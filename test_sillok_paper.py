import json, os, time, tempfile, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from memory_store import MemoryStore
from server import encode_batch

SILLOK_PATH = os.path.join(os.path.dirname(__file__), "data", "sillok_sample.json")
DB_PATH = os.path.join(os.path.dirname(__file__), "sillok_paper.db")

# ── Load sillok data ──────────────────────────────────────────────────────
print("=" * 65)
print("조선왕조실록 — TurboQuant Paper Algorithm 검증")
print("=" * 65)

with open(SILLOK_PATH) as f:
    sillok = json.load(f)
print(f"\n데이터: {len(sillok)}개 기록")

# Separate into text-only list for batch encoding
texts = [s["text"] for s in sillok]
importances = [s.get("importance", 0.5) for s in sillok]

# ── Encode embeddings ─────────────────────────────────────────────────────
print("\nStep 1: Embedding 생성 중... (SentenceTransformer bge-m3)")
t0 = time.time()
embeddings = encode_batch(texts)
t1 = time.time()
print(f"  {len(embeddings)}개 벡터, dim={embeddings.shape[1]}, {t1-t0:.1f}s")

# ── Ingest into paper-compressed store ────────────────────────────────────
print("\nStep 2: Paper 압축 모드로 저장 중...")
if os.path.exists(DB_PATH):
    os.unlink(DB_PATH)

store = MemoryStore(DB_PATH, compression='paper')
t0 = time.time()
for text, emb, imp in zip(texts, embeddings, importances):
    store.add(text, emb, importance=imp)
store.commit()
t1 = time.time()
print(f"  저장 완료: {t1-t0:.1f}s ({((t1-t0)/len(texts))*1e3:.1f}ms/개)")

stats = store.stats()
print(f"  압축률: {stats['compression_ratio']}x")
print(f"  FP32 필요: {stats['fp32_equivalent_bytes']/1e6:.1f}MB")
print(f"  실제 저장: {stats['actual_storage_bytes']/1e6:.1f}MB")

# ── Search queries ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 3: 의미 검색 (Semantic Search)")
print("=" * 65)

queries = [
    "태종 왕자의 난",
    "세종대왕 한글 창제",
    "임진왜란",
    "조선 초기 정치",
    "과거 시험",
]

for q_text in queries:
    print(f"\n  ▶ 검색어: '{q_text}'")
    q_vec = encode_batch([q_text])[0]
    
    # Paper compressed search
    t0 = time.time()
    results = store.search(q_text, q_vec, top_k=5)
    t1 = time.time()
    
    # FP32 exact search for comparison
    rows = store._db.execute("SELECT rowid, id, text, embedding, compression, importance, created_at FROM entries WHERE compression='fp32'").fetchall()
    if not rows:
        # Re-compute FP32 on the fly from stored paper vectors
        rows = store._db.execute("SELECT rowid, id, text, embedding, compression, importance, created_at FROM entries").fetchall()
        # For FP32 reference, compute exact cosine similarity using original embeddings
        fp32_scores = []
        for r in rows:
            rid, eid, text, blob, algo, imp, ts = r
            idx = texts.index(text) if text in texts else -1
            if idx >= 0:
                fp32_scores.append((idx, float(np.dot(embeddings[idx], q_vec) / (
                    np.linalg.norm(embeddings[idx]) * np.linalg.norm(q_vec) + 1e-9))))
        fp32_scores.sort(key=lambda x: x[1], reverse=True)
    else:
        fp32_scores = []
        for r in rows:
            rid, eid, text, blob, algo, imp, ts = r
            vec = np.frombuffer(blob, dtype=np.float32)
            fp32_scores.append((eid, float(np.dot(q_vec, vec) / (
                np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-9))))
        fp32_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Show paper results
    for rank, (eid, text, score) in enumerate(results):
        # Find FP32 rank of this result
        fp32_rank = -1
        for i, (fid, fscore) in enumerate(fp32_scores[:20]):
            if (isinstance(fid, int) and texts[fid] == text) or (isinstance(fid, str) and fid == eid):
                fp32_rank = i + 1
                break
        fp32_mark = f"  [FP32 순위: {fp32_rank}위]" if fp32_rank > 0 else ""
        print(f"    {rank+1}. ({score:.4f}) {text[:50]}... {fp32_mark}")
    
    print(f"    ⏱  검색 시간: {(t1-t0)*1e3:.1f}ms")

# ── Compare with FP32 ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 4: 압축 vs FP32 정확도 비교")
print("=" * 65)

# Pick a query and compute all scores
q_text = queries[0]
q_vec = encode_batch([q_text])[0]

# Get paper scores for all entries
paper_rows = store._db.execute(
    "SELECT id, text, embedding FROM entries WHERE compression='paper'"
).fetchall()

paper_scores = []
for eid, text, blob in paper_rows:
    score = store._score_paper(0, blob, 
        store._get_state_paper().rotation @ q_vec,
        store._get_state_paper().qjl_matrix @ q_vec)
    paper_scores.append((text, score))

# FP32 exact scores
fp32_scores = []
for i, v in enumerate(embeddings):
    score = float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-9))
    fp32_scores.append((texts[i], score))

paper_scores.sort(key=lambda x: x[1], reverse=True)
fp32_scores.sort(key=lambda x: x[1], reverse=True)

# Top-10 comparison
print(f"\n  Query: '{q_text}'")
print(f"  {'Rank':>4}  {'Paper Score':>11}  {'FP32 Score':>11}  {'Match?':>6}  Content")
print(f"  {'-'*4}  {'-'*11}  {'-'*11}  {'-'*6}  {'-'*30}")
overlap = 0
for rank in range(10):
    p_text = paper_scores[rank][0] if rank < len(paper_scores) else ""
    f_text = fp32_scores[rank][0] if rank < len(fp32_scores) else ""
    match = "✓" if p_text == f_text else "✗"
    if match == "✓": overlap += 1
    p_short = p_text[:30] if p_text else ""
    f_short = f_text[:30] if f_text else ""
    print(f"  {rank+1:>4}  {paper_scores[rank][1]:>11.4f}  {fp32_scores[rank][1]:>11.4f}  {match:>6}  {p_short}")

print(f"\n  Top-10 일치: {overlap}/10")
print(f"  Top-10 Recall@10: {overlap/10:.0%}")

# Overall MAE
all_paper = np.array([s[1] for s in paper_scores])
all_fp32 = np.array([s[1] for s in fp32_scores])
mae = np.mean(np.abs(all_paper - all_fp32))
rmse = np.sqrt(np.mean((all_paper - all_fp32)**2))
print(f"  전체 MAE: {mae:.4f}")
print(f"  전체 RMSE: {rmse:.4f}")

store.close()
print(f"\nDB: {DB_PATH} ({os.path.getsize(DB_PATH)/1e6:.1f}MB)")
print("완료!")
