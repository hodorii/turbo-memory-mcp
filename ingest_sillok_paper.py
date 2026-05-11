import xml.etree.ElementTree as ET
import glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from memory_store import MemoryStore
from server import encode_batch

XML_DIR = os.path.join(os.path.dirname(__file__), "ref", "data_local", "chosun")
DB_PATH = os.path.join(os.path.dirname(__file__), "sillok_paper.db")

print("=" * 65)
print("조선왕조실록 — Paper Algorithm 검증")
print("=" * 65)

# ── Parse all XML ─────────────────────────────────────────────────────────
print("\n[1/4] XML 파싱...")
files = sorted(glob.glob(os.path.join(XML_DIR, "*.xml")))
print(f"  XML: {len(files)}개")

all_texts = []
for f in files:
    try:
        tree = ET.parse(f)
        root = tree.getroot()
        for p in root.iter('paragraph'):
            if p.text:
                t = p.text.strip()
                if len(t) > 10:
                    all_texts.append(t)
    except: pass

all_texts = list(dict.fromkeys(all_texts))
N = len(all_texts)
print(f"  총 {N:,}개 단락")

# ── Encode & Store ────────────────────────────────────────────────────────
print("\n[2/4] Encoder loading...")
t0 = time.time()
test_vec = encode_batch(["test"])[0]
t1 = time.time()
print(f"  Model loaded, dim={test_vec.shape[0]} ({t1-t0:.1f}s)")

# Use a fresh DB
if os.path.exists(DB_PATH):
    os.unlink(DB_PATH)
store = MemoryStore(DB_PATH, compression='paper')

print(f"\n  Encoding & storing {N:,} items via paper algorithm...")
t0 = time.time()
BATCH = 64
count = 0
for i in range(0, N, BATCH):
    batch = all_texts[i:i+BATCH]
    embs = encode_batch(batch)
    for j, text in enumerate(batch):
        store.add(text, embs[j], importance=0.7, commit=False)
    count += len(batch)
    if count % 320 == 0:  # commit every 5 batches
        store.commit()
        elapsed = time.time() - t0
        rate = count / elapsed if elapsed > 0 else 0
        eta = (N - count) / rate if rate > 0 else 0
        print(f"  {count:,}/{N:,} ({rate:.0f} items/s, {elapsed:.0f}s, ETA {eta:.0f}s)")
store.commit()
t1 = time.time()
print(f"  저장 완료: {count:,}개, {t1-t0:.1f}s")

stats = store.stats()
print(f"\n  압축 모드: {stats['compression_modes']}")
print(f"  압축률: {stats['compression_ratio']}x")
print(f"  FP32 필요: {stats['fp32_equivalent_bytes']/1e6:.1f}MB")
print(f"  실제 저장: {stats['actual_storage_bytes']/1e6:.1f}MB")
print(f"  DB 파일: {os.path.getsize(DB_PATH)/1e6:.2f}MB")

# ── Search ────────────────────────────────────────────────────────────────
print("\n[3/4] 검색 테스트")

queries = [
    "임진왜란과 이순신",
    "세종대왕 한글 창제",
    "태종 왕자의 난",
    "조선 초기 정치 혼란",
    "성종과 경국대전",
]

for q_text in queries:
    q_vec = encode_batch([q_text])[0]
    t0 = time.time()
    results = store.search(q_text, q_vec, top_k=5)
    t1 = time.time()
    print(f"\n  ▶ '{q_text}' ({(t1-t0)*1e3:.1f}ms)")
    for rank, (eid, text, score) in enumerate(results):
        txt = text[:80].replace('\n', ' ')
        print(f"    {rank+1}. ({score:.4f}) {txt}")

# ── Accuracy on first 1000 ────────────────────────────────────────────────
print("\n[4/4] 정확도 (처음 1000개)")
q_vec = encode_batch(["임진왜란"])[0]
state = store._get_state_paper()
q_rot = state.rotation @ q_vec
q_qjl = state.qjl_matrix @ q_vec

rows = store._db.execute(
    "SELECT rowid, text, embedding FROM entries WHERE compression='paper' LIMIT 1000"
).fetchall()
print(f"  샘플: {len(rows)}개")

# First batch: pre-unpack cache
paper_scores = []
for rowid, text, blob in rows:
    score = store._score_paper(rowid, blob, q_rot, q_qjl)
    paper_scores.append((text, score))

# Second batch: FP32 (encode 1000 texts)
t0 = time.time()
fp32_embs = encode_batch([r[0] for r in rows])
t1 = time.time()
print(f"  FP32 re-encode: {t1-t0:.1f}s")
fp32_scores = [(t, float(np.dot(q_vec, e) / (np.linalg.norm(q_vec) * np.linalg.norm(e) + 1e-9)))
               for t, e in zip([r[0] for r in rows], fp32_embs)]

paper_scores.sort(key=lambda x: x[1], reverse=True)
fp32_scores.sort(key=lambda x: x[1], reverse=True)

print(f"  Query: '임진왜란'")
print(f"  {'Rank':>4}  {'Paper Score':>11}  {'FP32 Score':>11}  {'Match?':>6}")
print(f"  {'-'*4}  {'-'*11}  {'-'*11}  {'-'*6}")
overlap = 0
for rank in range(10):
    match = "✓" if paper_scores[rank][0] == fp32_scores[rank][0] else "✗"
    if match == "✓": overlap += 1
    pscore, fscore = paper_scores[rank][1], fp32_scores[rank][1]
    txt = paper_scores[rank][0][:45]
    print(f"  {rank+1:>4}  {pscore:>11.4f}  {fscore:>11.4f}  {match:>6}  {txt}")

print(f"\n  Top-10 recall: {overlap}/10 ({overlap/10:.0%})")
all_p = np.array([s[1] for s in paper_scores])
all_f = np.array([s[1] for s in fp32_scores])
mae = np.mean(np.abs(all_p - all_f))
rmse = np.sqrt(np.mean((all_p - all_f)**2))
print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}")
print(f"  Pre-unpacked cache: {len(store._paper_cache)} entries")

store.close()
print(f"\nDB: {DB_PATH}")
print("완료!")
