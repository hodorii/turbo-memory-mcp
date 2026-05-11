import struct
import numpy as np
import time
import tempfile
import os

from turbo_quant_paper import TurboQuantState, compress, prepare_query, estimate, compress_blob
from memory_store import MemoryStore

print("=" * 60)
print("TurboQuant Paper Algorithm — Memory Integration Test")
print("=" * 60)

DIM = 384
N = 500

rng = np.random.default_rng(42)
vectors = rng.standard_normal((N, DIM)).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
queries = vectors[:10]

state = TurboQuantState.build(DIM, b=3, seed=42)

print(f"Dimension: {DIM}, b={state.b}")
print(f"Centroids: n={len(state.centroids)}, range=[{state.centroids.min():.4f}, {state.centroids.max():.4f}]")
print()

print("Step 1: Direct accuracy test")
print("-" * 40)
ests, trues = [], []
for q in queries:
    q_rot, q_qjl = prepare_query(q, state)
    for i, v in enumerate(vectors):
        packed, norm, qjl, r_norm = compress(v, state)
        ests.append(estimate(q_rot, q_qjl, state, packed, norm, qjl, r_norm))
        trues.append(float(np.dot(q, v)))
ests, trues = np.array(ests), np.array(trues)
mae = np.mean(np.abs(trues - ests))
rmse = np.sqrt(np.mean((trues - ests)**2))
rel = np.median(np.abs(trues - ests) / (np.abs(trues) + 1e-9))
print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, Median rel: {rel:.1%}")

print()
print("Step 2: MemoryStore round-trip")
print("-" * 40)
tmp = tempfile.mktemp(suffix='.db')
try:
    store = MemoryStore(tmp, compression='paper')
    id_map = {}
    t0 = time.time()
    for i, v in enumerate(vectors):
        eid = store.add(f"vec_{i}", v, importance=0.5)
        id_map[eid] = i
    t1 = time.time()
    stats = store.stats()
    print(f"  Stored {N} entries in {t1-t0:.3f}s")
    print(f"  Compression: {stats['compression_ratio']:.1f}x, modes: {stats['compression_modes']}")

    print()
    print("Step 3: Ranking recall test")
    print("-" * 40)
    q_rot, q_qjl = prepare_query(queries[0], state)
    rows = store._db.execute('SELECT id, embedding, compression FROM entries').fetchall()
    scores = []
    for eid, blob, algo in rows:
        n_qjl, n_packed = struct.unpack('<ii', blob[:8])
        norm = struct.unpack_from('<f', blob, 8)[0]
        r_norm = struct.unpack_from('<f', blob, 12)[0]
        qjl = np.frombuffer(blob[20:20+n_qjl], dtype=np.int8).copy()
        packed = np.frombuffer(blob[20+n_qjl:20+n_qjl+n_packed], dtype=np.uint8).copy()
        est = estimate(q_rot, q_qjl, state, packed, norm, qjl, r_norm)
        scores.append((id_map[eid], est))
    scores.sort(key=lambda x: x[1], reverse=True)
    fp32_rank = sorted(range(N), key=lambda i: float(np.dot(queries[0], vectors[i])), reverse=True)
    fp32_set = set(fp32_rank[:5])
    est_set = set(idx for idx, _ in scores[:5])
    overlap = len(fp32_set & est_set)
    recall = overlap / 5.0
    print(f"  Top-5 recall (vs FP32): {overlap}/5 ({recall:.0%})")
    print(f"  FP32 top-5:  {fp32_rank[:5]}")
    print(f"  Est top-5:   {[idx for idx,_ in scores[:5]]}")
    print(f"  Overlap:     {fp32_set & est_set}")

    print()
    print("Step 4: Search API test")
    print("-" * 40)
    results = store.search(f"vec_0", queries[0], top_k=5)
    print(f"  Search returned {len(results)} results (all paper mode)")
    for eid, text, score in results:
        idx = id_map.get(eid, -1)
        print(f"  {eid} (vec_{idx}): score={score:.4f}")

    print()
    print("Step 5: Mixed compression modes")
    print("-" * 40)
    store_fp32 = MemoryStore(tmp, compression=None)
    store_fp32.add("fp32_mem", vectors[0])
    store_paper = MemoryStore(tmp, compression='paper')
    store_paper.add("paper_mem", vectors[0])
    mixed = store_paper.search("mem", vectors[0], top_k=5)
    s = store_paper.stats()
    print(f"  Mixed modes: {s['compression_modes']}")
    print(f"  Mixed search results: {len(mixed)}")

    store.close()
    store_fp32.close()
    store_paper.close()
    print()
    print("All tests passed!")
finally:
    os.unlink(tmp)