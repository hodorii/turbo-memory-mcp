#!/usr/bin/env python3
"""실제 검색 결과 예시 — EDEN 2/3/4bit vs V2 비교"""

import glob, os, shutil, sys, time
import xml.etree.ElementTree as ET
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.turboquant.eden import EdenConfig, EdenQuantizer
from src.turboquant.memory import TurboDiskStore

def extract_texts(file_paths, max_texts=500, max_length=512):
    texts = []
    for path in file_paths:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            for element in root.iter("paragraph"):
                text = "".join(element.itertext()).strip()
                if text and len(text) > 10:
                    texts.append(text[:max_length])
                    if len(texts) >= max_texts:
                        return texts
        except:
            pass
    return texts

# 1. Load model + data
model = SentenceTransformer("BAAI/bge-m3")
DIM = 1024
xml_files = sorted(glob.glob("data_local/chosun/*.xml"))
texts = extract_texts(xml_files[:30], max_texts=300)
print(f"Loaded {len(texts)} texts")

embs = torch.from_numpy(np.array(model.encode(texts, batch_size=4, show_progress_bar=True))).float()
embs = embs / torch.norm(embs, dim=-1, keepdim=True).clamp(min=1e-10)

store_embs = embs[:280]
query_embs = embs[280:300]
query_texts = texts[280:300]

# 2. Index with 2/3/4-bit EDEN + V2
stores = {}
for bits in [2, 3, 4]:
    for mode_name, mode in [("EDEN-biased", "biased")]:
        eq = EdenQuantizer(EdenConfig(dim=DIM, bits=bits, mode=mode, residual_bits=1, seed=42))
        path = f"/tmp/demo_b{bits}"
        if os.path.exists(path): shutil.rmtree(path)
        store = TurboDiskStore(DIM, bits, path, quantizer=eq)
        for v in store_embs:
            store.add(v)
        stores[(bits, mode_name)] = store

# V2 baseline
store_v2 = TurboDiskStore(DIM, 3, "/tmp/demo_v2")
for v in store_embs:
    store_v2.add(v)

# 3. Pick 4 diverse queries and show results
selected_queries = [0, 3, 7, 15]
k = 5

for qi in selected_queries:
    q_text = query_texts[qi][:80]
    q_vec = query_embs[qi]
    print(f"\n{'='*70}")
    print(f"쿼리: \"{q_text}…\"")
    print(f"{'='*70}")

    # Brute-force exact
    bf_scores = torch.matmul(store_embs, q_vec)
    bf_topk = torch.topk(bf_scores, k).indices.tolist()
    print(f"\n[Brute-force (정답)] 상위 {k}:")
    for j, idx in enumerate(bf_topk):
        print(f"  {j+1}. {texts[idx][:70]}…")

    for bits in [2, 3, 4]:
        store = stores[(bits, "EDEN-biased")]
        t0 = time.time()
        results = store.search(q_vec, top_k=k)
        et = time.time() - t0
        result_set = set(r[0] for r in results)
        recall = len(set(bf_topk) & result_set) / k
        print(f"\n[EDEN {bits}bit] recall@{k}={recall:.2f}  ({et*1000:.1f}ms):")
        for j, (ridx, score) in enumerate(results):
            print(f"  {j+1}. [{ridx:3d}] {texts[ridx][:60]}… (cos={score:.4f})")

    # V2
    t0 = time.time()
    v2_results = store_v2.search(q_vec, top_k=k)
    vt = time.time() - t0
    v2_set = set(r[0] for r in v2_results)
    v2_recall = len(set(bf_topk) & v2_set) / k
    print(f"\n[V2 3bit] recall@{k}={v2_recall:.2f}  ({vt*1000:.1f}ms):")
    for j, (ridx, score) in enumerate(v2_results):
        print(f"  {j+1}. [{ridx:3d}] {texts[ridx][:60]}… (cos={score:.4f})")

# Cleanup
for bits in [2, 3, 4]:
    shutil.rmtree(f"/tmp/demo_b{bits}", ignore_errors=True)
shutil.rmtree("/tmp/demo_v2", ignore_errors=True)

print(f"\n{'='*70}")
print("데모 완료")
print(f"{'='*70}")
