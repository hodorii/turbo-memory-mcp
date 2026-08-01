"""E2E sillok integration test for EDEN V3 quantizer.

Loads real 조선왕조실록 XML data, embeds with BGE-M3, stores in
TurboDiskStore with EdenQuantizer, and measures search quality vs
brute-force and vs V2 baseline.
"""

import glob
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.turboquant.eden import EdenConfig, EdenQuantizer
from src.turboquant.memory import TurboDiskStore


def extract_texts_from_xml(file_paths, max_texts=500, max_length=512):
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
        except Exception as e:
            print(f"  Error parsing {path}: {e}")
    return texts


def run_sillok_e2e():
    log = []
    def log_print(msg):
        print(msg)
        log.append(str(msg))

    log_print("=" * 60)
    log_print("EDEN V3 E2E Sillok Integration Test")
    log_print("=" * 60)

    # 1. Load embedding model
    log_print("\n[1] Loading BGE-M3 embedding model...")
    model = SentenceTransformer("BAAI/bge-m3")
    DIM = 1024  # BGE-M3 dimension
    log_print(f"    dim={DIM}")

    # 2. Load sillok data
    xml_dir = "data_local/chosun"
    xml_files = sorted(glob.glob(f"{xml_dir}/*.xml"))
    log_print(f"\n[2] Loading from {len(xml_files)} XML files ({xml_dir}/)...")

    texts = extract_texts_from_xml(xml_files[:30], max_texts=300)
    log_print(f"    Extracted {len(texts)} paragraphs")

    # 3. Generate embeddings
    log_print(f"\n[3] Generating {len(texts)} embeddings with BGE-M3...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=4, show_progress_bar=True)
    embeddings = torch.from_numpy(np.array(embeddings)).float()
    t_embed = time.time() - t0
    log_print(f"    Embedding time: {t_embed:.2f}s")

    # Normalize
    norms = torch.norm(embeddings, dim=-1, keepdim=True)
    embeddings = embeddings / norms.clamp(min=1e-10)

    # Split: 280 for store, 20 for queries
    n_store = min(280, len(embeddings) - 20)
    n_query = min(20, len(embeddings) - n_store)
    store_embs = embeddings[:n_store]
    query_embs = embeddings[n_store:n_store + n_query]
    query_texts = texts[n_store:n_store + n_query]
    log_print(f"    {n_store} stored, {n_query} queries")

    # 4. Build TurboDiskStore with EdenQuantizer (biased, 3-bit, residual)
    log_print("\n[4] Building TurboDiskStore with EdenQuantizer (biased, 3-bit, DRIVE)...")
    eden_cfg = EdenConfig(dim=DIM, bits=3, mode="biased", residual_bits=1, seed=42)
    eq = EdenQuantizer(eden_cfg)

    store_eden_dir = "/tmp/eden_sillok_store"
    if os.path.exists(store_eden_dir):
        shutil.rmtree(store_eden_dir)
    store_eden = TurboDiskStore(DIM, bits=3, storage_dir=store_eden_dir, quantizer=eq)

    t0 = time.time()
    for v in store_embs:
        store_eden.add(v)
    t_index = time.time() - t0
    log_print(f"    Index time: {t_index:.2f}s ({n_store / t_index:.0f} vec/s)")
    bpd_indices = 8 * store_eden.indices_mmap.itemsize
    bpd_signs = 8 * store_eden.signs_mmap.itemsize
    bpd_scale = 16 * store_eden.scales_mmap.itemsize
    bpd_s = 32 * store_eden.scale_s_mmap.itemsize
    bpd_total = (bpd_indices + bpd_signs + bpd_scale + bpd_s) / 8
    log_print(f"    Bits per dim: {bpd_total:.2f}")

    # 5. Build V2 baseline
    log_print("\n[5] Building V2 baseline for comparison...")
    store_v2_dir = "/tmp/v2_sillok_store"
    if os.path.exists(store_v2_dir):
        shutil.rmtree(store_v2_dir)
    store_v2 = TurboDiskStore(DIM, bits=3, storage_dir=store_v2_dir)

    t0 = time.time()
    for v in store_embs:
        store_v2.add(v)
    t_v2 = time.time() - t0
    log_print(f"    Index time: {t_v2:.2f}s ({n_store / t_v2:.0f} vec/s)")

    # 6. Search queries
    log_print(f"\n[6] Running {n_query} queries...")
    k = 5

    eden_recalls = []
    v2_recalls = []
    eden_times = []
    v2_times = []

    for i in range(n_query):
        q = query_embs[i]

        # Brute-force (exact) top-k
        bf_scores = torch.matmul(store_embs, q)
        bf_topk = set(torch.topk(bf_scores, k).indices.tolist())

        # EDEN search
        t0 = time.time()
        eden_results = store_eden.search(q, top_k=k)
        et = time.time() - t0
        eden_set = set(r[0] for r in eden_results)
        eden_recall = len(bf_topk & eden_set) / k
        eden_recalls.append(eden_recall)
        eden_times.append(et)

        # V2 search
        t0 = time.time()
        v2_results = store_v2.search(q, top_k=k)
        vt = time.time() - t0
        v2_set = set(r[0] for r in v2_results)
        v2_recall = len(bf_topk & v2_set) / k
        v2_recalls.append(v2_recall)
        v2_times.append(vt)

        if i < 3:
            qt = query_texts[i][:60]
            log_print(f"\n  Query[{i}]: {qt}")
            log_print(f"    EDEN recall@{k}={eden_recall:.2f}  time={et*1000:.2f}ms")
            log_print(f"    V2   recall@{k}={v2_recall:.2f}  time={vt*1000:.2f}ms")

    # 7. Summary
    log_print(f"\n{'='*60}")
    log_print("SUMMARY")
    log_print(f"{'='*60}")
    log_print(f"  Data: {n_store} vectors, dim={DIM}, 3-bit, k={k}")
    log_print(f"")
    log_print(f"  Recall@{k}:")
    log_print(f"    EDEN: mean={np.mean(eden_recalls):.4f}  "
              f"min={np.min(eden_recalls):.4f}  max={np.max(eden_recalls):.4f}")
    log_print(f"    V2:   mean={np.mean(v2_recalls):.4f}  "
              f"min={np.min(v2_recalls):.4f}  max={np.max(v2_recalls):.4f}")
    log_print(f"")
    log_print(f"  Search latency:")
    log_print(f"    EDEN: mean={np.mean(eden_times)*1000:.3f}ms  "
              f"total={np.sum(eden_times)*1000:.1f}ms")
    log_print(f"    V2:   mean={np.mean(v2_times)*1000:.3f}ms  "
              f"total={np.sum(v2_times)*1000:.1f}ms")
    log_print(f"")

    # 8. Assertions
    avg_eden = np.mean(eden_recalls)
    avg_v2 = np.mean(v2_recalls)
    log_print(f"  EDEN avg recall@{k}: {avg_eden:.4f}")
    log_print(f"  V2   avg recall@{k}: {avg_v2:.4f}")
    log_print(f"  Δ: {avg_eden - avg_v2:+.4f}")

    assert avg_eden >= avg_v2, (
        f"EDEN recall ({avg_eden:.4f}) < V2 recall ({avg_v2:.4f})")
    log_print(f"\n  ✓ EDEN matches or exceeds V2 recall")

    # Search quality baseline: at least 1/5 recall on average
    assert avg_eden >= 0.2, (
        f"EDEN recall@{k} too low: {avg_eden:.4f}")

    # Cleanup
    shutil.rmtree(store_eden_dir, ignore_errors=True)
    shutil.rmtree(store_v2_dir, ignore_errors=True)

    log_print(f"\n{'='*60}")
    log_print("ALL SILLOK E2E TESTS PASSED")
    log_print(f"{'='*60}")

    return log


if __name__ == "__main__":
    run_sillok_e2e()
