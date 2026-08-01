"""MemoryEngine E2E sillok integration test.

Creates two MemoryEngine instances sharing the same sillok data:
  - FAISS exact (use_quantization=False) — ground truth
  - V3 EDEN    (use_quantization=True)  — test target

Compares search quality (recall@k), latency, and text results.
"""

import glob
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine.memory_engine import model as _embed_model
from src.engine.memory_engine import MemoryEngine


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


def run_memory_engine_sillok_e2e():
    log = []
    def log_print(msg):
        print(msg)
        log.append(str(msg))

    log_print("=" * 60)
    log_print("MemoryEngine E2E Sillok Integration Test")
    log_print("FAISS exact vs V3 EDEN quantized")
    log_print("=" * 60)

    # Warm up the embedding model (first encode is ~10x slower)
    log_print("\n[Warmup] Pre-warming BGE-M3 encoder...")
    _embed_model.encode(["warmup"], batch_size=4)
    log_print("    Done")

    xml_dir = "data_local/chosun"
    xml_files = sorted(glob.glob(f"{xml_dir}/*.xml"))
    log_print(f"\n[1] Loading from {len(xml_files)} XML files ({xml_dir}/)...")

    texts = extract_texts_from_xml(xml_files[:10], max_texts=50)
    log_print(f"    Extracted {len(texts)} paragraphs")

    if len(texts) < 30:
        log_print("    Skipping test: too few texts")
        return log

    n_store = min(len(texts) - 3, 45)
    n_query = 3
    store_texts = texts[:n_store]
    query_texts = texts[n_store:n_store + n_query]
    log_print(f"    {n_store} stored, {n_query} queries")

    # FAISS engine (ground truth)
    faiss_db = "/tmp/mem_sillok_faiss.db"
    if os.path.exists(faiss_db):
        os.remove(faiss_db)
    for p in [faiss_db, faiss_db.replace(".db", "_turbo")]:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)

    log_print("\n[2] Building FAISS engine (exact, ground truth)...")
    t0 = time.time()
    faiss_engine = MemoryEngine(faiss_db, use_quantization=False)
    faiss_engine.add_batch(store_texts)
    t_faiss = time.time() - t0
    log_print(f"    Index time: {t_faiss:.2f}s ({n_store / max(t_faiss, 0.01):.0f} texts/s)")

    # EDEN engine
    eden_db = "/tmp/mem_sillok_eden.db"
    if os.path.exists(eden_db):
        os.remove(eden_db)
    for p in [eden_db, eden_db.replace(".db", "_turbo")]:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)

    log_print("\n[3] Building EDEN engine (3-bit quantized)...")
    t0 = time.time()
    eden_engine = MemoryEngine(eden_db, use_quantization=True, bits=3, eden_mode="biased")
    eden_engine.add_batch(store_texts)
    t_eden = time.time() - t0
    log_print(f"    Index time: {t_eden:.2f}s ({n_store / max(t_eden, 0.01):.0f} texts/s)")

    # Run queries
    log_print(f"\n[4] Running {n_query} queries (k=5)...")
    k = 5

    eden_recalls = []
    faiss_latency = []
    eden_latency = []

    for i, qt in enumerate(query_texts):
        t0 = time.time()
        faiss_results = faiss_engine.search(qt, top_k=k)
        ft = time.time() - t0
        faiss_latency.append(ft)
        faiss_texts = set(r[0] for r in faiss_results)

        t0 = time.time()
        eden_results = eden_engine.search(qt, top_k=k)
        et = time.time() - t0
        eden_latency.append(et)
        eden_texts = set(r[0] for r in eden_results)

        overlap = len(faiss_texts & eden_texts)
        recall = overlap / k
        eden_recalls.append(recall)

        if i < 3:
            log_print(f"\n  Query[{i}]: {qt[:60]}...")
            log_print(f"    FAISS: {[t[:40] for t,_ in faiss_results]}")
            log_print(f"    EDEN:  {[t[:40] for t,_ in eden_results]}")
            log_print(f"    recall@{k}={recall:.2f}")

    log_print(f"\n{'='*60}")
    log_print("SUMMARY")
    log_print(f"{'='*60}")
    log_print(f"  Store: {n_store} texts, Query: {n_query}, k={k}")
    log_print(f"")
    log_print(f"  Recall@{k} (EDEN vs FAISS exact):")
    log_print(f"    mean={np.mean(eden_recalls):.4f}")
    log_print(f"    min={np.min(eden_recalls):.4f}")
    log_print(f"    max={np.max(eden_recalls):.4f}")
    log_print(f"")
    log_print(f"  Search latency:")
    log_print(f"    FAISS: mean={np.mean(faiss_latency)*1000:.2f}ms")
    log_print(f"    EDEN:  mean={np.mean(eden_latency)*1000:.2f}ms")
    log_print(f"    EDEN/FAISS ratio: {np.mean(eden_latency)/max(np.mean(faiss_latency),1e-10):.2f}x")

    avg_recall = np.mean(eden_recalls)
    log_print(f"\n  EDEN avg recall@{k} vs FAISS exact: {avg_recall:.4f}")

    assert avg_recall >= 0.6, (
        f"EDEN recall@{k} too low vs FAISS: {avg_recall:.4f} < 0.6")
    log_print(f"  EDEN recall >= 0.60")

    # Test reusability
    log_print(f"\n[5] Reusability check...")
    extra_texts = texts[n_store:n_store + 3]
    if extra_texts:
        faiss_engine.add_batch(extra_texts)
        eden_engine.add_batch(extra_texts)
        result = eden_engine.search(query_texts[0], top_k=3)
        assert len(result) == 3, f"Expected 3 results, got {len(result)}"
        log_print(f"    Engine reusable after batch add")

    faiss_engine.close()
    eden_engine.close()
    for p in [faiss_db, eden_db,
              faiss_db.replace(".db", "_turbo"),
              eden_db.replace(".db", "_turbo")]:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)

    log_print(f"\n{'='*60}")
    log_print("ALL MEMORY ENGINE SILLOK E2E TESTS PASSED")
    log_print(f"{'='*60}")
    return log


if __name__ == "__main__":
    run_memory_engine_sillok_e2e()
