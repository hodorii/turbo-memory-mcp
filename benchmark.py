import time
import numpy as np
import os
from memory_store import MemoryStore
from datetime import datetime, timedelta

def run_benchmark(n_entries=100000):
    db_path = "bench_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    store = MemoryStore(db_path)
    print(f"--- Lifelong Memory Benchmark ({n_entries} entries) ---")
    
    # 1. Data Generation & Ingestion
    print(f"Injecting {n_entries} entries...")
    start_time = time.time()
    
    # Batch injection for speed
    batch_size = 5000
    for i in range(0, n_entries, batch_size):
        for j in range(batch_size):
            idx = i + j
            text = f"This is a dummy memory entry number {idx} for benchmarking."
            # Random vector (384-dim)
            vec = np.random.randn(384).astype(np.float32)
            vec /= np.linalg.norm(vec)
            
            # Varying metadata and timestamps
            category = "work" if idx % 2 == 0 else "personal"
            # Spread timestamps over last 10 years
            days_ago = np.random.randint(0, 3650)
            dt = datetime.now() - timedelta(days=days_ago)
            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Direct insert to bypass encoding but test store logic
            # Using current date if needed, but the point is to test ingestion speed
            store.add(text, vec, metadata={"category": category}, importance=np.random.random())
        
        store.commit()
        print(f"Progress: {idx+1}/{n_entries} ({(idx+1)/n_entries*100:.1f}%)")

    ingest_time = time.time() - start_time
    print(f"Ingestion finished in {ingest_time:.2f}s (Avg: {ingest_time/n_entries*1000:.2f}ms/entry)")
    print(f"DB Size: {os.path.getsize(db_path) / 1024 / 1024:.2f} MB")

    # 2. Performance Testing
    query_vec = np.random.randn(384).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    
    test_cases = [
        ("No Filter (Brute-force)", None),
        ("Metadata Filter (50% reduction)", "metadata LIKE '%work%'"),
        ("Temporal Filter (Recent 1 year)", f"created_at > '{(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')}'"),
        ("Strict Filter (Small subset)", "metadata LIKE '%work%' AND created_at > '2024-01-01'")
    ]

    print("\n--- Latency Results ---")
    for label, filters in test_cases:
        latencies = []
        for _ in range(5): # 5 runs for average
            start = time.time()
            res = store.search("benchmark", query_vec, top_k=5, filters=filters)
            latencies.append((time.time() - start) * 1000)
        
        avg_lat = sum(latencies) / len(latencies)
        print(f"{label:<35}: {avg_lat:8.2f} ms")

if __name__ == "__main__":
    run_benchmark(100000)
