import torch
import time
import os
import psutil
import numpy as np
from src.turboquant.memory import TurboDiskStore

def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def run_turbo_disk_benchmark():
    # Setup
    dim = 768
    num_vectors = 100000  # 100k vectors for a meaningful test
    top_k = 10
    storage_dir = "benchmark_storage"
    
    print(f"--- TurboDiskStore Benchmark Start ---")
    print(f"Vectors: {num_vectors}, Dim: {dim}")
    print(f"Initial RAM: {get_mem_usage():.2f} MB")
    
    # 1. Data Generation
    print("Generating dummy data...")
    data = torch.randn(num_vectors, dim)
    query = torch.randn(dim)
    
    # 2. Storage Phase
    store = TurboDiskStore(dim, bits=3, storage_dir=storage_dir)
    
    start_time = time.time()
    # Adding in batches for efficiency
    for i in range(num_vectors):
        store.add(data[i])
    storage_time = time.time() - start_time
    
    mem_after_storage = get_mem_usage()
    print(f"Storage Time: {storage_time:.2f}s")
    print(f"RAM after storage: {mem_after_storage:.2f} MB (Diff: {mem_after_storage - get_mem_usage():.2f} MB)")
    
    # 3. Search Phase (Warm-up)
    store.search(query, top_k=top_k)
    
    # 4. Search Performance Test
    print(f"Measuring search speed for {num_vectors} vectors...")
    start_search = time.time()
    results = store.search(query, top_k=top_k)
    search_time = time.time() - start_search
    
    print(f"--- Results ---")
    print(f"Search Time: {search_time:.4f}s")
    print(f"Queries Per Second (QPS): {1.0 / search_time:.2f}")
    print(f"Final RAM Usage: {get_mem_usage():.2f} MB")
    
    # Disk usage
    total_disk_size = sum(os.path.getsize(os.path.join(storage_dir, f)) for f in os.listdir(storage_dir)) / (1024 * 1024)
    print(f"Total Disk Space Used: {total_disk_size:.2f} MB")

if __name__ == "__main__":
    run_turbo_disk_benchmark()
