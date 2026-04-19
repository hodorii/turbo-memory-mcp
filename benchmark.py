import time
import json
import numpy as np
import os
import sys

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_startup_latency():
    print("--- 1. Startup Latency Test ---")
    start = time.perf_counter()
    import server
    # The moment of 'import' or 'server.store' creation should be fast now
    end = time.perf_counter()
    print(f"Server module load time: {(end - start) * 1000:.2f}ms")
    
    start = time.perf_counter()
    # Mock initialize
    res = server.dispatch({"method": "initialize", "id": 1})
    end = time.perf_counter()
    print(f"Initialize response time (Lazy): {(end - start) * 1000:.2f}ms")
    assert (end - start) < 0.5, "Initialize should be under 500ms"

def test_search_efficiency(n_items=1000):
    print(f"\n--- 2. Search Efficiency Test (N={n_items}) ---")
    import server
    from memory_store import MemoryStore
    
    test_db = "bench_test.db"
    if os.path.exists(test_db): os.remove(test_db)
    
    store = MemoryStore(path=test_db)
    dim = 384
    
    print(f"Inserting {n_items} random vectors...")
    texts = [f"text_{i}" for i in range(n_items)]
    # Use random vectors to avoid model loading in this step
    embs = np.random.randn(n_items, dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    
    for t, e in zip(texts, embs):
        store.add(t, e)
        
    query = np.random.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    
    # Warmup
    store.search(query_text="random", query_vec=query, top_k=5)
    
    start = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        results = store.search(query_text="random", query_vec=query, top_k=5)
    end = time.perf_counter()
    
    avg_ms = ((end - start) / iterations) * 1000
    print(f"Average search time for N={n_items}: {avg_ms:.2f}ms")
    
    # Simple O(N*d) vs O(N*d^2) check:
    # 1000 items, 384 dim. O(N*d) is ~384,000 ops.
    # Should be well under 50ms on modern CPU.
    assert avg_ms < 100, "Search is too slow, check O(N*d) optimization"

    if os.path.exists(test_db): os.remove(test_db)

def test_parallel_execution():
    print("\n--- 3. Parallel Execution Test ---")
    import server
    import threading
    
    results = []
    def call_recall(idx):
        start = time.perf_counter()
        # This will trigger model loading on the first call
        server.dispatch({
            "method": "tools/call",
            "params": {"name": "recall", "arguments": {"query": "test"}},
            "id": idx
        })
        results.append(time.perf_counter() - start)

    threads = [threading.Thread(target=call_recall, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    print(f"Parallel calls (5) completed. Max latency: {max(results)*1000:.2f}ms")

if __name__ == "__main__":
    test_startup_latency()
    test_search_efficiency(1000)
    test_parallel_execution()
