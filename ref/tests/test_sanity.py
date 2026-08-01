import torch
from src.turboquant.memory import TurboDiskStore
import os

def test_turbo_disk_store_sanity():
    print("--- TurboDiskStore Sanity Test ---")
    dim = 128
    bits = 3
    storage_dir = "sanity_storage"
    
    # Clean up old storage
    if os.path.exists(storage_dir):
        for f in os.listdir(storage_dir):
            os.remove(os.path.join(storage_dir, f))
    
    store = TurboDiskStore(dim, bits=bits, storage_dir=storage_dir)
    
    # Add vectors
    vecs = [torch.randn(dim) for _ in range(100)]
    for v in vecs:
        store.add(v)
    print(f"Stored {store.count} vectors.")
    
    # Search
    query = torch.randn(dim)
    results = store.search(query, top_k=3)
    
    print("Search results:")
    for idx, score in results:
        print(f"Index: {idx}, Score: {score:.4f}")
        assert idx < 100
        
    print("Sanity test passed!")

if __name__ == "__main__":
    test_turbo_disk_store_sanity()
