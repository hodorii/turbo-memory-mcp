import torch
from src.turboquant.memory import TurboDiskStore

def test_memory_recall():
    dim = 64
    store = TurboDiskStore(dim, bits=3)
    
    # Add dummy vectors
    v1 = torch.randn(dim)
    v2 = torch.randn(dim)
    
    store.add(v1)
    store.add(v2)
    
    # Search for v1
    results = store.search(v1, top_k=1)
    print(f"Top match index: {results[0][0]}, Score: {results[0][1]}")
    
    assert results[0][0] == 0 # Should match v1
    assert results[0][1] > 0.8 # Similarity should be reasonably high

if __name__ == "__main__":
    test_memory_recall()
    print("Memory recall test passed!")
