import torch
import sys
from src.turboquant.memory import TurboDiskStore

def verify_efficiency():
    dim = 768
    num_vectors = 1000
    
    # 1. Raw Float32 size
    raw_size = num_vectors * dim * 4 / (1024 * 1024)
    print(f"Raw Float32 size: {raw_size:.2f} MB")
    
    # 2. TurboMemoryStore size
    store = TurboDiskStore(dim, bits=3)
    for _ in range(num_vectors):
        store.add(torch.randn(dim))
    
    indices_size = sum(t.element_size() * t.nelement() for t in store.indices_storage) / (1024 * 1024)
    residuals_size = sum(t.element_size() * t.nelement() for t in store.residuals_storage) / (1024 * 1024)
    
    print(f"TurboMemoryStore Indices size: {indices_size:.2f} MB")
    print(f"TurboMemoryStore Residuals size: {residuals_size:.2f} MB")
    print(f"Total TurboMemoryStore storage: {indices_size + residuals_size:.2f} MB")
    
    efficiency_ratio = (indices_size + residuals_size) / raw_size
    print(f"Efficiency Ratio: {efficiency_ratio:.2f}x (Lower is better, >1.0 means it's worse than float32)")

if __name__ == "__main__":
    verify_efficiency()
