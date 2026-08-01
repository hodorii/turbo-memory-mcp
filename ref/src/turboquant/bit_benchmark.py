import numpy as np
import torch
from src.turboquant.quantizers import QuantizerRegistry

def run_bit_benchmark(bits_list=[2, 3, 4]):
    dim = 1024
    n_vectors = 1000
    # Load real data
    try:
        vectors = np.load('data/real_vectors.npy')[:n_vectors]
    except:
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)

    print(f"🚀 Bit-rate Benchmark: dim={dim}, n={n_vectors}")
    print("-" * 60)
    print(f"Bits | Algo     | vNMSE    | Recall@1 | Latency")
    print("-" * 60)

    for b in bits_list:
        for algo_id in ["DRIVE_V3", "QJL"]:
            # We simulate different bits by passing to the config (if supported)
            # For simplicity in this script, we instantiate with target bits
            if algo_id == "DRIVE_V3":
                from src.turboquant.quantizers import DriveV3Quantizer
                q = DriveV3Quantizer(dim=dim, bits=b)
            else:
                from src.turboquant.quantizers import QJLQuantizer
                q = QJLQuantizer(dim=dim, bits=b)
            
            # Simple metric calculation
            # (Using the logic from benchmark.py)
            # This is a placeholder for the actual benchmark.py call
            # To be accurate, I'll just run the main benchmark.py in a loop
            pass

if __name__ == "__main__":
    run_bit_benchmark()
