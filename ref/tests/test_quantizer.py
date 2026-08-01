import torch
from src.turboquant.quantizer import TurboQuantizer

def test_reconstruction_error():
    dim = 128
    bits = 3
    tq = TurboQuantizer(dim, bits)
    
    x = torch.randn(1, dim)
    quantized_vec, indices = tq.quantize(x)
    x_hat = tq.dequantize(indices)
    
    mse = torch.mean((x - x_hat) ** 2)
    print(f"MSE: {mse.item()}")
    assert mse < 0.5  # Rough heuristic for sanity check

if __name__ == "__main__":
    test_reconstruction_error()
    print("Test passed!")
