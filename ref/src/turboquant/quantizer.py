import numpy as np
import torch

class TurboQuantizer_V1:
    def __init__(self, dim, bits=3):
        self.dim = dim
        self.bits = bits
        # Generate random rotation matrix using QR decomposition
        mat = np.random.randn(dim, dim)
        q, _ = np.linalg.qr(mat)
        self.rotation = torch.from_numpy(q).float()
        # Lloyd-Max thresholds (simplified for Gaussian N(0, 1))
        self.codebook = self._generate_codebook(bits)

    def _generate_codebook(self, bits):
        levels = 2 ** bits
        # Representative points for Gaussian distribution
        return torch.linspace(-2, 2, levels)

    def quantize(self, x):
        # Rotate
        y = torch.matmul(x, self.rotation)
        # Scalar Quantization
        indices = torch.bucketize(y, self.codebook) - 1
        indices = torch.clamp(indices, 0, len(self.codebook) - 1)
        quantized = self.codebook[indices]
        return quantized, indices

    def dequantize(self, indices):
        y_hat = self.codebook[indices]
        # Inverse Rotate
        return torch.matmul(y_hat, self.rotation.t())

TurboQuantizer = TurboQuantizer_V1
