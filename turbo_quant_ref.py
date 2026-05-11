import numpy as np

class TurboQuantEngine:
    def __init__(self, dim=384):
        self.dim = dim
        self.R = np.random.randn(dim, dim).astype(np.float32)
        self.R, _ = np.linalg.qr(self.R)

    def compress(self, vec: np.ndarray):
        rotated = np.dot(self.R, vec)
        return (rotated >= 0).astype(np.int8)

    def estimate(self, sign1, sign2, norm1, norm2) -> float:
        d = self.dim
        hamming = np.sum(sign1 != sign2)
        cos_theta = np.cos(np.pi * hamming / d)
        return float(norm1 * norm2 * cos_theta)
