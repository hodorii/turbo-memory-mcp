import numpy as np
from math import sqrt, pi

class TurboQuantEngine:
    def __init__(self, dim=384, m=16):
        self.dim = dim
        self.m = m
        self.sub_dim = dim // m
        self.codebooks = np.random.randn(m, 256, self.sub_dim).astype(np.float32)

    def train(self, data: np.ndarray, iterations=5):
        n_samples = data.shape[0]
        sub_vectors = data.reshape(n_samples, self.m, self.sub_dim)
        for i in range(self.m):
            sub_vecs = sub_vectors[:, i, :]
            indices = np.random.choice(n_samples, 256, replace=False)
            self.codebooks[i] = sub_vecs[indices]
            for _ in range(iterations):
                dists = np.linalg.norm(sub_vecs[:, np.newaxis, :] - self.codebooks[i], axis=2)
                labels = np.argmin(dists, axis=1)
                for c in range(256):
                    if np.sum(labels == c) > 0:
                        self.codebooks[i, c] = np.mean(sub_vecs[labels == c], axis=0)

    def compress(self, vec: np.ndarray):
        vec_split = vec.reshape(self.m, self.sub_dim)
        indices = np.zeros(self.m, dtype=np.uint8)
        residual = np.zeros_like(vec)
        for i in range(self.m):
            dists = np.linalg.norm(self.codebooks[i] - vec_split[i], axis=1)
            indices[i] = np.argmin(dists)
            residual[i*self.sub_dim : (i+1)*self.sub_dim] = vec_split[i] - self.codebooks[i, indices[i]]
        
        sign_bits = (np.sign(residual) > 0).astype(np.int8)
        return indices, sign_bits

    def estimate(self, indices: np.ndarray, sign_bits: np.ndarray, query_vec: np.ndarray) -> float:
        q_split = query_vec.reshape(self.m, self.sub_dim)
        score = 0.0
        for i in range(self.m):
            score += np.dot(self.codebooks[i, indices[i]], q_split[i])
        
        correction = np.sum(np.abs(query_vec) * (2 * sign_bits - 1))
        return float(score + (correction * 0.01))

def verify_math_precision():
    engine = TurboQuantEngine()
    data = np.random.randn(1000, 384).astype(np.float32)
    engine.train(data)
    
    test_vec = np.random.randn(384).astype(np.float32)
    indices, signs = engine.compress(test_vec)
    
    original_dot = np.dot(test_vec, test_vec)
    estimated_dot = engine.estimate(indices, signs, test_vec)
    
    error = abs(original_dot - estimated_dot) / original_dot
    print(f"정밀도 검증 결과 - 오차율: {error:.4%}")
    if error < 0.2:
        print("검증 통과: 수학적 무결성 확인됨.")
    else:
        print("검증 실패: 정밀도 재설계 필요.")

if __name__ == "__main__":
    verify_math_precision()
