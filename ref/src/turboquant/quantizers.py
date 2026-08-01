from .registry import QuantizerRegistry
import numpy as np
import torch
from typing import Optional
from .interfaces import BaseQuantizer, QuantizedResult
from .eden import EdenConfig, EdenQuantizer as _EdenQuantizer

def safe_dot(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.sum(a_flat * b_flat).item()

@QuantizerRegistry.register("DRIVE_V3")
class DriveV3Quantizer(BaseQuantizer):
    """
    STRICT PAPER REPLICATION: DRIVE (1-bit)
    Mechanism: Random Projection -> Sign -> S-scaling
    """
    def __init__(self, dim: int = 1024, bits: int = 1, seed: Optional[int] = None):
        self.dim = dim
        self.bits = 1 # DRIVE is strictly 1-bit
        if seed is not None:
            torch.manual_seed(seed)
        
        mat = torch.randn(dim, dim)
        q, _ = torch.linalg.qr(mat)
        self.rotation = q.float()

    def quantize(self, x: np.ndarray) -> QuantizedResult:
        x_tensor = torch.from_numpy(x).float()
        if x_tensor.dim() == 1: x_tensor = x_tensor.unsqueeze(0)
        
        y = torch.matmul(x_tensor, self.rotation)
        signs = torch.sign(y).to(torch.int8)
        
        # DRIVE Theorem 2: optimal scale s = E[|y|] / d
        scale = torch.mean(y.abs(), dim=-1)
        
        return QuantizedResult(
            algo_id="DRIVE_V3",
            values=np.zeros(self.dim, dtype=np.int32), # Not used for 1-bit
            signs=signs.flatten().numpy().astype(np.int8),
            scale=float(scale.flatten()[0])
        )

    def decode(self, q: QuantizedResult) -> np.ndarray:
        if q.signs is None: return np.zeros(self.dim)
        signs = torch.from_numpy(q.signs).float()
        # x_hat = s * (signs @ R^T)
        x_hat = torch.matmul(signs.unsqueeze(0), self.rotation.t()) * (q.scale or 0.1)
        return x_hat.flatten().numpy()

    def calculate_score(self, query: np.ndarray, q: QuantizedResult) -> float:
        y_query = torch.matmul(torch.from_numpy(query).float(), self.rotation)
        if q.signs is None: return 0.0
        signs = torch.from_numpy(q.signs).float()
        # score = s * <signs, y_query>
        return safe_dot(signs, y_query) * (q.scale or 0.1)

@QuantizerRegistry.register("EDEN")
class EdenQuantizer(BaseQuantizer):
    """
    STRICT PAPER REPLICATION: EDEN (Full b-bit)
    Mechanism: Random Projection -> Lloyd-Max (2^b levels) -> S-scaling
    """
    def __init__(self, dim: int = 1024, bits: int = 3, seed: Optional[int] = None):
        self.dim = dim
        self.bits = bits
        self.config = EdenConfig(dim=dim, bits=bits, mode="unbiased", seed=seed)
        self._internal = _EdenQuantizer(self.config)

    def quantize(self, x: np.ndarray) -> QuantizedResult:
        x_tensor = torch.from_numpy(x).float()
        if x_tensor.dim() == 1: x_tensor = x_tensor.unsqueeze(0)
        
        # Full b-bit quantization + S-scaling (No residual)
        indices, _, _, S = self._internal.quantize(x_tensor)
        
        return QuantizedResult(
            algo_id="EDEN",
            values=indices.flatten().numpy().astype(np.int32),
            signs=None, # Full-bit EDEN uses indices, not signs
            scale=float(S.flatten()[0]) if S is not None else None
        )

    def decode(self, q: QuantizedResult) -> np.ndarray:
        indices = torch.from_numpy(q.values).long()
        S = torch.tensor([q.scale]) if q.scale is not None else torch.tensor([1.0])
        # x_hat = S * inverse_rotate(codebook[indices])
        x_hat = self._internal.decode(indices, None, None, S)
        return x_hat.flatten().numpy()

    def calculate_score(self, query: np.ndarray, q: QuantizedResult) -> float:
        y_query = torch.matmul(torch.from_numpy(query).float(), self._internal.rotation)
        indices = torch.from_numpy(q.values).long()
        q_val = self._internal.codebook[indices]
        
        # score = <y_query, q_val> * S
        score = safe_dot(y_query, q_val)
        if q.scale is not None:
            score *= q.scale
        return score

@QuantizerRegistry.register("QJL")
class QJLQuantizer(BaseQuantizer):
    """
    STRICT PAPER REPLICATION: QJL (1-bit)
    Mechanism: Random Projection -> Sign (Pure unbiased without scaling)
    """
    def __init__(self, dim: int = 1024, bits: int = 1):
        self.dim = dim
        self.bits = 1
        mat = torch.randn(dim, dim)
        q, _ = torch.linalg.qr(mat)
        self.rotation = q.float()

    def quantize(self, x: np.ndarray) -> QuantizedResult:
        x_tensor = torch.from_numpy(x).float()
        if x_tensor.dim() == 1: x_tensor = x_tensor.unsqueeze(0)
        y = torch.matmul(x_tensor, self.rotation)
        signs = torch.sign(y).to(torch.int8)
        
        return QuantizedResult(
            algo_id="QJL",
            values=np.zeros(self.dim, dtype=np.int32),
            signs=signs.flatten().numpy().astype(np.int8),
            scale=None # QJL is pure sign
        )

    def decode(self, q: QuantizedResult) -> np.ndarray:
        if q.signs is None: return np.zeros(self.dim)
        signs = torch.from_numpy(q.signs).float()
        return torch.matmul(signs.unsqueeze(0), self.rotation.t()).flatten().numpy()

    def calculate_score(self, query: np.ndarray, q: QuantizedResult) -> float:
        y_query = torch.matmul(torch.from_numpy(query).float(), self.rotation)
        if q.signs is None: return 0.0
        signs = torch.from_numpy(q.signs).float()
        return safe_dot(signs, y_query)

