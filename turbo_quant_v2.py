"""
TurboQuant Algorithm B — 3-bit Data-Driven Levels + 1-bit Residual (Sign + Scale)

Reference: turboquant/src/turboquant/memory.py (TurboQuantizer_V2)

Alternative approach:
  - 3-bit: 8 quantization levels derived from normal distribution quantiles
  - 1-bit: residual encoding via sign + per-vector scale factor
  - Search: LUT (Look-Up Table) outer product trick for O(d·L) scoring
  
Total: 3 bits + 1 bit + 16 bits (scale) per vector = ~7.9x theoretical compression vs FP32
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TurboQuantV2State:
    """Shared state for Algorithm B compression and estimation.
    
    Uses data-driven quantization levels (from normal quantiles)
    and per-vector residual scale.
    """
    dim: int                 # Vector dimension
    bits: int                # Total bits (default: 3 for levels)
    levels: np.ndarray       # (2^bits,) quantization levels from normal quantiles
    rotation: np.ndarray     # (dim, dim) orthogonal matrix Π


def build_state_v2(dim: int = 384, bits: int = 3, seed: int = 42) -> TurboQuantV2State:
    """Build Algorithm B state with data-driven levels.
    
    Args:
        dim: Vector dimension
        bits: Bits for level quantization (default: 3 → 8 levels)
        seed: Random seed for rotation reproducibility
    
    Returns:
        TurboQuantV2State
    """
    rng = np.random.RandomState(seed)
    
    # 1. Random orthogonal rotation matrix
    mat = rng.randn(dim, dim).astype(np.float64)
    q, _ = np.linalg.qr(mat)
    rotation = q.astype(np.float32)
    
    # 2. Data-driven quantization levels from normal quantiles
    # Sample 1M points from N(0,1) and compute quantiles
    num_levels = 2 ** bits
    # Use deterministic quantile positions
    probs = np.linspace(0, 1, num_levels)
    # Quantiles of standard normal (using percent point function)
    from scipy.stats import norm as norm_dist
    levels = norm_dist.ppf(probs).astype(np.float32)
    
    return TurboQuantV2State(
        dim=dim,
        bits=bits,
        levels=levels,
        rotation=rotation,
    )


def compress_v2(
    x: np.ndarray,
    state: TurboQuantV2State,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compress vector x using Algorithm B (levels + residual).
    
    Args:
        x: Input vector of shape (dim,), float32
        state: TurboQuantV2State
    
    Returns:
        Tuple of (indices, signs, scale):
            indices: uint8 array (dim,) — level indices (0..2^bits-1)
            signs:   int8 array (dim,) — residual signs (-1 or 1)
            scale:   float16 — residual scale factor (std of residual)
    """
    x = x.astype(np.float32, copy=False)
    
    # Normalize
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        return (
            np.zeros(state.dim, dtype=np.uint8),
            np.ones(state.dim, dtype=np.int8),
            np.float16(0.0),
        )
    
    x_normed = x / norm
    rotated = state.rotation @ x_normed  # (dim,)
    
    # Quantize to nearest level
    # rotated[:, None] - levels[None, :] → (dim, num_levels)
    indices = np.argmin(
        np.abs(rotated[:, np.newaxis] - state.levels[np.newaxis, :]),
        axis=1,
    ).astype(np.uint8)
    
    # Residual (in rotated space)
    y_hat = state.levels[indices]  # (dim,)
    residual = rotated - y_hat  # (dim,)
    
    # 1-bit sign + scale encoding
    signs = np.sign(residual).astype(np.int8)
    signs[signs == 0] = 1
    scale = np.float16(np.std(residual))
    
    return indices, signs, scale


def prepare_query_v2(
    query: np.ndarray,
    state: TurboQuantV2State,
) -> np.ndarray:
    """Pre-compute rotated query for fast estimation.
    
    Args:
        query: Query vector of shape (dim,), float32
        state: TurboQuantV2State
    
    Returns:
        q_rot: (dim,) — rotation @ (query / ||query||)
    """
    query = query.astype(np.float32, copy=False)
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-12:
        return np.zeros(state.dim, dtype=np.float32)
    return state.rotation @ (query / q_norm)


def estimate_v2(
    q_rot: np.ndarray,
    state: TurboQuantV2State,
    indices: np.ndarray,
    signs: np.ndarray,
    scale: float,
    query_norm: float,
) -> float:
    """Estimate cosine similarity (inner product after normalization).
    
    Uses LUT trick:
      score = sum(levels[indices] * q_rot) + scale * sum(signs * q_rot)
    
    The first term is the base quantization score (like LUT lookup).
    The second term compensates for quantization error using sign+scale.
    
    Args:
        q_rot:     Pre-computed rotated query, (dim,)
        state:     TurboQuantV2State
        indices:   Level indices from compress_v2(), (dim,), uint8
        signs:     Residual signs from compress_v2(), (dim,), int8
        scale:     Residual scale factor
        query_norm: L2 norm of original query vector
    
    Returns:
        Estimated inner product (float32)
    """
    # LUT score: levels[indices] · q_rot
    level_vals = state.levels[indices.astype(np.int32)]  # (dim,)
    base_score = float(np.dot(level_vals, q_rot))
    
    # Residual compensation: scale * signs · q_rot
    res_comp = float(scale * np.dot(signs.astype(np.float32), q_rot))
    
    # Scale back by query norm for true inner product estimate
    return (base_score + res_comp) * query_norm


def compress_batch_v2(
    vectors: np.ndarray,
    state: TurboQuantV2State,
) -> list:
    """Compress a batch of vectors."""
    return [compress_v2(vectors[i], state) for i in range(len(vectors))]
