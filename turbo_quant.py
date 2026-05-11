"""
TurboQuant Algorithm A — Lloyd-Max 2-bit + QJL 1-bit Residual Correction

Reference: arXiv:2504.19874 (TurboQuant)
            arXiv:2406.03482 (QJL)

2-stage compressor:
  Stage 1 (2-bit): Random orthogonal rotation → Lloyd-Max scalar quantization (4 levels)
  Stage 2 (1-bit): QJL (Quantized Johnson-Lindenstrauss) sign embedding for residual correction

Total: 3 bits per dimension + 2 scalar floats (norm, r_norm)
Theoretical compression vs FP32: ~10.1x
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Lloyd-Max centroids for Beta distribution (high-dimensional sphere coordinates)
# Source: Lloyd-Max algorithm applied to Beta(α, α) with α = (d/2 - 1) for d-sphere
_LLOYD_MAX_CENTROIDS = {
    1: np.array([-0.7979, 0.7979]),
    2: np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
    3: np.array([-2.1518, -1.3448, -0.5381, 0.2687, 1.0754, 1.8821, 2.6888, 3.4955]),
    4: np.array([-2.7332, -1.8216, -0.9117, 0.0, 0.9117, 1.8216, 2.7332]),
}


@dataclass
class TurboQuantState:
    """Shared state for TurboQuant compression and estimation.
    
    All clients must use the same state (same seed=42) for correctness.
    """
    dim: int                    # Vector dimension (384 for all-MiniLM-L6-v2)
    stage1_bits: int            # Bits for Stage 1 Lloyd-Max (default: 2)
    stage2_bits: int            # Bits for Stage 2 QJL (always 1)
    rotation: np.ndarray        # (dim, dim) orthogonal matrix Π
    centroids: np.ndarray       # (2^stage1_bits,) Lloyd-Max centroids, scaled by 1/√dim
    qjl_matrix: np.ndarray     # (dim, dim) standard Gaussian matrix S


def build_state(dim: int = 384, bits: int = 3, seed: int = 42) -> TurboQuantState:
    """Build TurboQuant state with fixed seed for reproducibility.
    
    Args:
        dim: Vector dimension
        bits: Total bits (stage1_bits + stage2_bits). stage2_bits is always 1.
              Default 3 means stage1=2, stage2=1.
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        TurboQuantState with initialized rotation, centroids, and QJL matrix
    """
    rng = np.random.RandomState(seed)
    
    # Stage 1 bits = total - 1 (QJL uses 1 bit)
    stage1_bits = bits - 1
    
    # 1. Random orthogonal rotation matrix via QR decomposition
    mat = rng.randn(dim, dim).astype(np.float64)
    q, _ = np.linalg.qr(mat)
    rotation = q.astype(np.float32)
    
    # 2. Lloyd-Max centroids
    raw_centroids = _LLOYD_MAX_CENTROIDS[stage1_bits].astype(np.float64)
    # Scale by 1/√dim for unit sphere coordinates
    centroids = (raw_centroids / np.sqrt(dim)).astype(np.float32)
    
    # 3. QJL Gaussian matrix
    qjl_matrix = rng.randn(dim, dim).astype(np.float32)
    
    return TurboQuantState(
        dim=dim,
        stage1_bits=stage1_bits,
        stage2_bits=1,
        rotation=rotation,
        centroids=centroids,
        qjl_matrix=qjl_matrix,
    )


def compress(
    x: np.ndarray,
    state: TurboQuantState,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Compress vector x using 2-stage TurboQuant.
    
    Args:
        x: Input vector of shape (dim,), float32
        state: TurboQuantState from build_state()
    
    Returns:
        Tuple of (idx, norm, qjl, r_norm):
            idx:    int32 array (dim,) — centroid indices (0..2^stage1_bits-1)
            norm:   float32 — L2 norm of x
            qjl:    int8 array (dim,) — QJL sign bits (-1 or 1)
            r_norm: float32 — L2 norm of residual
    """
    # Safety: ensure float32
    x = x.astype(np.float32, copy=False)
    
    # ── Stage 1: Lloyd-Max Quantization ──
    norm = float(np.linalg.norm(x))
    
    # Handle zero vector
    if norm < 1e-12:
        return (
            np.zeros(state.dim, dtype=np.int32),
            0.0,
            np.ones(state.dim, dtype=np.int8),
            0.0,
        )
    
    x_normalized = x / norm
    rotated = state.rotation @ x_normalized  # (dim,)
    
    # Nearest centroid per dimension
    # rotated[:, None] - centroids[None, :] → (dim, 2^stage1_bits)
    idx = np.argmin(
        np.abs(rotated[:, np.newaxis] - state.centroids[np.newaxis, :]),
        axis=1,
    ).astype(np.int32)
    
    # ── Stage 2: QJL Residual Correction ──
    # Dequantize stage 1
    # x_hat = Π^T @ centroids[idx] * norm
    x_hat = (state.rotation.T @ state.centroids[idx]) * norm
    
    residual = x - x_hat
    r_norm = float(np.linalg.norm(residual))
    
    # QJL sign embedding
    if r_norm > 1e-12:
        qjl_raw = state.qjl_matrix @ residual  # (dim,)
        qjl = np.sign(qjl_raw).astype(np.int8)
        # Ensure no zeros (map 0 → 1)
        qjl[qjl == 0] = 1
    else:
        qjl = np.ones(state.dim, dtype=np.int8)
    
    return idx, norm, qjl, r_norm


def prepare_query(
    query: np.ndarray,
    state: TurboQuantState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute query projections for fast inner product estimation.
    
    By pre-computing rotation @ query and sign(QJL @ query) once,
    the inner product loop becomes O(N·d) instead of O(N·d²).
    
    Args:
        query: Query vector of shape (dim,), float32
        state: TurboQuantState
    
    Returns:
        Tuple of (q_rot, q_qjl):
            q_rot: (dim,) — rotation @ query
            q_qjl: (dim,) — sign(QJL @ query), int8
    """
    query = query.astype(np.float32, copy=False)
    q_rot = state.rotation @ query
    q_qjl = state.qjl_matrix @ query  # raw projection (not signed)
    return q_rot, q_qjl


def estimate(
    q_rot: np.ndarray,
    q_qjl: np.ndarray,
    state: TurboQuantState,
    idx: np.ndarray,
    norm: float,
    qjl: np.ndarray,
    r_norm: float,
) -> float:
    """Estimate inner product <query, x> from compressed representation.
    
    Uses pre-computed query projections for O(d) computation.
    
    estimate = <centroids[idx], q_rot> * norm     (Stage 1)
             + sqrt(π/2)/d * r_norm * <qjl, q_qjl>  (Stage 2, QJL correction)
    
    The QJL correction is unbiased: E[estimate] = <query, x>
    
    Args:
        q_rot:  Pre-computed rotation @ query, (dim,)
        q_qjl:  Pre-computed sign(QJL @ query), (dim,), int8
        state:  TurboQuantState
        idx:    Centroid indices from compress(), (dim,), int32
        norm:   L2 norm of original vector
        qjl:    QJL sign bits from compress(), (dim,), int8
        r_norm: L2 norm of residual
    
    Returns:
        Estimated inner product (float32)
    """
    # Stage 1: centroid inner product via lookup
    # centroids[idx] → (dim,) array of centroid values
    stage1 = float(np.dot(state.centroids[idx], q_rot)) * norm
    
    # Stage 2: QJL unbiased correction
    # QJL estimator: ⟨q, r⟩ ≈ sqrt(π/2)/d · ⟨S@q, sign(S@r)⟩
    # This is unbiased: E[sqrt(π/2)·(sᵢᵀq)·sign(sᵢᵀr)] = ⟨q, r⟩
    qjl_dot = float(np.dot(qjl.astype(np.float32), q_qjl.astype(np.float32)))
    c = math.sqrt(math.pi / 2.0) / state.dim
    stage2 = c * r_norm * qjl_dot
    
    return stage1 + stage2


def compress_batch(
    vectors: np.ndarray,
    state: TurboQuantState,
) -> list:
    """Compress a batch of vectors.
    
    Args:
        vectors: (N, dim) array, float32
        state: TurboQuantState
    
    Returns:
        List of (idx, norm, qjl, r_norm) tuples
    """
    return [compress(vectors[i], state) for i in range(len(vectors))]
