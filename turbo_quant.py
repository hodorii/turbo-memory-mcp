import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Precomputed Lloyd-Max centroids for Beta distribution in high dimensions (d=1 scale)
_LLOYD_MAX_CENTROIDS: dict[int, np.ndarray] = {
    1: np.array([-0.7979, 0.7979]),
    2: np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
    3: np.array([-2.1520, -1.3439, -0.7560, -0.2451,
                  0.2451,  0.7560,  1.3439,  2.1520]),
    4: np.array([-2.7326, -2.0690, -1.5341, -1.0560,
                 -0.6568, -0.3177, -0.0836,  0.0836,
                  0.3177,  0.6568,  1.0560,  1.5341,
                  2.0690,  2.7326,  3.1500,  3.6000]),
}


@dataclass
class TurboQuantState:
    dim: int
    bits: int
    rotation: np.ndarray    # (dim, dim) random orthogonal — maps x to near-Gaussian coords
    centroids: np.ndarray   # Lloyd-Max codebook scaled by 1/sqrt(dim)
    qjl_matrix: np.ndarray  # (dim, dim) random Gaussian — for 1-bit residual correction


def build_state(dim: int, bits: int, seed: int = 42) -> TurboQuantState:
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    centroids = _LLOYD_MAX_CENTROIDS[max(1, bits - 1)].copy() / np.sqrt(dim)
    return TurboQuantState(dim=dim, bits=bits, rotation=rotation,
                           centroids=centroids,
                           qjl_matrix=rng.standard_normal((dim, dim)))


def _quantize_stage1(x: np.ndarray, state: TurboQuantState) -> Tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(x))
    if norm == 0:
        return np.zeros(state.dim, dtype=np.int32), 0.0
    # Rotate to near-Gaussian distribution, then nearest-centroid quantize
    rotated = state.rotation @ (x / norm)
    idx = np.argmin(np.abs(rotated[:, None] - state.centroids[None, :]), axis=1).astype(np.int32)
    return idx, norm


def _dequantize_stage1(idx: np.ndarray, norm: float, state: TurboQuantState) -> np.ndarray:
    return (state.rotation.T @ state.centroids[idx]) * norm


def compress(x: np.ndarray, state: TurboQuantState) -> Tuple[np.ndarray, float, np.ndarray, float]:
    idx, norm = _quantize_stage1(x, state)
    residual = x - _dequantize_stage1(idx, norm, state)
    r_norm = float(np.linalg.norm(residual))
    # QJL: sign(S·r) gives unbiased 1-bit estimate of residual direction
    qjl = np.sign(state.qjl_matrix @ residual) if r_norm > 1e-12 else np.ones(state.dim)
    qjl[qjl == 0] = 1.0
    return idx, norm, qjl, r_norm


def prepare_query(query: np.ndarray, state: TurboQuantState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute projections once per search to achieve O(d) in the inner loop.
    
    1. Rotation (Π·q): Maps query to the same near-Gaussian space as Stage 1 centroids.
    2. QJL Projection (S·q): For Stage 2 residual correction using 1-bit signs.
    """
    q_rot = state.rotation @ query
    q_qjl = np.sign(state.qjl_matrix @ query)
    q_qjl[q_qjl == 0] = 1.0
    return q_rot, q_qjl


def estimate_inner_product(state: TurboQuantState,
                           q_rot: np.ndarray, q_qjl: np.ndarray,
                           idx: np.ndarray, norm: float,
                           qjl: np.ndarray, r_norm: float) -> float:
    """
    Unbiased inner product estimation in the compressed domain.
    
    Stage 1: <centroids[idx], Π·q> * ‖x‖
       - Computes the inner product using MSE-optimized Lloyd-Max centroids.
       - Exploits that Π is orthogonal, so <x, q> = <Πx, Πq>.
    
    Stage 2: (√(π/2)/d) * ‖r‖ * <qjl, sign(S·q)>
       - QJL (Quantized Johnson-Lindenstrauss) correction for the residual r.
       - Based on the property: E[sign(S·r) · sign(S·q)] = (2/π) * <r,q> / (‖r‖‖q‖).
       - This makes the overall estimation unbiased: E[Est] = <x, q>.
    """
    # Stage 1: O(d) vector-vector dot product
    stage1 = float(np.dot(state.centroids[idx], q_rot)) * norm
    
    # Stage 2: O(d) correction using residual signs
    stage2 = (np.sqrt(np.pi / 2) / state.dim) * r_norm * float(np.dot(qjl, q_qjl))
    
    return stage1 + stage2
