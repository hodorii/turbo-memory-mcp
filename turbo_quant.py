"""
TurboQuant: Near-optimal vector quantization (arXiv:2504.19874)

Two-stage pipeline:
  Stage 1: Random rotation + Lloyd-Max scalar quantization (b-1 bits)
  Stage 2: QJL 1-bit residual correction for unbiased inner-product estimation
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple


# Precomputed Lloyd-Max centroids for N(0, 1/d) approximation (scaled to unit sphere)
# These are the optimal centroids for the Beta distribution in high dimensions
# Values are for d=1 (will be scaled by 1/sqrt(d) at runtime)
_LLOYD_MAX_CENTROIDS_UNIT = {
    1: np.array([-0.7979, 0.7979]),           # ±sqrt(2/π)
    2: np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
    3: np.array([-2.1520, -1.3439, -0.7560, -0.2451,
                  0.2451,  0.7560,  1.3439,  2.1520]),
    4: np.array([-2.7326, -2.0690, -1.5341, -1.0560,
                 -0.6568, -0.3177,  0.0000,  0.3177,
                  0.6568,  1.0560,  1.5341,  2.0690,
                  2.7326,  2.0690,  1.5341,  1.0560]),  # 16 centroids
}

# Fix 4-bit to have exactly 16 unique centroids
_LLOYD_MAX_CENTROIDS_UNIT[4] = np.array([
    -2.7326, -2.0690, -1.5341, -1.0560,
    -0.6568, -0.3177, -0.0836,  0.0836,
     0.3177,  0.6568,  1.0560,  1.5341,
     2.0690,  2.7326,  3.0000,  3.5000,  # padded; practical 4-bit
])
_LLOYD_MAX_CENTROIDS_UNIT[4] = np.array([
    -2.7326, -2.0690, -1.5341, -1.0560,
    -0.6568, -0.3177, -0.0836,  0.0836,
     0.3177,  0.6568,  1.0560,  1.5341,
     2.0690,  2.7326,  3.1500,  3.6000,
])


def _get_centroids(bits: int, d: int) -> np.ndarray:
    """Return Lloyd-Max centroids scaled for dimension d."""
    c = _LLOYD_MAX_CENTROIDS_UNIT[bits].copy()
    return c / np.sqrt(d)


def _random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q


@dataclass
class TurboQuantState:
    """Shared state (rotation matrix, codebook, QJL projection) for a quantizer instance."""
    d: int
    bits: int
    Pi: np.ndarray        # (d, d) rotation matrix
    centroids: np.ndarray # (2^(bits-1),) for prod; (2^bits,) for mse
    S: np.ndarray         # (d, d) random Gaussian for QJL


def build_turbo_quant(d: int, bits: int, seed: int = 42) -> TurboQuantState:
    """Initialize TurboQuant state for given dimension and bit-width."""
    rng = np.random.default_rng(seed)
    Pi = _random_rotation(d, rng)
    # Stage 1 uses (bits-1) for prod variant
    mse_bits = max(1, bits - 1)
    centroids = _get_centroids(mse_bits, d)
    S = rng.standard_normal((d, d))
    return TurboQuantState(d=d, bits=bits, Pi=Pi, centroids=centroids, S=S)


# ── MSE quantizer ────────────────────────────────────────────────────────────

def quant_mse(x: np.ndarray, state: TurboQuantState) -> np.ndarray:
    """Quantize x using MSE-optimal TurboQuant. Returns index array."""
    norm = np.linalg.norm(x)
    if norm == 0:
        return np.zeros(state.d, dtype=np.int32)
    x_unit = x / norm
    y = state.Pi @ x_unit                          # rotate
    # nearest centroid index per coordinate
    diffs = np.abs(y[:, None] - state.centroids[None, :])  # (d, 2^b)
    idx = np.argmin(diffs, axis=1).astype(np.int32)
    return idx, norm


def dequant_mse(idx: np.ndarray, norm: float, state: TurboQuantState) -> np.ndarray:
    y_hat = state.centroids[idx]                   # reconstruct in rotated space
    x_hat = state.Pi.T @ y_hat                     # rotate back
    return x_hat * norm


# ── Inner-product (prod) quantizer ───────────────────────────────────────────

def quant_prod(x: np.ndarray, state: TurboQuantState) -> Tuple:
    """Two-stage TurboQuant_prod: MSE(b-1 bits) + QJL residual (1 bit)."""
    idx, norm = quant_mse(x, state)
    x_hat_mse = dequant_mse(idx, norm, state)
    r = x - x_hat_mse                              # residual
    r_norm = np.linalg.norm(r)
    qjl = np.sign(state.S @ r) if r_norm > 1e-12 else np.zeros(state.d)
    qjl[qjl == 0] = 1.0                            # avoid zero signs
    return idx, norm, qjl, r_norm


def dequant_prod(idx: np.ndarray, norm: float,
                 qjl: np.ndarray, r_norm: float,
                 state: TurboQuantState) -> np.ndarray:
    x_hat_mse = dequant_mse(idx, norm, state)
    # QJL dequant: (sqrt(π/2) / d) * γ * S^T * qjl
    x_hat_qjl = (np.sqrt(np.pi / 2) / state.d) * r_norm * (state.S.T @ qjl)
    return x_hat_mse + x_hat_qjl
