"""EDEN-based vector quantizer with optimal scaling factors.

Implements the EDEN (DRIVE/EDEN) algorithm from NeurIPS 2021 / ICML 2022,
which provides analytically optimal scaling factors that TurboQuant omits.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor


_CODEBOOK_CACHE: dict[tuple[int, int], Tensor] = {}


# ── Bit-packing utilities ────────────────────────────────────────────────

def pack_bits(indices: Tensor, bits: int) -> Tensor:
    """Pack b-bit integer indices into uint8 bytes (LSB-first per byte).

    Fully vectorized — no Python loops over dim.

    Args:
        indices: [N] or [N, dim] uint8 tensor of indices in [0, 2^bits).
        bits: Number of bits per index (1-8).

    Returns:
        uint8 tensor of shape [n_bytes] or [N, n_bytes].
    """
    single = indices.dim() == 1
    if single:
        indices = indices.unsqueeze(0)
    N, dim = indices.shape
    n_bytes = (dim * bits + 7) // 8
    dev = indices.device

    # Flatten indices to [N, dim, 1], expand bits dim: [N, dim, bits]
    bit_slices = ((indices.unsqueeze(-1) >> torch.arange(bits, device=dev)) & 1).to(torch.uint8)
    bit_stream = bit_slices.reshape(N, dim * bits)  # [N, dim*bits]

    total_bits = n_bytes * 8
    if total_bits > dim * bits:
        pad = torch.zeros(N, total_bits - dim * bits, dtype=torch.uint8, device=dev)
        bit_stream = torch.cat([bit_stream, pad], dim=1)

    weights = (2 ** torch.arange(8, device=dev)).to(torch.uint8)
    out = (bit_stream.reshape(N, n_bytes, 8) * weights).sum(dim=-1).to(torch.uint8)

    if single:
        return out.squeeze(0)
    return out


def unpack_bits(packed: Tensor, bits: int, dim: int) -> Tensor:
    """Unpack b-bit indices from uint8 bytes.

    Fully vectorized — no Python loops over dim.

    Args:
        packed: uint8 tensor of shape [n_bytes] or [N, n_bytes].
        bits: Number of bits per index.
        dim: Expected dimension (number of indices to extract).

    Returns:
        uint8 tensor of shape [dim] or [N, dim].
    """
    single = packed.dim() == 1
    if single:
        packed = packed.unsqueeze(0)
    N, n_bytes = packed.shape
    dev = packed.device

    # Expand each byte into 8 bits: [N, n_bytes, 8]
    bit_slices = ((packed.unsqueeze(-1) >> torch.arange(8, device=dev)) & 1).to(torch.uint8)
    bit_stream = bit_slices.reshape(N, n_bytes * 8)[:, :dim * bits]

    # Group into bits-wide chunks
    total = dim * bits
    n_chunks = total // bits
    bit_groups = bit_stream[:, :n_chunks * bits].reshape(N, n_chunks, bits)
    weights = (2 ** torch.arange(bits, device=dev)).to(torch.uint8)
    out = (bit_groups * weights).sum(dim=-1).to(torch.uint8)

    if single:
        return out.squeeze(0)
    return out


def pack_signs(signs: Tensor) -> Tensor:
    """Pack int8 signs (-1/1) into 1-bit uint8 bytes.

    Fully vectorized. Sign(+1)=1, Sign(-1)=0 in packed form.

    Args:
        signs: int8 tensor of shape [dim] or [N, dim] with values -1 or +1.

    Returns:
        uint8 tensor of shape [n_bytes] or [N, n_bytes].
    """
    single = signs.dim() == 1
    if single:
        signs = signs.unsqueeze(0)
    N, dim = signs.shape
    dev = signs.device
    n_bytes = (dim + 7) // 8

    bits = (signs > 0).to(torch.uint8)
    bit_stream = bits.reshape(N, dim)  # [N, dim]
    total_bits = n_bytes * 8
    if total_bits > dim:
        pad = torch.zeros(N, total_bits - dim, dtype=torch.uint8, device=dev)
        bit_stream = torch.cat([bit_stream, pad], dim=1)

    weights = (2 ** torch.arange(8, device=dev)).to(torch.uint8)
    out = (bit_stream.reshape(N, n_bytes, 8) * weights).sum(dim=-1).to(torch.uint8)

    if single:
        return out.squeeze(0)
    return out


def unpack_signs(packed: Tensor, dim: int) -> Tensor:
    """Unpack 1-bit signs back to int8 (-1/1).

    Fully vectorized.

    Args:
        packed: uint8 tensor of shape [n_bytes] or [N, n_bytes].
        dim: Expected dimension.

    Returns:
        int8 tensor of shape [dim] or [N, dim] with values -1 or +1.
    """
    single = packed.dim() == 1
    if single:
        packed = packed.unsqueeze(0)
    N = packed.shape[0]
    dev = packed.device

    bit_slices = ((packed.unsqueeze(-1) >> torch.arange(8, device=dev)) & 1).to(torch.uint8)
    bit_stream = bit_slices.reshape(N, -1)[:, :dim]
    signs = torch.where(bit_stream > 0,
                        torch.tensor(1, dtype=torch.int8, device=dev),
                        torch.tensor(-1, dtype=torch.int8, device=dev))

    if single:
        return signs.squeeze(0)
    return signs


def packed_size(dim: int, bits: int) -> int:
    """Number of bytes needed to store dim elements at bits per element."""
    return (dim * bits + 7) // 8


@dataclass
class EdenConfig:
    dim: int
    bits: int = 3
    mode: str = "biased"
    residual_bits: int = 1
    seed: Optional[int] = None
    _rotation: Optional[Tensor] = field(default=None, repr=False)


def compute_beta_codebook(dim: int, bits: int, num_samples: int = 100_000) -> Tensor:
    """Compute Lloyd-Max codebook for the post-rotation coordinate distribution.

    After random rotation of a unit vector in R^d, each coordinate follows a
    distribution with density proportional to (1 - x^2)^{(d-3)/2} on [-1, 1].
    This is a scaled-shifted Beta(alpha, alpha) with alpha = (d-1)/2.

    Uses sampling-based Lloyd-Max iteration.
    """
    key = (dim, bits)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]

    # Isolate all tensor ops on CPU — this function is called *after* BGE-M3
    # has loaded weights on MPS, and Beta.sample(100_000) allocates ~400MB,
    # which triggers an MPS OOM hang if the default device leaks through.
    with torch.device('cpu'):
        alpha = (dim - 1) / 2.0
        beta_dist = torch.distributions.Beta(
            torch.tensor([alpha]),
            torch.tensor([alpha]),
        )
        samples = beta_dist.sample((num_samples,)).squeeze(-1)
        samples = 2.0 * samples - 1.0  # map to [-1, 1]

        levels = 2**bits
        quantile_bounds = torch.linspace(0, 1, levels + 1)
        quantile_midpoints = (quantile_bounds[:-1] + quantile_bounds[1:]) / 2.0
        codebook = torch.quantile(samples, quantile_midpoints)

        # Lloyd-Max iteration
        for _ in range(50):
            expanded = samples.unsqueeze(-1)
            dists = torch.abs(expanded - codebook.unsqueeze(0))
            assignments = torch.argmin(dists, dim=-1)
            new_codebook = torch.zeros_like(codebook)
            for i in range(levels):
                mask = assignments == i
                if mask.any():
                    new_codebook[i] = samples[mask].mean()
            # Handle empty bins: interpolate
            for i in range(levels):
                has_assignments = (assignments == i).any()
                if not has_assignments and new_codebook[i] == 0.0:
                    left = new_codebook[i - 1].item() if i > 0 else -1.0
                    right = new_codebook[i + 1].item() if i < levels - 1 else 1.0
                    new_codebook[i] = (left + right) / 2.0
            codebook = new_codebook

        codebook = codebook.sort()[0]
    _CODEBOOK_CACHE[key] = codebook
    return codebook


class EdenQuantizer:
    """EDEN vector quantizer with optimal scaling factor.

    Supports biased (MSE-minimizing) and unbiased (zero-bias) modes.
    """

    def __init__(self, config: EdenConfig):
        self.config = config
        self.dim = config.dim
        self.bits = config.bits
        self.mode = config.mode
        self.residual_bits = config.residual_bits

        if config.seed is not None:
            torch.manual_seed(config.seed)
        # Use CPU for QR decomposition (not supported on MPS)
        mat = torch.randn(config.dim, config.dim, device='cpu')
        q, _ = torch.linalg.qr(mat)
        self.rotation = q.float()

        self.codebook = compute_beta_codebook(config.dim, config.bits)
        self.levels = self.codebook.shape[0]

    def rotate(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return torch.matmul(x, self.rotation)
        return torch.matmul(x, self.rotation)

    def _inverse_rotate(self, y: Tensor) -> Tensor:
        if y.dim() == 1:
            return torch.matmul(y, self.rotation.t())
        return torch.matmul(y, self.rotation.t())

    def _scale_bias(self, y: Tensor, q_val: Tensor) -> Tensor:
        inner = torch.sum(y * q_val, dim=-1)
        q_norm_sq = torch.sum(q_val ** 2, dim=-1)
        q_norm_sq = torch.clamp(q_norm_sq, min=1e-10)
        return (inner / q_norm_sq).unsqueeze(-1)

    def _scale_unbias(self, x: Tensor, y: Tensor, q_val: Tensor) -> Tensor:
        x_norm_sq = torch.sum(x ** 2, dim=-1)
        inner = torch.sum(y * q_val, dim=-1)
        inner = torch.clamp(inner.abs(), min=1e-10)
        return (x_norm_sq / inner).unsqueeze(-1)

    def _quantize_scalar(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize each coordinate to nearest codebook entry."""
        diff = torch.abs(y.unsqueeze(-1) - self.codebook.unsqueeze(0).unsqueeze(0))
        indices = torch.argmin(diff, dim=-1)
        q_val = self.codebook[indices]
        return indices, q_val

    def quantize(self, x: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
        """Quantize vector(s) using EDEN quantization.

        Returns: (indices, r_signs, r_scale, S)
        """
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        y = self.rotate(x)
        indices, q_val = self._quantize_scalar(y)

        if self.mode == "unbiased":
            S = self._scale_unbias(x, y, q_val)
        else:
            S = self._scale_bias(y, q_val)

        r_signs: Optional[Tensor] = None
        r_scale: Optional[Tensor] = None

        if self.residual_bits >= 1:
            # DRIVE residual should be computed in ROTATED space (y - q_val)
            # because at search time we use: scores += (signs · y_query) * scale
            # This ensures consistency between encode and search
            residual_y = y - q_val  # [N, dim] in rotated space
            
            # Store signs in rotated space (matches how search uses them)
            r_signs = torch.sign(residual_y).to(torch.int8)
            
            # DRIVE Theorem 2: optimal scale = E[|r · y|] / dim
            # For query-unaware encoding, use mean |r| as estimate
            r_l1 = torch.sum(residual_y.abs(), dim=-1)
            r_scale = (r_l1 / self.dim).to(torch.float32)

        if single:
            indices = indices.squeeze(0)
            if r_signs is not None:
                r_signs = r_signs.squeeze(0)
            if r_scale is not None:
                r_scale = r_scale.squeeze(0)
            S = S.squeeze(0)

        return indices, r_signs, r_scale, S

    def decode(
        self,
        indices: Tensor,
        r_signs: Optional[Tensor] = None,
        r_scale: Optional[Tensor] = None,
        S: Optional[Tensor] = None,
    ) -> Tensor:
        """Reconstruct vector(s) from quantized representation.

        When S is None, uses S=1 (TurboQuant/naive mode for comparison).
        """
        single = (indices.dim() == 1)
        if single:
            indices = indices.unsqueeze(0)

        q_val = self.codebook[indices]
        q_rotated = self._inverse_rotate(q_val)

        if S is None:
            S = torch.ones(q_rotated.shape[0], 1, device=q_rotated.device)
        elif S.dim() == 0:
            S = S.unsqueeze(0).unsqueeze(-1)
        elif S.dim() == 1:
            S = S.unsqueeze(-1)

        x_hat = S * q_rotated

        if r_signs is not None and r_scale is not None:
            if r_signs.dim() == 1:
                r_signs = r_signs.unsqueeze(0)
            if r_scale.dim() == 0:
                r_scale = r_scale.unsqueeze(0)
            r_rotated = self._inverse_rotate(r_signs.float())
            x_hat = x_hat + r_scale.unsqueeze(-1) * r_rotated

        if single:
            x_hat = x_hat.squeeze(0)

        return x_hat


# ── Pre-warm codebook cache at import time ───────────────────────────────
# compute_beta_codebook allocates ~400 MB during Beta.sample(100_000).
# When called *after* BGE-M3 has loaded its weights on MPS, the MPS device
# may hang due to memory pressure.  Pre-computing the common codebooks here
# (at eden.py import time, before any model is loaded) caches them so that
# later EdenQuantizer.__init__ returns instantly from _CODEBOOK_CACHE.
_prewarm_configs = [(1024, 2), (1024, 3), (1024, 4), (768, 3)]
for _d, _b in _prewarm_configs:
    compute_beta_codebook(_d, _b)
