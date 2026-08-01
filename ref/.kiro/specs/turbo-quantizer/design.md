# turbo-quantizer V3 — Technical Design

## Overview

V3 rebases the quantization engine from TurboQuant (Google ICLR 2026) to EDEN (NeurIPS 2021 / ICML 2022), which provides analytically optimal scaling factors that TurboQuant omits. The result is a drop-in replacement for V2's quantizer that achieves lower MSE (biased mode) and lower inner-product error (unbiased mode) at the same bit-width, or matches accuracy at 1 bit fewer per coordinate.

## Goals / Non-Goals

### Goals
- Implement EDEN-biased (MSE-minimizing) and EDEN-unbiased (zero-bias) quantizers
- Use Lloyd-Max codebook derived from Beta distribution (post-rotation distribution)
- Provide DRIVE 1-bit unbiased residual as replacement for the current heuristic residual
- Achieve MSE below TurboQuant-mse (S=1) baseline at all practical dimensions (128-1024)
- Achieve unbiased property E[x̂] = x for EDEN-unbiased mode
- Preserve full backward compatibility for V1/V2

### Non-Goals
- Online codebook adaptation (future work)
- HIGGS-style blockwise Lloyd-Max (future work)
- AQUA-KV adaptive KV cache quantization (future work)
- NVFP4/hardware-specific quantization

## Out of Boundary (explicit)
- **FAISS/SQLite integration**: `memory_engine.py` hybrid engine and its sillok ingestion pipeline
- **Real-time codebook adaptation**: Dynamic retraining of codebooks as data distribution shifts
- **Hardware-specific optimization**: CUDA kernels, NPU, or custom accelerator backends
- **RaBitQ algorithm**: While RaBitQ shares the random rotation approach, V3 follows the EDEN lineage with closed-form scaling; RaBitQ-compatible mode is not provided
- **Weight quantization**: This design targets embedding/vector quantization only, not LLM weight quantization (HIGGS domain)
- **Cross-spec sillok/memory-engine**: Tests may consume sillok data for validation, but the data pipeline itself is owned by other specs

## Boundary Commitments

- **This spec owns**: The `EdenQuantizer` class, its configuration, and its test suite. The `TurboDiskStore` search pipeline that consumes it.
- **This spec does NOT own**: `memory_engine.py` (FAISS pipeline), ingestion scripts, sillok data parsing (those belong to `memory-engine` or `sillok-ingestion` specs).
- **Allowed dependencies**: `torch`, `numpy`, `scipy` (for Beta distribution numerical methods). No new external dependencies.
- **Revalidation triggers**: Changing the scaling factor formula, the codebook distribution assumption, or the rotation matrix construction invalidates the design's accuracy guarantees.

## Architecture

### Quantizer Hierarchy (V3 only)

```
EdenQuantizer (src/turboquant/eden.py)
├── __init__(dim, bits, mode="biased"|"unbiased")
├── rotation:  torch.Tensor [dim, dim]  # Random orthogonal matrix (QR)
├── codebook:  torch.Tensor [L]         # Lloyd-Max for Beta((d-1)/2, (d-1)/2)
├── quantize(x) → (indices, residual_signs, residual_scale, S)
│   ├── y = rotate(x)                       # Random rotation
│   ├── q_idx = closest_codebook(y)          # Lloyd-Max quantization per coord
│   ├── q_val = codebook[q_idx]
│   ├── S = compute_scale(x, y, q_val)      # EDEN-biased or EDEN-unbiased
│   ├── x_approx = S * inverse_rotate(q_val)
│   ├── residual = x - x_approx              # DRIVE 1-bit on the reconstruction error
│   ├── r_signs = sign(residual)            #    (not on y - q_val like V2)
│   └── r_scale = compute_drive_scale(...)  #    DRIVE's optimal scale
│   Returns: indices, r_signs, r_scale, S
│
├── encode(x) → packed_bytes                # Full encoding for storage
└── decode(packed) → x_hat                  # Full decoding from storage
```

### Data Flow (Storage + Search with V3)

```
Input Vector x [dim]
    │
    ▼
┌─────────────────────┐
│  Random Rotation     │  y = x @ rotation
│  (QR orthogonal)     │
└─────────┬───────────┘
          │ y [dim]
          ▼
┌─────────────────────┐
│  Lloyd-Max Scalar    │  for each coord: q[i] = argmin |y[i] - codebook|
│  Quantization (b-bit)│
└─────────┬───────────┘
          │ q_idx [dim] (uint8), q_val [dim]
          ▼
┌─────────────────────┐
│  Scaling Factor S    │  S_bias = ⟨y, q⟩ / ‖q‖²  or  S_unbias = ‖x‖² / ⟨y, q⟩
│  (closed-form)       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Inverse Rotation    │  x̂ = S · (q_val @ rotation^T)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  DRIVE 1-bit         │  r = x - x̂ (reconstruction error, NOT y - q_val)
│  Residual Encoding   │  r_signs = sign(r), r_scale = compute_drive_scale(r)
└─────────┬───────────┘
          │
          ▼
    Store: indices[dim] + r_signs[dim] + r_scale[1] + S[1]
```

### Search Flow (LUT-based, same as V2 but with S)

```
Query Vector q [dim]
    │
    ▼
┌─────────────────────┐
│  Rotate Query        │  y_q = q @ rotation
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Build LUT           │  LUT[d, L] = y_q[d] * codebook[L] × S
│  (pre-multiply S)    │  (S is stored per vector during encoding)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Score via LUT       │  score = Σ_d LUT[d, indices[n,d]]
│  (same as V2)        │  (already includes S scaling)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  DRIVE Residual      │  residual_score = r_scale[n] · Σ_d r_signs[n,d] · y_q[d]
│  Compensation        │  total = score + residual_score
└─────────┬───────────┘
          │
          ▼
    Return top_k results
```

## File Structure Plan

| File | Responsibility | Status |
|------|---------------|--------|
| `src/turboquant/eden.py` | NEW — `EdenQuantizer` class with biased/unbiased modes, Lloyd-Max Beta codebook, DRIVE residual, scaling factor computation | Create |
| `src/turboquant/quantizer.py` | LEGACY — V1 `TurboQuantizer_V1`, unchanged. Add import alias fix | Modify (alias) |
| `src/turboquant/memory.py` | LEGACY — V2 `TurboQuantizer_V2` + `TurboDiskStore`. Update `TurboDiskStore` to optionally accept `EdenQuantizer` | Modify (optional) |
| `tests/test_eden.py` | NEW — Unit tests for V3: MSE comparison vs S=1, unbiased property, DRIVE variance | Create |
| `tests/test_eden_sillok.py` | NEW — Integration test with sillok XML data + BGE-M3 | Create |
| `tests/test_quantizer.py` | LEGACY — Fix import: `TurboQuantizer` → `TurboQuantizer_V1` alias | Modify |

## Components & Interfaces

### EdenQuantizer (`src/turboquant/eden.py`)

**Intent**: EDEN-biased / EDEN-unbiased vector quantizer with Lloyd-Max Beta codebook and DRIVE 1-bit residual.

**Requirements**: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.3, 4.4

```
class EdenConfig:
    dim: int                    # Vector dimension
    bits: int = 3               # Quantization bits per coordinate (1-4)
    mode: str = "biased"        # "biased" for MSE-min, "unbiased" for zero-bias
    residual_bits: int = 1      # DRIVE 1-bit residual (default: 1, 0 to disable)
    seed: int | None = None     # For reproducible rotation (None = random)

class EdenQuantizer:
    def __init__(self, config: EdenConfig)
    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]
        # Returns: (indices [N,d] uint8, r_signs [N,d] int8|None, r_scale [N] float16|None, S [N] float32)
    def decode(self, indices: Tensor, r_signs: Tensor | None, r_scale: Tensor | None, S: Tensor) -> Tensor
        # Returns: x_hat [N,d]
    def rotate(self, x: Tensor) -> Tensor
        # Public for LUT-based search (used by TurboDiskStore)
    @property
    def scale_bias(self, y: Tensor, q: Tensor) -> Tensor
        # S_bias = ⟨y, q⟩ / ‖q‖²
    @property
    def scale_unbias(self, x: Tensor, y: Tensor, q: Tensor) -> Tensor
        # S_unbias = ‖x‖² / ⟨y, q⟩
```

### Codebook Generation (internal to EdenQuantizer)

**Requirements**: 1.2

- Static method `_compute_beta_codebook(dim: int, bits: int) -> Tensor`
- Uses Beta(α, α) where α = (dim - 1) / 2
- Lloyd-Max iteration: start with uniform quantiles of Beta distribution, then iteratively optimize decision boundaries and reconstruction levels
- Precompute once per (dim, bits) and cache
- Fallback: use scipy.stats.beta for quantile-based initialization

### Scaling Factor Computation

**Requirements**: 1.3, 2.1

- **EDEN-biased** (mode="biased"): `S_bias = sum(y * q) / sum(q²)` per vector where y = rotated x, q = codebook[y]
- **EDEN-unbiased** (mode="unbiased"): `S_unbias = sum(x²) / sum(y * q)` per vector
- Both are closed-form, O(d) per vector, no iteration needed

### DRIVE 1-bit Residual

**Requirements**: 3.1, 3.2, 3.3

- Compute reconstruction error: r = x - x̂ (NOT y - q_val like V2 does)
- r_signs = sign(r) ∈ {+1, -1}  (1 bit per coordinate)
- r_scale = E[|r·q|] / d  (DRIVE scale, see DRIVE NeurIPS 2021 Theorem 2)
- This differs from V2 which computed residual on y - q_val (pre-inverse-rotation) and used std(r) as scale
- Expected vNMSE ≈ 0.57 for DRIVE vs 1.57 for QJL (vs V2's heuristic which has no bounded variance)

## Testing Strategy

### Unit Tests (test_eden.py)

**Requirements**: 5.1, 5.2, 5.3, 5.4

| Test | What | How |
|------|------|-----|
| 5.1 MSE comparison | EDEN-biased MSE < TurboQuant-mse (S=1) | Quantize 1000 random vectors at d=128, b=3. Compare MSE of EDEN-biased vs same codebook with S=1. Expect: EDEN MSE < S=1 MSE. |
| 5.2 Unbiased property | E[x̂] ≈ x for EDEN-unbiased | Quantize 10000 random vectors at d=1024, b=3, mode=unbiased. Compute mean(x̂ - x). Expect: ‖mean‖ < 0.01. |
| 5.3 DRIVE variance | DRIVE vNMSE < QJL vNMSE | Implement QJL baseline. Compare 1-bit vNMSE. Expect: DRIVE ≈ 0.57, QJL ≈ 1.57. |
| 5.4 Batched/single | Both input shapes work | Test (N, d) and (d,) inputs produce consistent results. |
| 5.5 Bit-width sweep | MSE decreases with more bits | Test b=1,2,3,4. Expect: MSE(b+1) < MSE(b). |

### Integration Test with Sillok Data (test_eden_sillok.py)

**Requirements**: 5.4, implicit end-to-end validation

- Load 50 XML files from `data_local/chosun/*.xml`
- Extract ~500 paragraphs (as in `test_turbo_sillok.py`)
- Generate BGE-M3 embeddings (1024-dim)
- Store in TurboDiskStore configured with EdenQuantizer (mode="biased", bits=3)
- Search with sillok queries: "정도전의 정치 철학", "이방원의 권력 장악", "태조 이성계의 가계"
- Verify:
  - Quantization + search completes without error
  - Top-1 result has score > 0 (meaningful similarity)
  - Search latency < 0.1s for 500 vectors
  - MSE on reconstruction < V2 baseline

### Validation Data Pipeline

```
data_local/chosun/*.xml
    → XML parsing (paragraph elements)
    → BGE-M3 embedding (1024d)
    → EdenQuantizer encode
    → TurboDiskStore store
    → EdenQuantizer decode search
    → Compare: recall, MSE, latency vs V2
```

## Dependencies

| Dependency | Version | Role | Criticality |
|-----------|---------|------|-------------|
| torch | any | Tensor ops, QR decomposition, quantization | P0 |
| numpy | any | Type conversion, stats helpers | P1 |
| scipy | any | Beta distribution quantiles (codebook init) | P1 (optional fallback) |

No new external dependencies beyond what V1/V2 already use.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Beta Lloyd-Max convergence | Numerical instability for extreme (dim, bits) | Precomputed table, fallback to uniform quantiles |
| DRIVE residual not improving V2 heuristic on real data | No gain on sillok search | Benchmark both; V3 still has S factor advantage |
| S = 0 or NaN in edge cases | Division by zero | Clamp denominator: S = 1 when ‖q‖² < ε |

## Implementation Notes

1. **S factor must be stored per vector**: Unlike V2 which only stored indices + sign + std, V3 also needs S stored (one float32 per vector). This adds 4 bytes per vector — negligible vs 1024-dim float32 savings.
2. **LUT search must account for S**: The current V2 LUT computes `score = sum(LUT[dim, indices])`. V3 needs `score = S * sum(LUT[dim, indices])` or pre-multiply LUT by S per vector. For batched LUT, store S separately and multiply after gather.
3. **DRIVE residual is on x - x̂, not y - q̂**: This is the key difference from V2. V2 computes residual on the rotated space pre-inverse-rotation. DRIVE/EDEN computes it on the final reconstruction. This affects `TurboDiskStore.search()`.
4. **Codebook caching**: Lloyd-Max codebooks depend on (dim, bits). Cache precomputed codebooks in a module-level dict to avoid recomputation on every `EdenQuantizer` instantiation.
5. **V1 import fix**: `test_quantizer.py` does `from src.turboquant.quantizer import TurboQuantizer` but the class is `TurboQuantizer_V1`. Add `TurboQuantizer = TurboQuantizer_V1` alias.
