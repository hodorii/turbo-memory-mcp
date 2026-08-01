# turbo-quantizer — Requirements

## Project Description
Vector quantization engine for compressing high-dimensional embeddings using random-rotation-based scalar quantization. Originally implemented TurboQuant-style V1 (basic 3-bit) and V2 (3-bit + 1-bit residual correction). Following analysis of the TurboQuant/EDEN/RaBitQ academic controversy, the project rebases on the superior EDEN (DRIVE/EDEN, NeurIPS 2021 / ICML 2022) algorithm.

## Language
en

## Stakeholders
- AI/ML researchers needing efficient embedding storage
- Developers working on memory-augmented AI agents
- Teams processing large-scale historical text corpora

## Current Situation
- **V1 (quantizer.py)**: Basic 3-bit scalar quantization with random rotation matrix (QR decomposition), uniform codebook (linspace), no residual correction, no scaling factor (S=1 fixed). Analogous to TurboQuant-mse (degenerate EDEN case).
- **V2 (memory.py)**: Improved 3-bit quantization with data-driven codebook (random sample quantiles), 1-bit residual encoding (sign + std scale per vector). Custom heuristic — not proper QJL or DRIVE.
- **Code Duplication**: V1 and V2 are separate files with no shared base class.
- **Academic Context**: The TurboQuant paper (arXiv 2504.19874) has been shown to be a suboptimal special case of EDEN (ICML 2022) with S=1 fixed instead of analytically optimal scaling. EDEN-biased achieves lower MSE across all dimensions and bit-widths. EDEN-unbiased outperforms TurboQuant-prod by >1 bit per coordinate (arXiv 2604.18555).

## Desired Change (V3 — EDEN-based)
Implement a V3 quantizer based on the EDEN algorithm with:
1. **Random rotation** via random orthogonal matrix (QR decomposition) — shared with V1/V2
2. **Lloyd-Max codebook** optimized for the Beta distribution (post-rotation distribution), replacing the current Gaussian assumption
3. **Optimal scaling factor S** in closed form — the key innovation EDEN has over TurboQuant:
   - **EDEN-biased mode**: S chosen to minimize reconstruction MSE
   - **EDEN-unbiased mode**: S chosen for unbiased reconstruction (E[x̂] = x)
4. **DRIVE 1-bit unbiased residual** replacing the current heuristic sign+std residual — lower variance (vNMSE 0.57 vs QJL's 1.57)
5. **Unified design**: V3 becomes the primary quantizer; V1/V2 are preserved as legacy

## Requirements (numeric IDs)

### 1. EDEN-biased Quantization (MSE-minimizing)
1.1 The quantizer MUST apply a random orthogonal rotation to input vectors
1.2 The quantizer MUST use a Lloyd-Max codebook derived from the post-rotation Beta distribution
1.3 The quantizer MUST compute the optimal scaling factor S_bias = ⟨y, q⟩ / ‖q‖² to minimize reconstruction MSE
1.4 The quantizer MUST support configurable bit-width b ∈ {1, 2, 3, 4}

### 2. EDEN-unbiased Quantization (Unbiased estimation)
2.1 The quantizer MUST support an unbiased mode with S_unbiased = ‖x‖² / ⟨y, q⟩
2.2 The unbiased mode MUST satisfy E[x̂] = x (zero-bias property)
2.3 The unbiased mode MUST produce lower inner-product error than biased mode

### 3. 1-bit DRIVE Residual (replacing current heuristic)
3.1 The 1-bit unbiased residual MUST use the DRIVE method (NeurIPS 2021)
3.2 DRIVE residual MUST have lower variance (vNMSE ≈ 0.57) than QJL (vNMSE ≈ 1.57)
3.3 The combined V3 quantizer MUST support a 2-pass mode: EDEN + DRIVE residual

### 4. Backward Compatibility & Code Quality
4.1 V1 and V2 implementations MUST remain functional (legacy)
4.2 V3 MUST be in a new file `src/turboquant/eden.py`
4.3 V3 MUST use type hints and dataclasses for configuration
4.4 V3 MUST support batched (N, dim) and single (dim,) input vectors
4.5 Known defect: rename import alias `TurboQuantizer` → `TurboQuantizer_V1` in test

### 5. Testing & Validation
5.1 Unit tests MUST verify MSE of EDEN-biased < TurboQuant-mse (S=1) baseline
5.2 Unit tests MUST verify unbiased property: E[x̂] ≈ x for EDEN-unbiased
5.3 Unit tests MUST verify DRIVE residual variance < QJL residual variance
5.4 Integration test MUST verify V3 works with TurboDiskStore
