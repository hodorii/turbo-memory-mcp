# turbo-quantizer V3 — Implementation Tasks

## Implementation Notes
- Lloyd-Max codebook for Beta distribution requires numerical iteration; implement as static method with module-level caching
- S factor must be stored per vector as float32 (4 bytes/vector overhead, negligible vs 1024-dim embedding)
- DRIVE residual computes on reconstruction error r = x - x̂ (not on rotated residual y - q_val like V2)
- The search LUT must pre-multiply by S or apply S after gather

---

- [ ] 1. Foundation: test infrastructure and legacy fix
- [ ] 1.1 Fix V1 import alias in `test_quantizer.py`
  - `test_quantizer.py` imports `TurboQuantizer` but class is named `TurboQuantizer_V1`
  - Add `TurboQuantizer = TurboQuantizer_V1` alias in `quantizer.py`
  - Verify `python tests/test_quantizer.py` passes
  - _Requirements: 4.5_
  - _Boundary: quantizer.py, test_quantizer.py_

- [ ] 1.2 Create `tests/test_eden.py` test skeleton
  - Create `tests/test_eden.py` with import test and basic structure
  - Import `src.turboquant.eden.EdenQuantizer`, `EdenConfig`
  - Verify `python tests/test_eden.py` runs (may fail on EdenQuantizer not yet existing)
  - _Requirements: 5.1, 5.2, 5.3_
  - _Boundary: test_eden.py_

---

- [ ] 2. Core: Lloyd-Max codebook for Beta distribution
- [ ] 2.1 Implement `EdenConfig` dataclass + `compute_beta_codebook()`
  - `EdenConfig`: `dim`, `bits`, `mode="biased"`, `residual_bits=1`, `seed`
  - `compute_beta_codebook(dim, bits)` static method using scipy.stats.beta for initialization
  - Derive codebook from Beta(α, α) where α = (dim-1)/2
  - Implement Lloyd-Max iteration: alternate decision boundaries ↔ reconstruction levels
  - Module-level `_CODEBOOK_CACHE: dict[(dim, bits), Tensor]` for reuse
  - Verify codebook shape is `[2^bits]` and values are reasonable (within ±3σ of Beta)
  - _Requirements: 1.2, 1.4, 4.3_
  - _Boundary: eden.py_

---

- [ ] 3. Core: EdenQuantizer biased mode (EDEN-biased)
- [ ] 3.1 Implement `EdenQuantizer.__init__()`, rotation matrix, and `rotate()`
  - Generate random orthogonal matrix via QR decomposition (seed-controlled)
  - Store rotation as `[dim, dim]` float32 tensor
  - `rotate(x)` handles both `[d]` and `[N, d]` inputs
  - _Requirements: 1.1, 4.4_
  - _Boundary: eden.py_

- [ ] 3.2 Implement `quantize()` for biased mode
  - Rotate input: y = x @ rotation
  - Per-coordinate Lloyd-Max quantization: closest codebook entry
  - Compute S_bias = sum(y * q) / sum(q²) per vector
  - Inverse rotation with scaling: x̂ = S_bias · (q_val @ rotation^T)
  - Return `(indices, None, None, S)` (no residual when residual_bits=0)
  - Works for both `[d]` and `[N, d]` inputs
  - Observable: quantize + decode roundtrip preserves cosine similarity > 0.9 at b=3, d=128
  - _Requirements: 1.3, 1.4, 4.4_
  - _Boundary: eden.py_

- [ ] 3.3 Implement `decode()` for biased mode
  - `decode(indices, r_signs=None, r_scale=None, S=None) -> x_hat`
  - No residual path: x̂ = S · (codebook[indices] @ rotation^T)
  - _Requirements: 1.3_
  - _Boundary: eden.py_

---

- [ ] 4. Core: EDEN-unbiased mode + DRIVE 1-bit residual
- [ ] 4.1 Implement unbiased mode in `quantize()`
  - When mode="unbiased": compute S_unbias = sum(x²) / sum(y * q)
  - Document: E[x̂] = x property
  - Same codebook and rotation as biased mode — only S changes
  - _Requirements: 2.1, 2.2_
  - _Boundary: eden.py_

- [ ] 4.2 Implement DRIVE 1-bit residual
  - Enabled when `residual_bits >= 1`
  - Compute reconstruction: x̂ = S · inverse_rotate(q_val) — with current mode's S
  - Compute residual: r = x - x̂ (critical: on final reconstruction space, not rotated space)
  - r_signs = sign(r) ∈ {+1, -1} as int8
  - r_scale = mean(|r · q|) / dim per vector (DRIVE Theorem 2)
  - Store alongside indices and S for later decoding
  - Observable: with DRIVE enabled, reconstruction MSE < without DRIVE
  - _Requirements: 3.1, 3.3_
  - _Boundary: eden.py_

- [ ] 4.3 Implement `decode()` with DRIVE residual reconstruction
  - `decode(indices, r_signs, r_scale, S) -> x_hat`
  - Base reconstruction: x̂_base = S · (codebook[indices] @ rotation^T)
  - DRIVE correction: x̂ = x̂_base + r_scale · (r_signs @ rotation^T) — rotate residual signs back
  - _Requirements: 3.1_
  - _Boundary: eden.py_

---

- [ ] 5. Integration: TurboDiskStore + EdenQuantizer
- [ ] 5.1 Update `TurboDiskStore` to accept `EdenQuantizer`
  - `TurboDiskStore.__init__()` gains optional `quantizer=` parameter
  - If provided, use EdenQuantizer; if None, use existing TurboQuantizer_V2
  - Default stays V2 for backward compatibility
  - `add()` method stores indices, r_signs, r_scale, S from EdenQuantizer
  - Update memmap schema to include S (4 bytes per vector: float32)
  - Observable: `TurboDiskStore(dim, bits=3, quantizer=EdenQuantizer(...))` works
  - _Requirements: 4.1_
  - _Boundary: memory.py (TurboDiskStore)_

- [ ] 5.2 Update `TurboDiskStore.search()` for S factor
  - Read S per vector from storage
  - LUT scoring: `score = S[n] * sum(LUT[dim, indices[n]])` (per-vector scaling)
  - DRIVE residual compensation when r_signs/r_scale present
  - Observable: search with EdenQuantizer returns scores higher than V2 (better inner product preservation)
  - _Requirements: 3.3_
  - _Boundary: memory.py (TurboDiskStore)_

---

- [ ] 6. Validation: unit test suite and comparison benchmarks
- [ ] 6.1 MSE comparison: EDEN-biased vs S=1 baseline
  - Generate 1000 random vectors, d=128, b=3
  - Compute MSE(EDEN-biased) and MSE(S=1) with identical codebook and rotation
  - Assert: MSE(EDEN-biased) < MSE(S=1)
  - Sweep d ∈ {16, 64, 128, 256, 1024} and b ∈ {1,2,3,4}
  - Assert: gap closes as d increases (S_optimal → 1 for large d)
  - _Requirements: 5.1_
  - _Boundary: test_eden.py_

- [ ] 6.2 Unbiased property verification
  - Generate 10000 random vectors, d=1024, b=3, mode="unbiased"
  - Compute empirical mean error: mean(x̂ - x) over all vectors
  - Assert: ‖mean_error‖ < 0.01 (unbiased property holds empirically)
  - Compare with biased mode: biased mode should have ‖mean_error‖ > 0
  - _Requirements: 5.2_
  - _Boundary: test_eden.py_

- [ ] 6.3 DRIVE residual variance test
  - Implement QJL baseline for comparison
  - Generate 1000 vectors, d=1024, 1-bit quantize with DRIVE and QJL
  - Compute vNMSE for each: E[‖x - x̂‖²] / E[‖x‖²]
  - Assert: vNMSE(DRIVE) < vNMSE(QJL)  (≈0.57 vs ≈1.57)
  - _Requirements: 5.3_
  - _Boundary: test_eden.py_

---

- [ ] 7. Integration: E2E test with sillok real data (조선왕조실록)
- [ ] 7.1 Create `tests/test_eden_sillok.py`
  - Load 50 XML files from `data_local/chosun/*.xml`
  - Extract ~500 paragraphs via `paragraph` elements
  - Generate BGE-M3 (1024-dim) embeddings
  - Store in TurboDiskStore with EdenQuantizer (mode="biased", bits=3)
  - Search: "정도전의 정치 철학", "이방원의 권력 장악", "태조 이성계의 가계"
  - Verify: search completes, top-1 score > 0, latency < 0.1s
  - Observable: the test produces a search log with meaningful sillok results
  - _Requirements: 5.4_
  - _Boundary: test_eden_sillok.py, memory.py (TurboDiskStore)_
