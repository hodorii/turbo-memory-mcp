# turbo-quantizer — Research Log

## Summary
Comprehensive research conducted on the TurboQuant/EDEN/RaBitQ academic controversy to inform the V3 quantizer design. Decision: rebase from TurboQuant (Google ICLR 2026) to EDEN (NeurIPS 2021 / ICML 2022).

## Research Log

### 2026-05-09: TurboQuant Academic Controversy Investigation

**Source**: NextPlatform article + arXiv 2604.18555 + arXiv 2604.19528 + RaBitQ open letter + Google blog + ICLR OpenReview

**Key Findings**:

#### 1. TurboQuant is a Degenerate Case of EDEN
- TurboQuant-mse matches EDEN-biased at every step except one: it fixes scaling factor S=1 instead of computing the optimal closed-form S
- EDEN (ICML 2022) predates TurboQuant (ICLR 2026) by ~4 years
- EDEN evolved from DRIVE (NeurIPS 2021, 1-bit) generalized to arbitrary bit-widths
- Source: arXiv 2604.18555 (Amit Portnoy et al., EDEN authors)
- [arxiv.org/abs/2604.18555](https://arxiv.org/abs/2604.18555)

#### 2. RaBitQ (2024) Also Predates TurboQuant
- RaBitQ (SIGMOD 2024) by Jianyang Gao (ETH Zurich) first combined random rotations + vector quantization
- TurboQuant's second author Daliri contacted RaBitQ team in Jan 2025 for debugging help
- TurboQuant paper described RaBitQ as "grid-based PQ" omitting shared random rotation
- RaBitQ was tested on single-core CPU while TurboQuant on A100 GPU — unfair comparison
- Source: Jianyang Gao public statement, arXiv 2604.19528
- [medium.com/@gaojianyang0017](https://medium.com/@gaojianyang0017/turboquant-and-rabitq-what-the-public-story-gets-wrong-23df83209c22)

#### 3. EDEN Algorithm Mechanics
- 4 steps: Random Rotation → Scalar Quantization (Lloyd-Max) → Scaling (S) → Inverse Rotation
- S_bias = ⟨y, q⟩ / ‖q‖² minimizes MSE
- S_unbiased = ‖x‖² / ⟨y, q⟩ gives E[x̂] = x
- DRIVE (1-bit): vNMSE converges to π/2 - 1 ≈ 0.57 vs QJL ≈ π/2 ≈ 1.57 (2.75× lower variance)
- 2-bit EDEN beats 3-bit TurboQuant-prod; 1-bit EDEN beats 2-bit TurboQuant-prod
- Source: DRIVE NeurIPS 2021, EDEN ICML 2022

#### 4. Codebase Analysis (this project)
- V1 (quantizer.py): 32 lines, naive 3-bit, S=1 implicit, uniform codebook, no residual
- V2 (memory.py): 132 lines, 3-bit + custom 1-bit residual (sign + std), still S=1, heuristic residual
- Both use QR decomposition random rotation (V1: random, V2: seed=42)
- V2 search uses LUT-based fast scoring + residual compensation
- No test framework; individual file execution via `python file.py`
- BGE-M3 (1024d) for memory_engine.py; ko-sroberta (768d) for benchmarks

### Design Decisions

| Decision | Chosen | Rejected | Rationale |
|----------|--------|----------|-----------|
| Base algorithm | EDEN (DRIVE/EDEN) | TurboQuant, RaBitQ | EDEN is optimal, predates both, and our V1/V2 already structurally similar |
| Scaling factor | Closed-form S (biased/unbiased) | S=1 fixed | This is THE algorithmic gap between EDEN and TurboQuant |
| Codebook | Lloyd-Max for Beta distribution | Gaussian quantiles, uniform linspace | Post-rotation coordinates follow Beta distribution, not Gaussian |
| 1-bit residual | DRIVE | QJL, current heuristic (sign+std) | DRIVE: 2.75× lower variance than QJL |
| Code structure | New `eden.py` file | Modify V1/V2 in-place | Preserve backward compatibility; clear separation |

### Architecture Pattern Decision
- **Pattern**: Strategy + Factory — unified `EdenQuantizer` class with `mode="biased"|"unbiased"` parameter
- **Rationale**: EDEN-biased and EDEN-unbiased share rotation, codebook, and quantization — only S differs
- **Rejected**: Separate classes per mode (unnecessary duplication)
- **Rejected**: Modifying V1/V2 (would break existing tests and benchmarks)

### Risks
1. **Beta distribution Lloyd-Max**: Requires numerical integration or precomputed tables — moderate implementation complexity
2. **DRIVE integration**: Must ensure correct variance properties are maintained in batched mode
3. **Backward compatibility**: V1/V2 tests use `TurboMemoryStore` which doesn't exist (already broken) — pre-existing issue
