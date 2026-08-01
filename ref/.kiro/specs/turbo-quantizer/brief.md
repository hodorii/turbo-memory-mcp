# Brief: turbo-quantizer (EDEN V3 + Bit-Packing)

## Problem
TurboQuant V1/V2 uses S=1 fixed scaling (degenerate EDEN), losing 10-90% MSE vs optimal. Storage is uint8-padded (8 bits per index), wasting 50-75% space for 2-4 bit quantization.

## Current State
- V1 (quantizer.py): 3-bit naive scalar quantizer, S=1, no residual
- V2 (memory.py): 3-bit + 1-bit residual (sign+std), uint8 indices, S=1
- V3 (eden.py): EDEN algorithm with optimal S, Beta codebook, DRIVE residual
- TurboDiskStore (memory.py): memmap with uint8/int8 padding

## Desired Outcome
1. EDEN V3 quantizer with optimal S factor, Beta Lloyd-Max codebook, DRIVE residual
2. Bit-packed storage (indices 2-4 bit, signs 1 bit) reducing per-vector storage 75%
3. Fast LUT search with on-the-fly unpack (vectorized, no Python loops)
4. Full backward compatibility with V1/V2

## Approach
Implement EdenQuantizer with EDEN (ICML 2022) algorithm. Replace Gaussian codebook with Beta(alpha,alpha) Lloyd-Max. Add optimal scaling S_biased/S_unbias. Add DRIVE 1-bit residual with l1/d scaling. Implement bit-packing with vectorized torch operations (no Python loops).

## Scope
- **In**: EdenQuantizer, EdenConfig, Beta codebook, biased/unbiased modes, DRIVE residual, bit-packing pack/unpack, TurboDiskStore packing integration
- **Out**: FAISS integration, GPU support, online learning, product quantization

## Boundary Candidates
- `src/turboquant/eden.py`: Quantizer implementation
- `src/turboquant/memory.py`: Storage integration

## Upstream / Downstream
- **Upstream**: TurboDiskStore (consumer of quantized indices)
- **Downstream**: turbo-memory-mcp (consumer of TurboDiskStore)

## Existing Spec Touchpoints
- **Extends**: turbo-quantizer (adds V3), turbo-disk-store (adds bit-packing)

## Constraints
- Python 3.9+, torch, numpy only (no new dependencies)
- V1/V2 must remain functional
