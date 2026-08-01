# turbo-disk-store — Requirements

## Project Description
Disk-backed compressed memory store using TurboQuantizer V2 quantization. Stores compressed vectors (indices + signs + scales) via numpy memmap and provides LUT (Look-Up Table) based fast approximate search.

## Language
en

## Stakeholders
- Systems needing large-scale vector storage without loading everything into RAM
- Applications requiring sub-second semantic search over 100K+ vectors
- Edge/constrained environments with limited memory

## Current Situation
- **TurboDiskStore** (`memory.py`): Stores compressed vectors (uint8 indices, int8 signs, float16 scales) via numpy memmap, 1M max capacity. Search uses LUT outer product trick for O(dim * levels) score computation + 1-bit residual compensation.
- **TurboMemoryStore (missing)**: 16 test files import `TurboMemoryStore` from `memory.py` but it does not exist. This class was likely removed during refactoring to `TurboDiskStore`. All these tests are broken.
- **Test Coverage**: `test_sanity.py` (basic add/search), `benchmark_turbo_disk.py` (100K performance), `verify_efficiency.py` (compression ratio check).

## Desired Change
- Document existing TurboDiskStore implementation
- Flag missing TurboMemoryStore as a known defect
- No new implementation — capture current state
