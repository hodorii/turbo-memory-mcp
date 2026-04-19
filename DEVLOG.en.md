# Development Log

## 2026-04-17

### Initial Implementation (02:22 - 02:28)

**Goal**: Implement an MCP server with compressed vector memory based on Google's TurboQuant paper.

**Paper Analysis**:
- Reference: arXiv:2504.19874 (ICLR 2026)
- Core: 2-stage pipeline (MSE Quantization + QJL Residual Correction)
- Theoretical Guarantees: MSE Distortion ≤ (√3·π/2) · (1/4^b), unbiased inner product estimation.

**Implementation Completed**:
1. `turbo_quant.py` — TurboQuant Core
   - Precomputed Lloyd-Max codebooks (1~4 bits)
   - 2-stage quantization/dequantization logic
   - Initialization of random rotation matrix Π and QJL matrix S.

2. `memory_store.py` — Compressed Storage
   - `MemoryStore`: add/search/delete/stats
   - Compression statistics (3-bit: ~10x vs FP32)

3. `server.py` — MCP Server
   - JSON-RPC over stdio
   - Tools: remember, recall, forget, memory_stats

**Test Results**:
- Confirmed unbiased inner product estimation (bias < 0.007, n=100 trials)
- Compression ratio: ~10x (3-bit vs FP32)
- MCP protocol functioning correctly.

### Embedding Model Integration (02:28 - 02:39)

**Attempt 1**: OpenAI API (text-embedding-3-small, dim=1536)
- Direct calls via `requests`.
- Issue: Disk space limitations for persistent storage.

**Improvement**:
- Switched to `sentence-transformers` (all-MiniLM-L6-v2, dim=384).
- Local execution, no API keys required.
- Lazy-load model only on first tool call.
- Absolute paths for `STORE_PATH`.

### MCP Client Registration (02:39 - 02:46)

**Registered with**:
- `kiro-cli`
- `gemini`
- `antigravity`
All clients share the same `memory.db`, ensuring cross-client memory synchronization.

---

## 2026-04-19

### Performance Optimization & Simplification (Applying "The Algorithm")

**Goal**: Maximize startup speed, optimize search algorithms, and stabilize parallel processing.

**Application of Elon Musk's "The Algorithm"**:
1. **Simplify Requirements**: Removed the requirement that the model must be loaded during the `initialize` phase.
2. **Delete Process**: Replaced the unbounded embedding cache with an LRU cache.
3. **Optimize**: Reduced search complexity from $O(N \cdot d^2)$ to $O(N \cdot d)$.
4. **Accelerate**: Introduced multi-threading to handle parallel agent requests.

**Key Improvements**:
1. **Lazy Loading (Startup Speed)**:
   - Deferred `SentenceTransformer` loading until the first tool call.
   - MCP `initialize` response time: several seconds → **0ms** (instant).

2. **Pre-projection (Search Speed)**:
   - Performed random rotation (Π) and QJL projection (S) of the query vector once outside the search loop.
   - Replaced matrix-vector multiplications ($d^2$) with vector-vector dot products ($d$) inside the loop.
   - Search speed: ~**8ms** for N=1000 items.

3. **Concurrency Support (Parallel Agents)**:
   - Implemented `ThreadPoolExecutor` (stdio) and `ThreadingHTTPServer` (HTTP).
   - Non-blocking structure allows multiple sub-agents to read/write memory simultaneously.

4. **Stability & Concurrency Control**:
   - Introduced `RLock` (Reentrant Lock) to prevent deadlocks during lazy loading.
   - Resolved `SentenceTransformer` thread-safety issues (prevented segmentation faults) with dedicated locks.
   - Isolated `safe_print` lock to prevent interleaved log output.

5. **Memory Efficiency**:
   - Managed embedding cache with `functools.lru_cache` (max size: 1000).

6. **Hybrid Search**:
   - Integrated SQLite FTS5-based keyword search.
   - Updated weights (80% Keyword / 20% Vector) to maximize precision for technical symbols.

7. **Morpheme Analysis**:
   - Integrated `kiwipiepy` for Korean particle separation and noun extraction.
   - Normalizes "알고리즘은" (algorithm+is) to "알고리즘" (algorithm) for accurate keyword matching.

**Final Benchmarks**:
- **Startup Latency**: 0.00ms (Initialize response)
- **Search Efficiency**: 9.42ms (Avg for N=1000, Morpheme Hybrid Mode)
- **Parallel Robustness**: Successfully handled 5 parallel calls without deadlocks or crashes.

---

## Technical Decisions

### Why TurboQuant?
- **Information-Theoretic Optimality**: Within 2.7x of the lower bound (ICLR 2026).
- **Data-oblivious**: Precomputable codebooks, no calibration needed.
- **Unbiased Inner Product**: QJL correction ensures no bias in attention scores.

### Hybrid Search Necessity
- Vector search is flexible but weak at exact identifier (e.g., TASK-01) matching.
- Combining with FTS5 ensures both conceptual and literal accuracy.

### Morpheme Analysis (Kiwi)
- Korean's agglutinative nature makes whitespace tokenization ineffective for keywords.
- Using `kiwipiepy` ensures high recall for technical terms attached to particles.

### Compression Trade-offs
- 2-bit: 14x compression, MSE 0.117
- 3-bit: 10x compression, MSE 0.030 (Recommended)
- 4-bit: 7x compression, MSE 0.009

Current Implementation: **3-bit** (Sweet spot).

## Future Directions

1. **Outlier Quantization**: Separate 2.5-bit strategy for outlier channels.
2. **Entropy Coding**: Huffman encoding for codebook pointers.
3. **GPU Acceleration**: CUDA kernels for parallel rotation.

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874)
- [Vadim's blog: TurboQuant 3-bit KV cache](https://vadim.blog/turboquant-3-bit-kv-cache-zero-loss)
- [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression)
