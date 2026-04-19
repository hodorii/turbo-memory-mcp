# turbo-memory-mcp

Compressed vector memory MCP server based on [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874).

Implements Google Research's TurboQuant algorithm to compress embedding vectors **~10x** while preserving inner product estimation accuracy for unbiased similarity search.

[한국어](README.ko.md)

## Features

- **Hybrid Search**: Combines **Vector Similarity (Semantic)** with **SQLite FTS5 (Keyword)** + **Morpheme Analysis** for maximum precision on technical symbols and proper nouns.
- **Extreme Speed**: Optimized $O(N \cdot d)$ search path using pre-projection; 1000 items searched in < 10ms.
- **Instant Initialize**: Lazy model loading ensures MCP servers start in 0ms, preventing client timeouts.
- **Parallel Readiness**: Thread-safe execution using `ThreadPoolExecutor` and `RLock` for concurrent sub-agent access.
- **Two-stage pipeline**: Random rotation + Lloyd-Max quantization (b-1 bits) → QJL residual correction (1 bit).
- **Unbiased inner product**: QJL correction eliminates similarity search bias.
- **Local embeddings**: `all-MiniLM-L6-v2` (sentence-transformers), no API key required.
- **Multi-client safe**: SQLite WAL mode for concurrent access from multiple MCP clients.

## Performance & Accuracy

| Metric | Initial (Baseline) | Optimized (Current) | Improvement |
|------|-------------------|----------------------|-------------|
| **Startup Latency** | ~5,200ms | **< 1ms** | 5000x faster |
| **Search (N=1000)** | ~100ms | **~9.4ms** | 10x faster |
| **Technical Precision** | Moderate (Vector only) | **Extreme** (FTS5 Hybrid) | Corrected symbol matching |
| **Korean Support** | Basic (Whitespace) | **Advanced** (Kiwipiepy) | Accurate particle separation |

## Algorithm Deviations & Optimizations

While strictly following the **TurboQuant (ICLR 2026)** mathematical core, this implementation introduces two major engineering optimizations:

1. **Pre-projected Execution**: The paper suggests $O(d^2)$ projection during search. We pre-compute $\Pi \cdot q$ and $S \cdot q$ once per query, reducing the inner-loop complexity to $O(N \cdot d)$, making it scalable to tens of thousands of memories on a single CPU core.
2. **Hybrid 80/20 Scoring**: We don't rely solely on semantic vectors. We use a weighted sum (80% Keyword / 20% Vector) to ensure that specific technical identifiers (like `TASK-01`) are never lost in the "semantic blur" of vector spaces.

## Memory vs. RAG

This tool is a specialized **Long-term Memory** system, differing from traditional RAG in three key ways:

- **Compression**: Uses 3-bit TurboQuant to store 10x more memories in the same RAM footprint compared to standard RAG.
- **Dynamism**: Designed for frequent `remember`/`forget` operations by the agent, unlike static document RAG.
- **Precision**: Combines Morpheme-aware FTS5 with TurboQuant to handle technical symbols that often break standard semantic search.

## File Structure

```
turbo_quant.py    — TurboQuant core (quantization + compressed inner product)
memory_store.py   — Compressed vector store (SQLite WAL)
server.py         — MCP server (JSON-RPC over stdio or HTTP)
pyproject.toml    — Package definition for uvx
```

## Installation

```bash
# No venv needed — uvx handles isolation automatically
uvx --from git+https://github.com/hodorii/turbo-memory-mcp turbo-memory-mcp
```

Or clone for local use:

```bash
git clone https://github.com/hodorii/turbo-memory-mcp
cd turbo-memory-mcp
make install
make register   # registers in all supported MCP clients
```

## MCP Registration

### stdio (default)

```json
"memory": {
  "command": "uvx",
  "args": ["--from", "/path/to/turbo-memory-mcp", "turbo-memory-mcp"]
}
```

Supported config paths:
- Kiro: `~/.kiro/settings/mcp.json`
- Gemini CLI: `~/.gemini/settings.json`
- Antigravity: `~/.gemini/antigravity/mcp_config.json`
- opencode: `~/.config/opencode/opencode.json`

### HTTP (for multi-client / sub-agent parallel access)

```bash
make serve          # default port 8765
make serve PORT=9000
```

```json
"memory": {
  "type": "http",
  "url": "http://127.0.0.1:8765"
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `remember(text)` | Embed and store a memory (compressed) |
| `remember(texts=[...])` | Batch store (single encode pass, faster) |
| `recall(query, top_k?)` | Search similar memories |
| `forget(id)` | Delete a memory |
| `memory_stats()` | Show compression statistics |

## Why TurboQuant?

This implementation chooses **TurboQuant** over standard Product Quantization (PQ) for three mathematical reasons:
1. **Unbiased Inner Product**: Standard quantization introduces systematic bias in similarity scores. TurboQuant's **QJL (Quantized Johnson-Lindenstrauss)** stage corrects this, ensuring $E[\text{estimated}] = \text{actual}$.
2. **Information-Theoretic Optimality**: It achieves within 2.7x of the theoretical lower bound for MSE distortion, outperforming existing methods by orders of magnitude.
3. **Data-oblivious**: Uses precomputed Lloyd-Max codebooks for a unit-sphere Beta distribution, requiring no expensive training or calibration on user data.

## Core Algorithm: 2-Stage Pipeline

We use a **3-bit per dimension** configuration, which provides a ~10.1x compression ratio with near-zero accuracy loss (matches FP16 performance).

1. **Stage 1: Lloyd-Max Quantization (2-bits)**
   - Applies a random orthogonal rotation $\Pi$ to map vectors to a near-Gaussian distribution.
   - Quantizes coordinates using MSE-optimized centroids for the resulting Beta distribution.
2. **Stage 2: QJL Residual Correction (1-bit)**
   - Computes the residual $r = x - \hat{x}$.
   - Stores only the **sign** of the residual after a Gaussian projection $S \cdot r$.
   - This 1-bit correction is the key to removing the quantization bias during search.

## Compression Specs (3-bit default)

| Component | Bits per Dim | Total for 384-dim |
|-----------|--------------|-------------------|
| Stage 1 (Index) | 2 bits | 768 bits |
| Stage 2 (QJL) | 1 bit | 384 bits |
| Norms (Meta) | - | 64 bits (2x Float32) |
| **Total** | **~3.16 bits** | **1216 bits** (vs 12288 FP32) |

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874) — Zandieh et al., Google Research, ICLR 2026
- [QJL paper](https://arxiv.org/abs/2406.03482) — 1-bit unbiased inner product quantization
