# turbo-memory-mcp

Compressed vector memory MCP server based on [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874).

Implements Google Research's TurboQuant algorithm to compress embedding vectors **~10x** while preserving inner product estimation accuracy for unbiased similarity search.

[한국어](README.ko.md)

## Features

- **Two-stage pipeline**: Random rotation + Lloyd-Max quantization (b-1 bits) → QJL residual correction (1 bit)
- **Unbiased inner product**: QJL correction eliminates similarity search bias
- **Direct compressed search**: Inner product estimated without dequantization
- **Local embeddings**: `all-MiniLM-L6-v2` (sentence-transformers), no API key required
- **Multi-client safe**: SQLite WAL mode for concurrent access from multiple MCP clients
- **MCP standard**: Works with Gemini CLI, Kiro, Antigravity, opencode and any MCP client

## Compression

| bits | Ratio (vs FP32) | MSE distortion |
|------|----------------|----------------|
| 2    | ~14x           | 0.117          |
| 3    | ~10x           | 0.030          |
| 4    | ~7x            | 0.009          |

At 3-bit, matches FP16 LongBench score (50.06, per paper).

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

## How TurboQuant Works

```
input vector x
    │
    ▼ Stage 1 (b-1 bits): random rotation Π·x → Beta distribution → Lloyd-Max scalar quantization
    │
    ▼ Stage 2 (1 bit): residual r = x - x̂  →  sign(S·r)  [QJL, unbiased]
```

Search uses compressed representation directly:
```
score ≈ centroids[idx] · (Π·query) · norm        # Stage 1
      + (√π/2 / d) · r_norm · sign(S·query) · qjl  # Stage 2 QJL correction
```

Theoretical guarantee: MSE distortion ≤ (√3·π/2) · (1/4^b) — within 2.7x of information-theoretic lower bound.

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874) — Zandieh et al., Google Research, ICLR 2026
- [QJL paper](https://arxiv.org/abs/2406.03482) — 1-bit unbiased inner product quantization
