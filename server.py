import json
import os
import sys
import threading
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

from memory_store import MemoryStore
from concurrency import ConnectionPool, RetryPolicy

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_memory.db")
DIM = 384
MODEL_NAME = "intfloat/multilingual-e5-small"

COMPRESS = None
for arg in sys.argv:
    if arg == '--compress=algo_a':
        COMPRESS = 'algo_a'
    elif arg == '--compress=algo_b':
        COMPRESS = 'algo_b'
    elif arg == '--compress=paper':
        COMPRESS = 'paper'

_pool = ConnectionPool(path=STORE_PATH, pool_size=10, busy_timeout=5000)
_retry_policy = RetryPolicy()
_store = MemoryStore(pool=_pool, retry_policy=_retry_policy, compression=COMPRESS)
_encoder: SentenceTransformer | None = None
_model_lock = threading.RLock()
_executor = ThreadPoolExecutor(max_workers=10)


def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        with _model_lock:
            if _encoder is None:
                _encoder = SentenceTransformer(MODEL_NAME)
    return _encoder


@lru_cache(maxsize=1000)
def encode(text: str) -> np.ndarray:
    # E5 모델은 query/passage prefix 필요
    return get_encoder().encode(text, normalize_embeddings=True).astype(np.float32)


def encode_batch(texts: list[str]) -> np.ndarray:
    return get_encoder().encode(texts, normalize_embeddings=True, batch_size=64).astype(np.float32)


def encode_passage(text: str) -> np.ndarray:
    return encode(f"passage: {text}")


def encode_passage_batch(texts: list[str]) -> np.ndarray:
    prefixed = [f"passage: {t}" for t in texts]
    return encode_batch(prefixed)


def encode_query(text: str) -> np.ndarray:
    return encode(f"query: {text}")


# ── MCP Server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    "turbo-memory-mcp",
    instructions="TurboQuant-compressed vector memory store for MCP. "
                 "Store and retrieve memories using hybrid vector + FTS5 search. "
                 f"Compression mode: {'FP32' if COMPRESS is None else COMPRESS}. "
                 f"Embedding model: {MODEL_NAME} (multilingual, 384-dim).",
    host="127.0.0.1",
    port=8765,
)


@mcp.tool()
def remember(
    text: str | None = None,
    texts: list[str] | None = None,
    category: str = "",
    tags: str = "",
    source_ref: str = "",
    importance: float = 0.5,
    metadata: dict | None = None,
) -> str:
    """Store one or multiple memories with optional metadata.

    Args:
        text: Single text to remember.
        texts: Batch of texts (preferred for multiple memories).
        category: Memory category (e.g., session_context, source_code, report, thinking).
        tags: Comma-separated tags for filtering.
        source_ref: Source file path or reference URL.
        importance: Importance score 0.0-1.0 (default 0.5).
        metadata: Additional key-value metadata to store with memory.
    """
    entries = texts or ([text.strip()] if text else None)
    if not entries:
        return json.dumps({"error": "text or texts is required"})

    meta = dict(metadata or {})
    if category:
        meta['category'] = category
    if tags:
        meta['tags'] = tags
    if source_ref:
        meta['source_ref'] = source_ref

    embeddings = encode_passage_batch(entries)
    ids = [_store.add(t, e, dict(meta), importance) for t, e in zip(entries, embeddings)]
    return json.dumps({"ids": ids, "stored": len(ids)})


@mcp.tool()
def recall(
    query: str,
    top_k: int = 5,
    filters: str | None = None,
) -> str:
    """Retrieve memories similar to a query using hybrid vector + FTS5 search.

    Args:
        query: Search query text.
        top_k: Number of results (default 5).
        filters: Optional SQL WHERE clause for filtering
            (e.g., "category='source_code'" or "tags LIKE '%eden%'").
    """
    q = encode_query(query)
    results = _store.search(query, q, top_k=top_k, filters=filters)
    return json.dumps({
        "results": [
            {
                "id": r[0],
                "text": r[1],
                "score": round(r[2], 4),
                "metadata": r[3],
            }
            for r in results
        ]
    }, ensure_ascii=False)


@mcp.tool()
def forget(id: str) -> str:
    """Delete a stored memory by id.

    Args:
        id: The memory entry id to delete.
    """
    return json.dumps({"deleted": _store.delete(id)})


@mcp.tool()
def memory_stats() -> str:
    """Show memory store statistics including compression ratio."""
    return json.dumps(_store.stats(), ensure_ascii=False)


def main():
    use_sse = "--http" in sys.argv
    transport = "sse" if use_sse else "stdio"
    print(f"turbo-memory-mcp starting ({transport})", file=sys.stderr)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
