"""
Memory MCP Server — TurboQuant-compressed vector memory

Tools:
  remember(text, embedding?)  — store a memory
  recall(query, top_k?)       — retrieve similar memories
  forget(id)                  — delete a memory
  memory_stats()              — compression statistics

Embedding: sentence-transformers all-MiniLM-L6-v2 (dim=384, local, no API key needed).
"""
import json
import sys
import os
import numpy as np

from memory_store import CompressedMemoryStore

# ── Config ────────────────────────────────────────────────────────────────────
DIM = 384       # all-MiniLM-L6-v2 output dimension
BITS = 3        # TurboQuant bit-width (3-bit, ~9x compression)
STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")

# Eager-load embedding model at startup
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer('all-MiniLM-L6-v2')

def _get_model():
    return _model

# Query embedding cache (text → normalized embedding)
_emb_cache: dict = {}
def _cached_encode(text: str) -> np.ndarray:
    if text not in _emb_cache:
        v = _model.encode(text, normalize_embeddings=True)
        _emb_cache[text] = v.astype(np.float32)
    return _emb_cache[text]

# ── Global store ─────────────────────────────────────────────────────────────
store = CompressedMemoryStore(path=STORE_PATH, dim=DIM, bits=BITS)


def _text_to_embedding(text: str) -> np.ndarray:
    return _cached_encode(text)


# ── MCP protocol helpers ──────────────────────────────────────────────────────

def _send(obj: dict):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _error(msg: str) -> dict:
    return {"error": {"message": msg}}


# ── Tool handlers ─────────────────────────────────────────────────────────────

def handle_remember(params: dict) -> dict:
    # Batch: {"texts": [...]} or single: {"text": "..."}
    texts = params.get("texts") or ([params.get("text", "").strip()] if params.get("text") else None)
    if not texts:
        return _error("text or texts is required")

    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64)
    ids = [store.add(t, e.astype("float32")) for t, e in zip(texts, embeddings)]
    try:
        store.save(STORE_PATH)
    except OSError:
        pass
    return {"ids": ids, "stored": len(ids)}


def handle_recall(params: dict) -> dict:
    query_text = params.get("query", "").strip()
    if not query_text:
        return _error("query is required")
    top_k = int(params.get("top_k", 5))

    raw_emb = params.get("embedding")
    if raw_emb is not None:
        q_emb = np.array(raw_emb, dtype=np.float32)
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm
    else:
        q_emb = _text_to_embedding(query_text)

    results = store.search(q_emb, top_k=top_k)
    return {
        "results": [
            {"id": r[0], "text": r[1], "score": round(r[2], 4)}
            for r in results
        ]
    }


def handle_forget(params: dict) -> dict:
    entry_id = params.get("id", "").strip()
    if not entry_id:
        return _error("id is required")
    deleted = store.delete(entry_id)
    if deleted:
        try:
            store.save(STORE_PATH)
        except OSError:
            pass
    return {"deleted": deleted}


def handle_memory_stats(_params: dict) -> dict:
    return store.stats()


HANDLERS = {
    "remember": handle_remember,
    "recall": handle_recall,
    "forget": handle_forget,
    "memory_stats": handle_memory_stats,
}

# ── MCP tool definitions ──────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "remember",
        "description": "Store one or multiple memories with TurboQuant-compressed embeddings (~10x compression). Pass 'texts' array for batch (single encode pass, much faster).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text":  {"type": "string", "description": "Single text to remember"},
                "texts": {"type": "array", "items": {"type": "string"}, "description": "Batch of texts (preferred for multiple memories)"},
            },
        },
    },
    {
        "name": "recall",
        "description": "Retrieve memories similar to a query using inner-product search",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {"type": "integer", "default": 5, "description": "Number of results"},
                "embedding": {
                    "type": "array", "items": {"type": "number"},
                    "description": f"Optional query embedding of dim={DIM}",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "forget",
        "description": "Delete a stored memory by id",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "name": "memory_stats",
        "description": "Show memory store statistics including compression ratio",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

# ── JSON-RPC request handler (shared by stdio and http) ──────────────────────

def _handle_request(req: dict) -> dict | None:
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "memory-mcp", "version": "1.0.0"},
            },
        }
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_params = params.get("arguments", {})
        handler = HANDLERS.get(tool_name)
        if handler is None:
            result = _error(f"Unknown tool: {tool_name}")
        else:
            try:
                result = handler(tool_params)
            except Exception as e:
                result = _error(str(e))
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                "isError": "error" in result,
            },
        }
    elif method == "notifications/initialized":
        return None  # no response
    else:
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


# ── stdio mode ────────────────────────────────────────────────────────────────

def run_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle_request(req)
        if resp is not None:
            _send(resp)


# ── HTTP mode ─────────────────────────────────────────────────────────────────

def run_http(host: str = "127.0.0.1", port: int = 8765):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class MCPHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # suppress access logs

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                return
            resp = _handle_request(req)
            payload = json.dumps(resp).encode() if resp is not None else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer((host, port), MCPHandler)
    print(f"memory-mcp HTTP ready on {host}:{port}", file=sys.stderr, flush=True)
    server.serve_forever()


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1]) if len(sys.argv) > sys.argv.index("--http") + 1 else 8765
        run_http(port=port)
    else:
        run_stdio()


def main():
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1]) if len(sys.argv) > sys.argv.index("--http") + 1 else 8765
        run_http(port=port)
    else:
        run_stdio()
