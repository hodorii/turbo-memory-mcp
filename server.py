import json
import sys
import os
import numpy as np
import threading
from typing import Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

from memory_store import MemoryStore

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")
DIM = 384
BITS = 3

_encoder: Optional[SentenceTransformer] = None
_model_lock = threading.RLock() # 동일 스레드 재진입 허용
_print_lock = threading.Lock() # 출력용 락 분리
_executor = ThreadPoolExecutor(max_workers=10)
store = MemoryStore(path=STORE_PATH, dim=DIM, bits=BITS)


def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        with _model_lock:
            if _encoder is None:
                _encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return _encoder


@lru_cache(maxsize=1000)
def encode(text: str) -> np.ndarray:
    with _model_lock:
        return get_encoder().encode(text, normalize_embeddings=True).astype(np.float32)


def encode_batch(texts: list[str]) -> np.ndarray:
    with _model_lock:
        return get_encoder().encode(texts, normalize_embeddings=True, batch_size=64).astype(np.float32)


def safe_print(data: str):
    with _print_lock:
        sys.stdout.write(data + "\n")
        sys.stdout.flush()

# ... (handlers unchanged) ...

def dispatch(req: dict) -> Optional[dict]:
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "turbo-memory-mcp", "version": "0.1.0"},
                    "instructions": "TurboQuant-compressed vector memory store for MCP.",
                }
            }
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS}
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_params = params.get("arguments", {})
            if tool_name in HANDLERS:
                res = HANDLERS[tool_name](tool_params)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}],
                        "isError": False
                    }
                }
            return error(f"Unknown tool: {tool_name}", req_id)
        
        return None # Notifications (e.g. initialized)
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(e)},
            "isError": True
        }


def run_stdio():
    def worker(line):
        try:
            req = json.loads(line)
            resp = dispatch(req)
            if resp:
                safe_print(json.dumps(resp, ensure_ascii=False))
        except:
            pass

    for line in sys.stdin:
        if line.strip():
            _executor.submit(worker, line)


def run_http(port: int):
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            req = json.loads(self.rfile.read(content_length))
            resp = dispatch(req)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode('utf-8'))

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    safe_print(f"Server starting on http://127.0.0.1:{port} (Parallel Mode)")
    server.serve_forever()

def handle_remember(params: dict) -> dict:
    texts = params.get("texts") or ([params["text"].strip()] if params.get("text") else None)
    if not texts:
        return error("text or texts is required")
    embeddings = encode_batch(texts)
    ids = [store.add(t, e) for t, e in zip(texts, embeddings)]
    return {"ids": ids, "stored": len(ids)}


def handle_recall(params: dict) -> dict:
    query = params.get("query", "").strip()
    if not query:
        return error("query is required")
    raw = params.get("embedding")
    if raw is not None:
        q = np.array(raw, dtype=np.float32)
        if (n := float(np.linalg.norm(q))) > 0:
            q /= n
    else:
        q = encode(query)
    # Pass both query text and vector for hybrid (Vector + FTS5) search
    results = store.search(query, q, top_k=int(params.get("top_k", 5)))
    return {"results": [{"id": r[0], "text": r[1], "score": round(r[2], 4)} for r in results]}


def handle_forget(params: dict) -> dict:
    entry_id = params.get("id", "").strip()
    if not entry_id:
        return error("id is required")
    return {"deleted": store.delete(entry_id)}


def handle_memory_stats(_params: dict) -> dict:
    return store.stats()


HANDLERS = {
    "remember": handle_remember,
    "recall": handle_recall,
    "forget": handle_forget,
    "memory_stats": handle_memory_stats,
}

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
                "query":     {"type": "string", "description": "Search query text"},
                "top_k":     {"type": "integer", "default": 5, "description": "Number of results"},
                "embedding": {"type": "array", "items": {"type": "number"}, "description": f"Optional query embedding of dim={DIM}"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "forget",
        "description": "Delete a stored memory by id",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    },
    {
        "name": "memory_stats",
        "description": "Show memory store statistics including compression ratio",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

# ── JSON-RPC ──────────────────────────────────────────────────────────────────

def error(msg: str) -> dict:
    return {"error": {"message": msg}}


def dispatch(req: dict) -> Optional[dict]:
    req_id, method, params = req.get("id"), req.get("method", ""), req.get("params", {})

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": req_id, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "memory-mcp", "version": "1.0.0"},
        }}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        handler = HANDLERS.get(params.get("name"))
        try:
            result = handler(params.get("arguments", {})) if handler else error(f"Unknown tool: {params.get('name')}")
        except Exception as e:
            result = error(str(e))
        return {"jsonrpc": "2.0", "id": req_id, "result": {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            "isError": "error" in result,
        }}
    if method == "notifications/initialized":
        return None
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def send(obj: dict):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ── Transport modes ───────────────────────────────────────────────────────────

def run_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = dispatch(json.loads(line))
        except json.JSONDecodeError:
            continue
        if resp is not None:
            send(resp)


def run_http(port: int = 8765):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                resp = dispatch(json.loads(body))
            except json.JSONDecodeError:
                self.send_response(400); self.end_headers(); return
            payload = json.dumps(resp).encode() if resp is not None else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)

    print(f"memory-mcp HTTP ready on 127.0.0.1:{port}", file=sys.stderr, flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    if "--http" in sys.argv:
        idx = sys.argv.index("--http")
        port = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else 8765
        run_http(port)
    else:
        run_stdio()


if __name__ == "__main__":
    main()
