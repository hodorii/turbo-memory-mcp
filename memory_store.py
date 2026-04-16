import sqlite3
import numpy as np
from typing import List, Tuple

from turbo_quant import TurboQuantState, build_state, compress, estimate_inner_product


class MemoryStore:
    def __init__(self, path: str, dim: int = 384, bits: int = 3, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.state: TurboQuantState = build_state(dim, bits, seed)
        self._db = self._init_db(path)

    def _init_db(self, path: str) -> sqlite3.Connection:
        db = sqlite3.connect(path, check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id     TEXT PRIMARY KEY,
                text   TEXT NOT NULL,
                idx    BLOB NOT NULL,
                norm   REAL NOT NULL,
                qjl    BLOB NOT NULL,
                r_norm REAL NOT NULL
            )
        """)
        db.commit()
        return db

    def _next_id(self) -> str:
        count = self._db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        return f"mem_{count:06d}"

    def add(self, text: str, embedding: np.ndarray) -> str:
        idx, norm, qjl, r_norm = compress(embedding, self.state)
        entry_id = self._next_id()
        self._db.execute(
            "INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?)",
            (entry_id, text,
             idx.astype(np.int8).tobytes(), norm,
             np.sign(qjl).astype(np.int8).tobytes(), r_norm)
        )
        self._db.commit()
        return entry_id

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, float]]:
        rows = self._db.execute("SELECT id, text, idx, norm, qjl, r_norm FROM entries").fetchall()
        if not rows:
            return []
        scored = [
            (row_id, text, estimate_inner_product(
                query, self.state,
                np.frombuffer(idx_blob, dtype=np.int8).astype(np.int32), norm,
                np.frombuffer(qjl_blob, dtype=np.int8).astype(np.float32), r_norm
            ))
            for row_id, text, idx_blob, norm, qjl_blob, r_norm in rows
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def delete(self, entry_id: str) -> bool:
        cur = self._db.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        self._db.commit()
        return cur.rowcount > 0

    def stats(self) -> dict:
        n = self._db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        compressed_bits = n * ((self.bits - 1) * self.dim + self.dim + 64)
        original_bits = n * self.dim * 32
        return {
            "entries": n,
            "dim": self.dim,
            "bits": self.bits,
            "compression_ratio": round(original_bits / max(compressed_bits, 1), 2),
        }
