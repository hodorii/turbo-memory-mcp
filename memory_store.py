"""
Compressed vector memory store using TurboQuant.
Storage: SQLite with WAL mode (safe for multi-process access).
Search: direct inner-product estimation on compressed representation (no dequant).
"""
import sqlite3
import numpy as np
from typing import List, Tuple

from turbo_quant import TurboQuantState, build_turbo_quant, quant_prod, inner_prod_compressed

# TurboQuantState is deterministic given (dim, bits, seed) — no need to persist matrices.
_DEFAULT_SEED = 42


class CompressedMemoryStore:
    def __init__(self, path: str, dim: int = 384, bits: int = 3):
        self.dim = dim
        self.bits = bits
        self.path = path
        self.state: TurboQuantState = build_turbo_quant(dim, bits, _DEFAULT_SEED)
        self._db = self._open(path)

    def _open(self, path: str) -> sqlite3.Connection:
        db = sqlite3.connect(path, check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id      TEXT PRIMARY KEY,
                text    TEXT NOT NULL,
                idx     BLOB NOT NULL,
                norm    REAL NOT NULL,
                qjl     BLOB NOT NULL,
                r_norm  REAL NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        db.commit()
        return db

    def _next_id(self) -> str:
        row = self._db.execute("SELECT COUNT(*) FROM entries").fetchone()
        return f"mem_{row[0]:06d}"

    def add(self, text: str, embedding: np.ndarray) -> str:
        idx, norm, qjl, r_norm = quant_prod(embedding, self.state)
        entry_id = self._next_id()
        self._db.execute(
            "INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?)",
            (entry_id, text,
             idx.astype(np.int8).tobytes(), float(norm),
             np.sign(qjl).astype(np.int8).tobytes(), float(r_norm))
        )
        self._db.commit()
        return entry_id

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, float]]:
        rows = self._db.execute("SELECT id, text, idx, norm, qjl, r_norm FROM entries").fetchall()
        if not rows:
            return []

        scores = []
        for row_id, text, idx_blob, norm, qjl_blob, r_norm in rows:
            idx = np.frombuffer(idx_blob, dtype=np.int8).astype(np.int32)
            qjl = np.frombuffer(qjl_blob, dtype=np.int8).astype(np.float32)
            score = inner_prod_compressed(query, self.state, idx, norm, qjl, r_norm)
            scores.append((row_id, text, score))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

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
