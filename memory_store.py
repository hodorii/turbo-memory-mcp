import sqlite3
import json
import numpy as np
import struct
from datetime import datetime
from math import exp, ceil
from typing import List, Tuple, Optional, Literal

from turbo_quant import (
    TurboQuantState, build_state, compress, prepare_query, estimate as estimate_a,
)
from turbo_quant_v2 import (
    TurboQuantV2State, build_state_v2, compress_v2, prepare_query_v2, estimate_v2,
)
from turbo_quant_paper import (
    TurboQuantState as TurboQuantPaperState,
    compress as compress_paper,
    prepare_query as prepare_query_paper,
    estimate as estimate_paper,
    estimate_preunpacked,
    _unpack_numpy,
    batch_search,
)


class MemoryStore:
    """SQLite-backed memory with optional TurboQuant compression.

    compression=None:  FP32 embeddings stored as raw float32 blobs
    compression='algo_a':  Lloyd-Max 2-bit + QJL 1-bit
    compression='algo_b':  3-bit levels + 1-bit residual sign/scale
    compression='paper':  3-bit Beta Lloyd-Max + QJL + bit-packing
    """

    def __init__(self, path: str, compression: Optional[Literal['algo_a', 'algo_b', 'paper']] = None):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self.compression = compression
        self._state_a: Optional[TurboQuantState] = None
        self._state_b: Optional[TurboQuantV2State] = None
        self._state_paper: Optional[TurboQuantPaperState] = None
        self._paper_cache: dict = {}
        self._paper_cache_loaded: bool = False
        self._id_seq = 0
        self._init_db()

    def _get_state_a(self) -> TurboQuantState:
        if self._state_a is None:
            self._state_a = build_state(dim=384, bits=3, seed=42)
        return self._state_a

    def _get_state_b(self) -> TurboQuantV2State:
        if self._state_b is None:
            self._state_b = build_state_v2(dim=384, bits=3, seed=42)
        return self._state_b

    def _get_state_paper(self) -> TurboQuantPaperState:
        if self._state_paper is None:
            self._state_paper = TurboQuantPaperState.build(dim=384, b=3, seed=42)
        return self._state_paper

    def _init_db(self):
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                text TEXT,
                embedding BLOB,
                compression TEXT DEFAULT 'fp32' NOT NULL,
                importance REAL,
                created_at TIMESTAMP,
                category TEXT DEFAULT '',
                source_ref TEXT DEFAULT '',
                tags TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)
        # Migration: add columns if upgrading from old schema
        for col, typ in [('category', 'TEXT DEFAULT \'\''),
                         ('source_ref', 'TEXT DEFAULT \'\''),
                         ('tags', 'TEXT DEFAULT \'\''),
                         ('metadata', 'TEXT DEFAULT \'{}\'')]:
            try:
                self._db.execute(f"ALTER TABLE entries ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS archived_entries (
                id TEXT PRIMARY KEY, summary TEXT, period_start TIMESTAMP
            )
        """)
        try:
            self._db.execute("ALTER TABLE entries ADD COLUMN compression TEXT DEFAULT 'fp32'")
        except sqlite3.OperationalError:
            pass
        self._db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(text)")
        self._db.commit()

    # ── Compression ─────────────────────────────────────────────────────────

    def _pack_algo_a(self, dim: int, idx: np.ndarray, norm: float,
                     qjl: np.ndarray, r_norm: float) -> bytes:
        return struct.pack(f'<I{dim}if{dim}if',
                           dim, *idx.tolist(), norm, *qjl.tolist(), r_norm)

    def _pack_algo_b(self, dim: int, indices: np.ndarray,
                     signs: np.ndarray, scale: float) -> bytes:
        return struct.pack(f'<I{dim}B{dim}be',
                           dim, *indices.tolist(), *signs.tolist(), scale)

    def _pack_paper(self, state: TurboQuantPaperState, packed_idx, norm, qjl, r_norm) -> bytes:
        qjl_bytes = qjl.tobytes()
        packed_bytes = packed_idx.tobytes()
        header = struct.pack('<ii', len(qjl_bytes), len(packed_bytes))
        return header + struct.pack('<ffi', norm, r_norm, len(qjl_bytes)) + qjl_bytes + packed_bytes

    def _compress_vector(self, vec: np.ndarray) -> Tuple[bytes, str]:
        if self.compression is None or self.compression == 'fp32':
            return vec.astype(np.float32, copy=False).tobytes(), 'fp32'

        if self.compression == 'algo_a':
            state = self._get_state_a()
            idx, norm, qjl, r_norm = compress(vec, state)
            return self._pack_algo_a(state.dim, idx, norm, qjl, r_norm), 'algo_a'

        if self.compression == 'algo_b':
            state = self._get_state_b()
            indices, signs, scale = compress_v2(vec, state)
            return self._pack_algo_b(state.dim, indices, signs, scale), 'algo_b'

        if self.compression == 'paper':
            state = self._get_state_paper()
            p_idx, norm, qjl, r_norm = compress_paper(vec, state)
            return self._pack_paper(state, p_idx, norm, qjl, r_norm), 'paper'

        raise ValueError(f"Unknown compression: {self.compression}")

    def _score_algo_a(self, blob: bytes, q_rot: np.ndarray, q_qjl: np.ndarray) -> float:
        dim = struct.unpack_from('<I', blob, 0)[0]
        idx = np.frombuffer(blob, dtype=np.int32, count=dim, offset=4)
        norm = struct.unpack_from('<f', blob, offset=4 + dim * 4)[0]
        qjl = np.frombuffer(blob, dtype=np.int8, count=dim, offset=4 + dim * 4 + 4)
        r_norm = struct.unpack_from('<f', blob, offset=4 + dim * 4 + 4 + dim * 1)[0]
        return estimate_a(q_rot, q_qjl, self._get_state_a(), idx, norm, qjl, r_norm)

    def _score_algo_b(self, blob: bytes, q_rot: np.ndarray) -> float:
        dim = struct.unpack_from('<I', blob, 0)[0]
        indices = np.frombuffer(blob, dtype=np.uint8, count=dim, offset=4)
        signs = np.frombuffer(blob, dtype=np.int8, count=dim, offset=4 + dim * 1)
        scale = struct.unpack_from('<e', blob, offset=4 + dim * 2)[0]
        return estimate_v2(q_rot, self._get_state_b(), indices, signs, scale, query_norm=1.0)

    def _score_paper(self, rowid: int, blob: bytes, q_rot: np.ndarray, q_qjl: np.ndarray) -> float:
        cached = self._paper_cache.get(rowid)
        if cached is not None:
            indices, norm, qjl, r_norm = cached
            return estimate_preunpacked(indices, norm, qjl, r_norm, self._get_state_paper(), q_rot, q_qjl)
        n_qjl, n_packed = struct.unpack('<ii', blob[:8])
        norm = struct.unpack_from('<f', blob, 8)[0]
        r_norm = struct.unpack_from('<f', blob, 12)[0]
        packed = np.frombuffer(blob[20+n_qjl:20+n_qjl+n_packed], dtype=np.uint8).copy()
        indices = _unpack_numpy(packed, self._get_state_paper().dim)
        qjl = np.frombuffer(blob[20:20+n_qjl], dtype=np.int8).copy()
        self._paper_cache[rowid] = (indices, norm, qjl, r_norm)
        return estimate_preunpacked(indices, norm, qjl, r_norm, self._get_state_paper(), q_rot, q_qjl)

    # ── Paper Cache ──────────────────────────────────────────────────────────

    def _ensure_paper_cache(self):
        """Load all paper entries from DB, pre-unpack packed bits -> uint8 indices."""
        if self._paper_cache_loaded:
            return
        self._paper_cache = {}
        rows = self._db.execute(
            "SELECT rowid, embedding FROM entries WHERE compression='paper'"
        ).fetchall()
        for rowid, blob in rows:
            n_qjl, n_packed = struct.unpack('<ii', blob[:8])
            norm = struct.unpack_from('<f', blob, 8)[0]
            r_norm = struct.unpack_from('<f', blob, 12)[0]
            qjl = np.frombuffer(blob[20:20+n_qjl], dtype=np.int8).copy()
            packed = np.frombuffer(blob[20+n_qjl:20+n_qjl+n_packed], dtype=np.uint8).copy()
            indices = _unpack_numpy(packed, self._get_state_paper().dim)
            self._paper_cache[rowid] = (indices, norm, qjl, r_norm)
        self._paper_cache_loaded = True

    def _invalidate_paper_cache(self):
        self._paper_cache_loaded = False
        self._paper_cache = {}

    # ── Query Projections ───────────────────────────────────────────────────

    def _prepare_queries(self, query_vec: np.ndarray, rows: list):
        has_a = any(r[4] == 'algo_a' for r in rows)
        has_b = any(r[4] == 'algo_b' for r in rows)
        has_paper = any(r[4] == 'paper' for r in rows)
        q_a, q_qjl_a, q_b, q_qjl_p, q_rot_p = None, None, None, None, None
        if has_a:
            q_a, q_qjl_a = prepare_query(query_vec, self._get_state_a())
        if has_b:
            q_b = prepare_query_v2(query_vec, self._get_state_b())
        if has_paper:
            q_rot_p, q_qjl_p = prepare_query_paper(query_vec, self._get_state_paper())
        return q_a, q_qjl_a, q_b, q_qjl_p, q_rot_p

    # ── Public API ──────────────────────────────────────────────────────────

    def add(self, text: str, embedding: np.ndarray, metadata: dict = None,
            importance: float = 0.5, commit: bool = True):
        self._id_seq += 1
        entry_id = f"mem_{int(datetime.now().timestamp())}_{self._id_seq}"
        packed, algo = self._compress_vector(embedding)
        self._invalidate_paper_cache()
        # Extract typed fields from metadata, store raw metadata as JSON
        category = (metadata or {}).pop('category', '')
        source_ref = (metadata or {}).pop('source_ref', '')
        tags = (metadata or {}).pop('tags', '')
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        self._db.execute(
            "INSERT INTO entries (id, text, embedding, compression, importance, created_at, category, source_ref, tags, metadata) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (entry_id, text, packed, algo, importance, datetime.now(),
             category, source_ref, tags, meta_json))
        self._db.execute("INSERT INTO entries_fts (text) VALUES (?)", (text,))
        if commit:
            self._db.commit()
        return entry_id

    def sediment(self, threshold: float = 0.05):
        now = datetime.now()
        rows = self._db.execute(
            "SELECT id, text, importance, created_at FROM entries").fetchall()
        for rid, text, imp, ts in rows:
            dt = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S.%f")
            score = imp * exp(-0.01 * (now - dt).days / max(0.1, imp))
            if score < threshold:
                self._db.execute("INSERT INTO archived_entries VALUES (?,?,?)",
                                 (rid, text, ts))
                self._db.execute("DELETE FROM entries WHERE id = ?", (rid,))
                self._db.execute(
                    "DELETE FROM entries_fts WHERE rowid = (SELECT rowid FROM entries_fts WHERE text = ? LIMIT 1)",
                    (text,))
        self._db.commit()

    def search(self, query_text: str, query_vec: np.ndarray,
               top_k: int = 5, filters: str = None) -> List[Tuple[str, str, float, dict]]:
        query = "SELECT rowid, id, text, embedding, compression, importance, created_at, category, source_ref, tags, metadata FROM entries"
        if filters:
            query += f" WHERE {filters}"
        rows = self._db.execute(query).fetchall()
        if not rows:
            return []

        now = datetime.now()
        fts_scores = {}
        try:
            fts_rows = self._db.execute(
                "SELECT rowid, bm25(entries_fts) FROM entries_fts WHERE text MATCH ? ORDER BY bm25(entries_fts) LIMIT 50",
                (query_text,)).fetchall()
            fts_scores = {row[0]: -row[1] for row in fts_rows}
        except sqlite3.OperationalError:
            pass

        q_a, q_qjl_a, q_b, q_qjl_p, q_rot_p = self._prepare_queries(query_vec, rows)
        query_norm = float(np.linalg.norm(query_vec))

        scored = []
        for r in rows:
            rid, eid, text, blob, algo, imp, ts, cat, src, tags, meta_json = r
            try:
                if algo == 'fp32':
                    vec = np.frombuffer(blob, dtype=np.float32)
                    v_score = float(np.dot(query_vec, vec) / (
                            query_norm * np.linalg.norm(vec) + 1e-9))
                elif algo == 'algo_a':
                    v_score = self._score_algo_a(blob, q_a, q_qjl_a)
                elif algo == 'algo_b':
                    v_score = self._score_algo_b(blob, q_b)
                elif algo == 'paper':
                    v_score = self._score_paper(rid, blob, q_rot_p, q_qjl_p)
                else:
                    continue
            except Exception:
                continue

            k_score = fts_scores.get(rid, 0.0)
            hybrid_score = (k_score * 0.7) + (v_score * 0.3)
            dt = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S.%f")
            score = hybrid_score * imp * exp(
                -0.01 * (now - dt).days / max(0.1, imp))
            try:
                meta_dict = json.loads(meta_json) if meta_json else {}
            except (json.JSONDecodeError, TypeError):
                meta_dict = {}
            scored.append((eid, text, score, {
                'category': cat, 'source_ref': src, 'tags': tags,
                'importance': imp, 'created_at': str(ts)[:19],
                **meta_dict
            }))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def recall_deep(self, query: str) -> List[str]:
        return [r[0] for r in self._db.execute(
            "SELECT summary FROM archived_entries WHERE summary LIKE ?",
            (f'%{query}%',)).fetchall()]

    def delete(self, entry_id: str) -> bool:
        self._db.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        deleted = self._db.execute("SELECT changes()").fetchone()[0] > 0
        self._db.commit()
        self._invalidate_paper_cache()
        return deleted

    def stats(self) -> dict:
        count = self._db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        modes = self._db.execute(
            "SELECT compression, COUNT(*) FROM entries GROUP BY compression").fetchall()
        byte_sizes = {
    'fp32': 1536,
    'algo_a': ceil(384 * 3 / 8) + 4 + 4 + 384 + 4,
    'algo_b': 4 + 384 + 384 + 2,
    'paper': 8 + 4 + 4 + 384 + ceil(384 * 3 / 8),
}
        total_fp32 = count * 1536
        total_actual = sum(c * byte_sizes.get(m, 1536) for m, c in modes)
        return {
            "total_entries": count,
            "compression_modes": dict(modes),
            "fp32_equivalent_bytes": total_fp32,
            "actual_storage_bytes": total_actual,
            "compression_ratio": round(total_fp32 / max(total_actual, 1), 2),
        }

    def commit(self):
        self._db.commit()

    def close(self):
        self._db.close()
