import sqlite3
import numpy as np
from typing import List, Tuple
from math import exp

from turbo_quant import TurboQuantState, build_state, compress, estimate_inner_product


class MemoryStore:
    # Common stopwords for English and Korean to improve BM25 relevance
    STOPWORDS = {
        # English
        "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was", "were",
        "to", "for", "with", "at", "by", "from", "on", "in", "out", "of", "about",
        # Korean
        "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "만", "에서",
        "이다", "하고", "하고는", "그리고", "그래서", "하지만", "그런데",
    }

    def __init__(self, path: str, dim: int = 384, bits: int = 3, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.state: TurboQuantState = build_state(dim, bits, seed)
        self._db = self._init_db(path)
        # Initialize Kiwi for Korean morphological analysis
        try:
            from kiwipiepy import Kiwi
            self.kiwi = Kiwi()
        except:
            self.kiwi = None

    def _filter_stopwords(self, text: str) -> str:
        """
        Extract nouns and remove common stopwords.
        Uses morphological analysis for Korean and simple tokenization for English.
        """
        # 1. Handle Korean with Kiwi if available
        if self.kiwi:
            # Extract nouns (NNG, NNP) and technical terms
            analysis = self.kiwi.analyze(text)
            tokens = []
            for token, pos, _, _ in analysis[0][0]:
                # NNG: General Noun, NNP: Proper Noun, SL: Foreign Language (English)
                if pos in ('NNG', 'NNP', 'SL', 'SN'):
                    if token.lower() not in self.STOPWORDS:
                        tokens.append(token.lower())
            
            if tokens:
                return " ".join(tokens)

        # 2. Fallback for English or if Kiwi is missing
        tokens = text.lower().split()
        filtered = [t for t in tokens if t not in self.STOPWORDS]
        return " ".join(filtered) if filtered else text

    def _init_db(self, path: str) -> sqlite3.Connection:
        db = sqlite3.connect(path, check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")
        
        # 1. Base table for compressed vectors
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

        # 2. FTS5 Virtual Table for keyword search
        try:
            db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                    id UNINDEXED,
                    text,
                    tokenize='unicode61' -- Better for mixed technical tokens
                )
            """)
            # Triggers are removed to handle preprocessing in Python
            db.execute("DROP TRIGGER IF EXISTS entries_ai")
            db.execute("DROP TRIGGER IF EXISTS entries_ad")
            db.execute("DROP TRIGGER IF EXISTS entries_au")
        except sqlite3.OperationalError:
            pass
            
        db.commit()
        return db

    def _next_id(self) -> str:
        res = self._db.execute("SELECT id FROM entries ORDER BY CAST(SUBSTR(id, 5) AS INTEGER) DESC LIMIT 1").fetchone()
        if not res:
            return "mem_000000"
        last_num = int(res[0].split("_")[1])
        return f"mem_{last_num + 1:06d}"

    def add(self, text: str, embedding: np.ndarray) -> str:
        idx, norm, qjl, r_norm = compress(embedding, self.state)
        entry_id = self._next_id()
        
        # 1. Store base data
        self._db.execute(
            "INSERT OR REPLACE INTO entries (id, text, idx, norm, qjl, r_norm) VALUES (?,?,?,?,?,?)",
            (entry_id, text,
             idx.astype(np.int8).tobytes(), norm,
             np.sign(qjl).astype(np.int8).tobytes(), r_norm)
        )
        
        # 2. Store preprocessed text in FTS for precision matching
        try:
            clean_text = self._filter_stopwords(text)
            self._db.execute(
                "INSERT OR REPLACE INTO entries_fts (id, text) VALUES (?,?)",
                (entry_id, clean_text)
            )
        except:
            pass

        self._db.commit()
        return entry_id

    def search(self, query_text: str, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Hybrid Search: Vector Similarity (TurboQuant) + Keyword Matching (FTS5 BM25).
        
        1. Keyword Search (BM25):
           - Uses SQLite FTS5 to find exact tokens/phrases.
           - Crucial for technical symbols (e.g., function names, task IDs).
           
        2. Vector Search (Semantic):
           - Uses TurboQuant compressed representations to find conceptual matches.
           - O(N*d) complexity due to pre-projection of query.
           
        3. Scoring:
           - Combines Vector Score (0.7 weight) and Keyword Score (0.3 weight).
           - Exact keyword matches get a significant boost to ensure precision.
        """
        # 1. Keyword search (BM25) via FTS5
        keyword_scores = {}
        try:
            # Preprocess: Lowercase, remove special chars, and filter stopwords
            normalized_q = "".join(c if c.isalnum() or c.isspace() else " " for c in query_text.lower())
            clean_q = self._filter_stopwords(normalized_q).strip()
            # FTS5 works better with OR for multi-token search to increase recall
            fts_q = " OR ".join(clean_q.split())

            if fts_q:
                # FTS5 MATCH provides high performance keyword lookup
                fts_rows = self._db.execute(
                    "SELECT id, bm25(entries_fts) FROM entries_fts WHERE text MATCH ? LIMIT 100",
                    (fts_q,)
                ).fetchall()
                # bm25 in SQLite: lower is better (usually negative). 
                # Normalize: Map roughly -1.0 or less to 1.0, and 0.0 to 0.0
                for row_id, b_score in fts_rows:
                    # bm25 is typically negative. Smaller (more negative) is a better match.
                    # Increased sensitivity: -1.0 is now a full 1.0 score
                    k_val = max(0.0, min(1.0, -b_score))
                    keyword_scores[row_id] = k_val
        except:
            pass

        # 2. Vector search (TurboQuant)
        rows = self._db.execute("SELECT id, text, idx, norm, qjl, r_norm FROM entries").fetchall()
        if not rows:
            return []
        
        from turbo_quant import prepare_query, estimate_inner_product
        # Pre-project query once to rotate it into the same space as stored centroids
        q_rot, q_qjl = prepare_query(query_vec, self.state)

        scored = []
        for row_id, text, idx_blob, norm, qjl_blob, r_norm in rows:
            v_score = estimate_inner_product(
                self.state, q_rot, q_qjl,
                np.frombuffer(idx_blob, dtype=np.int8).astype(np.int32), norm,
                np.frombuffer(qjl_blob, dtype=np.int8).astype(np.float32), r_norm
            )
            
            # 3. Hybrid scoring integration
            k_score = keyword_scores.get(row_id, 0.0)
            
            # Final score: Keyword (0.8) for precision, Vector (0.2) for context
            # High weight on keyword ensures exact matches (especially technical ones) rank first.
            final_score = (k_score * 0.8) + (v_score * 0.2)
            scored.append((row_id, text, final_score))

        # Rank by combined hybrid score
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def delete(self, entry_id: str) -> bool:
        # Sync with FTS table
        self._db.execute("DELETE FROM entries_fts WHERE id=?", (entry_id,))
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
