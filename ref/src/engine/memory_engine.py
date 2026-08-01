"""Memory Engine: FAISS + SQLite hybrid with optional V3 EDEN compressed storage.

Architecture:
  - Search is ALWAYS done via FAISS IndexFlatIP (exact float32 inner product).
  - When use_quantization=True, embeddings are ALSO stored in TurboDiskStore
    (3-bit EDEN quantized) for compact persistent storage.
  - The quantized store is NEVER used as a search backend — search uses FAISS.

Both share SQLite for text storage and BGE-M3 for embeddings.
"""

import sqlite3
import os
import shutil
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# EDEN codebook pre-warming must run *before* faiss is imported, because
# FAISS initializes OpenMP thread pools that can deadlock with PyTorch's
# parallel CPU dispatcher during Beta.sample((100_000,)).
torch.set_num_threads(1)
import src.turboquant.eden  # noqa: F401  (side effect: _CODEBOOK_CACHE populated)

import faiss

# BGE-M3 로드 (성능과 다국어 지원이 뛰어남)
model = SentenceTransformer('BAAI/bge-m3')
DIM = 1024  # BGE-M3 기본 차원


class MemoryEngine:
    def __init__(self, db_path='memory.db', use_quantization=False, bits=3, eden_mode='biased'):
        """Initialize memory engine.

        Search is ALWAYS done via FAISS IndexFlatIP (exact float32 inner product).
        When use_quantization=True, embeddings are ALSO stored in a quantized
        TurboDiskStore for compact persistent storage (never used for search).

        Args:
            db_path: SQLite database path.
            use_quantization: If True, ALSO store embeddings in compressed format.
            bits: Quantization bits (2/3/4, only used when use_quantization=True).
            eden_mode: EDEN mode ('biased'/'unbiased', only used when use_quantization=True).
        """
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            'CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, text TEXT)'
        )

        self.use_quantization = use_quantization
        self.DIM = DIM
        self.db_path = db_path

        # Always create FAISS index for search
        self.index = faiss.IndexFlatIP(self.DIM)

        if use_quantization:
            # BGE-M3 loaded above holds MPS memory. Clear cache before
            # EdenQuantizer init, which allocates ~400 MB for Beta sampling.
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            from src.turboquant.eden import EdenConfig, EdenQuantizer
            from src.turboquant.memory import TurboDiskStore

            cfg = EdenConfig(dim=self.DIM, bits=bits, mode=eden_mode, residual_bits=1)
            eq = EdenQuantizer(cfg)

            # TurboDiskStore storage dir alongside the db file
            storage_dir = db_path.replace('.db', '_turbo')
            if os.path.exists(storage_dir):
                shutil.rmtree(storage_dir)

            self.store = TurboDiskStore(
                self.DIM, bits=bits, storage_dir=storage_dir, quantizer=eq
            )

    def add(self, text: str):
        self.add_batch([text])

    def add_batch(self, texts):
        # 1. DB 일괄 저장
        data_to_insert = [(text,) for text in texts]
        self.conn.executemany(
            "INSERT INTO memory (text) VALUES (?)", data_to_insert
        )
        self.conn.commit()

        # 2. 배치 임베딩
        embs = model.encode(texts).astype('float32')

        # 3. 정규화 (cosine similarity를 위한 L2 normalization)
        norms = np.linalg.norm(embs, axis=-1, keepdims=True)
        embs = embs / np.clip(norms, 1e-10, None)

        # 4. 항상 FAISS에 추가 (search 백엔드)
        self.index.add(embs)

        # 5. use_quantization 시 추가로 압축 저장
        if self.use_quantization:
            for emb in embs:
                self.store.add(torch.from_numpy(emb))

    def search(self, query, top_k=3):
        # 1. 질의 벡터화
        query_emb = model.encode([query]).astype('float32')

        # 2. 정규화
        norms = np.linalg.norm(query_emb, axis=-1, keepdims=True)
        query_emb = query_emb / np.clip(norms, 1e-10, None)

        # 3. 항상 FAISS로 검색 (양자화 여부 무관)
        return self._search_exact(query_emb, top_k)

    def _search_exact(self, query_emb, top_k):
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for i in range(top_k):
            idx = int(indices[0][i])
            if idx != -1:
                cursor = self.conn.execute(
                    "SELECT text FROM memory WHERE id = ?", (idx + 1,)
                )
                row = cursor.fetchone()
                if row:
                    results.append((row[0], float(scores[0][i])))
        return results

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Memory Engine Demo (FAISS exact search)")
    print("=" * 60)

    engine = MemoryEngine('/tmp/memory_demo.db')

    records = [
        "정도전은 재상 중심의 정치를 지향하였다.",
        "이방원은 왕자의 난을 통해 권력을 잡았다.",
        "태종은 정도전의 사병 혁파 정책에 반대하였다.",
        "단종은 수양대군에 의해 왕위에서 물러났다.",
        "세종대왕은 한글을 창제하였다.",
    ]
    engine.add_batch(records)

    query = "태종과 정도전의 갈등은?"
    print(f"\nQuery: {query}")
    for text, score in engine.search(query):
        print(f"  [{score:.4f}] {text}")

    engine.close()

    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Memory Engine Demo (V3 EDEN quantized search)")
    print("=" * 60)

    q_engine = MemoryEngine(
        '/tmp/memory_demo_quant.db', use_quantization=True, bits=3
    )
    q_engine.add_batch(records)

    print(f"\nQuery: {query}")
    for text, score in q_engine.search(query):
        print(f"  [{score:.4f}] {text}")

    q_engine.close()

    # Cleanup
    for p in ['/tmp/memory_demo.db', '/tmp/memory_demo_quant.db',
              '/tmp/memory_demo_quant_turbo']:
        if os.path.exists(p):
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
