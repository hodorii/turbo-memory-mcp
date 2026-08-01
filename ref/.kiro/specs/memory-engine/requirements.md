# memory-engine — Requirements

## Project Description
Hybrid memory engine combining FAISS (FlatIP inner product index) with SQLite persistent storage and BGE-M3 sentence embeddings (1024-dim) for semantic search over Korean historical texts.

## Language
en

## Stakeholders
- Historians and researchers analyzing Joseon Annals (Joseon Wangjo Sillok)
- AI agents needing persistent long-term memory with semantic retrieval
- Applications requiring both structured (SQL) and semantic (vector) search

## Current Situation
- **MemoryEngine** (`engine/memory_engine.py`): Simple hybrid architecture combining:
  - **SQLite**: Persistent storage with `id` and `text` columns, supports batch insert via `executemany`
  - **FAISS IndexFlatIP**: Inner product index for maximum inner product search, 1024-dim (BGE-M3)
  - **SentenceTransformer (BGE-M3)**: Multilingual embedding model, loaded as module-level singleton
- **Ingestion Pipeline** (tests/): Multiple scripts for parsing Joseon Annals XML files and ingesting into MemoryEngine:
  - `ingest_parallel.py`: multiprocessing XML parsing + batch ingest (5000/batch)
  - `ingest_all_batch.py`: batch ingest across all XML files (500/batch)
  - `ingest_with_progress.py`: progress bar variant with DB count tracking
  - `ingest_real_data.py`: sample ingest of 5 XML files
  - `profile_bge.py`: BGE-M3 embedding speed/throughput profiling
- **Key Limitation**: 1FAISS index is purely in-memory — no persistence across restarts, requires re-ingestion
- **Dimension Mismatch**: `benchmarks.py` uses 768d (ko-sroberta), `memory_engine.py` uses 1024d (BGE-M3)

## Desired Change
- Document existing MemoryEngine implementation as-is
- Flag lack of FAISS index persistence as known limitation
- Flag dimension mismatch (768d vs 1024d) as known issue
