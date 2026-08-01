# sillok-ingestion — Requirements

## Project Description
Data ingestion pipeline for parsing, embedding, and indexing Joseon Wangjo Sillok (Joseon Annals) XML data into the TurboQuant storage systems. Supports both TurboDiskStore and MemoryEngine backends.

## Language
en

## Stakeholders
- Researchers needing to query large historical text corpora
- Systems ingesting 320K+ records from 674+ XML files
- QA engineers verifying end-to-end search accuracy

## Current Situation
- **Data Source**: 674 XML files in `data_local/chosun/` containing Joseon Annals paragraphs, plus `danjong/` and `sillok/` directories
- **Ingestion Approaches**:
  1. **MemoryEngine path**: XML parse → BGE-M3 embed → FAISS index + SQLite store (`ingest_parallel.py`, `ingest_all_batch.py`)
  2. **TurboDiskStore path**: XML parse → ko-sroberta embed → TurboQuantizer_V2 quantize → memmap store (`optimized_ingestion.py`, `batch_ingestion_v2.py`)
  3. **TurboMemoryStore path**: (BROKEN) Legacy scripts referencing non-existent `TurboMemoryStore` class
- **Parallelism**: multiprocessing (Pool) for XML parsing, batch embedding for model inference
- **Performance**: 200K+ records indexed in ~5 minutes (BGE-M3 batch embedding)
- **Test Coverage**: 16+ ingestion/query scripts, most with hardcoded query verification (e.g., "태종은 정도전을 어떻게 했는가?")

## Desired Change
- Document all existing ingestion pipeline variants
- Flag broken TurboMemoryStore scripts as known defects
- No new implementation — capture current state
