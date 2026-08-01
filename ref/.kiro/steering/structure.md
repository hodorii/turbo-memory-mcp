# Project Structure

```
turboquant/
├── src/
│   ├── turboquant/
│   │   ├── quantizer.py          # TurboQuantizer_V1 (3-bit scalar quantization)
│   │   └── memory.py             # TurboQuantizer_V2 + TurboDiskStore
│   └── engine/
│       └── memory_engine.py       # FAISS + SQLite hybrid engine
├── tests/                         # 27 test/script files (no test framework)
│   ├── test_quantizer.py          # V1 reconstruction error test
│   ├── test_memory.py             # TurboMemoryStore recall test [BROKEN]
│   ├── test_sanity.py             # TurboDiskStore sanity test
│   ├── benchmarks.py              # V2 performance benchmark [BROKEN]
│   ├── benchmark_turbo_disk.py    # TurboDiskStore 100K benchmark
│   ├── verify_efficiency.py       # Compression efficiency verification
│   ├── profile_bge.py             # BGE-M3 profiling
│   ├── ingest_*.py                # Ingestion pipeline scripts
│   └── ...                        # Additional query/search tests
├── data/                          # Symlink → ../turbo-memory-mcp/data
├── data_local/                    # Local XML data (chosun, danjong, sillok)
├── benchmark_storage/             # Benchmark memmap files
├── sillok_turbo_storage/          # Sillok TurboDiskStore memmap files
├── sanity_storage/                # Sanity test memmap files
├── *.db                           # SQLite database files (git-ignored)
├── venv/                          # Python virtual environment
└── AGENTS.md                      # Project quick reference
```
