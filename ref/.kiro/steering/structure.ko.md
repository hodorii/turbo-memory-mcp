# 프로젝트 구조

```
turboquant/
├── src/
│   ├── turboquant/
│   │   ├── quantizer.py          # TurboQuantizer_V1 (3-bit 스칼라 양자화)
│   │   └── memory.py             # TurboQuantizer_V2 + TurboDiskStore
│   └── engine/
│       └── memory_engine.py       # FAISS + SQLite 하이브리드 엔진
├── tests/                         # 27개 테스트/스크립트 파일 (프레임워크 없음)
│   ├── test_quantizer.py          # V1 재구성 오류 테스트
│   ├── test_memory.py             # TurboMemoryStore 회귀 테스트 [깨짐]
│   ├── test_sanity.py             # TurboDiskStore 기본 테스트
│   ├── benchmarks.py              # V2 성능 벤치마크 [깨짐]
│   ├── benchmark_turbo_disk.py    # TurboDiskStore 100K 벤치마크
│   ├── verify_efficiency.py       # 압축 효율 검증
│   ├── profile_bge.py             # BGE-M3 프로파일링
│   ├── ingest_*.py                # 수집 파이프라인 스크립트
│   └── ...                        # 추가 질의/검색 테스트
├── data/                          # 심볼릭 링크 → ../turbo-memory-mcp/data
├── data_local/                    # 로컬 XML 데이터 (chosun, danjong, sillok)
├── benchmark_storage/             # 벤치마크 memmap 파일
├── sillok_turbo_storage/          # 실록 TurboDiskStore memmap 파일
├── sanity_storage/                # Sanity 테스트 memmap 파일
├── *.db                           # SQLite 데이터베이스 파일 (git 제외)
├── venv/                          # Python 가상환경
└── AGENTS.md                      # 프로젝트 퀵 레퍼런스
