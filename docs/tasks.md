# Tasks — mcp-foundation

> Phase 1: Foundation (Provider 추상화, EDEN 통합, 패키징)

## Phase 1. Foundation Implementation

- [ ] **TASK-01** `memory_store.py`: `QuantizerProvider` ABC 인터페이스 정의
- [ ] **TASK-02** `memory_store.py`: 기존 로직 기반 `V2Provider`, `PaperProvider`, `FP32Provider` 구현
- [ ] **TASK-03** `src/turboquant/eden.py`: EdenQuantizer를 numpy로 포팅 (`EdenProvider`용)
- [ ] **TASK-04** `memory_store.py`: `EdenProvider` 추가 및 `MemoryStore`에 통합
- [ ] **TASK-05** `server.py`: `STORAGE_PATH` 환경변수 지원 및 `~/.turbo-memory/` 자동 생성 로직
- [ ] **TASK-06** `pyproject.toml`, `Makefile`: 패키징 및 CLI 진입점 (`turbo-memory-mcp`) 설정
