# Roadmap

## Overview

turbo-memory-mcp를 정식 오픈소스 MCP 서버로 출시. 현재 베이스라인(알고리즘 3종, SQLite+FTS5, FastMCP) 위에 EDEN V3 양자화 통합, Provider 추상화, 세션관리, 지식조직화, 멀티에이전트 동시접근을 단계적으로 추가한다.

## Approach Decision

- **Chosen**: Embedded SQLite (single-process WAL mode) + 계층적 feature 확장
- **Why**: npx-like UX 유지 (설치=실행). 서버 프로세스 불필요. 현재 규모에서 SQLite WAL로 50+ concurrent reader 커버 가능
- **Rejected alternatives**:
  - Server-daemon architecture: "npx처럼" UX에 위배, 설치复杂度 증가
  - Separate repos per feature: 오버엔지니어링, 단일 MCP 서버로 통합

## Scope

- **In**:
  - QuantizerProvider 추상화 인터페이스 설계
  - EdenProvider (EDEN S_bias/S_unbias + DRIVE) numpy 구현
  - Provider별 compress / prepare_query / score 일원화
  - DB 자동 생성 (`~/.turbo-memory/`), STORAGE_PATH 환경변수
  - pyproject.toml 보강, Makefile, pip/uvx 설치
  - CLI runner (`turbo-memory-mcp` command)
  - Session 관리 (create/list/switch/delete, session-scoped isolation)
  - Knowledge 조직화 (topic CRUD, memory-topic linking, project scope)
  - Multi-agent concurrent access (connection pooling, WAL tuning)
- **Out**:
  - GPU 가속 (CUDA) — Phase 5+
  - FAISS/ANN 인덱스 통합 — Phase 5+
  - Outlier 채널 별도 양자화 — Phase 5+
  - PyPI 배포 (일단 GitHub+uvx, 추후 PyPI)

## Constraints

- Python 3.10+ (sentence-transformers, numpy)
- all-MiniLM-L6-v2 (384d) 유지 (향후 BGE-M3 1024d 옵션 추가 가능)
- SQLite WAL + `check_same_thread=False`
- MCP Protocol 2024-11-05
- 기존 DB 스키마와의 하위호환성 (migration path 필요)

## Boundary Strategy

- **Why this split**: 각 spec이 독립적인 MCP tool set에 대응. Foundation은 인프라, Session/Memory는 저장 도메인, Knowledge는 조직화 도메인, Concurrency는 성능/안정성
- **Shared seams to watch**:
  - `memory_store.py`가 모든 phase에서 변경됨 → 공유 인터페이스(Provider)를 먼저 확정
  - DB 스키마 변경이 연쇄됨 → migration 전략 필요

## Specs (dependency order)

- [ ] mcp-foundation -- QuantizerProvider 추상화, EdenProvider 구현, 패키징, DB 자동생성, CLI. Dependencies: none
- [ ] session-management -- Session CRUD, 세션별 memory 격리, auto-tracking. Dependencies: mcp-foundation
- [ ] knowledge-organization -- Topic CRUD, memory-topic linking, project scope, 필터검색. Dependencies: session-management
- [ ] multi-agent-concurrency -- Connection pooling, WAL tuning, read/write lock, sub-agent API. Dependencies: knowledge-organization

## TurboQuant Verification Track (병렬)

- [ ] turboquant-verification -- V3 EDEN 알고리즘 검증. turboquant repo에서 진행. Dependencies: none
