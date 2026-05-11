# Brief: mcp-foundation

## Problem
- **현재 상황**: 하드코딩된 DB 경로, `if/elif`로 얽힌 양자화 알고리즘 구조, 표준화되지 않은 패키징으로 인해 오픈소스 확장이 불가능함.
- **Gap**: 즉시 사용 가능한 표준 설치 방식(`uvx`) 부재 및 알고리즘 교체 시 코드 수정 필요.

## Desired Outcome
- **표준화**: `pyproject.toml`, `Makefile`을 통해 `pip install` 또는 `uvx`로 즉시 실행 가능.
- **추상화**: `QuantizerProvider` 인터페이스 도입으로 양자화 알고리즘 교체 용이성 확보 (Eden, V2, Paper, FP32).
- **자동화**: 설정 없이 `turbo-memory-mcp` 실행 시 `~/.turbo-memory/` 자동 생성.

## Approach
- **Provider 패턴**: `QuantizerProvider` ABC 정의 → `EdenProvider`, `V2Provider` 구현 → 기존 `server.py` 코드에서 분리.
- **패키징**: `hatchling` 기반 `pyproject.toml` 작성, `Makefile`에 `install`, `serve`, `register` 타겟 명시.
- **경로 관리**: 표준 경로(`~/.turbo-memory/memory.db`) 자동 할당.

## Scope
- **In**: `QuantizerProvider` interface 및 구현, `pyproject.toml`, `Makefile`, DB 자동 생성, CLI 진입점.
- **Out**: 세션관리(Phase 2), 주제/프로젝트 관리(Phase 3), 동시성 제어(Phase 4).

## Boundary Candidates
- `QuantizerProvider`: 알고리즘 인터페이스 분리
- `CLI/Packaging`: 인프라 로직 분리

## Upstream / Downstream
- **Upstream**: turboquant repo (EDEN/V2 알고리즘 로직)
- **Downstream**: Session Management (Phase 2)

## Constraints
- Python 3.10+, SQLite WAL mode, FastMCP 
