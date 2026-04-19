# Requirements — turbo-memory-mcp

> EARS (Easy Approach to Requirements Syntax) 형식  
> 작성일: 2026-04-17

---

## 1. Feature 목록

| ID | Feature | 우선순위 |
|----|---------|---------|
| F-01 | 압축 벡터 메모리 저장 | Must |
| F-02 | 의미 기반 유사도 검색 | Must |
| F-03 | 메모리 삭제 | Must |
| F-04 | 압축 통계 조회 | Should |
| F-05 | 배치 저장 (단일 인코딩 패스) | Should |
| F-06 | MCP stdio 트랜스포트 | Must |
| F-07 | MCP HTTP 트랜스포트 | Should |
| F-08 | 다중 클라이언트 동시 접근 | Should |
| F-09 | 로컬 임베딩 모델 (API 키 불필요) | Must |
| F-10 | uvx 원클릭 설치 | Should |

---

## 2. Ubiquitous Requirements (항상 적용)

- **REQ-U-01**: 시스템은 모든 메모리 저장 시 TurboQuant 2단계 파이프라인(랜덤 회전 + Lloyd-Max 양자화 + QJL 잔차 보정)으로 압축하여야 한다.
- **REQ-U-02**: 시스템은 inner product 추정 시 편향(bias)이 없어야 한다 (QJL unbiased 보장).
- **REQ-U-03**: 시스템은 SQLite WAL 모드로 데이터베이스를 운영하여야 한다.
- **REQ-U-04**: 시스템은 JSON-RPC 2.0 프로토콜을 준수하여야 한다.
- **REQ-U-05**: 시스템은 실행 후 500ms 이내에 `initialize` 응답을 반환해야 하며, 무거운 모델 로딩은 실제 도구 호출 시점까지 유예(Lazy Load)해야 한다.

---

## 3. Event-Driven Requirements

### F-01 압축 벡터 메모리 저장

- **REQ-01-01**: When MCP 클라이언트가 `remember(text)` 를 호출하면, 시스템은 `all-MiniLM-L6-v2` 모델로 384차원 임베딩을 생성하고 3-bit TurboQuant으로 압축하여 SQLite에 저장하여야 한다.
- **REQ-01-02**: When 저장이 완료되면, 시스템은 생성된 메모리 ID(`mem_XXXXXX`)를 반환하여야 한다.
- **REQ-01-03**: When 동일 ID가 이미 존재하면, 시스템은 `INSERT OR REPLACE`로 덮어쓰기하여야 한다.
- **REQ-01-04**: 시스템은 새로운 메모리 추가 시 기존에 사용된 적이 없는 고유한 ID를 생성하여야 한다 (삭제된 항목의 ID 재사용으로 인한 충돌 방지).
- **REQ-01-05**: 시스템은 임베딩 캐시의 크기를 제한(LRU 방식)하여 메모리 누수를 방지해야 한다.

### F-02 의미 기반 유사도 검색

- **REQ-02-01**: When MCP 클라이언트가 `recall(query)` 를 호출하면, 시스템은 query를 임베딩하고 압축된 표현에서 직접 inner product를 추정하여 상위 `top_k`개 결과를 반환하여야 한다. 이때 검색 루프 내 연산을 최소화하여 $O(N \cdot d)$의 복잡도를 유지하여야 한다.
- **REQ-02-02**: When `top_k` 파라미터가 생략되면, 시스템은 기본값 5를 사용하여야 한다.
- **REQ-02-03**: When `embedding` 파라미터가 제공되면, 시스템은 텍스트 인코딩 없이 해당 벡터를 직접 사용하여야 한다.
- **REQ-02-04**: When 저장된 메모리가 없으면, 시스템은 빈 배열을 반환하여야 한다.
- **REQ-02-05**: 시스템은 역양자화 없이 압축 표현에서 직접 inner product를 추정하여야 한다.
- **REQ-02-06**: 시스템은 정확한 단어 매칭을 위해 SQLite FTS5 키워드 검색을 수행하고, 벡터 유사도와 결합된 하이브리드 검색 결과를 반환해야 한다.
- **REQ-02-07**: 시스템은 키워드 검색 시 검색어에서 불용어를 제거하여 검색 품질을 높여야 한다.

### F-03 메모리 삭제

- **REQ-03-01**: When MCP 클라이언트가 `forget(id)` 를 호출하면, 시스템은 해당 ID의 메모리를 삭제하고 `{"deleted": true/false}` 를 반환하여야 한다.
- **REQ-03-02**: When 존재하지 않는 ID가 요청되면, 시스템은 `{"deleted": false}` 를 반환하여야 한다.

### F-04 압축 통계 조회

- **REQ-04-01**: When MCP 클라이언트가 `memory_stats()` 를 호출하면, 시스템은 총 항목 수, 벡터 차원, 비트 수, 압축률을 반환하여야 한다.
- **REQ-04-02**: 압축률은 `원본 비트 수 / 압축 비트 수` 로 계산하여야 한다.

### F-05 배치 저장

- **REQ-05-01**: When MCP 클라이언트가 `remember(texts=[...])` 를 호출하면, 시스템은 단일 인코딩 패스(batch_size=64)로 모든 텍스트를 처리하여야 한다.
- **REQ-05-02**: When 배치 저장이 완료되면, 시스템은 저장된 모든 ID 배열과 저장 개수를 반환하여야 한다.

### F-06 MCP stdio 트랜스포트

- **REQ-06-01**: When `--http` 플래그 없이 실행되면, 시스템은 stdin/stdout JSON-RPC 모드로 동작하여야 한다.
- **REQ-06-02**: When `initialize` 메서드가 수신되면, 시스템은 프로토콜 버전 `2024-11-05` 와 tool 목록을 반환하여야 한다.
- **REQ-06-03**: When `notifications/initialized` 가 수신되면, 시스템은 응답 없이 무시하여야 한다.

### F-07 MCP HTTP 트랜스포트

- **REQ-07-01**: When `--http [port]` 플래그로 실행되면, 시스템은 `127.0.0.1:{port}` 에서 HTTP POST JSON-RPC 서버를 시작하여야 한다.
- **REQ-07-02**: When `port` 가 생략되면, 시스템은 기본 포트 8765를 사용하여야 한다.

### F-08 다중 클라이언트 동시 접근

- **REQ-08-01**: 시스템은 SQLite WAL 모드를 통해 다수의 MCP 클라이언트가 동시에 읽기/쓰기를 수행할 수 있어야 한다.
- **REQ-08-02**: 시스템은 `check_same_thread=False` 로 멀티스레드 접근을 허용하여야 한다.

### F-09 로컬 임베딩 모델

- **REQ-09-01**: 시스템은 `sentence-transformers/all-MiniLM-L6-v2` 모델을 로컬에서 실행하여야 한다.
- **REQ-09-02**: 시스템은 외부 API 키 없이 동작하여야 한다.
- **REQ-09-03**: 시스템은 임베딩 생성 시 `normalize_embeddings=True` 를 적용하여 unit sphere를 보장하여야 한다.

### F-10 uvx 원클릭 설치

- **REQ-10-01**: When 사용자가 `uvx --from git+https://...` 를 실행하면, 시스템은 별도 venv 없이 격리된 환경에서 실행되어야 한다.
- **REQ-10-02**: `pyproject.toml` 의 `[project.scripts]` 에 `turbo-memory-mcp = "server:main"` 엔트리포인트가 정의되어야 한다.
- **REQ-10-03**: 시스템은 대량의 데이터를 한 번에 저장할 수 있는 CLI 도구(`ingest.py`)를 제공하여야 한다.

---

## 4. Unwanted Behaviour Requirements

- **REQ-X-01**: 시스템은 norm이 0인 벡터를 압축할 때 zero-division 없이 처리하여야 한다.
- **REQ-X-02**: 시스템은 알 수 없는 JSON-RPC 메서드 수신 시 `-32601 Method not found` 에러를 반환하여야 한다.
- **REQ-X-03**: 시스템은 tool handler 예외 발생 시 `isError: true` 와 에러 메시지를 반환하여야 한다.
- **REQ-X-04**: 시스템은 잘못된 JSON 수신 시 해당 요청을 무시하고 계속 동작하여야 한다.
