# Tasks — turbo-memory-mcp

> Kiro PSDD 기반 구현 태스크 목록  
> 작성일: 2026-04-17 | 상태: 완료

---

## Phase 1. 코어 알고리즘

- [x] **TASK-01** `turbo_quant.py` — TurboQuantState 데이터클래스 정의
  - `dim`, `bits`, `rotation`, `centroids`, `qjl_matrix` 필드
  - REQ-U-01, REQ-09-03

- [x] **TASK-02** `turbo_quant.py` — `build_state(dim, bits, seed)` 구현
  - `np.linalg.qr` 직교 회전 행렬 생성
  - Lloyd-Max 코드북 로드 + `1/√dim` 스케일링
  - QJL 행렬 생성

- [x] **TASK-03** `turbo_quant.py` — `compress(x, state)` 구현
  - Stage 1: 랜덤 회전 + nearest centroid 양자화
  - Stage 2: 잔차 계산 + QJL sign 압축
  - norm==0, r_norm<1e-12 엣지케이스 처리
  - REQ-X-01

- [x] **TASK-04** `turbo_quant.py` — `estimate_inner_product(query, state, ...)` 구현
  - Stage 1 코드북 inner product
  - Stage 2 QJL 보정 (`√(π/2)/d · r_norm · <qjl, sign(S·q)>`)
  - REQ-02-05

---

## Phase 2. 저장소

- [x] **TASK-05** `memory_store.py` — SQLite 스키마 초기화
  - `entries` 테이블 (id, text, idx, norm, qjl, r_norm)
  - `PRAGMA journal_mode=WAL`
  - `check_same_thread=False`
  - REQ-U-03, REQ-08-01, REQ-08-02

- [x] **TASK-06** `memory_store.py` — `MemoryStore.add(text, embedding)` 구현
  - ID 생성 (`mem_{n:06d}`)
  - idx → int8 bytes, qjl → int8 bytes 직렬화
  - `INSERT OR REPLACE`
  - REQ-01-01, REQ-01-02, REQ-01-03

- [x] **TASK-07** `memory_store.py` — `MemoryStore.search(query, top_k)` 구현
  - 전체 스캔 + blob 역직렬화
  - `estimate_inner_product` 호출
  - 내림차순 정렬 + top_k 슬라이싱
  - REQ-02-01, REQ-02-04

- [x] **TASK-08** `memory_store.py` — `MemoryStore.delete(id)` 구현
  - `DELETE WHERE id=?` + `rowcount` 반환
  - REQ-03-01, REQ-03-02

- [x] **TASK-09** `memory_store.py` — `MemoryStore.stats()` 구현
  - 압축률 계산: `original_bits / compressed_bits`
  - REQ-04-01, REQ-04-02

---

## Phase 3. MCP 서버

- [x] **TASK-10** `server.py` — 임베딩 모듈 초기화
  - `SentenceTransformer('all-MiniLM-L6-v2')` 로드
  - `encode()` 단건 + 캐시
  - `encode_batch()` 배치 (batch_size=64)
  - REQ-09-01, REQ-09-02, REQ-09-03

- [x] **TASK-11** `server.py` — `handle_remember` 구현
  - `texts` / `text` 파라미터 분기
  - 배치 인코딩 → 압축 → 저장
  - REQ-01-01, REQ-05-01, REQ-05-02

- [x] **TASK-12** `server.py` — `handle_recall` 구현
  - `embedding` 직접 사용 분기
  - `top_k` 기본값 5
  - REQ-02-01, REQ-02-02, REQ-02-03

- [x] **TASK-13** `server.py` — `handle_forget` / `handle_memory_stats` 구현
  - REQ-03-01, REQ-04-01

- [x] **TASK-14** `server.py` — `dispatch()` JSON-RPC 라우터 구현
  - `initialize` / `tools/list` / `tools/call` / `notifications/initialized` 처리
  - 알 수 없는 메서드 → `-32601` 에러
  - handler 예외 → `isError: true`
  - REQ-U-04, REQ-X-02, REQ-X-03, REQ-X-04

- [x] **TASK-15** `server.py` — `run_stdio()` 구현
  - stdin 라인 루프, JSON 파싱 실패 시 continue
  - REQ-06-01, REQ-06-02, REQ-06-03

- [x] **TASK-16** `server.py` — `run_http(port)` 구현
  - `HTTPServer(127.0.0.1, port)` 바인딩
  - 기본 포트 8765
  - REQ-07-01, REQ-07-02

- [x] **TASK-17** `server.py` — `main()` 진입점 구현
  - `--http` 플래그 분기

---

## Phase 4. 패키징 및 배포

- [x] **TASK-18** `pyproject.toml` — 패키지 정의
  - `requires-python = ">=3.10"`
  - `dependencies = ["sentence-transformers", "numpy"]`
  - `[project.scripts] turbo-memory-mcp = "server:main"`
  - REQ-10-01, REQ-10-02

- [x] **TASK-19** MCP 클라이언트 등록
  - `~/.kiro/settings/mcp.json`
  - `~/.gemini/settings.json`
  - `~/.gemini/antigravity/mcp_config.json`
  - `Makefile` `register` 타겟

---

## Phase 5. 문서화 (Kiro PSDD)

- [x] **TASK-20** `docs/requirements.md` — EARS 형식 요구사항 명세
- [x] **TASK-21** `docs/biz-process.md` — BPMN 기법 프로세스 드릴다운
- [x] **TASK-22** `docs/uis.md` — MCP Tool 인터페이스 화면설계
- [x] **TASK-23** `docs/design.md` — 기술 설계 (아키텍처, 알고리즘, 배포)
- [x] **TASK-24** `docs/tasks.md` — 구현 태스크 목록 (현재 문서)

---

## Phase 6. 버그 수정 및 유틸리티

- [x] **TASK-25** `memory_store.py` — ID 생성 로직 수정 (충돌 방지)
  - `MAX(id)` 기반 또는 `AUTOINCREMENT` 활용
  - REQ-01-04

- [x] **TASK-26** `ingest.py` — SQLite 및 최신 API 대응 리팩토링
  - `CompressedMemoryStore` 삭제 및 `MemoryStore` 연동
  - REQ-10-03, TASK-B04

---

## Phase 7. 성능 최적화 (Elon's Algorithm)

- [x] **TASK-27** `server.py` — 지연 로딩(Lazy Loading) 구현
  - `_encoder` 전역 변수 초기화 제거
  - `get_encoder()` 유틸리티 도입하여 첫 호출 시 로딩
  - REQ-U-05

- [x] **TASK-28** `server.py` — 임베딩 캐시 LRU 적용
  - `functools.lru_cache` 또는 전용 클래스 활용
  - REQ-01-05

- [x] **TASK-29** `turbo_quant.py` — 검색 프로젝션 최적화
  - `prepare_query(query, state)` 구현 (쿼리 투영 사전 계산)
  - `estimate_inner_product` 인터페이스 변경 (사전 투영 값 수신)
  - REQ-02-01

- [x] **TASK-30** `memory_store.py` — 검색 루프 최적화
  - `search()` 내부에서 `prepare_query()` 1회 호출
  - 루프 내 $O(d^2)$ 연산 제거 ($O(N \cdot d)$ 달성)
  - REQ-02-01

- [x] **TASK-31** `benchmark.py` — 성능 및 정확도 측정 도구
  - 시작 시간(Initialization Latency) 측정
  - 검색 속도 및 Recall 측정
  - TASK-B04

- [x] **TASK-32** `server.py` — 병렬 실행 지원
  - `ThreadPoolExecutor` 및 `ThreadingHTTPServer` 적용
  - `RLock` 및 `safe_print`를 통한 동시성 안정화
  - REQ-U-06 (Concurrency)

---

## Phase 8. 하이브리드 검색 (Vector + Keyword)

- [ ] **TASK-33** `memory_store.py` — FTS5 스키마 초기화
  - `entries_fts` 가상 테이블 및 `entries` 동기화 트리거 작성
  - REQ-02-06

- [ ] **TASK-34** `memory_store.py` — 하이브리드 검색 로직 구현
  - FTS `MATCH` 점수와 벡터 유사도 점수 결합 (RRF 또는 가중치 합)
  - 정확한 키워드 매칭 시 우선순위 부여
  - REQ-02-06

---

## Phase 9. 불용어 처리 (Stopword Filtering)

- [x] **TASK-35** `memory_store.py` — 불용어 사전 구축
  - 한글/영어 필수 불용어 리스트 정의
  - REQ-02-07

- [x] **TASK-36** `memory_store.py` — 텍스트 전처리 로직 강화
  - 검색 쿼리에서 불용어 제거 및 토큰화 개선
  - REQ-02-07

---

## 향후 태스크 (Backlog)

- [ ] **TASK-B01** Outlier 채널 별도 양자화 (논문 2.5-bit 전략)
- [ ] **TASK-B02** 코드북 포인터 허프만 인코딩 (5% 추가 압축)
- [ ] **TASK-B03** ANN 인덱스 통합 (FAISS) — O(N·d²) → O(log N)
- [ ] **TASK-B05** GPU 가속 (CUDA 커널, 회전/양자화 병렬화)
