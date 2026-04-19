# Business Process — turbo-memory-mcp

> BPMN 기법 기반 개조식 계층화  
> Process → Task → FunctionGroup/UI → Step → DetailStep → Logic  
> 작성일: 2026-04-17

---

## 프로세스 맵 (Overview)

```
[MCP Client]
    │
    ├─ P-01. 메모리 저장 프로세스
    ├─ P-02. 메모리 검색 프로세스
    ├─ P-03. 메모리 삭제 프로세스
    ├─ P-04. 통계 조회 프로세스
    └─ P-05. 서버 초기화 프로세스
```

---

## P-01. 메모리 저장 프로세스

> Trigger: `remember(text)` 또는 `remember(texts=[...])`

### T-01-1. 입력 파싱

- **FG: InputParser** (`server.py::handle_remember`)
  - Step 1. 파라미터 수신
    - DetailStep 1-1. `params.get("texts")` 우선 확인
    - DetailStep 1-2. 없으면 `params.get("text")` 단건 → 리스트 변환
    - DetailStep 1-3. 둘 다 없으면 에러 반환
    - Logic:
      ```python
      texts = params.get("texts") or (
          [params["text"].strip()] if params.get("text") else None
      )
      if not texts: return error("text or texts is required")
      ```

### T-01-2. 임베딩 생성

- **FG: Encoder** (`server.py::encode_batch`)
  - Step 1. 배치 인코딩
    - DetailStep 1-1. `all-MiniLM-L6-v2` 모델 로드 (최초 1회)
    - DetailStep 1-2. `encode(texts, normalize_embeddings=True, batch_size=64)`
    - DetailStep 1-3. `float32` 변환 → shape `(N, 384)`

### T-01-3. TurboQuant 압축

- **FG: Compressor** (`turbo_quant.py::compress`)
  - Step 1. Stage 1 — 랜덤 회전 + Lloyd-Max 양자화
    - DetailStep 1-1. `norm = ‖x‖₂`; norm==0 이면 zero 반환 (guard)
    - DetailStep 1-2. `rotated = Π · (x / norm)` — 직교 회전
    - DetailStep 1-3. nearest centroid 탐색 → `idx` (int32, dim개)
    - Logic:
      ```python
      rotated = state.rotation @ (x / norm)
      idx = np.argmin(np.abs(rotated[:,None] - state.centroids[None,:]), axis=1)
      ```
  - Step 2. Stage 2 — QJL 잔차 보정
    - DetailStep 2-1. `x̂ = Πᵀ · centroids[idx] · norm` 역양자화
    - DetailStep 2-2. `r = x - x̂`; `r_norm = ‖r‖₂`
    - DetailStep 2-3. `qjl = sign(S · r)` (r_norm < 1e-12 이면 ones)
    - Logic:
      ```python
      residual = x - (state.rotation.T @ state.centroids[idx]) * norm
      qjl = np.sign(state.qjl_matrix @ residual) if r_norm > 1e-12 else np.ones(dim)
      ```

### T-01-4. SQLite 저장

- **FG: Store** (`memory_store.py::MemoryStore.add`)
  - Step 1. ID 생성: `SELECT COUNT(*) FROM entries` → `mem_{n:06d}`
  - Step 2. 직렬화: `idx` → int8 bytes, `qjl` → int8 bytes
  - Step 3. 삽입
    - Logic:
      ```python
      self._db.execute(
          "INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?)",
          (entry_id, text, idx.astype(np.int8).tobytes(), norm,
           np.sign(qjl).astype(np.int8).tobytes(), r_norm)
      )
      self._db.commit()
      ```

### T-01-5. 응답 반환

- `{"ids": [...], "stored": N}` → JSON-RPC result envelope

---

## P-02. 메모리 검색 프로세스

> Trigger: `recall(query, top_k?, embedding?)`

### T-02-1. 입력 파싱

- **FG: InputParser** (`server.py::handle_recall`)
  - Step 1. query 검증: 빈 문자열이면 에러
  - Step 2. 임베딩 소스 결정
    - DetailStep 2-1. `embedding` 파라미터 있으면 → T-02-2a (직접 사용)
    - DetailStep 2-2. 없으면 → T-02-2b (텍스트 인코딩)

### T-02-2a. 사전 임베딩 사용 (선택적)

- **FG: EmbeddingLoader**
  - Step 1. `q = np.array(raw, float32)`; `n > 0` 이면 `q /= n` 정규화

### T-02-2b. 쿼리 임베딩 생성

- **FG: Encoder** (`server.py::encode`)
  - Step 1. `_encode_cache` 조회 → 캐시 히트 즉시 반환
  - Step 2. 캐시 미스 → 모델 인코딩 후 캐시 저장

### T-02-3. 전체 스캔 + 점수 계산

- **FG: Searcher** (`memory_store.py::MemoryStore.search`)
  - Step 1. 쿼리 사전 투영 (Pre-projection)
    - DetailStep 1-1. `q_rot = state.rotation @ query`
    - DetailStep 1-2. `q_qjl = np.sign(state.qjl_matrix @ query)`
  - Step 2. `SELECT id, text, idx, norm, qjl, r_norm FROM entries` 전체 로드
  - Step 3. 항목별 inner product 추정
    - DetailStep 3-1. blob → int8 → int32/float32 역직렬화
    - DetailStep 3-2. `estimate_inner_product(state, q_rot, q_qjl, idx, norm, qjl, r_norm)`
    - Logic:
      ```python
      stage1 = np.dot(state.centroids[idx], q_rot) * norm
      stage2 = (np.sqrt(np.pi/2) / state.dim) * r_norm * np.dot(qjl, q_qjl)
      score = stage1 + stage2
      ```
  - Step 4. 내림차순 정렬 → `[:top_k]` 슬라이싱

### T-02-4. 응답 반환

- `[{"id":..., "text":..., "score": round(s,4)}]` → JSON-RPC result envelope

---

## P-03. 메모리 삭제 프로세스

> Trigger: `forget(id)`

### T-03-1. 입력 검증

- **FG: InputParser** (`server.py::handle_forget`)
  - Step 1. `id` 빈 문자열이면 에러 반환

### T-03-2. DB 삭제

- **FG: Store** (`memory_store.py::MemoryStore.delete`)
  - Step 1. `DELETE FROM entries WHERE id=?`
  - Step 2. `rowcount > 0` → 삭제 성공 여부 판단
  - Logic:
    ```python
    cur = self._db.execute("DELETE FROM entries WHERE id=?", (entry_id,))
    self._db.commit()
    return cur.rowcount > 0
    ```

### T-03-3. 응답 반환

- `{"deleted": true/false}`

---

## P-04. 통계 조회 프로세스

> Trigger: `memory_stats()`

### T-04-1. 통계 계산

- **FG: StatsCalculator** (`memory_store.py::MemoryStore.stats`)
  - Step 1. `SELECT COUNT(*) FROM entries` → n
  - Step 2. 압축률 계산
    - DetailStep 2-1. `compressed_bits = n * ((bits-1)*dim + dim + 64)`
      - `(bits-1)*dim`: Stage1 코드북 인덱스 비트
      - `dim`: Stage2 QJL 1-bit
      - `64`: norm + r_norm (각 32-bit float)
    - DetailStep 2-2. `original_bits = n * dim * 32` (FP32 기준)
    - DetailStep 2-3. `ratio = original_bits / max(compressed_bits, 1)`

### T-04-2. 응답 반환

- `{"entries": N, "dim": 384, "bits": 3, "compression_ratio": X.XX}`

---

## P-05. 서버 초기화 프로세스

> Trigger: `server:main` 진입점 실행

### T-05-1. 트랜스포트 결정

- **FG: Launcher** (`server.py::main`)
  - Step 1. `"--http" in sys.argv` 확인
    - DetailStep 1-1. 있으면 → T-05-2b (HTTP 모드)
    - DetailStep 1-2. 없으면 → T-05-2a (stdio 모드)

### T-05-2a. stdio 서버

- **FG: StdioServer** (`server.py::run_stdio`)
  - Step 1. `for line in sys.stdin` 루프
    - DetailStep 1-1. 빈 줄 skip
    - DetailStep 1-2. `json.loads(line)` 실패 시 `continue`
    - DetailStep 1-3. `dispatch(req)` → `resp is not None` 이면 stdout 출력

### T-05-2b. HTTP 서버

- **FG: HttpServer** (`server.py::run_http`)
  - Step 1. 포트 파싱: `sys.argv[idx+1]` 없으면 8765
  - Step 2. `HTTPServer(("127.0.0.1", port), Handler).serve_forever()`
  - Step 3. `POST /` → `dispatch()` → JSON 응답

### T-05-3. 상태 초기화 (모듈 로드 시)

- **FG: StateInitializer**
  - Step 1. TurboQuantState 빌드 (`build_state(dim=384, bits=3, seed=42)`)
    - DetailStep 1-1. `np.linalg.qr(rng.standard_normal((384,384)))` → 직교 회전 행렬 Π
    - DetailStep 1-2. `_LLOYD_MAX_CENTROIDS[2]` 로드 (b-1=2, 4개 centroid)
    - DetailStep 1-3. `rng.standard_normal((384,384))` → QJL 행렬 S
  - Step 2. SQLite 연결
    - DetailStep 2-1. `sqlite3.connect(STORE_PATH, check_same_thread=False)`
    - DetailStep 2-2. `PRAGMA journal_mode=WAL`
    - DetailStep 2-3. `entries` 테이블 CREATE IF NOT EXISTS
  - Step 3. 모델 로딩 유예 (Lazy Loading)
    - DetailStep 3-1. `_encoder`를 `None`으로 설정하여 초기 `initialize` 속도 극대화

---

## 예외 흐름

| 상황 | 발생 위치 | 처리 |
|------|----------|------|
| `text`/`texts` 모두 없음 | handle_remember | `error("text or texts is required")` |
| `query` 빈 문자열 | handle_recall | `error("query is required")` |
| `id` 빈 문자열 | handle_forget | `error("id is required")` |
| 알 수 없는 tool | dispatch | `error("Unknown tool: ...")` |
| handler 예외 | dispatch | `isError: true` + 에러 메시지 |
| 잘못된 JSON | run_stdio | `continue` (무시) |
| norm == 0 벡터 | _quantize_stage1 | zero idx, norm=0.0 반환 |
| r_norm < 1e-12 | compress | `qjl = np.ones(dim)` |
