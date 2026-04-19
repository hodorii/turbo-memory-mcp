# Technical Design — turbo-memory-mcp

> 기술 설계 문서 (아키텍처, 알고리즘, 데이터 구조, 배포)  
> 작성일: 2026-04-17

---

## 1. 시스템 아키텍처

### 1.1 레이어 구조

```
┌─────────────────────────────────────────┐
│  MCP Client (Kiro, Gemini, Antigravity) │
└──────────────┬──────────────────────────┘
               │ JSON-RPC 2.0 (stdio/HTTP)
┌──────────────▼──────────────────────────┐
│  server.py — MCP Protocol Handler       │
│  - dispatch()                            │
│  - handle_remember/recall/forget/stats   │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼──────┐  ┌▼──────────────┐
│Encoder│  │MemStore │  │TurboQuant Core│
│(Mini  │  │(SQLite) │  │(compress/est) │
│LM-L6) │  │         │  │               │
└───────┘  └─────────┘  └───────────────┘
```

### 1.2 모듈 의존성

```
server.py
  ├─ memory_store.py
  │   └─ turbo_quant.py
  └─ sentence_transformers (외부)
```

### 1.3 데이터 흐름 (저장)

```
text → encode() → embedding (384, float32)
     → compress() → (idx, norm, qjl, r_norm)
     → SQLite → disk (memory.db)
```

### 1.4 데이터 흐름 (검색)

```
query → encode() → q_embedding (384, float32)
      → Parallel Search:
          1. Vector Search:
             - Pre-projection (q_rot, q_qjl)
             - estimate_inner_product() per entry (O(N·d))
          2. Keyword Search:
             - Text Normalization & Stopword Filtering (New!)
             - SQLite FTS5 MATCH query on entries_fts
      → Hybrid Scoring:
          - score = (Vector_Score * 0.7) + (FTS_Score * 0.3)
      → sort by score desc → top_k
```

---

## 2. TurboQuant 알고리즘

### 2.1 이론적 배경

- **목표**: 벡터 x ∈ ℝᵈ를 b-bit로 압축하되 inner product 추정 편향 제거
- **핵심 아이디어**:
  1. 랜덤 회전 Π로 좌표를 Beta 분포로 변환 → Lloyd-Max 양자화 최적
  2. QJL (Quantized Johnson-Lindenstrauss) 1-bit 잔차 보정 → unbiased

### 2.2 압축 파이프라인 (2단계)

#### Stage 1: Lloyd-Max Quantization (b-1 bits)

```
입력: x ∈ ℝ³⁸⁴
1. norm = ‖x‖₂
2. x_normalized = x / norm
3. rotated = Π · x_normalized  (Π: 384×384 직교 행렬)
4. 각 차원별 nearest centroid 탐색:
   idx[i] = argmin_j |rotated[i] - centroids[j]|
   (centroids: 사전 계산된 Lloyd-Max 코드북, 2-bit → 4개)
5. 출력: idx (384 × 2-bit = 768 bits), norm (32-bit float)
```

#### Stage 2: QJL Residual Correction (1 bit)

```
1. x̂ = Πᵀ · centroids[idx] · norm  (Stage 1 역양자화)
2. r = x - x̂  (잔차)
3. r_norm = ‖r‖₂
4. qjl = sign(S · r)  (S: 384×384 Gaussian 행렬)
   → 384 × 1-bit = 384 bits
5. 출력: qjl (384-bit), r_norm (32-bit float)
```

**총 압축 크기**: 768 + 384 + 64 = 1216 bits (vs FP32 12288 bits → 10.1x 압축)

### 2.3 Inner Product 추정 (검색 시)

```python
def estimate_inner_product(state, q_rot, q_qjl, idx, norm, qjl, r_norm):
    # Stage 1: 코드북 inner product (사전 투영된 쿼리 사용)
    # centroids[idx]와 q_rot의 dot product는 O(d)
    stage1 = np.dot(state.centroids[idx], q_rot) * norm
    
    # Stage 2: QJL 보정
    # qjl과 사전 투영된 q_qjl의 dot product는 O(d)
    stage2 = (np.sqrt(np.pi / 2) / state.dim) * r_norm * np.dot(qjl, q_qjl)
    
    return stage1 + stage2
```

**최적화 결과**: 검색 루프 내 복잡도가 O(d²)에서 O(d)로 감소.

### 2.4 Lloyd-Max 코드북 (사전 계산)

Beta 분포 (고차원 구 표면 좌표)에 최적화된 centroid:

```python
_LLOYD_MAX_CENTROIDS = {
    1: [-0.7979, 0.7979],  # 1-bit (2개)
    2: [-1.5104, -0.4528, 0.4528, 1.5104],  # 2-bit (4개) ← 현재 사용
    3: [...],  # 3-bit (8개)
    4: [...]   # 4-bit (16개)
}
# 실제 사용 시 1/√dim 스케일링
```

---

## 3. 데이터 구조

### 3.1 SQLite 스키마

```sql
-- 기본 데이터 및 압축 벡터 저장
CREATE TABLE entries (
    id     TEXT PRIMARY KEY,
    text   TEXT NOT NULL,
    idx    BLOB NOT NULL,
    norm   REAL NOT NULL,
    qjl    BLOB NOT NULL,
    r_norm REAL NOT NULL
);

-- 고속 키워드 검색용 가상 테이블 (FTS5)
CREATE VIRTUAL TABLE entries_fts USING fts5(
    id UNINDEXED,
    text,
    content='entries',
    content_rowid='rowid'
);
```

### 3.2 TurboQuantState (메모리 상주)

```python
@dataclass
class TurboQuantState:
    dim: int                # 384
    bits: int               # 3 (총 비트, Stage1=2, Stage2=1)
    rotation: np.ndarray    # (384, 384) 직교 행렬 Π
    centroids: np.ndarray   # (4,) Lloyd-Max 코드북 (2-bit)
    qjl_matrix: np.ndarray  # (384, 384) Gaussian 행렬 S
```

**초기화**: `build_state(dim=384, bits=3, seed=42)`
- `seed=42` 고정 → 재현 가능한 회전/QJL 행렬
- 모든 클라이언트가 동일한 state 공유 필수

### 3.3 인코딩 캐시 및 모델 로딩

```python
_encoder: Optional[SentenceTransformer] = None
_encode_cache: LRUCache[str, np.ndarray]  # functools.lru_cache 사용
```

- **Lazy Loading**: `server.py` 시작 시 모델을 로드하지 않고, `encode()` 또는 `encode_batch()`가 처음 호출될 때 로딩하여 `initialize` 지연 해결.
- **Cache**: 무제한 증가를 방지하기 위해 LRU 정책 적용.

---

## 4. 핵심 알고리즘 구현

### 4.1 compress() — 압축

```python
def compress(x: np.ndarray, state: TurboQuantState) -> Tuple[np.ndarray, float, np.ndarray, float]:
    # Stage 1
    norm = float(np.linalg.norm(x))
    if norm == 0:
        return np.zeros(state.dim, dtype=np.int32), 0.0, np.ones(state.dim), 0.0
    
    rotated = state.rotation @ (x / norm)
    idx = np.argmin(np.abs(rotated[:, None] - state.centroids[None, :]), axis=1).astype(np.int32)
    
    # Stage 2
    residual = x - (state.rotation.T @ state.centroids[idx]) * norm
    r_norm = float(np.linalg.norm(residual))
    qjl = np.sign(state.qjl_matrix @ residual) if r_norm > 1e-12 else np.ones(state.dim)
    qjl[qjl == 0] = 1.0
    
    return idx, norm, qjl, r_norm
```

**시간 복잡도**: O(d²) — 행렬 곱셈 지배

### 4.2 estimate_inner_product() — 검색

```python
def estimate_inner_product(query: np.ndarray, state: TurboQuantState,
                           idx: np.ndarray, norm: float,
                           qjl: np.ndarray, r_norm: float) -> float:
    stage1 = float(np.dot(state.centroids[idx], state.rotation @ query)) * norm
    stage2 = (np.sqrt(np.pi / 2) / state.dim) * r_norm \
             * float(np.dot(qjl, np.sign(state.qjl_matrix @ query)))
    return stage1 + stage2
```

**시간 복잡도**: O(d²) — 역양자화 없이 직접 추정

### 4.3 search() — 전체 스캔

```python
def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, float]]:
    rows = self._db.execute("SELECT id, text, idx, norm, qjl, r_norm FROM entries").fetchall()
    if not rows:
        return []
    
    scored = [
        (row_id, text, estimate_inner_product(
            query, self.state,
            np.frombuffer(idx_blob, dtype=np.int8).astype(np.int32), norm,
            np.frombuffer(qjl_blob, dtype=np.int8).astype(np.float32), r_norm
        ))
        for row_id, text, idx_blob, norm, qjl_blob, r_norm in rows
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]
```

**시간 복잡도**: O(N·d²) — N개 항목 전체 스캔

---

## 5. 성능 특성

### 5.1 압축률

| bits | 압축 크기 (per vector) | 압축률 (vs FP32) | MSE 왜곡 |
|------|----------------------|----------------|---------|
| 2    | 832 bits             | ~14.8x         | 0.117   |
| 3    | 1216 bits            | ~10.1x         | 0.030   |
| 4    | 1728 bits            | ~7.1x          | 0.009   |

현재 구현: **3-bit (10.1x 압축)**

### 5.2 정확도

- **Inner product bias**: < 0.007 (n=100 trials, 논문 실험)
- **LongBench score**: 50.06 (FP16와 동일, 3-bit 기준)
- **이론적 보장**: MSE ≤ (√3·π/2) · (1/4^b) ≈ 2.72 · (1/4^b)

### 5.3 시간 복잡도

| 연산 | 복잡도 | 병목 |
|------|--------|------|
| 임베딩 생성 | O(모델 추론) | Transformer forward pass |
| 압축 | O(d²) | 회전 행렬 곱셈 |
| 검색 (N개) | O(N·d²) | 전체 스캔 + inner product 추정 |

**최적화 여지**:
- 회전 행렬 sparse 근사 → O(d log d)
- FAISS 등 ANN 인덱스 (압축 벡터 직접 인덱싱)

---

## 6. 배포 아키텍처

### 6.1 패키지 구조

```
turbo-memory-mcp/
├── pyproject.toml       # uvx 진입점 정의
├── server.py            # MCP 서버 (main 함수)
├── memory_store.py      # 압축 저장소
├── turbo_quant.py       # TurboQuant 코어
└── memory.db            # SQLite 데이터베이스 (런타임 생성)
```

### 6.2 설치 방식

#### uvx (권장)

```bash
uvx --from git+https://github.com/hodorii/turbo-memory-mcp turbo-memory-mcp
```

- 자동 venv 격리
- 의존성 자동 설치: `sentence-transformers`, `numpy`

#### 로컬 개발

```bash
git clone https://github.com/hodorii/turbo-memory-mcp
cd turbo-memory-mcp
make install   # pip install -e .
make register  # MCP 클라이언트 설정 자동 등록
```

### 6.3 MCP 클라이언트 등록

#### stdio 모드 (단일 클라이언트)

```json
// ~/.kiro/settings/mcp.json
{
  "mcpServers": {
    "memory": {
      "command": "uvx",
      "args": ["--from", "/path/to/turbo-memory-mcp", "turbo-memory-mcp"]
    }
  }
}
```

#### HTTP 모드 (다중 클라이언트 / sub-agent)

```bash
# 터미널 1: 서버 시작
make serve  # 기본 포트 8765

# 터미널 2~N: 클라이언트 설정
{
  "mcpServers": {
    "memory": {
      "type": "http",
      "url": "http://127.0.0.1:8765"
    }
  }
}
```

**HTTP 모드 장점**:
- 여러 MCP 클라이언트가 동일 메모리 공유
- sub-agent 병렬 실행 시 메모리 동기화

---

## 7. 보안 및 제약사항

### 7.1 보안

- **로컬 실행**: 외부 API 호출 없음, 네트워크 격리 가능
- **HTTP 모드**: `127.0.0.1` 바인딩 (외부 접근 차단)
- **SQLite WAL**: 파일 잠금으로 동시 쓰기 충돌 방지

### 7.2 제약사항

| 항목 | 제약 | 이유 |
|------|------|------|
| 벡터 차원 | 384 고정 | all-MiniLM-L6-v2 모델 출력 |
| 양자화 비트 | 3-bit 고정 | 정확도/압축률 균형점 |
| 검색 방식 | 전체 스캔 | ANN 인덱스 미구현 |
| seed | 42 고정 | 재현성 보장 (변경 시 기존 DB 무효화) |

### 7.3 확장 가능성

- **다른 임베딩 모델**: `DIM` 변경 + state 재빌드
- **다른 비트 수**: `BITS` 변경 + 코드북 선택
- **ANN 인덱스**: FAISS/ScaNN 통합 (압축 벡터 직접 인덱싱)
- **GPU 가속**: CUDA 커널로 회전/양자화 병렬화

---

## 8. 참고 자료

- [TurboQuant paper (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)
- [QJL paper (arXiv:2406.03482)](https://arxiv.org/abs/2406.03482)
- [Vadim's blog: TurboQuant 3-bit KV cache](https://vadim.blog/turboquant-3-bit-kv-cache-zero-loss)
- [MCP Protocol Spec (2024-11-05)](https://spec.modelcontextprotocol.io/)
