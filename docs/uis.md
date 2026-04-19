# UI Specification — turbo-memory-mcp

> MCP Tool Interface 화면설계 (CLI/JSON-RPC 기반)  
> "화면" = MCP 클라이언트가 호출하는 Tool 인터페이스 + 응답 포맷  
> 작성일: 2026-04-17

---

## UI 목록

| ID | Tool | 방향 | 연결 프로세스 |
|----|------|------|-------------|
| UI-01 | `remember` | Client → Server | P-01 |
| UI-02 | `recall` | Client → Server | P-02 |
| UI-03 | `forget` | Client → Server | P-03 |
| UI-04 | `memory_stats` | Client → Server | P-04 |
| UI-05 | `tools/list` | Client → Server | P-05 |
| UI-06 | `initialize` | Client → Server | P-05 |

---

## UI-01. remember — 메모리 저장

### 입력 스키마

```json
{
  "name": "remember",
  "arguments": {
    "text":  "<string>  단건 텍스트 (text 또는 texts 중 하나 필수)",
    "texts": ["<string>", "..."]  // 배치 저장 (우선 사용 권장)
  }
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| text | string | 조건부 | 단건 저장 |
| texts | string[] | 조건부 | 배치 저장, 단일 인코딩 패스 |

> `text`와 `texts` 중 하나 이상 필수. `texts` 우선.

### 출력 스키마 (성공)

```json
{
  "ids": ["mem_000000", "mem_000001"],
  "stored": 2
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| ids | string[] | 생성된 메모리 ID 목록 (`mem_XXXXXX` 형식) |
| stored | integer | 저장된 항목 수 |

### 출력 스키마 (에러)

```json
{
  "error": { "message": "text or texts is required" }
}
```

### 호출 예시

```json
// 단건
{ "name": "remember", "arguments": { "text": "TurboQuant은 3-bit로 10x 압축한다" } }

// 배치
{ "name": "remember", "arguments": { "texts": ["메모리 A", "메모리 B", "메모리 C"] } }
```

---

## UI-02. recall — 메모리 검색

### 입력 스키마

```json
{
  "name": "recall",
  "arguments": {
    "query":     "<string>  검색 쿼리 텍스트 (필수)",
    "top_k":     5,          // 반환 개수 (기본값 5)
    "embedding": [0.1, ...]  // 사전 계산된 384차원 벡터 (선택)
  }
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| query | string | 필수 | — | 검색 쿼리 |
| top_k | integer | 선택 | 5 | 반환할 최대 결과 수 |
| embedding | number[] | 선택 | — | dim=384 사전 임베딩 (제공 시 인코딩 생략) |

### 출력 스키마 (성공)

```json
{
  "results": [
    { "id": "mem_000003", "text": "관련 메모리 텍스트", "score": 0.7823 },
    { "id": "mem_000001", "text": "두 번째 결과",       "score": 0.5412 }
  ]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| results | object[] | 유사도 내림차순 정렬 |
| results[].id | string | 메모리 ID |
| results[].text | string | 원본 텍스트 |
| results[].score | number | inner product 추정값 (소수점 4자리) |

### 출력 스키마 (저장 항목 없음)

```json
{ "results": [] }
```

### 호출 예시

```json
{ "name": "recall", "arguments": { "query": "벡터 압축 알고리즘", "top_k": 3 } }
```

---

## UI-03. forget — 메모리 삭제

### 입력 스키마

```json
{
  "name": "forget",
  "arguments": {
    "id": "mem_000002"  // 삭제할 메모리 ID (필수)
  }
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| id | string | 필수 | 삭제 대상 메모리 ID |

### 출력 스키마

```json
{ "deleted": true }   // 삭제 성공
{ "deleted": false }  // ID 없음
```

### 호출 예시

```json
{ "name": "forget", "arguments": { "id": "mem_000002" } }
```

---

## UI-04. memory_stats — 통계 조회

### 입력 스키마

```json
{
  "name": "memory_stats",
  "arguments": {}
}
```

파라미터 없음.

### 출력 스키마

```json
{
  "entries": 42,
  "dim": 384,
  "bits": 3,
  "compression_ratio": 9.14
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| entries | integer | 저장된 메모리 총 수 |
| dim | integer | 임베딩 차원 (384) |
| bits | integer | 양자화 비트 수 (3) |
| compression_ratio | number | FP32 대비 압축률 (예: 9.14x) |

---

## UI-05. tools/list — Tool 목록 조회

### 입력 (JSON-RPC)

```json
{ "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {} }
```

### 출력

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "tools": [
      {
        "name": "remember",
        "description": "Store one or multiple memories with TurboQuant-compressed embeddings (~10x compression).",
        "inputSchema": { "type": "object", "properties": { "text": {...}, "texts": {...} } }
      },
      { "name": "recall",       "description": "...", "inputSchema": {...} },
      { "name": "forget",       "description": "...", "inputSchema": {...} },
      { "name": "memory_stats", "description": "...", "inputSchema": {...} }
    ]
  }
}
```

---

## UI-06. initialize — 핸드셰이크

### 입력 (JSON-RPC)

```json
{
  "jsonrpc": "2.0", "id": 0,
  "method": "initialize",
  "params": { "protocolVersion": "2024-11-05", "clientInfo": {...} }
}
```

### 출력

```json
{
  "jsonrpc": "2.0", "id": 0,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": { "tools": {} },
    "serverInfo": { "name": "memory-mcp", "version": "1.0.0" }
  }
}
```

---

## JSON-RPC Envelope (공통)

모든 tool 호출은 `tools/call` 메서드를 통해 래핑됨:

### 요청 Envelope

```json
{
  "jsonrpc": "2.0",
  "id": "<request_id>",
  "method": "tools/call",
  "params": {
    "name": "<tool_name>",
    "arguments": { ... }
  }
}
```

### 응답 Envelope (성공)

```json
{
  "jsonrpc": "2.0",
  "id": "<request_id>",
  "result": {
    "content": [{ "type": "text", "text": "<JSON string>" }],
    "isError": false
  }
}
```

### 응답 Envelope (에러)

```json
{
  "jsonrpc": "2.0",
  "id": "<request_id>",
  "result": {
    "content": [{ "type": "text", "text": "{\"error\": {\"message\": \"...\"}}" }],
    "isError": true
  }
}
```

---

## 트랜스포트별 접근 방식

### stdio 모드 (기본)

```
stdin  → 한 줄 = 하나의 JSON-RPC 요청
stdout ← 한 줄 = 하나의 JSON-RPC 응답
```

MCP 클라이언트 설정:
```json
"memory": {
  "command": "uvx",
  "args": ["--from", "/path/to/turbo-memory-mcp", "turbo-memory-mcp"]
}
```

### HTTP 모드 (다중 클라이언트)

```
POST http://127.0.0.1:8765
Content-Type: application/json
Body: <JSON-RPC 요청>
```

MCP 클라이언트 설정:
```json
"memory": {
  "type": "http",
  "url": "http://127.0.0.1:8765"
}
```
