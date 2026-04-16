# turbo-memory-mcp

[English](README.md)

TurboQuant 알고리즘 기반 압축 벡터 메모리 MCP 서버.

Google Research의 [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874)을 구현하여,
임베딩 벡터를 **~10x 압축**하면서 inner product 추정의 정확도를 유지하는 메모리 서버.

## 특징

- **TurboQuant 2단계 파이프라인**: 랜덤 회전 + Lloyd-Max 양자화 (b-1 bits) → QJL 잔차 보정 (1 bit)
- **Unbiased inner product**: QJL 보정으로 유사도 검색 편향 제거
- **로컬 임베딩**: `all-MiniLM-L6-v2` (sentence-transformers), API 키 불필요
- **MCP 표준**: Gemini CLI, Kiro, Antigravity 등 MCP 클라이언트에서 바로 사용

## 압축률

| bits | 압축률 (vs FP32) | MSE 왜곡 |
|------|----------------|---------|
| 2    | ~14x           | 0.117   |
| 3    | ~10x           | 0.030   |
| 4    | ~7x            | 0.009   |

3-bit에서 LongBench 점수 FP16과 동일 (논문 결과: 50.06).

## 파일 구조

```
turbo_quant.py    — TurboQuant 코어 (양자화/역양자화)
memory_store.py   — 압축 벡터 저장소
server.py         — MCP 서버 (JSON-RPC over stdio)
mcp_config.json   — MCP 클라이언트 설정 예시
```

## 설치

```bash
git clone https://github.com/hodorii/turbo-memory-mcp
cd turbo-memory-mcp
python3 -m venv .venv
.venv/bin/pip install sentence-transformers numpy
```

## MCP 등록

`~/.kiro/settings/mcp.json`, `~/.gemini/settings.json`, `~/.gemini/antigravity/mcp_config.json`, `~/.config/opencode/mcp-servers.json` 중 원하는 클라이언트에 추가:

```json
"memory": {
  "command": "/path/to/.venv/bin/python",
  "args": ["/path/to/server.py"]
}
```

## MCP Tools

| Tool | 설명 |
|------|------|
| `remember(text)` | 텍스트를 임베딩 후 압축 저장 |
| `recall(query, top_k?)` | 유사 메모리 검색 |
| `forget(id)` | 메모리 삭제 |
| `memory_stats()` | 압축률 통계 |

## TurboQuant 원리

```
입력 벡터 x
    │
    ▼ Stage 1 (b-1 bits): 랜덤 회전 Π·x → Beta 분포 → Lloyd-Max 스칼라 양자화
    │
    ▼ Stage 2 (1 bit): residual r = x - x̂  →  sign(S·r)  [QJL, unbiased]
```

이론적 보장: MSE 왜곡 ≤ (√3·π/2) · (1/4^b) — 정보이론 하한의 2.7배 이내.

## 참고

- [TurboQuant 논문](https://arxiv.org/abs/2504.19874) — Zandieh et al., Google Research, ICLR 2026
- [QJL 논문](https://arxiv.org/abs/2406.03482) — 1-bit unbiased inner product quantization
