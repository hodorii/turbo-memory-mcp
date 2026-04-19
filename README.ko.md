# turbo-memory-mcp

[English](README.md)

TurboQuant 알고리즘 기반 압축 벡터 메모리 MCP 서버.

Google Research의 [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874)을 구현하여,
임베딩 벡터를 **~10x 압축**하면서 inner product 추정의 정확도를 유지하는 메모리 서버.

## 특징

- **하이브리드 검색**: **벡터 유사도(의미)**와 **SQLite FTS5(키워드)** + **형태소 분석(Kiwi)**을 결합하여 기술 심볼 및 고유 명사에 대한 검색 정확도 극대화.
- **초고속 성능**: 사전 투영(Pre-projection)을 통한 $O(N \cdot d)$ 최적화로 1,000개 항목 검색 시 10ms 미만 소요.
- **즉각적인 초기화**: 지연 로딩(Lazy Loading)을 통해 MCP 서버 시작 시간을 0ms로 단축, 클라이언트 타임아웃 방지.
- **병렬 처리 지원**: `ThreadPoolExecutor`와 `RLock`을 적용하여 여러 서브 에이전트의 동시 접근에 안전함.
- **TurboQuant 2단계 파이프라인**: 랜덤 회전 + Lloyd-Max 양자화 (b-1 bits) → QJL 잔차 보정 (1 bit)
- **Unbiased inner product**: QJL 보정으로 유사도 검색 편향 제거
- **로컬 임베딩**: `all-MiniLM-L6-v2` (sentence-transformers), API 키 불필요
- **MCP 표준**: Gemini CLI, Kiro, Antigravity 등 MCP 클라이언트에서 바로 사용

## 성능 및 정확도 비교

| 지표 | 초기 버전 (Baseline) | 최적화 후 (Current) | 개선 효과 |
|------|-------------------|----------------------|-------------|
| **시작 지연시간** | ~5,200ms | **< 1ms** | 5000배 단축 |
| **검색 속도 (N=1000)** | ~100ms | **~9.4ms** | 10배 향상 |
| **기술 심볼 정확도** | 보통 (벡터 전용) | **극대화** (FTS5 하이브리드) | 식별자 매칭 완벽 |
| **한글 지원** | 기초 (공백 분리) | **고급** (형태소 분석) | 조사 분리 및 정규화 |

## 알고리즘 최적화 및 변경점

본 프로젝트는 **TurboQuant (ICLR 2026)** 논문의 수학적 핵심을 엄격히 준수하면서도, 실제 운영 환경을 위해 두 가지 주요 엔지니어링 최적화를 도입했습니다.

1. **Pre-projected Execution**: 논문은 검색 시 매 레코드마다 $O(d^2)$ 투영을 제안하지만, 본 구현체는 쿼리당 1회의 사전 투영(Pre-projection)을 수행하여 루프 내 복잡도를 $O(N \cdot d)$로 낮췄습니다. 이를 통해 단일 CPU 코어에서 수만 개의 기억을 실시간 검색할 수 있습니다.
2. **Hybrid 80/20 Scoring**: 단순 벡터 유사도에만 의존하지 않고, **Keyword(80%) + Vector(20%)** 가중치 합산을 적용합니다. 이는 `TASK-01`과 같은 특정 기술 식별자가 벡터 공간의 "의미적 모호함"에 묻히지 않도록 보장합니다.

## Memory vs. RAG

이 도구는 에이전트 전용 **장기 기억(Long-term Memory)** 시스템으로, 일반적인 RAG와는 다음과 같은 차이가 있습니다.

- **압축률**: 3-bit TurboQuant를 사용하여 일반 RAG 대비 동일 메모리 공간에 10배 더 많은 기억을 저장합니다.
- **동적 관리**: 정적 문서 중심의 RAG와 달리, 에이전트가 실시간으로 사실을 기록(`remember`)하고 잊는(`forget`) 과정에 최적화되어 있습니다.
- **정밀도**: 형태소 분석 기반의 FTS5를 결합하여, 일반적인 시맨틱 검색이 놓치기 쉬운 기술적 심볼과 ID를 정확하게 추적합니다.

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

## 왜 TurboQuant인가?

본 구현체는 다음과 같은 수학적 이점 때문에 표준 PQ(Product Quantization) 대신 **TurboQuant**를 선택했습니다.
1. **무편향 내적 추정 (Unbiased Inner Product)**: 일반적인 양자화는 유사도 점수에 계통 오차(Bias)를 도입합니다. TurboQuant의 **QJL (Quantized Johnson-Lindenstrauss)** 단계는 이를 보정하여 $E[\text{추정치}] = \text{실제값}$을 보장합니다.
2. **정보이론적 최적성**: MSE 왜곡 측면에서 이론적 하한선의 2.7배 이내를 달성하며, 기존 방식들보다 수십 배 뛰어난 성능을 보입니다.
3. **데이터 독립성 (Data-oblivious)**: 단위 구(Unit-sphere) 상의 Beta 분포에 최적화된 사전 계산된 Lloyd-Max 코드북을 사용하므로, 사용자 데이터에 대한 별도의 학습이나 캘리브레이션이 필요 없습니다.

## 핵심 알고리즘: 2단계 파이프라인

본 프로젝트는 차원당 **3-bit** 설정을 기본으로 사용하며, 이는 FP16 수준의 정확도를 유지하면서 약 10.1배의 압축률을 제공합니다.

1. **Stage 1: Lloyd-Max 양자화 (2-bits)**
   - 랜덤 직교 회전 $\Pi$를 적용하여 벡터를 Near-Gaussian 분포로 매핑합니다.
   - MSE 최적화된 코드북을 사용하여 각 좌표를 양자화합니다.
2. **Stage 2: QJL 잔차 보정 (1-bit)**
   - 양자화 후 남은 잔차(Residual) $r = x - \hat{x}$를 계산합니다.
   - 가우시안 프로젝션 $S \cdot r$의 **부호(Sign)**만 저장합니다.
   - 이 1-bit 보정치가 검색 시 양자화 편향을 제거하는 핵심 열쇠입니다.

## 압축 명세 (3-bit 기본값)

| 구성 요소 | 차원당 비트 수 | 384차원 기준 크기 |
|-----------|--------------|-------------------|
| Stage 1 (인덱스) | 2 bits | 768 bits |
| Stage 2 (QJL) | 1 bit | 384 bits |
| Norms (메타) | - | 64 bits (2x Float32) |
| **합계** | **~3.16 bits** | **1216 bits** (vs 12288 FP32) |

## 참고

- [TurboQuant 논문](https://arxiv.org/abs/2504.19874) — Zandieh et al., Google Research, ICLR 2026
- [QJL 논문](https://arxiv.org/abs/2406.03482) — 1-bit unbiased inner product quantization
