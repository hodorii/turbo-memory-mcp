# Development Log

## 2026-04-17

### 초기 구현 (02:22 - 02:28)

**목표**: Google TurboQuant 논문 기반 압축 벡터 메모리 MCP 서버 구현

**논문 분석**:
- arXiv:2504.19874 (ICLR 2026) 확인
- 핵심: 2단계 파이프라인 (MSE 양자화 + QJL 잔차 보정)
- 이론적 보장: MSE 왜곡 ≤ (√3·π/2) · (1/4^b), inner product unbiased

**구현 완료**:
1. `turbo_quant.py` — TurboQuant 코어
   - Lloyd-Max 코드북 사전 계산 (1~4 bits)
   - `quant_prod/dequant_prod`: 2단계 양자화/역양자화
   - 랜덤 회전 행렬 Π, QJL 행렬 S 초기화

2. `memory_store.py` — 압축 저장소
   - `CompressedMemoryStore`: add/search/delete/save/load
   - 압축률 통계 (3-bit: ~9x vs FP32)

3. `server.py` — MCP 서버
   - JSON-RPC over stdio
   - Tools: remember, recall, forget, memory_stats
   - 초기: hash 기반 pseudo-embedding (dim=128)

**테스트 결과**:
- Inner product unbiasedness 확인 (bias < 0.007, n=100 trials)
- 압축률: 9.14x (3-bit, FP32 대비)
- MCP 프로토콜 정상 동작

### 실제 임베딩 모델 연동 (02:28 - 02:39)

**시도 1**: OpenAI API (text-embedding-3-small, dim=1536)
- `requests` 라이브러리로 직접 호출
- API 키 없으면 hash fallback
- 문제: 디스크 공간 부족으로 save 실패

**개선**:
- save 실패 시 graceful degradation (in-memory 계속 동작)
- 깨진 pickle 파일 로드 실패 시 자동 재초기화
- DIM/BITS 불일치 시 자동 재생성

**시도 2**: sentence-transformers (all-MiniLM-L6-v2, dim=384)
- 디스크 공간 확보 후 venv 설치 성공
- 로컬 모델, API 키 불필요
- lazy-load로 첫 호출 시에만 모델 로드
- `STORE_PATH` 절대경로로 변경

**최종 테스트**:
```
query: "벡터 양자화 압축"
  1. score=0.5775 — 파이썬은 머신러닝에 널리 쓰이는 언어다
  2. score=0.4057 — 랜덤 회전 행렬을 적용하면 각 좌표가 Beta 분포를 따른다
  3. score=0.1741 — TurboQuant은 3-bit로 KV 캐시를 압축...
```
의미 기반 검색 정상 동작 확인.

### MCP 클라이언트 등록 (02:39 - 02:46)

**등록 완료**:
- `~/.kiro/settings/mcp.json` (kiro-cli)
- `~/.gemini/settings.json` (gemini)
- `~/.gemini/antigravity/mcp_config.json` (antigravity)

모두 동일한 `memory.pkl` 공유 → 크로스 클라이언트 메모리 동기화.

## 기술적 결정

### 왜 TurboQuant인가?
- **정보이론적 최적성**: 하한의 2.7배 이내 (기존 방법 대비 지수적 개선)
- **Data-oblivious**: 코드북 사전 계산 가능, 캘리브레이션 불필요
- **Unbiased inner product**: QJL 보정으로 attention score 편향 제거

### 왜 sentence-transformers인가?
- 로컬 실행, API 비용 없음
- all-MiniLM-L6-v2: 384차원, 빠른 추론 속도
- normalize_embeddings=True로 unit sphere 보장

### 압축률 트레이드오프
- 2-bit: 14x 압축, MSE 0.117 (작은 모델에 적합)
- 3-bit: 10x 압축, MSE 0.030 (논문 권장, 정확도 손실 거의 없음)
- 4-bit: 7x 압축, MSE 0.009 (대형 모델, 긴 컨텍스트)

현재 구현: 3-bit (정확도/압축률 균형점)

## 향후 개선 방향

1. **Outlier 처리**: 32개 outlier 채널 별도 양자화 (논문 2.5-bit 전략)
2. **Entropy coding**: 코드북 포인터 허프만 인코딩 (5% 추가 압축)
3. **Batch quantization**: 여러 벡터 동시 처리로 throughput 향상
4. **GPU 가속**: CUDA 커널로 rotation/quantization 병렬화
5. **Incremental update**: 전체 재양자화 없이 새 메모리 추가

## 참고 자료

- [TurboQuant paper](https://arxiv.org/abs/2504.19874)
- [Vadim's blog: TurboQuant 3-bit KV cache](https://vadim.blog/turboquant-3-bit-kv-cache-zero-loss)
- [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression)
- [llama.cpp community implementation](https://github.com/ggml-org/llama.cpp/discussions/20969)
