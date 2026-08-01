# TurboQuant: 고성능 벡터 양자화 엔진

고성능 벡터 압축 및 LUT 기반 고속 검색 엔진.

## 양자화기

| 버전 | 파일 | 방식 | 잔차 | 스케일링 | 저장 |
|------|------|------|------|----------|------|
| V1 | `quantizer.py` | 3-bit scalar + Gaussian 코드북 | 없음 | S=1 (고정) | uint8 |
| V2 | `memory.py` | 3-bit scalar + Gaussian 코드북 | sign + std | S=1 (고정) | uint8 + int8 |
| **V3** | **`eden.py`** | **EDEN (최적 S) + Beta Lloyd-Max 코드북** | **DRIVE (l1/d)** | **최적 S_bias/S_unbias** | **bit-packed** |

## 핵심 결과 (V3 EDEN vs V2)

| 지표 | 개선 |
|------|------|
| MSE (d=1024, 3-bit) | **-96.7%** |
| Recall@5 (d=128, N=1000) | **+59%p** |
| 저장공간 (d=1024) | **-75%** (bit-packed) |
| 검색 속도 | **거의 동일** (<0.3% 차이) |

## 빠른 시작

```bash
# EDEN V3 양자화기
PYTHONPATH=. python -c "
from src.turboquant.eden import EdenConfig, EdenQuantizer
eq = EdenQuantizer(EdenConfig(dim=1024, bits=3, mode='biased'))
idx, rs, rsc, S = eq.quantize(torch.randn(1024))
"

# EDEN이 적용된 TurboDiskStore
from src.turboquant.memory import TurboDiskStore
store = TurboDiskStore(1024, 3, './storage', quantizer=eq)
store.add(torch.randn(1024))
store.search(torch.randn(1024), top_k=5)
```

## 테스트 실행

```bash
PYTHONPATH=. python tests/test_eden.py    # 15개 EDEN V3 테스트
PYTHONPATH=. python tests/test_quantizer.py  # V1 회귀 테스트
```

## 프로젝트 구조

```
src/turboquant/
  quantizer.py    V1 TurboQuantizer
  eden.py         V3 EdenQuantizer + bit-packing 유틸리티
  memory.py       V2/V3 TurboDiskStore (memmap + LUT 검색)
engine/
  memory_engine.py  FAISS + SQLite 하이브리드
tests/
  test_eden.py    EDEN V3 종합 테스트 스위트
```

## 의존성

torch, numpy, faiss, sentence-transformers (BGE-M3)
