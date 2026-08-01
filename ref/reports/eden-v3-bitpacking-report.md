# EDEN V3 + Bit-Packing 평가 보고서

> 생성일: 2026-05-09
> 태그: eden-v3, bit-packing, turboquant, quantization

## 요약

EDEN V3 양자화기는 TurboQuant의 S=1 고정 스케일링을 ICML 2022 EDEN 알고리즘의 분석적 최적 스케일링 계수로 대체한다. Bit-packing 저장 방식과 결합하여 V2 TurboQuant 대비 **MSE -90~99%**, **recall +26~59%p**, **저장공간 -75%** 를 달성했다.

## 아키텍처

```
quantizer.py  → TurboQuantizer_V1 (3-bit naive, S=1)
memory.py     → TurboQuantizer_V2 (3-bit + 1-bit residual, S=1)
               → TurboDiskStore (memmap + LUT 검색)
eden.py       → EdenQuantizer (EDEN: 최적 S + Beta 코드북 + DRIVE)
               → Bit-packing 유틸리티 (pack/unpack)
```

## 핵심 컴포넌트

### 1. EdenQuantizer (`src/turboquant/eden.py`)
- **Beta 코드북**: 회전 후 분포 Beta(alpha,alpha)에 대한 Lloyd-Max 양자화 레벨 (alpha=(d-1)/2)
- **최적 S 계수**: S_bias = <y, q>/||q||^2 (MSE 최소화), S_unbias = ||x||^2/|<y,q>| (편향 제거)
- **DRIVE 잔차**: 회전 공간에서 1-bit 부호 + l1/d 스케일
- **모드**: biased (고정밀) / unbiased (영평균 오차)

### 2. Bit-Packing (`src/turboquant/eden.py`)
- `pack_bits`: 벡터화된 torch pack (b-bit 인덱스 → uint8 바이트)
- `unpack_bits`: 벡터화된 torch unpack (uint8 바이트 → b-bit 인덱스)
- `pack_signs` / `unpack_signs`: 1-bit 부호 패킹
- 모든 연산: 텐서 연산만 사용, Python 루프 제로

### 3. TurboDiskStore (`src/turboquant/memory.py`)
- Packed mmap 저장 (인덱스 + 부호 압축, 스케일 + S는 float)
- LUT 기반 검색 + 실시간 unpack
- EdenQuantizer 선택적 적용 (`use_eden` 플래그로 자동 감지)

## 정밀도 결과

### MSE (합성 단위 벡터, N=500)

| bits | dim | V2 | EDEN-biased | EDEN-unbiased | V3 vs V2 |
|------|-----|------|-------------|---------------|----------|
| 2 | 64 | 5.72e-3 | 6.35e-4 | 7.64e-4 | **-88.9%** |
| 2 | 1024 | 3.56e-4 | 4.06e-5 | 4.78e-5 | **-88.6%** |
| 3 | 64 | 2.80e-3 | 1.84e-4 | 1.93e-4 | **-93.4%** |
| 3 | 1024 | 3.57e-4 | 1.19e-5 | 1.24e-5 | **-96.7%** |
| 4 | 64 | 6.24e-4 | 5.75e-5 | 5.86e-5 | **-90.8%** |
| 4 | 1024 | 3.46e-4 | 3.67e-6 | 3.73e-6 | **-98.9%** |

### Recall@5 (d=128, 합성)

| bits | N | V2 | EDEN-biased | Δ |
|------|---|-----|-------------|------|
| 2 | 1000 | 0.29 | **0.75** | **+46%p** |
| 3 | 1000 | 0.27 | **0.86** | **+59%p** |
| 4 | 1000 | 0.57 | **0.92** | **+35%p** |

### 실록 실제 데이터 (BGE-M3 d=1024, N=280)

| 모델 | Recall@5 |
|------|----------|
| V2 TurboQuant | 65.0% |
| **EDEN V3 biased** | **91.0%** |

## 속도 결과 (d=1024, 3-bit)

| 연산 | V2 | EDEN V3 | 차이 |
|------|-----|---------|------|
| 인덱싱 (N=2000) | 1907 vec/s | 1871 vec/s | **-1.9%** |
| 검색 (N=2000) | 17.45 ms/q | 17.50 ms/q | **+0.3%** |
| 벡터당 검색 | 8.7 us/vec | 8.8 us/vec | 무시 가능 |

### Pack/Unpack 마이크로벤치마크 (d=1024)

| 연산 | 2-bit | 3-bit | 4-bit |
|------|-------|-------|-------|
| pack_bits | 38 us | 45 us | 43 us |
| unpack_bits | 45 us | 55 us | 49 us |
| pack_signs | 27 us | 27 us | 27 us |
| unpack_signs | 31 us | 31 us | 31 us |

## 저장 결과 (d=1024, 최대용량=1M)

| 포맷 | 2-bit | 3-bit | 4-bit |
|------|-------|-------|-------|
| **Packed** | 390 MB | 518 MB | 640 MB |
| uint8 (기존) | 2048 MB | 2048 MB | 2048 MB |
| **절감** | **81%** | **75%** | **69%** |

### 저장량 세분화 (3-bit, 벡터당)

| 구성요소 | 이전 (uint8) | 이후 (packed) |
|----------|-------------|---------------|
| 인덱스 | 1024 B | 384 B |
| 부호 | 1024 B | 128 B |
| scale | 2 B | 2 B |
| S 계수 | — | 4 B |
| **합계** | **2050 B** | **518 B** |

## 테스트 결과
- 15/15 EDEN 단위 테스트 통과
- V1/V2 회귀 테스트 통과
- 실록 E2E: recall@5 91% (목표: >65%)
- Pack/unpack 정확성: 모든 비트 설정 검증 완료

## Memory MCP 연동
- 세션 컨텍스트 memory MCP에 저장 (40개 항목)
- 의미 검색 검증 완료: 관련 결과 정확히 반환
- 카테고리/태그 필터 검색 지원
