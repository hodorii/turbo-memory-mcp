# 백엔드 기술 사양: 알고리즘 비교 프레임워크

본 문서는 양자화 알고리즘을 체계적으로 추가하고 성능을 측정하기 위한 백엔드 아키텍처를 정의한다.

## 1. 플러그형 아키텍처 (Pluggable Architecture)

새로운 알고리즘을 코어 엔진 수정 없이 추가할 수 있도록 표준 인터페이스를 도입한다.

### 1.1 Python 인터페이스 (ABC)
`BaseQuantizer` 추상 베이스 클래스를 정의하여 모든 알고리즘이 동일한 메서드를 구현하게 한다.
- `quantize(vector: np.ndarray) -> QuantizedResult`
- `decode(quantized: QuantizedResult) -> np.ndarray`
- `calculate_score(query: np.ndarray, quantized: QuantizedResult) -> float`

### 1.2 Rust 인터페이스 (Trait)
`Quantizer` 트레이트를 정의하여 Rust MCP 서버에서도 동일한 구조를 유지한다.
- `fn quantize(&self, x: &[f32]) -> QuantizedResult`
- `fn decode(&self, q: &QuantizedResult) -> Vec<f32>`
- `fn score(&self, query: &[f32], q: &QuantizedResult) -> f32`

### 1.3 알고리즘 레지스트리
설정 파일(YAML/JSON)을 통해 사용할 알고리즘을 지정하고, 런타임에 해당 클래스/구조체를 로드하는 팩토리 패턴을 적용한다.

## 2. 벤치마크 하네스 설계 (Benchmarking Harness)

알고리즘 간의 객관적 비교를 위해 다음과 같은 측정 도구를 구축한다.

### 2.1 성능 지표 측정
- **지연 시간 (Latency)**: `time.perf_counter_ns` (Python) 및 `std::time::Instant` (Rust)를 사용하여 단일 연산 및 배치 연산의 나노초(ns) 단위 시간을 측정한다.
- **메모리 사용량 (Memory)**: 벡터당 할당되는 실제 바이트 수를 계산한다. (예: $b$ bits/dim $\times d$ dim)
- **처리량 (Throughput)**: 초당 처리 가능한 벡터 수 (Vectors/sec)를 측정한다.

### 2.2 측정 프로세스
1. 데이터셋 로드 $\rightarrow$ 2. 알고리즘별 양자화 $\rightarrow$ 3. 복원 및 점수 계산 $\rightarrow$ 4. 결과 수집 및 통계 분석

## 3. 통합 및 검증 전략 (Integration & Verification)

Python과 Rust 구현체의 일치성을 검증하여 포팅 오류를 방지한다.

- **데이터 교환 포맷**: 동일한 입력 벡터와 양자화 결과(q, r)를 JSON/CSV 포맷으로 저장하여 상호 교차 검증한다.
- **정확도 동기화**: 동일한 입력에 대해 Python의 `calculate_score` 결과와 Rust의 `score` 결과가 오차 범위 $\epsilon < 10^{-6}$ 이내인지 확인한다.

## 4. 비기능적 요구사항

- **오버헤드 최소화**: 벤치마크 도구가 측정 대상의 성능에 영향을 주지 않도록 가벼운 래퍼(Wrapper) 구조를 사용한다.
- **확장성**: 새로운 측정 지표(예: 에너지 효율, 캐시 미스율)가 추가될 수 있도록 측정 모듈을 분리한다.
