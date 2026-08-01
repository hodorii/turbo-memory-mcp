# 최종 구현 태스크: 알고리즘 비교 프레임워크

본 문서는 `algo-comparison-framework`의 설계를 실제 코드로 구현하기 위한 원자적 태스크 리스트이다. 모든 태스크는 `design.md`와 `biz-process.md`의 정의를 엄격히 따른다.

## Wave 1: 기반 구조 구축 (Foundation)
*목표: 알고리즘을 플러그인 형태로 추가하고 관리할 수 있는 표준 인터페이스 및 레지스트리 체계 완성*

### Task 1.1: Python 표준 인터페이스 및 데이터 구조 구현
- **설명**: `src/turboquant/interfaces.py`에 `BaseQuantizer` ABC와 `QuantizedResult` 데이터 클래스를 구현한다.
- **상세 요구사항**:
  - `QuantizedResult`: `algo_id`, `values`, `signs`, `scale` 필드 포함.
  - `BaseQuantizer`: `quantize`, `decode`, `calculate_score` 추상 메서드 정의.
- **수락 기준**: 
  - `interfaces.py` 파일이 존재하고 타입 힌트가 정확히 적용됨.
  - `BaseQuantizer`를 상속받지 않은 클래스는 인스턴스화할 수 없음.
- **카테고리**: `backend`

### Task 1.2: Rust 표준 인터페이스 및 데이터 구조 구현
- **설명**: `server-rs/src/traits.rs`에 `Quantizer` Trait과 `QuantizedResult` 구조체를 구현한다.
- **상세 요구사항**:
  - Python의 `QuantizedResult`와 필드 구성 및 타입이 일치해야 함.
  - `Quantizer` Trait에 `quantize`, `decode`, `score` 메서드 정의.
- **수락 기준**: 
  - `traits.rs` 파일이 존재하며 `serde` 직렬화/역직렬화가 가능함.
- **카테고리**: `backend`

### Task 1.3: 양방향 알고리즘 레지스트리 구현
- **설명**: ID를 통해 알고리즘 구현체를 동적으로 로드하는 팩토리 패턴을 구현한다.
- **상세 요구사항**:
  - [Python] `src/turboquant/registry.py`에 `@QuantizerRegistry.register` 데코레이터 구현.
  - [Rust] `server-rs/src/registry.rs`에 `QuantizerRegistry` 구조체 및 등록/조회 로직 구현.
- **수락 기준**: 
  - 특정 `algo_id`를 입력했을 때 해당 클래스/구조체의 인스턴스가 정상적으로 반환됨.
- **카테고리**: `backend`

---

## Wave 2: 알고리즘 플러그인화 (Algorithm Migration)
*목표: 기존 로직을 표준 인터페이스로 리팩토링하여 '플러그인'으로 전환*

### Task 2.1: DRIVE_V3 알고리즘 구현 (Python & Rust)
- **설명**: `ml-spec.md`의 수학적 정의를 바탕으로 `DriveV3Quantizer`를 구현한다.
- **핵심 구현 포인트**:
  - 회전 공간($y - q_{val}$)에서 잔차 부호 계산 및 저장.
  - DRIVE Theorem 2 기반의 최적 스케일 $s = \mathbb{E}[|\langle r, y \rangle|] / d$ 적용.
  - `calculate_score` 시 회전된 쿼리 $y_{query}$와 잔차 부호의 내적 보정 수행.
- **수락 기준**: 
  - Python과 Rust 버전의 `calculate_score` 결과가 동일한 입력에 대해 $\epsilon < 10^{-6}$ 이내로 일치함.
- **카테고리**: `ml`

### Task 2.2: QJL 알고리즘 구현 (Python & Rust)
- **설명**: 비교 베이스라인으로서 `QJLQuantizer`를 구현한다.
- **핵심 구현 포인트**:
  - 표준 인터페이스를 준수하되, 고정 스케일(Fixed Scale)을 사용하여 DRIVE와의 차별성을 둠.
  - 동일하게 회전 공간 기반 잔차 보정을 적용하여 '최소한의 정합성' 확보.
- **수락 기준**: 
  - `DRIVE_V3`보다 낮은 vNMSE와 Recall을 보이며 정상적으로 동작함.
- **카테고리**: `ml`

---

## Wave 3: 벤치마크 하네스 구현 (Benchmarking Harness)
*목표: 정밀한 성능 및 정확도 측정 도구 구축*

### Task 3.1: 지표 측정 모듈 구현 (Metrics Collector)
- **설명**: vNMSE, Recall@K, Latency, Memory 사용량을 계산하는 모듈을 구현한다.
- **상세 요구사항**:
  - **vNMSE**: $\mathbb{E}[\|x - \hat{x}\|^2 / \|x\|^2]$ 공식 적용.
  - **Recall**: Brute-force Float32 Top-K와의 중첩도 계산.
  - **Latency**: `perf_counter_ns` 등을 사용하여 연산당 순수 시간 측정 (오버헤드 제외).
- **수락 기준**: 
  - 측정된 지표가 `AlgoMetrics` 데이터 클래스에 정확히 저장됨.
- **카테고리**: `ml`

### Task 3.2: 벤치마크 실행 제어기 및 리포트 생성기 구현
- **설명**: 데이터셋 로드 $\rightarrow$ 알고리즘 순회 $\rightarrow$ 측정 $\rightarrow$ 리포트 출력 흐름을 제어한다.
- **상세 요구사항**:
  - `benchmark.py`에서 비트 수([2, 3, 4])별 루프 수행.
  - 결과물을 Markdown 테이블 및 JSON 파일로 저장.
- **수락 기준**: 
  - 모든 알고리즘에 대해 비트별 성능 비교표가 정상적으로 출력됨.
- **카테고리**: `backend`

---

## Wave 4: 최종 검증 및 리포팅 (Verification & Reporting)
*목표: 실데이터 기반의 최종 성능 확정 및 포팅 무결성 검증*

### Task 4.1: 교차 언어 일치성 검증 (Cross-Language Check)
- **설명**: 동일 데이터셋에 대해 Python 결과와 Rust 결과가 비트 단위로 일치하는지 확인하는 테스트 도구를 구현한다.
- **수락 기준**: 모든 테스트 케이스에서 두 언어의 결과값이 허용 오차 이내로 일치함.
- **카테고리**: `backend`

### Task 4.2: 실데이터 기반 최종 벤치마크 수행 및 리포트 작성
- **설명**: `sillok_sample.json` 등 실제 임베딩 데이터를 사용하여 최종 성능 지표를 도출한다.
- **수락 기준**: 
  - DRIVE_V3가 QJL 대비 압도적인 vNMSE 및 Recall 성능을 보임을 수치로 증명.
  - 최종 리포트가 `requirements.md`의 수락 기준을 모두 만족함.
- **카테고리**: `writing`
