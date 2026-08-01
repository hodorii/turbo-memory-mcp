# 비즈니스 프로세스 설계: 알고리즘 비교 프레임워크

본 문서는 양자화 알고리즘의 체계적인 통합과 비교 분석을 위한 상세 워크플로우를 정의한다.

## 1. 프로세스: 새로운 알고리즘 통합 (New Algorithm Integration)
**목표**: 코어 엔진의 수정 없이 새로운 양자화 알고리즘을 시스템에 추가하고 등록한다.
**성공 조건**: 신규 알고리즘이 레지스트리에 등록되어 `BaseQuantizer` 인터페이스를 통해 호출 가능함.

### 태스크 1.1: 알고리즘 구현 및 인터페이스 준수
- **기능 그룹**: `Quantizer Implementation`
- **단계**:
  1.1.1. [Python] `BaseQuantizer` 상속 및 필수 메서드(`quantize`, `decode`, `calculate_score`) 구현
  1.1.2. [Rust] `Quantizer` Trait 구현 및 `QuantizedResult` 구조체 매핑
- **로직 (AST)**:
  - `class NewAlgo(BaseQuantizer): def quantize(self, x) -> QuantizedResult: ...`
  - `impl Quantizer for NewAlgo { fn quantize(&self, x: &[f32]) -> QuantizedResult { ... } }`

### 태스크 1.2: 알고리즘 레지스트리 등록
- **기능 그룹**: `Algorithm Registry`
- **단계**:
  1.2.1. 알고리즘 고유 ID 및 설정값(bits, mode 등) 정의
  1.2.2. 팩토리 패턴을 통한 레지스트리 매핑 추가
- **로직 (AST)**:
  - `@QuantizerRegistry.register("ALGO_ID")`
  - `registry.register("ALGO_ID", || Box::new(NewAlgo::new()))`

---

## 2. 프로세스: 비교 벤치마킹 수행 (Comparative Benchmarking)
**목표**: 여러 알고리즘의 성능(Latency), 정확도(vNMSE, Recall), 메모리 효율을 동일 조건에서 측정한다.
**성공 조건**: 모든 대상 알고리즘에 대한 정량적 지표가 수집되어 비교 리포트로 생성됨.

### 태스크 2.1: 평가 데이터셋 및 쿼리 준비
- **기능 그룹**: `Dataset Manager`
- **단계**:
  2.1.1. 표준 벤치마크 데이터셋 로드 및 $\ell_2$-normalization 수행
  2.1.2. 데이터셋 외부의 독립적인 쿼리 벡터 생성 및 정규화
- **로직 (AST)**:
  - `data = load_dataset(); data /= np.linalg.norm(data, axis=1, keepdims=True)`

### 태스크 2.2: 알고리즘별 지표 측정
- **기능 그룹**: `Benchmark Runner`
- **단계**:
  2.2.1. [양자화] 각 알고리즘으로 데이터셋 전체 양자화 및 시간/메모리 측정
  2.2.2. [복원] 양자화된 결과로부터 벡터 복원 및 vNMSE 계산
  2.2.3. [검색] 쿼리 벡터에 대해 `calculate_score`를 수행하여 Recall@K 측정
- **로직 (AST)**:
  - `start = time.perf_counter_ns(); q_res = quantizer.quantize(x); end = time.perf_//counter_ns()`
  - `vnmse = mean(sum((x - x_hat)**2) / sum(x**2))`
  - `recall = len(intersect(gt_top_k, q_top_k)) / k`

### 태스크 2.3: 비교 리포트 생성
- **기능 그룹**: `Report Generator`
- **단계**:
  2.3.1. 수집된 `AlgoMetrics` 데이터를 기반으로 비교 테이블 생성
  2.3.2. Markdown 및 JSON 포맷으로 결과 파일 출력
- **로직 (AST)**:
  - `generate_markdown_table(metrics_list)` $\rightarrow$ `report.md`

---

## 3. 프로세스: 교차 언어 무결성 검증 (Cross-Language Verification)
**목표**: Python 구현체와 Rust 구현체의 결과가 수치적으로 일치하는지 확인하여 포팅 오류를 제거한다.
**성공 조건**: 동일 입력에 대해 두 언어의 스코어 차이가 $\epsilon < 10^{-6}$ 이내임.

### 태스크 3.1: 언어별 결과 덤프
- **기능 그룹**: `Consistency Checker`
- **단계**:
  3.1.1. Python 엔진에서 특정 벡터셋의 `QuantizedResult` 및 점수 추출 $\rightarrow$ JSON 저장
  3.1.2. Rust 엔진에서 동일 벡터셋의 결과 추출 $\rightarrow$ JSON 저장
- **로직 (AST)**:
  - `save_json(algo_id, input_vec, result)`

### 태스크 3.2: 결과 정밀 비교 및 분석
- **기능 그룹**: `Verification Engine`
- **단계**:
  3.2.1. 두 JSON 파일을 로드하여 인덱스별 점수 1:1 매핑
  3.2.2. 절대 오차(Absolute Difference) 계산 및 임계값 검증
- **로직 (AST)**:
  - `diff = abs(score_py - score_rs); assert diff < 1e-6`
