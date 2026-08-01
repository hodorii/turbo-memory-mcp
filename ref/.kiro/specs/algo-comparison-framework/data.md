# 데이터 설계: 알고리즘 비교 프레임워크

본 문서는 양자화 알고리즘 비교를 위해 사용되는 핵심 데이터 구조와 리포트 스키마를 정의한다.

## 1. 양자화 결과 구조 (QuantizedResult)

Python과 Rust 간의 데이터 교환 및 저장의 일관성을 위해 다음 구조를 사용한다.

| 필드명 | 타입 | 설명 | 비고 |
| :--- | :--- | :--- | :--- |
| `algo_id` | String | 사용된 알고리즘 식별자 (예: "DRIVE_V3", "QJL") | 필수 |
| `quantized_values` | IntArray | 양자화된 좌표 값들 | 필수 |
| `residual_signs` | BitArray/BoolArray | 잔차의 부호 (회전 공간 기준) | 선택 (DRIVE/QJL 전용) |
| `scale` | Float | 복원을 위한 최적 스케일 값 $S$ | 선택 |
| `meta` | Map | 알고리즘별 추가 파라미터 (예: $R$ 행렬 버전) | 선택 |

## 2. 벤치마크 리포트 스키마 (BenchmarkReport)

측정된 성능과 정확도를 저장하는 JSON 스키마를 정의한다.

### 2.1 개별 알고리즘 결과 (AlgoMetric)
```json
{
  "algo_id": "DRIVE_V3",
  "metrics": {
    "accuracy": {
      "vNMSE": 0.5712,
      "recall_at_1": 1.0,
      "recall_at_10": 1.0
    },
    "performance": {
      "latency_ns_per_op": 120,
      "throughput_vec_per_sec": 8333333,
      "memory_bytes_per_vec": 4.5
    }
  },
  "timestamp": "2026-05-15T10:00:00Z",
  "environment": {
    "language": "python",
    "device": "Apple M2 Max",
    "dim": 1024
  }
}
```

### 2.2 종합 비교 리포트 (ComparisonSummary)
- **구조**: `List<AlgoMetric>`
- **정렬 기준**: vNMSE $\rightarrow$ Latency $\rightarrow$ Memory 순으로 정렬하여 최적 알고리즘 도출.
