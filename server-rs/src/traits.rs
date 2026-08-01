use serde::{Serialize, Deserialize};

/// 양자화된 결과 및 복원을 위한 메타데이터를 담는 구조체
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedResult {
    pub algo_id: String,
    pub values: Vec<i32>,        // 양자화된 좌표 값 (또는 packed 데이터)
    pub signs: Option<Vec<i8>>, // 잔차의 부호 (-1, 0, 1)
    pub scale: Option<f32>,      // 복원을 위한 최적 스케일 S
    pub norm: f32,               // 입력 벡터의 norm
    pub r_norm: f32,             // 잔차 벡터의 norm
}

/// 모든 양자화 알고리즘이 구현해야 하는 표준 트레이트
pub trait Quantizer: Send + Sync {
    /// Quantizer의 차원을 반환한다.
    fn dim(&self) -> usize;

    /// 입력 벡터 x를 양자화하여 QuantizedResult를 반환
    fn quantize(&self, x: &[f32]) -> QuantizedResult;

    /// 양자화된 결과 q를 사용하여 원래 벡터 x를 복원
    fn decode(&self, q: &QuantizedResult) -> Vec<f32>;

    /// 쿼리 벡터와 양자화된 결과 사이의 보정된 내적 점수를 계산
    fn score(&self, query: &[f32], q: &QuantizedResult) -> f32;

    /// Score a query directly against packed bytes + S factor, bypassing unpack.
    /// Supports 2/3/4-bit packed formats when the quantizer supports it.
    /// Default implementation falls back to score() via unpack.
    fn score_packed(&self, query: &[f32], packed: &[u8], scale: f32) -> f32 {
        let q_res = QuantizedResult {
            algo_id: String::new(),
            values: packed.iter().map(|&v| v as i32).collect(),
            signs: None,
            scale: Some(scale),
            norm: 1.0,
            r_norm: 0.0,
        };
        self.score(query, &q_res)
    }
}
