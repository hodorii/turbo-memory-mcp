use crate::traits::Quantizer;
use std::collections::HashMap;
use std::sync::Arc;

pub type QuantizerPtr = Arc<dyn Quantizer>;

pub struct QuantizerRegistry {
    registry: HashMap<String, Box<dyn Fn() -> QuantizerPtr + Send + Sync>>,
}

impl QuantizerRegistry {
    pub fn new() -> Self {
        Self {
            registry: HashMap::new(),
        }
    }

    /// 알고리즘 ID와 생성자 함수를 등록한다.
    pub fn register<F>(&mut self, algo_id: &str, factory: F)
    where
        F: Fn() -> QuantizerPtr + 'static + Send + Sync,
    {
        self.registry.insert(algo_id.to_string(), Box::new(factory));
    }

    /// 알고리즘 ID를 통해 구현체 인스턴스를 생성하여 반환한다.
    pub fn get_quantizer(&self, algo_id: &str) -> Option<QuantizerPtr> {
        self.registry.get(algo_id).map(|factory| factory())
    }

    /// 등록된 모든 알고리즘 ID 목록을 반환한다.
    pub fn list_available(&self) -> Vec<String> {
        self.registry.keys().cloned().collect()
    }
}

impl Default for QuantizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
