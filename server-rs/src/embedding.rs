use fastembed::{EmbeddingModel as FastEmbedModel, InitOptions, TextEmbedding};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("FastEmbed error: {0}")]
    FastEmbed(#[from] fastembed::Error),
    #[error("No embedding returned for input")]
    EmptyResult,
}

/// E5-small embedding model wrapper.
///
/// Provides passage/query prefixed embedding via ONNX Runtime:
/// - `embed_passage()` → prepends `"passage: "` (for storage)
/// - `embed_query()` → prepends `"query: "` (for search)
///
/// Dimension: 384 (intfloat/multilingual-e5-small).
pub struct EmbeddingModel {
    model: TextEmbedding,
    dim: usize,
}

impl EmbeddingModel {
    /// Initialize the E5-small embedding model.
    ///
    /// Downloads the ONNX model on first run (~100MB) and caches it.
    pub fn new() -> Result<Self, EmbeddingError> {
        let model = TextEmbedding::try_new(
            InitOptions::new(FastEmbedModel::MultilingualE5Small)
                .with_show_download_progress(true),
        )?;
        Ok(Self { model, dim: 384 })
    }

    /// Embed a single passage for storage.
    ///
    /// Prepends E5-required `"passage: "` prefix.
    pub fn embed_passage(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let prefixed = format!("passage: {}", text);
        let mut embeddings = self.model.embed(vec![prefixed], Some(1))?;
        embeddings.pop().ok_or(EmbeddingError::EmptyResult)
    }

    /// Embed a single query for search.
    ///
    /// Prepends E5-required `"query: "` prefix.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, EmbeddingError> {
        let prefixed = format!("query: {}", query);
        let mut embeddings = self.model.embed(vec![prefixed], Some(1))?;
        embeddings.pop().ok_or(EmbeddingError::EmptyResult)
    }

    /// Return the embedding dimension (384).
    pub fn dim(&self) -> usize {
        self.dim
    }
}
