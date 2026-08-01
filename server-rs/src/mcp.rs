use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::*,
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::embedding::EmbeddingModel;
use crate::quantizers::{self, EdenQuantizer, EdenState};
use crate::schema;
use crate::storage::{self, MmapIndex};
use crate::traits::Quantizer;

// ── Tool parameter structs (Deserialize + JsonSchema for auto-schema) ──

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RememberParams {
    /// Text content to remember
    pub text: String,
    /// Category label for filtering
    #[serde(default)]
    pub category: Option<String>,
    /// Source reference URL or file path
    #[serde(default)]
    pub source_ref: Option<String>,
    /// Comma-separated tags
    #[serde(default)]
    pub tags: Option<String>,
    /// Importance score (0.0–1.0, default 0.5)
    #[serde(default = "default_importance")]
    pub importance: f64,
    /// Additional metadata as JSON object
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

fn default_importance() -> f64 {
    0.5
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallParams {
    /// Natural-language query (embedded internally)
    pub query: String,
    /// Number of results to return (default 5)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Optional SQL WHERE clause for metadata filtering
    #[serde(default)]
    pub filters: Option<String>,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetParams {
    /// Entry ID to delete
    pub id: String,
}

// ── Shared state (wrapped in Arc<Mutex<>> for tool handlers) ────────────

pub struct McpState {
    pub index_path: String,
    pub db_path: String,
    pub quantizer: Arc<dyn Quantizer>,
    pub bits: u16,
    pub db: rusqlite::Connection,
    pub mmap_index: Option<MmapIndex>,
}

impl McpState {
    pub fn new(
        index_path: String,
        db_path: String,
        eden_state: EdenState,
    ) -> Result<Self, String> {
        let db = rusqlite::Connection::open(&db_path)
            .map_err(|e| format!("DB open error: {e}"))?;
        schema::init_db(&db).map_err(|e| format!("DB init error: {e}"))?;
        tracing::info!("Database initialized at {db_path}");

        let bits = eden_state.bits as u16;

        let quantizer = Arc::new(EdenQuantizer {
            dim: eden_state.dim,
            bits: eden_state.bits,
            mode: eden_state.mode,
            rotation: eden_state.rotation,
            codebook: eden_state.codebook,
        });

        // Rebuild .tmd from SQLite so MmapIndex stays in sync
        let count = schema::count_entries(&db).unwrap_or(0);
        if count > 0 {
            match schema::load_entries(&db, None) {
                Ok(entries) => {
                    if let Err(e) = storage::rebuild(&index_path, &entries, bits) {
                        tracing::warn!("Failed to rebuild .tmd index: {e}");
                    } else {
                        tracing::info!("Rebuilt .tmd with {count} entries");
                    }
                }
                Err(e) => tracing::warn!("Failed to load entries for .tmd rebuild: {e}"),
            }
        }

        // Load mmap index
        let mmap_index = MmapIndex::open(&index_path).ok();
        if mmap_index.is_some() {
            tracing::info!("MmapIndex loaded from {index_path}");
        } else if count > 0 {
            tracing::warn!("MmapIndex could not be opened, falling back to SQLite search");
        } else {
            tracing::info!("No .tmd index yet (empty store)");
        }

        Ok(Self {
            index_path,
            db_path,
            quantizer,
            bits,
            db,
            mmap_index,
        })
    }
}

// ── MCP handler: tools exposed to the client ──────────────────────────

#[derive(Clone)]
pub struct McpHandler {
    pub state: Arc<Mutex<McpState>>,
    pub embedder: Arc<Mutex<EmbeddingModel>>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl McpHandler {
    pub fn new(state: McpState, embedder: EmbeddingModel) -> Self {
        Self {
            state: Arc::new(Mutex::new(state)),
            embedder: Arc::new(Mutex::new(embedder)),
            tool_router: Self::tool_router(),
        }
    }

    fn do_remember(
        state: &mut McpState,
        text: &str,
        embedding: &[f32],
        importance: f32,
        category: &str,
        source_ref: &str,
        tags: &str,
        meta_json: &str,
    ) -> Result<String, McpError> {
        let q_res = state.quantizer.quantize(embedding);
        let scale = q_res.scale.unwrap_or(1.0);

        let indices: Vec<u8> = q_res.values.iter().map(|&v| v as u8).collect();
        let packed_sz = quantizers::packed_size(state.quantizer.dim(), state.bits as usize);
        let mut packed_buf = vec![0u8; packed_sz];
        quantizers::pack_indices(&indices, state.bits as usize, &mut packed_buf);

        let entry_id = format!("mem_{}", chrono_now_nanos());

        schema::insert_entry(
            &state.db,
            &entry_id,
            text,
            q_res.norm,
            q_res.r_norm,
            scale,
            &packed_buf,
            &q_res.signs.unwrap_or_default(),
            importance,
            category,
            source_ref,
            tags,
            meta_json,
        )
        .map_err(|e| {
            McpError::internal_error(format!("DB insert error: {e}"), None)
        })?;

        tracing::info!(
            "Added entry {entry_id} norm={:.4} r_norm={:.4} scale={:.4}",
            q_res.norm,
            q_res.r_norm,
            scale
        );

        // Rebuild .tmd mmap
        if let Ok(entries) = schema::load_entries(&state.db, None) {
            if let Err(e) = storage::rebuild(&state.index_path, &entries, state.bits) {
                tracing::warn!("Failed to rebuild .tmd after add: {e}");
            } else {
                state.mmap_index = MmapIndex::open(&state.index_path).ok();
            }
        }

        Ok(entry_id)
    }

    #[tool(description = "Store a new memory entry from text. The text is automatically embedded using E5-small (384-dim).")]
    async fn remember(
        &self,
        Parameters(RememberParams { text, importance, category, source_ref, tags, metadata }): Parameters<RememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let importance = importance as f32;
        let category = category.as_deref().unwrap_or("");
        let source_ref = source_ref.as_deref().unwrap_or("");
        let tags = tags.as_deref().unwrap_or("");
        let metadata = metadata.unwrap_or(serde_json::json!({}));
        let meta_json =
            serde_json::to_string(&metadata).map_err(|e| McpError::invalid_params(format!("Invalid metadata: {e}"), None))?;

        // 1. Embed
        let mut embedder = self.embedder.lock().await;
        let embedding = embedder.embed_passage(&text).map_err(|e| {
            McpError::internal_error(format!("Embedding failed: {e}"), None)
        })?;
        drop(embedder);

        // 2. Quantize & store
        let mut state = self.state.lock().await;
        let entry_id = Self::do_remember(
            &mut state,
            &text,
            &embedding,
            importance,
            category,
            source_ref,
            tags,
            &meta_json,
        )?;
        drop(state);

        let result = serde_json::json!({"entry_id": entry_id});
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&result).unwrap_or_default(),
        )]))
    }

    #[tool(description = "Search memory by natural-language query. The query is automatically embedded using E5-small.")]
    async fn recall(
        &self,
        Parameters(RecallParams { query, top_k, filters }): Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let top_k = top_k.max(1);
        let filters = filters.as_deref();

        // 1. Embed query
        let mut embedder = self.embedder.lock().await;
        let query_vec = embedder.embed_query(&query).map_err(|e| {
            McpError::internal_error(format!("Query embedding failed: {e}"), None)
        })?;
        drop(embedder);

        // 2. Search
        let state = self.state.lock().await;

        let results = if let Some(ref mmap) = state.mmap_index {
            let entries = schema::load_metadata_only(&state.db, filters).map_err(|e| {
                McpError::internal_error(format!("DB load error: {e}"), None)
            })?;

            let mmap_count = mmap.len() as usize;
            let mut scored: Vec<(f32, &schema::SearchEntry)> = entries
                .iter()
                .enumerate()
                .map(|(row_idx, e)| {
                    if row_idx >= mmap_count || e.norm < 1e-12 {
                        return (0.0f32, e);
                    }
                    let idx = row_idx as u64;
                    let packed_slice = unsafe {
                        let base = mmap.packed_offset + idx * mmap.stride_packed;
                        std::slice::from_raw_parts(
                            mmap.mmap.as_ptr().add(base as usize),
                            mmap.stride_packed as usize,
                        )
                    };
                    let scale = mmap.scales()[idx as usize];
                    let score = state.quantizer.score_packed(&query_vec, packed_slice, scale);
                    (score, e)
                })
                .collect();

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(top_k);

            scored
                .into_iter()
                .map(|(score, e)| {
                    serde_json::json!({
                        "id": e.id,
                        "text": e.text,
                        "score": (score * 10000.0).round() / 10000.0,
                        "metadata": e.metadata,
                    })
                })
                .collect::<Vec<_>>()
        } else {
            let entries = schema::load_entries(&state.db, filters).map_err(|e| {
                McpError::internal_error(format!("DB load error: {e}"), None)
            })?;

            let mut scored: Vec<(f32, &schema::SearchEntry)> = entries
                .iter()
                .map(|e| {
                    if e.norm < 1e-12 {
                        return (0.0f32, e);
                    }
                    let score = state.quantizer.score_packed(&query_vec, &e.packed, e.scale);
                    (score, e)
                })
                .collect();

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(top_k);

            scored
                .into_iter()
                .map(|(score, e)| {
                    serde_json::json!({
                        "id": e.id,
                        "text": e.text,
                        "score": (score * 10000.0).round() / 10000.0,
                        "metadata": e.metadata,
                    })
                })
                .collect::<Vec<_>>()
        };
        drop(state);

        let result = serde_json::json!({"results": results});
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&result).unwrap_or_default(),
        )]))
    }

    #[tool(description = "Delete a memory entry by its ID")]
    async fn forget(
        &self,
        Parameters(ForgetParams { id }): Parameters<ForgetParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut state = self.state.lock().await;
        let deleted = schema::delete_entry(&state.db, &id)
            .map_err(|e| McpError::internal_error(format!("DB delete error: {e}"), None))?;

        if deleted {
            if let Ok(entries) = schema::load_entries(&state.db, None) {
                if let Err(e) = storage::rebuild(&state.index_path, &entries, state.bits) {
                    tracing::warn!("Failed to rebuild .tmd after delete: {e}");
                } else {
                    state.mmap_index = MmapIndex::open(&state.index_path).ok();
                }
            }
        }
        drop(state);

        let result = serde_json::json!({"success": deleted});
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&result).unwrap_or_default(),
        )]))
    }

    #[tool(description = "Return memory store statistics")]
    async fn memory_stats(&self) -> Result<CallToolResult, McpError> {
        let state = self.state.lock().await;
        let count = schema::count_entries(&state.db).unwrap_or(0);
        let compressed_bytes = state.quantizer.dim() as f32 * 1.0;
        let fp32_bytes = state.quantizer.dim() as f32 * 4.0;
        let ratio = if compressed_bytes > 0.0 {
            fp32_bytes / compressed_bytes
        } else {
            1.0
        };
        let mmap_entries = state.mmap_index.as_ref().map(|m| m.len()).unwrap_or(0);
        drop(state);

        let result = serde_json::json!({
            "total_entries": count,
            "mmap_entries": mmap_entries,
            "compression": "eden_v3",
            "bit_width": 3,
            "dimension": 384,
            "embedding_model": "intfloat/multilingual-e5-small",
            "fp32_bytes_per_entry": fp32_bytes,
            "compressed_bytes_per_entry": compressed_bytes,
            "compression_ratio": (ratio * 100.0).round() / 100.0,
        });
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&result).unwrap_or_default(),
        )]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for McpHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn chrono_now_nanos() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64
}
