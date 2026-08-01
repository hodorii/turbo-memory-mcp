use rmcp::ServiceExt;
use turbo_memory_rs::{eden, embedding::EmbeddingModel, mcp};

fn parse_args() -> (String, String, String) {
    let args: Vec<String> = std::env::args().collect();
    let mut index_path = "vectors.tmd".to_string();
    let mut db_path = "memory.db".to_string();
    let mut state_path = "state.bin".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--index" => {
                i += 1;
                if i < args.len() {
                    index_path = args[i].clone();
                }
            }
            "--db" => {
                i += 1;
                if i < args.len() {
                    db_path = args[i].clone();
                }
            }
            "--state" => {
                i += 1;
                if i < args.len() {
                    state_path = args[i].clone();
                }
            }
            "--help" | "-h" => {
                eprintln!(
                    "turbo-memory-rs [--index <path>] [--db <path>] [--state <path>]"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    (index_path, db_path, state_path)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let (index_path, db_path, state_path) = parse_args();

    tracing::info!("turbo-memory-rs starting");
    tracing::info!("  index: {index_path}");
    tracing::info!("  db:    {db_path}");
    tracing::info!("  state: {state_path}");

    // Load EDEN quantizer state
    let eden_state = eden::load_state(&state_path).unwrap_or_else(|e| {
        tracing::warn!("No EDEN state at {state_path}, creating default: {e}");
        turbo_memory_rs::quantizers::EdenState {
            dim: 384,
            bits: 3,
            mode: "unbiased".into(),
            codebook: vec![],
            rotation: vec![],
            qjl_matrix: vec![],
        }
    });
    tracing::info!(
        "EDEN state loaded: dim={}, bits={}",
        eden_state.dim,
        eden_state.bits
    );

    // Initialize E5-small embedding model
    tracing::info!("Initializing E5-small embedding model...");
    let embedder = EmbeddingModel::new()?;
    tracing::info!("Embedding model ready (dim={})", embedder.dim());

    // Initialize MCP state
    let state = mcp::McpState::new(index_path, db_path, eden_state)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Build handler and serve via stdio transport
    let handler = mcp::McpHandler::new(state, embedder);
    let service = handler.serve(rmcp::transport::stdio()).await?;
    tracing::info!("turbo-memory-rs running (stdio transport)");
    service.waiting().await?;

    tracing::info!("turbo-memory-rs shutting down");
    Ok(())
}
