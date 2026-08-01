use rusqlite::{params, Connection, Result};

/// An entry returned from a search query.
#[derive(Debug, Clone)]
pub struct SearchEntry {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub norm: f32,
    pub r_norm: f32,
    pub scale: f32,
    pub packed: Vec<u8>,
    pub qjl: Vec<i8>,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Initialize the SQLite database schema.
pub fn init_db(conn: &Connection) -> Result<()> {
    conn.execute_batch("PRAGMA journal_mode=WAL")?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            norm REAL NOT NULL DEFAULT 1.0,
            r_norm REAL NOT NULL DEFAULT 0.0,
            scale REAL NOT NULL DEFAULT 1.0,
            packed BLOB NOT NULL,
            qjl BLOB NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            category TEXT DEFAULT '',
            source_ref TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}'
        )",
    )?;
    // Migration: add scale column if missing (for databases created before v2 schema)
    let _ = conn.execute_batch("ALTER TABLE entries ADD COLUMN scale REAL NOT NULL DEFAULT 1.0");
    Ok(())
}

/// Insert a new memory entry.
pub fn insert_entry(
    conn: &Connection,
    id: &str,
    text: &str,
    norm: f32,
    r_norm: f32,
    scale: f32,
    packed: &[u8],
    qjl: &[i8],
    importance: f32,
    category: &str,
    source_ref: &str,
    tags: &str,
    metadata: &str,
) -> Result<()> {
    let qjl_bytes: &[u8] = bytemuck::cast_slice(qjl);
    conn.execute(
        "INSERT INTO entries (id, text, norm, r_norm, scale, packed, qjl, importance, category, source_ref, tags, metadata)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![id, text, norm, r_norm, scale, packed, qjl_bytes, importance, category, source_ref, tags, metadata],
    )?;
    Ok(())
}

/// Delete an entry by id.
pub fn delete_entry(conn: &Connection, id: &str) -> Result<bool> {
    let n = conn.execute("DELETE FROM entries WHERE id = ?1", params![id])?;
    Ok(n > 0)
}

/// Load all entries (optionally filtered) from the database, including vector data.
pub fn load_entries(
    conn: &Connection,
    filters: Option<&str>,
) -> Result<Vec<SearchEntry>> {
    let mut sql = String::from(
        "SELECT id, text, norm, r_norm, scale, packed, qjl, metadata FROM entries",
    );
    append_filters(&mut sql, filters);
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| row_to_entry(row))?;
    let mut entries = Vec::new();
    for row in rows {
        entries.push(row?);
    }
    Ok(entries)
}

/// Load only metadata fields from the database (skips packed/qjl blobs).
/// Use this when vector data is served from mmap instead of SQLite.
pub fn load_metadata_only(
    conn: &Connection,
    filters: Option<&str>,
) -> Result<Vec<SearchEntry>> {
    let mut sql = String::from(
        "SELECT id, text, norm, r_norm, scale, metadata FROM entries",
    );
    append_filters(&mut sql, filters);
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let id: String = row.get(0)?;
        let text: String = row.get(1)?;
        let norm: f32 = row.get(2)?;
        let r_norm: f32 = row.get(3)?;
        let scale: f32 = row.get(4)?;
        let metadata_json: String = row.get(5)?;
        let metadata: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&metadata_json).unwrap_or_default();
        Ok(SearchEntry {
            id,
            text,
            score: 0.0,
            norm,
            r_norm,
            scale,
            packed: Vec::new(),
            qjl: Vec::new(),
            metadata,
        })
    })?;
    let mut entries = Vec::new();
    for row in rows {
        entries.push(row?);
    }
    Ok(entries)
}

fn append_filters(sql: &mut String, filters: Option<&str>) {
    if let Some(f) = filters {
        sql.push_str(" WHERE ");
        sql.push_str(f);
    }
}

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<SearchEntry> {
    let id: String = row.get(0)?;
    let text: String = row.get(1)?;
    let norm: f32 = row.get(2)?;
    let r_norm: f32 = row.get(3)?;
    let scale: f32 = row.get(4)?;
    let packed: Vec<u8> = row.get(5)?;
    let qjl_bytes: Vec<u8> = row.get(6)?;
    let metadata_json: String = row.get(7)?;

    let qjl: Vec<i8> = if qjl_bytes.len() % std::mem::size_of::<i8>() == 0 {
        bytemuck::cast_slice(&qjl_bytes).to_vec()
    } else {
        Vec::new()
    };

    let metadata: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(&metadata_json).unwrap_or_default();

    Ok(SearchEntry {
        id,
        text,
        score: 0.0,
        norm,
        r_norm,
        scale,
        packed,
        qjl,
        metadata,
    })
}

/// Count total entries.
pub fn count_entries(conn: &Connection) -> Result<i64> {
    conn.query_row("SELECT COUNT(*) FROM entries", [], |row| row.get(0))
}
