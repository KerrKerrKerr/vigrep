use crate::text::{decode_embedding, encode_embedding};
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FileRecord {
    pub path: String,
    pub hash: String,
    pub size: i64,
    pub indexed_at: i64,
    pub complete: i64,
}

#[derive(Debug, Clone)]
pub struct FileState {
    pub hash: String,
    pub complete: bool,
}

#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub path: String,
    pub file_hash: String,
    pub chunk_index: i64,
    pub start_line: i64,
    pub end_line: i64,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SearchRow {
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub file_count: i64,
    pub chunk_count: i64,
    pub source_bytes: i64,
    pub embedded_bytes: i64,
    pub last_indexed_at: Option<i64>,
}

pub fn open_database(path: &Path) -> Result<Connection> {
    let connection = Connection::open(path)
        .with_context(|| format!("Failed to open database at {}", path.display()))?;
    connection.pragma_update(None, "journal_mode", "WAL")?;
    connection.pragma_update(None, "synchronous", "NORMAL")?;
    connection.busy_timeout(std::time::Duration::from_secs(5))?;
    initialize_schema(&connection)?;
    Ok(connection)
}

fn initialize_schema(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            size INTEGER NOT NULL,
            indexed_at INTEGER NOT NULL,
            complete INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            file_hash TEXT NOT NULL,
            UNIQUE(path, file_hash, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
        CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON chunks(file_hash);
        "#,
    )?;
    Ok(())
}

pub fn load_known_files(connection: &Connection) -> Result<HashMap<String, FileState>> {
    let mut statement = connection.prepare("SELECT path, hash, complete FROM files")?;
    let rows = statement.query_map([], |row| {
        let path: String = row.get(0)?;
        let hash: String = row.get(1)?;
        let complete: i64 = row.get(2)?;
        Ok((
            path,
            FileState {
                hash,
                complete: complete != 0,
            },
        ))
    })?;

    let mut hashes = HashMap::new();
    for row in rows {
        let (path, state) = row?;
        hashes.insert(path, state);
    }
    Ok(hashes)
}

pub fn load_chunk_indexes(
    connection: &Connection,
    path: &str,
    file_hash: &str,
) -> Result<HashSet<i64>> {
    let mut statement =
        connection.prepare("SELECT chunk_index FROM chunks WHERE path = ?1 AND file_hash = ?2")?;

    let rows = statement.query_map(params![path, file_hash], |row| row.get::<_, i64>(0))?;

    let mut chunk_indexes = HashSet::new();
    for row in rows {
        chunk_indexes.insert(row?);
    }

    Ok(chunk_indexes)
}

pub fn delete_paths(connection: &mut Connection, paths: &[String]) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }

    let transaction = connection.transaction()?;
    {
        let mut delete_chunks = transaction.prepare("DELETE FROM chunks WHERE path = ?1")?;
        let mut delete_files = transaction.prepare("DELETE FROM files WHERE path = ?1")?;
        for path in paths {
            delete_chunks.execute(params![path])?;
            delete_files.execute(params![path])?;
        }
    }
    transaction.commit()?;
    Ok(())
}

pub fn upsert_files(connection: &mut Connection, files: &[FileRecord]) -> Result<()> {
    if files.is_empty() {
        return Ok(());
    }

    let transaction = connection.transaction()?;
    {
        let mut statement = transaction.prepare(
            r#"
            INSERT INTO files (path, hash, size, indexed_at, complete)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(path) DO UPDATE SET
                hash = excluded.hash,
                size = excluded.size,
                indexed_at = excluded.indexed_at,
                complete = excluded.complete
            "#,
        )?;

        for file in files {
            statement.execute(params![
                file.path,
                file.hash,
                file.size,
                file.indexed_at,
                file.complete,
            ])?;
        }
    }
    transaction.commit()?;
    Ok(())
}

pub fn insert_chunks(connection: &mut Connection, chunks: &[ChunkRecord]) -> Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }

    let transaction = connection.transaction()?;
    {
        let mut statement = transaction.prepare(
            r#"
            INSERT INTO chunks (
                path,
                chunk_index,
                start_line,
                end_line,
                text,
                embedding,
                file_hash
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(path, file_hash, chunk_index) DO UPDATE SET
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                text = excluded.text,
                embedding = excluded.embedding
            "#,
        )?;

        for chunk in chunks {
            let bytes = encode_embedding(&chunk.embedding);
            statement.execute(params![
                chunk.path,
                chunk.chunk_index,
                chunk.start_line,
                chunk.end_line,
                chunk.text,
                bytes,
                chunk.file_hash,
            ])?;
        }
    }
    transaction.commit()?;
    Ok(())
}

pub fn load_chunks(connection: &Connection) -> Result<Vec<SearchRow>> {
    let mut statement = connection.prepare(
        r#"
        SELECT path, start_line, end_line, text, embedding
        FROM chunks
        ORDER BY path, chunk_index
        "#,
    )?;

    let rows = statement.query_map([], |row| {
        let embedding: Vec<u8> = row.get(4)?;
        Ok(SearchRow {
            path: row.get(0)?,
            start_line: row.get(1)?,
            end_line: row.get(2)?,
            text: row.get(3)?,
            embedding: decode_embedding(&embedding),
        })
    })?;

    let mut chunks = Vec::new();
    for row in rows {
        chunks.push(row?);
    }
    Ok(chunks)
}

pub fn load_stats(connection: &Connection) -> Result<IndexStats> {
    let (file_count, source_bytes, last_indexed_at) = connection.query_row(
        "SELECT COUNT(*), COALESCE(SUM(size), 0), MAX(indexed_at) FROM files",
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<i64>>(2)?,
            ))
        },
    )?;

    let (chunk_count, embedded_bytes) = connection.query_row(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(CAST(text AS BLOB))), 0) FROM chunks",
        [],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    )?;

    Ok(IndexStats {
        file_count,
        chunk_count,
        source_bytes,
        embedded_bytes,
        last_indexed_at,
    })
}
