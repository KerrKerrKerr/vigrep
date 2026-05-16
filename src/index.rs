use anyhow::{Context, Result};
use futures::{stream, StreamExt};
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::backend::EmbeddingBackend;
use crate::config::RuntimeConfig;
use crate::db::{self, ChunkRecord, FileRecord};
use crate::text::{chunk_text, normalize_embedding, sha256_hex};

#[derive(Debug, Clone)]
struct PendingFile {
    path: String,
    hash: String,
    size: i64,
    text: String,
}

#[derive(Debug, Clone)]
struct PendingChunk {
    path: String,
    file_hash: String,
    base_chunk_index: i64,
    split_path: u32,
    start_line: i64,
    end_line: i64,
    text: String,
}

impl PendingChunk {
    fn chunk_index(&self) -> i64 {
        (self.base_chunk_index << 32) | i64::from(self.split_path)
    }
}

pub async fn run_index(runtime: &RuntimeConfig, scan_path: &Path) -> Result<()> {
    let current_dir = std::env::current_dir().context("Failed to resolve the current directory")?;
    let current_dir = current_dir
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize {}", current_dir.display()))?;
    let scan_root = scan_path
        .canonicalize()
        .with_context(|| format!("Failed to resolve {}", scan_path.display()))?;
    let db_path = current_dir.join(".vigrep.x");
    let connection = Rc::new(RefCell::new(db::open_database(&db_path)?));
    let existing_files = {
        let connection_ref = connection.borrow();
        db::load_known_files(&connection_ref)?
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    install_ctrl_c_handler(Arc::clone(&cancelled))?;

    let mut discovered_files: Vec<PendingFile> = Vec::new();
    let mut seen_paths: HashSet<String> = HashSet::new();
    let mut skipped_non_text = 0usize;

    let walker = WalkBuilder::new(&scan_root).standard_filters(true).build();

    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                eprintln!("Skipping unreadable path: {error}");
                continue;
            }
        };

        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        if should_skip_file(path) {
            continue;
        }

        let relative = path
            .strip_prefix(&current_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();

        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("Skipping {}: {error}", relative);
                continue;
            }
        };

        let hash = sha256_hex(&bytes);
        let text = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => {
                skipped_non_text += 1;
                continue;
            }
        };

        seen_paths.insert(relative.clone());

        let size = text.len() as i64;
        discovered_files.push(PendingFile {
            path: relative.clone(),
            hash: hash.clone(),
            size,
            text,
        });
    }

    let mut paths_to_clear = Vec::new();
    for path in existing_files.keys() {
        if !seen_paths.contains(path) {
            paths_to_clear.push(path.clone());
        }
    }

    for file in &discovered_files {
        if existing_files.get(&file.path).map(|state| state.hash.as_str()) != Some(file.hash.as_str()) {
            paths_to_clear.push(file.path.clone());
        }
    }

    paths_to_clear.sort();
    paths_to_clear.dedup();
    {
        let mut connection_ref = connection.borrow_mut();
        db::delete_paths(&mut connection_ref, &paths_to_clear)?;
    }

    let backend = EmbeddingBackend::new(runtime.backend, runtime.backend_profile.clone(), runtime.debug_http)?;

    let files_to_process: Vec<&PendingFile> = discovered_files
        .iter()
        .filter(|file| {
            match existing_files.get(&file.path) {
                Some(state) if state.hash == file.hash && state.complete => false,
                _ => true,
            }
        })
        .collect();
    let processed_file_count = files_to_process.len();

    let mut file_plans = Vec::new();
    let mut total_base_chunks = 0u64;

    for file in files_to_process {
        let existing_chunk_indexes = {
            let connection_ref = connection.borrow();
            db::load_chunk_indexes(&connection_ref, &file.path, &file.hash)?
        };

        let mut pending_chunks = Vec::new();
        for chunk in chunk_text(
            &file.text,
            runtime.chunk_lines,
            runtime.chunk_max_chars,
            runtime.chunk_overlap,
        ) {
            let chunk_index = chunk_index_key(chunk.chunk_index as i64, 1);
            if !existing_chunk_indexes.contains(&chunk_index) {
                pending_chunks.push(chunk);
            }
        }

        if pending_chunks.is_empty() {
            db::upsert_files(
                &mut connection.borrow_mut(),
                &[FileRecord {
                    path: file.path.clone(),
                    hash: file.hash.clone(),
                    size: file.size,
                    indexed_at: current_unix_timestamp()?,
                    complete: 1,
                }],
            )?;
            continue;
        }

        total_base_chunks += pending_chunks.len() as u64;
        file_plans.push((file, pending_chunks));
    }

    let pb = ProgressBar::new(total_base_chunks);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks {msg}")
            .context("Failed to configure the progress bar")?
            .progress_chars("=>-"),
    );

    let mut processed_chunks = 0usize;
    let mut interrupted = false;
    for (file, base_chunks) in file_plans {
        if cancelled.load(Ordering::Relaxed) {
            interrupted = true;
            break;
        }

        db::upsert_files(
            &mut connection.borrow_mut(),
            &[FileRecord {
                path: file.path.clone(),
                hash: file.hash.clone(),
                size: file.size,
                indexed_at: current_unix_timestamp()?,
                complete: 0,
            }],
        )?;

        let mut chunk_stream = stream::iter(base_chunks.into_iter().map(|chunk| {
            let backend = backend.clone();
            let connection = Rc::clone(&connection);
            let cancelled = Arc::clone(&cancelled);
            let file_path = file.path.clone();
            let file_hash = file.hash.clone();
            let base_chunk_index = chunk.chunk_index as i64;
            async move {
                let records = embed_chunk_with_retry(
                    &backend,
                    &connection,
                    &cancelled,
                    PendingChunk {
                        path: file_path,
                        file_hash,
                        base_chunk_index,
                        split_path: 1,
                        start_line: chunk.start_line as i64,
                        end_line: chunk.end_line as i64,
                        text: chunk.text,
                    },
                )
                .await?;
                Result::<usize>::Ok(records)
            }
        }))
        .buffer_unordered(runtime.concurrent_requests);

        while let Some(result) = chunk_stream.next().await {
            let inserted = result?;
            processed_chunks += inserted;
            pb.inc(1);
            if cancelled.load(Ordering::Relaxed) {
                interrupted = true;
                break;
            }
        }

        if interrupted {
            break;
        }

        db::upsert_files(
            &mut connection.borrow_mut(),
            &[FileRecord {
                path: file.path.clone(),
                hash: file.hash.clone(),
                size: file.size,
                indexed_at: current_unix_timestamp()?,
                complete: 1,
            }],
        )?;
    }

    pb.finish_and_clear();

    if interrupted {
        println!("Interrupted. Progress up to the last processed request was saved to the database.");
    }

    println!(
        "Indexed {} file(s), {} chunk(s) with {} backend. Skipped {} binary/non-UTF8 file(s).",
        processed_file_count,
        processed_chunks,
        backend.label(),
        skipped_non_text
    );

    if processed_chunks == 0 {
        println!("No text chunks needed reindexing.");
    } else {
        println!("Database: {}", db_path.display());
    }

    Ok(())
}

fn install_ctrl_c_handler(cancelled: Arc<AtomicBool>) -> Result<()> {
    ctrlc::set_handler(move || {
        cancelled.store(true, Ordering::Relaxed);
    })
    .context("Failed to install Ctrl-C handler")
}

fn should_skip_file(path: &Path) -> bool {
    matches!(path.file_name().and_then(|name| name.to_str()), Some(".vigrep.x") | Some(".gitignore"))
}

fn current_unix_timestamp() -> Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("System clock is before the Unix epoch")?
        .as_secs() as i64)
}

async fn embed_chunk_with_retry(
    backend: &EmbeddingBackend,
    connection: &Rc<RefCell<rusqlite::Connection>>,
    cancelled: &AtomicBool,
    chunk: PendingChunk,
) -> Result<usize> {
    let mut stack = vec![chunk];
    let mut inserted_records = 0usize;

    while let Some(chunk) = stack.pop() {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(inserted_records);
        }

        match backend.embed_one(&chunk.text).await {
            Ok(embedding) => {
                let chunk_index = chunk.chunk_index();
                let path = chunk.path.clone();
                let file_hash = chunk.file_hash.clone();
                let text = chunk.text.clone();

                let record = ChunkRecord {
                    path,
                    file_hash,
                    chunk_index,
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    text,
                    embedding: normalize_embedding(embedding),
                };

                {
                    let mut connection_ref = connection.borrow_mut();
                    db::insert_chunks(&mut connection_ref, &[record])?;
                }

                inserted_records += 1;
            }
            Err(error) if is_too_large_error(&error) => {
                let Some((left, right)) = split_pending_chunk(&chunk) else {
                    return Err(error).context(format!(
                        "Chunk {}:{}-{} is still too large after splitting",
                        chunk.path, chunk.start_line, chunk.end_line
                    ));
                };

                stack.push(right);
                stack.push(left);
            }
            Err(error) => return Err(error),
        }
    }

    Ok(inserted_records)
}

fn is_too_large_error(error: &anyhow::Error) -> bool {
    let message = error.to_string().to_lowercase();
    message.contains("too large to process")
        || message.contains("input too large")
        || message.contains("context length")
        || message.contains("maximum context")
}

fn chunk_index_key(base_chunk_index: i64, split_path: u32) -> i64 {
    (base_chunk_index << 32) | i64::from(split_path)
}

fn split_pending_chunk(chunk: &PendingChunk) -> Option<(PendingChunk, PendingChunk)> {
    let lines: Vec<&str> = chunk.text.lines().collect();
    if lines.is_empty() {
        return None;
    }

    if lines.len() == 1 {
        return split_single_line_chunk(chunk);
    }

    let midpoint = lines.len() / 2;
    let left_text = lines[..midpoint].join("\n");
    let right_text = lines[midpoint..].join("\n");

    let left_end = chunk.start_line + midpoint as i64 - 1;
    let right_start = left_end + 1;

    Some((
        PendingChunk {
            path: chunk.path.clone(),
            file_hash: chunk.file_hash.clone(),
            base_chunk_index: chunk.base_chunk_index,
            split_path: chunk.split_path << 1,
            start_line: chunk.start_line,
            end_line: left_end,
            text: left_text,
        },
        PendingChunk {
            path: chunk.path.clone(),
            file_hash: chunk.file_hash.clone(),
            base_chunk_index: chunk.base_chunk_index,
            split_path: (chunk.split_path << 1) | 1,
            start_line: right_start,
            end_line: chunk.end_line,
            text: right_text,
        },
    ))
}

fn split_single_line_chunk(chunk: &PendingChunk) -> Option<(PendingChunk, PendingChunk)> {
    let characters: Vec<char> = chunk.text.chars().collect();
    if characters.len() <= 1 {
        return None;
    }

    let midpoint = characters.len() / 2;
    let left_text: String = characters[..midpoint].iter().collect();
    let right_text: String = characters[midpoint..].iter().collect();

    Some((
        PendingChunk {
            path: chunk.path.clone(),
            file_hash: chunk.file_hash.clone(),
            base_chunk_index: chunk.base_chunk_index,
            split_path: chunk.split_path << 1,
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            text: left_text,
        },
        PendingChunk {
            path: chunk.path.clone(),
            file_hash: chunk.file_hash.clone(),
            base_chunk_index: chunk.base_chunk_index,
            split_path: (chunk.split_path << 1) | 1,
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            text: right_text,
        },
    ))
}
