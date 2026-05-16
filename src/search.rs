use anyhow::{Context, Result};
use owo_colors::OwoColorize;
use std::cmp::Ordering;

use crate::backend::EmbeddingBackend;
use crate::config::RuntimeConfig;
use crate::db;
use crate::text::{cosine_similarity, normalize_embedding, preview};

#[derive(Debug, Clone)]
struct SearchHit {
    path: String,
    start_line: i64,
    end_line: i64,
    score: f32,
    snippet: String,
}

pub async fn run_search(runtime: &RuntimeConfig, query: &str, top_k: usize, min_score: f32) -> Result<()> {
    let current_dir = std::env::current_dir().context("Failed to resolve the current directory")?;
    let db_path = current_dir.join(".vigrep.x");
    if !db_path.exists() {
        anyhow::bail!("No .vigrep.x database found in {}. Run `vigrep index` first.", current_dir.display());
    }

    let connection = db::open_database(&db_path)?;
    let backend = EmbeddingBackend::new(runtime.backend, runtime.backend_profile.clone(), runtime.debug_http)?;

    let query_embedding = backend.embed_one(query).await?;
    let query_embedding = normalize_embedding(query_embedding);
    let effective_min_score = min_score.max(0.35);

    let chunks = db::load_chunks(&connection)?;
    let mut hits: Vec<SearchHit> = Vec::new();

    for chunk in chunks {
        let score = cosine_similarity(&query_embedding, &chunk.embedding);
        if score < effective_min_score {
            continue;
        }

        hits.push(SearchHit {
            path: chunk.path,
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            score,
            snippet: preview(&chunk.text, 512),
        });
    }

    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });

    if hits.is_empty() {
        println!("No matches at or above the 0.35 similarity floor.");
        return Ok(());
    }

    let limit = top_k.max(1).min(hits.len());
    println!(
        "{} {}  {} {}",
        "Query:".bold().cyan(),
        query.bold(),
        "Backend:".bold().cyan(),
        backend.label().bold()
    );
    println!("{} {}", "Showing top".bold().cyan(), limit);

    for (index, hit) in hits.into_iter().take(limit).enumerate() {
        let score_text = format!("{:.4}", hit.score);
        let score_style = if hit.score >= 0.8 {
            score_text.green().bold().to_string()
        } else if hit.score >= 0.6 {
            score_text.yellow().bold().to_string()
        } else {
            score_text.red().bold().to_string()
        };

        println!(
            "{} {}:{}  {}",
            format!("{:>2}.", index + 1).dimmed(),
            hit.path.cyan().bold(),
            format!("{}-{}", hit.start_line, hit.end_line).yellow(),
            score_style
        );
        println!("    {}", hit.snippet.dimmed());
    }

    Ok(())
}
