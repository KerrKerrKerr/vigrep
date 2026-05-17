use anyhow::{Context, Result};
use owo_colors::OwoColorize;
use std::cmp::Ordering;

use crate::backend::{EmbeddingBackend, RerankResult};
use crate::config::RuntimeConfig;
use crate::db;
use crate::text::{cosine_similarity, normalize_embedding};

const DEFAULT_PREVIEW_LINES: usize = 10;

#[derive(Debug, Clone)]
struct SearchHit {
    path: String,
    start_line: i64,
    end_line: i64,
    semantic_score: f32,
    rerank_score: Option<f32>,
    text: String,
    snippet: String,
}

pub async fn run_search(
    runtime: &RuntimeConfig,
    query: &str,
    top_k: Option<usize>,
    min_score: f32,
    full: bool,
) -> Result<()> {
    let current_dir = std::env::current_dir().context("Failed to resolve the current directory")?;
    let db_path = current_dir.join(".vigrep.x");
    if !db_path.exists() {
        anyhow::bail!(
            "No .vigrep.x database found in {}. Run `vigrep index` first.",
            current_dir.display()
        );
    }

    let connection = db::open_database(&db_path)?;
    let embedding_backend = EmbeddingBackend::new(
        runtime.backend,
        runtime.backend_profile.clone(),
        runtime.debug_http,
    )?;
    let rerank_backend = if runtime.rerank_enabled {
        Some(EmbeddingBackend::new(
            runtime.rerank_backend,
            runtime.rerank_backend_profile.clone(),
            runtime.debug_http,
        )?)
    } else {
        None
    };

    let query_embedding = embedding_backend.embed_one(query).await?;
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
            semantic_score: score,
            rerank_score: None,
            text: chunk.text.clone(),
            snippet: chunk.text.clone(),
        });
    }

    hits.sort_by(|left, right| {
        right
            .semantic_score
            .partial_cmp(&left.semantic_score)
            .unwrap_or(Ordering::Equal)
    });

    if hits.is_empty() {
        println!("No matches at or above the 0.35 similarity floor.");
        return Ok(());
    }

    let effective_top_k = top_k.unwrap_or(runtime.default_top_k).max(1);

    let mut rerank_applied = false;
    if let Some(rerank_backend) = rerank_backend.as_ref() {
        match apply_reranking(
            rerank_backend,
            query,
            &mut hits,
            runtime.rerank_top_n,
            effective_top_k,
        )
        .await
        {
            Ok(applied) => rerank_applied = applied,
            Err(error) => {
                eprintln!("Reranking request failed, falling back to semantic ranking: {error}")
            }
        }
    }

    let limit = effective_top_k.min(hits.len());
    println!(
        "{} {}  {} {}",
        "Query:".bold().cyan(),
        query.bold(),
        "Embedding backend:".bold().cyan(),
        embedding_backend.label().bold()
    );
    if let Some(rerank_backend) = rerank_backend.as_ref() {
        let status = if rerank_applied {
            "applied".green().bold().to_string()
        } else {
            "unavailable".yellow().bold().to_string()
        };
        let rerank_model = rerank_backend.rerank_model_label().unwrap_or("n/a");
        println!(
            "{} {}  {} {}  {} {}",
            "Reranking:".bold().cyan(),
            status,
            "Backend:".bold().cyan(),
            rerank_backend.label().bold(),
            "Model:".bold().cyan(),
            rerank_model.bold()
        );
    }
    println!("{} {}", "Showing top".bold().cyan(), limit);

    for (index, hit) in hits.into_iter().take(limit).enumerate() {
        let score = hit.rerank_score.unwrap_or(hit.semantic_score);
        let score_text = format!("{:.4}", score);
        let score_style = if score >= 0.8 {
            score_text.green().bold().to_string()
        } else if score >= 0.6 {
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
        print_snippet(&hit.snippet, full);
    }

    Ok(())
}

fn print_snippet(text: &str, full: bool) {
    let mut lines = text.lines();
    let preview_limit = if full { usize::MAX } else { DEFAULT_PREVIEW_LINES };
    let mut printed = 0usize;

    while printed < preview_limit {
        match lines.next() {
            Some(line) => {
                println!("    {}", line.dimmed());
                printed += 1;
            }
            None => return,
        }
    }

    if !full && lines.next().is_some() {
        println!("    {}", "...".dimmed());
    }
}

async fn apply_reranking(
    backend: &EmbeddingBackend,
    query: &str,
    hits: &mut Vec<SearchHit>,
    rerank_top_n: usize,
    requested_top_k: usize,
) -> Result<bool> {
    if hits.len() < 2 {
        return Ok(false);
    }

    let candidate_count = rerank_top_n.max(requested_top_k.max(1)).min(hits.len());

    let mut candidates: Vec<SearchHit> = hits.drain(..candidate_count).collect();
    let documents: Vec<String> = candidates.iter().map(|hit| hit.text.clone()).collect();
    let reranked = backend.rerank(query, &documents, candidate_count).await?;
    reorder_hits_by_rerank(candidates.as_mut_slice(), &reranked);

    let mut reordered = candidates;
    reordered.append(hits);
    *hits = reordered;
    Ok(true)
}

fn reorder_hits_by_rerank(candidates: &mut [SearchHit], reranked: &[RerankResult]) {
    let mut slots: Vec<Option<SearchHit>> = candidates.iter().cloned().map(Some).collect();
    let mut reordered = Vec::with_capacity(candidates.len());

    for rerank in reranked {
        if rerank.index >= slots.len() {
            continue;
        }
        if let Some(mut hit) = slots[rerank.index].take() {
            hit.rerank_score = Some(rerank.score);
            reordered.push(hit);
        }
    }

    for hit in slots.into_iter().flatten() {
        reordered.push(hit);
    }

    for (target, source) in candidates.iter_mut().zip(reordered.into_iter()) {
        *target = source;
    }
}
