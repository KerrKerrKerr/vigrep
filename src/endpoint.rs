use anyhow::{Context, Result};
use owo_colors::OwoColorize;
use reqwest::Client;
use serde_json::Value;
use std::io::{self, Write};
use std::path::Path;

use crate::config::{AppConfig, BackendProfile};

pub async fn run_endpoint(config_path: &Path) -> Result<()> {
    let mut config = AppConfig::load_or_create(config_path)?;

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("Failed to build HTTP client")?;

    println!(
        "{}",
        "\nInteractive Endpoint Configurator\n".bold().cyan()
    );
    println!("This wizard probes each backend URL and lets you choose which models to use.");
    println!("Press Enter to accept defaults, or type a new value.\n");

    // ---- llama.cpp ----
    println!("{}", "=== llama.cpp ===".bold().yellow());
    let llama_url = prompt_url("Base URL", &config.llama_cpp.base_url)?;
    if !llama_url.is_empty() {
        config.llama_cpp.base_url = llama_url.clone();
        let models = discover_models_llamacpp(&client, &llama_url).await;
        select_model_embedding(&mut config.llama_cpp, &models)?;
        select_model_rerank(&mut config.llama_cpp, &models)?;
    }

    // ---- Ollama ----
    println!("{}", "=== Ollama ===".bold().yellow());
    let ollama_url = prompt_url("Base URL", &config.ollama.base_url)?;
    if !ollama_url.is_empty() {
        config.ollama.base_url = ollama_url.clone();
        let models = discover_models_ollama(&client, &ollama_url).await;
        select_model_embedding(&mut config.ollama, &models)?;
        select_model_rerank(&mut config.ollama, &models)?;
    }

    // ---- vLLM ----
    println!("{}", "=== vLLM ===".bold().yellow());
    let vllm_url = prompt_url("Base URL", &config.vllm.base_url)?;
    if !vllm_url.is_empty() {
        config.vllm.base_url = vllm_url.clone();
        let models = discover_models_vllm(&client, &vllm_url).await;
        select_model_embedding(&mut config.vllm, &models)?;
        select_model_rerank(&mut config.vllm, &models)?;
    }

    // ---- Choose active backend ----
    println!("{}", "=== Active Backend ===".bold().yellow());
    let backends = [
        ("llama-cpp".to_string(), &config.llama_cpp),
        ("ollama".to_string(), &config.ollama),
        ("vllm".to_string(), &config.vllm),
    ];
    config.backend = choose_backend(&backends, &config.backend)?;

    // ---- Rerank ----
    println!("{}", "=== Reranking ===".bold().yellow());
    config.rerank_enabled = prompt_bool("Enable reranking", config.rerank_enabled)?;
    if config.rerank_enabled {
        config.rerank_backend = choose_backend(&backends, &config.rerank_backend)?;
        config.rerank_top_n = prompt_usize("Rerank top N", config.rerank_top_n)?;

        let rerank_profile = match config.rerank_backend.as_str() {
            "llama-cpp" => &config.llama_cpp,
            "ollama" => &config.ollama,
            _ => &config.vllm,
        };
        if rerank_profile.rerank_model.as_deref().unwrap_or("").is_empty() {
            eprintln!("{}", "  Warning: no rerank model set for the chosen rerank backend.".yellow());
        }
    }

    // ---- Save ----
    let serialized = toml::to_string_pretty(&config)?;
    std::fs::write(config_path, serialized)
        .with_context(|| format!("Failed to write config to {}", config_path.display()))?;
    println!("\n{}", "Configuration saved!".green().bold());
    println!("Run `vigrep index` or `vigrep search \"query\"` to use it.");

    Ok(())
}

fn prompt(label: &str, default: &str) -> Result<String> {
    print!("{} [{}]: ", label, default);
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).context("Failed to read input")?;
    let trimmed = input.trim();
    Ok(if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    })
}

fn prompt_url(label: &str, default: &str) -> Result<String> {
    prompt(&format!("{label} URL (empty=skip)"), default)
}

fn prompt_bool(label: &str, default: bool) -> Result<bool> {
    let default_str = if default { "yes" } else { "no" };
    let answer = prompt(&format!("{label} (yes/no)"), default_str)?;
    Ok(answer == "yes" || answer == "y")
}

fn prompt_usize(label: &str, default: usize) -> Result<usize> {
    let answer = prompt(label, &default.to_string())?;
    Ok(answer.parse::<usize>().unwrap_or(default).max(1))
}

async fn discover_models_ollama(client: &Client, base_url: &str) -> Vec<String> {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let response = match client.get(&url).send().await {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    if !response.status().is_success() {
        return Vec::new();
    }
    let body: Value = match response.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    body.get("models")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.get("name").and_then(Value::as_str))
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

async fn discover_models_vllm(client: &Client, base_url: &str) -> Vec<String> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let response = match client.get(&url).send().await {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    if !response.status().is_success() {
        return Vec::new();
    }
    let body: Value = match response.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    body.get("data")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.get("id").and_then(Value::as_str))
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

async fn discover_models_llamacpp(client: &Client, base_url: &str) -> Vec<String> {
    // Try /health endpoint for slot info
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    let response = match client.get(&url).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => return Vec::new(),
    };
    let body: Value = match response.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    // llama.cpp returns model slug in /health
    if let Some(model) = body.get("model").and_then(Value::as_str) {
        if !model.is_empty() {
            return vec![model.to_string()];
        }
    }
    // Also check /slots
    let url = format!("{}/slots", base_url.trim_end_matches('/'));
    let response = match client.get(&url).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => return Vec::new(),
    };
    let body: Value = match response.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    body.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.get("model").and_then(Value::as_str))
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn print_models(models: &[String]) {
    if models.is_empty() {
        println!("  {}", "(No models auto-discovered.)".dimmed());
        return;
    }
    println!("  {}", "Discovered models:".bold());
    for (i, model) in models.iter().enumerate() {
        println!("    {}. {}", i + 1, model);
    }
}

fn select_from_list(models: &[String], label: &str) -> Option<String> {
    if models.is_empty() {
        return None;
    }
    print!("  Pick {} (1-{}) or Enter to skip: ", label, models.len());
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).ok();
    let idx = input.trim().parse::<usize>().ok()?;
    models.get(idx.wrapping_sub(1)).cloned()
}

fn select_model_embedding(profile: &mut BackendProfile, models: &[String]) -> Result<()> {
    println!("  {}", "Select embedding model:".bold());
    print_models(models);
    match select_from_list(models, "embedding model") {
        Some(model) => {
            profile.model = prompt("Use model for embedding", &model)?;
        }
        None => {
            profile.model = prompt("Enter embedding model name", &profile.model)?;
        }
    }
    Ok(())
}

fn select_model_rerank(profile: &mut BackendProfile, models: &[String]) -> Result<()> {
    println!("  {}", "Select rerank model (optional):".bold());
    print_models(models);
    if let Some(model) = select_from_list(models, "rerank model") {
        let answer = prompt("Use model for reranking (Enter=skip)", &model)?;
        if !answer.is_empty() {
            profile.rerank_model = Some(answer);
        }
    }
    Ok(())
}

fn choose_backend(options: &[(String, &BackendProfile)], current: &str) -> Result<String> {
    let mut valid = Vec::new();
    println!("  Available backends:");
    for (i, (name, profile)) in options.iter().enumerate() {
        if profile.base_url.is_empty() {
            continue;
        }
        valid.push((i + 1, name.clone()));
        let marker = if name == current { " *" } else { "  " };
        println!("  {marker} {}. {} ({})", i + 1, name, profile.base_url);
    }
    if valid.is_empty() {
        anyhow::bail!("No backend has a URL configured. Run `vigrep endpoint` first.");
    }
    if valid.len() == 1 {
        return Ok(valid[0].1.clone());
    }
    print!("  Active backend (1-{}): ", valid.len());
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).ok();
    let idx = input.trim().parse::<usize>().ok();
    if let Some(idx) = idx {
        for (num, name) in &valid {
            if *num == idx {
                return Ok(name.clone());
            }
        }
    }
    // fallback to current if still valid
    if !current.is_empty() && options.iter().any(|(n, _)| n == current) {
        Ok(current.to_string())
    } else {
        Ok(valid[0].1.clone())
    }
}