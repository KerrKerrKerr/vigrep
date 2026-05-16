use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cli::Cli;

pub const DEFAULT_CHUNK_MAX_CHARS: usize = 2048;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub backend: String,
    pub concurrent_requests: usize,
    pub chunk_lines: usize,
    pub chunk_max_chars: usize,
    pub chunk_overlap: usize,
    pub llama_cpp: BackendProfile,
    pub ollama: BackendProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackendProfile {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub backend: BackendChoice,
    pub backend_profile: BackendProfile,
    pub concurrent_requests: usize,
    pub chunk_lines: usize,
    pub chunk_max_chars: usize,
    pub chunk_overlap: usize,
    pub debug_http: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    LlamaCpp,
    Ollama,
}

impl BackendChoice {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "llama-cpp" | "llama.cpp" | "llamacpp" => Ok(Self::LlamaCpp),
            "ollama" => Ok(Self::Ollama),
            other => bail!("unsupported backend '{other}'. Use llama-cpp or ollama"),
        }
    }
}

impl Default for BackendProfile {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            model: String::new(),
            api_key: None,
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            backend: "llama-cpp".to_string(),
            concurrent_requests: 1,
            chunk_lines: 32,
            chunk_max_chars: DEFAULT_CHUNK_MAX_CHARS,
            chunk_overlap: 4,
            llama_cpp: BackendProfile {
                base_url: "http://127.0.0.1:8080".to_string(),
                model: "nomic-embed-text".to_string(),
                api_key: None,
            },
            ollama: BackendProfile {
                base_url: "http://127.0.0.1:11434".to_string(),
                model: "nomic-embed-text".to_string(),
                api_key: None,
            },
        }
    }
}

pub fn default_config_dir() -> Result<PathBuf> {
    dirs::config_dir()
        .context("Unable to determine the user config directory")
        .map(|dir| dir.join("vigrep"))
}

pub fn default_config_path() -> Result<PathBuf> {
    Ok(default_config_dir()?.join("config.toml"))
}

impl AppConfig {
    pub fn load_or_create(path: &Path) -> Result<Self> {
        if path.exists() {
            let raw = fs::read_to_string(path)
                .with_context(|| format!("Failed to read config file at {}", path.display()))?;
            let config: Self = toml::from_str(&raw)
                .with_context(|| format!("Failed to parse config file at {}", path.display()))?;
            Ok(config)
        } else {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create config directory at {}", parent.display())
                })?;
            }

            let config = Self::default();
            let serialized = toml::to_string_pretty(&config)?;
            fs::write(path, serialized)
                .with_context(|| format!("Failed to write default config to {}", path.display()))?;
            Ok(config)
        }
    }
}

pub fn ensure_config_exists(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create config directory at {}", parent.display()))?;
    }

    let config = AppConfig::default();
    let serialized = toml::to_string_pretty(&config)?;
    fs::write(path, serialized)
        .with_context(|| format!("Failed to write default config to {}", path.display()))?;
    Ok(())
}

pub fn open_config_in_editor(path: &Path) -> Result<()> {
    for editor in preferred_editors() {
        match Command::new(&editor).arg(path).status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => {
                bail!("Editor '{}' exited with status {}", editor.to_string_lossy(), status)
            }
            Err(err) if err.kind() == ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("Failed to launch editor '{}'", editor.to_string_lossy())
                })
            }
        }
    }

    bail!("Unable to find an editor. Set VISUAL or EDITOR, or install nano.")
}

fn preferred_editors() -> Vec<std::ffi::OsString> {
    let mut editors = Vec::new();

    for candidate in [env::var_os("VISUAL"), env::var_os("EDITOR")] {
        if let Some(value) = candidate {
            if !value.to_string_lossy().trim().is_empty() {
                editors.push(value);
            }
        }
    }

    editors.push(std::ffi::OsString::from("nano"));
    editors
}

impl RuntimeConfig {
    pub fn from_cli_and_config(cli: &Cli, config: AppConfig) -> Result<Self> {
        let backend_name = cli
            .backend
            .as_deref()
            .unwrap_or(config.backend.as_str());
        let backend = BackendChoice::parse(backend_name)?;

        let concurrent_requests = cli.concurrent_requests.unwrap_or(config.concurrent_requests).max(1);
        let chunk_lines = config.chunk_lines.max(1);
        let chunk_max_chars = config.chunk_max_chars.max(1);
        let chunk_overlap = config.chunk_overlap.min(chunk_lines.saturating_sub(1));

        let mut backend_profile = match backend {
            BackendChoice::LlamaCpp => config.llama_cpp.clone(),
            BackendChoice::Ollama => config.ollama.clone(),
        };

        if let Some(base_url) = &cli.base_url {
            backend_profile.base_url = base_url.clone();
        }

        if let Some(model) = &cli.model {
            backend_profile.model = model.clone();
        }

        if backend_profile.base_url.trim().is_empty() {
            bail!("The selected backend has no base URL configured");
        }

        if matches!(backend, BackendChoice::Ollama) && backend_profile.model.trim().is_empty() {
            bail!("The selected backend has no model configured");
        }

        Ok(Self {
            backend,
            backend_profile,
            concurrent_requests,
            chunk_lines,
            chunk_max_chars,
            chunk_overlap,
            debug_http: cli.debug_http,
        })
    }
}
