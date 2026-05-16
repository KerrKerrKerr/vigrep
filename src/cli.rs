use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

const HELP_EPILOGUE: &str = r#"
Examples:
  vigrep index
  vigrep index src
  vigrep search "authentication flow"
  vigrep search "sqlite hash reindexing" --top-k 5
    vigrep stats
    vigrep completions zsh > _vigrep
        vigrep completions zsh --install

Config:
  Default config: ~/.config/vigrep/config.toml
    Config editor: nano by default, or VISUAL/EDITOR if set
        Chunk max chars: 2048 by default
  Index database:  .vigrep.x in the current directory
"#;

#[derive(Debug, Parser)]
#[command(
    name = "vigrep",
    version,
    about = "Vector semantic grep for local text files",
    long_about = "Index text files into a local SQLite database and search them with vector embeddings.",
    after_help = HELP_EPILOGUE,
    arg_required_else_help = true,
    subcommand_required = true,
    disable_help_subcommand = true
)]
pub struct Cli {
    #[arg(long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[arg(long, global = true, value_name = "BACKEND", help = "Backend to use: llama-cpp or ollama")]
    pub backend: Option<String>,

    #[arg(long, global = true, value_name = "URL", help = "Override the backend base URL")]
    pub base_url: Option<String>,

    #[arg(long, global = true, value_name = "MODEL", help = "Override the backend model name (Ollama)")]
    pub model: Option<String>,

    #[arg(long, global = true, value_name = "N", help = "Override the number of concurrent embedding requests")]
    pub concurrent_requests: Option<usize>,

    #[arg(long, global = true, help = "Print outbound and inbound HTTP request details")]
    pub debug_http: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(about = "Generate a shell completion script")]
    Completions {
        #[arg(value_name = "SHELL", value_enum)]
        shell: CompletionShell,

        #[arg(long, help = "Install the completion script into the standard location for the selected shell")]
        install: bool,
    },
    #[command(about = "Edit the config file in nano")]
    Configure,
    #[command(about = "Index or reindex a directory")]
    Index {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    #[command(about = "Search the index with a semantic query")]
    Search {
        #[arg(value_name = "QUERY")]
        query: String,

        #[arg(short = 'k', long = "top-k", default_value_t = 10, value_name = "N")]
        top_k: usize,

        #[arg(long = "min-score", default_value_t = 0.35, value_name = "SCORE")]
        min_score: f32,
    },
    #[command(about = "Show how much has been indexed")]
    Stats,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CompletionShell {
    Bash,
    Elvish,
    Fish,
    PowerShell,
    Zsh,
}
