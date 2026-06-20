use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

const HELP_EPILOGUE: &str = r#"
Examples:
  vigrep index
  vigrep index src
  vigrep search "authentication flow"
  vigrep search "sqlite hash reindexing" --top-k 5
  vigrep --backend ollama --base-url http://127.0.0.1:11434 --rerank-backend llama-cpp --rerank-base-url http://127.0.0.1:9090 search "oauth callback flow" --rerank
  vigrep search "oauth callback flow" --rerank --rerank-model bge-reranker-v2-m3
  vigrep stats
  vigrep endpoint
  vigrep completions zsh > _vigrep
  vigrep completions zsh --install

Config:
  Default config: ~/.config/vigrep/config.toml
  Config editor: nano by default, or VISUAL/EDITOR if set
  Chunk max chars: 2048 by default
    Default search results: 10 by default
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

    #[arg(
        long,
        global = true,
        value_name = "BACKEND",
        help = "Backend to use: llama-cpp, ollama, or vllm"
    )]
    pub backend: Option<String>,

    #[arg(
        long,
        global = true,
        value_name = "URL",
        help = "Override the backend base URL"
    )]
    pub base_url: Option<String>,

    #[arg(
        long,
        global = true,
        value_name = "URL",
        help = "Override the reranking base URL (defaults to --base-url)"
    )]
    pub rerank_base_url: Option<String>,

    #[arg(
        long,
        global = true,
        value_name = "BACKEND",
        help = "Override reranking backend: llama-cpp or ollama (defaults to --backend)"
    )]
    pub rerank_backend: Option<String>,

    #[arg(
        long,
        global = true,
        value_name = "MODEL",
        help = "Override the backend model name (Ollama / vLLM)"
    )]
    pub model: Option<String>,

    #[arg(
        long,
        global = true,
        value_name = "MODEL",
        help = "Override the backend reranking model name"
    )]
    pub rerank_model: Option<String>,

    #[arg(long, global = true, help = "Enable backend reranking during search")]
    pub rerank: bool,

    #[arg(
        long,
        global = true,
        value_name = "N",
        help = "Number of top candidates to rerank before truncating output"
    )]
    pub rerank_top_n: Option<usize>,

    #[arg(
        long,
        global = true,
        value_name = "N",
        help = "Override the number of concurrent embedding requests"
    )]
    pub concurrent_requests: Option<usize>,

    #[arg(
        long,
        global = true,
        help = "Print outbound and inbound HTTP request details"
    )]
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

        #[arg(
            long,
            help = "Install the completion script into the standard location for the selected shell"
        )]
        install: bool,
    },
    #[command(about = "Edit the config file in nano")]
    Configure,
    #[command(about = "Interactively configure endpoints with model discovery")]
    Endpoint,
    #[command(about = "Index or reindex a directory")]
    Index {
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,
    },
    #[command(about = "Search the index with a semantic query")]
    Search {
        #[arg(value_name = "QUERY")]
        query: String,

        #[arg(short = 'k', long = "top-k", value_name = "N")]
        top_k: Option<usize>,

        #[arg(long = "min-score", default_value_t = 0.35, value_name = "SCORE")]
        min_score: f32,

        #[arg(short = 'f', long = "full", help = "Print the full retrieved text with newlines preserved")]
        full: bool,
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
