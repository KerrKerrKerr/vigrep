# vigrep

`vigrep` is a local semantic grep tool for text/code repositories. It indexes files into a local SQLite database and retrieves relevant snippets using embedding similarity.

It aims to speed up onboarding on large projects with inredible ease of use. It also works great with Local AI agents.

Ollama and llama.cpp backends are the only ones supported due to the fact I only use them for inference of embedding models. You can request more backends if needed.

I personally use ollama for embeddings for comatibiliry with other infrastructure and llama.cpp for rerankings as ollama doesn't quite support it.

## What it does

- Indexes text files under a directory into `.vigrep.x` (SQLite + normalized embeddings).
- Skips binary/non-UTF-8 files automatically.
- Re-indexes incrementally by content hash (only changed files/chunks are re-embedded).
- Supports local embedding backends:
  - **llama.cpp** (`/embedding`)
  - **Ollama** (`/api/embed`)
- Optional backend reranking pass for top semantic candidates.
- Shows file path, line range, score, and snippet for search matches.

## Requirements

- Rust (for building from source)
- A running embedding backend:
  - llama.cpp server, or
  - Ollama with an embedding model available (default: `nomic-embed-text`)

## Install

Install the latest release binary to `~/.local/bin`:

```bash
curl -fsSL https://raw.githubusercontent.com/KerrKerrKerr/vigrep/master/install.sh | bash
```

Installer support is currently **Linux x86_64** only. Tested on Cachy OS.

### Install options
- `VIGREP_REPO`
- `VIGREP_VERSION`
- `VIGREP_INSTALL_DIR`
- `VIGREP_SOURCE_BRANCH` (`master` or `main`)

### Deinstallation is just removing executable at ~/.local/bin


## Build from source

```bash
cargo build --release
```

Binary path:

```bash
./target/release/vigrep
```

## Quick start

1. Create/edit config (first run writes `~/.config/vigrep/config.toml`):

   ```bash
   vigrep configure
   ```

2. Index the current directory:

   ```bash
   vigrep index
   ```

3. Search semantically:

   ```bash
   vigrep search "authentication flow"
   ```

4. Inspect index stats:

   ```bash
   vigrep stats
   ```

## Config Tutorial

Run this once to create or open your config file:

```bash
vigrep configure
```

The config file lives at `~/.config/vigrep/config.toml` by default. It is safe to edit directly after it has been created.

The main settings you will usually change are:

- `backend`: `llama-cpp` or `ollama`
- `base_url`: backend server URL
- `model`: embedding model name for the selected backend
- `rerank`: enable or disable reranking
- `rerank_backend`: separate backend for reranking
- `rerank_base_url`: separate endpoint for reranking
- `chunk_lines`: maximum number of lines per indexed chunk
- `chunk_max_chars`: hard character cap per chunk, default `2048`
- `chunk_overlap`: how many lines overlap between chunks
- `default_top_k`: default number of search results when `--top-k` is not passed
- `concurrent_requests`: embedding request concurrency during indexing
- `rerank_top_n`: how many candidates are reranked before truncating output


## Commands

```text
vigrep completions <shell> [--install]
vigrep configure
vigrep index [path]
vigrep search <query> [--top-k N] [--min-score SCORE]
vigrep stats
```

Global flags:

- `--config <FILE>`: use a custom config path
- `--backend <BACKEND>`: `llama-cpp` or `ollama`
- `--rerank-backend <BACKEND>`: rerank backend (`llama-cpp` or `ollama`, defaults to `--backend`)
- `--base-url <URL>`: override backend base URL
- `--rerank-base-url <URL>`: override rerank base URL (defaults to `--base-url`)
- `--model <MODEL>`: override model (used by Ollama)
- `--rerank`: enable backend reranking during search
- `--rerank-model <MODEL>`: override reranking model
- `--rerank-top-n <N>`: number of semantic candidates to rerank
- `--concurrent-requests <N>`: embedding request concurrency during indexing
- `--debug-http`: print request/response debug details

## Configuration

Default config file: `~/.config/vigrep/config.toml`

Default values:

```toml
backend = "llama-cpp"
rerank_backend = ""
concurrent_requests = 1
chunk_lines = 32
chunk_max_chars = 2048
chunk_overlap = 4
rerank_enabled = false
rerank_top_n = 40

[llama_cpp]
base_url = "http://127.0.0.1:8080"
rerank_base_url = ""
model = "nomic-embed-text"
rerank_model = ""
api_key = ""

[ollama]
base_url = "http://127.0.0.1:11434"
rerank_base_url = ""
model = "nomic-embed-text"
rerank_model = ""
api_key = ""
```

Notes:

- Search enforces a minimum similarity floor of `0.35`.
- You can mix backends (for example: Ollama embeddings + llama.cpp reranking) with `rerank_backend`.
- You can run embedding and reranking on different endpoints by setting `rerank_base_url`.
- Ollama reranking is performed through `/api/chat` by asking the reranker model to emit `{"score": ...}` per query-document pair.
- If reranking fails, search falls back to semantic ranking and prints a warning.
- `.vigrep.x` is always created in the current working directory.
- If indexing is interrupted, completed progress is preserved.

## Shell completions

Print completion script:

```bash
vigrep completions zsh > _vigrep
```

Install to standard shell location:

```bash
vigrep completions zsh --install
```
