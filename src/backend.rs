use anyhow::{anyhow, bail, Context, Result};
use reqwest::Client;
use serde::Serialize;
use serde_json::Value;
use std::time::Duration;

use crate::config::{BackendChoice, BackendProfile};

#[derive(Debug, Clone)]
pub enum EmbeddingBackend {
    LlamaCpp(BackendClient),
    Ollama(BackendClient),
}

#[derive(Debug, Clone)]
pub struct BackendClient {
    client: Client,
    profile: BackendProfile,
    debug_http: bool,
}

#[derive(Debug, Serialize)]
struct LlamaEmbeddingRequest<'a> {
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
    truncate: bool,
}

impl BackendClient {
    pub fn new(profile: BackendProfile, debug_http: bool) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            client,
            profile,
            debug_http,
        })
    }

    fn normalize_base_url(&self) -> String {
        self.profile.base_url.trim_end_matches('/').to_string()
    }

    fn apply_auth(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self
            .profile
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    fn log_request(&self, label: &str, url: &str, body: &str) {
        if !self.debug_http {
            return;
        }
        eprintln!("[vigrep:http] -> {label} {url}");
        eprintln!("[vigrep:http]    request body: {}", Self::preview(body, 4096));
    }

    fn log_response(&self, label: &str, url: &str, status: reqwest::StatusCode, body: &str) {
        if !self.debug_http {
            return;
        }
        eprintln!("[vigrep:http] <- {label} {url} [{status}]");
        eprintln!("[vigrep:http]    response body: {}", Self::preview(body, 4096));
    }

    fn preview(value: &str, max_chars: usize) -> String {
        let char_count = value.chars().count();
        if char_count <= max_chars {
            value.to_string()
        } else {
            let prefix: String = value.chars().take(max_chars).collect();
            format!("{prefix}… <trimmed {char_count} chars to {max_chars}>")
        }
    }

    fn llama_cpp_base_urls(&self) -> Vec<String> {
        vec![self.normalize_base_url()]
    }

    fn parse_embedding_array(values: &[Value]) -> Result<Vec<f32>> {
        values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|number| number as f32)
                    .ok_or_else(|| anyhow!("embedding response contained a non-numeric value"))
            })
            .collect()
    }

    fn parse_embedding_value(value: &Value) -> Result<Vec<f32>> {
        if let Some(array) = value.as_array() {
            if array.is_empty() {
                bail!("embedding response contained an empty array");
            }

            if array[0].is_object() {
                return Self::parse_embedding_value(&array[0]);
            }

            if array[0].is_array() {
                return Self::parse_embedding_value(&array[0]);
            }

            return Self::parse_embedding_array(array);
        }

        if let Some(object) = value.as_object() {
            if let Some(embedding) = object.get("embedding") {
                return Self::parse_embedding_value(embedding);
            }

            if let Some(embeddings) = object.get("embeddings") {
                return Self::parse_embedding_value(embeddings);
            }

            if let Some(data) = object.get("data") {
                return Self::parse_embedding_value(data);
            }
        }

        if let Some(number) = value.as_f64() {
            return Ok(vec![number as f32]);
        }

        bail!("embedding response did not contain an embedding array")
    }

    fn extract_embedding(value: &Value) -> Result<Vec<f32>> {
        if let Some(array) = value.get("embedding") {
            return Self::parse_embedding_value(array);
        }

        if let Some(array) = value.get("embeddings").and_then(Value::as_array) {
            if let Some(first) = array.first() {
                return Self::parse_embedding_value(first);
            }
        }

        if let Some(data) = value.get("data").and_then(Value::as_array) {
            if let Some(first) = data.first() {
                if let Some(array) = first.get("embedding") {
                    return Self::parse_embedding_value(array);
                }
            }
        }

        Self::parse_embedding_value(value)
    }

    async fn embed_llama_cpp(&self, input: &str) -> Result<Vec<f32>> {
        let request = LlamaEmbeddingRequest { content: input };
        let request_body = serde_json::to_string(&request).context("Failed to serialize llama.cpp request")?;

        for base_url in self.llama_cpp_base_urls() {
            let url = format!("{base_url}/embedding");
            self.log_request("llama.cpp", &url, &request_body);

            let response = self
                .apply_auth(self.client.post(url.clone()).json(&request))
                .send()
                .await
                .context("Failed to contact the llama.cpp embedding endpoint")?;

            let status = response.status();
            let body = response
                .text()
                .await
                .context("Failed to read the llama.cpp embedding response body")?;
            self.log_response("llama.cpp", &url, status, &body);

            if !status.is_success() {
                bail!("llama.cpp returned an error response: HTTP {status} body={body}");
            }

            let parsed: Value = serde_json::from_str(&body)
                .context("Failed to parse the llama.cpp embedding response")?;

            return Self::extract_embedding(&parsed);
        }

        bail!("Failed to contact the llama.cpp embedding endpoint")
    }

    async fn embed_ollama(&self, input: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embed", self.normalize_base_url());
        let request = OllamaEmbedRequest {
            model: &self.profile.model,
            input,
            truncate: true,
        };
        let request_body = serde_json::to_string(&request).context("Failed to serialize Ollama request")?;

        self.log_request("ollama", &url, &request_body);

        let response = self
            .apply_auth(self.client.post(url.clone()).json(&request))
            .send()
            .await
            .context("Failed to contact the Ollama embedding endpoint")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read the Ollama embedding response body")?;
        self.log_response("ollama", &url, status, &body);

        if !status.is_success() {
            bail!("Ollama returned an error response: HTTP {status} body={body}");
        }

        let parsed: Value = serde_json::from_str(&body)
            .context("Failed to parse the Ollama embedding response")?;

        Self::extract_embedding(&parsed)
    }
}

impl EmbeddingBackend {
    pub fn new(choice: BackendChoice, profile: BackendProfile, debug_http: bool) -> Result<Self> {
        let client = BackendClient::new(profile, debug_http)?;
        Ok(match choice {
            BackendChoice::LlamaCpp => Self::LlamaCpp(client),
            BackendChoice::Ollama => Self::Ollama(client),
        })
    }

    pub async fn embed_one(&self, input: &str) -> Result<Vec<f32>> {
        match self {
            EmbeddingBackend::LlamaCpp(client) => client.embed_llama_cpp(input).await,
            EmbeddingBackend::Ollama(client) => client.embed_ollama(input).await,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            EmbeddingBackend::LlamaCpp(_) => "llama.cpp",
            EmbeddingBackend::Ollama(_) => "ollama",
        }
    }
}
