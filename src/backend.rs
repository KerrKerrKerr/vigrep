use anyhow::{anyhow, bail, Context, Result};
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use std::time::Duration;

use crate::config::{BackendChoice, BackendProfile};

#[derive(Debug, Clone)]
pub enum EmbeddingBackend {
    LlamaCpp(BackendClient),
    Ollama(BackendClient),
    Vllm(BackendClient),
}

#[derive(Debug, Clone)]
pub struct BackendClient {
    client: Client,
    profile: BackendProfile,
    debug_http: bool,
}

#[derive(Debug, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
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

#[derive(Debug, Serialize)]
struct OllamaChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    stream: bool,
    format: &'a str,
    messages: Vec<OllamaChatMessage<'a>>,
}

#[derive(Debug, Serialize)]
struct VllmEmbedRequest<'a> {
    input: &'a str,
    model: &'a str,
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

    fn normalize_url(url: &str) -> String {
        url.trim_end_matches('/').to_string()
    }

    fn normalize_embed_base_url(&self) -> String {
        Self::normalize_url(self.profile.base_url.trim())
    }

    fn normalize_rerank_base_url(&self) -> String {
        let rerank_url = self
            .profile
            .rerank_base_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(self.profile.base_url.as_str());
        Self::normalize_url(rerank_url.trim())
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
        eprintln!(
            "[vigrep:http]    request body: {}",
            Self::preview(body, 4096)
        );
    }

    fn log_response(&self, label: &str, url: &str, status: reqwest::StatusCode, body: &str) {
        if !self.debug_http {
            return;
        }
        eprintln!("[vigrep:http] <- {label} {url} [{status}]");
        eprintln!(
            "[vigrep:http]    response body: {}",
            Self::preview(body, 4096)
        );
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
        vec![self.normalize_embed_base_url()]
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

    fn parse_index(value: &Value) -> Option<usize> {
        value
            .get("index")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .or_else(|| {
                value
                    .get("document_index")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
            })
            .or_else(|| {
                value
                    .get("doc_index")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
            })
            .or_else(|| {
                value
                    .get("input_index")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
            })
    }

    fn parse_score(value: &Value) -> Option<f32> {
        value
            .get("relevance_score")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .or_else(|| {
                value
                    .get("score")
                    .and_then(Value::as_f64)
                    .map(|value| value as f32)
            })
            .or_else(|| {
                value
                    .get("similarity")
                    .and_then(Value::as_f64)
                    .map(|value| value as f32)
            })
    }

    fn parse_rerank_items(items: &[Value]) -> Result<Vec<RerankResult>> {
        let mut parsed = Vec::new();
        for item in items {
            let Some(index) = Self::parse_index(item) else {
                continue;
            };
            let Some(score) = Self::parse_score(item) else {
                continue;
            };
            parsed.push(RerankResult { index, score });
        }

        if parsed.is_empty() {
            bail!("rerank response did not include index/score pairs");
        }

        parsed.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(parsed)
    }

    fn parse_score_value(value: &Value) -> Option<f32> {
        if let Some(number) = value
            .get("score")
            .and_then(Value::as_f64)
            .or_else(|| value.get("relevance_score").and_then(Value::as_f64))
            .or_else(|| value.get("similarity").and_then(Value::as_f64))
        {
            return Some(number as f32);
        }

        value.as_f64().map(|number| number as f32).or_else(|| {
            value
                .as_str()
                .and_then(|text| text.trim().parse::<f32>().ok())
        })
    }

    fn extract_rerank_results(value: &Value) -> Result<Vec<RerankResult>> {
        if let Some(results) = value.get("results").and_then(Value::as_array) {
            return Self::parse_rerank_items(results);
        }

        if let Some(data) = value.get("data").and_then(Value::as_array) {
            return Self::parse_rerank_items(data);
        }

        if let Some(rerank) = value.get("rerank").and_then(Value::as_array) {
            return Self::parse_rerank_items(rerank);
        }

        if let Some(array) = value.as_array() {
            return Self::parse_rerank_items(array);
        }

        bail!("rerank response did not include a supported result array shape")
    }

    fn extract_ollama_chat_score(value: &Value) -> Result<f32> {
        if let Some(score) = Self::parse_score_value(value) {
            return Ok(score.clamp(0.0, 1.0));
        }

        if let Some(object) = value.as_object() {
            if let Some(score_value) = object.get("result").or_else(|| object.get("data")) {
                if let Some(score) = Self::parse_score_value(score_value) {
                    return Ok(score.clamp(0.0, 1.0));
                }
            }
        }

        bail!("chat rerank response did not include a numeric score")
    }

    fn effective_rerank_model(&self) -> Option<&str> {
        self.profile
            .rerank_model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                let model = self.profile.model.trim();
                if model.is_empty() {
                    None
                } else {
                    Some(model)
                }
            })
    }

    async fn post_for_rerank(
        &self,
        label: &str,
        endpoints: &[&str],
        payloads: &[Value],
    ) -> Result<Vec<RerankResult>> {
        let base_url = self.normalize_rerank_base_url();
        let mut errors = Vec::new();

        for endpoint in endpoints {
            let url = format!("{base_url}{endpoint}");
            for payload in payloads {
                let body = serde_json::to_string(payload)
                    .context("Failed to serialize rerank request body")?;
                self.log_request(label, &url, &body);

                let response = self
                    .apply_auth(self.client.post(url.clone()).json(payload))
                    .send()
                    .await
                    .with_context(|| {
                        format!("Failed to contact {label} rerank endpoint at {url}")
                    })?;

                let status = response.status();
                let response_body = response
                    .text()
                    .await
                    .with_context(|| format!("Failed to read {label} rerank response body"))?;
                self.log_response(label, &url, status, &response_body);

                if !status.is_success() {
                    errors.push(format!("{url} -> HTTP {status} body={response_body}"));
                    continue;
                }

                let parsed: Value = serde_json::from_str(&response_body)
                    .with_context(|| format!("Failed to parse {label} rerank response"))?;
                return Self::extract_rerank_results(&parsed);
            }
        }

        bail!(
            "{label} rerank request failed on all endpoint/payload combinations: {}",
            errors.join(" | ")
        );
    }

    async fn embed_llama_cpp(&self, input: &str) -> Result<Vec<f32>> {
        let request = LlamaEmbeddingRequest { content: input };
        let request_body =
            serde_json::to_string(&request).context("Failed to serialize llama.cpp request")?;

        if let Some(base_url) = self.llama_cpp_base_urls().into_iter().next() {
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
        let url = format!("{}/api/embed", self.normalize_embed_base_url());
        let request = OllamaEmbedRequest {
            model: &self.profile.model,
            input,
            truncate: true,
        };
        let request_body =
            serde_json::to_string(&request).context("Failed to serialize Ollama request")?;

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

        let parsed: Value =
            serde_json::from_str(&body).context("Failed to parse the Ollama embedding response")?;

        Self::extract_embedding(&parsed)
    }

    async fn embed_vllm(&self, input: &str) -> Result<Vec<f32>> {
        let url = format!("{}/v1/embeddings", self.normalize_embed_base_url());
        let request = VllmEmbedRequest {
            input,
            model: &self.profile.model,
        };
        let request_body =
            serde_json::to_string(&request).context("Failed to serialize vLLM request")?;

        self.log_request("vllm", &url, &request_body);

        let response = self
            .apply_auth(self.client.post(url.clone()).json(&request))
            .send()
            .await
            .context("Failed to contact the vLLM embedding endpoint")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read the vLLM embedding response body")?;
        self.log_response("vllm", &url, status, &body);

        if !status.is_success() {
            bail!("vLLM returned an error response: HTTP {status} body={body}");
        }

        let parsed: Value =
            serde_json::from_str(&body).context("Failed to parse the vLLM embedding response")?;

        Self::extract_embedding(&parsed)
    }

    async fn rerank_vllm(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<RerankResult>> {
        let model = self.effective_rerank_model().unwrap_or_default();
        let payloads = vec![
            json!({
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }),
            json!({
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }),
        ];
        self.post_for_rerank("vllm", &["/v1/rerank", "/rerank"], &payloads)
            .await
    }

    async fn rerank_ollama(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<RerankResult>> {
        let model = self
            .effective_rerank_model()
            .context("No reranking model configured for Ollama")?;
        let url = format!("{}/api/chat", self.normalize_rerank_base_url());
        let candidate_count = top_n.min(documents.len()).max(1);
        let mut reranked = Vec::with_capacity(candidate_count);

        for (index, document) in documents.iter().take(candidate_count).enumerate() {
            let prompt = format!(
                "You are a reranker.\nRate the relevance between the query and document from 0 to 1.\nReturn JSON only with this exact shape: {{\"score\": number}}.\n\nQuery:\n{query}\n\nDocument:\n{document}\n"
            );
            let request = OllamaChatRequest {
                model,
                stream: false,
                format: "json",
                messages: vec![OllamaChatMessage {
                    role: "user",
                    content: &prompt,
                }],
            };
            let request_body = serde_json::to_string(&request)
                .context("Failed to serialize Ollama chat request")?;
            self.log_request("ollama-rerank", &url, &request_body);

            let response = self
                .apply_auth(self.client.post(url.clone()).json(&request))
                .send()
                .await
                .context("Failed to contact the Ollama chat endpoint for reranking")?;

            let status = response.status();
            let body = response
                .text()
                .await
                .context("Failed to read the Ollama chat rerank response body")?;
            self.log_response("ollama-rerank", &url, status, &body);

            if !status.is_success() {
                bail!("Ollama rerank chat returned an error response: HTTP {status} body={body}");
            }

            let parsed: Value = serde_json::from_str(&body)
                .context("Failed to parse the Ollama chat rerank response")?;
            let content = parsed
                .get("message")
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str)
                .context("Ollama chat rerank response did not include message.content")?;
            let content_value: Value = serde_json::from_str(content.trim())
                .context("Ollama chat rerank message.content was not valid JSON")?;
            let score = Self::extract_ollama_chat_score(&content_value)?;

            reranked.push(RerankResult { index, score });
        }

        reranked.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(reranked)
    }

    async fn rerank_llama_cpp(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<RerankResult>> {
        let model = self.effective_rerank_model().unwrap_or_default();
        let payloads = vec![
            json!({
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }),
            json!({
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }),
            json!({
                "model": model,
                "query": query,
                "input": documents,
                "top_n": top_n,
            }),
        ];
        self.post_for_rerank(
            "llama.cpp",
            &["/rerank", "/v1/rerank", "/api/rerank"],
            &payloads,
        )
        .await
    }
}

impl EmbeddingBackend {
    pub fn new(choice: BackendChoice, profile: BackendProfile, debug_http: bool) -> Result<Self> {
        let client = BackendClient::new(profile, debug_http)?;
        Ok(match choice {
            BackendChoice::LlamaCpp => Self::LlamaCpp(client),
            BackendChoice::Ollama => Self::Ollama(client),
            BackendChoice::Vllm => Self::Vllm(client),
        })
    }

    pub async fn embed_one(&self, input: &str) -> Result<Vec<f32>> {
        match self {
            EmbeddingBackend::LlamaCpp(client) => client.embed_llama_cpp(input).await,
            EmbeddingBackend::Ollama(client) => client.embed_ollama(input).await,
            EmbeddingBackend::Vllm(client) => client.embed_vllm(input).await,
        }
    }

    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<RerankResult>> {
        match self {
            EmbeddingBackend::LlamaCpp(client) => {
                client.rerank_llama_cpp(query, documents, top_n).await
            }
            EmbeddingBackend::Ollama(client) => client.rerank_ollama(query, documents, top_n).await,
            EmbeddingBackend::Vllm(client) => client.rerank_vllm(query, documents, top_n).await,
        }
    }

    pub fn rerank_model_label(&self) -> Option<&str> {
        match self {
            EmbeddingBackend::LlamaCpp(client) => client.effective_rerank_model(),
            EmbeddingBackend::Ollama(client) => client.effective_rerank_model(),
            EmbeddingBackend::Vllm(client) => client.effective_rerank_model(),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            EmbeddingBackend::LlamaCpp(_) => "llama.cpp",
            EmbeddingBackend::Ollama(_) => "ollama",
            EmbeddingBackend::Vllm(_) => "vllm",
        }
    }
}
