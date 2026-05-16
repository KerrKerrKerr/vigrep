use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct LineChunk {
    pub chunk_index: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{:02x}", byte)).collect()
}

pub fn chunk_text(content: &str, chunk_lines: usize, max_chars: usize, overlap: usize) -> Vec<LineChunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let lines_per_chunk = chunk_lines.max(1);
    let max_chars = max_chars.max(1);
    let overlap = overlap.min(lines_per_chunk.saturating_sub(1));

    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut chunk_index = 0usize;

    while start < lines.len() {
        let end = (start + lines_per_chunk).min(lines.len());
        let mut used_chars = 0usize;
        let mut split_single_line = false;
        let mut line_end = start;

        while line_end < end {
            let line = lines[line_end];
            let line_chars = line.chars().count();
            let line_cost = if line_end == start { line_chars } else { line_chars + 1 };

            if used_chars == 0 && line_chars > max_chars {
                for fragment in split_text_into_chunks(line, max_chars) {
                    chunks.push(LineChunk {
                        chunk_index,
                        start_line: start + 1,
                        end_line: start + 1,
                        text: fragment,
                    });
                    chunk_index += 1;
                }
                split_single_line = true;
                break;
            }

            if used_chars > 0 && used_chars + line_cost > max_chars {
                break;
            }

            used_chars += line_cost;
            line_end += 1;
        }

        if split_single_line {
            start += 1;
            continue;
        }

        if line_end == start {
            start += 1;
            continue;
        }

        chunks.push(LineChunk {
            chunk_index,
            start_line: start + 1,
            end_line: line_end,
            text: lines[start..line_end].join("\n"),
        });

        if line_end == lines.len() {
            break;
        }

        let chunk_len = line_end - start;
        let next_start = if chunk_len > overlap {
            line_end - overlap
        } else {
            line_end
        };
        start = next_start;
        chunk_index += 1;
    }

    chunks
}

fn split_text_into_chunks(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0usize;

    for character in text.chars() {
        current.push(character);
        current_chars += 1;
        if current_chars == max_chars {
            chunks.push(current);
            current = String::new();
            current_chars = 0;
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

pub fn normalize_embedding(mut embedding: Vec<f32>) -> Vec<f32> {
    let norm = embedding.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

pub fn encode_embedding(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn preview(text: &str, max_chars: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= max_chars {
        normalized
    } else {
        let trimmed: String = normalized.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{trimmed}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_sparse_text_by_lines_first() {
        let content = (1..=40)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");

        let chunks = chunk_text(&content, 8, 2048, 0);

        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 8);
        assert_eq!(chunks[1].start_line, 9);
        assert_eq!(chunks[1].end_line, 16);
    }

    #[test]
    fn chunks_long_line_by_char_budget() {
        let content = "a".repeat(10_000);

        let chunks = chunk_text(&content, 32, 2048, 0);

        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.text.chars().count() <= 2048));
        assert!(chunks.iter().all(|chunk| chunk.start_line == 1 && chunk.end_line == 1));
    }
}
