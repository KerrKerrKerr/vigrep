use anyhow::{Context, Result, bail};
use owo_colors::OwoColorize;

use crate::db;

fn format_bytes(bytes: i64) -> String {
    let mut value = bytes as f64;
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut unit_index = 0usize;

    while value >= 1024.0 && unit_index < units.len() - 1 {
        value /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, units[unit_index])
    } else {
        format!("{value:.1} {}", units[unit_index])
    }
}

pub fn run_stats() -> Result<()> {
    let current_dir = std::env::current_dir().context("Failed to resolve the current directory")?;
    let db_path = current_dir.join(".vigrep.x");

    if !db_path.exists() {
        bail!(
            "No .vigrep.x database found in {}. Run `vigrep index` first.",
            current_dir.display()
        );
    }

    let connection = db::open_database(&db_path)?;
    let stats = db::load_stats(&connection)?;

    println!("{} {}", "Database:".cyan().bold(), db_path.display());
    println!("{} {}", "Files indexed:".cyan().bold(), stats.file_count.to_string().green().bold());
    println!("{} {}", "Chunks indexed:".cyan().bold(), stats.chunk_count.to_string().green().bold());
    println!("{} {}", "Source bytes:".cyan().bold(), format_bytes(stats.source_bytes).yellow().bold());
    println!("{} {}", "Embedded bytes:".cyan().bold(), format_bytes(stats.embedded_bytes).yellow().bold());

    match stats.last_indexed_at {
        Some(value) => println!("{} {}", "Last indexed (unix):".cyan().bold(), value.to_string().magenta().bold()),
        None => println!("{} {}", "Last indexed:".cyan().bold(), "n/a".dimmed()),
    }

    Ok(())
}