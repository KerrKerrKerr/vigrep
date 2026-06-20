pub mod backend;
pub mod cli;
pub mod config;
pub mod db;
pub mod endpoint;
pub mod index;
pub mod search;
pub mod stats;
pub mod text;

pub type Result<T> = anyhow::Result<T>;
