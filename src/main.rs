use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use clap_complete::{generate, generate_to, shells};
use std::fs;
use std::path::PathBuf;

use vigrep::cli::{Cli, Commands, CompletionShell};
use vigrep::config::{AppConfig, RuntimeConfig};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = match cli.config.clone() {
        Some(path) => path,
        None => vigrep::config::default_config_path()?,
    };

    match &cli.command {
        Commands::Completions { shell, install } => {
            if *install {
                install_completions(*shell)?;
            } else {
                print_completions(*shell);
            }
        }
        Commands::Configure => {
            vigrep::config::ensure_config_exists(&config_path)?;
            vigrep::config::open_config_in_editor(&config_path)?;
        }
        Commands::Index { path } => {
            let app_config = AppConfig::load_or_create(&config_path)?;
            let runtime = RuntimeConfig::from_cli_and_config(&cli, app_config)?;
            vigrep::index::run_index(&runtime, path).await?
        }
        Commands::Search {
            query,
            top_k,
            min_score,
        } => {
            let app_config = AppConfig::load_or_create(&config_path)?;
            let runtime = RuntimeConfig::from_cli_and_config(&cli, app_config)?;
            vigrep::search::run_search(&runtime, query, *top_k, *min_score).await?
        }
        Commands::Stats => vigrep::stats::run_stats()?,
    }

    Ok(())
}

fn print_completions(shell: CompletionShell) {
    let mut command = Cli::command();
    let bin_name = command.get_name().to_string();

    match shell {
        CompletionShell::Bash => generate(shells::Bash, &mut command, bin_name, &mut std::io::stdout()),
        CompletionShell::Elvish => generate(shells::Elvish, &mut command, bin_name, &mut std::io::stdout()),
        CompletionShell::Fish => generate(shells::Fish, &mut command, bin_name, &mut std::io::stdout()),
        CompletionShell::PowerShell => {
            generate(shells::PowerShell, &mut command, bin_name, &mut std::io::stdout())
        }
        CompletionShell::Zsh => generate(shells::Zsh, &mut command, bin_name, &mut std::io::stdout()),
    }
}

fn install_completions(shell: CompletionShell) -> Result<()> {
    let mut command = Cli::command();
    let bin_name = command.get_name().to_string();
    let install_dir = completion_install_dir(shell)?;

    fs::create_dir_all(&install_dir)
        .with_context(|| format!("Failed to create completion directory at {}", install_dir.display()))?;

    let installed_path = match shell {
        CompletionShell::Bash => generate_to(shells::Bash, &mut command, bin_name, &install_dir)?,
        CompletionShell::Elvish => generate_to(shells::Elvish, &mut command, bin_name, &install_dir)?,
        CompletionShell::Fish => generate_to(shells::Fish, &mut command, bin_name, &install_dir)?,
        CompletionShell::PowerShell => {
            generate_to(shells::PowerShell, &mut command, bin_name, &install_dir)?
        }
        CompletionShell::Zsh => generate_to(shells::Zsh, &mut command, bin_name, &install_dir)?,
    };

    println!("Installed completion script to {}", installed_path.display());

    if matches!(shell, CompletionShell::Zsh) {
        println!(
            "If zsh does not pick it up automatically, add {} to your fpath and rerun compinit.",
            install_dir.display()
        );
    }

    Ok(())
}

fn completion_install_dir(shell: CompletionShell) -> Result<PathBuf> {
    let dir = match shell {
        CompletionShell::Bash => dirs::data_dir()
            .context("Unable to determine the user data directory")?
            .join("bash-completion/completions"),
        CompletionShell::Elvish => dirs::config_dir()
            .context("Unable to determine the user config directory")?
            .join("elvish/lib"),
        CompletionShell::Fish => dirs::config_dir()
            .context("Unable to determine the user config directory")?
            .join("fish/completions"),
        CompletionShell::PowerShell => dirs::data_dir()
            .context("Unable to determine the user data directory")?
            .join("powershell/Completions"),
        CompletionShell::Zsh => dirs::data_dir()
            .context("Unable to determine the user data directory")?
            .join("zsh/site-functions"),
    };

    Ok(dir)
}
