//! `jope`: the umbrella command line. Each verb delegates to a library crate's own clap
//! `Command` (no process is spawned); without a verb it prints the version.

use clap::{Parser, Subcommand};
use std::process::ExitCode;

fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[derive(Parser)]
#[command(name = "jope", version, about = "JOPE-Dime command-line interface")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Model bundles: fetch, verify, publish and locate content-addressed weights.
    Models(jope_models::cli::Cli),
    /// Dataset catalog: ingest, fetch, verify and link content-addressed datasets.
    Datasets(jope_datasets::cli::Cli),
}

fn main() -> ExitCode {
    match Cli::parse().cmd {
        None => {
            println!("jope {}", version());
            ExitCode::SUCCESS
        }
        Some(Cmd::Models(cli)) => jope_models::cli::exec(cli),
        Some(Cmd::Datasets(cli)) => jope_datasets::cli::exec(cli),
    }
}

#[cfg(test)]
mod tests {
    use super::{version, Cli};
    use clap::CommandFactory;

    #[test]
    fn version_is_semver() {
        let parts: Vec<_> = version().split('.').collect();
        assert_eq!(parts.len(), 3, "expected major.minor.patch");
        assert!(parts
            .iter()
            .all(|part| part.chars().all(|c| c.is_ascii_digit())));
    }

    #[test]
    fn umbrella_lists_models_and_datasets() {
        let cmd = Cli::command();
        let names: Vec<&str> = cmd.get_subcommands().map(|c| c.get_name()).collect();
        assert!(
            names.contains(&"models") && names.contains(&"datasets"),
            "{names:?}"
        );
        let models = cmd.find_subcommand("models").unwrap();
        let verbs: Vec<&str> = models.get_subcommands().map(|c| c.get_name()).collect();
        assert_eq!(verbs, ["fetch", "verify", "publish", "path"]);
        let datasets = cmd.find_subcommand("datasets").unwrap();
        assert!(datasets.find_subcommand("fetch").is_some());
        Cli::command().debug_assert();
    }
}
