use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use jope_cli::{encode_dataset, generate_fixtures, resolve_root, verify_inputs, Verification};
use std::{path::PathBuf, process::ExitCode};

#[derive(Parser)]
#[command(
    name = "jope",
    version,
    about = "JOPE-Dime offline input and CPU encode"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Verify the ten synthetic fixtures against the pinned deterministic generator.
    VerifyInputs {
        /// Explicitly select offline qualification; no network access is performed.
        #[arg(long)]
        offline: bool,
        /// Emit one machine-readable result, including on verification failure.
        #[arg(long)]
        json: bool,
        /// Directory containing catalog.json; otherwise resolve at runtime.
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Create the versioned synthetic corpus in a new output directory.
    GenerateFixtures {
        #[arg(long)]
        output: PathBuf,
        /// Check an existing directory for exact deterministic regeneration instead.
        #[arg(long)]
        check: bool,
        #[arg(long)]
        json: bool,
    },
    /// Encode one admitted dataset with CPU Hyperion. Needs HYPERION_V7_WEIGHTS; does not download.
    Encode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
}

fn execute(command: &Command) -> Result<Verification> {
    match command {
        Command::VerifyInputs { offline, root, .. } => {
            ensure!(
                *offline,
                "select --offline for synthetic input qualification"
            );
            verify_inputs(&resolve_root(root.clone())?)
        }
        Command::GenerateFixtures { output, check, .. } => {
            if !check {
                generate_fixtures(output)?;
            }
            verify_inputs(output)
        }
        Command::Encode { .. } => unreachable!("encode is dispatched separately"),
    }
}

fn fail(json: bool, schema: &str, error: anyhow::Error) -> ExitCode {
    if json {
        println!(
            "{}",
            serde_json::json!({"schema": schema, "ok": false, "error": format!("{error:#}")})
        );
    } else {
        eprintln!("jope: {error:#}");
    }
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let Some(command) = Cli::parse().command else {
        println!("jope {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    };
    if let Command::Encode { input, json } = &command {
        return match encode_dataset(input) {
            Ok(report) => {
                if *json {
                    println!(
                        "{}",
                        serde_json::to_string(&report).expect("serializable encode report")
                    );
                } else {
                    println!(
                        "Encoded {} ({} train rows → {} cols).",
                        report.dataset_id, report.train_rows, report.output_cols
                    );
                }
                ExitCode::SUCCESS
            }
            Err(error) => fail(*json, "jope.encode.v1", error),
        };
    }
    let json = match &command {
        Command::VerifyInputs { json, .. } | Command::GenerateFixtures { json, .. } => *json,
        Command::Encode { .. } => unreachable!("encode returned above"),
    };
    match execute(&command) {
        Ok(report) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string(&report).expect("serializable verification")
                );
            } else {
                println!(
                    "Verified {} synthetic datasets, {} bytes, offline.",
                    report.dataset_count, report.bytes
                );
            }
            ExitCode::SUCCESS
        }
        Err(error) => fail(json, "jope.input-verification.v1", error),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_semver() {
        let parts: Vec<_> = env!("CARGO_PKG_VERSION").split('.').collect();
        assert_eq!(parts.len(), 3, "expected major.minor.patch");
        assert!(parts
            .iter()
            .all(|part| part.chars().all(|character| character.is_ascii_digit())));
    }
}
