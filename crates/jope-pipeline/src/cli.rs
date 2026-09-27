//! `jope pipeline …`: the same `Cli`/`Cmd`/`exec`/`run` shape the other
//! subcommand crates expose, so the umbrella delegates to it in process.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{path::PathBuf, process::ExitCode};

use crate::{
    emit::Language,
    receipt,
    run::{Policy, RunOptions},
    search::Budget,
};

#[derive(Parser)]
#[command(
    name = "jope-pipeline",
    version,
    about = "JOPE-Dime product pipeline: dataset, prepared cuts, CPU GP search, emitted code and a receipt"
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Run dataset -> prepare -> search -> select -> emit and write the receipt.
    Run {
        /// A 16-hex catalog id resolved in the local cache, or a dataset directory.
        #[arg(long)]
        dataset: String,
        /// Where the preparation, searches, emitted sources and receipt are written.
        #[arg(long)]
        output: PathBuf,
        /// Total per-cut budget across search and comparison (1..=1000 attempts / 1..=600 active seconds).
        /// Multiple cuts reserve half for comparison; an in-flight fit may overrun time.
        #[arg(long, default_value = "evaluations=64,seconds=300")]
        budget: String,
        /// Fold seed for the cut preparation.
        #[arg(long, default_value_t = crate::DEFAULT_SEED)]
        seed: u64,
        /// Seed of the search's random streams.
        #[arg(long, default_value_t = crate::DEFAULT_SEED)]
        rng_seed: u64,
        /// How many of the fold partition's cuts to prepare and search (1..=5).
        #[arg(long, default_value_t = 1)]
        cuts: usize,
        /// Languages to emit: rust, python, cpp17, cpp20, js, or all.
        #[arg(long, default_value = "rust,python,cpp17,cpp20,js")]
        emit: String,
        /// The proposal stage: auto and skip record a skip, require fails closed.
        #[arg(long, default_value = "auto")]
        policy: String,
        /// A jope.pipeline-seeds/v1 file: its genomes enter generation 0 after the bootstrap
        /// (canonicalised, deduplicated, at most 4) and are re-measured on this run's cuts.
        #[arg(long)]
        seed_genomes: Option<PathBuf>,
        /// A parent receipt: verified against its run's evidence, its selected genome is seeded
        /// first and its sha256 recorded as this run's parent.
        #[arg(long)]
        incumbent: Option<PathBuf>,
        /// Print the receipt as one JSON document.
        #[arg(long)]
        json: bool,
    },
    /// Validate a receipt against the committed jope.pipeline-receipt/v1, v2 or v3 schema.
    VerifyReceipt {
        #[arg(long)]
        receipt: PathBuf,
        /// Also re-derive the receipt from its run directory's evidence (v2 and v3).
        #[arg(long)]
        evidence: bool,
        #[arg(long)]
        json: bool,
    },
}

pub fn exec(cli: Cli) -> ExitCode {
    match run(cli) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}

pub fn run(cli: Cli) -> Result<ExitCode> {
    match cli.cmd {
        Cmd::Run {
            dataset,
            output,
            budget,
            seed,
            rng_seed,
            cuts,
            emit,
            policy,
            seed_genomes,
            incumbent,
            json,
        } => {
            let options = RunOptions {
                dataset,
                output,
                budget: Budget::parse(&budget)?,
                seed,
                rng_seed,
                cuts,
                languages: Language::parse_list(&emit)?,
                policy: Policy::parse(&policy).ok_or_else(|| {
                    anyhow::anyhow!("--policy must be auto, skip or require, got {policy:?}")
                })?,
                seed_genomes,
                incumbent,
            };
            let receipt = crate::run::run(&options)?;
            if json {
                println!("{}", serde_json::to_string(&receipt)?);
            } else {
                let measured: u64 = receipt
                    .search
                    .cuts
                    .iter()
                    .map(|c| c.successful_measurements)
                    .sum();
                let score = match &receipt.search.best {
                    Some(best) => format!(
                        "hold RMSE {:.6}, hold R2 {:.6} over {} cut(s)",
                        best.validation_rmse, best.validation_r2, best.cuts_measured
                    ),
                    None => "no measurement".to_owned(),
                };
                println!(
                    "Pipeline ran {} on {} cut(s): {measured} measurement(s); \
                     selected {} by {} ({score}); emitted {}; receipt in {}.",
                    receipt.dataset.id,
                    receipt.prepared.cuts.len(),
                    receipt.search.selected_genome,
                    receipt.search.selection,
                    receipt
                        .artifacts
                        .iter()
                        .map(|a| a.language.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    options.output.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::VerifyReceipt {
            receipt: path,
            evidence,
            json,
        } => match if evidence {
            crate::evidence::verify_run(&path).map(|run| run.receipt)
        } else {
            receipt::read(&path)
        } {
            Ok(receipt) => {
                if json {
                    println!("{}", serde_json::to_string(&receipt)?);
                } else {
                    println!(
                        "ok {} ({} stage(s), {} artifact(s), {})",
                        path.display(),
                        receipt.stages.len(),
                        receipt.artifacts.len(),
                        receipt.schema
                    );
                }
                Ok(ExitCode::SUCCESS)
            }
            Err(error) => {
                if json {
                    println!(
                        "{}",
                        serde_json::json!({"schema": receipt::RECEIPT_SCHEMA_ID, "ok": false,
                                           "error": format!("{error:#}")})
                    );
                } else {
                    eprintln!("invalid receipt: {error:#}");
                }
                Ok(ExitCode::FAILURE)
            }
        },
    }
}
