//! `jope pipeline …`: the same `Cli`/`Cmd`/`exec`/`run` shape the other
//! subcommand crates expose, so the umbrella delegates to it in process.

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
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

/// How many CPUs the search may use. The worker count never changes a decision.
#[derive(Args, Clone, Debug)]
pub struct ParallelArgs {
    /// Worker threads evaluating each generation's candidates (1 = sequential). Refused above
    /// the CPUs sched_getaffinity allows, available_parallelism, or --cpu-ceiling.
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
    /// Independent random streams over the same cuts, each on --threads workers; islands x
    /// threads is the worker count the CPU checks bound.
    #[arg(long, default_value_t = 1)]
    pub islands: usize,
    /// With more than one island: every K generations each island's champion migrates into the
    /// others' elites.
    #[arg(long, default_value_t = crate::search::DEFAULT_MIGRATION_INTERVAL)]
    pub migration_interval: usize,
    /// The CPU admission in millicores (e.g. 8000 for eight cores) the workers must fit.
    #[arg(long)]
    pub cpu_ceiling: Option<u64>,
}

/// What a search runs on and with, shared by `run` and `evolve`.
#[derive(Args, Clone, Debug)]
pub struct SearchArgs {
    /// A 16-hex catalog id resolved in the local cache, or a dataset directory.
    #[arg(long)]
    pub dataset: String,
    /// Total per-cut budget across search and comparison (1..=1000 attempts / 1..=600 active seconds).
    /// Multiple cuts reserve half for comparison; an in-flight fit may overrun time.
    #[arg(long, default_value = "evaluations=64,seconds=300")]
    pub budget: String,
    /// Fold seed for the cut preparation.
    #[arg(long, default_value_t = crate::DEFAULT_SEED)]
    pub seed: u64,
    /// Seed of the search's random streams.
    #[arg(long, default_value_t = crate::DEFAULT_SEED)]
    pub rng_seed: u64,
    /// How many of the fold partition's cuts to prepare and search (1..=5).
    #[arg(long, default_value_t = 1)]
    pub cuts: usize,
    /// Languages to emit: rust, python, cpp17, cpp20, js, or all.
    #[arg(long, default_value = "rust,python,cpp17,cpp20,js")]
    pub emit: String,
    /// The proposal stage: auto and skip record a skip, require fails closed.
    #[arg(long, default_value = "auto")]
    pub policy: String,
    #[command(flatten)]
    pub parallel: ParallelArgs,
    /// The commit the caller runs (40 or 64 lowercase hex): recorded as the receipt's git_sha,
    /// and git is never spawned.
    #[arg(long)]
    pub source_commit: Option<String>,
}

impl SearchArgs {
    /// Run options for `output`, with the termination flag wired when `signals`.
    pub fn options(&self, output: PathBuf, signals: bool) -> Result<RunOptions> {
        let policy = &self.policy;
        Ok(RunOptions {
            budget: Budget::parse(&self.budget)?,
            seed: self.seed,
            rng_seed: self.rng_seed,
            cuts: self.cuts,
            languages: Language::parse_list(&self.emit)?,
            policy: Policy::parse(policy).ok_or_else(|| {
                anyhow::anyhow!("--policy must be auto, skip or require, got {policy:?}")
            })?,
            threads: self.parallel.threads,
            islands: self.parallel.islands,
            migration_interval: self.parallel.migration_interval,
            cpu_ceiling_millicores: self.parallel.cpu_ceiling,
            source_commit: self.source_commit.clone(),
            cancel: if signals {
                Some(crate::parallel::termination_flag()?)
            } else {
                None
            },
            ..RunOptions::new(self.dataset.clone(), output)
        })
    }
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Run dataset -> prepare -> search -> select -> emit and write the receipt.
    Run {
        #[command(flatten)]
        search: SearchArgs,
        /// Where the preparation, searches, emitted sources and receipt are written.
        #[arg(long)]
        output: PathBuf,
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
        Err(error) if crate::parallel::is_cancelled(&error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(crate::parallel::CANCELLED_EXIT)
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}

pub fn run(cli: Cli) -> Result<ExitCode> {
    match cli.cmd {
        Cmd::Run {
            search,
            output,
            seed_genomes,
            incumbent,
            json,
        } => {
            let options = RunOptions {
                seed_genomes,
                incumbent,
                ..search.options(output, true)?
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
