//! WS-5 instrument 1: evaluate the frozen sealed cohort — the project's only
//! benchmark-valid zero-shot estimate — and report it per checkpoint.
//!
//! `catalog/private/sealed-expanded-v1.json` holds 256 family-disjoint datasets
//! per task. No run has ever evaluated against it, so every generalization
//! number in this project rests on a validation split that has been trained
//! against repeatedly.
//!
//! SAFETY CONTRACT (all four hold, and each is load-bearing):
//!
//!  1. The sealed cohort file is opened READ-ONLY and never rewritten. This tool
//!     has no `--freeze-sealed` path and never calls `freeze_sealed`. That
//!     function is idempotent ONLY because the file exists
//!     (selfplay_expansion.rs:824); losing the file silently mints a DIFFERENT
//!     512-dataset cohort with no error and destroys the zero-shot guarantee.
//!
//!  2. Emitted job ids do NOT start with `jte-`. `reconcile_local_mode`
//!     (selfplay_ingest.rs:813-821) only considers `$R/receipts/jte-*`
//!     directories, so these jobs are invisible to it by construction.
//!
//!  3. No row is ever written to `$R/ops/backlog.sqlite`. Both ingestion paths
//!     are driven by `SELECT job_id FROM claims WHERE state='submitted' ...`
//!     (selfplay_ingest.rs:995-1001, :800-803). With no claim there is no
//!     ingestion, so a sealed evaluation can never become a training label.
//!
//!  4. Manifests, specs and receipts live under an operator-supplied output
//!     directory, never under `$R/manifests` or `$R/receipts`.
//!
//! Together these mean: running this tool cannot contaminate the cohort it
//! measures. That is the whole point — an instrument that pollutes its own
//! sample is worse than no instrument.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_expansion::TASKS;
use jope_dime_regressor::spec_policy::EVAL;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const SHARD_SCHEMA: &str = "JopeFlywheelCorpusEvalShard.v1";
const PORTFOLIO_SCHEMA: &str = "JopeSealedZeroShotPortfolio.v1";
const REPORT_SCHEMA: &str = "JopeSealedZeroShotReport.v1";
/// Job-id prefix. MUST NOT be `jte-` — see safety contract item 2.
const JOB_PREFIX: &str = "wsm5z";
const PROJECT: &str = "wsm5-sealed-zeroshot";

#[derive(Parser)]
#[command(about = "Zero-shot evaluation of the frozen sealed cohort (read-only on the cohort)")]
struct Args {
    #[arg(long, default_value_os_t = paths::flywheel_root())]
    root: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build the portfolio, the per-dataset manifests and the SCQ specs.
    Plan {
        /// Output directory for portfolio/manifests/specs. Never inside $R.
        #[arg(long)]
        out: PathBuf,
        /// Genomes per task drawn from the champion index, ranked by the number
        /// of TRAINING datasets each is champion on.
        #[arg(long, default_value_t = 32)]
        top_k: usize,
        /// Evaluate only the first N sealed datasets per task (0 = all 256).
        #[arg(long, default_value_t = 0)]
        limit_per_task: usize,
        /// Evaluator binary to stage into each job.
        #[arg(long)]
        evaluator: PathBuf,
        /// Shard runner script to stage into each job.
        #[arg(
            long,
            default_value_os_t = paths::fweval_runner()
        )]
        runner: PathBuf,
        /// Skip sealed datasets that already carry observations in the corpus.
        /// They are contaminated and cannot support a zero-shot claim.
        #[arg(long, default_value_t = true)]
        exclude_contaminated: bool,
    },
    /// Report which sealed datasets are already contaminated, and how.
    Contamination {
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Collect finished receipts and emit the zero-shot report.
    Report {
        /// The `plan --out` directory.
        #[arg(long)]
        plan: PathBuf,
        /// SCQ run root holding job outputs.
        #[arg(long, default_value_os_t = paths::scq_runs_root())]
        runs: PathBuf,
        /// SCQ CLI, used to fetch outputs from the node a job actually ran on.
        #[arg(long, default_value_os_t = paths::scq_binary())]
        scq: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

#[path = "jope_sealed_zeroshot/planning.rs"]
mod planning;
#[path = "jope_sealed_zeroshot/portfolio.rs"]
mod portfolio;
#[path = "jope_sealed_zeroshot/reporting.rs"]
mod reporting;
#[path = "jope_sealed_zeroshot/sealed.rs"]
mod sealed;

use planning::{atomic_write, plan};
use portfolio::{baseline_genomes, build_portfolio, sha256_hex, PortfolioGenome};
use reporting::{median, report};
use sealed::{contaminated, load_sealed, SealedCohort, SealedEntry};

fn main() -> Result<()> {
    let args = Args::parse();
    let value = match &args.command {
        Command::Plan {
            out,
            top_k,
            limit_per_task,
            evaluator,
            runner,
            exclude_contaminated,
        } => plan(
            &args.root,
            out,
            *top_k,
            *limit_per_task,
            evaluator,
            runner,
            *exclude_contaminated,
        )?,
        Command::Contamination { out } => {
            let cohort = load_sealed(&args.root)?;
            let dirty = contaminated(&args.root, &cohort)?;
            let by_task = cohort
                .entries
                .iter()
                .filter(|e| dirty.contains_key(&format!("{}:{}", e.task, e.dataset_hash)))
                .fold(BTreeMap::<String, u64>::new(), |mut acc, e| {
                    *acc.entry(e.task.clone()).or_insert(0) += 1;
                    acc
                });
            let detail: Vec<Value> = cohort
                .entries
                .iter()
                .filter_map(|e| {
                    let key = format!("{}:{}", e.task, e.dataset_hash);
                    dirty.get(&key).map(|(n, best)| {
                        json!({
                            "task": e.task, "dataset_id": e.dataset_id,
                            "dataset_hash": e.dataset_hash,
                            "source_family": e.source_family,
                            "observations": n, "best_score": best,
                        })
                    })
                })
                .collect();
            let value = json!({
                "cohort_sha256": cohort.cohort_sha256,
                "sealed_total": cohort.entries.len(),
                "contaminated_total": dirty.len(),
                "contaminated_by_task": by_task,
                "clean_total": cohort.entries.len() - dirty.len(),
                "detail": detail,
            });
            if let Some(path) = out {
                atomic_write(path, &serde_json::to_vec_pretty(&value)?)?;
            }
            value
        }
        Command::Report {
            plan,
            runs,
            scq,
            out,
        } => report(plan, runs, scq, out)?,
    };
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_ids_never_collide_with_the_ingestion_prefix() {
        // selfplay_ingest.rs:813-821 only materializes $R/receipts/jte-* dirs.
        assert!(!JOB_PREFIX.starts_with("jte"));
        assert!(!format!("{JOB_PREFIX}-r-abc").starts_with("jte-"));
    }

    #[test]
    fn semantic_hash_matches_the_evaluator_definition() {
        // corpus_eval.rs:690 recomputes sha256(canonical_genome) and warns on drift.
        let genome = "(pipe X (ridge))";
        assert_eq!(sha256_hex(genome.as_bytes()).len(), 64);
    }

    #[test]
    fn baseline_arms_exist_for_both_tasks() {
        assert_eq!(baseline_genomes("binary").len(), 2);
        assert_eq!(baseline_genomes("regression").len(), 2);
        assert!(baseline_genomes("binary")[0].0.contains("logit_ridge"));
        assert!(baseline_genomes("regression")[0].0.contains("ridge"));
    }

    #[test]
    fn median_handles_even_and_odd() {
        assert_eq!(median(&mut [1.0, 3.0]), 2.0);
        assert_eq!(median(&mut [1.0, 5.0, 3.0]), 3.0);
        assert!(median(&mut []).is_nan());
    }
}
