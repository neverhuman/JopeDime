//! Rust-only corpus ingestion, index export, and operational status entrypoint.

use anyhow::Result;
use clap::{Parser, Subcommand};
use jope_campaigns::paths;
use jope_campaigns::selfplay_ingest;
use serde_json::{json, Value};
use std::path::PathBuf;

jope_paths::once_str!(
    fn default_root_str => paths::flywheel_root()
);

#[derive(Parser)]
#[command(name = "jope-selfplay-status")]
#[command(about = "Native self-play receipt ingestion, indexes, and status v3")]
struct Args {
    #[arg(long, default_value = default_root_str(), global = true)]
    root: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Build and atomically publish operational status v3.
    Status,
    /// Ingest one already-durable evaluator receipt directory.
    Ingest {
        #[arg(long)]
        job_id: String,
        #[arg(long)]
        receipt_dir: PathBuf,
    },
    /// Ingest new durable receipt directories without calling SCQ.
    Reconcile {
        #[arg(long, default_value_t = 100)]
        limit: usize,
        /// Fetch completed SCQ outputs before local ingestion.
        #[arg(long)]
        fetch_completed: bool,
        #[arg(long, default_value_os_t = paths::scq_binary())]
        scq: PathBuf,
        #[arg(long, default_value_os_t = paths::scq_runs_root())]
        runs_root: PathBuf,
        /// Rebuild full champion/HOF indexes after ingestion. This is
        /// intentionally opt-in because it scans the complete corpus.
        #[arg(long)]
        rebuild_indexes: bool,
    },
    /// Rebuild deterministic champion and strict-win indexes.
    RebuildIndexes,
    /// Export current per-dataset champions for sync-gp-winners.
    ExportGeneEvidence {
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Answer protocol-scoped corpus winner, repeat, and origin-lift questions.
    EvidenceQuery {
        #[arg(long)]
        task: String,
        #[arg(long)]
        protocol_sha256: String,
        #[arg(long)]
        dataset: Option<String>,
        #[arg(long)]
        semantic: Option<String>,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    /// Export robust GP seeds directly from the latest indexed corpus evidence.
    ExportDynamicSeeds {
        #[arg(long)]
        task: String,
        #[arg(long)]
        protocol_sha256: String,
        #[arg(long, default_value_t = 256)]
        limit: usize,
        #[arg(long, default_value_t = 2)]
        minimum_datasets: usize,
    },
    /// Install the corpus and pair-registry composite query indexes.
    EnsureQueryIndexes,
    /// Build ops/seeding-curve.v1.json: per-checkpoint strict-win, champion,
    /// and DL-origin observation-share curve ("does the model seed better
    /// every cycle?"). Corpus access is read-only through the existing
    /// observations_checkpoint_v1 index with a persisted incremental cursor.
    SeedingCurve {
        /// Ceiling of new corpus rows walked in this pass; catch-up converges
        /// across repeated invocations.
        #[arg(long, default_value_t = selfplay_ingest::SEEDING_CURVE_DEFAULT_MAX_ROWS)]
        max_rows: usize,
    },
}

pub fn main(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(&args);
    let result: Value = match args.command.unwrap_or(Command::Status) {
        Command::Status => selfplay_ingest::build_status(&args.root)?,
        Command::Ingest {
            job_id,
            receipt_dir,
        } => {
            let summary = selfplay_ingest::ingest_receipt(
                &args.root.join("corpus/strict-test-exact.sqlite"),
                &job_id,
                &receipt_dir,
            )?;
            json!({
                "job_id": job_id,
                "rows": summary.rows,
                "valid": summary.valid,
                "failed": summary.failed,
                "already_ingested": summary.already_ingested,
            })
        }
        Command::Reconcile {
            limit,
            fetch_completed,
            scq,
            runs_root,
            rebuild_indexes,
        } => {
            let mut result = if fetch_completed {
                selfplay_ingest::reconcile_scheduler(
                    &args.root,
                    &scq,
                    &runs_root,
                    limit,
                    rebuild_indexes,
                )?
            } else {
                selfplay_ingest::reconcile_local_mode(&args.root, limit, rebuild_indexes)?
            };
            // Cheap, throttled, skip-on-lock: a curve failure must never fail
            // the reconcile that fed it.
            if let Some(object) = result.as_object_mut() {
                object.insert(
                    "seeding_curve".to_owned(),
                    selfplay_ingest::seeding_curve_after_reconcile(&args.root),
                );
            }
            result
        }
        Command::RebuildIndexes => selfplay_ingest::rebuild_indexes(&args.root)?,
        Command::ExportGeneEvidence { output } => {
            let output = output.unwrap_or_else(|| args.root.join("indexes/gene-evidence.v2.jsonl"));
            selfplay_ingest::export_gene_evidence(&args.root, &output)?
        }
        Command::EvidenceQuery {
            task,
            protocol_sha256,
            dataset,
            semantic,
            limit,
        } => selfplay_ingest::evidence_report(
            &args.root,
            &task,
            &protocol_sha256,
            dataset.as_deref(),
            semantic.as_deref(),
            limit,
        )?,
        Command::ExportDynamicSeeds {
            task,
            protocol_sha256,
            limit,
            minimum_datasets,
        } => selfplay_ingest::dynamic_seed_bank(
            &args.root.join("corpus/strict-test-exact.sqlite"),
            &task,
            &protocol_sha256,
            limit,
            minimum_datasets,
        )?,
        Command::SeedingCurve { max_rows } => {
            selfplay_ingest::seeding_curve_locked(&args.root, max_rows)?
        }
        Command::EnsureQueryIndexes => {
            let corpus_path = args.root.join("corpus/strict-test-exact.sqlite");
            drop(selfplay_ingest::open_corpus(&corpus_path)?);
            json!({
                "schema": "JopeEvidenceQueryIndexes.v1",
                "corpus": corpus_path,
                "pair_registry": selfplay_ingest::ensure_pair_query_indexes(
                    &args.root.join("ops/pair-registry.sqlite")
                )?,
                "corpus_indexes": [
                    "observations_dataset_protocol_best_v2",
                    "observations_semantic_protocol_best_v2",
                    "observations_origin_protocol_lift_v2",
                    "observations_checkpoint_protocol_lift_v2"
                ],
            })
        }
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
