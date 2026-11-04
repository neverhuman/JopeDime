//! Rust-only bridge between checkpoint proposals and long-running GP populations.
//!
//! The bridge never invokes Python or mutates SCQ. Every action validates the
//! checkpoint artifact, proposal pointer, semantic identities, and task-separated
//! robust seed bank before writing an immutable output plus a hash-chained,
//! append-only receipt.

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use feat_math::genome::GenomeIr;
use jope_dime_regressor::selfplay_data::{
    export_gp_seeds, inject_promoted_proposals, GpAncestry, GpPopulationMember, GpProposal,
    GP_INJECTION_SLOTS, GP_SEED_SLOTS,
};
use jope_dime_regressor::selfplay_ingest;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, canonical_json_bytes, now_ms, read_json, sha256_bytes, sha256_file,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[path = "jope_gp_bridge/lineage.rs"]
mod lineage;
#[path = "jope_gp_bridge/population.rs"]
mod population;
#[path = "jope_gp_bridge/receipts.rs"]
mod receipts;
#[cfg(test)]
#[path = "jope_gp_bridge/tests.rs"]
mod tests;
#[path = "jope_gp_bridge/verification.rs"]
mod verification;

use lineage::lineage_value;
use population::{export_value, injection_value, merge_generation_zero};
#[cfg(test)]
use receipts::validate_receipt_chain;
use receipts::{durable_write, proposal_request};
use verification::{read_dynamic_robust_bank, read_population, verify_proposals};

const EXPORT_SCHEMA: &str = "JopeGpSeedExport.v2";
const GENERATION_ZERO_SCHEMA: &str = "JopeGpGenerationZeroSeed.v1";
const INJECTION_SCHEMA: &str = "JopeGpProposalInjection.v2";
const LINEAGE_SCHEMA: &str = "JopeGpDlLineageContribution.v1";
const RECEIPT_SCHEMA: &str = "JopeGpBridgeReceipt.v1";
const ROBUST_BANK_SCHEMA: &str = "JopeRobustGpSeedBank.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Task {
    Regression,
    Binary,
    Multiclass,
    Timeseries,
}

impl Task {
    fn as_str(self) -> &'static str {
        match self {
            Self::Regression => "regression",
            Self::Binary => "binary",
            Self::Multiclass => "multiclass",
            Self::Timeseries => "timeseries",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "jope-gp-bridge",
    about = "Checkpoint-safe Rust DL/GP seed export, merge, injection, and lineage receipts",
    long_about = "Exports all 64 current-checkpoint DL proposals for a dataset, merges them \
with repeated cross-dataset GP winners without increasing generation-zero population size, \
and replaces exactly eight lowest non-elites after checkpoint advancement. The bridge never \
invokes Python or SCQ. All writes are idempotent and recorded in a hash-chained JSONL log."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Export exactly 64 unique, current-checkpoint DL proposals.
    Export(ProposalAction),
    /// Seed GP generation zero with 64 DL proposals and repeated robust GP winners.
    MergeGenerationZero(MergeArgs),
    /// On checkpoint advancement, replace exactly eight lowest non-elites.
    InjectCheckpoint(InjectArgs),
    /// Compute transitive DL lineage contribution and append its receipt.
    RecordLineage(LineageArgs),
}

#[derive(Clone, Debug, Args)]
struct ProposalInputs {
    /// JSON pointer containing checkpoint_sha256 and the checkpoint artifact path.
    #[arg(long)]
    checkpoint_pointer: PathBuf,
    /// JSON pointer containing proposals, proposals_sha256, and checkpoint_sha256.
    #[arg(long)]
    proposals_pointer: PathBuf,
    /// Model task. Regression and binary seeds are never mixed.
    #[arg(long, value_enum)]
    task: Task,
    /// Immutable dataset content identity used by proposal rows.
    #[arg(long)]
    dataset: String,
}

#[derive(Clone, Debug, Args)]
struct DurableOutput {
    /// New immutable JSON output. An existing nonidentical file is rejected.
    #[arg(long)]
    output: PathBuf,
    /// Append-only, hash-chained JSONL receipt log.
    #[arg(long)]
    receipts: PathBuf,
    /// Caller-stable key. Reuse with different inputs fails closed.
    #[arg(long)]
    idempotency_key: String,
}

#[derive(Debug, Args)]
struct ProposalAction {
    #[command(flatten)]
    input: ProposalInputs,
    #[command(flatten)]
    durable: DurableOutput,
}

#[derive(Debug, Args)]
struct MergeArgs {
    #[command(flatten)]
    input: ProposalInputs,
    /// Existing generation-zero population; its length is the immutable budget.
    #[arg(long)]
    population: PathBuf,
    /// Current strict test-exact corpus queried at merge time.
    #[arg(long)]
    corpus: PathBuf,
    /// Evaluator protocol whose evidence is eligible for robust GP seeding.
    #[arg(long)]
    protocol_sha256: String,
    /// Minimum distinct datasets required for one repeated robust seed.
    #[arg(long, default_value_t = 2)]
    minimum_repeat_datasets: usize,
    /// Requested robust winners after the 64 DL slots; capped by population budget.
    #[arg(long, default_value_t = 96)]
    robust_slots: usize,
    #[command(flatten)]
    durable: DurableOutput,
}

#[derive(Debug, Args)]
struct InjectArgs {
    #[command(flatten)]
    input: ProposalInputs,
    /// Current GP population. Its first elite-count rows are protected.
    #[arg(long)]
    population: PathBuf,
    /// Checkpoint used before this advancement; must differ from the active pointer.
    #[arg(long)]
    previous_checkpoint_sha256: String,
    /// Count of leading elite rows that must remain bit-for-bit unchanged.
    #[arg(long)]
    elite_count: usize,
    #[command(flatten)]
    durable: DurableOutput,
}

#[derive(Debug, Args)]
struct LineageArgs {
    /// JSON or JSONL lineage records containing semantic_sha256, parents, dl_origin, strict_win.
    #[arg(long)]
    lineage: PathBuf,
    /// Semantic identity of the final champion.
    #[arg(long)]
    final_champion_semantic_sha256: String,
    /// Model task for receipt provenance.
    #[arg(long, value_enum)]
    task: Task,
    /// Dataset content identity for receipt provenance.
    #[arg(long)]
    dataset: String,
    /// Stable GP run identity.
    #[arg(long)]
    run_id: String,
    #[command(flatten)]
    durable: DurableOutput,
}

#[derive(Clone, Debug, Deserialize)]
struct CheckpointPointer {
    checkpoint_sha256: String,
    #[serde(alias = "path", alias = "checkpoint_path")]
    checkpoint: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
struct ProposalPointer {
    proposals: PathBuf,
    proposals_sha256: String,
    checkpoint_sha256: String,
}

#[derive(Clone, Debug)]
struct VerifiedProposals {
    checkpoint_sha256: String,
    checkpoint_artifact: PathBuf,
    checkpoint_pointer_sha256: String,
    proposal_file: PathBuf,
    proposal_file_sha256: String,
    proposal_pointer_sha256: String,
    proposals: Vec<GpProposal>,
    task_invalid_skipped: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct RobustBank {
    schema: String,
    tasks: BTreeMap<String, Vec<RobustSeed>>,
}

#[derive(Clone, Debug, Deserialize)]
struct RobustSeed {
    task: String,
    semantic_sha256: String,
    canonical_genome: String,
    dataset_count: usize,
    mean_within_dataset_percentile: f64,
    worst_within_dataset_percentile: f64,
    strict_win_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct LineageRow {
    semantic_sha256: String,
    #[serde(default)]
    canonical_genome: Option<String>,
    #[serde(default)]
    dl_origin: bool,
    #[serde(default)]
    parents: Vec<String>,
    #[serde(default)]
    ancestry: Option<GpAncestry>,
    #[serde(default)]
    strict_win: bool,
}

impl LineageRow {
    fn all_parents(&self) -> impl Iterator<Item = &String> {
        self.parents
            .iter()
            .chain(self.ancestry.iter().flat_map(|row| row.parents.iter()))
    }
}

struct ReceiptLock {
    file: File,
}

impl ReceiptLock {
    fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let lock_path = path.with_extension(
            path.extension()
                .and_then(|value| value.to_str())
                .map(|value| format!("{value}.lock"))
                .unwrap_or_else(|| "lock".to_owned()),
        );
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("open receipt lock {}", lock_path.display()))?;
        ensure!(
            file.lock().is_ok(),
            "acquire receipt lock {}",
            lock_path.display()
        );
        Ok(Self { file })
    }
}

impl Drop for ReceiptLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    ensure!(is_sha256(value), "invalid {label}: {value}");
    Ok(())
}

fn resolve_pointer_path(pointer: &Path, target: &Path) -> PathBuf {
    if target.is_absolute() {
        target.to_owned()
    } else {
        pointer
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(target)
    }
}

fn run() -> Result<Value> {
    match Cli::parse().command {
        Command::Export(args) => {
            let verified = verify_proposals(&args.input)?;
            let value = export_value(&verified, args.input.task, &args.input.dataset);
            let request = proposal_request("export", &args.input, &verified);
            durable_write(
                "export",
                request,
                json!({ "task_invalid_skipped": verified.task_invalid_skipped }),
                &args.durable.output,
                &value,
                &args.durable.receipts,
                &args.durable.idempotency_key,
            )
        }
        Command::MergeGenerationZero(args) => {
            let verified = verify_proposals(&args.input)?;
            let population = read_population(&args.population)?;
            let dynamic_limit = args.robust_slots.saturating_add(GP_SEED_SLOTS).max(256);
            let (robust, robust_sha256, robust_evidence) = read_dynamic_robust_bank(
                &args.corpus,
                args.input.task,
                &args.protocol_sha256,
                dynamic_limit,
                args.minimum_repeat_datasets,
            )?;
            let value = merge_generation_zero(
                &population,
                &verified.proposals,
                &verified.checkpoint_sha256,
                &robust,
                args.robust_slots,
                args.input.task,
                &args.input.dataset,
            )?;
            let mut request = proposal_request("merge_generation_zero", &args.input, &verified);
            request
                .as_object_mut()
                .expect("JSON object")
                .extend(Map::from_iter([
                    (
                        "population_sha256".to_owned(),
                        Value::String(sha256_file(&args.population)?),
                    ),
                    (
                        "dynamic_robust_evidence_sha256".to_owned(),
                        Value::String(robust_sha256),
                    ),
                    (
                        "dynamic_robust_evidence_watermark".to_owned(),
                        robust_evidence["evidence_watermark"].clone(),
                    ),
                    (
                        "protocol_sha256".to_owned(),
                        Value::String(args.protocol_sha256),
                    ),
                    (
                        "corpus".to_owned(),
                        Value::String(args.corpus.to_string_lossy().into_owned()),
                    ),
                    (
                        "minimum_repeat_datasets".to_owned(),
                        Value::from(args.minimum_repeat_datasets),
                    ),
                    ("robust_slots".to_owned(), Value::from(args.robust_slots)),
                ]));
            durable_write(
                "merge_generation_zero",
                request,
                json!({ "task_invalid_skipped": verified.task_invalid_skipped }),
                &args.durable.output,
                &value,
                &args.durable.receipts,
                &args.durable.idempotency_key,
            )
        }
        Command::InjectCheckpoint(args) => {
            let verified = verify_proposals(&args.input)?;
            let population = read_population(&args.population)?;
            let value = injection_value(
                &population,
                &verified.proposals,
                &verified.checkpoint_sha256,
                &args.previous_checkpoint_sha256,
                args.elite_count,
                args.input.task,
                &args.input.dataset,
            )?;
            let mut request = proposal_request("inject_checkpoint", &args.input, &verified);
            request
                .as_object_mut()
                .expect("JSON object")
                .extend(Map::from_iter([
                    (
                        "population_sha256".to_owned(),
                        Value::String(sha256_file(&args.population)?),
                    ),
                    (
                        "previous_checkpoint_sha256".to_owned(),
                        Value::String(args.previous_checkpoint_sha256),
                    ),
                    ("elite_count".to_owned(), Value::from(args.elite_count)),
                ]));
            durable_write(
                "inject_checkpoint",
                request,
                json!({ "task_invalid_skipped": verified.task_invalid_skipped }),
                &args.durable.output,
                &value,
                &args.durable.receipts,
                &args.durable.idempotency_key,
            )
        }
        Command::RecordLineage(args) => {
            let value = lineage_value(
                &args.lineage,
                &args.final_champion_semantic_sha256,
                args.task,
                &args.dataset,
                &args.run_id,
            )?;
            let request = json!({
                "action": "record_lineage",
                "lineage_sha256": sha256_file(&args.lineage)?,
                "final_champion_semantic_sha256": args.final_champion_semantic_sha256,
                "task": args.task.as_str(),
                "dataset_hash": args.dataset,
                "run_id": args.run_id,
            });
            durable_write(
                "record_lineage",
                request,
                Value::Null,
                &args.durable.output,
                &value,
                &args.durable.receipts,
                &args.durable.idempotency_key,
            )
        }
    }
}

fn main() {
    match run() {
        Ok(receipt) => println!(
            "{}",
            serde_json::to_string(&receipt).expect("serialize receipt")
        ),
        Err(error) => {
            eprintln!("jope-gp-bridge: {error:#}");
            std::process::exit(1);
        }
    }
}
