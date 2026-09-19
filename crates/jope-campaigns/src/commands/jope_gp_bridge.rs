//! Rust-only bridge between checkpoint proposals and long-running GP populations.
//!
//! The bridge never invokes Python or mutates SCQ. Every action validates the
//! checkpoint artifact, proposal pointer, semantic identities, and task-separated
//! robust seed bank before writing an immutable output plus a hash-chained,
//! append-only receipt.

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use feat_math::genome::GenomeIr;
use crate::selfplay_data::{
    export_gp_seeds, inject_promoted_proposals, GpAncestry, GpPopulationMember, GpProposal,
    GP_INJECTION_SLOTS, GP_SEED_SLOTS,
};
use crate::selfplay_ingest;
use crate::selfplay_runtime::{
    atomic_json, canonical_json_bytes, now_ms, read_json, sha256_bytes, sha256_file,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

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
        let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        ensure!(status == 0, "acquire receipt lock {}", lock_path.display());
        Ok(Self { file })
    }
}

impl Drop for ReceiptLock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
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

fn task_valid(source: &str, task: &str) -> bool {
    let Ok(genome) = GenomeIr::parse(source) else {
        return false;
    };
    if genome.canonical() != source {
        return false;
    }
    let GenomeIr::Pipeline(pipeline) = genome else {
        return false;
    };
    let Some(head) = pipeline.stages.last().map(|stage| stage.op.as_str()) else {
        return false;
    };
    // Mirrors the wave-side allowlist in jope_selfplay_wave.rs exactly: timeseries
    // shares the regression arm because corpus_eval validates a timeseries genome
    // as TaskSpec::regression(Rmse) over the reduced schema, and multiclass admits
    // only the four multiclass_* heads (the same set corpus_eval's V3_EXTRA_HEADS
    // and the multiclass60 campaign pin). If these two allowlists ever disagree,
    // the bridge injects seeds the evaluator rejects -- silent seed loss.
    match task {
        "binary" => matches!(
            head,
            "logit_ridge" | "logistic" | "tab_boost" | "extra_trees" | "gam_boost" | "portfolio"
        ),
        "regression" | "timeseries" => {
            head != "logit_ridge"
                && head != "logistic"
                && (feat_math::heads::is_head_op(head)
                    || matches!(head, "tab_boost" | "extra_trees" | "gam_boost" | "pls"))
        }
        "multiclass" => matches!(
            head,
            "multiclass_ridge"
                | "multiclass_ovr_logistic"
                | "multiclass_diagonal_lda"
                | "multiclass_nearest_centroid"
        ),
        _ => false,
    }
}

fn verify_proposals(input: &ProposalInputs) -> Result<VerifiedProposals> {
    ensure!(
        !input.dataset.trim().is_empty(),
        "dataset identity must not be empty"
    );
    let checkpoint_pointer: CheckpointPointer =
        serde_json::from_value(read_json(&input.checkpoint_pointer)?).with_context(|| {
            format!(
                "parse checkpoint pointer {}",
                input.checkpoint_pointer.display()
            )
        })?;
    validate_sha256(
        &checkpoint_pointer.checkpoint_sha256,
        "checkpoint pointer SHA-256",
    )?;
    let checkpoint_artifact =
        resolve_pointer_path(&input.checkpoint_pointer, &checkpoint_pointer.checkpoint);
    ensure!(
        checkpoint_artifact.is_file(),
        "checkpoint artifact is absent: {}",
        checkpoint_artifact.display()
    );
    ensure!(
        sha256_file(&checkpoint_artifact)? == checkpoint_pointer.checkpoint_sha256,
        "checkpoint artifact content drift"
    );

    let proposal_pointer: ProposalPointer =
        serde_json::from_value(read_json(&input.proposals_pointer)?).with_context(|| {
            format!(
                "parse proposal pointer {}",
                input.proposals_pointer.display()
            )
        })?;
    validate_sha256(&proposal_pointer.proposals_sha256, "proposal file SHA-256")?;
    ensure!(
        proposal_pointer.checkpoint_sha256 == checkpoint_pointer.checkpoint_sha256,
        "proposal pointer does not reference the active checkpoint"
    );
    let proposal_file = resolve_pointer_path(&input.proposals_pointer, &proposal_pointer.proposals);
    ensure!(
        proposal_file.is_file(),
        "proposal file is absent: {}",
        proposal_file.display()
    );
    ensure!(
        sha256_file(&proposal_file)? == proposal_pointer.proposals_sha256,
        "proposal pointer content drift"
    );

    let file = File::open(&proposal_file)
        .with_context(|| format!("open proposal file {}", proposal_file.display()))?;
    let mut all = Vec::new();
    let mut task_invalid_skipped = 0usize;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: GpProposal = serde_json::from_str(&line)
            .with_context(|| format!("parse proposal JSONL line {}", index + 1))?;
        ensure!(
            row.model_checkpoint_sha256 == checkpoint_pointer.checkpoint_sha256,
            "proposal file mixes checkpoint epochs at line {}",
            index + 1
        );
        ensure!(
            sha256_bytes(row.canonical_genome.as_bytes()) == row.semantic_sha256,
            "proposal semantic identity drift at line {}",
            index + 1
        );
        // Task-invalid rows are skipped and counted rather than aborting the
        // whole file, matching the sanitize-proposals policy; the count is
        // surfaced in the durable receipt summary.
        if !task_valid(&row.canonical_genome, &row.task) {
            task_invalid_skipped += 1;
            continue;
        }
        all.push(row);
    }
    let proposals = export_gp_seeds(
        &all,
        input.task.as_str(),
        &input.dataset,
        &checkpoint_pointer.checkpoint_sha256,
    )?;
    Ok(VerifiedProposals {
        checkpoint_sha256: checkpoint_pointer.checkpoint_sha256,
        checkpoint_artifact,
        checkpoint_pointer_sha256: sha256_file(&input.checkpoint_pointer)?,
        proposal_file,
        proposal_file_sha256: proposal_pointer.proposals_sha256,
        proposal_pointer_sha256: sha256_file(&input.proposals_pointer)?,
        proposals,
        task_invalid_skipped,
    })
}

fn read_population(path: &Path) -> Result<Vec<GpPopulationMember>> {
    let value = read_json(path)?;
    let values = match &value {
        Value::Array(rows) => rows.clone(),
        Value::Object(object) => ["population", "members", "slots"]
            .iter()
            .find_map(|key| object.get(*key).and_then(Value::as_array).cloned())
            .context("population JSON requires an array or population/members/slots array")?,
        _ => bail!("population JSON must be an array or object"),
    };
    let mut members = Vec::with_capacity(values.len());
    let mut semantics = HashSet::new();
    for (index, value) in values.into_iter().enumerate() {
        let member: GpPopulationMember = serde_json::from_value(value)
            .with_context(|| format!("parse population member {index}"))?;
        validate_sha256(
            &member.semantic_sha256,
            &format!("population member {index} semantic SHA-256"),
        )?;
        ensure!(
            sha256_bytes(member.canonical_genome.as_bytes()) == member.semantic_sha256,
            "population semantic identity drift at slot {index}"
        );
        ensure!(
            member.score.into_iter().all(f64::is_finite),
            "population contains non-finite score at slot {index}"
        );
        ensure!(
            semantics.insert(member.semantic_sha256.clone()),
            "population repeats semantic identity at slot {index}"
        );
        members.push(member);
    }
    ensure!(!members.is_empty(), "GP population must not be empty");
    Ok(members)
}

fn read_dynamic_robust_bank(
    corpus: &Path,
    task: Task,
    protocol_sha256: &str,
    limit: usize,
    minimum_datasets: usize,
) -> Result<(Vec<RobustSeed>, String, Value)> {
    let value = selfplay_ingest::dynamic_seed_bank(
        corpus,
        task.as_str(),
        protocol_sha256,
        limit,
        minimum_datasets,
    )?;
    let bank: RobustBank =
        serde_json::from_value(value.clone()).context("parse dynamic robust seed bank")?;
    ensure!(
        bank.schema == ROBUST_BANK_SCHEMA,
        "unsupported robust seed-bank schema"
    );
    ensure!(
        bank.tasks.keys().all(|key| {
            matches!(
                key.as_str(),
                "regression" | "binary" | "timeseries" | "multiclass"
            )
        }),
        "robust seed bank contains an unsupported task"
    );
    let mut rows = bank
        .tasks
        .get(task.as_str())
        .cloned()
        .with_context(|| format!("robust seed bank lacks {} seeds", task.as_str()))?;
    let mut semantics = HashSet::new();
    for (index, row) in rows.iter().enumerate() {
        ensure!(
            row.task == task.as_str(),
            "robust seed task-map provenance drift at row {index}"
        );
        ensure!(
            row.dataset_count >= 2,
            "robust seed is not repeated across datasets at row {index}"
        );
        ensure!(
            row.mean_within_dataset_percentile.is_finite()
                && row.worst_within_dataset_percentile.is_finite(),
            "robust seed has non-finite rank evidence at row {index}"
        );
        validate_sha256(
            &row.semantic_sha256,
            &format!("robust seed {index} semantic SHA-256"),
        )?;
        ensure!(
            sha256_bytes(row.canonical_genome.as_bytes()) == row.semantic_sha256,
            "robust seed semantic identity drift at row {index}"
        );
        ensure!(
            task_valid(&row.canonical_genome, task.as_str()),
            "robust seed is invalid for {} at row {index}",
            task.as_str()
        );
        ensure!(
            semantics.insert(row.semantic_sha256.clone()),
            "robust seed bank repeats a semantic at row {index}"
        );
    }
    rows.sort_by(|left, right| {
        right
            .mean_within_dataset_percentile
            .total_cmp(&left.mean_within_dataset_percentile)
            .then_with(|| {
                right
                    .worst_within_dataset_percentile
                    .total_cmp(&left.worst_within_dataset_percentile)
            })
            .then_with(|| right.strict_win_count.cmp(&left.strict_win_count))
            .then_with(|| right.dataset_count.cmp(&left.dataset_count))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    let digest = sha256_bytes(&canonical_json_bytes(&value)?);
    Ok((rows, digest, value))
}

fn export_value(verified: &VerifiedProposals, task: Task, dataset: &str) -> Value {
    json!({
        "schema": EXPORT_SCHEMA,
        "task": task.as_str(),
        "dataset_hash": dataset,
        "model_checkpoint_sha256": verified.checkpoint_sha256,
        "checkpoint_artifact": verified.checkpoint_artifact,
        "checkpoint_artifact_sha256": verified.checkpoint_sha256,
        "checkpoint_pointer_sha256": verified.checkpoint_pointer_sha256,
        "proposal_file": verified.proposal_file,
        "proposal_set_sha256": verified.proposal_file_sha256,
        "proposal_pointer_sha256": verified.proposal_pointer_sha256,
        "population_budget_delta": 0,
        "generation_zero_consumption": "replace_existing_seed_slots",
        "proposals": verified.proposals,
    })
}

fn proposal_member(proposal: &GpProposal, checkpoint: &str, operation: &str) -> GpPopulationMember {
    GpPopulationMember {
        canonical_genome: proposal.canonical_genome.clone(),
        semantic_sha256: proposal.semantic_sha256.clone(),
        // Proposal-model priority is not GP fitness on this dataset. The GP must
        // evaluate every injected seed before selection can use its fitness.
        score: None,
        dl_origin: true,
        injected_checkpoint_sha256: Some(checkpoint.to_owned()),
        ancestry: Some(GpAncestry {
            operation: operation.to_owned(),
            parents: vec![proposal.semantic_sha256.clone()],
        }),
    }
}

fn merge_generation_zero(
    existing: &[GpPopulationMember],
    proposals: &[GpProposal],
    checkpoint: &str,
    robust: &[RobustSeed],
    robust_slots: usize,
    task: Task,
    dataset: &str,
) -> Result<Value> {
    ensure!(
        proposals.len() == GP_SEED_SLOTS,
        "generation zero requires exactly 64 DL proposals"
    );
    ensure!(
        existing.len() > GP_SEED_SLOTS,
        "population budget must leave at least one robust GP seed after 64 DL proposals"
    );
    ensure!(robust_slots > 0, "robust slot request must be positive");
    let required_robust = robust_slots.min(existing.len() - GP_SEED_SLOTS);
    let mut used = HashSet::new();
    let mut members = Vec::with_capacity(existing.len());
    for proposal in proposals {
        ensure!(
            used.insert(proposal.semantic_sha256.clone()),
            "DL proposal semantics repeat"
        );
        members.push(proposal_member(
            proposal,
            checkpoint,
            "generation_zero_current_checkpoint_dl_seed",
        ));
    }
    let mut selected_robust = Vec::with_capacity(required_robust);
    for row in robust {
        if !used.insert(row.semantic_sha256.clone()) {
            continue;
        }
        members.push(GpPopulationMember {
            canonical_genome: row.canonical_genome.clone(),
            semantic_sha256: row.semantic_sha256.clone(),
            // Cross-dataset evidence selects this seed but is not reusable as
            // fitness for the requested dataset.
            score: None,
            dl_origin: false,
            injected_checkpoint_sha256: None,
            ancestry: Some(GpAncestry {
                operation: "generation_zero_repeated_cross_dataset_gp_winner".to_owned(),
                parents: Vec::new(),
            }),
        });
        selected_robust.push(json!({
            "semantic_sha256": row.semantic_sha256,
            "dataset_count": row.dataset_count,
            "mean_within_dataset_percentile": row.mean_within_dataset_percentile,
            "worst_within_dataset_percentile": row.worst_within_dataset_percentile,
            "strict_win_count": row.strict_win_count,
        }));
        if selected_robust.len() == required_robust {
            break;
        }
    }
    ensure!(
        selected_robust.len() == required_robust,
        "robust seed bank has only {} usable distinct repeated winners; {required_robust} required",
        selected_robust.len()
    );
    for member in existing {
        if members.len() == existing.len() {
            break;
        }
        if used.insert(member.semantic_sha256.clone()) {
            members.push(member.clone());
        }
    }
    ensure!(
        members.len() == existing.len(),
        "deduplication left generation zero below its existing population budget"
    );
    Ok(json!({
        "schema": GENERATION_ZERO_SCHEMA,
        "task": task.as_str(),
        "dataset_hash": dataset,
        "checkpoint_sha256": checkpoint,
        "population_budget": existing.len(),
        "population_budget_delta": 0,
        "composition": {
            "current_checkpoint_dl": GP_SEED_SLOTS,
            "repeated_cross_dataset_gp_winners": required_robust,
            "retained_existing": existing.len() - GP_SEED_SLOTS - required_robust,
        },
        "robust_selection": selected_robust,
        "population": members,
    }))
}

fn injection_value(
    existing: &[GpPopulationMember],
    proposals: &[GpProposal],
    checkpoint: &str,
    previous_checkpoint: &str,
    elite_count: usize,
    task: Task,
    dataset: &str,
) -> Result<Value> {
    validate_sha256(previous_checkpoint, "previous checkpoint SHA-256")?;
    ensure!(
        checkpoint != previous_checkpoint,
        "checkpoint did not advance; injection is not permitted"
    );
    ensure!(
        existing.len().saturating_sub(elite_count) >= GP_INJECTION_SLOTS,
        "population has fewer than eight non-elites"
    );
    let elites_before = canonical_json_bytes(&serde_json::to_value(&existing[..elite_count])?)?;
    let (mut members, receipt) =
        inject_promoted_proposals(existing, proposals, checkpoint, elite_count)?;
    ensure!(
        receipt.injected == GP_INJECTION_SLOTS,
        "checkpoint advancement has only {} new proposals; exactly eight required",
        receipt.injected
    );
    ensure!(
        elites_before == canonical_json_bytes(&serde_json::to_value(&members[..elite_count])?)?,
        "checkpoint injection changed an elite"
    );
    let proposal_by_semantic = proposals
        .iter()
        .map(|row| (row.semantic_sha256.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let mut proposal_priority = Vec::with_capacity(GP_INJECTION_SLOTS);
    for replacement in &receipt.replacements {
        let proposal = proposal_by_semantic
            .get(replacement.injected_semantic_sha256.as_str())
            .context("injection receipt references an unknown proposal")?;
        proposal_priority.push(json!({
            "slot": replacement.slot,
            "semantic_sha256": proposal.semantic_sha256,
            "proposal_score": proposal.proposal_score,
            "observed_score": proposal.score,
        }));
        // The score ranks proposals for replacement but is not fitness on the
        // GP's current dataset. Force evaluation before selection.
        members[replacement.slot].score = None;
    }
    Ok(json!({
        "schema": INJECTION_SCHEMA,
        "task": task.as_str(),
        "dataset_hash": dataset,
        "previous_checkpoint_sha256": previous_checkpoint,
        "checkpoint_sha256": checkpoint,
        "population_budget": existing.len(),
        "population_budget_delta": 0,
        "population": members,
        "injection": receipt,
        "injected_proposal_priority": proposal_priority,
        "injected_fitness_state": "unevaluated",
    }))
}

fn read_json_or_jsonl(path: &Path) -> Result<Vec<Value>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
        return match value {
            Value::Array(rows) => Ok(rows),
            Value::Object(mut object) => {
                for key in ["lineage", "records", "members", "population"] {
                    if let Some(Value::Array(rows)) = object.remove(key) {
                        return Ok(rows);
                    }
                }
                Ok(vec![Value::Object(object)])
            }
            _ => bail!("lineage JSON must be an object or array"),
        };
    }
    let mut rows = Vec::new();
    for (index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        rows.push(
            serde_json::from_slice(line)
                .with_context(|| format!("parse lineage JSONL line {}", index + 1))?,
        );
    }
    Ok(rows)
}

fn lineage_value(
    path: &Path,
    champion: &str,
    task: Task,
    dataset: &str,
    run_id: &str,
) -> Result<Value> {
    validate_sha256(champion, "final champion semantic SHA-256")?;
    ensure!(!run_id.trim().is_empty(), "run identity must not be empty");
    let mut rows = Vec::new();
    let mut semantics = HashSet::new();
    for (index, value) in read_json_or_jsonl(path)?.into_iter().enumerate() {
        let row: LineageRow = serde_json::from_value(value)
            .with_context(|| format!("parse lineage record {index}"))?;
        validate_sha256(
            &row.semantic_sha256,
            &format!("lineage record {index} semantic SHA-256"),
        )?;
        if let Some(genome) = &row.canonical_genome {
            ensure!(
                sha256_bytes(genome.as_bytes()) == row.semantic_sha256,
                "lineage semantic identity drift at record {index}"
            );
        }
        ensure!(
            semantics.insert(row.semantic_sha256.clone()),
            "lineage repeats a semantic identity at record {index}"
        );
        for parent in row.all_parents() {
            validate_sha256(parent, "lineage parent semantic SHA-256")?;
        }
        rows.push(row);
    }
    ensure!(
        semantics.contains(champion),
        "final champion is absent from lineage"
    );
    let origins = rows
        .iter()
        .filter(|row| row.dl_origin)
        .map(|row| row.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    let mut contributed = origins.clone();
    loop {
        let before = contributed.len();
        for row in &rows {
            if row.all_parents().any(|parent| contributed.contains(parent)) {
                contributed.insert(row.semantic_sha256.clone());
            }
        }
        if contributed.len() == before {
            break;
        }
    }
    let descendants = contributed
        .difference(&origins)
        .cloned()
        .collect::<BTreeSet<_>>();
    let strict_winners = rows
        .iter()
        .filter(|row| row.strict_win && contributed.contains(&row.semantic_sha256))
        .map(|row| row.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    Ok(json!({
        "schema": LINEAGE_SCHEMA,
        "task": task.as_str(),
        "dataset_hash": dataset,
        "run_id": run_id,
        "lineage_sha256": sha256_file(path)?,
        "final_champion_semantic_sha256": champion,
        "dl_origin_parents": origins,
        "dl_origin_parent_count": origins.len(),
        "dl_origin_descendants": descendants,
        "dl_origin_descendant_count": descendants.len(),
        "dl_origin_strict_wins": strict_winners,
        "dl_origin_strict_win_count": strict_winners.len(),
        "final_champion_has_dl_contribution": contributed.contains(champion),
    }))
}

fn validate_receipt_chain(path: &Path) -> Result<Vec<Value>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    let mut previous: Option<String> = None;
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut value: Value = serde_json::from_str(&line)
            .with_context(|| format!("parse receipt line {}", index + 1))?;
        let object = value
            .as_object_mut()
            .context("receipt row must be a JSON object")?;
        ensure!(
            object.get("schema").and_then(Value::as_str) == Some(RECEIPT_SCHEMA),
            "unsupported receipt schema at line {}",
            index + 1
        );
        ensure!(
            object.get("sequence").and_then(Value::as_u64) == Some((rows.len() + 1) as u64),
            "receipt sequence drift at line {}",
            index + 1
        );
        let stored = object
            .remove("receipt_hash")
            .and_then(|value| value.as_str().map(str::to_owned))
            .context("receipt hash is absent")?;
        ensure!(
            object.get("previous_receipt_hash").and_then(Value::as_str) == previous.as_deref(),
            "receipt chain predecessor drift at line {}",
            index + 1
        );
        let actual = sha256_bytes(&canonical_json_bytes(&value)?);
        ensure!(
            stored == actual,
            "receipt content hash drift at line {}",
            index + 1
        );
        value
            .as_object_mut()
            .expect("object checked")
            .insert("receipt_hash".to_owned(), Value::String(stored.clone()));
        previous = Some(stored);
        rows.push(value);
    }
    Ok(rows)
}

fn durable_write(
    action: &str,
    request: Value,
    summary: Value,
    output: &Path,
    value: &Value,
    receipts: &Path,
    idempotency_key: &str,
) -> Result<Value> {
    ensure!(
        !idempotency_key.trim().is_empty(),
        "idempotency key must not be empty"
    );
    ensure!(
        output != receipts,
        "output and receipt log paths must differ"
    );
    let request_sha256 = sha256_bytes(&canonical_json_bytes(&request)?);
    let _lock = ReceiptLock::acquire(receipts)?;
    let existing = validate_receipt_chain(receipts)?;
    for receipt in &existing {
        if receipt.get("idempotency_key").and_then(Value::as_str) != Some(idempotency_key) {
            continue;
        }
        ensure!(
            receipt.get("action").and_then(Value::as_str) == Some(action)
                && receipt.get("request_sha256").and_then(Value::as_str) == Some(&request_sha256),
            "idempotency key was already used for different GP bridge inputs"
        );
        let expected = receipt
            .get("output_sha256")
            .and_then(Value::as_str)
            .context("idempotent receipt lacks output hash")?;
        ensure!(
            output.is_file() && sha256_file(output)? == expected,
            "idempotent output is absent or changed"
        );
        return Ok(receipt.clone());
    }

    let serialized = canonical_json_bytes(value)?;
    let output_sha256 = sha256_bytes(&serialized);
    if output.exists() {
        let existing_value = read_json(output)?;
        ensure!(
            canonical_json_bytes(&existing_value)? == serialized,
            "refusing to overwrite a nonidentical GP bridge output"
        );
    } else {
        atomic_json(output, value)?;
    }
    ensure!(
        sha256_file(output)? == sha256_bytes(&[serialized.as_slice(), b"\n"].concat()),
        "atomic output verification failed"
    );
    let output_file_sha256 = sha256_file(output)?;
    let previous = existing
        .last()
        .and_then(|row| row.get("receipt_hash"))
        .and_then(Value::as_str);
    let mut receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "sequence": existing.len() + 1,
        "created_at_ms": now_ms(),
        "action": action,
        "idempotency_key": idempotency_key,
        "request_sha256": request_sha256,
        "output": output,
        "output_sha256": output_file_sha256,
        "content_sha256": output_sha256,
        "previous_receipt_hash": previous,
    });
    if !summary.is_null() {
        receipt
            .as_object_mut()
            .expect("JSON object")
            .insert("summary".to_owned(), summary);
    }
    let receipt_hash = sha256_bytes(&canonical_json_bytes(&receipt)?);
    receipt
        .as_object_mut()
        .expect("JSON object")
        .insert("receipt_hash".to_owned(), Value::String(receipt_hash));
    if let Some(parent) = receipts.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(receipts)
        .with_context(|| format!("open receipt log {}", receipts.display()))?;
    file.write_all(&canonical_json_bytes(&receipt)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(receipt)
}

fn proposal_request(action: &str, args: &ProposalInputs, verified: &VerifiedProposals) -> Value {
    json!({
        "action": action,
        "checkpoint_pointer": args.checkpoint_pointer,
        "checkpoint_pointer_sha256": verified.checkpoint_pointer_sha256,
        "proposals_pointer": args.proposals_pointer,
        "proposals_pointer_sha256": verified.proposal_pointer_sha256,
        "proposal_file_sha256": verified.proposal_file_sha256,
        "checkpoint_sha256": verified.checkpoint_sha256,
        "task": args.task.as_str(),
        "dataset_hash": args.dataset,
    })
}

fn run(args: Vec<String>) -> Result<Value> {
    match Cli::parse_from(&args).command {
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

pub fn main(args: Vec<String>) -> Result<()> {
    match run(args) {
        Ok(receipt) => println!(
            "{}",
            serde_json::to_string(&receipt).expect("serialize receipt")
        ),
        Err(error) => {
            eprintln!("jope-gp-bridge: {error:#}");
            std::process::exit(1);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn digest(value: &str) -> String {
        sha256_bytes(value.as_bytes())
    }

    fn proposal(slot: usize, checkpoint: &str) -> GpProposal {
        let genome = format!("(pipe X (rff_cos n=32 seed={}) (ridge))", slot + 1);
        GpProposal {
            task: "regression".to_owned(),
            dataset_hash: "dataset".to_owned(),
            canonical_genome: genome.clone(),
            semantic_sha256: digest(&genome),
            model_checkpoint_sha256: checkpoint.to_owned(),
            slot,
            score: Some(slot as f64),
            proposal_score: None,
        }
    }

    fn member(name: &str, score: f64) -> GpPopulationMember {
        GpPopulationMember {
            canonical_genome: name.to_owned(),
            semantic_sha256: digest(name),
            score: Some(score),
            dl_origin: false,
            injected_checkpoint_sha256: None,
            ancestry: None,
        }
    }

    fn proposal_fixture(temporary: &TempDir) -> (ProposalInputs, String) {
        let checkpoint_artifact = temporary.path().join("checkpoint.model");
        fs::write(&checkpoint_artifact, b"checkpoint").unwrap();
        let checkpoint = sha256_file(&checkpoint_artifact).unwrap();
        let checkpoint_pointer = temporary.path().join("checkpoint.json");
        atomic_json(
            &checkpoint_pointer,
            &json!({
                "checkpoint_sha256": checkpoint,
                "checkpoint": checkpoint_artifact,
            }),
        )
        .unwrap();
        let proposal_file = temporary.path().join("proposals.jsonl");
        let mut file = File::create(&proposal_file).unwrap();
        for row in (0..64).map(|slot| proposal(slot, &checkpoint)) {
            file.write_all(&canonical_json_bytes(&serde_json::to_value(row).unwrap()).unwrap())
                .unwrap();
            file.write_all(b"\n").unwrap();
        }
        file.sync_all().unwrap();
        let proposal_pointer = temporary.path().join("proposals.json");
        atomic_json(
            &proposal_pointer,
            &json!({
                "checkpoint_sha256": checkpoint,
                "proposals": proposal_file,
                "proposals_sha256": sha256_file(&proposal_file).unwrap(),
            }),
        )
        .unwrap();
        (
            ProposalInputs {
                checkpoint_pointer,
                proposals_pointer: proposal_pointer,
                task: Task::Regression,
                dataset: "dataset".to_owned(),
            },
            checkpoint,
        )
    }

    #[test]
    fn export_verifies_checkpoint_artifact_pointer_and_all_64_slots() {
        let temporary = TempDir::new().unwrap();
        let (input, checkpoint) = proposal_fixture(&temporary);
        let verified = verify_proposals(&input).unwrap();
        assert_eq!(verified.checkpoint_sha256, checkpoint);
        assert_eq!(verified.proposals.len(), 64);
        assert_eq!(verified.proposals[0].slot, 0);
        assert_eq!(verified.proposals[63].slot, 63);
        assert_eq!(verified.task_invalid_skipped, 0);
        fs::write(&verified.checkpoint_artifact, b"changed").unwrap();
        assert!(verify_proposals(&input).is_err());
    }

    #[test]
    fn task_invalid_proposals_are_skipped_and_counted() {
        let temporary = TempDir::new().unwrap();
        let (input, checkpoint) = proposal_fixture(&temporary);
        let genome = "(pipe X (ridge))";
        let invalid = GpProposal {
            task: "binary".to_owned(),
            dataset_hash: "other".to_owned(),
            canonical_genome: genome.to_owned(),
            semantic_sha256: digest(genome),
            model_checkpoint_sha256: checkpoint.clone(),
            slot: 0,
            score: None,
            proposal_score: None,
        };
        let proposal_file = temporary.path().join("proposals.jsonl");
        let mut file = OpenOptions::new()
            .append(true)
            .open(&proposal_file)
            .unwrap();
        file.write_all(&canonical_json_bytes(&serde_json::to_value(&invalid).unwrap()).unwrap())
            .unwrap();
        file.write_all(b"\n").unwrap();
        file.sync_all().unwrap();
        atomic_json(
            &input.proposals_pointer,
            &json!({
                "checkpoint_sha256": checkpoint,
                "proposals": proposal_file,
                "proposals_sha256": sha256_file(&proposal_file).unwrap(),
            }),
        )
        .unwrap();
        let verified = verify_proposals(&input).unwrap();
        assert_eq!(verified.proposals.len(), 64);
        assert_eq!(verified.task_invalid_skipped, 1);
    }

    #[test]
    fn generation_zero_is_exact_budget_with_dl_and_robust_seeds() {
        let checkpoint = "a".repeat(64);
        let population = (0..80)
            .map(|slot| member(&format!("existing-{slot}"), slot as f64))
            .collect::<Vec<_>>();
        let proposals = (0..64)
            .map(|slot| proposal(slot, &checkpoint))
            .collect::<Vec<_>>();
        let robust = (0..20)
            .map(|slot| {
                let genome = format!("robust-{slot}");
                RobustSeed {
                    task: "regression".to_owned(),
                    semantic_sha256: digest(&genome),
                    canonical_genome: genome,
                    dataset_count: 3,
                    mean_within_dataset_percentile: 0.9 - slot as f64 / 100.0,
                    worst_within_dataset_percentile: 0.5,
                    strict_win_count: 1,
                }
            })
            .collect::<Vec<_>>();
        let value = merge_generation_zero(
            &population,
            &proposals,
            &checkpoint,
            &robust,
            12,
            Task::Regression,
            "dataset",
        )
        .unwrap();
        assert_eq!(value["population"].as_array().unwrap().len(), 80);
        assert_eq!(value["composition"]["current_checkpoint_dl"], 64);
        assert_eq!(
            value["composition"]["repeated_cross_dataset_gp_winners"],
            12
        );
        assert_eq!(value["composition"]["retained_existing"], 4);
        assert!(value["population"]
            .as_array()
            .unwrap()
            .iter()
            .take(76)
            .all(|member| member["score"].is_null()));
    }

    #[test]
    fn bridge_reads_protocol_scoped_robust_seeds_from_current_corpus() {
        let temporary = TempDir::new().unwrap();
        let corpus_path = temporary.path().join("corpus.sqlite");
        let connection = selfplay_ingest::open_corpus(&corpus_path).unwrap();
        let protocol = "9".repeat(64);
        let genome = "(pipe X (ridge))";
        let semantic = digest(genome);
        for (index, dataset) in ["dataset-a", "dataset-b"].into_iter().enumerate() {
            connection
                .execute(
                    "INSERT INTO observations(
                       task,dataset_hash,semantic_sha256,canonical_genome,candidate_source,
                       metric,score,runtime_seconds,train_sha256,test_sha256,
                       receipt_sha256,receipt_path,manifest_sha256,frozen_seed_sha256,
                       model_checkpoint_sha256,hall_of_fame_sha256,created_at_ms,
                       authority_kind,protocol_sha256,origin_class
                     ) VALUES(
                       'regression',?1,?2,?3,'dataset_local_elite',
                       'r2',?4,1.0,?5,?6,?7,'receipt',?8,?9,?10,?11,?12,
                       'corrected',?13,'gp_origin'
                     )",
                    rusqlite::params![
                        dataset,
                        semantic,
                        genome,
                        0.5 + index as f64 / 10.0,
                        "a".repeat(64),
                        "b".repeat(64),
                        "c".repeat(64),
                        "d".repeat(64),
                        "e".repeat(64),
                        "f".repeat(64),
                        "7".repeat(64),
                        index as i64,
                        protocol,
                    ],
                )
                .unwrap();
        }
        drop(connection);
        let (first, first_hash, first_value) =
            read_dynamic_robust_bank(&corpus_path, Task::Regression, &protocol, 10, 2).unwrap();
        let (second, second_hash, second_value) =
            read_dynamic_robust_bank(&corpus_path, Task::Regression, &protocol, 10, 2).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].semantic_sha256, semantic);
        assert_eq!(first[0].dataset_count, 2);
        assert_eq!(first_hash, second_hash);
        assert_eq!(first_value, second_value);
        assert_eq!(second.len(), 1);
    }

    #[test]
    fn checkpoint_injection_replaces_exactly_eight_non_elites() {
        let current = "b".repeat(64);
        let previous = "a".repeat(64);
        let population = (0..20)
            .map(|slot| member(&format!("existing-{slot}"), slot as f64))
            .collect::<Vec<_>>();
        let proposals = (0..64)
            .map(|slot| proposal(slot, &current))
            .collect::<Vec<_>>();
        let value = injection_value(
            &population,
            &proposals,
            &current,
            &previous,
            4,
            Task::Regression,
            "dataset",
        )
        .unwrap();
        assert_eq!(value["injection"]["injected"], 8);
        assert_eq!(
            &value["population"].as_array().unwrap()[..4],
            &serde_json::to_value(&population)
                .unwrap()
                .as_array()
                .unwrap()[..4]
        );
        for replacement in value["injection"]["replacements"].as_array().unwrap() {
            let slot = replacement["slot"].as_u64().unwrap() as usize;
            assert!(value["population"][slot]["score"].is_null());
        }
    }

    #[test]
    fn lineage_is_transitive_and_attributes_champion() {
        let temporary = TempDir::new().unwrap();
        let origin = digest("origin");
        let child = digest("child");
        let champion = digest("champion");
        let path = temporary.path().join("lineage.json");
        atomic_json(
            &path,
            &json!([
                {"semantic_sha256": origin, "dl_origin": true},
                {"semantic_sha256": child, "parents": [origin]},
                {"semantic_sha256": champion, "parents": [child], "strict_win": true}
            ]),
        )
        .unwrap();
        let value = lineage_value(&path, &champion, Task::Regression, "dataset", "run").unwrap();
        assert_eq!(value["dl_origin_parent_count"], 1);
        assert_eq!(value["dl_origin_descendant_count"], 2);
        assert_eq!(value["dl_origin_strict_win_count"], 1);
        assert_eq!(value["final_champion_has_dl_contribution"], true);
    }

    #[test]
    fn receipts_are_idempotent_and_hash_chained() {
        let temporary = TempDir::new().unwrap();
        let output = temporary.path().join("output.json");
        let receipts = temporary.path().join("receipts.jsonl");
        let first = durable_write(
            "test",
            json!({"input": 1}),
            Value::Null,
            &output,
            &json!({"output": 1}),
            &receipts,
            "stable",
        )
        .unwrap();
        let second = durable_write(
            "test",
            json!({"input": 1}),
            Value::Null,
            &output,
            &json!({"output": 1}),
            &receipts,
            "stable",
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(validate_receipt_chain(&receipts).unwrap().len(), 1);
        assert!(durable_write(
            "test",
            json!({"input": 2}),
            Value::Null,
            &output,
            &json!({"output": 1}),
            &receipts,
            "stable",
        )
        .is_err());
    }
}
