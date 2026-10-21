//! Rust-only continuous trainer admission for the self-play DL lane.
//!
//! Python is confined to the submitted PyTorch payload. This process only
//! validates immutable inputs, inspects SCQ, renders one deterministic spec,
//! and performs idempotent `check`/`submit` calls.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, default_scq_str, now_ms, read_json, scq_json, sha256_bytes,
    sha256_file,
};
use jope_dime_regressor::spec_policy::TRAINER;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

#[path = "common/flock.rs"]
mod flock;

const PROJECT: &str = "jope-selfplay-train";
const TRAINER_SPEC_CONTRACT: &str = "packed-v4-minimum-cpu";
/// The single schema string `ops/trainer-watchdog-v1.json` ever carries —
/// success or failure. See `CycleReport::degraded`.
const WATCHDOG_SCHEMA: &str = "JopeContinuousTrainerWatchdog.v1";
const REQUIRED_PRIORITY: f64 = 10.0;
const LABEL_STEP: u64 = 50_000;
const STALE_INTENT_CONFIRMATIONS: u8 = 3;
const LIVE_STATES: &[&str] = &[
    "queued",
    "starting",
    "dispatching",
    "running",
    "preempting",
    "held",
];
jope_paths::once_str!(
    fn default_trainer_str => paths::research_script("test_exact_train.py")
);
jope_paths::once_str!(
    fn default_common_str => paths::research_script("test_exact_common.py")
);
jope_paths::once_str!(
    fn default_packer_str => paths::research_script("test_exact_packed_corpus.py")
);
jope_paths::once_str!(
    fn default_prime_str => paths::runtime_bundle_pytorch()
);
jope_paths::once_str!(
    fn legacy_checkpoint_root_str => paths::legacy_checkpoints_root()
);
jope_paths::once_str!(
    fn default_runs_root_str => paths::scq_runs_root()
);
/// Prefixes of the trainer run directories whose `train-result.json` forms the
/// warm-start lineage. SCQ prunes finished jobs from `ls` long before the run
/// directory is collected (GC only touches `jte-`/`jcv-`), so the filesystem —
/// not the scheduler inventory — is the authority on completed trainers.
const LINEAGE_RUN_PREFIX: &str = "jope-selfplay-train-";
/// Parameter groups that must be present in a policy_v3.3 warm-start checkpoint.
/// These are exactly the 22 tensors the original 130-tensor anchor lacks.
const REQUIRED_CHECKPOINT_GROUPS: &[&str] = &[
    "binary_reward_heads.",
    "binary_routing_head.",
    "binary_stability_head.",
];
/// policy_v3.4 adds the reward branch's own context encoder (8 tensors: two Linear
/// weight/bias pairs and two LayerNorm weight/bias pairs), taking the full model to
/// 160 tensors / 32,240,687 parameters.
const REQUIRED_CHECKPOINT_GROUPS_V34: &[&str] = &[
    "binary_reward_heads.",
    "binary_routing_head.",
    "binary_stability_head.",
    "reward_encoder.",
];

/// One complete-model shape. A checkpoint may only be pinned into a successor spec
/// when it satisfies exactly one of these, because a short checkpoint warm-starts a
/// lobotomised network with no error from PyTorch.
///
/// `marker` makes the choice unambiguous rather than "whichever bar it clears":
/// a checkpoint that CONTAINS `reward_encoder.` is judged against the v3.4 bar
/// (160 tensors), so a truncated v3.4 archive can never squeak through on the v3.3
/// count. Schemas are ordered most-specific first; the last entry is the fallback.
struct CheckpointSchema {
    name: &'static str,
    marker: Option<&'static str>,
    tensors: usize,
    groups: &'static [&'static str],
}

const CHECKPOINT_SCHEMAS: &[CheckpointSchema] = &[
    CheckpointSchema {
        name: "policy_v3.4-split-context",
        marker: Some("reward_encoder."),
        tensors: 160,
        groups: REQUIRED_CHECKPOINT_GROUPS_V34,
    },
    CheckpointSchema {
        name: "policy_v3.3-fused-heads",
        marker: None,
        tensors: 152,
        groups: REQUIRED_CHECKPOINT_GROUPS,
    },
];

/// Fallback schema for archives that carry no schema marker (every checkpoint
/// written before the v3.4 un-sharing).
fn legacy_checkpoint_schema() -> &'static CheckpointSchema {
    CHECKPOINT_SCHEMAS
        .last()
        .expect("checkpoint schema table is never empty")
}

/// Pick the schema a checkpoint is claiming by the marker groups it carries.
fn resolve_checkpoint_schema(present: &[String]) -> &'static CheckpointSchema {
    CHECKPOINT_SCHEMAS
        .iter()
        .find(|schema| {
            schema
                .marker
                .is_some_and(|marker| present.iter().any(|group| group == marker))
        })
        .unwrap_or_else(|| legacy_checkpoint_schema())
}
/// Cheap-path lineage rescan throttle. The spec-rendering path always forces a
/// fresh scan, so a successor can never be pinned against a stale lineage.
const LINEAGE_RESCAN_MS: u128 = 60_000;

#[derive(Parser, Debug)]
#[command(
    name = "jope-selfplay-trainer-watchdog",
    about = "Keep exactly one priority-10 xbabe2 PyTorch trainer continuously present"
)]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    /// Permit SCQ check/submit. Without this flag the watchdog is audit-only.
    #[arg(long)]
    submit: bool,
    #[arg(long)]
    once: bool,
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..))]
    interval_seconds: u64,
    #[arg(long, default_value = "c", value_parser = ["a", "b", "c"])]
    recipe: String,
    #[arg(long, default_value_t = 20_260_726)]
    seed_base: u64,
    #[arg(long)]
    label_pointer: Option<PathBuf>,
    #[arg(long)]
    corpus_pointer: Option<PathBuf>,
    #[arg(long, default_value = default_trainer_str())]
    trainer_script: PathBuf,
    #[arg(long, default_value = default_common_str())]
    common_script: PathBuf,
    #[arg(long, default_value = default_packer_str())]
    packed_script: PathBuf,
    #[arg(long, default_value = default_prime_str())]
    prime_source: PathBuf,
    #[arg(long, default_value = default_runs_root_str())]
    runs_root: PathBuf,
    /// Resolve the warm-start checkpoint, print the decision as JSON and exit.
    /// Takes no lock and writes nothing: safe to run against the live root while
    /// the daemon owns it.
    #[arg(long)]
    explain_checkpoint: bool,
}

impl Args {
    fn label_pointer(&self) -> PathBuf {
        self.label_pointer
            .clone()
            .unwrap_or_else(|| self.root.join("ops/last-ingest.json"))
    }

    fn corpus_pointer(&self) -> PathBuf {
        self.corpus_pointer
            .clone()
            .unwrap_or_else(|| self.root.join("training/current-corpus.json"))
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct DurableState {
    schema: String,
    max_observed_labels: u64,
    max_observed_snapshot_watermark: u64,
    last_evidence_watermark: u64,
    next_sequence: u64,
    intent: Option<SubmissionIntent>,
    updated_at_ms: u128,
}

impl DurableState {
    fn new() -> Self {
        Self {
            schema: "JopeContinuousTrainerWatchdogState.v1".to_owned(),
            next_sequence: 1,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct SubmissionIntent {
    job_id: String,
    watermark: u64,
    sequence: u64,
    checkpoint_sha256: String,
    spec: PathBuf,
    spec_sha256: String,
    mode: String,
    seed: u64,
    created_at_ms: u128,
    last_error: Option<String>,
    checkpoint_mismatch_observations: u8,
}

#[derive(Clone, Debug)]
struct TrainerJob {
    id: String,
    watermark: u64,
    sequence: u64,
    state: String,
    project: String,
    tier: String,
    node: Option<String>,
    gpus: u64,
}

#[derive(Clone, Debug, Default)]
struct Checkpoint {
    path: PathBuf,
    sha256: String,
    /// Provenance of the choice: `accepted_pointer:<relative>`,
    /// `lineage:<job_id>` or `original_anchor`.
    origin: String,
    /// Parameter-storage count read from the torch archive index.
    tensors: usize,
    /// Required parameter groups the archive does not contain.
    missing_groups: Vec<String>,
    /// Name of the complete-model schema this archive is judged against.
    schema: &'static str,
    /// Tensor floor of that schema. Zero means "no schema resolved", which is
    /// never full — a defaulted Checkpoint must not pass the guard.
    required_tensors: usize,
    selected_epoch: Option<u64>,
    recipe: Option<String>,
    /// Completion time of the producing run (`train-result.json` mtime); 0 for
    /// pointer- and anchor-sourced checkpoints.
    completed_at_ms: u128,
}

impl Checkpoint {
    /// A checkpoint may only be pinned into a successor spec when it carries the
    /// whole model. Short tensor counts warm-start a lobotomised network without
    /// any error from PyTorch, so this is the load-bearing guard.
    fn is_full(&self) -> bool {
        self.missing_groups.is_empty()
            && self.required_tensors > 0
            && self.tensors >= self.required_tensors
    }

    fn deficiency(&self) -> Option<String> {
        if self.is_full() {
            return None;
        }
        Some(format!(
            "{} ({}) carries {} tensors, schema {} requires >= {}; missing groups [{}]",
            self.sha256,
            self.origin,
            self.tensors,
            self.schema,
            self.required_tensors,
            self.missing_groups.join(","),
        ))
    }

    fn describe(&self) -> Value {
        json!({
            "sha256": self.sha256,
            "path": self.path.to_string_lossy(),
            "origin": self.origin,
            "tensors": self.tensors,
            "schema": self.schema,
            "required_tensors": self.required_tensors,
            "missing_groups": self.missing_groups,
            "full": self.is_full(),
            "selected_checkpoint_epoch": self.selected_epoch,
            "recipe": self.recipe,
            "completed_at_ms": self.completed_at_ms as u64,
        })
    }
}

/// Tensor inventory of a torch `.pt` archive, read from the zip index only.
#[derive(Clone, Debug, Default)]
struct CheckpointShape {
    tensors: usize,
    missing_groups: Vec<String>,
    schema: &'static str,
    required_tensors: usize,
}

/// Cheap-path memo of the lineage scan. The scan itself is a single readdir over
/// ~75k run directories (~60-100 ms warm), which is wasteful at the 5 s cycle
/// interval but must never be stale when a successor is actually rendered.
#[derive(Debug, Default)]
struct LineageCache {
    runs: Vec<LineageRun>,
    scanned_at_ms: u128,
}

impl LineageCache {
    fn get(&mut self, runs_root: &Path, force: bool) -> &[LineageRun] {
        let stale = self
            .scanned_at_ms
            .checked_add(LINEAGE_RESCAN_MS)
            .map(|deadline| now_ms() >= deadline)
            .unwrap_or(true);
        if force || stale || self.scanned_at_ms == 0 {
            self.runs = scan_lineage(runs_root);
            self.scanned_at_ms = now_ms();
        }
        &self.runs
    }
}

/// One completed trainer run discovered on disk.
#[derive(Clone, Debug)]
struct LineageRun {
    job_id: String,
    checkpoint: Checkpoint,
}

#[derive(Clone, Debug)]
struct TrainingSnapshot {
    watermark: u64,
    manifest: PathBuf,
    packed_manifest: PathBuf,
    packed_catalog: PathBuf,
    packed_files: Vec<(String, PathBuf)>,
}

#[derive(Clone, Debug)]
struct SpecInput {
    name: String,
    source: PathBuf,
    target: String,
    sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContinuationKind {
    CoalescedLabels,
    RecentReplay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IntentCheckpointDisposition {
    Current,
    Confirming(u8),
    Supersede,
}

impl ContinuationKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::CoalescedLabels => "coalesced_50k_labels",
            Self::RecentReplay => "continuous_recent_plus_replay",
        }
    }
}

#[derive(Debug, Serialize)]
struct CycleReport {
    schema: &'static str,
    generated_at_ms: u128,
    mode: &'static str,
    continuous: bool,
    label_step: u64,
    labels: u64,
    snapshot_watermark: u64,
    pending_unfrozen_labels: u64,
    last_evidence_watermark: u64,
    next_label_threshold: u64,
    checkpoint_sha256: String,
    checkpoint_origin: String,
    checkpoint_tensors: usize,
    checkpoint_full: bool,
    lineage_full_runs: usize,
    family_priority: f64,
    priority_ready: bool,
    active_trainers: Vec<String>,
    action: String,
    planned_job_id: Option<String>,
    continuation_kind: Option<String>,
    submitted: bool,
    protected_work_mutations: u64,
    promotion_mutations: u64,
}

impl CycleReport {
    /// A failed cycle, expressed IN THE SAME SCHEMA as a successful one.
    ///
    /// The old error path wrote a five-key `JopeContinuousTrainerWatchdogError.v1`
    /// document over this file. That is the shape of defect that took the
    /// cluster down on 2026-07-28: the queue keeper does the same thing
    /// (jope_selfplay_queue.rs:273-282) and supply's strict `QueueStatus`
    /// struct — whose `queued_target` is not an `Option` — died on it 5,893
    /// times, so supply stopped supplying while the keeper was already stuck.
    /// A daemon's canonical status file is a CONTRACT with every consumer; a
    /// failing cycle is not permission to change the contract.
    ///
    /// So: same schema, every field present, and two unambiguous markers —
    /// `degraded: true` and `error` — that
    /// `jope_selfplay_alarms::check_status_contracts` raises as
    /// `status_degraded:ops/trainer-watchdog-v1.json`. Telemetry counters go to
    /// zero rather than carrying forward stale values, because a report that
    /// keeps the last good numbers while advancing `generated_at_ms` is exactly
    /// the "freshest possible file, no work being done" illusion this whole
    /// workstream exists to remove.
    ///
    /// Constructed as a struct literal on purpose: adding a field to
    /// `CycleReport` without deciding its degraded value is a compile error
    /// here, which is a local, loud failure instead of a silent one.
    fn degraded(mode: &'static str, error: &str) -> Value {
        let report = Self {
            schema: WATCHDOG_SCHEMA,
            generated_at_ms: now_ms(),
            mode,
            continuous: true,
            label_step: 0,
            labels: 0,
            snapshot_watermark: 0,
            pending_unfrozen_labels: 0,
            last_evidence_watermark: 0,
            next_label_threshold: 0,
            checkpoint_sha256: String::new(),
            checkpoint_origin: String::new(),
            checkpoint_tensors: 0,
            checkpoint_full: false,
            lineage_full_runs: 0,
            family_priority: 0.0,
            priority_ready: false,
            active_trainers: Vec::new(),
            action: "cycle_failed".to_owned(),
            planned_job_id: None,
            continuation_kind: None,
            submitted: false,
            protected_work_mutations: 0,
            promotion_mutations: 0,
        };
        let mut value =
            serde_json::to_value(&report).unwrap_or_else(|_| json!({ "schema": WATCHDOG_SCHEMA }));
        if let Some(object) = value.as_object_mut() {
            object.insert("degraded".to_owned(), Value::Bool(true));
            object.insert("error".to_owned(), Value::String(error.to_owned()));
            object.insert("admission_allowed".to_owned(), Value::Bool(false));
        }
        value
    }
}

struct ProcessLock {
    _file: File,
}

impl ProcessLock {
    fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        if flock::exclusive(&file, false).is_err() {
            bail!("another trainer watchdog owns {}", path.display());
        }
        Ok(Self { _file: file })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.explain_checkpoint {
        return explain_checkpoint(&args);
    }
    let ops = args.root.join("ops");
    fs::create_dir_all(&ops)?;
    let _lock = ProcessLock::acquire(&ops.join("trainer-watchdog.lock"))?;
    let mut lineage = LineageCache::default();
    loop {
        match cycle(&args, &mut lineage) {
            Ok(report) => {
                atomic_json(
                    &ops.join("trainer-watchdog-v1.json"),
                    &serde_json::to_value(&report)?,
                )?;
                println!("{}", serde_json::to_string(&report)?);
            }
            Err(error) => {
                // Degrade IN SCHEMA. Never swap the document shape under the
                // consumers — see CycleReport::degraded for why.
                let report = CycleReport::degraded(
                    if args.submit { "submit" } else { "audit" },
                    &format!("{error:#}"),
                );
                atomic_json(&ops.join("trainer-watchdog-v1.json"), &report)?;
                eprintln!("{}", serde_json::to_string(&report)?);
                if args.once {
                    return Err(error);
                }
            }
        }
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds));
    }
}

/// Read-only dry run of the warm-start decision. Takes no process lock and
/// writes nothing, so it can be exercised against the live root while the daemon
/// is running.
fn explain_checkpoint(args: &Args) -> Result<()> {
    let lineage = scan_lineage(&args.runs_root);
    let candidates = checkpoint_candidates(&args.root, &lineage);
    let selected = select_checkpoint(&candidates);
    let report = json!({
        "schema": "JopeContinuousTrainerWatchdogCheckpointPlan.v1",
        "generated_at_ms": now_ms() as u64,
        "runs_root": args.runs_root.to_string_lossy(),
        "checkpoint_schemas": CHECKPOINT_SCHEMAS
            .iter()
            .map(|schema| json!({
                "name": schema.name,
                "marker": schema.marker,
                "required_tensors": schema.tensors,
                "required_groups": schema.groups,
            }))
            .collect::<Vec<_>>(),
        "lineage_runs": lineage
            .iter()
            .map(|run| json!({"job_id": run.job_id, "checkpoint": run.checkpoint.describe()}))
            .collect::<Vec<_>>(),
        "candidates": candidates.iter().map(Checkpoint::describe).collect::<Vec<_>>(),
        "selected": selected.as_ref().ok().map(Checkpoint::describe),
        "error": selected.as_ref().err().map(|error| format!("{error:#}")),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    selected.map(|_| ())
}

fn cycle(args: &Args, lineage: &mut LineageCache) -> Result<CycleReport> {
    let state_path = args.root.join("ops/trainer-watchdog-state.v1.json");
    let mut state = load_state(&state_path)?;
    let labels = load_label_watermark(&args.label_pointer())?;
    ensure!(
        labels >= state.max_observed_labels,
        "label pointer regressed from {} to {}",
        state.max_observed_labels,
        labels
    );
    state.max_observed_labels = labels;

    let jobs = parse_jobs(&scq_json(&args.scq, &["ls"])?)?;
    let priority = parse_priority(&scq_json(&args.scq, &["priority", "ls"])?)?;
    let trainers = jobs
        .iter()
        .filter_map(parse_trainer_job)
        .collect::<Result<Vec<_>>>()?;
    let active = active_trainers(&trainers)?;

    let completed_watermark = trainers
        .iter()
        .filter(|job| job.state == "completed")
        .map(|job| job.watermark)
        .max()
        .unwrap_or(0);
    state.last_evidence_watermark = state.last_evidence_watermark.max(completed_watermark);
    let observed_sequence = trainers.iter().map(|job| job.sequence).max().unwrap_or(0);
    state.next_sequence = state.next_sequence.max(observed_sequence.saturating_add(1));

    if let Some(intent) = &state.intent {
        if let Some(job) = trainers.iter().find(|job| job.id == intent.job_id) {
            if job.state == "completed" {
                state.last_evidence_watermark = state.last_evidence_watermark.max(intent.watermark);
            }
            state.intent = None;
        }
    }

    let next_label_threshold = state.last_evidence_watermark.saturating_add(LABEL_STEP);

    if let Some(job) = active.first() {
        if let Some(intent) = state.intent.take() {
            append_event(
                &args.root,
                "intent_superseded_by_active_trainer",
                json!({
                    "job_id": intent.job_id,
                    "active_job_id": job.id,
                    "spec_sha256": intent.spec_sha256,
                }),
            )?;
        }
        // Do not repeatedly hash the large checkpoint and packed corpus while
        // an active trainer already blocks admission. Full content validation
        // is mandatory immediately before rendering or retrying a submission.
        let checkpoint = load_checkpoint_reference(&args.root, &args.runs_root, lineage)?;
        let lineage_full_runs = lineage
            .get(&args.runs_root, false)
            .iter()
            .filter(|run| run.checkpoint.is_full())
            .count();
        let snapshot_watermark = load_snapshot_watermark(&args.corpus_pointer(), labels)?;
        ensure!(
            snapshot_watermark >= state.max_observed_snapshot_watermark,
            "training corpus pointer regressed from {} to {}",
            state.max_observed_snapshot_watermark,
            snapshot_watermark
        );
        state.max_observed_snapshot_watermark = snapshot_watermark;
        state.updated_at_ms = now_ms();
        save_state(&state_path, &state)?;
        return Ok(CycleReport {
            schema: WATCHDOG_SCHEMA,
            generated_at_ms: now_ms(),
            mode: if args.submit { "submit" } else { "audit" },
            continuous: true,
            label_step: LABEL_STEP,
            labels,
            snapshot_watermark,
            pending_unfrozen_labels: labels.saturating_sub(snapshot_watermark),
            last_evidence_watermark: state.last_evidence_watermark,
            next_label_threshold,
            checkpoint_sha256: checkpoint.sha256.clone(),
            checkpoint_origin: checkpoint.origin.clone(),
            checkpoint_tensors: checkpoint.tensors,
            checkpoint_full: checkpoint.is_full(),
            lineage_full_runs,
            family_priority: priority,
            priority_ready: priority_is_ready(priority),
            active_trainers: vec![job.id.clone()],
            action: "active_trainer_preserved".to_owned(),
            planned_job_id: None,
            continuation_kind: None,
            submitted: false,
            protected_work_mutations: 0,
            promotion_mutations: 0,
        });
    }

    ensure!(
        priority_is_ready(priority),
        "{PROJECT} effective priority is {priority}, expected {REQUIRED_PRIORITY}"
    );
    let checkpoint = load_checkpoint(&args.root, &args.runs_root, lineage)?;
    let lineage_full_runs = lineage
        .get(&args.runs_root, false)
        .iter()
        .filter(|run| run.checkpoint.is_full())
        .count();
    let snapshot = load_training_snapshot(&args.corpus_pointer(), labels)?;
    ensure!(
        snapshot.watermark >= state.max_observed_snapshot_watermark,
        "training corpus pointer regressed from {} to {}",
        state.max_observed_snapshot_watermark,
        snapshot.watermark
    );
    state.max_observed_snapshot_watermark = snapshot.watermark;
    let pending_unfrozen_labels = labels.saturating_sub(snapshot.watermark);
    let kind = continuation_kind(snapshot.watermark, state.last_evidence_watermark);

    // Specs from an older contract either stage the 9+ GiB source JSONL files
    // or reserve CPU capacity that SCQ cannot currently admit despite an idle
    // GPU and bounded physical CPU headroom. Retire such an unsubmitted intent
    // explicitly and create a new immutable job identity below.
    if state
        .intent
        .as_ref()
        .is_some_and(intent_uses_stale_contract)
    {
        let stale = state
            .intent
            .take()
            .context("legacy trainer intent disappeared")?;
        append_event(
            &args.root,
            "intent_superseded_stale_spec_contract",
            json!({
                "job_id": stale.job_id,
                "spec_sha256": stale.spec_sha256,
                "checkpoint_sha256": stale.checkpoint_sha256,
            }),
        )?;
        state.updated_at_ms = now_ms();
        save_state(&state_path, &state)?;
    }

    let intent_checkpoint_disposition = state
        .intent
        .as_mut()
        .map(|intent| observe_intent_checkpoint(intent, &checkpoint.sha256))
        .unwrap_or(IntentCheckpointDisposition::Current);
    if let IntentCheckpointDisposition::Confirming(confirmation) = intent_checkpoint_disposition {
        state.updated_at_ms = now_ms();
        save_state(&state_path, &state)?;
        bail!(
            "pending trainer intent checkpoint is stale; confirmation {confirmation}/{STALE_INTENT_CONFIRMATIONS} before superseding confirmed-unsubmitted intent"
        );
    }
    if intent_checkpoint_disposition == IntentCheckpointDisposition::Supersede {
        let stale = state
            .intent
            .take()
            .context("stale trainer intent disappeared")?;
        append_event(
            &args.root,
            "intent_superseded_unsubmitted",
            json!({
                "job_id": stale.job_id,
                "old_checkpoint_sha256": stale.checkpoint_sha256,
                "current_checkpoint_sha256": checkpoint.sha256,
                "spec_sha256": stale.spec_sha256,
                "confirmation_samples": STALE_INTENT_CONFIRMATIONS,
            }),
        )?;
        state.updated_at_ms = now_ms();
        save_state(&state_path, &state)?;
    }

    let intent = match state.intent.clone() {
        Some(intent) => {
            validate_intent(&intent, &snapshot, &checkpoint)?;
            intent
        }
        None => {
            let sequence = state.next_sequence;
            let seed = args
                .seed_base
                .checked_add(sequence)
                .context("trainer seed overflow")?;
            let job_id = trainer_job_id(snapshot.watermark, &args.recipe, sequence);
            let spec = args.root.join("specs").join(format!("{job_id}.toml"));
            let content = render_spec(args, &job_id, seed, &snapshot, &checkpoint)?;
            let intent = SubmissionIntent {
                job_id,
                watermark: snapshot.watermark,
                sequence,
                checkpoint_sha256: checkpoint.sha256.clone(),
                spec,
                spec_sha256: sha256_bytes(content.as_bytes()),
                mode: kind.as_str().to_owned(),
                seed,
                created_at_ms: now_ms(),
                last_error: None,
                checkpoint_mismatch_observations: 0,
            };
            if args.submit {
                write_immutable(&intent.spec, content.as_bytes())?;
                state.intent = Some(intent.clone());
                state.next_sequence = sequence.saturating_add(1);
                state.updated_at_ms = now_ms();
                save_state(&state_path, &state)?;
                append_event(
                    &args.root,
                    "submission_intent",
                    json!({
                        "job_id": intent.job_id,
                        "watermark": intent.watermark,
                        "sequence": intent.sequence,
                        "checkpoint_sha256": intent.checkpoint_sha256,
                        "checkpoint": checkpoint.describe(),
                        "spec_sha256": intent.spec_sha256,
                        "mode": intent.mode,
                    }),
                )?;
            }
            intent
        }
    };

    let submitted = if args.submit {
        let content_hash = sha256_file(&intent.spec)?;
        ensure!(
            content_hash == intent.spec_sha256,
            "durable trainer spec content drift"
        );
        match scq_check_submit(&args.scq, &intent.spec) {
            Ok(()) => {
                append_event(
                    &args.root,
                    "submission_accepted",
                    json!({"job_id": intent.job_id, "spec_sha256": intent.spec_sha256}),
                )?;
                true
            }
            Err(error) => {
                if let Some(saved) = state.intent.as_mut() {
                    saved.last_error = Some(format!("{error:#}"));
                }
                state.updated_at_ms = now_ms();
                save_state(&state_path, &state)?;
                return Err(error);
            }
        }
    } else {
        false
    };
    state.updated_at_ms = now_ms();
    save_state(&state_path, &state)?;

    Ok(CycleReport {
        schema: WATCHDOG_SCHEMA,
        generated_at_ms: now_ms(),
        mode: if args.submit { "submit" } else { "audit" },
        continuous: true,
        label_step: LABEL_STEP,
        labels,
        snapshot_watermark: snapshot.watermark,
        pending_unfrozen_labels,
        last_evidence_watermark: state.last_evidence_watermark,
        next_label_threshold,
        checkpoint_sha256: checkpoint.sha256.clone(),
        checkpoint_origin: checkpoint.origin.clone(),
        checkpoint_tensors: checkpoint.tensors,
        checkpoint_full: checkpoint.is_full(),
        lineage_full_runs,
        family_priority: priority,
        priority_ready: true,
        active_trainers: Vec::new(),
        action: if submitted {
            "submitted_continuous_trainer"
        } else {
            "would_submit_continuous_trainer"
        }
        .to_owned(),
        planned_job_id: Some(intent.job_id.clone()),
        continuation_kind: Some(intent.mode),
        submitted,
        protected_work_mutations: 0,
        promotion_mutations: 0,
    })
}

fn load_state(path: &Path) -> Result<DurableState> {
    if !path.is_file() {
        return Ok(DurableState::new());
    }
    let state: DurableState = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        state.schema == "JopeContinuousTrainerWatchdogState.v1",
        "unsupported trainer watchdog state schema"
    );
    ensure!(state.next_sequence > 0, "trainer sequence must be positive");
    Ok(state)
}

fn save_state(path: &Path, state: &DurableState) -> Result<()> {
    atomic_json(path, &serde_json::to_value(state)?)
}

fn load_label_watermark(path: &Path) -> Result<u64> {
    let value = read_json(path)?;
    let labels = value
        .get("exact_rows_total")
        .or_else(|| value.get("watermark"))
        .and_then(Value::as_u64)
        .with_context(|| format!("{} lacks a nonnegative label watermark", path.display()))?;
    ensure!(labels > 0, "label watermark is zero");
    Ok(labels)
}

fn parse_jobs(value: &Value) -> Result<Vec<Value>> {
    value
        .get("Jobs")
        .and_then(|value| value.get("jobs"))
        .and_then(Value::as_array)
        .cloned()
        .context("SCQ ls JSON lacks Jobs.jobs")
}

fn parse_priority(value: &Value) -> Result<f64> {
    value
        .get("FamilyPriorities")
        .and_then(|value| value.get("families"))
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| row.get("family").and_then(Value::as_str) == Some(PROJECT))
        })
        .and_then(|row| row.get("effective_weight"))
        .and_then(Value::as_f64)
        .context("SCQ priority inventory lacks jope-selfplay-train")
}

fn priority_is_ready(value: f64) -> bool {
    value.is_finite() && (value - REQUIRED_PRIORITY).abs() < 1e-9
}

fn parse_trainer_job(value: &Value) -> Option<Result<TrainerJob>> {
    let id = value.get("id").and_then(Value::as_str)?;
    if !id.starts_with("jope-selfplay-train-") && !id.starts_with("jope-testonly-train-") {
        return None;
    }
    let parsed = parse_trainer_id(id);
    Some((|| {
        let (watermark, sequence) =
            parsed.with_context(|| format!("malformed trainer job id {id}"))?;
        Ok(TrainerJob {
            id: id.to_owned(),
            watermark,
            sequence,
            state: value
                .get("state")
                .and_then(Value::as_str)
                .context("trainer job lacks state")?
                .to_owned(),
            project: value
                .get("project")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            tier: value
                .get("tier")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            node: value.get("node").and_then(Value::as_str).map(str::to_owned),
            gpus: value.get("gpus").and_then(Value::as_u64).unwrap_or(0),
        })
    })())
}

fn active_trainers(trainers: &[TrainerJob]) -> Result<Vec<&TrainerJob>> {
    let active = trainers
        .iter()
        .filter(|job| LIVE_STATES.contains(&job.state.as_str()))
        .collect::<Vec<_>>();
    ensure!(
        active.len() <= 1,
        "duplicate active trainers: {}",
        active
            .iter()
            .map(|job| job.id.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    for job in &active {
        validate_active_trainer(job)?;
    }
    Ok(active)
}

fn parse_trainer_id(id: &str) -> Option<(u64, u64)> {
    let suffix = id
        .strip_prefix("jope-selfplay-train-")
        .or_else(|| id.strip_prefix("jope-testonly-train-"))?;
    let mut fields = suffix.split('-');
    let watermark = fields.next()?.parse().ok()?;
    let sequence = fields
        .find_map(|field| field.strip_prefix('r').and_then(|row| row.parse().ok()))
        .unwrap_or(0);
    Some((watermark, sequence))
}

fn validate_active_trainer(job: &TrainerJob) -> Result<()> {
    ensure!(
        matches!(job.project.as_str(), PROJECT | "jope-testonly"),
        "active trainer {} has unexpected project {}",
        job.id,
        job.project
    );
    ensure!(
        job.tier == "normal",
        "active trainer {} has unexpected tier {}",
        job.id,
        job.tier
    );
    ensure!(
        job.gpus == 1,
        "active trainer {} does not reserve exactly one GPU",
        job.id
    );
    if matches!(job.state.as_str(), "running" | "dispatching") {
        ensure!(
            job.node.as_deref() == Some("xbabe2"),
            "active trainer {} is not on xbabe2",
            job.id
        );
    }
    Ok(())
}

/// Tensor inventory of a torch `.pt` archive without loading a single tensor and
/// without touching CUDA: `<prefix>/data/<n>` holds exactly one storage per saved
/// tensor, and `<prefix>/data.pkl` carries the state-dict keys as plain ASCII
/// pickle strings. Cost is the zip central directory plus a ~100 KiB pickle.
fn checkpoint_shape(path: &Path) -> Result<CheckpointShape> {
    let file = File::open(path).with_context(|| format!("open checkpoint {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("checkpoint is not a torch archive: {}", path.display()))?;
    let names = archive
        .file_names()
        .map(|name| name.map(|name| name.into_owned()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut tensors = 0usize;
    let mut pickle = None;
    for name in &names {
        if let Some((_, index)) = name.rsplit_once("/data/") {
            if !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()) {
                tensors += 1;
            }
        }
        if name.ends_with("data.pkl") {
            pickle = Some(name.clone());
        }
    }
    let pickle =
        pickle.with_context(|| format!("checkpoint lacks data.pkl: {}", path.display()))?;
    let mut bytes = Vec::new();
    archive
        .by_name(&pickle)?
        .read_to_end(&mut bytes)
        .with_context(|| format!("read {pickle} from {}", path.display()))?;
    // Every group name any schema cares about, resolved once against the pickle.
    let mut present = Vec::new();
    for schema in CHECKPOINT_SCHEMAS {
        for group in schema.groups.iter().chain(schema.marker.iter()) {
            if !present.iter().any(|seen: &String| seen == group)
                && contains_subslice(&bytes, group.as_bytes())
            {
                present.push((*group).to_owned());
            }
        }
    }
    let schema = resolve_checkpoint_schema(&present);
    let missing_groups = schema
        .groups
        .iter()
        .filter(|group| !present.iter().any(|seen| seen == *group))
        .map(|group| (*group).to_owned())
        .collect::<Vec<_>>();
    Ok(CheckpointShape {
        tensors,
        missing_groups,
        schema: schema.name,
        required_tensors: schema.tensors,
    })
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack.len() >= needle.len()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

fn probe_checkpoint(path: PathBuf, sha256: String, origin: String) -> Result<Checkpoint> {
    ensure!(
        path.is_file(),
        "required checkpoint is absent: {}",
        path.display()
    );
    let shape = checkpoint_shape(&path)?;
    Ok(Checkpoint {
        path,
        sha256,
        origin,
        tensors: shape.tensors,
        missing_groups: shape.missing_groups,
        schema: shape.schema,
        required_tensors: shape.required_tensors,
        ..Checkpoint::default()
    })
}

fn mtime_ms(path: &Path) -> u128 {
    path.metadata()
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|delta| delta.as_millis())
        .unwrap_or(0)
}

/// Every completed trainer run on disk, newest completion first.
///
/// A run is a candidate when its `train-result.json` names a checkpoint that is
/// still present. Unreadable or vanished candidates are skipped rather than
/// failing the scan: one corrupt historical run must not wedge admission.
fn scan_lineage(runs_root: &Path) -> Vec<LineageRun> {
    let Ok(entries) = fs::read_dir(runs_root) else {
        return Vec::new();
    };
    let mut runs = Vec::new();
    for entry in entries.flatten() {
        let Ok(job_id) = entry.file_name().into_string() else {
            continue;
        };
        if !job_id.starts_with(LINEAGE_RUN_PREFIX) {
            continue;
        }
        let Ok(attempts) = fs::read_dir(entry.path().join("attempts")) else {
            continue;
        };
        for attempt in attempts.flatten() {
            let Ok(watermarks) = fs::read_dir(attempt.path().join("outputs/training")) else {
                continue;
            };
            for watermark in watermarks.flatten() {
                let result = watermark.path().join("train-result.json");
                if !result.is_file() {
                    continue;
                }
                if let Some(run) = lineage_run(&job_id, &result) {
                    runs.push(run);
                }
            }
        }
    }
    runs.sort_by(|left, right| {
        right
            .checkpoint
            .completed_at_ms
            .cmp(&left.checkpoint.completed_at_ms)
            .then_with(|| right.job_id.cmp(&left.job_id))
    });
    runs
}

fn lineage_run(job_id: &str, result: &Path) -> Option<LineageRun> {
    let value = read_json(result).ok()?;
    let sha256 = required_sha256(&value, "checkpoint_sha256").ok()?;
    // `checkpoint_path` is the digest-named artefact; `checkpoint` is the mutable
    // best.model.pt alias, so only the former may be pinned.
    let named = value
        .get("checkpoint_path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .filter(|path| path.is_file());
    let path = named.or_else(|| {
        let sibling = result.parent()?.join(format!("{sha256}.model.pt"));
        sibling.is_file().then_some(sibling)
    })?;
    let shape = checkpoint_shape(&path).ok()?;
    Some(LineageRun {
        job_id: job_id.to_owned(),
        checkpoint: Checkpoint {
            path,
            sha256,
            origin: format!("lineage:{job_id}"),
            tensors: shape.tensors,
            missing_groups: shape.missing_groups,
            schema: shape.schema,
            required_tensors: shape.required_tensors,
            selected_epoch: value
                .get("selected_checkpoint_epoch")
                .and_then(Value::as_u64),
            recipe: value
                .get("recipe")
                .and_then(Value::as_str)
                .map(str::to_owned),
            completed_at_ms: mtime_ms(result),
        },
    })
}

/// Ordered warm-start candidates: accepted pointer, then completed lineage runs
/// newest first, then the original anchor.
///
/// Before this ordering existed the watchdog fell straight through to the anchor
/// whenever `training/accepted.json` was absent, which is how successor
/// `c-r000002` was auto-submitted warm-starting the 130-tensor fbcd5459 and
/// silently re-randomising the binary branch.
fn checkpoint_candidates(root: &Path, lineage: &[LineageRun]) -> Vec<Checkpoint> {
    let mut candidates = Vec::new();
    for relative in ["training/learning.json", "training/accepted.json"] {
        let pointer = root.join(relative);
        if !pointer.is_file() {
            continue;
        }
        let Ok(value) = read_json(&pointer) else {
            continue;
        };
        let (Some(path), Ok(digest)) = (
            value.get("checkpoint").and_then(Value::as_str),
            required_sha256(&value, "checkpoint_sha256"),
        ) else {
            continue;
        };
        if let Ok(checkpoint) = probe_checkpoint(
            PathBuf::from(path),
            digest,
            format!("accepted_pointer:{relative}"),
        ) {
            candidates.push(checkpoint);
        }
    }
    candidates.extend(lineage.iter().map(|run| run.checkpoint.clone()));
    if let Ok(value) = read_json(&root.join("proposals/current.json")) {
        if value.get("legacy_adopted").and_then(Value::as_bool) == Some(true) {
            if let Ok(digest) = required_sha256(&value, "checkpoint_sha256") {
                let path =
                    PathBuf::from(legacy_checkpoint_root_str()).join(format!("{digest}.model.pt"));
                if let Ok(checkpoint) = probe_checkpoint(path, digest, "original_anchor".to_owned())
                {
                    candidates.push(checkpoint);
                }
            }
        }
    }
    let mut seen = Vec::new();
    candidates.retain(|candidate| {
        if seen.contains(&candidate.sha256) {
            return false;
        }
        seen.push(candidate.sha256.clone());
        true
    });
    candidates
}

/// Pick the first candidate that carries the whole model.
///
/// Deficient candidates are never pinned, no matter how preferred they are: a
/// short checkpoint would train a lobotomised model with no error anywhere.
fn select_checkpoint(candidates: &[Checkpoint]) -> Result<Checkpoint> {
    ensure!(
        !candidates.is_empty(),
        "no warm-start checkpoint is present"
    );
    candidates
        .iter()
        .find(|candidate| candidate.is_full())
        .cloned()
        .with_context(|| {
            format!(
                "every warm-start candidate is tensor-deficient: {}",
                candidates
                    .iter()
                    .filter_map(Checkpoint::deficiency)
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        })
}

/// Cheap resolution used while an active trainer already blocks admission: no
/// 125 MiB rehash, and the lineage scan may be up to `LINEAGE_RESCAN_MS` stale.
fn load_checkpoint_reference(
    root: &Path,
    runs_root: &Path,
    cache: &mut LineageCache,
) -> Result<Checkpoint> {
    let lineage = cache.get(runs_root, false).to_vec();
    let candidates = checkpoint_candidates(root, &lineage);
    let selected = select_checkpoint(&candidates);
    publish_warm_start_guard(root, &candidates, selected.as_ref().ok());
    selected
}

/// Full resolution used immediately before rendering or retrying a submission:
/// forces a fresh lineage scan and re-verifies the pinned bytes.
fn load_checkpoint(root: &Path, runs_root: &Path, cache: &mut LineageCache) -> Result<Checkpoint> {
    let lineage = cache.get(runs_root, true).to_vec();
    let candidates = checkpoint_candidates(root, &lineage);
    let selected = select_checkpoint(&candidates);
    publish_warm_start_guard(root, &candidates, selected.as_ref().ok());
    let checkpoint = selected?;
    verify_file(&checkpoint.path, &checkpoint.sha256)?;
    // Belt and braces: the guard already filtered on shape, but a spec must never
    // leave this process pinning a deficient checkpoint.
    ensure!(
        checkpoint.is_full(),
        "refusing to pin a tensor-deficient warm-start checkpoint: {}",
        checkpoint.deficiency().unwrap_or_default()
    );
    Ok(checkpoint)
}

/// Publish the warm-start guard verdict for `jope_selfplay_alarms` to lift into
/// `ops/health-alarms.v1.json`. Written on every resolution so the alarm clears
/// itself once a full checkpoint reappears. Never fails the cycle.
fn publish_warm_start_guard(root: &Path, candidates: &[Checkpoint], selected: Option<&Checkpoint>) {
    let deficient = selected.is_none();
    let report = json!({
        "schema": "JopeTrainerWarmStartGuard.v1",
        "generated_at_ms": now_ms() as u64,
        "alarm": "trainer_warm_start_deficient",
        "deficient": deficient,
        "severity": if deficient { "critical" } else { "ok" },
        "observed": selected.map_or_else(
            || format!("{} candidate(s), none full", candidates.len()),
            |checkpoint| format!(
                "{} tensors from {} ({})",
                checkpoint.tensors,
                checkpoint.origin,
                &checkpoint.sha256[..12.min(checkpoint.sha256.len())],
            ),
        ),
        "threshold": CHECKPOINT_SCHEMAS
            .iter()
            .map(|schema| format!(
                "{} (>={} tensors including {})",
                schema.name,
                schema.tensors,
                schema.groups.join(", "),
            ))
            .collect::<Vec<_>>()
            .join(" | "),
        "selected": selected.map(Checkpoint::describe),
        "candidates": candidates.iter().map(Checkpoint::describe).collect::<Vec<_>>(),
    });
    let _ = atomic_json(&root.join("ops/trainer-warm-start-guard.v1.json"), &report);
}

fn load_snapshot_watermark(pointer_path: &Path, labels: u64) -> Result<u64> {
    let pointer = read_json(pointer_path)?;
    let watermark = pointer
        .get("watermark")
        .and_then(Value::as_u64)
        .context("training corpus pointer lacks watermark")?;
    ensure!(
        watermark <= labels,
        "training corpus watermark {watermark} exceeds label pointer {labels}"
    );
    Ok(watermark)
}

fn load_training_snapshot(pointer_path: &Path, labels: u64) -> Result<TrainingSnapshot> {
    let pointer = read_json(pointer_path)?;
    let watermark = load_snapshot_watermark(pointer_path, labels)?;
    let manifest = PathBuf::from(
        pointer
            .get("manifest")
            .and_then(Value::as_str)
            .context("training corpus pointer lacks manifest")?,
    );
    verify_file(&manifest, &required_sha256(&pointer, "manifest_sha256")?)?;
    let value = read_json(&manifest)?;
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some("JopeTestOnlyTrainingCorpus.v1"),
        "training corpus schema mismatch"
    );
    ensure!(
        value.get("source_family_disjoint").and_then(Value::as_bool) == Some(true),
        "training corpus is not family-disjoint"
    );
    ensure!(
        value.get("source_label_watermark").and_then(Value::as_u64) == Some(watermark),
        "training manifest watermark mismatch"
    );
    ensure!(
        value
            .pointer("/strict_filter/test_adaptive")
            .and_then(Value::as_bool)
            == Some(true)
            && value
                .pointer("/strict_filter/benchmark_valid")
                .and_then(Value::as_bool)
                == Some(false),
        "training corpus authority mismatch"
    );
    let packed_manifest = PathBuf::from(
        pointer
            .get("packed_manifest")
            .and_then(Value::as_str)
            .context("training pointer lacks packed manifest")?,
    );
    verify_file(
        &packed_manifest,
        &required_sha256(&pointer, "packed_manifest_sha256")?,
    )?;
    let packed = read_json(&packed_manifest)?;
    ensure!(
        packed.get("source_manifest_sha256").and_then(Value::as_str)
            == pointer.get("manifest_sha256").and_then(Value::as_str),
        "packed corpus source manifest binding mismatch"
    );
    for key in ["train_sha256", "validation_sha256"] {
        ensure!(
            packed.get(key).and_then(Value::as_str) == value.get(key).and_then(Value::as_str),
            "packed corpus {key} binding mismatch"
        );
    }
    let packed_catalog = verified_path(&packed, "catalog_data", "catalog_data_sha256")?;
    let files = packed
        .get("files")
        .and_then(Value::as_object)
        .context("packed manifest lacks files")?;
    let mut packed_files = Vec::with_capacity(files.len());
    for (name, metadata) in files {
        ensure_safe_filename(name)?;
        let digest = required_sha256(metadata, "sha256")?;
        let path = packed_manifest
            .parent()
            .context("packed manifest has no parent")?
            .join(name);
        verify_file(&path, &digest)?;
        packed_files.push((name.clone(), path));
    }
    packed_files.sort_by(|left, right| left.0.cmp(&right.0));
    ensure!(!packed_files.is_empty(), "packed training corpus is empty");
    Ok(TrainingSnapshot {
        watermark,
        manifest,
        packed_manifest,
        packed_catalog,
        packed_files,
    })
}

fn required_sha256(value: &Value, field: &str) -> Result<String> {
    let digest = value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("missing {field}"))?;
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{field} is not a lowercase SHA-256"
    );
    Ok(digest.to_owned())
}

fn verified_path(value: &Value, path_field: &str, hash_field: &str) -> Result<PathBuf> {
    let path = PathBuf::from(
        value
            .get(path_field)
            .and_then(Value::as_str)
            .with_context(|| format!("missing {path_field}"))?,
    );
    verify_file(&path, &required_sha256(value, hash_field)?)?;
    Ok(path)
}

fn verify_file(path: &Path, digest: &str) -> Result<()> {
    ensure!(
        path.is_file(),
        "required file is absent: {}",
        path.display()
    );
    ensure!(
        sha256_file(path)? == digest,
        "content hash mismatch: {}",
        path.display()
    );
    Ok(())
}

fn ensure_safe_filename(value: &str) -> Result<()> {
    let path = Path::new(value);
    ensure!(
        path.components().count() == 1
            && matches!(path.components().next(), Some(Component::Normal(_))),
        "unsafe packed filename {value}"
    );
    Ok(())
}

fn continuation_kind(watermark: u64, last_evidence: u64) -> ContinuationKind {
    if watermark >= last_evidence.saturating_add(LABEL_STEP) {
        ContinuationKind::CoalescedLabels
    } else {
        ContinuationKind::RecentReplay
    }
}

fn observe_intent_checkpoint(
    intent: &mut SubmissionIntent,
    current_checkpoint_sha256: &str,
) -> IntentCheckpointDisposition {
    if intent.checkpoint_sha256 == current_checkpoint_sha256 {
        intent.checkpoint_mismatch_observations = 0;
        return IntentCheckpointDisposition::Current;
    }
    intent.checkpoint_mismatch_observations =
        intent.checkpoint_mismatch_observations.saturating_add(1);
    if intent.checkpoint_mismatch_observations >= STALE_INTENT_CONFIRMATIONS {
        IntentCheckpointDisposition::Supersede
    } else {
        IntentCheckpointDisposition::Confirming(intent.checkpoint_mismatch_observations)
    }
}

fn trainer_job_id(watermark: u64, recipe: &str, sequence: u64) -> String {
    format!("jope-selfplay-train-{watermark:09}-{recipe}-r{sequence:06}")
}

fn validate_intent(
    intent: &SubmissionIntent,
    snapshot: &TrainingSnapshot,
    checkpoint: &Checkpoint,
) -> Result<()> {
    ensure!(
        intent.watermark <= snapshot.watermark,
        "pending trainer intent uses a future corpus snapshot"
    );
    ensure!(
        intent.checkpoint_sha256 == checkpoint.sha256,
        "pending trainer intent uses a superseded checkpoint"
    );
    ensure!(
        parse_trainer_id(&intent.job_id) == Some((intent.watermark, intent.sequence)),
        "pending trainer intent identity mismatch"
    );
    Ok(())
}

fn render_spec(
    args: &Args,
    job_id: &str,
    seed: u64,
    snapshot: &TrainingSnapshot,
    checkpoint: &Checkpoint,
) -> Result<String> {
    let trainer_source = args.prime_source.join("jope/policy/train_v3.py");
    let mut inputs = vec![
        spec_input(
            "script",
            &args.trainer_script,
            "scripts/test_exact_train.py",
        )?,
        spec_input(
            "common",
            &args.common_script,
            "scripts/test_exact_common.py",
        )?,
        spec_input(
            "packed_source",
            &args.packed_script,
            "scripts/test_exact_packed_corpus.py",
        )?,
        spec_input(
            "manifest",
            &snapshot.manifest,
            "corpus/corpus-manifest.json",
        )?,
        spec_input(
            "checkpoint",
            &checkpoint.path,
            &format!("checkpoint/{}.model.pt", checkpoint.sha256),
        )?,
        spec_input("trainer_source", &trainer_source, "provenance/train_v3.py")?,
        spec_input(
            "packed_manifest",
            &snapshot.packed_manifest,
            "packed/manifest.json",
        )?,
        spec_input(
            "packed_catalog",
            &snapshot.packed_catalog,
            "packed/catalog-data.f32le",
        )?,
    ];
    for (index, (name, path)) in snapshot.packed_files.iter().enumerate() {
        inputs.push(spec_input(
            &format!("packed_{index}"),
            path,
            &format!("packed/{name}"),
        )?);
    }

    let output_name = format!("watermark-{:09}", snapshot.watermark);
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = {}", toml_string(job_id)?),
        format!("project = \"{PROJECT}\""),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/python3\"".to_owned(),
        format!(
            "args = [\"{{input:script}}\", \"train\", \"--corpus-manifest\", \
             \"{{input:manifest}}\", \"--output\", \"{{output:training}}/{output_name}\", \
             \"--device\", \"cuda\", \"--warm\", \"{{input:checkpoint}}\", \"--recipe\", \
             {}, \"--seed\", \"{seed}\", \"--packed-manifest\", \"{{input:packed_manifest}}\", \
             \"--packed-catalog-data\", \"{{input:packed_catalog}}\"]",
            toml_string(&args.recipe)?,
        ),
        String::new(),
    ];
    lines.extend(TRAINER.render_resource_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend(TRAINER.render_placement_block().lines().map(str::to_owned));
    lines.extend([
        String::new(),
        "[env]".to_owned(),
        format!(
            "PYTHONPATH = {}",
            toml_string(&args.prime_source.to_string_lossy())?
        ),
        format!("JOPE_TRAINER_SPEC_CONTRACT = \"{TRAINER_SPEC_CONTRACT}\""),
        format!("OMP_NUM_THREADS = \"{}\"", TRAINER.omp_threads),
        String::new(),
        "[preempt]".to_owned(),
        format!("preemptible = {}", TRAINER.preemptible),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
    ]);
    lines.extend(TRAINER.render_limits_block().lines().map(str::to_owned));
    lines.push(String::new());
    let mut text = lines.join("\n");
    text.push('\n');
    for input in inputs {
        text.push_str(&format!(
            "[[input]]\nname = {}\nsource = {}\ntarget = {}\nsha256 = {}\n\n",
            toml_string(&input.name)?,
            toml_string(&input.source.to_string_lossy())?,
            toml_string(&input.target)?,
            toml_string(&input.sha256)?,
        ));
    }
    text.push_str("[[output]]\nname = \"training\"\nsource = \"training\"\ndirectory = true\n");
    Ok(text)
}

fn spec_input(name: &str, source: &Path, target: &str) -> Result<SpecInput> {
    ensure!(
        source.is_file(),
        "spec input is absent: {}",
        source.display()
    );
    Ok(SpecInput {
        name: name.to_owned(),
        source: source.to_owned(),
        target: target.to_owned(),
        sha256: sha256_file(source)?,
    })
}

fn toml_string(value: &str) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(
            sha256_file(path)? == sha256_bytes(bytes),
            "existing trainer spec content drift: {}",
            path.display()
        );
        return Ok(());
    }
    let parent = path.parent().context("trainer spec has no parent")?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("trainer spec has no filename")?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn append_event(root: &Path, kind: &str, detail: Value) -> Result<()> {
    let path = root.join("ops/trainer-watchdog-events.v1.jsonl");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)?;
    serde_json::to_writer(
        &mut file,
        &json!({
            "schema": "JopeContinuousTrainerWatchdogEvent.v1",
            "kind": kind,
            "detail": detail,
            "created_at_ms": now_ms(),
        }),
    )?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}

fn scq_check_submit(scq: &Path, spec: &Path) -> Result<()> {
    let check = Command::new(scq)
        .arg("check")
        .arg(spec)
        .output()
        .with_context(|| format!("run {} check", scq.display()))?;
    let check_message = String::from_utf8_lossy(if check.stderr.is_empty() {
        &check.stdout
    } else {
        &check.stderr
    });
    if !check.status.success() && !temporarily_lacks_launch_capacity(&check_message) {
        bail!("SCQ check rejected trainer: {}", check_message.trim());
    }
    let submit = Command::new(scq)
        .arg("submit")
        .arg(spec)
        .output()
        .with_context(|| format!("run {} submit", scq.display()))?;
    let message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    if !submit.status.success() && !message.to_ascii_lowercase().contains("already exists") {
        bail!("SCQ submit rejected trainer: {}", message.trim());
    }
    Ok(())
}

fn temporarily_lacks_launch_capacity(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && message.contains("no eligible launch-ready node currently")
}

fn intent_uses_stale_contract(intent: &SubmissionIntent) -> bool {
    fs::read_to_string(&intent.spec)
        .map(|text| {
            !text.contains(&format!(
                "JOPE_TRAINER_SPEC_CONTRACT = \"{TRAINER_SPEC_CONTRACT}\""
            )) || text.contains("--packed-source-train")
                || text.contains("target = \"corpus/corpus.train.jsonl\"")
        })
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// A failed cycle must not change the document's shape. Consumers read this
    /// file with structs built against the success shape; the old error
    /// document dropped 19 of 24 keys, which is the same defect that killed
    /// jope-selfplay-supply 5,893 times on 2026-07-28.
    #[test]
    fn a_failed_cycle_degrades_in_schema() {
        let degraded = CycleReport::degraded("submit", "scq unreachable: connection refused");
        assert_eq!(degraded["schema"], WATCHDOG_SCHEMA);
        assert_eq!(degraded["degraded"], true);
        assert_eq!(degraded["error"], "scq unreachable: connection refused");
        assert_eq!(degraded["action"], "cycle_failed");
        assert_eq!(degraded["submitted"], false);

        // Every key a successful report carries is still present.
        let healthy = serde_json::to_value(CycleReport {
            schema: WATCHDOG_SCHEMA,
            generated_at_ms: 1,
            mode: "submit",
            continuous: true,
            label_step: LABEL_STEP,
            labels: 2,
            snapshot_watermark: 3,
            pending_unfrozen_labels: 4,
            last_evidence_watermark: 5,
            next_label_threshold: 6,
            checkpoint_sha256: "abc".into(),
            checkpoint_origin: "lineage:x".into(),
            checkpoint_tensors: 152,
            checkpoint_full: true,
            lineage_full_runs: 10,
            family_priority: 10.0,
            priority_ready: true,
            active_trainers: vec!["run".into()],
            action: "active_trainer_preserved".into(),
            planned_job_id: None,
            continuation_kind: None,
            submitted: false,
            protected_work_mutations: 0,
            promotion_mutations: 0,
        })
        .unwrap();
        for key in healthy.as_object().unwrap().keys() {
            assert!(degraded.get(key).is_some(), "degraded report dropped {key}");
        }
    }

    fn job(id: &str, project: &str, state: &str, node: Option<&str>, gpus: u64) -> Value {
        json!({
            "id": id,
            "project": project,
            "tier": "normal",
            "state": state,
            "node": node,
            "gpus": gpus,
        })
    }

    fn checkpoint(sha: &str, origin: &str, tensors: usize, missing: &[&str]) -> Checkpoint {
        schema_checkpoint(sha, origin, tensors, missing, legacy_checkpoint_schema())
    }

    fn schema_checkpoint(
        sha: &str,
        origin: &str,
        tensors: usize,
        missing: &[&str],
        schema: &'static CheckpointSchema,
    ) -> Checkpoint {
        Checkpoint {
            path: PathBuf::from(format!("/nonexistent/{sha}.model.pt")),
            sha256: sha.to_owned(),
            origin: origin.to_owned(),
            tensors,
            missing_groups: missing.iter().map(|group| (*group).to_owned()).collect(),
            schema: schema.name,
            required_tensors: schema.tensors,
            ..Checkpoint::default()
        }
    }

    fn split_context_schema() -> &'static CheckpointSchema {
        CHECKPOINT_SCHEMAS
            .iter()
            .find(|schema| schema.marker == Some("reward_encoder."))
            .expect("the v3.4 split-context schema is registered")
    }

    /// The exact shapes observed on disk: the original anchor fbcd5459 carries
    /// 130 tensors and no binary branch, a-g8's d36274de carries all 152.
    fn anchor() -> Checkpoint {
        checkpoint(
            &"f".repeat(64),
            "original_anchor",
            130,
            REQUIRED_CHECKPOINT_GROUPS,
        )
    }

    fn full_lineage(job: &str) -> Checkpoint {
        checkpoint(&"d".repeat(64), &format!("lineage:{job}"), 152, &[])
    }

    #[test]
    fn tensor_deficient_anchor_never_wins_over_a_full_lineage_checkpoint() {
        // Candidate order is preference order: the anchor sorts last, but even a
        // *preferred* deficient candidate must lose to a full one.
        let selected = select_checkpoint(&[anchor(), full_lineage("a-g8")]).unwrap();
        assert_eq!(selected.origin, "lineage:a-g8");
        assert_eq!(selected.tensors, 152);
        assert!(selected.is_full());
    }

    #[test]
    fn a_deficient_accepted_pointer_does_not_pin_a_lobotomised_checkpoint() {
        let pointer = checkpoint(
            &"e".repeat(64),
            "accepted_pointer:training/accepted.json",
            130,
            &["binary_reward_heads."],
        );
        let selected = select_checkpoint(&[pointer, full_lineage("a-g8")]).unwrap();
        assert_eq!(selected.origin, "lineage:a-g8");
    }

    #[test]
    fn a_full_accepted_pointer_outranks_the_lineage() {
        let pointer = checkpoint(
            &"e".repeat(64),
            "accepted_pointer:training/accepted.json",
            152,
            &[],
        );
        let selected = select_checkpoint(&[pointer, full_lineage("a-g8")]).unwrap();
        assert_eq!(selected.origin, "accepted_pointer:training/accepted.json");
    }

    #[test]
    fn no_full_candidate_stops_admission_instead_of_degrading_silently() {
        let error = select_checkpoint(&[anchor()]).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("tensor-deficient"), "{message}");
        assert!(message.contains("130 tensors"), "{message}");
        assert!(select_checkpoint(&[]).is_err());
    }

    #[test]
    fn a_full_tensor_count_with_a_missing_group_is_still_deficient() {
        let sneaky = checkpoint(&"a".repeat(64), "lineage:x", 152, &["binary_routing_head."]);
        assert!(!sneaky.is_full());
        assert!(sneaky.deficiency().is_some());
    }

    /// v3.4 un-shared the context encoder (8 new tensors, 152 -> 160). Both
    /// schemas are legitimate warm-start sources during the transition, so the
    /// guard must judge each archive against the schema it actually claims.
    #[test]
    fn the_split_context_schema_is_resolved_by_its_marker_group() {
        let legacy = resolve_checkpoint_schema(&[
            "binary_reward_heads.".to_owned(),
            "binary_routing_head.".to_owned(),
            "binary_stability_head.".to_owned(),
        ]);
        assert_eq!(legacy.name, "policy_v3.3-fused-heads");
        assert_eq!(legacy.tensors, 152);
        let split = resolve_checkpoint_schema(&[
            "binary_reward_heads.".to_owned(),
            "binary_routing_head.".to_owned(),
            "binary_stability_head.".to_owned(),
            "reward_encoder.".to_owned(),
        ]);
        assert_eq!(split.name, "policy_v3.4-split-context");
        assert_eq!(split.tensors, 160);
    }

    #[test]
    fn a_full_split_context_checkpoint_is_accepted_at_one_hundred_and_sixty() {
        let full = schema_checkpoint(
            &"1".repeat(64),
            "lineage:v34",
            160,
            &[],
            split_context_schema(),
        );
        assert!(full.is_full(), "{:?}", full.deficiency());
        assert_eq!(full.schema, "policy_v3.4-split-context");
    }

    /// The whole point of the schema table: an archive that CARRIES
    /// `reward_encoder.` but stops at the v3.3 tensor count is 8 tensors short of
    /// its own schema. Judging it against 152 would warm-start a model whose
    /// reward-side context encoder is random — the v3.4 shape of the lobotomy.
    #[test]
    fn a_truncated_split_context_checkpoint_cannot_pass_on_the_legacy_count() {
        let truncated = schema_checkpoint(
            &"2".repeat(64),
            "lineage:v34-truncated",
            152,
            &[],
            split_context_schema(),
        );
        assert!(!truncated.is_full());
        let message = truncated.deficiency().unwrap();
        assert!(message.contains("152 tensors"), "{message}");
        assert!(message.contains("160"), "{message}");
    }

    #[test]
    fn a_split_context_checkpoint_missing_the_reward_encoder_group_is_deficient() {
        let stripped = schema_checkpoint(
            &"3".repeat(64),
            "lineage:v34-stripped",
            160,
            &["reward_encoder."],
            split_context_schema(),
        );
        assert!(!stripped.is_full());
    }

    /// A defaulted Checkpoint has no resolved schema. It must never be pinnable:
    /// `required_tensors == 0` is "unknown", not "no floor".
    #[test]
    fn a_checkpoint_with_no_resolved_schema_is_never_full() {
        let unresolved = Checkpoint::default();
        assert_eq!(unresolved.required_tensors, 0);
        assert!(!unresolved.is_full());
        assert!(select_checkpoint(&[unresolved]).is_err());
    }

    /// Both schemas may be warm-start sources during the transition, and the
    /// newest completion wins on ordering, not on schema.
    #[test]
    fn a_legacy_and_a_split_context_checkpoint_are_both_admissible() {
        let legacy = full_lineage("a-g8");
        let split = schema_checkpoint(
            &"4".repeat(64),
            "lineage:v34",
            160,
            &[],
            split_context_schema(),
        );
        assert!(legacy.is_full());
        assert!(split.is_full());
        assert_eq!(
            select_checkpoint(&[split.clone(), legacy]).unwrap().origin,
            "lineage:v34"
        );
    }

    #[test]
    fn lineage_orders_newest_completion_first_and_skips_missing_checkpoints() {
        let directory = tempdir().unwrap();
        let runs = directory.path();
        // A run whose train-result.json names a checkpoint that no longer exists
        // must be skipped rather than fail the whole scan.
        let orphan = runs.join(
            "jope-selfplay-train-000000001-a-g1/attempts/0/outputs/training/watermark-000000001",
        );
        fs::create_dir_all(&orphan).unwrap();
        fs::write(
            orphan.join("train-result.json"),
            json!({"checkpoint_sha256": "a".repeat(64), "recipe": "a"}).to_string(),
        )
        .unwrap();
        // A directory that is not a trainer run must be ignored entirely.
        fs::create_dir_all(runs.join("jte-12345/attempts/0/outputs/training/watermark-1")).unwrap();
        assert!(scan_lineage(runs).is_empty());
    }

    #[test]
    fn torch_archive_pickle_scan_finds_required_group_names() {
        assert!(contains_subslice(
            b"\x8c\x18binary_reward_heads.0.0.weight",
            b"binary_reward_heads."
        ));
        assert!(!contains_subslice(
            b"\x8c\x0ereward_heads.0",
            b"binary_reward_heads."
        ));
        assert!(!contains_subslice(b"", b"binary_routing_head."));
    }

    #[test]
    fn trainer_ids_preserve_watermark_and_continuation_sequence() {
        assert_eq!(
            parse_trainer_id("jope-selfplay-train-001805437-c-r000042"),
            Some((1_805_437, 42))
        );
        assert_eq!(
            parse_trainer_id("jope-selfplay-train-001805437-c"),
            Some((1_805_437, 0))
        );
        assert_eq!(
            parse_trainer_id("jope-testonly-train-1000"),
            Some((1_000, 0))
        );
        assert_eq!(parse_trainer_id("jope-selfplay-propose-123"), None);
    }

    #[test]
    fn fifty_thousand_labels_coalesce_into_one_latest_snapshot() {
        assert_eq!(
            continuation_kind(260_000, 100_000),
            ContinuationKind::CoalescedLabels
        );
        assert_eq!(
            trainer_job_id(260_000, "c", 7),
            "jope-selfplay-train-000260000-c-r000007"
        );
    }

    #[test]
    fn no_new_threshold_still_selects_continuous_recent_replay() {
        assert_eq!(
            continuation_kind(120_000, 100_000),
            ContinuationKind::RecentReplay
        );
        assert_eq!(
            continuation_kind(100_000, 100_000),
            ContinuationKind::RecentReplay
        );
    }

    #[test]
    fn protected_nontrainer_work_is_not_part_of_duplicate_guard() {
        let protected = job(
            "ratchet-shadow-protected",
            "jope",
            "running",
            Some("xbabe3"),
            1,
        );
        assert!(parse_trainer_job(&protected).is_none());
        let proposal = job("jope-selfplay-propose-abcdef", PROJECT, "queued", None, 1);
        assert!(parse_trainer_job(&proposal).is_none());
        let trainer = job(
            "jope-selfplay-train-000100000-c-r000001",
            PROJECT,
            "running",
            Some("xbabe2"),
            1,
        );
        let parsed = parse_trainer_job(&trainer).unwrap().unwrap();
        validate_active_trainer(&parsed).unwrap();
    }

    #[test]
    fn malformed_or_misplaced_trainers_fail_closed() {
        let malformed = job(
            "jope-selfplay-train-not-a-watermark",
            PROJECT,
            "running",
            Some("xbabe2"),
            1,
        );
        assert!(parse_trainer_job(&malformed).unwrap().is_err());
        let misplaced = job(
            "jope-selfplay-train-000100000-c-r000001",
            PROJECT,
            "running",
            Some("xbabe3"),
            1,
        );
        let parsed = parse_trainer_job(&misplaced).unwrap().unwrap();
        assert!(validate_active_trainer(&parsed).is_err());
    }

    #[test]
    fn duplicate_active_trainers_fail_closed_without_touching_other_work() {
        let trainers = [
            job(
                "jope-selfplay-train-000100000-c-r000001",
                PROJECT,
                "running",
                Some("xbabe2"),
                1,
            ),
            job(
                "jope-selfplay-train-000100000-c-r000002",
                PROJECT,
                "queued",
                None,
                1,
            ),
        ]
        .iter()
        .map(|row| parse_trainer_job(row).unwrap().unwrap())
        .collect::<Vec<_>>();
        assert!(active_trainers(&trainers).is_err());
    }

    #[test]
    fn priority_must_be_exactly_ten() {
        assert!(priority_is_ready(10.0));
        assert!(!priority_is_ready(1.0));
        assert!(!priority_is_ready(f64::NAN));
    }

    #[test]
    fn durable_intent_round_trips_and_prevents_identity_drift() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("state.json");
        let state = DurableState {
            schema: "JopeContinuousTrainerWatchdogState.v1".to_owned(),
            max_observed_labels: 250_000,
            max_observed_snapshot_watermark: 250_000,
            last_evidence_watermark: 200_000,
            next_sequence: 9,
            intent: Some(SubmissionIntent {
                job_id: "jope-selfplay-train-000250000-c-r000008".to_owned(),
                watermark: 250_000,
                sequence: 8,
                checkpoint_sha256: "a".repeat(64),
                spec: directory.path().join("trainer.toml"),
                spec_sha256: "b".repeat(64),
                mode: "coalesced_50k_labels".to_owned(),
                seed: 20_260_734,
                created_at_ms: 1,
                last_error: None,
                checkpoint_mismatch_observations: 0,
            }),
            updated_at_ms: 2,
        };
        save_state(&path, &state).unwrap();
        let restored = load_state(&path).unwrap();
        assert_eq!(restored.next_sequence, 9);
        assert_eq!(restored.max_observed_snapshot_watermark, 250_000);
        assert_eq!(restored.intent.unwrap().watermark, 250_000);
    }

    #[test]
    fn durable_unsubmitted_intent_survives_newer_corpus_but_not_checkpoint() {
        let directory = tempdir().unwrap();
        let intent = SubmissionIntent {
            job_id: "jope-selfplay-train-000250000-c-r000008".to_owned(),
            watermark: 250_000,
            sequence: 8,
            checkpoint_sha256: "a".repeat(64),
            spec: directory.path().join("trainer.toml"),
            spec_sha256: "b".repeat(64),
            mode: "coalesced_50k_labels".to_owned(),
            seed: 20_260_734,
            created_at_ms: 1,
            last_error: None,
            checkpoint_mismatch_observations: 0,
        };
        let snapshot = TrainingSnapshot {
            watermark: 300_000,
            manifest: PathBuf::new(),
            packed_manifest: PathBuf::new(),
            packed_catalog: PathBuf::new(),
            packed_files: Vec::new(),
        };
        let checkpoint = Checkpoint {
            path: PathBuf::new(),
            sha256: "a".repeat(64),
            ..Checkpoint::default()
        };
        validate_intent(&intent, &snapshot, &checkpoint).unwrap();
        let changed_checkpoint = Checkpoint {
            path: PathBuf::new(),
            sha256: "c".repeat(64),
            ..Checkpoint::default()
        };
        assert!(validate_intent(&intent, &snapshot, &changed_checkpoint).is_err());
    }

    #[test]
    fn stale_unsubmitted_intent_requires_three_checkpoint_observations() {
        let directory = tempdir().unwrap();
        let mut intent = SubmissionIntent {
            job_id: "jope-selfplay-train-000250000-c-r000008".to_owned(),
            watermark: 250_000,
            sequence: 8,
            checkpoint_sha256: "a".repeat(64),
            spec: directory.path().join("trainer.toml"),
            spec_sha256: "b".repeat(64),
            mode: "coalesced_50k_labels".to_owned(),
            seed: 20_260_734,
            created_at_ms: 1,
            last_error: None,
            checkpoint_mismatch_observations: 0,
        };
        assert_eq!(
            observe_intent_checkpoint(&mut intent, &"c".repeat(64)),
            IntentCheckpointDisposition::Confirming(1)
        );
        assert_eq!(
            observe_intent_checkpoint(&mut intent, &"c".repeat(64)),
            IntentCheckpointDisposition::Confirming(2)
        );
        assert_eq!(
            observe_intent_checkpoint(&mut intent, &"c".repeat(64)),
            IntentCheckpointDisposition::Supersede
        );
        assert_eq!(
            observe_intent_checkpoint(&mut intent, &"a".repeat(64)),
            IntentCheckpointDisposition::Current
        );
        assert_eq!(intent.checkpoint_mismatch_observations, 0);
    }

    #[test]
    fn immutable_spec_write_is_idempotent_and_rejects_drift() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("trainer.toml");
        write_immutable(&path, b"one\n").unwrap();
        write_immutable(&path, b"one\n").unwrap();
        assert!(write_immutable(&path, b"two\n").is_err());
    }

    #[test]
    fn temporary_capacity_shortage_queues_but_real_dependency_failure_does_not() {
        assert!(temporarily_lacks_launch_capacity(
            "error[dependency-unsatisfied]: dependencies cannot be verified\n\
             no eligible launch-ready node currently"
        ));
        assert!(!temporarily_lacks_launch_capacity(
            "error[dependency-unsatisfied]: /usr/bin/python3 is absent"
        ));
        assert!(!temporarily_lacks_launch_capacity(
            "no eligible launch-ready node currently"
        ));
    }

    #[test]
    fn stale_spec_contract_is_identified_for_immutable_replacement() {
        let directory = tempdir().unwrap();
        let legacy = directory.path().join("legacy.toml");
        fs::write(
            &legacy,
            "args = [\"--packed-source-train\", \"{input:train}\"]\n",
        )
        .unwrap();
        let lean = directory.path().join("lean.toml");
        fs::write(
            &lean,
            format!(
                "args = [\"--packed-manifest\", \"{{input:packed_manifest}}\"]\n\
                 JOPE_TRAINER_SPEC_CONTRACT = \"{TRAINER_SPEC_CONTRACT}\"\n"
            ),
        )
        .unwrap();
        assert!(intent_uses_stale_contract(&SubmissionIntent {
            spec: legacy,
            ..SubmissionIntent::default()
        }));
        assert!(!intent_uses_stale_contract(&SubmissionIntent {
            spec: lean,
            ..SubmissionIntent::default()
        }));
    }

    #[test]
    fn rendered_contract_has_one_gpu_xbabe2_and_python_only_as_payload() {
        let text = format!(
            "project = \"{PROJECT}\"\n\
             command = \"/usr/bin/python3\"\n\
             gpus_per_task = 1\n\
             args = [\"-qx\", \"xbabe2\", \"/etc/hostname\"]\n"
        );
        assert!(text.contains("project = \"jope-selfplay-train\""));
        assert!(text.contains("gpus_per_task = 1"));
        assert!(text.contains("\"xbabe2\""));
        assert!(text.contains("command = \"/usr/bin/python3\""));
    }

    /// Placement must be constrained by `[[dependencies.smoke]]` and NOTHING else.
    ///
    /// Two wrong answers were tried first, both in production:
    ///   1. `[dependencies] hostname = [...]` -- a serde NO-OP. `hostname` is not
    ///      a field of scq's DependencySpec, so serde drops it, `dependencies`
    ///      becomes empty and `prepare_dependencies` CLEARS
    ///      `dependency_allowed_nodes`. Placement was never constrained: scq put
    ///      the trainer on xbabe3, the launch RPC died with "Broken pipe" 92
    ///      times, and each failure counted against xbabe3's launch budget until
    ///      scq excluded the whole node -- black-holing 432 slots, 41% of the
    ///      cluster, 9 times in one hour.
    ///   2. `[resource] node = "xbabe2"` -- rejected outright: "node is
    ///      controller-only; normal jobs must be node-agnostic". This took the
    ///      trainer to ZERO running jobs.
    ///
    /// `smoke` IS in the schema and IS enforced at placement. Probe the
    /// capability (the live flywheel root, which exists only where the trainer
    /// can actually read its corpus) rather than the hostname, so a second
    /// qualifying node auto-qualifies instead of being hard-coded out.
    #[test]
    fn spec_constrains_placement_via_smoke_dependency_only() {
        let code = TRAINER.render_placement_block();
        assert!(
            code.contains("[[dependencies.smoke]]\n"),
            "smoke dependency missing -- placement would be UNCONSTRAINED and can \
             black-hole a remote node"
        );
        assert!(
            !code.contains("node = \"xbabe2\""),
            "[resource] node pin present -- scq rejects it as controller-only and \
             the trainer will not launch at all"
        );
        assert!(
            !code.contains("hostname"),
            "the hostname dependency is a serde no-op and must not come back"
        );
    }

    #[test]
    fn packed_filenames_cannot_escape_the_staging_directory() {
        ensure_safe_filename("scores.npy").unwrap();
        assert!(ensure_safe_filename("../scores.npy").is_err());
        assert!(ensure_safe_filename("nested/scores.npy").is_err());
    }

    #[test]
    fn training_and_label_pointers_are_content_verified() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let train = root.join("train.jsonl");
        let validation = root.join("validation.jsonl");
        let packed_dir = root.join("packed");
        fs::create_dir(&packed_dir).unwrap();
        let scores = packed_dir.join("scores.npy");
        let catalog = packed_dir.join("catalog.f32le");
        fs::write(&train, b"train\n").unwrap();
        fs::write(&validation, b"validation\n").unwrap();
        fs::write(&scores, b"scores\n").unwrap();
        fs::write(&catalog, b"catalog\n").unwrap();

        let manifest = root.join("corpus-manifest.json");
        atomic_json(
            &manifest,
            &json!({
                "schema": "JopeTestOnlyTrainingCorpus.v1",
                "source_label_watermark": 100_000,
                "source_family_disjoint": true,
                "strict_filter": {"test_adaptive": true, "benchmark_valid": false},
                "train": train,
                "train_sha256": sha256_file(&train).unwrap(),
                "validation": validation,
                "validation_sha256": sha256_file(&validation).unwrap(),
            }),
        )
        .unwrap();
        let packed_manifest = packed_dir.join("manifest.json");
        atomic_json(
            &packed_manifest,
            &json!({
                "source_manifest_sha256": sha256_file(&manifest).unwrap(),
                "train_sha256": sha256_file(&train).unwrap(),
                "validation_sha256": sha256_file(&validation).unwrap(),
                "catalog_data": catalog,
                "catalog_data_sha256": sha256_file(&catalog).unwrap(),
                "files": {
                    "scores.npy": {
                        "sha256": sha256_file(&scores).unwrap(),
                    }
                }
            }),
        )
        .unwrap();
        let pointer = root.join("current-corpus.json");
        atomic_json(
            &pointer,
            &json!({
                "watermark": 100_000,
                "manifest": manifest,
                "manifest_sha256": sha256_file(&manifest).unwrap(),
                "packed_manifest": packed_manifest,
                "packed_manifest_sha256": sha256_file(&packed_manifest).unwrap(),
            }),
        )
        .unwrap();
        assert_eq!(
            load_training_snapshot(&pointer, 120_000).unwrap().watermark,
            100_000
        );
        fs::write(&scores, b"drift\n").unwrap();
        assert!(load_training_snapshot(&pointer, 120_000).is_err());

        let labels = root.join("labels.json");
        atomic_json(&labels, &json!({"exact_rows_total": 120_000})).unwrap();
        assert_eq!(load_label_watermark(&labels).unwrap(), 120_000);
    }
}
