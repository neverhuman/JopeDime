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

use super::flock;

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
        match &self.label_pointer {
            Some(path) => path.clone(),
            None => self.root.join("ops/last-ingest.json"),
        }
    }

    fn corpus_pointer(&self) -> PathBuf {
        match &self.corpus_pointer {
            Some(path) => path.clone(),
            None => self.root.join("training/current-corpus.json"),
        }
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
        let mut value = serde_json::to_value(&report)
            .expect("CycleReport fields have infallible JSON serialization");
        if let Some(object) = value.as_object_mut() {
            object.insert("degraded".to_owned(), Value::Bool(true));
            object.insert("error".to_owned(), Value::String(error.to_owned()));
            object.insert("admission_allowed".to_owned(), Value::Bool(false));
        }
        value
    }
}

#[path = "implementation/daemon.rs"]
mod daemon;
pub(crate) use daemon::run;

/// Read-only dry run of the warm-start decision. Takes no process lock and
/// writes nothing, so it can be exercised against the live root while the daemon
/// is running.
mod cycle;
use cycle::*;
mod lineage;
pub(crate) use lineage::*;
mod state;
pub(crate) use state::*;
mod submission;
pub(crate) use submission::*;
mod schema;
pub(crate) use schema::*;

#[cfg(test)]
mod tests;
