//! Checkpoint-fresh Rust-only SCQ queue keeper.
//!
//! The keeper admits only the newest immutable 128-dataset micro-wave for the
//! active checkpoint.  Audit mode is the default.  `--submit` is required for
//! any scheduler or registry mutation.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_core::{
    dynamic_queue_targets, queue_refill_needed, QueueTargets,
};
use jope_dime_regressor::selfplay_runtime::{
    active_checkpoint, atomic_json, default_evaluator_str, default_root_str, default_runner_str,
    default_scq_str, now_ms, protocol_for_manifest, read_json, scq_json, sha256_bytes, sha256_file,
};
use jope_dime_regressor::spec_policy::EVAL;
use jope_dime_regressor::task::Task;
use jope_dime_regressor::task_preflight::{
    preflight_task_manifest, GeneratedOperatorRegistry, TaskPreflightDisposition,
    TaskPreflightOutcomeV1, TaskPreflightRequest,
};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

mod attempts;
mod database;
mod dispatch;
mod keeper_claims;
mod keeper_cycle;
mod manifests;
mod screening;
#[cfg(test)]
mod tests;

use attempts::*;
use database::*;
use dispatch::*;
use keeper_claims::*;
use keeper_cycle::*;
use manifests::*;
use screening::*;

const PROJECT: &str = "jope-selfplay-eval";
const MANIFEST_SCHEMA: &str = "JopeFlywheelCorpusEvalShard.v1";
/// Timeseries shards only: they must pin a reduction version, which a v1
/// shard is forbidden to carry (corpus_eval.rs `resolve_reduction`).
const MANIFEST_SCHEMA_V2: &str = "JopeFlywheelCorpusEvalShard.v2";
const MICRO_WAVE_KIND: &str = "deterministic_128_dataset_micro_wave";
const VALIDATION_HEADROOM: usize = 64;
const MAX_ADMISSION_WAVES: usize = 8;
const MAX_SUBMISSIONS_PER_CYCLE: usize = 2_048;
const INFRA_FAILURES: &[&str] = &[
    "rpc-error",
    "storage",
    "storage-error",
    "load-error",
    "dependencyunknown",
    "dependency-unknown",
];
const ACTIVE_JOB_STATES: &[&str] = &["queued", "starting", "dispatching", "running", "held"];
const QUEUE_STATES: &[&str] = &["queued", "starting"];
const TERMINAL_STATES: &[&str] = &["completed", "failed", "cancelled"];
const REGRESSION_SCREENING_COST_THRESHOLD: u64 = 30;
const REGRESSION_SCREENING_BUDGET_SECONDS: u64 = 30;
/// Upper bound on the keeper's validation/preparation workers.
const MAX_KEEPER_WORKERS: usize = 32;

/// Worker threads for manifest validation and spec preparation: the CPUs this process may
/// actually use, capped at [`MAX_KEEPER_WORKERS`]. `available_parallelism` honours both the
/// affinity mask and the cgroup `cpu.max` quota, so under the shipped unit (`CPUQuota=400%`)
/// this is 4. Each worker runs a validator and, for bounded screening, a wall-clock-budgeted
/// evaluator subprocess in the same cgroup; a fixed 32 oversubscribed those 4 CPUs ~8x and
/// let CPU contention alone exhaust the 30 s screening budget, which is recorded as a
/// permanent rejection.
fn keeper_workers() -> usize {
    thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(MAX_KEEPER_WORKERS)
}

#[derive(Parser, Debug)]
#[command(
    name = "jope-selfplay-queue",
    about = "Rust checkpoint-fresh queue keeper (audit-only unless --submit)",
    long_about = "Runs every five seconds by default. It admits only provenance-verified \
                  jope-selfplay-eval jobs from the active checkpoint's newest immutable \
                  128-dataset micro-wave. jope-testonly, old epochs/waves, held work, \
                  checkpoint gates, sealed audits, and canaries are never admitted."
)]
struct Args {
    /// Flywheel state root.
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    /// SCQ client binary.
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    /// Precompiled Rust corpus evaluator.
    #[arg(long, default_value = default_evaluator_str())]
    evaluator: PathBuf,
    /// Portable evaluator runner (must not invoke Python).
    #[arg(long, default_value = default_runner_str())]
    runner: PathBuf,
    /// Generated operator/task registry. Defaults to the active immutable
    /// runtime bundle and is unreadable => admission closed.
    #[arg(long)]
    operator_registry: Option<PathBuf>,
    /// Run one cycle and exit.
    #[arg(long)]
    once: bool,
    /// Enable claim creation, infrastructure retry, and SCQ submission.
    #[arg(long)]
    submit: bool,
    /// Seconds between daemon cycles.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..))]
    interval_seconds: u64,
    /// Bound scheduler submissions in one cycle.
    #[arg(long, default_value_t = 256)]
    max_submit_per_cycle: usize,
    /// Keep provenance-valid held jobs held instead of running the release
    /// janitor. Used while a higher-priority GPU campaign needs host RAM/core
    /// headroom; submissions and all non-held work continue normally.
    #[arg(long)]
    preserve_held: bool,
}

#[derive(Clone, Debug)]
struct SchedulerJob {
    id: String,
    project: String,
    state: String,
    failure_class: Option<String>,
    queue_ts: Option<String>,
}

#[derive(Clone, Debug)]
struct PairKey {
    task: String,
    dataset_hash: String,
    semantic_sha256: String,
    protocol_sha256: String,
}

type PairReservationRow = (
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);

#[derive(Clone, Debug)]
struct ValidManifest {
    path: PathBuf,
    sha256: String,
    generated_at_ms: u128,
    pairs: Vec<PairKey>,
    value: Value,
}

#[derive(Clone, Debug)]
struct Claim {
    manifest_sha256: String,
    manifest: PathBuf,
    job_id: String,
    spec: PathBuf,
    state: String,
    attempts: i64,
    owner: Option<String>,
}

#[derive(Clone, Debug)]
struct ReadyClaim {
    claim: Claim,
    manifest: ValidManifest,
    owner: String,
}

#[derive(Clone, Debug)]
struct PairSchema {
    owner_column: &'static str,
    has_infrastructure_retries: bool,
    attempts: Option<AttemptSchema>,
}

#[derive(Clone, Debug)]
struct AttemptSchema {
    number_column: &'static str,
    owner_column: &'static str,
}

#[derive(Debug, Serialize)]
struct NodeSlots {
    node: String,
    physical_cores: i64,
    effective_cores: i64,
    cpu_job_slot_multiplier: f64,
    slot_target: i64,
    running_slots: i64,
}

#[derive(Clone, Debug)]
struct MicroWave {
    wave: String,
    summary_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct AdmissionWave {
    wave: String,
    summary_sha256: String,
    manifest_count: usize,
    unclaimed_manifests: usize,
    candidate_evaluation_count: u64,
    complete: bool,
}

#[derive(Debug, Serialize)]
struct CycleReport {
    schema: &'static str,
    generated_at_ms: u128,
    mode: &'static str,
    admission_allowed: bool,
    degraded: bool,
    error: Option<String>,
    fail_closed_reason: Option<String>,
    active_checkpoint_sha256: Option<String>,
    checkpoint_epoch: Option<String>,
    newest_micro_wave: Option<String>,
    admission_waves: Vec<AdmissionWave>,
    protected_reserve_waves: Vec<AdmissionWave>,
    complete_current_checkpoint_waves: usize,
    partial_current_checkpoint_waves: usize,
    complete_reserve_waves: usize,
    reserve_manifests: usize,
    admission_manifests: usize,
    unclaimed_current_checkpoint_manifests: usize,
    queued_target: i64,
    queued_hard_cap: i64,
    p95_60m_jobs_per_minute: f64,
    p95_60m_candidate_evaluations_per_minute: f64,
    candidate_throughput_estimator: &'static str,
    queued: usize,
    starting: usize,
    queued_starting: usize,
    running: usize,
    dispatching: usize,
    held: usize,
    released_held: usize,
    queue_maximum_age_seconds: f64,
    stale_or_unproven_active: usize,
    launch_ready: usize,
    validated_ready: usize,
    useful_work: usize,
    queued_candidate_evaluations: u64,
    starting_candidate_evaluations: u64,
    dispatching_candidate_evaluations: u64,
    validated_ready_candidate_evaluations: u64,
    candidate_evaluation_work: u64,
    launch_ready_deferred_by_horizon: usize,
    manifests_unclaimed: usize,
    manifests_validated: usize,
    manifest_validation_failures: usize,
    preflight_accepted: usize,
    preflight_permanent_rejections: usize,
    preflight_bounded_screenings: usize,
    candidate_evaluations_preflighted: u64,
    candidate_evaluations_avoided: u64,
    predicted_cpu_seconds_preflighted: u64,
    predicted_cpu_seconds_avoided: u64,
    pair_reservation_conflicts: usize,
    prepared: usize,
    infrastructure_retries_prepared: usize,
    adopted_after_restart: usize,
    submitted: usize,
    submission_failures: usize,
    effective_cluster_slots: i64,
    nodes: Vec<NodeSlots>,
    state_counts: BTreeMap<String, usize>,
}

#[derive(Default)]
struct PrepareOutcome {
    unclaimed: usize,
    validated: usize,
    validation_failures: usize,
    preflight_accepted: usize,
    preflight_permanent_rejections: usize,
    preflight_bounded_screenings: usize,
    candidate_evaluations_preflighted: u64,
    candidate_evaluations_avoided: u64,
    predicted_cpu_seconds_preflighted: u64,
    predicted_cpu_seconds_avoided: u64,
    reservation_conflicts: usize,
    prepared: usize,
}

impl PrepareOutcome {
    fn merge(&mut self, other: Self) {
        self.unclaimed = self.unclaimed.saturating_add(other.unclaimed);
        self.validated = self.validated.saturating_add(other.validated);
        self.validation_failures = self
            .validation_failures
            .saturating_add(other.validation_failures);
        self.preflight_accepted = self
            .preflight_accepted
            .saturating_add(other.preflight_accepted);
        self.preflight_permanent_rejections = self
            .preflight_permanent_rejections
            .saturating_add(other.preflight_permanent_rejections);
        self.preflight_bounded_screenings = self
            .preflight_bounded_screenings
            .saturating_add(other.preflight_bounded_screenings);
        self.candidate_evaluations_preflighted = self
            .candidate_evaluations_preflighted
            .saturating_add(other.candidate_evaluations_preflighted);
        self.candidate_evaluations_avoided = self
            .candidate_evaluations_avoided
            .saturating_add(other.candidate_evaluations_avoided);
        self.predicted_cpu_seconds_preflighted = self
            .predicted_cpu_seconds_preflighted
            .saturating_add(other.predicted_cpu_seconds_preflighted);
        self.predicted_cpu_seconds_avoided = self
            .predicted_cpu_seconds_avoided
            .saturating_add(other.predicted_cpu_seconds_avoided);
        self.reservation_conflicts = self
            .reservation_conflicts
            .saturating_add(other.reservation_conflicts);
        self.prepared = self.prepared.saturating_add(other.prepared);
    }
}

struct Keeper {
    args: Args,
    backlog: Connection,
    pairs: Connection,
    pair_schema: PairSchema,
    manifest_cache: HashMap<(PathBuf, String), ValidManifest>,
    reservation_cache: HashMap<(String, Option<String>, Option<String>), String>,
    terminal_seen: HashSet<String>,
    operator_registry: GeneratedOperatorRegistry,
    /// One pool for the keeper's lifetime, shared by validation and preparation (which run one
    /// after the other). It used to be rebuilt twice per cycle, i.e. 64 threads every 5 s.
    workers: rayon::ThreadPool,
}

fn bounded_validation_budget(unclaimed: usize, admission_limit: usize) -> usize {
    let headroom = VALIDATION_HEADROOM.min(admission_limit);
    unclaimed.min(admission_limit.saturating_add(headroom))
}

fn scheduler_job_is_adoptable(job: &SchedulerJob) -> bool {
    job.project == PROJECT
        && (ACTIVE_JOB_STATES.contains(&job.state.as_str())
            || TERMINAL_STATES.contains(&job.state.as_str()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !(1..=MAX_SUBMISSIONS_PER_CYCLE).contains(&args.max_submit_per_cycle) {
        bail!("--max-submit-per-cycle must be in 1..={MAX_SUBMISSIONS_PER_CYCLE}");
    }
    reject_python_runtime(&args.runner)?;
    let lock_path = args.root.join("ops/queue-keeper.lock");
    let _lock = SingleInstanceLock::acquire(&lock_path)?;
    let mut keeper = Keeper::open(args)?;
    loop {
        match keeper.cycle() {
            Ok(report) => {
                atomic_json(
                    &keeper.args.root.join("ops/queue-keeper-v3.json"),
                    &serde_json::to_value(&report)?,
                )?;
                println!("{}", serde_json::to_string(&report)?);
            }
            Err(error) => {
                let failure = failure_report(keeper.args.submit, &format!("{error:#}"));
                atomic_json(
                    &keeper.args.root.join("ops/queue-keeper-v3.json"),
                    &serde_json::to_value(&failure)?,
                )?;
                eprintln!("{}", serde_json::to_string(&failure)?);
                if keeper.args.once {
                    return Err(error);
                }
            }
        }
        if keeper.args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(keeper.args.interval_seconds));
    }
}

fn parse_jobs(value: &Value) -> Result<Vec<SchedulerJob>> {
    let rows = value
        .get("Jobs")
        .and_then(|value| value.get("jobs"))
        .and_then(Value::as_array)
        .context("SCQ ls JSON lacks Jobs.jobs")?;
    rows.iter()
        .map(|row| {
            Ok(SchedulerJob {
                id: row
                    .get("id")
                    .and_then(Value::as_str)
                    .context("SCQ job lacks id")?
                    .to_owned(),
                project: row
                    .get("project")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                state: row
                    .get("state")
                    .and_then(Value::as_str)
                    .context("SCQ job lacks state")?
                    .to_owned(),
                failure_class: row
                    .get("failure_class")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                queue_ts: row
                    .get("queue_ts")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            })
        })
        .collect()
}

fn parse_nodes(value: &Value) -> Result<Vec<NodeSlots>> {
    let rows = value
        .get("Nodes")
        .and_then(|value| value.get("nodes"))
        .and_then(Value::as_array)
        .context("SCQ nodes JSON lacks Nodes.nodes")?;
    rows.iter()
        .map(|row| {
            let node = row
                .get("node")
                .and_then(Value::as_str)
                .context("SCQ node lacks identity")?;
            let effective = integer_field(row, "effective_cores").max(0);
            let multiplier = row
                .get("cpu_job_slot_multiplier")
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            if !multiplier.is_finite() || !(0.1..=4.0).contains(&multiplier) {
                bail!("SCQ node has an invalid CPU slot multiplier");
            }
            let computed = (effective as f64 * multiplier).floor() as i64;
            let target = row
                .get("job_slot_target")
                .or_else(|| row.get("slot_target"))
                .and_then(Value::as_i64)
                .unwrap_or(computed);
            Ok(NodeSlots {
                node: node.to_owned(),
                physical_cores: integer_field(row, "cores").max(0),
                effective_cores: effective,
                cpu_job_slot_multiplier: multiplier,
                slot_target: target.max(0),
                running_slots: row
                    .get("running_job_slots")
                    .and_then(|field| {
                        field
                            .as_i64()
                            .or_else(|| field.as_u64().map(|value| value as i64))
                    })
                    .unwrap_or_else(|| integer_field(row, "running_jobs"))
                    .max(0),
            })
        })
        .collect()
}

fn integer_field(value: &Value, key: &str) -> i64 {
    value
        .get(key)
        .and_then(|field| field.as_i64().or_else(|| field.as_u64().map(|v| v as i64)))
        .unwrap_or(0)
}

struct SingleInstanceLock {
    file: File,
}

impl SingleInstanceLock {
    fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        if file.try_lock().is_err() {
            bail!("another Rust queue keeper owns {}", path.display());
        }
        Ok(Self { file })
    }
}

impl Drop for SingleInstanceLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}
