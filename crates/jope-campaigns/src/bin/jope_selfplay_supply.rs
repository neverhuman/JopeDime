//! Proactive Rust-only supply supervisor for immutable self-play micro-waves.
//!
//! This process never submits, holds, releases, or cancels SCQ jobs. It keeps a
//! bounded reserve of current-checkpoint manifests ahead of the queue keeper by
//! invoking only the Rust `jope_selfplay_wave` planner/sanitizer/builder.

use anyhow::{bail, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, checkpoint_epoch, now_ms, sha256_bytes, sha256_file, DEFAULT_ROOT, DEFAULT_SCQ,
};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const MICRO_WAVE_KIND: &str = "deterministic_128_dataset_micro_wave";
const WAVE_SCHEMA: &str = "JopeRealtimeSelfplayWave.v1";
const DATASETS_PER_WAVE: usize = 128;
const CANDIDATES_PER_DATASET: usize = 256;
const COHORT_PROPOSAL_SEED: u64 = 20_260_726;
const COHORT_NATIVE_D: usize = 8;
const SUPPLY_STATE_V1_SCHEMA: &str = "CohortSupplyState.v1";
const SUPPLY_STATE_V2_SCHEMA: &str = "CohortSupplyState.v2";
const RUNTIME_PINNED_INTENT_SCHEMA: &str = "RuntimePinnedIntent.v1";
const COHORT_CONTEXT_MIGRATION_SCHEMA: &str = "CohortContextMigrationReceipt.v1";
const COHORT_CONTEXT_EXCLUSION_SCHEMA: &str = "CohortContextExclusionReceipt.v1";
const ERROR_BACKOFF_MAX_SECONDS: u64 = 120;

#[derive(Parser, Debug)]
#[command(
    name = "jope-selfplay-supply",
    about = "Maintain a bounded reserve of immutable current-checkpoint micro-waves"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    /// Build waves. Without this flag the supervisor is read-only.
    #[arg(long)]
    build: bool,
    #[arg(long)]
    once: bool,
    #[arg(long, default_value_t = 5)]
    interval_seconds: u64,
    #[arg(long, default_value_t = 1_000)]
    queue_floor: usize,
    #[arg(long, default_value_t = 2_048)]
    queued_target: usize,
    #[arg(long, default_value_t = 4_096)]
    queued_hard_cap: usize,
    /// Maintain this many current-checkpoint manifests beyond scheduler demand.
    #[arg(long, default_value_t = 1_024)]
    ready_reserve_min: usize,
    /// Maximum current-checkpoint waves with unclaimed manifests.
    #[arg(long, default_value_t = 8)]
    max_overlap_waves: usize,
    /// Exactly one wave is built before waiting for keeper acknowledgment.
    #[arg(long, default_value_t = 1)]
    max_builds_per_cycle: usize,
    /// Kept at 100 normally; values below 25 could overshoot the hard cap.
    #[arg(long, default_value_t = 100)]
    max_candidates_per_manifest: usize,
    #[arg(long, default_value_t = 30)]
    queue_status_max_age_seconds: u64,
    #[arg(long, default_value = DEFAULT_SCQ)]
    scq: PathBuf,
    #[arg(
        long,
        default_value = "/home/ubuntu/jope-dime/target/release/jope_selfplay_wave"
    )]
    wave_builder: PathBuf,
    #[arg(
        long,
        default_value = "/home/ubuntu/jope-dime/scripts/test_exact_context_proposals.py"
    )]
    proposal_generator: PathBuf,
    #[arg(
        long,
        default_value = "/home/ubuntu/jope-dime/scripts/test_exact_dl_generate.py"
    )]
    proposal_legacy_generator: PathBuf,
    #[arg(
        long,
        default_value = "/home/ubuntu/jope-dime/scripts/test_exact_common.py"
    )]
    proposal_common: PathBuf,
    #[arg(long, default_value = "/home/ubuntu/jope-prime/src")]
    proposal_pythonpath: PathBuf,
    #[arg(long)]
    catalog_pointer: Option<PathBuf>,
    #[arg(long)]
    checkpoint_pointer: Option<PathBuf>,
    #[arg(long)]
    proposals_pointer: Option<PathBuf>,
    #[arg(long)]
    corpus: Option<PathBuf>,
    #[arg(long)]
    sealed_cohort: Option<PathBuf>,
    #[arg(long)]
    robust_seed_bank: Option<PathBuf>,
    #[arg(long)]
    pair_registry: Option<PathBuf>,
    #[arg(long)]
    queue_status: Option<PathBuf>,
}

impl Args {
    fn catalog_pointer(&self) -> PathBuf {
        self.catalog_pointer
            .clone()
            .unwrap_or_else(|| self.root.join("catalog/current.json"))
    }

    fn proposals_pointer(&self) -> PathBuf {
        self.proposals_pointer
            .clone()
            .unwrap_or_else(|| self.root.join("proposals/current.json"))
    }

    fn corpus(&self) -> PathBuf {
        self.corpus
            .clone()
            .unwrap_or_else(|| self.root.join("corpus/strict-test-exact.sqlite"))
    }

    fn sealed_cohort(&self) -> PathBuf {
        self.sealed_cohort
            .clone()
            .unwrap_or_else(|| self.root.join("catalog/private/sealed-current-v1.json"))
    }

    fn robust_seed_bank(&self) -> PathBuf {
        self.robust_seed_bank
            .clone()
            .unwrap_or_else(|| self.root.join("corpus/robust-gp-seed-bank.v1.json"))
    }

    fn pair_registry(&self) -> PathBuf {
        self.pair_registry
            .clone()
            .unwrap_or_else(|| self.root.join("ops/pair-registry.sqlite"))
    }

    fn queue_status(&self) -> PathBuf {
        self.queue_status
            .clone()
            .unwrap_or_else(|| self.root.join("ops/queue-keeper-v3.json"))
    }
}

#[derive(Debug, Deserialize)]
struct QueueStatus {
    schema: String,
    generated_at_ms: u128,
    admission_allowed: bool,
    #[serde(default)]
    degraded: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    fail_closed_reason: Option<String>,
    active_checkpoint_sha256: Option<String>,
    checkpoint_epoch: Option<String>,
    queued_target: i64,
    queued_hard_cap: i64,
    queued_starting: usize,
    dispatching: usize,
    admission_waves: Vec<KeeperAdmissionWave>,
    unclaimed_current_checkpoint_manifests: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct KeeperAdmissionWave {
    wave: String,
    summary_sha256: String,
    unclaimed_manifests: usize,
}

#[derive(Debug)]
struct CheckpointInputs {
    checkpoint_sha256: String,
    checkpoint_pointer: PathBuf,
    checkpoint_artifact: PathBuf,
    proposals_pointer: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
struct ReadyWave {
    wave: String,
    summary_sha256: String,
    manifests: usize,
    unclaimed_manifests: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SupplyDecision {
    below_floor: bool,
    overloaded: bool,
    projected_jobs: usize,
    estimated_manifests_per_wave: usize,
    builds_planned: usize,
}

#[derive(Debug, Serialize)]
struct BuiltWave {
    wave: String,
    cohort_sha256: String,
    proposal_pointer_sha256: String,
    summary_sha256: String,
    manifests: usize,
}

#[derive(Debug, Serialize)]
struct CycleReport {
    schema: &'static str,
    generated_at_ms: u128,
    mode: &'static str,
    active_checkpoint_sha256: String,
    checkpoint_epoch: String,
    queue_floor: usize,
    queued_target: usize,
    queued_hard_cap: usize,
    ready_reserve_min: usize,
    keeper_queued_target: i64,
    keeper_queued_hard_cap: i64,
    queued_starting: usize,
    dispatching: usize,
    queue_depth: usize,
    below_floor: bool,
    overloaded: bool,
    current_ready_waves: Vec<ReadyWave>,
    ready_manifests: usize,
    projected_jobs_before_build: usize,
    projected_jobs_after_build: usize,
    max_overlap_waves: usize,
    builds_planned: usize,
    builds_completed: Vec<BuiltWave>,
    awaiting_keeper_ack: Option<String>,
    awaiting_keeper_ack_since_ms: Option<u128>,
    keeper_status_generated_at_ms: u128,
    admission_allowed: bool,
    hold_reason: Option<String>,
}

struct PendingAck {
    wave: String,
    since_ms: u128,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PersistedSupplyStage {
    FreezeCohort,
    SubmitCohortProposer,
    FetchValidateProposals,
    CacheFilterResample,
    BuildWave,
    KeeperAcknowledgment,
}

impl PersistedSupplyStage {
    fn ordinal(self) -> u8 {
        match self {
            Self::FreezeCohort => 0,
            Self::SubmitCohortProposer => 1,
            Self::FetchValidateProposals => 2,
            Self::CacheFilterResample => 3,
            Self::BuildWave => 4,
            Self::KeeperAcknowledgment => 5,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedSupplyState {
    schema: String,
    wave: String,
    checkpoint_sha256: String,
    cohort: PathBuf,
    cohort_sha256: String,
    catalog_sha256: String,
    decoder_configuration_sha256: String,
    seed: u64,
    #[serde(default = "default_proposal_reservoir")]
    proposal_reservoir_per_context: usize,
    stage: PersistedSupplyStage,
    revision: u64,
    recovery_accounting_only: bool,
    #[serde(default)]
    proposal_attempt: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    proposal_job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cohort_proposal_pointer: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cohort_proposal_pointer_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sanitized_proposal_pointer: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sanitized_proposal_pointer_sha256: Option<String>,
    #[serde(default)]
    runtime_pinned_attempts: Vec<RuntimePinnedIntentV1>,
}

fn default_proposal_reservoir() -> usize {
    64
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PinnedProposerInput {
    name: String,
    path: PathBuf,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RuntimePinnedIntentV1 {
    schema: String,
    runtime_bundle: PathBuf,
    runtime_bundle_sha256: String,
    catalog_snapshot: PathBuf,
    catalog_snapshot_sha256: String,
    proposer_inputs: Vec<PinnedProposerInput>,
    immutable_spec_path: PathBuf,
    immutable_spec_sha256: String,
    job_identity: String,
    attempt: u32,
}

impl RuntimePinnedIntentV1 {
    fn validate(&self) -> Result<()> {
        if self.schema != RUNTIME_PINNED_INTENT_SCHEMA {
            bail!("runtime-pinned proposer intent schema is not v1");
        }
        validate_sha256(&self.runtime_bundle_sha256)?;
        validate_sha256(&self.catalog_snapshot_sha256)?;
        validate_sha256(&self.immutable_spec_sha256)?;
        if self.job_identity.trim().is_empty() {
            bail!("runtime-pinned proposer job identity is empty");
        }
        if !self.runtime_bundle.is_dir()
            || self
                .runtime_bundle
                .file_name()
                .and_then(|value| value.to_str())
                != Some(self.runtime_bundle_sha256.as_str())
        {
            bail!("runtime-pinned proposer bundle identity drift");
        }
        let bundle: Value =
            serde_json::from_slice(&fs::read(self.runtime_bundle.join("bundle.json"))?)?;
        if bundle.get("bundle_sha256").and_then(Value::as_str)
            != Some(self.runtime_bundle_sha256.as_str())
        {
            bail!("runtime-pinned proposer bundle manifest drift");
        }
        if !self.immutable_spec_path.is_file()
            || sha256_file(&self.immutable_spec_path)? != self.immutable_spec_sha256
        {
            bail!("runtime-pinned proposer immutable spec drift");
        }
        if self.proposer_inputs.is_empty() {
            bail!("runtime-pinned proposer has no immutable inputs");
        }
        let mut names = BTreeSet::new();
        for input in &self.proposer_inputs {
            validate_sha256(&input.sha256)?;
            if input.name.trim().is_empty()
                || !names.insert(input.name.as_str())
                || !input.path.is_file()
                || sha256_file(&input.path)? != input.sha256
            {
                bail!("runtime-pinned proposer input identity drift");
            }
        }
        if !names.contains("generator")
            || !names.contains("common")
            || !names.contains("pointer")
            || !names.contains("catalog")
            || !names.contains("data")
            || !names.contains("checkpoint")
            || !names.contains("cohort")
        {
            bail!("runtime-pinned proposer input set is incomplete");
        }
        let pointer = self
            .proposer_inputs
            .iter()
            .find(|input| input.name == "pointer")
            .context("runtime-pinned proposer has no catalog snapshot input")?;
        if pointer.path != self.catalog_snapshot || pointer.sha256 != self.catalog_snapshot_sha256 {
            bail!("runtime-pinned proposer catalog snapshot drift");
        }
        Ok(())
    }
}

impl PersistedSupplyState {
    fn validate(&self, expected_wave: &str, expected_checkpoint: &str) -> Result<()> {
        if !matches!(
            self.schema.as_str(),
            SUPPLY_STATE_V1_SCHEMA | SUPPLY_STATE_V2_SCHEMA
        ) || self.wave != expected_wave
            || self.checkpoint_sha256 != expected_checkpoint
        {
            bail!("persisted cohort supply identity drift");
        }
        validate_sha256(&self.cohort_sha256)?;
        validate_sha256(&self.catalog_sha256)?;
        validate_sha256(&self.decoder_configuration_sha256)?;
        if !matches!(self.proposal_reservoir_per_context, 64 | 128 | 256) {
            bail!("cohort proposal reservoir must be 64, 128, or 256");
        }
        if self.stage.ordinal() >= PersistedSupplyStage::SubmitCohortProposer.ordinal()
            && self.proposal_job_id.is_none()
        {
            bail!("submitted cohort supply state lacks a proposer job identity");
        }
        if self.stage.ordinal() >= PersistedSupplyStage::FetchValidateProposals.ordinal()
            && (self.cohort_proposal_pointer.is_none()
                || self.cohort_proposal_pointer_sha256.is_none())
        {
            bail!("fetched cohort supply state lacks an immutable proposal pointer");
        }
        if self.stage.ordinal() >= PersistedSupplyStage::CacheFilterResample.ordinal()
            && (self.sanitized_proposal_pointer.is_none()
                || self.sanitized_proposal_pointer_sha256.is_none())
        {
            bail!("cache-filtered supply state lacks its proposal pointer");
        }
        for (index, intent) in self.runtime_pinned_attempts.iter().enumerate() {
            intent.validate()?;
            if intent.attempt as usize != index {
                bail!("runtime-pinned proposer attempts are not contiguous");
            }
        }
        if self.schema == SUPPLY_STATE_V2_SCHEMA
            && self.stage.ordinal() >= PersistedSupplyStage::SubmitCohortProposer.ordinal()
        {
            let intent = self
                .runtime_pinned_attempts
                .last()
                .context("submitted v2 supply state lacks a runtime-pinned intent")?;
            if Some(intent.job_identity.as_str()) != self.proposal_job_id.as_deref()
                || intent.attempt != self.proposal_attempt
            {
                bail!("submitted v2 supply state proposer attempt identity drift");
            }
        }
        Ok(())
    }

    fn advance(&mut self, next: PersistedSupplyStage) -> Result<()> {
        if next.ordinal() != self.stage.ordinal() + 1 {
            bail!("cohort supply stage must advance exactly once");
        }
        self.stage = next;
        self.revision += 1;
        self.validate(&self.wave, &self.checkpoint_sha256)
    }

    fn retry_proposer(&mut self, root: &Path) -> Result<()> {
        if self.stage != PersistedSupplyStage::SubmitCohortProposer {
            bail!("cohort proposer retry is allowed only in its submitted stage");
        }
        if self.proposal_attempt >= 5 {
            bail!("cohort proposer retry budget exhausted");
        }
        let current = self
            .proposal_job_id
            .as_deref()
            .context("cohort proposer retry lacks a job identity")?;
        let base = current
            .rsplit_once("-retry")
            .filter(|(_, suffix)| suffix.bytes().all(|byte| byte.is_ascii_digit()))
            .map(|(base, _)| base)
            .unwrap_or(current)
            .to_owned();
        self.proposal_attempt += 1;
        let job_identity = format!("{base}-retry{}", self.proposal_attempt);
        let prior = self
            .runtime_pinned_attempts
            .last()
            .cloned()
            .context("cohort proposer retry lacks its pinned original spec")?;
        prior.validate()?;
        let prior_body = fs::read_to_string(&prior.immutable_spec_path)?;
        let replacement = format!("id = {}", toml_quote(&job_identity)?);
        let mut replaced = false;
        let body = prior_body
            .lines()
            .map(|line| {
                if !replaced && line.starts_with("id = ") {
                    replaced = true;
                    replacement.as_str()
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !replaced {
            bail!("pinned cohort proposer spec lacks a job identity");
        }
        let directory = root.join("specs/cohort-propose");
        fs::create_dir_all(&directory)?;
        let spec = directory.join(format!("{job_identity}.toml"));
        if spec.is_file() {
            if fs::read_to_string(&spec)? != body {
                bail!("immutable cohort proposer retry spec drift");
            }
        } else {
            fs::write(&spec, &body)?;
        }
        let pinned = runtime_pinned_intent_from_spec(&spec, self.proposal_attempt)?;
        if pinned.runtime_bundle_sha256 != prior.runtime_bundle_sha256
            || pinned.catalog_snapshot_sha256 != prior.catalog_snapshot_sha256
            || pinned.proposer_inputs != prior.proposer_inputs
        {
            bail!("cohort proposer retry changed its pinned bundle or inputs");
        }
        self.runtime_pinned_attempts.push(pinned);
        self.proposal_job_id = Some(job_identity);
        self.revision += 1;
        self.validate(&self.wave, &self.checkpoint_sha256)
    }
}

struct SingleInstanceLock {
    _file: File,
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
            .mode(0o600)
            .open(path)?;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            bail!(
                "another micro-wave supply supervisor owns {}",
                path.display()
            );
        }
        Ok(Self { _file: file })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let _lock = SingleInstanceLock::acquire(&args.root.join("ops/micro-wave-supply-builder.lock"))?;
    let circuit_path = args.root.join("ops/cohort-supply-circuit-breaker.v1.json");
    let (mut previous_error, mut identical_failures) =
        load_supply_failure_streak(&circuit_path).unwrap_or_default();
    loop {
        let mut retry_after_seconds = args.interval_seconds;
        match cycle(&args) {
            Ok(report) => {
                let value = serde_json::to_value(&report)?;
                atomic_json(&args.root.join("ops/micro-wave-supply-v1.json"), &value)?;
                previous_error = None;
                identical_failures = 0;
                atomic_json(
                    &circuit_path,
                    &json!({
                        "schema": "SupplyCircuitBreaker.v1",
                        "generated_at_ms": now_ms(),
                        "open": false,
                        "identical_failures": 0,
                        "critical": false,
                        "retry_after_seconds": args.interval_seconds,
                        "recoverable_state": args.root.join("ops/cohort-supply-state-v2"),
                    }),
                )?;
                println!("{}", serde_json::to_string(&report)?);
            }
            Err(error) => {
                let error_text = format!("{error:#}");
                if previous_error.as_deref() == Some(error_text.as_str()) {
                    identical_failures = identical_failures.saturating_add(1);
                } else {
                    previous_error = Some(error_text.clone());
                    identical_failures = 1;
                }
                let exponent = identical_failures.saturating_sub(1).min(31);
                retry_after_seconds = args
                    .interval_seconds
                    .saturating_mul(1_u64 << exponent)
                    .min(ERROR_BACKOFF_MAX_SECONDS);
                let failure = json!({
                    "schema": "JopeMicroWaveSupplyError.v1",
                    "generated_at_ms": now_ms(),
                    "mode": if args.build { "build" } else { "audit" },
                    "error": error_text,
                    "builds_completed": 0,
                    "identical_failures": identical_failures,
                    "critical": identical_failures >= 3,
                    "retry_after_seconds": retry_after_seconds,
                });
                atomic_json(&args.root.join("ops/micro-wave-supply-v1.json"), &failure)?;
                atomic_json(
                    &circuit_path,
                    &json!({
                        "schema": "SupplyCircuitBreaker.v1",
                        "generated_at_ms": now_ms(),
                        "open": true,
                        "error": previous_error,
                        "identical_failures": identical_failures,
                        "critical": identical_failures >= 3,
                        "retry_after_seconds": retry_after_seconds,
                        "recoverable_state": args.root.join("ops/cohort-supply-state-v2"),
                    }),
                )?;
                eprintln!("{}", serde_json::to_string(&failure)?);
                if args.once {
                    return Err(error);
                }
            }
        }
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(retry_after_seconds));
    }
}

fn load_supply_failure_streak(path: &Path) -> Result<(Option<String>, u32)> {
    if !path.is_file() {
        return Ok((None, 0));
    }
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    if value.get("schema").and_then(Value::as_str) != Some("SupplyCircuitBreaker.v1")
        || value.get("open").and_then(Value::as_bool) != Some(true)
    {
        return Ok((None, 0));
    }
    let error = value
        .get("error")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let count = value
        .get("identical_failures")
        .and_then(Value::as_u64)
        .unwrap_or_default()
        .min(u32::MAX as u64) as u32;
    Ok((error, count))
}

fn validate_args(args: &Args) -> Result<()> {
    if args.queue_floor == 0
        || args.queue_floor > args.queued_target
        || args.queued_target > args.queued_hard_cap
    {
        bail!("queue thresholds must satisfy 0 < floor <= target <= hard cap");
    }
    if args.ready_reserve_min == 0 || args.ready_reserve_min > args.queued_target {
        bail!("ready reserve must satisfy 0 < reserve <= target");
    }
    if !(1..=64).contains(&args.max_overlap_waves) {
        bail!("--max-overlap-waves must be in 1..=64");
    }
    if args.max_builds_per_cycle != 1 {
        bail!("--max-builds-per-cycle must be exactly 1 for keeper acknowledgment");
    }
    if !(25..=100).contains(&args.max_candidates_per_manifest) {
        bail!("--max-candidates-per-manifest must be in 25..=100");
    }
    let maximum_wave_manifests = manifests_per_wave(args.max_candidates_per_manifest);
    if args.queued_target.saturating_add(maximum_wave_manifests) > args.queued_hard_cap {
        bail!("one micro-wave could overshoot the configured queued hard cap");
    }
    Ok(())
}

fn dynamic_queued_target(args: &Args) -> usize {
    let performance = args.root.join("ops/supply-performance.v1.json");
    let Ok(value) = fs::read(&performance)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .context("supply performance unavailable")
    else {
        return args.queued_target;
    };
    let burn = value
        .get("evaluator_burn_per_second_p95")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let proposer_seconds = value
        .get("proposer_to_ready_seconds_p95")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    if !burn.is_finite() || !proposer_seconds.is_finite() || burn <= 0.0 || proposer_seconds <= 0.0
    {
        return args.queued_target;
    }
    (1.5 * burn * proposer_seconds)
        .ceil()
        .clamp(2_048.0, args.queued_hard_cap.min(4_096) as f64) as usize
}

fn cycle(args: &Args) -> Result<CycleReport> {
    let status = load_queue_status(args)?;
    if !status.admission_allowed {
        return Ok(held_report(args, &status));
    }
    let inputs = checkpoint_inputs(args, &status)?;
    let epoch = checkpoint_epoch(&inputs.checkpoint_sha256)?;
    if status.checkpoint_epoch.as_deref() != Some(epoch.as_str()) {
        bail!("queue keeper checkpoint epoch differs from its checkpoint identity");
    }
    let claimed = claimed_manifest_hashes(&args.root.join("ops/backlog.sqlite"))?;
    let waves = ready_waves(
        &args.root.join("waves"),
        &inputs.checkpoint_sha256,
        &epoch,
        &claimed,
        args.max_overlap_waves + 1,
    )?;
    if waves.len() > args.max_overlap_waves {
        bail!("more than the bounded number of current-checkpoint waves remain unclaimed");
    }
    let mut pending_ack = load_pending_ack(&args.root, &inputs.checkpoint_sha256)?;
    if let Some(pending) = &pending_ack {
        if keeper_acknowledged(args, &status, &inputs, &epoch, &claimed, pending)? {
            mark_keeper_acknowledged(&args.root, &pending.wave, &inputs.checkpoint_sha256)?;
            pending_ack = None;
        }
    }
    validate_keeper_admission(
        &status,
        &waves,
        &args.root.join("waves"),
        pending_ack.as_ref().map(|pending| pending.wave.as_str()),
        args.max_overlap_waves,
    )?;
    let queue_depth = status
        .queued_starting
        .checked_add(status.dispatching)
        .context("queue depth overflow")?;
    if queue_depth > args.queued_hard_cap {
        bail!("current queued+dispatching depth exceeds the configured hard cap");
    }
    let ready_before = status.unclaimed_current_checkpoint_manifests;
    let queued_target = dynamic_queued_target(args);
    let decision = supply_decision(
        queue_depth,
        ready_before,
        waves.len(),
        args.queue_floor,
        queued_target,
        args.queued_hard_cap,
        args.ready_reserve_min,
        args.max_overlap_waves,
        args.max_builds_per_cycle,
        args.max_candidates_per_manifest,
    );
    let mut completed = Vec::new();
    let builds_planned = if pending_ack.is_none() {
        decision.builds_planned.min(1)
    } else {
        0
    };
    if args.build && builds_planned == 1 {
        ensure_native_wave_builder(&args.wave_builder)?;
        assert_checkpoint_unchanged(args, &inputs.checkpoint_sha256)?;
        let built = build_next_wave(args, &inputs)?;
        assert_checkpoint_unchanged(args, &inputs.checkpoint_sha256)?;
        if let Some(built) = built {
            pending_ack = Some(PendingAck {
                wave: built.wave.clone(),
                since_ms: now_ms(),
            });
            completed.push(built);
        }
    }
    let report = CycleReport {
        schema: "JopeMicroWaveSupplyStatus.v1",
        generated_at_ms: now_ms(),
        mode: if args.build { "build" } else { "audit" },
        active_checkpoint_sha256: inputs.checkpoint_sha256,
        checkpoint_epoch: epoch,
        queue_floor: args.queue_floor,
        queued_target,
        queued_hard_cap: args.queued_hard_cap,
        ready_reserve_min: args.ready_reserve_min,
        keeper_queued_target: status.queued_target,
        keeper_queued_hard_cap: status.queued_hard_cap,
        queued_starting: status.queued_starting,
        dispatching: status.dispatching,
        queue_depth,
        below_floor: decision.below_floor,
        overloaded: decision.overloaded,
        current_ready_waves: waves,
        ready_manifests: ready_before,
        projected_jobs_before_build: decision.projected_jobs,
        // A built wave is deliberately not counted until a newer keeper
        // status acknowledges it in the bounded admission set.
        projected_jobs_after_build: decision.projected_jobs,
        max_overlap_waves: args.max_overlap_waves,
        builds_planned,
        builds_completed: completed,
        awaiting_keeper_ack: pending_ack.as_ref().map(|pending| pending.wave.clone()),
        awaiting_keeper_ack_since_ms: pending_ack.as_ref().map(|pending| pending.since_ms),
        keeper_status_generated_at_ms: status.generated_at_ms,
        admission_allowed: status.admission_allowed,
        hold_reason: None,
    };
    if !report.builds_completed.is_empty() {
        append_event(
            &args.root.join("ops/micro-wave-supply-events.v1.jsonl"),
            &serde_json::to_value(&report)?,
        )?;
    }
    Ok(report)
}

fn held_report(args: &Args, status: &QueueStatus) -> CycleReport {
    let active = status.active_checkpoint_sha256.clone().unwrap_or_default();
    let epoch = status.checkpoint_epoch.clone().unwrap_or_default();
    let queue_depth = status.queued_starting.saturating_add(status.dispatching);
    CycleReport {
        schema: "JopeMicroWaveSupplyStatus.v1",
        generated_at_ms: now_ms(),
        mode: if args.build { "build" } else { "audit" },
        active_checkpoint_sha256: active,
        checkpoint_epoch: epoch,
        queue_floor: args.queue_floor,
        queued_target: dynamic_queued_target(args),
        queued_hard_cap: args.queued_hard_cap,
        ready_reserve_min: args.ready_reserve_min,
        keeper_queued_target: status.queued_target,
        keeper_queued_hard_cap: status.queued_hard_cap,
        queued_starting: status.queued_starting,
        dispatching: status.dispatching,
        queue_depth,
        below_floor: queue_depth < args.queue_floor,
        overloaded: queue_depth.saturating_add(status.unclaimed_current_checkpoint_manifests)
            > dynamic_queued_target(args),
        current_ready_waves: Vec::new(),
        ready_manifests: status.unclaimed_current_checkpoint_manifests,
        projected_jobs_before_build: queue_depth
            .saturating_add(status.unclaimed_current_checkpoint_manifests),
        projected_jobs_after_build: queue_depth
            .saturating_add(status.unclaimed_current_checkpoint_manifests),
        max_overlap_waves: args.max_overlap_waves,
        builds_planned: 0,
        builds_completed: Vec::new(),
        awaiting_keeper_ack: None,
        awaiting_keeper_ack_since_ms: None,
        keeper_status_generated_at_ms: status.generated_at_ms,
        admission_allowed: false,
        hold_reason: status
            .error
            .clone()
            .or_else(|| status.fail_closed_reason.clone())
            .or_else(|| {
                status
                    .degraded
                    .then(|| "queue keeper is degraded".to_owned())
            })
            .or_else(|| Some("queue admission is held".to_owned())),
    }
}

fn load_queue_status(args: &Args) -> Result<QueueStatus> {
    let path = args.queue_status();
    let status: QueueStatus = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("parse queue status {}", path.display()))?;
    if status.schema != "JopeSelfplayQueueKeeper.v3" {
        bail!("queue keeper status schema is not v3");
    }
    let maximum_age_ms = u128::from(args.queue_status_max_age_seconds) * 1_000;
    if now_ms().saturating_sub(status.generated_at_ms) > maximum_age_ms {
        bail!("queue keeper status is stale");
    }
    Ok(status)
}

fn load_pending_ack(root: &Path, checkpoint_sha256: &str) -> Result<Option<PendingAck>> {
    let path = root.join("ops/micro-wave-supply-v1.json");
    if !path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("parse prior supply status {}", path.display()))?;
    if value.get("schema").and_then(Value::as_str) != Some("JopeMicroWaveSupplyStatus.v1")
        || value
            .get("active_checkpoint_sha256")
            .and_then(Value::as_str)
            != Some(checkpoint_sha256)
    {
        return Ok(None);
    }
    let Some(wave) = value.get("awaiting_keeper_ack").and_then(Value::as_str) else {
        return Ok(None);
    };
    if !valid_wave_id(wave) {
        bail!("prior supply status has an invalid acknowledgment wave");
    }
    let since_ms = value
        .get("awaiting_keeper_ack_since_ms")
        .and_then(Value::as_u64)
        .context("prior supply status lacks acknowledgment timestamp")?;
    Ok(Some(PendingAck {
        wave: wave.to_owned(),
        since_ms: u128::from(since_ms),
    }))
}

fn keeper_acknowledged(
    args: &Args,
    status: &QueueStatus,
    inputs: &CheckpointInputs,
    epoch: &str,
    claimed: &BTreeSet<String>,
    pending: &PendingAck,
) -> Result<bool> {
    if status.generated_at_ms <= pending.since_ms {
        return Ok(false);
    }
    if status
        .admission_waves
        .iter()
        .any(|wave| wave.wave == pending.wave)
    {
        return Ok(true);
    }
    let directory = args.root.join("waves").join(&pending.wave);
    let Some(wave) = validate_ready_wave(
        &pending.wave,
        &directory,
        &inputs.checkpoint_sha256,
        epoch,
        claimed,
    )?
    else {
        bail!("keeper did not acknowledge the built current-checkpoint wave");
    };
    Ok(wave.unclaimed_manifests == 0)
}

fn validate_keeper_admission(
    status: &QueueStatus,
    ready: &[ReadyWave],
    waves_root: &Path,
    pending_wave: Option<&str>,
    max_overlap_waves: usize,
) -> Result<()> {
    if status.admission_waves.len() > max_overlap_waves {
        bail!("keeper admission wave set exceeds the bounded overlap");
    }
    if status
        .admission_waves
        .windows(2)
        .any(|pair| pair[0].wave >= pair[1].wave)
    {
        bail!("keeper admission waves are not deterministic oldest-first");
    }
    let reported_unclaimed = status
        .admission_waves
        .iter()
        .try_fold(0_usize, |total, wave| {
            total.checked_add(wave.unclaimed_manifests)
        })
        .context("keeper unclaimed manifest count overflow")?;
    if reported_unclaimed != status.unclaimed_current_checkpoint_manifests {
        bail!("keeper admission wave counts do not match its aggregate");
    }
    let ready_by_wave = ready
        .iter()
        .map(|wave| (wave.wave.as_str(), wave))
        .collect::<BTreeMap<_, _>>();
    let mut admitted = BTreeSet::new();
    for keeper_wave in &status.admission_waves {
        if !admitted.insert(keeper_wave.wave.as_str()) {
            bail!("keeper admission wave is duplicated");
        }
        if let Some(local) = ready_by_wave.get(keeper_wave.wave.as_str()) {
            if keeper_wave.summary_sha256 != local.summary_sha256
                || local.unclaimed_manifests > keeper_wave.unclaimed_manifests
            {
                bail!("keeper admission wave provenance or count drift");
            }
        } else {
            if !valid_wave_id(&keeper_wave.wave) {
                bail!("keeper admission wave identity is invalid");
            }
            let summary = waves_root.join(&keeper_wave.wave).join("summary.json");
            if !summary.is_file() || sha256_file(&summary)? != keeper_wave.summary_sha256 {
                bail!("keeper admitted a wave with no matching local immutable summary");
            }
        }
    }
    for wave in ready {
        if !admitted.contains(wave.wave.as_str()) && pending_wave != Some(wave.wave.as_str()) {
            bail!("current-checkpoint unclaimed wave is outside keeper admission");
        }
    }
    Ok(())
}

fn checkpoint_inputs(args: &Args, status: &QueueStatus) -> Result<CheckpointInputs> {
    let active = status
        .active_checkpoint_sha256
        .as_deref()
        .context("queue keeper has no active checkpoint")?;
    validate_sha256(active)?;
    let checkpoint_pointer = if let Some(path) = &args.checkpoint_pointer {
        path.clone()
    } else {
        ["training/learning.json", "training/accepted.json"]
            .into_iter()
            .map(|relative| args.root.join(relative))
            .find(|path| path.is_file())
            .unwrap_or_else(|| args.proposals_pointer())
    };
    if pointer_checkpoint(&checkpoint_pointer)? != active {
        bail!("authoritative checkpoint pointer differs from queue keeper status");
    }
    let checkpoint_value: Value = serde_json::from_slice(&fs::read(&checkpoint_pointer)?)?;
    let declared_checkpoint = checkpoint_value
        .get("checkpoint")
        .or_else(|| checkpoint_value.pointer("/legacy_adopted/checkpoint"))
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let checkpoint_artifact = [
        declared_checkpoint,
        Some(
            args.root
                .join("training/immutable/checkpoints")
                .join(format!("{active}.model.pt")),
        ),
        Some(
            PathBuf::from("/home/ubuntu/jope-flywheel/training/round1/checkpoints")
                .join(format!("{active}.model.pt")),
        ),
    ]
    .into_iter()
    .flatten()
    .find(|path| path.is_file() && sha256_file(path).ok().as_deref() == Some(active))
    .context("active checkpoint artifact is absent or content-invalid")?;
    let proposals_pointer = args.proposals_pointer();
    if pointer_checkpoint(&proposals_pointer)? != active {
        bail!("proposal pointer is not from the active checkpoint");
    }
    Ok(CheckpointInputs {
        checkpoint_sha256: active.to_owned(),
        checkpoint_pointer,
        checkpoint_artifact,
        proposals_pointer,
    })
}

fn assert_checkpoint_unchanged(args: &Args, expected: &str) -> Result<()> {
    let status = load_queue_status(args)?;
    let inputs = checkpoint_inputs(args, &status)?;
    if inputs.checkpoint_sha256 != expected {
        bail!("checkpoint advanced while a micro-wave was being built");
    }
    Ok(())
}

fn pointer_checkpoint(path: &Path) -> Result<String> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("parse checkpoint pointer {}", path.display()))?;
    let digest = value
        .get("checkpoint_sha256")
        .and_then(Value::as_str)
        .context("checkpoint pointer lacks checkpoint_sha256")?;
    validate_sha256(digest)?;
    Ok(digest.to_owned())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("invalid SHA-256 identity");
    }
    Ok(())
}

fn claimed_manifest_hashes(path: &Path) -> Result<BTreeSet<String>> {
    if !path.is_file() {
        return Ok(BTreeSet::new());
    }
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let mut statement = connection
        .prepare("SELECT DISTINCT manifest_sha256 FROM claims WHERE manifest_sha256!=''")?;
    let mut hashes = statement
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<BTreeSet<_>>>()?;
    drop(statement);
    let has_rejections: bool = connection.query_row(
        "SELECT EXISTS(
           SELECT 1 FROM sqlite_master
           WHERE type='table' AND name='rejected_manifests'
         )",
        [],
        |row| row.get(0),
    )?;
    if !has_rejections {
        bail!("rejection-aware keeper backlog lacks rejected_manifests");
    }
    let mut rejected =
        connection.prepare("SELECT manifest,manifest_sha256 FROM rejected_manifests")?;
    for row in rejected.query_map([], |row| {
        Ok((
            PathBuf::from(row.get::<_, String>(0)?),
            row.get::<_, Option<String>>(1)?,
        ))
    })? {
        let (manifest, recorded) = row?;
        let digest = match recorded.filter(|value| !value.is_empty()) {
            Some(digest) => digest,
            None if manifest.is_file() => sha256_file(&manifest)?,
            None => bail!(
                "rejected manifest lacks both a content hash and readable file: {}",
                manifest.display()
            ),
        };
        validate_sha256(&digest)?;
        hashes.insert(digest);
    }
    Ok(hashes)
}

fn ready_waves(
    waves_root: &Path,
    checkpoint_sha256: &str,
    checkpoint_epoch: &str,
    claimed: &BTreeSet<String>,
    limit: usize,
) -> Result<Vec<ReadyWave>> {
    if !waves_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut directories = fs::read_dir(waves_root)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_owned();
            valid_wave_id(&name).then_some((name, entry.path()))
        })
        .collect::<Vec<_>>();
    directories.sort_by(|left, right| right.0.cmp(&left.0));
    let mut waves = Vec::new();
    for (wave, directory) in directories {
        if waves.len() == limit {
            break;
        }
        let Some(ready) = validate_ready_wave(
            &wave,
            &directory,
            checkpoint_sha256,
            checkpoint_epoch,
            claimed,
        )?
        else {
            continue;
        };
        if ready.unclaimed_manifests > 0 {
            waves.push(ready);
        }
    }
    waves.sort_by(|left, right| left.wave.cmp(&right.wave));
    Ok(waves)
}

fn validate_ready_wave(
    wave: &str,
    directory: &Path,
    checkpoint_sha256: &str,
    checkpoint_epoch: &str,
    claimed: &BTreeSet<String>,
) -> Result<Option<ReadyWave>> {
    let summary_path = directory.join("summary.json");
    if !summary_path.is_file() {
        return Ok(None);
    }
    let summary: Value = serde_json::from_slice(&fs::read(&summary_path)?)
        .with_context(|| format!("parse {}", summary_path.display()))?;
    if summary.get("checkpoint_sha256").and_then(Value::as_str) != Some(checkpoint_sha256) {
        return Ok(None);
    }
    let realtime_schema = summary.get("schema").and_then(Value::as_str) == Some(WAVE_SCHEMA);
    let micro_wave_kind = summary.get("wave_kind").and_then(Value::as_str) == Some(MICRO_WAVE_KIND);
    if !realtime_schema && !micro_wave_kind {
        return Ok(None);
    }
    if !realtime_schema
        || !micro_wave_kind
        || summary.get("immutable").and_then(Value::as_bool) != Some(true)
        || summary.get("wave").and_then(Value::as_str) != Some(wave)
        || summary.get("checkpoint_epoch").and_then(Value::as_str) != Some(checkpoint_epoch)
    {
        bail!("{wave}: current-checkpoint immutable micro-wave summary invariants failed");
    }
    let entries = summary
        .get("manifests")
        .and_then(Value::as_array)
        .context("wave summary lacks manifests")?;
    if summary.get("manifest_count").and_then(Value::as_u64) != Some(entries.len() as u64) {
        bail!("{wave}: manifest count differs from immutable summary");
    }
    let mut hashes = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut unclaimed = 0;
    for entry in entries {
        let digest = entry
            .get("sha256")
            .and_then(Value::as_str)
            .context("wave manifest receipt lacks SHA-256")?;
        validate_sha256(digest)?;
        let relative = entry
            .get("relative_path")
            .and_then(Value::as_str)
            .context("wave manifest receipt lacks relative path")?;
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            || relative_path.parent() != Some(Path::new("manifests"))
        {
            bail!("{wave}: manifest path escapes its immutable wave");
        }
        if !hashes.insert(digest.to_owned()) || !paths.insert(relative.to_owned()) {
            bail!("{wave}: duplicate manifest receipt");
        }
        if !directory.join(relative_path).is_file() {
            bail!("{wave}: immutable manifest is absent");
        }
        if !claimed.contains(digest) {
            unclaimed += 1;
        }
    }
    Ok(Some(ReadyWave {
        wave: wave.to_owned(),
        summary_sha256: sha256_file(&summary_path)?,
        manifests: entries.len(),
        unclaimed_manifests: unclaimed,
    }))
}

fn manifests_per_wave(max_candidates_per_manifest: usize) -> usize {
    DATASETS_PER_WAVE * CANDIDATES_PER_DATASET.div_ceil(max_candidates_per_manifest)
}

#[allow(clippy::too_many_arguments)]
fn supply_decision(
    queue_depth: usize,
    ready_manifests: usize,
    current_ready_waves: usize,
    queue_floor: usize,
    queued_target: usize,
    queued_hard_cap: usize,
    ready_reserve_min: usize,
    max_overlap_waves: usize,
    max_builds_per_cycle: usize,
    max_candidates_per_manifest: usize,
) -> SupplyDecision {
    let projected_jobs = queue_depth.saturating_add(ready_manifests);
    let estimated_manifests_per_wave = manifests_per_wave(max_candidates_per_manifest);
    let queue_deficit = queued_target.saturating_sub(projected_jobs);
    let reserve_deficit = ready_reserve_min.saturating_sub(ready_manifests);
    let desired_manifests = queue_deficit.max(reserve_deficit);
    let waves_for_deficit = desired_manifests.div_ceil(estimated_manifests_per_wave);
    let waves_with_hard_cap_room =
        queued_hard_cap.saturating_sub(projected_jobs) / estimated_manifests_per_wave;
    let overlap_slots = max_overlap_waves.saturating_sub(current_ready_waves);
    SupplyDecision {
        below_floor: queue_depth < queue_floor,
        overloaded: projected_jobs > queued_target,
        projected_jobs,
        estimated_manifests_per_wave,
        builds_planned: waves_for_deficit
            .min(waves_with_hard_cap_room)
            .min(overlap_slots)
            .min(max_builds_per_cycle),
    }
}

fn next_wave_id(waves_root: &Path) -> Result<String> {
    let mut maximum = 0_u64;
    if waves_root.is_dir() {
        for entry in fs::read_dir(waves_root)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                maximum = maximum.max(parse_wave_id(name).unwrap_or(0));
            }
        }
        let pending = waves_root.join("pending");
        if pending.is_dir() {
            for entry in fs::read_dir(pending)? {
                let entry = entry?;
                let path = entry.path();
                let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                    continue;
                };
                maximum = maximum.max(
                    stem.strip_suffix(".cohort")
                        .and_then(parse_wave_id)
                        .unwrap_or(0),
                );
            }
        }
    }
    if maximum >= 999_999 {
        bail!("strict wave identity space exhausted");
    }
    Ok(format!("strict-wave-{:06}", maximum + 1))
}

fn wave_to_build(waves_root: &Path) -> Result<String> {
    let pending = waves_root.join("pending");
    let mut unresolved = Vec::new();
    if pending.is_dir() {
        for entry in fs::read_dir(&pending)? {
            let entry = entry?;
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let Some(wave) = stem.strip_suffix(".cohort") else {
                continue;
            };
            if valid_wave_id(wave)
                && !waves_root.join(wave).is_dir()
                && !pending_wave_quarantined(waves_root, wave, &path)?
            {
                unresolved.push(wave.to_owned());
            }
        }
    }
    unresolved.sort();
    unresolved.dedup();
    match unresolved.as_slice() {
        [] => next_wave_id(waves_root),
        [wave, ..] => Ok(wave.clone()),
    }
}

fn pending_wave_quarantined(waves_root: &Path, wave: &str, cohort: &Path) -> Result<bool> {
    let receipt_path = waves_root
        .join("quarantine")
        .join(format!("{wave}.receipt.json"));
    if !receipt_path.is_file() {
        return Ok(false);
    }
    let receipt: Value = serde_json::from_slice(&fs::read(&receipt_path)?)?;
    if receipt.get("schema").and_then(Value::as_str) != Some("CohortSupplyQuarantineReceipt.v1")
        || receipt.get("wave").and_then(Value::as_str) != Some(wave)
        || receipt.get("operator_authorized").and_then(Value::as_bool) != Some(true)
        || receipt
            .get("reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
            .is_none()
    {
        bail!("{wave}: invalid cohort-supply quarantine authority");
    }
    let cohort_sha256 = receipt
        .get("cohort_sha256")
        .and_then(Value::as_str)
        .context("cohort-supply quarantine lacks cohort hash")?;
    validate_sha256(cohort_sha256)?;
    if sha256_file(cohort)? != cohort_sha256 {
        bail!("{wave}: quarantined cohort content drift");
    }
    let state_sha256 = receipt
        .get("supply_state_sha256")
        .and_then(Value::as_str)
        .context("cohort-supply quarantine lacks supply-state hash")?;
    validate_sha256(state_sha256)?;
    let root = waves_root
        .parent()
        .context("waves root has no flywheel parent")?;
    let state = supply_state_path(root, wave);
    if !state.is_file() || sha256_file(&state)? != state_sha256 {
        bail!("{wave}: quarantined supply-state content drift");
    }
    Ok(true)
}

fn valid_wave_id(value: &str) -> bool {
    parse_wave_id(value).is_some()
}

fn parse_wave_id(value: &str) -> Option<u64> {
    let digits = value.strip_prefix("strict-wave-")?;
    if digits.len() != 6 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn supply_state_v1_path(root: &Path, wave: &str) -> PathBuf {
    root.join("ops/cohort-supply-state-v1")
        .join(format!("{wave}.json"))
}

fn supply_state_path(root: &Path, wave: &str) -> PathBuf {
    root.join("ops/cohort-supply-state-v2")
        .join(format!("{wave}.json"))
}

fn persist_supply_state(root: &Path, state: &PersistedSupplyState) -> Result<()> {
    if state.schema != SUPPLY_STATE_V2_SCHEMA {
        bail!("only CohortSupplyState.v2 may be persisted by the active supply lane");
    }
    state.validate(&state.wave, &state.checkpoint_sha256)?;
    atomic_json(
        &supply_state_path(root, &state.wave),
        &serde_json::to_value(state)?,
    )
}

fn load_supply_state(
    root: &Path,
    wave: &str,
    checkpoint: &str,
) -> Result<Option<PersistedSupplyState>> {
    let v2_path = supply_state_path(root, wave);
    if v2_path.is_file() {
        let state: PersistedSupplyState = serde_json::from_slice(&fs::read(&v2_path)?)?;
        state.validate(wave, checkpoint)?;
        if state.schema != SUPPLY_STATE_V2_SCHEMA {
            bail!("v2 cohort supply path contains a non-v2 document");
        }
        return Ok(Some(state));
    }
    let v1_path = supply_state_v1_path(root, wave);
    if !v1_path.is_file() {
        return Ok(None);
    }
    let mut state: PersistedSupplyState = serde_json::from_slice(&fs::read(&v1_path)?)?;
    state.validate(wave, checkpoint)?;
    if state.schema != SUPPLY_STATE_V1_SCHEMA {
        bail!("legacy cohort supply path contains an unexpected schema");
    }
    state.schema = SUPPLY_STATE_V2_SCHEMA.to_owned();
    let job_identity = state
        .proposal_job_id
        .as_deref()
        .context("legacy cohort supply state lacks proposer identity")?;
    let spec = root
        .join("specs/cohort-propose")
        .join(format!("{job_identity}.toml"));
    if spec.is_file() {
        state
            .runtime_pinned_attempts
            .push(runtime_pinned_intent_from_spec(
                &spec,
                state.proposal_attempt,
            )?);
    } else if state.stage.ordinal() >= PersistedSupplyStage::SubmitCohortProposer.ordinal() {
        bail!("submitted legacy cohort supply state lacks its immutable proposer spec");
    }
    state.revision = state.revision.saturating_add(1);
    persist_supply_state(root, &state)?;
    Ok(Some(state))
}

fn mark_keeper_acknowledged(root: &Path, wave: &str, checkpoint: &str) -> Result<()> {
    let Some(mut state) = load_supply_state(root, wave, checkpoint)? else {
        return Ok(());
    };
    if state.stage == PersistedSupplyStage::KeeperAcknowledgment {
        return Ok(());
    }
    if state.stage != PersistedSupplyStage::BuildWave {
        bail!("{wave}: keeper acknowledged an incomplete supply state");
    }
    state.advance(PersistedSupplyStage::KeeperAcknowledgment)?;
    persist_supply_state(root, &state)
}

fn toml_quote(value: &str) -> Result<String> {
    if value.contains(['"', '\\', '\n', '\r']) {
        bail!("value is not safe for generated TOML");
    }
    Ok(format!("\"{value}\""))
}

fn toml_string_assignment(line: &str, name: &str) -> Option<String> {
    let value = line.trim().strip_prefix(&format!("{name} = "))?;
    serde_json::from_str::<String>(value).ok()
}

fn proposer_inputs_from_spec(body: &str) -> Result<Vec<PinnedProposerInput>> {
    let mut inputs = Vec::new();
    let mut current: Option<PinnedProposerInput> = None;
    let flush = |current: &mut Option<PinnedProposerInput>,
                 inputs: &mut Vec<PinnedProposerInput>|
     -> Result<()> {
        if let Some(input) = current.take() {
            if input.name.is_empty() || input.path.as_os_str().is_empty() {
                bail!("cohort proposer spec contains an incomplete input");
            }
            validate_sha256(&input.sha256)?;
            inputs.push(input);
        }
        Ok(())
    };
    for line in body.lines() {
        if line.trim() == "[[input]]" {
            flush(&mut current, &mut inputs)?;
            current = Some(PinnedProposerInput {
                name: String::new(),
                path: PathBuf::new(),
                sha256: String::new(),
            });
            continue;
        }
        let Some(input) = current.as_mut() else {
            continue;
        };
        if let Some(value) = toml_string_assignment(line, "name") {
            input.name = value;
        } else if let Some(value) = toml_string_assignment(line, "source") {
            input.path = PathBuf::from(value);
        } else if let Some(value) = toml_string_assignment(line, "sha256") {
            input.sha256 = value;
        } else if line.trim().starts_with("[[") || line.trim().starts_with('[') {
            flush(&mut current, &mut inputs)?;
        }
    }
    flush(&mut current, &mut inputs)?;
    inputs.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(inputs)
}

fn runtime_pinned_intent_from_spec(spec: &Path, attempt: u32) -> Result<RuntimePinnedIntentV1> {
    let body = fs::read_to_string(spec)
        .with_context(|| format!("read immutable proposer spec {}", spec.display()))?;
    let job_identity = body
        .lines()
        .find_map(|line| toml_string_assignment(line, "id"))
        .context("cohort proposer spec lacks job identity")?;
    let pythonpath = fs::canonicalize(proposer_pythonpath(&body)?)
        .context("canonicalize pinned proposer PYTHONPATH")?;
    if pythonpath.file_name().and_then(|value| value.to_str()) != Some("pytorch") {
        bail!("pinned proposer PYTHONPATH is not inside a runtime bundle");
    }
    let runtime_bundle = pythonpath
        .parent()
        .context("pinned proposer PYTHONPATH has no runtime bundle parent")?
        .to_path_buf();
    let runtime_bundle_sha256 = runtime_bundle
        .file_name()
        .and_then(|value| value.to_str())
        .context("pinned proposer runtime bundle has no content identity")?
        .to_owned();
    validate_sha256(&runtime_bundle_sha256)?;
    let proposer_inputs = proposer_inputs_from_spec(&body)?;
    let pointer = proposer_inputs
        .iter()
        .find(|input| input.name == "pointer")
        .context("pinned proposer spec has no catalog pointer input")?;
    let value = RuntimePinnedIntentV1 {
        schema: RUNTIME_PINNED_INTENT_SCHEMA.to_owned(),
        runtime_bundle,
        runtime_bundle_sha256,
        catalog_snapshot: pointer.path.clone(),
        catalog_snapshot_sha256: pointer.sha256.clone(),
        proposer_inputs,
        immutable_spec_path: fs::canonicalize(spec).unwrap_or_else(|_| spec.to_path_buf()),
        immutable_spec_sha256: sha256_file(spec)?,
        job_identity,
        attempt,
    };
    value.validate()?;
    Ok(value)
}

fn adaptive_proposal_reservoir(root: &Path) -> usize {
    let waves = root.join("waves");
    let Ok(entries) = fs::read_dir(waves) else {
        return 64;
    };
    let mut summaries = entries
        .flatten()
        .filter_map(|entry| {
            let wave = entry.file_name().to_string_lossy().into_owned();
            let number = parse_wave_id(&wave)?;
            let summary = entry.path().join("summary.json");
            summary.is_file().then_some((number, summary))
        })
        .collect::<Vec<_>>();
    summaries.sort_by_key(|(wave, _)| std::cmp::Reverse(*wave));
    let deficient = summaries
        .into_iter()
        .take(2)
        .map(|(_, path)| {
            fs::read(path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|value| {
                    value
                        .pointer("/wave_summary/actual_d_by_task/binary")
                        .and_then(Value::as_array)
                        .map(|values| {
                            values
                                .iter()
                                .filter_map(Value::as_u64)
                                .any(|actual| actual < COHORT_NATIVE_D as u64)
                        })
                })
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    match deficient.as_slice() {
        [true, true] => 256,
        [true, ..] => 128,
        _ => 64,
    }
}

fn proposal_decoder_configuration_sha256(args: &Args, reservoir: usize) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(&json!({
        "generator_sha256": sha256_file(&args.proposal_generator)?,
        "legacy_generator_sha256": sha256_file(&args.proposal_legacy_generator)?,
        "common_sha256": sha256_file(&args.proposal_common)?,
        "seed": COHORT_PROPOSAL_SEED,
        "proposal_reservoir_start": 64,
        "proposal_reservoir_maximum": 256,
        "legacy_adaptive_reservoir": reservoir,
        "selected_native_d": COHORT_NATIVE_D,
        "context_addressed": true,
        "selection_policy": "calibrated-or-empirical-dispersion-head-diversity-v3",
    }))?))
}

fn append_spec_input(
    lines: &mut Vec<String>,
    name: &str,
    source: &Path,
    target: &str,
) -> Result<()> {
    lines.push("[[input]]".into());
    lines.push(format!("name = {}", toml_quote(name)?));
    lines.push(format!(
        "source = {}",
        toml_quote(&source.to_string_lossy())?
    ));
    lines.push(format!("target = {}", toml_quote(target)?));
    lines.push(format!("sha256 = {}", toml_quote(&sha256_file(source)?)?));
    lines.push(String::new());
    Ok(())
}

fn normalized_proposer_spec(body: &str) -> String {
    body.lines()
        .map(|line| {
            if line.starts_with("PYTHONPATH = ") {
                "PYTHONPATH = \"<content-verified-runtime>\""
            } else if line.starts_with("source = ")
                && (line.ends_with("/pytorch/proposals/test_exact_context_proposals.py\"")
                    || line.ends_with("/pytorch/proposals/test_exact_dl_generate.py\"")
                    || line.ends_with("/pytorch/shared/test_exact_common.py\""))
            {
                "source = \"<content-verified-runtime-input>\""
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn proposer_pythonpath(body: &str) -> Result<PathBuf> {
    let line = body
        .lines()
        .find(|line| line.starts_with("PYTHONPATH = "))
        .context("cohort proposer spec lacks PYTHONPATH")?;
    let value = line
        .strip_prefix("PYTHONPATH = \"")
        .and_then(|value| value.strip_suffix('"'))
        .context("cohort proposer PYTHONPATH is malformed")?;
    if value.is_empty() || value.contains(['\\', '\n', '\r', '"']) {
        bail!("cohort proposer PYTHONPATH is unsafe");
    }
    Ok(PathBuf::from(value))
}

fn validate_proposer_runtime(existing_body: &str, expected_body: &str, args: &Args) -> Result<()> {
    if normalized_proposer_spec(existing_body) != normalized_proposer_spec(expected_body) {
        bail!("immutable cohort proposer spec drift");
    }
    let pythonpath = proposer_pythonpath(existing_body)?;
    if !pythonpath.is_dir()
        || sha256_file(&pythonpath.join("proposals/test_exact_context_proposals.py"))?
            != sha256_file(&args.proposal_generator)?
        || sha256_file(&pythonpath.join("proposals/test_exact_dl_generate.py"))?
            != sha256_file(&args.proposal_legacy_generator)?
        || sha256_file(&pythonpath.join("shared/test_exact_common.py"))?
            != sha256_file(&args.proposal_common)?
        || !pythonpath.join("jope/__init__.py").is_file()
        || !pythonpath.join("jope/policy/model_v3.py").is_file()
        || !pythonpath.join("jope/policy/decode.py").is_file()
    {
        bail!("immutable cohort proposer runtime dependency closure drift");
    }
    Ok(())
}

fn immutable_json(path: &Path, value: &Value) -> Result<()> {
    if path.is_file() {
        let existing: Value = serde_json::from_slice(&fs::read(path)?)?;
        if existing != *value {
            bail!("immutable JSON artifact drift: {}", path.display());
        }
        return Ok(());
    }
    atomic_json(path, value)
}

fn pointer_target(pointer_path: &Path, target: &str) -> PathBuf {
    let path = PathBuf::from(target);
    if path.is_absolute() {
        path
    } else {
        pointer_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}

fn context_proposal_request(
    args: &Args,
    state: &PersistedSupplyState,
    pointer_snapshot: &Path,
    catalog_pointer: &Value,
    runtime_bundle_sha256: &str,
) -> Result<PathBuf> {
    let cohort: Value = serde_json::from_slice(&fs::read(&state.cohort)?)?;
    if sha256_file(&state.cohort)? != state.cohort_sha256 {
        bail!("frozen cohort changed before proposal request creation");
    }
    let source_schema = cohort
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or("JopeMicroCohort.v2");
    let entries = cohort
        .get("entries")
        .and_then(Value::as_array)
        .context("frozen cohort lacks entries")?;
    if entries.is_empty() {
        bail!("context proposal request cohort is empty");
    }

    if catalog_pointer
        .get("catalog_sha256")
        .and_then(Value::as_str)
        != Some(state.catalog_sha256.as_str())
    {
        bail!("frozen catalog pointer does not match the cohort binding");
    }
    let catalog_target = catalog_pointer
        .get("catalog")
        .and_then(Value::as_str)
        .context("frozen catalog pointer lacks catalog")?;
    let catalog_path = pointer_target(pointer_snapshot, catalog_target);
    if !catalog_path.is_file() || sha256_file(&catalog_path)? != state.catalog_sha256 {
        bail!("frozen catalog artifact content drift");
    }
    let catalog: Value = serde_json::from_slice(&fs::read(&catalog_path)?)?;
    let catalog_schema = catalog
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or("legacy")
        .to_owned();
    let catalog_entries = catalog
        .get("entries")
        .and_then(Value::as_array)
        .context("frozen catalog lacks entries")?;
    let data_sha256 = catalog_pointer
        .get("data_sha256")
        .and_then(Value::as_str)
        .context("frozen catalog pointer lacks context data identity")?;
    validate_sha256(data_sha256)?;
    if catalog.get("data_sha256").and_then(Value::as_str) != Some(data_sha256) {
        bail!("frozen catalog and pointer disagree on context data");
    }
    let data_target = catalog_pointer
        .get("data")
        .and_then(Value::as_str)
        .context("frozen catalog pointer lacks context data")?;
    let data_path = pointer_target(pointer_snapshot, data_target);
    if !data_path.is_file() || sha256_file(&data_path)? != data_sha256 {
        bail!("frozen catalog context data drift");
    }
    let stride = catalog
        .get("record_stride_bytes")
        .and_then(Value::as_u64)
        .context("frozen catalog lacks context stride")?;
    if stride == 0 || data_path.metadata()?.len() != stride * catalog_entries.len() as u64 {
        bail!("frozen catalog context topology drift");
    }
    let mut by_dataset = BTreeMap::<(String, String), Vec<String>>::new();
    let mut data = File::open(&data_path)?;
    let mut context_bytes = vec![0_u8; stride as usize];
    for entry in catalog_entries {
        let task = entry
            .get("task")
            .and_then(Value::as_str)
            .context("frozen catalog entry lacks task")?;
        let dataset_hash = entry
            .get("dataset_hash")
            .and_then(Value::as_str)
            .context("frozen catalog entry lacks dataset hash")?;
        let context_sha256 = entry
            .get("context_sha256")
            .and_then(Value::as_str)
            .context("frozen catalog entry lacks context hash")?;
        validate_sha256(context_sha256)?;
        let index = entry
            .get("catalog_index")
            .and_then(Value::as_u64)
            .context("frozen catalog entry lacks context index")?;
        if index >= catalog_entries.len() as u64 {
            bail!("frozen catalog context index is out of range");
        }
        data.seek(SeekFrom::Start(index * stride))?;
        data.read_exact(&mut context_bytes)?;
        if sha256_bytes(&context_bytes) != context_sha256 {
            bail!("frozen catalog entry context bytes drift");
        }
        by_dataset
            .entry((task.to_owned(), dataset_hash.to_owned()))
            .or_default()
            .push(context_sha256.to_owned());
    }

    let mut identities = BTreeSet::new();
    let mut contexts = Vec::with_capacity(entries.len());
    let mut compatibility = Vec::with_capacity(entries.len());
    let mut exclusions = Vec::new();
    for (entry_index, entry) in entries.iter().enumerate() {
        let task = entry
            .get("task")
            .and_then(Value::as_str)
            .context("frozen cohort entry lacks task")?;
        let dataset_hash = entry
            .get("dataset_hash")
            .and_then(Value::as_str)
            .context("frozen cohort entry lacks dataset hash")?;
        if !matches!(task, "binary" | "regression" | "timeseries" | "multiclass") {
            bail!("frozen cohort has an invalid task");
        }
        let matches = by_dataset
            .get(&(task.to_owned(), dataset_hash.to_owned()))
            .cloned()
            .unwrap_or_default();
        let source_context = entry.get("context_sha256").and_then(Value::as_str);
        let (context_sha256, resolution) = match (source_context, matches.as_slice()) {
            (Some(source), [catalog_context]) if source == catalog_context => {
                validate_sha256(source)?;
                (Some(source.to_owned()), "preserved")
            }
            (None, [catalog_context]) => (Some(catalog_context.clone()), "resolved"),
            (Some(_), [..]) => (None, "context_catalog_mismatch"),
            (_, []) => (None, "absent_context"),
            (_, [_, ..]) => (None, "ambiguous_context"),
        };
        let Some(context_sha256) = context_sha256 else {
            compatibility.push(json!({
                "entry_index": entry_index,
                "task": task,
                "dataset_hash": dataset_hash,
                "status": "excluded",
                "reason": resolution,
                "context_sha256": Value::Null,
            }));
            exclusions.push(json!({
                "schema": COHORT_CONTEXT_EXCLUSION_SCHEMA,
                "source_cohort_sha256": state.cohort_sha256,
                "source_entry_index": entry_index,
                "task": task,
                "dataset_hash": dataset_hash,
                "reason": resolution,
                "frozen_catalog_sha256": state.catalog_sha256,
            }));
            continue;
        };
        if !identities.insert((
            task.to_owned(),
            dataset_hash.to_owned(),
            context_sha256.clone(),
        )) {
            bail!("frozen cohort has a duplicate resolved context identity");
        }
        compatibility.push(json!({
            "entry_index": entry_index,
            "task": task,
            "dataset_hash": dataset_hash,
            "status": resolution,
            "reason": Value::Null,
            "context_sha256": context_sha256,
        }));
        contexts.push(json!({
            "task": task,
            "dataset_hash": dataset_hash,
            "context_sha256": context_sha256,
            "micro_lane": entry.get("micro_lane")
                .or_else(|| entry.get("lane"))
                .cloned()
                .unwrap_or(Value::Null),
        }));
    }
    let directory = args.root.join("proposals/cohort-inputs");
    fs::create_dir_all(&directory)?;
    if contexts.is_empty() {
        let receipt = json!({
            "schema": COHORT_CONTEXT_MIGRATION_SCHEMA,
            "source_cohort_sha256": state.cohort_sha256,
            "source_schema": source_schema,
            "frozen_catalog_sha256": state.catalog_sha256,
            "frozen_catalog_data_identity": data_sha256,
            "request_sha256": Value::Null,
            "resolved_entries": 0,
            "excluded_entries": exclusions.len(),
            "exclusions": exclusions,
            "compatibility_resolution": compatibility,
        });
        let receipt_path = directory.join(format!(
            "{}.{}.context-migration.v1.json",
            state.cohort_sha256, state.catalog_sha256
        ));
        immutable_json(&receipt_path, &receipt)?;
        bail!("no frozen cohort entries resolve against the frozen catalog");
    }
    let resolved_entries = contexts.len();
    let request = json!({
        "schema": "ContextProposalRequest.v4",
        "task": "mixed",
        "source_cohort_schema": source_schema,
        "source_cohort_hash": state.cohort_sha256,
        "catalog_schema": catalog_schema,
        "catalog_sha256": state.catalog_sha256,
        "catalog_data_identity": data_sha256,
        "accepted_parent_hash": state.checkpoint_sha256,
        "accepted_checkpoint_sha256": state.checkpoint_sha256,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "seed": state.seed,
        "cohort_sha256": state.cohort_sha256,
        "requested_contexts": contexts.clone(),
        "contexts": contexts.clone(),
        "compatibility_resolution": compatibility.clone(),
    });
    let path = directory.join(format!(
        "{}.{}.request.v4.json",
        state.cohort_sha256, state.checkpoint_sha256
    ));
    immutable_json(&path, &request)?;
    let request_sha256 = sha256_file(&path)?;
    let receipt = json!({
        "schema": COHORT_CONTEXT_MIGRATION_SCHEMA,
        "source_cohort_sha256": state.cohort_sha256,
        "source_schema": source_schema,
        "frozen_catalog_sha256": state.catalog_sha256,
        "frozen_catalog_data_identity": data_sha256,
        "request_sha256": request_sha256.clone(),
        "resolved_entries": resolved_entries,
        "excluded_entries": exclusions.len(),
        "exclusions": exclusions,
        "compatibility_resolution": compatibility,
    });
    let receipt_path = directory.join(format!(
        "{}.{}.{}.context-migration.v1.json",
        state.cohort_sha256, state.catalog_sha256, request_sha256
    ));
    immutable_json(&receipt_path, &receipt)?;
    Ok(path)
}

fn frozen_catalog_pointer(
    live_pointer: &Path,
    pointer_snapshot: &Path,
    expected_catalog_sha256: &str,
) -> Result<Value> {
    let source = if pointer_snapshot.is_file() {
        pointer_snapshot
    } else {
        live_pointer
    };
    let pointer: Value = serde_json::from_slice(&fs::read(source)?)?;
    if pointer.get("catalog_sha256").and_then(Value::as_str) != Some(expected_catalog_sha256) {
        bail!(
            "{} catalog identity does not match the frozen cohort",
            if source == pointer_snapshot {
                "snapshot"
            } else {
                "live"
            }
        );
    }
    if source == live_pointer {
        atomic_json(pointer_snapshot, &pointer)?;
    }
    Ok(pointer)
}

fn render_cohort_proposer_spec(
    args: &Args,
    inputs: &CheckpointInputs,
    state: &PersistedSupplyState,
) -> Result<PathBuf> {
    let pointer_directory = args.root.join("proposals/cohort-inputs");
    fs::create_dir_all(&pointer_directory)?;
    let pointer_snapshot = pointer_directory.join(format!("{}.pointer.json", state.catalog_sha256));
    // Once a cohort is frozen, resume exclusively from its immutable catalog
    // snapshot. Breadth encoding may advance the live catalog independently
    // while a proposer is queued or running.
    let catalog_pointer = frozen_catalog_pointer(
        &args.catalog_pointer(),
        &pointer_snapshot,
        &state.catalog_sha256,
    )?;
    let catalog = PathBuf::from(
        catalog_pointer
            .get("catalog")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog")?,
    );
    let data = PathBuf::from(
        catalog_pointer
            .get("data")
            .and_then(Value::as_str)
            .context("catalog pointer lacks data")?,
    );
    let job_id = state
        .proposal_job_id
        .as_deref()
        .context("cohort proposer job identity is absent")?;
    let pythonpath = fs::canonicalize(&args.proposal_pythonpath)
        .unwrap_or_else(|_| args.proposal_pythonpath.clone());
    let runtime_bundle_sha256 = pythonpath
        .parent()
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        .context("cohort proposer runtime bundle has no content identity")?
        .to_owned();
    validate_sha256(&runtime_bundle_sha256)?;
    let proposal_generator = fs::canonicalize(&args.proposal_generator)
        .unwrap_or_else(|_| args.proposal_generator.clone());
    let proposal_legacy_generator = fs::canonicalize(&args.proposal_legacy_generator)
        .unwrap_or_else(|_| args.proposal_legacy_generator.clone());
    let proposal_common =
        fs::canonicalize(&args.proposal_common).unwrap_or_else(|_| args.proposal_common.clone());
    let request = context_proposal_request(
        args,
        state,
        &pointer_snapshot,
        &catalog_pointer,
        &runtime_bundle_sha256,
    )?;
    let cache_root = args.root.join("proposals/context-cache.v3");
    let args_line = format!(
        "args = [\"{{input:generator}}\", \"--catalog-pointer\", \"{{input:pointer}}\", \
         \"--catalog\", \"{{input:catalog}}\", \"--data\", \"{{input:data}}\", \
         \"--checkpoint\", \"{{input:checkpoint}}\", \"--request\", \"{{input:request}}\", \
         \"--cache-root\", {}, \"--output\", \"{{output:proposals}}/dl-proposals.jsonl\", \
         \"--device\", \"cuda\"]",
        toml_quote(&cache_root.to_string_lossy())?
    );
    let mut lines = vec![
        "[job]".into(),
        format!("id = {}", toml_quote(job_id)?),
        "project = \"jope-selfplay-train\"".into(),
        "tier = \"normal\"".into(),
        "command = \"/usr/bin/python3\"".into(),
        args_line,
        String::new(),
        "[resource]".into(),
        "gpus_per_task = 1".into(),
        "cpus_reserved = 4".into(),
        "omp_threads = 4".into(),
        "ram_mb = 12288".into(),
        "gang_size = 1".into(),
        "placement = \"single_node\"".into(),
        String::new(),
        "[[dependencies.smoke]]".into(),
        "name = \"proposal GPU capacity\"".into(),
        "command = \"/usr/bin/nvidia-smi\"".into(),
        "args = [\"--query-gpu=uuid,memory.total\", \"--format=csv,noheader\"]".into(),
        "timeout_secs = 5".into(),
        String::new(),
        "[[dependencies.smoke]]".into(),
        "name = \"RuntimeBundle JOPE Python package\"".into(),
        "command = \"/usr/bin/env\"".into(),
        format!(
            "args = [{}, \"/usr/bin/python3\", \"-c\", \"import jope\"]",
            toml_quote(&format!("PYTHONPATH={}", pythonpath.to_string_lossy()))?
        ),
        "timeout_secs = 30".into(),
        String::new(),
        "[env]".into(),
        format!(
            "PYTHONPATH = {}",
            toml_quote(&pythonpath.to_string_lossy())?
        ),
        "OMP_NUM_THREADS = \"4\"".into(),
        String::new(),
        "[preempt]".into(),
        "preemptible = true".into(),
        "checkpoint = \"none\"".into(),
        "resume = \"rerun\"".into(),
        String::new(),
        "[limits]".into(),
        "max_runtime_hours = 24".into(),
        "max_restarts = 1".into(),
        String::new(),
    ];
    append_spec_input(
        &mut lines,
        "generator",
        &proposal_generator,
        "test_exact_context_proposals.py",
    )?;
    append_spec_input(
        &mut lines,
        "legacy_generator",
        &proposal_legacy_generator,
        "test_exact_dl_generate.py",
    )?;
    append_spec_input(
        &mut lines,
        "common",
        &proposal_common,
        "test_exact_common.py",
    )?;
    append_spec_input(
        &mut lines,
        "pointer",
        &pointer_snapshot,
        "catalog/pointer.json",
    )?;
    append_spec_input(&mut lines, "catalog", &catalog, "catalog/catalog.json")?;
    append_spec_input(&mut lines, "data", &data, "catalog/contexts.f32.bin")?;
    append_spec_input(
        &mut lines,
        "checkpoint",
        &inputs.checkpoint_artifact,
        &format!("checkpoint/{}.model.pt", state.checkpoint_sha256),
    )?;
    append_spec_input(&mut lines, "cohort", &state.cohort, "cohort/cohort.json")?;
    append_spec_input(
        &mut lines,
        "request",
        &request,
        "cohort/context-proposal-request.v4.json",
    )?;
    lines.extend([
        "[[output]]".into(),
        "name = \"proposals\"".into(),
        "source = \"proposals\"".into(),
        "directory = true".into(),
        String::new(),
    ]);
    let directory = args.root.join("specs/cohort-propose");
    fs::create_dir_all(&directory)?;
    let spec = directory.join(format!("{job_id}.toml"));
    let body = lines.join("\n");
    if spec.is_file() {
        bail!(
            "refusing to re-render existing immutable cohort proposer spec {}",
            spec.display()
        );
    }
    fs::write(&spec, body)?;
    Ok(spec)
}

fn resolve_pinned_proposer_spec(
    args: &Args,
    inputs: &CheckpointInputs,
    state: &mut PersistedSupplyState,
) -> Result<PathBuf> {
    if let Some(intent) = state.runtime_pinned_attempts.last() {
        intent.validate()?;
        if Some(intent.job_identity.as_str()) != state.proposal_job_id.as_deref()
            || intent.attempt != state.proposal_attempt
        {
            bail!("current runtime-pinned proposer attempt identity drift");
        }
        return Ok(intent.immutable_spec_path.clone());
    }
    let job_identity = state
        .proposal_job_id
        .as_deref()
        .context("cohort proposer job identity is absent")?;
    let candidate = args
        .root
        .join("specs/cohort-propose")
        .join(format!("{job_identity}.toml"));
    let spec = if candidate.is_file() {
        candidate
    } else {
        render_cohort_proposer_spec(args, inputs, state)?
    };
    let pinned = runtime_pinned_intent_from_spec(&spec, state.proposal_attempt)?;
    if pinned.job_identity != job_identity {
        bail!("cohort proposer spec job identity drift");
    }
    state.runtime_pinned_attempts.push(pinned);
    state.revision = state.revision.saturating_add(1);
    state.validate(&state.wave, &state.checkpoint_sha256)?;
    Ok(spec)
}

fn run_scq(args: &Args, arguments: &[&str]) -> Result<String> {
    let output = Command::new(&args.scq)
        .args(arguments)
        .output()
        .with_context(|| format!("spawn {} {:?}", args.scq.display(), arguments))?;
    if !output.status.success() {
        bail!(
            "scq {:?} failed: {}",
            arguments,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn temporarily_lacks_launch_capacity(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && (message.contains("no eligible launch-ready node currently")
            || message.contains("eligible_node launch-ready-capacity")
            || (message.contains("smoke proposal GPU capacity")
                && message.contains("smoke test exited")))
}

fn register_proposer_admission(args: &Args, spec: &Path) -> Result<()> {
    let job_id = fs::read_to_string(spec)?
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("id = ")
                .and_then(|value| serde_json::from_str::<String>(value).ok())
        })
        .context("cohort proposer spec lacks job identity")?;
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    let controller = executable
        .parent()
        .context("supply executable has no bin directory")?
        .join("jope_gpu_lane_admission");
    let output = Command::new(&controller)
        .args([
            "--root",
            args.root.to_string_lossy().as_ref(),
            "--register-role",
            "cohort-proposer",
            "--spec",
            spec.to_string_lossy().as_ref(),
            "--job-id",
            &job_id,
        ])
        .output()
        .with_context(|| {
            format!(
                "register proposer admission through {}",
                controller.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "cohort proposer GPU admission request failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Capacity-only failures are handed to the crash-safe GPU admission owner.
/// The supply state remains at the submit stage until that owner has placed
/// the immutable spec.
fn scq_check_submit(args: &Args, spec: &Path) -> Result<bool> {
    let spec_text = spec.to_string_lossy();
    let check = Command::new(&args.scq)
        .args(["check", spec_text.as_ref()])
        .output()
        .with_context(|| format!("spawn {} check {}", args.scq.display(), spec.display()))?;
    let check_message = String::from_utf8_lossy(if check.stderr.is_empty() {
        &check.stdout
    } else {
        &check.stderr
    });
    if !check.status.success() && temporarily_lacks_launch_capacity(&check_message) {
        register_proposer_admission(args, spec)?;
        return Ok(false);
    }
    if !check.status.success() {
        bail!(
            "SCQ check rejected cohort proposer: {}",
            check_message.trim()
        );
    }

    let submit = Command::new(&args.scq)
        .args(["submit", spec_text.as_ref()])
        .output()
        .with_context(|| format!("spawn {} submit {}", args.scq.display(), spec.display()))?;
    let submit_message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    if submit.status.success()
        || submit_message
            .to_ascii_lowercase()
            .contains("already exists")
    {
        return Ok(true);
    }
    if temporarily_lacks_launch_capacity(&submit_message) {
        register_proposer_admission(args, spec)?;
        return Ok(false);
    }
    bail!(
        "SCQ submit rejected cohort proposer: {}",
        submit_message.trim()
    )
}

fn proposer_job_state(args: &Args, job_id: &str) -> Result<Option<String>> {
    let value: Value = serde_json::from_str(&run_scq(args, &["--json", "ls"])?)?;
    Ok(value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .and_then(|jobs| {
            jobs.iter().find_map(|job| {
                (job.get("id").and_then(Value::as_str) == Some(job_id)).then(|| {
                    job.get("state")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_owned()
                })
            })
        }))
}

fn recursive_find(directory: &Path, name: &str, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            recursive_find(&path, name, found);
        } else if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            found.push(path);
        }
    }
}

fn fetch_cohort_proposals(args: &Args, state: &PersistedSupplyState) -> Result<(PathBuf, String)> {
    let job_id = state
        .proposal_job_id
        .as_deref()
        .context("proposal job identity is absent")?;
    let incoming = args.root.join("proposals/_incoming").join(job_id);
    if incoming.exists() {
        fs::remove_dir_all(&incoming)?;
    }
    fs::create_dir_all(&incoming)?;
    run_scq(
        args,
        &["fetch", job_id, "--output", &incoming.to_string_lossy()],
    )?;
    let mut proposals = Vec::new();
    let mut summaries = Vec::new();
    recursive_find(&incoming, "dl-proposals.jsonl", &mut proposals);
    recursive_find(&incoming, "dl-proposals.summary.json", &mut summaries);
    let proposal_source = proposals
        .into_iter()
        .next()
        .context("cohort proposer output lacks proposals")?;
    let summary_source = summaries
        .into_iter()
        .next()
        .context("cohort proposer output lacks summary")?;
    let summary: Value = serde_json::from_slice(&fs::read(&summary_source)?)?;
    let proposal_sha256 = sha256_file(&proposal_source)?;
    if summary
        .get("model_checkpoint_sha256")
        .and_then(Value::as_str)
        != Some(state.checkpoint_sha256.as_str())
        || summary.get("catalog_sha256").and_then(Value::as_str)
            != Some(state.catalog_sha256.as_str())
        || summary.get("cohort_sha256").and_then(Value::as_str)
            != Some(state.cohort_sha256.as_str())
        || summary.get("output_sha256").and_then(Value::as_str) != Some(proposal_sha256.as_str())
        || summary.get("datasets").and_then(Value::as_u64) != Some(DATASETS_PER_WAVE as u64)
        || summary
            .get("zero_cross_context_backfill")
            .and_then(Value::as_bool)
            != Some(true)
    {
        bail!("cohort proposer output provenance drift");
    }
    let validity = summary
        .get("proposal_validity")
        .and_then(Value::as_f64)
        .context("cohort proposer summary lacks validity")?;
    if validity < 0.99 {
        bail!("cohort proposal validity {validity:.6} is below 0.99");
    }
    let cohort_value: Value = serde_json::from_slice(&fs::read(&state.cohort)?)?;
    let mut task_counts = BTreeMap::<String, usize>::new();
    for entry in cohort_value
        .get("entries")
        .and_then(Value::as_array)
        .context("frozen cohort lacks entries")?
    {
        let task = entry
            .get("task")
            .and_then(Value::as_str)
            .context("frozen cohort entry lacks task")?;
        *task_counts.entry(task.to_owned()).or_default() += 1;
    }
    if task_counts.values().any(|count| *count != 64) {
        bail!("cohort proposal task scope is not 64 contexts per task");
    }
    let destination = args
        .root
        .join("proposals/by-cohort")
        .join(&state.cohort_sha256);
    fs::create_dir_all(&destination)?;
    let proposals_path = destination.join(format!("{proposal_sha256}.jsonl"));
    if !proposals_path.is_file() {
        fs::copy(&proposal_source, &proposals_path)?;
    }
    if sha256_file(&proposals_path)? != proposal_sha256 {
        bail!("published cohort proposal content drift");
    }
    let summary_sha256 = sha256_file(&summary_source)?;
    let summary_path = destination.join(format!("{summary_sha256}.summary.json"));
    if !summary_path.is_file() {
        fs::copy(&summary_source, &summary_path)?;
    }
    let task_scopes = task_counts
        .into_iter()
        .map(|(task, context_count)| {
            json!({
                "task": task,
                "accepted_checkpoint_sha256": state.checkpoint_sha256,
                "context_count": context_count,
            })
        })
        .collect::<Vec<_>>();
    let pointer = json!({
        "schema": "CohortProposalPointer.v1",
        "immutable": true,
        "task_scopes": task_scopes,
        "cohort": state.cohort,
        "cohort_sha256": state.cohort_sha256,
        "catalog_sha256": state.catalog_sha256,
        "checkpoint_sha256": state.checkpoint_sha256,
        "decoder_configuration_sha256": state.decoder_configuration_sha256,
        "seed": state.seed,
        "proposal_job_id": job_id,
        "proposals": proposals_path,
        "proposals_sha256": proposal_sha256,
        "summary": summary_path,
        "summary_sha256": summary_sha256,
        "dl_slots_by_task": {"binary": 8, "regression": 8},
        "cross_context_backfill_rows": 0,
    });
    let mut bytes = serde_json::to_vec(&pointer)?;
    bytes.push(b'\n');
    let pointer_sha256 = sha256_bytes(&bytes);
    let pointer_path = destination.join(format!("{pointer_sha256}.pointer.json"));
    if pointer_path.is_file() {
        if fs::read(&pointer_path)? != bytes {
            bail!("immutable cohort proposal pointer drift");
        }
    } else {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pointer_path)?;
        file.write_all(&bytes)?;
        file.sync_data()?;
    }
    Ok((pointer_path, pointer_sha256))
}

fn built_wave_from_disk(args: &Args, state: &PersistedSupplyState) -> Result<BuiltWave> {
    let directory = args.root.join("waves").join(&state.wave);
    let summary: Value = serde_json::from_slice(&fs::read(directory.join("summary.json"))?)?;
    Ok(BuiltWave {
        wave: state.wave.clone(),
        cohort_sha256: state.cohort_sha256.clone(),
        proposal_pointer_sha256: state
            .sanitized_proposal_pointer_sha256
            .clone()
            .context("built supply state lacks sanitized proposal identity")?,
        summary_sha256: sha256_file(&directory.join("summary.json"))?,
        manifests: summary
            .get("manifest_count")
            .and_then(Value::as_u64)
            .context("built wave summary lacks manifest count")? as usize,
    })
}

fn create_supply_state(
    args: &Args,
    inputs: &CheckpointInputs,
    waves_root: &Path,
    wave: &str,
) -> Result<PersistedSupplyState> {
    let cohort = waves_root
        .join("pending")
        .join(format!("{wave}.cohort.json"));
    let plan = run_builder(
        &args.wave_builder,
        vec![
            "plan".into(),
            "--catalog-pointer".into(),
            args.catalog_pointer().into_os_string(),
            "--checkpoint-pointer".into(),
            inputs.checkpoint_pointer.clone().into_os_string(),
            "--corpus".into(),
            args.corpus().into_os_string(),
            "--sealed-cohort".into(),
            args.sealed_cohort().into_os_string(),
            "--output-root".into(),
            waves_root.to_path_buf().into_os_string(),
            "--wave-id".into(),
            wave.into(),
            "--allow-pending-lookahead".into(),
        ],
    )?;
    require_output_identity(&plan, wave, &inputs.checkpoint_sha256)?;
    if plan.get("cohort").and_then(Value::as_str) != Some(cohort.to_string_lossy().as_ref()) {
        bail!("{wave}: planner returned an unexpected cohort path");
    }
    let cohort_sha256 = require_sha(&plan, "cohort_sha256")?;
    let catalog_sha256 = require_sha(&plan, "catalog_sha256")?;
    let proposal_reservoir_per_context = adaptive_proposal_reservoir(&args.root);
    let decoder_configuration_sha256 =
        proposal_decoder_configuration_sha256(args, proposal_reservoir_per_context)?;
    let recovery_accounting_only = wave == "strict-wave-000189";
    let proposal_job_id = if recovery_accounting_only {
        format!(
            "recovery-source-{}",
            &sha256_file(&inputs.proposals_pointer)?[..16]
        )
    } else {
        format!(
            "jope-selfplay-cohort-propose-{}-{}-{}-s{}",
            &inputs.checkpoint_sha256[..12],
            &cohort_sha256[..12],
            &decoder_configuration_sha256[..12],
            COHORT_PROPOSAL_SEED
        )
    };
    let state = PersistedSupplyState {
        schema: SUPPLY_STATE_V2_SCHEMA.into(),
        wave: wave.to_owned(),
        checkpoint_sha256: inputs.checkpoint_sha256.clone(),
        cohort,
        cohort_sha256,
        catalog_sha256,
        decoder_configuration_sha256,
        seed: COHORT_PROPOSAL_SEED,
        proposal_reservoir_per_context,
        stage: PersistedSupplyStage::FreezeCohort,
        revision: 0,
        recovery_accounting_only,
        proposal_attempt: 0,
        proposal_job_id: Some(proposal_job_id),
        cohort_proposal_pointer: None,
        cohort_proposal_pointer_sha256: None,
        sanitized_proposal_pointer: None,
        sanitized_proposal_pointer_sha256: None,
        runtime_pinned_attempts: Vec::new(),
    };
    let mut state = state;
    if !state.recovery_accounting_only {
        resolve_pinned_proposer_spec(args, inputs, &mut state)?;
    }
    persist_supply_state(&args.root, &state)?;
    Ok(state)
}

fn ensure_cohort_lookahead(
    args: &Args,
    inputs: &CheckpointInputs,
    waves_root: &Path,
    desired_pending: usize,
) -> Result<()> {
    loop {
        let pending = waves_root.join("pending");
        let unresolved = fs::read_dir(&pending)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let stem = path.file_stem()?.to_str()?;
                let wave = stem.strip_suffix(".cohort")?;
                (valid_wave_id(wave) && !waves_root.join(wave).is_dir()).then_some(())
            })
            .count();
        if unresolved >= desired_pending {
            return Ok(());
        }
        let wave = next_wave_id(waves_root)?;
        create_supply_state(args, inputs, waves_root, &wave)?;
    }
}

fn build_next_wave(args: &Args, inputs: &CheckpointInputs) -> Result<Option<BuiltWave>> {
    let waves_root = args.root.join("waves");
    let wave = wave_to_build(&waves_root)?;
    let cohort = waves_root
        .join("pending")
        .join(format!("{wave}.cohort.json"));
    let mut state = if let Some(state) =
        load_supply_state(&args.root, &wave, &inputs.checkpoint_sha256)?
    {
        let expected_decoder_configuration_sha256 =
            proposal_decoder_configuration_sha256(args, state.proposal_reservoir_per_context)?;
        if state.runtime_pinned_attempts.is_empty()
            && state.decoder_configuration_sha256 != expected_decoder_configuration_sha256
        {
            bail!(
                "{wave}: persisted proposer decoder configuration differs from the active immutable runtime"
            );
        }
        state
    } else {
        create_supply_state(args, inputs, &waves_root, &wave)?
    };
    if !state.cohort.is_file() || sha256_file(&state.cohort)? != state.cohort_sha256 {
        bail!("{wave}: persisted frozen cohort content drift");
    }

    if state.stage == PersistedSupplyStage::FreezeCohort {
        if state.recovery_accounting_only {
            state.advance(PersistedSupplyStage::SubmitCohortProposer)?;
            persist_supply_state(&args.root, &state)?;
        } else {
            let spec = resolve_pinned_proposer_spec(args, inputs, &mut state)?;
            persist_supply_state(&args.root, &state)?;
            let job_id = state
                .proposal_job_id
                .as_deref()
                .context("cohort proposer job id is absent")?;
            match proposer_job_state(args, job_id)? {
                None => {
                    if !scq_check_submit(args, &spec)? {
                        return Ok(None);
                    }
                }
                Some(existing) if existing == "completed" => {}
                Some(existing)
                    if matches!(
                        existing.as_str(),
                        "queued" | "starting" | "dispatching" | "running" | "held"
                    ) => {}
                Some(existing) => bail!("{job_id}: proposer is terminal in state {existing}"),
            }
            state.advance(PersistedSupplyStage::SubmitCohortProposer)?;
            persist_supply_state(&args.root, &state)?;
            ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
            return Ok(None);
        }
    }

    if state.stage == PersistedSupplyStage::SubmitCohortProposer {
        let (pointer, pointer_sha256) = if state.recovery_accounting_only {
            if wave != "strict-wave-000189" {
                bail!("whole-catalog proposal sources are recovery-only");
            }
            (
                inputs.proposals_pointer.clone(),
                sha256_file(&inputs.proposals_pointer)?,
            )
        } else {
            let mut job_id = state
                .proposal_job_id
                .clone()
                .context("cohort proposer job id is absent")?;
            match proposer_job_state(args, &job_id)?.as_deref() {
                Some("completed") => {
                    ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                    fetch_cohort_proposals(args, &state)?
                }
                Some("queued" | "starting" | "dispatching" | "running" | "held") => {
                    ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                    return Ok(None);
                }
                Some(_) => {
                    state.retry_proposer(&args.root)?;
                    persist_supply_state(&args.root, &state)?;
                    job_id = state
                        .proposal_job_id
                        .clone()
                        .context("retried proposer lacks a job identity")?;
                    let spec = resolve_pinned_proposer_spec(args, inputs, &mut state)?;
                    match proposer_job_state(args, &job_id)?.as_deref() {
                        None => {
                            if !scq_check_submit(args, &spec)? {
                                return Ok(None);
                            }
                        }
                        Some(
                            "completed" | "queued" | "starting" | "dispatching" | "running"
                            | "held",
                        ) => {}
                        Some(other) => {
                            bail!("{job_id}: retried proposer is terminal in state {other}")
                        }
                    }
                    ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                    return Ok(None);
                }
                None => {
                    let spec = resolve_pinned_proposer_spec(args, inputs, &mut state)?;
                    if !scq_check_submit(args, &spec)? {
                        return Ok(None);
                    }
                    ensure_cohort_lookahead(args, inputs, &waves_root, 3)?;
                    return Ok(None);
                }
            }
        };
        state.cohort_proposal_pointer = Some(pointer);
        state.cohort_proposal_pointer_sha256 = Some(pointer_sha256);
        state.advance(PersistedSupplyStage::FetchValidateProposals)?;
        persist_supply_state(&args.root, &state)?;
    }

    if state.stage == PersistedSupplyStage::FetchValidateProposals {
        assert_checkpoint_unchanged(args, &inputs.checkpoint_sha256)?;
        let source_pointer = state
            .cohort_proposal_pointer
            .clone()
            .context("validated state lacks proposal pointer")?;
        if sha256_file(&source_pointer)?
            != state
                .cohort_proposal_pointer_sha256
                .clone()
                .context("validated state lacks proposal pointer hash")?
        {
            bail!("{wave}: immutable cohort proposal pointer drift");
        }
        let sanitized = run_builder(
            &args.wave_builder,
            vec![
                "sanitize-proposals".into(),
                "--cohort".into(),
                cohort.clone().into_os_string(),
                "--proposals-pointer".into(),
                source_pointer.into_os_string(),
                "--output-root".into(),
                args.root.join("proposals/sanitized").into_os_string(),
            ],
        )?;
        require_output_identity(&sanitized, &wave, &inputs.checkpoint_sha256)?;
        state.sanitized_proposal_pointer_sha256 = Some(require_sha(&sanitized, "pointer_sha256")?);
        state.sanitized_proposal_pointer = Some(PathBuf::from(
            sanitized
                .get("pointer")
                .and_then(Value::as_str)
                .context("sanitizer output lacks proposal pointer")?,
        ));
        state.advance(PersistedSupplyStage::CacheFilterResample)?;
        persist_supply_state(&args.root, &state)?;
    }

    if state.stage == PersistedSupplyStage::CacheFilterResample {
        assert_checkpoint_unchanged(args, &inputs.checkpoint_sha256)?;
        let proposal_pointer = state
            .sanitized_proposal_pointer
            .clone()
            .context("cache-filtered state lacks proposal pointer")?;
        let built = run_builder(
            &args.wave_builder,
            vec![
                "build".into(),
                "--cohort".into(),
                cohort.into_os_string(),
                "--catalog-pointer".into(),
                args.catalog_pointer().into_os_string(),
                "--proposals-pointer".into(),
                proposal_pointer.into_os_string(),
                "--corpus".into(),
                args.corpus().into_os_string(),
                "--robust-seed-bank".into(),
                args.robust_seed_bank().into_os_string(),
                "--sealed-cohort".into(),
                args.sealed_cohort().into_os_string(),
                "--pair-registry".into(),
                args.pair_registry().into_os_string(),
                "--output-root".into(),
                waves_root.into_os_string(),
                "--max-candidates-per-manifest".into(),
                args.max_candidates_per_manifest.to_string().into(),
            ],
        )?;
        if built.get("wave").and_then(Value::as_str) != Some(wave.as_str()) {
            bail!("{wave}: Rust wave builder returned a mismatched wave");
        }
        let manifests = built
            .get("manifests")
            .and_then(Value::as_u64)
            .context("builder output lacks manifest count")? as usize;
        if manifests == 0 || manifests > manifests_per_wave(args.max_candidates_per_manifest) {
            bail!("{wave}: builder manifest count is outside the bounded micro-wave size");
        }
        state.advance(PersistedSupplyStage::BuildWave)?;
        persist_supply_state(&args.root, &state)?;
    }

    if state.stage == PersistedSupplyStage::BuildWave {
        return Ok(Some(built_wave_from_disk(args, &state)?));
    }
    Ok(None)
}

fn require_output_identity(output: &Value, wave: &str, checkpoint: &str) -> Result<()> {
    if output.get("wave").and_then(Value::as_str) != Some(wave)
        || output.get("checkpoint_sha256").and_then(Value::as_str) != Some(checkpoint)
    {
        bail!("{wave}: Rust wave stage returned stale or mismatched provenance");
    }
    Ok(())
}

fn require_sha(output: &Value, field: &str) -> Result<String> {
    let digest = output
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("Rust wave stage lacks {field}"))?;
    validate_sha256(digest)?;
    Ok(digest.to_owned())
}

fn ensure_native_wave_builder(binary: &Path) -> Result<()> {
    let mut magic = [0_u8; 4];
    File::open(binary)
        .with_context(|| format!("open Rust wave builder {}", binary.display()))?
        .read_exact(&mut magic)?;
    if magic != [0x7f, b'E', b'L', b'F'] {
        bail!("wave builder must be a native ELF binary; scripts and Python are forbidden");
    }
    Ok(())
}

fn run_builder(binary: &Path, arguments: Vec<OsString>) -> Result<Value> {
    let output = Command::new(binary)
        .args(arguments)
        .env_remove("PYTHONPATH")
        .output()
        .with_context(|| format!("execute Rust wave builder {}", binary.display()))?;
    if !output.status.success() {
        bail!(
            "Rust wave builder failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    serde_json::from_slice(&output.stdout).context("parse Rust wave builder JSON output")
}

fn append_event(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn pinned_test_spec(root: &Path, job_identity: &str) -> PathBuf {
        let bundle_sha256 = "e".repeat(64);
        let bundle = root.join("runtime-bundles/bundles").join(&bundle_sha256);
        let pythonpath = bundle.join("pytorch");
        fs::create_dir_all(pythonpath.join("proposals")).unwrap();
        fs::create_dir_all(pythonpath.join("shared")).unwrap();
        fs::write(
            bundle.join("bundle.json"),
            serde_json::to_vec(&json!({"bundle_sha256": bundle_sha256})).unwrap(),
        )
        .unwrap();
        let generator = pythonpath.join("proposals/test_exact_dl_generate.py");
        let common = pythonpath.join("shared/test_exact_common.py");
        fs::write(&generator, b"generator").unwrap();
        fs::write(&common, b"common").unwrap();
        let input_root = root.join("inputs");
        fs::create_dir_all(&input_root).unwrap();
        let pointer = input_root.join("pointer.json");
        let catalog = input_root.join("catalog.json");
        let data = input_root.join("contexts.f32.bin");
        let checkpoint = input_root.join("checkpoint.pt");
        let cohort = input_root.join("cohort.json");
        for (path, body) in [
            (&pointer, b"pointer".as_slice()),
            (&catalog, b"catalog".as_slice()),
            (&data, b"data".as_slice()),
            (&checkpoint, b"checkpoint".as_slice()),
            (&cohort, b"cohort".as_slice()),
        ] {
            fs::write(path, body).unwrap();
        }
        let mut lines = vec![
            "[job]".to_owned(),
            format!("id = {}", toml_quote(job_identity).unwrap()),
            String::new(),
            "[env]".to_owned(),
            format!(
                "PYTHONPATH = {}",
                toml_quote(&pythonpath.to_string_lossy()).unwrap()
            ),
            String::new(),
        ];
        for (name, path) in [
            ("generator", generator.as_path()),
            ("common", common.as_path()),
            ("pointer", pointer.as_path()),
            ("catalog", catalog.as_path()),
            ("data", data.as_path()),
            ("checkpoint", checkpoint.as_path()),
            ("cohort", cohort.as_path()),
        ] {
            append_spec_input(&mut lines, name, path, name).unwrap();
        }
        let directory = root.join("specs/cohort-propose");
        fs::create_dir_all(&directory).unwrap();
        let spec = directory.join(format!("{job_identity}.toml"));
        fs::write(&spec, lines.join("\n")).unwrap();
        spec
    }

    fn audit_args(root: &Path) -> Args {
        Args {
            root: root.to_owned(),
            build: false,
            once: true,
            interval_seconds: 5,
            queue_floor: 1_000,
            queued_target: 2_048,
            queued_hard_cap: 4_096,
            ready_reserve_min: 1_024,
            max_overlap_waves: 8,
            max_builds_per_cycle: 1,
            max_candidates_per_manifest: 100,
            queue_status_max_age_seconds: 30,
            scq: PathBuf::from("/unused-scq"),
            wave_builder: PathBuf::from("/unused"),
            proposal_generator: PathBuf::from("/unused-generator"),
            proposal_legacy_generator: PathBuf::from("/unused-legacy-generator"),
            proposal_common: PathBuf::from("/unused-common"),
            proposal_pythonpath: PathBuf::from("/unused-pythonpath"),
            catalog_pointer: None,
            checkpoint_pointer: None,
            proposals_pointer: None,
            corpus: None,
            sealed_cohort: None,
            robust_seed_bank: None,
            pair_registry: None,
            queue_status: None,
        }
    }

    #[test]
    fn decision_builds_to_target_with_bounded_overlap() {
        let decision = supply_decision(900, 0, 0, 1_000, 2_048, 4_096, 1_024, 8, 1, 100);
        assert!(decision.below_floor);
        assert!(!decision.overloaded);
        assert_eq!(decision.estimated_manifests_per_wave, 384);
        assert_eq!(decision.builds_planned, 1);

        let bounded = supply_decision(0, 0, 7, 1_000, 2_048, 4_096, 1_024, 8, 1, 100);
        assert_eq!(bounded.builds_planned, 1);

        let full = supply_decision(1_500, 1_024, 2, 1_000, 2_048, 4_096, 1_024, 8, 1, 100);
        assert!(!full.below_floor);
        assert!(full.overloaded);
        assert_eq!(full.builds_planned, 0);

        let reserve = supply_decision(2_048, 600, 2, 1_000, 2_048, 4_096, 1_024, 8, 1, 100);
        assert!(reserve.overloaded);
        assert_eq!(reserve.builds_planned, 1);

        let capped = supply_decision(4_096, 0, 0, 1_000, 2_048, 4_096, 1_024, 8, 1, 100);
        assert_eq!(capped.builds_planned, 0);
    }

    #[test]
    fn configuration_rejects_a_wave_that_can_overshoot_the_hard_cap() {
        let directory = tempdir().unwrap();
        let mut args = audit_args(directory.path());
        assert!(validate_args(&args).is_ok());
        args.max_candidates_per_manifest = 24;
        assert!(validate_args(&args).is_err());
        args.max_candidates_per_manifest = 25;
        args.queued_hard_cap = 3_000;
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn immutable_proposer_spec_survives_equivalent_bundle_activation() {
        let directory = tempdir().unwrap();
        let old_runtime = directory.path().join("old/pytorch");
        let new_runtime = directory.path().join("new/pytorch");
        for runtime in [&old_runtime, &new_runtime] {
            fs::create_dir_all(runtime.join("proposals")).unwrap();
            fs::create_dir_all(runtime.join("shared")).unwrap();
            fs::create_dir_all(runtime.join("jope/policy")).unwrap();
            fs::write(
                runtime.join("proposals/test_exact_context_proposals.py"),
                b"context-generator",
            )
            .unwrap();
            fs::write(
                runtime.join("proposals/test_exact_dl_generate.py"),
                b"legacy-generator",
            )
            .unwrap();
            fs::write(runtime.join("shared/test_exact_common.py"), b"common").unwrap();
            fs::write(runtime.join("jope/__init__.py"), b"").unwrap();
            fs::write(runtime.join("jope/policy/model_v3.py"), b"model").unwrap();
            fs::write(runtime.join("jope/policy/decode.py"), b"decode").unwrap();
        }
        let mut args = audit_args(directory.path());
        args.proposal_generator = new_runtime.join("proposals/test_exact_context_proposals.py");
        args.proposal_legacy_generator = new_runtime.join("proposals/test_exact_dl_generate.py");
        args.proposal_common = new_runtime.join("shared/test_exact_common.py");
        let existing = format!(
            "[env]\nPYTHONPATH = \"{}\"\nOMP_NUM_THREADS = \"4\"\n",
            old_runtime.display()
        );
        let expected = format!(
            "[env]\nPYTHONPATH = \"{}\"\nOMP_NUM_THREADS = \"4\"\n",
            new_runtime.display()
        );
        assert!(validate_proposer_runtime(&existing, &expected, &args).is_ok());

        fs::write(
            old_runtime.join("proposals/test_exact_context_proposals.py"),
            b"drift",
        )
        .unwrap();
        assert!(validate_proposer_runtime(&existing, &expected, &args).is_err());
    }

    #[test]
    fn v2_cohort_contexts_resolve_only_from_the_frozen_catalog_with_receipts() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let data_path = root.join("contexts.bin");
        let context_a = [1_u8, 2, 3, 4];
        let context_b = [5_u8, 6, 7, 8];
        let context_c = [9_u8, 10, 11, 12];
        fs::write(
            &data_path,
            [
                context_a.as_slice(),
                context_b.as_slice(),
                context_c.as_slice(),
            ]
            .concat(),
        )
        .unwrap();
        let data_sha256 = sha256_file(&data_path).unwrap();
        let catalog_path = root.join("catalog.json");
        atomic_json(
            &catalog_path,
            &json!({
                "schema": "JopeExpandedCatalog.v2",
                "data_sha256": data_sha256,
                "record_stride_bytes": 4,
                "entries": [
                    {
                        "task": "binary",
                        "dataset_hash": "valid",
                        "catalog_index": 0,
                        "context_sha256": sha256_bytes(&context_a),
                    },
                    {
                        "task": "regression",
                        "dataset_hash": "ambiguous",
                        "catalog_index": 1,
                        "context_sha256": sha256_bytes(&context_b),
                    },
                    {
                        "task": "regression",
                        "dataset_hash": "ambiguous",
                        "catalog_index": 2,
                        "context_sha256": sha256_bytes(&context_c),
                    }
                ]
            }),
        )
        .unwrap();
        let catalog_sha256 = sha256_file(&catalog_path).unwrap();
        let pointer_path = root.join("frozen-pointer.json");
        let pointer = json!({
            "catalog": catalog_path,
            "catalog_sha256": catalog_sha256,
            "data": data_path,
            "data_sha256": data_sha256,
        });
        atomic_json(&pointer_path, &pointer).unwrap();
        let cohort_path = root.join("cohort.json");
        atomic_json(
            &cohort_path,
            &json!({
                "schema": "JopeMicroCohort.v2",
                "entries": [
                    {"task": "binary", "dataset_hash": "valid", "micro_lane": "a"},
                    {"task": "regression", "dataset_hash": "ambiguous", "micro_lane": "b"},
                    {"task": "regression", "dataset_hash": "absent", "micro_lane": "c"}
                ]
            }),
        )
        .unwrap();
        let cohort_sha256 = sha256_file(&cohort_path).unwrap();
        let state = PersistedSupplyState {
            schema: SUPPLY_STATE_V2_SCHEMA.to_owned(),
            wave: "strict-wave-000223".to_owned(),
            checkpoint_sha256: "c".repeat(64),
            cohort: cohort_path,
            cohort_sha256: cohort_sha256.clone(),
            catalog_sha256: catalog_sha256.clone(),
            decoder_configuration_sha256: "e".repeat(64),
            seed: COHORT_PROPOSAL_SEED,
            proposal_reservoir_per_context: 64,
            stage: PersistedSupplyStage::FreezeCohort,
            revision: 0,
            recovery_accounting_only: false,
            proposal_attempt: 0,
            proposal_job_id: Some("proposal".to_owned()),
            cohort_proposal_pointer: None,
            cohort_proposal_pointer_sha256: None,
            sanitized_proposal_pointer: None,
            sanitized_proposal_pointer_sha256: None,
            runtime_pinned_attempts: Vec::new(),
        };
        let args = audit_args(root);
        let request_path =
            context_proposal_request(&args, &state, &pointer_path, &pointer, &"d".repeat(64))
                .unwrap();
        let request: Value = serde_json::from_slice(&fs::read(&request_path).unwrap()).unwrap();
        assert_eq!(
            request.get("schema").and_then(Value::as_str),
            Some("ContextProposalRequest.v4")
        );
        assert_eq!(
            request
                .get("requested_contexts")
                .and_then(Value::as_array)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            request
                .pointer("/requested_contexts/0/context_sha256")
                .and_then(Value::as_str),
            Some(sha256_bytes(&context_a).as_str())
        );
        let receipt_path = root.join("proposals/cohort-inputs").join(format!(
            "{cohort_sha256}.{catalog_sha256}.{}.context-migration.v1.json",
            sha256_file(&request_path).unwrap()
        ));
        let receipt: Value = serde_json::from_slice(&fs::read(receipt_path).unwrap()).unwrap();
        assert_eq!(
            receipt.get("resolved_entries").and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            receipt.get("excluded_entries").and_then(Value::as_u64),
            Some(2)
        );

        fs::write(&data_path, b"catalog drift").unwrap();
        assert!(
            context_proposal_request(&args, &state, &pointer_path, &pointer, &"d".repeat(64),)
                .is_err()
        );
    }

    #[test]
    fn terminal_proposer_retry_preserves_cohort_and_advances_attempt() {
        let directory = tempdir().unwrap();
        let original_spec = pinned_test_spec(directory.path(), "cohort-proposer");
        let pinned = runtime_pinned_intent_from_spec(&original_spec, 0).unwrap();
        let mut state: PersistedSupplyState = serde_json::from_value(json!({
            "schema": "CohortSupplyState.v2",
            "wave": "strict-wave-000190",
            "checkpoint_sha256": "c".repeat(64),
            "cohort": "/tmp/cohort.json",
            "cohort_sha256": "a".repeat(64),
            "catalog_sha256": "b".repeat(64),
            "decoder_configuration_sha256": "d".repeat(64),
            "seed": 20260726,
            "stage": "submit_cohort_proposer",
            "revision": 1,
            "recovery_accounting_only": false,
            "proposal_job_id": "cohort-proposer",
            "runtime_pinned_attempts": [pinned]
        }))
        .unwrap();
        state.retry_proposer(directory.path()).unwrap();
        assert_eq!(state.proposal_attempt, 1);
        assert_eq!(
            state.proposal_job_id.as_deref(),
            Some("cohort-proposer-retry1")
        );
        assert_eq!(state.revision, 2);
        assert_eq!(state.cohort_sha256, "a".repeat(64));
        assert_eq!(state.runtime_pinned_attempts.len(), 2);
        assert_eq!(
            state.runtime_pinned_attempts[0].runtime_bundle_sha256,
            state.runtime_pinned_attempts[1].runtime_bundle_sha256
        );
        assert_eq!(
            state.runtime_pinned_attempts[0].proposer_inputs,
            state.runtime_pinned_attempts[1].proposer_inputs
        );
    }

    #[test]
    fn legacy_state_migrates_by_adopting_its_exact_existing_spec() {
        let directory = tempdir().unwrap();
        let spec = pinned_test_spec(directory.path(), "cohort-proposer");
        let immutable_spec_sha256 = sha256_file(&spec).unwrap();
        let wave = "strict-wave-000200";
        let checkpoint = "c".repeat(64);
        let legacy: PersistedSupplyState = serde_json::from_value(json!({
            "schema": "CohortSupplyState.v1",
            "wave": wave,
            "checkpoint_sha256": checkpoint,
            "cohort": "/tmp/cohort.json",
            "cohort_sha256": "a".repeat(64),
            "catalog_sha256": "b".repeat(64),
            "decoder_configuration_sha256": "d".repeat(64),
            "seed": 20260726,
            "stage": "submit_cohort_proposer",
            "revision": 1,
            "recovery_accounting_only": false,
            "proposal_job_id": "cohort-proposer"
        }))
        .unwrap();
        let legacy_path = supply_state_v1_path(directory.path(), wave);
        fs::create_dir_all(legacy_path.parent().unwrap()).unwrap();
        fs::write(&legacy_path, serde_json::to_vec(&legacy).unwrap()).unwrap();

        let migrated = load_supply_state(directory.path(), wave, &checkpoint)
            .unwrap()
            .unwrap();
        assert_eq!(migrated.schema, SUPPLY_STATE_V2_SCHEMA);
        assert_eq!(migrated.runtime_pinned_attempts.len(), 1);
        assert_eq!(
            migrated.runtime_pinned_attempts[0].immutable_spec_sha256,
            immutable_spec_sha256
        );
        assert!(legacy_path.is_file());
        assert!(supply_state_path(directory.path(), wave).is_file());
    }

    #[test]
    fn proposer_capacity_pressure_is_a_normal_backoff() {
        assert!(temporarily_lacks_launch_capacity(
            "error[dependency-unsatisfied]: dependencies cannot be verified on any eligible \
             node; eligible_node launch-ready-capacity: no eligible launch-ready node currently"
        ));
        assert!(temporarily_lacks_launch_capacity(
            "error[dependency-unsatisfied]: candidate node: smoke proposal GPU capacity: \
             smoke test exited with exit 6"
        ));
        assert!(!temporarily_lacks_launch_capacity(
            "error[dependency-unsatisfied]: unrelated smoke test exited with exit 6"
        ));
        assert!(!temporarily_lacks_launch_capacity(
            "no eligible launch-ready node currently"
        ));
    }

    #[test]
    fn frozen_catalog_snapshot_survives_independent_live_advancement() {
        let directory = tempdir().unwrap();
        let live = directory.path().join("current.json");
        let snapshot = directory.path().join("frozen.json");
        let first = json!({
            "catalog_sha256": "a".repeat(64),
            "catalog": "/catalog-a.json",
            "data": "/catalog-a.f32",
        });
        std::fs::write(&live, serde_json::to_vec(&first).unwrap()).unwrap();
        assert_eq!(
            frozen_catalog_pointer(&live, &snapshot, &"a".repeat(64)).unwrap(),
            first
        );

        let advanced = json!({
            "catalog_sha256": "b".repeat(64),
            "catalog": "/catalog-b.json",
            "data": "/catalog-b.f32",
        });
        std::fs::write(&live, serde_json::to_vec(&advanced).unwrap()).unwrap();
        assert_eq!(
            frozen_catalog_pointer(&live, &snapshot, &"a".repeat(64)).unwrap(),
            first
        );
    }

    #[test]
    fn keeper_handshake_allows_only_one_unacknowledged_wave() {
        let directory = tempdir().unwrap();
        let ready = vec![
            ReadyWave {
                wave: "strict-wave-000001".to_owned(),
                summary_sha256: "a".repeat(64),
                manifests: 384,
                unclaimed_manifests: 300,
            },
            ReadyWave {
                wave: "strict-wave-000002".to_owned(),
                summary_sha256: "b".repeat(64),
                manifests: 384,
                unclaimed_manifests: 384,
            },
        ];
        let status = QueueStatus {
            schema: "JopeSelfplayQueueKeeper.v3".to_owned(),
            generated_at_ms: now_ms(),
            admission_allowed: true,
            degraded: false,
            error: None,
            fail_closed_reason: None,
            active_checkpoint_sha256: Some("c".repeat(64)),
            checkpoint_epoch: Some(checkpoint_epoch(&"c".repeat(64)).unwrap()),
            queued_target: 2_048,
            queued_hard_cap: 4_096,
            queued_starting: 0,
            dispatching: 0,
            admission_waves: vec![KeeperAdmissionWave {
                wave: ready[0].wave.clone(),
                summary_sha256: ready[0].summary_sha256.clone(),
                unclaimed_manifests: ready[0].unclaimed_manifests,
            }],
            unclaimed_current_checkpoint_manifests: 300,
        };
        assert!(validate_keeper_admission(
            &status,
            &ready,
            directory.path(),
            Some("strict-wave-000002"),
            8
        )
        .is_ok());
        assert!(validate_keeper_admission(&status, &ready, directory.path(), None, 8).is_err());
    }

    #[test]
    fn keeper_handshake_accepts_monotonic_claim_progress_after_its_snapshot() {
        let directory = tempdir().unwrap();
        let ready = vec![ReadyWave {
            wave: "strict-wave-000001".to_owned(),
            summary_sha256: "a".repeat(64),
            manifests: 384,
            unclaimed_manifests: 299,
        }];
        let status = QueueStatus {
            schema: "JopeSelfplayQueueKeeper.v3".to_owned(),
            generated_at_ms: now_ms(),
            admission_allowed: true,
            degraded: false,
            error: None,
            fail_closed_reason: None,
            active_checkpoint_sha256: Some("c".repeat(64)),
            checkpoint_epoch: Some(checkpoint_epoch(&"c".repeat(64)).unwrap()),
            queued_target: 2_048,
            queued_hard_cap: 4_096,
            queued_starting: 0,
            dispatching: 0,
            admission_waves: vec![KeeperAdmissionWave {
                wave: ready[0].wave.clone(),
                summary_sha256: ready[0].summary_sha256.clone(),
                unclaimed_manifests: 300,
            }],
            unclaimed_current_checkpoint_manifests: 300,
        };
        assert!(validate_keeper_admission(&status, &ready, directory.path(), None, 8).is_ok());

        let impossible = vec![ReadyWave {
            unclaimed_manifests: 301,
            ..ready[0].clone()
        }];
        assert!(
            validate_keeper_admission(&status, &impossible, directory.path(), None, 8).is_err()
        );
    }

    #[test]
    fn keeper_acknowledgment_must_be_newer_than_the_build() {
        let directory = tempdir().unwrap();
        let checkpoint = "c".repeat(64);
        let wave = "strict-wave-000002";
        let pending = PendingAck {
            wave: wave.to_owned(),
            since_ms: 10_000,
        };
        let mut status = QueueStatus {
            schema: "JopeSelfplayQueueKeeper.v3".to_owned(),
            generated_at_ms: pending.since_ms,
            admission_allowed: true,
            degraded: false,
            error: None,
            fail_closed_reason: None,
            active_checkpoint_sha256: Some(checkpoint.clone()),
            checkpoint_epoch: Some(checkpoint_epoch(&checkpoint).unwrap()),
            queued_target: 2_048,
            queued_hard_cap: 4_096,
            queued_starting: 0,
            dispatching: 0,
            admission_waves: vec![KeeperAdmissionWave {
                wave: wave.to_owned(),
                summary_sha256: "a".repeat(64),
                unclaimed_manifests: 384,
            }],
            unclaimed_current_checkpoint_manifests: 384,
        };
        let inputs = CheckpointInputs {
            checkpoint_sha256: checkpoint.clone(),
            checkpoint_pointer: PathBuf::new(),
            checkpoint_artifact: PathBuf::new(),
            proposals_pointer: PathBuf::new(),
        };
        assert!(!keeper_acknowledged(
            &audit_args(directory.path()),
            &status,
            &inputs,
            &checkpoint_epoch(&checkpoint).unwrap(),
            &BTreeSet::new(),
            &pending,
        )
        .unwrap());
        status.generated_at_ms += 1;
        assert!(keeper_acknowledged(
            &audit_args(directory.path()),
            &status,
            &inputs,
            &checkpoint_epoch(&checkpoint).unwrap(),
            &BTreeSet::new(),
            &pending,
        )
        .unwrap());
    }

    #[test]
    fn rejected_manifests_are_not_counted_as_unclaimed_supply() {
        let directory = tempdir().unwrap();
        let backlog = directory.path().join("backlog.sqlite");
        let rejected_hash = "d".repeat(64);
        let connection = Connection::open(&backlog).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE claims(manifest_sha256 TEXT);
                 CREATE TABLE rejected_manifests(
                   manifest TEXT PRIMARY KEY,
                   manifest_sha256 TEXT,
                   reason TEXT NOT NULL,
                   updated_at_ms INTEGER NOT NULL
                 ) WITHOUT ROWID;",
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO rejected_manifests VALUES('/absent',?1,'invalid',1)",
                [&rejected_hash],
            )
            .unwrap();
        drop(connection);
        let unavailable = claimed_manifest_hashes(&backlog).unwrap();
        assert!(unavailable.contains(&rejected_hash));

        let missing_table = directory.path().join("old-backlog.sqlite");
        Connection::open(&missing_table)
            .unwrap()
            .execute_batch("CREATE TABLE claims(manifest_sha256 TEXT);")
            .unwrap();
        assert!(claimed_manifest_hashes(&missing_table).is_err());
    }

    #[test]
    fn next_wave_identity_includes_unresolved_pending_cohort() {
        let directory = tempdir().unwrap();
        let waves = directory.path();
        fs::create_dir(waves.join("strict-wave-000004")).unwrap();
        fs::create_dir(waves.join("strict-wave-invalid")).unwrap();
        fs::create_dir(waves.join("pending")).unwrap();
        File::create(waves.join("pending").join("strict-wave-000006.cohort.json")).unwrap();
        assert_eq!(next_wave_id(waves).unwrap(), "strict-wave-000007");
        assert_eq!(wave_to_build(waves).unwrap(), "strict-wave-000006");
    }

    #[test]
    fn operator_quarantine_skips_only_hash_bound_pending_wave() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let waves = root.join("waves");
        fs::create_dir_all(waves.join("pending")).unwrap();
        fs::create_dir_all(waves.join("quarantine")).unwrap();
        fs::create_dir_all(root.join("ops/cohort-supply-state-v2")).unwrap();
        let wave = "strict-wave-000006";
        let cohort = waves.join("pending").join(format!("{wave}.cohort.json"));
        let state = supply_state_path(root, wave);
        fs::write(&cohort, b"frozen-cohort").unwrap();
        fs::write(&state, b"frozen-state").unwrap();
        atomic_json(
            &waves
                .join("quarantine")
                .join(format!("{wave}.receipt.json")),
            &json!({
                "schema": "CohortSupplyQuarantineReceipt.v1",
                "wave": wave,
                "cohort_sha256": sha256_file(&cohort).unwrap(),
                "supply_state_sha256": sha256_file(&state).unwrap(),
                "operator_authorized": true,
                "reason": "proposal validity below immutable gate",
            }),
        )
        .unwrap();
        assert_eq!(wave_to_build(&waves).unwrap(), "strict-wave-000007");
    }

    #[test]
    fn stale_checkpoint_wave_is_not_launch_ready() {
        let directory = tempdir().unwrap();
        let wave = "strict-wave-000001";
        let wave_directory = directory.path().join(wave);
        fs::create_dir_all(wave_directory.join("manifests")).unwrap();
        fs::write(wave_directory.join("manifests/000.json"), b"{}").unwrap();
        let current = "a".repeat(64);
        let stale = "b".repeat(64);
        atomic_json(
            &wave_directory.join("summary.json"),
            &json!({
                "schema": WAVE_SCHEMA,
                "immutable": true,
                "wave_kind": MICRO_WAVE_KIND,
                "wave": wave,
                "checkpoint_sha256": stale,
                "checkpoint_epoch": checkpoint_epoch(&"b".repeat(64)).unwrap(),
                "manifest_count": 1,
                "manifests": [{
                    "relative_path": "manifests/000.json",
                    "sha256": "c".repeat(64)
                }]
            }),
        )
        .unwrap();
        let ready = validate_ready_wave(
            wave,
            &wave_directory,
            &current,
            &checkpoint_epoch(&current).unwrap(),
            &BTreeSet::new(),
        )
        .unwrap();
        assert!(ready.is_none());
    }

    #[test]
    fn legacy_summary_is_ignored_before_current_micro_wave_validation() {
        let directory = tempdir().unwrap();
        let wave = "strict-wave-000010";
        let checkpoint = "a".repeat(64);
        let wave_directory = directory.path().join(wave);
        fs::create_dir_all(&wave_directory).unwrap();
        atomic_json(
            &wave_directory.join("summary.json"),
            &json!({
                "schema": "JopeTestOnlyStrictWave.v1",
                "wave": wave,
                "checkpoint_sha256": checkpoint
            }),
        )
        .unwrap();
        assert!(validate_ready_wave(
            wave,
            &wave_directory,
            &"a".repeat(64),
            &checkpoint_epoch(&"a".repeat(64)).unwrap(),
            &BTreeSet::new(),
        )
        .unwrap()
        .is_none());

        let mut current_claim: Value =
            serde_json::from_slice(&fs::read(wave_directory.join("summary.json")).unwrap())
                .unwrap();
        current_claim["schema"] = json!(WAVE_SCHEMA);
        atomic_json(&wave_directory.join("summary.json"), &current_claim).unwrap();
        assert!(validate_ready_wave(
            wave,
            &wave_directory,
            &"a".repeat(64),
            &checkpoint_epoch(&"a".repeat(64)).unwrap(),
            &BTreeSet::new(),
        )
        .is_err());
    }

    #[test]
    fn malformed_manifest_path_fails_closed() {
        let directory = tempdir().unwrap();
        let wave = "strict-wave-000001";
        let checkpoint = "a".repeat(64);
        let wave_directory = directory.path().join(wave);
        fs::create_dir_all(&wave_directory).unwrap();
        atomic_json(
            &wave_directory.join("summary.json"),
            &json!({
                "schema": WAVE_SCHEMA,
                "immutable": true,
                "wave_kind": MICRO_WAVE_KIND,
                "wave": wave,
                "checkpoint_sha256": checkpoint,
                "checkpoint_epoch": checkpoint_epoch(&"a".repeat(64)).unwrap(),
                "manifest_count": 1,
                "manifests": [{
                    "relative_path": "../escape.json",
                    "sha256": "c".repeat(64)
                }]
            }),
        )
        .unwrap();
        assert!(validate_ready_wave(
            wave,
            &wave_directory,
            &"a".repeat(64),
            &checkpoint_epoch(&"a".repeat(64)).unwrap(),
            &BTreeSet::new(),
        )
        .is_err());
    }

    #[test]
    fn audit_cycle_counts_only_current_checkpoint_immutable_supply() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("ops")).unwrap();
        fs::create_dir_all(root.join("proposals")).unwrap();
        let checkpoint = "a".repeat(64);
        atomic_json(
            &root.join("proposals/current.json"),
            &json!({"checkpoint_sha256": checkpoint}),
        )
        .unwrap();
        let wave = "strict-wave-000001";
        let wave_directory = root.join("waves").join(wave);
        fs::create_dir_all(wave_directory.join("manifests")).unwrap();
        fs::write(wave_directory.join("manifests/000.json"), b"{}").unwrap();
        atomic_json(
            &wave_directory.join("summary.json"),
            &json!({
                "schema": WAVE_SCHEMA,
                "immutable": true,
                "wave_kind": MICRO_WAVE_KIND,
                "wave": wave,
                "checkpoint_sha256": "a".repeat(64),
                "checkpoint_epoch": checkpoint_epoch(&"a".repeat(64)).unwrap(),
                "manifest_count": 1,
                "manifests": [{
                    "relative_path": "manifests/000.json",
                    "sha256": "c".repeat(64)
                }]
            }),
        )
        .unwrap();
        let summary_sha256 = sha256_file(&wave_directory.join("summary.json")).unwrap();
        atomic_json(
            &root.join("ops/queue-keeper-v3.json"),
            &json!({
                "schema": "JopeSelfplayQueueKeeper.v3",
                "generated_at_ms": now_ms(),
                "admission_allowed": false,
                "active_checkpoint_sha256": checkpoint,
                "checkpoint_epoch": checkpoint_epoch(&"a".repeat(64)).unwrap(),
                "queued_target": 2048,
                "queued_hard_cap": 4096,
                "queued_starting": 900,
                "dispatching": 50,
                "admission_waves": [{
                    "wave": wave,
                    "summary_sha256": summary_sha256,
                    "unclaimed_manifests": 1
                }],
                "unclaimed_current_checkpoint_manifests": 1
            }),
        )
        .unwrap();

        let report = cycle(&audit_args(root)).unwrap();
        assert_eq!(report.queue_depth, 950);
        assert!(report.below_floor);
        assert_eq!(report.ready_manifests, 1);
        assert_eq!(report.projected_jobs_before_build, 951);
        assert_eq!(report.builds_planned, 0);
        assert!(!report.admission_allowed);
        assert_eq!(
            report.hold_reason.as_deref(),
            Some("queue admission is held")
        );

        atomic_json(
            &root.join("proposals/current.json"),
            &json!({"checkpoint_sha256": "b".repeat(64)}),
        )
        .unwrap();
        assert!(
            cycle(&audit_args(root)).is_ok(),
            "a queue hold must not cascade into unrelated pointer validation"
        );
    }

    #[test]
    fn in_schema_queue_error_is_a_healthy_supply_hold() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("ops")).unwrap();
        atomic_json(
            &root.join("ops/queue-keeper-v3.json"),
            &json!({
                "schema": "JopeSelfplayQueueKeeper.v3",
                "generated_at_ms": now_ms(),
                "admission_allowed": false,
                "degraded": true,
                "error": "native attempt reservation conflict",
                "fail_closed_reason": "queue cycle failed closed",
                "active_checkpoint_sha256": Value::Null,
                "checkpoint_epoch": Value::Null,
                "queued_target": 0,
                "queued_hard_cap": 0,
                "queued_starting": 0,
                "dispatching": 0,
                "admission_waves": [],
                "unclaimed_current_checkpoint_manifests": 0
            }),
        )
        .unwrap();
        let report = cycle(&audit_args(root)).unwrap();
        assert!(!report.admission_allowed);
        assert_eq!(report.builds_planned, 0);
        assert_eq!(
            report.hold_reason.as_deref(),
            Some("native attempt reservation conflict")
        );
    }
}
