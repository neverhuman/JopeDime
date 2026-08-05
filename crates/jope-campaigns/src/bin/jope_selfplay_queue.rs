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
    active_checkpoint, atomic_json, now_ms, protocol_for_manifest, read_json, scq_json,
    sha256_bytes, sha256_file, DEFAULT_EVALUATOR, DEFAULT_ROOT, DEFAULT_RUNNER, DEFAULT_SCQ,
};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const PROJECT: &str = "jope-selfplay-eval";
const MANIFEST_SCHEMA: &str = "JopeFlywheelCorpusEvalShard.v1";
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
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    /// SCQ client binary.
    #[arg(long, default_value = DEFAULT_SCQ)]
    scq: PathBuf,
    /// Precompiled Rust corpus evaluator.
    #[arg(long, default_value = DEFAULT_EVALUATOR)]
    evaluator: PathBuf,
    /// Portable evaluator runner (must not invoke Python).
    #[arg(long, default_value = DEFAULT_RUNNER)]
    runner: PathBuf,
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
    unclaimed_manifests: usize,
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
    unclaimed_current_checkpoint_manifests: usize,
    queued_target: i64,
    queued_hard_cap: i64,
    p95_60m_jobs_per_minute: f64,
    queued_starting: usize,
    running: usize,
    dispatching: usize,
    held: usize,
    released_held: usize,
    queue_maximum_age_seconds: f64,
    stale_or_unproven_active: usize,
    launch_ready: usize,
    launch_ready_deferred_by_horizon: usize,
    manifests_unclaimed: usize,
    manifests_validated: usize,
    manifest_validation_failures: usize,
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

impl Keeper {
    fn open(args: Args) -> Result<Self> {
        fs::create_dir_all(args.root.join("ops"))?;
        let backlog = Connection::open(args.root.join("ops/backlog.sqlite"))?;
        backlog.busy_timeout(Duration::from_secs(60))?;
        initialize_backlog(&backlog)?;
        let terminal_seen = load_terminal_seen(&backlog)?;
        let pairs = Connection::open(args.root.join("ops/pair-registry.sqlite"))?;
        pairs.busy_timeout(Duration::from_secs(60))?;
        let pair_schema = inspect_pair_schema(&pairs)?;
        Ok(Self {
            args,
            backlog,
            pairs,
            pair_schema,
            manifest_cache: HashMap::new(),
            reservation_cache: HashMap::new(),
            terminal_seen,
        })
    }

    fn cycle(&mut self) -> Result<CycleReport> {
        self.manifest_cache.clear();
        self.reservation_cache.clear();
        let scheduler_value = scq_json(&self.args.scq, &["ls"])?;
        let jobs = parse_jobs(&scheduler_value)?;
        let nodes_value = scq_json(&self.args.scq, &["nodes"])?;
        let nodes = parse_nodes(&nodes_value)?;
        let effective_cluster_slots = nodes.iter().map(|row| row.slot_target).sum();
        let burn = record_burn_rate(&mut self.backlog, &jobs, &mut self.terminal_seen)?;
        let targets = dynamic_queue_targets(&burn, effective_cluster_slots)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let gpu_pause = self.args.root.join("ops/gpu-lane.pause.json");
        if gpu_pause.is_file() {
            return Ok(empty_report(
                self.args.submit,
                targets,
                nodes,
                "GPU lane admission transaction paused evaluator queue submission",
            ));
        }
        let checkpoint = active_checkpoint(&self.args.root)?;

        let Some((checkpoint_sha256, checkpoint_epoch)) = checkpoint else {
            return Ok(empty_report(
                self.args.submit,
                targets,
                nodes,
                "active checkpoint pointer is absent",
            ));
        };
        let claims = load_claims(&self.backlog)?;
        let current_waves =
            valid_micro_waves(&self.args.root, &checkpoint_sha256, &checkpoint_epoch)?;
        let Some(newest_wave) = current_waves.last().map(|wave| wave.wave.clone()) else {
            let mut report = empty_report(
                self.args.submit,
                targets,
                nodes,
                "no complete current-checkpoint 128-dataset micro-wave",
            );
            report.active_checkpoint_sha256 = Some(checkpoint_sha256);
            report.checkpoint_epoch = Some(checkpoint_epoch);
            return Ok(report);
        };
        let current_wave_names = current_waves
            .iter()
            .map(|wave| wave.wave.clone())
            .collect::<BTreeSet<_>>();
        let rejected = load_rejected_manifests(&self.backlog)?;
        let mut admission_waves =
            current_admission_waves(&self.args.root, &current_waves, &claims, &rejected)?;
        if admission_waves.len() > MAX_ADMISSION_WAVES {
            bail!(
                "current checkpoint has {} unclaimed waves; maximum admission overlap is {}",
                admission_waves.len(),
                MAX_ADMISSION_WAVES
            );
        }
        let scheduler_by_id = jobs
            .iter()
            .map(|job| (job.id.as_str(), job))
            .collect::<HashMap<_, _>>();
        let mut manifests = HashMap::new();
        let mut useful = BTreeSet::new();
        let mut stale_or_unproven = 0_usize;
        let mut counts = BTreeMap::new();
        let mut max_queue_age = 0.0_f64;
        let now_seconds = (now_ms() / 1_000) as i64;
        for job in &jobs {
            if job.project != PROJECT || !ACTIVE_JOB_STATES.contains(&job.state.as_str()) {
                continue;
            }
            *counts.entry(job.state.clone()).or_insert(0) += 1;
            let valid = match claims.get(&job.id) {
                Some(claim) if matches!(claim.state.as_str(), "submitted" | "submitting") => self
                    .validate_claim_manifest(
                        claim,
                        &checkpoint_sha256,
                        &checkpoint_epoch,
                        &current_wave_names,
                    ),
                _ => Err(anyhow::anyhow!("scheduler job has no active claim")),
            };
            match valid {
                Ok(manifest) => {
                    manifests.insert(job.id.clone(), manifest);
                    useful.insert(job.id.clone());
                    if QUEUE_STATES.contains(&job.state.as_str()) {
                        if let Some(timestamp) =
                            job.queue_ts.as_deref().and_then(parse_rfc3339_seconds)
                        {
                            max_queue_age =
                                max_queue_age.max((now_seconds - timestamp).max(0) as f64);
                        }
                    }
                }
                Err(_) => stale_or_unproven += 1,
            }
        }

        let queued_starting = jobs
            .iter()
            .filter(|job| useful.contains(&job.id) && QUEUE_STATES.contains(&job.state.as_str()))
            .count();
        let running = jobs
            .iter()
            .filter(|job| useful.contains(&job.id) && job.state == "running")
            .count();
        let dispatching = jobs
            .iter()
            .filter(|job| useful.contains(&job.id) && job.state == "dispatching")
            .count();
        let held = jobs
            .iter()
            .filter(|job| useful.contains(&job.id) && job.state == "held")
            .count();

        // 2026-08-01 incident: a host CPU-overload episode (load ~194 on
        // xbabe2) made the scheduler guard-hold ~996 jope-selfplay-eval jobs
        // ("priority preemption for cpu-overload-guard", "CPU target
        // reached"). SCQ's `held` state is sticky — no daemon releases it —
        // so the flywheel sat at 3 running jobs on an otherwise idle cluster
        // (329/408 cores free) for ~3 hours until an operator released them
        // by hand. The keeper is the one process that knows which held jobs
        // are still provenance-useful, so it now releases them, rate-limited,
        // whenever the useful queue is under target. If the host is genuinely
        // overloaded the guard simply holds them again; a release is one
        // cheap socket call, so the worst case is a slow benign loop.
        let mut released_held = 0;
        if self.args.submit && !self.args.preserve_held {
            let budget = held_release_budget(queued_starting, targets.target, held);
            for job in jobs
                .iter()
                .filter(|job| useful.contains(&job.id) && job.state == "held")
            {
                if released_held >= budget {
                    break;
                }
                if scq_release(&self.args.scq, &job.id).is_ok() {
                    released_held += 1;
                }
            }
        }

        let mut adopted = 0;
        let mut retries = 0;
        if self.args.submit {
            adopted = self.adopt_scheduler_jobs(
                &claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
            )?;
            retries = self.prepare_infrastructure_retries(
                &claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
            )?;
        }

        let mut refreshed_claims = load_claims(&self.backlog)?;
        if self.args.submit {
            self.rebalance_launch_ready(
                &refreshed_claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
                targets.target.max(0) as usize,
            )?;
            refreshed_claims = load_claims(&self.backlog)?;
        }
        let mut ready = self.ready_claims(
            &refreshed_claims,
            &scheduler_by_id,
            &checkpoint_sha256,
            &checkpoint_epoch,
            &current_wave_names,
        )?;
        let launch_ready_limit = targets.target.max(0) as usize;
        ready.sort_by(|left, right| {
            left.manifest
                .generated_at_ms
                .cmp(&right.manifest.generated_at_ms)
                .then_with(|| left.claim.job_id.cmp(&right.claim.job_id))
        });
        if ready.len() > launch_ready_limit {
            ready.truncate(launch_ready_limit);
        }
        let mut deferred = self.count_deferred_current(
            &refreshed_claims,
            &scheduler_by_id,
            &checkpoint_sha256,
            &checkpoint_epoch,
            &current_wave_names,
        )?;

        let needed = queue_refill_needed(queued_starting as i64, targets.target, targets.hard_cap)
            .map_err(|error| anyhow::anyhow!(error.to_string()))? as usize;
        let outside_room = launch_ready_limit.saturating_sub(ready.len());
        // Never make launch-ready work wait behind validation of the next
        // batch. Submit it now; the following five-second cycle prepares more.
        let prepare_limit = if ready.is_empty() {
            needed.min(outside_room).min(self.args.max_submit_per_cycle)
        } else {
            0
        };
        let mut prepare = PrepareOutcome::default();
        if self.args.submit && prepare_limit > 0 {
            let mut remaining = prepare_limit;
            for wave in &admission_waves {
                if remaining == 0 {
                    break;
                }
                let outcome = self.prepare_current_manifests(
                    remaining,
                    &checkpoint_sha256,
                    &checkpoint_epoch,
                    &wave.wave,
                    &refreshed_claims,
                )?;
                remaining = remaining.saturating_sub(outcome.prepared);
                prepare.merge(outcome);
                refreshed_claims = load_claims(&self.backlog)?;
            }
            refreshed_claims = load_claims(&self.backlog)?;
            ready = self.ready_claims(
                &refreshed_claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
            )?;
            ready.sort_by(|left, right| left.claim.job_id.cmp(&right.claim.job_id));
            if ready.len() > launch_ready_limit {
                ready.truncate(launch_ready_limit);
            }
            deferred = self.count_deferred_current(
                &refreshed_claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
            )?;
        }

        let submit_count = needed
            .min(self.args.max_submit_per_cycle)
            .min(ready.len())
            .min((targets.hard_cap as usize).saturating_sub(queued_starting));
        let (submitted, submission_failures) = if self.args.submit && submit_count > 0 {
            self.submit_claims(&ready[..submit_count])?
        } else {
            (0, 0)
        };
        let report_claims = load_claims(&self.backlog)?;
        let report_rejected = load_rejected_manifests(&self.backlog)?;
        admission_waves = current_admission_waves(
            &self.args.root,
            &current_waves,
            &report_claims,
            &report_rejected,
        )?;
        if admission_waves.len() > MAX_ADMISSION_WAVES {
            bail!(
                "current checkpoint has {} unclaimed waves after admission; maximum overlap is {}",
                admission_waves.len(),
                MAX_ADMISSION_WAVES
            );
        }
        let unclaimed_current_checkpoint_manifests = admission_waves
            .iter()
            .map(|wave| wave.unclaimed_manifests)
            .sum();

        Ok(CycleReport {
            schema: "JopeSelfplayQueueKeeper.v3",
            generated_at_ms: now_ms(),
            mode: if self.args.submit { "submit" } else { "audit" },
            admission_allowed: true,
            degraded: false,
            error: None,
            fail_closed_reason: None,
            active_checkpoint_sha256: Some(checkpoint_sha256),
            checkpoint_epoch: Some(checkpoint_epoch),
            newest_micro_wave: Some(newest_wave),
            admission_waves,
            unclaimed_current_checkpoint_manifests,
            queued_target: targets.target,
            queued_hard_cap: targets.hard_cap,
            p95_60m_jobs_per_minute: targets.p95_jobs_per_minute,
            queued_starting,
            running,
            dispatching,
            held,
            released_held,
            queue_maximum_age_seconds: max_queue_age,
            stale_or_unproven_active: stale_or_unproven,
            launch_ready: ready.len(),
            launch_ready_deferred_by_horizon: deferred,
            manifests_unclaimed: prepare.unclaimed,
            manifests_validated: prepare.validated,
            manifest_validation_failures: prepare.validation_failures,
            pair_reservation_conflicts: prepare.reservation_conflicts,
            prepared: prepare.prepared,
            infrastructure_retries_prepared: retries,
            adopted_after_restart: adopted,
            submitted,
            submission_failures,
            effective_cluster_slots,
            nodes,
            state_counts: counts,
        })
    }

    fn validate_claim_manifest(
        &mut self,
        claim: &Claim,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<ValidManifest> {
        let wave = manifest_wave(&claim.manifest)?;
        if !waves.contains(&wave) {
            bail!("claim manifest wave is not current-checkpoint immutable supply");
        }
        let cache_key = (claim.manifest.clone(), claim.manifest_sha256.clone());
        if let Some(manifest) = self.manifest_cache.get(&cache_key) {
            return Ok(manifest.clone());
        }
        let manifest = validate_manifest(&claim.manifest, checkpoint_sha256, epoch, &wave)?;
        if manifest.sha256 != claim.manifest_sha256 {
            bail!("claim manifest hash drift");
        }
        self.manifest_cache.insert(cache_key, manifest.clone());
        Ok(manifest)
    }

    fn ready_claims(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<Vec<ReadyClaim>> {
        let mut ready = Vec::new();
        for claim in claims.values() {
            if !matches!(claim.state.as_str(), "checked" | "submitting")
                || scheduler.contains_key(claim.job_id.as_str())
            {
                continue;
            }
            let manifest =
                match self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
            let owner = self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                (claim.state == "submitting").then_some(claim.job_id.as_str()),
            )?;
            ready.push(ReadyClaim {
                claim: claim.clone(),
                manifest,
                owner,
            });
        }
        Ok(ready)
    }

    fn rebalance_launch_ready(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
        limit: usize,
    ) -> Result<()> {
        let mut eligible = Vec::new();
        for claim in claims.values() {
            if !matches!(
                claim.state.as_str(),
                "checked" | "submitting" | "deferred_horizon"
            ) || scheduler.contains_key(claim.job_id.as_str())
            {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                (claim.state == "submitting").then_some(claim.job_id.as_str()),
            )?;
            eligible.push((claim, manifest.generated_at_ms));
        }
        eligible.sort_by(|(left, left_time), (right, right_time)| {
            // A crash-window `submitting` claim keeps priority so its
            // deterministic SCQ identity is reconciled before fresh work.
            let left_submit = left.state == "submitting";
            let right_submit = right.state == "submitting";
            right_submit
                .cmp(&left_submit)
                .then_with(|| left_time.cmp(right_time))
                .then_with(|| left.job_id.cmp(&right.job_id))
        });
        let tx = self
            .backlog
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        for (index, (claim, _)) in eligible.into_iter().enumerate() {
            let state = if index < limit {
                if claim.state == "submitting" {
                    "submitting"
                } else {
                    "checked"
                }
            } else {
                "deferred_horizon"
            };
            tx.execute(
                "UPDATE claims SET state=?,updated_at_ms=? WHERE manifest_sha256=?",
                params![state, now_ms() as i64, claim.manifest_sha256],
            )?;
            tx.execute(
                "UPDATE keeper_attempts SET state=? WHERE manifest_sha256=? \
                 AND state IN ('checked','submitting','deferred_horizon')",
                params![state, claim.manifest_sha256],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    fn count_deferred_current(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<usize> {
        let mut count = 0;
        for claim in claims.values() {
            if claim.state != "deferred_horizon" || scheduler.contains_key(claim.job_id.as_str()) {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            self.validate_pair_reservations(&manifest, claim.owner.as_deref(), None)?;
            count += 1;
        }
        Ok(count)
    }

    fn validate_pair_reservations(
        &mut self,
        manifest: &ValidManifest,
        claimed_owner: Option<&str>,
        allow_submitted_job: Option<&str>,
    ) -> Result<String> {
        let cache_key = (
            manifest.sha256.clone(),
            claimed_owner.map(str::to_owned),
            allow_submitted_job.map(str::to_owned),
        );
        if let Some(owner) = self.reservation_cache.get(&cache_key) {
            return Ok(owner.clone());
        }
        let owner_col = self.pair_schema.owner_column;
        let query = format!(
            "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,state,{owner_col},\
             manifest_path,manifest_sha256,job_id FROM pairs WHERE manifest_sha256=?"
        );
        let expected_pairs = manifest
            .pairs
            .iter()
            .map(|pair| {
                (
                    pair.task.clone(),
                    pair.dataset_hash.clone(),
                    pair.semantic_sha256.clone(),
                    pair.protocol_sha256.clone(),
                )
            })
            .collect::<BTreeSet<_>>();
        let (owner, observed_pairs) = {
            let mut statement = self.pairs.prepare_cached(&query)?;
            let rows = statement.query_map([manifest.sha256.as_str()], |row| {
                Ok((
                    (
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ),
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, Option<String>>(8)?,
                ))
            })?;
            let mut owner = None;
            let mut observed_pairs = BTreeSet::new();
            for row in rows {
                let (pair, state, pair_owner, path, digest, submitted_job) = row?;
                observed_pairs.insert(pair);
                let launch_ready = matches!(state.as_str(), "reserved" | "infrastructure_retry");
                let crash_adoptable = state == "submitted"
                    && allow_submitted_job.is_some()
                    && submitted_job.as_deref() == allow_submitted_job;
                if !launch_ready && !crash_adoptable {
                    bail!("manifest protocol pair is not launch-ready");
                }
                if path.as_deref() != Some(manifest.path.to_string_lossy().as_ref())
                    || digest.as_deref() != Some(manifest.sha256.as_str())
                {
                    bail!("pair/manifest binding drift");
                }
                if let Some(expected) = claimed_owner {
                    if pair_owner != expected {
                        bail!("claim/pair owner drift");
                    }
                }
                if owner
                    .replace(pair_owner.clone())
                    .is_some_and(|prior| prior != pair_owner)
                {
                    bail!("manifest spans reservation owners");
                }
            }
            (owner, observed_pairs)
        };
        if observed_pairs != expected_pairs {
            bail!("manifest reserved protocol-pair set mismatch");
        }
        let owner = owner.context("manifest has no reserved protocol pairs")?;
        self.reservation_cache.insert(cache_key, owner.clone());
        Ok(owner)
    }

    fn prepare_current_manifests(
        &mut self,
        limit: usize,
        checkpoint_sha256: &str,
        epoch: &str,
        wave: &str,
        claims: &HashMap<String, Claim>,
    ) -> Result<PrepareOutcome> {
        let directory = wave_manifest_directory(&self.args.root, wave);
        let mut paths = fs::read_dir(&directory)?
            .filter_map(|entry| entry.ok().map(|value| value.path()))
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
            .collect::<Vec<_>>();
        paths.sort();
        let claimed = claims
            .values()
            .map(|claim| claim.manifest.clone())
            .collect::<BTreeSet<_>>();
        let rejected = load_rejected_manifests(&self.backlog)?;
        let candidate_paths = paths
            .into_iter()
            .filter(|path| !claimed.contains(path))
            .filter(|path| !rejected.contains(path))
            .collect::<Vec<_>>();
        let unclaimed = candidate_paths.len();
        // Admission must not wait for an eager sweep of an entire micro-wave.
        // Validate enough manifests to fill this cycle plus bounded headroom for
        // invalid or concurrently reserved pairs; later cycles consume the rest.
        let validation_budget = bounded_validation_budget(unclaimed, limit);
        let validation_pool = ThreadPoolBuilder::new()
            .num_threads(32)
            .thread_name(|index| format!("jope-manifest-validate-{index}"))
            .build()?;
        let validation_results = validation_pool.install(|| {
            candidate_paths
                .par_iter()
                .take(validation_budget)
                .map(|path| {
                    let manifest = match validate_manifest(path, checkpoint_sha256, epoch, wave) {
                        Ok(manifest) => manifest,
                        Err(error) => return (path.clone(), Err(error), true),
                    };
                    match validate_with_evaluator(&self.args.evaluator, path) {
                        Ok(()) => (path.clone(), Ok(manifest), false),
                        Err(error) => {
                            let permanent = evaluator_failure_is_permanent(&error);
                            (path.clone(), Err(error), permanent)
                        }
                    }
                })
                .collect::<Vec<_>>()
        });
        let validation_failures = validation_results
            .iter()
            .filter(|(_, result, _)| result.is_err())
            .count();
        let validated = validation_results.len().saturating_sub(validation_failures);
        for (path, result, permanent) in &validation_results {
            if *permanent {
                let error = result
                    .as_ref()
                    .expect_err("permanent validation outcome is an error");
                reject_manifest(
                    &self.backlog,
                    path,
                    None,
                    &format!("validation_failed:{error:#}"),
                )?;
            }
        }
        let mut manifests = validation_results
            .into_iter()
            .filter_map(|(_, result, _)| result.ok())
            .collect::<Vec<_>>();
        manifests.sort_by(|left, right| left.path.cmp(&right.path));

        let owner_column = self.pair_schema.owner_column;
        let tx = self
            .pairs
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let halted: Option<String> = tx
            .query_row(
                "SELECT value FROM controls WHERE key='admission_halted'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if halted.as_deref().is_some_and(|value| value != "0") {
            bail!("pair admission is halted by the uniqueness authority");
        }
        let mut accepted = Vec::new();
        let mut conflicts = Vec::new();
        let mut reservation_conflicts = 0;
        for manifest in manifests {
            if accepted.len() == limit {
                break;
            }
            let owner = match reserve_manifest_pairs_in_tx(&tx, owner_column, &manifest)? {
                Some(value) => value,
                None => {
                    reservation_conflicts += 1;
                    conflicts.push((manifest.path.clone(), manifest.sha256.clone()));
                    continue;
                }
            };
            accepted.push((manifest, owner));
        }
        tx.commit()?;
        for (path, digest) in conflicts {
            reject_manifest(
                &self.backlog,
                &path,
                Some(&digest),
                "pair_reservation_conflict",
            )?;
        }

        let preparation_pool = ThreadPoolBuilder::new()
            .num_threads(32)
            .thread_name(|index| format!("jope-manifest-prepare-{index}"))
            .build()?;
        let prepared_rows = preparation_pool.install(|| {
            accepted
                .into_par_iter()
                .map(|(manifest, owner)| {
                    let job_id = format!("jte-{}", &manifest.sha256[..24]);
                    let spec = self
                        .args
                        .root
                        .join("specs/evaluations")
                        .join(format!("{job_id}.toml"));
                    render_spec(
                        &spec,
                        &job_id,
                        &manifest,
                        &self.args.runner,
                        &self.args.evaluator,
                    )?;
                    // `submit-batch` runs SCQ's complete authoritative
                    // validation path for every spec before it mutates the
                    // scheduler. Running `scq check` here repeated the same
                    // controller round trip once per manifest and could leave
                    // the cluster starved while a large fresh wave waited.
                    // The manifest/evaluator/input hashes have already been
                    // validated above; permanent batch rejections remain
                    // recorded by `submit_claims`.
                    Ok((manifest, owner, job_id, spec))
                })
                .collect::<Vec<Result<_>>>()
        });
        let prepared_rows = prepared_rows.into_iter().collect::<Result<Vec<_>>>()?;
        let prepared = prepared_rows.len();
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            for (manifest, owner, job_id, spec) in &prepared_rows {
                tx.execute(
                    "INSERT INTO claims(manifest_sha256,manifest,job_id,spec,state,attempts,\
                     reservation_owner,updated_at_ms) VALUES(?,?,?,?, 'checked',0,?,?)",
                    params![
                        manifest.sha256,
                        manifest.path.to_string_lossy(),
                        job_id,
                        spec.to_string_lossy(),
                        owner,
                        now_ms() as i64
                    ],
                )?;
                append_keeper_attempts(&tx, manifest, owner, job_id, "checked", 1)?;
            }
            tx.commit()?;
        }
        for (manifest, owner, _, _) in prepared_rows {
            self.manifest_cache.insert(
                (manifest.path.clone(), manifest.sha256.clone()),
                manifest.clone(),
            );
            self.reservation_cache
                .insert((manifest.sha256.clone(), Some(owner.clone()), None), owner);
        }
        Ok(PrepareOutcome {
            unclaimed,
            validated,
            validation_failures,
            reservation_conflicts,
            prepared,
        })
    }

    fn adopt_scheduler_jobs(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<usize> {
        let mut rows = Vec::new();
        for claim in claims.values() {
            if !matches!(claim.state.as_str(), "checked" | "submitting") {
                continue;
            }
            let Some(job) = scheduler.get(claim.job_id.as_str()) else {
                continue;
            };
            if !scheduler_job_is_adoptable(job) {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            let owner = self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                Some(claim.job_id.as_str()),
            )?;
            rows.push((manifest, owner, claim.job_id.clone()));
        }
        if rows.is_empty() {
            return Ok(0);
        }
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            {
                let mut attempts = tx.prepare_cached(
                    "SELECT COUNT(*) FROM keeper_attempts WHERE manifest_sha256=? AND job_id=? \
                     AND state IN ('checked','submitting','submitted')",
                )?;
                for (manifest, _, job_id) in &rows {
                    let count: i64 =
                        attempts.query_row(params![manifest.sha256, job_id], |row| row.get(0))?;
                    if count != manifest.pairs.len() as i64 {
                        bail!("scheduler adoption has incomplete active attempt history");
                    }
                }
            }
            tx.commit()?;
        }
        let submissions = rows
            .iter()
            .map(|(manifest, owner, job_id)| (manifest, owner.as_str(), job_id.as_str()))
            .collect::<Vec<_>>();
        self.record_submissions(&submissions)?;
        Ok(rows.len())
    }

    fn record_submissions(&mut self, submissions: &[(&ValidManifest, &str, &str)]) -> Result<()> {
        if submissions.is_empty() {
            return Ok(());
        }
        {
            let tx = self
                .pairs
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            for (manifest, owner, job_id) in submissions {
                ensure_native_reserved_attempts(&tx, &self.pair_schema, manifest, owner, job_id)?;
                update_pairs_submitted(&tx, &self.pair_schema, manifest, owner, job_id)?;
                update_native_attempts_submitted(&tx, &self.pair_schema, manifest, owner, job_id)?;
            }
            tx.commit()?;
        }
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            {
                let mut update_claim = tx.prepare_cached(
                    "UPDATE claims SET state='submitted',attempts=attempts+1,\
                     reservation_owner=?,updated_at_ms=? WHERE manifest_sha256=?",
                )?;
                let mut update_attempt = tx.prepare_cached(
                    "UPDATE keeper_attempts SET state='submitted',job_id=? \
                     WHERE manifest_sha256=? AND state IN ('checked','submitting')",
                )?;
                for (manifest, owner, job_id) in submissions {
                    update_claim.execute(params![owner, now_ms() as i64, manifest.sha256])?;
                    update_attempt.execute(params![job_id, manifest.sha256])?;
                }
            }
            tx.commit()?;
        }
        Ok(())
    }

    fn submit_claims(&mut self, claims: &[ReadyClaim]) -> Result<(usize, usize)> {
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            for row in claims {
                let changed = tx.execute(
                    "UPDATE claims SET state='submitting',updated_at_ms=? \
                     WHERE manifest_sha256=? AND state IN ('checked','submitting')",
                    params![now_ms() as i64, row.manifest.sha256],
                )?;
                if changed != 1 {
                    bail!("launch-ready claim changed before submission");
                }
                let attempts_changed = tx.execute(
                    "UPDATE keeper_attempts SET state='submitting' \
                     WHERE manifest_sha256=? AND state IN ('checked','submitting')",
                    [row.manifest.sha256.as_str()],
                )?;
                if attempts_changed != row.manifest.pairs.len() {
                    bail!("launch-ready claim has incomplete active attempt history");
                }
            }
            tx.commit()?;
        }
        let batch_outcomes = submit_batch(&self.args.scq, claims)?;
        let outcomes = claims
            .iter()
            .cloned()
            .zip(batch_outcomes)
            .map(|(row, (accepted, transient, detail))| (row, accepted, transient, detail))
            .collect::<Vec<_>>();
        let mut submitted = 0;
        let mut failed = 0;
        let accepted_rows = outcomes
            .iter()
            .filter(|(_, accepted, _, _)| *accepted)
            .map(|(row, _, _, _)| (&row.manifest, row.owner.as_str(), row.claim.job_id.as_str()))
            .collect::<Vec<_>>();
        self.record_submissions(&accepted_rows)?;
        for (row, accepted, transient, detail) in outcomes {
            if accepted {
                submitted += 1;
            } else {
                failed += 1;
                self.backlog.execute(
                    "UPDATE claims SET state=?,updated_at_ms=? WHERE manifest_sha256=?",
                    params![
                        if transient {
                            "checked".to_owned()
                        } else {
                            format!(
                                "submit_failed:{}",
                                detail.chars().take(500).collect::<String>()
                            )
                        },
                        now_ms() as i64,
                        row.manifest.sha256
                    ],
                )?;
                self.backlog.execute(
                    "UPDATE keeper_attempts SET state=? WHERE manifest_sha256=? \
                     AND state='submitting'",
                    params![
                        if transient {
                            "checked"
                        } else {
                            "submit_failed"
                        },
                        row.manifest.sha256
                    ],
                )?;
            }
        }
        Ok((submitted, failed))
    }

    fn prepare_infrastructure_retries(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<usize> {
        let mut recovered = 0;
        for claim in claims.values() {
            if claim.state != "submitted" || claim.attempts >= 5 {
                continue;
            }
            let Some(job) = scheduler.get(claim.job_id.as_str()) else {
                // Missing scheduler state is ambiguous. Never infer an
                // infrastructure failure and never retry it.
                continue;
            };
            if job.state != "failed"
                || !job
                    .failure_class
                    .as_deref()
                    .is_some_and(is_infrastructure_failure_class)
            {
                continue;
            }
            if authoritative_receipt_exists(&self.args.root, &claim.job_id) {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            let owner = self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                Some(claim.job_id.as_str()),
            )?;
            let retry_number = claim.attempts;
            let new_job_id = format!("{}-r{retry_number}", claim.job_id);
            let new_spec = claim.spec.with_file_name(format!("{new_job_id}.toml"));
            rewrite_spec_job_id(&claim.spec, &new_spec, &claim.job_id, &new_job_id)?;
            scq_check(&self.args.scq, &new_spec)?;
            let failure = job.failure_class.as_deref().unwrap_or("infrastructure");
            self.mark_infrastructure_retry(&manifest, failure, &new_job_id, &owner)?;
            self.backlog.execute(
                "UPDATE claims SET job_id=?,spec=?,state='checked',updated_at_ms=? \
                 WHERE manifest_sha256=? AND state='submitted' AND job_id=?",
                params![
                    new_job_id,
                    new_spec.to_string_lossy(),
                    now_ms() as i64,
                    manifest.sha256,
                    claim.job_id
                ],
            )?;
            recovered += 1;
        }
        Ok(recovered)
    }

    fn mark_infrastructure_retry(
        &mut self,
        manifest: &ValidManifest,
        failure: &str,
        new_job_id: &str,
        owner: &str,
    ) -> Result<()> {
        let tx = self
            .pairs
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let retry_increment = if self.pair_schema.has_infrastructure_retries {
            ",infrastructure_retries=infrastructure_retries+1"
        } else {
            ""
        };
        let changed = tx.execute(
            &format!(
                "UPDATE pairs SET state='infrastructure_retry',failure_class=?{retry_increment},\
                 job_id=NULL,updated_at_ms=? WHERE manifest_sha256=? AND state='submitted'"
            ),
            params![failure, now_ms() as i64, manifest.sha256],
        )?;
        if changed != manifest.pairs.len() {
            let prior: i64 = tx.query_row(
                "SELECT COUNT(*) FROM pairs WHERE manifest_sha256=? \
                 AND state='infrastructure_retry' AND job_id IS NULL",
                [manifest.sha256.as_str()],
                |row| row.get(0),
            )?;
            if changed != 0 || prior != manifest.pairs.len() as i64 {
                bail!("infrastructure retry pair transition count mismatch");
            }
        }
        finalize_native_attempts_for_retry(&tx, &self.pair_schema, manifest, failure)?;
        append_native_retry_attempts(&tx, &self.pair_schema, manifest, owner, new_job_id)?;
        tx.commit()?;
        let tx = self
            .backlog
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "UPDATE keeper_attempts SET state='infrastructure_retry',failure_class=?,\
             finalized_at_ms=? WHERE manifest_sha256=? AND state='submitted'",
            params![failure, now_ms() as i64, manifest.sha256],
        )?;
        append_keeper_attempts(&tx, manifest, owner, new_job_id, "checked", 1)?;
        tx.commit()?;
        Ok(())
    }
}

fn reserve_manifest_pairs_in_tx(
    tx: &Transaction<'_>,
    owner_column: &str,
    manifest: &ValidManifest,
) -> Result<Option<String>> {
    let owner = format!("queue-keeper:{}", &manifest.sha256[..24]);
    let manifest_path = manifest.path.to_string_lossy().into_owned();
    let candidates = manifest
        .value
        .get("candidates")
        .and_then(Value::as_array)
        .context("validated manifest lost its candidates")?;
    ensure!(
        candidates.len() == manifest.pairs.len(),
        "validated manifest pair/candidate count drift"
    );
    let genomes = candidates
        .iter()
        .map(|candidate| {
            let semantic = candidate
                .get("semantic_sha256")
                .and_then(Value::as_str)
                .context("validated candidate lost its semantic identity")?;
            let genome = candidate
                .get("canonical_genome")
                .and_then(Value::as_str)
                .context("validated candidate lost its canonical genome")?;
            Ok((semantic.to_owned(), genome.to_owned()))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    ensure!(
        genomes.len() == manifest.pairs.len(),
        "validated manifest repeats a semantic identity"
    );

    let lookup = format!(
        "SELECT state,{owner_column},manifest_path,manifest_sha256,job_id,\
         receipt_sha256,canonical_genome FROM pairs WHERE task=? AND dataset_hash=? \
         AND semantic_sha256=? AND protocol_sha256=?"
    );
    let mut lookup_statement = tx.prepare_cached(&lookup)?;
    for pair in &manifest.pairs {
        let existing: Option<PairReservationRow> = lookup_statement
            .query_row(
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                },
            )
            .optional()?;
        let Some((state, pair_owner, path, digest, job_id, receipt, canonical)) = existing else {
            continue;
        };
        let expected_genome = genomes
            .get(&pair.semantic_sha256)
            .context("validated pair has no canonical genome")?;
        if canonical
            .as_deref()
            .is_some_and(|value| value != expected_genome)
        {
            bail!("pair registry semantic hash collision");
        }
        let reusable = state == "available" && receipt.is_none() && job_id.is_none();
        let already_reserved = state == "reserved"
            && pair_owner == owner
            && path.as_deref() == Some(manifest_path.as_str())
            && digest.as_deref() == Some(manifest.sha256.as_str())
            && job_id.is_none()
            && receipt.is_none();
        if !reusable && !already_reserved {
            return Ok(None);
        }
    }

    let timestamp = now_ms() as i64;
    let insert = format!(
        "INSERT OR IGNORE INTO pairs(task,dataset_hash,semantic_sha256,protocol_sha256,\
         canonical_genome,state,{owner_column},manifest_path,manifest_sha256,job_id,\
         receipt_sha256,failure_class,created_at_ms,updated_at_ms) \
         VALUES(?,?,?,?,?,'reserved',?,?,?,NULL,NULL,NULL,?,?)"
    );
    let update = format!(
        "UPDATE pairs SET canonical_genome=COALESCE(canonical_genome,?),state='reserved',\
         {owner_column}=?,manifest_path=?,manifest_sha256=?,job_id=NULL,\
         receipt_sha256=NULL,failure_class=NULL,updated_at_ms=? \
         WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
         AND state='available' AND receipt_sha256 IS NULL AND job_id IS NULL"
    );
    let mut insert_statement = tx.prepare_cached(&insert)?;
    let mut update_statement = tx.prepare_cached(&update)?;
    let mut event_statement = tx.prepare_cached(
        "INSERT INTO events(kind,task,dataset_hash,semantic_sha256,protocol_sha256,\
         detail_json,created_at_ms) VALUES('reserved',?,?,?,?,?,?)",
    )?;
    for pair in &manifest.pairs {
        let genome = genomes
            .get(&pair.semantic_sha256)
            .context("validated pair has no canonical genome")?;
        let inserted = insert_statement.execute(params![
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            genome,
            owner,
            manifest_path,
            manifest.sha256,
            timestamp,
            timestamp
        ])?;
        let changed = if inserted == 1 {
            1
        } else {
            update_statement.execute(params![
                genome,
                owner,
                manifest_path,
                manifest.sha256,
                timestamp,
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256
            ])?
        };
        if changed == 1 {
            event_statement.execute(params![
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256,
                serde_json::to_string(&json!({
                    "owner": owner,
                    "manifest_path": manifest_path,
                    "manifest_sha256": manifest.sha256,
                }))?,
                timestamp
            ])?;
        }
    }
    Ok(Some(owner))
}

fn empty_report(
    submit: bool,
    targets: QueueTargets,
    nodes: Vec<NodeSlots>,
    reason: &str,
) -> CycleReport {
    CycleReport {
        schema: "JopeSelfplayQueueKeeper.v3",
        generated_at_ms: now_ms(),
        mode: if submit { "submit" } else { "audit" },
        admission_allowed: false,
        degraded: false,
        error: None,
        fail_closed_reason: Some(reason.to_owned()),
        active_checkpoint_sha256: None,
        checkpoint_epoch: None,
        newest_micro_wave: None,
        admission_waves: Vec::new(),
        unclaimed_current_checkpoint_manifests: 0,
        queued_target: targets.target,
        queued_hard_cap: targets.hard_cap,
        p95_60m_jobs_per_minute: targets.p95_jobs_per_minute,
        queued_starting: 0,
        running: 0,
        dispatching: 0,
        held: 0,
        released_held: 0,
        queue_maximum_age_seconds: 0.0,
        stale_or_unproven_active: 0,
        launch_ready: 0,
        launch_ready_deferred_by_horizon: 0,
        manifests_unclaimed: 0,
        manifests_validated: 0,
        manifest_validation_failures: 0,
        pair_reservation_conflicts: 0,
        prepared: 0,
        infrastructure_retries_prepared: 0,
        adopted_after_restart: 0,
        submitted: 0,
        submission_failures: 0,
        effective_cluster_slots: nodes.iter().map(|row| row.slot_target).sum(),
        nodes,
        state_counts: BTreeMap::new(),
    }
}

fn failure_report(submit: bool, error: &str) -> CycleReport {
    let mut report = empty_report(
        submit,
        QueueTargets {
            target: 0,
            hard_cap: 0,
            p95_jobs_per_minute: 0.0,
        },
        Vec::new(),
        "queue cycle failed closed",
    );
    report.degraded = true;
    report.error = Some(error.to_owned());
    report
}

fn initialize_backlog(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        r#"
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS claims(
          manifest_sha256 TEXT PRIMARY KEY,
          manifest TEXT NOT NULL UNIQUE,
          job_id TEXT NOT NULL UNIQUE,
          spec TEXT NOT NULL,
          state TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0,
          reservation_owner TEXT,
          updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS terminal_jobs(
          job_id TEXT PRIMARY KEY,
          observed_at_ms INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS burn_samples(
          observed_at_ms INTEGER PRIMARY KEY,
          jobs_per_minute REAL NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS rejected_manifests(
          manifest TEXT PRIMARY KEY,
          manifest_sha256 TEXT,
          reason TEXT NOT NULL,
          updated_at_ms INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS keeper_attempts(
          id INTEGER PRIMARY KEY,
          task TEXT NOT NULL,
          dataset_hash TEXT NOT NULL,
          semantic_sha256 TEXT NOT NULL,
          protocol_sha256 TEXT NOT NULL,
          attempt_number INTEGER NOT NULL,
          state TEXT NOT NULL,
          owner TEXT NOT NULL,
          manifest_path TEXT NOT NULL,
          manifest_sha256 TEXT NOT NULL,
          job_id TEXT NOT NULL,
          failure_class TEXT,
          created_at_ms INTEGER NOT NULL,
          finalized_at_ms INTEGER,
          UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_number)
        );
        CREATE INDEX IF NOT EXISTS keeper_attempt_manifest_state
          ON keeper_attempts(manifest_sha256,state);
        "#,
    )?;
    // This index covers millions of attempt rows. Rebuilding it on every
    // daemon restart turns a five-second keeper recovery into minutes of disk
    // saturation. Migrate only when the declared predicate actually changes.
    const ACTIVE_INDEX_SQL: &str = "CREATE UNIQUE INDEX keeper_one_active_attempt \
        ON keeper_attempts(task,dataset_hash,semantic_sha256,protocol_sha256) \
        WHERE state IN ('checked','submitting','submitted','deferred_horizon')";
    let existing: Option<String> = connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='keeper_one_active_attempt'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let normalized = |sql: &str| {
        sql.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase()
    };
    if existing.as_deref().map(|sql| normalized(sql)) != Some(normalized(ACTIVE_INDEX_SQL)) {
        connection.execute_batch(
            "DROP INDEX IF EXISTS keeper_one_active_attempt;
             CREATE UNIQUE INDEX keeper_one_active_attempt
               ON keeper_attempts(task,dataset_hash,semantic_sha256,protocol_sha256)
               WHERE state IN ('checked','submitting','submitted','deferred_horizon');",
        )?;
    }
    Ok(())
}

fn inspect_pair_schema(connection: &Connection) -> Result<PairSchema> {
    let columns = table_columns(connection, "pairs")?;
    if columns.is_empty() {
        bail!("pair registry lacks the pairs table");
    }
    let owner_column = if columns.contains("reservation_owner") {
        "reservation_owner"
    } else if columns.contains("owner") {
        "owner"
    } else {
        bail!("pair registry lacks an owner column");
    };
    let attempt_columns = table_columns(connection, "attempts")?;
    let attempts = if attempt_columns.is_empty() {
        None
    } else {
        let number_column = if attempt_columns.contains("attempt_no") {
            "attempt_no"
        } else if attempt_columns.contains("attempt_number") {
            "attempt_number"
        } else {
            bail!("attempt ledger lacks an attempt number");
        };
        let attempt_owner = if attempt_columns.contains("reservation_owner") {
            "reservation_owner"
        } else if attempt_columns.contains("owner") {
            "owner"
        } else {
            bail!("attempt ledger lacks an owner column");
        };
        Some(AttemptSchema {
            number_column,
            owner_column: attempt_owner,
        })
    };
    Ok(PairSchema {
        owner_column,
        has_infrastructure_retries: columns.contains("infrastructure_retries"),
        attempts,
    })
}

fn table_columns(connection: &Connection, table: &str) -> Result<BTreeSet<String>> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let rows = statement.query_map([], |row| row.get::<_, String>(1))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn load_claims(connection: &Connection) -> Result<HashMap<String, Claim>> {
    let mut statement = connection.prepare(
        "SELECT manifest_sha256,manifest,job_id,spec,state,attempts,reservation_owner \
         FROM claims",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(Claim {
            manifest_sha256: row.get(0)?,
            manifest: PathBuf::from(row.get::<_, String>(1)?),
            job_id: row.get(2)?,
            spec: PathBuf::from(row.get::<_, String>(3)?),
            state: row.get(4)?,
            attempts: row.get(5)?,
            owner: row.get(6)?,
        })
    })?;
    let mut result = HashMap::new();
    for row in rows {
        let claim = row?;
        if result.insert(claim.job_id.clone(), claim).is_some() {
            bail!("backlog contains duplicate job identities");
        }
    }
    Ok(result)
}

fn load_rejected_manifests(connection: &Connection) -> Result<BTreeSet<PathBuf>> {
    let mut statement = connection.prepare("SELECT manifest FROM rejected_manifests")?;
    let rows = statement.query_map([], |row| Ok(PathBuf::from(row.get::<_, String>(0)?)))?;
    let mut rejected = BTreeSet::new();
    for row in rows {
        rejected.insert(row?);
    }
    Ok(rejected)
}

fn reject_manifest(
    connection: &Connection,
    path: &Path,
    digest: Option<&str>,
    reason: &str,
) -> Result<()> {
    connection.execute(
        "INSERT INTO rejected_manifests(manifest,manifest_sha256,reason,updated_at_ms) \
         VALUES(?,?,?,?) ON CONFLICT(manifest) DO UPDATE SET \
         manifest_sha256=excluded.manifest_sha256,reason=excluded.reason,\
         updated_at_ms=excluded.updated_at_ms",
        params![
            path.to_string_lossy(),
            digest,
            reason.chars().take(500).collect::<String>(),
            now_ms() as i64
        ],
    )?;
    Ok(())
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

fn valid_micro_waves(root: &Path, checkpoint_sha256: &str, epoch: &str) -> Result<Vec<MicroWave>> {
    let directory = root.join("waves");
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut waves = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let Some(wave) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !valid_wave_id(wave) {
            continue;
        }
        let summary_path = path.join("summary.json");
        if !summary_path.is_file() {
            continue;
        }
        let summary = match read_json(&summary_path) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let valid = summary.get("wave_kind").and_then(Value::as_str) == Some(MICRO_WAVE_KIND)
            && summary.get("wave").and_then(Value::as_str) == Some(wave)
            && summary.get("checkpoint_epoch").and_then(Value::as_str) == Some(epoch)
            && summary
                .get("checkpoint_sha256")
                .or_else(|| summary.get("model_checkpoint_sha256"))
                .and_then(Value::as_str)
                == Some(checkpoint_sha256)
            // Shape check driven by the summary's own task_datasets map: every
            // declared task carries exactly one 64-dataset lane, the mandatory
            // regression/binary backbone is present, and the dataset total is
            // the lane sum. The previous literal (exactly 128 = 64+64) silently
            // dropped valid 192/256 waves -- a whole wave of evaluation work
            // vanishing with no error is exactly the failure class epoch-06
            // lane 09 warned about.
            && summary
                .get("task_datasets")
                .and_then(Value::as_object)
                .is_some_and(|tasks| {
                    !tasks.is_empty()
                        && tasks.values().all(|count| count.as_u64() == Some(64))
                        && tasks.contains_key("regression")
                        && tasks.contains_key("binary")
                        && summary.get("datasets").and_then(Value::as_u64)
                            == Some(64 * tasks.len() as u64)
                })
            && summary
                .get("candidates_per_dataset")
                .and_then(Value::as_u64)
                == Some(256)
            && summary
                .get("max_candidates_per_manifest")
                .and_then(Value::as_u64)
                .is_some_and(|count| count <= 100)
            && wave_manifest_directory(root, wave).is_dir();
        if valid {
            waves.push(MicroWave {
                wave: wave.to_owned(),
                summary_sha256: sha256_file(&summary_path)?,
            });
        }
    }
    waves.sort_by(|left, right| left.wave.cmp(&right.wave));
    Ok(waves)
}

fn current_admission_waves(
    root: &Path,
    waves: &[MicroWave],
    claims: &HashMap<String, Claim>,
    rejected: &BTreeSet<PathBuf>,
) -> Result<Vec<AdmissionWave>> {
    let claimed = claims
        .values()
        .map(|claim| claim.manifest.as_path())
        .collect::<BTreeSet<_>>();
    let mut admission = Vec::new();
    for wave in waves {
        let directory = wave_manifest_directory(root, &wave.wave);
        let unclaimed_manifests = fs::read_dir(&directory)?
            .filter_map(|entry| entry.ok().map(|value| value.path()))
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
            .filter(|path| !claimed.contains(path.as_path()))
            .filter(|path| !rejected.contains(path))
            .count();
        if unclaimed_manifests > 0 {
            admission.push(AdmissionWave {
                wave: wave.wave.clone(),
                summary_sha256: wave.summary_sha256.clone(),
                unclaimed_manifests,
            });
        }
    }
    Ok(admission)
}

fn wave_manifest_directory(root: &Path, wave: &str) -> PathBuf {
    let canonical = root.join("waves").join(wave).join("manifests");
    if canonical.is_dir() {
        canonical
    } else {
        // Read-only compatibility for waves created before the immutable
        // wave bundle owned its manifest directory.
        root.join("manifests").join(wave)
    }
}

fn valid_wave_id(value: &str) -> bool {
    value
        .strip_prefix("strict-wave-")
        .is_some_and(|suffix| suffix.len() == 6 && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

fn manifest_wave(path: &Path) -> Result<String> {
    let value = read_json(path)?;
    let wave = value
        .get("strict_wave")
        .and_then(Value::as_str)
        .context("manifest lacks strict_wave")?;
    if !valid_wave_id(wave) {
        bail!("manifest strict_wave is malformed");
    }
    Ok(wave.to_owned())
}

/// Per-task authority metric (R1/D2); must agree with
/// `jope_dime_regressor::task::TaskKind::primary_metric`.
fn expected_manifest_metric(task: &str) -> Option<&'static str> {
    match task {
        "regression" => Some("r2"),
        "binary" => Some("auc"),
        "timeseries" => Some("negative_mase"),
        "multiclass" => Some("macro_f1"),
        _ => None,
    }
}

fn validate_manifest(
    path: &Path,
    checkpoint_sha256: &str,
    epoch: &str,
    wave: &str,
) -> Result<ValidManifest> {
    let value = read_json(path)?;
    if value.get("schema").and_then(Value::as_str) != Some(MANIFEST_SCHEMA) {
        bail!("manifest schema is outside self-play");
    }
    if value
        .get("evaluation_lane")
        .and_then(Value::as_str)
        .is_some_and(|lane| {
            matches!(
                lane,
                "checkpoint_gate" | "sealed_audit" | "evaluator_canary"
            )
        })
    {
        bail!("protected evaluation lane is not adaptive admission");
    }
    if value.get("strict_wave").and_then(Value::as_str) != Some(wave)
        || value.get("checkpoint_epoch").and_then(Value::as_str) != Some(epoch)
        || value.get("checkpoint_sha256").and_then(Value::as_str) != Some(checkpoint_sha256)
    {
        bail!("manifest is not from the newest current-checkpoint wave");
    }
    for field in [
        "checkpoint_sha256",
        "catalog_sha256",
        "protocol_sha256",
        "train_sha256",
        "test_sha256",
    ] {
        if !value
            .get(field)
            .and_then(Value::as_str)
            .is_some_and(is_sha256)
        {
            bail!("manifest lacks immutable {field}");
        }
    }
    if protocol_for_manifest(&value)? != value["protocol_sha256"].as_str().unwrap_or_default() {
        bail!("manifest evaluator protocol hash drift");
    }
    let generated_at_ms = value
        .get("generated_at_ms")
        .and_then(Value::as_u64)
        .context("manifest lacks generation timestamp")? as u128;
    let task = value
        .get("task")
        .and_then(Value::as_str)
        .filter(|task| matches!(*task, "regression" | "binary" | "timeseries" | "multiclass"))
        .context("manifest task is invalid")?;
    let dataset_hash = value
        .get("dataset_hash")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .context("manifest dataset identity is absent")?;
    if !matches!(
        value.get("micro_lane").and_then(Value::as_str),
        Some("stable_anchor" | "least_observed" | "hard_uncertain")
    ) {
        bail!("manifest is not assigned to a deterministic micro-wave lane");
    }
    let authority = value
        .get("label_authority")
        .context("manifest authority is absent")?;
    let expected_metric = expected_manifest_metric(task).expect("task filtered above");
    if authority.get("score_tier").and_then(Value::as_str) != Some("test_exact")
        || authority.get("metric").and_then(Value::as_str) != Some(expected_metric)
        || authority.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || authority.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || authority.get("generation").and_then(Value::as_str) != Some(wave)
        || authority
            .get("model_checkpoint_sha256")
            .and_then(Value::as_str)
            != Some(checkpoint_sha256)
    {
        bail!("manifest authority is not adaptive test-exact");
    }
    let protocol = value["protocol_sha256"]
        .as_str()
        .context("protocol hash absent")?;
    let candidates = value
        .get("candidates")
        .and_then(Value::as_array)
        .filter(|rows| !rows.is_empty() && rows.len() <= 100)
        .context("manifest candidate count is outside 1..=100")?;
    let mut semantics = BTreeSet::new();
    let mut pairs = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let genome = candidate
            .get("canonical_genome")
            .and_then(Value::as_str)
            .context("candidate genome is absent")?;
        let semantic = candidate
            .get("semantic_sha256")
            .and_then(Value::as_str)
            .context("candidate semantic hash is absent")?;
        if sha256_bytes(genome.as_bytes()) != semantic {
            bail!("candidate semantic hash drift");
        }
        if !semantics.insert(semantic.to_owned()) {
            bail!("manifest duplicates a semantic candidate");
        }
        let is_dl_proposal = candidate.get("slot_group").and_then(Value::as_str)
            == Some("dl_proposal")
            || candidate.get("source").and_then(Value::as_str)
                == Some("current_checkpoint_dl_proposal");
        if is_dl_proposal
            && candidate
                .get("model_checkpoint_sha256")
                .and_then(Value::as_str)
                != Some(checkpoint_sha256)
        {
            bail!("DL proposal checkpoint provenance drift");
        }
        pairs.push(PairKey {
            task: task.to_owned(),
            dataset_hash: dataset_hash.to_owned(),
            semantic_sha256: semantic.to_owned(),
            protocol_sha256: protocol.to_owned(),
        });
    }
    let sha256 = sha256_file(path)?;
    Ok(ValidManifest {
        path: path.to_owned(),
        sha256,
        generated_at_ms,
        pairs,
        value,
    })
}

fn load_terminal_seen(connection: &Connection) -> Result<HashSet<String>> {
    let mut statement = connection.prepare("SELECT job_id FROM terminal_jobs")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows.collect::<rusqlite::Result<HashSet<_>>>()?)
}

fn record_burn_rate(
    connection: &mut Connection,
    jobs: &[SchedulerJob],
    terminal_seen: &mut HashSet<String>,
) -> Result<Vec<f64>> {
    let timestamp = now_ms() as i64;
    let terminal = jobs
        .iter()
        .filter(|job| job.project == PROJECT && TERMINAL_STATES.contains(&job.state.as_str()))
        .filter(|job| !terminal_seen.contains(&job.id))
        .map(|job| job.id.clone())
        .collect::<Vec<_>>();
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let mut new = 0_i64;
    for job_id in &terminal {
        new += tx.execute(
            "INSERT OR IGNORE INTO terminal_jobs(job_id,observed_at_ms) VALUES(?,?)",
            params![job_id, timestamp],
        )? as i64;
    }
    let previous: Option<i64> =
        tx.query_row("SELECT MAX(observed_at_ms) FROM burn_samples", [], |row| {
            row.get(0)
        })?;
    let elapsed_minutes = previous
        .map(|prior| ((timestamp - prior).max(5_000) as f64) / 60_000.0)
        .unwrap_or(1.0);
    let rate = if previous.is_some() {
        new as f64 / elapsed_minutes
    } else {
        0.0
    };
    tx.execute(
        "INSERT OR REPLACE INTO burn_samples(observed_at_ms,jobs_per_minute) VALUES(?,?)",
        params![timestamp, rate],
    )?;
    tx.execute(
        "DELETE FROM burn_samples WHERE observed_at_ms<?",
        [timestamp - 60 * 60_000],
    )?;
    let values = {
        let mut statement = tx.prepare(
            "SELECT jobs_per_minute FROM burn_samples \
             WHERE observed_at_ms>=? ORDER BY observed_at_ms",
        )?;
        let rows = statement.query_map([timestamp - 60 * 60_000], |row| row.get::<_, f64>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    tx.commit()?;
    terminal_seen.extend(terminal);
    Ok(values)
}

fn append_keeper_attempts(
    tx: &Transaction<'_>,
    manifest: &ValidManifest,
    owner: &str,
    job_id: &str,
    state: &str,
    initial_only: i64,
) -> Result<()> {
    // New self-play identities overwhelmingly begin at attempt one. Trying
    // that insert first avoids two indexed reads for every pair in a wave.
    // `OR IGNORE` is safe here because both uniqueness constraints remain the
    // authorities; only the uncommon retry/idempotency path needs the reads.
    let mut first_attempt_statement = tx.prepare_cached(
        "INSERT OR IGNORE INTO keeper_attempts(task,dataset_hash,semantic_sha256,\
         protocol_sha256,attempt_number,state,owner,manifest_path,manifest_sha256,\
         job_id,created_at_ms) VALUES(?,?,?,?,1,?,?,?,?,?,?)",
    )?;
    let mut active_statement = tx.prepare_cached(
        "SELECT manifest_sha256,job_id FROM keeper_attempts WHERE task=? AND dataset_hash=? \
         AND semantic_sha256=? AND protocol_sha256=? \
         AND state IN ('checked','submitting','submitted','deferred_horizon')",
    )?;
    let mut attempt_statement = tx.prepare_cached(
        "SELECT COALESCE(MAX(attempt_number),0)+1 FROM keeper_attempts \
         WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=?",
    )?;
    let mut insert_statement = tx.prepare_cached(
        "INSERT INTO keeper_attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
         attempt_number,state,owner,manifest_path,manifest_sha256,job_id,created_at_ms) \
         VALUES(?,?,?,?,?,?,?,?,?,?,?)",
    )?;
    for pair in &manifest.pairs {
        let inserted = first_attempt_statement.execute(params![
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            state,
            owner,
            manifest.path.to_string_lossy(),
            manifest.sha256,
            job_id,
            now_ms() as i64
        ])?;
        if inserted == 1 {
            continue;
        }
        let active: Option<(String, String)> = active_statement
            .query_row(
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((active_manifest, active_job)) = active {
            if initial_only != 0 {
                if active_manifest != manifest.sha256 || active_job != job_id {
                    bail!("protocol pair has a conflicting active queue attempt");
                }
                continue;
            }
            bail!("protocol pair already has an active queue attempt");
        }
        let attempt: i64 = attempt_statement.query_row(
            params![
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256
            ],
            |row| row.get(0),
        )?;
        insert_statement.execute(params![
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            attempt,
            state,
            owner,
            manifest.path.to_string_lossy(),
            manifest.sha256,
            job_id,
            now_ms() as i64
        ])?;
    }
    Ok(())
}

fn update_native_attempts_submitted(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    owner: &str,
    job_id: &str,
) -> Result<()> {
    let Some(attempts) = &schema.attempts else {
        return Ok(());
    };
    let select = format!(
        "SELECT {},state,{},manifest_sha256,job_id FROM attempts \
         WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
           AND state IN ('reserved','submitted') ORDER BY {} DESC",
        attempts.number_column, attempts.owner_column, attempts.number_column
    );
    let update = format!(
        "UPDATE attempts SET state='submitted',job_id=? \
         WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
           AND {}=? AND {}=? AND manifest_sha256=? AND state='reserved'",
        attempts.number_column, attempts.owner_column
    );
    let mut select_statement = tx.prepare_cached(&select)?;
    let mut update_statement = tx.prepare_cached(&update)?;
    for pair in &manifest.pairs {
        let rows = select_statement
            .query_map(
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                },
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if rows.len() != 1 {
            bail!(
                "native attempt conflict for {}:{}:{}: expected one active attempt, found {}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                rows.len()
            );
        }
        let (attempt_number, state, attempt_owner, attempt_manifest, attempt_job) = &rows[0];
        if attempt_owner != owner || attempt_manifest.as_deref() != Some(manifest.sha256.as_str()) {
            bail!(
                "native attempt reservation conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        if state == "submitted" {
            if attempt_job.as_deref() == Some(job_id) {
                continue;
            }
            bail!(
                "native attempt job conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        let changed = update_statement.execute(params![
            job_id,
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            attempt_number,
            owner,
            manifest.sha256,
        ])?;
        if changed != 1 {
            bail!(
                "native attempt changed during exact submission transition for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
    }
    Ok(())
}

fn ensure_native_reserved_attempts(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    owner: &str,
    job_id: &str,
) -> Result<()> {
    let Some(attempts) = &schema.attempts else {
        return Ok(());
    };
    let active_query = format!(
        "SELECT {},state,{},manifest_sha256,job_id FROM attempts \
         WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
           AND state IN ('reserved','submitted') ORDER BY {} DESC",
        attempts.number_column, attempts.owner_column, attempts.number_column
    );
    let next_query = format!(
        "SELECT COALESCE(MAX({}),0)+1 FROM attempts WHERE task=? AND dataset_hash=? \
         AND semantic_sha256=? AND protocol_sha256=?",
        attempts.number_column
    );
    let insert = format!(
        "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
         {},state,{},manifest_path,manifest_sha256,job_id,created_at_ms) \
         VALUES(?,?,?,?,?,'reserved',?,?,?,NULL,?)",
        attempts.number_column, attempts.owner_column
    );
    let mut active_statement = tx.prepare_cached(&active_query)?;
    let mut next_statement = tx.prepare_cached(&next_query)?;
    let mut insert_statement = tx.prepare_cached(&insert)?;
    for pair in &manifest.pairs {
        let rows = active_statement
            .query_map(
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                },
            )?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if rows.len() > 1 {
            bail!(
                "native attempt conflict for {}:{}:{}: found multiple active attempts",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        if let Some((_, state, attempt_owner, attempt_manifest, attempt_job)) = rows.first() {
            ensure!(
                matches!(state.as_str(), "reserved" | "submitted")
                    && attempt_owner == owner
                    && attempt_manifest.as_deref() == Some(manifest.sha256.as_str())
                    && (state == "reserved" || attempt_job.as_deref() == Some(job_id)),
                "native attempt reservation conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
            continue;
        }
        // Migration repair for reservations created before the native attempt
        // ledger was wired into queue admission. The pair row already pins the
        // exact owner and manifest; this transaction supplies only the missing
        // initial attempt before submission.
        let attempt_number: i64 = next_statement.query_row(
            params![
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256
            ],
            |row| row.get(0),
        )?;
        insert_statement.execute(params![
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            attempt_number,
            owner,
            manifest.path.to_string_lossy(),
            manifest.sha256,
            now_ms() as i64
        ])?;
    }
    Ok(())
}

fn update_pairs_submitted(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    owner: &str,
    job_id: &str,
) -> Result<()> {
    let select = format!(
        "SELECT state,{},manifest_sha256,job_id FROM pairs \
         WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=?",
        schema.owner_column
    );
    let update = format!(
        "UPDATE pairs SET state='submitted',job_id=?,updated_at_ms=? \
         WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=? \
           AND {}=? AND manifest_sha256=? \
           AND state IN ('reserved','infrastructure_retry')",
        schema.owner_column
    );
    let mut select_statement = tx.prepare_cached(&select)?;
    let mut update_statement = tx.prepare_cached(&update)?;
    for pair in &manifest.pairs {
        let row: Option<(String, String, Option<String>, Option<String>)> = select_statement
            .query_row(
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        let Some((state, pair_owner, pair_manifest, pair_job)) = row else {
            bail!(
                "submission pair is absent for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        };
        if pair_owner != owner || pair_manifest.as_deref() != Some(manifest.sha256.as_str()) {
            bail!(
                "submission pair reservation conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        if state == "submitted" {
            if pair_job.as_deref() == Some(job_id) {
                continue;
            }
            bail!(
                "submission pair job conflict for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        if !matches!(state.as_str(), "reserved" | "infrastructure_retry") {
            bail!(
                "submission pair is not launch-ready for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
        let changed = update_statement.execute(params![
            job_id,
            now_ms() as i64,
            pair.task,
            pair.dataset_hash,
            pair.semantic_sha256,
            pair.protocol_sha256,
            owner,
            manifest.sha256,
        ])?;
        if changed != 1 {
            bail!(
                "submission pair changed during exact transition for {}:{}:{}",
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256
            );
        }
    }
    Ok(())
}

fn finalize_native_attempts_for_retry(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    failure: &str,
) -> Result<()> {
    let Some(_) = &schema.attempts else {
        return Ok(());
    };
    let columns = table_columns(tx, "attempts")?;
    let finalized = if columns.contains("finalized_at_ms") {
        ",finalized_at_ms=?"
    } else {
        ""
    };
    let query = format!(
        "UPDATE attempts SET state='infrastructure_retry',failure_class=?{finalized} \
         WHERE manifest_sha256=? AND state='submitted'"
    );
    let changed = if finalized.is_empty() {
        tx.execute(&query, params![failure, manifest.sha256])?
    } else {
        tx.execute(&query, params![failure, now_ms() as i64, manifest.sha256])?
    };
    if changed != manifest.pairs.len() {
        let prior: i64 = tx.query_row(
            "SELECT COUNT(*) FROM attempts WHERE manifest_sha256=? \
             AND state='infrastructure_retry'",
            [manifest.sha256.as_str()],
            |row| row.get(0),
        )?;
        if changed != 0 || prior < manifest.pairs.len() as i64 {
            bail!("native attempt infrastructure retry transition count mismatch");
        }
    }
    Ok(())
}

fn append_native_retry_attempts(
    tx: &Transaction<'_>,
    schema: &PairSchema,
    manifest: &ValidManifest,
    owner: &str,
    new_job_id: &str,
) -> Result<()> {
    let Some(attempts) = &schema.attempts else {
        return Ok(());
    };
    for pair in &manifest.pairs {
        let active_job: Option<String> = tx
            .query_row(
                "SELECT job_id FROM attempts WHERE task=? AND dataset_hash=? \
                 AND semantic_sha256=? AND protocol_sha256=? AND state='reserved'",
                params![
                    pair.task,
                    pair.dataset_hash,
                    pair.semantic_sha256,
                    pair.protocol_sha256
                ],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(job_id) = active_job {
            if job_id != new_job_id {
                bail!("native retry attempt is bound to a different job");
            }
            continue;
        }
        let attempt_number: i64 = tx.query_row(
            &format!(
                "SELECT COALESCE(MAX({}),0)+1 FROM attempts WHERE task=? \
                 AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=?",
                attempts.number_column
            ),
            params![
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256
            ],
            |row| row.get(0),
        )?;
        tx.execute(
            &format!(
                "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
                 {},state,{},manifest_path,manifest_sha256,job_id,created_at_ms) \
                 VALUES(?,?,?,?,?,'reserved',?,?,?,?,?)",
                attempts.number_column, attempts.owner_column
            ),
            params![
                pair.task,
                pair.dataset_hash,
                pair.semantic_sha256,
                pair.protocol_sha256,
                attempt_number,
                owner,
                manifest.path.to_string_lossy(),
                manifest.sha256,
                new_job_id,
                now_ms() as i64
            ],
        )?;
    }
    Ok(())
}

fn validate_with_evaluator(evaluator: &Path, manifest: &Path) -> Result<()> {
    let output = Command::new(evaluator)
        .args(["validate-manifest", "--manifest"])
        .arg(manifest)
        .output()
        .with_context(|| format!("run Rust evaluator {}", evaluator.display()))?;
    if !output.status.success() {
        bail!(
            "Rust evaluator rejected {}: {}",
            manifest.display(),
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    Ok(())
}

fn evaluator_failure_is_permanent(error: &anyhow::Error) -> bool {
    format!("{error:#}").contains("Rust evaluator rejected")
}

fn render_spec(
    path: &Path,
    job_id: &str,
    manifest: &ValidManifest,
    runner: &Path,
    evaluator: &Path,
) -> Result<()> {
    let train = path_field(&manifest.value, "train_csv")?;
    let test = path_field(&manifest.value, "test_csv")?;
    let inputs = [
        ("runner", runner, "runner.sh", sha256_file(runner)?),
        (
            "manifest",
            manifest.path.as_path(),
            "manifest.json",
            manifest.sha256.clone(),
        ),
        (
            "train",
            train.as_path(),
            "train.csv",
            string_field(&manifest.value, "train_sha256")?.to_owned(),
        ),
        (
            "test",
            test.as_path(),
            "test.csv",
            string_field(&manifest.value, "test_sha256")?.to_owned(),
        ),
        (
            "evaluator",
            evaluator,
            "corpus_eval",
            sha256_file(evaluator)?,
        ),
    ];
    for (_, source, _, digest) in &inputs {
        if !source.is_file() || sha256_file(source)? != *digest {
            bail!("spec input content pin drift: {}", source.display());
        }
    }
    let quoted = |value: &str| serde_json::to_string(value).expect("string JSON");
    let mut text = format!(
        "[job]\nid = {}\nproject = \"{}\"\ntier = \"normal\"\n\
         command = \"/usr/bin/env\"\n\
         args = [\"{{input:runner}}\", \"{{input:manifest}}\", \"{{input:train}}\", \
         \"{{input:test}}\", \"{{input:evaluator}}\", \"{{output:receipt}}\"]\n\n\
         [resource]\ngpus_per_task = 0\ncpus_reserved = 1\nomp_threads = 1\n\
         ram_mb = 512\ngang_size = 1\nplacement = \"single_node\"\n\n\
         [env]\nRAYON_NUM_THREADS = \"1\"\nOMP_NUM_THREADS = \"1\"\n\
         OPENBLAS_NUM_THREADS = \"1\"\nMKL_NUM_THREADS = \"1\"\n\
         NUMEXPR_NUM_THREADS = \"1\"\n\n\
         [preempt]\npreemptible = true\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n\
         [limits]\nmax_runtime_hours = 24\nmax_restarts = 1\n\n",
        quoted(job_id),
        PROJECT
    );
    for (name, source, target, digest) in inputs {
        text.push_str("[[input]]\n");
        text.push_str(&format!("name = {}\n", quoted(name)));
        text.push_str(&format!(
            "source = {}\n",
            quoted(source.to_string_lossy().as_ref())
        ));
        text.push_str(&format!("target = {}\n", quoted(target)));
        text.push_str(&format!("sha256 = {}\n\n", quoted(&digest)));
    }
    text.push_str("[[output]]\nname = \"receipt\"\nsource = \"receipt\"\ndirectory = true\n");
    atomic_write(path, text.as_bytes())
}

/// How many guard-held jobs one cycle may release. Deliberately bounded so a
/// 5-second cycle cannot flood a recovering host: draining 996 held jobs (the
/// 2026-08-01 pileup) at 64/cycle takes ~80 s, while the overload guard gets a
/// fresh look between batches. Release only fills the gap left under the
/// queue target — a full queue releases nothing.
const HELD_RELEASE_BATCH_MAX: usize = 64;

fn held_release_budget(queued_starting: usize, queued_target: i64, held: usize) -> usize {
    let target = queued_target.max(0) as usize;
    target
        .saturating_sub(queued_starting)
        .min(held)
        .min(HELD_RELEASE_BATCH_MAX)
}

fn scq_release(scq: &Path, job_id: &str) -> Result<()> {
    let output = Command::new(scq)
        .arg("release")
        .arg(job_id)
        .output()
        .with_context(|| format!("run SCQ release {job_id}"))?;
    if !output.status.success() {
        bail!(
            "SCQ refused release of {job_id}: {}",
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    Ok(())
}

fn scq_check(scq: &Path, spec: &Path) -> Result<()> {
    let output = Command::new(scq)
        .arg("check")
        .arg(spec)
        .output()
        .with_context(|| format!("run SCQ check {}", spec.display()))?;
    if !output.status.success() {
        bail!(
            "SCQ rejected {}: {}",
            spec.display(),
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    Ok(())
}

fn rewrite_spec_job_id(old: &Path, new: &Path, old_id: &str, new_id: &str) -> Result<()> {
    let text = fs::read_to_string(old)?;
    let old_line = format!("id = {}", serde_json::to_string(old_id)?);
    let new_line = format!("id = {}", serde_json::to_string(new_id)?);
    if !text.contains(&old_line) {
        bail!("retry source spec does not bind its claimed job id");
    }
    atomic_write(new, text.replacen(&old_line, &new_line, 1).as_bytes())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("write target has no filename")?;
    let temporary = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let mut file = File::create(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn authoritative_receipt_exists(root: &Path, job_id: &str) -> bool {
    root.join("receipts")
        .join(job_id)
        .join("SHARD_OK")
        .is_file()
}

fn reject_python_runtime(runner: &Path) -> Result<()> {
    let content = fs::read_to_string(runner)
        .with_context(|| format!("read evaluator runner {}", runner.display()))?;
    let lowered = content.to_ascii_lowercase();
    if ["python", "python3", ".py "]
        .iter()
        .any(|marker| lowered.contains(marker))
    {
        bail!("evaluator runner invokes Python");
    }
    Ok(())
}

fn path_field(value: &Value, key: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(string_field(value, key)?))
}

fn string_field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("manifest lacks {key}"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_infrastructure_failure_class(value: &str) -> bool {
    INFRA_FAILURES.contains(&value.to_ascii_lowercase().as_str())
}

fn is_transient_cli_error(detail: &str) -> bool {
    let detail = detail.to_ascii_lowercase();
    [
        "controller dropped request",
        "connection reset",
        "connection refused",
        "broken pipe",
        "timed out",
        "socket",
    ]
    .iter()
    .any(|marker| detail.contains(marker))
}

fn submit_batch(scq: &Path, claims: &[ReadyClaim]) -> Result<Vec<(bool, bool, String)>> {
    ensure!(
        !claims.is_empty() && claims.len() <= 1_024,
        "SCQ bulk submission requires 1..=1024 claims"
    );
    let output = Command::new(scq)
        .arg("--json")
        .arg("submit-batch")
        .args(claims.iter().map(|row| row.claim.spec.as_path()))
        .output()
        .with_context(|| format!("run SCQ bulk submit with {} specs", claims.len()))?;
    let value: Value = match serde_json::from_slice(&output.stdout) {
        Ok(value) => value,
        Err(error) => {
            let detail = format!(
                "SCQ bulk response parse failed: {error}; stdout={}; stderr={}",
                String::from_utf8_lossy(&output.stdout).trim(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
            return Ok((0..claims.len())
                .map(|_| (false, true, detail.clone()))
                .collect());
        }
    };
    let results = value
        .get("BatchSubmitted")
        .and_then(|value| value.get("results"))
        .and_then(Value::as_array);
    let Some(results) = results else {
        let detail = format!("SCQ rejected bulk request: {value}");
        return Ok((0..claims.len())
            .map(|_| (false, true, detail.clone()))
            .collect());
    };
    ensure!(
        results.len() == claims.len(),
        "SCQ bulk response count mismatch"
    );
    let mut outcomes = Vec::with_capacity(results.len());
    for (index, (result, claim)) in results.iter().zip(claims).enumerate() {
        ensure!(
            result.get("index").and_then(Value::as_u64) == Some(index as u64),
            "SCQ bulk response ordering mismatch"
        );
        let submitted = result
            .get("submitted")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if submitted {
            ensure!(
                result.get("id").and_then(Value::as_str) == Some(claim.claim.job_id.as_str()),
                "SCQ bulk response job identity mismatch"
            );
        }
        let detail = result
            .get("error")
            .and_then(Value::as_str)
            .or_else(|| result.get("code").and_then(Value::as_str))
            .unwrap_or(if submitted { "submitted" } else { "rejected" })
            .to_owned();
        let lowered = detail.to_ascii_lowercase();
        let accepted =
            submitted || lowered.contains("already exists") || lowered.contains("already active");
        outcomes.push((
            accepted,
            !accepted && is_transient_cli_error(&detail),
            detail,
        ));
    }
    Ok(outcomes)
}

fn parse_rfc3339_seconds(value: &str) -> Option<i64> {
    // SCQ emits UTC RFC3339 timestamps.  Parsing them locally avoids pulling
    // a high-level time runtime into the always-on keeper.
    let year = value.get(0..4)?.parse::<i64>().ok()?;
    let month = value.get(5..7)?.parse::<i64>().ok()?;
    let day = value.get(8..10)?.parse::<i64>().ok()?;
    let hour = value.get(11..13)?.parse::<i64>().ok()?;
    let minute = value.get(14..16)?.parse::<i64>().ok()?;
    let second = value.get(17..19)?.parse::<i64>().ok()?;
    let timezone = value.find('Z').map(|_| 0).or_else(|| {
        let index = value.get(19..)?.find(['+', '-'])? + 19;
        let sign = if value.as_bytes()[index] == b'+' {
            1
        } else {
            -1
        };
        let hours = value.get(index + 1..index + 3)?.parse::<i64>().ok()?;
        let minutes = value.get(index + 4..index + 6)?.parse::<i64>().ok()?;
        Some(sign * (hours * 3_600 + minutes * 60))
    })?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let days = days_from_civil(year, month, day)?;
    Some(days * 86_400 + hour * 3_600 + minute * 60 + second - timezone)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let adjusted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some(era * 146_097 + day_of_era - 719_468)
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
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            bail!("another Rust queue keeper owns {}", path.display());
        }
        Ok(Self { file })
    }
}

impl Drop for SingleInstanceLock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn manifest_metric_map_agrees_with_task_authority() {
        use jope_dime_regressor::task::TaskKind;
        for kind in [
            TaskKind::Regression,
            TaskKind::Binary,
            TaskKind::Timeseries,
            TaskKind::Multiclass,
        ] {
            assert_eq!(
                expected_manifest_metric(kind.as_str()).unwrap(),
                kind.primary_metric()
            );
        }
        assert_eq!(expected_manifest_metric("multi_label"), None);
    }

    fn checkpoint() -> String {
        "a".repeat(64)
    }

    fn manifest_value(wave: &str, epoch: &str) -> Value {
        let checkpoint = checkpoint();
        let genome = "(pipe X (ridge))";
        let train_sha256 = "d".repeat(64);
        let test_sha256 = "e".repeat(64);
        let protocol_sha256 = jope_dime_regressor::selfplay_runtime::evaluation_protocol_sha256(
            "regression",
            "r2",
            "test_exact",
            &train_sha256,
            &test_sha256,
        )
        .unwrap();
        json!({
            "schema": MANIFEST_SCHEMA,
            "strict_wave": wave,
            "task": "regression",
            "dataset_hash": "dataset",
            "checkpoint_epoch": epoch,
            "checkpoint_sha256": checkpoint,
            "catalog_sha256": "b".repeat(64),
            "protocol_sha256": protocol_sha256,
            "generated_at_ms": 10,
            "micro_lane": "least_observed",
            "train_csv": "/train.csv",
            "test_csv": "/test.csv",
            "train_sha256": train_sha256,
            "test_sha256": test_sha256,
            "label_authority": {
                "score_tier": "test_exact",
                "metric": "r2",
                "test_adaptive": true,
                "benchmark_valid": false,
                "generation": wave,
                "model_checkpoint_sha256": checkpoint,
            },
            "candidates": [{
                "canonical_genome": genome,
                "semantic_sha256": sha256_bytes(genome.as_bytes()),
                "model_checkpoint_sha256": checkpoint,
            }]
        })
    }

    #[test]
    fn queue_policy_is_fixed_and_held_is_not_a_queue_state() {
        let targets = dynamic_queue_targets(&[20.0, 30.0, 40.0], 200).unwrap();
        assert_eq!(targets.target, 2_048);
        assert_eq!(targets.hard_cap, 4_096);
        assert!(QUEUE_STATES.contains(&"queued"));
        assert!(QUEUE_STATES.contains(&"starting"));
        assert!(!QUEUE_STATES.contains(&"held"));
        assert!(!QUEUE_STATES.contains(&"running"));
    }

    #[test]
    fn held_release_only_fills_the_gap_under_target() {
        // The 2026-08-01 pileup: 996 held, queue empty, target 2048 — release
        // a full batch, never more than the batch cap.
        assert_eq!(held_release_budget(0, 2_048, 996), HELD_RELEASE_BATCH_MAX);
        // Queue already at/over target: a busy cluster releases nothing, so
        // the overload guard's decision is respected while pressure is real.
        assert_eq!(held_release_budget(2_048, 2_048, 996), 0);
        assert_eq!(held_release_budget(3_000, 2_048, 996), 0);
        // Fewer held than gap: release exactly what exists.
        assert_eq!(held_release_budget(2_040, 2_048, 3), 3);
        // Gap smaller than batch cap: release exactly the gap.
        assert_eq!(held_release_budget(2_000, 2_048, 996), 48);
        // A zero/negative target (fail-closed cycles) releases nothing.
        assert_eq!(held_release_budget(0, 0, 996), 0);
        assert_eq!(held_release_budget(0, -5, 996), 0);
    }

    #[test]
    fn failure_report_preserves_the_queue_status_envelope() {
        let value = serde_json::to_value(failure_report(true, "boom")).unwrap();
        assert_eq!(value["schema"], "JopeSelfplayQueueKeeper.v3");
        assert_eq!(value["admission_allowed"], false);
        assert_eq!(value["degraded"], true);
        assert_eq!(value["error"], "boom");
        for field in [
            "queued_target",
            "queued_hard_cap",
            "queued_starting",
            "dispatching",
            "admission_waves",
            "unclaimed_current_checkpoint_manifests",
        ] {
            assert!(value.get(field).is_some(), "missing {field}");
        }
    }

    #[test]
    fn backlog_initialization_does_not_rebuild_the_active_attempt_index() {
        let connection = Connection::open_in_memory().unwrap();
        initialize_backlog(&connection).unwrap();
        let before: i64 = connection
            .query_row("PRAGMA schema_version", [], |row| row.get(0))
            .unwrap();
        initialize_backlog(&connection).unwrap();
        let after: i64 = connection
            .query_row("PRAGMA schema_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(after, before);
        let sql: String = connection
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='keeper_one_active_attempt'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(sql.contains("deferred_horizon"));
    }

    #[test]
    fn manifest_validation_is_bounded_to_the_next_admission_batch() {
        assert_eq!(bounded_validation_budget(1_408, 512), 576);
        assert_eq!(bounded_validation_budget(40, 512), 40);
        assert_eq!(bounded_validation_budget(100, 1), 2);
        assert_eq!(bounded_validation_budget(100, 0), 0);
        assert!(evaluator_failure_is_permanent(&anyhow::anyhow!(
            "Rust evaluator rejected manifest"
        )));
        assert!(!evaluator_failure_is_permanent(&anyhow::anyhow!(
            "run Rust evaluator: resource temporarily unavailable"
        )));
    }

    #[test]
    fn admission_waves_are_oldest_first_and_exclude_claimed_manifests() {
        let root = tempdir().unwrap();
        let first = root
            .path()
            .join("waves/strict-wave-000001/manifests/first.json");
        let second = root
            .path()
            .join("waves/strict-wave-000002/manifests/second.json");
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        fs::create_dir_all(second.parent().unwrap()).unwrap();
        fs::write(&first, b"{}").unwrap();
        fs::write(&second, b"{}").unwrap();
        let waves = vec![
            MicroWave {
                wave: "strict-wave-000001".into(),
                summary_sha256: "a".repeat(64),
            },
            MicroWave {
                wave: "strict-wave-000002".into(),
                summary_sha256: "b".repeat(64),
            },
        ];
        let mut claims = HashMap::new();
        claims.insert(
            "claimed".into(),
            Claim {
                manifest_sha256: "c".repeat(64),
                manifest: first,
                job_id: "jte-claimed".into(),
                spec: root.path().join("claimed.toml"),
                state: "submitted".into(),
                attempts: 1,
                owner: Some("owner".into()),
            },
        );
        let admission =
            current_admission_waves(root.path(), &waves, &claims, &BTreeSet::new()).unwrap();
        assert_eq!(admission.len(), 1);
        assert_eq!(admission[0].wave, "strict-wave-000002");
        assert_eq!(admission[0].unclaimed_manifests, 1);
        let rejected = BTreeSet::from([second]);
        assert!(
            current_admission_waves(root.path(), &waves, &claims, &rejected)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn filters_stale_epoch_old_wave_and_protected_lane() {
        let root = tempdir().unwrap();
        let path = root.path().join("manifest.json");
        let current_epoch = format!("checkpoint-{}", checkpoint());
        atomic_json(&path, &manifest_value("strict-wave-000009", &current_epoch)).unwrap();
        assert!(
            validate_manifest(&path, &checkpoint(), &current_epoch, "strict-wave-000009").is_ok()
        );
        assert!(validate_manifest(
            &path,
            &checkpoint(),
            "checkpoint-stale",
            "strict-wave-000009"
        )
        .is_err());
        assert!(
            validate_manifest(&path, &checkpoint(), &current_epoch, "strict-wave-000008").is_err()
        );
        let mut protected = manifest_value("strict-wave-000009", &current_epoch);
        protected["evaluation_lane"] = json!("sealed_audit");
        atomic_json(&path, &protected).unwrap();
        assert!(
            validate_manifest(&path, &checkpoint(), &current_epoch, "strict-wave-000009").is_err()
        );
    }

    #[test]
    fn restart_submitting_claim_is_idempotently_adoptable() {
        let root = tempdir().unwrap();
        let connection = Connection::open(root.path().join("backlog.sqlite")).unwrap();
        initialize_backlog(&connection).unwrap();
        let genome = "(pipe X (ridge))";
        let manifest = ValidManifest {
            path: root.path().join("manifest.json"),
            sha256: "a".repeat(64),
            generated_at_ms: 1,
            pairs: vec![PairKey {
                task: "regression".into(),
                dataset_hash: "dataset".into(),
                semantic_sha256: sha256_bytes(genome.as_bytes()),
                protocol_sha256: "b".repeat(64),
            }],
            value: json!({}),
        };
        for _ in 0..2 {
            let tx = connection.unchecked_transaction().unwrap();
            append_keeper_attempts(&tx, &manifest, "wave", "jte-a", "submitting", 1).unwrap();
            tx.commit().unwrap();
        }
        let attempts: i64 = connection
            .query_row("SELECT COUNT(*) FROM keeper_attempts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(attempts, 1);
        let live = SchedulerJob {
            id: "jte-a".into(),
            project: PROJECT.into(),
            state: "queued".into(),
            failure_class: None,
            queue_ts: None,
        };
        assert_eq!(live.project, PROJECT);
        assert!(ACTIVE_JOB_STATES.contains(&live.state.as_str()));
        assert!(scheduler_job_is_adoptable(&live));
        let completed = SchedulerJob {
            state: "completed".into(),
            ..live
        };
        assert!(scheduler_job_is_adoptable(&completed));
    }

    fn submission_transition_fixture(
        number_column: &'static str,
        owner_column: &'static str,
    ) -> (Connection, PairSchema, ValidManifest, String) {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(&format!(
                "CREATE TABLE pairs(
                   task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
                   semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
                   state TEXT NOT NULL,{owner_column} TEXT NOT NULL,
                   manifest_sha256 TEXT,job_id TEXT,updated_at_ms INTEGER NOT NULL,
                   PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256));
                 CREATE TABLE attempts(
                   id INTEGER PRIMARY KEY,task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
                   semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
                   {number_column} INTEGER NOT NULL,state TEXT NOT NULL,
                   {owner_column} TEXT NOT NULL,manifest_path TEXT,
                   manifest_sha256 TEXT,job_id TEXT,
                   created_at_ms INTEGER NOT NULL DEFAULT 0,
                   UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,{number_column}));"
            ))
            .unwrap();
        let owner = "queue-keeper:fixture".to_owned();
        let manifest_sha256 = "a".repeat(64);
        let protocol_sha256 = "b".repeat(64);
        let selected_semantic = "c".repeat(64);
        let unrelated_semantic = "d".repeat(64);
        for semantic in [&selected_semantic, &unrelated_semantic] {
            connection
                .execute(
                    &format!(
                        "INSERT INTO pairs(task,dataset_hash,semantic_sha256,protocol_sha256,\
                         state,{owner_column},manifest_sha256,updated_at_ms) \
                         VALUES('regression','dataset',?,?,'reserved',?,?,1)"
                    ),
                    params![semantic, protocol_sha256, owner, manifest_sha256],
                )
                .unwrap();
            connection
                .execute(
                    &format!(
                        "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
                         {number_column},state,{owner_column},manifest_sha256) \
                         VALUES('regression','dataset',?,?,1,'reserved',?,?)"
                    ),
                    params![semantic, protocol_sha256, owner, manifest_sha256],
                )
                .unwrap();
        }
        let manifest = ValidManifest {
            path: PathBuf::from("/manifest.json"),
            sha256: manifest_sha256,
            generated_at_ms: 1,
            pairs: vec![PairKey {
                task: "regression".into(),
                dataset_hash: "dataset".into(),
                semantic_sha256: selected_semantic,
                protocol_sha256,
            }],
            value: json!({}),
        };
        (
            connection,
            PairSchema {
                owner_column,
                has_infrastructure_retries: false,
                attempts: Some(AttemptSchema {
                    number_column,
                    owner_column,
                }),
            },
            manifest,
            owner,
        )
    }

    fn assert_exact_submission_transition(number_column: &'static str, owner_column: &'static str) {
        let (connection, schema, manifest, owner) =
            submission_transition_fixture(number_column, owner_column);
        for _ in 0..2 {
            let tx = connection.unchecked_transaction().unwrap();
            ensure_native_reserved_attempts(&tx, &schema, &manifest, &owner, "jte-fixture")
                .unwrap();
            update_pairs_submitted(&tx, &schema, &manifest, &owner, "jte-fixture").unwrap();
            update_native_attempts_submitted(&tx, &schema, &manifest, &owner, "jte-fixture")
                .unwrap();
            tx.commit().unwrap();
        }
        let selected_pair: (String, Option<String>) = connection
            .query_row(
                "SELECT state,job_id FROM pairs WHERE semantic_sha256=?",
                [manifest.pairs[0].semantic_sha256.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            selected_pair,
            ("submitted".into(), Some("jte-fixture".into()))
        );
        let unrelated_pair: (String, Option<String>) = connection
            .query_row(
                "SELECT state,job_id FROM pairs WHERE semantic_sha256=?",
                ["d".repeat(64)],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(unrelated_pair, ("reserved".into(), None));
        let selected_attempt: (String, Option<String>) = connection
            .query_row(
                "SELECT state,job_id FROM attempts WHERE semantic_sha256=?",
                [manifest.pairs[0].semantic_sha256.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            selected_attempt,
            ("submitted".into(), Some("jte-fixture".into()))
        );
        let unrelated_attempt: (String, Option<String>) = connection
            .query_row(
                "SELECT state,job_id FROM attempts WHERE semantic_sha256=?",
                ["d".repeat(64)],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(unrelated_attempt, ("reserved".into(), None));
    }

    #[test]
    fn exact_submission_transition_is_idempotent_on_live_attempt_no_schema() {
        assert_exact_submission_transition("attempt_no", "reservation_owner");
    }

    #[test]
    fn exact_submission_repairs_a_pre_attempt_ledger_reservation() {
        let (connection, schema, manifest, owner) =
            submission_transition_fixture("attempt_no", "reservation_owner");
        connection
            .execute(
                "DELETE FROM attempts WHERE semantic_sha256=?",
                [manifest.pairs[0].semantic_sha256.as_str()],
            )
            .unwrap();
        let tx = connection.unchecked_transaction().unwrap();
        ensure_native_reserved_attempts(&tx, &schema, &manifest, &owner, "jte-repaired").unwrap();
        update_pairs_submitted(&tx, &schema, &manifest, &owner, "jte-repaired").unwrap();
        update_native_attempts_submitted(&tx, &schema, &manifest, &owner, "jte-repaired").unwrap();
        tx.commit().unwrap();
        let row: (String, Option<String>, String) = connection
            .query_row(
                "SELECT state,job_id,reservation_owner FROM attempts \
                 WHERE semantic_sha256=?",
                [manifest.pairs[0].semantic_sha256.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            row,
            ("submitted".into(), Some("jte-repaired".into()), owner)
        );
    }

    #[test]
    fn exact_submission_transition_is_idempotent_on_legacy_attempt_number_schema() {
        assert_exact_submission_transition("attempt_number", "owner");
    }

    #[test]
    fn exact_submission_transition_rejects_a_different_job() {
        let (connection, schema, manifest, owner) =
            submission_transition_fixture("attempt_no", "reservation_owner");
        let tx = connection.unchecked_transaction().unwrap();
        update_pairs_submitted(&tx, &schema, &manifest, &owner, "jte-first").unwrap();
        update_native_attempts_submitted(&tx, &schema, &manifest, &owner, "jte-first").unwrap();
        tx.commit().unwrap();
        let tx = connection.unchecked_transaction().unwrap();
        assert!(
            update_pairs_submitted(&tx, &schema, &manifest, &owner, "jte-other")
                .unwrap_err()
                .to_string()
                .contains("job conflict")
        );
    }

    #[test]
    fn rfc3339_parser_handles_fractional_utc_and_offset() {
        assert_eq!(parse_rfc3339_seconds("1970-01-01T00:00:00.000Z"), Some(0));
        assert_eq!(parse_rfc3339_seconds("1970-01-01T01:00:00+01:00"), Some(0));
    }

    #[test]
    fn only_explicit_infrastructure_classes_retry() {
        assert!(is_infrastructure_failure_class("storage-error"));
        assert!(is_infrastructure_failure_class("RPC-ERROR"));
        assert!(!is_infrastructure_failure_class("task-invalid"));
        assert!(!is_infrastructure_failure_class("timeout"));
    }

    #[test]
    fn node_slots_use_live_schema_without_fabricating_xbabe4_multiplier() {
        let value = json!({
            "Nodes": {
                "nodes": [
                    {
                        "node": "xbabe4",
                        "cores": 24,
                        "effective_cores": 20,
                        "running_jobs": 3
                    },
                    {
                        "node": "xbabe5",
                        "cores": 64,
                        "effective_cores": 56,
                        "cpu_job_slot_multiplier": 1.5,
                        "job_slot_target": 84,
                        "running_job_slots": 80
                    }
                ]
            }
        });
        let nodes = parse_nodes(&value).unwrap();
        assert_eq!(nodes[0].cpu_job_slot_multiplier, 1.0);
        assert_eq!(nodes[0].slot_target, 20);
        assert_eq!(nodes[0].running_slots, 3);
        assert_eq!(nodes[1].cpu_job_slot_multiplier, 1.5);
        assert_eq!(nodes[1].slot_target, 84);
        assert_eq!(nodes[1].running_slots, 80);
    }
}
