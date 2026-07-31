//! Crash-safe admission for the three reserved self-play GPU lanes.

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, now_ms, read_json, scq_json, sha256_file, DEFAULT_ROOT, DEFAULT_SCQ,
};
use jope_dime_regressor::selfplay_shadow::{
    GpuAdmissionStage, GpuLaneAdmissionV1, GpuLaneRole, RecoveryDisposition, Sha256Digest,
    GPU_ADMISSION_RETRY_MS,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

const EVALUATOR_PROJECT: &str = "jope-selfplay-eval";
const SENTINEL_SCHEMA: &str = "GpuLanePauseSentinel.v1";

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RoleArg {
    CohortProposer,
    AlternatingTrainer,
    BreadthEncoder,
}

impl From<RoleArg> for GpuLaneRole {
    fn from(value: RoleArg) -> Self {
        match value {
            RoleArg::CohortProposer => Self::CohortProposer,
            RoleArg::AlternatingTrainer => Self::AlternatingTrainer,
            RoleArg::BreadthEncoder => Self::BreadthEncoder,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "jope-gpu-lane-admission",
    about = "Persist pause/hold/capacity/submit/release transactions for xbabe1/2/3"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    #[arg(long, default_value = DEFAULT_SCQ)]
    scq: PathBuf,
    #[arg(long)]
    once: bool,
    #[arg(
        long,
        default_value_t = (GPU_ADMISSION_RETRY_MS / 1_000) as u64,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    interval_seconds: u64,
    /// Register one immutable request and exit.
    #[arg(long, value_enum, requires_all = ["spec", "job_id"])]
    register_role: Option<RoleArg>,
    #[arg(long)]
    spec: Option<PathBuf>,
    #[arg(long)]
    job_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct SchedulerJob {
    id: String,
    project: String,
    state: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct NodeCapacity {
    node: String,
    up: bool,
    gpus_free: u64,
    free_core_budget: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PauseSentinel {
    schema: String,
    created_at_ms: u128,
    updated_at_ms: u128,
    roles: Vec<GpuLaneRole>,
    held_evaluator_ids: Vec<String>,
    released_evaluator_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hold_inflight_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    release_inflight_id: Option<String>,
}

impl PauseSentinel {
    fn new(roles: Vec<GpuLaneRole>) -> Self {
        let timestamp = now_ms();
        Self {
            schema: SENTINEL_SCHEMA.to_owned(),
            created_at_ms: timestamp,
            updated_at_ms: timestamp,
            roles,
            held_evaluator_ids: Vec::new(),
            released_evaluator_ids: Vec::new(),
            hold_inflight_id: None,
            release_inflight_id: None,
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SENTINEL_SCHEMA,
            "GPU pause sentinel schema is not v1"
        );
        let held = self.held_evaluator_ids.iter().collect::<BTreeSet<_>>();
        let released = self.released_evaluator_ids.iter().collect::<BTreeSet<_>>();
        ensure!(
            held.len() == self.held_evaluator_ids.len(),
            "pause sentinel has duplicate held evaluator IDs"
        );
        ensure!(
            released.is_subset(&held),
            "pause sentinel released an evaluator it did not hold"
        );
        Ok(())
    }

    fn unreleased(&self) -> Vec<String> {
        let released = self
            .released_evaluator_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        self.held_evaluator_ids
            .iter()
            .filter(|id| !released.contains(id.as_str()))
            .cloned()
            .collect()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(role) = args.register_role {
        let spec = args.spec.as_deref().context("--spec is required")?;
        let job_id = args.job_id.as_deref().context("--job-id is required")?;
        let request = register_request(&args.root, role.into(), spec, job_id)?;
        println!("{}", serde_json::to_string(&request)?);
        return Ok(());
    }
    loop {
        let cycle_started = Instant::now();
        if let Err(error) = cycle(&args) {
            let failure = json!({
                "schema": "GpuLaneAdmissionError.v1",
                "generated_at_ms": now_ms(),
                "error": format!("{error:#}"),
            });
            atomic_json(
                &args.root.join("ops/gpu-lane-admission/error.v1.json"),
                &failure,
            )?;
            eprintln!("{}", serde_json::to_string(&failure)?);
            if args.once {
                return Err(error);
            }
        }
        if args.once {
            return Ok(());
        }
        let interval = Duration::from_secs(args.interval_seconds);
        if let Some(remaining) = interval.checked_sub(cycle_started.elapsed()) {
            thread::sleep(remaining);
        }
    }
}

fn request_path(root: &Path, role: GpuLaneRole) -> PathBuf {
    root.join("ops/gpu-lane-admission/requests")
        .join(format!("{}.json", role.as_str()))
}

fn state_path(root: &Path, role: GpuLaneRole) -> PathBuf {
    root.join("ops/gpu-lane-admission/state")
        .join(format!("{}.v1.json", role.as_str()))
}

fn sentinel_path(root: &Path) -> PathBuf {
    root.join("ops/gpu-lane.pause.json")
}

fn register_request(
    root: &Path,
    role: GpuLaneRole,
    spec: &Path,
    job_id: &str,
) -> Result<GpuLaneAdmissionV1> {
    ensure!(spec.is_file(), "GPU immutable spec is absent");
    let digest = Sha256Digest::parse(sha256_file(spec)?)?;
    let request =
        GpuLaneAdmissionV1::new(role, spec.to_owned(), digest, job_id.to_owned(), now_ms())?;
    let path = request_path(root, role);
    if path.is_file() {
        let existing: GpuLaneAdmissionV1 = serde_json::from_value(read_json(&path)?)?;
        existing.validate()?;
        ensure!(
            existing.immutable_job_identity == request.immutable_job_identity
                && existing.immutable_spec_sha256 == request.immutable_spec_sha256,
            "a different immutable GPU request already owns this role"
        );
        return Ok(existing);
    }
    atomic_json(&path, &serde_json::to_value(&request)?)?;
    append_event(root, "request_registered", &request)?;
    Ok(request)
}

fn cycle(args: &Args) -> Result<()> {
    let jobs = scheduler_jobs(&scq_json(&args.scq, &["ls"])?)?;
    let nodes = node_capacity(&scq_json(&args.scq, &["nodes"])?)?;
    let mut states = load_active_states(&args.root)?;
    if states.is_empty() {
        if sentinel_path(&args.root).is_file() {
            recover_orphan_sentinel(args, &jobs)?;
        }
        publish_status(args, &states, &nodes, false)?;
        return Ok(());
    }
    for state in states.values_mut() {
        state.recover();
        if jobs.iter().any(|job| {
            job.id == state.immutable_job_identity
                && !matches!(job.state.as_str(), "failed" | "finished" | "cancelled")
        }) {
            state.stage = GpuAdmissionStage::Submitted;
            state.recovery_disposition = RecoveryDisposition::ConfirmSubmissionThenRelease;
            state.updated_at_ms = now_ms();
            save_state(&args.root, state)?;
        }
    }

    if states.values().all(|state| {
        matches!(
            state.stage,
            GpuAdmissionStage::Submitted | GpuAdmissionStage::DependencyFailed
        )
    }) {
        if sentinel_path(&args.root).is_file() {
            recover_orphan_sentinel(args, &jobs)?;
        }
        for state in states.values_mut() {
            if state.stage == GpuAdmissionStage::Submitted {
                if state.held_evaluator_ids != state.released_evaluator_ids {
                    state.released_evaluator_ids = archived_exact_release(&args.root, state)?
                        .with_context(|| {
                            format!(
                                "submitted {} admission has no exact archived release receipt",
                                state.role.as_str()
                            )
                        })?;
                }
                state.stage = GpuAdmissionStage::Complete;
            }
            state.updated_at_ms = now_ms();
            state.recover();
            save_state(&args.root, state)?;
            archive_request(&args.root, state)?;
        }
        return publish_status(args, &states, &nodes, false);
    }

    let mut roles = Vec::new();
    for state in states.values_mut() {
        if matches!(
            state.stage,
            GpuAdmissionStage::Submitted | GpuAdmissionStage::DependencyFailed
        ) {
            continue;
        }
        let capacity = nodes
            .get(&state.target_node)
            .with_context(|| format!("target node {} is absent", state.target_node))?;
        if capacity_available(state, capacity) {
            roles.push(state.role);
        } else {
            state.stage = GpuAdmissionStage::WaitingForCapacity;
            state.recovery_disposition = RecoveryDisposition::ResumeCapacityWait;
            state.last_capacity_backoff = Some(format!(
                "{}: up={} free_gpus={} free_core_budget={} requires 1 GPU + 4 cores",
                capacity.node, capacity.up, capacity.gpus_free, capacity.free_core_budget
            ));
            state.updated_at_ms = now_ms();
            save_state(&args.root, state)?;
        }
    }
    if roles.is_empty() {
        if sentinel_path(&args.root).is_file() {
            recover_orphan_sentinel(args, &jobs)?;
        }
        return publish_status(args, &states, &nodes, false);
    }

    let mut sentinel = load_or_create_sentinel(&args.root, roles)?;
    recover_inflight(args, &mut sentinel, &jobs)?;
    for state in states.values_mut() {
        if sentinel.roles.contains(&state.role) && state.stage == GpuAdmissionStage::Requested {
            state.stage = GpuAdmissionStage::QueuePaused;
            state.recovery_disposition = RecoveryDisposition::ResumeHolding;
            state.updated_at_ms = now_ms();
            save_state(&args.root, state)?;
        }
    }
    hold_queued_evaluators(args, &mut sentinel, &jobs)?;
    for state in states.values_mut() {
        if !sentinel.roles.contains(&state.role) {
            continue;
        }
        state.held_evaluator_ids = sentinel.held_evaluator_ids.clone();
        if !matches!(
            state.stage,
            GpuAdmissionStage::Submitted | GpuAdmissionStage::DependencyFailed
        ) {
            state.stage = GpuAdmissionStage::WaitingForCapacity;
            state.recovery_disposition = RecoveryDisposition::ResumeCapacityWait;
        }
        state.updated_at_ms = now_ms();
        save_state(&args.root, state)?;
    }

    for state in states.values_mut() {
        if !sentinel.roles.contains(&state.role) {
            continue;
        }
        if matches!(
            state.stage,
            GpuAdmissionStage::Submitted | GpuAdmissionStage::DependencyFailed
        ) {
            continue;
        }
        let capacity = nodes
            .get(&state.target_node)
            .with_context(|| format!("target node {} is absent", state.target_node))?;
        if !capacity.up
            || capacity.gpus_free < state.resource_reservation.gpus as u64
            || capacity.free_core_budget < state.resource_reservation.companion_cpu_cores as i64
        {
            state.last_capacity_backoff = Some(format!(
                "{}: up={} free_gpus={} free_core_budget={} requires 1 GPU + 4 cores",
                capacity.node, capacity.up, capacity.gpus_free, capacity.free_core_budget
            ));
            state.updated_at_ms = now_ms();
            save_state(&args.root, state)?;
            continue;
        }
        match validate_and_submit(args, state, &jobs) {
            Ok(true) => {
                state.stage = GpuAdmissionStage::Submitted;
                state.recovery_disposition = RecoveryDisposition::ConfirmSubmissionThenRelease;
                state.last_capacity_backoff = None;
                state.updated_at_ms = now_ms();
                save_state(&args.root, state)?;
                append_event(&args.root, "gpu_spec_submitted", state)?;
            }
            Ok(false) => {
                state.last_capacity_backoff =
                    Some("capacity changed during dependency verification; retrying".to_owned());
                state.updated_at_ms = now_ms();
                save_state(&args.root, state)?;
            }
            Err(error) => {
                state.stage = GpuAdmissionStage::DependencyFailed;
                state.dependency_failure = Some(format!("{error:#}"));
                state.recovery_disposition = RecoveryDisposition::NothingToRecover;
                state.updated_at_ms = now_ms();
                save_state(&args.root, state)?;
                append_event(&args.root, "dependency_failed", state)?;
            }
        }
    }

    // The pause is a narrow crash-safe submit transaction, never a capacity
    // wait. Release exactly what this transaction held even when capacity
    // changed between the inventory read and SCQ's dependency check.
    release_exact_evaluators(args, &mut sentinel)?;
    for role in &sentinel.roles {
        if let Some(state) = states.get_mut(role) {
            state.released_evaluator_ids = sentinel.released_evaluator_ids.clone();
            if state.stage == GpuAdmissionStage::Submitted {
                state.stage = GpuAdmissionStage::Complete;
            }
            state.updated_at_ms = now_ms();
            state.recover();
            state.validate()?;
            save_state(&args.root, state)?;
            if matches!(
                state.stage,
                GpuAdmissionStage::Complete | GpuAdmissionStage::DependencyFailed
            ) {
                archive_request(&args.root, state)?;
            }
        }
    }
    archive_sentinel(&args.root, &sentinel)?;
    publish_status(args, &states, &nodes, false)
}

fn capacity_available(state: &GpuLaneAdmissionV1, capacity: &NodeCapacity) -> bool {
    capacity.up
        && capacity.gpus_free >= state.resource_reservation.gpus as u64
        && capacity.free_core_budget >= state.resource_reservation.companion_cpu_cores as i64
}

fn load_active_states(root: &Path) -> Result<BTreeMap<GpuLaneRole, GpuLaneAdmissionV1>> {
    let mut output = BTreeMap::new();
    for role in [
        GpuLaneRole::CohortProposer,
        GpuLaneRole::AlternatingTrainer,
        GpuLaneRole::BreadthEncoder,
    ] {
        let request = request_path(root, role);
        let state = state_path(root, role);
        let selected = if request.is_file() {
            request
        } else if state.is_file() {
            state.clone()
        } else {
            continue;
        };
        let mut value: GpuLaneAdmissionV1 = serde_json::from_value(read_json(&selected)?)?;
        value.validate()?;
        if matches!(
            value.stage,
            GpuAdmissionStage::Complete | GpuAdmissionStage::DependencyFailed
        ) && !request_path(root, role).is_file()
        {
            continue;
        }
        if state.is_file() {
            let persisted: GpuLaneAdmissionV1 = serde_json::from_value(read_json(&state)?)?;
            persisted.validate()?;
            if persisted.immutable_job_identity == value.immutable_job_identity {
                value = persisted;
            } else if !matches!(
                persisted.stage,
                GpuAdmissionStage::Complete | GpuAdmissionStage::DependencyFailed
            ) {
                bail!("active GPU role state conflicts with a new request");
            }
        }
        output.insert(role, value);
    }
    Ok(output)
}

fn load_or_create_sentinel(root: &Path, mut roles: Vec<GpuLaneRole>) -> Result<PauseSentinel> {
    roles.sort();
    roles.dedup();
    let path = sentinel_path(root);
    let mut sentinel = if path.is_file() {
        serde_json::from_value(read_json(&path)?)?
    } else {
        PauseSentinel::new(roles.clone())
    };
    sentinel.validate()?;
    sentinel.roles = roles;
    sentinel.updated_at_ms = now_ms();
    save_sentinel(root, &sentinel)?;
    Ok(sentinel)
}

fn recover_inflight(
    args: &Args,
    sentinel: &mut PauseSentinel,
    jobs: &[SchedulerJob],
) -> Result<()> {
    if let Some(id) = sentinel.hold_inflight_id.take() {
        if jobs.iter().any(|job| job.id == id && job.state == "held")
            && !sentinel.held_evaluator_ids.contains(&id)
        {
            sentinel.held_evaluator_ids.push(id);
            sentinel.held_evaluator_ids.sort();
        }
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
    }
    if let Some(id) = sentinel.release_inflight_id.take() {
        if !jobs.iter().any(|job| job.id == id && job.state == "held")
            && !sentinel.released_evaluator_ids.contains(&id)
        {
            sentinel.released_evaluator_ids.push(id);
            sentinel.released_evaluator_ids.sort();
        }
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
    }
    Ok(())
}

fn hold_queued_evaluators(
    args: &Args,
    sentinel: &mut PauseSentinel,
    jobs: &[SchedulerJob],
) -> Result<()> {
    let already = sentinel
        .held_evaluator_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let ids = jobs
        .iter()
        .filter(|job| job.project == EVALUATOR_PROJECT && job.state == "queued")
        .map(|job| job.id.clone())
        .filter(|id| !already.contains(id.as_str()))
        .collect::<Vec<_>>();
    for id in ids {
        sentinel.hold_inflight_id = Some(id.clone());
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
        let output = Command::new(&args.scq).args(["hold", &id]).output()?;
        if output.status.success() {
            sentinel.held_evaluator_ids.push(id);
            sentinel.held_evaluator_ids.sort();
        }
        sentinel.hold_inflight_id = None;
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
    }
    sentinel.validate()
}

fn release_exact_evaluators(args: &Args, sentinel: &mut PauseSentinel) -> Result<()> {
    for id in sentinel.unreleased() {
        sentinel.release_inflight_id = Some(id.clone());
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
        let output = Command::new(&args.scq).args(["release", &id]).output()?;
        if !output.status.success() {
            let jobs = scheduler_jobs(&scq_json(&args.scq, &["ls"])?)?;
            if jobs.iter().any(|job| job.id == id && job.state == "held") {
                bail!(
                    "failed to release exact evaluator {id}: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
        }
        sentinel.released_evaluator_ids.push(id);
        sentinel.released_evaluator_ids.sort();
        sentinel.release_inflight_id = None;
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
    }
    sentinel.validate()
}

fn validate_and_submit(
    args: &Args,
    state: &GpuLaneAdmissionV1,
    jobs: &[SchedulerJob],
) -> Result<bool> {
    if jobs
        .iter()
        .any(|job| job.id == state.immutable_job_identity)
    {
        return Ok(true);
    }
    ensure!(
        sha256_file(&state.immutable_spec_path)? == state.immutable_spec_sha256.as_str(),
        "immutable GPU spec hash drift"
    );
    let spec = fs::read_to_string(&state.immutable_spec_path)?;
    ensure!(
        spec.contains("gpus_per_task = 1") && spec.contains("cpus_reserved = 4"),
        "GPU spec does not reserve one GPU and four companion cores"
    );
    let output = Command::new(&args.scq)
        .arg("check")
        .arg(&state.immutable_spec_path)
        .output()?;
    if !output.status.success() {
        let message = command_message(&output);
        if capacity_backoff(&message) {
            return Ok(false);
        }
        bail!("dependency verification rejected immutable spec: {message}");
    }
    let output = Command::new(&args.scq)
        .arg("submit")
        .arg(&state.immutable_spec_path)
        .output()?;
    let message = command_message(&output);
    if output.status.success() || message.to_ascii_lowercase().contains("already exists") {
        return Ok(true);
    }
    if capacity_backoff(&message) {
        return Ok(false);
    }
    bail!("SCQ rejected immutable GPU submission: {message}")
}

fn command_message(output: &std::process::Output) -> String {
    String::from_utf8_lossy(if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    })
    .trim()
    .to_owned()
}

fn capacity_backoff(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && (message.contains("no eligible launch-ready node currently")
            || message.contains("eligible_node launch-ready-capacity"))
}

fn scheduler_jobs(value: &Value) -> Result<Vec<SchedulerJob>> {
    Ok(serde_json::from_value(
        value
            .get("Jobs")
            .and_then(|value| value.get("jobs"))
            .cloned()
            .context("SCQ jobs response is missing Jobs.jobs")?,
    )?)
}

fn node_capacity(value: &Value) -> Result<BTreeMap<String, NodeCapacity>> {
    let rows: Vec<NodeCapacity> = serde_json::from_value(
        value
            .get("Nodes")
            .and_then(|value| value.get("nodes"))
            .cloned()
            .context("SCQ nodes response is missing Nodes.nodes")?,
    )?;
    Ok(rows
        .into_iter()
        .map(|row| (row.node.clone(), row))
        .collect())
}

fn save_state(root: &Path, state: &GpuLaneAdmissionV1) -> Result<()> {
    state.validate()?;
    atomic_json(&state_path(root, state.role), &serde_json::to_value(state)?)
}

fn save_sentinel(root: &Path, sentinel: &PauseSentinel) -> Result<()> {
    sentinel.validate()?;
    atomic_json(&sentinel_path(root), &serde_json::to_value(sentinel)?)
}

fn archive_request(root: &Path, state: &GpuLaneAdmissionV1) -> Result<()> {
    let source = request_path(root, state.role);
    if !source.is_file() {
        return Ok(());
    }
    let directory = root.join("ops/gpu-lane-admission/completed");
    fs::create_dir_all(&directory)?;
    let target = directory.join(format!(
        "{}-{}-{}.json",
        state.role.as_str(),
        state.requested_at_ms,
        state.immutable_job_identity
    ));
    fs::rename(source, target)?;
    Ok(())
}

fn archive_sentinel(root: &Path, sentinel: &PauseSentinel) -> Result<()> {
    let source = sentinel_path(root);
    if !source.is_file() {
        return Ok(());
    }
    let directory = root.join("ops/gpu-lane-admission/completed");
    fs::create_dir_all(&directory)?;
    fs::rename(
        source,
        directory.join(format!("pause-{}.json", sentinel.created_at_ms)),
    )?;
    Ok(())
}

fn archived_exact_release(root: &Path, state: &GpuLaneAdmissionV1) -> Result<Option<Vec<String>>> {
    let directory = root.join("ops/gpu-lane-admission/completed");
    if !directory.is_dir() {
        return Ok(None);
    }
    let expected = state
        .held_evaluator_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut paths = fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("pause-") && name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths.reverse();
    for path in paths {
        let sentinel: PauseSentinel = serde_json::from_value(read_json(&path)?)?;
        sentinel.validate()?;
        let held = sentinel
            .held_evaluator_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let released = sentinel
            .released_evaluator_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if sentinel.roles.contains(&state.role) && held == expected && released == expected {
            return Ok(Some(sentinel.released_evaluator_ids));
        }
    }
    Ok(None)
}

fn recover_orphan_sentinel(args: &Args, jobs: &[SchedulerJob]) -> Result<()> {
    let mut sentinel: PauseSentinel =
        serde_json::from_value(read_json(&sentinel_path(&args.root))?)?;
    recover_inflight(args, &mut sentinel, jobs)?;
    release_exact_evaluators(args, &mut sentinel)?;
    archive_sentinel(&args.root, &sentinel)
}

fn publish_status(
    args: &Args,
    states: &BTreeMap<GpuLaneRole, GpuLaneAdmissionV1>,
    nodes: &BTreeMap<String, NodeCapacity>,
    pause_active: bool,
) -> Result<()> {
    let alarms = states
        .values()
        .filter(|state| state.idle_alarm_due(now_ms()))
        .map(|state| {
            json!({
                "role": state.role,
                "job_id": state.immutable_job_identity,
                "idle_ms": now_ms().saturating_sub(state.eligible_idle_since_ms),
                "stage": state.stage,
                "last_capacity_backoff": state.last_capacity_backoff,
            })
        })
        .collect::<Vec<_>>();
    let status = json!({
        "schema": "GpuLaneAdmissionStatus.v1",
        "generated_at_ms": now_ms(),
        "pause_active": pause_active,
        "retry_interval_ms": GPU_ADMISSION_RETRY_MS,
        "states": states,
        "nodes": nodes,
        "alarms": alarms,
    });
    atomic_json(
        &args.root.join("ops/gpu-lane-admission/status.v1.json"),
        &status,
    )?;
    atomic_json(
        &args.root.join("ops/gpu-lane-admission/alarm.v1.json"),
        &json!({
            "schema": "GpuLaneIdleAlarm.v1",
            "generated_at_ms": now_ms(),
            "active": !alarms.is_empty(),
            "alarms": alarms,
        }),
    )
}

fn append_event<T: Serialize>(root: &Path, kind: &str, detail: &T) -> Result<()> {
    let path = root.join("ops/gpu-lane-admission/events.v1.jsonl");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(
        &mut file,
        &json!({
            "schema": "GpuLaneAdmissionEvent.v1",
            "kind": kind,
            "created_at_ms": now_ms(),
            "detail": detail,
        }),
    )?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> GpuLaneAdmissionV1 {
        GpuLaneAdmissionV1::new(
            GpuLaneRole::CohortProposer,
            "/tmp/spec".into(),
            Sha256Digest::parse("a".repeat(64)).unwrap(),
            "proposer-job".to_owned(),
            1,
        )
        .unwrap()
    }

    #[test]
    fn capacity_wait_does_not_enter_the_pause_transaction() {
        let state = request();
        for capacity in [
            NodeCapacity {
                node: "xbabe1".to_owned(),
                up: false,
                gpus_free: 1,
                free_core_budget: 4,
            },
            NodeCapacity {
                node: "xbabe1".to_owned(),
                up: true,
                gpus_free: 0,
                free_core_budget: 4,
            },
            NodeCapacity {
                node: "xbabe1".to_owned(),
                up: true,
                gpus_free: 1,
                free_core_budget: 3,
            },
        ] {
            assert!(!capacity_available(&state, &capacity));
        }
        assert!(capacity_available(
            &state,
            &NodeCapacity {
                node: "xbabe1".to_owned(),
                up: true,
                gpus_free: 1,
                free_core_budget: 4,
            }
        ));
    }

    #[test]
    fn submitted_recovery_requires_an_exact_archived_release() {
        let root = std::env::temp_dir().join(format!(
            "jope-gpu-release-recovery-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let completed = root.join("ops/gpu-lane-admission/completed");
        fs::create_dir_all(&completed).unwrap();
        let mut state = request();
        state.held_evaluator_ids = vec!["eval-a".to_owned(), "eval-b".to_owned()];
        state.stage = GpuAdmissionStage::Submitted;
        state.recovery_disposition = RecoveryDisposition::ConfirmSubmissionThenRelease;
        let mut sentinel = PauseSentinel::new(vec![GpuLaneRole::CohortProposer]);
        sentinel.held_evaluator_ids = state.held_evaluator_ids.clone();
        sentinel.released_evaluator_ids = vec!["eval-a".to_owned()];
        atomic_json(
            &completed.join("pause-1.json"),
            &serde_json::to_value(&sentinel).unwrap(),
        )
        .unwrap();
        assert!(archived_exact_release(&root, &state).unwrap().is_none());

        sentinel.released_evaluator_ids = state.held_evaluator_ids.clone();
        atomic_json(
            &completed.join("pause-2.json"),
            &serde_json::to_value(&sentinel).unwrap(),
        )
        .unwrap();
        assert_eq!(
            archived_exact_release(&root, &state).unwrap().unwrap(),
            state.held_evaluator_ids
        );
        fs::remove_dir_all(root).unwrap();
    }
}
