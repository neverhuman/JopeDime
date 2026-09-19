//! Rust-only pressure-gated self-play capacity supervisor.
//!
//! Audit is the default. `--apply` records a content-addressed, idempotent
//! runtime policy decision; it never restarts SCQ, changes xbabe1/xbabe3, or
//! touches GPU allocations. A compatible SCQ build must expose multiplier
//! fields before this tool reports the decision as scheduler-enforced.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use crate::selfplay_core::{
    node_slot_target, ramp_decision, xbabe2_burn_eligible, BurnSample, PressureBaseline,
    PressureSample, RampStage, RampState,
};
use crate::selfplay_runtime::{
    atomic_json, canonical_json_bytes, default_root_str, default_scq_str, now_ms, read_json,
    scq_json, sha256_bytes, sha256_file,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_POLICY: &str = "manifests/selfplay/scq-nodes.v3.toml";
const SAMPLE_MIN_GAP_MS: u128 = 45_000;
const SAMPLE_MAX_GAP_MS: u128 = 75_000;
const HISTORY_LIMIT: usize = 180;

#[derive(Parser, Debug)]
#[command(
    name = "jope-capacity",
    about = "Audit or apply pressure-gated JOPE CPU slot multipliers"
)]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(long, default_value = DEFAULT_POLICY)]
    policy: PathBuf,
    /// Optional JSON metrics used for failure/runtime/throughput gates.
    #[arg(long)]
    metrics: Option<PathBuf>,
    /// Persist one due telemetry sample and the resulting runtime policy.
    #[arg(long)]
    apply: bool,
    /// Required with --apply. Reusing it returns the original receipt.
    #[arg(long, requires = "apply")]
    idempotency_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct StaticPolicy {
    poll_ms: u64,
    telemetry_ms: u64,
    multipliers: BTreeMap<String, f64>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ExternalMetrics {
    failure_rate: f64,
    p95_runtime_seconds: f64,
    throughput_per_slot: f64,
    baseline_failure_rate: f64,
    baseline_p95_runtime_seconds: f64,
    baseline_throughput_per_slot: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct StoredPressure {
    cpu_psi: f64,
    memory_psi: f64,
    io_psi: f64,
    available_memory_fraction: Option<f64>,
    failure_rate: f64,
    p95_runtime_seconds: f64,
    throughput_per_slot: f64,
}

impl StoredPressure {
    fn core(&self) -> PressureSample {
        PressureSample {
            cpu_psi: self.cpu_psi,
            memory_psi: self.memory_psi,
            io_psi: self.io_psi,
            available_memory_fraction: self.available_memory_fraction,
            failure_rate: self.failure_rate,
            p95_runtime_seconds: self.p95_runtime_seconds,
            throughput_per_slot: self.throughput_per_slot,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct StoredSample {
    sampled_at_ms: u128,
    cluster_cpu_fraction: f64,
    xbabe4: StoredPressure,
    cluster: StoredPressure,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct StoredBaseline {
    failure_rate: f64,
    p95_runtime_seconds: f64,
    throughput_per_slot: f64,
}

impl StoredBaseline {
    fn core(&self) -> PressureBaseline {
        PressureBaseline {
            failure_rate: self.failure_rate,
            p95_runtime_seconds: self.p95_runtime_seconds,
            throughput_per_slot: self.throughput_per_slot,
        }
    }
}

/// The single schema string `ops/capacity/state-v1.json` carries. Without it
/// the document cannot be version-gated at all: `load_state` would happily
/// deserialize any object that happens to have the right field names, and a
/// future shape change would surface as a wrong ramp decision rather than a
/// rejected file.
const CAPACITY_STATE_SCHEMA: &str = "JopeCapacityRampState.v1";

fn capacity_state_schema() -> String {
    CAPACITY_STATE_SCHEMA.to_owned()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct StoredRampState {
    /// `#[serde(default)]` so a state file written before this key existed
    /// still loads; `load_state` rejects a WRONG value, not a missing one.
    #[serde(default = "capacity_state_schema")]
    schema: String,
    stage: String,
    effective_cores: i64,
    multiplier: f64,
    slot_target: i64,
    stage_started_at_ms: Option<i64>,
    rollback_at_ms: Option<i64>,
    rollback_reasons: Vec<String>,
    baseline: StoredBaseline,
}

impl StoredRampState {
    fn from_core(value: &RampState) -> Self {
        Self {
            schema: capacity_state_schema(),
            stage: stage_name(value.stage).to_owned(),
            effective_cores: value.effective_cores,
            multiplier: value.multiplier,
            slot_target: value.slot_target,
            stage_started_at_ms: value.stage_started_at_ms,
            rollback_at_ms: value.rollback_at_ms,
            rollback_reasons: value.rollback_reasons.clone(),
            baseline: StoredBaseline {
                failure_rate: value.baseline.failure_rate,
                p95_runtime_seconds: value.baseline.p95_runtime_seconds,
                throughput_per_slot: value.baseline.throughput_per_slot,
            },
        }
    }

    fn core(&self) -> Result<RampState> {
        Ok(RampState {
            stage: parse_stage(&self.stage)?,
            effective_cores: self.effective_cores,
            multiplier: self.multiplier,
            slot_target: self.slot_target,
            stage_started_at_ms: self.stage_started_at_ms,
            rollback_at_ms: self.rollback_at_ms,
            rollback_reasons: self.rollback_reasons.clone(),
            baseline: self.baseline.core(),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct NodeReport {
    node: String,
    physical_cores: i64,
    effective_cores: i64,
    configured_multiplier: f64,
    runtime_multiplier: f64,
    slot_target: i64,
    scheduler_multiplier: Option<f64>,
    scheduler_slot_target: Option<i64>,
    running_slots: i64,
    cpu_psi: f64,
    memory_psi: f64,
    io_psi: f64,
    available_memory_fraction: f64,
    protected_gpu_allocations: Vec<String>,
}

#[derive(Clone, Debug)]
struct NodeTelemetry {
    node: String,
    physical_cores: i64,
    effective_cores: i64,
    running_slots: i64,
    scheduler_multiplier: Option<f64>,
    scheduler_slot_target: Option<i64>,
    cpu_busy_fraction: f64,
    pressure: StoredPressure,
}

#[derive(Clone, Debug, Serialize)]
struct Decision {
    schema: &'static str,
    generated_at_ms: u128,
    mode: &'static str,
    policy_sha256: String,
    scheduler_supports_multipliers: bool,
    scheduler_enforced: bool,
    requires_compatible_scq_deployment: bool,
    xbabe4_stage: String,
    xbabe4_rollback_reasons: Vec<String>,
    xbabe2_burn_eligible: bool,
    fresh_queue_depth: i64,
    telemetry_sample_recorded: bool,
    one_minute_samples: usize,
    nodes: Vec<NodeReport>,
    protected_nodes_unchanged: bool,
    protected_gpu_allocations_unchanged: bool,
    compile_cache: Option<Value>,
}

pub fn main(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(&args);
    if args.apply
        && args
            .idempotency_key
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
    {
        bail!("--apply requires a nonempty --idempotency-key");
    }
    if args.apply {
        if let Some(receipt) = replay_receipt(
            &args.root,
            args.idempotency_key.as_deref().expect("checked"),
        )? {
            println!("{}", serde_json::to_string(&receipt)?);
            return Ok(());
        }
    }

    let policy_text = fs::read_to_string(&args.policy)
        .with_context(|| format!("read capacity policy {}", args.policy.display()))?;
    let policy = parse_policy(&policy_text)?;
    validate_policy(&policy)?;
    let policy_sha256 = sha256_file(&args.policy)?;
    let nodes_value = scq_json(&args.scq, &["nodes"])?;
    let (nodes, gpu_allocations, scheduler_supports_multipliers) = parse_scq_nodes(&nodes_value)?;
    let metrics = args
        .metrics
        .as_ref()
        .map(|path| -> Result<ExternalMetrics> {
            Ok(serde_json::from_value(read_json(path)?)
                .with_context(|| format!("parse external metrics {}", path.display()))?)
        })
        .transpose()?
        .unwrap_or_default();
    validate_metrics(&metrics)?;

    let generated_at_ms = now_ms();
    let current_sample = build_sample(generated_at_ms, &nodes, metrics)?;
    let capacity_root = args.root.join("ops/capacity");
    let history_path = capacity_root.join("telemetry-v1.json");
    let mut history = load_history(&history_path)?;
    let sample_recorded = if args.apply && sample_due(&history, generated_at_ms) {
        history.push(current_sample);
        if history.len() > HISTORY_LIMIT {
            history.drain(..history.len() - HISTORY_LIMIT);
        }
        true
    } else {
        false
    };
    let consecutive = trailing_minute_samples(&history);

    let xbabe4 = nodes
        .iter()
        .find(|node| node.node == "xbabe4")
        .context("SCQ telemetry does not contain xbabe4")?;
    let baseline = PressureBaseline {
        failure_rate: metrics.baseline_failure_rate,
        p95_runtime_seconds: metrics.baseline_p95_runtime_seconds,
        throughput_per_slot: metrics.baseline_throughput_per_slot,
    };
    let state_path = capacity_root.join("state-v1.json");
    let prior_state = load_state(&state_path, xbabe4.effective_cores, baseline)?;
    let x4_samples = consecutive
        .iter()
        .map(|row| row.xbabe4.core())
        .collect::<Vec<_>>();
    let next_state = if args.apply {
        ramp_decision(&prior_state, &x4_samples, generated_at_ms as i64)?
    } else {
        prior_state
    };

    let queue_status = args.root.join("ops/queue-keeper-v3.json");
    let fresh_queue_depth = fresh_queue_depth(&queue_status)?;
    let burn_samples = consecutive
        .iter()
        .map(|row| BurnSample {
            cluster_cpu_fraction: row.cluster_cpu_fraction,
            pressure: row.cluster.core(),
            baseline,
        })
        .collect::<Vec<_>>();
    let xbabe2_eligible = xbabe2_burn_eligible(&burn_samples, fresh_queue_depth)?;
    let runtime_multipliers = BTreeMap::from([
        ("xbabe1".to_owned(), 1.0),
        ("xbabe2".to_owned(), if xbabe2_eligible { 1.2 } else { 1.0 }),
        ("xbabe3".to_owned(), 1.0),
        ("xbabe4".to_owned(), next_state.multiplier),
    ]);
    let reports = build_reports(&nodes, &policy, &runtime_multipliers, &gpu_allocations)?;
    let scheduler_enforced =
        scheduler_supports_multipliers && reports.iter().all(scheduler_matches_runtime);
    let decision = Decision {
        schema: "JopeSelfplayCapacityDecision.v1",
        generated_at_ms,
        mode: if args.apply { "apply" } else { "audit" },
        policy_sha256,
        scheduler_supports_multipliers,
        scheduler_enforced,
        requires_compatible_scq_deployment: !scheduler_supports_multipliers,
        xbabe4_stage: stage_name(next_state.stage).to_owned(),
        xbabe4_rollback_reasons: next_state.rollback_reasons.clone(),
        xbabe2_burn_eligible: xbabe2_eligible,
        fresh_queue_depth,
        telemetry_sample_recorded: sample_recorded,
        one_minute_samples: consecutive.len(),
        nodes: reports,
        protected_nodes_unchanged: runtime_multipliers["xbabe1"] == 1.0
            && runtime_multipliers["xbabe3"] == 1.0,
        protected_gpu_allocations_unchanged: true,
        compile_cache: compile_cache_status(&args.root),
    };
    let value = serde_json::to_value(&decision)?;

    if args.apply {
        fs::create_dir_all(&capacity_root)?;
        atomic_json(&history_path, &serde_json::to_value(&history)?)?;
        atomic_json(
            &state_path,
            &serde_json::to_value(StoredRampState::from_core(&next_state))?,
        )?;
        atomic_json(&capacity_root.join("runtime-policy-v1.json"), &value)?;
        let receipt = write_receipt(
            &args.root,
            args.idempotency_key.as_deref().expect("checked"),
            &value,
        )?;
        println!("{}", serde_json::to_string(&receipt)?);
    } else {
        println!("{}", serde_json::to_string(&value)?);
    }
    Ok(())
}

fn parse_policy(text: &str) -> Result<StaticPolicy> {
    let mut poll_ms = None;
    let mut telemetry_ms = None;
    let mut current_node: Option<String> = None;
    let mut multipliers = BTreeMap::new();
    let mut seen_nodes = BTreeSet::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[node]]" {
            current_node = Some(String::new());
            continue;
        }
        let Some((key, raw_value)) = line.split_once('=') else {
            bail!("unsupported policy line: {line}");
        };
        let key = key.trim();
        let value = raw_value.trim().trim_matches('"');
        if let Some(node) = current_node.as_mut() {
            match key {
                "name" => {
                    if node.is_empty() {
                        *node = value.to_owned();
                        if !seen_nodes.insert(node.clone()) {
                            bail!("duplicate node policy for {node}");
                        }
                        multipliers.insert(node.clone(), 1.0);
                    } else {
                        bail!("duplicate node name in policy");
                    }
                }
                "cpu_job_slot_multiplier" => {
                    if node.is_empty() {
                        bail!("node multiplier appears before node name");
                    }
                    let multiplier = value
                        .parse::<f64>()
                        .with_context(|| format!("parse multiplier for {node}"))?;
                    multipliers.insert(node.clone(), multiplier);
                }
                _ => {}
            }
        } else {
            match key {
                "poll_ms" => poll_ms = Some(value.parse()?),
                "telemetry_ms" => telemetry_ms = Some(value.parse()?),
                _ => {}
            }
        }
    }
    if current_node.as_deref() == Some("") {
        bail!("node policy is missing its name");
    }
    Ok(StaticPolicy {
        poll_ms: poll_ms.context("policy lacks poll_ms")?,
        telemetry_ms: telemetry_ms.context("policy lacks telemetry_ms")?,
        multipliers,
    })
}

fn validate_policy(policy: &StaticPolicy) -> Result<()> {
    if policy.poll_ms != 2_000 || policy.telemetry_ms != 1_000 {
        bail!("self-play policy must use a 2s scheduler tick and 1s telemetry");
    }
    for node in ["xbabe1", "xbabe2", "xbabe3"] {
        if policy.multipliers.get(node).copied() != Some(1.0) {
            bail!("{node} must start at multiplier 1.0");
        }
    }
    if policy.multipliers.get("xbabe4").copied() != Some(2.0) {
        bail!("xbabe4 policy target must be multiplier 2.0");
    }
    if policy
        .multipliers
        .values()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        bail!("invalid CPU slot multiplier");
    }
    Ok(())
}

fn validate_metrics(metrics: &ExternalMetrics) -> Result<()> {
    for (name, value) in [
        ("failure_rate", metrics.failure_rate),
        ("p95_runtime_seconds", metrics.p95_runtime_seconds),
        ("throughput_per_slot", metrics.throughput_per_slot),
        ("baseline_failure_rate", metrics.baseline_failure_rate),
        (
            "baseline_p95_runtime_seconds",
            metrics.baseline_p95_runtime_seconds,
        ),
        (
            "baseline_throughput_per_slot",
            metrics.baseline_throughput_per_slot,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            bail!("{name} must be finite and nonnegative");
        }
    }
    Ok(())
}

fn parse_scq_nodes(
    value: &Value,
) -> Result<(Vec<NodeTelemetry>, BTreeMap<String, Vec<String>>, bool)> {
    let rows = value
        .get("Nodes")
        .and_then(|row| row.get("nodes"))
        .and_then(Value::as_array)
        .context("SCQ nodes response lacks Nodes.nodes")?;
    let mut nodes = Vec::new();
    let mut scheduler_supports = true;
    for row in rows {
        let node = row
            .get("node")
            .and_then(Value::as_str)
            .context("SCQ node lacks name")?
            .to_owned();
        let physical = integer(row, "cores")?;
        let effective = integer(row, "effective_cores")?;
        let running = integer(row, "running_jobs")?;
        if physical < effective || effective < 0 || running < 0 {
            bail!("invalid capacity counters for {node}");
        }
        let total_memory = number(row, "mem_total_gb")?;
        let available_memory = number(row, "mem_avail_gb")?;
        if total_memory <= 0.0 || !(0.0..=total_memory).contains(&available_memory) {
            bail!("invalid memory telemetry for {node}");
        }
        let scheduler_multiplier = row.get("cpu_job_slot_multiplier").and_then(Value::as_f64);
        let scheduler_slot_target = row.get("slot_target").and_then(Value::as_i64);
        scheduler_supports &= scheduler_multiplier.is_some() && scheduler_slot_target.is_some();
        if scheduler_multiplier.is_some_and(|value| !value.is_finite() || value <= 0.0)
            || scheduler_slot_target.is_some_and(|value| value < 0)
        {
            bail!("invalid scheduler multiplier telemetry for {node}");
        }
        nodes.push(NodeTelemetry {
            node,
            physical_cores: physical,
            effective_cores: effective,
            running_slots: running,
            scheduler_multiplier,
            scheduler_slot_target,
            cpu_busy_fraction: number(row, "cpu_busy_pct")? / 100.0,
            pressure: StoredPressure {
                cpu_psi: number(row, "psi_cpu10")?,
                memory_psi: number(row, "psi_mem10")?,
                io_psi: number(row, "psi_io10")?,
                available_memory_fraction: Some(available_memory / total_memory),
                ..zero_pressure()
            },
        });
    }
    let expected = BTreeSet::from(["xbabe1", "xbabe2", "xbabe3", "xbabe4"]);
    let actual = nodes.iter().map(|row| row.node.as_str()).collect();
    if expected != actual {
        bail!("SCQ node inventory must contain exactly xbabe1..xbabe4");
    }
    let mut gpu_allocations = BTreeMap::<String, Vec<String>>::new();
    for row in value
        .get("Nodes")
        .and_then(|row| row.get("gpus"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(node) = row.get("node").and_then(Value::as_str) else {
            continue;
        };
        let status = row
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if status != "free" {
            gpu_allocations
                .entry(node.to_owned())
                .or_default()
                .push(status.to_owned());
        }
    }
    Ok((nodes, gpu_allocations, scheduler_supports))
}

fn zero_pressure() -> StoredPressure {
    StoredPressure {
        cpu_psi: 0.0,
        memory_psi: 0.0,
        io_psi: 0.0,
        available_memory_fraction: None,
        failure_rate: 0.0,
        p95_runtime_seconds: 0.0,
        throughput_per_slot: 0.0,
    }
}

fn integer(value: &Value, key: &str) -> Result<i64> {
    value
        .get(key)
        .and_then(|field| {
            field
                .as_i64()
                .or_else(|| field.as_u64().map(|row| row as i64))
        })
        .with_context(|| format!("SCQ node lacks integer {key}"))
}

fn number(value: &Value, key: &str) -> Result<f64> {
    let result = value
        .get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("SCQ node lacks numeric {key}"))?;
    if !result.is_finite() {
        bail!("SCQ node {key} is not finite");
    }
    Ok(result)
}

fn build_sample(
    sampled_at_ms: u128,
    nodes: &[NodeTelemetry],
    metrics: ExternalMetrics,
) -> Result<StoredSample> {
    let xbabe4 = nodes
        .iter()
        .find(|row| row.node == "xbabe4")
        .context("xbabe4 telemetry is absent")?;
    let total_cores: i64 = nodes.iter().map(|row| row.physical_cores).sum();
    let busy = nodes
        .iter()
        .map(|row| row.cpu_busy_fraction * row.physical_cores as f64)
        .sum::<f64>();
    let mut cluster = zero_pressure();
    cluster.cpu_psi = nodes
        .iter()
        .map(|row| row.pressure.cpu_psi)
        .fold(0.0, f64::max);
    cluster.memory_psi = nodes
        .iter()
        .map(|row| row.pressure.memory_psi)
        .fold(0.0, f64::max);
    cluster.io_psi = nodes
        .iter()
        .map(|row| row.pressure.io_psi)
        .fold(0.0, f64::max);
    cluster.available_memory_fraction = nodes
        .iter()
        .filter_map(|row| row.pressure.available_memory_fraction)
        .min_by(f64::total_cmp);
    let set_metrics = |pressure: &mut StoredPressure| {
        pressure.failure_rate = metrics.failure_rate;
        pressure.p95_runtime_seconds = metrics.p95_runtime_seconds;
        pressure.throughput_per_slot = metrics.throughput_per_slot;
    };
    let mut x4_pressure = xbabe4.pressure.clone();
    set_metrics(&mut x4_pressure);
    set_metrics(&mut cluster);
    Ok(StoredSample {
        sampled_at_ms,
        cluster_cpu_fraction: if total_cores == 0 {
            0.0
        } else {
            busy / total_cores as f64
        },
        xbabe4: x4_pressure,
        cluster,
    })
}

fn load_history(path: &Path) -> Result<Vec<StoredSample>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    serde_json::from_value(read_json(path)?)
        .with_context(|| format!("parse capacity telemetry {}", path.display()))
}

fn sample_due(history: &[StoredSample], now: u128) -> bool {
    history
        .last()
        .is_none_or(|sample| now.saturating_sub(sample.sampled_at_ms) >= SAMPLE_MIN_GAP_MS)
}

fn trailing_minute_samples(history: &[StoredSample]) -> Vec<StoredSample> {
    if history.is_empty() {
        return Vec::new();
    }
    let mut start = history.len() - 1;
    while start > 0 {
        let gap = history[start]
            .sampled_at_ms
            .saturating_sub(history[start - 1].sampled_at_ms);
        if !(SAMPLE_MIN_GAP_MS..=SAMPLE_MAX_GAP_MS).contains(&gap) {
            break;
        }
        start -= 1;
    }
    history[start..].to_vec()
}

fn load_state(path: &Path, effective_cores: i64, baseline: PressureBaseline) -> Result<RampState> {
    if path.is_file() {
        // Version gate FIRST, on a Value, before the strict struct sees the
        // bytes — the pattern at jope_selfplay_supply.rs:423-430. The inverted
        // form (deserialize, then check `schema`) is unreachable for exactly
        // the documents it is meant to reject.
        let value = read_json(path)?;
        let schema = value.get("schema").and_then(Value::as_str);
        if let Some(schema) = schema {
            ensure!(
                schema == CAPACITY_STATE_SCHEMA,
                "capacity ramp state {} carries schema {schema:?}, expected {CAPACITY_STATE_SCHEMA:?}",
                path.display(),
            );
        }
        let stored: StoredRampState = serde_json::from_value(value)?;
        let state = stored.core()?;
        if state.effective_cores != effective_cores {
            bail!("xbabe4 effective cores changed; re-baseline explicitly");
        }
        return Ok(state);
    }
    RampState::baseline(effective_cores, baseline).map_err(Into::into)
}

fn stage_name(stage: RampStage) -> &'static str {
    match stage {
        RampStage::Baseline => "baseline",
        RampStage::Ramp30 => "ramp_30",
        RampStage::Target40 => "target_40",
        RampStage::RolledBack => "rolled_back",
    }
}

fn parse_stage(value: &str) -> Result<RampStage> {
    match value {
        "baseline" => Ok(RampStage::Baseline),
        "ramp_30" => Ok(RampStage::Ramp30),
        "target_40" => Ok(RampStage::Target40),
        "rolled_back" => Ok(RampStage::RolledBack),
        _ => bail!("unknown xbabe4 ramp stage {value}"),
    }
}

fn fresh_queue_depth(path: &Path) -> Result<i64> {
    if !path.is_file() {
        return Ok(0);
    }
    let value = read_json(path)?;
    if value.get("admission_allowed").and_then(Value::as_bool) != Some(true) {
        return Ok(0);
    }
    Ok(value
        .get("queued_starting")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0))
}

fn build_reports(
    nodes: &[NodeTelemetry],
    policy: &StaticPolicy,
    runtime: &BTreeMap<String, f64>,
    gpu_allocations: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<NodeReport>> {
    nodes
        .iter()
        .map(|node| {
            let configured = policy.multipliers.get(&node.node).copied().unwrap_or(1.0);
            let runtime_multiplier = runtime.get(&node.node).copied().unwrap_or(1.0);
            if matches!(node.node.as_str(), "xbabe1" | "xbabe3") && runtime_multiplier != 1.0 {
                bail!("protected node {} multiplier changed", node.node);
            }
            Ok(NodeReport {
                node: node.node.clone(),
                physical_cores: node.physical_cores,
                effective_cores: node.effective_cores,
                configured_multiplier: configured,
                runtime_multiplier,
                slot_target: node_slot_target(node.effective_cores, runtime_multiplier)?,
                scheduler_multiplier: node.scheduler_multiplier,
                scheduler_slot_target: node.scheduler_slot_target,
                running_slots: node.running_slots,
                cpu_psi: node.pressure.cpu_psi,
                memory_psi: node.pressure.memory_psi,
                io_psi: node.pressure.io_psi,
                available_memory_fraction: node.pressure.available_memory_fraction.unwrap_or(0.0),
                protected_gpu_allocations: gpu_allocations
                    .get(&node.node)
                    .cloned()
                    .unwrap_or_default(),
            })
        })
        .collect()
}

fn scheduler_matches_runtime(report: &NodeReport) -> bool {
    report.scheduler_multiplier == Some(report.runtime_multiplier)
        && report.scheduler_slot_target == Some(report.slot_target)
}

fn compile_cache_status(root: &Path) -> Option<Value> {
    for path in [
        root.join("ops/status-v3.json"),
        root.join("ops/status.json"),
    ] {
        if let Ok(value) = read_json(&path) {
            if let Some(cache) = value.get("compile_cache") {
                return Some(cache.clone());
            }
        }
    }
    None
}

fn receipt_dirs(root: &Path) -> (PathBuf, PathBuf) {
    let base = root.join("ops/capacity");
    (base.join("receipts"), base.join("idempotency"))
}

fn replay_receipt(root: &Path, key: &str) -> Result<Option<Value>> {
    let (receipts, idempotency) = receipt_dirs(root);
    let key_sha256 = sha256_bytes(key.as_bytes());
    let index_path = idempotency.join(format!("{key_sha256}.json"));
    if !index_path.is_file() {
        return Ok(None);
    }
    let index = read_json(&index_path)?;
    let digest = index
        .get("receipt_sha256")
        .and_then(Value::as_str)
        .context("idempotency index lacks receipt SHA-256")?;
    let receipt_path = receipts.join(format!("{digest}.json"));
    if sha256_file(&receipt_path)? != digest {
        bail!("capacity receipt content drift");
    }
    Ok(Some(read_json(&receipt_path)?))
}

fn write_receipt(root: &Path, key: &str, decision: &Value) -> Result<Value> {
    let (receipts, idempotency) = receipt_dirs(root);
    fs::create_dir_all(&receipts)?;
    fs::create_dir_all(&idempotency)?;
    let key_sha256 = sha256_bytes(key.as_bytes());
    let receipt = json!({
        "schema": "JopeSelfplayCapacityReceipt.v1",
        "idempotency_key_sha256": key_sha256,
        "decision": decision,
    });
    // atomic_json terminates every document with exactly one newline. Hash the
    // bytes that are actually persisted so filename and on-disk content agree.
    let mut persisted = canonical_json_bytes(&receipt)?;
    persisted.push(b'\n');
    let digest = sha256_bytes(&persisted);
    let receipt_path = receipts.join(format!("{digest}.json"));
    if receipt_path.is_file() {
        if sha256_file(&receipt_path)? != digest {
            bail!("existing capacity receipt content drift");
        }
    } else {
        atomic_json(&receipt_path, &receipt)?;
    }
    atomic_json(
        &idempotency.join(format!("{key_sha256}.json")),
        &json!({
            "schema": "JopeSelfplayCapacityIdempotency.v1",
            "receipt_sha256": digest,
        }),
    )?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay_core::{unsafe_sample, RAMP_MID_DURATION_MS};
    use tempfile::TempDir;

    fn policy() -> &'static str {
        r#"
poll_ms = 2000
telemetry_ms = 1000
[[node]]
name = "xbabe1"
[[node]]
name = "xbabe2"
cpu_job_slot_multiplier = 1.0
[[node]]
name = "xbabe3"
[[node]]
name = "xbabe4"
cpu_job_slot_multiplier = 2.0
"#
    }

    fn pressure(cpu: f64) -> StoredPressure {
        StoredPressure {
            cpu_psi: cpu,
            memory_psi: 0.0,
            io_psi: 0.0,
            available_memory_fraction: Some(0.5),
            failure_rate: 0.0,
            p95_runtime_seconds: 10.0,
            throughput_per_slot: 5.0,
        }
    }

    fn sample(at: u128, cpu: f64, cluster_cpu: f64) -> StoredSample {
        StoredSample {
            sampled_at_ms: at,
            cluster_cpu_fraction: cluster_cpu,
            xbabe4: pressure(cpu),
            cluster: pressure(cpu),
        }
    }

    #[test]
    fn policy_defaults_other_nodes_to_one_and_pins_xbabe4() {
        let parsed = parse_policy(policy()).unwrap();
        validate_policy(&parsed).unwrap();
        assert_eq!(parsed.multipliers["xbabe1"], 1.0);
        assert_eq!(parsed.multipliers["xbabe4"], 2.0);
    }

    #[test]
    fn minute_windows_break_on_dense_or_missing_samples() {
        let valid = vec![sample(0, 0.0, 0.5), sample(60_000, 0.0, 0.5)];
        assert_eq!(trailing_minute_samples(&valid).len(), 2);
        let dense = vec![sample(0, 0.0, 0.5), sample(1_000, 0.0, 0.5)];
        assert_eq!(trailing_minute_samples(&dense).len(), 1);
        let missing = vec![sample(0, 0.0, 0.5), sample(120_000, 0.0, 0.5)];
        assert_eq!(trailing_minute_samples(&missing).len(), 1);
    }

    #[test]
    fn exactly_three_minute_samples_trigger_rollback() {
        let baseline = PressureBaseline {
            failure_rate: 0.0,
            p95_runtime_seconds: 10.0,
            throughput_per_slot: 5.0,
        };
        let mut state = RampState::baseline(20, baseline).unwrap();
        state = ramp_decision(&state, &[], 0).unwrap();
        state = ramp_decision(&state, &[], RAMP_MID_DURATION_MS).unwrap();
        let samples = vec![
            sample(0, 0.25, 0.5),
            sample(60_000, 0.25, 0.5),
            sample(120_000, 0.25, 0.5),
        ];
        let two = samples[..2]
            .iter()
            .map(|row| row.xbabe4.core())
            .collect::<Vec<_>>();
        assert_eq!(
            ramp_decision(&state, &two, 180_000).unwrap().stage,
            RampStage::Target40
        );
        let three = samples
            .iter()
            .map(|row| row.xbabe4.core())
            .collect::<Vec<_>>();
        let rolled = ramp_decision(&state, &three, 180_000).unwrap();
        assert_eq!(rolled.stage, RampStage::RolledBack);
        assert_eq!(rolled.multiplier, 1.0);
    }

    #[test]
    fn xbabe2_requires_fifteen_clear_minutes_and_fresh_work() {
        let baseline = PressureBaseline {
            failure_rate: 0.0,
            p95_runtime_seconds: 10.0,
            throughput_per_slot: 5.0,
        };
        let samples = (0..15)
            .map(|minute| {
                let row = sample(minute * 60_000, 0.0, 0.74);
                BurnSample {
                    cluster_cpu_fraction: row.cluster_cpu_fraction,
                    pressure: row.cluster.core(),
                    baseline,
                }
            })
            .collect::<Vec<_>>();
        assert!(xbabe2_burn_eligible(&samples, 1).unwrap());
        assert!(!xbabe2_burn_eligible(&samples, 0).unwrap());
    }

    #[test]
    fn receipts_are_content_addressed_and_idempotent() {
        let temporary = TempDir::new().unwrap();
        let decision = json!({"schema": "test", "slot_target": 30});
        let first = write_receipt(temporary.path(), "same-key", &decision).unwrap();
        let replay = replay_receipt(temporary.path(), "same-key")
            .unwrap()
            .unwrap();
        assert_eq!(first, replay);
        let (receipts, _) = receipt_dirs(temporary.path());
        let files = fs::read_dir(receipts).unwrap().count();
        assert_eq!(files, 1);
    }

    #[test]
    fn all_gate_boundaries_remain_exact() {
        let baseline = PressureBaseline {
            failure_rate: 0.1,
            p95_runtime_seconds: 10.0,
            throughput_per_slot: 5.0,
        };
        let reasons = unsafe_sample(
            &PressureSample {
                cpu_psi: 0.25,
                memory_psi: 0.05,
                io_psi: 0.10,
                available_memory_fraction: Some(0.149),
                failure_rate: 0.1026,
                p95_runtime_seconds: 15.1,
                throughput_per_slot: 4.0,
            },
            &baseline,
        )
        .unwrap();
        assert_eq!(reasons.len(), 6);
    }
}
