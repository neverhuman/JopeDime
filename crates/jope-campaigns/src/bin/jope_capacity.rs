//! Rust-only pressure-gated self-play capacity supervisor.
//!
//! Audit is the default. `--apply` records a content-addressed, idempotent
//! runtime policy decision; it never restarts SCQ, changes xbabe1/xbabe3, or
//! touches GPU allocations. A compatible SCQ build must expose multiplier
//! fields before this tool reports the decision as scheduler-enforced.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_core::{
    node_slot_target, ramp_decision, xbabe2_burn_eligible, BurnSample, PressureBaseline,
    PressureSample, RampStage, RampState,
};
use jope_dime_regressor::selfplay_runtime::{
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

fn main() -> Result<()> {
    let args = Args::parse();
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
            serde_json::from_value(read_json(path)?)
                .with_context(|| format!("parse external metrics {}", path.display()))
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

#[path = "jope_capacity/policy.rs"]
mod policy;
pub(crate) use policy::{parse_policy, validate_metrics, validate_policy};
#[path = "jope_capacity/scq_nodes.rs"]
mod scq_nodes;
pub(crate) use scq_nodes::{integer, number, parse_scq_nodes, zero_pressure, ScqNodes};
#[path = "jope_capacity/samples.rs"]
mod samples;
pub(crate) use samples::{build_sample, load_history, sample_due, trailing_minute_samples};
#[path = "jope_capacity/state.rs"]
mod state;
pub(crate) use state::{fresh_queue_depth, load_state, parse_stage, stage_name};
#[path = "jope_capacity/reports.rs"]
mod reports;
pub(crate) use reports::{
    build_reports, compile_cache_status, receipt_dirs, replay_receipt, scheduler_matches_runtime,
};
#[path = "jope_capacity/receipts.rs"]
mod receipts;
pub(crate) use receipts::write_receipt;

#[cfg(test)]
#[path = "jope_capacity/tests.rs"]
mod tests;
