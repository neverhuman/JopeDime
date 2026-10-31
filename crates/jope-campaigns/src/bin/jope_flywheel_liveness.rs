//! Join the flywheel's independent control-plane status into one fail-closed view.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, default_scq_str, now_ms, read_json, scq_json,
};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const MIN_USEFUL_WORK_TARGET: usize = 6_144;
const MIN_CANDIDATE_EVALUATION_WORK_TARGET: usize = 8 * 128 * 256;
const STALE_CONTROL_MS: u128 = 2 * 60 * 1_000;
const IDLE_GPU_MS: u128 = 5 * 60 * 1_000;

#[derive(Parser, Debug)]
#[command(name = "jope-flywheel-liveness")]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(long)]
    once: bool,
    #[arg(long, default_value_t = 30)]
    interval_seconds: u64,
}

fn optional_json(path: &Path) -> Value {
    read_json(path).unwrap_or(Value::Null)
}

fn timestamp(value: &Value) -> u128 {
    value
        .get("generated_at_ms")
        .or_else(|| value.get("updated_at_ms"))
        .and_then(Value::as_u64)
        .unwrap_or_default() as u128
}

fn count_pending(path: &Path) -> usize {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| {
                    entry.path().extension().and_then(|value| value.to_str()) == Some("json")
                })
                .count()
        })
        .unwrap_or_default()
}

fn scheduler_job_active(scheduler: &Value, exact_id: Option<&str>, prefixes: &[&str]) -> bool {
    scheduler
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|job| {
            let id = job.get("id").and_then(Value::as_str).unwrap_or_default();
            let state = job.get("state").and_then(Value::as_str).unwrap_or_default();
            matches!(state, "queued" | "dispatching" | "running" | "preempting")
                && (exact_id.is_some_and(|expected| id == expected)
                    || prefixes.iter().any(|prefix| id.starts_with(prefix)))
        })
}

fn disk_used_percent(path: &Path) -> Result<u8> {
    let material = std::ffi::CString::new(path.to_string_lossy().as_bytes())?;
    // Zeroed, not uninit: statvfs holds only integer fields, so all-zero is a valid value and
    // the read below never depends on libc writing every field.
    let mut status = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    // SAFETY: `material` is a CString that outlives the call, so the path is NUL-terminated;
    // `status` is a writable, aligned MaybeUninit<statvfs> that statvfs may overwrite.
    let result = unsafe { libc::statvfs(material.as_ptr(), status.as_mut_ptr()) };
    ensure!(result == 0, "statvfs failed for {}", path.display());
    // SAFETY: `status` started as a valid all-zero statvfs and statvfs only writes integers.
    let status = unsafe { status.assume_init() };
    let blocks = status.f_blocks as u128;
    let available = status.f_bavail as u128;
    let used_percent = blocks
        .saturating_sub(available)
        .saturating_mul(100)
        .checked_div(blocks)
        .unwrap_or(0);
    Ok(used_percent.min(100) as u8)
}

fn dynamic_target(p95_jobs_per_minute: f64, proposer_to_ready_minutes_p95: f64) -> usize {
    MIN_USEFUL_WORK_TARGET
        .max((120.0 * p95_jobs_per_minute.max(0.0)).ceil() as usize)
        .max(
            (1.5 * p95_jobs_per_minute.max(0.0) * proposer_to_ready_minutes_p95.max(0.0)).ceil()
                as usize,
        )
}

fn candidate_dynamic_target(
    p95_candidate_evaluations_per_minute: f64,
    proposer_to_ready_minutes_p95: f64,
) -> usize {
    MIN_CANDIDATE_EVALUATION_WORK_TARGET
        .max((120.0 * p95_candidate_evaluations_per_minute.max(0.0)).ceil() as usize)
        .max(
            (1.5 * p95_candidate_evaluations_per_minute.max(0.0)
                * proposer_to_ready_minutes_p95.max(0.0))
            .ceil() as usize,
        )
}

fn percentile_95(mut values: Vec<f64>) -> f64 {
    values.retain(|value| value.is_finite() && *value > 0.0);
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() * 95).div_ceil(100)).saturating_sub(1);
    values[index]
}

fn modified_ms(path: &Path) -> Option<u128> {
    path.metadata()
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis())
}

fn observed_supply_performance(root: &Path, queue: &Value) -> Value {
    let burn = queue
        .get("p95_60m_jobs_per_minute")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or_default();
    let candidate_burn = queue
        .get("p95_60m_candidate_evaluations_per_minute")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or_default();
    let mut lead_times = Vec::new();
    let states = root.join("ops/cohort-supply-state-v2");
    if let Ok(entries) = fs::read_dir(states) {
        for entry in entries.flatten() {
            let state = optional_json(&entry.path());
            let Some(wave) = state.get("wave").and_then(Value::as_str) else {
                continue;
            };
            let Some(spec) = state
                .pointer("/runtime_pinned_attempts/0/immutable_spec_path")
                .and_then(Value::as_str)
                .map(Path::new)
            else {
                continue;
            };
            let summary = root.join("waves").join(wave).join("summary.json");
            if let (Some(start), Some(ready)) = (modified_ms(spec), modified_ms(&summary)) {
                if ready > start {
                    lead_times.push((ready - start) as f64 / 1_000.0);
                }
            }
        }
    }
    let proposer = percentile_95(lead_times.clone());
    json!({
        "schema": "SupplyPerformance.v1",
        "generated_at_ms": now_ms(),
        "p95_jobs_per_minute": burn,
        "p95_candidate_evaluations_per_minute": candidate_burn,
        "evaluator_burn_per_second_p95": burn / 60.0,
        "proposer_to_ready_minutes_p95": proposer / 60.0,
        "proposer_to_ready_seconds_p95": proposer,
        "proposer_samples": lead_times.len(),
        "source": "queue-keeper p95 burn plus immutable proposer-spec-to-wave-summary timestamps",
    })
}

fn cycle(root: &Path, scq: &Path) -> Result<Value> {
    let now = now_ms();
    let queue = optional_json(&root.join("ops/queue-keeper-v3.json"));
    let supply = optional_json(&root.join("ops/micro-wave-supply-v2.json"));
    let ingest = optional_json(&root.join("ops/last-ingest.json"));
    let corpus = optional_json(&root.join("training/current-corpus.json"));
    let trainer_status = optional_json(&root.join("ops/task-learning-controller.v2.json"));
    let trainer_state = optional_json(&root.join("training/task-learning-controller.v2.json"));
    let encoder = optional_json(&root.join("ops/expansion-driver-v1.json"));
    let gpu = optional_json(&root.join("ops/gpu-lane-admission/status.v1.json"));
    let canary = optional_json(&root.join("ops/canary-executor.v2.json"));
    let pair_repair = optional_json(&root.join("ops/causal-pair-repair-executor.v1.json"));
    let pair_repair_queue = optional_json(&root.join("ops/causal-pair-repair-queue.v1.json"));
    let scheduler = scq_json(scq, &["ps"]).unwrap_or(Value::Null);
    let performance = observed_supply_performance(root, &queue);
    atomic_json(&root.join("ops/supply-performance.v1.json"), &performance)?;

    let queued = queue
        .get("queued")
        .and_then(Value::as_u64)
        .unwrap_or_default() as usize;
    let starting = queue
        .get("starting")
        .and_then(Value::as_u64)
        .unwrap_or_default() as usize;
    let dispatching = queue
        .get("dispatching")
        .and_then(Value::as_u64)
        .unwrap_or_default() as usize;
    let ready = queue
        .get("validated_ready")
        .and_then(Value::as_u64)
        .unwrap_or_default() as usize;
    let burn = performance
        .get("p95_jobs_per_minute")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let candidate_burn = performance
        .get("p95_candidate_evaluations_per_minute")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let proposer_time = performance
        .get("proposer_to_ready_minutes_p95")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let legacy_job_target = dynamic_target(burn, proposer_time);
    let candidate_target = candidate_dynamic_target(candidate_burn, proposer_time);
    let candidate_evaluation_work = queue
        .get("candidate_evaluation_work")
        .and_then(Value::as_u64)
        .unwrap_or_default() as usize;
    let labels = ingest
        .get("exact_rows_total")
        .or_else(|| ingest.get("watermark"))
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let corpus_watermark = corpus
        .get("watermark")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let supply_age = now.saturating_sub(timestamp(&supply));
    let trainer_progress_ms = timestamp(&trainer_status).max(timestamp(&trainer_state));
    let trainer_age = now.saturating_sub(trainer_progress_ms);
    let encoder_age = now.saturating_sub(timestamp(&encoder));
    let queue_age = now.saturating_sub(timestamp(&queue));
    let pending_shadow = count_pending(&root.join("training/shadow-canaries/pending"));
    let pending_production = count_pending(&root.join("training/task-canaries/pending"));
    let canary_age = now.saturating_sub(timestamp(&canary));
    let canary_action = canary
        .pointer("/result/action")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let canary_consuming = matches!(
        canary_action,
        "proposal_job_pending" | "evaluation_jobs_pending" | "deferred_capacity" | "completed"
    ) && canary_age <= STALE_CONTROL_MS;
    let pair_repair_depth = pair_repair_queue
        .get("repairs")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default();
    let pair_repair_age = now.saturating_sub(timestamp(&pair_repair));
    let disk = disk_used_percent(root)?;
    let useful_work = queued
        .saturating_add(starting)
        .saturating_add(dispatching)
        .saturating_add(ready);
    let proposer_eligible = candidate_evaluation_work < candidate_target;
    let proposer_active = supply
        .get("builds_planned")
        .and_then(Value::as_u64)
        .unwrap_or_default()
        > 0
        || supply
            .get("awaiting_keeper_ack")
            .is_some_and(Value::is_string)
        || scheduler_job_active(
            &scheduler,
            None,
            &["jope-selfplay-cohort-propose-", "jope-canary-propose-"],
        );
    // The status document may outlive a controller process and still describe
    // the last submitted intent. The crash-safe state is the authority for
    // whether an intent remains recoverable/active.
    let trainer_intent = trainer_state
        .get("intent")
        .is_some_and(|value| !value.is_null());
    let trainer_job_id = trainer_state
        .pointer("/intent/trainer/immutable_job_identity")
        .and_then(Value::as_str);
    let trainer_active = scheduler_job_active(&scheduler, trainer_job_id, &[]);
    // Eligible when ANY task has corpus the trainer has not yet shadowed.
    // Hard-coding binary/regression here meant a multiclass or timeseries lane
    // could stall invisibly: the corpus grows, no shadow ever trains, and this
    // gauge still reports the trainer as caught up.
    let trainer_eligible = ["binary", "regression", "multiclass", "timeseries"]
        .iter()
        .any(|task| {
            corpus_watermark
                > trainer_state
                    .pointer(&format!("/last_shadow_corpus_watermark/{task}"))
                    .and_then(Value::as_u64)
                    .unwrap_or_default()
        });
    let encoder_eligible = encoder
        .get("remaining")
        .or_else(|| encoder.get("pending_datasets"))
        .and_then(Value::as_u64)
        .unwrap_or_default()
        > 0;
    let encoder_active = scheduler_job_active(&scheduler, None, &["jope-selfplay-breadth-encode-"]);

    let mut alarms = Vec::<String>::new();
    if queue_age > STALE_CONTROL_MS {
        alarms.push("queue_status_stale_over_two_minutes".to_owned());
    }
    if supply_age > STALE_CONTROL_MS
        || !matches!(
            supply.get("schema").and_then(Value::as_str),
            Some("JopeMicroWaveSupplyStatus.v2" | "JopeMicroWaveSupplyError.v1")
        )
    {
        alarms.push("supply_off_schema_or_stale_over_two_minutes".to_owned());
    }
    if trainer_eligible && trainer_age > STALE_CONTROL_MS {
        alarms.push("trainer_progress_stale_over_two_minutes".to_owned());
    }
    if proposer_eligible && !proposer_active && supply_age > IDLE_GPU_MS {
        alarms.push("eligible_proposer_gpu_idle_over_five_minutes".to_owned());
    }
    if trainer_eligible && !trainer_active && trainer_age > IDLE_GPU_MS {
        alarms.push("eligible_trainer_gpu_idle_over_five_minutes".to_owned());
    }
    if encoder_eligible && !encoder_active && encoder_age > IDLE_GPU_MS {
        alarms.push("eligible_encoder_gpu_idle_over_five_minutes".to_owned());
    }
    if candidate_evaluation_work < candidate_target
        || supply.get("target_feasible").and_then(Value::as_bool) != Some(true)
        || supply
            .get("complete_reserve_waves")
            .and_then(Value::as_u64)
            .unwrap_or_default()
            < 2
        || supply.get("performance_fresh").and_then(Value::as_bool) != Some(true)
    {
        alarms.push("queue_plus_ready_reserve_below_projected_low_watermark".to_owned());
    }
    if pending_shadow.saturating_add(pending_production) > 0 && !canary_consuming {
        alarms.push("pending_canary_intents_unconsumed".to_owned());
    }
    if pair_repair_depth > 0 && pair_repair_age > STALE_CONTROL_MS {
        alarms.push("causal_pair_repair_executor_stale_over_two_minutes".to_owned());
    }
    if pair_repair
        .get("critical_identical_failures")
        .and_then(Value::as_array)
        .is_some_and(|failures| !failures.is_empty())
    {
        alarms.push("causal_pair_repair_critical_identical_failures".to_owned());
    }
    if disk >= 90 {
        alarms.push("disk_utilization_at_or_above_90_percent".to_owned());
    }
    let gpu_alarm = gpu.get("alarm").and_then(Value::as_str).map(str::to_owned);
    if let Some(alarm) = gpu_alarm {
        alarms.push(format!("gpu_lane:{alarm}"));
    }

    Ok(json!({
        "schema": "FlywheelLiveness.v1",
        "generated_at_ms": now,
        "queue_depth": queued.saturating_add(starting).saturating_add(dispatching),
        "ready_reserve": ready,
        "useful_work": useful_work,
        "dynamic_target": legacy_job_target,
        "target_formula": "legacy_job_count_only_not_used_for_capacity_decisions",
        "candidate_evaluation_work": candidate_evaluation_work,
        "candidate_evaluation_dynamic_target": candidate_target,
        "candidate_target_formula": "max(262144,ceil(120*p95_candidate_evaluations_per_minute),ceil(1.5*p95_proposer_to_ready_minutes*p95_candidate_evaluations_per_minute))",
        "p95_jobs_per_minute": burn,
        "p95_candidate_evaluations_per_minute": candidate_burn,
        "p95_proposer_to_ready_minutes": proposer_time,
        "ingestion_lag_labels": 0,
        "corpus_lag_labels": labels.saturating_sub(corpus_watermark),
        "pending_canaries": {
            "shadow": pending_shadow,
            "production": pending_production,
            "executor_action": canary_action,
            "executor_last_progress_ms": timestamp(&canary),
        },
        "causal_pair_repairs": {
            "queue_depth": pair_repair_depth,
            "inflight": pair_repair.get("inflight").and_then(Value::as_u64).unwrap_or_default(),
            "last_progress_ms": timestamp(&pair_repair),
            "critical_identical_failures": pair_repair
                .get("critical_identical_failures")
                .cloned()
                .unwrap_or_else(|| json!([])),
        },
        "proposer": {
            "eligible": proposer_eligible,
            "active": proposer_active,
            "last_progress_ms": timestamp(&supply),
        },
        "trainer": {
            "eligible": trainer_eligible,
            "active": trainer_active,
            "recoverable_intent": trainer_intent,
            "last_progress_ms": trainer_progress_ms,
        },
        "encoder": {
            "eligible": encoder_eligible,
            "active": encoder_active,
            "last_progress_ms": timestamp(&encoder),
        },
        "disk_used_percent": disk,
        "alarms": alarms,
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    loop {
        let value = cycle(&args.root, &args.scq).context("build joined flywheel liveness")?;
        atomic_json(&args.root.join("ops/flywheel-liveness.v1.json"), &value)?;
        println!("{}", serde_json::to_string(&value)?);
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_used_percent_reads_a_real_filesystem_and_rejects_a_missing_path() {
        let used = disk_used_percent(&std::env::temp_dir()).unwrap();
        assert!(used <= 100);
        assert!(disk_used_percent(Path::new("/nonexistent/jope-liveness-probe")).is_err());
    }

    #[test]
    fn supply_target_has_runway_floor_and_dynamic_growth() {
        assert_eq!(dynamic_target(0.0, 0.0), 6_144);
        assert_eq!(dynamic_target(10.0, 20.0), 6_144);
        assert_eq!(dynamic_target(60.0, 1.0), 7_200);
        assert_eq!(dynamic_target(100.0, 200.0), 30_000);
    }

    #[test]
    fn p95_is_deterministic_and_ignores_invalid_samples() {
        assert_eq!(percentile_95(vec![]), 0.0);
        assert_eq!(percentile_95(vec![f64::NAN, -1.0, 10.0]), 10.0);
        assert_eq!(percentile_95((1..=100).map(f64::from).collect()), 95.0);
    }

    #[test]
    fn scheduler_activity_joins_exact_and_lane_prefixed_jobs() {
        let scheduler = json!({
            "Jobs": {
                "jobs": [
                    {"id": "jope-selfplay-shadow-binary-1", "state": "running"},
                    {"id": "jope-selfplay-breadth-encode-1", "state": "dispatching"},
                    {"id": "jope-canary-propose-1", "state": "completed"}
                ]
            }
        });
        assert!(scheduler_job_active(
            &scheduler,
            Some("jope-selfplay-shadow-binary-1"),
            &[]
        ));
        assert!(scheduler_job_active(
            &scheduler,
            None,
            &["jope-selfplay-breadth-encode-"]
        ));
        assert!(!scheduler_job_active(
            &scheduler,
            None,
            &["jope-canary-propose-"]
        ));
    }
}
