//! Native, non-scheduling operational materialization loop.
//!
//! This service owns periodic status/telemetry refresh and visibility into
//! terminal promotion candidates. It never consumes panels, schedules work,
//! advances a model pointer, or converts adaptive ratings into learning claims.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::four_task::JopeFourTaskTelemetryV1;
use jope_dime_regressor::recovery_contracts::RecoveryContinuityObservationV1;
use jope_dime_regressor::selfplay_ingest;
use jope_dime_regressor::selfplay_runtime::{atomic_json, default_root_str, now_ms, sha256_file};
use jope_dime_regressor::selfplay_shadow::{
    Task, TaskPromotionEvaluationV3, TaskPromotionObservationSetV3,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const SCHEMA: &str = "JopeReportingStatus.v1";
const RECOVERY_SCHEMA: &str = "JopeRecoveryReport.v2";
const CONTINUITY_SCHEMA: &str = "JopeRecoveryContinuity.v1";
const REQUIRED_GREEN_SECONDS: u64 = 86_400;
const MAXIMUM_OBSERVATION_GAP_SECONDS: u64 = 120;

#[derive(Clone, Copy)]
struct RecoveryComponent {
    name: &'static str,
    path: &'static str,
    schema: &'static str,
    freshness_budget_seconds: u64,
}

const RECOVERY_COMPONENTS: &[RecoveryComponent] = &[
    RecoveryComponent {
        name: "controller",
        path: "ops/task-learning-controller.v2.json",
        schema: "TaskLearningControllerStatus.v2",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "ingestion",
        path: "ops/last-ingest.json",
        schema: "JopeSelfplayIngestReceipt.v1",
        freshness_budget_seconds: 300,
    },
    RecoveryComponent {
        name: "queue",
        path: "ops/queue-keeper-v3.json",
        schema: "JopeSelfplayQueueKeeper.v3",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "supply",
        path: "ops/micro-wave-supply-v2.json",
        schema: "JopeMicroWaveSupplyStatus.v2",
        freshness_budget_seconds: 300,
    },
    RecoveryComponent {
        name: "freezer",
        path: "ops/corpus-freezer-v1.json",
        schema: "JopeSelfplayCorpusFreezer.v1",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "trainer",
        path: "ops/task-learning-controller.v2.json",
        schema: "TaskLearningControllerStatus.v2",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "promotion_executor",
        path: "ops/canary-executor.v2.json",
        schema: "CanaryExecutorStatus.v2",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "repair",
        path: "ops/causal-pair-repair-executor.v1.json",
        schema: "CausalPairRepairExecutor.v1",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "causal_evidence",
        path: "ops/causal-evidence-audit.v2.json",
        schema: "CausalEvidenceAudit.v3",
        freshness_budget_seconds: 900,
    },
    RecoveryComponent {
        name: "freshness",
        path: "ops/freshness-status.json",
        schema: "JopeSelfplayFreshnessStatus.v1",
        freshness_budget_seconds: 31 * 86_400,
    },
    RecoveryComponent {
        name: "liveness_watchdog",
        path: "ops/flywheel-liveness.v1.json",
        schema: "FlywheelLiveness.v1",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "effective_unit_audit",
        path: "ops/effective-unit-audit.v1.json",
        schema: "JopeEffectiveUnitAudit.v1",
        freshness_budget_seconds: 86_400,
    },
];

#[derive(Parser, Debug)]
#[command(name = "jope-reporting")]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value_t = 30)]
    interval_seconds: u64,
    #[arg(long)]
    once: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
struct TerminalLearningInventory {
    terminal_evaluations: usize,
    promotion_cas_receipts: usize,
    immutable_learning_reports: usize,
    unpublished_terminal_candidates: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReportingStatus {
    schema: &'static str,
    generated_at_ms: u128,
    four_task_telemetry_path: PathBuf,
    four_task_telemetry_sha256: String,
    status_path: PathBuf,
    status_sha256: String,
    recovery_report_sha256: Option<String>,
    recovery_continuity_sha256: Option<String>,
    learning: TerminalLearningInventory,
    learning_reports_published: usize,
    scheduling_authority: bool,
    pointer_authority: bool,
    healthy: bool,
    blockers: Vec<String>,
}

fn json_files(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut paths = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn collect_named(directory: &Path, suffix: &str, output: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.filter_map(|entry| entry.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_named(&path, suffix, output);
        } else if path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(suffix))
        {
            output.push(path);
        }
    }
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    )?)
}

fn read_optional_json(path: &Path) -> Value {
    read_json(path).unwrap_or_else(|_| json!({}))
}

fn native_timestamp(value: &Value) -> Option<u128> {
    [
        "heartbeat_at_ms",
        "last_heartbeat_at_ms",
        "generated_at_ms",
        "updated_at_ms",
        "activated_at_ms",
    ]
    .into_iter()
    .find_map(|key| value.get(key).and_then(Value::as_u64).map(u128::from))
}

fn useful_queue_depth(value: &Value) -> u64 {
    value
        .get("useful_work")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            ["queued", "starting", "dispatching", "validated_ready"]
                .into_iter()
                .filter_map(|key| value.get(key).and_then(Value::as_u64))
                .sum()
        })
}

fn recovery_component_check(name: &str, value: &Value) -> (bool, String) {
    let error = value
        .get("error")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    match name {
        "queue" => {
            let fail_closed = value
                .get("fail_closed_reason")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            let healthy = error.is_none()
                && fail_closed.is_none()
                && value.get("admission_allowed").and_then(Value::as_bool) == Some(true)
                && value.get("degraded").and_then(Value::as_bool) != Some(true)
                && useful_queue_depth(value) > 0;
            (
                healthy,
                if healthy {
                    "fresh useful queue depth is nonzero".to_owned()
                } else {
                    error
                        .or(fail_closed)
                        .unwrap_or("queue admission is closed, degraded, or empty")
                        .to_owned()
                },
            )
        }
        "supply" => {
            let blocked = value
                .get("hold_reason")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            let useful_work = value
                .get("useful_work")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let target = value
                .get("useful_work_target")
                .and_then(Value::as_u64)
                .unwrap_or(u64::MAX);
            let healthy = error.is_none()
                && blocked.is_none()
                && value.get("admission_allowed").and_then(Value::as_bool) == Some(true)
                && value.get("performance_fresh").and_then(Value::as_bool) == Some(true)
                && value.get("target_feasible").and_then(Value::as_bool) == Some(true)
                && value
                    .get("complete_reserve_waves")
                    .and_then(Value::as_u64)
                    .unwrap_or_default()
                    >= 2
                && useful_work >= target;
            (
                healthy,
                if healthy {
                    format!("useful work {useful_work} meets target {target}")
                } else {
                    error
                        .or(blocked)
                        .unwrap_or("supply admission is closed")
                        .to_owned()
                },
            )
        }
        "ingestion" => {
            let errors_empty = value.get("errors").is_none_or(|errors| {
                errors.is_null() || errors.as_array().is_some_and(Vec::is_empty)
            });
            let healthy =
                value.get("failures").and_then(Value::as_u64).unwrap_or(0) == 0 && errors_empty;
            (healthy, "ingestion reconcile failure state".to_owned())
        }
        "freezer" => {
            let healthy = error.is_none()
                && value.get("degraded").and_then(Value::as_bool) != Some(true)
                && value.get("timed_out").and_then(Value::as_bool) != Some(true);
            (healthy, "freezer health state".to_owned())
        }
        "controller" => {
            let detail = value
                .get("detail")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            (
                error.is_none() && detail.is_none(),
                error
                    .or(detail)
                    .unwrap_or("controller state is current")
                    .to_owned(),
            )
        }
        "trainer" => {
            let action = value.get("action").and_then(Value::as_str);
            let healthy = matches!(
                action,
                Some(
                    "active_trainer_preserved"
                        | "trainer_submitted"
                        | "gpu_admission_requested"
                        | "trainer_result_harvested"
                )
            );
            (
                healthy,
                action
                    .unwrap_or("trainer state is not proven active or recoverable")
                    .to_owned(),
            )
        }
        "promotion_executor" => {
            let result_error = value
                .get("result")
                .and_then(|result| result.get("error"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty());
            (
                error.is_none() && result_error.is_none(),
                error
                    .or(result_error)
                    .unwrap_or("promotion executor is current")
                    .to_owned(),
            )
        }
        "repair" => {
            let healthy = error.is_none()
                && value
                    .get("permanent_failures_are_imputed")
                    .and_then(Value::as_bool)
                    == Some(false);
            (
                healthy,
                "repair evidence must remain non-imputed".to_owned(),
            )
        }
        "causal_evidence" => {
            let healthy = error.is_none()
                && value.get("authoritative").and_then(Value::as_bool) == Some(true);
            (healthy, "causal evidence authority state".to_owned())
        }
        "freshness" => {
            let healthy = value.get("healthy").and_then(Value::as_bool) == Some(true)
                && value
                    .get("stale_epoch_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    == 0
                && value.get("admission_paused").and_then(Value::as_bool) != Some(true);
            (healthy, "freshness authority state".to_owned())
        }
        "liveness_watchdog" => {
            let healthy = value
                .get("alarms")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty);
            (healthy, "liveness alarm state".to_owned())
        }
        "effective_unit_audit" => {
            let healthy = value.get("healthy").and_then(Value::as_bool) == Some(true)
                && value
                    .get("violations")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty);
            (healthy, "effective unit inventory state".to_owned())
        }
        _ => (true, "native document is current".to_owned()),
    }
}

fn materialize_recovery(root: &Path, at_ms: u128) -> Result<Value> {
    let mut components = serde_json::Map::new();
    let mut gates = serde_json::Map::new();
    for component in RECOVERY_COMPONENTS {
        let path = root.join(component.path);
        let value = read_optional_json(&path);
        let present = value.as_object().is_some_and(|object| !object.is_empty());
        let schema_valid = value.get("schema").and_then(Value::as_str) == Some(component.schema);
        let timestamp = native_timestamp(&value);
        let age_seconds = timestamp.map(|timestamp| at_ms.saturating_sub(timestamp) / 1_000);
        let fresh =
            age_seconds.is_some_and(|age| age <= u128::from(component.freshness_budget_seconds));
        let (component_healthy, check_reason) = if present && schema_valid {
            recovery_component_check(component.name, &value)
        } else {
            (false, String::new())
        };
        let healthy = present && schema_valid && fresh && component_healthy;
        let reason = if !present {
            "native component document is missing".to_owned()
        } else if !schema_valid {
            format!(
                "schema mismatch: expected {}, got {}",
                component.schema,
                value.get("schema").and_then(Value::as_str).unwrap_or("N/A")
            )
        } else if timestamp.is_none() {
            "native component document has no timestamp".to_owned()
        } else if !fresh {
            format!(
                "native component document is stale by {}s",
                age_seconds.unwrap_or(0)
            )
        } else {
            check_reason
        };
        gates.insert(component.name.to_owned(), json!(healthy));
        components.insert(
            component.name.to_owned(),
            json!({
                "path": component.path,
                "schema": value.get("schema"),
                "sha256": path.is_file().then(|| sha256_file(&path)).transpose()?,
                "timestamp_ms": timestamp,
                "age_seconds": age_seconds,
                "freshness_budget_seconds": component.freshness_budget_seconds,
                "fresh": fresh,
                "healthy": healthy,
                "reason": reason,
            }),
        );
    }
    let queue = read_optional_json(&root.join("ops/queue-keeper-v3.json"));
    let queue_depth = useful_queue_depth(&queue);
    let gates_green = gates.values().all(|value| value.as_bool() == Some(true));
    let queue_fresh = components
        .get("queue")
        .and_then(|value| value.get("fresh"))
        .and_then(Value::as_bool)
        == Some(true);
    let observation_path = root.join("ops/recovery-continuity-observation.v1.json");
    let observation_value = read_optional_json(&observation_path);
    let observation =
        serde_json::from_value::<RecoveryContinuityObservationV1>(observation_value.clone());
    let now_ms = at_ms.min(u128::from(u64::MAX)) as u64;
    let (observation_green, observation_reason, observation_epoch, observation_inputs) =
        match observation {
            Ok(observation) => {
                let validated = observation
                    .validate_at(now_ms)
                    .and_then(|()| verify_continuity_observation(root, &observation, at_ms))
                    .and_then(|()| observation.epoch_identity_sha256(now_ms));
                match validated {
                    Ok(epoch) => (
                        true,
                        "fresh digest-complete continuity observation is green".to_owned(),
                        Some(epoch),
                        observation.input_digests,
                    ),
                    Err(error) => (
                        false,
                        format!("continuity observation rejected: {error:#}"),
                        None,
                        observation.input_digests,
                    ),
                }
            }
            Err(error) => (
                false,
                format!("continuity observation is absent or malformed: {error}"),
                None,
                Vec::new(),
            ),
        };
    gates.insert(
        "continuity_observation".to_owned(),
        json!(observation_green),
    );
    components.insert(
        "continuity_observation".to_owned(),
        json!({
            "path": "ops/recovery-continuity-observation.v1.json",
            "schema": observation_value.get("schema"),
            "sha256": observation_path.is_file().then(|| sha256_file(&observation_path)).transpose()?,
            "healthy": observation_green,
            "reason": observation_reason,
        }),
    );
    let green = gates_green && queue_fresh && queue_depth > 0 && observation_green;
    let continuity_path = root.join("ops/recovery-continuity.v1.json");
    let previous = read_optional_json(&continuity_path);
    let previous_observed = previous.get("last_observed_at_ms").and_then(Value::as_u64);
    let gap_ok = previous_observed.is_some_and(|previous| {
        at_ms.saturating_sub(u128::from(previous))
            <= u128::from(MAXIMUM_OBSERVATION_GAP_SECONDS) * 1_000
    });
    let continuing = previous.get("schema").and_then(Value::as_str) == Some(CONTINUITY_SCHEMA)
        && previous
            .get("last_observation_green")
            .and_then(Value::as_bool)
            == Some(true)
        && previous
            .get("green_since_ms")
            .and_then(Value::as_u64)
            .is_some()
        && previous
            .get("epoch_identity_sha256")
            .and_then(Value::as_str)
            == observation_epoch.as_deref()
        && gap_ok
        && green;
    let green_since = if green {
        if continuing {
            previous.get("green_since_ms").and_then(Value::as_u64)
        } else {
            Some(at_ms.min(u128::from(u64::MAX)) as u64)
        }
    } else {
        None
    };
    let continuously_green_seconds = green_since
        .map(|since| at_ms.saturating_sub(u128::from(since)) / 1_000)
        .unwrap_or(0);
    let release_ready = green && continuously_green_seconds >= u128::from(REQUIRED_GREEN_SECONDS);
    let reset_reason = if green {
        (!continuing && previous.get("green_since_ms").is_some())
            .then_some("continuity epoch identity changed; a new uninterrupted epoch started")
    } else {
        Some(observation_reason.as_str())
    };
    let continuity = json!({
        "schema": CONTINUITY_SCHEMA,
        "generated_at_ms": at_ms,
        "freshness_budget_seconds": MAXIMUM_OBSERVATION_GAP_SECONDS,
        "input_digests": observation_inputs,
        "blockers": (!green).then_some(vec![observation_reason.clone()]).unwrap_or_default(),
        "epoch_identity_sha256": observation_epoch,
        "green_since_ms": green_since,
        "last_observed_at_ms": at_ms,
        "last_observation_green": green,
        "last_useful_queue_nonzero_at_ms": if green {
            Some(at_ms)
        } else {
            previous
                .get("last_useful_queue_nonzero_at_ms")
                .and_then(Value::as_u64)
                .map(u128::from)
        },
        "maximum_observation_gap_seconds": MAXIMUM_OBSERVATION_GAP_SECONDS,
        "required_green_seconds": REQUIRED_GREEN_SECONDS,
        "continuously_green_seconds": continuously_green_seconds,
        "release_ready": release_ready,
        "reset_reason": reset_reason,
    });
    atomic_json(&continuity_path, &continuity)?;
    let telemetry = read_optional_json(&root.join("ops/four-task-telemetry.v1.json"));
    let report = json!({
        "schema": RECOVERY_SCHEMA,
        "generated_at_ms": at_ms,
        "healthy": green,
        "release_gate": if release_ready { "ready" } else if green { "warming" } else { "blocked" },
        "gates": gates,
        "components": components,
        "queue": {
            "useful_depth": queue_depth,
            "queued": queue.get("queued").and_then(Value::as_u64).unwrap_or(0),
            "dispatching": queue.get("dispatching").and_then(Value::as_u64).unwrap_or(0),
            "running": queue.get("running").and_then(Value::as_u64).unwrap_or(0),
            "launch_ready": queue.get("launch_ready").and_then(Value::as_u64).unwrap_or(0),
            "fresh": queue_fresh,
        },
        "tasks": telemetry.get("rows").cloned(),
        "continuity": continuity,
        "claims_policy": {
            "corpus_growth_is_not_improvement": true,
            "completed_training_is_not_improvement": true,
            "test_adaptive_evidence_is_benchmark_invalid": true,
            "accepted_improvement_requires_three_family_disjoint_single_use_panels": true,
        },
    });
    atomic_json(&root.join("ops/recovery-report.v2.json"), &report)?;
    Ok(report)
}

fn verify_continuity_observation(
    root: &Path,
    observation: &RecoveryContinuityObservationV1,
    at_ms: u128,
) -> Result<()> {
    let unit_audit_path = root.join("ops/effective-unit-audit.v1.json");
    let unit_audit = read_json(&unit_audit_path)?;
    ensure!(
        unit_audit.get("bundle_sha256").and_then(Value::as_str)
            == Some(observation.runtime_bundle_sha256.as_str()),
        "effective-unit audit runtime bundle differs from continuity observation"
    );
    let expansion_inactive = unit_audit
        .get("units")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter().find(|row| {
                row.get("unit").and_then(Value::as_str) == Some("jope-selfplay-expansion.service")
            })
        })
        .is_some_and(|row| {
            row.get("active_state").and_then(Value::as_str) != Some("active")
                && row.get("main_pid").and_then(Value::as_u64).unwrap_or(0) == 0
        });
    ensure!(
        expansion_inactive,
        "effective-unit audit does not prove expansion stopped and disabled"
    );

    let telemetry_path = root.join("ops/four-task-telemetry.v1.json");
    ensure!(
        sha256_file(&telemetry_path)? == observation.telemetry_sha256,
        "four-task telemetry digest differs from continuity observation"
    );
    let telemetry: JopeFourTaskTelemetryV1 = serde_json::from_value(read_json(&telemetry_path)?)?;
    telemetry.validate()?;
    ensure!(
        telemetry.generated_at_ms <= at_ms
            && at_ms.saturating_sub(telemetry.generated_at_ms)
                <= u128::from(observation.freshness_budget_seconds) * 1_000,
        "four-task telemetry is stale"
    );

    let causal_path = root.join("ops/causal-evidence-audit.v2.json");
    ensure!(
        sha256_file(&causal_path)? == observation.causal_evidence_sha256,
        "causal evidence digest differs from continuity observation"
    );
    let supply = read_json(&root.join("ops/micro-wave-supply-v2.json"))?;
    ensure!(
        supply.get("useful_work").and_then(Value::as_u64) == Some(observation.useful_work)
            && supply.get("useful_work_target").and_then(Value::as_u64)
                == Some(observation.useful_work_target)
            && supply.get("complete_reserve_waves").and_then(Value::as_u64)
                == Some(observation.protected_complete_waves),
        "supply measurements differ from continuity observation"
    );
    ensure!(
        sha256_file(&root.join("training/accepted-by-task.json"))?
            == observation.accepted_model_pointer_sha256,
        "accepted-model pointer changed or is not directly observed"
    );
    Ok(())
}

fn task_name(task: Task) -> &'static str {
    task.as_str()
}

fn promotion_cas_valid(root: &Path, task: Task, checkpoint: &str) -> Result<Option<PathBuf>> {
    let path = root
        .join("training/task-promotions")
        .join(format!("{}-{checkpoint}.v1.json", task_name(task)));
    if !path.is_file() {
        return Ok(None);
    }
    let receipt = read_json(&path)?;
    let valid = receipt.get("schema").and_then(Value::as_str) == Some("TaskCheckpointPromotion.v2")
        && receipt.get("task").and_then(Value::as_str) == Some(task_name(task))
        && receipt
            .get("child_checkpoint_sha256")
            .and_then(Value::as_str)
            == Some(checkpoint)
        && receipt
            .get("independent_single_use_panel_acceptances")
            .and_then(Value::as_u64)
            == Some(3)
        && receipt
            .get("external_pass_receipt_ids")
            .and_then(Value::as_array)
            .is_some_and(|rows| rows.len() == 3);
    ensure!(valid, "task promotion CAS receipt contract drift");
    let pointer = read_json(&root.join("training/accepted-by-task.json"))?;
    ensure!(
        pointer
            .get("tasks")
            .and_then(|tasks| tasks.get(task_name(task)))
            .and_then(|row| row.get("checkpoint_sha256"))
            .and_then(Value::as_str)
            == Some(checkpoint),
        "promotion CAS receipt is not reflected by the accepted task pointer"
    );
    Ok(Some(path))
}

fn observation_sets(
    root: &Path,
    panels: &BTreeSet<String>,
) -> Result<BTreeMap<String, (TaskPromotionObservationSetV3, PathBuf)>> {
    let mut paths = Vec::new();
    collect_named(
        &root.join("training/task-promotion-evaluations/work"),
        ".observations.v3.json",
        &mut paths,
    );
    paths.sort();
    let mut output = BTreeMap::new();
    for path in paths {
        let value: TaskPromotionObservationSetV3 = serde_json::from_value(read_json(&path)?)?;
        value.validate()?;
        let panel = value.panel_consumption_sha256.as_str().to_owned();
        if !panels.contains(&panel) {
            continue;
        }
        let replace = output.get(&panel).is_none_or(
            |(prior, _): &(TaskPromotionObservationSetV3, PathBuf)| {
                value.observations.len() > prior.observations.len()
            },
        );
        if replace {
            output.insert(panel, (value, path));
        }
    }
    Ok(output)
}

fn immutable_learning_report(
    root: &Path,
    task: Task,
    checkpoint: &str,
    evaluations: Vec<(TaskPromotionEvaluationV3, PathBuf)>,
) -> Result<Option<PathBuf>> {
    let mut by_run = BTreeMap::new();
    for (evaluation, path) in evaluations {
        evaluation.validate()?;
        ensure!(
            evaluation.task == task
                && evaluation.candidate_checkpoint_sha256.as_str() == checkpoint,
            "terminal evaluation scope drift"
        );
        if let Some((prior, _)) = by_run.insert(evaluation.run_index, (evaluation.clone(), path)) {
            ensure!(
                prior == evaluation,
                "conflicting terminal evaluation for one run"
            );
        }
    }
    let rejected = by_run.values().any(|(evaluation, _)| !evaluation.accepted);
    let cas = promotion_cas_valid(root, task, checkpoint)?;
    let three_accepted = by_run.len() == 3
        && (0..3).all(|run| by_run.get(&run).is_some_and(|(row, _)| row.accepted));
    if !rejected && !(three_accepted && cas.is_some()) {
        return Ok(None);
    }
    let panels = by_run
        .values()
        .map(|(row, _)| row.panel_consumption_sha256.as_str().to_owned())
        .collect::<BTreeSet<_>>();
    let observations = observation_sets(root, &panels)?;
    let mut context_panel = BTreeMap::new();
    let mut family_panel = BTreeMap::new();
    let mut complete_runs = Vec::new();
    let mut source_paths = by_run
        .values()
        .map(|(_, path)| path.clone())
        .collect::<Vec<_>>();
    let mut total_contexts = 0_usize;
    let mut valid = 0_usize;
    let mut replay = 0_usize;
    let mut failures = 0_usize;
    let mut latencies = Vec::new();
    let mut strict_wins = 0_usize;
    let mut family_lifts = BTreeMap::<String, Vec<f64>>::new();
    let mut family_disjoint = panels.len() == by_run.len();
    for (run, (evaluation, _)) in &by_run {
        let panel = evaluation.panel_consumption_sha256.as_str();
        let Some((set, path)) = observations.get(panel) else {
            continue;
        };
        if set.observations.len() != evaluation.metrics.contexts
            || evaluation.metrics.replay_verification != 1.0
            || !set.observations.iter().all(|row| row.replay_verified)
        {
            continue;
        }
        source_paths.push(path.clone());
        complete_runs.push(*run);
        for row in &set.observations {
            family_disjoint &= context_panel
                .insert(row.context_sha256.as_str().to_owned(), panel.to_owned())
                .is_none();
            if let Some(prior) = family_panel.insert(
                row.source_family_sha256.as_str().to_owned(),
                panel.to_owned(),
            ) {
                family_disjoint &= prior == panel;
            }
            total_contexts += 1;
            valid += usize::from(row.valid);
            replay += usize::from(row.replay_verified);
            failures += usize::from(row.candidate_failed);
            latencies.push(row.candidate_latency_ms);
            strict_wins += usize::from(row.strict_win);
            family_lifts
                .entry(row.source_family_sha256.as_str().to_owned())
                .or_default()
                .push(row.candidate_score - row.accepted_score);
        }
    }
    complete_runs.sort_unstable();
    let complete_family_disjoint = family_disjoint && !complete_runs.is_empty();
    let hard_regression = by_run.values().any(|(evaluation, _)| {
        complete_runs.contains(&evaluation.run_index)
            && (evaluation.metrics.confidence.candidate_vs_accepted_lower < 0.0
                || evaluation.metrics.maximum_slice_regression > 0.005
                || evaluation.metrics.validity < 0.99
                || evaluation.metrics.failure_rate_increase > 0.0025
                || evaluation.metrics.latency_ratio > 1.25)
    });
    let cas_succeeded = cas.is_some();
    let improved =
        three_accepted && cas_succeeded && complete_runs == vec![0, 1, 2] && family_disjoint;
    let disposition = if improved {
        "improved"
    } else if complete_family_disjoint && hard_regression {
        "regressed"
    } else {
        "inconclusive"
    };
    if let Some(path) = cas {
        source_paths.push(path);
        source_paths.push(root.join("training/accepted-by-task.json"));
    }
    source_paths.sort();
    source_paths.dedup();
    latencies.sort_unstable();
    let p95_latency_ms = (!latencies.is_empty())
        .then(|| latencies[(latencies.len() * 95).div_ceil(100).saturating_sub(1)]);
    let family_means = family_lifts
        .values()
        .filter(|rows| !rows.is_empty())
        .map(|rows| rows.iter().sum::<f64>() / rows.len() as f64)
        .collect::<Vec<_>>();
    let sealed_panel_lift = (!family_means.is_empty())
        .then(|| family_means.iter().sum::<f64>() / family_means.len() as f64);
    let generated_at_ms = by_run
        .values()
        .map(|(row, _)| row.evaluated_at_ms)
        .max()
        .unwrap_or(0);
    let source_artifacts = source_paths
        .iter()
        .map(|path| {
            Ok(json!({
                "path": path,
                "sha256": sha256_file(path)?,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let report = json!({
        "schema": "JopeLearningReport.v2",
        "publisher": "jope_reporting.v1",
        "immutable": true,
        "checkpoint_sha256": checkpoint,
        "task": task,
        "generated_at_ms": generated_at_ms,
        "disposition": disposition,
        "promotion_disposition": {
            "evaluation_runs": by_run.iter().map(|(run, (row, _))| json!([run, row.accepted])).collect::<Vec<_>>(),
            "rejection_reasons": by_run.values().flat_map(|(row, _)| row.rejection_reasons.clone()).collect::<BTreeSet<_>>(),
            "pointer_compare_and_swap_succeeded": cas_succeeded,
        },
        "metrics": {
            "proposal_validity": (total_contexts > 0).then(|| valid as f64 / total_contexts as f64),
            "matched_strict_win_rate": (total_contexts > 0).then(|| strict_wins as f64 / total_contexts as f64),
            "p95_latency_ms": p95_latency_ms,
            "failure_rate": (total_contexts > 0).then(|| failures as f64 / total_contexts as f64),
            "replay_coverage": (total_contexts > 0).then(|| replay as f64 / total_contexts as f64),
            "sealed_panel_lift": sealed_panel_lift,
            "panel_coverage": panels.len(),
            "run_coverage": by_run.keys().copied().collect::<Vec<_>>(),
            "observation_run_coverage": complete_runs,
            "context_count": total_contexts,
            "source_family_count": family_panel.len(),
            "family_disjoint": family_disjoint,
            "gp_relative_lift": Value::Null,
            "causal_metrics_na_reason": "no complete authoritative causal-pair projection supplied to the reporter",
        },
        "source_artifacts": source_artifacts,
        "claims_policy": {
            "corpus_growth_is_not_improvement": true,
            "completed_training_is_not_improvement": true,
            "test_adaptive_evidence_is_benchmark_invalid": true,
            "family_disjoint_replay_is_required": true,
            "caller_supplied_metric_json_is_forbidden": true,
        },
    });
    let output = root
        .join("ops/learning-reports")
        .join(task_name(task))
        .join(format!("{checkpoint}.v2.json"));
    if output.is_file() {
        let existing = read_json(&output)?;
        if existing.get("publisher").and_then(Value::as_str) == Some("jope_reporting.v1") {
            ensure!(
                existing == report,
                "immutable learning report content collision"
            );
        } else {
            ensure!(
                existing.get("schema").and_then(Value::as_str) == Some("JopeLearningReport.v2")
                    && existing.get("task").and_then(Value::as_str) == Some(task_name(task))
                    && existing.get("checkpoint_sha256").and_then(Value::as_str)
                        == Some(checkpoint),
                "pre-existing immutable learning report scope collision"
            );
        }
    } else {
        atomic_json(&output, &report)?;
    }
    Ok(Some(output))
}

fn publish_terminal_learning_reports(root: &Path) -> Result<usize> {
    let mut grouped = BTreeMap::<(Task, String), Vec<(TaskPromotionEvaluationV3, PathBuf)>>::new();
    for tier in ["accepted", "rejected"] {
        for path in json_files(&root.join("training/task-promotion-evaluations").join(tier)) {
            let evaluation: TaskPromotionEvaluationV3 = serde_json::from_value(read_json(&path)?)?;
            evaluation.validate()?;
            grouped
                .entry((
                    evaluation.task,
                    evaluation.candidate_checkpoint_sha256.as_str().to_owned(),
                ))
                .or_default()
                .push((evaluation, path));
        }
    }
    let mut published = 0;
    for ((task, checkpoint), evaluations) in grouped {
        let output = root
            .join("ops/learning-reports")
            .join(task_name(task))
            .join(format!("{checkpoint}.v2.json"));
        let existed = output.is_file();
        if immutable_learning_report(root, task, &checkpoint, evaluations)?.is_some() && !existed {
            published += 1;
        }
    }
    Ok(published)
}

fn terminal_learning_inventory(root: &Path) -> TerminalLearningInventory {
    let mut inventory = TerminalLearningInventory::default();
    let mut terminal = Vec::<(String, String)>::new();
    for tier in ["accepted", "rejected"] {
        for path in json_files(&root.join("training/task-promotion-evaluations").join(tier)) {
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
                continue;
            };
            if value.get("schema").and_then(Value::as_str) != Some("TaskPromotionEvaluation.v3") {
                continue;
            }
            let Some(task) = value.get("task").and_then(Value::as_str) else {
                continue;
            };
            let Some(checkpoint) = value
                .get("candidate_checkpoint_sha256")
                .and_then(Value::as_str)
            else {
                continue;
            };
            inventory.terminal_evaluations += 1;
            terminal.push((task.to_owned(), checkpoint.to_owned()));
        }
    }
    inventory.promotion_cas_receipts = json_files(&root.join("training/task-promotions")).len();
    for task in ["regression", "binary", "multiclass", "timeseries"] {
        inventory.immutable_learning_reports +=
            json_files(&root.join("ops/learning-reports").join(task)).len();
    }
    terminal.sort();
    terminal.dedup();
    inventory.unpublished_terminal_candidates = terminal
        .into_iter()
        .filter_map(|(task, checkpoint)| {
            (!root
                .join("ops/learning-reports")
                .join(&task)
                .join(format!("{checkpoint}.v2.json"))
                .is_file())
            .then(|| format!("{task}:{checkpoint}"))
        })
        .collect();
    inventory
}

fn refresh(root: &Path) -> Result<ReportingStatus> {
    let mut status = selfplay_ingest::build_status(root)?;
    let generated_at_ms = now_ms();
    let recovery = materialize_recovery(root, generated_at_ms)?;
    let release_ready = recovery.get("release_gate").and_then(Value::as_str) == Some("ready");
    let failed_gates = recovery
        .get("gates")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(name, value)| (value.as_bool() != Some(true)).then(|| name.clone()))
        .collect::<Vec<_>>();
    status["release_gate"] = json!({
        "healthy": release_ready,
        "blocking_gates": failed_gates,
        "report": "ops/recovery-report.v2.json",
    });
    atomic_json(&root.join("ops/status-v3.json"), &status)?;
    atomic_json(&root.join("ops/status.json"), &status)?;
    let telemetry_path = root.join("ops/four-task-telemetry.v1.json");
    let status_path = root.join("ops/status-v3.json");
    let learning_reports_published = publish_terminal_learning_reports(root)?;
    let learning = terminal_learning_inventory(root);
    let mut blockers = Vec::new();
    let recovery_path = root.join("ops/recovery-report.v2.json");
    let continuity_path = root.join("ops/recovery-continuity.v1.json");
    let recovery = fs::read(&recovery_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    let continuity = fs::read(&continuity_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    if recovery.as_ref().is_none_or(|value| {
        value.get("schema").and_then(Value::as_str) != Some("JopeRecoveryReport.v2")
            || value.get("healthy").and_then(Value::as_bool) != Some(true)
            || value.get("release_gate").and_then(Value::as_str) != Some("ready")
    }) {
        blockers.push("recovery report is absent, invalid, or not ready".to_owned());
    }
    if continuity.as_ref().is_none_or(|value| {
        value.get("schema").and_then(Value::as_str) != Some("JopeRecoveryContinuity.v1")
            || value.get("release_ready").and_then(Value::as_bool) != Some(true)
    }) {
        blockers.push("86,400-second recovery continuity is not ready".to_owned());
    }
    if !learning.unpublished_terminal_candidates.is_empty() {
        blockers.push("terminal candidates await immutable JopeLearningReport.v2".to_owned());
    }
    let report = ReportingStatus {
        schema: SCHEMA,
        generated_at_ms,
        four_task_telemetry_sha256: sha256_file(&telemetry_path)?,
        status_sha256: sha256_file(&status_path)?,
        recovery_report_sha256: recovery_path
            .is_file()
            .then(|| sha256_file(&recovery_path))
            .transpose()?,
        recovery_continuity_sha256: continuity_path
            .is_file()
            .then(|| sha256_file(&continuity_path))
            .transpose()?,
        four_task_telemetry_path: telemetry_path,
        status_path,
        learning,
        learning_reports_published,
        scheduling_authority: false,
        pointer_authority: false,
        healthy: blockers.is_empty(),
        blockers,
    };
    atomic_json(
        &root.join("ops/reporting-status.v1.json"),
        &serde_json::to_value(&report)?,
    )?;
    Ok(report)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.interval_seconds > 0 && args.interval_seconds <= 300,
        "reporting interval must be within 1..=300 seconds"
    );
    loop {
        let report = refresh(&args.root)?;
        println!("{}", serde_json::to_string(&report)?);
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_dime_regressor::selfplay_runtime::atomic_json;
    use serde_json::json;
    use tempfile::tempdir;

    fn write_green_recovery_components(root: &Path, at_ms: u128) {
        for component in RECOVERY_COMPONENTS {
            let mut value = json!({
                "schema": component.schema,
                "generated_at_ms": at_ms,
            });
            let object = value.as_object_mut().unwrap();
            match component.name {
                "controller" | "trainer" => {
                    object.insert("action".to_owned(), json!("active_trainer_preserved"));
                }
                "ingestion" => {
                    object.insert("failures".to_owned(), json!(0));
                    object.insert("errors".to_owned(), json!([]));
                }
                "queue" => {
                    object.insert("admission_allowed".to_owned(), json!(true));
                    object.insert("useful_work".to_owned(), json!(10));
                    object.insert("queued".to_owned(), json!(10));
                }
                "supply" => {
                    object.insert("admission_allowed".to_owned(), json!(true));
                    object.insert("performance_fresh".to_owned(), json!(true));
                    object.insert("target_feasible".to_owned(), json!(true));
                    object.insert("complete_reserve_waves".to_owned(), json!(2));
                    object.insert("useful_work".to_owned(), json!(6_144));
                    object.insert("useful_work_target".to_owned(), json!(6_144));
                }
                "promotion_executor" => {
                    object.insert("result".to_owned(), json!({"action":"idle"}));
                }
                "repair" => {
                    object.insert("permanent_failures_are_imputed".to_owned(), json!(false));
                }
                "causal_evidence" => {
                    object.insert("authoritative".to_owned(), json!(true));
                }
                "freshness" => {
                    object.insert("healthy".to_owned(), json!(true));
                    object.insert("stale_epoch_count".to_owned(), json!(0));
                    object.insert("admission_paused".to_owned(), json!(false));
                }
                "liveness_watchdog" => {
                    object.insert("alarms".to_owned(), json!([]));
                }
                "effective_unit_audit" => {
                    object.insert("healthy".to_owned(), json!(true));
                    object.insert("violations".to_owned(), json!([]));
                    object.insert("bundle_sha256".to_owned(), json!("a".repeat(64)));
                    object.insert(
                        "units".to_owned(),
                        json!([{
                            "unit": "jope-selfplay-expansion.service",
                            "active_state": "inactive",
                            "main_pid": 0,
                        }]),
                    );
                }
                _ => {}
            }
            atomic_json(&root.join(component.path), &value).unwrap();
        }
        let typed_na = || json!({"na_reason": "not observed in continuity fixture"});
        let count = || json!({"value": 0});
        let rows = [
            ("regression", "r2"),
            ("binary", "auc"),
            ("multiclass", "macro_f1"),
            ("timeseries", "negative_mase"),
        ]
        .into_iter()
        .map(|(task, primary_metric)| {
            json!({
                "task": task,
                "counts": {
                    "catalog_entries": count(), "source_families": count(),
                    "encoded": count(), "active": count(), "observed": count(),
                    "queued": count(), "running": count(), "completed": count(),
                    "failed": count(), "quarantined": count(),
                },
                "latest_observation_at_ms": typed_na(),
                "valid_observations_per_minute": typed_na(),
                "evaluator_failures_per_minute": typed_na(),
                "freshness_reason": "typed absence",
                "metrics": {
                    "primary_metric": primary_metric,
                    "primary_performance": typed_na(),
                    "native_errors": {},
                    "bounded_utility": typed_na(),
                    "operational_loss": typed_na(),
                },
                "accepted_checkpoint": typed_na(),
                "bootstrap_anchor_state": "absent",
                "latest_shadow": typed_na(),
                "active_candidate": typed_na(),
                "eligibility": {
                    "evidence_eligible": false,
                    "training_eligible": false,
                    "promotion_eligible": false,
                    "blockers": ["fixture_absence"],
                },
                "adaptive_rating": typed_na(),
                "diagnostic_rating": typed_na(),
                "sealed_promotion": typed_na(),
                "trainer_validation": {"metrics": {}, "age_seconds": typed_na()},
                "sources": [],
            })
        })
        .collect::<Vec<_>>();
        atomic_json(
            &root.join("ops/four-task-telemetry.v1.json"),
            &json!({
                "schema": "JopeFourTaskTelemetry.v1",
                "generated_at_ms": at_ms,
                "corpus_watermark": count(),
                "rows": rows,
            }),
        )
        .unwrap();
        atomic_json(
            &root.join("training/accepted-by-task.json"),
            &json!({"schema": "JopeAcceptedByTask.v1", "tasks": {}}),
        )
        .unwrap();
        atomic_json(
            &root.join("ops/reporting-status.v1.json"),
            &json!({"schema": "JopeReportingStatus.v1", "generated_at_ms": at_ms}),
        )
        .unwrap();
        write_green_continuity_observation(root, at_ms);
    }

    fn write_green_continuity_observation(root: &Path, at_ms: u128) {
        let runtime = "a".repeat(64);
        let accepted = sha256_file(&root.join("training/accepted-by-task.json")).unwrap();
        let causal = sha256_file(&root.join("ops/causal-evidence-audit.v2.json")).unwrap();
        let telemetry = sha256_file(&root.join("ops/four-task-telemetry.v1.json")).unwrap();
        let service_paths = [
            ("freezer", "ops/corpus-freezer-v1.json"),
            ("freshness", "ops/freshness-status.json"),
            ("ingestion", "ops/last-ingest.json"),
            ("liveness", "ops/flywheel-liveness.v1.json"),
            ("queue", "ops/queue-keeper-v3.json"),
            ("reporting", "ops/reporting-status.v1.json"),
            ("supply", "ops/micro-wave-supply-v2.json"),
        ];
        let services = service_paths
            .iter()
            .map(|(service, path)| {
                json!({
                    "service": service,
                    "status_sha256": sha256_file(&root.join(path)).unwrap(),
                    "fresh": true,
                })
            })
            .collect::<Vec<_>>();
        let nodes = ["xbabe0", "xbabe1", "xbabe2", "xbabe3", "xbabe4"]
            .into_iter()
            .enumerate()
            .map(|(index, node)| {
                json!({
                    "node": node,
                    "receipt_sha256": format!("{:064x}", 100 + index),
                    "runtime_bundle_sha256": runtime,
                    "active_process_count": 1,
                    "verified": true,
                })
            })
            .collect::<Vec<_>>();
        let incident = format!("{:064x}", 1);
        let quarantine = format!("{:064x}", 2);
        let model = format!("{:064x}", 3);
        let held = format!("{:064x}", 4);
        let v1_inventory = format!("{:064x}", 5);
        let mut inputs = BTreeSet::from([
            incident.clone(),
            quarantine.clone(),
            runtime.clone(),
            causal.clone(),
            telemetry.clone(),
            accepted.clone(),
            model.clone(),
            held.clone(),
            v1_inventory.clone(),
        ]);
        for row in &nodes {
            inputs.insert(row["receipt_sha256"].as_str().unwrap().to_owned());
        }
        for row in &services {
            inputs.insert(row["status_sha256"].as_str().unwrap().to_owned());
        }
        atomic_json(
            &root.join("ops/recovery-continuity-observation.v1.json"),
            &json!({
                "schema": "JopeRecoveryContinuityObservation.v1",
                "generated_at_unix_ms": at_ms,
                "freshness_budget_seconds": 120,
                "input_digests": inputs,
                "incident_containment_sha256": incident,
                "quarantine_index_sha256": quarantine,
                "runtime_bundle_sha256": runtime,
                "nodes": nodes,
                "active_process_bundle_sha256": vec!["a".repeat(64); 5],
                "causal_evidence_sha256": causal,
                "causal_success": true,
                "telemetry_sha256": telemetry,
                "telemetry_rows_exact_and_fresh": true,
                "useful_work": 6_144,
                "useful_work_target": 6_144,
                "protected_complete_waves": 2,
                "services": services,
                "accepted_model_pointer_sha256": accepted,
                "model_pointer_sha256": model,
                "held_job_inventory_sha256": held,
                "held_job_count": 3,
                "ownerless_hold_count": 0,
                "evaluation_cancellation_count_since_epoch": 0,
                "v1_output_inventory_sha256": v1_inventory,
                "new_v1_output_count_since_epoch": 0,
                "v1_quarantine_consumption_count_since_epoch": 0,
                "expansion_active": false,
                "blockers": [],
            }),
        )
        .unwrap();
    }

    #[test]
    fn recovery_continuity_requires_86400_seconds_and_resets_on_queue_loss() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let at_ms = 100_000_000_u128;
        write_green_recovery_components(root, at_ms);
        let warming = materialize_recovery(root, at_ms).unwrap();
        assert_eq!(warming["healthy"], true);
        assert_eq!(warming["release_gate"], "warming");

        let epoch = warming["continuity"]["epoch_identity_sha256"].clone();
        atomic_json(
            &root.join("ops/recovery-continuity.v1.json"),
            &json!({
                "schema": CONTINUITY_SCHEMA,
                "epoch_identity_sha256": epoch,
                "green_since_ms": at_ms - u128::from(REQUIRED_GREEN_SECONDS) * 1_000,
                "last_observed_at_ms": at_ms - 30_000,
                "last_observation_green": true,
                "last_useful_queue_nonzero_at_ms": at_ms - 30_000,
                "maximum_observation_gap_seconds": MAXIMUM_OBSERVATION_GAP_SECONDS,
                "required_green_seconds": REQUIRED_GREEN_SECONDS,
                "continuously_green_seconds": REQUIRED_GREEN_SECONDS - 30,
                "release_ready": false,
                "reset_reason": Value::Null
            }),
        )
        .unwrap();
        let ready = materialize_recovery(root, at_ms).unwrap();
        assert_eq!(ready["release_gate"], "ready");
        assert_eq!(ready["continuity"]["release_ready"], true);

        let observation_path = root.join("ops/recovery-continuity-observation.v1.json");
        let mut observation = read_json(&observation_path).unwrap();
        let old_model = observation["model_pointer_sha256"]
            .as_str()
            .unwrap()
            .to_owned();
        let new_model = "f".repeat(64);
        observation["model_pointer_sha256"] = json!(new_model);
        observation["generated_at_unix_ms"] = json!(at_ms + 30_000);
        let mut inputs = observation["input_digests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        *inputs
            .iter_mut()
            .find(|digest| **digest == old_model)
            .unwrap() = "f".repeat(64);
        inputs.sort();
        observation["input_digests"] = json!(inputs);
        atomic_json(&observation_path, &observation).unwrap();
        let reset = materialize_recovery(root, at_ms + 30_000).unwrap();
        assert_eq!(reset["release_gate"], "warming");
        assert!(reset["continuity"]["reset_reason"]
            .as_str()
            .unwrap()
            .contains("epoch identity changed"));

        let queue_path = root.join("ops/queue-keeper-v3.json");
        let mut queue = read_json(&queue_path).unwrap();
        queue["generated_at_ms"] = json!(at_ms + 60_000);
        queue["useful_work"] = json!(0);
        queue["queued"] = json!(0);
        atomic_json(&queue_path, &queue).unwrap();
        let blocked = materialize_recovery(root, at_ms + 60_000).unwrap();
        assert_eq!(blocked["release_gate"], "blocked");
        assert_eq!(blocked["continuity"]["green_since_ms"], Value::Null);
        assert_eq!(blocked["continuity"]["last_observation_green"], false);
    }

    #[test]
    fn terminal_candidates_are_detected_without_consuming_or_claiming_them() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let checkpoint = "a".repeat(64);
        atomic_json(
            &root.join("training/task-promotion-evaluations/rejected/evaluation.v3.json"),
            &json!({
                "schema": "TaskPromotionEvaluation.v3",
                "task": "binary",
                "candidate_checkpoint_sha256": checkpoint
            }),
        )
        .unwrap();
        let inventory = terminal_learning_inventory(root);
        assert_eq!(inventory.terminal_evaluations, 1);
        assert_eq!(inventory.unpublished_terminal_candidates.len(), 1);

        atomic_json(
            &root
                .join("ops/learning-reports/binary")
                .join(format!("{checkpoint}.v2.json")),
            &json!({"schema":"JopeLearningReport.v2"}),
        )
        .unwrap();
        let published = terminal_learning_inventory(root);
        assert!(published.unpublished_terminal_candidates.is_empty());
        assert_eq!(published.immutable_learning_reports, 1);
    }

    #[test]
    fn complete_family_disjoint_hard_regression_publishes_create_once() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let checkpoint = "a".repeat(64);
        let panel = "b".repeat(64);
        atomic_json(
            &root.join("training/task-promotion-evaluations/rejected/evaluation.v3.json"),
            &json!({
                "schema": "TaskPromotionEvaluation.v3",
                "task": "binary",
                "accepted_parent_checkpoint_sha256": "c".repeat(64),
                "candidate_checkpoint_sha256": checkpoint,
                "panel_consumption_sha256": panel,
                "run_index": 0,
                "metrics": {
                    "contexts": 128,
                    "confidence": {
                        "resamples": 10000,
                        "seed": 1,
                        "candidate_vs_accepted_lower": -0.1,
                        "candidate_vs_gp_lower": -0.1,
                        "conditional_advantage_lower": 0.01
                    },
                    "maximum_slice_regression": 0.0,
                    "validity": 1.0,
                    "replay_verification": 1.0,
                    "failure_rate_increase": 0.0,
                    "latency_ratio": 1.0,
                    "strict_wins": 1,
                    "equal_budget_wins": 1
                },
                "accepted": false,
                "rejection_reasons": ["candidate_vs_accepted_lower"],
                "evaluated_at_ms": 7
            }),
        )
        .unwrap();
        let observations = (0..128)
            .map(|index| {
                json!({
                    "context_sha256": format!("{index:064x}"),
                    "source_family_sha256": format!("{:064x}", 10_000 + index / 2),
                    "candidate_score": 0.0,
                    "accepted_score": 1.0,
                    "gp_score": 1.0,
                    "candidate_conditional_advantage": 0.01,
                    "accepted_conditional_advantage": 0.0,
                    "valid": true,
                    "replay_verified": true,
                    "candidate_failed": false,
                    "accepted_failed": false,
                    "candidate_latency_ms": 100,
                    "accepted_latency_ms": 100,
                    "strict_win": index == 0,
                    "equal_budget_win": index == 1,
                    "slices": {
                        "origin": "real",
                        "size": "small",
                        "missingness": "none",
                        "imbalance": "balanced",
                        "target_shape": "balanced"
                    }
                })
            })
            .collect::<Vec<_>>();
        atomic_json(
            &root.join("training/task-promotion-evaluations/work/run0.observations.v3.json"),
            &json!({
                "schema": "TaskPromotionObservationSet.v3",
                "panel_consumption_sha256": panel,
                "observations": observations
            }),
        )
        .unwrap();
        assert_eq!(publish_terminal_learning_reports(root).unwrap(), 1);
        assert_eq!(publish_terminal_learning_reports(root).unwrap(), 0);
        let report_path = root
            .join("ops/learning-reports/binary")
            .join(format!("{checkpoint}.v2.json"));
        let report = read_json(&report_path).unwrap();
        assert_eq!(report["disposition"], "regressed");
        assert_eq!(report["metrics"]["family_disjoint"], true);

        let mut tampered = report;
        tampered["disposition"] = json!("improved");
        atomic_json(&report_path, &tampered).unwrap();
        assert!(publish_terminal_learning_reports(root).is_err());
    }

    #[test]
    fn improved_requires_three_complete_disjoint_passes_and_pointer_cas() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let checkpoint = "a".repeat(64);
        for run in 0..3 {
            let panel = format!("{:064x}", 50_000 + run);
            atomic_json(
                &root
                    .join("training/task-promotion-evaluations/accepted")
                    .join(format!("run{run}.v3.json")),
                &json!({
                    "schema": "TaskPromotionEvaluation.v3",
                    "task": "regression",
                    "accepted_parent_checkpoint_sha256": "c".repeat(64),
                    "candidate_checkpoint_sha256": checkpoint,
                    "panel_consumption_sha256": panel,
                    "run_index": run,
                    "metrics": {
                        "contexts": 128,
                        "confidence": {
                            "resamples": 10000,
                            "seed": run,
                            "candidate_vs_accepted_lower": 0.01,
                            "candidate_vs_gp_lower": 0.01,
                            "conditional_advantage_lower": 0.01
                        },
                        "maximum_slice_regression": 0.0,
                        "validity": 1.0,
                        "replay_verification": 1.0,
                        "failure_rate_increase": 0.0,
                        "latency_ratio": 1.0,
                        "strict_wins": 1,
                        "equal_budget_wins": 1
                    },
                    "accepted": true,
                    "rejection_reasons": [],
                    "evaluated_at_ms": 10 + run
                }),
            )
            .unwrap();
            let observations = (0..128)
                .map(|index| {
                    let identity = run * 1_000 + index;
                    json!({
                        "context_sha256": format!("{identity:064x}"),
                        "source_family_sha256": format!("{:064x}", 20_000 + identity / 2),
                        "candidate_score": 0.9,
                        "accepted_score": 0.8,
                        "gp_score": 0.8,
                        "candidate_conditional_advantage": 0.02,
                        "accepted_conditional_advantage": 0.01,
                        "valid": true,
                        "replay_verified": true,
                        "candidate_failed": false,
                        "accepted_failed": false,
                        "candidate_latency_ms": 100,
                        "accepted_latency_ms": 100,
                        "strict_win": index == 0,
                        "equal_budget_win": index == 1,
                        "slices": {
                            "origin": "real",
                            "size": "small",
                            "missingness": "none",
                            "imbalance": "balanced",
                            "target_shape": "balanced"
                        }
                    })
                })
                .collect::<Vec<_>>();
            atomic_json(
                &root
                    .join("training/task-promotion-evaluations/work")
                    .join(format!("run{run}.observations.v3.json")),
                &json!({
                    "schema": "TaskPromotionObservationSet.v3",
                    "panel_consumption_sha256": panel,
                    "observations": observations
                }),
            )
            .unwrap();
        }
        atomic_json(
            &root
                .join("training/task-promotions")
                .join(format!("regression-{checkpoint}.v1.json")),
            &json!({
                "schema": "TaskCheckpointPromotion.v2",
                "task": "regression",
                "parent_checkpoint_sha256": "c".repeat(64),
                "child_checkpoint_sha256": checkpoint,
                "external_pass_receipt_ids": ["run0", "run1", "run2"],
                "independent_single_use_panel_acceptances": 3
            }),
        )
        .unwrap();
        atomic_json(
            &root.join("training/accepted-by-task.json"),
            &json!({
                "tasks": {"regression": {"checkpoint_sha256": checkpoint}}
            }),
        )
        .unwrap();
        assert_eq!(publish_terminal_learning_reports(root).unwrap(), 1);
        let report = read_json(
            &root
                .join("ops/learning-reports/regression")
                .join(format!("{checkpoint}.v2.json")),
        )
        .unwrap();
        assert_eq!(report["disposition"], "improved");
        assert_eq!(
            report["promotion_disposition"]["pointer_compare_and_swap_succeeded"],
            true
        );
        assert_eq!(
            report["metrics"]["observation_run_coverage"],
            json!([0, 1, 2])
        );
    }
}
