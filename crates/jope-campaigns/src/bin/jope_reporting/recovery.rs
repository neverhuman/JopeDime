pub(super) const RECOVERY_SCHEMA: &str = "JopeRecoveryReport.v2";
pub(super) const CONTINUITY_SCHEMA: &str = "JopeRecoveryContinuity.v1";
pub(super) const REQUIRED_GREEN_SECONDS: u64 = 86_400;
pub(super) const MAXIMUM_OBSERVATION_GAP_SECONDS: u64 = 120;

use super::*;

#[derive(Clone, Copy)]
pub(super) struct RecoveryComponent {
    pub(super) name: &'static str,
    pub(super) path: &'static str,
    pub(super) schema: &'static str,
    pub(super) freshness_budget_seconds: u64,
}

pub(super) const RECOVERY_COMPONENTS: &[RecoveryComponent] = &[
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

pub(super) fn json_files(directory: &Path) -> Vec<PathBuf> {
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

pub(super) fn collect_named(directory: &Path, suffix: &str, output: &mut Vec<PathBuf>) {
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

pub(super) fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(std::io::BufReader::new(
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    ))?)
}

pub(super) fn read_optional_json(path: &Path) -> Result<Value> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(json!({})),
        Err(error) => return Err(error).with_context(|| format!("open {}", path.display())),
    };
    serde_json::from_reader(std::io::BufReader::new(file))
        .with_context(|| format!("parse {}", path.display()))
}

pub(super) fn native_timestamp(value: &Value) -> Option<u128> {
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

pub(super) fn useful_queue_depth(value: &Value) -> u64 {
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

pub(super) fn recovery_component_check(name: &str, value: &Value) -> (bool, String) {
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

pub(super) fn materialize_recovery(root: &Path, at_ms: u128) -> Result<Value> {
    let mut components = serde_json::Map::new();
    let mut gates = serde_json::Map::new();
    for component in RECOVERY_COMPONENTS {
        let path = root.join(component.path);
        let value = read_optional_json(&path)?;
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
    let queue = read_optional_json(&root.join("ops/queue-keeper-v3.json"))?;
    let queue_depth = useful_queue_depth(&queue);
    let gates_green = gates.values().all(|value| value.as_bool() == Some(true));
    let queue_fresh = components
        .get("queue")
        .and_then(|value| value.get("fresh"))
        .and_then(Value::as_bool)
        == Some(true);
    let observation_path = root.join("ops/recovery-continuity-observation.v1.json");
    let observation_value = read_optional_json(&observation_path)?;
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
    let previous = read_optional_json(&continuity_path)?;
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
    let telemetry = read_optional_json(&root.join("ops/four-task-telemetry.v1.json"))?;
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

pub(super) fn verify_continuity_observation(
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
