use super::*;

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
