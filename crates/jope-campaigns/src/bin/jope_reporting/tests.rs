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
