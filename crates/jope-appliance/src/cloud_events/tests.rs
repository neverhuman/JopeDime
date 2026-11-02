use super::*;

fn snapshot() -> RunSnapshot {
    serde_json::from_value(json!({
        "run_id": "run_01ARZ3NDEKTSV4RRFFQ69G5FAV", "revision": "3", "state": "completed",
        "phase": "finalizing", "pending": null, "effort": "fast",
        "budget_used": {"evaluations": "2", "wall_seconds": "12.5"},
        "budget_total": {"evaluations": "10", "wall_seconds": "60"},
        "best": null, "event_watermark": "9", "warnings": [], "completion_reason": null
    }))
    .expect("snapshot fixture")
}

fn events() -> Vec<jope_api::events::PublicEvent> {
    let run = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";
    let t = "2026-11-02T14:00:00Z";
    serde_json::from_value(json!([
        {"run_id": run, "sequence": "3", "time": t, "kind": "experiment.started",
            "body": {"experiment_id": "exp 1", "hypothesis": "ridge on CANARY_COL_7f3a"}},
        {"run_id": run, "sequence": "4", "time": t, "kind": "experiment.finished",
            "body": {"experiment_id": "exp 1", "outcome": "improved", "metric": "rmse",
                "value": 0.61, "failure": null}},
        {"run_id": run, "sequence": "5", "time": t, "kind": "experiment.finished",
            "body": {"experiment_id": "exp2", "outcome": "failed", "metric": "rmse",
                "value": null, "failure": null}},
        {"run_id": run, "sequence": "9", "time": t, "kind": "run.finished",
            "body": {"completion_reason": "converged", "champion": null,
                "scores": {"cv": 0.61, "holdout": 0.65, "test": null},
                "budget_used": {"evaluations": "2", "wall_seconds": "12.5"},
                "nondeterminism": {"count": "0", "experiment_ids": []}}}
    ]))
    .expect("event fixture")
}

fn profile() -> jope_api::DatasetProfile {
    serde_json::from_value(json!({"dataset_id": "ds_0123456789abcdef01234567",
            "name": "CANARY_NAME", "bytes": "100", "rows": "7342917",
            "columns": [{"name": "CANARY_COL_7f3a", "data_type": "numeric", "missing": "3", "distinct": "7"},
                        {"name": "y", "data_type": "categorical", "missing": "0", "distinct": "2"}],
            "target_suggestions": [], "warnings": []}))
        .expect("profile fixture")
}

#[test]
fn app_events_carry_trials_finish_and_column_summaries_only() {
    let app = project(&snapshot(), &events(), Some(&profile()));
    let kinds: Vec<_> = app.iter().map(|e| e["event"]["kind"].clone()).collect();
    assert_eq!(
        kinds,
        [
            "dataset_profiled",
            "trial_scored",
            "trial_scored",
            "run_finished"
        ]
    );
    assert_eq!(app[1]["event"]["algorithm"], "ridge on CANARY_COL_7f3a");
    assert_eq!(app[1]["event"]["trialId"], "exp_1");
    assert_eq!(app[1]["event"]["cv"], 0.61);
    assert_eq!(app[2]["event"]["outcome"], "failed");
    assert_eq!(app[3]["event"]["holdout"], 0.65);
    assert_eq!(app[3]["event"]["elapsedSeconds"], "12.5");
    // The profile is column summaries: these four keys and nothing else, and not the name.
    for column in app[0]["event"]["columns"].as_array().unwrap() {
        let mut keys: Vec<_> = column.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["dataType", "distinct", "missing", "name"]);
    }
    assert!(!app[0].to_string().contains("CANARY_NAME"));
    // Re-projection is byte-identical, so a re-sync is a duplicate, not a conflict.
    assert_eq!(app, project(&snapshot(), &events(), Some(&profile())));
}

#[test]
fn telemetry_has_no_names_no_exact_counts_and_no_local_ids() {
    let app = project(&snapshot(), &events(), Some(&profile()));
    let ids = Pseudonyms {
        install_id: "0123456789abcdef0123456789abcdef",
        session_id: "1123456789abcdef0123456789abcdef",
        run_id: "2123456789abcdef0123456789abcdef",
        environment: "development",
    };
    let mut n = 0;
    let sent = telemetry(&app, &ids, 1_790_000_000_000, || {
        n += 1;
        format!("{n:032x}")
    });
    assert_eq!(sent.len(), 4);
    let text = serde_json::to_string(&sent.iter().map(|(_, v)| v).collect::<Vec<_>>()).unwrap();
    for canary in [
        "CANARY",
        "7342917",
        "run_01ARZ",
        "exp 1",
        "exp_1",
        "ridge",
        "rmse",
        "0.61",
    ] {
        assert!(!text.contains(canary), "{canary} leaked: {text}");
    }
    assert_eq!(sent[0].1["event"]["dataset"]["rows"], "B1mTo10m");
    assert_eq!(sent[1].1["event"]["score"], "T6");
    assert_eq!(sent[2].1["event"]["outcome"], "failed");
    assert_eq!(sent[3].1["event"]["outcome"], "succeeded");
    assert_eq!(sent[3].1["event"]["algorithm"]["trials"], 2);
    assert_eq!(sent[0].1["source"], "cli");
    assert_eq!(sent[0].0, "profile-train");
}

#[test]
fn buckets_and_tenths_match_veox_telemetry() {
    assert_eq!(bucket(0), "Zero");
    assert_eq!(bucket(999), "B100To1k");
    assert_eq!(bucket(1_000), "B1kTo10k");
    assert_eq!(bucket(u64::MAX), "B1bOrMore");
    assert_eq!(tenth(f64::NAN), "T0");
    assert_eq!(tenth(0.61), "T6");
    assert_eq!(tenth(4.0), "T10");
    assert_eq!(environment("https://app.veox.ai"), "production");
    assert_eq!(environment("http://127.0.0.1:8091"), "development");
}
