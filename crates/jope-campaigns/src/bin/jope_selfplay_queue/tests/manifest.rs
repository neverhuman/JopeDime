use super::*;
use tempfile::tempdir;

#[test]
fn keeper_workers_follow_usable_cpus_and_stay_bounded() {
    let usable = thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    assert_eq!(keeper_workers(), usable.min(MAX_KEEPER_WORKERS));
    assert!((1..=MAX_KEEPER_WORKERS).contains(&keeper_workers()));
}

#[test]
fn rendered_spec_pins_every_input_and_tool_drift_fails_the_cycle() {
    let root = tempdir().unwrap();
    let write = |name: &str, bytes: &[u8]| {
        let path = root.path().join(name);
        fs::write(&path, bytes).unwrap();
        path
    };
    let runner = write("runner.sh", b"#!/bin/sh\n");
    let evaluator = write("corpus_eval", b"evaluator-v1");
    let train = write("train.csv", b"x,y\n1,2\n");
    let test = write("test.csv", b"x,y\n3,4\n");
    let manifest_path = write("manifest.json", b"{}");
    let manifest = ValidManifest {
        path: manifest_path.clone(),
        sha256: sha256_file(&manifest_path).unwrap(),
        generated_at_ms: 0,
        pairs: Vec::new(),
        value: json!({
            "train_csv": train.to_string_lossy(),
            "test_csv": test.to_string_lossy(),
            "train_sha256": sha256_file(&train).unwrap(),
            "test_sha256": sha256_file(&test).unwrap(),
        }),
    };
    let runner_sha256 = sha256_file(&runner).unwrap();
    let evaluator_sha256 = sha256_file(&evaluator).unwrap();
    let tools = PinnedTools {
        runner: (runner.as_path(), &runner_sha256),
        evaluator: (evaluator.as_path(), &evaluator_sha256),
    };
    let spec = root.path().join("job.toml");
    render_spec(&spec, "jte-test", &manifest, &tools).unwrap();
    let text = fs::read_to_string(&spec).unwrap();
    for digest in [
        &runner_sha256,
        &evaluator_sha256,
        &manifest.sha256,
        &sha256_file(&train).unwrap(),
        &sha256_file(&test).unwrap(),
    ] {
        assert!(text.contains(&format!("sha256 = \"{digest}\"")), "{text}");
    }
    tools.verify_unchanged().unwrap();

    // A manifest-declared input that drifted is still refused per manifest.
    fs::write(&train, b"x,y\n9,9\n").unwrap();
    let error = render_spec(&spec, "jte-test", &manifest, &tools).unwrap_err();
    assert!(
        format!("{error:#}").contains("content pin drift"),
        "{error:#}"
    );

    // A tool replaced after the cycle hashed it fails the cycle.
    fs::write(&evaluator, b"evaluator-v2").unwrap();
    let error = tools.verify_unchanged().unwrap_err();
    assert!(format!("{error:#}").contains("corpus_eval"), "{error:#}");
}

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

#[test]
fn generated_registry_preflight_runs_before_reservation() {
    let root = tempdir().unwrap();
    let path = root.path().join("manifest.json");
    fs::write(&path, b"manifest").unwrap();
    let mut manifest = ValidManifest {
        path,
        sha256: "b".repeat(64),
        generated_at_ms: 1,
        pairs: Vec::new(),
        value: manifest_value("strict-wave-000001", "epoch"),
    };
    let accepted = preflight_manifest(&generated_registry(), &manifest, None).unwrap();
    assert_eq!(accepted.disposition, TaskPreflightDisposition::Accepted);

    manifest.value["task"] = json!("binary");
    let rejected = preflight_manifest(&generated_registry(), &manifest, None).unwrap();
    assert_eq!(
        rejected.disposition,
        TaskPreflightDisposition::PermanentRejection
    );
    assert_eq!(rejected.estimated_compute_avoided, rejected.requested_work);
}

#[test]
fn typed_preflight_receipts_are_append_only() {
    let root = tempdir().unwrap();
    let connection = Connection::open_in_memory().unwrap();
    initialize_backlog(&connection).unwrap();
    let path = root.path().join("manifest.json");
    fs::write(&path, b"manifest").unwrap();
    let manifest = ValidManifest {
        path: path.clone(),
        sha256: sha256_file(&path).unwrap(),
        generated_at_ms: 1,
        pairs: Vec::new(),
        value: manifest_value("strict-wave-000001", "epoch"),
    };
    let accepted = preflight_manifest(&generated_registry(), &manifest, None).unwrap();
    record_preflight_receipt(&connection, &path, &accepted).unwrap();
    record_preflight_receipt(&connection, &path, &accepted).unwrap();
    let count: i64 = connection
        .query_row("SELECT COUNT(*) FROM task_preflight_receipts", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1);

    let rejected = TaskPreflightOutcomeV1::failed_regression_screening(
        &accepted,
        REGRESSION_SCREENING_BUDGET_SECONDS,
    )
    .unwrap();
    assert!(record_preflight_receipt(&connection, &path, &rejected).is_err());
}

#[test]
fn bounded_screening_requires_valid_candidate_receipts() {
    let root = tempdir().unwrap();
    let receipts = root.path().join("dataset.receipts.jsonl");
    fs::write(
        &receipts,
        b"{\"schema\":\"receipt\",\"valid\":true,\"failure\":null}\n",
    )
    .unwrap();
    validate_bounded_screening_receipts(root.path()).unwrap();

    fs::write(&receipts, b"{\"valid\":false,\"failure\":\"timeout\"}\n").unwrap();
    assert!(validate_bounded_screening_receipts(root.path())
        .unwrap_err()
        .to_string()
        .contains("timeout"));
}
