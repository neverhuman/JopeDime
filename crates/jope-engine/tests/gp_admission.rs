use jope_parallel::{corpus::ContextRow, label_admission, manifest::*, prepare::*};
use serde_json::{json, Value};
use std::{fs, path::Path, process::Command};

fn make_prepared(root: &Path) -> std::path::PathBuf {
    let prepared = root.join("prepared");
    let cut = prepared.join("cut-0");
    fs::create_dir_all(&cut).unwrap();
    let raw = (0..32)
        .map(|i| vec![i.to_string(), (i * i).to_string(), (7 + i * 3).to_string()])
        .collect::<Vec<_>>();
    let preprocessing = Preprocessing::fit(&raw[..24], "regression").unwrap();
    write_json(&cut.join("preprocessing.json"), &preprocessing).unwrap();
    fs::write(
        cut.join("train.csv"),
        csv_bytes(&preprocessing.transform(&raw[..24]).unwrap()).unwrap(),
    )
    .unwrap();
    fs::write(
        cut.join("validation.csv"),
        csv_bytes(&preprocessing.transform(&raw[24..]).unwrap()).unwrap(),
    )
    .unwrap();
    let p:Prepared=serde_json::from_value(json!({"schema":"parallel.prepared.v1","dataset":"fixture",
        "family":"fixture","task":"regression","meta_partition":"meta_train","seed":17,
        "family_partition_frozen":false,"clean_evaluation":true,"clean_status":"synthetic_unit_fixture",
        "original_train_sha256":"synthetic","original_test_sha256":"not_opened","selected_train_indices":[],
        "selected_test_indices":[],"target_last":true,"rows_total":32,"cuts":[{"index":0,
        "train_indices":(0..24).collect::<Vec<_>>(),"validation_indices":(24..32).collect::<Vec<_>>(),
        "training_sha256":file_hash(&cut.join("train.csv")).unwrap(),
        "validation_sha256":file_hash(&cut.join("validation.csv")).unwrap(),
        "preprocessing_sha256":file_hash(&cut.join("preprocessing.json")).unwrap()}]})).unwrap();
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    prepared
}

fn search_command(root: &Path, prepared: &Path, output: &Path) -> Command {
    let vocabulary = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
    let binary = Path::new(env!("CARGO_BIN_EXE_jope-parallel"));
    let mut command = Command::new(binary);
    command
        .env("JOPE_CPU_EVALUATOR", binary)
        .arg("--root")
        .arg(root)
        .arg("search")
        .arg("--prepared")
        .arg(prepared)
        .args([
            "--cut",
            "0",
            "--evaluations",
            "1",
            "--seconds",
            "60",
            "--qualification",
        ])
        .arg("--vocabulary")
        .arg(&vocabulary)
        .arg("--output")
        .arg(output)
        .env_remove("JOPE_EXPERIMENT_DIR")
        .env_remove("JOPE_EXPERIMENT_CRASH_AFTER")
        .env_remove("JOPE_FIT_AUTHORITY")
        .env_remove("JOPE_FIT_WITNESS")
        .env_remove("JOPE_FIT_SCOPE")
        .env_remove("JOPE_FIT_SERVICE_SOCKET")
        .env_remove("JOPE_FIT_DEADLINE_MONOTONIC_NS")
        .env("OMP_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1");
    command
}

#[test]
fn search_without_authority_cannot_publish_an_admissible_quality_label() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    let prepared = make_prepared(root);
    let output = root.join("search");
    let result = search_command(root, &prepared, &output).output().unwrap();
    assert!(
        result.status.success(),
        "accepted search failed to close: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    let binary = Path::new(env!("CARGO_BIN_EXE_jope-parallel"));
    let vocabulary = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), 1);
    for row in &rows {
        assert!(
            !row["failure"].is_null(),
            "missing authority produced a successful fit: {row}"
        );
        assert!(
            row["metrics"].is_null(),
            "failed dispatch fabricated quality"
        );
    }
    let receipt: Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(receipt["terminal"], true);
    assert_eq!(receipt["successful_measurements"], 0);
    assert_eq!(receipt["failed_or_unavailable_measurements"], 1);
    assert_eq!(receipt["best_program"], Value::Null);
    assert_eq!(
        receipt["observations_sha256"],
        file_hash(&output.join("observations.json")).unwrap()
    );
    let context = ContextRow {
        dataset: "fixture".into(),
        family: "fixture".into(),
        partition: "meta_train".into(),
        cut: 0,
        path: "unused-context".into(),
        sha256: "unused".into(),
        prepared: "prepared".into(),
        prepared_sha256: file_hash(&prepared.join("prepared.json")).unwrap(),
    };
    let policy = json!({"schema":"parallel.label_admission.v2","evaluator_protocol":label_admission::PROTOCOL,
        "evaluator_sha256":file_hash(binary).unwrap()});
    for row in &rows {
        let completion = label_admission::artifact(root, &row["clean_completion"]).unwrap();
        let log = label_admission::artifact_bytes(root, &completion["worker_log"]).unwrap();
        assert!(
            String::from_utf8_lossy(&log).contains("mandatory fit authority service unavailable")
        );
        assert!(completion["fit_completion"].is_null());
        assert_eq!(completion["exit_observed"], true);
        assert!(
            label_admission::validate(
                root,
                row,
                &context,
                &p,
                &file_hash(&vocabulary).unwrap(),
                &policy
            )
            .is_err(),
            "unregistered execution became an admissible label"
        );
        label_admission::validate_receipt(&receipt, row, rows.len(), &policy).unwrap();
    }
}

#[test]
fn invalid_artifact_roots_are_rejected_before_search_acceptance() {
    use std::os::unix::fs::symlink;
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    let prepared = make_prepared(root);
    let outside = tempfile::tempdir().unwrap();
    let escaped = outside.path().join("escaped-search");
    let linked = root.join("linked");
    symlink(outside.path(), &linked).unwrap();
    for output in [
        escaped.clone(),
        linked.join("escaped-search"),
        root.join("../escaped-search"),
    ] {
        let result = search_command(root, &prepared, &output).output().unwrap();
        assert!(!result.status.success(), "invalid output root accepted");
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            error.contains("outside declared root")
                || error.contains("symlink")
                || error.contains("unsafe relative path"),
            "{error}"
        );
        assert!(!output.join("search-request.json").exists());
        assert!(!output.join("candidate-0.attempt.json").exists());
    }
    let prep_alias = root.join("prepared-alias");
    symlink(&prepared, &prep_alias).unwrap();
    let output = root.join("search");
    let result = search_command(root, &prep_alias, &output).output().unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("symlink"));
    assert!(!output.exists());
    let result = search_command(&root.join("missing-root"), &prepared, &output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("search artifact root unavailable"));
    assert!(!output.exists());
}

#[test]
fn unreachable_authority_closes_the_search_with_original_failed_exit() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    let prepared = make_prepared(root);
    let output = root.join("search");
    let result = search_command(root, &prepared, &output)
        .env(
            "JOPE_FIT_SERVICE_SOCKET",
            root.join("missing-authority.sock"),
        )
        .env("JOPE_FIT_SCOPE", "private-synthetic-denial")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value = read_json(&output.join("search.json")).unwrap();
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(receipt["successful_measurements"], 0);
    assert_eq!(rows.len(), 1);
    let completion = label_admission::artifact(root, &rows[0]["clean_completion"]).unwrap();
    assert_eq!(completion["outcome"], "infrastructure_failure");
    assert_eq!(completion["exit_observed"], true);
    assert_ne!(completion["exit_code"], 0);
    assert!(completion["fit_completion"].is_null());
    assert!(rows[0]["metrics"].is_null());
}

#[test]
fn interrupted_denial_recovers_original_completion_without_redelivery() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    let prepared = make_prepared(root);
    let output = root.join("search");
    write_json(
        &root.join("manifest.json"),
        &json!({"schema":"parallel.seed_experiment.v2"}),
    )
    .unwrap();
    let result = search_command(root, &prepared, &output)
        .env("JOPE_EXPERIMENT_DIR", root)
        .env("JOPE_EXPERIMENT_CRASH_AFTER", "candidate_completion")
        .output()
        .unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(result.status.signal(), Some(libc::SIGKILL));
    assert!(!output.join("search.json").exists());
    let original_completion = file_hash(&output.join("candidate-0.completion.json")).unwrap();
    let original_log = file_hash(&output.join("candidate-0.log")).unwrap();
    let result = search_command(root, &prepared, &output)
        .env("JOPE_EXPERIMENT_DIR", root)
        .env_remove("JOPE_EXPERIMENT_CRASH_AFTER")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        original_completion,
        file_hash(&output.join("candidate-0.completion.json")).unwrap()
    );
    assert_eq!(
        original_log,
        file_hash(&output.join("candidate-0.log")).unwrap()
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["recovered_completed_artifact"], true);
    assert!(rows[0]["metrics"].is_null());
    let receipt: Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(receipt["successful_measurements"], 0);
    assert_eq!(receipt["terminal"], true);
}
