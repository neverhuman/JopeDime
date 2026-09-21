use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn jope(cwd: &Path, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_jope"))
        .current_dir(cwd)
        .env_remove("JOPE_HOME")
        .env_remove("JOPE_INPUTS_ROOT")
        .args(arguments)
        .output()
        .unwrap()
}

#[test]
fn cli_generates_checks_and_reports_tampering_from_unrelated_directory() {
    let temporary = tempfile::tempdir().unwrap();
    let output = jope(
        temporary.path(),
        &["generate-fixtures", "--output", "portable-inputs", "--json"],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let generated: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(generated["dataset_count"], 10);
    for arguments in [
        vec![
            "verify-inputs",
            "--offline",
            "--json",
            "--root",
            "portable-inputs",
        ],
        vec![
            "generate-fixtures",
            "--check",
            "--output",
            "portable-inputs",
            "--json",
        ],
    ] {
        let output = jope(temporary.path(), &arguments);
        assert!(output.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap(),
            generated
        );
    }
    fs::write(
        temporary.path().join("portable-inputs/binary/linear.json"),
        b"corrupt",
    )
    .unwrap();
    let failed = jope(
        temporary.path(),
        &[
            "verify-inputs",
            "--offline",
            "--json",
            "--root",
            "portable-inputs",
        ],
    );
    assert!(!failed.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&failed.stdout).unwrap()["ok"],
        false
    );
}

#[test]
fn runtime_environment_and_ancestor_resolution_work_without_build_paths() {
    let temporary = tempfile::tempdir().unwrap();
    let checkout = temporary.path().join("independent-checkout");
    fs::create_dir_all(checkout.join("datasets")).unwrap();
    jope_cli::generate_fixtures(&checkout.join("datasets/synthetic")).unwrap();
    fs::create_dir_all(checkout.join("deep/nested")).unwrap();
    let output = jope(
        &checkout.join("deep/nested"),
        &["verify-inputs", "--offline", "--json"],
    );
    assert!(output.status.success());
    for (key, value) in [
        ("JOPE_HOME", checkout.clone()),
        ("JOPE_INPUTS_ROOT", checkout.join("datasets/synthetic")),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_jope"))
            .current_dir(temporary.path())
            .env_remove("JOPE_HOME")
            .env_remove("JOPE_INPUTS_ROOT")
            .env(key, value)
            .args(["verify-inputs", "--offline", "--json"])
            .output()
            .unwrap();
        assert!(output.status.success());
    }
}

#[test]
fn explicit_missing_root_and_missing_offline_flag_fail() {
    let temporary = tempfile::tempdir().unwrap();
    for arguments in [
        vec!["verify-inputs", "--offline", "--json", "--root", "missing"],
        vec!["verify-inputs", "--json"],
    ] {
        let output = jope(temporary.path(), &arguments);
        assert!(!output.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap()["ok"],
            false
        );
    }
}

#[test]
fn committed_corpus_matches_regeneration() {
    // Test-only source location; the installed CLI resolves every path at runtime.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic");
    let report = jope_cli::verify_inputs(&root).unwrap();
    assert_eq!(report.dataset_count, 10);
}

#[test]
fn encode_without_weights_fails_closed() {
    let input =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic/binary/linear.json");
    let output = Command::new(env!("CARGO_BIN_EXE_jope"))
        .env_remove("JOPE_HOME")
        .env_remove("JOPE_INPUTS_ROOT")
        .env_remove("HYPERION_V7_WEIGHTS")
        .env_remove("JAIN_HYPERION_WEIGHTS")
        .env_remove("JAIN_HYPERION_MANIFEST")
        .args([
            "encode",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert_eq!(report["schema"], "jope.encode.v1");
}

fn regression_linear() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic/regression/linear.json")
}

#[test]
fn fit_ridge_on_regression_linear_reports_finite_r2() {
    let input = regression_linear();
    let output = jope(
        Path::new("."),
        &[
            "fit",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["schema"], "jope.fit.v1");
    assert!(report["train_r2"].as_f64().unwrap().is_finite());
    assert!(report["hold_r2"].as_f64().unwrap().is_finite());
}

#[test]
fn emit_all_languages_writes_four_artifacts() {
    let temporary = tempfile::tempdir().unwrap();
    let out = temporary.path().join("emitted");
    let output = jope(
        temporary.path(),
        &[
            "emit",
            "--json",
            "--lang",
            "all",
            "--output",
            out.to_str().expect("utf8 path"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["artifacts"].as_array().unwrap().len(), 4);
    for name in ["model.py", "model.rs", "model.cpp", "model.js"] {
        assert!(out.join(name).is_file(), "missing {name}");
    }
}

#[test]
fn search_picks_a_ridge_family_genome() {
    let input = regression_linear();
    let output = jope(
        Path::new("."),
        &[
            "search",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["schema"], "jope.search.v1");
    assert_eq!(report["considered"], 3);
    assert!(report["genome"].as_str().unwrap().contains("ridge"));
}

fn synthetic_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic")
}

#[test]
fn champions_pull_and_push_fail_closed() {
    for flag in ["--pull", "--push"] {
        let output = jope(Path::new("."), &["champions", "--json", flag]);
        assert!(!output.status.success());
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["ok"], false);
        assert_eq!(report["schema"], "jope.champions.v1");
    }
}

#[test]
fn train_step_without_local_fails_closed() {
    let output = jope(Path::new("."), &["train-step", "--json", "--env", "dev"]);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert_eq!(report["schema"], "jope.train-step.v1");
}

#[test]
fn local_champions_and_train_step_cover_ten_datasets() {
    let root = synthetic_root();
    let root = root.to_str().expect("utf8 path");
    let champions = jope(
        Path::new("."),
        &["champions", "--local", "--json", "--root", root],
    );
    assert!(
        champions.status.success(),
        "{}",
        String::from_utf8_lossy(&champions.stderr)
    );
    let bank: Value = serde_json::from_slice(&champions.stdout).unwrap();
    assert_eq!(bank["ok"], true);
    assert_eq!(bank["dataset_count"], 10);
    assert_eq!(bank["env"], "local");

    let step = jope(
        Path::new("."),
        &["train-step", "--local", "--json", "--root", root],
    );
    assert!(
        step.status.success(),
        "{}",
        String::from_utf8_lossy(&step.stderr)
    );
    let report: Value = serde_json::from_slice(&step.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["labels"].as_array().unwrap().len(), 10);
}
