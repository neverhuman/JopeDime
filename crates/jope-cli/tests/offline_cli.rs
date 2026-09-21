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
