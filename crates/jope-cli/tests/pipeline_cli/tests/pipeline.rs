use super::*;

#[test]
fn pipeline_cli_emits_fitted_cpp_and_records_executable_build() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("fitted-run");
    let result = jope()
        .args(["pipeline", "run", "--dataset"])
        .arg(dataset())
        .args([
            "--emit",
            "cpp17-fitted",
            "--policy",
            "skip",
            "--budget",
            "evaluations=1,seconds=60",
            "--source-commit",
            "0123456789abcdef0123456789abcdef01234567",
            "--output",
        ])
        .arg(&out)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["ok"], true);
    assert_eq!(receipt["artifacts"].as_array().unwrap().len(), 3);
    let build = receipt["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["language"] == "cpp17-fitted-standalone")
        .unwrap()["build"]
        .as_str()
        .unwrap();
    let mut words = build.split_whitespace();
    let compiled = Command::new(words.next().unwrap())
        .args(words)
        .current_dir(out.join("emit"))
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert!(out.join("emit/model-cpp17-fitted").is_file());
    let verified = jope()
        .args(["pipeline", "verify-receipt", "--evidence", "--receipt"])
        .arg(out.join("receipt.json"))
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
}
