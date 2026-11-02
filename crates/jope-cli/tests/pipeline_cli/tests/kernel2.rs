use super::*;

#[test]
fn kernel2_profile_cli_publishes_v4_verifies_evidence_and_defaults_to_fitted_cpp() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("run");
    let mut argv = run_args(&directory, "evaluations=1,seconds=60");
    argv.truncate(argv.len() - 2); // No explicit --emit: exercise the profile's default.
    let result = jope()
        .args(argv)
        .args([
            "--search-profile",
            "kernel2-baseline",
            "--refine-top-k",
            "0",
            "--refine-trials",
            "0",
            "--source-commit",
            &"a".repeat(40),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(receipt["schema"], "jope.pipeline-receipt/v4");
    assert_eq!(receipt["search"]["profile_id"], "kernel2-baseline");
    assert_eq!(receipt["search"]["fault_policy"]["kind"], "reject_hard");
    assert_eq!(receipt["search"]["completed_evaluations"], 1);
    assert!(directory.join("emit/cpp17-fitted/standalone.cpp").is_file());
    let result = jope()
        .args(["pipeline", "verify-receipt", "--receipt"])
        .arg(directory.join("receipt.json"))
        .args(["--evidence", "--json"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap(),
        receipt
    );
    std::fs::write(directory.join("search/cut-0/observations.json"), b"[]").unwrap();
    let result = jope()
        .args(["pipeline", "verify-receipt", "--receipt"])
        .arg(directory.join("receipt.json"))
        .args(["--evidence", "--json"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap()["ok"],
        false
    );
}

#[test]
fn an_objective_ranked_run_records_its_objective_and_bad_objectives_are_refused_first() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("run");
    for bad in [
        vec!["--fitness", "sortino"],
        vec!["--fitness", "no_such_objective"],
        vec!["--fitness", "pinball:quantile=2"],
        vec![
            "--fitness",
            "mae",
            "--search-profile",
            "kernel2-baseline",
            "--refine-top-k",
            "1",
            "--refine-trials",
            "4",
        ],
    ] {
        let result = jope()
            .args(run_args(&directory, "evaluations=4,seconds=60"))
            .args(&bad)
            .output()
            .unwrap();
        assert_eq!(
            result.status.code(),
            Some(2),
            "{bad:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!directory.exists(), "{bad:?}");
    }
    let result = jope()
        .args(run_args(&directory, "evaluations=4,seconds=60"))
        .args(["--fitness", "pinball:quantile=0.8", "--json"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(receipt["search"]["objective"]["name"], "pinball");
    assert_eq!(
        receipt["search"]["objective"]["specification"]["quantile"],
        0.8
    );
    assert!(receipt["search"]["best"]["validation_objective"].is_number());
    let verified = jope()
        .args(["pipeline", "verify-receipt", "--receipt"])
        .arg(directory.join("receipt.json"))
        .args(["--evidence", "--json"])
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
}

#[test]
fn kernel2_cli_refuses_unsupported_bounds_and_the_corrected_command_succeeds() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("run");
    for bad in [
        ["--cuts", "2"],
        ["--threads", "2"],
        ["--islands", "2"],
        ["--incumbent", "missing"],
    ] {
        let result = jope()
            .args(run_args(&directory, "evaluations=4,seconds=60"))
            .args([
                "--search-profile",
                "kernel2-baseline",
                "--refine-top-k",
                "1",
                "--refine-trials",
                "4",
            ])
            .args(bad)
            .output()
            .unwrap();
        assert_eq!(
            result.status.code(),
            Some(2),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!directory.exists());
    }
    let result = jope()
        .args(run_args(&directory, "evaluations=4,seconds=60"))
        .args([
            "--search-profile",
            "kernel2-baseline",
            "--refine-top-k",
            "1",
            "--refine-trials",
            "4",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(directory.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["schema"], "jope.pipeline-receipt/v4");
}

// laneD:B17-lm-refinement begin
/// `--refine-method lm` through the real binary: a kernel2 run whose one-literal seed (the one
/// jope-pipeline's LM run tests use) gives the damped least-squares refinement a `derive` stage to
/// fit; 8 trials hold that stage's surrogate minimum (one site plus five). The receipt records the
/// selection, a published refinement carries a surrogate stage, and `verify-receipt --evidence`,
/// which re-derives every refinement trace, accepts the run.
#[test]
fn kernel2_lm_refinement_records_its_selection_and_verifies_evidence() {
    use sha2::{Digest, Sha256};
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("run");
    let seed = "(pipe X (derive expr=\"1*c0\") (ridge))";
    let seeds = temp.path().join("seeds.json");
    let file = serde_json::json!({
        "schema": "jope.pipeline-seeds/v1",
        "genomes": [{"genome": seed, "program_sha256": format!("{:x}", Sha256::digest(seed))}],
    });
    std::fs::write(&seeds, serde_json::to_vec(&file).unwrap()).unwrap();
    let result = jope()
        .args(run_args(&directory, "evaluations=20,seconds=600"))
        .args([
            "--search-profile",
            "kernel2-baseline",
            "--refine-top-k",
            "1",
            "--refine-trials",
            "8",
            "--refine-method",
            "lm",
            "--rng-seed",
            "7",
            "--source-commit",
            &"a".repeat(40),
            "--json",
        ])
        .arg("--seed-genomes")
        .arg(&seeds)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(receipt["schema"], "jope.pipeline-receipt/v4");
    assert_eq!(
        receipt["search"]["engine_selection"],
        serde_json::json!({"refine": {
            "id": "feat-math.constant-refinement.damped-ls.v1",
            "jacobian": "sites",
            "tau": 1e-3,
        }})
    );
    let refinements: Value = serde_json::from_slice(
        &std::fs::read(directory.join("search/cut-0/refinements.json")).unwrap(),
    )
    .unwrap();
    assert!(
        refinements
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["refinement"]["trace"]["stages"]
                .as_array()
                .is_some_and(|stages| stages.iter().any(|s| s["kind"] == "surrogate"))),
        "{refinements}"
    );
    let verified = jope()
        .args(["pipeline", "verify-receipt", "--receipt"])
        .arg(directory.join("receipt.json"))
        .args(["--evidence", "--json"])
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&verified.stdout).unwrap(),
        receipt
    );
}
