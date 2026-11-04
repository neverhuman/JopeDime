use super::*;

#[test]
fn cli_qualify_rounds_create_only_and_canonical_digest() {
    let (identity, plan, ratchets, scores, parity, parent_map) = fixture_bundle();
    let direct = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("direct report");
    assert_eq!(direct.verdict, Verdict::Greater);
    assert!(direct.margin_cleared);

    let dir = tempfile::tempdir().expect("tempdir");
    let plan_path = dir.path().join("plan.json");
    let id_path = dir.path().join("identity.json");
    let ratchets_path = dir.path().join("ratchets.json");
    let scores_path = dir.path().join("scores.json");
    let parity_path = dir.path().join("parity.json");
    let parents_path = dir.path().join("parents.json");
    let out_path = dir.path().join("report.json");

    std::fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
    std::fs::write(&id_path, serde_json::to_vec(&identity).unwrap()).unwrap();
    std::fs::write(&ratchets_path, serde_json::to_vec(&ratchets).unwrap()).unwrap();
    std::fs::write(&scores_path, serde_json::to_vec(&scores).unwrap()).unwrap();
    std::fs::write(&parity_path, serde_json::to_vec(&parity).unwrap()).unwrap();
    std::fs::write(&parents_path, serde_json::to_vec(&parent_map).unwrap()).unwrap();

    let bin = env!("CARGO_BIN_EXE_jope_pools");
    let first = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_path.to_str().unwrap(),
            "--identity",
            id_path.to_str().unwrap(),
            "--ratchets",
            ratchets_path.to_str().unwrap(),
            "--test-scores",
            scores_path.to_str().unwrap(),
            "--parity",
            parity_path.to_str().unwrap(),
            "--parent-map",
            parents_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run jope_pools qualify-rounds");
    assert!(
        first.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let stdout = String::from_utf8_lossy(&first.stdout);
    let expected_sha = direct.canonical_sha256().unwrap();
    assert!(stdout.contains(&expected_sha), "stdout={stdout}");

    let written: Value = serde_json::from_slice(&std::fs::read(&out_path).unwrap()).unwrap();
    assert_eq!(written["schema"], ROUND_QUAL_SCHEMA);

    // Second invocation against the same --out path must fail closed (create-only).
    let second = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_path.to_str().unwrap(),
            "--identity",
            id_path.to_str().unwrap(),
            "--ratchets",
            ratchets_path.to_str().unwrap(),
            "--test-scores",
            scores_path.to_str().unwrap(),
            "--parity",
            parity_path.to_str().unwrap(),
            "--parent-map",
            parents_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("re-run jope_pools qualify-rounds");
    assert!(!second.status.success());
}
