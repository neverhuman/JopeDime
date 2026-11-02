use super::*;

#[test]
fn sigterm_mid_search_stops_between_evaluations_and_writes_no_receipt() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let mut child = jope()
        .args(run_args(&output, "evaluations=400,seconds=600"))
        .spawn()
        .unwrap();
    // The search is running once the cuts are prepared.
    wait_for(
        "the search to start",
        Duration::from_secs(120),
        || output.join("prepared/prepared.json").exists(),
        &mut child,
    );
    terminate(&child);
    let result = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(143), "{stderr}");
    assert!(stderr.contains("termination signal"), "{stderr}");
    assert!(
        !output.join("receipt.json").exists(),
        "an interrupted run wrote a receipt"
    );
    assert!(!output.join("search/cut-0/search.json").exists());
}

#[test]
fn sigterm_during_a_round_keeps_only_completed_rounds_in_the_lineage() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("lineage");
    let mut child = jope()
        .args([
            "pipeline",
            "evolve",
            "--dataset",
            dataset().to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--rounds",
            "3",
            "--control",
            "--budget",
            "evaluations=60,seconds=600",
            "--policy",
            "skip",
            "--emit",
            "rust",
        ])
        .spawn()
        .unwrap();
    let lineage = output.join("lineage.json");
    let rounds = || -> usize {
        std::fs::read(&lineage)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .map_or(0, |v| v["rounds"].as_array().map_or(0, Vec::len))
    };
    // Round 1 completed and is counted in the fsynced lineage; round 2's search has started.
    wait_for(
        "round 2 to start",
        Duration::from_secs(600),
        || rounds() == 1 && output.join("round-002/prepared/prepared.json").exists(),
        &mut child,
    );
    terminate(&child);
    let result = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(143), "{stderr}");
    assert_eq!(rounds(), 1, "only the completed round stays in the lineage");
    assert!(output.join("round-001/receipt.json").exists());
    assert!(!output.join("round-002/receipt.json").exists());
    assert!(!output.join("round-003").exists() && !output.join("control").exists());
    // What remains still verifies.
    let verify = jope()
        .args(["pipeline", "lineage", "verify", "--lineage"])
        .arg(&lineage)
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );
}
