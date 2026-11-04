use super::*;

#[test]
fn matched_search_seed_is_recorded_and_cannot_change_on_resume() {
    let (temp, prepared) = fixture();
    let output = temp.path().join("seeded");
    let run = |seed: &str| seeded_search(temp.path(), &prepared, &output, seed);
    let initial = run("20260908");
    assert!(
        initial.status.success(),
        "{}",
        String::from_utf8_lossy(&initial.stderr)
    );
    let search: serde_json::Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(search["rng_seed"], 20260908);
    let rows: Vec<serde_json::Value> = read_json(&output.join("observations.json")).unwrap();
    assert!(rows.iter().all(|row| row["rng_seed"] == 20260908));
    let request: serde_json::Value = read_json(&output.join("search-request.json")).unwrap();
    assert_eq!(request["rng_seed"], 20260908);
    fs::remove_file(output.join("search.json")).unwrap();
    let changed = run("20260909");
    assert!(!changed.status.success());
    assert!(
        String::from_utf8_lossy(&changed.stderr).contains("RNG seed mismatch"),
        "{}",
        String::from_utf8_lossy(&changed.stderr)
    );
    assert_eq!(
        read_json::<serde_json::Value>(&output.join("search-request.json")).unwrap(),
        request
    );
    assert_eq!(
        read_json::<Vec<serde_json::Value>>(&output.join("observations.json")).unwrap(),
        rows
    );
}

#[test]
fn legacy_resume_refuses_a_changed_seed_before_recording_its_request() {
    // Outputs written before search-request.json existed resume from progress.json
    // alone. A refused resume must not record its changed request, or the original
    // identity could never resume again.
    let (temp, prepared) = fixture();
    let output = temp.path().join("legacy-seeded");
    let run = |seed: &str| seeded_search(temp.path(), &prepared, &output, seed);
    let initial = run("20260908");
    assert!(
        initial.status.success(),
        "{}",
        String::from_utf8_lossy(&initial.stderr)
    );
    let rows: Vec<serde_json::Value> = read_json(&output.join("observations.json")).unwrap();
    fs::remove_file(output.join("search.json")).unwrap();
    fs::remove_file(output.join("search-request.json")).unwrap();
    let changed = run("20260909");
    assert!(!changed.status.success());
    assert!(
        String::from_utf8_lossy(&changed.stderr).contains("resume RNG seed mismatch"),
        "{}",
        String::from_utf8_lossy(&changed.stderr)
    );
    assert!(
        !output.join("search-request.json").exists(),
        "a refused resume recorded its changed request"
    );
    let resumed = run("20260908");
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    let request: serde_json::Value = read_json(&output.join("search-request.json")).unwrap();
    assert_eq!(request["rng_seed"], 20260908);
    assert_eq!(
        read_json::<Vec<serde_json::Value>>(&output.join("observations.json")).unwrap(),
        rows
    );
}
