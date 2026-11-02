use super::*;

#[test]
fn mirror_conflict_is_fail_closed_and_receipted() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "0014b9dad5725a1e";
    dataset(&paths.source_root, "regression", id, "1");
    dataset(&paths.mirror_root, "regression", id, "2");
    let error = discover(&paths, Some("regression"), None).unwrap_err();
    assert!(error.to_string().contains("mirror content conflict"));
    assert_eq!(fs::read_dir(&paths.receipt_root).unwrap().count(), 1);
}

#[test]
fn exhaustive_audit_receipts_every_conflict_and_writes_stable_summary() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    for (id, source_seed, mirror_seed) in [
        ("0014b9dad5725a1e", "1", "9"),
        ("006bca140f93e411", "2", "8"),
        ("00709d315efacf0c", "3", "3"),
    ] {
        dataset(&paths.source_root, "regression", id, source_seed);
        dataset(&paths.mirror_root, "regression", id, mirror_seed);
    }
    let first = audit_discover(&paths, Some("regression"), None).unwrap();
    assert_eq!(first.inspected, 3);
    assert_eq!(first.entries.len(), 1);
    assert_eq!(first.conflicts.len(), 2);
    assert_eq!(first.conflicts[0].dataset_id, "0014b9dad5725a1e");
    assert_eq!(first.conflicts[1].dataset_id, "006bca140f93e411");
    assert!(first
        .conflicts
        .iter()
        .all(|conflict| conflict.receipt.is_file()));
    let first_summary = write_conflict_summary(&paths, &first).unwrap();
    let first_bytes = fs::read(&first_summary).unwrap();

    let second = audit_discover(&paths, Some("regression"), None).unwrap();
    let second_summary = write_conflict_summary(&paths, &second).unwrap();
    assert_eq!(first_summary, second_summary);
    assert_eq!(first_bytes, fs::read(second_summary).unwrap());
    let pointer = read_json(&paths.receipt_root.join("latest-summary.json")).unwrap();
    assert_eq!(pointer["conflict_count"], 2);
    assert_eq!(pointer["inspected"], 3);
}

#[test]
fn cache_is_atomic_and_validates_all_hashes() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "0014b9dad5725a1e";
    dataset(&paths.source_root, "regression", id, "1");
    let entry = discover(&paths, Some("regression"), None)
        .unwrap()
        .remove(0);
    let manifest = materialize_cache(&paths, &entry, None).unwrap();
    assert!(manifest.train_only_encoding);
    let cache = cache_path(&paths, "regression", id);
    validate_cache(&cache, Some(&entry)).unwrap();
    assert_eq!(
        materialize_cache(&paths, &entry, None).unwrap(),
        manifest,
        "a complete cache must be resumed without rewriting"
    );
    fs::write(cache.join("train.csv"), "drift").unwrap();
    assert!(validate_cache(&cache, Some(&entry)).is_err());
}

#[test]
fn legacy_sidecar_cache_is_upgraded_with_verified_source_files() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "0014b9dad5725a1e";
    dataset(&paths.source_root, "timeseries_reduced_v2", id, "1");
    let entry = discover(&paths, Some("timeseries"), None)
        .unwrap()
        .remove(0);
    materialize_cache(&paths, &entry, None).unwrap();
    let cache = cache_path(&paths, "timeseries", id);
    let current = read_json(&cache.join("cache-manifest.v1.json")).unwrap();
    let old_files = current["files"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| {
            !matches!(
                name.as_str(),
                "train.csv" | "test.csv" | "signature_manifest.json"
            )
        })
        .map(|(name, digest)| (name.clone(), digest.clone()))
        .collect::<serde_json::Map<_, _>>();
    fs::remove_file(cache.join("cache-manifest.v1.json")).unwrap();
    fs::remove_file(cache.join("signature_manifest.json")).unwrap();
    fs::remove_file(cache.join("train.csv")).unwrap();
    fs::remove_file(cache.join("test.csv")).unwrap();
    atomic_json(
        &cache.join("cache-manifest.v1.json"),
        &json!({
            "schema": "JopeExpandedEncodingCache.v1",
            "task": entry.task,
            "source": entry.source,
            "dataset_hash": entry.dataset_hash,
            "encoder_protocol": ENCODER_PROTOCOL,
            "train_only_encoding": true,
            "immutable": true,
            "staged_by": "legacy-fixture",
            "files": old_files,
        }),
    )
    .unwrap();

    let manifest = materialize_cache(&paths, &entry, None).unwrap();
    assert_eq!(manifest.dataset_id, id);
    assert!(cache.join("train.csv").is_file());
    assert!(cache.join("test.csv").is_file());
    assert!(cache.join("signature_manifest.json").is_file());
    validate_cache(&cache, Some(&entry)).unwrap();
}

#[test]
fn missing_encodings_fail_without_leaving_a_partial_cache() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "0014b9dad5725a1e";
    dataset(&paths.source_root, "binary", id, "1");
    fs::remove_file(
        paths
            .source_root
            .join("binary")
            .join(id)
            .join("train_target.vs"),
    )
    .unwrap();
    let entry = discover(&paths, Some("binary"), None).unwrap().remove(0);
    let error = materialize_cache(&paths, &entry, None).unwrap_err();
    assert!(error
        .to_string()
        .contains("no native Rust Hyperion encoder"));
    assert!(!cache_path(&paths, "binary", id).exists());
}

#[test]
fn native_encoder_rejects_scripts() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("encoder.py");
    fs::write(&path, "#!/usr/bin/env python3\n").unwrap();
    assert!(verify_native_encoder(&path).is_err());
}
