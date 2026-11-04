use super::*;

#[test]
fn tranche_activation_is_hashed_validated_and_idempotent() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let id = "feedb9dad5725a1e";
    dataset(&paths.source_root, "regression", id, "1");
    let entry = discover(&paths, Some("regression"), None)
        .unwrap()
        .remove(0);
    materialize_cache(&paths, &entry, None).unwrap();

    let foundation = temp.path().join("foundation");
    fs::create_dir_all(&foundation).unwrap();
    fs::write(
        foundation.join(format!("{id}.npy")),
        npy_bytes(&[2048], &vec![0.0; 2048]),
    )
    .unwrap();
    let projection = temp.path().join("projection.npz");
    write_projection(&projection);
    atomic_json(
        &paths.standardization,
        &json!({
            "dim": 512,
            "mean": vec![0.0_f32; 512],
            "scale": vec![1.0_f32; 512],
            "foundation": {
                "cache_dir": foundation,
                "projection_file": projection,
                "raw_dim": 2048,
                "proj_dim": 1024,
            }
        }),
    )
    .unwrap();
    let mut sealed = Vec::new();
    for task in TASKS {
        for index in 0..sealed_target(task) {
            sealed.push(InventoryEntry {
                task: task.to_owned(),
                dataset_id: format!("{index:016x}"),
                dataset_hash: format!("{index:016x}"),
                source: PathBuf::from("private"),
                source_family: format!("{task}-sealed-{index}"),
                train_sha256: "a".repeat(64),
                test_sha256: "b".repeat(64),
                meta_sha256: "c".repeat(64),
                mirror: None,
            });
        }
    }
    freeze_sealed(&paths, &sealed, 7).unwrap();
    let first = activate_tranche(&paths, "tranche-test", std::slice::from_ref(&entry)).unwrap();
    assert_eq!(
        first,
        activate_tranche(&paths, "tranche-test", std::slice::from_ref(&entry)).unwrap()
    );
    let status = validate_catalog_pointer(&paths).unwrap();
    assert_eq!(status["healthy"], true);
    assert_eq!(status["row_count"], 1);
    assert_eq!(
        fs::read_to_string(paths.catalog_root.join("activations.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}
