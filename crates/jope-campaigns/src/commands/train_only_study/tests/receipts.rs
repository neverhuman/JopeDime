use super::*;

#[test]
fn sealed_receipts_detect_all_required_tamper_classes() {
    let directory = tempdir().unwrap();
    for artifact in [
        "checkpoint",
        "proposal",
        "transfer-seed",
        "split",
        "observation",
        "generation-checkpoint",
        "baseline-report",
    ] {
        let path = directory.path().join(format!("{artifact}.json"));
        write_receipt(
            &path,
            &json!({"schema": "Fixture.v2", "artifact": artifact, "value": 1}),
        )
        .unwrap();
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["value"] = json!(2);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(
            read_receipt::<Value>(&path).is_err(),
            "{artifact} tampering was not rejected"
        );
    }
}
#[test]
fn checkpoint_overlap_is_disclosed_without_zero_shot_rejection() {
    let directory = tempdir().unwrap();
    let registry = directory.path().join("registry");
    let benchmark_id = GOLDEN42[0];
    let benchmark_root = registry.join(benchmark_id);
    fs::create_dir_all(&benchmark_root).unwrap();
    let meta = json!({
        "source_family": "same-family",
        "source_platform": "fixture",
        "source_dataset_id": "same-source",
        "source_dataset_name": "same-name",
        "source_target_column_original": "y",
        "target_column": "target",
        "target_names": ["target"],
    });
    fs::write(
        benchmark_root.join("meta.json"),
        serde_json::to_vec(&meta).unwrap(),
    )
    .unwrap();
    let train = (0..100)
        .map(|index| format!("{},{}\n", index as f64, index as f64 / 10.0))
        .collect::<String>();
    fs::write(benchmark_root.join("train.csv"), train).unwrap();
    let train_hash = sha256_file(&benchmark_root.join("train.csv")).unwrap();
    let context_hash = "a".repeat(64);
    let catalog = Catalog {
        dataset_dimensions: 1536,
        target_dimensions: 512,
        record_stride_bytes: CONTEXT_BYTES,
        entries: vec![CatalogEntry {
            dataset_hash: benchmark_id.to_string(),
            task: "regression".to_string(),
            train_csv: benchmark_root.join("train.csv"),
            train_sha256: train_hash,
            source_family: "same-family".to_string(),
            context_sha256: context_hash,
            catalog_index: 0,
        }],
    };
    let other_id = "checkpoint-source";
    let other_root = registry.join(other_id);
    fs::create_dir_all(&other_root).unwrap();
    fs::write(
        other_root.join("meta.json"),
        serde_json::to_vec(&meta).unwrap(),
    )
    .unwrap();
    let corpus_train = directory.path().join("train.jsonl");
    let corpus_validation = directory.path().join("validation.jsonl");
    fs::write(
        &corpus_train,
        format!("{{\"dataset_id\":\"{other_id}\"}}\n{{\"dataset_id\":\"{benchmark_id}\"}}\n"),
    )
    .unwrap();
    fs::write(&corpus_validation, "").unwrap();
    let checkpoint = CheckpointProvenance {
        accepted_pointer_path: PathBuf::new(),
        accepted_pointer_sha256: String::new(),
        checkpoint_path: PathBuf::new(),
        checkpoint_sha256: "b".repeat(64),
        model_version: "fixture".to_string(),
        training_corpus_manifest_path: PathBuf::new(),
        training_corpus_manifest_sha256: String::new(),
        training_corpus_train_path: corpus_train,
        training_corpus_train_sha256: "c".repeat(64),
        training_corpus_validation_path: corpus_validation,
        training_corpus_validation_sha256: "d".repeat(64),
    };
    let exclusions = checkpoint_exclusions("fixture", &registry, &checkpoint, &catalog).unwrap();
    assert!(exclusions
        .source_families
        .contains(&"same-family".to_string()));
    assert!(exclusions
        .target_identities
        .contains(&target_identity(&registry, benchmark_id).unwrap()));
    let (records, selected) = build_eligibility(
        "fixture",
        &registry,
        &catalog,
        &exclusions,
        GENERATIONS,
        &[],
        None,
    )
    .unwrap();
    let record = records
        .iter()
        .find(|record| record.dataset_hash == benchmark_id)
        .unwrap();
    assert!(record.checkpoint_source_family_overlap);
    assert!(record.checkpoint_exact_overlap);
    assert!(record.checkpoint_target_identity_overlap);
    assert!(!record
        .reasons
        .iter()
        .any(|reason| reason.starts_with("checkpoint_corpus_")));
    assert!(selected.is_empty());
}
