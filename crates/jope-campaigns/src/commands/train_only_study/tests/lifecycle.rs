use super::*;

#[test]
fn refill_is_deterministic_preserves_three_elites_and_history_uniqueness() {
    let mut ranked = (0..POPULATION)
        .map(|index| {
            scored(
                member(
                    index,
                    &format!("(pipe X (ridge l2={}))", index as f64 + 0.25),
                    1,
                ),
                1.0 - index as f64 / 100.0,
                0.5 - index as f64 / 1000.0,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(rank_scored);
    let initial = ranked
        .iter()
        .map(|member| member.member.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let mut first_history = initial.clone();
    let mut second_history = initial;
    let first = refill_population("fixture", "dataset", 4, 1, &ranked, &mut first_history).unwrap();
    let second =
        refill_population("fixture", "dataset", 4, 1, &ranked, &mut second_history).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), POPULATION);
    assert_eq!(
        first[..ELITES]
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<Vec<_>>(),
        ranked[..ELITES]
            .iter()
            .map(|member| &member.member.semantic_sha256)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        first
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<HashSet<_>>()
            .len(),
        POPULATION
    );
    assert_eq!(first_history, second_history);
    assert_eq!(first_history.len(), POPULATION + (POPULATION - ELITES));
}
#[test]
#[ignore = "legacy v1 strict-firewall fixture; v2 production registry is covered separately"]
fn preparation_completes_without_test_or_baseline_files() {
    let directory = tempdir().unwrap();
    let registry = directory.path().join("registry");
    fs::create_dir_all(&registry).unwrap();
    let input_root = directory.path().join("inputs");
    fs::create_dir_all(&input_root).unwrap();

    let checkpoint_path = input_root.join("checkpoint.model");
    fs::write(&checkpoint_path, b"fixture checkpoint").unwrap();
    let checkpoint_sha256 = sha256_file(&checkpoint_path).unwrap();
    let accepted_pointer = input_root.join("accepted.json");
    fs::write(
        &accepted_pointer,
        serde_json::to_vec(&json!({
            "schema": "JopeSelfplayAcceptedByTask.v1",
            "tasks": {
                "regression": {
                    "checkpoint": checkpoint_path,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let corpus_train = input_root.join("corpus.train.jsonl");
    let corpus_validation = input_root.join("corpus.val.jsonl");
    fs::write(&corpus_train, "").unwrap();
    fs::write(&corpus_validation, "").unwrap();
    let corpus_manifest = input_root.join("corpus-manifest.json");
    fs::write(
        &corpus_manifest,
        serde_json::to_vec(&json!({
            "outputs": {
                "corpus.train.jsonl": sha256_file(&corpus_train).unwrap(),
                "corpus.val.jsonl": sha256_file(&corpus_validation).unwrap(),
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let provenance = input_root.join("provenance.json");
    fs::write(
        &provenance,
        serde_json::to_vec(&json!({
            "checkpoint_sha": checkpoint_sha256,
            "corpus_manifest": corpus_manifest,
            "checkpoint_config": {"model_version": "fixture-v1"},
        }))
        .unwrap(),
    )
    .unwrap();

    let context_slice = vec![0_u8; CONTEXT_BYTES];
    let context_sha256 = sha256_bytes(&context_slice);
    let mut context_blob = Vec::new();
    let mut catalog_entries = Vec::new();
    for (catalog_index, dataset_hash) in GOLDEN42[..10].iter().enumerate() {
        let dataset_root = registry.join(dataset_hash);
        fs::create_dir_all(&dataset_root).unwrap();
        let train = (0..100)
            .map(|row| {
                format!(
                    "{},{},{}\n",
                    row as f64 / 100.0,
                    (row as f64).sin(),
                    row as f64 / 10.0 + (row as f64).cos()
                )
            })
            .collect::<String>();
        let train_path = dataset_root.join("train.csv");
        fs::write(&train_path, train).unwrap();
        fs::write(
            dataset_root.join("meta.json"),
            serde_json::to_vec(&json!({
                "source_family": format!("fixture-family-{catalog_index}"),
                "source_platform": "fixture",
                "source_dataset_id": format!("fixture-source-{catalog_index}"),
                "source_dataset_name": format!("fixture-name-{catalog_index}"),
                "source_target_column_original": "y",
                "target_column": "target",
                "target_names": ["target"],
            }))
            .unwrap(),
        )
        .unwrap();
        catalog_entries.push(json!({
            "dataset_hash": dataset_hash,
            "task": "regression",
            "train_csv": train_path,
            "train_sha256": sha256_file(&train_path).unwrap(),
            "source_family": format!("fixture-family-{catalog_index}"),
            "context_sha256": context_sha256,
            "catalog_index": catalog_index,
        }));
        context_blob.extend_from_slice(&context_slice);
    }
    let context_data = input_root.join("contexts.bin");
    fs::write(&context_data, &context_blob).unwrap();
    let catalog_path = input_root.join("catalog.json");
    fs::write(
        &catalog_path,
        serde_json::to_vec(&json!({
            "dataset_dimensions": 1536,
            "target_dimensions": 512,
            "record_stride_bytes": CONTEXT_BYTES,
            "entries": catalog_entries,
        }))
        .unwrap(),
    )
    .unwrap();
    let catalog_pointer = input_root.join("catalog-pointer.json");
    fs::write(
        &catalog_pointer,
        serde_json::to_vec(&json!({
            "catalog": catalog_path,
            "catalog_sha256": sha256_file(&catalog_path).unwrap(),
            "data": context_data,
            "data_sha256": sha256_file(&context_data).unwrap(),
        }))
        .unwrap(),
    )
    .unwrap();

    let proposals_path = input_root.join("proposals.jsonl");
    let mut proposal_lines = String::new();
    for dataset_hash in &GOLDEN42[..10] {
        for index in 0..64 {
            let row = proposal_row(dataset_hash, &checkpoint_sha256, &context_sha256, index);
            writeln!(proposal_lines, "{}", serde_json::to_string(&row).unwrap()).unwrap();
        }
    }
    fs::write(&proposals_path, proposal_lines).unwrap();

    let observation_db = input_root.join("observations.sqlite");
    let connection = Connection::open(&observation_db).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE observations(
                    task TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    semantic_sha256 TEXT NOT NULL,
                    canonical_genome TEXT NOT NULL,
                    candidate_source TEXT NOT NULL,
                    score REAL NOT NULL,
                    origin_class TEXT NOT NULL
                );",
        )
        .unwrap();
    for index in 0..35 {
        let genome = canonical(&format!(
            "(pipe X (elasticnet alpha={} l1_ratio=0.5))",
            index as f64 + 0.75
        ));
        connection
                .execute(
                    "INSERT INTO observations VALUES('regression','external',?1,?2,'global',?3,'gp_origin')",
                    rusqlite::params![
                        semantic_identity(&genome),
                        genome,
                        index as f64
                    ],
                )
                .unwrap();
    }
    drop(connection);

    let study_root = directory.path().join("studies");
    let args = PrepareArgs {
        run_id: "fixture-study".to_string(),
        study_root: study_root.clone(),
        checkpoint_pointer: accepted_pointer,
        checkpoint_provenance: provenance,
        catalog_pointer,
        observation_db,
        dl_proposals: Some(proposals_path),
        proposal_generator: PathBuf::new(),
        python: PathBuf::new(),
        proposal_device: "cpu".to_string(),
        sotarena_root: registry.clone(),
        workers: 1,
        smoke: false,
        proposal_manifest_only: false,
    };
    let manifest_path = prepare(&registry, args.clone()).unwrap();
    let first_bytes = fs::read(&manifest_path).unwrap();
    let (manifest, _): (TrainOnlyStudyManifest, _) = read_receipt(&manifest_path).unwrap();
    assert_eq!(manifest.datasets.len(), 10);
    assert!(manifest.firewall.train_only);
    let lower = String::from_utf8_lossy(&first_bytes).to_ascii_lowercase();
    assert!(!lower.contains("/test.csv"));
    assert!(!lower.contains("tabpfn"));
    assert!(!lower.contains("tabfm"));
    assert!(registry.read_dir().unwrap().all(|entry| !entry
        .unwrap()
        .path()
        .join("test.csv")
        .exists()));

    let repeated = prepare(&registry, args).unwrap();
    assert_eq!(manifest_path, repeated);
    assert_eq!(first_bytes, fs::read(repeated).unwrap());
}
#[test]
fn score_refuses_before_validation_receipt_exists() {
    let directory = tempdir().unwrap();
    let error = score(directory.path(), "fixture", directory.path(), 1).unwrap_err();
    assert!(format!("{error:#}").contains("requires a successful independent validation"));
    assert!(!directory.path().join("fixture/test.csv").exists());
}
