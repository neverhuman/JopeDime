use super::*;
use tempfile::TempDir;

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn dataset(index: usize, family: &str) -> Dataset {
    Dataset {
        dataset_hash: format!("d{index:03}"),
        source_family: family.to_owned(),
        train_csv: PathBuf::from(format!("/train/{index}")),
        test_csv: PathBuf::from(format!("/test/{index}")),
        train_sha256: digest((index + 1) as u8),
        test_sha256: digest((index + 2) as u8),
        protocol_sha256: digest((index + 3) as u8),
        incumbent_score: Some(0.5),
    }
}

fn request() -> Request {
    let datasets = (0..40)
        .map(|index| dataset(index, &format!("f{index:03}")))
        .collect::<Vec<_>>();
    let genomes = [
        "(pipe X (ridge))",
        "(pipe X (ridge l2=0.1))",
        "(pipe X (mean))",
    ];
    let candidates = genomes
        .iter()
        .enumerate()
        .map(|(index, genome)| Candidate {
            dataset_hash: "d000".to_owned(),
            semantic_sha256: sha256_bytes(genome.as_bytes()),
            canonical_genome: (*genome).to_owned(),
            score: 0.7 - index as f64 / 10.0,
            valid: true,
        })
        .collect();
    Request {
        schema: REQUEST_SCHEMA.to_owned(),
        task: "regression".to_owned(),
        checkpoint_sha256: digest(240),
        catalog_sha256: digest(241),
        compiler_bundle_sha256: digest(242),
        target_triple: "x86_64-unknown-linux-gnu".to_owned(),
        frozen_seed_sha256: digest(243),
        hall_of_fame_sha256: digest(244),
        generation: "strict-wave-000001".to_owned(),
        panel_seed: default_panel_seed(),
        datasets,
        candidates,
    }
}

#[test]
fn panel_is_exact_family_diverse_and_deterministic() {
    let request = request();
    let first = select_panel(&request).unwrap();
    let second = select_panel(&request).unwrap();
    assert_eq!(first.len(), PANEL_SIZE);
    assert_eq!(
        first
            .iter()
            .map(|row| &row.dataset_hash)
            .collect::<Vec<_>>(),
        second
            .iter()
            .map(|row| &row.dataset_hash)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        first
            .iter()
            .map(|row| &row.source_family)
            .collect::<BTreeSet<_>>()
            .len(),
        PANEL_SIZE
    );
}

#[test]
fn finalists_are_top_two_or_strict_and_bounded() {
    let request = request();
    let finalists = qualified_candidates(&request).unwrap();
    assert_eq!(finalists.len(), 2);
    assert!(finalists.iter().all(|row| row.top_two));
    assert!(finalists.iter().all(|row| row.strict_incumbent_beat));
}

#[test]
fn native_executor_rejects_scripts() {
    let temporary = TempDir::new().unwrap();
    let path = temporary.path().join("fake");
    fs::write(&path, b"#!/usr/bin/python3\n").unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&path, permissions).unwrap();
    assert!(validate_native_elf(&path, "evaluator").is_err());
}

#[test]
fn parity_is_bit_exact() {
    assert!(exact_parity(&[0.0, 1.0], &[0.0, 1.0], 1.0, 1.0).qualified);
    assert!(!exact_parity(&[-0.0], &[0.0], 1.0, 1.0).qualified);
}

#[test]
fn cache_status_counts_append_only_operations() {
    let temporary = TempDir::new().unwrap();
    let operations = temporary.path().join("operations");
    fs::create_dir_all(&operations).unwrap();
    atomic_json(
        &operations.join("a.json"),
        &json!({"schema": OPERATION_SCHEMA, "outcome": "hit"}),
    )
    .unwrap();
    atomic_json(
        &operations.join("b.json"),
        &json!({"schema": OPERATION_SCHEMA, "outcome": "miss"}),
    )
    .unwrap();
    let status = cache_status(temporary.path()).unwrap();
    assert_eq!(status["hits"], 1);
    assert_eq!(status["misses"], 1);
    assert_eq!(status["hit_rate"], 0.5);
}

#[test]
fn native_build_emits_dataset_bound_bit_parity_artifact() {
    let temporary = TempDir::new().unwrap();
    let train = temporary.path().join("train.csv");
    let test = temporary.path().join("test.csv");
    fs::write(&train, b"0,0\n1,1\n2,2\n3,3\n").unwrap();
    fs::write(&test, b"4,4\n5,5\n6,6\n7,7\n").unwrap();
    let genome = "(pipe X (ridge))";
    let candidate = Candidate {
        dataset_hash: "fixture".to_owned(),
        semantic_sha256: sha256_bytes(genome.as_bytes()),
        canonical_genome: genome.to_owned(),
        score: 0.9,
        valid: true,
    };
    let dataset = Dataset {
        dataset_hash: candidate.dataset_hash.clone(),
        source_family: "fixture-family".to_owned(),
        train_csv: train.clone(),
        test_csv: test.clone(),
        train_sha256: sha256_file(&train).unwrap(),
        test_sha256: sha256_file(&test).unwrap(),
        protocol_sha256: digest(230),
        incumbent_score: Some(0.8),
    };
    let mut request = request();
    request.datasets = vec![dataset.clone()];
    request.candidates = vec![candidate.clone()];
    let key = compiled_cache_key(
        &request.task,
        &dataset.train_sha256,
        &candidate.semantic_sha256,
        &dataset.protocol_sha256,
        &request.compiler_bundle_sha256,
        &request.target_triple,
    )
    .unwrap();
    let entry = temporary.path().join("entry");
    let runtime = FittedCppRuntime::new(temporary.path().join("runtime")).unwrap();
    build_entry(
        &entry,
        &request,
        &dataset,
        &candidate,
        &Qualified {
            dataset_hash: dataset.dataset_hash.clone(),
            semantic_sha256: candidate.semantic_sha256.clone(),
            score: candidate.score,
            rank: 1,
            top_two: true,
            strict_incumbent_beat: true,
            incumbent_score: dataset.incumbent_score,
        },
        &key,
        &digest(231),
        &runtime,
    )
    .unwrap();
    validate_entry(&entry, &request, &dataset, &candidate, &key).unwrap();
    let manifest = read_json(&entry.join("artifact-manifest.json")).unwrap();
    assert_eq!(manifest["prediction_bit_parity"], true);
    assert_eq!(manifest["score_bit_parity"], true);
    assert_eq!(
        manifest["fitted_state_scope"]["dataset_train_sha256"],
        dataset.train_sha256
    );
}
