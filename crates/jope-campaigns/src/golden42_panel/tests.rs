use super::*;

#[test]
fn cpu_selection_prefers_distinct_physical_cores() {
    let cpus = choose_cpus(80).unwrap();
    assert_eq!(cpus.len(), 80);
    assert_eq!(cpus.iter().copied().collect::<BTreeSet<_>>().len(), 80);
    assert_eq!(&cpus[..64], &(0..64).collect::<Vec<_>>());
}

#[test]
fn strict_rmse_is_sequential_and_rejects_bad_tapes() {
    assert_eq!(raw_rmse(&[0.0, 1.0], &[0.0, 1.0]).unwrap().to_bits(), 0);
    assert!(raw_rmse(&[], &[]).is_err());
    assert!(raw_rmse(&[0.0], &[f64::NAN]).is_err());
}

#[test]
fn worker_group_cancellation_leaves_no_survivor() {
    let mut child = Command::new("setsid")
        .args(["sleep", "60"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id() as i32;
    let stdin = BufWriter::new(child.stdin.take().unwrap());
    let mut workers = vec![WorkerProcess {
        id: 0,
        cpu: 0,
        child,
        stdin,
    }];
    terminate_workers(&mut workers);
    assert!(workers[0].child.try_wait().unwrap().is_some());
    // SAFETY: kill with signal 0 sends nothing and reads no memory; pid came from
    // child.id() (never 0 or 1), so -pid names one process group.
    let status = unsafe { libc::kill(-pid, 0) };
    assert_eq!(status, -1, "cancelled worker process group survived");
}

#[test]
fn checkpoint_resume_is_hash_chained_and_fail_closed() {
    let root = std::env::temp_dir().join(format!(
        "golden42-v3-checkpoint-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("checkpoints")).unwrap();
    let config = V3EvolutionConfig::default();
    let population = initial_population(&config, &[]);
    let mut scored = population
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, candidate)| V3ScoredCandidate {
            candidate,
            rmse: 0.1 + index as f64 * 1e-6,
            novelty: 1.0,
        })
        .collect::<Vec<_>>();
    scored.sort_by(scored_order);
    let next = next_generation(&config, 1, &scored, &[0; 4]).unwrap();
    let checkpoint = Checkpoint {
        schema: "jope-dime.golden42-v3-checkpoint.v1".to_string(),
        run_id: "fixture".to_string(),
        dataset_hash: DATASET.to_string(),
        generation: 0,
        previous_sha256: None,
        ranked_semantic_hashes: scored
            .iter()
            .map(|item| item.candidate.semantic_hash.clone())
            .collect(),
        scored: scored.clone(),
        archive: V3Archive::default(),
        next_population: next.clone(),
        created_at_unix_ms: 1,
    };
    let first = json_bytes(&checkpoint).unwrap();
    atomic_immutable_write(&root.join("checkpoints/generation-000000.v1.json"), &first).unwrap();
    let loaded = validate_checkpoint_chain(&root, "fixture")
        .unwrap()
        .unwrap();
    assert_eq!(loaded.0.next_population, next);
    assert_eq!(loaded.1, sha256_bytes(&first));
    let broken = Checkpoint {
        generation: 1,
        previous_sha256: Some("wrong".to_string()),
        next_population: next_generation(&config, 2, &scored, &[0; 4]).unwrap(),
        ..checkpoint
    };
    atomic_immutable_write(
        &root.join("checkpoints/generation-000001.v1.json"),
        &json_bytes(&broken).unwrap(),
    )
    .unwrap();
    assert!(validate_checkpoint_chain(&root, "fixture").is_err());
    fs::remove_dir_all(root).unwrap();
}

fn assert_target_campaign_v3_fit_preserves_frozen_near_winner_bits() {
    let root = paths::jope_registry().join(DATASET);
    if !root.join("train.csv").is_file() || !root.join("test.csv").is_file() {
        return;
    }
    validate_dataset_hashes(&paths::jope_registry()).unwrap();
    let train = read_regression_csv(&root.join("train.csv")).unwrap();
    let test = read_regression_csv(&root.join("test.csv")).unwrap();
    let genome = GenomeIr::parse(NEAR_WINNER).unwrap();
    let training = campaign_training(&train.x).unwrap();
    let v3 = fit_campaign_candidate(&genome, &training, &train.y).unwrap();
    let legacy = fit_legacy_numeric_bound(&genome, &train.x, &train.y).unwrap();
    let test_table = numeric_raw_table(&test.x).unwrap();
    let expected = v3.predict_table(&test_table).unwrap();
    assert_eq!(
        prediction_bits(&expected),
        prediction_bits(&legacy.predict_table(&test_table).unwrap())
    );
    let codec = encode_fitted_cpp17(&v3).unwrap();
    let runtime_root = workspace_root().join("target/fitted-cpp-target-parity-runtime");
    fs::create_dir_all(&runtime_root).unwrap();
    let runtime = FittedCppRuntime::new(&runtime_root).unwrap();
    let cpp = runtime.load(&codec.bytes).unwrap();
    let encoded = EncodedTable::from_numeric_rows(&test.x).unwrap();
    let mut rust_times = Vec::with_capacity(30);
    let mut cpp_times = Vec::with_capacity(30);
    let mut actual = Vec::new();
    for _ in 0..30 {
        let started = Instant::now();
        assert_eq!(
            prediction_bits(&v3.predict_table(&test_table).unwrap()),
            prediction_bits(&expected)
        );
        rust_times.push(started.elapsed().as_nanos());
        let started = Instant::now();
        actual = cpp.predict(&encoded).unwrap();
        cpp_times.push(started.elapsed().as_nanos());
    }
    assert_eq!(expected.len(), actual.len());
    if let Some((index, rust, cpp)) = first_prediction_mismatch(&expected, &actual) {
        panic!(
            "target C++ parity mismatch at {index}: Rust {rust:?} ({:#018x}), C++ {cpp:?} ({:#018x})",
            rust.to_bits(),
            cpp.to_bits()
        );
    }
    rust_times.sort_unstable();
    cpp_times.sort_unstable();
    assert!(
        cpp_times[15] <= rust_times[15].saturating_mul(4).saturating_add(1_000_000),
        "target C++ median {}ns exceeds Rust {}ns correctness envelope",
        cpp_times[15],
        rust_times[15]
    );
}

#[test]
#[ignore = "unavailable: real-data parity must reuse original registry model artifacts; legacy refits are disallowed"]
fn target_campaign_v3_fit_preserves_frozen_near_winner_bits() {
    std::thread::Builder::new()
        .name("golden42-v3-frozen-near-winner".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(assert_target_campaign_v3_fit_preserves_frozen_near_winner_bits)
        .expect("spawn Golden42 V3 parity test with its qualified stack")
        .join()
        .expect("Golden42 V3 parity test thread panicked");
}
