use super::*;

#[test]
fn cas_retries_are_idempotent_and_collisions_or_corruption_fail() {
    let temp = TempDir::new().unwrap();
    let store = ArtifactStoreV1::new(temp.path());
    let identity = digest("logical");
    let first = store.publish(&identity, &vec![1u32, 2, 3]).unwrap();
    assert_eq!(store.publish(&identity, &vec![1u32, 2, 3]).unwrap(), first);
    assert!(store
        .publish(&identity, &vec![4u32])
        .unwrap_err()
        .contains("collision"));
    let loaded: Vec<u32> = store.load(&identity).unwrap();
    assert_eq!(loaded, vec![1, 2, 3]);
    let object = temp
        .path()
        .join("objects/sha256")
        .join(&first[..2])
        .join(format!("{first}.json"));
    fs::write(object, b"corrupt").unwrap();
    assert!(store
        .load::<Vec<u32>>(&identity)
        .unwrap_err()
        .contains("corrupt"));
}

#[test]
fn prediction_cache_is_arm_free_and_rejects_mismatched_evidence() {
    let temp = TempDir::new().unwrap();
    let candidate = genome(GpTaskV1::Regression, 0);
    let key = PredictionCacheKeyV1 {
        schema: PredictionCacheKeyV1::SCHEMA.into(),
        dataset_sha256: digest("cache dataset"),
        training_sha256: digest("cache training"),
        split_sha256: digest("cache split"),
        candidate_semantic_sha256: candidate.semantic_sha256.clone(),
        evaluator_sha256: digest("cache evaluator"),
        math_policy_sha256: digest("cache math"),
        backend_policy_sha256: digest("cache backend"),
    };
    let prediction = PredictionObjectV1 {
        schema: PredictionObjectV1::SCHEMA.into(),
        training_view_sha256: key.training_sha256.clone(),
        fold_plan_sha256: key.split_sha256.clone(),
        fold_index: 0,
        candidate_semantic_sha256: key.candidate_semantic_sha256.clone(),
        evaluator_sha256: key.evaluator_sha256.clone(),
        fit_sha256: digest("cache fit"),
        math_policy_sha256: key.math_policy_sha256.clone(),
        backend_policy_sha256: key.backend_policy_sha256.clone(),
        validation_row_ids: vec!["r0".into(), "r1".into()],
        prediction_shape: vec![2],
        prediction_f64_bits: vec![0.1f64.to_bits(), 0.2f64.to_bits()],
        valid: true,
        invalid_reason: None,
    };
    let cache = PredictionCacheV1::new(temp.path());
    cache.publish(&key, &prediction).unwrap();
    assert_eq!(cache.load(&key).unwrap(), Some(prediction.clone()));
    let mut stale = prediction;
    stale.fold_plan_sha256 = digest("another split");
    assert!(cache
        .publish(&key, &stale)
        .unwrap_err()
        .contains("does not match"));
}

#[test]
fn operational_adapter_preserves_legacy_shards_and_seals_test_access() {
    let evaluator_path = env!("CARGO_BIN_EXE_corpus_eval");
    let evaluator_bytes = fs::read(evaluator_path).unwrap();
    let evaluator = EvaluatorFreezeV1 {
        schema: EvaluatorFreezeV1::SCHEMA.into(),
        artifact: FrozenArtifactV1 {
            path: evaluator_path.into(),
            sha256: raw_sha256(&evaluator_bytes),
            byte_length: evaluator_bytes.len() as u64,
        },
        evaluator_protocol_sha256: digest("evaluator protocol"),
        math_policy_sha256: digest("math policy"),
        backend_policy_sha256: digest("backend policy"),
    };
    let lines: Vec<_> = (0..20)
        .map(|index| format!("{},{}", index as f64 / 10.0, index as f64 / 5.0))
        .collect();
    let table_rows: Vec<_> = lines
        .iter()
        .enumerate()
        .map(|(index, line)| DatasetRowV1 {
            split: SplitRowV1 {
                stable_row_id: format!("r{index}"),
                atomic_group: format!("g{index}"),
                regression_target: Some(index as f64 / 5.0),
                binary_target: None,
            },
            csv_line: line.clone(),
        })
        .collect();
    let mut bytes = lines.join("\n").into_bytes();
    bytes.push(b'\n');
    let dataset = DatasetTableV1 {
        task: GpTaskV1::Regression,
        dataset_sha256: digest("adapter dataset"),
        training_sha256: raw_sha256(&bytes),
        rows: table_rows,
    };
    let split = RowSplitV1 {
        split_index: 0,
        seed: 1,
        train_row_ids: (0..16).map(|index| format!("r{index}")).collect(),
        validation_row_ids: (16..20).map(|index| format!("r{index}")).collect(),
        validation_groups: (16..20).map(|index| format!("g{index}")).collect(),
        balance_counts: BTreeMap::new(),
    };
    let temp = TempDir::new().unwrap();
    let adapter = CorpusEvalAdapter::new(evaluator, temp.path(), 30).unwrap();
    let outcome = adapter
        .evaluate_search_split(&dataset, &split, &[genome(GpTaskV1::Regression, 0)])
        .unwrap();
    assert_eq!(outcome.len(), 1);
    assert!(outcome[0].prediction.valid);
    assert_eq!(
        outcome[0].prediction.validation_row_ids,
        split.validation_row_ids
    );
    assert_eq!(
        adapter
            .evaluate_search_split(&dataset, &split, &[genome(GpTaskV1::Regression, 0)])
            .unwrap(),
        outcome
    );

    let prefix = fs::read_dir(temp.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let invocation = fs::read_dir(prefix)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let done = fs::read_dir(invocation.join("output"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension == "done")
        })
        .unwrap();
    fs::write(
        done,
        format!("{{\"manifest_sha256\":\"{}\"}}", "0".repeat(64)),
    )
    .unwrap();
    assert!(adapter
        .evaluate_search_split(&dataset, &split, &[genome(GpTaskV1::Regression, 0)])
        .unwrap_err()
        .contains("stale"));

    let finalists = (0..10)
        .map(|index| Cv7CandidateEvidenceV1 {
            candidate: genome(GpTaskV1::Regression, index),
            fold_primary_scores: vec![0.5; 7],
            pooled_oof_primary: 0.5,
            secondary_metric: 1.0,
            runtime_seconds: 1.0,
            valid: true,
        })
        .collect();
    let wrong_selection = select_cv7_winner(
        CampaignArmV1::RotatingMean,
        &digest("wrong dataset"),
        &digest("archive"),
        &digest("cv"),
        finalists,
    )
    .unwrap();
    let sealed = SealedTestTableV1 {
        dataset_sha256: dataset.dataset_sha256.clone(),
        test_csv_path: temp
            .path()
            .join("must-not-be-read.csv")
            .display()
            .to_string(),
        test_sha256: digest("test"),
        test_row_ids: vec!["test-0".into()],
    };
    let error = adapter
        .evaluate_sealed_test(&dataset, &sealed, &wrong_selection)
        .unwrap_err();
    assert!(error.contains("selection freeze"));
    assert!(!error.contains("read sealed"));
}

#[test]
fn malformed_heads_missing_artifacts_and_incomplete_plans_fail_closed() {
    let binary_with_regression_head =
        GenomeV1::from_canonical(GpTaskV1::Binary, "(pipe X (standardize) (ridge))");
    assert!(binary_with_regression_head
        .unwrap_err()
        .contains("task-incompatible"));

    let missing = FrozenArtifactV1 {
        path: "/definitely/not/a/generalized-gp-checkpoint".into(),
        sha256: digest("missing"),
        byte_length: 1,
    };
    assert!(missing
        .verify()
        .unwrap_err()
        .contains("read frozen artifact"));

    let task = GpTaskV1::Regression;
    let source_rows = rows(task, 70);
    let mut plan = build_rotating_split_plan(
        task,
        &digest("fault dataset"),
        &digest("fault training"),
        &digest("fault seed"),
        &source_rows,
    )
    .unwrap();
    plan.splits[0].train_row_ids.pop();
    assert!(plan.validate().unwrap_err().contains("complete disjoint"));

    // Timeouts are typed invalid evidence, not crashes or synthetic scores.
    let timeout = CandidateSplitEvaluationV1 {
        valid: false,
        invalid_reason: Some("timeout".into()),
        runtime: RuntimeObservationV1 {
            cpu_seconds: 30.0,
            winner_latency_seconds: 30.0,
            timed_out: true,
        },
        ..evaluation(
            CampaignArmV1::RotatingMean,
            &digest("timeout dataset"),
            0,
            0,
            &digest("timeout split"),
            &genome(task, 0),
            0.0,
        )
    };
    timeout.validate().unwrap();
}
