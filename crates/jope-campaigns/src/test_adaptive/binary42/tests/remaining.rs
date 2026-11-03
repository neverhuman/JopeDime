#[test]
fn shards_account_for_exactly_sixty_slots_and_resume_ignores_partial_tail() {
    let members = (0..POPULATION)
        .map(|index| member(&format!("genome-{index}"), "seed", Vec::new()))
        .collect::<Vec<_>>();
    let shards = shard_members(&members).unwrap();
    assert_eq!(shards.len(), 4);
    assert!(shards.iter().all(|shard| shard.len() == SHARD_SIZE));
    assert_eq!(shards.iter().map(Vec::len).sum::<usize>(), POPULATION);

    let directory = tempdir().unwrap();
    fs::create_dir_all(generation_root(directory.path(), 0)).unwrap();
    assert!(super::indexes::load_checkpoints(directory.path())
        .unwrap()
        .is_empty());
}

#[test]
#[ignore = "deferred-classification: binary42 inputs require the explicit host-real-corpus lane"]
fn all_frozen_dataset_bindings_and_binary_labels_are_valid() {
    let (cohort, _) = read_cohort().unwrap();
    assert_eq!(cohort.datasets.len(), 42);
    for frozen in cohort.datasets {
        let spec = qualified_dataset(frozen);
        require_binary_input(&spec);
        verify_dataset_binding(&spec).unwrap();
        for path in [&spec.train_path, &spec.test_path] {
            let data = read_binary_csv(Path::new(path)).unwrap();
            assert!(data.y.iter().all(|label| *label == 0.0 || *label == 1.0));
            assert!(data.y.contains(&0.0));
            assert!(data.y.contains(&1.0));
        }
    }
}

#[test]
fn all_shards_and_six_dataset_cpu_budgets_are_exact() {
    let members = (0..POPULATION)
        .map(|index| member(&format!("all-genome-{index}"), "seed", Vec::new()))
        .collect::<Vec<_>>();
    let shards = shard_members_with_sizes(&members, &ALL_SHARD_SIZES).unwrap();
    assert_eq!(
        shards.iter().map(Vec::len).collect::<Vec<_>>(),
        ALL_SHARD_SIZES
    );
    assert_eq!(shards.iter().map(Vec::len).sum::<usize>(), POPULATION);
    let config = AllCohortConfig::fixed();
    assert_eq!(config.node_cpu_budget_per_active_set["xbabe1"], 114);
    assert_eq!(config.node_cpu_budget_per_active_set["xbabe2"], 114);
    assert_eq!(config.node_cpu_budget_per_active_set["xbabe3"], 114);
    assert_eq!(config.node_cpu_budget_per_active_set["xbabe4"], 18);
    for slot in 0..POPULATION {
        let expected = if slot < 19 {
            "xbabe1"
        } else if slot < 38 {
            "xbabe2"
        } else if slot < 57 {
            "xbabe3"
        } else {
            "xbabe4"
        };
        assert_eq!(
            assigned_node_for_slot(&ALL_SHARD_SIZES, slot).unwrap(),
            expected
        );
    }
}

#[test]
fn cost_order_work_conserving_admission_and_failover_are_deterministic() {
    let (cohort, _) = read_cohort().unwrap();
    let ordered = admission_order(&cohort);
    assert_eq!(ordered.len(), 42);
    for pair in ordered.windows(2) {
        let left_cost = pair[0].train_rows * pair[0].features;
        let right_cost = pair[1].train_rows * pair[1].features;
        assert!(
            left_cost > right_cost || (left_cost == right_cost && pair[0].order < pair[1].order)
        );
    }
    let mut pending = VecDeque::from(
        ordered
            .iter()
            .map(|spec| spec.id.clone())
            .collect::<Vec<_>>(),
    );
    let mut active = (0..ALL_ACTIVE_DATASETS)
        .map(|_| pending.pop_front().unwrap())
        .collect::<Vec<_>>();
    while let Some(next) = pending.pop_front() {
        active.remove(0);
        active.push(next);
        assert_eq!(active.len(), ALL_ACTIVE_DATASETS);
    }
    assert_eq!(failover_node(0, 0).unwrap(), "xbabe1");
    assert_eq!(failover_node(0, 1).unwrap(), "xbabe2");
    assert_eq!(failover_node(0, 2).unwrap(), "xbabe3");
    assert_eq!(failover_node(3, 1).unwrap(), "xbabe1");
    assert_eq!(failover_node(3, 2).unwrap(), "xbabe2");
    assert!(failover_node(0, 3).is_err());
}

#[test]
fn strict_saturated_and_exhausted_synthetic_cohort_outcomes_are_distinct() {
    let seeds = BTreeSet::from([sha256_bytes(b"seed")]);
    let strict_child = logical(0, "strict-child", "mutation", 0.81, 0.2, 2);
    assert!(qualifying_winner_for_policy(
        1,
        std::slice::from_ref(&strict_child),
        &seeds,
        "strictly_greater_auc",
        0.80,
        0.805,
    )
    .is_some());
    let saturated_child = logical(0, "saturated-child", "crossover", 1.0, 0.1, 2);
    assert!(qualifying_winner_for_policy(
        1,
        std::slice::from_ref(&saturated_child),
        &seeds,
        "exact_auc_1_tie",
        1.0,
        1.0,
    )
    .is_some());
    let near_saturated = logical(0, "near-saturated", "mutation", 0.999, 0.1, 2);
    assert!(qualifying_winner_for_policy(
        1,
        std::slice::from_ref(&near_saturated),
        &seeds,
        "exact_auc_1_tie",
        1.0,
        1.0,
    )
    .is_none());
    let exhausted_candidate = logical(0, "exhausted-child", "mutation", 0.70, 0.6, 2);
    for generation in 0..ALL_MAX_GENERATIONS {
        assert!(qualifying_winner_for_policy(
            generation,
            std::slice::from_ref(&exhausted_candidate),
            &seeds,
            "strictly_greater_auc",
            0.80,
            0.75,
        )
        .is_none());
    }
    let config = CampaignConfig::all(&read_cohort().unwrap().0.datasets[0]);
    assert!(!generation_cap_reached(&config, 8));
    assert!(generation_cap_reached(&config, 9));
    assert_eq!(config.generation_cap, Some(10));
}

#[test]
#[ignore = "deferred-classification: binary42 inputs require the explicit host-real-corpus lane"]
fn binary42_fitted_cpp17_predictions_match_rust_bits() {
    let (cohort, _) = read_cohort().unwrap();
    let spec = qualified_dataset(
        cohort
            .datasets
            .into_iter()
            .find(|dataset| dataset.id == ILPD)
            .unwrap(),
    );
    require_binary_input(&spec);
    let train = read_binary_csv(Path::new(&spec.train_path)).unwrap();
    let test = read_binary_csv(Path::new(&spec.test_path)).unwrap();
    let member = seed_population(spec.features).unwrap().remove(0);
    let rust = evaluate_test_adaptive_binary(&member.canonical_genome, &train, &test).unwrap();
    let directory = tempdir().unwrap();
    let cpp = evaluate_test_adaptive_binary_cpp17(
        &member.canonical_genome,
        &train,
        &test,
        directory.path(),
    )
    .unwrap();
    assert_eq!(
        cpp.evaluation
            .predictions
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        rust.predictions
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(cpp.evaluation.auc.to_bits(), rust.auc.to_bits());
    assert_eq!(cpp.evaluation.logloss.to_bits(), rust.logloss.to_bits());
    let replayed = parse_prediction_bytes(&binary_prediction_roundtrip_bytes(
        &cpp.evaluation.predictions,
    ))
    .unwrap();
    assert_eq!(
        auc(&test.y, &replayed).to_bits(),
        cpp.evaluation.auc.to_bits()
    );
    assert!(logloss_replays(
        cpp.evaluation.logloss,
        feat_math::metrics::logloss(&test.y, &replayed)
    ));
    assert!(is_sha256_hex(&cpp.source_sha256));
    assert!(is_sha256_hex(&cpp.codec_sha256));
    assert!(cpp.compile_seconds >= 0.0);
}

#[test]
fn pilot_serialization_omits_all_cohort_extensions() {
    let bytes = canonical_json(&CampaignConfig::fixed()).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(!text.contains("shard_sizes"));
    assert_eq!(CampaignConfig::fixed().generation_cap, None);
    let all = CampaignConfig::all(&read_cohort().unwrap().0.datasets[0]);
    assert_eq!(all.shard_sizes, ALL_SHARD_SIZES);
    assert_eq!(all.infrastructure_retries, 2);
}

#[test]
fn semantic_caches_are_dataset_local() {
    let directory = tempdir().unwrap();
    let first = directory.path().join("dataset-a");
    let second = directory.path().join("dataset-b");
    let record = CacheEntry {
        schema: CACHE_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        semantic_sha256: "a".repeat(64),
        receipt_sha256: "b".repeat(64),
        prediction_sha256: "c".repeat(64),
        score: Some(CandidateScore {
            auc: 0.75,
            logloss: 0.5,
        }),
        failure: None,
        evaluation_node: "xbabe1".to_string(),
        scq_job_id: "job".to_string(),
        first_generation: 0,
        first_slot: 0,
        runtime_seconds: Some(1.0),
    };
    write_json(&cache_path(&first, &record.semantic_sha256), &record).unwrap();
    assert_eq!(load_cache(&first).unwrap().len(), 1);
    assert!(load_cache(&second).unwrap().is_empty());
}
