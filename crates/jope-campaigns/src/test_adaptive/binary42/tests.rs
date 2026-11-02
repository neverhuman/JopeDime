use super::*;
use tempfile::tempdir;

fn qualified_dataset(mut spec: FrozenDataset) -> FrozenDataset {
    if let Some(root) = std::env::var_os("JOPE_BINARY42_TEST_DATA_ROOT") {
        let root = PathBuf::from(root).join(&spec.id);
        spec.train_path = root.join("train.csv").display().to_string();
        spec.test_path = root.join("test.csv").display().to_string();
    }
    let filename = Path::new(&spec.leader_report_path)
        .file_name()
        .expect("frozen leader report path has a filename")
        .to_owned();
    let historical = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/binary42-historical-reports/v1")
        .join(&spec.id)
        .join(&filename);
    if historical.is_file() {
        spec.leader_report_path = historical.display().to_string();
    } else if let Some(root) = std::env::var_os("JOPE_BINARY42_TEST_REPORT_ROOT") {
        spec.leader_report_path = PathBuf::from(root)
            .join(&spec.id)
            .join("reports")
            .join(&filename)
            .display()
            .to_string();
    }
    spec
}

/// Binary classification remains deferred. Its real-corpus inputs are named
/// when absent; the test still fails rather than skipping.
fn require_binary_input(spec: &FrozenDataset) {
    for path in [&spec.train_path, &spec.test_path] {
        assert!(
            Path::new(path).is_file(),
            "unavailable: binary42 real-corpus input {path} (set JOPE_BINARY42_TEST_DATA_ROOT)"
        );
    }
}

#[test]
fn confirmed_binary_winners_are_first_in_generation_zero() {
    let genomes = [
        "(pipe X (gam_boost lr=0.2 rounds=100))",
        "(pipe X (extra_trees depth=6 trees=32))",
    ];
    let sources = genomes
        .iter()
        .enumerate()
        .map(|(index, genome)| WinnerSeedSource {
            source_run_id: format!("source-{index}"),
            source_dataset: format!("{index:016x}"),
            complete_file: format!("complete-{index}"),
            complete_sha256: format!("complete-sha-{index}"),
            final_receipt_file: format!("receipt-{index}"),
            final_receipt_sha256: format!("receipt-sha-{index}"),
            confirmation_file: format!("confirmation-{index}"),
            confirmation_sha256: format!("confirmation-sha-{index}"),
            canonical_genome: (*genome).to_string(),
            semantic_sha256: sha256_bytes(genome.as_bytes()),
            auc_bits: 1.0f64.to_bits(),
            frozen_leader_auc_bits: 0.9f64.to_bits(),
            generation_zero_best_auc_bits: 0.95f64.to_bits(),
        })
        .collect::<Vec<_>>();
    let members = seed_population_with_winners(128, &sources).unwrap();
    assert_eq!(members.len(), POPULATION);
    assert_eq!(members[0].canonical_genome, genomes[0]);
    assert_eq!(members[1].canonical_genome, genomes[1]);
    assert_eq!(
        members
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<BTreeSet<_>>()
            .len(),
        POPULATION
    );
    assert!(members
        .iter()
        .all(|member| member.ancestry.operation == "seed"));
}

fn member(name: &str, operation: &str, parents: Vec<String>) -> MemberRecord {
    MemberRecord {
        canonical_genome: name.to_string(),
        semantic_sha256: sha256_bytes(name.as_bytes()),
        complexity: name.len(),
        ancestry: BinaryAncestry {
            operation: operation.to_string(),
            parents,
            retry_count: 0,
        },
    }
}

fn logical(
    slot: usize,
    name: &str,
    operation: &str,
    auc: f64,
    logloss: f64,
    complexity: usize,
) -> LogicalSlot {
    let mut member = member(name, operation, Vec::new());
    member.complexity = complexity;
    LogicalSlot {
        generation: 1,
        slot,
        assigned_node: "xbabe1".to_string(),
        member,
        cache_disposition: "fresh".to_string(),
        cache_source_generation: None,
        cache_source_slot: None,
        evaluation_node: "xbabe1".to_string(),
        scq_job_id: "job".to_string(),
        receipt_sha256: "r".repeat(64),
        prediction_sha256: "p".repeat(64),
        score: Some(CandidateScore { auc, logloss }),
        failure: None,
        runtime_seconds: None,
    }
}

#[test]
fn frozen_ilpd_and_cohort_contract_are_exact() {
    let (cohort, _) = read_cohort().unwrap();
    assert_eq!(cohort.datasets.len(), 42);
    assert_eq!(
        cohort
            .datasets
            .iter()
            .filter(|dataset| dataset.leader_auc == 1.0)
            .count(),
        6
    );
    let ilpd = cohort
        .datasets
        .iter()
        .find(|dataset| dataset.id == ILPD)
        .unwrap();
    assert_eq!(ilpd.train_rows, 485);
    assert_eq!(ilpd.test_rows, 98);
    assert_eq!(ilpd.features, 10);
    assert_eq!(ilpd.leader_auc.to_bits(), 0.7681159420289855f64.to_bits());
}

#[test]
fn binary_auc_uses_midranks_and_probabilities_are_bounded() {
    assert_eq!(auc(&[0.0, 1.0], &[0.5, 0.5]), 0.5);
    assert_eq!(
        parse_prediction_bytes(b"prediction\n0.0\n0.5\n1.0\n").unwrap(),
        vec![0.0, 0.5, 1.0]
    );
    assert!(parse_prediction_bytes(b"prediction\n1.0001\n").is_err());
    assert!(parse_prediction_bytes(b"prediction\nNaN\n").is_err());
    let tiny = [1e-20, 2e-20, f64::MIN_POSITIVE, 1.0 - f64::EPSILON];
    let roundtrip = parse_prediction_bytes(&binary_prediction_roundtrip_bytes(&tiny)).unwrap();
    assert_eq!(
        roundtrip
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        tiny.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
    );
    let score = 1.0967166826203898_f64;
    assert!(logloss_replays(score, f64::from_bits(score.to_bits() - 1)));
    assert!(logloss_replays(score, score + 9e-15));
    assert!(!logloss_replays(score, score + 2e-12));
}

#[test]
fn rank_chain_applies_all_tie_breakers() {
    let mut slots = [
        logical(0, "z", "mutation", 0.8, 0.4, 1),
        logical(1, "a", "mutation", 0.9, 0.5, 1),
        logical(2, "b", "mutation", 0.9, 0.4, 2),
        logical(3, "c", "mutation", 0.9, 0.4, 1),
        logical(4, "d", "mutation", 0.9, 0.4, 1),
    ];
    slots.sort_by(rank_slots);
    assert_eq!(slots[0].score.as_ref().unwrap().auc, 0.9);
    assert_eq!(slots[0].score.as_ref().unwrap().logloss, 0.4);
    assert_eq!(slots[0].member.complexity, 1);
    assert!(
        slots[0].member.semantic_sha256 < slots[1].member.semantic_sha256
            || slots[1].member.complexity > slots[0].member.complexity
    );
    assert_eq!(slots.last().unwrap().score.as_ref().unwrap().auc, 0.8);
}

#[test]
fn strict_and_saturated_comparators_are_distinct() {
    assert!(cohort_win(0.81, 0.8));
    assert!(!cohort_win(0.8, 0.8));
    assert!(cohort_win(1.0, 1.0));
    assert!(!cohort_win(0.999999999999, 1.0));
    assert!(strict_win(0.81, 0.8, 0.79));
    assert!(!strict_win(0.81, 0.8, 0.81));
}

#[test]
fn canonical_binary_offspring_are_deterministic() {
    let seeds = seed_population(10).unwrap();
    let ranked = seeds
        .iter()
        .enumerate()
        .map(|(index, member)| BinaryAdaptiveRanked {
            canonical_genome: member.canonical_genome.clone(),
            semantic_sha256: member.semantic_sha256.clone(),
            complexity: member.complexity,
            auc: 1.0 - index as f64 / 1000.0,
            logloss: index as f64 / 1000.0,
        })
        .collect::<Vec<_>>();
    let first = test_adaptive_next_population(RNG_SEED, 1, &ranked, 10)
        .unwrap()
        .into_iter()
        .map(MemberRecord::from_binary)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let second = test_adaptive_next_population(RNG_SEED, 1, &ranked, 10)
        .unwrap()
        .into_iter()
        .map(MemberRecord::from_binary)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), POPULATION);
    assert_eq!(
        first
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<BTreeSet<_>>()
            .len(),
        POPULATION
    );
    assert_eq!(
        first
            .iter()
            .filter(|member| member.ancestry.operation == "elite")
            .count(),
        ELITES
    );
}

#[test]
fn semantic_cache_is_global_and_collisions_fail_closed() {
    let directory = tempdir().unwrap();
    let root = directory.path();
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
        runtime_seconds: None,
    };
    write_json(&cache_path(root, &record.semantic_sha256), &record).unwrap();
    write_json(&cache_path(root, &record.semantic_sha256), &record).unwrap();
    let loaded = load_cache(root).unwrap();
    assert_eq!(loaded.len(), 1);
    let mut changed = record.clone();
    changed.receipt_sha256 = "d".repeat(64);
    assert!(write_json(&cache_path(root, &record.semantic_sha256), &changed).is_err());
}

#[test]
fn two_generation_fixture_selects_only_an_evolved_winner() {
    let seed = logical(0, "seed", "seed", 0.77, 0.5, 1);
    let seeds = BTreeSet::from([seed.member.semantic_sha256.clone()]);
    assert!(qualifying_winner(0, std::slice::from_ref(&seed), &seeds, 0.76, 0.77).is_none());
    let child = logical(0, "child", "mutation", 0.78, 0.4, 2);
    let winner = qualifying_winner(1, std::slice::from_ref(&child), &seeds, 0.76, 0.77).unwrap();
    assert_eq!(winner.semantic_sha256, child.member.semantic_sha256);
    assert_eq!(winner.auc, 0.78);
    let records = [seed, child];
    assert!(records.iter().all(|record| record.score.is_some()));
    assert_eq!(
        records
            .iter()
            .map(|record| &record.member.semantic_sha256)
            .collect::<BTreeSet<_>>()
            .len(),
        2
    );
}

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
    assert!(load_checkpoints(directory.path()).unwrap().is_empty());
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
