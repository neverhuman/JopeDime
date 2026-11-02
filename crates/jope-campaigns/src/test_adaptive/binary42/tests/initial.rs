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
