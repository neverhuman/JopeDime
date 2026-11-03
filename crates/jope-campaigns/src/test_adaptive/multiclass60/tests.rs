use super::*;

fn score(f1: f64, accuracy: f64, complexity: usize, hash: &str) -> RankedScore {
    RankedScore {
        macro_f1: f1,
        accuracy,
        complexity,
        semantic_sha256: hash.repeat(64),
    }
}

#[test]
fn frozen_config_matches_campaign_budget_contract() {
    let config = Config::fixed();
    assert_eq!(config.population, 60);
    assert_eq!(config.shard_sizes, [15, 15, 15, 15]);
    assert_eq!(config.elites, 7);
    assert_eq!(config.parent_pool, 30);
    assert_eq!(config.tournament, 3);
    assert_eq!(config.mutation_probability, 0.75);
    assert_eq!(config.crossover_probability, 0.25);
    assert_eq!(config.minimum_generations, 0);
    assert_eq!(config.maximum_generations, 20);
    assert_eq!(config.candidate_timeout_seconds, 180);
    assert_eq!(config.infrastructure_retries, 1);
}

#[test]
fn replay_specs_route_three_roles_by_fail_closed_capability() {
    let dataset = DatasetBinding {
        order: 0,
        wave: "fixture".to_string(),
        dataset_hash: "fixture".to_string(),
        train_path: "train.csv".to_string(),
        train_sha256: "1".repeat(64),
        test_path: "test.csv".to_string(),
        test_sha256: "2".repeat(64),
        meta_path: "meta.json".to_string(),
        meta_sha256: "3".repeat(64),
        class_count: 4,
        n_features: 12,
        fast_baselines_complete: true,
        completed_incumbent_count: 43,
        comparator_sha256: "4".repeat(64),
        strongest_available_incumbent: None,
    };
    let champion = SeedMember {
        slot: 0,
        canonical_genome: "input | standardize | multiclass_ridge".to_string(),
        semantic_sha256: "5".repeat(64),
        source: "fixture".to_string(),
    };
    for (node, marker) in REPLAY_ROLE_CAPABILITIES {
        let rendered = render_replay_spec(
            &dataset,
            &champion,
            node,
            0,
            Path::new("manifest.json"),
            &"6".repeat(64),
            &"7".repeat(64),
        )
        .unwrap();
        assert!(rendered.contains("[[dependencies.smoke]]"));
        assert!(rendered.contains(marker));
        assert!(!rendered.contains("/etc/hostname"));
        assert!(!rendered.contains("\nnode = "));
        assert!(rendered.contains("-r3\""));
    }
    assert!(replay_role_capability(NODES[3]).is_err());
}

#[test]
fn scheduler_kill_without_exit_status_is_preemption_not_retry_failure() {
    let preempted = json!({
        "state": "failed",
        "failure_class": "crash",
        "error_reason": "killed (no exit status)",
    });
    assert!(scheduler_preemption(&preempted));
    assert!(!scheduler_preemption(&json!({
        "state": "failed",
        "failure_class": "crash",
        "error_reason": "segmentation fault",
    })));
}

#[test]
fn scq_specs_use_integer_runtime_limits_and_revisioned_ids() {
    let dataset = DatasetBinding {
        order: 0,
        wave: "fixture".to_string(),
        dataset_hash: "fixture".to_string(),
        train_path: "train.csv".to_string(),
        train_sha256: "1".repeat(64),
        test_path: "test.csv".to_string(),
        test_sha256: "2".repeat(64),
        meta_path: "meta.json".to_string(),
        meta_sha256: "3".repeat(64),
        class_count: 4,
        n_features: 12,
        fast_baselines_complete: true,
        completed_incumbent_count: 41,
        comparator_sha256: "4".repeat(64),
        strongest_available_incumbent: None,
    };
    let rendered = render_spec(
        Path::new("/tmp"),
        &dataset,
        0,
        0,
        0,
        Path::new("manifest.json"),
        &"5".repeat(64),
        &"6".repeat(64),
    );
    assert!(rendered.contains("max_runtime_hours = 1"));
    assert!(!rendered.contains("max_runtime_hours = 0.25"));
    assert_eq!(
        shard_job_id("fixture", 0, 0, 0, &"5".repeat(64)),
        "jope-mc272-fixture-g000-s0-a0-m555555555555-r2"
    );
    assert_ne!(
        shard_job_id("fixture", 0, 0, 0, &"5".repeat(64)),
        shard_job_id("fixture", 0, 0, 0, &"7".repeat(64)),
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("job.toml");
    fs::write(&path, rendered).unwrap();
    assert_eq!(
        spec_job_id(&path).unwrap(),
        "jope-mc272-fixture-g000-s0-a0-m555555555555-r2"
    );
}

#[test]
fn missing_comparator_keeps_dataset_waiting_without_blocking_plan() {
    let directory = tempfile::tempdir().unwrap();
    let dataset = DatasetBinding {
        order: 0,
        wave: "all-272".to_string(),
        dataset_hash: "fixture".to_string(),
        train_path: "train.csv".to_string(),
        train_sha256: "1".repeat(64),
        test_path: "test.csv".to_string(),
        test_sha256: "2".repeat(64),
        meta_path: "meta.json".to_string(),
        meta_sha256: "3".repeat(64),
        class_count: 4,
        n_features: 12,
        fast_baselines_complete: false,
        completed_incumbent_count: 0,
        comparator_sha256: String::new(),
        strongest_available_incumbent: None,
    };
    let gate = CoverageGate {
        multiclass_receipt_sha256: "a".repeat(64),
        timeseries_receipt_sha256: "b".repeat(64),
        comparator_set_path: "fixture".to_string(),
        comparator_set_sha256: "c".repeat(64),
    };
    assert!(bind_frozen_comparator(dataset, directory.path(), &gate)
        .unwrap()
        .is_none());
}

#[test]
fn started_datasets_keep_admission_priority_over_new_comparators() {
    let directory = tempfile::tempdir().unwrap();
    let binding = |order, dataset_hash: &str| DatasetBinding {
        order,
        wave: "fixture".to_string(),
        dataset_hash: dataset_hash.to_string(),
        train_path: "train.csv".to_string(),
        train_sha256: "1".repeat(64),
        test_path: "test.csv".to_string(),
        test_sha256: "2".repeat(64),
        meta_path: "meta.json".to_string(),
        meta_sha256: "3".repeat(64),
        class_count: 4,
        n_features: 12,
        fast_baselines_complete: true,
        completed_incumbent_count: 41,
        comparator_sha256: "4".repeat(64),
        strongest_available_incumbent: None,
    };
    let generation = directory.path().join("datasets/started/generation-000");
    fs::create_dir_all(&generation).unwrap();
    fs::write(generation.join("shard-0.manifest.json"), b"{}").unwrap();
    let ready = vec![binding(0, "waiting"), binding(1, "started")];
    let active = super::ledger::active_dataset_bindings(&ready, directory.path());
    assert_eq!(active[0].dataset_hash, "started");
    assert_eq!(active[1].dataset_hash, "waiting");
}

#[test]
fn scheduler_never_invents_post_generation_19_continuations() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoints = directory.path().join("datasets/dataset/checkpoints");
    fs::create_dir_all(&checkpoints).unwrap();
    for generation in 0..MAX_GENERATIONS {
        fs::write(
            checkpoints.join(format!("generation-{generation:03}.v1.json")),
            b"{}",
        )
        .unwrap();
    }
    assert_eq!(
        next_generation_index(directory.path(), "dataset"),
        MAX_GENERATIONS
    );
    assert!(initial_budget_complete(directory.path(), "dataset"));
}

#[test]
fn ranking_uses_macro_f1_accuracy_complexity_then_hash() {
    let mut values = [
        score(0.8, 0.9, 4, "c"),
        score(0.8, 0.9, 3, "b"),
        score(0.8, 0.8, 1, "a"),
        score(0.9, 0.1, 99, "d"),
    ];
    values.sort_by(rank_order);
    assert_eq!(values[0].macro_f1, 0.9);
    assert_eq!(values[1].complexity, 3);
    assert_eq!(values[2].complexity, 4);
}

#[test]
fn strict_win_and_ceiling_match_can_terminate_at_generation_zero() {
    let winner = score(0.91, 0.8, 3, "a");
    assert!(qualifies(&winner, 0, false, 0.8, 0.9, true));
    assert!(qualifies(&winner, 8, true, 0.8, 0.9, true));
    assert!(qualifies(&winner, 9, false, 0.8, 0.9, true));
    let saturated = score(1.0, 1.0, 3, "b");
    assert!(qualifies(&saturated, 0, true, 1.0, 1.0, false));
    assert!(qualifies(&saturated, 9, false, 0.9, 1.0, true));
    let tied = score(0.9, 1.0, 1, "c");
    assert!(!qualifies(&tied, 9, false, 0.8, 0.9, true));
}

#[test]
fn generation_zero_rewrite_is_native_multiclass_and_deterministic() {
    let source = "(pipe X (standardize) (ridge lambda_frac=0.02))";
    let first = rewrite_seed(source, 0).unwrap();
    let second = rewrite_seed(source, 0).unwrap();
    assert_eq!(first, second);
    assert!(first.canonical_genome.contains("multiclass_ridge"));
    assert!(!first.canonical_genome.contains("(ridge "));
    assert!(rewrite_seed("(pipe X (target kind=log1p) (ridge lambda_frac=0.02))", 0).is_err());
    let binary_source = rewrite_seed("(pipe X (standardize) (logistic))", 0).unwrap();
    assert!(binary_source.canonical_genome.contains("multiclass_ridge"));
    assert!(!binary_source.canonical_genome.contains("logistic"));
}

#[test]
fn checked_in_seed_file_rewrites_to_exact_unique_population() {
    let members = seed_population(Path::new("seeds/generation-zero.txt")).unwrap();
    assert_eq!(members.len(), 60);
    assert_eq!(
        members
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<BTreeSet<_>>()
            .len(),
        60
    );
    let mut families = BTreeMap::new();
    for member in &members {
        assert!(!member.canonical_genome.contains("(target "));
        *families
            .entry(multiclass_head_family(&member.canonical_genome).unwrap())
            .or_insert(0usize) += 1;
    }
    assert_eq!(families.len(), 4);
    assert!(families.values().all(|count| *count == 15));
}

#[test]
fn mutation_and_crossover_keep_one_native_multiclass_head_and_family_cap() {
    let seeds = seed_population(Path::new("seeds/generation-zero.txt")).unwrap();
    let mut ranked = seeds
        .iter()
        .enumerate()
        .map(|(slot, member)| LogicalSlot {
            slot,
            member: member.clone(),
            score: Some(RankedScore {
                macro_f1: 1.0 - slot as f64 / 1_000.0,
                accuracy: 1.0 - slot as f64 / 2_000.0,
                complexity: genome_complexity(&member.canonical_genome).unwrap(),
                semantic_sha256: member.semantic_sha256.clone(),
            }),
            valid: true,
            failure: None,
            receipt_sha256: "a".repeat(64),
            predictions_sha256: Some("b".repeat(64)),
            probabilities_sha256: Some("c".repeat(64)),
            node: "xbabe1".to_string(),
        })
        .collect::<Vec<_>>();
    ranked.sort_by(slot_rank_order);
    let dataset = DatasetBinding {
        order: 0,
        wave: "all-272".to_string(),
        dataset_hash: "fixture".to_string(),
        train_path: "train.csv".to_string(),
        train_sha256: "1".repeat(64),
        test_path: "test.csv".to_string(),
        test_sha256: "2".repeat(64),
        meta_path: "meta.json".to_string(),
        meta_sha256: "3".repeat(64),
        class_count: 4,
        n_features: 48,
        fast_baselines_complete: true,
        completed_incumbent_count: 41,
        comparator_sha256: "4".repeat(64),
        strongest_available_incumbent: None,
    };
    let first = next_population(&dataset, &ranked, 1).unwrap();
    let second = next_population(&dataset, &ranked, 1).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 60);
    let mut families = BTreeMap::new();
    for member in &first {
        let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(&member.canonical_genome).unwrap()
        else {
            panic!("member is not a pipeline")
        };
        assert_eq!(
            pipeline
                .stages
                .iter()
                .filter(|stage| MULTICLASS_HEADS.contains(&stage.op.as_str()))
                .count(),
            1
        );
        assert!(pipeline.stages.iter().all(|stage| stage.op != "target"));
        *families
            .entry(multiclass_head_family(&member.canonical_genome).unwrap())
            .or_insert(0usize) += 1;
    }
    assert!(families.values().all(|count| *count <= 15));

    let mut sparse = ranked.clone();
    for slot in sparse.iter_mut().skip(21) {
        slot.valid = false;
        slot.score = None;
        slot.failure = Some("candidate-invalid fixture".to_string());
    }
    sparse.sort_by(slot_rank_order);
    let sparse_first = next_population(&dataset, &sparse, 1).unwrap();
    let sparse_second = next_population(&dataset, &sparse, 1).unwrap();
    assert_eq!(sparse_first, sparse_second);
    assert_eq!(sparse_first.len(), 60);

    let mut all_invalid = ranked.clone();
    for slot in &mut all_invalid {
        slot.valid = false;
        slot.score = None;
        slot.failure = Some("candidate-invalid fixture".to_string());
    }
    all_invalid.sort_by(slot_rank_order);
    let recovered_first = next_population(&dataset, &all_invalid, 1).unwrap();
    let recovered_second = next_population(&dataset, &all_invalid, 1).unwrap();
    assert_eq!(recovered_first, recovered_second);
    assert_eq!(recovered_first.len(), 60);
    assert!(recovered_first.iter().all(|member| member
        .source
        .starts_with("all-invalid-reseed:generation=1:")));
}

#[test]
fn scq_spec_content_addresses_native_evaluator_and_manifest() {
    let dataset = DatasetBinding {
        order: 0,
        wave: "all-272".to_string(),
        dataset_hash: "fixture".to_string(),
        train_path: "train.csv".to_string(),
        train_sha256: "1".repeat(64),
        test_path: "test.csv".to_string(),
        test_sha256: "2".repeat(64),
        meta_path: "meta.json".to_string(),
        meta_sha256: "3".repeat(64),
        class_count: 4,
        n_features: 48,
        fast_baselines_complete: true,
        completed_incumbent_count: 41,
        comparator_sha256: "4".repeat(64),
        strongest_available_incumbent: None,
    };
    let manifest_sha256 = "a".repeat(64);
    let evaluator_sha256 = "b".repeat(64);
    let spec = render_spec(
        &paths::jope_state(),
        &dataset,
        0,
        0,
        0,
        Path::new("/tmp/manifest.json"),
        &manifest_sha256,
        &evaluator_sha256,
    );
    assert!(spec.contains("command = \"/usr/bin/env\""));
    assert!(spec.contains("{input:evaluator}"));
    assert!(spec.contains("{input:manifest}"));
    assert!(spec.contains(&format!(
        "source = {corpus_eval_str:?}",
        corpus_eval_str = corpus_eval_str()
    )));
    assert!(spec.contains(&format!("sha256 = {evaluator_sha256:?}")));
    assert!(spec.contains(&format!("sha256 = {manifest_sha256:?}")));
    assert!(!spec.to_ascii_lowercase().contains("python"));
}
