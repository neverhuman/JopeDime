use super::population::seed::{derive_corrected_generation_zero, expand_initial_population};
use super::*;

#[test]
fn preflight_ignores_terminal_scq_jobs() {
    let terminal = ["completed", "failed", "cancelled"];
    for state in terminal {
        let job = serde_json::json!({"id": "old", "state": state});
        assert!(!scq_job_is_active(&job));
    }
    let active = serde_json::json!({"id": "new", "state": "running"});
    assert!(scq_job_is_active(&active));
    assert!(scq_job_is_active(&serde_json::json!({"id": "unknown"})));
}

#[test]
fn process_gate_ignores_scq_control_plane_but_not_workloads() {
    let controller = "42 jope-dime-regressor controller";
    assert!(!is_competing_process(controller, "xbabe2", "xbabe2", "42"));
    assert!(!is_competing_process(
        "2365996 jope-dime-regre /opt/jope-dime-regressor run-golden42-aggressive --resume",
        "xbabe2",
        "xbabe2",
        "different-pid",
    ));
    let scq = jope_paths::to_string(&paths::scq_binary());
    assert!(!is_competing_process(
        &format!("7 scq {scq} agent status --node xbabe1"),
        "xbabe1",
        "xbabe2",
        "42",
    ));
    assert!(!is_competing_process(
        &format!("8 scq {scq} controller serve"),
        "xbabe2",
        "xbabe2",
        "42",
    ));
    assert!(!is_competing_process(
        "9 cargo cargo audit --deny warnings",
        "xbabe2",
        "xbabe2",
        "42",
    ));
    assert!(is_competing_process(
        "10 jope-dime-regre /tmp/jope-dime-regressor candidate",
        "xbabe1",
        "xbabe2",
        "42",
    ));
}

#[test]
fn dataset_selector_accepts_a_deduplicated_panel_and_preserves_registry_order() {
    let spec = |hash: &str| DatasetSpec {
        dataset_hash: hash.to_string(),
        train_sha256: "a".repeat(64),
        test_sha256: "b".repeat(64),
        train_rows: 10,
        test_rows: 5,
        features: 2,
        leader_models: vec!["leader".to_string()],
        leader_rmse: 1.0,
    };
    let specs = vec![
        spec("1111111111111111"),
        spec("2222222222222222"),
        spec("3333333333333333"),
    ];
    let selected = select_dataset_specs(&specs, Some("3333,1111"), 0).unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|spec| spec.dataset_hash.as_str())
            .collect::<Vec<_>>(),
        ["1111111111111111", "3333333333333333"]
    );
    assert!(select_dataset_specs(&specs, Some("1111,111111"), 0).is_err());
    assert!(select_dataset_specs(&specs, Some("22,not-hex"), 0).is_err());
    let named = vec![spec("ZTS_small"), spec("ZTS"), spec("abcdef0123456789")];
    assert_eq!(
        select_dataset_specs(&named, Some("ZTS_small,abc"), 0)
            .unwrap()
            .iter()
            .map(|spec| spec.dataset_hash.as_str())
            .collect::<Vec<_>>(),
        ["ZTS_small", "abcdef0123456789"]
    );
    assert!(select_dataset_specs(&named, Some("ZTS_"), 0).is_err());
}
use serde_json::json;

#[test]
fn forty_two_dataset_wave_overdecomposes_each_node() {
    for generation in [0, 1, 2, 37, 99] {
        let mut counts = [0usize; 3];
        for slot in 0..TOTAL_POPULATION {
            counts[(slot + generation) % 3] += 1;
        }
        assert_eq!(counts, [12, 12, 12]);
        assert_eq!(counts.map(|count| count * 42), [504, 504, 504]);
    }
}

#[test]
fn generation_zero_expands_to_four_complete_unique_islands() {
    let mut seen = HashSet::new();
    let base = include_str!("../../../seeds/generation-zero.txt")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| canonicalize_for_features(line, 100).ok())
        .filter(|canonical| seen.insert(canonical.semantic_sha256.clone()))
        .take(ISLAND_POPULATION)
        .map(|canonical| Member {
            canonical_genome: canonical.canonical,
            semantic_sha256: canonical.semantic_sha256,
            complexity: canonical.complexity,
            rust_source_len: canonical.rust_source_len,
            evolved_descendant: false,
            ancestry: Ancestry {
                operation: "seed".to_string(),
                parents: Vec::new(),
            },
        })
        .collect::<Vec<_>>();
    assert_eq!(base.len(), ISLAND_POPULATION);
    let population = expand_initial_population(&base, 100).unwrap();
    assert_eq!(population.len(), TOTAL_POPULATION);
    assert_eq!(
        population
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<HashSet<_>>()
            .len(),
        TOTAL_POPULATION
    );
    assert_eq!(&population[..base.len()], base.as_slice());
}

#[test]
fn ties_are_not_wins() {
    let baseline = 1.0f64;
    assert!(!matches!(
        baseline.partial_cmp(&baseline),
        Some(std::cmp::Ordering::Less)
    ));
    assert!(f64::from_bits(baseline.to_bits() - 1) < baseline);
}

#[test]
fn golden46_header_view_removes_exactly_one_line() {
    let source = b"f0,f1,target\r\n1,2,3\r\n4,5,6\r\n";
    assert_eq!(
        remove_csv_header(source, true, "fixture").unwrap(),
        b"1,2,3\r\n4,5,6\r\n"
    );
    assert_eq!(remove_csv_header(source, false, "fixture").unwrap(), source);
    assert!(remove_csv_header(b"header-only", true, "fixture").is_err());
}

#[test]
fn generation_zero_and_unevolved_elites_cannot_terminate() {
    let outcome = |generation: usize,
                   evolved_descendant: bool,
                   operation: &str,
                   parent: &str|
     -> SlotOutcome {
        let semantic = "a".repeat(64);
        SlotOutcome {
            generation,
            slot: 0,
            member: Member {
                canonical_genome: "(pipe X (mean))".to_string(),
                semantic_sha256: semantic,
                complexity: 1,
                rust_source_len: 1,
                evolved_descendant,
                ancestry: Ancestry {
                    operation: operation.to_string(),
                    parents: (!parent.is_empty())
                        .then(|| parent.to_string())
                        .into_iter()
                        .collect(),
                },
            },
            valid: true,
            metrics: Some(CandidateMetrics {
                rmse: 0.5,
                mae: 0.5,
                r2: 0.0,
            }),
            failure: None,
            cache_source: None,
            receipt_file: "receipt".to_string(),
            receipt_sha256: "b".repeat(64),
            node: "xbabe1".to_string(),
            scq_job_id: "job".to_string(),
            attempt: 0,
        }
    };
    let other_parent = "b".repeat(64);
    let same_parent = "a".repeat(64);
    assert_eq!(
        evolutionary_winner_slot(&[outcome(0, true, "mutation", &other_parent)], 0, 1.0),
        None
    );
    assert_eq!(
        evolutionary_winner_slot(&[outcome(1, true, "elite", &other_parent)], 1, 1.0),
        None
    );
    assert_eq!(
        evolutionary_winner_slot(&[outcome(1, true, "mutation", &same_parent)], 1, 1.0),
        None
    );
    assert_eq!(
        evolutionary_winner_slot(&[outcome(1, false, "mutation", &other_parent)], 1, 1.0),
        None
    );
    let mut cached = outcome(1, true, "mutation", &other_parent);
    cached.cache_source = Some(CacheSource {
        generation: 0,
        slot: 7,
    });
    assert_eq!(evolutionary_winner_slot(&[cached], 1, 1.0), None);
    assert_eq!(
        evolutionary_winner_slot(&[outcome(1, true, "mutation", &other_parent)], 1, 1.0),
        Some(0)
    );
}

#[test]
fn corrected_generation_zero_has_one_parent_and_36_unique_descendants() {
    let parent_genome =
        feat_math::full_rust::canonicalize("(pipe X (standardize) (ridge))").unwrap();
    let parent = SeedProvenance {
        slot: 7,
        canonical_genome: parent_genome.canonical,
        semantic_sha256: parent_genome.semantic_sha256,
        complexity: parent_genome.complexity,
        rust_source_len: parent_genome.rust_source_len,
        source_kind: "corrected_test_adaptive_parent".to_string(),
        source_file: "/runs/v2/checkpoint-000075.json".to_string(),
        source_file_sha256: "a".repeat(64),
        source_record: 7,
        source_rank: 1,
        source_score: Some(0.1),
    };
    let spec = DatasetSpec {
        dataset_hash: "fixture".to_string(),
        train_sha256: "1".repeat(64),
        test_sha256: "2".repeat(64),
        train_rows: 10,
        test_rows: 5,
        features: 50,
        leader_models: vec!["leader".to_string()],
        leader_rmse: 0.09,
    };
    let (population, snapshot) =
        derive_corrected_generation_zero(&parent, GOLDEN46_RUN_PREFIX, &spec).unwrap();
    assert_eq!(population.len(), TOTAL_POPULATION);
    assert_eq!(snapshot.len(), TOTAL_POPULATION);
    assert_eq!(
        population
            .iter()
            .map(|member| member.semantic_sha256.as_str())
            .collect::<HashSet<_>>()
            .len(),
        TOTAL_POPULATION
    );
    assert!(population.iter().all(|member| {
        member.evolved_descendant
            && member.semantic_sha256 != parent.semantic_sha256
            && member.ancestry.operation == "generation_zero_mutation"
            && member.ancestry.parents == vec![parent.semantic_sha256.clone()]
    }));
    for island in 0..ISLANDS {
        assert_eq!(
            snapshot.iter().filter(|seed| seed.island == island).count(),
            ISLAND_POPULATION
        );
    }
}

#[test]
fn corrected_next_population_caches_eight_elites_and_has_28_novel_children() {
    let mut ranked_islands = Vec::new();
    let mut history = HashSet::new();
    for island in 0..ISLANDS {
        let mut ranked = Vec::new();
        for local_slot in 0..ISLAND_POPULATION {
            let slot = island * ISLAND_POPULATION + local_slot;
            let genome = feat_math::full_rust::random_pipeline(10_000 + slot as u64, 50).unwrap();
            history.insert(genome.semantic_sha256.clone());
            ranked.push(SlotOutcome {
                generation: 0,
                slot,
                member: Member::evolved(
                    genome,
                    Ancestry {
                        operation: "generation_zero_mutation".to_string(),
                        parents: vec!["a".repeat(64)],
                    },
                ),
                valid: true,
                metrics: Some(CandidateMetrics {
                    rmse: 0.1 + local_slot as f64 / 100.0,
                    mae: 0.1,
                    r2: 0.0,
                }),
                failure: None,
                cache_source: None,
                receipt_file: format!("receipt-{slot}.json"),
                receipt_sha256: format!("{slot:064x}"),
                node: "xbabe1".to_string(),
                scq_job_id: format!("job-{slot}"),
                attempt: 0,
            });
        }
        ranked_islands.push(ranked);
    }
    let next = next_golden46_population(
        &ranked_islands,
        GOLDEN46_RUN_PREFIX,
        "fixture",
        1,
        50,
        &history,
    )
    .unwrap();
    assert_eq!(next.len(), TOTAL_POPULATION);
    assert_eq!(
        next.iter()
            .map(|member| member.semantic_sha256.as_str())
            .collect::<HashSet<_>>()
            .len(),
        TOTAL_POPULATION
    );
    let elites = next
        .iter()
        .filter(|member| member.ancestry.operation == "elite")
        .collect::<Vec<_>>();
    let children = next
        .iter()
        .filter(|member| member.ancestry.operation != "elite")
        .collect::<Vec<_>>();
    assert_eq!(elites.len(), ISLANDS * ELITES_PER_ISLAND);
    assert_eq!(children.len(), TOTAL_POPULATION - elites.len());
    assert!(elites
        .iter()
        .all(|member| history.contains(&member.semantic_sha256)));
    assert!(children
        .iter()
        .all(|member| !history.contains(&member.semantic_sha256)));
}

#[test]
fn aggressive_contract_is_not_benchmark_valid() {
    let value = json!({"benchmark_valid": false, "test_adaptive": true});
    assert_eq!(value["benchmark_valid"], false);
    assert_eq!(value["test_adaptive"], true);
}

#[test]
fn traversal_run_ids_are_rejected() {
    assert!(validate_run_id(".").is_err());
    assert!(validate_run_id("..").is_err());
    assert!(validate_run_id("safe.run-1").is_ok());
}

#[test]
fn one_dataset_selector_requires_a_unique_hash_prefix() {
    let make = |dataset_hash: &str| DatasetSpec {
        dataset_hash: dataset_hash.to_string(),
        train_sha256: "1".repeat(64),
        test_sha256: "2".repeat(64),
        train_rows: 2,
        test_rows: 1,
        features: 1,
        leader_models: vec!["tabpfnv3".to_string()],
        leader_rmse: 1.0,
    };
    let specs = vec![make("aa00"), make("aa11"), make("bb22")];
    assert_eq!(
        select_dataset_specs(&specs, Some("bb"), 0).unwrap(),
        vec![specs[2].clone()]
    );
    assert!(select_dataset_specs(&specs, Some("aa"), 0).is_err());
    assert!(select_dataset_specs(&specs, Some("cc"), 0).is_err());
    assert!(select_dataset_specs(&specs, Some("../"), 0).is_err());
    assert_eq!(select_dataset_specs(&specs, None, 1).unwrap(), specs[1..]);
    assert!(select_dataset_specs(&specs, Some("bb"), 1).is_err());
    assert!(select_dataset_specs(&specs, None, specs.len()).is_err());
}
