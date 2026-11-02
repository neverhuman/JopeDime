use super::*;

fn fixture_dataset(name: &str) -> EncodedDataset {
    let binding = TabArenaSourceBinding {
        dataset: name.to_string(),
        source_train_sha256: "a".repeat(64),
        source_test_sha256: "b".repeat(64),
        metadata_sha256: "c".repeat(64),
        encoded_train_sha256: "d".repeat(64),
        encoded_test_sha256: "e".repeat(64),
        train_rows: 10,
        test_rows: 5,
        features: 2,
        categorical_features: 0,
        leader_models: vec!["leader".to_string()],
        leader_rmse: 1.0,
    };
    EncodedDataset {
        spec: DatasetSpec {
            dataset_hash: name.to_string(),
            train_sha256: binding.encoded_train_sha256.clone(),
            test_sha256: binding.encoded_test_sha256.clone(),
            train_rows: binding.train_rows,
            test_rows: binding.test_rows,
            features: binding.features,
            leader_models: binding.leader_models.clone(),
            leader_rmse: binding.leader_rmse,
        },
        binding,
    }
}

#[test]
fn dataset_selector_is_exact_ordered_and_fail_closed() {
    let datasets = vec![
        fixture_dataset("a"),
        fixture_dataset("b"),
        fixture_dataset("c"),
    ];
    let selected = select_datasets(datasets.clone(), Some("c,a")).unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|dataset| dataset.spec.dataset_hash.as_str())
            .collect::<Vec<_>>(),
        ["a", "c"]
    );
    assert!(select_datasets(datasets.clone(), Some("a,a")).is_err());
    assert!(select_datasets(datasets, Some("missing")).is_err());
}

#[test]
fn canonical_leaders_are_fold_zero_only_and_parse_quoted_models() {
    let canonical = concat!(
        "method,variant,dataset,fold,metric,metric_value,metric_error\n",
        "\"Slow, quoted\",,data,0,rmse,2.0,2.0\n",
        "wrong-fold,,data,1,rmse,0.1,0.1\n",
        "winner,,data,0,rmse,1.0,1.0\n",
    );
    let mut leaders = HashMap::new();
    collect_canonical_leaders(canonical, &mut leaders).unwrap();
    assert_eq!(leaders["data"].0, 1.0);
    assert_eq!(leaders["data"].1, BTreeSet::from(["winner".to_string()]));
}

#[test]
fn leader_updates_preserve_exact_ties_and_replace_slower_models() {
    let mut leaders = HashMap::new();
    update_leader(&mut leaders, "data", "one", 2.0);
    update_leader(&mut leaders, "data", "two", 1.0);
    update_leader(&mut leaders, "data", "three", 1.0);
    assert_eq!(leaders["data"].0, 1.0);
    assert_eq!(
        leaders["data"].1,
        BTreeSet::from(["three".to_string(), "two".to_string()])
    );
}

/// Seeds exactly as production freezes them (canonicalized, then
/// deduplicated by semantic hash), from the bank checked in with this
/// crate rather than whichever bank the host happens to publish.
fn checked_in_transfer_seeds(features: usize) -> Vec<SeedProvenance> {
    let sources = fast30_bank_sources_from(&checked_in_fast30_bank()).unwrap();
    assert!(!sources.is_empty());
    for source in &sources {
        if let Err(error) = canonicalize_for_features(&source.genome, features) {
            panic!(
                "checked-in bank line {} does not canonicalize: {:?}: {error:#}",
                source.record + 1,
                source.genome
            );
        }
    }
    // The transfer-manifest limit (60), not this module's per-island 8.
    canonical_distinct_seeds(sources, features, super::super::POPULATION)
}

#[test]
fn initial_islands_are_deterministic_and_complete() {
    let seeds = checked_in_transfer_seeds(10);
    assert!(
        seeds.len() >= TOTAL_POPULATION,
        "checked-in bank holds only {} distinct seeds",
        seeds.len()
    );
    let first = initial_population_from_seeds(&seeds, "run-a", "dataset").unwrap();
    let second = initial_population_from_seeds(&seeds, "run-a", "dataset").unwrap();
    let third = initial_population_from_seeds(&seeds, "run-b", "dataset").unwrap();
    assert_eq!(first, second);
    assert_ne!(first, third);
    assert_eq!(first.len(), TOTAL_POPULATION);
    assert_eq!(
        first
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<BTreeSet<_>>()
            .len(),
        TOTAL_POPULATION
    );
}

#[test]
fn initial_islands_refuse_too_few_or_repeated_seeds() {
    let seeds = checked_in_transfer_seeds(10);
    let error = initial_population_from_seeds(&seeds[..TOTAL_POPULATION - 1], "run-a", "dataset")
        .unwrap_err();
    assert!(
        error.to_string().contains(&format!(
            "at least {TOTAL_POPULATION} frozen transfer seeds"
        )),
        "{error:#}"
    );
    let mut repeated = seeds.clone();
    let last = repeated.len() - 1;
    repeated[last] = SeedProvenance {
        slot: last,
        ..repeated[0].clone()
    };
    let error = initial_population_from_seeds(&repeated, "run-a", "dataset").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires distinct transfer seeds"),
        "{error:#}"
    );
}

#[test]
fn all_dataset_wave_targets_high_node_utilization() {
    let generation_zero = (0..TOTAL_POPULATION).fold([0usize; 3], |mut counts, slot| {
        counts[slot % 3] += 1;
        counts
    });
    assert_eq!(generation_zero, [11, 11, 10]);
    let descendants = (0..TOTAL_POPULATION)
        .filter(|slot| slot % POPULATION >= ELITES)
        .fold([0usize; 3], |mut counts, slot| {
            counts[(slot + 1) % 3] += 1;
            counts
        });
    assert_eq!(descendants, [8, 8, 8]);
    assert_eq!(
        generation_zero.map(|count| count * DATASET_LANES),
        [143, 143, 130]
    );
    assert_eq!(
        descendants.map(|count| count * DATASET_LANES),
        [104, 104, 104]
    );
}

#[test]
fn tabarena_offspring_stream_is_reproducible_and_run_salted() {
    let first = offspring_seed("run-a", "dataset", 7, 2, 5, 0);
    assert_eq!(first, offspring_seed("run-a", "dataset", 7, 2, 5, 0));
    assert_ne!(first, offspring_seed("run-b", "dataset", 7, 2, 5, 0));
}

#[test]
fn two_valid_candidates_per_island_are_enough_to_continue() {
    let outcomes = fast30_bank_sources()
        .unwrap()
        .into_iter()
        .take(TOTAL_POPULATION)
        .enumerate()
        .map(|(slot, source)| {
            let canonical = canonicalize_for_features(&source.genome, 10).unwrap();
            let valid = slot % POPULATION < ELITES;
            SlotOutcome {
                generation: 0,
                slot,
                member: Member {
                    canonical_genome: canonical.canonical,
                    semantic_sha256: canonical.semantic_sha256,
                    complexity: canonical.complexity,
                    rust_source_len: canonical.rust_source_len,
                    evolved_descendant: false,
                    ancestry: Ancestry {
                        operation: "tabarena_seed".to_string(),
                        parents: Vec::new(),
                    },
                },
                valid,
                metrics: valid.then_some(CandidateMetrics {
                    rmse: slot as f64 + 1.0,
                    mae: 0.0,
                    r2: 0.0,
                }),
                failure: (!valid).then_some(FailureState {
                    class: "timeout".to_string(),
                    message: "fixture".to_string(),
                    retryable: false,
                }),
                cache_source: None,
                receipt_file: format!("slot-{slot}.json"),
                receipt_sha256: "a".repeat(64),
                node: scq_batch::NODES[slot % scq_batch::NODES.len()].to_string(),
                scq_job_id: format!("job-{slot}"),
                attempt: 0,
            }
        })
        .collect::<Vec<_>>();
    let next = next_population("run", "dataset", 0, 10, &outcomes).unwrap();
    assert_eq!(next.len(), TOTAL_POPULATION);
}

#[test]
fn deficient_island_recovers_from_globally_valid_parents() {
    let outcomes = fast30_bank_sources()
        .unwrap()
        .into_iter()
        .take(TOTAL_POPULATION)
        .enumerate()
        .map(|(slot, source)| {
            let canonical = canonicalize_for_features(&source.genome, 10).unwrap();
            let island = slot / POPULATION;
            let local_slot = slot % POPULATION;
            let valid = if island == 0 {
                local_slot == 0
            } else {
                local_slot < 3
            };
            SlotOutcome {
                generation: 0,
                slot,
                member: Member {
                    canonical_genome: canonical.canonical,
                    semantic_sha256: canonical.semantic_sha256,
                    complexity: canonical.complexity,
                    rust_source_len: canonical.rust_source_len,
                    evolved_descendant: false,
                    ancestry: Ancestry {
                        operation: "tabarena_seed".to_string(),
                        parents: Vec::new(),
                    },
                },
                valid,
                metrics: valid.then_some(CandidateMetrics {
                    rmse: slot as f64 + 1.0,
                    mae: 0.0,
                    r2: 0.0,
                }),
                failure: (!valid).then_some(FailureState {
                    class: "timeout".to_string(),
                    message: "fixture".to_string(),
                    retryable: false,
                }),
                cache_source: None,
                receipt_file: format!("slot-{slot}.json"),
                receipt_sha256: "a".repeat(64),
                node: scq_batch::NODES[slot % scq_batch::NODES.len()].to_string(),
                scq_job_id: format!("job-{slot}"),
                attempt: 0,
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(outcomes.iter().filter(|outcome| outcome.valid).count(), 10);
    let next = next_population("run", "dataset", 0, 10, &outcomes).unwrap();
    assert_eq!(next.len(), TOTAL_POPULATION);
    assert_eq!(next[0].ancestry.operation, "elite");
    assert_eq!(next[1].ancestry.operation, "elite");
    assert!(outcomes.iter().any(|outcome| {
        outcome.valid && outcome.member.semantic_sha256 == next[1].semantic_sha256
    }));
}

#[test]
fn only_fresh_post_g0_novel_descendants_can_terminate() {
    let outcome = |generation: usize, operation: &str, parent: &str, cached: bool| SlotOutcome {
        generation,
        slot: 0,
        member: Member {
            canonical_genome: "(pipe X (mean))".to_string(),
            semantic_sha256: "a".repeat(64),
            complexity: 1,
            rust_source_len: 1,
            evolved_descendant: true,
            ancestry: Ancestry {
                operation: operation.to_string(),
                parents: vec![parent.to_string()],
            },
        },
        valid: true,
        metrics: Some(CandidateMetrics {
            rmse: 0.5,
            mae: 0.5,
            r2: 0.0,
        }),
        failure: None,
        cache_source: cached.then_some(CacheSource {
            generation: 0,
            slot: 0,
        }),
        receipt_file: "receipt".to_string(),
        receipt_sha256: "b".repeat(64),
        node: "xbabe1".to_string(),
        scq_job_id: "job".to_string(),
        attempt: 0,
    };
    let other = "c".repeat(64);
    let same = "a".repeat(64);
    assert!(discovery_winner(&[outcome(0, "mutation", &other, false)], 0, 1.0).is_none());
    assert!(discovery_winner(&[outcome(1, "elite", &other, false)], 1, 1.0).is_none());
    assert!(discovery_winner(&[outcome(1, "mutation", &same, false)], 1, 1.0).is_none());
    assert!(discovery_winner(&[outcome(1, "mutation", &other, true)], 1, 1.0).is_none());
    assert!(discovery_winner(&[outcome(1, "mutation", &other, false)], 1, 1.0).is_some());
}
