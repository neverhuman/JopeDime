use super::*;

#[test]
fn global_slot_mapping_is_exact_and_reversible() {
    for island in 0..ISLANDS {
        for local in 0..ISLAND_POPULATION {
            let global = global_slot(island, local);
            assert_eq!(island_slot(global).unwrap(), (island, local));
        }
    }
    assert!(island_slot(TOTAL_POPULATION).is_err());
}

#[test]
fn island_rng_namespaces_are_deterministic_and_distinct() {
    let one = island_offspring_seed(20, 0, 4, 0);
    assert_eq!(one, island_offspring_seed(20, 0, 4, 0));
    let values = (0..ISLANDS)
        .map(|island| island_offspring_seed(20, island, 4, 0))
        .collect::<BTreeSet<_>>();
    assert_eq!(values.len(), ISLANDS);
    assert_ne!(island_sequence(0), island_sequence(1));
}

#[test]
fn generation_20_initialization_is_deterministic_and_distributes_distinct_elites() {
    let combined = synthetic_combined();
    let first = generation_20_populations(&combined, 20).unwrap();
    let second = generation_20_populations(&combined, 20).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), ISLANDS);
    assert!(first
        .iter()
        .all(|population| population.len() == ISLAND_POPULATION));
    let elites = first
        .iter()
        .flat_map(|population| &population[..ISLAND_ELITES])
        .map(|member| member.member.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(elites.len(), ISLANDS * ISLAND_ELITES);
    for population in &first {
        assert!(population[..ISLAND_ELITES]
            .iter()
            .all(|member| member.parents[0].island.is_none()));
    }
    let offspring = first
        .iter()
        .map(|population| population[ISLAND_ELITES].member.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(offspring.len(), ISLANDS);
}

#[test]
fn later_populations_use_only_their_local_parent_pool() {
    let combined = synthetic_combined();
    let terminal = &combined.last().unwrap().0.slots;
    let ranked = terminal[..ISLAND_POPULATION]
        .iter()
        .enumerate()
        .map(|(local_slot, old)| IslandSlotOutcome {
            generation: START_GENERATION,
            island: 2,
            local_slot,
            global_slot: global_slot(2, local_slot),
            member: old.member.clone(),
            parents: Vec::new(),
            valid: true,
            metrics: old.metrics.clone(),
            failure: None,
            cache_source: None,
            receipt_file: String::new(),
            receipt_sha256: String::new(),
            node: EXECUTION_NODE.to_string(),
            scq_job_id: String::new(),
            attempt: 0,
        })
        .collect::<Vec<_>>();
    let next = next_island_population(&ranked, START_GENERATION + 1, 2, 20).unwrap();
    assert!(next.iter().all(|member| member
        .parents
        .iter()
        .all(|parent| { parent.generation == START_GENERATION && parent.island == Some(2) })));
}

#[test]
fn ring_migration_replaces_exact_bottom_two() {
    let ranked = (0..ISLANDS)
        .map(|island| {
            (0..ISLAND_POPULATION)
                .map(|slot| outcome(island, slot, island as f64 + slot as f64 / 100.0))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (migrated, record) = migrate_rankings("run", "data", 21, &ranked).unwrap();
    assert_eq!(record.transfers.len(), 8);
    for source in 0..ISLANDS {
        let destination = (source + 1) % ISLANDS;
        for rank in 0..2 {
            let transfer = &record.transfers[source * 2 + rank];
            assert_eq!(transfer.source_island, source);
            assert_eq!(transfer.destination_island, destination);
            assert_eq!(transfer.source_rank, rank);
            assert_eq!(
                transfer.destination_replaced_rank,
                ISLAND_POPULATION - 2 + rank
            );
            assert_eq!(
                transfer.migrant.member.semantic_sha256,
                ranked[source][rank].member.semantic_sha256
            );
            assert_eq!(
                transfer.replaced.member.semantic_sha256,
                ranked[destination][ISLAND_POPULATION - 2 + rank]
                    .member
                    .semantic_sha256
            );
            assert!(migrated[destination].iter().any(|outcome| {
                outcome.member.semantic_sha256 == ranked[source][rank].member.semantic_sha256
            }));
        }
    }
}

#[test]
fn checkpoint_loader_rejects_incomplete_islands_and_broken_chain() {
    let identity = test_identity();
    let mut checkpoint = checkpoint_envelope("prior-terminal");
    validate_checkpoint_envelope(&checkpoint, &identity, START_GENERATION, "prior-terminal")
        .unwrap();
    checkpoint.islands[2].slots.pop();
    assert!(validate_checkpoint_envelope(
        &checkpoint,
        &identity,
        START_GENERATION,
        "prior-terminal"
    )
    .is_err());
    let checkpoint = checkpoint_envelope("wrong-chain");
    assert!(validate_checkpoint_envelope(
        &checkpoint,
        &identity,
        START_GENERATION,
        "prior-terminal"
    )
    .is_err());
}

#[test]
fn migration_policy_is_exact() {
    assert_eq!(
        MigrationPolicy::fixed().after_generations,
        vec![21, 23, 25, 27]
    );
    assert_eq!(
        MigrationPolicy::fixed().destination_replaced_ranks,
        vec![28, 29]
    );
    assert!(MigrationPolicy::fixed().semantic_duplicates_allowed);
}

#[test]
fn duplicate_logical_slots_bind_to_one_cached_receipt() {
    let population = IslandPopulationMember {
        member: simple_member("duplicate"),
        parents: Vec::new(),
    };
    let receipt = CandidateReceipt {
        schema: RECEIPT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: None,
        purpose: "candidate".to_string(),
        run_id: "run".to_string(),
        generation: 20,
        slot: 0,
        attempt: 0,
        semantic_sha256: "duplicate".to_string(),
        canonical_genome: population.member.canonical_genome.clone(),
        binary_sha256: "binary".to_string(),
        dataset_hash: "dataset".to_string(),
        train_sha256: "train".to_string(),
        test_sha256: "test".to_string(),
        train_rows: 1,
        test_rows: 1,
        features: 20,
        execution_host: EXECUTION_HOST.to_string(),
        node: EXECUTION_NODE.to_string(),
        scq_job_id: "job".to_string(),
        runtime_seconds: 1.0,
        thread_counts: ThreadCounts::single(),
        predictions: vec![0.0],
        predictions_sha256: sha256_bytes(&prediction_bytes(&[0.0])),
        metrics: Some(CandidateMetrics {
            rmse: 0.0,
            mae: 0.0,
            r2: 1.0,
        }),
        failure: None,
    };
    let evaluation = CachedEvaluation {
        source: CacheSource {
            generation: 20,
            slot: 0,
        },
        receipt_file: "receipt.json".to_string(),
        receipt_sha256: "receipt-hash".to_string(),
        receipt,
    };
    let cached = island_outcome_from_evaluation(20, 1, 0, population, &evaluation, true);
    assert_eq!(cached.global_slot, 30);
    assert_eq!(cached.cache_source, Some(evaluation.source));
    assert_eq!(cached.receipt_file, evaluation.receipt_file);
}

#[test]
fn receipt_and_binary_tampering_are_rejected() {
    let identity = test_identity();
    let member = simple_member("candidate");
    let worker = worker_identity(&identity);
    let request = make_request(
        &worker,
        Path::new("registry"),
        "candidate",
        20,
        0,
        0,
        &member,
        EXECUTION_NODE,
    );
    let metrics = CandidateMetrics {
        rmse: 0.0,
        mae: 0.0,
        r2: 1.0,
    };
    let receipt = CandidateReceipt {
        schema: RECEIPT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: None,
        purpose: request.purpose.clone(),
        run_id: request.run_id.clone(),
        generation: request.generation,
        slot: request.slot,
        attempt: request.attempt,
        semantic_sha256: request.semantic_sha256.clone(),
        canonical_genome: request.canonical_genome.clone(),
        binary_sha256: request.binary_sha256.clone(),
        dataset_hash: request.dataset_hash.clone(),
        train_sha256: request.train_sha256.clone(),
        test_sha256: request.test_sha256.clone(),
        train_rows: request.train_rows,
        test_rows: request.test_rows,
        features: request.features,
        execution_host: request.execution_host.clone(),
        node: request.node.clone(),
        scq_job_id: request.scq_job_id.clone(),
        runtime_seconds: 1.0,
        thread_counts: request.thread_counts.clone(),
        predictions: vec![0.0],
        predictions_sha256: sha256_bytes(&prediction_bytes(&[0.0])),
        metrics: Some(metrics.clone()),
        failure: None,
    };
    let outcome = IslandSlotOutcome {
        generation: 20,
        island: 0,
        local_slot: 0,
        global_slot: 0,
        member,
        parents: Vec::new(),
        valid: true,
        metrics: Some(metrics),
        failure: None,
        cache_source: None,
        receipt_file: String::new(),
        receipt_sha256: String::new(),
        node: EXECUTION_NODE.to_string(),
        scq_job_id: request.scq_job_id.clone(),
        attempt: 0,
    };
    validate_island_receipt_binding(&identity, &outcome, &receipt, &request).unwrap();
    let mut changed_request = request.clone();
    changed_request.binary_sha256 = "altered".to_string();
    assert!(
        validate_island_receipt_binding(&identity, &outcome, &receipt, &changed_request).is_err()
    );
    let mut changed_receipt = receipt;
    changed_receipt.semantic_sha256 = "altered".to_string();
    assert!(
        validate_island_receipt_binding(&identity, &outcome, &changed_receipt, &request).is_err()
    );
}

#[test]
fn island_lineage_reconstructs_through_generation_19() {
    let combined = synthetic_combined();
    let source = &combined.last().unwrap().0.slots[0];
    let selected = IslandSlotOutcome {
        generation: 20,
        island: 0,
        local_slot: 0,
        global_slot: 0,
        member: {
            let mut member = source.member.clone();
            member.ancestry = Ancestry {
                operation: "island_elite".to_string(),
                parents: vec![source.member.semantic_sha256.clone()],
            };
            member
        },
        parents: vec![old_parent_ref(19, source)],
        valid: true,
        metrics: source.metrics.clone(),
        failure: None,
        cache_source: None,
        receipt_file: String::new(),
        receipt_sha256: String::new(),
        node: EXECUTION_NODE.to_string(),
        scq_job_id: String::new(),
        attempt: 0,
    };
    let checkpoint = IslandGenerationCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: "run".to_string(),
        dataset_hash: "dataset".to_string(),
        generation: 20,
        previous_checkpoint_sha256: "g19".to_string(),
        logical_slots: 1,
        fresh_evaluations: 1,
        cumulative_fresh_evaluations: 1,
        valid_candidates: 1,
        fresh_g0_rmse: 1.0,
        global_best: genome_location(&selected),
        global_best_rmse: selected.rmse(),
        qualifying_descendant_win: false,
        qualifying_genomes: Vec::new(),
        result_label: None,
        migration_file: None,
        migration_sha256: None,
        islands: vec![IslandGenerationState {
            island: 0,
            logical_slots: 1,
            fresh_evaluations: 1,
            valid_candidates: 1,
            best_local_slot: 0,
            best_global_slot: 0,
            best_semantic_sha256: selected.member.semantic_sha256.clone(),
            best_rmse: selected.rmse(),
            ranked_local_slots: vec![0],
            slots: vec![selected.clone()],
            next_population: Vec::new(),
        }],
    };
    let lineage =
        reconstruct_island_lineage(&combined, &[(checkpoint, "g20".to_string())], &selected)
            .unwrap();
    assert!(lineage.iter().any(|record| {
        record.generation == 19
            && record.island.is_none()
            && record.semantic_sha256 == source.member.semantic_sha256
    }));
    assert!(lineage
        .iter()
        .any(|record| { record.generation == 20 && record.island == Some(0) }));
}

#[test]
fn altered_population_migration_and_qualifying_lists_change_bindings() {
    let combined = synthetic_combined();
    let populations = generation_20_populations(&combined, 20).unwrap();
    let original_hash = sha256_bytes(&canonical_json(&populations[0]).unwrap());
    let mut altered = populations[0].clone();
    altered[0].parents[0].semantic_sha256 = "altered".to_string();
    assert_ne!(
        original_hash,
        sha256_bytes(&canonical_json(&altered).unwrap())
    );

    let ranked = (0..ISLANDS)
        .map(|island| {
            (0..ISLAND_POPULATION)
                .map(|slot| outcome(island, slot, slot as f64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (_, migration) = migrate_rankings("run", "dataset", 21, &ranked).unwrap();
    let original_hash = sha256_bytes(&canonical_json(&migration).unwrap());
    let mut altered_migration = migration;
    altered_migration.transfers[0].destination_island = 3;
    assert_ne!(
        original_hash,
        sha256_bytes(&canonical_json(&altered_migration).unwrap())
    );

    let expected = vec![genome_location(&ranked[0][0])];
    let mut altered_qualifying = expected.clone();
    altered_qualifying.push(genome_location(&ranked[0][1]));
    assert_ne!(expected, altered_qualifying);
}

#[test]
#[ignore = "real-corpus: private inputs require the explicit host-real-corpus lane"]
fn prior_no_win_cohort_reconstructs_in_order() {
    require_historical_run("test-adaptive-golden42-v1");
    require_historical_run(&extension_run_id("test-adaptive-golden42-v1"));
    assert_eq!(
        prior_no_win_ids("test-adaptive-golden42-v1").unwrap(),
        vec![
            "21fbfe8b53dbcd60",
            "3f781a51af45387e",
            "55f06d147bcc2a32",
            "5f91a00bd3b66742",
            "860e5558f732f638",
            "b1f909fcfceeef4b",
            "b5beff4a53fa12c0",
        ]
    );
}
