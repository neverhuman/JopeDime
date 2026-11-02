use super::*;

/// Sealed historical runs are real inputs and are never synthesized here.
/// Their absence is reported as an unavailable input and remains a failure.
fn require_historical_run(run_id: &str) {
    let root = run_root(run_id);
    assert!(
        root.is_dir(),
        "unavailable: sealed historical run {} (set JOPE_STATE to a tree whose runs/ holds the preserved test-adaptive runs)",
        root.display()
    );
}

fn test_spec() -> DatasetSpec {
    DatasetSpec {
        dataset_hash: "dataset".to_string(),
        train_sha256: "train".to_string(),
        test_sha256: "test".to_string(),
        train_rows: 1,
        test_rows: 1,
        features: 20,
        leader_models: vec!["leader".to_string()],
        leader_rmse: 0.5,
    }
}

fn test_identity() -> IslandCampaignIdentity {
    IslandCampaignIdentity {
        schema: CAMPAIGN_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: "run".to_string(),
        dataset: test_spec(),
        release_binary_sha256: "binary".to_string(),
        config: IslandConfig::fixed(),
        migration_policy: MigrationPolicy::fixed(),
        rng_policy: RngPolicy::fixed(),
        selection_objective: "test".to_string(),
    }
}

fn simple_member(hash: &str) -> Member {
    Member {
        canonical_genome: format!("genome-{hash}"),
        semantic_sha256: hash.to_string(),
        complexity: 1,
        rust_source_len: 2,
        evolved_descendant: true,
        ancestry: Ancestry {
            operation: "mutation".to_string(),
            parents: Vec::new(),
        },
    }
}

fn outcome(island: usize, local_slot: usize, rmse: f64) -> IslandSlotOutcome {
    IslandSlotOutcome {
        generation: 21,
        island,
        local_slot,
        global_slot: global_slot(island, local_slot),
        member: simple_member(&format!("{island}-{local_slot}")),
        parents: Vec::new(),
        valid: true,
        metrics: Some(CandidateMetrics {
            rmse,
            mae: rmse,
            r2: 0.0,
        }),
        failure: None,
        cache_source: None,
        receipt_file: String::new(),
        receipt_sha256: String::new(),
        node: EXECUTION_NODE.to_string(),
        scq_job_id: String::new(),
        attempt: 0,
    }
}

fn old_checkpoint(generation: usize, slots: Vec<SlotOutcome>) -> GenerationCheckpoint {
    GenerationCheckpoint {
        schema: super::super::CHECKPOINT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: "prior".to_string(),
        dataset_hash: "dataset".to_string(),
        generation,
        previous_checkpoint_sha256: None,
        logical_slots: slots.len(),
        fresh_evaluations: slots.len(),
        cumulative_fresh_evaluations: slots.len(),
        valid_candidates: slots.len(),
        g0_best_rmse: 1.0,
        best_slot: 0,
        best_semantic_sha256: slots
            .first()
            .map_or_else(String::new, |slot| slot.member.semantic_sha256.clone()),
        best_rmse: slots.first().map_or(f64::INFINITY, SlotOutcome::rmse),
        qualifying_descendant_win: false,
        result_label: None,
        ranked_slots: slots.iter().map(|slot| slot.slot).collect(),
        slots,
        next_population: Vec::new(),
    }
}

fn synthetic_combined() -> Vec<(GenerationCheckpoint, String)> {
    let sources = include_str!("../../seeds/generation-zero.txt")
        .lines()
        .take(POPULATION)
        .enumerate()
        .map(|(slot, source)| {
            let canonical = canonicalize(source).unwrap();
            let metrics = CandidateMetrics {
                rmse: (slot + 1) as f64 / 100.0,
                mae: (slot + 1) as f64 / 200.0,
                r2: 0.0,
            };
            SlotOutcome {
                generation: START_GENERATION - 1,
                slot,
                member: Member {
                    canonical_genome: canonical.canonical,
                    semantic_sha256: canonical.semantic_sha256,
                    complexity: canonical.complexity,
                    rust_source_len: canonical.rust_source_len,
                    evolved_descendant: true,
                    ancestry: Ancestry {
                        operation: "seed".to_string(),
                        parents: Vec::new(),
                    },
                },
                valid: true,
                metrics: Some(metrics),
                failure: None,
                cache_source: None,
                receipt_file: String::new(),
                receipt_sha256: String::new(),
                node: EXECUTION_NODE.to_string(),
                scq_job_id: String::new(),
                attempt: 0,
            }
        })
        .collect::<Vec<_>>();
    (0..START_GENERATION)
        .map(|generation| {
            let slots = if generation + 1 == START_GENERATION {
                sources.clone()
            } else {
                Vec::new()
            };
            (old_checkpoint(generation, slots), format!("g{generation}"))
        })
        .collect()
}

fn complete_state(island: usize, generation: usize) -> IslandGenerationState {
    let slots = (0..ISLAND_POPULATION)
        .map(|local| {
            let mut value = outcome(island, local, local as f64);
            value.generation = generation;
            value
        })
        .collect::<Vec<_>>();
    IslandGenerationState {
        island,
        logical_slots: ISLAND_POPULATION,
        fresh_evaluations: ISLAND_POPULATION,
        valid_candidates: ISLAND_POPULATION,
        best_local_slot: 0,
        best_global_slot: global_slot(island, 0),
        best_semantic_sha256: slots[0].member.semantic_sha256.clone(),
        best_rmse: slots[0].rmse(),
        ranked_local_slots: (0..ISLAND_POPULATION).collect(),
        slots,
        next_population: (0..ISLAND_POPULATION)
            .map(|slot| IslandPopulationMember {
                member: simple_member(&format!("next-{island}-{slot}")),
                parents: Vec::new(),
            })
            .collect(),
    }
}

fn checkpoint_envelope(previous: &str) -> IslandGenerationCheckpoint {
    IslandGenerationCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: "run".to_string(),
        dataset_hash: "dataset".to_string(),
        generation: START_GENERATION,
        previous_checkpoint_sha256: previous.to_string(),
        logical_slots: TOTAL_POPULATION,
        fresh_evaluations: TOTAL_POPULATION,
        cumulative_fresh_evaluations: TOTAL_POPULATION,
        valid_candidates: TOTAL_POPULATION,
        fresh_g0_rmse: 1.0,
        global_best: IslandGenomeLocation {
            generation: START_GENERATION,
            island: 0,
            local_slot: 0,
            global_slot: 0,
            semantic_sha256: "0-0".to_string(),
        },
        global_best_rmse: 0.0,
        qualifying_descendant_win: false,
        qualifying_genomes: Vec::new(),
        result_label: None,
        migration_file: None,
        migration_sha256: None,
        islands: (0..ISLANDS)
            .map(|island| complete_state(island, START_GENERATION))
            .collect(),
    }
}

#[path = "tests/champion.rs"]
mod champion;
#[path = "tests/extension.rs"]
mod extension;
