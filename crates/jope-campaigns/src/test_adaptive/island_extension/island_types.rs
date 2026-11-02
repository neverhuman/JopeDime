use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandConfig {
    pub(super) islands: usize,
    pub(super) population_per_island: usize,
    pub(super) total_population: usize,
    pub(super) elites_per_island: usize,
    pub(super) generation_20_parent_pool: usize,
    pub(super) later_parent_pool_per_island: usize,
    pub(super) tournament: usize,
    pub(super) crossover_probability: f64,
    pub(super) mutation_probability: f64,
    pub(super) start_generation: usize,
    pub(super) generations: usize,
    pub(super) end_generation_exclusive: usize,
    pub(super) minimum_valid_per_island: usize,
    pub(super) minimum_valid_global: usize,
    pub(super) maximum_concurrent_candidates: usize,
    pub(super) candidate_threads: usize,
    pub(super) replay_count: usize,
    pub(super) global_slot_mapping: String,
}

impl IslandConfig {
    pub(super) fn fixed() -> Self {
        Self {
            islands: ISLANDS,
            population_per_island: ISLAND_POPULATION,
            total_population: TOTAL_POPULATION,
            elites_per_island: ISLAND_ELITES,
            generation_20_parent_pool: GENERATION_20_PARENT_POOL,
            later_parent_pool_per_island: ISLAND_PARENT_POOL,
            tournament: TOURNAMENT,
            crossover_probability: 0.25,
            mutation_probability: 0.75,
            start_generation: START_GENERATION,
            generations: GENERATIONS,
            end_generation_exclusive: END_GENERATION,
            minimum_valid_per_island: ISLAND_MIN_VALID,
            minimum_valid_global: GLOBAL_MIN_VALID,
            maximum_concurrent_candidates: scq_batch::SCQ_CPUS,
            candidate_threads: 1,
            replay_count: 3,
            global_slot_mapping: "island * 30 + local_slot".to_string(),
        }
    }

    pub(super) fn champion() -> Self {
        Self {
            start_generation: CHAMPION_START_GENERATION,
            generations: 0,
            end_generation_exclusive: 0,
            global_slot_mapping: "island * 30 + local_slot; continue until strict leader crossing"
                .to_string(),
            ..Self::fixed()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct MigrationPolicy {
    pub(super) after_generations: Vec<usize>,
    pub(super) direction: String,
    pub(super) migrants_per_island: usize,
    pub(super) source_ranks: Vec<usize>,
    pub(super) destination_replaced_ranks: Vec<usize>,
    pub(super) semantic_duplicates_allowed: bool,
}

impl MigrationPolicy {
    pub(super) fn fixed() -> Self {
        Self {
            after_generations: MIGRATION_GENERATIONS.to_vec(),
            direction: "island i sends to (i + 1) mod 4".to_string(),
            migrants_per_island: 2,
            source_ranks: vec![0, 1],
            destination_replaced_ranks: vec![28, 29],
            semantic_duplicates_allowed: true,
        }
    }

    pub(super) fn champion() -> Self {
        Self {
            after_generations: Vec::new(),
            direction: "after every odd generation >=31, island i sends to (i + 1) mod 4"
                .to_string(),
            migrants_per_island: 2,
            source_ranks: vec![0, 1],
            destination_replaced_ranks: vec![28, 29],
            semantic_duplicates_allowed: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct RngPolicy {
    pub(super) seed: u64,
    pub(super) namespace: String,
    pub(super) generator: String,
    pub(super) retry_limit: usize,
}

impl RngPolicy {
    pub(super) fn fixed() -> Self {
        Self {
            seed: RNG_SEED,
            namespace: "sha256(seed:generation:island:local_slot:retry)".to_string(),
            generator: "Pcg32 with island-specific sequence".to_string(),
            retry_limit: 100,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandPopulationHash {
    pub(super) island: usize,
    pub(super) sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandDatasetBinding {
    pub(super) dataset_hash: String,
    pub(super) prior_campaign_file: String,
    pub(super) prior_campaign_sha256: String,
    pub(super) prior_final_result_file: String,
    pub(super) prior_final_result_sha256: String,
    pub(super) prior_terminal_checkpoint_file: String,
    pub(super) prior_terminal_checkpoint_sha256: String,
    pub(super) generation_20_island_population_hashes: Vec<IslandPopulationHash>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandManifest {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) base_run_id: String,
    pub(super) prior_extension_run_id: String,
    pub(super) island_extension_run_id: String,
    pub(super) base_panel_manifest_sha256: String,
    pub(super) base_panel_summary_sha256: String,
    pub(super) base_winner_map_sha256: String,
    pub(super) prior_extension_manifest_sha256: String,
    pub(super) prior_extension_summary_sha256: String,
    pub(super) ordered_dataset_ids: Vec<String>,
    pub(super) ordered_cohort_sha256: String,
    pub(super) release_binary_sha256: String,
    pub(super) scq_binary_sha256: String,
    pub(super) scq_staging_receipt_sha256: String,
    pub(super) scq_policy: scq_batch::ScqPolicy,
    pub(super) config: IslandConfig,
    pub(super) migration_policy: MigrationPolicy,
    pub(super) rng_policy: RngPolicy,
    pub(super) datasets: Vec<IslandDatasetBinding>,
    pub(super) started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandCampaignIdentity {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) dataset: DatasetSpec,
    pub(super) release_binary_sha256: String,
    pub(super) config: IslandConfig,
    pub(super) migration_policy: MigrationPolicy,
    pub(super) rng_policy: RngPolicy,
    pub(super) selection_objective: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandParentRef {
    pub(super) generation: usize,
    pub(super) island: Option<usize>,
    pub(super) local_slot: usize,
    pub(super) global_slot: usize,
    pub(super) semantic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandPopulationMember {
    pub(super) member: Member,
    pub(super) parents: Vec<IslandParentRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandSlotOutcome {
    pub(super) generation: usize,
    pub(super) island: usize,
    pub(super) local_slot: usize,
    pub(super) global_slot: usize,
    pub(super) member: Member,
    pub(super) parents: Vec<IslandParentRef>,
    pub(super) valid: bool,
    pub(super) metrics: Option<CandidateMetrics>,
    pub(super) failure: Option<FailureState>,
    pub(super) cache_source: Option<CacheSource>,
    pub(super) receipt_file: String,
    pub(super) receipt_sha256: String,
    pub(super) node: String,
    pub(super) scq_job_id: String,
    pub(super) attempt: usize,
}

impl IslandSlotOutcome {
    pub(super) fn rmse(&self) -> f64 {
        self.metrics
            .as_ref()
            .map_or(f64::INFINITY, |metrics| metrics.rmse)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandGenerationState {
    pub(super) island: usize,
    pub(super) logical_slots: usize,
    pub(super) fresh_evaluations: usize,
    pub(super) valid_candidates: usize,
    pub(super) best_local_slot: usize,
    pub(super) best_global_slot: usize,
    pub(super) best_semantic_sha256: String,
    pub(super) best_rmse: f64,
    pub(super) ranked_local_slots: Vec<usize>,
    pub(super) slots: Vec<IslandSlotOutcome>,
    pub(super) next_population: Vec<IslandPopulationMember>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandGenomeLocation {
    pub(super) generation: usize,
    pub(super) island: usize,
    pub(super) local_slot: usize,
    pub(super) global_slot: usize,
    pub(super) semantic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandGenerationCheckpoint {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) generation: usize,
    pub(super) previous_checkpoint_sha256: String,
    pub(super) logical_slots: usize,
    pub(super) fresh_evaluations: usize,
    pub(super) cumulative_fresh_evaluations: usize,
    pub(super) valid_candidates: usize,
    pub(super) fresh_g0_rmse: f64,
    pub(super) global_best: IslandGenomeLocation,
    pub(super) global_best_rmse: f64,
    pub(super) qualifying_descendant_win: bool,
    pub(super) qualifying_genomes: Vec<IslandGenomeLocation>,
    pub(super) result_label: Option<String>,
    pub(super) migration_file: Option<String>,
    pub(super) migration_sha256: Option<String>,
    pub(super) islands: Vec<IslandGenerationState>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct MigrationGenome {
    pub(super) generation: usize,
    pub(super) island: usize,
    pub(super) local_slot: usize,
    pub(super) global_slot: usize,
    pub(super) rank: usize,
    pub(super) member: Member,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct MigrationTransfer {
    pub(super) source_island: usize,
    pub(super) destination_island: usize,
    pub(super) source_rank: usize,
    pub(super) destination_replaced_rank: usize,
    pub(super) migrant: MigrationGenome,
    pub(super) replaced: MigrationGenome,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct MigrationRecord {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) generation: usize,
    pub(super) policy: MigrationPolicy,
    pub(super) transfers: Vec<MigrationTransfer>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandLineageRecord {
    pub(super) generation: usize,
    pub(super) island: Option<usize>,
    pub(super) local_slot: usize,
    pub(super) global_slot: usize,
    pub(super) semantic_sha256: String,
    pub(super) operation: String,
    pub(super) parents: Vec<IslandParentRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandGenomeRecord {
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    pub(super) generation: usize,
    pub(super) island: usize,
    pub(super) local_slot: usize,
    pub(super) global_slot: usize,
    pub(super) rmse: f64,
    pub(super) mae: f64,
    pub(super) r2: f64,
    pub(super) margin_vs_g0: f64,
    pub(super) margin_vs_leader: f64,
    pub(super) complexity: usize,
    pub(super) rust_source_len: usize,
    pub(super) runtime_seconds: f64,
    pub(super) lineage: Vec<IslandLineageRecord>,
    pub(super) prediction_sha256: String,
    pub(super) receipt_file: String,
    pub(super) receipt_sha256: String,
    pub(super) binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandDatasetResult {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) base_run_id: String,
    pub(super) island_extension_run_id: String,
    pub(super) dataset_hash: String,
    pub(super) outcome: String,
    pub(super) start_generation: usize,
    pub(super) stopping_generation: usize,
    pub(super) generations_evaluated: usize,
    pub(super) total_generations_evaluated: usize,
    pub(super) leader_models: Vec<String>,
    pub(super) leader_rmse: f64,
    pub(super) fresh_g0_rmse: f64,
    pub(super) first_winning_generation: Option<usize>,
    pub(super) first_winning_island: Option<usize>,
    pub(super) generation_29_selected_genome: IslandGenomeRecord,
    pub(super) qualifying_descendants: Vec<IslandGenomeRecord>,
    pub(super) logical_evaluations: usize,
    pub(super) fresh_evaluations: usize,
    pub(super) started_at_unix_ms: u128,
    pub(super) finished_at_unix_ms: u128,
    pub(super) replay: Vec<ReplayRecord>,
    pub(super) release_binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct IslandSummary {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) base_run_id: String,
    pub(super) prior_extension_run_id: String,
    pub(super) island_extension_run_id: String,
    pub(super) base_panel_manifest_sha256: String,
    pub(super) base_panel_summary_sha256: String,
    pub(super) base_winner_map_sha256: String,
    pub(super) prior_extension_manifest_sha256: String,
    pub(super) prior_extension_summary_sha256: String,
    pub(super) release_binary_sha256: String,
    pub(super) ordered_dataset_ids: Vec<String>,
    pub(super) ordered_cohort_sha256: String,
    pub(super) dataset_count: usize,
    pub(super) existing_surpassed_count: usize,
    pub(super) new_leaders: usize,
    pub(super) still_no_leader: usize,
    pub(super) cumulative_surpassed_count: usize,
    pub(super) total_qualifying_genomes: usize,
    pub(super) total_logical_evaluations: usize,
    pub(super) total_fresh_evaluations: usize,
    pub(super) best_solutions_file: String,
    pub(super) best_solutions_sha256: String,
    pub(super) results: Vec<IslandDatasetResult>,
    pub(super) artifacts: Vec<ArtifactHash>,
}
