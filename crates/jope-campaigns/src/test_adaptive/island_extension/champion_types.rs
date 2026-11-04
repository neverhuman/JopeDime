use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Golden42BestSolution {
    pub(super) dataset_hash: String,
    pub(super) source_campaign: String,
    pub(super) source_artifact: String,
    pub(super) source_artifact_sha256: String,
    pub(super) benchmark_status: String,
    pub(super) leader_rmse: f64,
    pub(super) selected_rmse: f64,
    pub(super) margin_vs_leader: f64,
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    pub(super) generation: usize,
    pub(super) island: Option<usize>,
    pub(super) slot: usize,
    pub(super) binary_sha256: String,
    pub(super) prediction_sha256: String,
    pub(super) receipt_sha256: String,
    pub(super) replay_receipt_sha256: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Golden42BestSolutions {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) base_run_id: String,
    pub(super) island_extension_run_id: String,
    pub(super) ordered_dataset_ids: Vec<String>,
    pub(super) ordered_cohort_sha256: String,
    pub(super) dataset_count: usize,
    pub(super) prior_winner_count: usize,
    pub(super) island_selected_count: usize,
    pub(super) solutions: Vec<Golden42BestSolution>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ChampionDatasetBinding {
    pub(super) dataset_hash: String,
    pub(super) initial_result_file: String,
    pub(super) initial_result_sha256: String,
    pub(super) generation_29_checkpoint_file: String,
    pub(super) generation_29_checkpoint_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct ChampionManifest {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) base_run_id: String,
    pub(super) initial_island_run_id: String,
    pub(super) champion_run_id: String,
    pub(super) initial_manifest_sha256: String,
    pub(super) initial_summary_sha256: String,
    pub(super) initial_best_solutions_sha256: String,
    pub(super) ordered_dataset_ids: Vec<String>,
    pub(super) ordered_cohort_sha256: String,
    pub(super) release_binary_sha256: String,
    pub(super) scq_binary_sha256: String,
    pub(super) scq_staging_receipt_sha256: String,
    pub(super) scq_policy: scq_batch::ScqPolicy,
    pub(super) config: IslandConfig,
    pub(super) migration_policy: MigrationPolicy,
    pub(super) rng_policy: RngPolicy,
    pub(super) complexity_schedule: String,
    pub(super) stop_policy: String,
    pub(super) datasets: Vec<ChampionDatasetBinding>,
    pub(super) started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct ChampionDatasetResult {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) base_run_id: String,
    pub(super) champion_run_id: String,
    pub(super) dataset_hash: String,
    pub(super) outcome: String,
    pub(super) start_generation: usize,
    pub(super) stopping_generation: usize,
    pub(super) generations_evaluated: usize,
    pub(super) logical_evaluations: usize,
    pub(super) fresh_evaluations: usize,
    pub(super) leader_models: Vec<String>,
    pub(super) leader_rmse: f64,
    pub(super) champion: IslandGenomeRecord,
    pub(super) replay: Vec<ReplayRecord>,
    pub(super) release_binary_sha256: String,
    pub(super) started_at_unix_ms: u128,
    pub(super) finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct ChampionSummary {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) base_run_id: String,
    pub(super) initial_island_run_id: String,
    pub(super) champion_run_id: String,
    pub(super) release_binary_sha256: String,
    pub(super) ordered_dataset_ids: Vec<String>,
    pub(super) dataset_count: usize,
    pub(super) inherited_champion_count: usize,
    pub(super) continuation_champion_count: usize,
    pub(super) cumulative_surpassed_count: usize,
    pub(super) total_logical_evaluations: usize,
    pub(super) total_fresh_evaluations: usize,
    pub(super) best_solutions_file: String,
    pub(super) best_solutions_sha256: String,
    pub(super) results: Vec<ChampionDatasetResult>,
    pub(super) artifacts: Vec<ArtifactHash>,
}
