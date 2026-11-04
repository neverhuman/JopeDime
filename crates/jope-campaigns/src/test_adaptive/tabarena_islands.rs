use super::*;
use feat_math::advanced_v3::RawTable;
use jope_dime_regressor::paths;

jope_paths::once_str!(
    fn source_root_str => paths::legacy_tree("tabArenaLite").join("data/tabarena-lite-v0.1/datasets/regression")
);
jope_paths::once_str!(
    fn canonical_scores_str => paths::legacy_tree("tabArenaLite").join("artifacts/leaderboard-20260723/canonical_scores.csv")
);
jope_paths::once_str!(
    fn local_results_str => paths::legacy_tree("tabArenaLite").join("results")
);
const MANIFEST_SCHEMA: &str = "JopeTabArenaLiteIslandManifest.v1";
const CHECKPOINT_SCHEMA: &str = "JopeTabArenaLiteIslandCheckpoint.v1";
const RESULT_SCHEMA: &str = "JopeTabArenaLiteIslandResult.v1";
const SUMMARY_SCHEMA: &str = "JopeTabArenaLiteIslandSummary.v1";
const MANIFEST_FILE: &str = "tabarena-island-manifest.v1.json";
const STAGING_FILE: &str = "scq-staging-receipt.v1.json";
const SUMMARY_FILE: &str = "tabarena-island-summary.v1.json";
const ISLANDS: usize = 4;
const POPULATION: usize = 8;
const TOTAL_POPULATION: usize = ISLANDS * POPULATION;
const ELITES: usize = 2;
const PARENT_POOL: usize = 4;
const MIN_VALID_ISLAND: usize = ELITES;
const MIN_VALID_GLOBAL: usize = ISLANDS * ELITES;
const DATASET_LANES: usize = 13;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TabArenaSourceBinding {
    dataset: String,
    source_train_sha256: String,
    source_test_sha256: String,
    metadata_sha256: String,
    encoded_train_sha256: String,
    encoded_test_sha256: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    categorical_features: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct TabArenaManifest {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    source_root: String,
    canonical_scores_sha256: String,
    local_leader_artifacts: Vec<ArtifactHash>,
    release_binary_sha256: String,
    scq_binary_sha256: String,
    scq_staging_receipt_sha256: String,
    scq_policy: scq_batch::ScqPolicy,
    ordered_datasets: Vec<String>,
    ordered_cohort_sha256: String,
    datasets: Vec<TabArenaSourceBinding>,
    started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TabArenaCheckpoint {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    generation: usize,
    previous_checkpoint_sha256: Option<String>,
    logical_slots: usize,
    fresh_evaluations: usize,
    cumulative_fresh_evaluations: usize,
    valid_candidates: usize,
    island_valid_candidates: Vec<usize>,
    island_best_rmse: Vec<f64>,
    best_slot: usize,
    best_rmse: f64,
    leader_rmse: f64,
    leader_beaten: bool,
    migration_complete: bool,
    slots: Vec<SlotOutcome>,
    next_population: Vec<Member>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct TabArenaResult {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    stopping_generation: usize,
    generations_evaluated: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
    selected_rmse: f64,
    margin_vs_leader: f64,
    canonical_genome: String,
    semantic_sha256: String,
    selected_slot: usize,
    prediction_sha256: String,
    receipt_sha256: String,
    binary_sha256: String,
    logical_evaluations: usize,
    fresh_evaluations: usize,
    replay: Vec<ReplayRecord>,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TabArenaSummary {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset_count: usize,
    leaders_beaten: usize,
    total_generations: usize,
    total_logical_evaluations: usize,
    total_fresh_evaluations: usize,
    results: Vec<TabArenaResult>,
    artifacts: Vec<ArtifactHash>,
}

#[derive(Clone, Debug)]
pub(super) struct EncodedDataset {
    binding: TabArenaSourceBinding,
    spec: DatasetSpec,
}

mod artifacts;
mod data;
mod evolution;
mod execution;
#[cfg(test)]
mod tests;

pub(crate) use artifacts::{
    recover_tabarena_lite_result, run_tabarena_lite_islands, validate_tabarena_lite_islands,
};
pub(super) use data::{campaign_root, identity};
#[cfg(test)]
pub(super) use data::{collect_canonical_leaders, update_leader};
use data::{prepare_datasets, select_datasets};
pub(super) use evolution::{checkpoint_path, initial_population, replay_records};
#[cfg(test)]
pub(super) use evolution::{
    discovery_winner, initial_population_from_seeds, next_population, offspring_seed,
};
use evolution::{load_cache, load_checkpoints, make_checkpoint};
use execution::run_tabarena_dataset_waves;
