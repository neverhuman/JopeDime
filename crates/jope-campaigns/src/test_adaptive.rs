//! Sequential, intentionally test-adaptive Golden42 GP campaign.
//!
//! This path intentionally selects on the test target. Artifacts therefore
//! carry `benchmark_valid: false` and never touch the public registry.

use crate::data::{read_regression_csv, RegressionData};
use anyhow::{bail, Context, Result};
use feat_math::full_rust::{canonicalize, crossover_pipelines, mutate_pipeline, CanonicalGenome};
use feat_math::{GenomeIr, Pcg32};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod aggressive;
mod binary42;
mod island_extension;
mod scq_batch;
mod tabarena_islands;
mod timeseries42;

pub(crate) use aggressive::{run_aggressive, run_golden46, validate_aggressive, validate_golden46};
pub(crate) use binary42::{
    batch_worker as binary42_batch_worker, run as run_binary42, run_all as run_binary42_all,
    validate as validate_binary42, validate_all as validate_binary42_all,
};
pub(crate) use island_extension::{
    run_champion_continuation, run_island_extension, validate_champion_continuation,
    validate_island_extension,
};
pub(crate) use scq_batch::{aggressive_batch_worker, island_batch_worker};
pub(crate) use tabarena_islands::{
    recover_tabarena_lite_result, run_tabarena_lite_islands, validate_tabarena_lite_islands,
};
pub(crate) use timeseries42::{
    batch_worker as timeseries42_batch_worker, run_all as run_timeseries42_all,
    run_gen30 as run_timeseries42_gen30, run_winall as run_timeseries42_winall,
    validate_all as validate_timeseries42_all, validate_gen30 as validate_timeseries42_gen30,
};

const LEGACY_DATASET: &str = "b78d323566930bd7";
const PANEL_PATH: &str = "/home/ubuntu/golden42/golden42_foundry/data/panel.json";
const LEADER_MANIFEST: &str = "manifests/panel-analysis.v1.json";
const POPULATION: usize = 60;
const ELITES: usize = 7;
const PARENT_POOL: usize = 30;
const TOURNAMENT: usize = 3;
const MAX_GENERATIONS: usize = 10;
const RNG_SEED: u64 = 42;
const MIN_VALID: usize = 30;
const EXECUTION_HOST: &str = "localhost";
const EXECUTION_NODE: &str = "xbabe2";
const PANEL_SCHEMA: &str = "JopeTestAdaptivePanelCampaign.v1";
const CAMPAIGN_SCHEMA: &str = "JopeTestAdaptiveDatasetCampaign.v2";
const REQUEST_SCHEMA: &str = "JopeTestAdaptiveCandidateRequest.v2";
const RECEIPT_SCHEMA: &str = "JopeTestAdaptiveCandidateReceipt.v2";
const CHECKPOINT_SCHEMA: &str = "JopeTestAdaptiveGenerationCheckpoint.v2";
const SEED_SCHEMA: &str = "JopeTestAdaptiveSeedManifest.v2";
const STAGING_SCHEMA: &str = "JopeTestAdaptiveStagingReceipt.v2";
const RESULT_SCHEMA: &str = "JopeTestAdaptiveDatasetResult.v1";
const CONTINUATION_SCHEMA: &str = "JopeTestAdaptivePanelContinuation.v1";
const CONTINUATION_AMENDMENT_SCHEMA: &str = "JopeTestAdaptivePanelContinuationAmendment.v1";
const SUMMARY_SCHEMA: &str = "JopeTestAdaptivePanelSummary.v2";
const WINNER_MAP_SCHEMA: &str = "JopeTestAdaptiveWinnerMap.v1";
const RESULT_PREFIX: &str = "JOPE_TEST_ADAPTIVE_RECEIPT ";
const CONTINUATION_FILE: &str = "panel-continuation.v1.json";
const CONTINUATION_AMENDMENT_FILE: &str = "panel-continuation-002.v1.json";
const SUMMARY_FILE: &str = "panel-summary.v1.json";
const WINNER_MAP_FILE: &str = "winner-map.v1.json";
const FAST30_FIX_REASON: &str = "After canonicalization and semantic deduplication, the dataset-specific fast30 champion was not guaranteed to remain in the 60-slot generation-zero population; the corrected binary deterministically replaces slot 59 when that champion is absent.";
const FAST30_ABSENCE_FIX_REASON: &str = "Dataset cf5e74b4e9594944 has a frozen failed fast30 report with no genome; the amended binary records the champion as absent and does not invent a replacement, while retaining all other frozen generation-zero sources.";
const EXTENSION_MANIFEST_SCHEMA: &str = "JopeTestAdaptiveExtensionManifest.v1";
const EXTENSION_RESULT_SCHEMA: &str = "JopeTestAdaptiveDatasetExtensionResult.v1";
const EXTENSION_SUMMARY_SCHEMA: &str = "JopeTestAdaptiveExtensionSummary.v1";
const EXTENSION_START_GENERATION: usize = 10;
const EXTENSION_GENERATIONS: usize = 10;
const EXTENSION_END_GENERATION: usize = EXTENSION_START_GENERATION + EXTENSION_GENERATIONS;
const EXTENSION_MANIFEST_FILE: &str = "extension-manifest.v1.json";
const EXTENSION_SUMMARY_FILE: &str = "extension-summary.v1.json";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PanelFile {
    schema: String,
    dataset_root: String,
    panel_dataset_sha256: String,
    dataset_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct LeaderAnalysis {
    datasets: Vec<LeaderDataset>,
}

#[derive(Clone, Debug, Deserialize)]
struct LeaderDataset {
    dataset_hash: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    model_order: Vec<String>,
    rmse: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetSpec {
    dataset_hash: String,
    train_sha256: String,
    test_sha256: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PanelConfig {
    population: usize,
    elites: usize,
    parent_pool: usize,
    tournament: usize,
    crossover_probability: f64,
    mutation_probability: f64,
    max_transforms: usize,
    generations: usize,
    rng_seed: u64,
    maximum_concurrent_candidates: usize,
    candidate_threads: usize,
    infrastructure_retries: usize,
    candidate_timeout_seconds: usize,
    replay_count: usize,
    replay_rmse_tolerance: f64,
    replay_prediction_tolerance: f64,
}

impl PanelConfig {
    fn fixed() -> Self {
        Self {
            population: POPULATION,
            elites: ELITES,
            parent_pool: PARENT_POOL,
            tournament: TOURNAMENT,
            crossover_probability: 0.25,
            mutation_probability: 0.75,
            max_transforms: feat_math::full_rust::MAX_TRANSFORMS,
            generations: MAX_GENERATIONS,
            rng_seed: RNG_SEED,
            maximum_concurrent_candidates: POPULATION,
            candidate_threads: 1,
            infrastructure_retries: 1,
            candidate_timeout_seconds: 900,
            replay_count: 3,
            replay_rmse_tolerance: 1e-12,
            replay_prediction_tolerance: 1e-10,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PanelManifest {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    ordered_dataset_ids: Vec<String>,
    ordered_cohort_sha256: String,
    panel_file: String,
    panel_file_sha256: String,
    leader_manifest: String,
    leader_manifest_sha256: String,
    release_binary_sha256: String,
    registry_root: String,
    panel_declared_dataset_root: String,
    execution_host: String,
    execution_node: String,
    config: PanelConfig,
    datasets: Vec<DatasetSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ContinuationRecord {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    original_panel_manifest_sha256: String,
    original_binary_sha256: String,
    corrected_binary_sha256: String,
    completed_dataset_count: usize,
    first_remaining_dataset_index: usize,
    first_remaining_dataset_ordinal: usize,
    completed_dataset_ids: Vec<String>,
    remaining_dataset_ids: Vec<String>,
    completed_artifact_count: usize,
    completed_artifacts_sha256: String,
    seed_fix_reason: String,
    continuation_started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ContinuationAmendment {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    previous_continuation_file: String,
    previous_continuation_sha256: String,
    previous_binary_sha256: String,
    amended_binary_sha256: String,
    completed_dataset_count: usize,
    first_remaining_dataset_index: usize,
    first_remaining_dataset_ordinal: usize,
    completed_dataset_ids: Vec<String>,
    remaining_dataset_ids: Vec<String>,
    completed_artifact_count: usize,
    completed_artifacts_sha256: String,
    correction_reason: String,
    continuation_started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct BinarySegment {
    binary_sha256: String,
    first_dataset_index: usize,
    last_dataset_index: usize,
    dataset_count: usize,
    dataset_ids: Vec<String>,
}

#[derive(Clone, Debug)]
struct PanelExecution {
    dataset_binary_sha256: Vec<String>,
    binary_segments: Vec<BinarySegment>,
    continuation: Option<ContinuationRecord>,
    amendment: Option<ContinuationAmendment>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CampaignIdentity {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset_hash: String,
    train_sha256: String,
    test_sha256: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
    population: usize,
    elites: usize,
    parent_pool: usize,
    tournament: usize,
    crossover_probability: f64,
    mutation_probability: f64,
    max_transforms: usize,
    generations: usize,
    rng_seed: u64,
    binary_sha256: String,
    execution_host: String,
    execution_node: String,
    selection_objective: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SeedProvenance {
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    rust_source_len: usize,
    source_kind: String,
    source_file: String,
    source_file_sha256: String,
    source_record: usize,
    source_rank: usize,
    source_score: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SeedManifest {
    schema: String,
    benchmark_valid: bool,
    dataset_hash: String,
    population: usize,
    known_fast30_semantic_sha256: Option<String>,
    seeds: Vec<SeedProvenance>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ConfirmedGoldenTransferResult {
    schema: String,
    benchmark_valid: bool,
    test_adaptive: bool,
    run_id: String,
    dataset_hash: String,
    outcome: String,
    stopping_generation: usize,
    leader_rmse: f64,
    best_rmse: f64,
    best_slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    terminal_checkpoint_sha256: String,
    confirmation_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ConfirmedTabArenaTransferResult {
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Ancestry {
    operation: String,
    parents: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Member {
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    rust_source_len: usize,
    evolved_descendant: bool,
    ancestry: Ancestry,
}

impl Member {
    fn from_seed(seed: &SeedProvenance) -> Self {
        Self {
            canonical_genome: seed.canonical_genome.clone(),
            semantic_sha256: seed.semantic_sha256.clone(),
            complexity: seed.complexity,
            rust_source_len: seed.rust_source_len,
            evolved_descendant: false,
            ancestry: Ancestry {
                operation: "seed".to_string(),
                parents: Vec::new(),
            },
        }
    }

    fn evolved(value: feat_math::full_rust::EvolvedGenome, ancestry: Ancestry) -> Self {
        Self {
            canonical_genome: value.canonical,
            semantic_sha256: value.semantic_sha256,
            complexity: value.complexity,
            rust_source_len: value.rust_source_len,
            evolved_descendant: true,
            ancestry,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CandidateRequest {
    schema: String,
    benchmark_valid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    test_adaptive: Option<bool>,
    purpose: String,
    run_id: String,
    generation: usize,
    slot: usize,
    attempt: usize,
    semantic_sha256: String,
    canonical_genome: String,
    binary_sha256: String,
    dataset_hash: String,
    train_sha256: String,
    test_sha256: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    dataset_root: String,
    execution_host: String,
    node: String,
    scq_job_id: String,
    thread_counts: ThreadCounts,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CandidateMetrics {
    rmse: f64,
    mae: f64,
    r2: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct FailureState {
    class: String,
    message: String,
    retryable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ThreadCounts {
    rayon: usize,
    rust: usize,
    omp: usize,
    openblas: usize,
    mkl: usize,
    veclib: usize,
    numexpr: usize,
}

impl ThreadCounts {
    fn single() -> Self {
        Self {
            rayon: 1,
            rust: 1,
            omp: 1,
            openblas: 1,
            mkl: 1,
            veclib: 1,
            numexpr: 1,
        }
    }

    fn all_single(&self) -> bool {
        [
            self.rayon,
            self.rust,
            self.omp,
            self.openblas,
            self.mkl,
            self.veclib,
            self.numexpr,
        ]
        .iter()
        .all(|value| *value == 1)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CandidateReceipt {
    schema: String,
    benchmark_valid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    test_adaptive: Option<bool>,
    purpose: String,
    run_id: String,
    generation: usize,
    slot: usize,
    attempt: usize,
    semantic_sha256: String,
    canonical_genome: String,
    binary_sha256: String,
    dataset_hash: String,
    train_sha256: String,
    test_sha256: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    execution_host: String,
    node: String,
    scq_job_id: String,
    runtime_seconds: f64,
    thread_counts: ThreadCounts,
    predictions: Vec<f64>,
    predictions_sha256: String,
    metrics: Option<CandidateMetrics>,
    failure: Option<FailureState>,
}

impl CandidateReceipt {
    fn valid(&self, test_rows: usize) -> bool {
        self.failure.is_none()
            && self.metrics.as_ref().is_some_and(|metrics| {
                metrics.rmse.is_finite() && metrics.mae.is_finite() && metrics.r2.is_finite()
            })
            && self.predictions.len() == test_rows
            && self.predictions.iter().all(|value| value.is_finite())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CacheSource {
    generation: usize,
    slot: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SlotOutcome {
    generation: usize,
    slot: usize,
    member: Member,
    valid: bool,
    metrics: Option<CandidateMetrics>,
    failure: Option<FailureState>,
    cache_source: Option<CacheSource>,
    receipt_file: String,
    receipt_sha256: String,
    node: String,
    scq_job_id: String,
    attempt: usize,
}

impl SlotOutcome {
    fn rmse(&self) -> f64 {
        self.metrics
            .as_ref()
            .map_or(f64::INFINITY, |metrics| metrics.rmse)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GenerationCheckpoint {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    previous_checkpoint_sha256: Option<String>,
    logical_slots: usize,
    fresh_evaluations: usize,
    cumulative_fresh_evaluations: usize,
    valid_candidates: usize,
    g0_best_rmse: f64,
    best_slot: usize,
    best_semantic_sha256: String,
    best_rmse: f64,
    qualifying_descendant_win: bool,
    result_label: Option<String>,
    ranked_slots: Vec<usize>,
    slots: Vec<SlotOutcome>,
    next_population: Vec<Member>,
}

#[derive(Clone, Debug)]
struct CachedEvaluation {
    source: CacheSource,
    receipt_file: String,
    receipt_sha256: String,
    receipt: CandidateReceipt,
}

#[derive(Clone, Debug)]
struct CandidateSource {
    genome: String,
    kind: String,
    path: PathBuf,
    file_sha256: String,
    record: usize,
    rank: usize,
    score: Option<f64>,
    order_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct StagingReceipt {
    schema: String,
    benchmark_valid: bool,
    binary_sha256: String,
    campaign_sha256: String,
    bundle_root: String,
    nodes: BTreeMap<String, NodeStaging>,
    doctor_output_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct NodeStaging {
    binary_sha256: String,
    train_sha256: String,
    test_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ReplayRecord {
    receipt_file: String,
    receipt_sha256: String,
    rmse: f64,
    maximum_prediction_difference: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct LineageRecord {
    generation: usize,
    slot: usize,
    semantic_sha256: String,
    operation: String,
    parents: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetResult {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset_hash: String,
    outcome: String,
    stopping_generation: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
    g0_rmse: f64,
    final_rmse: f64,
    final_mae: f64,
    final_r2: f64,
    margin_vs_g0: f64,
    margin_vs_leader: f64,
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    rust_source_len: usize,
    generation: usize,
    slot: usize,
    prediction_file: String,
    prediction_sha256: String,
    runtime_seconds: f64,
    logical_evaluations: usize,
    fresh_evaluations: usize,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
    replay: Vec<ReplayRecord>,
    lineage: Vec<LineageRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ArtifactHash {
    file: String,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GenomeRecord {
    canonical_genome: String,
    semantic_sha256: String,
    generation: usize,
    slot: usize,
    rmse: f64,
    mae: f64,
    r2: f64,
    margin_vs_g0: f64,
    margin_vs_leader: f64,
    complexity: usize,
    rust_source_len: usize,
    runtime_seconds: f64,
    lineage: Vec<LineageRecord>,
    prediction_sha256: String,
    receipt_file: String,
    receipt_sha256: String,
    binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetWinnerMap {
    dataset_hash: String,
    final_selected_genome: GenomeRecord,
    qualifying_descendants: Vec<GenomeRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WinnerMap {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    ordered_cohort_sha256: String,
    binary_segments: Vec<BinarySegment>,
    total_qualifying_genomes: usize,
    datasets: Vec<DatasetWinnerMap>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PanelDatasetSummary {
    dataset_hash: String,
    leader_models: Vec<String>,
    leader_rmse: f64,
    fresh_g0_rmse: f64,
    outcome: String,
    stopping_generation: usize,
    total_generations_evaluated: usize,
    final_selected_semantic_sha256: String,
    qualifying_genome_count: usize,
    logical_evaluations: usize,
    fresh_evaluations: usize,
    binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct PanelSummary {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    ordered_cohort_sha256: String,
    leader_manifest_sha256: String,
    dataset_count: usize,
    early_wins: usize,
    no_wins: usize,
    total_qualifying_genomes: usize,
    total_logical_evaluations: usize,
    total_fresh_evaluations: usize,
    binary_segments: Vec<BinarySegment>,
    datasets: Vec<PanelDatasetSummary>,
    winner_map_file: String,
    winner_map_sha256: String,
    artifacts: Vec<ArtifactHash>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExtensionDatasetBinding {
    dataset_hash: String,
    base_binary_sha256: String,
    base_campaign_sha256: String,
    base_final_result_sha256: String,
    base_terminal_checkpoint_sha256: String,
    generation_10_population_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExtensionManifest {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    extension_run_id: String,
    base_panel_manifest_sha256: String,
    base_panel_summary_sha256: String,
    base_winner_map_sha256: String,
    extension_binary_sha256: String,
    start_generation: usize,
    generations: usize,
    end_generation_exclusive: usize,
    ordered_dataset_ids: Vec<String>,
    ordered_cohort_sha256: String,
    datasets: Vec<ExtensionDatasetBinding>,
    extension_started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExtensionDatasetResult {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    extension_run_id: String,
    dataset_hash: String,
    outcome: String,
    start_generation: usize,
    stopping_generation: usize,
    extension_generations_evaluated: usize,
    total_generations_evaluated: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
    fresh_g0_rmse: f64,
    first_qualifying_generation: Option<usize>,
    final_selected_genome: GenomeRecord,
    qualifying_descendants: Vec<GenomeRecord>,
    logical_evaluations: usize,
    fresh_evaluations: usize,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
    replay: Vec<ReplayRecord>,
    extension_binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExtensionSummary {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    extension_run_id: String,
    base_panel_summary_sha256: String,
    base_winner_map_sha256: String,
    extension_binary_sha256: String,
    dataset_count: usize,
    new_leaders: usize,
    still_no_leader: usize,
    total_qualifying_genomes: usize,
    total_logical_evaluations: usize,
    total_fresh_evaluations: usize,
    results: Vec<ExtensionDatasetResult>,
    artifacts: Vec<ArtifactHash>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn run_root(run_id: &str) -> PathBuf {
    workspace_root().join("runs").join(run_id)
}

fn extension_run_id(base_run_id: &str) -> String {
    format!("{base_run_id}-extension-g10-g19")
}

fn extension_root(base_run_id: &str) -> PathBuf {
    run_root(&extension_run_id(base_run_id))
}

fn validate_run_id(run_id: &str) -> Result<()> {
    if run_id.is_empty()
        || matches!(run_id, "." | "..")
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("run-id must contain only ASCII letters, digits, '.', '_' or '-'");
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&fs::read(path)?))
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn prediction_bytes(predictions: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for prediction in predictions {
        bytes.extend_from_slice(format!("{prediction:.17}\n").as_bytes());
    }
    bytes
}

fn ordered_cohort_sha256(dataset_ids: &[String]) -> String {
    let mut bytes = Vec::new();
    for dataset_id in dataset_ids {
        bytes.extend_from_slice(dataset_id.as_bytes());
        bytes.push(b'\n');
    }
    sha256_bytes(&bytes)
}

fn select_leader_with_model_count(
    dataset: &LeaderDataset,
    expected_model_count: usize,
) -> Result<(Vec<String>, f64)> {
    if dataset.model_order.len() != expected_model_count {
        bail!(
            "dataset {} records {} comparator models, expected {}",
            dataset.dataset_hash,
            dataset.model_order.len(),
            expected_model_count
        );
    }
    let mut best = f64::INFINITY;
    for model in &dataset.model_order {
        let value = *dataset.rmse.get(model).with_context(|| {
            format!(
                "dataset {} is missing RMSE for {model}",
                dataset.dataset_hash
            )
        })?;
        if value.is_finite() {
            best = best.min(value);
        }
    }
    if !best.is_finite() {
        bail!(
            "dataset {} has no finite comparator RMSE",
            dataset.dataset_hash
        );
    }
    let models = dataset
        .model_order
        .iter()
        .filter(|model| dataset.rmse.get(*model).copied() == Some(best))
        .cloned()
        .collect::<Vec<_>>();
    Ok((models, best))
}

fn select_leader(dataset: &LeaderDataset) -> Result<(Vec<String>, f64)> {
    select_leader_with_model_count(dataset, 6)
}

fn load_panel_specs(registry: &Path) -> Result<(PanelFile, Vec<DatasetSpec>, String, String)> {
    let panel_path = Path::new(PANEL_PATH);
    let panel_bytes = fs::read(panel_path)?;
    let panel: PanelFile = serde_json::from_slice(&panel_bytes)?;
    if panel.schema != "golden42.panel/v1" || panel.dataset_ids.len() != 42 {
        bail!("Golden42 panel schema or cohort size mismatch");
    }
    let cohort_sha256 = ordered_cohort_sha256(&panel.dataset_ids);
    if cohort_sha256 != panel.panel_dataset_sha256 {
        bail!("Golden42 ordered cohort hash mismatch");
    }
    let unique = panel.dataset_ids.iter().collect::<BTreeSet<_>>();
    if unique.len() != panel.dataset_ids.len() {
        bail!("Golden42 panel contains duplicate dataset ids");
    }

    let leader_path = workspace_root().join(LEADER_MANIFEST);
    let leader_bytes = fs::read(&leader_path)?;
    let analysis: LeaderAnalysis = serde_json::from_slice(&leader_bytes)?;
    let by_hash = analysis
        .datasets
        .iter()
        .map(|dataset| (dataset.dataset_hash.as_str(), dataset))
        .collect::<HashMap<_, _>>();
    if by_hash.len() != analysis.datasets.len() {
        bail!("leader manifest contains duplicate dataset records");
    }

    let mut specs = Vec::with_capacity(panel.dataset_ids.len());
    for dataset_hash in &panel.dataset_ids {
        let leader = by_hash
            .get(dataset_hash.as_str())
            .with_context(|| format!("leader manifest is missing dataset {dataset_hash}"))?;
        let (leader_models, leader_rmse) = select_leader(leader)?;
        let dataset_root = registry.join(dataset_hash);
        let train_path = dataset_root.join("train.csv");
        let test_path = dataset_root.join("test.csv");
        let train_sha256 = sha256_file(&train_path)?;
        let test_sha256 = sha256_file(&test_path)?;
        let train = read_regression_csv(&train_path)?;
        let test = read_regression_csv(&test_path)?;
        if train.x.len() != leader.train_rows
            || test.x.len() != leader.test_rows
            || train.features != leader.features
            || test.features != leader.features
        {
            bail!("dataset {dataset_hash} shape differs from the leader manifest");
        }
        specs.push(DatasetSpec {
            dataset_hash: dataset_hash.clone(),
            train_sha256,
            test_sha256,
            train_rows: leader.train_rows,
            test_rows: leader.test_rows,
            features: leader.features,
            leader_models,
            leader_rmse,
        });
    }
    Ok((
        panel,
        specs,
        sha256_bytes(&panel_bytes),
        sha256_bytes(&leader_bytes),
    ))
}

fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("immutable create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn node_for(_generation: usize, _slot: usize) -> &'static str {
    EXECUTION_NODE
}

fn metrics(truth: &[f64], predictions: &[f64]) -> Result<CandidateMetrics> {
    if truth.is_empty() || truth.len() != predictions.len() {
        bail!("metric vectors are empty or have different lengths");
    }
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let mut squared = 0.0;
    let mut absolute = 0.0;
    let mut total = 0.0;
    for (actual, predicted) in truth.iter().zip(predictions) {
        if !actual.is_finite() || !predicted.is_finite() {
            bail!("metric input is nonfinite");
        }
        let error = actual - predicted;
        squared += error * error;
        absolute += error.abs();
        total += (actual - mean).powi(2);
    }
    let rmse = (squared / n).sqrt();
    let mae = absolute / n;
    let r2 = if total > 0.0 {
        1.0 - squared / total
    } else if squared == 0.0 {
        1.0
    } else {
        0.0
    };
    if [rmse, mae, r2].iter().any(|value| !value.is_finite()) {
        bail!("computed metric is nonfinite");
    }
    Ok(CandidateMetrics { rmse, mae, r2 })
}

fn current_binary_sha256() -> Result<String> {
    sha256_file(&std::env::current_exe()?)
}

fn identity(run_id: &str, binary_sha256: String, spec: &DatasetSpec) -> CampaignIdentity {
    CampaignIdentity {
        schema: CAMPAIGN_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: run_id.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        train_sha256: spec.train_sha256.clone(),
        test_sha256: spec.test_sha256.clone(),
        train_rows: spec.train_rows,
        test_rows: spec.test_rows,
        features: spec.features,
        leader_models: spec.leader_models.clone(),
        leader_rmse: spec.leader_rmse,
        population: POPULATION,
        elites: ELITES,
        parent_pool: PARENT_POOL,
        tournament: TOURNAMENT,
        crossover_probability: 0.25,
        mutation_probability: 0.75,
        max_transforms: feat_math::full_rust::MAX_TRANSFORMS,
        generations: MAX_GENERATIONS,
        rng_seed: RNG_SEED,
        binary_sha256,
        execution_host: EXECUTION_HOST.to_string(),
        execution_node: EXECUTION_NODE.to_string(),
        selection_objective:
            "fit complete train; rank directly by test RMSE (intentionally test-adaptive)"
                .to_string(),
    }
}

fn report_sources(registry: &Path, dataset_hash: &str) -> Result<Vec<CandidateSource>> {
    let reports = registry.join(dataset_hash).join("reports");
    let mut paths = fs::read_dir(&reports)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|path| path.extension().and_then(|value| value.to_str()) == Some("json"));
    paths.sort();
    let mut sources = Vec::new();
    for path in paths {
        let bytes = fs::read(&path)?;
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let file_sha256 = sha256_bytes(&bytes);
        let report_rmse = value
            .pointer("/metrics/rmse")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .unwrap_or(f64::INFINITY);
        let mut record = 0;
        if let Some(genome) = value.get("genome").and_then(Value::as_str) {
            sources.push(CandidateSource {
                genome: genome.to_string(),
                kind: "frozen_target_report_winner".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: 0,
                score: report_rmse.is_finite().then_some(report_rmse),
                order_score: report_rmse,
            });
            record += 1;
        }
        if let Some(genome) = value
            .pointer("/algorithm/canonical_genome")
            .and_then(Value::as_str)
        {
            sources.push(CandidateSource {
                genome: genome.to_string(),
                kind: "frozen_target_report_winner".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: 0,
                score: report_rmse.is_finite().then_some(report_rmse),
                order_score: report_rmse,
            });
            record += 1;
        }
        if let Some(finalists) = value.get("finalists").and_then(Value::as_array) {
            for (index, finalist) in finalists.iter().enumerate() {
                let genome = finalist
                    .get("genome")
                    .or_else(|| finalist.get("gene"))
                    .and_then(Value::as_str);
                let Some(genome) = genome else {
                    continue;
                };
                let rank = finalist
                    .get("rank")
                    .and_then(Value::as_u64)
                    .map_or(index + 1, |rank| rank as usize);
                sources.push(CandidateSource {
                    genome: genome.to_string(),
                    kind: "frozen_target_report_finalist".to_string(),
                    path: path.clone(),
                    file_sha256: file_sha256.clone(),
                    record,
                    rank,
                    score: report_rmse.is_finite().then_some(report_rmse),
                    order_score: report_rmse,
                });
                record += 1;
            }
        }
    }
    sources.sort_by(|left, right| {
        left.order_score
            .total_cmp(&right.order_score)
            .then_with(|| left.rank.cmp(&right.rank))
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.record.cmp(&right.record))
    });
    Ok(sources)
}

fn jope_generation_sources(dataset_hash: &str) -> Result<Vec<CandidateSource>> {
    let mut ledgers = Vec::new();
    for entry in fs::read_dir(workspace_root().join("runs"))? {
        let entry = entry?;
        let path = entry.path().join(dataset_hash).join("slots.v1.jsonl");
        if path.is_file() {
            ledgers.push(path);
        }
    }
    ledgers.sort();
    let mut sources = Vec::new();
    for path in ledgers {
        let bytes = fs::read(&path)?;
        let file_sha256 = sha256_bytes(&bytes);
        let text = std::str::from_utf8(&bytes)?;
        for (record, line) in text.lines().enumerate() {
            let value: Value = match serde_json::from_str(line) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if value.get("valid").and_then(Value::as_bool) != Some(true) {
                continue;
            }
            let Some(genome) = value.get("canonical_genome").and_then(Value::as_str) else {
                continue;
            };
            let Some(score) = value.get("robust_fitness").and_then(Value::as_f64) else {
                continue;
            };
            if !score.is_finite() {
                continue;
            }
            sources.push(CandidateSource {
                genome: genome.to_string(),
                kind: "jope_generation_artifact".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: record,
                score: Some(score),
                order_score: -score,
            });
        }
    }
    sources.sort_by(|left, right| {
        left.order_score
            .total_cmp(&right.order_score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.record.cmp(&right.record))
    });
    Ok(sources)
}

fn prior_test_adaptive_checkpoint_sources(dataset_hash: &str) -> Result<Vec<CandidateSource>> {
    let mut checkpoint_paths = Vec::new();
    for entry in fs::read_dir(workspace_root().join("runs"))? {
        let run_root = entry?.path();
        let Some(run_name) = run_root.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !run_name.starts_with("golden42-all-regression-")
            && !run_name.starts_with("tabarena-lite-all-regression-")
        {
            continue;
        }
        let checkpoint_root = run_root
            .join("datasets")
            .join(dataset_hash)
            .join("checkpoints");
        if !checkpoint_root.is_dir() {
            continue;
        }
        checkpoint_paths.extend(
            fs::read_dir(checkpoint_root)?
                .map(|entry| Ok(entry?.path()))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    checkpoint_paths.sort();
    let mut sources = Vec::new();
    for path in checkpoint_paths {
        let bytes = fs::read(&path)?;
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let schema = value.get("schema").and_then(Value::as_str);
        let recorded_dataset = value
            .get("dataset_hash")
            .or_else(|| value.get("dataset"))
            .and_then(Value::as_str);
        if value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
            || !matches!(
                schema,
                Some("JopeGolden42AggressiveCheckpoint.v1" | "JopeTabArenaLiteIslandCheckpoint.v1")
            )
            || recorded_dataset != Some(dataset_hash)
        {
            continue;
        }
        let file_sha256 = sha256_bytes(&bytes);
        let generation = value.get("generation").and_then(Value::as_u64).unwrap_or(0) as usize;
        let Some(slots) = value.get("slots").and_then(Value::as_array) else {
            continue;
        };
        for (slot, outcome) in slots.iter().enumerate() {
            if outcome.get("valid").and_then(Value::as_bool) != Some(true) {
                continue;
            }
            let Some(genome) = outcome
                .pointer("/member/canonical_genome")
                .and_then(Value::as_str)
            else {
                continue;
            };
            let Some(rmse) = outcome.pointer("/metrics/rmse").and_then(Value::as_f64) else {
                continue;
            };
            if !rmse.is_finite() {
                continue;
            }
            sources.push(CandidateSource {
                genome: genome.to_string(),
                kind: "prior_test_adaptive_checkpoint_elite".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record: generation.saturating_mul(1_000).saturating_add(slot),
                rank: 0,
                score: Some(rmse),
                order_score: rmse,
            });
        }
    }
    sources.sort_by(|left, right| {
        left.order_score
            .total_cmp(&right.order_score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.record.cmp(&right.record))
    });
    let mut seen = HashSet::new();
    sources.retain(|source| seen.insert(source.genome.clone()));
    sources.truncate(16);
    for (rank, source) in sources.iter_mut().enumerate() {
        source.rank = rank + 1;
    }
    Ok(sources)
}

fn confirmed_campaign_winner_sources(target_dataset_hash: &str) -> Result<Vec<CandidateSource>> {
    confirmed_campaign_winner_sources_at(&workspace_root().join("runs"), target_dataset_hash)
}

/// Confirmed winners suitable for transfer to a different target.  A target's
/// own prior answer must never enter its discovery seed bank.
fn confirmed_transfer_winner_sources(target_dataset_hash: &str) -> Result<Vec<CandidateSource>> {
    confirmed_transfer_winner_sources_at(&workspace_root().join("runs"), target_dataset_hash)
}

fn confirmed_transfer_winner_sources_at(
    runs_root: &Path,
    target_dataset_hash: &str,
) -> Result<Vec<CandidateSource>> {
    let mut sources = confirmed_campaign_winner_sources_at(runs_root, target_dataset_hash)?
        .into_iter()
        .filter(|source| {
            source
                .path
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                != Some(target_dataset_hash)
        })
        .collect::<Vec<_>>();
    for (rank, source) in sources.iter_mut().enumerate() {
        source.rank = rank + 1;
    }
    Ok(sources)
}

fn confirmed_campaign_winner_sources_at(
    runs_root: &Path,
    target_dataset_hash: &str,
) -> Result<Vec<CandidateSource>> {
    let mut run_roots = fs::read_dir(&runs_root)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    run_roots.sort();
    let mut best_by_dataset = BTreeMap::<String, (f64, CandidateSource)>::new();
    for run_root in run_roots {
        let Some(run_name) = run_root.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let is_golden = run_name.starts_with("golden42-all-regression-")
            || run_name.starts_with("golden46-new4-")
            || run_name.starts_with("golden46-new3-");
        let is_tabarena = run_name.starts_with("tabarena-lite-all-regression-");
        if !is_golden && !is_tabarena {
            continue;
        }
        let datasets_root = run_root.join("datasets");
        if !datasets_root.is_dir() {
            continue;
        }
        let mut result_paths = fs::read_dir(&datasets_root)?
            .map(|entry| Ok(entry?.path().join("final-result.v1.json")))
            .collect::<Result<Vec<_>>>()?;
        result_paths.sort();
        for path in result_paths {
            if !path.is_file() {
                continue;
            }
            let bytes = fs::read(&path)?;
            let value: Value = serde_json::from_slice(&bytes)?;
            let canonical_result = if is_golden {
                serde_json::from_slice::<ConfirmedGoldenTransferResult>(&bytes)
                    .ok()
                    .and_then(|result| canonical_json(&result).ok())
                    .is_some_and(|canonical| canonical == bytes)
            } else {
                serde_json::from_slice::<ConfirmedTabArenaTransferResult>(&bytes)
                    .ok()
                    .and_then(|result| canonical_json(&result).ok())
                    .is_some_and(|canonical| canonical == bytes)
            };
            if !canonical_result
                || value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
            {
                continue;
            }
            let valid_result = if is_golden {
                value.get("schema").and_then(Value::as_str)
                    == Some("JopeGolden42AggressiveDatasetResult.v1")
                    && value.get("test_adaptive").and_then(Value::as_bool) == Some(true)
                    && value.get("outcome").and_then(Value::as_str) == Some("confirmed_strict_win")
            } else {
                matches!(
                    value.get("schema").and_then(Value::as_str),
                    Some("JopeTabArenaLiteIslandResult.v1" | "JopeTabArenaLiteIslandResult.v2")
                ) && value
                    .get("margin_vs_leader")
                    .and_then(Value::as_f64)
                    .is_some_and(|margin| margin.is_finite() && margin > 0.0)
            };
            if !valid_result {
                continue;
            }
            let dataset_field = if is_golden { "dataset_hash" } else { "dataset" };
            let score_field = if is_golden {
                "best_rmse"
            } else {
                "selected_rmse"
            };
            let Some(dataset_hash) = value.get(dataset_field).and_then(Value::as_str) else {
                continue;
            };
            let Some(genome) = value.get("canonical_genome").and_then(Value::as_str) else {
                continue;
            };
            let Some(best_rmse) = value.get(score_field).and_then(Value::as_f64) else {
                continue;
            };
            let Some(leader_rmse) = value.get("leader_rmse").and_then(Value::as_f64) else {
                continue;
            };
            if !best_rmse.is_finite()
                || !leader_rmse.is_finite()
                || leader_rmse <= 0.0
                || best_rmse >= leader_rmse
            {
                continue;
            }
            let result_root = path
                .parent()
                .context("winner result has no dataset directory")?;
            let provenance_valid = if is_golden {
                let Some(confirmation_sha256) =
                    value.get("confirmation_sha256").and_then(Value::as_str)
                else {
                    continue;
                };
                let Some(checkpoint_sha256) = value
                    .get("terminal_checkpoint_sha256")
                    .and_then(Value::as_str)
                else {
                    continue;
                };
                let Some(generation) = value.get("stopping_generation").and_then(Value::as_u64)
                else {
                    continue;
                };
                sha256_file(&result_root.join("confirmation.v1.json"))
                    .ok()
                    .as_deref()
                    == Some(confirmation_sha256)
                    && sha256_file(
                        &result_root
                            .join("checkpoints")
                            .join(format!("generation-{generation:03}.v1.json")),
                    )
                    .ok()
                    .as_deref()
                        == Some(checkpoint_sha256)
            } else {
                value
                    .get("replay")
                    .and_then(Value::as_array)
                    .is_some_and(|replays| {
                        replays.len() == scq_batch::NODES.len()
                            && replays.iter().all(|replay| {
                                let Some(file) = replay.get("receipt_file").and_then(Value::as_str)
                                else {
                                    return false;
                                };
                                let Some(expected) =
                                    replay.get("receipt_sha256").and_then(Value::as_str)
                                else {
                                    return false;
                                };
                                sha256_file(&result_root.join(file)).ok().as_deref()
                                    == Some(expected)
                            })
                    })
            };
            if !provenance_valid {
                continue;
            }
            let relative_rmse = best_rmse / leader_rmse;
            let source = CandidateSource {
                genome: genome.to_string(),
                kind: "confirmed_multi_dataset_campaign_winner".to_string(),
                path: path.clone(),
                file_sha256: sha256_bytes(&bytes),
                record: 0,
                rank: 0,
                score: Some(best_rmse),
                order_score: relative_rmse,
            };
            match best_by_dataset.get(dataset_hash) {
                Some((incumbent, _)) if *incumbent <= relative_rmse => {}
                _ => {
                    best_by_dataset.insert(dataset_hash.to_string(), (relative_rmse, source));
                }
            }
        }
    }
    let mut winners = best_by_dataset.into_iter().collect::<Vec<_>>();
    winners.sort_by(
        |(left_dataset, (left_score, left)), (right_dataset, (right_score, right))| {
            (left_dataset != target_dataset_hash)
                .cmp(&(right_dataset != target_dataset_hash))
                .then_with(|| left_score.total_cmp(right_score))
                .then_with(|| left.path.cmp(&right.path))
        },
    );
    Ok(winners
        .into_iter()
        .enumerate()
        .map(|(rank, (_, (_, mut source)))| {
            source.rank = rank + 1;
            source
        })
        .collect())
}

fn fast30_bank_sources() -> Result<Vec<CandidateSource>> {
    let published =
        Path::new("/home/ubuntu/quant/regression/reports/jain_regressor_fast30_gen0_seeds.txt");
    let path = if published.is_file() {
        published.to_path_buf()
    } else {
        workspace_root().join("seeds/generation-zero.txt")
    };
    let bytes = fs::read(&path)?;
    let file_sha256 = sha256_bytes(&bytes);
    let text = std::str::from_utf8(&bytes)?;
    Ok(text
        .lines()
        .enumerate()
        .filter_map(|(record, line)| {
            let genome = line.trim();
            (!genome.is_empty()).then(|| CandidateSource {
                genome: genome.to_string(),
                kind: "global_ranked_fast30_seed_bank".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: record + 1,
                score: None,
                order_score: record as f64,
            })
        })
        .collect())
}

fn canonicalize_for_features(source: &str, features: usize) -> Result<CanonicalGenome> {
    let canonical = canonicalize(source).map_err(anyhow::Error::msg)?;
    let genome = GenomeIr::parse(&canonical.canonical)?;
    let GenomeIr::Pipeline(pipeline) = genome else {
        bail!("FullRust seed is not a pipeline");
    };
    for stage in pipeline.stages {
        if !matches!(stage.op.as_str(), "map" | "derive") {
            continue;
        }
        let Some(feat_math::dsl::ParamValue::Str(expression)) = stage.params.get("expr") else {
            continue;
        };
        let bytes = expression.as_bytes();
        let mut index = 0;
        while index + 4 <= bytes.len() {
            if &bytes[index..index + 4] != b"col(" {
                index += 1;
                continue;
            }
            let start = index + 4;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end == start || end >= bytes.len() || bytes[end] != b')' {
                bail!("invalid column reference in expression {expression:?}");
            }
            let column = expression[start..end].parse::<usize>()?;
            if column >= features {
                bail!("expression column {column} is outside dataset feature bound {features}");
            }
            index = end + 1;
        }
    }
    Ok(canonical)
}

fn known_fast30(registry: &Path, spec: &DatasetSpec) -> Result<Option<CanonicalGenome>> {
    let path = registry
        .join(&spec.dataset_hash)
        .join("reports/jain_regressor_fast30.json");
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let genome = value
        .get("genome")
        .or_else(|| value.get("gene"))
        .or_else(|| value.pointer("/algorithm/canonical_genome"))
        .and_then(Value::as_str)
        .map(str::to_string);
    match genome {
        Some(genome) => canonicalize_for_features(&genome, spec.features).map(Some),
        None if value.get("status").and_then(Value::as_str) == Some("failed") => Ok(None),
        None => bail!("fast30 report has no genome and is not marked failed"),
    }
}

fn force_fast30_inclusion(seeds: &mut [SeedProvenance], champion: SeedProvenance) -> bool {
    if seeds
        .iter()
        .any(|seed| seed.semantic_sha256 == champion.semantic_sha256)
    {
        return false;
    }
    let slot = seeds.len() - 1;
    seeds[slot] = SeedProvenance { slot, ..champion };
    true
}

fn build_seed_manifest(registry: &Path, spec: &DatasetSpec) -> Result<SeedManifest> {
    let mut ordered = confirmed_campaign_winner_sources(&spec.dataset_hash)?;
    ordered.extend(report_sources(registry, &spec.dataset_hash)?);
    ordered.extend(jope_generation_sources(&spec.dataset_hash)?);
    ordered.extend(fast30_bank_sources()?);
    let mut seen = HashSet::new();
    let mut seeds = Vec::new();
    for source in ordered {
        let canonical = match canonicalize_for_features(&source.genome, spec.features) {
            Ok(canonical) => canonical,
            Err(_) => continue,
        };
        if !seen.insert(canonical.semantic_sha256.clone()) {
            continue;
        }
        seeds.push(SeedProvenance {
            slot: seeds.len(),
            canonical_genome: canonical.canonical,
            semantic_sha256: canonical.semantic_sha256,
            complexity: canonical.complexity,
            rust_source_len: canonical.rust_source_len,
            source_kind: source.kind,
            source_file: source.path.to_string_lossy().into_owned(),
            source_file_sha256: source.file_sha256,
            source_record: source.record,
            source_rank: source.rank,
            source_score: source.score,
        });
        if seeds.len() == POPULATION {
            break;
        }
    }
    if seeds.len() != POPULATION {
        bail!(
            "only {} canonical Rust-emittable, deduplicated generation-zero seeds were found",
            seeds.len()
        );
    }
    let champion = known_fast30(registry, spec)?;
    let champion_path = registry
        .join(&spec.dataset_hash)
        .join("reports/jain_regressor_fast30.json");
    // Keep the population exactly 60 while guaranteeing that the required
    // dataset-specific fast30 champion is freshly evaluated in G0.
    if let Some(champion) = &champion {
        force_fast30_inclusion(
            &mut seeds,
            SeedProvenance {
                slot: POPULATION - 1,
                canonical_genome: champion.canonical.clone(),
                semantic_sha256: champion.semantic_sha256.clone(),
                complexity: champion.complexity,
                rust_source_len: champion.rust_source_len,
                source_kind: "dataset_fast30_champion".to_string(),
                source_file: champion_path.to_string_lossy().into_owned(),
                source_file_sha256: sha256_file(&champion_path)?,
                source_record: 0,
                source_rank: 0,
                source_score: None,
            },
        );
    }
    Ok(SeedManifest {
        schema: SEED_SCHEMA.to_string(),
        benchmark_valid: false,
        dataset_hash: spec.dataset_hash.clone(),
        population: POPULATION,
        known_fast30_semantic_sha256: champion.map(|champion| champion.semantic_sha256),
        seeds,
    })
}

/// Freeze a leave-one-dataset-out generation-zero bank.  Unlike the legacy
/// seed manifest, this deliberately excludes target-specific reports,
/// historical target ledgers, and the target's fast30 champion.
fn build_transfer_seed_manifest(registry: &Path, spec: &DatasetSpec) -> Result<SeedManifest> {
    let _ = registry;
    let mut ordered = prior_test_adaptive_checkpoint_sources(&spec.dataset_hash)?;
    ordered.extend(confirmed_transfer_winner_sources(&spec.dataset_hash)?);
    ordered.extend(fast30_bank_sources()?);
    let mut seen = HashSet::new();
    let mut seeds = Vec::new();
    for source in ordered {
        let canonical = match canonicalize_for_features(&source.genome, spec.features) {
            Ok(canonical) => canonical,
            Err(_) => continue,
        };
        if !seen.insert(canonical.semantic_sha256.clone()) {
            continue;
        }
        seeds.push(SeedProvenance {
            slot: seeds.len(),
            canonical_genome: canonical.canonical,
            semantic_sha256: canonical.semantic_sha256,
            complexity: canonical.complexity,
            rust_source_len: canonical.rust_source_len,
            source_kind: source.kind,
            source_file: source.path.to_string_lossy().into_owned(),
            source_file_sha256: source.file_sha256,
            source_record: source.record,
            source_rank: source.rank,
            source_score: source.score,
        });
        if seeds.len() == POPULATION {
            break;
        }
    }
    if seeds.len() != POPULATION {
        bail!(
            "only {} canonical leave-one-dataset-out seeds were found",
            seeds.len()
        );
    }
    Ok(SeedManifest {
        schema: SEED_SCHEMA.to_string(),
        benchmark_valid: false,
        dataset_hash: spec.dataset_hash.clone(),
        population: POPULATION,
        known_fast30_semantic_sha256: None,
        seeds,
    })
}

fn load_or_create_transfer_seed_manifest(
    path: &Path,
    registry: &Path,
    spec: &DatasetSpec,
) -> Result<SeedManifest> {
    if path.is_file() {
        let bytes = fs::read(path)?;
        let manifest: SeedManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&manifest)?
            || manifest.schema != SEED_SCHEMA
            || manifest.benchmark_valid
            || manifest.dataset_hash != spec.dataset_hash
            || manifest.population != POPULATION
            || manifest.known_fast30_semantic_sha256.is_some()
            || manifest.seeds.len() != POPULATION
            || manifest
                .seeds
                .iter()
                .enumerate()
                .any(|(slot, seed)| seed.slot != slot)
            || manifest
                .seeds
                .iter()
                .map(|seed| &seed.semantic_sha256)
                .collect::<HashSet<_>>()
                .len()
                != POPULATION
        {
            bail!("invalid frozen transfer seed manifest");
        }
        return Ok(manifest);
    }
    let manifest = build_transfer_seed_manifest(registry, spec)?;
    immutable_write(path, &canonical_json(&manifest)?)?;
    Ok(manifest)
}

fn rank_outcomes(left: &SlotOutcome, right: &SlotOutcome) -> Ordering {
    left.rmse()
        .total_cmp(&right.rmse())
        .then_with(|| left.member.complexity.cmp(&right.member.complexity))
        .then_with(|| {
            left.member
                .rust_source_len
                .cmp(&right.member.rust_source_len)
        })
        .then_with(|| {
            left.member
                .semantic_sha256
                .cmp(&right.member.semantic_sha256)
        })
}

fn offspring_seed(generation: usize, slot: usize, retry: usize) -> u64 {
    let text = format!("{RNG_SEED}:{generation}:{slot}:{retry}");
    let digest = Sha256::digest(text.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

fn tournament_winner(rng: &mut Pcg32, pool: usize) -> usize {
    (0..TOURNAMENT)
        .map(|_| rng.next_u32() as usize % pool)
        .min()
        .expect("tournament is nonempty")
}

fn next_population(
    ranked: &[SlotOutcome],
    generation: usize,
    features: usize,
) -> Result<Vec<Member>> {
    if ranked.len() < PARENT_POOL || ranked[..PARENT_POOL].iter().any(|slot| !slot.valid) {
        bail!("parent pool contains invalid candidates");
    }
    let mut next = Vec::with_capacity(POPULATION);
    for elite in ranked.iter().take(ELITES) {
        let mut member = elite.member.clone();
        member.ancestry = Ancestry {
            operation: "elite".to_string(),
            parents: vec![elite.member.semantic_sha256.clone()],
        };
        next.push(member);
    }
    for slot in ELITES..POPULATION {
        let mut produced = None;
        for retry in 0..100 {
            let seed = offspring_seed(generation, slot, retry);
            let mut rng = Pcg32::new(seed, 0x5445_5354_4144_4150);
            let first = tournament_winner(&mut rng, PARENT_POOL);
            let first_parent = &ranked[first].member;
            let candidate = if rng.next_f64() < 0.25 {
                let second = tournament_winner(&mut rng, PARENT_POOL);
                let second_parent = &ranked[second].member;
                crossover_pipelines(
                    &first_parent.canonical_genome,
                    &second_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "crossover".to_string(),
                            parents: vec![
                                first_parent.semantic_sha256.clone(),
                                second_parent.semantic_sha256.clone(),
                            ],
                        },
                    )
                })
            } else {
                mutate_pipeline(
                    &first_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    features,
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "mutation".to_string(),
                            parents: vec![first_parent.semantic_sha256.clone()],
                        },
                    )
                })
            };
            if let Ok(candidate) = candidate {
                produced = Some(candidate);
                break;
            }
        }
        next.push(produced.context("could not create a valid FullRust descendant")?);
    }
    Ok(next)
}

fn job_id(run_id: &str, purpose: &str, generation: usize, slot: usize, attempt: usize) -> String {
    let compact_purpose = purpose
        .bytes()
        .filter(|byte| byte.is_ascii_alphanumeric())
        .map(char::from)
        .collect::<String>();
    format!("jope-{run_id}-{compact_purpose}-g{generation:02}-s{slot:02}-a{attempt}")
}

fn make_request(
    identity: &CampaignIdentity,
    registry: &Path,
    purpose: &str,
    generation: usize,
    slot: usize,
    attempt: usize,
    member: &Member,
    node: &str,
) -> CandidateRequest {
    CandidateRequest {
        schema: REQUEST_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: None,
        purpose: purpose.to_string(),
        run_id: identity.run_id.clone(),
        generation,
        slot,
        attempt,
        semantic_sha256: member.semantic_sha256.clone(),
        canonical_genome: member.canonical_genome.clone(),
        binary_sha256: identity.binary_sha256.clone(),
        dataset_hash: identity.dataset_hash.clone(),
        train_sha256: identity.train_sha256.clone(),
        test_sha256: identity.test_sha256.clone(),
        train_rows: identity.train_rows,
        test_rows: identity.test_rows,
        features: identity.features,
        dataset_root: registry
            .join(&identity.dataset_hash)
            .to_string_lossy()
            .into_owned(),
        execution_host: identity.execution_host.clone(),
        node: node.to_string(),
        scq_job_id: job_id(&identity.run_id, purpose, generation, slot, attempt),
        thread_counts: ThreadCounts::single(),
    }
}

fn receipt_relative(receipt: &CandidateReceipt) -> PathBuf {
    PathBuf::from("receipts")
        .join(&receipt.purpose)
        .join(format!("generation-{:03}", receipt.generation))
        .join(format!(
            "slot-{:03}-attempt-{}.v1.json",
            receipt.slot, receipt.attempt
        ))
}

fn validate_receipt_against_request(
    receipt: &CandidateReceipt,
    request: &CandidateRequest,
    test: &RegressionData,
) -> Result<()> {
    if receipt.schema != RECEIPT_SCHEMA
        || receipt.benchmark_valid
        || receipt.test_adaptive != request.test_adaptive
        || receipt.purpose != request.purpose
        || receipt.run_id != request.run_id
        || receipt.generation != request.generation
        || receipt.slot != request.slot
        || receipt.attempt != request.attempt
        || receipt.semantic_sha256 != request.semantic_sha256
        || receipt.canonical_genome != request.canonical_genome
        || receipt.binary_sha256 != request.binary_sha256
        || receipt.dataset_hash != request.dataset_hash
        || receipt.train_sha256 != request.train_sha256
        || receipt.test_sha256 != request.test_sha256
        || receipt.train_rows != request.train_rows
        || receipt.test_rows != request.test_rows
        || receipt.features != request.features
        || receipt.execution_host != request.execution_host
        || receipt.node != request.node
        || receipt.scq_job_id != request.scq_job_id
    {
        bail!("candidate receipt does not bind exactly to its request");
    }
    if receipt.thread_counts != request.thread_counts || !receipt.thread_counts.all_single() {
        bail!("candidate did not use single-thread execution");
    }
    match (&receipt.metrics, &receipt.failure) {
        (Some(recorded), None) => {
            if receipt.predictions.len() != test.y.len() {
                bail!("candidate prediction count mismatch");
            }
            let recomputed = metrics(&test.y, &receipt.predictions)?;
            if recorded.rmse.to_bits() != recomputed.rmse.to_bits()
                || recorded.mae.to_bits() != recomputed.mae.to_bits()
                || recorded.r2.to_bits() != recomputed.r2.to_bits()
            {
                bail!("candidate metrics do not exactly match predictions");
            }
            if receipt.predictions_sha256 != sha256_bytes(&prediction_bytes(&receipt.predictions)) {
                bail!("candidate prediction hash mismatch");
            }
        }
        (None, Some(_))
            if receipt.predictions.is_empty()
                && receipt.predictions_sha256 == sha256_bytes(&[]) => {}
        _ => bail!("receipt must contain either metrics/predictions or a failure"),
    }
    Ok(())
}

fn read_campaign_data(
    registry: &Path,
    spec: &DatasetSpec,
) -> Result<(RegressionData, RegressionData)> {
    let dataset = registry.join(&spec.dataset_hash);
    let train_path = dataset.join("train.csv");
    let test_path = dataset.join("test.csv");
    if sha256_file(&train_path)? != spec.train_sha256
        || sha256_file(&test_path)? != spec.test_sha256
    {
        bail!("pinned train/test SHA-256 mismatch");
    }
    let train = read_regression_csv(&train_path)?;
    let test = read_regression_csv(&test_path)?;
    if train.x.len() != spec.train_rows
        || test.x.len() != spec.test_rows
        || train.features != spec.features
        || test.features != spec.features
    {
        bail!("pinned dataset shape mismatch");
    }
    Ok((train, test))
}

fn run_command(mut command: Command, label: &str) -> Result<std::process::Output> {
    let output = command
        .output()
        .with_context(|| format!("launch {label}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output)
}

fn stage_bundle(id: &CampaignIdentity, root: &Path) -> Result<(PathBuf, StagingReceipt)> {
    let source_binary = std::env::current_exe()?;
    let campaign_bytes = canonical_json(id)?;
    let campaign_sha256 = sha256_bytes(&campaign_bytes);
    let mut nodes = BTreeMap::new();
    nodes.insert(
        "xbabe2".to_string(),
        NodeStaging {
            binary_sha256: sha256_file(&source_binary)?,
            train_sha256: id.train_sha256.clone(),
            test_sha256: id.test_sha256.clone(),
        },
    );
    let receipt = StagingReceipt {
        schema: STAGING_SCHEMA.to_string(),
        benchmark_valid: false,
        binary_sha256: id.binary_sha256.clone(),
        campaign_sha256,
        bundle_root: source_binary.to_string_lossy().into_owned(),
        nodes,
        doctor_output_sha256: sha256_bytes(b"localhost-only\n"),
    };
    let receipt_path = root.join("staging-receipt.v1.json");
    if receipt_path.exists() {
        let recorded: StagingReceipt = serde_json::from_slice(&fs::read(&receipt_path)?)?;
        if recorded != receipt {
            bail!("staging receipt changed on resume");
        }
    } else {
        immutable_write(&receipt_path, &canonical_json(&receipt)?)?;
    }
    Ok((source_binary, receipt))
}

fn remote_request_path(request: &CandidateRequest) -> PathBuf {
    PathBuf::from("/home/ubuntu/jope-test-adaptive/runs")
        .join(&request.run_id)
        .join("requests")
        .join(&request.purpose)
        .join(format!("generation-{:03}", request.generation))
        .join(format!(
            "slot-{:03}-attempt-{}.json",
            request.slot, request.attempt
        ))
}

fn parse_worker_receipt(stdout: &[u8]) -> Result<CandidateReceipt> {
    let text = std::str::from_utf8(stdout)?;
    let payload = text
        .lines()
        .find_map(|line| line.strip_prefix(RESULT_PREFIX))
        .context("worker emitted no receipt record")?;
    serde_json::from_str(payload).context("parse worker receipt")
}

fn evaluate_remote(
    identity: &CampaignIdentity,
    registry: &Path,
    root: &Path,
    remote_binary: &Path,
    purpose: &str,
    generation: usize,
    slot: usize,
    member: &Member,
    node: &str,
    test: &RegressionData,
) -> Result<CachedEvaluation> {
    for attempt in 0..=1 {
        let mut request = make_request(
            identity, registry, purpose, generation, slot, attempt, member, node,
        );
        request.scq_job_id = format!("localhost:{}", request.scq_job_id);
        let relative = PathBuf::from("receipts")
            .join(purpose)
            .join(format!("generation-{generation:03}"))
            .join(format!("slot-{slot:03}-attempt-{attempt}.v1.json"));
        let destination = root.join(&relative);
        if destination.exists() {
            let bytes = fs::read(&destination)?;
            let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
            validate_receipt_against_request(&receipt, &request, test)?;
            return Ok(CachedEvaluation {
                source: CacheSource { generation, slot },
                receipt_file: relative.to_string_lossy().into_owned(),
                receipt_sha256: sha256_bytes(&bytes),
                receipt,
            });
        }
    }
    let mut last_error = None;
    let started = Instant::now();
    let mut last_request = None;
    for attempt in 0..=1 {
        let mut request = make_request(
            identity, registry, purpose, generation, slot, attempt, member, node,
        );
        request.scq_job_id = format!("localhost:{}", request.scq_job_id);
        last_request = Some(request.clone());
        let local_request = root
            .join("requests")
            .join(purpose)
            .join(format!("generation-{generation:03}"))
            .join(format!("slot-{slot:03}-attempt-{attempt}.v1.json"));
        let request_bytes = canonical_json(&request)?;
        if local_request.exists() {
            if fs::read(&local_request)? != request_bytes {
                bail!("immutable request collision at {}", local_request.display());
            }
        } else {
            immutable_write(&local_request, &request_bytes)?;
        }
        let relative = PathBuf::from("receipts")
            .join(purpose)
            .join(format!("generation-{generation:03}"))
            .join(format!("slot-{slot:03}-attempt-{attempt}.v1.json"));
        let destination = root.join(&relative);
        if destination.exists() {
            let bytes = fs::read(&destination)?;
            let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
            validate_receipt_against_request(&receipt, &request, test)?;
            return Ok(CachedEvaluation {
                source: CacheSource { generation, slot },
                receipt_file: relative.to_string_lossy().into_owned(),
                receipt_sha256: sha256_bytes(&bytes),
                receipt,
            });
        }
        let mut command = Command::new("timeout");
        command
            .arg("900")
            .arg("/usr/bin/env")
            .args([
                "RAYON_NUM_THREADS=1",
                "RUST_NUM_THREADS=1",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "VECLIB_MAXIMUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
            ])
            .arg(remote_binary)
            .arg("test-adaptive-worker")
            .arg("--request")
            .arg(&local_request);
        match run_command(command, "localhost candidate worker")
            .and_then(|output| parse_worker_receipt(&output.stdout))
            .and_then(|receipt| {
                validate_receipt_against_request(&receipt, &request, test)?;
                Ok(receipt)
            }) {
            Ok(receipt) => {
                let relative = receipt_relative(&receipt);
                let bytes = canonical_json(&receipt)?;
                let sha256 = sha256_bytes(&bytes);
                let destination = root.join(&relative);
                if destination.exists() {
                    if fs::read(&destination)? != bytes {
                        bail!("immutable receipt collision at {}", destination.display());
                    }
                } else {
                    immutable_write(&destination, &bytes)?;
                }
                return Ok(CachedEvaluation {
                    source: CacheSource { generation, slot },
                    receipt_file: relative.to_string_lossy().into_owned(),
                    receipt_sha256: sha256,
                    receipt,
                });
            }
            Err(error) => last_error = Some(error),
        }
    }
    let request = last_request.context("worker produced no request")?;
    let message = last_error
        .map(|error| format!("{error:#}"))
        .unwrap_or_else(|| "worker failed without diagnostic".to_string());
    let class = if message.contains("status 124") {
        "timeout"
    } else {
        "infrastructure"
    };
    let receipt = CandidateReceipt {
        schema: RECEIPT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: request.test_adaptive,
        purpose: request.purpose.clone(),
        run_id: request.run_id.clone(),
        generation,
        slot,
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
        runtime_seconds: started.elapsed().as_secs_f64(),
        thread_counts: request.thread_counts.clone(),
        predictions: Vec::new(),
        predictions_sha256: sha256_bytes(&[]),
        metrics: None,
        failure: Some(FailureState {
            class: class.to_string(),
            message,
            retryable: true,
        }),
    };
    validate_receipt_against_request(&receipt, &request, test)?;
    let relative = receipt_relative(&receipt);
    let bytes = canonical_json(&receipt)?;
    immutable_write(&root.join(&relative), &bytes)?;
    Ok(CachedEvaluation {
        source: CacheSource { generation, slot },
        receipt_file: relative.to_string_lossy().into_owned(),
        receipt_sha256: sha256_bytes(&bytes),
        receipt,
    })
}

fn outcome_from_evaluation(
    generation: usize,
    slot: usize,
    member: Member,
    evaluation: &CachedEvaluation,
    cached: bool,
) -> SlotOutcome {
    SlotOutcome {
        generation,
        slot,
        member,
        valid: evaluation.receipt.valid(evaluation.receipt.test_rows),
        metrics: evaluation.receipt.metrics.clone(),
        failure: evaluation.receipt.failure.clone(),
        cache_source: cached.then(|| evaluation.source.clone()),
        receipt_file: evaluation.receipt_file.clone(),
        receipt_sha256: evaluation.receipt_sha256.clone(),
        node: evaluation.receipt.node.clone(),
        scq_job_id: evaluation.receipt.scq_job_id.clone(),
        attempt: evaluation.receipt.attempt,
    }
}

fn evaluate_generation(
    identity: &CampaignIdentity,
    registry: &Path,
    root: &Path,
    remote_binary: &Path,
    generation: usize,
    population: Vec<Member>,
    cache: &mut HashMap<String, CachedEvaluation>,
    test: &RegressionData,
) -> Result<(Vec<SlotOutcome>, usize)> {
    let mut outcomes = vec![None; POPULATION];
    let mut pending = BTreeMap::<String, Vec<(usize, Member)>>::new();
    for (slot, member) in population.into_iter().enumerate() {
        if let Some(evaluation) = cache.get(&member.semantic_sha256) {
            outcomes[slot] = Some(outcome_from_evaluation(
                generation, slot, member, evaluation, true,
            ));
        } else {
            pending
                .entry(member.semantic_sha256.clone())
                .or_default()
                .push((slot, member));
        }
    }
    let fresh = pending.len();
    let mut handles = Vec::with_capacity(fresh);
    for entries in pending.into_values() {
        let (slot, member) = entries[0].clone();
        let identity = identity.clone();
        let registry = registry.to_path_buf();
        let root = root.to_path_buf();
        let remote_binary = remote_binary.to_path_buf();
        let test = test.clone();
        let node = node_for(generation, slot).to_string();
        handles.push((
            entries,
            thread::spawn(move || {
                evaluate_remote(
                    &identity,
                    &registry,
                    &root,
                    &remote_binary,
                    "candidate",
                    generation,
                    slot,
                    &member,
                    &node,
                    &test,
                )
            }),
        ));
    }
    for (entries, handle) in handles {
        let evaluation = handle
            .join()
            .map_err(|_| anyhow::anyhow!("candidate worker controller thread panicked"))??;
        cache.insert(
            evaluation.receipt.semantic_sha256.clone(),
            evaluation.clone(),
        );
        for (index, (slot, member)) in entries.into_iter().enumerate() {
            outcomes[slot] = Some(outcome_from_evaluation(
                generation,
                slot,
                member,
                &evaluation,
                index > 0,
            ));
        }
    }
    Ok((
        outcomes
            .into_iter()
            .map(|outcome| outcome.expect("all logical slots evaluated"))
            .collect(),
        fresh,
    ))
}

fn checkpoint_path(root: &Path, generation: usize) -> PathBuf {
    root.join("checkpoints")
        .join(format!("generation-{generation:03}.v1.json"))
}

fn load_checkpoints(
    root: &Path,
    identity: &CampaignIdentity,
) -> Result<Vec<(GenerationCheckpoint, String)>> {
    let mut checkpoints = Vec::new();
    let mut previous = None;
    for generation in 0..MAX_GENERATIONS {
        let path = checkpoint_path(root, generation);
        if !path.exists() {
            break;
        }
        let bytes = fs::read(&path)?;
        let checkpoint: GenerationCheckpoint = serde_json::from_slice(&bytes)?;
        if checkpoint.schema != CHECKPOINT_SCHEMA
            || checkpoint.benchmark_valid
            || checkpoint.run_id != identity.run_id
            || checkpoint.dataset_hash != identity.dataset_hash
            || checkpoint.generation != generation
            || checkpoint.previous_checkpoint_sha256 != previous
            || checkpoint.logical_slots != POPULATION
            || checkpoint.slots.len() != POPULATION
        {
            bail!("invalid checkpoint {}", path.display());
        }
        let sha256 = sha256_bytes(&bytes);
        previous = Some(sha256.clone());
        checkpoints.push((checkpoint, sha256));
    }
    Ok(checkpoints)
}

fn replay_champion(
    identity: &CampaignIdentity,
    registry: &Path,
    root: &Path,
    remote_binary: &Path,
    generation: usize,
    champion: &Member,
    test: &RegressionData,
) -> Result<Vec<ReplayRecord>> {
    let purpose = "replay-final";
    let mut evaluations = Vec::new();
    for slot in 0..3 {
        evaluations.push(evaluate_remote(
            identity,
            registry,
            root,
            remote_binary,
            purpose,
            generation,
            slot,
            champion,
            EXECUTION_NODE,
            test,
        )?);
    }
    let receipts = evaluations
        .iter()
        .map(|evaluation| &evaluation.receipt)
        .collect::<Vec<_>>();
    let base = &receipts[0];
    let base_rmse = base.metrics.as_ref().context("replay failed")?.rmse;
    let mut records = Vec::new();
    for (index, receipt) in receipts.iter().enumerate() {
        let rmse = receipt.metrics.as_ref().context("replay failed")?.rmse;
        if (rmse - base_rmse).abs() > 1e-12 {
            bail!("champion replay RMSE disagreement");
        }
        let max_difference = base
            .predictions
            .iter()
            .zip(&receipt.predictions)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        if max_difference > 1e-10 {
            bail!("champion replay prediction disagreement");
        }
        records.push(ReplayRecord {
            receipt_file: evaluations[index].receipt_file.clone(),
            receipt_sha256: evaluations[index].receipt_sha256.clone(),
            rmse,
            maximum_prediction_difference: max_difference,
        });
    }
    Ok(records)
}

fn qualifying_descendant(slot: &SlotOutcome, generation: usize, g0: f64, leader: f64) -> bool {
    generation > 0
        && slot.member.evolved_descendant
        && slot.valid
        && slot.rmse() < g0
        && slot.rmse() < leader
}

fn reconstruct_lineage(
    checkpoints: &[(GenerationCheckpoint, String)],
    generation: usize,
    semantic_sha256: &str,
) -> Result<Vec<LineageRecord>> {
    fn visit(
        checkpoints: &[(GenerationCheckpoint, String)],
        generation: usize,
        semantic_sha256: &str,
        seen: &mut BTreeSet<(usize, String)>,
        output: &mut Vec<LineageRecord>,
    ) -> Result<()> {
        let key = (generation, semantic_sha256.to_string());
        if !seen.insert(key) {
            return Ok(());
        }
        let slot = checkpoints[generation]
            .0
            .slots
            .iter()
            .find(|slot| slot.member.semantic_sha256 == semantic_sha256)
            .with_context(|| {
                format!("lineage member {semantic_sha256} missing in generation {generation}")
            })?;
        for parent in &slot.member.ancestry.parents {
            let parent_generation = (0..generation).rev().find(|candidate_generation| {
                checkpoints[*candidate_generation]
                    .0
                    .slots
                    .iter()
                    .any(|candidate| candidate.member.semantic_sha256 == *parent)
            });
            if let Some(parent_generation) = parent_generation {
                visit(checkpoints, parent_generation, parent, seen, output)?;
            } else if slot.member.ancestry.operation != "seed" {
                bail!("lineage parent {parent} is not present in an earlier generation");
            }
        }
        output.push(LineageRecord {
            generation,
            slot: slot.slot,
            semantic_sha256: semantic_sha256.to_string(),
            operation: slot.member.ancestry.operation.clone(),
            parents: slot.member.ancestry.parents.clone(),
        });
        Ok(())
    }

    let mut output = Vec::new();
    visit(
        checkpoints,
        generation,
        semantic_sha256,
        &mut BTreeSet::new(),
        &mut output,
    )?;
    output.sort_by(|left, right| {
        left.generation
            .cmp(&right.generation)
            .then_with(|| left.slot.cmp(&right.slot))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    Ok(output)
}

fn reconstruct_slot_lineage(
    checkpoints: &[(GenerationCheckpoint, String)],
    generation: usize,
    slot: usize,
) -> Result<Vec<LineageRecord>> {
    fn visit(
        checkpoints: &[(GenerationCheckpoint, String)],
        generation: usize,
        slot: usize,
        seen: &mut BTreeSet<(usize, usize)>,
        output: &mut Vec<LineageRecord>,
    ) -> Result<()> {
        if !seen.insert((generation, slot)) {
            return Ok(());
        }
        let outcome = checkpoints
            .get(generation)
            .and_then(|checkpoint| checkpoint.0.slots.get(slot))
            .with_context(|| format!("lineage slot {slot} missing in generation {generation}"))?;
        for parent in &outcome.member.ancestry.parents {
            let parent_location = (0..generation).rev().find_map(|parent_generation| {
                checkpoints[parent_generation]
                    .0
                    .slots
                    .iter()
                    .find(|candidate| candidate.member.semantic_sha256 == *parent)
                    .map(|candidate| (parent_generation, candidate.slot))
            });
            if let Some((parent_generation, parent_slot)) = parent_location {
                visit(checkpoints, parent_generation, parent_slot, seen, output)?;
            } else if outcome.member.ancestry.operation != "seed" {
                bail!("lineage parent {parent} is not present in an earlier generation");
            }
        }
        output.push(LineageRecord {
            generation,
            slot,
            semantic_sha256: outcome.member.semantic_sha256.clone(),
            operation: outcome.member.ancestry.operation.clone(),
            parents: outcome.member.ancestry.parents.clone(),
        });
        Ok(())
    }

    let mut output = Vec::new();
    visit(
        checkpoints,
        generation,
        slot,
        &mut BTreeSet::new(),
        &mut output,
    )?;
    output.sort_by(|left, right| {
        left.generation
            .cmp(&right.generation)
            .then_with(|| left.slot.cmp(&right.slot))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    Ok(output)
}

fn started_at(root: &Path) -> Result<u128> {
    let path = root.join("dataset-start.v1.json");
    if path.exists() {
        let value: Value = serde_json::from_slice(&fs::read(path)?)?;
        return value
            .get("started_at_unix_ms")
            .and_then(Value::as_u64)
            .map(u128::from)
            .context("invalid dataset start receipt");
    }
    let started = unix_millis();
    immutable_write(
        &path,
        &canonical_json(&serde_json::json!({
            "schema": "JopeTestAdaptiveDatasetStart.v1",
            "benchmark_valid": false,
            "started_at_unix_ms": started,
        }))?,
    )?;
    Ok(started)
}

fn load_checkpoint_cache(
    root: &Path,
    checkpoints: &[(GenerationCheckpoint, String)],
    identity: &CampaignIdentity,
    test: &RegressionData,
) -> Result<HashMap<String, CachedEvaluation>> {
    let mut cache = HashMap::new();
    for (checkpoint, _) in checkpoints {
        for outcome in &checkpoint.slots {
            if outcome.cache_source.is_some() {
                continue;
            }
            let receipt_path = root.join(&outcome.receipt_file);
            let bytes = fs::read(&receipt_path)?;
            if sha256_bytes(&bytes) != outcome.receipt_sha256 {
                bail!("receipt hash mismatch at {}", receipt_path.display());
            }
            let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
            let request_path = root
                .join("requests")
                .join(&receipt.purpose)
                .join(format!("generation-{:03}", receipt.generation))
                .join(format!(
                    "slot-{:03}-attempt-{}.v1.json",
                    receipt.slot, receipt.attempt
                ));
            let request: CandidateRequest = serde_json::from_slice(&fs::read(request_path)?)?;
            validate_receipt_against_request(&receipt, &request, test)?;
            if receipt.run_id != identity.run_id || receipt.dataset_hash != identity.dataset_hash {
                bail!("cached receipt campaign identity mismatch");
            }
            cache.insert(
                outcome.member.semantic_sha256.clone(),
                CachedEvaluation {
                    source: CacheSource {
                        generation: outcome.generation,
                        slot: outcome.slot,
                    },
                    receipt_file: outcome.receipt_file.clone(),
                    receipt_sha256: outcome.receipt_sha256.clone(),
                    receipt,
                },
            );
        }
    }
    Ok(cache)
}

fn finalize_dataset(
    identity: &CampaignIdentity,
    registry: &Path,
    root: &Path,
    binary: &Path,
    checkpoints: &[(GenerationCheckpoint, String)],
    test: &RegressionData,
    started_at_unix_ms: u128,
) -> Result<DatasetResult> {
    let (terminal, _) = checkpoints.last().context("dataset has no checkpoints")?;
    if !terminal.qualifying_descendant_win && terminal.generation + 1 != MAX_GENERATIONS {
        bail!("cannot finalize a nonterminal dataset");
    }
    let selected = terminal
        .slots
        .iter()
        .find(|slot| slot.slot == terminal.best_slot)
        .context("terminal best slot is missing")?;
    let replay = replay_champion(
        identity,
        registry,
        root,
        binary,
        terminal.generation,
        &selected.member,
        test,
    )?;
    let receipt_bytes = fs::read(root.join(&selected.receipt_file))?;
    if sha256_bytes(&receipt_bytes) != selected.receipt_sha256 {
        bail!("selected receipt hash mismatch");
    }
    let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
    let selected_metrics = selected
        .metrics
        .as_ref()
        .context("selected candidate failed")?;
    let prediction_file = "final-predictions.csv".to_string();
    let prediction_bytes = prediction_bytes(&receipt.predictions);
    let prediction_sha256 = sha256_bytes(&prediction_bytes);
    let prediction_path = root.join(&prediction_file);
    if prediction_path.exists() {
        if fs::read(&prediction_path)? != prediction_bytes {
            bail!("immutable final prediction collision");
        }
    } else {
        immutable_write(&prediction_path, &prediction_bytes)?;
    }
    let result = DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: identity.run_id.clone(),
        dataset_hash: identity.dataset_hash.clone(),
        outcome: if terminal.qualifying_descendant_win {
            "early_win".to_string()
        } else {
            "no_win_after_10_generations".to_string()
        },
        stopping_generation: terminal.generation,
        leader_models: identity.leader_models.clone(),
        leader_rmse: identity.leader_rmse,
        g0_rmse: terminal.g0_best_rmse,
        final_rmse: selected_metrics.rmse,
        final_mae: selected_metrics.mae,
        final_r2: selected_metrics.r2,
        margin_vs_g0: terminal.g0_best_rmse - selected_metrics.rmse,
        margin_vs_leader: identity.leader_rmse - selected_metrics.rmse,
        canonical_genome: selected.member.canonical_genome.clone(),
        semantic_sha256: selected.member.semantic_sha256.clone(),
        complexity: selected.member.complexity,
        rust_source_len: selected.member.rust_source_len,
        generation: terminal.generation,
        slot: selected.slot,
        prediction_file,
        prediction_sha256,
        runtime_seconds: receipt.runtime_seconds,
        logical_evaluations: checkpoints.len() * POPULATION,
        fresh_evaluations: terminal.cumulative_fresh_evaluations,
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
        replay,
        lineage: reconstruct_lineage(
            checkpoints,
            terminal.generation,
            &selected.member.semantic_sha256,
        )?,
    };
    immutable_write(
        &root.join("final-result.v1.json"),
        &canonical_json(&result)?,
    )?;
    Ok(result)
}

fn run_dataset(
    registry: &Path,
    panel_run_id: &str,
    root: &Path,
    spec: &DatasetSpec,
    binary_sha256: &str,
    resume: bool,
) -> Result<DatasetResult> {
    let result_path = root.join("final-result.v1.json");
    if result_path.exists() {
        return serde_json::from_slice(&fs::read(result_path)?)
            .context("read final dataset result");
    }
    let (_train, test) = read_campaign_data(registry, spec)?;
    let campaign_path = root.join("campaign.v1.json");
    let expected_identity = identity(panel_run_id, binary_sha256.to_string(), spec);
    let expected_manifest = build_seed_manifest(registry, spec)?;
    if campaign_path.exists() {
        if !resume {
            bail!("run already exists; pass --resume or choose a new run-id");
        }
        let recorded: CampaignIdentity = serde_json::from_slice(&fs::read(&campaign_path)?)?;
        if recorded != expected_identity {
            bail!("campaign identity or binary changed; refusing non-reproducible resume");
        }
        let manifest: SeedManifest =
            serde_json::from_slice(&fs::read(root.join("seed-manifest.v1.json"))?)?;
        if manifest != expected_manifest {
            bail!("generation-zero seed manifest changed");
        }
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(&campaign_path, &canonical_json(&expected_identity)?)?;
        immutable_write(
            &root.join("seed-manifest.v1.json"),
            &canonical_json(&expected_manifest)?,
        )?;
    }
    let (remote_binary, _) = stage_bundle(&expected_identity, &root)?;
    let started_at_unix_ms = started_at(root)?;
    let checkpoints = load_checkpoints(root, &expected_identity)?;
    let mut cache = load_checkpoint_cache(root, &checkpoints, &expected_identity, &test)?;
    let mut population = checkpoints.last().map_or_else(
        || {
            expected_manifest
                .seeds
                .iter()
                .map(Member::from_seed)
                .collect()
        },
        |(checkpoint, _)| checkpoint.next_population.clone(),
    );
    let mut previous_sha = checkpoints.last().map(|(_, sha)| sha.clone());
    let mut cumulative_fresh = checkpoints
        .last()
        .map_or(0, |(checkpoint, _)| checkpoint.cumulative_fresh_evaluations);
    let mut g0_best = checkpoints
        .first()
        .map_or(f64::INFINITY, |(checkpoint, _)| checkpoint.g0_best_rmse);
    if checkpoints
        .last()
        .is_some_and(|(checkpoint, _)| checkpoint.qualifying_descendant_win)
        || checkpoints.len() == MAX_GENERATIONS
    {
        return finalize_dataset(
            &expected_identity,
            registry,
            root,
            &remote_binary,
            &checkpoints,
            &test,
            started_at_unix_ms,
        );
    }

    let mut completed = checkpoints;
    for generation in completed.len()..MAX_GENERATIONS {
        let (slots, fresh) = evaluate_generation(
            &expected_identity,
            registry,
            root,
            &remote_binary,
            generation,
            population,
            &mut cache,
            &test,
        )?;
        let valid_candidates = slots.iter().filter(|slot| slot.valid).count();
        if valid_candidates < MIN_VALID {
            bail!("generation {generation} has only {valid_candidates} valid candidates");
        }
        let mut ranked = slots.clone();
        ranked.sort_by(rank_outcomes);
        if generation == 0 {
            g0_best = ranked[0].rmse();
        }
        let winning_descendant = ranked.iter().find(|slot| {
            slot.member.evolved_descendant
                && slot.valid
                && slot.rmse() < g0_best
                && qualifying_descendant(slot, generation, g0_best, spec.leader_rmse)
        });
        let result_label = winning_descendant.map(|_| "early_win".to_string());
        cumulative_fresh += fresh;
        let next = if generation + 1 < MAX_GENERATIONS {
            next_population(&ranked, generation + 1, spec.features)?
        } else {
            Vec::new()
        };
        let checkpoint = GenerationCheckpoint {
            schema: CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: panel_run_id.to_string(),
            dataset_hash: spec.dataset_hash.clone(),
            generation,
            previous_checkpoint_sha256: previous_sha.clone(),
            logical_slots: POPULATION,
            fresh_evaluations: fresh,
            cumulative_fresh_evaluations: cumulative_fresh,
            valid_candidates,
            g0_best_rmse: g0_best,
            best_slot: ranked[0].slot,
            best_semantic_sha256: ranked[0].member.semantic_sha256.clone(),
            best_rmse: ranked[0].rmse(),
            qualifying_descendant_win: winning_descendant.is_some(),
            result_label,
            ranked_slots: ranked.iter().map(|slot| slot.slot).collect(),
            slots,
            next_population: next.clone(),
        };
        let bytes = canonical_json(&checkpoint)?;
        let checkpoint_sha256 = sha256_bytes(&bytes);
        immutable_write(&checkpoint_path(root, generation), &bytes)?;
        previous_sha = Some(checkpoint_sha256.clone());
        population = next;
        eprintln!(
            "generation {generation}: best RMSE {:.17}, valid {valid_candidates}/{POPULATION}, fresh {fresh}",
            checkpoint.best_rmse
        );

        completed.push((checkpoint.clone(), checkpoint_sha256));
        if checkpoint.qualifying_descendant_win || generation + 1 == MAX_GENERATIONS {
            return finalize_dataset(
                &expected_identity,
                registry,
                root,
                &remote_binary,
                &completed,
                &test,
                started_at_unix_ms,
            );
        }
    }
    bail!("dataset campaign ended without a terminal result")
}

/// Preserve the original one-dataset entry point as a localhost smoke path.
pub(crate) fn run(registry: &Path, run_id: &str, resume: bool) -> Result<()> {
    validate_run_id(run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let spec = specs
        .iter()
        .find(|spec| spec.dataset_hash == LEGACY_DATASET)
        .context("legacy smoke dataset is absent from Golden42")?;
    let binary_sha256 = current_binary_sha256()?;
    run_dataset(
        registry,
        run_id,
        &run_root(run_id),
        spec,
        &binary_sha256,
        resume,
    )?;
    Ok(())
}

/// Evaluate one candidate request locally (the SCQ wrapper simply forwards its
/// JSON request to this command and captures the machine-readable receipt).
pub(crate) fn worker(request_path: &Path) -> Result<()> {
    let request: CandidateRequest = serde_json::from_slice(&fs::read(request_path)?)?;
    let receipt = scq_batch::evaluate_request(&request)?;
    println!(
        "{}{}",
        RESULT_PREFIX,
        String::from_utf8(canonical_json(&receipt)?).unwrap()
    );
    Ok(())
}

fn expected_panel_manifest(
    registry: &Path,
    run_id: &str,
    panel: &PanelFile,
    specs: &[DatasetSpec],
    panel_file_sha256: String,
    leader_manifest_sha256: String,
    binary_sha256: String,
) -> PanelManifest {
    PanelManifest {
        schema: PANEL_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: run_id.to_string(),
        ordered_dataset_ids: panel.dataset_ids.clone(),
        ordered_cohort_sha256: panel.panel_dataset_sha256.clone(),
        panel_file: PANEL_PATH.to_string(),
        panel_file_sha256,
        leader_manifest: workspace_root()
            .join(LEADER_MANIFEST)
            .to_string_lossy()
            .into_owned(),
        leader_manifest_sha256,
        release_binary_sha256: binary_sha256,
        registry_root: registry.to_string_lossy().into_owned(),
        panel_declared_dataset_root: panel.dataset_root.clone(),
        execution_host: EXECUTION_HOST.to_string(),
        execution_node: EXECUTION_NODE.to_string(),
        config: PanelConfig::fixed(),
        datasets: specs.to_vec(),
    }
}

fn walk_files(directory: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk_files(&path, output)?;
        } else {
            output.push(path);
        }
    }
    Ok(())
}

fn completed_artifact_fingerprint(
    root: &Path,
    completed_dataset_ids: &[String],
) -> Result<(usize, String)> {
    let mut paths = Vec::new();
    for dataset_id in completed_dataset_ids {
        let dataset_root = root.join("datasets").join(dataset_id);
        if !dataset_root.is_dir() {
            bail!("completed dataset {dataset_id} has no artifact directory");
        }
        walk_files(&dataset_root, &mut paths)?;
    }
    paths.sort();
    let mut binding = Vec::new();
    for path in &paths {
        binding.extend_from_slice(path.strip_prefix(root)?.to_string_lossy().as_bytes());
        binding.push(0);
        binding.extend_from_slice(sha256_file(path)?.as_bytes());
        binding.push(b'\n');
    }
    Ok((paths.len(), sha256_bytes(&binding)))
}

fn dataset_directory_has_artifacts(path: &Path) -> Result<bool> {
    if !path.is_dir() {
        return Ok(false);
    }
    Ok(fs::read_dir(path)?.next().transpose()?.is_some())
}

fn clean_continuation_boundary(root: &Path, dataset_ids: &[String]) -> Result<usize> {
    let mut first_incomplete = None;
    for (index, dataset_id) in dataset_ids.iter().enumerate() {
        let dataset_root = root.join("datasets").join(dataset_id);
        let terminal = dataset_root.join("final-result.v1.json").is_file();
        if terminal {
            if first_incomplete.is_some() {
                bail!("terminal dataset {dataset_id} appears after the continuation boundary");
            }
            continue;
        }
        first_incomplete.get_or_insert(index);
        if dataset_directory_has_artifacts(&dataset_root)? {
            bail!(
                "dataset {dataset_id} was already started; refusing a binary or seed change within a partially evaluated dataset"
            );
        }
    }
    Ok(first_incomplete.unwrap_or(dataset_ids.len()))
}

fn binary_segments(
    dataset_ids: &[String],
    boundary: usize,
    original_binary_sha256: &str,
    corrected_binary_sha256: &str,
) -> Vec<BinarySegment> {
    let mut segments = Vec::new();
    if boundary > 0 {
        segments.push(BinarySegment {
            binary_sha256: original_binary_sha256.to_string(),
            first_dataset_index: 0,
            last_dataset_index: boundary - 1,
            dataset_count: boundary,
            dataset_ids: dataset_ids[..boundary].to_vec(),
        });
    }
    if boundary < dataset_ids.len() {
        segments.push(BinarySegment {
            binary_sha256: corrected_binary_sha256.to_string(),
            first_dataset_index: boundary,
            last_dataset_index: dataset_ids.len() - 1,
            dataset_count: dataset_ids.len() - boundary,
            dataset_ids: dataset_ids[boundary..].to_vec(),
        });
    }
    segments
}

fn validate_continuation_shape(
    manifest: &PanelManifest,
    manifest_sha256: &str,
    continuation: &ContinuationRecord,
) -> Result<()> {
    let boundary = continuation.completed_dataset_count;
    if continuation.schema != CONTINUATION_SCHEMA
        || continuation.benchmark_valid
        || continuation.run_id != manifest.run_id
        || continuation.original_panel_manifest_sha256 != manifest_sha256
        || continuation.original_binary_sha256 != manifest.release_binary_sha256
        || boundary > manifest.ordered_dataset_ids.len()
        || continuation.first_remaining_dataset_index != boundary
        || continuation.first_remaining_dataset_ordinal != boundary + 1
        || continuation.completed_dataset_ids != manifest.ordered_dataset_ids[..boundary]
        || continuation.remaining_dataset_ids != manifest.ordered_dataset_ids[boundary..]
        || continuation.seed_fix_reason != FAST30_FIX_REASON
    {
        bail!("panel continuation identity or boundary mismatch");
    }
    Ok(())
}

fn validate_continuation_artifacts(
    root: &Path,
    manifest: &PanelManifest,
    continuation: &ContinuationRecord,
) -> Result<()> {
    let (count, sha256) =
        completed_artifact_fingerprint(root, &continuation.completed_dataset_ids)?;
    if count != continuation.completed_artifact_count
        || sha256 != continuation.completed_artifacts_sha256
    {
        bail!("a completed pre-continuation dataset artifact changed");
    }
    for (index, dataset_id) in manifest.ordered_dataset_ids.iter().enumerate() {
        let dataset_root = root.join("datasets").join(dataset_id);
        let campaign_path = dataset_root.join("campaign.v1.json");
        if index < continuation.completed_dataset_count {
            if !dataset_root.join("final-result.v1.json").is_file() {
                bail!("completed dataset {dataset_id} lost its terminal result");
            }
            let campaign: CampaignIdentity = serde_json::from_slice(&fs::read(&campaign_path)?)?;
            if campaign.binary_sha256 != continuation.original_binary_sha256 {
                bail!("completed dataset {dataset_id} binary binding changed");
            }
        }
    }
    Ok(())
}

fn validate_amendment_shape(
    manifest: &PanelManifest,
    continuation: &ContinuationRecord,
    continuation_sha256: &str,
    amendment: &ContinuationAmendment,
) -> Result<()> {
    let boundary = amendment.completed_dataset_count;
    if amendment.schema != CONTINUATION_AMENDMENT_SCHEMA
        || amendment.benchmark_valid
        || amendment.run_id != manifest.run_id
        || amendment.previous_continuation_file != CONTINUATION_FILE
        || amendment.previous_continuation_sha256 != continuation_sha256
        || amendment.previous_binary_sha256 != continuation.corrected_binary_sha256
        || boundary <= continuation.completed_dataset_count
        || boundary > manifest.ordered_dataset_ids.len()
        || amendment.first_remaining_dataset_index != boundary
        || amendment.first_remaining_dataset_ordinal != boundary + 1
        || amendment.completed_dataset_ids != manifest.ordered_dataset_ids[..boundary]
        || amendment.remaining_dataset_ids != manifest.ordered_dataset_ids[boundary..]
        || amendment.correction_reason != FAST30_ABSENCE_FIX_REASON
    {
        bail!("panel continuation amendment identity or boundary mismatch");
    }
    Ok(())
}

fn validate_amendment_artifacts(
    root: &Path,
    manifest: &PanelManifest,
    continuation: &ContinuationRecord,
    amendment: &ContinuationAmendment,
) -> Result<()> {
    let (count, sha256) = completed_artifact_fingerprint(root, &amendment.completed_dataset_ids)?;
    if count != amendment.completed_artifact_count || sha256 != amendment.completed_artifacts_sha256
    {
        bail!("a completed pre-amendment dataset artifact changed");
    }
    for (index, dataset_id) in amendment.completed_dataset_ids.iter().enumerate() {
        let dataset_root = root.join("datasets").join(dataset_id);
        if !dataset_root.join("final-result.v1.json").is_file() {
            bail!("completed dataset {dataset_id} lost its terminal result");
        }
        let campaign: CampaignIdentity =
            serde_json::from_slice(&fs::read(dataset_root.join("campaign.v1.json"))?)?;
        let expected_binary = if index < continuation.completed_dataset_count {
            &continuation.original_binary_sha256
        } else {
            &continuation.corrected_binary_sha256
        };
        if &campaign.binary_sha256 != expected_binary {
            bail!("completed dataset {dataset_id} binary binding changed");
        }
    }
    if amendment.completed_dataset_ids
        != manifest.ordered_dataset_ids[..amendment.completed_dataset_count]
    {
        bail!("amendment completed cohort is not a panel prefix");
    }
    Ok(())
}

fn push_binary_segment(
    segments: &mut Vec<BinarySegment>,
    dataset_ids: &[String],
    start: usize,
    end: usize,
    binary_sha256: &str,
) {
    if start == end {
        return;
    }
    segments.push(BinarySegment {
        binary_sha256: binary_sha256.to_string(),
        first_dataset_index: start,
        last_dataset_index: end - 1,
        dataset_count: end - start,
        dataset_ids: dataset_ids[start..end].to_vec(),
    });
}

fn prepare_panel_execution(
    root: &Path,
    manifest: &PanelManifest,
    current_binary_sha256: &str,
    allow_create_continuation: bool,
) -> Result<PanelExecution> {
    let manifest_path = root.join("panel-manifest.v1.json");
    let manifest_sha256 = sha256_file(&manifest_path)?;
    let continuation_path = root.join(CONTINUATION_FILE);
    let continuation = if continuation_path.is_file() {
        let recorded: ContinuationRecord = serde_json::from_slice(&fs::read(&continuation_path)?)?;
        validate_continuation_shape(manifest, &manifest_sha256, &recorded)?;
        validate_continuation_artifacts(root, manifest, &recorded)?;
        Some(recorded)
    } else if current_binary_sha256 == manifest.release_binary_sha256 {
        None
    } else {
        if !allow_create_continuation {
            None
        } else {
            let boundary = clean_continuation_boundary(root, &manifest.ordered_dataset_ids)?;
            let completed_dataset_ids = manifest.ordered_dataset_ids[..boundary].to_vec();
            let (completed_artifact_count, completed_artifacts_sha256) =
                completed_artifact_fingerprint(root, &completed_dataset_ids)?;
            let record = ContinuationRecord {
                schema: CONTINUATION_SCHEMA.to_string(),
                benchmark_valid: false,
                run_id: manifest.run_id.clone(),
                original_panel_manifest_sha256: manifest_sha256.clone(),
                original_binary_sha256: manifest.release_binary_sha256.clone(),
                corrected_binary_sha256: current_binary_sha256.to_string(),
                completed_dataset_count: boundary,
                first_remaining_dataset_index: boundary,
                first_remaining_dataset_ordinal: boundary + 1,
                completed_dataset_ids,
                remaining_dataset_ids: manifest.ordered_dataset_ids[boundary..].to_vec(),
                completed_artifact_count,
                completed_artifacts_sha256,
                seed_fix_reason: FAST30_FIX_REASON.to_string(),
                continuation_started_at_unix_ms: unix_millis(),
            };
            immutable_write(&continuation_path, &canonical_json(&record)?)?;
            validate_continuation_artifacts(root, manifest, &record)?;
            Some(record)
        }
    };

    let amendment_path = root.join(CONTINUATION_AMENDMENT_FILE);
    let amendment = if let Some(continuation) = &continuation {
        let continuation_sha256 = sha256_file(&continuation_path)?;
        if amendment_path.is_file() {
            let recorded: ContinuationAmendment =
                serde_json::from_slice(&fs::read(&amendment_path)?)?;
            validate_amendment_shape(manifest, continuation, &continuation_sha256, &recorded)?;
            validate_amendment_artifacts(root, manifest, continuation, &recorded)?;
            Some(recorded)
        } else if current_binary_sha256 != continuation.corrected_binary_sha256
            && allow_create_continuation
        {
            let boundary = clean_continuation_boundary(root, &manifest.ordered_dataset_ids)?;
            if boundary <= continuation.completed_dataset_count {
                bail!("continuation amendment does not advance the completed boundary");
            }
            let completed_dataset_ids = manifest.ordered_dataset_ids[..boundary].to_vec();
            let (completed_artifact_count, completed_artifacts_sha256) =
                completed_artifact_fingerprint(root, &completed_dataset_ids)?;
            let record = ContinuationAmendment {
                schema: CONTINUATION_AMENDMENT_SCHEMA.to_string(),
                benchmark_valid: false,
                run_id: manifest.run_id.clone(),
                previous_continuation_file: CONTINUATION_FILE.to_string(),
                previous_continuation_sha256: continuation_sha256,
                previous_binary_sha256: continuation.corrected_binary_sha256.clone(),
                amended_binary_sha256: current_binary_sha256.to_string(),
                completed_dataset_count: boundary,
                first_remaining_dataset_index: boundary,
                first_remaining_dataset_ordinal: boundary + 1,
                completed_dataset_ids,
                remaining_dataset_ids: manifest.ordered_dataset_ids[boundary..].to_vec(),
                completed_artifact_count,
                completed_artifacts_sha256,
                correction_reason: FAST30_ABSENCE_FIX_REASON.to_string(),
                continuation_started_at_unix_ms: unix_millis(),
            };
            immutable_write(&amendment_path, &canonical_json(&record)?)?;
            validate_amendment_artifacts(root, manifest, continuation, &record)?;
            Some(record)
        } else {
            None
        }
    } else {
        None
    };

    let dataset_ids = &manifest.ordered_dataset_ids;
    let mut binary_segments = Vec::new();
    let mut dataset_binary_sha256 = vec![manifest.release_binary_sha256.clone(); dataset_ids.len()];
    if let Some(continuation) = &continuation {
        let first_boundary = continuation.completed_dataset_count;
        let second_boundary = amendment
            .as_ref()
            .map_or(dataset_ids.len(), |record| record.completed_dataset_count);
        push_binary_segment(
            &mut binary_segments,
            dataset_ids,
            0,
            first_boundary,
            &continuation.original_binary_sha256,
        );
        push_binary_segment(
            &mut binary_segments,
            dataset_ids,
            first_boundary,
            second_boundary,
            &continuation.corrected_binary_sha256,
        );
        for binary in &mut dataset_binary_sha256[first_boundary..second_boundary] {
            *binary = continuation.corrected_binary_sha256.clone();
        }
        if let Some(amendment) = &amendment {
            push_binary_segment(
                &mut binary_segments,
                dataset_ids,
                second_boundary,
                dataset_ids.len(),
                &amendment.amended_binary_sha256,
            );
            for binary in &mut dataset_binary_sha256[second_boundary..] {
                *binary = amendment.amended_binary_sha256.clone();
            }
        }
    } else {
        push_binary_segment(
            &mut binary_segments,
            dataset_ids,
            0,
            dataset_ids.len(),
            &manifest.release_binary_sha256,
        );
    }

    for (index, dataset_id) in dataset_ids.iter().enumerate() {
        let dataset_root = root.join("datasets").join(dataset_id);
        let campaign_path = dataset_root.join("campaign.v1.json");
        if campaign_path.is_file() {
            let campaign: CampaignIdentity = serde_json::from_slice(&fs::read(&campaign_path)?)?;
            validate_dataset_binary(&campaign, &dataset_binary_sha256[index])?;
        } else if dataset_directory_has_artifacts(&dataset_root)? {
            bail!("dataset {dataset_id} has artifacts but no campaign identity");
        }
    }
    Ok(PanelExecution {
        dataset_binary_sha256,
        binary_segments,
        continuation,
        amendment,
    })
}

fn collect_artifacts_excluding(root: &Path, excluded_file: &str) -> Result<Vec<ArtifactHash>> {
    let mut paths = Vec::new();
    walk_files(root, &mut paths)?;
    paths.retain(|path| path != &root.join(excluded_file));
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            Ok(ArtifactHash {
                file: path.strip_prefix(root)?.to_string_lossy().into_owned(),
                sha256: sha256_file(&path)?,
            })
        })
        .collect()
}

fn collect_artifacts(root: &Path) -> Result<Vec<ArtifactHash>> {
    collect_artifacts_excluding(root, SUMMARY_FILE)
}

fn genome_record(
    root: &Path,
    checkpoints: &[(GenerationCheckpoint, String)],
    outcome: &SlotOutcome,
    g0_rmse: f64,
    leader_rmse: f64,
    binary_sha256: &str,
) -> Result<GenomeRecord> {
    let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
    if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
        bail!("genome receipt hash mismatch at {}", outcome.receipt_file);
    }
    let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
    let metrics = outcome
        .metrics
        .as_ref()
        .context("reported genome is invalid")?;
    if receipt.semantic_sha256 != outcome.member.semantic_sha256
        || receipt.canonical_genome != outcome.member.canonical_genome
        || receipt.binary_sha256 != binary_sha256
        || receipt.metrics.as_ref() != Some(metrics)
        || receipt.predictions_sha256 != sha256_bytes(&prediction_bytes(&receipt.predictions))
    {
        bail!("genome does not reconstruct from its immutable receipt");
    }
    Ok(GenomeRecord {
        canonical_genome: outcome.member.canonical_genome.clone(),
        semantic_sha256: outcome.member.semantic_sha256.clone(),
        generation: outcome.generation,
        slot: outcome.slot,
        rmse: metrics.rmse,
        mae: metrics.mae,
        r2: metrics.r2,
        margin_vs_g0: g0_rmse - metrics.rmse,
        margin_vs_leader: leader_rmse - metrics.rmse,
        complexity: outcome.member.complexity,
        rust_source_len: outcome.member.rust_source_len,
        runtime_seconds: receipt.runtime_seconds,
        lineage: reconstruct_slot_lineage(checkpoints, outcome.generation, outcome.slot)?,
        prediction_sha256: receipt.predictions_sha256,
        receipt_file: outcome.receipt_file.clone(),
        receipt_sha256: outcome.receipt_sha256.clone(),
        binary_sha256: binary_sha256.to_string(),
    })
}

fn collect_qualifying_genomes(
    root: &Path,
    checkpoints: &[(GenerationCheckpoint, String)],
    spec: &DatasetSpec,
    binary_sha256: &str,
) -> Result<Vec<GenomeRecord>> {
    let g0_rmse = checkpoints
        .first()
        .context("dataset has no generation-zero checkpoint")?
        .0
        .g0_best_rmse;
    let mut records = Vec::new();
    for (checkpoint, _) in checkpoints {
        for outcome in &checkpoint.slots {
            if qualifying_descendant(outcome, checkpoint.generation, g0_rmse, spec.leader_rmse) {
                records.push(genome_record(
                    root,
                    checkpoints,
                    outcome,
                    g0_rmse,
                    spec.leader_rmse,
                    binary_sha256,
                )?);
            }
        }
    }
    Ok(records)
}

fn make_winner_map(
    manifest: &PanelManifest,
    specs: &[DatasetSpec],
    results: &[DatasetResult],
    execution: &PanelExecution,
    root: &Path,
) -> Result<WinnerMap> {
    if specs.len() != results.len()
        || specs.len() != execution.dataset_binary_sha256.len()
        || specs.len() != manifest.ordered_dataset_ids.len()
    {
        bail!("winner-map input lengths differ");
    }
    let mut datasets = Vec::with_capacity(specs.len());
    for (index, ((spec, result), binary_sha256)) in specs
        .iter()
        .zip(results)
        .zip(&execution.dataset_binary_sha256)
        .enumerate()
    {
        if spec.dataset_hash != manifest.ordered_dataset_ids[index]
            || result.dataset_hash != spec.dataset_hash
        {
            bail!("winner-map dataset order mismatch");
        }
        let dataset_root = root.join("datasets").join(&spec.dataset_hash);
        let identity: CampaignIdentity =
            serde_json::from_slice(&fs::read(dataset_root.join("campaign.v1.json"))?)?;
        validate_dataset_binary(&identity, binary_sha256)?;
        let checkpoints = load_checkpoints(&dataset_root, &identity)?;
        let terminal = &checkpoints
            .last()
            .context("dataset has no terminal checkpoint")?
            .0;
        let selected = terminal
            .slots
            .iter()
            .find(|outcome| outcome.slot == terminal.best_slot)
            .context("terminal selected slot is missing")?;
        let final_selected_genome = genome_record(
            &dataset_root,
            &checkpoints,
            selected,
            terminal.g0_best_rmse,
            spec.leader_rmse,
            binary_sha256,
        )?;
        if final_selected_genome.semantic_sha256 != result.semantic_sha256 {
            bail!("winner-map final genome differs from dataset result");
        }
        datasets.push(DatasetWinnerMap {
            dataset_hash: spec.dataset_hash.clone(),
            final_selected_genome,
            qualifying_descendants: collect_qualifying_genomes(
                &dataset_root,
                &checkpoints,
                spec,
                binary_sha256,
            )?,
        });
    }
    Ok(WinnerMap {
        schema: WINNER_MAP_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.run_id.clone(),
        ordered_cohort_sha256: manifest.ordered_cohort_sha256.clone(),
        binary_segments: execution.binary_segments.clone(),
        total_qualifying_genomes: datasets
            .iter()
            .map(|dataset| dataset.qualifying_descendants.len())
            .sum(),
        datasets,
    })
}

fn make_summary(
    manifest: &PanelManifest,
    results: &[DatasetResult],
    winner_map: &WinnerMap,
    execution: &PanelExecution,
    root: &Path,
) -> Result<PanelSummary> {
    let early_wins = results
        .iter()
        .filter(|result| result.outcome == "early_win")
        .count();
    let no_wins = results
        .iter()
        .filter(|result| result.outcome == "no_win_after_10_generations")
        .count();
    Ok(PanelSummary {
        schema: SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.run_id.clone(),
        ordered_cohort_sha256: manifest.ordered_cohort_sha256.clone(),
        leader_manifest_sha256: manifest.leader_manifest_sha256.clone(),
        dataset_count: results.len(),
        early_wins,
        no_wins,
        total_qualifying_genomes: winner_map.total_qualifying_genomes,
        total_logical_evaluations: results
            .iter()
            .map(|result| result.logical_evaluations)
            .sum(),
        total_fresh_evaluations: results.iter().map(|result| result.fresh_evaluations).sum(),
        binary_segments: execution.binary_segments.clone(),
        datasets: results
            .iter()
            .zip(&winner_map.datasets)
            .zip(&execution.dataset_binary_sha256)
            .map(|((result, winner), binary_sha256)| PanelDatasetSummary {
                dataset_hash: result.dataset_hash.clone(),
                leader_models: result.leader_models.clone(),
                leader_rmse: result.leader_rmse,
                fresh_g0_rmse: result.g0_rmse,
                outcome: result.outcome.clone(),
                stopping_generation: result.stopping_generation,
                total_generations_evaluated: result.stopping_generation + 1,
                final_selected_semantic_sha256: result.semantic_sha256.clone(),
                qualifying_genome_count: winner.qualifying_descendants.len(),
                logical_evaluations: result.logical_evaluations,
                fresh_evaluations: result.fresh_evaluations,
                binary_sha256: binary_sha256.clone(),
            })
            .collect(),
        winner_map_file: WINNER_MAP_FILE.to_string(),
        winner_map_sha256: sha256_file(&root.join(WINNER_MAP_FILE))?,
        artifacts: collect_artifacts(root)?,
    })
}

pub(crate) fn run_panel(registry: &Path, run_id: &str, resume: bool) -> Result<()> {
    validate_run_id(run_id)?;
    let (panel, specs, panel_sha256, leader_sha256) = load_panel_specs(registry)?;
    let current_binary_sha256 = current_binary_sha256()?;
    let root = run_root(run_id);
    let manifest_path = root.join("panel-manifest.v1.json");
    let manifest = if manifest_path.exists() {
        if !resume {
            bail!("panel run already exists; pass --resume or choose a new run-id");
        }
        let recorded: PanelManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        let expected = expected_panel_manifest(
            registry,
            run_id,
            &panel,
            &specs,
            panel_sha256,
            leader_sha256,
            recorded.release_binary_sha256.clone(),
        );
        if recorded != expected {
            bail!("panel identity, inputs, or configuration changed on resume");
        }
        recorded
    } else {
        if root.exists() && !resume {
            bail!("panel run directory already exists");
        }
        fs::create_dir_all(root.join("datasets"))?;
        let manifest = expected_panel_manifest(
            registry,
            run_id,
            &panel,
            &specs,
            panel_sha256,
            leader_sha256,
            current_binary_sha256.clone(),
        );
        immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    let execution = prepare_panel_execution(&root, &manifest, &current_binary_sha256, resume)?;
    if execution.continuation.is_some() {
        eprintln!("validated immutable mixed-binary continuation record");
    }
    if execution.amendment.is_some() {
        eprintln!("validated immutable continuation amendment");
    }
    if root.join(SUMMARY_FILE).exists() {
        validate_panel(registry, run_id)?;
        return Ok(());
    }

    let mut results = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let dataset_root = root.join("datasets").join(&spec.dataset_hash);
        eprintln!(
            "dataset {}/{} {}",
            index + 1,
            specs.len(),
            spec.dataset_hash
        );
        results.push(run_dataset(
            registry,
            run_id,
            &dataset_root,
            spec,
            &execution.dataset_binary_sha256[index],
            resume,
        )?);
    }
    let winner_map = make_winner_map(&manifest, &specs, &results, &execution, &root)?;
    let winner_map_bytes = canonical_json(&winner_map)?;
    let winner_map_path = root.join(WINNER_MAP_FILE);
    if winner_map_path.exists() {
        if fs::read(&winner_map_path)? != winner_map_bytes {
            bail!("immutable winner-map collision");
        }
    } else {
        immutable_write(&winner_map_path, &winner_map_bytes)?;
    }
    let summary = make_summary(&manifest, &results, &winner_map, &execution, &root)?;
    immutable_write(&root.join(SUMMARY_FILE), &canonical_json(&summary)?)?;
    validate_panel(registry, run_id)
}

fn validate_dataset_binary(
    identity: &CampaignIdentity,
    expected_binary_sha256: &str,
) -> Result<()> {
    if identity.binary_sha256 != expected_binary_sha256 {
        bail!(
            "dataset {} is bound to binary {}, expected {}",
            identity.dataset_hash,
            identity.binary_sha256,
            expected_binary_sha256
        );
    }
    Ok(())
}

fn validate_staging_receipt(root: &Path, identity: &CampaignIdentity) -> Result<()> {
    let staging: StagingReceipt =
        serde_json::from_slice(&fs::read(root.join("staging-receipt.v1.json"))?)?;
    let expected_node = staging.nodes.get(EXECUTION_NODE);
    if staging.schema != STAGING_SCHEMA
        || staging.benchmark_valid
        || staging.binary_sha256 != identity.binary_sha256
        || staging.campaign_sha256 != sha256_bytes(&canonical_json(identity)?)
        || staging.nodes.len() != 1
        || expected_node.is_none_or(|node| {
            node.binary_sha256 != identity.binary_sha256
                || node.train_sha256 != identity.train_sha256
                || node.test_sha256 != identity.test_sha256
        })
        || staging.doctor_output_sha256 != sha256_bytes(b"localhost-only\n")
    {
        bail!("dataset {} staging receipt mismatch", identity.dataset_hash);
    }
    Ok(())
}

fn validate_dataset(
    registry: &Path,
    panel_run_id: &str,
    root: &Path,
    spec: &DatasetSpec,
    binary_sha256: &str,
) -> Result<DatasetResult> {
    let recorded_identity: CampaignIdentity =
        serde_json::from_slice(&fs::read(root.join("campaign.v1.json"))?)?;
    validate_dataset_binary(&recorded_identity, binary_sha256)?;
    let expected_identity = identity(panel_run_id, binary_sha256.to_string(), spec);
    if recorded_identity != expected_identity {
        bail!("dataset {} campaign identity mismatch", spec.dataset_hash);
    }
    validate_staging_receipt(root, &recorded_identity)?;
    let seed_manifest: SeedManifest =
        serde_json::from_slice(&fs::read(root.join("seed-manifest.v1.json"))?)?;
    let expected_seeds = build_seed_manifest(registry, spec)?;
    if seed_manifest != expected_seeds {
        bail!("dataset {} seed manifest mismatch", spec.dataset_hash);
    }
    let (_, test) = read_campaign_data(registry, spec)?;
    let checkpoints = load_checkpoints(root, &recorded_identity)?;
    if checkpoints.is_empty() {
        bail!("dataset {} has no checkpoints", spec.dataset_hash);
    }

    let mut expected_population = expected_seeds
        .seeds
        .iter()
        .map(Member::from_seed)
        .collect::<Vec<_>>();
    let mut expected_previous = None;
    let mut expected_cumulative = 0;
    let mut g0 = f64::INFINITY;
    let mut saw_win = false;
    for (generation, (checkpoint, checkpoint_sha256)) in checkpoints.iter().enumerate() {
        if saw_win {
            bail!(
                "dataset {} continued after a qualifying win",
                spec.dataset_hash
            );
        }
        if checkpoint.previous_checkpoint_sha256 != expected_previous {
            bail!("dataset {} checkpoint chain mismatch", spec.dataset_hash);
        }
        if checkpoint.slots.len() != POPULATION
            || checkpoint.slots.iter().enumerate().any(|(slot, outcome)| {
                outcome.generation != generation
                    || outcome.slot != slot
                    || outcome.member != expected_population[slot]
            })
        {
            bail!("dataset {} logical population mismatch", spec.dataset_hash);
        }
        let mut fresh = 0;
        for outcome in &checkpoint.slots {
            let canonical =
                canonicalize_for_features(&outcome.member.canonical_genome, spec.features)?;
            if canonical.canonical != outcome.member.canonical_genome
                || canonical.semantic_sha256 != outcome.member.semantic_sha256
                || canonical.complexity != outcome.member.complexity
                || canonical.rust_source_len != outcome.member.rust_source_len
            {
                bail!(
                    "dataset {} member canonicalization mismatch",
                    spec.dataset_hash
                );
            }
            let receipt_path = root.join(&outcome.receipt_file);
            let receipt_bytes = fs::read(&receipt_path)?;
            if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
                bail!(
                    "dataset {} candidate receipt hash mismatch",
                    spec.dataset_hash
                );
            }
            let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
            let request_path = root
                .join("requests")
                .join(&receipt.purpose)
                .join(format!("generation-{:03}", receipt.generation))
                .join(format!(
                    "slot-{:03}-attempt-{}.v1.json",
                    receipt.slot, receipt.attempt
                ));
            let request: CandidateRequest = serde_json::from_slice(&fs::read(request_path)?)?;
            validate_receipt_against_request(&receipt, &request, &test)?;
            if receipt.dataset_hash != spec.dataset_hash
                || receipt.run_id != panel_run_id
                || receipt.binary_sha256 != binary_sha256
                || request.binary_sha256 != binary_sha256
                || receipt.semantic_sha256 != outcome.member.semantic_sha256
                || receipt.execution_host != EXECUTION_HOST
                || receipt.node != EXECUTION_NODE
                || !receipt.scq_job_id.starts_with("localhost:")
            {
                bail!("dataset {} receipt provenance mismatch", spec.dataset_hash);
            }
            if outcome.valid != receipt.valid(spec.test_rows)
                || outcome.metrics != receipt.metrics
                || outcome.failure != receipt.failure
                || outcome.node != receipt.node
                || outcome.scq_job_id != receipt.scq_job_id
                || outcome.attempt != receipt.attempt
            {
                bail!("dataset {} slot/receipt mismatch", spec.dataset_hash);
            }
            if let Some(source) = &outcome.cache_source {
                if source.generation > generation
                    || (source.generation == generation && source.slot >= outcome.slot)
                {
                    bail!("dataset {} cache points forward", spec.dataset_hash);
                }
                let source_outcome = &checkpoints[source.generation].0.slots[source.slot];
                if source_outcome.member.semantic_sha256 != outcome.member.semantic_sha256
                    || source_outcome.receipt_file != outcome.receipt_file
                    || source_outcome.receipt_sha256 != outcome.receipt_sha256
                {
                    bail!("dataset {} cache source mismatch", spec.dataset_hash);
                }
            } else {
                fresh += 1;
            }
        }
        let valid = checkpoint.slots.iter().filter(|slot| slot.valid).count();
        if valid != checkpoint.valid_candidates || valid < MIN_VALID {
            bail!(
                "dataset {} valid candidate count mismatch",
                spec.dataset_hash
            );
        }
        if fresh != checkpoint.fresh_evaluations {
            bail!(
                "dataset {} fresh evaluation count mismatch",
                spec.dataset_hash
            );
        }
        expected_cumulative += fresh;
        if expected_cumulative != checkpoint.cumulative_fresh_evaluations {
            bail!(
                "dataset {} cumulative evaluation count mismatch",
                spec.dataset_hash
            );
        }
        let mut ranked = checkpoint.slots.clone();
        ranked.sort_by(rank_outcomes);
        let ranked_slots = ranked.iter().map(|slot| slot.slot).collect::<Vec<_>>();
        if checkpoint.ranked_slots != ranked_slots
            || checkpoint.best_slot != ranked[0].slot
            || checkpoint.best_semantic_sha256 != ranked[0].member.semantic_sha256
            || checkpoint.best_rmse.to_bits() != ranked[0].rmse().to_bits()
        {
            bail!("dataset {} rank checkpoint mismatch", spec.dataset_hash);
        }
        if generation == 0 {
            g0 = ranked[0].rmse();
        }
        if checkpoint.g0_best_rmse.to_bits() != g0.to_bits() {
            bail!(
                "dataset {} generation-zero baseline mismatch",
                spec.dataset_hash
            );
        }
        let qualifies = ranked
            .iter()
            .any(|slot| qualifying_descendant(slot, generation, g0, spec.leader_rmse));
        if checkpoint.qualifying_descendant_win != qualifies
            || checkpoint.result_label != qualifies.then(|| "early_win".to_string())
        {
            bail!(
                "dataset {} strict win qualification mismatch",
                spec.dataset_hash
            );
        }
        let expected_next = if generation + 1 < MAX_GENERATIONS {
            next_population(&ranked, generation + 1, spec.features)?
        } else {
            Vec::new()
        };
        if checkpoint.next_population != expected_next {
            bail!(
                "dataset {} deterministic evolution mismatch",
                spec.dataset_hash
            );
        }
        expected_population = expected_next;
        expected_previous = Some(checkpoint_sha256.clone());
        saw_win = qualifies;
    }
    let terminal = &checkpoints.last().unwrap().0;
    if !terminal.qualifying_descendant_win && checkpoints.len() != MAX_GENERATIONS {
        bail!(
            "dataset {} stopped before generation nine",
            spec.dataset_hash
        );
    }

    let result_bytes = fs::read(root.join("final-result.v1.json"))?;
    let result: DatasetResult = serde_json::from_slice(&result_bytes)?;
    let selected = &terminal.slots[terminal.best_slot];
    let selected_metrics = selected
        .metrics
        .as_ref()
        .context("final candidate is invalid")?;
    let expected_outcome = if terminal.qualifying_descendant_win {
        "early_win"
    } else {
        "no_win_after_10_generations"
    };
    if result.schema != RESULT_SCHEMA
        || result.benchmark_valid
        || result.run_id != panel_run_id
        || result.dataset_hash != spec.dataset_hash
        || result.outcome != expected_outcome
        || result.stopping_generation != terminal.generation
        || result.leader_models != spec.leader_models
        || result.leader_rmse.to_bits() != spec.leader_rmse.to_bits()
        || result.g0_rmse.to_bits() != g0.to_bits()
        || result.final_rmse.to_bits() != selected_metrics.rmse.to_bits()
        || result.final_mae.to_bits() != selected_metrics.mae.to_bits()
        || result.final_r2.to_bits() != selected_metrics.r2.to_bits()
        || result.margin_vs_g0.to_bits() != (g0 - selected_metrics.rmse).to_bits()
        || result.margin_vs_leader.to_bits() != (spec.leader_rmse - selected_metrics.rmse).to_bits()
        || result.canonical_genome != selected.member.canonical_genome
        || result.semantic_sha256 != selected.member.semantic_sha256
        || result.complexity != selected.member.complexity
        || result.rust_source_len != selected.member.rust_source_len
        || result.generation != terminal.generation
        || result.slot != selected.slot
        || result.logical_evaluations != checkpoints.len() * POPULATION
        || result.fresh_evaluations != terminal.cumulative_fresh_evaluations
        || result.finished_at_unix_ms < result.started_at_unix_ms
    {
        bail!("dataset {} final result mismatch", spec.dataset_hash);
    }
    if result.lineage
        != reconstruct_lineage(
            &checkpoints,
            terminal.generation,
            &selected.member.semantic_sha256,
        )?
    {
        bail!("dataset {} final lineage mismatch", spec.dataset_hash);
    }
    let selected_receipt: CandidateReceipt =
        serde_json::from_slice(&fs::read(root.join(&selected.receipt_file))?)?;
    let expected_prediction_bytes = prediction_bytes(&selected_receipt.predictions);
    let recorded_prediction_bytes = fs::read(root.join(&result.prediction_file))?;
    if recorded_prediction_bytes != expected_prediction_bytes
        || result.prediction_sha256 != sha256_bytes(&recorded_prediction_bytes)
    {
        bail!("dataset {} final prediction mismatch", spec.dataset_hash);
    }
    if result.replay.len() != 3 {
        bail!(
            "dataset {} does not have three replay receipts",
            spec.dataset_hash
        );
    }
    let mut replay_receipts = Vec::new();
    for (slot, replay) in result.replay.iter().enumerate() {
        let bytes = fs::read(root.join(&replay.receipt_file))?;
        if sha256_bytes(&bytes) != replay.receipt_sha256 {
            bail!("dataset {} replay hash mismatch", spec.dataset_hash);
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
        let request_path = root
            .join("requests/replay-final")
            .join(format!("generation-{:03}", terminal.generation))
            .join(format!(
                "slot-{slot:03}-attempt-{}.v1.json",
                receipt.attempt
            ));
        let request: CandidateRequest = serde_json::from_slice(&fs::read(request_path)?)?;
        validate_receipt_against_request(&receipt, &request, &test)?;
        if receipt.semantic_sha256 != result.semantic_sha256
            || receipt.purpose != "replay-final"
            || receipt.binary_sha256 != binary_sha256
            || request.binary_sha256 != binary_sha256
        {
            bail!("dataset {} replay genome mismatch", spec.dataset_hash);
        }
        replay_receipts.push(receipt);
    }
    let replay_base = &replay_receipts[0];
    let base_rmse = replay_base.metrics.as_ref().context("failed replay")?.rmse;
    for (index, receipt) in replay_receipts.iter().enumerate() {
        let replay_rmse = receipt.metrics.as_ref().context("failed replay")?.rmse;
        let maximum_difference = replay_base
            .predictions
            .iter()
            .zip(&receipt.predictions)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        if (replay_rmse - base_rmse).abs() > 1e-12
            || maximum_difference > 1e-10
            || result.replay[index].rmse.to_bits() != replay_rmse.to_bits()
            || result.replay[index].maximum_prediction_difference.to_bits()
                != maximum_difference.to_bits()
        {
            bail!("dataset {} replay disagreement", spec.dataset_hash);
        }
    }
    Ok(result)
}

pub(crate) fn validate_panel(registry: &Path, run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let root = run_root(run_id);
    let (panel, specs, panel_sha256, leader_sha256) = load_panel_specs(registry)?;
    if specs.len() != 42 {
        bail!("panel validator requires exactly 42 datasets");
    }
    let current_binary_sha256 = current_binary_sha256()?;
    let manifest_bytes = fs::read(root.join("panel-manifest.v1.json"))?;
    let manifest: PanelManifest = serde_json::from_slice(&manifest_bytes)?;
    let expected_manifest = expected_panel_manifest(
        registry,
        run_id,
        &panel,
        &specs,
        panel_sha256,
        leader_sha256,
        manifest.release_binary_sha256.clone(),
    );
    if manifest != expected_manifest || manifest_bytes != canonical_json(&manifest)? {
        bail!("original panel manifest does not match its frozen inputs");
    }
    let execution = prepare_panel_execution(&root, &manifest, &current_binary_sha256, false)?;
    let mut results = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        results.push(validate_dataset(
            registry,
            run_id,
            &root.join("datasets").join(&spec.dataset_hash),
            spec,
            &execution.dataset_binary_sha256[index],
        )?);
    }
    if results.len() != 42
        || results
            .iter()
            .map(|result| &result.dataset_hash)
            .collect::<BTreeSet<_>>()
            .len()
            != 42
    {
        bail!("panel does not contain exactly 42 distinct terminal results");
    }
    for pair in results.windows(2) {
        if pair[0].finished_at_unix_ms > pair[1].started_at_unix_ms {
            bail!(
                "datasets {} and {} overlap in execution time",
                pair[0].dataset_hash,
                pair[1].dataset_hash
            );
        }
    }

    let winner_map_bytes = fs::read(root.join(WINNER_MAP_FILE))?;
    let recorded_winner_map: WinnerMap = serde_json::from_slice(&winner_map_bytes)?;
    let expected_winner_map = make_winner_map(&manifest, &specs, &results, &execution, &root)?;
    if recorded_winner_map != expected_winner_map
        || winner_map_bytes != canonical_json(&expected_winner_map)?
    {
        bail!("winner map does not reconstruct from immutable receipts");
    }

    let summary_bytes = fs::read(root.join(SUMMARY_FILE))?;
    let recorded_summary: PanelSummary = serde_json::from_slice(&summary_bytes)?;
    let expected_summary =
        make_summary(&manifest, &results, &expected_winner_map, &execution, &root)?;
    if recorded_summary != expected_summary || summary_bytes != canonical_json(&expected_summary)? {
        bail!("panel summary or artifact hashes do not reconstruct");
    }
    eprintln!(
        "validated {} datasets: {} early wins, {} no-wins, {} qualifying genomes, {} logical evaluations, {} fresh evaluations; winner map {}, summary {}",
        recorded_summary.dataset_count,
        recorded_summary.early_wins,
        recorded_summary.no_wins,
        recorded_summary.total_qualifying_genomes,
        recorded_summary.total_logical_evaluations,
        recorded_summary.total_fresh_evaluations,
        sha256_bytes(&winner_map_bytes),
        sha256_bytes(&summary_bytes),
    );
    Ok(())
}

fn generation_10_population(
    base_checkpoints: &[(GenerationCheckpoint, String)],
    spec: &DatasetSpec,
) -> Result<Vec<Member>> {
    let terminal = &base_checkpoints
        .last()
        .context("base dataset has no terminal checkpoint")?
        .0;
    if terminal.generation + 1 != EXTENSION_START_GENERATION || terminal.qualifying_descendant_win {
        bail!("extension requires a generation-nine base no-win");
    }
    let mut ranked = terminal.slots.clone();
    ranked.sort_by(rank_outcomes);
    next_population(&ranked, EXTENSION_START_GENERATION, spec.features)
}

fn extension_identity(
    extension_run_id: &str,
    binary_sha256: &str,
    spec: &DatasetSpec,
) -> CampaignIdentity {
    let mut value = identity(extension_run_id, binary_sha256.to_string(), spec);
    value.generations = EXTENSION_GENERATIONS;
    value.selection_objective = format!(
        "continue immutable base generation 9 through generations {}-{}; rank directly by test RMSE (intentionally test-adaptive)",
        EXTENSION_START_GENERATION,
        EXTENSION_END_GENERATION - 1
    );
    value
}

fn base_no_win_specs<'a>(
    base_root: &Path,
    specs: &'a [DatasetSpec],
) -> Result<Vec<&'a DatasetSpec>> {
    let mut selected = Vec::new();
    for spec in specs {
        let result: DatasetResult = serde_json::from_slice(&fs::read(
            base_root
                .join("datasets")
                .join(&spec.dataset_hash)
                .join("final-result.v1.json"),
        )?)?;
        if result.outcome == "no_win_after_10_generations" {
            selected.push(spec);
        }
    }
    Ok(selected)
}

fn extension_binding(base_root: &Path, spec: &DatasetSpec) -> Result<ExtensionDatasetBinding> {
    let dataset_root = base_root.join("datasets").join(&spec.dataset_hash);
    let campaign_path = dataset_root.join("campaign.v1.json");
    let base_identity: CampaignIdentity = serde_json::from_slice(&fs::read(&campaign_path)?)?;
    let base_checkpoints = load_checkpoints(&dataset_root, &base_identity)?;
    let start_population = generation_10_population(&base_checkpoints, spec)?;
    Ok(ExtensionDatasetBinding {
        dataset_hash: spec.dataset_hash.clone(),
        base_binary_sha256: base_identity.binary_sha256,
        base_campaign_sha256: sha256_file(&campaign_path)?,
        base_final_result_sha256: sha256_file(&dataset_root.join("final-result.v1.json"))?,
        base_terminal_checkpoint_sha256: base_checkpoints
            .last()
            .context("base dataset has no terminal checkpoint")?
            .1
            .clone(),
        generation_10_population_sha256: sha256_bytes(&canonical_json(&start_population)?),
    })
}

fn expected_extension_manifest(
    base_run_id: &str,
    extension_binary_sha256: &str,
    specs: &[DatasetSpec],
    started_at_unix_ms: u128,
) -> Result<ExtensionManifest> {
    let base_root = run_root(base_run_id);
    let selected = base_no_win_specs(&base_root, specs)?;
    let ordered_dataset_ids = selected
        .iter()
        .map(|spec| spec.dataset_hash.clone())
        .collect::<Vec<_>>();
    let datasets = selected
        .iter()
        .map(|spec| extension_binding(&base_root, spec))
        .collect::<Result<Vec<_>>>()?;
    Ok(ExtensionManifest {
        schema: EXTENSION_MANIFEST_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: base_run_id.to_string(),
        extension_run_id: extension_run_id(base_run_id),
        base_panel_manifest_sha256: sha256_file(&base_root.join("panel-manifest.v1.json"))?,
        base_panel_summary_sha256: sha256_file(&base_root.join(SUMMARY_FILE))?,
        base_winner_map_sha256: sha256_file(&base_root.join(WINNER_MAP_FILE))?,
        extension_binary_sha256: extension_binary_sha256.to_string(),
        start_generation: EXTENSION_START_GENERATION,
        generations: EXTENSION_GENERATIONS,
        end_generation_exclusive: EXTENSION_END_GENERATION,
        ordered_cohort_sha256: ordered_cohort_sha256(&ordered_dataset_ids),
        ordered_dataset_ids,
        datasets,
        extension_started_at_unix_ms: started_at_unix_ms,
    })
}

fn load_extension_checkpoints(
    root: &Path,
    identity: &CampaignIdentity,
    base_terminal_checkpoint_sha256: &str,
) -> Result<Vec<(GenerationCheckpoint, String)>> {
    let mut checkpoints = Vec::new();
    let mut previous = Some(base_terminal_checkpoint_sha256.to_string());
    for generation in EXTENSION_START_GENERATION..EXTENSION_END_GENERATION {
        let path = checkpoint_path(root, generation);
        if !path.exists() {
            break;
        }
        let bytes = fs::read(&path)?;
        let checkpoint: GenerationCheckpoint = serde_json::from_slice(&bytes)?;
        if checkpoint.schema != CHECKPOINT_SCHEMA
            || checkpoint.benchmark_valid
            || checkpoint.run_id != identity.run_id
            || checkpoint.dataset_hash != identity.dataset_hash
            || checkpoint.generation != generation
            || checkpoint.previous_checkpoint_sha256 != previous
            || checkpoint.logical_slots != POPULATION
            || checkpoint.slots.len() != POPULATION
        {
            bail!("invalid extension checkpoint {}", path.display());
        }
        let sha256 = sha256_bytes(&bytes);
        previous = Some(sha256.clone());
        checkpoints.push((checkpoint, sha256));
    }
    Ok(checkpoints)
}

fn combined_checkpoints(
    base: &[(GenerationCheckpoint, String)],
    extension: &[(GenerationCheckpoint, String)],
) -> Result<Vec<(GenerationCheckpoint, String)>> {
    if base.len() != EXTENSION_START_GENERATION {
        bail!("base checkpoint history does not end at generation nine");
    }
    let mut combined = base.to_vec();
    combined.extend_from_slice(extension);
    Ok(combined)
}

fn extension_genome_records(
    root: &Path,
    combined: &[(GenerationCheckpoint, String)],
    extension: &[(GenerationCheckpoint, String)],
    spec: &DatasetSpec,
    binary_sha256: &str,
) -> Result<Vec<GenomeRecord>> {
    let g0_rmse = combined[0].0.g0_best_rmse;
    let mut records = Vec::new();
    for (checkpoint, _) in extension {
        for outcome in &checkpoint.slots {
            if qualifying_descendant(outcome, checkpoint.generation, g0_rmse, spec.leader_rmse) {
                records.push(genome_record(
                    root,
                    combined,
                    outcome,
                    g0_rmse,
                    spec.leader_rmse,
                    binary_sha256,
                )?);
            }
        }
    }
    Ok(records)
}

fn finalize_extension_dataset(
    base_run_id: &str,
    identity: &CampaignIdentity,
    registry: &Path,
    root: &Path,
    binary: &Path,
    spec: &DatasetSpec,
    base_checkpoints: &[(GenerationCheckpoint, String)],
    extension_checkpoints: &[(GenerationCheckpoint, String)],
    test: &RegressionData,
    started_at_unix_ms: u128,
) -> Result<ExtensionDatasetResult> {
    if extension_checkpoints.len() != EXTENSION_GENERATIONS {
        bail!("extension dataset is not complete");
    }
    let terminal = &extension_checkpoints.last().unwrap().0;
    let selected = terminal
        .slots
        .iter()
        .find(|slot| slot.slot == terminal.best_slot)
        .context("extension terminal best slot is missing")?;
    let replay = replay_champion(
        identity,
        registry,
        root,
        binary,
        terminal.generation,
        &selected.member,
        test,
    )?;
    let combined = combined_checkpoints(base_checkpoints, extension_checkpoints)?;
    let qualifying_descendants = extension_genome_records(
        root,
        &combined,
        extension_checkpoints,
        spec,
        &identity.binary_sha256,
    )?;
    let final_selected_genome = genome_record(
        root,
        &combined,
        selected,
        combined[0].0.g0_best_rmse,
        spec.leader_rmse,
        &identity.binary_sha256,
    )?;
    let first_qualifying_generation = qualifying_descendants
        .iter()
        .map(|record| record.generation)
        .min();
    let result = ExtensionDatasetResult {
        schema: EXTENSION_RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: base_run_id.to_string(),
        extension_run_id: identity.run_id.clone(),
        dataset_hash: spec.dataset_hash.clone(),
        outcome: if first_qualifying_generation.is_some() {
            "new_leader_found".to_string()
        } else {
            "no_win_after_20_total_generations".to_string()
        },
        start_generation: EXTENSION_START_GENERATION,
        stopping_generation: EXTENSION_END_GENERATION - 1,
        extension_generations_evaluated: EXTENSION_GENERATIONS,
        total_generations_evaluated: EXTENSION_END_GENERATION,
        leader_models: spec.leader_models.clone(),
        leader_rmse: spec.leader_rmse,
        fresh_g0_rmse: combined[0].0.g0_best_rmse,
        first_qualifying_generation,
        final_selected_genome,
        qualifying_descendants,
        logical_evaluations: EXTENSION_GENERATIONS * POPULATION,
        fresh_evaluations: terminal.cumulative_fresh_evaluations,
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
        replay,
        extension_binary_sha256: identity.binary_sha256.clone(),
    };
    immutable_write(
        &root.join("final-result.v1.json"),
        &canonical_json(&result)?,
    )?;
    Ok(result)
}

fn run_extension_dataset(
    registry: &Path,
    manifest: &ExtensionManifest,
    binding: &ExtensionDatasetBinding,
    spec: &DatasetSpec,
    resume: bool,
) -> Result<ExtensionDatasetResult> {
    let root = extension_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let result_path = root.join("final-result.v1.json");
    if result_path.is_file() {
        return serde_json::from_slice(&fs::read(result_path)?)
            .context("read final extension result");
    }
    let (_, test) = read_campaign_data(registry, spec)?;
    let base_dataset_root = run_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let base_identity: CampaignIdentity =
        serde_json::from_slice(&fs::read(base_dataset_root.join("campaign.v1.json"))?)?;
    let base_checkpoints = load_checkpoints(&base_dataset_root, &base_identity)?;
    let start_population = generation_10_population(&base_checkpoints, spec)?;
    if sha256_bytes(&canonical_json(&start_population)?) != binding.generation_10_population_sha256
    {
        bail!("generation-10 population binding changed");
    }
    let expected_identity = extension_identity(
        &manifest.extension_run_id,
        &manifest.extension_binary_sha256,
        spec,
    );
    let campaign_path = root.join("campaign.v1.json");
    if campaign_path.is_file() {
        if !resume {
            bail!("extension dataset already exists; pass --resume");
        }
        let recorded: CampaignIdentity = serde_json::from_slice(&fs::read(&campaign_path)?)?;
        if recorded != expected_identity
            || fs::read(root.join("base-binding.v1.json"))? != canonical_json(binding)?
        {
            bail!("extension dataset identity or base binding changed");
        }
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(&campaign_path, &canonical_json(&expected_identity)?)?;
        immutable_write(
            &root.join("base-binding.v1.json"),
            &canonical_json(binding)?,
        )?;
    }
    let (binary, _) = stage_bundle(&expected_identity, &root)?;
    let started_at_unix_ms = started_at(&root)?;
    let checkpoints = load_extension_checkpoints(
        &root,
        &expected_identity,
        &binding.base_terminal_checkpoint_sha256,
    )?;
    let mut cache = load_checkpoint_cache(&root, &checkpoints, &expected_identity, &test)?;
    let mut population = checkpoints.last().map_or(start_population, |checkpoint| {
        checkpoint.0.next_population.clone()
    });
    let mut previous_sha = checkpoints.last().map_or_else(
        || binding.base_terminal_checkpoint_sha256.clone(),
        |checkpoint| checkpoint.1.clone(),
    );
    let mut cumulative_fresh = checkpoints
        .last()
        .map_or(0, |checkpoint| checkpoint.0.cumulative_fresh_evaluations);
    let g0_rmse = base_checkpoints[0].0.g0_best_rmse;
    if checkpoints.len() == EXTENSION_GENERATIONS {
        return finalize_extension_dataset(
            &manifest.base_run_id,
            &expected_identity,
            registry,
            &root,
            &binary,
            spec,
            &base_checkpoints,
            &checkpoints,
            &test,
            started_at_unix_ms,
        );
    }

    let mut completed = checkpoints;
    for generation in (EXTENSION_START_GENERATION + completed.len())..EXTENSION_END_GENERATION {
        let (slots, fresh) = evaluate_generation(
            &expected_identity,
            registry,
            &root,
            &binary,
            generation,
            population,
            &mut cache,
            &test,
        )?;
        let valid_candidates = slots.iter().filter(|slot| slot.valid).count();
        if valid_candidates < MIN_VALID {
            bail!("extension generation {generation} has only {valid_candidates} valid candidates");
        }
        let mut ranked = slots.clone();
        ranked.sort_by(rank_outcomes);
        let qualifies = ranked
            .iter()
            .any(|slot| qualifying_descendant(slot, generation, g0_rmse, spec.leader_rmse));
        cumulative_fresh += fresh;
        let next = if generation + 1 < EXTENSION_END_GENERATION {
            next_population(&ranked, generation + 1, spec.features)?
        } else {
            Vec::new()
        };
        let checkpoint = GenerationCheckpoint {
            schema: CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: expected_identity.run_id.clone(),
            dataset_hash: spec.dataset_hash.clone(),
            generation,
            previous_checkpoint_sha256: Some(previous_sha.clone()),
            logical_slots: POPULATION,
            fresh_evaluations: fresh,
            cumulative_fresh_evaluations: cumulative_fresh,
            valid_candidates,
            g0_best_rmse: g0_rmse,
            best_slot: ranked[0].slot,
            best_semantic_sha256: ranked[0].member.semantic_sha256.clone(),
            best_rmse: ranked[0].rmse(),
            qualifying_descendant_win: qualifies,
            result_label: qualifies.then(|| "new_leader_found".to_string()),
            ranked_slots: ranked.iter().map(|slot| slot.slot).collect(),
            slots,
            next_population: next.clone(),
        };
        let bytes = canonical_json(&checkpoint)?;
        let sha256 = sha256_bytes(&bytes);
        immutable_write(&checkpoint_path(&root, generation), &bytes)?;
        eprintln!(
            "extension generation {generation}: best RMSE {:.17}, valid {valid_candidates}/{POPULATION}, fresh {fresh}, new leader {qualifies}",
            checkpoint.best_rmse
        );
        previous_sha = sha256.clone();
        population = next;
        completed.push((checkpoint, sha256));
    }
    finalize_extension_dataset(
        &manifest.base_run_id,
        &expected_identity,
        registry,
        &root,
        &binary,
        spec,
        &base_checkpoints,
        &completed,
        &test,
        started_at_unix_ms,
    )
}

fn make_extension_summary(
    manifest: &ExtensionManifest,
    results: Vec<ExtensionDatasetResult>,
    root: &Path,
) -> Result<ExtensionSummary> {
    Ok(ExtensionSummary {
        schema: EXTENSION_SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        extension_run_id: manifest.extension_run_id.clone(),
        base_panel_summary_sha256: manifest.base_panel_summary_sha256.clone(),
        base_winner_map_sha256: manifest.base_winner_map_sha256.clone(),
        extension_binary_sha256: manifest.extension_binary_sha256.clone(),
        dataset_count: results.len(),
        new_leaders: results
            .iter()
            .filter(|result| result.outcome == "new_leader_found")
            .count(),
        still_no_leader: results
            .iter()
            .filter(|result| result.outcome == "no_win_after_20_total_generations")
            .count(),
        total_qualifying_genomes: results
            .iter()
            .map(|result| result.qualifying_descendants.len())
            .sum(),
        total_logical_evaluations: results
            .iter()
            .map(|result| result.logical_evaluations)
            .sum(),
        total_fresh_evaluations: results.iter().map(|result| result.fresh_evaluations).sum(),
        results,
        artifacts: collect_artifacts_excluding(root, EXTENSION_SUMMARY_FILE)?,
    })
}

pub(crate) fn run_extension(registry: &Path, base_run_id: &str, resume: bool) -> Result<()> {
    validate_run_id(base_run_id)?;
    validate_panel(registry, base_run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let binary_sha256 = current_binary_sha256()?;
    let root = extension_root(base_run_id);
    let manifest_path = root.join(EXTENSION_MANIFEST_FILE);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("extension already exists; pass --resume");
        }
        let recorded: ExtensionManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        let expected = expected_extension_manifest(
            base_run_id,
            &recorded.extension_binary_sha256,
            &specs,
            recorded.extension_started_at_unix_ms,
        )?;
        if recorded != expected || recorded.extension_binary_sha256 != binary_sha256 {
            bail!("extension identity, base artifacts, cohort, or binary changed");
        }
        recorded
    } else {
        if root.exists() && !resume {
            bail!("extension run directory already exists");
        }
        fs::create_dir_all(root.join("datasets"))?;
        let manifest =
            expected_extension_manifest(base_run_id, &binary_sha256, &specs, unix_millis())?;
        immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    if root.join(EXTENSION_SUMMARY_FILE).is_file() {
        return validate_extension(registry, base_run_id);
    }
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let mut results = Vec::with_capacity(manifest.datasets.len());
    for (index, binding) in manifest.datasets.iter().enumerate() {
        let spec = by_hash
            .get(binding.dataset_hash.as_str())
            .context("extension dataset is absent from panel specs")?;
        eprintln!(
            "extension dataset {}/{} {}",
            index + 1,
            manifest.datasets.len(),
            binding.dataset_hash
        );
        results.push(run_extension_dataset(
            registry, &manifest, binding, spec, resume,
        )?);
    }
    let summary = make_extension_summary(&manifest, results, &root)?;
    immutable_write(
        &root.join(EXTENSION_SUMMARY_FILE),
        &canonical_json(&summary)?,
    )?;
    validate_extension(registry, base_run_id)
}

fn validate_extension_dataset(
    registry: &Path,
    manifest: &ExtensionManifest,
    binding: &ExtensionDatasetBinding,
    spec: &DatasetSpec,
) -> Result<ExtensionDatasetResult> {
    let root = extension_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let recorded_identity: CampaignIdentity =
        serde_json::from_slice(&fs::read(root.join("campaign.v1.json"))?)?;
    let expected_identity = extension_identity(
        &manifest.extension_run_id,
        &manifest.extension_binary_sha256,
        spec,
    );
    if recorded_identity != expected_identity
        || fs::read(root.join("base-binding.v1.json"))? != canonical_json(binding)?
    {
        bail!("extension dataset {} identity mismatch", spec.dataset_hash);
    }
    validate_staging_receipt(&root, &recorded_identity)?;
    let (_, test) = read_campaign_data(registry, spec)?;
    let base_root = run_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let base_identity: CampaignIdentity =
        serde_json::from_slice(&fs::read(base_root.join("campaign.v1.json"))?)?;
    if base_identity.binary_sha256 != binding.base_binary_sha256
        || sha256_file(&base_root.join("campaign.v1.json"))? != binding.base_campaign_sha256
        || sha256_file(&base_root.join("final-result.v1.json"))? != binding.base_final_result_sha256
    {
        bail!(
            "extension dataset {} base binding mismatch",
            spec.dataset_hash
        );
    }
    let base_checkpoints = load_checkpoints(&base_root, &base_identity)?;
    let start_population = generation_10_population(&base_checkpoints, spec)?;
    if sha256_bytes(&canonical_json(&start_population)?) != binding.generation_10_population_sha256
        || base_checkpoints.last().unwrap().1 != binding.base_terminal_checkpoint_sha256
    {
        bail!(
            "extension dataset {} start state mismatch",
            spec.dataset_hash
        );
    }
    let checkpoints = load_extension_checkpoints(
        &root,
        &recorded_identity,
        &binding.base_terminal_checkpoint_sha256,
    )?;
    if checkpoints.len() != EXTENSION_GENERATIONS {
        bail!(
            "extension dataset {} has incomplete checkpoints",
            spec.dataset_hash
        );
    }

    let mut expected_population = start_population;
    let mut expected_previous = binding.base_terminal_checkpoint_sha256.clone();
    let mut expected_cumulative = 0;
    let g0_rmse = base_checkpoints[0].0.g0_best_rmse;
    for (offset, (checkpoint, checkpoint_sha256)) in checkpoints.iter().enumerate() {
        let generation = EXTENSION_START_GENERATION + offset;
        if checkpoint.previous_checkpoint_sha256.as_deref() != Some(&expected_previous)
            || checkpoint.slots.len() != POPULATION
            || checkpoint.slots.iter().enumerate().any(|(slot, outcome)| {
                outcome.generation != generation
                    || outcome.slot != slot
                    || outcome.member != expected_population[slot]
            })
        {
            bail!(
                "extension dataset {} population mismatch",
                spec.dataset_hash
            );
        }
        let mut fresh = 0;
        for outcome in &checkpoint.slots {
            let canonical =
                canonicalize_for_features(&outcome.member.canonical_genome, spec.features)?;
            if canonical.canonical != outcome.member.canonical_genome
                || canonical.semantic_sha256 != outcome.member.semantic_sha256
                || canonical.complexity != outcome.member.complexity
                || canonical.rust_source_len != outcome.member.rust_source_len
            {
                bail!("extension dataset {} genome mismatch", spec.dataset_hash);
            }
            let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
            if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
                bail!(
                    "extension dataset {} receipt hash mismatch",
                    spec.dataset_hash
                );
            }
            let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
            let request_path = root
                .join("requests")
                .join(&receipt.purpose)
                .join(format!("generation-{:03}", receipt.generation))
                .join(format!(
                    "slot-{:03}-attempt-{}.v1.json",
                    receipt.slot, receipt.attempt
                ));
            let request: CandidateRequest = serde_json::from_slice(&fs::read(request_path)?)?;
            validate_receipt_against_request(&receipt, &request, &test)?;
            if receipt.run_id != manifest.extension_run_id
                || receipt.dataset_hash != spec.dataset_hash
                || receipt.binary_sha256 != manifest.extension_binary_sha256
                || request.binary_sha256 != manifest.extension_binary_sha256
                || receipt.semantic_sha256 != outcome.member.semantic_sha256
                || outcome.valid != receipt.valid(spec.test_rows)
                || outcome.metrics != receipt.metrics
                || outcome.failure != receipt.failure
            {
                bail!(
                    "extension dataset {} receipt provenance mismatch",
                    spec.dataset_hash
                );
            }
            if let Some(source) = &outcome.cache_source {
                if source.generation < EXTENSION_START_GENERATION
                    || source.generation > generation
                    || (source.generation == generation && source.slot >= outcome.slot)
                {
                    bail!(
                        "extension dataset {} cache points outside extension history",
                        spec.dataset_hash
                    );
                }
                let source_outcome = &checkpoints[source.generation - EXTENSION_START_GENERATION]
                    .0
                    .slots[source.slot];
                if source_outcome.member.semantic_sha256 != outcome.member.semantic_sha256
                    || source_outcome.receipt_file != outcome.receipt_file
                    || source_outcome.receipt_sha256 != outcome.receipt_sha256
                {
                    bail!(
                        "extension dataset {} cache source mismatch",
                        spec.dataset_hash
                    );
                }
            } else {
                fresh += 1;
            }
        }
        let valid = checkpoint
            .slots
            .iter()
            .filter(|outcome| outcome.valid)
            .count();
        expected_cumulative += fresh;
        if valid != checkpoint.valid_candidates
            || valid < MIN_VALID
            || fresh != checkpoint.fresh_evaluations
            || expected_cumulative != checkpoint.cumulative_fresh_evaluations
        {
            bail!(
                "extension dataset {} evaluation count mismatch",
                spec.dataset_hash
            );
        }
        let mut ranked = checkpoint.slots.clone();
        ranked.sort_by(rank_outcomes);
        if checkpoint.ranked_slots != ranked.iter().map(|slot| slot.slot).collect::<Vec<_>>()
            || checkpoint.best_slot != ranked[0].slot
            || checkpoint.best_semantic_sha256 != ranked[0].member.semantic_sha256
            || checkpoint.best_rmse.to_bits() != ranked[0].rmse().to_bits()
            || checkpoint.g0_best_rmse.to_bits() != g0_rmse.to_bits()
        {
            bail!("extension dataset {} ranking mismatch", spec.dataset_hash);
        }
        let qualifies = ranked
            .iter()
            .any(|slot| qualifying_descendant(slot, generation, g0_rmse, spec.leader_rmse));
        if checkpoint.qualifying_descendant_win != qualifies
            || checkpoint.result_label != qualifies.then(|| "new_leader_found".to_string())
        {
            bail!(
                "extension dataset {} qualification mismatch",
                spec.dataset_hash
            );
        }
        let expected_next = if generation + 1 < EXTENSION_END_GENERATION {
            next_population(&ranked, generation + 1, spec.features)?
        } else {
            Vec::new()
        };
        if checkpoint.next_population != expected_next {
            bail!("extension dataset {} evolution mismatch", spec.dataset_hash);
        }
        expected_population = expected_next;
        expected_previous = checkpoint_sha256.clone();
    }

    let result_bytes = fs::read(root.join("final-result.v1.json"))?;
    let result: ExtensionDatasetResult = serde_json::from_slice(&result_bytes)?;
    let combined = combined_checkpoints(&base_checkpoints, &checkpoints)?;
    let qualifying_descendants = extension_genome_records(
        &root,
        &combined,
        &checkpoints,
        spec,
        &manifest.extension_binary_sha256,
    )?;
    let terminal = &checkpoints.last().unwrap().0;
    let selected = terminal
        .slots
        .iter()
        .find(|slot| slot.slot == terminal.best_slot)
        .context("extension final selected slot is absent")?;
    let final_selected_genome = genome_record(
        &root,
        &combined,
        selected,
        g0_rmse,
        spec.leader_rmse,
        &manifest.extension_binary_sha256,
    )?;
    let first_qualifying_generation = qualifying_descendants
        .iter()
        .map(|record| record.generation)
        .min();
    let expected_outcome = if first_qualifying_generation.is_some() {
        "new_leader_found"
    } else {
        "no_win_after_20_total_generations"
    };
    if result.schema != EXTENSION_RESULT_SCHEMA
        || result.benchmark_valid
        || result.base_run_id != manifest.base_run_id
        || result.extension_run_id != manifest.extension_run_id
        || result.dataset_hash != spec.dataset_hash
        || result.outcome != expected_outcome
        || result.start_generation != EXTENSION_START_GENERATION
        || result.stopping_generation + 1 != EXTENSION_END_GENERATION
        || result.extension_generations_evaluated != EXTENSION_GENERATIONS
        || result.total_generations_evaluated != EXTENSION_END_GENERATION
        || result.leader_models != spec.leader_models
        || result.leader_rmse.to_bits() != spec.leader_rmse.to_bits()
        || result.fresh_g0_rmse.to_bits() != g0_rmse.to_bits()
        || result.first_qualifying_generation != first_qualifying_generation
        || result.final_selected_genome != final_selected_genome
        || result.qualifying_descendants != qualifying_descendants
        || result.logical_evaluations != EXTENSION_GENERATIONS * POPULATION
        || result.fresh_evaluations != expected_cumulative
        || result.extension_binary_sha256 != manifest.extension_binary_sha256
        || result.finished_at_unix_ms < result.started_at_unix_ms
    {
        bail!(
            "extension dataset {} final result mismatch",
            spec.dataset_hash
        );
    }
    if result.replay.len() != 3 {
        bail!(
            "extension dataset {} replay count mismatch",
            spec.dataset_hash
        );
    }
    let mut replay_receipts = Vec::new();
    for (slot, replay) in result.replay.iter().enumerate() {
        let bytes = fs::read(root.join(&replay.receipt_file))?;
        if sha256_bytes(&bytes) != replay.receipt_sha256 {
            bail!(
                "extension dataset {} replay hash mismatch",
                spec.dataset_hash
            );
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
        let request_path = root
            .join("requests/replay-final")
            .join(format!("generation-{:03}", EXTENSION_END_GENERATION - 1))
            .join(format!(
                "slot-{slot:03}-attempt-{}.v1.json",
                receipt.attempt
            ));
        let request: CandidateRequest = serde_json::from_slice(&fs::read(request_path)?)?;
        validate_receipt_against_request(&receipt, &request, &test)?;
        if receipt.semantic_sha256 != result.final_selected_genome.semantic_sha256
            || receipt.binary_sha256 != manifest.extension_binary_sha256
        {
            bail!(
                "extension dataset {} replay genome mismatch",
                spec.dataset_hash
            );
        }
        replay_receipts.push(receipt);
    }
    let replay_base = &replay_receipts[0];
    let base_rmse = replay_base
        .metrics
        .as_ref()
        .context("failed extension replay")?
        .rmse;
    for (index, receipt) in replay_receipts.iter().enumerate() {
        let replay_rmse = receipt
            .metrics
            .as_ref()
            .context("failed extension replay")?
            .rmse;
        let maximum_difference = replay_base
            .predictions
            .iter()
            .zip(&receipt.predictions)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        if (replay_rmse - base_rmse).abs() > 1e-12
            || maximum_difference > 1e-10
            || result.replay[index].rmse.to_bits() != replay_rmse.to_bits()
            || result.replay[index].maximum_prediction_difference.to_bits()
                != maximum_difference.to_bits()
        {
            bail!(
                "extension dataset {} replay disagreement",
                spec.dataset_hash
            );
        }
    }
    Ok(result)
}

pub(crate) fn validate_extension(registry: &Path, base_run_id: &str) -> Result<()> {
    validate_run_id(base_run_id)?;
    validate_panel(registry, base_run_id)?;
    let root = extension_root(base_run_id);
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let manifest_bytes = fs::read(root.join(EXTENSION_MANIFEST_FILE))?;
    let manifest: ExtensionManifest = serde_json::from_slice(&manifest_bytes)?;
    let expected_manifest = expected_extension_manifest(
        base_run_id,
        &manifest.extension_binary_sha256,
        &specs,
        manifest.extension_started_at_unix_ms,
    )?;
    if manifest != expected_manifest || manifest_bytes != canonical_json(&expected_manifest)? {
        bail!("extension manifest does not reconstruct from the immutable base campaign");
    }
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let mut results = Vec::with_capacity(manifest.datasets.len());
    for binding in &manifest.datasets {
        let spec = by_hash
            .get(binding.dataset_hash.as_str())
            .context("extension dataset is absent from panel specs")?;
        results.push(validate_extension_dataset(
            registry, &manifest, binding, spec,
        )?);
    }
    for pair in results.windows(2) {
        if pair[0].finished_at_unix_ms > pair[1].started_at_unix_ms {
            bail!("extension datasets overlap in execution time");
        }
    }
    let summary_bytes = fs::read(root.join(EXTENSION_SUMMARY_FILE))?;
    let recorded_summary: ExtensionSummary = serde_json::from_slice(&summary_bytes)?;
    let expected_summary = make_extension_summary(&manifest, results, &root)?;
    if recorded_summary != expected_summary || summary_bytes != canonical_json(&expected_summary)? {
        bail!("extension summary or artifact hashes do not reconstruct");
    }
    eprintln!(
        "validated extension for {} datasets: {} new leaders, {} still no leader, {} qualifying genomes, {} logical evaluations, {} fresh evaluations; summary {}",
        recorded_summary.dataset_count,
        recorded_summary.new_leaders,
        recorded_summary.still_no_leader,
        recorded_summary.total_qualifying_genomes,
        recorded_summary.total_logical_evaluations,
        recorded_summary.total_fresh_evaluations,
        sha256_bytes(&summary_bytes),
    );
    Ok(())
}

pub(crate) fn validate(registry: &Path, run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let spec = specs
        .iter()
        .find(|spec| spec.dataset_hash == LEGACY_DATASET)
        .context("legacy smoke dataset is absent from Golden42")?;
    validate_dataset(
        registry,
        run_id,
        &run_root(run_id),
        spec,
        &current_binary_sha256()?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEST_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let counter = TEST_DIRECTORY_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "jope-dime-{label}-{}-{counter}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn spec(dataset_hash: &str) -> DatasetSpec {
        DatasetSpec {
            dataset_hash: dataset_hash.to_string(),
            train_sha256: "train".to_string(),
            test_sha256: "test".to_string(),
            train_rows: 1,
            test_rows: 1,
            features: 1,
            leader_models: vec!["leader".to_string()],
            leader_rmse: 0.9,
        }
    }

    fn manifest(dataset_ids: &[&str], binary_sha256: &str) -> PanelManifest {
        let ordered_dataset_ids = dataset_ids
            .iter()
            .map(|dataset| (*dataset).to_string())
            .collect::<Vec<_>>();
        PanelManifest {
            schema: PANEL_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: "run".to_string(),
            ordered_cohort_sha256: ordered_cohort_sha256(&ordered_dataset_ids),
            ordered_dataset_ids,
            panel_file: "panel".to_string(),
            panel_file_sha256: "panel-hash".to_string(),
            leader_manifest: "leaders".to_string(),
            leader_manifest_sha256: "leader-hash".to_string(),
            release_binary_sha256: binary_sha256.to_string(),
            registry_root: "registry".to_string(),
            panel_declared_dataset_root: "registry".to_string(),
            execution_host: EXECUTION_HOST.to_string(),
            execution_node: EXECUTION_NODE.to_string(),
            config: PanelConfig::fixed(),
            datasets: dataset_ids.iter().map(|dataset| spec(dataset)).collect(),
        }
    }

    fn seed(slot: usize, semantic_sha256: &str, source_kind: &str) -> SeedProvenance {
        SeedProvenance {
            slot,
            canonical_genome: format!("genome-{semantic_sha256}"),
            semantic_sha256: semantic_sha256.to_string(),
            complexity: slot,
            rust_source_len: slot + 1,
            source_kind: source_kind.to_string(),
            source_file: "source".to_string(),
            source_file_sha256: "source-hash".to_string(),
            source_record: slot,
            source_rank: slot,
            source_score: None,
        }
    }

    fn continuation(
        panel: &PanelManifest,
        boundary: usize,
        artifact_count: usize,
        artifacts_sha256: String,
    ) -> ContinuationRecord {
        ContinuationRecord {
            schema: CONTINUATION_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: panel.run_id.clone(),
            original_panel_manifest_sha256: "manifest-hash".to_string(),
            original_binary_sha256: panel.release_binary_sha256.clone(),
            corrected_binary_sha256: "corrected".to_string(),
            completed_dataset_count: boundary,
            first_remaining_dataset_index: boundary,
            first_remaining_dataset_ordinal: boundary + 1,
            completed_dataset_ids: panel.ordered_dataset_ids[..boundary].to_vec(),
            remaining_dataset_ids: panel.ordered_dataset_ids[boundary..].to_vec(),
            completed_artifact_count: artifact_count,
            completed_artifacts_sha256: artifacts_sha256,
            seed_fix_reason: FAST30_FIX_REASON.to_string(),
            continuation_started_at_unix_ms: 1,
        }
    }

    #[test]
    fn fast30_champion_is_forced_into_slot_59_with_exact_provenance() {
        let mut seeds = (0..POPULATION)
            .map(|slot| seed(slot, &format!("seed-{slot}"), "ranked"))
            .collect::<Vec<_>>();
        let champion = seed(7, "fast30", "dataset_fast30_champion");
        assert!(force_fast30_inclusion(&mut seeds, champion.clone()));
        assert_eq!(seeds.len(), POPULATION);
        assert_eq!(
            seeds[POPULATION - 1],
            SeedProvenance {
                slot: POPULATION - 1,
                ..champion
            }
        );
        assert!(!force_fast30_inclusion(
            &mut seeds,
            seed(0, "fast30", "different-provenance")
        ));
        assert_eq!(seeds[POPULATION - 1].source_kind, "dataset_fast30_champion");
    }

    #[test]
    fn mixed_binary_continuation_has_an_exact_boundary() {
        let panel = manifest(&["a", "b", "c", "d"], "original");
        let record = continuation(&panel, 2, 0, sha256_bytes(&[]));
        validate_continuation_shape(&panel, "manifest-hash", &record).unwrap();
        let segments = binary_segments(
            &panel.ordered_dataset_ids,
            record.completed_dataset_count,
            &record.original_binary_sha256,
            &record.corrected_binary_sha256,
        );
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].dataset_ids, vec!["a", "b"]);
        assert_eq!(segments[1].dataset_ids, vec!["c", "d"]);
        assert!(validate_continuation_shape(&panel, "wrong-manifest-hash", &record).is_err());
        let mut shifted = record;
        shifted.completed_dataset_count = 3;
        assert!(validate_continuation_shape(&panel, "manifest-hash", &shifted).is_err());
    }

    #[test]
    fn corrected_binary_is_refused_for_a_partially_started_dataset() {
        let directory = TestDirectory::new("partial-boundary");
        fs::create_dir_all(directory.0.join("datasets/a")).unwrap();
        fs::create_dir_all(directory.0.join("datasets/b")).unwrap();
        fs::write(
            directory.0.join("datasets/a/final-result.v1.json"),
            b"terminal\n",
        )
        .unwrap();
        let datasets = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(
            clean_continuation_boundary(&directory.0, &datasets).unwrap(),
            1
        );
        fs::write(
            directory.0.join("datasets/b/campaign.v1.json"),
            b"partially-started\n",
        )
        .unwrap();
        assert!(clean_continuation_boundary(&directory.0, &datasets).is_err());
    }

    #[test]
    fn completed_dataset_artifacts_are_immutable_across_continuation() {
        let directory = TestDirectory::new("immutable-completed");
        let panel = manifest(&["a", "b"], "original");
        let dataset_root = directory.0.join("datasets/a");
        fs::create_dir_all(&dataset_root).unwrap();
        let campaign = identity("run", "original".to_string(), &spec("a"));
        fs::write(
            dataset_root.join("campaign.v1.json"),
            canonical_json(&campaign).unwrap(),
        )
        .unwrap();
        fs::write(dataset_root.join("final-result.v1.json"), b"terminal\n").unwrap();
        let completed = vec!["a".to_string()];
        let (count, fingerprint) =
            completed_artifact_fingerprint(&directory.0, &completed).unwrap();
        let record = continuation(&panel, 1, count, fingerprint);
        validate_continuation_artifacts(&directory.0, &panel, &record).unwrap();
        fs::write(dataset_root.join("final-result.v1.json"), b"changed\n").unwrap();
        assert!(validate_continuation_artifacts(&directory.0, &panel, &record).is_err());
    }

    #[test]
    fn dataset_identity_is_validated_against_its_own_binary_segment() {
        let old = identity("run", "original".to_string(), &spec("a"));
        let new = identity("run", "corrected".to_string(), &spec("b"));
        validate_dataset_binary(&old, "original").unwrap();
        validate_dataset_binary(&new, "corrected").unwrap();
        assert!(validate_dataset_binary(&old, "corrected").is_err());
        assert!(validate_dataset_binary(&new, "original").is_err());
    }

    fn member(hash: &str, evolved_descendant: bool) -> Member {
        Member {
            canonical_genome: format!("canonical-{hash}"),
            semantic_sha256: hash.to_string(),
            complexity: 3,
            rust_source_len: 17,
            evolved_descendant,
            ancestry: Ancestry {
                operation: if evolved_descendant {
                    "mutation"
                } else {
                    "seed"
                }
                .to_string(),
                parents: Vec::new(),
            },
        }
    }

    fn checkpoint(
        generation: usize,
        g0_best_rmse: f64,
        slots: Vec<SlotOutcome>,
    ) -> GenerationCheckpoint {
        GenerationCheckpoint {
            schema: CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: "run".to_string(),
            dataset_hash: "a".to_string(),
            generation,
            previous_checkpoint_sha256: None,
            logical_slots: slots.len(),
            fresh_evaluations: slots.len(),
            cumulative_fresh_evaluations: slots.len(),
            valid_candidates: slots.len(),
            g0_best_rmse,
            best_slot: 0,
            best_semantic_sha256: slots
                .first()
                .map_or_else(String::new, |slot| slot.member.semantic_sha256.clone()),
            best_rmse: slots.first().map_or(f64::INFINITY, SlotOutcome::rmse),
            qualifying_descendant_win: generation > 0,
            result_label: (generation > 0).then(|| "early_win".to_string()),
            ranked_slots: slots.iter().map(|slot| slot.slot).collect(),
            slots,
            next_population: Vec::new(),
        }
    }

    fn qualifying_outcome(
        root: &Path,
        generation: usize,
        slot: usize,
        hash: &str,
        rmse: f64,
    ) -> SlotOutcome {
        let candidate = member(hash, true);
        let candidate_metrics = CandidateMetrics {
            rmse,
            mae: rmse / 2.0,
            r2: 1.0 - rmse,
        };
        let predictions = vec![rmse];
        let receipt = CandidateReceipt {
            schema: RECEIPT_SCHEMA.to_string(),
            benchmark_valid: false,
            test_adaptive: None,
            purpose: "candidate".to_string(),
            run_id: "run".to_string(),
            generation,
            slot,
            attempt: 0,
            semantic_sha256: hash.to_string(),
            canonical_genome: candidate.canonical_genome.clone(),
            binary_sha256: "corrected".to_string(),
            dataset_hash: "a".to_string(),
            train_sha256: "train".to_string(),
            test_sha256: "test".to_string(),
            train_rows: 1,
            test_rows: 1,
            features: 1,
            execution_host: EXECUTION_HOST.to_string(),
            node: EXECUTION_NODE.to_string(),
            scq_job_id: format!("job-{slot}"),
            runtime_seconds: slot as f64 + 0.25,
            thread_counts: ThreadCounts::single(),
            predictions_sha256: sha256_bytes(&prediction_bytes(&predictions)),
            predictions,
            metrics: Some(candidate_metrics.clone()),
            failure: None,
        };
        let relative = format!("receipt-{slot}.json");
        let bytes = canonical_json(&receipt).unwrap();
        fs::write(root.join(&relative), &bytes).unwrap();
        SlotOutcome {
            generation,
            slot,
            member: candidate,
            valid: true,
            metrics: Some(candidate_metrics),
            failure: None,
            cache_source: None,
            receipt_file: relative,
            receipt_sha256: sha256_bytes(&bytes),
            node: EXECUTION_NODE.to_string(),
            scq_job_id: format!("job-{slot}"),
            attempt: 0,
        }
    }

    #[test]
    fn every_qualifying_genome_reconstructs_not_only_the_selected_winner() {
        let directory = TestDirectory::new("qualifying-genomes");
        let checkpoints = vec![
            (checkpoint(0, 1.0, Vec::new()), "g0".to_string()),
            (
                checkpoint(
                    1,
                    1.0,
                    vec![
                        qualifying_outcome(&directory.0, 1, 0, "winner", 0.3),
                        qualifying_outcome(&directory.0, 1, 1, "also-qualifies", 0.4),
                    ],
                ),
                "g1".to_string(),
            ),
        ];
        let records =
            collect_qualifying_genomes(&directory.0, &checkpoints, &spec("a"), "corrected")
                .unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].semantic_sha256, "winner");
        assert_eq!(records[1].semantic_sha256, "also-qualifies");
        assert_eq!(records[0].generation, 1);
        assert_eq!(records[1].slot, 1);
        assert_eq!(
            records[1].receipt_sha256,
            checkpoints[1].0.slots[1].receipt_sha256
        );
    }

    #[test]
    fn confirmed_winner_seed_mix_keeps_strongest_per_dataset_and_target_first() {
        let directory = TestDirectory::new("confirmed-winner-seeds");
        let write_result =
            |run: &str, dataset: &str, genome: &str, best_rmse: f64, leader_rmse: f64| {
                let path = directory
                    .0
                    .join(run)
                    .join("datasets")
                    .join(dataset)
                    .join("final-result.v1.json");
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                let confirmation_path = path.parent().unwrap().join("confirmation.v1.json");
                let checkpoint_path = path
                    .parent()
                    .unwrap()
                    .join("checkpoints/generation-000.v1.json");
                fs::create_dir_all(checkpoint_path.parent().unwrap()).unwrap();
                fs::write(&confirmation_path, b"confirmation\n").unwrap();
                fs::write(&checkpoint_path, b"checkpoint\n").unwrap();
                let result = ConfirmedGoldenTransferResult {
                    schema: "JopeGolden42AggressiveDatasetResult.v1".to_string(),
                    benchmark_valid: false,
                    test_adaptive: true,
                    run_id: run.to_string(),
                    dataset_hash: dataset.to_string(),
                    outcome: "confirmed_strict_win".to_string(),
                    stopping_generation: 0,
                    leader_rmse,
                    best_rmse,
                    best_slot: 0,
                    canonical_genome: genome.to_string(),
                    semantic_sha256: sha256_bytes(genome.as_bytes()),
                    terminal_checkpoint_sha256: sha256_file(&checkpoint_path).unwrap(),
                    confirmation_sha256: Some(sha256_file(&confirmation_path).unwrap()),
                };
                fs::write(path, canonical_json(&result).unwrap()).unwrap();
            };
        write_result("golden42-all-regression-old", "a", "genome-a-old", 0.8, 0.9);
        write_result(
            "golden42-all-regression-new",
            "a",
            "genome-a-best",
            0.7,
            0.9,
        );
        write_result("golden42-all-regression-new", "b", "genome-b", 0.5, 0.8);

        let sources = confirmed_campaign_winner_sources_at(&directory.0, "b").unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].genome, "genome-b");
        assert_eq!(sources[0].rank, 1);
        assert_eq!(sources[1].genome, "genome-a-best");
        assert_eq!(sources[1].rank, 2);
        assert!(sources
            .iter()
            .all(|source| source.kind == "confirmed_multi_dataset_campaign_winner"));

        let transfer = confirmed_transfer_winner_sources_at(&directory.0, "b").unwrap();
        assert_eq!(transfer.len(), 1);
        assert_eq!(transfer[0].genome, "genome-a-best");
        assert_eq!(transfer[0].rank, 1);
    }
}
