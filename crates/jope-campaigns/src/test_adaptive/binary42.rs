//! Frozen Binary42 held-out-test-adaptive campaigns.
//!
//! This lane is intentionally benchmark-invalid: held-out test ROC-AUC is the
//! evolutionary fitness. Every persisted record carries both
//! `test_adaptive: true` and `benchmark_valid: false`.

use super::*;
use crate::commands::binary_cv::{
    binary_genome_complexity, evaluate_test_adaptive_binary, evaluate_test_adaptive_binary_cpp17,
    test_adaptive_next_population, test_adaptive_seed_population, BinaryAdaptiveRanked,
    BinaryAncestry, BinaryMember, SEED_BINARY_SHA256,
};
use crate::data::read_binary_csv;
use feat_math::metrics::auc;
use jope_dime_regressor::paths;
use serde_json::json;
use std::collections::VecDeque;
use std::os::unix::fs::PermissionsExt;
use std::sync::{Arc, Mutex};

const COHORT_BYTES: &[u8] = include_bytes!("../../manifests/binary42-test-adaptive.v1.json");
const BINARY_SEEDS: &[u8] = include_bytes!("../../seeds/generation-zero-binary.txt");
jope_paths::once_str!(
    fn scq_binary_str => paths::scq_binary()
);
jope_paths::once_str!(
    fn all_scq_binary_str => paths::super_cluster_scq_binary()
);
jope_paths::once_str!(
    fn linux_remote_root_str => paths::legacy_tree("jope-test-adaptive")
);
const PROJECT: &str = "jope";
const ILPD: &str = "962a35a53b0803a8";
const NODES: [&str; 4] = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"];
const PHYSICAL_HOSTNAMES: [&str; 4] = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"];
const POPULATION: usize = 60;
const SHARD_SIZE: usize = 15;
const ALL_SHARD_SIZES: [usize; 4] = [19, 19, 19, 3];
const ALL_ACTIVE_DATASETS: usize = 6;
const ALL_MAX_GENERATIONS: usize = 10;
const CANDIDATE_STACK_BYTES: usize = 32 * 1024 * 1024;
const CPP17_BACKEND: &str = "fitted-cpp17-dynamic-v2";
const CPP17_STANDARD: &str = "C++17";
const CPP17_FLAGS: [&str; 5] = [
    "-std=c++17",
    "-O3",
    "-march=znver2",
    "-ffp-contract=off",
    "-fno-fast-math",
];
const ELITES: usize = 7;
const PARENT_POOL: usize = 30;
const TOURNAMENT: usize = 3;
const LOGLOSS_REPLAY_MAX_ULPS: u64 = 16;
const LOGLOSS_REPLAY_MAX_ABS: f64 = 1e-12;
const RNG_SEED: u64 = 42 ^ 0x6269_6e34_325f_696c;
const CAMPAIGN_SCHEMA: &str = "JopeBinary42TestAdaptiveCampaign.v1";
const BASELINE_SCHEMA: &str = "JopeBinary42TestAdaptiveBaseline.v1";
const SEED_SCHEMA: &str = "JopeBinary42TestAdaptiveSeedManifest.v1";
const STAGING_SCHEMA: &str = "JopeBinary42TestAdaptiveStaging.v1";
const SHARD_SCHEMA: &str = "JopeBinary42TestAdaptiveShard.v1";
const WORKER_SCHEMA: &str = "JopeBinary42TestAdaptiveWorkerReceipt.v1";
const CANDIDATE_SCHEMA: &str = "JopeBinary42TestAdaptiveCandidateReceipt.v1";
const GENERATION_SCHEMA: &str = "JopeBinary42TestAdaptiveGeneration.v1";
const CHECKPOINT_SCHEMA: &str = "JopeBinary42TestAdaptiveCheckpoint.v1";
const CACHE_SCHEMA: &str = "JopeBinary42TestAdaptiveCacheEntry.v1";
const GENOME_SCHEMA: &str = "JopeBinary42TestAdaptiveGenome.v1";
const CONFIRMATION_SCHEMA: &str = "JopeBinary42TestAdaptiveConfirmation.v1";
const WINNER_SCHEMA: &str = "JopeBinary42TestAdaptiveWinner.v1";
const COMPLETE_SCHEMA: &str = "JopeBinary42TestAdaptiveComplete.v1";
const EXHAUSTED_SCHEMA: &str = "JopeBinary42TestAdaptiveExhausted.v1";
const ALL_CAMPAIGN_SCHEMA: &str = "JopeBinary42AllCampaign.v1";
const ALL_STAGING_SCHEMA: &str = "JopeBinary42AllStaging.v1";
const ALL_SUMMARY_SCHEMA: &str = "JopeBinary42AllSummary.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenWinPolicy {
    non_saturated: String,
    saturated: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct FrozenDataset {
    order: usize,
    id: String,
    display_name: String,
    category: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    leader_method: String,
    leader_auc: f64,
    leader_report_path: String,
    leader_report_sha256: String,
    win_policy: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenCohort {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    source: String,
    track_id: String,
    cohort_registry_sha256: String,
    dataset_registry_sha256: String,
    scores_sha256: String,
    win_policy: FrozenWinPolicy,
    datasets: Vec<FrozenDataset>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CampaignConfig {
    population: usize,
    shards: usize,
    candidates_per_shard: usize,
    elites: usize,
    parent_pool: usize,
    tournament: usize,
    mutation_probability_numerator: usize,
    mutation_probability_denominator: usize,
    crossover_probability_numerator: usize,
    crossover_probability_denominator: usize,
    candidate_threads: usize,
    cpus_per_shard: usize,
    gpus_per_shard: usize,
    generation_cap: Option<usize>,
    infrastructure_retries: usize,
    confirmation_nodes: Vec<String>,
    ranking: String,
    win_rule: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    shard_sizes: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    execution_backend: Option<String>,
}

impl CampaignConfig {
    fn fixed() -> Self {
        Self {
            population: POPULATION,
            shards: NODES.len(),
            candidates_per_shard: SHARD_SIZE,
            elites: ELITES,
            parent_pool: PARENT_POOL,
            tournament: TOURNAMENT,
            mutation_probability_numerator: 3,
            mutation_probability_denominator: 4,
            crossover_probability_numerator: 1,
            crossover_probability_denominator: 4,
            candidate_threads: 1,
            cpus_per_shard: SHARD_SIZE,
            gpus_per_shard: 0,
            generation_cap: None,
            infrastructure_retries: 1,
            confirmation_nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            ranking: "test_auc descending, test_logloss ascending, complexity ascending, semantic_sha256 ascending".to_string(),
            win_rule: "evolved descendant with test_auc strictly greater than both frozen leader and generation-zero best".to_string(),
            shard_sizes: Vec::new(),
            execution_backend: None,
        }
    }

    fn all(spec: &FrozenDataset) -> Self {
        Self {
            population: POPULATION,
            shards: NODES.len(),
            candidates_per_shard: ALL_SHARD_SIZES[0],
            elites: ELITES,
            parent_pool: PARENT_POOL,
            tournament: TOURNAMENT,
            mutation_probability_numerator: 3,
            mutation_probability_denominator: 4,
            crossover_probability_numerator: 1,
            crossover_probability_denominator: 4,
            candidate_threads: 1,
            cpus_per_shard: ALL_SHARD_SIZES[0],
            gpus_per_shard: 0,
            generation_cap: Some(ALL_MAX_GENERATIONS),
            infrastructure_retries: 2,
            confirmation_nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            ranking: "test_auc descending, test_logloss ascending, complexity ascending, semantic_sha256 ascending".to_string(),
            win_rule: match spec.win_policy.as_str() {
                "exact_auc_1_tie" => {
                    "non-seed evolved descendant from generation >= 1 with exact test_auc 1.0"
                }
                _ => "non-seed evolved descendant from generation >= 1 with test_auc strictly greater than both frozen leader and generation-zero best",
            }
            .to_string(),
            shard_sizes: ALL_SHARD_SIZES.to_vec(),
            execution_backend: Some(CPP17_BACKEND.to_string()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CampaignManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    frozen_cohort_file: String,
    frozen_cohort_sha256: String,
    binary_seed_sha256: String,
    release_binary_sha256: String,
    scq_binary_sha256: String,
    dataset_binding: FrozenDataset,
    config: CampaignConfig,
    started_at_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cohort_staging_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    winner_seed_snapshot_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    dataset: String,
    leader_method: String,
    leader_auc: f64,
    leader_report_path: String,
    leader_report_sha256: String,
    generation_zero_best_auc: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct MemberRecord {
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    ancestry: BinaryAncestry,
}

impl MemberRecord {
    fn from_binary(member: BinaryMember) -> Result<Self> {
        Ok(Self {
            complexity: binary_genome_complexity(&member.canonical_genome)?,
            canonical_genome: member.canonical_genome,
            semantic_sha256: member.semantic_sha256,
            ancestry: member.ancestry,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct WinnerSeedSource {
    source_run_id: String,
    source_dataset: String,
    complete_file: String,
    complete_sha256: String,
    final_receipt_file: String,
    final_receipt_sha256: String,
    confirmation_file: String,
    confirmation_sha256: String,
    canonical_genome: String,
    semantic_sha256: String,
    auc_bits: u64,
    frozen_leader_auc_bits: u64,
    generation_zero_best_auc_bits: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    seed_file_sha256: String,
    population: usize,
    members: Vec<MemberRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    winner_seed_sources: Vec<WinnerSeedSource>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagedNode {
    node: String,
    hostname: String,
    bundle_root: String,
    binary_path: String,
    binary_sha256: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    canary_job_id: String,
    canary_show_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StagingReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    release_binary_sha256: String,
    scq_binary_sha256: String,
    doctor_output_sha256: String,
    dataset: String,
    nodes: Vec<StagedNode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cohort_staging_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CandidateScore {
    auc: f64,
    logloss: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheEntry {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    semantic_sha256: String,
    receipt_sha256: String,
    prediction_sha256: String,
    score: Option<CandidateScore>,
    failure: Option<String>,
    evaluation_node: String,
    scq_job_id: String,
    first_generation: usize,
    first_slot: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenomeObject {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    semantic_sha256: String,
    canonical_genome: String,
    complexity: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShardSlot {
    generation: usize,
    slot: usize,
    member: MemberRecord,
    fresh: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShardManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    kind: String,
    run_id: String,
    dataset: String,
    generation: usize,
    shard: usize,
    node: String,
    job_id: String,
    attempt: usize,
    release_binary_sha256: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    cpus_reserved: usize,
    gpus_per_task: usize,
    candidate_threads: usize,
    slots: Vec<ShardSlot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cohort_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scq_binary_path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    kind: String,
    run_id: String,
    dataset: String,
    generation: usize,
    slot: usize,
    attempt: usize,
    semantic_sha256: String,
    canonical_genome: String,
    complexity: usize,
    release_binary_sha256: String,
    train_sha256: String,
    test_sha256: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    node: String,
    hostname: String,
    scq_job_id: String,
    candidate_threads: usize,
    cpu: usize,
    runtime_seconds: f64,
    fit_seconds: Option<f64>,
    predict_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    backend: Option<CandidateBackendReceipt>,
    predictions_sha256: String,
    score: Option<CandidateScore>,
    failure: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateBackendReceipt {
    name: String,
    language_standard: String,
    compiler: String,
    compiler_flags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    codec_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compile_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerCandidate {
    generation: usize,
    slot: usize,
    semantic_sha256: String,
    receipt_file: String,
    receipt_sha256: String,
    predictions_file: String,
    predictions_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    kind: String,
    run_id: String,
    dataset: String,
    generation: usize,
    shard: usize,
    node: String,
    hostname: String,
    job_id: String,
    manifest_sha256: String,
    logical_slots: usize,
    fresh_candidates: usize,
    cpus_reserved: usize,
    candidate_threads: usize,
    candidates: Vec<WorkerCandidate>,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JobReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    job_id: String,
    node: String,
    attempt: usize,
    manifest_sha256: String,
    scq_show_sha256: String,
    worker_receipt_sha256: String,
    state: String,
    exit_code: i64,
}

#[derive(Clone, Debug)]
struct FreshOutput {
    receipt: CandidateReceipt,
    receipt_bytes: Vec<u8>,
    prediction_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LogicalSlot {
    generation: usize,
    slot: usize,
    assigned_node: String,
    member: MemberRecord,
    cache_disposition: String,
    cache_source_generation: Option<usize>,
    cache_source_slot: Option<usize>,
    evaluation_node: String,
    scq_job_id: String,
    receipt_sha256: String,
    prediction_sha256: String,
    score: Option<CandidateScore>,
    failure: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    generation: usize,
    logical_slots: usize,
    shard_size: usize,
    shard_nodes: Vec<String>,
    members: Vec<MemberRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    shard_sizes: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualifyingWinner {
    generation: usize,
    slot: usize,
    semantic_sha256: String,
    auc: f64,
    logloss: f64,
    complexity: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationCheckpoint {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    generation: usize,
    previous_checkpoint_sha256: Option<String>,
    logical_slots: usize,
    fresh_evaluations: usize,
    cache_reuses: usize,
    valid_scores: usize,
    generation_zero_best_auc: f64,
    best_slot: usize,
    best_auc: f64,
    slots: Vec<LogicalSlot>,
    ranked_semantics: Vec<String>,
    next_population: Vec<MemberRecord>,
    qualifying_winner: Option<QualifyingWinner>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AllCohortConfig {
    active_datasets: usize,
    population: usize,
    generation_numbers: Vec<usize>,
    shard_sizes: Vec<usize>,
    node_cpu_budget_per_active_set: BTreeMap<String, usize>,
    candidate_threads: usize,
    gpus_per_task: usize,
    infrastructure_failover_attempts: usize,
    admission_order: String,
    completion_policy: String,
    execution_backend: String,
    compiler_requirement: String,
    compiler_flags: Vec<String>,
}

impl AllCohortConfig {
    fn fixed() -> Self {
        Self {
            active_datasets: ALL_ACTIVE_DATASETS,
            population: POPULATION,
            generation_numbers: (0..ALL_MAX_GENERATIONS).collect(),
            shard_sizes: ALL_SHARD_SIZES.to_vec(),
            node_cpu_budget_per_active_set: NODES
                .iter()
                .zip(ALL_SHARD_SIZES)
                .map(|(node, shard)| ((*node).to_string(), shard * ALL_ACTIVE_DATASETS))
                .collect(),
            candidate_threads: 1,
            gpus_per_task: 0,
            infrastructure_failover_attempts: 2,
            admission_order:
                "train_rows * features descending, frozen manifest order ascending".to_string(),
            completion_policy:
                "work-conserving admission; each dataset stops after a completed winning barrier or generation 9 exhaustion"
                    .to_string(),
            execution_backend: CPP17_BACKEND.to_string(),
            compiler_requirement: "g++ 13.3".to_string(),
            compiler_flags: CPP17_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllCohortManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    frozen_cohort_file: String,
    frozen_cohort_sha256: String,
    release_binary_sha256: String,
    scq_configured_path: String,
    scq_resolved_path: String,
    scq_binary_sha256: String,
    config: AllCohortConfig,
    admission_dataset_ids: Vec<String>,
    winner_seed_sources: Vec<WinnerSeedSource>,
    started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AllStagedDataset {
    dataset: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AllStagedNode {
    node: String,
    hostname: String,
    logical_cpus: usize,
    cpp17_compiler: String,
    bundle_root: String,
    binary_path: String,
    binary_sha256: String,
    canary_job_id: String,
    canary_show_sha256: String,
    canary_cpus_reserved: usize,
    datasets: Vec<AllStagedDataset>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllStagingReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    release_binary_sha256: String,
    scq_configured_path: String,
    scq_resolved_path: String,
    scq_binary_sha256: String,
    doctor_output_sha256: String,
    nodes: Vec<AllStagedNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExhaustedReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    status: String,
    run_id: String,
    dataset: String,
    completed_generations: usize,
    logical_slots: usize,
    best_generation: usize,
    best_slot: usize,
    best_semantic_sha256: String,
    best_auc: f64,
    best_logloss: f64,
    best_prediction_sha256: String,
    best_receipt_sha256: String,
    lineage_sha256: String,
    release_binary_sha256: String,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExhaustedMarker {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    status: String,
    run_id: String,
    dataset: String,
    completed_generations: usize,
    logical_slots: usize,
    best_semantic_sha256: String,
    best_auc: f64,
    final_receipt_sha256: String,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetValidationSummary {
    dataset: String,
    display_name: String,
    frozen_order: usize,
    status: String,
    completed_generations: usize,
    logical_slots: usize,
    unique_genomes: usize,
    best_auc: Option<f64>,
    winner_generation: Option<usize>,
    winner_semantic_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllTerminalSummary {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    terminal_datasets: usize,
    winners: usize,
    exhausted: usize,
    logical_evaluations: usize,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationIndexRecord {
    test_adaptive: bool,
    benchmark_valid: bool,
    generation: usize,
    slot: usize,
    semantic_sha256: String,
    ancestry_operation: String,
    ancestry_parents: Vec<String>,
    cache_disposition: String,
    assigned_node: String,
    evaluation_node: String,
    scq_job_id: String,
    auc: Option<f64>,
    logloss: Option<f64>,
    complexity: usize,
    receipt_sha256: String,
    prediction_sha256: String,
    failure: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenomeIndexRecord {
    semantic_sha256: String,
    canonical_genome: String,
    complexity: usize,
    ancestry_operation: String,
    ancestry_parents: Vec<String>,
    first_generation: usize,
    first_slot: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmationRecord {
    node: String,
    job_id: String,
    receipt_sha256: String,
    prediction_sha256: String,
    auc: f64,
    logloss: f64,
    auc_bits: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmationManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    generation: usize,
    selected_slot: usize,
    semantic_sha256: String,
    attempt: usize,
    original_prediction_sha256: String,
    confirmations: Vec<ConfirmationRecord>,
    prediction_hashes_identical: bool,
    auc_bits_identical: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LineageRecord {
    semantic_sha256: String,
    canonical_genome: String,
    complexity: usize,
    ancestry: BinaryAncestry,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WinnerReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    generation: usize,
    slot: usize,
    semantic_sha256: String,
    canonical_genome: String,
    complexity: usize,
    auc: f64,
    logloss: f64,
    frozen_leader_auc: f64,
    generation_zero_best_auc: f64,
    margin_vs_frozen_leader: f64,
    margin_vs_generation_zero: f64,
    prediction_sha256: String,
    source_receipt_sha256: String,
    four_node_confirmation_sha256: String,
    lineage_sha256: String,
    release_binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WinnerArtifact {
    path: String,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WinnerArtifactManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    artifacts: Vec<WinnerArtifact>,
}
#[derive(Clone, Debug, Serialize)]
struct AllEvaluationIndexRecord {
    dataset: String,
    display_name: String,
    frozen_order: usize,
    #[serde(flatten)]
    evaluation: EvaluationIndexRecord,
}

mod campaign_setup;
use campaign_setup::*;
mod worker;
use worker::*;
mod staging;
use staging::*;
mod execution;
use execution::*;
mod generation;
use generation::*;
mod indexes;
use indexes::*;
mod confirmation;
use confirmation::*;
mod publication;
use publication::*;
mod campaign;
use campaign::*;
mod validation;
use validation::*;

#[cfg(test)]
mod tests;

pub(crate) fn batch_worker(manifest_path: &Path, output_dir: &Path) -> Result<()> {
    worker::batch_worker_impl(manifest_path, output_dir)
}

pub(crate) fn run(run_id: &str, dataset: &str, resume: bool) -> Result<()> {
    campaign::run_impl(run_id, dataset, resume)
}

pub(crate) fn run_all(run_id: &str, resume: bool) -> Result<()> {
    campaign::run_all_impl(run_id, resume)
}

pub(crate) fn validate_all(run_id: &str) -> Result<()> {
    validation::validate_all_impl(run_id)
}

pub(crate) fn validate(run_id: &str, dataset: &str) -> Result<()> {
    validation::validate_impl(run_id, dataset)
}
