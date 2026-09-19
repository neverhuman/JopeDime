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
use crate::paths;
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

fn campaign_root(run_id: &str, dataset: &str) -> PathBuf {
    run_root(run_id).join(dataset)
}

fn generation_root(root: &Path, generation: usize) -> PathBuf {
    root.join("generations")
        .join(format!("generation-{generation:06}"))
}

fn immutable_or_equal(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        if fs::read(path)? != bytes {
            bail!("immutable artifact collision at {}", path.display());
        }
        return Ok(());
    }
    immutable_write(path, bytes)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<String> {
    let bytes = canonical_json(value)?;
    immutable_or_equal(path, &bytes)?;
    Ok(sha256_bytes(&bytes))
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path)?).with_context(|| format!("parse {}", path.display()))
}

fn read_cohort() -> Result<(FrozenCohort, String)> {
    let manifest: FrozenCohort =
        serde_json::from_slice(COHORT_BYTES).context("parse frozen Binary42 cohort")?;
    if manifest.schema != "JopeBinary42FrozenCohort.v1"
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.track_id != "tabular-binary-v1"
        || manifest.datasets.len() != 42
        || manifest
            .datasets
            .iter()
            .enumerate()
            .any(|(order, dataset)| dataset.order != order)
        || manifest
            .datasets
            .iter()
            .filter(|dataset| dataset.leader_auc == 1.0)
            .count()
            != 6
        || manifest.datasets.iter().any(|dataset| {
            if dataset.leader_auc.to_bits() == 1.0f64.to_bits() {
                dataset.win_policy != "exact_auc_1_tie"
            } else {
                dataset.win_policy != "strictly_greater_auc"
            }
        })
    {
        bail!("frozen Binary42 cohort contract mismatch");
    }
    let ids = manifest
        .datasets
        .iter()
        .map(|dataset| dataset.id.as_str())
        .collect::<BTreeSet<_>>();
    if ids.len() != 42 {
        bail!("frozen Binary42 cohort contains duplicate datasets");
    }
    let ilpd = manifest
        .datasets
        .iter()
        .find(|dataset| dataset.id == ILPD)
        .context("frozen Binary42 cohort omits ILPD")?;
    if ilpd.train_rows != 485
        || ilpd.test_rows != 98
        || ilpd.features != 10
        || ilpd.train_sha256 != "f2a030306747738f28cb60c9f2f3fe1591724427ee63cb9d0bf8fb43b75cc114"
        || ilpd.test_sha256 != "0b86467f1692851213f7adbf6194e91c2eda39ff3e340a4ef81805b53e388aac"
        || ilpd.leader_method != "pytorch_tabular_fttransformer_epochs5"
        || ilpd.leader_auc.to_bits() != 0.7681159420289855f64.to_bits()
        || ilpd.leader_report_sha256
            != "5018f76223d4445f87960a076c9766bfe4140bb43d8b8bf9ae68ebb5ac85bad4"
    {
        bail!("frozen ILPD binding differs from the pilot contract");
    }
    Ok((manifest, sha256_bytes(COHORT_BYTES)))
}

fn selected_dataset(dataset: &str) -> Result<(FrozenDataset, String)> {
    if dataset != ILPD {
        bail!("the Binary42 pilot is frozen to ILPD ({ILPD}); no other dataset may launch");
    }
    let (cohort, sha256) = read_cohort()?;
    Ok((
        cohort
            .datasets
            .into_iter()
            .find(|record| record.id == dataset)
            .context("selected dataset is absent from Binary42")?,
        sha256,
    ))
}

fn cohort_dataset(dataset: &str) -> Result<(FrozenDataset, String)> {
    let (cohort, sha256) = read_cohort()?;
    Ok((
        cohort
            .datasets
            .into_iter()
            .find(|record| record.id == dataset)
            .with_context(|| format!("dataset {dataset} is absent from frozen Binary42"))?,
        sha256,
    ))
}

fn admission_order(cohort: &FrozenCohort) -> Vec<FrozenDataset> {
    let mut datasets = cohort.datasets.clone();
    datasets.sort_by(|left, right| {
        let left_cost = (left.train_rows as u128) * (left.features as u128);
        let right_cost = (right.train_rows as u128) * (right.features as u128);
        right_cost
            .cmp(&left_cost)
            .then_with(|| left.order.cmp(&right.order))
    });
    datasets
}

fn verify_dataset_binding(spec: &FrozenDataset) -> Result<()> {
    let train_path = Path::new(&spec.train_path);
    let test_path = Path::new(&spec.test_path);
    let report_path = Path::new(&spec.leader_report_path);
    if sha256_file(train_path)? != spec.train_sha256
        || sha256_file(test_path)? != spec.test_sha256
        || sha256_file(report_path)? != spec.leader_report_sha256
    {
        bail!(
            "Binary42 dataset or leader report hash mismatch for {}",
            spec.id
        );
    }
    let train = read_binary_csv(train_path)?;
    let test = read_binary_csv(test_path)?;
    if train.x.len() != spec.train_rows
        || test.x.len() != spec.test_rows
        || train.features != spec.features
        || test.features != spec.features
    {
        bail!("Binary42 dataset shape mismatch for {}", spec.id);
    }
    Ok(())
}

fn seed_population(features: usize) -> Result<Vec<MemberRecord>> {
    seed_population_with_winners(features, &[])
}

fn winner_seed_snapshot_sha256(sources: &[WinnerSeedSource]) -> Result<String> {
    Ok(sha256_bytes(&canonical_json(&sources)?))
}

fn seed_population_with_winners(
    features: usize,
    winner_sources: &[WinnerSeedSource],
) -> Result<Vec<MemberRecord>> {
    if sha256_bytes(BINARY_SEEDS) != SEED_BINARY_SHA256 {
        bail!("canonical binary seed hash mismatch");
    }
    if winner_sources.len() > POPULATION {
        bail!("Binary42 confirmed winner seed bank exceeds the population");
    }
    let base = test_adaptive_seed_population(BINARY_SEEDS, features)?;
    let mut seen = BTreeSet::new();
    let mut genomes = Vec::with_capacity(POPULATION);
    for source in winner_sources {
        if sha256_bytes(source.canonical_genome.as_bytes()) != source.semantic_sha256 {
            bail!("Binary42 winner seed semantic identity changed");
        }
        if !seen.insert(source.semantic_sha256.clone()) {
            bail!("Binary42 confirmed winner seed bank contains duplicate semantics");
        }
        genomes.push(source.canonical_genome.clone());
    }
    for member in base {
        if seen.insert(member.semantic_sha256) {
            genomes.push(member.canonical_genome);
        }
        if genomes.len() == POPULATION {
            break;
        }
    }
    if genomes.len() != POPULATION {
        bail!("Binary42 winner-augmented seed bank produced a short population");
    }
    let bytes = format!("{}\n", genomes.join("\n")).into_bytes();
    let members = test_adaptive_seed_population(&bytes, features)?
        .into_iter()
        .map(MemberRecord::from_binary)
        .collect::<Result<Vec<_>>>()?;
    if winner_sources
        .iter()
        .zip(&members)
        .any(|(source, member)| source.semantic_sha256 != member.semantic_sha256)
    {
        bail!("Binary42 confirmed winners are not first in generation zero");
    }
    Ok(members)
}

fn rank_slots(left: &LogicalSlot, right: &LogicalSlot) -> Ordering {
    let left_score = left.score.as_ref().expect("ranked slot has score");
    let right_score = right.score.as_ref().expect("ranked slot has score");
    right_score
        .auc
        .total_cmp(&left_score.auc)
        .then_with(|| left_score.logloss.total_cmp(&right_score.logloss))
        .then_with(|| left.member.complexity.cmp(&right.member.complexity))
        .then_with(|| {
            left.member
                .semantic_sha256
                .cmp(&right.member.semantic_sha256)
        })
}

fn strict_win(candidate_auc: f64, leader_auc: f64, generation_zero_best_auc: f64) -> bool {
    candidate_auc > leader_auc && candidate_auc > generation_zero_best_auc
}

fn policy_win(
    win_policy: &str,
    candidate_auc: f64,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> bool {
    match win_policy {
        "strictly_greater_auc" => strict_win(candidate_auc, leader_auc, generation_zero_best_auc),
        "exact_auc_1_tie" => candidate_auc.to_bits() == 1.0f64.to_bits(),
        _ => false,
    }
}

fn physical_hostname(node: &str) -> Result<&'static str> {
    NODES
        .iter()
        .position(|candidate| *candidate == node)
        .map(|index| PHYSICAL_HOSTNAMES[index])
        .with_context(|| format!("unknown Binary42 SCQ node {node}"))
}

#[cfg_attr(not(test), allow(dead_code))]
fn cohort_win(candidate_auc: f64, leader_auc: f64) -> bool {
    if leader_auc == 1.0 {
        candidate_auc == 1.0
    } else {
        candidate_auc > leader_auc
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn qualifying_winner(
    generation: usize,
    ranked: &[LogicalSlot],
    seed_semantics: &BTreeSet<String>,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> Option<QualifyingWinner> {
    qualifying_winner_for_policy(
        generation,
        ranked,
        seed_semantics,
        "strictly_greater_auc",
        leader_auc,
        generation_zero_best_auc,
    )
}

fn qualifying_winner_for_policy(
    generation: usize,
    ranked: &[LogicalSlot],
    seed_semantics: &BTreeSet<String>,
    win_policy: &str,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> Option<QualifyingWinner> {
    if generation == 0 {
        return None;
    }
    ranked.iter().find_map(|slot| {
        let score = slot.score.as_ref()?;
        (!seed_semantics.contains(&slot.member.semantic_sha256)
            && matches!(
                slot.member.ancestry.operation.as_str(),
                "mutation" | "crossover" | "elite"
            )
            && policy_win(win_policy, score.auc, leader_auc, generation_zero_best_auc))
        .then(|| QualifyingWinner {
            generation,
            slot: slot.slot,
            semantic_sha256: slot.member.semantic_sha256.clone(),
            auc: score.auc,
            logloss: score.logloss,
            complexity: slot.member.complexity,
        })
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn shard_members(members: &[MemberRecord]) -> Result<Vec<Vec<MemberRecord>>> {
    shard_members_with_sizes(members, &[SHARD_SIZE; 4])
}

fn shard_members_with_sizes(
    members: &[MemberRecord],
    shard_sizes: &[usize],
) -> Result<Vec<Vec<MemberRecord>>> {
    if members.len() != POPULATION {
        bail!("Binary42 generation requires exactly {POPULATION} members");
    }
    if shard_sizes.len() != NODES.len() || shard_sizes.iter().sum::<usize>() != POPULATION {
        bail!("Binary42 shard sizes do not account for the 60-member population");
    }
    let mut offset = 0;
    let mut shards = Vec::with_capacity(shard_sizes.len());
    for size in shard_sizes {
        shards.push(members[offset..offset + size].to_vec());
        offset += size;
    }
    Ok(shards)
}

fn campaign_shard_sizes(config: &CampaignConfig) -> Vec<usize> {
    if config.shard_sizes.is_empty() {
        vec![SHARD_SIZE; NODES.len()]
    } else {
        config.shard_sizes.clone()
    }
}

fn shard_start(shard_sizes: &[usize], shard: usize) -> Result<usize> {
    if shard >= shard_sizes.len() {
        bail!("Binary42 shard index is outside the configured partition");
    }
    Ok(shard_sizes[..shard].iter().sum())
}

fn assigned_node_for_slot(shard_sizes: &[usize], slot: usize) -> Result<&'static str> {
    let mut end = 0;
    for (index, size) in shard_sizes.iter().enumerate() {
        end += size;
        if slot < end {
            return Ok(NODES[index]);
        }
    }
    bail!("Binary42 slot {slot} is outside the configured shard partition")
}

fn generation_cap_reached(config: &CampaignConfig, generation: usize) -> bool {
    config
        .generation_cap
        .is_some_and(|cap| generation + 1 >= cap)
}

fn failover_node(shard: usize, attempt: usize) -> Result<&'static str> {
    if shard >= NODES.len() || attempt > 2 {
        bail!("Binary42 failover index is outside the immutable policy");
    }
    Ok(NODES[(shard + attempt) % NODES.len()])
}

fn binary_prediction_bytes(predictions: &[f64]) -> Vec<u8> {
    let mut bytes = b"prediction\n".to_vec();
    for prediction in predictions {
        bytes.extend_from_slice(format!("{prediction:.17}\n").as_bytes());
    }
    bytes
}

fn binary_prediction_roundtrip_bytes(predictions: &[f64]) -> Vec<u8> {
    let mut bytes = b"prediction\n".to_vec();
    for prediction in predictions {
        bytes.extend_from_slice(format!("{prediction:?}\n").as_bytes());
    }
    bytes
}

fn parse_prediction_bytes(bytes: &[u8]) -> Result<Vec<f64>> {
    let text = std::str::from_utf8(bytes).context("prediction object is not UTF-8")?;
    let mut lines = text.lines();
    if lines.next() != Some("prediction") {
        bail!("prediction object has the wrong header");
    }
    lines
        .enumerate()
        .map(|(index, line)| {
            let value = line
                .parse::<f64>()
                .with_context(|| format!("prediction row {} is not numeric", index + 1))?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                bail!("prediction row {} is outside [0, 1]", index + 1);
            }
            Ok(value)
        })
        .collect()
}

fn candidate_file(generation: usize, slot: usize) -> String {
    format!("candidate-g{generation:06}-s{slot:02}.receipt.v1.json")
}

fn prediction_file(generation: usize, slot: usize) -> String {
    format!("candidate-g{generation:06}-s{slot:02}.predictions.csv")
}

fn expected_job_id(manifest: &ShardManifest) -> Result<String> {
    let mut binding = manifest.clone();
    binding.job_id.clear();
    Ok(format!(
        "jope-b42-{}",
        &sha256_bytes(&canonical_json(&binding)?)[..24]
    ))
}

fn validate_shard_manifest(manifest: &ShardManifest) -> Result<()> {
    let expected_slots = if manifest.kind == "generation" {
        if manifest.cohort_mode.as_deref() == Some("all") {
            *ALL_SHARD_SIZES
                .get(manifest.shard)
                .context("Binary42 all-dataset shard index is invalid")?
        } else {
            SHARD_SIZE
        }
    } else if manifest.kind == "confirmation" {
        1
    } else {
        bail!("unknown Binary42 shard kind");
    };
    let expected_cpus = expected_slots;
    let hostname = String::from_utf8(
        scq_batch::command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string();
    let expected_hostname = physical_hostname(&manifest.node)?;
    let (dataset, _) = cohort_dataset(&manifest.dataset)?;
    if manifest.schema != SHARD_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || (manifest.cohort_mode.is_none() && manifest.dataset != ILPD)
        || (manifest.cohort_mode.is_some() && manifest.cohort_mode.as_deref() != Some("all"))
        || !NODES.contains(&manifest.node.as_str())
        || hostname != expected_hostname
        || manifest.slots.len() != expected_slots
        || manifest.cpus_reserved != expected_cpus
        || manifest.gpus_per_task != 0
        || manifest.candidate_threads != 1
        || manifest.job_id != expected_job_id(manifest)?
        || current_binary_sha256()? != manifest.release_binary_sha256
    {
        bail!("Binary42 shard identity or resource contract mismatch");
    }
    if manifest.train_sha256 != dataset.train_sha256
        || manifest.test_sha256 != dataset.test_sha256
        || manifest.train_rows != dataset.train_rows
        || manifest.test_rows != dataset.test_rows
        || manifest.features != dataset.features
        || (manifest.cohort_mode.as_deref() == Some("all")
            && manifest.scq_binary_path.as_deref() != Some(all_scq_binary_str()))
        || (manifest.cohort_mode.is_none() && manifest.scq_binary_path.is_some())
    {
        bail!("Binary42 shard dataset binding mismatch");
    }
    let mut slot_ids = BTreeSet::new();
    let mut semantics = BTreeSet::new();
    for slot in &manifest.slots {
        if slot.generation != manifest.generation
            || slot.member.semantic_sha256 != sha256_bytes(slot.member.canonical_genome.as_bytes())
            || !slot_ids.insert(slot.slot)
            || !semantics.insert(slot.member.semantic_sha256.clone())
        {
            bail!("Binary42 shard contains an invalid or duplicate slot");
        }
    }
    if manifest.kind == "confirmation" && manifest.slots.iter().any(|slot| !slot.fresh) {
        bail!("Binary42 confirmations must be independently fresh");
    }
    Ok(())
}

fn failed_candidate_receipt(
    manifest: &ShardManifest,
    slot: &ShardSlot,
    hostname: &str,
    cpu: usize,
    runtime_seconds: f64,
    backend: Option<CandidateBackendReceipt>,
    failure: String,
) -> CandidateReceipt {
    let empty = binary_prediction_bytes(&[]);
    CandidateReceipt {
        schema: CANDIDATE_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        kind: manifest.kind.clone(),
        run_id: manifest.run_id.clone(),
        dataset: manifest.dataset.clone(),
        generation: manifest.generation,
        slot: slot.slot,
        attempt: manifest.attempt,
        semantic_sha256: slot.member.semantic_sha256.clone(),
        canonical_genome: slot.member.canonical_genome.clone(),
        complexity: slot.member.complexity,
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        train_sha256: manifest.train_sha256.clone(),
        test_sha256: manifest.test_sha256.clone(),
        train_rows: manifest.train_rows,
        test_rows: manifest.test_rows,
        features: manifest.features,
        node: manifest.node.clone(),
        hostname: hostname.to_string(),
        scq_job_id: manifest.job_id.clone(),
        candidate_threads: 1,
        cpu,
        runtime_seconds,
        fit_seconds: None,
        predict_seconds: None,
        backend,
        predictions_sha256: sha256_bytes(&empty),
        score: None,
        failure: Some(failure),
    }
}

fn cpp17_toolchain() -> Result<CandidateBackendReceipt> {
    let output = scq_batch::command_output(
        {
            let mut command = Command::new("g++");
            command.arg("--version");
            command
        },
        "Binary42 fitted C++17 compiler identity",
    )?;
    let compiler = String::from_utf8(output.stdout)?
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
    if !compiler.contains("13.3") {
        bail!("Binary42 fitted C++17 requires g++ 13.3, found {compiler:?}");
    }
    Ok(CandidateBackendReceipt {
        name: CPP17_BACKEND.to_string(),
        language_standard: CPP17_STANDARD.to_string(),
        compiler,
        compiler_flags: CPP17_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        source_sha256: None,
        codec_sha256: None,
        compile_seconds: None,
    })
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_candidate_backend(receipt: &CandidateReceipt, all_dataset_cohort: bool) -> Result<()> {
    match (&receipt.backend, all_dataset_cohort) {
        (None, false) => Ok(()),
        (Some(backend), true) => {
            if backend.name != CPP17_BACKEND
                || backend.language_standard != CPP17_STANDARD
                || !backend.compiler.contains("13.3")
                || backend.compiler_flags
                    != CPP17_FLAGS
                        .iter()
                        .map(|flag| (*flag).to_string())
                        .collect::<Vec<_>>()
            {
                bail!("Binary42 fitted C++17 backend identity mismatch");
            }
            match (&receipt.score, &receipt.failure) {
                (Some(_), None)
                    if backend.source_sha256.as_deref().is_some_and(is_sha256_hex)
                        && backend.codec_sha256.as_deref().is_some_and(is_sha256_hex)
                        && backend
                            .compile_seconds
                            .is_some_and(|seconds| seconds >= 0.0) =>
                {
                    Ok(())
                }
                (None, Some(_))
                    if backend.source_sha256.is_none()
                        && backend.codec_sha256.is_none()
                        && backend.compile_seconds.is_none() =>
                {
                    Ok(())
                }
                _ => bail!("Binary42 fitted C++17 build provenance is incomplete"),
            }
        }
        _ => bail!("Binary42 candidate used the wrong execution backend"),
    }
}

fn evaluate_worker_candidate(
    manifest: &ShardManifest,
    slot: &ShardSlot,
    train: &RegressionData,
    test: &RegressionData,
    cpu: usize,
    cpp17_toolchain: Option<&CandidateBackendReceipt>,
    runtime_root: &Path,
) -> (CandidateReceipt, Vec<u8>) {
    let started = Instant::now();
    let all_dataset_cohort = manifest.cohort_mode.as_deref() == Some("all");
    let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if all_dataset_cohort {
            let built = evaluate_test_adaptive_binary_cpp17(
                &slot.member.canonical_genome,
                train,
                test,
                runtime_root,
            )?;
            let mut backend = cpp17_toolchain
                .context("Binary42 fitted C++17 toolchain was not initialized")?
                .clone();
            backend.source_sha256 = Some(built.source_sha256);
            backend.codec_sha256 = Some(built.codec_sha256);
            backend.compile_seconds = Some(built.compile_seconds);
            Ok((built.evaluation, Some(backend)))
        } else {
            evaluate_test_adaptive_binary(&slot.member.canonical_genome, train, test)
                .map(|evaluation| (evaluation, None))
        }
    }));
    match evaluated {
        Ok(Ok((result, backend))) => {
            let predictions = if all_dataset_cohort {
                binary_prediction_roundtrip_bytes(&result.predictions)
            } else {
                binary_prediction_bytes(&result.predictions)
            };
            (
                CandidateReceipt {
                    schema: CANDIDATE_SCHEMA.to_string(),
                    test_adaptive: true,
                    benchmark_valid: false,
                    kind: manifest.kind.clone(),
                    run_id: manifest.run_id.clone(),
                    dataset: manifest.dataset.clone(),
                    generation: manifest.generation,
                    slot: slot.slot,
                    attempt: manifest.attempt,
                    semantic_sha256: slot.member.semantic_sha256.clone(),
                    canonical_genome: slot.member.canonical_genome.clone(),
                    complexity: result.complexity,
                    release_binary_sha256: manifest.release_binary_sha256.clone(),
                    train_sha256: manifest.train_sha256.clone(),
                    test_sha256: manifest.test_sha256.clone(),
                    train_rows: manifest.train_rows,
                    test_rows: manifest.test_rows,
                    features: manifest.features,
                    node: manifest.node.clone(),
                    hostname: manifest.node.clone(),
                    scq_job_id: manifest.job_id.clone(),
                    candidate_threads: 1,
                    cpu,
                    runtime_seconds: started.elapsed().as_secs_f64(),
                    fit_seconds: Some(result.fit_seconds),
                    predict_seconds: Some(result.predict_seconds),
                    backend,
                    predictions_sha256: sha256_bytes(&predictions),
                    score: Some(CandidateScore {
                        auc: result.auc,
                        logloss: result.logloss,
                    }),
                    failure: None,
                },
                predictions,
            )
        }
        Ok(Err(error)) => {
            let predictions = binary_prediction_bytes(&[]);
            (
                failed_candidate_receipt(
                    manifest,
                    slot,
                    &manifest.node,
                    cpu,
                    started.elapsed().as_secs_f64(),
                    cpp17_toolchain.cloned(),
                    format!("{error:#}"),
                ),
                predictions,
            )
        }
        Err(_) => {
            let predictions = binary_prediction_bytes(&[]);
            (
                failed_candidate_receipt(
                    manifest,
                    slot,
                    &manifest.node,
                    cpu,
                    started.elapsed().as_secs_f64(),
                    cpp17_toolchain.cloned(),
                    "candidate panicked".to_string(),
                ),
                predictions,
            )
        }
    }
}

pub(crate) fn batch_worker(manifest_path: &Path, output_dir: &Path) -> Result<()> {
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest: ShardManifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest_bytes != canonical_json(&manifest)? {
        bail!("Binary42 shard manifest is not canonical JSON");
    }
    validate_shard_manifest(&manifest)?;
    if sha256_file(Path::new(&manifest.train_path))? != manifest.train_sha256
        || sha256_file(Path::new(&manifest.test_path))? != manifest.test_sha256
    {
        bail!("Binary42 worker data hash mismatch");
    }
    let train = read_binary_csv(Path::new(&manifest.train_path))?;
    let test = read_binary_csv(Path::new(&manifest.test_path))?;
    if train.x.len() != manifest.train_rows
        || test.x.len() != manifest.test_rows
        || train.features != manifest.features
        || test.features != manifest.features
    {
        bail!("Binary42 worker data shape mismatch");
    }
    let hostname = manifest.node.clone();
    let fresh = manifest
        .slots
        .iter()
        .filter(|slot| slot.fresh)
        .cloned()
        .collect::<Vec<_>>();
    let allowed = scq_batch::job_allowed_cpus()?;
    if allowed.len() < manifest.cpus_reserved || fresh.len() > manifest.cpus_reserved {
        bail!(
            "Binary42 SCQ allocation exposes {} CPUs for {} reserved and {} fresh candidates",
            allowed.len(),
            manifest.cpus_reserved,
            fresh.len()
        );
    }
    fs::create_dir_all(output_dir)?;
    let cpp17_toolchain = manifest
        .cohort_mode
        .as_deref()
        .is_some_and(|mode| mode == "all")
        .then(cpp17_toolchain)
        .transpose()?;
    let started_at_unix_ms = unix_millis();
    let outputs = Arc::new(Mutex::new(Vec::<WorkerCandidate>::new()));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for (index, slot) in fresh.iter().enumerate() {
            let outputs = Arc::clone(&outputs);
            let thread_failures = Arc::clone(&failures);
            let spawn_failures = Arc::clone(&failures);
            let train = &train;
            let test = &test;
            let manifest = &manifest;
            let cpp17_toolchain = cpp17_toolchain.as_ref();
            let cpu = allowed[index];
            let spawn = thread::Builder::new()
                .name(format!(
                    "binary42-g{:02}-s{:02}",
                    slot.generation, slot.slot
                ))
                .stack_size(CANDIDATE_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    let result = (|| -> Result<WorkerCandidate> {
                        scq_batch::pin_current_thread(cpu)?;
                        let runtime_root = output_dir.join(format!(
                            ".cpp17-runtime-g{:02}-s{:02}",
                            slot.generation, slot.slot
                        ));
                        let (receipt, prediction_bytes) = evaluate_worker_candidate(
                            manifest,
                            slot,
                            train,
                            test,
                            cpu,
                            cpp17_toolchain,
                            &runtime_root,
                        );
                        let _ = fs::remove_dir_all(&runtime_root);
                        validate_candidate_backend(
                            &receipt,
                            manifest.cohort_mode.as_deref() == Some("all"),
                        )?;
                        let receipt_bytes = canonical_json(&receipt)?;
                        let receipt_file = candidate_file(slot.generation, slot.slot);
                        let predictions_file = prediction_file(slot.generation, slot.slot);
                        immutable_write(&output_dir.join(&receipt_file), &receipt_bytes)?;
                        immutable_write(&output_dir.join(&predictions_file), &prediction_bytes)?;
                        Ok(WorkerCandidate {
                            generation: slot.generation,
                            slot: slot.slot,
                            semantic_sha256: slot.member.semantic_sha256.clone(),
                            receipt_file,
                            receipt_sha256: sha256_bytes(&receipt_bytes),
                            predictions_file,
                            predictions_sha256: sha256_bytes(&prediction_bytes),
                        })
                    })();
                    match result {
                        Ok(output) => outputs.lock().unwrap().push(output),
                        Err(error) => thread_failures.lock().unwrap().push(format!(
                            "generation {} slot {}: {error:#}",
                            slot.generation, slot.slot
                        )),
                    }
                });
            if let Err(error) = spawn {
                spawn_failures.lock().unwrap().push(format!(
                    "generation {} slot {}: spawn candidate thread: {error}",
                    slot.generation, slot.slot
                ));
            }
        }
    });
    let failures = Arc::try_unwrap(failures)
        .expect("Binary42 worker failure references remain")
        .into_inner()
        .unwrap();
    if !failures.is_empty() {
        bail!(
            "Binary42 worker infrastructure failures: {}",
            failures.join("; ")
        );
    }
    let mut candidates = Arc::try_unwrap(outputs)
        .expect("Binary42 worker output references remain")
        .into_inner()
        .unwrap();
    candidates.sort_by_key(|candidate| candidate.slot);
    if candidates.len() != fresh.len() {
        bail!("Binary42 worker produced a short fresh-candidate set");
    }
    let receipt = WorkerReceipt {
        schema: WORKER_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        kind: manifest.kind.clone(),
        run_id: manifest.run_id.clone(),
        dataset: manifest.dataset.clone(),
        generation: manifest.generation,
        shard: manifest.shard,
        node: manifest.node.clone(),
        hostname,
        job_id: manifest.job_id.clone(),
        manifest_sha256: sha256_bytes(&manifest_bytes),
        logical_slots: manifest.slots.len(),
        fresh_candidates: fresh.len(),
        cpus_reserved: manifest.cpus_reserved,
        candidate_threads: 1,
        candidates,
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
    };
    immutable_write(
        &output_dir.join("batch-worker-receipt.v1.json"),
        &canonical_json(&receipt)?,
    )
}

fn remote_bundle_root(node: &str, release_binary_sha256: &str) -> Result<String> {
    if !NODES.contains(&node) {
        bail!("unknown Binary42 SCQ node {node}");
    }
    Ok(format!(
        "{linux_remote_root_str}/bundles/{release_binary_sha256}",
        linux_remote_root_str = linux_remote_root_str()
    ))
}

fn controller_hostname() -> Result<String> {
    Ok(String::from_utf8(
        scq_batch::command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "controller hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string())
}

fn is_controller_node(node: &str) -> Result<bool> {
    Ok(controller_hostname()? == physical_hostname(node)?)
}

fn validate_local_immutable(
    path: &Path,
    expected_sha256: &str,
    expected_permissions: u32,
) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("local immutable path is absent: {}", path.display()))?;
    if metadata.file_type().is_symlink()
        || !metadata.file_type().is_file()
        || metadata.permissions().mode() & 0o7777 != expected_permissions
    {
        bail!(
            "local immutable mode mismatch at {}: expected regular file with permissions {expected_permissions:#o}",
            path.display()
        );
    }
    let actual_sha256 = sha256_file(path)?;
    if actual_sha256 != expected_sha256 {
        bail!(
            "local immutable hash mismatch at {}: got {actual_sha256}, expected {expected_sha256}",
            path.display()
        );
    }
    Ok(())
}

fn copy_local_atomic(source: &Path, destination: &Path, executable: bool) -> Result<()> {
    let expected_sha256 = sha256_file(source)?;
    let expected_permissions = if executable { 0o555 } else { 0o444 };
    if destination.exists() {
        return validate_local_immutable(destination, &expected_sha256, expected_permissions);
    }
    let parent = destination
        .parent()
        .context("local staging destination has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".jope-stage.{}.{}.{}",
        std::process::id(),
        unix_millis(),
        &expected_sha256[..12]
    ));
    let mut source_file = fs::File::open(source)?;
    let mut temporary_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    std::io::copy(&mut source_file, &mut temporary_file)?;
    temporary_file.sync_all()?;
    fs::set_permissions(&temporary, fs::Permissions::from_mode(expected_permissions))?;
    validate_local_immutable(&temporary, &expected_sha256, expected_permissions)?;
    let publish = fs::hard_link(&temporary, destination);
    fs::remove_file(&temporary)?;
    if let Err(error) = publish {
        validate_local_immutable(destination, &expected_sha256, expected_permissions)
            .with_context(|| {
                format!(
                    "exclusive local publish failed at {}: {error}",
                    destination.display()
                )
            })?;
    }
    validate_local_immutable(destination, &expected_sha256, expected_permissions)
}

fn submit_canary(node: &str, suffix: &str) -> Result<(String, String)> {
    let id = format!("jope-b42-canary-{node}-{suffix}");
    let spec = format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/bin/true\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {SHARD_SIZE}\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"Binary42 C++17 compiler available\"\ncommand = \"/usr/bin/test\"\nargs = [\"-x\", \"/usr/bin/g++\"]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n"
    );
    let spec_path = std::env::temp_dir().join(format!("{id}.toml"));
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&spec_path)?;
        file.write_all(spec.as_bytes())?;
        file.sync_all()?;
    }
    let submitted = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("submit").arg(&spec_path);
            command
        },
        "Binary42 pinned SCQ canary",
    );
    fs::remove_file(&spec_path)?;
    submitted?;
    let show = scq_batch::wait_for_job(&id)?;
    let job = scq_batch::scq_job(&show, &id)?;
    if job.get("state").and_then(Value::as_str) != Some("completed")
        || job.get("exit_code").and_then(Value::as_i64) != Some(0)
        || job.get("cpus").and_then(Value::as_u64) != Some(SHARD_SIZE as u64)
    {
        bail!("Binary42 pinned SCQ canary failed its node/resource contract");
    }
    Ok((id, sha256_bytes(&canonical_json(&show)?)))
}

fn submit_canary_at(
    scq_binary: &str,
    node: &str,
    suffix: &str,
    cpus_reserved: usize,
) -> Result<(String, String)> {
    let id = format!("jope-b42-all-canary-{node}-{suffix}");
    let spec = format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/bin/true\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {cpus_reserved}\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"Binary42 C++17 compiler available\"\ncommand = \"/usr/bin/test\"\nargs = [\"-x\", \"/usr/bin/g++\"]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n"
    );
    let spec_path = std::env::temp_dir().join(format!("{id}.toml"));
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&spec_path)?;
        file.write_all(spec.as_bytes())?;
        file.sync_all()?;
    }
    let submitted = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary);
            command.arg("submit").arg(&spec_path);
            command
        },
        "Binary42 all-dataset aggregate pinned SCQ canary",
    );
    fs::remove_file(&spec_path)?;
    submitted?;
    let show = wait_for_job_at(scq_binary, &id)?;
    let job = scq_batch::scq_job(&show, &id)?;
    if job.get("state").and_then(Value::as_str) != Some("completed")
        || job.get("exit_code").and_then(Value::as_i64) != Some(0)
        || job.get("cpus").and_then(Value::as_u64) != Some(cpus_reserved as u64)
    {
        bail!("Binary42 aggregate pinned SCQ canary failed its node/resource contract");
    }
    Ok((id, sha256_bytes(&canonical_json(&show)?)))
}

fn node_command_output(node: &str, args: &[&str], label: &str) -> Result<std::process::Output> {
    if is_controller_node(node)? {
        let mut command = Command::new(args.first().context("empty local node command")?);
        command.args(&args[1..]);
        scq_batch::command_output(command, label)
    } else {
        scq_batch::ssh_output(node, args, label)
    }
}

fn validate_all_staging(
    staging: &AllStagingReceipt,
    cohort: &FrozenCohort,
    release_binary_sha256: &str,
    scq_resolved_path: &str,
    scq_binary_sha256: &str,
    verify_remote: bool,
) -> Result<()> {
    if staging.schema != ALL_STAGING_SCHEMA
        || !staging.test_adaptive
        || staging.benchmark_valid
        || staging.release_binary_sha256 != release_binary_sha256
        || staging.scq_configured_path != all_scq_binary_str()
        || staging.scq_resolved_path != scq_resolved_path
        || staging.scq_binary_sha256 != scq_binary_sha256
        || staging.nodes.len() != NODES.len()
    {
        bail!("Binary42 all-dataset staging identity mismatch");
    }
    let budget = AllCohortConfig::fixed().node_cpu_budget_per_active_set;
    for (node_index, node) in staging.nodes.iter().enumerate() {
        let expected_budget = budget
            .get(NODES[node_index])
            .copied()
            .context("Binary42 node CPU budget is absent")?;
        if node.node != NODES[node_index]
            || node.hostname != PHYSICAL_HOSTNAMES[node_index]
            || node.logical_cpus < expected_budget
            || !node.cpp17_compiler.contains("13.3")
            || node.canary_cpus_reserved != expected_budget
            || node.binary_sha256 != release_binary_sha256
            || node.bundle_root != remote_bundle_root(&node.node, release_binary_sha256)?
            || node.datasets.len() != cohort.datasets.len()
        {
            bail!("Binary42 all-dataset staged node binding mismatch");
        }
        for (dataset_index, staged) in node.datasets.iter().enumerate() {
            let spec = &cohort.datasets[dataset_index];
            if staged.dataset != spec.id
                || staged.train_sha256 != spec.train_sha256
                || staged.test_sha256 != spec.test_sha256
            {
                bail!("Binary42 all-dataset staged data binding mismatch");
            }
            if verify_remote {
                if is_controller_node(&node.node)? {
                    validate_local_immutable(
                        Path::new(&staged.train_path),
                        &staged.train_sha256,
                        0o444,
                    )?;
                    validate_local_immutable(
                        Path::new(&staged.test_path),
                        &staged.test_sha256,
                        0o444,
                    )?;
                } else {
                    scq_batch::validate_remote_immutable(
                        &node.node,
                        &staged.train_path,
                        &staged.train_sha256,
                        0o444,
                    )?;
                    scq_batch::validate_remote_immutable(
                        &node.node,
                        &staged.test_path,
                        &staged.test_sha256,
                        0o444,
                    )?;
                }
            }
        }
        if verify_remote {
            if is_controller_node(&node.node)? {
                validate_local_immutable(Path::new(&node.binary_path), &node.binary_sha256, 0o555)?;
            } else {
                scq_batch::validate_remote_immutable(
                    &node.node,
                    &node.binary_path,
                    &node.binary_sha256,
                    0o555,
                )?;
            }
        }
    }
    Ok(())
}

fn stage_all_campaign(
    cohort_root: &Path,
    cohort: &FrozenCohort,
    release_binary_sha256: &str,
    scq_resolved_path: &str,
    scq_binary_sha256: &str,
) -> Result<(AllStagingReceipt, String)> {
    let path = cohort_root.join("staging.v1.json");
    if path.exists() {
        let bytes = fs::read(&path)?;
        let staging: AllStagingReceipt = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&staging)? {
            bail!("Binary42 all-dataset staging receipt is not canonical JSON");
        }
        validate_all_staging(
            &staging,
            cohort,
            release_binary_sha256,
            scq_resolved_path,
            scq_binary_sha256,
            true,
        )?;
        return Ok((staging, sha256_bytes(&bytes)));
    }
    if controller_hostname()? != "xbabe2" {
        bail!("Binary42 all-dataset controller must run on xbabe2");
    }
    let release_binary = std::env::current_exe()?;
    if sha256_file(&release_binary)? != release_binary_sha256 {
        bail!("release binary changed before Binary42 all-dataset staging");
    }
    let doctor = scq_batch::command_output(
        {
            let mut command = Command::new(all_scq_binary_str());
            command.arg("doctor");
            command
        },
        "Binary42 all-dataset SCQ doctor",
    )?;
    let doctor_output_sha256 = sha256_bytes(&[doctor.stdout, doctor.stderr].concat());
    let suffix = format!("{}-{}", &release_binary_sha256[..8], unix_millis());
    let budgets = AllCohortConfig::fixed().node_cpu_budget_per_active_set;
    let canaries = Arc::new(Mutex::new(Vec::new()));
    thread::scope(|scope| {
        for node in NODES {
            let canaries = Arc::clone(&canaries);
            let suffix = suffix.clone();
            let cpus = budgets[node];
            scope.spawn(move || {
                let mut result = None;
                for attempt in 0..3 {
                    match submit_canary_at(
                        all_scq_binary_str(),
                        node,
                        &format!("{suffix}-a{attempt}"),
                        cpus,
                    ) {
                        Ok(receipt) => {
                            result = Some(Ok(receipt));
                            break;
                        }
                        Err(error) if attempt == 2 => {
                            let _ = record_cohort_event(
                                cohort_root,
                                "aggregate_canary_failed",
                                None,
                                Some(&format!("node={node}, attempt={attempt}, error={error:#}")),
                            );
                            result = Some(Err(error));
                        }
                        Err(error) => {
                            let _ = record_cohort_event(
                                cohort_root,
                                "aggregate_canary_retry",
                                None,
                                Some(&format!("node={node}, attempt={attempt}, error={error:#}")),
                            );
                        }
                    }
                }
                canaries
                    .lock()
                    .unwrap()
                    .push((node.to_string(), result.unwrap()));
            });
        }
    });
    let mut canaries = Arc::try_unwrap(canaries)
        .expect("Binary42 all-dataset canary references remain")
        .into_inner()
        .unwrap();
    canaries.sort_by_key(|(node, _)| {
        NODES
            .iter()
            .position(|candidate| candidate == node)
            .unwrap()
    });
    let mut nodes = Vec::new();
    for (node_index, node) in NODES.iter().copied().enumerate() {
        let (canary_job_id, canary_show_sha256) = canaries
            .get_mut(node_index)
            .context("Binary42 aggregate canary result is absent")?
            .1
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error:#}"))?
            .clone();
        let hostname = String::from_utf8(
            node_command_output(node, &["hostname", "-s"], "Binary42 node hostname")?.stdout,
        )?
        .trim()
        .to_string();
        if hostname != physical_hostname(node)? {
            bail!("Binary42 node {node} reports unexpected hostname {hostname}");
        }
        let logical_cpus = String::from_utf8(
            node_command_output(node, &["nproc"], "Binary42 node CPU inventory")?.stdout,
        )?
        .trim()
        .parse::<usize>()
        .context("parse Binary42 node CPU inventory")?;
        if logical_cpus < budgets[node] {
            bail!(
                "Binary42 node {node} exposes {logical_cpus} CPUs but cohort needs {}",
                budgets[node]
            );
        }
        let cpp17_compiler = String::from_utf8(
            node_command_output(
                node,
                &["g++", "--version"],
                "Binary42 node fitted C++17 compiler identity",
            )?
            .stdout,
        )?
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
        if !cpp17_compiler.contains("13.3") {
            bail!("Binary42 node {node} requires g++ 13.3, found {cpp17_compiler:?}");
        }
        let bundle_root = remote_bundle_root(node, release_binary_sha256)?;
        let binary_path = format!("{bundle_root}/jope-dime-regressor");
        if is_controller_node(node)? {
            copy_local_atomic(&release_binary, Path::new(&binary_path), true)?;
        } else {
            scq_batch::copy_atomic(node, &release_binary, &binary_path, true)?;
        }
        let mut datasets = Vec::with_capacity(cohort.datasets.len());
        for spec in &cohort.datasets {
            let dataset_root = format!("{bundle_root}/datasets/{}", spec.id);
            let train_path = format!("{dataset_root}/train.csv");
            let test_path = format!("{dataset_root}/test.csv");
            if is_controller_node(node)? {
                copy_local_atomic(Path::new(&spec.train_path), Path::new(&train_path), false)?;
                copy_local_atomic(Path::new(&spec.test_path), Path::new(&test_path), false)?;
            } else {
                scq_batch::copy_atomic(node, Path::new(&spec.train_path), &train_path, false)?;
                scq_batch::copy_atomic(node, Path::new(&spec.test_path), &test_path, false)?;
            }
            datasets.push(AllStagedDataset {
                dataset: spec.id.clone(),
                train_path,
                train_sha256: spec.train_sha256.clone(),
                test_path,
                test_sha256: spec.test_sha256.clone(),
            });
        }
        nodes.push(AllStagedNode {
            node: node.to_string(),
            hostname,
            logical_cpus,
            cpp17_compiler,
            bundle_root,
            binary_path,
            binary_sha256: release_binary_sha256.to_string(),
            canary_job_id,
            canary_show_sha256,
            canary_cpus_reserved: budgets[node],
            datasets,
        });
    }
    let staging = AllStagingReceipt {
        schema: ALL_STAGING_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_configured_path: all_scq_binary_str().to_string(),
        scq_resolved_path: scq_resolved_path.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        doctor_output_sha256,
        nodes,
    };
    let sha256 = write_json(&path, &staging)?;
    validate_all_staging(
        &staging,
        cohort,
        release_binary_sha256,
        scq_resolved_path,
        scq_binary_sha256,
        true,
    )?;
    Ok((staging, sha256))
}

fn stage_campaign(
    root: &Path,
    spec: &FrozenDataset,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
) -> Result<StagingReceipt> {
    let staging_path = root.join("staging").join("staging.v1.json");
    if staging_path.exists() {
        let staging: StagingReceipt = load_json(&staging_path)?;
        validate_staging(
            &staging,
            spec,
            release_binary_sha256,
            scq_binary_sha256,
            true,
        )?;
        return Ok(staging);
    }
    let release_binary = std::env::current_exe()?;
    if sha256_file(&release_binary)? != release_binary_sha256 {
        bail!("release binary changed before Binary42 staging");
    }
    let doctor = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("doctor");
            command
        },
        "scq doctor",
    )?;
    let doctor_output_sha256 = sha256_bytes(&[doctor.stdout, doctor.stderr].concat());
    let local_hostname = controller_hostname()?;
    let suffix = format!("{}-{}", &release_binary_sha256[..8], unix_millis());
    let mut nodes = Vec::new();
    for node in NODES {
        let mut canary = None;
        for attempt in 0..3 {
            match submit_canary(node, &format!("{suffix}-a{attempt}")) {
                Ok(receipt) => {
                    canary = Some(receipt);
                    break;
                }
                Err(error) => eprintln!(
                    "Binary42 pinned canary attempt {attempt} failed on {node}: {error:#}"
                ),
            }
        }
        let (canary_job_id, canary_show_sha256) =
            canary.with_context(|| format!("Binary42 canary retries exhausted on {node}"))?;
        let hostname = if node == local_hostname {
            local_hostname.clone()
        } else {
            String::from_utf8(
                scq_batch::ssh_output(node, &["hostname", "-s"], "Binary42 remote hostname")?
                    .stdout,
            )?
            .trim()
            .to_string()
        };
        if hostname != physical_hostname(node)? {
            bail!("Binary42 SSH target {node} reports unexpected hostname {hostname}");
        }
        let bundle_root = remote_bundle_root(node, release_binary_sha256)?;
        let binary_path = format!("{bundle_root}/jope-dime-regressor");
        let dataset_root = format!("{bundle_root}/datasets/{}", spec.id);
        let train_path = format!("{dataset_root}/train.csv");
        let test_path = format!("{dataset_root}/test.csv");
        if node == local_hostname {
            copy_local_atomic(&release_binary, Path::new(&binary_path), true)?;
            copy_local_atomic(Path::new(&spec.train_path), Path::new(&train_path), false)?;
            copy_local_atomic(Path::new(&spec.test_path), Path::new(&test_path), false)?;
        } else {
            scq_batch::copy_atomic(node, &release_binary, &binary_path, true)?;
            scq_batch::copy_atomic(node, Path::new(&spec.train_path), &train_path, false)?;
            scq_batch::copy_atomic(node, Path::new(&spec.test_path), &test_path, false)?;
        }
        nodes.push(StagedNode {
            node: node.to_string(),
            hostname,
            bundle_root,
            binary_path,
            binary_sha256: release_binary_sha256.to_string(),
            train_path,
            train_sha256: spec.train_sha256.clone(),
            test_path,
            test_sha256: spec.test_sha256.clone(),
            canary_job_id,
            canary_show_sha256,
        });
    }
    let receipt = StagingReceipt {
        schema: STAGING_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        doctor_output_sha256,
        dataset: spec.id.clone(),
        nodes,
        cohort_staging_sha256: None,
    };
    write_json(&staging_path, &receipt)?;
    validate_staging(
        &receipt,
        spec,
        release_binary_sha256,
        scq_binary_sha256,
        true,
    )?;
    Ok(receipt)
}

fn validate_staging(
    staging: &StagingReceipt,
    spec: &FrozenDataset,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    verify_remote: bool,
) -> Result<()> {
    if staging.schema != STAGING_SCHEMA
        || !staging.test_adaptive
        || staging.benchmark_valid
        || staging.release_binary_sha256 != release_binary_sha256
        || staging.scq_binary_sha256 != scq_binary_sha256
        || staging.dataset != spec.id
        || staging.nodes.len() != NODES.len()
    {
        bail!("Binary42 staging receipt identity mismatch");
    }
    for (index, node) in staging.nodes.iter().enumerate() {
        if node.node != NODES[index]
            || node.hostname != PHYSICAL_HOSTNAMES[index]
            || node.binary_sha256 != release_binary_sha256
            || node.train_sha256 != spec.train_sha256
            || node.test_sha256 != spec.test_sha256
            || node.bundle_root != remote_bundle_root(&node.node, release_binary_sha256)?
        {
            bail!("Binary42 staged node binding mismatch");
        }
        if verify_remote {
            if is_controller_node(&node.node)? {
                validate_local_immutable(Path::new(&node.binary_path), &node.binary_sha256, 0o555)?;
                validate_local_immutable(Path::new(&node.train_path), &node.train_sha256, 0o444)?;
                validate_local_immutable(Path::new(&node.test_path), &node.test_sha256, 0o444)?;
            } else {
                scq_batch::validate_remote_immutable(
                    &node.node,
                    &node.binary_path,
                    &node.binary_sha256,
                    0o555,
                )?;
                scq_batch::validate_remote_immutable(
                    &node.node,
                    &node.train_path,
                    &node.train_sha256,
                    0o444,
                )?;
                scq_batch::validate_remote_immutable(
                    &node.node,
                    &node.test_path,
                    &node.test_sha256,
                    0o444,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn shard_manifest(
    campaign: &CampaignManifest,
    staged: &StagedNode,
    generation: usize,
    shard: usize,
    attempt: usize,
    kind: &str,
    members: &[MemberRecord],
    cache: &HashMap<String, CacheEntry>,
) -> Result<ShardManifest> {
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    let start = shard_start(&shard_sizes, shard)?;
    let slots = members
        .iter()
        .enumerate()
        .map(|(offset, member)| ShardSlot {
            generation,
            slot: if kind == "generation" {
                start + offset
            } else {
                offset
            },
            member: member.clone(),
            fresh: kind == "confirmation"
                || cache
                    .get(&member.semantic_sha256)
                    .is_none_or(|entry| entry.first_generation == generation),
        })
        .collect::<Vec<_>>();
    let mut manifest = ShardManifest {
        schema: SHARD_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        kind: kind.to_string(),
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation,
        shard,
        node: staged.node.clone(),
        job_id: String::new(),
        attempt,
        release_binary_sha256: campaign.release_binary_sha256.clone(),
        train_path: staged.train_path.clone(),
        train_sha256: campaign.dataset_binding.train_sha256.clone(),
        test_path: staged.test_path.clone(),
        test_sha256: campaign.dataset_binding.test_sha256.clone(),
        train_rows: campaign.dataset_binding.train_rows,
        test_rows: campaign.dataset_binding.test_rows,
        features: campaign.dataset_binding.features,
        cpus_reserved: if kind == "generation" {
            *shard_sizes
                .get(shard)
                .context("Binary42 shard index is invalid")?
        } else {
            1
        },
        gpus_per_task: 0,
        candidate_threads: 1,
        slots,
        cohort_mode: (!campaign.config.shard_sizes.is_empty()).then(|| "all".to_string()),
        scq_binary_path: (!campaign.config.shard_sizes.is_empty())
            .then(|| all_scq_binary_str().to_string()),
    };
    manifest.job_id = expected_job_id(&manifest)?;
    Ok(manifest)
}

fn scq_spec(manifest: &ShardManifest) -> Result<String> {
    let bundle_root = remote_bundle_root(&manifest.node, &manifest.release_binary_sha256)?;
    let binary = format!("{bundle_root}/jope-dime-regressor");
    let manifest_sha256 = sha256_bytes(&canonical_json(manifest)?);
    let remote_manifest = format!("{bundle_root}/jobs/manifests/{manifest_sha256}.json");
    let remote_output = format!("{bundle_root}/jobs/{}/output", manifest.job_id);
    let args = vec![
        "OMP_NUM_THREADS=1".to_string(),
        "OPENBLAS_NUM_THREADS=1".to_string(),
        "MKL_NUM_THREADS=1".to_string(),
        "NUMEXPR_NUM_THREADS=1".to_string(),
        binary,
        "binary-test-adaptive-batch-worker".to_string(),
        "--manifest".to_string(),
        remote_manifest.clone(),
        "--output-dir".to_string(),
        remote_output,
    ];
    let rendered_args = args
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join(", ");
    Ok(format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/usr/bin/env\"\nargs = [{rendered_args}]\nworkdir = \"/tmp\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {cpus}\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"Binary42 immutable worker manifest staged\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", {remote_manifest:?}]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = 24\nmax_restarts = 0\n",
        id = manifest.job_id,
        cpus = manifest.cpus_reserved,
    ))
}

fn copy_remote_output(node: &str, remote: &str, local: &Path) -> Result<()> {
    if local.exists() {
        fs::remove_dir_all(local)?;
    }
    fs::create_dir_all(local)?;
    if is_controller_node(node)? {
        copy_directory_contents(Path::new(remote), local)?;
    } else {
        scq_batch::command_output(
            {
                let mut command = Command::new("scp");
                command
                    .arg("-r")
                    .arg("--")
                    .arg(format!("{node}:{remote}/."))
                    .arg(local);
                command
            },
            "pull Binary42 SCQ output",
        )?;
    }
    Ok(())
}

fn copy_directory_contents(source: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(source)
        .with_context(|| format!("read local SCQ output {}", source.display()))?
    {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            bail!(
                "local SCQ output contains a symbolic link: {}",
                source_path.display()
            );
        }
        if file_type.is_dir() {
            fs::create_dir_all(&destination_path)?;
            copy_directory_contents(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &destination_path)?;
        } else {
            bail!(
                "local SCQ output contains a non-file entry: {}",
                source_path.display()
            );
        }
    }
    Ok(())
}

fn scq_json_at(binary: &str, args: &[&str]) -> Result<Value> {
    let output = scq_batch::command_output(
        {
            let mut command = Command::new(binary);
            command.args(args).arg("--json");
            command
        },
        "Binary42 SCQ JSON command",
    )?;
    serde_json::from_slice(&output.stdout).context("parse Binary42 SCQ JSON output")
}

fn wait_for_job_at(binary: &str, id: &str) -> Result<Value> {
    loop {
        let show = scq_json_at(binary, &["show", id])?;
        let state = scq_batch::scq_job(&show, id)?
            .get("state")
            .and_then(Value::as_str)
            .context("Binary42 SCQ job has no state")?;
        if matches!(state, "completed" | "failed" | "cancelled") {
            return Ok(show);
        }
        thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn submit_or_recover_job(manifest: &ShardManifest, spec_path: &Path) -> Result<Value> {
    let scq_binary = manifest
        .scq_binary_path
        .as_deref()
        .unwrap_or(scq_binary_str());
    let exists = scq_json_at(scq_binary, &["show", &manifest.job_id])
        .ok()
        .and_then(|show| scq_batch::scq_job(&show, &manifest.job_id).ok().map(|_| ()))
        .is_some();
    if !exists {
        scq_batch::command_output(
            {
                let mut command = Command::new(scq_binary);
                command.arg("submit").arg(spec_path);
                command
            },
            "submit Binary42 SCQ shard",
        )?;
    }
    wait_for_job_at(scq_binary, &manifest.job_id)
}

fn completed_shard_outputs(
    root: &Path,
    manifest: &ShardManifest,
    manifest_sha256: &str,
) -> Result<Option<Vec<FreshOutput>>> {
    let job_receipt_path = root
        .join("jobs")
        .join("receipts")
        .join(format!("{}.v1.json", manifest.job_id));
    if !job_receipt_path.exists() {
        return Ok(None);
    }
    let job_receipt_bytes = fs::read(&job_receipt_path)?;
    let job_receipt: JobReceipt = serde_json::from_slice(&job_receipt_bytes)?;
    if job_receipt_bytes != canonical_json(&job_receipt)?
        || job_receipt.schema != "JopeBinary42TestAdaptiveScqJob.v1"
        || !job_receipt.test_adaptive
        || job_receipt.benchmark_valid
        || job_receipt.job_id != manifest.job_id
        || job_receipt.node != manifest.node
        || job_receipt.attempt != manifest.attempt
        || job_receipt.manifest_sha256 != manifest_sha256
        || job_receipt.state != "completed"
        || job_receipt.exit_code != 0
    {
        bail!("Binary42 completed shard receipt changed or is invalid");
    }
    let raw = root.join("jobs").join("raw").join(&manifest.job_id);
    let worker_bytes = fs::read(raw.join("batch-worker-receipt.v1.json"))?;
    let worker: WorkerReceipt = serde_json::from_slice(&worker_bytes)?;
    if worker_bytes != canonical_json(&worker)?
        || sha256_bytes(&worker_bytes) != job_receipt.worker_receipt_sha256
        || worker.schema != WORKER_SCHEMA
        || !worker.test_adaptive
        || worker.benchmark_valid
        || worker.job_id != manifest.job_id
        || worker.node != manifest.node
        || worker.hostname != physical_hostname(&manifest.node)?
        || worker.manifest_sha256 != manifest_sha256
        || worker.logical_slots != manifest.slots.len()
        || worker.fresh_candidates != manifest.slots.iter().filter(|slot| slot.fresh).count()
    {
        bail!("Binary42 completed worker receipt changed or is invalid");
    }
    let mut outputs = Vec::with_capacity(worker.candidates.len());
    for candidate in &worker.candidates {
        let receipt_bytes = fs::read(raw.join(&candidate.receipt_file))?;
        let prediction_bytes = fs::read(raw.join(&candidate.predictions_file))?;
        if sha256_bytes(&receipt_bytes) != candidate.receipt_sha256
            || sha256_bytes(&prediction_bytes) != candidate.predictions_sha256
        {
            bail!("Binary42 completed candidate output changed");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
        if receipt_bytes != canonical_json(&receipt)?
            || receipt.schema != CANDIDATE_SCHEMA
            || !receipt.test_adaptive
            || receipt.benchmark_valid
            || receipt.scq_job_id != manifest.job_id
            || receipt.node != manifest.node
            || receipt.hostname != physical_hostname(&manifest.node)?
            || receipt.predictions_sha256 != candidate.predictions_sha256
            || receipt.semantic_sha256 != candidate.semantic_sha256
        {
            bail!("Binary42 completed candidate receipt is invalid");
        }
        validate_candidate_backend(&receipt, manifest.cohort_mode.as_deref() == Some("all"))?;
        outputs.push(FreshOutput {
            receipt,
            receipt_bytes,
            prediction_bytes,
        });
    }
    Ok(Some(outputs))
}

fn execute_shard(root: &Path, manifest: &ShardManifest) -> Result<Vec<FreshOutput>> {
    let manifest_bytes = canonical_json(manifest)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let local_manifest = root
        .join("jobs")
        .join("manifests")
        .join(format!("{manifest_sha256}.json"));
    immutable_or_equal(&local_manifest, &manifest_bytes)?;
    if let Some(outputs) = completed_shard_outputs(root, manifest, &manifest_sha256)? {
        return Ok(outputs);
    }
    let bundle_root = remote_bundle_root(&manifest.node, &manifest.release_binary_sha256)?;
    let remote_manifest = format!("{bundle_root}/jobs/manifests/{manifest_sha256}.json");
    let remote_output = format!("{bundle_root}/jobs/{}/output", manifest.job_id);
    if is_controller_node(&manifest.node)? {
        copy_local_atomic(&local_manifest, Path::new(&remote_manifest), false)?;
        fs::create_dir_all(&remote_output)?;
    } else {
        scq_batch::copy_atomic(&manifest.node, &local_manifest, &remote_manifest, false)?;
        scq_batch::ssh_output(
            &manifest.node,
            &["mkdir", "-p", &remote_output],
            "create Binary42 remote output directory",
        )?;
    }
    let spec = scq_spec(manifest)?;
    let spec_path = root
        .join("jobs")
        .join("specs")
        .join(format!("{}.toml", manifest.job_id));
    immutable_or_equal(&spec_path, spec.as_bytes())?;
    let show = submit_or_recover_job(manifest, &spec_path)?;
    let job = scq_batch::scq_job(&show, &manifest.job_id)?;
    let state = job
        .get("state")
        .and_then(Value::as_str)
        .context("Binary42 SCQ job has no state")?;
    let exit_code = job
        .get("exit_code")
        .and_then(Value::as_i64)
        .context("Binary42 SCQ job has no exit code")?;
    if state != "completed"
        || exit_code != 0
        || job.get("node").and_then(Value::as_str) != Some(manifest.node.as_str())
        || job.get("cpus").and_then(Value::as_u64) != Some(manifest.cpus_reserved as u64)
    {
        bail!(
            "Binary42 SCQ job {} failed: state={state}, exit={exit_code}",
            manifest.job_id
        );
    }
    let temporary = std::env::temp_dir().join(format!(
        "jope-b42-pull-{}-{}",
        std::process::id(),
        manifest.job_id
    ));
    copy_remote_output(&manifest.node, &remote_output, &temporary)?;
    let raw = root.join("jobs").join("raw").join(&manifest.job_id);
    let worker_bytes = fs::read(temporary.join("batch-worker-receipt.v1.json"))?;
    let worker: WorkerReceipt = serde_json::from_slice(&worker_bytes)?;
    if worker.schema != WORKER_SCHEMA
        || !worker.test_adaptive
        || worker.benchmark_valid
        || worker.job_id != manifest.job_id
        || worker.node != manifest.node
        || worker.hostname != physical_hostname(&manifest.node)?
        || worker.manifest_sha256 != manifest_sha256
        || worker.logical_slots != manifest.slots.len()
        || worker.fresh_candidates != manifest.slots.iter().filter(|slot| slot.fresh).count()
    {
        bail!("Binary42 worker receipt binding mismatch");
    }
    immutable_or_equal(&raw.join("batch-worker-receipt.v1.json"), &worker_bytes)?;
    let mut outputs = Vec::new();
    for candidate in &worker.candidates {
        let receipt_bytes = fs::read(temporary.join(&candidate.receipt_file))?;
        let prediction_bytes = fs::read(temporary.join(&candidate.predictions_file))?;
        if sha256_bytes(&receipt_bytes) != candidate.receipt_sha256
            || sha256_bytes(&prediction_bytes) != candidate.predictions_sha256
        {
            bail!("Binary42 worker candidate output hash mismatch");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
        if receipt.schema != CANDIDATE_SCHEMA
            || !receipt.test_adaptive
            || receipt.benchmark_valid
            || receipt.scq_job_id != manifest.job_id
            || receipt.node != manifest.node
            || receipt.hostname != physical_hostname(&manifest.node)?
            || receipt.predictions_sha256 != candidate.predictions_sha256
            || receipt.semantic_sha256 != candidate.semantic_sha256
        {
            bail!("Binary42 candidate receipt binding mismatch");
        }
        validate_candidate_backend(&receipt, manifest.cohort_mode.as_deref() == Some("all"))?;
        immutable_or_equal(&raw.join(&candidate.receipt_file), &receipt_bytes)?;
        immutable_or_equal(&raw.join(&candidate.predictions_file), &prediction_bytes)?;
        outputs.push(FreshOutput {
            receipt,
            receipt_bytes,
            prediction_bytes,
        });
    }
    fs::remove_dir_all(&temporary)?;
    let job_receipt = JobReceipt {
        schema: "JopeBinary42TestAdaptiveScqJob.v1".to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        job_id: manifest.job_id.clone(),
        node: manifest.node.clone(),
        attempt: manifest.attempt,
        manifest_sha256,
        scq_show_sha256: sha256_bytes(&canonical_json(&show)?),
        worker_receipt_sha256: sha256_bytes(&worker_bytes),
        state: state.to_string(),
        exit_code,
    };
    write_json(
        &root
            .join("jobs")
            .join("receipts")
            .join(format!("{}.v1.json", manifest.job_id)),
        &job_receipt,
    )?;
    Ok(outputs)
}

fn cache_path(root: &Path, semantic_sha256: &str) -> PathBuf {
    root.join("objects")
        .join("cache")
        .join(format!("{semantic_sha256}.v1.json"))
}

fn load_cache(root: &Path) -> Result<HashMap<String, CacheEntry>> {
    let cache_root = root.join("objects").join("cache");
    if !cache_root.exists() {
        return Ok(HashMap::new());
    }
    let mut entries = HashMap::new();
    for entry in fs::read_dir(cache_root)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            bail!("unexpected file in Binary42 semantic cache");
        }
        let record: CacheEntry = load_json(&path)?;
        if record.schema != CACHE_SCHEMA
            || !record.test_adaptive
            || record.benchmark_valid
            || path.file_name().and_then(|value| value.to_str())
                != Some(&format!("{}.v1.json", record.semantic_sha256))
            || entries
                .insert(record.semantic_sha256.clone(), record)
                .is_some()
        {
            bail!("invalid or duplicate Binary42 semantic cache entry");
        }
    }
    Ok(entries)
}

fn store_genome(root: &Path, member: &MemberRecord) -> Result<()> {
    if member.semantic_sha256 != sha256_bytes(member.canonical_genome.as_bytes()) {
        bail!("Binary42 member semantic hash mismatch");
    }
    let object = GenomeObject {
        schema: GENOME_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        semantic_sha256: member.semantic_sha256.clone(),
        canonical_genome: member.canonical_genome.clone(),
        complexity: member.complexity,
    };
    write_json(
        &root
            .join("objects")
            .join("genomes")
            .join(format!("{}.v1.json", member.semantic_sha256)),
        &object,
    )?;
    Ok(())
}

fn validate_fresh_output(
    output: &FreshOutput,
    test: &RegressionData,
) -> Result<Option<CandidateScore>> {
    if sha256_bytes(&output.receipt_bytes) != sha256_bytes(&canonical_json(&output.receipt)?)
        || sha256_bytes(&output.prediction_bytes) != output.receipt.predictions_sha256
    {
        bail!("Binary42 fresh output content hash mismatch");
    }
    let predictions = parse_prediction_bytes(&output.prediction_bytes)?;
    match (&output.receipt.score, &output.receipt.failure) {
        (Some(score), None) => {
            if predictions.len() != test.y.len() {
                bail!("Binary42 candidate prediction length mismatch");
            }
            let replay = CandidateScore {
                auc: auc(&test.y, &predictions),
                logloss: feat_math::metrics::logloss(&test.y, &predictions),
            };
            if replay.auc.to_bits() != score.auc.to_bits()
                || !logloss_replays(score.logloss, replay.logloss)
            {
                bail!("Binary42 candidate metrics fall outside replay tolerance");
            }
            Ok(Some(replay))
        }
        (None, Some(_)) if predictions.is_empty() => Ok(None),
        _ => bail!("Binary42 candidate success/failure fields are inconsistent"),
    }
}

fn logloss_replays(worker: f64, controller: f64) -> bool {
    worker.is_finite()
        && controller.is_finite()
        && (worker.to_bits().abs_diff(controller.to_bits()) <= LOGLOSS_REPLAY_MAX_ULPS
            || (worker - controller).abs() <= LOGLOSS_REPLAY_MAX_ABS)
}

fn import_fresh_output(
    root: &Path,
    output: FreshOutput,
    test: &RegressionData,
    extended: bool,
) -> Result<CacheEntry> {
    validate_candidate_backend(&output.receipt, extended)?;
    let authoritative_score = validate_fresh_output(&output, test)?;
    let receipt_sha256 = sha256_bytes(&output.receipt_bytes);
    let prediction_sha256 = sha256_bytes(&output.prediction_bytes);
    immutable_or_equal(
        &root
            .join("objects")
            .join("receipts")
            .join(format!("{receipt_sha256}.json")),
        &output.receipt_bytes,
    )?;
    immutable_or_equal(
        &root
            .join("objects")
            .join("predictions")
            .join(format!("{prediction_sha256}.csv")),
        &output.prediction_bytes,
    )?;
    let cache = CacheEntry {
        schema: CACHE_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        semantic_sha256: output.receipt.semantic_sha256.clone(),
        receipt_sha256,
        prediction_sha256,
        score: authoritative_score,
        failure: output.receipt.failure.clone(),
        evaluation_node: output.receipt.node.clone(),
        scq_job_id: output.receipt.scq_job_id.clone(),
        first_generation: output.receipt.generation,
        first_slot: output.receipt.slot,
        runtime_seconds: extended.then_some(output.receipt.runtime_seconds),
    };
    write_json(&cache_path(root, &cache.semantic_sha256), &cache)?;
    Ok(cache)
}

fn record_infrastructure_failure(
    root: &Path,
    manifest: &ShardManifest,
    error: &anyhow::Error,
) -> Result<()> {
    let failure = json!({
        "schema": "JopeBinary42TestAdaptiveInfrastructureFailure.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "job_id": manifest.job_id,
        "node": manifest.node,
        "generation": manifest.generation,
        "shard": manifest.shard,
        "attempt": manifest.attempt,
        "error": format!("{error:#}"),
    });
    write_json(
        &root
            .join("jobs")
            .join("failures")
            .join(format!("{}.v1.json", manifest.job_id)),
        &failure,
    )?;
    Ok(())
}

fn execute_generation_shards(
    root: &Path,
    campaign: &CampaignManifest,
    staging: &StagingReceipt,
    generation: usize,
    members: &[MemberRecord],
    cache: &HashMap<String, CacheEntry>,
) -> Result<Vec<FreshOutput>> {
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    let chunks = shard_members_with_sizes(members, &shard_sizes)?;
    let manifests = chunks
        .iter()
        .enumerate()
        .map(|(shard, members)| {
            shard_manifest(
                campaign,
                &staging.nodes[shard],
                generation,
                shard,
                0,
                "generation",
                members,
                cache,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let results = Arc::new(Mutex::new(Vec::<(
        usize,
        std::result::Result<Vec<FreshOutput>, anyhow::Error>,
    )>::new()));
    thread::scope(|scope| {
        for (index, manifest) in manifests.iter().enumerate() {
            let results = Arc::clone(&results);
            scope.spawn(move || {
                let result = execute_shard(root, manifest);
                results.lock().unwrap().push((index, result));
            });
        }
    });
    let mut results = Arc::try_unwrap(results)
        .expect("Binary42 shard result references remain")
        .into_inner()
        .unwrap();
    results.sort_by_key(|(index, _)| *index);
    let mut outputs = Vec::new();
    for (index, result) in results {
        match result {
            Ok(mut fresh) => outputs.append(&mut fresh),
            Err(error) => {
                record_infrastructure_failure(root, &manifests[index], &error)?;
                let mut recovered = None;
                let mut final_error = error;
                for attempt in 1..=campaign.config.infrastructure_retries {
                    let retry_node = failover_node(index, attempt)?;
                    let retry = shard_manifest(
                        campaign,
                        &staging.nodes[NODES.iter().position(|node| *node == retry_node).unwrap()],
                        generation,
                        index,
                        attempt,
                        "generation",
                        &chunks[index],
                        cache,
                    )?;
                    match execute_shard(root, &retry) {
                        Ok(fresh) => {
                            recovered = Some(fresh);
                            break;
                        }
                        Err(error) => {
                            record_infrastructure_failure(root, &retry, &error)?;
                            final_error = error;
                        }
                    }
                }
                let mut fresh = recovered.with_context(|| {
                    format!(
                        "Binary42 generation {generation} shard {index} exhausted {} failover attempts: {final_error:#}",
                        campaign.config.infrastructure_retries
                    )
                })?;
                outputs.append(&mut fresh);
            }
        }
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
fn process_generation(
    root: &Path,
    campaign: &CampaignManifest,
    staging: &StagingReceipt,
    generation: usize,
    members: Vec<MemberRecord>,
    previous: Option<&GenerationCheckpoint>,
    seed_semantics: &BTreeSet<String>,
    test: &RegressionData,
) -> Result<GenerationCheckpoint> {
    if members.len() != POPULATION {
        bail!("Binary42 generation population is not exactly {POPULATION}");
    }
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    let generation_dir = generation_root(root, generation);
    let generation_manifest = GenerationManifest {
        schema: GENERATION_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation,
        logical_slots: POPULATION,
        shard_size: shard_sizes[0],
        shard_nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
        members: members.clone(),
        shard_sizes: if campaign.config.shard_sizes.is_empty() {
            Vec::new()
        } else {
            shard_sizes.clone()
        },
    };
    write_json(
        &generation_dir.join("generation-manifest.v1.json"),
        &generation_manifest,
    )?;
    for member in &members {
        store_genome(root, member)?;
    }
    let mut cache = load_cache(root)?;
    let preexisting = cache
        .iter()
        .filter(|(_, entry)| entry.first_generation < generation)
        .map(|(semantic, _)| semantic.clone())
        .collect::<BTreeSet<_>>();
    let fresh_outputs =
        execute_generation_shards(root, campaign, staging, generation, &members, &cache)?;
    let expected_fresh = members
        .iter()
        .filter(|member| !preexisting.contains(&member.semantic_sha256))
        .count();
    if fresh_outputs.len() != expected_fresh {
        bail!(
            "Binary42 generation {generation} expected {expected_fresh} fresh outputs, got {}",
            fresh_outputs.len()
        );
    }
    for output in fresh_outputs {
        let semantic = output.receipt.semantic_sha256.clone();
        if preexisting.contains(&semantic) {
            bail!("Binary42 worker returned an already-cached semantic genome");
        }
        let imported =
            import_fresh_output(root, output, test, !campaign.config.shard_sizes.is_empty())?;
        cache.insert(semantic, imported);
    }
    let mut slots = Vec::with_capacity(POPULATION);
    for (slot, member) in members.iter().enumerate() {
        let cached = cache
            .get(&member.semantic_sha256)
            .context("Binary42 semantic cache is missing a generation member")?;
        slots.push(LogicalSlot {
            generation,
            slot,
            assigned_node: assigned_node_for_slot(&shard_sizes, slot)?.to_string(),
            member: member.clone(),
            cache_disposition: if preexisting.contains(&member.semantic_sha256) {
                "reused".to_string()
            } else {
                "fresh".to_string()
            },
            cache_source_generation: preexisting
                .contains(&member.semantic_sha256)
                .then_some(cached.first_generation),
            cache_source_slot: preexisting
                .contains(&member.semantic_sha256)
                .then_some(cached.first_slot),
            evaluation_node: cached.evaluation_node.clone(),
            scq_job_id: cached.scq_job_id.clone(),
            receipt_sha256: cached.receipt_sha256.clone(),
            prediction_sha256: cached.prediction_sha256.clone(),
            score: cached.score.clone(),
            failure: cached.failure.clone(),
            runtime_seconds: cached.runtime_seconds,
        });
    }
    if slots.len() != POPULATION {
        bail!("Binary42 generation barrier has a short logical slot ledger");
    }
    let mut ranked = slots
        .iter()
        .filter(|slot| slot.score.is_some())
        .cloned()
        .collect::<Vec<_>>();
    ranked.sort_by(rank_slots);
    if ranked.len() < PARENT_POOL {
        bail!(
            "Binary42 generation {generation} has only {} valid candidates",
            ranked.len()
        );
    }
    let best = ranked
        .first()
        .context("Binary42 generation has no best slot")?;
    let generation_zero_best_auc = if generation == 0 {
        best.score.as_ref().unwrap().auc
    } else {
        previous
            .context("Binary42 nonzero generation has no previous checkpoint")?
            .generation_zero_best_auc
    };
    let qualifying = qualifying_winner_for_policy(
        generation,
        &ranked,
        seed_semantics,
        &campaign.dataset_binding.win_policy,
        campaign.dataset_binding.leader_auc,
        generation_zero_best_auc,
    );
    let reached_cap = generation_cap_reached(&campaign.config, generation);
    let next_population = if qualifying.is_some() || reached_cap {
        Vec::new()
    } else {
        let parents = ranked
            .iter()
            .map(|slot| {
                let score = slot.score.as_ref().unwrap();
                BinaryAdaptiveRanked {
                    canonical_genome: slot.member.canonical_genome.clone(),
                    semantic_sha256: slot.member.semantic_sha256.clone(),
                    complexity: slot.member.complexity,
                    auc: score.auc,
                    logloss: score.logloss,
                }
            })
            .collect::<Vec<_>>();
        test_adaptive_next_population(
            RNG_SEED,
            generation + 1,
            &parents,
            campaign.dataset_binding.features,
        )?
        .into_iter()
        .map(MemberRecord::from_binary)
        .collect::<Result<Vec<_>>>()?
    };
    let previous_checkpoint_sha256 = previous
        .map(|checkpoint| canonical_json(checkpoint).map(|bytes| sha256_bytes(&bytes)))
        .transpose()?;
    let checkpoint = GenerationCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation,
        previous_checkpoint_sha256,
        logical_slots: POPULATION,
        fresh_evaluations: expected_fresh,
        cache_reuses: POPULATION - expected_fresh,
        valid_scores: ranked.len(),
        generation_zero_best_auc,
        best_slot: best.slot,
        best_auc: best.score.as_ref().unwrap().auc,
        slots,
        ranked_semantics: ranked
            .iter()
            .map(|slot| slot.member.semantic_sha256.clone())
            .collect(),
        next_population,
        qualifying_winner: qualifying,
    };
    write_json(&generation_dir.join("checkpoint.v1.json"), &checkpoint)?;
    eprintln!(
        "Binary42 generation {} complete: fresh={}, cache={}, best_auc={:.17}, \
         leader_auc={:.17}, generation_zero_best_auc={:.17}, qualifying_winner={}",
        generation,
        checkpoint.fresh_evaluations,
        checkpoint.cache_reuses,
        checkpoint.best_auc,
        campaign.dataset_binding.leader_auc,
        checkpoint.generation_zero_best_auc,
        checkpoint.qualifying_winner.is_some()
    );
    if generation == 0 {
        let baseline = BaselineManifest {
            schema: BASELINE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            dataset: campaign.dataset.clone(),
            leader_method: campaign.dataset_binding.leader_method.clone(),
            leader_auc: campaign.dataset_binding.leader_auc,
            leader_report_path: campaign.dataset_binding.leader_report_path.clone(),
            leader_report_sha256: campaign.dataset_binding.leader_report_sha256.clone(),
            generation_zero_best_auc: Some(generation_zero_best_auc),
        };
        write_json(
            &root
                .join("baselines")
                .join("generation-zero-baseline.v1.json"),
            &baseline,
        )?;
    }
    Ok(checkpoint)
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

fn load_checkpoints(root: &Path) -> Result<Vec<GenerationCheckpoint>> {
    let generations_root = root.join("generations");
    if !generations_root.exists() {
        return Ok(Vec::new());
    }
    let mut generation_numbers = Vec::new();
    for entry in fs::read_dir(&generations_root)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            bail!("Binary42 generation directory has a non-UTF8 name");
        };
        let Some(raw) = name.strip_prefix("generation-") else {
            bail!("unexpected entry under Binary42 generations");
        };
        generation_numbers.push(raw.parse::<usize>().context("parse generation number")?);
    }
    generation_numbers.sort_unstable();
    if generation_numbers
        .iter()
        .enumerate()
        .any(|(expected, actual)| expected != *actual)
    {
        bail!("Binary42 generation sequence contains a gap");
    }
    let mut checkpoints = Vec::new();
    for (position, generation) in generation_numbers.iter().copied().enumerate() {
        let path = generation_root(root, generation).join("checkpoint.v1.json");
        if !path.exists() {
            if position + 1 == generation_numbers.len() {
                break;
            }
            bail!("Binary42 generation {generation} lacks a completed checkpoint");
        }
        let bytes = fs::read(&path)?;
        let checkpoint: GenerationCheckpoint = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.schema != CHECKPOINT_SCHEMA
            || !checkpoint.test_adaptive
            || checkpoint.benchmark_valid
            || checkpoint.generation != generation
            || checkpoint.logical_slots != POPULATION
            || checkpoint.slots.len() != POPULATION
            || checkpoint
                .slots
                .iter()
                .enumerate()
                .any(|(slot, record)| record.generation != generation || record.slot != slot)
        {
            bail!("Binary42 generation {generation} checkpoint is invalid");
        }
        let expected_previous = checkpoints
            .last()
            .map(|prior| canonical_json(prior).map(|bytes| sha256_bytes(&bytes)))
            .transpose()?;
        if checkpoint.previous_checkpoint_sha256 != expected_previous {
            bail!("Binary42 checkpoint hash chain mismatch at generation {generation}");
        }
        checkpoints.push(checkpoint);
    }
    Ok(checkpoints)
}

fn atomic_rebuild(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    if temporary.exists() {
        fs::remove_file(&temporary)?;
    }
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)?;
    Ok(())
}

fn evaluation_records(checkpoints: &[GenerationCheckpoint]) -> Vec<EvaluationIndexRecord> {
    checkpoints
        .iter()
        .flat_map(|checkpoint| &checkpoint.slots)
        .map(|slot| EvaluationIndexRecord {
            test_adaptive: true,
            benchmark_valid: false,
            generation: slot.generation,
            slot: slot.slot,
            semantic_sha256: slot.member.semantic_sha256.clone(),
            ancestry_operation: slot.member.ancestry.operation.clone(),
            ancestry_parents: slot.member.ancestry.parents.clone(),
            cache_disposition: slot.cache_disposition.clone(),
            assigned_node: slot.assigned_node.clone(),
            evaluation_node: slot.evaluation_node.clone(),
            scq_job_id: slot.scq_job_id.clone(),
            auc: slot.score.as_ref().map(|score| score.auc),
            logloss: slot.score.as_ref().map(|score| score.logloss),
            complexity: slot.member.complexity,
            receipt_sha256: slot.receipt_sha256.clone(),
            prediction_sha256: slot.prediction_sha256.clone(),
            failure: slot.failure.clone(),
            runtime_seconds: slot.runtime_seconds,
        })
        .collect()
}

fn csv_field(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn rebuild_indexes(
    root: &Path,
    checkpoints: &[GenerationCheckpoint],
    extended: bool,
) -> Result<()> {
    let records = evaluation_records(checkpoints);
    let mut jsonl = Vec::new();
    for record in &records {
        jsonl.extend_from_slice(&canonical_json(record)?);
    }
    atomic_rebuild(&root.join("evaluations.jsonl"), &jsonl)?;
    let mut csv = if extended {
        b"test_adaptive,benchmark_valid,generation,slot,semantic_sha256,ancestry_operation,ancestry_parents,cache_disposition,assigned_node,evaluation_node,scq_job_id,auc,logloss,complexity,runtime_seconds,receipt_sha256,prediction_sha256,failure\n".to_vec()
    } else {
        b"test_adaptive,benchmark_valid,generation,slot,semantic_sha256,ancestry_operation,ancestry_parents,cache_disposition,assigned_node,evaluation_node,scq_job_id,auc,logloss,complexity,receipt_sha256,prediction_sha256,failure\n".to_vec()
    };
    for record in &records {
        let row = if extended {
            format!(
                "true,false,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                record.generation,
                record.slot,
                csv_field(&record.semantic_sha256),
                csv_field(&record.ancestry_operation),
                csv_field(&record.ancestry_parents.join(";")),
                csv_field(&record.cache_disposition),
                csv_field(&record.assigned_node),
                csv_field(&record.evaluation_node),
                csv_field(&record.scq_job_id),
                record
                    .auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record
                    .logloss
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record.complexity,
                record
                    .runtime_seconds
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                csv_field(&record.receipt_sha256),
                csv_field(&record.prediction_sha256),
                csv_field(record.failure.as_deref().unwrap_or("")),
            )
        } else {
            format!(
                "true,false,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                record.generation,
                record.slot,
                csv_field(&record.semantic_sha256),
                csv_field(&record.ancestry_operation),
                csv_field(&record.ancestry_parents.join(";")),
                csv_field(&record.cache_disposition),
                csv_field(&record.assigned_node),
                csv_field(&record.evaluation_node),
                csv_field(&record.scq_job_id),
                record
                    .auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record
                    .logloss
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                record.complexity,
                csv_field(&record.receipt_sha256),
                csv_field(&record.prediction_sha256),
                csv_field(record.failure.as_deref().unwrap_or("")),
            )
        };
        csv.extend_from_slice(row.as_bytes());
    }
    atomic_rebuild(&root.join("evaluations.csv"), &csv)?;
    if extended {
        let mut genomes = BTreeMap::<String, GenomeIndexRecord>::new();
        for slot in checkpoints.iter().flat_map(|checkpoint| &checkpoint.slots) {
            genomes
                .entry(slot.member.semantic_sha256.clone())
                .or_insert_with(|| GenomeIndexRecord {
                    semantic_sha256: slot.member.semantic_sha256.clone(),
                    canonical_genome: slot.member.canonical_genome.clone(),
                    complexity: slot.member.complexity,
                    ancestry_operation: slot.member.ancestry.operation.clone(),
                    ancestry_parents: slot.member.ancestry.parents.clone(),
                    first_generation: slot.generation,
                    first_slot: slot.slot,
                });
        }
        let mut genomes_jsonl = Vec::new();
        let mut genomes_csv = b"semantic_sha256,canonical_genome,complexity,ancestry_operation,ancestry_parents,first_generation,first_slot\n".to_vec();
        for genome in genomes.values() {
            genomes_jsonl.extend_from_slice(&canonical_json(genome)?);
            genomes_csv.extend_from_slice(
                format!(
                    "{},{},{},{},{},{},{}\n",
                    csv_field(&genome.semantic_sha256),
                    csv_field(&genome.canonical_genome),
                    genome.complexity,
                    csv_field(&genome.ancestry_operation),
                    csv_field(&genome.ancestry_parents.join(";")),
                    genome.first_generation,
                    genome.first_slot,
                )
                .as_bytes(),
            );
        }
        atomic_rebuild(&root.join("genomes.jsonl"), &genomes_jsonl)?;
        atomic_rebuild(&root.join("genomes.csv"), &genomes_csv)?;
    }
    let mut auc_sorted = records
        .iter()
        .filter(|record| record.auc.is_some())
        .collect::<Vec<_>>();
    auc_sorted.sort_by(|left, right| {
        right
            .auc
            .unwrap()
            .total_cmp(&left.auc.unwrap())
            .then_with(|| left.logloss.unwrap().total_cmp(&right.logloss.unwrap()))
            .then_with(|| left.complexity.cmp(&right.complexity))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
            .then_with(|| left.generation.cmp(&right.generation))
            .then_with(|| left.slot.cmp(&right.slot))
    });
    let index = json!({
        "schema": "JopeBinary42TestAdaptiveAucIndex.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "records": &auc_sorted,
    });
    atomic_rebuild(
        &root.join("indexes").join("auc-sorted.v1.json"),
        &canonical_json(&index)?,
    )?;
    if extended {
        let mut auc_jsonl = Vec::new();
        let mut auc_csv = b"generation,slot,semantic_sha256,auc,logloss,complexity,receipt_sha256,prediction_sha256\n".to_vec();
        for record in auc_sorted {
            auc_jsonl.extend_from_slice(&canonical_json(record)?);
            auc_csv.extend_from_slice(
                format!(
                    "{},{},{},{},{},{},{},{}\n",
                    record.generation,
                    record.slot,
                    csv_field(&record.semantic_sha256),
                    record.auc.unwrap(),
                    record.logloss.unwrap(),
                    record.complexity,
                    csv_field(&record.receipt_sha256),
                    csv_field(&record.prediction_sha256),
                )
                .as_bytes(),
            );
        }
        atomic_rebuild(&root.join("indexes").join("auc-sorted.jsonl"), &auc_jsonl)?;
        atomic_rebuild(&root.join("indexes").join("auc-sorted.csv"), &auc_csv)?;
    }
    Ok(())
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

fn winner_seed_path(path: &Path) -> Result<String> {
    Ok(path
        .strip_prefix(workspace_root())?
        .to_string_lossy()
        .into_owned())
}

fn winner_seed_file(path: &str) -> PathBuf {
    workspace_root().join(path)
}

fn matching_confirmation_file(dataset_root: &Path, expected_sha256: &str) -> Result<PathBuf> {
    let confirmations_root = dataset_root.join("winner").join("confirmations");
    let mut attempts = fs::read_dir(&confirmations_root)
        .with_context(|| {
            format!(
                "read Binary42 winner confirmations {}",
                confirmations_root.display()
            )
        })?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    attempts.sort();
    attempts
        .into_iter()
        .map(|attempt| attempt.join("confirmation.v1.json"))
        .find(|path| {
            path.is_file()
                && sha256_file(path)
                    .ok()
                    .as_deref()
                    .is_some_and(|sha256| sha256 == expected_sha256)
        })
        .with_context(|| {
            format!(
                "Binary42 winner confirmation {expected_sha256} is absent under {}",
                confirmations_root.display()
            )
        })
}

fn validate_winner_seed_source(source: &WinnerSeedSource, cohort: &FrozenCohort) -> Result<()> {
    let spec = cohort
        .datasets
        .iter()
        .find(|spec| spec.id == source.source_dataset)
        .context("Binary42 winner seed references an unknown dataset")?;
    let complete_path = winner_seed_file(&source.complete_file);
    let receipt_path = winner_seed_file(&source.final_receipt_file);
    let confirmation_path = winner_seed_file(&source.confirmation_file);
    if sha256_file(&complete_path)? != source.complete_sha256
        || sha256_file(&receipt_path)? != source.final_receipt_sha256
        || sha256_file(&confirmation_path)? != source.confirmation_sha256
    {
        bail!("Binary42 winner seed source hash changed");
    }
    let complete_bytes = fs::read(&complete_path)?;
    let complete: Value = serde_json::from_slice(&complete_bytes)?;
    if complete_bytes != canonical_json(&complete)?
        || complete.get("schema").and_then(Value::as_str) != Some(COMPLETE_SCHEMA)
        || complete.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || complete.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || complete.get("status").and_then(Value::as_str) != Some("complete")
        || complete.get("run_id").and_then(Value::as_str) != Some(source.source_run_id.as_str())
        || complete.get("dataset").and_then(Value::as_str) != Some(source.source_dataset.as_str())
        || complete.get("final_receipt_sha256").and_then(Value::as_str)
            != Some(source.final_receipt_sha256.as_str())
    {
        bail!("Binary42 winner seed RUN_COMPLETE is invalid");
    }
    let receipt_bytes = fs::read(&receipt_path)?;
    let receipt: WinnerReceipt = serde_json::from_slice(&receipt_bytes)?;
    if receipt_bytes != canonical_json(&receipt)?
        || receipt.schema != WINNER_SCHEMA
        || !receipt.test_adaptive
        || receipt.benchmark_valid
        || receipt.run_id != source.source_run_id
        || receipt.dataset != source.source_dataset
        || receipt.generation == 0
        || receipt.semantic_sha256 != source.semantic_sha256
        || receipt.canonical_genome != source.canonical_genome
        || receipt.auc.to_bits() != source.auc_bits
        || receipt.frozen_leader_auc.to_bits() != source.frozen_leader_auc_bits
        || receipt.generation_zero_best_auc.to_bits() != source.generation_zero_best_auc_bits
        || receipt.four_node_confirmation_sha256 != source.confirmation_sha256
        || sha256_bytes(receipt.canonical_genome.as_bytes()) != receipt.semantic_sha256
        || !policy_win(
            &spec.win_policy,
            receipt.auc,
            receipt.frozen_leader_auc,
            receipt.generation_zero_best_auc,
        )
    {
        bail!("Binary42 winner seed final receipt is invalid");
    }
    let confirmation_bytes = fs::read(&confirmation_path)?;
    let confirmation: ConfirmationManifest = serde_json::from_slice(&confirmation_bytes)?;
    if confirmation_bytes != canonical_json(&confirmation)?
        || confirmation.schema != CONFIRMATION_SCHEMA
        || !confirmation.test_adaptive
        || confirmation.benchmark_valid
        || confirmation.run_id != source.source_run_id
        || confirmation.dataset != source.source_dataset
        || confirmation.semantic_sha256 != source.semantic_sha256
        || !confirmation.prediction_hashes_identical
        || !confirmation.auc_bits_identical
        || confirmation.confirmations.len() != NODES.len()
        || confirmation
            .confirmations
            .iter()
            .enumerate()
            .any(|(index, record)| {
                record.node != NODES[index]
                    || record.auc_bits != source.auc_bits
                    || record.prediction_sha256 != confirmation.original_prediction_sha256
            })
    {
        bail!("Binary42 winner seed confirmation is invalid");
    }
    let dataset_root = complete_path
        .parent()
        .context("Binary42 winner seed marker has no dataset directory")?;
    let source_seeds: SeedManifest =
        load_json(&dataset_root.join("seeds").join("generation-zero.v1.json"))?;
    if source_seeds
        .members
        .iter()
        .any(|member| member.semantic_sha256 == source.semantic_sha256)
    {
        bail!("Binary42 winner seed was not an evolved descendant");
    }
    Ok(())
}

fn confirmed_winner_seed_sources(
    current_run_id: &str,
    cohort: &FrozenCohort,
) -> Result<Vec<WinnerSeedSource>> {
    let runs_root = workspace_root().join("runs");
    let current_run = run_root(current_run_id);
    let mut run_roots = fs::read_dir(&runs_root)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    run_roots.sort();
    let mut best_by_dataset = BTreeMap::<String, WinnerSeedSource>::new();
    for run_path in run_roots {
        if run_path == current_run
            || !run_path.is_dir()
            || !run_path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("binary42-test-adaptive-"))
        {
            continue;
        }
        let mut dataset_roots = fs::read_dir(&run_path)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_type().ok()?.is_dir().then_some(entry.path()))
            .collect::<Vec<_>>();
        dataset_roots.sort();
        for dataset_root in dataset_roots {
            let complete_path = dataset_root.join("RUN_COMPLETE");
            let receipt_path = dataset_root.join("winner").join("final-receipt.v1.json");
            if !complete_path.is_file() || !receipt_path.is_file() {
                continue;
            }
            let receipt_bytes = fs::read(&receipt_path)?;
            let receipt: WinnerReceipt = serde_json::from_slice(&receipt_bytes)?;
            let confirmation_path =
                matching_confirmation_file(&dataset_root, &receipt.four_node_confirmation_sha256)?;
            let source = WinnerSeedSource {
                source_run_id: receipt.run_id.clone(),
                source_dataset: receipt.dataset.clone(),
                complete_file: winner_seed_path(&complete_path)?,
                complete_sha256: sha256_file(&complete_path)?,
                final_receipt_file: winner_seed_path(&receipt_path)?,
                final_receipt_sha256: sha256_bytes(&receipt_bytes),
                confirmation_file: winner_seed_path(&confirmation_path)?,
                confirmation_sha256: sha256_file(&confirmation_path)?,
                canonical_genome: receipt.canonical_genome.clone(),
                semantic_sha256: receipt.semantic_sha256.clone(),
                auc_bits: receipt.auc.to_bits(),
                frozen_leader_auc_bits: receipt.frozen_leader_auc.to_bits(),
                generation_zero_best_auc_bits: receipt.generation_zero_best_auc.to_bits(),
            };
            validate_winner_seed_source(&source, cohort).with_context(|| {
                format!(
                    "validate Binary42 winner seed {} from {}",
                    source.source_dataset,
                    receipt_path.display()
                )
            })?;
            match best_by_dataset.get(&source.source_dataset) {
                Some(incumbent)
                    if f64::from_bits(incumbent.auc_bits) > f64::from_bits(source.auc_bits)
                        || (incumbent.auc_bits == source.auc_bits
                            && incumbent.semantic_sha256 <= source.semantic_sha256) => {}
                _ => {
                    best_by_dataset.insert(source.source_dataset.clone(), source);
                }
            }
        }
    }
    let frozen_order = cohort
        .datasets
        .iter()
        .map(|spec| (spec.id.as_str(), spec.order))
        .collect::<BTreeMap<_, _>>();
    let mut sources = best_by_dataset.into_values().collect::<Vec<_>>();
    sources.sort_by(|left, right| {
        frozen_order
            .get(left.source_dataset.as_str())
            .cmp(&frozen_order.get(right.source_dataset.as_str()))
            .then_with(|| left.source_dataset.cmp(&right.source_dataset))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    let mut seen = BTreeSet::new();
    sources.retain(|source| seen.insert(source.semantic_sha256.clone()));
    if sources.len() > cohort.datasets.len() || sources.len() > POPULATION {
        bail!("Binary42 confirmed winner seed snapshot is unexpectedly large");
    }
    Ok(sources)
}

fn validate_winner_seed_snapshot(
    sources: &[WinnerSeedSource],
    cohort: &FrozenCohort,
) -> Result<()> {
    if sources.len() > cohort.datasets.len() || sources.len() > POPULATION {
        bail!("Binary42 confirmed winner seed snapshot is unexpectedly large");
    }
    let frozen_order = cohort
        .datasets
        .iter()
        .map(|spec| (spec.id.as_str(), spec.order))
        .collect::<BTreeMap<_, _>>();
    let mut datasets = BTreeSet::new();
    let mut semantics = BTreeSet::new();
    let mut previous_order = None;
    for source in sources {
        validate_winner_seed_source(source, cohort)?;
        if !datasets.insert(source.source_dataset.clone())
            || !semantics.insert(source.semantic_sha256.clone())
        {
            bail!("Binary42 winner seed snapshot is not unique");
        }
        let order = *frozen_order
            .get(source.source_dataset.as_str())
            .context("Binary42 winner seed source has no frozen order")?;
        if previous_order.is_some_and(|previous| previous >= order) {
            bail!("Binary42 winner seed snapshot is not in frozen dataset order");
        }
        previous_order = Some(order);
    }
    Ok(())
}

fn confirmation_attempt(root: &Path) -> Result<usize> {
    let confirmation_root = root.join("winner").join("confirmations");
    if !confirmation_root.exists() {
        return Ok(0);
    }
    Ok(fs::read_dir(confirmation_root)?.count())
}

fn store_confirmation_output(
    root: &Path,
    output: FreshOutput,
    test: &RegressionData,
) -> Result<ConfirmationRecord> {
    let score =
        validate_fresh_output(&output, test)?.context("Binary42 confirmation candidate failed")?;
    let receipt_sha256 = sha256_bytes(&output.receipt_bytes);
    let prediction_sha256 = sha256_bytes(&output.prediction_bytes);
    immutable_or_equal(
        &root
            .join("objects")
            .join("receipts")
            .join(format!("{receipt_sha256}.json")),
        &output.receipt_bytes,
    )?;
    immutable_or_equal(
        &root
            .join("objects")
            .join("predictions")
            .join(format!("{prediction_sha256}.csv")),
        &output.prediction_bytes,
    )?;
    Ok(ConfirmationRecord {
        node: output.receipt.node,
        job_id: output.receipt.scq_job_id,
        receipt_sha256,
        prediction_sha256,
        auc: score.auc,
        logloss: score.logloss,
        auc_bits: score.auc.to_bits(),
    })
}

fn confirm_winner(
    root: &Path,
    campaign: &CampaignManifest,
    staging: &StagingReceipt,
    checkpoint: &GenerationCheckpoint,
    test: &RegressionData,
) -> Result<ConfirmationManifest> {
    let selected = checkpoint
        .qualifying_winner
        .as_ref()
        .context("Binary42 checkpoint has no qualifying winner")?;
    let slot = checkpoint
        .slots
        .get(selected.slot)
        .filter(|slot| slot.member.semantic_sha256 == selected.semantic_sha256)
        .context("Binary42 qualifying winner slot mismatch")?;
    let attempt = confirmation_attempt(root)?;
    let manifests = staging
        .nodes
        .iter()
        .enumerate()
        .map(|(node_index, staged)| {
            shard_manifest(
                campaign,
                staged,
                selected.generation,
                node_index,
                attempt * 3,
                "confirmation",
                std::slice::from_ref(&slot.member),
                &HashMap::new(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let results = Arc::new(Mutex::new(Vec::<(
        usize,
        std::result::Result<Vec<FreshOutput>, anyhow::Error>,
    )>::new()));
    thread::scope(|scope| {
        for (index, manifest) in manifests.iter().enumerate() {
            let results = Arc::clone(&results);
            scope.spawn(move || {
                results
                    .lock()
                    .unwrap()
                    .push((index, execute_shard(root, manifest)));
            });
        }
    });
    let mut results = Arc::try_unwrap(results)
        .expect("Binary42 confirmation result references remain")
        .into_inner()
        .unwrap();
    results.sort_by_key(|(index, _)| *index);
    let mut confirmations = Vec::new();
    for (index, result) in results {
        let mut outputs = match result {
            Ok(outputs) => outputs,
            Err(error) => {
                record_infrastructure_failure(root, &manifests[index], &error)?;
                let mut recovered = None;
                let mut final_error = error;
                for retry_attempt in 1..=2 {
                    let retry = shard_manifest(
                        campaign,
                        &staging.nodes[index],
                        selected.generation,
                        index,
                        attempt * 3 + retry_attempt,
                        "confirmation",
                        std::slice::from_ref(&slot.member),
                        &HashMap::new(),
                    )?;
                    match execute_shard(root, &retry) {
                        Ok(outputs) => {
                            recovered = Some(outputs);
                            break;
                        }
                        Err(error) => {
                            record_infrastructure_failure(root, &retry, &error)?;
                            final_error = error;
                        }
                    }
                }
                if recovered.is_none() {
                    let failure = json!({
                        "schema": "JopeBinary42TestAdaptiveConfirmationInfrastructureFailure.v1",
                        "test_adaptive": true,
                        "benchmark_valid": false,
                        "run_id": campaign.run_id,
                        "dataset": campaign.dataset,
                        "generation": selected.generation,
                        "node": manifests[index].node,
                        "attempt": attempt,
                        "error": format!("{final_error:#}"),
                    });
                    write_json(
                        &root
                            .join("winner")
                            .join("confirmations")
                            .join(format!("attempt-{attempt:03}"))
                            .join("INFRASTRUCTURE_FAILED.v1.json"),
                        &failure,
                    )?;
                }
                recovered.with_context(|| {
                    format!(
                        "Binary42 confirmation failed on {} after two retries: {final_error:#}",
                        manifests[index].node
                    )
                })?
            }
        };
        if outputs.len() != 1 {
            bail!("Binary42 confirmation job returned the wrong candidate count");
        }
        confirmations.push(store_confirmation_output(root, outputs.remove(0), test)?);
    }
    if confirmations
        .iter()
        .enumerate()
        .any(|(index, confirmation)| confirmation.node != NODES[index])
    {
        bail!("Binary42 confirmation node order mismatch");
    }
    let prediction_hashes_identical = confirmations
        .iter()
        .all(|confirmation| confirmation.prediction_sha256 == slot.prediction_sha256);
    let auc_bits_identical = confirmations
        .iter()
        .all(|confirmation| confirmation.auc_bits == selected.auc.to_bits());
    let confirmation = ConfirmationManifest {
        schema: CONFIRMATION_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation: selected.generation,
        selected_slot: selected.slot,
        semantic_sha256: selected.semantic_sha256.clone(),
        attempt,
        original_prediction_sha256: slot.prediction_sha256.clone(),
        confirmations,
        prediction_hashes_identical,
        auc_bits_identical,
    };
    let attempt_root = root
        .join("winner")
        .join("confirmations")
        .join(format!("attempt-{attempt:03}"));
    if !prediction_hashes_identical || !auc_bits_identical {
        write_json(
            &attempt_root.join("CONFIRMATION_FAILED.v1.json"),
            &confirmation,
        )?;
        bail!("Binary42 four-node confirmation diverged; campaign retained for resume");
    }
    write_json(&attempt_root.join("confirmation.v1.json"), &confirmation)?;
    Ok(confirmation)
}

fn lineage_for(
    checkpoints: &[GenerationCheckpoint],
    winner_semantic: &str,
) -> Result<Vec<LineageRecord>> {
    let mut members = BTreeMap::<String, MemberRecord>::new();
    for slot in checkpoints.iter().flat_map(|checkpoint| &checkpoint.slots) {
        members
            .entry(slot.member.semantic_sha256.clone())
            .or_insert_with(|| slot.member.clone());
    }
    fn visit(
        semantic: &str,
        members: &BTreeMap<String, MemberRecord>,
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
        lineage: &mut Vec<LineageRecord>,
    ) -> Result<()> {
        if visited.contains(semantic) {
            return Ok(());
        }
        if !visiting.insert(semantic.to_string()) {
            bail!("Binary42 lineage contains a cycle");
        }
        let member = members
            .get(semantic)
            .with_context(|| format!("Binary42 lineage parent {semantic} is absent"))?;
        for parent in &member.ancestry.parents {
            if parent != semantic {
                visit(parent, members, visiting, visited, lineage)?;
            }
        }
        visiting.remove(semantic);
        visited.insert(semantic.to_string());
        lineage.push(LineageRecord {
            semantic_sha256: member.semantic_sha256.clone(),
            canonical_genome: member.canonical_genome.clone(),
            complexity: member.complexity,
            ancestry: member.ancestry.clone(),
        });
        Ok(())
    }
    let mut lineage = Vec::new();
    visit(
        winner_semantic,
        &members,
        &mut BTreeSet::new(),
        &mut BTreeSet::new(),
        &mut lineage,
    )?;
    Ok(lineage)
}

fn collect_artifacts(root: &Path) -> Result<Vec<WinnerArtifact>> {
    fn walk(root: &Path, current: &Path, artifacts: &mut Vec<WinnerArtifact>) -> Result<()> {
        let mut entries = fs::read_dir(current)?
            .map(|entry| Ok(entry?.path()))
            .collect::<Result<Vec<_>>>()?;
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(root, &path, artifacts)?;
            } else {
                let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
                if relative == "RUN_COMPLETE" || relative == "winner/artifact-manifest.v1.json" {
                    continue;
                }
                artifacts.push(WinnerArtifact {
                    path: relative,
                    sha256: sha256_file(&path)?,
                });
            }
        }
        Ok(())
    }
    let mut artifacts = Vec::new();
    walk(root, root, &mut artifacts)?;
    artifacts.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(artifacts)
}

fn publish_winner(
    root: &Path,
    campaign: &CampaignManifest,
    checkpoints: &[GenerationCheckpoint],
    confirmation: &ConfirmationManifest,
) -> Result<()> {
    let checkpoint = checkpoints
        .last()
        .context("Binary42 winner publication has no checkpoint")?;
    let selected = checkpoint
        .qualifying_winner
        .as_ref()
        .context("Binary42 final checkpoint has no qualifying winner")?;
    let slot = checkpoint
        .slots
        .get(selected.slot)
        .context("Binary42 winner slot is absent")?;
    let winner_root = root.join("winner");
    let genome = GenomeObject {
        schema: GENOME_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        semantic_sha256: slot.member.semantic_sha256.clone(),
        canonical_genome: slot.member.canonical_genome.clone(),
        complexity: slot.member.complexity,
    };
    write_json(&winner_root.join("genome.v1.json"), &genome)?;
    let prediction_object = root
        .join("objects")
        .join("predictions")
        .join(format!("{}.csv", slot.prediction_sha256));
    immutable_or_equal(
        &winner_root.join("predictions.csv"),
        &fs::read(&prediction_object)?,
    )?;
    let lineage = lineage_for(checkpoints, &selected.semantic_sha256)?;
    let lineage_value = json!({
        "schema": "JopeBinary42TestAdaptiveLineage.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "winner_semantic_sha256": selected.semantic_sha256,
        "records": lineage,
    });
    let lineage_sha256 = write_json(&winner_root.join("lineage.v1.json"), &lineage_value)?;
    let confirmation_attempt_root = winner_root
        .join("confirmations")
        .join(format!("attempt-{:03}", confirmation.attempt));
    let confirmation_sha256 = sha256_file(&confirmation_attempt_root.join("confirmation.v1.json"))?;
    let receipt = WinnerReceipt {
        schema: WINNER_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation: selected.generation,
        slot: selected.slot,
        semantic_sha256: selected.semantic_sha256.clone(),
        canonical_genome: slot.member.canonical_genome.clone(),
        complexity: selected.complexity,
        auc: selected.auc,
        logloss: selected.logloss,
        frozen_leader_auc: campaign.dataset_binding.leader_auc,
        generation_zero_best_auc: checkpoint.generation_zero_best_auc,
        margin_vs_frozen_leader: selected.auc - campaign.dataset_binding.leader_auc,
        margin_vs_generation_zero: selected.auc - checkpoint.generation_zero_best_auc,
        prediction_sha256: slot.prediction_sha256.clone(),
        source_receipt_sha256: slot.receipt_sha256.clone(),
        four_node_confirmation_sha256: confirmation_sha256,
        lineage_sha256,
        release_binary_sha256: campaign.release_binary_sha256.clone(),
    };
    let final_receipt_sha256 = write_json(&winner_root.join("final-receipt.v1.json"), &receipt)?;
    let artifact_manifest = WinnerArtifactManifest {
        schema: "JopeBinary42TestAdaptiveArtifactManifest.v1".to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        artifacts: collect_artifacts(root)?,
    };
    let artifact_manifest_sha256 = write_json(
        &winner_root.join("artifact-manifest.v1.json"),
        &artifact_manifest,
    )?;
    let complete = json!({
        "schema": COMPLETE_SCHEMA,
        "test_adaptive": true,
        "benchmark_valid": false,
        "status": "complete",
        "run_id": campaign.run_id,
        "dataset": campaign.dataset,
        "winner_semantic_sha256": selected.semantic_sha256,
        "winner_auc": selected.auc,
        "final_receipt_sha256": final_receipt_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "completed_at_unix_ms": unix_millis(),
    });
    write_json(&root.join("RUN_COMPLETE"), &complete)?;
    Ok(())
}

fn publish_exhausted(
    root: &Path,
    campaign: &CampaignManifest,
    checkpoints: &[GenerationCheckpoint],
) -> Result<()> {
    if checkpoints.len() != ALL_MAX_GENERATIONS
        || checkpoints
            .iter()
            .enumerate()
            .any(|(generation, checkpoint)| {
                checkpoint.generation != generation || checkpoint.qualifying_winner.is_some()
            })
    {
        bail!("Binary42 exhaustion requires exactly generations 0-9 with no qualifying winner");
    }
    if generation_root(root, ALL_MAX_GENERATIONS).exists() {
        bail!("Binary42 exhaustion found a forbidden generation 10 directory");
    }
    let mut ranked = checkpoints
        .iter()
        .flat_map(|checkpoint| &checkpoint.slots)
        .filter(|slot| slot.score.is_some())
        .cloned()
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        rank_slots(left, right)
            .then_with(|| left.generation.cmp(&right.generation))
            .then_with(|| left.slot.cmp(&right.slot))
    });
    let best = ranked
        .first()
        .context("Binary42 exhausted campaign has no successful candidate")?;
    let score = best.score.as_ref().unwrap();
    let best_effort = root.join("best-effort");
    let genome = GenomeObject {
        schema: GENOME_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        semantic_sha256: best.member.semantic_sha256.clone(),
        canonical_genome: best.member.canonical_genome.clone(),
        complexity: best.member.complexity,
    };
    write_json(&best_effort.join("genome.v1.json"), &genome)?;
    let prediction_object = root
        .join("objects")
        .join("predictions")
        .join(format!("{}.csv", best.prediction_sha256));
    immutable_or_equal(
        &best_effort.join("predictions.csv"),
        &fs::read(&prediction_object)?,
    )?;
    let lineage = lineage_for(checkpoints, &best.member.semantic_sha256)?;
    let lineage_value = json!({
        "schema": "JopeBinary42TestAdaptiveLineage.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "winner_semantic_sha256": best.member.semantic_sha256,
        "records": lineage,
    });
    let lineage_sha256 = write_json(&best_effort.join("lineage.v1.json"), &lineage_value)?;
    let receipt = ExhaustedReceipt {
        schema: EXHAUSTED_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        status: "exhausted".to_string(),
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        completed_generations: checkpoints.len(),
        logical_slots: checkpoints.len() * POPULATION,
        best_generation: best.generation,
        best_slot: best.slot,
        best_semantic_sha256: best.member.semantic_sha256.clone(),
        best_auc: score.auc,
        best_logloss: score.logloss,
        best_prediction_sha256: best.prediction_sha256.clone(),
        best_receipt_sha256: best.receipt_sha256.clone(),
        lineage_sha256,
        release_binary_sha256: campaign.release_binary_sha256.clone(),
        completed_at_unix_ms: unix_millis(),
    };
    let receipt_sha256 = write_json(&best_effort.join("final-receipt.v1.json"), &receipt)?;
    let terminal = ExhaustedMarker {
        schema: EXHAUSTED_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        status: "exhausted".to_string(),
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        completed_generations: ALL_MAX_GENERATIONS,
        logical_slots: ALL_MAX_GENERATIONS * POPULATION,
        best_semantic_sha256: best.member.semantic_sha256.clone(),
        best_auc: score.auc,
        final_receipt_sha256: receipt_sha256,
        completed_at_unix_ms: unix_millis(),
    };
    write_json(&root.join("RUN_EXHAUSTED"), &terminal)?;
    Ok(())
}

fn existing_confirmation(root: &Path) -> Result<Option<ConfirmationManifest>> {
    let confirmation_root = root.join("winner").join("confirmations");
    if !confirmation_root.exists() {
        return Ok(None);
    }
    let mut attempts = fs::read_dir(confirmation_root)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    attempts.sort();
    for attempt in attempts.into_iter().rev() {
        let path = attempt.join("confirmation.v1.json");
        if path.exists() {
            let confirmation: ConfirmationManifest = load_json(&path)?;
            if confirmation.schema != CONFIRMATION_SCHEMA
                || !confirmation.test_adaptive
                || confirmation.benchmark_valid
                || !confirmation.prediction_hashes_identical
                || !confirmation.auc_bits_identical
                || confirmation.confirmations.len() != NODES.len()
            {
                bail!("existing Binary42 confirmation is invalid");
            }
            return Ok(Some(confirmation));
        }
    }
    Ok(None)
}

fn initialize_campaign(
    run_id: &str,
    dataset: &str,
    resume: bool,
) -> Result<(PathBuf, CampaignManifest, StagingReceipt)> {
    validate_run_id(run_id)?;
    let (spec, cohort_sha256) = selected_dataset(dataset)?;
    verify_dataset_binding(&spec)?;
    let root = campaign_root(run_id, dataset);
    let campaign_path = root.join("campaign").join("campaign.v1.json");
    let release_binary_sha256 = current_binary_sha256()?;
    let scq_binary_sha256 = sha256_file(Path::new(scq_binary_str()))?;
    let campaign = if campaign_path.exists() {
        if !resume && !root.join("RUN_COMPLETE").exists() {
            bail!("Binary42 campaign already exists; pass --resume");
        }
        let recorded: CampaignManifest = load_json(&campaign_path)?;
        if recorded.schema != CAMPAIGN_SCHEMA
            || !recorded.test_adaptive
            || recorded.benchmark_valid
            || recorded.run_id != run_id
            || recorded.dataset != dataset
            || recorded.frozen_cohort_sha256 != cohort_sha256
            || recorded.binary_seed_sha256 != SEED_BINARY_SHA256
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_binary_sha256 != scq_binary_sha256
            || recorded.dataset_binding != spec
            || recorded.config != CampaignConfig::fixed()
            || recorded.cohort_staging_sha256.is_some()
            || recorded.winner_seed_snapshot_sha256.is_some()
        {
            bail!("Binary42 resume identity changed");
        }
        recorded
    } else {
        if resume {
            bail!("cannot resume a Binary42 campaign that does not exist");
        }
        fs::create_dir_all(&root)?;
        immutable_write(
            &root.join("campaign").join("binary42-frozen-cohort.v1.json"),
            COHORT_BYTES,
        )?;
        let created = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            dataset: dataset.to_string(),
            frozen_cohort_file: "campaign/binary42-frozen-cohort.v1.json".to_string(),
            frozen_cohort_sha256: cohort_sha256,
            binary_seed_sha256: SEED_BINARY_SHA256.to_string(),
            release_binary_sha256,
            scq_binary_sha256,
            dataset_binding: spec,
            config: CampaignConfig::fixed(),
            started_at_unix_ms: unix_millis(),
            cohort_staging_sha256: None,
            winner_seed_snapshot_sha256: None,
        };
        write_json(&campaign_path, &created)?;
        let baseline = BaselineManifest {
            schema: BASELINE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            dataset: dataset.to_string(),
            leader_method: created.dataset_binding.leader_method.clone(),
            leader_auc: created.dataset_binding.leader_auc,
            leader_report_path: created.dataset_binding.leader_report_path.clone(),
            leader_report_sha256: created.dataset_binding.leader_report_sha256.clone(),
            generation_zero_best_auc: None,
        };
        write_json(
            &root.join("baselines").join("frozen-leader.v1.json"),
            &baseline,
        )?;
        let members = seed_population(created.dataset_binding.features)?;
        let seeds = SeedManifest {
            schema: SEED_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            seed_file_sha256: SEED_BINARY_SHA256.to_string(),
            population: POPULATION,
            members,
            winner_seed_sources: Vec::new(),
        };
        write_json(&root.join("seeds").join("generation-zero.v1.json"), &seeds)?;
        created
    };
    let staging = stage_campaign(
        &root,
        &campaign.dataset_binding,
        &campaign.release_binary_sha256,
        &campaign.scq_binary_sha256,
    )?;
    Ok((root, campaign, staging))
}

fn dataset_staging_from_cohort(
    spec: &FrozenDataset,
    cohort_staging: &AllStagingReceipt,
    cohort_staging_sha256: &str,
) -> Result<StagingReceipt> {
    let mut nodes = Vec::with_capacity(NODES.len());
    for staged_node in &cohort_staging.nodes {
        let dataset = staged_node
            .datasets
            .iter()
            .find(|dataset| dataset.dataset == spec.id)
            .with_context(|| format!("Binary42 cohort staging omits dataset {}", spec.id))?;
        nodes.push(StagedNode {
            node: staged_node.node.clone(),
            hostname: staged_node.hostname.clone(),
            bundle_root: staged_node.bundle_root.clone(),
            binary_path: staged_node.binary_path.clone(),
            binary_sha256: staged_node.binary_sha256.clone(),
            train_path: dataset.train_path.clone(),
            train_sha256: dataset.train_sha256.clone(),
            test_path: dataset.test_path.clone(),
            test_sha256: dataset.test_sha256.clone(),
            canary_job_id: staged_node.canary_job_id.clone(),
            canary_show_sha256: staged_node.canary_show_sha256.clone(),
        });
    }
    Ok(StagingReceipt {
        schema: STAGING_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        release_binary_sha256: cohort_staging.release_binary_sha256.clone(),
        scq_binary_sha256: cohort_staging.scq_binary_sha256.clone(),
        doctor_output_sha256: cohort_staging.doctor_output_sha256.clone(),
        dataset: spec.id.clone(),
        nodes,
        cohort_staging_sha256: Some(cohort_staging_sha256.to_string()),
    })
}

fn initialize_all_dataset(
    run_id: &str,
    spec: &FrozenDataset,
    cohort_sha256: &str,
    cohort_manifest: &AllCohortManifest,
    cohort_staging: &AllStagingReceipt,
    cohort_staging_sha256: &str,
    resume: bool,
) -> Result<(PathBuf, CampaignManifest, StagingReceipt)> {
    let root = campaign_root(run_id, &spec.id);
    let campaign_path = root.join("campaign").join("campaign.v1.json");
    let expected_config = CampaignConfig::all(spec);
    let winner_seed_snapshot_sha256 =
        winner_seed_snapshot_sha256(&cohort_manifest.winner_seed_sources)?;
    let campaign = if campaign_path.exists() {
        if !resume && !root.join("RUN_COMPLETE").exists() && !root.join("RUN_EXHAUSTED").exists() {
            bail!(
                "Binary42 all-dataset campaign {} already exists; pass --resume",
                spec.id
            );
        }
        let recorded: CampaignManifest = load_json(&campaign_path)?;
        if recorded.schema != CAMPAIGN_SCHEMA
            || !recorded.test_adaptive
            || recorded.benchmark_valid
            || recorded.run_id != run_id
            || recorded.dataset != spec.id
            || recorded.frozen_cohort_sha256 != cohort_sha256
            || recorded.binary_seed_sha256 != SEED_BINARY_SHA256
            || recorded.release_binary_sha256 != cohort_manifest.release_binary_sha256
            || recorded.scq_binary_sha256 != cohort_manifest.scq_binary_sha256
            || recorded.dataset_binding != *spec
            || recorded.config != expected_config
            || recorded.cohort_staging_sha256.as_deref() != Some(cohort_staging_sha256)
            || recorded.winner_seed_snapshot_sha256.as_deref()
                != Some(winner_seed_snapshot_sha256.as_str())
        {
            bail!(
                "Binary42 all-dataset resume identity changed for {}",
                spec.id
            );
        }
        recorded
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(
            &root.join("campaign").join("binary42-frozen-cohort.v1.json"),
            COHORT_BYTES,
        )?;
        let created = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            dataset: spec.id.clone(),
            frozen_cohort_file: "campaign/binary42-frozen-cohort.v1.json".to_string(),
            frozen_cohort_sha256: cohort_sha256.to_string(),
            binary_seed_sha256: SEED_BINARY_SHA256.to_string(),
            release_binary_sha256: cohort_manifest.release_binary_sha256.clone(),
            scq_binary_sha256: cohort_manifest.scq_binary_sha256.clone(),
            dataset_binding: spec.clone(),
            config: expected_config,
            started_at_unix_ms: unix_millis(),
            cohort_staging_sha256: Some(cohort_staging_sha256.to_string()),
            winner_seed_snapshot_sha256: Some(winner_seed_snapshot_sha256),
        };
        write_json(&campaign_path, &created)?;
        let baseline = BaselineManifest {
            schema: BASELINE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            dataset: spec.id.clone(),
            leader_method: spec.leader_method.clone(),
            leader_auc: spec.leader_auc,
            leader_report_path: spec.leader_report_path.clone(),
            leader_report_sha256: spec.leader_report_sha256.clone(),
            generation_zero_best_auc: None,
        };
        write_json(
            &root.join("baselines").join("frozen-leader.v1.json"),
            &baseline,
        )?;
        let members =
            seed_population_with_winners(spec.features, &cohort_manifest.winner_seed_sources)?;
        let seeds = SeedManifest {
            schema: SEED_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            seed_file_sha256: SEED_BINARY_SHA256.to_string(),
            population: POPULATION,
            members,
            winner_seed_sources: cohort_manifest.winner_seed_sources.clone(),
        };
        write_json(&root.join("seeds").join("generation-zero.v1.json"), &seeds)?;
        created
    };
    let staging = dataset_staging_from_cohort(spec, cohort_staging, cohort_staging_sha256)?;
    write_json(&root.join("staging").join("staging.v1.json"), &staging)?;
    validate_staging(
        &staging,
        spec,
        &campaign.release_binary_sha256,
        &campaign.scq_binary_sha256,
        false,
    )?;
    Ok((root, campaign, staging))
}

pub(crate) fn run(run_id: &str, dataset: &str, resume: bool) -> Result<()> {
    let (root, campaign, staging) = initialize_campaign(run_id, dataset, resume)?;
    if root.join("RUN_COMPLETE").exists() {
        return validate(run_id, dataset);
    }
    run_campaign_loop(&root, &campaign, &staging)?;
    validate(run_id, dataset)
}

fn run_campaign_loop(
    root: &Path,
    campaign: &CampaignManifest,
    staging: &StagingReceipt,
) -> Result<()> {
    if root.join("RUN_COMPLETE").exists() || root.join("RUN_EXHAUSTED").exists() {
        return Ok(());
    }
    let train = read_binary_csv(Path::new(&campaign.dataset_binding.train_path))?;
    let test = read_binary_csv(Path::new(&campaign.dataset_binding.test_path))?;
    if train.x.len() != campaign.dataset_binding.train_rows
        || test.x.len() != campaign.dataset_binding.test_rows
    {
        bail!("Binary42 campaign data changed after initialization");
    }
    let seeds: SeedManifest = load_json(&root.join("seeds").join("generation-zero.v1.json"))?;
    let seed_semantics = seeds
        .members
        .iter()
        .map(|member| member.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    if seed_semantics.len() != POPULATION {
        bail!("Binary42 seed manifest does not contain 60 unique semantics");
    }
    loop {
        let mut checkpoints = load_checkpoints(root)?;
        if let Some(final_checkpoint) = checkpoints
            .last()
            .filter(|checkpoint| checkpoint.qualifying_winner.is_some())
        {
            let confirmation = if let Some(existing) = existing_confirmation(root)? {
                existing
            } else {
                confirm_winner(root, campaign, staging, final_checkpoint, &test)?
            };
            rebuild_indexes(root, &checkpoints, !campaign.config.shard_sizes.is_empty())?;
            publish_winner(root, campaign, &checkpoints, &confirmation)?;
            return Ok(());
        }
        if campaign
            .config
            .generation_cap
            .is_some_and(|cap| checkpoints.len() >= cap)
        {
            rebuild_indexes(root, &checkpoints, !campaign.config.shard_sizes.is_empty())?;
            publish_exhausted(root, campaign, &checkpoints)?;
            return Ok(());
        }
        let generation = checkpoints.len();
        let population = if let Some(previous) = checkpoints.last() {
            if previous.next_population.len() != POPULATION {
                bail!("Binary42 resume checkpoint has no complete next population");
            }
            previous.next_population.clone()
        } else {
            seeds.members.clone()
        };
        let checkpoint = process_generation(
            root,
            campaign,
            staging,
            generation,
            population,
            checkpoints.last(),
            &seed_semantics,
            &test,
        )?;
        checkpoints.push(checkpoint);
        rebuild_indexes(root, &checkpoints, !campaign.config.shard_sizes.is_empty())?;
    }
}

fn record_cohort_event(
    cohort_root: &Path,
    event: &str,
    dataset: Option<&str>,
    detail: Option<&str>,
) -> Result<()> {
    let value = json!({
        "schema": "JopeBinary42AllProgressEvent.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "event": event,
        "dataset": dataset,
        "detail": detail,
        "at_unix_ms": unix_millis(),
    });
    let bytes = canonical_json(&value)?;
    let sha256 = sha256_bytes(&bytes);
    immutable_or_equal(
        &cohort_root
            .join("progress")
            .join(format!("{sha256}.event.v1.json")),
        &bytes,
    )
}

fn initialize_all_cohort(
    run_id: &str,
    resume: bool,
) -> Result<(
    FrozenCohort,
    String,
    AllCohortManifest,
    AllStagingReceipt,
    String,
)> {
    validate_run_id(run_id)?;
    let (cohort, cohort_sha256) = read_cohort()?;
    for spec in &cohort.datasets {
        verify_dataset_binding(spec)?;
    }
    let release_binary_sha256 = current_binary_sha256()?;
    let resolved_scq = fs::canonicalize(all_scq_binary_str()).with_context(|| {
        format!(
            "resolve Binary42 SCQ path {all_scq_binary_str}",
            all_scq_binary_str = all_scq_binary_str()
        )
    })?;
    let scq_resolved_path = resolved_scq.to_string_lossy().into_owned();
    let scq_binary_sha256 = sha256_file(&resolved_scq)?;
    let ordered = admission_order(&cohort);
    let run = run_root(run_id);
    let cohort_root = run.join("cohort");
    let campaign_path = cohort_root.join("campaign.v1.json");
    let manifest = if campaign_path.exists() {
        if !resume {
            bail!("Binary42 all-dataset cohort already exists; pass --resume");
        }
        let bytes = fs::read(&campaign_path)?;
        let recorded: AllCohortManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&recorded)?
            || recorded.schema != ALL_CAMPAIGN_SCHEMA
            || !recorded.test_adaptive
            || recorded.benchmark_valid
            || recorded.run_id != run_id
            || recorded.frozen_cohort_sha256 != cohort_sha256
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_configured_path != all_scq_binary_str()
            || recorded.scq_resolved_path != scq_resolved_path
            || recorded.scq_binary_sha256 != scq_binary_sha256
            || recorded.config != AllCohortConfig::fixed()
            || recorded.admission_dataset_ids
                != ordered
                    .iter()
                    .map(|dataset| dataset.id.clone())
                    .collect::<Vec<_>>()
            || sha256_file(&cohort_root.join(&recorded.frozen_cohort_file))? != cohort_sha256
        {
            bail!("Binary42 all-dataset cohort resume identity changed");
        }
        recorded
    } else {
        if resume {
            bail!("cannot resume a Binary42 all-dataset cohort that does not exist");
        }
        if run.exists() && fs::read_dir(&run)?.next().is_some() {
            bail!("Binary42 all-dataset run ID collides with an existing run");
        }
        fs::create_dir_all(&cohort_root)?;
        immutable_write(
            &cohort_root.join("binary42-frozen-cohort.v1.json"),
            COHORT_BYTES,
        )?;
        let winner_seed_sources = confirmed_winner_seed_sources(run_id, &cohort)?;
        validate_winner_seed_snapshot(&winner_seed_sources, &cohort)?;
        let created = AllCohortManifest {
            schema: ALL_CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            frozen_cohort_file: "binary42-frozen-cohort.v1.json".to_string(),
            frozen_cohort_sha256: cohort_sha256.clone(),
            release_binary_sha256,
            scq_configured_path: all_scq_binary_str().to_string(),
            scq_resolved_path,
            scq_binary_sha256,
            config: AllCohortConfig::fixed(),
            admission_dataset_ids: ordered.iter().map(|dataset| dataset.id.clone()).collect(),
            winner_seed_sources,
            started_at_unix_ms: unix_millis(),
        };
        write_json(&campaign_path, &created)?;
        record_cohort_event(
            &cohort_root,
            "cohort_created",
            None,
            Some(&format!(
                "{} provenance-verified winner genomes promoted into generation zero",
                created.winner_seed_sources.len()
            )),
        )?;
        created
    };
    validate_winner_seed_snapshot(&manifest.winner_seed_sources, &cohort)?;
    let (staging, staging_sha256) = stage_all_campaign(
        &cohort_root,
        &cohort,
        &manifest.release_binary_sha256,
        &manifest.scq_resolved_path,
        &manifest.scq_binary_sha256,
    )?;
    Ok((cohort, cohort_sha256, manifest, staging, staging_sha256))
}

#[derive(Clone, Debug, Serialize)]
struct AllEvaluationIndexRecord {
    dataset: String,
    display_name: String,
    frozen_order: usize,
    #[serde(flatten)]
    evaluation: EvaluationIndexRecord,
}

fn summarize_dataset(
    run_id: &str,
    spec: &FrozenDataset,
) -> Result<(DatasetValidationSummary, Vec<AllEvaluationIndexRecord>)> {
    let root = campaign_root(run_id, &spec.id);
    let checkpoints = load_checkpoints(&root)?;
    let complete = root.join("RUN_COMPLETE").exists();
    let exhausted = root.join("RUN_EXHAUSTED").exists();
    if complete && exhausted {
        bail!(
            "Binary42 dataset {} has conflicting terminal markers",
            spec.id
        );
    }
    let status = if complete {
        "complete"
    } else if exhausted {
        "exhausted"
    } else {
        "in_progress"
    };
    let winner = checkpoints
        .iter()
        .find_map(|checkpoint| checkpoint.qualifying_winner.as_ref());
    let best_auc = checkpoints
        .iter()
        .map(|checkpoint| checkpoint.best_auc)
        .max_by(f64::total_cmp);
    let unique_genomes = checkpoints
        .iter()
        .flat_map(|checkpoint| &checkpoint.slots)
        .map(|slot| &slot.member.semantic_sha256)
        .collect::<BTreeSet<_>>()
        .len();
    let evaluations = evaluation_records(&checkpoints)
        .into_iter()
        .map(|evaluation| AllEvaluationIndexRecord {
            dataset: spec.id.clone(),
            display_name: spec.display_name.clone(),
            frozen_order: spec.order,
            evaluation,
        })
        .collect();
    Ok((
        DatasetValidationSummary {
            dataset: spec.id.clone(),
            display_name: spec.display_name.clone(),
            frozen_order: spec.order,
            status: status.to_string(),
            completed_generations: checkpoints.len(),
            logical_slots: checkpoints.len() * POPULATION,
            unique_genomes,
            best_auc,
            winner_generation: winner.map(|record| record.generation),
            winner_semantic_sha256: winner.map(|record| record.semantic_sha256.clone()),
        },
        evaluations,
    ))
}

fn rebuild_cohort_indexes(
    run_id: &str,
    cohort: &FrozenCohort,
) -> Result<(Vec<DatasetValidationSummary>, usize)> {
    let cohort_root = run_root(run_id).join("cohort");
    let mut datasets = Vec::with_capacity(cohort.datasets.len());
    let mut evaluations = Vec::new();
    for spec in &cohort.datasets {
        let root = campaign_root(run_id, &spec.id);
        if root.exists() {
            let (summary, mut records) = summarize_dataset(run_id, spec)?;
            datasets.push(summary);
            evaluations.append(&mut records);
        } else {
            datasets.push(DatasetValidationSummary {
                dataset: spec.id.clone(),
                display_name: spec.display_name.clone(),
                frozen_order: spec.order,
                status: "pending".to_string(),
                completed_generations: 0,
                logical_slots: 0,
                unique_genomes: 0,
                best_auc: None,
                winner_generation: None,
                winner_semantic_sha256: None,
            });
        }
    }
    let dataset_index = json!({
        "schema": "JopeBinary42AllDatasetIndex.v1",
        "test_adaptive": true,
        "benchmark_valid": false,
        "datasets": &datasets,
    });
    atomic_rebuild(
        &cohort_root.join("datasets.json"),
        &canonical_json(&dataset_index)?,
    )?;
    let mut dataset_csv = b"dataset,display_name,frozen_order,status,completed_generations,logical_slots,unique_genomes,best_auc,winner_generation,winner_semantic_sha256\n".to_vec();
    for dataset in &datasets {
        dataset_csv.extend_from_slice(
            format!(
                "{},{},{},{},{},{},{},{},{},{}\n",
                csv_field(&dataset.dataset),
                csv_field(&dataset.display_name),
                dataset.frozen_order,
                csv_field(&dataset.status),
                dataset.completed_generations,
                dataset.logical_slots,
                dataset.unique_genomes,
                dataset
                    .best_auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                dataset
                    .winner_generation
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                csv_field(dataset.winner_semantic_sha256.as_deref().unwrap_or("")),
            )
            .as_bytes(),
        );
    }
    atomic_rebuild(&cohort_root.join("datasets.csv"), &dataset_csv)?;
    let mut evaluation_jsonl = Vec::new();
    let mut evaluation_csv = b"dataset,display_name,frozen_order,test_adaptive,benchmark_valid,generation,slot,semantic_sha256,ancestry_operation,ancestry_parents,cache_disposition,assigned_node,evaluation_node,scq_job_id,auc,logloss,complexity,runtime_seconds,receipt_sha256,prediction_sha256,failure\n".to_vec();
    for record in &evaluations {
        evaluation_jsonl.extend_from_slice(&canonical_json(record)?);
        let evaluation = &record.evaluation;
        evaluation_csv.extend_from_slice(
            format!(
                "{},{},{},true,false,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                csv_field(&record.dataset),
                csv_field(&record.display_name),
                record.frozen_order,
                evaluation.generation,
                evaluation.slot,
                csv_field(&evaluation.semantic_sha256),
                csv_field(&evaluation.ancestry_operation),
                csv_field(&evaluation.ancestry_parents.join(";")),
                csv_field(&evaluation.cache_disposition),
                csv_field(&evaluation.assigned_node),
                csv_field(&evaluation.evaluation_node),
                csv_field(&evaluation.scq_job_id),
                evaluation
                    .auc
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                evaluation
                    .logloss
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                evaluation.complexity,
                evaluation
                    .runtime_seconds
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                csv_field(&evaluation.receipt_sha256),
                csv_field(&evaluation.prediction_sha256),
                csv_field(evaluation.failure.as_deref().unwrap_or("")),
            )
            .as_bytes(),
        );
    }
    atomic_rebuild(
        &cohort_root.join("all-evaluations.jsonl"),
        &evaluation_jsonl,
    )?;
    atomic_rebuild(&cohort_root.join("all-evaluations.csv"), &evaluation_csv)?;
    let mut winners = b"dataset,display_name,generation,semantic_sha256,best_auc\n".to_vec();
    let mut exhausted = b"dataset,display_name,completed_generations,best_auc\n".to_vec();
    for dataset in &datasets {
        if dataset.status == "complete" {
            winners.extend_from_slice(
                format!(
                    "{},{},{},{},{}\n",
                    csv_field(&dataset.dataset),
                    csv_field(&dataset.display_name),
                    dataset.winner_generation.unwrap(),
                    csv_field(dataset.winner_semantic_sha256.as_deref().unwrap()),
                    dataset.best_auc.unwrap(),
                )
                .as_bytes(),
            );
        } else if dataset.status == "exhausted" {
            exhausted.extend_from_slice(
                format!(
                    "{},{},{},{}\n",
                    csv_field(&dataset.dataset),
                    csv_field(&dataset.display_name),
                    dataset.completed_generations,
                    dataset.best_auc.unwrap(),
                )
                .as_bytes(),
            );
        }
    }
    atomic_rebuild(&cohort_root.join("winners.csv"), &winners)?;
    atomic_rebuild(&cohort_root.join("exhausted.csv"), &exhausted)?;
    Ok((datasets, evaluations.len()))
}

pub(crate) fn run_all(run_id: &str, resume: bool) -> Result<()> {
    let (cohort, cohort_sha256, manifest, staging, staging_sha256) =
        initialize_all_cohort(run_id, resume)?;
    let cohort_root = run_root(run_id).join("cohort");
    if cohort_root.join("terminal-summary.v1.json").exists() {
        return validate_all(run_id);
    }
    let ordered = admission_order(&cohort);
    let queue = Arc::new(Mutex::new(VecDeque::from(ordered)));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for _ in 0..ALL_ACTIVE_DATASETS {
            let queue = Arc::clone(&queue);
            let errors = Arc::clone(&errors);
            let manifest = &manifest;
            let staging = &staging;
            let cohort_sha256 = &cohort_sha256;
            let staging_sha256 = &staging_sha256;
            let cohort_root = &cohort_root;
            scope.spawn(move || loop {
                let Some(spec) = queue.lock().unwrap().pop_front() else {
                    break;
                };
                let result = (|| -> Result<()> {
                    record_cohort_event(cohort_root, "dataset_admitted", Some(&spec.id), None)?;
                    let (root, campaign, dataset_staging) = initialize_all_dataset(
                        run_id,
                        &spec,
                        cohort_sha256,
                        manifest,
                        staging,
                        staging_sha256,
                        resume,
                    )?;
                    run_campaign_loop(&root, &campaign, &dataset_staging)?;
                    let terminal = if root.join("RUN_COMPLETE").exists() {
                        "dataset_complete"
                    } else if root.join("RUN_EXHAUSTED").exists() {
                        "dataset_exhausted"
                    } else {
                        bail!(
                            "Binary42 dataset {} returned without a terminal marker",
                            spec.id
                        );
                    };
                    record_cohort_event(cohort_root, terminal, Some(&spec.id), None)
                })();
                if let Err(error) = result {
                    let detail = format!("{error:#}");
                    let _ = record_cohort_event(
                        cohort_root,
                        "dataset_resumable_failure",
                        Some(&spec.id),
                        Some(&detail),
                    );
                    errors
                        .lock()
                        .unwrap()
                        .push(format!("{}: {detail}", spec.id));
                }
            });
        }
    });
    let (datasets, logical_evaluations) = rebuild_cohort_indexes(run_id, &cohort)?;
    let errors = Arc::try_unwrap(errors)
        .expect("Binary42 all-dataset error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "Binary42 all-dataset cohort retained resumable failures: {}",
            errors.join("; ")
        );
    }
    if datasets
        .iter()
        .any(|dataset| !matches!(dataset.status.as_str(), "complete" | "exhausted"))
    {
        bail!("Binary42 all-dataset cohort finished scheduling without 42 terminal datasets");
    }
    let summary = AllTerminalSummary {
        schema: ALL_SUMMARY_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        terminal_datasets: datasets.len(),
        winners: datasets
            .iter()
            .filter(|dataset| dataset.status == "complete")
            .count(),
        exhausted: datasets
            .iter()
            .filter(|dataset| dataset.status == "exhausted")
            .count(),
        logical_evaluations,
        completed_at_unix_ms: unix_millis(),
    };
    write_json(&cohort_root.join("terminal-summary.v1.json"), &summary)?;
    validate_all(run_id)
}

fn validate_shard_accounting(
    root: &Path,
    campaign: &CampaignManifest,
    checkpoints: &[GenerationCheckpoint],
) -> Result<()> {
    let manifest_root = root.join("jobs").join("manifests");
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    let mut primary = HashMap::<(usize, usize), ShardManifest>::new();
    if manifest_root.exists() {
        for entry in fs::read_dir(manifest_root)? {
            let path = entry?.path();
            let manifest: ShardManifest = load_json(&path)?;
            if manifest.schema != SHARD_SCHEMA
                || !manifest.test_adaptive
                || manifest.benchmark_valid
                || manifest.dataset != campaign.dataset
                || manifest.cohort_mode.as_deref()
                    != (!campaign.config.shard_sizes.is_empty()).then_some("all")
                || !matches!(manifest.kind.as_str(), "generation" | "confirmation")
                || manifest.job_id != expected_job_id(&manifest)?
            {
                bail!("invalid Binary42 immutable shard manifest");
            }
            if manifest.kind == "generation" {
                let start = shard_start(&shard_sizes, manifest.shard)?;
                if manifest.attempt > campaign.config.infrastructure_retries
                    || manifest.node != failover_node(manifest.shard, manifest.attempt)?
                    || manifest.slots.len() != shard_sizes[manifest.shard]
                    || manifest.cpus_reserved != shard_sizes[manifest.shard]
                    || manifest
                        .slots
                        .iter()
                        .enumerate()
                        .any(|(offset, slot)| slot.slot != start + offset)
                {
                    bail!("Binary42 immutable failover shard accounting mismatch");
                }
            } else if manifest.shard >= NODES.len()
                || manifest.node != NODES[manifest.shard]
                || manifest.slots.len() != 1
                || manifest.cpus_reserved != 1
            {
                bail!("Binary42 immutable confirmation shard accounting mismatch");
            }
            if manifest.kind == "generation"
                && manifest.attempt == 0
                && primary
                    .insert((manifest.generation, manifest.shard), manifest)
                    .is_some()
            {
                bail!("duplicate Binary42 primary shard manifest");
            }
        }
    }
    for checkpoint in checkpoints {
        for shard in 0..NODES.len() {
            let manifest = primary
                .get(&(checkpoint.generation, shard))
                .context("Binary42 checkpoint is missing a primary shard manifest")?;
            let start = shard_start(&shard_sizes, shard)?;
            if manifest.node != NODES[shard]
                || manifest.slots.len() != shard_sizes[shard]
                || manifest.cpus_reserved != shard_sizes[shard]
                || manifest.gpus_per_task != 0
                || manifest.candidate_threads != 1
                || manifest
                    .slots
                    .iter()
                    .enumerate()
                    .any(|(offset, slot)| slot.slot != start + offset)
            {
                bail!("Binary42 primary shard accounting mismatch");
            }
        }
    }
    Ok(())
}

fn validate_artifact_manifest(root: &Path, campaign: &CampaignManifest) -> Result<()> {
    let path = root.join("winner").join("artifact-manifest.v1.json");
    let recorded: WinnerArtifactManifest = load_json(&path)?;
    if recorded.schema != "JopeBinary42TestAdaptiveArtifactManifest.v1"
        || !recorded.test_adaptive
        || recorded.benchmark_valid
        || recorded.run_id != campaign.run_id
        || recorded.dataset != campaign.dataset
    {
        bail!("Binary42 winner artifact manifest identity mismatch");
    }
    for artifact in &recorded.artifacts {
        let target = root.join(&artifact.path);
        if !target.is_file() || sha256_file(&target)? != artifact.sha256 {
            bail!(
                "Binary42 artifact changed or disappeared: {}",
                artifact.path
            );
        }
    }
    let rebuilt = collect_artifacts(root)?;
    if rebuilt.len() != recorded.artifacts.len()
        || rebuilt
            .iter()
            .zip(&recorded.artifacts)
            .any(|(left, right)| left.path != right.path || left.sha256 != right.sha256)
    {
        bail!("Binary42 complete artifact manifest is not exhaustive");
    }
    Ok(())
}

fn validate_dataset_campaign(
    run_id: &str,
    dataset: &str,
    all_dataset_cohort: bool,
) -> Result<DatasetValidationSummary> {
    validate_run_id(run_id)?;
    let (spec, cohort_sha256) = if all_dataset_cohort {
        cohort_dataset(dataset)?
    } else {
        selected_dataset(dataset)?
    };
    verify_dataset_binding(&spec)?;
    let run = run_root(run_id);
    if !all_dataset_cohort {
        let entries = fs::read_dir(&run)
            .with_context(|| format!("Binary42 run {run_id} does not exist"))?
            .map(|entry| Ok(entry?.file_name()))
            .collect::<Result<Vec<_>>>()?;
        if entries.len() != 1 || entries[0] != std::ffi::OsStr::new(dataset) {
            bail!("Binary42 pilot run contains artifacts for a non-ILPD dataset");
        }
    }
    let root = campaign_root(run_id, dataset);
    let campaign: CampaignManifest = load_json(&root.join("campaign").join("campaign.v1.json"))?;
    let expected_config = if all_dataset_cohort {
        CampaignConfig::all(&spec)
    } else {
        CampaignConfig::fixed()
    };
    let expected_cohort_staging_sha256 = if all_dataset_cohort {
        Some(sha256_file(&run.join("cohort").join("staging.v1.json"))?)
    } else {
        None
    };
    let winner_seed_sources = if all_dataset_cohort {
        let cohort_campaign: AllCohortManifest =
            load_json(&run.join("cohort").join("campaign.v1.json"))?;
        validate_winner_seed_snapshot(&cohort_campaign.winner_seed_sources, &read_cohort()?.0)?;
        cohort_campaign.winner_seed_sources
    } else {
        Vec::new()
    };
    let expected_winner_seed_snapshot_sha256 = if all_dataset_cohort {
        Some(winner_seed_snapshot_sha256(&winner_seed_sources)?)
    } else {
        None
    };
    if campaign.schema != CAMPAIGN_SCHEMA
        || !campaign.test_adaptive
        || campaign.benchmark_valid
        || campaign.run_id != run_id
        || campaign.dataset != dataset
        || campaign.dataset_binding != spec
        || campaign.frozen_cohort_sha256 != cohort_sha256
        || campaign.binary_seed_sha256 != SEED_BINARY_SHA256
        || campaign.config != expected_config
        || campaign.cohort_staging_sha256 != expected_cohort_staging_sha256
        || campaign.winner_seed_snapshot_sha256 != expected_winner_seed_snapshot_sha256
        || sha256_file(&root.join(&campaign.frozen_cohort_file))? != cohort_sha256
    {
        bail!("Binary42 campaign manifest validation failed");
    }
    let staging: StagingReceipt = load_json(&root.join("staging").join("staging.v1.json"))?;
    validate_staging(
        &staging,
        &spec,
        &campaign.release_binary_sha256,
        &campaign.scq_binary_sha256,
        false,
    )?;
    if staging.cohort_staging_sha256 != campaign.cohort_staging_sha256 {
        bail!("Binary42 dataset staging does not bind the cohort staging receipt");
    }
    let seeds: SeedManifest = load_json(&root.join("seeds").join("generation-zero.v1.json"))?;
    let expected_seeds = seed_population_with_winners(spec.features, &winner_seed_sources)?;
    if seeds.schema != SEED_SCHEMA
        || !seeds.test_adaptive
        || seeds.benchmark_valid
        || seeds.seed_file_sha256 != SEED_BINARY_SHA256
        || seeds.population != POPULATION
        || seeds.members != expected_seeds
        || seeds.winner_seed_sources != winner_seed_sources
    {
        bail!("Binary42 seed manifest validation failed");
    }
    let seed_semantics = seeds
        .members
        .iter()
        .map(|member| member.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    let test = read_binary_csv(Path::new(&spec.test_path))?;
    let checkpoints = load_checkpoints(&root)?;
    if checkpoints.is_empty() {
        bail!("Binary42 campaign has no completed generation barrier");
    }
    if all_dataset_cohort
        && (checkpoints.len() < 2
            || checkpoints.len() > ALL_MAX_GENERATIONS
            || generation_root(&root, ALL_MAX_GENERATIONS).exists())
    {
        bail!("Binary42 all-dataset campaign must contain 2-10 generations numbered 0-9");
    }
    let cache = load_cache(&root)?;
    let mut validated_semantics = BTreeSet::new();
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    for (generation, checkpoint) in checkpoints.iter().enumerate() {
        if checkpoint.run_id != run_id
            || checkpoint.dataset != dataset
            || checkpoint.generation != generation
            || checkpoint.fresh_evaluations + checkpoint.cache_reuses != POPULATION
        {
            bail!("Binary42 checkpoint campaign binding mismatch");
        }
        let generation_manifest_path =
            generation_root(&root, generation).join("generation-manifest.v1.json");
        let generation_manifest_bytes = fs::read(&generation_manifest_path)?;
        let generation_manifest: GenerationManifest =
            serde_json::from_slice(&generation_manifest_bytes)?;
        let expected_manifest_shard_sizes = if all_dataset_cohort {
            shard_sizes.clone()
        } else {
            Vec::new()
        };
        if generation_manifest_bytes != canonical_json(&generation_manifest)?
            || generation_manifest.schema != GENERATION_SCHEMA
            || !generation_manifest.test_adaptive
            || generation_manifest.benchmark_valid
            || generation_manifest.run_id != run_id
            || generation_manifest.dataset != dataset
            || generation_manifest.generation != generation
            || generation_manifest.logical_slots != POPULATION
            || generation_manifest.shard_nodes
                != NODES
                    .iter()
                    .map(|node| (*node).to_string())
                    .collect::<Vec<_>>()
            || generation_manifest.shard_sizes != expected_manifest_shard_sizes
            || generation_manifest.members
                != checkpoint
                    .slots
                    .iter()
                    .map(|slot| slot.member.clone())
                    .collect::<Vec<_>>()
        {
            bail!("Binary42 generation manifest validation failed");
        }
        for slot in &checkpoint.slots {
            if slot.assigned_node != assigned_node_for_slot(&shard_sizes, slot.slot)? {
                bail!("Binary42 logical slot assigned-node binding mismatch");
            }
        }
        let mut ranked = checkpoint
            .slots
            .iter()
            .filter(|slot| slot.score.is_some())
            .cloned()
            .collect::<Vec<_>>();
        ranked.sort_by(rank_slots);
        if ranked.len() != checkpoint.valid_scores
            || ranked
                .iter()
                .map(|slot| slot.member.semantic_sha256.clone())
                .collect::<Vec<_>>()
                != checkpoint.ranked_semantics
            || ranked.first().map(|slot| slot.slot) != Some(checkpoint.best_slot)
            || ranked
                .first()
                .and_then(|slot| slot.score.as_ref())
                .map(|score| score.auc.to_bits())
                != Some(checkpoint.best_auc.to_bits())
        {
            bail!("Binary42 checkpoint ranking authority mismatch");
        }
        let expected_gen0 = if generation == 0 {
            checkpoint.best_auc
        } else {
            checkpoints[0].best_auc
        };
        if checkpoint.generation_zero_best_auc.to_bits() != expected_gen0.to_bits() {
            bail!("Binary42 generation-zero baseline changed");
        }
        let expected_winner = qualifying_winner_for_policy(
            generation,
            &ranked,
            &seed_semantics,
            &spec.win_policy,
            spec.leader_auc,
            checkpoint.generation_zero_best_auc,
        );
        if checkpoint.qualifying_winner.as_ref().map(|winner| {
            (
                winner.slot,
                winner.semantic_sha256.as_str(),
                winner.auc.to_bits(),
            )
        }) != expected_winner.as_ref().map(|winner| {
            (
                winner.slot,
                winner.semantic_sha256.as_str(),
                winner.auc.to_bits(),
            )
        }) {
            bail!("Binary42 qualifying-winner comparator mismatch");
        }
        for slot in &checkpoint.slots {
            let cached = cache
                .get(&slot.member.semantic_sha256)
                .context("Binary42 logical slot lacks a semantic cache entry")?;
            if cached.receipt_sha256 != slot.receipt_sha256
                || cached.prediction_sha256 != slot.prediction_sha256
                || cached.score != slot.score
                || cached.failure != slot.failure
                || cached.runtime_seconds != slot.runtime_seconds
            {
                bail!("Binary42 logical slot differs from semantic cache");
            }
            if validated_semantics.insert(slot.member.semantic_sha256.clone()) {
                let genome_path = root
                    .join("objects")
                    .join("genomes")
                    .join(format!("{}.v1.json", slot.member.semantic_sha256));
                let genome: GenomeObject = load_json(&genome_path)?;
                if genome.semantic_sha256 != slot.member.semantic_sha256
                    || genome.canonical_genome != slot.member.canonical_genome
                    || genome.complexity != slot.member.complexity
                {
                    bail!("Binary42 content-addressed genome mismatch");
                }
                let receipt_path = root
                    .join("objects")
                    .join("receipts")
                    .join(format!("{}.json", cached.receipt_sha256));
                let prediction_path = root
                    .join("objects")
                    .join("predictions")
                    .join(format!("{}.csv", cached.prediction_sha256));
                if sha256_file(&receipt_path)? != cached.receipt_sha256
                    || sha256_file(&prediction_path)? != cached.prediction_sha256
                {
                    bail!("Binary42 content-addressed evaluation object mismatch");
                }
                let receipt_bytes = fs::read(receipt_path)?;
                let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
                validate_candidate_backend(&receipt, all_dataset_cohort)?;
                validate_fresh_output(
                    &FreshOutput {
                        receipt,
                        receipt_bytes,
                        prediction_bytes: fs::read(prediction_path)?,
                    },
                    &test,
                )?;
            }
        }
        let reached_cap = generation_cap_reached(&campaign.config, generation);
        if checkpoint.qualifying_winner.is_none() && !reached_cap {
            let parents = ranked
                .iter()
                .map(|slot| {
                    let score = slot.score.as_ref().unwrap();
                    BinaryAdaptiveRanked {
                        canonical_genome: slot.member.canonical_genome.clone(),
                        semantic_sha256: slot.member.semantic_sha256.clone(),
                        complexity: slot.member.complexity,
                        auc: score.auc,
                        logloss: score.logloss,
                    }
                })
                .collect::<Vec<_>>();
            let expected_next =
                test_adaptive_next_population(RNG_SEED, generation + 1, &parents, spec.features)?
                    .into_iter()
                    .map(MemberRecord::from_binary)
                    .collect::<Result<Vec<_>>>()?;
            if checkpoint.next_population != expected_next {
                bail!("Binary42 deterministic next population changed");
            }
        } else if !checkpoint.next_population.is_empty() {
            bail!("Binary42 terminal checkpoint unexpectedly continues evolution");
        }
    }
    validate_shard_accounting(&root, &campaign, &checkpoints)?;
    rebuild_indexes(&root, &checkpoints, all_dataset_cohort)?;
    let complete = root.join("RUN_COMPLETE").exists();
    let exhausted = root.join("RUN_EXHAUSTED").exists();
    if complete && exhausted {
        bail!("Binary42 campaign has conflicting terminal receipts");
    }
    if complete {
        if checkpoints.last().unwrap().qualifying_winner.is_none()
            || checkpoints
                .iter()
                .flat_map(|checkpoint| &checkpoint.slots)
                .any(|slot| slot.score.is_none() || slot.failure.is_some())
        {
            bail!("Binary42 complete campaign has missing unique-genome scores");
        }
        let confirmation =
            existing_confirmation(&root)?.context("Binary42 completion lacks confirmations")?;
        if confirmation.confirmations.len() != NODES.len()
            || confirmation
                .confirmations
                .iter()
                .enumerate()
                .any(|(index, record)| record.node != NODES[index])
        {
            bail!("Binary42 completion lacks exact four-node replay evidence");
        }
        let final_receipt: WinnerReceipt =
            load_json(&root.join("winner").join("final-receipt.v1.json"))?;
        if final_receipt.schema != WINNER_SCHEMA
            || !final_receipt.test_adaptive
            || final_receipt.benchmark_valid
            || !policy_win(
                &spec.win_policy,
                final_receipt.auc,
                final_receipt.frozen_leader_auc,
                final_receipt.generation_zero_best_auc,
            )
            || seed_semantics.contains(&final_receipt.semantic_sha256)
            || confirmation.semantic_sha256 != final_receipt.semantic_sha256
            || confirmation.confirmations.iter().any(|record| {
                record.prediction_sha256 != final_receipt.prediction_sha256
                    || record.auc_bits != final_receipt.auc.to_bits()
            })
        {
            bail!("Binary42 final winner receipt validation failed");
        }
        validate_artifact_manifest(&root, &campaign)?;
        let complete_bytes = fs::read(root.join("RUN_COMPLETE"))?;
        let complete_marker: Value = serde_json::from_slice(&complete_bytes)?;
        let final_receipt_sha256 = sha256_bytes(&canonical_json(&final_receipt)?);
        let artifact_manifest_sha256 =
            sha256_file(&root.join("winner").join("artifact-manifest.v1.json"))?;
        if complete_bytes != canonical_json(&complete_marker)?
            || complete_marker.get("schema").and_then(Value::as_str) != Some(COMPLETE_SCHEMA)
            || complete_marker
                .get("test_adaptive")
                .and_then(Value::as_bool)
                != Some(true)
            || complete_marker
                .get("benchmark_valid")
                .and_then(Value::as_bool)
                != Some(false)
            || complete_marker.get("status").and_then(Value::as_str) != Some("complete")
            || complete_marker.get("run_id").and_then(Value::as_str) != Some(run_id)
            || complete_marker.get("dataset").and_then(Value::as_str) != Some(dataset)
            || complete_marker
                .get("winner_semantic_sha256")
                .and_then(Value::as_str)
                != Some(final_receipt.semantic_sha256.as_str())
            || complete_marker
                .get("final_receipt_sha256")
                .and_then(Value::as_str)
                != Some(final_receipt_sha256.as_str())
            || complete_marker
                .get("artifact_manifest_sha256")
                .and_then(Value::as_str)
                != Some(artifact_manifest_sha256.as_str())
        {
            bail!("Binary42 RUN_COMPLETE validation failed");
        }
    }
    if exhausted {
        if !all_dataset_cohort
            || checkpoints.len() != ALL_MAX_GENERATIONS
            || checkpoints
                .iter()
                .any(|checkpoint| checkpoint.qualifying_winner.is_some())
            || !checkpoint_next_populations_terminate_at_cap(&checkpoints)
        {
            bail!("Binary42 exhausted campaign violates the generation-9 terminal policy");
        }
        let receipt_path = root.join("best-effort").join("final-receipt.v1.json");
        let receipt_bytes = fs::read(&receipt_path)?;
        let receipt: ExhaustedReceipt = serde_json::from_slice(&receipt_bytes)?;
        let marker_bytes = fs::read(root.join("RUN_EXHAUSTED"))?;
        let marker: ExhaustedMarker = serde_json::from_slice(&marker_bytes)?;
        if receipt_bytes != canonical_json(&receipt)?
            || receipt.schema != EXHAUSTED_SCHEMA
            || !receipt.test_adaptive
            || receipt.benchmark_valid
            || receipt.status != "exhausted"
            || receipt.run_id != run_id
            || receipt.dataset != dataset
            || receipt.completed_generations != ALL_MAX_GENERATIONS
            || receipt.logical_slots != ALL_MAX_GENERATIONS * POPULATION
            || receipt.release_binary_sha256 != campaign.release_binary_sha256
            || marker_bytes != canonical_json(&marker)?
            || marker.schema != EXHAUSTED_SCHEMA
            || !marker.test_adaptive
            || marker.benchmark_valid
            || marker.status != "exhausted"
            || marker.run_id != run_id
            || marker.dataset != dataset
            || marker.completed_generations != ALL_MAX_GENERATIONS
            || marker.logical_slots != ALL_MAX_GENERATIONS * POPULATION
            || marker.best_semantic_sha256 != receipt.best_semantic_sha256
            || marker.best_auc.to_bits() != receipt.best_auc.to_bits()
            || marker.final_receipt_sha256 != sha256_bytes(&receipt_bytes)
        {
            bail!("Binary42 best-effort receipt validation failed");
        }
        let mut ranked = checkpoints
            .iter()
            .flat_map(|checkpoint| &checkpoint.slots)
            .filter(|slot| slot.score.is_some())
            .cloned()
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            rank_slots(left, right)
                .then_with(|| left.generation.cmp(&right.generation))
                .then_with(|| left.slot.cmp(&right.slot))
        });
        let best = ranked
            .first()
            .context("Binary42 exhausted campaign has no best score")?;
        let score = best.score.as_ref().unwrap();
        if receipt.best_generation != best.generation
            || receipt.best_slot != best.slot
            || receipt.best_semantic_sha256 != best.member.semantic_sha256
            || receipt.best_auc.to_bits() != score.auc.to_bits()
            || receipt.best_logloss.to_bits() != score.logloss.to_bits()
            || receipt.best_prediction_sha256 != best.prediction_sha256
            || receipt.best_receipt_sha256 != best.receipt_sha256
            || sha256_file(&root.join("best-effort").join("predictions.csv"))?
                != best.prediction_sha256
            || sha256_file(&root.join("best-effort").join("lineage.v1.json"))?
                != receipt.lineage_sha256
        {
            bail!("Binary42 best-effort artifact validation failed");
        }
        let genome: GenomeObject = load_json(&root.join("best-effort").join("genome.v1.json"))?;
        if genome.semantic_sha256 != best.member.semantic_sha256
            || genome.canonical_genome != best.member.canonical_genome
            || genome.complexity != best.member.complexity
        {
            bail!("Binary42 best-effort genome validation failed");
        }
    }
    if all_dataset_cohort && !complete && !exhausted {
        bail!("Binary42 all-dataset validation requires a terminal dataset");
    }
    let status = if complete {
        "complete"
    } else if exhausted {
        "exhausted"
    } else {
        "in_progress"
    };
    let winner = checkpoints
        .iter()
        .find_map(|checkpoint| checkpoint.qualifying_winner.as_ref());
    Ok(DatasetValidationSummary {
        dataset: spec.id,
        display_name: spec.display_name,
        frozen_order: spec.order,
        status: status.to_string(),
        completed_generations: checkpoints.len(),
        logical_slots: checkpoints.len() * POPULATION,
        unique_genomes: validated_semantics.len(),
        best_auc: checkpoints
            .iter()
            .map(|checkpoint| checkpoint.best_auc)
            .max_by(f64::total_cmp),
        winner_generation: winner.map(|record| record.generation),
        winner_semantic_sha256: winner.map(|record| record.semantic_sha256.clone()),
    })
}

fn checkpoint_next_populations_terminate_at_cap(checkpoints: &[GenerationCheckpoint]) -> bool {
    checkpoints
        .last()
        .is_some_and(|checkpoint| checkpoint.next_population.is_empty())
}

pub(crate) fn validate_all(run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let (cohort, cohort_sha256) = read_cohort()?;
    let run = run_root(run_id);
    let cohort_root = run.join("cohort");
    let campaign_path = cohort_root.join("campaign.v1.json");
    let campaign_bytes = fs::read(&campaign_path)
        .with_context(|| format!("Binary42 all-dataset run {run_id} does not exist"))?;
    let campaign: AllCohortManifest = serde_json::from_slice(&campaign_bytes)?;
    let resolved_scq = fs::canonicalize(all_scq_binary_str())?;
    let resolved_scq_path = resolved_scq.to_string_lossy().into_owned();
    let ordered = admission_order(&cohort);
    if campaign_bytes != canonical_json(&campaign)?
        || campaign.schema != ALL_CAMPAIGN_SCHEMA
        || !campaign.test_adaptive
        || campaign.benchmark_valid
        || campaign.run_id != run_id
        || campaign.frozen_cohort_sha256 != cohort_sha256
        || campaign.release_binary_sha256 != current_binary_sha256()?
        || campaign.scq_configured_path != all_scq_binary_str()
        || campaign.scq_resolved_path != resolved_scq_path
        || campaign.scq_binary_sha256 != sha256_file(&resolved_scq)?
        || campaign.config != AllCohortConfig::fixed()
        || campaign.admission_dataset_ids
            != ordered
                .iter()
                .map(|dataset| dataset.id.clone())
                .collect::<Vec<_>>()
        || sha256_file(&cohort_root.join(&campaign.frozen_cohort_file))? != cohort_sha256
    {
        bail!("Binary42 all-dataset cohort manifest validation failed");
    }
    validate_winner_seed_snapshot(&campaign.winner_seed_sources, &cohort)?;
    let staging_path = cohort_root.join("staging.v1.json");
    let staging_bytes = fs::read(&staging_path)?;
    let staging: AllStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    if staging_bytes != canonical_json(&staging)? {
        bail!("Binary42 all-dataset staging receipt is not canonical JSON");
    }
    validate_all_staging(
        &staging,
        &cohort,
        &campaign.release_binary_sha256,
        &campaign.scq_resolved_path,
        &campaign.scq_binary_sha256,
        false,
    )?;
    let expected_entries = std::iter::once("cohort".to_string())
        .chain(cohort.datasets.iter().map(|dataset| dataset.id.clone()))
        .collect::<BTreeSet<_>>();
    let actual_entries = fs::read_dir(&run)?
        .map(|entry| {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                bail!("unexpected non-directory artifact at Binary42 cohort run root");
            }
            Ok(entry.file_name().to_string_lossy().into_owned())
        })
        .collect::<Result<BTreeSet<_>>>()?;
    if actual_entries != expected_entries {
        bail!("Binary42 all-dataset run has missing or unknown dataset directories");
    }
    let mut validated = Vec::with_capacity(cohort.datasets.len());
    for spec in &cohort.datasets {
        let summary = validate_dataset_campaign(run_id, &spec.id, true)?;
        if !matches!(summary.status.as_str(), "complete" | "exhausted")
            || !(2..=ALL_MAX_GENERATIONS).contains(&summary.completed_generations)
            || summary.logical_slots != summary.completed_generations * POPULATION
        {
            bail!(
                "Binary42 dataset {} is not terminal and barrier-complete",
                spec.id
            );
        }
        validated.push(summary);
    }
    let expected_logical = validated
        .iter()
        .map(|summary| summary.logical_slots)
        .sum::<usize>();
    let (rebuilt, rebuilt_logical) = rebuild_cohort_indexes(run_id, &cohort)?;
    if rebuilt != validated || rebuilt_logical != expected_logical {
        bail!("Binary42 global index counts differ from per-dataset ledgers");
    }
    let jsonl_count = fs::read_to_string(cohort_root.join("all-evaluations.jsonl"))?
        .lines()
        .count();
    let csv_count = fs::read_to_string(cohort_root.join("all-evaluations.csv"))?
        .lines()
        .count()
        .saturating_sub(1);
    if jsonl_count != expected_logical || csv_count != expected_logical {
        bail!("Binary42 global evaluation indexes have the wrong logical row count");
    }
    let summary_path = cohort_root.join("terminal-summary.v1.json");
    let summary_bytes = fs::read(&summary_path)?;
    let summary: AllTerminalSummary = serde_json::from_slice(&summary_bytes)?;
    let winners = validated
        .iter()
        .filter(|dataset| dataset.status == "complete")
        .count();
    let exhausted = validated.len() - winners;
    if summary_bytes != canonical_json(&summary)?
        || summary.schema != ALL_SUMMARY_SCHEMA
        || !summary.test_adaptive
        || summary.benchmark_valid
        || summary.run_id != run_id
        || summary.terminal_datasets != 42
        || summary.winners != winners
        || summary.exhausted != exhausted
        || summary.logical_evaluations != expected_logical
    {
        bail!("Binary42 all-dataset terminal summary validation failed");
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "JopeBinary42AllValidation.v1",
            "test_adaptive": true,
            "benchmark_valid": false,
            "run_id": run_id,
            "terminal_datasets": validated.len(),
            "winners": winners,
            "exhausted": exhausted,
            "logical_evaluations": expected_logical,
            "four_node_confirmed_winners": winners,
            "status": "complete",
        }))?
    );
    Ok(())
}

pub(crate) fn validate(run_id: &str, dataset: &str) -> Result<()> {
    let summary = validate_dataset_campaign(run_id, dataset, false)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "JopeBinary42TestAdaptiveValidation.v1",
            "test_adaptive": true,
            "benchmark_valid": false,
            "run_id": run_id,
            "dataset": dataset,
            "status": summary.status,
            "completed_generations": summary.completed_generations,
            "logical_slots": summary.logical_slots,
            "unique_genomes": summary.unique_genomes,
            "four_node_confirmed": summary.status == "complete",
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn qualified_dataset(mut spec: FrozenDataset) -> FrozenDataset {
        if let Some(root) = std::env::var_os("JOPE_BINARY42_TEST_DATA_ROOT") {
            let root = PathBuf::from(root).join(&spec.id);
            spec.train_path = root.join("train.csv").display().to_string();
            spec.test_path = root.join("test.csv").display().to_string();
        }
        let filename = Path::new(&spec.leader_report_path)
            .file_name()
            .expect("frozen leader report path has a filename")
            .to_owned();
        let historical = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/binary42-historical-reports/v1")
            .join(&spec.id)
            .join(&filename);
        if historical.is_file() {
            spec.leader_report_path = historical.display().to_string();
        } else if let Some(root) = std::env::var_os("JOPE_BINARY42_TEST_REPORT_ROOT") {
            spec.leader_report_path = PathBuf::from(root)
                .join(&spec.id)
                .join("reports")
                .join(&filename)
                .display()
                .to_string();
        }
        spec
    }

    #[test]
    fn confirmed_binary_winners_are_first_in_generation_zero() {
        let genomes = [
            "(pipe X (gam_boost lr=0.2 rounds=100))",
            "(pipe X (extra_trees depth=6 trees=32))",
        ];
        let sources = genomes
            .iter()
            .enumerate()
            .map(|(index, genome)| WinnerSeedSource {
                source_run_id: format!("source-{index}"),
                source_dataset: format!("{index:016x}"),
                complete_file: format!("complete-{index}"),
                complete_sha256: format!("complete-sha-{index}"),
                final_receipt_file: format!("receipt-{index}"),
                final_receipt_sha256: format!("receipt-sha-{index}"),
                confirmation_file: format!("confirmation-{index}"),
                confirmation_sha256: format!("confirmation-sha-{index}"),
                canonical_genome: (*genome).to_string(),
                semantic_sha256: sha256_bytes(genome.as_bytes()),
                auc_bits: 1.0f64.to_bits(),
                frozen_leader_auc_bits: 0.9f64.to_bits(),
                generation_zero_best_auc_bits: 0.95f64.to_bits(),
            })
            .collect::<Vec<_>>();
        let members = seed_population_with_winners(128, &sources).unwrap();
        assert_eq!(members.len(), POPULATION);
        assert_eq!(members[0].canonical_genome, genomes[0]);
        assert_eq!(members[1].canonical_genome, genomes[1]);
        assert_eq!(
            members
                .iter()
                .map(|member| &member.semantic_sha256)
                .collect::<BTreeSet<_>>()
                .len(),
            POPULATION
        );
        assert!(members
            .iter()
            .all(|member| member.ancestry.operation == "seed"));
    }

    fn member(name: &str, operation: &str, parents: Vec<String>) -> MemberRecord {
        MemberRecord {
            canonical_genome: name.to_string(),
            semantic_sha256: sha256_bytes(name.as_bytes()),
            complexity: name.len(),
            ancestry: BinaryAncestry {
                operation: operation.to_string(),
                parents,
                retry_count: 0,
            },
        }
    }

    fn logical(
        slot: usize,
        name: &str,
        operation: &str,
        auc: f64,
        logloss: f64,
        complexity: usize,
    ) -> LogicalSlot {
        let mut member = member(name, operation, Vec::new());
        member.complexity = complexity;
        LogicalSlot {
            generation: 1,
            slot,
            assigned_node: "xbabe1".to_string(),
            member,
            cache_disposition: "fresh".to_string(),
            cache_source_generation: None,
            cache_source_slot: None,
            evaluation_node: "xbabe1".to_string(),
            scq_job_id: "job".to_string(),
            receipt_sha256: "r".repeat(64),
            prediction_sha256: "p".repeat(64),
            score: Some(CandidateScore { auc, logloss }),
            failure: None,
            runtime_seconds: None,
        }
    }

    #[test]
    fn frozen_ilpd_and_cohort_contract_are_exact() {
        let (cohort, _) = read_cohort().unwrap();
        assert_eq!(cohort.datasets.len(), 42);
        assert_eq!(
            cohort
                .datasets
                .iter()
                .filter(|dataset| dataset.leader_auc == 1.0)
                .count(),
            6
        );
        let ilpd = cohort
            .datasets
            .iter()
            .find(|dataset| dataset.id == ILPD)
            .unwrap();
        assert_eq!(ilpd.train_rows, 485);
        assert_eq!(ilpd.test_rows, 98);
        assert_eq!(ilpd.features, 10);
        assert_eq!(ilpd.leader_auc.to_bits(), 0.7681159420289855f64.to_bits());
    }

    #[test]
    fn binary_auc_uses_midranks_and_probabilities_are_bounded() {
        assert_eq!(auc(&[0.0, 1.0], &[0.5, 0.5]), 0.5);
        assert_eq!(
            parse_prediction_bytes(b"prediction\n0.0\n0.5\n1.0\n").unwrap(),
            vec![0.0, 0.5, 1.0]
        );
        assert!(parse_prediction_bytes(b"prediction\n1.0001\n").is_err());
        assert!(parse_prediction_bytes(b"prediction\nNaN\n").is_err());
        let tiny = [1e-20, 2e-20, f64::MIN_POSITIVE, 1.0 - f64::EPSILON];
        let roundtrip = parse_prediction_bytes(&binary_prediction_roundtrip_bytes(&tiny)).unwrap();
        assert_eq!(
            roundtrip
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            tiny.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
        let score = 1.0967166826203898_f64;
        assert!(logloss_replays(score, f64::from_bits(score.to_bits() - 1)));
        assert!(logloss_replays(score, score + 9e-15));
        assert!(!logloss_replays(score, score + 2e-12));
    }

    #[test]
    fn rank_chain_applies_all_tie_breakers() {
        let mut slots = [
            logical(0, "z", "mutation", 0.8, 0.4, 1),
            logical(1, "a", "mutation", 0.9, 0.5, 1),
            logical(2, "b", "mutation", 0.9, 0.4, 2),
            logical(3, "c", "mutation", 0.9, 0.4, 1),
            logical(4, "d", "mutation", 0.9, 0.4, 1),
        ];
        slots.sort_by(rank_slots);
        assert_eq!(slots[0].score.as_ref().unwrap().auc, 0.9);
        assert_eq!(slots[0].score.as_ref().unwrap().logloss, 0.4);
        assert_eq!(slots[0].member.complexity, 1);
        assert!(
            slots[0].member.semantic_sha256 < slots[1].member.semantic_sha256
                || slots[1].member.complexity > slots[0].member.complexity
        );
        assert_eq!(slots.last().unwrap().score.as_ref().unwrap().auc, 0.8);
    }

    #[test]
    fn strict_and_saturated_comparators_are_distinct() {
        assert!(cohort_win(0.81, 0.8));
        assert!(!cohort_win(0.8, 0.8));
        assert!(cohort_win(1.0, 1.0));
        assert!(!cohort_win(0.999999999999, 1.0));
        assert!(strict_win(0.81, 0.8, 0.79));
        assert!(!strict_win(0.81, 0.8, 0.81));
    }

    #[test]
    fn canonical_binary_offspring_are_deterministic() {
        let seeds = seed_population(10).unwrap();
        let ranked = seeds
            .iter()
            .enumerate()
            .map(|(index, member)| BinaryAdaptiveRanked {
                canonical_genome: member.canonical_genome.clone(),
                semantic_sha256: member.semantic_sha256.clone(),
                complexity: member.complexity,
                auc: 1.0 - index as f64 / 1000.0,
                logloss: index as f64 / 1000.0,
            })
            .collect::<Vec<_>>();
        let first = test_adaptive_next_population(RNG_SEED, 1, &ranked, 10)
            .unwrap()
            .into_iter()
            .map(MemberRecord::from_binary)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let second = test_adaptive_next_population(RNG_SEED, 1, &ranked, 10)
            .unwrap()
            .into_iter()
            .map(MemberRecord::from_binary)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), POPULATION);
        assert_eq!(
            first
                .iter()
                .map(|member| &member.semantic_sha256)
                .collect::<BTreeSet<_>>()
                .len(),
            POPULATION
        );
        assert_eq!(
            first
                .iter()
                .filter(|member| member.ancestry.operation == "elite")
                .count(),
            ELITES
        );
    }

    #[test]
    fn semantic_cache_is_global_and_collisions_fail_closed() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let record = CacheEntry {
            schema: CACHE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            semantic_sha256: "a".repeat(64),
            receipt_sha256: "b".repeat(64),
            prediction_sha256: "c".repeat(64),
            score: Some(CandidateScore {
                auc: 0.75,
                logloss: 0.5,
            }),
            failure: None,
            evaluation_node: "xbabe1".to_string(),
            scq_job_id: "job".to_string(),
            first_generation: 0,
            first_slot: 0,
            runtime_seconds: None,
        };
        write_json(&cache_path(root, &record.semantic_sha256), &record).unwrap();
        write_json(&cache_path(root, &record.semantic_sha256), &record).unwrap();
        let loaded = load_cache(root).unwrap();
        assert_eq!(loaded.len(), 1);
        let mut changed = record.clone();
        changed.receipt_sha256 = "d".repeat(64);
        assert!(write_json(&cache_path(root, &record.semantic_sha256), &changed).is_err());
    }

    #[test]
    fn two_generation_fixture_selects_only_an_evolved_winner() {
        let seed = logical(0, "seed", "seed", 0.77, 0.5, 1);
        let seeds = BTreeSet::from([seed.member.semantic_sha256.clone()]);
        assert!(qualifying_winner(0, std::slice::from_ref(&seed), &seeds, 0.76, 0.77).is_none());
        let child = logical(0, "child", "mutation", 0.78, 0.4, 2);
        let winner =
            qualifying_winner(1, std::slice::from_ref(&child), &seeds, 0.76, 0.77).unwrap();
        assert_eq!(winner.semantic_sha256, child.member.semantic_sha256);
        assert_eq!(winner.auc, 0.78);
        let records = [seed, child];
        assert!(records.iter().all(|record| record.score.is_some()));
        assert_eq!(
            records
                .iter()
                .map(|record| &record.member.semantic_sha256)
                .collect::<BTreeSet<_>>()
                .len(),
            2
        );
    }

    #[test]
    fn shards_account_for_exactly_sixty_slots_and_resume_ignores_partial_tail() {
        let members = (0..POPULATION)
            .map(|index| member(&format!("genome-{index}"), "seed", Vec::new()))
            .collect::<Vec<_>>();
        let shards = shard_members(&members).unwrap();
        assert_eq!(shards.len(), 4);
        assert!(shards.iter().all(|shard| shard.len() == SHARD_SIZE));
        assert_eq!(shards.iter().map(Vec::len).sum::<usize>(), POPULATION);

        let directory = tempdir().unwrap();
        fs::create_dir_all(generation_root(directory.path(), 0)).unwrap();
        assert!(load_checkpoints(directory.path()).unwrap().is_empty());
    }

    #[test]
    fn all_frozen_dataset_bindings_and_binary_labels_are_valid() {
        let (cohort, _) = read_cohort().unwrap();
        assert_eq!(cohort.datasets.len(), 42);
        for frozen in cohort.datasets {
            let spec = qualified_dataset(frozen);
            verify_dataset_binding(&spec).unwrap();
            for path in [&spec.train_path, &spec.test_path] {
                let data = read_binary_csv(Path::new(path)).unwrap();
                assert!(data.y.iter().all(|label| *label == 0.0 || *label == 1.0));
                assert!(data.y.contains(&0.0));
                assert!(data.y.contains(&1.0));
            }
        }
    }

    #[test]
    fn all_shards_and_six_dataset_cpu_budgets_are_exact() {
        let members = (0..POPULATION)
            .map(|index| member(&format!("all-genome-{index}"), "seed", Vec::new()))
            .collect::<Vec<_>>();
        let shards = shard_members_with_sizes(&members, &ALL_SHARD_SIZES).unwrap();
        assert_eq!(
            shards.iter().map(Vec::len).collect::<Vec<_>>(),
            ALL_SHARD_SIZES
        );
        assert_eq!(shards.iter().map(Vec::len).sum::<usize>(), POPULATION);
        let config = AllCohortConfig::fixed();
        assert_eq!(config.node_cpu_budget_per_active_set["xbabe1"], 114);
        assert_eq!(config.node_cpu_budget_per_active_set["xbabe2"], 114);
        assert_eq!(config.node_cpu_budget_per_active_set["xbabe3"], 114);
        assert_eq!(config.node_cpu_budget_per_active_set["xbabe4"], 18);
        for slot in 0..POPULATION {
            let expected = if slot < 19 {
                "xbabe1"
            } else if slot < 38 {
                "xbabe2"
            } else if slot < 57 {
                "xbabe3"
            } else {
                "xbabe4"
            };
            assert_eq!(
                assigned_node_for_slot(&ALL_SHARD_SIZES, slot).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn cost_order_work_conserving_admission_and_failover_are_deterministic() {
        let (cohort, _) = read_cohort().unwrap();
        let ordered = admission_order(&cohort);
        assert_eq!(ordered.len(), 42);
        for pair in ordered.windows(2) {
            let left_cost = pair[0].train_rows * pair[0].features;
            let right_cost = pair[1].train_rows * pair[1].features;
            assert!(
                left_cost > right_cost
                    || (left_cost == right_cost && pair[0].order < pair[1].order)
            );
        }
        let mut pending = VecDeque::from(
            ordered
                .iter()
                .map(|spec| spec.id.clone())
                .collect::<Vec<_>>(),
        );
        let mut active = (0..ALL_ACTIVE_DATASETS)
            .map(|_| pending.pop_front().unwrap())
            .collect::<Vec<_>>();
        while let Some(next) = pending.pop_front() {
            active.remove(0);
            active.push(next);
            assert_eq!(active.len(), ALL_ACTIVE_DATASETS);
        }
        assert_eq!(failover_node(0, 0).unwrap(), "xbabe1");
        assert_eq!(failover_node(0, 1).unwrap(), "xbabe2");
        assert_eq!(failover_node(0, 2).unwrap(), "xbabe3");
        assert_eq!(failover_node(3, 1).unwrap(), "xbabe1");
        assert_eq!(failover_node(3, 2).unwrap(), "xbabe2");
        assert!(failover_node(0, 3).is_err());
    }

    #[test]
    fn strict_saturated_and_exhausted_synthetic_cohort_outcomes_are_distinct() {
        let seeds = BTreeSet::from([sha256_bytes(b"seed")]);
        let strict_child = logical(0, "strict-child", "mutation", 0.81, 0.2, 2);
        assert!(qualifying_winner_for_policy(
            1,
            std::slice::from_ref(&strict_child),
            &seeds,
            "strictly_greater_auc",
            0.80,
            0.805,
        )
        .is_some());
        let saturated_child = logical(0, "saturated-child", "crossover", 1.0, 0.1, 2);
        assert!(qualifying_winner_for_policy(
            1,
            std::slice::from_ref(&saturated_child),
            &seeds,
            "exact_auc_1_tie",
            1.0,
            1.0,
        )
        .is_some());
        let near_saturated = logical(0, "near-saturated", "mutation", 0.999, 0.1, 2);
        assert!(qualifying_winner_for_policy(
            1,
            std::slice::from_ref(&near_saturated),
            &seeds,
            "exact_auc_1_tie",
            1.0,
            1.0,
        )
        .is_none());
        let exhausted_candidate = logical(0, "exhausted-child", "mutation", 0.70, 0.6, 2);
        for generation in 0..ALL_MAX_GENERATIONS {
            assert!(qualifying_winner_for_policy(
                generation,
                std::slice::from_ref(&exhausted_candidate),
                &seeds,
                "strictly_greater_auc",
                0.80,
                0.75,
            )
            .is_none());
        }
        let config = CampaignConfig::all(&read_cohort().unwrap().0.datasets[0]);
        assert!(!generation_cap_reached(&config, 8));
        assert!(generation_cap_reached(&config, 9));
        assert_eq!(config.generation_cap, Some(10));
    }

    #[test]
    fn binary42_fitted_cpp17_predictions_match_rust_bits() {
        let (cohort, _) = read_cohort().unwrap();
        let spec = qualified_dataset(
            cohort
                .datasets
                .into_iter()
                .find(|dataset| dataset.id == ILPD)
                .unwrap(),
        );
        let train = read_binary_csv(Path::new(&spec.train_path)).unwrap();
        let test = read_binary_csv(Path::new(&spec.test_path)).unwrap();
        let member = seed_population(spec.features).unwrap().remove(0);
        let rust = evaluate_test_adaptive_binary(&member.canonical_genome, &train, &test).unwrap();
        let directory = tempdir().unwrap();
        let cpp = evaluate_test_adaptive_binary_cpp17(
            &member.canonical_genome,
            &train,
            &test,
            directory.path(),
        )
        .unwrap();
        assert_eq!(
            cpp.evaluation
                .predictions
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            rust.predictions
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(cpp.evaluation.auc.to_bits(), rust.auc.to_bits());
        assert_eq!(cpp.evaluation.logloss.to_bits(), rust.logloss.to_bits());
        let replayed = parse_prediction_bytes(&binary_prediction_roundtrip_bytes(
            &cpp.evaluation.predictions,
        ))
        .unwrap();
        assert_eq!(
            auc(&test.y, &replayed).to_bits(),
            cpp.evaluation.auc.to_bits()
        );
        assert!(logloss_replays(
            cpp.evaluation.logloss,
            feat_math::metrics::logloss(&test.y, &replayed)
        ));
        assert!(is_sha256_hex(&cpp.source_sha256));
        assert!(is_sha256_hex(&cpp.codec_sha256));
        assert!(cpp.compile_seconds >= 0.0);
    }

    #[test]
    fn pilot_serialization_omits_all_cohort_extensions() {
        let bytes = canonical_json(&CampaignConfig::fixed()).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(!text.contains("shard_sizes"));
        assert_eq!(CampaignConfig::fixed().generation_cap, None);
        let all = CampaignConfig::all(&read_cohort().unwrap().0.datasets[0]);
        assert_eq!(all.shard_sizes, ALL_SHARD_SIZES);
        assert_eq!(all.infrastructure_retries, 2);
    }

    #[test]
    fn semantic_caches_are_dataset_local() {
        let directory = tempdir().unwrap();
        let first = directory.path().join("dataset-a");
        let second = directory.path().join("dataset-b");
        let record = CacheEntry {
            schema: CACHE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            semantic_sha256: "a".repeat(64),
            receipt_sha256: "b".repeat(64),
            prediction_sha256: "c".repeat(64),
            score: Some(CandidateScore {
                auc: 0.75,
                logloss: 0.5,
            }),
            failure: None,
            evaluation_node: "xbabe1".to_string(),
            scq_job_id: "job".to_string(),
            first_generation: 0,
            first_slot: 0,
            runtime_seconds: Some(1.0),
        };
        write_json(&cache_path(&first, &record.semantic_sha256), &record).unwrap();
        assert_eq!(load_cache(&first).unwrap().len(), 1);
        assert!(load_cache(&second).unwrap().is_empty());
    }
}
