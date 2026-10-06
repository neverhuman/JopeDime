//! Frozen TimeSeries42 held-out-test-adaptive corpus-eval campaign.
//!
//! This lane is intentionally benchmark-invalid: held-out test MASE is reused
//! for selection. Receipts stay compatible with `sotarena_pull --task
//! timeseries` by preserving the corpus-eval `negative_mase` authority.

use super::*;
use feat_math::advanced_v3::validate_v3_genome;
use feat_math::heads::is_head_op;
use feat_math::target_transform::is_target_op;
use feat_math::{ColumnSpec, DataSchema, Objective, RowSemantics, TaskSpec};
use jope_dime_regressor::paths;
use jope_dime_regressor::timeseries::{
    direct_feature_names_for, make_supervised_reduction, read_timeseries_csv,
    validate_timeseries_data, ReductionVersion, TimeSeriesManifestSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::os::unix::fs::PermissionsExt;
use std::process::Stdio;
use std::sync::{Arc, Mutex, OnceLock};

jope_paths::once_str!(
    fn registry_str => paths::quant_root().join("timeseries")
);
jope_paths::once_str!(
    fn output_root_str => paths::legacy_tree("jope-test-adaptive").join("timeseries42")
);
jope_paths::once_str!(
    fn scq_binary_str => paths::super_cluster_scq_binary()
);
jope_paths::once_str!(
    fn corpus_eval_str => paths::corpus_eval_binary()
);
jope_paths::once_str!(
    fn sotarena_pull_str => paths::legacy_tree("sotarena_pull").join("target/release/sotarena_pull")
);
const LEGACY_PROJECT: &str = "jope";
const FORECAST3_PROJECT: &str = "jope-timeseries-test-adaptive";
const NODES: [&str; 4] = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"];
const LOCAL_SCQ_NODE: &str = "xbabe2";
const POPULATION: usize = 60;
const SHARDS: usize = 4;
const SHARD_SIZE: usize = POPULATION / SHARDS;
const DEFAULT_DATASET_WORKERS: usize = 2;
const MAX_DATASET_WORKERS: usize = 16;
const DATASET_WORKERS_ENV: &str = "JOPE_TIMESERIES42_DATASET_WORKERS";
const SKIP_RESUME_STAGING_VERIFY_ENV: &str = "JOPE_TIMESERIES42_SKIP_RESUME_STAGING_VERIFY";
const FULL_RESUME_STAGING_VERIFY_ENV: &str = "JOPE_TIMESERIES42_FULL_RESUME_STAGING_VERIFY";
const GENERATIONS: usize = 2;
const FORECAST3_RUN_ID: &str = "timeseries42-test-adaptive-forecast3-gen20-v2";
const FORECAST3_GENERATIONS: usize = 20;
jope_paths::once_str!(
    fn coverage_comparator_root_str => paths::test_adaptive_root().join("comparators/post-coverage-v2")
);
jope_paths::once_str!(
    fn coverage_comparator_set_str => paths::test_adaptive_root().join("comparators/post-coverage-v2/comparator-set.v2.json")
);
jope_paths::once_str!(
    fn mc_coverage_manifest_str => paths::quant_root().join("releases/mc-union-runnable272-v5/manifest.json")
);
const MC_COVERAGE_MANIFEST_SHA256: &str =
    "93888beb0d24f7cb7f0def66c88127a6cee6d0606c25c11ddff82161506e4938";
jope_paths::once_str!(
    fn mc_coverage_receipt_str => paths::quant_root().join("releases/mc-union-runnable272-v5/receipts/final.json")
);
jope_paths::once_str!(
    fn ts_coverage_manifest_str => paths::quant_root().join("releases/timeseries42-forecast3-v10/manifest.json")
);
const TS_COVERAGE_MANIFEST_SHA256: &str =
    "dda881dab5eeebcd7293081df5838062c97cfb5e15b9383776a4a3630f32c06d";
jope_paths::once_str!(
    fn ts_coverage_receipt_str => paths::quant_root().join("releases/timeseries42-forecast3-v10/receipts/final.json")
);
jope_paths::once_str!(
    fn ts_baseline_receipt_str => paths::quant_root().join("releases/mc272-ts42-coverage-v1/receipts/timeseries-baseline-all21.json")
);
const GEN30_GENERATIONS: usize = 30;
const GEN30_START_GENERATION: usize = GENERATIONS;
const DEFAULT_WINALL_GENERATIONS: usize = 120;
const ELITES: usize = 7;
const PARENT_POOL: usize = 30;
const TOURNAMENT: usize = 3;
const CANDIDATE_TIMEOUT_SECONDS: u64 = 300;
const SCQ_CAPACITY_RETRY_SECONDS: u64 = 30;
const SCQ_QUEUED_STALL_TIMEOUT_SECONDS: u64 = 300;
const SCQ_RETRYABLE_STALL_MARKER: &str = "timeseries42-scq-retryable-stall";
const RNG_SEED: u64 = 42 ^ 0x7473_3432_5f6d_6173;
const NODE_MARKER_BYTES: &[u8] = b"JopeTimeseries42NodeMarker.v1\n";

const CAMPAIGN_SCHEMA: &str = "JopeTimeseries42Campaign.v1";
const DATASET_SCHEMA: &str = "JopeTimeseries42DatasetBinding.v1";
const SEED_SCHEMA: &str = "JopeTimeseriesSeedBank.v1";
const STAGING_SCHEMA: &str = "JopeTimeseries42Staging.v1";
const GENERATION_SCHEMA: &str = "JopeTimeseries42Generation.v1";
const WORKER_SCHEMA: &str = "JopeTimeseries42WorkerReceipt.v1";
const JOB_SCHEMA: &str = "JopeTimeseries42ScqJob.v1";
const WAVE_SCHEMA: &str = "JopeTimeseries42ScqWaveReceipt.v1";
const RESULT_SCHEMA: &str = "JopeTimeseries42DatasetResult.v1";
const SUMMARY_SCHEMA: &str = "JopeTimeseries42Summary.v1";
const CORPUS_SHARD_SCHEMA: &str = "JopeFlywheelCorpusEvalShard.v1";
/// Emitted only when a campaign opts into a non-default reduction: the shard
/// then carries a REQUIRED `reduction.version` pin that old corpus_eval
/// binaries fail closed on (they reject the unknown schema string).
const CORPUS_SHARD_SCHEMA_V2: &str = "JopeFlywheelCorpusEvalShard.v2";
const JOPE_TIMESERIES_REPORT_SCHEMA: &str = "JopeTimeseries42DatasetReport.v1";
const JOPE_TIMESERIES_REPORT_FILE: &str =
    "jope-dime-regressor-test-adaptive-forecast3-gen20-v2.json";
const BASELINE_MODELS: [&str; 21] = [
    "seasonal_naive",
    "auto_arima",
    "auto_ets",
    "auto_theta",
    "prophet",
    "mlforecast_lightgbm",
    "neuralforecast_nbeats",
    "neuralforecast_patchtst",
    "sktime_reduction_lightgbm",
    "darts_tft",
    "autogluon_timeseries",
    "tabpfn_ts_3",
    "tabicl_forecaster",
    "chronos_2_small",
    "timesfm_2_5_200m",
    "moirai_2_small",
    "tabfm_ensemble_regressor",
    "gpt5p6sol_regressor",
    "fable5_regressor",
    "kimik3_regressor",
    "grok4p5_regressor",
];

fn dataset_worker_count() -> Result<usize> {
    let Ok(value) = std::env::var(DATASET_WORKERS_ENV) else {
        return Ok(DEFAULT_DATASET_WORKERS);
    };
    let workers = value
        .trim()
        .parse::<usize>()
        .with_context(|| format!("parse {DATASET_WORKERS_ENV}"))?;
    if workers == 0 {
        bail!("{DATASET_WORKERS_ENV} must be at least 1");
    }
    Ok(workers.min(MAX_DATASET_WORKERS))
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct OfficialLeader {
    model: String,
    mase: f64,
    rmse: f64,
    report_path: String,
    report_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExternalReport {
    model: String,
    report_path: String,
    report_sha256: String,
    mase: f64,
    rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetSpec {
    schema: String,
    dataset_hash: String,
    display_name: String,
    category: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    meta_path: String,
    meta_sha256: String,
    train_rows: usize,
    test_rows: usize,
    csv_features: usize,
    reduction_features: usize,
    horizon: usize,
    seasonality: usize,
    frequency: String,
    target_is_final_column: bool,
    known_future_covariates: Vec<String>,
    past_only_covariates: Vec<String>,
    source_real_data: bool,
    synthetic_origin: bool,
    /// Empty means ts-reduction-v1 (every historical campaign); campaigns opt
    /// into v2 by stamping ts-reduction-v2 here AND in CampaignConfig — the
    /// two pins must agree, which new campaign constructors are responsible
    /// for. Skipped when empty so historical manifests stay byte-identical.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    reduction_version: String,
    /// The immutable external envelope used by the all-models campaign.  This
    /// is absent from historical manifests so their canonical bytes remain
    /// unchanged when they are resumed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    accepted_external_reports: Vec<ExternalReport>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    comparator_lock_sha256: String,
    official_leader: OfficialLeader,
}

impl DatasetSpec {
    fn reduction(&self) -> Result<ReductionVersion> {
        if self.reduction_version.is_empty() {
            Ok(ReductionVersion::V1)
        } else {
            ReductionVersion::parse(&self.reduction_version)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AncestryRecord {
    operation: String,
    parents: Vec<String>,
    retry_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Member {
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    rust_source_len: usize,
    evolved_descendant: bool,
    ancestry: AncestryRecord,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SeedRecord {
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    rust_source_len: usize,
    source_family: String,
    source_file: String,
    source_file_sha256: String,
    source_record: usize,
    source_rank: usize,
    source_score: Option<f64>,
}

impl SeedRecord {
    fn member(&self) -> Member {
        Member {
            canonical_genome: self.canonical_genome.clone(),
            semantic_sha256: self.semantic_sha256.clone(),
            complexity: self.complexity,
            rust_source_len: self.rust_source_len,
            evolved_descendant: false,
            ancestry: AncestryRecord {
                operation: format!("seed:{}", self.source_family),
                parents: Vec::new(),
                retry_count: 0,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SeedBank {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    dataset_hash: String,
    population: usize,
    selection: String,
    seeds: Vec<SeedRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CampaignConfig {
    population: usize,
    generations: usize,
    shards: usize,
    shard_size: usize,
    nodes: Vec<String>,
    candidate_threads: usize,
    candidate_timeout_seconds: u64,
    infrastructure_retries: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    minimum_generations: usize,
    ranking: String,
    win_rule: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    lineage_salt: String,
    /// Empty means ts-reduction-v1. Frozen constructors (fixed/gen30/winall)
    /// never set this, so their configs and every historical manifest stay
    /// byte-identical; config equality checks therefore also pin v1 for the
    /// frozen flows.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    reduction_version: String,
}

impl CampaignConfig {
    fn fixed() -> Self {
        Self::fixed_with_generations(
            GENERATIONS,
            "finite evolved descendant with raw test MASE strictly lower than both official leader MASE and generation-zero best-seed MASE",
        )
    }

    fn gen30() -> Self {
        Self::fixed_with_generations(
            GEN30_GENERATIONS,
            "base generations 0-1 are immutable; generations 2-29 continue with 60 candidates/generation and stop non-base-wins only after a generation >=2 evolved descendant strictly beats both official leader MASE and generation-zero best MASE",
        )
    }

    fn forecast3_gen20() -> Self {
        let mut config = Self::fixed_with_generations(
            FORECAST3_GENERATIONS,
            "stop after the first confirmed finite candidate with raw test MASE strictly lower than the frozen authorized external comparator, including generation 0; otherwise exhaust after generation 19",
        );
        config.minimum_generations = 0;
        config
    }

    fn winall(generations: usize, start_generation: usize, lineage_salt: &str) -> Self {
        let mut config = Self::fixed_with_generations(
            generations,
            &format!(
                "source generations 0-{} are immutable; generations {start_generation}-{} continue source non-winners with 60 candidates/generation and stop after an evolved descendant strictly beats both official leader MASE and generation-zero best MASE",
                start_generation - 1,
                generations - 1
            ),
        );
        config.lineage_salt = lineage_salt.to_string();
        config
    }

    fn fixed_with_generations(generations: usize, win_rule: &str) -> Self {
        Self {
            population: POPULATION,
            generations,
            shards: SHARDS,
            shard_size: SHARD_SIZE,
            nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            candidate_threads: SHARD_SIZE,
            candidate_timeout_seconds: CANDIDATE_TIMEOUT_SECONDS,
            infrastructure_retries: NODES.len() - 1,
            minimum_generations: 0,
            ranking: "test_mase ASC, rmse ASC, complexity ASC, semantic_sha256 ASC".to_string(),
            win_rule: win_rule.to_string(),
            lineage_salt: String::new(),
            reduction_version: String::new(),
        }
    }
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

fn is_forecast3_gen20(run_id: &str) -> bool {
    run_id == FORECAST3_RUN_ID
}

fn base_campaign_config(run_id: &str) -> CampaignConfig {
    if is_forecast3_gen20(run_id) {
        CampaignConfig::forecast3_gen20()
    } else {
        CampaignConfig::fixed()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CampaignManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_campaign_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    continuation_start_generation: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    continuation_terminal_generation: Option<usize>,
    registry_root: String,
    output_root: String,
    release_binary_sha256: String,
    corpus_eval_sha256: String,
    scq_binary: String,
    scq_binary_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    coverage_binding: Option<CoverageBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    seed_populations: Vec<SeedPopulationBinding>,
    ordered_dataset_ids: Vec<String>,
    ordered_cohort_sha256: String,
    config: CampaignConfig,
    datasets: Vec<DatasetSpec>,
    started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SeedPopulationBinding {
    dataset_hash: String,
    seed_bank_path: String,
    seed_bank_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CoverageBinding {
    multiclass_manifest_sha256: String,
    multiclass_final_receipt_sha256: String,
    timeseries_manifest_sha256: String,
    timeseries_final_receipt_sha256: String,
    timeseries_baseline_receipt_sha256: String,
    comparator_root: String,
    comparator_set_path: String,
    comparator_set_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagedDataset {
    dataset_hash: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    meta_path: String,
    meta_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagedNode {
    node: String,
    hostname: String,
    bundle_root: String,
    runner_path: String,
    runner_sha256: String,
    corpus_eval_path: String,
    corpus_eval_sha256: String,
    datasets: Vec<StagedDataset>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagingReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    release_binary_sha256: String,
    corpus_eval_sha256: String,
    nodes: Vec<StagedNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct LabelAuthority {
    score_tier: String,
    metric: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    selection_bias: Option<String>,
    generation: String,
    frozen_seed_sha256: String,
    model_checkpoint_sha256: String,
    hall_of_fame_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ShardCandidate {
    semantic_sha256: String,
    canonical_genome: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CorpusShardManifest {
    schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    campaign_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    release_binary_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    comparator_lock_sha256: Option<String>,
    task: String,
    dataset_hash: String,
    train_csv: String,
    test_csv: String,
    train_sha256: String,
    test_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint_epoch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    catalog_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    protocol_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    generated_at_ms: Option<u128>,
    label_authority: LabelAuthority,
    meta_json: String,
    meta_sha256: String,
    horizon: usize,
    seasonality: usize,
    frequency: String,
    target_is_final_column: bool,
    known_future_covariates: Vec<String>,
    past_only_covariates: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reduction: Option<CorpusReductionSpec>,
    candidates: Vec<ShardCandidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CorpusReductionSpec {
    version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ShardRecord {
    generation: usize,
    shard: usize,
    node: String,
    attempt: usize,
    job_id: String,
    corpus_manifest_file: String,
    corpus_manifest_sha256: String,
    scq_spec_file: String,
    scq_spec_sha256: String,
    receipt_file: String,
    receipt_sha256: String,
    valid_candidates: usize,
    total_candidates: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GenerationManifest {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    campaign_sha256: String,
    release_binary_sha256: String,
    dataset_hash: String,
    generation: usize,
    members: Vec<Member>,
    shards: Vec<ShardRecord>,
    best_semantic_sha256: Option<String>,
    best_mase: Option<f64>,
    valid_candidates: usize,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WorkerReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    dataset_hash: String,
    manifest_sha256: String,
    receipt_sha256: String,
    done_sha256: String,
    rows: usize,
    valid_rows: usize,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WaveReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    campaign_sha256: String,
    dataset_hash: String,
    generation: usize,
    nodes: Vec<String>,
    job_ids: Vec<String>,
    corpus_manifest_sha256: Vec<String>,
    scq_spec_sha256: Vec<String>,
    terminal_rows: usize,
    submitted_at_unix_ms: u128,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug)]
struct PreparedShard {
    generation: usize,
    shard: usize,
    node: String,
    attempt: usize,
    job_id: String,
    remote_manifest_sha256: String,
    local_manifest_path: PathBuf,
    spec_path: PathBuf,
    spec_text: String,
    remote_output: String,
    receipt_path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct JobReceipt {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    job_id: String,
    node: String,
    attempt: usize,
    state: String,
    exit_code: i64,
    corpus_manifest_sha256: String,
    scq_spec_sha256: String,
    scq_show_sha256: String,
    worker_receipt_sha256: String,
    receipt_sha256: String,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Deserialize)]
struct EvalReceipt {
    task: String,
    dataset_hash: String,
    semantic_sha256: String,
    canonical_genome: String,
    metric: String,
    score: Option<f64>,
    secondary_metric: String,
    secondary_score: Option<f64>,
    valid: bool,
    failure: Option<String>,
    runtime_seconds: f64,
    predictions_sha256: Option<String>,
    rmse: Option<f64>,
    protocol_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TimeSeriesScore {
    mase: f64,
    rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LogicalSlot {
    generation: usize,
    slot: usize,
    member: Member,
    score: Option<TimeSeriesScore>,
    valid: bool,
    failure: Option<String>,
    receipt_file: String,
    receipt_sha256: String,
    prediction_sha256: Option<String>,
    runtime_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetResult {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_strict_winner: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_best_jope_mase: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    continuation_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    best_evidence_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    best_receipt_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    best_receipt_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    best_prediction_sha256: Option<String>,
    campaign_sha256: String,
    release_binary_sha256: String,
    dataset_hash: String,
    status: String,
    official_leader_model: String,
    official_leader_mase: f64,
    generation_zero_best_mase: Option<f64>,
    best_jope_mase: Option<f64>,
    best_jope_rmse: Option<f64>,
    mase_delta_vs_official: Option<f64>,
    strict_winner: bool,
    best_beats_official: bool,
    best_semantic_sha256: Option<String>,
    best_canonical_genome: Option<String>,
    best_generation: Option<usize>,
    best_slot: Option<usize>,
    seed_source_family: Option<String>,
    logical_evaluations: usize,
    valid_evaluations: usize,
    completed_generations: usize,
    completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TerminalSummary {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inherited_base_wins: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    improved_base_wins: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    continuation_strict_wins: Option<usize>,
    dataset_count: usize,
    wins: usize,
    losses: usize,
    ties: usize,
    exhausted: usize,
    logical_evaluations: usize,
    valid_evaluations: usize,
    datasets: Vec<DatasetResult>,
    sotarena_scan_output: Option<String>,
    completed_at_unix_ms: u128,
}

fn campaign_root(run_id: &str) -> PathBuf {
    Path::new(output_root_str()).join(run_id)
}

fn dataset_root(root: &Path, dataset: &str) -> PathBuf {
    root.join("datasets").join(dataset)
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

fn replace_stale_artifact(path: &Path, bytes: &[u8], stale_stem: &str) -> Result<String> {
    if path.exists() {
        if fs::read(path)? == bytes {
            return Ok(sha256_bytes(bytes));
        }
        let prior = fs::read(path)?;
        let prior_sha256 = sha256_bytes(&prior);
        let archive = path.with_file_name(format!(
            "{stale_stem}.stale-{}.v1.json",
            &prior_sha256[..16]
        ));
        immutable_or_equal(&archive, &prior)?;
        fs::remove_file(path).with_context(|| format!("remove stale {}", path.display()))?;
    }
    immutable_write(path, bytes)?;
    Ok(sha256_bytes(bytes))
}

fn write_current_json<T: Serialize>(path: &Path, value: &T, stale_stem: &str) -> Result<String> {
    let bytes = canonical_json(value)?;
    replace_stale_artifact(path, &bytes, stale_stem)
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path)?).with_context(|| format!("parse {}", path.display()))
}

fn finite_metric(value: &Value, key: &str) -> Option<f64> {
    value
        .get("metrics")
        .and_then(|metrics| metrics.get(key))
        .and_then(Value::as_f64)
        .filter(|score| score.is_finite())
}

fn string_array(value: &Value, key: &str) -> Result<Vec<String>> {
    let Some(array) = value.get(key) else {
        return Ok(Vec::new());
    };
    array
        .as_array()
        .with_context(|| format!("meta_json {key} must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .with_context(|| format!("meta_json {key} contains a non-string"))
        })
        .collect()
}

fn required_usize(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|raw| usize::try_from(raw).ok())
        .with_context(|| format!("meta_json missing numeric {key}"))
}

fn external_reports(dataset_dir: &Path, dataset_hash: &str) -> Result<Vec<ExternalReport>> {
    let reports = dataset_dir.join("reports");
    let mut paths = fs::read_dir(&reports)
        .with_context(|| format!("read {}", reports.display()))?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    paths.sort();
    let mut accepted = Vec::new();
    for path in paths {
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes = fs::read(&path)?;
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if value.get("task_type").and_then(Value::as_str) != Some("timeseries")
            || !matches!(
                value.get("status").and_then(Value::as_str),
                Some("completed" | "completed-valid")
            )
            || value.get("test_adaptive").and_then(Value::as_bool) == Some(true)
        {
            continue;
        }
        if value
            .get("dataset_hash")
            .and_then(Value::as_str)
            .is_some_and(|bound| bound != dataset_hash)
        {
            continue;
        }
        let model = value
            .get("model_name")
            .or_else(|| value.get("algorithm"))
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("unknown")
            })
            .to_string();
        if model.to_ascii_lowercase().contains("jope") {
            continue;
        }
        let Some(mase) = finite_metric(&value, "mase") else {
            continue;
        };
        let Some(rmse) = finite_metric(&value, "rmse") else {
            continue;
        };
        accepted.push(ExternalReport {
            model,
            mase,
            rmse,
            report_path: path.to_string_lossy().into_owned(),
            report_sha256: sha256_bytes(&bytes),
        });
    }
    if accepted.is_empty() {
        bail!(
            "{} has no finite completed non-Jope external time-series report",
            dataset_dir.display()
        );
    }
    Ok(accepted)
}

fn official_leader(accepted: &[ExternalReport], dataset_dir: &Path) -> Result<OfficialLeader> {
    let best = accepted.iter().min_by(|left, right| {
        left.mase
            .total_cmp(&right.mase)
            .then(left.rmse.total_cmp(&right.rmse))
            .then(left.model.cmp(&right.model))
            .then(left.report_sha256.cmp(&right.report_sha256))
    });
    best.map(|report| OfficialLeader {
        model: report.model.clone(),
        mase: report.mase,
        rmse: report.rmse,
        report_path: report.report_path.clone(),
        report_sha256: report.report_sha256.clone(),
    })
    .with_context(|| {
        format!(
            "{} has no finite non-Jope official time-series MASE report",
            dataset_dir.display()
        )
    })
}

fn embedded_payload_sha256(value: &Value, key: &str) -> Result<String> {
    let embedded = value
        .get(key)
        .and_then(Value::as_str)
        .filter(|digest| digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .with_context(|| format!("missing or invalid {key}"))?
        .to_string();
    let mut unsigned = value.clone();
    unsigned
        .as_object_mut()
        .context("hashed payload must be an object")?
        .remove(key);
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("{key} digest drift");
    }
    Ok(embedded)
}

fn validate_coverage_final_receipt(
    path: &Path,
    campaign_id: &str,
    manifest_sha256: &str,
    task: &str,
    expected: usize,
) -> Result<String> {
    let value: Value = load_json(path)?;
    if value.get("schema").and_then(Value::as_str) != Some("coverage.final-receipt.v1")
        || value.get("campaign_id").and_then(Value::as_str) != Some(campaign_id)
        || value
            .get("campaign_manifest_sha256")
            .and_then(Value::as_str)
            != Some(manifest_sha256)
        || value.get("task").and_then(Value::as_str) != Some(task)
        || value.get("expected").and_then(Value::as_u64) != Some(expected as u64)
        || value.get("valid").and_then(Value::as_u64) != Some(expected as u64)
        || value.get("missing").and_then(Value::as_u64) != Some(0)
        || value.get("invalid").and_then(Value::as_u64) != Some(0)
        || value.get("complete").and_then(Value::as_bool) != Some(true)
        || value.get("reports").and_then(Value::as_array).map(Vec::len) != Some(expected)
    {
        bail!("invalid or incomplete coverage receipt {}", path.display());
    }
    let receipt_sha256 = embedded_payload_sha256(&value, "receipt_sha256")?;
    validate_receipt_report_files(&value, expected, true)?;
    Ok(receipt_sha256)
}

fn validate_timeseries_baseline_receipt() -> Result<String> {
    let value: Value = load_json(Path::new(ts_baseline_receipt_str()))?;
    if value.get("schema").and_then(Value::as_str)
        != Some("timeseries42.baseline-all21-final-receipt.v1")
        || value.get("parent_manifest_sha256").and_then(Value::as_str)
            != Some("75194fba4d85aefd9e065cd71bd95d3b9a9744cace74a7e2aa92859887ca8885")
        || value.get("repair_manifest_sha256").and_then(Value::as_str)
            != Some(TS_COVERAGE_MANIFEST_SHA256)
        || value.get("expected").and_then(Value::as_u64) != Some(882)
        || value.get("completed_numeric").and_then(Value::as_u64) != Some(877)
        || value.get("skipped_ineligible").and_then(Value::as_u64) != Some(5)
        || value.get("missing").and_then(Value::as_u64) != Some(0)
        || value.get("invalid").and_then(Value::as_u64) != Some(0)
        || value.get("errored").and_then(Value::as_u64) != Some(0)
        || value.get("complete").and_then(Value::as_bool) != Some(true)
        || value.get("reports").and_then(Value::as_array).map(Vec::len) != Some(882)
    {
        bail!("invalid exact 877+5 time-series baseline receipt");
    }
    let receipt_sha256 = embedded_payload_sha256(&value, "receipt_sha256")?;
    validate_receipt_report_files(&value, 882, false)?;
    Ok(receipt_sha256)
}

fn validate_coverage_binding() -> Result<CoverageBinding> {
    for (path, schema, campaign_id, expected_sha256) in [
        (
            mc_coverage_manifest_str(),
            "multiclass-union-runnable-v5",
            "mc-union-runnable272-v5",
            MC_COVERAGE_MANIFEST_SHA256,
        ),
        (
            ts_coverage_manifest_str(),
            "timeseries42-forecast-campaign-v10",
            "timeseries42-forecast3-v10",
            TS_COVERAGE_MANIFEST_SHA256,
        ),
    ] {
        let value: Value = load_json(Path::new(path))?;
        if value.get("schema_version").and_then(Value::as_str) != Some(schema)
            || value.get("campaign_id").and_then(Value::as_str) != Some(campaign_id)
            || embedded_payload_sha256(&value, "manifest_sha256")? != expected_sha256
        {
            bail!("coverage successor manifest drift at {path}");
        }
    }
    let multiclass_final_receipt_sha256 = validate_coverage_final_receipt(
        Path::new(mc_coverage_receipt_str()),
        "mc-union-runnable272-v5",
        MC_COVERAGE_MANIFEST_SHA256,
        "multiclass",
        11_696,
    )?;
    let timeseries_final_receipt_sha256 = validate_coverage_final_receipt(
        Path::new(ts_coverage_receipt_str()),
        "timeseries42-forecast3-v10",
        TS_COVERAGE_MANIFEST_SHA256,
        "timeseries",
        126,
    )?;
    let timeseries_baseline_receipt_sha256 = validate_timeseries_baseline_receipt()?;
    let set: Value = load_json(Path::new(coverage_comparator_set_str()))?;
    if set.get("schema").and_then(Value::as_str) != Some("test-adaptive-comparator-set.v2")
        || set.get("root").and_then(Value::as_str) != Some(coverage_comparator_root_str())
        || set
            .get("multiclass_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(multiclass_final_receipt_sha256.as_str())
        || set
            .get("timeseries_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(timeseries_final_receipt_sha256.as_str())
        || set
            .get("timeseries_baseline_receipt_sha256")
            .and_then(Value::as_str)
            != Some(timeseries_baseline_receipt_sha256.as_str())
        || set.get("lock_count").and_then(Value::as_u64) != Some(314)
        || set.get("locks").and_then(Value::as_array).map(Vec::len) != Some(314)
    {
        bail!("post-coverage comparator set identity drift");
    }
    let comparator_set_sha256 = embedded_payload_sha256(&set, "comparator_set_sha256")?;
    Ok(CoverageBinding {
        multiclass_manifest_sha256: MC_COVERAGE_MANIFEST_SHA256.to_string(),
        multiclass_final_receipt_sha256,
        timeseries_manifest_sha256: TS_COVERAGE_MANIFEST_SHA256.to_string(),
        timeseries_final_receipt_sha256,
        timeseries_baseline_receipt_sha256,
        comparator_root: coverage_comparator_root_str().to_string(),
        comparator_set_path: coverage_comparator_set_str().to_string(),
        comparator_set_sha256,
    })
}

fn bind_forecast3_comparators(specs: &mut [DatasetSpec], coverage: &CoverageBinding) -> Result<()> {
    for spec in specs {
        let authorized = BASELINE_MODELS
            .iter()
            .copied()
            .filter(|model| !declared_baseline_skip(&spec.dataset_hash, model))
            .collect::<BTreeSet<_>>();
        let path = Path::new(coverage_comparator_root_str())
            .join("timeseries")
            .join(&spec.dataset_hash)
            .join("comparator-lock.v2.json");
        let lock: Value = load_json(&path)?;
        if lock.get("schema").and_then(Value::as_str) != Some("test-adaptive-comparator-lock-v2")
            || lock.get("task").and_then(Value::as_str) != Some("timeseries")
            || lock.get("dataset_hash").and_then(Value::as_str) != Some(spec.dataset_hash.as_str())
            || lock.get("run_id").and_then(Value::as_str) != Some(FORECAST3_RUN_ID)
            || lock.get("coverage_campaign_id").and_then(Value::as_str)
                != Some("timeseries42-forecast3-v10")
            || lock.get("coverage_manifest_sha256").and_then(Value::as_str)
                != Some(TS_COVERAGE_MANIFEST_SHA256)
            || lock
                .get("coverage_final_receipt_sha256")
                .and_then(Value::as_str)
                != Some(coverage.timeseries_final_receipt_sha256.as_str())
            || lock
                .get("timeseries_baseline_receipt_sha256")
                .and_then(Value::as_str)
                != Some(coverage.timeseries_baseline_receipt_sha256.as_str())
            || lock.get("accepted_report_count").and_then(Value::as_u64)
                != Some(authorized.len() as u64)
        {
            bail!(
                "timeseries comparator lock identity drift at {}",
                path.display()
            );
        }
        let protocol_sha256 = lock
            .get("protocol_sha256")
            .and_then(Value::as_str)
            .context("timeseries comparator lacks protocol_sha256")?;
        let mut protocol = lock
            .get("protocol")
            .cloned()
            .context("timeseries comparator lacks protocol")?;
        let evaluator_sha256 = current_corpus_eval_sha256()?;
        if protocol.get("protocol_sha256").and_then(Value::as_str) != Some(protocol_sha256)
            || protocol
                .pointer("/evaluator/sha256")
                .and_then(Value::as_str)
                != Some(evaluator_sha256.as_str())
            || protocol.get("minimum_generations").and_then(Value::as_u64) != Some(0)
            || protocol.get("maximum_generations").and_then(Value::as_u64) != Some(20)
            || protocol.get("population").and_then(Value::as_u64) != Some(60)
        {
            bail!("timeseries comparator protocol drift at {}", path.display());
        }
        protocol
            .as_object_mut()
            .context("timeseries comparator protocol must be an object")?
            .remove("protocol_sha256");
        if sha256_bytes(&serde_json::to_vec(&protocol)?) != protocol_sha256 {
            bail!(
                "timeseries comparator protocol digest drift at {}",
                path.display()
            );
        }
        let files = lock
            .get("dataset_file_hashes")
            .and_then(Value::as_object)
            .context("timeseries comparator lacks dataset file hashes")?;
        if files.get("train.csv").and_then(Value::as_str) != Some(spec.train_sha256.as_str())
            || files.get("test.csv").and_then(Value::as_str) != Some(spec.test_sha256.as_str())
            || files.get("meta.json").and_then(Value::as_str) != Some(spec.meta_sha256.as_str())
        {
            bail!("timeseries comparator input drift at {}", path.display());
        }
        let accepted = lock
            .get("accepted_reports")
            .and_then(Value::as_array)
            .context("timeseries comparator lacks accepted reports")?;
        let mut reports = Vec::with_capacity(authorized.len());
        for row in accepted {
            let model = row
                .get("model")
                .and_then(Value::as_str)
                .context("timeseries comparator report lacks model")?;
            let report_path = row
                .get("report_path")
                .and_then(Value::as_str)
                .context("timeseries comparator report lacks path")?;
            let report_sha256 = row
                .get("report_sha256")
                .and_then(Value::as_str)
                .context("timeseries comparator report lacks digest")?;
            let mase = row
                .get("primary")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite());
            let rmse = row
                .get("secondary")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite());
            if !authorized.contains(model)
                || sha256_file(Path::new(report_path))? != report_sha256
                || mase.is_none()
                || rmse.is_none()
            {
                bail!("invalid authorized comparator report at {}", path.display());
            }
            reports.push(ExternalReport {
                model: model.to_string(),
                report_path: report_path.to_string(),
                report_sha256: report_sha256.to_string(),
                mase: mase.unwrap(),
                rmse: rmse.unwrap(),
            });
        }
        if reports.len() != authorized.len()
            || reports
                .iter()
                .map(|row| row.model.as_str())
                .collect::<BTreeSet<_>>()
                != authorized
        {
            bail!(
                "timeseries comparator report membership drift at {}",
                path.display()
            );
        }
        reports.sort_by(|left, right| {
            left.mase
                .total_cmp(&right.mase)
                .then(left.rmse.total_cmp(&right.rmse))
                .then(left.model.cmp(&right.model))
                .then(left.report_sha256.cmp(&right.report_sha256))
        });
        spec.official_leader = official_leader(&reports, Path::new(&spec.train_path))?;
        spec.accepted_external_reports = reports;
        spec.comparator_lock_sha256 = embedded_payload_sha256(&lock, "lock_sha256")?;
    }
    Ok(())
}

fn declared_baseline_skip(dataset_hash: &str, model: &str) -> bool {
    matches!(
        (dataset_hash, model),
        ("080d2674945eb5d7", "autogluon_timeseries")
            | ("c400e4876a9046ae", "autogluon_timeseries")
            | ("c43cfcfc9313d051", "autogluon_timeseries")
            | ("080d2674945eb5d7", "sktime_reduction_lightgbm")
            | ("c43cfcfc9313d051", "sktime_reduction_lightgbm")
    )
}

fn verify_allmodels_baseline(
    dataset_dir: &Path,
    dataset_hash: &str,
    accepted: &[ExternalReport],
) -> Result<()> {
    let accepted_models = accepted
        .iter()
        .map(|row| row.model.as_str())
        .collect::<BTreeSet<_>>();
    for model in BASELINE_MODELS {
        let path = dataset_dir.join("reports").join(format!("{model}.json"));
        let value: Value = serde_json::from_slice(
            &fs::read(&path)
                .with_context(|| format!("missing baseline report {}", path.display()))?,
        )
        .with_context(|| format!("parse baseline report {}", path.display()))?;
        if value.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
            || value.get("task_type").and_then(Value::as_str) != Some("timeseries")
            || value.get("test_adaptive").and_then(Value::as_bool) == Some(true)
        {
            bail!("baseline report identity drift at {}", path.display());
        }
        let status = value.get("status").and_then(Value::as_str).unwrap_or("");
        if declared_baseline_skip(dataset_hash, model) {
            if status != "skipped-ineligible" {
                bail!("declared eligibility terminal drift at {}", path.display());
            }
        } else if !matches!(status, "completed" | "completed-valid")
            || finite_metric(&value, "mase").is_none()
            || finite_metric(&value, "rmse").is_none()
            || !accepted_models.contains(model)
        {
            bail!("numeric baseline is incomplete at {}", path.display());
        }
    }
    let expected_numeric = BASELINE_MODELS
        .iter()
        .filter(|model| !declared_baseline_skip(dataset_hash, model))
        .copied()
        .collect::<BTreeSet<_>>();
    if accepted_models != expected_numeric {
        bail!("baseline comparator envelope drift for {dataset_hash}");
    }
    Ok(())
}

fn load_dataset_specs(require_allmodels_baseline: bool) -> Result<Vec<DatasetSpec>> {
    let registry = Path::new(registry_str());
    let mut dirs = fs::read_dir(registry)
        .with_context(|| format!("read {registry_str}", registry_str = registry_str()))?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    dirs.retain(|path| path.is_dir());
    dirs.sort();
    if dirs.len() != 42 {
        bail!(
            "TimeSeries42 expected 42 dataset directories, found {}",
            dirs.len()
        );
    }
    let mut specs = Vec::with_capacity(dirs.len());
    for dir in dirs {
        let dataset_hash = dir
            .file_name()
            .and_then(|name| name.to_str())
            .context("time-series dataset path has no UTF-8 name")?
            .to_string();
        let train_path = dir.join("train.csv");
        let test_path = dir.join("test.csv");
        let meta_path = dir.join("meta.json");
        for path in [&train_path, &test_path, &meta_path] {
            if !path.is_file() {
                bail!(
                    "TimeSeries42 dataset {dataset_hash} is missing {}",
                    path.display()
                );
            }
        }
        let meta_bytes = fs::read(&meta_path)?;
        let meta: Value = serde_json::from_slice(&meta_bytes)
            .with_context(|| format!("parse {}", meta_path.display()))?;
        let horizon = required_usize(&meta, "horizon")?;
        let seasonality = required_usize(&meta, "seasonality")?;
        let frequency = meta
            .get("frequency")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .context("meta_json missing frequency")?
            .to_string();
        let target_is_final_column = meta
            .get("target_is_final_column")
            .and_then(Value::as_bool)
            .context("meta_json missing target_is_final_column")?;
        let known_future_covariates = string_array(&meta, "known_dynamic_columns")?;
        let past_only_covariates = string_array(&meta, "past_dynamic_columns_excluded")?;
        let manifest_spec = TimeSeriesManifestSpec {
            horizon,
            seasonality,
            frequency: frequency.clone(),
            target_is_final_column,
            known_future_covariates: known_future_covariates.clone(),
            past_only_covariates: Vec::new(),
        };
        let train = read_timeseries_csv(&train_path)?;
        let test = read_timeseries_csv(&test_path)?;
        let train_rows = train.y.len();
        let test_rows = test.y.len();
        let csv_features = train.features;
        let data = validate_timeseries_data(train, test, &meta, &manifest_spec)
            .with_context(|| format!("validate time-series data for {dataset_hash}"))?;
        let reduction_features = make_supervised_reduction(&data)?.feature_names.len();
        let accepted_external_reports = external_reports(&dir, &dataset_hash)?;
        if require_allmodels_baseline {
            verify_allmodels_baseline(&dir, &dataset_hash, &accepted_external_reports)?;
        }
        let official_leader = official_leader(&accepted_external_reports, &dir)?;
        specs.push(DatasetSpec {
            schema: DATASET_SCHEMA.to_string(),
            dataset_hash,
            display_name: meta
                .get("display_name")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            category: meta
                .get("category")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            train_path: train_path.to_string_lossy().into_owned(),
            train_sha256: sha256_file(&train_path)?,
            test_path: test_path.to_string_lossy().into_owned(),
            test_sha256: sha256_file(&test_path)?,
            meta_path: meta_path.to_string_lossy().into_owned(),
            meta_sha256: sha256_bytes(&meta_bytes),
            train_rows,
            test_rows,
            csv_features,
            reduction_features,
            horizon,
            seasonality,
            frequency,
            target_is_final_column,
            known_future_covariates,
            past_only_covariates,
            source_real_data: meta
                .get("source_real_data")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            synthetic_origin: meta
                .get("synthetic_origin")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            reduction_version: String::new(),
            accepted_external_reports,
            comparator_lock_sha256: String::new(),
            official_leader,
        });
    }
    Ok(specs)
}

fn ordered_ids(specs: &[DatasetSpec]) -> Vec<String> {
    specs.iter().map(|spec| spec.dataset_hash.clone()).collect()
}

fn terminal_op(genome: &GenomeIr) -> Option<&str> {
    match genome {
        GenomeIr::Pipeline(pipeline) => pipeline.stages.last().map(|stage| stage.op.as_str()),
        _ => None,
    }
}

fn is_binary_head(op: &str) -> bool {
    matches!(op, "logit_ridge" | "logistic")
}

fn is_head_like(op: &str) -> bool {
    is_head_op(op)
        || matches!(
            op,
            "tab_boost" | "extra_trees" | "gam_boost" | "pls" | "logit_ridge"
        )
}

fn rewrite_binary_body_to_regression(canonical: &str) -> Result<String> {
    let genome = GenomeIr::parse(canonical).context("parse binary transfer genome")?;
    let GenomeIr::Pipeline(mut pipeline) = genome else {
        bail!("binary transfer genome is not a pipeline");
    };
    while pipeline
        .stages
        .last()
        .is_some_and(|stage| is_head_like(&stage.op) || is_binary_head(&stage.op))
    {
        pipeline.stages.pop();
    }
    pipeline.stages.retain(|stage| !is_target_op(&stage.op));
    let mut text = String::from("(pipe X");
    for stage in pipeline.stages {
        text.push(' ');
        text.push_str(&stage.to_string());
    }
    text.push_str(" (ridge))");
    Ok(GenomeIr::parse(&text)?.canonical())
}

fn timeseries_schema(spec: &DatasetSpec) -> DataSchema {
    let mut names = spec.known_future_covariates.clone();
    if names.is_empty() {
        names.extend((0..spec.csv_features).map(|column| format!("known_future_{column}")));
    }
    names.extend(
        [
            "lag_1",
            "lag_2",
            "lag_3",
            "seasonal_lag_1",
            "seasonal_lag_2",
            "rolling_mean_3",
            "rolling_mean_seasonal",
            "rolling_absdiff_3",
            "rolling_absdiff_seasonal",
            "row_index",
            "horizon_index",
            "seasonal_phase",
            "seasonal_sin",
            "seasonal_cos",
        ]
        .into_iter()
        .map(str::to_string),
    );
    DataSchema {
        columns: names.into_iter().map(ColumnSpec::numeric).collect(),
        row_semantics: RowSemantics::Iid,
    }
}

fn timeseries_schema_v2(spec: &DatasetSpec) -> DataSchema {
    DataSchema {
        columns: direct_feature_names_for(&spec.known_future_covariates, spec.csv_features)
            .into_iter()
            .map(ColumnSpec::numeric)
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

fn canonicalize_for_timeseries(source: &str, spec: &DatasetSpec) -> Result<CanonicalGenome> {
    // v1 keeps the pinned spec.reduction_features count; v2 derives its width
    // from the manifest-level facts so the pinned v1 count stays untouched.
    let (schema, features) = match spec.reduction()? {
        ReductionVersion::V1 => (timeseries_schema(spec), spec.reduction_features),
        ReductionVersion::V2 => {
            let schema = timeseries_schema_v2(spec);
            let features = schema.columns.len();
            (schema, features)
        }
    };
    let canonical = canonicalize_for_features(source, features)?;
    let genome = GenomeIr::parse(&canonical.canonical)?;
    if terminal_op(&genome).is_some_and(is_binary_head) {
        bail!("binary classification head is not valid for timeseries");
    }
    let task = TaskSpec::regression(Objective::Rmse)?;
    validate_v3_genome(&genome, &task, &schema)?;
    Ok(canonical)
}

fn hand_authored_sources() -> Result<Vec<CandidateSource>> {
    let path = PathBuf::from(file!());
    let file_sha256 = sha256_file(&workspace_root().join(file!())).unwrap_or_default();
    let priors = [
        "(pipe X (ridge))",
        "(pipe X (standardize) (ridge))",
        "(pipe X (robust_scale) (ridge))",
        "(pipe X (standardize) (elasticnet alpha=0.001 l1_ratio=0.2))",
        "(pipe X (standardize) (elasticnet alpha=0.01 l1_ratio=0.5))",
        "(pipe X (row_stats append=true) (ridge))",
        "(pipe X (pair_prod append=true topk=8) (ridge))",
        "(pipe X (rff_sin append=true gamma=0.1 n=16 seed=13) (ridge))",
        "(pipe X (rff_cos append=true gamma=0.1 n=16 seed=29) (ridge))",
        "(pipe X (hinge append=true knots=4) (ridge))",
        "(pipe X (winsor_clip append=true q=0.05) (ridge))",
        "(pipe X (quantile_bins append=true bins=8) (ridge))",
        "(pipe X (blom append=true) (ridge))",
        "(pipe X (standardize) (histgbm depth=3 lr=0.05 rounds=80))",
        "(pipe X (robust_scale) (histgbm2 leaves=31 lr=0.04 rounds=300))",
    ];
    Ok(priors
        .iter()
        .enumerate()
        .map(|(record, genome)| CandidateSource {
            genome: (*genome).to_string(),
            kind: "hand_authored_timeseries_prior".to_string(),
            path: path.clone(),
            file_sha256: file_sha256.clone(),
            record,
            rank: record + 1,
            score: None,
            order_score: record as f64,
        })
        .collect())
}

fn binary_rewrite_sources() -> Result<Vec<CandidateSource>> {
    let path = workspace_root().join("seeds/generation-zero-binary.txt");
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let file_sha256 = sha256_bytes(&bytes);
    let text = std::str::from_utf8(&bytes)?;
    let mut sources = Vec::new();
    for (record, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(genome) = rewrite_binary_body_to_regression(line) {
            sources.push(CandidateSource {
                genome,
                kind: "binary_body_to_regression_rewrite".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: record + 1,
                score: None,
                order_score: record as f64,
            });
        }
    }
    Ok(sources)
}

fn sotarena_scan_sources() -> Result<Vec<CandidateSource>> {
    let root = paths::legacy_tree("sotarena_pull").join("out/scans");
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(root)? {
        let path = entry?.path().join("new_candidates.jsonl");
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    let mut sources = Vec::new();
    for path in files {
        let bytes = fs::read(&path)?;
        let file_sha256 = sha256_bytes(&bytes);
        let text = std::str::from_utf8(&bytes)?;
        for (record, line) in text.lines().enumerate() {
            let value: Value = match serde_json::from_str(line) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let Some(genome) = value.get("canonical_genome").and_then(Value::as_str) else {
                continue;
            };
            let margin = value
                .get("margin_vs_best_comparator")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite())
                .unwrap_or(0.0);
            sources.push(CandidateSource {
                genome: genome.to_string(),
                kind: "sotarena_high_margin_seed".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: record + 1,
                score: Some(margin),
                order_score: -margin,
            });
        }
    }
    sources.sort_by(|left, right| {
        left.order_score
            .total_cmp(&right.order_score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.record.cmp(&right.record))
    });
    sources.truncate(128);
    Ok(sources)
}

fn local_flywheel_elite_sources() -> Result<Vec<CandidateSource>> {
    let roots = [
        paths::flywheel_root().join("receipts"),
        paths::runs_root(),
        paths::legacy_tree("jope-test-adaptive"),
    ];
    let mut receipt_files = Vec::new();
    for root in roots {
        if !root.exists() {
            continue;
        }
        let mut stack = vec![(root.to_path_buf(), 0usize)];
        while let Some((dir, depth)) = stack.pop() {
            if depth > 6 {
                continue;
            }
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push((path, depth + 1));
                } else if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(".receipts.jsonl"))
                {
                    receipt_files.push(path);
                }
            }
        }
    }
    receipt_files.sort();
    let mut sources = Vec::new();
    for path in receipt_files {
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
            let task = value
                .get("task")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let order_score = match task {
                "timeseries" => value
                    .get("secondary_score")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .unwrap_or(f64::INFINITY),
                "regression" => -value
                    .get("score")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .unwrap_or(f64::NEG_INFINITY),
                "binary" => -value
                    .get("score")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .unwrap_or(f64::NEG_INFINITY),
                _ => continue,
            };
            sources.push(CandidateSource {
                genome: genome.to_string(),
                kind: "local_flywheel_elite".to_string(),
                path: path.clone(),
                file_sha256: file_sha256.clone(),
                record,
                rank: record + 1,
                score: order_score.is_finite().then_some(order_score),
                order_score,
            });
        }
    }
    sources.sort_by(|left, right| {
        left.order_score
            .total_cmp(&right.order_score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.record.cmp(&right.record))
    });
    sources.truncate(256);
    Ok(sources)
}

fn cached_sources<F>(
    cell: &'static OnceLock<std::result::Result<Vec<CandidateSource>, String>>,
    build: F,
) -> Result<Vec<CandidateSource>>
where
    F: FnOnce() -> Result<Vec<CandidateSource>>,
{
    match cell.get_or_init(|| build().map_err(|error| format!("{error:#}"))) {
        Ok(sources) => Ok(sources.clone()),
        Err(error) => bail!("{error}"),
    }
}

fn cached_fast30_bank_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, fast30_bank_sources)
}

fn cached_binary_rewrite_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, binary_rewrite_sources)
}

fn cached_sotarena_scan_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, sotarena_scan_sources)
}

fn cached_local_flywheel_elite_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, local_flywheel_elite_sources)
}

fn cached_hand_authored_sources() -> Result<Vec<CandidateSource>> {
    static CACHE: OnceLock<std::result::Result<Vec<CandidateSource>, String>> = OnceLock::new();
    cached_sources(&CACHE, hand_authored_sources)
}

fn admit_seed_sources(
    spec: &DatasetSpec,
    sources: Vec<CandidateSource>,
    seen: &mut BTreeSet<String>,
    seeds: &mut Vec<SeedRecord>,
) {
    for source in sources {
        let canonical = match canonicalize_for_timeseries(&source.genome, spec) {
            Ok(canonical) => canonical,
            Err(_) => continue,
        };
        if !seen.insert(canonical.semantic_sha256.clone()) {
            continue;
        }
        seeds.push(SeedRecord {
            slot: seeds.len(),
            canonical_genome: canonical.canonical,
            semantic_sha256: canonical.semantic_sha256,
            complexity: canonical.complexity,
            rust_source_len: canonical.rust_source_len,
            source_family: source.kind,
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
}

fn build_seed_bank(spec: &DatasetSpec) -> Result<SeedBank> {
    let mut seen = BTreeSet::new();
    let mut seeds = Vec::new();
    if let Ok(mut sources) = confirmed_transfer_winner_sources(&spec.dataset_hash) {
        admit_seed_sources(spec, std::mem::take(&mut sources), &mut seen, &mut seeds);
    }
    if seeds.len() < POPULATION {
        if let Ok(sources) = cached_fast30_bank_sources() {
            admit_seed_sources(spec, sources, &mut seen, &mut seeds);
        }
    }
    if seeds.len() < POPULATION {
        if let Ok(sources) = cached_binary_rewrite_sources() {
            admit_seed_sources(spec, sources, &mut seen, &mut seeds);
        }
    }
    if seeds.len() < POPULATION {
        if let Ok(sources) = cached_sotarena_scan_sources() {
            admit_seed_sources(spec, sources, &mut seen, &mut seeds);
        }
    }
    if seeds.len() < POPULATION {
        if let Ok(sources) = cached_local_flywheel_elite_sources() {
            admit_seed_sources(spec, sources, &mut seen, &mut seeds);
        }
    }
    if seeds.len() < POPULATION {
        admit_seed_sources(spec, cached_hand_authored_sources()?, &mut seen, &mut seeds);
    }
    if seeds.len() != POPULATION {
        bail!(
            "only {} valid, deduplicated TimeSeries42 generation-zero seeds were found for {}",
            seeds.len(),
            spec.dataset_hash
        );
    }
    Ok(SeedBank {
        schema: SEED_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        dataset_hash: spec.dataset_hash.clone(),
        population: POPULATION,
        selection:
            "deterministic provenance-ordered canonicalization, task admission, semantic dedup"
                .to_string(),
        seeds,
    })
}

fn load_or_create_seed_bank(root: &Path, spec: &DatasetSpec) -> Result<(SeedBank, String)> {
    let path = dataset_root(root, &spec.dataset_hash)
        .join("seeds")
        .join("generation-zero.v1.json");
    if path.is_file() {
        return read_seed_bank(&path, spec);
    }
    let bank = build_seed_bank(spec)?;
    let sha = write_json(&path, &bank)?;
    Ok((bank, sha))
}

fn read_seed_bank(path: &Path, spec: &DatasetSpec) -> Result<(SeedBank, String)> {
    let bytes = fs::read(path)?;
    let bank: SeedBank = serde_json::from_slice(&bytes)?;
    if bytes != canonical_json(&bank)?
        || bank.schema != SEED_SCHEMA
        || !bank.test_adaptive
        || bank.benchmark_valid
        || bank.dataset_hash != spec.dataset_hash
        || bank.population != POPULATION
        || bank.seeds.len() != POPULATION
        || bank
            .seeds
            .iter()
            .enumerate()
            .any(|(slot, seed)| seed.slot != slot)
    {
        bail!("invalid TimeSeries42 seed bank for {}", spec.dataset_hash);
    }
    Ok((bank, sha256_bytes(&bytes)))
}

fn load_existing_seed_bank(root: &Path, spec: &DatasetSpec) -> Result<(SeedBank, String)> {
    let path = dataset_root(root, &spec.dataset_hash)
        .join("seeds")
        .join("generation-zero.v1.json");
    if !path.is_file() {
        bail!(
            "base TimeSeries42 seed bank is missing for {}",
            spec.dataset_hash
        );
    }
    read_seed_bank(&path, spec)
}

fn seed_population_bindings(
    root: &Path,
    specs: &[DatasetSpec],
    create: bool,
) -> Result<Vec<SeedPopulationBinding>> {
    specs
        .iter()
        .map(|spec| {
            let path = dataset_root(root, &spec.dataset_hash)
                .join("seeds")
                .join("generation-zero.v1.json");
            let (_, seed_bank_sha256) = if create {
                load_or_create_seed_bank(root, spec)?
            } else {
                load_existing_seed_bank(root, spec)?
            };
            Ok(SeedPopulationBinding {
                dataset_hash: spec.dataset_hash.clone(),
                seed_bank_path: path.display().to_string(),
                seed_bank_sha256,
            })
        })
        .collect()
}

fn current_corpus_eval_sha256() -> Result<String> {
    sha256_file(Path::new(corpus_eval_str()))
}

fn remote_bundle_root(release_binary_sha256: &str) -> String {
    format!(
        "{output_root_str}/bundles/{release_binary_sha256}",
        output_root_str = output_root_str()
    )
}

fn remote_dataset_root(release_binary_sha256: &str, dataset_hash: &str) -> String {
    format!(
        "{}/datasets/{dataset_hash}",
        remote_bundle_root(release_binary_sha256)
    )
}

fn node_marker_path(bundle_root: &str, hostname: &str) -> String {
    format!("{bundle_root}/nodes/{hostname}.ready")
}

fn controller_hostname() -> Result<String> {
    Ok(String::from_utf8(
        scq_batch::command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "TimeSeries42 controller hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string())
}

fn is_local_scq_node(node: &str, controller_hostname: &str) -> bool {
    node == LOCAL_SCQ_NODE || node == controller_hostname
}

fn is_controller_node(node: &str) -> Result<bool> {
    Ok(is_local_scq_node(node, &controller_hostname()?))
}

fn validate_local_checked(
    path: &Path,
    expected_sha256: &str,
    expected_permissions: u32,
) -> Result<()> {
    let metadata = fs::symlink_metadata(path).with_context(|| {
        format!(
            "local TimeSeries42 staging path is absent: {}",
            path.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!(
            "local TimeSeries42 staging path is not a regular file: {}",
            path.display()
        );
    }
    let mode = metadata.permissions().mode() & 0o7777;
    if mode != expected_permissions {
        bail!(
            "local TimeSeries42 staging mode mismatch on {}: got {mode:#o}, expected {expected_permissions:#o}",
            path.display()
        );
    }
    let actual_sha256 = sha256_file(path)?;
    if actual_sha256 != expected_sha256 {
        bail!(
            "local TimeSeries42 staging hash mismatch on {}: got {actual_sha256}, expected {expected_sha256}",
            path.display()
        );
    }
    Ok(())
}

fn copy_local_checked(source: &Path, destination: &Path, executable: bool) -> Result<()> {
    let expected_sha256 = sha256_file(source)?;
    let expected_permissions = staging_file_mode(executable);
    if destination.exists() {
        let validation =
            validate_local_checked(destination, &expected_sha256, expected_permissions);
        if validation.is_ok() {
            return Ok(());
        }
        fs::remove_file(destination).with_context(|| {
            format!(
                "remove invalid local TimeSeries42 staging file after mismatch: {:#}",
                validation.unwrap_err()
            )
        })?;
    }
    fs::create_dir_all(
        destination
            .parent()
            .context("local TimeSeries42 destination has no parent")?,
    )?;
    fs::copy(source, destination)
        .with_context(|| format!("copy {} to {}", source.display(), destination.display()))?;
    let permissions = std::os::unix::fs::PermissionsExt::from_mode(expected_permissions);
    fs::set_permissions(destination, permissions)?;
    validate_local_checked(destination, &expected_sha256, expected_permissions)
}

fn ssh_raw_output(node: &str, remote_args: &[&str], label: &str) -> Result<std::process::Output> {
    let mut command = Command::new("ssh");
    command.arg(node).arg("--").args(remote_args);
    command.output().with_context(|| format!("launch {label}"))
}

fn remote_path_exists(node: &str, path: &str) -> Result<bool> {
    let exists = ssh_raw_output(
        node,
        &["test", "-e", path],
        "probe TimeSeries42 remote path",
    )?;
    if exists.status.success() {
        return Ok(true);
    }
    if exists.status.code() != Some(1) {
        bail!(
            "TimeSeries42 remote existence probe failed on {node}:{path} ({}): {}",
            exists.status,
            String::from_utf8_lossy(&exists.stderr).trim()
        );
    }
    let symlink = ssh_raw_output(
        node,
        &["test", "-L", path],
        "probe TimeSeries42 remote symlink",
    )?;
    if symlink.status.success() {
        bail!("TimeSeries42 remote immutable path is a symbolic link: {node}:{path}");
    }
    if symlink.status.code() != Some(1) {
        bail!(
            "TimeSeries42 remote symlink probe failed on {node}:{path} ({}): {}",
            symlink.status,
            String::from_utf8_lossy(&symlink.stderr).trim()
        );
    }
    Ok(false)
}

fn staging_file_mode(executable: bool) -> u32 {
    if executable {
        0o555
    } else {
        0o444
    }
}

fn remove_remote_path(node: &str, path: &str) -> Result<()> {
    scq_batch::ssh_output(
        node,
        &["rm", "-f", "--", path],
        "remove TimeSeries42 remote staging file",
    )?;
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn upload_remote_file(node: &str, source: &Path, destination: &str) -> Result<()> {
    let remote_command = format!("cat > {}", shell_quote(destination));
    let mut child = Command::new("timeout")
        .arg("600s")
        .arg("ssh")
        .arg("-T")
        .arg(node)
        .arg(remote_command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("start TimeSeries42 remote upload to {node}:{destination}"))?;
    {
        let mut input = child.stdin.take().context("remote upload stdin missing")?;
        let mut file =
            fs::File::open(source).with_context(|| format!("open {}", source.display()))?;
        std::io::copy(&mut file, &mut input)
            .with_context(|| format!("stream {} to {node}:{destination}", source.display()))?;
    }
    let output = child
        .wait_with_output()
        .with_context(|| format!("finish TimeSeries42 remote upload to {node}:{destination}"))?;
    if !output.status.success() {
        bail!(
            "TimeSeries42 remote upload failed for {node}:{destination} ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn copy_remote_checked(
    node: &str,
    source: &Path,
    destination: &str,
    executable: bool,
) -> Result<()> {
    let parent = Path::new(destination)
        .parent()
        .context("remote TimeSeries42 destination has no parent")?
        .to_string_lossy()
        .into_owned();
    let expected_sha256 = sha256_file(source)?;
    let expected_permissions = staging_file_mode(executable);
    if remote_path_exists(node, destination)? {
        let validation = scq_batch::validate_remote_immutable(
            node,
            destination,
            &expected_sha256,
            expected_permissions,
        );
        if validation.is_ok() {
            return Ok(());
        }
        let error = validation.unwrap_err();
        remove_remote_path(node, destination).with_context(|| {
            format!("remove invalid TimeSeries42 staging file after mismatch: {error:#}")
        })?;
    }
    scq_batch::ssh_output(
        node,
        &["mkdir", "-p", &parent],
        "create TimeSeries42 remote staging directory",
    )?;
    let upload_result = (|| -> Result<()> {
        upload_remote_file(node, source, destination)?;
        let mode = if executable { "0555" } else { "0444" };
        scq_batch::ssh_output(
            node,
            &["chmod", mode, "--", destination],
            "chmod TimeSeries42 staged file",
        )?;
        scq_batch::validate_remote_immutable(
            node,
            destination,
            &expected_sha256,
            expected_permissions,
        )
    })();
    if upload_result.is_err() {
        let _ = remove_remote_path(node, destination);
    }
    upload_result
}

fn stage_campaign(
    root: &Path,
    specs: &[DatasetSpec],
    release_binary_sha256: &str,
    corpus_eval_sha256: &str,
) -> Result<(StagingReceipt, String)> {
    let path = root.join("staging.v1.json");
    if path.is_file() {
        let bytes = fs::read(&path)?;
        let staging: StagingReceipt = serde_json::from_slice(&bytes)?;
        validate_staging_receipt(&staging, specs, release_binary_sha256, corpus_eval_sha256)?;
        if bytes != canonical_json(&staging)? {
            bail!("invalid TimeSeries42 staging receipt");
        }
        verify_existing_staging_assets(&staging)?;
        return Ok((staging, sha256_bytes(&bytes)));
    }
    let release_binary = std::env::current_exe()?;
    if sha256_file(&release_binary)? != release_binary_sha256 {
        bail!("release binary changed before TimeSeries42 staging");
    }
    let corpus_eval = Path::new(corpus_eval_str());
    if sha256_file(corpus_eval)? != corpus_eval_sha256 {
        bail!("corpus_eval changed before TimeSeries42 staging");
    }
    let marker_source = root.join("node-marker.v1");
    immutable_or_equal(&marker_source, NODE_MARKER_BYTES)?;
    let mut nodes = Vec::new();
    let controller = controller_hostname()?;
    for node in NODES {
        let local_node = is_local_scq_node(node, &controller);
        let hostname = if local_node {
            controller.clone()
        } else {
            String::from_utf8(
                scq_batch::ssh_output(node, &["hostname", "-s"], "TimeSeries42 node hostname")?
                    .stdout,
            )?
            .trim()
            .to_string()
        };
        if hostname.is_empty() {
            bail!("TimeSeries42 node {node} reported an empty hostname");
        }
        let bundle_root = remote_bundle_root(release_binary_sha256);
        let runner_path = format!("{bundle_root}/jope-dime-regressor");
        let corpus_eval_path = format!("{bundle_root}/corpus_eval");
        let marker_path = node_marker_path(&bundle_root, &hostname);
        if local_node {
            copy_local_checked(&release_binary, Path::new(&runner_path), true)?;
            copy_local_checked(corpus_eval, Path::new(&corpus_eval_path), true)?;
            copy_local_checked(&marker_source, Path::new(&marker_path), false)?;
        } else {
            copy_remote_checked(node, &release_binary, &runner_path, true)?;
            copy_remote_checked(node, corpus_eval, &corpus_eval_path, true)?;
            copy_remote_checked(node, &marker_source, &marker_path, false)?;
        }
        let mut datasets = Vec::with_capacity(specs.len());
        for spec in specs {
            let remote_root = remote_dataset_root(release_binary_sha256, &spec.dataset_hash);
            let train_path = format!("{remote_root}/train.csv");
            let test_path = format!("{remote_root}/test.csv");
            let meta_path = format!("{remote_root}/meta.json");
            if local_node {
                copy_local_checked(Path::new(&spec.train_path), Path::new(&train_path), false)?;
                copy_local_checked(Path::new(&spec.test_path), Path::new(&test_path), false)?;
                copy_local_checked(Path::new(&spec.meta_path), Path::new(&meta_path), false)?;
            } else {
                copy_remote_checked(node, Path::new(&spec.train_path), &train_path, false)?;
                copy_remote_checked(node, Path::new(&spec.test_path), &test_path, false)?;
                copy_remote_checked(node, Path::new(&spec.meta_path), &meta_path, false)?;
            }
            datasets.push(StagedDataset {
                dataset_hash: spec.dataset_hash.clone(),
                train_path,
                train_sha256: spec.train_sha256.clone(),
                test_path,
                test_sha256: spec.test_sha256.clone(),
                meta_path,
                meta_sha256: spec.meta_sha256.clone(),
            });
        }
        nodes.push(StagedNode {
            node: node.to_string(),
            hostname,
            bundle_root,
            runner_path,
            runner_sha256: release_binary_sha256.to_string(),
            corpus_eval_path,
            corpus_eval_sha256: corpus_eval_sha256.to_string(),
            datasets,
        });
    }
    let staging = StagingReceipt {
        schema: STAGING_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        release_binary_sha256: release_binary_sha256.to_string(),
        corpus_eval_sha256: corpus_eval_sha256.to_string(),
        nodes,
    };
    validate_staging_receipt(&staging, specs, release_binary_sha256, corpus_eval_sha256)?;
    verify_staging_assets(&staging)?;
    let sha = write_json(&path, &staging)?;
    Ok((staging, sha))
}

fn verify_existing_staging_assets(staging: &StagingReceipt) -> Result<()> {
    if env_truthy(SKIP_RESUME_STAGING_VERIFY_ENV) {
        eprintln!(
            "TimeSeries42 resume staging verification skipped by {SKIP_RESUME_STAGING_VERIFY_ENV}=1"
        );
        return Ok(());
    }
    if env_truthy(FULL_RESUME_STAGING_VERIFY_ENV) {
        return verify_staging_assets(staging);
    }
    verify_representative_staging_assets(staging)
}

fn validate_staging_receipt(
    staging: &StagingReceipt,
    specs: &[DatasetSpec],
    release_binary_sha256: &str,
    corpus_eval_sha256: &str,
) -> Result<()> {
    if staging.schema != STAGING_SCHEMA
        || !staging.test_adaptive
        || staging.benchmark_valid
        || staging.release_binary_sha256 != release_binary_sha256
        || staging.corpus_eval_sha256 != corpus_eval_sha256
        || staging.nodes.len() != NODES.len()
    {
        bail!("invalid TimeSeries42 staging receipt");
    }
    for (index, node) in staging.nodes.iter().enumerate() {
        if node.node != NODES[index]
            || node.hostname.is_empty()
            || node.runner_sha256 != release_binary_sha256
            || node.corpus_eval_sha256 != corpus_eval_sha256
            || node.datasets.len() != specs.len()
        {
            bail!("invalid TimeSeries42 staging node {}", node.node);
        }
        for spec in specs {
            let staged = node
                .datasets
                .iter()
                .find(|candidate| candidate.dataset_hash == spec.dataset_hash)
                .with_context(|| format!("staging omits {} on {}", spec.dataset_hash, node.node))?;
            if staged.train_sha256 != spec.train_sha256
                || staged.test_sha256 != spec.test_sha256
                || staged.meta_sha256 != spec.meta_sha256
            {
                bail!("invalid TimeSeries42 staged dataset {}", spec.dataset_hash);
            }
        }
    }
    Ok(())
}

fn validate_staged_path(
    node: &StagedNode,
    path: &str,
    sha256: &str,
    executable: bool,
) -> Result<()> {
    let mode = staging_file_mode(executable);
    if is_controller_node(&node.node)? {
        validate_local_checked(Path::new(path), sha256, mode)
    } else {
        scq_batch::validate_remote_immutable(&node.node, path, sha256, mode)
    }
    .with_context(|| format!("verify TimeSeries42 staged path {}:{path}", node.node))
}

fn verify_staging_assets(staging: &StagingReceipt) -> Result<()> {
    let marker_sha256 = sha256_bytes(NODE_MARKER_BYTES);
    for node in &staging.nodes {
        validate_staged_path(node, &node.runner_path, &node.runner_sha256, true)?;
        validate_staged_path(node, &node.corpus_eval_path, &node.corpus_eval_sha256, true)?;
        validate_staged_path(
            node,
            &node_marker_path(&node.bundle_root, &node.hostname),
            &marker_sha256,
            false,
        )?;
        for dataset in &node.datasets {
            validate_staged_path(node, &dataset.train_path, &dataset.train_sha256, false)?;
            validate_staged_path(node, &dataset.test_path, &dataset.test_sha256, false)?;
            validate_staged_path(node, &dataset.meta_path, &dataset.meta_sha256, false)?;
        }
    }
    Ok(())
}

fn verify_representative_staging_assets(staging: &StagingReceipt) -> Result<()> {
    let marker_sha256 = sha256_bytes(NODE_MARKER_BYTES);
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for node in &staging.nodes {
            let errors = Arc::clone(&errors);
            let marker_sha256 = marker_sha256.clone();
            scope.spawn(move || {
                let result = (|| -> Result<()> {
                    validate_staged_path(node, &node.runner_path, &node.runner_sha256, true)?;
                    validate_staged_path(
                        node,
                        &node.corpus_eval_path,
                        &node.corpus_eval_sha256,
                        true,
                    )?;
                    validate_staged_path(
                        node,
                        &node_marker_path(&node.bundle_root, &node.hostname),
                        &marker_sha256,
                        false,
                    )?;
                    let dataset = node
                        .datasets
                        .first()
                        .with_context(|| format!("staging node {} has no datasets", node.node))?;
                    validate_staged_path(node, &dataset.train_path, &dataset.train_sha256, false)?;
                    validate_staged_path(node, &dataset.test_path, &dataset.test_sha256, false)?;
                    validate_staged_path(node, &dataset.meta_path, &dataset.meta_sha256, false)?;
                    Ok(())
                })();
                if let Err(error) = result {
                    errors
                        .lock()
                        .unwrap()
                        .push(format!("{}: {error:#}", node.node));
                }
            });
        }
    });
    let errors = Arc::try_unwrap(errors)
        .expect("TimeSeries42 staging verification error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "TimeSeries42 representative staging verification failed: {}",
            errors.join("; ")
        );
    }
    eprintln!(
        "TimeSeries42 representative resume staging verification passed; set {FULL_RESUME_STAGING_VERIFY_ENV}=1 for full re-hash"
    );
    Ok(())
}

fn staged_dataset<'a>(
    staging: &'a StagingReceipt,
    node: &str,
    dataset: &str,
) -> Result<&'a StagedDataset> {
    staging
        .nodes
        .iter()
        .find(|record| record.node == node)
        .and_then(|record| {
            record
                .datasets
                .iter()
                .find(|candidate| candidate.dataset_hash == dataset)
        })
        .with_context(|| format!("staging lacks {node}/{dataset}"))
}

fn staged_node<'a>(staging: &'a StagingReceipt, node: &str) -> Result<&'a StagedNode> {
    staging
        .nodes
        .iter()
        .find(|record| record.node == node)
        .with_context(|| format!("staging lacks node {node}"))
}

fn label_authority(
    generation: usize,
    seed_sha256: &str,
    release_sha256: &str,
    campaign_sha256: &str,
) -> LabelAuthority {
    LabelAuthority {
        score_tier: "test_exact".to_string(),
        metric: "negative_mase".to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        selection_bias: Some("test-selected".to_string()),
        generation: format!("strict-wave-{generation:06}"),
        frozen_seed_sha256: seed_sha256.to_string(),
        model_checkpoint_sha256: release_sha256.to_string(),
        hall_of_fame_sha256: campaign_sha256.to_string(),
    }
}

fn corpus_manifest(
    spec: &DatasetSpec,
    members: &[Member],
    paths: (&str, &str, &str),
    authority: LabelAuthority,
    protocol_sha256: Option<&str>,
    execution_binding: Option<(&str, &str, &str)>,
) -> Result<CorpusShardManifest> {
    let (schema, reduction) = match spec.reduction()? {
        ReductionVersion::V1 => (CORPUS_SHARD_SCHEMA, None),
        version => (
            CORPUS_SHARD_SCHEMA_V2,
            Some(CorpusReductionSpec {
                version: version.as_str().to_string(),
            }),
        ),
    };
    let checkpoint_sha256 = protocol_sha256.map(|_| authority.model_checkpoint_sha256.clone());
    let (run_id, campaign_sha256, release_binary_sha256) = execution_binding
        .map(|(run_id, campaign_sha256, release_binary_sha256)| {
            (
                Some(run_id.to_string()),
                Some(campaign_sha256.to_string()),
                Some(release_binary_sha256.to_string()),
            )
        })
        .unwrap_or((None, None, None));
    Ok(CorpusShardManifest {
        schema: schema.to_string(),
        run_id,
        campaign_sha256,
        release_binary_sha256,
        comparator_lock_sha256: execution_binding.map(|_| spec.comparator_lock_sha256.clone()),
        task: "timeseries".to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        train_csv: paths.0.to_string(),
        test_csv: paths.1.to_string(),
        train_sha256: spec.train_sha256.clone(),
        test_sha256: spec.test_sha256.clone(),
        checkpoint_epoch: checkpoint_sha256
            .as_ref()
            .map(|digest| format!("checkpoint-{digest}")),
        checkpoint_sha256,
        catalog_sha256: protocol_sha256.map(|_| authority.frozen_seed_sha256.clone()),
        protocol_sha256: protocol_sha256.map(str::to_string),
        generated_at_ms: protocol_sha256.map(|_| 0),
        label_authority: authority,
        meta_json: paths.2.to_string(),
        meta_sha256: spec.meta_sha256.clone(),
        horizon: spec.horizon,
        seasonality: spec.seasonality,
        frequency: spec.frequency.clone(),
        target_is_final_column: true,
        known_future_covariates: spec.known_future_covariates.clone(),
        past_only_covariates: Vec::new(),
        reduction,
        candidates: members
            .iter()
            .map(|member| ShardCandidate {
                semantic_sha256: member.semantic_sha256.clone(),
                canonical_genome: member.canonical_genome.clone(),
            })
            .collect(),
    })
}

fn dataset_protocol_sha256(spec: &DatasetSpec) -> Result<String> {
    let path = Path::new(coverage_comparator_root_str())
        .join("timeseries")
        .join(&spec.dataset_hash)
        .join("comparator-lock.v2.json");
    let lock: Value = load_json(&path)?;
    let protocol = lock
        .get("protocol_sha256")
        .and_then(Value::as_str)
        .context("timeseries comparator lacks protocol_sha256")?;
    if lock
        .pointer("/protocol/protocol_sha256")
        .and_then(Value::as_str)
        != Some(protocol)
    {
        bail!("timeseries comparator protocol drift at {}", path.display());
    }
    Ok(protocol.to_string())
}

fn validate_with_corpus_eval(manifest: &Path) -> Result<()> {
    let output = Command::new(corpus_eval_str())
        .args(["validate-manifest", "--manifest"])
        .arg(manifest)
        .output()
        .with_context(|| format!("run corpus_eval validate-manifest {}", manifest.display()))?;
    if !output.status.success() {
        bail!(
            "corpus_eval rejected {}: {}",
            manifest.display(),
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    Ok(())
}

fn expected_job_id(
    run_id: &str,
    dataset: &str,
    generation: usize,
    shard: usize,
    node: &str,
    attempt: usize,
    manifest_sha256: &str,
) -> String {
    let binding =
        format!("{run_id}:{dataset}:{generation}:{shard}:{node}:{attempt}:{manifest_sha256}");
    format!("jope-ts42-{}", &sha256_bytes(binding.as_bytes())[..24])
}

fn scq_spec(
    node: &StagedNode,
    job_id: &str,
    remote_manifest: &str,
    remote_output: &str,
    project: &str,
) -> Result<String> {
    let args = [
        "RAYON_NUM_THREADS=1".to_string(),
        "RUST_NUM_THREADS=1".to_string(),
        "OMP_NUM_THREADS=1".to_string(),
        "OPENBLAS_NUM_THREADS=1".to_string(),
        "MKL_NUM_THREADS=1".to_string(),
        "VECLIB_MAXIMUM_THREADS=1".to_string(),
        "NUMEXPR_NUM_THREADS=1".to_string(),
        node.runner_path.clone(),
        "timeseries-test-adaptive-batch-worker".to_string(),
        "--manifest".to_string(),
        remote_manifest.to_string(),
        "--output-dir".to_string(),
        remote_output.to_string(),
        "--corpus-eval".to_string(),
        node.corpus_eval_path.clone(),
        "--timeout-seconds".to_string(),
        CANDIDATE_TIMEOUT_SECONDS.to_string(),
    ];
    let rendered_args = args
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join(", ");
    let node_probe_args = [
        "-f".to_string(),
        node_marker_path(&node.bundle_root, &node.hostname),
    ];
    let rendered_node_probe_args = node_probe_args
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join(", ");
    Ok(format!(
        "[job]\nid = {job_id:?}\nproject = {project:?}\ncommand = \"/usr/bin/env\"\nargs = [{rendered_args}]\nworkdir = {workdir:?}\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {SHARD_SIZE}\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = {probe_name:?}\ncommand = \"/usr/bin/test\"\nargs = [{rendered_node_probe_args}]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = 24\nmax_restarts = 0\n",
        workdir = node.bundle_root,
        probe_name = format!("TimeSeries42 staged node {}", node.node),
    ))
}

fn scq_json(args: &[&str]) -> Result<Value> {
    let output = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("--json").args(args);
            command
        },
        "TimeSeries42 SCQ JSON command",
    )?;
    serde_json::from_slice(&output.stdout).context("parse TimeSeries42 SCQ JSON")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScqWaitDecision {
    Terminal,
    CancelQueuedStall,
    Wait,
}

fn scq_job_has_last_start(job: &serde_json::Map<String, Value>) -> bool {
    job.get("last_start").is_some_and(|value| !value.is_null())
}

fn scq_wait_decision(
    state: &str,
    has_last_start: bool,
    queued_since: Option<Instant>,
    now: Instant,
    timeout: std::time::Duration,
) -> ScqWaitDecision {
    if matches!(state, "completed" | "failed" | "cancelled") {
        return ScqWaitDecision::Terminal;
    }
    if state == "queued"
        && !has_last_start
        && queued_since.is_some_and(|started| now.duration_since(started) >= timeout)
    {
        return ScqWaitDecision::CancelQueuedStall;
    }
    ScqWaitDecision::Wait
}

fn retryable_scq_stall_error(job_id: &str, reason: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "TimeSeries42 SCQ retryable stall ({SCQ_RETRYABLE_STALL_MARKER}): job {job_id} {reason}; launch-ready-capacity retry"
    )
}

fn cancel_scq_job(id: &str) -> Result<()> {
    scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("cancel").arg(id);
            command
        },
        "cancel TimeSeries42 stalled SCQ shard",
    )?;
    Ok(())
}

fn wait_for_job(id: &str) -> Result<Value> {
    let timeout = std::time::Duration::from_secs(SCQ_QUEUED_STALL_TIMEOUT_SECONDS);
    let mut queued_since = None;
    loop {
        let show = scq_json(&["show", id])?;
        let job = scq_batch::scq_job(&show, id)?;
        let state = job
            .get("state")
            .and_then(Value::as_str)
            .context("SCQ job has no state")?;
        let has_last_start = scq_job_has_last_start(job);
        if state == "queued" && !has_last_start {
            queued_since.get_or_insert_with(Instant::now);
        } else {
            queued_since = None;
        }
        match scq_wait_decision(state, has_last_start, queued_since, Instant::now(), timeout) {
            ScqWaitDecision::Terminal => return Ok(show),
            ScqWaitDecision::CancelQueuedStall => {
                cancel_scq_job(id)?;
                return Err(retryable_scq_stall_error(
                    id,
                    &format!(
                        "stayed queued without last_start for at least {SCQ_QUEUED_STALL_TIMEOUT_SECONDS}s"
                    ),
                ));
            }
            ScqWaitDecision::Wait => {}
        }
        thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn is_scq_capacity_error(error: &anyhow::Error) -> bool {
    let text = format!("{error:#}");
    text.contains(SCQ_RETRYABLE_STALL_MARKER)
        || text.contains("launch-ready-capacity")
        || text.contains("no eligible launch-ready node currently has the requested resources")
        || text.contains("Wait for capacity or reduce the resource request")
}

fn is_wrong_node_marker_capacity_error(error: &anyhow::Error, target_node: &str) -> bool {
    let text = format!("{error:#}");
    if !text.contains("dependencies unsatisfied on all eligible nodes")
        || !text.contains("smoke TimeSeries42 staged node")
    {
        return false;
    }
    let target_diagnostic =
        format!("{target_node}: smoke TimeSeries42 staged node {target_node}: smoke test exited");
    !text
        .lines()
        .map(str::trim_start)
        .any(|line| line.starts_with(&target_diagnostic))
}

fn is_scq_capacity_error_for_node(error: &anyhow::Error, target_node: &str) -> bool {
    is_scq_capacity_error(error) || is_wrong_node_marker_capacity_error(error, target_node)
}

fn submit_or_recover(job_id: &str, spec_path: &Path, target_node: &str) -> Result<Value> {
    let exists = scq_json(&["show", job_id])
        .ok()
        .and_then(|show| scq_batch::scq_job(&show, job_id).ok().map(|_| ()))
        .is_some();
    if !exists {
        let result = scq_batch::command_output(
            {
                let mut command = Command::new(scq_binary_str());
                command.arg("submit").arg(spec_path);
                command
            },
            "submit TimeSeries42 shard",
        );
        match result {
            Ok(_) => {}
            Err(error) if is_scq_capacity_error_for_node(&error, target_node) => return Err(error),
            Err(error) => return Err(error),
        }
    }
    let show = wait_for_job(job_id)?;
    let job = scq_batch::scq_job(&show, job_id)?;
    if job.get("state").and_then(Value::as_str) == Some("cancelled") {
        return Err(retryable_scq_stall_error(
            job_id,
            "was already cancelled and has no local shard receipt",
        ));
    }
    Ok(show)
}

fn scq_check(spec: &Path, target_node: &str) -> Result<()> {
    let result = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("check").arg(spec);
            command
        },
        "TimeSeries42 SCQ check",
    );
    match result {
        Ok(_) => Ok(()),
        Err(error) if is_scq_capacity_error_for_node(&error, target_node) => Err(error),
        Err(error) => Err(error),
    }
}

fn scq_job_exists(job_id: &str) -> bool {
    scq_json(&["show", job_id])
        .ok()
        .and_then(|show| scq_batch::scq_job(&show, job_id).ok().map(|_| ()))
        .is_some()
}

fn submit_batch(spec_paths: &[PathBuf]) -> Result<()> {
    if spec_paths.is_empty() {
        return Ok(());
    }
    scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("submit-batch");
            for spec_path in spec_paths {
                command.arg(spec_path);
            }
            command
        },
        "submit TimeSeries42 generation shard batch",
    )?;
    Ok(())
}

fn preflight_representative_scq_checks(
    root: &Path,
    staging: &StagingReceipt,
    run_id: &str,
) -> Result<()> {
    let project = if is_forecast3_gen20(run_id) {
        FORECAST3_PROJECT
    } else {
        LEGACY_PROJECT
    };
    let mut passed = 0usize;
    for node in &staging.nodes {
        let job_id = format!(
            "jope-ts42-preflight-{}-{}",
            node.node,
            &staging.release_binary_sha256[..12]
        );
        let spec = scq_spec(
            node,
            &job_id,
            &format!("{}/preflight/manifest.json", node.bundle_root),
            &format!("{}/preflight/{job_id}/output", node.bundle_root),
            project,
        )?;
        let spec_sha256 = sha256_bytes(spec.as_bytes());
        let spec_path = root.join("preflight").join("scq-checks").join(format!(
            "{}-{}.toml",
            node.node,
            &spec_sha256[..16]
        ));
        immutable_or_equal(&spec_path, spec.as_bytes())?;
        match scq_check(&spec_path, &node.node) {
            Ok(()) => passed += 1,
            Err(error)
                if node.node == "xbabe4" || is_scq_capacity_error_for_node(&error, &node.node) =>
            {
                eprintln!(
                    "TimeSeries42 representative SCQ check did not pass on {}; shard retries will route around it when needed: {error:#}",
                    node.node
                );
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("TimeSeries42 representative SCQ check on {}", node.node)
                });
            }
        }
    }
    if passed == 0 {
        eprintln!(
            "TimeSeries42 representative SCQ checks found no currently launch-ready staged node; per-shard retry will keep probing capacity"
        );
    }
    Ok(())
}

fn copy_remote_output(node: &str, remote: &str, local: &Path) -> Result<()> {
    if local.exists() {
        fs::remove_dir_all(local)?;
    }
    fs::create_dir_all(local)?;
    if is_controller_node(node)? {
        scq_batch::command_output(
            {
                let mut command = Command::new("cp");
                command.arg("-a").arg(format!("{remote}/.")).arg(local);
                command
            },
            "copy local TimeSeries42 SCQ output",
        )?;
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
            "pull TimeSeries42 SCQ output",
        )?;
    }
    Ok(())
}

fn read_receipts(path: &Path) -> Result<Vec<EvalReceipt>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parse corpus_eval receipt"))
        .collect()
}

fn prepare_shard(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    seed_sha256: &str,
    release_sha256: &str,
    staging: &StagingReceipt,
    generation: usize,
    shard: usize,
    node_index: usize,
    attempt: usize,
    members: &[Member],
) -> Result<PreparedShard> {
    let node_name = NODES[node_index];
    let node = staged_node(staging, node_name)?;
    let staged = staged_dataset(staging, node_name, &spec.dataset_hash)?;
    let authority = label_authority(generation, seed_sha256, release_sha256, campaign_sha256);
    let protocol_sha256 = if is_forecast3_gen20(run_id) {
        Some(dataset_protocol_sha256(spec)?)
    } else {
        None
    };
    let execution_binding =
        is_forecast3_gen20(run_id).then_some((run_id, campaign_sha256, release_sha256));
    let remote_manifest_value = corpus_manifest(
        spec,
        members,
        (&staged.train_path, &staged.test_path, &staged.meta_path),
        authority.clone(),
        protocol_sha256.as_deref(),
        execution_binding,
    )?;
    let local_manifest_value = corpus_manifest(
        spec,
        members,
        (&spec.train_path, &spec.test_path, &spec.meta_path),
        authority,
        protocol_sha256.as_deref(),
        execution_binding,
    )?;
    let local_manifest_bytes = canonical_json(&local_manifest_value)?;
    let local_manifest_sha256 = sha256_bytes(&local_manifest_bytes);
    let dataset_dir = dataset_root(root, &spec.dataset_hash);
    let local_manifest_path = generation_root(&dataset_dir, generation)
        .join("corpus-manifests")
        .join(format!(
            "shard-{shard:02}-attempt-{attempt}-{short}.v1.json",
            short = &local_manifest_sha256[..16]
        ));
    immutable_or_equal(&local_manifest_path, &local_manifest_bytes)?;
    validate_with_corpus_eval(&local_manifest_path)?;

    let remote_manifest_bytes = canonical_json(&remote_manifest_value)?;
    let remote_manifest_sha256 = sha256_bytes(&remote_manifest_bytes);
    let remote_manifest_local = root
        .join("jobs")
        .join("manifests")
        .join(format!("{remote_manifest_sha256}.json"));
    immutable_or_equal(&remote_manifest_local, &remote_manifest_bytes)?;
    let job_id = expected_job_id(
        run_id,
        &spec.dataset_hash,
        generation,
        shard,
        node_name,
        attempt,
        &remote_manifest_sha256,
    );
    let remote_manifest = format!(
        "{}/jobs/manifests/{remote_manifest_sha256}.json",
        node.bundle_root
    );
    let remote_output = format!("{}/jobs/{job_id}/output", node.bundle_root);
    let project = if is_forecast3_gen20(run_id) {
        FORECAST3_PROJECT
    } else {
        LEGACY_PROJECT
    };
    let spec_text = scq_spec(node, &job_id, &remote_manifest, &remote_output, project)?;
    let spec_path = root
        .join("jobs")
        .join("specs")
        .join(format!("{job_id}.toml"));
    immutable_or_equal(&spec_path, spec_text.as_bytes())?;

    let receipt_path = root
        .join("jobs")
        .join("receipts")
        .join(format!("{job_id}.v1.json"));
    if !receipt_path.is_file() {
        if is_controller_node(node_name)? {
            copy_local_checked(&remote_manifest_local, Path::new(&remote_manifest), false)?;
            fs::create_dir_all(&remote_output)?;
        } else {
            copy_remote_checked(node_name, &remote_manifest_local, &remote_manifest, false)?;
            scq_batch::ssh_output(
                node_name,
                &["mkdir", "-p", &remote_output],
                "create TimeSeries42 remote output",
            )?;
        }
    }

    Ok(PreparedShard {
        generation,
        shard,
        node: node_name.to_string(),
        attempt,
        job_id,
        remote_manifest_sha256,
        local_manifest_path,
        spec_path,
        spec_text,
        remote_output,
        receipt_path,
    })
}

fn finalize_prepared_shard(
    root: &Path,
    spec: &DatasetSpec,
    prepared: &PreparedShard,
) -> Result<ShardRecord> {
    if !prepared.receipt_path.is_file() {
        let show = wait_for_job(&prepared.job_id)?;
        let show_bytes = canonical_json(&show)?;
        let job = scq_batch::scq_job(&show, &prepared.job_id)?;
        let state = job
            .get("state")
            .and_then(Value::as_str)
            .context("SCQ job has no state")?
            .to_string();
        let exit_code = job
            .get("exit_code")
            .and_then(Value::as_i64)
            .context("SCQ job has no exit_code")?;
        if state != "completed" || exit_code != 0 {
            bail!(
                "TimeSeries42 shard {} failed: state={state}, exit={exit_code}",
                prepared.job_id
            );
        }
        let temp = std::env::temp_dir().join(format!(
            "jope-ts42-{}-{}",
            std::process::id(),
            prepared.job_id
        ));
        copy_remote_output(&prepared.node, &prepared.remote_output, &temp)?;
        let raw = root.join("jobs").join("raw").join(&prepared.job_id);
        if raw.exists() {
            fs::remove_dir_all(&raw)?;
        }
        fs::create_dir_all(raw.parent().context("raw output has no parent")?)?;
        fs::rename(&temp, &raw)?;
        let worker_bytes = fs::read(raw.join("batch-worker-receipt.v1.json"))?;
        let worker: WorkerReceipt = serde_json::from_slice(&worker_bytes)?;
        let receipt_file = raw.join(format!("{}.receipts.jsonl", spec.dataset_hash));
        let receipt_bytes = fs::read(&receipt_file)?;
        if worker.schema != WORKER_SCHEMA
            || !worker.test_adaptive
            || worker.benchmark_valid
            || worker.dataset_hash != spec.dataset_hash
            || worker.manifest_sha256 != prepared.remote_manifest_sha256
            || worker.receipt_sha256 != sha256_bytes(&receipt_bytes)
        {
            bail!(
                "TimeSeries42 worker receipt mismatch for {}",
                prepared.job_id
            );
        }
        let job_receipt = JobReceipt {
            schema: JOB_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            job_id: prepared.job_id.clone(),
            node: prepared.node.clone(),
            attempt: prepared.attempt,
            state,
            exit_code,
            corpus_manifest_sha256: prepared.remote_manifest_sha256.clone(),
            scq_spec_sha256: sha256_bytes(prepared.spec_text.as_bytes()),
            scq_show_sha256: sha256_bytes(&show_bytes),
            worker_receipt_sha256: sha256_bytes(&worker_bytes),
            receipt_sha256: sha256_bytes(&receipt_bytes),
            completed_at_unix_ms: unix_millis(),
        };
        write_json(&prepared.receipt_path, &job_receipt)?;
    }
    let raw_receipt = root
        .join("jobs")
        .join("raw")
        .join(&prepared.job_id)
        .join(format!("{}.receipts.jsonl", spec.dataset_hash));
    let receipt_bytes = fs::read(&raw_receipt)?;
    let copied_receipt =
        generation_root(&dataset_root(root, &spec.dataset_hash), prepared.generation)
            .join("receipts")
            .join(format!(
                "shard-{shard:02}-attempt-{attempt}-{short}.receipts.jsonl",
                shard = prepared.shard,
                attempt = prepared.attempt,
                short = &prepared.remote_manifest_sha256[..16]
            ));
    immutable_or_equal(&copied_receipt, &receipt_bytes)?;
    let receipts = read_receipts(&raw_receipt)?;
    Ok(ShardRecord {
        generation: prepared.generation,
        shard: prepared.shard,
        node: prepared.node.clone(),
        attempt: prepared.attempt,
        job_id: prepared.job_id.clone(),
        corpus_manifest_file: prepared.local_manifest_path.to_string_lossy().into_owned(),
        corpus_manifest_sha256: sha256_file(&prepared.local_manifest_path)?,
        scq_spec_file: prepared.spec_path.to_string_lossy().into_owned(),
        scq_spec_sha256: sha256_bytes(prepared.spec_text.as_bytes()),
        receipt_file: copied_receipt.to_string_lossy().into_owned(),
        receipt_sha256: sha256_bytes(&receipt_bytes),
        valid_candidates: receipts.iter().filter(|receipt| receipt.valid).count(),
        total_candidates: receipts.len(),
    })
}

fn execute_transactional_wave(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    seed_sha256: &str,
    release_sha256: &str,
    staging: &StagingReceipt,
    generation: usize,
    chunks: &[Vec<Member>],
) -> Result<Vec<ShardRecord>> {
    loop {
        let submitted_at = unix_millis();
        let mut prepared = Vec::with_capacity(SHARDS);
        for (shard, members) in chunks.iter().enumerate() {
            prepared.push(prepare_shard(
                root,
                run_id,
                campaign_sha256,
                spec,
                seed_sha256,
                release_sha256,
                staging,
                generation,
                shard,
                shard % NODES.len(),
                0,
                members,
            )?);
        }
        let mut capacity_wait = None;
        for shard in &prepared {
            if shard.receipt_path.is_file() {
                continue;
            }
            match scq_check(&shard.spec_path, &shard.node) {
                Ok(()) => {}
                Err(error) if is_scq_capacity_error_for_node(&error, &shard.node) => {
                    capacity_wait = Some(format!("{error:#}"));
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        if let Some(error) = capacity_wait {
            eprintln!(
                "TimeSeries42 generation {generation} dataset {} waiting for SCQ capacity before transactional wave: {error}",
                spec.dataset_hash
            );
            thread::sleep(std::time::Duration::from_secs(SCQ_CAPACITY_RETRY_SECONDS));
            continue;
        }
        let missing_specs = prepared
            .iter()
            .filter(|shard| !shard.receipt_path.is_file() && !scq_job_exists(&shard.job_id))
            .map(|shard| shard.spec_path.clone())
            .collect::<Vec<_>>();
        match submit_batch(&missing_specs) {
            Ok(()) => {}
            Err(error) if is_scq_capacity_error(&error) => {
                eprintln!(
                    "TimeSeries42 generation {generation} dataset {} waiting for SCQ capacity before submit-batch: {error:#}",
                    spec.dataset_hash
                );
                thread::sleep(std::time::Duration::from_secs(SCQ_CAPACITY_RETRY_SECONDS));
                continue;
            }
            Err(error) => return Err(error),
        }

        let mut records = Vec::with_capacity(SHARDS);
        for shard in &prepared {
            records.push(finalize_prepared_shard(root, spec, shard)?);
        }
        records.sort_by_key(|record| record.shard);
        let wave = WaveReceipt {
            schema: WAVE_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            campaign_sha256: campaign_sha256.to_string(),
            dataset_hash: spec.dataset_hash.clone(),
            generation,
            nodes: prepared.iter().map(|shard| shard.node.clone()).collect(),
            job_ids: prepared.iter().map(|shard| shard.job_id.clone()).collect(),
            corpus_manifest_sha256: prepared
                .iter()
                .map(|shard| shard.remote_manifest_sha256.clone())
                .collect(),
            scq_spec_sha256: prepared
                .iter()
                .map(|shard| sha256_bytes(shard.spec_text.as_bytes()))
                .collect(),
            terminal_rows: records.iter().map(|record| record.total_candidates).sum(),
            submitted_at_unix_ms: submitted_at,
            completed_at_unix_ms: unix_millis(),
        };
        write_json(
            &generation_root(&dataset_root(root, &spec.dataset_hash), generation)
                .join("wave-receipt.v1.json"),
            &wave,
        )?;
        return Ok(records);
    }
}

fn execute_shard(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    seed_sha256: &str,
    release_sha256: &str,
    staging: &StagingReceipt,
    generation: usize,
    shard: usize,
    node_index: usize,
    attempt: usize,
    members: &[Member],
) -> Result<ShardRecord> {
    let node_name = NODES[node_index];
    let node = staged_node(staging, node_name)?;
    let staged = staged_dataset(staging, node_name, &spec.dataset_hash)?;
    let authority = label_authority(generation, seed_sha256, release_sha256, campaign_sha256);
    let protocol_sha256 = if is_forecast3_gen20(run_id) {
        Some(dataset_protocol_sha256(spec)?)
    } else {
        None
    };
    let execution_binding =
        is_forecast3_gen20(run_id).then_some((run_id, campaign_sha256, release_sha256));
    let remote_manifest_value = corpus_manifest(
        spec,
        members,
        (&staged.train_path, &staged.test_path, &staged.meta_path),
        authority.clone(),
        protocol_sha256.as_deref(),
        execution_binding,
    )?;
    let local_manifest_value = corpus_manifest(
        spec,
        members,
        (&spec.train_path, &spec.test_path, &spec.meta_path),
        authority,
        protocol_sha256.as_deref(),
        execution_binding,
    )?;
    let local_manifest_bytes = canonical_json(&local_manifest_value)?;
    let local_manifest_sha256 = sha256_bytes(&local_manifest_bytes);
    let dataset_dir = dataset_root(root, &spec.dataset_hash);
    let local_manifest_path = generation_root(&dataset_dir, generation)
        .join("corpus-manifests")
        .join(format!(
            "shard-{shard:02}-attempt-{attempt}-{short}.v1.json",
            short = &local_manifest_sha256[..16]
        ));
    immutable_or_equal(&local_manifest_path, &local_manifest_bytes)?;
    validate_with_corpus_eval(&local_manifest_path)?;

    let remote_manifest_bytes = canonical_json(&remote_manifest_value)?;
    let remote_manifest_sha256 = sha256_bytes(&remote_manifest_bytes);
    let remote_manifest_local = root
        .join("jobs")
        .join("manifests")
        .join(format!("{remote_manifest_sha256}.json"));
    immutable_or_equal(&remote_manifest_local, &remote_manifest_bytes)?;
    let job_id = expected_job_id(
        run_id,
        &spec.dataset_hash,
        generation,
        shard,
        node_name,
        attempt,
        &remote_manifest_sha256,
    );
    let remote_manifest = format!(
        "{}/jobs/manifests/{remote_manifest_sha256}.json",
        node.bundle_root
    );
    let remote_output = format!("{}/jobs/{job_id}/output", node.bundle_root);
    let project = if is_forecast3_gen20(run_id) {
        FORECAST3_PROJECT
    } else {
        LEGACY_PROJECT
    };
    let spec_text = scq_spec(node, &job_id, &remote_manifest, &remote_output, project)?;
    let spec_path = root
        .join("jobs")
        .join("specs")
        .join(format!("{job_id}.toml"));
    immutable_or_equal(&spec_path, spec_text.as_bytes())?;

    let receipt_path = root
        .join("jobs")
        .join("receipts")
        .join(format!("{job_id}.v1.json"));
    if !receipt_path.is_file() {
        if is_controller_node(node_name)? {
            copy_local_checked(&remote_manifest_local, Path::new(&remote_manifest), false)?;
            fs::create_dir_all(&remote_output)?;
        } else {
            copy_remote_checked(node_name, &remote_manifest_local, &remote_manifest, false)?;
            scq_batch::ssh_output(
                node_name,
                &["mkdir", "-p", &remote_output],
                "create TimeSeries42 remote output",
            )?;
        }
        scq_check(&spec_path, node_name)?;
        let show = submit_or_recover(&job_id, &spec_path, node_name)?;
        let show_bytes = canonical_json(&show)?;
        let job = scq_batch::scq_job(&show, &job_id)?;
        let state = job
            .get("state")
            .and_then(Value::as_str)
            .context("SCQ job has no state")?
            .to_string();
        let exit_code = job
            .get("exit_code")
            .and_then(Value::as_i64)
            .context("SCQ job has no exit_code")?;
        if state != "completed" || exit_code != 0 {
            bail!("TimeSeries42 shard {job_id} failed: state={state}, exit={exit_code}");
        }
        let temp = std::env::temp_dir().join(format!("jope-ts42-{}-{job_id}", std::process::id()));
        copy_remote_output(node_name, &remote_output, &temp)?;
        let raw = root.join("jobs").join("raw").join(&job_id);
        if raw.exists() {
            fs::remove_dir_all(&raw)?;
        }
        fs::create_dir_all(raw.parent().context("raw output has no parent")?)?;
        fs::rename(&temp, &raw)?;
        let worker_bytes = fs::read(raw.join("batch-worker-receipt.v1.json"))?;
        let worker: WorkerReceipt = serde_json::from_slice(&worker_bytes)?;
        let receipt_file = raw.join(format!("{}.receipts.jsonl", spec.dataset_hash));
        let receipt_bytes = fs::read(&receipt_file)?;
        if worker.schema != WORKER_SCHEMA
            || !worker.test_adaptive
            || worker.benchmark_valid
            || worker.dataset_hash != spec.dataset_hash
            || worker.manifest_sha256 != remote_manifest_sha256
            || worker.receipt_sha256 != sha256_bytes(&receipt_bytes)
        {
            bail!("TimeSeries42 worker receipt mismatch for {job_id}");
        }
        let job_receipt = JobReceipt {
            schema: JOB_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            job_id: job_id.clone(),
            node: node_name.to_string(),
            attempt,
            state,
            exit_code,
            corpus_manifest_sha256: remote_manifest_sha256.clone(),
            scq_spec_sha256: sha256_bytes(spec_text.as_bytes()),
            scq_show_sha256: sha256_bytes(&show_bytes),
            worker_receipt_sha256: sha256_bytes(&worker_bytes),
            receipt_sha256: sha256_bytes(&receipt_bytes),
            completed_at_unix_ms: unix_millis(),
        };
        write_json(&receipt_path, &job_receipt)?;
    }
    let raw_receipt = root
        .join("jobs")
        .join("raw")
        .join(&job_id)
        .join(format!("{}.receipts.jsonl", spec.dataset_hash));
    let receipt_bytes = fs::read(&raw_receipt)?;
    let copied_receipt = generation_root(&dataset_dir, generation)
        .join("receipts")
        .join(format!(
            "shard-{shard:02}-attempt-{attempt}-{short}.receipts.jsonl",
            short = &remote_manifest_sha256[..16]
        ));
    immutable_or_equal(&copied_receipt, &receipt_bytes)?;
    let receipts = read_receipts(&raw_receipt)?;
    Ok(ShardRecord {
        generation,
        shard,
        node: node_name.to_string(),
        attempt,
        job_id,
        corpus_manifest_file: local_manifest_path.to_string_lossy().into_owned(),
        corpus_manifest_sha256: sha256_file(&local_manifest_path)?,
        scq_spec_file: spec_path.to_string_lossy().into_owned(),
        scq_spec_sha256: sha256_bytes(spec_text.as_bytes()),
        receipt_file: copied_receipt.to_string_lossy().into_owned(),
        receipt_sha256: sha256_bytes(&receipt_bytes),
        valid_candidates: receipts.iter().filter(|receipt| receipt.valid).count(),
        total_candidates: receipts.len(),
    })
}

fn execute_shard_with_retry(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    seed_sha256: &str,
    release_sha256: &str,
    staging: &StagingReceipt,
    generation: usize,
    shard: usize,
    members: &[Member],
) -> Result<ShardRecord> {
    let primary = shard % NODES.len();
    let mut hard_errors = Vec::new();
    let mut retry_cycle = 0usize;
    loop {
        let mut saw_capacity_wait = false;
        for offset in 0..NODES.len() {
            let node_index = (primary + offset) % NODES.len();
            let attempt = retry_cycle * NODES.len() + offset;
            match execute_shard(
                root,
                run_id,
                campaign_sha256,
                spec,
                seed_sha256,
                release_sha256,
                staging,
                generation,
                shard,
                node_index,
                attempt,
                members,
            ) {
                Ok(record) => return Ok(record),
                Err(error) if is_scq_capacity_error_for_node(&error, NODES[node_index]) => {
                    eprintln!(
                        "TimeSeries42 shard {} generation {} dataset {} node {} attempt {} is waiting for SCQ capacity; trying next staged node: {error:#}",
                        shard, generation, spec.dataset_hash, NODES[node_index], attempt
                    );
                    saw_capacity_wait = true;
                }
                Err(error) => {
                    eprintln!(
                        "TimeSeries42 shard {} generation {} dataset {} node {} attempt {} failed: {error:#}",
                        shard, generation, spec.dataset_hash, NODES[node_index], attempt
                    );
                    hard_errors.push(format!(
                        "{} attempt {}: {error:#}",
                        NODES[node_index], attempt
                    ));
                }
            }
        }
        if saw_capacity_wait {
            eprintln!(
                "TimeSeries42 shard {} generation {} dataset {} found no launch-ready staged node; waiting {SCQ_CAPACITY_RETRY_SECONDS}s before retrying node pool",
                shard, generation, spec.dataset_hash
            );
            retry_cycle += 1;
            thread::sleep(std::time::Duration::from_secs(SCQ_CAPACITY_RETRY_SECONDS));
            continue;
        }
        bail!(
            "TimeSeries42 shard {} generation {} dataset {} failed on every staged node: {}",
            shard,
            generation,
            spec.dataset_hash,
            hard_errors.join(" | ")
        )
    }
}

fn receipt_mase(receipt: &EvalReceipt) -> Option<TimeSeriesScore> {
    if receipt.task != "timeseries" || !receipt.valid {
        return None;
    }
    let mase = if receipt.secondary_metric == "mase" {
        receipt.secondary_score?
    } else if receipt.metric == "negative_mase" {
        -receipt.score?
    } else {
        return None;
    };
    if !mase.is_finite() {
        return None;
    }
    Some(TimeSeriesScore {
        mase,
        rmse: receipt.rmse.unwrap_or(f64::INFINITY),
    })
}

fn slot_from_receipt(
    generation: usize,
    slot: usize,
    member: &Member,
    receipt: &EvalReceipt,
    receipt_file: &str,
    receipt_sha256: &str,
) -> LogicalSlot {
    let score = receipt_mase(receipt);
    LogicalSlot {
        generation,
        slot,
        member: member.clone(),
        valid: score.is_some(),
        score,
        failure: receipt.failure.clone(),
        receipt_file: receipt_file.to_string(),
        receipt_sha256: receipt_sha256.to_string(),
        prediction_sha256: receipt.predictions_sha256.clone(),
        runtime_seconds: receipt.runtime_seconds,
    }
}

fn rank_slots(left: &LogicalSlot, right: &LogicalSlot) -> Ordering {
    let left_score = left.score.as_ref();
    let right_score = right.score.as_ref();
    match (left_score, right_score) {
        (Some(left_score), Some(right_score)) => left_score
            .mase
            .total_cmp(&right_score.mase)
            .then_with(|| left_score.rmse.total_cmp(&right_score.rmse))
            .then_with(|| left.member.complexity.cmp(&right.member.complexity))
            .then_with(|| {
                left.member
                    .semantic_sha256
                    .cmp(&right.member.semantic_sha256)
            }),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left
            .member
            .semantic_sha256
            .cmp(&right.member.semantic_sha256),
    }
}

fn evaluate_generation(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    seed_sha256: &str,
    release_sha256: &str,
    staging: &StagingReceipt,
    generation: usize,
    members: &[Member],
) -> Result<Vec<LogicalSlot>> {
    if members.len() != POPULATION {
        bail!("TimeSeries42 generation requires exactly {POPULATION} members");
    }
    let chunks = members
        .chunks(SHARD_SIZE)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    let records = execute_transactional_wave(
        root,
        run_id,
        campaign_sha256,
        spec,
        seed_sha256,
        release_sha256,
        staging,
        generation,
        &chunks,
    )?;

    let expected_protocol = if is_forecast3_gen20(run_id) {
        Some(dataset_protocol_sha256(spec)?)
    } else {
        None
    };
    let mut slots = Vec::with_capacity(POPULATION);
    for record in &records {
        let receipts = read_receipts(Path::new(&record.receipt_file))?;
        if receipts.len() != SHARD_SIZE {
            bail!("TimeSeries42 shard produced {} receipts", receipts.len());
        }
        let start = record.shard * SHARD_SIZE;
        for (offset, receipt) in receipts.iter().enumerate() {
            let slot = start + offset;
            if receipt.dataset_hash != spec.dataset_hash
                || receipt.semantic_sha256 != members[slot].semantic_sha256
                || receipt.canonical_genome != members[slot].canonical_genome
                || expected_protocol
                    .as_ref()
                    .is_some_and(|expected| receipt.protocol_sha256.as_deref() != Some(expected))
            {
                bail!("TimeSeries42 receipt/member binding mismatch");
            }
            slots.push(slot_from_receipt(
                generation,
                slot,
                &members[slot],
                receipt,
                &record.receipt_file,
                &record.receipt_sha256,
            ));
        }
    }
    let mut ranked = slots.clone();
    ranked.sort_by(rank_slots);
    let best = ranked.iter().find(|slot| slot.valid);
    let manifest = GenerationManifest {
        schema: GENERATION_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        campaign_sha256: campaign_sha256.to_string(),
        release_binary_sha256: release_sha256.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        generation,
        members: members.to_vec(),
        shards: records,
        best_semantic_sha256: best.map(|slot| slot.member.semantic_sha256.clone()),
        best_mase: best.and_then(|slot| slot.score.as_ref().map(|score| score.mase)),
        valid_candidates: slots.iter().filter(|slot| slot.valid).count(),
        completed_at_unix_ms: unix_millis(),
    };
    write_current_json(
        &generation_root(&dataset_root(root, &spec.dataset_hash), generation)
            .join("generation.v1.json"),
        &manifest,
        "generation",
    )?;
    Ok(slots)
}

fn load_generation_slots(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    generation: usize,
) -> Result<Vec<LogicalSlot>> {
    let path = generation_root(&dataset_root(root, &spec.dataset_hash), generation)
        .join("generation.v1.json");
    let manifest: GenerationManifest = load_json(&path)?;
    if manifest.schema != GENERATION_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != run_id
        || manifest.campaign_sha256 != campaign_sha256
        || manifest.dataset_hash != spec.dataset_hash
        || manifest.generation != generation
        || manifest.members.len() != POPULATION
        || manifest.shards.len() != SHARDS
    {
        bail!(
            "TimeSeries42 generation artifact is stale or invalid for {} generation {}",
            spec.dataset_hash,
            generation
        );
    }
    let mut records = manifest.shards.clone();
    let expected_protocol = if is_forecast3_gen20(run_id) {
        Some(dataset_protocol_sha256(spec)?)
    } else {
        None
    };
    records.sort_by_key(|record| record.shard);
    let mut seen_shards = BTreeSet::new();
    let mut slots = Vec::with_capacity(POPULATION);
    for record in &records {
        if record.generation != generation
            || record.shard >= SHARDS
            || !seen_shards.insert(record.shard)
            || record.total_candidates != SHARD_SIZE
        {
            bail!(
                "TimeSeries42 shard accounting mismatch for {} generation {}",
                spec.dataset_hash,
                generation
            );
        }
        let receipt_path = Path::new(&record.receipt_file);
        if sha256_file(receipt_path)? != record.receipt_sha256 {
            bail!(
                "TimeSeries42 shard receipt hash mismatch for {} generation {} shard {}",
                spec.dataset_hash,
                generation,
                record.shard
            );
        }
        let receipts = read_receipts(receipt_path)?;
        if receipts.len() != SHARD_SIZE {
            bail!("TimeSeries42 shard produced {} receipts", receipts.len());
        }
        let start = record.shard * SHARD_SIZE;
        for (offset, receipt) in receipts.iter().enumerate() {
            let slot = start + offset;
            if receipt.dataset_hash != spec.dataset_hash
                || receipt.semantic_sha256 != manifest.members[slot].semantic_sha256
                || receipt.canonical_genome != manifest.members[slot].canonical_genome
                || expected_protocol
                    .as_ref()
                    .is_some_and(|expected| receipt.protocol_sha256.as_deref() != Some(expected))
            {
                bail!("TimeSeries42 base receipt/member binding mismatch");
            }
            slots.push(slot_from_receipt(
                generation,
                slot,
                &manifest.members[slot],
                receipt,
                &record.receipt_file,
                &record.receipt_sha256,
            ));
        }
    }
    if slots.len() != POPULATION {
        bail!(
            "TimeSeries42 loaded {} slots for {} generation {}",
            slots.len(),
            spec.dataset_hash,
            generation
        );
    }
    Ok(slots)
}

fn generation_best_mase(slots: &[LogicalSlot]) -> Option<f64> {
    let mut ranked = slots.to_vec();
    ranked.sort_by(rank_slots);
    ranked
        .iter()
        .find(|slot| slot.valid)
        .and_then(|slot| slot.score.as_ref().map(|score| score.mase))
}

fn strict_generation_floor(base_result: &DatasetResult) -> usize {
    if base_result.strict_winner {
        1
    } else {
        GEN30_START_GENERATION
    }
}

fn slot_is_strict_winner(
    slot: &LogicalSlot,
    spec: &DatasetSpec,
    generation_zero_best: Option<f64>,
    min_generation: usize,
) -> bool {
    let Some(score) = &slot.score else {
        return false;
    };
    slot.generation >= min_generation
        && slot.member.evolved_descendant
        && score.mase < spec.official_leader.mase
        && generation_zero_best.is_some_and(|g0| score.mase < g0)
}

fn slot_beats_external(slot: &LogicalSlot, spec: &DatasetSpec) -> bool {
    slot.valid
        && slot
            .score
            .as_ref()
            .is_some_and(|score| score.mase.is_finite() && score.mase < spec.official_leader.mase)
}

fn slot_is_base_campaign_winner(
    slot: &LogicalSlot,
    spec: &DatasetSpec,
    generation_zero_best: Option<f64>,
    allow_generation_zero_win: bool,
) -> bool {
    if allow_generation_zero_win {
        slot_beats_external(slot, spec)
    } else {
        slot_is_strict_winner(slot, spec, generation_zero_best, 1)
    }
}

fn has_strict_winner_from_generation(
    slots: &[LogicalSlot],
    spec: &DatasetSpec,
    generation_zero_best: Option<f64>,
    min_generation: usize,
) -> bool {
    slots
        .iter()
        .any(|slot| slot_is_strict_winner(slot, spec, generation_zero_best, min_generation))
}

fn continuation_should_stop(
    base_result: &DatasetResult,
    all_slots: &[LogicalSlot],
    spec: &DatasetSpec,
    generation_zero_best: Option<f64>,
    generation: usize,
) -> bool {
    if generation + 1 == GEN30_GENERATIONS {
        return true;
    }
    if base_result.strict_winner {
        return false;
    }
    has_strict_winner_from_generation(
        all_slots,
        spec,
        generation_zero_best,
        strict_generation_floor(base_result),
    )
}

fn offspring_seed(
    dataset_hash: &str,
    generation: usize,
    slot: usize,
    retry: usize,
    lineage_salt: &str,
) -> u64 {
    let text = if lineage_salt.is_empty() {
        format!("{RNG_SEED}:{dataset_hash}:{generation}:{slot}:{retry}")
    } else {
        format!("{RNG_SEED}:{lineage_salt}:{dataset_hash}:{generation}:{slot}:{retry}")
    };
    let digest = Sha256::digest(text.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

fn tournament_winner(rng: &mut Pcg32) -> usize {
    (0..TOURNAMENT)
        .map(|_| rng.next_u32() as usize % PARENT_POOL)
        .min()
        .expect("tournament is nonempty")
}

fn next_population(
    spec: &DatasetSpec,
    ranked: &[LogicalSlot],
    generation: usize,
    lineage_salt: &str,
) -> Result<Vec<Member>> {
    if ranked.len() < PARENT_POOL || ranked[..PARENT_POOL].iter().any(|slot| !slot.valid) {
        bail!("TimeSeries42 parent pool has fewer than {PARENT_POOL} valid candidates");
    }
    let mut next = Vec::with_capacity(POPULATION);
    let mut seen = BTreeSet::new();
    for elite in ranked.iter().take(ELITES) {
        seen.insert(elite.member.semantic_sha256.clone());
        let mut member = elite.member.clone();
        member.evolved_descendant = true;
        member.ancestry = AncestryRecord {
            operation: "elite".to_string(),
            parents: vec![elite.member.semantic_sha256.clone()],
            retry_count: 0,
        };
        next.push(member);
    }
    for slot in ELITES..POPULATION {
        let mut produced = None;
        for retry in 0..1_000 {
            let mut rng = Pcg32::new(
                offspring_seed(&spec.dataset_hash, generation, slot, retry, lineage_salt),
                0x7473_3432_6e65_7874,
            );
            let first = tournament_winner(&mut rng);
            let first_parent = &ranked[first].member;
            let maybe = if rng.next_f64() < 0.25 {
                let second = tournament_winner(&mut rng);
                let second_parent = &ranked[second].member;
                crossover_pipelines(
                    &first_parent.canonical_genome,
                    &second_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                )
                .map(|genome| {
                    (
                        genome,
                        "crossover",
                        vec![
                            first_parent.semantic_sha256.clone(),
                            second_parent.semantic_sha256.clone(),
                        ],
                    )
                })
            } else {
                mutate_pipeline(
                    &first_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    feat_math::full_rust::MAX_TRANSFORMS,
                )
                .map(|genome| {
                    (
                        genome,
                        "mutation",
                        vec![first_parent.semantic_sha256.clone()],
                    )
                })
            };
            let Ok((candidate, operation, parents)) = maybe else {
                continue;
            };
            let canonical = match canonicalize_for_timeseries(&candidate.canonical, spec) {
                Ok(canonical) => canonical,
                Err(_) => continue,
            };
            if !seen.insert(canonical.semantic_sha256.clone()) {
                continue;
            }
            produced = Some(Member {
                canonical_genome: canonical.canonical,
                semantic_sha256: canonical.semantic_sha256,
                complexity: canonical.complexity,
                rust_source_len: canonical.rust_source_len,
                evolved_descendant: true,
                ancestry: AncestryRecord {
                    operation: operation.to_string(),
                    parents,
                    retry_count: retry,
                },
            });
            break;
        }
        next.push(produced.with_context(|| {
            format!(
                "could not generate TimeSeries42 child for {} generation {generation} slot {slot}",
                spec.dataset_hash
            )
        })?);
    }
    Ok(next)
}

fn seed_family(bank: &SeedBank, semantic: &str) -> Option<String> {
    bank.seeds
        .iter()
        .find(|seed| seed.semantic_sha256 == semantic)
        .map(|seed| seed.source_family.clone())
}

fn final_result_path(root: &Path, dataset_hash: &str) -> PathBuf {
    dataset_root(root, dataset_hash).join("final-result.v1.json")
}

fn archive_stale_final_result(path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let bytes = fs::read(path)?;
    let sha256 = sha256_bytes(&bytes);
    let archive = path.with_file_name(format!("final-result.stale-{}.v1.json", &sha256[..16]));
    immutable_or_equal(&archive, &bytes)?;
    fs::remove_file(path).with_context(|| format!("remove stale {}", path.display()))?;
    Ok(())
}

fn result_matches_campaign(
    result: &DatasetResult,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    release_sha256: &str,
) -> bool {
    result.schema == RESULT_SCHEMA
        && result.test_adaptive
        && !result.benchmark_valid
        && result.run_id == run_id
        && result.campaign_sha256 == campaign_sha256
        && result.release_binary_sha256 == release_sha256
        && result.dataset_hash == spec.dataset_hash
        && result.official_leader_model == spec.official_leader.model
        && result
            .official_leader_mase
            .total_cmp(&spec.official_leader.mase)
            == Ordering::Equal
        && matches!(result.status.as_str(), "win" | "loss" | "tie" | "exhausted")
}

fn load_reusable_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    release_sha256: &str,
) -> Result<Option<DatasetResult>> {
    let final_path = final_result_path(root, &spec.dataset_hash);
    if !final_path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&final_path)?;
    match serde_json::from_slice::<DatasetResult>(&bytes) {
        Ok(result)
            if result_matches_campaign(&result, run_id, campaign_sha256, spec, release_sha256) =>
        {
            Ok(Some(result))
        }
        Ok(_) | Err(_) => {
            archive_stale_final_result(&final_path)?;
            Ok(None)
        }
    }
}

fn base_campaign_status(
    run_id: &str,
    strict_winner: bool,
    best_score: Option<&TimeSeriesScore>,
    official_mase: f64,
) -> &'static str {
    if is_forecast3_gen20(run_id) {
        return if strict_winner { "win" } else { "exhausted" };
    }
    match (strict_winner, best_score) {
        (true, _) => "win",
        (_, None) => "exhausted",
        (false, Some(score)) if score.mase.total_cmp(&official_mase) == Ordering::Equal => "tie",
        (false, Some(score)) if score.mase > official_mase => "loss",
        (false, Some(_)) => "exhausted",
    }
}

fn write_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    bank: &SeedBank,
    all_slots: &[LogicalSlot],
    generation_zero_best: Option<f64>,
    release_sha256: &str,
    allow_generation_zero_win: bool,
) -> Result<DatasetResult> {
    let mut ranked = all_slots.to_vec();
    ranked.sort_by(rank_slots);
    let best = ranked.iter().find(|slot| slot.valid);
    let strict_winner = best.is_some_and(|slot| {
        slot_is_base_campaign_winner(slot, spec, generation_zero_best, allow_generation_zero_win)
    });
    let best_score = best.and_then(|slot| slot.score.as_ref());
    let best_beats_official =
        best_score.is_some_and(|score| score.mase < spec.official_leader.mase);
    let status = base_campaign_status(run_id, strict_winner, best_score, spec.official_leader.mase)
        .to_string();
    let result = DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        base_run_id: None,
        base_status: None,
        base_strict_winner: None,
        base_best_jope_mase: None,
        continuation_status: None,
        best_evidence_run_id: None,
        best_receipt_file: best.map(|slot| slot.receipt_file.clone()),
        best_receipt_sha256: best.map(|slot| slot.receipt_sha256.clone()),
        best_prediction_sha256: best.and_then(|slot| slot.prediction_sha256.clone()),
        campaign_sha256: campaign_sha256.to_string(),
        release_binary_sha256: release_sha256.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        status,
        official_leader_model: spec.official_leader.model.clone(),
        official_leader_mase: spec.official_leader.mase,
        generation_zero_best_mase: generation_zero_best,
        best_jope_mase: best_score.map(|score| score.mase),
        best_jope_rmse: best_score.map(|score| score.rmse),
        mase_delta_vs_official: best_score.map(|score| spec.official_leader.mase - score.mase),
        strict_winner,
        best_beats_official,
        best_semantic_sha256: best.map(|slot| slot.member.semantic_sha256.clone()),
        best_canonical_genome: best.map(|slot| slot.member.canonical_genome.clone()),
        best_generation: best.map(|slot| slot.generation),
        best_slot: best.map(|slot| slot.slot),
        seed_source_family: best.and_then(|slot| seed_family(bank, &slot.member.semantic_sha256)),
        logical_evaluations: all_slots.len(),
        valid_evaluations: all_slots.iter().filter(|slot| slot.valid).count(),
        completed_generations: all_slots
            .iter()
            .map(|slot| slot.generation)
            .collect::<BTreeSet<_>>()
            .len(),
        completed_at_unix_ms: unix_millis(),
    };
    let target = final_result_path(root, &spec.dataset_hash);
    write_json(&target, &result)?;
    Ok(result)
}

fn classify_continuation_status(
    base_result: &DatasetResult,
    strict_winner: bool,
    best: Option<&LogicalSlot>,
    best_score: Option<&TimeSeriesScore>,
    spec: &DatasetSpec,
) -> String {
    if base_result.strict_winner {
        let improved = best_score
            .zip(base_result.best_jope_mase)
            .is_some_and(|(score, base_mase)| score.mase < base_mase)
            && best.is_some_and(|slot| slot.generation >= GEN30_START_GENERATION);
        if improved {
            "base_win_improved".to_string()
        } else {
            "inherited_base_win".to_string()
        }
    } else if strict_winner {
        "gen30_strict_win".to_string()
    } else {
        match best_score {
            None => "exhausted".to_string(),
            Some(score) if score.mase.total_cmp(&spec.official_leader.mase) == Ordering::Equal => {
                "tie".to_string()
            }
            Some(score) if score.mase > spec.official_leader.mase => "loss".to_string(),
            Some(_) => "exhausted".to_string(),
        }
    }
}

fn load_reusable_gen30_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    release_sha256: &str,
    base_run_id: &str,
) -> Result<Option<DatasetResult>> {
    let Some(result) =
        load_reusable_dataset_result(root, run_id, campaign_sha256, spec, release_sha256)?
    else {
        return Ok(None);
    };
    if result.base_run_id.as_deref() == Some(base_run_id)
        && result.continuation_status.is_some()
        && result.best_evidence_run_id.is_some()
    {
        Ok(Some(result))
    } else {
        archive_stale_final_result(&final_result_path(root, &spec.dataset_hash))?;
        Ok(None)
    }
}

fn write_gen30_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    bank: &SeedBank,
    all_slots: &[LogicalSlot],
    generation_zero_best: Option<f64>,
    release_sha256: &str,
    base_run_id: &str,
    base_result: &DatasetResult,
) -> Result<DatasetResult> {
    let mut ranked = all_slots.to_vec();
    ranked.sort_by(rank_slots);
    let best = ranked.iter().find(|slot| slot.valid);
    let strict_floor = strict_generation_floor(base_result);
    let strict_winner = best
        .is_some_and(|slot| slot_is_strict_winner(slot, spec, generation_zero_best, strict_floor));
    let best_score = best.and_then(|slot| slot.score.as_ref());
    let best_beats_official =
        best_score.is_some_and(|score| score.mase < spec.official_leader.mase);
    let status = match (strict_winner, best_score) {
        (true, _) => "win",
        (_, None) => "exhausted",
        (false, Some(score))
            if score.mase.total_cmp(&spec.official_leader.mase) == Ordering::Equal =>
        {
            "tie"
        }
        (false, Some(score)) if score.mase > spec.official_leader.mase => "loss",
        (false, Some(_)) => "exhausted",
    }
    .to_string();
    let continuation_status =
        classify_continuation_status(base_result, strict_winner, best, best_score, spec);
    let result = DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        base_run_id: Some(base_run_id.to_string()),
        base_status: Some(base_result.status.clone()),
        base_strict_winner: Some(base_result.strict_winner),
        base_best_jope_mase: base_result.best_jope_mase,
        continuation_status: Some(continuation_status),
        best_evidence_run_id: best.map(|slot| {
            if slot.generation < GEN30_START_GENERATION {
                base_run_id.to_string()
            } else {
                run_id.to_string()
            }
        }),
        best_receipt_file: best.map(|slot| slot.receipt_file.clone()),
        best_receipt_sha256: best.map(|slot| slot.receipt_sha256.clone()),
        best_prediction_sha256: best.and_then(|slot| slot.prediction_sha256.clone()),
        campaign_sha256: campaign_sha256.to_string(),
        release_binary_sha256: release_sha256.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        status,
        official_leader_model: spec.official_leader.model.clone(),
        official_leader_mase: spec.official_leader.mase,
        generation_zero_best_mase: generation_zero_best,
        best_jope_mase: best_score.map(|score| score.mase),
        best_jope_rmse: best_score.map(|score| score.rmse),
        mase_delta_vs_official: best_score.map(|score| spec.official_leader.mase - score.mase),
        strict_winner,
        best_beats_official,
        best_semantic_sha256: best.map(|slot| slot.member.semantic_sha256.clone()),
        best_canonical_genome: best.map(|slot| slot.member.canonical_genome.clone()),
        best_generation: best.map(|slot| slot.generation),
        best_slot: best.map(|slot| slot.slot),
        seed_source_family: best.and_then(|slot| seed_family(bank, &slot.member.semantic_sha256)),
        logical_evaluations: all_slots.len(),
        valid_evaluations: all_slots.iter().filter(|slot| slot.valid).count(),
        completed_generations: all_slots
            .iter()
            .map(|slot| slot.generation)
            .collect::<BTreeSet<_>>()
            .len(),
        completed_at_unix_ms: unix_millis(),
    };
    let target = final_result_path(root, &spec.dataset_hash);
    write_json(&target, &result)?;
    Ok(result)
}

fn run_dataset(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    staging: &StagingReceipt,
    release_sha256: &str,
    config: &CampaignConfig,
    allow_generation_zero_win: bool,
) -> Result<DatasetResult> {
    if let Some(result) =
        load_reusable_dataset_result(root, run_id, campaign_sha256, spec, release_sha256)?
    {
        return Ok(result);
    }
    let (bank, seed_sha256) = load_or_create_seed_bank(root, spec)?;
    let mut members = bank
        .seeds
        .iter()
        .map(SeedRecord::member)
        .collect::<Vec<_>>();
    let mut all_slots = Vec::new();
    let mut generation_zero_best = None;
    for generation in 0..config.generations {
        let mut slots = evaluate_generation(
            root,
            run_id,
            campaign_sha256,
            spec,
            &seed_sha256,
            release_sha256,
            staging,
            generation,
            &members,
        )?;
        let mut ranked = slots.clone();
        ranked.sort_by(rank_slots);
        if generation == 0 {
            generation_zero_best = ranked
                .iter()
                .find(|slot| slot.valid)
                .and_then(|slot| slot.score.as_ref().map(|score| score.mase));
        }
        all_slots.append(&mut slots);
        let has_strict_winner = ranked.iter().any(|slot| {
            slot_is_base_campaign_winner(
                slot,
                spec,
                generation_zero_best,
                allow_generation_zero_win,
            )
        });
        if (has_strict_winner && generation + 1 >= config.minimum_generations.max(1))
            || generation + 1 == config.generations
        {
            break;
        }
        members = next_population(spec, &ranked, generation + 1, "")?;
    }
    write_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        &bank,
        &all_slots,
        generation_zero_best,
        release_sha256,
        allow_generation_zero_win,
    )
}

fn run_gen30_dataset(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    staging: &StagingReceipt,
    release_sha256: &str,
    base: &BaseCampaign,
) -> Result<DatasetResult> {
    if let Some(result) = load_reusable_gen30_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        release_sha256,
        &base.manifest.run_id,
    )? {
        return Ok(result);
    }
    let (bank, seed_sha256) = load_existing_seed_bank(&base.root, spec)?;
    let base_result: DatasetResult = load_json(&final_result_path(&base.root, &spec.dataset_hash))?;
    if !result_matches_campaign(
        &base_result,
        &base.manifest.run_id,
        &base.campaign_sha256,
        spec,
        &base.manifest.release_binary_sha256,
    ) {
        bail!("invalid base TimeSeries42 result for {}", spec.dataset_hash);
    }
    let mut all_slots = Vec::with_capacity(GEN30_GENERATIONS * POPULATION);
    let generation_zero = load_generation_slots(
        &base.root,
        &base.manifest.run_id,
        &base.campaign_sha256,
        spec,
        0,
    )?;
    let generation_zero_best = generation_best_mase(&generation_zero);
    all_slots.extend(generation_zero);
    let generation_one = load_generation_slots(
        &base.root,
        &base.manifest.run_id,
        &base.campaign_sha256,
        spec,
        1,
    )?;
    let mut ranked = generation_one.clone();
    ranked.sort_by(rank_slots);
    all_slots.extend(generation_one);

    let mut members = next_population(spec, &ranked, GEN30_START_GENERATION, "")?;
    for generation in GEN30_START_GENERATION..GEN30_GENERATIONS {
        let mut slots = evaluate_generation(
            root,
            run_id,
            campaign_sha256,
            spec,
            &seed_sha256,
            release_sha256,
            staging,
            generation,
            &members,
        )?;
        let mut generation_ranked = slots.clone();
        generation_ranked.sort_by(rank_slots);
        all_slots.append(&mut slots);
        if continuation_should_stop(
            &base_result,
            &all_slots,
            spec,
            generation_zero_best,
            generation,
        ) {
            break;
        }
        members = next_population(spec, &generation_ranked, generation + 1, "")?;
    }
    write_gen30_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        &bank,
        &all_slots,
        generation_zero_best,
        release_sha256,
        &base.manifest.run_id,
        &base_result,
    )
}

fn load_reusable_winall_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    release_sha256: &str,
    source_run_id: &str,
) -> Result<Option<DatasetResult>> {
    let Some(result) =
        load_reusable_dataset_result(root, run_id, campaign_sha256, spec, release_sha256)?
    else {
        return Ok(None);
    };
    if result.base_run_id.as_deref() == Some(source_run_id)
        && result.continuation_status.is_some()
        && result.best_evidence_run_id.is_some()
    {
        Ok(Some(result))
    } else {
        archive_stale_final_result(&final_result_path(root, &spec.dataset_hash))?;
        Ok(None)
    }
}

fn write_winall_dataset_result(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    bank: &SeedBank,
    all_slots: &[LogicalSlot],
    generation_zero_best: Option<f64>,
    release_sha256: &str,
    source_run_id: &str,
    source_result: &DatasetResult,
    start_generation: usize,
) -> Result<DatasetResult> {
    let mut ranked = all_slots.to_vec();
    ranked.sort_by(rank_slots);
    let best = ranked.iter().find(|slot| slot.valid);
    let strict_winner = best.is_some_and(|slot| {
        slot_is_strict_winner(slot, spec, generation_zero_best, start_generation)
    });
    let best_score = best.and_then(|slot| slot.score.as_ref());
    let best_beats_official =
        best_score.is_some_and(|score| score.mase < spec.official_leader.mase);
    let status = match (strict_winner, best_score) {
        (true, _) => "win",
        (_, None) => "exhausted",
        (false, Some(score))
            if score.mase.total_cmp(&spec.official_leader.mase) == Ordering::Equal =>
        {
            "tie"
        }
        (false, Some(score)) if score.mase > spec.official_leader.mase => "loss",
        (false, Some(_)) => "exhausted",
    }
    .to_string();
    let continuation_status = if strict_winner {
        "winall_strict_win".to_string()
    } else {
        match best_score {
            None => "exhausted".to_string(),
            Some(score) if score.mase.total_cmp(&spec.official_leader.mase) == Ordering::Equal => {
                "tie".to_string()
            }
            Some(score) if score.mase > spec.official_leader.mase => "loss".to_string(),
            Some(_) => "exhausted".to_string(),
        }
    };
    let result = DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        base_run_id: Some(source_run_id.to_string()),
        base_status: Some(source_result.status.clone()),
        base_strict_winner: Some(source_result.strict_winner),
        base_best_jope_mase: source_result.best_jope_mase,
        continuation_status: Some(continuation_status),
        best_evidence_run_id: best.map(|slot| {
            if slot.generation < start_generation {
                source_run_id.to_string()
            } else {
                run_id.to_string()
            }
        }),
        best_receipt_file: best.map(|slot| slot.receipt_file.clone()),
        best_receipt_sha256: best.map(|slot| slot.receipt_sha256.clone()),
        best_prediction_sha256: best.and_then(|slot| slot.prediction_sha256.clone()),
        campaign_sha256: campaign_sha256.to_string(),
        release_binary_sha256: release_sha256.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        status,
        official_leader_model: spec.official_leader.model.clone(),
        official_leader_mase: spec.official_leader.mase,
        generation_zero_best_mase: generation_zero_best,
        best_jope_mase: best_score.map(|score| score.mase),
        best_jope_rmse: best_score.map(|score| score.rmse),
        mase_delta_vs_official: best_score.map(|score| spec.official_leader.mase - score.mase),
        strict_winner,
        best_beats_official,
        best_semantic_sha256: best.map(|slot| slot.member.semantic_sha256.clone()),
        best_canonical_genome: best.map(|slot| slot.member.canonical_genome.clone()),
        best_generation: best.map(|slot| slot.generation),
        best_slot: best.map(|slot| slot.slot),
        seed_source_family: best.and_then(|slot| seed_family(bank, &slot.member.semantic_sha256)),
        logical_evaluations: all_slots.len(),
        valid_evaluations: all_slots.iter().filter(|slot| slot.valid).count(),
        completed_generations: all_slots
            .iter()
            .map(|slot| slot.generation)
            .collect::<BTreeSet<_>>()
            .len(),
        completed_at_unix_ms: unix_millis(),
    };
    let target = final_result_path(root, &spec.dataset_hash);
    write_json(&target, &result)?;
    Ok(result)
}

fn run_winall_dataset(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    staging: &StagingReceipt,
    release_sha256: &str,
    source: &WinallSourceCampaign,
    max_generations: usize,
    lineage_salt: &str,
) -> Result<DatasetResult> {
    if let Some(result) = load_reusable_winall_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        release_sha256,
        &source.manifest.run_id,
    )? {
        return Ok(result);
    }
    let (bank, seed_sha256) = load_existing_seed_bank(&source.base.root, spec)?;
    let source_result: DatasetResult =
        load_json(&final_result_path(&source.root, &spec.dataset_hash))?;
    if source_result.strict_winner {
        bail!(
            "TimeSeries42 win-all source result is already a strict winner for {}",
            spec.dataset_hash
        );
    }
    let start_generation = source.terminal_generation + 1;
    let mut all_slots = Vec::with_capacity(max_generations * POPULATION);
    let generation_zero = load_generation_slots(
        &source.base.root,
        &source.base.manifest.run_id,
        &source.base.campaign_sha256,
        spec,
        0,
    )?;
    let generation_zero_best = generation_best_mase(&generation_zero);
    all_slots.extend(generation_zero);
    let generation_one = load_generation_slots(
        &source.base.root,
        &source.base.manifest.run_id,
        &source.base.campaign_sha256,
        spec,
        1,
    )?;
    all_slots.extend(generation_one);

    let source_start = source
        .manifest
        .continuation_start_generation
        .unwrap_or(GEN30_START_GENERATION);
    let mut ranked = Vec::new();
    for generation in source_start..=source.terminal_generation {
        let slots = load_generation_slots(
            &source.root,
            &source.manifest.run_id,
            &source.campaign_sha256,
            spec,
            generation,
        )?;
        if generation == source.terminal_generation {
            ranked = slots.clone();
        }
        all_slots.extend(slots);
    }
    ranked.sort_by(rank_slots);
    let mut members = next_population(spec, &ranked, start_generation, lineage_salt)?;
    for generation in start_generation..max_generations {
        let mut slots = evaluate_generation(
            root,
            run_id,
            campaign_sha256,
            spec,
            &seed_sha256,
            release_sha256,
            staging,
            generation,
            &members,
        )?;
        let mut generation_ranked = slots.clone();
        generation_ranked.sort_by(rank_slots);
        all_slots.append(&mut slots);
        if has_strict_winner_from_generation(
            &all_slots,
            spec,
            generation_zero_best,
            start_generation,
        ) || generation + 1 == max_generations
        {
            break;
        }
        members = next_population(spec, &generation_ranked, generation + 1, lineage_salt)?;
    }
    write_winall_dataset_result(
        root,
        run_id,
        campaign_sha256,
        spec,
        &bank,
        &all_slots,
        generation_zero_best,
        release_sha256,
        &source.manifest.run_id,
        &source_result,
        start_generation,
    )
}

fn initialize(run_id: &str, resume: bool) -> Result<(CampaignManifest, StagingReceipt, String)> {
    validate_run_id(run_id)?;
    let expected_coverage = if is_forecast3_gen20(run_id) {
        Some(validate_coverage_binding()?)
    } else {
        None
    };
    let root = campaign_root(run_id);
    fs::create_dir_all(&root)?;
    let release_sha256 = current_binary_sha256()?;
    let corpus_eval_sha256 = current_corpus_eval_sha256()?;
    let scq_binary_sha256 = sha256_file(Path::new(scq_binary_str()))?;
    let manifest_path = root.join("campaign.v1.json");
    let expected_config = base_campaign_config(run_id);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("TimeSeries42 campaign exists; pass --resume to reuse it");
        }
        let bytes = fs::read(&manifest_path)?;
        let manifest: CampaignManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&manifest)?
            || manifest.schema != CAMPAIGN_SCHEMA
            || !manifest.test_adaptive
            || manifest.benchmark_valid
            || manifest.run_id != run_id
            || manifest.config != expected_config
            || manifest.coverage_binding != expected_coverage
            || manifest.release_binary_sha256 != release_sha256
            || manifest.corpus_eval_sha256 != corpus_eval_sha256
            || manifest.scq_binary != scq_binary_str()
            || manifest.scq_binary_sha256 != scq_binary_sha256
            || (is_forecast3_gen20(run_id)
                && manifest.seed_populations
                    != seed_population_bindings(&root, &manifest.datasets, false)?)
        {
            bail!(
                "existing TimeSeries42 campaign manifest is invalid or bound to a different binary"
            );
        }
        manifest
    } else {
        let mut specs = load_dataset_specs(false)?;
        if let Some(coverage) = expected_coverage.as_ref() {
            bind_forecast3_comparators(&mut specs, coverage)?;
        }
        let ids = ordered_ids(&specs);
        let seed_populations = if is_forecast3_gen20(run_id) {
            seed_population_bindings(&root, &specs, true)?
        } else {
            Vec::new()
        };
        let manifest = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            base_run_id: None,
            base_campaign_sha256: None,
            continuation_start_generation: None,
            continuation_terminal_generation: None,
            registry_root: registry_str().to_string(),
            output_root: output_root_str().to_string(),
            release_binary_sha256: release_sha256.clone(),
            corpus_eval_sha256: corpus_eval_sha256.clone(),
            scq_binary: scq_binary_str().to_string(),
            scq_binary_sha256,
            coverage_binding: expected_coverage,
            seed_populations,
            ordered_dataset_ids: ids.clone(),
            ordered_cohort_sha256: ordered_cohort_sha256(&ids),
            config: expected_config,
            datasets: specs,
            started_at_unix_ms: unix_millis(),
        };
        write_json(&manifest_path, &manifest)?;
        manifest
    };
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let (staging, _) = stage_campaign(
        &root,
        &manifest.datasets,
        &manifest.release_binary_sha256,
        &manifest.corpus_eval_sha256,
    )?;
    preflight_representative_scq_checks(&root, &staging, run_id)?;
    Ok((manifest, staging, campaign_sha256))
}

#[derive(Clone)]
struct BaseCampaign {
    root: PathBuf,
    manifest: CampaignManifest,
    campaign_sha256: String,
    summary: TerminalSummary,
}

fn load_base_campaign(base_run_id: &str) -> Result<BaseCampaign> {
    validate_run_id(base_run_id)?;
    let root = campaign_root(base_run_id);
    let manifest_path = root.join("campaign.v1.json");
    let summary_path = root.join("terminal-summary.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let summary: TerminalSummary = load_json(&summary_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    if manifest.schema != CAMPAIGN_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != base_run_id
        || manifest.base_run_id.is_some()
        || manifest.config.generations != GENERATIONS
        || manifest.datasets.len() != 42
        || summary.schema != SUMMARY_SCHEMA
        || !summary.test_adaptive
        || summary.benchmark_valid
        || summary.run_id != base_run_id
        || summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("invalid TimeSeries42 base campaign {base_run_id}");
    }
    let mut summary_ids = summary
        .datasets
        .iter()
        .map(|row| row.dataset_hash.clone())
        .collect::<Vec<_>>();
    summary_ids.sort();
    let mut manifest_ids = ordered_ids(&manifest.datasets);
    manifest_ids.sort();
    if summary_ids != manifest_ids {
        bail!("TimeSeries42 base summary/dataset manifest mismatch");
    }
    for spec in &manifest.datasets {
        let result: DatasetResult = load_json(&final_result_path(&root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            base_run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) || result.completed_generations != GENERATIONS
        {
            bail!(
                "invalid TimeSeries42 base dataset result for {}",
                spec.dataset_hash
            );
        }
    }
    Ok(BaseCampaign {
        root,
        manifest,
        campaign_sha256,
        summary,
    })
}

#[derive(Clone)]
struct WinallSourceCampaign {
    base: BaseCampaign,
    root: PathBuf,
    manifest: CampaignManifest,
    campaign_sha256: String,
    summary: TerminalSummary,
    terminal_generation: usize,
}

fn load_winall_source_campaign(source_run_id: &str) -> Result<WinallSourceCampaign> {
    validate_run_id(source_run_id)?;
    let root = campaign_root(source_run_id);
    let manifest_path = root.join("campaign.v1.json");
    let summary_path = root.join("terminal-summary.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let summary: TerminalSummary = load_json(&summary_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let base_run_id = manifest
        .base_run_id
        .as_deref()
        .context("TimeSeries42 win-all source has no base_run_id")?;
    let base = load_base_campaign(base_run_id)?;
    let terminal_generation = manifest
        .continuation_terminal_generation
        .unwrap_or_else(|| manifest.config.generations.saturating_sub(1));
    if manifest.schema != CAMPAIGN_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != source_run_id
        || manifest.base_campaign_sha256.as_deref() != Some(&base.campaign_sha256)
        || manifest.continuation_start_generation != Some(GEN30_START_GENERATION)
        || terminal_generation + 1 != manifest.config.generations
        || manifest.datasets != base.manifest.datasets
        || summary.schema != SUMMARY_SCHEMA
        || !summary.test_adaptive
        || summary.benchmark_valid
        || summary.run_id != source_run_id
        || summary.dataset_count != manifest.datasets.len()
        || summary.wins + summary.losses + summary.ties + summary.exhausted
            != manifest.datasets.len()
    {
        bail!("invalid TimeSeries42 win-all source campaign {source_run_id}");
    }
    for spec in &manifest.datasets {
        let result: DatasetResult = load_json(&final_result_path(&root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            source_run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) {
            bail!(
                "invalid TimeSeries42 win-all source result for {}",
                spec.dataset_hash
            );
        }
    }
    Ok(WinallSourceCampaign {
        base,
        root,
        manifest,
        campaign_sha256,
        summary,
        terminal_generation,
    })
}

fn winall_source_non_winners(source: &WinallSourceCampaign) -> Result<Vec<DatasetSpec>> {
    let mut results = BTreeMap::new();
    for spec in &source.manifest.datasets {
        let result: DatasetResult =
            load_json(&final_result_path(&source.root, &spec.dataset_hash))?;
        results.insert(spec.dataset_hash.as_str(), result.strict_winner);
    }
    let specs = source
        .manifest
        .datasets
        .iter()
        .filter(|spec| {
            !results
                .get(spec.dataset_hash.as_str())
                .copied()
                .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();
    if specs.is_empty() {
        bail!(
            "TimeSeries42 source campaign {} already has strict winners for every dataset",
            source.manifest.run_id
        );
    }
    Ok(specs)
}

fn initialize_gen30(
    base_run_id: &str,
    run_id: &str,
    resume: bool,
) -> Result<(BaseCampaign, CampaignManifest, StagingReceipt, String)> {
    validate_run_id(run_id)?;
    if base_run_id == run_id {
        bail!("TimeSeries42 gen30 continuation run id must differ from the base run id");
    }
    let base = load_base_campaign(base_run_id)?;
    let root = campaign_root(run_id);
    fs::create_dir_all(&root)?;
    let release_sha256 = current_binary_sha256()?;
    let corpus_eval_sha256 = current_corpus_eval_sha256()?;
    let scq_binary_sha256 = sha256_file(Path::new(scq_binary_str()))?;
    let manifest_path = root.join("campaign.v1.json");
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("TimeSeries42 gen30 campaign exists; pass --resume to reuse it");
        }
        let bytes = fs::read(&manifest_path)?;
        let manifest: CampaignManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&manifest)?
            || manifest.schema != CAMPAIGN_SCHEMA
            || !manifest.test_adaptive
            || manifest.benchmark_valid
            || manifest.run_id != run_id
            || manifest.base_run_id.as_deref() != Some(base_run_id)
            || manifest.base_campaign_sha256.as_deref() != Some(&base.campaign_sha256)
            || manifest.continuation_start_generation != Some(GEN30_START_GENERATION)
            || manifest.continuation_terminal_generation != Some(GEN30_GENERATIONS - 1)
            || manifest.config != CampaignConfig::gen30()
            || manifest.ordered_dataset_ids != base.manifest.ordered_dataset_ids
            || manifest.datasets != base.manifest.datasets
        {
            bail!("existing TimeSeries42 gen30 manifest is invalid or bound to different inputs");
        }
        manifest
    } else {
        let ids = base.manifest.ordered_dataset_ids.clone();
        let manifest = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            base_run_id: Some(base_run_id.to_string()),
            base_campaign_sha256: Some(base.campaign_sha256.clone()),
            continuation_start_generation: Some(GEN30_START_GENERATION),
            continuation_terminal_generation: Some(GEN30_GENERATIONS - 1),
            registry_root: registry_str().to_string(),
            output_root: output_root_str().to_string(),
            release_binary_sha256: release_sha256.clone(),
            corpus_eval_sha256: corpus_eval_sha256.clone(),
            scq_binary: scq_binary_str().to_string(),
            scq_binary_sha256,
            coverage_binding: None,
            seed_populations: Vec::new(),
            ordered_dataset_ids: ids.clone(),
            ordered_cohort_sha256: ordered_cohort_sha256(&ids),
            config: CampaignConfig::gen30(),
            datasets: base.manifest.datasets.clone(),
            started_at_unix_ms: unix_millis(),
        };
        write_json(&manifest_path, &manifest)?;
        manifest
    };
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let (staging, _) = stage_campaign(
        &root,
        &manifest.datasets,
        &manifest.release_binary_sha256,
        &manifest.corpus_eval_sha256,
    )?;
    preflight_representative_scq_checks(&root, &staging, run_id)?;
    Ok((base, manifest, staging, campaign_sha256))
}

fn initialize_winall(
    source_run_id: &str,
    run_id: &str,
    resume: bool,
    max_generations: usize,
    lineage_salt: &str,
) -> Result<(
    WinallSourceCampaign,
    CampaignManifest,
    StagingReceipt,
    String,
)> {
    validate_run_id(run_id)?;
    if source_run_id == run_id {
        bail!("TimeSeries42 win-all run id must differ from the source run id");
    }
    let source = load_winall_source_campaign(source_run_id)?;
    let start_generation = source.terminal_generation + 1;
    if max_generations <= start_generation {
        bail!("TimeSeries42 win-all max_generations must be greater than {start_generation}");
    }
    let winall_specs = winall_source_non_winners(&source)?;
    let root = campaign_root(run_id);
    fs::create_dir_all(&root)?;
    let release_sha256 = current_binary_sha256()?;
    let corpus_eval_sha256 = current_corpus_eval_sha256()?;
    let scq_binary_sha256 = sha256_file(Path::new(scq_binary_str()))?;
    let manifest_path = root.join("campaign.v1.json");
    let expected_config = CampaignConfig::winall(max_generations, start_generation, lineage_salt);
    let ids = ordered_ids(&winall_specs);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("TimeSeries42 win-all campaign exists; pass --resume to reuse it");
        }
        let bytes = fs::read(&manifest_path)?;
        let manifest: CampaignManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&manifest)?
            || manifest.schema != CAMPAIGN_SCHEMA
            || !manifest.test_adaptive
            || manifest.benchmark_valid
            || manifest.run_id != run_id
            || manifest.base_run_id.as_deref() != Some(source_run_id)
            || manifest.base_campaign_sha256.as_deref() != Some(&source.campaign_sha256)
            || manifest.continuation_start_generation != Some(start_generation)
            || manifest.continuation_terminal_generation != Some(max_generations - 1)
            || manifest.config != expected_config
            || manifest.ordered_dataset_ids != ids
            || manifest.datasets != winall_specs
        {
            bail!("existing TimeSeries42 win-all manifest is invalid or bound to different inputs");
        }
        manifest
    } else {
        let manifest = CampaignManifest {
            schema: CAMPAIGN_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            base_run_id: Some(source_run_id.to_string()),
            base_campaign_sha256: Some(source.campaign_sha256.clone()),
            continuation_start_generation: Some(start_generation),
            continuation_terminal_generation: Some(max_generations - 1),
            registry_root: registry_str().to_string(),
            output_root: output_root_str().to_string(),
            release_binary_sha256: release_sha256.clone(),
            corpus_eval_sha256: corpus_eval_sha256.clone(),
            scq_binary: scq_binary_str().to_string(),
            scq_binary_sha256,
            coverage_binding: None,
            seed_populations: Vec::new(),
            ordered_dataset_ids: ids.clone(),
            ordered_cohort_sha256: ordered_cohort_sha256(&ids),
            config: expected_config,
            datasets: winall_specs,
            started_at_unix_ms: unix_millis(),
        };
        write_json(&manifest_path, &manifest)?;
        manifest
    };
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let (staging, _) = stage_campaign(
        &root,
        &manifest.datasets,
        &manifest.release_binary_sha256,
        &manifest.corpus_eval_sha256,
    )?;
    preflight_representative_scq_checks(&root, &staging, run_id)?;
    Ok((source, manifest, staging, campaign_sha256))
}

fn rebuild_summary(
    root: &Path,
    run_id: &str,
    scan_output: Option<String>,
) -> Result<TerminalSummary> {
    let manifest_path = root.join("campaign.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let mut datasets = Vec::with_capacity(manifest.datasets.len());
    for spec in &manifest.datasets {
        let result: DatasetResult = load_json(&final_result_path(root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) {
            bail!(
                "TimeSeries42 dataset result is stale for {}",
                spec.dataset_hash
            );
        }
        datasets.push(result);
    }
    datasets.sort_by(|left, right| left.dataset_hash.cmp(&right.dataset_hash));
    let logical_evaluations = datasets.iter().map(|row| row.logical_evaluations).sum();
    let valid_evaluations = datasets.iter().map(|row| row.valid_evaluations).sum();
    let summary = TerminalSummary {
        schema: SUMMARY_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        base_run_id: manifest.base_run_id.clone(),
        inherited_base_wins: manifest.base_run_id.as_ref().map(|_| {
            datasets
                .iter()
                .filter(|row| row.continuation_status.as_deref() == Some("inherited_base_win"))
                .count()
        }),
        improved_base_wins: manifest.base_run_id.as_ref().map(|_| {
            datasets
                .iter()
                .filter(|row| row.continuation_status.as_deref() == Some("base_win_improved"))
                .count()
        }),
        continuation_strict_wins: manifest.base_run_id.as_ref().map(|_| {
            datasets
                .iter()
                .filter(|row| {
                    matches!(
                        row.continuation_status.as_deref(),
                        Some("gen30_strict_win" | "winall_strict_win")
                    )
                })
                .count()
        }),
        dataset_count: datasets.len(),
        wins: datasets.iter().filter(|row| row.status == "win").count(),
        losses: datasets.iter().filter(|row| row.status == "loss").count(),
        ties: datasets.iter().filter(|row| row.status == "tie").count(),
        exhausted: datasets
            .iter()
            .filter(|row| row.status == "exhausted")
            .count(),
        logical_evaluations,
        valid_evaluations,
        datasets,
        sotarena_scan_output: scan_output,
        completed_at_unix_ms: unix_millis(),
    };
    write_current_json(
        &root.join("terminal-summary.v1.json"),
        &summary,
        "terminal-summary",
    )?;
    write_ranked_csv(root, &summary)?;
    Ok(summary)
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn write_ranked_csv(root: &Path, summary: &TerminalSummary) -> Result<()> {
    let mut rows = summary.datasets.clone();
    rows.sort_by(|left, right| {
        let left_mase = left.best_jope_mase.unwrap_or(f64::INFINITY);
        let right_mase = right.best_jope_mase.unwrap_or(f64::INFINITY);
        left.status
            .cmp(&right.status)
            .then(left_mase.total_cmp(&right_mase))
            .then(left.dataset_hash.cmp(&right.dataset_hash))
    });
    let mut bytes = b"dataset,status,official_leader,official_mase,best_mase,best_rmse,mase_delta,best_generation,best_slot,semantic_sha256,seed_family\n".to_vec();
    for row in rows {
        bytes.extend_from_slice(
            format!(
                "{},{},{},{:.17},{},{},{},{},{},{},{}\n",
                row.dataset_hash,
                row.status,
                csv_field(&row.official_leader_model),
                row.official_leader_mase,
                row.best_jope_mase
                    .map(|value| format!("{value:.17}"))
                    .unwrap_or_default(),
                row.best_jope_rmse
                    .map(|value| format!("{value:.17}"))
                    .unwrap_or_default(),
                row.mase_delta_vs_official
                    .map(|value| format!("{value:.17}"))
                    .unwrap_or_default(),
                row.best_generation
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                row.best_slot
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                row.best_semantic_sha256.unwrap_or_default(),
                csv_field(row.seed_source_family.as_deref().unwrap_or("")),
            )
            .as_bytes(),
        );
    }
    immutable_or_equal(&root.join("ranked-summary.csv"), &bytes)
}

fn run_sotarena_scan(root: &Path) -> Result<String> {
    let output = Command::new(sotarena_pull_str())
        .args(["scan", "--task", "timeseries"])
        .output()
        .context("run sotarena_pull scan --task timeseries")?;
    let scan_dir = root.join("sotarena-scan");
    fs::create_dir_all(&scan_dir)?;
    fs::write(scan_dir.join("stdout.txt"), &output.stdout)?;
    fs::write(scan_dir.join("stderr.txt"), &output.stderr)?;
    if !output.status.success() {
        bail!(
            "sotarena_pull scan failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn timeseries_report_value(
    spec: &DatasetSpec,
    result: &DatasetResult,
    terminal_summary_path: &Path,
    terminal_summary_sha256: &str,
) -> Value {
    json!({
        "report_schema_version": JOPE_TIMESERIES_REPORT_SCHEMA,
        "algorithm": "jope-dime-regressor",
        "model_name": "jope-dime-regressor",
        "model_class": "aspfm_generated_rust_timeseries_reduction",
        "task_type": "timeseries",
        "status": "completed",
        "dataset_hash": result.dataset_hash,
        "source_path": result.best_receipt_file,
        "benchmark_valid": false,
        "test_adaptive": true,
        "selection_bias": "test-selected",
        "train_rows": spec.train_rows,
        "test_rows": spec.test_rows,
        "n_features": spec.reduction_features,
        "metrics": {
            "mase": result.best_jope_mase,
            "negative_mase": result.best_jope_mase.map(|value| -value),
            "rmse": result.best_jope_rmse,
            "mase_delta_vs_official": result.mase_delta_vs_official,
            "official_leader_mase": result.official_leader_mase,
            "strict_winner": result.strict_winner,
            "best_beats_official": result.best_beats_official
        },
        "comparison": {
            "official_leader_model": result.official_leader_model,
            "official_leader_mase": result.official_leader_mase,
            "generation_zero_best_mase": result.generation_zero_best_mase,
            "terminal_status": result.status,
            "strict_winner": result.strict_winner,
            "continuation_status": result.continuation_status,
            "base_status": result.base_status,
            "base_strict_winner": result.base_strict_winner,
            "base_best_jope_mase": result.base_best_jope_mase
        },
        "genome": result.best_canonical_genome,
        "genome_hash": result.best_semantic_sha256,
        "semantic_sha256": result.best_semantic_sha256,
        "provenance": {
            "run_id": result.run_id,
            "base_run_id": result.base_run_id,
            "best_evidence_run_id": result.best_evidence_run_id,
            "campaign_sha256": result.campaign_sha256,
            "release_binary_sha256": result.release_binary_sha256,
            "terminal_summary_path": terminal_summary_path.to_string_lossy(),
            "terminal_summary_sha256": terminal_summary_sha256,
            "receipt_file": result.best_receipt_file,
            "receipt_sha256": result.best_receipt_sha256,
            "prediction_sha256": result.best_prediction_sha256,
            "best_generation": result.best_generation,
            "best_slot": result.best_slot,
            "logical_evaluations": result.logical_evaluations,
            "valid_evaluations": result.valid_evaluations,
            "completed_generations": result.completed_generations,
            "completed_at_unix_ms": result.completed_at_unix_ms
        },
        "environment": {
            "execution_host": "scq",
            "nodes": NODES,
            "candidate_timeout_seconds": CANDIDATE_TIMEOUT_SECONDS,
            "candidate_threads": SHARD_SIZE,
            "corpus_eval": corpus_eval_str()
        }
    })
}

fn write_report_file(path: &Path, value: &Value) -> Result<String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    immutable_or_equal(path, &bytes)?;
    Ok(sha256_bytes(&bytes))
}

fn write_timeseries_dataset_reports(
    manifest: &CampaignManifest,
    summary: &TerminalSummary,
    summary_path: &Path,
) -> Result<usize> {
    let summary_sha256 = sha256_file(summary_path)?;
    let specs = manifest
        .datasets
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<BTreeMap<_, _>>();
    let mut written = 0usize;
    for result in &summary.datasets {
        if result.best_jope_mase.is_none() {
            continue;
        }
        let spec = specs
            .get(result.dataset_hash.as_str())
            .with_context(|| format!("missing spec for {}", result.dataset_hash))?;
        let mut report = timeseries_report_value(spec, result, summary_path, &summary_sha256);
        let checkpoint_path = result.best_generation.map(|generation| {
            dataset_root(&campaign_root(&manifest.run_id), &result.dataset_hash)
                .join("generations")
                .join(format!("generation-{generation:06}"))
                .join("generation.v1.json")
        });
        let checkpoint_sha256 = checkpoint_path
            .as_ref()
            .filter(|path| path.is_file())
            .map(|path| sha256_file(path))
            .transpose()?;
        let object = report
            .as_object_mut()
            .context("timeseries report must be an object")?;
        object.insert("official_combined_rank_eligible".to_string(), json!(false));
        object.insert("ranking_eligible".to_string(), json!(false));
        object.insert(
            "comparator".to_string(),
            json!({
                "lock_sha256": spec.comparator_lock_sha256,
                "accepted_report_count": spec.accepted_external_reports.len(),
                "winner_model": spec.official_leader.model,
                "winner_report_sha256": spec.official_leader.report_sha256,
            }),
        );
        object.insert(
            "checkpoint".to_string(),
            json!({
                "path": checkpoint_path,
                "sha256": checkpoint_sha256,
            }),
        );
        object.insert(
            "confirmation".to_string(),
            json!({
                "strict_beat_confirmed": result.strict_winner,
                "receipt_path": result.best_receipt_file,
                "receipt_sha256": result.best_receipt_sha256,
                "prediction_sha256": result.best_prediction_sha256,
            }),
        );
        let report_path = Path::new(registry_str())
            .join(&result.dataset_hash)
            .join("reports")
            .join(JOPE_TIMESERIES_REPORT_FILE);
        write_report_file(&report_path, &report)?;
        written += 1;
    }
    Ok(written)
}

pub(crate) fn run_all(run_id: &str, resume: bool) -> Result<()> {
    let (manifest, staging, campaign_sha256) = initialize(run_id, resume)?;
    let root = campaign_root(run_id);
    if root.join("terminal-summary.v1.json").is_file() {
        validate_all(run_id)?;
        let summary: TerminalSummary = load_json(&root.join("terminal-summary.v1.json"))?;
        write_timeseries_dataset_reports(
            &manifest,
            &summary,
            &root.join("terminal-summary.v1.json"),
        )?;
        return Ok(());
    }
    let dataset_workers = dataset_worker_count()?;
    let allow_generation_zero_win = is_forecast3_gen20(run_id);
    println!(
        "TimeSeries42 dataset_workers={dataset_workers} shards_per_generation={SHARDS} shard_cpus={SHARD_SIZE}"
    );
    let queue = Arc::new(Mutex::new(VecDeque::from(manifest.datasets.clone())));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for _ in 0..dataset_workers {
            let queue = Arc::clone(&queue);
            let errors = Arc::clone(&errors);
            let staging = staging.clone();
            let release_sha256 = manifest.release_binary_sha256.clone();
            let campaign_sha256 = campaign_sha256.clone();
            let root = root.clone();
            let run_id = run_id.to_string();
            let config = manifest.config.clone();
            scope.spawn(move || loop {
                let Some(spec) = queue.lock().unwrap().pop_front() else {
                    break;
                };
                if let Err(error) = run_dataset(
                    &root,
                    &run_id,
                    &campaign_sha256,
                    &spec,
                    &staging,
                    &release_sha256,
                    &config,
                    allow_generation_zero_win,
                ) {
                    errors
                        .lock()
                        .unwrap()
                        .push(format!("{}: {error:#}", spec.dataset_hash));
                }
            });
        }
    });
    let errors = Arc::try_unwrap(errors)
        .expect("TimeSeries42 error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "TimeSeries42 retained dataset failures: {}",
            errors.join("; ")
        );
    }
    let scan_output = run_sotarena_scan(&root).ok();
    let summary = rebuild_summary(&root, run_id, scan_output)?;
    if summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("TimeSeries42 did not finish with 42 terminal datasets");
    }
    let reports_written = write_timeseries_dataset_reports(
        &manifest,
        &summary,
        &root.join("terminal-summary.v1.json"),
    )?;
    println!(
        "TimeSeries42 complete: wins={} losses={} ties={} exhausted={} valid_evaluations={} dataset_reports_written={}",
        summary.wins, summary.losses, summary.ties, summary.exhausted, summary.valid_evaluations, reports_written
    );
    Ok(())
}

fn validate_gen30_dataset_result(
    result: &DatasetResult,
    base_result: &DatasetResult,
    base_run_id: &str,
) -> Result<()> {
    if result.base_run_id.as_deref() != Some(base_run_id)
        || result.base_status.as_deref() != Some(base_result.status.as_str())
        || result.base_strict_winner != Some(base_result.strict_winner)
        || result.base_best_jope_mase != base_result.best_jope_mase
        || result.continuation_status.is_none()
        || result.best_evidence_run_id.is_none()
        || result.completed_generations < GEN30_START_GENERATION + 1
        || result.completed_generations > GEN30_GENERATIONS
    {
        bail!(
            "invalid TimeSeries42 gen30 continuation fields for {}",
            result.dataset_hash
        );
    }
    match result.continuation_status.as_deref() {
        Some("inherited_base_win") => {
            if !base_result.strict_winner
                || !result.strict_winner
                || result
                    .best_generation
                    .is_some_and(|generation| generation >= GEN30_START_GENERATION)
                || result.best_evidence_run_id.as_deref() != Some(base_run_id)
                || result.completed_generations != GEN30_GENERATIONS
            {
                bail!(
                    "invalid inherited base win status for {}",
                    result.dataset_hash
                );
            }
        }
        Some("base_win_improved") => {
            if !base_result.strict_winner
                || !result.strict_winner
                || result
                    .best_generation
                    .is_none_or(|generation| generation < GEN30_START_GENERATION)
                || result.best_evidence_run_id.as_deref() == Some(base_run_id)
                || result.completed_generations != GEN30_GENERATIONS
            {
                bail!(
                    "invalid improved base win status for {}",
                    result.dataset_hash
                );
            }
        }
        Some("gen30_strict_win") => {
            if base_result.strict_winner
                || !result.strict_winner
                || result
                    .best_generation
                    .is_none_or(|generation| generation < GEN30_START_GENERATION)
                || result.best_evidence_run_id.as_deref() == Some(base_run_id)
            {
                bail!(
                    "invalid gen30 strict win status for {}",
                    result.dataset_hash
                );
            }
        }
        Some("loss" | "tie" | "exhausted") => {
            if result.strict_winner || result.completed_generations != GEN30_GENERATIONS {
                bail!(
                    "invalid terminal non-win status for {}",
                    result.dataset_hash
                );
            }
        }
        Some(other) => bail!(
            "unknown TimeSeries42 gen30 continuation status {other} for {}",
            result.dataset_hash
        ),
        None => unreachable!("checked above"),
    }
    Ok(())
}

pub(crate) fn run_gen30(base_run_id: &str, run_id: &str, resume: bool) -> Result<()> {
    let (base, manifest, staging, campaign_sha256) = initialize_gen30(base_run_id, run_id, resume)?;
    let root = campaign_root(run_id);
    if root.join("terminal-summary.v1.json").is_file() {
        validate_gen30(base_run_id, run_id)?;
        return Ok(());
    }
    let dataset_workers = dataset_worker_count()?;
    println!(
        "TimeSeries42 gen30 dataset_workers={dataset_workers} shards_per_generation={SHARDS} shard_cpus={SHARD_SIZE}"
    );
    let queue = Arc::new(Mutex::new(VecDeque::from(manifest.datasets.clone())));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for _ in 0..dataset_workers {
            let queue = Arc::clone(&queue);
            let errors = Arc::clone(&errors);
            let staging = staging.clone();
            let release_sha256 = manifest.release_binary_sha256.clone();
            let campaign_sha256 = campaign_sha256.clone();
            let root = root.clone();
            let run_id = run_id.to_string();
            let base = base.clone();
            scope.spawn(move || loop {
                let Some(spec) = queue.lock().unwrap().pop_front() else {
                    break;
                };
                if let Err(error) = run_gen30_dataset(
                    &root,
                    &run_id,
                    &campaign_sha256,
                    &spec,
                    &staging,
                    &release_sha256,
                    &base,
                ) {
                    errors
                        .lock()
                        .unwrap()
                        .push(format!("{}: {error:#}", spec.dataset_hash));
                }
            });
        }
    });
    let errors = Arc::try_unwrap(errors)
        .expect("TimeSeries42 gen30 error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "TimeSeries42 gen30 retained dataset failures: {}",
            errors.join("; ")
        );
    }
    let scan_output = run_sotarena_scan(&root).ok();
    let summary = rebuild_summary(&root, run_id, scan_output)?;
    if summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("TimeSeries42 gen30 did not finish with 42 terminal datasets");
    }
    let reports_written = write_timeseries_dataset_reports(
        &manifest,
        &summary,
        &root.join("terminal-summary.v1.json"),
    )?;
    println!(
        "TimeSeries42 gen30 complete: wins={} losses={} ties={} exhausted={} inherited_base_wins={} improved_base_wins={} continuation_strict_wins={} valid_evaluations={} dataset_reports_written={}",
        summary.wins,
        summary.losses,
        summary.ties,
        summary.exhausted,
        summary.inherited_base_wins.unwrap_or(0),
        summary.improved_base_wins.unwrap_or(0),
        summary.continuation_strict_wins.unwrap_or(0),
        summary.valid_evaluations,
        reports_written
    );
    Ok(())
}

pub(crate) fn run_winall(
    source_run_id: &str,
    run_id: &str,
    resume: bool,
    max_generations: Option<usize>,
    lineage_salt: Option<&str>,
) -> Result<()> {
    let max_generations = max_generations.unwrap_or(DEFAULT_WINALL_GENERATIONS);
    let lineage_salt = lineage_salt.unwrap_or("");
    let (source, manifest, staging, campaign_sha256) =
        initialize_winall(source_run_id, run_id, resume, max_generations, lineage_salt)?;
    let root = campaign_root(run_id);
    if root.join("terminal-summary.v1.json").is_file() {
        let summary = rebuild_summary(&root, run_id, None)?;
        let reports_written = write_timeseries_dataset_reports(
            &manifest,
            &summary,
            &root.join("terminal-summary.v1.json"),
        )?;
        println!(
            "TimeSeries42 win-all already complete: source_wins={} continuation_wins={} total_wins={} remaining_losses={} ties={} exhausted={} dataset_reports_written={}",
            source.summary.wins,
            summary.wins,
            source.summary.wins + summary.wins,
            summary.losses,
            summary.ties,
            summary.exhausted,
            reports_written
        );
        return Ok(());
    }
    let dataset_workers = dataset_worker_count()?;
    println!(
        "TimeSeries42 win-all dataset_workers={dataset_workers} source_non_winners={} start_generation={} max_generations={} shards_per_generation={SHARDS} shard_cpus={SHARD_SIZE}",
        manifest.datasets.len(),
        manifest.continuation_start_generation.unwrap_or(GEN30_GENERATIONS),
        max_generations
    );
    let queue = Arc::new(Mutex::new(VecDeque::from(manifest.datasets.clone())));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for _ in 0..dataset_workers {
            let queue = Arc::clone(&queue);
            let errors = Arc::clone(&errors);
            let staging = staging.clone();
            let release_sha256 = manifest.release_binary_sha256.clone();
            let campaign_sha256 = campaign_sha256.clone();
            let root = root.clone();
            let run_id = run_id.to_string();
            let source = source.clone();
            let lineage_salt = manifest.config.lineage_salt.clone();
            scope.spawn(move || loop {
                let Some(spec) = queue.lock().unwrap().pop_front() else {
                    break;
                };
                if let Err(error) = run_winall_dataset(
                    &root,
                    &run_id,
                    &campaign_sha256,
                    &spec,
                    &staging,
                    &release_sha256,
                    &source,
                    max_generations,
                    &lineage_salt,
                ) {
                    errors
                        .lock()
                        .unwrap()
                        .push(format!("{}: {error:#}", spec.dataset_hash));
                }
            });
        }
    });
    let errors = Arc::try_unwrap(errors)
        .expect("TimeSeries42 win-all error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "TimeSeries42 win-all retained dataset failures: {}",
            errors.join("; ")
        );
    }
    let summary = rebuild_summary(&root, run_id, None)?;
    if summary.dataset_count != manifest.datasets.len()
        || summary.wins + summary.losses + summary.ties + summary.exhausted
            != manifest.datasets.len()
    {
        bail!("TimeSeries42 win-all did not finish with terminal source non-winners");
    }
    let reports_written = write_timeseries_dataset_reports(
        &manifest,
        &summary,
        &root.join("terminal-summary.v1.json"),
    )?;
    println!(
        "TimeSeries42 win-all complete: source_wins={} continuation_wins={} total_wins={} remaining_losses={} ties={} exhausted={} valid_evaluations={} dataset_reports_written={}",
        source.summary.wins,
        summary.wins,
        source.summary.wins + summary.wins,
        summary.losses,
        summary.ties,
        summary.exhausted,
        summary.valid_evaluations,
        reports_written
    );
    Ok(())
}

pub(crate) fn validate_all(run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let expected_coverage = if is_forecast3_gen20(run_id) {
        Some(validate_coverage_binding()?)
    } else {
        None
    };
    let root = campaign_root(run_id);
    let manifest_path = root.join("campaign.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    if manifest.schema != CAMPAIGN_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != run_id
        || manifest.datasets.len() != 42
        || manifest.config != base_campaign_config(run_id)
        || manifest.coverage_binding != expected_coverage
    {
        bail!("invalid TimeSeries42 campaign manifest");
    }
    for spec in &manifest.datasets {
        if is_forecast3_gen20(run_id)
            && (spec.accepted_external_reports.len()
                != BASELINE_MODELS
                    .iter()
                    .filter(|model| !declared_baseline_skip(&spec.dataset_hash, model))
                    .count()
                || spec.comparator_lock_sha256.len() != 64)
        {
            bail!(
                "TimeSeries42 dataset {} is not bound to the forecast3 comparator",
                spec.dataset_hash
            );
        }
        let (bank, _) = load_or_create_seed_bank(&root, spec)?;
        if bank.seeds.len() != POPULATION {
            bail!("TimeSeries42 seed bank changed for {}", spec.dataset_hash);
        }
        let result: DatasetResult = load_json(&final_result_path(&root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) {
            bail!(
                "invalid TimeSeries42 dataset result for {}",
                spec.dataset_hash
            );
        }
        if result.completed_generations == 0
            || result.completed_generations < manifest.config.minimum_generations.max(1)
            || result.completed_generations > manifest.config.generations
            || (!result.strict_winner
                && result.completed_generations != manifest.config.generations)
            || (is_forecast3_gen20(run_id) && result.strict_winner != result.best_beats_official)
            || (is_forecast3_gen20(run_id) && !result.strict_winner && result.status != "exhausted")
        {
            bail!(
                "invalid TimeSeries42 generation accounting for {}",
                spec.dataset_hash
            );
        }
    }
    let prior_scan = root
        .join("terminal-summary.v1.json")
        .is_file()
        .then(|| load_json::<TerminalSummary>(&root.join("terminal-summary.v1.json")).ok())
        .flatten()
        .and_then(|summary| summary.sotarena_scan_output);
    let summary = rebuild_summary(&root, run_id, prior_scan)?;
    if summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("TimeSeries42 terminal summary accounting mismatch");
    }
    println!(
        "{}",
        json!({
            "schema": "JopeTimeseries42Validation.v1",
            "status": "complete",
            "datasets": summary.dataset_count,
            "wins": summary.wins,
            "losses": summary.losses,
            "ties": summary.ties,
            "exhausted": summary.exhausted,
            "valid_evaluations": summary.valid_evaluations,
        })
    );
    Ok(())
}

pub(crate) fn validate_gen30(base_run_id: &str, run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    if base_run_id == run_id {
        bail!("TimeSeries42 gen30 continuation run id must differ from the base run id");
    }
    let base = load_base_campaign(base_run_id)?;
    let root = campaign_root(run_id);
    let manifest_path = root.join("campaign.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    if manifest.schema != CAMPAIGN_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.run_id != run_id
        || manifest.base_run_id.as_deref() != Some(base_run_id)
        || manifest.base_campaign_sha256.as_deref() != Some(&base.campaign_sha256)
        || manifest.continuation_start_generation != Some(GEN30_START_GENERATION)
        || manifest.continuation_terminal_generation != Some(GEN30_GENERATIONS - 1)
        || manifest.config != CampaignConfig::gen30()
        || manifest.datasets != base.manifest.datasets
    {
        bail!("invalid TimeSeries42 gen30 campaign manifest");
    }
    for spec in &manifest.datasets {
        let (_bank, _) = load_existing_seed_bank(&base.root, spec)?;
        for generation in 0..GENERATIONS {
            let slots = load_generation_slots(
                &base.root,
                base_run_id,
                &base.campaign_sha256,
                spec,
                generation,
            )?;
            if slots.len() != POPULATION {
                bail!("invalid base slot count for {}", spec.dataset_hash);
            }
        }
        let base_result: DatasetResult =
            load_json(&final_result_path(&base.root, &spec.dataset_hash))?;
        let result: DatasetResult = load_json(&final_result_path(&root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) {
            bail!(
                "invalid TimeSeries42 gen30 dataset result for {}",
                spec.dataset_hash
            );
        }
        validate_gen30_dataset_result(&result, &base_result, base_run_id)?;
    }
    let prior_scan = root
        .join("terminal-summary.v1.json")
        .is_file()
        .then(|| load_json::<TerminalSummary>(&root.join("terminal-summary.v1.json")).ok())
        .flatten()
        .and_then(|summary| summary.sotarena_scan_output);
    let summary = rebuild_summary(&root, run_id, prior_scan)?;
    if summary.base_run_id.as_deref() != Some(base_run_id)
        || summary.dataset_count != 42
        || summary.wins + summary.losses + summary.ties + summary.exhausted != 42
    {
        bail!("TimeSeries42 gen30 terminal summary accounting mismatch");
    }
    let reports_written = write_timeseries_dataset_reports(
        &manifest,
        &summary,
        &root.join("terminal-summary.v1.json"),
    )?;
    println!(
        "{}",
        json!({
            "schema": "JopeTimeseries42Gen30Validation.v1",
            "status": "complete",
            "base_run_id": base_run_id,
            "run_id": run_id,
            "datasets": summary.dataset_count,
            "wins": summary.wins,
            "losses": summary.losses,
            "ties": summary.ties,
            "exhausted": summary.exhausted,
            "inherited_base_wins": summary.inherited_base_wins.unwrap_or(0),
            "improved_base_wins": summary.improved_base_wins.unwrap_or(0),
            "continuation_strict_wins": summary.continuation_strict_wins.unwrap_or(0),
            "valid_evaluations": summary.valid_evaluations,
            "dataset_reports_written": reports_written,
            "dataset_report_file": JOPE_TIMESERIES_REPORT_FILE,
        })
    );
    Ok(())
}

pub(crate) fn batch_worker(
    manifest_path: &Path,
    output_dir: &Path,
    corpus_eval: &Path,
    timeout_seconds: u64,
) -> Result<()> {
    let started = unix_millis();
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    if !matches!(
        manifest.get("schema").and_then(Value::as_str),
        Some(CORPUS_SHARD_SCHEMA | CORPUS_SHARD_SCHEMA_V2)
    ) || manifest.get("task").and_then(Value::as_str) != Some("timeseries")
    {
        bail!("TimeSeries42 worker manifest is not a corpus_eval time-series shard");
    }
    fs::create_dir_all(output_dir)?;
    let status = Command::new(corpus_eval)
        .arg("--manifest")
        .arg(manifest_path)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--threads")
        .arg(SHARD_SIZE.to_string())
        .arg("--timeout-seconds")
        .arg(timeout_seconds.to_string())
        .status()
        .with_context(|| format!("run corpus_eval {}", corpus_eval.display()))?;
    if !status.success() {
        let failure = json!({
            "schema": "JopeTimeseries42WorkerFailure.v1",
            "manifest_sha256": manifest_sha256,
            "exit_status": status.code(),
            "failed_at_unix_ms": unix_millis(),
        });
        fs::write(
            output_dir.join("batch-worker-failure.v1.json"),
            canonical_json(&failure)?,
        )?;
        bail!("corpus_eval failed with {status}");
    }
    let dataset = manifest
        .get("dataset_hash")
        .and_then(Value::as_str)
        .context("worker manifest lacks dataset_hash")?;
    let receipt_path = output_dir.join(format!("{dataset}.receipts.jsonl"));
    let done_path = output_dir.join(format!("{dataset}.done"));
    let receipt_bytes = fs::read(&receipt_path)?;
    let done_bytes = fs::read(&done_path)?;
    let receipts = read_receipts(&receipt_path)?;
    let worker = WorkerReceipt {
        schema: WORKER_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        dataset_hash: dataset.to_string(),
        manifest_sha256,
        receipt_sha256: sha256_bytes(&receipt_bytes),
        done_sha256: sha256_bytes(&done_bytes),
        rows: receipts.len(),
        valid_rows: receipts.iter().filter(|receipt| receipt.valid).count(),
        started_at_unix_ms: started,
        finished_at_unix_ms: unix_millis(),
    };
    fs::write(
        output_dir.join("batch-worker-receipt.v1.json"),
        canonical_json(&worker)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn leader(model: &str, mase: f64, rmse: f64) -> Value {
        json!({
            "task_type": "timeseries",
            "model_name": model,
            "status": "completed",
            "metrics": {"mase": mase, "rmse": rmse}
        })
    }

    #[test]
    fn official_leader_uses_raw_mase_then_rmse() {
        let dir = tempdir().unwrap();
        let reports = dir.path().join("reports");
        fs::create_dir_all(&reports).unwrap();
        fs::write(
            reports.join("a.json"),
            serde_json::to_vec(&leader("a", 0.8, 10.0)).unwrap(),
        )
        .unwrap();
        fs::write(
            reports.join("b.json"),
            serde_json::to_vec(&leader("b", 0.7, 20.0)).unwrap(),
        )
        .unwrap();
        fs::write(
            reports.join("c.json"),
            serde_json::to_vec(&leader("c", 0.7, 5.0)).unwrap(),
        )
        .unwrap();
        let accepted = external_reports(dir.path(), "dataset").unwrap();
        let best = official_leader(&accepted, dir.path()).unwrap();
        assert_eq!(accepted.len(), 3);
        assert_eq!(best.model, "c");
        assert_eq!(best.mase, 0.7);
    }

    #[test]
    fn four_node_shards_include_xbabe4_and_spec_is_rust_only() {
        let node = StagedNode {
            node: "xbabe4".to_string(),
            hostname: "xbabe4".to_string(),
            bundle_root: "/remote/bundle".to_string(),
            runner_path: "/remote/bundle/jope-dime-regressor".to_string(),
            runner_sha256: "a".repeat(64),
            corpus_eval_path: "/remote/bundle/corpus_eval".to_string(),
            corpus_eval_sha256: "b".repeat(64),
            datasets: Vec::new(),
        };
        let spec = scq_spec(
            &node,
            "job",
            "/remote/manifest.json",
            "/remote/output",
            LEGACY_PROJECT,
        )
        .unwrap();
        assert!(spec.contains("xbabe4"));
        assert!(spec.contains("[[dependencies.smoke]]"));
        assert!(spec.contains("command = \"/usr/bin/test\""));
        assert!(spec.contains("args = [\"-f\", \"/remote/bundle/nodes/xbabe4.ready\"]"));
        assert!(!spec.contains("/etc/hostname"));
        assert!(!spec.contains("command = \"/bin/sh\""));
        assert!(!spec.contains("\nnode = "));
        assert!(spec.contains("timeseries-test-adaptive-batch-worker"));
        assert!(spec.contains("corpus_eval"));
        assert!(!spec.to_ascii_lowercase().contains("python"));
    }

    #[test]
    fn shard_partition_is_stable() {
        assert_eq!(NODES, ["xbabe1", "xbabe2", "xbabe3", "xbabe4"]);
        assert_eq!(SHARD_SIZE, 15);
        assert_eq!(POPULATION, SHARDS * SHARD_SIZE);
    }

    #[test]
    fn wrong_node_marker_failure_is_capacity_not_staging_failure() {
        let wrong_node = anyhow::anyhow!(
            "TimeSeries42 SCQ check failed (exit status: 1): error[dependency-unsatisfied]: dependencies unsatisfied on all eligible nodes\n\
dependency diagnostics:\n  xbabe2: smoke TimeSeries42 staged node xbabe1: smoke test exited with exit 1\n  xbabe3: smoke TimeSeries42 staged node xbabe1: smoke test exited with exit 1"
        );
        assert!(is_scq_capacity_error_for_node(&wrong_node, "xbabe1"));

        let target_node = anyhow::anyhow!(
            "TimeSeries42 SCQ check failed (exit status: 1): error[dependency-unsatisfied]: dependencies unsatisfied on all eligible nodes\n\
dependency diagnostics:\n  xbabe1: smoke TimeSeries42 staged node xbabe1: smoke test exited with exit 1"
        );
        assert!(!is_scq_capacity_error_for_node(&target_node, "xbabe1"));
    }

    #[test]
    fn queued_scq_timeout_is_retryable_capacity_status() {
        let now = Instant::now();
        let timeout = std::time::Duration::from_secs(SCQ_QUEUED_STALL_TIMEOUT_SECONDS);
        let queued_since = now - timeout - std::time::Duration::from_secs(1);
        assert_eq!(
            scq_wait_decision("queued", false, Some(queued_since), now, timeout),
            ScqWaitDecision::CancelQueuedStall
        );
        assert_eq!(
            scq_wait_decision("running", true, Some(queued_since), now, timeout),
            ScqWaitDecision::Wait
        );
        assert_eq!(
            scq_wait_decision("queued", true, Some(queued_since), now, timeout),
            ScqWaitDecision::Wait
        );
        let error = retryable_scq_stall_error("job", "stayed queued without last_start");
        assert!(is_scq_capacity_error_for_node(&error, "xbabe1"));
    }

    #[test]
    fn stale_cancelled_jobs_are_retryable_not_terminal_failures() {
        let error = retryable_scq_stall_error(
            "job",
            "was already cancelled and has no local shard receipt",
        );
        assert!(is_scq_capacity_error(&error));
    }

    #[test]
    fn shard_retry_attempt_ids_stay_unique_across_nodes_and_cycles() {
        let manifest_sha256 = "a".repeat(64);
        let mut ids = BTreeSet::new();
        for attempt in 0..8 {
            let node = NODES[attempt % NODES.len()];
            ids.insert(expected_job_id(
                "run",
                "dataset",
                2,
                1,
                node,
                attempt,
                &manifest_sha256,
            ));
        }
        assert_eq!(ids.len(), 8);
    }

    #[test]
    fn xbabe2_is_treated_as_local_without_ssh_alias() {
        assert!(is_local_scq_node("xbabe2", "controller-alias"));
        assert!(is_local_scq_node("controller-alias", "controller-alias"));
        assert!(!is_local_scq_node("xbabe1", "controller-alias"));
    }

    #[test]
    fn receipt_reads_mase_from_negative_score() {
        let receipt = EvalReceipt {
            task: "timeseries".to_string(),
            dataset_hash: "d".to_string(),
            semantic_sha256: "s".to_string(),
            canonical_genome: "(pipe X (ridge))".to_string(),
            metric: "negative_mase".to_string(),
            score: Some(-0.42),
            secondary_metric: "mase".to_string(),
            secondary_score: Some(0.41),
            valid: true,
            failure: None,
            runtime_seconds: 0.0,
            predictions_sha256: None,
            rmse: Some(1.5),
            protocol_sha256: None,
        };
        let score = receipt_mase(&receipt).unwrap();
        assert_eq!(score.mase, 0.41);
        assert_eq!(score.rmse, 1.5);
    }

    fn test_spec(official_mase: f64) -> DatasetSpec {
        DatasetSpec {
            schema: DATASET_SCHEMA.to_string(),
            dataset_hash: "dataset".to_string(),
            display_name: "Dataset".to_string(),
            category: "test".to_string(),
            train_path: "/tmp/train.csv".to_string(),
            train_sha256: "a".repeat(64),
            test_path: "/tmp/test.csv".to_string(),
            test_sha256: "b".repeat(64),
            meta_path: "/tmp/meta.json".to_string(),
            meta_sha256: "c".repeat(64),
            train_rows: 10,
            test_rows: 5,
            csv_features: 1,
            reduction_features: 4,
            horizon: 1,
            seasonality: 1,
            frequency: "D".to_string(),
            target_is_final_column: true,
            known_future_covariates: Vec::new(),
            past_only_covariates: Vec::new(),
            source_real_data: true,
            synthetic_origin: false,
            reduction_version: String::new(),
            accepted_external_reports: Vec::new(),
            comparator_lock_sha256: String::new(),
            official_leader: OfficialLeader {
                model: "official".to_string(),
                mase: official_mase,
                rmse: 1.0,
                report_path: "/tmp/report.json".to_string(),
                report_sha256: "d".repeat(64),
            },
        }
    }

    fn test_member(semantic: &str, evolved_descendant: bool) -> Member {
        Member {
            canonical_genome: "(pipe X (ridge))".to_string(),
            semantic_sha256: semantic.to_string(),
            complexity: 1,
            rust_source_len: 10,
            evolved_descendant,
            ancestry: AncestryRecord {
                operation: "test".to_string(),
                parents: Vec::new(),
                retry_count: 0,
            },
        }
    }

    #[test]
    fn forecast3_worker_manifest_binds_campaign_binary_and_comparator() {
        let mut spec = test_spec(1.0);
        spec.comparator_lock_sha256 = "e".repeat(64);
        let member = test_member(&"f".repeat(64), false);
        let seed = "1".repeat(64);
        let release = "2".repeat(64);
        let campaign = "3".repeat(64);
        let protocol = "4".repeat(64);
        let manifest = corpus_manifest(
            &spec,
            &[member],
            ("train.csv", "test.csv", "meta.json"),
            label_authority(0, &seed, &release, &campaign),
            Some(&protocol),
            Some((FORECAST3_RUN_ID, &campaign, &release)),
        )
        .unwrap();
        assert_eq!(manifest.run_id.as_deref(), Some(FORECAST3_RUN_ID));
        assert_eq!(manifest.campaign_sha256.as_deref(), Some(campaign.as_str()));
        assert_eq!(
            manifest.release_binary_sha256.as_deref(),
            Some(release.as_str())
        );
        assert_eq!(
            manifest.comparator_lock_sha256.as_deref(),
            Some(spec.comparator_lock_sha256.as_str())
        );
    }

    fn test_slot(
        generation: usize,
        slot: usize,
        mase: f64,
        evolved_descendant: bool,
    ) -> LogicalSlot {
        LogicalSlot {
            generation,
            slot,
            member: test_member(&format!("{generation:02}{slot:062}"), evolved_descendant),
            score: Some(TimeSeriesScore { mase, rmse: 1.0 }),
            valid: true,
            failure: None,
            receipt_file: "/tmp/receipt.jsonl".to_string(),
            receipt_sha256: "e".repeat(64),
            prediction_sha256: Some("f".repeat(64)),
            runtime_seconds: 0.1,
        }
    }

    fn base_result(strict_winner: bool) -> DatasetResult {
        DatasetResult {
            schema: RESULT_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: "base".to_string(),
            base_run_id: None,
            base_status: None,
            base_strict_winner: None,
            base_best_jope_mase: None,
            continuation_status: None,
            best_evidence_run_id: None,
            best_receipt_file: None,
            best_receipt_sha256: None,
            best_prediction_sha256: None,
            campaign_sha256: "a".repeat(64),
            release_binary_sha256: "b".repeat(64),
            dataset_hash: "dataset".to_string(),
            status: if strict_winner { "win" } else { "loss" }.to_string(),
            official_leader_model: "official".to_string(),
            official_leader_mase: 0.5,
            generation_zero_best_mase: Some(0.4),
            best_jope_mase: Some(if strict_winner { 0.35 } else { 0.45 }),
            best_jope_rmse: Some(1.0),
            mase_delta_vs_official: Some(if strict_winner { 0.15 } else { 0.05 }),
            strict_winner,
            best_beats_official: true,
            best_semantic_sha256: Some("s".to_string()),
            best_canonical_genome: Some("(pipe X (ridge))".to_string()),
            best_generation: Some(if strict_winner { 1 } else { 0 }),
            best_slot: Some(0),
            seed_source_family: None,
            logical_evaluations: 120,
            valid_evaluations: 120,
            completed_generations: 2,
            completed_at_unix_ms: 1,
        }
    }

    #[test]
    fn gen30_stop_rule_requires_generation_two_descendant_for_base_losses() {
        let spec = test_spec(0.5);
        let base = base_result(false);
        let generation_zero_best = Some(0.4);
        let mut slots = vec![test_slot(0, 0, 0.39, false)];
        assert!(!continuation_should_stop(
            &base,
            &slots,
            &spec,
            generation_zero_best,
            GEN30_START_GENERATION,
        ));
        slots.push(test_slot(GEN30_START_GENERATION, 1, 0.45, true));
        assert!(!continuation_should_stop(
            &base,
            &slots,
            &spec,
            generation_zero_best,
            GEN30_START_GENERATION,
        ));
        slots.push(test_slot(GEN30_START_GENERATION, 2, 0.35, true));
        assert!(continuation_should_stop(
            &base,
            &slots,
            &spec,
            generation_zero_best,
            GEN30_START_GENERATION,
        ));
    }

    #[test]
    fn gen30_base_wins_chase_until_terminal_generation() {
        let spec = test_spec(0.5);
        let base = base_result(true);
        let slots = vec![test_slot(1, 0, 0.35, true)];
        assert!(!continuation_should_stop(
            &base,
            &slots,
            &spec,
            Some(0.4),
            GEN30_START_GENERATION,
        ));
        assert!(continuation_should_stop(
            &base,
            &slots,
            &spec,
            Some(0.4),
            GEN30_GENERATIONS - 1,
        ));
    }

    #[test]
    fn forecast3_campaign_can_win_at_generation_zero_and_runs_at_most_twenty_generations() {
        let config = base_campaign_config(FORECAST3_RUN_ID);
        assert_eq!(config.generations, FORECAST3_GENERATIONS);
        assert_eq!(config.minimum_generations, 0);

        let spec = test_spec(0.5);
        let seed = test_slot(0, 0, 0.49, false);
        assert!(slot_is_base_campaign_winner(&seed, &spec, Some(0.49), true,));
        assert!(!slot_is_base_campaign_winner(
            &seed,
            &spec,
            Some(0.49),
            false,
        ));
    }

    #[test]
    fn forecast3_non_winners_are_exhausted_after_the_terminal_generation() {
        let loss = TimeSeriesScore {
            mase: 2.0,
            rmse: 3.0,
        };
        let tie = TimeSeriesScore {
            mase: 1.0,
            rmse: 3.0,
        };
        assert_eq!(
            base_campaign_status(FORECAST3_RUN_ID, false, Some(&loss), 1.0),
            "exhausted"
        );
        assert_eq!(
            base_campaign_status(FORECAST3_RUN_ID, false, Some(&tie), 1.0),
            "exhausted"
        );
        assert_eq!(
            base_campaign_status("legacy", false, Some(&loss), 1.0),
            "loss"
        );
        assert_eq!(
            base_campaign_status("legacy", false, Some(&tie), 1.0),
            "tie"
        );
    }

    #[test]
    fn forecast3_win_rule_targets_external_leader_not_generation_zero_seed() {
        let spec = test_spec(0.5);
        let child = test_slot(1, 0, 0.45, true);
        assert!(slot_is_base_campaign_winner(&child, &spec, Some(0.4), true,));
        assert!(!slot_is_base_campaign_winner(
            &child,
            &spec,
            Some(0.4),
            false,
        ));
    }

    #[test]
    fn mase_ranking_minimizes_before_rmse_and_complexity() {
        let mut low_mase = test_slot(2, 0, 0.2, true);
        low_mase.score.as_mut().unwrap().rmse = 100.0;
        let mut high_mase = test_slot(2, 1, 0.3, true);
        high_mase.score.as_mut().unwrap().rmse = 0.1;
        assert_eq!(rank_slots(&low_mase, &high_mase), Ordering::Less);

        let mut low_rmse = test_slot(2, 2, 0.2, true);
        low_rmse.score.as_mut().unwrap().rmse = 1.0;
        assert_eq!(rank_slots(&low_rmse, &low_mase), Ordering::Less);
    }

    fn write_generation_fixture(
        root: &Path,
        run_id: &str,
        campaign_sha256: &str,
        spec: &DatasetSpec,
        generation: usize,
    ) {
        let generation_dir = generation_root(&dataset_root(root, &spec.dataset_hash), generation);
        fs::create_dir_all(generation_dir.join("receipts")).unwrap();
        let members = (0..POPULATION)
            .map(|slot| {
                test_member(
                    &format!("{generation:02}{slot:062}"),
                    generation >= GEN30_START_GENERATION,
                )
            })
            .collect::<Vec<_>>();
        let mut shards = Vec::new();
        for (shard, node) in NODES[..SHARDS].iter().enumerate() {
            let mut receipt_bytes = Vec::new();
            for offset in 0..SHARD_SIZE {
                let slot = shard * SHARD_SIZE + offset;
                let row = json!({
                    "task": "timeseries",
                    "dataset_hash": spec.dataset_hash,
                    "semantic_sha256": members[slot].semantic_sha256,
                    "canonical_genome": members[slot].canonical_genome,
                    "metric": "negative_mase",
                    "score": -0.5,
                    "secondary_metric": "mase",
                    "secondary_score": 0.5,
                    "valid": true,
                    "failure": null,
                    "runtime_seconds": 0.1,
                    "predictions_sha256": "p",
                    "rmse": 1.0
                });
                receipt_bytes.extend_from_slice(serde_json::to_string(&row).unwrap().as_bytes());
                receipt_bytes.push(b'\n');
            }
            let receipt_file = generation_dir
                .join("receipts")
                .join(format!("shard-{shard:02}.receipts.jsonl"));
            fs::write(&receipt_file, &receipt_bytes).unwrap();
            let receipt_sha256 = sha256_bytes(&receipt_bytes);
            shards.push(ShardRecord {
                generation,
                shard,
                node: node.to_string(),
                attempt: 0,
                job_id: format!("job-{shard}"),
                corpus_manifest_file: "/tmp/manifest.json".to_string(),
                corpus_manifest_sha256: "a".repeat(64),
                scq_spec_file: "/tmp/spec.toml".to_string(),
                scq_spec_sha256: "b".repeat(64),
                receipt_file: receipt_file.to_string_lossy().into_owned(),
                receipt_sha256,
                valid_candidates: SHARD_SIZE,
                total_candidates: SHARD_SIZE,
            });
        }
        let manifest = GenerationManifest {
            schema: GENERATION_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            run_id: run_id.to_string(),
            campaign_sha256: campaign_sha256.to_string(),
            release_binary_sha256: "c".repeat(64),
            dataset_hash: spec.dataset_hash.clone(),
            generation,
            members,
            shards,
            best_semantic_sha256: None,
            best_mase: Some(0.5),
            valid_candidates: POPULATION,
            completed_at_unix_ms: 1,
        };
        fs::write(
            generation_dir.join("generation.v1.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn base_generation_loader_rejects_stale_artifacts() {
        let dir = tempdir().unwrap();
        let spec = test_spec(0.5);
        let campaign_sha256 = "a".repeat(64);
        write_generation_fixture(dir.path(), "other-run", &campaign_sha256, &spec, 0);
        let error = load_generation_slots(dir.path(), "base-run", &campaign_sha256, &spec, 0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("stale or invalid"));
    }

    #[test]
    fn timeseries_dataset_report_is_benchmark_invalid_and_mase_authoritative() {
        let spec = test_spec(0.5);
        let mut result = base_result(false);
        result.run_id = "gen30".to_string();
        result.base_run_id = Some("base".to_string());
        result.continuation_status = Some("gen30_strict_win".to_string());
        result.status = "win".to_string();
        result.strict_winner = true;
        result.best_jope_mase = Some(0.3);
        result.best_jope_rmse = Some(2.0);
        result.mase_delta_vs_official = Some(0.2);
        result.best_receipt_file = Some("/tmp/best.receipts.jsonl".to_string());
        result.best_receipt_sha256 = Some("e".repeat(64));
        let report = timeseries_report_value(
            &spec,
            &result,
            Path::new("/tmp/terminal-summary.v1.json"),
            &"f".repeat(64),
        );
        assert_eq!(
            report["report_schema_version"],
            JOPE_TIMESERIES_REPORT_SCHEMA
        );
        assert_eq!(report["algorithm"], "jope-dime-regressor");
        assert_eq!(report["task_type"], "timeseries");
        assert_eq!(report["benchmark_valid"], false);
        assert_eq!(report["test_adaptive"], true);
        assert_eq!(report["source_path"], "/tmp/best.receipts.jsonl");
        assert_eq!(report["metrics"]["mase"], 0.3);
        assert_eq!(report["metrics"]["negative_mase"], -0.3);
        assert_eq!(
            report["comparison"]["continuation_status"],
            "gen30_strict_win"
        );
        assert_eq!(report["provenance"]["base_run_id"], "base");
    }
}
