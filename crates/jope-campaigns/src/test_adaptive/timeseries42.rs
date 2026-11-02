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

include!("timeseries42/dataset_config.rs");

include!("timeseries42/campaign_records.rs");

include!("timeseries42/dataset_io.rs");

include!("timeseries42/coverage_binding.rs");

include!("timeseries42/candidate_sources.rs");

include!("timeseries42/seed_bank.rs");

include!("timeseries42/remote_nodes.rs");

include!("timeseries42/staging.rs");

include!("timeseries42/scq_jobs.rs");

include!("timeseries42/shard_execution.rs");

include!("timeseries42/generation_evaluation.rs");

include!("timeseries42/dataset_results.rs");

include!("timeseries42/continuation.rs");

include!("timeseries42/campaign_initialization.rs");

include!("timeseries42/reporting.rs");

include!("timeseries42/run_base.rs");

include!("timeseries42/run_continuations.rs");

include!("timeseries42/validation.rs");

#[cfg(test)]
mod tests;
