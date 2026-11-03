//! Frozen held-out-test-adaptive multiclass campaign contract.
//!
//! This lane is deliberately exploratory. Test macro-F1 is evolutionary
//! fitness, so every artifact is benchmark-invalid and excluded from official
//! combined rankings.

use super::*;
use feat_math::advanced_v3::validate_v3_genome;
use feat_math::dsl::{ParamValue, Stage};
use feat_math::{ColumnSpec, DataSchema, GenomeIr, Objective, RowSemantics, TaskSpec};
use jope_dime_regressor::paths;
use serde_json::json;

pub(crate) const RUN_ID: &str = "multiclass272-test-adaptive-union43-gen20-v2";
const PLAN_SCHEMA: &str = "JopeMulticlass272TestAdaptivePlan.v2";
const LEDGER_SCHEMA: &str = "JopeMulticlass60ChampionLedger.v2";
const CORPUS_SCHEMA: &str = "multiclass-union-runnable-v5";
const COMPARATOR_SCHEMA: &str = "test-adaptive-comparator-lock-v2";
const COMPARATOR_SET_SCHEMA: &str = "test-adaptive-comparator-set.v2";
const COMPARATOR_SET_FILE: &str = "comparator-set.v2.json";
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
jope_paths::once_str!(
    fn data_root_str => paths::quant_root().join("multi_label")
);
jope_paths::once_str!(
    fn scq_str => paths::super_cluster_scq_binary()
);
jope_paths::once_str!(
    fn corpus_eval_str => paths::corpus_eval_binary()
);
const PROJECT: &str = "jope-multiclass-test-adaptive";
const JOPE_MULTICLASS_REPORT_FILE: &str = "jope-dime-test-adaptive-union43-gen20-v2.json";
const POPULATION: usize = 60;
const SHARD_SIZES: [usize; 4] = [15, 15, 15, 15];
const NODES: [&str; 4] = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"];
const PRIORITY_DATASETS: usize = 272;
// Frozen into the existing v1 plan. Capacity sharing with TimeSeries42 is
// enforced by SCQ admission, not by mutating this immutable identity.
const ACTIVE_DATASETS: usize = 5;
// Shards are often sub-second on small datasets, while a complete SCQ
// reconcile/harvest pass takes several seconds.  Keep multiple waves visible
// to SCQ so its global core and memory accounting, rather than controller
// latency, controls the number that actually run at once.
const QUEUE_BUFFER_DATASETS: usize = 64;
const SPEC_REVISION: usize = 2;
const REPLAY_SPEC_REVISION: usize = 3;
const REPLAY_ROLE_CAPABILITIES: [(&str, &str); 3] = [
    (
        "xbabe1",
        "/etc/jope/capabilities/multiclass-replay-slot-a.v1",
    ),
    (
        "xbabe2",
        "/etc/jope/capabilities/multiclass-replay-slot-b.v1",
    ),
    (
        "xbabe3",
        "/etc/jope/capabilities/multiclass-replay-slot-c.v1",
    ),
];
const MIN_GENERATIONS: usize = 0;
const MAX_GENERATIONS: usize = 20;
const MAX_SCHEDULER_PREEMPTIONS: usize = 64;
const MULTICLASS_HEADS: [&str; 4] = [
    "multiclass_ridge",
    "multiclass_ovr_logistic",
    "multiclass_diagonal_lda",
    "multiclass_nearest_centroid",
];
const FEATURE_BUDGETS: [usize; 4] = [32, 64, 128, 256];
const FAST_BASELINES: [&str; 3] = [
    "mc20_hist_gradient_boosting",
    "mc20_extra_trees",
    "mc20_random_forest",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Config {
    population: usize,
    elites: usize,
    parent_pool: usize,
    tournament: usize,
    mutation_probability: f64,
    crossover_probability: f64,
    minimum_generations: usize,
    maximum_generations: usize,
    shard_sizes: Vec<usize>,
    nodes: Vec<String>,
    active_datasets: usize,
    candidate_threads: usize,
    candidate_timeout_seconds: usize,
    infrastructure_retries: usize,
    replay_nodes: usize,
    ranking: String,
    win_rule: String,
}

impl Config {
    fn fixed() -> Self {
        Self {
            population: POPULATION,
            elites: 7,
            parent_pool: 30,
            tournament: 3,
            mutation_probability: 0.75,
            crossover_probability: 0.25,
            minimum_generations: MIN_GENERATIONS,
            maximum_generations: MAX_GENERATIONS,
            shard_sizes: SHARD_SIZES.to_vec(),
            nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            active_datasets: ACTIVE_DATASETS,
            candidate_threads: 1,
            candidate_timeout_seconds: 180,
            infrastructure_retries: 1,
            replay_nodes: 3,
            ranking: "test macro_f1 descending, test accuracy descending, complexity ascending, semantic_sha256 ascending".to_string(),
            win_rule: "stop after the first confirmed strict beat of the frozen 43-model comparator, including generation 0; exact 1.0 matches an exact 1.0 incumbent as ceiling_matched; otherwise publish exhausted after generation 19".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SeedMember {
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Incumbent {
    method: String,
    macro_f1: f64,
    accuracy: f64,
    report_path: String,
    report_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetBinding {
    order: usize,
    wave: String,
    dataset_hash: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    meta_path: String,
    meta_sha256: String,
    class_count: usize,
    n_features: usize,
    fast_baselines_complete: bool,
    completed_incumbent_count: usize,
    comparator_sha256: String,
    strongest_available_incumbent: Option<Incumbent>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    run_id: String,
    test_adaptive: bool,
    selection_bias: String,
    benchmark_valid: bool,
    official_combined_rank_eligible: bool,
    project: String,
    project_weight: usize,
    project_max_gpus: usize,
    scq_binary: String,
    scq_binary_sha256: String,
    release_binary_sha256: String,
    corpus_eval: String,
    corpus_eval_sha256: String,
    corpus_manifest: String,
    corpus_manifest_sha256: String,
    multiclass_coverage_final_receipt_sha256: String,
    timeseries_coverage_manifest_sha256: String,
    timeseries_coverage_final_receipt_sha256: String,
    comparator_root: String,
    comparator_set_path: String,
    comparator_set_sha256: String,
    seed_catalog: String,
    seed_catalog_sha256: String,
    seed_population_sha256: String,
    priority_count: usize,
    execution_order: Vec<String>,
    fast_baselines: Vec<String>,
    config: Config,
    seeds: Vec<SeedMember>,
    datasets: Vec<DatasetBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CoverageGate {
    multiclass_receipt_sha256: String,
    timeseries_receipt_sha256: String,
    comparator_set_path: String,
    comparator_set_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct RankedScore {
    pub macro_f1: f64,
    pub accuracy: f64,
    pub complexity: usize,
    pub semantic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LogicalSlot {
    slot: usize,
    member: SeedMember,
    score: Option<RankedScore>,
    valid: bool,
    failure: Option<String>,
    receipt_sha256: String,
    predictions_sha256: Option<String>,
    probabilities_sha256: Option<String>,
    node: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GenerationCheckpoint {
    schema: String,
    run_id: String,
    dataset_hash: String,
    comparator_sha256: String,
    generation: usize,
    test_adaptive: bool,
    selection_bias: String,
    benchmark_valid: bool,
    slots: Vec<LogicalSlot>,
    best: Option<RankedScore>,
    best_member: Option<SeedMember>,
    initial_budget_complete: bool,
    continuation_lineage_salt: Option<String>,
    next_population: Vec<SeedMember>,
    completed_at_unix_ms: u128,
}

const CHECKPOINT_SCHEMA: &str = "JopeMulticlass272GenerationCheckpoint.v1";

pub(crate) fn rank_order(left: &RankedScore, right: &RankedScore) -> Ordering {
    right
        .macro_f1
        .total_cmp(&left.macro_f1)
        .then_with(|| right.accuracy.total_cmp(&left.accuracy))
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

pub(crate) fn qualifies(
    candidate: &RankedScore,
    generation: usize,
    is_seed: bool,
    generation_zero_best: f64,
    incumbent: f64,
    final_twenty_complete: bool,
) -> bool {
    let _ = (is_seed, generation_zero_best, final_twenty_complete);
    if !candidate.macro_f1.is_finite() {
        return false;
    }
    let _ = generation;
    candidate.macro_f1 > incumbent || (incumbent == 1.0 && candidate.macro_f1 == 1.0)
}

mod evolution;
mod ledger;
mod plan;
mod replay;
mod runtime;
mod validation;

use self::plan as plan_module;
use self::plan::{
    multiclass_head_family, read_plan, read_value, ready_dataset_bindings, required_hash,
    rewrite_seed, run_root,
};
use evolution::{
    checkpoint_path, load_checkpoint, next_population, slot_rank_order, slots_from_receipts,
};
use ledger::{
    comparator_protocol_sha256, dataset_terminal, fetch_shard_receipt, ledger_path,
    read_receipt_values, render_spec, scheduler_preemption, scq_rows, shard_job_id, shard_manifest,
    spec_job_id, terminal_infrastructure_state, write_multiclass_terminal_report,
};
#[cfg(test)]
use ledger::{initial_budget_complete, next_generation_index};
use replay::{manage_winner_replay, write_exhausted_ledger};

pub(crate) fn plan(
    run_id: &str,
    campaign_root: &Path,
    corpus_manifest: &Path,
    comparator_root: &Path,
    seed_catalog: &Path,
) -> Result<()> {
    plan_module::plan(
        run_id,
        campaign_root,
        corpus_manifest,
        comparator_root,
        seed_catalog,
    )
}

pub(crate) fn run_all(run_id: &str, campaign_root: &Path, resume: bool) -> Result<()> {
    runtime::run_all(run_id, campaign_root, resume)
}

pub(crate) fn validate_all(run_id: &str, campaign_root: &Path) -> Result<()> {
    validation::validate_all(run_id, campaign_root)
}

#[cfg(test)]
mod tests;
