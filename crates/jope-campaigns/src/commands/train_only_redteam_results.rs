use super::*;

const FINAL_METHOD_ID: &str = "jope-dime-train-only-redteam-v1";
const GEN9_VARIANT: &str = "existing-v2-generation-9-control";
const G20_RAW_VARIANT: &str = "raw-generation-19-champion";
const G20_OOF_VARIANT: &str = "generation-0-19-train-only-oof-selection";
const SMART_SINGLE_VARIANT: &str = "smart-v3-single-pipeline";
const SMART_NO_JITTER_VARIANT: &str = "smart-v3-no-jitter-counterfactual";
const SMART_ORIGINAL_SEED_VARIANT: &str = "smart-v3-original-decoder-seed-only-counterfactual";
const FINAL_VARIANT: &str = "smart-v3-final";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DatasetValidationShard {
    schema: String,
    run_id: String,
    manifest_sha256: String,
    owner_node: String,
    replay: execution::DatasetValidationReplay,
    #[serde(default)]
    scq_execution: Option<ScqExecutionProvenance>,
    baseline_paths_accessed: bool,
    test_paths_accessed: bool,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GlobalValidationReceipt {
    schema: String,
    run_id: String,
    manifest_sha256: String,
    g20_arm_sha256: String,
    smart_arm_sha256: String,
    selector_cohort_sha256: String,
    dataset_validation_sha256: BTreeMap<String, String>,
    #[serde(default)]
    scq_execution: BTreeMap<String, ScqExecutionProvenance>,
    every_selection_receipt_validated: bool,
    baseline_paths_accessed: bool,
    test_paths_accessed: bool,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BaselineRow {
    prior_train_only_rmse: f64,
    tabpfn_v3_rmse: f64,
    stronger_tabfm_method_id: String,
    stronger_tabfm_rmse: f64,
    oracle_aspfm_rmse: f64,
    prior_train_only_source_sha256: String,
    tabpfn_v3_source_sha256: String,
    stronger_tabfm_source_sha256: String,
    oracle_source_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FrozenTestScore {
    variant: String,
    semantic_sha256: Vec<String>,
    weights: Vec<f64>,
    rmse: f64,
    predictions_sha256: String,
    full_data_refit_parity: bool,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DatasetScoreReceipt {
    schema: String,
    run_id: String,
    dataset_hash: String,
    validation_sha256: String,
    selector_sha256: String,
    train_sha256: String,
    test_sha256: String,
    baseline_manifest_sha256: String,
    oracle_index_sha256: String,
    variants: Vec<FrozenTestScore>,
    baselines: BaselineRow,
    selection_frozen_before_test_access: bool,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ScoreIndex {
    schema: String,
    run_id: String,
    validation_sha256: String,
    baseline_manifest_sha256: String,
    oracle_index_sha256: String,
    dataset_score_sha256: BTreeMap<String, String>,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WinCounts {
    tabpfn_v3: usize,
    stronger_tabfm: usize,
    both_tabpfn_and_stronger_tabfm: usize,
    prior_train_only: usize,
    generation_9_control: usize,
    oracle_aspfm_envelope: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReportDataset {
    dataset_hash: String,
    final_rmse: f64,
    generation_9_control_rmse: f64,
    prior_train_only_rmse: f64,
    tabpfn_v3_rmse: f64,
    stronger_tabfm_rmse: f64,
    oracle_aspfm_rmse: f64,
    final_variant: String,
    ensemble_accepted: bool,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StudyReport {
    schema: String,
    run_id: String,
    validation_sha256: String,
    score_index_sha256: String,
    fair_ladder: Vec<EloRating>,
    oracle_reference_ladder: Vec<EloRating>,
    wins: WinCounts,
    median_final_rmse: f64,
    median_generation_9_control_rmse: f64,
    development_strategy_improved: bool,
    oracle_comparison_determines_success: bool,
    authoritative_method_labels: BTreeMap<String, String>,
    datasets: Vec<ReportDataset>,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SmartV3TestRow {
    dataset_hash: String,
    final_variant: String,
    assigned_scq_node: String,
    selector_sha256: String,
    train_sha256: String,
    test_sha256: String,
    test_rmse: f64,
    test_mae: f64,
    test_r2: f64,
    previous_train_only_rmse: f64,
    relative_rmse_change: f64,
    tabpfn_v3_rmse: f64,
    relative_rmse_change_tabpfn_v3: f64,
    stronger_tabfm_method_id: String,
    stronger_tabfm_rmse: f64,
    relative_rmse_change_stronger_tabfm: f64,
    prior_comparison: String,
    tabpfn_v3_comparison: String,
    stronger_tabfm_comparison: String,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SmartV3Aggregate {
    wins: usize,
    ties: usize,
    losses: usize,
    median_relative_rmse_change: f64,
    geometric_mean_rmse_ratio: f64,
    minimum_wins_required: usize,
    smart_v3_improved: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SmartV3BaselineAggregate {
    wins: usize,
    ties: usize,
    losses: usize,
    median_relative_rmse_change: f64,
    geometric_mean_rmse_ratio: f64,
    minimum_wins_required: usize,
    smart_v3_beats_baseline: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SmartV3AggregateComparisons {
    previous_train_only: SmartV3BaselineAggregate,
    tabpfn_v3: SmartV3BaselineAggregate,
    stronger_tabfm: SmartV3BaselineAggregate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SmartV3TestResults {
    schema: String,
    run_id: String,
    manifest_sha256: String,
    baseline_manifest_sha256: String,
    selector_count: usize,
    test_result_rows: Vec<SmartV3TestRow>,
    aggregate: SmartV3Aggregate,
    baseline_comparisons: SmartV3AggregateComparisons,
    selection_frozen_before_test_access: bool,
    test_paths_opened_only_by_this_command: bool,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
}

struct SmartComparisonBaselines {
    prior_train_only_rmse: f64,
    tabpfn_v3_rmse: f64,
    stronger_tabfm_method_id: String,
    stronger_tabfm_rmse: f64,
}

#[path = "train_only_redteam_results/report.rs"]
mod report;
#[path = "train_only_redteam_results/scoring.rs"]
mod scoring;
#[path = "train_only_redteam_results/smart_v3.rs"]
mod smart_v3;
#[cfg(test)]
#[path = "train_only_redteam_results/tests.rs"]
mod tests;
#[path = "train_only_redteam_results/validation.rs"]
mod validation;

pub(super) use report::report_run;
pub(super) use scoring::score_run;
#[cfg(test)]
pub(super) use smart_v3::regression_metrics;
pub(super) use smart_v3::score_smart_v3_run;
#[cfg(test)]
use smart_v3::{smart_v3_aggregate, smart_v3_baseline_aggregate};
pub(super) use validation::{selector_path, validate_run, validation_shard_path};
