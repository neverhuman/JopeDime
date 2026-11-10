//! Same-rows comparator evidence attached to a finished run.
//!
//! Every comparator is fitted on exactly the champion's FIT rows and scored on exactly its
//! HOLDOUT rows with the run objective. The champion is measured the same way and reported in the
//! same shape, so a reader can compare rows directly. Comparators that time out or fail stay in
//! the report with their status; they are never dropped.

use crate::runs::Language;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// `ComparisonReport` is a struct for the same-rows comparator evidence contract in this module:
/// the champion and every requested comparator measured on identical FIT and HOLDOUT rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ComparisonReport {
    /// `medium`, `high` or `ultra`.
    pub tier: String,
    /// The run objective, for example `rmse`.
    pub metric: String,
    /// The score scope shared by every row, for example `holdout`.
    pub scope: String,
    /// Digest of the FIT partition every model was fitted on.
    pub fit_rows_sha256: String,
    /// Digest of the HOLDOUT partition every model was scored on.
    pub holdout_rows_sha256: String,
    /// The run champion, measured the same way as the comparators.
    pub champion: ComparedModel,
    /// Requested comparators in roster order, including timed-out and failed ones.
    pub comparators: Vec<ComparedModel>,
    pub value: ValueProposition,
    pub budget: ComparatorBudgetUsed,
}

/// `ComparedModel` is a struct for the per-model comparison row contract in this module: one
/// model's identity, outcome and measured cost on the shared rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ComparedModel {
    pub id: String,
    pub label: String,
    /// `jope_champion` or `rust_port_analogue`.
    pub implementation: String,
    pub status: ComparedStatus,
    pub holdout: Option<f64>,
    pub cv: Option<f64>,
    pub fit_seconds: Option<f64>,
    pub predict_us_per_row: Option<f64>,
    pub model_bytes: Option<u64>,
}

/// `ComparedStatus` is an enum for the comparison row outcome contract in this module: whether
/// a model completed, hit the stage ceiling, or failed with a reason.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ComparedStatus {
    Completed,
    TimedOut,
    Failed { reason: String },
}

/// `ValueProposition` is a struct for the champion-versus-alternatives summary contract in this
/// module: where the champion ranks among completed models and by how much. Positive deltas mean
/// the champion is better.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ValueProposition {
    /// The champion's 1-based rank among completed models, champion included.
    pub rank: u32,
    /// Completed models ranked, champion included.
    pub entrants: u32,
    /// The best completed comparator's id.
    pub best_alternative: Option<String>,
    pub holdout_delta_pct_vs_best: Option<f64>,
    pub holdout_delta_pct_vs_median: Option<f64>,
    pub beats: u32,
    pub loses_to: u32,
    /// Best alternative's predict time divided by the champion's, when both were measured.
    pub predict_speedup_vs_best: Option<f64>,
    /// Champion model bytes divided by the best alternative's, when both were measured.
    pub size_ratio_vs_best: Option<f64>,
    /// Languages the champion exports to as readable source; comparators export none.
    pub export_languages: Vec<Language>,
    pub deterministic_replay: bool,
}

/// `ComparatorBudgetUsed` is a struct for the comparator stage cost contract in this module: the
/// wall time spent against the stage ceiling and how many comparators hit it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ComparatorBudgetUsed {
    pub wall_seconds: f64,
    pub ceiling_seconds: f64,
    pub timed_out: u32,
}
