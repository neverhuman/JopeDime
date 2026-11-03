//! Portable regression objectives. Configuration is data, never executable code.
//!
//! A specification is a strict JSON document (`schemas/jope.regression-fitness.v1.schema.json`):
//!
//! - a **built-in** metric with explicit parameters (`{"metric":"pinball","quantile":0.9}`);
//! - a **resource** metric (`{"metric":"resource","name":"fit_seconds"}`) reading one
//!   [`ResourceVar`] of the candidate's fit or prediction pass;
//! - a **custom** bounded row expression over `y`, `pred`, named numeric *auxiliary columns*
//!   (prices, unit costs, exposures) and [`ResourceVar`] scalars followed by an aggregation;
//! - a **weighted** form of another specification (`weights` is itself a row expression, so an
//!   exposure column, a notional or `abs(y)` can weight rows);
//! - a positive, scaled **composite** of specifications;
//! - a **constrained** specification (`{"metric":"constrained", ...}`) that scores by an inner
//!   objective only when every [`ResourceLimit`] holds and fails the candidate otherwise.
//!
//! Auxiliary columns are read from the training table, excluded from the model's features and
//! used only to score held-out predictions. Every parameter, default and expression is part of the
//! serialized objective identity ([`Fitness::identity`]).
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub mod resources;
pub use resources::{Infeasible, ResourceLimit, ResourceVar, Resources};

pub const FITNESS_VERSION: &str = "jope.regression-fitness/v1";
pub const MAX_SPEC_BYTES: usize = 32 * 1024;
/// Expression nodes in one specification, summed over every custom expression and weight, so
/// the cost of scoring a row stays bounded however the expressions are composed.
pub const MAX_EXPRESSION_NODES: usize = 256;
/// Most distinct auxiliary columns one specification may read.
pub const MAX_COLUMNS: usize = 16;
const MAX_COLUMN_NAME_BYTES: usize = 256;

/// Named auxiliary columns aligned with the targets being scored.
pub type Columns = BTreeMap<String, Vec<f64>>;

/// Complete schema shared by tooling and MCP discovery. Runtime validation also bounds recursion.
pub fn schema() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../schemas/jope.regression-fitness.v1.schema.json"
    ))
    .expect("committed fitness schema")
}

pub use jope_api::Direction;
/// Compatibility path; all runtime objective directions share the API enum.
pub type FitnessDirection = Direction;

/// Every parameter is part of the serialized objective identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "metric", rename_all = "snake_case", deny_unknown_fields)]
pub enum Fitness {
    Rmse {},
    Mse {},
    Mae {},
    MedianAbsoluteError {},
    MaxError {},
    Mape {
        epsilon: f64,
    },
    Smape {
        epsilon: f64,
    },
    Wape {},
    Msle {},
    Rmsle {},
    R2 {},
    ExplainedVariance {},
    Pearson {},
    Spearman {},
    Huber {
        delta: f64,
    },
    PseudoHuber {
        delta: f64,
    },
    LogCosh {},
    Pinball {
        quantile: f64,
    },
    Expectile {
        quantile: f64,
    },
    PoissonDeviance {},
    GammaDeviance {},
    TweedieDeviance {
        power: f64,
    },
    /// `b·(exp(a·e) − a·e − 1)` with `e = y − pred` and `b = 1`: a > 0 makes under-prediction
    /// exponentially costly, a < 0 over-prediction.
    Linex {
        a: f64,
    },
    /// `under_cost·max(y−pred,0) + over_cost·max(pred−y,0)`: the newsvendor cost in target units.
    AsymmetricAbsolute {
        under_cost: f64,
        over_cost: f64,
    },
    /// The squared form of [`Fitness::AsymmetricAbsolute`].
    AsymmetricSquared {
        under_cost: f64,
        over_cost: f64,
    },
    /// Root mean squared percentage error with denominator floor `epsilon`.
    Rmspe {
        epsilon: f64,
    },
    /// MAE divided by the MAE of the seasonal naive forecast `y[t−season]` on the same scored,
    /// chronologically ordered rows.
    Mase {
        season: usize,
    },
    /// y is the realized simple return; clipped prediction is a signed position.
    MeanReturn {
        max_position: f64,
        cost: f64,
    },
    Sharpe {
        max_position: f64,
        cost: f64,
        target: f64,
        periods_per_year: f64,
    },
    Sortino {
        max_position: f64,
        cost: f64,
        target: f64,
        periods_per_year: f64,
    },
    /// Largest peak-to-trough loss of the compounded equity curve, as a fraction in `[0,1]`.
    MaxDrawdown {
        max_position: f64,
        cost: f64,
    },
    /// Compound annual growth rate divided by the maximum drawdown.
    Calmar {
        max_position: f64,
        cost: f64,
        periods_per_year: f64,
    },
    /// Sum of net returns above `threshold` divided by the sum of shortfalls below it.
    Omega {
        max_position: f64,
        cost: f64,
        threshold: f64,
    },
    /// Mean loss over the worst `alpha` fraction of net returns (CVaR), reported as a positive loss.
    ExpectedShortfall {
        max_position: f64,
        cost: f64,
        alpha: f64,
    },
    /// Share of nonzero targets whose sign the prediction gets right (a zero prediction misses).
    HitRate {},
    /// Lorenz-curve Gini of targets ranked by prediction, divided by the Gini of the perfect
    /// ranking; with weights it is the exposure-weighted actuarial Gini.
    NormalizedGini {},
    /// Kling–Gupta efficiency: `1 − sqrt((r−1)² + (σp/σy−1)² + (μp/μy−1)²)`.
    Kge {},
    /// Share of the total target captured by the top `fraction` of rows ranked by prediction.
    CaptureAt {
        fraction: f64,
    },
    /// One resource reading as a metric in its own right (for [`Fitness::Composite`] terms or
    /// pure cost objectives). Throughput is maximized; every other resource is minimized.
    Resource {
        name: ResourceVar,
    },
    /// Wall-clock time in seconds spent fitting the candidate model (`>= 0`).
    TrainTimeSeconds {},
    /// Wall-clock time in seconds spent running candidate inference (`>= 0`).
    InferenceTimeSeconds {},
    /// Combined training and inference wall-clock time in seconds (`>= 0`).
    TotalTimeSeconds {},
    /// Per-row inference latency in microseconds (`>= 0`).
    InferenceUsPerRow {},
    /// Training throughput in rows per second (`>= 0`, maximize for faster training).
    TrainRowsPerSecond {},
    /// Inference throughput in rows per second (`>= 0`, maximize for faster inference).
    InferenceRowsPerSecond {},
    /// Peak/working heap memory in bytes used during candidate training (`>= 0`).
    TrainMemoryBytes {},
    /// Peak/working heap memory in bytes used during candidate inference (`>= 0`).
    InferenceMemoryBytes {},
    /// Combined training and inference memory footprint in bytes (`>= 0`).
    TotalMemoryBytes {},
    /// Fitted model artifact footprint in bytes (`>= 0`).
    ModelBytes {},
    /// Pipeline stage count of the candidate (`>= 0`).
    ComplexityStages {},
    /// Fitted scalar parameter/node count of the candidate (`>= 0`).
    ComplexityParameters {},
    /// Active input feature count consumed by the candidate (`>= 0`).
    FeatureCount {},
    /// Standard deviation of loss across cross-validation folds or regime cuts (`>= 0`).
    FoldStd {},
    /// Mean absolute sequential change `mean(|pred[t] - pred[t-1]|)` (`>= 0`).
    PredictionTurnover {},
    Custom {
        name: String,
        direction: Direction,
        expression: Expression,
        aggregation: RegressionAggregation,
    },
    /// Another specification with nonnegative per-row weights (an expression, usually a column).
    Weighted {
        weights: Expression,
        fitness: Box<Fitness>,
    },
    /// Terms are normalized to minimization before the positive weighted sum.
    Composite {
        terms: Vec<Term>,
    },
    /// Scores by `fitness` when every resource limit is met; fails with [`Infeasible`] otherwise.
    Constrained {
        fitness: Box<Fitness>,
        limits: Vec<ResourceLimit>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Term {
    pub weight: f64,
    pub scale: f64,
    pub fitness: Fitness,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Expression {
    Y {},
    Pred {},
    Constant {
        value: f64,
    },
    /// A numeric auxiliary column of the training table, aligned with the scored rows.
    Column {
        name: String,
    },
    /// A scalar resource reading of the candidate, broadcast to every scored row.
    Resource {
        name: ResourceVar,
    },
    /// Candidate training time in seconds (`>= 0`).
    #[serde(alias = "train_time")]
    TrainTimeSeconds {},
    /// Candidate inference time in seconds (`>= 0`).
    #[serde(alias = "inference_time")]
    InferenceTimeSeconds {},
    /// Candidate combined training + inference time in seconds (`>= 0`).
    #[serde(alias = "total_time")]
    TotalTimeSeconds {},
    /// Candidate per-row inference latency in microseconds (`>= 0`).
    InferenceUsPerRow {},
    /// Candidate training throughput in rows per second (`>= 0`).
    TrainRowsPerSecond {},
    /// Candidate inference throughput in rows per second (`>= 0`).
    InferenceRowsPerSecond {},
    /// Candidate training memory footprint in bytes (`>= 0`).
    #[serde(alias = "train_memory")]
    TrainMemoryBytes {},
    /// Candidate inference memory footprint in bytes (`>= 0`).
    #[serde(alias = "inference_memory")]
    InferenceMemoryBytes {},
    /// Candidate combined training + inference memory footprint in bytes (`>= 0`).
    #[serde(alias = "total_memory")]
    TotalMemoryBytes {},
    /// Candidate fitted model footprint in bytes (`>= 0`).
    ModelBytes {},
    /// Candidate pipeline stage count (`>= 0`).
    ComplexityStages {},
    /// Candidate fitted parameter count (`>= 0`).
    ComplexityParameters {},
    /// Candidate active input feature count (`>= 0`).
    FeatureCount {},
    /// Cross-fold loss standard deviation (`>= 0`).
    FoldStd {},
    /// Mean sequential prediction turnover `mean(|pred[t] - pred[t-1]|)` (`>= 0`).
    PredictionTurnover {},
    /// Zero-based row index `0..n-1`.
    RowIndex {},
    /// Normalized row position `row / max(n - 1, 1)` in `[0, 1]` for regime recency weighting.
    RowFraction {},
    /// Number of scored rows `n`.
    RowCount {},
    /// Signed residual `y - pred`.
    Residual {},
    /// Absolute residual `|y - pred|`.
    AbsError {},
    /// Squared residual `(y - pred)^2`.
    SquaredError {},
    /// Mean of target `y` across scored rows.
    YMean {},
    /// Population standard deviation of target `y` across scored rows.
    YStd {},
    /// Mean of prediction `pred` across scored rows.
    PredMean {},
    /// Population standard deviation of prediction `pred` across scored rows.
    PredStd {},
    Add {
        left: Box<Self>,
        right: Box<Self>,
    },
    Sub {
        left: Box<Self>,
        right: Box<Self>,
    },
    Mul {
        left: Box<Self>,
        right: Box<Self>,
    },
    Div {
        left: Box<Self>,
        right: Box<Self>,
    },
    Min {
        left: Box<Self>,
        right: Box<Self>,
    },
    Max {
        left: Box<Self>,
        right: Box<Self>,
    },
    /// `left` raised to `right`; a non-finite result fails the candidate.
    Pow {
        left: Box<Self>,
        right: Box<Self>,
    },
    Abs {
        arg: Box<Self>,
    },
    Neg {
        arg: Box<Self>,
    },
    Square {
        arg: Box<Self>,
    },
    Sqrt {
        arg: Box<Self>,
    },
    Ln {
        arg: Box<Self>,
    },
    Log1p {
        arg: Box<Self>,
    },
    Exp {
        arg: Box<Self>,
    },
    /// Sign of `arg` (`-1`, `0`, or `1`).
    Sign {
        arg: Box<Self>,
    },
    /// Hyperbolic tangent `tanh(arg)`.
    Tanh {
        arg: Box<Self>,
    },
    /// Logistic sigmoid `1 / (1 + exp(-arg))`.
    Sigmoid {
        arg: Box<Self>,
    },
    /// Cube root `cbrt(arg)`.
    Cbrt {
        arg: Box<Self>,
    },
    /// 1 when `arg > 0`, otherwise 0.
    Step {
        arg: Box<Self>,
    },
    /// `arg` limited to `[min, max]`.
    Clip {
        arg: Box<Self>,
        min: f64,
        max: f64,
    },
    /// `then` when `condition > 0`, otherwise `else`. Only the chosen branch is evaluated.
    Where {
        condition: Box<Self>,
        then: Box<Self>,
        #[serde(rename = "else")]
        otherwise: Box<Self>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RegressionAggregation {
    Mean {},
    Sum {},
    Min {},
    Max {},
    Median {},
    Std {},
    Variance {},
    Quantile {
        quantile: f64,
    },
    /// Mean of the largest values holding `fraction` of the rows (or weight): CVaR of a row loss.
    TailMean {
        fraction: f64,
    },
    /// Square root of the mean, for RMSE-like custom losses.
    RootMean {},
}

/// Backwards-compatible name for regression-objective aggregation.
pub type Aggregation = RegressionAggregation;

#[derive(Clone, Debug, Serialize)]
pub struct Score {
    pub value: f64,
    /// The selection machinery always minimizes this value.
    pub loss: f64,
    pub direction: Direction,
    pub fitness_sha256: String,
}

/// One catalog entry: its short name, default specification and a one-line description.
pub struct CatalogEntry {
    pub name: &'static str,
    pub fitness: Fitness,
    pub description: &'static str,
}

mod catalog;
mod evaluate;
mod math;
use math::*;
pub use evaluate::{require_no_columns, require_no_resources, ExecutionTelemetry};

#[cfg(test)]
mod tests;
