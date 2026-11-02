//! Portable regression objectives. Configuration is data, never executable code.
//!
//! A specification is a strict JSON document (`schemas/jope.regression-fitness.v1.schema.json`):
//!
//! - a **built-in** metric with explicit parameters (`{"metric":"pinball","quantile":0.9}`);
//! - a **custom** bounded row expression over `y`, `pred` and named numeric *auxiliary columns*
//!   (prices, unit costs, exposures) followed by an aggregation;
//! - a **weighted** form of another specification (`weights` is itself a row expression, so an
//!   exposure column, a notional or `abs(y)` can weight rows);
//! - a positive, scaled **composite** of specifications.
//!
//! Auxiliary columns are read from the training table, excluded from the model's features and
//! used only to score held-out predictions. Every parameter, default and expression is part of the
//! serialized objective identity ([`Fitness::identity`]).
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

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
    Max {},
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

fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}
fn probability(x: f64) -> bool {
    x.is_finite() && x > 0.0 && x < 1.0
}
fn finite(x: f64) -> Result<f64> {
    ensure!(x.is_finite(), "non-finite fitness value");
    Ok(x)
}
fn mean(v: &[f64]) -> f64 {
    v.iter().map(|x| x / v.len() as f64).sum()
}
/// Weighted mean; weights are validated nonnegative with a positive finite total.
fn wmean(v: &[f64], w: Option<&[f64]>) -> f64 {
    match w {
        None => mean(v),
        Some(w) => {
            let total: f64 = w.iter().sum();
            v.iter().zip(w).map(|(x, w)| x * (w / total)).sum()
        }
    }
}
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    let p = q * (v.len() - 1) as f64;
    let i = p.floor() as usize;
    v[i] + (v[p.ceil() as usize] - v[i]) * (p - i as f64)
}
/// Linear interpolation at `q·(n−1)` over the sorted values. Weighted, each value sits at the
/// share of weight below it, scaled so the largest value sits at 1: equal weights give exactly the
/// unweighted quantile. Zero-weight rows are ignored.
fn wquantile(v: &[f64], w: Option<&[f64]>, q: f64) -> Result<f64> {
    let Some(w) = w else {
        return Ok(quantile(v, q));
    };
    let mut order: Vec<usize> = (0..v.len()).filter(|&i| w[i] > 0.0).collect();
    ensure!(!order.is_empty(), "every row has zero weight");
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]).then(a.cmp(&b)));
    let last = order.len() - 1;
    let span: f64 = order[..last].iter().map(|&i| w[i]).sum();
    if last == 0 || q <= 0.0 {
        return Ok(v[order[0]]);
    }
    let mut below = 0.0;
    let positions: Vec<f64> = order
        .iter()
        .map(|&i| {
            let p = below / span;
            below += w[i];
            p
        })
        .collect();
    let k = positions.partition_point(|&p| p < q).clamp(1, last);
    let (p0, p1) = (positions[k - 1], positions[k]);
    let (v0, v1) = (v[order[k - 1]], v[order[k]]);
    Ok(v0 + (v1 - v0) * ((q - p0) / (p1 - p0)).min(1.0))
}
/// Mean of the largest values that hold `fraction` of the total weight (each row weighs 1 when
/// unweighted); the boundary row contributes the part of its weight that is needed.
fn tail_mean(v: &[f64], w: Option<&[f64]>, fraction: f64) -> Result<f64> {
    let weight = |i: usize| w.map_or(1.0, |w| w[i]);
    let total: f64 = (0..v.len()).map(weight).sum();
    let mass = fraction * total;
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    let (mut taken, mut sum) = (0.0, 0.0);
    for i in order {
        let take = weight(i).min(mass - taken);
        if take <= 0.0 {
            if weight(i) > 0.0 {
                break;
            }
            continue;
        }
        sum += take * v[i];
        taken += take;
    }
    ensure!(taken > 0.0, "tail holds no weight");
    finite(sum / taken)
}
fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    mean(&v.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>())
}
fn wvariance(v: &[f64], w: Option<&[f64]>) -> f64 {
    let m = wmean(v, w);
    wmean(&v.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>(), w)
}
fn correlation(a: &[f64], b: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let am = wmean(a, w);
    let bm = wmean(b, w);
    let av = wvariance(a, w);
    let bv = wvariance(b, w);
    ensure!(
        av.is_finite() && bv.is_finite(),
        "correlation overflows: the variance of a vector is not finite"
    );
    ensure!(
        av > 0.0 && bv > 0.0,
        "correlation undefined for a constant vector"
    );
    finite(
        wmean(
            &a.iter()
                .zip(b)
                .map(|(x, y)| (x - am) * (y - bm))
                .collect::<Vec<_>>(),
            w,
        ) / av.sqrt()
            / bv.sqrt(),
    )
}
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite inputs"));
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && v[order[i]] == v[order[j]] {
            j += 1;
        }
        for k in i..j {
            out[order[k]] = (i + j - 1) as f64 / 2.0;
        }
        i = j;
    }
    out
}
/// `(weight, target)` in descending order of `key`. A group of tied keys shares its weighted mean
/// target, so a tie is one straight segment of a Lorenz or capture curve (unweighted, the expected
/// curve over every order of the tied rows): the rows' file order never counts as ranking skill.
fn ranked(y: &[f64], key: &[f64], w: Option<&[f64]>) -> Vec<(f64, f64)> {
    let weight = |i: usize| w.map_or(1.0, |w| w[i]);
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| key[b].total_cmp(&key[a]));
    let mut out = Vec::with_capacity(order.len());
    for group in order.chunk_by(|&a, &b| key[a] == key[b]) {
        let total: f64 = group.iter().map(|&i| weight(i)).sum();
        let target = if total > 0.0 {
            group.iter().map(|&i| weight(i) * y[i]).sum::<f64>() / total
        } else {
            group.iter().map(|&i| y[i]).sum::<f64>() / group.len() as f64
        };
        out.extend(group.iter().map(|&i| (weight(i), target)));
    }
    out
}
/// Twice the trapezoid area under the cumulative-weight vs cumulative-(w·y) curve, minus one.
fn lorenz_gini(ranked: &[(f64, f64)]) -> Result<f64> {
    let total_w: f64 = ranked.iter().map(|(w, _)| w).sum();
    let total_y: f64 = ranked.iter().map(|(w, y)| w * y).sum();
    ensure!(
        total_w > 0.0 && total_y.is_finite() && total_y != 0.0,
        "Gini needs a nonzero weighted target total"
    );
    let (mut x0, mut y0, mut area) = (0.0, 0.0, 0.0);
    for &(w, y) in ranked {
        let x1 = x0 + w / total_w;
        let y1 = y0 + w * y / total_y;
        area += (x1 - x0) * (y0 + y1) / 2.0;
        (x0, y0) = (x1, y1);
    }
    finite(2.0 * area - 1.0)
}
/// Net per-interval returns of the documented position convention.
fn strategy_returns(y: &[f64], p: &[f64], max_position: f64, cost: f64) -> Result<Vec<f64>> {
    let mut last = 0.0;
    let returns: Vec<_> = y
        .iter()
        .zip(p)
        .map(|(y, p)| {
            let position = p.clamp(-max_position, max_position);
            let r = position * y - cost * (position - last).abs();
            last = position;
            r
        })
        .collect();
    ensure!(
        returns.iter().all(|x| x.is_finite()),
        "non-finite strategy return"
    );
    Ok(returns)
}
/// Compounded equity curve from 1; a period losing everything ends the curve at zero. An
/// overflowing curve is an error, never a finite score.
fn equity_curve(returns: &[f64]) -> Result<Vec<f64>> {
    let mut equity = 1.0f64;
    let mut curve = Vec::with_capacity(returns.len());
    for r in returns {
        let next = equity * (1.0 + r);
        ensure!(next.is_finite(), "the equity curve overflowed");
        equity = next.max(0.0);
        curve.push(equity);
    }
    Ok(curve)
}
fn max_drawdown(returns: &[f64]) -> Result<f64> {
    let mut peak = 1.0f64;
    let mut worst = 0.0f64;
    for e in equity_curve(returns)? {
        peak = peak.max(e);
        worst = worst.max(1.0 - e / peak);
    }
    Ok(worst)
}
fn validate_column_name(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.len() <= MAX_COLUMN_NAME_BYTES
            && !name.chars().any(char::is_control),
        "column names need 1..{MAX_COLUMN_NAME_BYTES} bytes without control characters"
    );
    Ok(())
}

impl Expression {
    fn validate(&self, depth: usize, nodes: &mut usize) -> Result<()> {
        *nodes += 1;
        ensure!(
            depth <= 16 && *nodes <= MAX_EXPRESSION_NODES,
            "fitness expressions exceed depth 16 or {MAX_EXPRESSION_NODES} nodes in one \
             specification"
        );
        match self {
            Self::Constant { value } => {
                finite(*value)?;
            }
            Self::Column { name } => validate_column_name(name)?,
            Self::Y {} | Self::Pred {} => (),
            Self::Abs { arg }
            | Self::Neg { arg }
            | Self::Square { arg }
            | Self::Sqrt { arg }
            | Self::Ln { arg }
            | Self::Log1p { arg }
            | Self::Exp { arg }
            | Self::Step { arg } => arg.validate(depth + 1, nodes)?,
            Self::Clip { arg, min, max } => {
                ensure!(
                    min.is_finite() && max.is_finite() && min <= max,
                    "clip needs finite min <= max"
                );
                arg.validate(depth + 1, nodes)?;
            }
            Self::Add { left, right }
            | Self::Sub { left, right }
            | Self::Mul { left, right }
            | Self::Div { left, right }
            | Self::Min { left, right }
            | Self::Max { left, right }
            | Self::Pow { left, right } => {
                left.validate(depth + 1, nodes)?;
                right.validate(depth + 1, nodes)?;
            }
            Self::Where {
                condition,
                then,
                otherwise,
            } => {
                condition.validate(depth + 1, nodes)?;
                then.validate(depth + 1, nodes)?;
                otherwise.validate(depth + 1, nodes)?;
            }
        }
        Ok(())
    }
    fn collect_columns(&self, out: &mut BTreeSet<String>) {
        match self {
            Self::Column { name } => {
                out.insert(name.clone());
            }
            Self::Y {} | Self::Pred {} | Self::Constant { .. } => (),
            Self::Abs { arg }
            | Self::Neg { arg }
            | Self::Square { arg }
            | Self::Sqrt { arg }
            | Self::Ln { arg }
            | Self::Log1p { arg }
            | Self::Exp { arg }
            | Self::Step { arg }
            | Self::Clip { arg, .. } => arg.collect_columns(out),
            Self::Add { left, right }
            | Self::Sub { left, right }
            | Self::Mul { left, right }
            | Self::Div { left, right }
            | Self::Min { left, right }
            | Self::Max { left, right }
            | Self::Pow { left, right } => {
                left.collect_columns(out);
                right.collect_columns(out);
            }
            Self::Where {
                condition,
                then,
                otherwise,
            } => {
                condition.collect_columns(out);
                then.collect_columns(out);
                otherwise.collect_columns(out);
            }
        }
    }
    fn eval(&self, y: f64, p: f64, row: usize, columns: &Columns) -> Result<f64> {
        let e = |x: &Self| x.eval(y, p, row, columns);
        finite(match self {
            Self::Y {} => y,
            Self::Pred {} => p,
            Self::Constant { value } => *value,
            Self::Column { name } => *columns
                .get(name)
                .and_then(|values| values.get(row))
                .ok_or_else(|| anyhow::anyhow!("fitness column {name:?} is not available"))?,
            Self::Abs { arg } => e(arg)?.abs(),
            Self::Neg { arg } => -e(arg)?,
            Self::Square { arg } => e(arg)?.powi(2),
            Self::Sqrt { arg } => e(arg)?.sqrt(),
            Self::Ln { arg } => e(arg)?.ln(),
            Self::Log1p { arg } => e(arg)?.ln_1p(),
            Self::Exp { arg } => e(arg)?.exp(),
            Self::Step { arg } => {
                if e(arg)? > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Clip { arg, min, max } => e(arg)?.clamp(*min, *max),
            Self::Add { left, right } => e(left)? + e(right)?,
            Self::Sub { left, right } => e(left)? - e(right)?,
            Self::Mul { left, right } => e(left)? * e(right)?,
            Self::Div { left, right } => e(left)? / e(right)?,
            Self::Min { left, right } => e(left)?.min(e(right)?),
            Self::Max { left, right } => e(left)?.max(e(right)?),
            Self::Pow { left, right } => e(left)?.powf(e(right)?),
            Self::Where {
                condition,
                then,
                otherwise,
            } => {
                if e(condition)? > 0.0 {
                    e(then)?
                } else {
                    e(otherwise)?
                }
            }
        })
    }
    fn rows(&self, y: &[f64], p: &[f64], columns: &Columns) -> Result<Vec<f64>> {
        y.iter()
            .zip(p)
            .enumerate()
            .map(|(row, (&y, &p))| self.eval(y, p, row, columns))
            .collect()
    }
}

/// One catalog entry: its short name, default specification and a one-line description.
pub struct CatalogEntry {
    pub name: &'static str,
    pub fitness: Fitness,
    pub description: &'static str,
}

mod catalog;
mod evaluate;

#[cfg(test)]
mod tests;
