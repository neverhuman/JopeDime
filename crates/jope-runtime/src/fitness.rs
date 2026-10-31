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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Minimize,
    Maximize,
}

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
        aggregation: Aggregation,
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
pub enum Aggregation {
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

impl Fitness {
    /// A catalog name, `name:key=value,…` overriding a catalog default's numeric parameters
    /// (`pinball:quantile=0.9`), or a strict inline JSON specification. File access belongs to
    /// the CLI.
    pub fn parse(text: &str) -> Result<Self> {
        ensure!(
            text.len() <= MAX_SPEC_BYTES,
            "fitness specification is too large"
        );
        let spec = if text.trim_start().starts_with('{') {
            serde_json::from_str(text)?
        } else {
            let (name, overrides) = match text.trim().split_once(':') {
                Some((name, overrides)) => (name, Some(overrides)),
                None => (text.trim(), None),
            };
            let base = Self::catalog()
                .into_iter()
                .find(|entry| entry.name == name)
                .map(|entry| entry.fitness)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown fitness {name:?}; use `jopedime fitness list`")
                })?;
            match overrides {
                None => base,
                Some(overrides) => {
                    let mut value = serde_json::to_value(&base)?;
                    let object = value.as_object_mut().expect("tagged specification");
                    for pair in overrides.split(',').filter(|p| !p.trim().is_empty()) {
                        let (key, raw) = pair
                            .split_once('=')
                            .ok_or_else(|| anyhow::anyhow!("override {pair:?} is not key=value"))?;
                        let key = key.trim();
                        ensure!(
                            key != "metric" && object.contains_key(key),
                            "{name} has no parameter {key:?}"
                        );
                        let number: serde_json::Value = serde_json::from_str(raw.trim())
                            .ok()
                            .filter(serde_json::Value::is_number)
                            .ok_or_else(|| {
                                anyhow::anyhow!("parameter {key} needs a number, got {raw:?}")
                            })?;
                        object.insert(key.to_owned(), number);
                    }
                    serde_json::from_value(value)?
                }
            }
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Short names with their exact default parameters.
    pub fn catalog() -> Vec<CatalogEntry> {
        let strategy = |metric: &str| -> Fitness {
            let (max_position, cost) = (1.0, 0.0);
            match metric {
                "mean_return" => Self::MeanReturn { max_position, cost },
                "sharpe" => Self::Sharpe {
                    max_position,
                    cost,
                    target: 0.0,
                    periods_per_year: 252.0,
                },
                "sortino" => Self::Sortino {
                    max_position,
                    cost,
                    target: 0.0,
                    periods_per_year: 252.0,
                },
                "max_drawdown" => Self::MaxDrawdown { max_position, cost },
                "calmar" => Self::Calmar {
                    max_position,
                    cost,
                    periods_per_year: 252.0,
                },
                "omega" => Self::Omega {
                    max_position,
                    cost,
                    threshold: 0.0,
                },
                _ => Self::ExpectedShortfall {
                    max_position,
                    cost,
                    alpha: 0.05,
                },
            }
        };
        let entry = |name, fitness, description| CatalogEntry {
            name,
            fitness,
            description,
        };
        vec![
            entry(
                "rmse",
                Self::Rmse {},
                "Root mean squared error; emphasizes large misses.",
            ),
            entry("mse", Self::Mse {}, "Mean squared error."),
            entry(
                "mae",
                Self::Mae {},
                "Mean absolute error; linear cost per unit missed.",
            ),
            entry(
                "median_absolute_error",
                Self::MedianAbsoluteError {},
                "Median absolute error; ignores the worst half of misses.",
            ),
            entry(
                "max_error",
                Self::MaxError {},
                "Worst absolute error on scored rows.",
            ),
            entry(
                "mape",
                Self::Mape { epsilon: 1e-8 },
                "Mean absolute percentage error (ratio) with a denominator floor.",
            ),
            entry(
                "smape",
                Self::Smape { epsilon: 1e-8 },
                "Symmetric mean absolute percentage error (ratio, 0..2).",
            ),
            entry(
                "wape",
                Self::Wape {},
                "Total absolute error over total absolute target volume.",
            ),
            entry(
                "msle",
                Self::Msle {},
                "Mean squared log1p error; nonnegative values.",
            ),
            entry(
                "rmsle",
                Self::Rmsle {},
                "Root mean squared log1p error; nonnegative values.",
            ),
            entry(
                "r2",
                Self::R2 {},
                "Coefficient of determination (maximize).",
            ),
            entry(
                "explained_variance",
                Self::ExplainedVariance {},
                "Explained variance; ignores a constant bias (maximize).",
            ),
            entry(
                "pearson",
                Self::Pearson {},
                "Pearson correlation (maximize).",
            ),
            entry(
                "spearman",
                Self::Spearman {},
                "Rank correlation, the information coefficient (maximize).",
            ),
            entry(
                "huber",
                Self::Huber { delta: 1.0 },
                "Huber loss: quadratic inside delta.",
            ),
            entry(
                "pseudo_huber",
                Self::PseudoHuber { delta: 1.0 },
                "Smooth Huber loss.",
            ),
            entry(
                "log_cosh",
                Self::LogCosh {},
                "log(cosh(error)); smooth and robust.",
            ),
            entry(
                "pinball",
                Self::Pinball { quantile: 0.5 },
                "Quantile (pinball) loss; quantile=0.9 makes shortages 9x as costly.",
            ),
            entry(
                "expectile",
                Self::Expectile { quantile: 0.5 },
                "Asymmetric squared error.",
            ),
            entry(
                "poisson_deviance",
                Self::PoissonDeviance {},
                "Poisson deviance for counts; y>=0, pred>0.",
            ),
            entry(
                "gamma_deviance",
                Self::GammaDeviance {},
                "Gamma deviance for positive severities.",
            ),
            entry(
                "tweedie_deviance",
                Self::TweedieDeviance { power: 1.5 },
                "Tweedie deviance for zero-inflated positive costs.",
            ),
            entry(
                "linex",
                Self::Linex { a: 1.0 },
                "Linear-exponential loss; a>0 punishes under-prediction exponentially.",
            ),
            entry(
                "asymmetric_absolute",
                Self::AsymmetricAbsolute {
                    under_cost: 1.0,
                    over_cost: 1.0,
                },
                "Newsvendor cost: under_cost per unit short, over_cost per unit over.",
            ),
            entry(
                "asymmetric_squared",
                Self::AsymmetricSquared {
                    under_cost: 1.0,
                    over_cost: 1.0,
                },
                "Squared newsvendor cost.",
            ),
            entry(
                "rmspe",
                Self::Rmspe { epsilon: 1e-8 },
                "Root mean squared percentage error (ratio).",
            ),
            entry(
                "mase",
                Self::Mase { season: 1 },
                "MAE relative to the seasonal naive forecast; chronological rows.",
            ),
            entry(
                "mean_return",
                strategy("mean_return"),
                "Mean net strategy return per interval (maximize).",
            ),
            entry(
                "sharpe",
                strategy("sharpe"),
                "Annualized Sharpe ratio of net strategy returns (maximize).",
            ),
            entry(
                "sortino",
                strategy("sortino"),
                "Annualized Sortino ratio; penalizes only downside (maximize).",
            ),
            entry(
                "max_drawdown",
                strategy("max_drawdown"),
                "Largest peak-to-trough loss of compounded equity.",
            ),
            entry(
                "calmar",
                strategy("calmar"),
                "Compound annual growth over maximum drawdown (maximize).",
            ),
            entry(
                "omega",
                strategy("omega"),
                "Gains above threshold over shortfalls below it (maximize).",
            ),
            entry(
                "expected_shortfall",
                strategy("expected_shortfall"),
                "Mean loss of the worst alpha fraction of net returns (CVaR).",
            ),
            entry(
                "hit_rate",
                Self::HitRate {},
                "Share of nonzero targets whose sign is predicted (maximize).",
            ),
            entry(
                "normalized_gini",
                Self::NormalizedGini {},
                "Ranking quality: Gini over perfect Gini (maximize).",
            ),
            entry(
                "kge",
                Self::Kge {},
                "Kling-Gupta efficiency for hydrology (maximize).",
            ),
            entry(
                "capture_at",
                Self::CaptureAt { fraction: 0.1 },
                "Share of total target in the top-ranked fraction (maximize).",
            ),
        ]
    }

    pub fn name(&self) -> String {
        match self {
            Self::Custom { name, .. } => name.clone(),
            Self::Weighted { fitness, .. } => format!("weighted_{}", fitness.name()),
            Self::Composite { .. } => "composite".to_owned(),
            other => serde_json::to_value(other)
                .ok()
                .and_then(|v| v["metric"].as_str().map(str::to_owned))
                .expect("tagged built-in"),
        }
    }
    pub fn direction(&self) -> Direction {
        match self {
            Self::R2 {}
            | Self::ExplainedVariance {}
            | Self::Pearson {}
            | Self::Spearman {}
            | Self::MeanReturn { .. }
            | Self::Sharpe { .. }
            | Self::Sortino { .. }
            | Self::Calmar { .. }
            | Self::Omega { .. }
            | Self::HitRate {}
            | Self::NormalizedGini {}
            | Self::Kge {}
            | Self::CaptureAt { .. } => Direction::Maximize,
            Self::Custom { direction, .. } => *direction,
            Self::Weighted { fitness, .. } => fitness.direction(),
            _ => Direction::Minimize,
        }
    }
    /// Whether scoring depends on row order, so it needs chronological forward validation.
    pub fn requires_order(&self) -> bool {
        match self {
            Self::MeanReturn { .. }
            | Self::Sharpe { .. }
            | Self::Sortino { .. }
            | Self::MaxDrawdown { .. }
            | Self::Calmar { .. }
            | Self::Omega { .. }
            | Self::ExpectedShortfall { .. }
            | Self::Mase { .. } => true,
            Self::Weighted { fitness, .. } => fitness.requires_order(),
            Self::Composite { terms } => terms.iter().any(|t| t.fitness.requires_order()),
            _ => false,
        }
    }
    /// Auxiliary columns the specification reads (custom expressions and weights).
    pub fn columns(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.collect_columns(&mut out);
        out
    }
    fn collect_columns(&self, out: &mut BTreeSet<String>) {
        match self {
            Self::Custom { expression, .. } => expression.collect_columns(out),
            Self::Weighted { weights, fitness } => {
                weights.collect_columns(out);
                fitness.collect_columns(out);
            }
            Self::Composite { terms } => {
                for term in terms {
                    term.fitness.collect_columns(out);
                }
            }
            _ => (),
        }
    }
    /// Whether rows may carry weights here.
    fn supports_weights(&self) -> bool {
        match self {
            Self::MaxError {}
            | Self::Spearman {}
            | Self::Mase { .. }
            | Self::MeanReturn { .. }
            | Self::Sharpe { .. }
            | Self::Sortino { .. }
            | Self::MaxDrawdown { .. }
            | Self::Calmar { .. }
            | Self::Omega { .. }
            | Self::ExpectedShortfall { .. }
            | Self::Kge {}
            | Self::CaptureAt { .. }
            | Self::Weighted { .. } => false,
            Self::Custom { aggregation, .. } => !matches!(aggregation, Aggregation::Max {}),
            Self::Composite { terms } => terms.iter().all(|t| t.fitness.supports_weights()),
            _ => true,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.validate_at(0, &mut 0, &mut 0, false)?;
        let columns = self.columns();
        ensure!(
            columns.len() <= MAX_COLUMNS,
            "a fitness specification may read at most {MAX_COLUMNS} columns"
        );
        ensure!(
            serde_json::to_vec(self)?.len() <= MAX_SPEC_BYTES,
            "fitness specification is too large"
        );
        Ok(())
    }
    fn validate_at(
        &self,
        depth: usize,
        nodes: &mut usize,
        expression_nodes: &mut usize,
        weighted: bool,
    ) -> Result<()> {
        *nodes += 1;
        ensure!(
            depth <= 8 && *nodes <= 64,
            "fitness composition exceeds depth/node limit"
        );
        if weighted {
            ensure!(
                self.supports_weights(),
                "{} does not accept row weights",
                self.name()
            );
        }
        let strategy = |max_position: f64, cost: f64| {
            ensure!(
                positive(max_position) && cost.is_finite() && cost >= 0.0,
                "position must be positive; cost must be nonnegative; both finite"
            );
            Ok(())
        };
        match self {
            Self::Mape { epsilon } | Self::Smape { epsilon } | Self::Rmspe { epsilon } => {
                ensure!(positive(*epsilon), "epsilon must be positive and finite")
            }
            Self::Huber { delta } | Self::PseudoHuber { delta } => {
                ensure!(positive(*delta), "delta must be positive and finite")
            }
            Self::Pinball { quantile } | Self::Expectile { quantile } => {
                ensure!(probability(*quantile), "quantile must be in (0,1)")
            }
            Self::TweedieDeviance { power } => ensure!(
                power.is_finite() && *power > 1.0 && *power < 2.0,
                "Tweedie power must be in (1,2); use Poisson/Gamma at endpoints"
            ),
            Self::Linex { a } => ensure!(
                a.is_finite() && *a != 0.0,
                "LinEx a must be finite and nonzero"
            ),
            Self::AsymmetricAbsolute {
                under_cost,
                over_cost,
            }
            | Self::AsymmetricSquared {
                under_cost,
                over_cost,
            } => ensure!(
                positive(*under_cost) && positive(*over_cost),
                "under_cost and over_cost must be positive and finite"
            ),
            Self::Mase { season } => {
                ensure!((1..=10_000).contains(season), "season must be in 1..=10000")
            }
            Self::MeanReturn { max_position, cost } | Self::MaxDrawdown { max_position, cost } => {
                strategy(*max_position, *cost)?
            }
            Self::Sharpe {
                max_position,
                cost,
                target,
                periods_per_year,
            }
            | Self::Sortino {
                max_position,
                cost,
                target,
                periods_per_year,
            } => {
                strategy(*max_position, *cost)?;
                ensure!(
                    target.is_finite() && positive(*periods_per_year),
                    "target must be finite and periods_per_year positive"
                );
            }
            Self::Calmar {
                max_position,
                cost,
                periods_per_year,
            } => {
                strategy(*max_position, *cost)?;
                ensure!(
                    positive(*periods_per_year),
                    "periods_per_year must be positive"
                );
            }
            Self::Omega {
                max_position,
                cost,
                threshold,
            } => {
                strategy(*max_position, *cost)?;
                ensure!(threshold.is_finite(), "threshold must be finite");
            }
            Self::ExpectedShortfall {
                max_position,
                cost,
                alpha,
            } => {
                strategy(*max_position, *cost)?;
                ensure!(probability(*alpha), "alpha must be in (0,1)");
            }
            Self::CaptureAt { fraction } => ensure!(
                fraction.is_finite() && *fraction > 0.0 && *fraction <= 1.0,
                "fraction must be in (0,1]"
            ),
            Self::Custom {
                name,
                expression,
                aggregation,
                ..
            } => {
                ensure!(
                    !name.is_empty()
                        && name.len() <= 64
                        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_'),
                    "custom name needs 1..64 ASCII letters, digits or underscores"
                );
                expression.validate(0, expression_nodes)?;
                match aggregation {
                    Aggregation::Quantile { quantile } => ensure!(
                        quantile.is_finite() && (0.0..=1.0).contains(quantile),
                        "aggregation quantile must be in [0,1]"
                    ),
                    Aggregation::TailMean { fraction } => ensure!(
                        fraction.is_finite() && *fraction > 0.0 && *fraction <= 1.0,
                        "tail_mean fraction must be in (0,1]"
                    ),
                    _ => (),
                }
            }
            Self::Weighted { weights, fitness } => {
                ensure!(!weighted, "weighting cannot be nested");
                weights.validate(0, expression_nodes)?;
                fitness.validate_at(depth + 1, nodes, expression_nodes, true)?;
            }
            Self::Composite { terms } => {
                ensure!(
                    !terms.is_empty() && terms.len() <= 16,
                    "composite needs 1..16 terms"
                );
                for t in terms {
                    ensure!(
                        positive(t.weight) && positive(t.scale),
                        "term weight and scale must be positive and finite"
                    );
                    t.fitness
                        .validate_at(depth + 1, nodes, expression_nodes, weighted)?;
                }
            }
            _ => (),
        }
        Ok(())
    }
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        // Compact JSON, except that −0.0 is written as 0.0: they score identically, so they are
        // one objective. Every other specification hashes exactly as `serde_json::to_vec`.
        struct Canonical;
        impl serde_json::ser::Formatter for Canonical {
            fn write_f64<W: ?Sized + std::io::Write>(
                &mut self,
                writer: &mut W,
                value: f64,
            ) -> std::io::Result<()> {
                let value = if value == 0.0 { 0.0 } else { value };
                serde_json::ser::CompactFormatter.write_f64(writer, value)
            }
        }
        let mut bytes = Vec::new();
        serde::Serialize::serialize(
            &(FITNESS_VERSION, self),
            &mut serde_json::Serializer::with_formatter(&mut bytes, Canonical),
        )?;
        Ok(crate::sha256_hex(&bytes))
    }
    /// Score predictions of a specification that reads no auxiliary column.
    pub fn evaluate(&self, y: &[f64], pred: &[f64]) -> Result<Score> {
        self.evaluate_with(y, pred, &Columns::new())
    }
    /// Score predictions; `columns` holds every auxiliary column the specification reads,
    /// aligned with `y`.
    pub fn evaluate_with(&self, y: &[f64], pred: &[f64], columns: &Columns) -> Result<Score> {
        self.validate()?;
        let value = self.checked_value(y, pred, columns)?;
        let direction = self.direction();
        Ok(Score {
            value,
            loss: if direction == Direction::Maximize {
                -value
            } else {
                value
            },
            direction,
            fitness_sha256: self.identity()?,
        })
    }
    /// The minimization loss of an already validated specification, without hashing it. The
    /// calibration search calls this many times on one validated objective.
    pub fn loss_unhashed(&self, y: &[f64], pred: &[f64], columns: &Columns) -> Result<f64> {
        let value = self.checked_value(y, pred, columns)?;
        Ok(if self.direction() == Direction::Maximize {
            -value
        } else {
            value
        })
    }
    fn checked_value(&self, y: &[f64], pred: &[f64], columns: &Columns) -> Result<f64> {
        ensure!(
            !y.is_empty() && y.len() == pred.len(),
            "fitness needs equal, nonempty target and prediction vectors"
        );
        ensure!(
            y.iter().chain(pred).all(|x| x.is_finite()),
            "fitness inputs must be finite"
        );
        for name in self.columns() {
            let values = columns
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("fitness column {name:?} was not supplied"))?;
            ensure!(
                values.len() == y.len() && values.iter().all(|v| v.is_finite()),
                "fitness column {name:?} needs one finite value per scored row"
            );
        }
        finite(self.value(y, pred, columns, None)?)
    }
    fn value(&self, y: &[f64], p: &[f64], columns: &Columns, w: Option<&[f64]>) -> Result<f64> {
        if w.is_some() {
            ensure!(
                self.supports_weights(),
                "{} does not accept row weights",
                self.name()
            );
        }
        let errors: Vec<_> = y.iter().zip(p).map(|(y, p)| y - p).collect();
        if !matches!(
            self,
            Self::Custom { .. }
                | Self::Weighted { .. }
                | Self::Composite { .. }
                | Self::MeanReturn { .. }
                | Self::Sharpe { .. }
                | Self::Sortino { .. }
                | Self::MaxDrawdown { .. }
                | Self::Calmar { .. }
                | Self::Omega { .. }
                | Self::ExpectedShortfall { .. }
                | Self::Pearson {}
                | Self::Spearman {}
                | Self::HitRate {}
                | Self::NormalizedGini {}
                | Self::Kge {}
                | Self::CaptureAt { .. }
        ) {
            ensure!(
                errors.iter().all(|e| e.is_finite()),
                "non-finite regression residual"
            );
        }
        let squared = || wmean(&errors.iter().map(|e| e * e).collect::<Vec<_>>(), w);
        let abs: Vec<_> = errors.iter().map(|e| e.abs()).collect();
        let row_loss = |f: &dyn Fn(f64, f64) -> f64| {
            wmean(
                &y.iter().zip(p).map(|(&y, &p)| f(y, p)).collect::<Vec<_>>(),
                w,
            )
        };
        let error_loss =
            |f: &dyn Fn(f64) -> f64| wmean(&errors.iter().map(|&e| f(e)).collect::<Vec<_>>(), w);
        Ok(match self {
            Self::Rmse {} => squared().sqrt(),
            Self::Mse {} => squared(),
            Self::Mae {} => wmean(&abs, w),
            Self::MedianAbsoluteError {} => wquantile(&abs, w, 0.5)?,
            Self::MaxError {} => abs.iter().copied().fold(0.0, f64::max),
            Self::Mape { epsilon } => row_loss(&|y, p| (y - p).abs() / y.abs().max(*epsilon)),
            Self::Smape { epsilon } => row_loss(&|y, p| {
                // Normalize before adding or doubling: finite same-sign inputs
                // can overflow the unscaled denominator and hide real error.
                let scale = y.abs().max(p.abs()).max(*epsilon);
                let denominator = (y.abs() / scale + p.abs() / scale).max(epsilon / scale);
                ((y - p).abs() / scale) * (2.0 / denominator)
            }),
            Self::Rmspe { epsilon } => {
                row_loss(&|y, p| ((y - p) / y.abs().max(*epsilon)).powi(2)).sqrt()
            }
            Self::Wape {} => {
                let den = wmean(&y.iter().map(|y| y.abs()).collect::<Vec<_>>(), w);
                ensure!(den > 0.0, "WAPE undefined for all-zero targets");
                wmean(&abs, w) / den
            }
            Self::Msle {} | Self::Rmsle {} => {
                ensure!(
                    y.iter().chain(p).all(|x| *x >= 0.0),
                    "log error needs nonnegative targets and predictions"
                );
                let v = row_loss(&|y, p| (y.ln_1p() - p.ln_1p()).powi(2));
                if matches!(self, Self::Rmsle {}) {
                    v.sqrt()
                } else {
                    v
                }
            }
            Self::R2 {} | Self::ExplainedVariance {} => {
                let den = wvariance(y, w);
                ensure!(
                    den.is_finite(),
                    "R2/explained variance overflows: the target variance is not finite"
                );
                ensure!(
                    y.len() >= 2 && den > 0.0,
                    "R2/explained variance undefined for constant or singleton targets"
                );
                1.0 - if matches!(self, Self::R2 {}) {
                    squared() / den
                } else {
                    wvariance(&errors, w) / den
                }
            }
            Self::Pearson {} => correlation(y, p, w)?,
            Self::Spearman {} => correlation(&ranks(y), &ranks(p), None)?,
            Self::Huber { delta } => error_loss(&|e| {
                let e = e.abs();
                if e <= *delta {
                    0.5 * e * e
                } else {
                    delta * (e - 0.5 * delta)
                }
            }),
            Self::PseudoHuber { delta } => {
                error_loss(&|e| delta * delta * ((1.0 + (e / delta).powi(2)).sqrt() - 1.0))
            }
            Self::LogCosh {} => error_loss(&|e| {
                let e = e.abs();
                e + (-2.0 * e).exp().ln_1p() - std::f64::consts::LN_2
            }),
            Self::Pinball { quantile } => error_loss(&|e| {
                if e >= 0.0 {
                    quantile * e
                } else {
                    (quantile - 1.0) * e
                }
            }),
            Self::Expectile { quantile } => error_loss(&|e| {
                if e >= 0.0 {
                    quantile * e * e
                } else {
                    (1.0 - quantile) * e * e
                }
            }),
            Self::Linex { a } => error_loss(&|e| (a * e).exp_m1() - a * e),
            Self::AsymmetricAbsolute {
                under_cost,
                over_cost,
            } => error_loss(&|e| {
                if e >= 0.0 {
                    under_cost * e
                } else {
                    -over_cost * e
                }
            }),
            Self::AsymmetricSquared {
                under_cost,
                over_cost,
            } => error_loss(&|e| {
                if e >= 0.0 {
                    under_cost * e * e
                } else {
                    over_cost * e * e
                }
            }),
            Self::PoissonDeviance {} => {
                ensure!(
                    y.iter().all(|x| *x >= 0.0) && p.iter().all(|x| *x > 0.0),
                    "Poisson requires y>=0 and pred>0"
                );
                row_loss(&|y, p| {
                    2.0 * (if y == 0.0 {
                        p
                    } else {
                        y * (y / p).ln() - y + p
                    })
                })
            }
            Self::GammaDeviance {} => {
                ensure!(
                    y.iter().chain(p).all(|x| *x > 0.0),
                    "Gamma requires y>0 and pred>0"
                );
                row_loss(&|y, p| 2.0 * (-(y / p).ln() + y / p - 1.0))
            }
            Self::TweedieDeviance { power } => {
                ensure!(
                    y.iter().all(|x| *x >= 0.0) && p.iter().all(|x| *x > 0.0),
                    "Tweedie requires y>=0 and pred>0"
                );
                let q = *power;
                row_loss(&|y, p| {
                    2.0 * (y.powf(2.0 - q) / ((1.0 - q) * (2.0 - q))
                        - y * p.powf(1.0 - q) / (1.0 - q)
                        + p.powf(2.0 - q) / (2.0 - q))
                })
            }
            Self::Mase { season } => {
                ensure!(
                    y.len() > *season,
                    "MASE needs more scored rows than its season"
                );
                let naive = mean(
                    &y.windows(season + 1)
                        .map(|w| (w[*season] - w[0]).abs())
                        .collect::<Vec<_>>(),
                );
                ensure!(
                    naive > 0.0,
                    "MASE undefined when the seasonal naive forecast is exact"
                );
                mean(&abs) / naive
            }
            Self::MeanReturn { max_position, cost } => {
                mean(&strategy_returns(y, p, *max_position, *cost)?)
            }
            Self::Sharpe {
                max_position,
                cost,
                target,
                periods_per_year,
            }
            | Self::Sortino {
                max_position,
                cost,
                target,
                periods_per_year,
            } => {
                let returns = strategy_returns(y, p, *max_position, *cost)?;
                let den = if matches!(self, Self::Sortino { .. }) {
                    mean(
                        &returns
                            .iter()
                            .map(|r| (r - target).min(0.0).powi(2))
                            .collect::<Vec<_>>(),
                    )
                    .sqrt()
                } else {
                    variance(&returns).sqrt()
                };
                ensure!(
                    den > 0.0 && den.is_finite(),
                    "risk ratio undefined with a zero or non-finite risk denominator"
                );
                finite((mean(&returns) - target) / den * periods_per_year.sqrt())?
            }
            Self::MaxDrawdown { max_position, cost } => {
                max_drawdown(&strategy_returns(y, p, *max_position, *cost)?)?
            }
            Self::Calmar {
                max_position,
                cost,
                periods_per_year,
            } => {
                let returns = strategy_returns(y, p, *max_position, *cost)?;
                let drawdown = max_drawdown(&returns)?;
                ensure!(drawdown > 0.0, "Calmar undefined without a drawdown");
                let end = *equity_curve(&returns)?.last().expect("nonempty");
                let growth = end.powf(periods_per_year / returns.len() as f64) - 1.0;
                growth / drawdown
            }
            Self::Omega {
                max_position,
                cost,
                threshold,
            } => {
                let returns = strategy_returns(y, p, *max_position, *cost)?;
                let gains: f64 = returns.iter().map(|r| (r - threshold).max(0.0)).sum();
                let losses: f64 = returns.iter().map(|r| (threshold - r).max(0.0)).sum();
                ensure!(gains.is_finite() && losses.is_finite(), "Omega overflowed");
                ensure!(losses > 0.0, "Omega undefined without a shortfall");
                gains / losses
            }
            Self::ExpectedShortfall {
                max_position,
                cost,
                alpha,
            } => {
                let losses: Vec<f64> = strategy_returns(y, p, *max_position, *cost)?
                    .iter()
                    .map(|r| -r)
                    .collect();
                tail_mean(&losses, None, *alpha)?
            }
            Self::HitRate {} => {
                let weight = |i: usize| w.map_or(1.0, |w| w[i]);
                let (mut hits, mut total) = (0.0, 0.0);
                for i in (0..y.len()).filter(|&i| y[i] != 0.0) {
                    total += weight(i);
                    if p[i] != 0.0 && (p[i] > 0.0) == (y[i] > 0.0) {
                        hits += weight(i);
                    }
                }
                ensure!(total > 0.0, "hit rate undefined for all-zero targets");
                hits / total
            }
            Self::NormalizedGini {} => {
                let model = lorenz_gini(&ranked(y, p, w))?;
                let perfect = lorenz_gini(&ranked(y, y, w))?;
                ensure!(
                    perfect != 0.0,
                    "normalized Gini undefined for constant targets"
                );
                model / perfect
            }
            Self::Kge {} => {
                ensure!(y.len() >= 2, "KGE needs at least two rows");
                let r = correlation(y, p, None)?;
                let alpha = (variance(p) / variance(y)).sqrt();
                let my = mean(y);
                ensure!(my != 0.0, "KGE undefined for a zero-mean target");
                let beta = mean(p) / my;
                1.0 - ((r - 1.0).powi(2) + (alpha - 1.0).powi(2) + (beta - 1.0).powi(2)).sqrt()
            }
            Self::CaptureAt { fraction } => {
                let total: f64 = y.iter().sum();
                ensure!(
                    total > 0.0 && total.is_finite(),
                    "capture needs a positive target total"
                );
                let mass = fraction * y.len() as f64;
                let (mut taken, mut captured) = (0.0f64, 0.0);
                for (_, target) in ranked(y, p, None) {
                    let take = (mass - taken).min(1.0);
                    if take <= 0.0 {
                        break;
                    }
                    captured += take * target;
                    taken += take;
                }
                finite(captured / total)?
            }
            Self::Custom {
                expression,
                aggregation,
                ..
            } => {
                let v = expression.rows(y, p, columns)?;
                match aggregation {
                    Aggregation::Mean {} => wmean(&v, w),
                    Aggregation::Sum {} => match w {
                        None => v.iter().sum(),
                        Some(w) => v.iter().zip(w).map(|(v, w)| v * w).sum(),
                    },
                    Aggregation::Max {} => v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    Aggregation::Quantile { quantile: q } => wquantile(&v, w, *q)?,
                    Aggregation::TailMean { fraction } => tail_mean(&v, w, *fraction)?,
                    Aggregation::RootMean {} => {
                        let m = wmean(&v, w);
                        ensure!(m >= 0.0, "root_mean needs a nonnegative mean");
                        m.sqrt()
                    }
                }
            }
            Self::Weighted { weights, fitness } => {
                ensure!(w.is_none(), "weighting cannot be nested");
                let weights = weights.rows(y, p, columns)?;
                let total: f64 = weights.iter().sum();
                ensure!(
                    weights.iter().all(|w| *w >= 0.0) && total.is_finite() && total > 0.0,
                    "weights must be nonnegative with a positive finite total"
                );
                fitness.value(y, p, columns, Some(&weights))?
            }
            Self::Composite { terms } => {
                let mut sum = 0.0;
                for t in terms {
                    let natural = finite(t.fitness.value(y, p, columns, w)?)?;
                    let loss = if t.fitness.direction() == Direction::Maximize {
                        -natural
                    } else {
                        natural
                    };
                    sum += t.weight * loss / t.scale;
                }
                sum
            }
        })
    }
}

/// Build [`Columns`] from `(name, values)` pairs.
pub fn columns_from_pairs(pairs: &[(&str, Vec<f64>)]) -> Columns {
    pairs
        .iter()
        .map(|(name, values)| ((*name).to_owned(), values.clone()))
        .collect()
}

/// Bail with a uniform message when a specification reads columns the caller cannot supply.
pub fn require_no_columns(fitness: &Fitness, context: &str) -> Result<()> {
    let columns = fitness.columns();
    if !columns.is_empty() {
        bail!(
            "{context} cannot supply fitness columns {:?}",
            columns.into_iter().collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
    fn error_expr() -> Expression {
        Expression::Sub {
            left: Box::new(Expression::Y {}),
            right: Box::new(Expression::Pred {}),
        }
    }
    fn value(f: &Fitness, y: &[f64], p: &[f64]) -> f64 {
        f.evaluate(y, p).unwrap().value
    }
    #[test]
    fn standard_metrics_match_independent_hand_calculations() {
        for (spec, expected) in [
            (Fitness::Rmse {}, (2.0f64 / 3.0).sqrt()),
            (Fitness::Mse {}, 2.0 / 3.0),
            (Fitness::Mae {}, 2.0 / 3.0),
            (Fitness::MedianAbsoluteError {}, 1.0),
            (Fitness::MaxError {}, 1.0),
            (Fitness::Mape { epsilon: 1e-8 }, 4.0 / 9.0),
            (Fitness::Smape { epsilon: 1e-8 }, 16.0 / 45.0),
            (Fitness::Wape {}, 1.0 / 3.0),
            (Fitness::R2 {}, 0.0),
            (Fitness::ExplainedVariance {}, 0.0),
            (Fitness::Huber { delta: 1.0 }, 1.0 / 3.0),
            (
                Fitness::PseudoHuber { delta: 1.0 },
                2.0 * (2.0f64.sqrt() - 1.0) / 3.0,
            ),
            (Fitness::LogCosh {}, 2.0 * 1.0f64.cosh().ln() / 3.0),
            (Fitness::Pinball { quantile: 0.9 }, 1.0 / 3.0),
            (Fitness::Expectile { quantile: 0.8 }, 1.0 / 3.0),
            (
                Fitness::Linex { a: 1.0 },
                ((-1.0f64).exp() + 1.0f64.exp() - 2.0) / 3.0,
            ),
            (
                Fitness::AsymmetricAbsolute {
                    under_cost: 3.0,
                    over_cost: 1.0,
                },
                4.0 / 3.0,
            ),
        ] {
            near(value(&spec, &[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]), expected);
        }
        // A bounded relative error must not collapse to zero when only the
        // unscaled denominator overflows for finite, same-sign inputs.
        near(
            value(&Fitness::Smape { epsilon: 1e-8 }, &[1.2e308], &[9e307]),
            2.0 / 7.0,
        );
        near(
            value(
                &Fitness::AsymmetricSquared {
                    under_cost: 3.0,
                    over_cost: 1.0,
                },
                &[0.0, 2.0],
                &[2.0, 0.0],
            ),
            8.0,
        );
        near(
            value(
                &Fitness::Rmspe { epsilon: 1e-8 },
                &[1.0, 2.0, 4.0],
                &[2.0, 2.0, 2.0],
            ),
            (1.25f64 / 3.0).sqrt(),
        );
    }
    #[test]
    fn log_and_deviance_metrics_match_independent_calculations() {
        near(
            value(&Fitness::Msle {}, &[0.0, 3.0], &[1.0, 1.0]),
            2.0f64.ln().powi(2),
        );
        near(
            value(&Fitness::Rmsle {}, &[0.0, 3.0], &[1.0, 1.0]),
            2.0f64.ln(),
        );
        near(
            value(
                &Fitness::PoissonDeviance {},
                &[0.0, 1.0, 2.0],
                &[1.0, 1.0, 1.0],
            ),
            4.0 * 2.0f64.ln() / 3.0,
        );
        near(
            value(&Fitness::GammaDeviance {}, &[1.0, 2.0], &[2.0, 1.0]),
            0.5,
        );
        near(
            value(
                &Fitness::TweedieDeviance { power: 1.5 },
                &[0.0, 1.0, 4.0],
                &[1.0, 1.0, 1.0],
            ),
            8.0 / 3.0,
        );
    }
    #[test]
    fn correlations_handle_ties_signed_zero_and_direction() {
        near(
            Fitness::Pearson {}
                .evaluate(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0])
                .unwrap()
                .loss,
            1.0,
        );
        near(
            Fitness::Spearman {}
                .evaluate(&[0.0, -0.0, 2.0], &[2.0, 2.0, 5.0])
                .unwrap()
                .loss,
            -1.0,
        );
        let r = Fitness::R2 {}
            .evaluate(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0])
            .unwrap();
        assert_eq!(r.direction, Direction::Maximize);
        near(r.value, 1.0);
        near(r.loss, -1.0);
    }
    #[test]
    fn ranking_hydrology_and_forecast_scale_metrics_match_hand_calculations() {
        // A variance that overflows fails the score instead of returning a finite wrong value.
        assert!(Fitness::Pearson {}
            .evaluate(&[-1e155, 1e155], &[-1.0, 1.0])
            .is_err());
        assert!(Fitness::R2 {}
            .evaluate(&[-1e200, 1e200, 0.0], &[0.0, 0.0, 0.0])
            .is_err());
        // Kaggle-style normalized Gini: 0.25 / 0.3125.
        let claims = [1.0, 0.0, 0.0, 3.0];
        let risk = [0.2, 0.1, 0.3, 0.9];
        near(value(&Fitness::NormalizedGini {}, &claims, &risk), 0.8);
        near(value(&Fitness::NormalizedGini {}, &claims, &claims), 1.0);
        near(
            value(&Fitness::CaptureAt { fraction: 0.25 }, &claims, &risk),
            0.75,
        );
        near(
            value(&Fitness::CaptureAt { fraction: 0.375 }, &claims, &risk),
            0.75,
        );
        // Tied predictions earn nothing from the rows' order: a constant prediction on rows
        // sorted by claims ranks nothing, and a tie shares its claims across the boundary.
        let sorted = [3.0, 1.0, 0.0, 0.0];
        near(value(&Fitness::NormalizedGini {}, &sorted, &[0.5; 4]), 0.0);
        near(
            value(&Fitness::CaptureAt { fraction: 0.25 }, &sorted, &[0.5; 4]),
            0.25,
        );
        near(
            value(
                &Fitness::CaptureAt { fraction: 0.25 },
                &sorted,
                &[0.9, 0.9, 0.1, 0.1],
            ),
            0.5,
        );
        near(
            value(&Fitness::Kge {}, &[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]),
            1.0 - 2.0f64.sqrt(),
        );
        near(
            value(
                &Fitness::HitRate {},
                &[1.0, -1.0, 2.0, 0.0],
                &[0.5, 0.5, -1.0, 3.0],
            ),
            1.0 / 3.0,
        );
        let mase = Fitness::Mase { season: 1 };
        assert!(mase.requires_order());
        near(
            value(&mase, &[1.0, 2.0, 3.0, 5.0], &[1.0, 1.0, 1.0, 1.0]),
            21.0 / 16.0,
        );
        assert!(mase.evaluate(&[1.0, 1.0], &[0.0, 0.0]).is_err());
    }
    #[test]
    fn malformed_nonfinite_and_undefined_values_never_become_winners() {
        for (y, p) in [
            (vec![], vec![]),
            (vec![1.0], vec![]),
            (vec![f64::NAN], vec![0.0]),
            (vec![1.0], vec![f64::INFINITY]),
        ] {
            assert!(Fitness::Rmse {}.evaluate(&y, &p).is_err());
        }
        assert!(Fitness::Mse {}.evaluate(&[f64::MAX], &[-f64::MAX]).is_err());
        assert!(Fitness::MedianAbsoluteError {}
            .evaluate(&[f64::MAX, 1.0, 2.0], &[-f64::MAX, 1.0, 2.0])
            .is_err());
        for f in [
            Fitness::R2 {},
            Fitness::ExplainedVariance {},
            Fitness::Pearson {},
            Fitness::Spearman {},
            Fitness::Wape {},
            Fitness::HitRate {},
            Fitness::NormalizedGini {},
            Fitness::Kge {},
            Fitness::CaptureAt { fraction: 0.5 },
        ] {
            assert!(f.evaluate(&[0.0, 0.0], &[1.0, 2.0]).is_err(), "{f:?}");
        }
        assert!(Fitness::Msle {}.evaluate(&[-1.0], &[1.0]).is_err());
        assert!(Fitness::PoissonDeviance {}
            .evaluate(&[1.0], &[0.0])
            .is_err());
        assert!(Fitness::GammaDeviance {}.evaluate(&[0.0], &[1.0]).is_err());
        assert!(Fitness::Linex { a: 1000.0 }
            .evaluate(&[10.0], &[0.0])
            .is_err());
        assert!(Fitness::parse("{\"metric\":\"mae\",\"typo\":1}").is_err());
        assert!(Fitness::parse("import os").is_err());
    }
    #[test]
    fn configuration_parameters_and_expression_limits_are_enforced() {
        for f in [
            Fitness::Huber { delta: 0.0 },
            Fitness::Mape { epsilon: f64::NAN },
            Fitness::Pinball { quantile: 1.0 },
            Fitness::Expectile { quantile: -0.1 },
            Fitness::TweedieDeviance { power: 2.0 },
            Fitness::Composite { terms: vec![] },
            Fitness::Linex { a: 0.0 },
            Fitness::AsymmetricAbsolute {
                under_cost: 0.0,
                over_cost: 1.0,
            },
            Fitness::Mase { season: 0 },
            Fitness::ExpectedShortfall {
                max_position: 1.0,
                cost: 0.0,
                alpha: 1.0,
            },
            Fitness::CaptureAt { fraction: 0.0 },
            // Weights on a metric whose meaning they would change, and nested weighting.
            Fitness::Weighted {
                weights: Expression::Constant { value: 1.0 },
                fitness: Box::new(Fitness::MaxError {}),
            },
            Fitness::Weighted {
                weights: Expression::Constant { value: 1.0 },
                fitness: Box::new(Fitness::Weighted {
                    weights: Expression::Constant { value: 1.0 },
                    fitness: Box::new(Fitness::Mae {}),
                }),
            },
        ] {
            assert!(f.validate().is_err(), "{f:?}");
        }
        let mut expr = Expression::Y {};
        for _ in 0..18 {
            expr = Expression::Abs {
                arg: Box::new(expr),
            };
        }
        assert!(Fitness::Custom {
            name: "deep".into(),
            direction: Direction::Minimize,
            expression: expr,
            aggregation: Aggregation::Mean {}
        }
        .validate()
        .is_err());
        assert!(Fitness::parse(&" ".repeat(MAX_SPEC_BYTES + 1)).is_err());
        // The node limit covers the whole specification, not each expression in it.
        let chain = |n: usize| {
            let mut e = Expression::Y {};
            for _ in 0..n {
                e = Expression::Add {
                    left: Box::new(e),
                    right: Box::new(Expression::Constant { value: 1.0 }),
                };
            }
            e
        };
        let custom = |n: usize| Fitness::Custom {
            name: "wide".into(),
            direction: Direction::Minimize,
            expression: chain(n),
            aggregation: Aggregation::Mean {},
        };
        assert!(custom(7).validate().is_ok());
        let composite = |n: usize| Fitness::Composite {
            terms: (0..2)
                .map(|_| Term {
                    weight: 1.0,
                    scale: 1.0,
                    fitness: custom(n),
                })
                .collect(),
        };
        assert!(composite(7).validate().is_ok());
        assert!(composite(100).validate().is_err());
        let f = Fitness::Custom {
            name: "zero_division".into(),
            direction: Direction::Minimize,
            expression: Expression::Div {
                left: Box::new(Expression::Y {}),
                right: Box::new(Expression::Constant { value: 0.0 }),
            },
            aggregation: Aggregation::Mean {},
        };
        assert!(f.evaluate(&[1.0], &[0.0]).is_err());
    }
    #[test]
    fn custom_aggregations_and_scaled_composition_preserve_semantics() {
        let expr = Expression::Abs {
            arg: Box::new(error_expr()),
        };
        for (aggregation, expected) in [
            (Aggregation::Mean {}, 1.0),
            (Aggregation::Sum {}, 3.0),
            (Aggregation::Max {}, 2.0),
            (Aggregation::Quantile { quantile: 0.75 }, 1.5),
            // Largest 2 of 3 rows: (2 + 1) / 2.
            (
                Aggregation::TailMean {
                    fraction: 2.0 / 3.0,
                },
                1.5,
            ),
            // Half a row of mass: the largest value alone.
            (
                Aggregation::TailMean {
                    fraction: 1.0 / 6.0,
                },
                2.0,
            ),
            (Aggregation::RootMean {}, 1.0),
        ] {
            let f = Fitness::Custom {
                name: "business_loss".into(),
                direction: Direction::Minimize,
                expression: expr.clone(),
                aggregation,
            };
            near(value(&f, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]), expected);
        }
        let f = Fitness::Composite {
            terms: vec![
                Term {
                    weight: 2.0,
                    scale: 4.0,
                    fitness: Fitness::Mae {},
                },
                Term {
                    weight: 1.0,
                    scale: 2.0,
                    fitness: Fitness::R2 {},
                },
            ],
        };
        near(
            f.evaluate(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]).unwrap().loss,
            -0.5,
        );
    }
    #[test]
    fn expression_operators_columns_and_branches_compute_business_costs() {
        // Newsvendor profit with per-row price and unit cost columns.
        let profit = Fitness::parse(
            r#"{"metric":"custom","name":"profit","direction":"maximize",
            "expression":{"op":"sub",
              "left":{"op":"mul","left":{"op":"column","name":"price"},
                      "right":{"op":"min","left":{"op":"y"},"right":{"op":"pred"}}},
              "right":{"op":"mul","left":{"op":"column","name":"unit cost"},"right":{"op":"pred"}}},
            "aggregation":{"kind":"sum"}}"#,
        )
        .unwrap();
        assert_eq!(
            profit.columns().into_iter().collect::<Vec<_>>(),
            ["price", "unit cost"]
        );
        assert!(profit.evaluate(&[1.0], &[1.0]).is_err(), "columns required");
        let columns =
            columns_from_pairs(&[("price", vec![10.0, 4.0]), ("unit cost", vec![3.0, 1.0])]);
        // Row 0: 10*min(5,6) - 3*6 = 32; row 1: 4*min(4,2) - 1*2 = 6.
        let score = profit
            .evaluate_with(&[5.0, 4.0], &[6.0, 2.0], &columns)
            .unwrap();
        near(score.value, 38.0);
        near(score.loss, -38.0);
        // where/step/neg/pow/log1p/clip: a fixed stockout penalty plus a capped overstock cost.
        let ops = Fitness::parse(
            r#"{"metric":"custom","name":"ops","direction":"minimize",
            "expression":{"op":"add",
              "left":{"op":"mul","left":{"op":"constant","value":7},
                      "right":{"op":"step","arg":{"op":"sub","left":{"op":"y"},"right":{"op":"pred"}}}},
              "right":{"op":"where","condition":{"op":"sub","left":{"op":"pred"},"right":{"op":"y"}},
                "then":{"op":"clip","arg":{"op":"pow","left":{"op":"sub","left":{"op":"pred"},"right":{"op":"y"}},
                        "right":{"op":"constant","value":2}},"min":0,"max":5},
                "else":{"op":"neg","arg":{"op":"log1p","arg":{"op":"constant","value":0}}}}},
            "aggregation":{"kind":"sum"}}"#,
        )
        .unwrap();
        // Short by 1 (7), over by 3 (min(9,5) = 5), exact (0).
        near(value(&ops, &[3.0, 1.0, 2.0], &[2.0, 4.0, 2.0]), 12.0);
    }
    #[test]
    fn weights_reweight_rows_and_refuse_bad_values() {
        let weighted = |fitness: Fitness| Fitness::Weighted {
            weights: Expression::Column {
                name: "exposure".into(),
            },
            fitness: Box::new(fitness),
        };
        let columns = columns_from_pairs(&[("exposure", vec![1.0, 1.0, 2.0])]);
        let y = [1.0, 2.0, 3.0];
        let p = [2.0, 2.0, 2.0];
        let mae = weighted(Fitness::Mae {});
        assert_eq!(mae.name(), "weighted_mae");
        near(mae.evaluate_with(&y, &p, &columns).unwrap().value, 0.75);
        near(
            weighted(Fitness::Rmse {})
                .evaluate_with(&y, &p, &columns)
                .unwrap()
                .value,
            0.75f64.sqrt(),
        );
        // Weighted Wape: (1 + 0 + 2) / (1 + 2 + 6).
        near(
            weighted(Fitness::Wape {})
                .evaluate_with(&y, &p, &columns)
                .unwrap()
                .value,
            3.0 / 9.0,
        );
        // Unit weights reproduce every unweighted metric exactly, quantiles included.
        let ones = columns_from_pairs(&[("exposure", vec![1.0; 3])]);
        let p90 = Fitness::Custom {
            name: "p90_error".into(),
            direction: Direction::Minimize,
            expression: Expression::Abs {
                arg: Box::new(Expression::Sub {
                    left: Box::new(Expression::Y {}),
                    right: Box::new(Expression::Pred {}),
                }),
            },
            aggregation: Aggregation::Quantile { quantile: 0.9 },
        };
        for f in [
            Fitness::Mae {},
            Fitness::R2 {},
            Fitness::Pinball { quantile: 0.8 },
            Fitness::MedianAbsoluteError {},
            p90.clone(),
        ] {
            near(
                weighted(f.clone())
                    .evaluate_with(&y, &p, &ones)
                    .unwrap()
                    .value,
                value(&f, &y, &p),
            );
        }
        // Ten distinct errors: the 0.9 quantile is 9.1 weighted by ones, as unweighted.
        let ten: Vec<f64> = (1..=10).map(f64::from).collect();
        let ten_ones = columns_from_pairs(&[("exposure", vec![1.0; 10])]);
        near(value(&p90, &ten, &[0.0; 10]), 9.1);
        near(
            weighted(p90)
                .evaluate_with(&ten, &[0.0; 10], &ten_ones)
                .unwrap()
                .value,
            9.1,
        );
        let negative = columns_from_pairs(&[("exposure", vec![1.0, -1.0, 2.0])]);
        assert!(mae.evaluate_with(&y, &p, &negative).is_err());
        let zero = columns_from_pairs(&[("exposure", vec![0.0; 3])]);
        assert!(mae.evaluate_with(&y, &p, &zero).is_err());
    }
    #[test]
    fn return_ratios_use_positions_turnover_full_sample_downside_and_annualization() {
        let y = [0.1, -0.2, 0.3];
        let p = [1.0, 1.0, 1.0];
        let sortino = Fitness::Sortino {
            max_position: 1.0,
            cost: 0.0,
            target: 0.0,
            periods_per_year: 1.0,
        };
        near(
            value(&sortino, &y, &p),
            (0.2 / 3.0) / (0.04f64 / 3.0).sqrt(),
        );
        let sharpe = Fitness::Sharpe {
            max_position: 1.0,
            cost: 0.0,
            target: 0.0,
            periods_per_year: 4.0,
        };
        near(
            value(&sharpe, &y, &p),
            (0.2 / 3.0) / (0.12666666666666668f64 / 3.0).sqrt() * 2.0,
        );
        near(
            value(
                &Fitness::MeanReturn {
                    max_position: 1.0,
                    cost: 0.01,
                },
                &[0.1, 0.1],
                &[2.0, -2.0],
            ),
            -0.015,
        );
        assert!(sortino.requires_order());
        assert!(sortino.evaluate(&[0.1, 0.2], &[1.0, 1.0]).is_err());
        assert!(sharpe.evaluate(&[0.1, 0.1], &[1.0, 1.0]).is_err());
        // Overflow is an error, not a finite score: an infinite variance, an exploding equity.
        assert!(sharpe.evaluate(&[1e200, -1e200], &[1.0, 1.0]).is_err());
        assert!(Fitness::MaxDrawdown {
            max_position: 1.0,
            cost: 0.0
        }
        .evaluate(&[1e200, 1e200], &[1.0, 1.0])
        .is_err());
        // −0.0 and 0.0 are one objective.
        assert_eq!(
            Fitness::parse("sharpe:cost=-0.0")
                .unwrap()
                .identity()
                .unwrap(),
            Fitness::parse("sharpe:cost=0").unwrap().identity().unwrap()
        );
    }
    #[test]
    fn drawdown_calmar_omega_and_expected_shortfall_follow_their_definitions() {
        let y = [0.1, -0.2, 0.3];
        let p = [1.0, 1.0, 1.0];
        let (max_position, cost) = (1.0, 0.0);
        // Equity 1.1, 0.88, 1.144: the drawdown from 1.1 to 0.88 is 20%.
        near(
            value(&Fitness::MaxDrawdown { max_position, cost }, &y, &p),
            0.2,
        );
        near(
            value(
                &Fitness::Calmar {
                    max_position,
                    cost,
                    periods_per_year: 1.0,
                },
                &y,
                &p,
            ),
            (1.144f64.powf(1.0 / 3.0) - 1.0) / 0.2,
        );
        near(
            value(
                &Fitness::Omega {
                    max_position,
                    cost,
                    threshold: 0.0,
                },
                &y,
                &p,
            ),
            2.0,
        );
        let shortfall = |alpha| Fitness::ExpectedShortfall {
            max_position,
            cost,
            alpha,
        };
        near(value(&shortfall(1.0 / 3.0), &y, &p), 0.2);
        // 1.5 rows of tail mass: all of the -0.2 loss and half of the +0.1 gain.
        near(value(&shortfall(0.5), &y, &p), 0.1);
        // A total loss ends the equity curve: 100% drawdown.
        near(
            value(
                &Fitness::MaxDrawdown { max_position, cost },
                &[-1.5, 0.5],
                &[1.0, 1.0],
            ),
            1.0,
        );
        for f in [
            Fitness::Calmar {
                max_position,
                cost,
                periods_per_year: 1.0,
            },
            Fitness::Omega {
                max_position,
                cost,
                threshold: 0.0,
            },
        ] {
            assert!(f.evaluate(&[0.1, 0.2], &[1.0, 1.0]).is_err(), "{f:?}");
        }
    }
    #[test]
    fn catalog_and_custom_identities_round_trip_and_bind_every_parameter() {
        let catalog = Fitness::catalog();
        assert_eq!(catalog.len(), 38);
        let names: BTreeSet<_> = catalog.iter().map(|e| e.name).collect();
        assert_eq!(names.len(), catalog.len(), "duplicate catalog name");
        for entry in catalog {
            assert_eq!(entry.fitness.name(), entry.name);
            assert!(!entry.description.is_empty());
            let json = serde_json::to_string(&entry.fitness).unwrap();
            assert_eq!(Fitness::parse(entry.name).unwrap(), entry.fitness);
            assert_eq!(
                Fitness::parse(&json).unwrap().identity().unwrap(),
                entry.fitness.identity().unwrap()
            );
        }
        assert_ne!(
            Fitness::Pinball { quantile: 0.5 }.identity().unwrap(),
            Fitness::Pinball { quantile: 0.9 }.identity().unwrap()
        );
        assert_ne!(
            Fitness::Mae {}.identity().unwrap(),
            Fitness::Mse {}.identity().unwrap()
        );
    }
    #[test]
    fn shorthand_overrides_numeric_defaults_and_refuses_unknown_parameters() {
        assert_eq!(
            Fitness::parse("pinball:quantile=0.9").unwrap(),
            Fitness::Pinball { quantile: 0.9 }
        );
        assert_eq!(
            Fitness::parse("sortino:cost=0.0005, periods_per_year=52").unwrap(),
            Fitness::Sortino {
                max_position: 1.0,
                cost: 0.0005,
                target: 0.0,
                periods_per_year: 52.0
            }
        );
        assert_eq!(
            Fitness::parse("mase:season=7").unwrap(),
            Fitness::Mase { season: 7 }
        );
        for bad in [
            "pinball:quantile=1.5",
            "pinball:quantlie=0.9",
            "pinball:metric=1",
            "pinball:quantile=high",
            "pinball:quantile",
            "mase:season=0.5",
        ] {
            assert!(Fitness::parse(bad).is_err(), "{bad}");
        }
    }
    #[test]
    fn all_checked_in_fitness_examples_parse_and_score() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/objectives");
        let mut count = 0;
        let mut paths: Vec<_> = std::fs::read_dir(root)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        paths.sort();
        for path in paths {
            if path.extension().is_some_and(|e| e == "json")
                && !path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .contains("request")
            {
                let f = Fitness::parse(&std::fs::read_to_string(&path).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
                let n = 12;
                let positive = !(f.requires_order() || matches!(f, Fitness::HitRate {}));
                let targets: Vec<f64> = (0..n)
                    .map(|i| {
                        let v = 0.01 * (1 + i % 5) as f64;
                        if positive || i % 3 != 0 {
                            v
                        } else {
                            -v
                        }
                    })
                    .collect();
                let predictions: Vec<f64> = (0..n).map(|i| 0.012 * (1 + i % 4) as f64).collect();
                let columns: Columns = f
                    .columns()
                    .into_iter()
                    .map(|name| (name, (0..n).map(|i| 1.0 + (i % 3) as f64).collect()))
                    .collect();
                f.evaluate_with(&targets, &predictions, &columns)
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
                count += 1;
            }
        }
        assert!(count >= 24, "{count} recipes");
    }
}
