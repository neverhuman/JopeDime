//! Objective-aware output calibration: the decision layer between a fitted model and the cost.
//!
//! Every native method fits its parameters with its own loss, usually squared error. Selecting
//! among such methods by a custom objective can only pick the least-bad mean predictor; it cannot
//! produce a 90th-percentile forecast, a newsvendor order quantity or a position size. The
//! calibration layer closes that gap without touching any estimator: it learns a monotone map
//!
//! ```text
//! pred' = max(intercept + slope·pred, lower)        slope > 0
//! ```
//!
//! that minimizes the user's objective on **out-of-fold** predictions. `lower` is set only when
//! the objective's prediction domain requires it (Poisson/Gamma/Tweedie deviances need positive
//! predictions; log errors need nonnegative ones).
//!
//! Selection compares candidates by a **cross-fitted** calibrated score: each fold's predictions
//! are transformed by a map learned on the other folds' out-of-fold prediction/label pairs (with
//! forward validation, on earlier folds only), so no row is scored by a map fitted to its own
//! pair. As in stacking, a fold's labels still trained the models behind the other folds'
//! predictions; forward validation removes even that.
//! A candidate keeps its raw predictions unless calibration improves that cross-fitted score.
//! The winner's final map is learned on all of its out-of-fold predictions and stored with the
//! fitted model; prediction applies it after the native model. When the raw predictions win but
//! the objective's domain needs a floor, the stored map is the floor alone, so a refitted model
//! cannot leave the domain on new rows.
//!
//! The search is a deterministic Nelder–Mead over `(intercept / scale, ln slope)` from fixed
//! starts, bounded in iterations, so the same inputs always give the same map. Cross-fitting
//! during selection also stops at the selection's time limit (a budget failure, as for folds).
use crate::fitness::{Columns, Fitness};
use crate::Failure;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const CALIBRATION_METHOD: &str = "cross-fitted-monotone-affine/v1";
/// Fewer rows than this leave the map at identity (plus any domain floor).
pub const MIN_CALIBRATION_ROWS: usize = 8;
const MAX_EVALUATIONS: usize = 400;
const SLOPE_BOUNDS: (f64, f64) = (1e-3, 1e3);
const INTERCEPT_BOUND: f64 = 50.0;

/// Which map family the selection may learn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Calibration {
    /// `affine` when an objective is given, otherwise `none`.
    #[default]
    Auto,
    /// Raw native predictions.
    None,
    /// Learn the intercept only.
    Shift,
    /// Learn the intercept and a positive slope.
    Affine,
}

impl Calibration {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "auto" => Some(Self::Auto),
            "none" => Some(Self::None),
            "shift" => Some(Self::Shift),
            "affine" => Some(Self::Affine),
            _ => None,
        }
    }
    /// The concrete family for a selection with or without an objective.
    pub fn resolve(self, has_objective: bool) -> Self {
        match self {
            Self::Auto if has_objective => Self::Affine,
            Self::Auto => Self::None,
            other => other,
        }
    }
}

/// Where an objective requires predictions to lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PredictionDomain {
    Any,
    NonNegative,
    Positive,
}

/// The objective's prediction domain; the strictest over composite and weighted parts.
pub fn prediction_domain(fitness: &Fitness) -> PredictionDomain {
    match fitness {
        Fitness::Msle {} | Fitness::Rmsle {} => PredictionDomain::NonNegative,
        Fitness::PoissonDeviance {}
        | Fitness::GammaDeviance {}
        | Fitness::TweedieDeviance { .. } => PredictionDomain::Positive,
        Fitness::Weighted { fitness, .. } => prediction_domain(fitness),
        Fitness::Composite { terms } => terms
            .iter()
            .map(|t| prediction_domain(&t.fitness))
            .max()
            .unwrap_or(PredictionDomain::Any),
        _ => PredictionDomain::Any,
    }
}

/// The learned monotone output map stored with a fitted model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputTransform {
    pub intercept: f64,
    pub slope: f64,
    /// Predictions below this value are raised to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower: Option<f64>,
}

impl OutputTransform {
    pub const IDENTITY: Self = Self {
        intercept: 0.0,
        slope: 1.0,
        lower: None,
    };
    pub fn apply(&self, prediction: f64) -> f64 {
        let value = self.intercept + self.slope * prediction;
        match self.lower {
            Some(lower) => value.max(lower),
            None => value,
        }
    }
    pub fn apply_all(&self, predictions: &[f64]) -> Vec<f64> {
        predictions.iter().map(|p| self.apply(*p)).collect()
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.intercept.is_finite()
                && self.slope.is_finite()
                && self.slope > 0.0
                && self.lower.is_none_or(f64::is_finite),
            "output transform needs a finite intercept, a positive finite slope and a finite floor"
        );
        Ok(())
    }
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }
}

/// The floor the objective's domain needs, learned from the calibration rows' targets.
fn domain_floor(domain: PredictionDomain, y: &[f64]) -> Option<f64> {
    match domain {
        PredictionDomain::Any => None,
        PredictionDomain::NonNegative => Some(0.0),
        PredictionDomain::Positive => {
            let smallest = y
                .iter()
                .copied()
                .filter(|v| *v > 0.0)
                .fold(f64::INFINITY, f64::min);
            Some(if smallest.is_finite() {
                smallest * 1e-3
            } else {
                1e-12
            })
        }
    }
}

fn subset(values: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|r| values[*r]).collect()
}

/// `columns` restricted to `rows`, in that order.
pub fn subset_columns(columns: &Columns, rows: &[usize]) -> Columns {
    columns
        .iter()
        .map(|(name, values)| (name.clone(), subset(values, rows)))
        .collect()
}

/// Loss of `transform` on predictions; any domain error or non-finite value is `+inf`.
fn loss_of(
    fitness: &Fitness,
    transform: &OutputTransform,
    y: &[f64],
    p: &[f64],
    columns: &Columns,
) -> f64 {
    let calibrated = transform.apply_all(p);
    if !calibrated.iter().all(|v| v.is_finite()) {
        return f64::INFINITY;
    }
    fitness
        .loss_unhashed(y, &calibrated, columns)
        .ok()
        .filter(|v| v.is_finite())
        .unwrap_or(f64::INFINITY)
}

/// Deterministic Nelder–Mead minimization in one or two dimensions; `None` when `stop()` asked
/// it to give up.
fn nelder_mead(
    f: &dyn Fn(&[f64]) -> f64,
    start: &[f64],
    step: f64,
    stop: &dyn Fn() -> bool,
) -> Option<(Vec<f64>, f64)> {
    let dim = start.len();
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(dim + 1);
    simplex.push((start.to_vec(), f(start)));
    for axis in 0..dim {
        let mut point = start.to_vec();
        point[axis] += step;
        let value = f(&point);
        simplex.push((point, value));
    }
    let mut evaluations = dim + 1;
    let order = |s: &mut Vec<(Vec<f64>, f64)>| {
        s.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0[0].total_cmp(&b.0[0])));
    };
    while evaluations < MAX_EVALUATIONS {
        if stop() {
            return None;
        }
        order(&mut simplex);
        let (best, worst) = (simplex[0].1, simplex[dim].1);
        let spread: f64 = (1..=dim)
            .map(|i| {
                simplex[i]
                    .0
                    .iter()
                    .zip(&simplex[0].0)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        let flat = best.is_finite() && worst - best <= 1e-13 * (1.0 + best.abs());
        if spread < 1e-9 || (flat && spread < 1e-6) {
            break;
        }
        let centroid: Vec<f64> = (0..dim)
            .map(|j| simplex[..dim].iter().map(|(p, _)| p[j]).sum::<f64>() / dim as f64)
            .collect();
        let towards = |t: f64| -> Vec<f64> {
            centroid
                .iter()
                .zip(&simplex[dim].0)
                .map(|(c, w)| c + t * (c - w))
                .collect()
        };
        let reflected = towards(1.0);
        let r = f(&reflected);
        evaluations += 1;
        if r < simplex[0].1 {
            let expanded = towards(2.0);
            let e = f(&expanded);
            evaluations += 1;
            simplex[dim] = if e < r { (expanded, e) } else { (reflected, r) };
        } else if r < simplex[dim - 1].1 {
            simplex[dim] = (reflected, r);
        } else {
            let contracted = if r < simplex[dim].1 {
                towards(0.5)
            } else {
                towards(-0.5)
            };
            let c = f(&contracted);
            evaluations += 1;
            if c < simplex[dim].1.min(r) {
                simplex[dim] = (contracted, c);
            } else {
                // Shrink toward the best vertex.
                let anchor = simplex[0].0.clone();
                for vertex in simplex.iter_mut().skip(1) {
                    let point: Vec<f64> = vertex
                        .0
                        .iter()
                        .zip(&anchor)
                        .map(|(v, a)| a + 0.5 * (v - a))
                        .collect();
                    let value = f(&point);
                    *vertex = (point, value);
                }
                evaluations += dim;
            }
        }
    }
    order(&mut simplex);
    Some(simplex.swap_remove(0))
}

/// The map a model keeps when its raw predictions win: identity plus the floor the objective's
/// domain needs, or `None` when the domain needs no floor.
pub fn floor_only(fitness: &Fitness, y: &[f64]) -> Option<OutputTransform> {
    domain_floor(prediction_domain(fitness), y).map(|lower| OutputTransform {
        lower: Some(lower),
        ..OutputTransform::IDENTITY
    })
}

/// Learn the map of `family` minimizing `fitness` on `(y, p)`. Never fails: when nothing beats
/// identity (plus the domain floor), that is the map. The search is bounded by its evaluation
/// count, not by a clock.
pub fn fit(
    fitness: &Fitness,
    family: Calibration,
    y: &[f64],
    p: &[f64],
    columns: &Columns,
) -> OutputTransform {
    fit_until(fitness, family, y, p, columns, &|| false)
        .expect("a search that is never stopped finishes")
}

/// [`fit`], abandoned with a budget failure as soon as `stop()` reports the time limit.
pub fn fit_until(
    fitness: &Fitness,
    family: Calibration,
    y: &[f64],
    p: &[f64],
    columns: &Columns,
    stop: &dyn Fn() -> bool,
) -> Result<OutputTransform> {
    let base = OutputTransform {
        lower: domain_floor(prediction_domain(fitness), y),
        ..OutputTransform::IDENTITY
    };
    if matches!(family, Calibration::None | Calibration::Auto) || y.len() < MIN_CALIBRATION_ROWS {
        return Ok(base);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (my, mp) = (mean(y), mean(p));
    let spread = (y.iter().map(|v| (v - my).powi(2)).sum::<f64>() / y.len() as f64).sqrt();
    let scale = if spread.is_finite() && spread > 0.0 {
        spread
    } else {
        my.abs().max(1.0)
    };
    let affine = family == Calibration::Affine;
    let decode = |x: &[f64]| OutputTransform {
        intercept: scale * x[0].clamp(-INTERCEPT_BOUND, INTERCEPT_BOUND),
        slope: if affine {
            x[1].clamp(SLOPE_BOUNDS.0.ln(), SLOPE_BOUNDS.1.ln()).exp()
        } else {
            1.0
        },
        lower: base.lower,
    };
    let objective = |x: &[f64]| loss_of(fitness, &decode(x), y, p, columns);
    let mut best = (base, loss_of(fitness, &base, y, p, columns));
    let mean_shift = if scale > 0.0 { (my - mp) / scale } else { 0.0 };
    let starts: Vec<Vec<f64>> = if affine {
        vec![vec![0.0, 0.0], vec![mean_shift, 0.0]]
    } else {
        vec![vec![0.0], vec![mean_shift]]
    };
    for start in starts {
        let Some((point, value)) = nelder_mead(&objective, &start, 0.25, stop) else {
            return Err(Failure::budget(
                "time limit reached during output calibration",
            ));
        };
        if value < best.1 {
            best = (decode(&point), value);
        }
    }
    Ok(best.0)
}

/// Cross-fitted calibrated predictions: rows of fold `f` use a map learned on `fit_rows(f)`.
/// `folds[i]` is the fold of scored row `i`; `fit_rows(f)` lists scored-row indices. Fails with
/// a budget failure when `stop()` reports the time limit.
#[allow(clippy::too_many_arguments)]
pub fn cross_fit(
    fitness: &Fitness,
    family: Calibration,
    y: &[f64],
    p: &[f64],
    columns: &Columns,
    folds: &[usize],
    fit_rows: &dyn Fn(usize) -> Vec<usize>,
    stop: &dyn Fn() -> bool,
) -> Result<Vec<f64>> {
    let mut out = vec![0.0; p.len()];
    let mut distinct: Vec<usize> = folds.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    for fold in distinct {
        let rows = fit_rows(fold);
        let transform = fit_until(
            fitness,
            family,
            &subset(y, &rows),
            &subset(p, &rows),
            &subset_columns(columns, &rows),
            stop,
        )?;
        for (i, f) in folds.iter().enumerate() {
            if *f == fold {
                out[i] = transform.apply(p[i]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
