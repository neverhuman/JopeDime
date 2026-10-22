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
//! fitted model; prediction applies it after the native model.
//!
//! The search is a deterministic Nelder–Mead over `(intercept / scale, ln slope)` from fixed
//! starts, bounded in iterations, so the same inputs always give the same map.
use crate::fitness::{Columns, Fitness};
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

/// Deterministic Nelder–Mead minimization in one or two dimensions.
fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, start: &[f64], step: f64) -> (Vec<f64>, f64) {
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
    let (point, value) = simplex.swap_remove(0);
    (point, value)
}

/// Learn the map of `family` minimizing `fitness` on `(y, p)`. Never fails: when nothing beats
/// identity (plus the domain floor), that is the map.
pub fn fit(
    fitness: &Fitness,
    family: Calibration,
    y: &[f64],
    p: &[f64],
    columns: &Columns,
) -> OutputTransform {
    let base = OutputTransform {
        lower: domain_floor(prediction_domain(fitness), y),
        ..OutputTransform::IDENTITY
    };
    if matches!(family, Calibration::None | Calibration::Auto) || y.len() < MIN_CALIBRATION_ROWS {
        return base;
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
        let (point, value) = nelder_mead(&objective, &start, 0.25);
        if value < best.1 {
            best = (decode(&point), value);
        }
    }
    best.0
}

/// Cross-fitted calibrated predictions: rows of fold `f` use a map learned on `fit_rows(f)`.
/// `folds[i]` is the fold of scored row `i`; `fit_rows(f)` lists scored-row indices.
pub fn cross_fit(
    fitness: &Fitness,
    family: Calibration,
    y: &[f64],
    p: &[f64],
    columns: &Columns,
    folds: &[usize],
    fit_rows: &dyn Fn(usize) -> Vec<usize>,
) -> Vec<f64> {
    let mut out = vec![0.0; p.len()];
    let mut distinct: Vec<usize> = folds.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    for fold in distinct {
        let rows = fit_rows(fold);
        let transform = fit(
            fitness,
            family,
            &subset(y, &rows),
            &subset(p, &rows),
            &subset_columns(columns, &rows),
        );
        for (i, f) in folds.iter().enumerate() {
            if *f == fold {
                out[i] = transform.apply(p[i]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize) -> (Vec<f64>, Vec<f64>) {
        // A model that predicts the conditional mean of a target with level-dependent noise.
        let p: Vec<f64> = (0..n).map(|i| 10.0 + (i % 17) as f64).collect();
        let y: Vec<f64> = p
            .iter()
            .enumerate()
            .map(|(i, p)| p + (((i * 7919) % 101) as f64 / 100.0 - 0.5) * 0.4 * p)
            .collect();
        (y, p)
    }

    #[test]
    fn affine_calibration_moves_a_mean_forecast_toward_the_requested_quantile() {
        let (y, p) = rows(400);
        let fitness = Fitness::Pinball { quantile: 0.9 };
        let columns = Columns::new();
        let raw = fitness.evaluate(&y, &p).unwrap().value;
        let transform = fit(&fitness, Calibration::Affine, &y, &p, &columns);
        transform.validate().unwrap();
        let calibrated = transform.apply_all(&p);
        let tuned = fitness.evaluate(&y, &calibrated).unwrap().value;
        assert!(tuned < 0.7 * raw, "{tuned} vs {raw}");
        let covered = y.iter().zip(&calibrated).filter(|(y, c)| y <= c).count();
        let share = covered as f64 / y.len() as f64;
        assert!((share - 0.9).abs() < 0.05, "coverage {share}");
        // The same inputs give the same map.
        assert_eq!(
            transform,
            fit(&fitness, Calibration::Affine, &y, &p, &columns)
        );
    }

    #[test]
    fn calibration_never_worsens_the_fitted_rows_and_honors_the_domain_floor() {
        let (y, p) = rows(200);
        let columns = Columns::new();
        for fitness in [
            Fitness::Rmse {},
            Fitness::Mae {},
            Fitness::Spearman {},
            Fitness::AsymmetricAbsolute {
                under_cost: 5.0,
                over_cost: 1.0,
            },
        ] {
            for family in [Calibration::Shift, Calibration::Affine] {
                let t = fit(&fitness, family, &y, &p, &columns);
                let before = fitness.evaluate(&y, &p).unwrap().loss;
                let after = fitness.evaluate(&y, &t.apply_all(&p)).unwrap().loss;
                assert!(after <= before + 1e-12, "{fitness:?} {family:?}");
                if family == Calibration::Shift {
                    assert_eq!(t.slope, 1.0);
                }
            }
        }
        // Negative raw predictions make a deviance undefined; the floor rescues them.
        let fitness = Fitness::PoissonDeviance {};
        let y = [0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0, 1.0, 4.0];
        let p = [-0.5, 1.0, 2.0, 0.2, 2.5, 1.0, -0.1, 1.5, 1.2, 3.0];
        assert!(fitness.evaluate(&y, &p).is_err());
        let t = fit(&fitness, Calibration::Affine, &y, &p, &columns);
        assert_eq!(t.lower, Some(1e-3));
        assert!(fitness.evaluate(&y, &t.apply_all(&p)).is_ok());
        assert_eq!(
            fit(&fitness, Calibration::None, &y, &p, &columns),
            OutputTransform {
                lower: Some(1e-3),
                ..OutputTransform::IDENTITY
            }
        );
    }

    #[test]
    fn cross_fitting_applies_only_maps_learned_without_the_scored_fold() {
        let (y, p) = rows(120);
        let folds: Vec<usize> = (0..120).map(|i| i % 3).collect();
        let fitness = Fitness::Pinball { quantile: 0.8 };
        let columns = Columns::new();
        let others = |f: usize| (0..120).filter(|i| folds[*i] != f).collect::<Vec<_>>();
        let crossed = cross_fit(
            &fitness,
            Calibration::Affine,
            &y,
            &p,
            &columns,
            &folds,
            &others,
        );
        // Recompute fold 1 independently.
        let rows1 = others(1);
        let t1 = fit(
            &fitness,
            Calibration::Affine,
            &subset(&y, &rows1),
            &subset(&p, &rows1),
            &columns,
        );
        for i in (0..120).filter(|i| folds[*i] == 1) {
            assert_eq!(crossed[i], t1.apply(p[i]));
        }
        // Changing a fold-1 label cannot change fold 1's map.
        let mut altered = y.clone();
        altered[1] += 1e6;
        let again = cross_fit(
            &fitness,
            Calibration::Affine,
            &altered,
            &p,
            &columns,
            &folds,
            &others,
        );
        for i in (0..120).filter(|i| folds[*i] == 1) {
            assert_eq!(crossed[i], again[i]);
        }
    }

    #[test]
    fn transforms_validate_and_families_resolve() {
        assert!(OutputTransform {
            slope: 0.0,
            ..OutputTransform::IDENTITY
        }
        .validate()
        .is_err());
        assert!(OutputTransform::IDENTITY.is_identity());
        assert_eq!(Calibration::Auto.resolve(true), Calibration::Affine);
        assert_eq!(Calibration::Auto.resolve(false), Calibration::None);
        assert_eq!(Calibration::parse("shift"), Some(Calibration::Shift));
        assert_eq!(Calibration::parse("other"), None);
    }
}
