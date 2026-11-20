//! Cross-fitted Temperature Scaling (`Guo et al., ICML 2017`) of native K-class (`K >= 3`)
//! probability vectors.
//!
//! Input probability rows MUST already be out-of-fold base-head predictions aligned with
//! `class_order`. For each validation fold, fit the scalar temperature `T in [0.05, 20.0]` on
//! other folds only, minimizing the user's `jope.multiclass-fitness/v1` objective via deterministic
//! coarse grid + golden-section refinement over `ln(T)`. Refit the stored scalar temperature on
//! all out-of-fold rows for future predictions. Because `T > 0` scales every class log-probability
//! by the same positive constant, `argmax` class order is strictly preserved on every row.
use crate::fitness_multiclass::{Inputs, MulticlassFitness, PROBABILITY_SUM_TOLERANCE};
use crate::Failure;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA: &str = "jope.temperature-calibration/v1";
pub const CAPABILITY: &str = "org.jopedime/temperature-calibration/v1";
pub const PART: &str = "temperature.json";
const SCORE_KIND: &str = "native-class-probabilities/v1";

pub const MIN_TEMPERATURE: f64 = 0.05;
pub const MAX_TEMPERATURE: f64 = 20.0;
pub const PROBABILITY_FLOOR: f64 = 1e-15;
pub const MAX_FOLDS: usize = 20;
const GRID_STEPS: usize = 40;
const GOLDEN_ITERATIONS: usize = 50;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    pub schema: String,
    pub score_kind: String,
    pub class_order: Vec<String>,
    pub fitting_rows: usize,
    pub cross_fit_folds: usize,
    pub temperature: f64,
}

impl Calibration {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA && self.score_kind == SCORE_KIND,
            "unsupported Temperature calibration version or score domain"
        );
        validate_class_names(&self.class_order)?;
        ensure!(
            (2..=MAX_FOLDS).contains(&self.cross_fit_folds)
                && self.fitting_rows >= self.cross_fit_folds,
            "invalid Temperature cross-fitting record"
        );
        ensure!(
            self.temperature.is_finite()
                && (MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&self.temperature),
            "invalid Temperature scalar {} (expected finite in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}])",
            self.temperature
        );
        Ok(())
    }

    pub fn apply(&self, row: &[f64]) -> Result<Vec<f64>> {
        self.validate()?;
        ensure!(
            row.len() == self.class_order.len(),
            "Temperature calibration expected {} classes, got {}",
            self.class_order.len(),
            row.len()
        );
        apply_temperature(row, self.temperature)
    }
}

fn validate_class_names<S: AsRef<str>>(class_order: &[S]) -> Result<()> {
    ensure!(
        (3..=1024).contains(&class_order.len()),
        "Temperature calibration requires 3..=1024 distinct class names"
    );
    let mut seen = BTreeSet::new();
    for raw in class_order {
        let name = raw.as_ref();
        ensure!(
            !name.is_empty() && name.chars().count() <= 256 && seen.insert(name),
            "Temperature calibration requires distinct class names of 1..=256 characters"
        );
    }
    Ok(())
}

/// Apply scalar temperature `T in [0.05, 20.0]` to one K-class probability vector:
/// `z_k = ln(max(p_k, 1e-15)) / T`, followed by shift-invariant softmax.
pub fn apply_temperature(row: &[f64], temperature: f64) -> Result<Vec<f64>> {
    ensure!(
        temperature.is_finite() && (MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&temperature),
        "Temperature must be finite in [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}]"
    );
    ensure!(
        row.len() >= 3 && row.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
        "Temperature input must be >= 3 finite class probabilities in [0, 1]"
    );
    let sum: f64 = row.iter().sum();
    ensure!(
        (sum - 1.0).abs() <= PROBABILITY_SUM_TOLERANCE,
        "Temperature input probabilities sum to {sum}, outside 1 +- {PROBABILITY_SUM_TOLERANCE:e}"
    );
    let inv_t = 1.0 / temperature;
    let mut logits = Vec::with_capacity(row.len());
    let mut max_logit = f64::NEG_INFINITY;
    for &p in row {
        let z = p.max(PROBABILITY_FLOOR).ln() * inv_t;
        if z > max_logit {
            max_logit = z;
        }
        logits.push(z);
    }
    let mut exps = Vec::with_capacity(row.len());
    let mut denom = 0.0;
    for z in logits {
        let e = (z - max_logit).exp();
        denom += e;
        exps.push(e);
    }
    ensure!(
        denom.is_finite() && denom > 0.0,
        "non-finite softmax normalizer"
    );
    Ok(exps.into_iter().map(|e| e / denom).collect())
}

#[derive(Clone, Debug)]
pub struct CrossFitted {
    pub probabilities: Vec<Vec<f64>>,
    pub fold_temperatures: BTreeMap<usize, f64>,
    pub calibration: Calibration,
}

/// Cross-fit Temperature Scaling under the default `logloss` objective.
pub fn cross_fit(
    probabilities: &[Vec<f64>],
    labels: &[&str],
    class_order: &[&str],
    folds: &[usize],
) -> Result<CrossFitted> {
    let classes: Vec<String> = class_order.iter().map(|s| (*s).to_owned()).collect();
    let fitness = MulticlassFitness::from_shorthand("logloss", classes)
        .map_err(|e| Failure::input(format!("Temperature cross-fit: {e:#}")))?;
    cross_fit_with_fitness(probabilities, labels, class_order, folds, &fitness)
}

/// Cross-fit Temperature Scaling minimizing `fitness` on out-of-fold K-class probability rows.
pub fn cross_fit_with_fitness(
    probabilities: &[Vec<f64>],
    labels: &[&str],
    class_order: &[&str],
    folds: &[usize],
    fitness: &MulticlassFitness,
) -> Result<CrossFitted> {
    let bad = |s: String| Failure::input(format!("Temperature cross-fit: {s}"));
    fitness.validate().map_err(|e| bad(format!("{e:#}")))?;
    validate_class_names(class_order).map_err(|e| bad(format!("{e:#}")))?;
    if fitness.class_order.len() != class_order.len()
        || fitness
            .class_order
            .iter()
            .zip(class_order.iter())
            .any(|(a, b)| a.as_str() != *b)
    {
        return Err(bad(
            "fitness class_order differs from supplied class_order".into()
        ));
    }
    let n = probabilities.len();
    if n == 0 || n != labels.len() || n != folds.len() {
        return Err(bad(
            "probabilities, labels and fold IDs must be nonempty and aligned".into(),
        ));
    }
    let k = class_order.len();
    for (row_idx, (label, row)) in labels.iter().zip(probabilities.iter()).enumerate() {
        if !class_order.contains(label) {
            return Err(bad(format!("unknown label at row {row_idx}")));
        }
        if row.len() != k
            || row
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err(bad(format!(
                "row {row_idx} must contain {k} finite probabilities in [0, 1]"
            )));
        }
        let sum: f64 = row.iter().sum();
        if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err(bad(format!(
                "row {row_idx} probabilities sum to {sum}, outside 1 +- {PROBABILITY_SUM_TOLERANCE:e}"
            )));
        }
    }
    let unique: BTreeSet<usize> = folds.iter().copied().collect();
    if !(2..=MAX_FOLDS).contains(&unique.len()) {
        return Err(bad("between 2 and 20 nonempty folds are required".into()));
    }

    let mut calibrated_rows = vec![Vec::new(); n];
    let mut fold_temperatures = BTreeMap::new();
    for fold in unique {
        let train_probs: Vec<Vec<f64>> = (0..n)
            .filter(|&i| folds[i] != fold)
            .map(|i| probabilities[i].clone())
            .collect();
        let train_labels: Vec<&str> = (0..n)
            .filter(|&i| folds[i] != fold)
            .map(|i| labels[i])
            .collect();
        let temp = fit_temperature(&train_probs, &train_labels, class_order, fitness)?;
        for i in 0..n {
            if folds[i] == fold {
                calibrated_rows[i] = apply_temperature(&probabilities[i], temp)?;
            }
        }
        fold_temperatures.insert(fold, temp);
    }

    let final_temperature = fit_temperature(probabilities, labels, class_order, fitness)?;
    let calibration = Calibration {
        schema: SCHEMA.to_owned(),
        score_kind: SCORE_KIND.to_owned(),
        class_order: class_order.iter().map(|s| (*s).to_owned()).collect(),
        fitting_rows: n,
        cross_fit_folds: fold_temperatures.len(),
        temperature: final_temperature,
    };
    calibration.validate()?;
    Ok(CrossFitted {
        probabilities: calibrated_rows,
        fold_temperatures,
        calibration,
    })
}

fn evaluate_log_temp(
    log_t: f64,
    probabilities: &[Vec<f64>],
    labels: &[&str],
    class_order: &[&str],
    fitness: &MulticlassFitness,
) -> f64 {
    let t = log_t
        .clamp(MIN_TEMPERATURE.ln(), MAX_TEMPERATURE.ln())
        .exp()
        .clamp(MIN_TEMPERATURE, MAX_TEMPERATURE);
    let mut scaled = Vec::with_capacity(probabilities.len());
    for row in probabilities {
        let Ok(mapped) = apply_temperature(row, t) else {
            return f64::INFINITY;
        };
        scaled.push(mapped);
    }
    fitness
        .evaluate(Inputs {
            class_order,
            labels,
            probabilities: &scaled,
        })
        .map(|s| s.loss)
        .ok()
        .filter(|loss| loss.is_finite())
        .unwrap_or(f64::INFINITY)
}

fn golden_section_min<F: Fn(f64) -> f64>(mut a: f64, mut b: f64, f: &F) -> (f64, f64) {
    // invphi = (sqrt(5) - 1) / 2
    let invphi = (5.0_f64.sqrt() - 1.0) * 0.5;
    let mut c = b - invphi * (b - a);
    let mut d = a + invphi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..GOLDEN_ITERATIONS {
        if (b - a).abs() <= 1e-11 {
            break;
        }
        if fc <= fd {
            b = d;
            d = c;
            fd = fc;
            c = b - invphi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + invphi * (b - a);
            fd = f(d);
        }
    }
    let mid = 0.5 * (a + b);
    let fmid = f(mid);
    if fc <= fd && fc <= fmid {
        (c, fc)
    } else if fd <= fmid {
        (d, fd)
    } else {
        (mid, fmid)
    }
}

/// Fit scalar temperature `T in [0.05, 20.0]` on a partition of OOF rows. Returns `1.0` whenever
/// no candidate temperature strictly improves over `T = 1.0`.
pub fn fit_temperature(
    probabilities: &[Vec<f64>],
    labels: &[&str],
    class_order: &[&str],
    fitness: &MulticlassFitness,
) -> Result<f64> {
    ensure!(
        !probabilities.is_empty() && probabilities.len() == labels.len(),
        "Temperature fitting partition must be nonempty and aligned"
    );
    let lo = MIN_TEMPERATURE.ln();
    let hi = MAX_TEMPERATURE.ln();
    let eval = |u: f64| evaluate_log_temp(u, probabilities, labels, class_order, fitness);

    // Baseline at T = 1.0 (ln T = 0.0).
    let base_loss = eval(0.0);
    let mut best_u = 0.0;
    let mut best_loss = base_loss;

    // Coarse deterministic grid over [ln 0.05, ln 20.0] (symmetric around 0.0 so u = 0.0 is hit).
    let mut grid = Vec::with_capacity(GRID_STEPS + 1);
    for idx in 0..=GRID_STEPS {
        let u = lo + (hi - lo) * (idx as f64 / GRID_STEPS as f64);
        let loss = eval(u);
        grid.push((u, loss));
        if loss < best_loss {
            best_loss = loss;
            best_u = u;
        }
    }

    ensure!(
        best_loss.is_finite(),
        "FITNESS_UNDEFINED: Temperature fitting partition produced no finite objective score"
    );

    let best_grid_idx = grid
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.1.total_cmp(&b.1))
        .map_or(GRID_STEPS / 2, |(idx, _)| idx);
    let left_bracket = grid[best_grid_idx.saturating_sub(1)].0;
    let right_bracket = grid[(best_grid_idx + 1).min(GRID_STEPS)].0;

    for (a, b) in [
        (left_bracket, right_bracket),
        (-1.5_f64.max(lo), 1.5_f64.min(hi)),
    ] {
        if a < b {
            let (cand_u, cand_loss) = golden_section_min(a, b, &eval);
            if cand_loss < best_loss {
                best_loss = cand_loss;
                best_u = cand_u;
            }
        }
    }

    if best_loss < base_loss {
        Ok(best_u
            .clamp(lo, hi)
            .exp()
            .clamp(MIN_TEMPERATURE, MAX_TEMPERATURE))
    } else {
        Ok(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_temperature_preserves_simplex_and_argmax_order() {
        let row = [0.65, 0.25, 0.10];
        for t in [MIN_TEMPERATURE, 0.4, 1.0, 2.75, MAX_TEMPERATURE] {
            let out = apply_temperature(&row, t).unwrap();
            assert_eq!(out.len(), 3);
            assert!((out.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
            assert!(out[0] > out[1] && out[1] > out[2], "t={t}: {out:?}");
        }
        let identity = apply_temperature(&row, 1.0).unwrap();
        for (a, b) in row.iter().zip(identity.iter()) {
            assert!((a - b).abs() <= 1e-12);
        }
        assert!(apply_temperature(&row, 0.01).is_err());
        assert!(apply_temperature(&row, 25.0).is_err());
        assert!(apply_temperature(&[0.5, 0.5], 1.0).is_err());
        assert!(apply_temperature(&[0.2, 0.2, 0.2], 1.0).is_err());
    }

    #[test]
    fn cross_fit_improves_overconfident_multiclass_logloss_and_validates_bounds() {
        let classes = ["alpha", "beta", "gamma"];
        let mut probs = Vec::new();
        let mut labels = Vec::new();
        let mut folds = Vec::new();
        // Overconfident base predictions (0.96 on predicted class, ~25% error rate -> optimal T > 1).
        for i in 0..60 {
            let pred_cls = i % 3;
            let true_cls = if i % 4 == 0 {
                (pred_cls + 1) % 3
            } else {
                pred_cls
            };
            let mut row = vec![0.02; 3];
            row[pred_cls] = 0.96;
            probs.push(row);
            labels.push(classes[true_cls]);
            folds.push(i % 5);
        }
        let fitness = MulticlassFitness::from_shorthand(
            "logloss",
            classes.iter().map(|s| (*s).to_owned()).collect(),
        )
        .unwrap();
        let raw_loss = fitness
            .evaluate(Inputs {
                class_order: &classes,
                labels: &labels,
                probabilities: &probs,
            })
            .unwrap()
            .loss;
        let cf = cross_fit_with_fitness(&probs, &labels, &classes, &folds, &fitness).unwrap();
        cf.calibration.validate().unwrap();
        assert!(
            cf.calibration.temperature > 1.2,
            "expected softening temperature > 1.2, got {}",
            cf.calibration.temperature
        );
        let cal_loss = fitness
            .evaluate(Inputs {
                class_order: &classes,
                labels: &labels,
                probabilities: &cf.probabilities,
            })
            .unwrap()
            .loss;
        assert!(
            cal_loss < raw_loss,
            "expected cross-fitted logloss {cal_loss} < raw {raw_loss}"
        );

        let mut bad = cf.calibration.clone();
        bad.temperature = 0.01;
        assert!(bad.validate().is_err());
        bad.temperature = 25.0;
        assert!(bad.validate().is_err());
    }
}
