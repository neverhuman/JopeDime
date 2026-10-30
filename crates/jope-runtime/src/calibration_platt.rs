//! Cross-fitted monotone Platt scaling of the native binary head's positive probability.
//!
//! Input scores MUST already be out-of-fold base-head predictions. For each validation fold,
//! fit the sigmoid on other folds only; refit the stored map on all OOF pairs for future rows.
//! As in stacking, this does not remove indirect dependence through other folds' base heads.
//! No row is scored by a sigmoid fitted to its own label. This API does not fit base heads.
use crate::Failure;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA: &str = "jope.platt-calibration/v1";
pub const CAPABILITY: &str = "org.jopedime/platt-calibration/v1";
pub const PART: &str = "calibration-platt.json";
const SCORE_KIND: &str = "native-positive-probability/v1";
const RIDGE: f64 = 1e-8;
const MAX_ITERATIONS: usize = 100;
/// Bound cross-fitting work, matching the runtime's 2..=20 fold contract.
pub const MAX_FOLDS: usize = 20;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sigmoid {
    pub intercept: f64,
    pub slope: f64,
    /// Affine input is `score / scale - center`; scale is strictly positive.
    pub scale: f64,
    pub center: f64,
}

impl Sigmoid {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.intercept.is_finite()
                && self.slope.is_finite()
                && self.slope >= 0.0
                && self.scale.is_finite()
                && self.scale > 0.0
                && self.center.is_finite(),
            "invalid monotone Platt coefficients"
        );
        Ok(())
    }

    pub fn apply(&self, score: f64) -> Result<f64> {
        self.validate()?;
        ensure!(
            score.is_finite() && (0.0..=1.0).contains(&score),
            "Platt score must be a finite native positive probability"
        );
        // Avoid 0 * infinity when a constant map sees a score far outside its fitting range.
        let z = if self.slope == 0.0 {
            self.intercept
        } else {
            self.intercept + self.slope * (score / self.scale - self.center)
        };
        Ok(sigmoid(z))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    pub schema: String,
    pub score_kind: String,
    pub class_order: [String; 2],
    pub fitting_rows: usize,
    pub cross_fit_folds: usize,
    pub sigmoid: Sigmoid,
}

impl Calibration {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA && self.score_kind == SCORE_KIND,
            "unsupported Platt calibration version or score domain"
        );
        ensure!(
            self.class_order[0] != self.class_order[1]
                && self
                    .class_order
                    .iter()
                    .all(|s| !s.is_empty() && s.chars().count() <= 256),
            "Platt calibration needs two explicit distinct class names"
        );
        ensure!(
            (2..=MAX_FOLDS).contains(&self.cross_fit_folds)
                && self.fitting_rows >= self.cross_fit_folds,
            "invalid Platt cross-fitting record"
        );
        self.sigmoid.validate()
    }

    pub fn apply(&self, score: f64) -> Result<f64> {
        self.validate()?;
        self.sigmoid.apply(score)
    }
}

#[derive(Clone, Debug)]
pub struct CrossFitted {
    pub probabilities: Vec<[f64; 2]>,
    pub fold_maps: BTreeMap<usize, Sigmoid>,
    pub calibration: Calibration,
}

/// Class order is explicit: `class_order[1]` names the positive head score.
pub fn cross_fit(
    scores: &[f64],
    labels: &[&str],
    class_order: [&str; 2],
    folds: &[usize],
) -> Result<CrossFitted> {
    let bad = |s| Failure::input(format!("Platt cross-fit: {s}"));
    if scores.is_empty() || scores.len() != labels.len() || scores.len() != folds.len() {
        return Err(bad(
            "scores, labels and fold IDs must be nonempty and aligned",
        ));
    }
    if class_order[0] == class_order[1]
        || class_order
            .iter()
            .any(|s| s.is_empty() || s.chars().count() > 256)
    {
        return Err(bad("two explicit distinct class names are required"));
    }
    if scores
        .iter()
        .any(|s| !s.is_finite() || !(0.0..=1.0).contains(s))
    {
        return Err(bad("scores must be finite native positive probabilities"));
    }
    let y = labels
        .iter()
        .map(|label| {
            class_order
                .iter()
                .position(|c| c == label)
                .map(|i| i as f64)
                .ok_or_else(|| bad("label absent from class order"))
        })
        .collect::<Result<Vec<_>>>()?;
    let unique: BTreeSet<_> = folds.iter().copied().collect();
    if !(2..=MAX_FOLDS).contains(&unique.len()) {
        return Err(bad("between 2 and 20 nonempty folds are required"));
    }
    let mut probabilities = vec![[0.0; 2]; scores.len()];
    let mut fold_maps = BTreeMap::new();
    for fold in unique {
        let training: Vec<_> = (0..scores.len()).filter(|i| folds[*i] != fold).collect();
        let x: Vec<_> = training.iter().map(|i| scores[*i]).collect();
        let targets: Vec<_> = training.iter().map(|i| y[*i]).collect();
        let map = fit(&x, &targets)?;
        for i in 0..scores.len() {
            if folds[i] == fold {
                let p = map.apply(scores[i])?;
                probabilities[i] = [1.0 - p, p];
            }
        }
        fold_maps.insert(fold, map);
    }
    let calibration = Calibration {
        schema: SCHEMA.into(),
        score_kind: SCORE_KIND.into(),
        class_order: class_order.map(str::to_owned),
        fitting_rows: scores.len(),
        cross_fit_folds: fold_maps.len(),
        sigmoid: fit(scores, &y)?,
    };
    calibration.validate()?;
    Ok(CrossFitted {
        probabilities,
        fold_maps,
        calibration,
    })
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn objective(x: &[f64], targets: &[f64], intercept: f64, slope: f64) -> f64 {
    let sum: f64 = x
        .iter()
        .zip(targets)
        .map(|(x, t)| {
            let z = intercept + slope * x;
            if z >= 0.0 {
                (1.0 - t) * z + (-z).exp().ln_1p()
            } else {
                -t * z + z.exp().ln_1p()
            }
        })
        .sum();
    sum / x.len() as f64 + 0.5 * RIDGE * slope * slope
}

fn fit(scores: &[f64], labels: &[f64]) -> Result<Sigmoid> {
    let positive = labels.iter().filter(|y| **y == 1.0).count();
    let negative = labels.len() - positive;
    ensure!(
        positive > 0 && negative > 0,
        "FITNESS_UNDEFINED: Platt fitting partition needs both classes"
    );
    let scale = scores
        .iter()
        .copied()
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let center = scores.iter().map(|s| s / scale).sum::<f64>() / scores.len() as f64;
    let x: Vec<_> = scores.iter().map(|s| s / scale - center).collect();
    // Platt smoothing depends only on this fitting partition, never the held-out labels.
    let hi = (positive as f64 + 1.0) / (positive as f64 + 2.0);
    let lo = 1.0 / (negative as f64 + 2.0);
    let targets: Vec<_> = labels
        .iter()
        .map(|y| if *y == 1.0 { hi } else { lo })
        .collect();
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let mut b = (mean / (1.0 - mean)).ln();
    let mut a = 0.0;
    for _ in 0..MAX_ITERATIONS {
        let mut ga = RIDGE * a;
        let mut gb = 0.0;
        let mut haa = RIDGE;
        let mut hab = 0.0;
        let mut hbb = 0.0;
        let n = x.len() as f64;
        for (x, t) in x.iter().zip(&targets) {
            let p = sigmoid(b + a * x);
            let g = (p - t) / n;
            let h = p * (1.0 - p) / n;
            ga += g * x;
            gb += g;
            haa += h * x * x;
            hab += h * x;
            hbb += h;
        }
        if gb.abs() < 1e-10
            && (if a == 0.0 {
                ga.min(0.0).abs()
            } else {
                ga.abs()
            }) < 1e-10
        {
            let map = Sigmoid {
                intercept: b,
                slope: a,
                scale,
                center,
            };
            map.validate()?;
            return Ok(map);
        }
        let det = haa * hbb - hab * hab;
        ensure!(det.is_finite() && det > 0.0, "Platt Hessian is singular");
        let mut da = (-hbb * ga + hab * gb) / det;
        let mut db = (hab * ga - haa * gb) / det;
        if a + da < 0.0 {
            da = -a;
            db = -(gb + hab * da) / hbb;
        }
        let old = objective(&x, &targets, b, a);
        let mut step = 1.0;
        let mut accepted = false;
        for _ in 0..50 {
            let next_a = (a + step * da).max(0.0);
            let next_b = b + step * db;
            if next_a.is_finite()
                && next_b.is_finite()
                && objective(&x, &targets, next_b, next_a)
                    <= old + 1e-4 * step * (ga * da + gb * db)
            {
                a = next_a;
                b = next_b;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        ensure!(accepted, "Platt line search failed to converge");
    }
    anyhow::bail!("Platt optimizer did not converge within its bounded iteration limit")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness_classification::{
        Aggregation, ClassificationFitness, Inputs, Metric, UndefinedPolicy, FITNESS_VERSION,
    };

    fn fixture() -> (Vec<f64>, Vec<&'static str>, Vec<usize>) {
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        let mut folds = Vec::new();
        for fold in [2, 7, 19] {
            for (score, positives) in [(0.001, 2), (0.999, 8)] {
                for row in 0..10 {
                    scores.push(score);
                    labels.push(if row < positives { "yes" } else { "no" });
                    folds.push(fold);
                }
            }
        }
        (scores, labels, folds)
    }

    #[test]
    fn cross_fitted_logloss_improves_deliberately_overconfident_scores() {
        let (scores, labels, folds) = fixture();
        let result = cross_fit(&scores, &labels, ["no", "yes"], &folds).unwrap();
        let spec = ClassificationFitness {
            schema: FITNESS_VERSION.into(),
            metric: Metric::Logloss,
            class_order: ["no".into(), "yes".into()],
            aggregation: Aggregation::Mean,
            undefined: UndefinedPolicy::FailCandidate,
            threshold: None,
            cost_matrix: None,
        };
        let raw: Vec<_> = scores.iter().map(|p| [1.0 - p, *p]).collect();
        let loss = |p: &[[f64; 2]]| {
            spec.evaluate(Inputs {
                class_order: ["no", "yes"],
                labels: &labels,
                probabilities: p,
            })
            .unwrap()
            .loss
        };
        let raw_loss = loss(&raw);
        let calibrated_loss = loss(&result.probabilities);
        println!("deliberately miscalibrated fixture: raw_logloss={raw_loss:.12}, cross_fitted_logloss={calibrated_loss:.12}");
        assert!(calibrated_loss < raw_loss * 0.5);
        for map in result.fold_maps.values() {
            let p: Vec<_> = (0..=1000)
                .map(|i| map.apply(i as f64 / 1000.0).unwrap())
                .collect();
            assert!(p.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
            assert!(p.windows(2).all(|w| w[0] <= w[1]));
        }
    }

    #[test]
    fn held_out_labels_do_not_train_their_own_map() {
        let (scores, labels, folds) = fixture();
        let original = cross_fit(&scores, &labels, ["no", "yes"], &folds).unwrap();
        let changed: Vec<_> = labels
            .iter()
            .zip(&folds)
            .map(|(y, f)| {
                if *f == 7 {
                    if *y == "yes" {
                        "no"
                    } else {
                        "yes"
                    }
                } else {
                    *y
                }
            })
            .collect();
        let other = cross_fit(&scores, &changed, ["no", "yes"], &folds).unwrap();
        assert_eq!(original.fold_maps[&7], other.fold_maps[&7]);
        for (i, fold) in folds.iter().enumerate() {
            if *fold == 7 {
                assert_eq!(original.probabilities[i], other.probabilities[i]);
            }
        }
        assert_ne!(original.calibration.sigmoid, other.calibration.sigmoid);
        let rows: Vec<_> = (0..folds.len()).filter(|i| folds[*i] != 7).collect();
        let x: Vec<_> = rows.iter().map(|i| scores[*i]).collect();
        let y: Vec<_> = rows
            .iter()
            .map(|i| if labels[*i] == "yes" { 1.0 } else { 0.0 })
            .collect();
        assert_eq!(original.fold_maps[&7], fit(&x, &y).unwrap());
    }

    #[test]
    fn invalid_inputs_fail_instead_of_inventing_scores() {
        let (scores, labels, folds) = fixture();
        assert!(cross_fit(&scores[..2], &labels, ["no", "yes"], &folds).is_err());
        assert!(cross_fit(&scores, &labels, ["no", "no"], &folds).is_err());
        assert!(cross_fit(&scores, &labels, ["unknown", "yes"], &folds).is_err());
        assert!(cross_fit(&scores, &labels, ["no", "yes"], &vec![0; scores.len()]).is_err());
        assert!(cross_fit(&[0.0, 1.0], &["no", "yes"], ["no", "yes"], &[0, 1]).is_err());
        for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut bad = scores.clone();
            bad[0] = invalid;
            assert!(cross_fit(&bad, &labels, ["no", "yes"], &folds).is_err());
        }
    }

    #[test]
    fn constant_and_reverse_scores_remain_monotone_and_serializable() {
        for scores in [
            vec![0.5; 8],
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            vec![0.0; 8],
        ] {
            let labels = ["no", "yes", "no", "yes", "no", "yes", "no", "yes"];
            let result =
                cross_fit(&scores, &labels, ["no", "yes"], &[0, 0, 1, 1, 2, 2, 3, 3]).unwrap();
            let c = result.calibration;
            let restored: Calibration =
                crate::strict_json::decode(&serde_json::to_vec(&c).unwrap()).unwrap();
            assert_eq!(c, restored);
            assert!(c.apply(0.0).unwrap() <= c.apply(1.0).unwrap());
            assert_eq!(c.sigmoid.slope, 0.0);
        }
        for intercept in [-f64::MAX, f64::MAX] {
            let map = Sigmoid {
                intercept,
                slope: 1.0,
                center: 0.0,
                scale: f64::MIN_POSITIVE,
            };
            for score in [0.0, 0.5, 1.0] {
                assert!((0.0..=1.0).contains(&map.apply(score).unwrap()));
            }
        }
    }

    #[test]
    fn imbalanced_tiny_and_close_scores_fit_deterministically_with_bounded_work() {
        let folds: Vec<_> = (0..120).map(|i| i % 3).collect();
        let labels: Vec<_> = (0..120)
            .map(|i| if i % 11 == 0 { "yes" } else { "no" })
            .collect();
        for scores in [
            (0..120).map(|i| (i % 13) as f64 / 12.0).collect::<Vec<_>>(),
            (0..120).map(|i| (i % 13) as f64 * 1e-300).collect(),
            (0..120).map(|i| 0.5 + (i % 13) as f64 * 1e-15).collect(),
        ] {
            let result = cross_fit(&scores, &labels, ["no", "yes"], &folds).unwrap();
            let repeat = cross_fit(&scores, &labels, ["no", "yes"], &folds).unwrap();
            assert_eq!(result.calibration, repeat.calibration);
            assert_eq!(result.probabilities, repeat.probabilities);
            for row in result.probabilities {
                assert!(row.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
            }
        }
        let scores = vec![0.5; 42];
        let labels: Vec<_> = (0..42)
            .map(|i| if i % 2 == 0 { "no" } else { "yes" })
            .collect();
        assert!(cross_fit(
            &scores,
            &labels,
            ["no", "yes"],
            &(0..42).map(|i| i / 2).collect::<Vec<_>>()
        )
        .is_err());
    }
}
