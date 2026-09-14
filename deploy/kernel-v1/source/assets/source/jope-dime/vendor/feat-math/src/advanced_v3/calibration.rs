use super::heads::sigmoid;
use crate::metrics::{auc, logloss};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum CalibrationModel {
    None,
    Platt { slope: f64, intercept: f64 },
    Isotonic { upper: Vec<f64>, values: Vec<f64> },
}

impl CalibrationModel {
    fn apply_one(&self, probability: f64) -> f64 {
        match self {
            CalibrationModel::None => probability,
            CalibrationModel::Platt { slope, intercept } => {
                let probability = probability.clamp(1e-12, 1.0 - 1e-12);
                sigmoid(slope * (probability / (1.0 - probability)).ln() + intercept)
            }
            CalibrationModel::Isotonic { upper, values } => {
                let probability = probability.clamp(1e-12, 1.0 - 1e-12);
                let index = upper
                    .partition_point(|bound| *bound < probability)
                    .min(values.len().saturating_sub(1));
                values
                    .get(index)
                    .copied()
                    .unwrap_or(probability)
                    .clamp(0.0, 1.0)
            }
        }
    }

    pub(super) fn apply(&self, probabilities: Vec<f64>) -> Vec<f64> {
        probabilities
            .into_iter()
            .map(|value| self.apply_one(value))
            .collect()
    }
}

pub(super) fn fit_calibration(kind: &str, y: &[f64], probabilities: &[f64]) -> CalibrationModel {
    if kind == "none" || y.len() != probabilities.len() || y.is_empty() {
        return CalibrationModel::None;
    }
    let candidate = match kind {
        "platt" => fit_platt(y, probabilities),
        "isotonic" => fit_isotonic(y, probabilities),
        _ => CalibrationModel::None,
    };
    let calibrated = candidate.apply(probabilities.to_vec());
    let improves_loss = logloss(y, &calibrated) + 1e-12 < logloss(y, probabilities);
    let preserves_auc = auc(y, &calibrated) + 0.005 >= auc(y, probabilities);
    if improves_loss && preserves_auc {
        candidate
    } else {
        CalibrationModel::None
    }
}

fn fit_platt(y: &[f64], probabilities: &[f64]) -> CalibrationModel {
    let logits: Vec<f64> = probabilities
        .iter()
        .map(|p| {
            let p = p.clamp(1e-6, 1.0 - 1e-6);
            (p / (1.0 - p)).ln()
        })
        .collect();
    let mut slope = 1.0;
    let mut intercept = 0.0;
    for _ in 0..100 {
        let mut g0 = 0.0;
        let mut g1 = 0.0;
        let mut h00 = 1e-9;
        let mut h01 = 0.0;
        let mut h11 = 1e-9;
        for ((&logit, &label), _) in logits.iter().zip(y).zip(0..) {
            let probability = sigmoid(slope * logit + intercept);
            let error = probability - label;
            let curvature = probability * (1.0 - probability);
            g0 += error;
            g1 += error * logit;
            h00 += curvature;
            h01 += curvature * logit;
            h11 += curvature * logit * logit;
        }
        let determinant = h00 * h11 - h01 * h01;
        if determinant.abs() < 1e-18 {
            break;
        }
        let intercept_delta = (g0 * h11 - g1 * h01) / determinant;
        let slope_delta = (g1 * h00 - g0 * h01) / determinant;
        intercept -= intercept_delta.clamp(-1.0, 1.0);
        slope -= slope_delta.clamp(-1.0, 1.0);
        if intercept_delta.abs().max(slope_delta.abs()) < 1e-10 {
            break;
        }
    }
    CalibrationModel::Platt { slope, intercept }
}

fn fit_isotonic(y: &[f64], probabilities: &[f64]) -> CalibrationModel {
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| {
        probabilities[a]
            .total_cmp(&probabilities[b])
            .then(a.cmp(&b))
    });
    let mut blocks: Vec<(f64, f64, usize)> = Vec::new();
    for &row in &order {
        blocks.push((probabilities[row], y[row], 1));
        while blocks.len() >= 2 {
            let n = blocks.len();
            let left_mean = blocks[n - 2].1 / blocks[n - 2].2 as f64;
            let right_mean = blocks[n - 1].1 / blocks[n - 1].2 as f64;
            if left_mean <= right_mean {
                break;
            }
            let right = blocks.pop().expect("block exists");
            let left = blocks.pop().expect("block exists");
            blocks.push((right.0, left.1 + right.1, left.2 + right.2));
        }
    }
    CalibrationModel::Isotonic {
        upper: blocks.iter().map(|block| block.0).collect(),
        values: blocks
            .iter()
            .map(|block| block.1 / block.2 as f64)
            .collect(),
    }
}
