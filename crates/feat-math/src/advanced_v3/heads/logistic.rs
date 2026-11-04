use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct LogitRidgeModel {
    pub(crate) means: Vec<f64>,
    pub(crate) scales: Vec<f64>,
    pub(crate) weights: Vec<f64>,
}

impl LogitRidgeModel {
    pub(crate) fn fit(x: &Rows, y: &[f64], l2: f64, class_weight: ClassWeight) -> Result<Self> {
        validate_binary_fit_target(y, x.len())?;
        Self::fit_binary_unchecked(x, y, l2, class_weight)
    }

    pub(crate) fn fit_ovr(x: &Rows, y: &[f64], l2: f64) -> Result<Self> {
        if y.len() != x.len() || x.is_empty() {
            bail!("multiclass OvR target length differs from nonempty training rows");
        }
        let positives = y.iter().filter(|value| **value == 1.0).count();
        if positives == 0
            || positives == y.len()
            || y.iter().any(|value| !matches!(*value, 0.0 | 1.0))
        {
            bail!("multiclass OvR targets must contain both zero and one");
        }
        Self::fit_binary_unchecked(x, y, l2, ClassWeight::Balanced)
    }

    fn fit_binary_unchecked(
        x: &Rows,
        y: &[f64],
        l2: f64,
        class_weight: ClassWeight,
    ) -> Result<Self> {
        let n = x.len();
        let cols = x[0].len();
        let mut means = vec![0.0; cols];
        for row in x {
            for j in 0..cols {
                means[j] += row[j] / n as f64;
            }
        }
        let mut scales = vec![0.0; cols];
        for row in x {
            for j in 0..cols {
                let delta = row[j] - means[j];
                scales[j] += delta * delta;
            }
        }
        for scale in &mut scales {
            *scale = (*scale / n as f64).sqrt().max(1e-12);
        }
        let positives = y.iter().filter(|v| **v == 1.0).count();
        let negatives = n - positives;
        let pos_weight = if class_weight == ClassWeight::Balanced {
            n as f64 / (2.0 * positives as f64)
        } else {
            1.0
        };
        let neg_weight = if class_weight == ClassWeight::Balanced {
            n as f64 / (2.0 * negatives as f64)
        } else {
            1.0
        };
        let mut weights = vec![0.0; cols + 1];
        weights[0] = ((positives as f64 + 0.5) / (negatives as f64 + 0.5)).ln();
        let l2 = l2.clamp(1e-9, 1e6);
        for _ in 0..100 {
            let mut gradient = vec![0.0; cols + 1];
            let mut diagonal = vec![1e-9; cols + 1];
            for (row, &label) in x.iter().zip(y) {
                let sample_weight = if label == 1.0 { pos_weight } else { neg_weight };
                let mut linear = weights[0];
                for j in 0..cols {
                    linear += weights[j + 1] * ((row[j] - means[j]) / scales[j]).clamp(-30.0, 30.0);
                }
                let probability = sigmoid(linear);
                let error = sample_weight * (probability - label);
                let curvature = sample_weight * probability * (1.0 - probability);
                gradient[0] += error;
                diagonal[0] += curvature;
                for j in 0..cols {
                    let value = ((row[j] - means[j]) / scales[j]).clamp(-30.0, 30.0);
                    gradient[j + 1] += error * value;
                    diagonal[j + 1] += curvature * value * value;
                }
            }
            let mut max_delta = 0.0f64;
            let intercept_delta = (gradient[0] / diagonal[0]).clamp(-2.0, 2.0);
            weights[0] -= intercept_delta;
            max_delta = max_delta.max(intercept_delta.abs());
            for j in 0..cols {
                let gradient = gradient[j + 1] + l2 * weights[j + 1];
                let hessian = diagonal[j + 1] + l2;
                let delta = (gradient / hessian).clamp(-1.0, 1.0);
                weights[j + 1] -= delta;
                max_delta = max_delta.max(delta.abs());
            }
            if max_delta < 1e-9 {
                break;
            }
        }
        Ok(Self {
            means,
            scales,
            weights,
        })
    }

    pub(crate) fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut linear = self.weights[0];
                // `j` indexes four arrays in lockstep (row, means, scales, and weights at an
                // offset). One index says this more plainly than a three-way zip plus a skip.
                #[allow(clippy::needless_range_loop)]
                for j in 0..self.means.len().min(row.len()) {
                    linear += self.weights[j + 1]
                        * ((row[j] - self.means[j]) / self.scales[j]).clamp(-30.0, 30.0);
                }
                sigmoid(linear)
            })
            .collect()
    }
}

pub(crate) fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value.min(40.0)).exp())
    } else {
        let z = value.max(-40.0).exp();
        z / (1.0 + z)
    }
}
