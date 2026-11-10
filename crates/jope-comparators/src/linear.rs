//! Standardized ridge regression, the linear member of the AutoML-style comparator portfolios.

use crate::error::{ComparatorError, ComparatorResult};
use crate::matrix::{mean, predict_rows, validate_training, FeatureMatrix, MedianImputer};

/// Columns whose training standard deviation is at or below this carry no signal.
const MIN_SCALE: f64 = 1e-12;
/// Feature limit that keeps the `O(p³)` normal-equation solve bounded.
pub const MAX_RIDGE_FEATURES: usize = 2048;

/// `RidgeParams` is a struct for the ridge penalty contract in this module.
#[derive(Clone, Debug, PartialEq)]
pub struct RidgeParams {
    /// L2 penalty on standardized coefficients (finite, non-negative).
    pub alpha: f64,
}

/// `RidgeModel` is a struct for the fitted linear model contract in this module: the FIT-only
/// imputer and standardization plus coefficients on the standardized features.
#[derive(Clone, Debug, PartialEq)]
pub struct RidgeModel {
    imputer: MedianImputer,
    means: Vec<f64>,
    scales: Vec<f64>,
    coefficients: Vec<f64>,
    intercept: f64,
}

impl RidgeModel {
    /// Fit `min ‖y − ȳ − Zw‖² + α‖w‖²` on standardized training features `Z`.
    pub fn fit(
        features: FeatureMatrix<'_>,
        targets: &[f64],
        params: &RidgeParams,
    ) -> ComparatorResult<Self> {
        if !(params.alpha >= 0.0 && params.alpha.is_finite()) {
            return Err(ComparatorError::InvalidParameter { name: "alpha" });
        }
        validate_training(features, targets)?;
        let (rows, cols) = (features.rows(), features.cols());
        if cols > MAX_RIDGE_FEATURES {
            return Err(ComparatorError::TooManyFeatures {
                limit: MAX_RIDGE_FEATURES,
            });
        }
        let imputer = MedianImputer::fit(features);
        let dense = imputer.transform(features)?;
        let (means, scales) = standardization(&dense, rows, cols);
        let intercept = mean(targets);
        let mut gram = vec![0.0; cols * cols];
        let mut rhs = vec![0.0; cols];
        let mut z = vec![0.0; cols];
        for row in 0..rows {
            standardize(
                &dense[row * cols..(row + 1) * cols],
                &means,
                &scales,
                &mut z,
            );
            let centered = targets[row] - intercept;
            for i in 0..cols {
                rhs[i] += z[i] * centered;
                for j in 0..=i {
                    gram[i * cols + j] += z[i] * z[j];
                }
            }
        }
        for i in 0..cols {
            // Constant columns standardize to zero; pin their coefficient to zero.
            gram[i * cols + i] += if scales[i] > 0.0 { params.alpha } else { 1.0 };
        }
        let coefficients = solve_cholesky(&mut gram, &rhs, cols)?;
        Ok(Self {
            imputer,
            means,
            scales,
            coefficients,
            intercept,
        })
    }

    /// Intercept plus the coefficient-weighted standardized features, for each row.
    pub fn predict(&self, features: FeatureMatrix<'_>) -> ComparatorResult<Vec<f64>> {
        let mut z = vec![0.0; self.coefficients.len()];
        predict_rows(features, &self.imputer, |row| {
            standardize(row, &self.means, &self.scales, &mut z);
            self.intercept
                + z.iter()
                    .zip(&self.coefficients)
                    .map(|(value, weight)| value * weight)
                    .sum::<f64>()
        })
    }

    /// Approximate in-memory parameter footprint in bytes, for model-size comparison.
    pub fn model_bytes(&self) -> u64 {
        let scalars = self.imputer.medians().len() + 3 * self.coefficients.len() + 1;
        (scalars * std::mem::size_of::<f64>()) as u64
    }
}

fn standardization(dense: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut means = vec![0.0; cols];
    for row in dense.chunks_exact(cols) {
        for (sum, value) in means.iter_mut().zip(row) {
            *sum += value;
        }
    }
    for value in &mut means {
        *value /= rows as f64;
    }
    let mut scales = vec![0.0; cols];
    for row in dense.chunks_exact(cols) {
        for ((sum, value), center) in scales.iter_mut().zip(row).zip(&means) {
            *sum += (value - center).powi(2);
        }
    }
    for value in &mut scales {
        let deviation = (*value / rows as f64).sqrt();
        *value = if deviation > MIN_SCALE {
            deviation
        } else {
            0.0
        };
    }
    (means, scales)
}

fn standardize(row: &[f64], means: &[f64], scales: &[f64], out: &mut [f64]) {
    for (((slot, value), center), scale) in out.iter_mut().zip(row).zip(means).zip(scales) {
        *slot = if *scale > 0.0 {
            (value - center) / scale
        } else {
            0.0
        };
    }
}

/// Solve `A x = b` for symmetric positive-definite `A`, whose lower triangle is filled.
fn solve_cholesky(lower: &mut [f64], rhs: &[f64], n: usize) -> ComparatorResult<Vec<f64>> {
    for j in 0..n {
        let mut diagonal = lower[j * n + j];
        for k in 0..j {
            diagonal -= lower[j * n + k] * lower[j * n + k];
        }
        if diagonal.is_nan() || diagonal <= MIN_SCALE {
            return Err(ComparatorError::SingularSystem);
        }
        let diagonal = diagonal.sqrt();
        lower[j * n + j] = diagonal;
        for i in j + 1..n {
            let mut value = lower[i * n + j];
            for k in 0..j {
                value -= lower[i * n + k] * lower[j * n + k];
            }
            lower[i * n + j] = value / diagonal;
        }
    }
    let mut forward = vec![0.0; n];
    for i in 0..n {
        let partial: f64 = (0..i).map(|k| lower[i * n + k] * forward[k]).sum();
        forward[i] = (rhs[i] - partial) / lower[i * n + i];
    }
    let mut solution = vec![0.0; n];
    for i in (0..n).rev() {
        let partial: f64 = (i + 1..n).map(|k| lower[k * n + i] * solution[k]).sum();
        solution[i] = (forward[i] - partial) / lower[i * n + i];
    }
    Ok(solution)
}
