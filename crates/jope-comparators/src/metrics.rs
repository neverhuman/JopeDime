//! Regression scores shared by comparator models and the run report.

use crate::error::{ComparatorError, ComparatorResult};

fn check_pairs(targets: &[f64], predictions: &[f64]) -> ComparatorResult<()> {
    if targets.is_empty() {
        return Err(ComparatorError::EmptyScore);
    }
    if predictions.len() != targets.len() {
        return Err(ComparatorError::ShapeMismatch {
            expected: targets.len(),
            actual: predictions.len(),
        });
    }
    if let Some(row) = targets.iter().position(|target| !target.is_finite()) {
        return Err(ComparatorError::NonFiniteTarget { row });
    }
    match predictions.iter().position(|value| !value.is_finite()) {
        Some(row) => Err(ComparatorError::NonFinitePrediction { row }),
        None => Ok(()),
    }
}

fn mean_squared_error(targets: &[f64], predictions: &[f64]) -> f64 {
    targets
        .iter()
        .zip(predictions)
        .map(|(target, prediction)| (target - prediction).powi(2))
        .sum::<f64>()
        / targets.len() as f64
}

/// Root mean squared error.
pub fn rmse(targets: &[f64], predictions: &[f64]) -> ComparatorResult<f64> {
    check_pairs(targets, predictions)?;
    Ok(mean_squared_error(targets, predictions).sqrt())
}

/// Mean absolute error.
pub fn mae(targets: &[f64], predictions: &[f64]) -> ComparatorResult<f64> {
    check_pairs(targets, predictions)?;
    Ok(targets
        .iter()
        .zip(predictions)
        .map(|(target, prediction)| (target - prediction).abs())
        .sum::<f64>()
        / targets.len() as f64)
}

/// Coefficient of determination. With constant targets it is `1.0` for a perfect fit and `0.0`
/// otherwise, matching scikit-learn's finite convention.
pub fn r_squared(targets: &[f64], predictions: &[f64]) -> ComparatorResult<f64> {
    check_pairs(targets, predictions)?;
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let total: f64 = targets.iter().map(|target| (target - mean).powi(2)).sum();
    let residual = mean_squared_error(targets, predictions) * targets.len() as f64;
    if total > 0.0 {
        Ok(1.0 - residual / total)
    } else if residual == 0.0 {
        Ok(1.0)
    } else {
        Ok(0.0)
    }
}
