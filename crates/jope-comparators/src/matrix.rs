//! Borrowed feature matrices, training validation and the FIT-only median imputer.

use crate::error::{ComparatorError, ComparatorResult};

/// `FeatureMatrix` is a struct for the borrowed row-major feature contract in this crate:
/// `rows × cols` values where NaN marks a missing value and infinities are refused.
#[derive(Clone, Copy, Debug)]
pub struct FeatureMatrix<'a> {
    values: &'a [f64],
    rows: usize,
    cols: usize,
}

impl<'a> FeatureMatrix<'a> {
    /// Borrow `values` as a `rows × cols` row-major matrix.
    pub fn new(values: &'a [f64], rows: usize, cols: usize) -> ComparatorResult<Self> {
        if cols == 0 {
            return Err(ComparatorError::NoFeatures);
        }
        let expected = rows
            .checked_mul(cols)
            .ok_or(ComparatorError::ShapeOverflow)?;
        if values.len() != expected {
            return Err(ComparatorError::ShapeMismatch {
                expected,
                actual: values.len(),
            });
        }
        if let Some(index) = values.iter().position(|value| value.is_infinite()) {
            return Err(ComparatorError::InfiniteFeature {
                row: index / cols,
                column: index % cols,
            });
        }
        Ok(Self { values, rows, cols })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of feature columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The values of one row. `row` must be below [`FeatureMatrix::rows`].
    pub fn row(&self, row: usize) -> &'a [f64] {
        &self.values[row * self.cols..(row + 1) * self.cols]
    }
}

/// Largest training row count: row indices are stored as `u32`.
pub(crate) const MAX_TRAINING_ROWS: usize = u32::MAX as usize;

/// Refuse an empty, oversized or mis-shaped training set, or a non-finite target.
pub(crate) fn validate_training(
    features: FeatureMatrix<'_>,
    targets: &[f64],
) -> ComparatorResult<()> {
    if features.rows() == 0 {
        return Err(ComparatorError::EmptyFit);
    }
    if features.rows() > MAX_TRAINING_ROWS {
        return Err(ComparatorError::TooManyRows {
            limit: MAX_TRAINING_ROWS,
        });
    }
    if targets.len() != features.rows() {
        return Err(ComparatorError::ShapeMismatch {
            expected: features.rows(),
            actual: targets.len(),
        });
    }
    match targets.iter().position(|target| !target.is_finite()) {
        Some(row) => Err(ComparatorError::NonFiniteTarget { row }),
        None => Ok(()),
    }
}

/// `MedianImputer` is a struct for the FIT-only missing-value contract in this module: one median
/// per column, learned from training rows only and applied unchanged to every later row.
#[derive(Clone, Debug, PartialEq)]
pub struct MedianImputer {
    medians: Vec<f64>,
}

impl MedianImputer {
    /// Learn each column's median over its non-missing training values. A column with no
    /// observed value imputes `0.0`.
    pub fn fit(features: FeatureMatrix<'_>) -> Self {
        let mut column = Vec::with_capacity(features.rows());
        let medians = (0..features.cols())
            .map(|col| {
                column.clear();
                column.extend(
                    (0..features.rows())
                        .map(|row| features.row(row)[col])
                        .filter(|value| !value.is_nan()),
                );
                median(&mut column)
            })
            .collect();
        Self { medians }
    }

    /// The learned per-column medians.
    pub fn medians(&self) -> &[f64] {
        &self.medians
    }

    /// Copy `row` into `out` with missing values replaced by the learned medians.
    pub(crate) fn fill_row(&self, row: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.extend(row.iter().zip(&self.medians).map(|(value, median)| {
            if value.is_nan() {
                *median
            } else {
                *value
            }
        }));
    }

    /// A dense row-major copy of `features` with every missing value imputed.
    pub fn transform(&self, features: FeatureMatrix<'_>) -> ComparatorResult<Vec<f64>> {
        self.check_width(features)?;
        let mut dense = Vec::with_capacity(features.rows() * features.cols());
        let mut buffer = Vec::with_capacity(features.cols());
        for row in 0..features.rows() {
            self.fill_row(features.row(row), &mut buffer);
            dense.extend_from_slice(&buffer);
        }
        Ok(dense)
    }

    fn check_width(&self, features: FeatureMatrix<'_>) -> ComparatorResult<()> {
        if features.cols() == self.medians.len() {
            Ok(())
        } else {
            Err(ComparatorError::ShapeMismatch {
                expected: self.medians.len(),
                actual: features.cols(),
            })
        }
    }
}

/// Impute each row, evaluate `model` on it and refuse any non-finite prediction.
pub(crate) fn predict_rows(
    features: FeatureMatrix<'_>,
    imputer: &MedianImputer,
    mut model: impl FnMut(&[f64]) -> f64,
) -> ComparatorResult<Vec<f64>> {
    imputer.check_width(features)?;
    let mut buffer = Vec::with_capacity(features.cols());
    (0..features.rows())
        .map(|row| {
            imputer.fill_row(features.row(row), &mut buffer);
            let prediction = model(&buffer);
            if prediction.is_finite() {
                Ok(prediction)
            } else {
                Err(ComparatorError::NonFinitePrediction { row })
            }
        })
        .collect()
}

/// Arithmetic mean; `0.0` for an empty slice.
pub(crate) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        values[middle - 1] + (values[middle] - values[middle - 1]) / 2.0
    }
}
