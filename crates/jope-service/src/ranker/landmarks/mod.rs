mod descriptors;
mod probes;
mod stats;

use super::*;
pub(crate) use descriptors::{
    building_blocks, inter_descriptors, report_descriptors, table_descriptors,
};
pub(crate) use probes::extract_probes;
pub(crate) use stats::{
    abs_corr_with_y, corr, gini, mean, probe_bounds, quantile, recipe_argsort,
    scale_numeric_regression, squash, stddev,
};

/// Full CPU implementation of the numeric-regression `high` landmark recipe.
/// `x` is row-major FIT data; no validation/test rows are accepted by this API.
pub fn fit_transform_and_extract_bundle_high(
    x: &[f32],
    rows: usize,
    features: usize,
    y: &[f32],
) -> Result<LandmarkOutput, RecipeError> {
    if rows > 1_024 {
        return Err(RecipeError::FitSampleOverLimit);
    }
    build_high_full_table(x, rows, features, y)
}

pub(crate) fn build_high_full_table(
    x: &[f32],
    rows: usize,
    features: usize,
    y: &[f32],
) -> Result<LandmarkOutput, RecipeError> {
    if rows == 0 || features == 0 {
        return Err(RecipeError::EmptyTable);
    }
    if x.len() != rows.saturating_mul(features) || y.len() != rows {
        return Err(RecipeError::ShapeMismatch);
    }
    let (scaled_x, scaled_y) = scale_numeric_regression(x, rows, features, y);
    let stage = extract_probes(&scaled_x, &scaled_y, rows, features)?;
    let blocks = building_blocks(&stage, rows, features);
    let report = report_descriptors(&stage, &blocks, rows, features);
    let table = table_descriptors(&stage, rows, features);
    let inter = inter_descriptors(&report, &table, &blocks, features);

    let mut raw = vec![[0.0_f32; RAW_COLUMN_WIDTH]; features];
    for (j, column) in raw.iter_mut().enumerate() {
        column[..9].copy_from_slice(&report.column_spectrum[j]);
        column[9..17].copy_from_slice(&table.column[j]);
        column[17..24].copy_from_slice(&blocks.column[j]);
    }
    Ok(LandmarkOutput {
        vector: assemble_vector(&report.vector, &table.vector, &inter, &blocks.vector),
        columns: pad_ranker_columns(&raw),
        raw_columns: raw,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecipeError {
    EmptyTable,
    FitSampleOverLimit,
    ShapeMismatch,
    InvalidProbeBounds,
}

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTable => f.write_str("landmark FIT table must be nonempty"),
            Self::FitSampleOverLimit => f.write_str("landmark FIT sample exceeds 1,024 rows"),
            Self::ShapeMismatch => f.write_str("landmark FIT matrix and target shapes differ"),
            Self::InvalidProbeBounds => {
                f.write_str("landmark probe bounds must be finite and ordered")
            }
        }
    }
}

impl std::error::Error for RecipeError {}

#[derive(Clone)]
pub(crate) struct ProbeStage {
    pub(crate) y_scaled: Vec<f64>,
    pub(crate) z: Vec<f64>,
    pub(crate) z_unc: Vec<f64>,
    pub(crate) y_z: Vec<f64>,
    pub(crate) probes: Vec<[f64; 14]>,
    pub(crate) table_cols: Vec<[f64; 6]>,
    pub(crate) row_corr: [f64; 8],
}

#[derive(Clone)]
pub(crate) struct BlockStage {
    pub(crate) raw_gain: [f64; 16],
    pub(crate) vector: [f32; BUILDING_BLOCK_WIDTH],
    pub(crate) column: Vec<[f32; 7]>,
}

#[derive(Clone)]
pub(crate) struct ReportStage {
    pub(crate) vector: [f32; REPORT_WIDTH],
    pub(crate) gains: [f64; 23],
    pub(crate) ses: [f64; 23],
    pub(crate) lambdas: [f64; 23],
    pub(crate) present: [bool; 23],
    pub(crate) column_spectrum: Vec<[f32; 9]>,
}

#[derive(Clone)]
pub(crate) struct TableStage {
    pub(crate) vector: [f32; TABLE_WIDTH],
    pub(crate) column: Vec<[f32; 8]>,
}
