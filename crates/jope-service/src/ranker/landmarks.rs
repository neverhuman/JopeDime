//! CPU implementation of the high-resolution landmark recipe.
//!
//! The FIT transform and descriptor extraction port the pinned
//! `FusedLandmarkWorkspace.fit_transform_and_extract_bundle_high` path into
//! Rust, including its 1,024-row FIT cap. The public builder returns the
//! 256-value vector and raw 24-column descriptors; ranker-facing columns are
//! projected into the existing 64-by-17 shape. Numeric feature scaling is
//! regression-only, matching the source recipe's numeric input contract.
#[path = "landmarks/mod.rs"]
mod recipe;
pub(crate) use crate::ranker::adapter;
pub use recipe::{fit_transform_and_extract_bundle_high, RecipeError};

pub const VECTOR_WIDTH: usize = 256;
pub const REPORT_WIDTH: usize = 116;
pub const TABLE_WIDTH: usize = 24;
pub const INTERACTION_WIDTH: usize = 48;
pub const BUILDING_BLOCK_WIDTH: usize = 68;
pub const RAW_COLUMN_WIDTH: usize = 24;
pub const RANKER_COLUMN_WIDTH: usize = 17;
pub const RANKER_COLUMN_COUNT: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkOutput {
    vector: [f32; VECTOR_WIDTH],
    columns: [[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT],
    raw_columns: Vec<[f32; RAW_COLUMN_WIDTH]>,
}

impl LandmarkOutput {
    pub fn from_descriptor_parts(
        report: &[f32; REPORT_WIDTH],
        table: &[f32; TABLE_WIDTH],
        interactions: &[f32; INTERACTION_WIDTH],
        building_blocks: &[f32; BUILDING_BLOCK_WIDTH],
        raw_columns: &[[f32; RAW_COLUMN_WIDTH]],
    ) -> Self {
        Self {
            vector: assemble_vector(report, table, interactions, building_blocks),
            columns: pad_ranker_columns(raw_columns),
            raw_columns: raw_columns.to_vec(),
        }
    }

    pub fn vector(&self) -> &[f32; VECTOR_WIDTH] {
        &self.vector
    }

    pub fn columns(&self) -> &[[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT] {
        &self.columns
    }

    pub fn raw_columns(&self) -> &[[f32; RAW_COLUMN_WIDTH]] {
        &self.raw_columns
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgreementSummary {
    pub sampled_rows: usize,
    pub full_rows: usize,
    pub values_compared: usize,
    pub exact_matches: usize,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgreementError {
    EmptyTable,
    SampleOverLimit,
    SampleExceedsFullTable,
}

impl std::fmt::Display for AgreementError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTable => formatter.write_str("landmark tables must contain FIT rows"),
            Self::SampleOverLimit => {
                formatter.write_str("landmark FIT sample exceeds the 1,024-row cap")
            }
            Self::SampleExceedsFullTable => {
                formatter.write_str("landmark FIT sample exceeds the full-table row count")
            }
        }
    }
}

impl std::error::Error for AgreementError {}

/// Summarize sampled-versus-full outputs without imposing a pass threshold.
/// T099 requires reporting agreement; the qualification owner sets the
/// acceptance interpretation when RES-09 fixtures and approved inputs land.
pub fn compare_sampled_to_full(
    sampled: &LandmarkOutput,
    full: &LandmarkOutput,
    sampled_rows: usize,
    full_rows: usize,
) -> Result<AgreementSummary, AgreementError> {
    if sampled_rows == 0 || full_rows == 0 {
        return Err(AgreementError::EmptyTable);
    }
    if sampled_rows > 1_024 {
        return Err(AgreementError::SampleOverLimit);
    }
    if sampled_rows > full_rows {
        return Err(AgreementError::SampleExceedsFullTable);
    }

    let mut values_compared = 0;
    let mut exact_matches = 0;
    let mut max_abs_error = 0.0_f64;
    let mut total_abs_error = 0.0_f64;

    for (sampled_value, full_value) in sampled
        .vector
        .iter()
        .chain(sampled.columns.iter().flatten())
        .zip(full.vector.iter().chain(full.columns.iter().flatten()))
    {
        let abs_error = f64::from((*sampled_value - *full_value).abs());
        values_compared += 1;
        exact_matches += usize::from(abs_error == 0.0);
        max_abs_error = max_abs_error.max(abs_error);
        total_abs_error += abs_error;
    }

    Ok(AgreementSummary {
        sampled_rows,
        full_rows,
        values_compared,
        exact_matches,
        max_abs_error,
        mean_abs_error: total_abs_error / values_compared as f64,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencySummary {
    pub samples: usize,
    pub min_ms: f64,
    pub median_ms: f64,
    pub max_ms: f64,
}

/// Summarize repeated `landmarks.ms` measurements for a pre-stage report.
pub fn summarize_latency(samples: &[std::time::Duration]) -> Option<LatencySummary> {
    if samples.is_empty() {
        return None;
    }

    let mut millis = samples
        .iter()
        .map(std::time::Duration::as_secs_f64)
        .map(|seconds| seconds * 1_000.0)
        .collect::<Vec<_>>();
    millis.sort_by(f64::total_cmp);

    let middle = millis.len() / 2;
    let median_ms = if millis.len() % 2 == 0 {
        (millis[middle - 1] + millis[middle]) / 2.0
    } else {
        millis[middle]
    };

    Some(LatencySummary {
        samples: millis.len(),
        min_ms: millis[0],
        median_ms,
        max_ms: millis[millis.len() - 1],
    })
}

/// Join the four high-level descriptor blocks in the training order.
///
/// The source concatenates `vector116`, `table_vec24`, `inter_vec48`, and
/// `building_block_vec68`, then replaces NaN and infinities with zero.
pub fn assemble_vector(
    report: &[f32; REPORT_WIDTH],
    table: &[f32; TABLE_WIDTH],
    interactions: &[f32; INTERACTION_WIDTH],
    building_blocks: &[f32; BUILDING_BLOCK_WIDTH],
) -> [f32; VECTOR_WIDTH] {
    let mut result = [0.0; VECTOR_WIDTH];

    for (offset, value) in report
        .iter()
        .chain(table)
        .chain(interactions)
        .chain(building_blocks)
        .enumerate()
    {
        result[offset] = finite_or_zero(*value);
    }

    result
}

/// Adapt raw 24-wide per-column landmarks to the ranker's `[64, 17]` input.
///
/// The training consumer copies the first 17 values for the first 64 columns,
/// truncates excess columns, zero-pads missing columns, and sanitizes
/// non-finite values to zero.
pub fn pad_ranker_columns(
    raw_columns: &[[f32; RAW_COLUMN_WIDTH]],
) -> [[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT] {
    let mut result = [[0.0; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT];

    for (target, source) in result.iter_mut().zip(raw_columns.iter()) {
        for (target_value, source_value) in target.iter_mut().zip(source.iter()) {
            *target_value = finite_or_zero(*source_value);
        }
    }

    result
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "landmarks/tests.rs"]
mod tests;
