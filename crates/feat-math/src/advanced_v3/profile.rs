//! Deterministic, train-only dataset profiling and cheap landmarks.
//!
//! The profile vector is deliberately fixed at exactly 24 finite values.  The
//! landmark scores and sketch receipt are separate fields so adding execution
//! evidence cannot silently change the router feature contract.

#![allow(dead_code)]

use super::csv::RawTable;
use super::heads::fit_v3_head;
use super::input::{FittedInput, InputConfig};
use super::schema::{ColumnKind, DataSchema, RowSemantics};
use super::types::{stable_hash, stratified_fold_assignments, TaskKind, TaskSpec};
use crate::dsl::{ParamValue, Stage};
use crate::ops::quantile_sorted;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const DATASET_PROFILE_VERSION: &str = "advanced-v3-dataset-profile-v1";
pub(crate) const PROFILE_STAT_COUNT: usize = 24;
pub(crate) const MAX_PROFILE_ROWS: usize = 4_096;
pub(crate) const MAX_PROFILE_COLUMNS: usize = 256;
pub(crate) const MAX_PROFILE_NUMERIC_PAIRS: usize = 1_024;
pub(crate) const MAX_LANDMARK_ROWS: usize = 512;
pub(crate) const LANDMARK_FOLDS: usize = 3;

pub(crate) const PROFILE_STAT_NAMES: [&str; PROFILE_STAT_COUNT] = [
    "ln1p_rows",
    "ln1p_raw_columns",
    "log_aspect_ratio",
    "numeric_column_fraction",
    "missing_cell_fraction",
    "rows_with_missing_fraction",
    "median_column_missingness",
    "maximum_column_missingness",
    "numeric_zero_fraction",
    "constant_column_fraction",
    "dominant_99pct_column_fraction",
    "median_numeric_unique_fraction",
    "p90_numeric_unique_fraction",
    "median_log_categorical_cardinality",
    "maximum_log_categorical_cardinality",
    "rare_category_cell_fraction",
    "median_absolute_skew",
    "median_absolute_excess_kurtosis",
    "numeric_3iqr_outlier_fraction",
    "duplicate_raw_row_fraction",
    "median_absolute_pairwise_numeric_correlation",
    "maximum_numeric_target_association",
    "maximum_categorical_target_correlation_ratio",
    "target_skew_or_binary_minority_fraction",
];

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CheapLandmarkKind {
    RegressionMean,
    RegressionRidge,
    RegressionShallowTree,
    RegressionShallowGam,
    RegressionPls,
    BinaryPrior,
    BinaryLogitRidge,
    BinaryShallowTree,
    BinaryShallowGam,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct CheapLandmarkScore {
    pub kind: CheapLandmarkKind,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct CheapLandmarksV1 {
    pub rows: usize,
    pub folds: usize,
    pub scores: Vec<CheapLandmarkScore>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ProfileSketchReceipt {
    pub source_rows: usize,
    pub source_columns: usize,
    pub selected_rows: usize,
    pub selected_columns: usize,
    pub selected_numeric_pairs: usize,
    pub landmark_rows: usize,
    pub selection_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct DatasetProfileV1 {
    version: String,
    statistics: [f64; PROFILE_STAT_COUNT],
    landmarks: CheapLandmarksV1,
    sketch: ProfileSketchReceipt,
}

impl DatasetProfileV1 {
    pub(crate) fn statistics(&self) -> &[f64; PROFILE_STAT_COUNT] {
        &self.statistics
    }

    pub(crate) fn statistic(&self, name: &str) -> Option<f64> {
        PROFILE_STAT_NAMES
            .iter()
            .position(|candidate| *candidate == name)
            .map(|index| self.statistics[index])
    }

    pub(crate) fn landmarks(&self) -> &CheapLandmarksV1 {
        &self.landmarks
    }

    pub(crate) fn sketch(&self) -> &ProfileSketchReceipt {
        &self.sketch
    }

    pub(crate) fn sha256(&self) -> Result<String> {
        stable_hash(self)
    }

    fn validate(&self) -> Result<()> {
        if self.version != DATASET_PROFILE_VERSION {
            bail!("dataset profile version mismatch");
        }
        if self.statistics.iter().any(|value| !value.is_finite()) {
            bail!("dataset profile contains a nonfinite statistic");
        }
        if self
            .landmarks
            .scores
            .iter()
            .any(|entry| !entry.score.is_finite())
        {
            bail!("dataset profile contains a nonfinite landmark");
        }
        if self.sketch.selected_rows > MAX_PROFILE_ROWS
            || self.sketch.selected_columns > MAX_PROFILE_COLUMNS
            || self.sketch.selected_numeric_pairs > MAX_PROFILE_NUMERIC_PAIRS
            || self.sketch.landmark_rows > MAX_LANDMARK_ROWS
        {
            bail!("dataset profile exceeds a deterministic sketch bound");
        }
        Ok(())
    }
}

enum ColumnSketch<'a> {
    Numeric(Vec<Option<f64>>),
    Categorical(Vec<Option<&'a str>>),
}

/// Build a fixed-width, train-only dataset profile and deterministic cheap
/// landmarks.  No held-out rows, validation labels, or external metadata are
/// accepted by this API.
pub(crate) fn build_dataset_profile(
    table: &RawTable,
    schema: &DataSchema,
    task: &TaskSpec,
    targets: &[f64],
) -> Result<DatasetProfileV1> {
    task.validate()?;
    schema.validate()?;
    table.validate_against(schema)?;
    if table.rows.is_empty() || targets.len() != table.rows.len() {
        bail!("dataset profile requires a nonempty table and one target per row");
    }
    if targets.iter().any(|value| !value.is_finite()) {
        bail!("dataset profile targets must be finite");
    }
    if task.kind == TaskKind::Binary && targets.iter().any(|value| *value != 0.0 && *value != 1.0) {
        bail!("binary dataset profile targets must be exactly 0 or 1");
    }

    let row_indices = deterministic_indices(table.rows.len(), MAX_PROFILE_ROWS);
    let column_indices = deterministic_indices(schema.columns.len(), MAX_PROFILE_COLUMNS);
    let numeric_columns: Vec<usize> = column_indices
        .iter()
        .copied()
        .filter(|column| schema.columns[*column].kind == ColumnKind::Numeric)
        .collect();
    let numeric_pairs = deterministic_numeric_pairs(&numeric_columns, MAX_PROFILE_NUMERIC_PAIRS);
    let landmark_rows = deterministic_indices(table.rows.len(), MAX_LANDMARK_ROWS);
    let selection_sha256 = stable_hash(&(
        DATASET_PROFILE_VERSION,
        &row_indices,
        &column_indices,
        &numeric_pairs,
        &landmark_rows,
    ))?;

    let (statistics, sketches) = compute_statistics(
        table,
        schema,
        task,
        targets,
        &row_indices,
        &column_indices,
        &numeric_pairs,
    )?;
    // Keep the borrow-backed sketches alive through all association work in
    // `compute_statistics`; only the fixed statistics cross this boundary.
    drop(sketches);

    let landmarks = compute_landmarks(
        table,
        schema,
        task,
        targets,
        &landmark_rows,
        &column_indices,
    )?;
    let profile = DatasetProfileV1 {
        version: DATASET_PROFILE_VERSION.to_string(),
        statistics,
        landmarks,
        sketch: ProfileSketchReceipt {
            source_rows: table.rows.len(),
            source_columns: schema.columns.len(),
            selected_rows: row_indices.len(),
            selected_columns: column_indices.len(),
            selected_numeric_pairs: numeric_pairs.len(),
            landmark_rows: landmark_rows.len(),
            selection_sha256,
        },
    };
    profile.validate()?;
    Ok(profile)
}

mod profile_landmarks;
mod profile_statistics;

use profile_landmarks::compute_landmarks;
use profile_statistics::{
    compute_statistics, deterministic_indices, deterministic_numeric_pairs, finite_stat,
    subset_table,
};

#[cfg(test)]
mod tests;
