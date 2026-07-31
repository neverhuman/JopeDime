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

fn compute_statistics<'a>(
    table: &'a RawTable,
    schema: &DataSchema,
    task: &TaskSpec,
    targets: &[f64],
    rows: &[usize],
    columns: &[usize],
    numeric_pairs: &[(usize, usize)],
) -> Result<([f64; PROFILE_STAT_COUNT], Vec<ColumnSketch<'a>>)> {
    let mut sketches = Vec::with_capacity(columns.len());
    let mut row_has_missing = vec![false; rows.len()];
    let mut column_missingness = Vec::with_capacity(columns.len());
    let mut numeric_unique_fractions = Vec::new();
    let mut categorical_log_cardinalities = Vec::new();
    let mut absolute_skews = Vec::new();
    let mut absolute_kurtoses = Vec::new();
    let mut missing_cells = 0usize;
    let mut observed_numeric_cells = 0usize;
    let mut numeric_zero_cells = 0usize;
    let mut constant_columns = 0usize;
    let mut dominant_columns = 0usize;
    let mut categorical_cells = 0usize;
    let mut rare_categorical_cells = 0usize;
    let mut outlier_numeric_cells = 0usize;

    for &column in columns {
        match schema.columns[column].kind {
            ColumnKind::Numeric => {
                let mut values = Vec::with_capacity(rows.len());
                for (sample_row, &source_row) in rows.iter().enumerate() {
                    let value = parse_numeric_profile(&table.rows[source_row][column])
                        .with_context(|| {
                            format!(
                                "invalid numeric profile value at row {}, column '{}'",
                                source_row + 1,
                                schema.columns[column].name
                            )
                        })?;
                    if value.is_none() {
                        missing_cells += 1;
                        row_has_missing[sample_row] = true;
                    }
                    values.push(value);
                }
                let observed: Vec<f64> = values.iter().flatten().copied().collect();
                observed_numeric_cells += observed.len();
                numeric_zero_cells += observed.iter().filter(|value| **value == 0.0).count();
                let unique = numeric_unique_count(&observed);
                numeric_unique_fractions.push(if observed.is_empty() {
                    0.0
                } else {
                    unique as f64 / observed.len() as f64
                });
                let (skew, kurtosis) = skew_and_excess_kurtosis(&observed);
                absolute_skews.push(skew.abs());
                absolute_kurtoses.push(kurtosis.abs());
                outlier_numeric_cells += three_iqr_outlier_count(&observed);
                constant_columns += usize::from(unique <= 1);
                dominant_columns += usize::from(dominant_numeric(&observed) >= 0.99);
                column_missingness
                    .push(missing_fraction(values.len() - observed.len(), rows.len()));
                sketches.push(ColumnSketch::Numeric(values));
            }
            ColumnKind::Categorical => {
                let mut values = Vec::with_capacity(rows.len());
                let mut counts = BTreeMap::<&str, usize>::new();
                for (sample_row, &source_row) in rows.iter().enumerate() {
                    let value = categorical_profile_value(&table.rows[source_row][column]);
                    if let Some(value) = value {
                        *counts.entry(value).or_default() += 1;
                    } else {
                        missing_cells += 1;
                        row_has_missing[sample_row] = true;
                    }
                    values.push(value);
                }
                let observed = counts.values().sum::<usize>();
                categorical_cells += observed;
                let rare_limit = (observed / 100).max(1);
                rare_categorical_cells += counts
                    .values()
                    .filter(|count| **count <= rare_limit)
                    .sum::<usize>();
                categorical_log_cardinalities.push((counts.len() as f64).ln_1p());
                constant_columns += usize::from(counts.len() <= 1);
                let dominant = counts.values().copied().max().unwrap_or(0);
                let dominant_fraction = if observed == 0 {
                    1.0
                } else {
                    dominant as f64 / observed as f64
                };
                dominant_columns += usize::from(dominant_fraction >= 0.99);
                column_missingness.push(missing_fraction(rows.len() - observed, rows.len()));
                sketches.push(ColumnSketch::Categorical(values));
            }
        }
    }

    let selected_targets: Vec<f64> = rows.iter().map(|row| targets[*row]).collect();
    let column_to_sketch: BTreeMap<usize, usize> = columns
        .iter()
        .copied()
        .enumerate()
        .map(|(sketch, column)| (column, sketch))
        .collect();
    let pair_correlations: Vec<f64> = numeric_pairs
        .iter()
        .map(|(left, right)| {
            let ColumnSketch::Numeric(left_values) = &sketches[column_to_sketch[left]] else {
                unreachable!("pair selection contains only numeric columns")
            };
            let ColumnSketch::Numeric(right_values) = &sketches[column_to_sketch[right]] else {
                unreachable!("pair selection contains only numeric columns")
            };
            optional_correlation(left_values, right_values).abs()
        })
        .collect();

    let mut maximum_numeric_target = 0.0f64;
    let mut maximum_categorical_target = 0.0f64;
    for sketch in &sketches {
        match sketch {
            ColumnSketch::Numeric(values) => {
                maximum_numeric_target = maximum_numeric_target
                    .max(optional_target_correlation(values, &selected_targets).abs());
            }
            ColumnSketch::Categorical(values) => {
                maximum_categorical_target = maximum_categorical_target
                    .max(categorical_correlation_ratio(values, &selected_targets));
            }
        }
    }

    let selected_column_count = columns.len().max(1);
    let selected_cell_count = rows.len().saturating_mul(columns.len()).max(1);
    let numeric_column_count = schema
        .columns
        .iter()
        .filter(|column| column.kind == ColumnKind::Numeric)
        .count();
    let target_stat = match task.kind {
        TaskKind::Regression | TaskKind::Timeseries => {
            skew_and_excess_kurtosis(&selected_targets).0
        }
        TaskKind::Binary => {
            let positive = selected_targets
                .iter()
                .filter(|target| **target == 1.0)
                .count();
            positive.min(selected_targets.len() - positive) as f64
                / selected_targets.len().max(1) as f64
        }
    };
    let statistics = [
        (table.rows.len() as f64).ln_1p(),
        (schema.columns.len() as f64).ln_1p(),
        finite_stat((table.rows.len() as f64 / schema.columns.len() as f64).ln()),
        numeric_column_count as f64 / schema.columns.len() as f64,
        missing_cells as f64 / selected_cell_count as f64,
        row_has_missing.iter().filter(|missing| **missing).count() as f64
            / rows.len().max(1) as f64,
        quantile(&column_missingness, 0.5),
        column_missingness.iter().copied().fold(0.0, f64::max),
        numeric_zero_cells as f64 / observed_numeric_cells.max(1) as f64,
        constant_columns as f64 / selected_column_count as f64,
        dominant_columns as f64 / selected_column_count as f64,
        quantile(&numeric_unique_fractions, 0.5),
        quantile(&numeric_unique_fractions, 0.9),
        quantile(&categorical_log_cardinalities, 0.5),
        categorical_log_cardinalities
            .iter()
            .copied()
            .fold(0.0, f64::max),
        rare_categorical_cells as f64 / categorical_cells.max(1) as f64,
        quantile(&absolute_skews, 0.5),
        quantile(&absolute_kurtoses, 0.5),
        outlier_numeric_cells as f64 / observed_numeric_cells.max(1) as f64,
        duplicate_raw_row_fraction(table),
        quantile(&pair_correlations, 0.5),
        maximum_numeric_target,
        maximum_categorical_target,
        target_stat,
    ]
    .map(finite_stat);
    Ok((statistics, sketches))
}

fn compute_landmarks(
    table: &RawTable,
    schema: &DataSchema,
    task: &TaskSpec,
    targets: &[f64],
    selected_rows: &[usize],
    selected_columns: &[usize],
) -> Result<CheapLandmarksV1> {
    let landmark_table = subset_table(table, selected_rows, selected_columns)?;
    let landmark_schema = DataSchema {
        columns: selected_columns
            .iter()
            .map(|column| schema.columns[*column].clone())
            .collect(),
        // Landmarks contain no row operation.  Treating the deterministic row
        // sample as IID prevents a sampled-out group/order column from
        // changing this cheap baseline contract.
        row_semantics: RowSemantics::Iid,
    };
    landmark_schema.validate()?;
    let landmark_targets: Vec<f64> = selected_rows.iter().map(|row| targets[*row]).collect();
    let encoded_width = landmark_schema
        .columns
        .iter()
        .map(|column| match column.kind {
            ColumnKind::Numeric => 2usize,
            ColumnKind::Categorical => 1usize,
        })
        .sum::<usize>();
    let include_pls = matches!(task.kind, TaskKind::Regression | TaskKind::Timeseries)
        && encoded_width > landmark_targets.len();
    let kinds: Vec<CheapLandmarkKind> = match task.kind {
        TaskKind::Regression | TaskKind::Timeseries => {
            let mut kinds = vec![
                CheapLandmarkKind::RegressionMean,
                CheapLandmarkKind::RegressionRidge,
                CheapLandmarkKind::RegressionShallowTree,
                CheapLandmarkKind::RegressionShallowGam,
            ];
            if include_pls {
                kinds.push(CheapLandmarkKind::RegressionPls);
            }
            kinds
        }
        TaskKind::Binary => vec![
            CheapLandmarkKind::BinaryPrior,
            CheapLandmarkKind::BinaryLogitRidge,
            CheapLandmarkKind::BinaryShallowTree,
            CheapLandmarkKind::BinaryShallowGam,
        ],
    };
    let assignments = stratified_fold_assignments(&landmark_targets, LANDMARK_FOLDS);
    let mut predictions = vec![vec![0.0; landmark_targets.len()]; kinds.len()];

    for fold in 0..LANDMARK_FOLDS {
        let fit_rows: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter_map(|(row, assigned)| (*assigned != fold).then_some(row))
            .collect();
        let valid_rows: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter_map(|(row, assigned)| (*assigned == fold).then_some(row))
            .collect();
        if fit_rows.is_empty() || valid_rows.is_empty() {
            continue;
        }
        let fit_table = subset_table(
            &landmark_table,
            &fit_rows,
            &(0..landmark_schema.columns.len()).collect::<Vec<_>>(),
        )?;
        let valid_table = subset_table(
            &landmark_table,
            &valid_rows,
            &(0..landmark_schema.columns.len()).collect::<Vec<_>>(),
        )?;
        let fit_targets: Vec<f64> = fit_rows.iter().map(|row| landmark_targets[*row]).collect();
        let fallback = fold_baseline(task.kind, &fit_targets);
        let input_stage = landmark_input_stage();
        let input_config = InputConfig::from_stage(Some(&input_stage))?;
        let encoded = FittedInput::fit_transform(
            input_config,
            &landmark_schema,
            &fit_table,
            &fit_targets,
            task.kind,
        );
        let (fitted_input, fit_block) = match encoded {
            Ok(encoded) => encoded,
            Err(_) => {
                for candidate in &mut predictions {
                    for &row in &valid_rows {
                        candidate[row] = fallback;
                    }
                }
                continue;
            }
        };
        let valid_block = match fitted_input.transform(&valid_table) {
            Ok(block) => block,
            Err(_) => {
                for candidate in &mut predictions {
                    for &row in &valid_rows {
                        candidate[row] = fallback;
                    }
                }
                continue;
            }
        };

        for (candidate, kind) in kinds.iter().enumerate() {
            let fold_predictions = match kind {
                CheapLandmarkKind::RegressionMean | CheapLandmarkKind::BinaryPrior => {
                    vec![fallback; valid_rows.len()]
                }
                _ => landmark_head_stage(*kind)
                    .and_then(|stage| {
                        fit_v3_head(&stage, &fit_block.values, &fit_targets, task.kind)
                    })
                    .map(|head| head.predict(&valid_block.values))
                    .unwrap_or_else(|_| vec![fallback; valid_rows.len()]),
            };
            for (&row, prediction) in valid_rows.iter().zip(fold_predictions) {
                predictions[candidate][row] = finite_prediction(task.kind, prediction, fallback);
            }
        }
    }

    let scores = kinds
        .into_iter()
        .zip(predictions)
        .map(|(kind, predictions)| CheapLandmarkScore {
            kind,
            score: finite_stat(task.score(&landmark_targets, &predictions, 0.5)),
        })
        .collect();
    Ok(CheapLandmarksV1 {
        rows: selected_rows.len(),
        folds: LANDMARK_FOLDS,
        scores,
    })
}

fn landmark_input_stage() -> Stage {
    let mut stage = Stage::new("input");
    set_string(&mut stage, "numeric", "median");
    set_string(&mut stage, "missing", "indicator");
    set_string(&mut stage, "categorical", "frequency");
    stage
}

fn landmark_head_stage(kind: CheapLandmarkKind) -> Result<Stage> {
    let mut stage = match kind {
        CheapLandmarkKind::RegressionRidge => Stage::new("ridge"),
        CheapLandmarkKind::RegressionShallowTree | CheapLandmarkKind::BinaryShallowTree => {
            let mut stage = Stage::new("extra_trees");
            set_integer(&mut stage, "trees", 16);
            set_integer(&mut stage, "depth", 2);
            set_integer(&mut stage, "min_leaf", 2);
            set_integer(&mut stage, "seed", 7);
            stage
        }
        CheapLandmarkKind::RegressionShallowGam | CheapLandmarkKind::BinaryShallowGam => {
            let mut stage = Stage::new("gam_boost");
            set_integer(&mut stage, "rounds", 16);
            set_integer(&mut stage, "bins", 8);
            set_float(&mut stage, "lr", 0.05);
            stage
        }
        CheapLandmarkKind::RegressionPls => {
            let mut stage = Stage::new("pls");
            set_integer(&mut stage, "k", 8);
            stage
        }
        CheapLandmarkKind::BinaryLogitRidge => {
            let mut stage = Stage::new("logit_ridge");
            set_float(&mut stage, "l2", 0.02);
            stage
        }
        CheapLandmarkKind::RegressionMean | CheapLandmarkKind::BinaryPrior => {
            bail!("baseline landmarks do not have fitted head stages")
        }
    };
    if matches!(
        kind,
        CheapLandmarkKind::BinaryLogitRidge
            | CheapLandmarkKind::BinaryShallowTree
            | CheapLandmarkKind::BinaryShallowGam
    ) {
        set_string(&mut stage, "class_weight", "none");
    }
    Ok(stage)
}

fn fold_baseline(task: TaskKind, targets: &[f64]) -> f64 {
    let mean = targets.iter().sum::<f64>() / targets.len().max(1) as f64;
    match task {
        TaskKind::Regression | TaskKind::Timeseries => finite_stat(mean),
        TaskKind::Binary => mean.clamp(1e-6, 1.0 - 1e-6),
    }
}

fn finite_prediction(task: TaskKind, prediction: f64, fallback: f64) -> f64 {
    if !prediction.is_finite() {
        return fallback;
    }
    match task {
        TaskKind::Regression | TaskKind::Timeseries => prediction,
        TaskKind::Binary => prediction.clamp(0.0, 1.0),
    }
}

fn set_integer(stage: &mut Stage, name: &str, value: i64) {
    stage
        .params
        .insert(name.to_string(), ParamValue::Int(value));
}

fn set_float(stage: &mut Stage, name: &str, value: f64) {
    stage
        .params
        .insert(name.to_string(), ParamValue::Float(value));
}

fn set_string(stage: &mut Stage, name: &str, value: &str) {
    stage
        .params
        .insert(name.to_string(), ParamValue::Str(value.to_string()));
}

fn deterministic_indices(total: usize, limit: usize) -> Vec<usize> {
    let selected = total.min(limit);
    if selected == total {
        return (0..total).collect();
    }
    (0..selected)
        .map(|index| ((index as u128 * total as u128) / selected as u128) as usize)
        .collect()
}

fn deterministic_numeric_pairs(columns: &[usize], limit: usize) -> Vec<(usize, usize)> {
    let mut all = Vec::new();
    for left in 0..columns.len() {
        for right in (left + 1)..columns.len() {
            all.push((columns[left], columns[right]));
        }
    }
    deterministic_indices(all.len(), limit)
        .into_iter()
        .map(|index| all[index])
        .collect()
}

fn subset_table(table: &RawTable, rows: &[usize], columns: &[usize]) -> Result<RawTable> {
    RawTable::new(
        columns
            .iter()
            .map(|column| table.headers[*column].clone())
            .collect(),
        rows.iter()
            .map(|row| {
                columns
                    .iter()
                    .map(|column| table.rows[*row][*column].clone())
                    .collect()
            })
            .collect(),
    )
}

fn parse_numeric_profile(raw: &str) -> Result<Option<f64>> {
    let text = raw.trim();
    if text.is_empty()
        || text.eq_ignore_ascii_case("na")
        || text.eq_ignore_ascii_case("nan")
        || text.eq_ignore_ascii_case("null")
    {
        return Ok(None);
    }
    let value = text
        .parse::<f64>()
        .with_context(|| format!("invalid numeric value '{raw}'"))?;
    Ok(value.is_finite().then_some(value))
}

fn categorical_profile_value(raw: &str) -> Option<&str> {
    let text = raw.trim();
    (!text.is_empty() && !text.eq_ignore_ascii_case("na") && !text.eq_ignore_ascii_case("null"))
        .then_some(raw)
}

fn numeric_unique_count(values: &[f64]) -> usize {
    values
        .iter()
        .map(|value| normalized_bits(*value))
        .collect::<BTreeSet<_>>()
        .len()
}

fn normalized_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

fn dominant_numeric(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let mut counts = BTreeMap::<u64, usize>::new();
    for value in values {
        *counts.entry(normalized_bits(*value)).or_default() += 1;
    }
    counts.values().copied().max().unwrap_or(0) as f64 / values.len() as f64
}

fn missing_fraction(missing: usize, total: usize) -> f64 {
    missing as f64 / total.max(1) as f64
}

fn quantile(values: &[f64], probability: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    finite_stat(quantile_sorted(&sorted, probability))
}

fn skew_and_excess_kurtosis(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        return (0.0, 0.0);
    }
    let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return (0.0, 0.0);
    }
    let scaled: Vec<f64> = values.iter().map(|value| value / scale).collect();
    let mean = scaled.iter().sum::<f64>() / scaled.len() as f64;
    let mut second = 0.0;
    let mut third = 0.0;
    let mut fourth = 0.0;
    for value in scaled {
        let centered = value - mean;
        let square = centered * centered;
        second += square;
        third += square * centered;
        fourth += square * square;
    }
    second /= values.len() as f64;
    third /= values.len() as f64;
    fourth /= values.len() as f64;
    if second <= 1e-30 {
        return (0.0, 0.0);
    }
    (
        finite_stat(third / second.powf(1.5)),
        finite_stat(fourth / (second * second) - 3.0),
    )
}

fn three_iqr_outlier_count(values: &[f64]) -> usize {
    if values.len() < 2 {
        return 0;
    }
    let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return 0;
    }
    let mut scaled: Vec<f64> = values.iter().map(|value| value / scale).collect();
    scaled.sort_by(f64::total_cmp);
    let q1 = quantile_sorted(&scaled, 0.25);
    let q3 = quantile_sorted(&scaled, 0.75);
    let iqr = q3 - q1;
    let lower = q1 - 3.0 * iqr;
    let upper = q3 + 3.0 * iqr;
    scaled
        .iter()
        .filter(|value| **value < lower || **value > upper)
        .count()
}

fn optional_correlation(left: &[Option<f64>], right: &[Option<f64>]) -> f64 {
    let pairs: Vec<(f64, f64)> = left
        .iter()
        .zip(right)
        .filter_map(|(left, right)| Some((left.as_ref()?, right.as_ref()?)))
        .map(|(left, right)| (*left, *right))
        .collect();
    stable_correlation(&pairs)
}

fn optional_target_correlation(values: &[Option<f64>], targets: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = values
        .iter()
        .zip(targets)
        .filter_map(|(value, target)| value.map(|value| (value, *target)))
        .collect();
    stable_correlation(&pairs)
}

fn stable_correlation(pairs: &[(f64, f64)]) -> f64 {
    if pairs.len() < 2 {
        return 0.0;
    }
    let left_scale = pairs.iter().map(|(left, _)| left.abs()).fold(0.0, f64::max);
    let right_scale = pairs
        .iter()
        .map(|(_, right)| right.abs())
        .fold(0.0, f64::max);
    if left_scale == 0.0 || right_scale == 0.0 {
        return 0.0;
    }
    let left_mean =
        pairs.iter().map(|(left, _)| left / left_scale).sum::<f64>() / pairs.len() as f64;
    let right_mean = pairs
        .iter()
        .map(|(_, right)| right / right_scale)
        .sum::<f64>()
        / pairs.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left, right) in pairs {
        let left = left / left_scale - left_mean;
        let right = right / right_scale - right_mean;
        covariance += left * right;
        left_variance += left * left;
        right_variance += right * right;
    }
    if left_variance <= 1e-30 || right_variance <= 1e-30 {
        0.0
    } else {
        finite_stat(covariance / (left_variance * right_variance).sqrt()).clamp(-1.0, 1.0)
    }
}

fn categorical_correlation_ratio(values: &[Option<&str>], targets: &[f64]) -> f64 {
    let observed: Vec<(&str, f64)> = values
        .iter()
        .zip(targets)
        .filter_map(|(value, target)| value.map(|value| (value, *target)))
        .collect();
    if observed.len() < 2 {
        return 0.0;
    }
    let scale = observed
        .iter()
        .map(|(_, target)| target.abs())
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let mean = observed
        .iter()
        .map(|(_, target)| target / scale)
        .sum::<f64>()
        / observed.len() as f64;
    let total = observed
        .iter()
        .map(|(_, target)| {
            let centered = target / scale - mean;
            centered * centered
        })
        .sum::<f64>();
    if total <= 1e-30 {
        return 0.0;
    }
    let mut groups = BTreeMap::<&str, (usize, f64)>::new();
    for (value, target) in observed {
        let entry = groups.entry(value).or_default();
        entry.0 += 1;
        entry.1 += target / scale;
    }
    let between = groups
        .values()
        .map(|(count, sum)| {
            let delta = sum / *count as f64 - mean;
            *count as f64 * delta * delta
        })
        .sum::<f64>();
    finite_stat((between / total).clamp(0.0, 1.0).sqrt())
}

fn duplicate_raw_row_fraction(table: &RawTable) -> f64 {
    let mut unique = BTreeSet::<[u8; 32]>::new();
    for row in &table.rows {
        let mut hasher = Sha256::new();
        hasher.update((row.len() as u64).to_le_bytes());
        for field in row {
            hasher.update((field.len() as u64).to_le_bytes());
            hasher.update(field.as_bytes());
        }
        unique.insert(hasher.finalize().into());
    }
    (table.rows.len() - unique.len()) as f64 / table.rows.len().max(1) as f64
}

fn finite_stat(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::super::schema::ColumnSpec;
    use super::super::types::Objective;
    use super::*;

    fn mixed_schema() -> DataSchema {
        DataSchema {
            columns: vec![
                ColumnSpec::numeric("number"),
                ColumnSpec::categorical("category"),
                ColumnSpec::numeric("constant"),
            ],
            row_semantics: RowSemantics::Iid,
        }
    }

    fn mixed_table() -> RawTable {
        RawTable::new(
            vec![
                "number".to_string(),
                "category".to_string(),
                "constant".to_string(),
            ],
            vec![
                vec!["0".to_string(), "a".to_string(), "7".to_string()],
                vec!["1".to_string(), "a".to_string(), "7".to_string()],
                vec!["2".to_string(), "b".to_string(), "7".to_string()],
                vec!["NA".to_string(), "nan".to_string(), "7".to_string()],
                vec!["4".to_string(), "null".to_string(), "7".to_string()],
                vec!["5".to_string(), "c".to_string(), "7".to_string()],
            ],
        )
        .unwrap()
    }

    #[test]
    fn profile_has_exactly_24_named_finite_statistics() {
        assert_eq!(PROFILE_STAT_NAMES.len(), PROFILE_STAT_COUNT);
        assert_eq!(PROFILE_STAT_NAMES.iter().collect::<BTreeSet<_>>().len(), 24);
        let profile = build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        assert_eq!(profile.statistics().len(), 24);
        assert!(profile.statistics().iter().all(|value| value.is_finite()));
        assert_eq!(
            profile.statistic("numeric_column_fraction"),
            Some(2.0 / 3.0)
        );
        // Numeric NA and categorical null are missing; categorical "nan" is
        // deliberately a real category.
        assert_eq!(profile.statistic("missing_cell_fraction"), Some(2.0 / 18.0));
        assert!(profile.validate().is_ok());
    }

    #[test]
    fn repeated_profiles_are_byte_hash_deterministic() {
        let table = mixed_table();
        let schema = mixed_schema();
        let task = TaskSpec::regression(Objective::R2).unwrap();
        let targets = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        let first = build_dataset_profile(&table, &schema, &task, &targets).unwrap();
        let second = build_dataset_profile(&table, &schema, &task, &targets).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
        assert_eq!(
            first.sketch().selection_sha256,
            second.sketch().selection_sha256
        );
    }

    #[test]
    fn deterministic_selection_respects_every_bound() {
        for total in [0, 1, 2, 255, 256, 257, 4_095, 4_096, 4_097, 100_000] {
            let selected = deterministic_indices(total, MAX_PROFILE_ROWS);
            assert_eq!(selected.len(), total.min(MAX_PROFILE_ROWS));
            assert!(selected.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(selected.iter().all(|index| *index < total));
        }
        let columns: Vec<usize> = (0..MAX_PROFILE_COLUMNS).collect();
        let pairs = deterministic_numeric_pairs(&columns, MAX_PROFILE_NUMERIC_PAIRS);
        assert_eq!(pairs.len(), MAX_PROFILE_NUMERIC_PAIRS);
        assert!(pairs.iter().all(|(left, right)| left < right));
        assert_eq!(
            pairs,
            deterministic_numeric_pairs(&columns, MAX_PROFILE_NUMERIC_PAIRS)
        );
    }

    #[test]
    fn duplicate_hashing_scans_rows_outside_the_profile_sketch() {
        let mut rows: Vec<Vec<String>> = (0..(MAX_PROFILE_ROWS + 1))
            .map(|row| vec![row.to_string()])
            .collect();
        rows[MAX_PROFILE_ROWS][0] = "0".to_string();
        let table = RawTable::new(vec!["x".to_string()], rows).unwrap();
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("x")],
            row_semantics: RowSemantics::Iid,
        };
        let targets: Vec<f64> = (0..table.rows.len()).map(|row| row as f64).collect();
        let profile = build_dataset_profile(
            &table,
            &schema,
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &targets,
        )
        .unwrap();
        assert_eq!(
            profile.statistic("duplicate_raw_row_fraction"),
            Some(1.0 / (MAX_PROFILE_ROWS + 1) as f64)
        );
        assert_eq!(profile.sketch().selected_rows, MAX_PROFILE_ROWS);
        assert_eq!(profile.sketch().source_rows, MAX_PROFILE_ROWS + 1);
    }

    #[test]
    fn regression_and_binary_landmark_catalogs_are_fixed_and_finite() {
        let regression = build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        assert_eq!(regression.landmarks().scores.len(), 4);
        assert!(regression
            .landmarks()
            .scores
            .iter()
            .all(|entry| entry.score.is_finite()));

        let binary = build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::binary(Objective::LogLoss).unwrap(),
            &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
        assert_eq!(
            binary
                .landmarks()
                .scores
                .iter()
                .map(|entry| entry.kind)
                .collect::<Vec<_>>(),
            vec![
                CheapLandmarkKind::BinaryPrior,
                CheapLandmarkKind::BinaryLogitRidge,
                CheapLandmarkKind::BinaryShallowTree,
                CheapLandmarkKind::BinaryShallowGam,
            ]
        );
        assert!(binary
            .landmarks()
            .scores
            .iter()
            .all(|entry| entry.score.is_finite()));
    }

    #[test]
    fn pls_landmark_is_included_only_when_encoded_width_exceeds_rows() {
        let columns: Vec<ColumnSpec> = (0..4)
            .map(|column| ColumnSpec::numeric(format!("x{column}")))
            .collect();
        let headers: Vec<String> = columns.iter().map(|column| column.name.clone()).collect();
        let table = RawTable::new(
            headers,
            vec![
                vec!["0".to_string(); 4],
                vec!["1".to_string(); 4],
                vec!["2".to_string(); 4],
                vec!["3".to_string(); 4],
                vec!["4".to_string(); 4],
                vec!["5".to_string(); 4],
            ],
        )
        .unwrap();
        let schema = DataSchema {
            columns,
            row_semantics: RowSemantics::Iid,
        };
        let profile = build_dataset_profile(
            &table,
            &schema,
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        assert!(profile
            .landmarks()
            .scores
            .iter()
            .any(|entry| entry.kind == CheapLandmarkKind::RegressionPls));
    }

    #[test]
    fn extreme_finite_values_still_produce_finite_profile_values() {
        let table = RawTable::new(
            vec!["x".to_string(), "y".to_string()],
            vec![
                vec![f64::MAX.to_string(), (-f64::MAX).to_string()],
                vec![(-f64::MAX).to_string(), f64::MAX.to_string()],
                vec!["0".to_string(), "0".to_string()],
            ],
        )
        .unwrap();
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("x"), ColumnSpec::numeric("y")],
            row_semantics: RowSemantics::Iid,
        };
        let profile = build_dataset_profile(
            &table,
            &schema,
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[f64::MAX, -f64::MAX, 0.0],
        )
        .unwrap();
        assert!(profile.statistics().iter().all(|value| value.is_finite()));
        assert!(profile
            .landmarks()
            .scores
            .iter()
            .all(|entry| entry.score.is_finite()));
    }

    #[test]
    fn invalid_target_contracts_are_rejected() {
        assert!(build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::binary(Objective::LogLoss).unwrap(),
            &[0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        )
        .is_err());
        assert!(build_dataset_profile(
            &mixed_table(),
            &mixed_schema(),
            &TaskSpec::regression(Objective::SquaredError).unwrap(),
            &[0.0],
        )
        .is_err());
    }
}
