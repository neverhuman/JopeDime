//! Bounded automatic selection, `--method auto` (SPECIFICATION.md §6.3).
//!
//! `jope.runtime-select/v1`, a new runtime behaviour with its own identity (it is not the
//! research panel's search or a promoted proposal policy):
//!
//! 1. Candidates: the task's native roster in registry order, or the `--candidates` subset.
//! 2. Folds: `stratified-round-robin/v1` — the k-th row of each class (each row, for regression)
//!    goes to fold `k mod folds`; deterministic, no seed. `folds = min(requested, rows / 2,
//!    smallest class count)`, at least 2.
//! 3. Each candidate is fitted on every training part and predicts its held-out part through the
//!    same [`fit_native`]/[`predict_bound`] path as a final fit. All preprocessing is refitted per
//!    part: held-out rows contribute no labels, statistics or vocabulary.
//! 4. Loss on the pooled out-of-fold predictions, matching the panel's primary metric: RMSE,
//!    `1 - AUROC`, or log loss with probabilities clipped below at 1e-15.
//! 5. The lowest loss wins; an exact tie goes to the earlier roster entry. A candidate that fails
//!    any fold, by error or by panic, is recorded with its error and not selected.
//! 6. The deadline bounds selection: it is checked before every fold fit, and candidates not
//!    reached are recorded as skipped. The chosen recipe is then refitted once on all rows; that
//!    refit always runs and its time is charged to the run.
//!
//! `jope.runtime-select/v2` (regression only) replaces step 4's loss with a portable objective
//! ([`crate::fitness`]) and may add chronological forward folds and an objective-aware output
//! calibration ([`crate::calibration`]). Each candidate's score is the better of its raw
//! out-of-fold objective and its cross-fitted calibrated objective; the winner's final output map
//! is learned on all of its out-of-fold predictions and stored with the fitted model.
use crate::budget::Budget;
use crate::calibration::{self, Calibration, OutputTransform};
use crate::fitness::{Columns, Direction, Fitness};
use crate::methods::native::{fit_native, predict_bound, Predictions};
use crate::methods::RustMethod;
use crate::table::Training;
use crate::task::{self, TaskKind};
use crate::Failure;
use anyhow::{Context, Result};
use feat_math::advanced_v3::RawTable;
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub const SELECTION_POLICY: &str = "jope.runtime-select/v1";
/// Objective, forward-validation and calibration selection.
pub const SELECTION_POLICY_V2: &str = "jope.runtime-select/v2";
pub const FORWARD_FOLD_POLICY: &str = "expanding-window-gap/v1";
pub const FOLD_POLICY: &str = "stratified-round-robin/v1";
pub const DEFAULT_FOLDS: usize = 5;
/// Probability floor of the log loss, the panel's epsilon.
pub const PROBABILITY_EPSILON: f64 = 1e-15;

mod metrics;
pub use metrics::{folds, loss, loss_name};

/// Forward folds train only on earlier rows, with an explicit embargo of `gap` rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Validation {
    Iid {},
    Forward { gap: usize },
}

impl Default for Validation {
    fn default() -> Self {
        Self::Iid {}
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CandidateOutcome {
    pub id: String,
    /// `scored`, `failed` or `skipped`.
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f64>,
    /// Natural value (before maximization is negated); with an objective only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitness_value: Option<f64>,
    /// With calibration: the objective loss of the raw out-of-fold predictions (absent when the
    /// raw predictions are outside the objective's domain).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncalibrated_loss: Option<f64>,
    /// With calibration: the cross-fitted calibrated loss.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated_loss: Option<f64>,
    /// Whether `loss` is the calibrated one.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub calibrated: bool,
    pub folds_completed: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Selection {
    pub policy: &'static str,
    pub fold_policy: &'static str,
    pub folds: usize,
    pub metric: String,
    pub loss: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitness: Option<crate::fitness::Fitness>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitness_sha256: Option<String>,
    /// `v2` only (an objective or forward validation): a `v1` record keeps its original shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<Validation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scored_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_rows: Option<usize>,
    /// Present when the selection learned output maps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationRecord>,
    pub selected: String,
    pub candidates: Vec<CandidateOutcome>,
}

/// How output maps were learned, and the map the selected candidate keeps.
#[derive(Clone, Debug, Serialize)]
pub struct CalibrationRecord {
    pub family: Calibration,
    pub method: &'static str,
    /// The selected candidate's final map, learned on all its out-of-fold predictions; absent
    /// when its raw predictions scored at least as well.
    pub output_transform: Option<OutputTransform>,
}

fn subset(table: &RawTable, rows: &[usize]) -> Result<RawTable> {
    RawTable::new(
        table.headers.clone(),
        rows.iter().map(|r| table.rows[*r].clone()).collect(),
    )
}

/// Out-of-fold predictions of one candidate, assembled in row order.
fn out_of_fold(
    method: &RustMethod,
    training: &Training,
    assignment: &[usize],
    k: usize,
    budget: &Budget,
    completed: &mut usize,
    validation: &Validation,
) -> Result<(Predictions, Vec<usize>)> {
    let n = training.targets.len();
    let mut values: Vec<Option<Vec<f64>>> = vec![None; n];
    for fold in 0..k {
        budget.check(&format!("fold {fold} of {}", method.id))?;
        let valid_rows: Vec<usize> = (0..n).filter(|r| assignment[*r] == fold).collect();
        let train_rows: Vec<usize> = match validation {
            Validation::Iid {} => (0..n).filter(|r| assignment[*r] != fold).collect(),
            Validation::Forward { gap } => (0..valid_rows[0].saturating_sub(*gap)).collect(),
        };
        anyhow::ensure!(
            train_rows.len() >= 2,
            "forward fold needs at least two training rows after gap"
        );
        let train = subset(&training.table, &train_rows)?;
        let targets: Vec<f64> = train_rows.iter().map(|r| training.targets[*r]).collect();
        let model = fit_native(
            method,
            training.kind,
            &training.schema,
            &training.classes,
            &train,
            &targets,
        )
        .with_context(|| format!("fold {fold}"))?;
        let predicted = predict_bound(
            &model,
            &subset(&training.table, &valid_rows)?,
            &training.classes,
        )?;
        match predicted {
            Predictions::Values(v) => {
                for (row, value) in valid_rows.iter().zip(v) {
                    values[*row] = Some(vec![value]);
                }
            }
            Predictions::Probabilities { values: v, .. } => {
                for (row, value) in valid_rows.iter().zip(v) {
                    values[*row] = Some(value);
                }
            }
        }
        *completed += 1;
    }
    let mut scored = Vec::new();
    let mut rows = Vec::new();
    for (index, value) in values.into_iter().enumerate() {
        if assignment[index] == usize::MAX {
            continue;
        }
        rows.push(value.context("a scoring row was in no fold")?);
        scored.push(index);
    }
    Ok((
        if training.kind == TaskKind::Regression {
            Predictions::Values(rows.into_iter().map(|r| r[0]).collect())
        } else {
            Predictions::Probabilities {
                classes: training.classes.clone(),
                values: rows,
            }
        },
        scored,
    ))
}

/// The effective fold count, or an input failure when selection is impossible.
pub fn effective_folds(training: &Training, requested: usize) -> Result<usize> {
    let n = training.targets.len();
    if n < 4 {
        return Err(Failure::input(format!(
            "automatic selection needs at least 4 rows; the table has {n} (name a --method)"
        )));
    }
    let mut k = requested.max(2).min(n / 2);
    if training.kind != TaskKind::Regression {
        let mut counts = vec![0usize; training.classes.len()];
        for t in &training.targets {
            counts[*t as usize] += 1;
        }
        let smallest = counts.iter().copied().min().unwrap_or(0);
        if smallest < 2 {
            let class = &training.classes[counts.iter().position(|c| *c == smallest).unwrap_or(0)];
            return Err(Failure::input(format!(
                "class {class:?} has {smallest} row; automatic selection needs at least 2 per class \
                 (name a --method)"
            )));
        }
        k = k.min(smallest);
    }
    Ok(k.max(2))
}

/// Run the selection plan; returns the record and the winning recipe.
pub fn select(
    training: &Training,
    candidates: &[RustMethod],
    requested_folds: usize,
    budget: &Budget,
) -> Result<(Selection, RustMethod)> {
    select_with_fitness(
        training,
        candidates,
        requested_folds,
        budget,
        None,
        &Validation::Iid {},
        Calibration::None,
    )
}

mod selection;
pub use selection::{select, select_with_fitness};

#[cfg(test)]
mod tests;
