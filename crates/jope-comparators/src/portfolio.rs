//! Shared portfolio machinery for the AutoML-style comparators: member specifications, seeded
//! inner splits of the training rows, out-of-fold predictions and weighted ensembles.
//!
//! Every split here partitions the **training rows only**. The run's held-out rows never reach a
//! comparator, so inner validation cannot leak them.

use crate::deadline::Deadline;
use crate::ensemble::caruana_weights;
use crate::error::{ComparatorError, ComparatorResult};
use crate::forest::{ForestModel, ForestParams};
use crate::histgbm::{HistGbmModel, HistGbmParams};
use crate::linear::{RidgeModel, RidgeParams, MAX_RIDGE_FEATURES};
use crate::matrix::FeatureMatrix;
use crate::rng::{stream_seed, SplitMix64};

/// `MemberSpec` is an enum for the portfolio member configuration contract in this module.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MemberSpec {
    Forest(ForestParams),
    Boosting(HistGbmParams),
    Ridge(RidgeParams),
}

/// `FittedMember` is an enum for the fitted portfolio member contract in this module.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FittedMember {
    Forest(ForestModel),
    Boosting(HistGbmModel),
    Ridge(RidgeModel),
}

impl MemberSpec {
    /// Ridge members are skipped on tables wider than the bounded solve allows.
    pub(crate) fn applicable(&self, cols: usize) -> bool {
        !matches!(self, Self::Ridge(_)) || cols <= MAX_RIDGE_FEATURES
    }

    pub(crate) fn fit(
        &self,
        features: FeatureMatrix<'_>,
        targets: &[f64],
        seed: u64,
        deadline: &Deadline,
    ) -> ComparatorResult<FittedMember> {
        deadline.check()?;
        Ok(match self {
            Self::Forest(params) => {
                FittedMember::Forest(ForestModel::fit(features, targets, params, seed, deadline)?)
            }
            Self::Boosting(params) => FittedMember::Boosting(HistGbmModel::fit(
                features, targets, params, seed, deadline,
            )?),
            Self::Ridge(params) => FittedMember::Ridge(RidgeModel::fit(features, targets, params)?),
        })
    }
}

impl FittedMember {
    pub(crate) fn predict(&self, features: FeatureMatrix<'_>) -> ComparatorResult<Vec<f64>> {
        match self {
            Self::Forest(model) => model.predict(features),
            Self::Boosting(model) => model.predict(features),
            Self::Ridge(model) => model.predict(features),
        }
    }

    pub(crate) fn model_bytes(&self) -> u64 {
        match self {
            Self::Forest(model) => model.model_bytes(),
            Self::Boosting(model) => model.model_bytes(),
            Self::Ridge(model) => model.model_bytes(),
        }
    }
}

/// `OwnedRows` is a struct for the copied row subset contract in this module: the feature values
/// and targets of selected training rows, in selection order.
pub(crate) struct OwnedRows {
    values: Vec<f64>,
    targets: Vec<f64>,
    cols: usize,
}

impl OwnedRows {
    pub(crate) fn select(features: FeatureMatrix<'_>, targets: &[f64], rows: &[usize]) -> Self {
        let mut values = Vec::with_capacity(rows.len() * features.cols());
        for row in rows {
            values.extend_from_slice(features.row(*row));
        }
        Self {
            values,
            targets: rows.iter().map(|row| targets[*row]).collect(),
            cols: features.cols(),
        }
    }

    pub(crate) fn matrix(&self) -> ComparatorResult<FeatureMatrix<'_>> {
        FeatureMatrix::new(&self.values, self.targets.len(), self.cols)
    }

    pub(crate) fn targets(&self) -> &[f64] {
        &self.targets
    }
}

/// A seeded permutation of `0..rows`.
fn shuffled(rows: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows).collect();
    SplitMix64::new(seed).shuffle(&mut order);
    order
}

/// Seeded (fit, validation) split of the training rows; both sides keep at least one row.
pub(crate) fn holdout_split(
    rows: usize,
    fraction: f64,
    seed: u64,
) -> ComparatorResult<(Vec<usize>, Vec<usize>)> {
    if rows < 2 {
        return Err(ComparatorError::EmptyFit);
    }
    let mut order = shuffled(rows, seed);
    let held = ((rows as f64 * fraction).round() as usize).clamp(1, rows - 1);
    let fit = order.split_off(held);
    Ok((fit, order))
}

/// Seeded k-fold assignment of the training rows: `folds` disjoint validation sets.
pub(crate) fn fold_assignment(rows: usize, folds: usize, seed: u64) -> Vec<Vec<usize>> {
    let folds = folds.clamp(2, rows.max(2));
    let mut assignment = vec![Vec::new(); folds];
    for (position, row) in shuffled(rows, seed).into_iter().enumerate() {
        assignment[position % folds].push(row);
    }
    assignment
}

/// Out-of-fold predictions of `member` for every training row.
pub(crate) fn out_of_fold(
    member: &MemberSpec,
    features: FeatureMatrix<'_>,
    targets: &[f64],
    folds: &[Vec<usize>],
    seed: u64,
    deadline: &Deadline,
) -> ComparatorResult<Vec<f64>> {
    let mut predictions = vec![0.0; features.rows()];
    for (index, validation) in folds.iter().enumerate() {
        let mut in_fold = vec![false; features.rows()];
        for row in validation {
            in_fold[*row] = true;
        }
        let train_rows: Vec<usize> = (0..features.rows()).filter(|row| !in_fold[*row]).collect();
        let train = OwnedRows::select(features, targets, &train_rows);
        let held = OwnedRows::select(features, targets, validation);
        let fitted = member.fit(
            train.matrix()?,
            train.targets(),
            stream_seed(seed, index as u64),
            deadline,
        )?;
        for (row, prediction) in validation.iter().zip(fitted.predict(held.matrix()?)?) {
            predictions[*row] = prediction;
        }
    }
    Ok(predictions)
}

/// `WeightedPortfolio` is a struct for the fitted weighted ensemble contract in this module:
/// members refitted on every training row, each with its non-zero selection weight.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WeightedPortfolio {
    members: Vec<(f64, FittedMember)>,
}

impl WeightedPortfolio {
    /// Fit every applicable member on a seeded inner split, choose weights by greedy ensemble
    /// selection on the inner validation rows, then refit the chosen members on all rows.
    pub(crate) fn fit(
        specs: &[MemberSpec],
        features: FeatureMatrix<'_>,
        targets: &[f64],
        validation_fraction: f64,
        ensemble_rounds: usize,
        seed: u64,
        deadline: &Deadline,
    ) -> ComparatorResult<Self> {
        let specs: Vec<&MemberSpec> = specs
            .iter()
            .filter(|spec| spec.applicable(features.cols()))
            .collect();
        let (fit_rows, validation_rows) =
            holdout_split(features.rows(), validation_fraction, stream_seed(seed, 0))?;
        let inner = OwnedRows::select(features, targets, &fit_rows);
        let held = OwnedRows::select(features, targets, &validation_rows);
        let mut validation_predictions = Vec::with_capacity(specs.len());
        for (index, spec) in specs.iter().enumerate() {
            let member_seed = stream_seed(seed, index as u64 + 1);
            let fitted = spec.fit(inner.matrix()?, inner.targets(), member_seed, deadline)?;
            validation_predictions.push(fitted.predict(held.matrix()?)?);
        }
        let weights = caruana_weights(&validation_predictions, held.targets(), ensemble_rounds)?;
        let mut members = Vec::new();
        for (index, (spec, weight)) in specs.iter().zip(weights).enumerate() {
            if weight > 0.0 {
                let member_seed = stream_seed(seed, index as u64 + 1);
                members.push((weight, spec.fit(features, targets, member_seed, deadline)?));
            }
        }
        Ok(Self { members })
    }

    pub(crate) fn predict(&self, features: FeatureMatrix<'_>) -> ComparatorResult<Vec<f64>> {
        let mut total = vec![0.0; features.rows()];
        for (weight, member) in &self.members {
            for (sum, prediction) in total.iter_mut().zip(member.predict(features)?) {
                *sum += weight * prediction;
            }
        }
        Ok(total)
    }

    pub(crate) fn model_bytes(&self) -> u64 {
        self.members
            .iter()
            .map(|(_, member)| member.model_bytes() + 8)
            .sum()
    }

    pub(crate) fn member_count(&self) -> usize {
        self.members.len()
    }
}
