//! H2O AutoML-style stacked ensemble: a GLM, a default GBM, a random forest (DRF), extremely
//! randomized trees (XRT) and a small seeded GBM grid produce out-of-fold predictions on the
//! training rows; a non-negative linear metalearner (H2O's default GLM metalearner constraint)
//! combines them, and the base models with non-zero weight are refitted on every row.
//!
//! H2O's DeepLearning and native XGBoost members are omitted.

use crate::deadline::Deadline;
use crate::error::{ComparatorError, ComparatorResult};
use crate::forest::ForestParams;
use crate::histgbm::HistGbmParams;
use crate::linear::RidgeParams;
use crate::matrix::{mean, FeatureMatrix};
use crate::portfolio::{fold_assignment, out_of_fold, FittedMember, MemberSpec};
use crate::presets;
use crate::rng::{stream_seed, SplitMix64};
use crate::tier::TierBudget;

/// Coordinate-descent sweeps of the non-negative metalearner.
const METALEARNER_SWEEPS: usize = 200;

/// `StackedEnsemble` is a struct for the fitted H2O-style stack contract in this module: the
/// metalearner intercept and the refitted base models with their non-negative weights.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StackedEnsemble {
    intercept: f64,
    members: Vec<(f64, FittedMember)>,
}

fn base_models(budget: &TierBudget, seed: u64) -> Vec<MemberSpec> {
    let mut members = vec![
        MemberSpec::Ridge(RidgeParams { alpha: 1.0 }),
        MemberSpec::Boosting(presets::h2o_style(budget.boosting_rounds)),
        MemberSpec::Forest(ForestParams::random_forest(budget.member_forest_trees)),
        MemberSpec::Forest(ForestParams::extra_trees(budget.member_forest_trees)),
    ];
    let mut rng = SplitMix64::new(stream_seed(seed, 7));
    for _ in 0..budget.search_candidates / 4 {
        members.push(MemberSpec::Boosting(HistGbmParams {
            max_depth: Some(3 + rng.below(6)),
            learning_rate: [0.03, 0.05, 0.1, 0.2][rng.below(4)],
            min_samples_leaf: [1, 5, 10, 20][rng.below(4)],
            ..presets::h2o_style(budget.boosting_rounds)
        }));
    }
    members
}

/// Least squares with an intercept and non-negative weights, by cyclic coordinate descent.
fn nonnegative_weights(columns: &[Vec<f64>], targets: &[f64]) -> (f64, Vec<f64>) {
    let target_mean = mean(targets);
    let centered: Vec<Vec<f64>> = columns
        .iter()
        .map(|column| {
            let center = mean(column);
            column.iter().map(|value| value - center).collect()
        })
        .collect();
    let norms: Vec<f64> = centered
        .iter()
        .map(|column| column.iter().map(|value| value * value).sum())
        .collect();
    let mut residual: Vec<f64> = targets.iter().map(|target| target - target_mean).collect();
    let mut weights = vec![0.0; columns.len()];
    for _ in 0..METALEARNER_SWEEPS {
        for (index, column) in centered.iter().enumerate() {
            if norms[index] <= 0.0 {
                continue;
            }
            let correlation: f64 = column.iter().zip(&residual).map(|(x, r)| x * r).sum();
            let updated = (weights[index] + correlation / norms[index]).max(0.0);
            let delta = updated - weights[index];
            if delta != 0.0 {
                for (r, x) in residual.iter_mut().zip(column) {
                    *r -= delta * x;
                }
                weights[index] = updated;
            }
        }
    }
    let intercept = target_mean
        - columns
            .iter()
            .zip(&weights)
            .map(|(column, weight)| weight * mean(column))
            .sum::<f64>();
    (intercept, weights)
}

impl StackedEnsemble {
    pub(crate) fn fit(
        features: FeatureMatrix<'_>,
        targets: &[f64],
        budget: &TierBudget,
        seed: u64,
        deadline: &Deadline,
    ) -> ComparatorResult<Self> {
        let specs: Vec<MemberSpec> = base_models(budget, seed)
            .into_iter()
            .filter(|spec| spec.applicable(features.cols()))
            .collect();
        let folds = fold_assignment(features.rows(), budget.folds, stream_seed(seed, 0));
        let mut columns = Vec::with_capacity(specs.len());
        for (index, spec) in specs.iter().enumerate() {
            let member_seed = stream_seed(seed, index as u64 + 1);
            columns.push(out_of_fold(
                spec,
                features,
                targets,
                &folds,
                member_seed,
                deadline,
            )?);
        }
        let (intercept, weights) = nonnegative_weights(&columns, targets);
        let mut members = Vec::new();
        for (index, (spec, weight)) in specs.iter().zip(weights).enumerate() {
            if weight > 0.0 {
                let member_seed = stream_seed(seed, index as u64 + 1);
                members.push((weight, spec.fit(features, targets, member_seed, deadline)?));
            }
        }
        // With every weight at zero the intercept is the training mean, so an all-uninformative
        // stack predicts the mean rather than failing.
        Ok(Self { intercept, members })
    }

    pub(crate) fn predict(&self, features: FeatureMatrix<'_>) -> ComparatorResult<Vec<f64>> {
        let mut total = vec![self.intercept; features.rows()];
        for (weight, member) in &self.members {
            for (sum, prediction) in total.iter_mut().zip(member.predict(features)?) {
                *sum += weight * prediction;
            }
        }
        match total.iter().position(|value| !value.is_finite()) {
            Some(row) => Err(ComparatorError::NonFinitePrediction { row }),
            None => Ok(total),
        }
    }

    pub(crate) fn member_count(&self) -> usize {
        self.members.len()
    }

    pub(crate) fn model_bytes(&self) -> u64 {
        8 + self
            .members
            .iter()
            .map(|(_, member)| member.model_bytes() + 8)
            .sum::<u64>()
    }
}
