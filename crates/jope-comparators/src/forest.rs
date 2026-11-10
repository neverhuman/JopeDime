//! Random forests and extra trees for regression, re-homed from the jain-model-zoo
//! `random_forest_regression` and `extra_trees_regression` ports onto the shared histogram tree.

use crate::binning::{BinnedMatrix, MAX_BINS};
use crate::deadline::Deadline;
use crate::error::{ComparatorError, ComparatorResult};
use crate::matrix::{predict_rows, validate_training, FeatureMatrix, MedianImputer};
use crate::rng::{stream_seed, SplitMix64};
use crate::tree::{RegressionTree, SplitRule, TreeParams};

/// `ForestParams` is a struct for the tree ensemble configuration contract in this module.
#[derive(Clone, Debug, PartialEq)]
pub struct ForestParams {
    /// Number of trees (at least 1).
    pub trees: usize,
    /// Draw a bootstrap sample per tree (random forest) or use every row (extra trees).
    pub bootstrap: bool,
    /// One random cut per examined feature (extra trees) instead of the best cut.
    pub random_splits: bool,
    /// Fraction of features examined at each split, in `(0, 1]`.
    pub max_features: f64,
    /// Depth limit; `None` grows until leaves are pure or minimal.
    pub max_depth: Option<usize>,
    /// Minimum training rows per leaf (at least 1).
    pub min_samples_leaf: usize,
    /// Histogram bins per feature, in `2..=255`.
    pub max_bins: usize,
}

impl ForestParams {
    /// scikit-learn `RandomForestRegressor` defaults: bootstrap, best splits, all features.
    pub fn random_forest(trees: usize) -> Self {
        Self {
            trees,
            bootstrap: true,
            random_splits: false,
            max_features: 1.0,
            max_depth: None,
            min_samples_leaf: 1,
            max_bins: MAX_BINS,
        }
    }

    /// scikit-learn `ExtraTreesRegressor` defaults: every row, random cuts, all features.
    pub fn extra_trees(trees: usize) -> Self {
        Self {
            bootstrap: false,
            random_splits: true,
            ..Self::random_forest(trees)
        }
    }

    fn validate(&self) -> ComparatorResult<()> {
        let checks = [
            (self.trees >= 1, "trees"),
            (
                self.max_features > 0.0 && self.max_features <= 1.0,
                "max_features",
            ),
            (self.min_samples_leaf >= 1, "min_samples_leaf"),
            ((2..=MAX_BINS).contains(&self.max_bins), "max_bins"),
        ];
        match checks.iter().find(|(valid, _)| !valid) {
            Some((_, name)) => Err(ComparatorError::InvalidParameter { name }),
            None => Ok(()),
        }
    }
}

/// Features examined per split for a fraction of `cols`, at least one.
pub(crate) fn features_per_split(cols: usize, fraction: f64) -> usize {
    ((cols as f64 * fraction).ceil() as usize).clamp(1, cols)
}

/// `ForestModel` is a struct for the fitted tree ensemble contract in this module: the FIT-only
/// imputer and the trees whose average is the prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct ForestModel {
    imputer: MedianImputer,
    trees: Vec<RegressionTree>,
}

impl ForestModel {
    /// Fit on training rows only. `seed` fixes every bootstrap draw and random cut.
    pub fn fit(
        features: FeatureMatrix<'_>,
        targets: &[f64],
        params: &ForestParams,
        seed: u64,
        deadline: &Deadline,
    ) -> ComparatorResult<Self> {
        params.validate()?;
        validate_training(features, targets)?;
        let rows = features.rows();
        let imputer = MedianImputer::fit(features);
        let dense = imputer.transform(features)?;
        let binned = BinnedMatrix::build(&dense, rows, features.cols(), params.max_bins);
        let gradients: Vec<f64> = targets.iter().map(|target| -target).collect();
        let tree_params = TreeParams {
            max_depth: params.max_depth,
            max_leaves: None,
            min_samples_leaf: params.min_samples_leaf,
            l2: 0.0,
            features_per_split: features_per_split(features.cols(), params.max_features),
            split_rule: if params.random_splits {
                SplitRule::Random
            } else {
                SplitRule::Best
            },
        };
        let mut trees = Vec::with_capacity(params.trees);
        for index in 0..params.trees {
            deadline.check()?;
            let mut rng = SplitMix64::new(stream_seed(seed, index as u64));
            // `validate_training` bounds `rows` by `u32::MAX`, so the casts are lossless.
            let sample: Vec<u32> = if params.bootstrap {
                (0..rows).map(|_| rng.below(rows) as u32).collect()
            } else {
                (0..rows as u32).collect()
            };
            trees.push(RegressionTree::grow(
                &binned,
                sample,
                &gradients,
                &tree_params,
                &mut rng,
            ));
        }
        Ok(Self { imputer, trees })
    }

    /// Average of the trees' predictions for each row.
    pub fn predict(&self, features: FeatureMatrix<'_>) -> ComparatorResult<Vec<f64>> {
        let scale = 1.0 / self.trees.len() as f64;
        predict_rows(features, &self.imputer, |row| {
            self.trees.iter().map(|tree| tree.predict(row)).sum::<f64>() * scale
        })
    }

    /// Number of fitted trees.
    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    /// Approximate in-memory parameter footprint in bytes, for model-size comparison.
    pub fn model_bytes(&self) -> u64 {
        parameter_bytes(&self.imputer, &self.trees, 0)
    }
}

/// Footprint of an imputer, a set of trees and `extra` scalar parameters: every node is counted
/// as one feature index plus two `f64`-sized fields.
pub(crate) fn parameter_bytes(
    imputer: &MedianImputer,
    trees: &[RegressionTree],
    extra: usize,
) -> u64 {
    let nodes: usize = trees.iter().map(RegressionTree::node_count).sum();
    let scalars = imputer.medians().len() + extra;
    (nodes * 3 * std::mem::size_of::<u64>() + scalars * std::mem::size_of::<f64>()) as u64
}
