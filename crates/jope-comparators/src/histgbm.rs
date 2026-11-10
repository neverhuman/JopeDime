//! Histogram gradient boosting for squared loss, re-homed from the jain-model-zoo
//! `hist_gradient_boosting_regression` port and the AutoGluon port's hist-GBM members.

use crate::binning::{BinnedMatrix, MAX_BINS};
use crate::deadline::Deadline;
use crate::error::{ComparatorError, ComparatorResult};
use crate::forest::parameter_bytes;
use crate::matrix::{mean, predict_rows, validate_training, FeatureMatrix, MedianImputer};
use crate::rng::{stream_seed, SplitMix64};
use crate::tree::{RegressionTree, SplitRule, TreeParams};

/// Improvements of the inner-validation RMSE at or below this do not reset patience.
const MIN_IMPROVEMENT: f64 = 1e-12;

/// `EarlyStopping` is a struct for the inner-validation stopping contract in this module: a
/// seeded fraction of the training rows is held aside and boosting keeps the best round.
#[derive(Clone, Debug, PartialEq)]
pub struct EarlyStopping {
    /// Fraction of training rows held aside for validation, in `(0, 0.5]`.
    pub validation_fraction: f64,
    /// Rounds without improvement before stopping (at least 1).
    pub patience: usize,
}

/// `HistGbmParams` is a struct for the boosting configuration contract in this module.
#[derive(Clone, Debug, PartialEq)]
pub struct HistGbmParams {
    /// Maximum boosting rounds (at least 1).
    pub rounds: usize,
    /// Shrinkage applied to every tree, in `(0, 1]`.
    pub learning_rate: f64,
    /// Leaf limit per tree (leaf-wise growth); `None` with a depth limit grows depth-wise.
    pub max_leaves: Option<usize>,
    /// Depth limit per tree.
    pub max_depth: Option<usize>,
    /// Minimum training rows per leaf (at least 1).
    pub min_samples_leaf: usize,
    /// L2 penalty on leaf values (non-negative).
    pub l2: f64,
    /// Fraction of rows sampled without replacement per round, in `(0, 1]`.
    pub row_subsample: f64,
    /// Histogram bins per feature, in `2..=255`.
    pub max_bins: usize,
    /// Optional early stopping on an inner validation split of the training rows.
    pub early_stopping: Option<EarlyStopping>,
}

impl HistGbmParams {
    /// scikit-learn `HistGradientBoostingRegressor`-style settings with `rounds` iterations:
    /// learning rate 0.05, 31 leaves, L2 0.01, 20 rows per leaf, no early stopping.
    pub fn scikit_learn_style(rounds: usize) -> Self {
        Self {
            rounds,
            learning_rate: 0.05,
            max_leaves: Some(31),
            max_depth: None,
            min_samples_leaf: 20,
            l2: 0.01,
            row_subsample: 1.0,
            max_bins: MAX_BINS,
            early_stopping: None,
        }
    }

    fn validate(&self) -> ComparatorResult<()> {
        let stopping_ok = self.early_stopping.as_ref().is_none_or(|stopping| {
            stopping.validation_fraction > 0.0
                && stopping.validation_fraction <= 0.5
                && stopping.patience >= 1
        });
        let checks = [
            (self.rounds >= 1, "rounds"),
            (
                self.learning_rate > 0.0 && self.learning_rate <= 1.0,
                "learning_rate",
            ),
            (
                self.max_leaves.is_none_or(|leaves| leaves >= 2),
                "max_leaves",
            ),
            (self.min_samples_leaf >= 1, "min_samples_leaf"),
            (self.l2 >= 0.0 && self.l2.is_finite(), "l2"),
            (
                self.row_subsample > 0.0 && self.row_subsample <= 1.0,
                "row_subsample",
            ),
            ((2..=MAX_BINS).contains(&self.max_bins), "max_bins"),
            (stopping_ok, "early_stopping"),
        ];
        match checks.iter().find(|(valid, _)| !valid) {
            Some((_, name)) => Err(ComparatorError::InvalidParameter { name }),
            None => Ok(()),
        }
    }
}

/// `HistGbmModel` is a struct for the fitted boosting contract in this module: the FIT-only
/// imputer, the initial prediction and the shrunken trees added to it.
#[derive(Clone, Debug, PartialEq)]
pub struct HistGbmModel {
    imputer: MedianImputer,
    initial: f64,
    trees: Vec<RegressionTree>,
}

struct Schedule {
    train: Vec<u32>,
    validation: Vec<u32>,
}

fn schedule(rows: usize, stopping: Option<&EarlyStopping>, rng: &mut SplitMix64) -> Schedule {
    // `validate_training` bounds `rows` by `u32::MAX`, so the casts are lossless.
    let mut all: Vec<u32> = (0..rows as u32).collect();
    match stopping {
        Some(stopping) if rows >= 4 => {
            rng.shuffle(&mut all);
            let held =
                ((rows as f64 * stopping.validation_fraction).round() as usize).clamp(1, rows / 2);
            let train = all.split_off(held);
            Schedule {
                train,
                validation: all,
            }
        }
        _ => Schedule {
            train: all,
            validation: Vec::new(),
        },
    }
}

impl HistGbmModel {
    /// Fit on training rows only. `seed` fixes the validation split and row subsamples.
    pub fn fit(
        features: FeatureMatrix<'_>,
        targets: &[f64],
        params: &HistGbmParams,
        seed: u64,
        deadline: &Deadline,
    ) -> ComparatorResult<Self> {
        params.validate()?;
        validate_training(features, targets)?;
        let (rows, cols) = (features.rows(), features.cols());
        let imputer = MedianImputer::fit(features);
        let dense = imputer.transform(features)?;
        let binned = BinnedMatrix::build(&dense, rows, cols, params.max_bins);
        let mut rng = SplitMix64::new(stream_seed(seed, 0));
        let plan = schedule(rows, params.early_stopping.as_ref(), &mut rng);
        let train_targets: Vec<f64> = plan
            .train
            .iter()
            .map(|row| targets[*row as usize])
            .collect();
        let initial = mean(&train_targets);
        let tree_params = TreeParams {
            max_depth: params.max_depth,
            max_leaves: params.max_leaves,
            min_samples_leaf: params.min_samples_leaf,
            l2: params.l2,
            features_per_split: cols,
            split_rule: SplitRule::Best,
        };
        let sample_size = ((plan.train.len() as f64 * params.row_subsample).ceil() as usize)
            .clamp(1, plan.train.len());
        let mut predictions = vec![initial; rows];
        let mut gradients = vec![0.0; rows];
        let mut trees = Vec::with_capacity(params.rounds);
        let (mut best_loss, mut best_len, mut stale) = (f64::INFINITY, 0_usize, 0_usize);
        for round in 0..params.rounds {
            deadline.check()?;
            for (gradient, (prediction, target)) in
                gradients.iter_mut().zip(predictions.iter().zip(targets))
            {
                *gradient = prediction - target;
            }
            let mut sample = plan.train.clone();
            if sample_size < sample.len() {
                SplitMix64::new(stream_seed(seed, round as u64 + 1)).shuffle(&mut sample);
                sample.truncate(sample_size);
            }
            let mut tree =
                RegressionTree::grow(&binned, sample, &gradients, &tree_params, &mut rng);
            tree.scale(params.learning_rate);
            for (row, prediction) in predictions.iter_mut().enumerate() {
                *prediction += tree.predict(&dense[row * cols..(row + 1) * cols]);
            }
            trees.push(tree);
            let Some(stopping) = &params.early_stopping else {
                continue;
            };
            if plan.validation.is_empty() {
                continue;
            }
            let loss = validation_rmse(&plan.validation, &predictions, targets);
            if loss < best_loss - MIN_IMPROVEMENT {
                (best_loss, best_len, stale) = (loss, trees.len(), 0);
            } else {
                stale += 1;
                if stale >= stopping.patience {
                    break;
                }
            }
        }
        if best_len > 0 {
            trees.truncate(best_len);
        }
        Ok(Self {
            imputer,
            initial,
            trees,
        })
    }

    /// Initial prediction plus the sum of the shrunken trees, for each row.
    pub fn predict(&self, features: FeatureMatrix<'_>) -> ComparatorResult<Vec<f64>> {
        predict_rows(features, &self.imputer, |row| {
            self.initial + self.trees.iter().map(|tree| tree.predict(row)).sum::<f64>()
        })
    }

    /// Number of boosting rounds kept after early stopping.
    pub fn rounds_kept(&self) -> usize {
        self.trees.len()
    }

    /// Approximate in-memory parameter footprint in bytes, for model-size comparison.
    pub fn model_bytes(&self) -> u64 {
        parameter_bytes(&self.imputer, &self.trees, 1)
    }
}

fn validation_rmse(rows: &[u32], predictions: &[f64], targets: &[f64]) -> f64 {
    let squared: f64 = rows
        .iter()
        .map(|row| (predictions[*row as usize] - targets[*row as usize]).powi(2))
        .sum();
    (squared / rows.len() as f64).sqrt()
}
