//! Library-style hyperparameter presets for the shared histogram boosting learner.
//!
//! These are **analogues**: the same learner configured with each library's characteristic
//! defaults. They do not implement XGBoost's exact or approximate split finding, LightGBM's
//! GOSS/EFB, or CatBoost's oblivious trees and ordered boosting. Settings come from the
//! jain-model-zoo AutoGluon port's `GbmXgb`, `GbmGbdt` and `GbmCat` members and the
//! `hist_gradient_boosting_regression` port.

use crate::binning::MAX_BINS;
use crate::histgbm::{EarlyStopping, HistGbmParams};

/// Inner-validation early stopping shared by every boosting preset.
fn stopping() -> Option<EarlyStopping> {
    Some(EarlyStopping {
        validation_fraction: 0.1,
        patience: 20,
    })
}

/// XGBoost-style: learning rate 0.10, depth 3, L2 1.0, 90 % row subsample.
pub fn xgboost_style(rounds: usize) -> HistGbmParams {
    HistGbmParams {
        rounds,
        learning_rate: 0.10,
        max_leaves: None,
        max_depth: Some(3),
        min_samples_leaf: 1,
        l2: 1.0,
        row_subsample: 0.9,
        max_bins: MAX_BINS,
        early_stopping: stopping(),
    }
}

/// LightGBM-style: leaf-wise growth to 31 leaves, learning rate 0.05, 20 rows per leaf.
pub fn lightgbm_style(rounds: usize) -> HistGbmParams {
    HistGbmParams {
        rounds,
        learning_rate: 0.05,
        max_leaves: Some(31),
        max_depth: None,
        min_samples_leaf: 20,
        l2: 0.0,
        row_subsample: 1.0,
        max_bins: MAX_BINS,
        early_stopping: stopping(),
    }
}

/// CatBoost-style: depth 6, learning rate 0.03, strong L2 (3.0). Depth-wise, not oblivious.
pub fn catboost_style(rounds: usize) -> HistGbmParams {
    HistGbmParams {
        rounds,
        learning_rate: 0.03,
        max_leaves: None,
        max_depth: Some(6),
        min_samples_leaf: 1,
        l2: 3.0,
        row_subsample: 1.0,
        max_bins: MAX_BINS,
        early_stopping: stopping(),
    }
}

/// scikit-learn `HistGradientBoostingRegressor`-style, with early stopping.
pub fn scikit_learn_style(rounds: usize) -> HistGbmParams {
    HistGbmParams {
        early_stopping: stopping(),
        ..HistGbmParams::scikit_learn_style(rounds)
    }
}

/// H2O GBM defaults: depth 5, learning rate 0.1, 10 rows per leaf.
pub fn h2o_style(rounds: usize) -> HistGbmParams {
    HistGbmParams {
        rounds,
        learning_rate: 0.10,
        max_leaves: None,
        max_depth: Some(5),
        min_samples_leaf: 10,
        l2: 0.0,
        row_subsample: 1.0,
        max_bins: MAX_BINS,
        early_stopping: stopping(),
    }
}
