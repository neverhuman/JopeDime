//! Pure-Rust comparator models for same-rows run comparison.
//!
//! JopeDime reports how a run's champion compares with established tabular learners fitted on
//! exactly the same training rows and scored on exactly the same held-out rows. This crate holds
//! the learners: one de-duplicated, dependency-free, panic-free core re-homed from the
//! pure-Rust strategy ports in the `jain-model-zoo` repository (`reference/ported/*`).
//!
//! - [`ForestModel`]: random forest and extra trees ([`ForestParams`]).
//! - [`HistGbmModel`]: histogram gradient boosting ([`HistGbmParams`]); the XGBoost-, LightGBM-
//!   and CatBoost-style comparators are parameter presets of this one learner, not ports of
//!   those libraries.
//! - [`RidgeModel`]: standardized ridge regression.
//! - [`caruana_weights`]: greedy weighted ensemble selection for AutoML-style portfolios.
//! - [`fit_comparator`]: the roster entry point. Every [`ComparatorId`] is fitted at a
//!   [`ComparatorTier`] (`medium`, `high`, `ultra`; there is no `fast` tier) whose
//!   [`TierBudget`] fixes deterministic work counts and the stage's wall-clock ceiling.
//!   [`ComparatorId::descriptor`] gives the honest display label and known gaps.
//!
//! Every fit is a deterministic function of its rows, parameters and seed. Missing values (NaN)
//! are imputed with medians learned from the training rows only. Failures are typed
//! [`ComparatorError`] values with [`ComparatorError::guidance`] repair hints; nothing here
//! panics on bad input, performs IO or reads a clock except through the caller's [`Deadline`].
//!
//! ```
//! use jope_comparators::{Deadline, FeatureMatrix, ForestModel, ForestParams, rmse};
//!
//! let train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//! let targets = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
//! let features = FeatureMatrix::new(&train, 6, 1)?;
//! let model = ForestModel::fit(features, &targets, &ForestParams::random_forest(10), 7,
//!     &Deadline::no_limit())?;
//! let holdout = [0.5, 4.5];
//! let predictions = model.predict(FeatureMatrix::new(&holdout, 2, 1)?)?;
//! assert!(rmse(&[0.0, 10.0], &predictions)? < 5.0);
//! # Ok::<(), jope_comparators::ComparatorError>(())
//! ```

mod automl;
mod binning;
mod deadline;
mod ensemble;
mod error;
mod forest;
mod histgbm;
mod linear;
mod matrix;
mod metrics;
mod portfolio;
mod presets;
mod rng;
mod roster;
mod tier;
mod tree;

pub use deadline::Deadline;
pub use ensemble::caruana_weights;
pub use error::{ComparatorError, ComparatorGuidance, ComparatorResult, COMPARATORS_DOCS_URL};
pub use forest::{ForestModel, ForestParams};
pub use histgbm::{EarlyStopping, HistGbmModel, HistGbmParams};
pub use linear::{RidgeModel, RidgeParams, MAX_RIDGE_FEATURES};
pub use matrix::{FeatureMatrix, MedianImputer};
pub use metrics::{mae, r_squared, rmse};
pub use roster::{fit_comparator, ComparatorDescriptor, ComparatorId, FittedComparator};
pub use tier::{ComparatorTier, TierBudget};
