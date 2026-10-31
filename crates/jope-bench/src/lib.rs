//! `jope-bench`: one cut for everyone.
//!
//! For every dataset in a directory tree, the benchmark builds train/held-out cuts with the one
//! fold authority (`feat_math::fold_of` keyed on `"<dataset_hash>:<seed>"`, the same key the
//! product pipeline uses), fits every tier-1 baseline as a single-head program such as
//! `(pipe X (ridge))`, fits any extra program the caller names, optionally runs the search on the
//! cut's training rows only, and scores **all of them on the same held-out rows with the same
//! evaluator**. Then it follows the honest screen-then-deepen protocol: every method is scored
//! on one screen cut, the top N are re-scored on every cut, and the number that counts is the
//! pooled out-of-fold one, penalised by fold variance ([`eval::robust_fitness_for`]).
//!
//! Three tasks ([`task::Task`]): regression on the frozen kernel (R², RMSE), binary and
//! multiclass on the schema-bound v3 table API (ROC AUC, log loss, Brier; macro-F1, accuracy).
//! The search is regression-only for now.
//!
//! Tier-2 baselines are external engines (XGBoost, LightGBM, CatBoost, a scikit-learn forest)
//! run as subprocesses on the same cuts ([`adapter`]): the controller hands them a cut's training
//! rows and the held-out features only, and scores the predictions they return with the same
//! evaluator, so their rows are as honest as a program's. Frozen General Regressor Challenge
//! entrants written in safe Rust ([`regressors`], `run --entrant <name>`) are compiled into this
//! binary and run through that same adapter contract, so they are scored like any external engine.
//!
//! What it deliberately does not do: touch the catalog's sealed `test.csv` (cuts are made from
//! training rows only), reuse the pipeline's private evaluator (it carries its own, [`eval`], so
//! a change there cannot silently change a benchmark), or depend on any crate outside this
//! workspace (adapters are processes, not dependencies).
//!
//! Outputs are `jope.bench-report/v2` files ([`report`]) and, from those, `scores.csv` rows plus
//! registries for a local `jope-arena` track ([`emit`]), so the same Elo ladder code that ranks the
//! published benchmark ranks this one.

/// Equal-budget comparison of search profiles (`jope-bench ablation`).
pub mod ablation;
/// Process adapters for external baseline engines and their prediction contract.
pub mod adapter;
/// Frozen original Rust learners and their AutoML comparison adapters.
pub mod automl;
/// Deterministic train and held-out fold construction for same-cut comparisons.
pub mod cuts;
/// Conversion of benchmark reports into arena tracks and score registries.
pub mod emit;
/// Task-specific metrics and robust fitness aggregation.
pub mod eval;
/// Dataset discovery, metadata validation and table loading.
pub mod load;
/// Registered benchmark methods and their same-cut fitting paths.
pub mod methods;
/// Original native AutoML candidates, evaluated with the same cuts and scorer.
pub mod native;
/// Frozen development panel: raw inputs, task metrics and fitted predictors.
pub mod panel;
/// Validation and representation of the benchmark's versioned protocol.
pub mod protocol;
/// Frozen challenge regressors and their benchmark adapter entry points.
pub mod regressors;
/// Versioned benchmark report records and their serialization helpers.
pub mod report;
/// Supported task kinds and task-specific metric definitions.
pub mod task;

/// Identity of the evaluator every score in a report was produced by. v2: regression on the
/// frozen kernel, binary and multiclass on the v3 table API, scores in a task-keyed metrics map.
pub const EVALUATOR: &str = "jope-bench.feat-math-f64.v2";
/// Default fold seed: the product pipeline's `DEFAULT_SEED`, so cuts agree with `jope pipeline run`.
pub const DEFAULT_SEED: u64 = 20_260_907;
/// Default number of folds.
pub const DEFAULT_FOLDS: usize = 5;
