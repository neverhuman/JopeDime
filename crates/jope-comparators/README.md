# jope-comparators

Pure-Rust comparator models for **same-rows comparison**. A run's champion can be compared with
established tabular learners that are fitted on exactly the same training rows and scored on
exactly the same held-out rows.

This crate is a library only. It is not wired into runs yet. When it is, comparators are **off by
default**: a run includes them only when the user opts in (`--comparators` or the MCP
`comparators` field), and never at `fast` effort.

## Models

| Type | What it is |
| --- | --- |
| `ForestModel` + `ForestParams::random_forest` | Random forest with scikit-learn `RandomForestRegressor` defaults: bootstrap, best split, all features |
| `ForestModel` + `ForestParams::extra_trees` | Extra trees with scikit-learn `ExtraTreesRegressor` defaults: every row, one random cut per feature |
| `HistGbmModel` + `HistGbmParams` | Squared-loss histogram gradient boosting with shrinkage, L2, row subsampling and seeded inner-validation early stopping |
| `RidgeModel` + `RidgeParams` | Standardized ridge regression (Cholesky), at most `MAX_RIDGE_FEATURES` columns |
| `caruana_weights` | Greedy weighted ensemble selection (the AutoGluon `WeightedEnsemble_L2` step) |

## Roster

`fit_comparator(id, tier, features, targets, seed, deadline)` fits any roster entry. A tier
fixes deterministic work counts, so results do not depend on machine speed. Its ceiling bounds
the whole comparator stage: 60 s at `medium`, 300 s at `high` and 1,800 s at `ultra`. There is
no `fast` tier.

| Key | Label | Strategy |
| --- | --- | --- |
| `autogluon_style` | AutoGluon-style weighted ensemble (Rust port) | Six members on a 20 % inner holdout, Caruana weights, refit on all rows |
| `flaml_style` | FLAML-style cost-frugal search (Rust port) | Cheap starting points, seeded local search, k-fold CV, refit the best |
| `h2o_style` | H2O AutoML-style stacked ensemble (Rust port) | GLM, GBM, DRF, XRT and a GBM grid; out-of-fold non-negative metalearner |
| `mljar_style` | MLJAR-style AutoML (Rust port) | Seven default members on a 25 % inner holdout, greedy ensemble |
| `xgboost_style` | XGBoost-style hist-GBM (Rust) | Depth 3, learning rate 0.10, L2 1, 90 % subsample |
| `lightgbm_style` | LightGBM-style leaf-wise hist-GBM (Rust) | 31 leaves, learning rate 0.05, 20 rows per leaf |
| `catboost_style` | CatBoost-style hist-GBM, strong L2 (Rust) | Depth 6, learning rate 0.03, L2 3 |
| `random_forest` | Random forest (scikit-learn defaults, Rust) | Bootstrap, best splits |
| `extra_trees` | Extra trees (scikit-learn defaults, Rust) | Every row, random cuts |
| `hist_gradient_boosting` | HistGradientBoosting (scikit-learn-style, Rust) | 31 leaves, learning rate 0.05, L2 0.01 |

Every boosting model uses seeded inner-validation early stopping. Every inner split partitions
the training rows only.

These are **re-homed strategies, not the upstream libraries**. The XGBoost-, LightGBM- and
CatBoost-style comparators are parameter presets of `HistGbmModel`; they do
not implement oblivious trees, ordered boosting or GOSS. The strategies come from the pure-Rust
ports in the `jain-model-zoo` repository (`reference/ported/random_forest_regression`,
`extra_trees_regression`, `hist_gradient_boosting_regression` and `autogluon_*`). Those ports
were rebuilt here on one shared core instead of being copied.

Trees bin each feature into at most 255 quantile bins, so split thresholds can differ slightly
from exact-split implementations.

## Contract

- **Input.** `FeatureMatrix` borrows a row-major `rows × cols` buffer. NaN marks a missing value;
  infinities are refused.
- **Missing values.** `MedianImputer` learns one median per column from the **training rows only**
  and applies them unchanged at prediction.
- **Determinism.** A fit is a pure function of its rows, parameters and seed. There is no global
  state, IO or clock, except the caller's cooperative `Deadline`. A deadline can stop a fit with
  `DeadlineExceeded`; it never changes the model a completed fit returns.
- **Failures.** Every failure is a typed `ComparatorError`. Nothing panics on bad input.
  `ComparatorError::guidance()` returns the refusing control, the reason, common fixes, this page
  and the local rerun that proves a fix.

## Verification

```sh
cargo nextest run --locked -P pr -p jope-comparators
cargo test --locked -p jope-comparators --doc
```

The `comparators` test target covers three areas:

- **Learning.** Every model beats the training-mean baseline on unseen rows.
- **Properties.** Bitwise seeded determinism, row locality, FIT-only imputation and constant
  targets, all over generated tables.
- **Refusals.** Every invalid input, parameter and expired deadline.
- **Roster.** Every comparator learns at the medium tier, reproduces bit-for-bit under the
  same seed, honours an expired deadline, and carries an honest `Rust` label.
