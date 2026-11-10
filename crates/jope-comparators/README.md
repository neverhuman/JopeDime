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

These are **re-homed strategies, not the upstream libraries**. The XGBoost-, LightGBM- and
CatBoost-style comparators that later PRs add are parameter presets of `HistGbmModel`; they do
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
