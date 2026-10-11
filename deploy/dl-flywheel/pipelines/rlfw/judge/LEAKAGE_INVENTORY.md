# DSL inner-fold fit inventory

Read 2026-10-11. Fixture judgment only. This file does not spend a seal.

The honest inner-fold path is `cv_oof_predictions` / `cv_r2` in `crates/feat-math/src/model.rs`: each fold calls `fit` on the rows whose `fold_of` assignment is not that fold. `target_aware_ops_are_oof_in_cv` checks that a noise target keeps CV r² under 0.15 for `target_bin_encode`, `select_corr`, and `pair_ratio`. It does not poison one fold's held-out rows and compare the fitted model. `advanced_v3` `target_bin_encode` with `oof=5` fits fold transforms for the training block and still stores a full-block `fit_op` for the deployed operator (`transforms/mod.rs`). That is not the inner-fold fixture below.

| stage | inner-fold fit-only fixture before this change | fixture |
|---|---|---|
| `target_bin_encode` | noise-CV bound only (`target_aware_ops_are_oof_in_cv`) | `inner_fold_learned_stages_fit_only_on_fitting_rows` |
| `quantile_bins` | none found | same test |
| `winsor_clip` | none found | same test |
| `standardize` | none found for this operator. `test_trainonly.py` poisons an outer-test min-max scaler, which is a different transform | same test |
| `impute` | behavior tests in `ops.rs`, not a held-out-row poison | same test |
| head (`ridge` through `fit_head`) | accuracy and determinism tests, not a held-out-row poison | same test, `(pipe X (ridge))` |

The new test requires three things for each stage. A fit on every row must change predictions on the held-out rows, so a leaked fit cannot hide. `cv_oof_predictions` on the clean matrix must match a model fit only on that fold's fitting rows. Poisoning the held-out targets and calling `cv_oof_predictions` with the original features must leave those predictions bit-identical, and poisoning the held-out features must leave a complement-only fit's predictions on the original features bit-identical. The prediction check does not score the poisoned feature rows, because that would change the input rather than the fit.

Local `cargo test -p feat-math --lib` on 2026-10-11, filtered to this test plus `explicit_portfolio_form_fits_through_model_dispatch` and `target_aware_ops_are_oof_in_cv`: 3 passed, 0 failed. `CARGO_TARGET_DIR` was unset, `CARGO_INCREMENTAL=0`, and `CARGO_PROFILE_DEV_DEBUG=line-tables-only`. Receipt: `lanes/G-JUDGE/gap-close-20261011/leakage-fixture.json`.
