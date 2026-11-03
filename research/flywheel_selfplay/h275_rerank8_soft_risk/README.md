# H275: Soft Invalid-Risk Penalized Top-8 Re-Ranking (`h275_rerank8_soft_risk`)

- **Wave ID**: `h275-rerank8-soft-risk-20261103` (`H275`)
- **Owner**: `Argon`
- **Status**: `PREREGISTERED` (`outcomes_read = 0`)
- **Preregistration Specification**: See [PREREG.md](PREREG.md) and [fold_manifest.s3.json](fold_manifest.s3.json) (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H275/prereg/fold_manifest.json`, `VersionId = SuUQLfyvQvJ_u07AyESxvQIOTR6wYPkG`).

## Summary of Preregistered Design
- **Source**: Argon's `H273` (`29fdbd8be15e`) clean-room `K = 8` (`568` cols) runner, ranker, invalid-gate `LGBMClassifier`, margin grid, and strict inner-`FIT` fallback.
- **Confirmatory Arm (`RERANK8_CR_SOFT`)**: Per-query standardized ranker score `z_k = (score_k - mean) / max(std, 1e-6)` with slot `0` (`r882`) untouched (`score'_0 = z_0`) and soft invalid-risk penalty `score'_k = z_k - lambda * p_invalid_k` for `k in {1..7}`, where `lambda in {0.0, 0.5, 1.0, 2.0, 4.0, 8.0}` is selected jointly with `margin` strictly inside inner `FIT` maximizing inner-`FIT` equal-family utility gain subject to `fit_invalid <= fit_ctrl_invalid`.
- **Fresh Seeds**: `fold_seeds = (275140, 275141, 275142)` x `5` folds x `model_seeds = (275101, 275102, 275103)` (`45` cells/arm), `boot_seed = 275150`, `Holm m = 3` over `{PRIMARY (DL-Train), SECONDARY (DL-Val91 supporting only), SOFT - VETO (DL-Train)}`, plus `TERTIARY` `H272` `41`-family least-similar leave-cluster-out panel (`201` contexts).
