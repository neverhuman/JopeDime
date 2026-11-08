# Full Elo list

`full-elo.csv` and `full-elo.jsonl` are the same 615 rows. They are copied from receipts already in this repository. No new model was fit to build this file. `protocol_id` is required reading: a 2259.8 and a 1690.8 are not on one ladder.

| Protocol | Rows | What it is | Best row |
| --- | ---: | --- | --- |
| `tabular-regression-v2-published` | 305 | Golden45 tabular ladder as published, 45 datasets, 8,763 rankable scores | `jope-dime` (evolved), Elo 2259.8, 45/45, 11 wins |
| `tabular-regression-v2-benchmark-valid` | 304 | The same scores after the 45 test-adaptive `jope-dime` rows are removed | `tabfm_ensemble_regressor`, Elo 1934.9, 45/45 |
| `automl-64-family-rmse-bt-v3` | 6 | Native Rust regressors, 64 families × 3 folds × 3 seeds, all 576 cells complete | `Opus5p5_Extra`, Elo 1690.8345518061888 |

## Our best model

On the published ladder the highest row is our evolved `jope-dime` program at **2259.8**. Every one of its 45 rows was selected with test labels. The repository's own correction file removes those rows from the fair fit. That 2259.8 is the best-known test-adaptive envelope, not a held-out rating.

On the fair tabular ladder `jope-dime` does not appear. Our reproducible entry is `jope-dime-regressor`: **rank 31 of 304, Elo 1687.0, 42/45, 1 dataset win**. The best fair tabular model is TabFM Ensemble at 1934.9.

On the native-regressor panel the best model is `Opus5p5_Extra` at 1690.8345518061888. `kimik3p1_max` is 3rd at 1510.6689563931964. Those six numbers are the v3 refit. They are not comparable to 1687.0 or 2259.8.

The H-wave neural ladder is not in this file. Its admitted best arm, `h129/BLEND_025` at 1523.48 versus `r882` at 1500, is in `assets/comparisons/protocol-elo/v2/`.

## Sources

- `assets/regression/elo-tabular-regression-v2.json` (`leaderboard_sha256` `2d28ffbd…`)
- `assets/regression/elo-tabular-regression-v2.benchmark-valid.json` (`leaderboard_sha256` `4c39f51c…`)
- `assets/automl-elo/v3/RESULTS.json`
- Provenance for the two tabular ladders: `assets/regression/SOURCES.md`
