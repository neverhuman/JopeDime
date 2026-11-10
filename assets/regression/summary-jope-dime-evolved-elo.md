# Summary: JopeDime (evolved) Elo 2259.8

Reviewed 2026-11-08 from the committed scores, the protocol-corrections file, and the arena tests. No new model was fit for this note.

## Finding

**2259.8 is a real output of the published Elo fitter. It is not the held-out score of one frozen JopeDime model.**

The published ladder ranks `jope-dime`, displayed as `jope-dime (evolved)`, first of 305 methods: Elo 2259.8, average rank 4.622, 11 dataset wins, 45/45 coverage. `crates/jope-arena/tests/elo_regression.rs` recomputes that ladder from the committed `scores.csv` and checks the top row to within 0.05 Elo. The arithmetic is not a typo, a rounding bug, or a missing opponent.

The 45 inputs that produce the rating are the problem. They are the best test-scored result the evolution search found on each dataset, not the score of one program chosen before the test labels were opened. The fair refit of the same scores drops all 45 rows. On that ladder `jope-dime` does not appear.

Use these numbers instead when saying how a JopeDime model performs:

| Claim | Number | Where it is safe |
| --- | --- | --- |
| Best test-scored candidate in the published evolution envelope | Elo 2259.8, rank 1 of 305 | Only with the label “test-selected / oracle envelope” |
| Best fair tabular model on that benchmark | TabFM Ensemble, Elo 1934.9, rank 1 of 304 | `benchmark-valid` ladder |
| Our one recorded fixed-seed run of the same engine (run reports not committed) | `jope-dime-regressor`, Elo 1687.0, rank 31 of 304, 42/45, 1 win | `benchmark-valid` ladder |
| Best native Rust regressor on the later 64-family panel | `Opus5p5_Extra`, Elo 1690.8345518061888 | AutoML v3. A different panel. Do not subtract it from 2259.8 |

The gap from 2259.8 down to TabFM’s 1933.1 on the published ladder is about 327 Elo. That gap is the advantage of picking the minimum test RMSE out of a large search, not evidence that one frozen program is that much stronger than TabFM.

## What the 45 rows actually are

Every `jope-dime` row in the committed `scores.csv.zst` was classified on 2026-11-08. There are exactly 45 rows, one per dataset. The file marks all 45 `benchmark_valid=True`. It has no `test_adaptive` column, so the published fit includes them.

| Rows | `source_kind` | `source_revision` | What was scored |
| --- | ---: | --- | --- |
| 42 | `complete-evolution-ledger-snapshot` | `1784790556145` | The per-dataset minimum RMSE over the combined evolution ledger. The `source_uri` is `results/tabular-regression-v1/jope-dime-evolution/datasets/golden42/<dataset>/evaluations.jsonl`, the whole ledger, not one program receipt. |
| 3 | `accepted-current-best-rust-replay` | `golden46-new3-20260723-v4` | An accepted replay of the current best candidate from Golden46. Datasets `6054656956182296`, `ZTS_small`, and `a3bd0a400ec4d7f9`. The `source_uri` is `results/tabular-regression-v2/aspfm-acceptance-manifest.json`. |

Example values from those rows: dataset `019de5d76a718163` has value `0.01317412081339552` and points at its `evaluations.jsonl`. Dataset `ZTS_small` has value `0.1823874052422644` and points at the acceptance manifest. The manifest’s own `benchmark_valid` field is `true`, and its `source_evolution_receipt.run_id` is `golden46-new3-20260723-v4` for all three entries.

### The 42 Golden42 rows

`assets/regression/protocol-corrections-tabular-regression-v2.json`, correction id `golden42-evolution-envelope-test-selected`, records the selection rule. Each value is the minimum RMSE over every valid evaluation in that dataset’s ledger. The campaign red-team names the same quantity: method id `jope-dime`, label “best-test ASPFM envelope over 63,108 Golden42 candidate evaluations”, role `oracle_reference_only`, `candidate_evaluations: 63_108`. A configuration that labels it otherwise is refused. The value is read from the index field `best_rmse` and is kept out of the fair ladder. TabPFN, TabFM, and `jope-dime-regressor` are role `fair`.

A minimum over 63,108 test-scored candidates is a test-adaptive selection even if each individual candidate was fitted only on training rows. The search saw the test metric and kept the winner.

### The 3 Golden46 rows

Correction id `golden46-new3-20260723-v4-test-adaptive` covers the other three. The acceptance manifest records `benchmark_valid: true`. The campaign that produced run `golden46-new3-20260723-v4` lives in the test-adaptive module and writes its artefacts with `benchmark_valid: false` and `test_adaptive: true`. Generation-zero seeds are derived from the best test-scored candidate of the corrected parent run. The published flag contradicts the source run’s own label, so the correction demotes these rows.

The module has since been split (`crates/jope-campaigns/src/test_adaptive/aggressive.rs` is now a 69-line facade). The flags still exist under that tree, including `crates/jope-campaigns/src/test_adaptive/aggressive/runner.rs`. The corrections file still cites the pre-split line numbers. The row classification above does not depend on those line numbers. It depends on `source_kind`, `source_revision`, and `source_uri` in the committed scores.

## What the fair ladder says

`assets/regression/elo-tabular-regression-v2.benchmark-valid.json` is the same `scores.csv` after `protocol-corrections-tabular-regression-v2.json` is applied and every `benchmark_valid: False` row is left out.

| | Published ladder | Benchmark-valid ladder |
| --- | --- | --- |
| File | `elo-tabular-regression-v2.json` | `elo-tabular-regression-v2.benchmark-valid.json` |
| Methods | 305 | 304 |
| Rankable rows | 8,763 | 8,718 |
| Excluded rows | 0 | 45 |
| Corrections | none | 3 Golden46 rows + 42 Golden42 rows |
| `jope-dime` | rank 1, Elo 2259.8, 11 wins, 45/45 | absent |
| Best method | `jope-dime` (evolved) | `tabfm_ensemble_regressor`, Elo 1934.9, 7 wins, 45/45 |
| `jope-dime-regressor` | rank 32, Elo 1684.9, 1 win | rank 31, Elo 1687.0, 1 win, 42/45 |

`crates/jope-arena/tests/elo_regression.rs::benchmark_valid_ladder_excludes_the_rows_the_campaign_code_labels_test_adaptive` pins this. It checks that the corrections select exactly those 3 and 42 rows, that the published scores still say `benchmark_valid=True` for the Golden46 rows (“published as valid; that is the contradiction”), and that applying the corrections removes `jope-dime` from the fit.

`assets/regression/SOURCES.md` states the same conclusion in prose: the published ladder is the best known program per dataset, not a held-out result, and `jope-dime-regressor` is this repository’s entry on the benchmark-valid ladder.

## Other ladders, so the numbers are not mixed

These are real ratings. They answer different questions. Do not subtract them from 2259.8.

| Protocol | File | Best row | Our comparable row |
| --- | --- | --- | --- |
| Published Golden45 tabular, 45 datasets | `assets/regression/elo-tabular-regression-v2.json` | `jope-dime` 2259.8, test-selected | `jope-dime-regressor` 1684.9 |
| Fair Golden45 tabular | `assets/regression/elo-tabular-regression-v2.benchmark-valid.json` | TabFM Ensemble 1934.9 | `jope-dime-regressor` 1687.0, rank 31 of 304 |
| Native Rust regressors, 64 families × 3 folds × 3 seeds, 576/576 cells | `assets/automl-elo/v3/RESULTS.json` | `Opus5p5_Extra` 1690.8345518061888 | `kimik3p1_max` is 3rd at 1510.6689563931964. `grok4p7_xhigh` is 6th at 1308.4482213594254 |
| H-wave neural policy versus incumbent `r882` | `research/flywheel_selfplay/h129_gbm_blend_harm_gate/metrics.json` | Admitted arm `BLEND_025` at 1523.48 versus `r882` = 1500 | `PURE_GBM` is higher at 1528.02 and is not admitted (harm 18.5%, cap 10%) |

The combined copy of the two tabular ladders plus the six native regressors is `assets/comparisons/full-elo/v1/full-elo.csv` (615 rows). The path index is `assets/comparisons/full-elo/v1/PATHS.txt`.

## Concerns

1. **The headline number is an oracle envelope.** Reporting 2259.8 as “our model’s Elo” overstates a search result as if it were one frozen program. The source URI for 42 of 45 rows is the evaluations ledger.
2. **The published validity flag disagrees with the campaign.** All 45 `scores.csv` rows say `benchmark_valid=True`. The Golden46 campaign writes `benchmark_valid: false` and `test_adaptive: true`. The acceptance manifest flips the flag to true. Anyone who trusts only the scores file will refit 2259.8 and think it is a valid benchmark row.
3. **The fair number is much lower, and it is already computed.** `jope-dime-regressor` at 1687.0 is recorded as a fixed-seed run; its 42 run reports (`results/external-references/golden42/<id>/jope-dime-regressor.json`) are not committed and its `source_revision` is empty, so the run cannot be re-checked from the repository. It is rank 31, not rank 1. 17 of the 30 methods above it cover fewer than 42 of the 45 datasets (four are 1/45 singletons); refit on the 124 methods with at least 42/45 coverage it is 14th (1662.2), and on the 42 Golden42 datasets alone it is still 31st (1686.9). TabFM Ensemble leads the fair ladder by about 248 Elo.
4. **Partial coverage inflates some neighbors, but it does not create 2259.8.** Many methods on this ladder score 1 of 45 datasets. A full-coverage method is easier to rank highly. `jope-dime` does have 45/45 coverage, so the rating is not a small-sample fluke. The selection rule, not missing opponents, is what puts it first. TabFM also has 45/45 and still trails by hundreds of Elo on the published fit.
5. **Line citations in the corrections file have drifted.** `protocol-corrections-tabular-regression-v2.json` points at `aggressive.rs` lines 510–748. That file is now a module facade. The flags live in `crates/jope-campaigns/src/test_adaptive/aggressive/`. The scores and the test still identify the same 45 rows. Other citations have drifted too (`train_only_redteam.rs`, `train_only_redteam_results.rs`, `scq_batch.rs:32`, `main.rs:714`); the current locations are in the "Citation drift (2026-11-09)" table in `assets/regression/SOURCES.md`. A later editor should not treat a stale line number as a reason to drop the correction.
6. **The `notes` column contradicts the corrections.** All 45 `jope-dime` rows in `scores.csv.zst` say "frozen test targets were used only for scoring". The corrections file exists to refute exactly that: the rows are test-selected. Read the corrections, not the `notes` column, for what the rows are.
7. **Do not repair 2259.8 by editing the published snapshot in place.** The published ladder is pinned, including its `leaderboard_sha256`. The honest presentation is the pair of ladders that already exists: published, and benchmark-valid.

## Key files

Scores and ladders:

- `assets/regression/sotarena-tabular-regression-v2/results/tabular-regression-v2/scores.csv.zst` is the compressed score table. SHA-256 of the uncompressed `scores.csv` is `5ec868987c2f4286ae44728fde49215ec6e55e7d27e62100d3823744e2957db6`. 45 `jope-dime` rows, all `benchmark_valid=True`.
- `assets/regression/sotarena-tabular-regression-v2/results/tabular-regression-v2/leaderboard.csv` is the published leaderboard. SHA-256 `2d28ffbd8b5830c6adc39874dfcf98498419ef8c1df6ac5eb135c660013fb24f`.
- `assets/regression/elo-tabular-regression-v2.json` is the published Elo snapshot. Top row `jope-dime`, Elo `2259.8`, rank 1, 11 wins, 45/45, 305 methods, 8,763 rankable rows.
- `assets/regression/leaderboard-tabular-regression-v2.benchmark-valid.csv` is the fair leaderboard. SHA-256 `4c39f51c2e269114f7bbd32728c4627e9aba240518c8950cfb4facce540910fe`.
- `assets/regression/elo-tabular-regression-v2.benchmark-valid.json` is the fair snapshot. 304 methods, 8,718 rankable rows, 45 excluded. `jope-dime` is absent. TabFM Ensemble is rank 1 at 1934.9. `jope-dime-regressor` is rank 31 at 1687.0.
- `assets/regression/protocol-corrections-tabular-regression-v2.json` is the demotion record. SHA-256 `6a831c4a97c2fc29627399809e1860f358a6c3ce69d864a0091463c6d860816d`. Two corrections, 45 rows, both `method_id: jope-dime`.
- `assets/regression/sotarena-tabular-regression-v2/results/tabular-regression-v2/aspfm-acceptance-manifest.json` accepts the three Golden46 replays with `benchmark_valid: true` and `run_id: golden46-new3-20260723-v4`.
- `assets/regression/SOURCES.md` is the provenance note. It states that the published ladder is the best known program per dataset, not a held-out result.

Code that defines and checks the claim:

- `crates/jope-arena/tests/elo_regression.rs` asserts the published top row (`method_id` `jope-dime`, Elo `2259.8`, rank 1, 11 wins) and, in `benchmark_valid_ladder_excludes_the_rows_the_campaign_code_labels_test_adaptive`, asserts that the 3 Golden46 rows and 42 ledger rows are exactly the ones removed.
- `crates/jope-arena/src/elo.rs` records that the historical reporter wrote `jope-dime` 2259.8 on `tabular-regression-v2`.
- `crates/jope-campaigns/src/commands/train_only_redteam.rs` labels `jope-dime` `oracle_reference_only` and the best-test envelope over 63,108 evaluations. `jope-dime-regressor`, TabPFN v3, and TabFM are `fair`.
- `crates/jope-campaigns/src/test_adaptive/aggressive.rs` names the Golden46 run prefix `golden46-new3-20260723-v4` and the three datasets. The validity flags now live in the sibling modules, including `aggressive/runner.rs`.

Later panels, kept separate on purpose:

- `assets/automl-elo/v3/RESULTS.json` and `assets/automl-elo/v3/SOURCES.md` are the six-entrant native-regressor refit.
- `assets/comparisons/full-elo/v1/full-elo.csv` copies the 305 published rows, the 304 fair rows, and the 6 native regressors. `protocol_id` keeps them apart.
- `assets/comparisons/full-elo/v1/PATHS.txt` lists the repository-relative paths of the checked-in Elo receipts, including the Golden45 `scores.csv.zst`.
- `research/flywheel_selfplay/h129_gbm_blend_harm_gate/metrics.json` is the H-wave comparison against `r882` at 1500. It is not a tabular-regression Elo.

## How to cite it

Say: “The published Golden45 ladder rates the JopeDime evolution envelope at 2259.8. That figure is the best test-scored candidate per dataset, 42 ledger minima plus 3 test-adaptive Golden46 accepts. The fair ladder removes those rows. The recorded fixed-seed JopeDime regressor is Elo 1687.0, rank 31 of 304. TabFM Ensemble leads that ladder at 1934.9.”
