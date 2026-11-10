# Elo and ladders in this repository: which number is which

Several ladders exist. They use different datasets, splits and fields, so **an Elo number is only comparable to
numbers in the same table**.

**Where are the reference-model scores (TabFM, Kumo, TabPFN, TabICL, TabDPT, Mitra, AutoGluon, CatBoost …)?**
[`reference-board/README.md` section 1](reference-board/README.md#1-where-every-reference-model-score-is) has every
location, in git and S3. Machine-readable form: [`reference-board/SCORE_LOCATIONS.json`](reference-board/SCORE_LOCATIONS.json).
**They are never re-run.**

## Canonical: the reference boards (from 2026-11-05)

All three boards hold stored per-context scores in git, archived to S3. They are rebuilt only by `tools/ladder.py`
from those scores.

| Board | Contexts / split | Reference models stored | Our models stored |
| --- | --- | --- | --- |
| [`dev91-s1-two-fold-v1`](reference-board/boards/dev91-s1-two-fold-v1/LADDER.md) | DL-Val91, 91 / 19 families, H257 two-fold | TabFM, TabFM ensemble, Kumo L / S, TabPFN 3.5 (8 and 1 estimators), TabICL v2, TabDPT 1.3, Mitra v2, 4 LLM regressors | none yet |
| [`dev91-audit-v1`](reference-board/boards/dev91-audit-v1/LADDER.md) | DL-Val91, 91 / 19 families, H335 SEARCH/AUDIT | TabFM, TabFM ensemble, Kumo L, TabPFN 3.5, TabPFN v3, AutoGluon, CatBoost, XGBoost, LightGBM, random forest, ridge, 4 LLM regressors; all 91/91 | the 12 H335 entrants (`H335_AGG`, …) |
| [`p201b-train-audit-v1`](reference-board/boards/p201b-train-audit-v1/LADDER.md) | TRAIN, 201 / 201, H335 SEARCH/AUDIT; development only | CatBoost, XGBoost, LightGBM, random forest, ridge (TabPFN 3.5 33/201: stopped) | the 12 H335 entrants |

The two dev91 boards cover the same contexts but score different rows. Their numbers are not comparable with each
other.

## Other stored score sets

| Set | Where | What | Comparable with the boards? |
| --- | --- | --- | --- |
| dev91 S3 single-split (precomputed, 2026-10) | [`reference-board/external/dev91-s3-single-split-v1.jsonl`](reference-board/external/dev91-s3-single-split-v1.jsonl) (each line: S3 URI + version id) | TabPFN 3.5 outer-train → outer-test; Kumo L / S, TabICL v2, TabDPT 1.3, Mitra v2 from a 75/25 query split | No: different rows, one split, no fold penalty |
| panel_v1 (2026-10-29) | [`reference-board/external/panel_v1/`](reference-board/external/panel_v1/) (+ S3 archive) | 240 regression + 240 binary datasets; Kumo 1874.5, AutoGluon 1601.3, CatBoost 1523.7, `jope-fused-net-v3` 1383.4 | No: different datasets |
| SOTArena Golden45 (published, benchmark-valid) | [`../../assets/regression/SOURCES.md`](../../assets/regression/SOURCES.md), repo `README.md` arena block | 45 public datasets, 305 methods; TabFM Ensemble 1934.9, TabPFN v3 1909.5, `jope-dime-regressor` 1687.0 (#31) | No: different datasets |
| AutoML Elo v1 (4 LLM regressors) | [`../../assets/automl-elo/v1/SOURCES.md`](../../assets/automl-elo/v1/SOURCES.md) | 64 families × 3 folds × 3 seeds | No: different panel (the same 4 regressors are on the boards) |

## Earlier ladders: kept for the record

| Ladder | Where | Relation to the boards |
| --- | --- | --- |
| H335 AUDIT-Elo (12 entrants) | [`../flywheel_selfplay/h335_aggressive_locked_elo/README.md`](../flywheel_selfplay/h335_aggressive_locked_elo/README.md) | `dev91-audit-v1` restricted to the 12 H335 entrants. Reproduced exactly from the stored scores (dev91 `H335_AGG` 1742.6; TRAIN 1679.8). With the reference models added the field changes (dev91 `H335_AGG` 1708.0) |
| H257 phase-2 tournament (2026-11-04) | `origin/main` only: `research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/` (`REPORT.md`, `frontier_elo_comparison.json`; commit `571873a38c`, PR #2120) | Its 13 reference entrants are `dev91-s1-two-fold-v1`, per context. Its published tables (anchor R882, 25 entrants, including H257 "Joint Elo" 1749.51) have a different field. Its H257/H256/R882 entrants have no per-context score records. `evaluate_frontier_vs_h257_elo.py` recomputes them from the `.pt` prediction caches in xbabe2 `~/jope-data/h257/cache/` and `~/jope-data/h256/cache/` |
| Golden45 H-wave loader (pre-stage) | [`h-wave-golden45/README.md`](h-wave-golden45/README.md) | Input validation only; never produced a ladder |
| same-cut v1 / v2 | `../../assets/same-cut/` | Different datasets |
| Historical Elo genealogy | [`../../docs/internal/core-ip/work-orders/evidence/W12-elo-genealogy.md`](../../docs/internal/core-ip/work-orders/evidence/W12-elo-genealogy.md) | Which historical ladders can still be replayed |
