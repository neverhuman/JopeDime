# Gemini4_argon_high

The owner supplied this as a standalone Rust file on 2026-10-22, with no design notes. This page
summarises what the source does, as read during integration. It is not the author's own
description. The source is included byte-for-byte as
[`src/Gemini4_argon_high.rs`](../../sources/Gemini4_argon_high.rs) (SHA-256
`d5cd85fcf5e2bea8513d7c8665ba9f17b5fc4b70cd1570fdcedd569c376053d3`, 2,931 lines). It uses only
the standard library and forbids `unsafe`.

## Learning procedure

`fit` validates the shapes and requires finite targets; infinite features are refused and NaN
features are treated as missing. Constant targets, a single row and zero features take a
constant-mean fast path. Otherwise the fit is a stacked ensemble of five experts over a shared
feature pipeline:

0. a spline/ridge model over an expanded basis that includes pairwise interactions;
1. an oblivious (symmetric-tree) gradient-boosted model on 64-bin histograms with a NaN bin;
2. a leaf-wise gradient-boosted model boosted on a damped ridge prior;
3. a gradient-boosted model on an asinh-transformed target with smearing back-transformation;
4. a metric k-nearest-neighbour model, used for 8 to 3,000 rows, and the target mean otherwise.

The row count sets the work: 3 to 5 cross-validation folds, at most 224/144/96 active features,
and up to 180 oblivious and 140 leaf-wise trees. Above 30,000 rows a 20,000-row subset sorted by
target is used. Out-of-fold expert predictions fit non-negative simplex stacking weights (they sum
to 1, with a diversity penalty) and a calibration scale. The final state keeps the per-fold units
and a full-data unit.

## Prediction and state

`predict` validates the width and refuses infinite features, then combines the five expert
predictions with the stacking weights. The checks show it is deterministic for a seed and
row-local: a row's prediction does not depend on the other rows in the batch.

## Expected strengths and limitations

- **Strengths.** Broad coverage. It is the second-ranked entrant of four in the AutoML Elo
  ([`assets/automl-elo/v1`](../../../../../../assets/automl-elo/v1/SOURCES.md)), with Elo 1561 and 95%
  CI 1519–1607.
- **Cost.** It is the slowest of the four to fit on that panel: 1,986 s across 576 fits, at most
  56 s each.
- **Accuracy.** Its RMSE advantage over the other entrants is not significant after Holm
  adjustment. Its geometric-mean RMSE is about 15% higher than `GPT6p1_Sol_Max`'s, although it
  beats Sol on 66% of families.
- **Known defect.** With extreme finite inputs, such as features or targets near `±f64::MAX`, the
  source can return non-finite predictions. The source is not patched. The `jope-automl` adapter
  refuses any non-finite prediction with an error instead
  (`gemini4_argon_high_non_finite_output_is_refused_by_the_adapter`). No panel cell triggered it.

## Authorship and checks

- [`authorship.json`](authorship.json) records what is known: the file name, bytes, hash, the
  time the owner supplied it, and the first compilation in this repository. The authoring model,
  session and conditions were not supplied and stay `null`.
- [`verification.json`](verification.json) lists the checks run after integration.
- This source replaces the earlier `gemini-argon-4` candidate as the Gemini entrant.
