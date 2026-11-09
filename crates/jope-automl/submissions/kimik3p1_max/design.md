# kimik3p1_max

The owner supplied this as a standalone Rust file on 2026-11-07, with no design notes. This page
summarises what the source does, as read during integration. It is not the author's own
description. The source is included byte-for-byte as
[`src/kimik3p1_max.rs`](../../src/kimik3p1_max.rs) (SHA-256
`0c211694ec66605b994595fd8f4cf2b8894df6f156caa3b80f1fcc75f5d6ceb3`, 1,788 lines). It uses only
the standard library, contains no `unsafe` code and has no comments.

## Learning procedure

`fit` validates the shapes and requires finite targets; infinite features are refused and NaN
features are treated as missing. Zero features, fewer than two rows or a constant target take a
constant-mean fast path. Otherwise the fit is a weighted blend, on a standardized target, of a
constant and three base models:

1. a ridge model over up to 256 ranked raw features, with missing-value indicators, `asinh` and
   three hinge terms for up to 24 features, and pairwise products of the top 8; its row subsample
   is capped by design size (at most 50,000 rows);
2. gradient-boosted depth-3 trees (learning rate 0.04, at most 260 rounds, 80% row subsample per
   round);
3. bagged depth-7 trees (at most 140 rounds, 80% row subsample per tree, averaged).

The trees use up to 64 ranked features, each binned at up to 63 midpoint thresholds (64 value
bins) plus a separate missing bin, on at most 60,000 rows. Round counts shrink with a work cap.

The row count sets the blend weights. With 40 to 2,000 rows, 3-fold out-of-fold predictions fit
the four weights by coordinate descent on the simplex, shrunk toward a fixed prior; 2-fold is used
for 2,001 to 6,000 rows. Below 40 rows and above 6,000 rows there is no cross-validation, and the
fixed prior weights are used as they are: `[0.20, 0.35, 0.30, 0.15]` below 40 rows,
`[0.03, 0.22, 0.50, 0.25]` above 6,000 (constant, ridge, boosted, bagged). The final base models
are refitted on all rows.

## Prediction and state

`predict` validates the width and refuses infinite features, then blends the constant and the
three base-model predictions with the fitted weights. Standardized values are clamped to
`±1e6` and the back-transformed output to `±f64::MAX`, so extreme finite inputs stay finite. The checks show it is deterministic for a seed and row-local:
a row's prediction does not depend on the other rows in the batch. A different seed gives
different predictions.

## Expected strengths and limitations

- **Standing.** It is third of six in the AutoML Elo
  ([`assets/automl-elo/v3`](../../../../assets/automl-elo/v3/SOURCES.md)), with Elo 1511 and 95%
  CI 1469–1554. It ranks above `GPT6p1_Sol_Max` on family wins (55%), but Sol's geometric-mean
  RMSE is lower (ratio 0.764, 95% CI 0.546–0.976; not significant after Holm adjustment).
  Gemini's lower RMSE than Kimi's (ratio 0.902) is significant after Holm adjustment.
- **Cost.** Fast: 92 s of fit wall time across 576 fits, at most 1.1 s each, peak memory 14 MiB.
- **Large tables.** Above 6,000 rows the blend weights are fixed, not fitted, so large tables
  rely on the prior. In the integration probe, the in-sample error ratio on a synthetic
  6,001-row table was about twice that of the 6,000-row table.
- **No known defect.** The integration probe found no non-finite output, panic or
  nondeterminism.

## Authorship and checks

- [`authorship.json`](authorship.json) records what is known: the file name, bytes, hash, the
  time the owner supplied it, and the first compilation in this repository. The authoring model,
  session and conditions were not supplied and stay `null`.
- [`verification.json`](verification.json) lists the checks run after integration.
