# H119 reproduction index

The full report and compact, machine-readable record are
[README.md](README.md) and [metrics.json](metrics.json). The original prereg is
[PREREG.md](PREREG.md). This index copies no source, model, raw fit output, or
dataset payload.

## Protocol and inputs

- **Protocol:** prospective GBM_PROG412_CTX confirmation against the offline
  r882 reference. The preregistration was committed before scoring and fixes
  the two primary endpoints, Holm rule, harm threshold, and report-only
  secondary panel. Feature counts: 412 program + 48 LM48 + 629 frozen DOPE =
  1,089. HistGradientBoosting uses squared error, 300 iterations, learning
  rate 0.08, 63 leaves, and L2=1; complete parameters are in each
  `metrics.json.fit_receipts[]` entry.
- **Dataset and split:** 1,219 DL-Train contexts / 205 families; five
  whole-family folds, 41 families/fold, seed `11900`, held-context counts
  `[246, 234, 212, 235, 292]`. Sampled training-row identity SHA-256:
  `8c8beb28c1355f7dce1817a2677fddaffa035a0fac7975d54c78d339b98288fb`.
  GBM seeds are `51`, `52`, `53`; sampled unrevealed rows/context use seed
  `700`. The exact input identities and source hashes are in `metrics.json`.
- **Data boundary:** primary uses DL-Train only. The 91-context/19-family
  DL-Val panel is explicitly secondary/report-only and opened after primary
  sealing. No outer-test, holdout, or sealed meta-test payload was read.

## Source, metrics, and loss record

- **Run source:** executed commit `93e5d0e7480bae1890f05cc418ddd0f33526aa91`,
  runner SHA-256 `8ed175fa240c3769f1fcef81e23c6128736bf539f7e771358da84f4dc7278ab7`.
  Prereg commit `da81b26d3653b8f5ef6e25104f214cbceca40eeb`. Report PR
  [#1056](https://github.com/neverhuman/JopeDime/pull/1056), reviewed source
  head `d076ef6860b301960d171a3081b7ab7bfa67bc97`, merge
  `695eb7f6da6a92b13266077b7f356970f46ef905`.
- **Elo:** primary Track 2 U4 is 1549.87 [1525.42, 1575.12] vs r882=1500;
  Track 1 legacy is 1449.18 [1423.61, 1476.28]. Verdict: `NO_ADVANCE` because
  the harm and legacy gates fail. Complete intervals and seed-level ratings
  are in `metrics.json`; Elo, not AUC.
- **Loss availability:** all 48 fit receipts record configuration,
  iteration count, and prediction SHA, but no per-iteration training or
  validation loss series. HistGradientBoosting's configured validation
  fraction is not a retained validation curve. No curve is reconstructed or
  inferred from final metrics.
- **Exact replay custody:** the README lists the exact-version S3 replay
  archive and manifest identities; full raw replay is not copied into Git.

## Reproduction

Follow [PREREG.md](PREREG.md), then use the recorded source/input identities
and exact-version S3 archive in README §“Verification, resources and custody”.
The screen is conditional on the previously exposed corpus; it is not
independent-new-family or serving qualification.
