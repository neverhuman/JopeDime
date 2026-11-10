# Reference-board protocols

- **AUDIT boards** (`dev91-audit-v1`, `p201b-train-audit-v1`) follow sections 1–4. This is the H335 AUDIT protocol,
  unchanged, so the H335 entrants and the reference models sit on the same footing.
- **The two-fold board** `dev91-s1-two-fold-v1` follows H257's `jope.s1-two-fold/v1` (section 5). Its records were
  imported, not produced, by this directory.
- Numbers from the two protocols are never compared with each other.

## 1. Board
- A board is a fixed list of contexts. Each context has one **AUDIT envelope** (JTT1, `nq = 0`).
  - Fold byte 0 marks the **SEARCH rows** (the only rows a model may fit on).
  - Fold byte 1 marks the **AUDIT rows** (the rows that are scored).
- The split is `common_h317.search_and_audit_envelopes`: the AUDIT rows are a fixed ~30% of each context, chosen by
  the context id alone. It is the exact split H317 and H335 used.
- `BOARD.json` pins every envelope by sha256, with its family id and row counts. `board_sha256` is the sha256 of the
  concatenated `cid + envelope_sha256` strings in cid order. A score is valid only for those bytes.
- Boards never change. A different context list or split is a new board id (`…-v2`).

## 2. Score per context
- Fit on the SEARCH rows only. Predict the AUDIT rows, in envelope order.
- `RMSE_base` = RMSE of the SEARCH-row target mean, evaluated on the AUDIT rows.
- `u = −ln(RMSE_audit / RMSE_base)`, with the ratio clamped to ≥ 1e-12. Higher is better: u = 0 is no better than
  predicting the mean, and u = 1 is an RMSE 63% below it.
- **Failure rule.** A degenerate baseline, an exception, a non-finite prediction or a wall-clock timeout is recorded
  with `status` = `failed` or `timeout`, and **u = 0**. Failures are final, as in H335. A re-score needs a new model id.
- Reference models: one fit, `seed` from `models.json` (0).
- H335 GP entrants: the mean over their 2 search seeds (dev91) or their 1 seed (P201b). This is exactly the value the
  H335 verdict used.

## 3. Ladder (tools/ladder.py; functions imported unchanged from `h335_stats.py`)
- **AUDIT-Elo.** Per family, each pair of entrants scores the mean over the family's contexts of win (1) / tie (0.5,
  |Δu| ≤ 1e-4) / loss (0). Ratings come from an MM Bradley–Terry fit (prior 0.01) and are reported on the Elo scale.
  The anchor `h335.H257_REPLICA` is fixed at 1500. The 95% CI is a 10,000-draw family bootstrap (seed 335160).
  - Every family counts once, whatever its number of contexts.
  - Elo depends on the field: adding an entrant moves the others. Compare within one ladder table only.
- **Head to head** (focus entrant vs each other entrant; field-independent):
  - **gain:** the family-equal mean of u(focus) − u(other);
  - **RMSE cut:** 1 − exp(−gain);
  - **families won / tied / lost;**
  - **the H335 primary procedure:**
    - the contrast is clipped to ±0.5;
    - the one-sided sign-flip p is exact up to 24 families (dev91: 2^19), else from 2^20 Monte Carlo draws;
    - a pass needs p < 0.025, the leave-one-family-out minimum > 0, and the family median ≥ 0.

## 4. What a board number may be compared with
- **Comparable:** any two entrants on the same board, through stored scores.
- **Not comparable:**
  - numbers from a different board, including `dev91-s1-two-fold-v1` against `dev91-audit-v1` (same contexts,
    different rows);
  - H257's published two-fold tables (Joint Elo 1749.51 and the rest: a 25-entrant field anchored at R882, with
    best-of-arms entrants);
  - the S3 single-split records;
  - SOTArena Golden45, panel_v1, same-cut and the AutoML Elo.
  - See [`../README.md`](../README.md).
- **dev91 caveats:**
  - It is a reuse cohort: H257 tuned on it, and H335 was this lineage's first read.
  - It has 19 families, so family-level intervals are wide.
- **P201b caveats:**
  - It is TRAIN. H335 chose G\* and its stack on it, and our screeners trained on TRAIN folds.
  - It is a development board, never evidence of a test result.

## 5. The two-fold board (`dev91-s1-two-fold-v1`, H257 `jope.s1-two-fold/v1`)
- **Contexts.** The same 91 DL-Val91 contexts as `dev91-audit-v1`, with the same row count and family per context.
  The rows are split differently: in half, fold 0 / fold 1. Envelopes are on xbabe2 in
  `~/jope-data/h257/cache/val91_jtt1_envelopes/`, pinned by sha256 in `BOARD.json`.
- **Score per context.**
  - Fit on fold 1 and predict fold 0; fit on fold 0 and predict fold 1.
  - ratio_k = RMSE_k / RMSE of the training fold's target mean on fold k.
  - logratio = mean of the two log ratios; sd_logratio = half their absolute difference.
  - gate = exp(logratio) ≤ 0.90 and both folds beat their baseline.
  - u = `s1_utility_zero_comp` = −(0.9 · logratio + 0.2 · sd_logratio) when the gate passes, else −2.0. A fit
    failure is also −2.0.
  - Definition: the docstring of `evaluate_tabular_foundation_dl_val91.py` (`origin/main` `571873a38c`).
- **Records.** One line per context. `utility` = `s1_utility_zero_comp`. `h257_record` holds the H257 record verbatim,
  minus fold predictions (those are in the S3 archive, in the `sources/` copies).
- **Ladder.** The same `tools/ladder.py` statistics (section 3), anchored at `h257.Opus5p5_Extra` = 1500. It is
  labelled "Elo", not "AUDIT-Elo".
