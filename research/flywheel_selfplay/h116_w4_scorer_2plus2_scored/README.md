# H116 — W4 two-plus-two with the proposal-scorer prior: scored run of the H112 preregistration

Status: **NO_ADVANCE.** The full panel completed with no gaps. Adaptive two-plus-two adds nothing over the scorer's
top four: −0.0002 [−0.0013, +0.0007], with 40 of 44 families tied. The scorer's top four **again beats random4 by a
wide margin** on fresh seeds: +0.167 [+0.013, +0.445], median +0.005, +259 Elo. That replicates H110's primary A.

## Protocol

This run executes [H112's preregistration](../h112_w4_active_reveal_scorer/PREREG.md), merged in #1022. The run
started automatically on that merge from merged main `774773d9`, after a check that the merged runner was
byte-identical to the dry-run source (`run_h112.py` sha256 `de7d5882…`; the same in the run and analyze receipts).

- **Prior.** The H106-recipe scorer trained on 2,306 logged outcomes (H94 + H103 + H110), with each context's whole
  family held out in the W1 folds. Its out-of-fold utility-scale slopes are 0.84–0.91. The fitted priors are
  byte-identical to the dry run's (sha256 `326f349a…`).
- **Arms.** `s_top4` (anchor); `s_2plus2`, a level-free GP update on the first pair's centred residuals;
  `s_2plus2_swap` (placebo); and `random4`.
- **Panel.** H67 roster (50 contexts / 44 families) × fresh seeds 901–903, with all 16 proposals evaluated: 1,614
  distinct native fits, 0 failures, 0 timeouts.

## Outcome (44 families, 150 / 150 context-seeds; completeness gate passed)

| Contrast | Mean [95% CI] | One-sided LCB | Median [95% CI] | Losing > .01 | W/T/L | Gate |
|---|---|---:|---|---:|---|---|
| **s_2plus2 − s_top4** (primary) | **−0.0002** [−0.0013, +0.0007] | −0.0010 | 0.0000 [0, 0] | 2.3% | 2/40/2 | **fail** |
| s_2plus2 − s_2plus2_swap | +0.0004 [−0.0001, +0.0011] | −0.0000 | 0.0000 [0, 0] | 0.0% | 4/38/2 | — |
| **s_top4 − random4** (H110 replication) | **+0.1668** [+0.0130, +0.4446] | +0.0164 | **+0.0050** [+0.0025, +0.0118] | 2.3% | 35/1/8 | — |

| Arm | Family-mean best of four | Family Elo (s_top4 = 1500) |
|---|---:|---:|
| s_top4 | **2.0484** | 1500.0 |
| s_2plus2 | 2.0482 | 1504.8 |
| s_2plus2_swap | 2.0478 | 1483.6 |
| random4 | 1.8816 | **1240.7** |

## Verdict and reading

- **W4 does not advance, now with a prior that has signal.** The update changed the revealed set in only 11 of 150
  context-seeds, and its effect is a tie: 40/44 families tied, primary LCB −0.0010. H103 failed because the prior had
  no signal. H116 fails because a good scorer's top four leaves the two revealed outcomes almost nothing to correct.
  With 16 proposals and four picks, the level-free GP update on two residuals rarely moves the third and fourth picks
  past the scorer's own ranking.
- **The proposal-distribution scorer replicates.** On fresh seeds, with H110's own outcomes added to its training
  memory, the scorer's top four beats random4 by +0.167 (median +0.005, 35/1/8 families, about +259 Elo). This is the
  second prospective confirmation, after H110, that learning on proposer outcomes is what carries the gain, not
  adaptive revealing.
- **W4 status.** The two-plus-two variants (cached prior, H103; scorer prior, H116) are both NO_ADVANCE. Further W4
  work would need a different acquisition design, such as larger batches or uncertainty-driven picks, not this
  update rule.

## Red-team checks

- **Preregistration came first.** #1022 merged before the run, and the run was armed to start only on the merge
  after the runner-hash check.
- **Complete panel.** 0 gaps, 0 shortfalls and 0 execution errors, so this is the gated verdict.
- **Leakage.**
  - Each context is scored by the fold model that excluded its whole family, together with all of that family's
    logged rows.
  - The slopes are out of fold.
  - The adaptive pair sees only its first pair's outcomes, and all picks are fixed before the remaining fits.
- **Placebo.** The swap differed from the treatment in 19 units. Its contrast is +0.0004, with an LCB of
  −0.00004. The outcomes carried no exploitable information beyond the scorer.
- **Elo, not AUC.** Ties are scored 0.5. 46 of 50 contexts tie between the adaptive arms and the anchor, which is
  why their Elo sits near 1500.
- **Data scope.**
  - DL-Train only.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE is not loaded, and no GPU is used.

## Evidence

Run `opus-h112-20261031a` used xbabe2 at nice 15 with 12 workers: 10:27 wall and 1,614 native fits.
[metrics.json](metrics.json) holds every number above and the S3 object versions. The objects live under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/opus-h112-20261031a/`: manifest, metrics,
run receipt, memory, fold manifest, priors, rows, fits, ledger, logs, the merged-main SHA and the runner. Each was
written create-only with a version-pinned readback hash.
