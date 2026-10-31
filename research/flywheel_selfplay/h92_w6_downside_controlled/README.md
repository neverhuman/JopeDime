# H92 — W6 downside-controlled learner (cached held-family screen)

Status: **ADVANCE_CACHED_SCREEN on adaptive development data.** Three of the four preregistered primaries clear the
full common gate, including the 10% harm tail. The strongest is `HEDGE_CTX`. This is **not** a prospective or fresh
confirmation, and no native fits were spent.

## Hypothesis

Every [H83](../h83_w6_cross_family_cached/README.md) W6 arm failed only the common gate's harm tail (more than 10% of
families losing > .01), and TREAT's mean gain sat in a minority of families. **A learner that limits per-family
downside should clear the gate while keeping most of the mean gain.**

## Setup (fixed before any H92 outcome)

- **Harness.** H92 imports the merged H83 module, pinned to its full sha256, and does not copy it. It therefore uses
  the same W1 folds, the 1,219 DL-Train contexts and 205 families scored out of fold, the same four hash-ordered
  anchors, verified U4, and the same scorers: the W1 program-only PRIOR, the registered rank-8 bilinear TREAT, its
  context-shuffled SHUF, and W1's context GBM CTX.
- **Integrity gate.** The recomputed PRIOR/TREAT/SHUF/CTX U4 values must equal H83's stored per-context values
  exactly, or nothing is reported. Result: **0 mismatches across 1,219 contexts × 4 arms.**
- **Controlled arms.** Ties are broken by program SHA.
  - `HEDGE_s`: the prior's top-2 unrevealed picks plus the top-2 of `s` among the rest. Two of the four picks are
    always shared with PRIOR.
  - `SHORT_s`: the top-4 of `s` within the prior's top-16 unrevealed shortlist.
- **Primaries.** `HEDGE_TREAT`, `SHORT_TREAT`, `HEDGE_CTX` and `SHORT_CTX`, each minus PRIOR. Bonferroni m = 4,
  one-sided α = .05, so the adjusted LCB is the 1.25th percentile. The common gate requires LCB > +.001, family
  median ≥ 0, and at most 10% of families losing > .01.
- **Statistics.** Ten thousand paired family-cluster draws (seed 20261031) share one index matrix. Median and mean
  95% intervals are reported. Elo is family-grouped Bradley–Terry from the canonical `elo_tracker`, with PRIOR = 1500.

## Outcome

| Contrast | Mean U4 | Adj. LCB | Median [95% CI] | Losing > .01 | W/T/L | Gate |
|---|---:|---:|---|---:|---|---|
| **HEDGE_CTX − PRIOR** | **+0.0362** | +0.0169 | **+0.0026 [+0.0012, +0.0052]** | 8.3% | 134/22/49 | **pass** |
| HEDGE_TREAT − PRIOR | +0.0091 | +0.0034 | 0.0000 [0.0000, 0.0000] | 8.8% | 81/73/51 | pass |
| SHORT_TREAT − PRIOR | +0.0089 | +0.0034 | 0.0000 [0.0000, 0.0000] | 9.3% | 82/68/55 | pass |
| SHORT_CTX − PRIOR | +0.0317 | +0.0110 | +0.0036 [+0.0019, +0.0064] | 10.2% | 143/15/47 | fail (harm tail) |
| HEDGE_TREAT − HEDGE_SHUF (placebo) | +0.0045 | −0.0002 | 0.0000 [0.0000, 0.0000] | 13.2% | 82/58/65 | — |
| HEDGE_CTX − CTX (cost of hedging) | +0.0057 | −0.0058 | −0.0002 [−0.0008, 0.0000] | 18.0% | 63/37/105 | — |
| TREAT − PRIOR (H83, reproduced) | +0.0085 | +0.0038 | 0.0000 [0.0000, +0.0002] | 13.2% | 90/59/56 | fail |
| CTX − PRIOR (H83, reproduced) | +0.0306 | +0.0096 | +0.0040 [+0.0021, +0.0099] | 16.1% | 134/8/63 | fail |

Secondary LCBs are one-sided at α = .05 and serve as descriptive references only.

Family-mean U4 by arm:

| PRIOR | TREAT | SHUF | CTX | HEDGE_TREAT | SHORT_TREAT | HEDGE_CTX | SHORT_CTX | HEDGE_SHUF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.6498 | 1.6583 | 1.6548 | 1.6804 | 1.6589 | 1.6587 | **1.6860** | 1.6815 | 1.6544 |

Family Elo (PRIOR = 1500):

| CTX | SHORT_CTX | HEDGE_CTX | TREAT | HEDGE_TREAT | SHORT_TREAT | SHUF | HEDGE_SHUF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1570.2 | 1565.7 | **1550.5** | 1525.0 | 1520.1 | 1514.6 | 1509.0 | 1508.2 |

## Verdict and reading

- **Hedging converts W1's context model into a gate-passing learner.** HEDGE_CTX keeps PRIOR's two best picks and adds
  CTX's two best others. It gives **+0.036 U4 and +50.5 Elo over the generic prior**, with a positive median whose
  interval excludes zero and only 8.3% of families losing more than .01. Unhedged CTX had a higher mean Elo (+70) but
  failed the harm tail (16.1%); hedging trades a little mean for a much thinner downside tail.
- **The registered bilinear treatment passes only because of hedging, not context.** HEDGE_TREAT and SHORT_TREAT pass
  with median 0, but HEDGE_TREAT − HEDGE_SHUF is +0.0045 with LCB −0.0002. The context-specific contribution of the
  low-rank residual is still not established, as in H83.
- **SHORT_CTX misses by 0.2 points of harm** (10.2% against 10%). It is reported as a fail, and the threshold was not
  moved.
- Elo and the raw-utility gate rank CTX and HEDGE_CTX differently. The raw-utility gate is the preregistered decision
  rule; Elo is reported alongside it.

## Red-team checks

- **Whole-family out-of-fold scoring.** The scoring is inherited unchanged from H83: 0 of 205 families span more than
  one W1 fold, the manifest records 0 cross-family table aliases, every held score uses only fold-k objects, and the
  stacking prior on fit slots is W1's inner-pair out-of-fold prediction. The integrity gate proves the scorers are
  H83's.
- **Zero DL-Val, sealed or holdout payload reads.** The inputs are the same as H83's: DL-Train-only priors, context
  features and pools. `selected.json`'s 91 DL-Val metadata rows are parsed with the file and never used. DOPE
  c51a56d0 is not loaded, and no GPU is used.
- **Elo, not AUC; ties are not wins.** |Δ| ≤ 1e-4 counts as a tie, reported in W/T/L and scored 0.5 in Bradley–Terry.
- **Adaptive development.** The controlled arms were chosen after H83 was read on these same 205 families. The
  intervals are conditional development evidence, not fresh confirmation, and the result does not authorize a live
  change.
- **Missing placebo.** A context-shuffled placebo for CTX would require refitting W1's context GBM with shuffled
  context features. W1's fitted models were not retained (see H81, [PR #895](https://github.com/neverhuman/JopeDime/pull/895)),
  so the HEDGE_CTX gain is not yet separated into "context" and "a stronger nonlinear prior".

## Next step

1. **Prospective confirmation of HEDGE_CTX on new programs.** Run the H81 W4/W6 prospective stratum with HEDGE_CTX
   against PRIOR at four native fits per context. This needs W1's fitted fold models persisted (H81 blocker).
2. **A CTX context placebo.** Once W1's models, or a preregistered refit, are available, refit CTX with context
   features shuffled across fit contexts, so the context contribution can be separated from model strength.
3. **No serving or live-loop change follows from a cached development screen.**

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h92_w6_downside_controlled/test_run_downside.py   # 7 synthetic tests
cd research/flywheel_selfplay/h92_w6_downside_controlled
OMP_NUM_THREADS=8 nice -n 15 python3 -I run_downside.py --out ~/jope-data/h92/<run_id> --run-id <run_id> \
  --h83-per-context <H83 per_context.json>   # from s3://.../W6/opus-h83-20261031a/per_context.json.gz
```

## Evidence

Run `opus-h92-20261031a` used xbabe2, nice 15 and 8 threads: 18:28 wall, 2.1 GB peak, zero native fits and zero GPU.
The source sha256 recorded in the metrics equals the committed runner. [metrics.json](metrics.json) holds every number
above and the S3 object versions. The objects live under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/opus-h92-20261031a/` (metrics, gzipped
per-context results, logs and both sources), each written create-only with a version-pinned readback hash. No HWM
candidate was produced.
