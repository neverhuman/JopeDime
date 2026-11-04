# H315: margin-gated ENS_VOTE2 vs H303 NESTED (TRAIN-only nested CV, paired)

**Verdict: `MARGIN_NOT_ADVANCED`.** H316 (a dev91 test) is not triggered.

The predicted-margin gate does not remove harm on DL-Train; it removes gain. Paired against unchanged H303 NESTED over 9
(partition, seed) cells:
- The MARGIN − BASE drop-top-10 gain is **negative in 9/9 cells**: median −0.0101, paired CI [−0.0175, −0.0051].
- MARGIN keeps **57%** of BASE's family-equal gain: cell medians +0.0275 vs +0.0482.
- It harms fewer families: 6.8% vs 9.3%.

**Regression is exact.** BASE reproduced the archived H303/H308 picks on 9/9 cells and the published medians.

**Elo: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.**

**Records.**
- **Prereg:** [PREREG.md](PREREG.md), commit `ff6c46fec8263a53e7d1d70cb3811aee617638ba`, pushed non-force to
  `origin/v1/next` at about 10:28Z, before any fit (10:28:50Z).
- **Owner:** rd_opus [OPUS].
- **Assignment:** `~/nudges/hw_t1030_opus_h315.md`.

## Hypothesis

On dev91 (H310/H312), ENS_VOTE2's gain vs r3175 is concentrated: drop-top-10 is negative on 6/6 seeds. If marginal
switches are net harm, gating switches on the ensemble's predicted margin should raise the gain after the top families
are removed, without losing much total gain.

## Setup

**Shared.** H303 code with sha256 pins, H308's TRAIN-only loader (0 DL-Val rows), partitions P0/P1/P2 × root seeds
30001–30003 × 4 outer folds, giving 36 tasks.
- **Guard:** outer-test packets were guarded until each fsynced fit_lock: 0 accesses and 0 denials. 10,971 outer-test
  packets were present in memory (1,219 × 9), counted honestly.
- **Run:** xbabe2 cores 102/111/117/118/119/123, nice 19, 6 workers, `MemoryMax=8G` (never hit). Fit took 75 min
  (10:28–11:43Z); scoring took 1 min.

**BASE.** Unchanged H303 NESTED: 18 configs, inner family-equal mean, H303 ties.

**MARGIN.** The same members, folds and seeds, with 18 configs × gate m ∈ {0, q50, q75, q90}.
- **Margin:** the mean over the 3 members of (raw ranker score of the ensemble pick − raw score of slot 0).
- **Quantiles:** taken over the switched inner-train contexts of each inner fold.
- **Inner objective:** the family-equal mean after dropping each inner fold's top-5 family gains.
- **Ties:** larger m first.

**Outer.** Refit on outer_train with the chosen TAU. The threshold is the chosen quantile of the refit members' switched
outer-train margins.

**MARGIN's inner choices:**
- gate 0 in 21/36, q50 in 7, q75 in 1, q90 in 7;
- agreement a = 3 in 8/36 (BASE always chose a = 2);
- the same (TAU, k, a) as BASE in 9/36.

## Outcome (vs r882 slot0; out-of-fold over 1,219 contexts / 205 families per partition)

| partition | arm | per-seed gain | median [pooled 95% CI] | per-seed LCB95 | drop-top-10 per seed | harmed % | switches | family median | median Elo |
|---|---|---|---|---|---|---|---|---|---|
| P0 | BASE (= H303) | +0.0413 / +0.0643 / +0.0723 | **+0.0643** [+0.0223, +0.1336] | +0.0245 / +0.0209 / +0.0287 | +0.0132 / +0.0086 / +0.0166 | 6.3 / 8.8 / 9.3 | 219 / 300 / 320 | 0 / .0016 / .0070 | 1726 |
| P0 | MARGIN | +0.0275 / +0.0286 / +0.0439 | **+0.0286** [+0.0067, +0.0640] | +0.0159 / +0.0022 / +0.0117 | +0.0086 / −0.0035 / +0.0012 | 4.4 / 5.4 / 5.9 | 154 / 206 / 204 | 0 / 0 / 0 | 1738 |
| P1 | BASE (= H308) | +0.0482 / +0.0487 / +0.0383 | **+0.0482** [+0.0222, +0.0839] | +0.0216 / +0.0252 / +0.0196 | +0.0102 / +0.0141 / +0.0127 | 11.2 / 10.2 / 8.3 | 327 / 275 / 285 | .0030 / .0036 / .0033 | 1697 |
| P1 | MARGIN | +0.0230 / +0.0148 / +0.0251 | **+0.0230** [+0.0057, +0.0380] | +0.0104 / +0.0027 / +0.0076 | +0.0020 / −0.0025 / +0.0026 | 6.8 / 7.8 / 6.3 | 105 / 144 / 180 | 0 / 0 / 0 | 1688 |
| P2 | BASE (= H308) | +0.0436 / +0.0318 / +0.0547 | **+0.0436** [+0.0188, +0.0706] | +0.0215 / +0.0139 / +0.0366 | +0.0107 / +0.0088 / +0.0236 | 10.7 / 7.8 / 10.7 | 315 / 292 / 335 | .0063 / .0036 / .0081 | 1727 |
| P2 | MARGIN | +0.0361 / +0.0258 / +0.0455 | **+0.0361** [+0.0121, +0.0608] | +0.0149 / +0.0097 / +0.0184 | +0.0046 / +0.0072 / +0.0063 | 11.2 / 6.8 / 7.3 | 251 / 224 / 169 | .0007 / 0 / 0 | 1720 |

**Invalid picks** (r882 slot0 has 17 per cell):
- BASE: 11–17 per cell.
- MARGIN: 14–18. The 18 is on P2 seed 30002, one above r882.

**Paired drop-top-10 difference (MARGIN − BASE) per cell:** −0.0046, −0.0121, −0.0154, −0.0082, −0.0165, −0.0101, −0.0061,
−0.0015 and −0.0173. The median is −0.0101, with CI [−0.0175, −0.0051] and LCB95 −0.0162 (10k paired whole-family draws,
RNG 315150).

## Verdict

**Pre-registered rule.**
1. **FAIL:** the paired drop-top-10 median is −0.0101 and the LCB95 is −0.0162.
2. **FAIL:** MARGIN's gain is 57% of BASE's, below 90%.
3. **PASS:** MARGIN's median harmed is 6.8% vs 9.3%.

**MARGIN_NOT_ADVANCED.**

**Reading.**
- On DL-Train, BASE's switches outside its top-10 families are **net positive**: drop-top-10 is +0.006 to +0.024 in all
  9 cells.
- A margin gate (or the stricter a = 3 that the drop-top-5 objective also favoured) removes those small wins together
  with some harm.
- The drop-top-10 deficit seen on dev91 vs r3175 (H310/H312) is therefore **not** reproduced on DL-Train OOF vs r882.
  It is more likely specific to the dev91 / r3175-pool setting (19 families; a different candidate pool) than a general
  property of marginal switches.
- The gate trades gain for a lower harmed share (about −2.5 points). That trade is unattractive at a 43% gain cost.

## Deviations (declared)

1. **Regression inside the run.** The regression ran inside the same 36 tasks (BASE computed alongside MARGIN) and was
   checked before any MARGIN metric. That is a full regression rather than a single-task pre-check, as declared in the
   PREREG.
2. **Moot gates.** In 2/36 tasks MARGIN chose a gated config whose outer-train ensemble made no switch, so the outer
   threshold was undefined (null) and the gate was moot. Those cells score the ungated pick, which also switches rarely.
3. **Margin definition.** "Predicted utility margin" was implemented as the mean raw ranker-score margin of the 3
   members, as declared in the PREREG. The rankers' scores are relevance scores, not calibrated utilities.

## Next step

1. **Drop margin gating.** On TRAIN it removes net-positive marginal switches. Do not tune its quantiles further.
2. **Diagnose the dev91 tail at the pool level instead.** H310 already showed the chooser on r882's dev91 pool vs r3175
   at +0.0132 with a similar profile. The next useful step is an outcome-free comparison of r882 and r3175 pool
   composition on DL-Train-like contexts, or fresh data (H313: none yet), rather than another selector knob.
3. **The confirmation path is unchanged.** It still needs a new ingest, then a census, then H314 with H313's 6 locks.

## Files and custody

| File | Contents |
|---|---|
| [PREREG.md](PREREG.md) | frozen protocol (commit `ff6c46fec8`) |
| [run_h315.py](run_h315.py) | 36-task BASE+MARGIN fit with guards, outer scoring |
| [evaluate_h315.py](evaluate_h315.py) | regression vs archived picks, judges, paired decision |
| [test_h315.py](test_h315.py) | 3 data-free tests |
| [metrics.json](metrics.json) | both arms per partition, paired bootstrap, choices, regression, guard counts |
| [s3_manifest.json](s3_manifest.json), [manifest.json](manifest.json) | S3 VersionIds, pins |

**S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H315/` holds:
- `h315_locks_models_outputs.tar.gz` (VersionId `l0TNL39iFbqgDPNKuDCuKl54F9s4zEcU`, sha `89b7daec…`, 792 MB), with all 36
  locks and members;
- `metrics_full.json`, `picks.npz` and `fit_locks_pre_score.sha256`.

Every object passed an exact-version GET, streamed through sha256, before scratch was deleted.
