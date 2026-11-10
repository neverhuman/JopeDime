# H320: trimmed inner objective for the ENS_VOTE2 config choice vs H303 NESTED (TRAIN-only, paired)

**Verdict: `TRIM_NOT_ADVANCED`.** The choice is config-sensitive: TRIM matched BASE on only **4/36** folds, so this is
not CONFIG_INSENSITIVE. No dev91 test is triggered.

Choosing (TAU, k, a) by the inner gain that survives dropping each config's top-10 inner families (with q95 clipping)
moves the choice strongly toward conservative configs: k = 7 in 26/36 folds and a = 3 in 17/36. That costs gain without
raising the outer gain after the top families. Over 9 (partition, seed) cells:
- The paired TRIM − BASE drop-top-10 is **negative in 9/9 cells**: median −0.0096, CI [−0.0174, −0.0041].
- TRIM keeps **57%** of BASE's gain: +0.0274 vs +0.0482.
- It harms fewer families: 5.9% vs 9.3%.
- Its switches are less seed-stable: Jaccard 0.25 vs 0.45.

**Regressions are exact.** The archived H315 BASE equals the H303/H308 picks on 9/9 cells, and the 3 medians reproduce.
The in-run BASE, recomputed with refitted inner members, also equals the archived picks on 9/9 cells.

**Elo: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.**

**Records.**
- **Prereg:** [PREREG.md](PREREG.md), commit `c0cea7d0e41ff22c2d3deca4715b2aaadad979d9`, pushed non-force to
  `origin/v1/next` at 12:58Z, before the regression and any fit (12:59:21Z).
- **Owner:** rd_opus [OPUS].
- **Assignment:** `~/nudges/hw_t1300_opus_h320.md`.

## Hypothesis

Selecting the config by its inner drop-top-10 gain (q95-clipped, raw utility) should yield outer picks whose
drop-top-10 rises while most of the gain is kept. H316's family normalization and H315/H319's selector changes did not
achieve this.

## Setup

**Shared.**
- H308 partitions P0/P1/P2 × seeds 30001–30003 × 4 outer folds (36 tasks).
- H303 code and H287 pins via H308's TRAIN-only loader (0 DL-Val rows).
- Guard: 0 outer-test accesses before each fsynced lock and 0 denials (10,971 present in memory, counted honestly).
- **BASE** is H303 NESTED, recomputed in-run and equal to the archived picks.
- **TRIM** uses the same 18 configs, the same inner folds and the same inner members (refitted with H303 seeds: the
  archived locks lack the inner member metadata). Only the objective changes:
  1. Per inner fold, clip context gains at the q95 of |gain|.
  2. Take family-equal means.
  3. Drop the top 10 family means and average the rest.
  4. Average over the 4 inner folds.
  5. Choose the argmax; ties within 1e-4 go to fewer switches, then higher k/a, lower TAU.

**Run.**
- xbabe2 cores 55/58/76/78/115/120, nice 19, 6 workers, `MemoryMax=6G`, `MemorySwapMax=0`.
- Host CPU busy: max under 30%.
- Fit took 81 min (12:59–14:20Z); scoring took 1 min.

## Outcome (vs r882 slot0; 1,219 contexts / 205 families per partition)

| partition | arm | per-seed gain | median [pooled 95% CI] | per-seed LCB95 | drop-top-10 | harmed % | invalid | switches | median Elo |
|---|---|---|---|---|---|---|---|---|---|
| P0 | BASE | +0.0413 / +0.0643 / +0.0723 | **+0.0643** [+0.0223, +0.1336] | +.0245 / +.0209 / +.0287 | +.0132 / +.0086 / +.0166 | 6.3 / 8.8 / 9.3 | 13 / 14 / 11 | 219 / 300 / 320 | 1726 |
| P0 | TRIM | +0.0209 / +0.0436 / +0.0425 | **+0.0425** [+0.0131, +0.0765] | +.0097 / +.0210 / +.0106 | +.0036 / +.0068 / +.0009 | 1.0 / 4.9 / 5.4 | 15 / 14 / 13 | 61 / 138 / 196 | 1741 |
| P1 | BASE | +0.0482 / +0.0487 / +0.0383 | **+0.0482** [+0.0222, +0.0839] | +.0216 / +.0252 / +.0196 | +.0102 / +.0141 / +.0127 | 11.2 / 10.2 / 8.3 | 17 / 15 / 16 | 327 / 275 / 285 | 1697 |
| P1 | TRIM | +0.0239 / +0.0186 / +0.0309 | **+0.0239** [+0.0094, +0.0394] | +.0131 / +.0063 / +.0126 | +.0056 / +.0001 / +.0060 | 7.3 / 6.8 / 6.3 | 17 / 18 / 17 | 159 / 146 / 221 | 1701 |
| P2 | BASE | +0.0436 / +0.0318 / +0.0547 | **+0.0436** [+0.0188, +0.0706] | +.0215 / +.0139 / +.0366 | +.0107 / +.0088 / +.0236 | 10.7 / 7.8 / 10.7 | 16 / 17 / 15 | 315 / 292 / 335 | 1727 |
| P2 | TRIM | +0.0118 / +0.0274 / +0.0552 | **+0.0274** [+0.0078, +0.0600] | +.0025 / +.0053 / +.0233 | −.0016 / −.0003 / +.0111 | 4.4 / 5.9 / 7.8 | 15 / 18 / 17 | 58 / 164 / 203 | 1706 |

**Invalid picks** (r882 slot0 has 17 per cell): TRIM has 18 on two cells.

**Paired drop-top-10 (TRIM − BASE) per cell:** −0.0096, −0.0018, −0.0157, −0.0046, −0.0139, −0.0067, −0.0123, −0.0091
and −0.0125. The median is −0.0096, with CI [−0.0174, −0.0041] and LCB95 −0.0160 (10k paired whole-family draws, RNG
320191).

**Cell medians, BASE → TRIM:**
- gain +0.0482 → +0.0274;
- harmed 9.3% → 5.9%;
- LOFO min +0.0353 → +0.0193;
- seed-Jaccard 0.450 → 0.246.

**TRIM's chosen configs:**
- k: 7 in 26/36, 6 in 3/36, 5 in 7/36;
- a: 3 in 17/36;
- TAU: 0.05 in 14/36, 0.10 in 10/36, 0.15 in 12/36.

## Verdict

**Pre-registered rule.**
- **(a) FAIL:** the paired drop-top-10 median is −0.0096, LCB95 −0.0160.
- **(b) FAIL:** TRIM's gain is 57% of BASE's, below 90%.
- **(c) PASS:** harmed 5.9% ≤ 9.3%.

**TRIM_NOT_ADVANCED**, and not config-insensitive (4/36 same).

**Reading.**
- With about 37 families per inner fold, dropping the top 10 leaves a small, noisy remainder. Maximizing it favours
  configs that switch rarely (k = 7, a = 3).
- On DL-Train, ENS_VOTE2's marginal switches are net positive (H315/H319/H320 all agree). Rare-switch configs therefore
  lower both the total gain and the post-top-10 gain.
- This is the third selector-level route (after the H315 gate and H319 bagging) that trades gain for lower harm without
  improving robustness. The dev91 tail is not reproducible on DL-Train OOF.

## Deviations (declared)

1. **Inner members refit.** The archived H315 locks lack the inner member metadata, so the inner members were refit with
   H303's seeds and code, as declared in the PREREG. Their exactness is confirmed by the in-run BASE equalling the
   archived picks on 9/9 cells.
2. **Scratch compaction.** To stay under the 2 GB cap, each completed (locked) task's inner-member directories were
   compressed into a per-task `tar.gz`, file counts were verified, and the directories were removed during the run. All
   archives are inside the S3 tar.
3. **Cores.** The nudge's candidate core 63 was busy (42% idle) and was replaced by 55, which is disjoint from H317/H318
   including SMT siblings.

## Next step

**Stop tuning the ENS_VOTE2 selector on DL-Train for the dev91 tail.** The margin gate (H315), bagging (H319) and the
trimmed objective (H320) all trade gain for lower harm. The tail needs a pool- or data-level explanation (r3175 vs r882
pool composition on dev91, or the 19-family sample) or fresh data (H313: none yet).

## Files and custody

| File | Contents |
|---|---|
| [PREREG.md](PREREG.md) | frozen protocol (commit `c0cea7d0e4`) |
| [run_h320.py](run_h320.py) | regression / 36-task BASE+TRIM fit with guards / outer scoring |
| [evaluate_h320.py](evaluate_h320.py) | regression and in-run check, judges, LOFO, Jaccard, paired decision |
| [test_h320.py](test_h320.py) | data-free tests |
| [metrics.json](metrics.json) | both arms per partition, paired bootstrap, choices, regressions, guards |
| [regression_receipt.json](regression_receipt.json), [s3_manifest.json](s3_manifest.json), [manifest.json](manifest.json) | regression, S3 VersionIds, pins |

**S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H320/` holds:
- `h320_trim_locks_models_outputs.tar.gz` (VersionId `P6_rSyzB0._wHDPAnLWzWA9z3m1qM02L`, sha `e8c112d3…`, 794 MB);
- `metrics_full.json`, `picks.npz`, `regression_receipt.json` and `fit_locks_pre_score.sha256`.

Every object passed an exact-version GET, streamed through sha256, before scratch was deleted.
