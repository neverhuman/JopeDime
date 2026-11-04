# H319: bagged ENS_VOTE2 members vs H303 NESTED (TRAIN-only, paired)

**Verdict: `BAG_NOT_ADVANCED`.** No dev91 follow-up is triggered.

Replacing ENS_VOTE2's 3 members with 9 whole-family-bootstrap members, under a 6/9 (a = 2) or 9/9 (a = 3) vote, does not
make the switch decision more reliable. It removes most of the gain and makes the switches **less** stable across seeds.
Over 9 (partition, seed) cells:
- BAG keeps **28%** of BASE's gain: cell medians +0.0137 vs +0.0482.
- The paired drop-top-10 difference is **negative in 9/9 cells**: median −0.0147, CI [−0.0249, −0.0061].
- **Seed-Jaccard falls** from 0.45 to 0.34.
- Only the harmed share improves: 7.8% vs 9.3%.

**Regression first: exact.** The archived H315 BASE picks equal H303/H308's on 9/9 cells, and the three published medians
reproduce.

**Elo: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.**

**Records.**
- **Prereg:** [PREREG.md](PREREG.md), commit `066e6c0ee3a2fede886f5eceaf2c2c0217653c45`, pushed non-force to
  `origin/v1/next` at about 12:06Z. That was before the regression (12:06:24Z) and any fit (12:07:36Z).
- **Owner:** rd_opus [OPUS].
- **Assignment:** `~/nudges/hw_t1205_opus_h319.md`.

## Hypothesis

ENS_VOTE2's drop-top-10 is negative on dev91 (H310/H312/H314). If that comes from seed-noise switches, bootstrap-
aggregating the members should keep the robust switches and drop the noisy ones. That would raise drop-top-10 and
cross-seed agreement while losing little gain.

## Setup

**Shared.**
- H308 partitions P0/P1/P2 × root seeds 30001–30003 × 4 outer folds.
- H303 code and H287 pins, through H308's TRAIN-only loader (sha `65a86b6d…`, 0 DL-Val rows).
- Guards: 0 outer-test accesses before each fsynced lock and 0 denials; 10,971 outer-test packets were present in memory,
  counted honestly.

**BASE.** The archived H315 BASE picks and choices (S3 `H315/picks.npz` `jCll6z9F…`), used read-only.

**BAG.**
- For each (partition, fold, seed), BASE's (TAU, k, a), with 9 members of the unchanged H302 member recipe.
- Each member is fitted on a whole-family bootstrap of outer_train. Bag RNG: 319190 + 1000·partition + seed + 10·fold.
  Drawn family copies are relabelled as distinct families.
- Switch only when ≥ ceil(9a/3) members pick the same candidate; ties go to slot 0.

**Run.**
- xbabe2 cores 33/55/76/78/115/120, nice 19, 6 workers. Each core and its SMT sibling were disjoint from H316–H318.
- `MemoryMax=6G` (never hit). Host CPU busy: mean 31%, max 41%.
- Fit took 24 min; scoring took 2 min.

## Outcome (vs r882 slot0; 1,219 contexts / 205 families per partition)

| partition | arm | per-seed gain | median [pooled 95% CI] | per-seed LCB95 | drop-top-10 | LOFO min | harmed % | invalid | switches | Elo | seed-Jaccard (4 folds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | BASE | +0.0413 / +0.0643 / +0.0723 | **+0.0643** [+0.0223, +0.1336] | +.0245 / +.0209 / +.0287 | +.0132 / +.0086 / +.0166 | .0345 / .0371 / .0451 | 6.3 / 8.8 / 9.3 | 13 / 14 / 11 | 219 / 300 / 320 | 1726 | .52 .41 .48 .36 |
| P0 | BAG | +0.0187 / +0.0135 / +0.0293 | **+0.0187** [+0.0090, +0.0338] | +.0076 / +.0056 / +.0148 | +.0016 / +.0016 / +.0051 | .0142 / .0111 / .0230 | 8.8 / 6.8 / 9.3 | 16 / 17 / 14 | 162 / 127 / 139 | 1654 | .34 .37 .33 .23 |
| P1 | BASE | +0.0482 / +0.0487 / +0.0383 | **+0.0482** [+0.0222, +0.0839] | +.0216 / +.0252 / +.0196 | +.0102 / +.0141 / +.0127 | .0307 / .0353 / .0333 | 11.2 / 10.2 / 8.3 | 17 / 15 / 16 | 327 / 275 / 285 | 1697 | .55 .43 .41 .39 |
| P1 | BAG | +0.0137 / +0.0069 / +0.0091 | **+0.0091** [−0.0040, +0.0224] | −.0021 / −.0093 / +.0006 | −.0046 / −.0098 / −.0020 | .0088 / .0021 / .0067 | 9.3 / 7.3 / 7.8 | 18 / 18 / 16 | 167 / 149 / 135 | 1660 | .37 .32 .29 .39 |
| P2 | BASE | +0.0436 / +0.0318 / +0.0547 | **+0.0436** [+0.0188, +0.0706] | +.0215 / +.0139 / +.0366 | +.0107 / +.0088 / +.0236 | .0373 / .0265 / .0486 | 10.7 / 7.8 / 10.7 | 16 / 17 / 15 | 315 / 292 / 335 | 1727 | .57 .47 .49 .38 |
| P2 | BAG | +0.0094 / +0.0226 / +0.0312 | **+0.0226** [+0.0091, +0.0375] | −.0054 / +.0111 / +.0156 | −.0060 / +.0040 / +.0044 | .0047 / .0179 / .0249 | 6.8 / 7.8 / 6.3 | 17 / 15 / 12 | 111 / 136 / 140 | 1702 | .34 .39 .33 .31 |

**Invalid picks** (r882 slot0 has 17 per cell): BAG has 18 on two P1 cells, one above r882.

**Paired (BAG − BASE) drop-top-10 per cell:** −0.0115, −0.0070, −0.0115, −0.0148, −0.0239, −0.0147, −0.0166, −0.0047 and
−0.0192. The median is −0.0147, with CI [−0.0249, −0.0061] and LCB95 −0.0231 (10k paired whole-family draws, RNG 319191).

**Cell medians, BASE → BAG:**
- gain +0.0482 → +0.0137;
- harmed 9.3% → 7.8%;
- leave-one-family-out minimum +0.0353 → +0.0111;
- seed-Jaccard 0.450 → 0.338.

## Verdict

**Pre-registered rule.**
- **(a) FAIL:** the paired drop-top-10 median is −0.0147, LCB95 −0.0231.
- **(b) FAIL:** BAG's gain is 28% of BASE's, below 90%.
- **(c) PASS:** harmed 7.8% ≤ 9.3%.
- **(d) FAIL:** seed-Jaccard 0.34 is not > 0.45.

**BAG_NOT_ADVANCED.**

**Reading.**
- Each bootstrap member sees only about 63% of the outer-train families, so it is a weaker ranker.
- Requiring 6 of 9 such members to agree halves the switches.
- The switches that survive are neither the big wins (drop-top-1 falls too) nor more reproducible: Jaccard drops. Which
  candidates reach 6/9 depends on each seed's bootstrap draws.
- On DL-Train, BASE's marginal switches are net positive (as in H315), so removing them costs gain without fixing the
  dev91 tail.
- Selector-level variance reduction (bagging, as with H315's margin gate) is not the lever for the dev91 concentration.

## Deviations (declared)

1. **Cores.** The nudge's candidate cores 101 and 117 were busy at sampling (64% and 0% idle). Six other idle cores were
   used, verified disjoint from H316–H318 including SMT siblings, as recorded in the PREREG.
2. **Regression source.** The regression was done on archived picks, as the nudge directed: H315's BASE vs H303/H308. It
   did not refit BASE. The BAG fits reuse BASE's per-task choices from H315 `metrics.json`.

## Next step

1. **Stop at the selector layer.** Margin gating (H315) and bagging (H319) both trade most of the gain for little or no
   robustness. ENS_VOTE2 NESTED stays the reference selector.
2. **Explain the dev91 tail at the data or pool level.** The negative drop-top-10 on dev91 is not reproduced on DL-Train
   OOF for any variant. Candidates are r3175 vs r882 pool composition, or the 19-family dev91 sample itself. Otherwise
   wait for fresh data (H313: none yet).
3. **No dev91 test is warranted for BAG.**

## Files and custody

| File | Contents |
|---|---|
| [PREREG.md](PREREG.md) | frozen protocol (commit `066e6c0ee3`) |
| [run_h319.py](run_h319.py) | regression / bag fit with guards / outer scoring |
| [evaluate_h319.py](evaluate_h319.py) | regression vs archived picks, judges, LOFO, seed-Jaccard, paired decision |
| [test_h319.py](test_h319.py) | 4 data-free tests |
| [metrics.json](metrics.json) | both arms per partition, paired bootstrap, Jaccard, regression, guards, host CPU |
| [regression_receipt.json](regression_receipt.json), [s3_manifest.json](s3_manifest.json), [manifest.json](manifest.json) | regression, S3 VersionIds, pins |

**S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H319/` holds:
- `h319_bag_locks_models_outputs.tar.gz` (VersionId `xNFekNwVG8exw4OgwepykpEE9ojgdy6T`, sha `ea74bb31…`, 216 MB), with
  the 36 locks and 324 members;
- `metrics_full.json`, `picks.npz`, `regression_receipt.json` and `fit_locks_pre_score.sha256`.

Every object passed an exact-version GET, streamed through sha256, before scratch was deleted.
