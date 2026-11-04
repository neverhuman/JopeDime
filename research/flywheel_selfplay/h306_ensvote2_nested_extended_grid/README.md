# H306: nested ENS_VOTE2 with an extended grid

**Verdict: `XGRID_NOT_ADVANCED`.** The H303 grid stands. This is shadow research only, with no serving change.

**What was compared.** H306 extended H303's nested grid past both edges that H303 kept choosing:
- TAU ∈ {0.025, 0.05, 0.10};
- member vote k ∈ {4, 5, 6} of 8;
- agreement a = 2.

**Result.**
- The inner CV moved to the new k = 4 edge in **10/12** tasks.
- The extended-grid policy does **not** beat H303's nested policy: the paired median is −0.0048, and the paired LCB95 is
  below 0 on 3/3 seeds.
- It still beats r882 slot0 clearly: median +0.0675 [+0.0203, +0.1333].

**Elo reference: r882 slot0 (DL-Train OOF), NOT the flywheel incumbent r3175. r3175-anchored Elo is unmeasured.**

**Records.**
- **Prereg:** [PREREG.md](PREREG.md), commit `00e1a8ca50e01700e738cdc853b4a346dbbb5c6b`, pushed non-force to
  `origin/v1/next` at about 06:31Z. That was before any H306 fit (06:40Z) and any outer-test score (07:05Z).
- **Owner:** rd_opus [OPUS].
- **Assignment:** `~/nudges/hw_t0630_opus_h306.md`.

## Step 0: H303 Elo label correction

Slot 0 of every H287 top-8 packet is r882's lowest-cost candidate (`topk.load_packets`, `r882_u`, costs from
`r882-train-costs.npz`). H303's README and PREREG said "r3175 slot 0".

A label-only correction (no number changes) was pushed to the H303 README as `e7e6318cc1`: "reference = r882 slot0
(DL-Train OOF), NOT the flywheel incumbent r3175; r3175-anchored Elo unmeasured". H305 (rd_astra) reached the same
finding independently. H303's PREREG is frozen and left as written.

## Hypothesis

H303 chose TAU = 0.05 (its strictest veto) and k = 5 (its loosest bag vote) in 8/12 tasks each. If the inner optimum
lies beyond those edges, an extended grid should improve the out-of-fold gain over H303's nested policy.

## Setup

**Same as H303**, imported with sha256 pins:
- `run_h303.py` / `evaluate_h303.py` → H302 → H287 `213b30ad52`;
- `selected.json` `981960ad…` (1,219 DL-Train contexts, 205 families);
- outer folds SHA256("H302-outer-v1" NUL family) mod 4;
- root seeds 30001–30003 and their member seeds;
- H303's inner 4-fold family CV and 1e-4 conservative tie rule;
- bootstrap RNG 302150, 10,000 whole-family draws.

**Different:** only the 9-config grid.

**Regression first: PASS** ([regression_receipt.json](regression_receipt.json)). A fresh fit of the single H303 task
(outer fold 0, seed 30001) with H303's grid reproduced the archived H303 fit_lock exactly:
- the choice (0.05, 5, 2), with inner gain 0.0961251;
- the 18-row inner table;
- every member model sha256, lr, n_est and margin;
- the fold-0 outer-test picks: sha `18fdb2538cea…`, 335 contexts, 74 switches.

The H303 archive is S3 VersionId `hshnLPUZ…`, sha `8957cf0e…`.

**XGRID.** 12 tasks × (3 TAU × 4 inner folds × 3 members + 3 refit members) = 468 member fits:
- xbabe2 cores 120–127, nice 19, 6 forked 1-thread workers;
- systemd `MemoryMax=8G`, never hit; the MemAvailable watchdog never fired;
- fit 06:41:49–07:05:19Z.

The 12 lock hashes were inventoried before scoring. Outer_test was scored once, at 07:05Z. Paired comparison: XGRID
against H303's archived nested picks, hash-checked against H303's published hashes.

## Chosen hyperparameters (inside outer_train only)

| outer fold | 30001 (TAU, k) | 30002 | 30003 |
|---|---|---|---|
| 0 | 0.05, **4** | **0.025**, 5 | 0.05, **4** |
| 1 | 0.05, 6 | 0.05, **4** | 0.10, **4** |
| 2 | **0.025**, **4** | 0.05, **4** | 0.05, **4** |
| 3 | **0.025**, **4** | 0.05, **4** | **0.025**, **4** |

(a = 2 throughout. Bold marks a new edge.)

- **TAU:** 0.025 in 4/12, 0.05 in 7/12, 0.10 in 1/12.
- **k:** 4 in 10/12, 5 in 1/12, 6 in 1/12.

**The choices still pile at an edge: the new k = 4 edge (10/12).** TAU moved off the 0.05 edge only partly (4/12 at
0.025).

## Outcome (out-of-fold, 1,219 contexts / 205 families)

### Primary: paired XGRID − H303 NESTED

| seed | paired family-equal gain [95% CI] | one-sided LCB95 | family W/T/L |
|---|---|---|---|
| 30001 | −0.00517 [−0.01853, +0.00673] | −0.01600 | 58/114/33 |
| 30002 | +0.00321 [−0.00728, +0.01400] | −0.00546 | 48/115/42 |
| 30003 | −0.00480 [−0.01263, +0.00152] | −0.01113 | 34/138/33 |
| **median** | **−0.00480** | LCB > 0 on 0/3 | |

XGRID picks agree with H303's nested picks on 85.7% / 89.2% / 89.9% of contexts.

### XGRID vs r882 slot0

| seed | family-equal gain [95% CI] | LCB95 | family median [CI] | W/T/L | harmed | invalid vs 17 (excess, McNemar p) | switches | Elo [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 30001 | +0.03612 [+0.01743, +0.05739] | +0.01996 | +0.0035 [0, +0.0080] | 112/63/30 | 11.2% | 14 (−3, 0.938) | 289 | 1728.8 [1663.5, 1805.3] |
| 30002 | +0.06752 [+0.01894, +0.13445] | +0.02483 | +0.0064 [+0.0015, +0.0120] | 122/50/33 | 12.2% | 12 (−5, 0.980) | 362 | 1727.1 [1663.1, 1800.3] |
| 30003 | +0.06746 [+0.01688, +0.13703] | +0.02363 | +0.0070 [+0.0036, +0.0130] | 125/41/39 | 10.7% | 11 (−6, 0.989) | 388 | 1702.3 [1642.3, 1769.7] |
| **median** | **+0.06746** [pooled +0.02026, +0.13330] | min +0.01996 | +0.0064 [pooled +0.0015, +0.0115] | | max 12.2% | max excess −3 | | **1727.1** [1665.7, 1780.8] |

H303 NESTED on the same table had a median of +0.06432, harmed of 6.3–9.3% and switches of 219/300/320.

### Concentration (XGRID vs slot0)

| seed | gain after dropping top 1 / 5 / 10 | top-10 share | drop-top-5 LCB95 |
|---|---|---|---|
| 30001 | +0.0293 / +0.0206 / +0.0129 | 66.1% | +0.0099 |
| 30002 | +0.0415 / +0.0223 / +0.0114 | 83.9% | +0.0028 |
| 30003 | +0.0403 / +0.0226 / +0.0116 | 83.7% | +0.0023 |

## Verdict

**Pre-registered rule.**
1. **FAIL:** the median paired gain is −0.0048, not > +0.001.
2. **FAIL:** the paired LCB95 is > 0 on 0/3 seeds.
3. **PASS:** harmed ≤ 12.2%.
4. **PASS:** invalid excess ≤ −3, McNemar p ≥ 0.938.
5. **PASS:** drop-top-5 LCB95 > 0 on 3/3 seeds.

**XGRID_NOT_ADVANCED. The H303 grid stands.**

**Reading.** Looser member votes (k = 4) and a stricter veto (TAU 0.025) make ENS_VOTE2 switch more: about 60–70 more
contexts per seed.
- Those switches spread the gain more evenly. The family median is positive on every seed, and drop-top-5 is positive
  on 3/3 seeds, against 2/3 for H303.
- But the extra switches add as many harmed families as helped ones: harmed rises to 10.7–12.2% from 6.3–9.3%. The net
  mean gain is unchanged within noise.
- The inner CV keeps rewarding looser voting (the k-edge pile persists at k = 4), but the outer folds do not confirm it.
  That pattern is inner-CV optimism at the edge, not an unexplored optimum worth chasing further.

## Deviations (declared)

1. **First fit launch stopped to remove DL-Val decoding.** The first XGRID fit launch (06:40:11Z) was stopped at
   06:41:24Z, before any fit lock and with no outcome.
   - H305's audit (h), published during this wave, found that the H302/H303 loader decodes whole cache arrays whose
     final 91 rows are DL-Val (dev91) features.
   - H306's fit and score phases now memory-map the two `ZIP_STORED` cache members, so only TRAIN rows are decoded.
     The TRAIN values are byte-identical. This was tested against `np.load` on synthetic data and on the real files'
     ID arrays.
   - The XGRID fit was relaunched at 06:41:49Z.
   - **Still affected:** the regression step (06:30Z) and the stopped launch used H303's exact loader, which decodes
     those 91 DL-Val feature rows in memory. They were never used in any fit, selection or score.
   - Partial scratch of the stopped launch (19 MB, no locks) is archived in the S3 tar.
2. **Code edited after the prereg.** `run_h306.py` (the loader) and `test_h306.py` (one added test) changed after the
   prereg. [manifest.json](manifest.json) records both versions.
3. **Outer-test features in memory before locking.** As in H302/H303 (H305 audit), the loader builds all TRAIN contexts'
   features, including outer-test ones, in memory before fit locks. No outer-test row enters any fit or choice. The
   `outer_test_rows_read_before_lock` counter refers to use, not to presence in memory.

## Next step

1. **Stop extending the vote grid.** The k-edge pile moved with the edge and the outer folds did not follow, so keep
   H303's grid and treat k ≤ 5 choices as inner-CV optimism.
2. **Harm-aware selector.** XGRID's even spread (positive family median and drop-top-5 on 3/3) came with more harm.
   A harm-aware selector (for example H291 HARMVETO on top of a k = 4 ensemble) could keep the spread without the extra
   harmed families. That is a separate prereg.
3. **Fresh confirmation.** Confirmation of ENS_VOTE2 still needs a new authorized population (H299).
4. **r3175 Elo.** An r3175-anchored Elo would need r3175 pools, which is out of scope here.

## Files and custody

| File | Contents |
|---|---|
| [PREREG.md](PREREG.md) | frozen protocol (commit `00e1a8ca50`) |
| [run_h306.py](run_h306.py), [evaluate_h306.py](evaluate_h306.py), [test_h306.py](test_h306.py) | regression / fit / score (H303 code with the grid swapped), paired evaluation and decision, 6 data-free tests |
| [metrics.json](metrics.json) | compact metrics: paired, vs slot0, concentration, chosen configs, inner tables, lock hashes |
| [regression_receipt.json](regression_receipt.json), [s3_manifest.json](s3_manifest.json), [manifest.json](manifest.json) | regression, S3 VersionIds, pins |

**S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H306/` holds:
- `h306_models_locks_and_outputs.tar.gz` (VersionId `HuP6NVtw74H9WSBhjGXDtLzg7IN09XqW`, sha256 `073a1db5…`, 287 MB);
- `metrics_full.json` (`2Udw.SEJ…`);
- `regression_receipt.json` (`.Zk.JUtw…`);
- `picks.npz` (`wm3A0Ngy…`).

Local scratch was deleted after the S3 sha was verified.
