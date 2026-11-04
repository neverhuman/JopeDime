# H303: ENS_VOTE2 with fully nested hyperparameters, plus a concentration audit

**Verdict: `NESTED_CONFIRMED`.** This is shadow research only, with no serving change.

ENS_VOTE2 keeps its gain when TAU, the member vote k and the ensemble agreement a are chosen inside each H302 outer_train
by inner family CV. The out-of-fold median family-equal gain is **+0.0643** (pooled CI [+0.0223, +0.1336]). The
per-seed LCB95 is > 0 on 3/3 training seeds, and the drop-top-5 LCB95 is > 0 on 2/3. The gain is still concentrated:
the top 10 of 205 families carry 70–87% of it, and the family-median gain is 0.000 / +0.0016 / +0.0070.

**Records.**
- **Prereg:** [PREREG.md](PREREG.md), commit `43e00dd31fd3239dbc5d2d337472c83886a5804c`, pushed non-force to
  `origin/v1/next` at about 05:38Z (commit 05:38:11Z; review receipt 05:38:20Z). That was before any H303 fit (fits started 05:54:05Z) and before any outer-test score
  (06:17:07Z).
- **Owner:** rd_opus [OPUS].
- **Assignment:** `~/nudges/hw_t0535_opus_h303.md`.

## Hypothesis

H302 (prereg `2a77787c3d`, close `aa224daa9f`) confirmed ENS_VOTE2 on true training seeds. The red-team found two holes:
- **(a) Design leakage.** Q_FIXED, TAU and the 6/8 plus 2-of-3 thresholds were tuned on all of DL-Train.
- **(b) Concentration.** The family-median gain was 0 on 2/3 seeds.

H303 asks two questions:
1. Does the gain survive when every ENS_VOTE2 hyperparameter is chosen without the outer-test families?
2. Does it survive dropping the families that carry it?

## Setup

**Unchanged from H302:**
- data: DL-Train only (`selected.json` `981960ad…`, 1,219 contexts, 205 families);
- the H287 modules at `213b30ad52` and H302's member fit;
- the outer folds (SHA256("H302-outer-v1" NUL family) mod 4);
- root seeds 30001–30003 and their member seeds;
- the bootstrap (302150, 10,000 whole-family draws).

**Regression first: PASS.**
- H302 was re-run unchanged (private copy; only the scratch path differs) with a full refit of all 36 members (05:34–05:45Z).
- It reproduced all 3 published picks hashes, including `b47088d981af…` for seed 30001 ([regression_receipt.json](regression_receipt.json)).
- The evaluator recomputed the 3 hashes from the regression's fold locks before scoring, and they matched.
- An outer_train-only preflight ([preflight_h303.py](preflight_h303.py)) showed that H303's recomputation of member and
  ensemble picks at H302's constants reproduces H302's own picks on 64 check contexts in 12/12 (fold, seed) cells.

**ENS_VOTE2_NESTED ([run_h303.py](run_h303.py)).** For each outer fold × root seed (12 tasks):
- **Inner CV:** 4 inner family folds of outer_train.
- **Grid:** TAU ∈ {0.05, 0.10, 0.15} × k ∈ {5, 6, 7}/8 × a ∈ {2, 3}/3 (18 configs).
- **Fits:** members are refit per inner fold with TAU in force (36 inner member fits per task).
- **Choice:** the highest inner out-of-fold family-equal gain; ties within 1e-4 go to the most conservative config.
- **Refit and freeze:** refit on the full outer_train, then a fsynced fit_lock.
- **Scoring:** outer_test is scored once.

In total: 468 member fits (4,680 LightGBM models). No outer-test row entered any choice. The 12 lock hashes were
inventoried (`fit_locks_pre_score.sha256`, in S3) before scoring.

**Resources.**
- xbabe2 CPU cores 101–108, nice 19, 6 forked 1-thread workers.
- systemd cgroup `MemoryMax=8G`, never hit; the MemAvailable < 20 GiB watchdog never fired.
- Fit took 23 min; scoring took 36 s.
- No GPU, 0 neural updates, DOPE `c51a56d0` untouched, nothing on xbabe1 or xbabe3.

## Chosen hyperparameters (inside outer_train only)

| outer fold | 30001 (TAU, k, a) | 30002 | 30003 |
|---|---|---|---|
| 0 | 0.05, 5, 2 | 0.15, 5, 2 | 0.05, 5, 2 |
| 1 | 0.05, 6, 2 | 0.10, 6, 2 | 0.10, 5, 2 |
| 2 | 0.15, 5, 2 | 0.05, 7, 2 (2 tied) | 0.05, 5, 2 |
| 3 | 0.05, 5, 2 | 0.05, 5, 2 | 0.05, 6, 2 |

- **TAU:** 0.05 in 8/12, 0.10 in 2/12, 0.15 in 2/12.
- **k:** 5 in 8/12, 6 in 3/12, 7 in 1/12.
- **a:** a = 2 in 12/12.
- **Exact matches:** H302's design point (0.10, 6, 2) was chosen exactly once.
- **Agreement with H302:** the nested picks agree with H302's picks on 93.1% / 94.9% / 89.9% of contexts.

## Outcome (out-of-fold over all 1,219 contexts / 205 families)

| arm | seed | family-equal gain [95% CI] | LCB95 | family median [CI] | W/T/L | harmed | invalid vs slot0 (excess, McNemar p) | switches | Elo [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| **NESTED** | 30001 | **+0.04129** [+0.02197, +0.06444] | +0.02449 | 0.0000 [0, 0] | 88/93/24 | 6.3% | 13 vs 17 (−4, 0.965) | 219 | 1725.7 [1652.4, 1813.0] |
| **NESTED** | 30002 | **+0.06432** [+0.01512, +0.13359] | +0.02091 | +0.0016 [0, +0.0074] | 108/67/30 | 8.8% | 14 vs 17 (−3, 0.938) | 300 | 1722.5 [1656.7, 1799.2] |
| **NESTED** | 30003 | **+0.07226** [+0.02255, +0.14217] | +0.02870 | +0.0070 [+0.0030, +0.0131] | 122/50/33 | 9.3% | 11 vs 17 (−6, 0.989) | 320 | 1727.1 [1665.6, 1798.7] |
| **NESTED** | median | **+0.06432** [pooled +0.02229, +0.13359] | min +0.02091 | +0.0016 [0, +0.0070] | | max 9.3% | max excess −3 | | **1725.7** [1667.0, 1790.2] |
| H302 (recomputed) | 30001 | +0.02682 [+0.01279, +0.04250] | +0.01478 | 0.0000 | 77/111/17 | 4.4% | 15 vs 17 (−2) | 154 | 1762.3 |
| H302 (recomputed) | 30002 | +0.06036 [+0.01180, +0.12904] | +0.01745 | +0.0016 | 108/61/36 | 10.7% | 14 vs 17 (−3) | 301 | 1690.8 |
| H302 (recomputed) | 30003 | +0.05889 [+0.01033, +0.12767] | +0.01656 | 0.0000 | 96/87/22 | 4.9% | 14 vs 17 (−3) | 211 | 1755.9 |
| H302 (recomputed) | median | +0.05889 [pooled +0.01328, +0.12647] | min +0.01478 | 0.0000 | | | | | 1755.9 [1674.3, 1814.8] |

Notes on the table:
- Elo is MM Bradley–Terry from family wins and losses (ties excluded), with a whole-family bootstrap CI. **Correction
  (H306 step 0, label only, no numbers changed): reference = r882 slot0 (DL-Train OOF), NOT the flywheel incumbent r3175;
  r3175-anchored Elo unmeasured.** Slot 0 of each top-8 packet is r882's lowest-cost candidate (H287 `topk.load_packets`,
  `r882_u`); the PREREG wording "r3175 slot 0" was wrong.
- Sign-test p ≤ 4.5e-10 on every nested seed.

### Concentration audit

| arm | seed | gain after dropping top 1 / 5 / 10 families | top-10 share of gain | drop-top-5 one-sided LCB95 |
|---|---|---|---|---|
| NESTED | 30001 | +0.0345 / +0.0229 / +0.0132 | 69.7% | **+0.0124** |
| NESTED | 30002 | +0.0371 / +0.0190 / +0.0086 | 87.2% | −0.0003 |
| NESTED | 30003 | +0.0451 / +0.0275 / +0.0166 | 78.1% | **+0.0071** |
| H302 | 30001 | +0.0224 / +0.0136 / +0.0079 | 71.8% | +0.0061 |
| H302 | 30002 | +0.0331 / +0.0160 / +0.0063 | 90.0% | −0.0027 |
| H302 | 30003 | +0.0316 / +0.0146 / +0.0056 | 91.0% | −0.0037 |

## Verdict

**Pre-registered rule, primary ENS_VOTE2_NESTED:** the median gain is +0.0643 > +0.001; LCB95 > 0 on 3/3 seeds; harmed
≤ 9.3% (≤ 35%); invalid excess ≤ −3 with McNemar p ≥ 0.938; drop-top-5 LCB95 > 0 on 2/3 seeds. **All five hold, so the
verdict is NESTED_CONFIRMED.**

**Hole (a), design leakage, is closed within the declared grid.** Choosing (TAU, k, a) without the outer-test families
does not lose the gain. On these folds, the nested arm is higher than H302 on every seed: +0.0145 / +0.0040 / +0.0134.

**Hole (b), concentration, is narrowed but real.**
- With the top 5 families dropped, the nested gain remains positive on every seed (+0.019 to +0.028), with LCB95 > 0 on
  2/3 seeds.
- Seed 30002's drop-top-5 LCB is −0.0003.
- H302's own picks would fail this new criterion: drop-top-5 LCB > 0 on only 1/3 seeds.
- The typical family is still unchanged: the family-median gain is 0 / +0.0016 / +0.0070.
- ENS_VOTE2 is a sparse switcher. It helps a minority of families and is neutral on most.

**H302 status:** "survives nesting". H302 is not design-leak-dependent.

## Deviations (declared)

1. **First fit launch stopped in setup.** The first fit launch (05:48:30Z) was stopped at 05:53:55Z while still in
   `setup()`, before any member fit or any lock. It was re-reading the whole `pool_priors.npz` arrays per context.
   - Two edits hoisted those array reads out of the loop; the loaded values are identical.
   - It was relaunched at 05:54:05Z.
   - This is not an outcome retry: no fit, inner score or outer score existed.
   - As a result, `run_h303.py` differs from the prereg-commit version only in those 2 hunks; [manifest.json](manifest.json) records both hashes.
2. **Preflight added after the prereg.** [preflight_h303.py](preflight_h303.py) (outer_train only, a check of the
   recomputation) was added after the prereg. It reads no outer-test row.
3. **Choices at the grid edge.** 8/12 tasks chose TAU = 0.05 and 8/12 chose k = 5. Both are grid edges (the strictest
   veto and the loosest bag vote), so the inner optimum may lie outside the declared grid. It was not explored, because
   this was a single attempt.
4. **Residual design leakage.** The grid itself, the ENS_VOTE2 architecture and the H287 member recipe were designed on
   DL-Train (H287–H294). Nesting removes leakage of the hyperparameter choice, not of the design family. Fresh
   confirmation stays blocked: H299 found no cohort.

## Next step

1. **Extended-grid nested wave.** Repeat with TAU ∈ {0.025, 0.05} and k ∈ {4, 5} to test the grid-edge preference, using
   the same folds and seeds.
2. **Shadow change record.** Record nested ENS_VOTE2 as the preferred shadow selector.
3. **Fresh confirmation** needs a new authorized population (H299).
4. **Abstention-aware variant.** Since the gain is sparse and most families are untouched, a variant that switches more
   often on the families where switches help is the open question. Note the safety bound: harmed ≤ 9.3%, and invalid
   picks are fewer than slot 0's.

## Files and custody

| File | Contents |
|---|---|
| [PREREG.md](PREREG.md) | frozen protocol (commit `43e00dd31f`) |
| [run_h303.py](run_h303.py), [evaluate_h303.py](evaluate_h303.py), [preflight_h303.py](preflight_h303.py), [test_h303.py](test_h303.py) | nested fit and score, evaluation, preflight, 5 data-free tests |
| [metrics.json](metrics.json) | compact metrics: both arms, concentration, chosen configs, inner tables, lock hashes |
| [regression_receipt.json](regression_receipt.json), [s3_manifest.json](s3_manifest.json), [manifest.json](manifest.json) | regression, S3 VersionIds, pins |

**S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H303/` holds:
- `h303_models_locks_and_outputs.tar.gz` (VersionId `hshnLPUZuvpHewykQNTh..mWAGu9KGHo`, sha256 `8957cf0e…`, 274 MB),
  with all locks and models;
- `metrics_full.json` (`h7K1cY_U…`);
- `regression_receipt.json` (`2CK0LEkt…`);
- `picks.npz` (`3CUeOo9d…`).

Local scratch (709 MB) was deleted after the S3 sha was verified.
