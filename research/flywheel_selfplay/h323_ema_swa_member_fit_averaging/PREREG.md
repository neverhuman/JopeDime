# H323: SWA / EMA snapshot averaging of ENS_VOTE2 member predictions vs H303 NESTED (TRAIN-only, paired)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1500_opus_h323.md` (Grok Bot H-wave pass, 2026-11-04 15:00Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H323_claim.txt`.

**Scope.**
- DL-Train only, through H308's TRAIN-only loader (0 DL-Val rows).
- No dev91 / DL-Val, fresh, sealed, holdout, meta_test or DL-Test.
- CPU selection layer only: DOPE `c51a56d0` is not loaded, and there is no GPU.
- This is not H296's adapter SWA, and no H317/H321/H322 work is touched.

**Publication.** This prereg is pushed non-force to `origin/v1/next` **before any fit or score**.

## Hypothesis

Single-endpoint member scores may be noisy enough that the ENS_VOTE2 switch vote chases a few families. Averaging each
member's predictions over late boosting snapshots of the same fitted booster should stabilize the scores and raise
drop-top-10.

## Design ([run_h323.py](run_h323.py), [evaluate_h323.py](evaluate_h323.py))

**Cells.** H308 partitions P0/P1/P2 × root seeds 30001–30003 × 4 family-grouped outer folds: 36 tasks, 9 cells.

**Regression first.** The archived H315 BASE picks (S3 `H315/picks.npz`, exact version `jCll6z9F…`) must equal the
H303/H308 ENS_VOTE2_NESTED picks on 9/9 cells. The medians must equal .0643163334285782, .04821480030677507 and
.04360142642936269. Otherwise **REGRESSION_FAIL**, and nothing is fitted. In-run, BASE recomputed with the refitted
members must also equal the archived picks on 9/9 cells.

**Declared deviation (before any fit).** The archived H315/H320 locks do not carry the inner members' metadata
(selected n_est, strict margin, file shas). The members are therefore **refitted only with the archived seeds and
parameters**: H303's code and member seeds, unchanged. Each LightGBM booster already contains every tree up to its
fitted count, so snapshots are predictions of the **same booster** at `num_iteration` < n_est. No callbacks or new fits
are needed beyond the refit.

### Arms (same 18 configs, inner folds, member streams and fitted boosters; only the member score changes)

- **BASE.** H303's member score: the ranker and the 8 bag rankers predicted at `num_iteration = n_est`.
- **SWA (primary).**
  - The member score is the uniform mean (weights 0.2 each) of predictions at `num_iteration = max(1, round(p·n_est))`,
    p ∈ {0.6, 0.7, 0.8, 0.9, 1.0}.
  - It is applied to the ranker raw scores and to each of the 8 bag rankers' scores.
- **EMA (secondary).** The same snapshots, with weights w_j ∝ 0.7^(4−j), j = 0…4 from oldest to newest, normalized
  (≈ 0.0837, 0.1195, 0.1707, 0.2439, 0.3484).
- **Unchanged in every arm:**
  - the veto gate p_invalid (full model);
  - the TAU mask, top-4 margin selection, bag votes / Q_FIXED, BAG_SWITCH and the ENS_VOTE2 agreement;
  - H303's inner objective and tie rule (no H320 trimming).

Each arm makes its own inner choice, is refitted on outer_train with its chosen TAU, and is locked with an fsynced
fit_lock. Outer-test packets are guarded until the lock exists. Outer test is scored once per partition × seed.

## Metrics (raw utility vs r882 slot0, paired with BASE)

- family-equal gain (per seed, median, whole-family bootstrap CI with RNG 302150 and 10k draws);
- drop-top-1/5/10 and the LOFO minimum;
- W/T/L, harmed (> 0.01), invalid and switches;
- seed-Jaccard and config agreement with BASE;
- Elo labelled **"reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN"**;
- `metrics.json` carries `"dope": {"composite_sha256_prefix": "c51a56d0", "loaded": false}`.

No AUC, and no speed verdicts.

## Decision (per arm, Bonferroni over 2 arms; ties or margins below +0.002 never advance)

**ADVANCE** iff all four hold:
- **(a)** The median over the 9 cells of the paired (ARM − BASE) drop-top-10 gain is **≥ +0.002**, **and** the paired
  whole-family bootstrap **one-sided LCB at 97.5%** (0.025 quantile) is > 0. The bootstrap uses 10,000 draws, RNG
  323191, and the median over cells as the statistic. Each arm drops its own top 10.
- **(b)** The ARM's median gain is ≥ 90% of BASE's.
- **(c)** The ARM's median harmed share is ≤ BASE's.
- **(d)** On ≥ 2 of 3 root seeds, the median over the 3 partitions of that seed's paired drop-top-10 difference is > 0.

**Else NOT_ADVANCED.** **CONFIG_INSENSITIVE** is reported if the arm's (TAU, k, a) equals BASE's on ≥ 30/36 folds. If an
arm advances, the next step is a one-shot dev91 test vs r3175. That is not part of H323.

## Resources and window

- **Hosts:** xbabe2 CPU, nice 19, `taskset` 32, 34, 35, 63, 71, 78 (93–98% idle; no pair are siblings).
  - Each core and its sibling n±64 are disjoint from H321 (48–50, 52–54, 112–118), H322 (13, 15, 57, 59, 60, 62, 77, 79,
    121, 123, 124, 126), H317 (24–27, 44–47, 88–91, 108–111) and 58/122.
  - The nudge's candidates 76 and 120 were no longer idle, and 115 belongs to H321's range.
- **Memory and load:** `MemoryMax=6G`, `MemorySwapMax=0`; stop if MemAvailable < 20 GiB, host CPU > 85% or load > 85.
- **Scratch:** ≤ 2 GB (inner members of locked tasks are compacted into verified per-task archives), archived create-only
  to S3 `…/H323/` (exact-version GET and sha verify), then deleted.
- **Window:** compute stop 16:45Z; close 17:00Z. If late, the wave closes as INCOMPLETE with no partial scoring.
