# H319: bagged ENS_VOTE2 members vs H303 NESTED (TRAIN-only, paired)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1205_opus_h319.md` (Grok Bot H-wave pass, 2026-11-04 12:05Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H319_claim.txt`.

**Scope.**
- DL-Train only, through H308's TRAIN-only loader (sha `65a86b6d…`, 0 DL-Val rows).
- No dev91 / DL-Val rows, no fresh cohorts (v1–v7), no sealed, holdout, meta_test or DL-Test.
- CPU selection layer only: DOPE `c51a56d0` is untouched, and there is no GPU.
- Nothing from H316/H317/H318 is duplicated or touched.

**Publication.** This prereg is pushed non-force to `origin/v1/next` **before any fit**.

## Hypothesis

ENS_VOTE2's switches include seed-noise switches that pay off in only a few families: drop-top-10 is negative on dev91
(H310/H312/H314). Bootstrap-aggregating the members, so that a switch needs agreement across family-resampled members,
should remove those switches. It should raise drop-top-10 and cross-seed stability without losing most of the gain.

## Design ([run_h319.py](run_h319.py), [evaluate_h319.py](evaluate_h319.py))

**Cells.** H308 partitions P0 (SHA256("H302-outer-v1" NUL family) mod 4), P1 and P2 ("H308-outer-v1"/"v2") × root seeds
30001–30003 × 4 outer folds: 36 tasks, 9 (partition, seed) cells.

### BASE: unchanged H303 NESTED, archived by H315, used read-only

- **Picks:** S3 `H315/picks.npz`, VersionId `jCll6z9FGwmpNg.cocb305LcVzFYuieR`, sha `e9e6b16a…`.
- **Per-task choices (TAU, k, a):** from H315 `metrics.json` on `origin/v1/next`.

**Regression first (before any BAG fit).** The H315 BASE picks must equal the archived ENS_VOTE2_NESTED picks on **9/9
cells**: H303 `picks.npz` `3CUeOo9d…` for P0, H308 `picks.npz` `uXsQkA27…` for P1/P2. Their medians, recomputed with
H303's judge, must equal the published P0 +0.0643163334285782, P1 +0.04821480030677507 and P2 +0.04360142642936269.
A mismatch means **REGRESSION_FAIL and a stop**.

### BAG

For each task, use **BASE's chosen (TAU, k, a)**. There is no new inner CV; only the members change.

**Members.** B = 9 members per task. Each is the unchanged H302/H287 BAG_SWITCH member recipe
(`r302.fit_and_freeze_member`, with TAU in force), fitted on one whole-family bootstrap resample of outer_train.

**Draws (exact formula).** `rng = numpy.random.default_rng(319190 + 1000*partition_idx + root_seed + 10*outer_fold)`,
with P0 = 0, P1 = 1, P2 = 2. Then, for b = 0…8 in order:
1. `families = rng.choice(sorted(outer_train families), size=n_families, replace=True)`;
2. `member_seed = int(rng.integers(1, 2**31 - 1))`.

**Relabel.** The c-th copy of a drawn family becomes the distinct family `"<family>#<c>"`, with contexts `"<cid>#<c>"`
sharing the same packets (cluster-bootstrap relabel).

**Vote.**
- Each bag member gives its BAG_SWITCH pick at (TAU, k).
- Switch from slot 0 **iff ≥ ceil(9·a/3) bags pick the same nonzero candidate**: a = 2 needs 6/9, a = 3 needs 9/9.
- The pick is the most-voted nonzero candidate; ties go to slot 0.

**Isolation.** Outer-test packets are guarded (any access raises) until the task's fsynced fit_lock exists. The lock
records built and accessed counts. Outer test is scored once per partition × seed.

## Metrics (raw utility vs r882 slot0; both arms, paired)

- family-equal gain (per seed, median over seeds, whole-family bootstrap CI with RNG 302150 and 10k draws);
- drop-top-1/5/10;
- leave-one-family-out minimum;
- W/T/L, harmed share (loss > 0.01), invalid and switches;
- Elo labelled **"reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN"**.

**Seed stability.** For each partition × outer fold, the mean pairwise **Jaccard** of the switched-context sets across the
3 root seeds, for BASE vs BAG (12 values per arm).

No AUC, and no speed verdicts.

## Decision (ties never advance)

**BAG_ADVANCE** iff all four hold:
- **(a)** The median over the 9 cells of the **paired (BAG − BASE) drop-top-10 gain** is > 0 **and** its paired
  whole-family bootstrap LCB95 is > 0. The bootstrap uses 10,000 draws, RNG 319191, the same family resample for both
  arms, and the median over cells as the statistic. Each arm drops its own top 10.
- **(b)** BAG's median (over cells) gain is ≥ 90% of BASE's.
- **(c)** BAG's median harmed share is ≤ BASE's.
- **(d)** BAG's median seed-Jaccard is > BASE's.

**Else BAG_NOT_ADVANCED.** If ADVANCE, the next step is a one-shot dev91 test vs r3175. That is not part of H319.

## Resources

- **Hosts:** xbabe2 CPU, nice 19, `taskset` 33, 55, 76, 78, 115, 120.
  - These were 93–96% idle in a 10 s sample, with no two of them SMT siblings.
  - Each core and its sibling n±64 are disjoint from H316 (52, 58, 60–63), H317 (24–27, 44–47) and H318 (13, 15, 38, 57,
    59, 125).
  - The nudge's candidates 101 and 117 were busy (64% and 0% idle) at sampling.
  - 6 forked 1-thread workers.
- **Memory:** systemd `MemoryMax=6G`; stop if MemAvailable < 20 GiB or host CPU > 85%.
- **Scratch:** ≤ 2 GB under `/home/ubuntu/jope-data/h319-opus`, archived to S3 `…/H319/` (exact-version GET and sha
  verify), then deleted.
- **Deadlines:** compute stop 14:45Z; close 15:00Z.
