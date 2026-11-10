# H320: trimmed inner objective for the ENS_VOTE2 config choice vs H303 NESTED (TRAIN-only, paired)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1300_opus_h320.md` (Grok Bot H-wave pass, 2026-11-04 13:00Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H320_claim.txt`.

**Scope.**
- DL-Train only, through H308's TRAIN-only loader (sha `65a86b6d…`, 0 DL-Val rows).
- No dev91 / DL-Val, no fresh (v1–v7), no sealed, holdout, meta_test or DL-Test.
- CPU selection layer only: DOPE `c51a56d0` is untouched, and there is no GPU.
- No H317/H318 work is touched or duplicated.

**Publication.** This prereg is pushed non-force to `origin/v1/next` **before any fit**.

## Hypothesis

ENS_VOTE2's dev91 drop-top-10 is negative on 6/6 seeds. If the inner CV selects (TAU, k, a) by a trimmed objective (the
inner gain left after removing the config's own best inner families, with outlier contexts clipped), the outer picks'
drop-top-10 should rise while most of the gain is kept.

## Design ([run_h320.py](run_h320.py), [evaluate_h320.py](evaluate_h320.py))

**Cells.** H308 partitions P0/P1/P2 × root seeds 30001–30003 × 4 outer folds: 36 tasks, 9 cells.

**Regression first.** The archived H315 BASE picks (S3 `H315/picks.npz`, VersionId `jCll6z9F…`, sha `e9e6b16a…`) must
equal the H303 (P0) / H308 (P1, P2) ENS_VOTE2_NESTED picks on 9/9 cells, and their medians must equal the published ones
(+0.0643163334285782, +0.04821480030677507, +0.04360142642936269). Otherwise **REGRESSION_FAIL**, with no fit.

**BASE.** H303 NESTED, unchanged. It is recomputed in the same tasks, and its outer picks must also equal the archived
BASE picks on 9/9 cells (in-run regression).

**Inner members.** The archived H315 locks do not carry the inner members' metadata (n_est, margins, shas), so the inner
members are **refit** with H303's code and member seeds. This is exactly H303's inner CV: the same inner folds
SHA256("H303-inner-v1" NUL str(fold) NUL family) mod 4, and the same streams.

**TRIM.** The same 18 configs and the same inner predictions; only the selection objective changes.
- **Per inner fold i and config:**
  1. Compute g_c = u(pick_c) − u(slot0) for inner-test contexts c of fold i, using raw utility.
  2. Set q = `numpy.quantile(|g|, 0.95)` (linear) over those contexts.
  3. Clip: g′_c = clip(g_c, −q, q).
  4. Take the family-equal means of g′ over the fold's families.
  5. Drop the 10 largest family means and take the mean of the rest.
- **Objective:** the mean over the 4 inner folds.
- **Choice:** argmax; ties within 1e-4 go to fewer inner switches, then higher k, higher a, lower TAU.

**Outer.** Refit on outer_train with each chosen TAU. The fit_lock is fsynced, and outer-test packets are guarded (any
access raises) until it exists. Outer test is scored once per partition × seed.

## Metrics (raw, unclipped utility vs r882 slot0; both arms, paired)

- family-equal gain (per seed, median over seeds, whole-family bootstrap CI with RNG 302150 and 10k draws);
- drop-top-1/5/10 and the LOFO minimum;
- W/T/L, harmed (> 0.01), invalid and switches;
- seed-Jaccard of the switched sets;
- config agreement with BASE;
- Elo labelled **"reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN"**.

No AUC, and no speed verdicts.

## Decision (ties never advance)

**TRIM_ADVANCE** iff all three hold:
- **(a)** The median over the 9 cells of the paired (TRIM − BASE) drop-top-10 gain is > 0 **and** its paired whole-family
  bootstrap LCB95 is > 0. The bootstrap uses 10,000 draws, RNG 320191, and the median over cells as the statistic. Each
  arm drops its own top 10.
- **(b)** TRIM's median gain is ≥ 90% of BASE's.
- **(c)** TRIM's median harmed share is ≤ BASE's.

**Else TRIM_NOT_ADVANCED.** If TRIM's (TAU, k, a) equals BASE's on ≥ 30/36 folds, **CONFIG_INSENSITIVE** is reported
alongside. If ADVANCE, the next step is a one-shot dev91 test vs r3175. That is not part of H320.

## Resources

- **Hosts:** xbabe2 CPU, nice 19, `taskset` 55, 58, 76, 78, 115, 120 (91–97% idle; host 77% idle).
  - Each core and its SMT sibling are disjoint from H318 (13, 15, 38, 57, 59, 125) and H317 (24–27, 44–47).
  - The nudge's candidate 63 was 42% idle and is replaced by 55.
  - 6 forked 1-thread workers.
- **Memory:** systemd `MemoryMax=6G`, `MemorySwapMax=0`; stop if MemAvailable < 20 GiB or host CPU > 85%.
- **Scratch:** ≤ 2 GB under `/home/ubuntu/jope-data/h320-opus`, archived to S3 `…/H320/` (exact-version GET and sha
  verify), then deleted.
- **Deadlines:** compute stop 14:45Z; close 15:00Z.
  - The 36 tasks take about 60 min on 6 workers.
  - If the stop arrives first, unfinished tasks are reported as such, and no verdict is claimed on missing cells.
