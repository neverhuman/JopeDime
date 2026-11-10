# H315: margin-gated ENS_VOTE2 vs H303 NESTED (TRAIN-only nested CV, paired)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1030_opus_h315.md` (Grok Bot H-wave pass, 2026-11-04 10:30Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H315_claim.txt`.

**Scope.**
- DL-Train only, through H308's TRAIN-only prefix loader. No dev91 / DL-Val rows, no fresh cohorts (v1–v7), no sealed,
  holdout, meta_test or DL-Test.
- CPU selection layer only: DOPE `c51a56d0` is untouched, and the xbabe2 GPU is not touched.
- Shadow research only.

**Publication.** This prereg is pushed non-force to `origin/v1/next` **before any fit**.

## Why

The H310/H312 gain vs r3175 slot0 is concentrated: drop-top-10 is negative on 6/6 seeds. ENS_VOTE2 switches that are not
big wins are slight net harm. H303's 18-config grid has no knob to stop marginal switches, which is why H309's trimmed
objective changed nothing. H315 adds a gate on the ensemble's predicted margin.

## Design ([run_h315.py](run_h315.py), [evaluate_h315.py](evaluate_h315.py))

**Shared setup.**
- Code: H303's, imported with sha256 pins (H287 `213b30ad52`, `selected.json` `981960ad…`), via H308's TRAIN-only
  loader.
- Partitions: P0 (SHA256("H302-outer-v1" NUL family) mod 4), P1 and P2 (H308 salts "H308-outer-v1"/"v2").
- Root seeds 30001–30003 with their member seeds, giving 36 (partition, fold, seed) tasks.
- In each task, outer-test packets are guarded (any access raises) until the task's fsynced fit_lock exists. The fit_lock
  records built and accessed counts. Outer-test rows are scored once, in the score phase.

**BASE.** Unchanged H303 NESTED:
- 18 configs, TAU {0.05, 0.10, 0.15} × k {5, 6, 7} × a {2, 3};
- inner 4-fold family CV;
- highest inner OOF family-equal mean, with 1e-4 ties going to fewest switches, then higher k, higher a, lower TAU.

**MARGIN.** The same members, folds and seeds (paired), with 18 configs × gate **m ∈ {0, q50, q75, q90}** = 72 configs.
- **Predicted margin of a context:** the mean over the 3 members of the raw ranker score of the ensemble's pick minus
  the raw score of slot 0. It is 0 when the ensemble keeps slot 0.
- **Gate:** keep a switch only if its margin ≥ the threshold. m = 0 means no gate.
- **Thresholds:** q50/q75/q90 are quantiles of the margins of the **switched inner-TRAIN contexts of that inner fold**,
  scored in-sample by that inner fold's members. They are computed inside inner folds only.
- **Inner objective:** the family-equal mean of the inner-OOF family gains after **dropping each inner fold's top-5
  family gains**.
- **Ties (1e-4):** larger m first, then H303's order.
- **Outer:** refit on outer_train with the chosen TAU. The threshold is the chosen quantile of the switched outer_train
  margins of the refit members (in-sample).

## Regression first

BASE's outer picks must equal the archived ENS_VOTE2_NESTED picks for **every partition and seed**: H303 for P0
(S3 `H303/picks.npz` `3CUeOo9d…`), H308 for P1 and P2 (S3 `H308/picks.npz` `uXsQkA27…`). The BASE medians must equal the
published ones (P0 +0.0643163, P1 +0.0482148, P2 +0.0436014). A mismatch means **REGRESSION_FAIL**, and MARGIN is not
evaluated.

BASE is computed in the same tasks as MARGIN, which is a stronger regression (all 36 tasks) than a single-task check. It
is evaluated **before** any MARGIN metric.

## Metrics (per arm, partition, seed; vs r882 slot0)

- family-equal gain (median and pooled CI over seeds, RNG 302150, 10k whole-family draws);
- drop-top-1/5/10 gains;
- W/T/L, harmed share (loss > 0.01), invalid and switch rate;
- Elo labelled **"reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN"**.

No AUC, and no speed verdicts.

## Decision (ties never win)

**MARGIN_ADVANCE** iff all three hold:
1. The median over the 9 (partition, seed) cells of the **paired (MARGIN − BASE) drop-top-10 gain** is > 0, **and** its
   paired whole-family bootstrap LCB95 is > 0. The bootstrap uses 10,000 draws, RNG 315150, the same family resample for
   both arms and all cells, and the median over cells as the statistic. Each arm drops its own top 10 families.
2. MARGIN's median (over cells) family-equal gain is ≥ 90% of BASE's.
3. MARGIN's median harmed share is ≤ BASE's.

**Else MARGIN_NOT_ADVANCED.** If ADVANCE, the next step is H316, a one-shot dev91 test vs r3175. That is not part of
H315.

## Resources

- **Hosts:** xbabe2 CPU, nice 19, `taskset` 102, 111, 117, 118, 119, 123 (93–96% idle; host 87% idle; the qualification
  services on 0–95 are avoided), with 6 forked 1-thread workers.
- **Memory:** systemd `MemoryMax=8G`; stop if MemAvailable < 20 GiB or host CPU > 85%.
- **Scratch:** ≤ 3 GB, archived to S3 `…/H315/` (exact-version GET and sha verify), then deleted.
- **Deadlines:** compute stop 12:45Z; close 13:00Z.
