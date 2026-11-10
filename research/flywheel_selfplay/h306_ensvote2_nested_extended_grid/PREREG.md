# H306: nested ENS_VOTE2 with an extended grid (does the inner optimum lie beyond H303's edges?)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0630_opus_h306.md` (Grok Bot H-wave pass, 2026-11-04 06:30Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H306_claim.txt`.

**Integrity.**
- CPU only (xbabe2) and DL-Train only (`selected.json` `981960ad…`, 1,219 contexts, 205 families).
- H302's audit guard refuses fresh cohorts, DL-Val, DL-Test, sealed and holdout paths.
- No neural forward or update; DOPE `c51a56d0` is untouched.
- Nothing on xbabe1 or xbabe3. Shadow research only: no serving change.

**Publication.** This prereg is pushed non-force to `origin/v1/next` **before any H306 fit and before any outer-test
score**.

## Why

H303 (prereg `43e00dd31f`, close `7bef8870f7`) is NESTED_CONFIRMED: median family-equal gain +0.0643 vs r882 slot0. But
its inner CV chose two grid edges:
- TAU = 0.05 (the strictest veto) in 8/12 tasks;
- k = 5 (the loosest bag vote) in 8/12 tasks.

So the inner optimum may lie outside H303's grid.

## Step 0 (done before this prereg)

H303's README said its Elo reference was "r3175 slot 0". Slot 0 of every H287 top-8 packet is in fact **r882's**
lowest-cost candidate (`topk.load_packets`, `r882_u`).

A label-only correction was pushed as `e7e6318cc1`: **reference = r882 slot0 (DL-Train OOF), NOT the flywheel incumbent
r3175; r3175-anchored Elo is unmeasured.** H306 uses the same wording.

## Design (H303's code imported with sha256 pins: [manifest.json](manifest.json))

**Step 1: regression first** ([run_h306.py](run_h306.py) `regress`).
- A fresh fit of the single H303 task (outer fold 0, root seed 30001) with H303's own grid.
- It must reproduce the archived H303 fit_lock exactly:
  - the choice;
  - the full 18-row inner table;
  - every member's ranker/veto/bag model sha256, lr, n_est, margin and TAU.
- It must also reproduce the archived fold-0 outer-test picks of seed 30001, by sha256 of the int32 picks.
- Source: S3 `H303/h303_models_locks_and_outputs.tar.gz`, VersionId `hshnLPUZuvpHewykQNTh..mWAGu9KGHo`,
  sha256 `8957cf0e…`.
- A mismatch means **stop and write `REGRESSION_FAIL`**. The fit phase refuses to start without a PASS receipt.

**Step 2: ENS_VOTE2_NESTED_XGRID.** Identical to H303 in:
- the H287 pins (`213b30ad52`) and `selected.json` (`981960ad…`);
- the outer folds (SHA256("H302-outer-v1" NUL family) mod 4);
- root seeds 30001–30003 and their member seeds;
- H303's inner 4-fold family CV (SHA256("H303-inner-v1" NUL str(fold) NUL family) mod 4) and its 1e-4 tie rule (fewest
  inner switches, then higher k, higher a, lower TAU).

The only difference is the **grid** (9 configs, declared here before any fit):
- **TAU ∈ {0.025, 0.05, 0.10}**;
- **member vote k ∈ {4, 5, 6} of 8**;
- **agreement a ∈ {2} of 3**.

After the choice, the members are refit on outer_train and a fit_lock is fsynced. Outer_test is scored once.

## Reporting

**Per seed, plus the median over the 3 seeds:**
- family-equal gain vs r882 slot0 with a 10,000-draw whole-family bootstrap CI (RNG 302150) and LCB95;
- family-median gain with its CI;
- W/T/L;
- harmed share;
- invalid excess with McNemar p;
- switches;
- drop-top-1/5/10 gains and the drop-top-5 LCB95;
- top-10 share.

**Chosen-config histogram:** how many tasks land at the new edges TAU 0.025 and k = 4.

**Paired XGRID − H303 NESTED per seed**, with a whole-family bootstrap CI (RNG 302150, 10,000 draws). H303's picks come
from the S3 tar `picks.npz`, hash-checked against H303's published nested hashes.

**Elo:** MM Bradley–Terry with the family bootstrap CI, **reference = r882 slot0 (DL-Train OOF) = 1500, NOT r3175.**

No AUC, and no speed verdicts.

## Decision ([evaluate_h306.py](evaluate_h306.py); ties never win)

**XGRID_ADVANCES** iff all five hold:
1. The median over seeds of the paired gain (XGRID − H303 NESTED) is > +0.001.
2. The paired one-sided LCB95 is > 0 on ≥ 2/3 seeds.
3. XGRID's harmed share is ≤ 35% each seed.
4. XGRID's invalid excess vs slot 0 is ≤ 3 with McNemar p ≥ 0.05, each seed.
5. XGRID's drop-top-5 LCB95 is > 0 on ≥ 2/3 seeds.

**Otherwise XGRID_NOT_ADVANCED:** the H303 grid stands, and the histogram reports whether choices still pile at an edge.

## Resources

- **Hosts:** xbabe2 CPU only, nice 19. The regression runs on core 120. The fit uses `taskset` 120–127 (84–92% idle in
  an 8 s sample; host 17% busy; rd_astra's 60–63 avoided), with 6 forked 1-thread workers.
- **Memory:** systemd `MemoryMax=8G`; stop if MemAvailable < 20 GiB.
- **Scratch:** ≤ 5 GB under `~/jope-data/h306`, archived to S3 `…/H306/` and deleted at close.
- **Deadlines:** compute stop 08:15Z; close 08:30Z.
