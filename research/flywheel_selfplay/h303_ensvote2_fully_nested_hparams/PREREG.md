# H303: ENS_VOTE2 with fully nested hyperparameters, plus a concentration audit

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0535_opus_h303.md` (Grok Bot H-wave pass, 2026-11-04 05:35Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H303_claim.txt`.

**Integrity.**
- CPU only (xbabe2) and DL-Train only (`selected.json` sha256 `981960ad…`, 1,219 contexts, 205 families).
- H302's audit guard refuses fresh cohorts, DL-Val, DL-Test, sealed and holdout paths.
- No neural forward or update; DOPE `c51a56d0` is untouched.
- Nothing on xbabe1 or xbabe3. Even a pass is shadow research only: there is no serving change.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) **before any H303 fit and before any outer-test
score**.

## Why

H302 (prereg `2a77787c3d`, close `aa224daa9f`) is INNERCV_CONFIRMED_TRAINING_SEEDS for ENS_VOTE2: median family-equal gain
+0.0589 across the true training seeds 30001–30003. The red-team raised two holes:
- **(a) Design leakage.** `bagged.Q_FIXED`, `veto.TAU` and the 6/8 plus 2-of-3 vote thresholds were tuned on all of
  DL-Train (H287–H294) and imported as constants into every outer fold.
- **(b) Concentration.** The family-median gain is 0 on 2/3 seeds, and the CIs are right-skewed.

## Design (H302's code, pins and seeds, imported unchanged)

**Reuse.** H302's [`run_h302.py`](../h302_ensvote2_trainseed_nested_cv/run_h302.py) and
[`analysis_h302.py`](../h302_ensvote2_trainseed_nested_cv/analysis_h302.py) are imported with sha256 pins
([manifest.json](manifest.json)). The member fit function, the H287 modules at `213b30ad52`, the outer fold rule
(SHA256("H302-outer-v1" NUL family) last byte mod 4), the root training seeds 30001–30003 and their member seeds, and the
shared bootstrap (seed 302150, 10,000 whole-family draws) are all H302's.

### Step 1: regression first

H302 is re-run unchanged in a private copy (only its scratch path differs) with a full refit. Its ENS_VOTE2 picks must
reproduce H302's published hashes for all three seeds, including seed 30001's `b47088d981af…`. The evaluator also
recomputes them from the regression's fold locks and checks them again. Any mismatch means **stop and write
`REGRESSION_FAIL`**.

### Step 2: the fully nested arm ENS_VOTE2_NESTED ([run_h303.py](run_h303.py))

The procedure runs for each outer fold k (4) and root seed (3), **inside outer_train only**.

**Grid** (18 configs, declared here before any fit):
- **TAU** ∈ {0.05, 0.10, 0.15};
- **member vote k** (H287's Q_FIXED, the bag-vote threshold of each member's BAG_SWITCH) ∈ {5, 6, 7} of 8;
- **ensemble agreement a** ∈ {2, 3} of 3.

Q_FIXED and "member vote k" are the same parameter in ENS_VOTE2, so it is varied once.

**Inner CV.**
- **Folds:** 4 family-grouped folds of the outer_train families, with fold = SHA256("H303-inner-v1" NUL str(k) NUL family)
  last byte mod 4.
- **Fitting:** for each TAU, the root seed's 3 members are **refit on each inner_train with that TAU in force**.
  H287's `veto.TAU` and the `veto.mask` default are set, so TAU also governs each member's own inner lr/n_est/margin
  selection. k and a act only at aggregation, recomputed exactly from each frozen member's raw outputs.
- **Scoring:** each inner_test is scored once per config. The out-of-fold picks cover all outer_train contexts.

**Choice.** The config with the **highest inner out-of-fold family-equal mean gain** over the outer_train families. Configs
within 1e-4 of the best are tied, and the tie resolves to the **more conservative** setting: fewest inner switches, then
higher k, then higher a, then lower TAU (lower TAU vetoes more).

**Refit, freeze, score.** The 3 members are refit on the full outer_train with the chosen TAU (with TAU 0.10 this is
exactly H302's member fit). A `fit_lock.json` holding the chosen config, the inner table and the member hashes is
fsynced. **Only then** is outer_test scored, once, with the chosen (TAU, k, a). Outer-test rows never enter any choice.

### Step 3: concentration audit, for both ENS_VOTE2 (H302 picks, recomputed) and ENS_VOTE2_NESTED

Per seed: the gain after dropping the top 1/5/10 families by gain; the share of the total gain from the top 10 families;
and the family-median gain with its CI.

## Decision (primary = ENS_VOTE2_NESTED; ties never win)

**NESTED_CONFIRMED** iff all five hold:
1. The median over the 3 seeds of the family-equal gain is > +0.001.
2. The per-seed one-sided LCB95 (0.05 quantile of the shared bootstrap) is > 0 on **3/3 seeds**.
3. The harmed-family share (loss > 0.01) is ≤ 35% each seed.
4. The invalid excess is ≤ 3 and the exact one-sided McNemar p is ≥ 0.05, each seed.
5. **After dropping the top-5 families by gain**, the family-equal gain's one-sided LCB95 (whole-family bootstrap of the
   remaining 200 families, seed 302150) is > 0 on **≥ 2/3 seeds**.

**Otherwise NESTED_NOT_CONFIRMED.** In that case, H302 is labelled **"design-leak-dependent"** if its recomputed picks
pass parts 1–4 while the nested arm does not.

**Reporting.** MM Bradley–Terry Elo (family-level, r3175 slot 0 = 1500) with a family-bootstrap CI (H302's
`summarize_three_seeds`), plus the median and CI over the 3 seeds. No AUC, and no speed verdicts.

## Resources

- **Hosts:** xbabe2 CPU only, nice 19, `taskset` 101–108 (8 cores, each 100% idle in a 5 s sample; rd_argon is busy on
  0–7). LightGBM n_jobs 1 (H287's pin); 6 forked worker processes over the 12 (fold, seed) tasks, in a
  systemd unit with `MemoryMax=8G` (the whole cgroup, so shared pages count once).
- **Memory:** RSS ≤ 8 GiB in total; stop if MemAvailable < 20 GiB.
- **Scratch:** ≤ 5 GB under `~/jope-data/h303`, archived to S3 and deleted at close.
- **Deadlines:** compute stop 07:45Z; close by 08:00Z.

## Custody

- **S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H303/` holds the fit locks, the picks, the
  full metrics and the H302 regression receipt.
- **Git:** no models. This PREREG, the README, `metrics.json` under 50 KB, the code, the tests and the index rows.
