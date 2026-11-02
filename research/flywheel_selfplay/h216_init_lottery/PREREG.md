# H216: random-init lottery with B-locked strict-margin selection

Authority: `~/nudges/hw_t0300_argon_h216.md` (2026-11-02). Owner: `rd_argon` (`[ARGON]`).
Status: **PREREGISTERED, NOT RUN.** Preregister before any scored `A` outcome is read.
CPU `xbabe2` only (`nice -n 19`, `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, no `GPU0`, no `qualification.lock`); offline frozen `r882` copy, `DOPE` frozen (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) verified before and after every forward pass. Zero optimizer updates (`actual_optimizer_updates = 0`). No sealed, holdout, or meta-test rows; no live-flywheel interaction; no speed or AUC claim; no serving promotion.

## Hypothesis and fixed arms

`H180`, `H183`, and `H184` showed that the `adapter_10tok` architecture (`47` tensors / `29,211` parameters) acts as a structural kernel carrier, trained weights add nothing detectable over initial states (`ENS10 - RENS10 = -0.001153` in `H183`), and the untrained 10-member ensemble `RENS10` scored `+0.016972` (`95% CI [-0.014237, +0.050233]`).

`H216` tests whether **selecting a single untrained adapter initialization on the reserved `B` split**—requiring its `10,000`-draw paired family-bootstrap median utility gain over `r882` on `B` to have 10th-percentile `> +0.005` (`LCB80 > +0.005`), else keeping `r882`—yields positive held-out utility lift on `A`. This is a **single-model selection test**, not an ensemble (distinct from `H183`/`H185`/`H190`/`H215`) and not an optimizer sweep.

Fixed arms (`ARMS = ("R882", "LOTTERY16", "PLACEBO", "ORACLE_A")`):
1. `R882` (control): frozen `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`).
2. `LOTTERY16` (sole primary arm): `K = 16` fresh constructor-initialized `adapter_10tok` states (`47` tensors / `29,211` parameters each, zero updates). On each `(seed, outer_fold)`, compute each candidate `k in 1..16`'s equal-family utility gain vector over `R882` across the `41` families in `B_outer` (`41` families) and draw `10,000` paired family bootstrap samples on `B_outer` (`stream_seed("H216", "B_BOOT", seed, outer)`). Candidate `k` is eligible iff its `B_outer` bootstrap-median gain over `R882` exceeds `+0.005` and its 10th-percentile (`q10`) bootstrap gain on `B_outer` exceeds `+0.005`. Among eligible candidates, `LOTTERY16` selects the single candidate `k` with the largest `B_outer` bootstrap-median gain (tie-broken by largest `q10`, then smallest candidate index `k`); if no candidate is eligible, `LOTTERY16` keeps `R882`.
3. `PLACEBO` (null-selection control): applies the exact same strict-margin `B_outer` selection rule (`q10 > +0.005`, max bootstrap-median gain, else `R882`) after family-shuffling the `B_outer` family gain assignments across families and candidate labels using deterministic seed `stream_seed("H216", "PLACEBO_SHUFFLE", seed, outer)`.
4. `ORACLE_A` (diagnostic only, cannot win): hindsight single-model ceiling that picks the single model in `{"R882", 1, ..., 16}` maximizing equal-family mean utility on `A_outer` for each `(seed, outer)`. Strictly diagnostic (`diagnostic_only = True`); excluded from Holm family and cannot trigger `CANDIDATE`.

## Fixed cohort, outer-fold manifest, and random streams

- Cohort: inherited `H183` real-S3 regression closure (`1,219` `DL-Train` contexts across `205` families; `91` `DL-Val` contexts across `19` disjoint families for the `r882` native baseline parity gate `1.1377941898164525`).
- Outer-fold manifest: reuse `H183`'s family-grouped 5-fold outer partition (`fold_seed = 183140`, `fold_manifest.json` SHA-256 `92c4ca7852db8a09072df7bcfec46ee08d7024aa6d4572f33f8eff90f590ff30`). For outer fold `outer in 0..4`:
  - `A_outer = fold[outer]` (`41` families)
  - `B_outer = fold[(outer + 1) % 5]` (`41` families)
  - `FIT_outer = remaining 123 families` (unused; zero optimizer updates).
- Replicate seeds: `SEEDS = (216101, 216102, 216103)`.
- Candidate initialization seeds (`K = 16` per replicate seed, `48` distinct untrained adapter states total):
  - `216101`: `216201..216216`
  - `216102`: `216301..216316`
  - `216103`: `216401..216416`
  All `48` constructor-produced adapter states must have distinct SHA-256 tensor digests and differ from `R882`.
- Paired family bootstrap seed for held-out `A` inference: `BOOT_SEED = 216141` (`10,000` paired family draws across the `205` families).

## Two-stage lock order and statistical decision rule

1. **Baseline & Integrity Gate:** Verify all pinned file hashes, package versions, and `DOPE` composite SHA-256 (`c51a56d0...`), then verify bit-exact `91`-context `DL-Val` native `r882` utilities (`mean = 1.1377941898164525`) and candidate-cost bytes before opening any `TRAIN` outcome pool.
2. **Stage 1 (`B`-Selection Lock Before `A`):** Evaluate `R882` and all `K = 16` untrained candidate adapters on `B_outer` (`outer = 0..4`), compute `LOTTERY16` and `PLACEBO` selections for all `15` `(seed, outer)` cells using `B_outer` outcomes only, persist and `fsync` `b_selection_lock.json` (and create-only S3-verify in full runs) before any `A_outer` selection evaluation or `ORACLE_A` calculation.
3. **Stage 2 (`A` Evaluation & Paired-Family Bootstrap):** Evaluate locked `LOTTERY16`, `PLACEBO`, `R882`, and diagnostic `ORACLE_A` across `A_0 union ... union A_4` (`1,219` contexts / `205` families).
   - Utility averages contexts within each family, then weights the `205` families equally, and takes the median across the `3` replicate seeds (`216101-216103`).
   - `10,000` paired family bootstrap draws (`BOOT_SEED = 216141`) compute 95% confidence intervals and one-sided bootstrap `p`-values.
   - Holm step-down adjustment (`m = 2`) over the two non-oracle contrasts `LOTTERY16 - R882` and `LOTTERY16 - PLACEBO`.
   - Family harm is the fraction of the `205` families whose 3-seed mean utility delta (`LOTTERY16 - R882`) is `< -0.01` (required `<= 10%`, i.e. `<= 20 / 205` families; exact `< 0` family counts also reported).
   - Canonical 400-iteration Bradley-Terry Elo (`R882 = 1500.0`, prior `0.01`, tie margin `1e-4`) with `10,000` paired family bootstrap draws.
   - **Verdict Gate:** `CANDIDATE` iff `LOTTERY16` achieves median family-equal utility gain vs `R882 > 0` (ties = `NO_ADVANCE`), strict `LCB95 > 0`, Holm-adjusted `p < 0.05` (and unadjusted `p < 0.05`), and family harm `<= 10%` (`<= 20 / 205`). Otherwise `NO_ADVANCE`.

## Resource guards and custody

- CPU on `xbabe2` only (`nice -n 19`, `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, torch threads `1`).
- Load guard: `load1 < 85.0`; memory guard: `MemAvailable >= 16 GiB` (records `free -g` and swap `19/19 GiB`); disk guard: `disk_used_pct <= 96.0%`.
- Do not touch `GPU0` (`H215` owns it) and do not take `qualification.lock`.
- Bulk arrays and per-cell score shards go to versioned S3 (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H216/argon-t0300/`) with create-only `IfNoneMatch="*"` and exact `VersionId` + `SHA256` readback verification; only compact `README.md`, `PREREG.md`, `metrics.json`, `manifest.json`, `fold_manifest.json`, `provenance.json`, and source/tests are committed to Git.
