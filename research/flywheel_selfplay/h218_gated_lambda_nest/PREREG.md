# H218 — nested blend-weight (`lambda in {0.125, 0.25, 0.5}`) + outcome-free `zgap` gate on FIT, xbabe2 CPU, 0 updates

**Owner:** rd_argon [ARGON]. **Assignment:** `~/nudges/hw_t0400_argon_h218.md`. **Claim:**
`~/jope_plan/ns/receipts/H218_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h218-argon`, branch
`rd/h218-gated-lambda-nest`.

This is an offline CPU diagnostic study using a frozen `r882` copy, not the live flywheel loop. `DOPE` stays
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, and `0` optimizer updates are performed.
**This prereg merges before the 1-cell non-scored smoke or full run launches.**

## Hypothesis

[H210](../h210_blend025_gated_retry/README.md) `GATED_025` (`BLEND` with fixed `lambda = 0.25` of calibrated HistGBM vs
`r882`, gated per context by the outcome-free `zgap` statistic with `tau` chosen on `FIT` only) achieved pooled-`A`
native gain `+0.0140 [+0.0022, +0.0355]` (`3/3` seeds positive, harm `3.4%`, Elo `1505.6`), but moved `DL-Val`
`flywheel91` by only `+0.0010` with `lambda` fixed at `0.25`.

**`H218` tests whether choosing `lambda in {0.125, 0.25, 0.5}` jointly with `tau`, nested on `FIT` only (inner 3-fold
family CV out-of-fold predictions, maximizing `FIT` equal-family native gain subject to `FIT` harm `<= 5%`, else
`r882`), lifts the `DL-Val` `flywheel91` plateau beyond `+0.002` while preserving reliable held-out `A` native gain.**

## Recipe (core functions byte-identical to H210/H217; only `lambda` becomes a grid)

A unit test compares the function bodies of `zgap`, `disagreement`, `tau_grid`, `gated`, `choose_tau`, and `blend`
against `H210` and `H217` byte-for-byte.
- **Blend grid (`LAMBDA_GRID = (0.125, 0.25, 0.5)`):** `blend(r882, GBM_cal, lam) = r882 + lam * (GBM_cal - r882)` in
  cost space (`H129`).
- **Outcome-free gate (`zgap`):** `zgap(r882, blend_lam) = (r882[b] - r882[a]) / std(r882)`, where `a` is `r882`'s
  first argmin and `b` is `blend_lam`'s first argmin (`0` when picks agree, `+inf` for constant `r882`). Uses costs
  only (no fitness, gate, or utility). The gate is open iff `zgap <= tau`; otherwise the arm falls back to `r882`.
- **Nested `(lambda, tau)` procedure on `FIT` (`choose_lambda_tau`):**
  - **Inner 3-fold family CV out-of-fold predictions:** For cell `(seed, outer=k)` with `A = k`, `B = (k+1) % 5`, and
    `FIT = the other three folds`, the three inner HistGBM models (each trained excluding `{A, B, j}` for `j in FIT`)
    produce out-of-fold predictions on `FIT`, calibrated with positive slope `(fa, fb)` on `FIT` only.
  - **Per-`lambda` threshold search:** For each `lam in {0.125, 0.25, 0.5}`, `choose_tau` evaluates `[-inf, FIT q10,
    q25, q50, q75, q90 of switching zgap > 0, +inf]` on `FIT` and picks the `tau` maximizing `FIT` equal-family native
    gain subject to `FIT` harm `<= 5%` (ties -> smaller `tau`).
  - **Joint selection across `lambda in {0.125, 0.25, 0.5}`:** `choose_lambda_tau` picks the `(lam, tau)` maximizing
    `FIT` equal-family native gain across `lam in {0.125, 0.25, 0.5}` subject to `FIT` harm `<= 5%` (ties -> smaller
    `tau`, then smaller `lam`). When no open threshold achieves positive `FIT` gain with `FIT` harm `<= 5%`, the gate
    stays closed (`tau = -inf`, `selected_lambda = "R882"`, falling back to `r882`).
  - **No outcome leakage:** No `A`, `B`, or `DL-Val` outcome is used to choose `lambda`, `tau`, or calibration.
- **Endpoint:** The flywheel protocol `fixed-dev-cost-argmin-negative-fitness/v1` (merged `H139` `native`): full-pool
  first argmin, utility `-fitness` when valid and `-2.0` when invalid. Verified on every scored context against `H124`'s
  first-argmin utility. **Exact ties are zero gain.**

## Arms

| Arm | Cost | Role |
|---|---|---|
| `R882` | `H124` baseline (frozen `round-882.pt`, `0` updates) | reference (`Elo 1500`); `DL-Val` native `91/91` bit-exact (`1.1377941898164525`) first |
| **`GATED_NEST`** | `blend(r882, GBM_cal, lam_cell)` if `zgap <= tau_cell`, else `r882`, with `(lam_cell, tau_cell)` nested on `FIT` only over `lam in {0.125, 0.25, 0.5}` | **primary and co-primary** |
| `GATED_NEST_SHUF` | same nested `(lam, tau)` rule with `zgap` permuted across contexts (`SHUF_SEED = 218160`) on `FIT` and at evaluation | placebo, report-only |
| `GATED_025` | fixed `lam = 0.25` with `tau_cell` chosen on `FIT` only (`H210`/`H217` recipe) | fixed-`lambda` reference, report-only |

`lambda_selection_counts` records how many of the `15` `(seed, outer)` cells selected `0.125`, `0.25`, `0.5`, or closed
`R882` fallback for each gated arm.

## Fresh partitions

- **Seeds and folds:** Seeds **`218101–218103`**, with fold seeds **`218141–218143`** (one distinct 5-fold family
  partition per seed). Sample seed `218140`, bootstrap seed `218150`, Elo seed `218151`, plateau bootstrap seed
  `218152`, `zgap`-permutation seed `218160`. None appears anywhere on `origin/main` (`git grep -w`).
- **Partitions:** Built with `H124` `folds` on the `205` `DL-Train` families (`1,219` contexts), cyclic cells `A = k`,
  `B = (k+1) % 5`, `FIT = the other three`. Pinned in [fold_manifest.json](fold_manifest.json).
- **Overlap (refuses reuse against `H203`, `H210`, and `H217`):**
  - Within `H218`, pairwise mean best-match Jaccard is `0.145–0.152`.
  - Against `H203` partitions, max best-match Jaccard is `0.176`.
  - Against `H210` partitions, max best-match Jaccard is `0.185`.
  - Against `H217` partitions, max best-match Jaccard is `0.178`.
  - Every `H218` partition is distinct from every other seed and from `H203`/`H210`/`H217`.
- **Shared across seeds (caveat disclosed upfront):** As in `H210` and `H217`, the `192`-program training subsample per
  context (`SAMPLE_SEED = 218140`) and the deterministic `r882` costs are shared across the three seeds, while the 5-fold
  family partitions and HistGBM random states vary by seed.

## Fits, smoke, lock, and scoring

- **Fits:** Per seed, `5` outer HistGBM fits (exclude `{A, B}`) and `10` inner HistGBM fits (exclude `{A, B, j}`):
  **`45` fits total** (because `blend(r882, GBM_cal, lam)` interpolates calibrated predictions in cost space without
  re-fitting HistGBM per `lambda`).
- **Step 2 — 1-cell non-scored smoke (`--smoke`):** After this prereg PR is squash-merged to `main`, `monitor.py --smoke`
  executes `1` cell (`seed = 218101, outer = 0`): verifies `DL-Val` `91/91` `r882` bit-exact (`1.1377941898164525`),
  `TRAIN` `r882` forward + `DOPE` checksum, fits the `4` HistGBM models for cell `(218101, 0)` (`(0,1)`, `(0,1,2)`,
  `(0,1,3)`, `(0,1,4)`), runs Stage-1 `FIT`-only `(lambda, tau)` selection for cell `(218101, 0)`, and writes
  `smoke_cell_0.json` with **`0` `A` or `DL-Val` arm outcomes scored**.
- **Step 3 — Full run (`--after-smoke`):** Verifies and reuses the `4` hash-checked fits from the 1-cell smoke, fits the
  remaining `41` HistGBM models, exclusively writes and `fsync`s `stage1_lock.json` **before any `A` or `DL-Val` arm
  scoring**, and then scores pooled `A` (`1,219` contexts / `205` families) and `DL-Val` (`91` contexts / `19` families).

## Statistics and decision

**Primary (`GATED_NEST` vs `R882`, pooled-`A` native, `205` families, Holm over primary only `m = 1`):**
- Statistic: median over the `3` seeds of the per-seed equal-family native utility gain.
- `CI95`: `10,000`-draw paired family-cluster bootstrap (`BOOT_SEED = 218150`), shared family draws across seeds,
  reduced by median over seeds on axis `0`.
- Harm: fraction of families (`205`) whose 3-seed mean delta is `< -0.01`.
- **`CANDIDATE: GATED_NEST`** iff all three hold:
  1. **`LCB95 > 0`**;
  2. **`3/3` seeds positive**;
  3. **`harm <= 10%`**.
  Otherwise **`NO_ADVANCE`**.

**Co-primary: `DL-Val` `flywheel91` plateau (`GATED_NEST` vs `R882` on `DL-Val`, `91` contexts, `19` families):**
- Statistic: `91`-context mean native utility gain over `r882` (`1.1377941898164525`), where each context's arm utility
  is averaged across the `15` `(seed, outer)` cells.
- `CI95`: `10,000`-draw family-cluster bootstrap (`PLATEAU_SEED = 218152`, resampling the `19` families with replacement
  and pooling drawn contexts).
- **`PLATEAU_WIN`** iff **`gain > +0.002` AND `LCB95 > 0`**. Exact ties or gains `<= +0.002` are **`NO_PLATEAU_WIN`**.

**Combined verdict:** `<primary> + <co-primary>` (e.g., `CANDIDATE: GATED_NEST + PLATEAU_WIN` or
`NO_ADVANCE + NO_PLATEAU_WIN`). Reported alongside: Bradley–Terry Elo (`r882 = 1500`, `1,000` family draws,
`seed = 218151`), `U4` report-only contrasts/Elo, `-2.0` pick counts, gate open/switching rates, and per-cell
`(lambda, tau)` choices and `lambda_selection_counts`. No AUC and no speed metrics.

## Read boundary, host guards, and custody

- **Reads:** Real-S3-derived `DL-Train` packet (`1,219` contexts / `205` families; `selected.json` `981960ad…`) and `91`
  `DL-Val` contexts (`19` families). Staging `~/jope-data/h218/inputs -> ~/jope-data/h195/inputs`.
- **Never read:** Sealed, holdout, or meta-test rows; live flywheel loop; GPU (`CUDA_VISIBLE_DEVICES=""`, no
  `~/qualification.lock`).
- **Host & runtime guards (`xbabe2` CPU only, `nice 19`):**
  - Worker cap: `OMP_NUM_THREADS=4`, all other thread pools `1`, torch `1` thread; worker tree `<= 4.5` cores over a
    `30 s` window.
  - Admission: `load1 <= 80`, `load5 <= 80`, `host CPU <= 80%`, `MemAvailable >= 24 GiB`, `disk free >= 220 GB`.
  - Runtime stops: `load5 > 85`, `load1 > 85` or `host CPU > 85%` continuously for `300 s`, `MemAvailable < 20 GiB`
    (swap is full), `xbabe2 free disk < 200 GB`, `aggregate RSS >= 6 GB`, own root `>= 500 MB`, or absolute deadline
    (`claim time + 4 h`, finish before `14:30Z`).
- **Custody:** Bulk artifacts go create-only to
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H218/<run_id>/`; git commits only small JSON/MD
  summaries (`README.md`, `PREREG.md`, `manifest.json`, `fold_manifest.json`, `metrics.json` `< 100 KB`, `custody.json`,
  and the index row).
