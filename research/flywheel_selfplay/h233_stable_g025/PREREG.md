# H233 — `STABLE_G025` (pick-stability unanimity vote gate inside the frozen H217 `GATED_025` recipe), `xbabe2` CPU, 0 updates

**Owner:** `rd_argon` `[ARGON]`. **Assignment:** `~/nudges/hw_t2205_argon_h233.md` (`watcher`, 2026-11-02 22:05Z).
**Claim:** `~/jope_plan/ns/receipts/H233_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/claude-klguard`.

This study is a zero-update offline evaluation on read-only copies of `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`). It cannot advance a model, a serving release, or a live flywheel checkpoint. **This preregistration merges to `main` before any scientific run executes.**

## Context & Hypothesis

- **Context:** [H228](../h228_family_invariant/README.md) (`#2062`) established that gradient continuation on the static 1,219-context `DL-Train` pool is closed. The only confirmed positive lift over `r882` is the zero-update `GATED_025` recipe ([H217](../h217_gated025_confirm/README.md): `+0.0216` native on `DL-Train` OOF, `NO_PLATEAU_WIN` `+0.00145` on `DL-Val91`). [H223](../h223_bagged_blend_gate/README.md) (mean-of-3 bagged blend costs) and [H215](../h215_pess_disagreement/README.md) (disagreement cost penalty) were `NO_ADVANCE`. [H231](../h231_coswr_gated025_retest/PREREG.md) (`rd_astra`) is testing `COSWR x GATED_025` and is not duplicated here.
- **Hypothesis (`STABLE_G025`, pick-stability vote gate):** Apply the frozen H217 `GATED_025` blend to a context **only** when:
  1. **(a)** the frozen H217 `zgap` gate passes (`zgap(r882, blend_m0) <= tau_cell`), **AND**
  2. **(b)** the blended first-argmin pick is **unanimous across 3 outer HistGBM members** (`argmin(blend_m0) == argmin(blend_m1) == argmin(blend_m2)`), where member `0` is the unweighted H217 recipe fit and members `1–2` are `FIT`-family bootstrap refits with the same recipe.
  Otherwise fall back to the `r882` pick (`r882` cost vector).
- **Rationale:** Residual harm in `GATED_025` comes from unstable blend flips near the decision boundary. A unanimity vote across 3 `FIT`-family bootstrap members acts as a discrete pick-level stability filter rather than averaging continuous cost predictions (H223) or penalizing candidate costs (H215).

## Arms (0 optimizer updates; `DOPE c51a56d0` and `r882` read-only)

| Arm | Definition | Role |
|---|---|---|
| `R882` | Pinned H124 baseline (`round-882.pt`), 0 updates | Reference (`Elo = 1500`); `DL-Val91` native `91/91` (`1.1377941898164525`) verified bit-exact first |
| `G025` | Member-0 blend `blend_m0` if `zgap <= tau_cell`, else `r882` (frozen H217 `GATED_025` recipe) | Replication anchor (`STABLE_G025` must beat `G025`) |
| **`STABLE_G025`** | Member-0 blend `blend_m0` if `(zgap <= tau_cell)` **AND** `(pick_m0 == pick_m1 == pick_m2)`, else `r882` | **Sole primary (`Holm m = 1`)** |
| `STABLE_SHUF` | Member-0 blend `blend_m0` if `(zgap <= tau_cell)` **AND** `shuf_agree`, else `r882`, where `shuf_agree` permutes the 3-member boolean unanimity mask across contexts within each `FIT`-calibrated cell (`seed, outer_k`), preserving the exact vote count and vote rate | Placebo (`STABLE_G025` must beat `STABLE_SHUF`) |
| `BAG_VOTE_UNGATED` | Member-0 blend `blend_m0` if `(pick_m0 == pick_m1 == pick_m2)` (vote without `zgap`), else `r882` | Report-only |

## Frozen H217 Recipe & 75-Fit Budget

- **Blend:** `blend_m = r882 + 0.25 * (prior_cal_m - r882)` in cost space (`lambda = 0.25` fixed).
- **Calibration (`FIT`-only):** For each cell `(seed, outer_k)` (`A = k`, `B = (k+1)%5`, `FIT = the other 3 folds`), positive-slope linear calibration `(alpha, beta) = h124.calibration(...)` is fit on member `0`'s out-of-fold inner predictions over the 3 `FIT` folds (`exclude {A, B, j}` for `j in FIT`). That cell's `(alpha, beta)` calibrates all 3 outer members (`prior_cal_m[c] = alpha - beta * outer_pred_m[c]`).
- **`zgap` & `tau_cell` (`FIT`-only, frozen from H210/H217):** `zgap = (r882[b] - r882[a]) / std(r882)` where `a = argmin(r882)` and `b = argmin(blend_m0)` (`0` when picks agree). `tau_cell` is selected on `FIT` folds only from `[-inf, q10, q25, q50, q75, q90, +inf]` (quantiles over switching `FIT` contexts) to maximize `FIT` equal-family native gain subject to `FIT` family harm (`< -0.01`) `<= 5%`, with ties broken toward the smaller (more conservative) `tau`.
- **Fit budget (`25` HistGBM fits/seed, `75` total):**
  - **5 outer exclusion units per seed (`exclude {A, B}`):** Each builds the design matrix once (`192` subsampled candidates/context, `SAMPLE_SEED = 233140`) and fits **3 HistGBM members** (`m = 0, 1, 2`):
    - Member `0`: unweighted (`sample_weight = None`), tested bit-for-bit identical to H217's `fit_gbm`.
    - Members `1, 2`: `FIT`-family cluster bootstrap refits (`sample_weight = member_weights(fams, seed, excluded, m)` with `BAG_SEED = 233170`, drawing `|FIT families|` with replacement and weighting every row of a family by its multiplicity).
  - **10 inner exclusion units per seed (`exclude {A, B, j}` for `j in FIT`):** Each fits **member `0` only** (`1` HistGBM fit/unit) to supply the `FIT` out-of-fold predictions for `h124.calibration` and `choose_tau`.
  - **Total:** `3 seeds x (5 outer x 3 members + 10 inner x 1 member) = 3 x 25 = 75` HistGBM fits.
- **Stage-1 lock (`stage1_lock.json`):** Written exclusively and fsynced **before any `A` or `DL-Val91` outcome scoring**, recording per cell: `(alpha, beta)`, `tau_cell` and the `FIT` grid, the `A`-fold unanimous vote counts and permuted `STABLE_SHUF` counts, the `B`-panel hash, and all `25` prediction `.npz` SHA-256 hashes per seed.

## Fresh Seeds & Family Partitions (`Jaccard <= 0.20` vs `H203–H231`)

- **Seeds:** Outer seeds **`233101, 233102, 233103`**; fold seeds **`233141, 233142, 233143`**; subsample seed **`233140`**; bootstrap seed **`233150`** (`Elo` seed `233151`, plateau seed `233152`); agreement-mask permutation seed **`233160`**; family-bootstrap member weight seed **`233170`**.
- **Partitions:** Pinned in [fold_manifest.json](fold_manifest.json) over the 205 `DL-Train` families (`1,219` contexts).
  - Within `H233`, pairwise mean best-match Jaccard is `0.1519–0.1715`.
  - Across all 24 registered partition waves in `H203–H231`, the maximum mean best-match Jaccard is **`0.1993 <= 0.20`** (full per-wave table pinned in [fold_manifest.json](fold_manifest.json)).

## Endpoints & Preregistered Admission Criteria

- **Primary endpoint:** Flywheel native utility `fixed-dev-cost-argmin-negative-fitness/v1` (first argmin over the full candidate pool; utility is `-fitness` when gate-valid and `-2.0` when gate-invalid) on the 205-family / 1,219-context pooled-`A` cross-fit panel.
- **Admission criteria (`STABLE_G025`, sole primary, `Holm m = 1`, ties = `NO_ADVANCE`):**
  1. Median equal-family gain over 3 seeds vs `R882` **`> +0.002`**.
  2. 10,000-draw paired family-cluster bootstrap 95% CI lower bound **`> 0`**.
  3. **`3/3` positive seeds** vs `R882`.
  4. Family harm fraction (205 families with 3-seed mean gain `< -0.01`) **`<= 10%`**.
  5. Must **strictly beat `STABLE_SHUF`** (paired median equal-family gain of `STABLE_G025 - STABLE_SHUF > 0`; ties = `NO_ADVANCE`).
  6. Must **strictly beat `G025`** (paired median equal-family gain of `STABLE_G025 - G025 > 0`; ties = `NO_ADVANCE`).
- **Secondary endpoint (`DL-Val91`, 19 families, 91 contexts):** Equal-family 3-seed median gain, 10k family-bootstrap CI95, harm fraction, paired contrasts vs `G025` and `STABLE_SHUF`, Bradley–Terry Elo vs `R882 = 1500`, and the 15-cell plateau readout (`> +0.002` and `LCB95 > 0`).
- **Reporting:** Utility and Bradley–Terry Elo (`R882 = 1500`) only; never AUC, no speed verdicts. Assert `0` protected (`sealed`, `holdout`, `meta-test`) reads in `metrics.json`.

## Resource Guards & Schedule (`xbabe2` CPU only)

- **Host & priority:** `xbabe2` CPU only (`ALLOWED_HOSTS = ("xbabe2",)`), `nice 19`, `CUDA_VISIBLE_DEVICES=""` (no GPU, no `qualification.lock`).
- **Cores & threads:** `taskset <= 4` cores (`3` OpenMP threads) when `load5 >= 50.0`, scaling up to `8` cores (`7` OpenMP threads) when `load5 < 50.0`, with a hard 30 s window cap of `8.0` worker-tree cores.
- **Hard aborts:** `MemAvailable < 20 GiB` (swap is full on `xbabe2`), free disk `< 300 GB`, process-tree RSS `>= 4 GiB`, own scratch `>= 1 GB`, or reaching the fixed compute stop **`2026-11-03T05:00:00Z`**.
- **Load pause/resume:** `SIGSTOP` own worker process group when `load5 > 60` (or sustained `load1`/CPU `> 85%` for `300 s`), resuming (`SIGCONT`) when `load5 <= 55`; never kill or pause any other job.
- **Schedule:** Prereg PR merged by `2026-11-02T23:30:00Z`; `TRAIN` smoke + **one** primary attempt; compute stop **`2026-11-03T05:00:00Z`**; closeout PR merged by **`2026-11-03T05:30:00Z`** (`INCOMPLETE` with no retry if stopped early). Artifacts stored in S3 under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H233/`.
