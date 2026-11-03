# H246 Preregistration — Fresh-Seed `STABLE_G025` Pick-Stability Unanimity Vote Gate Rerun (`h246-stable-g025-rerun-20261103`)

- **Wave ID**: `h246-stable-g025-rerun-20261103` (`H246`)
- **Owner**: `rd_argon` (`[ARGON]`, screen `4187849.rd_argon`)
- **Authority**: Owner-delegated H-wave routine (`/home/ubuntu/nudges/hw_t0505_argon_h246.md`)
- **Study Path**: `research/flywheel_selfplay/h246_stable_g025_rerun`
- **Host / Resource Contract**: `xbabe2` CPU ONLY, `taskset -c 52-55` (`4` cores), `nice -n 19`, HistGBM/OpenMP threads `<= 4` (`OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`), process-tree RSS `< 4 GiB`, free disk `>= 300 GB`. Cooperative unit-boundary / member-boundary PAUSE guard (H239 A1):
  - **Load**: admit `load5 <= 100.0`, cooperative pause when `load5 > 108.0`, resume when `load5 <= 100.0`.
  - **Memory**: admit `MemAvailable >= 30 GiB`, cooperative PAUSE when `MemAvailable < 20 GiB`, resume when `MemAvailable >= 25 GiB` sustained continuously for `30 s`; hard stop ONLY if `MemAvailable < 12 GiB` sustained continuously for `30 s` or `MemAvailable < 6 GiB` instant.
  - **No GPU**, never `xbabe0` or `xbabe3`, never pause/kill other jobs, cores `40-51` untouched.
- **Deadlines**:
  - Preregistration PR merged to `main`: `<= 2026-11-03T06:15:00Z`
  - Primary compute hard stop: `<= 2026-11-03T10:00:00Z` (`epoch 1793700000.0`)
  - Closeout PR merged to `main`: `<= 2026-11-03T10:45:00Z`

---

## 1. Scientific Motivation & Hypothesis

`GATED_025` (`H210`, confirmed in `H217`) is the only confirmed positive post-`r882` line on the 205-family pooled cross-fit (`H217` pooled-`A` OOF equal-family native gain `+0.0216`, harm `4.4%`), but its `DL-Val91` gain (`+0.00145`) misses the `+0.002` plateau margin because a residual fraction of contexts where `zgap <= tau_cell` opens the gate still switch to an unstable HistGBM top-1 candidate due to finite-sample variance of a single HistGBM fit on `3/5` of the `205` families (`~123` FIT families).

Wave `H233` preregistered `STABLE_G025` (require both `zgap <= tau_cell` AND unanimity of the blended first-argmin pick across `3` outer HistGBM members: member `0` = unweighted `H217` fit, members `1–2` = FIT-family cluster-bootstrap refits), but `H233` was stopped `CLOSED INCOMPLETE` at `44/75` subfits (`2026-11-03T01:20:51Z`) when its hard `MemAvailable < 20 GiB` abort fired during a transient host-wide `xbabe2` memory dip (`19.86 GiB`) while `H233`'s own worker was already paused at `0.43 GiB` RSS, so the 3-seed hypothesis was never tested.

**H246 Hypothesis (`STABLE_G025` Fresh-Seed Rerun)**:
Evaluating the exact unchanged `H233` `STABLE_G025` recipe on three fresh seeds (`246101–246103`, fold seeds `246141–246143`) with sequential member fitting (holding at most one `HistGradientBoostingRegressor` model in memory at a time and releasing heap pages between members) and the cooperative H239 A1 unit/member-boundary memory/load pause guard will complete all `75/75` HistGBM subfits within `< 4 GiB` RSS and improve both `DL-Val91` (`flywheel91`) and `205`-family pooled-`A` OOF native utility over `r882` by `> +0.002` with `LCB95 > 0`, Holm `p < 0.05` (`m = 1`), `>= 2/3` positive seeds, and harm `<= 10%`, while outperforming both the unvoted `G025` anchor and a vote-rate-matched permuted-agreement placebo (`STABLE_SHUF`).

---

## 2. Experimental Design, Arms & Fit Plan

### 2.1 Seeds and Fold Partitions (`fold_manifest.json`)
- **Outer Seeds**: `SEEDS = (246101, 246102, 246103)`
- **Fold Seeds**: `FOLD_SEEDS = {246101: 246141, 246102: 246142, 246103: 246143}`
- **Subsample / Bootstrap / Permutation Seeds**: `SAMPLE_SEED = 246140`, `BOOT_SEED = 246150`, `SHUF_SEED = 246160`, `BAG_SEED = 246170`
- **Algorithm**: Exact `H124` 5-fold family partition on the `205` `DL-Train` families (`1,219` contexts; `has_meta_test` rows excluded):
  - Fold sizes per seed: `[41, 41, 41, 41, 41]` families (`1,219` contexts total).
  - For each outer cell `k in {0, 1, 2, 3, 4}`:
    - `A = fold[k]` (outer evaluation fold; `5` cells pool to all `205` families / `1,219` contexts per seed)
    - `B = fold[(k + 1) % 5]` (disjoint holdout fold)
    - `FIT = the remaining 3 folds` (`123` families; used exclusively for HistGBM fitting, inner OOF linear calibration, and FIT-only `tau_cell` selection)
- **Overlap Independence Check (`H203–H245`)**:
  - Pairwise best-match Jaccard across `246101–246103`: `0.1583`, `0.1651`, `0.1520`.
  - Maximum best-match Jaccard against all `38` earlier waves (`H203–H245`): **`0.192801 <= 0.20`** (recorded in [fold_manifest.json](fold_manifest.json)).

### 2.2 75-Fit Sequential Plan (`25` HistGBM Fits per Seed × `3` Seeds)
- **Zero neural network updates**: `0` optimizer steps; `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) are read-only and verified before and after execution.
- Per seed `s in {246101, 246102, 246103}`:
  1. **`10` Inner Calibration Fits (`exclude {A, B, j}`, `j in FIT`, `member = 0` only)**:
     - Unweighted `H217` `HistGradientBoostingRegressor(max_iter=30, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=s)` on the `2` remaining FIT folds (`82` families).
     - Used strictly to produce out-of-fold predictions on the `3` `FIT` folds of each cell `(s, k)` for `H124` linear cost calibration `(cal_a, cal_b)` and `H217` FIT-only threshold selection `tau_cell = choose_tau(zgap_fit, r882_fit, blend0_fit, fit_families, target=0.05)`.
  2. **`5` Outer Units × `3` Members = `15` Outer Fits (`exclude {A, B}`, `member in {0, 1, 2}`)**:
     - Fitted **sequentially** on the `3` `FIT` folds (`123` families) of cell `(s, k)`, writing each member's `.npz` and `.json` receipt and releasing the fitted estimator and temporary arrays (`gc.collect()` + `malloc_trim(0)`) before fitting the next member:
       - `member 0`: Unweighted (`sample_weight = None`) — bit-identical to the `H217` outer HistGBM fit.
       - `members 1, 2`: Family-cluster bootstrap refits (`stat.member_weights(row_families, s, excluded, m)` using `stream_seed(BAG_SEED, s, m, *excluded)`), drawing `123` FIT families with replacement and weighting each context's `192` sampled rows by its family multiplicity.
     - For each context `c`, each member `m in {0, 1, 2}` produces calibrated prior cost `prior_m(c) = cal_a - cal_b * pred_m(c)` and blended cost vector:
       $$b_m(c) = c_{\text{r882}}(c) + 0.25 \cdot (\text{prior}_m(c) - c_{\text{r882}}(c))$$
- **Total Fits**: `(10 * 1 + 5 * 3) * 3 = 25 * 3 = 75` HistGBM subfits.

### 2.3 Registered Arms (`5` Arms; `1` Sole Primary `m = 1`)
For every context `c` in outer cell `(s, k)` with FIT-selected threshold `tau_cell`:
- Let `b_0(c)` be member `0`'s blended cost vector (identical to `H217`'s `BLEND_025`).
- Let `zgap(c) = (r882(c)[argmin(b_0(c))] - r882(c)[argmin(r882(c))]) / std(r882(c))` (`0.0` when `argmin(b_0(c)) == argmin(r882(c))`).
- Let `unanimous(c) = I(argmin(b_0(c)) == argmin(b_1(c)) == argmin(b_2(c)))` (outcome-free 3-member top-1 pick unanimity).
- Let `shuf_unanimous(c)` be a deterministic outcome-free permutation of the boolean vector `{unanimous(c)}` across contexts within cell `(s, k)` (`stream_seed(SHUF_SEED, s, k, stage_tag)`), preserving the exact cell unanimity count and rate.

1. **`R882` (Incumbent Baseline)**: Uses `c_r882(c)`.
2. **`G025` (Replication Anchor — Frozen `H217` `GATED_025`)**: Uses `b_0(c)` iff `zgap(c) <= tau_cell`, else `c_r882(c)`.
3. **`STABLE_G025` (Sole Primary Treatment, Holm `m = 1`)**: Uses `b_0(c)` iff `zgap(c) <= tau_cell` **AND** `unanimous(c) == True`, else `c_r882(c)`.
4. **`STABLE_SHUF` (Vote-Rate-Matched Permutation Placebo)**: Uses `b_0(c)` iff `zgap(c) <= tau_cell` **AND** `shuf_unanimous(c) == True`, else `c_r882(c)`.
5. **`BAG_VOTE_UNGATED` (Report-Only Ablation)**: Uses `b_0(c)` iff `unanimous(c) == True` (no `zgap` gate), else `c_r882(c)`.

---

## 3. Stage-1 Lock, Endpoints & Admission Rules

### 3.1 Mandatory Pre-Scoring Gates
1. **Exact `R882` `DL-Val91` Regression Gate (`Native91`)**: Before any `TRAIN` or `DL-Val` HistGBM scoring, `r882` must reproduce all `91/91` `DL-Val` per-context utilities bit-for-bit (`mean = 1.1377941898164525`). Any discrepancy aborts immediately.
2. **Stage-1 `fsync` Lock (`stage1_lock.json`)**: All `15` cells' inner calibrations `cal`, FIT-only thresholds `tau_cell`, outcome-free unanimity counts/rates, `STABLE_SHUF` masks, `B`-fold `BLEND_025` SHA-256 digests, and all `75` prediction `.npz` SHA-256 digests must be written and `os.fsync`-locked to `stage1_lock.json` **before** any `A`-fold or `DL-Val91` treatment outcome is computed.

### 3.2 Primary & Secondary Endpoints and Admission Gate (`m = 1`)
- **Metric**: Flywheel native first-argmin utility (`higher = better`; `-fitness` if `gate == 1`, `-2.0` if `gate == 0`). Report median and `10,000`-draw family-bootstrap `CI95` across the `3` seeds, `1,000`-draw family-bootstrap Bradley–Terry Elo vs `R882` (`= 1500.0`) and Elo lift (`Elo - 1500.0`), contrast vs plateau best `1.1389572403748425`, and family harm fraction (`mean delta < -0.01`). No AUC; no speed verdicts.
- **Admission Verdict (`CANDIDATE: STABLE_G025` vs `NO_ADVANCE`)**:
  `STABLE_G025` (sole primary, Holm `m = 1`) advances **iff** on `DL-Val91` (`19` families, `91` contexts):
  1. Median equal-family native utility gain over `3` seeds vs `R882` **`> +0.002`**
  2. `10,000`-draw family-bootstrap `95%` lower confidence bound **`LCB95 > 0.0`**
  3. Holm `m = 1` adjusted one-sided `p < 0.05`
  4. **`>= 2/3` seeds positive** (`per_seed_gain > 0`) vs `R882`
  5. Family harm fraction (`3`-seed family mean delta `< -0.01`) **`<= 10%`**
  AND on the `205`-family (`1,219`-context) pooled-`A` OOF contrast:
  6. Median equal-family native gain vs `R882 > +0.002`, `LCB95 > 0.0`, `>= 2/3` positive seeds, harm `<= 10%`, and strictly beats both `STABLE_SHUF` (`median_gain(STABLE_G025 - STABLE_SHUF) > 0`) and `G025` (`median_gain(STABLE_G025 - G025) > 0`).
  - **Ties on any gate resolve to `NO_ADVANCE`.**
