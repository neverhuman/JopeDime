# H266 (`h266_rerank4_confirm`) — Preregistration

- **Wave ID**: `h266-rerank4-confirm-20261103`
- **Study Path**: `research/flywheel_selfplay/h266_rerank4_confirm`
- **Owner**: Argon (`xbabe2` CPU lane `taskset -c 57-60`, `nice -n 19`, `threads <= 4`, `RSS <= 4 GiB`, `0` GPU)
- **Status**: `PREREGISTERED` (`0` outcomes read for `H266`)
- **Predecessor**: `H265` (`da5a08afb9e7df397a24a65d249cfa07189cf11a`)

---

## 1. Motivation & Confirmatory Question

In `H265` (`h265_top4_listwise_reranker`), a zero-update (`0` optimizer steps on `r882` or DOPE) `FIT`-only listwise LambdaMART re-ranker (`RERANK4`) achieved the first zero-reveal single-pick `ADVANCE` over `r882` on `DL-Val91` (`+0.018685` equal-family utility, `Holm 97.5% CI [+0.005100, +0.033073]`, `Holm p = 0.000200`, `3/3` seeds positive, `0/455` invalid `-2.0` gates, beating `RERANK4_SHUF`). However, red-team review noted four reasons requiring confirmatory replication before adoption:
1. `DL-Val91` has only `19` families and has been reused across ~`265` waves (`Holm m = 2` does not adjust for cross-wave selection).
2. On the $10\times$ larger `DL-Train` OOF (`1,219` contexts / `205` families) under single fold seed `265140`, strict `RERANK4 - CONTROL` (`+0.023578` fam, `3/3` seeds) had `95% CI [-0.005946, +0.061002]`, `raw p = 0.0623` (`4/15` cells fell back to `m* = inf`), even though unconstrained `RERANK4_UNC - CONTROL` reached `+0.039468` fam (`95% CI [+0.004005, +0.080631]`, `raw p = 0.0143`).
3. On `DL-Val91`, seed `265103` (`+0.009591` fam, `3/5` folds admitted) had roughly half the gain of seeds `265101` (`+0.018685`) and `265102` (`+0.018698`).
4. `3–5` `DL-Val` families (`43–55` `DL-Train` families) experienced harm (`Δ < -1e-4`) per seed.

**H266 Question**: With the **exact `H265` feature builder, `LGBMRanker` hyperparameter grid, strict `H257` inner-`FIT` selector + margin rule, and relevance labels frozen (`0` re-tuning)**, does `RERANK4` replicate under **fresh seeds** on:
- **PRIMARY (`205` families, `1,219` contexts)**: `DL-Train` repeated nested family-grouped cross-validation across **`3` fresh fold seeds (`266140, 266141, 266142`) $\times$ `5` folds $\times$ `3` fresh model seeds (`266101, 266102, 266103`)** (`45` cells total), evaluated by `10,000`-draw paired family bootstrap (`boot_seed = 266150`) on the `205` families using per-seed medians across the `3` fold seeds?
- **SECONDARY (`19` families, `91` contexts)**: `DL-Val91` fresh-seed replication (`model_seeds = 266101, 266102, 266103`, `fold_seed = 266140`, plus `3`-fold-seed aggregate)?
- **Placebo (`RERANK4_SHUF`)**: Within-family label shuffle (`Holm m = 2` over `{PRIMARY, SECONDARY}` and within-split `Holm m = 2` over `{RERANK4, RERANK4_SHUF}`)?

---

## 2. Frozen Design Specification (`0` Changes from `H265`)

1. **Candidate Shortlist**: Fixed `r882` `top-4` (`top4 = np.argsort(rcost, kind="mergesort")[:4]`).
2. **Feature Vector (`560` float32 outcome-free features per candidate)**:
   - `cols 0..19` (`block1`, `20` features): `r882` score, pool z-score, top-4 z-score, rank slot `j`, `j/3`, `is_slot_0`, raw diff to slot-0, exact `H217` `zgap`, context top-2 `zgap`, context top-4 spread, softmax probabilities (`tau=1.0`, `tau=0.2`, top-4 z), pool-standardized `base_prior_cost`, and pairwise score/z-score diffs to the other 3 candidates.
   - `cols 20..31` (`block2`, `12` features): within-top-4 and top-8 `sem24`/`proj16` cosine geometry, `cos_rev4_mean` (`sem64[:4]` = structural semantic signature of fixed pool rows `0..3`, zero outcomes), operator count and diff to slot-0, and estimator family match indicators.
   - `cols 32..443` (`prog_412`, `412` features): frozen candidate program descriptor vector.
   - `cols 444..467` (`sem24_z`, `24` features): pool-standardized 24D structural semantic signature `compute_sem24_numpy(prog)`.
   - `cols 468..483` (`proj16_z`, `16` features): pool-standardized 16D program projection from `pool_priors.npz`.
   - `cols 484..559` (`ctx_vec76`, `76` features): `lm48_z` (`48` cols) + leading `25` cols and `3` summary stats (`mean, std, rms`) of `dope629` from `context_features.npz`.
3. **Training Relevance Labels (`DL-Train` `FIT` folds only)**:
   - `compute_true_top4_relevance(fitness_4, gate_4)` maps true revealed cost order within `top-4` to `rel in {0, 1, 2, 3}` (`3` = best valid, `-2.0` gate failure = `0` worst).
4. **Model & Inner-FIT Selector (`run_cell_fit_and_select`)**:
   - `lightgbm.LGBMRanker(objective="lambdarank", metric="ndcg", lambdarank_truncation_level=4, max_depth=3, num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, deterministic=True, force_row_wise=True)`.
   - 4-fold inner cross-validation across the `4` `FIT` folds (`164` families) over `lr in {0.03, 0.05, 0.10}`, `n_estimators in {50, 100, 200}`, and fallback margin `m in {0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, inf}`.
   - Strict `H257` inner-FIT selector: admit `m* = m_unc` iff `inner_fit_gain > +0.001 + 1e-12` AND `2,000`-draw paired family bootstrap `inner_lcb95 > 0.0`, else fallback `m* = inf` (keep `r882` rank-1).
   - All `90` confirmatory cell selections (`45` `RERANK4` + `45` `RERANK4_SHUF`) and `75` ablation cell selections are `os.fsync`-locked to `fit_selections_lock.json` **before any `DL-Val` feature or pool read for model scoring**.

---

## 3. Descriptive Feature-Block Ablations (on `fold_seed = 266140`, `seeds = 266101–266103`)

Not included in the Holm family (`descriptive only`):
1. `SCORE_ONLY` (`cols 0..31`, `32` features): `r882` score + cosine/program geometry block only.
2. `SCORE_ONLY_20` (`cols 0..19`, `20` features): strictly scalar `r882` + `base_prior_cost` scores/ranks/gaps/diffs only (tests `"just recalibrated r882 scores"` with zero structural/cosine features).
3. `NO_SCOREGEOM` (`cols 32..559`, `528` features): drops `cols 0..31`.
4. `NO_SEM24` (`531` features): drops `sem24_z` (`cols 444..467`) and all `sem24` cosine columns (`cols 20, 21, 22, 26, 27`).
5. `NO_CTX76` (`cols 0..483`, `484` features): drops `ctx_vec76` (`lm48_z` + `dope629`).

---

## 4. Confirmatory Decision Rule (`ADVANCE = CONFIRMED` vs `NO_ADVANCE = NOT_CONFIRMED`)

`H266` declares **`ADVANCE` (`CONFIRMED`)** iff ALL of the following hold:
1. **Baseline regression**: `CONTROL` (`r882`) reproduces `Native91 = 1.1377941898164525` (`91/91` bit-exact) first.
2. **PRIMARY (`DL-Train` `205`-family OOF `RERANK4 - CONTROL`, per-seed medians across `3` fold seeds `266140/266141/266142`)**:
   - `median_family_gain > +0.002 + 1e-12`
   - `Holm LCB95 > 0.0` (`ci975_holm2[0] > 0.0` and `holm_p_primary_secondary_m2 < 0.05`)
   - `3/3` model seeds (`266101, 266102, 266103`) positive
   - `RERANK4` beats `RERANK4_SHUF` (`median_family_gain(RERANK4 - RERANK4_SHUF) > 0.0` and `raw_p_one_sided < 0.05`)
3. **SECONDARY (`DL-Val91` `19`-family `RERANK4 - CONTROL`, `fold_seed = 266140`, `model_seeds = 266101–266103`)**:
   - `median_family_gain > 0.0` with `3/3` model seeds positive.

Any failure of the above gates yields **`NO_ADVANCE` (`NOT_CONFIRMED`)**, reported honestly with full diagnostics.
