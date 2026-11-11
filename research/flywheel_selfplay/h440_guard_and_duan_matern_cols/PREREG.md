# H440 Pre-Registration: `ya`-Free Extrapolation Guard + Duan-Smeared Matérn/Entity/DAG Columns (`W1–W4` + Cross-Lane `H430`/`H433`)

- **Wave ID:** `H440` (`research/flywheel_selfplay/h440_guard_and_duan_matern_cols/`)
- **Agent / Lane:** `agy_apex_w10` (Gemini, `xbabe2` CPU-only)
- **Panels:** Stage A design check on `FIT-SEL` (`106` contexts, `TRAIN`); Stage B confirmation on `TRAIN P201b` (`201` families, plus pre-declared `191`-context `excl10` unprobed subset excluding `PROBED_P201B.txt`).
- **Holdout & `DEV` Discipline:** **Zero `APEX-DEV` reads.** `APEX-CONFIRM`, `dev91-audit-v1`, `DL-Val`, `DL-Holdout`, and `sealed` are 100% untouched.
- **Purity:** 100% ours-only (`NumPy`, `SciPy`, `scikit-learn`, `CatBoostRegressor(thread_count=1)`). Foundation models (`Kumo L`, `TabPFN 3.5`, `TabFM`) are never loaded, queried, or distilled.

---

## 1. Motivation & Hypotheses

1. **Extrapolation Catastrophes & Near-Exact Precision Dilution (`W1`):** Stored GP/library candidates (`51e71f0b3089` in `P201b`) and unguarded external columns (`d8134cd42a` in `H430`) occasionally achieve good SEARCH OOF scores while extrapolating wildly (`10x–1000x` `sd(y)`) on `Xa` due to denominator singularities or ill-conditioned ridge fallbacks, while on near-exact tables (`7d3e30f33f` in `H436`, stored `u = 17.05`), finite-step Caruana / column-subsampling can assign `0.01–0.03` weight to a `u = 6.7` column and dilute stack utility by `-6.9` nats. Our 100% `ya`-free **Peer-Consensus Extrapolation + Precision-Dilution Guard**:
   - Compares every candidate's `BAG`/`REF` audit predictions against the median of the top-7 bounded OOF peers (`[min(ys) - 4*span, max(ys) + 4*span]`), demoting unstable refits or dropping exploding candidates without ever reading `ya`;
   - Enforces a multiplicative precision-tier pool filter (`s_oof <= 3.0 * max(min_oof, 1e-12 * sd_y)`) whenever `min_oof < 0.03 * sd_y` (`u_oof > 3.5`, preventing near-exact precision dilution across both internal and external columns) plus `H434`'s `(overlap_rho, r_rat)` diversity bound on noisy tables (`min_oof >= 0.03 * sd_y`);
   - Combines surviving columns via Multi-Loss BCAR (`0.50*fold_log + 0.35*huber + 0.15*br_mse` with adaptive `eps = min(1e-12*var_y, 1e-4*min_col_mse)`) and a 4/5-fold OOF-ratcheted `eigh` Level-2 residual gate.
2. **Duan-Smeared Target Warping, Entity Hierarchy, Smooth Matérn-RQ & Residual DAGs (`W2–W4`):**
   - `rel_cb_yjduan`: Yeo-Johnson + 15-quantile Duan smearing CatBoost over pre-scale missingness indicators (`NaN` + `0.0` sentinel) and James-Stein shrunk entity-key / foreign-key pair relational aggregations (`EntityHierarchyEngine`).
   - `matern_rq_duan`: Exact Spectral-LOO (`eigh`) Active-Subspace Matérn-3/2 + Matérn-5/2 + Rational-Quadratic GP choosing `(identity, yeo_johnson)` Duan warp and noise `(0.015, 0.06, 0.22)` strictly by analytical LOO MSE in raw `y` units.
   - `subgp_rgduan`: Sharp-ARD MultiView `[Xs, 0.55*Xb]` Subspace GP in Rank-Gauss + linear-tail + 15-quantile Duan smearing space.
   - `spline_ratio_yjduan`: Exact SVD-LOO Ridge over 7-knot quantile hinges + top-10 bilinear products + log-ratios in Yeo-Johnson Duan space.
   - `dag_gp_on_cb`: OOF-gated Active-Matérn-RQ GP trained on out-of-bag residuals of relational CatBoost.
   - `dag_hgb_on_spline`: OOF-gated bagged `HistGradientBoostingRegressor` trained on exact SVD-LOO spline-ridge residuals.
   - `regime_route_gp`: Soft-sigmoid regime-split tree with local Active-Matérn-RQ GP leaf experts blended 50/50 with global GP.
   - **Plain Controls (`CONTROL_NAMES`):** `cb_plain` and `gp_plain` (median/IQR features only, `identity` target space, no Duan smearing, no `EntityHierarchyEngine`, no residual DAG).

### Pre-Registered Primary Hypotheses (on `TRAIN P201b`, 201 families; Holm-Bonferroni FWER = 0.025)
- **Primary 1 (`G_ONLY` vs `BASE`):** Applying the 100% `ya`-free Peer-Consensus Extrapolation Guard + Multi-Loss BCAR + `eigh` `L2` router to the **stored `h416` candidates alone** improves family-mean utility over `h416.SXB` (`BASE`).
- **Primary 2 (`GM_FULL` vs `BASE`):** Admitting the 7 primary `W2–W4` columns (`PRIMARY_NAMES`) into the guarded 2L combiner improves family-mean utility over `h416.SXB` (`BASE`), with `M_BCAR` (`CA.bcar` on stored + 7 primary columns) beating `C_PLAIN` (`CA.bcar` on stored + 2 plain controls) in clipped family-mean utility (`beats_plain_control == true`).
- **Holm-Bonferroni Step-Down (`alpha = 0.025`):** Sort `(p_G_ONLY, p_GM_FULL)` ascending; the smaller one-sided family sign-flip `p`-value must satisfy `p_(1) < 0.0125` (with `T16.pair_stats["pass"] == True`), and the second must satisfy `p_(2) < 0.025` (with `pass == True`).

### Pre-Declared Secondary & Cross-Lane Readouts
- **Unprobed `P201b` (`excl10`, `191` families):** Same readout excluding the 10 contexts in `PROBED_P201B.txt`.
- **Cross-Lane `H430` Exact-Key Admission (`G_H430`, `M_H430_BCAR`, `GM_H430`):** Read-only admission of `/home/ubuntu/jope-data/h430/pub/cols_{FITSEL,P201B}` (`key_direct`, `key_gbm`, `factor_glm`), testing whether fit-range clipping + `ya`-free peer guard eliminates `H430`'s `d8134cd42a` blowup on `P201b` (`201` and `191` `excl10`) and whether `W2–W4` + `H430` (`GM_H430`) stacks additively over `G_H430` and `GM_FULL`.
- **Cross-Lane `H430 + H433` (`G_CW12`, `GM_CW12`) & `H436` (`GM_CW12H`):** Read-only cross-admission of `/home/ubuntu/jope-data/h433/pub/cols_{FITSEL,P201B}` (`form_gbm`, `form_gam`, `form_sr`, `form_interp`) and `/home/ubuntu/jope-data/h436/pub/cols_{FITSEL,P201B}` (`key_hybrid`, `ridge_z`), including `@cl_fwriter`'s `cid 7d3e30f33f` near-exact precision-dilution check.

---

## 2. Pinned Inputs & Compute Budget

- **Base Stage:** `/home/ubuntu/apex-stage/h424/pub-r1` (`pub` stage, verified against `CODE.sha256`).
- **Banks & Stored Runs:**
  - `FIT-SEL` bank: `/home/ubuntu/jope-data/h335-cl/h424/dev/bank_FITSEL.npz` (`sha256: f8849ccb7343cfec1c0fd6688aaafef7580b85001046f2707fab58c5ecebe069`)
  - `P201b` bank: `/home/ubuntu/jope-data/h430/bank_P201B.npz` (`sha256: da29e8ec64f623e0fa2ab29b31bbd40b91bef36adf8ddc8e2e773f51c96d01af`)
  - `FIT-SEL` stored run: `/home/ubuntu/jope_plan/evidence/H420/pulled/H420_FIT-SEL/H420_FIT-SEL` (`tag: TBOTH`, `106` contexts)
  - `P201b` stored run: `/home/ubuntu/jope_plan/evidence/H416/pulled/H416_CONFIRM/H416_CONFIRM` (`tag: P201B`, `201` contexts)
- **Compute Ceiling:** `<= 16` single-thread `nice 19` CPU processes (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""`), per-context timeout `900s`, wall-clock cap `14,400s` (`4 hours`).

---

## 3. Execution Stages & Stop Rule

1. **Unit & Synthetic Smoke Check:** `test_guard_matern.py` (`6/6` invariants: `ya`-free `bag`/`ref` bomb rejection, `W2–W4` vs plain control on skewed relational DGP, strict `Xa` row-permutation invariance, near-exact precision-dilution prevention + refit-improved key admission) + `smoke_matern.py` (2-context synthetic bank end-to-end check).
2. **Stage A (`FIT-SEL`, `106` contexts):** Build `cols_FITSEL`, run `APPLY_FITSEL.json` and `READOUT_FITSEL.json`. **Stop Rule:** Halt before `P201b` if `READOUT_FITSEL.json` reports `harm == true` OR more than `5%` (`> 5`) of `FIT-SEL` contexts are not `ok`.
3. **Stage B (`P201b`, `201` contexts):** Build `cols_P201B` and run `APPLY_P201B.json`. Driver stops **before** `READOUT_P201B.json` so `gates_h440.sh` executes first.

---

## 4. Integrity Gates (`gates_h440.sh`, Run Before Stage B Readout)

1. `g1_prereg_before_output`: PREREG commit is an ancestor of `origin/v1/next`, staged `PREREG_H440.md` matches byte-for-byte, earliest output timestamp strictly exceeds PREREG commit timestamp, and `READOUT_P201B.json` does not yet exist.
2. `g2_pinned_banks`: `APPLY_FITSEL.json` and `APPLY_P201B.json` record exact `bank_sha256` matches and `aligned_all == true`.
3. `g3_replay_exact`: Stored `h416.SXB` replay satisfies `replay_max_w_diff < 1e-12` and `replay_max_u_diff < 1e-12` on both panels.
4. `g4_cpu_budget`: Wall time `<= 14400s` and shard count `<= 16`.
5. `g5_columns_complete`: At least `95%` of `FIT-SEL` (`>= 101/106`) and `P201b` (`>= 191/201`) contexts completed `column_ok == true`.
6. `g6_unit_and_smoke`: `unit_tests.json` reports `all_passed == true`.
7. `g7_confirm_untouched`: `/home/ubuntu/jope-data/h335-cl/apex/tables` count equals `661`, and zero `APEX-DEV` files are opened.
8. `g8_pub_stage`: Stage mode is `pub` and `sha256sum -c CODE.sha256` passes.

---

## 5. Amendment Policy

Code changes after `PREREG.md` is published on `origin/v1/next` are restricted strictly to fixing an unhandled crash or non-finite output before `READOUT_P201B.json` is generated, documented in a numbered `AMENDMENT_k.md` committed on `origin/v1/next` before staging a new `pub-r<k+1>` directory.
