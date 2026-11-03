# H275 Preregistration: Soft Invalid-Risk Penalized Top-8 Re-Ranking (`h275_rerank8_soft_risk`)

- **Wave ID**: `h275-rerank8-soft-risk-20261103` (`H275`)
- **Study Path**: `research/flywheel_selfplay/h275_rerank8_soft_risk`
- **Owner**: `Argon`
- **Authority**: `/home/ubuntu/nudges/hw_t2030_argon_h275.md` (`DECIDED`)
- **Claim Receipt**: `/home/ubuntu/jope_plan/ns/receipts/H275_claim.txt`
- **Parent `origin/v1/next` SHA**: `29fdbd8be15e2d2aa4ddf7b33e96f637f11dbd85`
- **Outcomes Read Before Preregistration**: `0` (`outcomes_read = 0`)
- **Preregistered Fold Manifest (`579,610` bytes)**:
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H275/prereg/fold_manifest.json`
  - `VersionId = SuUQLfyvQvJ_u07AyESxvQIOTR6wYPkG`
  - `sha256 = 95dc827b887a77b55cc0438e15f9573fb91f0c4159a0870fb59c5c0c980cf92d`

---

## 1. Scientific Motivation & Hypothesis

In `H273` (`29fdbd8be15e`), clean-room `RERANK8_CR` (`K = 8`, `568` feature columns) replicated `H270` on fresh fold and model seeds (`DL-Train` OOF median family gain `+0.0571 [+0.0192, +0.1166]`, `Holm p = 0.0013`, MM Elo `1554.73`; `DL-Val91` `+0.0365 [+0.0151, +0.0568]`, MM Elo `1568.99`), while the hard invalid-gate veto (`RERANK8_CR_VETO`, `tau = 0.10`) slashed `DL-Train` families harmed `> 0.01` from median `33` to `9` and eliminated `100%` of `DL-Val91` invalid picks (`0` vs `2`), but forfeited roughly half of the raw median lift (`+0.0278` on `DL-Train`, `+0.0158` on `DL-Val91`) due to hard `tau = 0.10` masking and Tier-1 zero-new-invalid-switch filtering on inner `FIT`. Meanwhile, `H272` (`7b5358963dc6`) showed that `RERANK8`'s lift shrinks to `+0.0173` (n.s.) on the `41` least-similar `DL-Train` families (`leave-family-cluster-out`).

`H275` tests whether replacing hard `tau = 0.10` invalid-gate masking with a **per-query standardized soft invalid-risk penalty (`RERANK8_CR_SOFT`)** tuned jointly with `margin` strictly inside inner `FIT` subject to `fit_invalid <= fit_ctrl_invalid` recovers the utility lift lost by `RERANK8_CR_VETO` while preserving low tail harm and zero `DL-Val91` invalid picks.

---

## 2. Frozen Core & Exact `RERANK8_CR_SOFT` Specification

### 2.1 Frozen Core from `H273` (`29fdbd8be15e`)
`runner.py` reuses `H273`'s clean-room core byte-for-byte:
- `568`-column outcome-free feature builder (`build_topk_feature_matrix_for_context` at `K = 8`),
- Relevance label builder (`compute_true_topk_relevance`),
- Base `LGBMRanker` (`fit_lgbm_ranker_200`, `LR_GRID = (0.03, 0.05, 0.10)`, `N_EST_GRID = (50, 100, 200)`),
- Outcome-free `LGBMClassifier` invalid-gate predictor (`fit_veto_classifier`, `VETO_CLF_PARAMS`),
- Margin grid (`FINITE_MARGIN_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)`, `MARGIN_GRID = (*FINITE_MARGIN_GRID, inf)`),
- Strict `H257`/`H273` inner-`FIT` fallback (`inner_fit_family_gain > +0.001`, `inner_lcb95 > 0.0`, `fit_invalid <= fit_ctrl_invalid`).

### 2.2 Per-Query Standardized Soft Invalid-Risk Score (`RERANK8_CR_SOFT`)
For each context (query) $i$ with raw `LGBMRanker` scores $s_{i,0}, \dots, s_{i,7}$ and out-of-fold / outer-`FIT` `LGBMClassifier` invalid-gate probabilities $p_{\text{invalid}}(i, 0), \dots, p_{\text{invalid}}(i, 7)$:
1. **Per-query standardization (`z(score_k)`)**:
   $$\mu_i = \frac{1}{8}\sum_{k=0}^{7} s_{i,k}, \qquad \sigma_i = \sqrt{\frac{1}{8}\sum_{k=0}^{7} (s_{i,k} - \mu_i)^2}, \qquad z_{i,k} = \frac{s_{i,k} - \mu_i}{\max(\sigma_i, 10^{-6})} \quad (k \in \{0,\dots,7\})$$
2. **Soft invalid-risk penalty (`slot 0 untouched`)**:
   $$\text{score}'_{i,0} = z_{i,0}, \qquad \text{score}'_{i,k} = z_{i,k} - \lambda \cdot p_{\text{invalid}}(i, k) \quad \text{for } k \in \{1,\dots,7\}$$
3. **Single-pick margin rule (ties keep `r882`)**:
   Let $k^* = \arg\max_{k \in \{0..7\}} \text{score}'_{i,k}$. The selected candidate slot is:
   $$\text{pick}_i = \begin{cases} k^* & \text{if } k^* > 0 \text{ and } \text{score}'_{i,k^*} - \text{score}'_{i,0} > m \\ 0 & \text{otherwise} \end{cases}$$
4. **Inner-`FIT` Joint `(lr, n_estimators, lambda, margin)` Selection**:
   Inside each `(fold_seed, model_seed, assess_fold)` cell (`164` `FIT` families, `4`-fold inner CV), `(lr, n_est, lambda, m)` is chosen **ONLY inside inner `FIT`** over:
   $$\text{LR\_GRID} \times \text{N\_EST\_GRID} \times \underbrace{\{0.0, 0.5, 1.0, 2.0, 4.0, 8.0\}}_{\text{SOFT\_LAMBDA\_GRID}} \times \text{MARGIN\_GRID}$$
   by maximizing the inner-`FIT` equal-family utility gain subject to `fit_invalid <= fit_ctrl_invalid` (ties broken by lower `fit_invalid`, higher `lambda`, and higher `margin`), followed by the strict inner-`FIT` fallback (`best_gain > +0.001`, `inner_lcb95 > 0.0`, `fit_invalid <= fit_ctrl_invalid`).

---

## 3. Experimental Arms

1. **`CONTROL`**: Frozen `r882` baseline (`slot 0` pick).
2. **`RERANK8_CR` (Reference)**: Base `K = 8` clean-room `LGBMRanker` (`H273` specification).
3. **`RERANK8_CR_VETO` (Reference)**: Hard `tau = 0.10` invalid-gate veto (`H273` specification).
4. **`RERANK8_CR_SOFT` (Confirmatory New Arm)**: Soft invalid-risk penalized `K = 8` reranker (`score'_k = z_k - lambda * p_invalid_k` for `k >= 1`, slot `0` untouched) with strict inner-`FIT` fallback.
5. **`RERANK8_CR_SOFT_SHUF_UNC` (Primary Unconditional Shuffle Placebo)**: Within-family permuted relevance labels (`rel_8`) and within-family permuted invalid-gate labels (`y_invalid`) on inner `FIT`, joint `(lr, n_est, lambda, margin)` selection over finite margins subject to `fit_invalid <= fit_ctrl_invalid`, with **inner-`FIT` fallback disabled** so the placebo is forced to act on all `45` cells.
6. **Supporting / Diagnostic Arms**: `RERANK8_CR_SOFT_SHUF` (strict fallback), `RERANK8_CR_SOFT_UNC` (unconditional `SOFT`), `RERANK8_CR_SHUF`, `RERANK8_CR_SHUF_UNC`, `RERANK8_CR_SHUF_UNC_M0`, `RERANK8_CR_UNC`, `TOP4_PICK_ORACLE`, `TOP8_PICK_ORACLE`, `POOL_ORACLE`.

---

## 4. Evaluation Panels, Fresh Seeds, & Multiplicity Control (`Holm m = 3`)

- **Fresh Seeds (`H275`)**:
  - `fold_seeds = (275140, 275141, 275142)` $\times$ `5` family-grouped folds $\times$ `model_seeds = (275101, 275102, 275103)` = `45` fit cells/arm.
  - `boot_seed = 275150` (`10,000` paired family-cluster bootstrap draws).
- **Panels**:
  1. **`PRIMARY` (Confirmatory)**: `RERANK8_CR_SOFT - CONTROL` on `DL-Train` OOF (`205` families, `1,219` contexts, `45` cells/arm).
  2. **`SECONDARY` (Supporting Only — `DL-Val91` Heavily Reused)**: `RERANK8_CR_SOFT - CONTROL` on `DL-Val91` (`19` families, `91` contexts, `fold_seed = 275140`, `3` model seeds + `3`-fold-seed aggregate). `DL-Val91` is unopened until all `DL-Train` + `TERTIARY` selections are `fsync`-locked in `fit_selections_lock.json`.
  3. **`TERTIARY` (`Cluster41` Leave-Family-Cluster-Out Panel)**: Evaluates all arms across `model_seeds = (275101, 275102, 275103)` on the `41` least-similar `DL-Train` families (`201` contexts) reused from `H272`'s S3 `fold_manifest.json` (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H272/prereg-f629cb79/fold_manifest.json`, `VersionId = 6NAoV3LWmW0.ZY5lbGs8fawZ_hUE47bP`), where the remaining `164` families form the `4` inner `FIT` splits (`fold_seed = 275140`).
- **Multiplicity Control (`Holm m = 3`)**:
  Step-down `Holm-Bonferroni` (`m = 3`) is applied over:
  1. `PRIMARY`: `RERANK8_CR_SOFT - CONTROL` on `DL-Train`,
  2. `SECONDARY`: `RERANK8_CR_SOFT - CONTROL` on `DL-Val91`,
  3. `SOFT - VETO`: `RERANK8_CR_SOFT - RERANK8_CR_VETO` on `DL-Train`.
  Both `95%` bootstrap CIs (`[2.5%, 97.5%]`) and `Holm m = 3` (`98.33%` two-sided / one-sided `alpha = 0.025 / 3 = 0.00833`) bootstrap CIs (`[0.833%, 99.167%]`) are reported.

---

## 5. Pre-Declared `ADVANCE` Decision Rule & Key Harm Readout

### 5.1 `ADVANCE` Criteria (All 8 Required)
`H275` returns `verdict = "ADVANCE"` (`winning_arm = "RERANK8_CR_SOFT"`) if and only if all 8 conditions hold:
1. **`PRIMARY` Median Gain**: `RERANK8_CR_SOFT - CONTROL` on `DL-Train` median family gain `> +0.020`;
2. **`Holm m = 3` Lower Confidence Bound**: `PRIMARY` `Holm m = 3` LCB `> 0.0` (`ci9833_holm3[0] > 0.0` and `holm_p_m3 < 0.025`);
3. **Seed & Repetition Consistency**: `3/3` model seeds positive AND `>= 7/9` `(model_seed × fold_seed)` repetitions positive on `DL-Train`;
4. **`DL-Train` Invalid Cap**: `RERANK8_CR_SOFT` `-2.0` invalid picks across `45` cells `<= 153` (`CONTROL`);
5. **`DL-Val91` Invalid Cap**: `RERANK8_CR_SOFT` `-2.0` invalid picks `<= CONTROL` (`0`) on `3/3` model seeds;
6. **Harmed-Family Cap vs `VETO`**: `RERANK8_CR_SOFT` `DL-Train` harmed-family median (`#families with delta < -0.01`, median across the 3 model seeds) `<= RERANK8_CR_VETO harmed-family median + 10`;
7. **Unconditional Shuffle Placebo Cap**: `RERANK8_CR_SOFT_SHUF_UNC - CONTROL` median family gain `<= +0.005` on `DL-Train`;
8. **Non-Fallback Identity**: `RERANK8_CR_SOFT` changes `> 0` picks (`total_changed_picks_45cells > 0`; fallback identity = `NO_ADVANCE`).

### 5.2 Pre-Declared Key Harm Readout
1. `RERANK8_CR_SOFT - RERANK8_CR` and `RERANK8_CR_SOFT - RERANK8_CR_VETO` medians + `95%` / `Holm m=3` CIs on `DL-Train`, `DL-Val91`, and `Cluster41`;
2. Harmed families (`> 0.01`) per seed and median across seeds for all arms;
3. Worst family loss per seed and median across seeds;
4. Raw `-2.0` invalid picks vs `CONTROL` (`153` on `DL-Train`, `0` on `DL-Val91`, `Cluster41` on `TERTIARY`);
5. Per-seed and median utility deltas on recurring `DL-Val91` family `fam-769243ee920953f082bd6784c94795f919abe67c243fae912657abc080863c0c` (plus `fam-d73e184e...` and `fam-ccf8d821...`);
6. `TERTIARY` `41`-family least-similar leave-cluster-out panel (`201` contexts) for all arms;
7. MM Bradley-Terry Elo (`CONTROL = 1500.00`, `1,000` family-cluster bootstrap draws) and family W/T/L on all panels. No AUC, no speed verdicts.

---

## 6. Resource & Hygiene Contract (`xbabe2`)
- `taskset -c 57-60`, `nice -n 19`, `threads <= 4`, `RSS <= 4 GiB`, `load5 <= 100.0` (`pause > 108.0`), swap-in-rate (`pswpin`) + `MemAvailable >= 12 GiB` guard (never `swap_free`), CPU only.
- Preregistration pushed to `origin/v1/next` by `21:05Z` before outcomes; compute stop `23:00Z`; closeout pushed to `origin/v1/next` by `23:45Z`.
