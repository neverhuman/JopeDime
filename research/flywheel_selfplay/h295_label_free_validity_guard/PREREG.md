# H295 Preregistration — Label-Free Validity Guard on `ENS_VOTE2` (`h295_label_free_validity_guard`)

- **Wave**: `H295` (`h295_label_free_validity_guard`)
- **Lane / Agent**: `rd_argon` (`[ARGON]`)
- **Pre-outer-scoring Fit-Lock Content SHA256**: `c378913b35a3aacba85672a80b7ca52e0d8854762c28f421d767c1c67baccc3e`
- **Pre-outer-scoring `fit_lock.json` File SHA256**: `adf8912732b45c9f7bebf13f6d8440bed68ff8041b9d3d06086235c0edee2223`
- **Outer-Fold Validation Labels Read Before Lock**: `0` (outer validation fold `va` utilities and validity labels are masked to `NaN` / `-1` during `--mode fit_and_lock`)
- **Forbidden Cohort / Split Reads (`DL-Val`, `DL-Test`, holdout, sealed, `H286v2`/`v3`/`v4`/`v5`/fresh)**: `0` (enforced via `sys.addaudithook`)

---

## 1. Context & Hypothesis

In `H292` (`f9e31c57`), `H287` `ENS_VOTE2` passed every statistical clause on fresh ODD cohort v4 (`+0.0665 [+0.021, +0.122]`, Holm LCB `+0.0057, +0.0060, +0.0058` on `3/3` seeds, sign test $p = 3 \times 10^{-5}$, harmed `5.4%`, Elo `1687`), failing **only** on invalid (`-2.0`) picks (`7` vs `slot 0`'s `6`).

**Hypothesis (`H295`, `h295_label_free_validity_guard`)**:
A **label-free validity guard** on top of `ENS_VOTE2` (at inference: if the voted pick's predicted-invalid probability $> \tau$, fall back to the next voted candidate, else `slot 0`) cuts invalid picks to $\le$ `slot 0`'s count (and strictly below `ENS_VOTE2`'s count on `DL-Train`) while keeping $\ge 85\%$ of `ENS_VOTE2`'s median gain and not increasing harmed families above `ENS_VOTE2`, on `DL-Train` 4-fold family-grouped outer CV.

---

## 2. Pre-Fit Regression Gate (Verified Before Any Fit)

Before fitting any `H295` model, `valguard.verify_h287_ens_vote2_regression` reproduced the published `H287` `ENS_VOTE2` `DL-Train` inner-CV cell bit-exactly from published artifacts:
- `n_families = 205`, `n_contexts = 1219`
- `per_seed_gain_exact = [0.053902857747975874, 0.059931533238705134, 0.062203897397489716]` (`round6 = [0.053903, 0.059932, 0.062204]`)
- `median_gain_exact = 0.059931533238705134` (`round6 = 0.059932`)
- `utility_vector_max_abs_diff_by_seed = [0.0, 0.0, 0.0]`
- `harmed_001_by_seed = [24, 17, 22]` (median `22.0`)
- `invalid_by_seed = [14, 13, 14]` (median `14.0`, vs `slot 0` `[17, 17, 17]`)
- `changed_by_seed = [296, 268, 265]`

---

## 3. Experimental Design & Guard Architecture

- **Data**: `DL-Train` only (`205` families, `1,219` contexts, $K = 8$ candidates per context from `H287`/`H291` cached `dl_train_member_oof.npz`). Zero reads of `DL-Val`, `DL-Test`, holdout, sealed, or any fresh cohort (`v1`–`v5`).
- **Outer CV**: 4-fold family-grouped outer CV across 3 outer seeds `295101`, `295102`, `295103` (controlling both the 4-fold family partition `outer_fold_ids` and model fit random streams).
- **Label-Free Validity Guard (`VALGUARD`)**:
  - **Input features (candidate features only, zero fitness, zero labels at inference)**:
    - `568` raw H278 candidate features `X[i, k, :]` for the shallow regularized `LGBMClassifier` (`veto.fit_gate`: `n_estimators=70, lr=0.05, num_leaves=15, min_child_samples=25, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0`).
    - `652` augmented candidate features (`568` raw cols + `40` slot-0 difference cols `X[i, k, :40] - X[i, 0, :40]` + `40` within-context mean-centered cols `X[i, k, :40] - mean_k(X[i, :, :40])` + `4` `sem_z`/`proj_z` L2/L-inf norm cols) for the standardized L2 `LogisticRegression(C=0.05, max_iter=500)`.
    - Predicted invalid probability: $\hat{p}_{\text{inv}}(i, k) = 0.5 \, \hat{p}_{\text{GBM}}(i, k) + 0.5 \, \hat{p}_{\text{LR}}(i, k)$.
  - **Training labels**: Binary invalidity $y_{\text{inv}}(i, k) = \mathbb{1}[\neg \text{gate\_k}(i, k)]$, used **only** inside each outer training split `tr` (never at inference and never on `va`).
  - **Nested inner-CV threshold selection**: Inside each outer training split `tr` (`3` of `4` family folds), a 3-fold family-grouped nested inner CV selects $\tau \in \{0.030, 0.035, 0.040, 0.045, 0.050, 0.060\}$ maximizing feasibility ($\text{inv} \le \text{inv}_{\text{slot0}}$, $\text{gain} \ge 0.85 \times \text{gain}_{\text{ENS\_VOTE2}}$, $\text{harmed} \le \text{harmed}_{\text{ENS\_VOTE2}}$) and then minimizing invalid count, harmed families, and maximizing gain on `tr`.
  - **Inference fallback rule**:
    - Let $c_0$ be `ENS_VOTE2`'s primary pick ($\ge 2$ member `u_pbag` votes on $j > 0$, else $0$), and `next_cands` be any other candidate $j > 0$ ($j \ne c_0$) with $\ge 1$ member `u_pbag` vote ordered by `(vote_count desc, bag_vote_sum desc, mean_raw_score desc, -j)`.
    - If $c_0 > 0$: keep $c_0$ if $\hat{p}_{\text{inv}}(i, c_0) \le \tau$; else fall back to the first $c_{\text{alt}} \in \text{next\_cands}$ with $\hat{p}_{\text{inv}}(i, c_{\text{alt}}) \le \tau$, else `slot 0` (unless $\hat{p}_{\text{inv}}(i, 0) > \hat{p}_{\text{inv}}(i, c_0)$, avoiding a fallback onto a strictly riskier `slot 0`).
    - If $c_0 = 0$: stay at `slot 0` unless $\hat{p}_{\text{inv}}(i, 0) > \max(\tau, 0.055)$, in which case fall back to the first $c_{\text{alt}} \in \text{next\_cands}$ with $\hat{p}_{\text{inv}}(i, c_{\text{alt}}) \le \min(\tau, 0.015)$, else `slot 0`.
- **Arms**:
  1. `SLOT0`: Baseline `r882` slot 0.
  2. `ENS_VOTE2`: Reference `H287` 2-of-3 member agreement policy.
  3. `ENS_VOTE2+VALGUARD` (**Primary**): `ENS_VOTE2` with the nested-inner-CV-tuned label-free validity guard.
  4. `ENS_VOTE2+VALGUARD+HARMVETO` (**Secondary**): `ENS_VOTE2+VALGUARD` plus an in-fold shallow LightGBM harm-veto gate (nested inner-CV $\tau_{\text{harm}} \in \{0.15, 0.18, 0.20, 0.22, 0.25, 0.30\}$) that abstains to `slot 0` only when $\hat{p}_{\text{harm}} \ge \tau_{\text{harm}}$ and $\hat{p}_{\text{inv}}(i, 0) \le \tau_{\text{val}}$.
  5. `SHUFFLED_NULL` (**Null**, seeds `295201–295203`): Identical hybrid validity guard architecture and nested inner-CV $\tau$ selector trained on permuted validity labels $y_{\text{inv}}$ within each training fold.

---

## 4. Strict Decision Rule (`VALGUARD_ADVANCE` vs `VALGUARD_NOT_ADVANCED`)

Primary (`ENS_VOTE2+VALGUARD`) passes (`VALGUARD_ADVANCE`) iff on **all 3/3 seeds**:
1. `invalid picks <= slot 0` (`17`) on `3/3` seeds AND strictly cuts invalid picks relative to `ENS_VOTE2` (`invalid_by_seed <= ENS_VOTE2` on `3/3` seeds with strict reduction in total/median invalid picks).
2. `10,000`-draw family-bootstrap `LCB95 > 0` vs `slot 0` (`boot_seed = 295150`) on `3/3` seeds.
3. `3`-seed median family-equal mean gain $\ge 0.85 \times$ `ENS_VOTE2` median gain ($\ge 0.85 \times 0.05993153 = +0.050942$).
4. Harmed families (losing $> 0.01$ vs `slot 0`) not above `ENS_VOTE2` (`harmed_by_seed <= [24, 17, 22]` on `3/3` seeds and median $\le 22.0$).
5. `SHUFFLED_NULL` (`295201–295203`) fails to reproduce the invalid-pick cut and fails the primary gate.

Ties ($|\Delta u_{\text{fam}}| \le 10^{-4}$) never count as wins. If `VALGUARD_ADVANCE`, the frozen full-`DL-Train` validity guard models locked in `fit_lock.json` are exported to S3 for a later fresh test (no serving promotion in `H295`).
