# H291 Preregistration (`h291_harm_veto_abstention`)

- **Hypothesis ID**: `H291` (`h291_harm_veto_abstention`)
- **Lane**: `rd_argon` (`[ARGON]`)
- **Wave**: `rd_breakthrough_20261030`
- **Parent Artifacts**:
  - `H287` (`h287_bagswitch_seed_ensemble_export`, commit `213b30ad52e1ce48927bba960ed609bd98ab4dfe`, verdict `EXPORTED_ADVANCE`)
  - `H284` (`h284_r882_bagged_ranker_fresh200_transfer`, commit `7346a8fc63`)
  - `H278` (`h278_FullDLTrain_RankerVeto_C100`, `568`-column `topk` feature representation)
- **Host / CPU**: `xbabe2` CPU cores `53-56`, `nice 19`, numeric threads `1`, aggregate threads `4`, `RSS <= 4 GiB`, `MemAvailable >= 10 GiB`. Zero GPU / zero neural forwards or updates. `qualification.lock` and cohort v3 / ODD are untouched (`H288` owns them).

## 1. Hypothesis

A harm-aware abstention gate — a family-grouped, inner-CV-trained binary classifier predicting $P(\text{switch from slot } 0 \text{ loses } > 0.01)$, plus a seed-agreement requirement — layered on the frozen `H287` `CLIP`/`ENS` members keeps $\ge 70\%$ of the `CLIP` inner-CV gain while cutting harmed families to $\le \text{ENS\_VOTE3}$ levels and makes all 3 seeds $\text{LCB}_{95} > 0$.

## 2. Protocol & Chronological Separation

1. **Step 1 — Bit-Exact Regression Check**:
   - Reproduce `H287` inner-CV cell `CLIP` seed `287101` on `DL-Train` (`205` families, `1,219` contexts) bit-exact from published artifacts (`utility_vector_max_abs_diff == 0.0`, `gain_abs_diff == 0.0`, `harmed == 31`, `raw -2 == 13`, `changed == 472`).
2. **Step 2 — Part A (`DL-Train` Only, `205` Families / `1,219` Contexts, 4-Fold Family-Grouped Inner CV, Seeds `291101–291103`)**:
   - **Features (`644` columns)**:
     - `568` `H278` `topk` columns of the proposed candidate slot $j > 0$ (`X[i, j, :]`).
     - `44` member/bag disagreement and slot-0 margin columns across the 3 members and $3 \times 8 = 24$ family-bootstrapped bags (member margin mean/std/min/max/spread, bag vote count sum/min/max/std, 24-bag margin mean/std/min/q25/median/q75/max and exceedance fractions $>0, >0.10, >0.20$, invalid-veto probability stats on slot $j$ and slot $0$, slot-0 margin over best alternative, and member agreement counts across `CLIP` and `H284` members).
     - `32` candidate-minus-slot-0 `topk` feature difference columns (`X[i, j, :32] - X[i, 0, :32]`).
   - **In-Fold Harm Classifier**:
     - Regularized shallow LightGBM binary classifier (`n_estimators=60`, `learning_rate=0.05`, `num_leaves=7`, `max_depth=3`, `min_child_samples=15`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=1.0`, `reg_lambda=2.0`) trained in-fold on proposed switches ($j > 0$) with binary target $y_{\text{harm}} = \mathbb{1}[u(i, j) - u(i, 0) < -0.01]$.
   - **Inner-Inner CV Threshold Selection (No Outer-Fold Peeking)**:
     - Inside each outer training split `tr` (3 of 4 outer folds), split `tr` into 3 family-grouped inner-inner folds `(itr, iva)` and compute inner-inner OOF harm probabilities $\hat{p}_{\text{harm}}^{\text{ii}}$ on `tr` over $\mathcal{T} = \{0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30\}$:
       - For `ENS_VOTE2+HARMVETO`: select $\tau_h^* \in \mathcal{T}$ maximizing `tr` family-equal mean gain subject to `tr` harmed families $\le H_{\text{ENS\_VOTE3}}(\text{tr})$ and gain $\ge 0.02$.
       - For `CLIP+HARMVETO`: select $\tau_h^* \in \mathcal{T}$ minimizing `tr` harmed families subject to retaining $\ge 0.70 \times G_{\text{CLIP}}(\text{tr})$ on `tr`.
   - **Part A Arms**:
     - `SLOT0` (anchor baseline)
     - `CLIP` (frozen `H287` `-1.0` clipped single-seed `BAG_SWITCH` member)
     - `ENS_VOTE3` (frozen `H287` 3-of-3 unanimous member agreement)
     - `CLIP+HARMVETO` (`CLIP` gated by in-fold harm-veto classifier at inner-inner CV threshold $\tau_h^*$)
     - `ENS_VOTE2+HARMVETO` (`ENS_VOTE2` 2-of-3 member agreement gated by in-fold harm-veto classifier at inner-inner CV threshold $\tau_h^*$)
     - `PLACEBO` (harm-veto classifier trained on within-family permuted harm labels at the same $\tau_h^*$)
     - `NULL_LF` (random abstention on proposed switches at the exact fold-matched abstention rate)
3. **Step 3 — Fit-Lock Hash + Preregistration on `origin/v1/next` BEFORE Part B**:
   - Export frozen full-`DL-Train` harm-veto boosters and lock their `sha256` digests and inner-CV-selected thresholds in `fit_lock.json`, then push `PREREG.md` and `fit_lock.json` to `origin/v1/next` (non-force) before any Part B descriptive scoring.
   - **Locked `fit_lock.json` hashes (locked at `2026-11-04T01:21:50Z`, `part_b_pools_read_during_part_a = 0`)**:
     - `fit_lock_content_sha256`: `cd04dec452bacd84ea68a1c08e53a214ba2452b09c982652b5aeaf7fff2498cf`
     - `fit_lock.json` file `sha256`: `3d05461907ab6d4513204f70d98800c7af2fe8ca244f0517f40bb26af913f1da`
     - `h287_fit_lock_sha256`: `8a20c58bdb4bfcbadfa5ba927aa0577b630e3174e6dc4a7bfd579175ca08f35f`
     - `dl_train_oof_cache_sha256`: `7bbad071d17edc6c4c76a3c81acf67443e30f2443603a9e9af52fab5c87882de`
     - `best_by_prereg_arm`: `ENS_VOTE2+HARMVETO` (`locked_part_b_taus = [0.135, 0.165, 0.135]`; `CLIP+HARMVETO` `locked_part_b_taus = [0.235, 0.175, 0.250]`)
4. **Step 4 — Part B (`SEMI_FRESH`, Descriptive Only, Labeled `reused`)**:
   - Score the frozen best-by-prereg arm (`ENS_VOTE2+HARMVETO`, alongside the full frozen arm panel for completeness) on `H286` v2 `200` EVEN `r3175` pools (already opened by `H286`/`H290`). No retuning after Step 3. Part B only annotates and is NOT a fresh claim.

## 3. Metrics & Verdict Rule

- **Metrics per seed (`291101–291103`)**:
  - Family-equal mean gain vs `slot 0`
  - `10,000` family bootstrap draws (`BOOT_SEED = 291150`), median of 3 seeds + `CI95`, per-seed `CI95`
  - Family-median gain
  - Exact binomial sign test on non-tied families ($|d| > 10^{-4}$; ties never count as wins)
  - Harmed families ($\text{loss} > 0.01$)
  - Raw `-2.0` invalid picks
  - Abstain rate ($\text{mean}(\text{pick} == 0)$)
  - Minorization-Maximization (MM) Bradley-Terry Elo (`slot 0 = 1500`, $10^{-4}$ tie band)
- **Step 6 Verdict Rule**:
  - `HARMVETO_ADVANCE` iff on **Part A**:
    1. Median gain $\ge 0.7 \times \text{CLIP}$ median gain,
    2. $\text{LCB}_{95} > 0$ on `3/3` seeds,
    3. Harmed families median $\le \text{ENS\_VOTE3}$ harmed families median,
    4. Raw `-2` picks $\le \text{slot } 0$,
    5. `PLACEBO` and `NULL_LF` fail the same gate.
  - Otherwise `HARMVETO_NOT_ADVANCED`. Part B only annotates.
