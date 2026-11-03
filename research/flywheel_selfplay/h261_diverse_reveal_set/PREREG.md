# H261 Preregistration (`h261_diverse_reveal_set`)

## 1. Scientific Question & Hypothesis

**Question**: Is the 4-reveal set in `TOP4_PICK` (`r882` top-4 by predicted cost) too redundant, and does diversity-aware 4-reveal selection (`MMR4_CV` or `STRUCT4`) beat `TOP4_PICK` at a strictly matched 4-reveal budget (`0` optimizer updates, CPU only)?

**Core Claim Under Test**:
- H258 established that `TOP4_PICK` (`4` non-adaptive reveals on gated contexts `top_zgap < 0.25`) reaches `DL-Val` (`Native91`, `91` contexts / `19` families) mean context utility `1.2403601682250796` and mean family utility `1.2519997168684853`, beating adaptive `ACTIVE_2PLUS2_HURDLE` (`1.1578028336339840` ctx / `1.1687895062564112` fam) at matched 4-reveal budget.
- However, the top-4 candidates ranked by `r882` may share almost identical embeddings or learner/DSL head structure, wasting reveal budget on near-duplicate candidates.
- H261 tests whether selecting `4` diverse candidates from `r882`'s `top-16` pool—either by maximal marginal relevance (`MMR4_CV`) over standardized candidate embeddings or by structural diversity across distinct learner families / DSL heads (`STRUCT4`)—improves revealed winner quality over `TOP4_PICK` at the exact same `4`-reveal budget on gated contexts (`0` reveals on non-gated contexts).
- **Important Taxonomy Framing**: Any positive gain from `MMR4_CV` or `STRUCT4` vs `TOP4_PICK` is an **equal-budget search-policy lift (4 reveals vs 4 reveals on gated contexts, 0 reveals elsewhere)**, **NOT** a zero-reveal checkpoint lift.

## 2. Locked Inputs & Provenance

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Primary Anchor Checkpoint (`R882`) | `/home/ubuntu/jope_plan/evidence/H261/argon-t1530/inputs/r882/round-882.pt` | `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` |
| Diagnostic Anchor Checkpoint (`R3175`) | `/home/ubuntu/jope_plan/evidence/H261/argon-t1530/inputs/r3175/round-3175.pt` | `1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b` |
| Canonical Split Manifest (`dope.json`) | `/home/ubuntu/jope-data/h50/splits/dope.json` (`DL-Train`: `1,219` ctx / `205` fam; `DL-Val`: `91` ctx / `19` fam) | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` |
| `DL-Train` 5-Fold Family-Grouped Split (`fold_seed = 261140`) | `research/flywheel_selfplay/h261_diverse_reveal_set/fold_manifest.json` | Computed before any outcome read (`outcomes_read = 0`) |

## 3. Experimental Arms (Matched 4-Reveal Budget on Gated Contexts)

All arms use the frozen H258 code and constants, uncertainty gate `top_zgap < 0.25`, `0` reveals on non-gated contexts (`top_zgap >= 0.25`, picking `r882` rank-1), and `4` unique reveals from `top-16` on gated contexts (`top_zgap < 0.25`).
For every reveal-4 arm, final pick on a gated context is:
$$\text{pick} = \arg\min_{i \in S_4} \text{cost}_{\text{true}}(i), \quad \text{cost}_{\text{true}}(i) = \begin{cases} \text{fitness}(i) & \text{if } \text{gate}(i) = 1 \\ 2.0 & \text{otherwise} \end{cases}$$
with ties broken deterministically by ascending `r882` predicted-cost order.

1. **`R882` (`0` reveals, baseline anchor)**:
   - Picks `argmin rcost` from `round-882.pt`.
   - Must reproduce `Native91` mean context utility `1.1377941898164525` bit-exact (`<= 1e-12`) BEFORE any other arm runs.
2. **`TOP4_PICK` (`4` reveals on gated contexts, baseline)**:
   - Reveals `r882` `top-4` (`[r_0, r_1, r_2, r_3]`) on gated contexts (`top_zgap < 0.25`).
   - Must reproduce H258 `Native91` mean context utility `1.2403601682250796` and mean family utility `1.2519997168684853` bit-exact (`<= 1e-12`) IMMEDIATELY after `R882`.
3. **`MMR4_CV` (**Primary**, `4` reveals on gated contexts)**:
   - Candidate embedding `emb40(i)`: pool-standardized concatenation of `[sem24(i), proj16(i)]` (where `proj16(i) = model.prog_proj(prog_412(i))`), L2-normalized to unit length so `cos_sim(i, j) = dot(emb40(i), emb40(j))`.
   - On gated contexts (`top_zgap < 0.25`), starts with `S_1 = [r_0]` (`r882` rank-1) and greedily selects `3` additional candidates from `top_16 = [r_0, ..., r_15]`:
     $$\text{MMR\_score}(i \mid S) = -z_{\text{cost}}(i) - \lambda \max_{j \in S} \cos\_\text{sim}(i, j)$$
     with ties broken by `r882` order.
   - Cross-validation protocol for `lambda`: selected **exclusively** on `DL-Train` (`1,219` contexts / `205` families) via 5-fold family-grouped OOF (`fold_seed = 261140`) over grid `LAMBDA_GRID = {0.1, 0.2, 0.3, 0.5, 1.0}`. For each fold `f in 0..4`, `lambda*_f` maximizes training-folds (`!= f`) mean family utility (ties broken toward smaller `lambda`); overall `lambda*_global` maximizes `DL-Train` family utility and is frozen for `DL-Val` (`Native91`).
4. **`STRUCT4` (**Secondary**, `4` reveals on gated contexts)**:
   - Uses each candidate's outcome-free structural identifiers extracted from `prog_412` (`prog_412[:, 24:29]` one-hot estimator family `est_id` and `dsl_head_id` encoding `(est_id, active_op_bitmask)` over the 23 DSL operators `prog_412[:, 0:23] > 0`).
   - Scans `top_16 = [r_0, ..., r_15]` in ascending `r882` predicted-cost order: first takes the best-`r882`-cost candidate from each distinct learner family `est_id`; if `< 4` distinct learner families exist in `top_16`, fills from distinct `dsl_head_id` in ascending `r882` order; if still `< 4`, falls back to next by `r882` order.
5. **Descriptive & Diagnostic Comparators**:
   - `RAND4_OF16`: on gated contexts, reveals `r_0` (`r882` rank-1) plus `3` candidates sampled uniformly without replacement from `top_16[1:16]` using seed `s in {261101, 261102, 261103}`.
   - `STRUCT4_EST_ONLY`: diagnostic sensitivity check diversifying only over the `5` estimator families (`est_id`) in `top_16` (fallback by `r882` order).
   - `R3175`, `TOP4_PICK@r3175`, `MMR4@r3175`, and `STRUCT4@r3175`.

## 4. Hypotheses, Multiple-Testing Control (`Holm m = 2`), and Decision Rule

- **Primary Contrast (`Holm m = 2`)**: `MMR4_CV - TOP4_PICK` on `DL-Val` (`Native91`, `91` contexts / `19` families) paired family-level utility gain.
- **Secondary Contrast (`Holm m = 2`)**: `STRUCT4 - TOP4_PICK` on `DL-Val` (`Native91`, `91` contexts / `19` families) paired family-level utility gain.
- **Statistical Inference**:
  - Paired family-level bootstrap (`B = 10,000` resamples over the `19` `DL-Val` families and `205` `DL-Train` families, `boot_seed = 261150`).
  - Reports median family gain, mean family gain, raw 95% CI, Holm-Bonferroni step-down adjusted `p`-value and Simultaneous 95% CI (`m = 2`, 97.5% quantile bounds), family Win/Tie/Loss counts (`|delta| <= 1e-4` = tie), and harmed families (`delta < -0.01`).
  - Reports `400`-iteration MM Bradley-Terry Elo (`10,000` family-bootstrap draws, `seed = 261150`) **anchored at `TOP4_PICK = 1500.0`**.
  - No AUC or speed claims are used for verdicts.
- **Seeds (`261101, 261102, 261103`)**:
  - `R882`, `TOP4_PICK`, `MMR4_CV`, and `STRUCT4` are deterministic given the frozen checkpoint and `fold_manifest.json` (`fold_seed = 261140`), whereas evaluation seeds `261101–261103` vary `RAND4_OF16` sampling and the permuted-fitness negative control. We explicitly report that deterministic arms have identical values across evaluation seeds (`weak seed evidence` for stochastic variance).
- **Required Redundancy & Mechanism Diagnostics**:
  - Mean pairwise cosine similarity (`emb40` and `sem24`) among the `4` chosen candidates on gated contexts (`TOP4_PICK`, `MMR4_CV`, `STRUCT4`, `RAND4_OF16`).
  - Mean distinct learner families (`1..5`) and distinct DSL heads (`1..4`) among the `4` chosen candidates on gated contexts.
  - Oracle-best `recall@4` on gated contexts (`oracle_top16_recall_at_4` and `oracle_pool_recall_at_4`).
  - Per-family harm audit on `DL-Val`.
- **Negative Control**:
  - Sign-permutation null and permuted-fitness negative control on `DL-Val` (`95% CI` covers `0`).
- **ADVANCE Criteria (Equal-Budget Search-Policy Lift Only)**:
  - A tested co-primary/secondary arm (`MMR4_CV` or `STRUCT4`) advances if and only if on `DL-Val` (`Native91`):
    1. Median family utility gain vs `TOP4_PICK` `> +0.002`,
    2. Holm-adjusted 95% LCB `> 0`,
    3. Holm-adjusted `p < 0.05`,
    4. `>= 2/3` evaluation seeds positive,
    5. Permuted negative control 95% CI covers `0`.
  - Ties or failures on any criterion yield `NO_ADVANCE`.
