# H291: Harm-Aware Abstention Gate on Frozen H287 CLIP/ENS Members (`h291_harm_veto_abstention`)

- **Verdict**: `HARMVETO_NOT_ADVANCED` (Part A Pareto split: `CLIP+HARMVETO` retains `82.65%` of `CLIP` median gain (`+0.072198 >= +0.061148`) with `3/3` seeds `LCB95 > 0` and `raw -2 <= slot 0` while cutting harmed families from `26.0` to `16.0` (`> 7.0`), whereas `ENS_VOTE2+HARMVETO` cuts harmed families to `6.0 <= 7.0` (`ENS_VOTE3` level) with `3/3` seeds `LCB95 > 0` and `+72.0%` higher gain than `ENS_VOTE3` (`+0.037399` vs `+0.021749`), retaining `42.81% < 70%` of `CLIP` median gain; Part B descriptive scoring on `H286` v2 `200` EVEN `r3175` pools shows `ENS_VOTE2+HARMVETO` cuts harmed families to `[0, 1, 1]` (median `1.0` vs `R882_FIT` `17.0` `[21, 7, 17]`) with median gain `+0.027144` (`95% CI [+0.002637, +0.064238]`)).
- **Lane**: `rd_argon` (`[ARGON]`)
- **Preregistration + Fit-Lock Commit (`origin/v1/next` BEFORE Part B)**: `d8bc373b80c80892600ddaaa83272616f2b1bda9` (`fit_lock_content_sha256 = cd04dec452bacd84ea68a1c08e53a214ba2452b09c982652b5aeaf7fff2498cf`, `part_b_pools_read_during_part_a = 0`)
- **Parent Artifacts**: `H287` (`213b30ad52e1ce48927bba960ed609bd98ab4dfe`), `H284` (`7346a8fc63`), `H278` (`568`-col `topk`)
- **Neural Forwards / Updates**: `0` forwards, `0` updates (`DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` untouched; `qualification.lock` and cohort v3/ODD untouched)

---

## 1. Hypothesis

A harm-aware abstention gate — a family-grouped, inner-CV-trained binary classifier predicting $P(\text{switch from slot } 0 \text{ loses } > 0.01)$, plus a seed-agreement requirement — layered on the frozen `H287` `CLIP`/`ENS` members keeps $\ge 70\%$ of the `CLIP` inner-CV gain while cutting harmed families to $\le \text{ENS\_VOTE3}$ levels and makes all 3 seeds $\text{LCB}_{95} > 0$.

---

## 2. Setup & Chronological Separation

1. **Step 1 — Bit-Exact Regression on Published `H287` Inner-CV Cell**:
   - Reproduced `H287` inner-CV cell `CLIP` seed `287101` across all `205` `DL-Train` families (`1,219` contexts) bit-for-bit (`utility_vector_max_abs_diff = 0.0`, `gain_abs_diff = 0.0`, `family_equal_mean_gain = +0.09355105605170717` (`float64`) / `+0.09355079434338156` (`float32`), `harmed_families_gt_0p01 = 31`, `raw_neg2_invalid_picks = 13`, `changed_contexts = 472`).
2. **Step 2 — Part A (`DL-Train` Only, `205` Families / `1,219` Contexts, 4-Fold Family-Grouped Inner CV, Seeds `291101–291103`)**:
   - **Features (`644` columns)**: `568` `H278` `topk` columns of proposed candidate slot $j > 0$ + `44` member/bag disagreement and slot-0 margin features across the 3 members and 24 family-bootstrapped bags + `32` candidate-minus-slot-0 `topk` difference columns.
   - **In-Fold Harm Classifier & Inner-Inner CV Threshold Selection**:
     - Shallow regularized LightGBM binary classifier (`n_estimators=60`, `learning_rate=0.05`, `num_leaves=7`, `max_depth=3`, `min_child_samples=15`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=1.0`, `reg_lambda=2.0`) trained in-fold on proposed switches ($j > 0$) with binary target $y_{\text{harm}} = \mathbb{1}[u(i, j) - u(i, 0) < -0.01]$.
     - Veto threshold $\tau_h^* \in \{0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30\}$ selected by 3-fold family-grouped inner-inner CV inside each outer training split `tr` (zero outer-fold `va` peeking):
       - `ENS_VOTE2+HARMVETO`: inner-inner CV maximizes `tr` family-equal mean gain subject to `tr` harmed families $\le H_{\text{ENS\_VOTE3}}(\text{tr})$ and gain $\ge 0.02$ (`inner_inner_selected_taus = [[0.15, 0.18, 0.12, 0.12], [0.25, 0.15, 0.18, 0.12], [0.15, 0.12, 0.18, 0.12]]`, locked Part B median $\bar{\tau}_h^* = [0.135, 0.165, 0.135]$).
       - `CLIP+HARMVETO`: inner-inner CV minimizes `tr` harmed families subject to retaining $\ge 0.70 \times G_{\text{CLIP}}(\text{tr})$ (`inner_inner_selected_taus = [[0.25, 0.22, 0.30, 0.20], [0.15, 0.15, 0.20, 0.22], [0.30, 0.20, 0.30, 0.15]]`, locked Part B median $\bar{\tau}_h^* = [0.235, 0.175, 0.250]$).
3. **Step 3 — Fit-Lock Hash + Preregistration on `origin/v1/next` BEFORE Part B**:
   - Exported all 18 frozen full-`DL-Train` harm-veto and placebo boosters to `/home/ubuntu/jope-data/h291/export/`, verified `part_b_pools_read_during_part_a == 0`, and pushed `PREREG.md` and `fit_lock.json` (`fit_lock_content_sha256 = cd04dec452bacd84ea68a1c08e53a214ba2452b09c982652b5aeaf7fff2498cf`) to `origin/v1/next` at commit `d8bc373b80c80892600ddaaa83272616f2b1bda9` (`2026-11-04T01:24:28Z`, 26 minutes before the `01:50Z` deadline) before touching Part B.
4. **Step 4 — Part B (`SEMI_FRESH`, Descriptive Only, Labeled `reused`)**:
   - Verified `fit_lock.json` and all 78 model `sha256` hashes (`60` `H287` boosters + `18` `H291` boosters), disabled all LightGBM `.fit()` methods, and scored the frozen `best_by_prereg_arm` (`ENS_VOTE2+HARMVETO`) and full arm panel on `H286` v2 `200` EVEN `r3175` pools (`200` families, `200` contexts; `R882_FIT` reproduced `H290` `561ae6519f` bit-exact: median gain `+0.062995`, harmed `[21, 7, 17]`, raw `-2` `[2, 2, 4]`).

---

## 3. Key Metrics

### Part A — Primary Evaluation (`DL-Train` 4-Fold Family-Grouped Inner CV, `205` Families / `1,219` Contexts, `10,000` Bootstrap Draws Seed `291150`)

| Arm | Median Gain vs `SLOT0` | 95% Bootstrap CI | Seed Gains (`291101–291103`) | `LCB95 > 0` Seeds | Family-Median Gain | Harmed Families (`>0.01`, Med / Per Seed) | Raw `-2` Picks (Per Seed) | Abstain Rate (Med / Per Seed) | MM BT Elo (Med / Per Seed) | Non-Tied W/L/T (Per Seed) | Sign-Test Two-Sided $p$ (Per Seed) |
| --- | ---: | --- | --- | :---: | ---: | --- | --- | --- | --- | --- | --- |
| `SLOT0` | `+0.000000` | `[+0.000000, +0.000000]` | `[+0.00000, +0.00000, +0.00000]` | `0/3` | `+0.000000` | `0.0` `[0, 0, 0]` | `[17, 17, 17]` | `1.000` `[1.000, 1.000, 1.000]` | `1500.00` `[1500.0, 1500.0, 1500.0]` | `0/0/205` $\times 3$ | `[1.0, 1.0, 1.0]` |
| `CLIP` | `+0.087354` | `[+0.042728, +0.147661]` | `[+0.09355, +0.06172, +0.08735]` | `3/3` | `+0.013052` | `26.0` `[31, 12, 26]` | `[13, 13, 10]` | `0.671` `[0.613, 0.797, 0.671]` | `1697.67` `[1714.2, 1667.8, 1697.7]` | `142/39/24`, `113/26/66`, `134/41/30` | `[1.1e-14, 7.6e-14, 1.1e-12]` |
| `ENS_VOTE3` | `+0.021749` | `[+0.008589, +0.032032]` | `[+0.02175, +0.01436, +0.02362]` | `3/3` | `+0.000000` | `7.0` `[9, 7, 7]` | `[18, 16, 14]` | `0.925` `[0.921, 0.925, 0.936]` | `1553.51` `[1562.1, 1552.3, 1553.5]` | `58/14/133`, `48/17/140`, `52/14/139` | `[1.6e-07, 1.5e-04, 2.5e-06]` |
| `CLIP+HARMVETO` | **`+0.072198`** | `[+0.033559, +0.128666]` | `[+0.07647, +0.04761, +0.07220]` | **`3/3`** | `+0.006632` | `16.0` `[24, 6, 16]` | **`[15, 14, 13]`** | `0.760` `[0.692, 0.865, 0.760]` | `1667.25` `[1678.0, 1611.8, 1667.2]` | `123/34/48`, `87/17/101`, `119/29/57` | `[7.2e-13, 2.6e-12, 6.2e-14]` |
| `ENS_VOTE2+HARMVETO` | **`+0.037399`** | `[+0.013765, +0.068738]` | `[+0.02108, +0.03740, +0.04652]` | **`3/3`** | `+0.000000` | **`6.0` `[6, 6, 5]`** | `[18, 14, 15]` | `0.911` `[0.925, 0.884, 0.911]` | `1581.65` `[1572.7, 1597.0, 1581.7]` | `57/12/136`, `67/16/122`, `60/13/132` | `[3.8e-08, 1.5e-08, 2.2e-08]` |
| `PLACEBO` | `+0.018858` | `[+0.006661, +0.037141]` | `[+0.01886, +0.03769, +0.00898]` | `3/3` | `+0.000000` | `8.0` `[8, 7, 12]` | `[16, 13, 17]` | `0.916` `[0.916, 0.881, 0.923]` | `1563.60` `[1563.6, 1583.7, 1546.9]` | `57/18/130`, `70/20/115`, `47/18/140` | `[7.0e-06, 1.1e-07, 4.2e-04]` |
| `NULL_LF` | `+0.028161` | `[+0.012576, +0.042242]` | `[+0.01927, +0.02816, +0.03449]` | `3/3` | `+0.000000` | `13.0` `[13, 11, 13]` | `[17, 12, 14]` | `0.911` `[0.925, 0.884, 0.911]` | `1555.78` `[1546.3, 1576.5, 1555.8]` | `53/21/131`, `71/25/109`, `61/22/122` | `[2.6e-04, 2.9e-06, 2.2e-05]` |

### Part B — Descriptive Annotation (`H286` v2 `200` EVEN `r3175` Pools, `SEMI_FRESH`, Labeled `"reused": true`, NOT a Fresh Claim)

| Arm | Median Gain vs `SLOT0` | 95% Bootstrap CI | Seed Gains (`291101–291103`) | `LCB95 > 0` Seeds | Family-Median Gain | Harmed Families (`>0.01`, Med / Per Seed) | Raw `-2` Picks (Per Seed) | Abstain Rate (Med / Per Seed) | MM BT Elo (Med / Per Seed) | Non-Tied W/L/T (Per Seed) | Sign-Test Two-Sided $p$ (Per Seed) |
| --- | ---: | --- | --- | :---: | ---: | --- | --- | --- | --- | --- | --- |
| `SLOT0` | `+0.000000` | `[+0.000000, +0.000000]` | `[+0.00000, +0.00000, +0.00000]` | `0/3` | `+0.000000` | `0.0` `[0, 0, 0]` | `[4, 4, 4]` | `1.000` `[1.000, 1.000, 1.000]` | `1500.00` `[1500.0, 1500.0, 1500.0]` | `0/0/200` $\times 3$ | `[1.0, 1.0, 1.0]` |
| `R882_FIT` (`H284` ref) | `+0.062995` | `[+0.018816, +0.115992]` | `[+0.06917, +0.06300, +0.03556]` | `2/3` | `+0.000000` | `17.0` `[21, 7, 17]` | `[2, 2, 4]` | `0.710` `[0.615, 0.815, 0.710]` | `1522.29` `[1522.3, 1530.8, 1513.0]` | `44/32/124`, `27/10/163`, `32/24/144` | `[0.20674, 0.00763, 0.34968]` |
| `CLIP` | `+0.047420` | `[+0.010493, +0.102325]` | `[+0.04742, +0.06300, +0.03556]` | `2/3` | `+0.000000` | `17.0` `[24, 7, 17]` | `[2, 2, 4]` | `0.710` `[0.680, 0.815, 0.710]` | `1512.96` `[1500.2, 1530.8, 1513.0]` | `32/31/137`, `27/10/163`, `32/24/144` | `[1.00000, 0.00763, 0.34968]` |
| `ENS_VOTE3` | `+0.016280` | `[-0.002732, +0.049255]` | `[+0.01628, +0.01628, +0.01628]` | `0/3` | `+0.000000` | `2.0` `[2, 2, 2]` | `[3, 3, 3]` | `0.960` `[0.960, 0.960, 0.960]` | `1503.48` `[1503.6, 1503.5, 1503.5]` | `5/3/192` $\times 3$ | `[0.72656, 0.72656, 0.72656]` |
| `CLIP+HARMVETO` | **`+0.046143`** | `[+0.010516, +0.089596]` | `[+0.04614, +0.04918, +0.03771]` | `2/3` | `+0.000000` | **`12.0` `[16, 4, 12]`** | `[2, 3, 4]` | `0.790` `[0.790, 0.875, 0.780]` | `1515.80` `[1503.6, 1523.1, 1515.8]` | `22/20/158`, `19/6/175`, `26/17/157` | `[0.87761, 0.01463, 0.22205]` |
| `ENS_VOTE2+HARMVETO` (`best_by_prereg`) | **`+0.027144`** | **`[+0.002637, +0.064238]`** | `[+0.00238, +0.02714, +0.03312]` | `2/3` | `+0.000000` | **`1.0` `[0, 1, 1]`** | `[4, 3, 3]` | `0.945` `[0.980, 0.940, 0.945]` | `1508.69` `[1503.8, 1510.4, 1508.7]` | `3/1/196`, `9/3/188`, `8/3/189` | `[0.62500, 0.14600, 0.22656]` |
| `PLACEBO` | `+0.015980` | `[-0.000284, +0.047474]` | `[+0.02967, +0.00081, +0.01598]` | `1/3` | `+0.000000` | `1.0` `[1, 2, 1]` | `[3, 4, 3]` | `0.965` `[0.940, 0.965, 0.975]` | `1501.90` `[1513.3, 1501.9, 1498.6]` | `10/2/188`, `4/3/193`, `2/3/195` | `[0.03857, 1.00000, 1.00000]` |
| `NULL_LF` | `+0.003873` | `[-0.001253, +0.012774]` | `[+0.00101, +0.03304, +0.00387]` | `1/3` | `+0.000000` | `1.0` `[1, 1, 2]` | `[4, 3, 4]` | `0.945` `[0.980, 0.940, 0.945]` | `1508.06` `[1503.8, 1513.0, 1508.1]` | `3/1/196`, `10/2/188`, `8/3/189` | `[0.62500, 0.03857, 0.22656]` |
| `ENS_VOTE2` (ungated ref) | `+0.039620` | `[+0.008442, +0.079990]` | `[+0.03962, +0.03962, +0.03962]` | **`3/3`** | `+0.000000` | `7.0` `[7, 7, 7]` | `[3, 3, 3]` | `0.825` `[0.825, 0.825, 0.825]` | `1518.49` `[1518.5, 1518.2, 1519.3]` | `23/12/165` $\times 3$ | `[0.08953, 0.08953, 0.08953]` |
| `CLIP_VOTE2` (ungated ref) | `+0.046073` | `[+0.008586, +0.091584]` | `[+0.04607, +0.04607, +0.04607]` | **`3/3`** | `+0.000000` | `9.0` `[9, 9, 9]` | `[2, 2, 2]` | `0.830` `[0.830, 0.830, 0.830]` | `1506.64` `[1506.6, 1505.9, 1508.1]` | `19/15/166` $\times 3$ | `[0.60759, 0.60759, 0.60759]` |
| `CLIP_VOTE2+HARMVETO` (aux) | `+0.022876` | `[+0.003314, +0.052417]` | `[+0.00891, +0.02288, +0.02890]` | **`3/3`** | `+0.000000` | **`0.0` `[0, 0, 2]`** | `[4, 3, 3]` | `0.955` `[0.975, 0.955, 0.925]` | `1512.02` `[1509.5, 1512.0, 1513.0]` | `5/0/195`, `8/1/191`, `11/4/185` | `[0.06250, 0.03906, 0.11847]` |

---

## 4. Verdict & Next Step

- **Verdict**: **`HARMVETO_NOT_ADVANCED`**.
  - On Part A (`DL-Train` 4-fold inner CV), the two candidate arms split the joint gate along the fundamental abstention-vs-gain Pareto frontier:
    1. `CLIP+HARMVETO` passes the gain-retention criterion (`+0.072198 >= +0.061148`, retaining `82.65%` of `CLIP` median gain), passes `LCB95 > 0` on `3/3` seeds, and passes `raw -2 <= slot 0` (`[15, 14, 13] <= 17`), while cutting harmed families from `26.0` to `16.0` (which remains above `ENS_VOTE3`'s `7.0`).
    2. `ENS_VOTE2+HARMVETO` (`best_by_prereg_arm`) passes the harmed-family criterion (`6.0 <= 7.0` `ENS_VOTE3`) and `LCB95 > 0` on `3/3` seeds, and beats `ENS_VOTE3` on median gain by `+72.0%` (`+0.037399` vs `+0.021749`) and MM BT Elo by `+28.14` (`1581.65` vs `1553.51`), while beating both `PLACEBO` (`+0.018858`, harmed `8.0`) and `NULL_LF` (`+0.028161`, harmed `13.0`), but retains `42.81% < 70%` of single-seed `CLIP`'s median gain and has `18 > 17` raw `-2` on seed `291101` (`[18, 14, 15]`).
- **Next Step**:
  - Do not adopt `HARMVETO` directly. On `H286` v2 `200` EVEN (`SEMI_FRESH`), 2-of-3 member agreement (`CLIP_VOTE2` / `ENS_VOTE2`) plus the harm-veto gate (`CLIP_VOTE2+HARMVETO`) eliminates harmed families (`[0, 0, 2]`, median `0.0` out of `200` families!) with `3/3` seeds `LCB95 > 0`, showing that the harm classifier genuinely isolates losing switches on unseen families, but at a `~94–95%` abstention rate. Closing the `70%` gain-retention gap without relaxing the `<= 7` harmed-family cap requires training the base ranker directly with an asymmetric pairwise hinge/focal penalty on slot-0 losses rather than post-hoc abstention on top of an already-fitted proposer.
