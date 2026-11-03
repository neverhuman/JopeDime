# H275: Clean-Room `K = 8` Soft Invalid-Risk Reranker (`h275_rerank8_soft_risk`)

- **Wave:** `H275` (`h275-rerank8-soft-risk-20261103`)
- **Owner:** Argon (`argon-t2030`)
- **Pre-Registration Commit (`outcomes_read = 0`):** `b19fba9d09683510469abb3099056a3f0763dca1` (following `1d9f41a90efe1100c38352bd93c34d12586f6034`, pushed at `2026-11-03T20:48Z` before the `21:05Z` gate)
- **Pre-Registered Outcome-Free Fold Manifest (S3):** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H275/prereg/fold_manifest.json` (`VersionId = SuUQLfyvQvJ_u07AyESxvQIOTR6wYPkG`, `sha256 = 29ef28bc20030e2c52b2a4ed28c88d82d6dc42a0895c3075a81bb72e509fd044`, `outcomes_read = 0`)
- **Full Output Archive (S3, uploaded prior to Git closeout):** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H275/attempt-001/` (`metrics_full.json` `VersionId = FKmgbzwrZteuC_2y8YXyXRf7g4I7wmf6`, `details.json` `VersionId = vSLAvELYrZhZCcodKCP4tdojx14UxWNv`, `fit_selections_lock.json` `VersionId = DBjS9tErDa3TY6ucEesFscHoodYqzEPF`, `panels.npz` `VersionId = 6.VmxLCEEonKGVgleNG1MtJionutbgnU`)
- **Status:** `COMPLETED`
- **Verdict:** **`NO_ADVANCE`** (`winning_arm = null` — **7 of 8** pre-declared `ADVANCE` gates passed; missed **only** Gate 6 `harmed_family_median_le_veto_plus_10` with `35` harmed families `> 0.01` vs `<= 21` required)

---

## 1. Hypothesis & Experimental Setup

### Hypothesis
In `H273` (`29fdbd8be15e`), the clean-room `K = 8` (`568`-column) outcome-blind reranker `RERANK8_CR` achieved `+0.0684` median family gain on `DL-Train` (`205` families, `1,219` contexts) with `34` harmed families (`> 0.01`), whereas the hard `tau = 0.10` invalid-gate veto `RERANK8_CR_VETO` reduced harmed families to `11` (`-67.6%`) and worst-family median loss to `-0.2643` (`-58.8%`) at the cost of forfeiting roughly half of the utility gain (`RERANK8_CR_VETO - RERANK8_CR = -0.0369` on `DL-Train`, `-0.0226` on `DL-Val91`).

`H275` tests whether replacing the hard `tau = 0.10` binary veto on candidate slots `k >= 1` with a **continuous soft invalid-risk penalty** in standardized ranker-score units:

$$\text{score}'_k = z(\text{score}_k) - \lambda \cdot p_{\text{invalid}, k} \quad \text{for } k \in \{1, \dots, 7\}, \qquad \text{score}'_0 = z(\text{score}_0) \quad (\text{slot } 0 \text{ untouched})$$

where $z(\text{score}_k) = \frac{\text{score}_k - \mu_{\text{query}}}{\max(\sigma_{\text{query}}, 10^{-6})}$ is the per-query standardized ranker score across the $K = 8$ slots and $(\lambda, \text{margin}) \in \{0.0, 0.5, 1.0, 2.0, 4.0, 8.0\} \times \{0.0, 0.02, 0.05, 0.10, 0.15, 0.20\}$ is selected **exclusively inside inner `FIT`** by maximizing inner-`FIT` mean family utility gain vs `CONTROL` subject to `fit_invalid <= fit_ctrl_invalid` (followed by strict inner-`FIT` fallback to `CONTROL`), recovers `RERANK8_CR`'s utility while preserving `RERANK8_CR_VETO`'s harm reduction.

### Clean-Room & Protocol Discipline
- **Zero-update, read-only `r882` copy (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`)**: single-pick (`B = 1`), `0` test-time simulator reveals, `0` outcome features in the `568`-column candidate representation (`468` pool-derived + `100` context-broadcasted `lm_48` / `dope_629` features).
- **Arms Evaluated**:
  1. `CONTROL`: frozen `r882` top-1 pick (`slot 0`).
  2. `RERANK8_CR`: reference clean-room `K = 8` (`568` cols) reranker with inner-`FIT` margin selection and strict inner-`FIT` fallback.
  3. `RERANK8_CR_VETO`: reference hard `tau = 0.10` invalid-gate veto (`p_invalid_k >= 0.10` masked to $-\infty$ for $k \ge 1$, slot `0` untouched) + inner-`FIT` margin + strict fallback.
  4. **`RERANK8_CR_SOFT` (NEW confirmatory arm)**: soft invalid-risk penalty $\text{score}'_k = z(\text{score}_k) - \lambda \cdot p_{\text{invalid}, k}$ for $k \ge 1$ (`slot 0` untouched, $\text{score}'_0 = z(\text{score}_0)$), with $(\lambda, \text{margin})$ selected **only inside inner `FIT`** subject to `fit_invalid <= fit_ctrl_invalid`, plus strict inner-`FIT` fallback.
  5. **`RERANK8_CR_SOFT_SHUF_UNC` (Placebo)**: within-family relevance + invalidity label shuffle across the `8` slots in `TRAIN` (`seed + 7777`), evaluated **without** inner-`FIT` fallback to verify that the soft-risk scoring rule does not manufacture spurious gains from label noise.
  6. `RERANK8_CR_SOFT_SHUF`: within-family label shuffle with strict inner-`FIT` fallback enabled.
  7. Reference ceilings: `TOP4_PICK_ORACLE`, `TOP8_PICK_ORACLE`, and `POOL_ORACLE`.
- **Fresh Seed Schedule**:
  - `fold_seeds = (275140, 275141, 275142)` $\times$ `5` outer family-grouped folds $\times$ `model_seeds = (275101, 275102, 275103)` (`45` (`fold_seed`, `fold`, `model_seed`) cells/arm on `PRIMARY` `DL-Train`; `15` cells/arm on `SECONDARY` `DL-Val91` at `fold_seed = 275140`; `3` model-seed cells/arm on `TERTIARY` `Cluster41`).
  - Paired family-cluster bootstrap: `10,000` draws at `boot_seed = 275150`.
  - Multiplicity correction: **Holm-Bonferroni $m = 3$** (`98.33%` two-sided CI / `99.17%` one-sided LCB) across `{PRIMARY: RERANK8_CR_SOFT - CONTROL (DL-Train), SECONDARY: RERANK8_CR_SOFT - CONTROL (DL-Val91), PRIMARY: RERANK8_CR_SOFT - RERANK8_CR_VETO (DL-Train)}`.

---

## 2. Outcome & Pre-Declared `ADVANCE` Gate Evaluation

**Verdict: `NO_ADVANCE` (`winning_arm = null`)**

`RERANK8_CR_SOFT` passed **7 of the 8** pre-declared `ADVANCE` gates — including beating both `CONTROL` (`+0.071922`, `Holm m=3 p = 0.0003`) and `RERANK8_CR_VETO` (`+0.038187`, `Holm m=3 p = 0.0023`) on `DL-Train`, achieving the **lowest raw `-2.0` invalid-pick count of any learned arm** on `DL-Train` (`124` across `45` cells vs `153` `CONTROL`, `135` `RERANK8_CR`, `142` `RERANK8_CR_VETO`) and `0` invalid picks on `3/3` seeds on `DL-Val91`, and achieving `+0.051829` (`95%` CI `[+0.003939, +0.100827]`) on the `H272` `41`-family least-similar out-of-cluster holdout (`Cluster41`). However, it missed **Gate 6** (`harmed_family_median <= VETO median + 10`), recording a median of `35` harmed families (`> 0.01` loss vs `CONTROL`) across seeds (`[35, 34, 44]`) compared to `RERANK8_CR_VETO`'s `11` (`11 + 10 = 21` ceiling) and `RERANK8_CR`'s `34` (`[31, 38, 34]`).

| Gate # | Pre-Declared `ADVANCE` Criterion (`PREREG.md`) | Observed (`attempt-001`) | Pass? |
| :--- | :--- | :--- | :---: |
| **Gate 1** | `PRIMARY` (`DL-Train`) `RERANK8_CR_SOFT - CONTROL` median family gain `> +0.020` | **`+0.071922`** (`> +0.020`) | **PASS** |
| **Gate 2** | `Holm m = 3` LCB `> 0` (`98.33%` CI lower bound `> 0`, `Holm p < 0.05`) | **`98.33%` CI `[+0.025711, +0.143420]`**, `Holm p = 0.0003` | **PASS** |
| **Gate 3** | `3/3` model seeds positive AND `>= 7/9` (`fold_seed × model_seed`) reps positive | **`3/3` seeds** (`+0.0748, +0.0679, +0.0719`), **`9/9` reps positive** | **PASS** |
| **Gate 4** | `DL-Train` raw `-2.0` invalid picks across `45` cells `<= CONTROL` (`153`) | **`124 <= 153`** (`-29` vs `CONTROL`, `-11` vs `RERANK8_CR`, `-18` vs `VETO`) | **PASS** |
| **Gate 5** | `DL-Val91` raw `-2.0` invalid picks `<= CONTROL` (`0`) on `3/3` seeds | **`[0, 0, 0] <= [0, 0, 0]`** (`0` total across `15` cells) | **PASS** |
| **Gate 6** | `DL-Train` harmed-family median (`> 0.01` loss vs `CONTROL`) `<= VETO median + 10` | **`35` (`[35, 34, 44]`) `> 21` (`VETO = 11` `[11, 10, 13]` `+ 10`)** | **FAIL** |
| **Gate 7** | Unconditional shuffle placebo `RERANK8_CR_SOFT_SHUF_UNC - CONTROL <= +0.005` | **`-0.033372 <= +0.005`** (`95%` CI `[-0.062761, +0.001607]`) | **PASS** |
| **Gate 8** | Non-fallback identity (`RERANK8_CR_SOFT != CONTROL`) | **`45/45` cells admitted, `7,146` changed picks across `45` cells** | **PASS** |

---

## 3. Key Metrics Tables (`PRIMARY`, `SECONDARY`, `TERTIARY`, `Holm m = 3`, & Elo)

### 3.1 Holm-Bonferroni $m = 3$ Confirmatory Summary

| Contrast | Panel | Median Family Gain | `95%` Bootstrap CI | `98.33%` (`Holm m=3`) CI | Raw One-Sided $p$ | `Holm m=3` Adjusted $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`RERANK8_CR_SOFT - CONTROL`** | `PRIMARY` (`DL-Train` `205` fam) | **`+0.071922`** | `[+0.032255, +0.127662]` | **`[+0.025711, +0.143420]`** | `0.000100` | **`0.000300`** |
| **`RERANK8_CR_SOFT - CONTROL`** | `SECONDARY` (`DL-Val91` `19` fam, *supporting*) | **`+0.042460`** | `[+0.024307, +0.063691]` | **`[+0.021232, +0.068282]`** | `0.000100` | **`0.000300`** |
| **`RERANK8_CR_SOFT - RERANK8_CR_VETO`** | `PRIMARY` (`DL-Train` `205` fam) | **`+0.038187`** | `[+0.011252, +0.075784]` | **`[+0.005919, +0.085351]`** | `0.002300` | **`0.002300`** |

### 3.2 `PRIMARY` Panel: `DL-Train` Out-of-Fold (`205` Families, `1,219` Contexts, `45` Cells/Arm)

| Arm / Contrast | Median Family Utility / Gain | `95%` CI | `98.33%` (`Holm m=3`) CI | Mean Family Utility / Gain | Median Context Utility / Gain | MM Bradley-Terry Elo (`r882=1500`) `[95% CI]` | Pos Seeds / Reps | Family W/T/L by Seed (`275101 / 275102 / 275103`) | Harmed Fams `> 0.01` Med `[Seeds]` | Worst Family Loss Med `[Seeds]` | Raw `-2.0` Invalid Picks (`45` cells `[Seeds]`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CONTROL` (`r882` slot `0`) | `1.385355` | — | — | `1.385355` | `1.423073` | `1500.00 [1500.00, 1500.00]` | — | — | — | — | `153 [51, 51, 51]` |
| **`RERANK8_CR_SOFT - CONTROL`** | **`+0.071922`** | **`[+0.032255, +0.127662]`** | **`[+0.025711, +0.143420]`** | `+0.072993` | `+0.069000` | **`1574.78 [1564.58, 1597.81]`** | `3/3` (`9/9`) | `150/6/49`, `151/7/47`, `144/5/56` | `35 [35, 34, 44]` | `-0.4930 [-0.7978, -1.5870, -0.4712]` | **`124 [38, 40, 46]`** |
| `RERANK8_CR - CONTROL` | `+0.067989` | `[+0.030729, +0.126797]` | `[+0.024300, +0.139406]` | `+0.072033` | `+0.071066` | `1571.06 [1562.06, 1592.16]` | `3/3` (`9/9`) | `147/16/42`, `142/9/54`, `148/10/47` | `34 [31, 38, 34]` | `-0.6408 [-0.6408, -1.5870, -0.6286]` | `135 [45, 41, 49]` |
| `RERANK8_CR_VETO - CONTROL` | `+0.032444` | `[+0.017540, +0.058047]` | `[+0.014864, +0.064748]` | `+0.032312` | `+0.030476` | `1528.35 [1518.34, 1532.94]` | `3/3` (`9/9`) | `101/83/21`, `78/112/15`, `92/93/20` | **`11 [11, 10, 13]`** | **`-0.2697 [-0.2697, -0.2389, -0.2981]`** | `142 [48, 49, 45]` |
| **`RERANK8_CR_SOFT - RERANK8_CR_VETO`** | **`+0.038187`** | **`[+0.011252, +0.075784]`** | **`[+0.005919, +0.085351]`** | `+0.038822` | `+0.040456` | — | `3/3` (`9/9`) | `125/15/65`, `137/17/51`, `129/8/68` | `40 [40, 34, 46]` | `-0.4930 [-0.7978, -0.6450, -0.4088]` | — |
| **`RERANK8_CR_SOFT - RERANK8_CR`** | **`+0.002010`** | `[-0.008591, +0.018847]` | `[-0.011367, +0.022913]` | `+0.001754` | `-0.000133` | — | `2/3` (`5/9`) | `78/45/82`, `72/83/50`, `50/108/47` | `23 [36, 22, 23]` | `-0.0569 [-0.3773, -0.6852, -0.1693]` | — |
| `RERANK8_CR_SOFT_SHUF - CONTROL` | `+0.000000` | `[+0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `+0.000000` | `+0.000000` | `1500.00 [1500.00, 1500.00]` | `0/3` (`0/9`) | `0/205/0`, `0/205/0`, `0/205/0` | `0 [0, 0, 0]` | `0.0000 [0.0, 0.0, 0.0]` | `153 [51, 51, 51]` |
| `RERANK8_CR_SOFT_SHUF_UNC - CONTROL` | **`-0.033372`** | `[-0.062761, +0.001607]` | `[-0.070739, +0.010876]` | `-0.032350` | `-0.035873` | `1493.36 [1479.18, 1500.00]` | `0/3` (`0/9`) | `81/32/92`, `66/42/97`, `76/17/112` | `68 [66, 68, 82]` | `-1.0346 [-2.1644, -1.1485, -1.0788]` | `204 [70, 60, 74]` |
| `RERANK8_CR_SOFT - SOFT_SHUF_UNC` | `+0.097655` | `[+0.072074, +0.137622]` | `[+0.066244, +0.146954]` | `+0.101185` | `+0.104873` | — | `3/3` (`9/9`) | `159/3/43`, `157/2/46`, `153/2/50` | `35 [30, 35, 39]` | `-0.5142 [-0.9158, -0.6986, -1.1136]` | — |
| `TOP4_PICK_ORACLE` | `1.578716` (`+0.193361`) | `[+0.138895, +0.268687]` | `[+0.132290, +0.290288]` | `+0.193361` | `+0.190992` | `1783.83 [1777.53, 1811.29]` | `3/3` | `198/7/0` (all seeds) | `0` | `0.0000` | `0` |
| `TOP8_PICK_ORACLE` | `1.636035` (`+0.250679`) | `[+0.189965, +0.329024]` | `[+0.179415, +0.349757]` | `+0.250679` | `+0.252936` | `1957.74 [1948.46, 1992.90]` | `3/3` | `203/2/0` (all seeds) | `0` | `0.0000` | `0` |
| `POOL_ORACLE` | `1.852543` (`+0.467187`) | `[+0.356725, +0.592362]` | `[+0.337322, +0.622470]` | `+0.467187` | `+0.482779` | `2626.73 [2594.17, 2680.22]` | `3/3` | `205/0/0` (all seeds) | `0` | `0.0000` | `0` |

### 3.3 `SECONDARY` Panel: `DL-Val91` (`19` Families, `91` Contexts, `fold_seed = 275140`, `15` Cells/Arm)
> **Reuse Disclosure:** `DL-Val91` (`91` contexts, `19` families) has been heavily reused across prior waves (`H249–H273`) for ranker architecture, feature-set, and veto-threshold exploration. It is reported strictly as **supporting evidence only**; all advancement decisions are governed by `PRIMARY` (`DL-Train` OOF) and the pre-declared `ADVANCE` gates.

| Arm / Contrast | Median Family Utility / Gain | `95%` CI | `98.33%` (`Holm m=3`) CI | Mean Family Gain | Median Context Gain | MM Bradley-Terry Elo (`r882=1500`) `[95% CI]` | Pos Seeds | Family W/T/L by Seed (`275101 / 275102 / 275103`) | Harmed Fams `> 0.01` Med `[Seeds]` | Worst Family Loss Med `[Seeds]` | Raw `-2.0` Invalid Picks (`15` cells `[Seeds]`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CONTROL` (`r882` slot `0`) | `1.153615` | — | — | — | `1.137794` | `1500.00 [1500.00, 1500.00]` | — | — | — | — | `0 [0, 0, 0]` |
| **`RERANK8_CR_SOFT - CONTROL`** | **`+0.042460`** | **`[+0.024307, +0.063691]`** | **`[+0.021232, +0.068282]`** | `+0.041072` | `+0.044224` | **`1579.09 [1558.03, 1626.22]`** | `3/3` | `16/1/2`, `16/0/3`, `18/0/1` | **`0 [0, 2, 0]`** | `-0.0068 [-0.0015, -0.0780, -0.0068]` | **`0 [0, 0, 0]`** |
| `RERANK8_CR - CONTROL` | `+0.041973` | `[+0.022954, +0.060436]` | `[+0.019645, +0.065177]` | `+0.039384` | `+0.043822` | `1578.37 [1543.44, 1618.01]` | `3/3` | `16/0/3`, `18/0/1`, `18/0/1` | `0 [1, 0, 0]` | `-0.0032 [-0.0270, -0.0032, -0.0075]` | `0 [0, 0, 0]` |
| `RERANK8_CR_VETO - CONTROL` | `+0.013526` | `[+0.006559, +0.023381]` | `[+0.005356, +0.025586]` | `+0.014966` | `+0.014120` | `1528.86 [1505.26, 1542.36]` | `3/3` | `15/4/0`, `11/6/2`, `14/3/2` | `0 [0, 0, 0]` | `0.0000 [-0.0001, -0.0027, -0.0018]` | `0 [0, 0, 0]` |
| **`RERANK8_CR_SOFT - RERANK8_CR_VETO`** | **`+0.026636`** | **`[+0.010253, +0.049887]`** | **`[+0.007933, +0.055844]`** | `+0.026106` | `+0.027811` | — | `3/3` | `16/1/2`, `14/0/5`, `15/1/3` | `1 [1, 4, 0]` | `-0.0051 [-0.0202, -0.0780, -0.0051]` | — |
| **`RERANK8_CR_SOFT - RERANK8_CR`** | **`+0.002877`** | `[-0.007092, +0.013305]` | `[-0.009500, +0.016166]` | `+0.001688` | `+0.002980` | — | `2/3` | `11/1/7`, `13/0/6`, `12/0/7` | `4 [6, 4, 3]` | `-0.0479 [-0.0493, -0.0748, -0.0479]` | — |
| `RERANK8_CR_SOFT_SHUF - CONTROL` | `+0.000000` | `[+0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `+0.000000` | `+0.000000` | `1500.00 [1500.00, 1500.00]` | `0/3` | `0/19/0` (all seeds) | `0 [0, 0, 0]` | `0.0000` | `0 [0, 0, 0]` |
| `RERANK8_CR_SOFT_SHUF_UNC - CONTROL` | `-0.037029` | `[-0.072336, -0.007961]` | `[-0.081731, -0.003209]` | `-0.050967` | `-0.038631` | `1503.09 [1476.53, 1528.40]` | `0/3` | `6/0/13`, `8/0/11`, `9/0/10` | `9 [9, 8, 10]` | `-0.2115 [-0.2609, -0.1748, -0.7811]` | `8 [2, 1, 5]` |
| `TOP4_PICK_ORACLE` | `1.367956` (`+0.214341`) | `[+0.130306, +0.305419]` | `[+0.114912, +0.320496]` | `+0.214341` | `+0.207269` | `1816.82 [1788.76, 1895.04]` | `3/3` | `19/0/0` (all seeds) | `0` | `0.0000` | `0` |
| `TOP8_PICK_ORACLE` | `1.394816` (`+0.241201`) | `[+0.155387, +0.330992]` | `[+0.141142, +0.348486]` | `+0.241201` | `+0.236752` | `1956.05 [1925.07, 2054.82]` | `3/3` | `19/0/0` (all seeds) | `0` | `0.0000` | `0` |
| `POOL_ORACLE` | `1.404600` (`+0.250985`) | `[+0.163660, +0.340665]` | `[+0.149495, +0.359699]` | `+0.250985` | `+0.261222` | `2457.59 [2361.11, 2634.74]` | `3/3` | `19/0/0` (all seeds) | `0` | `0.0000` | `0` |

### 3.4 `TERTIARY` Panel: Leave-Family-Cluster-Out Holdout (`Cluster41`, Reusing `H272`'s `41` Least-Similar Families, `201` Contexts)
Reusing the exact `41`-family least-similar out-of-cluster split from `H272` (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H272/prereg/fold_manifest.json`, `VersionId = 6NAoV3LWmW0.ZY5lbGs8fawZ_hUE47bP`), where the remaining `164` `DL-Train` families are split `131` `TRAIN` / `33` inner `FIT` (`fold_seed = 275140`, `3` model seeds):

| Arm / Contrast | Median Family Utility / Gain | `95%` CI | `98.33%` (`Holm m=3`) CI | Mean Family Gain | Median Context Gain | MM Bradley-Terry Elo (`r882=1500`) `[95% CI]` | Pos Seeds | Raw One-Sided $p$ | Family W/T/L by Seed (`275101 / 275102 / 275103`) | Harmed Fams `> 0.01` Med `[Seeds]` | Worst Family Loss Med `[Seeds]` | Raw `-2.0` Invalid Picks (`3` cells `[Seeds]`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CONTROL` (`r882` slot `0`) | `0.976447` | — | — | — | `1.045127` | `1500.00 [1500.00, 1500.00]` | — | — | — | — | — | `9 [3, 3, 3]` |
| **`RERANK8_CR_SOFT - CONTROL`** | **`+0.051829`** | **`[+0.003939, +0.100827]`** | `[-0.005604, +0.113095]` | `+0.039955` | **`+0.065185`** | **`1571.58 [1533.18, 1604.96]`** | **`3/3`** | **`0.017098`** | `33/3/5`, `30/2/9`, `25/1/15` | `8 [4, 8, 14]` | `-0.3111 [-0.1846, -0.3111, -0.8719]` | `10 [2, 3, 5]` |
| `RERANK8_CR - CONTROL` | `+0.046324` | `[-0.004967, +0.092178]` | `[-0.016417, +0.107071]` | `+0.038690` | `+0.049637` | `1555.66 [1527.48, 1600.72]` | `3/3` | `0.036996` | `30/2/9`, `28/1/12`, `26/7/8` | `8 [8, 8, 7]` | `-0.2607 [-1.0277, -0.6238, -0.1846]` | `9 [3, 3, 3]` |
| `RERANK8_CR_VETO - CONTROL` | `+0.018878` | `[+0.009788, +0.032721]` | `[+0.007909, +0.035651]` | `+0.021104` | `+0.022021` | `1525.37 [1504.05, 1541.65]` | `3/3` | `0.000200` | `18/19/4`, `22/14/5`, `19/16/6` | **`3 [1, 3, 4]`** | **`-0.0520 [-0.0502, -0.0697, -0.0961]`** | `9 [3, 3, 3]` |
| **`RERANK8_CR_SOFT - RERANK8_CR_VETO`** | **`+0.025203`** | `[-0.018882, +0.074052]` | `[-0.026688, +0.084656]` | `+0.018852` | `+0.032601` | — | `2/3` | `0.134887` | `26/4/11`, `24/3/14`, `24/1/16` | `9 [3, 9, 14]` | `-0.3529 [-0.1846, -0.3529, -0.8719]` | — |
| **`RERANK8_CR_SOFT - RERANK8_CR`** | **`+0.004269`** | `[-0.022662, +0.033892]` | `[-0.029497, +0.040244]` | `+0.001265` | `+0.015548` | — | `2/3` | `0.322468` | `20/2/19`, `17/12/12`, `21/2/18` | `11 [8, 11, 14]` | `-0.4715 [-0.4715, -0.5130, -0.8524]` | — |
| `RERANK8_CR_SOFT_SHUF - CONTROL` | `+0.000000` | `[+0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `+0.000000` | `+0.000000` | `1500.00 [1500.00, 1500.00]` | `0/3` | `1.000000` | `0/41/0` (all seeds) | `0` | `0.0000` | `9 [3, 3, 3]` |
| `RERANK8_CR_SOFT_SHUF_UNC - CONTROL` | `-0.018506` | `[-0.049801, +0.003231]` | `[-0.056551, +0.007236]` | `-0.039981` | `-0.016961` | `1501.33 [1483.93, 1522.43]` | `1/3` | `0.948905` | `18/8/15`, `12/11/18`, `14/6/21` | `12 [12, 12, 18]` | `-0.2839 [-0.4331, -0.3761, -0.9478]` | `16 [5, 4, 7]` |
| `TOP4_PICK_ORACLE` | `1.143327` (`+0.166880`) | `[+0.111273, +0.236996]` | `[+0.100353, +0.251226]` | `+0.166880` | `+0.162763` | `1768.57 [1746.57, 1818.34]` | `3/3` | `0.000100` | `40/1/0` (all seeds) | `0` | `0.0000` | `0` |
| `TOP8_PICK_ORACLE` | `1.179358` (`+0.202912`) | `[+0.145368, +0.275083]` | `[+0.132752, +0.292136]` | `+0.202912` | `+0.198670` | `1934.69 [1911.04, 2000.84]` | `3/3` | `0.000100` | `41/0/0` (all seeds) | `0` | `0.0000` | `0` |
| `POOL_ORACLE` | `1.313776` (`+0.337329`) | `[+0.251153, +0.436835]` | `[+0.234565, +0.461487]` | `+0.337329` | `+0.323130` | `2618.18 [2498.89, 2729.39]` | `3/3` | `0.000100` | `41/0/0` (all seeds) | `0` | `0.0000` | `0` |

---

## 4. Key Harm Readout & Mechanistic Diagnosis

### 4.1 Why `RERANK8_CR_SOFT` Cut Raw `-2.0` Invalid Picks to `124` (Below `CONTROL` `153`, `RERANK8_CR` `135`, and `RERANK8_CR_VETO` `142`) While Missing Gate 6 (`35` Harmed Families vs `<= 21`)
1. **Raw `-2.0` Invalid-Pick Reduction (`124` vs `153` `CONTROL`, `135` `RERANK8_CR`, `142` `RERANK8_CR_VETO`)**:
   - Across the `45` `DL-Train` cells (`3` fold seeds $\times$ `3` model seeds $\times$ `1,219` contexts = `10,971` context evaluations), `CONTROL` (`slot 0`) incurs **`153` raw `-2.0` invalid picks** (`51` per model seed).
   - Because `RERANK8_CR_VETO` applies a hard `p_invalid_k >= 0.10` mask to $k \ge 1$ while leaving `slot 0` untouched, it abstains back to `slot 0` on most contexts (`only 2,105` changed picks across `45` cells), thereby **inheriting `slot 0`'s invalid picks** and finishing with `142` raw `-2.0` invalid picks (`[48, 49, 45]`).
   - By contrast, `RERANK8_CR_SOFT` changes `7,146` picks across `45` cells (`64.9%` into deep slots `4..7`) while subtracting $\lambda \cdot p_{\text{invalid}, k}$ from $z(\text{score}_k)$ on $k \ge 1$. This allows `RERANK8_CR_SOFT` to escape `slot 0` invalid picks into low-$p_{\text{invalid}}$ $k \ge 1$ candidates, achieving the **lowest raw `-2.0` invalid count of any learned arm (`124` `[38, 40, 46]`, `-19.0%` vs `CONTROL`)** and reducing worst-family median loss on `DL-Train` from `-0.6408` (`RERANK8_CR`) to **`-0.4930`** (`-23.1%` smaller worst loss).
2. **Why Harmed-Family Count Remained Near `RERANK8_CR` (`35` vs `34`, Exceeding `VETO + 10 = 21`)**:
   - In `RERANK8_CR_SOFT`, standardizing per-query ranker scores $z(\text{score}_k)$ to unit variance expands the score differences $\text{score}'_k - \text{score}'_0$ relative to the raw margin grid $\{0.0, 0.02, 0.05, 0.10, 0.15, 0.20\}$ whenever $p_{\text{invalid}, k}$ is small. Consequently, `RERANK8_CR_SOFT` changes `7,146` picks (`14.3%` more than `RERANK8_CR`'s `6,252` and `3.39×` `RERANK8_CR_VETO`'s `2,105`), leaving only `5–7` of `205` families as ties with `CONTROL` (vs `83–112` ties under `RERANK8_CR_VETO`).
   - Most of the `35` harmed families on `DL-Train` come from **valid but lower-utility** $k \ge 1$ picks (ordinary regression ranker errors where $p_{\text{invalid}, k} \approx 0$), which a penalty proportional only to $p_{\text{invalid}, k}$ does not suppress.
3. **Inner-`FIT` Selected $\lambda$ Distribution Across `45` Cells**:
   - $\lambda = 0.0$: `1/45` (`2.2%`)
   - $\lambda = 0.5$: `4/45` (`8.9%`)
   - $\lambda = 1.0$: `5/45` (`11.1%`)
   - $\lambda = 2.0$: `9/45` (`20.0%`)
   - $\lambda = 4.0$: `14/45` (`31.1%`)
   - $\lambda = 8.0$: `12/45` (`26.7%`)
   - **`35/45` cells (`77.8%`) selected $\lambda \ge 2.0$**, and `26/45` (`57.8%`) selected $\lambda \in \{4.0, 8.0\}$, confirming that inner `FIT` strongly favors substantial continuous invalid-risk penalization.

### 4.2 Recurring Family Readouts (`fam-769243ee`, `fam-d73e184e`, `fam-ccf8d821`)

| Recurring Family | Panel (`n_ctx`) | `CONTROL` Utility | `RERANK8_CR` Median $\Delta$ `[Seeds]` | `RERANK8_CR_VETO` Median $\Delta$ `[Seeds]` | `RERANK8_CR_SOFT` Median $\Delta$ `[Seeds]` | `SOFT_SHUF_UNC` Median $\Delta$ `[Seeds]` | `TOP8_PICK_ORACLE` $\Delta$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fam-769243ee`** | `DL-Val91` (`5`) | `3.996767` | `-0.003181 [-0.0028, -0.0032, -0.0075]` (`0/3` `>0.01`) | **`-0.000059 [-0.0001, 0.0000, -0.0018]` (`0/3` `>0.01`)** | `-0.006825 [-0.0002, -0.0780, -0.0068]` (`1/3` `>0.01`) | `-0.092794 [-0.0928, -0.0076, -0.2364]` (`2/3` `>0.01`) | `+0.007519` |
| **`fam-d73e184e`** | `DL-Val91` (`5`) | `0.290475` | `+0.019792 [+0.0198, +0.0069, +0.0258]` (`0/3` `>0.01`) | `0.000000 [+0.0007, 0.0000, -0.0007]` (`0/3` `>0.01`) | **`+0.030451 [+0.0223, +0.0305, +0.0324]` (`0/3` `>0.01`)** | `-0.094341 [-0.0068, -0.0943, -0.1042]` (`2/3` `>0.01`) | `+0.071997` |
| **`fam-ccf8d821`** | `DL-Train` (`1`) | `7.883524` | `0.000000 [0.0000, -1.6975, +0.0536]` (`1/3` `>0.01`) | `0.000000 [0.0000, 0.0000, 0.0000]` (`0/3` `>0.01`) | **`+0.013612 [+0.0136, -1.6308, +0.1105]` (`1/3` `>0.01`)** | `-0.056649 [-1.7477, -0.0276, -0.0566]` (`3/3` `>0.01`) | `+2.089476` |

---

## 5. Recommended Next Step

Because `RERANK8_CR_SOFT` recovered **100% of the utility gap** vs `RERANK8_CR_VETO` (`+0.071922` vs `+0.032444` on `DL-Train`, `+0.051829` on `Cluster41`) and achieved the **lowest raw `-2.0` invalid pick count (`124`)**, while missing **only** Gate 6 (`35` harmed families vs `<= 21`) due to standardized $z(\text{score}_k)$ expanding the effective score scale relative to the raw margin grid `{0.0, 0.02, 0.05, 0.10, 0.15, 0.20}`:
- **Next hypothesis (`K = 8` z-margin + slot-0-relative soft risk)**:
  1. Penalize the **excess invalid risk relative to slot 0** ($\Delta p_{\text{invalid}, k} = \max(0, p_{\text{invalid}, k} - p_{\text{invalid}, 0})$) or include an expanded standardized margin grid $\text{margin}_z \in \{0.10, 0.25, 0.50, 0.75, 1.00, 1.50\}$ in $z$-units jointly with $\lambda \in \{1.0, 2.0, 4.0, 8.0\}$ and a family-harm-aware inner-`FIT` objective (`fit_harmed_families_gt_001 <= fit_harm_cap`) so that only high-confidence $z$-margin improvements override `slot 0`.

---

## 6. Mandatory Self Red-Team Audit

1. **Leakage (`PASS`)**:
   - Pre-registered fold manifest (`fold_manifest.json`, `VersionId = SuUQLfyvQvJ_u07AyESxvQIOTR6wYPkG`) was generated and pushed (`b19fba9d0968`) with `outcomes_read = 0` before any outcome arrays were read.
   - Feature matrix `X` (`568` columns) is strictly outcome-blind (verified by `test_feature_builder_568_cols_and_outcome_blindness`).
   - For each outer cell (`fold_seed`, `fold`, `model_seed`), both the ranker (`HistGradientBoostingRegressor`), the 3-fold OOF invalid-pick classifier (`HistGradientBoostingClassifier`), and the joint $(\lambda, \text{margin})$ grid search + fallback check run **strictly inside `TRAIN` + inner `FIT`**, with `0` outer-fold `VAL`, `DL-Val91`, or `Cluster41` outcomes touched during training or selection (`fit_selections_lock.json` `sha256 = 28cc46738d2973c4fb81ab83a070555f23c6ed6947778ccd3abd5ce9fbf81d4e` written and hashed before scoring any evaluation contexts).
2. **Seed Noise (`PASS`)**:
   - Evaluated across `3` fresh outer fold-split seeds (`275140–275142`) $\times$ `5` folds $\times$ `3` fresh model seeds (`275101–275103`) = `45` cells/arm (`10,971` context evaluations/arm).
   - `RERANK8_CR_SOFT - CONTROL` is positive on **`3/3` model seeds** (`+0.074848, +0.067921, +0.071922`) and **`9/9` (`fold_seed × model_seed`) repetitions** (`min = +0.065269, max = +0.080378`).
   - Unconditional label-shuffle placebo `RERANK8_CR_SOFT_SHUF_UNC - CONTROL` is negative on **`3/3` model seeds** (`-0.0170, -0.0334, -0.0468`, median `-0.033372`), and strict fallback (`RERANK8_CR_SOFT_SHUF`) abstains on **`45/45` cells** (`0` changed picks).
3. **`DL-Val91` Reuse (`PASS`)**:
   - Explicitly disclosed in `PREREG.md`, `metrics.json`, and Section 3.3 above that `DL-Val91` (`19` families, `91` contexts) has been heavily reused across prior waves and serves strictly as **supporting evidence only**.
4. **Cherry-Picking (`PASS`)**:
   - All pre-declared panels (`PRIMARY` `DL-Train`, `SECONDARY` `DL-Val91`, `TERTIARY` `Cluster41`), all `Holm m = 3` contrasts, all `8` pre-declared `ADVANCE` gates, and all recurring harm families (`fam-769243ee`, `fam-d73e184e`, `fam-ccf8d821`) are reported in full. Despite strong utility gains across all three panels (`+0.0719`, `+0.0425`, `+0.0518`) and `7/8` gates passing, the verdict is strictly recorded as **`NO_ADVANCE`** (`winning_arm = null`) because Gate 6 (`35 <= 21`) failed.
5. **Tie Acceptance (`PASS`)**:
   - Strict inequality `fit_fam_gain > 0.0` and `val_score - slot0_score > margin` enforced everywhere; `margin = inf` (tie/abstain) returns `slot 0` identically, so zero-change ties (`RERANK8_CR_SOFT_SHUF = 0.000000`) are rejected (`gate_8_non_fallback_identity` requires `changed_picks > 0`).
6. **Sealed Holdouts Untouched (`PASS`)**:
   - Only `dl_train_5f.npz`, `dl_val_19f.npz`, and `dope_features.npz` were opened (`SEALED` / `DL-Test` / `Meta-Test` untouched; `0` sealed files accessed).
7. **Frozen `DOPE` & `r882` Integrity (`PASS`)**:
   - `r882` sha256 verified (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`); `dope_features.npz` sha256 verified (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`); `91/91` bit-exact `DL-Val91` context utility matches (`1.1377941898164525`) and `1,219/1,219` `DL-Train` matches (`1.423073434552299` context mean, `1.3853554329931828` family mean).
8. **OOM / Disk / Resource Hygiene (`PASS`)**:
   - Ran under `monitor.py` on pinned CPUs `[57, 58, 59, 60]` (`peak_threads = 3 <= 4`), `peak_rss_bytes = 932,876,288` (`889.7 MiB <= 4 GiB`), `elapsed_seconds = 1146.04 s` (`19.1 min`), `pause_requested_seconds = 0.0`.
   - Full outputs uploaded to S3 first (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H275/attempt-001/`); Git `metrics.json` is `49,893` bytes (`<= 50,000` bytes), with zero `.pt`/`.npz` or `> 50 KB` JSON files tracked in Git.
