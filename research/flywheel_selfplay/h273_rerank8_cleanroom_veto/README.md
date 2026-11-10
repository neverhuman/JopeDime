# H273: Independent-Lane Clean-Room Replication of `RERANK8` (`K=8`, `568` Cols) + `H269` Invalid-Gate Veto (`RERANK8_CR_VETO`) & Unconditional Shuffle Placebo (`RERANK8_CR_SHUF_UNC`)

- **Wave ID**: `h273-rerank8-cleanroom-veto-20261103` (`H273`)
- **Owner**: `Argon` (independent lane from `H270` owner `Cobalt`)
- **Status**: `COMPLETED` (`2026-11-03T20:06:22Z`)
- **Preregistered SHA**: `81dc6738b2b113d8895425f3bd148176054077c3` (`outcomes_read = 0`)
- **Verdict**: **`REPLICATED`** (`winning_arm = "RERANK8_CR"`; both confirmatory `RERANK8_CR` and descriptive `RERANK8_CR_VETO` pass all `6/6` preregistered replication gates under `Holm-Bonferroni m = 2`)

---

## 1. Executive Summary & Hypothesis

`H270` (`h270_rerank_top8`, owner `Cobalt`) reported that expanding the pairwise-delta `LGBMRanker` candidate horizon from `K = 4` (`RERANK4`, `+0.0258` on `DL-Train`) to `K = 8` (`RERANK8`, `568` feature columns) more than doubled the `PRIMARY` `DL-Train` out-of-fold median family gain to `+0.0523 [+0.0171, +0.1060]` (`p = 0.0029`, `9/9` `(model_seed × fold_seed)` repetitions positive) and improved `SECONDARY` `DL-Val91` to `+0.0256 [+0.0096, +0.0513]`. Meanwhile, `H269` (`h269_nosg_veto_rerank4`) showed that while dropping SG features (`NOSG`) fails (`NO_ADVANCE`), a frozen out-of-fold `LGBMClassifier` invalid-gate veto (`p_invalid >= 0.10`) reduced `-2.0` invalid picks on `NOSG` from `161` to `143` (`< 153` baseline).

`H273` executes a **two-stage independent-lane clean-room replication and extension** written from the `H270` `PREREG.md` / `README.md` specification text ONLY (zero imports or code copies from `topk.py`, `h265_core.py`, or `h270/runner.py`):

1. **Stage 1 (`H270` Seed Reproduction Gate)**: Evaluates clean-room `RERANK8_CR` at `H270`'s exact seeds (`fold_seeds = 270140–270142`, `model_seeds = 270101–270103`, `boot_seed = 270150`) to measure per-cell pick agreement and `PRIMARY` median family gain reproduction against `H270`'s published artifact before fresh-seed evaluation.
2. **Stage 2 (`H273` Fresh-Seed Confirmatory Evaluation, `Holm m = 2`)**: Evaluates `RERANK8_CR` (confirmatory), `RERANK8_CR_VETO` (`H269` frozen `LGBMClassifier` invalid-gate veto at `tau = 0.10` applied to slots `k in {1..7}`), strict shuffle placebo `RERANK8_CR_SHUF`, and **unconditional shuffle placebo `RERANK8_CR_SHUF_UNC` (inner-`FIT` selector fallback disabled so the placebo has teeth)** across `3` fresh `fold_seeds` (`273140, 273141, 273142`) $\times$ `5` family-grouped folds $\times$ `3` fresh `model_seeds` (`273101, 273102, 273103`) = `45` fit cells/arm (`boot_seed = 273150`, `10,000` paired family-cluster bootstrap draws).

### Headline Findings

- **Stage 1 Reproduction Gate (`PASS`)**: At `H270`'s seeds (`270140–270142` $\times$ `270101–270103`), clean-room `RERANK8_CR` strictly admits `45/45` cells (`100%`, matching `H270`'s `45/45`) and achieves `PRIMARY` median family gain **`+0.054085 [+0.016302, +0.115280]`** (`raw_p = 0.0028`, per-seed `[+0.066836, +0.053059, +0.054085]`, **`9/9` repetitions positive**, `132 <= 153` invalid picks), reproducing `H270`'s published **`+0.052315 [+0.0171, +0.1060]`** within `0.0018`. Across `10,971` context-cell evaluations (`45` cells), clean-room `RERANK8_CR` agrees with `H270`'s published picks on **`71.48%` (`7,842 / 10,971`)** of binary `r882`-vs-switch decisions and **`54.06%` (`5,931 / 10,971`, `4.3×` the `12.5%` 8-slot uniform rate)** of exact slot indices despite independent random streams.
- **Stage 2 Confirmatory `RERANK8_CR` (`REPLICATED`, `6/6` gates passed)**: On fresh fold and model seeds, `RERANK8_CR - CONTROL` achieves `PRIMARY` `DL-Train` median family gain **`+0.057130`** (`per_seed = [+0.057130, +0.075119, +0.048951]`, **`3/3` model seeds positive** and **`9/9` repetitions positive**), `95%` CI **`[+0.019198, +0.116590]`**, `Holm m=2` (`97.5%`) CI **`[+0.014119, +0.126979]`**, `raw_p = 0.0013`, **`Holm m=2 adjusted p = 0.0013`**, with `43/45` cells admitted and **`144 <= 153` `-2.0` invalid picks**. Deep slots `4..7` account for **`65.8%` (`3,865 / 5,871`)** of all changed picks (`35.2%` of all `10,971` picks). On `SECONDARY` `DL-Val91` (`fseed = 273140`), `RERANK8_CR - CONTROL` achieves **`+0.036470 [+0.015136, +0.056771]`** (`Holm2 CI = [+0.012507, +0.060083]`, `Holm p = 0.0006`, `3/3` seeds positive; `3`-fold-seed aggregate **`+0.033369 [+0.017827, +0.052956]`**, `p = 0.0001`).
- **Stage 2 Descriptive `RERANK8_CR_VETO` (`tau = 0.10`, `6/6` gates passed vs `CONTROL`)**: `RERANK8_CR_VETO - CONTROL` achieves `PRIMARY` `DL-Train` median family gain **`+0.027807`** (`per_seed = [+0.024727, +0.032535, +0.027807]`, **`3/3` seeds positive** and **`9/9` repetitions positive**), `95%` CI **`[+0.010083, +0.052092]`**, `Holm m=2` CI **`[+0.007958, +0.056603]`**, `raw_p = 0.0001`, **`Holm m=2 adjusted p = 0.0002`**, with **`45/45` (`100%`) cells admitted** and **`148 <= 153` invalid picks**. While `RERANK8_CR_VETO` sacrifices roughly half of `RERANK8_CR`'s raw median gain (`RERANK8_CR_VETO - RERANK8_CR = -0.018415 [-0.048663, +0.004540]`) because high-upside deep-slot (`4..7`) candidates also carry higher predicted invalid probabilities, it **slashes `DL-Train` families harmed by `> 0.01` from `[40, 27, 34]` (median `33`, worst `-1.6865`) down to `[6, 11, 10]` (median `9`, worst `-0.4724`)**, and on `DL-Val91` (`fseed = 273140`, `+0.015792 [+0.007071, +0.026629]`, `Holm p = 0.0002`) **eliminates `100%` of `-2.0` invalid picks (`0 / 15` cells vs `2 / 15` on `RERANK8_CR`) and reduces `DL-Val91` families harmed `> 0.01` to `0 / 19` (`[0, 0, 0]`) across all `3` seeds**.
- **Unconditional Shuffle Placebo (`RERANK8_CR_SHUF_UNC`) Has Real Teeth**: When the inner-`FIT` selector fallback is active (`RERANK8_CR_SHUF`), only `2/45` cells are admitted (`median = 0.000000`), showing that the selector rejects shuffled-label noise. With fallback disabled (`RERANK8_CR_SHUF_UNC`, `45/45` cells active, `642` changed picks), shuffled-label rankers **degrade utility on both splits**: `DL-Train` median family gain **`-0.008564 [-0.014086, -0.000515]`** (`0/3` seeds positive, `164` invalid picks) and `DL-Val91` **`-0.009746 [-0.021863, -0.001053]`** (`0/3` seeds positive). Forcing `margin = 0.0` (`RERANK8_CR_SHUF_UNC_M0`, `9,208` changed picks) degrades `DL-Train` to **`-0.082706 [-0.135613, -0.014595]`** (`294` invalid picks). Consequently, `RERANK8_CR - RERANK8_CR_SHUF_UNC = +0.057515 [+0.030248, +0.120839]` (`p = 0.0001`), confirming that `RERANK8_CR`'s gain is driven by genuine candidate-ranking signal.

---

## 2. Clean-Room Architecture & Feature Contract (`K = 8`, `568` Columns)

`runner.py` was implemented from scratch using only standard scientific Python (`numpy`, `torch`, `lightgbm`) and the frozen `dope` backbone loader:
- **Zero code reuse from `H265`/`H266`/`H270`**: Verified by AST/source inspection in `test_h273.py::test_cleanroom_no_prohibited_imports_or_files` (`0` references to `topk`, `h265_core`, or `h270`).
- **`568`-Column Feature Matrix (`K = 8`)**:
  - `56` context summary features (`20` `ctx_x` + `20` `pool_mean_x` + `4` `pool_std_x` + `12` pool-score & `r882` entropy/margin/spread statistics),
  - `8 × 28 = 224` concatenated top-`8` candidate descriptors (`20` `cand_x` + `6` pool priors + `r882` logit + `r882` probability),
  - `8 × 28 = 224` candidate-minus-slot-`0` (`r882` pick) pairwise deltas (`cand_k - cand_0`),
  - `8 × 8 = 64` per-candidate rank/gap/z-score features (`rank_norm`, `prob`, `gap_to_slot0`, `gap_to_prev`, `gap_to_next`, `ratio_to_slot0`, `logit_z`, `prior_gap_to_slot0`).
  - All `568` columns are verified strictly outcome-blind via synthetic target-mutation invariance in `test_h273.py::test_feature_builder_568_cols_and_outcome_blindness`.
- **Base `LGBMRanker` & `H269` Invalid-Gate Veto (`RERANK8_CR_VETO`)**:
  - Base ranker: `LGBMRanker(objective="lambdarank", n_estimators=90, learning_rate=0.05, num_leaves=15, min_child_samples=25, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, n_jobs=1)` trained on `8`-row query groups with integer utility-rank relevance labels `0..7` (ties broken in favor of slot `0`).
  - Veto classifier (`RERANK8_CR_VETO`): `LGBMClassifier(objective="binary", n_estimators=70, learning_rate=0.05, num_leaves=15, min_child_samples=25, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, n_jobs=1)` trained on the `8 × N_train` candidate rows of the `3` `TRAIN` folds (`48`-dim per-candidate + context feature vector) to predict `y_invalid = 1[u_k <= -1.999]`. Any non-baseline candidate (`k in {1..7}`) with `p_invalid(x, k) >= tau` (`tau = 0.10` preregistered) has its ranker score masked to `-inf` on both `FIT` and `ASSESS`/`DL-Val91` before margin selection (`margin in {0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60}`) and inner-`FIT` strict fallback (`fit_fam_gain > 0` and `fit_invalid <= fit_ctrl_invalid`).

---

## 3. Baseline Regression & Stage 1 (`H270` Seed Reproduction Gate)

### 3.1 Baseline Regression Gate (`Native91` First)

| Check | Expected | Observed | Status |
| --- | --- | --- | --- |
| `Native91` (`DL-Val` `91` contexts, `19` families) mean utility | `1.1377941898164525` (`91/91` bit-exact) | `1.1377941898164525` (`max_abs_diff = 0.0`, `91/91` exact) | **`PASS`** |
| `DL-Train` (`1,219` contexts, `205` families) `r882` mean utility | `1.3266996728990814` (`1219/1219` exact) | `1.3266996728990814` (`max_abs_diff = 0.0`, `1219/1219` exact) | **`PASS`** |
| `Context-27` (`27` contexts, `11` families) `r882` mean utility | `0.7336108206469107` (`27/27` exact) | `0.7336108206469107` (`max_abs_diff = 0.0`, `27/27` exact) | **`PASS`** |

### 3.2 Stage 1: Clean-Room `RERANK8_CR` at `H270` Seeds (`270140–270142` $\times$ `270101–270103`, `boot_seed = 270150`)

| Metric | `H270` Published (`RERANK8`) | `H273` Stage 1 Clean-Room (`RERANK8_CR`) | Delta / Agreement |
| --- | --- | --- | --- |
| `PRIMARY` `DL-Train` median family gain | **`+0.052315`** | **`+0.054085`** | **`+0.001770`** (within `0.0018`) |
| `95%` family-cluster bootstrap CI | `[+0.017066, +0.106024]` | `[+0.016302, +0.115280]` | Overlapping `95%` CIs |
| One-sided bootstrap `raw_p` | `0.0029` | `0.0028` (`Holm m=2 CI = [+0.011202, +0.125747]`) | Matched significance |
| Per-model-seed median family gains (`270101..270103`) | `[+0.052315, +0.068080, +0.039815]` | `[+0.066836, +0.053059, +0.054085]` | **`3/3` model seeds positive** |
| Positive `(model_seed × fold_seed)` repetitions | **`9 / 9`** | **`9 / 9`** | **`100%` match** |
| Strict inner-`FIT` cells admitted | **`45 / 45` (`100.0%`)** | **`45 / 45` (`100.0%`)** | **`100%` match** |
| `DL-Train` `-2.0` invalid picks (`45` cells vs `153` `CONTROL`) | `150 <= 153` | **`132 <= 153`** (`[45, 42, 45]` by seed) | Both `<= 153` cap |
| Changed picks (`45` cells = `10,971` evals) | `4,734 / 10,971` (`43.15%`) | `5,836 / 10,971` (`53.20%`, `3,803` from slots `4..7`) | Both heavily use slots `4..7` (`~65%`) |
| Per-cell binary switch agreement (`pick == 0` vs `pick > 0`) | — | **`7,842 / 10,971` (`71.48%`)** | High switch concordance |
| Per-cell exact slot agreement (`0..7`) | — | **`5,931 / 10,971` (`54.06%`)** | **`4.32×`** 8-slot uniform (`12.5%`) |

---

## 4. Stage 2 (`H273` Fresh-Seed Evaluation: `fold_seeds = 273140–273142`, `model_seeds = 273101–273103`)

### 4.1 `PRIMARY` Metric: `DL-Train` OOF Family-Equal (`205` Families, `1,219` Contexts, `45` Cells/Arm, `boot_seed = 273150`)

| Contrast (`DL-Train` OOF `205` Families) | Median Family Gain | `95%` Bootstrap CI | `Holm m=2` (`97.5%`) CI | `raw_p` | `Holm m=2 p` | Per-Model-Seed Gains (`273101, 273102, 273103`) | Pos Seeds / `9` Reps | Admitted Cells | `-2.0` Invalid Picks (`45` cells) | Changed Picks (`Slots 4..7`) | Family W/T/L by Seed | Harmed Families `> 0.01` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`RERANK8_CR - CONTROL` (Confirmatory)** | **`+0.057130`** | **`[+0.019198, +0.116590]`** | **`[+0.014119, +0.126979]`** | **`0.0013`** | **`0.0013`** | `[+0.057130, +0.075119, +0.048951]` | **`3/3` (`9/9`)** | `43 / 45` (`95.6%`) | **`144 <= 153`** (`[50, 44, 50]`) | `5,871` (`3,865` = `65.8%`) | `143/10/52`, `155/11/39`, `131/34/40` | `[40, 27, 34]` (med `33`) |
| **`RERANK8_CR_VETO - CONTROL` (`tau = 0.10`)** | **`+0.027807`** | **`[+0.010083, +0.052092]`** | **`[+0.007958, +0.056603]`** | **`0.0001`** | **`0.0002`** | `[+0.024727, +0.032535, +0.027807]` | **`3/3` (`9/9`)** | **`45 / 45` (`100%`)** | **`148 <= 153`** (`[52, 46, 50]`) | `3,530` (`2,248` = `63.7%`) | `126/43/36`, `130/39/36`, `127/44/34` | **`[6, 11, 10]` (med `9`)** |
| `RERANK8_CR_VETO - RERANK8_CR` (Veto vs Base) | `-0.018415` | `[-0.048663, +0.004540]` | `[-0.055436, +0.007922]` | `0.9541` | `0.9541` | `[-0.018415, -0.043406, -0.013250]` | `0/3` (`0/9`) | — | `148` vs `144` | — | `55/19/131`, `55/17/133`, `56/36/113` | `[94, 77, 72]` |
| `RERANK8_CR_UNC - CONTROL` (No Fallback) | `+0.062564` | `[+0.019530, +0.128280]` | `[+0.013712, +0.137025]` | `0.0014` | — | `[+0.062564, +0.075119, +0.059882]` | `3/3` (`9/9`) | `45 / 45` (forced) | `146 <= 153` (`[51, 44, 51]`) | `6,127` (`4,033` = `65.8%`) | `145/6/54`, `155/11/39`, `145/9/51` | `[40, 27, 41]` |
| `RERANK8_CR_SHUF - CONTROL` (Strict Placebo) | `+0.000000` | `[+0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `1.0000` | — | `[+0.000000, -0.002552, +0.000000]` | `0/3` (`0/9`) | `2 / 45` (`4.4%`) | `153` (`[51, 51, 51]`) | `22` (`14` = `63.6%`) | `0/205/0`, `6/188/11`, `0/205/0` | `[0, 1, 0]` |
| **`RERANK8_CR_SHUF_UNC - CONTROL` (Uncond. Placebo)** | **`-0.008564`** | **`[-0.014086, -0.000515]`** | **`[-0.015660, +0.000000]`** | `0.9828` | — | `[-0.000119, -0.005245, -0.008564]` | **`0/3` (`1/9`)** | `45 / 45` (forced) | **`164 > 153`** (`[53, 56, 55]`) | `642` (`420` = `65.4%`) | `46/98/61`, `43/100/62`, `36/105/64` | `[9, 5, 8]` |
| **`RERANK8_CR_SHUF_UNC_M0 - CONTROL` (`margin = 0`)** | **`-0.082706`** | **`[-0.135613, -0.014595]`** | **`[-0.145551, -0.006694]`** | `0.9923` | — | `[-0.097253, -0.082706, -0.081110]` | **`0/3` (`0/9`)** | `45 / 45` (forced) | **`294 > 153`** (`[102, 92, 100]`) | `9,208` (`6,158` = `66.9%`) | `84/0/121`, `86/0/119`, `86/0/119` | `[111, 111, 109]` |
| `RERANK8_CR - RERANK8_CR_SHUF` | `+0.057130` | `[+0.019946, +0.116210]` | `[+0.014658, +0.126950]` | `0.0013` | — | `[+0.057130, +0.075119, +0.048951]` | `3/3` (`9/9`) | — | `144` vs `153` | — | `143/10/52`, `154/11/40`, `131/34/40` | `[40, 28, 34]` |
| **`RERANK8_CR - RERANK8_CR_SHUF_UNC`** | **`+0.057515`** | **`[+0.030248, +0.120839]`** | **`[+0.025279, +0.130602]`** | **`0.0001`** | — | `[+0.057130, +0.080744, +0.057515]` | **`3/3` (`9/9`)** | — | `144` vs `164` | — | `142/8/55`, `155/8/42`, `139/23/43` | `[41, 30, 32]` |
| **`RERANK8_CR_VETO - RERANK8_CR_SHUF_UNC`** | **`+0.034994`** | **`[+0.020574, +0.060455]`** | **`[+0.018508, +0.064234]`** | **`0.0001`** | — | `[+0.024727, +0.035568, +0.034994]` | **`3/3` (`9/9`)** | — | `148` vs `164` | — | `127/28/50`, `133/24/48`, `135/29/41` | `[8, 13, 8]` |
| `TOP4_PICK_ORACLE - CONTROL` (Diagnostic) | `+0.193361` | `[+0.139503, +0.267152]` | `[+0.130541, +0.278859]` | `0.0001` | — | `[+0.193361, +0.193361, +0.193361]` | `3/3` | — | `27` (`[9, 9, 9]`) | `4,797` (`0` slots `4..7`) | `197/8/0` | `[0, 0, 0]` |
| `TOP8_PICK_ORACLE - CONTROL` (Diagnostic) | `+0.250679` | `[+0.189796, +0.331142]` | `[+0.178223, +0.343537]` | `0.0001` | — | `[+0.250679, +0.250679, +0.250679]` | `3/3` | — | `18` (`[6, 6, 6]`) | `6,867` (`3,177` = `46.3%`) | `201/4/0` | `[0, 0, 0]` |
| `POOL_ORACLE - CONTROL` (Diagnostic) | `+0.467187` | `[+0.361262, +0.594715]` | `[+0.349093, +0.614308]` | `0.0001` | — | `[+0.467187, +0.467187, +0.467187]` | `3/3` | — | `0` (`[0, 0, 0]`) | `9,342` (`1,773` = `19.0%`) | `205/0/0` | `[0, 0, 0]` |

---

### 4.2 `SECONDARY` Metric: `DL-Val91` Family-Equal (`19` Families, `91` Contexts)

Per preregistration, `DL-Val91` is evaluated by fold-averaging the `5` fold models of `fold_seed = 273140` (`15` cell evaluations/arm) and also reported across all `3` fold seeds (`45` cell evaluations/arm). **Reuse Disclosure**: `DL-Val91` has been observed across prior waves (`H245–H272`) and is strictly a secondary guardrail, not a fresh holdout.

| Contrast (`DL-Val91`, `19` Families, `91` Contexts) | `fseed = 273140` Median Family Gain | `95%` Bootstrap CI | `Holm m=2` (`97.5%`) CI | `raw_p` (`Holm p`) | Per-Seed Gains (`273101..273103`) | `-2.0` Invalid Picks (`15` cells) | Harmed Families `> 0.01` (`19` fams) | `3`-Fold-Seed Aggregate Median (`95%` CI, `p`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`RERANK8_CR - CONTROL` (Confirmatory)** | **`+0.036470`** | **`[+0.015136, +0.056771]`** | **`[+0.012507, +0.060083]`** | **`0.0003` (`0.0006`)** | `[+0.036470, +0.029009, +0.037394]` (`3/3`) | `2` (`[1, 1, 0]` vs `[0, 0, 0]`) | `[1, 1, 0]` (med `1`) | **`+0.033369 [+0.017827, +0.052956]` (`p = 0.0001`)** |
| **`RERANK8_CR_VETO - CONTROL` (`tau = 0.10`)** | **`+0.015792`** | **`[+0.007071, +0.026629]`** | **`[+0.006192, +0.028522]`** | **`0.0001` (`0.0002`)** | `[+0.014889, +0.018870, +0.015792]` (`3/3`) | **`0` (`[0, 0, 0]` vs `[0, 0, 0]`)** | **`[0, 0, 0]` (med `0`)** | **`+0.013203 [+0.005767, +0.020897]` (`p = 0.0001`)** |
| `RERANK8_CR_VETO - RERANK8_CR` | `-0.018725` | `[-0.035556, -0.001129]` | `[-0.038388, +0.001480]` | `0.9806` (`0.9806`) | `[-0.018725, -0.010139, -0.018944]` (`0/3`) | **`0` vs `2`** | `[9, 9, 9]` | `-0.017263 [-0.034254, -0.005689]` (`p = 0.9964`) |
| `RERANK8_CR_UNC - CONTROL` (No Fallback) | `+0.036470` | `[+0.014969, +0.056603]` | `[+0.012194, +0.060083]` | `0.0003` | `[+0.036470, +0.029009, +0.036139]` (`3/3`) | `2` (`[1, 1, 0]`) | `[1, 1, 0]` | `+0.032998 [+0.017761, +0.051822]` (`p = 0.0001`) |
| `RERANK8_CR_SHUF - CONTROL` (Strict Placebo) | `+0.000000` | `[-0.002975, +0.000000]` | `[-0.003382, +0.000000]` | `1.0000` | `[+0.000000, -0.004400, +0.000000]` (`0/3`) | `0` (`[0, 0, 0]`) | `[0, 1, 0]` | `+0.000000 [-0.000992, +0.000000]` (`p = 1.0000`) |
| **`RERANK8_CR_SHUF_UNC - CONTROL` (Uncond. Placebo)** | **`-0.009746`** | **`[-0.021863, -0.001053]`** | **`[-0.023978, +0.000000]`** | `0.9844` | `[-0.006378, -0.016047, -0.009746]` (`0/3`) | `0` (`[0, 0, 0]`) | `[2, 5, 4]` | **`-0.004966 [-0.012661, -0.000511]` (`p = 0.9873`)** |
| **`RERANK8_CR_SHUF_UNC_M0 - CONTROL` (`margin = 0`)** | **`-0.029567`** | **`[-0.058183, +0.009266]`** | **`[-0.061964, +0.014912]`** | `0.9225` | `[-0.028542, -0.052862, -0.029567]` (`0/3`) | `2` (`[0, 0, 2]`) | `[11, 11, 11]` | **`-0.026520 [-0.055805, +0.010126]` (`p = 0.9102`)** |
| **`RERANK8_CR - RERANK8_CR_SHUF_UNC`** | **`+0.043934`** | **`[+0.026646, +0.066353]`** | **`[+0.023879, +0.070632]`** | **`0.0001`** | `[+0.043934, +0.041944, +0.046848]` (`3/3`) | `2` vs `0` | `[1, 1, 0]` | **`+0.039815 [+0.026468, +0.058756]` (`p = 0.0001`)** |
| **`RERANK8_CR_VETO - RERANK8_CR_SHUF_UNC`** | **`+0.025209`** | **`[+0.015524, +0.041690]`** | **`[+0.014074, +0.044530]`** | **`0.0001`** | `[+0.025209, +0.031805, +0.021750]` (`3/3`) | **`0` vs `0`** | **`[0, 0, 0]`** | **`+0.019649 [+0.012649, +0.030656]` (`p = 0.0001`)** |

---

## 5. MM Bradley-Terry Elo Ratings (`r882 = CONTROL = 1500.00`)

Ratings are computed via Minorization-Maximization (MM) Bradley-Terry maximum likelihood over family-equal pairwise win/loss/tie counts (`0.5` credit for ties within `1e-9` utility, anchored at `CONTROL = 1500.00`, scale `400 / ln(10)`), with `95%` confidence intervals over `1,000` family-cluster bootstrap draws (`205` families on `DL-Train`; `19` families on `DL-Val91` at `fold_seed = 273140`).

| Arm | `DL-Train` OOF (`205` Families) MM Elo | `DL-Train` `95%` Bootstrap CI | `DL-Train` Per-Seed Elo (`273101..273103`) | `DL-Val91` (`19` Families) MM Elo | `DL-Val91` `95%` Bootstrap CI | `DL-Val91` Per-Seed Elo (`273101..273103`) |
| --- | --- | --- | --- | --- | --- | --- |
| `CONTROL` (`r882` Top-1 Anchor) | `1500.00` | `[1500.00, 1500.00]` | `[1500.00, 1500.00, 1500.00]` | `1500.00` | `[1500.00, 1500.00]` | `[1500.00, 1500.00, 1500.00]` |
| **`RERANK8_CR` (Confirmatory `K=8`)** | **`1554.73`** (`+54.73`) | **`[1538.99, 1576.19]`** | `[1554.73, 1564.12, 1546.79]` | **`1568.99`** (`+68.99`) | **`[1550.14, 1609.09]`** | `[1591.22, 1561.99, 1568.99]` |
| **`RERANK8_CR_VETO` (`tau = 0.10`)** | **`1522.40`** (`+22.40`) | **`[1513.56, 1533.17]`** | `[1521.48, 1522.97, 1522.40]` | **`1530.97`** (`+30.97`) | **`[1505.45, 1546.59]`** | `[1533.90, 1530.97, 1522.34]` |
| `RERANK8_CR_UNC` (Unconditional) | `1556.08` (`+56.08`) | `[1540.48, 1577.77]` | `[1554.20, 1564.17, 1556.08]` | `1568.98` (`+68.98`) | `[1550.09, 1609.10]` | `[1591.22, 1561.98, 1568.98]` |
| `RERANK8_CR_SHUF` (Strict Placebo) | `1500.00` (`+0.00`) | `[1499.64, 1500.00]` | `[1500.00, 1499.51, 1500.00]` | `1500.00` (`+0.00`) | `[1496.60, 1500.00]` | `[1500.00, 1496.17, 1500.00]` |
| **`RERANK8_CR_SHUF_UNC` (Uncond. Placebo)** | **`1494.64`** (`-5.36`) | **`[1487.58, 1498.78]`** | `[1496.26, 1494.64, 1492.41]` | **`1484.17`** (`-15.83`) | **`[1464.59, 1496.51]`** | `[1484.17, 1472.73, 1491.37]` |
| **`RERANK8_CR_SHUF_UNC_M0` (`margin = 0`)** | **`1456.60`** (`-43.40`) | **`[1438.49, 1471.40]`** | `[1454.02, 1456.60, 1458.18]` | **`1474.16`** (`-25.84`) | **`[1417.64, 1532.39]`** | `[1483.86, 1453.30, 1474.16]` |
| `TOP4_PICK_ORACLE` (Diagnostic) | `1649.30` (`+149.30`) | `[1625.87, 1673.92]` | `[1649.11, 1649.49, 1649.30]` | `1635.75` (`+135.75`) | `[1580.60, 1697.81]` | `[1635.92, 1635.75, 1635.33]` |
| `TOP8_PICK_ORACLE` (Diagnostic) | `1693.09` (`+193.09`) | `[1669.03, 1718.30]` | `[1692.90, 1693.29, 1693.09]` | `1720.23` (`+220.23`) | `[1658.47, 1785.56]` | `[1720.41, 1720.23, 1719.80]` |
| `POOL_ORACLE` (Diagnostic) | `1804.60` (`+304.60`) | `[1779.13, 1829.83]` | `[1804.40, 1804.80, 1804.60]` | `1876.43` (`+376.43`) | `[1817.16, 1933.91]` | `[1876.62, 1876.43, 1875.99]` |

---

## 6. Replication Gate Assessment (`6 / 6` Passed for Both `RERANK8_CR` and `RERANK8_CR_VETO`)

| Gate # | Preregistered Criterion (`PREREG.md` Section 5) | `RERANK8_CR` (Confirmatory) | `RERANK8_CR_VETO` (`tau = 0.10`) | Status |
| --- | --- | --- | --- | --- |
| **Gate 1** | `DL-Train` OOF median family gain `>= +0.020` and `95%` CI lower bound `> 0.0` | **`+0.057130`**, `CI95 = [+0.019198, +0.116590]` | **`+0.027807`**, `CI95 = [+0.010083, +0.052092]` | **`PASS` (Both)** |
| **Gate 2** | `3/3` model seeds positive on `DL-Train` and `>= 7/9` `(model_seed × fold_seed)` repetitions positive | **`3/3` seeds**, **`9/9` repetitions positive** | **`3/3` seeds**, **`9/9` repetitions positive** | **`PASS` (Both)** |
| **Gate 3** | `Holm-Bonferroni m = 2` adjusted one-sided `p < 0.025` | **`Holm p = 0.0013 < 0.025`** (`97.5% CI = [+0.014119, +0.126979]`) | **`Holm p = 0.0002 < 0.025`** (`97.5% CI = [+0.007958, +0.056603]`) | **`PASS` (Both)** |
| **Gate 4** | `DL-Val91` secondary median family gain `> 0.0` | **`+0.036470 > 0.0`** (`3`-fseed `+0.033369 > 0.0`) | **`+0.015792 > 0.0`** (`3`-fseed `+0.013203 > 0.0`) | **`PASS` (Both)** |
| **Gate 5** | `DL-Train` `-2.0` invalid picks across `45` cells `<= 153` (`CONTROL`) | **`144 <= 153`** (`[50, 44, 50]`) | **`148 <= 153`** (`[52, 46, 50]`; **`0` on `DL-Val91`**) | **`PASS` (Both)** |
| **Gate 6** | Both `RERANK8_CR_SHUF` and `RERANK8_CR_SHUF_UNC` `<= +0.005` on `DL-Train` | `SHUF = 0.000000 <= +0.005`; **`SHUF_UNC = -0.008564 <= +0.005`** | `SHUF = 0.000000 <= +0.005`; **`SHUF_UNC = -0.008564 <= +0.005`** | **`PASS` (Both)** |

---

## 7. Next Step

1. **Promote `RERANK8` (`K = 8`, `568` cols) as the Verified Top-`K` Reranker Baseline**: `H273` independently confirms `H270` across both clean-room code (`Stage 1`: `+0.0541` vs `+0.0523`, `9/9` positive repetitions, `71.48%` switch agreement) and fresh fold/model seeds (`Stage 2`: `+0.0571` on `DL-Train`, `+0.0365` on `DL-Val91`, `Holm m=2 p = 0.0013`, MM Elo `1554.73` on `DL-Train` and `1568.99` on `DL-Val91`).
2. **Use `RERANK8_CR_VETO` (`tau = 0.10`) or Soft Invalid-Risk Penalization for Risk-Constrained Deployment**: Hard invalid-gate veto at `tau = 0.10` (`RERANK8_CR_VETO`) passes all `6/6` replication gates on its own (`+0.0278` on `DL-Train`, `+0.0158` on `DL-Val91`, `45/45` cells admitted), reduces `DL-Train` harmed families (`> 0.01`) from `33` to `9` (`-73%`), and eliminates `100%` of `DL-Val91` invalid picks and harmed families (`[0, 0, 0]`). However, because hard masking at `tau = 0.10` forfeits roughly half of raw median utility gain (`-0.0184` vs `RERANK8_CR`), any follow-up invalid-risk refinement on `RERANK8` should replace the hard `tau = 0.10` cutoff with a soft expected-utility score penalty (`score_k - lambda * p_invalid_k`) tuned strictly inside `TRAIN`/`FIT` folds.

---

## 8. Durable S3 Artifacts (`VersionId`s) & Resource Hygiene

All full artifacts were uploaded to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H273/` (`AWS_PROFILE=veox-jepson-prod`) prior to git closeout:

- **Preregistered Fold Manifest (`694,857` bytes, `outcomes_read = 0`)**:
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H273/prereg/fold_manifest.json` (`VersionId = oCb50mvrbyR0CCN_nhxc16GuBOpfzSQZ`, `sha256 = 2bb389098ebdf1720f9e88fb543f9df6edaa7b3a91e8327bb45be91dd5d6f5c9`)
- **Primary Compute Outputs (`attempt-001`, `s3_manifest.json` `VersionId = brvDVgTSnSiS3b1pFjPxC7e_uMSwl4mv`)**:
  - `metrics_full.json` (`108,383` bytes): `VersionId = DVSxAwQf8Dch8bJODBBNKylROtJwgXmb` (`sha256 = 2c5f13f2802d59f549356f9fd7f30a67abb99b797555794330fdd5c6bf65f8ea`)
  - `details.json` (`121,076` bytes): `VersionId = l9fqH6VbEq_1cGZwoedJSMNoivH2xc8q` (`sha256 = 8ab9da9caeb23ffd3f4a7eec9f68eecbdd29d2e4b17b5c4ea05e42a151dbf780`)
  - `panels.npz` (`555,214` bytes): `VersionId = drzIo47Ohb2cwE8kBjkhWQ8gy9QB5GV2` (`sha256 = d2c5ac5a6539b635569c9b8a2d80adb3b391fe9d2efa34431c490240de78683a`)
  - `fit_selections_lock.json` (`91,329` bytes): `VersionId = 2IPtjdf5BwgCGxeDJ8A9emP5S4pS2RBH` (`sha256 = a84863c1b30aa3e9b516724c587f684164f07ba5e2de3e9340986d6423bd5409`)
  - `stage1_h270_repro.json` (`15,572` bytes): `VersionId = rr5Ca8xcJOD6m_X4bdzk2O2sQvpER16K` (`sha256 = 340a122146c2939c7319b00f10db71c498181f2fa1f6593482db668c64053e13`)
  - `baseline.json` (`726` bytes): `VersionId = VUOWQeRtpx_hx.XdP3FEYjeOuiZ3EfyO` (`sha256 = c237cb76be9c6fb8cc13fa63503609e6943f7df06ed6f5d56cf7094127e70a32`)
  - `attempt-001.resource_summary.json` (`834` bytes): `VersionId = rreAEquGttCaO0ldYdz6lZ53lccHBNiF` (`wall = 1294.31 s`, `peak_rss = 912.8 MB <= 4 GiB`, `peak_threads = 3 <= 4`, `cpu_affinity = [57, 58, 59, 60]`, `pause_requested_seconds = 0.0`)
  - `smoke-001/smoke.json` (`371` bytes): `VersionId = OOCxF.xGeeCt1SfKozw2smB6QzfEqDuq` (`wall = 4.83 s`, `peak_rss = 602.5 MB`, `peak_threads = 3 <= 4`)
