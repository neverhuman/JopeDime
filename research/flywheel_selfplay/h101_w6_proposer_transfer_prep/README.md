# H101 — W6 Cross-Family Outcome Transfer: Complete Registered Design, 4-Node Real-S3 CPU Smoke & Prospective H94/H95 True-W1 Audit

- **Wave / ID**: `W6` / `H101` (renumbered from `H98`/`H100` per `rd_argon_idfix_t0420.md` and `rd_astra_h102_t0428.md`; `H98` is PR #975 `h98_w5_provenance`, `H100` is PR #994 ASAM confirmation wave)
- **Executed Run ID**: `argon-h98-proposer-transfer-20261031T0938Z` (immutable 4-node S3 execution; derived under `H101` without re-running fits)
- **Scope**: `DL-Train` only (`1,219` contexts / `205` families across 5 family-grouped folds + `50`-context / `44`-family / `150`-context-seed real-S3 prospective audit on `H94`/`H95`); **zero `DL-Val`, outer-test, sealed, or holdout reads; zero GPU (`0` native fits, `0` GPU seconds)**
- **Registered W6 Study Verdict (`PROPOSALS.md:209`)**:
  - **Overall Registered W6 Verdict**: **`KILL_NO_ADVANCE`** (`KILL_NO_ADVANCE (registered PROPOSALS.md:209 kill conditions met: TREAT_FAMBAL - RETR fails Common Advance Gate; unseen structural-feature proxy template stratum fails Common Advance Gate; context-shuffle placebo (TREAT_FAMBAL - SHUF) matches / fails Common Advance Gate; cached TREAT_FAMBAL - PRIOR pass=YES retained as diagnostic only)`)
  - **Registered Gate Checks (`PROPOSALS.md:209`)**:
    - Advance over generic prior (`TREAT_FAMBAL - PRIOR` Track-2 `U4`): **`PASS` (diagnostic only)** (`+24.97` Elo, mean `+0.012264`, median `+0.000000`, 4-primary Bonferroni LCB `+0.004938 > +0.001`, harm `9.27% <= 10.0%`, family W/T/L `89 / 67 / 49`)
    - Advance over current retrieval (`TREAT_FAMBAL - RETR` Track-2 `U4`): **`FAIL`** (`+65.39` Elo, mean `+0.102915`, Bonferroni LCB `+0.039653`, **harm `15.12% > 10.0%`**, family W/T/L `138 / 7 / 60`)
    - Benefit survives on unseen template proxy (`TREAT_FAMBAL - PRIOR` Unseen-Proxy-Hash Track-2 `U4`): **`FAIL` (kill condition met)** (`+21.19` Elo, mean `+0.008730`, **Bonferroni LCB `+0.000639 < +0.001`**, **harm `10.24% > 10.0%`**, family W/T/L `102 / 48 / 55`)
    - Beats context-shuffle placebo (`TREAT_FAMBAL - SHUF` Track-2 `U4`): **`FAIL` (kill condition met — context-shuffle matches)** (`+15.90` Elo, mean `+0.007285`, **1-sided LCB `+0.000463 < +0.001`**, **harm `16.10% > 10.0%`**, family W/T/L `86 / 42 / 77`)
  - **Cached Screen Track 1 (`val_reg_inner` Legacy Single-Pick)**: **`KILL_NO_ADVANCE`** (`TREAT_FAMBAL - PRIOR` `+24.14` Elo, mean `+0.037160`, Bonferroni LCB `-0.016833`, harm `21.46% > 10.0%`, W/T/L `88 / 55 / 62`)
  - **Prospective New-Program Audit (`H94` + `H95` + `H85` Persisted `W1` Models, Restricted Evaluated-Subset Comparison)**: **`NO_ADVANCE`** (`H94_HEDGE_CTX - H94_PRIOR_top4`: `-48.71` Elo, mean `-0.010545`, Bonferroni LCB `-0.040062`, harm `13.64%`; restricted evaluated-subset `TRUE_W1_HEDGE_CTX_EVAL_SUBSET - TRUE_W1_PRIOR_EVAL_SUBSET` across the `556` `H94`-evaluated proposals where `51 / 150` context-seeds have all 12 true-`W1` top-4 picks covered: `-76.80` Elo, mean `-0.014417`, Bonferroni LCB `-0.042997`, harm `15.91%`)

---

## 1. Executive Summary & Scientific Findings (Elo First)

### 1.1 Registered W6 Outcome Transfer Verdict (`KILL_NO_ADVANCE`)
1. **Why Registered W6 Outcome Transfer Is Killed (`PROPOSALS.md:209`)**:
   - Under `PROPOSALS.md` (line 209), advancing a `W6` cross-family outcome transfer scorer requires advancing over **both** the generic prior (`PRIOR`) and current retrieval (`RETR`), and triggers a **Kill** if the benefit disappears on unseen templates or if the context-shuffle placebo (`SHUF`) matches.
   - Although the family-balanced rank-8 bilinear residual (`TREAT_FAMBAL`) clears the individual `TREAT_FAMBAL - PRIOR` contrast on cached Track-2 `U4` (`1524.97` vs `1500.00` Elo, mean `+0.012264`, Bonferroni LCB `+0.004938`, harm `9.27%`), **all three registered kill/control gates fail**:
     1. **Retrieval control (`TREAT_FAMBAL - RETR`) fails harm**: Despite `+65.39` Elo (`1524.97` vs `1459.58`) and mean `+0.102915`, `15.12%` of families (`31 / 205`) lose more than `0.01` `U4` against `RETR` (exceeding the `10.0%` Common Advance Gate ceiling).
     2. **Unseen structural-feature proxy stratum (`TREAT_FAMBAL - PRIOR` on unseen proxy hashes) fails both LCB and harm**: On the `91.95%` (`1,286,695 / 1,399,411`) of unrevealed candidate slots whose `program412` structural-feature proxy hash was not seen in fit families, the Bonferroni LCB drops to `+0.000639 < +0.001` and family harm rises to `10.24% > 10.0%` (`W/T/L 102 / 48 / 55`).
     3. **Context-shuffle placebo (`TREAT_FAMBAL - SHUF`) matches**: Permuting context embeddings across fit contexts (`SHUF`) reaches `1509.07` Elo (`+9.07` Elo over `PRIOR`), and the direct `TREAT_FAMBAL - SHUF` contrast fails (`+15.90` Elo, mean `+0.007285`, 1-sided LCB `+0.000463 < +0.001`, harm `16.10% > 10.0%`, `W/T/L 86 / 42 / 77`). Thus, a substantial fraction of the apparent cached `TREAT_FAMBAL - PRIOR` gain is absorbed by the shuffled-context bilinear baseline rather than true context-specific bilinear outcome transfer.
   - **Comparison with equal-context weighting (`TREAT`)**: In this single cached 5-fold fit, inverse-family-size weighting (`TREAT_FAMBAL`, `1524.97` Elo, harm `9.27%`) had lower family harm than equal-context weighting (`TREAT`, `1524.30` Elo, harm `13.17%`), which is why `TREAT_FAMBAL - PRIOR` crossed the individual `10%` harm threshold on cached pools. However, no direct `TREAT_FAMBAL - TREAT` bootstrap CI was registered, and because `TREAT_FAMBAL` fails the retrieval, unseen-proxy, and context-shuffle placebo kill gates, the cached `TREAT_FAMBAL - PRIOR` pass is retained strictly as a **diagnostic**.
2. **Retrieval (`RETR`) & Structural-Feature Proxy (`FIT_TEMPLATE_MEAN`) Implementation Definitions**:
   - **`RETR` (4-NN raw-`lm12` $L_2$ context retrieval + `proj16` $L_2$ champion scoring)**: In `run_fold_shard_from_bundle`, `RETR` selects the `RETR_CONTEXTS = 4` nearest out-of-fold fit contexts by Euclidean ($L_2$) distance on the **raw first-12 `lm48` coordinates** (`lm[i][:12]`), collects the `proj16` (`16D` orthogonal projection of `prog412`) vectors of those fit contexts' champion programs from `champions_featurized.json`, and scores each unrevealed candidate by negative minimum Euclidean ($L_2$) distance in `proj16` to those retrieved champion vectors.
   - **`compute_template_hashes` / `FIT_TEMPLATE_MEAN` (`program412` structural-feature proxy hash)**: Because `pools/{cid}.npz` stores pre-featurized `412`-dim `program` vectors rather than raw DSL AST strings, `compute_template_hashes` constructs a **structural-feature proxy hash** by excluding the `26` continuous numeric threshold/constant-magnitude quantile columns (`338..363` inclusive) from `program412`, rounding the remaining `386` structural coordinates to `10^-3` (`np.rint(prog_matrix[:, STRUCTURAL_COLS] * 1000.0)`), and taking the 16-hex-char SHA-256 prefix. Across the 5 `W1` folds, this structural-feature proxy covers **`8.0545%` (`112,716 / 1,399,411`)** of unrevealed candidate slots (`15,919` unique proxy hashes vs `16,580` unique exact program SHAs across `1,404,287` total slots), only `+1.045 percentage points` above exact `program_sha256` coverage (`7.0094%`, `98,091 / 1,399,411`). Empirical-Bayes shrinkage on this proxy (`FIT_TEMPLATE_MEAN`) yields `1493.42` Elo (`-6.58` Elo vs `PRIOR`, 1-sided LCB `-0.009857`, harm `12.20%`, **FAIL**).
3. **Exact-ID Overlap Reconciliation (`H83` vs `H84`)**:
   - `H83` (`98,091 / 1,399,411 = 7.0094%`) counted **unrevealed slots (`>= 4`)** across the **`W1` fold manifest** (`dd211134...`).
   - `H84` (`98,820 / 1,404,287 = 7.0370%`) counted **all slots (`0..N-1`)** across the **`H72` fold manifest** (`129002bc...`).
   - Our shard runner (`run_fold_shard_from_bundle`) computes **both** simultaneously on the exact same pools and reproduces both numbers to the exact slot (`98,091 / 1,399,411` and `98,820 / 1,404,287`). Even with clean fallback to `PRIOR` on the `92.99%` unseen IDs (fixing `H84`'s zero-fill bug), `FIT_PROG_MEAN` changes top-4 picks on `799 / 1,219` (`65.55%`) contexts and reaches `1495.13` Elo on Track 2 (`-4.87` Elo vs `PRIOR`, Bonferroni LCB `-0.005800`, harm `12.68%`, **FAIL**).

### 1.2 Real-S3 Diagnosis of the `H92 -> H94` Cached-to-Prospective Reversal (`W1` Persisted Models vs `H94` Interleaved Refits)
By re-generating `H94`'s exact deterministic proposal streams (`150` context-seeds across `50` contexts / `44` families, `16` fresh GP proposals per context-seed; `100%` SHA-256 match against all `H94` revealed IDs) and scoring them with the persisted `H85` `W1` fold-grouped calibrated outer models (`w1_outer_fold_models.pkl`, SHA-256 `2ebc4859...`), we isolate two distinct mechanisms behind `H94`'s prospective failure:
1. **`H94`'s Interleaved Fold Refit Diverged Materially from True `W1` Scoring**:
   - On the exact same 16-candidate prospective slates, the mean top-4 set overlap (`|A ∩ B| / 4`) between the persisted `W1` models (`w1_outer_fold_models.pkl`, which use `family_id` fold grouping, `sqrt(inv_family_size)` sample weights, and inner-fold linear calibration `a + b * raw`) and `H94`'s uncalibrated `i % 5` interleaved refits is:
     - **`PRIOR_top4`**: **`76.67%`** (`3.07 / 4` picks in common)
     - **`CTX_top4`**: **`65.83%`** (`2.63 / 4` picks in common)
     - **`HEDGE_CTX`**: **`70.50%`** (`2.82 / 4` picks in common)
   - Because `H94` only ran native Rust evaluations on the union of its own 3 arms' top-4 picks (`556` unique `(context, program_sha256)` evaluations out of `2,400` generated candidates), only **`51 / 150` (`34.0%`)** context-seeds have all 12 of the true `W1` top-4 picks covered by `H94`'s `556` native evaluations.
2. **Restricted Evaluated-Subset Comparison & `H95` Confirmed-HWM Recall**:
   - **Important restriction qualification**: The `TRUE_W1_*_EVAL_SUBSET` arms below re-rank **only** the subset of proposals per context-seed that `H94` had already natively evaluated (`>= 4` evaluated proposals per context-seed, with full 12-pick coverage on `51 / 150` context-seeds). Therefore, the `+8.79` Elo difference of `TRUE_W1_PRIOR_EVAL_SUBSET` (`1508.79` Elo) vs `H94_PRIOR_top4` (`1500.00` Elo) is a **restricted evaluated-subset diagnostic**, NOT a like-for-like 16-proposal evaluation (mean `-0.000627`, 1-sided LCB `-0.003061`, family W/T/L `10 / 26 / 8`). Furthermore, even on this restricted subset, `TRUE_W1_HEDGE_CTX_EVAL_SUBSET` (`1431.99` Elo) loses to `TRUE_W1_PRIOR_EVAL_SUBSET` (`1508.79` Elo) by `-76.80` Elo (mean `-0.014417`, Bonferroni LCB `-0.042997`, harm `15.91%`, family W/T/L `6 / 19 / 19`, verdict **`NO_ADVANCE`**).
   - Across the full 16-candidate slates, scoring with `TRUE_W1_PRIOR_top4` recalls **`13 / 30` (`43.33%`)** of `H95`'s resplit-confirmed high-water marks (`11 / 27 = 40.74%` material HWMs) and `TRUE_W1_HEDGE_CTX` recalls **`7 / 30` (`23.33%`)** (`6 / 27 = 22.22%` material), compared to `H94_PRIOR_top4` (`18 / 30 = 60.00%` confirmed; `15 / 27 = 55.56%` material) and `H94_HEDGE_CTX` (`19 / 30 = 63.33%` confirmed; `19 / 27 = 70.37%` material), because `H94` only natively evaluated the union of its own 3 arms' top-4 picks (`99 / 150` context-seeds have at least one un-evaluated `TRUE_W1` top-4 pick).

---

## 2. Registered Arm Panel & Elo Leaderboard (`1,219` `DL-Train` Contexts / `205` Families)

*All values below are generated directly from `metrics.json` (`SHA-256: 78186916cfcc341f189a3f60474da7f9040d0837fea43389865ee76980206a80`). Family-grouped Bradley-Terry Elo (`prior=0.01, tie_margin=1e-4, n_iters=400`, canonical `elo_tracker.py` `9ece9832...`). Family-mean `U0` (4 revealed anchors) = `1.458233`.*

| Arm | Track 2 `U4` Elo (`PRIOR=1500`) | Track 2 `U4` Elo (`R882=1500`) | Track 2 Family-Mean `U4` | Track 2 Context-Mean `U4` | Track 1 `val_reg_inner` Elo (`PRIOR=1500`) | Track 1 Family-Mean `val_reg_inner` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CTX` | `1569.76` | `1533.21` | `1.680362` | `1.716926` | `1573.99` | `1.317445` |
| `HEDGE_CTX` | `1551.88` | `1515.33` | `1.686021` | `1.729732` | `1500.00` | `1.182765` |
| `R882_CONTINUITY` | `1536.55` | `1500.00` | `1.609212` | `1.643161` | `1579.14` | `1.355472` |
| `TREAT_FAMBAL` | `1524.97` | `1488.42` | `1.662061` | `1.705083` | `1524.14` | `1.219924` |
| `NEAR_CHAMP_TREAT` | `1524.43` | `1487.88` | `1.656972` | `1.697797` | `1526.42` | `1.220847` |
| `TREAT` | `1524.30` | `1487.75` | `1.658271` | `1.701007` | `1529.69` | `1.237736` |
| `HEDGE_NEAR_CHAMP` | `1521.16` | `1484.61` | `1.657804` | `1.700995` | `1500.00` | `1.182765` |
| `HEDGE_TREAT_FAMBAL` | `1519.26` | `1482.71` | `1.660032` | `1.704533` | `1500.00` | `1.182765` |
| `SHUF` | `1509.07` | `1472.52` | `1.654776` | `1.700825` | `1517.90` | `1.227244` |
| `PRIOR` | `1500.00` | `1463.45` | `1.649797` | `1.695609` | `1500.00` | `1.182765` |
| `FIT_PROG_MEAN` | `1495.13` | `1458.58` | `1.660159` | `1.704359` | `1497.84` | `1.159549` |
| `FIT_TEMPLATE_MEAN` | `1493.42` | `1456.87` | `1.655063` | `1.698653` | `1497.90` | `1.169970` |
| `RETR` | `1459.58` | `1423.03` | `1.559147` | `1.597712` | `1489.23` | `0.918211` |

---

## 3. Primary & Secondary Contrasts Across Both Decision Tracks

### 3.1 Track 2 — Verified Search (`U4`, 205 Families, 10,000 Bootstrap Draws, Seed `20261031`)
*Panel verdict in `metrics.json`: `KILL_NO_ADVANCE (fails registered W6 gates: TREAT_FAMBAL - RETR, TREAT_FAMBAL - SHUF; diagnostic primaries passing: TREAT_FAMBAL - PRIOR)`.*

| Contrast | Family Mean $\Delta U_4$ | Family Median $\Delta U_4$ | One-Sided LCB | Harm (`<-0.01`) | Family W / T / L | Individual Gate (`LCB > +0.001`, `med >= 0`, `harm <= 10%`) |
| --- | ---: | ---: | ---: | ---: | :---: | :---: |
| `TREAT_FAMBAL - PRIOR` (Primary 1, Bonferroni p1.25) | `+0.012264` | `+0.000000` | `+0.004938` | `9.27%` | `89 / 67 / 49` | **PASS** |
| `TREAT_FAMBAL - RETR` (Primary 2, Bonferroni p1.25) | `+0.102915` | `+0.005873` | `+0.039653` | `15.12%` | `138 / 7 / 60` | **FAIL** |
| `FIT_PROG_MEAN - PRIOR` (Primary 3, Bonferroni p1.25) | `+0.010361` | `+0.000000` | `-0.005800` | `12.68%` | `43 / 108 / 54` | **FAIL** |
| `HEDGE_NEAR_CHAMP - PRIOR` (Primary 4, Bonferroni p1.25) | `+0.008007` | `+0.000000` | `+0.000831` | `8.78%` | `89 / 72 / 44` | **FAIL** |
| `TREAT_FAMBAL - SHUF` (Secondary placebo, p5) | `+0.007285` | `+0.000000` | `+0.000463` | `16.10%` | `86 / 42 / 77` | **FAIL** |
| `HEDGE_TREAT_FAMBAL - PRIOR` (Secondary hedge, p5) | `+0.010235` | `+0.000000` | `+0.005677` | `9.76%` | `79 / 75 / 51` | **PASS** |
| `TREAT - PRIOR` (Equal-context H83 continuity, p5) | `+0.008473` | `+0.000000` | `+0.003755` | `13.17%` | `90 / 59 / 56` | **FAIL** |
| `FIT_TEMPLATE_MEAN - PRIOR` (Structural-proxy EB, p5) | `+0.005266` | `+0.000000` | `-0.009857` | `12.20%` | `38 / 113 / 54` | **FAIL** |
| `CTX - PRIOR` (W1 context-aware GBM, p5) | `+0.030564` | `+0.003986` | `+0.009645` | `16.10%` | `134 / 8 / 63` | **FAIL** |
| `HEDGE_CTX - PRIOR` (W1 hedged context GBM, p5) | `+0.036223` | `+0.002615` | `+0.021008` | `8.29%` | `134 / 22 / 49` | **PASS** |
| `HEDGE_CTX - R882_CONTINUITY` (vs frozen r882, p5) | `+0.076809` | `+0.003000` | `+0.044018` | `21.46%` | `119 / 8 / 78` | **FAIL** |

### 3.2 Track 1 — Legacy Single-Pick Selection (`val_reg_inner`, 205 Families)
*Panel verdict in `metrics.json`: `KILL_NO_ADVANCE (fails registered W6 gates: TREAT_FAMBAL - PRIOR, TREAT_FAMBAL - RETR, TREAT_FAMBAL - SHUF; diagnostic primaries passing: none)`. Every row below is regenerated directly from `metrics["track1_legacy_val_reg_inner"]`.*

| Contrast | Family Mean $\Delta$ | Family Median $\Delta$ | One-Sided LCB | Harm (`<-0.01`) | Family W / T / L | Individual Gate |
| --- | ---: | ---: | ---: | ---: | :---: | :---: |
| `TREAT_FAMBAL - PRIOR` (Primary 1, Bonferroni p1.25) | `+0.037160` | `+0.000000` | `-0.016833` | `21.46%` | `88 / 55 / 62` | **FAIL** |
| `TREAT_FAMBAL - RETR` (Primary 2, Bonferroni p1.25) | `+0.301713` | `+0.012655` | `+0.079462` | `40.49%` | `110 / 0 / 95` | **FAIL** |
| `FIT_PROG_MEAN - PRIOR` (Primary 3, Bonferroni p1.25) | `-0.023216` | `+0.000000` | `-0.082948` | `28.29%` | `80 / 34 / 91` | **FAIL** |
| `HEDGE_NEAR_CHAMP - PRIOR` (Primary 4, Bonferroni p1.25) | `+0.000000` | `+0.000000` | `+0.000000` | `0.00%` | `0 / 205 / 0` | **FAIL** |
| `CTX - PRIOR` (W1 context-aware GBM, p5) | `+0.134680` | `+0.065253` | `+0.075864` | `24.39%` | `143 / 0 / 62` | **FAIL** |
| `HEDGE_CTX - R882_CONTINUITY` (vs frozen r882, p5) | `-0.172707` | `-0.043248` | `-0.256120` | `55.12%` | `85 / 0 / 120` | **FAIL** |

---

## 4. Within-Context Paired Seen-Proxy-Hash vs Unseen-Proxy-Hash Evaluation (`1,219` Contexts / `205` Families)

Every one of the `1,219` `DL-Train` contexts has at least 4 unrevealed slots whose `program412` structural-feature proxy hash (`compute_template_hashes`: `program412` rounded to `10^-3` with constant-magnitude quantile columns `338..363` excluded) was seen in fit families (`112,716 / 1,399,411 = 8.0545%` of unrevealed slots) and at least 4 unrevealed slots whose structural-feature proxy hash is unseen (`1,286,695 / 1,399,411 = 91.9455%` of unrevealed slots). Evaluating both strata within every context eliminates family-composition confounding and tests the registered `W6` unseen-template kill gate:

| Stratum (`1,219` Contexts / `205` Families) | Contrast | Track 2 `U4` Elo (`PRIOR=1500`) | Family Mean $\Delta U_4$ | Family Median $\Delta U_4$ | One-Sided LCB | Harm (`<-0.01`) | Family W / T / L | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **Seen-Proxy-Hash (`8.05%` slots)** | `TREAT_FAMBAL - PRIOR` | `1500.41` | `+0.004835` | `+0.000000` | `+0.000286` (Bonf. p1.25) | `3.90%` | `34 / 144 / 27` | **FAIL** |
| **Seen-Proxy-Hash (`8.05%` slots)** | `HEDGE_TREAT_FAMBAL - PRIOR` | `1502.03` | `+0.005043` | `+0.000000` | `+0.001561` (1-sided p5) | `2.44%` | `34 / 151 / 20` | **PASS** |
| **Seen-Proxy-Hash (`8.05%` slots)** | `FIT_TEMPLATE_MEAN - PRIOR` | `1497.56` | `-0.001455` | `+0.000000` | `-0.009034` (1-sided p5) | `13.66%` | `46 / 103 / 56` | **FAIL** |
| **Seen-Proxy-Hash (`8.05%` slots)** | `HEDGE_CTX - PRIOR` | `1511.41` | `+0.006529` | `+0.000000` | `-0.000292` (1-sided p5) | `11.71%` | `69 / 89 / 47` | **FAIL** |
| **Unseen-Proxy-Hash (`91.95%` slots)** | `TREAT_FAMBAL - PRIOR` | `1521.19` | `+0.008730` | `+0.000079` | `+0.000639` (Bonf. p1.25) | `10.24%` | `102 / 48 / 55` | **FAIL** |
| **Unseen-Proxy-Hash (`91.95%` slots)** | `HEDGE_TREAT_FAMBAL - PRIOR` | `1518.33` | `+0.009085` | `+0.000000` | `+0.003751` (1-sided p5) | `10.24%` | `91 / 55 / 59` | **FAIL** |
| **Unseen-Proxy-Hash (`91.95%` slots)** | `FIT_TEMPLATE_MEAN - PRIOR` | `1500.00` | `+0.000000` | `+0.000000` | `+0.000000` (1-sided p5) | `0.00%` | `0 / 205 / 0` | **FAIL** |
| **Unseen-Proxy-Hash (`91.95%` slots)** | `HEDGE_CTX - PRIOR` | `1532.86` | `+0.010228` | `+0.002135` | `-0.001257` (1-sided p5) | `14.15%` | `128 / 14 / 63` | **FAIL** |

- **Finding**: On seen structural-feature proxy hashes (`8.05%` of unrevealed slots), `TREAT_FAMBAL - PRIOR` has Bonferroni LCB `+0.000286` and harm `3.90%` (**FAIL**). On unseen structural-feature proxy hashes (`91.95%` of unrevealed slots), `TREAT_FAMBAL - PRIOR` has Bonferroni LCB `+0.000639 < +0.001` with harm `10.24% > 10.0%` (**FAIL**), triggering the registered `PROPOSALS.md:209` unseen-template kill rule.

---

## 5. 4-Node Sharded Execution Receipt & Exact-ID / Proxy-Hash Census

Executed across 4 nodes with CPU load strictly below `85%` on every host (`0` GPU usage, `xbabe0` untouched). Every per-fold numerator and denominator below is generated directly from `metrics["shard_receipts"]`:

| Fold | Host (`nice`) | Held Contexts / Families | `W1`-Fold Unrevealed Seen Exact IDs | `W1`-Fold Unrevealed Seen Proxy Hashes | `H72`-Fold Total Seen Exact IDs | Wall / CPU Time | Shard JSON SHA-256 |
| :---: | --- | :---: | :---: | :---: | :---: | :---: | --- |
| `0` | `xbabe1` (`nice -n 19`) | `227 / 41` | `18,591 / 260,595` (`7.13%`) | `21,197 / 260,595` (`8.13%`) | `21,783 / 302,976` (`7.19%`) | `74.28s / 101.37s` | `3b20258968d0a2a3...` |
| `1` | `xbabe3` (`nice -n 10`) | `234 / 41` | `17,332 / 268,632` (`6.45%`) | `20,000 / 268,632` (`7.45%`) | `18,793 / 248,832` (`7.55%`) | `14.60s / 35.78s` | `3c5aadbe238f85e0...` |
| `2` | `xbabe3` (`nice -n 10`) | `269 / 41` | `21,419 / 308,812` (`6.94%`) | `24,607 / 308,812` (`7.97%`) | `23,126 / 319,104` (`7.25%`) | `14.44s / 36.91s` | `04069e8cab6fe832...` |
| `3` | `xbabe4` (`nice -n 10`) | `271 / 41` | `22,181 / 311,108` (`7.13%`) | `25,555 / 311,108` (`8.21%`) | `18,728 / 276,480` (`6.77%`) | `12.65s / 12.65s` | `99d8ac40fc515dd9...` |
| `4` | `xbabe2` (`nice -n 10`) | `218 / 41` | `18,568 / 250,264` (`7.42%`) | `21,357 / 250,264` (`8.53%`) | `16,390 / 256,895` (`6.38%`) | `92.41s / 114.28s` | `a05b0d6b914bbecf...` |
| **Total** | **4 hosts** | **`1,219 / 205`** | **`98,091 / 1,399,411` (`7.0094%`)** | **`112,716 / 1,399,411` (`8.0545%`)** | **`98,820 / 1,404,287` (`7.0370%`)** | **`208.38s / 300.98s`** | — |

---

## 6. Real-S3 Prospective New-Program Audit (`H94` + `H95` + `H85` Persisted `W1` Models)

*Note: `H94_*` arms evaluate full 16-candidate prospective slates (`150` context-seeds across `50` contexts / `44` families). `TRUE_W1_*_EVAL_SUBSET` arms are a **restricted evaluated-subset comparison** over the `556` `(context, program_sha256)` pairs natively evaluated by `H94` (`51 / 150` context-seeds have all 12 true-`W1` top-4 picks covered).*

| Prospective Arm (`50` Contexts / `44` Families / `150` Context-Seeds) | Evaluation Scope | Family-Grouped Elo (`H94_PRIOR=1500`) | Family-Mean `U4` | Context-Mean `U4` | `H95` Resplit-Confirmed HWM Recall (`30` total) | `H95` Material HWM Recall (`27` total) | Mean Top-4 Overlap vs `H94` Refit |
| --- | --- | ---: | ---: | ---: | :---: | :---: | :---: |
| `H94_PRIOR_top4` (`H94` interleaved refit) | Full 16-candidate slate | `1500.00` | `2.080933` | `1.993979` | `18 / 30` (`60.00%`) | `15 / 27` (`55.56%`) | `100.0%` |
| `H94_HEDGE_CTX` (`H94` interleaved refit) | Full 16-candidate slate | `1451.29` | `2.070388` | `1.986361` | `19 / 30` (`63.33%`) | `19 / 27` (`70.37%`) | `100.0%` |
| `H94_CTX_top4` (`H94` interleaved refit) | Full 16-candidate slate | `1425.97` | `2.059191` | `1.976466` | `19 / 30` (`63.33%`) | `19 / 27` (`70.37%`) | `100.0%` |
| `TRUE_W1_HEDGE_CTX` (`H85` persisted `W1` models) | Restricted `H94`-evaluated subset (`51/150` full) | `1431.99` | `2.065889` | `1.980091` | `7 / 30` (`23.33%`) | `6 / 27` (`22.22%`) | `70.50%` |
| `TRUE_W1_PRIOR_top4` (`H85` persisted `W1` models) | Restricted `H94`-evaluated subset (`51/150` full) | `1508.79` | `2.080306` | `1.993128` | `13 / 30` (`43.33%`) | `11 / 27` (`40.74%`) | `76.67%` |
| `TRUE_W1_CTX_top4` (`H85` persisted `W1` models) | Restricted `H94`-evaluated subset (`51/150` full) | `1424.64` | `2.055858` | `1.971402` | `9 / 30` (`30.00%`) | `9 / 27` (`33.33%`) | `65.83%` |

---

## 7. Execution Provenance, Derivation Lineage & S3 Version-Pinned Custody

### 7.1 Execution vs Post-Review Formatted Source Provenance
- **Executed Script (`w6_transfer_prep.py` at 2026-10-31T09:38Z execution time)**:
  - **SHA-256**: `3e6eda5859e9ec387ee57b8dc188fb0f82baedf9567eec583aa623889c63bdcc` (`65,971` bytes)
  - **Pinned S3 URI**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w6/argon-h98-proposer-transfer-20261031T0938Z/w6_transfer_prep.py` (`VersionId: rul15f0jKDZewt7R_wN5NZyRg8Wul4NV`)
- **Immutable Raw Execution `metrics.json` (written by `merge` at 2026-10-31T09:38Z)**:
  - **SHA-256**: `0ca473ed3de4ad0012148bb9f670e5a4562888617879a7285aea7b7899efe47e` (`85,336` bytes)
  - **Pinned S3 URI**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w6/argon-h98-proposer-transfer-20261031T0938Z/metrics.json` (`VersionId: PqrsUh5VE1NL2wrZFVZ2w31DhSS5X762`)
- **Post-Review Formatted `H101` Source (`w6_transfer_prep.py` in this commit)**:
  - **SHA-256**: `d79358c7c668e72034df0595872749baaf0a45040487d023a04af1557dd6e802`
- **Committed Derived `H101` `metrics.json` (in this commit)**:
  - **SHA-256**: `78186916cfcc341f189a3f60474da7f9040d0837fea43389865ee76980206a80`
  - **Derivation**: Derived deterministically without re-execution from the immutable S3 execution `metrics.json` (`0ca473ed3de4ad0012148bb9f670e5a4562888617879a7285aea7b7899efe47e`, `VersionId: PqrsUh5VE1NL2wrZFVZ2w31DhSS5X762`) by applying the `H98 -> H100 -> H101` wave-ID rename, recording `derivation_provenance`, and applying `derive_registered_w6_verdict()` per `PROPOSALS.md:209`. All numerical arm utilities, Elo ratings, bootstrap CIs, and shard receipts are 100% identical to `0ca473ed3de4ad0012148bb9f670e5a4562888617879a7285aea7b7899efe47e`.

The derivation also adds diagnostic-primary lists and the `micro_rust_evaluator` pin that was absent from the immutable metrics. That later metadata addition is not independent proof of the evaluator used historically. The integration review checked all scientific numeric leaves and shard receipts against the original metrics; no numerical change or new fit occurred.

### 7.2 S3 Version-Pinned Custody Objects (`s3://veox-jopedime-learning-use2`)

The following exact versions were independently downloaded and checked against their SHA-256 and byte counts during the ASTRA integration review. This corrects erroneous hand-copied receipt entries in the prior README; the stored numerical metrics are unchanged. Historical create-only behavior is not newly established by these reads.

- **Original execution prefix**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w6/argon-h98-proposer-transfer-20261031T0938Z/`
  - `manifest.json`: `VersionId: pxkd5g.nfU_zG_ShoAvyRYZOookpwRv8`, `SHA-256: e73cb69429943887e0450b67fda39c20fa37d254fcd81c8d9e468882ca442657`, `14,519` bytes
  - `metrics.json`: `VersionId: PqrsUh5VE1NL2wrZFVZ2w31DhSS5X762`, `SHA-256: 0ca473ed3de4ad0012148bb9f670e5a4562888617879a7285aea7b7899efe47e`, `85,336` bytes
  - `per_context_vectors.json`: `VersionId: gPqzXGY6YpcRh_31h.ip9IsDt7.Ma8RT`, `SHA-256: 9875492fc764da9dab0b7578f724415a847d4ff535c585e17dbc0efe0a69093e`, `1,435,174` bytes
  - `shard_0.json`: `VersionId: fY4wRfOG4cZZlij4EeuGv__F_pRxO0Oo`, `SHA-256: 3b20258968d0a2a37196e9f1ff37527dbe92a53c2136c98ca6fe6b0ff36c345f`, `785,884` bytes
  - `shard_1.json`: `VersionId: dnOzZdKwBQ6QBeO424Pa3Ki2j319.jby`, `SHA-256: 3c5aadbe238f85e06093bb7665dda7d2f22dab19d4400e6645c4a186db8dcc2f`, `808,752` bytes
  - `shard_2.json`: `VersionId: 98Z033eHXV.D9fH.HA_ZuomOoK5YAwiW`, `SHA-256: 04069e8cab6fe8321d60fc5cb101ccd84455077a928f8a78b4004afcd92db59c`, `931,152` bytes
  - `shard_3.json`: `VersionId: OrqqJ6s99RY.LNsSuo2uGhTJF0R2fwRH`, `SHA-256: 99d8ac40fc515dd992aff7c9c4cee12e4ca63ddd9d0dde343ab39c7e9950c9e1`, `935,981` bytes
  - `shard_4.json`: `VersionId: iP.289KXexCCE6wCh1f9FBivNpYI284t`, `SHA-256: a05b0d6b914bbecf499065b7582ad60434713fe52e14b019e712411654f2d86e`, `751,426` bytes
  - `test_w6_transfer_prep.py`: `VersionId: 13AsbPOnMlV8iU6bqOxhDfLrob8Az5kG`, `SHA-256: a86f0fb65f16d50fbd8ebcfefbe8dcac2948a240cfa60dcc6940a4cc6806ae24`, `7,278` bytes
  - `w6_transfer_prep.py`: `VersionId: rul15f0jKDZewt7R_wN5NZyRg8Wul4NV`, `SHA-256: 3e6eda5859e9ec387ee57b8dc188fb0f82baedf9567eec583aa623889c63bdcc`, `65,971` bytes

---

## 8. Reproduction & Verification Commands

```bash
# 1. Run unit tests for weighted bilinear SVD, empirical-Bayes shrinkage, anchor-boundary hedging,
#    structural-feature proxy hashing, panel completeness guards, and registered W6 verdict derivation
nice -n 10 pytest -p no:cacheprovider -q research/flywheel_selfplay/h101_w6_proposer_transfer_prep/test_w6_transfer_prep.py

# 2. Lint and format check
nice -n 10 ruff check research/flywheel_selfplay/h101_w6_proposer_transfer_prep/
nice -n 10 ruff format --check research/flywheel_selfplay/h101_w6_proposer_transfer_prep/
```
