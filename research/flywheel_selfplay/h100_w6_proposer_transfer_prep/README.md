# H100 — W6 Cross-Context Proposer Transfer Prep, Family-Balanced Bilinear Screen & Prospective True-W1 Audit (CPU-Only, 4-Node)

## 1. Executive Summary & Verdicts (Elo Reported First)

`H100` completes the Wave 6 (`W6`, `RES-03`) CPU-only design, 4-node outer-fold smoke screen, and prospective true-`W1` re-scoring audit across all `1,219` `DL-Train` contexts (`205` families, `1,404,287` total pool slots, `1,399,411` unrevealed slots after the 4 hash-ordered anchors) and the `50`-context / `44`-family `H94`/`H95` prospective new-program panel. All inputs and outputs are strictly `DL-Train` only (`0` `DL-Val`, outer-test, sealed, or holdout reads; `0` GPU seconds; `<= 85%` CPU per host across `xbabe1`, `xbabe2`, `xbabe3`, `xbabe4`).

### Headline Family-Grouped Bradley-Terry Elo Summary (`1,219` Contexts / `205` Families)

| Contender Arm | Track 2 (`U_4`) Elo (`PRIOR = 1500`) | Track 2 (`U_4`) Elo (`R882 = 1500`) | Track 1 (`val_reg_inner`) Elo (`PRIOR = 1500`) | Track 1 (`val_reg_inner`) Elo (`R882 = 1500`) | Track 2 (`U_4`) Family Mean | Track 1 (`val_reg_inner`) Family Mean | Unseen-Template Track 2 (`U_4`) Elo (`PRIOR = 1500`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CTX` (`W1` context-aware GBDT prior) | `1569.76` | `1533.21` | `1573.99` | `1494.85` | `1.680362` | `1.317445` | `1548.93` |
| `HEDGE_CTX` (`PRIOR` top-2 + `CTX` top-2) | `1551.88` | `1515.33` | `1500.00` | `1420.86` | `1.686021` | `1.182765` | `1532.86` |
| `R882_CONTINUITY` (16D linear ridge baseline) | `1536.55` | `1500.00` | **`1579.14`** | **`1500.00`** | `1.609212` | **`1.355472`** | — |
| **`TREAT_FAMBAL` (registered family-balanced rank-8 bilinear)** | **`1524.97`** | **`1488.42`** | `1524.14` | `1445.00` | **`1.662061`** | `1.219924` | `1521.19` |
| `NEAR_CHAMP_TREAT` (passing top-32 near-champion bilinear) | `1524.43` | `1487.88` | `1526.42` | `1447.28` | `1.656972` | `1.220847` | `1522.71` |
| `TREAT` (equal-context rank-8 bilinear, `H83`/`H92`) | `1524.30` | `1487.75` | `1529.69` | `1450.55` | `1.658271` | `1.237736` | `1525.50` |
| `HEDGE_NEAR_CHAMP` (`PRIOR` top-2 + `NEAR_CHAMP_TREAT` top-2) | `1521.16` | `1484.61` | `1500.00` | `1420.86` | `1.657804` | `1.182765` | `1520.15` |
| `HEDGE_TREAT_FAMBAL` (`PRIOR` top-2 + `TREAT_FAMBAL` top-2) | `1519.26` | `1482.71` | `1500.00` | `1420.86` | `1.660032` | `1.182765` | `1518.33` |
| `SHUF` (context-shuffled bilinear placebo) | `1509.07` | `1472.52` | `1517.90` | `1438.76` | `1.654776` | `1.227244` | `1505.38` |
| `PRIOR` (`W1` program-only GBDT prior) | `1500.00` | `1463.45` | `1500.00` | `1420.86` | `1.649797` | `1.182765` | `1500.00` |
| `FIT_PROG_MEAN` (exact-SHA empirical-Bayes residual + `PRIOR`) | `1495.13` | `1458.58` | `1497.84` | `1418.71` | `1.660159` | `1.159549` | `1500.00` |
| `FIT_TEMPLATE_MEAN` (AST-template empirical-Bayes residual + `PRIOR`) | `1493.42` | `1456.87` | `1497.90` | `1418.76` | `1.655063` | `1.169970` | `1500.00` |
| `RETR` (`k=5` `lm48_z` champion retrieval cosine) | `1459.58` | `1423.03` | `1489.23` | `1410.09` | `1.559147` | `0.918211` | `1433.08` |

### Gate Verdicts Across the 4 Evaluation Regimes

| Regime | Verdict | Reason |
| --- | --- | --- |
| **Cached Pools — Track 2 Verified Search (`U_4`)** | **`ADVANCE_CACHED_SCREEN (TREAT_FAMBAL - PRIOR)`** | `TREAT_FAMBAL - PRIOR` achieves `+24.97` Elo (`1524.97` vs `1500.00`), family mean `+0.012264`, Bonferroni `.0125` LCB `+0.004938 > +0.001` (one-sided `.05` LCB `+0.006731`), median `0.000000 >= 0`, and reduces family harm ($< -0.01$) from `13.17%` (equal-context `TREAT`) to **`9.27%` (`19 / 205 <= 10%`)**. |
| **Cached Pools — Track 1 Legacy Single-Pick (`val_reg_inner`)** | **`NO_ADVANCE (primary gate not met)`** | `TREAT_FAMBAL - PRIOR` has `+24.14` Elo (`1524.14`) and family mean `+0.037160`, but `21.95%` of families lose $> 0.01$ (`> 10%`), and `R882_CONTINUITY` (`1579.14` Elo, `1.355472`) dominates single-pick selection. |
| **Cached Pools — Within-Context Unseen AST Templates (`U_4`)** | **`NO_ADVANCE (primary gate not met)`** | On unseen AST templates (`91.95%` of unrevealed slots), `TREAT_FAMBAL - PRIOR` drops to `+21.19` Elo (`1521.19`), mean `+0.008730`, Bonferroni `.0125` LCB `+0.000639 < +0.001`, and family harm rises to `10.24%` (`21 / 205 > 10%`). In addition, `TREAT_FAMBAL - SHUF` fails the `+0.001` LCB gate on both all-slots (`.05` LCB `+0.000463`, harm `16.10%`) and unseen templates (`.05` LCB `-0.000431`). |
| **Prospective New Programs (`H94`/`H95` Re-Scored with True `W1` Models)** | **`NO_ADVANCE (primary gate not met)`** | On `50` contexts / `44` families (`150` context-seeds, `556` natively evaluated proposals), true persisted `W1` outer models (`H85`) confirm `H94`'s negative finding: `TRUE_W1_PRIOR_EVAL_SUBSET` reaches `1508.79` Elo (`+8.79` vs `H94_PRIOR_top4 = 1500.0`), while `TRUE_W1_HEDGE_CTX_EVAL_SUBSET` (`1431.99` Elo, mean `-0.014417`, `.025` LCB `-0.042997`, harm `15.91%`) and `H94_HEDGE_CTX` (`1451.29` Elo, mean `-0.010545`, `.025` LCB `-0.040062`, harm `13.64%`) lose to `PRIOR`. |

---

## 2. Why `H100` Was Needed & What It Adds Beyond `H83` / `H84` / `H92` / `H94`

1. **Resolves the `H72`/`H84` vs `W1`/`H83` Fold-Manifest Discrepancy**:
   - `H72` and `H84` used `w6-roster-prep-res03-20261031T0451Z-d4/fold_manifest.json` (`SHA-256: 129002bc...`, generated via Python `random.Random(4900)`), whereas `W1`, `H83`, `H92`, and `H94` used `w1/fold_manifest.json` (`SHA-256: dd211134...`, generated via `np.random.default_rng(4900)`). The two manifests assign the same fold index `k` to only `261 / 1,219` contexts (`21.41%`).
   - Because `W1`'s out-of-fold priors (`PRIOR` and `CTX`) are fit on `W1` folds, evaluating transfer models stacked on `PRIOR` under `H72` folds would leak cross-fold `W1` prior information. `H100` fits all stacked models strictly on `W1` folds (`dd211134...`) while simultaneously computing exact-ID coverage under both fold manifests:
     - **On `H72` folds (`129002bc...`, all slots)**: `21,783 + 18,793 + 23,126 + 18,728 + 16,390 = 98,820 / 1,404,287` (`7.0370%`), reproducing `H84`'s exact slot counts on every single fold (`0..4`).
     - **On `W1` folds (`dd211134...`, unrevealed slots `4..N`)**: `18,591 + 17,332 + 21,419 + 22,181 + 18,568 = 98,091 / 1,399,411` (`7.0094%`), reproducing `H83`'s exact unrevealed coverage.
2. **Tests Registered Family-Balanced Bilinear Transfer (`TREAT_FAMBAL`) vs Equal-Context `TREAT`**:
   - `PROPOSALS.md` (line 147) specified family-balanced weighting ($1/|F_g|$ per family) for the rank-8 bilinear residual model, whereas `H83` and `H92` used equal-context ridge weighting. `H100` implements both and proves that family-balanced weighting is the exact reason `TREAT_FAMBAL` clears the `<= 10%` family-harm gate on cached pools (`9.27%` vs `13.17%`).
3. **Tests Structural AST-Template Transfer (`FIT_TEMPLATE_MEAN`) and the Seen-vs-Unseen Template Kill Rule**:
   - Canonicalizes all `1,404,287` DSL expressions by stripping numeric leaf literals (`1,316,986` unique exact SHAs $\to$ `1,305,507` unique structural AST templates) and evaluates within-context paired Seen-Template vs Unseen-Template `U_4` and `val_reg_inner` across all `1,219` contexts.
4. **Re-Scores `H94` Prospective New-Program Proposals with Persisted `W1` Outer Models (`H85`) & Audits `H95` HWMs**:
   - Replaces `H94`'s in-script refits with `W1`'s true persisted outer-fold models (`w1_outer_fold_models.pkl` from `H85`, `SHA-256: 2ebc4859...`) across all `150` context-seeds (`2,400` regenerated proposals, `556` native evaluations) and measures recall of `H95`'s `30` resplit-confirmed (`801..803`) and `27` material-confirmed high-water marks.
5. **Enforces a Fail-Closed Completeness Gate (`analyze_w6_panel`)**:
   - Verifies exact seed sets (`set(seeds) == expected_seeds` and `len(seeds) == len(expected_seeds)`), duplicate `(context, seed, arm)` rejection, incomplete-row rejection, and proposal shortfall rejection, covered by 6 unit tests in `test_w6_transfer_prep.py`.

---

## 3. Primary & Secondary Contrast Tables (Cached Pools, `1,219` Contexts / `205` Families)

### 3.1 Track 2 Verified Search (`U_4`) Contrasts (`family_mean_U0 = 1.458233`)

Primary contrasts use a 4-way Bonferroni-adjusted one-sided LCB at the `1.25`th percentile (`alpha = 0.05 / 4 = 0.0125`); secondary contrasts report the one-sided `5.0`th percentile LCB.

| Contrast (`A - B`) | Type | Elo $\Delta$ (`PRIOR=1500`) | Family Mean $\Delta$ | Family Median $\Delta$ | Adj. One-Sided LCB | 95% Bootstrap CI | Harm ($< -0.01$) | Family W / T / L | Exact Tie % | Common Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| **`TREAT_FAMBAL - PRIOR`** | **Primary** | **`+24.97`** | **`+0.012264`** | `0.000000` | **`+0.004938`** | `[+0.005743, +0.019214]` | **`9.27%`** | `89 / 67 / 49` | `32.68%` | **`PASS`** |
| `TREAT_FAMBAL - RETR` | Primary | `+65.39` | `+0.102915` | `+0.005873` | `+0.039653` | `[+0.044560, +0.182658]` | `15.12%` | `138 / 7 / 60` | `3.41%` | `FAIL` (harm) |
| `FIT_PROG_MEAN - PRIOR` | Primary | `-4.87` | `+0.010361` | `0.000000` | `-0.005800` | `[-0.004777, +0.031014]` | `12.68%` | `43 / 108 / 54` | `52.68%` | `FAIL` |
| `HEDGE_NEAR_CHAMP - PRIOR` | Primary | `+21.16` | `+0.008007` | `0.000000` | `+0.000831` | `[+0.001674, +0.015203]` | `8.78%` | `89 / 72 / 44` | `35.12%` | `FAIL` (LCB) |
| `HEDGE_TREAT_FAMBAL - PRIOR` | Secondary | `+19.26` | `+0.010235` | `0.000000` | `+0.005677` | `[+0.004839, +0.016308]` | `9.76%` | `79 / 75 / 51` | `36.59%` | `PASS` (sec.) |
| `HEDGE_CTX - PRIOR` (`H92` ref) | Secondary | `+51.88` | `+0.036223` | `+0.002615` | `+0.021008` | `[+0.018853, +0.058727]` | `8.29%` | `134 / 22 / 49` | `10.73%` | `PASS` (sec.) |
| `TREAT - PRIOR` (`H83` ref) | Secondary | `+24.30` | `+0.008473` | `0.000000` | `+0.003755` | `[+0.002863, +0.014259]` | `13.17%` | `90 / 59 / 56` | `28.78%` | `FAIL` (harm) |
| `TREAT_FAMBAL - SHUF` | Secondary | `+15.90` | `+0.007285` | `0.000000` | `+0.000463` | `[-0.000600, +0.016418]` | `16.10%` | `86 / 42 / 77` | `20.49%` | `FAIL` |
| `NEAR_CHAMP_TREAT - PRIOR` | Secondary | `+24.43` | `+0.007174` | `0.000000` | `+0.001234` | `[+0.000263, +0.014801]` | `11.71%` | `89 / 61 / 55` | `29.76%` | `FAIL` (harm) |
| `NEAR_CHAMP_TREAT - SHUF` | Secondary | `+15.36` | `+0.002195` | `0.000000` | `-0.004429` | `[-0.005616, +0.011237]` | `16.59%` | `83 / 54 / 68` | `26.34%` | `FAIL` |
| `FIT_TEMPLATE_MEAN - PRIOR` | Secondary | `-6.58` | `+0.005266` | `0.000000` | `-0.009857` | `[-0.012136, +0.029521]` | `12.20%` | `38 / 113 / 54` | `55.12%` | `FAIL` |
| `CTX - PRIOR` | Secondary | `+69.76` | `+0.030564` | `+0.003986` | `+0.009645` | `[+0.006040, +0.058652]` | `16.10%` | `134 / 8 / 63` | `3.90%` | `FAIL` (harm) |
| `PRIOR - R882_CONTINUITY` | Secondary | `-36.55` | `+0.040585` | `-0.000520` | `+0.013443` | `[+0.009041, +0.076685]` | `33.66%` | `100 / 2 / 103` | `0.98%` | `FAIL` |
| `HEDGE_CTX - R882_CONTINUITY` | Secondary | `+15.33` | `+0.076809` | `+0.003000` | `+0.044018` | `[+0.038700, +0.120986]` | `21.46%` | `119 / 8 / 78` | `3.90%` | `FAIL` (harm) |

### 3.2 Track 1 Legacy Single-Pick (`val_reg_inner`) Contrasts

Note that any top-2 prior hedge (`HEDGE_CTX`, `HEDGE_NEAR_CHAMP`, `HEDGE_TREAT_FAMBAL`) places `PRIOR`'s #1 pick in slot 0 of its 4-pick slate, so its single-pick `val_reg_inner` is identically equal to `PRIOR` (`1.182765`, `1500.00` Elo).

| Contrast (`A - B`) | Type | Elo $\Delta$ (`PRIOR=1500`) | Family Mean $\Delta$ | Family Median $\Delta$ | Adj. One-Sided LCB | Harm ($< -0.01$) | Family W / T / L | Common Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `TREAT_FAMBAL - PRIOR` | Primary | `+24.14` | `+0.037160` | `0.000000` | `-0.004788` | `21.95%` | `86 / 64 / 55` | `FAIL` |
| `TREAT_FAMBAL - RETR` | Primary | `+34.91` | `+0.301713` | `+0.004416` | `+0.175341` | `35.12%` | `124 / 3 / 78` | `FAIL` |
| `FIT_PROG_MEAN - PRIOR` | Primary | `-2.16` | `-0.023216` | `0.000000` | `-0.082948` | `14.15%` | `38 / 116 / 51` | `FAIL` |
| `TREAT - PRIOR` | Secondary | `+29.69` | `+0.054971` | `0.000000` | `+0.024286` | `19.51%` | `90 / 57 / 58` | `FAIL` (harm) |
| `CTX - PRIOR` | Secondary | `+73.99` | `+0.134680` | `+0.008074` | `+0.072578` | `24.88%` | `135 / 5 / 65` | `FAIL` (harm) |
| `PRIOR - R882_CONTINUITY` | Secondary | `-79.14` | `-0.172707` | `-0.043248` | `-0.256120` | `55.12%` | `85 / 0 / 120` | `FAIL` |

### 3.3 Within-Context Seen vs Unseen AST-Template Stratum Decomposition (`U_4`)

All `1,219` `DL-Train` contexts (`205` families) contain at least 4 seen-template unrevealed candidates and at least 4 unseen-template unrevealed candidates (`112,716 / 1,399,411 = 8.0545%` seen-template unrevealed slots; `1,286,695 / 1,399,411 = 91.9455%` unseen-template unrevealed slots).

| Contrast (`A - B`) | Seen-Template Elo $\Delta$ | Seen-Template Mean $\Delta$ | Seen-Template Adj. LCB | Seen-Template Harm % | Unseen-Template Elo $\Delta$ | Unseen-Template Mean $\Delta$ | Unseen-Template Adj. LCB | Unseen-Template Harm % | Unseen Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TREAT_FAMBAL - PRIOR` | `+0.41` | `+0.004835` | `+0.000286` | `3.90%` | `+21.19` | `+0.008730` | `+0.000639` | `10.24%` | `FAIL` |
| `HEDGE_TREAT_FAMBAL - PRIOR` | `+2.03` | `+0.005043` | `+0.001561` | `2.44%` | `+18.33` | `+0.009085` | `+0.003751` | `10.24%` | `FAIL` |
| `TREAT - PRIOR` | `+2.04` | `+0.005458` | `+0.001615` | `4.39%` | `+25.50` | `+0.010497` | `+0.003757` | `12.20%` | `FAIL` |
| `HEDGE_CTX - PRIOR` | `+11.41` | `+0.006529` | `-0.000292` | `11.71%` | `+32.86` | `+0.010228` | `-0.001257` | `14.15%` | `FAIL` |
| `TREAT_FAMBAL - SHUF` | `+0.00` | `+0.004916` | `+0.000498` | `6.34%` | `+15.81` | `+0.007354` | `-0.000431` | `14.63%` | `FAIL` |

---

## 4. Prospective `H94` / `H95` Re-Scoring with Persisted `W1` Outer-Fold Models (`H85`)

`H94` evaluated new live-proposer candidates (`16` new programs per context-seed across `50` contexts / `44` families, seeds `701..703` = `150` context-seeds, `556` unique native evaluations) using in-script refits (`models.pkl` `c8bfbe94...`) because `W1`'s original fitted outer-fold models had not been persisted. `H85` subsequently remediated `W1` and persisted the exact calibrated outer-fold models (`w1_outer_fold_models.pkl`, `SHA-256: 2ebc4859...`, verified at `0.0` max absolute utility difference against `w1_train_only_priors.npz`).

`H100` regenerates all `150` `H94` proposal streams (verifying 100% SHA parity on every revealed program) and re-scores all `2,400` prospective proposals with `H85`'s true persisted `W1` outer-fold models:

1. **Top-4 Selection Overlap (`True W1` vs `H94` Refits across `150` Context-Seeds)**:
   - `PRIOR_top4` mean top-4 set overlap: **`76.67%`**
   - `HEDGE_CTX` mean top-4 set overlap: **`70.50%`**
   - `CTX_top4` mean top-4 set overlap: **`65.83%`**
   - Context-seeds where all 12 picks (`TRUE_W1_PRIOR_top4`, `TRUE_W1_CTX_top4`, `TRUE_W1_HEDGE_CTX`) were already natively evaluated inside `H94`'s `556` fits: **`51 / 150`** (`34.0%`).
2. **Evaluated-Subset Panel (`50` Contexts / `44` Families, Seeds `701..703`)**:
   - `TRUE_W1_PRIOR_EVAL_SUBSET`: **`1508.79` Elo**, family mean `2.080306` (`-0.000627` vs `H94_PRIOR_top4`, `59.09%` exact family ties).
   - `H94_PRIOR_top4` (anchor): **`1500.00` Elo**, family mean `2.080933`.
   - `H94_HEDGE_CTX`: **`1451.29` Elo**, family mean `2.070388` (`-0.010545` vs `H94_PRIOR_top4`, `.025` LCB `-0.040062`, harm `13.64%`, W/T/L `13 / 11 / 20`).
   - `TRUE_W1_HEDGE_CTX_EVAL_SUBSET`: **`1431.99` Elo**, family mean `2.065889` (`-0.014417` vs `TRUE_W1_PRIOR_EVAL_SUBSET`, `.025` LCB `-0.042997`, harm `15.91%`, W/T/L `6 / 19 / 19`).
   - `H94_CTX_top4`: **`1425.97` Elo**, family mean `2.059191`.
   - `TRUE_W1_CTX_EVAL_SUBSET`: **`1424.64` Elo**, family mean `2.055858`.
3. **Recall of `H95` Resplit-Confirmed (`30`) and Material-Confirmed (`27`) High-Water Marks**:
   - Note that `H95`'s `30` confirmed HWMs were discovered inside the union of programs natively evaluated by `H94`'s three refit arms (`556 / 2,400` proposals); any `TRUE_W1_*` top-4 pick outside `H94`'s evaluated union was never natively fit or resplit-tested in `H95`.
   - Within `H94`'s evaluated union:
     - `H94_CTX_top4`: recalls **`19 / 30` (`63.33%`)** confirmed HWMs and **`19 / 27` (`70.37%`)** material HWMs.
     - `H94_HEDGE_CTX`: recalls **`19 / 30` (`63.33%`)** confirmed HWMs and **`19 / 27` (`70.37%`)** material HWMs (`+4` material HWMs over `H94_PRIOR_top4`).
     - `H94_PRIOR_top4`: recalls **`18 / 30` (`60.00%`)** confirmed HWMs and **`15 / 27` (`55.56%`)** material HWMs.
     - Un-refit `TRUE_W1_PRIOR_top4` / `TRUE_W1_CTX_top4` / `TRUE_W1_HEDGE_CTX` (evaluated only where their top-4 overlapped `H94`'s fits): recall `13 / 30` (`11 / 27` material), `9 / 30` (`9 / 27` material), and `7 / 30` (`6 / 27` material) respectively.

---

## 5. 4-Node CPU Execution Receipts (`<= 85%` CPU/Host, `0` GPU)

| Host | Outer Fold(s) | W1 Held Contexts / Families | W1 Unrevealed Seen Exact Slots | H72 Seen Exact Slots (`H84` Parity) | Priority / Container | Wall Seconds | CPU Seconds | Shard JSON SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `xbabe1` | Fold `0` | `227` / `41` | `18,591 / 261,806` | `21,783 / 287,648` | `nice -n 19` (102 cores) | `74.28s` | `101.37s` | `3b20258968d0a2a37196e9f1ff37527dbe92a53c2136c98ca6fe6b0ff36c345f` |
| `xbabe3` | Fold `1` | `234` / `41` | `17,332 / 267,565` | `18,793 / 270,583` | `nice -n 10` (102 cores) | `14.60s` | `35.78s` | `3c5aadbe238f85e06093bb7665dda7d2f22dab19d4400e6645c4a186db8dcc2f` |
| `xbabe3` | Fold `2` | `269` / `41` | `21,419 / 309,500` | `23,126 / 294,404` | `nice -n 10` (102 cores) | `14.44s` | `36.91s` | `04069e8cab6fe8321d60fc5cb101ccd84455077a928f8a78b4004afcd92db59c` |
| `xbabe4` | Fold `3` | `271` / `41` | `22,181 / 311,488` | `18,728 / 287,976` | `nice -n 10`, Colima `jope-ci` (`jope-elo-worker:arm64`) | `12.65s` | `12.65s` | `99d8ac40fc515dd992aff7c9c4cee12e4ca63ddd9d0dde343ab39c7e9950c9e1` |
| `xbabe2` | Fold `4` | `218` / `41` | `18,568 / 249,052` | `16,390 / 263,676` | `nice -n 10` (102 cores) | `92.41s` | `114.28s` | `a05b0d6b914bbecf499065b7582ad60434713fe52e14b019e712411654f2d86e` |

---

## 6. S3 Custody Receipts (Create-Only `put-object --if-none-match '*'`, Readback Verified)

Run ID: `argon-h98-proposer-transfer-20261031T0938Z`

- **Lowercase `w6/` Prefix (per `rd_argon_t0325.md`)**:
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w6/argon-h98-proposer-transfer-20261031T0938Z/manifest.json`
    - `VersionId`: `pxkd5g.nfU_zG_ShoAvyRYZOookpwRv8`
    - `SHA-256`: `e73cb69429943887e0450b67fda39c20fa37d254fcd81c8d9e468882ca442657` (`14,519` bytes)
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w6/argon-h98-proposer-transfer-20261031T0938Z/metrics.json`
    - `VersionId`: `PqrsUh5VE1NL2wrZFVZ2w31DhSS5X762`
    - `SHA-256`: `0ca473ed3de4ad0012148bb9f670e5a4562888617879a7285aea7b7899efe47e` (`85,336` bytes)
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w6/argon-h98-proposer-transfer-20261031T0938Z/per_context_vectors.json`
    - `VersionId`: `gPqzXGY6YpcRh_31h.ip9IsDt7.Ma8RT`
    - `SHA-256`: `9875492fc764da9dab0b7578f724415a847d4ff535c585e17dbc0efe0a69093e` (`1,435,174` bytes)
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w6/argon-h100-proposer-transfer-20261031T1003Z/metrics.json`
    - `VersionId`: `rggwL25qG51LmaUHGll3PXSSbB2IGLb4`
    - `SHA-256`: `b3d7e2eae1d183a91f11b4b9dc5067811028a6e534dee5c5256ee202b6f9a6fa` (`92,786` bytes)
- **Uppercase `W6/` Prefix (per `rd_cpu_s3_t2101.md`)**:
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/argon-h98-proposer-transfer-20261031T0938Z/manifest.json`
    - `VersionId`: `JUHdvdk4LnpDMBcK3hbpH7fkoNTqCZxL`
    - `SHA-256`: `e73cb69429943887e0450b67fda39c20fa37d254fcd81c8d9e468882ca442657` (`14,519` bytes)
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/argon-h98-proposer-transfer-20261031T0938Z/metrics.json`
    - `VersionId`: `nvPezCrO7VPD8NGrTvuxMHNWTmXO28rQ`
    - `SHA-256`: `0ca473ed3de4ad0012148bb9f670e5a4562888617879a7285aea7b7899efe47e` (`85,336` bytes)
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/argon-h98-proposer-transfer-20261031T0938Z/per_context_vectors.json`
    - `VersionId`: `hLKyT9mqiSIFe_lpTMh66VI5zTqe.Cgd`
    - `SHA-256`: `9875492fc764da9dab0b7578f724415a847d4ff535c585e17dbc0efe0a69093e` (`1,435,174` bytes)
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/argon-h100-proposer-transfer-20261031T1003Z/metrics.json`
    - `VersionId`: `aa1AWHQsHvGHqB2UUdDsLcJdkfdb1Hgl`
    - `SHA-256`: `b3d7e2eae1d183a91f11b4b9dc5067811028a6e534dee5c5256ee202b6f9a6fa` (`92,786` bytes)

## 7. Reproduction & Verification Commands

```bash
# Unit test suite (6 tests covering exact-ID shrinkage, AST template canonicalization,
# family-balanced vs equal-context bilinear ridge, near-champion ridge, top-2 hedge,
# and fail-closed completeness gates):
nice -n 10 pytest -q research/flywheel_selfplay/h100_w6_proposer_transfer_prep/test_w6_transfer_prep.py

# Lint check:
nice -n 10 ruff check research/flywheel_selfplay/h100_w6_proposer_transfer_prep/
```
