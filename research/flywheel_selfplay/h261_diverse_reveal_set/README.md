# H261: Diversity-Aware 4-Reveal Selection (`h261_diverse_reveal_set`) — `NO_ADVANCE`

- **Wave**: `H261` (`h261-diverse-reveal-set-20261103` / `h261_diverse_reveal_set`)
- **Owner**: Argon
- **Status**: `COMPLETE`
- **Verdict**: **`NO_ADVANCE`** (both Primary `MMR4_CV - TOP4_PICK` and Secondary `STRUCT4 - TOP4_PICK` are negative on held-out `DL-Val` `Native91`)
- **Claim Scope / Taxonomy**: **Equal-budget 4-reveal test-time search policy comparison (`4` reveals on gated contexts `top_zgap < 0.25`, `0` elsewhere; `0` optimizer updates, CPU-only). NOT a zero-reveal checkpoint lift.**
- **Preregistration Commit**: `f5f41629ca5feef2321b4560eb06497bd988f601` (with outcome-free dict-key parsing fix `8e470d4237bd1ce2784cb818790604b36dec6ca0`, `0` outcomes read; S3 manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H261/prereg-f5f41629ca5f/prereg_s3_manifest.json`)
- **Primary Run Artifact Prefix**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H261/attempt-002/`

---

## 1. Executive Summary

H261 tested whether the non-adaptive 4-reveal baseline `TOP4_PICK` (`r882` top-4 by predicted cost on gated contexts `top_zgap < 0.25`) suffers from candidate redundancy and whether diversity-aware 4-reveal selection from `r882` `top-16`—either via Maximal Marginal Relevance on standardized 40D candidate embeddings (`MMR4_CV`, primary) or structural diversification across learner families and DSL operator heads (`STRUCT4`, secondary)—improves revealed winner utility at a strictly matched 4-reveal budget (`0` optimizer updates).

1. **Bit-Exact Baseline Reproduction (`Step 1 & Step 2b`)**:
   - `R882` (`0` reveals) reproduced `Native91` mean context utility **`1.1377941898164525`** bit-exact (`91/91` contexts, `abs_diff = 0.0`).
   - `TOP4_PICK` (`4` reveals on gated contexts) reproduced H258 `Native91` mean context utility **`1.2403601682250796`** (`abs_diff = 0.0`) and mean family utility **`1.2519997168684853`** (`abs_diff = 0.0`) bit-exact prior to evaluating diversity arms.
2. **In-Distribution `DL-Train` OOF (`1,219` contexts / `205` families, 5-fold family-grouped CV, `fold_seed = 261140`)**:
   - Every training fold selected `lambda* = 1.0` from `{0.1, 0.2, 0.3, 0.5, 1.0}` (`frozen_dl_val_lambda = 1.0`).
   - On `DL-Train` OOF, both diversity policies beat `TOP4_PICK`:
     - `MMR4_CV - TOP4_PICK`: median family gain **`+0.006127`** (`95% CI [+0.002951, +0.009881]`, Holm `97.5% CI [+0.002591, +0.010514]`, `Holm p = 0.00020`, family W/T/L `69/90/46`).
     - `STRUCT4 - TOP4_PICK`: median family gain **`+0.025811`** (`95% CI [+0.009538, +0.044202]`, Holm `97.5% CI [+0.007362, +0.046646]`, `Holm p = 0.00030`, family W/T/L `111/11/83`).
3. **Held-Out `DL-Val` (`Native91`, `91` contexts / `19` families) — `NO_ADVANCE`**:
   - Neither `MMR4_CV` nor `STRUCT4` generalized its `DL-Train` OOF gain to `DL-Val`:
     - **Primary (`MMR4_CV - TOP4_PICK`, `Holm m = 2`)**: median family gain **`-0.000320`** (`mean = -0.000320`, context gain `-0.000334`, `95% CI [-0.002240, +0.001407]`, Holm `97.5% CI [-0.002573, +0.001642]`, `raw p = 0.6241`, `Holm p = 1.0`, `0/3` seeds positive, family W/T/L `3/13/3`, harm rate `5.26%` (`1/19` family `< -0.01`)). Across smaller penalties `lambda in {0.1, 0.2, 0.3, 0.5}`, `MMR4` selected the exact same winners as `TOP4_PICK` (`0.0` gain, W/T/L `0/19/0`).
     - **Secondary (`STRUCT4 - TOP4_PICK`, `Holm m = 2`)**: median family gain **`-0.003366`** (`mean = -0.003366`, context gain `-0.003513`, `95% CI [-0.017231, +0.008492]`, Holm `97.5% CI [-0.019507, +0.010111]`, `raw p = 0.6888`, `Holm p = 1.0`, `0/3` seeds positive, family W/T/L `10/2/7`, harm rate `31.58%` (`6/19` families `< -0.01`, worst `fam-4286560c...` `-0.095968`)).
4. **Mechanism Diagnosis**:
   - On `DL-Val`, forcing structural diversity in `STRUCT4` successfully lowers `emb40` pairwise cosine similarity from `0.5833` (`TOP4_PICK`) to `0.4009` and increases mean distinct learner families from `1.473` to `2.264` and distinct DSL heads from `2.659` to `3.846`.
   - However, replacing `r882`'s `rank-2..4` candidates within the preferred learner family with lower-ranked (`rank 5..15`) candidates from alternative learner families **reduces `top-16` oracle `recall@4` from `32.97%` (`30/91`) to `25.27%` (`23/91`)** and **reduces full-pool oracle `recall@4` from `6.59%` (`6/91`) to `4.40%` (`4/91`)**. Greedy `TOP4_PICK` remains the strongest equal-budget 4-reveal baseline on `DL-Val`.

---

## 2. Primary & Secondary Contrasts (`Holm m = 2`, `10,000` Paired Family Bootstrap Draws, `boot_seed = 261150`)

| Split | Contrast | Median Fam Gain | Mean Fam Gain | Median Ctx Gain | Raw 95% CI | Holm 97.5% CI (`m=2`) | Raw `p` (1-sided) | Holm `p` (`m=2`) | Pos Seeds | Family W/T/L | Harm `>0.01` | Verdict |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| `DL-Val` (`91` ctx / `19` fam) | **`MMR4_CV - TOP4_PICK` (Primary)** | **`-0.000320`** | `-0.000320` | `-0.000334` | `[-0.002240, +0.001407]` | `[-0.002573, +0.001642]` | `0.6241` | `1.0000` | `0/3` | `3 / 13 / 3` | `5.26%` (`1/19`) | **`NO_ADVANCE`** |
| `DL-Val` (`91` ctx / `19` fam) | **`STRUCT4 - TOP4_PICK` (Secondary)** | **`-0.003366`** | `-0.003366` | `-0.003513` | `[-0.017231, +0.008492]` | `[-0.019507, +0.010111]` | `0.6888` | `1.0000` | `0/3` | `10 / 2 / 7` | `31.58%` (`6/19`) | **`NO_ADVANCE`** |
| `DL-Val` (`91` ctx / `19` fam) | `STRUCT4_EST_ONLY - TOP4_PICK` | `-0.001325` | `-0.001325` | `-0.001353` | `[-0.008118, +0.006244]` | `[-0.009048, +0.007400]` | `0.6621` | — | `0/3` | `6 / 4 / 9` | `26.32%` (`5/19`) | Descriptive |
| `DL-Val` (`91` ctx / `19` fam) | `RAND4_OF16 - TOP4_PICK` | `-0.002701` | `+0.002080` | `-0.002790` | `[-0.015409, +0.016474]` | `[-0.018353, +0.018135]` | `0.3816` | — | `1/3` | `11 / 0 / 8` (s0) | `21.05%` (`4/19`) | Descriptive |
| `DL-Val` (`91` ctx / `19` fam) | `MMR4@r3175 - TOP4_PICK@r3175` | `-0.000040` | `-0.000040` | `-0.000041` | `[-0.001473, +0.001441]` | `[-0.001663, +0.001657]` | `0.5318` | — | `0/3` | `2 / 14 / 3` | `0.00%` (`0/19`) | Descriptive |
| `DL-Val` (`91` ctx / `19` fam) | `STRUCT4@r3175 - TOP4_PICK@r3175` | `+0.000752` | `+0.000752` | `+0.000786` | `[-0.016151, +0.017742]` | `[-0.018409, +0.020548]` | `0.4730` | — | `3/3`* | `9 / 2 / 8` | `31.58%` (`6/19`) | Descriptive |
| `DL-Val` (`91` ctx / `19` fam) | `TOP4_PICK - R882` (4-rev vs 0-rev) | `+0.098384` | `+0.098384` | `+0.102566` | `[+0.059711, +0.146769]` | `[+0.055560, +0.153950]` | `0.0001` | — | `3/3`* | `19 / 0 / 0` | `0.00%` (`0/19`) | Reference |
| `DL-Train` OOF (`1,219` ctx / `205` fam) | `MMR4_CV - TOP4_PICK` | `+0.006127` | `+0.006127` | `+0.004844` | `[+0.002951, +0.009881]` | `[+0.002591, +0.010514]` | `0.0001` | `0.0002` | `3/3`* | `69 / 90 / 46` | `2.44%` (`5/205`) | OOF Reference |
| `DL-Train` OOF (`1,219` ctx / `205` fam) | `STRUCT4 - TOP4_PICK` | `+0.025811` | `+0.025811` | `+0.027583` | `[+0.009538, +0.044202]` | `[+0.007362, +0.046646]` | `0.0003` | `0.0003` | `3/3`* | `111 / 11 / 83` | `14.63%` (`30/205`) | OOF Reference |
| `DL-Train` OOF (`1,219` ctx / `205` fam) | `STRUCT4_EST_ONLY - TOP4_PICK` | `+0.011446` | `+0.011446` | `+0.012069` | `[-0.000393, +0.027844]` | `[-0.001387, +0.030474]` | `0.0315` | — | `3/3`* | `65 / 67 / 73` | `13.66%` (`28/205`) | OOF Reference |
| `DL-Train` OOF (`1,219` ctx / `205` fam) | `RAND4_OF16 - TOP4_PICK` | `+0.007973` | `+0.007978` | `+0.012140` | `[-0.013456, +0.028520]` | `[-0.016902, +0.031291]` | `0.2281` | — | `2/3` | `127 / 3 / 75` | `21.46%` (`44/205`) | OOF Reference |

*\*Seed caveat: Seeds `261101–261103` vary only `RAND4_OF16` sampling and permutation RNG. `R882`, `TOP4_PICK`, `MMR4_CV`, and `STRUCT4` are deterministic given `rcost`, `fold_manifest.json` (`fold_seed = 261140`), and frozen candidate embeddings, so `3/3` or `0/3` seeds on deterministic arms constitutes weak seed evidence.*

---

## 3. Arm Utilities & MM Bradley-Terry Elo (`Anchored at TOP4_PICK = 1500.0`)

| Arm | Reveals (`DL-Val` / `DL-Train` Gated) | `DL-Val` Mean Ctx Utility | `DL-Val` Mean Fam Utility | `DL-Val` MM-BT Elo (`95% CI`) | `DL-Train` OOF Mean Ctx Utility | `DL-Train` OOF Mean Fam Utility |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `TOP4_PICK` (`r882` anchor) | `4.000` (`91/91` gated; `1200/1219` train) | **`1.240360`** | **`1.252000`** | **`1500.00`** (`[1500.00, 1500.00]`) | `1.606115` | `1.569233` |
| `MMR4_L0.1..0.5` (`r882`) | `4.000` | `1.240360` | `1.252000` | `1500.00` (`[1500.00, 1500.00]`) | `1.606349–1.607598` | `1.569590–1.570413` |
| `MMR4_CV` (`lambda*=1.0`, Primary) | `4.000` | `1.240026` | `1.251680` | `1499.48` (`[1485.16, 1514.03]`) | `1.610960` | `1.575360` |
| `STRUCT4_EST_ONLY` (`r882`) | `4.000` | `1.239007` | `1.250675` | `1457.58` (`[1415.44, 1492.35]`) | `1.618184` | `1.580679` |
| `RAND4_OF16` (`r882`, median of 3 seeds) | `4.000` | `1.237570` | `1.249298` | `1512.24` (`[1457.43, 1538.88]`) | `1.618255` | `1.577206` |
| `STRUCT4` (`r882`, Secondary) | `4.000` | `1.236847` | `1.248634` | `1459.14` (`[1424.43, 1495.89]`) | `1.633698` | `1.595044` |
| `TOP4_PICK@r3175` | `4.000` | `1.240323` | `1.251964` | `1501.04` (`[1493.80, 1509.16]`) | `1.606920` | `1.570568` |
| `MMR4@r3175` (`lambda*=1.0`) | `4.000` | `1.240281` | `1.251924` | `1498.18` (`[1481.11, 1514.48]`) | `1.609635` | `1.573197` |
| `STRUCT4@r3175` | `4.000` | `1.241109` | `1.252716` | `1452.93` (`[1411.75, 1492.30]`) | `1.633658` | `1.596053` |
| `R882` (`0` reveals) | `0.000` | `1.137794` | `1.153615` | `1158.09` (`[1088.76, 1209.25]`) | `1.423073` | `1.385355` |
| `R3175` (`0` reveals) | `0.000` | `1.136683` | `1.152551` | `1152.15` (`[1080.30, 1206.62]`) | `1.419535` | `1.381525` |

---

## 4. Redundancy & Oracle Recall Diagnostics on Gated Contexts

| Split | Arm | Gated Ctxs | Mean Pairwise Cos (`emb40`) | Mean Pairwise Cos (`sem24`) | Distinct Learner Families (`1..5`) | Distinct DSL Heads (`1..4`) | `top-16` Oracle `Recall@4` | Full-Pool Oracle `Recall@4` | Mean `top-16` Oracle Regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DL-Val` | `TOP4_PICK` | `91` | `0.583297` | `0.999313` | `1.4725` | `2.6593` | **`32.97%`** (`30/91`) | **`6.59%`** (`6/91`) | **`0.062272`** |
| `DL-Val` | `MMR4_L0.1` | `91` | `0.580289` | `0.999296` | `1.4835` | `2.6593` | `32.97%` | `6.59%` | `0.062272` |
| `DL-Val` | `MMR4_L0.3` | `91` | `0.559185` | `0.999260` | `1.5275` | `2.7253` | `32.97%` | `6.59%` | `0.062272` |
| `DL-Val` | `MMR4_L0.5` | `91` | `0.555031` | `0.999252` | `1.5385` | `2.7363` | `32.97%` | `6.59%` | `0.062272` |
| `DL-Val` | `MMR4_CV` (`lambda*=1.0`) | `91` | `0.513824` | `0.999193` | `1.5604` | `2.8462` | `32.97%` | `6.59%` | `0.062607` |
| `DL-Val` | `STRUCT4_EST_ONLY` | `91` | `0.447834` | `0.999056` | `2.2637` | `3.0330` | `27.47%` | `6.59%` | `0.063626` |
| `DL-Val` | `STRUCT4` | `91` | **`0.400948`** | `0.998945` | **`2.2637`** | **`3.8462`** | **`25.27%`** (`23/91`) | **`4.40%`** (`4/91`) | `0.065785` |
| `DL-Val` | `RAND4_OF16` (3-seed mean) | `91` | `0.456136` | `0.999083` | `1.5604` | `3.2125` | `26.37%` | `2.20%` | `0.060116` |
| `DL-Train` | `TOP4_PICK` | `1,200` | `0.606367` | `0.999263` | `1.4300` | `2.5883` | `27.00%` | `0.92%` | `0.112208` |
| `DL-Train` | `MMR4_CV` (`lambda*=1.0`) | `1,200` | `0.528619` | `0.999148` | `1.5325` | `2.7800` | `25.83%` | `1.00%` | `0.107287` |
| `DL-Train` | `STRUCT4` | `1,200` | `0.400265` | `0.998873` | `2.1508` | `3.8642` | `22.75%` | `0.83%` | `0.084189` |

---

## 5. Harm Audit & Falsification Gates

- **Harmed `DL-Val` Families (`delta < -0.01` vs `TOP4_PICK`)**:
  - `MMR4_CV`: `1 / 19` (`5.26%`) — `fam-f8fd51f35678f2c68a7b99cc0dfdbd9ed0e9556bb029cb24c8df3ba1b5c030b4` (`-0.012934`).
  - `STRUCT4`: `6 / 19` (`31.58%`) — `fam-4286560c...` (`-0.095968`), `fam-ce33f2a8...` (`-0.029252`), `fam-469b72b7...` (`-0.021136`), `fam-598bbb77...` (`-0.017869`), `fam-f2893c38...` (`-0.014763`), `fam-d73e184e...` (`-0.011154`).
- **Audit A (Zero Leakage)**: `PASS` (`has_meta_test_any = false`, `dl_val_in_lambda_selection = false`, `dl_train_val_family_overlap = 0`, `dl_train_fold_family_overlap = 0`).
- **Audit B (Matched Reveal Budget)**: `PASS` (all 4-reveal arms use exactly `4` unique reveals on gated contexts and `0` on non-gated contexts).
- **Audit C (Permuted Negative Control)**: `PASS` (sign-permutation null 95% CI `[-0.001886, +0.001886]` covers `0`; permuting fitness labels within the 4-reveal set collapses utility vs `TOP4_PICK` to `-0.135568` for `PERM_MMR4_CV` and `-0.273070` for `PERM_STRUCT4`).
- **Compute & Resource Hygiene**: CPU-only (`CUDA_VISIBLE_DEVICES=""`), pinned to cores `57–60` (`nice 19`), `0` optimizer updates, `attempt-002` wall time `33.22s`, peak RSS `0.92 GiB` (`989,523,968 B`), peak threads `1`, `0` pause transitions.
