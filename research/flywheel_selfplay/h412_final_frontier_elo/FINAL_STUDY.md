# H412 FINAL study: our strongest model vs every frontier LLM on dev91-audit-v1

**Board:** `dev91-audit-v1`, 91 dev contexts / 19 families, board sha256 `40cb4828cf5c9a0d…`. Every entrant fits on the SEARCH rows and predicts the AUDIT rows. u = −ln(RMSE / RMSE of the SEARCH-mean predictor). AUDIT-Elo anchor `h335.H257_REPLICA` = 1500.
**Reads stored scores only.** No model was refit, rerun or rescored. The one new board row is `h412.grok4p7_xhigh`, scored once under the H412 LOCK. Our model is the stored `h364.XSTACK_2L` (owner's choice, 2026-11-06).

## Result

- **Our strongest model, XSTACK_2L, ranks 6 of 25 (AUDIT-Elo 1724.6).**
- **It beats every one of the 5 frontier LLM regressors.** Each contrast passes the H335 primary test (p < 0.025, LOFO > 0).
  - Against the highest-rated of them, Opus5p5_Extra, the RMSE cut is +23.4% and ours wins 13 of 19 families.
  - Across the 5, the RMSE cut runs from +18.3% (Gemini4_argon_high) to +50.2% (grok4p7_xhigh).
- **It does not beat any of the 5 foundation models.** Kumo, TabFM, the TabFM ensemble, TabPFN 3.5 and TabPFN v3 all rank above it.
  - Kumo-Tabular large leads the board: its RMSE is 31.3% lower than ours.
  - Foundation models significantly better than ours (mirrored primary test): Kumo-Tabular large, TabFM ensemble.
- **The new frontier entrant grok4p7_xhigh ranks 25 of 25, last on the board.**
  - The frontier-only order on dev91 is the same as AutoML Elo v2: yes. AutoML Elo is a separate TRAIN panel.

## A. Full ladder (25 entrants, all 91/91)

| Rank | Entrant | Class | AUDIT-Elo | 95% CI | Family-mean u | Geo-mean RMSE / mean baseline |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Kumo-Tabular large (`kumo_tabular_large_e8.v1`) | foundation model | 2045.3 | 1859–2336 | 1.9356 | 0.144 |
| 2 | TabFM ensemble (`tabfm_ensemble.v1`) | foundation model | 1898.5 | 1732–2133 | 1.7675 | 0.171 |
| 3 | TabFM (`tabfm.v1`) | foundation model | 1873.4 | 1706–2104 | 1.7202 | 0.179 |
| 4 | TabPFN 3.5 (`tabpfn_v3p5.v1`) | foundation model | 1820.3 | 1649–2072 | 1.6411 | 0.194 |
| 5 | TabPFN v3 (`tabpfn_v3.v1`) | foundation model | 1785.8 | 1651–1960 | 1.6427 | 0.193 |
| 6 | **XSTACK_2L (ours)** (`h364.XSTACK_2L`) | JopeDime | **1724.6** | 1632–1850 | 1.5607 | 0.210 |
| 7 | H335_AGG (ours) (`h335.H335_AGG`) | JopeDime | 1705.7 | 1618–1823 | 1.5592 | 0.210 |
| 8 | h335.OLD_MATCHED_CV3 (`h335.OLD_MATCHED_CV3`) | JopeDime | 1625.4 | 1563–1705 | 1.4607 | 0.232 |
| 9 | Opus5p5_Extra (`h335.Opus5p5_Extra`) | frontier LLM | 1577.8 | 1462–1698 | 1.2943 | 0.274 |
| 10 | h335.OLD_MATCHED (`h335.OLD_MATCHED`) | JopeDime | 1530.9 | 1478–1593 | 1.4465 | 0.235 |
| 11 | h335.H317_BASE (`h335.H317_BASE`) | JopeDime | 1526.6 | 1492–1565 | 1.4073 | 0.245 |
| 12 | AutoGluon (300 s) (`autogluon_300s.v1`) | AutoML | 1526.0 | 1411–1638 | 1.2524 | 0.286 |
| 13 | Gemini4_argon_high (`h335.Gemini4_argon_high`) | frontier LLM | 1524.4 | 1415–1640 | 1.3587 | 0.257 |
| 14 | h335.H257_REPLICA (`h335.H257_REPLICA`) | JopeDime | 1500.0 | 1500–1500 | 1.3703 | 0.254 |
| 15 | GPT6p1_Sol_Max (`h335.GPT6p1_Sol_Max`) | frontier LLM | 1497.3 | 1412–1585 | 1.3543 | 0.258 |
| 16 | CatBoost (`catboost.v1`) | classic learner | 1480.2 | 1364–1592 | 1.1886 | 0.305 |
| 17 | h335.BANK_NN8_PORTFOLIO (`h335.BANK_NN8_PORTFOLIO`) | JopeDime | 1453.3 | 1381–1527 | 1.2715 | 0.280 |
| 18 | h335.BANK_NN8_PICK (`h335.BANK_NN8_PICK`) | JopeDime | 1421.3 | 1359–1482 | 1.2636 | 0.283 |
| 19 | XGBoost (`xgboost.v1`) | classic learner | 1374.4 | 1263–1482 | 1.0742 | 0.342 |
| 20 | Random forest (`random_forest.v1`) | classic learner | 1366.6 | 1247–1471 | 1.1245 | 0.325 |
| 21 | LightGBM (`lightgbm.v1`) | classic learner | 1316.2 | 1193–1422 | 1.0492 | 0.350 |
| 22 | GPT6_Astra_Max (`h335.GPT6_Astra_Max`) | frontier LLM | 1308.4 | 1175–1429 | 1.0215 | 0.360 |
| 23 | Ridge (`ridge.v1`) | classic learner | 1223.2 | 1009–1364 | 0.9208 | 0.398 |
| 24 | h335.BANK_NN1 (`h335.BANK_NN1`) | JopeDime | 1189.6 | 1084–1273 | 0.9031 | 0.405 |
| 25 | grok4p7_xhigh (`h412.grok4p7_xhigh`) | frontier LLM | 1127.4 | 1004–1222 | 0.8639 | 0.422 |

## B. XSTACK_2L head to head

- **Gain** is the family-equal mean of u(XSTACK_2L) − u(other), with its 95% family-bootstrap interval.
- **RMSE** is the cut 1 − exp(−gain). Positive means ours is better.
- **p** and **LOFO** come from the H335 primary test (clip ±0.5, one-sided exact sign-flip over 19 families).
- **Verdict:** "better" means the test passes. "worse" means the mirrored test passes.

### vs the frontier LLM regressors

| vs | Class | Gain [95% CI] | RMSE | Families W/T/L | p | LOFO min | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Opus5p5_Extra | frontier LLM | +0.2664 [+0.097, +0.476] | +23.4% | 13/0/6 | 0.00066 | +0.1209 | **better** |
| Gemini4_argon_high | frontier LLM | +0.2020 [+0.044, +0.414] | +18.3% | 14/0/5 | 0.0031 | +0.0839 | **better** |
| GPT6p1_Sol_Max | frontier LLM | +0.2064 [+0.061, +0.390] | +18.6% | 18/0/1 | 0.001 | +0.0968 | **better** |
| GPT6_Astra_Max | frontier LLM | +0.5392 [+0.245, +0.970] | +41.7% | 19/0/0 | 1.9e-06 | +0.2280 | **better** |
| grok4p7_xhigh | frontier LLM | +0.6969 [+0.326, +1.158] | +50.2% | 19/0/0 | 1.9e-06 | +0.2524 | **better** |

### vs the foundation models, AutoML and the classic learners

| vs | Class | Gain [95% CI] | RMSE | Families W/T/L | p | LOFO min | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kumo-Tabular large | foundation model | -0.3749 [-0.657, -0.154] | -45.5% | 1/0/18 | 1 | -0.2028 | worse |
| TabFM ensemble | foundation model | -0.2068 [-0.402, -0.042] | -23.0% | 3/0/16 | 0.98 | -0.1568 | worse |
| TabFM | foundation model | -0.1595 [-0.319, -0.017] | -17.3% | 4/0/15 | 0.97 | -0.1451 | not separated |
| TabPFN 3.5 | foundation model | -0.0804 [-0.191, +0.033] | -8.4% | 5/0/14 | 0.93 | -0.0938 | not separated |
| TabPFN v3 | foundation model | -0.0820 [-0.194, +0.050] | -8.5% | 2/0/17 | 0.95 | -0.0971 | not separated |
| AutoGluon (300 s) | AutoML | +0.3083 [+0.088, +0.602] | +26.5% | 18/0/1 | 0.00018 | +0.1153 | **better** |
| CatBoost | classic learner | +0.3722 [+0.127, +0.693] | +31.1% | 17/0/2 | 0.00066 | +0.1383 | **better** |
| XGBoost | classic learner | +0.4865 [+0.194, +0.868] | +38.5% | 18/0/1 | 4.6e-05 | +0.1749 | **better** |
| Random forest | classic learner | +0.4362 [+0.146, +0.837] | +35.3% | 18/0/1 | 1.3e-05 | +0.1582 | **better** |
| LightGBM | classic learner | +0.5115 [+0.232, +0.868] | +40.0% | 18/0/1 | 3.8e-06 | +0.2052 | **better** |
| Ridge | classic learner | +0.6399 [+0.271, +1.118] | +47.3% | 17/0/2 | 8.8e-05 | +0.2388 | **better** |

### vs the other JopeDime entrants

| vs | Class | Gain [95% CI] | RMSE | Families W/T/L | p | LOFO min | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H335_AGG (ours) | JopeDime | +0.0015 [-0.000, +0.003] | +0.2% | 14/3/2 | 0.045 | +0.0011 | not separated |
| h335.OLD_MATCHED_CV3 | JopeDime | +0.1001 [+0.029, +0.199] | +9.5% | 17/0/2 | 0.004 | +0.0411 | **better** |
| h335.OLD_MATCHED | JopeDime | +0.1142 [+0.033, +0.227] | +10.8% | 18/0/1 | 0.0026 | +0.0556 | **better** |
| h335.H317_BASE | JopeDime | +0.1534 [+0.046, +0.295] | +14.2% | 18/0/1 | 0.0027 | +0.0683 | **better** |
| h335.H257_REPLICA | JopeDime | +0.1904 [+0.061, +0.353] | +17.3% | 17/0/2 | 0.001 | +0.0907 | **better** |
| h335.BANK_NN8_PORTFOLIO | JopeDime | +0.2893 [+0.134, +0.478] | +25.1% | 18/0/1 | 4e-05 | +0.1246 | **better** |
| h335.BANK_NN8_PICK | JopeDime | +0.2972 [+0.139, +0.487] | +25.7% | 19/0/0 | 2.9e-05 | +0.1314 | **better** |
| h335.BANK_NN1 | JopeDime | +0.6576 [+0.349, +1.007] | +48.2% | 19/0/0 | 1.9e-06 | +0.2458 | **better** |

### Second row: H335_AGG (ours, the XSTACK_2L input) vs the frontier LLMs

| vs | Class | Gain [95% CI] | RMSE | Families W/T/L | p | LOFO min | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Opus5p5_Extra | frontier LLM | +0.2649 [+0.097, +0.473] | +23.3% | 13/0/6 | 0.00072 | +0.1195 | **better** |
| Gemini4_argon_high | frontier LLM | +0.2005 [+0.043, +0.407] | +18.2% | 14/0/5 | 0.0033 | +0.0823 | **better** |
| GPT6p1_Sol_Max | frontier LLM | +0.2049 [+0.061, +0.397] | +18.5% | 16/0/3 | 0.0011 | +0.0949 | **better** |
| GPT6_Astra_Max | frontier LLM | +0.5377 [+0.245, +0.960] | +41.6% | 19/0/0 | 1.9e-06 | +0.2268 | **better** |
| grok4p7_xhigh | frontier LLM | +0.6953 [+0.328, +1.154] | +50.1% | 19/0/0 | 1.9e-06 | +0.2506 | **better** |

## C. The frontier LLMs: dev91 next to AutoML Elo

- **The two panels answer different questions.**
  - dev91 AUDIT: 91 dev contexts, one SEARCH → AUDIT fit.
  - AutoML Elo: 64 TRAIN families × 3 folds × 3 seeds.
- Their numbers are on different scales; compare the order only (PROTOCOL §4).
- Our GP is never placed on the AutoML panel, because it was selected on TRAIN.
- The "frontier-only" column re-fits the board's Elo on the 5 LLMs alone and centres their mean on 1500, which is AutoML v2's convention.

| Model | dev91 AUDIT-Elo (25-field) | dev91 frontier-only | AutoML v1 | AutoML v2 [95% CI] | Rank dev91 / v1 / v2 | Same source on both panels |
| --- | --- | --- | --- | --- | --- | --- |
| Opus5p5_Extra | 1577.8 | 1700.7 | 1641.3 | 1681.3 [1632–1737] | 1 / 1 / 1 | yes |
| Gemini4_argon_high | 1524.4 | 1659.0 | 1560.8 | 1602.7 [1562–1648] | 2 / 2 / 2 | yes |
| GPT6p1_Sol_Max | 1497.3 | 1613.0 | 1455.8 | 1504.0 [1464–1546] | 3 / 3 / 3 | **no**: dev91 `f7597dd23e123a95…`, AutoML `fe5bb21ed62edb3a…` |
| GPT6_Astra_Max | 1308.4 | 1330.5 | 1342.1 | 1393.2 [1345–1437] | 4 / 4 / 4 | yes |
| grok4p7_xhigh | 1127.4 | 1196.7 | — | 1318.8 [1255–1375] | 5 / — / 5 | yes |

- **The order is the same on both panels:** dev91 vs AutoML v2 yes; dev91 vs AutoML v1 (4 entrants) yes.
- **Sol caveat:** the dev91 Sol row (H335 build tree) was compiled from source `f7597dd2…`. That is the 2026-11-02 refactor of `crates/jope-automl/src/GPT6p1_Sol_Max.rs`, split into 3 modules (commits `d4c48d20f9` … `793667e62a`).
  - The AutoML Sol score belongs to the earlier blob `fe5bb21e…`, as the Grok lane records in `assets/comparisons/protocol-elo/v1/tokens.json` (commit `3309223b45`, PR #2154, open).
  - Nobody has checked that the two sources behave the same, so Sol's agreement across panels is not a same-code check. The other four rows use the same source on both panels.
- AutoML inputs, read from git and sha-checked:
  - v1: `269a1bc24d:assets/automl-elo/v1/RESULTS.json`, sha256 `f1f7f3c304110bb1…`.
  - v2: `1177c4f9c1:assets/automl-elo/v2/RESULTS.json`, sha256 `4f14193ab3058ce9…`.

## Limits

- **dev91 is a reuse cohort.** H257 was tuned on it.
  - The 5 H381 / H410 contexts were adaptively exposed: `ac07466e`, `8eeca816`, `016648dc`, `6ecbb7d5` and `b1a5c1a1`.
  - XSTACK_2L was selected on TRAIN (H364) and scored once (H369). It was not tuned on those contexts, and nothing in this study was chosen on dev91.
- **Elo depends on the field.** Adding grok moved XSTACK_2L from 1726.8 to 1724.6; its family-mean u is unchanged.
  - Effect sizes come from the head-to-head table, not from Elo differences.
- **One run per entrant (seed 42).** The confidence intervals are family bootstraps; there is no seed variance.
- **grok4p7_xhigh provenance:** its authorship record says `fresh_session: false`. This is recorded and does not affect scoring.
- **Not done:** no new candidate of ours (owner's choice).
  - The H410 TRAIN lead (pooled runs + strong-learner library, +8.1% RMSE on 120 TRAIN families, p 0.0006) is the next candidate. It needs its own LOCK before any dev91 read.

Generated by `final_study_h412.py` from the stored board files; `final_study.json` has every number.
