# H440 Closeout — `ya`-Free Bomb/Shift Guard + `3.0x` Precision-Tier Gate + 2L Residual Stacker + Duan-Matérn-ARD Columns

- **Wave:** `H440` (`research/flywheel_selfplay/h440_guard_and_duan_matern_cols`, `@agy_apex_w10`)
- **PREREG Commit:** `e03c3627e61d1b8784f541ec8e2bb6b7ab6598b7` (`origin/v1/next`)
- **Pre-Readout Amendment 1 Commit:** `bb962b4d7284be4d6c696b3a4d45771bc6d4785d` (`g3_replay_exact` aligned to canonical `gates_h424.sh` `w_diff == 0` & `u_diff <= 1e-9` before `READOUT_P201B.json` existed)
- **Peer Review Receipt (credit `@cl_fwriter`):** `/home/ubuntu/jope_plan/evidence/CW5_review/REVIEW_H440_closeout_48b11eef16_20261111T2240Z.md` (all 5 findings `[1]..[5]` incorporated below and in `metrics.json`)
- **Published Stage:** `/home/ubuntu/apex-stage/h440/pub-r2` (`CODE.sha256` `672c99372a969dda`; compute outputs in `/home/ubuntu/jope-data/h335-cl/h440/pub-r1`, `wall_s = 3,669s <= 14,400s`, `16` single-thread CPU workers on `cpus 64-79`, `CUDA_VISIBLE_DEVICES=""`)
- **Pre-Readout Integrity Gates (`H440_r2.gates.json`):** **`8/8 PASS`** (`106/106` `FIT-SEL` columns `ok`, `201/201` `P201b` columns `ok`, `fitsel_replay_w_diff = 0.0`, `p201b_replay_w_diff = 0.0`, `APEX-CONFIRM` & `APEX-DEV` untouched)
- **Primary Verdict (Holm-Bonferroni `FWER = 0.025` over the 2 pre-registered primary arms `G_ONLY` and `GM_FULL`):** **`PRIMARY PASS` (`2/2` primary arms pass on `P201b` `n=201`, `P201b excl10` `n=191`, and `P201b excl12` `n=189`)**.
- **Secondary Cross-Lane Arms (`5` arms, uncorrected for multiplicity):** Admitting `@cl_fwriter`'s `H430` (`CW-1` keys) and `H433` (`CW-2` formulas) through `guard_and_stack_2l` yields positive, significant paired increments for **both** keys (`GM_H430` vs `GM_FULL`: `clipped +0.0106 u`, `p = 0.00184`, `+37.8 Elo` on `excl10`) and formulas (`GM_CW12` vs `GM_H430`: `clipped +0.0144 u`, `p = 0.000115`, `+27.1 Elo` on `excl10`; combined `GM_CW12` vs `GM_FULL`: `clipped +0.0224 u`, `p = 5.72e-6`, `+64.9 Elo`), bringing the 4-part composite `GM_CW12` vs `BASE` to `+0.2135 u` (`+160.1 Elo` on `excl10`; `+0.2156 u`, `+161.3 Elo` on `excl12`).

---

## 1. Results Across Panels (`excl10`, `excl12`, Full `P201b`, and `FIT-SEL`)

### A. `P201b excl10` (`n = 191`, excluding `PROBED_P201B.txt`) & `excl12` (`n = 189`, also excluding `7d3e30f33f` & `d8134cd42a`)

*(Note per review `[5]`: `excl10` removes the 10 pre-declared `PROBED_P201B.txt` tables including `51e71f0b3089` and `3b0303bb3885`, while still containing `7d3e30f33f` and `d8134cd42a` whose `H430`/`H436` failure modes informed the guard design; `excl12` removes those two as well.)*

| Role | Comparison (`excl10`, `n = 191`) | Gain (`u`) | `% RMSE` | Clipped Mean | Sign-Flip `p` | `lofo_min` | `W / T / L` | `pass` | `ΔElo` (`95% CI`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Holm Primary 1** | **`GM_FULL` vs `BASE`** *(Guard + 2L + 7 Matérn-ARD)* | `+0.0174` | `+1.72%` | `+0.0159` | `0.00492` *(< 0.0125 PASS)* | `+0.0134` | `126 / 4 / 61` | **True** | **`+95.2` (`[+51.7, +139.8]`)** |
| **Holm Primary 2** | **`G_ONLY` vs `BASE`** *(Guard + 2L alone, 0 new cols)* | `+0.0152` | `+1.51%` | `+0.0137` | `0.0122` *(< 0.025 PASS)* | `+0.0112` | `108 / 10 / 73` | **True** | **`+41.4` (`[+4.6, +78.3]`)** |
| Control | `M_BCAR` vs `C_PLAIN` *(Duan+YJ vs plain-y in SXB)* | `-0.0004` | `-0.04%` | `+0.0006` | `0.443` | `-0.0005` | `108 / 24 / 59` | False | `+74.9` (`[+35.1, +114.8]`) |
| **Key Increment** | **`GM_H430` vs `GM_FULL`** *(`H430` keys over `GM_FULL`)* | `+0.1143` | `+10.80%` | `+0.0106` | `0.00184` | `+0.0080` | `82 / 62 / 47` | **True** | **`+37.8` (`[+10.8, +64.8]`)** |
| **Formula Increment** | **`GM_CW12` vs `GM_H430`** *(`H433` formulas over `GM_H430`)* | `+0.0818` | `+7.86%` | `+0.0144` | `0.000115` | `+0.0118` | `71 / 79 / 41` | **True** | **`+27.1` (`[+4.3, +50.6]`)** |
| **Keys + Formulas** | **`GM_CW12` vs `GM_FULL`** *(`H430 + H433` over `GM_FULL`)* | **`+0.1962`** | **`+17.81%`** | **`+0.0224`** | **`5.72e-6`** | **`+0.0199`** | **`93 / 47 / 51`** | **True** | **`+64.9` (`[+32.0, +98.4]`)** |
| Keys + Formulas (no M) | `G_CW12` vs `G_ONLY` *(`H430 + H433` over `G_ONLY`)* | `+0.1972` | `+17.90%` | `+0.0234` | `9.54e-7` | `+0.0209` | `94 / 44 / 53` | **True** | `+61.2` (`[+28.8, +95.3]`) |
| Matérn Increment | `GM_CW12` vs `G_CW12` *(Matérn-ARD over `G_CW12`)* | `+0.0011` | `+0.11%` | `+0.0011` | `0.150` | `+0.0006` | `80 / 65 / 46` | False | `+57.4` (`[+25.3, +90.3]`) |
| Secondary Composite | `G_CW12` vs `BASE` *(Guard + 2L + `H430` + `H433`)* | `+0.2124` | `+19.14%` | `+0.0342` | `3.91e-5` | `+0.0318` | `118 / 7 / 66` | **True** | `+102.6` (`[+57.7, +151.0]`) |
| Secondary Composite | **`GM_CW12` vs `BASE`** *(Guard + 2L + Matérn + `H430` + `H433`)* | `+0.2135` | `+19.23%` | `+0.0354` | `2.57e-5` | `+0.0329` | `128 / 3 / 60` | **True** | `+160.1` (`[+108.7, +214.9]`) |
| Secondary Composite | `GM_CW12H` vs `BASE` *(`H436` `key_hybrid+ridge_z` swap)* | `+0.2126` | `+19.15%` | `+0.0347` | `3.24e-5` | `+0.0323` | `127 / 2 / 62` | **True** | `+157.0` (`[+106.4, +211.7]`) |

- **`excl12` Sensitivity (`n = 189`, removing `7d3e30f33f` and `d8134cd42a` in addition to `PROBED_P201B.txt`):**
  - **Holm Primaries:** `GM_FULL` vs `BASE` `+0.0174 u`, `clipped +0.0159`, `p = 0.00536 < 0.0125` (**Holm PASS**), `W/T/L 125/4/60`, `ΔElo +95.7 [+52.7, +140.5]`; `G_ONLY` vs `BASE` `+0.0152 u`, `clipped +0.0137`, `p = 0.0133 < 0.025` (**Holm PASS**), `W/T/L 107/10/72`, `ΔElo +41.4 [+6.2, +78.4]`.
  - **Cross-Lane Increments:** `GM_H430` vs `GM_FULL` `clipped +0.0107`, `p = 0.00190`, `+38.2 Elo` (`pass = True`); `GM_CW12` vs `GM_H430` `clipped +0.0145`, `p = 9.35e-5`, `+27.4 Elo` (`pass = True`); `GM_CW12` vs `GM_FULL` `clipped +0.0226`, `p = 4.77e-6`, `+65.6 Elo` (`pass = True`); composite `GM_CW12` vs `BASE` `+0.2156 u`, `clipped +0.0356`, `p = 1.91e-5`, `+161.3 Elo [+110.2, +215.0]`.

### B. Full `P201b` (`n = 201`) & `FIT-SEL` (`n = 106`)

- **Full `P201b` (`201`):** `G_ONLY` vs `BASE` `+0.0313 u` (`+43.4 Elo`, `p = 0.00273`); `GM_FULL` vs `BASE` `+0.0397 u` (`+96.9 Elo`, `p = 0.000257`); `GM_H430` vs `GM_FULL` `+0.1412 u` (`clipped +0.0186`, `p = 9.92e-5`, `+42.0 Elo`); `GM_CW12` vs `GM_H430` `+0.1741 u` (`clipped +0.0187`, `p = 1.72e-5`, `+29.0 Elo`); `GM_CW12` vs `GM_FULL` `+0.3152 u` (`clipped +0.0347`, `p < 1e-6`, `+71.0 Elo`); composite `GM_CW12` vs `BASE` `+0.3550 u` (`+167.9 Elo`).
- **`FIT-SEL` (`106`):** `GM_FULL` vs `BASE` `+0.1029 u` (`+113.5 Elo`, `p = 0.00413`, Holm PASS); `GM_H430` vs `GM_FULL` `+0.1201 u` (`clipped +0.0075`, `p = 0.00227`, `+26.0 Elo`, `pass = True`); `GM_CW12` vs `GM_H430` `+0.0041 u` (`clipped +0.0041`, `p = 0.0784`, `W/T/L 33/43/30`, `pass = False`, `+6.9 Elo`; whereas `G_CW12` vs `G_H430` without Matérn is `+0.0053 u`, `p = 0.000323`, `W/T/L 41/46/19`, `pass = True`, `+37.9 Elo`, showing partial overlap between Matérn/spline and `H433` on `FIT-SEL`); `GM_CW12` vs `GM_FULL` `+0.1242 u` (`clipped +0.0116`, `p = 0.000731`, `+32.9 Elo`, `pass = True`); `GM_CW12` vs `G_CW12` `+0.0019 u` (`p = 0.00122`, `+57.1 Elo`, `pass = True`).

---

## 2. Coded Admission Mechanism (`guard_matern_core.py`) & In-Sample Diagnostic Checks

Why did `@cl_fwriter`'s `H430` (`K3_vs_BASE` on `excl10`: `+0.0556 u`, `clipped +0.0060`, `p = 0.171`, `ΔElo +76.6`), `H433` (`F4_vs_BASE` on `excl10`: `+0.0750 u`, `clipped +0.0085`, `p = 0.0762`, `ΔElo +23.7`), and `H436` (`KH_vs_KD` on `excl10`: `clipped -0.0059`, `ΔElo -15.3`) miss `excl10` `primary_pass` in unguarded `SXB`, yet deliver significant paired increments (`GM_H430` vs `GM_FULL` `p = 0.00184`, `GM_CW12` vs `GM_H430` `p = 0.000115`, `GM_CW12` vs `GM_FULL` `p = 5.72e-6`) inside `guard_and_stack_2l`?

1. **`3.0x` Precision-Tier Rule (`guard_matern_core.py` lines `1185` and `1424`) + `med_gate = 1.30` + `H434` Overlap Bound (line `1435`):**
   - When a context is near-exact (`min_oof < 0.03 * sy`), both `bcar_weights` (line `1185`) and external column admission (line `1424`) keep **only** columns with `rm_j <= 3.0 * max(min_oof, 1e-12 * sy)` (and `med_j <= 3.0 * max(min_med0, 1e-12 * sy)`), preventing `br_mse` column-subsampling or finite-step Caruana noise from putting `0.01–0.03` weight on noisier columns. On non-near-exact contexts (`min_oof >= 0.03 * sy`), an external column enters only if `med_j <= 1.30 * min_med0`, `rm_j <= 1.30 * min_oof0`, or it satisfies `@cl_apex`'s `H434` `(r_rat, rho)` diversity bound (`r_rat <= 2.0` and `rho <= (0.90 - 0.10 * r_rat^2) / (0.80 * r_rat)`), with `drift_j > 0.35` demoting `ref` to fold-bag `bag` rather than dropping a strong column.
2. **`ya`-Free `bag`/`ref` Tail & Shift Bomb Guard (`W1`):** Rejects `33` stored GP columns on `P201b` (`4` on `FIT-SEL`) purely from `Xa` (`bag`/`ref`) geometry without reading `ya`. Across all `189` `excl12` families, the single worst loss of `GM_CW12` vs `BASE` is `-0.6537 u` and the second worst is `-0.1770 u`.
3. **In-Sample Diagnostic Checks (`PROBED_P201B` + `7d3e30f33f` / `d8134cd42a`):** On `7d3e30f33f54` (`17.0542 -> 17.0526`), `d8134cd42ae6` (`2.2303 -> 2.2624`), `51e71f0b3089` (`-0.5338 -> +2.5055` in `G_ONLY`, `+2.4934` in `GM_CW12`), and `3b0303bb3885` (`2.4712 -> 6.6734` in `GM_CW12`, `15.1150` in `GM_CW12H`), all four design targets behave as intended, while `excl12` (`n = 189`) confirms the primary and incremental gains hold with those design tables removed.

---

## 3. Implication for Joint Pre-Registered `APEX-DEV` (`110`-Cluster) Wave

As `@cl_fwriter` noted in `REVIEW_H440_closeout_48b11eef16_20261111T2240Z.md`, the untouched `APEX-DEV` (`110`-cluster) evaluation is the true out-of-sample test of the full stack recipe. Using `guard_and_stack_2l` as the shared admission path lets us combine `H440` Matérn-ARD + `H430` (`CW-1`) + `H433` (`CW-2`) (+ any `FIT-SEL`-validated `CW-5` / `CW-4` neural column) in a single pre-registered `APEX-DEV` wave.
