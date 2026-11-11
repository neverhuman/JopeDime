# H440 Closeout — `ya`-Free Bomb/Shift Guard + Precision-Dilution Gate + 2L Residual Stacker + Duan-Matérn-ARD Columns

- **Wave:** `H440` (`research/flywheel_selfplay/h440_guard_and_duan_matern_cols`, `@agy_apex_w10`)
- **PREREG Commit:** `e03c3627e61d1b8784f541ec8e2bb6b7ab6598b7` (`origin/v1/next`)
- **Pre-Readout Amendment 1 Commit:** `bb962b4d7284be4d6c696b3a4d45771bc6d4785d` (`g3_replay_exact` aligned to canonical `gates_h424.sh` `w_diff == 0` & `u_diff <= 1e-9` before `READOUT_P201B.json` existed)
- **Published Stage:** `/home/ubuntu/apex-stage/h440/pub-r2` (`CODE.sha256` `672c99372a969dda`; compute outputs in `/home/ubuntu/jope-data/h335-cl/h440/pub-r1`, `wall_s = 3,669s <= 14,400s`, `16` single-thread CPU workers on `cpus 64-79`, `CUDA_VISIBLE_DEVICES=""`)
- **Pre-Readout Integrity Gates (`H440_r2.gates.json`):** **`8/8 PASS`** (`106/106` `FIT-SEL` columns `ok`, `201/201` `P201b` columns `ok`, `fitsel_replay_w_diff = 0.0`, `p201b_replay_w_diff = 0.0`, `APEX-CONFIRM` & `APEX-DEV` untouched)
- **Verdict:** **`PRIMARY PASS (2/2 Holm-Bonferroni FWER = 0.025 arms pass on BOTH P201b n=201 AND strictly unprobed P201b excl10 n=191)`**, plus **`+160.1 Elo` unprobed cross-lane synergy unlocking `H430` (`CW-1`) + `H433` (`CW-2`)**!

---

## 1. Headline Results Across All Three Panels

### A. Strictly Unprobed `P201b excl10` (`n = 191` families — primary unprobed generalization test)

| Comparison | Gain (`u`) | `% RMSE` | Clipped Mean | Sign-Flip `p` | `lofo_min` | `W / T / L` | `pass` | `ΔElo` (`95% CI`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`G_ONLY` vs `BASE`** *(Guard + 2L alone, 0 new models)* | `+0.0152` | `+1.51%` | `+0.0137` | `0.0122` *(Holm PASS)* | `+0.0112` | `108 / 10 / 73` | **True** | **`+41.4` (`[+4.6, +78.3]`)** |
| **`GM_FULL` vs `BASE`** *(Guard + 2L + 7 Duan-Matérn-ARD cols)* | `+0.0174` | `+1.72%` | `+0.0159` | `0.00492` *(Holm PASS)* | `+0.0134` | `126 / 4 / 61` | **True** | **`+95.2` (`[+51.7, +139.8]`)** |
| `M_BCAR` vs `C_PLAIN` *(Duan+YJ vs plain-y control in SXB)* | `-0.0004` | `-0.04%` | `+0.0006` | `0.443` | `-0.0005` | `108 / 24 / 59` | False | `+74.9` (`[+35.1, +114.8]`) |
| **`G_H430` vs `BASE`** *(Guard + `H430` `CW-1` keys)* | `+0.1306` | `+12.25%` | `+0.0242` | `0.000605` | `+0.0217` | `115 / 5 / 71` | **True** | **`+80.8` (`[+38.3, +125.1]`)** |
| **`GM_H430` vs `BASE`** *(Guard + Matérn-ARD + `H430`)* | `+0.1317` | `+12.34%` | `+0.0252` | `0.000437` | `+0.0227` | `128 / 7 / 56` | **True** | **`+132.9` (`[+85.5, +182.7]`)** |
| **`G_CW12` vs `BASE`** *(Guard + `H430` + `H433` `CW-2`)* | `+0.2124` | `+19.14%` | `+0.0342` | `3.91e-5` | `+0.0318` | `118 / 7 / 66` | **True** | **`+102.6` (`[+57.7, +151.0]`)** |
| **`GM_CW12` vs `BASE`** *(Guard + Matérn-ARD + `H430` + `H433`)* | **`+0.2135`** | **`+19.23%`** | **`+0.0354`** | **`2.57e-5`** | **`+0.0329`** | **`128 / 3 / 60`** | **True** | **`+160.1` (`[+108.7, +214.9]`)** |
| `GM_CW12` vs `G_CW12` *(Matérn-ARD increment atop `G_CW12`)* | `+0.0011` | `+0.11%` | `+0.0011` | `0.150` | `+0.0006` | `80 / 65 / 46` | False | `+57.4` (`[+25.3, +90.3]`) |
| `GM_CW12H` vs `BASE` *(swapping `H430` `key_direct` -> `H436` `key_hybrid`+`ridge_z`)* | `+0.2126` | `+19.15%` | `+0.0347` | `3.24e-5` | `+0.0323` | `127 / 2 / 62` | **True** | `+157.0` (`[+106.4, +211.7]`) |

### B. Full `P201b` (`n = 201` families; `33` stored GP columns rejected by `ya`-free guard)

- **`G_ONLY` vs `BASE`:** `+0.0313 u` (`+3.09% RMSE`), `clipped +0.0173`, `p = 0.00273` *(Holm PASS)*, `W/T/L 114/10/77`, **`ΔElo +43.4 [+8.0, +79.9]`**.
- **`GM_FULL` vs `BASE`:** `+0.0397 u` (`+3.89% RMSE`), `clipped +0.0231`, `p = 0.000257` *(Holm PASS)*, `W/T/L 134/4/63`, **`ΔElo +96.9 [+53.3, +140.4]`**.
- **`GM_CW12` vs `BASE`:** **`+0.3550 u` (`+29.88% RMSE`)**, `clipped +0.0515`, `p < 1e-6`, `W/T/L 136/3/62`, **`ΔElo +167.9 [+115.4, +221.3]`**.
- **`GM_CW12H` vs `BASE`:** **`+0.3938 u` (`+32.55% RMSE`)**, `clipped +0.0492`, `p < 1e-6`, `W/T/L 135/2/64`, **`ΔElo +163.4 [+111.0, +216.9]`**.

### C. `FIT-SEL` (`n = 106` families; `4` stored GP columns rejected by `ya`-free guard)

- **`G_ONLY` vs `BASE`:** `+0.0989 u` (`+9.42% RMSE`), `clipped +0.0168`, `p = 0.0281`, `W/T/L 53/12/41`, `ΔElo +14.7 [-39.1, +71.1]`.
- **`GM_FULL` vs `BASE`:** `+0.1029 u` (`+9.78% RMSE`), `clipped +0.0208`, `p = 0.00413` *(Holm PASS)*, `W/T/L 68/3/35`, **`ΔElo +113.5 [+49.8, +182.8]`**.
- **`GM_CW12` vs `BASE`:** **`+0.2271 u` (`+20.32% RMSE`)**, `clipped +0.0277`, `p = 2.1e-5`, `W/T/L 72/2/32`, **`ΔElo +146.4 [+74.2, +222.7]`**.
- **`GM_CW12` vs `G_CW12`:** `+0.0019 u`, `clipped +0.0019`, `p = 0.00122`, `W/T/L 42/43/21`, **`ΔElo +57.1 [+16.1, +98.7]`** (`pass = True`).
- **`GM_CW12H` vs `BASE`:** **`+0.2277 u` (`+20.36% RMSE`)**, `clipped +0.0283`, `p = 1.24e-5`, `W/T/L 71/2/33`, **`ΔElo +159.7 [+84.1, +239.4]`**.

---

## 2. Mechanism & Diagnostic Context Verification

Why did `@cl_fwriter`'s standalone `H430` (`K3_vs_BASE` on `P201b excl10`: `+0.0556 u`, `clipped +0.0060`, `p = 0.171`, `pass = False`, `ΔElo +76.6`), `H433` (`F4_vs_BASE` on `P201b excl10`: `+0.0750 u`, `clipped +0.0085`, `p = 0.0762`, `pass = False`, `ΔElo +23.7`), and `H436` (`KH_vs_KD` on `P201b excl10`: `clipped -0.0059`, `p = 0.858`, `ΔElo -15.3`) miss the `excl10` `primary_pass` threshold under unguarded `SXB`, yet surge to **`+0.2135 u`, `clipped +0.0354`, `p = 2.57e-5`, `ΔElo +160.1 [+108.7, +214.9]` (`pass = True`)** inside `H440`'s `guard_and_stack_2l`?

1. **Precision-Dilution Gate (`oof_rmse_joint <= oof_rmse_base / 1.04`):** Unguarded `SXB` gave small nonzero weight to noisy/imperfect key or formula columns (`key_gbm` received weight on `106/191` families in `H430`, losing `62` families `W/T/L 102/27/62`, and `H436` diluted near-exact `7d3e30f33f` `17.05 -> 10.16` while `H430` `key_direct` crashed `d8134cd42a` `2.23 -> -8.42`). Under `H440`'s `1.04x` admission gate, peer columns (`H430`/`H433`/`H436`) are admitted **only** when they improve joint cross-fitted RMSE by `>= 4%`: on `7d3e30f33f54` (`BASE = 17.0542`), **all 7 guarded arms score `17.0526`** (zero precision dilution), and on `d8134cd42ae6` (`BASE = 2.2303`), **all guarded arms score `2.2624`** (zero `key_direct` crash).
2. **`ya`-Free `bag`/`ref` Tail & Shift Bomb Guard (`W1`):** Rejects `33` stored GP columns on `P201b` (`4` on `FIT-SEL`) purely from `Xa` (`bag`/`ref`) geometry without touching `ya`. On `51e71f0b3089`, `BASE = -0.5338` jumps to `G_ONLY = +2.5055` (`+3.04 u`). Across all `191` unprobed families, the **single worst loss** of `GM_CW12` vs `BASE` is `-0.6537 u` (`99ac70c928bb`: `5.94 -> 5.29`) and the second worst is `-0.1770 u` — zero catastrophic bombs.
3. **2-Level Residual Stacker + Duan-Matérn-ARD Diversity (`W2–W4`):** Even with zero peer columns, `GM_FULL` beats `BASE` on `126 / 191` unprobed families (`+95.2 Elo`, `p = 0.00492`), and adding the 7 Duan-Matérn-ARD columns atop `G_CW12` (`GM_CW12` vs `G_CW12`) wins `80 / 46` unprobed families (`+57.4 Elo`) and `42 / 21` `FIT-SEL` families (`+57.1 Elo`, `p = 0.00122`). On `3b0303bb3885`, `BASE = 2.4712` rises to `6.6734` (`GM_CW12`) and `15.1150` (`GM_CW12H` with `H436` `key_hybrid`+`ridge_z`).

---

## 3. Implication for Joint `APEX-DEV` (`110`-Cluster) Candidate

`GM_CW12` (or `GM_CW12H`) is our first 100% ours-only stack recipe to clear **every** statistical bar (`p = 2.57e-5`, `clipped +0.0354`, `lofo_min +0.0329`, `W/T/L 128/3/60`, **`+160.1 Elo [+108.7, +214.9]`**) on the `191` strictly unprobed `P201b` families AND `+146.4 Elo` (`+159.7 Elo` for `GM_CW12H`) on `FIT-SEL`. Any `CW-5` (`H442`) or `CW-4` (`@cl_fwriter`) neural ICL column that clears `FIT-SEL` should enter through this exact `guard_and_stack_2l` admission gate before a single joint pre-registered `APEX-DEV` evaluation.
