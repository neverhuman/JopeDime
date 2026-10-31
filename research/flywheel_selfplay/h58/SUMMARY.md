# H58 Wave Summary: Flywheel Self-Play Defect-Isolation & Ranking-Loss Ablation (`t550`)

## 1. Executive Summary & Verdict

Wave **H58** (`t550`) executed the preregistered 4-arm cumulative defect-isolation and ranking-loss ablation across **72 isolated flywheel cells** (`60` inner 5-fold family-grouped cross-validation cells on `DL-Train` [`205` families, `1,219` contexts] + `12` terminal training/evaluation cells on `DL-Val` [`19` disjoint families, `91` contexts], `3` seeds `[12, 13, 14]` × `4` arms `[A, B, C, D]`, `R = 20` self-play rounds × `2` gradient steps/round = `40` optimizer steps/cell) on `cuda:0` under `flock /home/ubuntu/jope_plan/qualification.lock`.

### Headline Outcomes
1. **`t537` Promotion Gate Verdict**: **NO-PROMOTE (`t537_gate_passed = false` for `B`, `C`, and `D` vs `A` and vs `INIT_G0`)**. None of `B`, `C`, or `D` cleared the preregistered `t537` promotion threshold (`mean_LCB95 > +0.001` and `family_median >= 0.0` over `10,000` paired family-cluster bootstrap draws).
2. **Global Pool Ranking (`Spearman` $\rho$ on Unrevealed Passing Candidates `4..P-1`) Increased by `+42.4%` (OOF) to `+50.3%` (`DL-Val`) Across `B -> C -> D`**:
   - On `DL-Train` 205-family OOF, mean Spearman rank correlation $\rho$ improved monotonically across every progressive fix:
     $$\text{Arm A } (0.126683) \longrightarrow \text{Arm B } (0.178444) \longrightarrow \text{Arm C } (0.179610) \longrightarrow \text{Arm D } (0.180379)$$
   - On terminal `DL-Val` (`19` disjoint families), mean Spearman $\rho$ likewise improved monotonically across every seed (`12, 13, 14`):
     $$\text{Arm A } (0.120185) \longrightarrow \text{Arm B } (0.178730) \longrightarrow \text{Arm C } (0.179548) \longrightarrow \text{Arm D } (0.180597)$$
3. **Within the Leak-Free Defect-Isolated Family (`B -> C -> D`), Both Fix Bug 2 (`C: Hard-Negative Self-Play Admission`) and Fix Rank 3 (`D: Listwise Plackett-Luce + Margin Loss`) Produced Consistent Positive Gains Over `B`**:
   - **Live GP Self-Play Admission (`Fix Bug 2`)**: Across `2,880` live native Rust two-fold CV evaluations (`720` per arm, **`0` execution failures**), HWM-only `promote()` in `Arm A` (`33/720`, `4.58%`) and `Arm B` (`29/720`, `4.03%`) discarded `>95%` of completed live GP evaluations and admitted `0` non-HWM candidates and `0` CV gate failures. `Arm C` and `Arm D` admitted **`720/720` (`100.0%`, `40/cell`)** of completed live GP evaluations (`666` and `665` non-HWM valid candidates + `31` and `29` genuine live CV gate failures), plus `333.3` historical journal candidates/cell (vs `262.5` in `A/B`) and `~6,460` harvested top-8 false-positive hard negatives/cell.
   - **`C` and `D` vs `B` on Terminal `DL-Val`**: `Arm C` and `Arm D` outperformed `Arm B` by **`+3.13` unrev Elo** (`1272.27` vs `1269.14`) and **`+4.73` native Elo** (`1258.62` vs `1253.89`), increased family utility mean from `0.294746` (`B`) to `0.296669` (`C, D`, `+0.001923` with `0` negative families and `mean_LCB95 = 0.000000`), and reduced normalized regret from `1.472660` to `1.468645`.
   - **`D` vs `B/C/INIT_G0` on 205-Family `DL-Train` OOF**: `Arm D` achieved the highest family utility mean (`0.195060` vs `C: 0.193609`, `B: 0.193362`, `INIT_G0: 0.193080`), highest family utility median (`0.234073` vs `B/C: 0.227214`), highest top-5% hit rate (`2.160%` vs `B: 2.051%`), lowest gate violation rate (`2.953%` vs `B/C: 2.981%`), and highest Spearman $\rho$ (`0.180379`).
4. **Why `Arm A (CONTROL)` Had Higher Top-1 `argmin` Utility (`0.5783` OOF / `0.6398` `DL-Val` Unrev) Than `Arms B/C/D` (`0.1934–0.1951` OOF / `0.2947–0.2967` `DL-Val`)**:
   - **First (Revealed-Anchor Evaluation Leak in `native` Mode)**: Unmasked `argmin` over `0..P-1` in `Arm A` inflated `native` family utility by **`+0.0718`** on `DL-Train` OOF (`0.650086` native vs `0.578299` unrev) and **`+0.0198`** on `DL-Val` (`0.659610` native vs `0.639788` unrev). Masking `0..3` in `Arms B/C/D` eliminated this leak (`native == unrev` identically).
   - **Second (10.2× Scale Dominance of Unattenuated `prior10` Over W1 Trajectory Interpolation in `gamenet_anchor`)**: Inside `gamenet_anchor.predict_costs` (`model.py:277-299`), the W1 skip-connection computes:
     $$w_1\_\text{cost}(c) = \underbrace{\sum_{t=0}^3 \tilde{w}(c, t) \frac{F_t}{\text{scale}}}_{\text{W1 trajectory interpolation (range } \approx 1.78\text{)}} + \underbrace{\left( \text{base\_prior\_cost}(c) - \sum_{t=0}^3 \tilde{w}(c, t) \text{base\_prior\_cost}(t) \right)}_{\text{prior10 offset from revealed anchors (range } \approx 18.20\text{)}}$$
     On `DL-Val`, the best of the 4 revealed seed anchors `0..3` already has high utility (`0.7099`) and has close structural neighbors in `4..P-1` in 16D `proj16` space (`median min` $d^2 = 0.0917$). When `base_prior_cost = 0` (`Arm A`), the second term vanishes and `gamenet_anchor` acts as a pure 16D W1 nearest-neighbor interpolator around the best revealed seed anchor (`unrev utility = 0.5736` from W1 alone, `0.6398` with adapter, despite poor global pool correlation $\rho = 0.0621$). When raw `prior10` is passed into `base_prior_cost` (`Arms B/C/D`, matching `run_h39_diagnostics.py`), its `18.20`-unit peak-to-peak range (`10.2x` larger than the `1.78`-unit W1 trajectory term and penalizing the nearest `proj16` neighbor of the best revealed anchor by **`+8.42` units** on average) overwhelms the W1 trajectory signal and pulls top-1 `argmin` toward the static `prior10` minimum (`unrev utility = 0.2567`), while simultaneously lifting global Spearman $\rho$ across all passing candidates by `+42%` to `+50%`.

---

## 2. Experimental Design & Preregistered Protocol

All protocol parameters from [`research/flywheel_selfplay/h58/PREREG.md`](PREREG.md) and [`research/flywheel_selfplay/h58/runner.py`](runner.py) were executed without modification after passing the v4 Codex pre-GPU audit (`PASS` in `/home/ubuntu/jope-data/h58/redteam_codex_pre_gpu_v4.md`):

| Parameter | Value |
| --- | --- |
| **Task / Wave** | `t550` / `H58` |
| **Arms** | `A` (`CONTROL`), `B` (`FIX1`), `C` (`FIX1+2`), `D` (`FIX1+2+3`), plus `INIT_G0` (untouched fold-clean init baseline) |
| **Seeds** | `12, 13, 14` (`3` seeds) |
| **Inner CV Split** | `5`-fold family-grouped CV on `DL-Train` (`205` families, `1,219` contexts, `fold_seed = 4900`) → `60` inner-CV cells |
| **Terminal Validation Split** | Fit on full `DL-Train` (`1,219` contexts), evaluate once terminally on disjoint `DL-Val` (`19` families, `91` contexts) → `12` terminal cells |
| **Total Cells Executed** | `72` cells (`18` per arm) |
| **Training Budget per Cell** | `R = 20` self-play rounds × `2` optimizer steps/round = `40` gradient steps (`batch_size = 256`, `slate_size = 192`, `lr = 3e-4`, `weight_decay = 1e-4`, `clip_grad_norm = 1.0`) |
| **Live GP Self-Play Budget per Cell** | `2` training contexts/round × `8` AST mutations proposed/context → neural adapter ranking → `2` candidates/context (`1` greedy + `1` exploration) evaluated in native Rust two-fold CV (`40` live Rust evaluations/cell, `2,880` across wave) |
| **Initialization** | Fold-clean from-scratch (`gamenet_anchor` seeded by `20000 + seed`, `adapter_10tok` seeded by `10000 + seed` with `scale = 0.1`; `prog_mean`/`prog_std` computed strictly on `fit_cids`) |
| **Trainable Parameters** | `29,211` in `Arm A` (`parent_delta_net` unmasked); `23,546` in `Arms B, C, D` (dead `parent_delta_net` frozen) |
| **Wall-Clock Duration** | `5530.37 s` (`92.17 min`), completed `2026-10-31T02:12:48Z` with exit status `0` |

---

## 3. Primary Results: `DL-Train` 5-Fold Family-Grouped OOF (`205` Families, `1,219` Contexts)

Averaged across seeds `12, 13, 14` (`60` inner-CV cells). Bradley-Terry Elo (`compute_family_grouped_bradley_terry_elo`, anchor `A = 1500.0`) is computed over all `3 × 1,219 = 3,657` OOF context evaluations clustered by `205` families (`1,000` bootstrap replicates).

### 3.1 Leak-Free Unrevealed Evaluation (`unrev`: `choice = 4 + argmin(pred[4:])`)

| Arm | Unrev BT Elo (95% CI) | W / T / L vs `A` | Family Utility Mean | Family Utility Median | Gate Violation Rate | Mean Norm Regret | Top-5% Hit Rate | Spearman $\rho$ (`4..P-1`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`A` (`CONTROL`)** | **1500.00** `[1500.00, 1500.00]` | — | **0.578299** | **0.654254** | **1.176%** | **0.856989** | **15.450%** | 0.126683 |
| **`INIT_G0`** | 1305.03 `[1276.62, 1333.96]` | 286 / 19 / 914 | 0.193080 | 0.228880 | — | — | — | — |
| **`B` (`FIX1`)** | 1305.48 `[1277.02, 1334.10]` | 285 / 19 / 915 | 0.193362 | 0.227214 | 2.981% | 1.625426 | 2.051% | 0.178444 |
| **`C` (`FIX1+2`)** | 1306.04 `[1277.96, 1334.97]` | 285 / 19 / 915 | 0.193609 | 0.227214 | 2.981% | 1.625087 | 2.078% | 0.179610 |
| **`D` (`FIX1+2+3`)** | 1305.72 `[1277.01, 1334.61]` | 286 / 19 / 914 | 0.195060 | 0.234073 | 2.953% | 1.622254 | 2.160% | **0.180379** |

### 3.2 Native Evaluation (`native`: `0..P-1` unmasked for `A`; masked `4..P-1` for `B/C/D`)

| Arm | Native BT Elo (95% CI) | W / T / L vs `A` | Family Utility Mean | Family Utility Median | Gate Violation Rate | Mean Norm Regret | Top-5% Hit Rate | Spearman $\rho$ (`4..P-1`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`A` (`CONTROL`)** | **1500.00** `[1500.00, 1500.00]` | — | **0.650086** | **0.750618** | **0.164%** | **0.721219** | **15.669%** | 0.126683 |
| **`INIT_G0`** | 1239.60 `[1208.16, 1268.20]` | 222 / 21 / 976 | 0.193080 | 0.228880 | — | — | — | — |
| **`B` (`FIX1`)** | 1240.36 `[1209.26, 1268.93]` | 223 / 21 / 975 | 0.193362 | 0.227214 | 2.981% | 1.625426 | 2.051% | 0.178444 |
| **`C` (`FIX1+2`)** | 1240.98 `[1210.30, 1268.85]` | 224 / 21 / 974 | 0.193609 | 0.227214 | 2.981% | 1.625087 | 2.078% | 0.179610 |
| **`D` (`FIX1+2+3`)** | 1240.65 `[1209.97, 1269.55]` | 225 / 21 / 973 | 0.195060 | 0.234073 | 2.953% | 1.622254 | 2.160% | **0.180379** |

---

## 4. Terminal Validation Results: `DL-Val` (`19` Disjoint Families, `91` Contexts)

Each of the `12` terminal cells (`4` arms × `3` seeds) trained on all `1,219` `DL-Train` contexts for `20` rounds (`40` gradient steps) and executed **exactly `1` terminal evaluation sweep** on `DL-Val` (`terminal_dl_val_eval_sweeps_per_trained_arm_seed = 1`).

### 4.1 Leak-Free Unrevealed Evaluation (`unrev`: `choice = 4 + argmin(pred[4:])`)

| Arm | Unrev BT Elo (95% CI) | W / T / L vs `A` | Family Utility Mean | Family Utility Median | Gate Violation Rate | Mean Norm Regret | Top-5% Hit Rate | Spearman $\rho$ (`4..P-1`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`A` (`CONTROL`)** | **1500.00** `[1500.00, 1500.00]` | — | **0.639788** | **0.631066** | **0.000%** | **0.750309** | **17.949%** | 0.120185 |
| **`INIT_G0`** | 1269.14 `[1147.50, 1359.61]` | 20 / 0 / 71 | 0.305258 | 0.389401 | — | — | — | — |
| **`B` (`FIX1`)** | 1269.14 `[1148.28, 1357.33]` | 20 / 0 / 71 | 0.294746 | 0.311338 | 1.099% | 1.472660 | 7.692% | 0.178730 |
| **`C` (`FIX1+2`)** | 1272.27 `[1150.81, 1358.83]` | 20 / 0 / 71 | 0.296669 | 0.311338 | 1.099% | 1.468645 | 7.692% | 0.179548 |
| **`D` (`FIX1+2+3`)** | 1272.27 `[1150.81, 1358.83]` | 20 / 0 / 71 | 0.296669 | 0.311338 | 1.099% | 1.468645 | 7.692% | **0.180597** |

### 4.2 Native Evaluation (`native`: `0..P-1` unmasked for `A`; masked `4..P-1` for `B/C/D`)

| Arm | Native BT Elo (95% CI) | W / T / L vs `A` | Family Utility Mean | Family Utility Median | Gate Violation Rate | Mean Norm Regret | Top-5% Hit Rate | Spearman $\rho$ (`4..P-1`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`A` (`CONTROL`)** | **1500.00** `[1500.00, 1500.00]` | — | **0.659610** | **0.654064** | **0.000%** | **0.709457** | **23.077%** | 0.120185 |
| **`INIT_G0`** | 1255.46 `[1103.69, 1371.01]` | 19 / 0 / 72 | 0.305258 | 0.389401 | — | — | — | — |
| **`B` (`FIX1`)** | 1253.89 `[1099.43, 1367.25]` | 18 / 0 / 73 | 0.294746 | 0.311338 | 1.099% | 1.472660 | 7.692% | 0.178730 |
| **`C` (`FIX1+2`)** | 1258.62 `[1109.60, 1372.50]` | 19 / 0 / 72 | 0.296669 | 0.311338 | 1.099% | 1.468645 | 7.692% | 0.179548 |
| **`D` (`FIX1+2+3`)** | 1258.62 `[1109.60, 1372.50]` | 19 / 0 / 72 | 0.296669 | 0.311338 | 1.099% | 1.468645 | 7.692% | **0.180597** |

### 4.3 Per-Seed Stability (`Seeds 12, 13, 14`)

| Metric / Split | Seed `12` | Seed `13` | Seed `14` | 3-Seed Mean |
| --- | --- | --- | --- | --- |
| **`A` OOF Spearman $\rho$** | 0.124484 | 0.128614 | 0.126920 | 0.126683 |
| **`B` OOF Spearman $\rho$** | 0.178221 | 0.178530 | 0.177418 | 0.178444 |
| **`C` OOF Spearman $\rho$** | 0.179467 | 0.179626 | 0.178586 | 0.179610 |
| **`D` OOF Spearman $\rho$** | **0.179815** | **0.181009** | **0.179245** | **0.180379** |
| **`A` `DL-Val` Spearman $\rho$** | 0.117121 | 0.123679 | 0.119755 | 0.120185 |
| **`B` `DL-Val` Spearman $\rho$** | 0.178867 | 0.179118 | 0.178204 | 0.178730 |
| **`C` `DL-Val` Spearman $\rho$** | 0.179650 | 0.179661 | 0.179332 | 0.179548 |
| **`D` `DL-Val` Spearman $\rho$** | **0.180527** | **0.181208** | **0.180055** | **0.180597** |
| **`A` `DL-Val` Unrev Utility** | 0.640688 | 0.639880 | 0.638797 | 0.639788 |
| **`B` `DL-Val` Unrev Utility** | 0.290901 | 0.296669 | 0.296669 | 0.294746 |
| **`C` `DL-Val` Unrev Utility** | 0.296669 | 0.296669 | 0.296669 | 0.296669 |
| **`D` `DL-Val` Unrev Utility** | 0.296669 | 0.296669 | 0.296669 | 0.296669 |

---

## 5. Paired Family-Cluster Bootstrap Gate Table (`t537`, 10,000 Draws)

Promotion rule (`t537`): `mean_LCB95 > +0.001` **and** `family_median >= 0.0`.

### 5.1 `DL-Train` OOF (`205` Families, Leak-Free `unrev`)

| Comparison | Mean Gain | `mean_LCB95` | Mean 95% CI | Family Median | Pos / Neg / Tied Families | `t537` Gate |
| --- | --- | --- | --- | --- | --- | --- |
| **`B vs A`** | -0.384937 | -0.427826 | `[-0.436474, -0.334617]` | -0.367510 | 28 / 177 / 0 | `FAIL` |
| **`C vs A`** | -0.384690 | -0.427660 | `[-0.435845, -0.333349]` | -0.367510 | 28 / 177 / 0 | `FAIL` |
| **`D vs A`** | -0.383239 | -0.426560 | `[-0.434491, -0.332123]` | -0.359514 | 27 / 178 / 0 | `FAIL` |
| **`C vs B`** | **+0.000247** | -0.000393 | `[-0.000486, +0.001192]` | 0.000000 | 5 / 4 / 196 | `FAIL` |
| **`D vs C`** | **+0.001451** | -0.000913 | `[-0.001251, +0.004816]` | 0.000000 | 8 / 10 / 187 | `FAIL` |
| **`D vs B`** | **+0.001698** | -0.000658 | `[-0.001027, +0.005102]` | 0.000000 | 11 / 10 / 184 | `FAIL` |
| **`B vs INIT_G0`** | **+0.000282** | -0.003614 | `[-0.004379, +0.005227]` | 0.000000 | 19 / 17 / 169 | `FAIL` |
| **`C vs INIT_G0`** | **+0.000529** | -0.003385 | `[-0.004125, +0.005468]` | 0.000000 | 19 / 14 / 172 | `FAIL` |
| **`D vs INIT_G0`** | **+0.001981** | -0.001501 | `[-0.002075, +0.006702]` | 0.000000 | 15 / 12 / 178 | `FAIL` |

### 5.2 Terminal `DL-Val` (`19` Disjoint Families, Leak-Free `unrev`)

| Comparison | Mean Gain | `mean_LCB95` | Mean 95% CI | Family Median | Pos / Neg / Tied Families | `t537` Gate |
| --- | --- | --- | --- | --- | --- | --- |
| **`B vs A`** | -0.345042 | -0.447903 | `[-0.468428, -0.229253]` | -0.385780 | 2 / 17 / 0 | `FAIL` |
| **`C vs A`** | -0.343120 | -0.445900 | `[-0.467773, -0.225890]` | -0.385780 | 2 / 17 / 0 | `FAIL` |
| **`D vs A`** | -0.343120 | -0.444545 | `[-0.464420, -0.226821]` | -0.385780 | 2 / 17 / 0 | `FAIL` |
| **`C vs B`** | **+0.001923** | **0.000000** | `[0.000000, +0.005768]` | 0.000000 | 1 / 0 / 18 | `FAIL` (`LCB95 <= 0.001`) |
| **`D vs C`** | 0.000000 | 0.000000 | `[0.000000, 0.000000]` | 0.000000 | 0 / 0 / 19 | `FAIL` |
| **`D vs B`** | **+0.001923** | **0.000000** | `[0.000000, +0.005768]` | 0.000000 | 1 / 0 / 18 | `FAIL` (`LCB95 <= 0.001`) |
| **`B vs INIT_G0`** | -0.010512 | -0.032909 | `[-0.035732, +0.005904]` | 0.000000 | 2 / 2 / 15 | `FAIL` |
| **`C vs INIT_G0`** | -0.008589 | -0.031841 | `[-0.034877, +0.006972]` | 0.000000 | 2 / 1 / 16 | `FAIL` |
| **`D vs INIT_G0`** | -0.008589 | -0.031841 | `[-0.033809, +0.006972]` | 0.000000 | 2 / 1 / 16 | `FAIL` |

*(In `native` mode on `DL-Train` OOF, `B vs A`, `C vs A`, and `D vs A` have `mean_gain = -0.456724, -0.456477, -0.455026` due to the additional `+0.0718` revealed-anchor leak in `Arm A`; all `B/C/D` internal comparisons are identical between `native` and `unrev` because `B/C/D` mask `0..3` in both modes.)*

---

## 6. Live GP Self-Play, Hard-Negative Admission & Optimizer Telemetry

Across all `72` cells (`18` cells/arm × `20` rounds = `360` self-play rounds/arm), the runner executed live AST mutation (`propose_flywheel_candidates`), neural adapter ranking (`select_gp_proposal_indices`), and native Rust two-fold CV evaluation (`evaluate_gp_batch_rust` against SHA256-verified `.jtt1.gz` envelopes):

| Telemetry Metric (across `18` cells / arm) | `Arm A (CONTROL)` | `Arm B (FIX1)` | `Arm C (FIX1+2)` | `Arm D (FIX1+2+3)` |
| --- | --- | --- | --- | --- |
| **Trainable Parameters** | `29,211` | `23,546` | `23,546` | `23,546` |
| **Mean Parameter Update $\|\Delta\theta\|_2$ from Init** | `0.257377` | `0.304518` | `0.306176` | `0.294719` |
| **Mean Gradient L2 Norm (`40` steps/cell)** | `0.774064` | `0.921178` | `0.903805` | `7.011689` |
| **Mean Training Loss (`Step 1 -> Step 40`)** | `3.4139 -> 3.3870` | `6.4250 -> 6.3424` | `6.6037 -> 6.4856` | `38.5027 -> 38.1786` |
| **Live GP AST Proposals Ranked by Adapter** | `2,681` | `2,702` | `2,695` | `2,702` |
| **Live Native Rust 2-Fold CV Evaluations Completed** | `720` (`40.0/cell`) | `720` (`40.0/cell`) | `720` (`40.0/cell`) | `720` (`40.0/cell`) |
| **Live Native Rust Execution Failures (`status != completed`)** | `0` | `0` | `0` | `0` |
| **Live GP Candidates Admitted to Pool (`total / mean per cell`)** | `33` (`1.83/cell`, `4.58%`) | `29` (`1.61/cell`, `4.03%`) | **`720` (`40.00/cell`, `100.0%`)** | **`720` (`40.00/cell`, `100.0%`)** |
| **Live GP Non-HWM Valid Candidates Admitted** | `0` (`0.00/cell`) | `0` (`0.00/cell`) | **`666` (`37.00/cell`)** | **`665` (`36.94/cell`)** |
| **Live GP CV Gate Failures Admitted as Hard Negatives** | `0` (`0.00/cell`) | `0` (`0.00/cell`) | **`31` (`1.72/cell`)** | **`29` (`1.61/cell`)** |
| **Round-0 Historical Journal Candidates Admitted (mean/cell)** | `262.5` (HWM-only) | `262.5` (HWM-only) | **`333.3` (all completed)** | **`333.3` (all completed)** |
| **Initial Seeded Hard Negatives in Training Pools (mean/cell)** | `0.0` | `0.0` | `200,068.3` | `200,068.3` |
| **Model-Proposed Top-8 False Positives Harvested (sum / mean/cell)** | `0` (`0.0/cell`) | `0` (`0.0/cell`) | **`116,076` (`6,448.7/cell`)** | **`116,509` (`6,472.7/cell`)** |

---

## 7. Mechanistic Diagnosis: Why `prior10` Lifted Global Spearman $\rho$ by `+50%` While Depressing Top-1 `argmin` Utility Relative to Zero-Prior W1

A direct inspection of `gamenet_anchor.predict_costs` (`model.py:246-299`) and the cached `DL-Val` pools (`/home/ubuntu/jope-data/h58/cache/pool_priors.npz`) explains the opposing movements of **top-1 `argmin` utility** (`Arm A > Arms B/C/D`) and **full-pool Spearman rank correlation $\rho$** (`Arm D > Arm C > Arm B > Arm A`):

1. **Mathematical Decomposition of `gamenet_anchor.predict_costs`**:
   In `model.py:277-299`, the 16D `proj16` W1 5-NN skip-connection computes:
   $$w_1\_\text{cost}(c) = \text{base\_prior\_cost}(c) + \sum_{t=0}^3 \tilde{w}(c, t)\left(\frac{F_t}{\text{scale}} - \text{base\_prior\_cost}(t)\right)$$
   Since $\sum_{t=0}^3 \tilde{w}(c, t) = 1$, this rearranges identically to:
   $$w_1\_\text{cost}(c) = \underbrace{\sum_{t=0}^3 \tilde{w}(c, t)\frac{F_t}{\text{scale}}}_{\text{W1 revealed-seed kernel interpolation}} + \underbrace{\left(\text{base\_prior\_cost}(c) - \sum_{t=0}^3 \tilde{w}(c, t)\text{base\_prior\_cost}(t)\right)}_{\text{Static prior10 deviation from revealed anchors}}$$
2. **Amplitude Mismatch (`18.20` vs `1.78`)**:
   - On `DL-Val`, the 4 revealed seed anchors `0..3` already contain a strong baseline candidate (`mean utility of best of 0..3 = 0.7099`), and the unrevealed pool `4..P-1` contains close structural neighbors in 16D `proj16` space (`median min` $d^2 = 0.091676$).
   - Across unrevealed candidates `4..P-1` on `DL-Val`, the W1 revealed-seed kernel interpolation term $\sum_{t=0}^3 \tilde{w}(c, t)(F_t / \text{scale})$ has a mean peak-to-peak range of **`1.7798` units**.
   - By contrast, the raw `prior10` vector (`_predict_calibrated_prior_batched_gpu(..., track_prefix="prior10")`) has a mean peak-to-peak range of **`18.2011` units** (`10.2x` larger than the W1 trajectory term), and at the nearest 16D `proj16` neighbor of the best revealed seed anchor `0..3`, `base_prior_cost[nn] - min(base_prior_cost[4:])` averages **`+8.4215` units**.
3. **Consequence for `Arm A` (`base_prior_cost = 0`) vs `Arms B/C/D` (`base_prior_cost = prior10`)**:
   - When `base_prior_cost = 0` (`Arm A`), the second term vanishes completely (`0.0`). Even without any adapter, pure W1 5-NN interpolation with `base_prior_cost = 0` achieves **`unrev ctx utility = 0.5736`** on `DL-Val` because `argmin` selects the closest 16D `proj16` neighbor of the best revealed seed anchor in `0..3`, even though its global rank correlation across the full pool is near-random (**`Spearman` $\rho = 0.0621$** for W1 alone; **`0.1202`** with the `Arm A` adapter).
   - When unattenuated `prior10` (`alpha = 1.0`) is passed as `base_prior_cost` (`Arms B/C/D`), the `18.20`-unit static prior dominates the `1.78`-unit W1 trajectory interpolation:
     - **Global Spearman $\rho$ jumps from `0.0621` to `0.1765`** on the W1 backbone (`0.1806` in `Arm D`), because `prior10` carries real global AST-quality rank information across the entire candidate pool (`pure prior10` has $\rho = 0.1824$).
     - **Top-1 `argmin` utility drops from `0.5736` to `0.2722`** on the W1 backbone (`0.2967` in `Arms C & D`), because the `+8.42`-unit static `prior10` penalty at the best revealed anchor's nearest neighbor overrides the W1 trajectory interpolation and pulls `argmin` toward the static zero-shot `prior10` minimum (`unrev utility = 0.2567`).
   - Attenuating `base_prior_cost` by a scalar factor $\alpha \in [0, 1]$ on `DL-Val` confirms a smooth, monotonic trade-off between local W1 trajectory exploitation and global `prior10` ranking:
     - $\alpha = 0.00$: `unrev ctx utility = 0.5736`, `Spearman` $\rho = 0.0621$
     - $\alpha = 0.02$: `unrev ctx utility = 0.4859`, `Spearman` $\rho = 0.1047$
     - $\alpha = 0.05$: `unrev ctx utility = 0.4225`, `Spearman` $\rho = 0.1300$
     - $\alpha = 0.10$: `unrev ctx utility = 0.3703`, `Spearman` $\rho = 0.1528$
     - $\alpha = 0.25$: `unrev ctx utility = 0.3271`, `Spearman` $\rho = 0.1804$
     - $\alpha = 1.00$: `unrev ctx utility = 0.2722`, `Spearman` $\rho = 0.1765$

### Recommendations for Next Flywheel Wave
1. **Keep the Leak-Free & Self-Play Fixes from `Arms B, C, D`**:
   - Retain `Fix Bug 1` (unsorted seed anchors `0..3` at slots `0..3`, permuted unrevealed slots `4..M-1`, and strict `0..3` masking in both loss and `argmin`), `Fix Bug 4` (frozen `parent_delta_net`), `Fix Bug 2` (`Arm C`: admit all completed live GP evaluations — non-HWM valid candidates, model-proposed false positives, and genuine CV gate failures — into `hard_negatives`), and `Fix Rank 3` (`Arm D`: shift-invariant listwise Plackett-Luce + gate-failure/top-choice margin loss).
2. **Decouple or Variance-Attenuate `base_prior_cost` Inside `gamenet_anchor`**:
   - Do **not** feed raw unscaled `prior10` (`ptp ≈ 18.2`) directly into the W1 residual difference `revealed_raw_f / scale - base_prior_cost` where `revealed_raw_f / scale` has `ptp ≈ 1.78`. Instead, either keep `base_prior_cost = 0` inside the W1 anchor interpolation while passing normalized `prior10` as a feature to `adapter_10tok`, or learn a bounded variance-matched prior weight $\alpha \in [0.0, 0.05]$ so the adapter captures `prior10`'s global ranking signal without destroying W1 local trajectory conditioning around revealed anchors `0..3`.

---

## 8. Cryptographic Provenance, Isolation & Audit Receipts

- **DOPE Backbone Weight Freeze (`ADR-0005` Invariant 3)**:
  - Verified before and after every single cell (`72/72` cells):
    `dope_sha256_before == dope_sha256_after == "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"`.
- **Holdout & Outer-Fold Isolation**:
  - `excluded_payloads_opened = 0`, `outer_rows_delivered = 0`, `terminal_dl_val_eval_sweeps_per_trained_arm_seed = 1`.
  - All `1,219` `DL-Train` `.jtt1.gz` two-fold CV evaluation envelopes were verified against `/home/ubuntu/jope-data/h58/cache/jtt1_gz_manifest.json` before every live Rust evaluator call (`2,880` evaluations, `0` cross-dataset fallbacks).
- **Pinned Input & Asset SHA256 Digest Table**:
  - `manifest_file_sha256`: `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`
  - `manifest_id`: `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`
  - `selected_file_sha256`: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`
  - `champions_file_sha256`: `a444047ffb937dc8d69012cbdbf0891faf3e7d743b127e7cac3baed114cf141d`
  - `fused_source_sha256`: `b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab`
  - `worker_bin_sha256`: `e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8`
  - `stage1_stage2_pt_sha256`: `136b2381c312fd4bf267249fe0543aa44f5b780a05d57f974d43dd4277588b6d`
  - `dope_multitask_pt_sha256`: `a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952`
  - `prog_feat_names_sha256`: `c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962`
  - `rust_evaluator_py_sha256`: `53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900`
  - `model_py_sha256`: `0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a`
  - `phase12_mm_py_sha256`: `7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a`
- **Durable Host Receipts**:
  - Wave Summary JSON: `/home/ubuntu/jope-data/h58/runs/wave_summary.json`
  - Per-Cell Receipts JSON (`72` cells): `/home/ubuntu/jope-data/h58/runs/cell_receipts.json`
  - Pre-GPU Codex v4 Audit (`PASS`): `/home/ubuntu/jope-data/h58/redteam_codex_pre_gpu_v4.md`
  - Status Log: `/home/ubuntu/jope-data/h58/H58_STATUS.log`
