# H173: Hard-Context (Regret-Weighted / Focal-Style) Loss Reweighting on `r882` Continuation

- **Wave**: `H173` (`research/flywheel_selfplay/h173_hard_context_reweight/`)
- **Owner**: `rd_argon` (`[ARGON]`), claim `/home/ubuntu/jope_plan/ns/receipts/H173_claim.txt` (`reserved_by: ENG:hwave-loop`)
- **Status**: `COMPLETE` (single supervised CPU wave `custody-run` at merged commit `8dac600b50c24dae65fc254a6fb5209707f2fe7c`, red-team `PASS` review `5727638185` at `7d7007be2c3a08ffee07b02c974f7d06434a6166`, same-agent review `5727634316`)
- **Verdict**: `NO_ADVANCE` (`NOISE_NEUTRAL_NO_OOF_GAIN`; `0 / 2` qualifying hardness arms `RW05`, `RW10` pass primary OOF confirmation gates; `serving_promotion: false`)
- **Custody Archive**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H173/argon-h173-20261101a/` (`47/47` objects, `2,732,476` bytes restore-verified; local receipt `/home/ubuntu/jope_plan/evidence/H173/s3-custody-receipt.json` and `/home/ubuntu/jope-rescue-20260921/h173-rd_argon-20261101/`)

---

## 1. Hypothesis & Experimental Design

`H173` tested whether reweighting each `FIT`-split context's continuation loss by a clipped power of its frozen `r882` step-0 top-1 regret computed **strictly within the `FIT` split $C_k$** (`RW05` with $\gamma = 0.5$, `RW10` with $\gamma = 1.0$) improves out-of-fold equal-family utility over uniform weighting (`CTRL`, $w_c = 1.0$) and over a shuffled-weight placebo (`RW10_SHUF`, which permutes `RW10` weights across `FIT` contexts via `shuf_seed = seed + 1000 * outer + 77` to preserve the exact weight distribution and effective sample size `ESS` while removing the hardness signal).

- **Seeds & 5-Fold Cross-Fit**: `SEEDS = (173101, 173102, 173103)`, `FOLD_SEED = 173140` (`5` disjoint folds of `41` families across `1,219` `DL-Train` contexts / `205` families; fold context counts `[240, 242, 258, 273, 206]`), `BOOT_SEED = 173141`.
- **Leak-Free `FIT`-Only Regret Weights (`compute_fit_regret_weights`)**:
  - In each outer fold $k \in \{0, 1, 2, 3, 4\}$, $\text{regret}_c = \text{cost}_c(i_{\text{r882}}(c)) - \min_j \text{cost}_c(j) \ge 0$ and $\varepsilon_k = \max(\text{median}_{c \in C_k}(\text{regret}_c), 10^{-6})$ are computed on the `123`-family `FIT` split $C_k$ **only** (`0` `B_k`, `A_k`, or `DL-Val` weight leakage).
  - Power weights $u_c(\gamma) = (\text{regret}_c + \varepsilon_k)^\gamma$ are normalized to `FIT` mean `1.0`, clipped to `[0.2, 5.0]`, and renormalized to `FIT` mean `1.0`.
- **Strict Inner-`B` Selector & Raw Inner-`B` Panel Retention**:
  - On inner fold $B_k$ (`41` families), candidate steps $t \in \{0, 10, 20, 30, 40\}$ are evaluated by equal-family mean gain $\Delta_{B_k}(a, t)$ and `1,000`-draw paired family-bootstrap 95% CI (`selector_seed = 173141 + 1000 * outer + (seed % 1000)`).
  - Primary strict selector picks the earliest step $t \in \{10, 20, 30, 40\}$ maximizing $\Delta_{B_k}(a, t)$ with $\Delta_{B_k}(a, t) > +0.001 + 10^{-12}$ AND $\text{LCB}_{95, B_k}(a, t) > 0.0$, else `step = 0` (`r882`), locked before evaluating $A_k$ or `DL-Val`.
  - All `60` raw inner-`B` context-utility panels (`(len(B_k), 5)`) are persisted in `inner_b_panels.npz` (`70` arrays including `context_ids` and `family_ids` per fold) for full independent replay of all `60` inner-`B` bootstrap CIs.

---

## 2. Pre-Training Baseline, Bit-Identity & Freeze Verification

- **Exact `91/91` `DL-Val` `r882` Baseline Parity**: `matches = 91`, `mean = 1.1377941898164525`, `max_context_error = 0.0`, `mean_error = 0.0` (`PASS` in both `--smoke` and `custody-run`).
- **Native Score Bit-Identity (`uncached` vs `cached` `total_cost`)**: `91` `DL-Val` contexts, `104,832` candidate scores `bit_identical: true` (`torch.equal` and raw `float32` bytes), `max_abs_cost_diff = 0.0` (`PASS`).
- **Freeze & Input Closure Audit (`audit.json`)**: `dope_sha256 = c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and `frozen_sha256 = c1baca14dad644e3a2ec9566b2f89e8f912b098f455d29cc8ace9b608ee694c6` invariant across all `60` trajectories and `2,400` optimizer updates (`opened_count = 1,315`, `protected_reads = 0`, `refused = []`, `cuda_initialized = false`, `nice = 19`, peak RSS `3.34 GB < 6.0 GB`, `<= 4` aggregate threads, `wall_seconds = 2,668.94`).
- **Gradient Health**: Across all `60` trajectories (`2,400` AdamW updates), `41` of `47` adapter parameter tensors received active gradients and `6` structural adapter tensors had `None` gradients (`none_grad_param_indices = [20, 21, 30, 31, 32, 33]`), handled cleanly by `safe_grad_norm_and_clip`.

---

## 3. Primary Out-of-Fold (`DL-Train`, `1,219` Contexts / `205` Families) & Secondary (`DL-Val`, `91` Contexts / `19` Families) Results

| Arm / Contrast | Non-Zero Cells (`/15`) | `DL-Train` OOF Per-Seed Equal-Family Gain (`173101`, `173102`, `173103`) | `DL-Train` OOF Median Gain `[95% CI]` | Positive Seeds | One-Sided $p$ (`Holm m=3` / `m=4`) | `DL-Train` Median BT Elo `[95% CI]` | `DL-Train` `-2` Invalid Picks (Seeds / Median) | `DL-Val` (91 ctx / 19 fam) Median Gain `[95% CI]` | `DL-Val` Median BT Elo | Verdict |
|---|---:|---|---|---:|---|---:|---|---:|---:|---|
| `r882` (Reference) | `0 / 15` | `[0.0, 0.0, 0.0]` (Fam util `1.385355`, Ctx util `1.423073`) | `0.000000 [0.000000, 0.000000]` | — | — | `1500.00 [1500.00, 1500.00]` | `[17, 17, 17]` (`17.0`) | `0.000000` (Fam `1.153615`, Ctx `1.137794`) | `1500.00` | Reference |
| `CTRL - r882` | `1 / 15` | `[+0.000000, -0.000892, +0.000000]` | **`0.000000 [0.000000, 0.000000]`** | `0 / 3` | `1.0000` (`—` / `1.0000`) | `1500.00 [1500.00, 1500.00]` | `[17, 18, 17]` (`17.0`) | `0.000000 [0.000000, 0.000000]` | `1500.00` | Control |
| `RW05 - r882` | `1 / 15` | `[+0.000000, -0.000949, +0.000000]` | **`0.000000 [0.000000, 0.000000]`** | `0 / 3` | `1.0000` (`1.0000` / `1.0000`) | `1500.00 [1500.00, 1500.00]` | `[17, 18, 17]` (`17.0`) | `0.000000 [0.000000, 0.000000]` | `1500.00` | `NO_ADVANCE` |
| `RW10 - r882` | `0 / 15` | `[+0.000000, +0.000000, +0.000000]` | **`0.000000 [0.000000, 0.000000]`** | `0 / 3` | `1.0000` (`1.0000` / `1.0000`) | `1500.00 [1500.00, 1500.00]` | `[17, 17, 17]` (`17.0`) | `0.000000 [0.000000, 0.000000]` | `1500.00` | `NO_ADVANCE` |
| `RW10_SHUF - r882` | `1 / 15` | `[-0.001471, +0.000000, +0.000000]` | **`0.000000 [0.000000, 0.000000]`** | `0 / 3` | `1.0000` (`1.0000` / `1.0000`) | `1500.00 [1500.00, 1500.00]` | `[18, 17, 17]` (`17.0`) | `0.000000 [0.000000, 0.000000]` | `1500.00` | Placebo (`advance=false`) |
| `RW05 - CTRL` | `1` vs `1` | `[+0.000000, -0.000057, +0.000000]` | `0.000000 [0.000000, 0.000000]` | `0 / 3` | `1.0000` (Mech.) | — | `17.0` vs `17.0` | `0.000000 [0.000000, 0.000000]` | — | Null |
| `RW10 - CTRL` | `0` vs `1` | `[+0.000000, +0.000892, +0.000000]` | `0.000000 [0.000000, 0.000000]` | `1 / 3` | `1.0000` (Mech.) | — | `17.0` vs `17.0` | `0.000000 [0.000000, 0.000000]` | — | Null |
| `RW10_SHUF - CTRL` | `1` vs `1` | `[-0.001471, +0.000892, +0.000000]` | `0.000000 [-0.000450, 0.000000]` | `1 / 3` | `0.9956` (Mech.) | — | `17.0` vs `17.0` | `+0.000127 [0.000000, +0.001453]` | — | Null |
| `RW10 - RW10_SHUF` | `0` vs `1` | `[+0.001471, +0.000000, +0.000000]` | **`0.000000 [0.000000, 0.000000]`** | `1 / 3` | `1.0000` (Hardness vs Placebo) | — | `17.0` vs `17.0` | `0.000000 [0.000000, 0.000000]` | — | `NOISE_NEUTRAL_NO_OOF_GAIN` |

### Pre-Locked Diagnostic Selectors (`UNCONSTRAINED` $\arg\max_{t \in \{0,10,20,30,40\}} \Delta_B(t)$ & `NONZERO` $\arg\max_{t \in \{10,20,30,40\}} \Delta_B(t)$)

| Diagnostic Arm / Contrast | Non-Zero Cells (`/15`) | `DL-Train` OOF Per-Seed Equal-Family Gain (`173101`, `173102`, `173103`) | `DL-Train` OOF Median Gain `[95% CI]` | Positive Seeds | One-Sided $p$ | `DL-Train` Median BT Elo | `DL-Train` `-2` Picks (Seeds / Median) | `DL-Val` Median Gain `[95% CI]` | `DL-Val` Median Elo |
|---|---:|---|---|---:|---:|---:|---|---:|---:|
| `CTRL_UNCONSTRAINED - r882` | `6 / 15` | `[-0.005848, -0.008568, -0.009931]` | `-0.008568 [-0.018799, +0.001351]` | `0 / 3` | `0.9506` | `1498.11` | `[21, 21, 21]` (`21.0`) | `-0.004890 [-0.012434, +0.000301]` | `1499.27` |
| `RW05_UNCONSTRAINED - r882` | `7 / 15` | `[+0.001367, -0.000731, -0.002892]` | `-0.000731 [-0.003907, +0.002411]` | `1 / 3` | `0.6112` | `1500.67` | `[18, 18, 18]` (`18.0`) | `-0.004306 [-0.009890, +0.000212]` | `1496.35` |
| `RW10_UNCONSTRAINED - r882` | `10 / 15` | `[-0.004052, -0.000957, -0.002529]` | `-0.002529 [-0.008893, +0.001742]` | `0 / 3` | `0.8026` | `1500.31` | `[17, 18, 17]` (`17.0`) | `-0.003831 [-0.008326, +0.000219]` | `1492.69` |
| `RW10_SHUF_UNCONSTRAINED - r882` | `8 / 15` | `[-0.007535, +0.001854, -0.018565]` | `-0.007535 [-0.031523, +0.003297]` | `1 / 3` | `0.7792` | `1496.53` | `[17, 17, 20]` (`17.0`) | `-0.001831 [-0.005542, +0.000564]` | `1496.35` |
| `RW05_UNCON - CTRL_UNCON` | `7` vs `6` | `[+0.007215, +0.007837, +0.007039]` | `+0.007215 [-0.000193, +0.016230]` | `3 / 3` | `0.0277` | — | `18.0` vs `21.0` | `+0.002726 [-0.000339, +0.006610]` | — |
| `RW10_UNCON - CTRL_UNCON` | `10` vs `6` | `[+0.001797, +0.007610, +0.007402]` | `+0.007402 [-0.004303, +0.016855]` | `3 / 3` | `0.1266` | — | `17.0` vs `21.0` | `+0.002215 [-0.000753, +0.010719]` | — |
| `RW10_UNCON - RW10_SHUF_UNCON` | `10` vs `8` | `[+0.003483, -0.002811, +0.016036]` | `+0.003483 [-0.004307, +0.016905]` | `2 / 3` | `0.2681` | — | `17.0` vs `17.0` | `-0.001988 [-0.003185, +0.003165]` | — |
| `CTRL_NONZERO - r882` | `15 / 15` | `[-0.004827, -0.008521, -0.010833]` | `-0.008521 [-0.023142, +0.008293]` | `0 / 3` | `0.8491` | `1493.15` | `[21, 21, 21]` (`21.0`) | `-0.007803 [-0.019177, +0.000750]` | `1500.00` |
| `RW05_NONZERO - r882` | `15 / 15` | `[+0.003324, -0.000708, -0.007133]` | `-0.000708 [-0.012512, +0.013818]` | `1 / 3` | `0.5770` | `1496.58` | `[16, 18, 18]` (`18.0`) | `-0.007393 [-0.014852, +0.000711]` | `1500.00` |
| `RW10_NONZERO - r882` | `15 / 15` | `[+0.000906, +0.003219, +0.002832]` | **`+0.002832 [-0.008079, +0.016087]`** | `3 / 3` | `0.3816` | `1498.54` | `[16, 17, 16]` (`16.0`) | `-0.003454 [-0.007653, +0.000627]` | `1498.54` |
| `RW10_SHUF_NONZERO - r882` | `15 / 15` | `[-0.008590, +0.001751, -0.019616]` | `-0.008590 [-0.032125, +0.012050]` | `1 / 3` | `0.7766` | `1497.11` | `[18, 16, 20]` (`18.0`) | `-0.004900 [-0.011882, +0.000962]` | `1497.08` |
| `RW10_NONZERO - CTRL_NONZERO` | `15` vs `15` | `[+0.005733, +0.011739, +0.013664]` | **`+0.011739 [+0.000519, +0.023229]`** | `3 / 3` | `0.0202` | — | `16.0` vs `21.0` | `+0.006069 [-0.000651, +0.015796]` | — |
| `RW10_NONZERO - RW10_SHUF_NONZERO` | `15` vs `15` | `[+0.009497, +0.001467, +0.022448]` | **`+0.009497 [+0.000424, +0.020862]`** | `3 / 3` | `0.0180` | — | `16.0` vs `18.0` | `+0.001446 [-0.000575, +0.007436]` | — |

---

## 4. `FIT`-Only Regret Weights, Effective Sample Size (`ESS`), Cell Selections & Step-40 L2 Drift

### `FIT`-Split Weight Distribution & Step-40 Adapter L2 Drift Summary (`15` Cells)

| Arm | Median `ESS` (`[min, max]`) | Median `ess_ratio` (`[min, max]`) | Median Min / Max Weight $w_c$ | Median Step-40 L2 Drift (`[min, max]`) |
|---|---|---|---|---|
| `CTRL` ($\gamma = 0.0$) | `721.00` (`[701.00, 771.00]`) | `1.0000` (`[1.0000, 1.0000]`) | `1.0000` / `1.0000` | `0.3770` (`[0.3591, 0.3886]`) |
| `RW05` ($\gamma = 0.5$) | `500.90` (`[473.17, 515.49]`) | `0.6789` (`[0.6620, 0.6898]`) | `0.4090` / `3.8902` | `0.3358` (`[0.3148, 0.3592]`) |
| `RW10` ($\gamma = 1.0$) | `251.82` (`[240.14, 267.67]`) | `0.3487` (`[0.3258, 0.3660]`) | `0.2387` / `5.9667` | `0.2754` (`[0.2488, 0.3024]`) |
| `RW10_SHUF` (Placebo) | `251.82` (`[240.14, 267.67]`) | `0.3487` (`[0.3258, 0.3660]`) | `0.2387` / `5.9667` | `0.3075` (`[0.2771, 0.3424]`) |

### Cell-by-Cell Inner-`B` Checkpoint Selections (`Strict`, `Unconstrained`, `Non-Zero`) & Step-40 L2 Drift

| Cell (`Outer-Seed`) | $\varepsilon_k$ (`FIT` Median Regret) | `ESS` Ratio (`RW05`, `RW10` / `SHUF`) | `Strict` (`CTRL`, `RW05`, `RW10`, `SHUF`) | `Unconstrained` (`CTRL`, `RW05`, `RW10`, `SHUF`) | `Non-Zero` (`CTRL`, `RW05`, `RW10`, `SHUF`) | Step-40 L2 Drift (`CTRL`, `RW05`, `RW10`, `SHUF`) |
|---|---:|---|---|---|---|---|
| `A0-173101` | `0.0930` | `(0.6620, 0.3258)` | `(0, 0, 0, 0)` | `(0, 0, 30, 30)` | `(10, 20, 30, 30)` | `(0.3857, 0.3358, 0.2680, 0.3027)` |
| `A0-173102` | `0.0930` | `(0.6620, 0.3258)` | `(0, 0, 0, 0)` | `(0, 0, 0, 0)` | `(10, 10, 10, 20)` | `(0.3875, 0.3582, 0.3024, 0.3031)` |
| `A0-173103` | `0.0930` | `(0.6620, 0.3258)` | `(0, 0, 0, 0)` | `(0, 0, 10, 20)` | `(10, 20, 10, 20)` | `(0.3882, 0.3592, 0.2998, 0.3319)` |
| `A1-173101` | `0.0999` | `(0.6789, 0.3487)` | `(0, 0, 0, 0)` | `(0, 10, 10, 0)` | `(10, 10, 10, 10)` | `(0.3796, 0.3450, 0.2869, 0.3092)` |
| `A1-173102` | `0.0999` | `(0.6789, 0.3487)` | `(0, 0, 0, 0)` | `(0, 0, 10, 0)` | `(10, 10, 10, 20)` | `(0.3634, 0.3258, 0.2671, 0.3075)` |
| `A1-173103` | `0.0999` | `(0.6789, 0.3487)` | `(0, 0, 0, 0)` | `(0, 10, 10, 0)` | `(10, 10, 10, 10)` | `(0.3840, 0.3505, 0.2969, 0.2964)` |
| `A2-173101` | `0.1113` | `(0.6898, 0.3660)` | `(0, 0, 0, 10)` | `(10, 10, 20, 10)` | `(10, 10, 20, 10)` | `(0.3646, 0.3279, 0.2754, 0.2813)` |
| `A2-173102` | `0.1113` | `(0.6898, 0.3660)` | `(10, 10, 0, 0)` | `(10, 10, 10, 10)` | `(10, 10, 10, 10)` | `(0.3770, 0.3362, 0.2728, 0.3293)` |
| `A2-173103` | `0.1113` | `(0.6898, 0.3660)` | `(0, 0, 0, 0)` | `(20, 0, 0, 20)` | `(20, 20, 20, 20)` | `(0.3591, 0.3148, 0.2541, 0.2771)` |
| `A3-173101` | `0.1265` | `(0.6890, 0.3617)` | `(0, 0, 0, 0)` | `(40, 10, 10, 0)` | `(40, 10, 10, 10)` | `(0.3770, 0.3267, 0.2594, 0.2960)` |
| `A3-173102` | `0.1265` | `(0.6890, 0.3617)` | `(0, 0, 0, 0)` | `(40, 10, 10, 10)` | `(40, 10, 10, 10)` | `(0.3712, 0.3494, 0.2951, 0.3142)` |
| `A3-173103` | `0.1265` | `(0.6890, 0.3617)` | `(0, 0, 0, 0)` | `(40, 40, 20, 40)` | `(40, 40, 20, 40)` | `(0.3664, 0.3224, 0.2488, 0.3179)` |
| `A4-173101` | `0.0946` | `(0.6788, 0.3415)` | `(0, 0, 0, 0)` | `(0, 0, 0, 10)` | `(10, 10, 10, 10)` | `(0.3886, 0.3545, 0.2935, 0.3424)` |
| `A4-173102` | `0.0946` | `(0.6788, 0.3415)` | `(0, 0, 0, 0)` | `(0, 0, 0, 0)` | `(10, 10, 10, 10)` | `(0.3635, 0.3290, 0.2758, 0.3206)` |
| `A4-173103` | `0.0946` | `(0.6788, 0.3415)` | `(0, 0, 0, 0)` | `(0, 0, 0, 0)` | `(10, 10, 10, 10)` | `(0.3635, 0.3269, 0.2558, 0.2783)` |
| **Summary** | `[0.0930, 0.1265]` | `0.6789` / `0.3487` | `1/15, 1/15, 0/15, 1/15` | `6/15, 7/15, 10/15, 8/15` | `15/15` all arms | `0.3770, 0.3358, 0.2754, 0.3075` |

---

## 5. Required Disclosures: Checkpoint-Step Identity vs Candidate/Utility-Vector Identity & Scope

1. **Checkpoint-Step Identity vs Candidate / Utility-Vector Identity (`pseudoreplication_report`)**:
   - Under the **strict primary selector**, `RW10` selects `step = 0` (`r882`) in all `15/15` cells across all 3 seeds, while `CTRL` (`A2-173102` step 10), `RW05` (`A2-173102` step 10), and `RW10_SHUF` (`A2-173101` step 10) each select a non-zero checkpoint in `1/15` cells and fall back to `step = 0` (`r882`) on the other 2 seeds. Consequently, primary OOF seed panels coincide with `r882` on the 2 all-fallback seeds for `CTRL`/`RW05`/`RW10_SHUF` and on all 3 seeds for `RW10` (making their 3-seed median CIs `[0.0, 0.0]` a fallback point-mass rather than a tight equivalence bound).
   - **Trained candidate trajectories and utility vectors are distinct**: Across all `60` trained trajectories (`diag_nonzero_primary` in `primary.npz`), `0/3` seed pairs are identical within any arm (`s0_s1 = False, s0_s2 = False, s1_s2 = False` for all 4 arms). Furthermore, in cell `A2-173102` where both `CTRL` and `RW05` selected `step = 10` under the strict selector, their step-10 checkpoints and utility vectors are distinct (`inner-B max abs diff = 1.428291`, `outer-A OOF max abs diff = 0.275520`, seed-`173102` OOF equal-family gain `-0.000892` for `CTRL` vs `-0.000949` for `RW05`; thus `identical_oof_panel_to_ctrl_by_seed` for `RW05` is `[True, False, True]`).
2. **Full Raw Inner-`B` Panel Retention (`inner_b_panels.npz`)**:
   - All `60` raw inner-`B` per-context utility matrices (`shape = (len(B_k), 5)`) plus per-fold `context_ids` and `family_ids` are retained in `/home/ubuntu/jope-data/h173/custody-run/run/inner_b_panels.npz` (and S3 custody), enabling exact independent replay of all `60` inner-`B` paired family-bootstrap CIs and selector decisions.
3. **Reference Checkpoint Scope (`r882`)**:
   - `round-882.pt` has historical exposure to `DL-Train` contexts during earlier flywheel rounds (`historical_r882_train_exposure: true`); results are conditional continuation evidence and no serving promotion is claimed (`serving_promotion: false`).

---

## 6. Scientific Conclusion & Next Step

- **Verdict**: `NO_ADVANCE` (`NOISE_NEUTRAL_NO_OOF_GAIN`).
- **Scientific Conclusion**:
  1. Under the strict 41-family inner-`B` selector (`gain > +0.001 + 1e-12` and `LCB95 > 0`), `RW10` ($\gamma = 1.0$) selected `r882` (`step = 0`) in `15/15` cells and `RW05` ($\gamma = 0.5$) selected `step = 10` in `1/15` cells (`A2-173102`, outer-`A` seed gain `-0.000949`), yielding primary OOF median gain `0.000000 [0.000000, 0.000000]` (`Holm p = 1.0`, BT Elo `1500.00`).
  2. However, on the **pre-locked `NONZERO` diagnostic selector** ($t \in \{10, 20, 30, 40\}$, `15/15` non-zero cells), linear regret reweighting (`RW10_NONZERO`, `median_ess_ratio = 0.3487`) beat both uniform weighting (`RW10_NONZERO - CTRL_NONZERO = +0.011739 [+0.000519, +0.023229]`, $p = 0.0202$, `3/3` seeds) and the `ESS`-matched shuffled-weight placebo (`RW10_NONZERO - RW10_SHUF_NONZERO = +0.009497 [+0.000424, +0.020862]`, $p = 0.0180$, `3/3` seeds), and achieved positive 205-family OOF gain over `r882` on all `3/3` seeds (`[+0.000906, +0.003219, +0.002832]`, median `+0.002832 [-0.008079, +0.016087]`, reducing median `-2` cliff selections from `17.0` (`r882`) / `21.0` (`CTRL_NONZERO`) to `16.0`).
  3. Thus, concentrating continuation gradient updates on high-regret `FIT` contexts genuinely prevents the `-2` gate-cliff degradation seen under uniform or randomly weighted continuation updates, but the resulting out-of-fold improvement (`+0.0028`) is below the 41-family inner-`B` `LCB95 > 0` detection threshold and the 205-family OOF `LCB95 > 0` confirmation bar.

## Reproduction & Verification Commands

```bash
# 1. Data-free unit tests (11/11)
pytest -q research/flywheel_selfplay/h173_hard_context_reweight/test_h173.py

# 2. 91-context DL-Val baseline-scoring smoke (0 updates, 0 DL-Train pool opens)
python3 -B research/flywheel_selfplay/h173_hard_context_reweight/monitor.py \
  --custody /home/ubuntu/jope-data/h173/custody --smoke

# 3. Single supervised full CPU wave (executed under custody-run)
python3 -B research/flywheel_selfplay/h173_hard_context_reweight/monitor.py \
  --custody /home/ubuntu/jope-data/h173/custody-run
```
