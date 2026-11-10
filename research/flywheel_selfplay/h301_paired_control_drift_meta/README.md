# H301 (`h301_paired_control_drift_meta`) — Paired-Control Drift Meta-Analysis & Winner's-Curse Audit

- **Wave ID**: `H301`
- **Slug**: `h301_paired_control_drift_meta`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Mode**: **ANALYSIS ONLY** (`xbabe2`, `nice -n 19 taskset -c 0-1`, peak RSS `100.0 MiB <= 2 GiB`, `0` neural updates, `0` neural forwards, `0` new `dev91` scorings, no GPU)
- **Preregistration Commit**: `1dfbc2736e580288f7dbef0e13b1bdb93a986b1e` (pushed to `origin/v1/next` before executing the meta-analysis)
- **DOPE SHA-256**: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (untouched)
- **Primary Verdict**: **`SELECTION_NOISE_PLATEAU`**

---

## 1. Executive Summary & Decision Gate

Every recent adapter-training wave initialized from `r3175` (`1.1389572403748425` on `dev91` identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`) lost to `r3175`, **including its paired baseline-loss control** (`SmoothL1 + 0.1 KL`). `H301` tests whether `r3175` is a selected maximum of a noisy `dev91` search (winner's curse over `~4,800` flywheel rounds) so that any further `TRAIN` update regresses toward `~1.13` (`SELECTION_NOISE_PLATEAU`), or whether multi-step unconstrained updates actively drift away from a genuinely superior basin (`OPTIMIZER_PLATEAU`).

All pre-declared criteria in [`PREREG.md`](PREREG.md) Section 4(c) pass for **`SELECTION_NOISE_PLATEAU`**:

| Pre-Declared Criterion | Value | 19-Family `10k` Bootstrap `95%` CI / Threshold | Gate Result |
| --- | --- | --- | --- |
| **Step 1: `H297` & `H298` S3 exact-`VersionId` regression** | `H297` `-0.008461025318455606`; `H298` `-0.033528458894059865` | Bit-for-bit match against committed `metrics.json` (`True` / `True`) | **PASS** |
| **Criterion 1 (Primary — All 7 Wave/Horizon Groups `2..256` updates)**: Median-over-waves control lift `< 0` with `CI95` excluding `0` | **`-0.0334624321`** (`1484.72` Elo) | **`[-0.0908240568, -0.0003781744]`** (`98.33%` of draws `< 0`, `UCB95 < 0`) | **PASS (`True`)** |
| **Criterion 1 (Sensitivity — Short-Horizon `2..8` update waves `t1820, t1845, H297, H298`)** | `-0.0084131995` (`1490.45` Elo) | `[-0.0190918802, +0.0001758830]` (`96.12%` of draws `< 0`; `n=2` changes only 2 of 91 contexts) | Sensitivity reported |
| **(a) Drift Slope vs `n_updates` (Linear per update, `2..256`)** | **`-0.0004519110` / update** | **`[-0.0008234340, -0.0001124856]`** (strictly `< 0`) | **Strictly negative** |
| **(a) Drift Slope vs $\log_2(\text{n\_updates})$ (Per doubling, `2..256`)** | **`-0.0170536518` / doubling** | **`[-0.0275982059, -0.0068708343]`** (strictly `< 0`) | **Strictly negative** |
| **Criterion 2 (Marginal 19-family SE)**: $(u_{r3175} - q_{0.95}) < 2 \cdot \text{SE}_{\text{fam}}(r3175)$ | $u_{r3175} - q_{0.95} = +0.0035184519$ | $\text{SE}_{\text{fam}}(r3175) = 0.2386372862 \implies \mathbf{0.0147 \times \text{SE}_{\text{fam}}} < 2.0$ | **PASS (`True`)** |
| **Criterion 2 (Paired 19-family SE)**: $(u_{r3175} - q_{0.95}) < 2 \cdot \text{SE}_{\text{fam}}(r3175 - q_{0.95})$ | $u_{r3175} - q_{0.95} = +0.0035184519$ | $\text{SE}_{\text{fam}}(\Delta_{q95}) = 0.0024511857 \implies \mathbf{1.4354 \times \text{SE}_{\text{fam}}} < 2.0$ | **PASS (`True`)** |

---

## 2. Step 1 — Exact-`VersionId` S3 Regression Verification (`regression_receipt.json`)

Before running the meta-analysis, both `H297` and `H298` S3 archives were fetched by exact `VersionId` and recomputed from scratch:

1. **`H297` (`h297_robust_ctxnorm_clipped_loss`)**:
   - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H297/astra-t0240/results.tar.gz` (`VersionId = K7zNgkztjDJ32iTNlNc9clWVrzZDw2tF`, `sha256 = d3829dd9591d39428062f8c50ea99b276872c4126452b7c853fe83d088cf2216`).
   - Primary `CTXNORM_CLIP` median lift `-0.008461025318455606`, `CI95 = [-0.019228082690923506, +0.0003517660130095988]`, median Elo `1488.5444469574622`, `CI95 = [1467.9687997051508, 1504.185164900202]`; `CONTROL` median lift `-0.008461025318455606`, `CI95 = [-0.019228082690923506, +0.0003517660130095988]`. **Bit-for-bit identical (`True`)** to committed `H297/metrics.json`.
2. **`H298` (`h298_family_balanced_gradaccum`)**:
   - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H298/h298_checkpoints_and_results.tar.gz` (`VersionId = lG1UYi3tUz3vNfkFllycDAWzTvOLIbOE`, `sha256 = 95494ce2efc99eead40e5f40d503aa750f8783e1b6f5cb472bfcdb8d5f72865a`).
   - Primary `FAMBAL_ACC4` median lift `-0.033528458894059865`, `CI95 = [-0.09813891237891575, +0.0035028817498420945]`; `PAIRED_CONTROL` median lift `-0.03683030173874502`, `CI95 = [-0.10260735527823622, +0.0029918138749592755]`. **Bit-for-bit identical (`True`)** to committed `H298/metrics.json`.

---

## 3. Complete Paired-Control / Baseline-Loss Trajectory Table (`19` Trajectories Across `7` Wave/Horizon Groups)

*(Note: `H296` remained `PREREGISTERED_NOT_EXECUTED` on `origin/v1/next` at execution time and is recorded as unclosed.)*

| Wave / Group | Seed / Label | `n_updates` | LR | `dev91` Mean | Lift vs `r3175` | 19-Family Bootstrap `95%` CI | `W / T / L` | Changed Contexts | MM BT Elo (`r3175 = 1500`) | Elo `95%` CI |
| --- | --- | ---: | ---: | ---: | ---: | --- | :---: | ---: | ---: | --- |
| **`r3175` Baseline** | `round-3175.pt` | `0` | — | `1.138957` | `0.000000` | `[0.000000, 0.000000]` | `0 / 91 / 0` | `0` | `1500.00` | `[1500.00, 1500.00]` |
| **`t1820`** (median of 639 rounds) | `r3182` (`3176..3814`) | `2` | v2 sched | `1.130592` | `-0.008365` | `[-0.019080, +0.000443]` | `2 / 85 / 4` | `6` | `1492.36` | `[1469.38, 1515.98]` |
| *`t1820` (mean of 639 rounds)* | *`3176..3814` mean* | *`2`* | *v2 sched* | *`1.132162`* | *`-0.006795`* | *`[-0.015148, -0.000019]`* | *`4 / 80 / 7`* | *`4` (med)* | *`1484.72`* | *`[1459.90, 1510.97]`* |
| **`t1845`** | `control_v2-184501` | `2` | `3.69e-5` | `1.135439` | `-0.003518` | `[-0.009080, 0.000000]` | `0 / 89 / 2` | `2` | `1492.36` | `[1480.90, 1500.00]` |
| **`t1845`** | `control_v2-184502` | `2` | `2.64e-5` | `1.135439` | `-0.003518` | `[-0.009080, 0.000000]` | `0 / 89 / 2` | `2` | `1492.36` | `[1480.90, 1500.00]` |
| **`t1845`** | `control_v2-184503` | `2` | `4.22e-5` | `1.135439` | `-0.003518` | `[-0.009080, 0.000000]` | `0 / 89 / 2` | `2` | `1492.36` | `[1480.90, 1500.00]` |
| **`H297`** | `CONTROL-29701` | `8` | `3.69e-5` | `1.130592` | `-0.008365` | `[-0.019080, +0.000443]` | `2 / 85 / 4` | `6` | `1492.36` | `[1469.38, 1515.98]` |
| **`H297`** | `CONTROL-29702` | `8` | `3.69e-5` | `1.130496` | `-0.008461` | `[-0.019143, +0.000352]` | `1 / 86 / 4` | `5` | `1488.54` | `[1467.97, 1504.40]` |
| **`H297`** | `CONTROL-29703` | `8` | `3.69e-5` | `1.130496` | `-0.008461` | `[-0.019143, +0.000352]` | `1 / 86 / 4` | `5` | `1488.54` | `[1467.97, 1504.40]` |
| **`H298`** | `PAIRED_CONTROL-29801` | `8` | `2.19e-4` | `1.102127` | `-0.036830` | `[-0.103384, +0.002834]` | `7 / 73 / 11` | `18` | `1484.72` | `[1457.81, 1511.46]` |
| **`H298`** | `PAIRED_CONTROL-29802` | `8` | `6.53e-5` | `1.127301` | `-0.011656` | `[-0.024901, -0.000559]` | `3 / 80 / 8` | `11` | `1480.90` | `[1459.60, 1500.00]` |
| **`H298`** | `PAIRED_CONTROL-29803` | `8` | `2.87e-4` | `1.076404` | `-0.062553` | `[-0.133477, -0.010709]` | `7 / 64 / 20` | `27` | `1450.04` | `[1406.17, 1488.54]` |
| **`t2115_step16`** | `lr2.5e-05-step16` | `16` | `2.50e-5` | `1.130101` | `-0.008856` | `[-0.019459, -0.000378]` | `2 / 83 / 6` | `8` | `1484.72` | `[1470.68, 1496.34]` |
| **`t2115_step16`** | `lr0.0001-step16` | `16` | `1.00e-4` | `1.105495` | `-0.033462` | `[-0.097792, +0.003514]` | `5 / 77 / 9` | `14` | `1484.72` | `[1459.60, 1507.99]` |
| **`t2115_step16`** | `lr0.0003-step16` | `16` | `3.00e-4` | `1.040800` | `-0.098157` | `[-0.176799, -0.031734]` | `14 / 49 / 28` | `42` | `1446.13` | `[1389.54, 1496.34]` |
| **`t2115_step64`** | `lr2.5e-05-step64` | `64` | `2.50e-5` | `1.105495` | `-0.033462` | `[-0.097792, +0.003514]` | `5 / 77 / 9` | `14` | `1484.72` | `[1459.60, 1507.99]` |
| **`t2115_step64`** | `lr0.0001-step64` | `64` | `1.00e-4` | `1.059286` | `-0.079672` | `[-0.145025, -0.026960]` | `15 / 36 / 40` | `55` | `1402.06` | `[1322.27, 1469.38]` |
| **`t2115_step64`** | `lr0.0003-step64` | `64` | `3.00e-4` | `1.013149` | `-0.125808` | `[-0.249418, -0.030639]` | `25 / 16 / 50` | `75` | `1402.06` | `[1322.27, 1476.01]` |
| **`t2115_step256`** | `lr2.5e-05-step256` | `256` | `2.50e-5` | `1.065405` | `-0.073552` | `[-0.139724, -0.020473]` | `16 / 39 / 36` | `52` | `1422.39` | `[1352.84, 1484.72]` |
| **`t2115_step256`** | `lr0.0001-step256` | `256` | `1.00e-4` | `0.999819` | `-0.139138` | `[-0.237817, -0.051802]` | `24 / 17 / 50` | `74` | `1397.91` | `[1321.70, 1465.54]` |
| **`t2115_step256`** | `lr0.0003-step256` | `256` | `3.00e-4` | `1.011711` | `-0.127246` | `[-0.235314, -0.039401]` | `25 / 22 / 44` | `69` | `1430.37` | `[1378.24, 1478.03]` |

**19 out of 19 (`100%`)** paired-control / baseline-loss trajectories from `r3175` have negative `dev91` lift (`< 0.0`) and MM Bradley-Terry Elo `< 1500.0`.

---

## 4. Winner's-Curse Analysis on Flywheel `dev91` History

From the read-only copy of `H289` `history.npz` (`VersionId = dj7fcZrNhjkNwY2s5DkfOah1nfAUObnL`, `1,499` post-`r3175` 2-update rounds `3176..4675`) and `adapter_admissions.jsonl` (`4,038` rounds under fixed evaluation identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`):

1. **Post-`r3175` 2-Update Candidate Distribution (`N = 1,499` rounds `3176..4675`)**:
   - Mean $\mu_{\text{round}} = 1.1320995299$, SD $\sigma_{\text{round}} = 0.0026069589$, median $= 1.1305918667$, IQR $= 0.0053098016$.
   - `90th` percentile $q_{0.90} = 1.1354387884$, **`95th` percentile $q_{0.95} = 1.1354387884$**, `99th` percentile $q_{0.99} = 1.1366829009$, maximum rejected candidate $= 1.1385566141$.
   - **Expected Gaussian Order-Statistic Maximum** over `1,499` draws ($\mu + \sigma \Phi^{-1}(1499/1500)$): **`1.1404644969`**, which **exceeds** the observed `r3175` best (`1.1389572404`) by `+0.001507`!
   - Observed `r3175` sits at $z = +2.6305 \sigma_{\text{round}}$ above the post-`r3175` 2-update mean — completely typical for the maximum of `~1,500–4,000` noisy proposals ($E[\max Z_{1499}] \approx 3.21 \sigma$).
   - **Gap Above $q_{0.95}$ vs Family-Bootstrap SE**:
     - $u_{r3175} - q_{0.95} = 1.1389572404 - 1.1354387884 = \mathbf{+0.0035184519}$ (which equals the exact 2-context discrete step drop seen in `t1845`!).
     - Marginal 19-family bootstrap SE of `r3175`: $\text{SE}_{\text{fam}}(r3175) = 0.2386372862 \implies (u_{r3175} - q_{0.95}) / \text{SE}_{\text{fam}}(r3175) = \mathbf{0.0147 < 2.0}$.
     - Paired 19-family bootstrap SE of $(u_{r3175} - u_{q95})$: $\text{SE}_{\text{fam}}(\Delta_{q95}) = 0.0024511857 \implies (u_{r3175} - q_{0.95}) / \text{SE}_{\text{fam}}(\Delta_{q95}) = \mathbf{1.4354 < 2.0}$.
2. **All Identity-Matched Flywheel Admissions (`N = 4,038` rounds)**:
   - Mean $= 1.1331339087$, SD $= 0.0028000638$, $q_{0.95} = 1.1364930980$, $q_{0.99} = 1.1375519109$.
   - Expected Gaussian maximum over `4,038` draws: **`1.1428875238`** ($> 1.1389572404$), and $u_{r3175} - q_{0.95} = +0.0024641424$ (`0.0103 x` marginal family SE).

### Mechanistic Synthesis: Both Selection Noise at `n = 2` AND Linear Optimizer Drift for `n >= 8`

The meta-analysis resolves the apparent paradox between the two hypotheses:
1. **Why every 2-update step drops immediately by `-0.003518` to `-0.008365` (`SELECTION_NOISE_PLATEAU`)**:
   - `r3175` (`1.138957`) owes `+0.003518` (`r3175 - q95`) to a 2-context lucky discrete argmax flip on `dev91` (`0.0147x` marginal family SE, `1.435x` paired family SE). Any generic 2-step gradient perturbation (`t1820`, `t1845`) immediately loses those 2 fragile contexts and regresses to the `95th`-percentile / modal plateau (`1.135439` or `1.130592`).
2. **Why taking `8..256` updates degrades further (`-0.017054` per doubling of `n_updates`, `CI95 = [-0.027598, -0.006871]`)**:
   - Once off the 2-context argmax spike, unconstrained multi-step SGD on `TRAIN` exhibits a statistically significant negative drift slope (`-0.000452` utility/update linear, `-0.017054` utility/doubling of steps), expanding changed contexts from `2` (`n=2`) $\to$ `5–18` (`n=8`) $\to$ `14–42` (`n=16`) $\to$ `55` (`n=64`) $\to$ `69` (`n=256`).
   - Therefore, **no single-checkpoint multi-step SGD recipe from `r3175` can beat `1.138957 + 0.001 = 1.139957` unless it simultaneously (i) suppresses per-step argmax churn on the `85+` already-solved `dev91` contexts and (ii) evaluates/selects across an ensemble or trust-region constraint rather than a single unconstrained `TRAIN` trajectory.**

---

## 5. Reproduction Commands

```bash
nice -n 19 taskset -c 0-1 python3 research/flywheel_selfplay/h301_paired_control_drift_meta/run_h301.py
nice -n 19 taskset -c 0-1 pytest -q research/flywheel_selfplay/h301_paired_control_drift_meta/test_h301.py
```
