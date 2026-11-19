# H470-DVP Results: DOPE Value Probe (`WP0`–`WP4`) & Zero-Retraining Accuracy Harvester (`WP3` & `B03`)

- **Wave ID**: `H470-DVP`
- **Overall Verdict**: `INSTRUMENT_VERIFIED_PLATEAU_DIAGNOSED`
- **Frozen DOPE Composite SHA-256 (`before == after`)**: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
- **Cohort**: `1219` `DL-Train` contexts (`205` families) / `91` `DL-Val91` contexts (`19` families), `0` family overlap, `0` outer/query rows (`nq == 0`)
- **DVP Config SHA-256**: `b3e5a0d56726e5e1a3d88ecddfd08c852f0b7b2f5cbecd3e0cdbb0260e5387bf`

---

## 1. Gate & Criterion Scorecard

| Criterion | Status | Empirical Evidence |
| --- | --- | --- |
| `WP01_cache_auth` | **PASS** | 7-field SHA-256 manifests verified; uncovered 0.0939 median token drift in unauthenticated h452..h469 tokens_cache.pt vs H58/H256. |
| `DVP-S1_placebo_clean` | **PASS** | C4a R_shuffled-R_none CI=[-0.02416, +0.01470]; C4b R_noise-R_none CI=[-0.04160, +0.00306]. |
| `DVP-S2_dope_freeze_and_h58_parity` | **PASS** | dope_before == dope_after == c51a56d031a4af40..., H58 parity max diff = 0.0. |
| `DVP-S4_power_mde` | **PASS** | 80%-power MDE (205 families, alpha=0.05) = 0.06019 utility. |
| `G0_O-S2_outcome_ceiling` | **FAIL** | profile_oracle-R_none CI=[-0.00914, +0.02118]; R_oracle64-R_dope CI=[-0.00010, +0.02035]. |
| `U-S1_U-S2_U-S3_ultra_view` | **NO_LIFT** | T4 fine-landmark R² from R_dope = 0.4611; C6a R_dope_ultra-R_dope T1 CI=[-0.00441, +0.02631]; fine-5 median latency = 389.6ms (1T). |
| `P-S1_P-S2_P-S3_packet_v1` | **NO_LIFT** | C7b (R_dope+R_packet_v1)-R_dope T1 CI=[-0.01215, +0.01406], T2 neg-logloss CI=[-0.04296, +0.07400]; P-S2 target-bundle delta = -0.00766. |
| `V-S1_V-S2_V-S3_wp3_predicted_reveals` | **NO_LIFT** | V-S1 lam=0 diff=0.0; WP3_fixed_lowvar-incumbent_0rev Δ=+0.15883 CI=[+0.04333, +0.31439] (40.1% of 4-rev gain, harm=42.9%). |
| `B03_stage1b_svd_and_residual_adapter` | **NO_LIFT** | B03_svd96-pairnf_z40 Δ=-0.000012 CI=[-0.000188, +0.000156]; B03_ultra_packet-pairnf_z40 Δ=+0.000000 CI=[-0.000001, +0.000001]. |

---

## 2. Part A — Instrument & Bottleneck Diagnostics (`DL-Train` 1,219 Contexts)

### A1. Stage-1A Conditioning Bottleneck & `WP01` Cache Drift
- `FusedLandmarkWorkspace` (`hpc_high`) vs H58/H256 cache max abs diff: **`0.0e+00`** (`DVP-S2` exact bit-parity).
- Unauthenticated `tokens_cache.pt` (`h452..h469`) vs `hpc_high` token drift: median **`0.0939`**, p95 **`0.4225`**, max **`2.5333`** — directly validating `WP01`.
- `Stage1ADistilledBridge` (`landmark_bridge.blend_alpha = 0.0000`) `vec256` cosine vs `hpc_high`: **`0.4669`** (held-out family Ridge $R^2 =$ **`0.4179`**; vs native 23-landmark `high`: cosine **`0.4811`**, Ridge $R^2 =$ **`0.3201`**).

| Stage-1A Mode (`round-3175`) | 0-Rev Family Util | 0-Rev Gate Fails | 4-Rev Family Util | 4-Rev Gate Fails | 4-Rev Top-1 Agree vs `hpc_high` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `campaign_tokens` | `0.986389` | `72` | `1.382530` | `19` | `0.9992` |
| `hpc_high` | `0.983967` | `73` | `1.382600` | `19` | `1.0000` |
| `native_high` | `0.983736` | `80` | `1.390310` | `17` | `0.9877` |
| `sketch_landmark_bridge` | `0.963600` | `85` | `1.386958` | `18` | `0.9869` |
| `sketch_nolm_bridge` | `0.999647` | `76` | `1.380960` | `20` | `0.9828` |

### A2. `L7` Audit (`lm48_z` 48D SVD Projection vs `R_lm80` 80D Triple Reshape)
- Verdict: **`SVD_PROJECTION_CONFIRMED_DISTORTED_BY_16X3_TRIPLE_RESHAPE`** (`||B^T B - I_48||_max = 5.00e-08`).
- Reshaping `lm48` into `(16, 3)` `(pm, ps, px)` triples encounters negative `ps` before `abs()` on **`51.7%`** of slots and `px < pm` on **`50.8%`** of triples.
- Family `1-NN` / `5-NN` retrieval: `lm48_z` **`0.9918`** / **`0.9951`** vs `R_lm80` **`0.9918`** / **`0.9967`** (synthesized `extra32` alone: **`0.9893`**).

### A3. True Masked 0-Reveal vs Wave-0 Fake `fitness=100, gate=0` Pseudo-Reveal
- Wave-0 passes uncapped `100.0 / 0.4208 = +237.64` into `hist_token_proj` (`h_in`) and `w1_residual_adj`.
- Family-mean utility: `true_masked_0rev` **`0.986389`** (`72` gate fails) vs `wave0_fake_k1` **`0.726729`** (`104` gate fails) and `wave0_fake_k4` **`0.767161`** (`116` gate fails); `real_4rev` = **`1.382530`**.

### A4. `H410` Sample-Size Stability, `U-S3` Latency, & `DVP-S4` MDE
- `DVP-S4` 80%-power MDE ($\alpha=0.05$, 205 families): paired 0-reveal selector MDE = **`0.060186`** utility; paired 4-reveal vs 0-reveal MDE = **`0.220124`**.
- `U-S3` 5 fine FIT-CV landmarks (`gbdt`, `knn1`, `knn16`, `net`, `gbdt_quarter`) latency: 1-thread median **`389.6 ms`** (p95 **`527.7 ms`**; ~**`92.8 ms`** at 8 threads).
- Row-disjoint split-half cosine reliability (`all`): `vec256_sketch` **`0.9337`**, `vec256_hpc` **`0.9966`**, `dope629_hpc` **`0.9987`**, `dope629_ultra` **`0.9970`**, `R_packet_v1` **`0.9973`**.

---

## 3. Part B — Full CPU Value Probe Matrix (`T1`–`T4`, `h452a/` + `h452b/` 5 Folds × 3 Seeds)

### 3.1 Representation Arms (`DL-Train` OOF)

| Arm | Width | T1 0-Rev Fam Util | T1 0-Rev NDCG@1 | T1 0-Rev GateFail | T1 4-Rev Fam Util | T2 Head LogLoss | T2 Head F1 | T3 Med $R^2$ | T4 Fine-LM Mean $R^2$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R_none` | `0` | `1.733680` | `0.8820` | `0.0088` | `1.734592` | `1.3019` | `0.1237` | `-0.0106` | `—` |
| `R_shape` | `3` | `1.726478` | `0.8834` | `0.0075` | `1.741785` | `1.3135` | `0.1554` | `0.0233` | `-0.0068` |
| `lm48_z` | `48` | `1.721985` | `0.8856` | `0.0071` | `1.737103` | `1.3336` | `0.3293` | `0.1565` | `0.4697` |
| `R_lm80` | `80` | `1.717081` | `0.8855` | `0.0079` | `1.718519` | `1.3545` | `0.3046` | `0.1217` | `0.4073` |
| `R_desc` | `290` | `1.723847` | `0.8856` | `0.0077` | `1.724648` | `1.2646` | `0.3324` | `0.0700` | `0.5027` |
| `R_learned` | `373` | `1.726396` | `0.8856` | `0.0068` | `1.725600` | `1.1911` | `0.3000` | `0.0801` | `0.2882` |
| `R_dope` | `663` | `1.713631` | `0.8841` | `0.0082` | `1.715893` | `1.2018` | `0.3058` | `0.0725` | `0.4611` |
| `R_hidden` | `256` | `1.720992` | `0.8841` | `0.0086` | `1.724262` | `1.1728` | `0.3233` | `0.1859` | `0.4088` |
| `R_tok` | `320` | `1.723430` | `0.8857` | `0.0074` | `1.714938` | `1.3229` | `0.3148` | `0.0080` | `0.3181` |
| `R_shuffled` | `663` | `1.730326` | `0.8862` | `0.0079` | `1.738073` | `1.4280` | `0.1568` | `-0.2105` | `—` |
| `R_noise` | `663` | `1.717733` | `0.8844` | `0.0092` | `1.720969` | `1.3695` | `0.1649` | `-0.0119` | `—` |
| `R_desc_native_high` | `290` | `1.719607` | `0.8849` | `0.0072` | `1.719383` | `1.1135` | `0.4106` | `0.3729` | `0.7964` |
| `R_dope_native_high` | `663` | `1.722779` | `0.8831` | `0.0072` | `1.719715` | `1.0785` | `0.3948` | `0.3965` | `0.8033` |
| `R_desc_ultra` | `290` | `1.724511` | `0.8841` | `0.0081` | `1.719220` | `1.0588` | `0.4230` | `0.4447` | `—` |
| `R_dope_ultra` | `663` | `1.723229` | `0.8841` | `0.0089` | `1.712532` | `1.0380` | `0.4028` | `0.4595` | `—` |
| `R_dope_plus_ultra_delta` | `1326` | `1.726189` | `0.8861` | `0.0072` | `1.720797` | `1.0079` | `0.3959` | `0.4190` | `—` |
| `R_packet_v1` | `196` | `1.722508` | `0.8845` | `0.0078` | `1.732085` | `1.2530` | `0.3343` | `0.2605` | `0.5585` |
| `R_packet_notarget` | `196` | `1.730170` | `0.8852` | `0.0055` | `1.726783` | `1.4346` | `0.2848` | `-0.0078` | `0.5532` |
| `R_dope_plus_packet_v1` | `859` | `1.715163` | `0.8848` | `0.0079` | `1.727306` | `1.1940` | `0.3489` | `0.2833` | `0.7020` |
| `R_dope_ultra_plus_packet_v1` | `859` | `1.716437` | `0.8829` | `0.0096` | `1.721962` | `1.0300` | `0.3838` | `0.4741` | `—` |
| `R_randrouter` | `663` | `1.723537` | `0.8845` | `0.0076` | `1.718334` | `1.2508` | `0.3248` | `0.1190` | `—` |

### 3.2 Pre-Registered Contrasts (`C0`–`C7`, 10,000-Draw Paired Family Bootstrap + Holm)

| Contrast | Δ T1 0-Rev Fam Util | 95% Family Boot CI | Raw $p$ | Holm $p$ | Fam W/T/L |
| --- | ---: | --- | ---: | ---: | --- |
| `C0a_profile_oracle_minus_R_none_halfB` | `+0.006476` | `[ -0.009142, +0.021183 ]` | `0.4002` | `—` | `122/3/80` |
| `C0b_profile_oracle_minus_R_dope_halfB` | `-0.002075` | `[ -0.028495, +0.028902 ]` | `0.8199` | `1.0000` | `109/3/93` |
| `C0c_R_oracle64_minus_R_none_nb64` | `-0.009634` | `[ -0.025837, +0.002005 ]` | `0.1288` | `—` | `104/0/101` |
| `C0d_R_oracle64_minus_R_dope_nb64` | `+0.009955` | `[ -0.000098, +0.020352 ]` | `0.0520` | `0.5199` | `123/0/82` |
| `C0e_R_oracle16_minus_R_dope_nb64` | `+0.017504` | `[ +0.007033, +0.031158 ]` | `0.0006` | `—` | `118/0/87` |
| `C1_R_dope_minus_R_desc` | `-0.010216` | `[ -0.027562, +0.003848 ]` | `0.1860` | `1.0000` | `106/0/99` |
| `C1b_R_learned_minus_R_desc` | `+0.002549` | `[ -0.009473, +0.013900 ]` | `0.6613` | `—` | `106/0/99` |
| `C2_R_dope_minus_R_randrouter` | `-0.009906` | `[ -0.026843, +0.001959 ]` | `0.1314` | `1.0000` | `100/1/104` |
| `C3_R_desc_minus_R_shape` | `-0.002631` | `[ -0.019730, +0.014224 ]` | `0.7769` | `1.0000` | `103/1/101` |
| `C3b_R_dope_minus_R_none` | `-0.020049` | `[ -0.058037, +0.006562 ]` | `0.1940` | `—` | `101/0/104` |
| `C4a_R_shuffled_minus_R_none` | `-0.003354` | `[ -0.024160, +0.014701 ]` | `0.7739` | `—` | `105/0/100` |
| `C4b_R_noise_minus_R_none` | `-0.015947` | `[ -0.041602, +0.003064 ]` | `0.1272` | `—` | `100/0/105` |
| `C5_R_dope_minus_R_tok` | `-0.009799` | `[ -0.022798, +0.002236 ]` | `0.1146` | `1.0000` | `95/1/109` |
| `C5b_R_lm80_minus_lm48_z` | `-0.004904` | `[ -0.018001, +0.006198 ]` | `0.4422` | `—` | `108/1/96` |
| `C6a_R_dope_ultra_minus_R_dope` | `+0.009597` | `[ -0.004407, +0.026312 ]` | `0.1972` | `1.0000` | `104/0/101` |
| `C6b_R_dope_ultra_minus_R_dope_native_high` | `+0.000449` | `[ -0.011613, +0.014947 ]` | `0.9761` | `1.0000` | `95/0/110` |
| `C6c_R_desc_ultra_minus_R_desc_native_high` | `+0.004904` | `[ -0.007122, +0.018450 ]` | `0.4394` | `—` | `105/0/100` |
| `C6d_R_dope_plus_ultra_delta_minus_R_dope` | `+0.012558` | `[ +0.002950, +0.023506 ]` | `0.0090` | `—` | `110/3/92` |
| `C7a_R_packet_v1_minus_R_dope` | `+0.008877` | `[ -0.002396, +0.021884 ]` | `0.1246` | `—` | `110/0/95` |
| `C7b_R_dope_plus_packet_v1_minus_R_dope` | `+0.001531` | `[ -0.012153, +0.014064 ]` | `0.7987` | `1.0000` | `116/1/88` |
| `C7c_R_packet_v1_minus_R_packet_notarget` | `-0.007662` | `[ -0.024410, +0.008496 ]` | `0.3476` | `1.0000` | `100/1/104` |
| `C7d_R_dope_ultra_plus_packet_v1_minus_R_dope` | `+0.002806` | `[ -0.009352, +0.015932 ]` | `0.6537` | `—` | `99/1/105` |

---

## 4. Part C — Zero-Retraining Accuracy Upgrades (`WP3` & `B03`, Contrast `C8`)

### 4.1 `WP3` Variance-Shrunk Predicted Reveals (0 Real Candidate Evaluations)
- `V-S1` (`lam = 0` identity vs `incumbent_0rev`): `max_abs_diff = 0.0e+00` (**PASS**).
- Mean top-4 Empirical-Bayes shrinkage weight $\bar{\lambda}_4 = 0.9500$; real 4-reveal minus 0-reveal headroom = `+0.396141`.

| Arm (0 Real Evals unless noted) | Fam Mean Utility | Ctx Mean Utility | GateFail Rate | Δ vs `incumbent_0rev` (95% CI) | % 4-Rev Gain | Fam Harm Rate |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `incumbent_0rev` | `0.986389` | `1.038181` | `0.0591` | `baseline` | `—` | `—` |
| `wave0_fake_k4` | `0.767161` | `0.805814` | `—` | `baseline` | `—` | `—` |
| `incumbent_4rev_upper` | `1.382530` | `1.420532` | `0.0156` | `baseline` | `—` | `—` |
| `WP3_hist_only_lowvar` | `1.132317` | `1.154663` | `0.0327` | `+0.145928` (`[+0.078394, +0.218917]`) | `36.8%` | `34.6%` |
| `WP3_fixed_lowvar` | `1.145216` | `1.191259` | `0.0495` | `+0.158827` (`[+0.043330, +0.314390]`) | `40.1%` | `42.9%` |
| `WP3_mimic_acq` | `1.122781` | `1.164162` | `0.0546` | `+0.136392` (`[+0.018562, +0.296796]`) | `34.4%` | `45.9%` |
| `WP3_ultra_packet` | `1.125228` | `1.166192` | `0.0494` | `+0.138839` (`[+0.020172, +0.297311]`) | `35.0%` | `43.4%` |
| `WP3_shuffled_ctrl` | `1.122149` | `1.176696` | `0.0507` | `+0.135760` (`[+0.010998, +0.297899]`) | `34.3%` | `44.9%` |

### 4.2 `B03` Folded Stage-1B SVD `629 -> 96` & Zero-Init `Ultra+Packet` Adapter (4-Reveal Flywheel CV)
- Step-0 exact bit-identity (`B03_svd96` & `B03_ultra_packet` vs `round-3175`): `max_abs_diff = 0.0e+00` / `0.0e+00` (fast cached-anchor parity `max_abs_diff = 3.05e-05`).

| Seed | Arm | Pooled OOF Gain | Gain 95% CI | Pooled Elo | Non-Floor Elo | Non-Floor Elo 95% CI | W / T / L | Strict |
| ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `1` | `pairnf_z40` | `+0.004781` | `[+0.000555, +0.009951]` | `+4.70` | `+4.34` | `[-1.62, +10.13]` | `122/2227/89` | `False` |
| `1` | `B03_svd96` | `+0.004842` | `[+0.000471, +0.009973]` | `+5.13` | `+4.78` | `[-1.46, +10.76]` | `127/2220/91` | `False` |
| `1` | `B03_ctrl96` | `+0.005107` | `[+0.000795, +0.010393]` | `+4.85` | `+4.49` | `[-1.39, +10.38]` | `123/2226/89` | `False` |
| `1` | `B03_ultra_packet` | `+0.004784` | `[+0.000556, +0.009953]` | `+4.85` | `+4.49` | `[-1.50, +10.32]` | `123/2226/89` | `False` |
| `2` | `pairnf_z40` | `+0.006861` | `[+0.001502, +0.013448]` | `+6.56` | `+5.94` | `[+0.15, +11.59]` | `128/2228/82` | `True` |
| `2` | `B03_svd96` | `+0.006697` | `[+0.001331, +0.013315]` | `+5.70` | `+5.07` | `[-1.24, +11.26]` | `129/2220/89` | `False` |
| `2` | `B03_ctrl96` | `+0.006861` | `[+0.001502, +0.013448]` | `+6.56` | `+5.94` | `[+0.15, +11.59]` | `128/2228/82` | `True` |
| `2` | `B03_ultra_packet` | `+0.006858` | `[+0.001499, +0.013438]` | `+6.41` | `+5.79` | `[+0.00, +11.46]` | `128/2227/83` | `False` |
| `3` | `pairnf_z40` | `+0.005586` | `[+0.001166, +0.011034]` | `+3.85` | `+3.33` | `[-2.39, +8.94]` | `115/2235/88` | `False` |
| `3` | `B03_svd96` | `+0.005786` | `[+0.001412, +0.011229]` | `+4.13` | `+3.62` | `[-2.10, +9.20]` | `120/2227/91` | `False` |
| `3` | `B03_ctrl96` | `+0.005586` | `[+0.001166, +0.011034]` | `+3.85` | `+3.33` | `[-2.39, +8.94]` | `115/2235/88` | `False` |
| `3` | `B03_ultra_packet` | `+0.005586` | `[+0.001166, +0.011034]` | `+3.85` | `+3.33` | `[-2.39, +8.94]` | `115/2235/88` | `False` |

---

## 5. One-Time Frozen `DL-Val91` `REGCHECK` (`91` Contexts / `19` Families)

- Incumbent 4-reveal `val91` parity: **`1.138957240375`** (`parity_ok = True`).
- `WP3` selected arm (`WP3_fixed_lowvar`, 0 real evals): `val91` family utility **`0.656460`** vs `incumbent_0rev` **`0.769448`** (`Δ = -0.112987`, 95% CI `[-0.418344, +0.139593]`, `not_worse = False`).

| Seed | CV-Selected B03 Arm | `val91` Ctx Util | `val91` Ctx Gain | `val91` Fam Gain (95% CI) | `val91` Elo | W / T / L | New GateFails |
| ---: | --- | ---: | ---: | --- | ---: | --- | ---: |
| `1` | `B03_svd96` | `1.138957` | `-0.000000` | `-0.000000` (`[-0.000867, +0.000739]`) | `+3.82` | `3/86/2` | `0` |
| `2` | `pairnf_z40` | `1.140027` | `+0.001070` | `+0.001025` (`[-0.000407, +0.002750]`) | `+11.46` | `4/86/1` | `0` |
| `3` | `B03_svd96` | `1.139919` | `+0.000961` | `+0.000921` (`[-0.000496, +0.002585]`) | `+7.64` | `4/85/2` | `0` |
