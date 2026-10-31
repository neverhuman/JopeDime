# H108 — W1 GPU Campaign Readout: Prior Replacement and DOPE Neural Residual (`run_w1_gpu_campaign.py`)

- **Hypothesis ID**: `H108`
- **Wave**: `W1` (`drv_t0520.md` / `rd_argon_h108_t0630.md` / `PROPOSALS.md` §W1)
- **Owner**: `rd_argon` (`[ARGON]`)
- **Verdict**: **`NO_NEURAL_RESIDUAL_ADVANCE_OVER_GBM_CONTROLS`** (`H70` official gate decision: **`NO_ADVANCE`**, `search_claim_advanced=False`, `legacy_claim_advanced=False`)
- **GPU Budget & Lock Custody**: **`865.99` GPU-seconds** (`0.2406` GPU-hours <= `1.5` GPU-h = `5,400.0`s ceiling) on `xbabe2` `NVIDIA GeForce RTX 3090` under `/home/ubuntu/jope_plan/qualification.lock` (`mode=inherited_fd`, `fd=9`, `inode=103:07:108299230`, lock released and `/home/ubuntu/jope_plan/qualification.queue/` verified clean).
- **Holdout & Frozen-DOPE Hygiene**: Strictly `1,219` `DL-Train` contexts across `205` families (`5` disjoint `41`-family outer folds, `seed=4900`); `0` `DL-Val`, outer-test, or sealed meta-test reads. Frozen DOPE backbone SHA `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` verified before and after every single cell (`45/45` cells, `23,546` trainable `h40b` adapter parameters per cell, positive `update_l2_from_init` across `[0.230441, 1.723185]`).

---

## 1. Hypothesis & Experimental Setup

**Hypothesis (`W1`)**: The legacy `RIDGE_PRIOR10` prior (`10` hand-engineered features) is substantially weaker than a 412-feature out-of-fold tree prior (`GBM_PROG412_ONLY`, Track 2 $U_4 = 1.649797$ vs $1.511488$). W1 tests whether:
1. Removing `RIDGE_PRIOR10` in favor of a zero prior (`NEURAL_ZERO_PRIOR`) eliminates prior-induced drag inside the `h40b` neural adapter.
2. Replacing `RIDGE_PRIOR10` with the fold-calibrated out-of-fold `GBM_PROG412` prior (`NEURAL_GBM_RESIDUAL`) allows the frozen-DOPE `h40b` multimodal neural residual adapter (`23,546` trainable parameters; `parent_delta` frozen at zero) to beat both `R882_CONTINUITY` and the strongest matched standalone tree controls (`GBM_PROG412_ONLY` and `GBM_PROG412_CTX`) on out-of-fold `DL-Train` evaluation.

### W2 Gate Confirmation (`FINDINGS.md` §7 Line 561 & `H62`)
- Per `drv_t0520.md`, W2 gate status in [`FINDINGS.md`](../rd_breakthrough/FINDINGS.md) §7 (`lines 545, 552, 561`) and [`h62_slate_consistency/SUMMARY.md`](../h62_slate_consistency/SUMMARY.md) (`lines 42–46`) was verified prior to launch:
  - The W2 kill rule (`PROPOSALS.md:130`) applies only to W2's own slate-consistency training follow-up.
  - ASTRA and OPUS recorded that live slate-12 changed choices on `24/800 = 3.0%` of `(context, draw)` rows (`15/800 = 1.875%` at slate 20, `2/800 = 0.250%` at slate 192) and issued the explicit downstream contract disposition: **retain current live slate-centering for all matched W1/W3 arms and `R882_CONTINUITY`, pin training slate size (`192`: `4` revealed anchors + `188` unrevealed candidates) and evaluation slate size (`full_pool` $P_c$), and disclose the known slate-size dependence**.

---

## 2. Per-Fold Fit-Only Sensitivity Pilot (`seed=700`, `64` Fit-Only Contexts/Fold)

Before training the 45 main cells, a fit-only sensitivity pilot (`NEURAL_GBM_RESIDUAL`, `seed=700`, `64` stratified fit-only contexts per outer fold) selected the earliest endpoint $E_k \in \{40, 160, 480\}$ where at least `10%` of pilot contexts changed their unrevealed top-4 slate relative to step 0:

| Outer Fold $k$ | Held Contexts (`41` Families) | Pilot Fit Contexts | Top-4 Changed @ 40 Steps | Top-4 Changed @ 160 Steps | Selected Endpoint $E_k$ | Pilot GPU Wall (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fold_0` | `227` | `64` | `7.81%` | `15.62%` | **`160`** | `14.18s` |
| `fold_1` | `234` | `64` | `7.81%` | `15.62%` | **`160`** | `12.06s` |
| `fold_2` | `269` | `64` | `18.75%` | `—` | **`40`** | `7.31s` |
| `fold_3` | `271` | `64` | `7.81%` | `18.75%` | **`160`** | `11.90s` |
| `fold_4` | `218` | `64` | `14.06%` | `—` | **`40`** | `7.73s` |

---

## 3. Full 10-Contender Outcome Panel: Elo + 95% CI & Utility (`1,219` `DL-Train` Contexts, `205` Families)

All `10` contenders were scored out-of-fold across all `1,219` `DL-Train` contexts (`205` families, `0` contexts excluded across global, within-generation, seen-template, and unseen-template strata). Family-grouped Bradley-Terry Elo (`compute_family_grouped_bradley_terry_elo`, `tie_margin=1e-4`, `prior=0.01`, `n_iters=400`, `1,000` family-bootstrap draws) is anchored at `R882_CONTINUITY = 1500.0`:

| Contender | Track 2 Verified `U_4` Elo (`95% CI`) | Track 2 `U_4` Family Mean | `U_4 - U_0` Family Mean | `U_4 - R_4` Family Mean | Track 1 Legacy Elo (`95% CI`) | Track 1 Legacy Family Mean | Track 1 `val_reg_inner` Context Mean | Track 1 Unrev-Only Argmin Family Mean | Within-Gen `W_4 - WR_4` | Seen-Template `U_4` | Unseen-Template `U_4` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `GBM_PROG412_CTX` | **1524.50** (`[1502.83, 1545.84]`) | `1.680321` | `+0.222088` | `+0.107224` | **1489.81** (`[1468.33, 1512.48]`) | `1.317384` | `1.360410` | `1.316602` | `+0.066055` | `1.639563` | `1.668578` |
| `R882_CONTINUITY` | **1500.00** (`[1500.00, 1500.00]`) | `1.609212` | `+0.150979` | `+0.036114` | **1500.00** (`[1500.00, 1500.00]`) | `1.385355` | `1.423073` | `1.355472` | `+0.037264` | `1.617654` | `1.607129` |
| `GBM_RESIDUAL_STEP0` | **1489.48** (`[1471.21, 1507.57]`) | `1.647172` | `+0.188939` | `+0.074074` | **1437.27** (`[1413.99, 1461.68]`) | `1.234085` | `1.295048` | `1.228541` | `+0.030456` | `1.636261` | `1.657658` |
| `NEURAL_ZERO_PRIOR` | **1487.52** (`[1472.35, 1503.43]`) | `1.559999` | `+0.101766` | `-0.013099` | **1568.88** (`[1549.87, 1587.40]`) | `1.426594` | `1.452810` | `1.339813` | `+0.012292` | `1.620038` | `1.543153` |
| `NEURAL_GBM_RESIDUAL` | **1487.17** (`[1469.05, 1505.46]`) | `1.646411` | `+0.188177` | `+0.073313` | **1436.93** (`[1413.88, 1461.48]`) | `1.236941` | `1.297418` | `1.231982` | `+0.031185` | `1.636121` | `1.656750` |
| `ZERO_PRIOR_STEP0` | **1481.60** (`[1465.87, 1497.31]`) | `1.552575` | `+0.094342` | `-0.020523` | **1566.89** (`[1547.25, 1585.97]`) | `1.423838` | `1.449459` | `1.345974` | `+0.011818` | `1.600245` | `1.542070` |
| `GBM_PROG412_ONLY` | **1456.82** (`[1436.16, 1476.22]`) | `1.649797` | `+0.191564` | `+0.076700` | **1408.13** (`[1380.32, 1435.27]`) | `1.183562` | `1.238190` | `1.179366` | `+0.057457` | `1.636833` | `1.656648` |
| `NEURAL_RIDGE_PRIOR10` | **1393.93** (`[1374.08, 1413.00]`) | `1.527448` | `+0.069215` | `-0.045650` | **1318.26** (`[1291.57, 1345.19]`) | `0.789555` | `0.820358` | `0.789539` | `+0.009581` | `1.535536` | `1.526081` |
| `RIDGE_PRIOR10_STEP0` | **1393.49** (`[1373.90, 1412.29]`) | `1.524924` | `+0.066691` | `-0.048174` | **1319.62** (`[1292.54, 1346.79]`) | `0.791603` | `0.821361` | `0.791603` | `+0.009628` | `1.533109` | `1.523631` |
| `RIDGE_PRIOR10_ONLY` | **1381.85** (`[1362.32, 1400.98]`) | `1.511488` | `+0.053255` | `-0.061610` | **1322.03** (`[1293.89, 1349.87]`) | `0.779950` | `0.794549` | `0.780117` | `+0.012863` | `1.539141` | `1.510541` |

### Explicit Disclosure: `NEURAL_ZERO_PRIOR` Is NOT a Win (Track-1 Legacy Revealed-Slot / Tie Artifact)

> [!IMPORTANT]
> **Do not report or interpret `NEURAL_ZERO_PRIOR` as a win.**
> On Track 1 Legacy (`val_reg_inner` whole-pool argmin utility), `NEURAL_ZERO_PRIOR` records **`+68.88` Elo** vs `R882_CONTINUITY` (`1568.88`, `95% CI [1549.87, 1587.40]`, `545W / 342T / 332L`), but **untrained `ZERO_PRIOR_STEP0` is already `+66.89` Elo** (`1566.89`, `95% CI [1547.90, 1585.13]`, `540W / 341T / 338L` with `341` context ties vs `R882_CONTINUITY`), and training moves it by only `+1.99` Elo.
> Meanwhile, on **Track 2 Verified `U_4`** (unrevealed top-4 search), `NEURAL_ZERO_PRIOR` is **`-12.48` Elo** vs `R882_CONTINUITY` (`1487.52`, `95% CI [1472.35, 1503.43]`, `373W / 393T / 453L`, family mean `1.559999` vs `R882_CONTINUITY` `1.609212` and random-4 `1.573098`), and on **Track 1 unrevealed-only argmin** (`track1_unrev_argmin_family_mean`), `NEURAL_ZERO_PRIOR` (`1.339813`) also trails `R882_CONTINUITY` (`1.355472`).
> Therefore, the Track 1 Legacy elevation of `ZERO_PRIOR_STEP0` and `NEURAL_ZERO_PRIOR` is strictly a **scoring/tie artifact of the legacy whole-pool argmin track** (where revealed anchor slots `0..3` are eligible to win the argmin when `base_prior_cost = 0` and `parent_delta = 0`), **not neural learning**.

---

## 4. Joint Primary Contrasts (`m = 9`) & Official `H70` Gate (`m = 4`)

All contrasts use `10,000` paired family-clustered bootstrap draws (`seed=20261031`, `205` families). Exact integer harm counts (`harm_gt_0p01_count`) are computed directly from the unrounded IEEE-754 `float64` paired-family deltas (`deltas_f64 < -0.01`):

### Joint 9-Contrast Primary Family (`m = 9`)

| Endpoint | Contrast | Family Mean $\Delta$ | Median $\Delta$ | One-Sided `LCB95` | Adjusted `LCB` ($m=9$) | Family W/T/L (`205`) | Exact Harm Count ($\Delta < -0.01$, `<=10%` Gate) | Common Gate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Track 2 `U_4` | `NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_CTX` | `-0.033911` | `-0.002077` | `-0.064049` | `-0.084168` | `80/7/118` | `72/205 (35.12%)` | `False` |
| Track 2 `U_4` | `NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_ONLY` | `-0.003387` | `+0.000146` | `-0.016748` | `-0.026449` | `111/35/59` | `26/205 (12.68%)` | `False` |
| Track 2 `U_4` | `NEURAL_GBM_RESIDUAL_vs_NEURAL_ZERO_PRIOR` | `+0.086412` | `+0.005322` | `+0.048709` | `+0.033982` | `138/0/67` | `32/205 (15.61%)` | `False` |
| Track 2 `U_4` | `NEURAL_ZERO_PRIOR_vs_NEURAL_RIDGE_PRIOR10` | `+0.032551` | `+0.003274` | `+0.010504` | `+0.002860` | `126/1/78` | `47/205 (22.93%)` | `False` |
| Track 1 Legacy | `NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_CTX` | `-0.080443` | `-0.015741` | `-0.142357` | `-0.176410` | `78/0/127` | `112/205 (54.63%)` | `False` |
| Track 1 Legacy | `NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_ONLY` | `+0.053379` | `+0.004612` | `+0.014413` | `-0.008850` | `114/30/61` | `39/205 (19.02%)` | `False` |
| Track 1 Legacy | `NEURAL_GBM_RESIDUAL_vs_NEURAL_ZERO_PRIOR` | `-0.189653` | `-0.074978` | `-0.264544` | `-0.307121` | `54/0/151` | `131/205 (63.90%)` | `False` |
| Track 1 Legacy | `NEURAL_GBM_RESIDUAL_vs_R882_CONTINUITY` | `-0.148415` | `-0.046834` | `-0.223808` | `-0.268314` | `83/0/122` | `117/205 (57.07%)` | `False` |
| Track 1 Legacy | `NEURAL_ZERO_PRIOR_vs_NEURAL_RIDGE_PRIOR10` | `+0.637039` | `+0.274207` | `+0.475661` | `+0.418067` | `195/0/10` | `8/205 (3.90%)` | `True` |

### Official `H70` 4-Contrast Gate (`m = 4`, `research/flywheel_selfplay/h70_w1_prelaunch_gate_audit/gate.py`)

| Contrast (`jope.w1-gate-evidence/v1`) | Adjusted Family Mean `LCB` ($m=4$) | Median Family $\Delta$ | Exact Harm Count ($\Delta < -0.01$) | Family Count | Common Gate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `u4_vs_program_gbm` | `-0.022643` | `+0.000146` | `26` (`12.68%`) | `205` | `False` |
| `u4_vs_program_context_gbm` | `-0.077376` | `-0.002077` | `72` (`35.12%`) | `205` | `False` |
| `legacy_vs_program_gbm` | `-0.001907` | `+0.004612` | `39` (`19.02%`) | `205` | `False` |
| `legacy_vs_program_context_gbm` | `-0.163040` | `-0.015741` | `112` (`54.63%`) | `205` | `False` |

### Verdict & Next Step

> **Errata (2026-10-31, hwave overseer).** The verdict bullet originally quoted stale pre-replay numbers (`+0.064984`, `155W/12T/38L`, `1.653159`, `-0.006748`, `32–35/205`). They are corrected below to match the tables above and `metrics.json`. The verdict (`NO_ADVANCE`) is unchanged.

- **Verdict**: **`NO_NEURAL_RESIDUAL_ADVANCE_OVER_GBM_CONTROLS`** (`H70` official decision: **`NO_ADVANCE`**). While replacing `RIDGE_PRIOR10` with zero prior (`+0.032551` $U_4$) and replacing zero prior with the calibrated `GBM_PROG412` prior (`+0.086412` $U_4$, `LCB_adj(m=9) = +0.033982 > +0.001`, `138W / 0T / 67L`) produce large gains inside the neural adapter architecture, `NEURAL_GBM_RESIDUAL` ($U_4 = 1.646411$, `1487.17` Elo) does **not** beat standalone `GBM_PROG412_ONLY` ($U_4 = 1.649797$, $\Delta = -0.003387$, `26/205 = 12.68%` harmed families) or `GBM_PROG412_CTX` ($U_4 = 1.680321$, $\Delta = -0.033911$, `72/205 = 35.12%` harmed families) on Track 2 Verified $U_4$.
- **Next Step**: Do **not** append any W1 neural contender to `HWM_CANDIDATES.jsonl` and do **not** promote to outer-fold or holdout evaluation. Retain the non-neural `GBM_PROG412_CTX` / `GBM_PROG412_ONLY` priors for downstream waves (`W4`, `W6`, `W7`, `W9`) while diagnosing why the slate-centered `smooth_l1 + 0.1 * KL` residual loss harms `26/205` (vs `GBM_PROG412_ONLY`) and `72/205` (vs `GBM_PROG412_CTX`) families relative to the calibrated tree prior.

---

## 5. Closure of `REDTEAM_astra.md` Findings (`#1020`, Commit `4a3e10be92befd27215399d4c0186eaa8f09326b`)

All items raised in [`REDTEAM_astra.md`](REDTEAM_astra.md) (`#1020`, commit `4a3e10be92befd27215399d4c0186eaa8f09326b`) are closed in the Phase 4 CPU replay and versioned S3 custody (`run_id = w1_gpu_20261031T123502Z_326209`):
1. **P1 (`H70` Schema & Exact Integer Harm Counts)**: Closed. `run_w1_gpu_campaign.py` constructs the exact `jope.w1-gate-evidence/v1` payload (`bootstrap` + `contrasts`) for `h70_w1_prelaunch_gate_audit/gate.py` and computes `harm_gt_0p01_count` directly from the original-precision `float64` paired-family deltas (`int(np.sum((lhs_f64 - rhs_f64) < -0.01))`), never from rounded fractions. Both the 9-contrast runner verdict (`NO_NEURAL_RESIDUAL_ADVANCE_OVER_GBM_CONTROLS`) and the 4-contrast `H70` verdict (`NO_ADVANCE`) are preserved and reported separately.
2. **P2 (Original-Precision Context/Seed/Arm Utilities for Independent Elo & Harm Replay)**: Closed. `w1_per_seed_family_vectors.json` (`12,663,161` bytes, SHA-256 `101bcb9d82e3cc53dcf5a43ccbd4c8f813660a455a212c12d60b8f6605c75274`, VersionId `pOeS6mq9APnevXPLa8eBkS1I2W2s7CbI`) and `w1_wave_summary.json` (`339,716` bytes, SHA-256 `2c37f7f25587d2d2cd78b95b47b70bceabd8e246812753c929d0463723de8951`, VersionId `i3d6.gwClv0BbVMCjUb4MRtK4QBDX3gN`) in versioned S3 training-only custody publish full unrounded IEEE-754 `float64` per-context utilities (`track2_u4_ctx_mean_by_cid`, `track1_legacy_ctx_mean_by_cid`, `track1_unrev_ctx_mean_by_cid`), per-seed per-context utilities (`track2_u4_by_seed_cid`, `track1_legacy_by_seed_cid`), per-family utilities (`track2_u4_family_mean_vector`, `track1_legacy_family_mean_vector`), unrounded `paired_family_deltas_float64`, `_metadata` (`train_cids`, `family_by_cid`, `fold_by_cid`, `u0_by_cid`, `r4_ctx_mean_by_cid`), and explicit stratum exclusion lists (`0` excluded across all strata).
3. **P2 (S3 Readback Verification by Exact `VersionId`)**: Closed. `s3_create_only_upload_and_verify` enforces a non-empty `VersionId` from `aws s3api put-object --if-none-match "*"` and verifies every object via `aws s3api get-object --version-id VERSION`, checking response `VersionId`, byte length, and SHA-256 (`version_id_readback_verified: true` on all 10 uploaded objects across `w1/` and `W1/`).
4. **Provenance & Budget Ledger Disclosure**: Phases 1–3 executed on `xbabe2` `cuda:0` under `/home/ubuntu/jope_plan/qualification.lock` (`fd=9`, `inode=103:02:118242855`) with runner SHA `e4531b98f29106eb96cf4e636ae6ba1f553d11020e3dabde6a7adae84bee48c8`, writing the `205 MiB` Phase 3 checkpoint (`SHA-256 = c7377a1318f164aec8b847ed0f27d61ba7611dd380f3e831aed79b343ac00135`, kept outside Git in `/home/ubuntu/jope-data/w1/runs/w1_gpu_campaign_h108/w1_gpu_phase3_checkpoint.pkl`); Phase 4 readout and exact-version S3 upload ran on CPU with runner SHA `a7a2996734e48c77176108a4cd00429560fcba918ff6664a1f79ecdb843788fe`. All prior failed initialization/step-0 events remain preserved in `gpu_budget.failed_pre_run_events_preserved` (`890.42` cumulative GPU-seconds <= `5,400.0`s ceiling; `check_in_flight` inner-loop limitation noted per `REDTEAM_astra.md`).

---

## 6. Versioned S3 Custody Receipts (`--if-none-match "*"` + `get-object --version-id` Verified)

- **Lowercase `w1/` Manifest (`drv_t0520.md`)**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w1/w1_gpu_20261031T123502Z_326209/manifest.json` (`VersionId = O9E8sbnggmN.NQalGuVQ_n8c4f_Ko1Ti`, `SHA-256 = e34e243aa327f1dec830a97b91638f5c51c99fff33f2434655fb97823dda6900`, `2648` B, `version_id_readback_verified = True`).
- **Uppercase `W1/` Mirror Manifest**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W1/w1_gpu_20261031T123502Z_326209/manifest.json` (`VersionId = rvDTY5gnhR8uINZmpYyjc2q3pposwn3X`, `SHA-256 = fed8ec35fc9241f3fe82aa2bce228a76edb3f55bb46fb6153690952ebfab4d0b`, `2648` B, `version_id_readback_verified = True`).
- **`w1_wave_summary.json`**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w1/w1_gpu_20261031T123502Z_326209/w1_wave_summary.json` (`VersionId = i3d6.gwClv0BbVMCjUb4MRtK4QBDX3gN`, `SHA-256 = 2c37f7f25587d2d2cd78b95b47b70bceabd8e246812753c929d0463723de8951`, `339716` B).
- **`w1_cell_receipts.json`**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w1/w1_gpu_20261031T123502Z_326209/w1_cell_receipts.json` (`VersionId = L4ojrTOwTQKXXbriZrmA3lRHdUbcw5_0`, `SHA-256 = 45e899639307207b1c367335780ae945a53c2f2aa4e9f74b34a736236dc859ca`, `90372` B).
- **`w1_per_seed_family_vectors.json` (Original-Precision Replay Custody)**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w1/w1_gpu_20261031T123502Z_326209/w1_per_seed_family_vectors.json` (`VersionId = pOeS6mq9APnevXPLa8eBkS1I2W2s7CbI`, `SHA-256 = 101bcb9d82e3cc53dcf5a43ccbd4c8f813660a455a212c12d60b8f6605c75274`, `12663161` B).
- **`run_w1_gpu_campaign.py`**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/w1/w1_gpu_20261031T123502Z_326209/run_w1_gpu_campaign.py` (`VersionId = IwGu70zONwJKufzFrV2KfLc4R7nVHltu`, `SHA-256 = a7a2996734e48c77176108a4cd00429560fcba918ff6664a1f79ecdb843788fe`, `79510` B).
