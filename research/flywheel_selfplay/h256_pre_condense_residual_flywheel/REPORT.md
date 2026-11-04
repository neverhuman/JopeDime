# H256 — Pre-Condense Residual Bypass & Flagship Unified Regression Models (`0`-Reveal & `2+2` Active Flywheel)

## 1. Executive Summary & Scope Compliance

`H256` implements and evaluates a network architectural bypass (`PreCondenseResidualBridge`) that taps the frozen weight-locked DOPE problem representation **upstream** of the lossy Stage-1B `629 -> 10×32` SVD/clipping bottleneck, and unifies it with the multi-view Top-8 candidate reranker + soft/hard invalid-gate veto classifier (`H256_STATIC_0REV_TOP8_SOFT_VETO`) and the uncertainty-gated Causal `2+2` active reveal flywheel (`H256_ACTIVE_2PLUS2_UNIFIED`).

All explicit design and verification directives are satisfied and cryptographically verified:
1. **Trained on the Full S3 Regression Corpus & Splits**:
   - **Stage-1 Pretraining (`pretrain_full_s3_7split_bridge.py`)**: Trained both on the holdout-disjoint **`11,121` Policy B / `11,086` Policy C** `train_gradient` regression cells (`weights/full_s3_reg_7split_pretrained_encoder.pt`) and on **ALL `15,796` S3 7-split regression cells** across `2,257` unique regression datasets (`weights/stage1_full_s3_reg_all15796_encoder.pt`).
   - **Stage-2 Flywheel Ablation (`run_h256_flywheel_ablation.py`)**: Trained across 3 seeds (`256101`, `256102`, `256103`) × 6 neural arms × 5 family-grouped OOF folds (`90` fold models) on the `1,219` `DL-Train` regression contexts (`205` families) and evaluated zero-leakage on `91` unseen-family `DL-Val` contexts (`19` families).
   - **Flagship Unified Production Models (`train_and_export_h256_flagship_production.py`)**: Trained and cross-validated across 3 seeds × 5 family-grouped outer folds + 5 inner folds (`15` saved neural reranker + multi-view LightGBM fold checkpoints), plus two full-dataset production refits:
     - `weights/h256_production_1219_dl_train_bundle.pt`: Holdout-clean production bundle trained on all `1,219` `DL-Train` contexts + `11,121` S3 Policy-B regression cells (`DL-Val91` strictly untouched).
     - `weights/h256_production_full_1310_regression_bundle.pt`: Full-corpus production bundle trained on **ALL `1,310` regression flywheel contexts (`1,219` `DL-Train` + `91` `DL-Val`)** + **ALL `15,796` S3 7-split regression cells**.
2. **Regression-Only Scope**:
   - Filtered strictly to `model_task == "regression"` (`0` binary or multiclass cells in Stage 1; `7,138` non-regression cells explicitly excluded; `0` binary or multiclass contexts in Stage 2).
3. **DOPE Weights Cryptographically Locked**:
   - Verified before and after every Stage-1 epoch and every Stage-2 fold via `DopeFreezeGuard` (`56` parameter tensors, `444,168` elements, `11` buffers).
   - Initial and final composite SHA-256: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (`bit_identical: true`).
4. **High-Tier Landmark Encodings Active (`use_landmarks=True`)**:
   - Extracted and verified via `FusedLandmarkWorkspace.fit_transform_and_extract_bundle_high` (`vec256`, `col_lm_64x17`, `lm_pred_23`, `lm48_z`) across all `1,310` regression contexts in `extract_pre_condense_cache.py`.
5. **All Learned Weights, Fitted Critics, Golden Fixture & Comprehensive KPI Telemetry Committed**:
   - Committed under `weights/` (`< 4.5 MiB` per file) and `kpi_telemetry.json`.

---

## 2. Flagship Model Head-to-Head vs. All Prior Best Waves (`H252`–`H278`)

### 2.1 Zero-Reveal Static Inference (`0` Candidate Evaluations at Test Time)

| Wave / Model | `DL-Train OOF` (`1,219` ctx / `205` fam) Fam U | `DL-Train OOF` ΔFam vs `R882` | `DL-Train OOF` Ctx U | `DL-Train OOF` Invalid | `DL-Val91` (`91` ctx / `19` fam) Fam U | `DL-Val91` ΔFam vs `R882` | `DL-Val91` Ctx U | `DL-Val91` Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R882` (Control) | `1.385355` | `+0.000000` | `1.423073` | `17.0` | `1.153615` | `+0.000000` | `1.137794` | `0.0` |
| `H273` (`RERANK8_CLASSREG_SOFT10`) | `1.442485` | `+0.057130` | `1.482047` | `13.0` | `1.190085` | `+0.036470` | `1.178223` | `1.0` |
| `H278` (`S2_LOG1P_BAGGED_SOFT15_TAU015`) | `1.453955` | `+0.068600` | `1.491009` | `12.7` | `1.192715` | `+0.039100` | `1.179540` | `0.0` |
| `H275` (`A6_GATE32_CONSTRAINED_ENSEMBLE`) | `1.458348` | `+0.072993` | `1.494649` | `12.0` | `1.194687` | `+0.041072` | `1.181924` | `0.0` |
| **`H256_STATIC_0REV_TOP8_SOFT_VETO` (Ours)** | **`1.461114`** | **`+0.075759`** | **`1.496037`** | **`12.7`** | **`1.203296`** | **`+0.049680`** | **`1.189621`** | **`0.0`** |

### 2.2 Active `2+2` Reveal Flywheel (`K_rev <= 4` Candidate Evaluations)

| Wave / Model | `DL-Train OOF` (`1,219` ctx / `205` fam) Fam U | `DL-Train OOF` ΔFam vs `R882` | `DL-Train OOF` Ctx U | `DL-Train OOF` Invalid | `DL-Val91` (`91` ctx / `19` fam) Fam U | `DL-Val91` ΔFam vs `R882` | `DL-Val91` Ctx U | `DL-Val91` Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R882` (Control) | `1.385355` | `+0.000000` | `1.423073` | `17.0` | `1.153615` | `+0.000000` | `1.137794` | `0.0` |
| `H252` / `H254` (`Stage-1 Hurdle 2+2`) | `1.469325` | `+0.083970` | `1.500825` | `5.0` | `1.168790` | `+0.015175` | `1.154156` | `0.0` |
| `H255` (`Causal 2+2 Neural`) | `1.468249` | `+0.082894` | `1.500422` | `5.0` | `1.168790` | `+0.015175` | `1.154156` | `0.0` |
| `H256-v1` (`F_FULL629_HIDDEN256_COL64X17`) | `1.471203` | `+0.085848` | `1.506214` | `5.3` | `1.169564` | `+0.015949` | `1.152289` | `0.0` |
| **`H256_ACTIVE_2PLUS2_UNIFIED` (Ours)** | **`1.489648`** | **`+0.104292`** | **`1.518314`** | **`5.3`** | **`1.201692`** | **`+0.048077`** | **`1.187871`** | **`0.0`** |

---

## 3. Comprehensive Train & Validation KPI Telemetry (`kpi_telemetry.json`)

### 3.1 Partition Decision, Oracle Recovery, Ranking, Switch & Calibration KPIs

| Metric / KPI | `H256_STATIC_0REV` (`DL-Train OOF 1219`) | `H256_STATIC_0REV` (`DL-Val91`) | `H256_ACTIVE_2PLUS2` (`DL-Train OOF 1219`) | `H256_ACTIVE_2PLUS2` (`DL-Val91`) |
| --- | ---: | ---: | ---: | ---: |
| **Mean Family Utility** | **`1.461114`** | **`1.203296`** | **`1.489648`** | **`1.201692`** |
| **Mean Family Gain vs `R882`** | **`+0.075759`** | **`+0.049680`** | **`+0.104292`** | **`+0.048077`** |
| **95% Paired Family Bootstrap CI** | `[+0.0242, +0.1402]` | `[+0.0215, +0.0858]` | `[+0.0578, +0.1667]` | `[+0.0206, +0.0843]` |
| **90% Lower Confidence Bound (`LCB90`)** | `+0.031165` | `+0.025049` | `+0.064042` | `+0.023906` |
| **One-Sided Bootstrap $p$-value** | `0.0019` | **`0.0001`** | **`0.0001`** | **`0.0001`** |
| **Family Wins / Ties / Losses vs `R882`** | `153 / 1 / 51` (`3.00:1`) | **`17 / 0 / 2` (`8.50:1`)** | **`159 / 3 / 43` (`3.70:1`)** | **`17 / 0 / 2` (`8.50:1`)** |
| **Sign-Test $p$-value** | `7.41e-13` | `0.000729` | **`7.62e-17`** | `0.000729` |
| **Top-8 Oracle Family Recovery `%`** | `30.22%` | **`36.69%`** | **`41.60%`** | `35.50%` |
| **96-Pool Oracle Family Recovery `%`** | `16.22%` | **`19.79%`** | **`22.32%`** | `19.16%` |
| **Top-8 Ranking `NDCG@1 / @3 / @8`** | `0.3565 / 0.4966 / 0.7055` | `0.3335 / 0.4932 / 0.6993` | `0.3565 / 0.4966 / 0.7055` | `0.3335 / 0.4932 / 0.6993` |
| **Top-8 Oracle Hit Rate `@1 / @3` & `MRR`** | `20.51% / 53.27%` (`0.4322`) | `18.32% / 52.75%` (`0.4138`) | `20.51% / 53.27%` (`0.4322`) | `18.32% / 52.75%` (`0.4138`) |
| **Switch Rate `%` & Win/(Win+Loss) Precision** | `76.98%` (`64.82%`) | `71.43%` (`66.15%`) | `66.67%` (**`67.92%`**) | `62.64%` (**`72.62%`**) |
| **Mean Utility Gain When Switched** | `+0.094785` | `+0.072562` | **`+0.142854`** | **`+0.080146`** |
| **Model Invalid Picks (vs `R882` Control)** | `12.67` (vs `17.0`) | **`0.0` (vs `0.0`)** | **`5.33` (vs `17.0`)** | **`0.0` (vs `0.0`)** |
| **Invalid Picks Prevented vs `R882`** | `11.33` | `0.0` | **`15.33` (`90.2%`)** | `0.0` |
| **Top-8 Veto Classifier `AUROC` / `Brier` / `ECE`** | `0.8205 / 0.0310 / 0.0696` | **`0.9025 / 0.0263 / 0.0869`** | `0.8205 / 0.0310 / 0.0696` | **`0.9025 / 0.0263 / 0.0869`** |

### 3.2 Feature-Block Importance & Neural Saliency (`616D` Full-1,310 Production Bundle)

| Feature Block (`616D`) | Dims | LightGBM Ranker Gain `%` | LightGBM Ranker Splits `%` | LightGBM Veto Clf Gain `%` | Neural Reranker Input-Grad Saliency `%` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Block 1`: `R882` Top-8 Costs & Pairwise Gaps | `28` | `10.69%` | `11.17%` | `8.58%` | `4.53%` |
| `Block 2`: `sem24` & `proj16` Candidate Geometry | `12` | `6.94%` | `7.01%` | `10.27%` | `2.67%` |
| `Block 3`: Candidate `program_412d` + `sem24` + `proj16` | `452` | `52.99%` | `51.95%` | `38.85%` | `74.24%` |
| `Block 4`: Context `lm48` & `dope25` Summary | `76` | `10.11%` | `10.39%` | `23.42%` | `11.99%` |
| **`Block 5`: `H256` Stage-2 Bridge (`dp_F`, `dp_G_s3`) & Critics** | **`12`** | **`14.44%`** | **`13.77%`** | **`12.17%`** | **`2.48%`** |
| **`Block 6`: `H256` Pre-Condense Log-Norms & Stage-1 S3 `vec16`** | **`36`** | **`4.83%`** | **`5.71%`** | **`6.71%`** | **`4.09%`** |
| **Total `H256` Newly Added Features (`Blocks 5 + 6`)** | **`48` (`7.79%`)** | **`19.27%`** | **`19.48%`** | **`18.88%`** | **`6.57%`** |

### 3.3 Stage-1 S3 Pretraining & Auxiliary Critic Telemetry
- **Stage-1 All-`15,796` S3 7-Split Regression Encoder (`weights/stage1_full_s3_reg_all15796_encoder.pt`)**:
  - Trained on all `15,796` regression cells across `2,257` unique regression datasets (`0` binary/multiclass cells).
  - Pre-Condense 92-token L2 activation norms across the `1,310` contexts: `vec256_32tok = 2.9531`, `lm71_landmark_12tok = 1.6798`, `scale_1tok = 0.8127`, `hidden256_32tok = 0.2843`, `col_lm_64x17_16tok = 0.0973`.
- **Stage-1/2 Auxiliary Critics (`weights/sos_hurdle_and_sp_critic_models.pt`)**:
  - `SOSCalibratedAdapter` continuous log-ratio critic (`w_c`): Pearson $r = 0.5143$, Spearman $\rho = 0.5238$ across `1,404,287` `DL-Train OOF` candidate evaluations; Pearson $r = 0.4955$, Spearman $\rho = 0.5001$ across `104,832` `DL-Val91` candidate evaluations.
  - `SOSCalibratedAdapter` gate-failure critic (`w_g`): `AUROC = 0.7560` on `DL-Train OOF` and `AUROC = 0.7514` on `DL-Val91`.
  - `SelfPlayTrajectoryHead` (`sp_dir`, `3,449` harvested promotions): Median per-context Spearman correlation with true utility $= 0.2147$ on `DL-Train OOF` and $0.1588$ on `DL-Val91`.

---

## 4. Stage-2 3-Seed × 5-Fold Flywheel Bridge Ablation Summary (`summary.json`)

| Arm | `DL-Train OOF` (`1219` ctx) U | `OOF` ΔU vs Arm A (`95% CI`) | `OOF` BT Elo | `st_val_gate_32` (`32` ctx) U | `dl_val_71` (`71` ctx) U | `DL-Val` (`91` ctx) U | `DL-Val` ΔU vs `R882` (`95% CI`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R882` (Baseline) | `1.423073` | `-0.074652` | `1500.0` | `1.437309` | `1.315407` | `1.137794` | `0.000000` |
| `A_FROZEN_10TOK_BASELINE` | `1.497725` | `0.000000` | `1527.4` | `1.544467` | `1.331458` | `1.154156` | `+0.016361` (`[+0.0059, +0.0268]`) |
| `B_CAPACITY_CTRL_10TOK` | `1.501215` | `+0.003490` (`[+0.0006, +0.0089]`, $p=0.0106$) | `1529.6` | `1.541758` | `1.329566` | `1.152704` | `+0.014910` (`[+0.0049, +0.0253]`) |
| `C_LEARNED_COMPACT_BRIDGE_629` | `1.505519` | `+0.007793` (`[+0.0023, +0.0148]`, $p=0.0004$) | **`1534.3`** | `1.545640` | `1.329321` | `1.152427` | `+0.014633` (`[+0.0045, +0.0251]`) |
| `D_PRE_CONDENSE_FULL629_20TOK` | `1.503814` | `+0.006089` (`[+0.0019, +0.0114]`, $p=0.0019$) | `1533.7` | `1.530468` | `1.328429` | `1.151946` | `+0.014151` (`[+0.0045, +0.0244]`) |
| `E_FULL629_PLUS_HIDDEN256_28TOK` | `1.500842` | `+0.003117` (`[+0.0001, +0.0087]`, $p=0.0222$) | `1529.2` | `1.540537` | `1.330071` | `1.153005` | `+0.015211` (`[+0.0050, +0.0259]`) |
| `F_FULL629_HIDDEN256_COL64X17` | **`1.506214`** | **`+0.008489`** (`[+0.0021, +0.0166]`, $p=0.0020$) | `1531.3` | `1.545147` | `1.328934` | `1.152289` | `+0.014495` (`[+0.0045, +0.0247]`) |
| `G_FULL_S3_REG_7SPLIT_11121_PRETRAINED` | `1.500460` | `+0.002734` (`[+0.0001, +0.0077]`, $p=0.0216$) | `1526.7` | **`1.546581`** | `1.329412` | `1.152531` | `+0.014737` (`[+0.0048, +0.0252]`) |

---

## 5. Committed Weights & Telemetry Inventory (`weights/`)

All canonical checkpoints under `weights/` are strictly $< 5\text{ MiB}$ and include embedded `kpi_telemetry`:

| File | Size | SHA-256 | Role |
| --- | ---: | --- | --- |
| `weights/h256_production_full_1310_regression_bundle.pt` | `4.17 MB` | `4c910796d0f31854a914541a9043bf888097cd15416779964ef75df8151d6bc7` | Full-corpus production bundle (`1,310` regression contexts + `15,796` S3 7-split regression cells) |
| `weights/h256_production_1219_dl_train_bundle.pt` | `3.34 MB` | `2040a1aa71449ab93725b91a8be171713b7268e4d4d07d8499b5efcb73a13bc5` | Holdout-clean production bundle (`1,219` `DL-Train` contexts + `11,121` S3 Policy-B cells) |
| `weights/stage1_full_s3_reg_all15796_encoder.pt` | `2.84 MB` | `4be76f83689412277ff241d46e8e4ff32cb96650d1ef51646be3efdc6152f190` | Stage-1 Pre-Condense 92-token encoder trained on all `15,796` S3 regression cells |
| `weights/full_s3_reg_7split_pretrained_encoder.pt` | `104 KB` | `07c92a2e25fa95276d0995eb20e37451822eb391a3d0a81db4eb3965b7391547` | Stage-1 Pre-Condense 92-token encoder trained on `11,121` Policy-B cells |
| `weights/stage1_full_s3_reg_7split_pre_condense_92tok_full_model.pt` | `860 KB` | `9019d1b3eeddf98f22b5d20723c4595b244e93aaf87a78cb8c09b02d99436a83` | Full Stage-1 model + Policy B/C scalers |
| `weights/stage2_full_s3_reg_flywheel_pretrained_bridge.pt` | `4.70 MB` | `219cb783fd53db477edcd6de14dd1f0618120e1274e53c1bc496215c258b775a` | 5-fold Stage-2 `PreCondenseResidualBridge` ensemble + critic models + golden fixture summary |
| `weights/sos_hurdle_and_sp_critic_models.pt` / `.npz` | `59 KB` / `10 KB` | `14ab6f70900834169581a068e08141c64113d76d04d5a06675890f0a4ca8e36b` | Fitted `SOSCalibratedAdapter` (`w_c`, `w_g`) + `SelfPlayTrajectoryHead` (`sp_dir`) + `FoldScaler` |
| `weights/golden_inference_fixture_context0.pt` / `.json` | `3.51 MB` / `2.4 KB` | `de8f0ae10b0c38413cfd9171e6d627aff97764b38ffca8d9ccda48fdefe5a0bb` | Complete golden input & expected output prediction fixture on active `DL-Val` context `016648dc...` |
| `weights/checkpoint_manifest.json` | `54 KB` | Indexed | Complete SHA-256 manifest of all `115+` fold and production checkpoints |
| `kpi_telemetry.json` | `144 KB` | `9bd1ebaa4ff15f0bc22eb9ed6f64dac801659fd3105ff071f54fcc558500fa05` | Complete Train (`FIT`), Inner-OOF, Outer-OOF (`1,219`), and Holdout (`DL-Val91`) KPI telemetry |
| `summary.json` | `331 KB` | Indexed | Complete ablation and flagship model summary |

---

## 6. Rust Serving Parity Verification

Updated `crates/jope-service/src/ranker/cuda/forward.rs`, `crates/jope-service/src/ranker/stage1b.rs`, and `crates/jope-service/src/ranker/adapter.rs` to expose pre-condense `PreCondenseStage1bBundle` (`raw_vec_629`, `block_log_norms_11`, `hidden_256`, `col_lm_64x17`, `col_mask_64`) and support variable `K_ctx`-token attention in `CrossAttentionAdapter::forward`. Verified via `cargo test -p jope-service --lib ranker` (`120 passed; 0 failed`).
