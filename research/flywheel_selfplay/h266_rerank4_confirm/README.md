# H266 — Confirmatory Fresh-Split Replication & Feature-Block Ablation of `H265` (`RERANK4`)

- **Wave ID**: `h266-rerank4-confirm-20261103` (`H266`)
- **Study Path**: `research/flywheel_selfplay/h266_rerank4_confirm`
- **Owner**: `Argon`
- **Status**: `COMPLETE`
- **Verdict**: **`NO_ADVANCE`** (`confirmation_status = NOT_CONFIRMED`: **SECONDARY `DL-Val91` REPLICATED (`PASS`)**, **PRIMARY `DL-Train` OOF positive (`+0.018938`, `3/3` seeds, `raw p = 0.0495`, `Holm p = 0.0495`) but missed bootstrap `LCB95 > 0` (`[-0.004329, +0.051845]`) under the strict `H257` inner-`FIT` gate**)
- **Preregistration Commit (`origin/v1/next`)**: `6d662d708b859ac181ea745912727f38ef4b0a74` (`bound: 793964789b3c9c8b3eaea0ff249d9d1389f1467c`, `0` outcomes read)
- **S3 Preregistration Snapshot**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H266/prereg-6d662d708b85/prereg_s3_manifest.json` (`VersionId = 3ubnOj0RBjMye3wF9V6ZjOo9WXsViRVC`, `sha256 = 7e8b79df5469bc1a6c539cfe837fec6826cffc879552a36c502aa93fbc20c42f`)
- **S3 Run Custody (`attempt-001`)**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H266/attempt-001/s3_manifest.json` (`VersionId = MKn2ALWFB2Akgew3Uhhtp47i_cKA1GpN`, `sha256 = 76d6d425c0903ac2edf2aebce1343fc93d29cfa6c3d7e59dd5ec2df795cb9711`)
- **Untouched Full `metrics.json` on S3 (`218,715` B)**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H266/attempt-001/metrics.json` (`VersionId = RYGkclV0aJZ4YvamtM9Yo8VM_KBAEGi6`, `sha256 = 083edb422ce1d97ff5277ff5b3ddee8f0ebe39ce63df2e96591d8efe2ee5d160`)
- **Full `details.json` on S3 (`376,575` B)**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H266/attempt-001/details.json` (`VersionId = LvbWHQX6LOd9D2wl8yErvM.v4egl3Tuq`, `sha256 = 8d234f5f371f9bdb72b1cc6d452292d472740b01a12f29d829b5df5d277f5dcc`)
- **Full `panels.npz` on S3 (`509,867` B)**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H266/attempt-001/panels.npz` (`VersionId = Ua9dfJiAuXfw1l8_JymyUavWnr09GDC1`, `sha256 = c549f3ffb428af8dc14fa0830bd309c3d22271fc80f7475ecfbfcee4866a0365`)
- **Global FIT Selection Lock (`fit_selections_lock.json`)**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H266/attempt-001/fit_selections_lock.json` (`VersionId = 1ZFL4gTjf.6OT_9CD2fIZDa1sYyUTXQk`, `sha256 = ad5be106f53d754b8bf48c2601c39809c4269254ca0f296eeb0167855149f40d`)

---

## 1. Hypothesis & Experimental Setup

`H265` (`da5a08afb`) reported an exploratory `ADVANCE` (`+0.018685` equal-family utility gain on `DL-Val91`, `95% CI [+0.006369, +0.030822]`, `Holm p = 0.0002`, `3/3` seeds, `0` invalid `-2.0` gates) using `RERANK4`, a zero-update (`0` neural weight updates, `0` test-time reveals, scored **SINGLE-PICK**) listwise LambdaMART (`LightGBM LGBMRanker`, `objective="lambdarank"`) over `r882`'s top-`4` candidates per context using a `560`-dimensional outcome-free feature representation.

`H266` tests whether the frozen `H265` `RERANK4` pipeline (zero hyperparameter, feature, selector, or margin-grid adjustments) confirms out-of-fold across:
1. **PRIMARY (`DL-Train` OOF, `205` families / `1,219` contexts)**: Repeated nested `5`-fold family-grouped cross-validation across `3` fresh fold seeds (`266140, 266141, 266142`) $\times$ `5` folds $\times$ `3` fresh model seeds (`266101, 266102, 266103`) = `45` model cells per arm (`10,000`-draw paired family bootstrap, `boot_seed = 266150`, per-seed medians across the `3` fold seeds).
2. **SECONDARY (`DL-Val91`, `19` families / `91` contexts)**: Fresh-seed replication (`fold_seed = 266140`, `model_seeds = 266101–266103`, plus `3`-fold-seed aggregate) after reproducing `Native91 = 1.1377941898164525` (`91/91` bit-exact).
3. **Descriptive Feature-Block Ablations (`fold_seed = 266140`, `model_seeds = 266101–266103`, not in Holm family)**: `SCORE_ONLY_20` (`20` cols), `SCORE_ONLY` (`32` cols), `NO_SCOREGEOM` (`528` cols), `NO_SEM24` (`531` cols), and `NO_CTX76` (`484` cols).

---

## 2. Baseline Regression & Hygiene Audits

- **Baseline Regression (`baseline.json`, `sha256 = 56d9501eb482190ed3ebf83bab12441a80528c46e0e3d44ff1ce0218aa447150`)**:
  - `CONTROL` (`r882` top-1): `DL-Val91` context mean = `1.1377941898164525` (`abs_diff = 0.0`, `91/91` contexts bit-exact), family mean = `1.1489359425732622` (`abs_diff = 0.0`).
  - `ORACLE_TOP4` (`TOP4_PICK@r882`): context mean = `1.2403601682250796` (`abs_diff = 0.0`), family mean = `1.2519997168684853` (`abs_diff = 0.0`), oracle gain `+0.103064` fam / `+0.102566` ctx.
  - `DL-Train` (`1,219` contexts / `205` families): `CONTROL` (`r882`) family mean = `1.385355432993183`, context mean = `1.423073434552299`; `ORACLE_TOP4` family mean = `1.6626179025810164` (`+0.277262`), context mean = `1.7016030392341578` (`+0.278530`).
- **Checkpoint & Compute Hygiene**:
  - `dope_backbone` (`stage1_stage2.pt`) `sha256` before/after: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (unchanged).
  - `r882` checkpoint `sha256` before/after: `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` (unchanged).
  - `optimizer_updates = 0`, `test_time_reveals_used = 0`, `CUDA_VISIBLE_DEVICES = ""` (CPU-only, `2` threads on cores `57–60`, peak RSS `857,968,640` bytes = `0.799 GiB`, wall time `1096.13s`).
- **Required Provenance & Disjointness Audits**:
  1. **`h58` Cache Provenance (`jope.h58.context-feature-cache/v3`)**: Verified `context_features.npz` (`sha256 = 75704e82...`, keys `cid, ctx76, split_code`) and `pool_priors.npz` (`sha256 = 6db567ef...`, keys `prog, base_prior_cost, h36`) contain `0` `DL-Val` or `DL-Train` fitness/gate outcomes (`excluded_payloads_opened = 0`, `outer_rows_delivered = 0`). `base_prior_cost` is the native prior model input extracted from `stage1_stage2.pt` (`sha256 = 136b2381c312fd4bf267249fe0543aa44f5b780a05d57f974d43dd4277588b6d`).
  2. **`FIT` / `ASSESS` / `DL-Val` Family Disjointness**: Verified `0` family overlap between `DL-Train` (`205` families) and `DL-Val` (`19` families), and `0` family overlap across all `5` folds in each of the `3` fresh fold seeds (`266140, 266141, 266142`). All `90` confirmatory and `75` ablation inner-`FIT` selections were locked and `fsync`-written to `fit_selections_lock.json` (`sha256 = ad5be106...`, `dl_val_reads_before_fit_lock = 0`) before any `DL-Val` evaluation.
  3. **`sem_rev03` Definition**: Verified `sem_rev03 = sem64[:4]` is the `24`-dim structural semantic signature `compute_sem24_numpy(prog[:4])` of fixed pool rows `0..3` (candidate program descriptors only; `0` outcomes read).

---

## 3. Primary & Secondary Confirmatory Results

### 3.1 Confirmatory Gate Summary (`Holm m = 2` over `{PRIMARY, SECONDARY}`)

| Gate | Criterion | Observed | Status |
| --- | --- | --- | --- |
| Global `FIT` lock admission | `>= 1/45` cells admitted for `RERANK4` | `33 / 45` (`8/15`, `13/15`, `12/15` vs `1/45` `SHUF`) | **PASS** |
| **PRIMARY (`DL-Train` `205` fam)** median family gain | `> +0.0020` | **`+0.018938`** (ctx `+0.018354`) | **PASS** |
| **PRIMARY (`DL-Train` `205` fam)** per-seed family gains | `3 / 3` seeds `> 0` | **`[+0.026254, +0.018938, +0.014425]`** (`3/3`) | **PASS** |
| **PRIMARY (`DL-Train` `205` fam)** step-down Holm `p` (`m=2`) | `< 0.05` | **`0.049495`** (`raw p = 0.049495`) | **PASS** |
| **PRIMARY (`DL-Train` `205` fam)** beats `RERANK4_SHUF` | `gain > 0` and `p < 0.05` | **`+0.018938`** (`raw p = 0.049495`, `3/3` seeds) | **PASS** |
| **PRIMARY (`DL-Train` `205` fam)** paired bootstrap `LCB95` / `Holm LCB97.5` | `> 0.0` | **`95% CI [-0.004329, +0.051845]`**, **`97.5% CI [-0.008610, +0.055949]`** | **FAIL** |
| **SECONDARY (`DL-Val91` `19` fam)** median family gain | `> 0.0` (`> +0.0020`) | **`+0.012861`** (ctx `+0.013426`) | **PASS** |
| **SECONDARY (`DL-Val91` `19` fam)** per-seed family gains | `3 / 3` seeds `> 0` | **`[+0.012861, +0.018362, +0.010068]`** (`3/3`) | **PASS** |
| **SECONDARY (`DL-Val91` `19` fam)** `LCB95` & `Holm LCB97.5` | `> 0.0` | **`95% CI [+0.004095, +0.022738]`**, **`97.5% CI [+0.003281, +0.024504]`** (`Holm p = 0.001600`) | **PASS** |
| **SECONDARY (`DL-Val91` `19` fam)** beats `RERANK4_SHUF` | `gain > 0` and `p < 0.05` | **`+0.013983`** (`95% CI [+0.003785, +0.023410]`, `raw p = 0.000800`) | **PASS** |

### 3.2 PRIMARY Endpoint Breakdown (`DL-Train` OOF, `205` Families / `1,219` Contexts)

| Contrast / Slice | Median Fam Gain | Median Ctx Gain | `95% CI` (Fam) | `Holm 97.5% CI` | Raw `p` | Seeds (`266101 / 266102 / 266103`) | Fam W/T/L (by seed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`RERANK4 - CONTROL` (Primary `3`-fseed median)** | **`+0.018938`** | **`+0.018354`** | `[-0.004329, +0.051845]` | `[-0.008610, +0.055949]` | `0.049495` | `+0.026254 / +0.018938 / +0.014425` (`3/3`) | `96/75/34`, `77/89/39`, `86/80/39` |
| `RERANK4 - CONTROL` (`3`-fseed mean) | `+0.019559` | `+0.017609` | `[-0.004380, +0.047281]` | `[-0.009123, +0.051212]` | `0.064194` | `+0.025596 / +0.012704 / +0.019559` (`3/3`) | `124/23/58`, `121/25/59`, `124/22/59` |
| `RERANK4 - CONTROL` (`fold_seed = 266140`, `8/15` adm) | `+0.009074` | `+0.012470` | `[-0.015737, +0.042375]` | `[-0.020961, +0.048046]` | `0.180082` | `+0.009074 / +0.007706 / +0.018003` (`3/3`) | `84/80/41`, `55/116/34`, `79/84/42` |
| `RERANK4 - CONTROL` (`fold_seed = 266141`, `13/15` adm) | `+0.027734` | `+0.018231` | `[-0.007268, +0.058970]` | `[-0.011902, +0.064231]` | `0.068393` | `+0.038383 / +0.027734 / +0.008688` (`3/3`) | `102/53/50`, `99/54/52`, `79/79/47` |
| `RERANK4 - CONTROL` (`fold_seed = 266142`, `12/15` adm) | `+0.021733` | `+0.024313` | `[-0.007866, +0.055434]` | `[-0.012649, +0.059889]` | `0.081192` | `+0.029332 / +0.002671 / +0.021733` (`3/3`) | `106/55/44`, `76/80/49`, `100/57/48` |
| `RERANK4_SHUF - CONTROL` (Placebo, `1/45` adm) | `+0.000000` | `+0.000000` | `[+0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `1.000000` | `0.000000 / 0.000000 / 0.000000` (`0/3`) | `0/205/0`, `0/205/0`, `0/205/0` |
| `RERANK4 - RERANK4_SHUF` | `+0.018938` | `+0.018354` | `[-0.004329, +0.051845]` | `[-0.008610, +0.055949]` | `0.049495` | `+0.026254 / +0.018938 / +0.014425` (`3/3`) | `96/75/34`, `77/89/39`, `86/80/39` |
| **`RERANK4_UNC - CONTROL` (Diagnostic, no `LCB` fallback)** | **`+0.035589`** | **`+0.035404`** | **`[+0.005382, +0.079481]`** | **`[+0.001054, +0.085623]`** | **`0.010499`** | **`+0.034400 / +0.036830 / +0.035589` (`3/3`)** | `110/47/48`, `104/52/49`, `105/50/50` |
| **`RERANK4_M0 - CONTROL` (Diagnostic `m = 0` fixed)** | **`+0.043218`** | **`+0.045420`** | **`[+0.000365, +0.099038]`** | `[-0.006354, +0.107268]` | **`0.024098`** | **`+0.026303 / +0.043218 / +0.049141` (`3/3`)** | `107/38/60`, `111/35/59`, `114/33/58` |

### 3.3 SECONDARY Endpoint Breakdown (`DL-Val91`, `19` Families / `91` Contexts) — **REPLICATED (`PASS`)**

| Contrast / Slice | Median Fam Gain | Median Ctx Gain | `95% CI` (Fam) | `Holm 97.5% CI` | Raw `p` | Seeds (`266101 / 266102 / 266103`) | Fam W/T/L (by seed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`RERANK4 - CONTROL` (`fold_seed = 266140`)** | **`+0.012861`** | **`+0.013426`** | **`[+0.004095, +0.022738]`** | **`[+0.003281, +0.024504]`** | **`0.000800`** | **`+0.012861 / +0.018362 / +0.010068` (`3/3`)** | `13/2/4`, `15/0/4`, `15/1/3` |
| **`RERANK4 - CONTROL` (`3`-fold-seed aggregate)** | **`+0.015518`** | **`+0.015926`** | **`[+0.005870, +0.029510]`** | **`[+0.004425, +0.031526]`** | **`0.000300`** | **`+0.017180 / +0.014574 / +0.015518` (`3/3`)** | `15/1/3`, `14/0/5`, `15/1/3` |
| `RERANK4_SHUF - CONTROL` (`fold_seed = 266140`) | `+0.000000` | `+0.000000` | `[+0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `1.000000` | `0.000000 / 0.000000 / 0.000000` (`0/3`) | `0/19/0`, `0/19/0`, `0/19/0` |
| `RERANK4 - RERANK4_SHUF` (`fold_seed = 266140`) | `+0.013983` | `+0.014331` | `[+0.003785, +0.023410]` | `[+0.002975, +0.025216]` | `0.000800` | `+0.013983 / +0.018362 / +0.010068` (`3/3`) | `14/1/4`, `15/0/4`, `15/1/3` |
| `RERANK4_UNC - CONTROL` (`fold_seed = 266140`) | `+0.015219` | `+0.014075` | `[-0.002320, +0.029434]` | `[-0.004968, +0.031582]` | `0.044596` | `+0.018412 / +0.015219 / +0.008564` (`3/3`) | `16/0/3`, `15/0/4`, `15/0/4` |
| `RERANK4_M0 - CONTROL` (`fold_seed = 266140`) | `+0.003728` | `+0.001815` | `[-0.017603, +0.023828]` | `[-0.021163, +0.027314]` | `0.369463` | `+0.003728 / +0.016143 / -0.000723` (`2/3`) | `14/0/5`, `15/0/4`, `13/0/6` |

### 3.4 `400`-Iteration MM Bradley-Terry Elo (`CONTROL = r882` Anchored at `1500.0`)

| Arm | `DL-Train` OOF (`205` fam) Elo (`95% CI`) | `DL-Val91` (`19` fam, `fseed=266140`) Elo (`95% CI`) | `DL-Train` `-2.0` Invalid Picks (of `3,657`/seed) | `DL-Val91` `-2.0` Invalid Picks (of `455`/seed) |
| --- | --- | --- | --- | --- |
| `CONTROL` (`r882`) | `1500.00 [1500.00, 1500.00]` | `1500.00 [1500.00, 1500.00]` | `41 / 41 / 41` (`1.12%`) | `0 / 0 / 0` (`0.00%`) |
| **`RERANK4` (Strict)** | **`1531.50 [1520.12, 1540.78]`** | **`1520.99 [1508.94, 1548.00]`** | `55 / 55 / 56` (`1.50%`) | **`0 / 0 / 0` (`0.00%`)** |
| `RERANK4_SHUF` | `1500.00 [1500.00, 1500.00]` | `1500.00 [1500.00, 1500.00]` | `41 / 41 / 41` (`1.12%`) | `0 / 0 / 0` (`0.00%`) |
| `RERANK4_UNC` | `1542.17 [1532.75, 1558.75]` | `1531.69 [1508.35, 1570.39]` | `65 / 67 / 68` (`1.80%`) | `1 / 1 / 1` (`0.22%`) |
| `RERANK4_M0` | `1544.70 [1531.96, 1565.84]` | `1517.85 [1484.56, 1561.73]` | `84 / 76 / 83` (`2.22%`) | `3 / 1 / 3` (`0.44%`) |
| `ORACLE_TOP4` | `1763.93 [1733.36, 1791.47]` | `1649.14 [1586.57, 1732.02]` | `0 / 0 / 0` (`0.00%`) | `0 / 0 / 0` (`0.00%`) |
| `ORACLE_POOL` | `1911.42 [1877.67, 1941.32]` | `1804.64 [1725.55, 1923.38]` | `0 / 0 / 0` (`0.00%`) | `0 / 0 / 0` (`0.00%`) |

---

## 4. Descriptive Feature-Block Ablations (`fold_seed = 266140`, `model_seeds = 266101–266103`)

| Ablation Arm | Cols | Admitted (of `15`) | `DL-Train` OOF Strict Gain (`95% CI`, `p`) | `DL-Train` OOF UNC Gain (`95% CI`, `p`) | `DL-Val91` Strict Gain (`95% CI`, `p`) |
| --- | --- | --- | --- | --- | --- |
| **`FULL RERANK4`** | `560` | `8 / 15` | `+0.009074` (`[-0.015737, +0.042375]`, `p = 0.1801`, `3/3`) | `+0.032998` (`[-0.005609, +0.078986]`, `p = 0.0434`, `3/3`) | **`+0.012861` (`[+0.004095, +0.022738]`, `p = 0.0008`, `3/3`)** |
| **`SCORE_ONLY_20`** (only scalar `r882` score/gap cols `0..19`) | `20` | **`0 / 15`** | **`+0.000000`** (`[0.0, 0.0]`, `p = 1.0`, `0/3`) | `-0.001470` (`[-0.017061, +0.016552]`, `p = 0.5567`, `1/3`) | **`+0.000000`** (`[0.0, 0.0]`, `p = 1.0`, `0/3`; UNC `-0.016407`) |
| **`SCORE_ONLY`** (score + cosines cols `0..31`) | `32` | `6 / 15` | **`+0.001988`** (`[-0.008074, +0.015014]`, `p = 0.3437`, `2/3`) | `+0.013865` (`[-0.011464, +0.045791]`, `p = 0.1464`, `2/3`) | **`+0.000390`** (`[-0.001809, +0.012543]`, `p = 0.2836`, `2/3`) |
| **`NO_SCOREGEOM`** (drop cols `0..31`; keep `sem24`, `prog48`, `h36`, `ctx76`) | `528` | `10 / 15` | **`+0.012210` (`[+0.000462, +0.047014]`, `p = 0.0228`, `3/3`)** | **`+0.038922` (`[+0.000605, +0.088267]`, `p = 0.0235`, `3/3`)** | **`+0.020092` (`[+0.003153, +0.032188]`, `p = 0.0068`, `3/3`)** |
| **`NO_SEM24`** (drop `sem24` cols `444..467` + `sem24` cosines) | `531` | `9 / 15` | `+0.008827` (`[-0.017997, +0.044680]`, `p = 0.2484`, `2/3`) | `+0.025634` (`[-0.014407, +0.072579]`, `p = 0.1112`, `3/3`) | `+0.014961` (`[+0.006024, +0.028191]`, `p = 0.0004`, `3/3`) |
| **`NO_CTX76`** (drop `ctx76` interaction cols `484..559`) | `484` | `9 / 15` | `+0.011909` (`[-0.019066, +0.039942]`, `p = 0.2513`, `3/3`) | `+0.022670` (`[-0.021163, +0.070894]`, `p = 0.1510`, `3/3`) | `+0.010854` (`[+0.003404, +0.022164]`, `p = 0.0006`, `3/3`) |

**Key Ablation Takeaway**:
- `SCORE_ONLY_20` (`0/15` cells admitted, `+0.000000` strict, `-0.001470` UNC on `DL-Train`, `-0.016407` UNC on `DL-Val91`) and `SCORE_ONLY` (`+0.001988` train `p=0.3437`, `+0.000390` val `p=0.2836`) **completely fail**, proving that `RERANK4`'s gain does **not** come from recalibrating `r882`'s scalar scores.
- Conversely, **`NO_SCOREGEOM`** (which drops all `32` score/gap/cosine features and uses only candidate descriptors `prog48`, `sem24`, `h36`, `base_prior_cost`, and `ctx76` interactions) achieves **`+0.012210` (`95% CI [+0.000462, +0.047014]`, `p = 0.0228`, `3/3` seeds) on `DL-Train` OOF** and **`+0.020092` (`95% CI [+0.003153, +0.032188]`, `p = 0.0068`, `3/3` seeds) on `DL-Val91`**.
- Feature importance share on `fold_seed = 266140, seed = 266101, fold = 0`: `cand_raw_440` (`72.64%`), `ctx_interactions_76` (`15.65%`), `score_geometry_32` (`6.06%`), `cand_sem24_24` (`5.21%`), `cand_z_within_top4_16` (`0.43%`).

---

## 5. Harmed Families & Context `#27` Audit

- **Per-Family Harm Summary (`RERANK4 - CONTROL`)**:
  - **PRIMARY (`DL-Train`, `205` families)**: `19` families have median harm `< -0.01` (`23, 26, 27` by seed). Top-5 worst harmed families on `DL-Train` (`3`-fold-seed median):
    1. `c0a9f578dc73188c` (`1` ctx): `-0.391666`
    2. `0d7375e424cc7bf0` (`3` ctx): `-0.303693`
    3. `b7e90814c3c192eb` (`2` ctx): `-0.194840`
    4. `a24c28d6cb19f209` (`2` ctx): `-0.160516`
    5. `05bf4bc85c490ba8` (`2` ctx): `-0.150610`
  - **SECONDARY (`DL-Val91`, `19` families, `fold_seed = 266140`)**: `0` families have median harm `< -0.01` (`0, 0, 1` by seed); `4` families have small negative deltas (`f07b6deaa9393372`: `-0.003051`, `43a7767b56b30541`: `-0.001776`, `8c897157a497fb31`: `-0.001425`, `a325c6b0a5e9d9aa`: `-0.001387`).
- **Context `#27` (`sha256 = 71f2df6c4fb2f9c02d4deffca5eddeae3efcbcfbf8be79e90944ca1bc39ab780`, family `c5e3a4af253d445d`)**:
  - `r882` slot `0` utility = `0.9420064388407493`; `TOP4_PICK` oracle slot `3` utility = `2.0757762987447575`.
  - Under `RERANK4` (`fold_seed = 266140`, `15` cells): `12/15` cells pick slot `0` (`r882`), `2/15` pick slot `3` (`+1.1338` oracle gain), `0/15` pick invalid `-2.0` slot `1`. Under `RERANK4_M0` (`m=0`), `5/15` cells pick oracle slot `3` and `2/15` pick invalid slot `1`.

---

## 6. Disclosure of Post-Write Packaging Assert & Watcher Ruling (`hw_t1808_argon_h266_metricscap.md`)

1. **What Happened**: `attempt-001` (`17:39Z–17:57Z`) completed all `90` confirmatory cell fits (`PRIMARY` `DL-Train` OOF + `SECONDARY` `DL-Val91`) and all `75` descriptive ablation cell fits (`165` total fits), and `os.fsync`-wrote `baseline.json`, `fit_selections_lock.json`, `panels.npz`, `details.json`, and `metrics.json` (`218,715` bytes, `sha256 = 083edb422ce1d97ff5277ff5b3ddee8f0ebe39ce63df2e96591d8efe2ee5d160`) to `/home/ubuntu/jope_plan/evidence/H266/argon-t1730/runs/attempt-001/`. After writing `metrics.json`, line `2001` (`assert metrics_size <= 50_000`) raised `AssertionError: metrics.json size 218715 exceeds 50 KB cap!` because `worst_harmed_families_all` (`205` families $\times$ `5` ablations) and `worst_harmed_families_top10` across ~`65` sub-contrasts remained inside `metrics.json`.
2. **Compliance with Watcher Ruling (`hw_t1808_argon_h266_metricscap.md`)**:
   - **Zero recompute (`0` post-outcome retries)**: `attempt-001` is the single preregistered run for both `PRIMARY` and `SECONDARY`.
   - **Full S3 Custody**: The untouched `218,715`-byte `metrics.json` (`sha256 = 083edb422ce1d97ff5277ff5b3ddee8f0ebe39ce63df2e96591d8efe2ee5d160`, `VersionId = RYGkclV0aJZ4YvamtM9Yo8VM_KBAEGi6`), `details.json` (`VersionId = LvbWHQX6LOd9D2wl8yErvM.v4egl3Tuq`), `panels.npz` (`VersionId = Ua9dfJiAuXfw1l8_JymyUavWnr09GDC1`), and `fit_selections_lock.json` (`VersionId = 1ZFL4gTjf.6OT_9CD2fIZDa1sYyUTXQk`) were uploaded directly to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H266/attempt-001/`.
   - **Deterministic Compaction for Git**: The `48,380`-byte (`<= 50 KB`) Git `metrics.json` was derived deterministically from the untouched `218,715`-byte `metrics.json` and records `full_metrics_json_sha256 = 083edb422ce1d97ff5277ff5b3ddee8f0ebe39ce63df2e96591d8efe2ee5d160`.
   - **Packaging-Only `runner.py` Diff vs Preregistration Commit `6d662d708b859ac181ea745912727f38ef4b0a74`**:

```diff
diff --git a/research/flywheel_selfplay/h266_rerank4_confirm/runner.py b/runner.py
index 6e2e924c8..8ad8b790f 100644
--- a/research/flywheel_selfplay/h266_rerank4_confirm/runner.py
+++ b/runner.py
@@ -1869,12 +1869,87 @@ def run_primary(out_dir: Path) -> int:
     }
     details_sha256 = fsync_write_json(out_dir / "details.json", details_doc)

-    # Build compact metrics.json (guaranteed <= 50 KB for Git publication)
+    # Build compact metrics.json (guaranteed <= 50 KB for Git publication; full tables in details.json)
+    core_contrasts = (
+        "RERANK4 - CONTROL",
+        "RERANK4_SHUF - CONTROL",
+        "RERANK4 - RERANK4_SHUF",
+        "RERANK4_UNC - CONTROL",
+        "RERANK4_M0 - CONTROL",
+        "ORACLE_TOP4 - CONTROL",
+    )
+    core_arms = (
+        "CONTROL",
+        "RERANK4",
+        "RERANK4_SHUF",
+        "RERANK4_UNC",
+        "RERANK4_M0",
+        "ORACLE_TOP4",
+        "ORACLE_POOL",
+    )
```

---

## 7. Next Step Recommendation

Because:
1. `SECONDARY` (`DL-Val91`) replicated cleanly (`+0.012861` fam, `Holm p = 0.0016`, `3/3` seeds, `0` invalid `-2.0` gates),
2. `NO_SCOREGEOM` proved that non-score structural/context descriptors (`prog48`, `sem24`, `h36`, `ctx76`) drive the re-ranking signal (`+0.012210` `p = 0.0228` on `DL-Train` OOF; `+0.020092` `p = 0.0068` on `DL-Val91`), and
3. `PRIMARY` (`DL-Train` OOF) missed `LCB95 > 0` under the strict `H257` inner-`FIT` `LCB95` fallback (`+0.018938`, `LCB95 = -0.004329`) while clearing `LCB95 > 0` without that inner fallback (`RERANK4_UNC = +0.035589`, `95% CI [+0.005382, +0.079481]`, `p = 0.0105`, except for `55–68` `-2.0` invalid gate picks on `DL-Train`),

any follow-up top-`4` re-ranker wave should pair the `NO_SCOREGEOM` / full `560`-dim listwise ranker with an explicit **outcome-free `-2.0` invalid-gate classifier veto** trained on inner `FIT` so that margin `m*` does not have to default to `inf` (`CONTROL`) across entire folds just to avoid rare `-2.0` gate traps on `DL-Train`.
