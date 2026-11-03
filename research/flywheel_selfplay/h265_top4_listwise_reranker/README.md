# H265 (`h265_top4_listwise_reranker`) — Zero-Update Listwise LambdaMART Top-4 Re-Ranker (`0` Test-Time Reveals, Single-Pick)

- **Wave ID**: `h265-top4-listwise-reranker-20261103`
- **Owner**: Argon (`xbabe2` CPU lane `taskset -c 57-60`, `nice -n 19`, `threads <= 2`, `0` GPU)
- **Status**: `COMPLETE`
- **Verdict**: **`ADVANCE`** (on primary `DL-Val` `Native91` **single-pick (`0` test-time reveals)** equal-family utility contrast `RERANK4 - CONTROL` with `Holm m = 2` over `RERANK4` and `RERANK4_SHUF`)
- **Preregistration Commit (`origin/v1/next`)**: `11d8b1f1895f823476a7d9a4f6b9099518cf835d` (bound in `manifest.json` at `ad5ff5c9cd25817126fca7c1d0253d1bf155ddfd` with `0` outcomes read)
- **S3 Preregistration Snapshot**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H265/prereg-11d8b1f1895f/prereg_s3_manifest.json` (`VersionId = JwX9Xt8.yDtbUTrtY3H_GhFg7gS1j1.H`, SHA256 `5a7544d4e8b2e3096275081fbf7a47783747afdb92249d87b225eec56259b868`)
- **S3 Attempt-002 Manifest**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H265/attempt-002/s3_manifest.json` (`VersionId = 1LEMA3bIwPFe4tyT7tWB_l9jltEhlo7F`, SHA256 `b1a866b841f5f17a14ae38ce62f4a7deedcafaf5a25eb92daa51557d7af064ad`)

---

## 1. Scientific Question & Claim Scope

In `H258` (`h258_top4_reveal_baseline`), `r882`'s top-4 shortlist (`TOP4_PICK` with `4` test-time reveals on `top_zgap < 0.25`) reached `1.2403601682250796` context mean / `1.2519997168684853` family mean on `DL-Val` (`Native91`, `91` contexts / `19` families) versus `r882` single-pick (`0` reveals) `1.1377941898164525` context mean / `1.153615276991245` family mean (`+0.098384` family ceiling). Moreover, within `r882`'s own `top-4`, `r882`'s rank-1 pick (`slot 0`) is the true best of the 4 only **`23.08%`** (`21/91`) of the time on `DL-Val` and **`27.32%`** (`333/1,219`) of the time on `DL-Train`.

**H265 Question**: Can a **zero-update (`0` optimizer steps on `r882` or DOPE)** `FIT`-only listwise LambdaMART re-ranker (`lightgbm.LGBMRanker(objective="lambdarank")`) trained **exclusively on `r882`'s `top-4`** (`group = [4, 4, ...]`) using `560` frozen outcome-free features and **NO test-time reveals (`0` test-time reveals, scored SINGLE-PICK)** significantly improve single-pick utility over `r882` (`CONTROL`) and over a within-family label-shuffled placebo (`RERANK4_SHUF`)?

---

## 2. Baseline Regression & Hygiene Verification

Executed first before any model training or selection:
- **`DOPE` Backbone SHA256**: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (unchanged before/after; `0` optimizer updates).
- **`r882` Checkpoint SHA256**: `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` (unchanged before/after; `0` optimizer updates).
- **`CONTROL` (`r882` Single-Pick `Native91`)**: `1.1377941898164525` (`abs_diff = 0.0`, **`91/91` bit-exact context matches** against `best_dev_checkpoint.json`), equal-family mean `1.153615276991245`.
- **`TOP4_PICK@r882` Oracle 4-Reveal Ceiling**: `1.2403601682250796` context mean (`abs_diff = 0.0`), `1.2519997168684853` equal-family mean (`abs_diff = 0.0`).
- **Global FIT Selection Lock (`fit_selections_lock.json`)**: All `15` cell selections (`5` folds $\times$ `3` seeds `265101, 265102, 265103`) for `RERANK4` and `RERANK4_SHUF` were selected via 4-fold inner cross-validation on the `4` `DL-Train` `FIT` folds (`164` families) and `os.fsync`-locked to `fit_selections_lock.json` (`sha256 = a8505f63a9a8ba58b65b6f1daccaf17681a8be665500ddee0644eca4c24a6566`) with `dl_val_feature_or_pool_reads_during_fit = 0` before any `DL-Val` feature/pool read for model scoring.

---

## 3. Primary & Secondary Results on `DL-Val` (`Native91`, `91` ctx / `19` fam, Single-Pick, `0` Reveals)

### 3.1 `DL-Val` Arm Utilities & Choice Diagnostics

| Arm | Description | Test Reveals | Median Context Mean | Median Family Mean | Per-Seed Family Means (`265101 / 265102 / 265103`) | Top-1-in-Top-4 Acc (`3` seeds) | Mean Changed Choices `/ 91` (`3` seeds) | Invalid (`-2.0`) Picks `/ 455` (`3` seeds) | `400`-Iter MM Elo (`[95% CI]`) |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `CONTROL` (`r882`) | Frozen `r882` slot-0 single pick | `0` | `1.137794190` | `1.153615277` | `1.153615 / 1.153615 / 1.153615` | `23.08% / 23.08% / 23.08%` | `0.0 / 0.0 / 0.0` | `0 / 0 / 0` | `1500.00 [1500.00, 1500.00]` |
| **`RERANK4` (PRIMARY)** | Strict `H257`-selected `LGBMRanker` (`11/15` cells admitted) | **`0`** | **`1.157278226`** | **`1.172300641`** | **`1.172301 / 1.172313 / 1.163206`** | **`27.91% / 28.57% / 25.93%`** | `30.2 / 35.8 / 23.8` | **`0 / 0 / 0`** | **`1536.53 [1502.13, 1568.99]`** |
| **`RERANK4_SHUF` (SECONDARY)** | Within-family label-shuffled placebo (`1/15` cells admitted) | `0` | `1.137794190` | `1.153615277` | `1.153615 / 1.153615 / 1.156218` | `23.08% / 23.08% / 22.86%` | `0.0 / 0.0 / 2.0` | `0 / 0 / 0` | `1500.00 [1493.90, 1500.67]` |
| `RERANK4_UNC` (Diagnostic) | Unconstrained inner-FIT-best `(lr*, n_est*, m*)` | `0` | `1.161840954` | `1.176671255` | `1.176671 / 1.176789 / 1.169901` | `29.67% / 29.01% / 28.13%` | `40.4 / 41.0 / 38.6` | `0 / 0 / 0` | `1555.50 [1506.76, 1606.18]` |
| `RERANK4_SHUF_UNC` (Diagnostic) | Unconstrained inner-FIT-best placebo | `0` | `1.137551885` | `1.153383174` | `1.142516 / 1.153383 / 1.156108` | `20.66% / 22.42% / 25.93%` | `25.6 / 8.2 / 22.2` | `1 / 0 / 0` | `1490.34 [1480.91, 1541.99]` |
| `RERANK4_M0` (Diagnostic) | Fixed `lr=0.05, n_est=100, m=0.0` (pure argmax) | `0` | `1.159108682` | `1.174108838` | `1.184832 / 1.174109 / 1.171342` | **`34.95% / 34.07% / 34.29%`** | `62.2 / 61.8 / 60.8` | `0 / 0 / 0` | `1601.12 [1527.37, 1653.70]` |
| `RERANK4_SHUF_M0` (Diagnostic) | Fixed `lr=0.05, n_est=100, m=0.0` placebo | `0` | `1.126163807` | `1.142532220` | `1.143706 / 1.142532 / 1.119157` | `26.81% / 26.37% / 27.91%` | `65.4 / 61.8 / 64.8` | `1 / 2 / 4` | `1511.01 [1427.91, 1593.62]` |
| `TOP4_PICK_ORACLE` (Ceiling) | 4-reveal oracle ceiling inside `r882` top-4 | `4` | `1.240360168` | `1.251999717` | `1.252000 / 1.252000 / 1.252000` | `100.0%` | `70.0 / 70.0 / 70.0` | `0 / 0 / 0` | `1988.06 [1906.87, 2073.94]` |

### 3.2 `DL-Val` Paired Family Bootstrap Contrasts (`10,000` draws, `boot_seed = 265150`, `Holm m = 2`)

| Contrast (`DL-Val` `Native91`) | Median Family Gain | Median Context Gain | `95% CI` (Family) | Holm `97.5% CI` (`m=2`) | One-Sided `raw p` | `Holm p` (`m=2`) | Positive Seeds | Family W/T/L (`265101 / 265102 / 265103`) | Harmed Fams `Δ < -1e-4` (`< -0.01`) |
| --- | ---: | ---: | --- | --- | ---: | ---: | :---: | --- | :---: |
| **`RERANK4 - CONTROL` (PRIMARY)** | **`+0.018685`** | **`+0.019484`** | **`[+0.006369, +0.030822]`** | **`[+0.005100, +0.033073]`** | **`0.000100`** | **`0.000200`** | **`3 / 3`** | `16/0/3`, `15/1/3`, `13/1/5` | `3, 3, 5` (`1, 2, 2`) |
| **`RERANK4_SHUF - CONTROL` (SECONDARY)** | **`+0.000000`** | **`+0.000000`** | `[+0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `1.000000` | `1.000000` | `1 / 3` | `0/19/0`, `0/19/0`, `4/13/2` | `0, 0, 2` (`0, 0, 0`) |
| **`RERANK4 - RERANK4_SHUF` (Placebo Guard)** | **`+0.018685`** | **`+0.019484`** | **`[+0.006360, +0.030822]`** | **`[+0.005044, +0.033073]`** | **`0.000200`** | — | **`3 / 3`** | `16/0/3`, `15/1/3`, `12/1/6` | `3, 3, 6` (`1, 2, 2`) |
| `RERANK4_UNC - CONTROL` (Unconstrained) | `+0.023056` | `+0.024047` | `[+0.008356, +0.038581]` | `[+0.006813, +0.040972]` | `0.000100` | — | `3 / 3` | `15/0/4`, `15/1/3`, `14/0/5` | `4, 3, 5` (`1, 1, 3`) |
| `RERANK4_UNC - RERANK4_SHUF_UNC` | `+0.023405` | `+0.024434` | `[+0.009815, +0.043510]` | `[+0.008552, +0.047513]` | `0.000100` | — | `3 / 3` | `15/0/4`, `15/1/3`, `12/0/7` | `4, 3, 7` (`0, 1, 3`) |
| `RERANK4_M0 - CONTROL` (`m=0` fixed) | `+0.020494` | `+0.021314` | `[+0.004814, +0.039940]` | `[+0.002750, +0.042901]` | `0.004300` | — | `3 / 3` | `16/0/3`, `15/0/4`, `13/1/5` | `3, 4, 5` (`0, 3, 3`) |
| `RERANK4_M0 - RERANK4_SHUF_M0` | `+0.031577` | `+0.032945` | `[+0.004354, +0.068139]` | `[+0.000270, +0.073688]` | `0.011799` | — | `3 / 3` | `15/0/4`, `13/0/6`, `15/0/4` | `4, 6, 4` (`2, 3, 2`) |
| `RERANK4_SHUF_UNC - CONTROL` | `-0.000232` | `-0.000242` | `[-0.010926, +0.006687]` | `[-0.012329, +0.007810]` | `0.615338` | — | `1 / 3` | `6/1/12`, `6/5/8`, `12/1/6` | `12, 8, 6` (`8, 1, 2`) |
| `RERANK4_SHUF_M0 - CONTROL` | `-0.011083` | `-0.011630` | `[-0.040581, +0.010365]` | `[-0.045010, +0.013291]` | `0.857114` | — | `0 / 3` | `9/0/10`, `9/0/10`, `7/0/12` | `10, 10, 12` (`5, 8, 8`) |
| `TOP4_PICK_ORACLE - CONTROL` (4-Reveal Ceiling) | `+0.098384` | `+0.102566` | `[+0.059056, +0.148119]` | `[+0.055099, +0.157268]` | `0.000100` | — | `3 / 3` | `19/0/0`, `19/0/0`, `19/0/0` | `0, 0, 0` (`0, 0, 0`) |

### 3.3 All Preregistered `ADVANCE` Gates Verified

1. **`median_family_gain(RERANK4 - CONTROL) > +0.002 + 1e-12`**: **`+0.018685364 > +0.002`** (`PASS`, recovers `19.0%` of the entire `TOP4_PICK` 4-reveal oracle gap `+0.098384` with **`0` test-time reveals**!).
2. **`Holm-adjusted LCB95 (m=2, 97.5% LCB) > 0`**: **`+0.005100 > 0`** (unadjusted `95% LCB = +0.006369 > 0`, `PASS`).
3. **`Holm-adjusted p-value (m=2) < 0.05`**: **`Holm p = 0.000200 < 0.05`** (`raw p = 0.000100`, `PASS`).
4. **`3/3` seeds positive**: **`+0.018685`, `+0.018698`, `+0.009591`** (`3/3` positive, `PASS`).
5. **`RERANK4` beats `RERANK4_SHUF` placebo**: **`RERANK4 - RERANK4_SHUF = +0.018685`** (`95% CI [+0.006360, +0.030822]`, `raw p = 0.000200`, `3/3` seeds positive; `PASS`).
6. **Non-trivial admission (`strict_fallback_identity == False`)**: `RERANK4` was admitted by the strict inner-FIT selector (`inner_fit_gain > +0.001 + 1e-12` and `inner_fit_lcb95 > 0`) on **`11 / 15` cells** vs only **`1 / 15` cells** for `RERANK4_SHUF`.

---

## 4. Descriptive Out-of-Fold (`OOF`) Results on `DL-Train` (`1,219` ctx / `205` fam, Single-Pick, `0` Reveals)

Unlike `H261` (where `DL-Train` OOF gains failed to transfer to `DL-Val`), `H265` shows **consistent positive gains on both `DL-Train` OOF and `DL-Val`**, and negative gains for the shuffled placebo on both splits:

| Contrast (`DL-Train` OOF, `1,219` ctx / `205` fam) | Median Family Gain | Median Context Gain | `95% CI` (Family) | One-Sided `raw p` | Positive Seeds | Per-Seed Family Gains (`265101 / 265102 / 265103`) | Family W/T/L (`265101 / 265102 / 265103`) |
| --- | ---: | ---: | --- | ---: | :---: | --- | --- |
| **`RERANK4 - CONTROL`** | **`+0.023578`** | **`+0.028540`** | `[-0.005946, +0.061002]` | `0.062294` | **`3 / 3`** | `+0.023578 / +0.032104 / +0.016850` | `95/59/51`, `100/50/55`, `63/99/43` |
| **`RERANK4 - RERANK4_SHUF`** | **`+0.023578`** | **`+0.028540`** | `[-0.005817, +0.061244]` | `0.059094` | **`3 / 3`** | `+0.023578 / +0.032104 / +0.017613` | `95/59/51`, `100/50/55`, `59/98/48` |
| **`RERANK4_UNC - CONTROL`** | **`+0.039468`** | **`+0.036634`** | **`[+0.004005, +0.080631]`** | **`0.014299`** | **`3 / 3`** | `+0.030203 / +0.047757 / +0.039468` | `117/24/64`, `129/10/66`, `112/18/75` |
| **`RERANK4_UNC - RERANK4_SHUF_UNC`** | **`+0.057744`** | **`+0.054027`** | **`[+0.018298, +0.090866]`** | **`0.000900`** | **`3 / 3`** | `+0.026926 / +0.057744 / +0.057947` | `125/8/72`, `134/9/62`, `122/12/71` |
| `RERANK4_M0 - CONTROL` | `+0.022058` | `+0.019018` | `[-0.018359, +0.062760]` | `0.145985` | `3 / 3` | `+0.026065 / +0.014565 / +0.022058` | `127/6/72`, `129/6/70`, `125/8/72` |
| `RERANK4_M0 - RERANK4_SHUF_M0` | `+0.054824` | `+0.062350` | `[+0.015156, +0.094191]` | `0.005599` | `3 / 3` | `+0.066172 / +0.054824 / +0.048964` | `148/1/56`, `132/3/70`, `136/2/67` |
| `RERANK4_SHUF - CONTROL` | `+0.000000` | `+0.000000` | `[+0.000000, +0.000000]` | `1.000000` | `0 / 3` | `0.000000 / 0.000000 / -0.000763` | `0/205/0`, `0/205/0`, `17/177/11` |
| `RERANK4_SHUF_UNC - CONTROL` | `-0.009987` | `-0.016704` | `[-0.024400, +0.005250]` | `0.900910` | `1 / 3` | `+0.003277 / -0.009987 / -0.018479` | `65/61/79`, `33/127/45`, `47/94/64` |
| `RERANK4_SHUF_M0 - CONTROL` | `-0.040107` | `-0.051722` | `[-0.086498, +0.028868]` | `0.888111` | `0 / 3` | `-0.040107 / -0.040259 / -0.026906` | `78/4/123`, `90/4/111`, `89/6/110` |
| `TOP4_PICK_ORACLE - CONTROL` | `+0.193361` | `+0.190992` | `[+0.139247, +0.268021]` | `0.000100` | `3 / 3` | `+0.193361 / +0.193361 / +0.193361` | `198/7/0`, `198/7/0`, `198/7/0` |

On `DL-Train` OOF (`1,219` contexts):
- Top-1-within-top-4 accuracy rises from **`27.32%`** (`CONTROL` `r882`, `333/1,219`) to **`33.47% / 33.88% / 30.84%`** under `RERANK4` (`35.27% / 34.45% / 35.11%` under `RERANK4_M0`), while `RERANK4_SHUF_M0` drops to **`25.10% / 26.91% / 26.91%`**.
- `400`-iter MM Bradley-Terry Elo on `DL-Train` OOF: `CONTROL = 1500.00`, `RERANK4 = 1525.88 [1512.79, 1532.73]`, `RERANK4_UNC = 1534.15 [1518.87, 1542.96]`, `RERANK4_M0 = 1538.34 [1520.93, 1550.92]`, `RERANK4_SHUF_M0 = 1487.23 [1473.89, 1502.09]`.

---

## 5. Mechanism, Feature Importance Audit, Harmed Families & Context `#27`

### 5.1 Why `RERANK4` Succeeds Where Full-Pool Re-Training (`H260`) and Diversity Search (`H261`) Did Not

1. **Restricted Action Space (`top-4` Only)**: Full-pool adapter updates (`1,152` candidates/context) suffer from ranking drift across the 1,148 tail candidates (`H260`), whereas `RERANK4` never promotes a tail candidate (`ranks 5..1152`) and therefore inherits `r882`'s `0` invalid (`-2.0`) gate rate on `DL-Val` (`0/455` invalid picks across all `15` cells).
2. **Correction of `r882`'s Systematic Slot-0 Bias**: Within `r882`'s `top-4`, `r882` picks the true best of the 4 only `23.08%` of the time on `DL-Val`. Auditing the fitted `LGBMRanker` gain shares (`seed 265101, fold 0`) shows why:
   - `sem24_z_24` (standardized 24D structural semantic program signature): **`28.79%` of total gain** (`#1` feature `sem24_z[20]`, `#6` `sem24_z[13]`, `#9` `sem24_z[4]`, `#10` `sem24_z[21]`).
   - `score_geom_32` (`r882` pairwise score diffs, `aprior` residual, and within-top-4 / top-8 cosine similarity): **`28.27%` of total gain** (`#2` `cos_emb40_top8_mean`, `#3` `aprior_raw`, `#4` `diff_to_slot0`, `#5` `diff_to_slot3`, `#7` `cos_emb40_to_slot0`, `#8` `cos_proj16_to_slot0`).
   - `prog_412` (412D program descriptors): **`21.76%` of total gain** (`#17` `prog_412[108]`, `#20` `prog_412[196]`).
   - `lm48_z_48` (48D context landmark meta-features): **`14.35%` of total gain**.
   - `proj16_z_16` (`4.08%`) + `dope_28` (`2.75%`): **`6.83%` of total gain**.

### 5.2 Harmed Families on `DL-Val` (`RERANK4 - CONTROL`)

Across the `19` `DL-Val` families (median across seeds), only `2` families have median family delta `< -1e-4` (`1` family `< -0.01`):
- `fam-469b72b7c8b4fe39c95e136c0ba59f4264e5eb1ed572d53f9e5f131e9c21c279`: `Δ = -0.020687`
- `fam-769243ee920953f082bd6784c94795f919abe67c243fae912657abc080863c0c`: `Δ = -0.004228`

### 5.3 Context `#27` (`43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`)

- Family: `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f`
- Within `r882`'s `top-4` on Context `#27`, `slot 0` (`r882` pick) has valid utility `1.122695`, `slot 3` (`TOP4_PICK_ORACLE`) has valid utility `1.128496` (`+0.005801`), while **`slot 1` and `slot 2` fail the gate (`utility = -2.0`)**!
- **`RERANK4` (`15/15` cells)**: avoids the two `-2.0` gate-failing traps (`slots 1, 2`) in **`15/15` cells** (`0` invalid picks), selecting `slot 0` (`1.122695`) in `14/15` cells and `slot 3` (`1.128496`, the oracle best!) in `1/15` cells (`seed 265103, fold 4`; and `4/15` cells under `RERANK4_M0`, also with `0` invalid picks).
- **`RERANK4_SHUF_M0` (Placebo)**: hits the `-2.0` invalid gate traps (`slot 1` or `slot 2`) on **`5/15` cells** of Context `#27`!

---

## 6. Post-Preregistration Runner Disclosure & Resource Summary

- **Pre-Fit Pool Loader Fix (`attempt-001` $\to$ `attempt-002`)**: `attempt-001` exited after `10s` at the first pool file in `load_cohort_top4_packets` (`KeyError: 'sem24 is not a file in the archive'`) before any model fit or inner CV selection. In `runner.py`, `z["sem24"]` was replaced by the deterministic helper `compute_sem24_numpy(prog)` matching `common_h252.py` (computing the exact 24D structural semantic signature specified in `PREREG.md`). No model, feature definition, hyperparameter grid, selector rule, seed, or scoring logic was modified.
- **Resource Envelope (`attempt-002` under `monitor.py`)**:
  - Wall-clock time: `254.07s` (`4.23 min`, well before the `19:45Z` compute stop).
  - Peak RSS: `861,552,640` bytes (`0.802 GiB`, limit `<= 4.0 GiB`).
  - Peak threads: `2` (`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`, limit `<= 4`).
  - CPU affinity: cores `[57, 58, 59, 60]`, `nice = 19`, `0` GPU processes.

## 7. Reproduction Commands

```bash
cd research/flywheel_selfplay/h265_top4_listwise_reranker
pytest -q test_h265.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 57-60 nice -n 19 python3 monitor.py \
  --out-dir /home/ubuntu/jope_plan/evidence/H265/argon-t1658/runs/attempt-002 \
  --run-id attempt-002 \
  -- python3 runner.py --mode primary --out-dir /home/ubuntu/jope_plan/evidence/H265/argon-t1658/runs/attempt-002
```
