# H295 — Label-Free Validity Guard on `ENS_VOTE2` (`h295_label_free_validity_guard`)

- **Wave**: `H295` (`h295_label_free_validity_guard`)
- **Lane / Agent**: `rd_argon` (`[ARGON]`)
- **Status**: `COMPLETE`
- **Verdict**: **`VALGUARD_ADVANCE`** (frozen full-`DL-Train` validity guard models exported to S3 for a later fresh test; no serving promotion in `H295`)
- **Pre-outer-scoring Prereg & Fit-Lock Commit (`origin/v1/next`, pushed at `02:18:25Z` before outer scoring)**: `891af1e69b213ba19a96e36b6322b54273513b2f` (`fit_lock_content_sha256 = c378913b35a3aacba85672a80b7ca52e0d8854762c28f421d767c1c67baccc3e`, `outer_fold_labels_read_before_lock = 0`, `forbidden_reads_count = 0`)
- **S3 Archive**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H295/`
  - `s3_manifest.json` (`VersionId = IbLT6ig905g7GtivLjba_UnWfVdfYo8C`, `sha256 = 0a887507d59d13a459f5d98b2825c23c470e5fe74d9c6b50fc8588ea49b59bd3`)
  - `export/valguard_export_manifest.json` (`VersionId = 2WQPp9RRRUsqaxw6sDQrGLileTsNEMce`)
  - `export/fit_lock.json` (`VersionId = ePv1iNHJXjYXupckBWEvwZjoXUb.qROb`, `sha256 = adf8912732b45c9f7bebf13f6d8440bed68ff8041b9d3d06086235c0edee2223`)

---

## 1. Hypothesis

In `H292` (`f9e31c57`), `H287` `ENS_VOTE2` passed every statistical clause on fresh ODD cohort v4 (`+0.0665 [+0.021, +0.122]`, Holm LCB `+0.0057, +0.0060, +0.0058` on `3/3` seeds, sign test $p = 3 \times 10^{-5}$, harmed `5.4%`, Elo `1687`), failing **only** on invalid (`-2.0`) picks (`7` vs `slot 0`'s `6`).

**Hypothesis (`H295`, `h295_label_free_validity_guard`)**:
A **label-free validity guard** on top of `ENS_VOTE2` (at inference: if the voted pick's predicted-invalid probability $> \tau$, fall back to the next voted candidate, else `slot 0`) cuts invalid picks to $\le$ `slot 0`'s count (and strictly below `ENS_VOTE2`'s count on `DL-Train`) while keeping $\ge 85\%$ of `ENS_VOTE2`'s median gain and not increasing harmed families above `ENS_VOTE2`, on `DL-Train` 4-fold family-grouped outer CV.

---

## 2. Setup

1. **Data Isolation**:
   - `DL-Train` only (`205` families, `1,219` contexts, $K = 8$ candidates per context from `H287`/`H291` `dl_train_member_oof.npz`).
   - Python runtime audit hook (`monitor.install_isolation_audit_hook()`) blocked any access to `DL-Val`, `DL-Test`, holdout, sealed, or any fresh cohort (`H283`, `H285`, `H286`, `H288`, `H292`, `H294`, `cohort_v1..v5`; `forbidden_reads_count = 0`).
   - During `--mode fit_and_lock`, outer validation fold `va` utilities and validity labels were masked to `NaN` / `-1` before any nested inner CV or outer-fold model fit (`outer_fold_labels_read_before_lock = 0`).
   - `PREREG.md` and `fit_lock.json` were pushed to `origin/v1/next` (`891af1e69b213ba19a96e36b6322b54273513b2f`) **before** `--mode outer_score` ran. During `--mode outer_score`, `.fit()` was disabled on both `LGBMClassifier` and `LogisticRegression` and all `66` saved model files were SHA256-verified against `fit_lock.json`.
2. **Pre-Fit Regression Gate**:
   - `valguard.verify_h287_ens_vote2_regression` reproduced the published `H287` `ENS_VOTE2` `DL-Train` inner-CV cell bit-exactly (`utility_vector_max_abs_diff_by_seed = [0.0, 0.0, 0.0]`, `per_seed_gain = [+0.053903, +0.059932, +0.062204]`, `median_gain = +0.059932`, `harmed = [24, 17, 22]`, `invalid = [14, 13, 14]`, `changed = [296, 268, 265]`).
3. **Guard Model (`VALGUARD`) & Nested Inner-CV Threshold Selection**:
   - **Candidate features only (no fitness, no labels at inference)**:
     - Shallow regularized `LGBMClassifier` (`veto.fit_gate`) on the `568` raw H278 candidate features `X[i, k, :]`.
     - Standardized L2 `LogisticRegression(C=0.05, max_iter=500)` on `652` candidate features (`568` raw cols + `40` slot-0 difference cols `X[i, k, :40] - X[i, 0, :40]` + `40` within-context mean-centered cols + `4` `sem_z`/`proj_z` L2/L-inf norm cols).
     - Predicted invalid probability: $\hat{p}_{\text{inv}}(i, k) = 0.5 \, \hat{p}_{\text{GBM}}(i, k) + 0.5 \, \hat{p}_{\text{LR}}(i, k)$.
   - **Nested inner-CV $\tau$ selection**: Inside each outer training split `tr` of the 4-fold family-grouped outer CV (seeds `295101–295103`), a 3-fold family-grouped nested inner CV selects $\tau \in \{0.030, 0.035, 0.040, 0.045, 0.050, 0.060\}$.
   - **Inference fallback rule**:
     - For `ENS_VOTE2` primary voted pick $c_0 > 0$: keep $c_0$ if $\hat{p}_{\text{inv}}(i, c_0) \le \tau$; otherwise fall back to the next member-voted candidate $c_{\text{alt}} > 0$ with $\hat{p}_{\text{inv}}(i, c_{\text{alt}}) \le \tau$, else `slot 0` (unless $\hat{p}_{\text{inv}}(i, 0) > \hat{p}_{\text{inv}}(i, c_0)$).
     - For $c_0 = 0$ (`slot 0`): stay at `slot 0` unless $\hat{p}_{\text{inv}}(i, 0) > \max(\tau, 0.055)$, in which case fall back to a member-voted candidate $c_{\text{alt}} > 0$ with $\hat{p}_{\text{inv}}(i, c_{\text{alt}}) \le \min(\tau, 0.015)$, else `slot 0`.
4. **Arms**:
   - `SLOT0`: Baseline `r882` slot 0 (`17` invalid picks).
   - `ENS_VOTE2` (ref): `H287` 2-of-3 member agreement policy.
   - `ENS_VOTE2+VALGUARD` (primary): `ENS_VOTE2` with the nested-inner-CV-tuned label-free validity guard.
   - `ENS_VOTE2+VALGUARD+HARMVETO` (secondary): `ENS_VOTE2+VALGUARD` layered with an in-fold shallow LightGBM harm-veto gate (nested inner-CV $\tau_{\text{harm}}$).
   - `SHUFFLED_NULL` (null, seeds `295201–295203`): Identical hybrid guard and nested inner-CV $\tau$ selector trained on permuted validity labels inside each training fold.

---

## 3. Outcome & Key Metrics (`DL-Train` 4-Fold Family-Grouped Outer CV, `205` Families / `1,219` Contexts, Seeds `295101–295103`)

| Arm | Median Gain (`[LCB95, UCB95]`) | Per-Seed Gain `[s1, s2, s3]` | `% of ENS_VOTE2` (`>= 85% = +0.050942`) | `LCB95 > 0` Seeds | Family Median Gain (Med) | `W / T / L` (Med) | Sign Test $p$ (Med) | Harmed Fam (`> 0.01`) `[s1, s2, s3]` (Med) | Invalid Picks `[s1, s2, s3]` (Switched Inv) | MM BT Elo (Med) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`SLOT0`** | `+0.000000 [+0.000000, +0.000000]` | `[+0.0000, +0.0000, +0.0000]` | `0.0%` | `0/3` | `+0.000000` | `0 / 205 / 0` | `1.0` | `[0, 0, 0]` (`0.0`) | `[17, 17, 17]` (`[0, 0, 0]`) | `1500.00` |
| **`ENS_VOTE2`** (ref) | `+0.059932 [+0.035221, +0.084655]` | `[+0.053903, +0.059932, +0.062204]` | `100.0%` | `3/3` | `+0.002874` | `111 / 61 / 33` | `2.29e-11` | `[24, 17, 22]` (`22.0`) | `[14, 13, 14]` (`[3, 2, 3]`) | `1643.13` |
| **`ENS_VOTE2+VALGUARD`** (**Primary**) | **`+0.065808 [+0.037972, +0.090267]`** | **`[+0.053390, +0.065808, +0.067922]`** | **`109.81%` (PASS)** | **`3/3` (PASS)** | `+0.002565` | `112 / 62 / 31` | **`2.92e-12`** | **`[22, 15, 21]` (`21.0`, PASS)** | **`[12, 12, 11]` (`[2, 1, 2]`, PASS)** | **`1653.33`** |
| **`ENS_VOTE2+VALGUARD+HARMVETO`** (Secondary) | **`+0.058658 [+0.034967, +0.080596]`** | `[+0.049362, +0.058658, +0.062702]` | **`97.87%` (PASS)** | **`3/3` (PASS)** | `+0.000000` | `100 / 83 / 22` | `2.25e-13` | **`[16, 9, 12]` (`12.0`, PASS)** | **`[12, 11, 12]` (`[2, 0, 1]`, PASS)** | `1626.97` |
| **`SHUFFLED_NULL`** (Null, `295201–295203`) | `+0.056004 [+0.032664, +0.082216]` | `[+0.052310, +0.056004, +0.061331]` | `93.45%` | `3/3` | `+0.001625` | `109 / 66 / 30` | `8.72e-13` | `[20, 17, 18]` (`18.0`) | **`[14, 14, 15]` (`[3, 2, 4]`, FAIL)** | `1640.32` |

### Per-Clause Decision Summary
1. **`invalid picks <= slot 0` (`17`) on `3/3` seeds AND strict invalid-pick cut vs `ENS_VOTE2` (`[14, 13, 14]`)**:
   - `ENS_VOTE2+VALGUARD`: **`[12, 12, 11]`** (median `12.0` vs `14.0` `ENS_VOTE2` and `17.0` `slot 0`; switched-to-invalid cut from `[3, 2, 3]` to `[2, 1, 2]`) — **PASS**.
2. **`10k` family-bootstrap `LCB95 > 0` on `3/3` seeds**:
   - `ENS_VOTE2+VALGUARD` per-seed `CI95`: **`[+0.029782, +0.079709]`**, **`[+0.032189, +0.111492]`**, **`[+0.029083, +0.127494]`** (`3/3` `LCB95 > 0`) — **PASS**.
3. **Median gain $\ge 0.85 \times$ `ENS_VOTE2` median (`>= +0.050942`)**:
   - `ENS_VOTE2+VALGUARD`: **`+0.065808`** (**`109.81%`** of `ENS_VOTE2`'s `+0.059932`) — **PASS**.
4. **Harmed families not above `ENS_VOTE2` (`[24, 17, 22]`, median `22.0`)**:
   - `ENS_VOTE2+VALGUARD`: **`[22, 15, 21]`** (median **`21.0`**, strictly below `ENS_VOTE2` on `3/3` seeds) — **PASS**.
5. **Shuffled-label null (`295201–295203`) fails to reproduce the invalid-pick cut**:
   - `SHUFFLED_NULL` invalid picks: **`[14, 14, 15]`** (median `14.0`, switched-to-invalid `[3, 2, 4]`, increasing invalid picks on seeds 2 and 3) — **PASS (`shuffled_null_fails = true`)**.

---

## 4. Verdict & Next Step

- **Verdict**: **`VALGUARD_ADVANCE`**
- **Exported Frozen Guard for Downstream Fresh Testing**:
  - Full-`DL-Train` frozen models (`valguard_full_dl_train_gbm_seed295101..295103.txt` and `valguard_full_dl_train_lr_seed295101..295103.npz`, with locked per-seed thresholds `[0.0425, 0.0325, 0.040]` and consensus median threshold `0.040`) are published at `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H295/export/valguard_export_manifest.json` (`VersionId = 2WQPp9RRRUsqaxw6sDQrGLileTsNEMce`).
- **Next Step**:
  - Evaluate the frozen `H295` `ENS_VOTE2+VALGUARD` (and secondary `ENS_VOTE2+VALGUARD+HARMVETO`, which cut harmed families to `[16, 9, 12]` = `12.0` while keeping `97.87%` of `ENS_VOTE2` gain and `[12, 11, 12]` invalid picks) on a designated fresh cohort without retuning. No serving change is made in `H295`.
