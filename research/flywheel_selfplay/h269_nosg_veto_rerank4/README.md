# H269 (`h269_nosg_veto_rerank4`) — Fresh-Fold Confirmation of `H266` `NO_SCOREGEOM` (`528` Cols) + Outcome-Free `LGBMClassifier` `-2.0` Invalid-Gate Veto

- **Wave ID**: `h269-nosg-veto-rerank4-20261103` (`H269`)
- **Owner**: `Argon`
- **Status**: `COMPLETED`
- **Verdict**: **`NO_ADVANCE`** (`winning_arm = null`)
- **Preregistration commit (`origin/v1/next`)**: `ff9c177a6ddbfa18d644a46089f2792e6dc19609` (`outcomes_read = 0`)
- **S3 preregistration snapshot**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H269/prereg-ff9c177a6ddb/prereg_s3_manifest.json` (`VersionId = J5QjR19h0fAZ5lTLgi8IVb3szLVLFZT.`)
- **S3 primary evidence bundle (`attempt-001`)**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H269/attempt-001/s3_manifest.json` (`VersionId = .XPgSnC.y.fPK1SqCMs6L3gfFXKU9iI2`)
- **Zero post-prereg code/protocol edits**: `runner.py`, `PREREG.md`, `fold_manifest.json`, `monitor.py`, and `test_h269.py` are byte-identical to preregistration commit `ff9c177a6ddbfa18d644a46089f2792e6dc19609`. Full outputs (`metrics_full.json` `62,910 B`, `VersionId = FDeKBSZUh1jhQQkpsEEtYt280ilk8Ul1`; `details.json` `406,827 B`, `VersionId = hy3ayOqeLTC1CZqJJ6LtIAyde6LJdeEa`; `panels.npz` `384,223 B`, `VersionId = Gc9eN7aA9bQEDR5gDrwzjSBuKychN9tA`; `fit_selections_lock.json` `125,064 B`, `VersionId = MJq1YH.FdweFSHUSs_d0ssYtuKHjINCE`) were uploaded to S3 first, and the `<= 50 KB` Git `metrics.json` (`50,922 B`, `VersionId = pfINOavnvqHwZ2XjRzvhTSucVZwwQBl2`, `100%` lossless JSON round-trip) was derived as a packaging step.

---

## 1. Executive Summary & Decision Gate Audit

`H269` tested whether the `H266` `NO_SCOREGEOM` (`528`-column `LGBMRanker` dropping all `32` `r882` scalar score/gap/cosine columns `0..31`) and `NOSG_VETO` (`NOSG` plus an outcome-free inner-`FIT` `LGBMClassifier` with `class_weight = {0: 1.0, 1: 10.0}` vetoing any non-`r882` pick `j > 0` with `P(invalid) > tau` back to `r882` `slot 0`) replicate on `3` **fresh** `5`-fold family-grouped CV fold seeds (`269140, 269141, 269142`, disjoint from `H265`–`H268`) $\times$ `3` fresh model seeds (`269101, 269102, 269103`) = `45` cells per arm, under `Holm m = 3` (`{NOSG train, NOSG_VETO train, NOSG_VETO val91}`).

### Key Scientific Findings

1. **`NOSG_VETO` solved the inner-`FIT` admission collapse and achieved `LCB95 > 0` inside every single fresh fold seed (`3/3` fold seeds, `9/9` `(fold_seed, model_seed)` evaluations positive), but narrowly missed the preregistered cross-fold-seed context-median `Holm` gate (`NO_ADVANCE`)**:
   - **Inner-`FIT` strict `H257` admission (`gain > +0.001` & `inner_lcb95 > 0` & invalid-gate non-regression)**: `NOSG_VETO` strictly admitted **`45/45` cells (`100.0%`)**, compared to **`29/45` (`64.4%`)** for `NOSG`, **`29/45` (`64.4%`)** for `RERANK4`, and **`0/45` (`0.0%`)** for `NOSG_SHUF`.
   - **Within each individual fresh fold seed on `DL-Train` (`205` families, `1,219` contexts)**, `NOSG_VETO - CONTROL` achieved positive `95%` family bootstrap `LCB95 > 0` and `3/3` model seeds positive:
     - `fold_seed = 269140`: **`+0.013785`** (`CI95 = [+0.002723, +0.031593]`, `p = 0.0071`, `3/3` seeds positive)
     - `fold_seed = 269141`: **`+0.014685`** (`CI95 = [+0.004768, +0.044364]`, `p = 0.0006`, `3/3` seeds positive)
     - `fold_seed = 269142`: **`+0.011417`** (`CI95 = [+0.001489, +0.028753]`, `p = 0.0120`, `3/3` seeds positive)
   - **Preregistered cross-fold-seed context-median contrast (`PRIMARY`)**: Taking the per-context median utility across the `3` fold seeds before family aggregation yielded `NOSG_VETO - CONTROL = +0.011363` (`per_seed = [+0.011363, +0.009586, +0.018618]`, `3/3` seeds positive, `raw_p = 0.0282`, `Holm m=3 adjusted p = 0.0564`, `CI95 = [-0.000199, +0.027251]`, `Holm step-down CI = [-0.001611, +0.030366]`). Because `LCB95 = -0.000199 <= 0` (`Holm LCB = -0.001611 <= 0`) and `model_seed = 269102` had `56` vs `51` `-2.0` invalid picks (`[48, 56, 49]` across the 3 model seeds; median `49 <= 51`, `2/3` seeds `<= CONTROL`), the preregistered verdict is **`NO_ADVANCE`**.
2. **`NOSG` (`528` cols without veto) remained positive on `3/3` model seeds (`+0.015274`) but suffered `-2.0` invalid-gate regressions (`[55, 56, 59]` vs `51`) that widened its tail (`CI95 = [-0.007128, +0.044223]`, `raw_p = 0.1060`)**:
   - Without the invalid-gate classifier veto, `NOSG` picked `[55, 56, 59]` `-2.0` invalid gates across the 3 fold seeds (vs `51` for `CONTROL`), causing two severe family losses (`fam-05246449...` `-0.734610`, `fam-f3dabbc8...` `-0.611491`).
   - `NOSG_VETO` eliminated those catastrophic tail losses (worst harmed family on `NOSG_VETO` was `-0.300046`, with only `9` families harmed `> 0.01` vs `22` on `NOSG`), shrinking the `95%` bootstrap lower bound error by `35x` (from `-0.007128` to `-0.000199`).
3. **Strong replication on `DL-Val91` (`19` families, `91` contexts, `SECONDARY` with reuse disclosure)**:
   - On `fold_seed = 269140`, `NOSG_VETO - CONTROL = +0.009521` (`CI95 = [+0.004580, +0.025024]`, `Holm3 CI = [+0.003219, +0.027924]`, `raw_p = 0.0001`, `Holm m=3 adjusted p = 0.0003`, `3/3` seeds positive, `0` families harmed `> 0.01`, `0` `-2.0` invalid picks).
   - `NOSG - CONTROL = +0.014889` (`CI95 = [+0.004320, +0.026271]`, `raw_p = 0.0014`, `3/3` seeds positive).
   - `RERANK4 - CONTROL = +0.012465` (`CI95 = [+0.005029, +0.019926]`, `raw_p = 0.0001`, `3/3` seeds positive).
4. **Within-family label shuffle placebo (`NOSG_SHUF`) confirmed zero leakage**:
   - `NOSG_SHUF` admitted `0/45` cells (`0.0%`), abstained to `r882` `slot 0` everywhere, and achieved `0.000000` gain (`p = 1.0000`) on both `DL-Train` and `DL-Val91`.

| Gate Criterion | Preregistered Requirement | `NOSG` (`528` cols) | `NOSG_VETO` (`528` cols + Veto) | Pass? |
| --- | --- | --- | --- | --- |
| Strict `H257` inner-`FIT` cells admitted | Diagnostic (`/ 45`) | `29 / 45` (`64.4%`) | **`45 / 45` (`100.0%`)** | — |
| `PRIMARY` (`DL-Train`) median family gain | `> +0.002` | `+0.015274` | `+0.011363` | **PASS** (both) |
| `PRIMARY` (`DL-Train`) per-fold-seed `LCB95 > 0` | Diagnostic (`3` fold seeds) | `0 / 3` (`-0.0094, -0.0142, -0.0059`) | **`3 / 3` (`+0.0027, +0.0048, +0.0015`)** | — |
| `PRIMARY` (`DL-Train`) cross-fold-seed `Holm LCB > 0` | `> 0.0` (`Holm m = 3`) | `-0.007128` (`p_Holm = 0.1060`) | `-0.001611` (`LCB95 = -0.000199`, `p_Holm = 0.0564`) | **FAIL** |
| `PRIMARY` (`DL-Train`) positive model seeds | `3 / 3` | `3 / 3` (`+0.0153, +0.0130, +0.0200`) | `3 / 3` (`+0.0114, +0.0096, +0.0186`) | **PASS** (both) |
| `PRIMARY` beats `NOSG_SHUF` (`LCB95 > 0` or point) | Point `> 0` (`NOSG_SHUF = 0`) | `+0.015274` (`LCB95 = -0.007128`) | `+0.011363` (`LCB95 = -0.000199`) | `NOSG_VETO` point PASS |
| `SECONDARY` (`DL-Val91`) gain & positive seeds | `> 0` on `3 / 3` seeds | `+0.014889` (`3 / 3`, `p = 0.0014`) | `+0.009521` (`3 / 3`, `p_Holm = 0.0003`) | **PASS** (both) |
| `DL-Train` `-2.0` invalid picks `<= CONTROL` (`51`) | `<= 51` | `[55, 56, 59]` (median `56 > 51`) | `[48, 56, 49]` (median `49 <= 51`, `2/3` seeds `<= 51`) | Median PASS / All-seed FAIL |
| **Overall Verdict** | All gates true | `arm_advance = false` | `arm_advance = false` | **`NO_ADVANCE`** |

---

## 2. Primary Endpoint Results (`DL-Train` OOF: `205` Families, `1,219` Contexts, `45` Cells/Arm)

### 2.1 Cross-Fold-Seed Contrasts (`boot_seed = 269150`, `10,000` Paired Family Bootstrap Draws)

| Contrast | Median Family Gain | Per-Seed Family Gains (`269101, 269102, 269103`) | `95%` Bootstrap CI | `Holm m=3` Simultaneous `98.33%` CI | `raw_p` (1-sided) | `Holm m=3 p` | Pos Seeds | Median Context Gain | Harmed Fams (`>0.01`) | `-2.0` Invalid Picks (by seed, sum over 3 fold seeds) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `NOSG - CONTROL` | `+0.015274` | `[+0.015274, +0.012996, +0.019987]` | `[-0.007128, +0.044223]` | `[-0.010303, +0.052798]` | `0.1060` | `0.1060` | `3 / 3` | `+0.015928` | `22 / 205` | `[55, 56, 59]` (vs `51`) |
| `NOSG_VETO - CONTROL` | **`+0.011363`** | `[+0.011363, +0.009586, +0.018618]` | **`[-0.000199, +0.027251]`** | `[-0.002263, +0.031806]` | **`0.0282`** | **`0.0564`** | **`3 / 3`** | `+0.012039` | **`9 / 205`** | **`[48, 56, 49]`** (vs `51`) |
| `NOSG_SHUF - CONTROL` | `+0.000000` | `[+0.000000, +0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `1.0000` | — | `0 / 3` | `+0.000000` | `0 / 205` | `[51, 51, 51]` |
| `RERANK4 - CONTROL` | `+0.013261` | `[+0.005001, +0.023182, +0.013261]` | `[-0.020404, +0.045399]` | `[-0.027342, +0.053271]` | `0.2710` | — | `3 / 3` | `+0.014529` | `22 / 205` | `[58, 58, 59]` (vs `51`) |
| `NOSG_VETO - NOSG` | `-0.003410` | `[-0.003911, -0.003410, -0.001368]` | `[-0.023715, +0.024033]` | `[-0.029674, +0.028768]` | `0.4604` | — | `0 / 3` | `-0.003889` | `20 / 205` | `[-7, 0, -10]` vs `NOSG` |
| `NOSG_UNC - CONTROL` (diag) | `+0.030381` | `[+0.034428, +0.008545, +0.030381]` | `[+0.000689, +0.069395]` | `[-0.005663, +0.077913]` | `0.0229` | — | `3 / 3` | `+0.025108` | `35 / 205` | `[52, 61, 58]` |
| `NOSG_VETO_UNC_SAFE - CONTROL` (diag) | `+0.010846` | `[+0.010846, +0.009586, +0.018618]` | `[-0.000027, +0.027509]` | `[-0.002060, +0.032142]` | `0.0257` | — | `3 / 3` | `+0.010824` | `9 / 205` | `[48, 56, 49]` |
| `TOP4_PICK_ORACLE - CONTROL` | `+0.193361` | `[+0.193361, +0.193361, +0.193361]` | `[+0.139927, +0.270176]` | `[+0.131622, +0.289472]` | `0.0001` | — | `3 / 3` | `+0.190992` | `0 / 205` | `[0, 0, 0]` |

### 2.2 Per-Fold-Seed Breakdown on `DL-Train` (`205` Families)

| Fold Seed | Contrast | Median Family Gain | `95%` Bootstrap CI | `raw_p` (1-sided) | Pos Seeds | Harmed Fams (`>0.01`) |
| --- | --- | --- | --- | --- | --- | --- |
| `269140` | `NOSG_VETO - CONTROL` | **`+0.013785`** | **`[+0.002723, +0.031593]`** | **`0.0071`** | `3 / 3` | `9 / 205` |
| `269140` | `NOSG - CONTROL` | `+0.014656` | `[-0.009360, +0.042674]` | `0.1355` | `3 / 3` | `20 / 205` |
| `269140` | `RERANK4 - CONTROL` | `+0.012078` | `[-0.006626, +0.036266]` | `0.0947` | `3 / 3` | `17 / 205` |
| `269141` | `NOSG_VETO - CONTROL` | **`+0.014685`** | **`[+0.004768, +0.044364]`** | **`0.0006`** | `3 / 3` | `12 / 205` |
| `269141` | `NOSG - CONTROL` | `+0.005989` | `[-0.014248, +0.056432]` | `0.1411` | `2 / 3` | `25 / 205` |
| `269141` | `RERANK4 - CONTROL` | `+0.004385` | `[-0.022946, +0.045598]` | `0.3015` | `2 / 3` | `31 / 205` |
| `269142` | `NOSG_VETO - CONTROL` | **`+0.011417`** | **`[+0.001489, +0.028753]`** | **`0.0120`** | `3 / 3` | `12 / 205` |
| `269142` | `NOSG - CONTROL` | `+0.015545` | `[-0.005883, +0.048055]` | `0.0778` | `3 / 3` | `26 / 205` |
| `269142` | `RERANK4 - CONTROL` | `+0.021690` | `[-0.002957, +0.063075]` | `0.0385` | `3 / 3` | `29 / 205` |

---

## 3. Secondary Endpoint Results (`DL-Val91`: `19` Families, `91` Contexts)

> **Reuse Disclosure**: `DL-Val91` (`19` families, `91` contexts) has been heavily reused across prior waves; `DL-Train` `205`-family OOF on fresh fold seeds `269140, 269141, 269142` is the primary decision endpoint.

### 3.1 `fold_seed = 269140` (`Native91 = 1.1377941898164525` Bit-Exact `91/91`)

| Contrast | Median Family Gain | Per-Seed Family Gains (`269101, 269102, 269103`) | `95%` Bootstrap CI | `Holm m=3` Simultaneous `98.33%` CI | `raw_p` (1-sided) | `Holm m=3 p` | Pos Seeds | Median Context Gain | Harmed Fams (`>0.01`) | `-2.0` Invalid Picks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `NOSG_VETO - CONTROL` | **`+0.009521`** | `[+0.007005, +0.017071, +0.009521]` | **`[+0.004580, +0.025024]`** | **`[+0.003219, +0.027924]`** | **`0.0001`** | **`0.0003`** | **`3 / 3`** | `+0.009939` | **`0 / 19`** | `[0, 0, 0]` |
| `NOSG - CONTROL` | **`+0.014889`** | `[+0.014889, +0.016811, +0.012717]` | **`[+0.004320, +0.026271]`** | `[+0.002548, +0.029229]` | **`0.0014`** | — | **`3 / 3`** | `+0.015543` | `2 / 19` | `[0, 0, 0]` |
| `RERANK4 - CONTROL` | **`+0.012465`** | `[+0.007291, +0.018200, +0.012465]` | **`[+0.005029, +0.019926]`** | `[+0.003712, +0.021711]` | **`0.0001`** | — | **`3 / 3`** | `+0.012984` | `1 / 19` | `[0, 0, 0]` |
| `NOSG_SHUF - CONTROL` | `+0.000000` | `[+0.000000, +0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `[+0.000000, +0.000000]` | `1.0000` | — | `0 / 3` | `+0.000000` | `0 / 19` | `[0, 0, 0]` |
| `NOSG_VETO - NOSG` | `-0.003196` | `[-0.007883, +0.000260, -0.003196]` | `[-0.015069, +0.007460]` | `[-0.017002, +0.010684]` | `0.8190` | — | `1 / 3` | `-0.003314` | `2 / 19` | `[0, 0, 0]` |
| `TOP4_PICK_ORACLE - CONTROL` | `+0.098384` | `[+0.098384, +0.098384, +0.098384]` | `[+0.059714, +0.147994]` | `[+0.052957, +0.161633]` | `0.0001` | — | `3 / 3` | `+0.102566` | `0 / 19` | `[0, 0, 0]` |

### 3.2 `3`-Fold-Seed Aggregate on `DL-Val91` (`45` Models per Arm)

- **`NOSG - CONTROL`**: `+0.015240` (`CI95 = [+0.003980, +0.024949]`, `raw_p = 0.0019`, `3/3` seeds positive)
- **`NOSG_VETO - CONTROL`**: `+0.011379` (`CI95 = [+0.002401, +0.018881]`, `raw_p = 0.0024`, `3/3` seeds positive)
- **`RERANK4 - CONTROL`**: `+0.011548` (`CI95 = [+0.002839, +0.019991]`, `raw_p = 0.0025`, `3/3` seeds positive)

---

## 4. Diagnostic Audit: Feature Importance (`528` Cols) & Context `27`

- **528-Col Block Gain Split (`NOSG`, `fold_seed = 269140`, `model_seed = 269101`, `fold 0`)**:
  - `sem24_z_24` (cols `412..435`): **`42.04%`** (`4,135.27` gain)
  - `prog_412` (cols `0..411`): **`30.21%`** (`2,971.47` gain)
  - `lm48_z_48` (cols `452..499`): **`16.48%`** (`1,621.39` gain)
  - `proj16_z_16` (cols `436..451`): **`7.62%`** (`749.67` gain)
  - `dope_28` (cols `500..527`): **`3.64%`** (`357.93` gain)
- **Context `27` (`fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f`, `DL-Val91`)**:
  - Both `NOSG` and `NOSG_VETO` made `0 / 15` invalid picks on Context `27` (`r882` utility `1.122695`, `slot 3` oracle utility `1.128496`).

---

## 5. Resource Usage & S3 Custody Receipts

- **Smoke run (`smoke-001`)**: `status = PASS`, `exit_code = 0`, `elapsed = 4.98 s`, `peak_rss = 593.8 MB`, `peak_threads = 2`, `affinity = [57, 58, 59, 60]`.
- **Primary run (`attempt-001`)**: `status = PASS`, `exit_code = 0`, `elapsed = 1,079.47 s` (`17.99 min`), `peak_rss = 869.1 MB` (`<= 4 GiB` cap), `peak_threads = 2` (`<= 4` cap), `affinity = [57, 58, 59, 60]`, `pause_requested_seconds = 0.0`.
- **S3 Custody (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H269/attempt-001/`)**:
  - `s3_manifest.json`: `VersionId = .XPgSnC.y.fPK1SqCMs6L3gfFXKU9iI2`
  - `metrics_full.json`: `62,910` bytes, `sha256 = 29018b3733801e8df9e257021c970f4dc87cc7e89b37028f7eb55ebfbbe5925d`, `VersionId = FDeKBSZUh1jhQQkpsEEtYt280ilk8Ul1`
  - `metrics.json` (Git `<= 50 KB`, `100%` lossless): `50,922` bytes, `sha256 = e4957e23d4ae5069324a5e446cb7241dc3a464893909b91bb044cf7bf27b10ab`, `VersionId = pfINOavnvqHwZ2XjRzvhTSucVZwwQBl2`
  - `details.json`: `406,827` bytes, `sha256 = 8e76f75a7fe743eedc733b4b3b0dc925bec65e3d07b327fd4ef37c285e9ecd1e`, `VersionId = hy3ayOqeLTC1CZqJJ6LtIAyde6LJdeEa`
  - `panels.npz`: `384,223` bytes, `sha256 = a7a78ca7327029c462f7ab89643d780388ba7d2c0b2e9b859a87acdf69054560`, `VersionId = Gc9eN7aA9bQEDR5gDrwzjSBuKychN9tA`
  - `fit_selections_lock.json`: `125,064` bytes, `sha256 = 6d912a206f569ac49e6a08ecade69da65112119ac07fffacf2a07efcb2c073c3`, `VersionId = MJq1YH.FdweFSHUSs_d0ssYtuKHjINCE`
  - `baseline.json`: `726` bytes, `sha256 = a5c6a71a79d1527bf742dcbb0b3d7a368a136eb3cb57452132e3c2020f72b1a8`, `VersionId = yd4mnLB15d5LNT3L9SEIzV3.OGuxJYXJ`
  - `attempt-001.resource_summary.json`: `840` bytes, `VersionId = AOD6lE_UMlvwSn4mH8q0qgoEAHuFb_hg`
  - `attempt-001.stdout`: `10,144` bytes, `VersionId = TaMGIPzZwpWd5m3sEYbGjq1YKWhm3yeK`
