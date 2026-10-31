# H85 — W1 CPU Pre-Review Remediation, Persisted Outer-Fold Models (Unblocking H81 W4 Prospective), Replayable Score Vectors, Matched-Comparator Contrasts, and Within-Context Template Pairing

- **Hypothesis ID**: `H85`
- **Wave**: `W1` (`overnight16_t0050` / `t2058` / `t2101`)
- **Owner**: `rd_argon` (`[ARGON]`)
- **GPU Budget Used**: `0.0` GPU-hours (100% CPU across `xbabe1`, `xbabe2`, `xbabe3`, `xbabe4`)
- **Holdout Hygiene**: Strictly `1,219` `DL-Train` contexts (`205` families, `nq == 0` train-only 2-fold CV `.jtt1` tables); `0` `DL-Val` payloads opened; `0` outer-test or sealed meta-test rows touched.

---

## 1. Executive Summary & Pre-Review CPU Resolution

Per `~/nudges/overnight16_t0050.md`:
> *"`rd_argon`: if `W1` pre-review is signed, queue `W1` on `qualification.lock` AFTER `H57` and `Fused D`; otherwise fix what pre-review flagged on CPU."*

While `qualification.lock` is occupied by active GPU waves and `W1` pre-review awaits formal `launch_admission.json` signoff, `H85` resolves **every CPU-side item** flagged across the three `W1` pre-review documents (`sol_v1_mcp01` `W1-R1..R8`, `H70`, `H78` PR #876, and `H81` PR #895):

1. **Persisted Outer-Fold HistGBM Models & Exact H81 W4 Unblocker (`H81` & `H78`)**:
   - Diagnosed the exact root cause of `H81`'s `1.8620` vs `1.9159` refit discrepancy: `H81` (`run_prospective.py`) concatenated sampled training rows in `selected.json` interleaved-fold order (`X[FK != k]`), whereas `W1` (`prepare_w1_cpu.py`) concatenated training rows grouped by fit fold (`for f in fit_folds:`). Because `HistGradientBoostingRegressor._BinMapper` subsamples rows using `random_state` to compute quantile bin edges when `n_samples > 100,000`, changing row ordering altered bin edges (`r = 0.952`).
   - Refitted and persisted all 5 fold-grouped outer-fold `HistGradientBoostingRegressor` models (`outer_0..4` for `program412` and `program412+lm48_z+dope629`, plus linear calibrators) across 4-node CPU fan-out (`xbabe1`, `xbabe3`, `xbabe4` `colima` `jope-ci` Linux container, and `xbabe2`) in both `w1_outer_fold_models.pkl` (`28,886,930` B, SHA-256 `2ebc4859d10cef0bfff9849cf01bc831ca9a4cb1222b1ad6dd956a5ef8f5b09f`) and drop-in `h81_models.pkl` (`17,512,005` B, SHA-256 `e378eb68c21c718e62b6ec68b7d8ccbc285d268fddd831abd01309b5c7412b4e`).
   - Verified `max |Δ| == 0.0` against stored `w1_train_only_priors.npz` (`max |Δ_cost| == 4.77e-6`) and verified `H81`'s 50-context W4 comparator strength guard: persisted models achieve `1.915928` family mean (`1.915872` context mean) vs stored `1.915928` (`gap == 0.000000`, passing `H81`'s `0.02` guard and immediately unblocking `@rd_opus` H81 W4 prospective).
2. **Replayable Per-Context (`1,219`) and Per-Family (`205`) Score Vectors & 10,000-Draw Elo CIs (`H78`)**:
   - Emitted full `1,219`-context and `205`-family score vectors for all contenders (`R882_CONTINUITY`, `RIDGE_PRIOR10_ONLY`, `GBM_PROG412_ONLY`, `GBM_PROG412_CTX`, `ZERO_PRIOR_STEP0`, `RIDGE_PRIOR10_STEP0`, `GBM_RESIDUAL_STEP0`, plus `RANDOM4` and `U_0`) in `w1_cpu_per_context_and_family_vectors.json` (`3,134,730` B, SHA-256 `120d34990001feab338a5ea856007c71fe012b9e272f674de74b9dc2bfa2dc3d`).
   - Computed both `10,000`-draw and `1,000`-draw family-cluster bootstrap Bradley-Terry Elo CIs (`seed=20261031..20261033`) and reported both `205`-family W/T/L and `1,219`-context W/T/L against `R882_CONTINUITY`.
3. **Matched-Comparator DOPE Kernel-Anchoring Contrasts & Official `H70` Gate Evaluation (`H70` & `H78`)**:
   - Computed matched-comparator contrasts (`GBM_RESIDUAL_STEP0` vs `GBM_PROG412_ONLY`, `GBM_RESIDUAL_STEP0` vs `GBM_PROG412_CTX`, `RIDGE_PRIOR10_STEP0` vs `RIDGE_PRIOR10_ONLY`) on both `Track 2 U_4` and `Track 1 Legacy`.
   - Evaluated `H70`'s official `gate.py` `verdict` calculator at step 0 (`decision: "NO_ADVANCE"`, `scientific_result: false`, `u4_common_gate_pass: false`, `legacy_common_gate_pass: false`, `legacy_no_harm_lcb: -0.166939`, `gpu_used: false`).
4. **Within-Context Paired Seen-vs-Unseen Template Resolution (`H78`)**:
   - Resolved `H78`'s subgroup-composition critique (`unseen U_4 > seen U_4`) by computing within-context paired `U_4_unseen(c) - U_4_seen(c)` across all `1,219` contexts (`205` families), every one of which contains both `>= 4` seen-in-fit and `>= 4` unseen-in-fit unrevealed templates.
5. **Import-Time Script Hash Binding, Shard Sidecars, andMeasured Process CPU Accounting (`W1-R1..R3` & `H78`)**:
   - Added `SELF_SHA256_AT_IMPORT` and `PREP_SHA256_AT_IMPORT` binding to `w1_runner.py` (`w1_runner_h85.py`, SHA-256 `9269bff1ca33de5da8577c791aa3bfc50b01d836d714bd3a0c1338d46bab5425`), verified cryptographic shard sidecars (`input_npz_sha256`, `worker_py_sha256`, `output_pkl_sha256`), and recorded measured `ru_utime + ru_stime` process CPU seconds on every shard.

---

## 2. Bradley-Terry Elo Panel (`1,219` `DL-Train` Contexts, `205` Families, `10,000` & `1,000` Draws)

Anchored at `R882_CONTINUITY = 1500.0` Elo:

| Contender | Track 2 Verified `U_4` Elo (`10k` CI95 / `1k` CI95) | `U_4` Family W/T/L (`205`) | `U_4` Context W/T/L (`1,219`) | Track 1 Legacy Elo (`10k` CI95 / `1k` CI95) | Legacy Family W/T/L (`205`) | Legacy Context W/T/L (`1,219`) | `U_4` Family Mean | Legacy Family Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `GBM_PROG412_CTX` | **1527.93** (`[1504.64, 1550.92]` / `[1503.84, 1551.39]`) | `129/2/74` | `529/274/416` | **1490.16** (`[1467.56, 1512.04]` / `[1469.12, 1512.80]`) | `102/0/103` | `588/9/622` | **1.6803** | `1.3174` |
| `RANDOM4` (`R_4`) | **1507.26** (`[1488.82, 1525.81]` / `[1489.38, 1526.10]`) | `94/0/111` | `471/211/537` | — | — | — | `1.5731` | — |
| `R882_CONTINUITY` | **1500.00** (anchor) | `0/205/0` | `0/1219/0` | **1500.00** (anchor) | `0/205/0` | `0/1219/0` | `1.6092` | `1.3854` |
| `GBM_RESIDUAL_STEP0` | **1489.02** (`[1469.76, 1508.23]` / `[1469.47, 1508.19]`) | `105/1/99` | `431/330/458` | **1438.29** (`[1412.65, 1463.18]` / `[1415.01, 1462.64]`) | `82/0/123` | `498/22/699` | `1.6472` | `1.2341` |
| `ZERO_PRIOR_STEP0` | **1479.86** (`[1464.42, 1495.89]` / `[1464.67, 1496.14]`) | `59/7/139` | `346/425/448` | **1566.92** (`[1548.09, 1586.34]` / `[1547.59, 1585.95]`) | `134/4/67` | `540/341/338` | `1.5526` | **1.4238** |
| `GBM_PROG412_ONLY` | **1456.33** (`[1435.27, 1477.56]` / `[1435.38, 1475.65]`) | `100/2/103` | `378/342/499` | **1409.22** (`[1380.90, 1436.79]` / `[1381.45, 1436.23]`) | `77/0/128` | `448/9/762` | `1.6498` | `1.1836` |
| `RIDGE_PRIOR10_STEP0` | **1389.17** (`[1369.71, 1408.50]` / `[1369.14, 1408.39]`) | `46/3/156` | `233/375/611` | **1320.56** (`[1292.03, 1346.73]` / `[1293.94, 1348.02]`) | `32/0/173` | `313/5/901` | `1.5249` | `0.7916` |
| `RIDGE_PRIOR10_ONLY` | **1376.24** (`[1356.63, 1395.31]` / `[1356.79, 1395.33]`) | `35/4/166` | `216/381/622` | **1322.89** (`[1294.21, 1349.93]` / `[1294.60, 1350.49]`) | `35/0/170` | `314/4/901` | `1.5115` | `0.7800` |

---

## 3. Matched-Comparator DOPE Kernel-Anchoring Contrasts (`10,000` Draws, `seed=20261031`)

Isolating the effect of wrapping a fixed prior inside DOPE's 4-reveal Nadaraya-Watson kernel + fold-clean step-0 `h40b` adapter (`scale=0.1`) against its own standalone base prior:

| Matched Contrast | Endpoint | Elo Delta | Family Mean $\Delta$ | Median $\Delta$ | One-Sided `LCB95` | Adjusted `LCB` ($m=4$) | Two-Sided 95% CI | Family W/T/L (`205`) | Harm Tail ($<-0.01$) | Common Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `GBM_RESIDUAL_STEP0` vs `GBM_PROG412_ONLY` | Track 2 `U_4` | **+32.69 Elo** | `-0.0026` | `+0.00004` | `-0.0161` | `-0.0220` | `[-0.0191, +0.0106]` | `109/36/60` | `12.68%` (`26/205`) | `False` |
| `GBM_RESIDUAL_STEP0` vs `GBM_PROG412_CTX` | Track 2 `U_4` | **-38.91 Elo** | `-0.0331` | `-0.00158` | `-0.0634` | `-0.0765` | `[-0.0704, -0.0055]` | `82/7/116` | `35.12%` (`72/205`) | `False` |
| `RIDGE_PRIOR10_STEP0` vs `RIDGE_PRIOR10_ONLY` | Track 2 `U_4` | **+12.93 Elo** | `+0.0134` | `0.00000` | `+0.0006` | `-0.0011` | `[-0.0004, +0.0365]` | `81/63/61` | `9.76%` (`20/205`) | `False` |
| `GBM_RESIDUAL_STEP0` vs `GBM_PROG412_ONLY` | Track 1 Legacy | **+29.07 Elo** | `+0.0505` | `+0.00506` | `+0.0117` | `-0.0042` | `[+0.0039, +0.0960]` | `116/31/58` | `18.54%` (`38/205`) | `False` |
| `GBM_RESIDUAL_STEP0` vs `GBM_PROG412_CTX` | Track 1 Legacy | **-51.87 Elo** | `-0.0833` | `-0.01574` | `-0.1454` | `-0.1669` | `[-0.1561, -0.0094]` | `78/0/127` | `54.63%` (`112/205`) | `False` |
| `RIDGE_PRIOR10_STEP0` vs `RIDGE_PRIOR10_ONLY` | Track 1 Legacy | **-2.33 Elo** | `+0.0117` | `0.00000` | `-0.0307` | `-0.0451` | `[-0.0384, +0.0650]` | `97/13/95` | `37.07%` (`76/205`) | `False` |

- **Interpretation**: Wrapping `GBM_PROG412_ONLY` inside DOPE's 4-reveal residual kernel at step 0 improves pairwise family win rate (`109W/36T/60L` on `U_4`, `+32.69` Elo; `116W/31T/58L` on Legacy, `+29.07` Elo and `+0.0505` family mean), because the 4 revealed seed anchors provide context calibration that `GBM_PROG412_ONLY` lacks. However, without gradient training on the `h40b` multimodal context tokens, `GBM_RESIDUAL_STEP0` trails `GBM_PROG412_CTX` (`-38.91` Elo on `U_4`, `-51.87` Elo on Legacy) and properly returns `decision: "NO_ADVANCE"` under `H70`'s official `gate.py` calculator.

---

## 4. Within-Context Paired Seen-vs-Unseen Template Resolution (`H78`)

Every single one of the `1,219` `DL-Train` contexts (`205` families) contains both `>= 4` seen-in-fit and `>= 4` unseen-in-fit unrevealed templates, so pooled and within-context paired family means are evaluated on the exact same `1,219` contexts:

- **`GBM_PROG412_CTX`**: within-context unseen family mean `1.668578` vs seen `1.639563` -> paired within-context $\Delta = +0.029015$ (context mean $\Delta = +0.029282$, median $+0.011076$, two-sided 95% CI `[+0.006029, +0.050471]`, one-sided `LCB95 = +0.009793`, `158W/7T/40L`, `common_gate_passed = True`).
- **`GBM_PROG412_ONLY`**: within-context unseen family mean `1.656648` vs seen `1.636833` -> paired within-context $\Delta = +0.019815$ (context mean $\Delta = +0.022653$, median $+0.005497$, two-sided 95% CI `[+0.000895, +0.037778]`, one-sided `LCB95 = +0.003886`, `144W/13T/48L`).
- **Mechanism**: This rules out between-context subgroup composition (`Simpson's paradox`). Within the same context, the unseen-in-fit candidate stratum consists of context-specialized mutated/champion DSL templates not shared across other families' pools (`H84`), which carry a higher utility ceiling than generic templates shared across folds, and `program412` continuous AST/operator features transfer cleanly to those unseen templates.

---

## 5. S3 Custody & Verification

All artifacts are uploaded create-only (`--if-none-match "*"`) and verified by version-pinned S3 download and SHA-256 readback under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W1/argon-h85-cpu-remediation-20261031T0653Z/`:

- `manifest.json`: VersionId `UQHpDyBbR3be29GfmerauoLcB2.vOhLz`, SHA-256 `a7f7b91f5f25d74499f6af6fcdca0f6b53c212d3fec76d03d62bf44dddd288f1`
- `w1_outer_fold_models.pkl`: VersionId `n6BnfyMERez1CZFbU.oonOMWZfO7Sdsx`, SHA-256 `2ebc4859d10cef0bfff9849cf01bc831ca9a4cb1222b1ad6dd956a5ef8f5b09f`
- `h81_models.pkl`: VersionId `aOf8X2rJ3CXs7LoeJC8E3pfxIP1d6Nly`, SHA-256 `e378eb68c21c718e62b6ec68b7d8ccbc285d268fddd831abd01309b5c7412b4e`
- `w1_cpu_per_context_and_family_vectors.json`: VersionId `Ar.0rJGiIGDbeA77qkVvAzOupQqzCc__`, SHA-256 `120d34990001feab338a5ea856007c71fe012b9e272f674de74b9dc2bfa2dc3d`
- `w1_cpu_baselines_report_v2.json`: VersionId `76SWPxMqwignfH3Ck1xuKphDLiiGFxGT`, SHA-256 `84a678def31f6934ba95312d339abf3a40c70b27e034a059607093898feb4fc7`
- `w1_runner_h85.py`: VersionId `jEcNm1wEzpBXGwNAYlsNp4Vu.op6bDHP`, SHA-256 `9269bff1ca33de5da8577c791aa3bfc50b01d836d714bd3a0c1338d46bab5425`
- `build_and_verify_h85.py`: VersionId `lcCIWQ7qObsz2bSVimVUPVISN2ur6OOT`, SHA-256 `8201bd9066186e78b07efd4640a23b178563f3ea12a6be153e8b1b13fcc71e47`

Local verification commands:
```bash
nice -n 10 pytest -q research/flywheel_selfplay/h85_w1_cpu_remediation_and_models/test_w1_remediation.py
ruff check research/flywheel_selfplay/h85_w1_cpu_remediation_and_models/
```
