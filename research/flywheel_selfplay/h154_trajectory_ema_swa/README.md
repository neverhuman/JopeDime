# H154 — Trajectory EMA and Uniform SWA Weight Averaging with Train-Only Stat Recalibration on the r882 Continuation

**Status:** `COMPLETE` — **`NO_ADVANCE`** vs `r882 = 1500.0` (`r882_verdict = "NO_ADVANCE"`, `trajectory_vs_ctrl_verdict = "NO_TRAJECTORY_AVERAGING_GAIN_OVER_CTRL"`, `serving_promotion = False`).
**Owner:** `gpu_fused_agy` `[FUSED-AGY]`
**Authority:** `~/nudges/gpu_fused_agy_lift_t2215.md` (`ENG:watcher t2215`), `~/nudges/hwave_t2220_gpu_fused_agy.md`, and `~/nudges/hwave_t2300_gpu_fused_agy.md`.
**Source Provenance:** Preregistered in PR `#1315` (`8874ef4fb216969d43d0c41a9f99faf472de659e`); pre-scoring technical amendment [PREREG_AMENDMENT_A1.md](PREREG_AMENDMENT_A1.md) merged in PR `#1331` (`4c40a03224b1f313faa250c032bd6eb959014cd7`, reviewed head `0754d5f7aef700cf0071e297fab55c96a901bfa5`). Executed from merged `main` (`4c40a03224b1f313faa250c032bd6eb959014cd7`) under `/home/ubuntu/jope_plan/evidence/H154/run/` (`120` optimizer updates across `3` seeds $\times 40$ steps, `21` arm-seed cells, `xbabe2` CPU at `nice 19`, peak `3` OS threads $\le 4$, peak RSS `3.57 GB < 6 GB`, scratch `9.36 MB < 1 GB`, wall clock `1,451.0s`).

## 1. Exact Baseline & Integrity Gates

- **Exact `91/91` `r882` baseline gate (`baseline.json`):** `PASS` (`91/91` matches on `DL-Val`, `max_context_error == 0.0`, `mean_error == 0.0`, context mean utility `1.1377941898164525`, 19-family equal-family mean utility `1.1536152769912449`, `1` invalid pick).
- **Secondary `r882` baselines:** `H145` `1,219`-context / `205`-family `DL-Train` context mean `1.423073434552299` and equal-family mean `1.3853554329931828` (`17` invalid picks); held-out `Fold 0` (`254` contexts, `41` families) equal-family mean `1.5545684485943823` (`2` invalid picks).
- **Bitwise freeze & non-adapter anchor verification:** `DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, `adapter_10tok.anchor` (`assert_frozen_anchor`), and all `222` non-trainable tensors (`frozen_sha256 = 29e53de6...`) remained 100% bitwise identical to `r882` across all `21` cells.
- **Dose manipulation check (`decision.doses`):** All `7` arms are `DOSE_VALID` (`CTRL` median L2 drift `0.447721`; `EMA_099` `0.083229`; `EMA_0995` `0.044349`; `EMA_0999` `0.009347`; `SWA_K5` `0.401441`; `SWA_K10` `0.346557`; `SWA_K20` `0.237065`).

## 2. Primary Readout: `DL-Val` (`91` Contexts, `19` Families; `r882 = 1500.0`, Unified `Holm m = 7`)

| Arm | Median Adapter L2 Drift | Recal Scales `(mm, traj)` (Seed Median) | Equal-Family Gain vs `r882` (95% CI) | Seeds `> 0` vs `r882` | 1-Sided $p$ (`Holm m=7` $p$) | **Elo vs `r882 = 1500.0`** (95% CI) | Equal-Family Gain vs `CTRL` (95% CI) | `Holm m=6` $p$ vs `CTRL` | **Elo vs `CTRL = 1500.0`** (95% CI) | Verdict vs `r882` |
| --- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CTRL` | `0.447721` | None (`False`) | `-0.051565` `[-0.116598, -0.006343]` | `0/3` | `0.99790` (`1.00000`) | **`1386.53`** `[1241.88, 1518.28]` | `0.000000` `[0.000000, 0.000000]` | — | **`1500.00`** `[1500.00, 1500.00]` | `NO_ADVANCE` |
| `EMA_099` | `0.083229` | `(0.90560, 0.91841)` | `-0.001904` `[-0.007162, +0.002033]` | `0/3` | `0.76842` (`1.00000`) | **`1481.72`** `[1406.47, 1555.26]` | `+0.049660` `[+0.004994, +0.114100]` | `0.01400` | **`1613.47`** `[1500.00, 1729.29]` | `NO_ADVANCE` |
| `EMA_0995` | `0.044349` | `(0.94848, 0.95562)` | `-0.002438` `[-0.009192, +0.002110]` | `0/3` | `0.76442` (`1.00000`) | **`1481.72`** `[1425.83, 1536.67]` | `+0.049127` `[+0.007112, +0.107844]` | `0.00540` | **`1613.47`** `[1500.00, 1729.29]` | `NO_ADVANCE` |
| `EMA_0999` | `0.009347` | `(0.98890, 0.99047)` | `-0.003219` `[-0.009424, 0.000000]` | `0/3` | `1.00000` (`1.00000`) | **`1463.33`** `[1406.47, 1500.00]` | `+0.048346` `[+0.006058, +0.107210]` | `0.01300` | **`1593.53`** `[1481.72, 1729.29]` | `NO_ADVANCE` |
| `SWA_K5` | `0.401441` | `(0.85000, 0.85000)` | `-0.012798` `[-0.028321, +0.000619]` | `0/3` | `0.96450` (`1.00000`) | **`1463.33`** `[1344.20, 1574.17]` | `+0.018909` `[+0.000923, +0.040862]` | `0.01580` | **`1536.67`** `[1463.33, 1613.47]` | `NO_ADVANCE` |
| `SWA_K10` | `0.346557` | `(0.85000, 0.85000)` | `-0.009880` `[-0.022781, +0.000211]` | `0/3` | `0.97200` (`1.00000`) | **`1444.74`** `[1321.36, 1555.26]` | `+0.039291` `[+0.004880, +0.090733]` | `0.01400` | **`1555.26`** `[1444.74, 1655.80]` | `NO_ADVANCE` |
| `SWA_K20` | `0.237065` | `(0.85000, 0.85000)` | `-0.010139` `[-0.023123, +0.000051]` | `0/3` | `0.97450` (`1.00000`) | **`1444.74`** `[1321.36, 1555.26]` | `+0.041425` `[+0.004601, +0.097137]` | `0.01400` | **`1536.67`** `[1444.74, 1634.16]` | `NO_ADVANCE` |

## 3. Secondary Readouts: `H145` `1,219`-Context / `205`-Family `DL-Train` & Held-Out `Fold 0` (`254` Contexts, `41` Families)

| Arm | `H145` `DL-Train` (`205` fam) Gain vs `r882` (95% CI) | **`H145` `DL-Train` Elo vs `r882 = 1500.0`** (95% CI) | `H145` `DL-Train` Elo vs `CTRL` | `H145` Invalid Picks (`r882 = 17`) | Out-of-Fold `Fold 0` (`41` fam) Gain vs `r882` (95% CI) | **`Fold 0` Elo vs `r882 = 1500.0`** (95% CI) | `Fold 0` Elo vs `CTRL` | `Fold 0` Invalid Picks (`r882 = 2`) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CTRL` | `-0.034936` `[-0.065633, -0.007146]` | **`1438.35`** `[1397.11, 1479.64]` | `1500.00` | `[25, 26, 23]` | `+0.002139` `[-0.034699, +0.055913]` | **`1448.81`** `[1346.80, 1533.99]` | `1500.00` | `[1, 1, 1]` |
| `EMA_099` | `-0.010911` `[-0.022975, -0.000987]` | **`1498.31`** `[1477.94, 1518.66]` | `1558.15` | `[20, 20, 20]` | `+0.000678` `[-0.001414, +0.002828]` | **`1491.53`** `[1431.36, 1542.56]` | `1525.46` | `[2, 2, 2]` |
| `EMA_0995` | `-0.003067` `[-0.010118, +0.001299]` | **`1498.31`** `[1483.04, 1515.26]` | `1556.41` | `[18, 19, 19]` | `+0.000464` `[-0.001188, +0.002357]` | **`1500.00`** `[1457.44, 1542.56]` | `1542.56` | `[2, 2, 2]` |
| `EMA_0999` | `+0.000309` `[-0.000212, +0.001137]` | **`1503.39`** `[1494.92, 1513.56]` | `1561.65` | `[17, 17, 17]` | `-0.000390` `[-0.001338, +0.000168]` | **`1491.53`** `[1457.44, 1516.95]` | `1551.19` | `[2, 2, 2]` |
| `SWA_K5` | `-0.033196` `[-0.060730, -0.008820]` | **`1447.06`** `[1408.10, 1484.74]` | `1523.76` | `[25, 26, 24]` | `-0.010183` `[-0.028324, +0.005462]` | **`1431.36`** `[1346.80, 1508.47]` | `1500.00` | `[2, 2, 2]` |
| `SWA_K10` | `-0.028470` `[-0.051152, -0.004732]` | **`1457.42`** `[1420.70, 1491.57]` | `1520.36` | `[21, 23, 23]` | `-0.013576` `[-0.028855, +0.002892]` | **`1422.51`** `[1346.80, 1508.47]` | `1491.53` | `[2, 2, 2]` |
| `SWA_K20` | `-0.015337` `[-0.035457, +0.003590]` | **`1466.00`** `[1433.09, 1494.92]` | `1544.30` | `[21, 21, 21]` | `-0.006296` `[-0.015214, +0.003595]` | **`1431.36`** `[1356.90, 1500.00]` | `1542.56` | `[2, 2, 2]` |

## 4. Key Findings & Interpretation

1. **Trajectory averaging + train-only RMS recalibration eliminates `96.3%` of raw `CTRL` continuation regression on `DL-Val`, but does not beat `r882` (`NO_ADVANCE`):**
   - Raw 40-step continuation (`CTRL`) drifts `0.4477` L2 from `r882` and regresses by `-0.051565` family mean (`Elo 1386.53` vs `r882 = 1500.0`) on `DL-Val` and `-0.034936` (`Elo 1438.35`) on `1,219`-ctx `DL-Train`.
   - High-decay trajectory `EMA_099` (`+0.049660` vs `CTRL`, `Holm p = 0.01400`, `Elo 1613.47` vs `CTRL`) and `EMA_0995` (`+0.049127` vs `CTRL`, `Holm p = 0.00540`, `Elo 1613.47` vs `CTRL`) beat `CTRL` on `3/3` seeds with positive 95% CIs on `DL-Val` and `DL-Train`, and `EMA_0999` reaches `+0.000309` (`3/3` seeds `> 0`, `Elo 1503.39`) vs `r882` on `1,219`-ctx `H145` `DL-Train` and `EMA_0995` reaches `+0.000464` (`3/3` seeds `> 0`, `Elo 1500.00`) vs `r882` on held-out `Fold 0`.
   - However, on primary `91`-context `DL-Val`, all 6 trajectory-averaged arms remain slightly below `r882` (`-0.001904` to `-0.012798`, `Elo 1444.74–1481.72` vs `1500.0`), so `H154` closes as **`NO_ADVANCE`** with no serving promotion.
