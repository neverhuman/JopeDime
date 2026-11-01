# H147: Matched-Support Training/Serving Reveal Alignment (5-Fold Out-of-Fold CV on 1,219-Context / 205-Family DL-Train)

## Verdict: `NO_ADVANCE` (`COMPLETE`, `45/45` cells, `1,800/1,800` optimizer updates, `serving_promotion = false`)

- **Authority**: `/home/ubuntu/nudges/hwave_t0000_h147exec_gpu_fused_agy.md` (`gpu_fused_agy`, executing preregistration by `rd_astra`).
- **One-Use Execution Ledger**: `/home/ubuntu/jope_plan/ns/receipts/H147_exec_claim.txt` (`1659f00ce907d1ad55a3502e55b168e2da31f0587be60a486c4162d5541750c1`).
- **Executed Source Commit**: `c63be43808ec66ebc74ae10fab20736e92a44ce7` (merged PR `#1383`, reviewed head `44d6e8c88e945d539599eb2f3ba9240811d12b41`).
- **Preregistration Integrity**:
  - `PREREG.md`: `f694b4386939ca18bee165de92aeb18a00d61080f415f096ee79b1396ae9ea89` (byte-identical to merged preregistration).
  - `fold_manifest.json`: `a03f5aa9d9d00d40ea2f990009e64e0844d72729e3ff2ca020ae9cb9be7f0954` (byte-identical to merged preregistration).
  - `PREREG_AMENDMENT_A1.md`: `3d2cb96996746a7f99024c31cda02a56d72f5c30ef74169081080ce0255cab13` (pre-execution asset bindings and harmonization).
- **Hardware & Resource Guards**: `xbabe2` CPU (`nice 19`, `1` math thread, peak `3` OS threads `<= 4`, peak RSS `3,691,053,056` B `= 3.69 GB < 6 GB`, peak scratch `12,369,682` B `= 12.37 MB < 3 GB`, `2,072.65` CPU s `<= 7,200` s, `2,074.76` wall s `<= 10,800` s).
- **Freeze & Boundary Audit**:
  - `DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, frozen `adapter_10tok.anchor` (`5127e7e866dc60f479be203fc109b885e9625d2d70804f397fd23c8a665f5bc0`), and all `222` non-adapter tensors (`bc032d12396cf398d06262a1674dc14f74ab19c3479a3852e819937e8922d543`) verified bitwise identical after every single optimizer step (`1,800` checks across `45` cells).
  - Only the `47` trainable `adapter_10tok.*` tensors (`29,211` parameters) updated.
  - `0` protected/sealed/holdout/meta-test/outer-test reads (`known_catalog_sealed_id_overlap = 0` against `S3_INVENTORY`; global sealed alias roster incompleteness disclosed per `PREREG.md`).
  - `0` reads of the mixed `1,310`-context `/home/ubuntu/jope-data/h58/cache/context_features.npz`; `DL-Train` (`1,219` contexts) and `DL-Val` (`91` contexts) loaded exclusively from isolated feature packets (`train_only_context_features_1219.npz` `5194c29a...`, `val_only_context_features_91.npz` `c480fee5...`).
  - Historical `r882` `DL-Train` exposure is explicitly disclosed (`historical_r882_train_exposure = true`, `continuation_held_out_only = true`).

---

## 1. Baseline Gates (Passed Pre-Continuation)

| Gate | Cohort | Contexts / Families | Max Abs Error | Context Mean Utility | Equal-Family Mean Utility | Status |
|---|---|---:|---:|---:|---:|---|
| Exact `91/91` `r882` `DL-Val` parity | `DL-Val` | `91` / `19` | `0.0` | `1.1377941898164525` | `1.1536152769912449` | `PASS` |
| Two-load `1,219/1,219` `r882` `DL-Train` parity | `DL-Train` | `1,219` / `205` | `0.0` | `1.4230734345522990` | `1.3853554329931828` | `PASS` (`d28fe298...`) |

---

## 2. Primary Readout: 5-Fold Out-of-Fold `DL-Train` (`1,219` Contexts, `205` Families)

For each seed $s \in \{147101, 147102, 147103\}$ and fold $k \in \{0, 1, 2, 3, 4\}$ (`fold_seed = 147140`, `41` assessment families / `164` fit families per fold), all three arms trained for `40` AdamW steps (`lr = 1e-4`, `weight_decay = 1e-4`, grad clip `1.0`) on the exact same `192`-candidate common support $S = A \cup T_{48} \cup \text{draw}_{192 - |A \cup T_{48}|}$:
- `SUPPORT_CTRL`: reveals $B$ (the `4` best-fitness members of $S$) at positions `0..3`; loss over all `192` slots.
- `STORAGE_REVEAL`: reveals $A = [0, 1, 2, 3]$ (storage-order anchors, matching evaluation) at positions `0..3`; loss over all `192` slots.
- `STORAGE_MASK`: identical candidate permutation and revealed anchors $A = [0, 1, 2, 3]$ to `STORAGE_REVEAL`, but loss (`SmoothL1` + `0.1` listwise KL $\tau=0.2$ with target softmax renormalized on `4..191`) is computed over the `188` unrevealed positions (`4..191`) only (`0.0` gradient wrt `pred[:, :4]`).

### 2.1 Equal-Family Utility Contrasts & Pinned Multi-Contender Bradley–Terry MM Elo (`205` Families)

| Arm / Contrast | Seed `147101` $\Delta$ | Seed `147102` $\Delta$ | Seed `147103` $\Delta$ | Median Seed $\Delta$ | 95% Family Bootstrap CI | $p_{\text{one-sided}}$ | Holm $p$ ($m=2$) | Seeds $>0$ | 3-Seed Harm Families (`>10%` cap) | Top-3 Family Share (`<=80%`) | Paired Elo [95% CI] | Pinned BT MM Elo (`r882=1500`) [95% CI] | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`SUPPORT_CTRL` vs `r882`** | `-0.034535` | `-0.029339` | `-0.029655` | **`-0.029655`** | `[-0.061102, -0.002781]` | `0.9834` | — | `0/3` | `103/205` (`50.24%`) | `52.28%` | `1440.10` `[1400.79, 1483.04]` | **`1485.89`** `[1474.43, 1496.58]` | Control |
| **`STORAGE_REVEAL` vs `r882`** | `-0.031457` | `-0.035780` | `-0.026817` | **`-0.031457`** | `[-0.051371, -0.009508]` | `0.9968` | `1.0000` | `0/3` | `107/205` (`52.20%`) | `53.53%` | `1445.32` `[1402.63, 1483.04]` | **`1485.30`** `[1473.90, 1496.88]` | `NO_ADVANCE` |
| **`STORAGE_MASK` vs `r882`** | `-0.019987` | `-0.033643` | `-0.031671` | **`-0.031671`** | `[-0.056254, -0.005939]` | `0.9933` | `1.0000` | `0/3` | `98/205` (`47.80%`) | `50.95%` | `1443.59` `[1402.63, 1483.04]` | **`1487.12`** `[1475.81, 1497.77]` | `NO_ADVANCE` |
| **`STORAGE_REVEAL` vs `SUPPORT_CTRL`** | `+0.003079` | `-0.006441` | `+0.002837` | **`+0.002837`** | `[-0.016672, +0.018101]` | `0.5010` | `1.0000` | `2/3` | `74/205` (`36.10%`) | `58.27%` | `1489.83` `[1462.57, 1523.76]` | — | `NO_ADVANCE` |
| **`STORAGE_MASK` vs `SUPPORT_CTRL`** | `+0.014549` | `-0.004304` | `-0.002016` | **`-0.002016`** | `[-0.016528, +0.013105]` | `0.5583` | `1.0000` | `1/3` | `65/205` (`31.71%`) | `50.22%` | `1503.39` `[1474.53, 1528.88]` | — | `NO_ADVANCE` |
| **`STORAGE_MASK` vs `STORAGE_REVEAL`** | `+0.011470` | `+0.002137` | `-0.004854` | **`+0.002137`** | `[-0.009162, +0.013883]` | `0.3379` | — | `2/3` | `42/205` (`20.49%`) | `77.92%` | `1513.56` `[1493.22, 1532.29]` | — | Diagnostic |

---

## 3. Mechanism & Diagnostic Readouts (`DL-Train`)

1. **Support-Matched Reveal Distribution Shift ($A$ vs $B$)**:
   - Across all `9,600` training slates (`40` steps $\times 16$ contexts $\times 5$ folds $\times 3$ seeds), $A = \{0, 1, 2, 3\}$ and $B$ (the `4` best-fitness members of common support $S$) matched in **`0 / 9,600` slates (`0.0%`)**, and the common support $S$ added at least one storage-order anchor outside $T_{48}$ in **`9,600 / 9,600` slates (`100.0%`)**.
2. **Revealed-Slot Loss & Gradient Mass Concentration**:
   - In `SUPPORT_CTRL` (revealing top-fitness $B$), positions `0..3` accounted for `2.25%` of loss mass and `12.44%` of $\partial \mathcal{L}/\partial \text{pred}$ squared gradient norm (`median L2 drift = 0.4417`).
   - In `STORAGE_REVEAL` (revealing storage-order $A$), positions `0..3` accounted for `1.51%` of loss mass but **`60.65%` of $\partial \mathcal{L}/\partial \text{pred}$ squared gradient norm** (`median L2 drift = 0.4056`).
   - In `STORAGE_MASK`, positions `0..3` accounted for **`0.00%` of loss mass and `0.00%` of $\partial \mathcal{L}/\partial \text{pred}$ squared gradient norm** across every probed step (`median L2 drift = 0.4049`).
   - Removing the `60.65%` revealed-anchor output-gradient mass (`STORAGE_MASK - STORAGE_REVEAL`) yielded a small positive median family gain (`+0.002137`, `2/3` seeds $>0$, paired Elo `1513.56 [1493.22, 1532.29]`,BT MM Elo `1487.12` vs `1485.30`), and aligning revealed training anchors to storage order (`STORAGE_REVEAL - SUPPORT_CTRL`) yielded `+0.002837` (`2/3` seeds $>0$), but neither effect was statistically distinguishable from zero (`95% CI` straddles `0.0`) or overcame the ~`-0.030` continuation degradation relative to `r882` at `lr = 1e-4`.
3. **Anchor-Picking Rate, Invalid (`-2.0`) Counts, & Unrevealed / `U4` Secondary Readouts**:
   - **Anchor choice fraction (`choice in {0,1,2,3}`)**: `r882` `29.04%`; `SUPPORT_CTRL` `28.30%` on all 3 seeds; `STORAGE_REVEAL` `27.73%, 27.65%, 27.65%`; `STORAGE_MASK` `28.14%, 27.97%, 28.14%`.
   - **Invalid (`-2.0` gate-failure) picks out of `1,219` contexts**: `r882` `17`; `SUPPORT_CTRL` `[25, 22, 22]`; `STORAGE_REVEAL` `[24, 25, 23]`; `STORAGE_MASK` `[21, 24, 22]`.
   - **Unrevealed top-1 (`unrev_top1` on slots `4..N-1`) vs `r882` (`1.355472`)**: `SUPPORT_CTRL` `-0.029127 [-0.062741, +0.000176]`; `STORAGE_REVEAL` `-0.029868 [-0.054164, -0.006599]`; `STORAGE_MASK` `-0.028498 [-0.057292, -0.001558]`.
   - **Best-of-4 unrevealed (`u4_top4_unrev`) vs `r882` (`1.576611`)**: `SUPPORT_CTRL` `-0.000589 [-0.010040, +0.009959]`; `STORAGE_REVEAL` `-0.006070 [-0.012823, -0.000454]`; `STORAGE_MASK` `-0.005325 [-0.012811, +0.000255]`.
   - **Best-of-8 union (`anchors 0..3 U top-4 unrevealed`) vs `r882` (`1.609212`)**: `SUPPORT_CTRL` `+0.001916 [-0.007094, +0.012099]`; `STORAGE_REVEAL` `-0.004366 [-0.010101, +0.001028]`; `STORAGE_MASK` `-0.003375 [-0.010532, +0.001586]`.

---

## 4. Descriptive Readout: `91`-Context / `19`-Family `DL-Val` (Fold-Averaged per Seed)

| Arm | Context Mean Utility (Seeds `1..3`) | Equal-Family Mean Utility (Seeds `1..3`) | Equal-Family Median $\Delta$ vs `r882` [95% CI] | Equal-Family Median $\Delta$ vs `SUPPORT_CTRL` [95% CI] | Paired Elo vs `r882` | Pinned BT MM Elo (`r882=1500`) [95% CI] | Mean Invalid Picks / `91` |
|---|---|---|---|---|---:|---:|---:|
| **`r882`** | `1.137794` | `1.153615` | `0.000000` | — | `1500.00` | **`1500.00`** `[1500.00, 1500.00]` | `0.0` |
| **`SUPPORT_CTRL`** | `1.088527, 1.094095, 1.098959` | `1.106531, 1.111864, 1.116524` | `-0.041751` `[-0.108388, -0.003086]` | `0.000000` `[0.000000, 0.000000]` | `1365.84` | **`1500.00`** `[1453.69, 1561.75]` | `1.0, 1.0, 0.6` |
| **`STORAGE_REVEAL`** | `1.094289, 1.097344, 1.094544` | `1.112007, 1.114912, 1.112251` | `-0.041364` `[-0.076716, -0.012579]` | `+0.003047` `[-0.026318, +0.049255]` | `1321.36` | **`1453.18`** `[1409.20, 1501.85]` | `0.4, 0.2, 0.4` |
| **`STORAGE_MASK`** | `1.093096, 1.085857, 1.092894` | `1.110908, 1.103973, 1.110714` | `-0.042901` `[-0.091647, -0.006425]` | `-0.005809` `[-0.014901, +0.008624]` | `1386.53` | **`1456.88`** `[1408.28, 1508.87]` | `0.6, 0.8, 0.6` |

---

## 5. Artifacts & Custody

- Committed artifacts in `research/flywheel_selfplay/h147_reveal_alignment/`:
  - `PREREG.md` (`f694b4386939ca18bee165de92aeb18a00d61080f415f096ee79b1396ae9ea89`)
  - `PREREG_AMENDMENT_A1.md` (`3d2cb96996746a7f99024c31cda02a56d72f5c30ef74169081080ce0255cab13`)
  - `fold_manifest.json` (`a03f5aa9d9d00d40ea2f990009e64e0844d72729e3ff2ca020ae9cb9be7f0954`)
  - `manifest.json`, `preflight.json`, `metrics.json` (`47d1cb044a752d330fa51ff163e84b560b1912bb0913d1ecd31d6b9a2a915857`)
  - `runner.py`, `runtime.py`, `science.py`, `monitor.py`, `test_h147.py`
- Host custody directory: `/home/ubuntu/jope_plan/evidence/H147/{launch.json,run-monitor.json,run-resources.jsonl,run/}`
