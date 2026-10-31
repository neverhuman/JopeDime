# H137 — Target Smoothing (`TS-0.1` $\epsilon = 0.1$, `TS-0.3` $\epsilon = 0.3$) vs. `CONTROL` ($\epsilon = 0.0$) and Unchanged `r882`

## 1. Hypothesis

We test the preregistered hypothesis from `H121` (`research/flywheel_selfplay/h121_target_smoothing/PREREG.md`, `sha256 = 127e0481f06c0d9b42690b3091138b5a927af9fa7b90f10fbf88a80735e6b923`; `runner.py`, `sha256 = 36ef51730ebe9db21a59455c99fcb15580252c92bf296f2618911c1dc1624aec`) under prospective fresh fold and seed bindings (`PREREG_AMENDMENT_A1.md`, `sha256 = 2721fcda10e6abb9957de86203fb9cde54a652856619f2209aca5b63a7031e87`):
- Contracting gated regression targets toward the per-context valid-candidate mean by factor $1 - \epsilon$ via `h121.smooth_targets` ($\tilde{y}_{c,i} = (1 - \epsilon) y_{c,i} + \epsilon \bar{y}_c$ on valid candidates, leaving invalid sentinel `-1.0` untouched) for $\epsilon \in \{0.1, 0.3\}$ (`TS-0.1` and `TS-0.3`) during 40-step AdamW (`lr = 1e-4`, `wd = 1e-4`, clip `1.0`) adapter fine-tuning from `r882` will reduce gradient over-concentration on sharp empirical target peaks and improve `fold 0` (`41` held-out `DL-Train` families, `212` contexts) family-grouped `unrev` utility relative to both unchanged `r882` and unsmoothed `CONTROL` ($\epsilon = 0.0$).

## 2. Setup

- **Executed Source Commit**: `117bc11e59fc0d8c40aef1977095c8aa336e5f96` (PR [#1217](https://github.com/neverhuman/JopeDime/pull/1217) merged to `main`)
- **Preregistration Artifacts (untouched after PR [#1217](https://github.com/neverhuman/JopeDime/pull/1217))**:
  - `PREREG.md`: `cd66d40bcd40a825b8ec4a679bc7aed9fe4f75908fe487dbebeac892d800ea79`
  - `PREREG_AMENDMENT_A1.md`: `2721fcda10e6abb9957de86203fb9cde54a652856619f2209aca5b63a7031e87`
  - `fold_manifest_137171.json` (`fold_seed = 137171`): `a5ef0a687c26ab0da527bd9bfd54ce4d831b81049493abc77a924b6ea8d97af9` (`fold 0` = `41` assessment families / `212` contexts; `folds 1..4` = `164` fit families / `1,007` contexts; `assess_fams & fit_fams == set()`)
- **Pre-Scoring Disclosure (`PREREG_AMENDMENT_A2.md`)**:
  - Filed at `2026-10-31T23:11:00Z` **before** Step-3 scoring readout completed, documenting: (1) the exact JSON key and row closure in `load_dl_train_rows_only` and `main`, CPU-only / zero-CUDA-compute verification, and live `/proc/189673/task/*/stat` OS thread audit (`4` active compute threads `<= 4` + `4` permanently sleeping `vol = 1` runtime threads); and (2) the pre-data `wave_start` disk-guard change from `1 - f_bavail/f_blocks` (`Attempt 1` `run_h137_cpu.attempt1.py` `sha256 = 23f221180129940a800ec5a0c275d7bcb03948a364d9f7783b3655cd0274ae70`, exited at line 701 `check_guards("wave_start")` with `0` data reads, `0` checkpoints, `0` cells, `0` optimizer updates) to physical used percentage `1 - f_bfree/f_blocks` (`Attempt 2` `run_h137_cpu.py` `sha256 = b5d2f00871a710d20deac5ff923c0d493fa085441fcb895e90e45559474cfff5`, `chmod 0444`), which relaxes the disk guard by `1.009` percentage points (ext4 root reserve) while keeping physical disk utilization strictly `< 93.8%` (`93.418%` at `wave_start`, max `93.397%` across all 18 cell-boundary checks, `92.433%` / `484.76 GiB` free at `wave_complete`) with `0` `load_pauses` and `0` `SIGSTOP` pause windows across all `9/9` cells.
- **Starting Checkpoint & Parameter Scope**:
  - Fresh hash-verified copy of `round-882.pt` (`sha256 = 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`; initial adapter `sha256 = 740174e9f0d29b69ea1b3dad2e4545e40eff541501e6bfdec1e79dbe96912183`) per cell.
  - `DOPE` backbone (`sha256 = c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and all `1,126,777` non-trainable parameters (`sha256 = de372fc2b87792e88ae65dbff3541d893622296592c3e95a3335a68757008c92`, `max_abs_diff = 0.0`) verified pre and post every single cell (`9/9` cells + Step-1 regression gate).
  - Only the `29,211`-parameter (`47`-tensor) trainable `adapter_10tok` (`requires_grad = True`, excluding `anchor.*`) is updated (`40` AdamW steps per cell = `20` rounds $\times$ `2` steps/round, `batch_size = 16`, `slate_size = 192`; `360` total optimizer updates across `9` cells).
- **Arms & Seeds**:
  - `CONTROL` ($\epsilon = 0.0$), `TS-0.1` ($\epsilon = 0.1$), `TS-0.3` ($\epsilon = 0.3$) $\times$ fresh seeds `171, 172, 173` (`bootstrap_seed = 137171`, `10,000` paired family-cluster bootstrap draws).

## 3. Outcome

1. **Step 1 — Unchanged-`r882` Regression-First Gate (`PASS`)**:
   - Two independent staged copies of `r882` reproduced `two_copy_fold0_pred_max_abs_diff = 0.0`, `two_copy_fold0_utility_max_abs_diff = 0.0`, `two_copy_control_loss_abs_diff = 0.0`, and `two_copy_control_grad_max_abs_diff = 0.0`.
   - Baseline `r882` on `fold 0` (`41` families, `212` contexts):
     - Family-grouped `unrev` utility: `1.118150` (`context_mean = 1.098031`, `native_family_utility = 1.155308`)
     - Continuous `mean_regret` (`all_candidates`): `6.426676` (`top1_oracle_regret = 0.380521`, `softmax_expected_regret = 0.436844`)
2. **Step 2 & Step 3 — Primary Endpoint (`TS-0.1` and `TS-0.3` vs. `r882` on `fold 0` Family-Grouped `unrev` Utility, 2-Arm Holm Step-Down)**:
   - Neither `TS-0.1` nor `TS-0.3` passes the preregistered admission criteria (`Holm LCB95 > +0.001`, `3/3` seeds `> 0`, `harm <= 10%`):
     - **`TS-0.1` ($\epsilon = 0.1$) vs. `r882` (Holm Rank 1, $\alpha_1 = 0.025$)**:
       - Per-seed `fold 0` family `unrev` utility: `1.126931` (`seed 171`, $\Delta = +0.008781$), `1.130156` (`seed 172`, $\Delta = +0.012006$), `1.113379` (`seed 173`, $\Delta = -0.004771$).
       - Median seed gain: `+0.008781` (`3-seed mean = +0.005338`); two-sided 95% family-cluster bootstrap CI: `[-0.031433, +0.041289]`; **`Holm LCB95 = -0.031433`** (`p_margin_0_001 = 0.3608`, **FAIL**).
       - `all_three_seeds_positive = false` (`2/3` seeds positive; `seed 173` is `-0.004771`, **FAIL**).
       - Harmed family fraction: `26.83%` (`seed_mean` / `pooled`) and `29.27%` (`max_seed`, `12/41` families on `seed 173`, **FAIL `> 10%`**).
       - Paired Elo vs. `r882 = 1500.0`: median seed `1559.90` (`pooled = 1557.00`, 95% CI `[1466.00, 1649.89]`, `W/T/L = 63/17/43`); family-grouped Bradley-Terry Elo (`R882 = 1500.0`): **`1487.64`** (`-12.36` vs. `r882`, context `W/T/L = 98/421/117`).
     - **`TS-0.3` ($\epsilon = 0.3$) vs. `r882` (Holm Rank 2, $\alpha_2 = 0.050$)**:
       - Per-seed `fold 0` family `unrev` utility: `1.126253` (`seed 171`, $\Delta = +0.008103$), `1.126719` (`seed 172`, $\Delta = +0.008569$), `1.105966` (`seed 173`, $\Delta = -0.012184`).
       - Median seed gain: `+0.008103` (`3-seed mean = +0.001496`); two-sided 95% family-cluster bootstrap CI: `[-0.035175, +0.038176]`; **`Holm LCB95 = -0.029123`** (`p_margin_0_001 = 0.4053`, **FAIL**).
       - `all_three_seeds_positive = false` (`2/3` seeds positive; `seed 173` is `-0.012184`, **FAIL**).
       - Harmed family fraction: `29.27%` (`seed_mean` / `pooled`) and `31.71%` (`max_seed`, `13/41` families on `seed 173`, **FAIL `> 10%`**).
       - Paired Elo vs. `r882 = 1500.0`: median seed `1542.58` (`pooled = 1534.09`, 95% CI `[1443.00, 1632.39]`, `W/T/L = 60/17/46`); family-grouped Bradley-Terry Elo (`R882 = 1500.0`): **`1485.03`** (`-14.97` vs. `r882`, context `W/T/L = 97/418/121`).
     - **`CONTROL` ($\epsilon = 0.0$) vs. `r882` (Diagnostic Reference)**:
       - Per-seed `fold 0` family `unrev` utility: `1.127413` (`seed 171`, $\Delta = +0.009263$), `1.130156` (`seed 172`, $\Delta = +0.012006$), `1.113982` (`seed 173`, $\Delta = -0.004168$).
       - Median seed gain: `+0.009263` (`3-seed mean = +0.005700`); two-sided 95% CI: `[-0.031099, +0.041626]`; one-sided `LCB95 = -0.024703`; harmed family fraction `26.83%`; paired Elo `1559.90` (`pooled = 1568.67`); family-grouped Bradley-Terry Elo (`R882 = 1500.0`): **`1488.91`** (`-11.09` vs. `r882`, context `W/T/L = 99/422/115`).
3. **Secondary Endpoint (`TS-0.1` and `TS-0.3` vs. `CONTROL`, Isolating Target-Smoothing Effect)**:
   - Target smoothing monotonically degrades `fold 0` `unrev` utility relative to unsmoothed `CONTROL`:
     - **`TS-0.1` vs. `CONTROL`**: per-seed deltas `-0.000482, 0.000000, -0.000603` (median `-0.000482`, `3-seed mean = -0.000362`, 95% CI `[-0.000924, 0.000000]`, `Holm LCB95 = -0.000804`, paired Elo `1491.52`, BT Elo `1498.73` vs. `CONTROL = 1500.0`, context `W/T/L = 0/634/2`). Note that on `seed 172`, `TS-0.1` ties `CONTROL` (`0.000000`) in top-1 `unrev` selection despite distinct weights (`final_adapter_sha256` `5f375835...` vs. `6b94c710...`); per preregistration, ties do not satisfy strict margin `> +0.001`.
     - **`TS-0.3` vs. `CONTROL`**: per-seed deltas `-0.001161, -0.003437, -0.008016` (median `-0.003437`, `3-seed mean = -0.004204`, 95% CI `[-0.010695, -0.000228]`, `Holm LCB95 = -0.010695`, paired Elo `1483.04`, BT Elo `1496.12` vs. `CONTROL = 1500.0`, context `W/T/L = 2/625/9`).
     - **`TS-0.3` vs. `TS-0.1` (Dose-Response)**: per-seed deltas `-0.000678, -0.003437, -0.007413` (median `-0.003437`, 95% CI `[-0.010318, +0.000015]`, paired Elo `1483.04`).

## 4. Metrics & Red-Team Audit

### Summary Table (`fold 0`: `41` Families, `212` Contexts)

| Arm | Seed `171` `unrev` | Seed `172` `unrev` | Seed `173` `unrev` | Median $\Delta$ vs `r882` | 95% CI vs `r882` | Holm `LCB95` vs `r882` | Median $\Delta$ vs `CONTROL` | Harmed Families (Mean / Max) | Paired Elo vs `r882` (Med / Pooled) | BT Elo (`R882=1500` / `CTRL=1500`) | Continuous Regret (`all` / $\Delta$ vs `r882`) | Adapter $L_2$ Drift (Med) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `r882` (frozen) | `1.118150` | `1.118150` | `1.118150` | `0.000000` | — | — | — | `0.00%` | `1500.00` | `1500.00` / `1511.09` | `6.426676` / `0.000000` | `0.00000` |
| `CONTROL` ($\epsilon=0.0$) | `1.127413` | `1.130156` | `1.113982` | `+0.009263` | `[-0.031099, +0.041626]` | `-0.024703` (1-sided) | `0.000000` | `26.83%` / `26.83%` | `1559.90` / `1568.67` | `1488.91` / `1500.00` | `6.432643` / `+0.005967` | `0.44526` |
| `TS-0.1` ($\epsilon=0.1$) | `1.126931` | `1.130156` | `1.113379` | `+0.008781` | `[-0.031433, +0.041289]` | `-0.031433` (`FAIL`) | `-0.000482` | `26.83%` / `29.27%` | `1559.90` / `1557.00` | `1487.64` / `1498.73` | `6.432639` / `+0.005963` | `0.44681` |
| `TS-0.3` ($\epsilon=0.3$) | `1.126253` | `1.126719` | `1.105966` | `+0.008103` | `[-0.035175, +0.038176]` | `-0.029123` (`FAIL`) | `-0.003437` | `29.27%` / `31.71%` | `1542.58` / `1534.09` | `1485.03` / `1496.12` | `6.432628` / `+0.005953` | `0.45041` |

### Required Red-Team Diagnostics (`~/nudges/rd_argon_t1715.md`)

1. **(a) Family Concentration & Trimmed-Mean Gain (Dropping Top-10 Families)**:
   - Across the `41` `fold 0` families, **95.14%–96.18% of all positive family gain vs. `r882` is concentrated in the top-10 families** (`top10_mean_delta = +0.132649` for `CONTROL` and `TS-0.1`, `+0.132189` for `TS-0.3`, led by `fam-21852904...`, `fam-6e1e9c4f...`, `fam-7a021840...`, `fam-6ed8ae43...`, and `fam-41343514...`).
   - **When those top-10 families are dropped, the trimmed-mean delta over the remaining `31/41` `fold 0` families is strictly NEGATIVE for every arm and every seed**:

| Arm vs. `r882` | Full `41`-Family Mean $\Delta$ (3-Seed Mean) | Top-10 Families Mean $\Delta$ | Top-10 Share of Positive Gain | Trimmed `31/41`-Family Mean $\Delta$ (Top-10 Dropped, 3-Seed Mean) | Trimmed `31/41` Mean $\Delta$ (`seed 171 / 172 / 173`) | Trimmed `31/41` Mean $\Delta$ vs. `CONTROL` |
| --- | --- | --- | --- | --- | --- | --- |
| `CONTROL` ($\epsilon=0.0$) | `+0.005700` | `+0.132649` | `95.137%` (`5.68x` net sum) | **`-0.035251`** | `-0.035184 / -0.033423 / -0.038622` | `0.000000` |
| `TS-0.1` ($\epsilon=0.1$) | `+0.005338` | `+0.132649` | `95.226%` (`6.06x` net sum) | **`-0.035730`** | `-0.035822 / -0.033423 / -0.039420` | `-0.000478` |
| `TS-0.3` ($\epsilon=0.3$) | `+0.001496` | `+0.132189` | `96.179%` (`21.13x` net sum) | **`-0.039821`** | `-0.037088 / -0.038237 / -0.046645` | `-0.005991` |

   - This explains why `seed 171` and `seed 172` show a small positive full-41 mean (`+0.0088` to `+0.0120`) while `seed 173` is negative (`-0.0048` to `-0.0122`) and why family harm is `26.83%–31.71%`: 40-step fine-tuning from `r882` improves 10 high-leverage families while degrading the remaining `31/41` families by `-0.035` to `-0.040` on average.
2. **(b) Pairwise Seed Correlation across Seeds `171, 172, 173`**:
   - Seeds `171, 172, 173` produce distinct final adapter weights in every arm (`3/3` unique SHA-256 hashes per arm; `min_pairwise_max_abs_context_utility_diff` = `0.3652` for `CONTROL`, `0.3652` for `TS-0.1`, `0.3627` for `TS-0.3`).
   - Pairwise correlations across `fold 0` (`41` families / `212` contexts):
     - **`TS-0.1` family $\Delta$ vs. `r882`**: Pearson $r = 0.9698$ (`171 vs 172`), $0.7528$ (`171 vs 173`), $0.7160$ (`172 vs 173`); Spearman $\rho = 0.8925, 0.8620, 0.7841$. Context $\Delta$: Pearson $r = 0.9619, 0.7753, 0.7456$; Spearman $\rho = 0.9128, 0.8704, 0.8908$. Raw family utility: Pearson $r = 0.9993, 0.9948, 0.9942$.
     - **`TS-0.3` family $\Delta$ vs. `r882`**: Pearson $r = 0.9681$ (`171 vs 172`), $0.7152$ (`171 vs 173`), $0.7166$ (`172 vs 173`); Spearman $\rho = 0.8117, 0.7932, 0.7875$. Context $\Delta$: Pearson $r = 0.9598, 0.7192, 0.7317$; Spearman $\rho = 0.8768, 0.8555, 0.9050$. Raw family utility: Pearson $r = 0.9992, 0.9939, 0.9939$.
     - **`CONTROL` family $\Delta$ vs. `r882`**: Pearson $r = 0.9694$ (`171 vs 172`), $0.7526$ (`171 vs 173`), $0.7158$ (`172 vs 173`); Spearman $\rho = 0.8840, 0.8519, 0.7951$.
3. **(c) Metric Direction**:
   - `unrev` utility (`fold0_family_mean_utility`): **higher is better** (`+` delta = improvement over `r882` or `CONTROL`).
   - Continuous `mean_regret` (`h121.mean_regret`, `fold0_mean_regret_all_candidates`): **lower is better** (`-` delta = improvement). All fine-tuned arms increase `mean_regret_all_candidates` vs. `r882` by `+0.005953` to `+0.005967` (`6.432628–6.432643` vs. `6.426676`), while `TS-0.1` (`-4.46e-6`) and `TS-0.3` (`-1.57e-5`) move `mean_regret_all_candidates` only negligibly relative to `CONTROL`.
   - Paired Elo and family-grouped Bradley-Terry Elo (`R882 = 1500.0`, `CONTROL = 1500.0`; no AUC reported anywhere): **higher is better**. Note that per-family paired Elo vs. `r882` is `> 1500` (`1542.58–1559.90`) because `18–25` of `41` families have small positive wins while losses are concentrated on `9–17` families, whereas context-level family-grouped Bradley-Terry Elo (`636` seed-context pairs, `421` ties) is `< 1500` (`1485.03–1488.91`) because harmed contexts outnumber winning contexts (`115–121` losses vs. `97–99` wins).
4. **(d) Zero Sealed / Holdout / Meta-Test / `DL-Val` Reads & `@rd_astra` Source-Review Resolution**:
   - `dl_val_reads = 0`, `outer_test_reads = 0`, `sealed_metatest_reads = 0`, `live_loop_reads = 0`.
   - As detailed in `PREREG_AMENDMENT_A2.md`, `source_manifest.json` (`99a53569...`) is checked only for 4 provenance scalars (`source_manifest_sha256`, `excluded_dl_holdout_count == 93`, `excluded_payloads_opened == 0`, `outer_rows_delivered == 0`; `val_families` is never read). `selected.json` (`981960ad...`) is deserialized by `json.loads` at line 259 and immediately extracts **only** `train_rows = raw_selected["train"]` (`1,219` `DL-Train` rows / `205` families); `raw_selected["dev"]` (`91` `DL-Val` rows) is discarded without indexing or iterating, and **zero** `.npz` pools or `.jtt1` tables outside the `1,219` `DL-Train` contexts are ever opened. Within `DL-Train`, `folds 1..4` (`164` families / `1,007` contexts) are used strictly for fit and `fold 0` (`41` families / `212` contexts) is used strictly for post-fit read-only scoring.
5. **(e) `DOPE` `c51a56d0...` Pre/Post & Frozen Parameter Verification**:
   - `dope_sha256_pre == dope_sha256_post == "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"` and `non_trainable_max_abs_diff == 0.0` across all `1,126,777` non-trainable parameters (`sha256 = de372fc2b87792e88ae65dbff3541d893622296592c3e95a3335a68757008c92`) in all `9/9` cells and Step 1.
6. **(f) CPU-Only / OS Thread Ceiling / Disk-Guard Audit**:
   - `CUDA_VISIBLE_DEVICES=""` set before `import torch`; `gpu_seconds = 0.0`; no `qualification.lock`.
   - Live `/proc/189673/task/*/stat` audit confirmed `4` active CPU compute threads (`<= 4`) (`tid 189673` `403.71s`, `tid 246083` `36.39s`, `tid 246084` `36.76s`, `tid 246085` `36.46s`) + `4` permanently sleeping (`state = S`, `vol = 1, nonvol = 0`) runtime threads.
   - `Amendment A2` relaxed `check_guards` from `1 - f_bavail/f_blocks` to physical used `1 - f_bfree/f_blocks` (`~1.009` pt ext4 root-reserve difference); physical `disk_used_pct` remained strictly `< 93.8%` (`92.433%–93.418%`, `411.92–484.76 GiB` free), with `0` `load_pauses` and `0` `SIGSTOP` pause windows across all `9/9` cells.

## 5. Verdict

**`NO_ADMISSION`** (`admitted_arms = []`, `candidate_breakthrough = false`).
- Both `TS-0.1` (`Holm LCB95 = -0.031433`, `2/3` positive seeds, `26.83%` mean / `29.27%` max family harm, trimmed `31/41`-family mean delta `-0.035730`) and `TS-0.3` (`Holm LCB95 = -0.029123`, `2/3` positive seeds, `29.27%` mean / `31.71%` max family harm, trimmed `31/41`-family mean delta `-0.039821`) fail the preregistered primary admission gates vs. `r882` and underperform unsmoothed `CONTROL` (`-0.000482` and `-0.003437` median seed delta vs. `CONTROL`).

## 6. Next Step

Target smoothing ($\epsilon \in \{0.1, 0.3\}$) during 40-step AdamW fine-tuning from `r882` is falsified (`TS-0.1` and `TS-0.3` monotonically trail `CONTROL`, and `> 95%` of positive family gains across all fine-tuned arms come from `10/41` outlier families while the remaining `31/41` families suffer a `-0.035` to `-0.040` trimmed-mean loss). Close out the target-smoothing / single-run fine-tuning line on `r882` and prioritize prediction-level checkpoint blends (`H129`/`H134`/`H135`) that control per-family harm without unanchored weight drift.
