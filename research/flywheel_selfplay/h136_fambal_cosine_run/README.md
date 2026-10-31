# H136: Family-Balanced 4x Accumulation (`FAMBAL_ACC4`) & Warmup + Cosine Warm Restarts (`COSWR`) vs `CONTROL` and `r882`

- **Wave ID**: `H136`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Status**: `COMPLETE` (`Verdict = NO_ADMISSION`; `candidate_breakthrough = false`; no serving promotion)
- **Execution Mode**: CPU-only (`nice 19`, `<= 4` threads, `0.0` GPU seconds, `360` AdamW optimizer updates across `9` cells)
- **Lineage**: Executes the preregistered `H109` family-balanced accumulation & cosine warm-restart design (`research/flywheel_selfplay/h109_family_balanced_accum/runner.py`, `sha256 = 68ed53d5ba80ce52b2450de526c54f2468129eb13d0cb93cba93a8562953866b`) on a fresh 5-fold family-grouped `DL-Train` split (`fold_seed = 136161`).
- **Documents**:
  - [PREREG.md](PREREG.md) (`sha256 = a9bbce83d944ab1fab9d88a96b82ea01207fc093f85a08e7cfef117dfda1aa74`, untouched since PR [#1199](https://github.com/neverhuman/JopeDime/pull/1199))
  - [PREREG_AMENDMENT_A1.md](PREREG_AMENDMENT_A1.md) (pre-forward-pass loader unpacking disclosure `a8fa2908...` -> `b0eda24a...` and read-only Step-3 summary finalizer `92e0b425...`)
  - [fold_manifest_136161.json](fold_manifest_136161.json) (`sha256 = 6ff1a5c48564a2b13b83f3b50403bfa113f8e1d0361bd5f1c3cb3c83ce872496`: `fold 0` = `41` assessment families / `245` contexts; `folds 1..4` = `164` fit families / `974` contexts)
  - [manifest.json](manifest.json)
  - [metrics.json](metrics.json) (`49,780` bytes `< 100 KB`, `sha256 = 7a8d7d56751b7ef6f3879313e40bdc84c33df47e92ab9399511862708ae33bb5`)

## 1. Unchanged-`r882` Regression-First Gate (`Step 1`)

- **Status**: `PASS` ([regression_receipt.json](file:///home/ubuntu/jope-rescue-20260921/h136-fambal-cosine-run/regression_receipt.json), `sha256 = e30990a5e8b5e887b6b96d433152f2d43ad14f358b066644ed4964de7f5bec73`)
- Two staged copies of `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`, frozen `DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, initial adapter `740174e9f0d29b69ea1b3dad2e4545e40eff541501e6bfdec1e79dbe96912183`) verified on `fold 0` before any training cell:
  - `two_copy_fold0_pred_max_abs_diff = 0.0`
  - `two_copy_fold0_utility_max_abs_diff = 0.0`
  - `two_copy_control_loss_abs_diff = 0.0`
  - `two_copy_control_grad_max_abs_diff = 0.0`
- **Unchanged `r882` Baseline on `fold 0` (`41` families / `245` contexts)**:
  - Family mean `unrev` utility: `1.148793` (`1.1487925631665765`)
  - Context mean `unrev` utility: `1.228365` (`1.2283650391113055`)
  - Family mean `native` utility: `1.182967` (`1.1829669000847078`)
  - Continuous mean regret (`all_candidates` / `top1_oracle` / `softmax_expected`): `9.409206` / `0.230762` / `0.295250`

## 2. Primary Readout: Paired Family-Cluster Bootstrap vs `r882` (`10,000` draws, `bootstrap_seed = 136161`)

| Arm | Seed 161 `unrev` ($\Delta$ vs `r882`) | Seed 162 `unrev` ($\Delta$ vs `r882`) | Seed 163 `unrev` ($\Delta$ vs `r882`) | Median Seed Gain vs `r882` | Mean Seed Gain vs `r882` | 95% Bootstrap CI | Holm Rank ($\alpha_k$) | Holm One-Sided LCB95 | Family Harm (`< -0.01`) | Paired Elo vs `r882=1500` (95% CI) | Joint BT Elo (`r882=1500`) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CONTROL` | `1.128230` (`-0.020563`) | `1.129708` (`-0.019085`) | `1.144269` (`-0.004523`) | `-0.019085` | `-0.014724` | `[-0.041260, +0.004019]` | Comparator | `-0.036222` (unadj.) | `26.83%` (`11/41`) | `1451.67` (`[1363.44, 1534.08]`, W/T/L `34/38/51`) | `1490.69` | Baseline |
| `COSWR` | `1.148202` (`-0.000590`) | `1.129225` (`-0.019567`) | `1.147661` (`-0.001131`) | `-0.001131` | `-0.007096` | `[-0.020474, +0.002470]` | `1` (`0.025`) | `-0.020474` (`p=0.9046`) | `19.51%` (`8/41`) | `1466.00` (`[1395.25, 1534.00]`, W/T/L `26/59/38`) | `1495.26` | `NO_ADMISSION` |
| `FAMBAL_ACC4` | `1.126887` (`-0.021905`) | `1.124690` (`-0.024102`) | `1.126986` (`-0.021806`) | `-0.021905` | `-0.022605` | `[-0.060905, +0.002730]` | `2` (`0.050`) | `-0.053562` (`p=0.9460`) | `29.27%` (`12/41`) | `1443.00` (`[1346.71, 1534.00]`, W/T/L `35/33/55`) | `1486.28` | `NO_ADMISSION` |

## 3. Secondary Readout: Treatment Arms vs `CONTROL` (`10,000` draws)

| Comparison | Per-Seed Family Mean Deltas (`161, 162, 163`) | Median Seed Gain | Mean Seed Gain | 95% Bootstrap CI | Holm One-Sided LCB95 | Family Harm (`< -0.01`) | Paired Elo (`CONTROL=1500`) | Joint BT Elo (`CONTROL=1500`) | Seeds `> 0` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `COSWR` vs `CONTROL` | `+0.019972, -0.000482, +0.003392` | `+0.003392` | `+0.007627` | `[-0.003554, +0.022009]` | `-0.003554` (`p=0.1159`) | `14.63%` (`6/41`) | `1522.63` (`[1463.14, 1583.49]`, W/T/L `31/69/23`) | `1504.57` | `2/3` |
| `FAMBAL_ACC4` vs `CONTROL` | `-0.001343, -0.005017, -0.017283` | `-0.005017` | `-0.007881` | `[-0.020505, -0.000009]` | `-0.017959` (`p=0.9752`) | `9.76%` (`4/41`) | `1460.28` (`[1419.50, 1497.18]`, W/T/L `11/87/25`) | `1495.59` | `0/3` |
| `FAMBAL_ACC4` vs `COSWR` | `-0.021315, -0.004535, -0.020675` | `-0.020675` | `-0.015508` | `[-0.041677, +0.002805]` | `-0.036686` (unadj.) | `19.51%` (`8/41`) | `1466.00` (`[1398.33, 1534.00]`, W/T/L `25/61/37`) | — | `0/3` |

## 4. Continuous Mean-Regret Co-Readout & Adapter L2 Drift

| Contender / Arm | Per-Seed Adapter L2 Drift (`161, 162, 163`) | Median Adapter L2 Drift | All-Candidate Mean Regret (Median Seed / $\Delta$ vs `r882`) | Top-1 Oracle Regret (Median Seed / $\Delta$ vs `r882`) | Softmax-Expected Regret (Median Seed) | Fit Families Sampled per Cell |Non-Trainable Diff (`1,126,777` params) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `R882` (unchanged) | `0.00000, 0.00000, 0.00000` | `0.00000` | `9.409206` (`0.000000`) | `0.230762` (`0.000000`) | `0.295250` | `0/164` (`0` updates) | `0.0` |
| `COSWR` | `0.24531, 0.24658, 0.24585` | `0.24585` (`-44.1%` vs `CONTROL`) | `9.411057` (`+0.001851` vs `r882`, `-0.001019` vs `CONTROL`) | `0.230531` (`-0.000231` vs `r882`) | `0.296743` | `155, 157, 156 / 164` | `0.0` |
| `CONTROL` | `0.43980, 0.44093, 0.43652` | `0.43980` | `9.412185` (`+0.002979` vs `r882`) | `0.246166` (`+0.015404` vs `r882`) | `0.298377` | `155, 157, 156 / 164` | `0.0` |
| `FAMBAL_ACC4` | `0.50627, 0.51278, 0.50806` | `0.50806` (`+15.5%` vs `CONTROL`) | `9.412357` (`+0.003151` vs `r882`, `+0.000173` vs `CONTROL`) | `0.250526` (`+0.019764` vs `r882`) | `0.299866` | `164, 164, 164 / 164` | `0.0` |

## 5. Key Findings & Mechanistic Interpretation

1. **Both `COSWR` and `FAMBAL_ACC4` are `NO_ADMISSION` vs `r882`**:
   - Neither arm achieves positive median gain vs unchanged `r882` (`COSWR` `-0.001131`, `FAMBAL_ACC4` `-0.021905`).
2. **`COSWR` substantially mitigates `CONTROL` drift and regression**:
   - Linear warmup (`4` updates) + cosine warm restarts (`T_0 = 40/3`) cuts adapter L2 drift from `0.43980` (`CONTROL`) to `0.24585` (`-44.1%`), beats `CONTROL` on `2/3` seeds (`median_seed_gain = +0.003392`, `mean_seed_gain = +0.007627`, paired Elo `1522.63`), lowers continuous all-candidate regret on `3/3` seeds (`-0.001019` median delta vs `CONTROL`), and slightly improves median top-1 oracle regret over `r882` (`0.230531` vs `0.230762`).
3. **`FAMBAL_ACC4` at flat `lr = 1e-4` increases adapter drift and fold-0 degradation**:
   - Although `FAMBAL_ACC4` achieves `100%` fit-family coverage (`164/164` families per cell across `160` microbatches) and reduces gradient variance, taking `40` AdamW steps at flat `lr = 1e-4` along lower-noise fit-fold gradients drives `15.5%` larger parameter drift (`0.50806` vs `0.43980`) and worse out-of-family generalization (`-0.021905` vs `r882`, `-0.005017` vs `CONTROL`).
   - Across all three arms, `fold 0` utility loss, family harm fraction, and continuous regret scale monotonically with adapter L2 drift from `r882` (`COSWR` `0.24585` < `CONTROL` `0.43980` < `FAMBAL_ACC4` `0.50806`).

## 6. Next Step

Every fine-tune-from-`r882` arm (including `CONTROL`) loses `~0.02` on `fold 0`, so further optimizer-schedule waves are low prior; prediction-level blends (`H129`/`H134`/`H135`) are the live lead.
