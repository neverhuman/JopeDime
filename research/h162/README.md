# H162: Cosine Warm-Restart LR Schedule with Snapshot Selection vs LR-Matched Constant Control

- **Status**: `COMPLETE`
- **Verdict**: `NO_ADVANCE` (`rule_neutral = true`: all `15/15` cells × `4/4` arms selected `step = 0` under the preregistered strict inner-selector gate; `serving_promotion = false`)
- **Preregistration Commit**: `52be208096d00d0201d1b19703df25de1a85f085` ([PR #1365](https://github.com/neverhuman/JopeDime/pull/1365), [PREREG.md](PREREG.md))
- **Seeds**: `162101`, `162102`, `162103` (`fold_seed = 162140`, `boot_seed = 162141`)
- **Host / Audit**: `xbabe2` CPU (`nice 19`, peak `2` threads `<= 4`, peak RSS `3.34 GB < 6.0 GB`, `gpu_lock_held = false`, `cuda_initialized = false`, `2,400` AdamW updates across `15` cells, `0` protected/sealed/holdout reads, `0` refused paths)

---

## 1. Hypothesis

Continuation fine-tuning from `r882` with a constant learning rate (`1e-4`) has failed to produce an admitted improvement across recent waves, while `H151` showed that effective step-size differences can confound optimizer comparisons. Cosine warm restarts (`SGDR`) alternate exploration (`peak_lr = 2e-4`) with annealing (`min_lr = 1e-6`), producing cycle-end candidate checkpoints at each restart minimum. We tested whether cycle-end snapshot selection under `2`-cycle (`COS_R2`, `2 × 20` steps) or `4`-cycle (`COS_R4`, `4 × 10` steps) cosine warm restarts improves out-of-fold equal-family utility over both frozen `r882` and a mean-LR-matched constant control (`CTRL_LRMATCH`, `lr = 1.005e-4`).

---

## 2. Setup

- **Data & Folds**: `1,219` contexts across `205` `H145` real-S3 `DL-Train` families (`fold_seed = 162140`: `3` fit folds = `123` families, `1` inner selector fold $B_k$ = `41` families, `1` outer assessment fold $A_k$ = `41` families) + `91`-context / `19`-family `DL-Val` descriptive readout.
- **Model & Invariants**: Owned copy of `r882` (`/home/ubuntu/jope-data/h162/inputs/round-882.pt`, SHA256 `0084479a...`); only the `29,211`-parameter `adapter_10tok` trains (`AdamW`, `wd = 1e-4`, grad norm clip `1.0`). Frozen backbone (`8d1070fa...`) and `DOPE` (`c51a56d0...`) verified unchanged after every step and evaluation.
- **Exact `r882` Baseline Gate (`PASS`)**: `91/91` `DL-Val` contexts matched pinned reference with `max_context_error = 0.0`, `mean_error = 0.0`, `mean = 1.1377941898164525`.
- **Arms (`40` AdamW updates per cell, shared minibatch stream per `(outer, seed)`)**:
  - `CTRL`: constant `lr = 1.000e-4`, snapshots `{0, 10, 20, 30, 40}`
  - `COS_R2`: `2` cosine cycles of `20` steps (`peak = 2e-4`, `min = 1e-6`, exact mean `1.005e-4`), cycle-end snapshots `{0, 20, 40}`
  - `COS_R4`: `4` cosine cycles of `10` steps (`peak = 2e-4`, `min = 1e-6`, exact mean `1.005e-4`), cycle-end snapshots `{0, 10, 20, 30, 40}`
  - `CTRL_LRMATCH`: constant `lr = 1.005e-4` (exact step-mean LR of `COS_R2` and `COS_R4`), snapshots `{0, 10, 20, 30, 40}`
- **Preregistered Strict Inner Selector (`select` fold $B_k$ only)**:
  - Selects the snapshot step $s > 0$ with largest equal-family gain $\hat{\Delta}_s$ satisfying $\hat{\Delta}_s > +0.001 + 10^{-12}$ AND `1,000`-draw family bootstrap $\text{LCB}_{95}(s) > 0$ (ties within $10^{-12}$ rejected); otherwise falls back to `step = 0` (`r882`).

---

## 3. Outcome & Primary Preregistered Metrics

Across all `15` `(outer, seed)` cells (`60` arm trajectories, `225` non-zero snapshot evaluations on the `41`-family inner selector fold $B_k$), **no non-zero snapshot achieved a positive `95%` family bootstrap LCB** (`max` inner-selector $\text{LCB}_{95}$ was `-0.000304` for `CTRL`, `-0.008624` for `COS_R2`, `-0.000514` for `COS_R4`, and `-0.000310` for `CTRL_LRMATCH`, even on cells where inner point gain reached `+0.030191`). Consequently, the preregistered strict inner selector chose `step = 0` (`r882`) on `15/15` cells for all `4` arms (`rule_neutral = true`).

### Primary Cohort: `1,219`-Context / `205`-Family `H145` Train-Only Out-of-Fold Assessment (`3` Seeds)
*(Note: `r882` has historical in-sample `DL-Train` exposure.)*

| Arm | Selected Steps (`15` cells) | Equal-Family Mean | Median-Seed Equal-Family $\Delta$ vs `r882` (`95%` CI) | $\Delta$ vs `CTRL_LRMATCH` (`95%` CI) | Positive Seeds | Holm $p$ ($m=4$) | Elo vs `r882` (`1500.00`) | `-2` Invalid Count (Median / Per-Seed) | Step-40 L2 Drift (Median `[Min, Max]`) | Verdict |
| --- | :---: | ---: | --- | --- | :---: | ---: | ---: | :---: | --- | :---: |
| `R882` | — | `1.385355` | `0.000000` | — | — | — | `1500.00` | `17.0` (`[17, 17, 17]`) | `0.0000` | Baseline |
| `CTRL` | `0` (`15/15`) | `1.385355` | `0.000000` (`[0.000000, 0.000000]`) | `0.000000` (`[0.000000, 0.000000]`) | `0/3` | `1.0000` | `1500.00` | `17.0` (`[17, 17, 17]`) | `0.3750` (`[0.3468, 0.3897]`) | `NO_ADVANCE` |
| `COS_R2` | `0` (`15/15`) | `1.385355` | `0.000000` (`[0.000000, 0.000000]`) | `0.000000` (`[0.000000, 0.000000]`) | `0/3` | `1.0000` | `1500.00` | `17.0` (`[17, 17, 17]`) | `0.3773` (`[0.3443, 0.4032]`) | `NO_ADVANCE` |
| `COS_R4` | `0` (`15/15`) | `1.385355` | `0.000000` (`[0.000000, 0.000000]`) | `0.000000` (`[0.000000, 0.000000]`) | `0/3` | `1.0000` | `1500.00` | `17.0` (`[17, 17, 17]`) | `0.3772` (`[0.3417, 0.3972]`) | `NO_ADVANCE` |
| `CTRL_LRMATCH` | `0` (`15/15`) | `1.385355` | `0.000000` (`[0.000000, 0.000000]`) | `0.000000` (`[0.000000, 0.000000]`) | `0/3` | `1.0000` | `1500.00` | `17.0` (`[17, 17, 17]`) | `0.3770` (`[0.3486, 0.3917]`) | `NO_ADVANCE` |

### Secondary Descriptive Cohort: `91`-Context / `19`-Family `DL-Val` Set (`5`-Fold Policy Average)

| Arm | Raw Context Mean | Equal-Family Mean | Median-Seed Equal-Family $\Delta$ vs `r882` (`95%` CI) | Elo vs `r882` (`1500.00`) | `-2` Invalid Count (Median / Per-Seed) | Verdict |
| --- | ---: | ---: | --- | ---: | :---: | :---: |
| `R882` | `1.1377942` | `1.153615` | `0.000000` | `1500.00` | `0.0` (`[0, 0, 0]`) | Baseline |
| `CTRL` | `1.1377942` | `1.153615` | `0.000000` (`[0.000000, 0.000000]`) | `1500.00` | `0.0` (`[0, 0, 0]`) | `NO_ADVANCE` |
| `COS_R2` | `1.1377942` | `1.153615` | `0.000000` (`[0.000000, 0.000000]`) | `1500.00` | `0.0` (`[0, 0, 0]`) | `NO_ADVANCE` |
| `COS_R4` | `1.1377942` | `1.153615` | `0.000000` (`[0.000000, 0.000000]`) | `1500.00` | `0.0` (`[0, 0, 0]`) | `NO_ADVANCE` |
| `CTRL_LRMATCH` | `1.1377942` | `1.153615` | `0.000000` (`[0.000000, 0.000000]`) | `1500.00` | `0.0` (`[0, 0, 0]`) | `NO_ADVANCE` |

---

## 4. Post-Hoc Descriptive Diagnostics (Unconstrained Positive-Gain Selector & Final `Step = 40` Snapshot)

To measure underlying trajectory behavior when the strict `LCB > 0` and `+0.001` inner-selector gates are removed, we evaluated the saved cycle-end snapshots under (a) the **unconstrained positive-gain inner selector** ($\hat{\Delta}_s > 10^{-12}$ on selector fold $B_k$, picking non-zero snapshots on `7/15` cells for `CTRL`, `3/15` for `COS_R2`, `8/15` for `COS_R4`, and `7/15` for `CTRL_LRMATCH`) and (b) the **fixed `step = 40` final cycle-end checkpoint**:

| Diagnostic Policy | Cohort | Arm | Equal-Family $\Delta$ vs `r882` (`95%` CI) | $\Delta$ vs `CTRL_LRMATCH` | Positive Seeds | Elo vs `r882` (`1500.00`) | `-2` Invalid Count (Median) |
| --- | --- | --- | --- | ---: | :---: | ---: | ---: |
| **Unconstrained Inner Pick** ($\hat{\Delta}_{\text{sel}} > 0$) | `H145` Train (`1,219` ctx / `205` fam) | `CTRL` | `-0.001961` (`[-0.005425, +0.000511]`) | `0.000000` | `0/3` | `1497.40` | `18.0` |
| **Unconstrained Inner Pick** ($\hat{\Delta}_{\text{sel}} > 0$) | `H145` Train (`1,219` ctx / `205` fam) | `COS_R2` | `-0.004978` (`[-0.012862, -0.000451]`) | `-0.003016` | `0/3` | `1494.92` | `19.0` |
| **Unconstrained Inner Pick** ($\hat{\Delta}_{\text{sel}} > 0$) | `H145` Train (`1,219` ctx / `205` fam) | `COS_R4` | `-0.005961` (`[-0.015062, -0.000255]`) | `-0.003999` | `0/3` | `1496.98` | `19.0` |
| **Unconstrained Inner Pick** ($\hat{\Delta}_{\text{sel}} > 0$) | `H145` Train (`1,219` ctx / `205` fam) | `CTRL_LRMATCH` | `-0.001961` (`[-0.005526, +0.000511]`) | `0.000000` | `0/3` | `1497.40` | `18.0` |
| **Unconstrained Inner Pick** ($\hat{\Delta}_{\text{sel}} > 0$) | `DL-Val` (`91` ctx / `19` fam) | `CTRL` | `-0.001978` (`[-0.005421, +0.000642]`) | `0.000000` | `0/3` | `1502.20` | `0.0` |
| **Unconstrained Inner Pick** ($\hat{\Delta}_{\text{sel}} > 0$) | `DL-Val` (`91` ctx / `19` fam) | `COS_R2` | `-0.000731` (`[-0.002176, +0.000418]`) | `+0.001248` | `0/3` | `1505.12` | `0.0` |
| **Unconstrained Inner Pick** ($\hat{\Delta}_{\text{sel}} > 0$) | `DL-Val` (`91` ctx / `19` fam) | `COS_R4` | `-0.003241` (`[-0.007630, +0.000330]`) | `-0.001263` | `0/3` | `1486.82` | `0.0` |
| **Unconstrained Inner Pick** ($\hat{\Delta}_{\text{sel}} > 0$) | `DL-Val` (`91` ctx / `19` fam) | `CTRL_LRMATCH` | `-0.001978` (`[-0.005486, +0.000655]`) | `0.000000` | `0/3` | `1502.20` | `0.0` |
| **Fixed `Step = 40`** (Final Cycle End) | `H145` Train (`1,219` ctx / `205` fam) | `CTRL` | `-0.031721` (`[-0.060708, -0.003425]`) | `-0.000001` | `0/3` | `1484.28` | `21.0` |
| **Fixed `Step = 40`** (Final Cycle End) | `H145` Train (`1,219` ctx / `205` fam) | `COS_R2` | `-0.028960` (`[-0.057897, -0.001438]`) | `+0.002761` | `0/3` | `1484.63` | `22.0` |
| **Fixed `Step = 40`** (Final Cycle End) | `H145` Train (`1,219` ctx / `205` fam) | `COS_R4` | `-0.029831` (`[-0.058712, -0.002521]`) | `+0.001889` | `0/3` | `1484.47` | `21.0` |
| **Fixed `Step = 40`** (Final Cycle End) | `H145` Train (`1,219` ctx / `205` fam) | `CTRL_LRMATCH` | `-0.031720` (`[-0.061818, -0.004770]`) | `0.000000` | `0/3` | `1484.25` | `21.0` |

---

## 5. Verdict & Next Step

- **Verdict**: `NO_ADVANCE` (`serving_promotion = false`).
  1. Cosine warm restarts (`COS_R2`, `COS_R4`) match the L2 parameter drift of the mean-LR-matched constant control (`CTRL_LRMATCH`: `0.3770` vs `0.3773` / `0.3772` at `step = 40`) and slightly reduce step-40 degradation (`+0.0028` / `+0.0019` vs `CTRL_LRMATCH` on `H145` train-only), but all non-zero gradient trajectories still increase `-2` invalid gate-cliff picks (`17 -> 18..22`) and lose utility relative to frozen `r882`.
  2. The preregistered strict `LCB > 0` inner-selector gate functioned as intended: it prevented false-positive inner-fold selections that would have degraded out-of-fold utility by `-0.0020` to `-0.0060`.
- **Next Step**: Do not pursue further learning-rate schedule variations on unconstrained `rt.loss` continuation from `r882`; focus future continuation work on explicit gate-cliff preservation or non-gradient / trust-region combination with `r882`.

---

## 6. Durable Evidence & Custody

- **In-repo summary**: [`research/h162/metrics.json`](metrics.json)
- **Host custody**: `/home/ubuntu/jope_plan/evidence/H162/attempt1/{launch.json,smoke-monitor.json,run-monitor.json,run.log,s3-inventory.txt}`
- **S3 custody (`305` run objects + `5` evidence objects verified)**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H162/gpu_tokbench_agy-h162-20261101/`
