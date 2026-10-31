# H126 — Robust Per-Family Gradient Aggregation (`TRIM` & `TRIM_CLIP` vs `CONTROL`)

**NO_ADMISSION**. CPU-only offline `r882` fine-tuning under [PREREG.md](PREREG.md), preregistered in [PR #1097](https://github.com/neverhuman/JopeDime/pull/1097) (`12549fc659`) before calibration or execution. Status: `COMPLETE`; actual optimizer updates: `360` (`9` cells $\times$ `40` AdamW updates). No serving promotion.

## Hypothesis & Setup

- **Hypothesis**: In standard `CONTROL` (`r882` recipe: `batch_size = 16`, `slate_size = 192`, `20` rounds $\times$ `2` AdamW steps/round = `40` updates, `lr = 1e-4`, `weight_decay = 1e-4`), a single outlier family in a 16-context minibatch can dominate the `29,211`-parameter `adapter_10tok` update. Computing one `29,211`-parameter gradient vector per training family in the minibatch and aggregating coordinate-wise via a `20%`-each-tail trimmed mean across the $M_t$ unique families in step $t$ ($k_t = \lfloor 0.20 M_t \rfloor$, `TRIM`), optionally combined with global gradient-norm clipping at the frozen `CONTROL` calibration median norm (`TRIM_CLIP`), was hypothesized to improve held-out family utility on `DL-Train` `fold 0`.
- **Partition & Pairing**: Fresh 5-fold family-grouped split (`fold_seed = 126070`, `fold_manifest_sha256 = ab8abd3319c42b5a462653c2ca3cd226b99faeb96658aaae0e58294ea3d28117`):
  - **Assessment (`fold 0`)**: `41` held-out families (`211` contexts), disjoint from calibration and training.
  - **Fit (`folds 1..4`)**: `164` training families (`1,008` contexts).
  - **Seeds**: `70, 71, 72`, paired across `CONTROL`, `TRIM`, and `TRIM_CLIP` (`3 x 3 = 9` cells). Every cell starts from its own fresh SHA-verified copy of `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`); `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) is frozen and verified before and after every cell.

## Pre-Update Gates

1. **Unchanged-`r882` Regression-First Gate (`PASS`, `0` updates)**:
   - Two independent fresh `r882` copies achieved `0.0` max absolute prediction difference across all `211` `fold 0` assessment pools (`r882_fold0_unrev_family_utility = 1.087381942794`, `r882_fold0_unrev_context_utility = 1.264643806643`, `r882_fold0_native_family_utility = 1.128351062481`).
   - Single-minibatch `CONTROL` family-weighted gradient sum matched direct `loss.backward()` within float32 precision (`max_abs_diff = 9.31e-08`; the `5,665` `adapter_10tok.parent_delta_net.*` parameters unused by `predict_costs` retain `p.grad = None`, and all `23,546` active parameters match).
   - Receipt: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H126/h126_cpu_run_20261031T1635Z/regression_receipt.json` (`VersionId = CtYvvFO7WEii_DjwSh5eM_BvIxKFouN0`).
2. **Frozen `CONTROL` Calibration (`PASS`, `0` updates)**:
   - Measured across the `40` preregistered `seed = 70` fit-fold minibatches on unchanged `r882` without calling `optimizer.step()` (`adapter_mutated = false`, `adapter_sha256 = 740174e9f0d29b69ea1b3dad2e4545e40eff541501e6bfdec1e79dbe96912183` before and after):
     - `median_control_grad_norm = 0.427995637059` (`mean = 0.481397198141`)
     - `effective_trim_clip_threshold = min(1.0, 0.427995637059) = 0.427995637059`
   - Receipt: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H126/h126_cpu_run_20261031T1635Z/calibration_manifest.json` (`VersionId = HaUyiz7j_M4f0Kdf.Uy_27FQXG0F2hh3`).

## Outcome & Metrics Summary

Unchanged `r882` `fold 0` assessment utility: **`1.087382`** (`41` families) / **`1.264644`** (`211` contexts). Full per-cell and bootstrap metrics are in [metrics.json](metrics.json).

### Primary Endpoint: Paired Per-Family Utility Delta vs `CONTROL` (`fold 0`, `41` Families)

| Arm | Seed `fold 0` family utilities (`70, 71, 72`) | Median seed gain vs `CONTROL` | Family-bootstrap `LCB95` (`95% CI`) | Pooled paired Elo vs `CONTROL=1500` (`95% CI`) | BT Elo (`CONTROL=1500`) | Max seed family harm ($\Delta < -0.01$) | Holm ($\alpha=0.05$) | Verdict |
|---|---|---:|---|---|---:|---:|---|---|
| `CONTROL` | `1.040060, 1.047023, 1.042355` | `0.000000` | n/a | `1500.0` (ref) | `1500.00` | n/a | ref | BASELINE |
| `TRIM` | `1.032454, 1.035166, 1.020042` | `-0.011857` | `-0.039564` (`[-0.046070, -0.001372]`) | `1491.52` (`[1443.00, 1539.79]`) | `1498.38` | `17.07%` (`7/41`) | fail (`p=0.9905 > 0.025`) | **NO_ADMISSION** |
| `TRIM_CLIP` | `1.032454, 1.035166, 1.020042` | `-0.011857` | `-0.039564` (`[-0.046070, -0.001372]`) | `1491.52` (`[1443.00, 1539.79]`) | `1498.38` | `17.07%` (`7/41`) | fail (`p=0.9905 > 0.050`) | **NO_ADMISSION** |

### Co-Readout: Paired Per-Family Utility Delta vs Unchanged `r882` (`fold 0`, `41` Families)

| Arm | Median seed gain vs `r882` | Family-bootstrap `LCB95` (`95% CI`) | Pooled paired Elo vs `r882=1500` (`95% CI`) | BT Elo (`r882=1500`) | Max seed family harm ($\Delta < -0.01$) |
|---|---:|---|---|---:|---:|
| `CONTROL` | `-0.045027` | `-0.103962` (`[-0.117743, +0.006028]`) | `1448.79` (`[1369.96, 1528.31]`) | `1496.47` | `31.71%` (`13/41`) |
| `TRIM` | `-0.054928` | `-0.124118` (`[-0.140034, -0.004984]`) | `1463.14` (`[1376.40, 1548.33]`) | `1494.85` | `34.15%` (`14/41`) |
| `TRIM_CLIP` | `-0.054928` | `-0.124118` (`[-0.140034, -0.004984]`) | `1463.14` (`[1376.40, 1548.33]`) | `1494.85` | `34.15%` (`14/41`) |

### Diagnostic Findings

1. **Why `TRIM` underperformed `CONTROL`**: Across `16`-context minibatches (`14–16` unique families per step, $k_t = 2$ or $3$ trimmed per tail), coordinate-wise `20%`-each-tail trimming discarded high-magnitude informative gradients on discriminative adapter coordinates, reducing `fold 0` family mean utility in all three seeds (`-0.007606`, `-0.011857`, `-0.022313` vs `CONTROL`) and harming `14.63%–17.07%` of assessment families (`6–7` of `41` families with $\Delta < -0.01$, `26–27` families tied).
2. **Why `TRIM` and `TRIM_CLIP` tied on `fold 0` utility despite distinct trained weights**: Coordinate-wise family trimming already shrank the median pre-clip gradient norm from `0.4748..0.4761` (`CONTROL`) down to `0.3876..0.4081` (`TRIM`), below the frozen `CONTROL` calibration median threshold (`0.427996`). Consequently, the `0.427996` clip in `TRIM_CLIP` only clipped occasional upper-tail steps, producing slightly smaller L2 parameter drift from `r882` (`0.45357, 0.44547, 0.43152` vs `0.45594, 0.44676, 0.43282`) and distinct `final_adapter_sha256` hashes in every seed without changing any top-4 candidate argmax on the `211` `fold 0` assessment pools.
3. **Why `40`-step unanchored AdamW from `r882` regresses vs unchanged `r882`**: Consistent with `H125`, `40` unconstrained AdamW steps at `lr = 1e-4` drift `0.43–0.46` L2 units away from `r882` (`CONTROL` median `-0.045027` vs `r882`). Neither per-family coordinate trimming nor norm clipping prevents this adapter drift.

## Resource & Governance Audit

- **CPU-only execution (`xbabe2`)**: `nice 15`, `4` Torch/math threads (`OMP/MKL/NUMEXPR=4, OPENBLAS=1`), `0.0` GPU seconds, no `qualification.lock`, `0` reads of `DL-Val`, outer-test, sealed meta-test, or live-loop state.
- **Observed bounds**: elapsed `1,250.58 s`, peak RSS `4,802.52 MiB` (`5.036 GB <= 6 GB`), `MemAvailable` `80.28–83.84 GiB` (`>= 30 GB`), `load1` `33.12–35.26` (`<= 86.7`), disk `91.86%–92.00%` (`<= 95%`).
- **Create-only S3 receipts (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H126/h126_cpu_run_20261031T1635Z/`)**:
  - `h126_summary.json`: `VersionId = k3U_XLd4YCLryGRofDtm16bhFUxWN_gu`, `sha256 = 078f256906414aecded481b8e88476f58de7a8464708566f29981ee30772b19e`
  - `regression_receipt.json`: `VersionId = CtYvvFO7WEii_DjwSh5eM_BvIxKFouN0`, `sha256 = d4e3fc9e027d377a0fd75f1cc7f918ce9801343540e12fcd33752d5b49982e63`
  - `calibration_manifest.json`: `VersionId = HaUyiz7j_M4f0Kdf.Uy_27FQXG0F2hh3`, `sha256 = c217df2e63ef55f6ef2b18a43114ecbf3c918a86869a90eb7aaca08a9be0fd8b`
  - `run_h126_cpu.py`: `VersionId = KWR_1XnxRCA__HmhQRP6JCkuJCu_CRAV`, `sha256 = 7be10d3ac7474e71810e4d37189e9fe6bae738ead7781b754cafa00cc6d72b70`
  - `fold_manifest_126070.json`: `VersionId = pZgsRJrjxIbdJdYNziXalw_ii7rhEhoI`, `sha256 = ab8abd3319c42b5a462653c2ca3cd226b99faeb96658aaae0e58294ea3d28117`
  - Per-cell receipts `cells/{CONTROL,TRIM,TRIM_CLIP}_seed{70,71,72}.json`: all `9` uploaded create-only with verified `VersionId`s recorded in [metrics.json](metrics.json).

## Verdict & Next Step

- **Verdict**: `NO_ADMISSION` (`TRIM` and `TRIM_CLIP` rejected; `r882` remains the unpromoted reference).
- **Next step**: Do not adopt coordinate-wise per-family gradient trimming (`TRIM` / `TRIM_CLIP`) in the adapter training recipe; prioritize explicit `r882`-anchored or subspace-constrained updates rather than unanchored minibatch gradient trimming.
