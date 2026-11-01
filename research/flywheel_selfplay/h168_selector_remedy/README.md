# H168: Prospective Selector Remedy (`H156` `W10` / `W20` Adapter Continuations under `t0058` Dual Inner-`B` Checkpoint Selection)

- **Wave**: `H168` (`research/flywheel_selfplay/h168_selector_remedy/`)
- **Owner**: `rd_argon`
- **Status**: `COMPLETE` (`custody-run-a2` at merged commit `b1686766c27d5c2a452e77d281506507a11e2bff`, review `5727253679`; pre-baseline setup attempts `custody-run` and `custody-run-a1` `INCOMPLETE` with `0` optimizer updates and `0` contexts scored, preserved untouched)
- **Verdict**: `NO_ADVANCE` (`NEW_SELECTOR_NOISE_NEUTRAL` across both `W10` and `W20`; `0 / 4` primary arms pass Holm-Bonferroni / LCB / margin gates; `serving_promotion: false`)
- **Custody Archive**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H168/argon-h168-20261101a/` (`60/60` objects, `4,298,402` bytes restore-verified; local receipt `/home/ubuntu/jope_plan/evidence/H168/s3-custody-receipt.json` and `/home/ubuntu/jope-rescue-20260921/h168-rd_argon-20261101/`)

## Hypothesis & Lineage

- `H156` tested `W10` (`1.0x`, `29,211` trainable adapter params) and `W20` (`2.0x`, `46,595` trainable adapter params, research-only Net2WiderNet initialization from `r882`) under point-positive `XFIT` primary selection and returned `NULL`; in post-hoc strict-eligibility diagnostics (`evidence/redteam-t0058/selector-gate-counts.json`), `59 / 180` non-baseline inner-`B` candidates cleared $\Delta_B(t) > +0.001 + 10^{-12}$, whereas only `4 / 180` (`1` in `W10`, `3` in `W05`, `0` in `W20`) cleared both $\Delta_B(t) > +0.001 + 10^{-12}$ and 41-family paired bootstrap $\text{LCB}_{95,B}(t) > 0$.
- Red-Team audit `t0058` (`evidence/redteam-t0058/selector-power-and-oof.json`) showed via conditional empirical `H156` `W10_XFIT` noise diagnostics on a 41-family inner selector fold $B$ (not refit-training or out-of-sample generalization power) that a planted diffuse $+0.005$ equal-family improvement has `22.0%–62.4%` conditional 41-family admission under the strict `OLD` selector (`53.4%–94.2%` margin pass rate bottlenecked by `LCB > 0`), and that in the fixed single-family sparse control ($+0.205$ on 1 family, $0$ on 40 families, mean $+0.005$), $(40/41)^{41} \approx 36.33\%$ of 41-family bootstrap draws omit the improved family, forcing $\text{LCB}_{95,B} = 0.0$ (`0%` admission for that sparse fixed control).
- `H168` prospectively tested on **fresh seeds** (`168101`, `168102`, `168103`; fresh `fold_seed = 168140`; fresh `boot_seed = 168141`; zero reuse of seen panels) whether replacing the inner-`B` bootstrap LCB filter with the `t0058` **NEW** equal-family mean-utility selector ($\Delta_B(t) > +0.001 + 10^{-12}$ with strict baseline/margin ties, else step `0`) on the **exact same `W10` and `W20` continuation trajectories** (using the exact pinned `H156`/`H146` `runtime.loss`: valid `clip(fitness, -4.0, 2.0)`, invalid `2.0`, SmoothL1 + `0.1` KL at $T=0.2$) unlocks positive out-of-fold `DL-Train` utility or instead admits inner-`B` noise that degrades held-out outer-fold $A$ utility.

## Attempt & Amendment Record

1. **Attempt 1 (`custody-run` at `512897d3c384556ee682ac1618fe941a6da2050f`) — `INCOMPLETE` (`0` optimizer updates, `0` contexts scored)**:
   - Failed before scoring any baseline context (`baseline.json` not created) with `AttributeError: 'FusedJopeNetwork' object has no attribute 'predict_with_10tok_adapter'`. Preserved untouched in `/home/ubuntu/jope-data/h168/custody-run/` and S3 custody.
2. **Attempt A1 (`custody-run-a1` at `ac6221eb5657beee45335b39183290fa1f3c7534`, PR `#1441`, historical `BLOCK` review `5727210306`) — `INCOMPLETE` (`0` optimizer updates, `0` contexts scored, `0` `DL-Train` pools opened)**:
   - Failed in `1`s at `evaluate_uncached(val)` (`baseline.json` not created) with `TypeError: FlywheelBlockFormerAdapterNet.predict_costs() got an unexpected keyword argument 'enable_prior_correction'` (those 4 `enable_*` kwargs belong to `anchor.predict_costs`, not `adapter_10tok.predict_costs`). Preserved untouched in `/home/ubuntu/jope-data/h168/custody-run-a1/` and S3 custody.
3. **Amendment A2 & Single Scientific Run (`custody-run-a2` at `b1686766c27d5c2a452e77d281506507a11e2bff`, PR `#1449`, PASS review `5727253679` at `acd303e64996b585c680deec0dd0b7954eae42d5`) — `COMPLETE` (`1,200` optimizer updates)**:
   - Matched the exact `net.adapter_10tok.predict_costs(...)` call in `h146_lookahead_exec` and `h165_gradient_noise_dropout`, verified via `15/15` unit tests (`test_h168.py`) and 91-context `DL-Val` baseline-scoring `--smoke` (`0` updates, `0` `DL-Train` pool opens), then executed the single full CPU wave (`30` trajectories / `1,200` updates in `1,761.47`s, `nice = 19`, peak RSS `3.45 GB < 6 GB`, `<= 4` aggregate threads, `CUDA_VISIBLE_DEVICES=""`).

## Pre-Training Baseline & Invariance Verification

- **Exact `91/91` `DL-Val` `r882` Baseline Parity**: `matches = 91`, `mean = 1.1377941898164525`, `max_context_error = 0.0`, `mean_error = 0.0` (`PASS`).
- **Native `W10` Score Bit-Identity (`uncached` vs `cached` `total_cost`)**: `91` `DL-Val` contexts, `104,832` candidate scores `bit_identical: true` (`torch.equal` and raw `float32` bytes), `max_abs_cost_diff = 0.0` (`PASS`).
- **Research-Only `W20` (`46,595` params) Step-0 Utility Tolerance**: `w20_step0_val_utility_max_diff = 0.0`, raw `total_cost` `max_abs_diff = 1.52587890625e-05` (`<= 1e-6` utility tolerance `PASS`; explicitly research-only and not native-score bit-exact).
- **Freeze & Input Closure Audit**: `dope_sha256 = c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and `frozen_sha256 = c1baca14dad644e3a2ec9566b2f89e8f912b098f455d29cc8ace9b608ee694c6` invariant across all evaluations and `1,200` optimizer updates (`opened_count = 1,317`, `protected_reads = 0`, `refused = []`, `cuda_initialized = false`).

## Primary Out-of-Fold (`DL-Train`, `1,219` Contexts / `205` Families) & Secondary (`DL-Val`, `91` Contexts / `19` Families) Results

| Arm / Contrast | Non-Zero Cells (`/15`) | `DL-Train` OOF Per-Seed Equal-Family Gain (`168101`, `168102`, `168103`) | `DL-Train` OOF Median Gain `[95% CI]` | Positive Seeds | One-Sided $p$ (`Holm` $p$) | `DL-Train` Median BT Elo `[95% CI]` | `DL-Train` `-2` Invalid Picks (Seeds / Median) | `DL-Val` (91 ctx / 19 fam) Median Gain `[95% CI]` | `DL-Val` Median BT Elo | Verdict |
|---|---:|---|---|---:|---|---:|---|---:|---:|---|
| `r882` (Reference) | `0 / 15` | `[0.0, 0.0, 0.0]` (Fam util `1.385355`, Ctx util `1.423073`) | `0.000000 [0.000000, 0.000000]` | — | — | `1500.00 [1500.00, 1500.00]` | `[17, 17, 17]` (`17.0`) | `0.000000` (Fam `1.153615`, Ctx `1.137794`) | `1500.00` | Reference |
| `W10_OLD - r882` | `1 / 15` | `[+0.000000, +0.000000, -0.001301]` | **`0.000000 [0.000000, 0.000000]`** | `0 / 3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` | `[17, 17, 18]` (`17.0`) | `0.000000 [0.000000, 0.000000]` | `1500.00` | `NO_ADVANCE` |
| `W10_NEW - r882` | `7 / 15` | `[-0.014225, -0.006089, -0.005050]` | **`-0.006089 [-0.020896, +0.004070]`** | `0 / 3` | `0.9164` (`1.0000`) | `1499.28 [1495.81, 1502.39]` | `[20, 20, 19]` (`20.0`) | `-0.003189 [-0.007614, +0.000285]` | `1495.62` | `NO_ADVANCE` |
| `W20_OLD - r882` | `0 / 15` | `[+0.000000, +0.000000, +0.000000]` | **`0.000000 [0.000000, 0.000000]`** | `0 / 3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` | `[17, 17, 17]` (`17.0`) | `0.000000 [0.000000, 0.000000]` | `1500.00` | `NO_ADVANCE` |
| `W20_NEW - r882` | `9 / 15` | `[-0.007748, -0.008614, -0.002407]` | **`-0.007748 [-0.021481, +0.008389]`** | `0 / 3` | `0.8394` (`1.0000`) | `1497.92 [1492.68, 1503.47]` | `[20, 20, 18]` (`20.0`) | `-0.010664 [-0.025335, -0.000157]` | `1487.57` | `NO_ADVANCE` |
| `W10_NEW - W10_OLD` | `7` vs `1` | `[-0.014225, -0.006089, -0.003749]` | **`-0.006089 [-0.020896, +0.004070]`** | `0 / 3` | `0.9205` (Mech.) | — | `+3.0` median `-2` picks | `-0.003189 [-0.007614, +0.000158]` | — | `NEW_SELECTOR_NOISE_NEUTRAL` |
| `W20_NEW - W20_OLD` | `9` vs `0` | `[-0.007748, -0.008614, -0.002407]` | **`-0.007748 [-0.021481, +0.008389]`** | `0 / 3` | `0.8394` (Mech.) | — | `+3.0` median `-2` picks | `-0.010664 [-0.025335, -0.000157]` | — | `NEW_SELECTOR_NOISE_NEUTRAL` |
| `W20_NEW - W10_NEW` | `9` vs `7` | `[+0.006477, -0.002526, +0.002643]` | `+0.002643 [-0.004429, +0.007650]` | `2 / 3` | `0.2913` (Mech.) | — | `20.0` vs `20.0` | `-0.001382 [-0.003298, +0.000127]` | — | Width contrast null |

### Pre-Locked Secondary Diagnostic Selectors (`RELAXED` $\Delta_B > 10^{-12}$ & `NONZERO` $\arg\max_{t \in \{10,20,30,40\}} \Delta_B(t)$)

| Diagnostic Arm | Non-Zero Cells (`/15`) | `DL-Train` OOF Per-Seed Gain | `DL-Train` OOF Median Gain `[95% CI]` | `DL-Train` Median Elo | `DL-Train` `-2` Picks (Median) | `DL-Val` Median Gain `[95% CI]` | `DL-Val` Median Elo |
|---|---:|---|---:|---:|---:|---:|---:|
| `W10_RELAXED - r882` | `9 / 15` | `[-0.014225, -0.008637, -0.005050]` | `-0.008637 [-0.024749, +0.003507]` | `1499.24` | `[20, 19, 19]` (`19.0`) | `-0.005927 [-0.014675, +0.000312]` | `1497.81` |
| `W10_NONZERO - r882` | `15 / 15` | `[-0.014065, -0.008383, -0.008221]` | `-0.008383 [-0.024651, +0.004299]` | `1494.94` | `[20, 19, 19]` (`19.0`) | `-0.013455 [-0.031213, -0.000668]` | `1491.96` |
| `W20_RELAXED - r882` | `10 / 15` | `[-0.007748, -0.015523, -0.002407]` | `-0.007748 [-0.023036, +0.006972]` | `1496.21` | `[20, 21, 18]` (`20.0`) | `-0.012159 [-0.027373, -0.000394]` | `1494.88` |
| `W20_NONZERO - r882` | `15 / 15` | `[-0.007598, -0.015299, -0.000503]` | `-0.007598 [-0.023000, +0.007284]` | `1496.37` | `[20, 21, 17]` (`20.0`) | `-0.014044 [-0.031865, -0.000690]` | `1492.69` |

## Cell-by-Cell Inner-`B` Checkpoint Selections & Step-40 L2 Drift

| Cell (`Outer-Seed`) | `W10` (`OLD`, `NEW`, `RELAXED`, `NONZERO`) | `W20` (`OLD`, `NEW`, `RELAXED`, `NONZERO`) | `W10` Step-40 L2 Drift | `W20` Step-40 L2 Drift |
|---|---|---|---|---|
| `A0-168101` | `(0, 10, 10, 10)` | `(0, 10, 10, 10)` | `0.3620` | `0.4213` |
| `A0-168102` | `(0, 0, 10, 10)` | `(0, 0, 30, 30)` | `0.3717` | `0.4357` |
| `A0-168103` | `(0, 0, 0, 40)` | `(0, 10, 10, 10)` | `0.3516` | `0.4112` |
| `A1-168101` | `(0, 40, 40, 40)` | `(0, 40, 40, 40)` | `0.3901` | `0.4537` |
| `A1-168102` | `(0, 20, 20, 20)` | `(0, 40, 40, 40)` | `0.3472` | `0.4047` |
| `A1-168103` | `(10, 10, 10, 10)` | `(0, 10, 10, 10)` | `0.3760` | `0.4391` |
| `A2-168101` | `(0, 0, 0, 10)` | `(0, 0, 0, 10)` | `0.3775` | `0.4412` |
| `A2-168102` | `(0, 0, 0, 10)` | `(0, 0, 0, 10)` | `0.3724` | `0.4330` |
| `A2-168103` | `(0, 0, 0, 10)` | `(0, 0, 0, 10)` | `0.3878` | `0.4524` |
| `A3-168101` | `(0, 20, 20, 20)` | `(0, 10, 10, 10)` | `0.3647` | `0.4251` |
| `A3-168102` | `(0, 10, 10, 10)` | `(0, 10, 10, 10)` | `0.3830` | `0.4456` |
| `A3-168103` | `(0, 10, 10, 10)` | `(0, 10, 10, 10)` | `0.3738` | `0.4348` |
| `A4-168101` | `(0, 0, 0, 10)` | `(0, 0, 0, 10)` | `0.3733` | `0.4363` |
| `A4-168102` | `(0, 0, 30, 30)` | `(0, 10, 10, 10)` | `0.3690` | `0.4281` |
| `A4-168103` | `(0, 0, 0, 10)` | `(0, 0, 0, 10)` | `0.3907` | `0.4555` |
| **Summary** | `OLD: 1/15`, `NEW: 7/15`, `REL: 9/15`, `NZ: 15/15` | `OLD: 0/15`, `NEW: 9/15`, `REL: 10/15`, `NZ: 15/15` | Median `0.3733` (`[0.3472, 0.3907]`) | Median `0.4357` (`[0.4047, 0.4555]`) |

- **Note on `A0-168102` `W20` (`(0, 0, 30, 30)`)**: In cell `A0-168102` `W20`, step `30` achieved an inner-`B` equal-family gain of `+0.00086208 <= +0.001 + 1e-12` (`LCB95 = -0.06132288`) and a held-out outer-`A` equal-family gain of `-0.03454409` across `41` families; thus step `30` was selected only by the descriptive `RELAXED` / `NONZERO` selectors, while primary `W20_NEW` selected `step = 0`.
- **Gradient Health**: Across all `1,200` optimizer updates (`30` trajectories $\times$ `40` steps), all `47` trainable adapter parameter tensors received active gradients (`none_grad_param_indices = []` in all `30` trajectories).

## Required Limitations & Scope Disclosures

1. **Inner-`B` Raw Context Panel Retention vs Summary-Statistics Verification Scope**:
   - Retained custody locks (`custody-run-a2/run/locks/*.json`) record per-candidate step (`0, 10, 20, 30, 40`) inner-`B` equal-family mean gain, `10,000`-draw paired family bootstrap `95%` CI, one-sided $p$-value, and `OLD`/`NEW`/`RELAXED` eligibility flags, but do **not** retain raw per-context inner-`B` utility matrices across all candidate steps.
   - Consequently, inner-`B` selector bootstrap CIs cannot be independently recomputed from retained `B` context panels; independent verification of the `60` primary and `60` diagnostic selector decisions verifies eligibility and earliest-step tie-breaking against the recorded candidate summary statistics, whereas outer-`A` primary (`3 × 1,219 × 5`) and `DL-Val` secondary (`3 × 5 × 91 × 5`) raw context-utility panels are retained in `primary.npz` and support full independent replay.
2. **Reference Checkpoint Scope (`r882`)**:
   - `round-882.pt` is the preregistered `H`-wave reference checkpoint (not asserted as a verified live-serving incumbent) and has historical exposure to `DL-Train` contexts during earlier flywheel rounds (`historical_r882_train_exposure: true`); results are conditional continuation evidence.
3. **Research-Only `W20` Scope**:
   - `W20` (`2.0x` width, `46,595` trainable adapter parameters via Net2WiderNet initialization) is a research-only capacity diagnostic (`w20_step0_val_utility_max_diff = 0.0`, raw `total_cost` `max_abs_diff = 1.52587890625e-05`) and is not native-score bit-identical; no serving promotion is claimed (`serving_promotion: false`).

## Verdict & Next Step

- **Verdict**: `NO_ADVANCE` (`NEW_SELECTOR_NOISE_NEUTRAL`).
- **Scientific Conclusion & Next Step**: Removing the 41-family paired bootstrap `LCB95 > 0` gate in the `t0058` `NEW` inner-`B` selector (`mean_gain > +0.001 + 1e-12`) dramatically increased non-zero checkpoint admission from `1/15` to `7/15` cells on `W10` and from `0/15` to `9/15` cells on `W20`, directly confirming that `LCB95 > 0` was the binding filter on inner-`B`. However, those admitted non-zero `W10` and `W20` checkpoints degraded held-out outer-fold $A$ equal-family utility (`W10_NEW - r882 = -0.006089 [-0.020896, +0.004070]`, `W20_NEW - r882 = -0.007748 [-0.021481, +0.008389]`, `0/3` positive seeds, increasing median `-2` invalid cliff selections from `17.0` to `20.0`). Thus, under standard `lr=1e-4` AdamW `r882` continuation trajectories, the strict `OLD` `LCB95 > 0` gate was protecting out-of-fold utility against false-positive inner-`B` winners rather than masking true out-of-fold gains. Do not relax `LCB95 > 0` on 41-family inner folds without an explicit gate-invalid (`-2` cliff) veto or a lower-variance continuation objective.

## Reproduction & Verification Commands

```bash
# 1. Data-free unit tests (15/15)
pytest -q research/flywheel_selfplay/h168_selector_remedy/test_h168.py

# 2. 91-context DL-Val baseline-scoring smoke (0 updates, 0 DL-Train pool opens)
python3 -B research/flywheel_selfplay/h168_selector_remedy/monitor.py \
  --custody /home/ubuntu/jope-data/h168/custody --smoke

# 3. Full single-attempt wave (executed under custody-run-a2)
python3 -B research/flywheel_selfplay/h168_selector_remedy/monitor.py \
  --custody /home/ubuntu/jope-data/h168/custody-run-a2
```
