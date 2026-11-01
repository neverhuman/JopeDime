# H171: Large Effective Batch via Family-Stratified Gradient Accumulation `r882` Continuation

- **Wave**: `H171`
- **Owner**: `gpu_fused_agy` (`[FUSED-AGY]`)
- **Authority**: `/home/ubuntu/nudges/hwave_t0205_h171_fused_agy.md`
- **Preregistration**: [PREREG.md](PREREG.md) (PR `#1459`, `@gpu_fused_redteam_codex` `PASS` review `5727448562` at `bcfc3fe677f4ac8859992284646978e1745e35d9`, squash-merged into `main` at `b362b1762350df99989618d51ac9d136a0343948`)
- **Execution Host**: `xbabe2` (`CPU-only`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `<= 4` threads; finished `2026-11-01T10:39Z`)
- **Status / Verdict**: **`COMPLETE — NO_ADVANCE`** (`15/15` cells $\times$ `4/4` arms selected `step = 0` under strict inner-`B` selector; `gradient_noise_diagnostic` confirms strong monotonic gradient-alignment gains `CTRL` `0.8276` $\to$ `ACC4` `0.9489` $\to$ `ACC16` `0.9906` vs `CTRL_EX4` `0.7893`, while `ACC4` and `ACC16` curb `CTRL`'s unconstrained OOF degradation `-0.002056` back to `0.000000` median)

---

## 1. Hypothesis & Fixed Protocol

Small single-minibatch updates ($B = 8$ contexts across `8` fit families) during `40`-step `r882` continuation exhibit substantial inter-step gradient variance across heterogeneous problem families (`123` fit families per cross-fit fold), driving trajectory drift that forces the strict inner-fold `B` selector (`gain > +0.001 + 1e-12` and `95% LCB > 0`) to fall back to `step 0` (`r882`). We hypothesized that accumulating $K \in \{4, 16\}$ family-stratified minibatches per `AdamW(lr=1e-4, weight_decay=1e-4)` update (using the exact mean of accumulated gradients so step scale matches $B = 8$) would increase cosine alignment to the exact equal-family-weighted full-fit-split gradient $g_{\text{full}}(\theta) \in \mathbb{R}^{29211}$ and improve out-of-fold selection and utility.

### Registered Arms & Cross-Fit Design
- **Arms (`4`)**:
  1. `CTRL`: `1` family-stratified minibatch ($1B = 8$ contexts across `8` distinct fit families) per optimizer update for `40` updates (`snapshots 0/10/20/30/40`; `320` examples total).
  2. `ACC4`: accumulate `4` family-stratified minibatches ($4B = 32$ contexts across `32` distinct fit families, exact mean of `4` microbatch gradients) per optimizer update for `40` updates (`snapshots 0/10/20/30/40`; `1,280` examples total).
  3. `ACC16`: accumulate `16` family-stratified minibatches ($16B = 128$ contexts covering all `123/123` fit families per update — `118` singletons + `5` two-context families — exact mean of `16` microbatch gradients) per optimizer update for `40` updates (`snapshots 0/10/20/30/40`; `5,120` examples total).
  4. `CTRL_EX4`: live single-minibatch ($1B = 8$ contexts) updates for `160` optimizer steps on the **exact same `160` minibatches** seen by `ACC4`, with example-matched snapshots at `0/40/80/120/160` mapped to logical stages `0/10/20/30/40` (`1,280` examples total).
- **Cross-Fit Structure**:
  - `3` seeds (`171101, 171102, 171103`) $\times$ `5` disjoint outer assessment folds (`A0..A4`, `41` families each, `fold_seed = 171140`, `boot_seed = 171141`) over `1,219` `DL-Train` contexts (`205` families).
  - For outer fold $k$, inner selector fold `B` is $(k+1) \bmod 5$ (`41` families) and fit folds are the remaining `3` folds (`123` families; `712, 720, 754, 742, 729` fit contexts across folds `0..4`, `0` inner-select `B`, outer-assess `A`, or `DL-Val` contexts ever entering family-stratified minibatch sampling or full-fit-split gradient evaluation).
  - `15` seed-fold cells $\times$ (`40 + 40 + 40 + 160 = 280` updates/cell) = **`4,200` optimizer updates** across `60` trajectories (`240` saved `.npz` checkpoints).
  - **Strict inner selector on `B`**: Select `best_step` $\in \{10, 20, 30, 40\}$ iff equal-family utility gain vs `step 0` (`r882`) on `B` exceeds `+0.001 + 1e-12` **and** its `10,000`-draw family-bootstrap `LCB95 > 0.0`; otherwise select `step 0` (`r882`).

### Methodological & Attribution Disclosures (`PREREG.md` / Review `5727448562`)
1. **Conditional TRAIN-Exposed Scope**: Because `round-882.pt` was trained on the `1,219`-context / `205`-family `DL-Train` pool during the historical flywheel run, all `DL-Train` out-of-fold readouts are conditional continuation evidence rather than unexposed holdout generalization.
2. **Exact Instantaneous Full-Fit Gradient Schedule (`DIAG_STEPS`)**: `CTRL`, `ACC4`, and `ACC16` evaluate the exact equal-family-weighted full-fit-split gradient $g_{\text{full}}(\theta^{(t)})$ (`w_i = 1 / (123 * |fam(i)|)`, $\sum_i w_i = 1$) at **`40/40 = 100%`** of optimizer updates (`600` exact evaluations per arm across `15` cells, `0` lagged rows). `CTRL_EX4` evaluates $g_{\text{full}}(\theta)$ at `41` macro-steps (`{1, 4, 8, ..., 160}`, `615` exact instantaneous evaluations across `15` cells = `25.625%`) and explicitly separates its `1,785` off-grid micro-steps (`cos_to_full_fit_grad = None`, recording `reference_full_fit_update`, `reference_full_fit_lag_updates`, and `cos_to_lagged_macrostep_full_fit_grad`).
3. **`ACC16` Exposure Attribution**: `ACC16` is an update-matched (`40` updates) full-fit-family coverage contrast (`5,120` examples vs `1,280` for `ACC4` and `CTRL_EX4`), whereas `ACC4 vs CTRL_EX4` is the clean example-matched (`1,280` examples) and update-matched (`40` updates vs `CTRL`) isolation of gradient accumulation.

---

## 2. Pre-Training Parity Gates, `gradient_noise_diagnostic`, & Vector-Identity Audit

### Pre-Training Baseline & Static-Forward Parity Gates (`baseline.json` — `PASS`)
- **First-Scored Native `91/91` `DL-Val` Gate (`PASS`)**: Scored before any optimizer update; `91/91` contexts matched `/home/ubuntu/jope-data/h171/inputs/best_dev_checkpoint.json` with `max_abs_error = 0.0`, context mean utility **`1.1377941898164525`**, and `19`-family equal-weight mean utility **`1.153615276991245`**.
- **Forward & Gradient Bit-Exactness (`PASS`)**:
  - `eval_forward_bit_exact = true`: `torch.equal` across all `91` `DL-Val` contexts between `Engine.predict_costs` and `Engine.adapter_forward_static`.
  - `static_grad_parity_bit_exact = true` and `static_grad_parity_perturbed_bit_exact = true`: bit-exact equality (`max_abs_grad_diff = 0.0`) across all `47` trainable adapter tensors (`29,211` parameters: `41` active gradient tensors and `6` structural `None`-gradient tensors zero-filled at fixed offsets) between dynamic `adapter_forward` and cached `adapter_forward_static` at both initial `r882` weights and deterministically perturbed weights.
- **Frozen Backbone Invariance (`PASS`)**: `DOPE` composite SHA-256 (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and frozen non-adapter state SHA-256 (`8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`) remained bitwise identical across all `60` trajectories (`4,200` updates).

### Main Mechanism Diagnostic — `gradient_noise_diagnostic`
Cosine similarity $\cos(g_{\text{step}}(\theta), g_{\text{full}}(\theta))$ in $\mathbb{R}^{29211}$ (`41` active + `6` `None`-as-zero tensors) between each arm's pre-clip update gradient and the exact equal-family-weighted full-fit-split gradient at the **exact same pre-update parameter state $\theta$**:

| Arm | Effective Batch / Coverage | Updates / Cell | Instantaneous Exact Evals (Fraction) | Instantaneous Cosine Median `[Q25, Q75]` | Instantaneous Cosine Mean `[Min, Max]` | Step-1 Cosine to $g_{\text{full}}(\theta_{r882})$ Median `[Min, Max]` | Cosine to Fixed $g_{\text{full}}(\theta_{r882})$ Median (Mean) | Off-Grid Lagged Evals (Median Cosine) | Raw Pre-Clip Grad L2 Norm Median (Mean) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CTRL`** | `1B = 8` ctx (`8` fam) | `40` | `600 / 600` (`100%`) | **`0.827556`** `[0.784425, 0.863637]` | `0.819576` `[0.590007, 0.949869]` | **`0.864896`** `[0.795109, 0.910681]` | `0.825233` (`0.810370`) | `0` (`null`) | `0.534524` (`0.569644`) |
| **`ACC4`** | `4B = 32` ctx (`32` distinct fam) | `40` | `600 / 600` (`100%`) | **`0.948898`** `[0.938397, 0.958681]` | `0.946165` `[0.867141, 0.975269]` | **`0.959976`** `[0.885659, 0.970183]` | `0.944458` (`0.936245`) | `0` (`null`) | `0.507562` (`0.510409`) |
| **`ACC16`** | `16B = 128` ctx (`123/123` fit fam) | `40` | `600 / 600` (`100%`) | **`0.990610`** `[0.989586, 0.991568]` | `0.990396` `[0.983878, 0.994172]` | **`0.990833`** `[0.989626, 0.992802]` | `0.981733` (`0.978189`) | `0` (`null`) | `0.490097` (`0.492809`) |
| **`CTRL_EX4`** | `1B = 8` ctx (`160` live steps) | `160` | `615 / 2,400` (`25.625%`) | **`0.789314`** `[0.711801, 0.842713]` | `0.759439` `[0.027842, 0.946785]` | **`0.864896`** `[0.795109, 0.910681]` | `0.729147` (`0.711447`) | `1,785` (`0.790485`) | `0.814476` (`0.878801`) |

- **Mechanism Verification**: Family-stratified gradient accumulation sharply and monotonically eliminates inter-family gradient noise: median instantaneous cosine to $g_{\text{full}}(\theta)$ rises from **`0.827556`** (`CTRL`, `1B = 8`) to **`0.948898`** (`ACC4`, `4B = 32`, `+0.121342`) and **`0.990610`** (`ACC16`, `16B = 128`, `+0.163054`), while `CTRL_EX4` (`160` unaccumulated `1B = 8` steps on the exact same `160` minibatches as `ACC4`) drifts to higher curvature and lower instantaneous cosine (**`0.789314`**, minimum `0.027842`).

### Adapter L2 Drift (`drift_summary`) & Vector-Identity Audit (`inner_selector_summary`)

| Arm | Step-40 Adapter L2 Drift Median `[Min, Max]` | Drift Ratio vs `CTRL` | Distinct Step-40 Checkpoints vs `CTRL` | Inner-`B` Strict Selected Counts (`0/10/20/30/40`) | Inner-`B` Unconstrained `best_step` Counts (`0/10/20/30/40`) | Inner-`B` Max `best_step` Gain (`Max LCB95`) | `raw_b_identical_cells_to_ctrl` (`raw_b_vectors_identical_to_ctrl`) | `best_steps_identical_to_ctrl` / `family_means_identical_to_ctrl` | `selected_steps_identical_to_ctrl` (`primary` / `secondary` vectors identical) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CTRL`** | **`0.379384`** `[0.365892, 0.387524]` | `1.000x` | `15/15` (self ref) | `15/0/0/0/0` | `11/2/2/0/0` (`4/15 > 0`) | `+0.029204` (`0.0`) | — | — | — |
| **`ACC4`** | **`0.462063`** `[0.450991, 0.470444]` | `1.218x` | `15/15` (`0` hash match) | `15/0/0/0/0` | `13/2/0/0/0` (`2/15 > 0`) | `+0.002177` (`0.0`) | `0 / 15` (`false`) | `false` / `false` | `true` (`true` / `true`) |
| **`ACC16`** | **`0.532508`** `[0.527399, 0.537892]` | `1.404x` | `15/15` (`0` hash match) | `15/0/0/0/0` | `13/1/0/0/1` (`2/15 > 0`) | `+0.005217` (`0.0`) | `0 / 15` (`false`) | `false` / `false` | `true` (`true` / `true`) |
| **`CTRL_EX4`** | **`1.713380`** `[1.648444, 1.801226]` | `4.516x` | `15/15` (`0` hash match) | `15/0/0/0/0` | `15/0/0/0/0` (`0/15 > 0`) | `0.000000` (`0.0`) | `0 / 15` (`false`) | `false` / `false` | `true` (`true` / `true`) |

- **Explicit Vector-Identity & Selector-Neutrality Disclosure**:
  - Unlike small-perturbation waves where trajectories fail to move inner-`B` candidate argmins, **`ACC4`, `ACC16`, and `CTRL_EX4` produce genuinely distinct trajectories, distinct inner-`B` context utility panels in all `15/15` cells (`raw_b_identical_cells_to_ctrl = 0`), distinct `family_means`, and distinct unconstrained `best_step` profiles (`best_steps_identical_to_ctrl = false`)**.
  - Notice why `step-40` L2 drift increases smoothly from `0.3794` (`CTRL`) to `0.4621` (`ACC4`, `1.218x`) and `0.5325` (`ACC16`, `1.404x`): under `AdamW`, higher inter-step gradient directional consistency (`cos = 0.9489` and `0.9906`) reduces step-to-step cancellation in the first-moment accumulator $m_t$, yielding more coherent travel along the full-fit descent direction, whereas `CTRL_EX4` takes `4x` as many optimizer steps (`160` updates) and drifts `4.516x` (`1.7134`).
  - However, even at `cos = 0.9906` (`ACC16`), `40` steps of `AdamW(lr=1e-4)` continuation from `r882` on the `123`-family fit split do not produce a statistically reliable (`LCB95 > 0`) improvement on the `41`-family inner selection fold `B` (`max_best_step_lcb95 = 0.0` across all `60` trajectories). Consequently, the strict selector selects `step = 0` (`r882`) in all `15/15` cells for all `4` arms (`selected_adapter_hashes_identical_to_ctrl = 15`), making the strict primary and secondary utility contrasts identically `0.000000 [0.000000, 0.000000]`.

---

## 3. Primary & Secondary Readouts (Strict Selector — `NO_ADVANCE`)

Under the preregistered strict selector (`gain > +0.001 + 1e-12` and `LCB95 > 0`), all `15/15` cells across all `4` arms selected **`step = 0` (`r882`)** (`median_selected_l2_drift = 0.0`).

| Panel | Contrast / Arm | Selected Step Counts (`0/10/20/30/40`) | Family Mean Utility | Context Mean Utility | Median Seed $\Delta$ | 95% Family Bootstrap CI | Positive Seeds | One-Sided $p$ (`Holm p`) | BT MM Elo vs `r882=1500` (95% CI) | `-2` Gate Cliffs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `r882` (baseline) | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`—`) | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `CTRL - r882` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`—`) | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `ACC4 - r882` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `ACC16 - r882` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `CTRL_EX4 - r882` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `ACC4 - CTRL`, `ACC16 - CTRL`, `CTRL_EX4 - CTRL`, `ACC4 - CTRL_EX4`, `ACC16 - ACC4` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`—`) | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Secondary: `DL-Val` (`91` ctx / `19` fam)** | All `4` arms & all `9` contrasts | `15/0/0/0/0` | `1.153615` | `1.137794` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` | `0.0` |

---

## 4. Descriptive Unconstrained Inner-`B` Best-Step Readouts (`DL-Train` OOF & `DL-Val`)

Evaluating each cell at its unconstrained inner-`B` argmax checkpoint (`best_step` $\in \{0, 10, 20, 30, 40\}$ maximizing equal-family utility on `B` without the `+0.001` or `LCB95 > 0` gate):

| Panel (Unconstrained `best_step`) | Contrast / Arm | Per-Seed Equal-Family $\Delta$ (`171101, 171102, 171103`) | Median Seed $\Delta$ | 95% Family Bootstrap CI | Positive Seeds | One-Sided $p$ (`Holm p`) | BT MM Elo vs `r882=1500` (95% CI) | `-2` Cliffs (`171101/171102/171103`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `CTRL - r882` | `[-0.001816, -0.002056, -0.005182]` | **`-0.002056`** | `[-0.005384, -0.000143]` | `0/3` | `0.9896` (`—`) | `1499.89 [1497.51, 1501.58]` | `18 / 18 / 18` (med `18.0`) |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `ACC4 - r882` | `[-0.002253, +0.000324, 0.000000]` | **`0.000000`** | `[-0.004166, 0.000000]` | `1/3` | `0.9969` (`1.0000`) | `1500.28 [1499.04, 1501.94]` | `18 / 17 / 17` (med `17.0`) |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `ACC16 - r882` | `[+0.000125, 0.000000, -0.009462]` | **`0.000000`** | `[-0.009754, 0.000000]` | `1/3` | `0.9927` (`1.0000`) | `1500.00 [1497.82, 1500.00]` | `17 / 17 / 20` (med `17.0`) |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `CTRL_EX4 - r882` | `[0.000000, 0.000000, 0.000000]` | **`0.000000`** | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` | `17 / 17 / 17` (med `17.0`) |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `ACC4 - CTRL` | `[-0.000437, +0.002380, +0.005182]` | **`+0.002380`** | `[-0.000885, +0.007869]` | `2/3` | `0.3524` (`—`) | — | — |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `ACC16 - CTRL` | `[+0.001942, +0.002056, -0.004280]` | **`+0.001942`** | `[-0.006160, +0.005038]` | `2/3` | `0.3183` (`—`) | — | — |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `ACC4 - CTRL_EX4` | `[-0.002253, +0.000324, 0.000000]` | **`0.000000`** | `[-0.004166, 0.000000]` | `1/3` | `0.9969` (`—`) | — | — |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `ACC16 - ACC4` | `[+0.002378, -0.000324, -0.009462]` | **`-0.000324`** | `[-0.011250, +0.001329]` | `1/3` | `0.8755` (`—`) | — | — |
| **`DL-Val` (`91` ctx / `19` fam)** | `CTRL - r882` | `[-0.000651, -0.001306, -0.002103]` | **`-0.001306`** | `[-0.003251, +0.000268]` | `0/3` | `0.9261` (`—`) | `1494.15 [1474.38, 1510.96]` | `0 / 0 / 0` |
| **`DL-Val` (`91` ctx / `19` fam)** | `ACC4 - r882` | `[-0.000724, -0.001398, 0.000000]` | **`-0.000724`** | `[-0.002294, +0.000110]` | `0/3` | `0.9621` (`1.0000`) | `1492.69 [1470.71, 1507.31]` | `0 / 0 / 0` |
| **`DL-Val` (`91` ctx / `19` fam)** | `ACC16 - r882` | `[-0.001392, 0.000000, -0.004109]` | **`-0.001392`** | `[-0.003305, 0.000000]` | `0/3` | `0.9794` (`1.0000`) | `1494.88 [1469.26, 1505.88]` | `0 / 0 / 0` |
| **`DL-Val` (`91` ctx / `19` fam)** | `CTRL_EX4 - r882` | `[0.000000, 0.000000, 0.000000]` | **`0.000000`** | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` | `0 / 0 / 0` |

### Scientific Synthesis & Recommended Next Step
1. **What `H171` Proved**:
   - Even with near-exact full-fit-split gradients (`ACC16` median cosine **`0.990610`**, IQR `[0.989586, 0.991568]`, covering all `123/123` fit families on every single step), first-order `AdamW` continuation of the pairwise rank-loss objective from `round-882.pt` moves away from the discrete argmin utility optimum on held-out families: `ACC4` and `ACC16` reduce spurious inner-`B` false-positive step selections (`2/15` cells vs `4/15` for `CTRL`, recovering `+0.002380` and `+0.001942` median unconstrained OOF utility over `CTRL`), and `ACC4` (`171102`: `+0.000324`) and `ACC16` (`171101`: `+0.000125`) each achieve one tiny positive unconstrained OOF seed, but neither advances over `r882` (`NO_ADVANCE`).
   - This rules out **minibatch gradient noise / inter-family sampling variance** as the primary bottleneck preventing `r882` continuation gains: when the update direction aligns at `0.9906` with the exact equal-family full-fit gradient and still fails to improve held-out family utility, the mismatch is **objective-to-utility misalignment** (smooth pairwise softplus surrogate vs discrete top-1 candidate utility around a converged checkpoint), not stochastic gradient variance.
2. **Recommended Next Step**: Do not invest further compute in larger batch sizes or variance-reduced SGD/AdamW on the existing pairwise rank-loss objective from `r882`. Any continuation from `r882` must directly target top-1 decision boundaries / gate-cliff avoidance or direct utility-aligned selection rather than reducing gradient variance of the surrogate loss.

---

## 5. Host Telemetry, Independent Recount, & Versioned S3 Custody

- **Runtime & Host Guards (`PASS`)**:
  - `elapsed_seconds = 3,039.07` (`monitor.py` `3,042.95` s, `exit_code = 0`, `reasons = []`), `peak_rss_bytes = 4,594,749,440` B (`4.59 GB < 6.0 GB`), `threads_at_end = 2` (`<= 4`), `protected_reads = 0`, `refused = []`, `cuda_initialized = false`.
- **Independent Recount (`recount.json` — `PASS`)**:
  - Verified all `15` cells, `15` selection locks, `60` selector decisions, `240` `.npz` checkpoints, `4,200` trace rows (`2,415` exact instantaneous + `1,785` lagged off-grid), and exact match (`true`) on all `primary`, `secondary`, and `descriptive_unconstrained` contrasts plus `arm_identity_audit`.
- **Versioned S3 Custody ([`custody.json`](custody.json) — `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H171/gpu_fused_agy-t0205/`)**:

| Object | S3 `VersionId` | Size (Bytes) | SHA-256 (`readback_verified = true`) |
| :--- | :--- | :---: | :--- |
| `h171-evidence-bundle.tar.gz` (all `15` cells, `240` `.npz` checkpoints, traces, locks) | `l_mv6VvvP1jZ76hO5PWjtBb49W1zyGeo` | `33,393,659` | `86704320032b06ad6fde290f0e30ba9c62f1438f1c9a54806d927324db626a60` |
| `metrics.json` (raw full `1,331`-open audit) | `vlfVmeJ4dlj6rLv0QZEOBvHbeEJNgdAz` | `208,246` | `e07a09ed7ac1baa5524285b0e90abcfbc4fb370c7cd67733474588143d80da28` |
| `panels.npz` | `b5oSAH4yFSM.UeB17S6K_uRzJBFJDvoS` | `66,095` | `168a48fc5347d47169c5658bfbe26b52e4b427b80c665fc601def7d301192c98` |
| `audit.json` | `v9OFOU4LvM9Op25Uyl9SkTVLFUu13CPV` | `152,636` | `13439965075c72b44dff10e2ede210e6f4408443b99961998181c2d7944a630c` |
| `baseline.json` | `AH_okn5OTN2ysfcmWHUVWDdj4vjoNCk6` | `10,781` | `64fa9f657ff635a335d6e88d0cca4e29abc152b8bbeb71c28e99ae7ea06b3093` |
| `recount.json` | `jMfSiFyFvktNmrFJbsGv22.jtUfDSS3C` | `826` | `ed2bbbc8a5cc3b76849457f3f896822947ef52d5c76e017638b97dfa407a2a97` |

### Reproduction & Unit Verification
```bash
PYTHONPATH=research/flywheel_selfplay/h171_grad_accum pytest -q research/flywheel_selfplay/h171_grad_accum/test_h171.py
```
