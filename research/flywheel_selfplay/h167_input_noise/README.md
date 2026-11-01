# H167: Adapter Input-Feature Gaussian Noise Injection `r882` Continuation

- **Wave**: `H167`
- **Owner**: `gpu_fused_agy` (`[FUSED-AGY]`)
- **Authority**: `/home/ubuntu/nudges/hwave_t0125_h167_fused_agy.md` + `ADDENDUM 07:26Z` + `/home/ubuntu/nudges/hwave_t0120_h167close_fused_agy.md`
- **Preregistration**: [PREREG.md](PREREG.md) (PR `#1422`, `@gpu_fused_redteam_codex` `PASS` review `5727004397` at `3e4e21603e80aa286efb75db4f86b7b8bbcd0007`, squash-merged into `main` at `fa3c84b6875421aee8904415195d99907608f5a0`)
- **Execution Host**: `xbabe2` (`CPU-only`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `<= 4` threads; finished `2026-11-01T08:21Z`)
- **Status / Verdict**: **`COMPLETE - NO_ADVANCE`** (`RULE_NEUTRAL` under strict inner selector; **uninformative-equal-to-`CTRL` at $\tau \le 0.10 s_j$**)

---

## 1. Hypothesis & Fixed Protocol

Training with additive zero-mean Gaussian input noise $\tilde{x}_j = x_j + \sigma_j \varepsilon_j$, $\varepsilon_j \sim \mathcal{N}(0, 1)$, $\sigma_j = \tau \cdot s_j$ on the `10x32` frozen-DOPE multimodal tokens (`320` features) feeding `adapter_10tok` (`29,211` trainable parameters across `47` tensors; frozen DOPE composite SHA-256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) induces a data-dependent Tikhonov / Jacobian-norm penalty $\frac{1}{2}\sum_j \sigma_j^2 (\partial f / \partial x_j)^2$ (Bishop 1995) that we hypothesized would curb high-curvature logit spikes on rare modality configurations during `40`-step `r882` continuations.

### Registered Arms & Cross-Fit Design
- **Arms (`4`)**:
  1. `CTRL`: $\tau = 0.0$ (exact unperturbed `r882` continuation under common random numbers).
  2. `IN05`: $\tau = 0.05$ dynamic per-step i.i.d. Gaussian noise scaled by fit-fold feature standard deviation $s_j$.
  3. `IN10`: $\tau = 0.10$ dynamic per-step i.i.d. Gaussian noise scaled by fit-fold $s_j$.
  4. `IN10_FIXED`: $\tau = 0.10$ static per-context Gaussian offset drawn once at step `0` (seeded by `(seed, fold_idx, 167999)`) and held fixed across all `40` updates whenever that fit-fold context is sampled, isolating time-varying stochastic smoothing (`IN10`) from static point-cloud jitter (`IN10_FIXED`).
- **Cross-Fit Structure**:
  - `3` seeds (`167101, 167102, 167103`) $\times$ `5` disjoint outer assessment folds (`A0..A4`, `41` families each, `fold_seed = 167140`, `boot_seed = 167141`) over `1,219` `DL-Train` contexts (`205` families).
  - For outer fold $k$, inner selector fold `B` is $(k+1) \bmod 5$ (`41` families) and fit folds are the remaining `3` folds (`123` families).
  - `15` seed-fold cells $\times$ `4` arms = `60` trajectories $\times$ `40` AdamW updates (`lr = 1e-4`, `weight_decay = 1e-4`, `2` microbatches of `8` contexts per update) = **`2,400` optimizer updates** (`240` saved step snapshots at `steps = {10, 20, 30, 40}`).
  - **Strict inner selector on `B`**: Select `best_step` $\in \{10, 20, 30, 40\}$ iff equal-family utility gain vs `step 0` (`r882`) on `B` exceeds `+0.001 + 1e-12` **and** its `10,000`-draw family-bootstrap `LCB95 > 0.0`; otherwise select `step 0` (`r882`).

### Scope Clarifications (`@gpu_fused_redteam_codex` Review `5727004397`)
1. **Conditional TRAIN-Exposed Scope**: Because `round-882.pt` was produced by the historical flywheel trajectory over the `1,219`-context / `205`-family `DL-Train` pool, all `DL-Train` out-of-fold readouts constitute conditional continuation evidence rather than unexposed holdout generalization.
2. **Fit-Fold Candidate Pooling for $s_j$**: Per-feature standard deviations $s_j$ (`ddof=0`) are computed over all `N_fit * 24` unrevealed candidate rows in the `3` outer fit folds (`0` inner-select `B`, outer-assess `A`, or `DL-Val` rows). Since every context has exactly `24` unrevealed candidates, contexts are weighted equally and $s_j$ matches the exact token distribution entering `adapter_10tok`.
3. **`IN10_FIXED` Static Perturbation Indexing**: `IN10_FIXED` pre-generates a static tensor `z_fixed[ctx_pos, cand, slot, dim]` over `[N_fit, 24, 10, 32]` indexed by fit-fold position `0..N_fit-1` and reuses each context's slice whenever that context is sampled into a minibatch, whereas `IN05` and `IN10` draw fresh i.i.d. perturbations on every forward pass.

---

## 2. Baseline Verification, `None`-Gradient Guard (`07:26Z` Addendum), & Dose / Argmin-Invariance Disclosure

- **First-Scored Native `91/91` `DL-Val` Gate (`PASS`)**:
  - Scored before any optimizer update: `91/91` contexts matched `/home/ubuntu/jope-data/h167/inputs/best_dev_checkpoint.json` with `max_abs_error = 0.0`, context mean utility **`1.1377941898164525`**, and `19`-family equal-weight mean utility **`1.153615276991245`**.
  - **Eval-Mode Bit-Exactness (`PASS`)**: `eval_forward_bit_exact = true` (`torch.equal` between native `predict_costs` and cached-anchor `adapter_forward` with `InputFeatureNoise.eval()`).
- **`07:26Z` Addendum `None`-Gradient Guard Verification (`PASS`)**:
  - Across all `2,400` optimizer updates, `41/47` trainable adapter parameter tensors received active gradients and **`6/47` trainable adapter tensors had `p.grad is None`** on every single update (`active_grad_tensors = 41`, `none_grad_tensors = 6`). The `grad_norms()` guard skipped `None` gradients cleanly and recorded exact active/None tensor counts at every step.
- **Fit-Fold Feature Standard Deviations $s_j$ & Perturbation Telemetry**:
  - Across all `5` outer folds (`fit` context counts `775, 743, 690, 684, 765`), **`224/320` features** (`7` present modality slots $\times$ `32` dims) had strictly positive $s_j \in [0.078349, 1.749796]$ (median active $s_j \in [0.780157, 0.814896]$), while **`96/320` features** (`3` absent modality slots `3, 6, 8`) had exact $s_j = 0.0$.
  - On every noisy microbatch (`8` contexts $\times$ `224` active features = `1,792` perturbed activations per microbatch, `0` perturbations on the `96` absent features):
    - `CTRL`: `mean_noise_l2 = 0.0000` (expected `0.0000`), `perturbed_elem_count = 0`, `step40` median adapter L2 drift = **`0.374725`**.
    - `IN05`: `mean_noise_l2 = 1.7765` (expected `1.7765`), `perturbed_elem_count = 1792`, `step40` median adapter L2 drift = **`0.374743`**.
    - `IN10`: `mean_noise_l2 = 3.5410` (expected `3.5531`), `perturbed_elem_count = 1792`, `step40` median adapter L2 drift = **`0.374656`**.
    - `IN10_FIXED`: `mean_noise_l2 = 3.5477` (expected `3.5531`), `perturbed_elem_count = 1792`, `step40` median adapter L2 drift = **`0.374896`**.
  - All `60/60` (`15` cells $\times$ `4` arms) `step-40` adapter checkpoints are distinct in parameter space (`distinct_from_ctrl_step40 = 15/15` for `IN05`, `IN10`, and `IN10_FIXED`; `distinct_from_in10_step40 = 15/15` for `IN10_FIXED`; L2 drift differs by `~1e-4`).
- **Explicit Red-Team Dose / Discrete-Argmin Invariance Disclosure (`t0120`)**:
  - Even though input noise was applied on every training forward pass (`noise_l2 = 1.78 / 3.54`, `1,792` changed activations per microbatch) and the `step-40` adapter weights are distinct across all `4` arms (`L2 drift` differing by `~1e-4`), **`inner_selector_summary` is IDENTICAL across `CTRL`, `IN05`, `IN10`, and `IN10_FIXED`**:
    - Identical inner-`B` unconstrained `best_step_counts`: `{"0": 7, "10": 6, "20": 2, "30": 0, "40": 0}` (`8/15` cells positive on `B`).
    - Identical inner-`B` `median_best_step_gain = +0.00046809822976063423` and `max_best_step_gain = +0.01728633219852009`.
    - Identical inner-`B` `max_best_step_lcb95 = 0.0` (`strict_selected_step_counts = {"0": 15, "10": 0, "20": 0, "30": 0, "40": 0}`).
  - **Plain-Language Conclusion**: At doses $\tau \le 0.10 s_j$, the continuous weight perturbations (`~1e-4` L2 drift difference) are too small to flip the discrete argmin candidate choices on the inner selector panel (and flip only a single outer-assessment context on seed `167102` for `IN10_FIXED`, $\Delta = -0.000013$). **This wave is therefore uninformative-equal-to-`CTRL` at $\tau \le 0.10 s_j$, NOT evidence that input noise hurts.**

---

## 3. Primary & Secondary Readouts (Strict Selector — `RULE_NEUTRAL` / `NO_ADVANCE`)

Under the preregistered strict selector (`gain > +0.001 + 1e-12` and `LCB95 > 0`), all `15/15` cells across all `4` arms selected **`step = 0` (`r882`)** (`median_selected_l2_drift = 0.0`).

| Panel | Arm | Selected Step Counts (`0/10/20/30/40`) | Family Mean Utility | Context Mean Utility | Median Seed $\Delta$ vs `r882` | 95% Family Bootstrap CI | Positive Seeds | Holm $p$ vs `r882` | BT MM Elo vs `r882=1500` (95% CI) | `-2` Gate Cliffs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `r882` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `CTRL` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `IN05` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `IN10` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Primary: `DL-Train` OOF (`1,219` ctx / `205` fam)** | `IN10_FIXED` | `15/0/0/0/0` | `1.385355` | `1.423073` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `17.0` |
| **Secondary: `DL-Val` (`91` ctx / `19` fam)** | All `4` arms | `15/0/0/0/0` | `1.153615` | `1.137794` | `0.000000` | `[0.000000, 0.000000]` | `0/3` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `0.0` |

---

## 4. Descriptive Unconstrained Inner-`B` Best-Step Readout & Mechanism Contrast

Under the unconstrained inner-`B` argmax selector (`best_step` $\in \{0, 10, 20, 30, 40\}$ maximizing equal-family utility on `B`: `step=0` in `7` cells, `step=10` in `6` cells, `step=20` in `2` cells; median unconstrained L2 drift `0.09764` `CTRL`, `0.09769` `IN05`, `0.09762` `IN10`, `0.09772` `IN10_FIXED`):

| Panel (Unconstrained Inner-`B` `best_step`) | Contrast / Arm | Per-Seed Equal-Family $\Delta$ (`167101, 167102, 167103`) | Median Seed $\Delta$ | 95% Family Bootstrap CI | Positive Seeds | Holm $p$ | BT MM Elo vs `r882=1500` (95% CI) | Mean Harm Families |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `CTRL - r882` | `[-0.002092, -0.003020, -0.006478]` | `-0.003020` | `[-0.011818, +0.001220]` | `0/3` | `0.8982` (raw) | `1498.57 [1494.44, 1502.27]` | `4.00 / 205` (`1.95%`) |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `IN05 - r882` | `[-0.002092, -0.003020, -0.006478]` | `-0.003020` | `[-0.011818, +0.001220]` | `0/3` | `1.0000` | `1498.57 [1494.44, 1502.27]` | `4.00 / 205` (`1.95%`) |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `IN10 - r882` | `[-0.002092, -0.003020, -0.006478]` | `-0.003020` | `[-0.011818, +0.001220]` | `0/3` | `1.0000` | `1498.57 [1494.44, 1502.27]` | `4.00 / 205` (`1.95%`) |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `IN10_FIXED - r882` | `[-0.002092, -0.003032, -0.006478]` | `-0.003032` | `[-0.011818, +0.001218]` | `0/3` | `1.0000` | `1498.57 [1494.41, 1502.16]` | `4.00 / 205` (`1.95%`) |
| **`DL-Train` OOF (`1,219` ctx / `205` fam)** | `IN10 - IN10_FIXED` (mechanism) | `[0.000000, +0.000013, 0.000000]` | `0.000000` | `[0.000000, 0.000000]` | `1/3` | `0.5003` (raw) | — | `0.00 / 205` (`0.00%`) |
| **`DL-Val` (`91` ctx / `19` fam)** | `CTRL / IN05 / IN10 / IN10_FIXED - r882` | `[-0.004610, -0.001398, -0.002181]` | `-0.002181` | `[-0.005960, +0.000764]` | `0/3` | `0.9036` (raw) | `1489.04 [1463.33, 1518.28]` | `1.00 / 19` (`5.26%`) |

### Verdict & Recommended Next Step
- **Verdict**: **`COMPLETE - NO_ADVANCE`** (`0` serving promotions).
- **Next Step**: **Do NOT run any further input-noise doses at $\tau \le 0.10 s_j$** (which do not move discrete argmin choices relative to `CTRL`). **If input-noise regularization is revisited at all, run a `0.5 s_j` ($\tau = 0.50$) dose screen only** to verify that the perturbation alters discrete candidate argmins before committing to a cross-fit continuation wave.

---

## 5. Host Telemetry, Independent Recount, & Versioned S3 Custody

- **Runtime & Host Guards (`PASS`)**:
  - `elapsed_seconds = 2018.02` (`monitor.py` `2021.86` s), `peak_rss = 3,412,242,432` B (`3.41 GB < 6.0 GB`), `max_threads = 3` (`<= 4`), `max_load = 45.57` (`<= 85.0`), `min_avail_bytes = 71,872,155,648` (`66.94 GiB >= 16 GiB`), `min_free_bytes = 633,605,595,136` (`633.61 GB >= 40 GB`), `max_disk_used_frac = 0.9193` (`< 0.95`), `protected_reads = 0`, `refused = 0`, `cuda_initialized = false`.
- **Independent Recount (`PASS`)**:
  - Independent recount (`recount.json`) recomputed all `30,599` context-level choice/utility evaluations across `train_1219` and `val_91` plus all `4` statistical blocks (`primary`, `secondary`, `descriptive_unconstrained`, `descriptive_unconstrained_val`) with **`0` mismatches** (`max_abs_diff = 0.0`).
- **Versioned S3 Custody ([`custody.json`](custody.json) — `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H167/gpu_fused_agy-t0125/`)**:

| Object | S3 `VersionId` | Size (Bytes) | SHA-256 (`byte_readback_verified = true`) |
| :--- | :--- | :---: | :--- |
| `h167-evidence-bundle.tar.gz` (`293` members incl. all `240` `.npz` snapshots) | `efAkgybagVLq0brb9ZRZpw1uIjfBFDIw` | `33,531,767` | `fdcc2719bf0687ecb2d928a34e2f5f0443b54c9fd2fa508971767ce9993d8aee` |
| `metrics.json` (raw full `1,331`-open audit) | `v2uSYxLfzhcJdFgD3aCtJzq4C6nfGoWV` | `200,930` | `ec86a5f8249482098ab3cf5f40f35550a1f01c8a0324b0878491d2ad43a6c2b4` |
| `panels.npz` | `9E1KyQVGqGhS6QM521OmMUCmUD0.MoQh` | `65,043` | `bb29c737f0a80708772c992884e4236e102aea5b9a6e91a4c14a770a61f2b377` |
| `audit.json` | `EsOYYkHcFY6u.VbPgF.VD_9mnfplVYsw` | `152,636` | `40e68b57ddd1e14267716846f6dd5e06d071ecd7a9b423ee5234e902868e82af` |
| `recount.json` | `fjpy9PT4RxZIm6GLpvKh5gwr2Pd_7v.s` | `2,594` | `69538fa64be27f2508447273c19f03fbbf5dd406cdf090f5b13daa9067d9784b` |
| `launch.json` | `0Y5vPwBiWa9k4Q8kxEbjdyzMIogibBsE` | `336` | `bdeff345aadbbc571daf3e4cfa3aa27b0b841b3feae70b757047a660d30eeeea` |
| `custody.json` (full archive member inventory) | `cvi.I3sfyJgMXtf3pPUde8XMs4dZgGwH` | `42,005` | `23f567b7d77929377d935e2681b51dd789250db2d46df2e22f3dca5e3bb5aa31` |
