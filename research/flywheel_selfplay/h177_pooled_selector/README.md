# H177: Pooled Cross-Fit Confirmatory Selector for `r882` Continuation

- **Wave**: `H177`
- **Owner**: `gpu_fused_agy` (`[FUSED-AGY]`)
- **Authority**: `/home/ubuntu/nudges/hwave_t0400_h177_fused_agy.md` & `/home/ubuntu/nudges/hwave_t0440_h177_anchor_fusedagy.md` (`DECISIONS_20261029.md` `2026-11-01T11:28Z`)
- **Preregistration & Pre-Run Amendment**:
  - Initial preregistration: [PREREG.md](PREREG.md) (PR `#1492`, squash-merged into `main` at `725fed25dc2ec2dfcc16c50e39c2351913ac7c48`)
  - Pre-run `H175` power-anchor & `H178` calibration-gate amendment: [h175_power_anchor.json](h175_power_anchor.json), [PREREG.md](PREREG.md), [analysis.py](analysis.py), [runner.py](runner.py) (PR `#1500`, `@gpu_fused_redteam_codex` `PASS` review `5727935400`, squash-merged into `main` at `70bf6698301d544fcb522998fffe0bc8dd7ec278`)
- **Execution Host**: `xbabe2` (`CPU-only`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `<= 4` threads; finished `2026-11-01T12:26Z`)
- **Status / Verdict**: **`COMPLETE — NO_ADVANCE`** (`candidate_admitted = false`, `lift_qualified = false`, `h178_calibration_status = "PENDING_H178"`; `0/4` arms admitted on `205`-family pooled `B` under `m = 4` Holm multiplicity; selected system falls back to `step = 0` (`r882`) on all `4` arms, preserving exact `r882` utility (`0.000000` delta) and `1500.00` Elo on both `A` (`205` families) and `DL-Val` (`19` families))

---

## 1. Hypothesis, `H175` Power Anchor, & `H178` Calibration Gate

Prior `r882` continuation waves (`H170` `OneCycleLR` and `H171` family-stratified gradient accumulation) evaluated step selection independently inside each outer fold's `41`-family `B_k` selection split and fell back to `step 0` (`r882`). `H175` (`MIXED`, closeout commit `b0261812e125c8743370b7801b75d27e7f27977f`) quantified why single-fold `41`-family `B_k` selection is severely underpowered for family-clustered continuation gains:
- **`H175` Single-Fold (`41`-Family) Empirical `MDE80` ([`h175_power_anchor.json`](h175_power_anchor.json))**:
  - **Family-clustered `MDE80`**: **`+0.0640`** (`+0.063957` on `24`-slate) and **`+0.0947`** (`+0.094678` on `full`-slate).
  - **Scattered-context `MDE80`** (never cited without the family-clustered anchor): **`+0.0158`** (`+0.015834` on `24`-slate) and **`+0.0228`** (`+0.022802` on `full`-slate).
- **`H177` Pooled `205`-Family Planning Heuristic ($\div\sqrt{5}$)**:
  - Pooling out-of-fit `B_k` evaluations across all `5` cyclic `3/1/1` folds (`5 × 41 = 205` families, `1,219` contexts) before applying the selector decision rule yields an expected `205`-family planning heuristic of **`~+0.029`** (`+0.028602` / `+0.029155` on `24`-slate) and **`~+0.042`** (`+0.042341` / `+0.042894` on `full`-slate) for family-clustered `MDE80` (`~+0.0071 / +0.0102` scattered-context).

### Fixed Two-Stage Protocol & Reusable Decision Rule (`H177_DECISION_RULE`)
1. **Stage 1 (`B`-Only Training & Pooled `205`-Family Selection)**:
   - Across `5` outer folds ($k \in \{0,1,2,3,4\}$, `FOLD_SEED = 177140`) and `3` seeds (`177101, 177102, 177103`), each arm (`CTRL`, `ACC4`, `ACC16`, `OC15`) trains from `round-882.pt` (`29,211` trainable adapter parameters; `41` active + `6` structural `None`-grad tensors; frozen DOPE trunk `c51a56d0...`) for `40` optimizer updates on the `123`-family `FIT_k` split only (`15` cells $\times$ `4` arms $\times$ `40` updates = `2,400` total optimizer updates).
   - No intermediate checkpoints (`10/20/30`) are scored (no best-of-$k$ step shopping). Only the single prespecified candidate step (**`CANDIDATE_STEP = 40`**) is scored on the `41`-family `B_k` selection split.
   - Across $k = 0..4$, the disjoint `B_k` splits cover all **`205` families (`1,219` contexts)** exactly once per seed.
   - [`pooled_b_select`](analysis.py) (`H177_DECISION_RULE` in [analysis.py](analysis.py)) averages each family's utility gain $\Delta_{s,f}^{(a)} = U_{B,\text{step 40}}^{(a)}(s,f) - U_{B,\text{r882}}(s,f)$ across the `3` seeds first ($\bar{\Delta}_f^{(a)} = \frac{1}{3}\sum_{s=1}^3 \Delta_{s,f}^{(a)}$), runs `10,000` paired family-clustered bootstrap draws (`BOOT_SEED = 177141`) over the `205` families, and enforces `m = 4` Holm step-down multiplicity across all `4` arms (`CTRL`, `ACC4`, `ACC16`, `OC15`). An arm is admitted (`selected_step = 40`) iff:
     $$\widehat{\Delta}_{B,\text{pool}}^{(a)} > 0.001 + 10^{-12}, \qquad \text{LCB}_{95\%}\!\left(\widehat{\Delta}_{B,\text{pool}}^{(a)}\right) > 0, \qquad p_{\text{Holm},4}^{(a)} < 0.05.$$
   - Ties and boundary cases reject (`selected_step = 0`, falling back to `r882`). Selection decisions are locked in `pooled_b_selection_lock.json` via `StageOrderGuard` before any `A_k` or `DL-Val` evaluation occurs.
2. **Stage 2 (Correlated Cross-Fold Candidate Readout on `A` & Descriptive `DL-Val`)**:
   - After locking `pooled_b_selection_lock.json`, Stage 2 scores `A_k` (`205` families, `1,219` contexts across $k = 0..4$) and `DL-Val` (`19` families, `91` contexts) for the selected system (`step 40` if admitted on pooled `B`, else `step 0` / `r882`) and records the unconditional `step 40` readout as a diagnostic.
   - Because each fold's trained checkpoint depends on `FIT_k` (`123` of `205` families), pooled `A` is explicitly reported as a **correlated cross-fold candidate readout** rather than an independent sample split.
3. **`H178` Calibration Gate ([`resolve_h177_verdict`](analysis.py))**:
   - Per `/home/ubuntu/nudges/hwave_t0440_h177_anchor_fusedagy.md`, any `H177` admission is lift (`ADVANCE`) only if `H178` (`gpu_tokbench_agy`, calibration of `H177_DECISION_RULE` on the `H175` synthetic bank) verifies null/harmful admission $\le 0.05$; otherwise an admission is `DIAGNOSTIC` / `DIAGNOSTIC_PENDING_H178`. When `candidate_admitted = false` (`0/4` arms admitted), `resolve_h177_verdict` returns **`NO_ADVANCE`** unconditionally.

---

## 2. Primary Results: Stage-1 Pooled `B` Selection (`205` Families, `1,219` Contexts)

Baseline `r882` family-mean utility on `DL-Train` (`205` families, `1,219` contexts) is **`1.385355`** (context-mean utility `1.423073`), anchored at **`1500.00` Elo**.

### Table 1A — Pooled `B` Arm Decisions & Contrasts vs `r882` (`m = 4` Holm)

| Arm | Cand. Step | Seed Gains `[177101, 177102, 177103]` | `3`-Seed Family-Mean Gain | `95%` Family Bootstrap CI | Pos. Seeds | One-Sided $p$ (`Holm` $m=4$) | Admitted (`Selected Step`) | Seed-Averaged Elo `[95% CI]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`r882` (Baseline)** | `0` | `[0.000000, 0.000000, 0.000000]` | **`0.000000`** | `[0.000000, 0.000000]` | `0/3` | `1.0000` (`—`) | — (`0`) | `1500.00 [1500.00, 1500.00]` |
| **`CTRL`** (`1B = 8`, const LR) | `40` | `[-0.025393, -0.034180, -0.030760]` | **`-0.030111`** | `[-0.057122, -0.006241]` | `0/3` | `0.9929` (`1.0000`) | **`false` (`0`)** | `1487.03 [1476.62, 1496.76]` |
| **`ACC4`** (`4B = 32` accum) | `40` | `[-0.037862, -0.033201, -0.035227]` | **`-0.035430`** | `[-0.061550, -0.010540]` | `0/3` | `0.9970` (`1.0000`) | **`false` (`0`)** | `1483.47 [1471.91, 1493.84]` |
| **`ACC16`** (`16B = 128` accum) | `40` | `[-0.031323, -0.037485, -0.034108]` | **`-0.034305`** | `[-0.062021, -0.006952]` | `0/3` | `0.9937` (`1.0000`) | **`false` (`0`)** | `1478.97 [1466.88, 1489.56]` |
| **`OC15`** (`OneCycleLR` `1.5e-4`) | `40` | `[-0.009321, -0.018773, -0.029365]` | **`-0.019153`** | `[-0.040593, +0.000757]` | `0/3` | `0.9689` (`1.0000`) | **`false` (`0`)** | `1492.43 [1484.50, 1500.36]` |

### Table 1B — Pooled `B` Active-Control Contrasts (`Step 40`)

| Contrast (`205` Families) | Seed Gains `[177101, 177102, 177103]` | `3`-Seed Family-Mean Gain | `95%` Family Bootstrap CI | Median Seed Gain `[95% CI]` | Pos. Seeds | One-Sided $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`OC15 - CTRL`** | `[+0.016072, +0.015407, +0.001395]` | **`+0.010958`** | **`[+0.001921, +0.021367]`** | `+0.015407 [+0.000962, +0.026674]` | **`3/3`** | **`0.0081`** |
| **`OC15 - ACC4`** | `[+0.028541, +0.014428, +0.005862]` | **`+0.016277`** | **`[+0.003469, +0.029640]`** | `+0.014428 [+0.000468, +0.030816]` | **`3/3`** | **`0.0065`** |
| **`ACC4 - CTRL`** | `[-0.012469, +0.000979, -0.004467]` | **`-0.005319`** | `[-0.013372, +0.003078]` | `-0.004467 [-0.014238, +0.005329]` | `1/3` | `0.8892` |
| **`ACC16 - CTRL`** | `[-0.005930, -0.003306, -0.003348]` | **`-0.004195`** | `[-0.017344, +0.009681]` | `-0.003348 [-0.017585, +0.010929]` | `0/3` | `0.7259` |
| **`ACC16 - ACC4`** | `[+0.006540, -0.004285, +0.001119]` | **`+0.001125`** | `[-0.008665, +0.011909]` | `+0.001119 [-0.010292, +0.013677]` | `2/3` | `0.4226` |

---

## 3. Stage-2 Correlated Cross-Fold Candidate Readout on `A` (`205` Families) & Descriptive `DL-Val` (`19` Families)

Because all `4` arms were rejected by `pooled_b_select` (`selected_step = 0`), the selected system (`confirmation_a_selected` and `descriptive_val_selected`) is identically `r882` across all `205` `DL-Train` families and all `19` `DL-Val` families (**zero negative transfer admitted**). Table 2 also reports the unconditional `step 40` diagnostic panels (`confirmation_a_step40` and `descriptive_val_step40`), confirming that the step-`40` continuation checkpoints are indeed inferior to `r882` on both splits.

### Table 2A — Selected System Readout (`Selected Step = 0` Across All `4` Arms)

| Panel | Arm | `3`-Seed Family Utility | Gain vs `r882` `[95% CI]` | Pos. Seeds | One-Sided $p$ (`Holm` $m=4$) | Seed-Averaged Elo `[95% CI]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pooled `A` Selected (`205` fam / `1,219` ctx)** | `CTRL / ACC4 / ACC16 / OC15` | `1.385355` | **`0.000000 [0.000000, 0.000000]`** | `0/3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` |
| **Descriptive `DL-Val` Selected (`19` fam / `91` ctx)** | `CTRL / ACC4 / ACC16 / OC15` | `1.153615` (`1.137794` ctx) | **`0.000000 [0.000000, 0.000000]`** | `0/3` | `1.0000` (`1.0000`) | `1500.00 [1500.00, 1500.00]` |

### Table 2B — Unconditional `Step 40` Diagnostic Readout on Pooled `A` (`205` Families) & `DL-Val` (`19` Families)

| Panel | Contrast | Seed Gains `[177101, 177102, 177103]` | `3`-Seed Family-Mean Gain | `95%` Family Bootstrap CI | Pos. Seeds | One-Sided $p$ (`Holm` $m=4$) | Seed-Averaged Elo `[95% CI]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled `A` `Step 40` (`205` fam)** | `CTRL - r882` | `[-0.028195, -0.022532, -0.021259]` | **`-0.023995`** | `[-0.046617, -0.002637]` | `0/3` | `0.9865` (`1.0000`) | `1488.25 [1477.61, 1498.06]` |
| **Pooled `A` `Step 40` (`205` fam)** | `ACC4 - r882` | `[-0.025222, -0.030578, -0.035794]` | **`-0.030531`** | `[-0.056406, -0.005435]` | `0/3` | `0.9918` (`1.0000`) | `1487.14 [1475.40, 1498.27]` |
| **Pooled `A` `Step 40` (`205` fam)** | `ACC16 - r882` | `[-0.045252, -0.031550, -0.047305]` | **`-0.041369`** | `[-0.071116, -0.013484]` | `0/3` | `0.9983` (`1.0000`) | `1480.97 [1468.45, 1493.49]` |
| **Pooled `A` `Step 40` (`205` fam)** | `OC15 - r882` | `[-0.022482, -0.013224, -0.017645]` | **`-0.017784`** | `[-0.036321, -0.000837]` | `0/3` | `0.9805` (`1.0000`) | `1493.29 [1485.12, 1501.19]` |
| **Pooled `A` `Step 40` (`205` fam)** | `OC15 - CTRL` | `[+0.005712, +0.009307, +0.003615]` | **`+0.006211`** | `[-0.002531, +0.015735]` | `3/3` | `0.0804` (`—`) | — |
| **Pooled `A` `Step 40` (`205` fam)** | `OC15 - ACC4` | `[+0.002739, +0.017354, +0.018149]` | **`+0.012747`** | `[-0.000603, +0.026303]` | `3/3` | `0.0307` (`—`) | — |
| **`DL-Val` `Step 40` (`19` fam)** | `CTRL - r882` | `[-0.039012, -0.031745, -0.024226]` | **`-0.031661`** | `[-0.064202, -0.005040]` | `0/3` | `0.9972` (`1.0000`) | `1500.73 [1456.80, 1557.05]` |
| **`DL-Val` `Step 40` (`19` fam)** | `ACC4 - r882` | `[-0.049729, -0.041361, -0.033475]` | **`-0.041522`** | `[-0.087197, -0.006453]` | `0/3` | `0.9987` (`1.0000`) | `1483.90 [1439.06, 1539.91]` |
| **`DL-Val` `Step 40` (`19` fam)** | `ACC16 - r882` | `[-0.048345, -0.050519, -0.044759]` | **`-0.047875`** | `[-0.099460, -0.007483]` | `0/3` | `0.9969` (`1.0000`) | `1470.47 [1420.51, 1522.72]` |
| **`DL-Val` `Step 40` (`19` fam)** | `OC15 - r882` | `[-0.017714, -0.008212, -0.010339]` | **`-0.012089`** | `[-0.026978, -0.000565]` | `0/3` | `0.9833` (`1.0000`) | `1496.10 [1475.03, 1516.15]` |
| **`DL-Val` `Step 40` (`19` fam)** | `OC15 - CTRL` | `[+0.021298, +0.023533, +0.013887]` | **`+0.019572`** | **`[+0.003724, +0.039332]`** | **`3/3`** | **`0.0015`** (`—`) | — |
| **`DL-Val` `Step 40` (`19` fam)** | `OC15 - ACC4` | `[+0.032015, +0.033149, +0.023136]` | **`+0.029433`** | **`[+0.004190, +0.063828]`** | **`3/3`** | **`0.0014`** (`—`) | — |

### Table 3 — `Step 40` Adapter L2 Parameter Drift (`||\theta_{40} - \theta_0||_2` across `15` Cells)

| Arm | Median L2 Drift | Min L2 Drift | Max L2 Drift | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **`OC15`** (`OneCycleLR` warmup + cosine cooldown to `1.5e-6`) | **`0.260265`** | `0.245026` | `0.272165` | Cosine LR decay curtails late-trajectory weight excursion by `30.4%` vs `CTRL` |
| **`CTRL`** (`1B = 8`, constant `lr = 1e-4`) | **`0.373963`** | `0.350636` | `0.385813` | Standard stochastic continuation drift |
| **`ACC4`** (`4B = 32`, constant `lr = 1e-4`) | **`0.460061`** | `0.452741` | `0.466257` | Higher inter-step gradient coherence drives larger coherent step excursion under AdamW |
| **`ACC16`** (`16B = 128`, constant `lr = 1e-4`) | **`0.531724`** | `0.527856` | `0.536557` | Near-deterministic `123`-family full-split gradient drives largest excursion away from `r882` |

---

## 4. Scientific Synthesis & Recommended Next Step

1. **Empirical Verification of `205`-Family Pooled Selector Precision**:
   - Pooling `B_k` across all `205` families shrinks the 95% family-bootstrap CI half-width on real `DL-Train` to **`±0.0207`** for `OC15` (`[-0.04059, +0.00076]`) and **`±0.0254–0.0275`** for `CTRL`/`ACC4`/`ACC16`, matching the $\div\sqrt{5}$ reduction predicted from `H175`'s `41`-family `MDE80` (`+0.0640 / +0.0947` $\to$ `~+0.029 / +0.042`).
2. **Resolution of the `H170`/`H171` Continuation Question**:
   - `H177` decisively rules out the hypothesis that `H170` (`OC15`) and `H171` (`ACC4`, `ACC16`) failed to advance merely because single-fold `41`-family `B_k` selection was underpowered.
   - At the prespecified `step 40` checkpoint, all four pairwise softplus rank-loss continuation recipes (`CTRL`, `ACC4`, `ACC16`, `OC15`) exhibit **strictly negative true family-mean utility delta vs `r882`** across all `3/3` seeds on both pooled `B` (`-0.01915` to `-0.03543`, `95% UCB < 0` for `CTRL`, `ACC4`, `ACC16` and `+0.00076` for `OC15`) and pooled `A` (`-0.01778` to `-0.04137`, `95% UCB < 0` for all `4` arms).
   - Notably, adapter L2 drift (`0.2603` for `OC15` $\to$ `0.3740` for `CTRL` $\to$ `0.4601` for `ACC4` $\to$ `0.5317` for `ACC16`) is monotonically inversely related to held-out utility (`-0.01915` on `B` / `-0.01778` on `A` for `OC15` vs `-0.03431` on `B` / `-0.04137` on `A` for `ACC16`). That is why `OC15` beats `CTRL` (`+0.01096` on pooled `B`, `+0.01957` on `DL-Val`, `3/3` seeds) and `ACC4` (`+0.01628` on pooled `B`, `+0.02943` on `DL-Val`, `3/3` seeds): cosine LR cooldown limits how far the smooth pairwise surrogate pulls the adapter away from `r882`'s discrete top-1 argmin optimum, whereas variance-reduced gradient accumulation (`ACC4`, `ACC16`) moves more coherently along the misaligned surrogate gradient.
3. **Recommended Next Step**:
   - Keep `H177_DECISION_RULE` (`pooled_b_select` over `205` families with `m = 4` Holm + `H178` calibration gate) as the standard high-power selector for future `DL-Train` cross-fit evaluations, and terminate first-order pairwise rank-loss continuation from `round-882.pt`.

---

## 5. Host Telemetry, Audit Disclosure, Independent Recount, & Versioned S3 Custody

- **Prelaunch `attempt-001` Audit Disclosure (`0` Updates)**:
  - Initial invocation `/home/ubuntu/jope-data/h177/attempt-001` exited in `2.256s` at `runner.py:495` (`KeyError: 'merged_source_verified'` in `launch.json`) prior to `runtime()`, `cohorts()`, or any model/pool open (`0` optimizer updates, `0` forward passes, `0` contexts scored; preserved in `h177-evidence-bundle.tar.gz`). After adding `"merged_source_verified": true` to `launch.json`, `/home/ubuntu/jope-data/h177/attempt-002` executed the sole scientific run.
- **Runtime & Host Guards (`attempt-002` — `PASS`)**:
  - `optimizer_updates = 2,400`, `complete_cells = 15`, `elapsed_seconds = 1,162.61` (`monitor.py` `1,165.82` s, `exit_code = 0`, `reasons = []`), `peak_rss_bytes = 4,484,132,864` B (`4.48 GB < 6.0 GB`), `threads_at_end = 1` (`<= 4`), `protected_reads = 0`, `refused = []`, `cuda_initialized = false`.
  - `91/91` `DL-Val` baseline `1.1377941898164525` (`max_abs_error = 0.0`), `eval_forward_bit_exact = true`, `static_grad_parity_bit_exact = true`, `static_grad_parity_perturbed_bit_exact = true`.
- **Independent Recount (`recount_verification.json` — `PASS`)**:
  - Reconstructed `pooled_b`, `confirm_a_selected`, `confirm_a_step40`, `val_selected`, and `val_step40` from the `15` `.b_only.json` and `15` `.a_confirm.json` cell records (`max_abs_diff = 0.0`) and verified bit-exact equality (`true`) of `pooled_b_select` decisions and all `10,000`-draw family-bootstrap / `1,000`-draw MM Elo summaries.
- **Versioned S3 Custody ([`custody.json`](custody.json) — `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H177/gpu_fused_agy-t0400/`)**:

| Object | S3 `VersionId` | Size (Bytes) | SHA-256 (`readback_verified = true`) |
| :--- | :--- | :---: | :--- |
| `h177-evidence-bundle.tar.gz` (all `15` cells, `60` `.npz` step-40 snapshots, `.b_only.json`, `.a_confirm.json`, `attempt-001` & `attempt-002` logs) | `pYs2Qq1jmHW7vRTE39Pel2t0Npah8jKJ` | `10,165,869` | `950df2212f8385f0b9e694a4d4c31e4aa0e3b38b47403e4f85d7dcf014d6c25b` |
| `metrics.json` (raw full `1,331`-open audit) | `Pps6M4kJnOmgHn8bfdNjaztViKI4j0XU` | `232,212` | `aff5e5875780c6a8e278dcc02c38889204fac69cccbc130a9246753e565d258c` |
| `pooled_b_selection_lock.json` | `8NOh_prT3qM0JoHvJ_9IteK.5KND1UJu` | `34,168` | `3acc421fb5a46bbdee2373cd5586f9dff3c63986c6b521ef2f8e7e358158123a` |
| `panels.npz` | `rnk1Ye7jt0t0peipZVJPlSs9Rub7yGsN` | `121,279` | `713f2999c549966b7fc9f6c96cda1614d3fc54844a1ce0244eafe399963e87cc` |
| `audit.json` | `sGTBbN7cDoHDtKSGC_VVgyH4HlbI1JkT` | `159,459` | `007587e19144381ce5834f699e302364d91482c036b5136266db529cb28b6f86` |
| `baseline.json` | `gwFBmjDcVf1uRx1MciJdfcNMDpsyr0.8` | `10,781` | `64fa9f657ff635a335d6e88d0cca4e29abc152b8bbeb71c28e99ae7ea06b3093` |
| `recount_verification.json` | `cztv2ZobDi7zv91kQSuPNha7SeVpWrgG` | `854` | `746ed3ca7a5bf813c6239460ec5ab39e47bc6699e2237f566c930bfba73ef0c1` |

### Reproduction & Unit Verification
```bash
PYTHONPATH=research/flywheel_selfplay/h177_pooled_selector pytest -q research/flywheel_selfplay/h177_pooled_selector/test_h177.py
```
