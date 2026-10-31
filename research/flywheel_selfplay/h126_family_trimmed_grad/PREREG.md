# H126 — Robust Gradient Aggregation across Training Families (Coordinate-Wise Trimmed-Mean + Frozen Calibration Norm Clip)

Preregistered before calibration, optimizer updates, or H126 outcome reads under `/home/ubuntu/nudges/rd_argon_t1025.md` (`rd_argon` / `[ARGON]`). CPU-only offline candidate study; no serving promotion or GPU authorization.

## 1. Falsifiable Hypothesis & Distinctness from Prior Waves

Prior waves tested loss-element or per-context objective clipping (`H59`, `H125`), family-balanced context sampling (`H73`, `H109`), trajectory averaging (`H46`, `H48`, `H55`, `H97`), sharpness-aware perturbation (`H76`, `H82`, `H100`), and lookahead outer interpolation (`H93`, `H122`). None modified **how per-family gradients are aggregated within an update**: under the standard `r882` / `H97` / `H105` recipe, a single heavy-tailed or outlier training family in a 16-context minibatch can dominate individual coordinates of the `29,211`-parameter adapter gradient even after global L2 clipping at `1.0`.

**Hypothesis (`H126`)**: Computing one `29,211`-parameter adapter gradient vector per distinct training family represented in each minibatch and aggregating across families via a **coordinate-wise `20%`-each-tail trimmed mean** (`TRIM`), or `TRIM` combined with a **global gradient-norm clip at the median per-update norm measured on frozen `CONTROL` calibration** (`TRIM_CLIP`), improves held-out `DL-Train` family utility (`fold 0`, `41` families) over the exact `r882` plain-mean `CONTROL` recipe across paired seeds `70, 71, 72`.

---

## 2. Frozen Design, Data Closure & Fresh Family Split (`fold_seed = 126070`)

1. **Data Closure & Provenance**:
   - Use only the pinned real-S3 `H50` **`DL-Train`** packet (`/home/ubuntu/jope-data/h50/input/selected.json`, SHA-256 `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`; `source_manifest.json` SHA-256 `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`, manifest ID `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`, `1,219` regression contexts across `205` families) per `/home/ubuntu/jope_plan/master/source/S3_DATASET_CATALOG.md`.
   - Zero `DL-Val` reads (`0`), zero sealed meta-test reads (`0`), zero holdout/outer-test reads (`0`), zero live-loop (`xbabe1:~/jope-data/flywheel-full-20261030/`) reads (`0`).
2. **Seeds (`70, 71, 72`)**:
   - Verified unused across `/home/ubuntu/jope_plan/ns/receipts/`, `overnight-hwave-index.md`, `docs/research/waves/INDEX.jsonl`, and active branches/PRs.
   - `9` paired cells total: `3` arms (`CONTROL`, `TRIM`, `TRIM_CLIP`) $\times$ `3` seeds (`70, 71, 72`).
3. **Fresh Family-Grouped `DL-Train` Assessment Split (`fold_seed = 126070`)**:
   - Sort the `205` `DL-Train` family IDs lexicographically, permute with `np.random.default_rng(126070).permutation(205)`, and split via `np.array_split(order, 5)` into `5` disjoint `41`-family folds (`fold_sha256 = ab8abd3319c42b5a462653c2ca3cd226b99faeb96658aaae0e58294ea3d28117`).
   - **Assessment Fold (`fold 0`)**: `41` families, `211` contexts. Never touched during calibration or training; used only for the pre-training unchanged-`r882` regression check and terminal post-40-step evaluation.
   - **Fit Folds (`folds 1..4`)**: `164` families, `1,008` contexts (`fold 1`: `245`, `fold 2`: `262`, `fold 3`: `278`, `fold 4`: `223`). Used exclusively for `CONTROL` calibration and the `40` paired optimizer updates per cell.
4. **Model & Checkpoint Invariants**:
   - Every cell starts from its own fresh SHA-verified copy of `/home/ubuntu/jope-data/h51/ckpt/round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) staged in `/home/ubuntu/jope-rescue-20260921/h126-family-trimmed-grad/scratch/`.
   - Only the `29,211` non-anchor `adapter_10tok` parameters have `requires_grad = True`.
   - Frozen `DOPE` weights and buffers are locked in `.eval()` mode and verified (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) before and after every cell.

---

## 3. Per-Update Objective & Family Gradient Aggregation (`CONTROL`, `TRIM`, `TRIM_CLIP`)

All arms share the exact `H76` / `H97` / `H105` / `H125` batch plan and optimizer hyperparameters:
- `20` rounds $\times$ `2` AdamW steps/round = `40` updates per cell (`batch_size = 16` contexts per step, `slate_size = 192` candidates per context, `AdamW(lr=1e-4, weight_decay=1e-4)`, no LR scheduler).
- For each context $c$ in minibatch $B_t$ ($|B_t| = 16$) with sampled 192-candidate slate $S_c$, predicted costs $\hat{y}_c \in \mathbb{R}^{192}$, and targets $t_{c,i} = -\text{clip}(-f_{c,i}, -2.0, 4.0)$ on gate-passing candidates and $+2.0$ on gate failures:
  $$L_c(\theta) = \frac{1}{192}\sum_{i=1}^{192} \text{SmoothL1}(\hat{y}_{c,i}, t_{c,i}; \beta=1.0) + 0.1\,\text{KL}\!\left(\text{softmax}(-t_c / 0.2) \,\|\, \text{softmax}(-\hat{y}_c / 0.2)\right).$$
- Let $\mathcal{F}(B_t) = \{f_1, \dots, f_{M_t}\}$ be the lexicographically sorted distinct `family_id`s present in minibatch $B_t$ ($1 \le M_t \le 16$), and let $C_f(B_t) = \{c \in B_t : \text{fam}(c) = f\}$ ($n_f = |C_f(B_t)|$, $\sum_{f} n_f = 16$).
- For each family $f \in \mathcal{F}(B_t)$, define the mean family loss on the minibatch and its `29,211`-dimensional adapter gradient:
  $$\bar{L}_f(\theta) = \frac{1}{n_f}\sum_{c \in C_f(B_t)} L_c(\theta), \qquad g_f = \nabla_\theta \bar{L}_f(\theta) \in \mathbb{R}^{29,211}.$$

### Arm 1: `CONTROL` (Plain Mean — Exact `r882` Recipe)
- Aggregate gradient is the exact plain context-mean gradient over the 16 contexts:
  $$g_{\text{CONTROL}} = \nabla_\theta \!\left(\frac{1}{16}\sum_{c \in B_t} L_c(\theta)\right) = \sum_{f \in \mathcal{F}(B_t)} \frac{n_f}{16}\, g_f,$$
  followed by `torch.nn.utils.clip_grad_norm_(params, 1.0)` and `optimizer.step()`.

### Arm 2: `TRIM` (Coordinate-Wise `20%`-Each-Tail Trimmed Mean Across Families)
- Stack the $M_t$ family gradient vectors into $G_t = [g_{f_1}, \dots, g_{f_{M_t}}]^\top \in \mathbb{R}^{M_t \times 29,211}$.
- Set the per-tail trim count $k_t = \lfloor 0.20 \cdot M_t \rfloor$ (so $2 k_t < M_t$ for all $M_t \ge 1$; e.g. for $M_t \in [10, 14]$, $k_t = 2$ tails trimmed on each side; for $M_t \in [15, 16]$, $k_t = 3$ tails trimmed on each side).
- Sort each coordinate $j \in \{1, \dots, 29,211\}$ across the $M_t$ families, $g_{(1), j} \le g_{(2), j} \le \dots \le g_{(M_t), j}$, and compute the coordinate-wise trimmed mean:
  $$g_{\text{TRIM}, j} = \frac{1}{M_t - 2 k_t} \sum_{r = k_t + 1}^{M_t - k_t} g_{(r), j}.$$
- Assign $g_{\text{TRIM}}$ to `params`, apply the standard safety ceiling `torch.nn.utils.clip_grad_norm_(params, 1.0)`, and call `optimizer.step()`.

### Arm 3: `TRIM_CLIP` (`TRIM` + Frozen `CONTROL` Calibration Median Global Norm Clip)
- **Frozen Pre-Update Calibration**:
  - Before any optimizer step in any cell, evaluate the `40` paired minibatch steps of seed `70` on an unchanged `round-882.pt` copy (`optimizer.step()` **never** called; adapter SHA-256 verified identical before and after) using the `CONTROL` plain-mean gradient on fit families (`folds 1..4`).
  - Record all `40` pre-clip global L2 gradient norms $\nu_u = \|g_{\text{CONTROL}}^{(u)}\|_2$ ($u = 1, \dots, 40$) and freeze their median:
    $$C_{\text{med}} = \text{median}_{1 \le u \le 40}(\nu_u)$$
    in `/home/ubuntu/jope-rescue-20260921/h126-family-trimmed-grad/calibration_manifest.json` and create-only S3 (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H126/<run_id>/calibration_manifest.json`) **before** any optimizer update is executed.
- **Update Rule**:
  - Compute $g_{\text{TRIM}}$ exactly as in `TRIM`, then clip its global L2 norm at $\min(C_{\text{med}}, 1.0)$ via `torch.nn.utils.clip_grad_norm_(params, min(C_med, 1.0))` and call `optimizer.step()`.

---

## 4. Regression-First Gate (`INVALID` If Violated)

Before any calibration or optimizer step:
1. Two independently staged unchanged `round-882.pt` copies must produce identical cost predictions (`atol = 1e-6, rtol = 0`) and identical equal-family utility (`atol = 1e-12`) across all `211` assessment contexts (`41` families in `fold 0`).
2. Unchanged `round-882.pt` must reproduce the pinned `1,219`-context `DL-Train` harness utility (`1.1536152769912449`, within `atol = 1e-6`) and match the recorded `r882` validation metadata (`round == 882`, `best_dev_round == 882`, `dope_sha256 == c51a56d0...`, historical `H114` validation score `1.1377941898164525` within `1e-5` of `1.13779`, without reading `DL-Val` payloads).
3. On an identical minibatch, the `CONTROL` family-weighted gradient sum $\sum_{f} \frac{n_f}{16} \nabla_\theta \bar{L}_f$ must match the direct batch-mean gradient $\nabla_\theta (\frac{1}{16}\sum_{c} L_c)$ within `atol = 1e-6, rtol = 1e-6`.
4. Any failure marks the wave **`INVALID`** and halts before optimization.

---

## 5. Primary Endpoint, Statistical Gate & Resource Guardrails

- **Evaluation Metric**: Unchanged `H61`/`H97`/`H105`/`H125` terminal unrevealed-candidate selection utility (`eval_mode="unrev"`, eligible candidate indices `4..P-1`, passing utility $= \text{clip}(-\text{fitness}, -2.0, 4.0)$, gate failure $= -2.0$) on the `41` held-out assessment families (`fold 0`, `211` contexts).
- **Primary Comparison (Per Directive `/home/ubuntu/nudges/rd_argon_t1025.md`)**:
  - Paired per-family utility delta on `fold 0` (`41` families) versus `CONTROL` ($\Delta_{f,s}(A) = U_{f,s}(A) - U_{f,s}(\text{CONTROL})$ for $A \in \{\text{TRIM}, \text{TRIM_CLIP}\}$ and $s \in \{70, 71, 72\}$). Co-readout also reports paired deltas versus unchanged `r882`.
  - **Strict Admission Rule (All 3 Required; Ties = Reject; No Post-Hoc Arm Selection)**:
    1. **Positive Seed Median**: $\text{median}_{s \in \{70,71,72\}}\!\left(\frac{1}{41}\sum_{f \in F_0} \Delta_{f,s}(A)\right) > 0$.
    2. **Family-Bootstrap Lower Bound**: One-sided `10,000`-draw paired family-cluster bootstrap (`bootstrap_seed = 126070`, resampling `41` families with replacement paired across seeds and taking the seed-median gain per draw) satisfies:
       $$\text{LCB}_{95}(A \text{ vs CONTROL}) > +0.001 \quad \text{(strict; ties fail)},$$
       reported alongside two-sided `95%` percentile CIs and 2-arm Holm step-down across `TRIM` and `TRIM_CLIP`.
    3. **Per-Seed Harm Bound**: In **every** seed $s \in \{70, 71, 72\}$, at most `10%` of assessment families (`<= 4` of `41` families) have $\Delta_{f,s}(A) < -0.01$.
  - **Elo Reporting**: Report family-grouped Bradley-Terry Elo and family-paired win/tie/loss Elo (`CONTROL = 1500.0` and `r882 = 1500.0`) with `10,000`-draw family-bootstrap `95%` CIs. Never report AUC.
- **CPU-Only Resource Guardrails (`xbabe2`)**:
  - CPU only (`torch.device("cpu")`), `nice -n 15`, `<= 6` threads (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(4)`), sequential single-cell execution.
  - Abort immediately before launch and before/after every cell if:
    - `MemAvailable < 30 GiB` (`30 * 1024**3` bytes),
    - `1-min loadavg > 86.7` (`85%` of `102` cores),
    - `statvfs("/")` disk used `> 95.0%`, or
    - process `RSS > 6.0 GB` (`6,000,000,000` bytes).
  - External custody path: `/home/ubuntu/jope-rescue-20260921/h126-family-trimmed-grad/` plus create-only S3 receipts under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H126/`.
