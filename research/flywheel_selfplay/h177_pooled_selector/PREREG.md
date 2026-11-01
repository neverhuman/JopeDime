# H177 Preregistration: Pooled Cross-Fit Confirmatory Selector (`205`-Family Pooled `B` Selection + Single Prespecified Candidate per Arm + Separate `A`-Family Confirmation) on `r882` Continuation

- **Wave**: `H177`
- **Owner**: `gpu_fused_agy` (`[FUSED-AGY]`)
- **Authority**: `/home/ubuntu/nudges/hwave_t0400_h177_fused_agy.md` and `/home/ubuntu/jope_plan/DECISIONS_20261029.md` (`2026-11-01T11:05Z` H-wave selector policy)
- **Seeds & Partition Pins**: `SEEDS = (177101, 177102, 177103)`, `FOLD_SEED = 177140`, `BOOT_SEED = 177141`
- **Launch Gate (`PREREG + HARNESS ONLY` until `H175` `#1490` Verdict)**:
  - **No science run is permitted in this preregistration step.**
  - Execution is gated on the merged closeout verdict of `H175` (`#1490`, `research/flywheel_selfplay/h175_selector_power`):
    - If `H175` closes with **`SELECTOR_UNDERPOWERED`** or **`MIXED`**, update `h175_power_anchor.json` and Section 4 below with `H175`'s measured `MDE80` and closeout commit SHA, then launch the single guarded CPU run on `xbabe2`.
    - If `H175` closes with **`CEILING_LIKELY`**, `H177` closes immediately as **`NOT_RUN`** with reason `H175_CEILING_LIKELY` (`0` optimizer updates).

---

## 1. Motivation & Hypothesis

In waves `H156`–`H174`, per-cell strict inner selection decided on a single `41`-family fold $B_k$ and searched across $4$ intermediate trajectory snapshots (`steps = {10, 20, 30, 40}`). Deciding on only $N_B = 41$ families inflates the family-bootstrap standard error by $\sqrt{205 / 41} = \sqrt{5} \approx 2.2361\times$ relative to the full `205`-family `DL-Train` cohort, while searching across $4$ snapshots introduces winner's-curse selection bias. Per `DECISIONS_20261029.md` (`2026-11-01T11:05Z`), we do **not** loosen the `+0.001` margin or `LCB95 > 0` gate into noise acceptance; instead, `H177` increases selector statistical power at the **exact same `+0.001` margin and `LCB95 > 0` gate** by:
1. **Pooled cross-fit `B` selection across all `205` `DL-Train` families (`1,219` contexts)**: Every `DL-Train` family serves as an out-of-fit selection (`B`) family **exactly once** across the `5` cross-fit folds ($5 \times 41 = 205$ families).
2. **One prespecified candidate step per arm (`CANDIDATE_STEP = 40`, the final step)**: No best-of-$k$ intermediate snapshot search (`steps 10/20/30` are never scored on `B`, `A`, or `DL-Val`), eliminating snapshot multiplicity.
3. **Within-family seed averaging before resampling**: Averaging the `3` seeds (`177101, 177102, 177103`) within each of the `205` families before running the `10,000`-draw paired family bootstrap (`boot_seed = 177141`).

---

## 2. Registered Arms (`H171` / `H170` Recipes) & Single Prespecified Candidate

All `4` continuation arms start from an identical read-only copy of `round-882.pt` (`/home/ubuntu/jope-data/h177/inputs/round-882.pt`, SHA-256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) with frozen `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`), frozen non-adapter backbone (`8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`), and `29,211` trainable parameters in `adapter_10tok` (`47` tensors: `41` active gradients, `6` structural `None`-as-zero gradients).
- **Prespecified Candidate Step**: `PRESPECIFIED_CANDIDATE_STEP = {"CTRL": 40, "ACC4": 40, "ACC16": 40, "OC15": 40}` (`CANDIDATE_STEP = 40` chosen a priori; no intermediate step evaluation).
- **Arms (`4` continuation arms vs `r882` baseline, `60` trajectories, `2,400` total optimizer updates)**:
  1. `CTRL`: constant `AdamW(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)`, `1` family-stratified minibatch ($1B = 8$ contexts across `8` distinct fit families) per update for `40` updates (`320` fit examples/trajectory).
  2. `ACC4` (`H171` recipe): constant `AdamW(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)`, accumulating the exact mean gradient of `4` family-stratified minibatches ($4B = 32$ contexts across `32` distinct fit families) per update for `40` updates (`1,280` fit examples/trajectory).
  3. `ACC16` (`H171` recipe): constant `AdamW(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)`, accumulating the exact mean gradient of `16` family-stratified minibatches ($16B = 128$ contexts covering all `123/123` fit families per update — `118` singletons + `5` two-context families) per update for `40` updates (`5,120` fit examples/trajectory).
  4. `OC15` (`H170` recipe): one-cycle LR and inverse-momentum `AdamW(weight_decay=1e-4)` schedule (`12` linear warmup updates $t=1..12$ from `lr = 1e-5, beta1 = 0.95` to `lr = 1.5e-4, beta1 = 0.85`, followed by `28` cosine-annealing updates $t=13..40$ to `lr = 1e-6, beta1 = 0.95` at `step = 40`, $\beta_2 = 0.999$) on the exact same `1B = 8` family-stratified minibatches as `CTRL` (`320` fit examples/trajectory).

---

## 3. Two-Stage Pooled Cross-Fit Selection & Separate `A`-Family Confirmation Protocol

The `205` `DL-Train` families (`1,219` contexts) are partitioned by `fold_seed = 177140` (`fold_manifest.json`) into `5` disjoint blocks $F_0, F_1, F_2, F_3, F_4$ of `41` families each. For fold $k \in \{0, 1, 2, 3, 4\}$:
- **Confirmation block**: $A_k = F_k$ (`41` families).
- **Selection block**: $B_k = F_{(k+1) \bmod 5}$ (`41` families).
- **Training (fit) split**: $\text{FIT}_k = \mathcal{F}_{\text{TRAIN}} \setminus (A_k \cup B_k)$ (`123` families). Within every fold $k$, $\text{FIT}_k$, $B_k$, and $A_k$ are pairwise disjoint ($B_k \cap \text{FIT}_k = \emptyset$, $A_k \cap \text{FIT}_k = \emptyset$, $B_k \cap A_k = \emptyset$).

### Stage 1: Fit-Only Training, Pooled `B` Scoring (`205` Families), & `fsync` Selection Lock
1. For each fold $k \in \{0..4\}$ and seed $s \in \{177101, 177102, 177103\}$, train each arm on $\text{FIT}_k$ only for `40` updates to produce $\hat{\theta}_{s, k}^{(\text{arm})}$ (`step 40`), and score **only** $B_k = F_{(k+1)\bmod 5}$ (`41` families).
2. Pool the five disjoint selection blocks $B_0 \sqcup B_1 \sqcup B_2 \sqcup B_3 \sqcup B_4 = \mathcal{F}_{\text{TRAIN}}$ across $k = 0..4$. Every one of the `205` `DL-Train` families (`1,219` contexts) is represented **exactly once** in the pooled `B` panel under the unique fold model $k_B(f) = (m - 1) \bmod 5$ (for $f \in F_m$), for which $f \notin \text{FIT}_{k_B(f)}$.
3. **Within-Family Seed Averaging & `10,000`-Draw Paired Family Bootstrap (`boot_seed = 177141`)**:
   - For each family $f \in \{1..\text{205}\}$ and seed $s \in \{1..3\}$, compute the family mean utility $u_{s, f}^{(B)}(\text{arm}) = \frac{1}{|C_f|} \sum_{c \in C_f} u_{s, c}^{(B)}(\text{arm})$ and gain $\Delta_{s, f}^{(B)}(\text{arm}) = u_{s, f}^{(B)}(\text{arm}) - u_{f}^{(0)}$ vs `r882`.
   - Average across the `3` seeds **within each family $f$ before resampling**:
     $$\bar{\Delta}_{f}^{(B)}(\text{arm}) = \frac{1}{3} \sum_{s=1}^{3} \Delta_{s, f}^{(B)}(\text{arm}), \qquad \hat{\mu}^{(B)}(\text{arm}) = \frac{1}{205} \sum_{f=1}^{205} \bar{\Delta}_{f}^{(B)}(\text{arm}).$$
   - Resample `205` families with replacement for `10,000` draws (`boot_seed = 177141`, shared resample index matrix `[10000, 205]` across all arms) to obtain the two-sided percentile 95% CI $[\text{LCB}_{95}^{(B)}(\text{arm}), \text{UCB}_{95}^{(B)}(\text{arm})]$, raw one-sided $p$-value $p_{\text{one-sided}}^{(B)}(\text{arm}) = (1 + \#\{\Delta_d^{*(B)}(\text{arm}) \le 0\}) / 10001$, and **Holm step-down adjusted $p$-values** across all `4` arms (`holm_p`, $m = 4$; `holm_p_treatments3` over `ACC4, ACC16, OC15` also reported).
4. **Confirmatory Pooled-`B` Admission Rule**:
   - Arm $a \in \{\text{CTRL}, \text{ACC4}, \text{ACC16}, \text{OC15}\}$ is **admitted** (`admitted = True`, selecting `step = 40`; else `admitted = False`, retaining `step = 0` = `r882`) iff:
     $$\hat{\mu}^{(B)}(a) > 0.001 + 10^{-12} \quad\text{AND}\quad \text{LCB}_{95}^{(B)}(a) > 0.0 \quad\text{AND}\quad \text{Holm-}p^{(B)}(a) < 0.05.$$
   - Exact or numerical ties ($\hat{\mu}^{(B)}(a) \le 0.001 + 10^{-12}$ or $\text{LCB}_{95}^{(B)}(a) \le 0.0$) strictly reject (`selected_step = 0`).
5. **Hard Stage-Order File-System Barrier (`pooled_b_selection_lock.json`)**:
   - Before any `A_k` confirmation context or `DL-Val` context is scored on any trained checkpoint, `runner.py` writes and `os.fsync`s `pooled_b_selection_lock.json` containing the full `205`-family pooled `B` gains, CIs, Holm $p$-values, and admission decisions. `StageOrderGuard.require_b_locked()` verifies both in-memory lock state and on-disk `pooled_b_selection_lock.json` integrity before Stage 2 can execute.

### Stage 2: Separate Confirmation Readout on `A` Families & Descriptive `DL-Val`
- For each fold $k \in \{0..4\}$ and seed $s \in \{1..3\}$, model $\hat{\theta}_{s, k}^{(\text{arm})}$ scores its held-out confirmation block $A_k = F_k$ (`41` families, never used in $\text{FIT}_k$ or $B_k$ for fold $k$) and the `91`-context / `19`-family `DL-Val` panel.
- We report both:
  1. **Admitted/Selected Policy Confirmation on Pooled `A` (`205` families, `1,219` contexts)**: evaluates $\hat{\theta}_{s, k}^{(\text{arm})}$ (`step 40`) if arm $a$ was admitted in `pooled_b_selection_lock.json`, else `r882` (`step 0`).
  2. **Unconditional Prespecified Candidate (`step = 40`) Confirmation on Pooled `A` (`205` families) & Per-Fold Disjoint Confirmation ($B_k \to A_k$)**: always evaluates the `step 40` candidate $\hat{\theta}_{s, k}^{(\text{arm})}$ on $A_k$ so confirmation behavior is visible even when pooled `B` rejects (`admitted = False`).
  3. **Descriptive `91`-Context / `19`-Family `DL-Val` Readout** (both selected and unconditional `step 40`).
- **Elo Reporting**: Canonical `H148` 400-iteration Bradley–Terry MM ratings anchored at `r882 = 1500.0` (`tie_tol = 1e-4`, `prior = 0.01`, `1,000` family-bootstrap draws with `seed = BOOT_SEED + 1 = 177142`). No AUC or speed metrics are computed or reported.

### Scope & Cross-Fold Dependency Disclosures
1. **Conditional TRAIN-Exposed Scope**: Because `round-882.pt` was produced on the `1,219`-context / `205`-family `DL-Train` pool during the historical flywheel run, all `DL-Train` readouts represent conditional continuation evidence rather than unexposed holdout generalization.
2. **Cross-Fold Relationship Between Pooled `B` and Pooled `A`**: Within each fold $k$, $B_k = F_{(k+1)\bmod 5}$ and $A_k = F_k$ are 100% disjoint (`41` vs `41` families) and neither overlaps $\text{FIT}_k$ (`123` families). Across the 5-fold rotation, each family $f \in F_m$ is evaluated in pooled `B` under fold model $k_B(f) = (m-1)\bmod 5$ and in pooled `A` under a **distinct fold model** $k_A(f) = m$ (whose training sets share `82/123` fit families). Because the underlying ground-truth context outcomes of $F_m$ appear in $B_{(m-1)\bmod 5}$ and $A_m$, pooled `A` is a cross-fold out-of-selection confirmation under distinct fold models rather than a disjoint new-family cohort; `DL-Val` (`19` disjoint families) and per-fold $B_k \to A_k$ (`41` vs `41` disjoint families) provide family-disjoint descriptive companions.

---

## 4. Expected Statistical Power & `H175` `MDE80` Anchor (`h175_power_anchor.json`)

- **Variance & MDE Scaling From `41` Single-Fold `B` Families to `205` Pooled `B` Families**:
  - Pooling `5` disjoint `41`-family selection blocks into `N_fam = 205` families increases family sample size by $5\times$, reducing the between-family bootstrap standard error $\text{SE}_{\text{fam}} = \sigma_{\text{fam}} / \sqrt{N_{\text{fam}}}$ by $\sqrt{5} \approx 2.236068$ ($\text{SE}_{205} / \text{SE}_{41} = 1/\sqrt{5} \approx 0.447214$, a **$55.28\%$ reduction** in standard error and 95% CI half-width).
  - Averaging `3` seeds within each family before resampling further reduces the seed-stochastic within-family variance component by $3\times$, and testing a single prespecified candidate (`step = 40`) eliminates 4-snapshot winner's-curse inflation.
  - Given `H175`'s measured single-fold (`41`-family) strict-selector minimum detectable effect $\text{MDE}_{80}^{(41)}$, the expected `205`-family pooled-`B` $\text{MDE}_{80}^{(205)}$ under the $\sqrt{5}$ standard-error scaling law is:
    $$\text{MDE}_{80}^{(205, \text{expected})} = \max\!\left(0.001 + \frac{\text{MDE}_{80}^{(41)} - 0.001}{\sqrt{5}},\; \frac{\text{MDE}_{80}^{(41)}}{\sqrt{5}}\right).$$
- **Pinned `H175` Placeholder (`h175_power_anchor.json`)**:
  - At the time of `H177` preregistration, `H175` (PR `#1490`, reviewed head `1cb289d545912561f09ac332a0e8344948ac3e67`, `@gpu_fused_redteam_codex` PASS review `5727688555`) has not yet merged its closeout metrics.
  - `h175_power_anchor.json` pins `status = "PENDING_H175_CLOSEOUT"`, `h175_prereg_pr = 1490`, `h175_prereg_reviewed_head = "1cb289d545912561f09ac332a0e8344948ac3e67"`, `h175_closeout_commit = "PINNED_PLACEHOLDER_UPDATE_BEFORE_LAUNCH"`, `h175_verdict = "PINNED_PLACEHOLDER_UPDATE_BEFORE_LAUNCH"`, and `h175_mde80_41fam = null`.
  - **Programmatic Pre-Launch Gate**: `treatment.validate_launch()` allows `--smoke` preflight while `h175_power_anchor.json` is in `PENDING_H175_CLOSEOUT` state, but **hard-refuses any non-smoke science run** until `h175_power_anchor.json` is updated with `status = "READY_TO_LAUNCH"`, a valid 40-hex `h175_closeout_commit`, `h175_verdict in ("SELECTOR_UNDERPOWERED", "MIXED")`, and a finite positive `h175_mde80_41fam`.

---

## 5. Pre-Training Gates, Data-Free Unit Controls, & Resource Guards

1. **First-Scored Native `91/91` `DL-Val` Baseline & Static-Forward Parity Gates**:
   - Before any `DL-Train` pool is decoded or any optimizer step runs, `runner.py` verifies exact `91/91` `DL-Val` utility parity (`1.1377941898164525`, `max_abs_error = 0.0`), `eval_forward_bit_exact = True`, and initial + perturbed `47`-tensor (`41` active + `6` `None`-as-zero) `prove_static_gradient_parity`.
2. **Data-Free Unit Test Suite (`test_h177.py`)**:
   - `pooled B` never overlaps that fold's `FIT` (`B_k \cap \text{FIT}_k = \emptyset$, $B_k \cap A_k = \emptyset$) and $\bigsqcup_{k=0}^4 B_k$ covers all `205` families (`1,219` contexts) **exactly once**;
   - Prespecified candidate step (`CANDIDATE_STEP = 40`) is fixed before `B` is read, multi-step arrays are rejected, and `StageOrderGuard` blocks `A` scoring until `pooled_b_selection_lock.json` is `fsync`-ed;
   - Exact-null candidate (`candidate == r882`) admits **`0` times** across trials and boundary ties (`+0.001`) strictly reject;
   - Synthetic **`+0.01` shift** admits (`admitted = True`, `selected_step = 40`) with exact within-family 3-seed averaging prior to `10,000`-draw family resampling;
   - `H175` launch gate (`PENDING_H175_CLOSEOUT` and `CEILING_LIKELY` hard-refused for non-smoke runs) and all `xbabe2` CPU resource guards (`nice -n 19`, `<= 4` threads, `RSS < 6 GB`, `CPU <= 85%`, `MemAvailable >= 16 GiB`, `free_disk >= 40 GB`).
