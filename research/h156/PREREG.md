# H156 Preregistration — Family-Grouped Cross-Fit Early Stopping + Capacity Sweep (`{0.5x, 1x, 2x}`)

## 1. Hypothesis & Motivation

Prior `r882` continuation waves (`H144`, `H146`, `H148`) observed that fixed-horizon continuation at `1x` width (`29,211` adapter parameters) degrades out-of-sample utility unless early-stopped, and may suffer from a coupling between **selection noise** (selecting on the 91-context / 19-family `DL-Val` split or running to a fixed step) and **adapter capacity mismatch** (`1x` bottleneck/bilinear/head width).

**Hypothesis (`H156`)**: The `r882` plateau is driven by selection noise and capacity mismatch. Training adapter widths $\{0.5\times, 1\times, 2\times\}$ (`W05`, `W10`, `W20`) initialized from the owned copy of `r882` (`round-882.pt`, SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) with early stopping chosen by **5-fold family-grouped cross-fit on training families** improves out-of-fold family-weighted utility and Elo over `r882`.

---

## 2. Capacity Sweep Arms (`{0.5x, 1x, 2x}`)

All three width arms retain the exact `47` trainable adapter parameter tensors (`adapter_10tok.*` excluding `adapter_10tok.anchor.*`) while freezing `DOPE-MLC-v3` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`), `FlywheelGameNet`, and all Stage 1/2 projection buffers:

1. **`W05` (`0.5x` width, `20,807` trainable parameters across `47` tensors)**:
   - Internal widths scaled by `0.5x`: `d_bot=8, d_bi=8, d_mm=12, d_traj=12, d_par=24`.
   - Initialized deterministically and data-free from `r882` by selecting the top-$k$ channels ranked by joint incoming/outgoing weight product norm.
2. **`W10` (`1.0x` width, `29,211` trainable parameters across `47` tensors)**:
   - Exact native `r882` architecture (`d_bot=16, d_bi=16, d_mm=24, d_traj=24, d_par=48`) and exact `strict=True` load of `r882`.
3. **`W20` (`2.0x` width, `46,595` trainable parameters across `47` tensors)**:
   - Internal widths scaled by `2.0x`: `d_bot=32, d_bi=32, d_mm=48, d_traj=48, d_par=96`.
   - Initialized from `r882` via function-preserving Net2WiderNet channel duplication with symmetric $\pm 10^{-3}$ outgoing split perturbations so step `0` preserves the `r882` function while breaking gradient symmetry across widened units.

---

## 3. 5-Fold Family-Grouped Cross-Fit Protocol

1. **Family Partition (`FOLD_SEED = 156140`)**:
   - The `205` `DL-Train` families (`1,219` contexts) in `/home/ubuntu/jope-data/h50/input/selected.json` are partitioned into `5` disjoint folds of `41` families (`research/h156/fold_manifest.json`, context counts `[288, 218, 257, 223, 233]`).
   - For each outer fold $A \in \{0, 1, 2, 3, 4\}$:
     - **Assessment fold**: $F_A$ (`41` families)
     - **Inner selector fold**: $F_B$ where $B = (A + 1) \bmod 5$ (`41` families)
     - **Fit folds**: the remaining 3 folds (`123` families)
   - Zero family overlap between fit, inner selector $B$, outer assessment $A$, and the `91`-context / `19`-family `DL-Val` split is asserted prior to every cell and on every minibatch.
2. **Seeds & Continuation Budget**:
   - Seeds: `SEEDS = (156101, 156102, 156103)` (`BOOT_SEED = 156141`).
   - For each `(outer_fold A, seed)` and each width arm `W05`, `W10`, `W20`: `40` AdamW updates (`lr = 1e-4, weight_decay = 1e-4, batch_size = 8`, grad clip `1.0`), snapshots at `STEPS = (0, 10, 20, 30, 40)` (`3 * 5 * 3 * 40 = 1,800` total optimizer updates across `15` cross-fit cells).
3. **Cross-Fit Early Stopping & Joint Selector Lock**:
   - Before evaluating outer assessment fold $A$, inner selector fold $B$ utilities are evaluated across `STEPS = (0, 10, 20, 30, 40)` and locked to `A{outer}-{seed}-selection.json` (`selection_lock_sha256`):
     - `W05_XFIT`, `W10_XFIT`, `W20_XFIT`: select $t^* \in \{0, 10, 20, 30, 40\}$ maximizing equal-family mean utility on $F_B$ when gain over `r882` on $F_B$ exceeds `1e-12`, else fall back to `r882` (`W10` step `0`).
     - `XFIT_JOINT`: joint cross-fit capacity + step selector choosing $(w^*, t^*) \in \{\text{W05}, \text{W10}, \text{W20}\} \times \{0, 10, 20, 30, 40\}$ maximizing equal-family mean utility on $F_B$ when gain over `r882` exceeds `1e-12`, else `("W10", 0)`.
     - Conservative LCB selections (`gain > 0.001 + 1e-12` and 95% family bootstrap LB `> 0`) are also recorded in the selection lock.

---

## 4. Dual-Cohort Evaluation & Statistical Decision Rule

We report utility lift vs `r882`, 10,000-draw paired family-cluster bootstrap 95% CIs (`BOOT_SEED = 156141`), Holm-Bonferroni adjusted p-values across the 4 candidate arms (`m = 4`: `W05_XFIT`, `W10_XFIT`, `W20_XFIT`, `XFIT_JOINT`), and 400-iteration Bradley-Terry MM **Elo vs `r882` (`1500.0`)** on **both**:

1. **The `1,219`-context / `205`-family `H145` train-only set** (out-of-fold outer-$A$ cross-fit predictions assembled across $A = 0..4$), and
2. **The `91`-context / `19`-family `DL-Val` set** (verified at step `0` to match `r882` mean utility `1.1377941898164525` with `91/91` exact context parity).

**Limitations**: `r882` was historically trained on `DL-Train` families before this continuation wave (`historical_r882_train_exposure = True`), and family bootstrap resamples the assembled cross-fit panel conditional on the locked fold policies. No sealed, holdout, or meta-test data is ever accessed.
