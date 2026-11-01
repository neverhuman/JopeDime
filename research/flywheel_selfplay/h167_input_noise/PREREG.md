# H167 — Adapter Input-Feature Gaussian Noise Injection `r882` Continuation

Authority: `~/nudges/hwave_t0125_h167_fused_agy.md`; owner `gpu_fused_agy`, `ENG:hwave-loop` reservation (`/home/ubuntu/jope_plan/ns/receipts/H167_claim.txt`). This protocol and source are registered before any `H167` score, fit-fold feature-std calculation, or optimizer update.

## Hypothesis and Non-Overlap

Across prior `r882` continuation waves (`H148`, `H156`, `H161`, `H162`), deterministic continuation from `round-882.pt` exhibits an overfitting signal: strict inner-fold selection retains step `0` (`r882`), while unconstrained continuation drifts by roughly `-0.03` utility vs `r882` on `DL-Val` and `DL-Train`. Parameter-gradient noise and activation dropout are tested in `H165`; slate-level C-Mixup is tested in `H166`.

**Hypothesis (`H167`)**: Adding zero-mean Gaussian noise in input feature space on the frozen-DOPE multimodal tokens (`mm_tokens_10x32`) that feed the `29,211`-parameter trainable adapter (`adapter_10tok`) acts as an input-space Tikhonov / Jacobian regularizer ([Bishop 1995](https://doi.org/10.1162/neco.1995.7.1.108): training with input noise $\approx$ Tikhonov regularization), curbing continuation overfitting and improving out-of-fold equal-family utility over `r882`. To separate stochastic neighborhood smoothing (data augmentation / Tikhonov regularization) from a static coordinate shift, we include a matched-variance fixed-perturbation control (`IN10_FIXED`) that reuses one fixed noise draw per fit-fold context across all `40` updates.

No speed metric, AUC, binary classification proxy, or serving change is permitted.

## Fixed Experiment

- **Checkpoint & Freeze Contract**: Read-only private copy of `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`); frozen DOPE composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and all non-adapter tensors (`frozen_sha256`) verified before and after every update and evaluation. Exactly `47` adapter tensors (`29,211` parameters) have `requires_grad=True`.
- **Optimizer & Continuation Budget**: Every trajectory initializes at `r882` with fresh `AdamW(lr=1e-4, weight_decay=1e-4)`, `40` updates per trajectory, global gradient norm clip `L2 = 1.0`, and snapshots evaluated at steps `0, 10, 20, 30, 40`. Loss (`H59` SmoothL1 + `0.1` KL at $\tau = 0.2$) and frozen-anchor cache follow merged `H146`/`H156`/`H165`.
- **`None`-Gradient Guard (`07:26Z` Addendum)**: Any trainable adapter tensor whose `.grad` is `None` after `.backward()` is treated as zero in unclipped/clipped gradient-norm logging (`grad_norms` skips `None` gradients when forming the L2 norm and records active vs `None` gradient tensor counts), covered by a dedicated data-free unit test before launch.
- **First Scored Gate & Bit-Exact Eval Proof**: Before opening any `DL-Train` outcome pool or `DL-Train` context feature file, the runner evaluates the `91` native `DL-Val` contexts (`val_only_context_features_91.npz`) at step `0`, proving:
  1. Bit-exact equality (`torch.equal`) between native `net.adapter_10tok.predict_costs(...)` and the cached-anchor `adapter_forward(...)` with `InputFeatureNoise` in `.eval()` mode across all `91` `DL-Val` contexts, and
  2. Exact `91/91` native `r882` utility parity (`mean = 1.1377941898164525`, `max_error = 0.0`) against `best_dev_checkpoint.json`.
  A constructor/optimizer-only `--smoke` mode opens zero outcome pools and performs zero forwards/updates.
- **Minibatch Sampler**: Control minibatches follow merged `H156`/`H165`: `8` `fit`-fold contexts per step, `24` candidates each (`4` top-fitness reveals by raw fitness plus `20` uniformly sampled without replacement from the remaining candidates). Common minibatches are shared across all `4` arms within each `(seed, outer_fold)` cell.

## Registered Arms (`4` Arms $\times$ `3` Seeds $\times$ `5` Outer Folds = `60` Cells / `2,400` Updates)

Within each outer fold $k \in \{0..4\}$, let $\text{FIT}_k$ denote the `fit`-fold context set (`123` families, disjoint from `select`, `assess`, and `DL-Val`). Before any training in outer fold $k$, compute the per-feature fit-fold population standard deviation (`ddof = 0`) across $c \in \text{FIT}_k$:
$$s_j = \operatorname{std}_{c \in \text{FIT}_k}\!\bigl(x_{c, j}\bigr) \cdot \mathbf{1}[\text{modality slot of } j \text{ is present in } \text{FIT}_k], \quad j \in \{1..320\} \text{ (`10x32` tokens)},$$
and freeze $s$ (`shape (10, 32)`) for outer fold $k$. No `select`, `assess`, or `DL-Val` context is ever used to compute $s_j$.

All `4` arms initialize from `r882` and pass inputs through a parameter-free, buffer-free `InputFeatureNoise` site on the `10x32` multimodal context tokens entering the trainable adapter branch (`encode_multimodal_context_and_candidates`), while the cached frozen `FlywheelGameNet` anchor receives unmodified inputs:
1. **`CTRL`**: Clean input features (`scale = 0.0`; `InputFeatureNoise` inactive).
2. **`IN05`**: Additive Gaussian noise $\xi^{(t)}_{b, j} \sim \mathcal{N}\bigl(0, (0.05 \cdot s_j)^2\bigr)$ masked by `mm_mask_10`, drawn freshly on every training step $t \in \{1..40\}$ from an independent deterministic CPU generator seeded by `stream_seed(seed, outer_fold, "IN05", step)`.
3. **`IN10`**: Additive Gaussian noise $\xi^{(t)}_{b, j} \sim \mathcal{N}\bigl(0, (0.10 \cdot s_j)^2\bigr)$ masked by `mm_mask_10`, drawn freshly on every training step $t \in \{1..40\}$ from an independent deterministic CPU generator seeded by `stream_seed(seed, outer_fold, "IN10", step)`.
4. **`IN10_FIXED`**: Same coordinate scale $\sigma_j = 0.10 \cdot s_j$, but **one fixed noise draw per fit-fold context** $c \in \text{FIT}_k$, $\xi^{\text{fixed}}_{c, j} \sim \mathcal{N}\bigl(0, (0.10 \cdot s_j)^2\bigr)$ masked by `mm_mask_10[c]` (seeded by `stream_seed(seed, outer_fold, "IN10_FIXED", c)`), reused at every update $t \in \{1..40\}$ whenever context $c$ appears in the minibatch.

**Train-Mode Isolation**: Only `InputFeatureNoise` enters `.train(True)` during noisy training steps (`IN05`, `IN10`, `IN10_FIXED`). All evaluation, inner-selection, and assessment forwards set `InputFeatureNoise.eval()`, returning unmodified token tensors (`result is tokens`) and preserving bit-exact `r882` evaluation at step `0`. Other adapter modules, `anchor`, and `DOPE` remain in `.eval()` mode at all times. Every noisy training update records `changed_activations`, `active_inputs`, `noise_l2`, and `expected_noise_l2`, failing closed if `changed_activations == 0` or `noise_l2 <= 0`.

## Cross-Fit and Locked Selection

- **Cohorts**: `1,219` regression `DL-Train` contexts (`205` families) from approved real-S3-derived `H50`/`H58` split-only retained inputs (`train_only_selected_1219.json`, `train_only_context_features_1219.npz`); `91` `DL-Val` contexts (`19` disjoint families, `val_only_context_features_91.npz`) descriptive secondary only. Canonical S3 catalog origin join covers all `276` dataset IDs (`provenance.json`). Zero sealed/holdout/meta-test reads and zero reads of the mixed `1,310`-context `H58` cache.
- **Seeds & Partition**: Seeds `167101, 167102, 167103`, `fold_seed = 167140`, `boot_seed = 167141`. Sorted `205` `DL-Train` families are randomly partitioned (`fold_seed = 167140`) into `5` disjoint groups of `41` families (`fold_manifest.json`). For outer fold $A \in \{0..4\}$, selector fold is $B = (A + 1) \bmod 5$ (`41` families) and `fit` is the remaining `123` families. Each `DL-Train` context is assessed out-of-fold exactly once per seed.
- **Strict Selection Rule**: On selector fold $B$, choose the earliest step in `(0, 10, 20, 30, 40)` achieving the highest equal-family selector mean, treating differences $\le 10^{-12}$ as ties. Bootstrap that best step vs step `0` with `10,000` paired family draws (`stream_seed(BOOT_SEED, outer, seed, arm)`). Accept only if `gain > +0.001 + 1e-12` AND `95% LCB > 0`; otherwise select step `0` (`r882`). Do not search for a second step after rejection. Persist selection locks and SHA-256 with `os.fsync` before any trained-policy assessment on outer fold $A$ or `DL-Val`.
- **Descriptive Unconstrained & Drift Diagnostics**: The unconstrained inner-$B$ best step (`best_step`) is also evaluated on outer fold $A$ and `DL-Val` after the selection lock for descriptive reporting only (never used for advancement), alongside per-step and step-40 adapter L2 parameter drift $\|\theta_t - \theta_0\|_2$.

## Inference and Decision Rule

- **Primary Estimand**: Equal-family mean out-of-fold utility gain vs `r882` on `1,219`-context / `205`-family `DL-Train` for each seed, aggregated by the **median of the three scalar seed gains** (`10,000` shared family-bootstrap draws taking the median across the 3 resampled per-seed family means in each draw; percentile 95% CI; one-sided bootstrap $p$-value with Holm step-down across the `3` treatment arms `IN05`, `IN10`, `IN10_FIXED` vs `r882`, $m = 3$). We also report treatment-vs-`CTRL` contrasts and the mechanism contrast `IN10 - IN10_FIXED` (dynamic stochastic noise vs static perturbation).
- **Elo**: Canonical `H148` Bradley–Terry MM Elo (`1e-4` tie margin, `0.01` pseudo-win prior, `400` MM iterations, `r882 = 1500.0`, median over 3 seed ratings and `1,000` shared-family draws for 95% CI).
- **Secondary**: For each seed, average the `5` outer-fold selected policies' context utilities on the `91`-context / `19`-family `DL-Val` panel into one `91`-context vector before applying the same equal-family median-of-seeds summary.
- **Verdict**: `ADVANCE` only if any treatment arm (`IN05`, `IN10`, `IN10_FIXED`) achieves primary `DL-Train` equal-family median gain `> +0.001 + 1e-12`, 95% CI `LCB > 0`, and Holm-adjusted $p < 0.05$; otherwise `NO_ADVANCE`. No automatic serving promotion.

## Resources, Custody, and Flow

CPU only on `xbabe2`, `nice 19`, `1` Torch/math thread, aggregate supervisor+worker tree `<= 4` threads, aggregate `RSS < 6,000,000,000` B. Terminal stop on `load1 > 85`, `host CPU > 85%`, `MemAvailable < 16 GiB`, `free disk < 40 GB`, or `disk_used_fraction >= 0.95`. Exact `launch.json` path is read-only allowlisted before the audit hook and re-read after it; `audit.json` is atomically persisted in `finally:`. Raw panels, snapshots, and cell receipts are archived to versioned S3 (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H167/`) with byte-readback SHA verification; only small source/metrics/index files are committed to Git.
