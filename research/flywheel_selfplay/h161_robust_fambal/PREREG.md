# H161 Preregistration — Robust Per-Context Normalized/Huber Loss + Family-Balanced Sampling on `r882` Continuation

- **Wave**: `H161`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Authority**: `~/nudges/hwave_t0000_h161_rd_argon.md`
- **Claim**: `~/jope_plan/ns/receipts/H161_claim.txt`
- **Status**: `PREREGISTERED_NOT_EXECUTED` (committed and merged before any scored inference or optimizer update)

---

## 1. Scientific Hypothesis & Motivation

Diagnostics in `H143` and `H159` established that:
1. Every admitted flywheel checkpoint since `r774` moved `DL-Val` primarily via single `-2.0` invalid-gate cliff flips, while unconstrained `r882` continuations (`H144`, `H146`, `H148`, `H151`) fail strict family cross-fit admission (`0/15` cells admitted in `H148`).
2. Under the standard `H59`/`H140`/`H148` objective (`CTRL`), two sources of gradient concentration dominate:
   - **Per-context outlier loss concentration**: a small number of high-loss / `-2.0` gate-failure contexts dominate the batch-mean gradient.
   - **Family-size sampling skew**: uniform context sampling draws multi-context families proportionally to their context count (`H73` showed the top 10 of 205 families carry `23.55%` of loss mass on `10.91%` of contexts).

Prior waves tested single-fold/unrevealed-evaluator variants (`H125` quantile loss clipping on `H61` unrevealed evaluation; `H136` `FAMBAL_ACC4` on a single 41-family fold without `H148` 5-fold cross-fit or normalized Huber loss). Neither tested the factorial combination of **fit-fold median-normalized per-context Huber loss ($\delta = 1.0$)** and **equal-family minibatch sampling with gradient accumulation** on the canonical trajectory-conditioned `H148` 5-fold family cross-fit harness with `H151` drift logging.

**Hypothesis (`H161`)**: Normalizing each training context's loss by its detached fit-fold `r882` median scale and applying Huber ($\delta = 1.0$) (`CLIP`), drawing equal-family minibatches with gradient accumulation to keep update count fixed (`FAMBAL`), or combining both (`CLIP_FAMBAL`) on `r882` continuation (`40` AdamW updates per cell, `lr = 1e-4`, `wd = 1e-4`) produces checkpoints that pass the strict `H148` selection gate on fold `B` and improve out-of-fold equal-family utility on fold `A` over `r882` by $> +0.001$ with 95% family-bootstrap LCB $> 0$ across seeds `161101–161103`.

---

## 2. Experimental Design & Arms (`2x2` Factorial + Conditional Post-Hoc `CTRL_DRIFT`)

All cells start from an owned, SHA-verified copy of `round-882.pt` (`/home/ubuntu/jope-data/h161/inputs/round-882.pt`) and a fresh `AdamW` optimizer (`lr = 1e-4`, `weight_decay = 1e-4`, `clip_grad_norm_ = 1.0`), training **only the 47 adapter tensors (`29,211` parameters)** in `.eval()` mode (`0` effective dropout) for **40 optimizer updates per cell**. Frozen `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and all non-adapter tensors are verified bitwise before and after every update and evaluation.

| Arm | Per-Context Loss Objective | Minibatch Sampler (`40` updates/cell, `16` contexts/update) | Role |
| --- | --- | --- | --- |
| `CTRL` | Standard `H59`/`H148` mean loss: $\frac{1}{16}\sum_{c=1}^{16} L_c$ | Standard `H140`/`H148` context-uniform paired stream (`batch = 16`, `slate = 192`) | Unconstrained recipe control |
| `CLIP` | Normalized Huber ($\delta=1.0$): $\frac{1}{16}\sum_{c=1}^{16} \text{Huber}_{\delta=1.0}\!\left(\frac{L_c}{s_{\text{fit}}(A)}\right)$ | Identical `batch_plan` context/candidate stream as `CTRL` | Robust per-context loss treatment |
| `FAMBAL` | Standard `H59`/`H148` mean loss accumulated over $2\times 8$ microbatches | Equal-family sampler (`fambal_batch_plan`): `16` distinct fit families per step via stratified permutation cycling, `1` context/family, `2` microbatches of `8` contexts accumulated via `(loss / 2).backward()` before `optimizer.step()` | Equal-family sampling treatment |
| `CLIP_FAMBAL` | Normalized Huber ($\delta=1.0$) accumulated over $2\times 8$ microbatches | Identical `fambal_batch_plan` equal-family stream as `FAMBAL` | Joint robust loss + equal-family treatment |
| `CTRL_DRIFT` *(conditional)* | Standard `H59`/`H148` mean loss at `lr = 1e-4 * median_drift(winner) / median_drift(CTRL)` | Standard `batch_plan` | Executed automatically (`15` cells / `600` updates) if and only if at least one treatment arm (`CLIP`, `FAMBAL`, `CLIP_FAMBAL`) passes all primary efficacy gates vs `r882` |

### Per-Context Loss & Fit-Fold Scale Definition
For each context $c$ and candidate slate $i \in \{1, \dots, K\}$ with $\text{target}_{c,i} = \text{clip}(\text{fitness}_{c,i}, -4.0, 2.0)$ if $\text{gate}_{c,i}$ else $2.0$:
$$L_c = \frac{1}{K}\sum_{i=1}^K \text{SmoothL1}(\text{pred}_{c,i}, \text{target}_{c,i}) + 0.1 \sum_{i=1}^K \text{KL}\!\left(\text{softmax}(-\text{target}_c/0.2) \,\|\, \text{softmax}(-\text{pred}_c/0.2)\right)$$
For outer fold $A \in \{0, 1, 2, 3, 4\}$, let $\text{fit}(A)$ denote the 3 fit folds ($123$ families, strictly excluding selection fold $B$, assessment fold $A$, and `DL-Val`). The detached fit-fold median scale is:
$$s_{\text{fit}}(A) = \max\!\left(\text{median}_{c \in \text{fit}(A)} L_c^{(0)},\; 10^{-6}\right)$$
where $L_c^{(0)}$ is the step-0 `r882` per-context loss on context $c \in \text{fit}(A)$. In `CLIP` and `CLIP_FAMBAL`, the normalized per-context loss $u_c = L_c / s_{\text{fit}}(A) \ge 0$ is passed through `F.huber_loss(u, 0, reduction="mean", delta=1.0)`, capping $\partial \mathcal{L} / \partial L_c$ at $1 / (B \cdot s_{\text{fit}}(A))$ whenever $L_c > s_{\text{fit}}(A)$.

---

## 3. Family Cross-Fit Harness (`H148`) & Strict Checkpoint Selection

- **Cohort**: `1,219` contexts across `205` families in real-S3 `DL-Train` (`source_split in {"train", "inner"}`, `has_meta_test == False`), plus `91` contexts across `19` disjoint families in `DL-Val`.
- **Seeds**:
  - Training / sampling seeds: `SEEDS = (161101, 161102, 161103)`
  - 5-fold family partition seed: `FOLD_SEED = 161140` (`5` folds of `41` families each, context counts `[232, 261, 267, 224, 235]`, pinned in [fold_manifest.json](fold_manifest.json))
  - Family bootstrap seed: `BOOT_SEED = 161141` (`10,000` shared family draws; Elo bootstrap `161142` with `1,000` shared family draws)
- **Nested Fold Roles**: For each outer fold $A \in \{0, 1, 2, 3, 4\}$:
  - Selection fold: $B = (A + 1) \bmod 5$ (`41` families)
  - Fit folds: the remaining 3 folds (`123` families)
  - Assessment fold: $A$ (`41` families)
- **Strict Selection Rule (`choose`)**:
  - Snapshots are taken at steps `STEPS = (0, 10, 20, 30, 40)` and scored on selection fold $B$ by equal-family mean native utility ($-\text{fitness}$ if valid gate, else $-2.0$).
  - Let $s^*$ be the earliest step maximizing equal-family mean utility on $B$.
  - $s^*$ is selected (`selected_step = s*`) if and only if its paired equal-family gain over step `0` on $B$ satisfies **$\text{gain} > 0.001 + 10^{-12}$** AND its `10,000`-draw paired family-bootstrap 95% lower confidence bound on $B$ satisfies **$\text{LCB}_{95} > 0$** (seed `161141 + 100*A + 10*seed_index + arm_index + 1`). Otherwise `selected_step = 0` (`r882`).
  - A durable selection lock file (`{arm}-A{outer}-{seed}-selection.json`) is written and SHA-256 hashed before scoring assessment fold $A$.

---

## 4. Endpoints, Drift Logging & Decision Rules

1. **Exact Baseline Gate (Hard Precondition)**:
   - Before any training update or `DL-Train` scoring, `r882` is evaluated on all `91` `DL-Val` contexts and must reproduce `best_dev_checkpoint.json` with `91/91` exact matches (`max_context_error == 0.0`, `mean_error == 0.0`, context mean `1.1377941898164525`), and bitwise parity (`0.0` difference) between uncached and frozen-anchor-cached evaluation.
2. **Primary Endpoint (Out-of-Fold `DL-Train` Assessment Panel, `1,219` contexts / `205` families)**:
   - For each arm and seed, the 5 disjoint assessment folds $A \in \{0..4\}$ assemble into a complete `1,219`-context / `205`-family out-of-fold assessment panel.
   - Primary contrast for each arm vs `r882`: equal-family mean gain within each seed, **median across the 3 seeds**, and `10,000`-draw shared 205-family bootstrap 95% percentile CI (`[q025, q975]`) and one-sided $p = (1 + \sum \mathbb{I}[\Delta^{(b)} \le 0]) / 10001$.
   - Canonical Bradley-Terry Elo vs `r882 = 1500` (`400` MM iterations, prior `0.01`, family-averaged pairwise context wins with tie margin `1e-4`, `1,000` shared family bootstrap draws).
   - Report `-2.0` invalid pick counts per arm and seed, number of non-zero step selections (`out of 15` cells per arm), and per-arm step-40 and selected-step L2 parameter drift (`H151` drift logging), plus selector-fold ($B$) step-10/20/30/40 diagnostic trajectories.
3. **Secondary Endpoint (Descriptive Only)**:
   - `91`-context / `19`-family `DL-Val` utility averaged across the 5 outer selected policies per seed, with equal-family bootstrap CI, canonical BT Elo vs `r882 = 1500`, and `-2.0` invalid pick counts.
4. **Decision Rule**:
   - Holm step-down ($m = 3$) across the 3 treatment arms `TREATMENT_ARMS = ("CLIP", "FAMBAL", "CLIP_FAMBAL")` vs `r882`.
   - Primary efficacy gates for a treatment arm:
     1. `median_gain > +0.001 + 1e-12` (ties rejected);
     2. `ci95[0] > 0.0`;
     3. `positive_seeds == 3` (`3/3` seeds $> 0$);
     4. `top3_positive_share <= 0.80`;
     5. `holm_adjusted_p < 0.05`;
     6. `nonzero_selection > 0`.
   - Post-hoc drift-matched control gate (`H151` lesson): if any treatment arm passes the 6 primary efficacy gates, `CTRL_DRIFT` is run at `lr = 1e-4 * median_drift(winner) / median_drift(CTRL)` (`15` cells / `600` updates), and `ADVANCE` requires `0.85 <= median_drift(CTRL_DRIFT) / median_drift(winner) <= 1.15` AND `winner - CTRL_DRIFT` median equal-family gain $> 0$ with 95% CI LCB $> 0$.
   - Verdict is `RULE_NEUTRAL` if all `60` cells select step `0`, `ADVANCE` if a treatment arm passes all primary efficacy and `CTRL_DRIFT` gates, and `NO_ADVANCE` otherwise.

---

## 5. Resource & Custody Envelope

- **Host**: `xbabe2` CPU only (`CUDA_VISIBLE_DEVICES=""`, no GPU, no `qualification.lock`, never `xbabe0`/`xbabe3`).
- **Limits**: `nice 19`, `1` numerical thread (`OMP`/`MKL`/`OPENBLAS`/`torch` = `1`), `<= 4` total OS threads across supervisor + worker tree, `RSS < 6,000,000,000 B`, `load1 <= 85.0`, `MemAvailable >= 16 GiB`, `free disk >= 40 GB`, `user disk used < 95%`, owned scratch `< 3 GB`.
- **Single Attempt & Custody**: One supervised run after preregistration merge; bulk cell receipts and snapshots archived to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H161/argon-h161-20261101a/` and restore-verified before scratch cleanup.
