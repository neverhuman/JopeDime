# H173 Preregistration — Hard-Context (Regret-Weighted / Focal-Style) Loss Reweighting on `r882` Continuation

- **Wave**: `H173` (`research/flywheel_selfplay/h173_hard_context_reweight/`)
- **Owner**: `rd_argon` (`[ARGON]`), claim `/home/ubuntu/jope_plan/ns/receipts/H173_claim.txt` (`reserved_by: ENG:hwave-loop`)
- **Status**: `PREREGISTERED_NOT_EXECUTED`
- **Fresh Seeds (`no reuse of seen panels`)**:
  - `SEEDS = (173101, 173102, 173103)`
  - `FOLD_SEED = 173140` (`5` disjoint folds of `41` families across the `1,219`-context / `205`-family `DL-Train` cohort; fold context counts `[240, 242, 258, 273, 206]`)
  - `BOOT_SEED = 173141` (`10,000` bootstrap draws on outer fold $A$, `1,000` draws on inner selector fold $B$ and Bradley-Terry Elo)

---

## 1. Hypothesis & Motivation

Recent `r882` continuation waves (`H148`, `H160`, `H161`, `H162`, `H168`, `H170`) returned `RULE_NEUTRAL` or `NO_ADVANCE`. In particular, `H168` showed that under uniform per-context weighting ($w_c = 1.0$) with standard `lr=1e-4` AdamW continuation updates, removing the 41-family inner-`B` `LCB95 > 0` gate admitted non-zero checkpoints (`7/15` on `W10`, `9/15` on `W20`) that degraded held-out outer-fold $A$ equal-family utility (`W10_NEW - r882 = -0.006089`, `W20_NEW - r882 = -0.007748`) and increased median `-2` invalid cliff selections from `17.0` to `20.0`.

Most `DL-Train` contexts already have near-optimal `r882` top-1 picks ($\text{regret}_c \approx 0$). Under uniform context weighting ($w_c = 1.0$), optimizer gradients across each `8`-context minibatch are dominated by already-solved contexts, causing drift on the tail of hard / high-regret contexts where `r882` makes suboptimal or gate-invalid (`-2` cliff) picks.

**Hypothesis (`H173`)**: Reweighting each `FIT`-split context's continuation loss by a clipped power of its frozen `r882` step-0 top-1 regret computed **strictly within the `FIT` split $C_k$** (`RW05` with $\gamma = 0.5$, `RW10` with $\gamma = 1.0$) concentrates continuation updates on high-regret `FIT` contexts, yielding positive out-of-fold equal-family utility on outer fold $A$ under the strict inner-`B` selector (`gain > +0.001 + 1e-12` and `1,000`-draw paired family-bootstrap `LCB95 > 0`), whereas a **shuffled-weight placebo (`RW10_SHUF`)** with the **exact same weight distribution and effective sample size (`ESS`)** permuted randomly across `FIT` contexts does not.

---

## 2. Prespecified Design: 5-Fold Family Cross-Fit $\times$ 4 Paired Reweighting Arms (`CTRL`, `RW05`, `RW10`, `RW10_SHUF`)

1. **5-Fold Family Cross-Fit (`FOLD_SEED = 173140`)**:
   - The `205` real-S3 `DL-Train` families (`1,219` contexts) in `/home/ubuntu/jope-data/h50/input/selected.json` are partitioned into `5` disjoint folds of `41` families (`fold_manifest.json`, fold context counts `[240, 242, 258, 273, 206]`).
   - For each outer fold $k \in \{0, 1, 2, 3, 4\}$:
     - **Outer assessment fold $A_k$**: fold $k$ (`41` families)
     - **Inner selector fold $B_k$**: fold $(k + 1) \bmod 5$ (`41` families)
     - **Fit split $C_k$ (`FIT`)**: the remaining `3` folds (`123` families)
   - Disjointness between $C_k$, $B_k$, $A_k$, and the `91`-context / `19`-family `DL-Val` cohort is asserted before every cell and on every minibatch.

2. **Leak-Free Per-Context Regret Weights on `FIT` ($C_k$) Only (`compute_fit_regret_weights`)**:
   - Before any training in fold $k$, for each context $c \in C_k$ (`FIT` split **only**), let $\text{cost}_c(j) = \text{fitness}_j$ if $\text{gate}_j$ else $2.0$ ($= -\text{utility}_j$) be the realized candidate cost on context $c$, and let $i_{\text{r882}}(c) = \arg\min_j \hat{s}_{\text{r882}}(c, j)$ be the frozen `r882` step-0 top-1 candidate pick.
   - Define the step-0 `FIT` regret:
     $$\text{regret}_c = \text{cost}_c(i_{\text{r882}}(c)) - \min_{j} \text{cost}_c(j) \ge 0, \qquad c \in C_k$$
   - Let $\varepsilon_k = \max\left(\text{median}_{c \in C_k}(\text{regret}_c),\; 10^{-6}\right)$, computed **using `FIT` contexts $c \in C_k$ only** (never $B_k$, $A_k$, or `DL-Val`).
   - For exponent $\gamma \in \{0.0, 0.5, 1.0\}$:
     1. Raw power weight: $u_c(\gamma) = (\text{regret}_c + \varepsilon_k)^\gamma$ for $c \in C_k$.
     2. Normalize to `FIT` mean `1.0`: $\tilde{w}_c(\gamma) = \frac{u_c(\gamma)}{\frac{1}{|C_k|}\sum_{c' \in C_k} u_{c'}(\gamma)}$.
     3. Clip to $[0.2, 5.0]$: $\bar{w}_c(\gamma) = \text{clip}(\tilde{w}_c(\gamma), 0.2, 5.0)$.
     4. Renormalize to exact `FIT` mean `1.0`: $w_c(\gamma) = \frac{\bar{w}_c(\gamma)}{\frac{1}{|C_k|}\sum_{c' \in C_k} \bar{w}_{c'}(\gamma)}$.
   - Record per-split effective sample size $\text{ESS}(w) = \frac{(\sum_{c \in C_k} w_c)^2}{\sum_{c \in C_k} w_c^2}$ and $\text{ess\_ratio}(w) = \text{ESS}(w) / |C_k|$.

3. **Four Paired Arms per `(outer, seed)` Cell (`15` cells $\times$ `4` arms $= 60$ trajectories, `2,400` AdamW updates)**:
   - All 4 arms start from the exact `strict=True` `r882` (`1.0x`, `29,211` trainable adapter parameters across `47` tensors) checkpoint (`/home/ubuntu/jope-data/h173/inputs/round-882.pt`) and share the exact same `40`-step common-random-numbers minibatch plan (`batch_size = 8` contexts/step, `top = 24` candidates = `4` revealed + `20` sampled, `lr = 1e-4`, `weight_decay = 1e-4`, `clip_grad_norm_ = 1.0` via `None`-safe `safe_grad_norm_and_clip`, snapshots at `STEPS = (0, 10, 20, 30, 40)`):
     1. **`CTRL` ($\gamma = 0.0$, uniform $w_c = 1.0$)**: Exact unweighted `H146`/`H156` continuation loss $\mathcal{L}_{\text{batch}} = \frac{1}{|B_{\text{mb}}|}\sum_{c \in B_{\text{mb}}} \ell_c$, where $\ell_c = \text{SmoothL1}(\hat{y}_c, y_c) + 0.1 \cdot \text{KL}\!\left(\text{softmax}(-y_c / 0.2) \,\|\, \text{softmax}(-\hat{y}_c / 0.2)\right)$ with valid target $y_{c,j} = \text{clip}(\text{fitness}_{c,j}, -4.0, 2.0)$ and invalid target $2.0$.
     2. **`RW05` ($\gamma = 0.5$)**: Regret-weighted continuation loss normalized within each minibatch $B_{\text{mb}} \subset C_k$:
        $$\mathcal{L}_{\text{batch}}(w) = \frac{\sum_{c \in B_{\text{mb}}} w_c \ell_c}{\sum_{c \in B_{\text{mb}}} w_c}$$
        (preserving exact learning-rate scale when $w_c \equiv 1.0$).
     3. **`RW10` ($\gamma = 1.0$)**: Linear regret-weighted continuation loss with $w_c = w_c(1.0)$ computed on $C_k$.
     4. **`RW10_SHUF` (Placebo / Variance-Inflation Control)**: Permutes the `RW10` weight vector $(w_c(1.0))_{c \in C_k}$ across `FIT` contexts $c \in C_k$ using a deterministic seed-fixed permutation (`seed + 1000 * outer + 77`), preserving the **exact weight distribution, min/max/quantiles, and `ESS`** of `RW10` while breaking the alignment between $w_c$ and context hardness $\text{regret}_c$.

4. **Strict Inner-`B` Checkpoint Selector Locked Before Outer Fold $A$ (`choose`)**:
   - On inner fold $B_k$ (`41` families), each candidate step $t \in \{0, 10, 20, 30, 40\}$ is evaluated by unweighted equal-family mean gain $\Delta_{B_k}(a, t) = \frac{1}{41}\sum_{f \in B_k}(\bar{u}_f(a, t) - \bar{u}_f(r882))$ and a `1,000`-draw paired family-bootstrap 95% CI $[\text{LCB}_{95, B_k}(a, t), \text{UCB}_{95, B_k}(a, t)]$ (`seed = BOOT_SEED + 1000 * outer + (seed % 1000)`).
   - **Strict Primary Selector (`selected_step`)**: Step $t \in \{10, 20, 30, 40\}$ is eligible iff $\Delta_{B_k}(a, t) > +0.001 + 10^{-12}$ **AND** $\text{LCB}_{95, B_k}(a, t) > 0.0$. Selects the earliest eligible step maximizing $\Delta_{B_k}(a, t)$ (ties within $10^{-12}$ resolved to earliest step), else `step = 0` (`r882`).
   - **Pre-Locked Descriptive Selectors**:
     - `unconstrained_selected_step`: earliest step $t \in \{0, 10, 20, 30, 40\}$ maximizing $\Delta_{B_k}(a, t)$ (`> 1e-12` over step 0).
     - `nonzero_selected_step`: earliest non-zero step $t \in \{10, 20, 30, 40\}$ maximizing $\Delta_{B_k}(a, t)$.
   - All selector choices are written and `fsync`ed to `A{outer}-{seed}-selection.json` **before** any non-zero checkpoint is evaluated on outer fold $A_k$ or `DL-Val`.
   - Raw per-context inner-`B` utility panels (`shape = (len(B_k), 5)`) for all `60` `(outer, seed, arm)` trajectories are persisted in `inner_b_panels.npz` so that every inner-`B` bootstrap CI and selector decision is independently replayable from retained raw arrays.

---

## 3. Primary Estimand, Multiplicity, Placebo Contrasts & Decision Gates

- **Primary Cohort**: `1,219` contexts / `205` families on `DL-Train` assembled out-of-fold across $A_0 \dots A_4$ (`historical_r882_train_exposure = True` disclosed).
- **Secondary Cohort (Descriptive Only)**: `91` contexts / `19` families on `DL-Val` (exact `91/91` `r882` baseline gate `1.1377941898164525`, `max_context_error = 0.0`, and `104,832`-candidate `total_cost` bit-identity verified before training and in `--smoke`).
- **Primary Seed Estimand (`shared_family_median_seed`)**:
  - For each seed $s \in \{0, 1, 2\}$ and arm $a \in \{\text{CTRL}, \text{RW05}, \text{RW10}, \text{RW10\_SHUF}\}$, let $G_s(a) = \frac{1}{205}\sum_{f=1}^{205}(\bar{u}_{s,f}(a) - \bar{u}_{s,f}(r882))$.
  - Primary point estimate: $\text{median\_gain}(a) = \text{median}_{s \in \{0,1,2\}} G_s(a)$, with `10,000`-draw shared-family bootstrap (`BOOT_SEED = 173141`) `95%` CI and one-sided $p$-value.
- **Multiplicity & Decision Rule (`decide`)**:
  - Holm-Bonferroni adjustment across the $m = 3$ reweighted arms (`RW05`, `RW10`, `RW10_SHUF`) as primary (`holm_adjusted_p`), plus conservative $m = 4$ Holm across all 4 arms including `CTRL` (`holm_m4_adjusted_p`).
  - A hardness-reweighted arm $a \in \{\text{RW05}, \text{RW10}\}$ passes `advance = True` iff on the 205-family out-of-fold panel:
    1. `median_gain > +0.001 + 1e-12`
    2. `ci95[0] > 0.0` (`LCB95 > 0`)
    3. `positive_seeds >= 2`
    4. `holm_adjusted_p < 0.05`
    5. `nonzero_selection > 0`
  - **Paired Mechanism & Placebo Contrasts**: `RW05_minus_CTRL`, `RW10_minus_CTRL`, `RW10_SHUF_minus_CTRL`, and **`RW10_minus_RW10_SHUF`** (direct test of hardness alignment vs weight-variance inflation at identical `ESS`), plus canonical 400-iteration Bradley-Terry Elo (`r882 = 1500.0`, `tie_margin = 1e-4`, `prior = 0.01`, `1,000`-draw shared-family bootstrap 95% CI) and per-arm seed pseudoreplication diagnostics (`pseudoreplication_report`).

---

## 4. Resource & Safety Envelope (`xbabe2` CPU Only)

- **Hard stops (`monitor.py`)**: `nice 19`, aggregate supervisor + worker subtree (`psutil.Process(os.getpid())` plus worker and all children) capped at `<= 4` OS threads and `RSS < 6.0 GB`, `free disk >= 40 GB`, `disk_used < 95%`, `scratch < 3.0 GB`, `load <= 85.0`, `MemAvailable >= 16 GiB`, `wall <= 12h`, `CUDA_VISIBLE_DEVICES=""`.
- **External launch binding (`verify_external_launch_binding`)**: Before non-smoke execution, `runner.py` requires an external `launch.json` in the custody parent directory binding `merged_commit` (checked against `git rev-parse HEAD`), `review_id`, `manifest_sha256`, `prereg_sha256`, and `fold_manifest_sha256`.
- **Single-attempt rule**: One supervised full wave after preregistration review and merge; if it aborts after baseline scoring, the wave is recorded as `INCOMPLETE` without silent retry.
