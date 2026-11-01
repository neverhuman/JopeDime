# H172 Preregistration: Anchored Convex Last-Layer Refit + Family Bagging on Frozen `r882` Penultimate Features

- **Wave**: `H172`
- **Owner**: `gpu_tokbench_agy`
- **Authority**: `/home/ubuntu/nudges/hwave_t0225_h172_tokbench_agy.md`
- **Claim receipt**: `/home/ubuntu/jope_plan/ns/receipts/H172_claim.txt`
- **Status**: `PREREGISTERED` (pre-execution; zero `TRAIN` optimization or assessment scoring executed prior to preregistration merge)
- **Seeds**: `SEEDS = (172101, 172102, 172103)`, `FOLD_SEED = 172140`, `BOOT_SEED = 172141`

---

## 1. Scientific Rationale & Hypothesis

Across 25+ SGD-continuation waves (`H46`–`H171`: EMA/SWA, SAM, input/gradient noise, dropout, C-Mixup, rank smoothing, weight decay, warm restarts, optimizer geometry, lookahead, EWC, and `BITFIT`-by-SGD `H132`), short stochastic gradient trajectories on the `29,211`-parameter `adapter_10tok` have consistently overfit fit-fold contexts and fallen back to `step = 0` (`r882`) under the strict inner-fold selector (`gain > +0.001` and `LCB95 > 0`).

**Untested mechanism**: **Deterministic, fully-converged, strictly convex anchored last-layer refit on frozen `r882` penultimate features** (Deep Feature Reweighting / DFR style, Kirichenko et al., ICLR 2023), combined with **inner 5-fold family cross-validation** over a continuous anchor grid $\lambda \in \{10^4, 10^3, 10^2, 10^1, 1, 10^{-1}, \infty\}$ (`\lambda = \infty` $\equiv$ exact `r882`) and **8-model family-bootstrap bagging (`REFIT_BAG`)**. By freezing all internal representations at `r882` and solving the strictly convex output-head objective to gradient norm $\le 10^{-6}$ over full candidate pools with `L-BFGS`, candidate generation eliminates both SGD trajectory noise and candidate-subsampling variance.

---

## 2. Exact Output-Layer Parameter Subset (`H132` Subset) & Convex Formulation

1. **Frozen Backbone & `H132` Output-Layer Subset**:
   - All `222` DOPE and frozen-anchor tensors (`dope_sha256 = c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, `frozen_sha256 = 8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`) and all `41` non-output `adapter_10tok` tensors (`29,112` parameters) remain **100% frozen at `r882`**.
   - The `H132` output-layer subset comprises the `3` `nn.Linear(..., 1)` scalar readout heads of `adapter_10tok` (**`6` tensors, `99` total parameters = `96` weight parameters + `3` bias parameters**):
     - `adapter_10tok.mm_prior_out.2.weight` (`shape (1, 24)` = `24` params) and `adapter_10tok.mm_prior_out.2.bias` (`shape (1,)` = `1` param)
     - `adapter_10tok.traj_delta_head.2.weight` (`shape (1, 24)` = `24` params) and `adapter_10tok.traj_delta_head.2.bias` (`shape (1,)` = `1` param)
     - `adapter_10tok.parent_delta_net.3.weight` (`shape (1, 48)` = `48` params) and `adapter_10tok.parent_delta_net.3.bias` (`shape (1,)` = `1` param)
   - **Active Ranking Coordinates vs Structural Zero-Gradient Coordinates (`07:26Z` Addendum Guard)**:
     - `parent_delta_net.3.{weight,bias}` (`49` params) is outside the `predict_costs` candidate-ranking graph (`grad = None / 0.0`). Under any positive anchor $\lambda \|W - W_{r882}\|_2^2$ ($\lambda > 0$), its exact unique minimizer is $w_{\text{parent}}^* = w_{\text{parent}, r882},\; b_{\text{parent}}^* = b_{\text{parent}, r882}$ (`0.0` drift).
     - `mm_prior_out.2.bias` (`1` param) and `traj_delta_head.2.bias` (`1` param) are canceled identically by `adapter_10tok`'s within-slate mean centering (`raw - mean(raw)`), so $\nabla_b \mathcal{L}(W) \equiv 0$ and their exact unique minimizer for every $\lambda > 0$ is $b_{\text{mm}}^* = b_{\text{mm}, r882},\; b_{\text{traj}}^* = b_{\text{traj}, r882}$ (`0.0` drift).
     - The **`48` active ranking weight parameters** are $w = (w_{\text{mm}}, w_{\text{traj}}) \in \mathbb{R}^{48}$ (`24 + 24`), parameterized by displacement $\delta = w - w_{r882} = (\delta_{\text{mm}}, \delta_{\text{traj}}) \in \mathbb{R}^{48}$ with exact 99-parameter Euclidean head drift $\|W - W_{r882}\|_2 = \|w - w_{r882}\|_2 = \|\delta\|_2$.

2. **Precomputed Frozen `r882` Penultimate Features (DFR Readout)**:
   - For each context $c$ with full candidate pool size $M_c$, a single frozen `r882` forward pass extracts:
     - Native `r882` candidate costs $c_{r882}(c) \in \mathbb{R}^{M_c}$ (`torch.equal` bit-identical to `parent.predict`),
     - Penultimate 24D activations $\Phi_{\text{mm}}(c) \in \mathbb{R}^{M_c \times 24}$ (input to `mm_prior_out.2`) and $\Phi_{\text{traj}}(c) \in \mathbb{R}^{M_c \times 24}$ (input to `traj_delta_head.2`).
   - Defining the scaled, slate-centered penultimate feature matrix:
     $$X_c = \Bigl[\tanh(\text{mm\_prior\_scale}_{r882})\bigl(\Phi_{\text{mm}}(c) - \overline{\Phi}_{\text{mm}}(c)\bigr),\;\; \tanh(\text{traj\_attn\_scale}_{r882})\bigl(\Phi_{\text{traj}}(c) - \overline{\Phi}_{\text{traj}}(c)\bigr)\Bigr] \in \mathbb{R}^{M_c \times 48}$$
     the DFR candidate cost prediction for any head parameter vector $W$ is the exact affine map:
     $$\hat{c}(c; W) = c_{r882}(c) + X_c (w - w_{r882})$$
     At $W = W_{r882}$ ($\lambda = \infty$, `CTRL`), $\hat{c}(c; W_{r882}) \equiv c_{r882}(c)$ is `torch.equal` bit-identical to native `r882`. (As an additional descriptive diagnostic, `run.py` also evaluates full-graph module writeback into `net.adapter_10tok` where `mm_prior_adj` re-enters `eff_prior`.)

3. **Strictly Convex Objective & Converged `L-BFGS` Optimization**:
   - For each context $c$ with gate-resolved clipped target $t_c = \operatorname{where}(\text{gate}_c, \operatorname{clip}(\text{fitness}_c, -4.0, 2.0), 2.0)$ and target Boltzmann distribution $p_c = \operatorname{softmax}(-t_c / 0.2)$, the live adapter loss (`rt.loss` from `H140`/`H146`) is:
     $$\ell_c(\hat{c}_c, t_c) = \frac{1}{M_c}\sum_{m=1}^{M_c} \operatorname{SmoothL1}(\hat{c}_{c,m} - t_{c,m}) + 0.1 \sum_{m=1}^{M_c} p_{c,m}\bigl(\log p_{c,m} - \log \operatorname{softmax}_m(-\hat{c}_c / 0.2)\bigr)$$
   - Both $\operatorname{SmoothL1}$ and the listwise KL / log-sum-exp term are convex and $C^1$-smooth in $\hat{c}_c$; composed with the linear map $\delta \mapsto c_{r882}(c) + X_c \delta$ and the quadratic anchor $\lambda \|\delta\|_2^2$ ($\lambda > 0$), the weighted fit objective over any fit context set $S \subseteq F_k$ with nonnegative weights $\omega_c$ ($\sum_{c \in S} \omega_c = 1$):
     $$\mathcal{J}_{S, \omega, \lambda}(\delta) = \sum_{c \in S} \omega_c\, \ell_c\bigl(c_{r882}(c) + X_c \delta,\, t_c\bigr) + \lambda \|\delta\|_2^2$$
     is **strictly convex** with a unique global minimizer $\delta^*(\lambda)$.
   - Solved in `torch.float64` using deterministic `torch.optim.LBFGS` (`lr = 1.0`, `max_iter = 500`, `tolerance_grad = 1e-6`, `tolerance_change = 0.0`, `history_size = 20`, `line_search_fn = "strong_wolfe"`, warm-started along the descending $\lambda$ grid) with explicit post-solve verification that $\|\nabla_\delta \mathcal{J}_{S, \omega, \lambda}(\delta^*)\|_2 \le 10^{-6}$.

---

## 3. Experimental Design, Arms & Anti-Leakage Protocol

- **Outer 5-Fold Family-Grouped Split (`FOLD_SEED = 172140`)**:
  - `1,219` `H145` `TRAIN` contexts across `205` families are partitioned into `5` disjoint outer folds $G_0, \dots, G_4$ of `41` families each (`fold_manifest.json`).
  - For each outer fold $k \in \{0, 1, 2, 3, 4\}$:
    - **Outer assessment fold $A_k = G_k$**: `41` families (held out until all fit-split CV, $\lambda$ selection, and head refits for fold $k$ are locked).
    - **Outer FIT split $F_k = \bigcup_{j \neq k} G_j$**: `164` families (`4` outer folds). **All $\lambda$ selection and head estimation use $F_k$ only (`H169` leak lesson).**
- **Seed-Driven Inner 5-Fold Family CV on $F_k$ (`SEEDS = (172101, 172102, 172103)`)**:
  - For each `(outer = k, seed = s)`:
    - Partition the `164` sorted families of $F_k$ into `5` disjoint inner family folds $I_{k,s,0}, \dots, I_{k,s,4}$ (`33, 33, 33, 33, 32` families) using `np.random.default_rng(stream_seed("inner_cv", s, k))`.
    - For each inner fold $m \in \{0, \dots, 4\}$ and each $\lambda \in \Lambda_{\text{grid}} = (10^4, 10^3, 10^2, 10^1, 1.0, 10^{-1})$, fit $\delta_{k,s,m}^*(\lambda)$ on the inner-train contexts $F_k \setminus I_{k,s,m}$ (uniform context weights $\omega_c = 1 / |F_k \setminus I_{k,s,m}|$) and evaluate candidate utilities on inner-validation contexts $I_{k,s,m}$.
    - Pool out-of-inner-fold utility predictions across $m = 0..4$ covering all `164` families in $F_k$.
    - For each $\lambda \in \Lambda_{\text{grid}}$, compute the `164`-family equal-family mean utility gain vs `r882` ($\lambda = \infty$) and its 1,000-draw family-bootstrap 95% lower confidence bound (`ci_low_95`) under `stream_seed("inner_boot", s, k, lam_idx)`.
    - **Strict inner-CV rule (`selected_lambda`)**: $\lambda \in \Lambda_{\text{grid}}$ is eligible iff `delta > +0.001 + 1e-12` AND `ci_low_95 > 0.0` (ties within `1e-12` rejected $\to \lambda = \infty$). Among eligible $\lambda$, select $\lambda^*(k, s)$ maximizing inner-CV equal-family gain (ties broken toward larger $\lambda$); if none is eligible, $\lambda^*(k, s) = \infty$ (`r882`).
    - Also record unconstrained inner-CV best $\lambda_{\text{uncon}}^*(k, s)$ (`delta > 1e-12`, else $\infty$) and best finite $\lambda_{\text{finite}}^*(k, s) \in \Lambda_{\text{grid}}$.
- **Four Arms (`CTRL`, `REFIT_CV`, `REFIT_BAG`, `REFIT_SHUF`)**:
  1. `CTRL`: `r882` unchanged ($\lambda = \infty$, $\delta = 0$).
  2. `REFIT_CV`: Single converged L-BFGS refit $\delta_{\text{CV}}^*(k, s) = \delta_{F_k}^*(\lambda^*(k, s))$ on the full `164`-family fit split $F_k$ at the inner-CV selected $\lambda^*(k, s)$ ($\delta = 0$ if $\lambda^*(k, s) = \infty$).
  3. `REFIT_BAG`: Mean head $\delta_{\text{BAG}}^*(k, s) = \frac{1}{8}\sum_{b=0}^{7} \delta_{\text{bag}, b}^*(k, s, \lambda^*(k, s))$ of `8` converged L-BFGS refits on family-bootstrap resamples of the `164` families in $F_k$ (`stream_seed("bag_families", s, k, b)`, where each context $c \in F_k$ receives weight proportional to its family's bootstrap draw count $m_{b, \text{fam}(c)}$) at the `REFIT_CV` $\lambda^*(k, s)$ ($\delta = 0$ if $\lambda^*(k, s) = \infty$). Because $\hat{c}(c; W)$ is linear in $\delta$, averaging the 8 head vectors is mathematically identical to averaging the 8 ensemble cost predictions.
  4. `REFIT_SHUF` (within-family target-permuted placebo): Identical pipeline to `REFIT_CV` (inner 5-fold family CV on $F_k$ + full-$F_k$ refit), except that within each fit-split context $c \in F_k$ (within its family in $F_k$), the candidate target pairs $(\text{fitness}_c, \text{gate}_c)$ are randomly permuted across candidates using `np.random.default_rng(stream_seed("shuf_placebo", s, k, cid))` prior to fitting. Inner-CV validation utilities are evaluated against the true unpermuted inner-validation targets on $I_{k,s,m}$. `REFIT_SHUF` must not advance; if it advances, the harness leaks.

---

## 4. Endpoints, Statistical Protocol, Pseudoreplication Check & Hard Rules

- **Primary Endpoint & `ADVANCE` Gate**:
  - Out-of-fold equal-family utility gain vs `r882` across all `205` families (`1,219` contexts) of `H145` `TRAIN`: **median over the `3` seeds of per-seed equal-family mean gains**, with 10,000-draw shared-family bootstrap 95% CI of that median-over-seeds estimator under `BOOT_SEED = 172141` (`H148` protocol), Holm-Bonferroni step-down across the `2` active treatment arms (`REFIT_CV`, `REFIT_BAG`), and canonical `H145`/`H148` 400-iteration MM Bradley-Terry Elo anchored at `r882 = 1500.0` (`tie_margin = 1e-4`, `prior = 0.01`, strength clip `[1e-6, 1e6]`, 1,000-draw shared-family bootstrap 95% CI).
  - **`ADVANCE` iff**:
    1. `REFIT_SHUF` placebo does **not** pass the advancement criteria (`placebo_pass == False`), AND
    2. At least one active treatment arm (`REFIT_CV` or `REFIT_BAG`) achieves median out-of-fold equal-family gain `> +0.001 + 1e-12`, 10,000-draw shared-family bootstrap `ci_low_95 > 0.0`, AND 2-arm Holm-adjusted one-sided $p_{\text{Holm}} < 0.05$. Otherwise `NO_ADVANCE`.
- **Secondary & Descriptive Readouts**:
  - `91`-context / `19`-family `DL-Val` equal-family utility gain vs `r882` (averaged across the 5 outer-fold models per seed, then summarized across the 3 seeds via the same median-of-seed-gains + 10,000-draw shared-family bootstrap estimator).
  - Unconstrained inner-CV selector (`\lambda_{\text{uncon}}^*`, `delta > 1e-12` on $F_k$) and best-finite inner-CV selector (`\lambda_{\text{finite}}^* \in \Lambda_{\text{grid}}`) out-of-fold `TRAIN` and `DL-Val` panels for `REFIT_CV`, `REFIT_BAG`, and `REFIT_SHUF`.
  - **Full $\lambda$ Path Table**: Out-of-fold `205`-family `TRAIN` gain, `164`-family inner-CV gain, `91`-context `DL-Val` gain, Elo vs `r882 = 1500.0`, `-2` invalid pick counts, and head L2 drift $\|W^*(\lambda) - W_{r882}\|_2$ for every fixed $\lambda \in \{10^4, 10^3, 10^2, 10^1, 1.0, 10^{-1}, \infty\}$.
  - **Pseudoreplication Check (`H160` Lesson)**: Explicitly report for every arm (`REFIT_CV`, `REFIT_BAG`, `REFIT_SHUF`) under strict, unconstrained, and finite-best selection how many outer folds/seeds select identical $\lambda$, produce identical head weights $W$, or yield identical context argmin choices across the 3 seeds.
- **Hard Rules & Resource Caps (`xbabe2` CPU-only)**:
  - Pre-`TRAIN` `91/91` `r882` baseline gate (`mean = 1.1377941898164525`, `max_context_error == 0.0`, plus `torch.equal` raw-cost bit-identity vs native `r882`) before any `TRAIN` pool is opened; `launch.json` `source_commit` verified against `git rev-parse HEAD`.
  - `nice -n 19`, `<= 4` aggregate supervisor + worker threads, `RSS < 6 GB`, `CUDA_VISIBLE_DEVICES=""`, abort if `load1 > 85`, `MemAvailable < 16 GiB`, `disk free < 40 GB`, or `disk used >= 95%`.
