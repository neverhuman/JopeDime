# H166 Preregistration: C-Mixup (Label-Similarity Mixup) Regularized `r882` Adapter Continuation

- **Wave**: `H166`
- **Owner**: `gpu_tokbench_agy`
- **Authority**: `/home/ubuntu/nudges/hwave_t0125_h166_tokbench_agy.md`
- **Claim receipt**: `/home/ubuntu/jope_plan/ns/receipts/H166_claim.txt`
- **Status**: `PREREGISTERED` (pre-execution; zero wave optimizer updates executed prior to preregistration merge)
- **Seeds**: `SEEDS = (166101, 166102, 166103)`, `FOLD_SEED = 166140`, `BOOT_SEED = 166141`

---

## 1. Scientific Rationale & Hypothesis

Prior `r882` continuation waves (weight decay `H144`/`H149`, label smoothing `H137`/`H155`, EMA/SWA `H154`, lookahead `H146`/`H151`, trust region `H142`/`H157`, curriculum `H128`, cosine restarts `H162`, robust loss `H161`, KD `H158`) all exhibited rapid continuation overfitting on the 3-fold fit split (`~731` contexts), causing inner-fold strict selection (`gain > +0.001` and `LCB95 > 0`) to fall back to `step = 0` (`r882`).

**Untested mechanism**: **C-Mixup** (Yao et al., NeurIPS 2022, *C-Mixup: Improving Generalization in Regression*). Standard empirical risk minimization over discrete training contexts allows the 29,211-parameter `adapter_10tok` to memorize sharp local utility steps around individual fit-fold contexts. C-Mixup regularizes continuation by interpolating pairs of training contexts $(i, j)$ within each fit-fold minibatch using mixing weights $\lambda_i \sim \operatorname{Beta}(\alpha, \alpha)$ and sampling partner $j \neq i$ with probability proportional to a Gaussian kernel on target utility distribution similarity.

### Slate-Level Same-Shape C-Mixup Specification (Preregistered Before Scoring)

Because `rt.loss(pred, fitness, gate)` is a **per-slate ranking and cost-regression objective** over candidate slates rather than an independent pointwise regression loss, mixing is defined **at the slate level only between same-shape `(24, ...)` slates** within each fit-fold minibatch ($B = 8$, $K = 24$):

1. **Rank-aligned same-shape 24-candidate slates**:
   - Every training minibatch in `batch_plan` samples $B = 8$ fit-fold contexts, each subsampled to $K = 24$ candidates: positions `0..3` are the top-4 revealed candidates in ascending cost (`fitness`) order, and positions `4..23` are 20 unrevealed candidates sampled without replacement from `orders[cid][4:]` and **sorted in ascending cost (`fitness`) rank order**.
   - Consequently, every slate in the minibatch has identical tensor shapes (`progs` `(8, 24, 412)`, `sos` `(8, 24, 2)`, `tok` `(8, 10, 32)`, `mask` `(8, 10)`, `rev_raw_f` `(8, 4)`, `rev_gate` `(8, 4)`, `a_tot` `(8, 24)`, `a_eff` `(8, 24)`, `fitness` `(8, 24)`, `gate` `(8, 24)`) and aligned within-slate candidate rank positions `0..3` and `4..23`.
2. **Fit-fold target descriptor & median pairwise distance bandwidth**:
   - For each fit-fold context $c \in F_k$ (strictly excluding selector fold $S_k$, assessment fold $A_k$, and `DL-Val`), resolve the effective H59/H140 candidate cost target $t_c = \operatorname{where}(\text{gate}_c, \operatorname{clip}(\text{fitness}_c, -4.0, 2.0), 2.0)$ and corresponding effective utility $u_c = -t_c$.
   - Define the 4D target distribution summary $r_c = (\operatorname{mean}(u_c), \operatorname{std}(u_c), \max(u_c), \operatorname{median}(u_c)) \in \mathbb{R}^4$, standardized across fit-fold contexts $c \in F_k$:
     $$y_c = \frac{r_c - \mu_{F_k}}{\max(\sigma_{F_k}, 10^{-6})}$$
   - Set the Gaussian kernel bandwidth $\sigma_k$ to the **fit-fold median pairwise Euclidean distance**:
     $$\sigma_k = \max\Bigl(\operatorname{median}_{c < c' \in F_k} \|y_c - y_{c'}\|_2,\; 10^{-6}\Bigr)$$
3. **Minibatch partner sampling & gate-resolved convex target mixing**:
   - Within a minibatch $(c_0, \dots, c_{B-1})$ of size $B = 8$:
     - In `MIX02` ($\alpha = 0.2$) and `MIX04` ($\alpha = 0.4$), partner $j \neq i$ is sampled with probability:
       $$P_{\text{gauss}}(j \mid i) = \frac{\exp(-\|y_{c_i} - y_{c_j}\|_2^2 / (2\sigma_k^2))}{\sum_{j' \neq i} \exp(-\|y_{c_i} - y_{c_{j'}}\|_2^2 / (2\sigma_k^2))}, \quad j \neq i$$
     - In `RANDMIX02` ($\alpha = 0.2$, mechanism control), partner $j \neq i$ is sampled uniformly across the minibatch ($P_{\text{unif}}(j \mid i) = \frac{1}{B-1}$ for $j \neq i$).
     - Both `MIX02` and `RANDMIX02` use the identical deterministic RNG stream `stream_seed("cmixup", seed, outer, step)` to draw $u_{\text{partner}, i} \sim \operatorname{Uniform}(0, 1)$ and $\lambda_i \sim \operatorname{Beta}(\alpha, \alpha)$, so at $\alpha = 0.2$ `MIX02` and `RANDMIX02` share the **exact same $\lambda_i$ realizations** at every step and isolate the effect of the C-Mixup target-similarity kernel.
   - Continuous adapter inputs (`progs`, `sos`, `tok`, gate-resolved `rev_raw_f`, `rev_gate`, `a_tot`, `a_eff`) and gate-resolved clipped slate regression targets $t_i = \operatorname{where}(\text{gate}_i, \operatorname{clip}(\text{fitness}_i, -4.0, 2.0), 2.0)$ are convexly combined with $\lambda_i \in [0, 1]$:
     $$\tilde{x}_i = \lambda_i x_i + (1 - \lambda_i) x_j, \qquad \tilde{t}_i = \lambda_i t_i + (1 - \lambda_i) t_j \in [-4.0, 2.0]$$
     with token attention masks combined via $\tilde{\text{mask}}_i = \text{mask}_i \lor \text{mask}_j$ and $\tilde{\text{gate}}_i = \mathbf{1}$ passed to `rt.loss(pred, fit_in, gate_in)` so that `rt.loss` evaluates directly against the exact convex effective target $\tilde{t}_i$ without nonlinear post-mix gate switching.
4. **Train-Mode Only Mixing, Pre-TRAIN `91/91` Baseline Gate & Bit-Exact `step = 0` Parity**:
   - The `91/91` `DL-Val` `r882` baseline gate (`mean = 1.1377941898164525`, `max_context_error == 0.0`) AND `torch.equal` raw `total_cost` bit-identity against native `r882` (`parent.predict` / `net.adapter_10tok.predict_costs`) execute **before** opening any `1,219` `TRAIN` context pools.
   - Mixing is applied **strictly during training forward/backward updates** in `MIX02`, `MIX04`, and `RANDMIX02`.
   - All selector-fold (`select`), outer-assessment-fold (`assess`), and `DL-Val` (`val`) evaluations execute completely unmixed via `evaluate(rows)`. At `step = 0`, evaluation is bit-exact `r882` (`torch.equal` raw cost identity and `max_abs_diff == 0.0`).

---

## 2. Experimental Design & Arms

- **Cohort**: `H145` real S3 `TRAIN` (`1,219` contexts across `205` families, partitioned into `5` disjoint folds of `41` families each under `FOLD_SEED = 166140`: `3` fit folds = `123` families, `1` inner selector fold = `41` families, `1` outer assessment fold = `41` families). Secondary descriptive cohort: `91`-context / `19`-family `DL-Val`.
- **Arms (`4` continuation arms + `r882` anchor)**:
  1. `CTRL`: Unmixed live AdamW (`lr = 1e-4`, `weight_decay = 1e-4`, `40` updates, snapshots at steps `(0, 10, 20, 30, 40)`).
  2. `MIX02`: C-Mixup with Gaussian target-similarity partner kernel and $\alpha = 0.2$ ($\lambda_i \sim \operatorname{Beta}(0.2, 0.2)$), live AdamW (`lr = 1e-4`, `weight_decay = 1e-4`, `40` updates, snapshots at `(0, 10, 20, 30, 40)`).
  3. `MIX04`: C-Mixup with Gaussian target-similarity partner kernel and $\alpha = 0.4$ ($\lambda_i \sim \operatorname{Beta}(0.4, 0.4)$), live AdamW (`lr = 1e-4`, `weight_decay = 1e-4`, `40` updates, snapshots at `(0, 10, 20, 30, 40)`).
  4. `RANDMIX02`: Uniform-partner mixup mechanism control with $\alpha = 0.2$ ($\lambda_i \sim \operatorname{Beta}(0.2, 0.2)$ matched to `MIX02`), live AdamW (`lr = 1e-4`, `weight_decay = 1e-4`, `40` updates, snapshots at `(0, 10, 20, 30, 40)`).
- **Cells & Updates**: `5` outer folds × `3` seeds (`166101, 166102, 166103`) = `15` cells; `4` arms × `40` updates = `160` updates/cell (`2,400` total optimizer updates).
- **`None`-Grad Guard (Addendum 07:26Z)**: `sanitize_and_clip_grads` checks `p.grad is None` for every adapter parameter before norm calculation and clipping, safely skipping `None`-grad tensors while recording their names per cell (`none_grad_tensors`) and failing closed on any non-finite gradient.

---

## 3. Selection Gate, Metrics & Decision Rule

- **Inner-Fold Strict Selection (`selected_step`)**:
  - For each `(outer, seed, arm)`, evaluate snapshots `s in (0, 10, 20, 30, 40)` on the inner selector fold (`41` families) before touching the outer assessment fold.
  - A step `s > 0` is eligible iff its equal-family mean utility gain vs `step = 0` (`r882`) exceeds `+0.001 + 1e-12` (ties within `1e-12` rejected) AND its 1,000-draw family-bootstrap 95% lower confidence bound is strictly positive (`ci_low_95 > 0.0`). Otherwise `selected_step = 0` (`r882`).
- **Descriptive Unconstrained Best-Step & L2 Drift Diagnostics**:
  - In the same pass, `run.py` also records and evaluates:
    1. `unconstrained_selected_step`: step in `(0, 10, 20, 30, 40)` maximizing inner selector-fold equal-family gain (`delta > 1e-12`, else `0`).
    2. `best_nonzero_step`: step in `(10, 20, 30, 40)` maximizing inner selector-fold equal-family gain among positive-update steps.
    3. Adapter L2 parameter drift $\|w_s - w_0\|_2$ at `step = 40`, `selected_step`, `unconstrained_selected_step`, and `best_nonzero_step`.
- **Primary Endpoint & `ADVANCE` Gate**:
  - Equal-family out-of-fold utility gain vs `r882` across all `205` families (`1,219` contexts) of `H145` `TRAIN`: point estimate is the **median over the `3` seeds of per-seed equal-family mean gains**, with 10,000-draw shared-family bootstrap 95% CI of that median-over-seeds statistic under `BOOT_SEED = 166141` (`H148` protocol, immune to crossing-seed sign inversions), Holm-Bonferroni adjusted one-sided $p$-values across treatment arms `MIX02`, `MIX04`, `RANDMIX02`, and canonical `H145`/`H148` 400-iteration MM Bradley-Terry Elo anchored at `r882 = 1500.0` (`tie_margin = 1e-4`, `prior = 0.01`, strength clipping `[1e-6, 1e6]`, 1,000-draw shared-family bootstrap 95% CI).
  - **`ADVANCE` iff** at least one treatment arm (`MIX02`, `MIX04`, `RANDMIX02`) achieves median equal-family out-of-fold gain `> +0.001` (`> 0.001 + 1e-12`), 10,000-draw shared-family bootstrap 95% `ci_low_95 > 0.0`, AND treatment-family Holm-adjusted one-sided $p_{\text{Holm}} < 0.05$. Otherwise `NO_ADVANCE`.

---

## 4. Hard Rules & Resource Caps

1. Exact `91/91` `r882` baseline gate (`mean = 1.1377941898164525`, `max_context_error == 0.0`, plus `torch.equal` raw-cost bit-identity vs native `r882`) must pass before any `TRAIN` pool is opened or any continuation step runs; `launch.json` `source_commit` is verified against actual checkout `HEAD`.
2. Frozen DOPE composite SHA-256 (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and non-adapter parameters (`8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`) verified before and after every evaluation and optimizer step.
3. Only the `29,211`-parameter `adapter_10tok` (`47` tensors) trains, on an owned read-only copy of `r882` under `/home/ubuntu/jope-data/h166/inputs/round-882.pt`.
4. CPU-only on `xbabe2`: `nice -n 19`, `<= 4` aggregate threads and `RSS < 6 GB` across the full supervisor + worker process tree (`OMP/MKL/OPENBLAS/NUMEXPR/RAYON/TORCH_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""`), abort if `load1 > 85`, `MemAvailable < 16 GiB`, `disk free < 40 GB`, or `disk used >= 95%`.
