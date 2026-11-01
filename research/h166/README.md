# H166: C-Mixup (Label-Similarity Mixup) Regularized `r882` Adapter Continuation

- **Wave**: `H166`
- **Owner**: `gpu_tokbench_agy`
- **Status**: `COMPLETE` (`RULE_NEUTRAL` / `NO_ADVANCE`; `0/15` cells × `4/4` arms admitted `step > 0` under strict inner-fold `gain > +0.001` & `LCB95 > 0` gate; `r882` retained)
- **Preregistration PR**: [#1423](https://github.com/neverhuman/JopeDime/pull/1423) (reviewed by `@gpu_fused_redteam_codex` in `5727060095` at `9b8a2a0bca5ef7a8979e7b6b3150f224bac0a2a7`, merged at `86e7dbef1b840f19972555175ffb823a4beecdfb`)
- **Seeds**: `SEEDS = (166101, 166102, 166103)`, `FOLD_SEED = 166140`, `BOOT_SEED = 166141`
- **S3 Custody**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H166/gpu_tokbench_agy-h166-20261101/` (`334` run objects + `6` custody objects verified)

---

## 1. Hypothesis & Mechanism

Standard empirical risk minimization over discrete training contexts allows the `29,211`-parameter `adapter_10tok` (`47` tensors) to overfit local candidate utility steps on the 3-fold fit split (`718–749` contexts), causing inner-fold strict selection to fall back to `step = 0` (`r882`).

`H166` tests **C-Mixup** (Yao et al., NeurIPS 2022, *C-Mixup: Improving Generalization in Regression*) adapted at the **slate level between rank-aligned same-shape 24-candidate slates** within each training minibatch ($B = 8, K = 24$):
1. **Rank-aligned 24-candidate slates**: Positions `0..3` are the top-4 revealed candidates in ascending cost (`fitness`) order, and positions `4..23` are 20 unrevealed candidates sampled without replacement from `orders[cid][4:]` and sorted in ascending cost (`fitness`) rank order.
2. **Fit-fold target descriptor & median pairwise distance bandwidth**: For each fit-fold context $c \in F_k$ (strictly excluding selector fold $S_k$, assessment fold $A_k$, and `DL-Val`), effective candidate cost targets are resolved via $t_c = \operatorname{where}(\text{gate}_c, \operatorname{clip}(\text{fitness}_c, -4.0, 2.0), 2.0)$ ($u_c = -t_c$), summarized by $r_c = (\operatorname{mean}(u_c), \operatorname{std}(u_c), \max(u_c), \operatorname{median}(u_c)) \in \mathbb{R}^4$, standardized on $F_k$ to $y_c$, with Gaussian kernel bandwidth $\sigma_k = \operatorname{median}_{c < c' \in F_k} \|y_c - y_{c'}\|_2$. Across the 5 outer folds (`A0..A4`), fit-fold bandwidths were $\sigma_k \in \{2.094351, 1.924030, 1.732199, 1.791642, 1.986622\}$ (`median = 1.924030`).
3. **Gate-resolved convex target interpolation**: Continuous adapter inputs (`progs`, `sos`, `tok`, gate-resolved `rev_raw_f`, `rev_gate`, `a_tot`, `a_eff`) and gate-resolved clipped regression targets $t_i$ are interpolated with $\lambda_i \sim \operatorname{Beta}(\alpha, \alpha)$ ($\tilde{t}_i = \lambda_i t_i + (1 - \lambda_i) t_j \in [-4.0, 2.0]$, `gate_in = True`), while evaluation (`select`, `assess`, `DL-Val`) is strictly unmixed.

---

## 2. Experimental Setup

- **Cohort**: `H145` real S3 `TRAIN` (`1,219` contexts across `205` families, partitioned into `5` disjoint folds of `41` families each under `FOLD_SEED = 166140`: `123` fit / `41` selector / `41` assessment families per outer fold) and `91`-context / `19`-family `DL-Val`.
- **Arms (`4` continuation arms × `40` AdamW updates at `lr = 1e-4, weight_decay = 1e-4`, snapshots at `(0, 10, 20, 30, 40)`)**:
  - `CTRL`: Unmixed live AdamW continuation.
  - `MIX02`: C-Mixup with Gaussian target-similarity partner kernel and $\alpha = 0.2$ ($\lambda \sim \operatorname{Beta}(0.2, 0.2)$).
  - `MIX04`: C-Mixup with Gaussian target-similarity partner kernel and $\alpha = 0.4$ ($\lambda \sim \operatorname{Beta}(0.4, 0.4)$).
  - `RANDMIX02`: Uniform-partner mechanism control ($\alpha = 0.2$, sharing identical deterministic $\lambda \sim \operatorname{Beta}(0.2, 0.2)$ draws with `MIX02` and sampling partner $j \neq i$ uniformly within the minibatch).
- **`None`-Grad Guard (Addendum `07:26Z`)**: Across all `15` cells and `4` arms, `sanitize_and_clip_grads` detected and cleanly skipped `6` unused `parent_delta_net` tensors (`adapter_10tok.parent_delta_net.{0,1,3}.{weight,bias}`) that receive `grad = None`, while clipping and updating the `41` active adapter tensors across all `2,400` optimizer updates.
- **Resource & Audit Envelope (`xbabe2` CPU-only)**:
  - Pre-`TRAIN` `91/91` `r882` baseline gate: `PASS` (`mean = 1.1377941898164525`, `max_context_error = 0.0`, `raw_cost_bit_exact_91_91 = True` via `torch.equal` against native `r882`).
  - `2,400` optimizer updates across `15` cells in `2,760.13 s` (`~46.0 min`), `nice -n 19`, peak aggregate supervisor + worker threads `3` (`<= 4`), peak aggregate RSS `3.35 GB` (`< 6 GB`), `gpu_lock_held = false`, `cuda_initialized = false`, `protected_reads = 0`, `refused_reads = 0`, frozen DOPE SHA-256 (`c51a56d0...`) and non-adapter SHA-256 (`8d1070fa...`) unchanged after every step and evaluation.

---

## 3. Outcome & Key Metrics (Median + 95% CI over 3 Seeds)

### 3.1 Primary Endpoint: Strict Inner-Fold Selector (`gain > +0.001` & `LCB95 > 0`)

Under the preregistered strict inner-fold selector (`selected_step`), `0/15` cells in `CTRL`, `MIX02`, `MIX04`, and `RANDMIX02` cleared both `gain > +0.001` and 1,000-draw family-bootstrap `ci_low_95 > 0.0` on the 41-family selector fold `B`. Every cell selected `step = 0` (`r882`):

| Cohort | Arm | Strict Admitted Cells | Equal-Family Mean | Median Gain vs `r882` [95% CI] | Per-Seed Gains (`166101, 166102, 166103`) | $p_{\text{Holm}}$ | Canonical MM Elo [95% CI] | `-2` Picks (Median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `H145` `TRAIN` (`1,219` ctx / `205` fam OOF) | `r882` | — | `1.385355` | `0.000000` | — | — | `1500.00` | `17.0` |
| `H145` `TRAIN` (`1,219` ctx / `205` fam OOF) | `CTRL` | `0/15` | `1.385355` | `+0.000000 [+0.000000, +0.000000]` | `[0.0, 0.0, 0.0]` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `17.0` |
| `H145` `TRAIN` (`1,219` ctx / `205` fam OOF) | `MIX02` | `0/15` | `1.385355` | `+0.000000 [+0.000000, +0.000000]` | `[0.0, 0.0, 0.0]` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `17.0` |
| `H145` `TRAIN` (`1,219` ctx / `205` fam OOF) | `MIX04` | `0/15` | `1.385355` | `+0.000000 [+0.000000, +0.000000]` | `[0.0, 0.0, 0.0]` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `17.0` |
| `H145` `TRAIN` (`1,219` ctx / `205` fam OOF) | `RANDMIX02` | `0/15` | `1.385355` | `+0.000000 [+0.000000, +0.000000]` | `[0.0, 0.0, 0.0]` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `17.0` |
| `DL-Val` (`91` ctx / `19` fam) | `r882` / All Arms | `0/15` | `1.153615` | `+0.000000 [+0.000000, +0.000000]` | `[0.0, 0.0, 0.0]` | `1.0000` | `1500.00 [1500.00, 1500.00]` | `0.0` |

### 3.2 Mechanism Verification: Partner Similarity, L2 Drift & Inner-Fold (`B`) Regularization

C-Mixup's partner kernel and convex target interpolation operated as designed during training:
- **Exact $\lambda$ coupling & kernel distance reduction**: `MIX02` and `RANDMIX02` shared identical `mean_lambda` (`median = 0.509587`, `mean = 0.510465`), while the C-Mixup Gaussian kernel (`MIX02`, `MIX04`) reduced standardized partner target distance by **`33.1%`** (`median = 1.561284`, `mean = 1.574153` vs `median = 2.332362`, `mean = 2.328230` for `RANDMIX02`).
- **Adapter L2 drift damping**: Median step-40 adapter L2 drift $\|w_{40} - w_0\|_2$ decreased from `0.3720` (`CTRL`) to `0.3440` (`MIX02`, `0.925x`), `0.3417` (`MIX04`, `0.918x`), and `0.3320` (`RANDMIX02`, `0.893x`).
- **Inner selector fold (`B`, 41 families) trajectory**: Each arm achieved positive unconstrained best-step gain (`delta > 0`) on `9/15` inner selector cells, with median inner-fold best-step gain rising from `+0.001621` (`CTRL`) to `+0.003526` (`MIX02`), `+0.003996` (`MIX04`), and `+0.005047` (`RANDMIX02`), and median step-40 inner-fold degradation softened from `-0.047797` (`CTRL`) to `-0.034902` (`MIX02`), `-0.028658` (`MIX04`), and `-0.033196` (`RANDMIX02`).

### 3.3 Unconstrained Best-Step (`delta > 0` on `B`) & Best Non-Zero Step (`s in {10, 20, 30, 40}`) Out-of-Fold Panels

When the inner-fold unconstrained best step (`unconstrained_selected_step`) or best positive-update step (`best_nonzero_step`) was evaluated out-of-fold on the `205`-family `H145` `TRAIN` assessment folds and `19`-family `DL-Val`, none of the continuation arms improved over `r882`:

| Panel / Selector | Arm | Median Gain vs `r882` [95% CI] | Per-Seed Gains (`166101, 166102, 166103`) | Median Gain vs `CTRL` [95% CI] | Median Gain vs `RANDMIX02` [95% CI] | Canonical MM Elo [95% CI] | Selected Step L2 Drift (Med) | `-2` Picks (Med) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Unconstrained** `H145` `TRAIN` (`205` fam) | `CTRL` | `-0.006598 [-0.019996, +0.006103]` | `[+0.002716, -0.006598, -0.009185]` | `+0.000000 [+0.000000, +0.000000]` | `+0.002701 [-0.001386, +0.007579]` | `1499.57 [1493.72, 1504.43]` | `0.0973` | `19.0` |
| **Unconstrained** `H145` `TRAIN` (`205` fam) | `MIX02` | `-0.008319 [-0.015275, -0.000868]` | `[-0.000641, -0.008319, -0.010751]` | `-0.001721 [-0.015073, +0.008881]` | `+0.001136 [-0.014063, +0.006039]` | `1496.74 [1492.26, 1501.45]` | `0.0950` | `19.0` |
| **Unconstrained** `H145` `TRAIN` (`205` fam) | `MIX04` | `-0.005470 [-0.014562, +0.000520]` | `[-0.005470, -0.004187, -0.011151]` | `-0.001966 [-0.013435, +0.010013]` | `+0.000735 [-0.013528, +0.010245]` | `1497.09 [1493.40, 1501.14]` | `0.0935` | `19.0` |
| **Unconstrained** `H145` `TRAIN` (`205` fam) | `RANDMIX02` | `-0.006679 [-0.016540, +0.005043]` | `[-0.006679, +0.001664, -0.011886]` | `-0.002701 [-0.007700, +0.001399]` | `+0.000000 [+0.000000, +0.000000]` | `1497.79 [1491.75, 1501.50]` | `0.0905` | `19.0` |
| **Unconstrained** `DL-Val` (`19` fam) | `CTRL` | `-0.002525 [-0.006572, +0.000634]` | `[-0.002471, -0.002525, -0.005574]` | `+0.000000 [+0.000000, +0.000000]` | `+0.000239 [-0.000688, +0.000811]` | `1494.15 [1464.85, 1517.61]` | `0.0973` | `0.0` |
| **Unconstrained** `DL-Val` (`19` fam) | `MIX02` | `-0.002350 [-0.005683, +0.000559]` | `[-0.002350, -0.000895, -0.005613]` | `+0.000121 [-0.000219, +0.003052]` | `+0.000360 [-0.000266, +0.001276]` | `1494.88 [1475.04, 1516.83]` | `0.0950` | `0.0` |
| **Unconstrained** `DL-Val` (`19` fam) | `MIX04` | `-0.003846 [-0.010291, +0.000444]` | `[-0.006373, -0.001055, -0.003846]` | `+0.001470 [-0.000401, +0.003623]` | `-0.000032 [-0.001373, +0.001020]` | `1484.64 [1466.14, 1501.46]` | `0.0935` | `0.0` |
| **Unconstrained** `DL-Val` (`19` fam) | `RANDMIX02` | `-0.002710 [-0.006760, +0.000526]` | `[-0.002710, -0.001023, -0.006181]` | `-0.000239 [-0.000808, +0.000704]` | `+0.000000 [+0.000000, +0.000000]` | `1495.61 [1469.23, 1504.39]` | `0.0905` | `0.0` |
| **Best Non-Zero** `H145` `TRAIN` (`205` fam) | `CTRL` | `-0.003658 [-0.018257, +0.009250]` | `[-0.003504, -0.003658, -0.007009]` | `+0.000000 [+0.000000, +0.000000]` | `+0.001217 [-0.000986, +0.005475]` | `1498.50 [1492.02, 1504.13]` | `0.1001` | `18.0` |
| **Best Non-Zero** `H145` `TRAIN` (`205` fam) | `MIX02` | `-0.006794 [-0.021252, +0.006414]` | `[-0.006401, -0.006794, -0.009820]` | `-0.002897 [-0.008791, +0.001059]` | `-0.001680 [-0.007287, +0.003412]` | `1495.33 [1487.69, 1501.62]` | `0.0966` | `19.0` |
| **Best Non-Zero** `H145` `TRAIN` (`205` fam) | `MIX04` | `-0.006179 [-0.022065, +0.005589]` | `[-0.006179, -0.004764, -0.011555]` | `-0.002676 [-0.009904, +0.001814]` | `-0.001458 [-0.006500, +0.002732]` | `1496.14 [1489.18, 1501.81]` | `0.0967` | `19.0` |
| **Best Non-Zero** `H145` `TRAIN` (`205` fam) | `RANDMIX02` | `-0.004721 [-0.015461, +0.006632]` | `[-0.004721, +0.001315, -0.010996]` | `-0.001217 [-0.005275, +0.000978]` | `+0.000000 [+0.000000, +0.000000]` | `1496.97 [1490.51, 1502.54]` | `0.0943` | `18.0` |
| **Best Non-Zero** `DL-Val` (`19` fam) | `CTRL` | `-0.006284 [-0.014950, +0.000779]` | `[-0.006284, -0.003873, -0.013215]` | `+0.000000 [+0.000000, +0.000000]` | `-0.001055 [-0.003691, +0.000825]` | `1492.69 [1458.24, 1527.10]` | `0.1001` | `0.0` |
| **Best Non-Zero** `DL-Val` (`19` fam) | `MIX02` | `-0.007211 [-0.016370, +0.000761]` | `[-0.007483, -0.003150, -0.007211]` | `+0.000723 [-0.001970, +0.003277]` | `-0.000332 [-0.002536, +0.000471]` | `1490.49 [1458.89, 1524.89]` | `0.0966` | `0.0` |
| **Best Non-Zero** `DL-Val` (`19` fam) | `MIX04` | `-0.007328 [-0.016225, +0.000579]` | `[-0.007328, -0.001995, -0.011940]` | `+0.001275 [-0.000255, +0.003249]` | `+0.000824 [-0.000167, +0.001950]` | `1496.34 [1469.00, 1527.81]` | `0.0967` | `0.0` |
| **Best Non-Zero** `DL-Val` (`19` fam) | `RANDMIX02` | `-0.003751 [-0.009420, +0.000731]` | `[-0.003751, -0.002818, -0.013476]` | `+0.001055 [-0.000824, +0.003677]` | `+0.000000 [+0.000000, +0.000000]` | `1497.81 [1481.65, 1516.17]` | `0.0943` | `0.0` |

---

## 4. Verdict & Next Step

- **Verdict**: `COMPLETE — RULE_NEUTRAL (NO_ADVANCE)`. No serving promotion; `r882` remains the undefeated serving baseline.
- **Why C-Mixup did not advance**: While C-Mixup (`MIX02`, `MIX04`) and uniform partner mixup (`RANDMIX02`) reduced step-40 adapter L2 drift by `8–11%` and softened step-40 inner-fold degradation (`-0.0287` to `-0.0349` vs `-0.0478` `CTRL`), convexly interpolating projected program features (`progs`) and context tokens (`tok`) across distinct dataset contexts creates synthetic intermediate slate representations that do not correspond to realizable symbolic programs on unseen dataset families. Consequently, positive inner-fold unconstrained steps (`9/15` cells) failed to generalize to out-of-fold assessment families (`-0.0055` to `-0.0083` vs `r882`).
- **Next Step**: Close the cross-context input/target interpolation direction (`C-Mixup` / `Mixup`) on `adapter_10tok`; prioritize within-context candidate-ranking invariance or non-continuation structural bottlenecks where candidate program identity is preserved.
