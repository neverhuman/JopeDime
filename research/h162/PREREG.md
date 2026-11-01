# H162 Preregistration: Cosine Warm-Restart LR Schedule with Snapshot Selection vs LR-Matched Constant Control

- **Wave**: `H162`
- **Agent**: `gpu_tokbench_agy`
- **Authority**: `/home/ubuntu/nudges/hwave_t0000_h162_gpu_tokbench_agy.md`
- **Claim**: `/home/ubuntu/jope_plan/ns/receipts/H162_claim.txt`
- **Seeds**: `162101`, `162102`, `162103` (`fold_seed = 162140`, `boot_seed = 162141`)
- **Host / Budget**: `xbabe2` CPU (`nice 19`, `<= 4` threads, `RSS < 6 GB`, `load <= 85`, `MemAvailable >= 16 GiB`, `free disk >= 40 GB`, `used disk < 95%`, `0` GPU usage)

---

## 1. Scientific Hypothesis

Continuation fine-tuning from `r882` with a constant learning rate (`1e-4`) has failed to produce an admitted improvement across recent waves, while `H151` demonstrated that effective step-size differences can confound optimizer comparisons. Cosine warm restarts (`SGDR`) alternate exploration (`peak_lr = 2e-4`) with annealing (`min_lr = 1e-6`), producing distinct cycle-end candidate checkpoints at each restart minimum. We test whether cycle-end snapshot selection under `2`-cycle (`COS_R2`, `2 × 20` steps) or `4`-cycle (`COS_R4`, `4 × 10` steps) cosine warm restarts improves out-of-fold equal-family utility over both frozen `r882` and a mean-LR-matched constant control (`CTRL_LRMATCH`, `lr = 1.005e-4`).

---

## 2. Experimental Design & Arms (`2,400` Total AdamW Updates Across `15` Cells)

Each `(outer, seed)` cell (`outer in 0..4`, `seed in {162101, 162102, 162103}`) partitions the `205` `H145` `DL-Train` families (`1,219` contexts, `fold_seed = 162140`) into `3` fit folds (`123` families), `1` inner selector fold $B_k$ (`41` families, `select = (outer + 1) % 5`), and `1` outer assessment fold $A_k$ (`41` families).

All `4` arms start from an owned copy of `r882` (`/home/ubuntu/jope-data/h162/inputs/round-882.pt`), train **only** the `29,211`-parameter `adapter_10tok` (`weight_decay = 1e-4`, grad norm clip `1.0`), share the exact same `40`-step minibatch stream (`batch = 8` contexts × `24` candidates per step), and keep the frozen backbone (`493ba7dc...`) and `DOPE` (`c51a56d0...`) bit-identical:

| Arm | LR Schedule (`40` AdamW steps, `wd = 1e-4`) | Mean LR | Snapshot Grid (`+ step 0`) |
| --- | --- | ---: | --- |
| `CTRL` | Constant `lr = 1.000e-4` | `1.000e-4` | `{0, 10, 20, 30, 40}` |
| `COS_R2` | Cosine warm restart: `2` cycles of `20` steps (`peak = 2e-4`, `min = 1e-6`) | `1.005e-4` | `{0, 20, 40}` (cycle ends) |
| `COS_R4` | Cosine warm restart: `4` cycles of `10` steps (`peak = 2e-4`, `min = 1e-6`) | `1.005e-4` | `{0, 10, 20, 30, 40}` (cycle ends) |
| `CTRL_LRMATCH` | Constant `lr = 1.005e-4` (exact step-mean LR of `COS_R2` and `COS_R4`) | `1.005e-4` | `{0, 10, 20, 30, 40}` |

Within cycle length $T_c \in \{20, 10\}$, step $j \in \{0, \dots, T_c - 1\}$ uses:
$$\eta(j) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi j}{T_c - 1}\right)\right)$$
so step $j = 0$ is at $\eta_{\max} = 2\times 10^{-4}$, step $j = T_c - 1$ (cycle end) is at $\eta_{\min} = 1\times 10^{-6}$, and the exact arithmetic mean over `40` steps for both `COS_R2` and `COS_R4` is $\frac{1}{2}(\eta_{\max} + \eta_{\min}) = 1.005\times 10^{-4}$.

---

## 3. Inner-Selector Rule & Decision Gate

- **Inner Selector (`select` fold $B_k$ only, locked to `A{outer}-{seed}-selection.json` before evaluating $A_k$ or `DL-Val`)**:
  - Evaluates each arm's cycle-end snapshots (`+ step 0`) on the `41` selector families (`1,000`-draw family bootstrap vs `step 0`).
  - **Strict margin rule**: A snapshot step $s > 0$ is eligible iff equal-family gain $\hat{\Delta}_s > +0.001 + 10^{-12}$ AND `95%` family bootstrap lower confidence bound $\text{LCB}_{95}(s) > 0$. Among eligible steps, the step with largest $\hat{\Delta}_s$ (ties within $10^{-12}$ rejected) is selected; otherwise `step 0` (`r882`) is selected.
- **Primary Readout (`1,219`-context / `205`-family `H145` train-only out-of-fold assessment)**:
  - Median-seed equal-family out-of-fold utility gain vs `r882` + `10,000`-draw family bootstrap `95%` CI, Holm-Bonferroni adjusted one-sided $p$-values ($m = 4$), Elo vs `r882 = 1500.0`, `-2` invalid/gate-cliff counts per arm, and L2 adapter parameter drift (`step 40` and selected step).
  - Direct contrast of `COS_R2` and `COS_R4` vs `CTRL_LRMATCH` (`10,000`-draw family bootstrap).
  - *Note*: `r882` has historical in-sample `DL-Train` exposure (`historical_r882_train_exposure = true`).
- **Secondary Descriptive Readout (`91`-context / `19`-family `DL-Val` cohort)**:
  - Exact `91/91` `r882` baseline gate (`1.1377941898164525`, `max_context_error <= 1e-6`) verified prior to any training step.
  - 5-fold cross-fit policy average on `DL-Val` (`equal-family` $\Delta$, `95%` CI, Elo vs `r882 = 1500.0`, `-2` invalid counts).
- **Advancement Verdict**:
  - `ADVANCE` iff a cosine warm-restart arm (`COS_R2` or `COS_R4`) achieves median-seed equal-family out-of-fold gain $> +0.001$ AND `95%` bootstrap $\text{LCB} > 0$ vs `r882` AND strictly beats `CTRL_LRMATCH` ($\hat{\Delta}_{\text{vs CTRL\_LRMATCH}} > 10^{-12}$); ties or sub-threshold gains return `NO_ADVANCE`.
