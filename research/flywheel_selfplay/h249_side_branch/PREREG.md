# H249: Zero-Init Low-Rank Residual Side Branch on a Frozen r882 Adapter (`SIDE4`, `SIDE16`) vs Exact H170 `CTRL`, xbabe2 CPU

**Owner:** `rd_argon` `[ARGON]`. **Assignment:** `~/nudges/hw_t1035_argon_h249.md` (watcher, 2026-11-03 10:35Z, decided).
**Claim:** `~/jope_plan/ns/receipts/H249_claim.txt`, taken under `Hwave-admission.lock`.

This is an offline diagnostic study only. It cannot advance a serving model or mutate any live flywheel checkpoint.
`DOPE` stays frozen at `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`: it is checked at load, after
every optimizer step and evaluation, and before and after the study.

Training runs offline on a read-only copy of `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`).
The live flywheel is never read, written, paused, or preempted.

**This preregistration merges to `main` before any outcome read.** The pre-launch `--smoke` check performs `0` optimizer
updates and scores `0` contexts.

## Hypothesis

- **What has been tried:**
  - Full-adapter or subset-adapter continuations that move `r882` weights (`CTRL` AdamW H170, Lion H232, SF-AdamW H236,
    AdaBelief/RAdam H237, low-LR H238/H240, subset freeze H239, Muon/Grokfast H247) all drift from `r882` and lose utility
    at step 40 (e.g., H247 `CONTROL` step-40 `-0.0263`).
  - Zero-update weight-space interpolations, spectral truncations, module soups, and perturbations (H241–H244) either
    preserve `r882` argmins trivially or trigger cliff regressions.
- **What is missing:** The H-wave index has **no function-preserving capacity expansion**: a zero-initialised low-rank
  residual side branch (LoRA-style, Hu et al. 2021) trained while every `r882` adapter tensor stays strictly frozen
  (`requires_grad = False`), so update `0` is bit-identical to `r882` and all learned behaviour is purely additive
  ($W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$).
- **H249 asks:** At the exact H170 `CTRL` 40-update budget, loss, batches, and data order, does training a zero-init
  low-rank side branch (`SIDE4` rank $r=4$ primary, `SIDE16` rank $r=16$ secondary) on the 16 2-D `r882` adapter matrices
  lift `DL-Val91` utility over `r882` by more than `+0.002`?

## Arms & Frozen Hyperparameters

All arms share:
- the exact inherited H146/H170 `CTRL` loss (`smooth_l1_loss + 0.1 * kl_div` at temperature `0.2`);
- the H156 minibatches (`8` contexts $\times$ `24` candidates: top-4 revealed + 20 sampled) and identical data order per `(seed, outer_fold)`;
- global gradient norm clipping at `1.0` over the arm's trainable parameters;
- `40` optimizer updates per cell (`steps = [0, 10, 20, 30, 40]`);
- starting point bit-identical to `r882` (`47` adapter tensors, `29,211` parameters, of which `16` are 2-D matrices).

The arms:
1. **`CONTROL` (full-adapter baseline, `29,211` trainable parameters across `47` tensors)**:
   - Exact H170 `CTRL` AdamW over all `47` `r882` adapter tensors: `lr = 1e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8`,
     decoupled `weight_decay = 1e-4`.
2. **`SIDE4` (primary, rank $r = 4$, $\alpha = 4.0$, $\alpha/r = 1.0$, `6,232` trainable parameters across `32` tensors)**:
   - Every `r882` adapter tensor (`47` tensors, `29,211` parameters) has `requires_grad = False` and is verified
     bit-identical to `r882` after every optimizer step.
   - For each of the `16` 2-D adapter weight matrices $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, attach
     $\Delta W = (B A) \cdot (\alpha / r)$ with $A \in \mathbb{R}^{4 \times d_{\text{in}}} \sim \text{Kaiming-uniform}(a=\sqrt{5})$
     (seeded deterministically per `(seed, outer_fold, arm)`), $B = 0 \in \mathbb{R}^{d_{\text{out}} \times 4}$, and
     $\alpha = r = 4$.
   - Only $(A, B)$ (`32` tensors, `6,232` parameters = `21.33%` of full adapter) are trained using AdamW with
     `lr = 1e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8`, and decoupled `weight_decay = 1e-2` on the branch only.
3. **`SIDE16` (secondary, rank $r = 16$, $\alpha = 16.0$, $\alpha/r = 1.0$, `24,928` trainable parameters across `32` tensors)**:
   - Identical to `SIDE4` with rank $r = 16$ and $\alpha = 16.0$ (`32` tensors, `24,928` parameters = `85.34%` of full
     adapter), trained with AdamW (`lr = 1e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8`, decoupled `weight_decay = 1e-2`).

[`schedules.json`](schedules.json) records the per-update settings for all three arms (`step = 1..40`), verified by
[`runner.py`](runner.py) before training. Before any training step executes, [`runner.py`](runner.py) asserts that
update-0 (`B = 0`) scores for `SIDE4` and `SIDE16` are **bit-identical** to `r882` on all `91` `DL-Val` (`Native91`)
contexts (`mean = 1.1377941898164525`).

## Machinery & Protocol (Merged H247 / H232 / H170)

- **Cohorts & Provenance**: `1,219` `DL-Train` contexts across `205` families and `91` `DL-Val` contexts across `19`
  families (`1,310` total regression contexts across `276` real S3 datasets pinned in [`provenance.json`](provenance.json);
  `0` synthetic, `has_meta_test` excluded, `0` protected reads).
- **Folds**: One fresh family partition seed **`249140`** (`205` families $\to 5 \times 41$), shared by seeds
  **`249101, 249102, 249103`**, pinned in [`fold_manifest.json`](fold_manifest.json).
- **Cells & Budget**: `15` cells per arm (`5` outer folds $\times 3$ seeds), `40` updates per cell $\implies$
  $15 \times 3 \times 40 = \mathbf{1,800}$ total optimizer updates.
- **Strict Inner Selector**: Snapshots at steps `0, 10, 20, 30, 40` are scored on the `SELECT` fold (`41` families). The
  earliest step maximizing equal-family `SELECT` utility is accepted iff equal-family gain vs step `0` (`r882`) is
  `> +0.001` **and** `10,000`-draw family-bootstrap `LCB95 > 0`; otherwise step `0` (`r882`) is selected.
- **Pre-Assessment Lock**: Per-cell selections are written and `fsync`-locked to `{cellname}-selection.json` before any
  trained policy is evaluated on outer `ASSESS` or `DL-Val91`. Each selected policy is scored once.

## Statistics & Decision Rule

- **Primary Panel**: `DL-Val91` (`91` contexts, `19` families), each seed's equal mixture across its `5` outer-fold
  selected policies.
- **Secondary Panel**: `1,219`-context `DL-Train` strict out-of-fold (`OOF`) panel (`205` families).
- **Primary Statistic & Multiplicity**: Median across the `3` seeds of per-seed equal-family native utility gain vs
  `r882`, `10,000`-draw family-cluster bootstrap `CI95` and one-sided $p$-value (`boot_seed = 249150`), with
  **Holm-Bonferroni (`m = 2`)** across `SIDE4` (primary) and `SIDE16` (secondary).
- **ADVANCE (`CANDIDATE: <arm>`)** iff all four gates hold on `DL-Val91`:
  1. `median_gain > +0.002`
  2. family-bootstrap `LCB95 > 0`
  3. Holm-adjusted one-sided `p < 0.05` (`m = 2`)
  4. `>= 2/3` seeds positive
  Ties = `NO_ADVANCE`. No AUC, no speed verdicts.
- **Required Diagnostics**:
  - `UNCONSTRAINED` step-40 diagnostic on `DL-Val91` and `OOF` (`1,219` contexts) prior to strict selector fallback;
  - `step40_branch_norm` ($\|(A, B)\|_2$) and `step40_drift` (effective float64 L2 drift $\|\Delta W\|_2$ of merged
    adapter weights across all `29,211` coordinates);
  - trainable parameter counts (`CONTROL = 29,211`, `SIDE4 = 6,232`, `SIDE16 = 24,928`);
  - Elo vs `r882` (`= 1500.0`) with `1,000`-draw family-bootstrap `CI95` and gap vs plateau best `1.1389572` (`r3175`);
  - harm families (`< -1e-4`) and H152 gate-cliff context `#27` (`43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`)
    reported separately and never excluded.

## Host & H239 A1 Resource Guard

- **Host & Affinity**: `xbabe2` CPU only, `taskset -c 52-55`, `nice -n 19`, `1` math thread (`<= 4` total threads),
  tree RSS `< 4 GiB`, free disk `>= 300 GB`. Never touch cores `40-51`, `xbabe0`, `xbabe3`, or PR `#2071`.
- **Admission & Cooperative Pause (H239 A1)**:
  - Admit only if `load5 <= 100` and `MemAvailable >= 30 GiB`.
  - Cooperative pause (via pause file between steps/batches/loads, no `SIGSTOP`) on `load5 > 108` or `MemAvailable < 20 GiB`;
    resume at `load5 <= 100` and `MemAvailable >= 25 GiB` sustained `30 s`.
  - Hard stop only on `MemAvailable < 12 GiB` sustained `30 s`, `< 6 GiB` instant, RSS `>= 4 GiB`, free disk `< 300 GB`,
    threads `> 4`, or the absolute compute stop **`2026-11-03T15:00:00Z`** (`epoch 1793718000.0`).
- **Deadlines & Custody**:
  - Preregistration PR merged to `main` by **`2026-11-03T11:45:00Z`**.
  - Compute hard stop at **`2026-11-03T15:00:00Z`**.
  - Closeout PR merged to `main` by **`2026-11-03T15:45:00Z`**, with S3 custody under
    `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H249/<run_id>/` verified by `VersionId`.
