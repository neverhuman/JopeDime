# H132 preregistration — Fisher-anchored (`EWC`) and `BITFIT`-restricted adapter fine-tuning

Status: **PREREGISTERED / NOT RUN**. This document and [manifest.json](manifest.json) freeze the hypothesis, arms, seeds, family split, parameter masks, primary outcomes, admission gates, and resource limits before any H132 assessment-family payload is scored. No real-data training or `fold 0` scoring has occurred.

## Hypothesis

Prior waves show that uniform L2 interpolation or anchoring toward `r882` (`H47` L2-SP, `H105` shrink/anchor) still harms held-out families, whereas `H130` (`CVAR`) improved over `CONTROL` primarily by curbing adapter drift. This indicates that harm arises from **which parameter directions move**, not step size alone. We test two directional constraints starting from fresh hash-verified `r882` copies (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) with frozen `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`):

1. **`EWC` (Diagonal-Fisher-weighted anchor)**: An empirical diagonal Fisher penalty estimated at `r882` on **fit families only** (`folds 1..4`) selectively stiffens high-sensitivity adapter coordinates while allowing low-sensitivity coordinates to adapt.
2. **`BITFIT` (Bias + output-layer restricted fine-tune)**: Restricting optimization to the `17` adapter bias vectors (`595` parameters) plus the `3` adapter scalar-head output projection weight matrices (`mm_prior_out.2.weight` `24`, `traj_delta_head.2.weight` `24`, `parent_delta_net.3.weight` `48` = `96` parameters), for **`691` trainable parameters** across `20` tensors while freezing the remaining **`28,520`** adapter parameters across `27` tensors at `r882`, prevents internal representation drift while calibrating head offsets and linear readouts.

Either mechanism (`EWC` or `BITFIT`) is hypothesized to yield a strict paired family utility gain over unchanged `r882` on held-out assessment families (`fold 0`) with family harm fraction `<= 10%`.

## Design

- **Input Closure**: Only the `1,219` `DL-Train` contexts (`205` families) in the pinned H50 input (`/home/ubuntu/jope-data/h50/input`). Zero `DL-Val`, outer-test, sealed, holdout, or meta-test reads.
- **Fresh 5-Fold Family-Grouped Split (`fold_seed = 132100`)**: Deterministic `numpy.random.default_rng(132100)` permutation of the `205` sorted `DL-Train` family IDs partitioned into 5 equal family folds ([fold_manifest_132100.json](fold_manifest_132100.json), `sha256 = a2a9dd147601ecca76e247576a4575970450d1facc77c332232a5899ac54f440`):
  - `fold 0` = `41` assessment families (`263` contexts).
  - `folds 1..4` = `164` fit families (`956` contexts).
  - Family identities never cross folds.
- **Paired Seeds & Schedule**: Unused seeds `100, 101, 102` (`H131` uses `97–99`), `40` AdamW updates per cell (`20` rounds $	imes$ `2` steps/round, `batch_size = 16`, `slate_size = 192`, `lr = 1e-4`, `weight_decay = 1e-4`, `clip_grad_norm = 1.0`), `3` arms $	imes$ `3` seeds = `9` cells (`360` total optimizer updates).
- **Pre-Update Unchanged-`r882` Regression-First Gate**: Two independent fresh `r882` copies must produce exact-zero differences (`0.0`) on all `263` `fold 0` predictions, utilities, and one `CONTROL` forward+backward gradient probe before Fisher estimation or optimization.
- **Arms**:
  - **`CONTROL`** (`29,211` trainable adapter params across `47` tensors): Exact `r882` / `H130` uniform mean-context objective $\mathcal{L}_{	ext{CONTROL}}(	heta) = rac{1}{|B|} \sum_{c \in B} L_c(	heta)$.
  - **`EWC`** (`29,211` trainable adapter params across `47` tensors):
    $$\mathcal{L}_{	ext{EWC}}(	heta) = \mathcal{L}_{	ext{CONTROL}}(	heta) + \lambda \sum_{i=1}^{29211} F_i (	heta_i - 	heta_{0,i})^2$$
    where $F_i = rac{1}{|\mathcal{F}_{	ext{fit}}|} \sum_{f \in \mathcal{F}_{	ext{fit}}} (
abla_{	heta_i} L_f(	heta_0))^2$ is the frozen family-balanced empirical diagonal Fisher at $	heta_0 = r882$ computed across the `164` fit families only (`0` assessment families touched), and $\lambda$ is fixed a priori in closed form from fit folds only:
    $$\lambda = rac{ar{L}_{	ext{fit}}(	heta_0)}{D_{	ext{ref}}^2 \cdot \left(rac{1}{P} \sum_{i=1}^{P} F_iight)}, \quad D_{	ext{ref}} = 0.45, \quad P = 29,211$$
    so that at the reference 40-step `CONTROL` L2 drift $D_{	ext{ref}} = 0.45$, an isotropic displacement of norm $D_{	ext{ref}}$ incurs an EWC penalty equal to the frozen fit-family baseline loss $ar{L}_{	ext{fit}}(	heta_0)$, with zero hyperparameter tuning on assessment data.
  - **`BITFIT`** (`691` trainable adapter params across `20` tensors; `28,520` adapter params across `27` tensors frozen at `r882`): Same objective $\mathcal{L}_{	ext{CONTROL}}(	heta)$ and schedule, but `requires_grad = True` only on:
    - All `17` adapter `.bias` tensors (`modality_norm.bias` `32`, `prog_412_bottleneck.0.bias` `16`, `prog_412_bottleneck.1.bias` `16`, `cand_query_proj.0.bias` `32`, `cand_query_proj.1.bias` `32`, `mm_cross_attn.in_proj_bias` `96`, `mm_cross_attn.out_proj.bias` `32`, `mm_prior_out.0.bias` `24`, `mm_prior_out.2.bias` `1`, `hist_token_proj.0.bias` `32`, `hist_token_proj.1.bias` `32`, `traj_cross_attn.in_proj_bias` `96`, `traj_cross_attn.out_proj.bias` `32`, `traj_delta_head.0.bias` `24`, `traj_delta_head.2.bias` `1`, `parent_delta_net.0.bias` `48`, `parent_delta_net.1.bias` `48`, `parent_delta_net.3.bias` `1`; total `595` bias parameters), plus
    - The `3` adapter output-layer weight matrices (`mm_prior_out.2.weight` `(1, 24)` = `24`, `traj_delta_head.2.weight` `(1, 24)` = `24`, `parent_delta_net.3.weight` `(1, 48)` = `48`; total `96` weight parameters).
    - Total trainable = `691` parameters (`20` tensors). After every `BITFIT` cell, all `27` non-trainable adapter tensors (`28,520` parameters) are verified to have `0.0` max absolute difference from `r882`.

## Primary Endpoint & Admission Gates

Each primary treatment arm (`EWC`, `BITFIT`) is evaluated against unchanged `r882` on `fold 0` (`41` assessment families, `263` contexts) using a `10,000`-draw paired family-cluster bootstrap (`bootstrap_seed = 132100`) and 2-arm Holm step-down ($lpha_1 = 0.025, lpha_2 = 0.050$). An arm is **ADMITTED** only if all of the following hold simultaneously:

1. **Strict Median Gain**: `median_seed_gain > +0.001` vs `r882`.
2. **Holm Step-Down & Holm LCB**: One-sided bootstrap $p$-value passes 2-arm Holm step-down at family $lpha = 0.05$ AND the Holm-adjusted one-sided lower confidence bound (`LCB97.5` for rank 1, `LCB95` for rank 2) is strictly `> 0.0`.
3. **Family Harm Cap**: Pooled assessment-family harm fraction (seed-mean family gain `< -0.01`) is `<= 10%` (`<= 4/41` families).
4. **All Three Seeds Positive**: Per-seed family mean gain vs `r882` is `> 0.0` on all 3 seeds (`100, 101, 102`).
5. **Non-Degeneracy / Anti-Tie**: Seeds are mutually non-identical (`3` distinct final adapter SHA-256s and positive pairwise context utility differences) and strictly distinct from unchanged `r882` (`adapter_l2_drift > 0` and non-zero utility difference). Exact ties or zero gains are rejected.

Report family-paired Elo and joint Bradley-Terry Elo vs `r882=1500` (primary) and vs `CONTROL=1500` (secondary), plus per-seed family W/T/L counts. Never report AUC.

## Resource & Execution Guardrails

- CPU-only (`CUDA_VISIBLE_DEVICES=""`, `0` GPU seconds, no `qualification.lock`), launched at `nice 15`, at most `4` threads, RSS `<= 6 GiB`, `MemAvailable >= 30 GiB`, `load1 <= 86.7`, and `statvfs` disk used `< 94%` (poll/wait on transient spike, else abort).
- Custody receipts written to `/home/ubuntu/jope-rescue-20260921/h132-ewc-bitfit/`; temporary `.pt` scratch copies deleted immediately after load and on exit.
- Real-data execution (`--execute-wave`) is gated on this preregistration PR merging into `origin/main` (`git merge-base --is-ancestor HEAD origin/main`).
