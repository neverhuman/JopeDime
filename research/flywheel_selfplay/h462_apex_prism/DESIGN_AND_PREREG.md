# H462 `APEX-PRISM`: Sovereign Two-Clock Synthesis Wave (`Design & Preregistration`)

## 1. Grounded Empirical Premise (`2026-11-15` Fleet Measurements)

`H462 APEX-PRISM` unifies the five `tips/core/v2/*.md` specifications with tonight's verified empirical findings across `agy_apex_w10`, `cl_h458`, `cl_mtc`, and `cl_fwriter`:

1. **Win-Efficiency & Leverage (`+133.7 Elo` at Equal Mean Utility; `+36.6 Elo` on Singleton `866f`):**
   - `h448.APEX_FAIR_CHAMPION` (`1823.1 Elo`, `family_mean_u = 1.6737`) already exceeds `TabPFN 3.5` (`1.6411`), yet `TabPFN 3.5`'s win shape shifted to `1.6737 u` rates **`1956.8 Elo`**.
2. **The Five-Classic Floor (`+73.1 Elo` to `1896.2`) & Why Clipped Fold-Mean Selection Missed It:**
   - On `18/91` contexts (`98%` with `SEARCH tail > 12`), our fair champion trails `{ridge, CatBoost, LightGBM, XGBoost, RF}`.
   - `cl_h458`'s `06:05Z` measurement proved `dev91`'s 5 folds are **already target-stratified** (`rank spread = 0.0002`). The leak comes from **two selection/extrapolation defects**:
     1. `3 MAD` / `2.5 MAD` residual clipping plus geometric-mean `mean_k log(RMSE_k)` discounts a `10x` single-fold tail explosion to `0.2 * log(10)`, whereas `AUDIT` scores **unclipped pooled `log(RMSE)`** with `3` of every `10` sorted extreme rows guaranteed in `AUDIT` (`common_h317.py:582`).
     2. `Yeo-Johnson` (`yj`) and unconstrained GP polynomials explode in the tail (`line 28`: `-0.64 to -1.71 u`), whereas trees stay bounded in `[min(y_S), max(y_S)]` and `ridge` extrapolates linearly.
3. **Precision / Heavy-Tail Laws (`28` Contexts, `+55 Elo` Standalone / `+96 Elo` Combined):**
   - Smooth uncapped GP plateaued at `4.42 u` on `280b`. Inductive SVD + `1/|Q|` + Golub–Pereyra Variable Projection (`identity`/`log` warps only, `yj` banned) with **Anchor Cancellation (`delta_a in [-1, 0]`)** and a strict **`+0.50 u` across `5/5` folds** gate recovers exact/rational laws without harming noisy tables or `280b`.
4. **Ordinary Contexts (`63` Contexts) & Warm-Startable Dual-Path `PRISM-24M`:**
   - Per-context picking with a `Kumo-Small`-class own prior caps at `1996.8 Elo` (`1947` at `0.05 u` margin). Passing `Kumo L` (`2070+`) under sovereign `S0` ("ours only") requires decorrelated per-row fusion (`rho <= 0.85`).
   - Reconciling `cl_h458`'s `TWO CLOCKS` (`06:06Z`) warm-start requirement with `Solver-Trained Mechanism Synthesis`: `prism_solver_net.py` provides BOTH the warm-startable `warm_scalar_twohot` head (primary `24h/72h` arm) AND `warm_plus_cholesky_r64` — a **zero-perturbation residual `float64` Cholesky `r=64` mechanism adapter** (`gamma_init = 0.0`, bit-identical to the warm checkpoint at step 0, zero-shot testable on frozen latents, and fully differentiable end-to-end).

---

## 2. Architecture & Module Map (`research/flywheel_selfplay/h462_apex_prism/`)

| Stage / Clock | Module | Responsibility |
| --- | --- | --- |
| Custody & `S0` | `contracts_v3.py` | `FitClosureV3` transitive ancestor/corpus provenance, per-outer-fold base-block check (blocking global-OOF leakage), `S0` enforcement, and `T14` forbidden-path refusal. |
| Regime Assay | `assay_regime.py` | `SEARCH`-only `float64` `ASSAY` (`loo_u`, `slope`, `top1_err_share`, `tail_robust_sd`, `is_lattice`) via closed-form uncapped Matérn-5/2 LOO. |
| Fast Clock (`Stage A`) | `edge_certify.py` | Unclipped pooled scored-space certification + `common_h317`-shaped (`3-of-10` target block) `B=2,000` bootstrap maximizing expected family-weighted wins (`5x` singleton leverage), Five-Classic Floor, `8-bag` (`h462.APEX_PRISM_E8`), and `{cid}.npz` block packager. |
| Exact Readout (`Stage B`) | `exact_varpro_battery.py` | Strictly inductive (`T1` `max_abs_diff == 0.0`) SVD + `1/|Q|` + Golub–Pereyra `VarPro` rational/log-monomial battery (`yj` banned) with `delta_a in [-1, 0]` anchor cancellation and `+0.50 u` `5/5`-fold gate. |
| Hint Compiler | `pcr_hint_compiler.py` | `SYMBIONT §7` inner-5-fold OOF `PCR` hints (`Class A` ratios/products/log-ratios; `Class B/C` inner-OOF ARD-GP mean/SD, GCV ridge, ARD-kNN, key-TE + reliability channels) + `20%` counterfactual hint corruption. |
| Slow Clock (`Stage C`) | `prism_solver_net.py` | Row- & slot-equivariant, strictly query-independent (`T1=0`) `PRISM` neural synthesizer supporting `warm_scalar_twohot`, `warm_plus_cholesky_r64` (`gamma_init=0`), `cholesky_basis_r64`, and `stopgrad_cholesky_r64` with counterfactual mechanism-repair episodes. |
| Verification | `selftest_h462_apex_prism.py` | 12 deterministic CPU gates (`P0_1..P0_12`), including the `cl_h458` `06:05Z` target-stratified pooled-vs-clipped criterion fixture. |

---

## 3. Preregistered Product Budgets, Clean-Cohort Protocol & Promotion Gates

- **Sovereign Tier:** `S0` (`"ours_only"`) mandatory for all contenders; external FM columns prohibited.
- **Clean-Cohort Protocol (`be7de9bc`-safe):**
  - Design & hyperparameter freeze: `121` `P201b` non-S9 contexts (`TRAIN` only).
  - Pre-dev selection gate: `FIT-SEL` clean-88 (`0` `dev91` / `CONFIRM` / `FMGAP B/C` overlap).
  - Neural training (`WP3`, gated on `cl_h458` `lines 31, 35, 36`): `1,219` `DL-TRAIN` families + synthetic mechanism/hint-repair prior (`0` `APEX-DEV` / `CONFIRM` exposure).
- **Product-Tier Inference Budgets (`FitClosureV3` enforced):**
  - `medium`: `<= 128` solver evaluations, `<= 60s` wall/context.
  - `high`: `<= 512` solver evaluations, `<= 300s` wall/context.
  - `ultra`: `<= 4096` solver evaluations, `<= 1800s` wall/context.
- **Causal Kill / Promotion Gates:**
  - **Fast Clock (`Stage A + B` on `FIT-SEL` clean-88):** `>= +3` net family-games vs `Kumo L` and `TabPFN 3.5` yardsticks on `TRAIN` boards with `Delta family_mean_u >= -0.01`, `0` false `EXACT-VARPRO` fires on noisy synthetic/train contexts, and positive clipped contrast.
  - **Slow Clock (`Stage C` `24h / 72h` fresh-row probe):** Beat warm-start init on `>= 7/10` clean families by `>= +0.10 u` standalone, `rho(err_PRISM, err_stack) <= 0.85`, and `warm_plus_cholesky_r64 > stopgrad_cholesky_r64` before keeping the Cholesky adapter active.
