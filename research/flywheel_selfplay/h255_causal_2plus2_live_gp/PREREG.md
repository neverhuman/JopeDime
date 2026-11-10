# `H255` Preregistration: Causal `2+2` Neural Trajectory Critic & Live Rust GP `2+2` Closed-Loop Flywheel

- **Wave ID**: `H255` (`h255_causal_2plus2_live_gp`)
- **Preregistration Timestamp (UTC)**: `2026-11-03T13:29:00Z` (committed and pushed to `origin/v1/next` BEFORE any Part A or Part B execution)
- **Seeds**: `SEEDS = (255101, 255102, 255103)`, `FOLD_SEED = 255140`, `BOOT_SEED = 255150`, `BOOT_DRAWS = 10_000`
- **Frozen Anchors**:
  - `DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"`
  - `R882_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"`
  - `R882_NATIVE91_MEAN = 1.1377941898164525` (exact reproduction verified before any treatment evaluation)
- **Strict Split Hygiene**:
  - `DL-Train`: `1,219` contexts / `205` families (`dl_split == "DL-Train"`, `source_split in ("train", "inner")`, `has_meta_test == False`).
  - `DL-Val`: `91` contexts / `19` families (`dl_split == "DL-Val"`, `source_split in ("train", "inner")`, `has_meta_test == False`, zero family overlap with `DL-Train`).
  - Outer test folds and sealed meta-test families are never read (`run_outer_test=False`).

---

## 1. Scientific Hypothesis & Root-Cause Mechanism

1. **Root Cause in `run_fused_closed_loop.py` (`revealed_indices = [0, 1, 2, 3]` Self-Leakage & Train-Eval Mismatch)**:
   - In [`run_fused_closed_loop.py` lines 486–491](../../run_fused_closed_loop.py), `predict()` hardcodes `revealed_indices = [0, 1, 2, 3]`.
   - During training, [`training_indices()`](../../automatic_promotion.py) places the 4 oracle best candidates (`order[:4]`) at slate indices `[0, 1, 2, 3]`, causing `Phase12MultiModalAdapter.predict_costs` (`sem_adj` and `traj_cross_attn`) to evaluate $d^2_{\text{joint}} = 0$ from targets `0..3` to themselves and copy their ground-truth residual `r_rev` directly.
   - At `DL-Val` evaluation time (`utility()`) and during GP proposal ranking, `selections = np.arange(M)` places arbitrary raw pool indices `[0, 1, 2, 3]` (median best rank `181.0 / 1,152` on `DL-Val`) into `revealed_indices`.
2. **Part A Hypothesis (Causal `2+2` Neural Trajectory Critic under Equal 4-Reveal Budget)**:
   - Replacing the non-causal `revealed_indices = [0, 1, 2, 3]` call with a **2-stage causal trajectory pass**:
     - **Step 1 ($K_{\text{rev}} = 0 \to 2$)**: Zero-reveal prior pass (`revealed_indices=None`) + 5-fold family-grouped `DL-Train` OOF Hurdle prior selects $c_1, c_2$ (`2` inner-CV reveals).
     - **Step 2 ($K_{\text{rev}} = 2 \to 4$)**: Causal 2-reveal pass feeds the **actually revealed** Step-1 candidates `revealed_indices = [c_1, c_2]`, `revealed_raw_f = [f(c_1), f(c_2)]`, `revealed_gate = [g(c_1), g(c_2)]` into `adapter_10tok.predict_costs(...)` (`sem_adj` + `traj_cross_attn`) combined with the frozen `ACTIVE_2PLUS2_HURDLE` residual/diversity acquisition rule to select $c_3, c_4$ (`2 + 2 = 4` total reveals), and training `adapter_10tok` with **target loss masked strictly to unrevealed candidates $i \notin \{c_1, c_2\}$** will outperform `GATED_025` and `R882` on `DL-Val` (`19` families) and `DL-Train` OOF (`205` families).
3. **Part B Hypothesis (Live Rust GP `2+2` Closed-Loop Flywheel on 32 Stratified `DL-Train` Contexts)**:
   - In live outer GP AST generation + exact 2-fold `JTT1` Rust evaluation (`evaluator.propose_flywheel_candidates` + `evaluator.evaluate_gp_batch_rust(..., run_outer_test=False)`), evaluating `4` proposed ASTs per context via **Live `2+2` Sequential Rust GP Evaluation** (`LIVE_GP_CAUSAL_2PLUS2_FLY`: evaluate 2 AST proposals in Rust CV $\to$ observe their true 2-fold `JTT1` fitness & gate $\to$ condition on those 2 live Rust outcomes + true pool champions to select the next 2 AST proposals) will yield more strict incumbent promotions (`score < incumbent - 1e-6`, `cv_gate_passed=True`) and higher promoted utility gain $\Delta u_{\text{incumbent}}$ than `LIVE_GP_LEGACY_STATIC_4` (`run_fused_closed_loop.py` baseline).

---

## 2. Preregistered Arms, Endpoints, and Decision Rules

### Part A Arms (Equal 4-Reveal Budget on `DL-Val` `91` ctx / `19` fam & `DL-Train` OOF `1,219` ctx / `205` fam)
1. `R882` (static 4-reveal baseline)
2. `GATED_025` (`H217` static 4-reveal baseline)
3. `ACTIVE_2PLUS2_HURDLE` (frozen `H252`/`H254` baseline)
4. `CAUSAL_2PLUS2_R882_ZERO_SHOT` (zero-shot fix of `round-882.pt` using `K_rev=0` in Step 1 $\to$ `revealed_indices=[c_1, c_2]` in Step 2)
5. `CAUSAL_2PLUS2_NEURAL` (causally trained `adapter_10tok` with masked target loss $i \notin \{c_1, c_2\}$ + `PCGrad-EMA` + OOF trust-region gate + `2+2` reveal)
6. `NEGATIVE_CONTROL_PERMUTED_REVEAL` (`CAUSAL_2PLUS2_NEURAL` with Step-1 revealed fitnesses permuted within context; 95% CI vs Step-1-only baseline must cover `0`)

- **Part A Primary Endpoint**: Paired family mean & median gain of `CAUSAL_2PLUS2_NEURAL` minus `GATED_025` (and minus `R882`) on `DL-Val` (`19` families) under equal 4-reveal budget (`10,000`-draw paired family bootstrap 95% CI, Holm-Bonferroni adjusted across comparisons).
- **Part A Required Audits**:
  - Exact Native91 reproduction (`1.1377941898164525`, diff `< 1e-12`).
  - `DOPE` SHA-256 `c51a56d0...` unchanged before and after training.
  - Per-context reveal count histogram (`== 4` for every context).
  - Negative control (`NEGATIVE_CONTROL_PERMUTED_REVEAL`) 95% CI covers `0`.

### Part B Arms (Live Rust GP Evaluation on 32 Stratified `DL-Train` Contexts, 2 Rounds, 16 Proposals/Ctx, 4 Rust Evaluations/Ctx)
1. `LIVE_GP_LEGACY_STATIC_4`: Legacy `run_fused_closed_loop.py` proposal ranking (`p["program"][:4]` arbitrary slots `0..3` as revealed history, one-shot top-4).
2. `LIVE_GP_CHAMPION_COND_STATIC_4`: Champion-conditioned one-shot top-4 (`order_passing[:4]` true pool champions as revealed history).
3. `LIVE_GP_CAUSAL_2PLUS2_FLY`: Champion-conditioned Batch-1 (`2` Rust CV evaluations) $\to$ live residual & structural diversity conditioning on Batch-1's true Rust `JTT1` 2-fold outcomes $\to$ Batch-2 (`2` Rust CV evaluations; total `4` Rust CV evaluations/context/round).
