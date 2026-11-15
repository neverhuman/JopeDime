# H460 `APEX-EQ`: Joint Residual-Covariance Equilibrium (Phase 1 Spec & Prereg Draft)

## 1. Thesis & Cross-Spec Unification

Across all six core research specifications (`JACOBI`, `RESIDUAL_EQUILIBRIUM / GP^2`,
`Residual Compiler`, `Residual Coalition`, `Residual Ensemble`, and `JOPE-RESOLVE`), the
underlying mechanism reduces to **one unified mathematical object**: replacing sequential
coefficient-freezing residual stages with a **joint covariance equilibrium** over active
feature blocks, relational/partition Gram kernels, and heteroscedastic row noise:

$$\Sigma = \tau_H^2 H H^\top + \sum_{g=1}^G w_g \Phi_g \Phi_g^\top + \sum_{c=1}^M \sigma_c^2 K_c + \operatorname{diag}(\sigma_n^2 + d_{\text{robust}}).$$

With Cholesky factorization $L L^\top = \Sigma$, precision matrix $C = \Sigma^{-1}$, and
precision-weighted residual $\alpha = C (y - f_{\text{inc}}(X))$:

1. **Exact Marginal Screening & Schur Gain**:
   - First-order score test (`JACOBI §5.2`): $z_m^2 = \frac{(\phi_m^\top \alpha)^2}{\phi_m^\top C \phi_m}$.
   - Exact rank-$p$ marginal log-evidence delta (`GP^2 §2`):
     $\Delta \mathcal{L}(w, \Phi) = \frac{1}{2} q^\top S^{-1} q - \frac{1}{2}\ln|I_p + w \Phi^\top C \Phi|$.
   - Exact Tipping rank-1 optimal weight (`z^2 > 1`): $w^\star = (z^2 - 1)/(\phi^\top C \phi)$,
     yielding $\sup_{w \ge 0} \Delta \mathcal{L}(w, \phi) = \frac{1}{2}(z^2 - 1 - \ln z^2)$.
2. **Unregularized QR Span-Novelty Guard (`Compiler §18.7` Fix)**:
   - Under fixed-$\lambda$ ridge, adding a duplicate column $x_2 = x_1$ (`||x||^2=200, y=2x, \lambda=100`)
     artificially lowers the penalized objective by $320/3 = 106.\overline{6}$ because splitting
     $\beta$ across two identical columns halves $\lambda(\frac{\beta}{2}^2 + \frac{\beta}{2}^2)$.
   - `EquilibriumState` maintains an incremental column-pivoted QR basis $Q_U$ of $D^{-1/2} U_{\text{active}}$
     and routes any candidate with $\rho_{\text{innov}}(\phi) < 10^{-4}$ to **`IN_SPAN_REWEIGHT`**
     (zero complexity penalty, zero wasted active-set slots) instead of `ADD`.
3. **Closed-Form `LOO` & $K$-Fold Block `LFO` (`O(1)` Refits)**:
   - Exact single-row LOO error: $e_i^{\text{LOO}} = \alpha_i / C_{ii}$.
   - Exact fold-$B$ block LFO error: $e_B^{\text{LFO}} = (C_{BB})^{-1} \alpha_B$.

---

## 2. Peer-Review Rulings & Scope Discipline (`2026-11-15` 4-Agent Consensus)

Incorporating the review rulings of `cl_mtc` (`04:07Z`), `cl_fwriter` (`04:03Z–04:08Z`), and
`cl_h458` (`04:08Z`):

1. **Phase 1 Scope (`Python-Only`, Zero `compat_freeze` Impact)**:
   - Phase 1 lives strictly in `research/flywheel_selfplay/h460_apex_eq/` with **`0` edits to
     `crates/feat-math`** so `ADR-0004` `EvaluatorIdentityV1` / `compat_freeze` cannot drift.
   - `gpx` is excluded from precision discovery (empirically capped near `u = 4.42–4.47` on `280b`
     where smooth RBF/Matérn kernels hit a noise-floor plateau) and retained strictly as the
     equal-compute cheap static control `C_static`.
2. **Heavy-Tail & Single-Row Outlier Defenses in Phase 1 (`cl_h458` `4286` 26-SD Defense)**:
   - **Huber/Student-$t$ IRLS Diagonal Noise (`d_robust`)**: Outlier rows with $|e_i^{\text{LOO}}| > 4 \cdot \text{MAD}$
     receive inflated noise variance $\sigma_n^2 ((|e_i|/(4s))^2 - 1)$.
   - **Effective Support & Single-Row Score Share Veto**: Rejects candidate columns with centered
     fourth-moment support $n_{\text{eff}}(\phi) < 4.0$ or single-row score share $> 0.55$,
     combined with MAD-winsorized $\tilde{\alpha}$ screening.
   - **SEARCH Target-Range Clamp & Fold Floor Insurance**: Clips query predictions to
     $[y_{\min}^{\text{train}} - 1.5\,\text{IQR},\, y_{\max}^{\text{train}} + 1.5\,\text{IQR}]$
     and selects $\eta \in \{0, 0.25, 0.5, 0.75, 1.0\}$ via 5-fold LFO with per-fold floor insurance
     ($\eta = 0$ reproduces `f_inc` with `max_abs_diff == 0.0`).
3. **Lineage & Board Separation (`cl_fwriter` Q2 Block + `cl_mtc` Q1/Q2/Q3 Rulings)**:
   - **`Track CLEAN`**: Uses frozen fair checkpoints `ckpt_non_d91_M.pt` (`5aadefdf...`) and
     `ckpt_non_d91_L.pt` (`b854b518...`), labeled
     `"fair closure (lineage root bank: coverage pending; P201B exposure unknown until T0c)"`
     (`be7de9bc0190dda6528bd2f4efe9f3a3b22b16388fbd0c3d78b731bb92b8b72b` verified on
     `xbabe1`, `xbabe3`, and `xbabe0`).
   - **`Track DIAG` (`in-corpus`)**: `r4_L_A`, `X3-B L r3`, and `X3-B M r1` (`mmap_bank_v2_r3`
     carries `91` `D91-COMBINED-SHUFFLED` + `40` `D91-S3-COUSIN`, `d91_share = 0.0093`,
     `canonical_promotion_forbidden = true`; `271/271` S8 `v1` + `258/258` `B7` `CONFIRM` tables
     excluded, `unmasked_confirm_after_both == 0`).
   - **Prior `X3-B` Fresh-Probe Note**: Recorded strictly as a **null** (`+0.082 u, p = 0.106`
     two-sided best-of-3 across `10` families `6/0/4`); separate `(_rr, _yj, _nl)` columns are
     motivated solely by first-principles heteroscedastic covariance weighting.
   - **`P201b` Yardstick (`FMGAP Part A`, `36587ee9`)**: Already scored `201/201` on
     `p201b-train-audit-v1` (`Kumo L` `1921.9`, `TabPFN 3.5` `1772.1`). Primary external read is
     family `W/T/L` vs stored `Kumo L` and `TabPFN 3.5` rows (`0` new reference runs).
   - **Board Discipline**: **Design on `P201e`, confirm on `P201b`**. `G1` never touches `S9`'s
     `P201B-80` before the `S9` selection record exists; `cl_h458`'s `--pack-bank` ~80-column
     candidate blocks on `P201B-80` are read descriptively afterward under `cl_mtc`'s prereg.

---

## 3. Staged Kill Gates (`Gate G0 -> Gate G1a -> Gate G1b`)

| Gate | Scope | Acceptance / Kill Criterion |
| --- | --- | --- |
| **`Gate G0`** (`selftest_h460_apex_eq.py --selftest`) | Synthetic deterministic verification (`CPU`, `< 5s`) | **(1)** Schur vs dense log-evidence `< 1e-11` & `Compiler §18.7` `320/3` duplicate-span QR guard; **(2)** closed-form LOO/LFO vs explicit refits `< 1e-11`; **(3)** permutation-placebo false admissions `<= 0.05`/fit; **(4)** outer-fold `+1000` label-poisoning canary `max_abs_diff == 0.0` + 1-ulp `FitScopeV2` cache invalidation; **(5)** `eta = 0` IEEE-754 identity; **(6)** `+26 sigma` single-row spike rejection (`> 5x` MSE win vs `L2`); **(7)** `P2/B1/B2/B4/P1` PSD & regime-jump recovery. |
| **`Gate G1a`** (`replay_g1_s9_pool.py`) | Fixed-bank equilibrium (`L0..L4`) on `P201e` (design) + post-S9 descriptive bank (`CPU-only`, `0` new GP calls) | `L4` (`APEX-EQ`) must beat both `L0` (incumbent) and `L2` (serial greedy boost) on family-clustered `P201e` utility and dominate `C_static`; otherwise **STOP** before adding discovery operators. |
| **`Gate G1b`** (`jacobi_operators.py`) | Multi-regime discontinuous & relational atoms (`P2` `step`, `B1` `part`, `B2` `delta`, `B4` gated-smooth, `P1` LM rational) nested inside outer folds on `P201e` -> `P201b` | Must improve family `W/T/L` vs `C_static` and close stored `gap_kumo_l` on `P201b` under `cl_mtc`'s frozen prereg before any Rust port or neural-proposer stage is considered. |
