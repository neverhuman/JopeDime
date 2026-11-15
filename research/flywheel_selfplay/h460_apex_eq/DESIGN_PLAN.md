# H460 `APEX-EQ`: Joint Residual-Covariance Equilibrium & Router Non-Smooth Dictionary (Phase 1 Spec v2)

## 1. Thesis & Cross-Spec Unification

Across all six core research specifications (`JACOBI`, `RESIDUAL_EQUILIBRIUM / GP^2`,
`Residual Compiler`, `Residual Coalition`, `Residual Ensemble`, and `JOPE-RESOLVE`), the
underlying mechanism reduces to **one unified mathematical object**: replacing sequential
coefficient-freezing residual stages with a **joint covariance equilibrium** over active
feature blocks, relational/partition Gram kernels, and heteroscedastic row noise:

$$\Sigma = \tau_H^2 H H^\top + \sum_{g=1}^G w_g \Phi_g \Phi_g^\top + \sum_{c=1}^M \sigma_c^2 K_c + \operatorname{diag}(\sigma_n^2 + d_{\text{robust}}).$$

With Cholesky factorization $L L^\top = \Sigma$, precision matrix $C = \Sigma^{-1}$, and
precision-weighted residual $\alpha = C (y - f_{\text{inc}}(X))$:

1. **Scale-Covariant Marginal Screening & Exact Schur Gain**:
   - Base noise variance scales covariantly with residual variance (`\sigma_n^2 = \max(0.10\,\mathrm{Var}(r), 10^{-10})`, `Finding #8`).
   - Scale-normalized score test (`JACOBI §5.2`):
     $z_m^2 = \frac{(\phi_m^\top \alpha)^2}{(\phi_m^\top C \phi_m)\,(\|L^\top \alpha\|_2^2 / n)}$.
   - Exact rank-$p$ marginal log-evidence delta (`GP^2 §2`):
     $\Delta \mathcal{L}(w, \Phi) = \frac{1}{2} q^\top S^{-1} q - \frac{1}{2}\ln|I_p + w \Phi^\top C \Phi|$.
2. **Unregularized QR Span-Novelty Guard (`Compiler §18.7` Fix)**:
   - Under fixed-$\lambda$ ridge, adding a duplicate column $x_2 = x_1$ (`||x||^2=200, y=2x, \lambda=100`)
     artificially lowers the penalized objective by $320/3 = 106.\overline{6}$.
   - `EquilibriumState` maintains an incremental column-pivoted QR basis $Q_U$ of $D^{-1/2} U_{\text{active}}$
     and routes any candidate with $\rho_{\text{innov}}(\phi) < 10^{-4}$ to **`IN_SPAN_REWEIGHT`**
     instead of `ADD`, scanning across **all** ranked candidates without top-24 truncation (`Finding #11`).
3. **Closed-Form `LOO` & $K$-Fold Block `LFO` (`O(1)` Refits)**:
   - Exact single-row LOO error: $e_i^{\text{LOO}} = \alpha_i / C_{ii}$; fold-$B$ block LFO error: $e_B^{\text{LFO}} = (C_{BB})^{-1} \alpha_B$.

---

## 2. Placement, Realistic Sizing & Nesting Discipline (`2026-11-15` 4-Agent Consensus)

Incorporating `cl_mtc` (`04:07Z`, `04:32Z`, `04:37Z`), `cl_fwriter` (`04:34Z`, `04:35Z`, `04:36Z`, `04:37Z`),
and `cl_h458` (`04:08Z`, `04:30Z` `WHATIF_LADDER_v1`):

1. **Primary Placement (`gm_ts` L2 Router Slot) + Pre-Named Secondary (`JAC_V1` Standalone)**:
   - Inside `V0`, `guard_matern_core.fit_l2_residual_gate` (`4-of-5` fold ratchet + `0.004` relative-RMSE
     floor) transfers `1:1` on precision contexts (`Kumo L u >= 2`: `+0.0646` AUDIT vs `+0.0605` SEARCH on
     `P201B`) and `0.48–0.70` on ordinary contexts (`PREREG_G1_DRAFT_v1` §8b).
   - **Shared 4-Arm Router Harness (`router_ratchet.py`, `D1–D4`)**:
     * `R0 ROUTER_ASIS` (`27` Matérn-3/2 configs, `gm_ts` off, bit-identical to `V0`);
     * `R1 ROUTER_DICT` (`30` frozen configs: `27` `R0` + `3` `("dict_eq", 0, 0, eta)` calling
       `fit_router_fold_dictionary` over `scan_best_step_features`, `scan_best_equality_features`,
       `build_gated_smooth_features`, `fit_rational_law_vp`, and `B1/B2` kernels, with nested
       `0.5 * r_bag + 0.5 * r_full` per `D4` and `±2.5 MAD` clip alongside unclamped outputs per `N4`);
     * `R2 ROUTER_L3` (`30` frozen configs: `27` `R0` + `3` `("l3_ridge", 0, 0, eta)` simultaneous inner-CV
       dense ridge on the same fold-discovered banks);
     * `R3 C_STATIC` (`30` frozen configs: `27` `R0` + `3` `("c_static", 0, 0, eta)` smooth RFF ridge
       ensemble via `fit_router_fold_static_control`, matched to `R1`'s worker CPU-seconds per context
       within `±10%`).
   - **Pre-named secondary G1 arm**: Standalone `JAC_V1` (`solve_nested_router_dictionary`), compared against
     `JAC_K`, `L3` simultaneous dense ridge (inner-CV $\lambda$), and equal-compute `C_static` (`N3`).
   - **Honest Effect-Size Expectation (`cl_fwriter` Challenge 1, `WHATIF_LADDER_v1`, `PREREG_G1_DRAFT_v1` §0 & §8b)**:
     * The router non-smooth dictionary targets **`+0.005 to +0.01 u`** (`~+5 to +30 dev91 Elo`, measured lift,
       **not** Rank 1; `WHATIF_LADDER_v1` prices Rank 1 vs `Kumo L` at `+0.15 u` uniform).
2. **Board Correction & Lineage Separation (`cl_mtc` `04:37Z` §2 + `N6`)**:
   - **`P201e` Refusal**: `P201e` has **no** SEARCH/AUDIT bank or `V0` columns in Track S tooling;
     `replay_g1_s9_pool.py` unconditionally refuses `--board p201e` (`Finding #1`) and derives `"board"`
     strictly from verified loader input metadata + SHA-256s (`N6`).
   - **Design Board**: The **`121` `P201B` contexts outside `S9`'s `80`** (`bank_P201B`, `da29e8ec`),
     which carry stored `FMGAP Part A` (`36587ee9`) `Kumo L` (`1921.9`) and `TabPFN 3.5` (`1772.1`) rows
     (`0` new reference runs). `S9`'s `P201B-80` is untouched before `--s9-selection-receipt` exists.
   - **Frozen Decision Board**: **`FITSEL-106`** (`bank_FITSEL`, `f8849ccb`), reporting **`clean-88`** for
     fair-closure claims (`r4_e8` is copy-tier on `18/106`) alongside labelled `in-corpus` all-106.
   - **Lineage Labels**: `ckpt_non_d91_M.pt` (`5aadefdf...`) and `ckpt_non_d91_L.pt` (`b854b518...`) carry
     `"fair closure (lineage root bank: coverage pending; P201B exposure unknown until T0c)"` (`be7de9bc...`
     on `xbabe1`/`xbabe3`/`xbabe0`). `X3-B L r3` and `X3-B M r1` are `Track DIAG` (`in-corpus`,
     `d91_share = 0.0093`, `canonical_promotion_forbidden = true`, `0` `CONFIRM` opens).
3. **Nesting & Parity Gates (`N1–N6`, `cl_fwriter` `04:36Z`)**:
   - **`N1` (Fold-Internal Discovery)**: Every inner fold reruns `build_production_router_dictionary`,
     `calibrate_nested_discovery_placebo_c0`, admission, EB weights, and IRLS strictly on fold-training rows.
   - **`N2` (Nested OOF Damping & Floors)**: `eta`, the `0.004` floor, and the `4-of-5` ratchet use **only**
     nested OOF scores (`eta = 0` reproduces `incumbent_val` bit-identically).
   - **`N3` (Rivals)**: Must beat `(a)` switch-off `V0` router, `(b)` `L3` dense ridge with inner-CV $\lambda$,
     and `(c)` equal-compute `C_static` with positive recorded compute seconds (`L2` serial boost does not count).
   - **`N4` (Clamp Parity)**: Every arm uses the identical output clamp (`mad3` or `search_iqr`), with
     unclamped predictions reported beside clamped predictions.
   - **`N5` (On-Disk AUDIT-Poison Canary)**: `+1000` corruption of `y_val`/`ya`/`y_audit` in `.npz` files on
     disk produces `max_abs_diff == 0.0` across all query predictions; `FitScopeV2` hashes `bank_*`,
     `incumbent_*`, and `static_ctrl_val`.
   - **`N6` (Loader-Verified Board Provenance)**: Receipt `"board"` comes solely from verified input files
     with per-file SHA-256s and `context_ids`.

---

## 3. Staged Kill Gates (`Gate G0 v3 -> Gate G1`)

| Gate | Scope | Acceptance / Kill Criterion |
| --- | --- | --- |
| **`Gate G0` v3** (`selftest_h460_apex_eq.py --selftest`) | `9/9` deterministic verification checks (`CPU`) | **`G0_1`** Schur vs dense `< 1e-11`, Sherman-Morrison `< 1e-13`, `320/3` duplicate-span QR guard, `1000x` scale covariance, & 25-spike full-rank scan; **`G0_2`** LOO/LFO `< 1e-11`; **`G0_3`** `F=400` fixed-pool null `<= 0.05`; **`G0_4`** on-disk `.npz` CLI `+1000` AUDIT-poison canary `max_abs_diff == 0.0` + `bank_train` 1-ulp invalidation + CLI refusals; **`G0_5`** `eta=0` bit-identity (`iqr` & `mad3`); **`G0_6`** `+26 sigma` row-0 spike defense beating **both `L2` and `L3` dense ridge** by `> 3x` (clamped & unclamped); **`G0_7`** `P2/eq/B1/B2/B4/P1` PSD & 4-arm router harness (`R1` beats `R0`, `R2`, `R3` with `|CPU(R3)-CPU(R1)|/CPU(R1) <= 10%`); **`G0_8`** `F=400` pure-noise fits (`n=180, p=6`) in **both** standalone (`nested_rate <= 0.05` vs `38ff3faf` fixed-pool `>= 0.10`) and **router-ratchet (`D3`, exact `guard_matern_core.py` scaling)** forms (`R1` & `R0` null false-fire rates `<= 0.05`); **`G0_9`** `v2` `WP0` adversarial witnesses (`ARENA D1` affine equivariance `< 1e-14`, `D3/D4` regime dispatch, `F03/B.1` hard fold floor, `F04` `evidence_role`, `F05/B.3` Gaussian vs robust telemetry, `F06/B.2` full covariance digest, `F08` read-only CAS worker proposals, `F10` full-bank Golub-Pereyra VarPro + anchor cancellation `delta_a in [-1, 0]`, and target-stratified `lab` generator). |
| **`Gate G1`** (`PREREG_G1_DRAFT_v1.md` + `gm_ts` switch) | 4-arm router (`R0..R3`, primary `R1`) & `JAC_V1` (`S1`, secondary) on `121` non-S9 `P201B` -> `FITSEL-106` (`clean-88` primary + all-106 `in-corpus`) | Intersection gate (`G1-P & G1-R & G1-S & G1-B`, `Q1` accepted): `R1` beats `R0`, `R2`, and `R3` at `alpha=0.05` (bootstrap 95% LCB `> 0` vs `R0`), `R1 - R0 >= -0.02 u` per context (`>= -0.05 u` per family), and `>= 1.5x` ordinary fires with `>= 85%` AUDIT-positive (`§8b`), printing per-fire gain 95% CI, changed-set count, and `LOOK_LEDGER` Elo arithmetic; otherwise **STOP** cleanly. |
