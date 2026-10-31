# Waves H41–H44 Live — 4-Arm `2+1+1` Stage-2 Verification (`20.19s`, Status: `UNQUALIFIED_COMPLETED_WITH_SPQ1_KL_GATE_VIOLATION_AND_BELOW_UNGUIDED_CONTROL`)

## 1. Execution, Qualification Status & `SPQ-1` Policy Tilt Accounting
- **Status**: **`UNQUALIFIED_COMPLETED_WITH_SPQ1_KL_GATE_VIOLATION_AND_BELOW_UNGUIDED_CONTROL`** (`qualified_for_promotion = False`)
- **Original Preserved Bundle (`2026-10-26T13:48:53Z`)**: Preserved read-only at `/home/ubuntu/.codex-screen-monitor/20261026T135545Z-jope-h41-original/` and `h41_h44_live/original_preserved_20261026T134853Z/` (`results.json` SHA-256 `f13380496643d63063505d5a33c7e99431f852a44f3e3a9208a5414691976351`, `README.md` `deedfab8380ddf313a808d65902ad27d08b2177a9beb310af5c6d67650192be5`).
- **In-Guard Wall Clock**: **`20.188s`** in `launch_index=3` (`204.109s` across `launches 1..3`; `uncached_evals=10` in `launch_index=3` / `378` cumulative across `launches 1..3`; `reserved_fits=60` in `launch_index=3` / `2,268` cumulative, `1,594` actual CV fold fits; **`adapter_optimizer_steps = 0`**, **`encoder_optimizer_steps = 0`** — reuses frozen `H40-Fast` checkpoints).
- **Chained Global Campaign Wall-Clock Accounting**: Prior through `Track 2` `[3725.572, 3727.572]s` + `H41–H44` `204.109s` = **`[3929.681, 3931.681]s` guarded** (`[3929.681, 4813.681]s` including post-guard `H40B` `98.0s`, `task-11260` tail `64.0s`, and aborted reload runs `720.0s`; `0.0s` remaining allowance at the upper bound).
- **Wave H42 `SPQ-1` Policy Tilt (VIOLATED — UNQUALIFIED)**:
  - `kl_source = "measured_tilt_leaf_policy"`
  - **`max_conditional_base_kl = 1.148685 > 0.35` (`VIOLATED — UNQUALIFIED`; `all_rounds_kl_and_support_admissible = False`)**
  - `max_new_support_mass = 0.250000 <= 0.25`
  - **Root Cause & Fail-Closed Fix**: In `launch_index=2` and `launch_index=3`, `ns_mask` on the 16-candidate slate marked standard `"crossover"` / `"transfer"` operators (`up to 14/16 = 0.875` of base mass in uniform $p_0$) as new support. Downweighting `ns_mask` to $0.2125 \le 0.25$ forced $\mathrm{KL}(q \,\|\, p_0) \ge \mathrm{KL}(\mathrm{Ber}(0.2125) \,\|\, \mathrm{Ber}(0.875)) = 1.148685 > 0.35$, and `tilted_probs` was used to select `chosen_p` without an admission check. `h40_h45_perpetual_cluster_flywheel.py` (source file is not tracked in this repository snapshot) now enforces `enforce_spq1_tilt_admission_fail_closed` before candidate selection (verified by `test_32_spq1_kl_and_support_boundary_and_negative_fail_closed` in `flywheel_dope_high_only_fixture_verification_receipt_v15_20261026T141435Z.json`).
- **Wave H44 `ContextualSurrogateQHat` (`93D` Bilinear LOPO SoS Ridge — In-Sample Training Fit Diagnostic Only)**:
  - Track A (`FP_COMPLETE_SVD`) `train_spearman_rho = +0.390249` on `28,192` strict-train rows (`w_93_l2_norm = 22.213827`); Track B (`FP_ANALYTIC_ZERO_CKPT`) `train_spearman_rho = +0.390249`. This is an **in-sample training fit correlation**, not an independent LOPO or held-out qualification proof.

## 2. Stage-2 (`stage2_20`) `holdout30_verify` Mean Log-Ratio Gain vs. Incumbent (Matched Controls Explicit)
> **Finding**: While `H41_ONLINE_FP_COMPLETE_TILT_QHAT` (`+0.028847`) and `H41_ONLINE_FP_ANALYTIC_ZERO_CKPT_TILT_QHAT` (`+0.028847`) outperform the Round-0 frozen checkpoint control `H41_FROZEN_FP_COMPLETE_CONTROL` (`-0.000851`) under `G1C`, **both underperform `H41_UNGUIDED_CONTROL` (`+0.035839` under `G1C`; `+0.036732` vs. `+0.028502` under `G4_ADAPTIVE_TWO_TIER`)**. Therefore, these receipts **do not show neural superiority over the unguided control**.

### Under `G1C` (`cv_gate_passed` + Continuous 7F Gain)
- **`H41_ONLINE_FP_COMPLETE_TILT_QHAT` (Track A Checkpoint Reuse, `0` New Adapter Updates, Unqualified SPQ-1 Tilt)**: **`+0.028847`** (`2W / 15T / 3L`, `std=0.149214`, `CVaR10=-0.046131`, worst `-0.057422`)
- **`H41_FROZEN_FP_COMPLETE_CONTROL` (Round-0 Frozen Checkpoint)**: **`-0.000851`** (`1W / 16T / 3L`, `std=0.041172`, `CVaR10=-0.075751`, worst `-0.088840`)
- **`H41_ONLINE_FP_ANALYTIC_ZERO_CKPT_TILT_QHAT` (Track B Zero-Checkpoint Reuse, `0` New Adapter Updates, Unqualified SPQ-1 Tilt)**: **`+0.028847`** (`2W / 15T / 3L`, `std=0.149214`, `CVaR10=-0.046131`, worst `-0.057422`)
- **`H41_UNGUIDED_CONTROL` (Matched Unguided Control — Highest Mean Gain)**: **`+0.035839`** (`3W / 14T / 3L`, `std=0.172582`, `CVaR10=-0.031150`, worst `-0.036747`)

### Under `G4_ADAPTIVE_TWO_TIER` (Two-Tier Quorum + Worst-Fold Floor)
- **`H41_ONLINE_FP_COMPLETE_TILT_QHAT`**: **`+0.028502`** (`1W / 16T / 3L`, `std=0.149288`, worst `-0.057422`)
- **`H41_ONLINE_FP_ANALYTIC_ZERO_CKPT_TILT_QHAT`**: **`+0.028502`** (`1W / 16T / 3L`, `std=0.149288`, worst `-0.057422`)
- **`H41_UNGUIDED_CONTROL` (Matched Unguided Control — Highest Mean Gain)**: **`+0.036732`** (`1W / 17T / 2L`, `std=0.172293`, worst `-0.036747`)
