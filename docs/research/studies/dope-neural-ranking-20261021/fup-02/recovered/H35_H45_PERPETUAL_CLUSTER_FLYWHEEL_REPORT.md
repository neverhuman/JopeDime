# Waves H35–H45 Technical Whitepaper (`v4.7` Reconciled): Self-Contained First-Party `DOPE` Encoders, Multi-Split S3 Coverage (`15,151` Gate-Disjoint / `15,116` Maximal-Disjoint / `23,403` 4-Role Subset), `SPQ-1` KL Gate Audit (`1.148685 > 0.35`), and 4-Host (`xbabe1`–`xbabe4`) Cluster Flywheel

*Self-Improving Flywheel Metrics Over Time & Full S3 Multi-Split Scale plot is not mirrored in this repository; its object identity is listed in the [H40–H45 S3 custody receipt](h40_h45_s3_custody_receipt_v4_7_20261026.json).*

## 1. Executive Summary & Scientific Qualification Status

This whitepaper records the complete, receipt-verified results and scientific audit across **Waves `H35`–`H45`** (`v4.7`, reconciled against `/home/ubuntu/.codex-screen-monitor/20261026T135545Z-jope-kl-and-accounting-correction.txt`, `/home/ubuntu/.codex-screen-monitor/20261026T123434Z-jope-evidence-correction.txt`, and `/home/ubuntu/.codex-screen-monitor/20261026T113303Z-jope-full-s3-evidence-correction.txt`):

1. **First-Party `DOPE` High (`629D` $\to$ `256D`) + Analytic Landmarks (`LM_SMART_80D`) + Block-RMS Spectrum-Weighted SVD (`alpha = 0.65`) (`Waves H38–H39`)**:
   - Eliminates external foundation-model dependencies (`Kumo`, `TabICL v2`, `Mitra v2`) while preserving `4`-token cross-attention (`M0: AST+Schema+1D+Target`, `M1: Pairwise+Ridge+RF+Interactions`, `M2: DOPE High 256D`, `M3: Block-RMS SVD 64D`).
   - Even with **zero checkpoint dependencies** (`SMART_SVD_FIRST_PARTY_ANALYTIC_NO_CKPT`, `M0 + M1 + LM_SMART_80D + SVD`), pure first-party analytical descriptors achieve competitive Stage-A ranking (`st_val_gate_32` $u_w = -0.979425$ at Round 2).
2. **`Wave H40-Fast` (`112` Guarded Adapter Updates in `115.22s`) & Post-Guard `Wave H40B` (`48` Updates in `98.0s`) — Cached-Pool Replay Disclosure**:
   - `H40-Fast` (`launch_index=6`) executed `112` vectorized 5-horizon (`k in {1, 4, 8, 16, 32}`) adapter updates inside `RuntimeStepGuard` (`115.217s`, `1.0287s/step`, `peak CPU = 23.83% <= 85%`). `H40B` (`DOPE_SELF_CONTAINED_10TOK`) executed `16 + 32 = 48` updates post-guard (`98.0s`, `guard=None`, `cpu_telemetry_availability="UNAVAILABLE_EXECUTED_WITH_GUARD_NONE"`).
   - **Cached-Pool Replay & Repeated-Inspection Disclosure**: Both `H40-Fast` and `H40B` execute **`0` native Rust fits and `0` uncached candidate evaluations**. `dl_val_71` and `dl_holdout_91` are repeatedly inspected development/holdout panels across `H35–H45`, not single-use blind tests.
   - **Matched Controls & Uncertainty (`H40-Fast`)**:
     - On `stage2_20` cached-pool replay under `G1C`, `SMART_SVD_FIRST_PARTY_COMPLETE` mean `holdout30_verify` gain is `+0.010238` (`1W/16T/3L`, `std=0.097830`, `worst=-0.161407`) at Round 0 and `+0.007644` (`2W/14T/4L`, `std=0.099392`) at Rounds 1/2, while `N0_AST_PRIOR_ONLY` reaches `+0.048538` (`2W/13T/5L`, `std=0.217772`, driven by a single `+0.891158` outlier with 5 losses) and `UNGUIDED_PROPOSAL_CONTROL` is `-0.000834` (`0W/19T/1L`).
     - On `dl_holdout_91` (weighted budget utility $u_w$, **lower is better**), `SMART_SVD_FIRST_PARTY_COMPLETE` improves from `-1.133758` (Round 0) to `-1.133948` (Round 1) to `-1.134014` (Round 2, $\Delta = -0.000256$), `ANALYTIC_NO_CKPT` improves from `-1.133597` to `-1.133885` ($\Delta = -0.000288$), `DOPE_SELF_CONTAINED_10TOK` reaches `-1.133947`, and `N0_AST_PRIOR_ONLY` reaches `-1.134141`.
3. **`Track 2` Full-S3 7-Split Corpus (`26,679` Eligible Cells = `23,641` Ordinary 4-Role + `3,038` Excluded) & 3-Policy Disjointness**:
   - **Policy A (Historical Narrower 3-Split Exclusion Without `st_val_gate_32` Protection)**: Excludes `23` datasets (`161` cells) $\to$ `15,228` `train_gradient` cells (`23,480` 4-role subset), recorded strictly as a historical narrower count because `11` `st_val_gate_32` datasets (`77` cells) remained in `train_gradient`.
   - **Policy B (Canonical Gate-Disjoint 4-Split Exclusion Protecting `st_val_gate_32`)**: Excludes **`34` datasets (`238` cells = `161` + `77` `st_val_gate_32`)** $\to$ **`15,151` Gate-Disjoint `train_gradient` cells** across `2,165` datasets (`23,403` Gate-Disjoint 4-role subset: `15,151 + 7,000 + 783 + 469 = 23,403`; `23,403 + 238 + 3,038 = 26,679`), with `0` `st_val_gate_32` overlap.
   - **Policy C (Maximal 6-Set + Physical-Table Disjointness)**: Excludes **`39` datasets (`273` cells = `238` + `28` `excluded_cross_fold_102` + `7` extra `quarantined_3`)** $\to$ **`15,116` Maximal-Disjoint `train_gradient` cells** across `2,160` datasets (`23,368` 4-role subset), with `0` overlap against all 6 protected sets and `0` `eval_phys_set` physical-table overlap (`64D` Block-RMS SVD explains `94.75%` variance on `15,116` fit cells and `88.78%` out-of-sample on `7,000` `eval_gradient_disjoint_v2_panel` cells; `vec256 -> landmark_pred 23D` probe $R^2 = 0.6566$ out-of-sample).
4. **`Waves H41–H44` Live `2+1+1` Evolution (`UNQUALIFIED_COMPLETED_WITH_SPQ1_KL_GATE_VIOLATION_AND_BELOW_UNGUIDED_CONTROL`) & `Wave H45` 4-Host Ratchet (`REJECTED_RETAIN_INCUMBENT_GENERATION`)**:
   - In `h41_h44_live` (`launch_index=3`, preserved read-only at `/home/ubuntu/.codex-screen-monitor/20261026T135545Z-jope-h41-original/results.json`, SHA-256 `f13380496643d63063505d5a33c7e99431f852a44f3e3a9208a5414691976351`), `ns_mask` on the 16-candidate slate marked standard `"crossover"` / `"transfer"` operators (`14/16 = 0.875` base mass in $p_0$) as new support, forcing $\mathrm{KL}(q \,\|\, p_0) \ge \mathrm{KL}(\mathrm{Ber}(0.2125) \,\|\, \mathrm{Ber}(0.875)) = 1.148685 > 0.35$ (`all_rounds_kl_and_support_admissible = False`).
   - On `stage2_20` `holdout30_verify`, `H41_ONLINE_FP_COMPLETE_TILT_QHAT` (`+0.028847`, `2W/15T/3L` under `G1C`; `+0.028502`, `1W/16T/3L` under `G4`) and `H41_ONLINE_FP_ANALYTIC_ZERO_CKPT_TILT_QHAT` (`+0.028847` `G1C`; `+0.028502` `G4`) outperform `H41_FROZEN_FP_COMPLETE_CONTROL` (`-0.000851`, `1W/16T/3L` under `G1C`), **but both underperform `H41_UNGUIDED_CONTROL` (`+0.035839`, `3W/14T/3L` under `G1C`; `+0.036732`, `1W/17T/2L` under `G4`)**.
   - `H41–H44` executed **`0` adapter optimizer steps and `0` encoder optimizer steps** (reusing `H40-Fast` checkpoints), and `ContextualSurrogateQHatLopoSos` `train_spearman_rho = +0.390249` on `28,192` strict-train rows is an **in-sample training fit correlation**, not held-out qualification.
   - `enforce_spq1_tilt_admission_fail_closed` is now enforced in `h40_h45_perpetual_cluster_flywheel.py` (source file is not tracked in this repository snapshot) before candidate selection (verified by `test_32` in `flywheel_dope_high_only_fixture_verification_receipt_v15_20261026T141435Z.json`, `32/32 PASS`, SHA-256 `c29c4e85c255ef37eed9ba1dcd44cc5822ae5abfa63145dee2e84557658965fa`), and `Wave H45` across `xbabe1`, `xbabe2` (paused for CI lock), `xbabe3`, and `xbabe4` (`338` logical CPUs) rejected promotion (`REJECTED_RETAIN_INCUMBENT_GENERATION`).

---

## 2. Global Additive Campaign Wall-Clock & Compute Accounting (`v4.7`)

| Phase / Segment | Guarded / Unguarded | Launches | Phase Wall-Clock (`s`) | Adapter Micro-Steps | Uncached Candidate Evals | Reserved Native Fits (`actual` CV) | Cumulative Guarded Campaign Interval (`s`) | All-Inclusive Campaign Interval (`[Lower, Upper] s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prior Campaign (`H38–H39`)** | Guarded | `1..19` | `2634.716–2636.716` | — | — | — | `[2634.716, 2636.716]` | `[2634.716, 2636.716]` |
| **`Wave H40-Fast` (`launches 1..6`)** | Guarded | `6` (`L6` completed) | `940.700` (`L6=115.217`) | `562` (`L6=112`) | `0` | `0` (`0`) | `[3575.416, 3577.416]` | `[3575.416, 3577.416]` |
| **Post-Guard `H40B` (`10TOK`) + `task-11260` Tail** | Unguarded (`guard=None`) | `1` post-guard | `162.000` (`98.0 + 64.0`) | `48` (`16 + 32`) | `0` | `0` (`0`) | `[3575.416, 3577.416]` | `[3575.416, 3739.416]` |
| **Aborted Reload Runs (`task-11292`, `11300`, `11318`)** | Unguarded (`guard=None`) | `3` aborted | `720.000` (`189+239+292`) | `48` (`16 x 3`) | `0` | `0` (`0`) | `[3575.416, 3577.416]` | `[3575.416, 4459.416]` |
| **`Track 2` Full-S3 & All-Splits (`launches 1..4`)** | Guarded | `4` (`L4` completed) | `150.156` (`L4=120.251`) | `0` | `0` | `0` (`0`) | **`[3725.572, 3727.572]`** | `[3725.572, 4609.572]` |
| **`Waves H41–H44` Live `2+1+1` (`launches 1..3`)** | Guarded | `3` (`L2,L3` completed) | `204.109` (`L3=20.188`) | `0` | `378` (`cap 720`) | `2,268` (`1,594` actual) | **`[3929.681, 3931.681]`** | `[3929.681, 4813.681]` |
| **`Wave H45` 4-Host Cluster (`launches 1..4`)** | Guarded | `4` (`L3,L4` completed) | `15.435` (`L3=7.150, L4=5.276`) | `0` | `0` | `0` (`0`) | **`[3945.116, 3947.116]`** | `[3945.116, 4829.116]` |
| **Pre-Guard Startup / Reload Overhead (`task-11444..11684`)** | Outside Guard | `12` process starts | `389.000` | `0` | `0` | `0` (`0`) | **`[3945.116, 3947.116]`** | **`[3945.116, 5218.116]`** |
| **Remaining Allowance vs. `4500.0s` Ceiling** | — | — | — | — | `342` (nominal) | `2,052` (nominal) | `552.884s` (guarded only) | **`0.0s` (`[0.0, 554.884]s`; CLOSED)** |

---

## 3. Matched-Control Tables Across `Waves H40–H45`

### 3.1 `Wave H40-Fast` & `H40B` (`0` Native Fits, Cached-Pool Replay)
| Arm | `st_val_gate_32` $u_w$ (R0 $\to$ R2, Lower=Better) | `dl_val_71` (Reused Dev) $u_w$ (R0 $\to$ R2) | `dl_holdout_91` (Reused Holdout) $u_w$ (R0 $\to$ R2) | `stage2_20` `G1C` `holdout30_verify` Mean Gain (R0 $\to$ R2, `W/T/L`, `std`) | `stage2_20` `G4` `holdout30_verify` Mean Gain (R0 $\to$ R2, `W/T/L`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`SMART_SVD_FIRST_PARTY_COMPLETE`** | `-0.980614` $\to$ `-0.980747` | `-0.923415` $\to$ `-0.923519` | `-1.133758` $\to$ `-1.134014` ($\Delta=-0.000256$) | `+0.010238` (`1W/16T/3L`, `0.0978`) $\to$ `+0.007644` (`2W/14T/4L`, `0.0994`) | `-0.002093` (`0W/18T/2L`) $\to$ `-0.005594` (`0W/17T/3L`) |
| **`SMART_SVD_FIRST_PARTY_ANALYTIC_NO_CKPT`** | `-0.979325` $\to$ `-0.979425` | `-0.922845` $\to$ `-0.923012` | `-1.133597` $\to$ `-1.133885` ($\Delta=-0.000288$) | `+0.010238` (`1W/16T/3L`, `0.0978`) $\to$ `+0.007644` (`2W/14T/4L`, `0.0994`) | `-0.002093` (`0W/18T/2L`) $\to$ `-0.005594` (`0W/17T/3L`) |
| **`SMART_SVD_FIRST_PARTY_RESTART_NO_CKPT`** | `-0.979325` $\to$ `-0.980550` | `-0.922845` $\to$ `-0.923482` | `-1.133597` $\to$ `-1.133986` ($\Delta=-0.000389$) | `+0.010238` (`1W/16T/3L`, `0.0978`) $\to$ `+0.007644` (`2W/14T/4L`, `0.0994`) | `-0.002093` (`0W/18T/2L`) $\to$ `-0.005594` (`0W/17T/3L`) |
| **`DOPE_SELF_CONTAINED_10TOK` (`H40B`, Post-Guard)** | `-0.980614` $\to$ `-0.980268` | `-0.923415` $\to$ `-0.923491` | `-1.133758` $\to$ `-1.133947` ($\Delta=-0.000189$) | `+0.010238` $\to$ `+0.007644` (`2W/14T/4L`) | `-0.002093` $\to$ `-0.005594` (`0W/17T/3L`) |
| **`N0_AST_PRIOR_ONLY`** | `-0.980498` $\to$ `-0.980612` | `-0.923210` $\to$ `-0.923401` | `-1.133812` $\to$ `-1.134141` ($\Delta=-0.000329$) | `+0.010238` $\to$ `+0.048538` (`2W/13T/5L`, `0.2178`, outlier `+0.8912`) | `-0.002093` $\to$ `+0.038963` (`1W/16T/3L`, `0.1962`) |
| **`UNGUIDED_PROPOSAL_CONTROL`** | `-0.962155` | `-0.916322` | `-1.127700` | `-0.000834` (`0W/19T/1L`, `0.0037`) | `0.000000` (`0W/20T/0L`) |

### 3.2 `Waves H41–H44` Live `2+1+1` Stage-2 Verification (`0` Optimizer Steps; Status: `UNQUALIFIED_COMPLETED_WITH_SPQ1_KL_GATE_VIOLATION_AND_BELOW_UNGUIDED_CONTROL`)
| Arm | `G1C` `holdout30_verify` Mean Gain | `G1C` `W/T/L` | `G1C` `std` / `CVaR10` / `worst` | `G4_ADAPTIVE_TWO_TIER` `holdout30_verify` Mean Gain | `G4` `W/T/L` | Qualification Disposition |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`H41_ONLINE_FP_COMPLETE_TILT_QHAT`** | `+0.028847` | `2W / 15T / 3L` | `0.149214` / `-0.046131` / `-0.057422` | `+0.028502` | `1W / 16T / 3L` | **UNQUALIFIED** (`KL=1.148685 > 0.35`; below `UNGUIDED`) |
| **`H41_ONLINE_FP_ANALYTIC_ZERO_CKPT_TILT_QHAT`** | `+0.028847` | `2W / 15T / 3L` | `0.149214` / `-0.046131` / `-0.057422` | `+0.028502` | `1W / 16T / 3L` | **UNQUALIFIED** (`KL=1.148685 > 0.35`; below `UNGUIDED`) |
| **`H41_FROZEN_FP_COMPLETE_CONTROL` (Round-0)** | `-0.000851` | `1W / 16T / 3L` | `0.015641` / `-0.036518` / `-0.057422` | `-0.002093` | `0W / 18T / 2L` | Baseline Frozen Control |
| **`H41_UNGUIDED_CONTROL` (Matched Control)** | **`+0.035839`** | **`3W / 14T / 3L`** | `0.147615` / `-0.018508` / `-0.021802` | **`+0.036732`** | **`1W / 17T / 2L`** | **Highest Mean Gain Control** |
