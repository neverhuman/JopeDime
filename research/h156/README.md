# H156: Family-Grouped Cross-Fit Early Stopping + Capacity Sweep (`{0.5x, 1x, 2x}`)

- **Status**: `COMPLETE` (Attempt 1 `INCOMPLETE`; Attempt 2 `COMPLETE`)
- **Scientific Verdict**: `NULL` (`advance = false` on all 4 cross-fit arms across both the `1,219`-context / `205`-family `H145` train-only cohort and the `91`-context / `19`-family `DL-Val` cohort; `serving_promotion = false`)
- **Source Commit (Attempt 2)**: `cdd61b7aba3251bb48609d40d3cb5a51ce375107` ([PR #1332](https://github.com/neverhuman/JopeDime/pull/1332), following preregistration [PR #1316](https://github.com/neverhuman/JopeDime/pull/1316) at `e2c29474142db1d1e095cf15367fece95401616b`)
- **Protocols**: [PREREG.md](PREREG.md) and [PREREG_AMENDMENT_A1.md](PREREG_AMENDMENT_A1.md)
- **Seeds**: `156101`, `156102`, `156103` (`fold_seed = 156140`, `boot_seed = 156141`)
- **Adapter Width Arms**:
  - `W05` (`0.5x`, `20,807` trainable adapter parameters)
  - `W10` (`1.0x`, `29,211` trainable adapter parameters)
  - `W20` (`2.0x`, `46,595` trainable adapter parameters)
  - `XFIT_JOINT` (joint `(width, step)` cross-fit selector over `{W05, W10, W20} × {0, 10, 20, 30, 40}`)

- **Adoption scope**: `W05` (`20,807` params) and `W20` (`46,595` params) are **research-only** arms. Only the `29,211`-parameter `W10` shape is flywheel-compatible; `W05`/`W20` must never be adopted, promoted, or loaded into the live flywheel or serving path. No arm is authorized for adoption (`serving_promotion = false`).
- **Independent recount (luna_ns_hbaseline, 2026-11-01 05:29Z)**: rebuilt the primary panel from the 15 raw cell files; byte-for-byte equal to `panels.npz:primary`. The `W05_XFIT` sign mismatch (equal-family delta `+0.005647`, CI `[-0.017770, +0.032931]`, vs Elo `1492.62` / `-7.38`) is reproduced exactly and is a real difference between magnitude-sensitive mean utility and pairwise Elo (context W/T/L `83/1030/106`; `55` selected `-2` outcomes), not a metric-direction bug. Verdict remains `NULL`. DOPE `c51a56d0...` on all 15 cells; `protected_reads = 0`.

---

## 1. Attempt History & Baseline Parity Gate

1. **Attempt 1 (`INCOMPLETE`, `0` updates, `0` arms scored)**:
   - Aborted at the pre-training `r882` `baseline_gate` on the `91`-context `DL-Val` set (`max_context_error = 0.3224933702796032`, `mean_error = 0.002191645911309159`).
   - **Root cause**: `configure_adapter_width` in `run.py` reconstructed the five width-scalable `nn.Sequential` submodules (`prog_412_bottleneck`, `cand_query_proj`, `mm_prior_out`, `traj_delta_head`, `parent_delta_net`) with `nn.GELU()` instead of `FlywheelBlockFormerAdapterNet`'s native `nn.SiLU()`. Because activation modules hold no `state_dict` keys, `load_state_dict(strict=True)` succeeded while substituting `GELU` for `SiLU`.
   - **Resolution**: Recorded Attempt 1 as `INCOMPLETE` in `/home/ubuntu/jope_plan/evidence/H156/attempt1/attempt1-incomplete.json`, fixed `configure_adapter_width` to preserve `nn.SiLU()`, added data-free unit test `test_configure_adapter_width_preserves_silu_and_exact_w10_parity` (`10/10` unit tests PASS), and disclosed in [PREREG_AMENDMENT_A1.md](PREREG_AMENDMENT_A1.md) ([PR #1332](https://github.com/neverhuman/JopeDime/pull/1332)).
2. **Attempt 2 (`COMPLETE`, Verdict `NULL`, `1,800` AdamW updates across `15` cells)**:
   - Executed at merged commit `cdd61b7aba3251bb48609d40d3cb5a51ce375107` while holding `/home/ubuntu/jope_plan/ns/qualification.lock` (`gpu_lock_held = true`, released on exit).
   - **Baseline gate (`PASS`)**: `91/91` `DL-Val` contexts matched pinned `r882` reference with `max_context_error = 0.0`, `mean_error = 0.0`, `mean = 1.1377941898164525`.
   - **Audit & invariants (`PASS`)**: `1,800` actual AdamW updates (`15` cells × `3` width arms × `40` steps), positive adapter L2 drift on all `45` arm trajectories, frozen backbone SHA256 `493ba7dc8b7b2fa5efbdab90af2fb0b0f6b99fd3c3fa63efbb9f0177ca3218fc` unchanged throughout, `DOPE` SHA256 `c51a56d03d85bd11a33f1ee8a6ae9eeadbf20987cd20ec33ca50fe289cdcf847`, `0` protected/sealed/holdout reads, `0` refused paths.

---

## 2. Primary Readout: `1,219`-Context / `205`-Family `H145` Train-Only Cohort (5-Fold Out-of-Fold Cross-Fit)

*Note: `r882` was historically trained on `DL-Train`, so `r882` has historical in-sample exposure on this `1,219`-context / `205`-family cohort (`historical_r882_train_exposure = true`). Each candidate score on outer assessment fold $A_k$ (`k = 0..4`) uses only fit folds $F_k$ (`3` folds) for gradient updates and inner selector fold $B_k$ (`1` fold) for step/width selection, locked prior to evaluating $A_k$.*

| Arm | Trainable Params | Raw Context Mean | Equal-Family Mean | Equal-Family $\Delta$ vs `r882` | 95% Family Bootstrap CI | Positive Seeds | One-Sided $p$ | Holm $p$ ($m=4$) | Elo vs `r882` (`1500.00`) | Elo 95% CI | Verdict |
| --- | ---: | ---: | ---: | ---: | --- | :---: | ---: | ---: | ---: | --- | :---: |
| `R882` (baseline) | `29,211` | `1.423073` | `1.385355` | `0.000000` | — | — | — | — | `1500.00` | — | Baseline |
| `W05_XFIT` (`0.5x`) | `20,807` | `1.426214` | `1.391002` | `+0.005647` | `[-0.017770, +0.032931]` | `3/3` (`+0.000161, +0.008318, +0.002069`) | `0.3499` | `1.0000` | `1492.62` (`-7.38`) | `[1483.31, 1501.54]` | `NULL` |
| `W10_XFIT` (`1.0x`) | `29,211` | `1.421130` | `1.383338` | `-0.002018` | `[-0.010686, +0.006000]` | `1/3` (`-0.000084, +0.002400, -0.006597`) | `0.6885` | `1.0000` | `1498.10` (`-1.90`) | `[1493.41, 1502.73]` | `NULL` |
| `W20_XFIT` (`2.0x`) | `46,595` | `1.422562` | `1.385048` | `-0.000307` | `[-0.002458, +0.002122]` | `0/3` (`-0.000753, -0.002933, -0.000458`) | `0.6122` | `1.0000` | `1498.65` (`-1.35`) | `[1494.34, 1501.38]` | `NULL` |
| `XFIT_JOINT` | `{20.8k..46.6k}` | `1.426164` | `1.390278` | `+0.004923` | `[-0.017679, +0.031993]` | `2/3` (`+0.004802, +0.005929, -0.004905`) | `0.3640` | `1.0000` | `1498.74` (`-1.26`) | `[1491.55, 1505.73]` | `NULL` |

---

## 3. Secondary Readout: `91`-Context / `19`-Family `DL-Val` Cohort (5-Fold Cross-Fit Policy Average)

*Note: `DL-Val` was never used for fitting or early-stop selection (`DL-Val` evaluated only after outer-fold selection locks were written).*

| Arm | Raw Context Mean | Equal-Family Mean | Equal-Family $\Delta$ vs `r882` | 95% Family Bootstrap CI | Positive Seeds | One-Sided $p$ | Holm $p$ ($m=4$) | Elo vs `r882` (`1500.00`) | Elo 95% CI | Verdict |
| --- | ---: | ---: | ---: | --- | :---: | ---: | ---: | ---: | --- | :---: |
| `R882` (baseline) | `1.1377942` | `1.153615` | `0.000000` | — | — | — | — | `1500.00` | — | Baseline |
| `W05_XFIT` (`0.5x`) | `1.116047` | `1.132798` | `-0.020817` | `[-0.062786, +0.003208]` | `0/3` (`-0.018944, -0.022946, -0.010551`) | `0.8698` | `1.0000` | `1469.95` (`-30.05`) | `[1425.11, 1518.31]` | `NULL` |
| `W10_XFIT` (`1.0x`) | `1.135406` | `1.151157` | `-0.002459` | `[-0.006941, +0.000816]` | `0/3` (`-0.000716, -0.004415, -0.002588`) | `0.9017` | `1.0000` | `1498.53` (`-1.47`) | `[1475.62, 1520.53]` | `NULL` |
| `W20_XFIT` (`2.0x`) | `1.137030` | `1.153242` | `-0.000373` | `[-0.001515, +0.000492]` | `0/3` (`-0.000203, -0.002438, -0.000435`) | `0.7535` | `1.0000` | `1497.80` (`-2.20`) | `[1485.92, 1508.81]` | `NULL` |
| `XFIT_JOINT` | `1.123363` | `1.138964` | `-0.014651` | `[-0.043562, +0.002798]` | `0/3` (`-0.012833, -0.024344, -0.006950`) | `0.8226` | `1.0000` | `1482.42` (`-17.58`) | `[1435.06, 1530.05]` | `NULL` |

---

## 4. Inner-Selector Decisions Across All 15 `(outer, seed)` Cells

| Cell (`outer-seed`) | Selector Fold $B_k$ | `W05` Selected Step | `W10` Selected Step | `W20` Selected Step | `XFIT_JOINT` Selected `(Arm, Step)` |
| --- | :---: | :---: | :---: | :---: | :---: |
| `A0-156101` | `B1` (`41` fam) | `10` | `0` | `0` | `("W05", 10)` |
| `A0-156102` | `B1` (`41` fam) | `10` | `10` | `0` | `("W05", 10)` |
| `A0-156103` | `B1` (`41` fam) | `0` | `10` | `0` | `("W05", 0)` |
| `A1-156101` | `B2` (`41` fam) | `0` | `0` | `0` | `("W05", 0)` |
| `A1-156102` | `B2` (`41` fam) | `0` | `0` | `10` | `("W05", 0)` |
| `A1-156103` | `B2` (`41` fam) | `0` | `0` | `0` | `("W05", 0)` |
| `A2-156101` | `B3` (`41` fam) | `40` | `0` | `40` | `("W05", 40)` |
| `A2-156102` | `B3` (`41` fam) | `40` | `0` | `0` | `("W05", 40)` |
| `A2-156103` | `B3` (`41` fam) | `40` | `0` | `40` | `("W05", 40)` |
| `A3-156101` | `B4` (`41` fam) | `0` | `10` | `0` | `("W10", 10)` |
| `A3-156102` | `B4` (`41` fam) | `10` | `0` | `0` | `("W05", 10)` |
| `A3-156103` | `B4` (`41` fam) | `0` | `20` | `0` | `("W10", 20)` |
| `A4-156101` | `B0` (`41` fam) | `40` | `20` | `0` | `("W05", 40)` |
| `A4-156102` | `B0` (`41` fam) | `10` | `20` | `20` | `("W05", 10)` |
| `A4-156103` | `B0` (`41` fam) | `0` | `10` | `20` | `("W05", 0)` |

- Step selection counts (`15` cells):
  - `W05`: step `0` chosen `7/15`, step `10` chosen `4/15`, step `40` chosen `4/15`.
  - `W10`: step `0` chosen `9/15`, step `10` chosen `3/15`, step `20` chosen `3/15`.
  - `W20`: step `0` chosen `11/15`, step `10` chosen `1/15`, step `20` chosen `1/15`, step `40` chosen `2/15`.
  - `XFIT_JOINT`: `("W05", 0)` chosen `5/15`, `("W05", 10)` chosen `4/15`, `("W05", 40)` chosen `4/15`, `("W10", 10)` chosen `1/15`, `("W10", 20)` chosen `1/15`.

---

## 5. Sensitivity Analysis: Non-Fallback Resolution of `("W05", 0)`

In `run.py` (`L868-L896`), `selected_step == 0` maps to the `r882` fallback key `("W10", 0)`. For `W10` and `W20`, step `0` is exact `r882`, so `selected_step == 0` is identical to `("W10", 0)`. For `W05` (`0.5x`), however, step `0` is the rank-truncated `r882` projection, which achieved positive inner-selector gain on `5/15` cells (`A0-156103`, `A1-156101`, `A1-156102`, `A1-156103`, `A4-156103`). Evaluating the actual `("W05", 0)` checkpoint on those `5` cells (recorded in `metrics.json` under `w05_step0_nonfallback_sensitivity`) produces the identical `NULL` verdict:

| Cohort | Arm | Equal-Family $\Delta$ vs `r882` | 95% Family Bootstrap CI | Positive Seeds | Holm $p$ ($m=4$) | Elo vs `r882` (`1500.00`) | Verdict |
| --- | --- | ---: | --- | :---: | ---: | ---: | :---: |
| `H145` Train-Only (`1,219` ctx / `205` fam) | `W05_XFIT` (non-fallback `W05_0`) | `+0.003232` | `[-0.022044, +0.032099]` | `3/3` | `1.0000` | `1491.87` (`-8.13`) | `NULL` |
| `H145` Train-Only (`1,219` ctx / `205` fam) | `XFIT_JOINT` (non-fallback `W05_0`) | `+0.005351` | `[-0.018530, +0.034508]` | `3/3` | `1.0000` | `1498.41` (`-1.59`) | `NULL` |
| `DL-Val` (`91` ctx / `19` fam) | `W05_XFIT` (non-fallback `W05_0`) | `-0.024793` | `[-0.081760, +0.009200]` | `0/3` | `1.0000` | `1474.36` (`-25.64`) | `NULL` |
| `DL-Val` (`91` ctx / `19` fam) | `XFIT_JOINT` (non-fallback `W05_0`) | `-0.020026` | `[-0.063761, +0.006948]` | `0/3` | `1.0000` | `1478.03` (`-21.97`) | `NULL` |

---

## 6. Durable Evidence & Reproduction

- **In-repo metrics summary**: [`research/h156/metrics.json`](metrics.json)
- **Attempt 1 (`INCOMPLETE`) receipt**: `/home/ubuntu/jope_plan/evidence/H156/attempt1/attempt1-incomplete.json`
- **Attempt 2 (`COMPLETE`) custody receipts**:
  - `/home/ubuntu/jope_plan/evidence/H156/attempt2/launch.json`
  - `/home/ubuntu/jope_plan/evidence/H156/attempt2/smoke-monitor.json`
  - `/home/ubuntu/jope_plan/evidence/H156/attempt2/run-monitor.json`
  - `/home/ubuntu/jope_plan/evidence/H156/attempt2/run.log`
- **Attempt 2 data & per-cell receipts**:
  - `/home/ubuntu/jope-data/h156/run/metrics.json`
  - `/home/ubuntu/jope-data/h156/run/panels.npz`
  - `/home/ubuntu/jope-data/h156/run/audit.json`
  - `/home/ubuntu/jope-data/h156/run/baseline.json`
  - `/home/ubuntu/jope-data/h156/run/A{0..4}-{156101..156103}.json` and `A{0..4}-{156101..156103}-selection.json`
- **Commands**:
  - Data-free unit tests (`10/10` PASS): `python3 -B research/h156/test_h156.py`
  - Constructor smoke: `python3 -B research/h156/monitor.py --custody /home/ubuntu/jope_plan/evidence/H156/attempt2 --smoke`
