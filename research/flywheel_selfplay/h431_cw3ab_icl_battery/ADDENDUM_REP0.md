# H431 Addendum — Paired `rep == 0` (`68`-Task) Comparison of `L_h018` vs `M_h020` / `M_h024`

**Date:** 2026-11-11
**Credit:** Triggered by `cl_fwriter`'s peer review of `H431` (`2026-11-11T21:09:50Z`).
**Verdict impact:** `H431`'s pre-registered verdicts (`BATTERY_AS_EXPECTED` `6/6` on `M_h024`, `PARTB_READY` `6/6`) are **unchanged**. This addendum corrects the descriptive interpretation comparing `L_h018` (`104M`) against `M_h020` / `M_h024` (`24M`).

## 1. Why Unpaired `metrics.json` Over-Interpreted `L_h018`
In `metrics.json`, `M_h020` and `M_h024` were aggregated over all `340` `ICB_v1` tasks (`reps 0..4`, `10` noise-free tasks per family), whereas `L_h018` (`Amendment 2`) scored the `68` balanced `rep == 0` tasks (plus `38` partial `rep > 0` tasks = `106` total, `2` noise-free `rep == 0` tasks per family):
1. **Sample-size bias on `precision_ceiling_u8` (`max`):** A maximum over `2` noise-free tasks per family is not comparable to a maximum over `10` tasks.
2. **Task-difficulty difference between `rep == 0` and `reps 1..4`:** For example, on `B6 arithmetic_pair`, `M_h024` scores `0.8348` `ridgeres` capability on `rep == 0` (`n=6`) vs `0.5283` across all `30` tasks.
3. **All-or-nothing per-task hit rate, not a bin-mean ceiling:** On the shared `rep == 0` tasks, both `24M` and `104M` exhibit **all-or-nothing identification** of the active signal/key column across noise levels (`B7 smooth_1d` `1` active column in `16`: `L_h018` scores `-0.01, +0.02 u` at `noise=0` yet `+2.37, +2.89 u` at `noise=0.05`, where `M_h024` scores `4.28, 0.72 u` at `noise=0` and `0.00, 0.00 u` at `noise=0.05`). A readout bin-mean precision floor would cap `u` high rather than producing `u ~ 0` at `noise = 0`. Therefore, `H431` **does not confirm** a bin-mean precision floor (`A-007` Account 2); `CW-4` Arm `T4` (`--fine-head`) remains a prospective test, while `T3` (`--arch cell`) and `T1` (`--typed-cells`) directly target column/key identification.

## 2. Exact Paired `rep == 0` (`68` Tasks, `22` Noise-Free) Summary

| Checkpoint | Mode | Mean Cap (`68` tasks) | Mean Cap (`all12`) | Mean Cap (`structure6`) | Noise-Free Mean `u8` (`n=22`) | Noise-Free Hit Rate `u8 > 1` (`n=22`) | All-`68` Hit Rate `u8 > 1` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `M_h020` (`24M h20`) | `full` | `0.2601` | `0.2916` | `0.1195` | `0.8039` | `4/22` (`18.2%`) | `17/68` (`25.0%`) |
| `M_h024` (`24M h24`) | `full` | `0.2706` | `0.3036` | `0.1451` | `0.7811` | `6/22` (`27.3%`) | `17/68` (`25.0%`) |
| `L_h018` (`104M h18`) | `full` | `0.1610` | `0.1686` | `0.1161` | `0.2171` | `1/22` (`4.5%`) | `9/68` (`13.2%`) |
| `M_h020` (`24M h20`) | `ridgeres` | `0.3788` | `0.4132` | `0.1158` | `1.9121` | `8/22` (`36.4%`) | `25/68` (`36.8%`) |
| `M_h024` (`24M h24`) | `ridgeres` | `0.3891` | `0.4231` | `0.1361` | `1.8883` | `8/22` (`36.4%`) | `24/68` (`35.3%`) |
| `L_h018` (`104M h18`) | `ridgeres` | `0.3725` | `0.4070` | `0.1136` | `1.6272` | `6/22` (`27.3%`) | `24/68` (`35.3%`) |

See `metrics_rep0_paired.json` for full per-family paired breakdowns on both `rep0_68` and `all_shared_106`.
