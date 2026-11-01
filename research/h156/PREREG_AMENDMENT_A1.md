# H156 Preregistration Amendment A1 — Exact `r882` `SiLU` Submodule Parity (`t2300`)

- **Wave**: `H156`
- **Authority**: `/home/ubuntu/nudges/hwave_t2300_gpu_tokbench_agy.md`
- **Original Preregistration PR**: `#1316` (`e2c29474142db1d1e095cf15367fece95401616b`)
- **Attempt 1 Disposition**: `INCOMPLETE` (`0` optimizer updates, `0` arms scored; archived in `/home/ubuntu/jope_plan/evidence/H156/attempt1/attempt1-incomplete.json`)

## 1. Attempt 1 Failure & Root Cause Diagnosis

Attempt 1 aborted at the pre-training `r882` `baseline_gate` sanity check before any training step or arm selection executed:

```text
RuntimeError: baseline exact parity failed: 0.3224933702796032, 0.002191645911309159
```

**Root cause**:
1. In `research/h156/run.py`, `configure_adapter_width` reconstructed the five width-scalable `nn.Sequential` submodules of `FlywheelBlockFormerAdapterNet` (`prog_412_bottleneck`, `cand_query_proj`, `mm_prior_out`, `traj_delta_head`, and `parent_delta_net`) with `nn.GELU()` instead of `FlywheelBlockFormerAdapterNet`'s native `nn.SiLU()` activation.
2. Because `nn.SiLU()` and `nn.GELU()` have no parameters in `state_dict()`, `net.load_state_dict(r882_sd, strict=True)` succeeded without error after replacing `SiLU` with `GELU` during pre-flight width parameter counting (`width_counts`). Consequently, `net.adapter_10tok` evaluated the `r882` baseline with `GELU` activations instead of `SiLU`.
3. In addition, `eval_cache` precomputed frozen anchor terms by pooling `train + val` into a single `<= 8` batch stream instead of grouping `val` and each of the 5 disjoint `train` folds separately.

## 2. Amendment Scope (No Loosening of Gates or Scientific Rules)

1. **`nn.SiLU()` Submodule Reconstruction (`research/h156/run.py`)**:
   - Replaced all `nn.GELU()` constructors in `configure_adapter_width` with `nn.SiLU()`, matching `FlywheelBlockFormerAdapterNet` exactly across `0.5x`, `1.0x`, and `2.0x`.
   - Precomputed frozen anchor outputs `(a_tot, a_eff)` per cohort (`val` first, immediately verified by `parent.baseline_gate(baseline["values"], ref)` with `max_context_error == 0.0` and `mean_error == 0.0` on `91/91` contexts, mean `1.1377941898164525`, followed by each of the 5 disjoint `train` folds).
   - Outer fold `assess` and `DL-Val` `val` are evaluated strictly after `A{outer}-{seed}-selection.json` is locked to disk.
2. **Qualification Lock Discipline (`research/h156/monitor.py`)**:
   - Acquires `/home/ubuntu/jope_plan/ns/qualification.lock` (and `/home/ubuntu/jope_plan/qualification.lock`) via `fcntl.flock(..., LOCK_EX | LOCK_NB)` when GPU is used and explicitly unlocks and closes both file descriptors in `finally:` on exit.
3. **Data-Free Regression Test (`research/h156/test_h156.py`)**:
   - Added `test_configure_adapter_width_preserves_silu_and_exact_w10_parity` verifying that cycling `configure_adapter_width` through `0.5x -> 2.0x -> 1.0x` preserves `nn.SiLU` in all 5 sequential submodules and reproduces the unmodified `r882` forward output with `0.0` max absolute difference (`10/10` unit tests PASS).
4. **Unchanged Scientific Contract**:
   - Seeds (`156101, 156102, 156103`), fold seed (`156140`), bootstrap seed (`156141`), steps (`0, 10, 20, 30, 40`), widths (`0.5x, 1.0x, 2.0x`), 5-fold family-grouped cross-fit selection rule, Holm-Bonferroni step-down gate, Bradley-Terry MM Elo vs `r882 = 1500.0`, and the `0.0` tolerance `baseline_gate` on `91/91` contexts (`mean = 1.1377941898164525`) are 100% unchanged.
