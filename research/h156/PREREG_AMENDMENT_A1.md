# H156 Preregistration Amendment A1 — Exact `r882` `SiLU` Submodule Parity & Red-Team Resolutions (`t2300`)

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

## 2. Amendment Scope & Red-Team Resolutions (No Loosening of Gates)

1. **`nn.SiLU()` Submodule Reconstruction (`research/h156/run.py`)**:
   - Replaced all `nn.GELU()` constructors in `configure_adapter_width` with `nn.SiLU()`, matching `FlywheelBlockFormerAdapterNet` exactly across `0.5x`, `1.0x`, and `2.0x`.
   - Precomputed frozen anchor outputs `(a_tot, a_eff)` per cohort (`val` first, immediately verified by `parent.baseline_gate(baseline["values"], ref)` with `max_context_error == 0.0` and `mean_error == 0.0` on `91/91` contexts, mean `1.1377941898164525`, followed by each of the 5 disjoint `train` folds).
   - Outer fold `assess` and `DL-Val` `val` are evaluated strictly after `A{outer}-{seed}-selection.json` is locked to disk.
2. **`W05` / `W20` Step-0 Non-Fallback Identity (`research/h156/analysis.py`, `research/h156/run.py`)**:
   - `choose` and `choose_joint` explicitly record `fallback_r882`, `resolved_arm`, and `resolved_step`. When `W05` or `W20` at `step == 0` improves over `r882`, `fallback_r882` is `False` and `(resolved_arm, resolved_step)` remains `("W05", 0)` or `("W20", 0)` through outer assessment, `DL-Val` evaluation, and `verdict` (`RULE_NEUTRAL` requires `fallback_r882 == True` across all cells and arms).
3. **Parent Source Pinning (`research/h156/manifest.json`, `research/h156/run.py`)**:
   - Added `parent_source_sha256` pinning `../flywheel_selfplay/h146_lookahead_exec/{runner,runtime,science}.py` and verified all parent hashes in `verify(pins)` prior to `parent_executor()` import.
4. **Qualification Lock & Process Budget Discipline (`research/h156/monitor.py`)**:
   - Samples `rss_bytes` and `threads` across both the supervisor (`monitor.py`) and the worker subtree (`run.py`).
   - Enforces `MAX_THREADS = 4` on CPU (`1` worker thread + `1` supervisor thread = `2 <= 4`, matching `H148`/`H151`) and `MAX_GPU_THREADS = 8` when GPU is used under `/home/ubuntu/jope_plan/ns/qualification.lock` (and `/home/ubuntu/jope_plan/qualification.lock`).
   - Ensures any live child process is terminated and waited in `finally:` before unlocking and closing the qualification lock file descriptors.
5. **5-Fold Policy Averaging on `91`-Context `DL-Val`**:
   - On the `1,219`-context / `205`-family `H145` train-only set (`primary`), each context belongs to exactly one outer assessment fold `A`. On the `91`-context / `19`-family `DL-Val` cohort (`secondary`), utilities are averaged across the 5 outer fold policies (`secondary.mean(axis=1)`) per seed before computing family means, bootstrap CIs, and Elo vs `r882 = 1500.0`, so the `91`-context readout evaluates the 5-fold cross-fit policy ensemble rather than a single deployed adapter.
6. **Data-Free Regression Suite (`research/h156/test_h156.py`)**:
   - `13/13` data-free unit tests PASS (`test_configure_adapter_width_preserves_silu_and_exact_w10_parity`, `test_w05_and_w20_step0_preserve_arm_identity_not_replaced_by_r882`, `test_monitor_sample_includes_supervisor_and_worker_and_cpu_thread_cap`, `test_parent_source_sha256_pinned_and_verified`).
