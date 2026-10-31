# H137 Preregistration: Target Smoothing (`TS-0.1` $\epsilon=0.1$, `TS-0.3` $\epsilon=0.3$) vs `CONTROL` and Unchanged `r882`

- **Wave ID**: `H137` (`research/flywheel_selfplay/h137_target_smoothing_run/`)
- **Owner**: `rd_argon` (`[ARGON]`)
- **Timestamp (UTC)**: `2026-10-31T22:40:00Z`
- **Status**: `PREREGISTERED_NOT_EXECUTED` (`optimizer_updates = 0`, `dl_val_reads = 0`, `gpu_seconds = 0.0`)
- **Lineage**: Scored CPU execution of the merged `H121` preregistration (`research/flywheel_selfplay/h121_target_smoothing/PREREG.md`, `sha256 = 127e0481f06c0d9b42690b3091138b5a927af9fa7b90f10fbf88a80735e6b923`; `runner.py`, `sha256 = 36ef51730ebe9db21a59455c99fcb15580252c92bf296f2618911c1dc1624aec`) with prospective fresh `fold_seed = 137171` and fresh seeds `171, 172, 173` (disclosed in [PREREG_AMENDMENT_A1.md](PREREG_AMENDMENT_A1.md)).
- **Locked Custody Runner**: `/home/ubuntu/jope-rescue-20260921/h137-target-smoothing-run/run_h137_cpu.py` (`chmod 0444`, `sha256 = 23f221180129940a800ec5a0c275d7bcb03948a364d9f7783b3655cd0274ae70`)

## 1. Hypothesis & Arms

Target smoothing blends each context's sampled candidate targets toward its within-slate mean (`y' = (1 - \epsilon) y + \epsilon \cdot \text{mean}_{\text{ctx}}(y)` via `h121.smooth_targets(targets, epsilon)`) to test whether shrinking extreme target differences reduces fit-fold overfitting and improves out-of-family ranking utility on `fold 0`:

1. **`CONTROL` ($\epsilon = 0.0$)**:
   - Standard `r882`/`H82` recipe unchanged (`40` AdamW updates, `lr = 1e-4`, `weight_decay = 1e-4`, `batch_size = 16`, `slate_size = 192`, `clip_grad_norm = 1.0`).
2. **`TS-0.1` ($\epsilon = 0.1$)**:
   - Identical batch/candidate slates and optimizer recipe as `CONTROL`, with `targets = h121.smooth_targets(targets, 0.1)`.
3. **`TS-0.3` ($\epsilon = 0.3$)**:
   - Identical batch/candidate slates and optimizer recipe as `CONTROL`, with `targets = h121.smooth_targets(targets, 0.3)`.

All `9` cells (`3` arms $\times$ `3` seeds `171, 172, 173` = `360` total AdamW updates) start from independent SHA-verified copies of `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`). Only the `29,211` adapter parameters (`47` tensors) train; `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and all `1,126,777` non-trainable parameters are hash-verified before and after every cell (`max_abs_diff == 0.0`).

## 2. Corpus & Fresh 5-Fold Family-Grouped Partition (`fold_seed = 137171`)

- **Corpus**: Real S3 `DL-Train` only (`1,219` contexts / `205` families; `selected.json` `sha256 = 981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`, `source_manifest.json` `sha256 = 99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`).
- **Partition**: [fold_manifest_137171.json](fold_manifest_137171.json) (`sha256 = a5ef0a687c26ab0da527bd9bfd54ce4d831b81049493abc77a924b6ea8d97af9`):
  - `fold 0` (Assessment): `41` families / `212` contexts (zero gradient updates).
  - `folds 1..4` (Fit): `164` families / `1,007` contexts.
- **Holdout Isolation**: Strictly `0` `DL-Val` (`91` contexts), outer-test, sealed meta-test, or holdout metadata/payload reads.

## 3. Step 1 Regression-First Gate & Execution Guards

- **Step 1 Regression Gate**: Two independent copies of unchanged `r882` evaluated on `fold 0` before any training cell must satisfy `two_copy_fold0_pred_max_abs_diff == 0.0`, `two_copy_fold0_utility_max_abs_diff == 0.0`, `two_copy_control_loss_abs_diff == 0.0`, and `two_copy_control_grad_max_abs_diff == 0.0`.
- **Resource & Load Guards**:
  - CPU only (`CUDA_VISIBLE_DEVICES=""`, `0.0` GPU seconds, no `qualification.lock`).
  - `JE_ARROW_MALLOC_CONF=background_thread:false` and `RAYON_NUM_THREADS=1` set before imports; `nice 19`, `<= 4` threads.
  - **Pause-on-load guard**: whenever `load1 > 86.7`, the runner pauses (`sleep 15s` loop) until `load1 <= 86.7` and **never aborts** on load spikes.
  - `RSS <= 6 GiB`, `MemAvailable >= 30 GiB`, `disk / < 93.8%`, `scratch < 2 GiB` (cleaned on exit).

## 4. Primary Endpoint, Admission Rule & Co-Readouts

- **Primary Endpoint**: Paired per-family `unrev` argmin utility delta on `fold 0` (`41` families / `212` contexts), evaluated via `10,000` paired family-cluster bootstrap draws (`bootstrap_seed = 137171`) with 2-arm Holm step-down across `TS-0.1` and `TS-0.3` vs `r882` (and vs `CONTROL`).
- **Admission Gates (all required for `ADMITTED`)**:
  1. `median_seed_gain > +0.001` (strict margin `> +0.001` over both `r882` and `CONTROL`)
  2. `Holm LCB95 > +0.001` (one-sided family-cluster bootstrap lower confidence bound after 2-arm Holm step-down)
  3. `pooled_family_harm_fraction <= 0.10` (`<= 10%` of assessment families with 3-seed mean delta `< -0.01`)
  4. `all_three_seeds_positive == True` (`gain > 0` on all 3 seeds `171, 172, 173`)
  5. `seeds_not_identical == True` and `not_identical_to_r882 == True` (ties are not wins)
- **Co-Readouts**:
  - Paired family Elo + 4-contender family-grouped Bradley-Terry Elo (`R882 = 1500` and `CONTROL = 1500`; no AUC).
  - Continuous mean regret over all candidates (`h121.mean_regret`: `mean(max(true_u) - (-pred_cost))`), top-1 oracle regret, softmax-expected regret, and adapter L2 drift from `r882`.
