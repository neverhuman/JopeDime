# H136 — Family-Balanced 4x Accumulation (`FAMBAL_ACC4`) and Warmup/Cosine Warm Restarts (`COSWR`) vs `CONTROL` and Unchanged `r882`

- **Status**: `PREREGISTERED_NOT_RUN` (`0` real optimizer updates; awaiting merge of preregistration PR into `origin/main` before execution)
- **Preregistration**: [`PREREG.md`](PREREG.md) (`sha256 = a9bbce83d944ab1fab9d88a96b82ea01207fc093f85a08e7cfef117dfda1aa74`)
- **Pinned Executor (`chmod 0444`)**: `/home/ubuntu/jope-rescue-20260921/h136-fambal-cosine-run/run_h136_cpu.py` (`runner_sha256 = a8fa2908812964cfd5f33207a5bef2e3a2b5a9c2e23558888cd0e45e5b13e001`)
- **Pinned H109 Harness**: [`../h109_family_balanced_accum/runner.py`](../h109_family_balanced_accum/runner.py) (`h109_runner_sha256 = 68ed53d5ba80ce52b2450de526c54f2468129eb13d0cb93cba93a8562953866b`)
- **Fold Manifest**: [`fold_manifest_136161.json`](fold_manifest_136161.json) (`fold_seed = 136161`, `sha256 = 6ff1a5c48564a2b13b83f3b50403bfa113f8e1d0361bd5f1c3cb3c83ce872496`)
- **Manifest**: [`manifest.json`](manifest.json)
- **Metrics Placeholder**: [`metrics.json`](metrics.json)

---

## Summary of Preregistered Design

- **Hypothesis**: Sampling uniformly across the `164` fit families (and uniformly over contexts within family) with `4x` gradient accumulation (`FAMBAL_ACC4`, `4 x 16 = 64` effective batch size) or applying `4`-step linear warmup + cosine annealing with `2` warm restarts (`COSWR`, `T_0 = 40 / 3` updates) over `40` AdamW updates from fresh `r882` copies will yield a strict paired family gain (`> +0.001`) over both unchanged `r882` and `CONTROL` on held-out `fold 0` (`41` families, `245` contexts) with `<= 10%` harmed families.
- **Seeds**: `161, 162, 163` (`9` cells $\times$ `40` AdamW updates = `360` total optimizer updates).
- **Data Boundary**: Real S3 `DL-Train` only (`1,219` contexts / `205` families). Strictly zero `DL-Val`, outer-test, sealed, or holdout reads (metadata or payload).
- **Admission Gates (`FAMBAL_ACC4` and `COSWR` vs both `r882` and `CONTROL`, 2-arm Holm at $\alpha = 0.05$, `10,000` bootstrap draws)**:
  1. `median_seed_gain > +0.001` (strict)
  2. `Holm LCB95 > 0.0`
  3. `pooled_family_harm_fraction <= 0.10` (`< -0.01` 3-seed mean family delta)
  4. `all_three_seeds_positive == True`
  5. `seeds_not_identical == True` and `not_identical_to_r882 == True` (ties fail).
- **Co-Readouts**: Family-paired Elo and Bradley-Terry Elo (`r882 = 1500` and `CONTROL = 1500`; no AUC), continuous mean-regret co-readout (`mean_regret_all_candidates`, `top1_oracle_regret`, `softmax_expected_regret`), and per-arm adapter L2 drift.
