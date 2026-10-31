# H136 — Preregistration: Family-Balanced 4x Accumulation (`FAMBAL_ACC4`) and Warmup/Cosine Warm Restarts (`COSWR`) vs `CONTROL` and Unchanged `r882`

- **Wave**: `H136` (`research/flywheel_selfplay/h136_fambal_cosine_run/`)
- **Status at Preregistration**: `PREREGISTERED_NOT_RUN` (`0` real optimizer updates, `0` assessment scores read)
- **Owner**: `rd_argon` (`[ARGON]`)
- **Claim Receipt**: `/home/ubuntu/jope_plan/ns/receipts/H136_claim.txt`
- **Pinned Executor (`chmod 0444`)**: `/home/ubuntu/jope-rescue-20260921/h136-fambal-cosine-run/run_h136_cpu.py` (`runner_sha256 = a8fa2908812964cfd5f33207a5bef2e3a2b5a9c2e23558888cd0e45e5b13e001`)
- **Pinned H109 Harness**: `research/flywheel_selfplay/h109_family_balanced_accum/runner.py` (`h109_runner_sha256 = 68ed53d5ba80ce52b2450de526c54f2468129eb13d0cb93cba93a8562953866b`)
- **Execution Gate**: No real-data training or assessment scoring may begin until this preregistration PR is merged into `origin/main`. `run_h136_cpu.py` enforces `git merge-base --is-ancestor HEAD origin/main` at startup.

---

## 1. Motivation & Hypothesis

`H109` (`research/flywheel_selfplay/h109_family_balanced_accum/` on `main`) prepared family-balanced sampling with 4x gradient accumulation and warmup/cosine warm restarts on CPU, but was never scored on real `DL-Train` pools. Meanwhile, `H114` showed the live loop is a discrete utility-lattice trap, and `H124`, `H129`, `H130`, `H132`, and `H133` found that unconstrained `CONTROL` fine-tuning from `r882` concentrates harm in smaller / under-sampled families because uniform context sampling weights families proportionally to their context count (`1` to `50+` contexts/family).

**`H136` tests two distinct interventions against `CONTROL` and unchanged `r882`:**
1. **`FAMBAL_ACC4` (Family-Balanced Sampling + 4x Gradient Accumulation)**: sampling uniformly over the `164` fit families first and then uniformly over contexts within each sampled family across `4` accumulated microbatches (`4 x 16 = 64` effective contexts/step) per AdamW update for `40` updates at flat `lr = 1e-4`.
2. **`COSWR` (Warmup + Cosine Warm Restarts with `CONTROL` Sampler)**: using the exact `CONTROL` batch stream (`1` microbatch of `16` contexts/step) with `4` updates of linear warmup (`2.5e-5 -> 1.0e-4`) followed by cosine annealing with `2` warm restarts (`T_0 = 40 / 3 = 13.333...` updates, restarting at updates `18` and `31`) over `40` updates at peak `lr = 1e-4`.

---

## 2. Data Boundary, Fresh Fold (`fold_seed = 136161`), and Unused Paired Seeds (`161, 162, 163`)

- **Data Closure**: Real S3 `DL-Train` export (`/home/ubuntu/jope-data/h50/input`, `selected.json` `sha256 = 981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`, `source_manifest.json` `sha256 = 99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`, `cpu_exact_fused.py` `sha256 = b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab`): `1,219` regression contexts across `205` families.
- **Strictly Zero `DL-Val` / Outer-Test / Sealed / Holdout Reads**: `run_h136_cpu.py` loads `raw_selected["train"]` only and never reads `raw_selected["dev"]` (`dl_val_reads = 0`, `dl_val_metadata_reads = 0`, `outer_test_reads = 0`, `sealed_metatest_reads = 0`).
- **Fresh 5-Fold Family-Grouped Split (`fold_seed = 136161`)**:
  - `fold_manifest_136161.json` (`sha256 = 6ff1a5c48564a2b13b83f3b50403bfa113f8e1d0361bd5f1c3cb3c83ce872496`).
  - **Assessment fold (`fold 0`)**: `41` families, `245` contexts (never read during batch planning or optimization).
  - **Fit folds (`folds 1..4`)**: `164` families, `974` contexts (`245 + 974 = 1,219`).
- **Paired Seeds**: `161, 162, 163` (verified unused across git and `/home/ubuntu/jope_plan/ns/receipts/`).

---

## 3. Starting Checkpoint, Frozen `DOPE`, and Non-Trainable Hash Verification

- **Starting Checkpoint**: Each of the `9` cells (`3` arms $\times$ `3` seeds) copies a fresh, hash-verified `r882` checkpoint (`/home/ubuntu/jope-data/h51/ckpt/round-882.pt`, `sha256 = 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) into `/home/ubuntu/jope-rescue-20260921/h136-fambal-cosine-run/scratch/`, verifies its SHA-256, loads it, and deletes the staged copy immediately.
- **Frozen `DOPE` & Non-Trainable Parameter Verification**:
  - `DOPE` composite SHA-256 (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) is verified before and after every cell.
  - Only the `47` non-anchor tensors of `adapter_10tok` (`29,211` parameters) have `requires_grad = True`.
  - All `1,126,777` non-trainable parameters (`dope_model`, `gamenet_anchor`, `adapter_10tok.anchor`, `landmark_bridge`, `nolm_bridge`, `dope_to_lm48`) are snapshotted before each cell and verified after each cell (`non_trainable_sha256_pre == non_trainable_sha256_post` and `non_trainable_max_abs_diff == 0.0`).

---

## 4. Arms and Fixed Hyperparameters (`9` Cells $\times$ `40` Updates = `360` Total Updates)

All three arms optimize the exact `r882` / `H132` per-context surrogate objective (`smooth_l1(beta=1.0) + 0.1 * KL(softmax(-pred/0.2) || softmax(-target/0.2))` over `slate_size = 192` candidates per context) with `AdamW(lr=1e-4, weight_decay=1e-4)`, global gradient norm clip `1.0`, and `40` optimizer steps per cell:

| Arm | Context Sampler (`folds 1..4`: `164` fams / `974` ctxs) | Microbatches $\times$ Batch Size per Update | Learning Rate Schedule (`40` Updates, Peak `1e-4`) |
| --- | --- | --- | --- |
| `CONTROL` | Rotated uniform over the `974` fit contexts (`H132` `CONTROL` recipe) | `1 x 16 = 16` (`40` microbatches/cell) | Flat `1.0e-4` |
| `FAMBAL_ACC4` | Uniform over the `164` fit families, then uniform over contexts within family (`rng = default_rng(seed * 1_000_003 + 109_001)`) | `4 x 16 = 64` accumulated (`160` microbatches/cell; `(loss_mb / 4).backward()`) | Flat `1.0e-4` |
| `COSWR` | Exact `CONTROL` batch plan `(cids, ixs)` per seed | `1 x 16 = 16` (`40` microbatches/cell) | Linear warmup `4` updates (`u=0..3`: `1e-4 * (u+1)/4`), then cosine with `2` warm restarts (`u=4..39`: `T_0 = 40/3`, `0.5e-4 * (1 + cos(pi * ((u-4) % T_0) / T_0))`) |

---

## 5. Step 1 Regression-First Gate, Primary Admission Gates, Elo & Continuous-Regret Co-Readout

1. **Step 1 — Unchanged-`r882` Regression-First Gate**:
   - Before any training cell runs, two independent copies of `r882` are loaded and scored on `fold 0` (`41` families, `245` contexts) and `CONTROL` batch-0 gradients. Any non-zero difference (`two_copy_fold0_pred_max_abs_diff != 0.0`, `two_copy_fold0_utility_max_abs_diff != 0.0`, `two_copy_control_loss_abs_diff != 0.0`, or `two_copy_control_grad_max_abs_diff != 0.0`) aborts execution immediately.
2. **Primary Endpoint & Admission Gates (`FAMBAL_ACC4` and `COSWR` vs Unchanged `r882` and vs `CONTROL` on `fold 0`)**:
   - Metric: equal-family `unrev` normalized Top-1 utility gain on `fold 0` (`41` families).
   - Bootstrap: `10,000` paired family-cluster bootstrap draws (`bootstrap_seed = 136161`).
   - Multiple-testing control: 2-arm Holm-Bonferroni step-down across `FAMBAL_ACC4` and `COSWR` at family-wise $\alpha = 0.05$ ($\alpha_1 = 0.025$ / `LCB97.5 > 0` for rank 1; $\alpha_2 = 0.050$ / `LCB95 > 0` for rank 2), evaluated against **both** unchanged `r882` and `CONTROL`.
   - **Admission requires ALL of the following (against both `r882` and `CONTROL`)**:
     - `median_seed_gain > +0.001` (strict)
     - `Holm LCB95 > 0.0` ($p \le \alpha_k$ and Holm-adjusted one-sided lower confidence bound `> 0.0`)
     - `pooled_family_harm_fraction <= 0.10` (`<= 10%` of the `41` assessment families with 3-seed mean gain `< -0.01`)
     - `all_three_seeds_positive == True` (`gain > 0.0` on all 3 seeds `161, 162, 163`)
     - `seeds_not_identical == True` (`3` unique final adapter SHAs and `min_pairwise_max_abs_context_utility_diff > 0.0`) and `not_identical_to_r882 == True` (ties or zero drift fail admission).
3. **Required Co-Readouts**:
   - **Elo (never AUC)**: family-paired Elo (with 95% bootstrap CI) and 4-player Bradley-Terry Elo anchored at `r882 = 1500` and `CONTROL = 1500`.
   - **Continuous Mean-Regret Co-Readout**: `H121`/`H123` all-candidate mean regret $\frac{1}{|P_c|}\sum_{p \in P_c} (\max_{q \in P_c} u(q) - \hat{u}(p))$, top-1 oracle regret $\max_{q \in P_c} u(q) - u(\arg\min_p \hat{c}(p))$, and softmax-expected regret ($\tau = 0.2$) aggregated equally across the `41` assessment families.
   - **Per-Arm Adapter Drift**: L2 parameter drift $\|\theta - \theta_0\|_2$ from `r882`.

---

## 6. Resource & Thread Guardrails

- `JE_ARROW_MALLOC_CONF=background_thread:false`, `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=""` set before imports.
- CPU only (`0` GPU seconds), `nice >= 19`, `<= 4` threads, `RSS <= 6 GiB` abort, `MemAvailable >= 30 GiB`, `load1 <= 86.0`, `disk / < 93.8%` (pause up to 10 min on transient spike, else abort), `scratch < 2 GiB` deleted on exit.
