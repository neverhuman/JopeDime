# H142 Preregistration: Trust-Region Argmin-Aligned Listwise-KL Continuation (`CTRL`, `TR_LR`, `TR_L2SP`, `TR_NOFAIL`) from Frozen `round-882.pt`

- **Wave ID**: `H142` (`research/flywheel_selfplay/h142_trust_region_listwise/`)
- **Track**: `F` (Training objective / trust-region continuation alignment; direct follow-up to `H140`)
- **Owner**: `rd_argon` (`[ARGON]`)
- **Host / Compute**: `xbabe1` CPU-only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4`, `torch.set_num_threads(4)`, `torch.set_num_interop_threads(1)`, `0` GPU seconds, no `qualification.lock`)
- **Status**: `PREREGISTERED` (pre-execution; `0` real continuation updates executed prior to preregistration merge)

---

## 1. Motivation & Falsifiable Hypothesis

In `H140` ([`research/flywheel_selfplay/h140_listwise_weight/README.md`](../h140_listwise_weight/README.md)), increasing the argmin-aligned listwise KL weight from the live control $\lambda = 0.1$ (`L0.1`) to $\lambda = 1.0$ (`L1.0` and `L1.0_nofail`) reduced continuation degradation by more than half on `91`-context `DL-Val` (`+0.006338` and `+0.006807` median gain vs `L0.1`, beating `L0.1` on `2/3` seeds) and produced positive 95% family-cluster bootstrap CIs on `DL-Train` Fold `0` `u4` (`+0.0373` to `+0.0394`, 95% CIs strictly `> 0`). However, running `40` unconstrained AdamW updates at `lr = 1e-4` moved the `29,211`-parameter adapter by `adapter_l2_drift = 0.44438` (`~0.0111` L2 per step), flipping `1–2` additional contexts onto discrete `-2.0` invalid-pick cliffs on `DL-Val` (`3–4` invalid picks vs `r882`'s `2`) and losing to unchanged `round-882.pt` on `3/3` seeds (`-0.024713` median delta vs `r882`).

**Falsifiable Hypothesis (`H142`)**: Constraining `40`-step $\lambda = 1.0$ listwise continuation from frozen `round-882.pt` to an explicit trust region (`adapter_l2_drift <= 0.05`, i.e. $\le \frac{1}{9}$ of `H140`'s `0.444` drift) via either a $10\times$ smaller learning rate (`lr = 1e-5` in `TR_LR` and `TR_NOFAIL`) or an a priori calibrated L2-SP anchor penalty to `round-882.pt` weights (`TR_L2SP`) prevents discrete `-2.0` gate-cliff regressions while retaining the argmin-aligned listwise ranking signal, yielding a positive median `DL-Val` regression utility gain `> +0.001` over `round-882.pt` with Holm-adjusted 95% family-cluster bootstrap `LCB95 > 0`, `>= 2/3` winning seeds, and `<= 80%` top-3 family gain concentration.

---

## 2. Locked Inputs, Checksums & Split Identity

All inputs are staged in isolated read-only custody before execution; the live flywheel directory (`/home/ubuntu/JopeDime/data/flywheel_selfplay/`) is never read or touched during execution:

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Frozen `r882` checkpoint copy | `/home/ubuntu/jope-data/h142/inputs/round-882.pt` | `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` |
| `DL-Val` 91-ctx reference JSON | `/home/ubuntu/jope-data/h142/inputs/best_dev_checkpoint.json` | `30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a` |
| Fused CPU pool loader | `/home/ubuntu/jope-rescue-20260921/h48-holdout-blend/inputs/cpu_exact_fused.py` | `4c9fd1ae952ebefbd24d275da96fbeb56a3ac04b2ef5be163bb6a78ac5971f0e` |
| Selected `DL-Train` + `DL-Val` index | `/home/ubuntu/jope-rescue-20260921/h48-holdout-blend/inputs/selected.json` | `d0795d8fbccaa50aeabbc8efcf0ad5fdbd13ebac88ff70dc4dc2a4bc5fa7f382` |
| Source manifest (`1,310` pools) | `/home/ubuntu/jope-rescue-20260921/h48-holdout-blend/inputs/source_manifest.json` | `7b6ac233d1358be512d1a14a4ca5f023c6c2b483eb65ffbe9358eb2b1b2b3d05` |
| `H58` context feature cache | `/home/ubuntu/jope-rescue-20260921/h58-context- prior/context_features_1310.npz` | `7817f6e61a4533bf96342bb7e6fb621e0f20ff9282eb8bd6d1e811a209612661` |
| `H121` loss-primitive module | `/home/ubuntu/JopeDime/.agent/worktrees/agy_h61/research/flywheel_selfplay/h121_7day_wave_a/run_wave_a.py` | `e40ed007a5fb966dc23df1379291fd441fa1e0ef9a6b034717a26eb5ecdf7213` |
| `H142` fold manifest (`fold_seed=142140`) | `research/flywheel_selfplay/h142_trust_region_listwise/fold_manifest_142140.json` | `3ffa1e8bfb95390435fde3f3dcb50462e1689e91eac3218946e44d2f237d9f68` |
| `H142` CPU executor | `/home/ubuntu/jope-rescue-20260921/h142-trust-region-listwise/run_h142_cpu.py` | `d5215234c2659d2a470ac95c0900827bfad79704b7fb5ac7fe952cdb633d4a8e` |

- **Evaluation identity**: `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`
- **Fresh `DL-Train` 5-fold family split (`fold_seed = 142140`)**:
  - Fold `0` (held-out assessment): `41` unseen families, `274` contexts (disjoint from all continuation updates).
  - Folds `1..4` (continuation fit): `164` families, `945` contexts.
- **Primary `DL-Val` panel**: `19` families, `91` contexts (`best_dev_checkpoint.json`).
- **Fresh continuation seeds**: `142101, 142102, 142103` (never used in prior waves).

---

## 3. Fixed 4-Arm Experimental Design & A Priori Trust-Region Calibration

Every cell (`4` arms $\times$ `3` seeds = `12` cells) starts from a fresh copy of `round-882.pt` (`adapter.*` initial SHA-256 `740174e9f0d29b69ea1b3dad2e4545e40eff541501e6bfdec1e79dbe96912183`) and executes `40` AdamW steps (`5` continuation rounds $\times$ `8` steps/round, `batch_size = 32` contexts/step sampled from the `945` fit-fold contexts using paired stream seeds `paired_stream_seed(seed, round_idx, step_idx)` so all `4` arms see identical context batches on a given seed). Total real optimizer updates across the wave = `480`.

### Per-Context Loss Formulation

For a context $c$ with candidate pool utilities $u_c \in \mathbb{R}^{K_c}$ and predicted loss scores $s_c = \text{net}(X_c) \in \mathbb{R}^{K_c}$ (where inference selects $\arg\min_i s_{c,i}$ and target loss is $\ell_c = -u_c$):
- **Pairwise hinge loss (`H121.pairwise_hinge_loss`)**: $\mathcal{L}_{\text{hinge}}(s_c, \ell_c)$ over all candidate pairs (`margin = 0.05`).
- **Argmin-aligned listwise KL loss (`H121.listwise_kl_loss`, temperature $\tau = 0.1$)**:
  $$\mathcal{L}_{\text{KL}}(s_c, \ell_c; \tau) = \text{KL}\!\left(\text{softmax}\!\left(\frac{-(\ell_c - \min \ell_c)}{\tau}\right) \;\middle\|\; \text{softmax}\!\left(\frac{-(s_c - \min s_c)}{\tau}\right)\right)$$
- **Valid-only listwise KL (`TR_NOFAIL`)**: evaluates $\mathcal{L}_{\text{KL}}$ restricted to candidates with $u_{c,i} > -1.0$ whenever a context has $\ge 2$ valid candidates (0 otherwise), keeping $\mathcal{L}_{\text{hinge}}$ over all candidates so valid-vs-invalid separation remains active.
- **L2-SP trust-region anchor (`TR_L2SP`)**:
  - In `H140`, `L1.0` had `median_pre_clip_grad_norm = 2.623127` and 40-step L2 drift `0.44438` (`0.01111` per step at `lr = 1e-4`).
  - To bound 40-step adapter L2 drift to $\le 0.05$ at `lr = 1e-4`, `TR_L2SP` adds the L2-SP quadratic penalty $\frac{\beta}{2}\|\theta - \theta_{r882}\|_2^2$ with $\beta = 50.0$ ($\approx 2.623 / 0.05$, matching gradient norm at the `0.05` boundary) and applies decoupled L2-SP retraction $\theta \leftarrow \theta - \gamma(\theta - \theta_{r882})$ after each AdamW update with $\gamma = 0.25$ (so steady-state AdamW preconditioned drift is bounded by $\Delta_{\text{step}} / \gamma \approx 0.0111 / 0.25 \approx 0.044 \le 0.05$).

| Arm | $\lambda_{\text{KL}}$ | $\tau$ | KL Candidate Mask | `lr` | `weight_decay` | `l2sp_beta` | `l2sp_gamma` | Target 40-Step Adapter L2 Drift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CTRL` | `0.1` | `0.1` | All candidates | `1e-4` | `1e-4` | `0.0` | `0.0` | Unconstrained (`~0.44` baseline) |
| `TR_LR` | `1.0` | `0.1` | All candidates | `1e-5` | `1e-4` | `0.0` | `0.0` | $\le 0.05$ (`10x` step reduction) |
| `TR_L2SP` | `1.0` | `0.1` | All candidates | `1e-4` | `1e-4` | `50.0` | `0.25` | $\le 0.05$ (L2-SP anchor to `r882`) |
| `TR_NOFAIL` | `1.0` | `0.1` | Valid-only ($u > -1.0$, $\ge 2$ valid) | `1e-5` | `1e-4` | `0.0` | `0.0` | $\le 0.05$ (`10x` step reduction + valid-only KL) |

### Parameter-Freeze & Integrity Invariants

- Only the `29,211`-parameter adapter (`adapter.*`) has `requires_grad = True`.
- All `1,126,777` non-adapter parameters (`291` state-dict tensors, SHA-256 `bf37bcb9f65fffb40b61afb39b2fdec16239d1f234980d8e44e8e47b107324b4`) and the frozen `DOPE` backbone (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) are verified bitwise identical before and after every cell. Any mismatch aborts immediately.
- Log `adapter_l2_drift = ||theta - theta_r882||_2` and `adapter_linf_drift` for every cell.

---

## 4. Step-1 Unchanged-`r882` Regression Gate

Before any continuation cell runs:
1. Load two independent copies of `round-882.pt` (`reg_a`, `reg_b`) and evaluate native argmin utility across all `91` `DL-Val` contexts and all `274` `DL-Train` Fold-0 contexts.
2. Verify `reg_a` vs `reg_b` `max_abs_diff == 0.0` on both panels, and verify `reg_a` vs `best_dev_checkpoint.json` reproduces `expected_mean_utility = 1.137794200072984` (`1.1377942`) with `91/91` exact context matches (`tol = 1e-7`). Abort before training if this gate fails.

---

## 5. Primary & Secondary Endpoints, Invalid-Pick Accounting, Elo & Decision Rule

### Primary Endpoint (`91`-Context `DL-Val`)
- Family-weighted mean native argmin regression utility across the `19` `DL-Val` families (`91` contexts) under `evaluation_identity = e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83` (higher is better).
- For each arm, report:
  1. Per-seed and median utility and gain vs frozen `round-882.pt` (`1.1377942`) and vs paired `CTRL` (`L0.1`).
  2. `10,000`-draw family-cluster bootstrap 95% CI (`19` families resampled with replacement, `seed = 20261142`) on the 3-seed median family-mean delta vs `r882` and vs `CTRL`.
  3. Holm–Bonferroni step-down adjusted 95% CI lower bound (`Holm LCB95`) across the `3` non-control arms (`TR_LR`, `TR_L2SP`, `TR_NOFAIL`).

### Secondary Endpoints (`DL-Train` Held-Out Fold `0` + Audits)
1. **Unseen `DL-Train` Fold `0` (`41` families / `274` contexts)**:
   - Native argmin regression utility (`native_unrev`), top-4 `u4` utility, `top1_regret`, and `all_cand_expected_regret_tau01` (`tau = 0.1`), with `10,000`-draw family-cluster bootstrap 95% CIs (`seed = 20261143`) vs `r882` and vs `CTRL`.
2. **5-Fold Family Cross-Fit Selection Audit on Fold `0`**:
   - Partition the `41` Fold-0 families into `5` disjoint sub-folds (`seed = 142140 + 99`). For each held-out sub-fold $k \in \{0..4\}$, select the arm maximizing 3-seed median family-weighted utility on the other `4` sub-folds and evaluate out-of-fold delta vs `r882` and vs `CTRL` on sub-fold $k$.
3. **Invalid (`-2.0`) Pick Accounting (Mandatory Separate Reporting)**:
   - Separately report the exact count and rate of invalid picks ($u_{\text{pick}} \le -1.999999$, i.e. `-2.0` gate failures) for `r882` and for every seed and arm on both `91`-ctx `DL-Val` (`r882 = 2/91`) and `274`-ctx `DL-Train` Fold `0`, plus net invalid-pick delta vs `r882` (`new_invalid_vs_r882`, `rescued_invalid_vs_r882`).
4. **Elo Reporting (`r882 = 1500.0`; No AUC)**:
   - Paired family Elo (`400 * log10((w + 0.5*t) / (l + 0.5*t))` with `10,000`-draw family-cluster bootstrap 95% CI) vs `r882` and vs `CTRL`.
   - Joint 5-competitor (`r882`, `CTRL`, `TR_LR`, `TR_L2SP`, `TR_NOFAIL`) Bradley–Terry Elo anchored at `r882 = 1500.0` with `2,000` family-cluster bootstrap resamples (`seed = 20261144` on `DL-Val`, `seed = 20261145` on Fold `0`). Never report AUC.

### Strict Admission Decision Rule (`ADVANCE` vs `NO_ADMISSION`)
A non-control arm (`TR_LR`, `TR_L2SP`, `TR_NOFAIL`) is marked `ADVANCE` if and only if on the primary `91`-ctx `DL-Val` endpoint vs frozen `r882`:
1. **Median Seed Gain**: 3-seed median family-weighted utility gain vs `r882` is strictly `> +0.001`, AND
2. **Holm-Adjusted Bootstrap CI**: Holm–Bonferroni step-down 95% family-cluster bootstrap CI lower bound across the `3` non-control arms (`TR_LR`, `TR_L2SP`, `TR_NOFAIL`) is strictly `> 0.0`, AND
3. **Seed Replication**: `>= 2/3` seeds strictly beat `r882` (`seed_gain > 0.0`; ties count as non-wins), AND
4. **Anti-Concentration Gate**: No more than `80%` of total positive family gain vs `r882` comes from the top `<= 3` `DL-Val` families (`top3_pos_gain_share <= 0.80`), AND
5. **Ties Rejected**: Any zero or tied gain is rejected.

If no arm satisfies all 5 criteria, the wave verdict is `NO_ADMISSION` (`NO_ADVANCE`), and no checkpoint is promoted.

---

## 6. Resource & Hygiene Guards

- **CPU-only**: `CUDA_VISIBLE_DEVICES=""`, `nice = 19`, `<= 4` threads, no `qualification.lock`.
- **Memory & Disk**: Background `RssWatchdog` samples RSS every `0.25s` and aborts if `RSS >= 6.0 GiB` or host `MemAvailable < 40.0 GB` or scratch directory `>= 3.0 GB`.
- **Artifact Size & Cleanup**: `metrics.json` must be `< 100 KB` (`100,000` bytes). All staged inputs (`/home/ubuntu/jope-data/h142/inputs`) and scratch checkpoints (`/home/ubuntu/jope-rescue-20260921/h142-trust-region-listwise/scratch`) are deleted at closeout.
