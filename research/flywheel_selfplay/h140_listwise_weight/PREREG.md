# H140 Preregistration — Argmin-Aligned Listwise-KL Weight Sweep (`L0.1`, `L0.5`, `L1.0`, `L1.0_nofail`) vs `r882`

- **Wave**: `H140` (`h140_rd_argon_t1800`)
- **Owner**: `rd_argon` (`[ARGON]`)
- **Folder**: `research/flywheel_selfplay/h140_listwise_weight/`
- **Execution Mode**: CPU-only on `xbabe1` (`CUDA_VISIBLE_DEVICES=""`, `nice 19`, `<= 4` threads, `RSS < 6 GiB`, `host free >= 40 GB`, `scratch < 3 GB`, NO `qualification.lock`, 0 GPU seconds).

## 1. Scientific Hypothesis

Flywheel selection utility (`fixed-dev-cost-argmin-negative-fitness/v1`) evaluates candidate pools by taking the single top-1 argmin predicted cost (`-fitness[argmin(cost)]` if `gate[argmin(cost)]` else `-2.0`), whereas the live training loop (`run_fused_closed_loop.py`) optimizes a hybrid objective:

$$\mathcal{L}_b(\lambda) = \frac{1}{S}\sum_{j=1}^{S}\text{SmoothL1}_{\beta=1.0}(\hat{y}_{b,j}, y_{b,j}) + \lambda \cdot \text{KL}\!\left(\text{softmax}(-y_b / 0.2)\;\|\;\text{softmax}(-\hat{y}_b / 0.2)\right)$$

with $\lambda = 0.1$ and gate-failing candidates forced to cost $y_{b,j} = 2.0$. Up-weighting the listwise KL weight ($\lambda \in \{0.5, 1.0\}$) or computing the listwise KL distribution strictly over gate-passing candidates (`L1.0_nofail`, $\lambda = 1.0$ over $\{j : \text{gate}_{b,j} = \text{True}\}$ when $|G_b| \ge 2$, and $0.0$ otherwise) tests whether stronger listwise ranking pressure aligned with valid-candidate argmin selection lifts validation utility above the `r882` plateau (`1.1377942`).

## 2. Arms, Seeds, and Continuation Budget (Fixed A Priori)

- **Starting Checkpoint**: SHA-pinned claim-time copy `/home/ubuntu/jope-data/h140/inputs/round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`; never touches the live flywheel directory). Fresh byte-verified copy staged per cell.
- **Arms (4 total; 1 control + 3 primary treatments)**:
  1. `L0.1` (`control = live recipe`): $\lambda = 0.1$, listwise KL over all $S = 192$ slate candidates (gate failures at $2.0$).
  2. `L0.5`: $\lambda = 0.5$, listwise KL over all $S = 192$ slate candidates.
  3. `L1.0`: $\lambda = 1.0$, listwise KL over all $S = 192$ slate candidates.
  4. `L1.0_nofail`: $\lambda = 1.0$, listwise KL computed strictly over gate-passing candidates ($\text{gate}_{b,j} = \text{True}$) per context row $b$ when $|G_b| \ge 2$ (`0.0` when $|G_b| < 2$), while pointwise $\text{SmoothL1}$ remains over all $192$ candidates.
- **Seeds (3 paired seeds)**: `140101`, `140102`, `140103` (`4 arms x 3 seeds = 12 sequential cells`).
- **Continuation Budget (reusing `H137`)**:
  - `20` rounds $\times$ `2` steps/round = `40` AdamW updates per cell (`480` total updates across 12 cells).
  - `batch_size = 16`, `slate_size = 192`, `lr = 1e-4`, `weight_decay = 1e-4`, `grad_clip = 1.0`.
  - Paired batch plans: within each seed $s \in \{140101, 140102, 140103\}$, all 4 arms receive the exact same 40-step sequence of `(context_ids, candidate_indices)` sampled from the `DL-Train` fit split (Folds `1..4`, `164` families / `927` contexts under `fold_seed = 140140`, SHA-256 `0b4aa9b485f6eaec27f95b5242e48fcc6b47e918420c2689b7121705959de482`).
- **Trainable Scope & Bitwise Freeze Guards**:
  - Only the `29,211`-parameter adapter (`47` tensors in `adapter_10tok.*`, excluding `anchor.*`) trains.
  - Abort immediately if `DOPE` SHA-256 $\ne$ `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` or if any of the `291` non-adapter `state_dict` tensors/buffers (`1,126,777` non-trainable parameters + normalizer/anchor buffers, SHA-256 `bf37bcb9f65fffb40b61afb39b2fdec16239d1f234980d8e44e8e47b107324b4`) changes by even 1 bit pre- vs post-cell.

## 3. Data Closure & Step-1 Regression-First Gate

- **Data Closure**:
  - Real S3-backed corpus (`/home/ubuntu/jope-data/h50/input/selected.json`, SHA-256 `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`; `source_manifest.json` SHA-256 `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`; `/home/ubuntu/jope-data/h58/cache/context_features.npz` SHA-256 `75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf`).
  - Strictly **zero** reads of outer test folds, `DL-Holdout` (`93` contexts), or sealed meta-test families.
  - `DL-Val` (`91` contexts / `19` families) is never used in training or step selection; it is evaluated once for the Step-1 `r882` regression gate and once per cell at the end of its 40-step continuation.
- **Step-1 Regression-First Gate**:
  - Before any training cell executes, two independent copies of `round-882.pt` must reproduce `/home/ubuntu/jope-data/h140/inputs/best_dev_checkpoint.json` (`30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a`) under `evaluation_identity = e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83` across all `91/91` contexts within `<= 1e-6` and mean `1.1377942` (`1.1377941898164525`) within `<= 1e-6`, plus `0.0` two-copy prediction/loss/gradient difference.

## 4. Primary & Secondary Endpoints and Decision Rule

- **Primary Endpoint**:
  - Flywheel `91`-context regression validation utility (`fixed-dev-cost-argmin-negative-fitness/v1`, higher is better) under `evaluation_identity = e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`: per-arm median over the 3 seeds (`140101, 140102, 140103`) of gain vs control `L0.1` AND vs `r882`, with `10,000`-draw family-cluster bootstrap 95% CI (`bootstrap_seed = 140140`), reported for both the `91`-context mean (cluster-bootstrapping the `19` families weighted by context counts) and the `19`-family equal-family mean.
- **Secondary Endpoint**:
  - Out-of-sample `DL-Train` held-out Fold `0` (`41` unseen families / `292` contexts under `fold_seed = 140140`) native argmin utility, `unrev` (`u4`) utility, continuous regret co-readout, and `H139`-style 5-fold family-grouped cross-fit selection-optimism audit across `fold_seeds = (140140, 140141, 140142)`.
  - Family-grouped Bradley-Terry Elo (`r882 = 1500.0`, `prior = 0.01`, `1,000` bootstrap draws) on both `DL-Val` (`19` families) and `DL-Train` Fold `0` (`41` families) — **no AUC**.
- **Decision Rule**:
  - Holm-Bonferroni step-down at $\alpha = 0.05$ over the `3` non-control arms (`L0.5`, `L1.0`, `L1.0_nofail`).
  - `ADVANCE` an arm only if:
    1. Median over 3 seeds of gain $> +0.001$ vs `L0.1` AND vs `r882`,
    2. Family-cluster bootstrap 95% CI lower bound $> 0$ vs `L0.1` AND vs `r882` (and Holm step-down passes),
    3. $\ge 2/3$ seeds beat `r882` (and $\ge 2/3$ seeds beat `L0.1`),
    4. Positive family gain is not $> 80\%$ concentrated in $\le 3$ families (`top3_positive_family_gain_share <= 0.80`),
    5. Strict tie rejection (`gain > 0`, `adapter_l2_drift > 0`, `seeds_not_identical`).
  - Otherwise `Verdict = NO_ADMISSION`. Never modify the live flywheel.
- **Pinned Executor**: `/home/ubuntu/jope-rescue-20260921/h140-listwise-weight/run_h140_cpu.py` (`fee6b6f56245d41274124d5875abe82f978a24341561c1837fdce27f52bdbf7e`).
