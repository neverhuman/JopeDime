# H140 — Argmin-Aligned Listwise-KL Weight Sweep (`L0.1`, `L0.5`, `L1.0`, `L1.0_nofail`) vs `r882`

- **Status**: `COMPLETE`
- **Verdict**: **`NO_ADMISSION`** (`admitted_arms = []`; live flywheel untouched)
- **Owner**: `rd_argon` (`[ARGON]`, `h140_rd_argon_t1800`)
- **Preregistration**: [`PREREG.md`](PREREG.md) (PR [#1244](https://github.com/neverhuman/JopeDime/pull/1244), merged at `17118d28a9a0c12ab36a93be7193a09887566f7c`)
- **Fold Manifest (`fold_seed = 140140`)**: [`fold_manifest_140140.json`](fold_manifest_140140.json) (`0b4aa9b485f6eaec27f95b5242e48fcc6b47e918420c2689b7121705959de482`)
- **Metrics & Manifest**: [`metrics.json`](metrics.json) (`96,261` bytes `< 100 KB`), [`manifest.json`](manifest.json)
- **Pinned Executor**: `/home/ubuntu/jope-rescue-20260921/h140-listwise-weight/run_h140_cpu.py` (`cells_runner_sha256 = fee6b6f56245d41274124d5875abe82f978a24341561c1837fdce27f52bdbf7e`)

---

## 1. Hypothesis

Flywheel selection utility (`fixed-dev-cost-argmin-negative-fitness/v1`) scores candidate pools by taking the single top-1 argmin predicted cost (`-fitness[argmin(cost)]` if `gate[argmin(cost)]` else `-2.0`), whereas `run_fused_closed_loop.py` trains:

$$\mathcal{L}_b(\lambda) = \frac{1}{S}\sum_{j=1}^{S}\text{SmoothL1}_{\beta=1.0}(\hat{y}_{b,j}, y_{b,j}) + \lambda \cdot \text{KL}\!\left(\text{softmax}(-y_b / 0.2)\;\|\;\text{softmax}(-\hat{y}_b / 0.2)\right)$$

with $\lambda = 0.1$ and gate failures forced to $y_{b,j} = 2.0$. `H140` tested whether up-weighting the listwise KL weight ($\lambda \in \{0.5, 1.0\}$) or restricting the listwise KL softmax strictly to gate-passing candidates (`L1.0_nofail`, $\lambda = 1.0$ over $\{j : \text{gate}_{b,j} = \text{True}\}$ when $|G_b| \ge 2$) aligns continuation training with argmin selection and lifts validation utility above the `r882` plateau (`1.1377942`).

---

## 2. Setup & Guard Verification

- **Arms (fixed a priori)**:
  - `L0.1` (`control = live recipe`, $\lambda = 0.1$, `nofail = False`)
  - `L0.5` ($\lambda = 0.5$, `nofail = False`)
  - `L1.0` ($\lambda = 1.0$, `nofail = False`)
  - `L1.0_nofail` ($\lambda = 1.0$, `nofail = True`: listwise KL over gate-passing candidates only)
- **Seeds & Continuation Budget**:
  - `3` paired seeds: `140101`, `140102`, `140103` (`4 arms x 3 seeds = 12 sequential cells`).
  - Each cell started from a fresh byte-verified copy of `/home/ubuntu/jope-data/h140/inputs/round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) and ran `40` AdamW updates (`20` rounds $\times$ `2` steps/round, `batch_size = 16`, `slate_size = 192`, `lr = 1e-4`, `weight_decay = 1e-4`, `grad_clip = 1.0`, `480` total updates across `12` cells) on `DL-Train` Folds `1..4` (`164` families / `927` contexts under `fold_seed = 140140`).
- **Step-1 `r882` Regression-First Gate (`PASS`)**:
  - Before any training cell executed, two independent copies of `r882` reproduced `/home/ubuntu/jope-data/h140/inputs/best_dev_checkpoint.json` (`30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a`) under `evaluation_identity = e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83` on **`91/91` `DL-Val` contexts** with `max_abs_error = 0.0` and mean `1.1377941898164525` (`mean_error = 0.0`), plus `0.0` two-copy prediction/loss/gradient difference.
- **Bitwise `DOPE` & Non-Adapter Integrity (`PASS`)**:
  - Across all `12/12` cells, `DOPE` SHA-256 remained `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` pre/post cell, and all `291` non-adapter `state_dict` tensors/buffers (`1,126,777` non-trainable parameters + normalizer/anchor buffers) remained bitwise identical (`bf37bcb9f65fffb40b61afb39b2fdec16239d1f234980d8e44e8e47b107324b4`, `max_abs_diff = 0.0`), while all `47` trainable adapter tensors (`29,211` parameters) updated (`adapter_l2_drift_from_r882` in `0.44235–0.44866`, `3` unique final adapter SHAs per arm).
- **Data Closure & Resource Guards (`PASS`)**:
  - `0` reads of outer-test, `DL-Holdout`, or sealed meta-test splits; `0` writes to the live flywheel directory; `DL-Val` (`91` contexts / `19` families) was never used in training or step selection.
  - CPU-only on `xbabe1` (`0.0` GPU seconds, no `qualification.lock`), `nice 19`, `4` torch compute threads, peak RSS `5,722.37 MiB < 6,144 MiB` (`6 GiB` cap), host free disk `541.8–546.6 GB >= 40 GB`, scratch deleted on exit (`0` bytes remaining).

---

## 3. Outcome & Metrics

### 3.1 Primary Endpoint — `91`-Context `DL-Val` Regression Utility (`evaluation_identity = e79a7715...`)

Reference `r882`: context-mean utility = **`1.1377942`** (`1.1377941898164525`), equal-family mean (`19` families) = **`1.1536153`**, Bradley-Terry Elo = **`1500.0`**.

| Arm | Seed `140101` | Seed `140102` | Seed `140103` | Median Gain vs `L0.1` (`95% CI`) | Median Gain vs `r882` (`95% CI`) | Seeds Beat `L0.1` / `r882` | Top-3 Family Share (vs `L0.1` / `r882`) | BT Elo (`95% CI`) | Holm Pass / Admitted |
| --- | ---: | ---: | ---: | --- | --- | :---: | :---: | --- | :---: |
| `L0.1` (ctrl) | `1.114016` | `1.082952` | `1.090807` | `0.000000` (ref) | `-0.046987` (`[-0.092860, -0.004179]`) | `—` / `0/3` | `—` / `100.0%` | `1489.01` (`[1445.31, 1552.18]`) | control |
| `L0.5` | `1.120480` | `1.079549` | `1.113333` | **`+0.006465`** (`[-0.032213, +0.058517]`) | `-0.024461` (`[-0.070265, -0.004533]`) | `2/3` / `0/3` | `99.65%` / `95.10%` | `1480.96` (`[1437.18, 1529.29]`) | `False` / **`NO`** |
| `L1.0` | `1.120354` | `1.079157` | `1.113082` | **`+0.006338`** (`[-0.032210, +0.058176]`) | `-0.024713` (`[-0.070497, -0.004966]`) | `2/3` / `0/3` | `99.49%` / `96.83%` | `1464.12` (`[1424.75, 1500.02]`) | `False` / **`NO`** |
| `L1.0_nofail` | `1.120823` | `1.079549` | `1.113082` | **`+0.006807`** (`[-0.032041, +0.058623]`) | `-0.024713` (`[-0.070270, -0.004548]`) | `2/3` / `0/3` | `99.49%` / `96.83%` | `1470.71` (`[1435.12, 1504.39]`) | `False` / **`NO`** |

### 3.2 Secondary Endpoint — Held-Out `DL-Train` Fold `0` (`41` Unseen Families / `292` Contexts) & 5-Fold Cross-Fit

Reference `r882` on held-out Fold `0`: native equal-family mean = **`1.347417`**, `unrev` (`u4`) equal-family mean = **`1.535066`**, top-1 oracle regret = **`0.247382`**, Bradley-Terry Elo = **`1500.0`**.

| Arm | Native Fam Gain vs `L0.1` (`95% CI`) | Native Fam Gain vs `r882` (`95% CI`) | `u4` (`unrev`) Fam Gain vs `L0.1` (`95% CI`) | `u4` (`unrev`) Fam Gain vs `r882` (`95% CI`) | Top-1 Oracle Regret (med seed) | Fold-0 BT Elo (`95% CI`) |
| --- | --- | --- | --- | --- | ---: | --- |
| `L0.1` (ctrl) | `0.000000` (ref) | `-0.043983` (`[-0.100758, -0.002211]`) | `0.000000` (ref) | `-0.047642` (`[-0.102864, -0.004454]`) | `0.291544` | `1476.75` (`[1439.14, 1508.77]`) |
| `L0.5` | **`+0.018400`** (`[-0.003602, +0.056152]`) | `-0.027004` (`[-0.070968, +0.001952]`) | **`+0.037333`** (`[+0.000974, +0.085661]`) | `-0.011902` (`[-0.063260, +0.037012]`) | `0.274564` | `1472.26` (`[1443.65, 1500.91]`) |
| `L1.0` | **`+0.021190`** (`[-0.001284, +0.060040]`) | `-0.024214` (`[-0.068407, +0.005164]`) | **`+0.039352`** (`[+0.003038, +0.087781]`) | `-0.009114` (`[-0.060893, +0.039928]`) | **`0.271774`** | **`1481.59`** (`[1449.68, 1515.57]`) |
| `L1.0_nofail` | **`+0.018875`** (`[-0.004021, +0.057638]`) | `-0.026529` (`[-0.070694, +0.002664]`) | **`+0.037808`** (`[+0.001215, +0.086086]`) | `-0.011427` (`[-0.062700, +0.037922]`) | `0.274090` | `1477.03` (`[1449.25, 1507.95]`) |

- **5-Fold Cross-Fit Selection Audit (`fold_seeds = 140140, 140141, 140142`)**: Across all `15/15` folds on held-out `DL-Train` families, the 4-fold in-sample selector picks `R882` (`mean_in_sample_gain_vs_r882 = 0.0`, `mean_held_out_gain_vs_r882 = 0.0`).

---

## 4. Verdict & Next Step

- **Verdict**: **`NO_ADMISSION`** (`0` arms pass the primary gates against `r882`).
- **Scientific Takeaway & Next Step**:
  1. **Relative to the live recipe (`L0.1`, $\lambda = 0.1$)**, up-weighting the listwise KL term ($\lambda \in \{0.5, 1.0\}$) or masking gate-failing candidates (`L1.0_nofail`) cuts the 40-step continuation loss vs `r882` roughly in half on both `DL-Val` (`-0.0245` vs `-0.0470`, beating `L0.1` on `2/3` seeds by `+0.0063` to `+0.0068`) and unseen `DL-Train` Fold `0` (`-0.0242` vs `-0.0440` native; `+0.0373` to `+0.0394` gain over `L0.1` on `u4` with 95% CIs strictly `> 0`).
  2. **Relative to frozen `r882`**, however, unconstrained `lr = 1e-4` AdamW continuation (`L2` drift `~0.444`) still degrades native top-1 argmin utility on all `3/3` seeds because small score perturbations flip single contexts across `-2.0` gate cliffs. Any future test of $\lambda = 1.0$ listwise weighting should be paired with a strong trust-region anchor (`lr <= 1e-5` or weight-space shrinkage toward `r882`) rather than unconstrained `lr = 1e-4` continuation.
