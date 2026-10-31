# H132 — Fisher-Anchored (`EWC`) and Bias/Output-Restricted (`BITFIT`) Adapter Fine-Tuning vs `CONTROL` and Unchanged `r882`

- **Status**: `COMPLETE — NO_ADMISSION` (`0` arms admitted; unchanged `r882` remains the active serving adapter)
- **Preregistration**: [`PREREG.md`](PREREG.md) (`sha256 = 49f9cca4ad7d51e9e20605a60b76a05c2ee1d5ef2a5bca69ec129bc890c070df`, merged in PR [#1155](https://github.com/neverhuman/JopeDime/pull/1155) at `6306506e333bee32adcc7939842c5f38565b9d4b`)
- **Amendment A1 (Disclosed Runner Fixes & Attempt-1 / Attempt-2 Side-by-Side)**: [`PREREG_AMENDMENT_A1.md`](PREREG_AMENDMENT_A1.md) (`sha256 = 715548147a8ee7f1157c71ff5e13f2e838ac4a9524fc97ce31807805c3053826` — **runner SHA deviates from prereg; Amendment A1 included**)
- **Fold Manifest**: [`fold_manifest_132100.json`](fold_manifest_132100.json) (`fold_seed = 132100`, `sha256 = a2a9dd147601ecca76e247576a4575970450d1facc77c332232a5899ac54f440`)
- **Metrics Summary**: [`metrics.json`](metrics.json) (`42,325` bytes `< 100 KB`)

---

## 1. Hypothesis

Across recent `DL-Train` fine-tuning waves (`H126`, `H128`, `H130`), standard 40-step full-adapter (`29,211`-parameter) fine-tuning (`CONTROL`) from `r882` drifted by $\|\Delta \theta\|_2 \approx 0.43\text{--}0.45$ and degraded unseen-family utility on the held-out assessment fold (`fold 0`) relative to unchanged `r882`.

**`H132` tests whether constraining parameter-space movement around $\theta_0 = r882$ via:**
1. **`EWC` (Fit-Fold Empirical Diagonal Fisher Anchoring)**: adding $\lambda \sum_{i=1}^{29,211} F_i (\theta_i - \theta_{0,i})^2$ where $F_i = \frac{1}{164}\sum_{f \in \mathcal{F}_{\text{fit}}} (\nabla_{\theta_i} L_f(\theta_0))^2$ is estimated strictly on the `164` fit families (`0` assessment families touched) and $\lambda = \bar{L}_{\text{fit}}(\theta_0) / (0.45^2 \cdot \bar{F})$ is fixed a priori in closed form; or
2. **`BITFIT` (Bias + Output-Layer Fine-Tuning)**: freezing the `26` interior projection/attention weight matrices (`28,520` parameters, `97.63%` of the adapter) at `r882` and training only the `21` bias + final scalar output-layer tensors (`691` parameters, `2.37%` of the adapter)

eliminates the catastrophic family harm of `CONTROL` (`48.78%` harmed families on `fold 0`) and yields a strict paired family gain (`> +0.001`) over unchanged `r882` across all three paired seeds (`100, 101, 102`) with `Holm LCB95 > 0.0` and `<= 10%` harmed families.

---

## 2. Setup

- **Starting Checkpoint**: Fresh hash-verified copies of `r882` (`sha256 = 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) per cell; `DOPE` backbone frozen (`sha256 = c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`).
- **Partition (`fold_seed = 132100`)**: 5-fold family-grouped split of `1,219` `DL-Train` contexts (`205` families):
  - **Assessment fold (`fold 0`)**: `41` families, `263` contexts (never read during Fisher estimation, $\lambda$ calibration, or gradient updates).
  - **Fit folds (`folds 1..4`)**: `164` families, `956` contexts.
  - **Holdout isolation**: `0` `DL-Val`, outer-test, sealed, or meta-test reads.
- **Design**: `3` arms (`CONTROL`, `EWC`, `BITFIT`) $\times$ `3` paired seeds (`100, 101, 102`) = `9` cells $\times$ `40` AdamW updates/cell (`lr = 1e-4`, `weight_decay = 1e-4`, `batch_size = 8`, `slate_size = 32`, `grad_clip = 1.0`) = `360` updates per full wave.

---

## 3. Protocol Deviation Disclosure: Attempt-1 vs Attempt-2 (`EWC` Rerun)

Per [`PREREG_AMENDMENT_A1.md`](PREREG_AMENDMENT_A1.md), three defects in `/home/ubuntu/jope-rescue-20260921/h132-ewc-bitfit/run_h132_cpu.py` (`prereg_runner_sha256 = c1bb24c94d3ad7e83d8844d6fecd9076b5b078fe58c9bd6764282a4166c09f60` $\to$ `final_runner_sha256 = 4d3e8182e674410b179e13ec2cadba21e1715c65b677053fd776a2dfe4a65673`) were fixed during execution:

1. **Before any cell ran (`Edit 1`, `19:29:44Z`)**: fixed `.npz` pool key loading (`0` updates executed prior).
2. **After `CONTROL_seed100` (`1.037556`) and `EWC_seed100` (`1.081732`) were scored (`Edit 2`, `19:37:57Z`)**: replaced `h48.h46.checkpoint_guard` (which hardcoded `29,211` trainable params and crashed at the pre-check of `BITFIT_seed100`) with an inline check of `DOPE_SHA` and `expected_trainable` (`691` for `BITFIT`, `29,211` for `CONTROL`/`EWC`).
3. **After all 9 cells of Attempt-1 were scored (`Edit 3`, `19:55:51Z` – `19:56:31Z`)**:
   - In **Attempt-1** (`task-2366` + `task-2381`), Step 2 (`fit_family_fisher.json`) called `h53b.cached_training_indices` without first seeding `np.random` (Step 1's `build_arm_seed_batch_plan` had restored `np.random` to the unseeded OS-startup state). When `task-2381` resumed after the `BITFIT_seed100` pre-check crash, Step 2 re-ran with a different unseeded NumPy RNG state (`ewc_lambda = 770,199.1457` vs `772,275.3451` in `task-2366`). Thus, in Attempt-1, `EWC_seed100` was trained with $\lambda = 772,275.3451$ while `EWC_seed101` and `EWC_seed102` were trained with $\lambda = 770,199.1457$.
   - In **Edit 3**, Step 2 was seeded deterministically with `np.random.seed(FOLD_SEED)`, `random.seed(FOLD_SEED)`, and `torch.manual_seed(FOLD_SEED)` (`FOLD_SEED = 132100`, yielding $\lambda = 773,442.8941$), Step 4 was fixed to record `seeds_not_identical = False` instead of crashing on `BITFIT`'s seed `100/101` utility tie, and the 3 `EWC` cells were re-run as **Attempt-2** (`task-2411`).

### Side-by-Side Comparison: `EWC` Attempt-1 (Deleted/Inconsistent-$\lambda$) vs Attempt-2 (Deterministic-$\lambda$ Rerun)

| Cell / Statistic | Attempt-1 (`task-2366` + `task-2381`, Unseeded Step-2 `np.random`) | Attempt-2 (`task-2411`, Deterministic Step-2 `FOLD_SEED = 132100`) | Comparison & Materiality |
| --- | --- | --- | --- |
| Step-2 $\bar{L}_{\text{fit}}(\theta_0)$ / $\bar{F}$ | `3.975576 / 2.542160e-05` (`1a`)<br>`3.974131 / 2.548086e-05` (`1b`) | `3.975128 / 2.538036e-05` | Fit-fold only (`0` assessment families touched in all runs). |
| Step-2 $\lambda$ (`ewc_lambda_a_priori`) | `772,275.3451` (`seed 100`)<br>`770,199.1457` (`seeds 101, 102`) | `773,442.8941` (`all 3 seeds`) | Attempt-1 mixed two $\lambda$ values across seeds (`0.27%` apart); Attempt-2 uses one deterministic $\lambda$. |
| `EWC, seed=100` (`fold 0` family mean / $\Delta$ vs `r882`) | `1.081732` (`+0.012276` vs `r882`, `+0.044176` vs `CONTROL`, `drift=0.18583`, `f_drift=1.917e-09`, deleted receipt `304e9e84...`) | `1.081732` (`+0.012276` vs `r882`, `+0.044176` vs `CONTROL`, `drift=0.18650`, `f_drift=1.686e-09`, receipt `dc8771ce...`) | **Identical fold-0 utility** (`1.081732`). |
| `EWC, seed=101` (`fold 0` family mean / $\Delta$ vs `r882`) | `1.081571` (`+0.012115` vs `r882`, `+0.043723` vs `CONTROL`, `drift=0.20481`, `f_drift=1.769e-09`) | `1.067354` (`-0.002102` vs `r882`, `+0.029506` vs `CONTROL`, `drift=0.20367`, `f_drift=1.621e-09`, receipt `8d195d3e...`) | **Differs materially (`-0.014217`)**: a `+0.42%` shift in $\lambda$ flipped a discrete top-1 candidate argmax on fold 0. |
| `EWC, seed=102` (`fold 0` family mean / $\Delta$ vs `r882`) | `1.067282` (`-0.002174` vs `r882`, `+0.011148` vs `CONTROL`, `drift=0.25008`, `f_drift=1.881e-09`) | `1.067282` (`-0.002174` vs `r882`, `+0.011148` vs `CONTROL`, `drift=0.24832`, `f_drift=1.885e-09`, receipt `0c47cf52...`) | **Identical fold-0 utility** (`1.067282`). |
| `EWC` vs `r882` `median_seed_gain` / `mean_seed_gain` | `+0.012115` / `+0.007406` | `-0.002102` / `+0.002667` | Material shift in median (`+0.012115` $\to$ `-0.002102`) due to `seed=101` argmax flip. |
| `EWC` vs `r882` `all_three_seeds_positive` | `False` (`2/3` positive; `seed 102 = -0.002174`) | `False` (`1/3` positive; `seeds 101, 102 < 0`) | **Fails admission gate in both Attempt-1 and Attempt-2.** |
| **Verdict** | **`NO_ADMISSION`** | **`NO_ADMISSION`** | **Unchanged (`NO_ADMISSION` under both attempts).** |

---

## 4. Outcome & Key Metrics

### Step 1 — Unchanged-`r882` Regression-First Gate (`PASS`)
- Two independent `r882` loads evaluated on `fold 0` (`41` families, `263` contexts) and `CONTROL` batch-0 gradients:
  - `two_copy_fold0_pred_max_abs_diff = 0.0`
  - `two_copy_fold0_utility_max_abs_diff = 0.0`
  - `two_copy_control_loss_abs_diff = 0.0`
  - `two_copy_control_grad_max_abs_diff = 0.0`
  - Unchanged `r882` `fold 0` family mean utility (`unrev`): **`1.069456`** (context mean: `1.033949`, native family mean: `1.092172`).

### Per-Cell Summary (Attempt-2 Canonical Run; Attempt-1 `EWC` Shown in Brackets Where Different)

| Arm | Seed | Trainable Params (Tensors) | Frozen Adapter Max Diff | L2 Drift $\|\Delta\theta\|_2$ | Fisher Drift $\sum_i F_i \Delta\theta_i^2$ | `fold 0` Family Mean Utility | $\Delta$ vs `r882` (`1.069456`) | $\Delta$ vs `CONTROL` | Family W/T/L vs `r882` (`41` fams) | Harmed Families vs `r882` (`< -0.01`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CONTROL` | `100` | `29,211 (47)` | `0.0` | `0.44364` | `8.929e-06` | `1.037556` | `-0.031900` | `0.000000` | `8 / 8 / 25` | `21 / 41 (51.22%)` |
| `CONTROL` | `101` | `29,211 (47)` | `0.0` | `0.43069` | `9.848e-06` | `1.037848` | `-0.031608` | `0.000000` | `10 / 6 / 25` | `21 / 41 (51.22%)` |
| `CONTROL` | `102` | `29,211 (47)` | `0.0` | `0.44529` | `1.002e-05` | `1.056134` | `-0.013322` | `0.000000` | `12 / 7 / 22` | `18 / 41 (43.90%)` |
| `EWC` | `100` | `29,211 (47)` | `0.0` | `0.18650` (`[0.18583]`) | `1.686e-09` (`[1.917e-09]`) | `1.081732` (`[1.081732]`) | **`+0.012276`** (`[+0.012276]`) | **`+0.044176`** | `4 / 34 / 3` | `1 / 41 (2.44%)` |
| `EWC` | `101` | `29,211 (47)` | `0.0` | `0.20367` (`[0.20481]`) | `1.621e-09` (`[1.769e-09]`) | `1.067354` (`[1.081571]`) | `-0.002102` (`[+0.012115]`) | **`+0.029506`** (`[+0.043723]`) | `3 / 34 / 4` | `1 / 41 (2.44%)` |
| `EWC` | `102` | `29,211 (47)` | `0.0` | `0.24832` (`[0.25008]`) | `1.885e-09` (`[1.881e-09]`) | `1.067282` (`[1.067282]`) | `-0.002174` (`[-0.002174]`) | **`+0.011148`** | `2 / 35 / 4` | `1 / 41 (2.44%)` |
| `BITFIT` | `100` | `691 (21)` | `0.0` (`26` tensors) | `0.07621` | `6.904e-07` | `1.067102` | `-0.002353` | **`+0.029547`** | `3 / 34 / 4` | `2 / 41 (4.88%)` |
| `BITFIT` | `101` | `691 (21)` | `0.0` (`26` tensors) | `0.07255` | `6.545e-07` | `1.067102` | `-0.002353` | **`+0.029255`** | `3 / 34 / 4` | `2 / 41 (4.88%)` |
| `BITFIT` | `102` | `691 (21)` | `0.0` (`26` tensors) | `0.07623` | `7.150e-07` | `1.066963` | `-0.002493` | **`+0.010829`** | `3 / 33 / 5` | `2 / 41 (4.88%)` |

### Primary Endpoint — Treatment Arms vs Unchanged `r882` (`10,000` Paired Family-Cluster Bootstrap Draws, `bootstrap_seed = 132100`, 2-Arm Holm Step-Down at $\alpha = 0.05$)

| Arm | Holm Rank ($\alpha_k$) | Median Seed Gain vs `r882` | Mean Seed Gain vs `r882` | 95% Two-Sided Bootstrap CI | Holm-Adjusted `LCB95` | One-Sided $p$-value | Pooled Family Harm (`< -0.01`) | Pooled Family W/T/L (`123` pairs) | Paired Elo vs `r882=1500` (95% CI) | Bradley-Terry Elo (`r882=1500`) | All 3 Seeds $> 0$? | Seeds Independent? | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `EWC` (Attempt-2) | `1` (`0.025`) | `-0.002102` | `+0.002667` | `[-0.005791, +0.014229]` | `-0.005791` | `0.3520` | **`2.44%`** (`1/41`) | `9 / 103 / 11` | `1494.35 [1451.67, 1534.00]` | `1499.51` | `False` (`1/3`) | `True` (`min_diff=0.0148`) | **`NO_ADMISSION`** |
| `EWC` (Attempt-1 Disclosed) | — | `+0.012115` | `+0.007406` | — | — | — | — | — | — | — | `False` (`2/3`: `seed 102 = -0.002174`) | `True` | **`NO_ADMISSION`** |
| `BITFIT` | `2` (`0.050`) | `-0.002353` | `-0.002400` | `[-0.006854, +0.000240]` | `-0.006212` | `0.9290` | **`4.88%`** (`2/41`) | `9 / 101 / 13` | `1488.70 [1445.90, 1534.00]` | `1498.26` | `False` (`0/3`) | **`False`** (`seeds 100/101 tie`) | **`NO_ADMISSION`** |
| `CONTROL` (diagnostic vs `r882`) | — | `-0.031608` | `-0.025610` | `[-0.086549, +0.049565]` | `-0.077654` (`LCB95`) | `0.7882` | **`48.78%`** (`20/41`) | `30 / 21 / 72` | `1376.40 [1275.49, 1463.14]` | `1463.88` | `False` (`0/3`) | `True` (`min_diff=2.3369`) | — |

### Secondary Endpoint — Treatment Arms vs `CONTROL` and `EWC` vs `BITFIT`

| Comparison | Median Seed Gain | Mean Seed Gain | 95% Two-Sided Bootstrap CI | One-Sided `LCB95` | One-Sided $p$-value | Pooled Family Harm (`< -0.01`) | Pooled Family W/T/L (`123` pairs) | Paired Elo (95% CI) | Bradley-Terry Elo (`CONTROL=1500`) | All 3 Seeds $> 0$? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `EWC` vs `CONTROL` | **`+0.029506`** | **`+0.028277`** | `[-0.046892, +0.089982]` | `-0.032907` | `0.1968` | `19.51%` | `71 / 21 / 31` (`66.26%` win) | **`1617.24 [1534.00, 1716.27]`** | `1535.63` | **`True` (`3/3`)** |
| `BITFIT` vs `CONTROL` | **`+0.029255`** | **`+0.023210`** | `[-0.051392, +0.084395]` | `-0.037857` | `0.2329` | `24.39%` | `67 / 21 / 35` (`63.01%` win) | **`1592.52 [1505.65, 1688.97]`** | `1534.38` | **`True` (`3/3`)** |
| `EWC` vs `BITFIT` | **`+0.000319`** | **`+0.005067`** | `[-0.000117, +0.014960]` | `-0.000079` | `0.1131` | **`0.00%`** (`0/41`) | `6 / 115 / 2` (`51.63%` win) | **`1511.30 [1491.52, 1536.86]`** | — | **`True` (`3/3`)** |

---

## 5. Verdict & Mechanistic Takeaways

- **Verdict**: **`NO_ADMISSION`**. Neither `EWC` nor `BITFIT` satisfies the preregistered primary admission gates against unchanged `r882`. Do **not** write `H132_WIN.md` and do **not** touch the live flywheel.
- **What Worked Mechanistically**:
  1. **Massive Reduction in Harmed Families Relative to `CONTROL`**: Unconstrained 40-step `CONTROL` harmed **`48.78%`** (`20/41`) of held-out `fold 0` families relative to `r882` (per-seed `51.22%`, `51.22%`, `43.90%`, median seed delta `-0.031608`, paired Elo `1376.40`). Fit-fold Fisher anchoring (`EWC`) reduced the harmed family fraction by **20x** to **`2.44%`** (`1/41` families, passing the `<= 10%` harm gate on every seed) and cut Fisher-weighted parameter drift $\sum_i F_i \Delta\theta_i^2$ by **5,500x** (`1.62e-09 – 1.88e-09` vs `8.93e-06 – 1.00e-05`). `BITFIT` reduced harmed families by **10x** to **`4.88%`** (`2/41` families).
  2. **Consistent 3/3 Seed Dominance Over `CONTROL`**: Both `EWC` (`+0.044176`, `+0.029506`, `+0.011148`; paired Elo **`1617.24 [1534.00, 1716.27]`**) and `BITFIT` (`+0.029547`, `+0.029255`, `+0.010829`; paired Elo **`1592.52 [1505.65, 1688.97]`**) beat `CONTROL` on all 3 seeds, and `EWC` beat `BITFIT` on all 3 seeds with `0.00%` harmed families.
  3. **Why Neither Beats Unchanged `r882` Across All 3 Seeds**: `EWC` achieved a genuine `+0.012276` gain over `r882` on `seed=100` (`1.081732` vs `1.069456`, paired Elo `1508.48`), but on `seed=101` (`-0.002102` in Attempt-2 vs `+0.012115` in Attempt-1) and `seed=102` (`-0.002174` in both attempts) it suffered `1` net family loss (`34–35` ties out of `41` families). Because `r882` was already trained on the full `DL-Train` corpus, 40 gradient steps on a `4/5` subset (`956` fit contexts) under the standard smooth-L1 + KL surrogate loss can only prevent `CONTROL`'s forgetting when heavily Fisher-anchored; where it moves, discrete top-1 flips on `fold 0` remain sensitive to `0.42%` perturbations in $\lambda$.
- **Next Step**: Any future fine-tuning wave from `r882` should keep fit-fold diagonal Fisher anchoring (`EWC`) as the baseline drift stabilizer (since it reduces `CONTROL`'s `48.78%` family harm to `2.44%`), while pairing it with a margin/contrastive ranking signal around `r882`'s current top-1 selections rather than the global 32-slate surrogate loss.

---

## 6. Red-Team Audit

1. **Leakage (`PASS`)**:
   - `fold 0` (`41` assessment families, `263` contexts) was strictly excluded from `fit_cids` (`folds 1..4`: `164` families, `956` contexts).
   - In Step 2 (`fit_family_fisher.json`), line 839 explicitly asserts `set(fit_cids_by_fam.keys()) == fit_fams` and `(set(fit_cids_by_fam.keys()) & assess_fams) == set()`. `assessment_families_touched = 0` and `assessment_contexts_touched = 0`.
   - $\lambda = \bar{L}_{\text{fit}}(\theta_0) / (0.45^2 \cdot \bar{F})$ was computed in closed form strictly from fit-fold quantities at $\theta_0 = r882$. Zero assessment-fold, `DL-Val`, outer-test, sealed, or meta-test values influenced $F$ or $\lambda$.
2. **Protocol Deviation & Attempt-1 / Attempt-2 Disclosure (`DISCLOSED VIA AMENDMENT A1`)**:
   - All three runner edits (`c1bb24c9...` $\to$ `fd7d9214...` $\to$ `785806f0...` $\to$ `4d3e8182...`), their exact diffs, timestamps relative to scored cells, and the deleted Attempt-1 `EWC_seed100` (`1.081732`, $\lambda = 772,275.3451$) + Attempt-1 `EWC_seed101` (`1.081571`, $\lambda = 770,199.1457$) + Attempt-1 `EWC_seed102` (`1.067282`, $\lambda = 770,199.1457$) results are fully disclosed in [`PREREG_AMENDMENT_A1.md`](PREREG_AMENDMENT_A1.md), Section 3 above, and [`metrics.json`](metrics.json) (`attempt_1_disclosed_ewc`).
   - Both Attempt-1 (`2/3` seeds positive vs `r882`) and Attempt-2 (`1/3` seeds positive vs `r882`) result in **`NO_ADMISSION`**.
3. **Seed Noise, Seed Independence & Tie Acceptance (`ENFORCED — BITFIT FAILS SEED INDEPENDENCE`)**:
   - All 9 cells produced unique final adapter SHA-256 hashes (`3` unique SHAs per arm) and non-zero L2 drift from `r882`.
   - For `CONTROL` and `EWC`, pairwise max-abs context utility differences across seeds were strictly positive (`min_diff = 2.3369` for `CONTROL`, `0.0148` for `EWC` $\Rightarrow$ `seeds_not_identical = True`).
   - For `BITFIT`, `seed=100` (`bc1dd635df71...`, L2 drift `0.07621`) and `seed=101` (`fe6f41499797...`, L2 drift `0.07255`) selected the exact same top-1 candidates across all `263` fold-0 contexts (`fold0_fam_u = 1.067102`, `min_pairwise_max_abs_context_utility_diff = 0.0`). Per the preregistered tie rule, `seeds_not_identical = False` for `BITFIT`, failing admission regardless of median.
4. **Metric Direction & Elo-Only Reporting (`PASS`)**:
   - Higher `unrev` normalized Top-1 utility and higher Elo are better.
   - Both family-paired Elo (with 95% bootstrap CI) and 4-player Bradley-Terry Elo anchored at `r882 = 1500` and `CONTROL = 1500` are reported; no AUC is used anywhere.
5. **Cherry-Picking & Multiple-Testing Control (`PASS`)**:
   - All `9` cells (`3` arms $\times$ `3` seeds) are reported with zero exclusions.
   - Two-arm Holm-Bonferroni step-down ($\alpha_1 = 0.025$ for rank 1 `EWC`, $\alpha_2 = 0.050$ for rank 2 `BITFIT`) was applied over `10,000` paired family-cluster bootstrap draws (`bootstrap_seed = 132100`).
6. **DOPE Freeze & `BITFIT` Parameter Mask Verification (`PASS`)**:
   - Before and after every cell (`9/9`), `DOPE` composite SHA-256 equaled `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and `guard.assert_optimizer_excludes_dope(optimizer, net)` passed.
   - For all 3 `BITFIT` cells, `trainable_parameter_count = 691` (`21` tensors), `frozen_adapter_tensor_count = 26` (`28,520` parameters), and `frozen_adapter_max_abs_diff = 0.0`.
