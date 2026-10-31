# H132 Preregistration Amendment A1 — Runner Execution Fixes & Attempt-1 / Attempt-2 Disclosure

- **Wave**: `H132` (`h132_ewc_bitfit`)
- **Preregistration PR**: [#1155](https://github.com/neverhuman/JopeDime/pull/1155) (`6306506e333bee32adcc7939842c5f38565b9d4b`, merged `2026-10-31T19:26:40Z`)
- **Preregistration Document SHA-256 (unchanged)**: `49f9cca4ad7d51e9e20605a60b76a05c2ee1d5ef2a5bca69ec129bc890c070df` (`PREREG.md`)
- **Fold Manifest SHA-256 (unchanged)**: `a2a9dd147601ecca76e247576a4575970450d1facc77c332232a5899ac54f440` (`fold_manifest_132100.json`, `fold_seed = 132100`)
- **Preregistered Runner SHA-256**: `c1bb24c94d3ad7e83d8844d6fecd9076b5b078fe58c9bd6764282a4166c09f60` (`/home/ubuntu/jope-rescue-20260921/h132-ewc-bitfit/run_h132_cpu.py`)
- **Final Executed Runner SHA-256**: `4d3e8182e674410b179e13ec2cadba21e1715c65b677053fd776a2dfe4a65673` (`/home/ubuntu/jope-rescue-20260921/h132-ewc-bitfit/run_h132_cpu.py`)

---

## 1. Summary of Protocol Deviation and Why Amendment A1 Is Filed

During live CPU execution of `H132`, three defects in `/home/ubuntu/jope-rescue-20260921/h132-ewc-bitfit/run_h132_cpu.py` were encountered and fixed:

1. **Edit 1 (`2026-10-31T19:29:44Z`, before any cell or Step 1/2 ran; `0` cells scored)**: Fixed `.npz` pool key loading (`program`, `fitness`, `gate`, `program_sha256` and `W1_CTX_NPZ` keys `cids`, `tokens_10x32`, `mask_10`, matching `run_h130_cpu.py`). Runner SHA changed from `c1bb24c94d3ad7e83d8844d6fecd9076b5b078fe58c9bd6764282a4166c09f60` to `fd7d9214ee7840d451859e883fa6ed7e7d66d1b668b217a65eeed2762b03f8f2`.
2. **Edit 2 (`2026-10-31T19:37:57Z`, after `CONTROL_seed100` and `EWC_seed100` had been scored in Attempt 1a)**: `h48.h46.checkpoint_guard(guard_a, net_a, expected=DOPE_SHA)` crashed at the pre-check of `(BITFIT, seed=100)` (`0` updates executed for `BITFIT_seed100`) because `h46.checkpoint_guard` hardcodes `trainable_params == 29211` instead of `expected_trainable` (`691` for `BITFIT`, `29211` for `CONTROL`/`EWC`). Replaced lines 1020 and 1080 with an inline check of `snap.composite_sha256 == DOPE_SHA` and `trainable_params == expected_trainable`.
3. **Edit 3 (`2026-10-31T19:55:51Z` – `19:56:31Z`, after all 9 cells had been scored across Attempt 1a + Attempt 1b)**:
   - **Defect 3a (Step 4 crash on `BITFIT` seed utility tie)**: In Step 4, `(BITFIT, seed=100)` and `(BITFIT, seed=101)` had distinct final adapter weight SHA-256 hashes (`bc1dd635df71...` vs `fe6f41499797...`) and distinct L2 drifts (`0.07621` vs `0.07255`), but selected the identical top-1 candidates on `fold 0` (`fold0_fam_u = 1.067102`, `min_pairwise_max_abs_context_utility_diff = 0.0`). Line 1202 raised `RuntimeError` instead of recording `seeds_not_identical = False` and failing the preregistered `seeds_not_identical` admission gate. Fixed Step 4 to raise `RuntimeError` only if `unique_final_adapter_shas != len(SEEDS)` and to record `seeds_not_identical = False` (`gates["seeds_not_identical"] = False` $\Rightarrow$ `NO_ADMISSION` for `BITFIT`).
   - **Defect 3b (Unseeded NumPy RNG in Step 2 Fisher estimation across process restarts)**: When Attempt 1b (`task-2381`) resumed after the `BITFIT_seed100` pre-check crash in Attempt 1a (`task-2366`), Step 2 (`fit_family_fisher.json`) re-ran and produced a slightly different fit-fold Fisher and a-priori $\lambda$ (`ewc_lambda = 770199.1457`, `mean_fit_fam_loss = 3.974131`, receipt `0b8d2046...`) than Attempt 1a (`ewc_lambda = 772275.3451`, `mean_fit_fam_loss = 3.975576`, receipt `ea301a7a...`). Root cause: `h53b.cached_training_indices(pools[c], orders[c], SLATE_SIZE)` in Step 2 draws from `np.random`, and Step 1's `build_arm_seed_batch_plan` restored `np.random.set_state(rng_state)` to the unseeded OS-startup NumPy RNG state before Step 2 ran. Consequently, in Attempt 1 (`task-2366` + `task-2381`), `EWC_seed100` was trained with $\lambda = 772275.3451$ while `EWC_seed101` and `EWC_seed102` were trained with $\lambda = 770199.1457$.
   - Seeded `np.random.seed(FOLD_SEED)`, `random.seed(FOLD_SEED)`, and `torch.manual_seed(FOLD_SEED)` (`FOLD_SEED = 132100`) around Step 2 (saving and restoring `np.random.get_state()`), deleted `cells/EWC_seed*.json`, and re-ran the 3 `EWC` cells (`Attempt 2`, `task-2411`) so all three `EWC` seeds share the single deterministic `FOLD_SEED = 132100` fit-fold Fisher ($\lambda = 773442.8941$, receipt `1cd853ef...`).
   - Resulting final runner SHA-256: `4d3e8182e674410b179e13ec2cadba21e1715c65b677053fd776a2dfe4a65673`.

**Crucial Invariant**: Both **Attempt 1** and **Attempt 2** yield **`NO_ADMISSION`** (`0` admitted arms). Both attempts are disclosed side-by-side below and in `README.md` and `metrics.json`.

---

## 2. Chronological Audit Log of Runner Edits vs Scored Cells

| Event / Step | Timestamp (UTC) | Runner SHA-256 | Cells Scored Prior to Event | What Happened |
| --- | --- | --- | --- | --- |
| Prereg Merge (`PR #1155`) | `2026-10-31T19:26:40Z` | `c1bb24c94d3ad7e83d8844d6fecd9076b5b078fe58c9bd6764282a4166c09f60` | None (`0/9`) | Preregistered runner pinned in `manifest.json`. |
| Initial Launch (`task-2357`) | `2026-10-31T19:27:47Z` | `c1bb24c94d3ad7e83d8844d6fecd9076b5b078fe58c9bd6764282a4166c09f60` | None (`0/9`) | Failed during initial pool `.npz` load (`KeyError: 'semantic is not a file in the archive'`), `0` updates executed. |
| **Edit 1** (`.npz` loader keys) | `2026-10-31T19:29:44Z` | `fd7d9214ee7840d451859e883fa6ed7e7d66d1b668b217a65eeed2762b03f8f2` | None (`0/9`) | Updated lines 678–703 to load `program`, `fitness`, `gate`, `program_sha256` + `W1_CTX_NPZ` (`cids`, `tokens_10x32`, `mask_10`). |
| **Attempt 1a** (`task-2366`) | `2026-10-31T19:29:50Z` – `19:36:14Z` | `fd7d9214ee7840d451859e883fa6ed7e7d66d1b668b217a65eeed2762b03f8f2` | Scored `CONTROL_seed100` (`1.037556`) and `EWC_seed100` (`1.081732`, $\lambda = 772275.3451$) | Step 1 PASS (`1.069456`). Step 2 Fisher ($\lambda = 772275.3451$, `ea301a7a...`). Scored `CONTROL_seed100` and `EWC_seed100`. Crashed at pre-check of `BITFIT_seed100` (`h46.checkpoint_guard` expected `29211` instead of `691`). |
| **Edit 2** (`BITFIT` trainable guard) | `2026-10-31T19:37:57Z` | `785806f0f0f8f8c4cb1bc5d8e5c3e18b7e4074d61ef70c7f9ca2fb3607311e2b` (intermediate) | `CONTROL_seed100`, `EWC_seed100` (`2/9`) | Replaced `h48.h46.checkpoint_guard` calls at lines 1020 & 1080 with `snap.composite_sha256 == DOPE_SHA` and `trainable_params == expected_trainable`. |
| **Attempt 1b** (`task-2381`) | `2026-10-31T19:38:03Z` – `19:53:25Z` | `785806f0f0f8f8c4cb1bc5d8e5c3e18b7e4074d61ef70c7f9ca2fb3607311e2b` (intermediate) | Reused `CONTROL_seed100`, `EWC_seed100`; scored remaining 7 cells (`9/9` total) | Step 2 re-ran with unseeded `np.random` ($\lambda = 770199.1457$, `0b8d2046...`). Scored `BITFIT_seed100` (`1.067102`), `CONTROL_seed101` (`1.037848`), `EWC_seed101` (`1.081571`), `BITFIT_seed101` (`1.067102`), `CONTROL_seed102` (`1.056134`), `EWC_seed102` (`1.067282`), `BITFIT_seed102` (`1.066963`). Crashed in Step 4 on `BITFIT` seed 100/101 utility tie (`min_diff=0.0`). |
| **Edit 3** (Step 2 `FOLD_SEED` + Step 4 tie gate) | `2026-10-31T19:55:51Z` – `19:56:31Z` | `4d3e8182e674410b179e13ec2cadba21e1715c65b677053fd776a2dfe4a65673` | All 9 cells of Attempt 1 (`9/9`) | Seeded `np.random.seed(FOLD_SEED)`, `random.seed(FOLD_SEED)`, `torch.manual_seed(FOLD_SEED)` around Step 2; changed Step 4 tie check to record `seeds_not_identical = False` rather than crashing when `unique_shas == 3`. |
| **Attempt 2** (`task-2411`) | `2026-10-31T19:56:37Z` – `20:05:04Z` | `4d3e8182e674410b179e13ec2cadba21e1715c65b677053fd776a2dfe4a65673` | Reused `CONTROL` (3 seeds) & `BITFIT` (3 seeds); re-ran `EWC` (3 seeds) under $\lambda = 773442.8941$ | Step 2 deterministic Fisher ($\lambda = 773442.8941$, `1cd853ef...`). Scored `EWC_seed100` (`1.081732`), `EWC_seed101` (`1.067354`), `EWC_seed102` (`1.067282`). Step 4 completed (`Verdict = NO_ADMISSION`). |

---

## 3. Side-by-Side Comparison: Attempt 1 (Unseeded Step-2 Fisher) vs Attempt 2 (Deterministic `FOLD_SEED = 132100` Step-2 Fisher)

| Metric / Cell | Attempt 1 (`task-2366` + `task-2381`, Unseeded Step-2 `np.random`) | Attempt 2 (`task-2411`, Deterministic Step-2 `FOLD_SEED = 132100`) | Material Difference / Scientific Note |
| --- | --- | --- | --- |
| Step-2 `mean_fit_family_loss_at_r882` | `3.975576` (`1a`) / `3.974131` (`1b`) | `3.975128` | `< 0.036%` variation from 32-candidate slate sampling on fit contexts (`0` assessment families touched). |
| Step-2 `fisher_diag_mean` ($\bar{F}$) | `2.542160e-05` (`1a`) / `2.548086e-05` (`1b`) | `2.538036e-05` | `< 0.40%` variation across fit-family gradient averages. |
| Step-2 `ewc_lambda_a_priori` ($\lambda$) | `772,275.3451` (`seed 100`) / `770,199.1457` (`seeds 101, 102`) | `773,442.8941` (`all 3 seeds`) | Attempt 1 used two different $\lambda$ values across seeds (`0.27%` apart); Attempt 2 uses a single deterministic $\lambda$. |
| Step-2 `fit_family_fisher.json` SHA-256 | `ea301a7a6100...` (`1a`) / `0b8d20460ddc...` (`1b`) | `1cd853ef0abc28ce533ad936664b587d4cea9d13969d8aa3c94a40ca414296a6` | Both computed strictly on `164` fit families (`956` fit contexts). |
| `EWC, seed=100` `fold0_family_mean_utility` | `1.081732` (`+0.012276` vs `r882`, `+0.044176` vs `CONTROL`) | `1.081732` (`+0.012276` vs `r882`, `+0.044176` vs `CONTROL`) | **Identical fold-0 utility** (`L2 drift`: `0.18583` vs `0.18650`; `fisher_drift`: `1.917e-09` vs `1.686e-09`; adapter SHA `e9ecc7bd3827...` vs `cc4a8dbc3a05...`). |
| `EWC, seed=101` `fold0_family_mean_utility` | `1.081571` (`+0.012115` vs `r882`, `+0.043723` vs `CONTROL`) | `1.067354` (`-0.002102` vs `r882`, `+0.029506` vs `CONTROL`) | **Differs materially by `-0.014217`**: a `+0.42%` change in $\lambda$ (`770,199.15` $\to$ `773,442.89`) flipped a discrete top-1 candidate argmax on fold 0 (`L2 drift`: `0.20481` vs `0.20367`; adapter SHA `ef8f006b4a70...` vs `8ba054a5dee8...`). |
| `EWC, seed=102` `fold0_family_mean_utility` | `1.067282` (`-0.002174` vs `r882`, `+0.011148` vs `CONTROL`) | `1.067282` (`-0.002174` vs `r882`, `+0.011148` vs `CONTROL`) | **Identical fold-0 utility** (`L2 drift`: `0.25008` vs `0.24832`; `fisher_drift`: `1.881e-09` vs `1.885e-09`; adapter SHA `09234a93414d...` vs `181ba00d0200...`). |
| `EWC` vs `r882` `median_seed_gain` | `+0.012115` | `-0.002102` | Material shift due to `seed=101` discrete top-1 flip. |
| `EWC` vs `r882` `mean_seed_gain` | `+0.007406` | `+0.002667` | Both positive in mean, neither passes all 3 seeds. |
| `EWC` vs `r882` `all_three_seeds_positive` | `False` (`2/3`: `100, 101` positive; `102` negative) | `False` (`1/3`: `100` positive; `101, 102` negative) | **Fails admission gate in both Attempt 1 and Attempt 2.** |
| **Overall Verdict (`EWC`, `BITFIT`)** | **`NO_ADMISSION`** | **`NO_ADMISSION`** | **Identical `NO_ADMISSION` verdict.** |

---

## 4. Exact Diffs Applied to `run_h132_cpu.py`

### Diff A — Edit 2 (`BITFIT` trainable parameter guard, lines 1013–1105)

```diff
@@ -1018,7 +1018,19 @@
                 fisher_list = [fisher_by_name[name] for name, _ in all_adapter_params]
                 all_adapter_tensors = [p for _, p in all_adapter_params]
 
-                dope_pre = h48.h46.checkpoint_guard(guard_a, net_a, expected=DOPE_SHA)
+                snap_pre = guard_a.snapshot_dope_weights(net_a)
+                if snap_pre.composite_sha256 != DOPE_SHA:
+                    raise RuntimeError(f"DOPE guard pre failed: {snap_pre.composite_sha256}")
+                counts_pre = guard_a.get_trainable_param_counts_by_submodule(net_a)
+                if (
+                    counts_pre.get("__global_total__", {}).get("trainable_params") != expected_trainable
+                    or counts_pre.get("adapter_10tok", {}).get("trainable_params") != expected_trainable
+                ):
+                    raise RuntimeError(
+                        f"trainable parameter guard pre failed for {arm} (expected {expected_trainable}): {counts_pre}"
+                    )
+                dope_pre = snap_pre.composite_sha256
                 init_sha = h53b.adapter_sha(net_a)
                 init_full_vec = torch.cat([p.detach().reshape(-1).clone() for _, p in all_adapter_params], dim=0)
@@ -1078,7 +1090,19 @@
                     pre_clip_norms.append(pre_norm)
                     post_clip_norms.append(post_norm)
 
-                dope_post = h48.h46.checkpoint_guard(guard_a, net_a, expected=DOPE_SHA)
+                snap_post = guard_a.snapshot_dope_weights(net_a)
+                if snap_post.composite_sha256 != DOPE_SHA:
+                    raise RuntimeError(f"DOPE guard post failed: {snap_post.composite_sha256}")
+                counts_post = guard_a.get_trainable_param_counts_by_submodule(net_a)
+                if (
+                    counts_post.get("__global_total__", {}).get("trainable_params") != expected_trainable
+                    or counts_post.get("adapter_10tok", {}).get("trainable_params") != expected_trainable
+                ):
+                    raise RuntimeError(
+                        f"trainable parameter guard post failed for {arm} (expected {expected_trainable}): {counts_post}"
+                    )
+                dope_post = snap_post.composite_sha256
```

### Diff B — Edit 3 (Deterministic Step-2 `FOLD_SEED` seeding + Step-4 `seeds_not_identical` gate handling)

```diff
@@ -842,6 +842,11 @@
         fit_family_losses: dict[str, float] = {}
         fit_context_losses: dict[str, float] = {}
+        rng_state_fisher = np.random.get_state()
+        np.random.seed(FOLD_SEED)
+        random.seed(FOLD_SEED)
+        torch.manual_seed(FOLD_SEED)
+        try:
             for fam in sorted(fit_fams):
                 ...
+        finally:
+            np.random.set_state(rng_state_fisher)
@@ -1206,9 +1213,10 @@
             min_pair_diff = float(min(pairwise_ctx_diffs))
-            seeds_distinct = (len(set(shas)) == len(SEEDS)) and (min_pair_diff > 0.0)
-            if not seeds_distinct:
-                raise RuntimeError(f"Seed independence check failed for {arm}: shas={shas}, min_diff={min_pair_diff}")
+            unique_shas = len(set(shas))
+            if unique_shas != len(SEEDS):
+                raise RuntimeError(f"Duplicate final adapter SHA across seeds for {arm}: shas={shas}")
+            seeds_distinct = (unique_shas == len(SEEDS)) and (min_pair_diff > 0.0)
             seed_independence[arm] = {
-                "unique_final_adapter_shas": len(set(shas)),
+                "unique_final_adapter_shas": unique_shas,
                 "min_pairwise_max_abs_context_utility_diff": min_pair_diff,
                 "max_pairwise_max_abs_context_utility_diff": float(max(pairwise_ctx_diffs)),
                 "seeds_not_identical": seeds_distinct,
```

---

## 5. Zero-Leakage Confirmation for $\lambda$ and Fit-Fold Fisher

- In both Attempt 1 and Attempt 2, `fit_cids_by_fam` is constructed exclusively from `fit_cids` (`folds 1..4`: `164` families, `956` contexts), and line 839 enforces:
  `if set(fit_cids_by_fam.keys()) != fit_fams or (set(fit_cids_by_fam.keys()) & assess_fams): raise RuntimeError(...)`.
- `assessment_families_touched = 0` and `assessment_contexts_touched = 0` in Step 2.
- The reference drift $D_{\text{ref}} = 0.45$ was fixed in `PREREG.md` (`49f9cca4...`) prior to execution, and $\lambda = \bar{L}_{\text{fit}}(\theta_0) / (D_{\text{ref}}^2 \cdot \bar{F})$ was computed in closed form strictly from fit-fold quantities at $\theta_0 = r882$. No assessment-fold (`fold 0`), `DL-Val`, outer-test, or sealed meta-test value ever entered $F$ or $\lambda$.
