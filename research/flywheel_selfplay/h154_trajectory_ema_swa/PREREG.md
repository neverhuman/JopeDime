# H154 — Trajectory EMA and Uniform SWA Weight Averaging with Train-Only Stat Recalibration on the r882 Continuation

**Owner:** `gpu_fused_agy` `[FUSED-AGY]`. Authority: `~/nudges/gpu_fused_agy_lift_t2215.md` (`ENG:watcher t2215`). Claim: `~/jope_plan/ns/receipts/H154_claim.txt`.
Cross-reviewed read-only before any scored candidate execution; squash-merged centrally (`no stacking`).

**Status:** Preregistration committed before any optimizer update or candidate evaluation in H154 (`0` updates, `0` scored contexts in constructor smoke). Exploratory research on repeatedly exposed `DL-Val` (`91` contexts, `19` families) and `DL-Train` (`1,219` contexts, `205` families); not sealed-test qualification or serving promotion.

## 1. Question, Motivation, and Non-Overlap with H138 and H152 (`hwave_t2220_gpu_fused_agy.md`)

Across recent `r882` continuation waves (`H140`, `H144`, `H146`, `H151`), raw 40-step AdamW updates (`CTRL`, `lr=1e-4, wd=1e-4`) accumulate `~0.44` L2 adapter drift and regress on `DL-Val` (`-0.0237` family mean in `H146`), while variance-reducing trajectory mechanisms (`H97` `EMA(0.9)` and `H146` Lookahead) consistently outperform raw `CTRL` on `3/3` seeds.

**Explicit comparison against `H138` (`research/flywheel_selfplay/h138_swa_ema_soup`, `NO_ADVANCE`, merged) and `H152` (`research/flywheel_selfplay/h152_ckpt_soup`, merged `38897495460e8d91a25738a83d907409739fab05`), and what is new in `H154`:**
1. **`H138` and `H152` averaged only the 4 sparse historical checkpoints (`r782, r785, r859, r882`) at zero updates without activation-stat recalibration:** In `H138`, uniform/recency-weighted soups (`SOUP_ALL4`, `SOUP_LAST3`, `SOUP_782_882`, `EMA_05`, `EMA_07`) over those 4 saved checkpoints were tie-bound or negative on `DL-Val` (`-0.00534` to `-0.00148` family mean, `NO_ADVANCE`), and `H152` re-tested zero-update discrete blends of those same 4 saved checkpoints. Meanwhile, `H97` tested online trajectory `EMA` only at a fast decay (`0.9`, effective window `~10` steps, still drifting `0.346` L2 from `r882`) without output-scale recalibration.
2. **Infeasibility of "last $K \in \{5, 10, 20\}$ promoted adapters" and explicit disclosure of unadmitted trajectory snapshots:** Inspection of `adapter_admissions.jsonl` in `run-20261030` shows that only `4` rounds (`782, 785, 859, 882`; `5` including `r774`) were ever promoted across the entire flywheel run, making "last $K \in \{5, 10, 20\}$ promoted checkpoints" infeasible for $K > 5$ and redundant with `H138`/`H152` for $K \le 4$. In `H154`, `SWA_K5`, `SWA_K10`, and `SWA_K20` average the **last $K \in \{5, 10, 20\}$ end-of-round trajectory snapshots (steps `2, 4, ..., 40`) along the 20-round `r882` continuation**; we explicitly disclose that these intermediate round iterates are **unadmitted continuation trajectory snapshots (never promoted by the flywheel gate)**.
3. **What is new in `H154` — high-decay trajectory EMA (`0.99 / 0.995 / 0.999`) + trajectory-round SWA (`K in {5, 10, 20}`) + train-only RMS activation-stat recalibration:** Weight averaging shrinks the variance/RMS dispersion of the adapter's two centered additive residual branches (`mm_prior_adj` and `traj_attn_adj`), which neither `H97`, `H138`, nor `H152` corrected. `H154` tests high-decay online trajectory `EMA` ($\gamma \in \{0.99, 0.995, 0.999\}$ anchored at $\theta_{\text{r882}}$) and trajectory-round `SWA` ($K \in \{5, 10, 20\}$) paired with **zero-gradient, in-fold `DL-Train` (`Folds 1..4`) RMS output-scale recalibration (`[0.85, 1.15]`)** on the 4 trainable output-head tensors (`adapter_10tok.mm_prior_out.2.{weight,bias}` and `adapter_10tok.traj_delta_head.2.{weight,bias}`), while keeping `DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and all non-trainable tensors (`222` non-trainable `state_dict` tensors in `FusedJopeNetwork`, and all `291` non-adapter parameters/buffers across the full checkpoint) 100% bitwise identical to `r882`, verified per cell.

**Hypothesis (`H154`):** Averaging the `47` trainable non-anchor `adapter_10tok` tensors (`29,211` parameters) along the 20-round (`40`-step) `r882` continuation trajectory using high-decay exponential moving averages ($\gamma \in \{0.99, 0.995, 0.999\}$) or uniform Stochastic Weight Averaging (SWA) over the last $K \in \{5, 10, 20\}$ trajectory round snapshots—paired with zero-gradient, in-fold `DL-Train` output-scale recalibration to correct weight-averaging variance shrinkage—lands in a flatter basin than raw `CTRL` and `r882`.

**Code provenance:**
- Reuses the merged `H151` harness (`research/flywheel_selfplay/h151_lookahead_drift_control/{runner.py,runtime.py,science.py,monitor.py,test_h151.py}` at `38897495460e8d91a25738a83d907409739fab05`) without modifying `H151` files.
- Replaces Lookahead with online `float64` trajectory accumulators (`TrajectoryEMAState`, `TrajectorySWAState`), zero-gradient in-fold `DL-Train` RMS variance recalibration (`compute_recal_rms`, `apply_stat_recalibration`), a 7-arm dose manipulation check (`dose_check`), unified 7-arm Holm step-down vs `r882` (`HOLM_M = 7`) and 6-arm Holm step-down vs `CTRL` (`HOLM_M_CTRL = 6`), squash-compatible source blob binding (`bind_source`), and strict hard-stop (never pause) resource guards across `monitor.py` and `runtime.py`.

## 2. Fixed Arms, Seeds, Folds, and Train-Only Stat Recalibration

Each of the 3 seeds (`SEEDS = (154101, 154102, 154103)`) starts from a read-only SHA-verified copy of `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) staged at `/home/ubuntu/jope-data/h154/inputs/round-882.pt` and runs one `40`-step (`20` rounds $\times 2$ retained steps from the H140/H146/H151 8-draw paired sampler) AdamW trajectory (`lr = 1e-4`, `weight_decay = 1e-4`, batch `16`, slate `192` = `48` fitness-ranked + `144` random, grad norm clip `1.0`, SmoothL1 $\beta=1.0$ + `0.1` listwise KL $\tau=0.2$, eval mode throughout, only the `47` non-anchor `adapter_10tok` tensors trainable). All 7 arms for a given seed are extracted from that single 40-step trajectory (`3 seeds x 40 steps = 120` total optimizer updates across the entire wave):

| Arm | Trajectory Weight Rule (computed in `float64`, cast to `float32`) | Train-Only Stat Recalibration (`Folds 1..4`) |
| --- | --- | --- |
| `CTRL` | Raw step-40 adapter weights $\theta^{(40)}$ | `False` |
| `EMA_099` | $\theta_{\text{EMA}}^{(t)} = 0.99\,\theta_{\text{EMA}}^{(t-1)} + 0.01\,\theta^{(t)}$, $\theta_{\text{EMA}}^{(0)} = \theta_{\text{r882}}$, $t=1\dots 40$ | `True` |
| `EMA_0995` | $\theta_{\text{EMA}}^{(t)} = 0.995\,\theta_{\text{EMA}}^{(t-1)} + 0.005\,\theta^{(t)}$, $\theta_{\text{EMA}}^{(0)} = \theta_{\text{r882}}$, $t=1\dots 40$ | `True` |
| `EMA_0999` | $\theta_{\text{EMA}}^{(t)} = 0.999\,\theta_{\text{EMA}}^{(t-1)} + 0.001\,\theta^{(t)}$, $\theta_{\text{EMA}}^{(0)} = \theta_{\text{r882}}$, $t=1\dots 40$ | `True` |
| `SWA_K5` | Uniform average of last $K=5$ round snapshots: $\frac{1}{5}\sum_{r=16}^{20} \theta^{(2r)}$ | `True` |
| `SWA_K10` | Uniform average of last $K=10$ round snapshots: $\frac{1}{10}\sum_{r=11}^{20} \theta^{(2r)}$ | `True` |
| `SWA_K20` | Uniform average of all $K=20$ round snapshots: $\frac{1}{20}\sum_{r=1}^{20} \theta^{(2r)}$ | `True` |

- **Seeds:** `154101, 154102, 154103`.
- **Family fold seed (`FOLD_SEED = 154140`):** Splits the `205` `DL-Train` families (`1,219` contexts) into 5 disjoint 41-family folds ([fold_manifest.json](fold_manifest.json)). `Fold 0` (`41` families, `254` contexts) is held out from training and stat recalibration; `Folds 1..4` (`164` families, `965` contexts) are used for the 40-step continuation and stat recalibration.
- **Bootstrap seed (`BOOT_SEED = 154141`):** `10,000` shared family-cluster draws.
- **Train-only activation-stat recalibration (`compute_recal_rms`, `apply_stat_recalibration`):**
  - Weight averaging contracts the dispersion of the adapter's two centered residual branches (`mm_prior_adj` and `traj_attn_adj`).
  - Before any evaluation on `DL-Val` or `Fold 0`, we run a zero-gradient forward pass over the first `RECAL_SLATES = 8` deterministic in-fold `DL-Train` slates (`plan[:8]`, drawn exclusively from `Folds 1..4`) under `r882` ($\text{RMS}_{\text{ref}}$) and under each uncalibrated trajectory-averaged arm ($\text{RMS}_{\text{cand}}$).
  - We compute `mm_scale = clip(ref_mm_rms / cand_mm_rms, 0.85, 1.15)` and `traj_scale = clip(ref_traj_rms / cand_traj_rms, 0.85, 1.15)` (`RECAL_BOUNDS = (0.85, 1.15)`) and multiply the final linear head parameters `adapter_10tok.mm_prior_out.2.{weight,bias}` by `mm_scale` and `adapter_10tok.traj_delta_head.2.{weight,bias}` by `traj_scale`.
  - Every non-trainable tensor (`222` tensors including `adapter_10tok.anchor.*` and `DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) is asserted bitwise identical to `r882` before and after every step, recalibration, and evaluation.

## 3. Dose Manipulation Check (`dose_check`)

For each arm $a \in \text{ARMS}$, let $D_a = \text{median}_{s \in \text{SEEDS}} \|\theta_a(s) - \theta_{\text{r882}}\|_2$.
1. Every cell $(a, s)$ must have finite positive adapter L2 drift ($> 0$).
2. A trajectory-averaged arm $a \in \text{CANDIDATE\_ARMS}$ is marked `DOSE_INVALID` if its median drift lies within $\pm 5\%$ of `CTRL` (`0.95 * D_CTRL <= D_a <= 1.05 * D_CTRL`, `DOSE_INVALID_BOUNDS = (0.95, 1.05)`) **and** its `3 x 91` `DL-Val` utility panel is identical to `CTRL`. Any arm with `status == "DOSE_INVALID"` has `gates["dose"] = False` and `advance = False`.

## 4. Primary Estimand, Elo Readouts (`DL-Val`, `H145` 1,219-ctx `DL-Train`, `Fold 0`), and Decision Rules

1. **Exact baseline gate (before any trajectory step):** Unchanged `r882` must reproduce all `91/91` `DL-Val` utilities in `best_dev_checkpoint.json` with `max_context_error == 0.0` and `mean_error == 0.0` (`1.1377941898164525` context mean, `1.1536152769912449` 19-family mean).
2. **Primary readout (`primary_dl_val_91ctx_19fam`):** Equal-family native argmin utility (`-fitness` if valid else `-2.0`) across the `19` `DL-Val` families (`91` contexts), median across the 3 seeds, with 10,000-draw shared 19-family bootstrap 95% percentile CI (`BOOT_SEED = 154141`), one-sided $p = (1 + \#\{\text{boot} \le 0\}) / 10001$, `ties=reject`, and **paired family Elo vs `r882 = 1500.0`** (and vs `CTRL = 1500.0`) with 95% bootstrap CI.
3. **Secondary readouts (`secondary_h145_train_1219ctx_205fam` and `secondary_fold0_out_of_fold_254ctx_41fam`):**
   - Full `1,219`-context / `205`-family `H145` train-only panel (reporting equal-family utility gain, 95% bootstrap CI, **Elo vs `r882 = 1500.0`**, Elo vs `CTRL = 1500.0`, and invalid-pick counts; disclosed as partially in-sample because `Folds 1..4` are fit contexts).
   - Held-out `Fold 0` (`254` contexts, `41` `DL-Train` families, strictly out-of-fold for both training and stat recalibration), reporting the same equal-family utility gain, 95% CI, **Elo vs `r882 = 1500.0`**, Elo vs `CTRL = 1500.0`, and invalid-pick counts.
4. **Multiplicity and advancement rules (`decide` & `study_verdict`):**
   - **Unified 7-arm Holm step-down vs `r882` (`HOLM_M = 7`):** All 7 executed arms (`CTRL`, `EMA_099`, `EMA_0995`, `EMA_0999`, `SWA_K5`, `SWA_K10`, `SWA_K20`) are governed in a single 7-hypothesis Holm step-down family vs `r882` (`CTRL` cannot bypass Holm). An arm advances (`ADVANCE`) iff all 6 gates hold:
     1. `median_gain > 0.001`
     2. `ci95[0] > 0.0` (`ties=reject`)
     3. `seeds_beating == 3` (`3/3` seeds `> 0`)
     4. `top3_positive_share <= 0.8`
     5. 7-arm Holm-adjusted $p < 0.05$
     6. `dose == "DOSE_VALID"`
   - **6-arm Holm step-down vs `CTRL` (`HOLM_M_CTRL = 6`):** All 6 trajectory-averaged arms (`CANDIDATE_ARMS`) are tested against `CTRL` under the same per-arm gates with `HOLM_M_CTRL = 6`, yielding `trajectory_vs_ctrl_verdict = "TRAJECTORY_AVERAGING_BEATS_CTRL"` if any candidate arm passes all 6 gates vs `CTRL`, else `"NO_TRAJECTORY_AVERAGING_GAIN_OVER_CTRL"`.

## 5. Host, Hard-Stop Supervisor, and Squash-Compatible Source Binding

- **Host:** `xbabe2` CPU at `nice 19`, `CUDA_VISIBLE_DEVICES=""`, `OMP/MKL/OPENBLAS/NUMEXPR/RAYON=1`, total process-tree OS threads $\le 4$. Never touches `xbabe1` live flywheel (`PID 2297929`) or live checkpoints.
- **Hard-stop supervisor (`monitor.py`) and in-process guard (`runtime.guards`):** Every resource threshold is a **hard stop (`abort`, never pause or resume)**:
  - Process-tree threads `> 4` $\rightarrow$ `abort` (`TOTAL_THREADS`)
  - Process-tree RSS `>= 6,000,000,000` B $\rightarrow$ `abort` (`RSS`)
  - Disk free `< 40,000,000,000` B or disk used `>= 0.95` $\rightarrow$ `abort` (`DISK`)
  - Scratch usage across `/home/ubuntu/jope-data/h154` + custody dir `>= 1,000,000,000` B $\rightarrow$ `abort` (`SCRATCH`)
  - `load1 > 85.0` $\rightarrow$ `abort` (`LOAD`)
  - `MemAvailable < 16 GiB` (`16 * 1024**3` B) $\rightarrow$ `abort` (`MEMORY`)
  - Process `nice < 19` $\rightarrow$ `RuntimeError`
- **Squash-compatible source binding (`runtime.bind_source`):** Before running any trajectory step, `runner.py` verifies `launch.json` (`merged_source_verified == True`, `manifest_sha256` match) and checks via `git show` that all 6 `BOUND_FILES` (`PREREG.md`, `runner.py`, `runtime.py`, `science.py`, `monitor.py`, `dope_freeze_guard.py`) are byte-identical to both `reviewed_head` and the squash-merged `source_commit` on `main`.
