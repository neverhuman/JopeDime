# H159 — TRAIN-calibrated `fail_logit` veto at selection time (zero-update) on `r882`

## Hypothesis & Mechanism

`H143` found that every live admission since `r774` was a `-2` gate-cliff flip, and `H152`/`H139` deficits on 91-context `DL-Val` are driven by single `-2` cliff contexts. Inside `r882`, the frozen `FlywheelGameNet` anchor already emits per-candidate `fail_logit` alongside `total_cost` (`net.adapter_10tok.predict_costs(...)` returns both `total_cost` and `fail_logit`). While `anchor_total` adds a soft barrier `softplus(gate_barrier_weight) * sigmoid(fail_logit)`, `adapter_total_delta` adds multimodal and trajectory adjustments on top of `anchor_total` without a hard failure-probability cutoff at candidate selection time.

**Hypothesis (`H159`)**: A zero-update selection-time decision rule — *"pick $\arg\min$ `total_cost` among candidates with $\sigma(\text{fail\_logit}) < t$, else fall back to plain $\arg\min$ `total_cost`"* — with threshold $t \in \mathcal{T} = \{0.50, 0.60, 0.70, 0.80, 0.90, 0.95\}$ calibrated exclusively on `DL-Train` families cuts `-2` cliff picks and raises equal-family native utility vs `r882`'s plain $\arg\min$. Zero optimizer updates (`0` steps), adapter weights unchanged, DOPE (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) untouched.

If `fail_logit` is already folded into `total_cost` (or $\sigma(\text{fail\_logit}) < 0.50$ on selected candidates) so that the veto is a no-op across `VETO_FIX` and `VETO_CV`, the preregistered verdict is **`RULE_NEUTRAL`** (`r882_verdict: NO_ADVANCE`), recorded as a valid scientific finding.

## Arms & Calibration Design

1. **`R882` (Baseline & Regression Gate)**:
   - Plain $\arg\min_i \text{total\_cost}_i$ ($t = 1.0$).
   - **Regression-first gate**: Must reproduce `1.1377941898164525` (`91/91` exact context matches, `max_context_error = 0.0`, equal-family mean `1.1536152769912449`, `0` invalid picks) on the 91-context `DL-Val` panel before any contrast is evaluated.
2. **`VETO_FIX` (Primary Arm on 91-context `DL-Val` / 19 families)**:
   - Candidate threshold grid: $\mathcal{T} = \{0.50, 0.60, 0.70, 0.80, 0.90, 0.95\}$.
   - Tie-breaking: if multiple thresholds $t \in \mathcal{T}$ achieve the exact maximum equal-family mean utility on the fit families, select the **largest** $t \in \mathcal{T}$ (most conservative veto / closest to plain `R882` $\arg\min$).
   - Deterministic point fit: $t_{\text{full}}^* \in \mathcal{T}$ chosen by equal-family utility across all `205` `DL-Train` families (`1,219` contexts) and applied to 91-context `DL-Val`.
   - Selection-noise resampling seeds (`159101, 159102, 159103`): for each seed $s \in \{159101, 159102, 159103\}$, resample `205` families with replacement from the `205` `DL-Train` families (`np.random.default_rng(s)`), choose $t_s^* \in \mathcal{T}$ maximizing equal-family utility on the resampled `DL-Train` families, and apply $t_s^*$ to 91-context `DL-Val`.
3. **`VETO_CV` (Secondary Out-of-Fold Arm on 1,219-context / 205-family `DL-Train`)**:
   - 5-fold family-grouped partition of the `205` `DL-Train` families (`fold_seed = 159140`, `41` families per fold, context counts `[245, 240, 225, 268, 241]`; pinned in [fold_manifest.json](fold_manifest.json)).
   - For each outer fold $k \in \{0, 1, 2, 3, 4\}$, choose $t_{\text{cv}, k}^* \in \mathcal{T}$ by equal-family utility on the 4 fit folds (`164` families) and evaluate out-of-fold on held-out fold $k$ (`41` families), both deterministically and across the 3 family-resampling seeds `159101, 159102, 159103` (`np.random.default_rng(10 * s + k)`).
4. **`ORACLE_INFO` (Descriptive Diagnostic Only — Never Used for Selection)**:
   - Reports quantiles (`min, p50, p90, p99, max`) and fraction $\ge 0.50$ of $\sigma(\text{fail\_logit})$ across all candidates and across `R882` winners on `DL-Val` and `DL-Train`, plus the count of `-2` cliff picks under `R882` avoidable at any $t \in \mathcal{T}$ (`avoidable_on_grid_count`) and at any real threshold $t \in (0, 1)$ (`avoidable_at_any_t_count`).

## Statistical Inference & Advancement Criteria

- **Primary Contrast**: `VETO_FIX minus R882` on 91-context `DL-Val` (`19` families; historical exposure disclosed), equal-family mean within seed, median across seeds `159101, 159102, 159103`, 10,000-draw shared family-cluster bootstrap (`boot_seed = 159141`, strict `ties = "reject"`), Bradley-Terry Elo anchored at `R882 = 1500.0`.
- **Verdict Rule**:
  - `RULE_NEUTRAL` (`r882_verdict: NO_ADVANCE`) if `VETO_FIX` and `VETO_CV` change `0` candidate selections vs `R882` across all contexts and seeds.
  - `ADVANCE` only if all 5 strict primary gates pass: median equal-family gain $> +0.001$, `ci95[0] > 0.0`, $\ge 2/3$ seeds positive, `p_one_sided < 0.05`, and invalid (`-2`) picks $\le$ `R882` invalid picks; otherwise `NO_ADVANCE`.

## Execution Safeguards & Resource Bounds

- **Host & CPU bounds (`xbabe2` CPU only)**: `nice 19`, `CUDA_VISIBLE_DEVICES=""`, total OS threads $\le 4$ (supervisor + worker tree), RSS $< 6\times 10^9\text{ B}$, owned scratch $< 3\times 10^9\text{ B}$, free disk $\ge 40\times 10^9\text{ B}$, user-visible disk used `(blocks - bfree) / ((blocks - bfree) + bavail) < 0.95`, 1-min load $\le 85.0$, and `MemAvailable >= 16 GiB`. Every threshold breach is a terminal `INCOMPLETE` abort (never pause-and-resume).
- **Input & Freeze Closure**: `0` optimizer updates; `Audit` hook blocks any protected/holdout/sealed path and any write outside the run output directory; bytewise identity of all 47 adapter tensors (`29,211` params), all non-adapter parameters/buffers, and DOPE (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) is verified before and after scoring.
