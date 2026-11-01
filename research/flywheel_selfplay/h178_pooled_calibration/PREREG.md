# H178 Preregistration: Empirical Calibration of the H177 Pooled Cross-Fit Selector on the H175 Counterfactual Bank

- **Wave**: `H178`
- **Owner**: `gpu_tokbench_agy` (`[TOKBENCH-AGY]`, reserved by `ENG:hwave-loop`)
- **Authority**: `/home/ubuntu/nudges/hwave_t0440_h178_tokagy.md`, `/home/ubuntu/jope_plan/DECISIONS_20261029.md` (`2026-11-01T11:28Z` owner ruling on `H175 = MIXED` $\to$ calibration gate `H178`), and `/home/ubuntu/jope_plan/evidence/redteam-t0440/ruling_review.md`
- **Seeds & Partition Pins**:
  - `H178` bootstrap / decision RNG seeds: `SEEDS = (178101, 178102, 178103)`, `FOLD_SEED = 178140`, `BOOT_SEED = 178141`
  - Reused `H175` counterfactual slate & activation RNG seeds: `SLATE_SEEDS = (175101, 175102, 175103)`
- **Optimizer Budget**: `0` optimizer updates, `0` `.backward()` calls, `0` trainable parameters (strictly read-only statistical calibration on `xbabe2` CPU).

---

## 1. Scientific Question & Context

`H175` (*Selector Power, Minimum Detectable Effect & Oracle Headroom*, prereg PR `#1490`, closeout PR `#1498`, merge commit `b0261812e125c8743370b7801b75d27e7f27977f`) closed with verdict **`MIXED`**:
- Real S3 `TRAIN` (`1,219` contexts / `205` families) has large oracle headroom above `r882` (`+0.320167` equal-family utility on `24`-slate, `+0.467187` on `full`-pool), and under context-independent sparse oracle switches (`variant = "context"`) the single-fold `41`-family `STRICT` selector reaches `MDE80 = +0.015834` (`24`-slate) / `+0.022802` (`full`-pool, below the 6-study historical median maximum `+0.028149`) at $f = 5\%$.
- However, under **family-clustered** oracle switches (`variant = "family"`), the `41`-family `STRICT` selector has **only `15.0%` power at $f = 5\%$** (`+0.016364` / `+0.024480` gain) and `0.8%` power at $f = 2\%$ (`+0.006423` / `+0.009826`), requiring **$f = 20\%$ (`MDE80 = +0.063957` on `24`-slate, `+0.094678` on `full`-pool)** to reach `80%` power.
- Meanwhile, loosening to the mean-only rule `H168_NEW` (`gain > 0.001 + 1e-12`, no `LCB95 > 0` gate) inflates overall `random_nonoracle` mirror admission to **`9.4%`** on `full/context` $f = 0.02$ and conditional false admission given nonpositive full-`TRAIN` population gain to **`8.394%`** (`194/198/193` eligible nonpositive simulations per seed; note that `random_nonoracle` replacements can improve a poor baseline on some draws, so total mirror admission and conditional nonpositive-truth false admission must be reported separately).

`H177` (prereg PR `#1492`, amendment PR `#1500` merged at `70bf6698301d544fcb522998fffe0bc8dd7ec278`, `analysis.py` SHA-256 `707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4`) preregistered a **pooled `205`-family cross-fit `B` selector** (`research/flywheel_selfplay/h177_pooled_selector/analysis.py::pooled_b_select` / `H177_DECISION_RULE`) with a single prespecified candidate step (`step = 40`), `+0.001 + 1e-12` margin, `3`-seed within-family averaging before `10,000`-draw paired family bootstrap `LCB95 > 0`, `m = 4` step-down Holm adjustment across the `4` registered arms `("CTRL", "ACC4", "ACC16", "OC15")`, tie rejection, and separate Stage-2 `A`-family readout. Per `DECISIONS_20261029.md` (`11:28Z`) and `ruling_review.md`, `H177` lift is conditional on `H178`: `H178` empirically calibrates the **exact imported `H177` four-arm/three-seed decision function** on `H175`'s counterfactual bank.

---

## 2. Exact Imported Selector, Four-Arm Bank Mapping, & Counterfactual Protocol

1. **Exact Imported `H177` Selector (`pooled_b_select` / `H177_DECISION_RULE`)**:
   - `calibration.py` directly imports `pooled_b_select`, `H177_DECISION_RULE`, `resolve_h177_verdict`, `StageOrderGuard`, `verify_pooled_b_coverage`, and `expected_mde80_205fam` from `research/flywheel_selfplay/h177_pooled_selector/analysis.py` (`707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4`) and consumes the returned `d["admitted"]` and `d["holm_p"]` fields directly without ever reconstructing `m = 1` or Bonferroni `4 * p` decisions outside `pooled_b_select`.
   - `test_h178.py` proves both function-object identity and exact decision agreement on the frozen-source 10k-draw counterexample (`CTRL` constant `+0.01`, `ACC4` four of `205` families `+0.06`, `ACC16/OC15` null `0.0`, where step-down Holm `m = 4` returns `ACC4 ADMIT` with `holm_p = 0.04829517048 < 0.05` whereas naive `4 * p = 0.06439356` rejects).
   - For each counterfactual draw, Stage 1 injects the policy into each of the `5` disjoint `41`-family selection blocks $B_{\text{outer}} = F_{(\text{outer}+1)\bmod 5}$ ($\text{outer} \in \{0..4\}$) using `LocalOnlyPayload(payload, bscope, role_name="B")`, which raises `KeyError` on any read outside $B_{\text{outer}}$.
   - Pooling $B_0 \sqcup B_1 \sqcup B_2 \sqcup B_3 \sqcup B_4$ covers all `205` `TRAIN` families (`1,219` contexts) **exactly once** outside each fold's `123`-family `FIT` split and `41`-family `A` split.
   - `StageOrderGuard` writes and `os.fsync`s `lock-pooled-b-<cell>.json` to disk and verifies `require_b_locked()` **before** any Stage-2 `A`-family context is injected or scored.

2. **Prespecified Four-Arm Bank Mapping, Trial Units, & Selector Rules**:
   - Reuses `research/flywheel_selfplay/h175_selector_power/protocol.py` (`inject`, `slate_indices`, `family_mean`, `summarize`, `elo`, `headroom`, `mde`) and `H175`'s pinned cached `r882` cost/utility bank (`/home/ubuntu/jope-data/h178/inputs/counterfactual_inputs.npz`, SHA-256 `dd18f7b311b9da82c0efc6ee72a3303bbdced743485285f6476a5b84870132be`).
   - Across `56` cells (`slates` $\in \{\text{24}, \text{full}\}$, `variants` $\in \{\text{context}, \text{family}\}$, `directions` $\in \{\text{oracle}, \text{random\_nonoracle}\}$, $f \in \{0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20\}$), each cell generates `200` Monte Carlo simulation draws across the `3` slate seeds `175101, 175102, 175103`, yielding `200` three-seed pooled `B` panels $P_0, \dots, P_{199} \in \mathbb{R}^{3 \times 205}$.
   - **`H177_POOLED` (PRIMARY Rule — Exact 3-Seed Within-Family Averaged `(3, 205)` Panels + Exact 4-Arm `pooled_b_select` Decisions)**:
     - Partitions the `200` three-seed panels $P_0, \dots, P_{199}$ in deterministic simulation order into **`n_cohorts = 50` disjoint 4-arm selector cohorts** $b \in \{0..49\}$: `{"CTRL": P_{4b}, "ACC4": P_{4b+1}, "ACC16": P_{4b+2}, "OC15": P_{4b+3}}`, calls `pooled_b_select(rows_205, r882_zeros, c, seed=BOOT_SEED)`, and consumes `res[arm]["admitted"]` and `res[arm]["holm_p"]` directly.
     - **Explicit Trial / Clopper–Pearson Units**:
       1. `clopper_pearson_cohort_any_arm_50`: **`n = 50` independent 4-arm cohorts** per cell (`any(res[a]["admitted"] for a in ARMS)`);
       2. `clopper_pearson_cohort_treatment3_50`: **`n = 50` independent 4-arm cohorts** per cell where any of the 3 treatment arms (`ACC4, ACC16, OC15`) is admitted (matching `H177` `runner.py` line 866 `if arm != "CTRL"`);
       3. `clopper_pearson_marginal_arm_200`: **`200` marginal arm decisions** across the `50` 4-arm cohorts (explicitly disclosing within-cohort step-down Holm coupling across 4 arms).
     - **Harmful / Conditional False-Admission Denominator**: Uses the exact same 3-seed equal-family `TRAIN` gain functional $\bar{\Delta}_{\text{3seed}}(d) = \frac{1}{205}\sum_{f=1}^{205}\frac{1}{3}\sum_{s=1}^3 \Delta_{s, d, f}$ as `pooled_b_select`. Reports total `random_nonoracle` mirror admission (`n = 200` marginal / `n = 50` cohort) separately from conditional false admission given $\bar{\Delta}_{\text{3seed}}(d) \le 0$ (`None` / unavailable if zero eligible trials).
   - **Explicit Diagnostic Companion Rules**:
     - `H177_POOLED_CLONED4_DIAGNOSTIC`: passes each 3-seed panel $P_d$ to all 4 arms `{arm: P_d for arm in ARMS}` in `pooled_b_select` ($n = 200$ independent 3-seed trials under worst-case 4-arm tie where every candidate faces the rank-1 `4 * p` Holm factor), plus `H177_POOLED_UNADJUSTED_M1_DIAGNOSTIC` (`res[arm]["unadjusted_admitted"]`).
     - `H177_POOLED_SINGLE_SEED_DIAGNOSTIC`: clones each single slate seed (`175101, 175102, 175103`) into `(3, 205)` before calling `pooled_b_select` so the pure `205`-family pooling effect ($\sqrt{5}$) is separated from cross-seed averaging ($\sqrt{3}$) over independent random activation masks.
     - `STRICT` (`H175`'s `41`-family single-fold rule) and `H168_NEW` (`41`-family mean-only `> 0.001 + 1e-12` rule), verified bit-for-bit against `H175`'s `simulation_metrics.json`.
   - **Bootstrap Draw Schedule**:
     - Every exact-null ($f=0$) cell, every positive-mean cohort in `random_nonoracle` cells (since any arm with `mean_gain <= 0.001 + 1e-12` has `unadjusted_admitted = False` and `admitted = False` identically for every bootstrap draw count $D \ge 1$ in `pooled_b_select`), and every boundary `oracle` cohort (`mean_gain > 0.001 + 1e-12` and `0.0002 <= p_one_sided <= 0.30`) is evaluated by calling `pooled_b_select` at the full **`FINAL_DRAWS = 10000`** paired family bootstrap draws.

---

## 3. Required Outputs, Preregistered Pass/Fail Verdict, & Scope Limitations

1. **Admission Rate & Exact Clopper–Pearson 95% Bounds per Cell**:
   - Report observed admission rates and exact Clopper–Pearson 95% one-sided upper (`cp95_one_sided_upper`) and two-sided (`cp95_two_sided`) bounds for both the `n = 50` 4-arm cohort unit and the `n = 200` marginal arm unit, plus conditional false-admission rates given nonpositive 3-seed `TRAIN` gain.
2. **Side-by-Side `MDE80` Table**:
   - Report `MDE80` (smallest tested grid fraction with admission rate $\ge 80\%$, in equal-family utility units and `% of headroom`, with `10,000`-draw conditional family bootstrap 95% CIs) for both `context` and `family`-clustered variants on `24`-slate and `full`-pool across `H177_POOLED`, `H177_POOLED_CLONED4_DIAGNOSTIC`, `H177_POOLED_SINGLE_SEED_DIAGNOSTIC`, `H175 STRICT` (`41`-family), and `H168_NEW`, alongside both the simple $\text{MDE}_{41}/\sqrt{5}$ (`+0.028602` / `+0.042341`) and margin-adjusted (`+0.029155` / `+0.042894`) heuristics.
3. **Elo Reporting**:
   - Report expected pairwise-win `H148` Bradley–Terry Elo vs `r882 = 1500.0` (`0.01` pseudo-wins, tie margin `1e-4`) for every injected and selected policy. No AUC or speed metrics are computed or reported.
4. **Fixed Preregistered Pass/Fail Criterion & `H177` Programmatic Receipt (`h178_receipt.json`)**:
   - **Calibration Gate (`CALIBRATED_PASS` vs `CALIBRATED_FAIL`)**:
     - `CALIBRATED_PASS` iff across **all `4` exact-null ($f=0$) cells** and **all `28` `random_nonoracle` cells**, the observed point admission rate satisfies **$\le 0.05$** across `H177_POOLED` (`marginal_arm_200`, `cohort_any_arm_50`, and `cohort_treatment3_50`), `H177_POOLED_CLONED4_DIAGNOSTIC`, and `H177_POOLED_SINGLE_SEED_DIAGNOSTIC`. Otherwise `CALIBRATED_FAIL`.
     - Emits `h178_receipt.json` (and top-level `metrics.json` fields) with `wave = "H178"`, `status = "COMPLETE"`, `null_and_harmful_admission_le_005`, `max_null_or_harmful_admission_rate`, `replayed_four_arm_holm_m = 4`, and `h177_analysis_sha256 = "707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4"`, consumed programmatically by `h177_analysis.resolve_h177_verdict`.
   - **Power Classification (`POWER_OK` vs `POWER_LOW`)**:
     - `POWER_OK` iff `H177_POOLED` achieves `GRID_DETECTED` family-clustered `MDE80 <= 0.03` utility units on both `24`-slate and `full`-pool; otherwise `POWER_LOW`.
5. **Scope & Calibration Disclosures (`ruling_review.md`)**:
   - **Correlated Cross-Fold `A` Readout**: Pooled `A` and pooled `B` contain the exact same `205` families (`1,219` contexts). In the pointwise counterfactual bank, pooled `A` repeats the same `205`-family gain used in pooled `B` across rotated fold roles; in `H177`, pooled `A` is a correlated cross-fold candidate readout on shared family rosters, not an independent new-family confirmation sample.
   - **Structural Finite-Bank Gate vs External Near-Margin Noise**: Any policy with 3-seed finite-`TRAIN` gain $\le 0$ already fails the deterministic $> 0.001 + 10^{-12}$ mean gate in `pooled_b_select`. Passing the finite-bank `<= 0.05` point criterion does not certify external-population effects just below the `+0.001` margin (e.g., an external normal population with $\mu = 0.00099, \sigma_{\text{fam}} = 0.001, N = 205$ crosses the observed `+0.001` mean gate $\approx 44.3\%$ of the time), nor is a point rate $\le 0.05$ at $n = 200$ ($10/200$ has one-sided 95% CP upper bound `0.083335`) a 95% upper-bound certification at `0.05` unless $k \le 4/200$.

---

## 4. Hard Guardrails & Execution Lifecycle

- **Pre-`TRAIN` Native Baseline Gate**: Before any `TRAIN` outcome pool or cached counterfactual input is opened, `runner.py` loads `round-882.pt` with `DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and model state `6eb0e998896bd87553bbea815be06b534d5f3f1ae2a63cbf7ff4ec22e89c4f8a`, freezes all parameters, verifies the exact `91`-context `DL-Val` utility vector (`mean = 1.1377941898164525`) and native float32 cost bit-identity, and writes `baseline.json` with `TRAIN_outcome_pools_opened = 0`.
- **Post-Run State Integrity**: Verifies `DOPE` SHA-256 and whole-model state SHA-256 (`6eb0e998...`) match before and after, with `optimizer_updates = 0` and `audit.refused == []`.
- **Host & Resource Guards (`xbabe2` CPU only)**: `nice -n 19`, `1` math thread, `<= 4` aggregate supervisor/worker threads, `RSS < 6 GB`, `CUDA_VISIBLE_DEVICES=""`. Terminal stop if `load > 85%` of online cores, host `CPU > 85%`, `MemAvailable < 16 GiB`, `free disk < 40 GB`, or `disk_used_fraction >= 0.938`.
- **Custody**: Sync raw panels (`panels.npz`, `simulation_metrics.json`, `metrics.json`, `h178_receipt.json`, `baseline.json`, `provenance.json`, `lock-pooled-b-*.json`) to versioned S3 custody `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H178/gpu_tokbench_agy-h178-20261101/` and remove local bulk `.npz` artifacts after verified S3 readback.
