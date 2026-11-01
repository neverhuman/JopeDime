# H178 — Empirical Calibration of the H177 Pooled Cross-Fit Selector on the H175 Counterfactual Bank

- **Wave**: `H178`
- **Owner**: `gpu_tokbench_agy` (reserved by `ENG:hwave-loop`)
- **Status**: `COMPLETE`
- **Verdict**: **`CALIBRATED_PASS_POWER_OK`** (`calibration_verdict = CALIBRATED_PASS`, `power_verdict = POWER_OK`)
- **Authority**: `/home/ubuntu/nudges/hwave_t0440_h178_tokagy.md`, `/home/ubuntu/jope_plan/DECISIONS_20261029.md` (`2026-11-01T11:28Z` owner ruling on `H175 = MIXED` $\to$ calibration gate `H178`), and `/home/ubuntu/jope_plan/evidence/redteam-t0440/ruling_review.md`
- **Preregistration PRs**: [#1502](https://github.com/neverhuman/JopeDime/pull/1502) (`134764ca91debd35f081ddae41505b208fcec735`), preflight fix PRs [#1513](https://github.com/neverhuman/JopeDime/pull/1513) (`22e22e7c0f1c0ada2a4727bb3a4d51cd6b7a415c`) & [#1516](https://github.com/neverhuman/JopeDime/pull/1516) (`4de98a453de4214116330bb4e387c786d07f9211`)
- **Executed Source Commit**: `4de98a453de4214116330bb4e387c786d07f9211`
- **S3 Custody**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H178/gpu_tokbench_agy-h178-20261101/` (`67` versioned objects, `16,919,203` bytes; see [`custody.json`](custody.json))

---

## 1. Hypothesis

`H175` (`MIXED`, closeout [#1498](https://github.com/neverhuman/JopeDime/pull/1498)) demonstrated that the single-fold `41`-family `STRICT` selector (`margin +0.001`, `10,000`-draw paired family bootstrap `LCB95 > 0`, `3/3` seeds `> 0`, Holm, ties reject) has `0.000` false-admission rate across all `4` exact-null ($f=0$) and `24` nonzero `random_nonoracle` harmful cells, but achieves only `0.150` power at realistic family-clustered `5%` oracle improvements (`+0.016364` utility on `24`-slate and `+0.024480` on `full`-pool), requiring $f = 0.20$ (`+0.063957` on `24`-slate and `+0.094678` on `full`-pool) to reach `80%` detection power (`MDE80`).

`H177` ([#1492](https://github.com/neverhuman/JopeDime/pull/1492), amended by [#1500](https://github.com/neverhuman/JopeDime/pull/1500)) replaces single-fold `41`-family selection with pooled `205`-family cross-fit `B` selection (`pooled_b_select`), where each of the `5` disjoint `41`-family `B` folds is scored out-of-fold by its own `123`-family `FIT`-trained model and pooled into one `205`-family `B` vector before evaluating the strict selector. Under the owner's `2026-11-01T11:28Z` ruling (`DECISIONS_20261029.md`), before any `H177` `ADMIT` counts as lift, `H178` must empirically calibrate the exact imported `H177` decision function on the `H175` counterfactual policy bank (`1,219` `TRAIN` contexts / `205` families) to test two joint hypotheses:

1. **Calibration (`CALIBRATED_PASS`)**: Across all `4` exact-null ($f=0$) cells and all `24` nonzero `random_nonoracle` harmful cells (`28` null/harmful cells in total), the imported `H177` pooled `205`-family cross-fit selector admits at most `5%` of counterfactual policies (`admission_rate <= 0.05` in every null and harmful cell).
2. **Power (`POWER_OK`)**: Pooling all `205` out-of-fold `B` families reduces the family-clustered `80%` minimum detectable effect (`MDE80`) from `H175 STRICT`'s `+0.063957` (`24`-slate) / `+0.094678` (`full`-pool) down to `<= +0.030` utility units (`family MDE80 <= 0.03`).

---

## 2. Setup & Protocol

1. **Exact Imported `H177` Decision Function (`pooled_b_select`)**:
   - [`calibration.py`](calibration.py) directly imports `pooled_b_select`, `H177_DECISION_RULE`, `resolve_h177_verdict`, `StageOrderGuard`, `verify_pooled_b_coverage`, and `expected_mde80_205fam` from [`research/flywheel_selfplay/h177_pooled_selector/analysis.py`](../h177_pooled_selector/analysis.py) (SHA-256 `707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4`, `runner.py` SHA-256 `214f0fe82d3e81a5a30b380b7057555792a50437ccf82637c3fcfb4c6019a182`).
   - Both [`evaluate_counterfactual_draw_four_arm`](calibration.py#L209-L261) and [`evaluate_counterfactual_draw_cloned_four_arm`](calibration.py#L264-L320) consume `res[arm]["admitted"]` and `res[arm]["holm_p"]` directly from `pooled_b_select` without reimplementing any selection threshold or Bonferroni reconstruction (`test_holm_stepdown_vs_bonferroni_4p_counterexample` in [`test_h178.py`](test_h178.py) verifies on the frozen 10,000-draw counterexample `CTRL=+0.01`, `ACC4` four of 205 families `+0.06`, `ACC16/OC15=0.0` that `pooled_b_select` returns `ACC4 ADMIT` with `holm_p = 0.04829517048 < 0.05` whereas `4*p = 0.06439356` rejects).
2. **Counterfactual Policy Grid (`56` cells $\times$ `200` draws = `11,200` 3-seed counterfactual policies)**:
   - Reuses [`research/flywheel_selfplay/h175_selector_power/protocol.py`](../h175_selector_power/protocol.py) (`inject`, `slate_indices`, `family_mean`, `summarize`, `elo`, `headroom`, `mde`) and `H175`'s cached `r882` cost/utility vectors on `TRAIN` (`1,219` contexts / `205` families, `counterfactual_inputs.npz` SHA-256 `dd18f7b311b9da82c0efc6ee72a3303bbdced743485285f6476a5b84870132be`).
   - **Slates**: `24` (prespecified 24-program slate) and `full` (`500`-program pool).
   - **Variants**: `context` (independent context draws) and `family` (family-clustered draws).
   - **Directions**: `oracle` (beneficial shift toward pool oracle) and `random_nonoracle` (harmful shift toward a random non-oracle program).
   - **Fractions**: $f \in \{0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2\}$.
   - **Seeds**: Slate seeds `175101, 175102, 175103`; `H178` RNG seeds `178101, 178102, 178103`; `fold_seed = 178140`; `boot_seed = 178141`.
   - **Disjoint 4-Arm Cohort Mapping (`50` cohorts / `200` marginal arm decisions per cell)**: Within each cell, the `200` 3-seed counterfactual policy draws are deterministically partitioned into `50` disjoint 4-arm cohorts `(4*c, 4*c+1, 4*c+2, 4*c+3)` for $c \in \{0, \dots, 49\}$, mapped to `H177`'s four registered arms `("CTRL", "ACC4", "ACC16", "OC15")` and passed jointly to `pooled_b_select(..., n_boot=300, seed=178141)`. Every exact-null ($f=0$) and `random_nonoracle` cell—plus every boundary `oracle` candidate—is evaluated with `10,000` paired family bootstrap resamples (`FINAL_DRAWS = 10000`).
3. **Role Isolation, Stage-Order Barrier, & Guarded CPU Execution**:
   - Stage 1 constructs each outer fold's `41`-family `B` delta using `LocalOnlyPayload(payload, set(b))` (raising `KeyError` on any read outside `B_outer`), verifies disjoint `5 x 41 = 205` family coverage, evaluates `pooled_b_select` on the assembled `(3, 205)` `B` matrix, and writes + `os.fsync`s `lock-pooled-b-<cell>.json` via `StageOrderGuard.write_b_lock` before Stage 2 computes any `A`-fold outcome.
   - Executed on `xbabe2` CPU only (`nice 19`, `<= 4` threads, `RSS < 6 GB`, `0` GPU devices, `0` optimizer updates, `0` `.backward()` calls):
     - Exact `91/91` native `DL-Val` `r882` baseline verified first (`1.1377941898164525`, bit-exact cost vectors) with `TRAIN_outcome_pools_opened = 0`.
     - `DOPE` composite hash `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and model state hash `6eb0e998896bd87553bbea815be06b534d5f3f1ae2a63cbf7ff4ec22e89c4f8a` verified identical before and after the run; `audit.refused = []`.

---

## 3. Outcome & Key Metrics

### 3.1 Null ($f=0$) and Harmful (`random_nonoracle`) Calibration Across All 28 Cells

Across all `4` exact-null ($f=0$) cells and all `24` nonzero `random_nonoracle` harmful cells, **`H177_POOLED` (3-seed / 4-arm joint Holm stepdown), `H177_POOLED_CLONED4_DIAGNOSTIC` (`4*p` tied clone), `H177_POOLED_UNADJUSTED_M1_DIAGNOSTIC` ($m=1$), and `H177_POOLED_SINGLE_SEED_DIAGNOSTIC` admitted `0 / 200` counterfactual policies (`0.0000 <= 0.05`) and `0 / 50` 4-arm cohorts in every single cell**:

- **Marginal arm decision unit (`n = 200` per cell)**: `0 / 200` admitted (`rate = 0.0000`), exact Clopper–Pearson 95% one-sided upper bound **`0.014867`**, two-sided 95% CI **`[0.000000, 0.018275]`**.
- **Disjoint 4-arm cohort unit (`n = 50` cohorts per cell, `any(admitted)` across `CTRL, ACC4, ACC16, OC15`)**: `0 / 50` cohorts admitted (`rate = 0.0000`), exact Clopper–Pearson 95% one-sided upper bound **`0.058155`**, two-sided 95% CI **`[0.000000, 0.071122]`**.
- **Conditional false admission given nonpositive 3-seed `TRAIN` gain**: **`0.0000`** across all `28` null/harmful cells.

| Slate | Variant | Direction | Fraction $f$ | Injected Gain (Median) | Injected Elo | `H177_POOLED` Arm Rate (`k/200`, `CP95_1u`) | `H177_POOLED` Cohort Rate (`k/50`, `CP95_1u`) | `CLONED4` / `M1` / `SINGLE_SEED` | `H175 STRICT` (`41`-fam) | `H168_NEW` Total (Cond. $\le 0$) |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| `24` | `context` | `oracle` / `random_nonoracle` | `0` | `+0.000000` | `1500.00` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0000` (`0.0000`) |
| `24` | `context` | `random_nonoracle` | `0.005` | `-0.003024` | `1499.22` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0590` (`0.0445`) |
| `24` | `context` | `random_nonoracle` | `0.01` | `-0.006074` | `1498.35` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0020` | `0.0740` (`0.0680`) |
| `24` | `context` | `random_nonoracle` | `0.02` | `-0.011672` | `1496.70` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0030` | `0.0890` (`0.0807`) |
| `24` | `context` | `random_nonoracle` | `0.05` | `-0.029466` | `1491.67` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0010` | `0.0520` (`0.0520`) |
| `24` | `context` | `random_nonoracle` | `0.10` | `-0.058665` | `1483.40` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0230` (`0.0230`) |
| `24` | `context` | `random_nonoracle` | `0.20` | `-0.119536` | `1466.52` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0060` (`0.0060`) |
| `24` | `family` | `oracle` / `random_nonoracle` | `0` | `+0.000000` | `1500.00` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0000` (`0.0000`) |
| `24` | `family` | `random_nonoracle` | `0.005` | `-0.002642` | `1499.18` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0220` (`0.0114`) |
| `24` | `family` | `random_nonoracle` | `0.01` | `-0.005801` | `1498.37` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0260` (`0.0170`) |
| `24` | `family` | `random_nonoracle` | `0.02` | `-0.011547` | `1496.46` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0390` (`0.0315`) |
| `24` | `family` | `random_nonoracle` | `0.05` | `-0.030981` | `1491.32` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0550` (`0.0528`) |
| `24` | `family` | `random_nonoracle` | `0.10` | `-0.062385` | `1482.70` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0400` (`0.0400`) |
| `24` | `family` | `random_nonoracle` | `0.20` | `-0.122951` | `1465.95` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0200` (`0.0200`) |
| `full` | `context` | `oracle` / `random_nonoracle` | `0` | `+0.000000` | `1500.00` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0000` (`0.0000`) |
| `full` | `context` | `random_nonoracle` | `0.005` | `-0.003000` | `1499.28` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0570` (`0.0445`) |
| `full` | `context` | `random_nonoracle` | `0.01` | `-0.006013` | `1498.55` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0010` | `0.0750` (`0.0680`) |
| `full` | `context` | `random_nonoracle` | `0.02` | `-0.011816` | `1497.11` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0020` | `0.0940` (`0.0839`) |
| `full` | `context` | `random_nonoracle` | `0.05` | `-0.029602` | `1492.85` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0570` (`0.0570`) |
| `full` | `context` | `random_nonoracle` | `0.10` | `-0.058207` | `1485.63` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0200` (`0.0200`) |
| `full` | `context` | `random_nonoracle` | `0.20` | `-0.117602` | `1471.68` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0040` (`0.0040`) |
| `full` | `family` | `oracle` / `random_nonoracle` | `0` | `+0.000000` | `1500.00` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0000` (`0.0000`) |
| `full` | `family` | `random_nonoracle` | `0.005` | `-0.002437` | `1499.31` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0170` (`0.0115`) |
| `full` | `family` | `random_nonoracle` | `0.01` | `-0.005714` | `1498.65` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0300` (`0.0191`) |
| `full` | `family` | `random_nonoracle` | `0.02` | `-0.011284` | `1497.14` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0450` (`0.0355`) |
| `full` | `family` | `random_nonoracle` | `0.05` | `-0.029371` | `1493.02` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0600` (`0.0576`) |
| `full` | `family` | `random_nonoracle` | `0.10` | `-0.060375` | `1485.77` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0360` (`0.0360`) |
| `full` | `family` | `random_nonoracle` | `0.20` | `-0.119085` | `1471.94` | **`0.0000`** (`0/200`, `0.0149`) | **`0.0000`** (`0/50`, `0.0582`) | `0.0000` | `0.0000` | `0.0140` (`0.0140`) |

---

### 3.2 Minimum Detectable Effect at 80% Power (`MDE80`) Side-by-Side Comparison

- **Equal-family `TRAIN` headroom (`205` families, `1,219` contexts)**:
  - `24`-slate: median **`+0.320167`** `[+0.256873, +0.393324]` (Oracle Elo **`1958.53`** `[1933.43, 1988.13]` vs `r882 = 1500.00`).
  - `full`-pool: median **`+0.467187`** `[+0.360408, +0.596798]` (Oracle Elo **`2601.89`** `[2471.75, 2913.41]` vs `r882 = 1500.00`).
- Both the primary 3-seed / 4-arm `H177_POOLED` rule (independent counterfactual draws across the 3 slate seeds `175101–175103`) and the single-seed cloned diagnostic `H177_POOLED_SINGLE_SEED_DIAGNOSTIC` (worst-case `1.0` inter-seed correlation, isolating the pure $\sqrt{5}$ `205`-vs-`41` family pooling reduction) achieve **`family MDE80 <= 0.030` (`POWER_OK`)**:

| Slate | Variant | Selector Rule | Grid Fraction $f$ | `MDE80` (Utility Units, `[95% CI]`) | `MDE80` (`% of Headroom`, `[95% CI]`) | Extrapolated $\sqrt{5}$ Reference |
| --- | --- | --- | ---: | --- | --- | ---: |
| `24` | `context` | **`H177_POOLED` (3-seed / 4-arm primary)** | **`0.01`** | **`+0.003049`** `[+0.002461, +0.003769]` | **`0.95%`** `[0.88%, 1.06%]` | `+0.007634` |
| `24` | `context` | `H177_POOLED_SINGLE_SEED_DIAGNOSTIC` | `0.01` | `+0.003049` `[+0.002461, +0.003769]` | `0.95%` `[0.88%, 1.06%]` | `+0.007634` |
| `24` | `context` | `H175 STRICT` (`41`-family single-fold) | `0.05` | `+0.015834` `[+0.012770, +0.019565]` | `4.95%` `[4.72%, 5.30%]` | — |
| `24` | `context` | `H168_NEW` (`41`-family `> 0` relaxed) | `0.05` | `+0.015834` `[+0.012770, +0.019565]` | `4.95%` `[4.72%, 5.30%]` | — |
| `24` | `family` | **`H177_POOLED` (3-seed / 4-arm primary)** | **`0.01`** | **`+0.003038`** `[+0.002378, +0.003911]` | **`0.95%`** `[0.85%, 1.09%]` | `+0.029155` |
| `24` | `family` | `H177_POOLED_CLONED4_DIAGNOSTIC` (`4*p`) | `0.02` | `+0.006423` `[+0.004949, +0.008204]` | `2.01%` `[1.81%, 2.21%]` | `+0.029155` |
| `24` | `family` | `H177_POOLED_SINGLE_SEED_DIAGNOSTIC` (1-seed clone) | `0.05` | **`+0.016364`** `[+0.012737, +0.020876]` | **`5.11%`** `[4.76%, 5.52%]` | `+0.029155` |
| `24` | `family` | `H175 STRICT` (`41`-family single-fold) | `0.20` | `+0.063957` `[+0.050418, +0.078688]` | `19.98%` `[19.17%, 20.61%]` | — |
| `24` | `family` | `H168_NEW` (`41`-family `> 0` relaxed) | `0.05` | `+0.016364` `[+0.012737, +0.020876]` | `5.11%` `[4.76%, 5.52%]` | — |
| `full` | `context` | **`H177_POOLED` (3-seed / 4-arm primary)** | **`0.005`** | **`+0.002201`** `[+0.001726, +0.002814]` | **`0.47%`** `[0.44%, 0.53%]` | `+0.010750` |
| `full` | `context` | `H177_POOLED_SINGLE_SEED_DIAGNOSTIC` | `0.01` | `+0.004461` `[+0.003441, +0.005861]` | `0.95%` `[0.89%, 1.04%]` | `+0.010750` |
| `full` | `context` | `H175 STRICT` (`41`-family single-fold) | `0.05` | `+0.022802` `[+0.017594, +0.029180]` | `4.88%` `[4.74%, 5.06%]` | — |
| `full` | `context` | `H168_NEW` (`41`-family `> 0` relaxed) | `0.02` | `+0.009355` `[+0.007139, +0.011868]` | `2.00%` `[1.90%, 2.07%]` | — |
| `full` | `family` | **`H177_POOLED` (3-seed / 4-arm primary)** | **`0.01`** | **`+0.005057`** `[+0.003576, +0.006494]` | **`1.08%`** `[0.93%, 1.17%]` | `+0.042894` |
| `full` | `family` | `H177_POOLED_CLONED4_DIAGNOSTIC` (`4*p`) | `0.02` | `+0.009826` `[+0.007116, +0.012700]` | `2.10%` `[1.87%, 2.24%]` | `+0.042894` |
| `full` | `family` | `H177_POOLED_SINGLE_SEED_DIAGNOSTIC` (1-seed clone) | `0.05` | **`+0.024480`** `[+0.018345, +0.032226]` | **`5.24%`** `[4.86%, 5.64%]` | `+0.042894` |
| `full` | `family` | `H175 STRICT` (`41`-family single-fold) | `0.20` | `+0.094678` `[+0.072157, +0.121182]` | `20.27%` `[19.70%, 20.74%]` | — |
| `full` | `family` | `H168_NEW` (`41`-family `> 0` relaxed) | `0.05` | `+0.024480` `[+0.018345, +0.032226]` | `5.24%` `[4.86%, 5.64%]` | — |

---

### 3.3 Beneficial (`oracle`) Power Curves Across All 28 Cells

| Slate | Variant | Fraction $f$ | Injected Gain (Median) | Injected Elo | `H177_POOLED` Arm Power (`k/200`) | `H177_POOLED` Cohort Power (`k/50`) | Selected OOF Gain | Selected Elo | `CLONED4` | `SINGLE_SEED` | `H175 STRICT` (`41`-fam) | `H168_NEW` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `24` | `context` | `0.005` | `+0.001537` | `1501.46` | `0.6400` (`128/200`) | `0.9600` (`48/50`) | `+0.001301` | `1501.00` | `0.6400` | `0.4050` | `0.0190` | `0.2850` |
| `24` | `context` | `0.01` | `+0.003049` | `1502.92` | **`0.9850`** (`197/200`) | **`1.0000`** (`50/50`) | `+0.003030` | `1502.87` | `0.9850` | **`0.8100`** | `0.1230` | `0.4940` |
| `24` | `context` | `0.02` | `+0.006466` | `1505.91` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.006466` | `1505.91` | `1.0000` | `0.9950` | `0.5210` | `0.7780` |
| `24` | `context` | `0.05` | `+0.015834` | `1514.91` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.015834` | `1514.91` | `1.0000` | `1.0000` | **`0.9830`** | `0.9890` |
| `24` | `context` | `0.10` | `+0.031888` | `1529.82` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.031888` | `1529.82` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `24` | `context` | `0.20` | `+0.064411` | `1560.55` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.064411` | `1560.55` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `24` | `family` | `0.005` | `+0.001533` | `1501.52` | `0.1950` (`39/200`) | `0.5000` (`25/50`) | `+0.000611` | `1500.52` | `0.1600` | `0.0050` | `0.0000` | `0.1520` |
| `24` | `family` | `0.01` | `+0.003038` | `1503.07` | **`0.8000`** (`160/200`) | **`1.0000`** (`50/50`) | `+0.002900` | `1502.81` | `0.6650` | `0.0750` | `0.0000` | `0.2750` |
| `24` | `family` | `0.02` | `+0.006423` | `1506.19` | **`0.9900`** (`198/200`) | **`1.0000`** (`50/50`) | `+0.006417` | `1506.17` | **`0.9900`** | `0.4650` | `0.0080` | `0.4780` |
| `24` | `family` | `0.05` | `+0.016364` | `1515.30` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.016364` | `1515.30` | `1.0000` | **`0.9900`** | `0.1500` | **`0.8080`** |
| `24` | `family` | `0.10` | `+0.032799` | `1530.70` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.032799` | `1530.70` | `1.0000` | `1.0000` | `0.6070` | `0.9660` |
| `24` | `family` | `0.20` | `+0.063957` | `1560.71` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.063957` | `1560.71` | `1.0000` | `1.0000` | **`0.9760`** | `1.0000` |
| `full` | `context` | `0.005` | `+0.002201` | `1501.70` | **`0.8500`** (`170/200`) | **`1.0000`** (`50/50`) | `+0.002078` | `1501.50` | **`0.8500`** | `0.5600` | `0.0230` | `0.3230` |
| `full` | `context` | `0.01` | `+0.004461` | `1503.40` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.004461` | `1503.40` | `1.0000` | **`0.9200`** | `0.1760` | `0.5810` |
| `full` | `context` | `0.02` | `+0.009355` | `1506.81` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.009355` | `1506.81` | `1.0000` | `1.0000` | `0.6570` | **`0.8740`** |
| `full` | `context` | `0.05` | `+0.022802` | `1517.19` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.022802` | `1517.19` | `1.0000` | `1.0000` | **`0.9960`** | `0.9990` |
| `full` | `context` | `0.10` | `+0.045948` | `1534.53` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.045948` | `1534.53` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `full` | `context` | `0.20` | `+0.093858` | `1569.87` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.093858` | `1569.87` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `full` | `family` | `0.005` | `+0.002228` | `1501.72` | `0.2200` (`44/200`) | `0.5600` (`28/50`) | `+0.001023` | `1500.68` | `0.1800` | `0.0050` | `0.0000` | `0.1550` |
| `full` | `family` | `0.01` | `+0.005057` | `1503.54` | **`0.8300`** (`166/200`) | **`1.0000`** (`50/50`) | `+0.004806` | `1503.29` | `0.6750` | `0.0750` | `0.0000` | `0.3090` |
| `full` | `family` | `0.02` | `+0.009826` | `1507.07` | **`0.9950`** (`199/200`) | **`1.0000`** (`50/50`) | `+0.009819` | `1507.05` | **`0.9950`** | `0.4650` | `0.0080` | `0.5130` |
| `full` | `family` | `0.05` | `+0.024480` | `1517.55` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.024480` | `1517.55` | `1.0000` | **`0.9900`** | `0.1500` | **`0.8330`** |
| `full` | `family` | `0.10` | `+0.048789` | `1535.34` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.048789` | `1535.34` | `1.0000` | `1.0000` | `0.6090` | `0.9750` |
| `full` | `family` | `0.20` | `+0.094678` | `1570.08` | `1.0000` (`200/200`) | `1.0000` (`50/50`) | `+0.094678` | `1570.08` | `1.0000` | `1.0000` | **`0.9760`** | `1.0000` |

---

## 4. Verdict & `H177` Gate Binding

- **Verdict**: **`CALIBRATED_PASS_POWER_OK`** (`calibration_verdict = CALIBRATED_PASS`, `power_verdict = POWER_OK`).
- **Programmatic `H177` Receipt ([`h178_receipt.json`](h178_receipt.json))**:
  ```json
  {
    "wave": "H178",
    "status": "COMPLETE",
    "null_and_harmful_admission_le_005": true,
    "max_null_or_harmful_admission_rate": 0.0,
    "replayed_four_arm_holm_m": 4,
    "h177_analysis_sha256": "707c4903a77d48ffc8ae14c820a626df5c2858d3d24df35db3f9718d120bafc4"
  }
  ```
- Verified directly with `h177_analysis.resolve_h177_verdict(candidate_admitted, h178_receipt)`:
  - When `candidate_admitted=True`: `{"candidate_admitted": True, "lift_qualified": True, "h178_calibration_status": "PASS", "verdict": "ADVANCE"}`.
  - When `candidate_admitted=False`: `{"candidate_admitted": False, "lift_qualified": False, "h178_calibration_status": "EVALUATED_NO_ADMIT", "verdict": "NO_ADVANCE"}`.

---

## 5. Limitations & Next Step

1. **Scope & Limitations**:
   - **Correlated pooled `A` readout**: Under 5-fold cross-fitting where each `41`-family `B` fold is paired with a `41`-family `A` fold from the same `205`-family partition, the `5` `A` folds cover the same `205` families as pooled `B` (evaluated under a different `123`-family `FIT`-trained model per fold). Under counterfactual policy injections (which depend only on context/family identity and pool utilities, not on which `FIT` model was trained), pooled `A` equals pooled `B`; in live `H177` training, pooled `A` is a correlated out-of-fold check across the `5` trained fold models, not an untouched holdout cohort.
   - **Inter-seed correlation bracket**: In the counterfactual bank, the 3 slate seeds (`175101–175103`) draw independent family masks in `H177_POOLED` (`family MDE80 = +0.003038` on `24`-slate and `+0.005057` on `full`-pool at $f=0.01$), whereas `H177_POOLED_SINGLE_SEED_DIAGNOSTIC` clones each draw across all 3 seed slots (`family MDE80 = +0.016364` on `24`-slate and `+0.024480` on `full`-pool at $f=0.05$). Real `H177` 3-seed continuation runs share the `205` `TRAIN` families and `r882` initialization, so their effective family-clustered `MDE80` lies between those two bounds (`[+0.003038, +0.016364]` on `24`-slate and `[+0.005057, +0.024480]` on `full`-pool)—both of which are below `+0.030` (`POWER_OK`).
   - **Zero-update selector calibration**: `H178` calibrates selector false-admission and detection power on the `r882` counterfactual bank (`0` optimizer updates); it does not train a new checkpoint or promote a serving model (`serving_promotion = False`).
2. **Next Step**:
   - With `H178 = CALIBRATED_PASS_POWER_OK` and [`h178_receipt.json`](h178_receipt.json) committed, `H177`'s calibration gate is satisfied (`h178_calibration_status = PASS`). Any candidate admitted by `H177` (`pooled_b_select` + `A`-family confirmation) is now eligible to resolve as `ADVANCE` under `resolve_h177_verdict`.

---

## 6. Verification Commands

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q research/flywheel_selfplay/h178_pooled_calibration/test_h178.py
ruff check research/flywheel_selfplay/h178_pooled_calibration/
ruff format --check research/flywheel_selfplay/h178_pooled_calibration/
```
