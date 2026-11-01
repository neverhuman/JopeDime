# Wave H190: `ENS10` Confirmation with Independent Fold Seeds per Outer Seed (`190141, 190142, 190143`)

- **Status**: `COMPLETE` — **`NO_ADVANCE`** (`candidate_admitted = false`, `lift_qualified = false`, `h178_calibration_status = EVALUATED_NO_ADMIT`, `serving_promotion = false`)
- **Owner**: `gpu_fused_agy` (`[FUSED-AGY]`)
- **Authority**: `/home/ubuntu/nudges/hwave_t0835_h190.md`
- **Preregistration**: PR `#1579`, reviewed `PASS` by `@rd_argon` (review ID `5728910422` / `5728900711`), squash-merged at `796d1bad71cac306fb7070e425cab2ee56699649`
- **Execution Host**: `xbabe1` (`CPU-only`, `nice -n 19`, `1` torch thread, `<= 8` process threads, `RSS < 8 GB`, single guarded run in `/home/ubuntu/lane_work/h190-agy/attempt-001`)
- **S3 Custody**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H190/gpu_fused_agy-t0835/` (`14` versioned objects, `36,048,769` bytes, exact-version byte-readback verified in [custody.json](custody.json))

---

## 1. Executive Summary & Verdict

`H190` tests whether `H182`'s `ENS10` (`10`-member fresh-init `adapter_10tok` score ensemble, `40` `FIT`-only updates per member, predicted costs averaged before `argmin`) and its matched `0`-update init-ensemble control `RENS10` (`10` identical fresh member initializations, `0` updates) confirm against frozen `r882` (`R882`) under the imported `H177` pooled selector (`research/flywheel_selfplay/h177_pooled_selector/analysis.py`) and `H178` calibration receipt (`research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json`) when **each outer seed uses its own independent `5`-fold `3/1/1` family partition** (`FOLD_SEED_BY_SEED = {190101: 190141, 190102: 190142, 190103: 190143}`).

1. **Verdict — `NO_ADVANCE` (`serving_promotion = false`)**:
   - On Stage-1 pooled `B` (`205` `DL-Train` families, `1,219` contexts per seed), both `ENS10` and `RENS10` are positive on **`3/3` independent-partition seeds** vs `R882`:
     - `ENS10 - R882`: per-seed equal-family utility gains `[+0.017758, +0.020039, +0.018407]`, median-seed gain **`+0.018407`** (`95%` paired family bootstrap CI `[-0.012231, +0.053469]`, one-sided $p = 0.121188$, Holm $p_{m=2} = 0.242376$), mean gain **`+0.018735`** (`95%` CI `[-0.012133, +0.053318]`, one-sided $p = 0.122788$, Holm $p_{m=2} = 0.245575$), Bradley-Terry Elo vs `R882` **`1511.16`** (`95%` CI `[1494.18, 1525.37]`, seed-averaged `1510.40`), exact `-2.0` picks `[16, 15, 16]` vs `[17, 17, 17]` (`R882`).
     - `RENS10 - R882`: per-seed equal-family utility gains `[+0.015606, +0.016292, +0.018504]`, median-seed gain **`+0.016292`** (`95%` CI `[-0.013844, +0.051190]`, one-sided $p = 0.146085$, Holm $p_{m=2} = 0.242376`), mean gain **`+0.016801`** (`95%` CI `[-0.013970, +0.050976]`, one-sided $p = 0.148485$, Holm $p_{m=2} = 0.245575`), Elo vs `R882` **`1508.26`** (`95%` CI `[1491.07, 1522.65]`, seed-averaged `1507.43`), exact `-2.0` picks `[16, 16, 16]`.
   - Because neither arm's `10,000`-draw paired family bootstrap `95%` CI lower bound clears `> 0` (`LCB95 = -0.012133` for `ENS10`, `-0.013970` for `RENS10`; Holm $p_{m=2} = 0.245575 \ge 0.05$), the imported `H177_DECISION_RULE` (`pooled_b_select`) rejects both candidates at Stage 1 and locks `selected_step = 0` (`R882` fallback) in `pooled_b_selection_lock.json` (`locked_before_a_confirmation = true`) before any `A` or `DL-Val` context is loaded or scored.
2. **Stage-2 Pooled `A` Confirmation (`205` families, `1,219` contexts)**:
   - Under the Stage-1 locked selection (`selected_step = 0`), both arms revert to `R882` (`0.000000 [0.000000, 0.000000]`, Elo `1500.00`, `-2.0` picks `[17, 17, 17]`).
   - On the unconditional prespecified candidate readout (`step = 40` for `ENS10`, `step = 0` for `RENS10`):
     - `ENS10 - R882` is positive on **all `3/3` independent-partition seeds**: `[+0.021457, +0.023076, +0.018990]`, median-seed gain **`+0.021457`** (`95%` CI `[-0.009456, +0.055994]`, one-sided $p = 0.092191$, Holm $p_{m=2} = 0.184382$), mean gain **`+0.021174`** (`95%` CI `[-0.009697, +0.055344]`, one-sided $p = 0.093291$, Holm $p_{m=2} = 0.186581`), Elo vs `R882` **`1512.44`** (`95%` CI `[1495.48, 1526.89]`, seed-averaged `1511.65`), exact `-2.0` picks `[15, 14, 16]` vs `[17, 17, 17]`.
     - `ENS10 - RENS10` (diagnostic pure-training lift of `40` `FIT`-only updates per member over the exact same `10` fresh member initializations) is positive on **`3/3` independent-partition seeds** on `A`: `[+0.005851, +0.006784, +0.000486]`, median-seed gain **`+0.005851`** (`95%` CI `[+0.000907, +0.009853]`, one-sided $p = 0.002100$), mean gain **`+0.004374`** (`95%` CI `[+0.001450, +0.007981]`, one-sided $p = 0.000100$). Still, `79.3%` (`+0.016801 / +0.021174`) of `ENS10`'s unconditional lift over `R882` on `A` comes from the zero-update `10`-member random initialization ensemble `RENS10`.
3. **Partition-Variance Component Decomposition**:
   - Replacing `H182`'s shared fold partition (`182140`, between-seed SD `0.003172`, range `0.006281`) with three independent fold partitions (`190141, 190142, 190143`, pairwise fold Jaccard `0.1027–0.1285`) yields a between-seed SD of **`0.001175`** (range `0.002281`) on pooled `B` and **`0.002058`** (range `0.004086`) on pooled `A` for `ENS10 - R882`.
   - Variance decomposition across the `205` `DL-Train` families proves that the paired family bootstrap CI half-width (`~0.0327`) is dominated by **between-family heterogeneity** against the static `R882` anchor (`between_family_var_ddof1 = 0.055045`, family SD `0.2346`, $\text{SE}_{\text{family}} = 0.01639$, reflecting `~50%` context-choice disagreement with `R882`), which is **`23.8x` larger** than the within-family between-seed partition variance (`0.002311`, SD `0.04807`) and **`13,003x` larger** than the between-seed variance of the 3-seed mean (`4.233e-6`).

---

## 2. Guarded Execution & Integrity Gates

| Gate / Invariant | Requirement | Measured in `attempt-001` | Status |
| --- | --- | --- | --- |
| Execution host & priority | `xbabe1`, `CPU-only`, `nice 19`, `1` torch thread, `<= 8` threads, `RSS < 8 GB` | `host = xbabe1`, `nice = 19`, `status = PASS`, `exit_code = 0` | `PASS` |
| Timing budget pilot | First `4` updates of `seed190101-fseed190141-A0` `MEMBER1` on `FIT_0`; cap `9,000 s` | `0.052411 s/update`, projected `1,214.47 s < 9,000 s` (`fallback_triggered = false`, `active_ensemble_k = 10`) | `PASS` |
| Exact `91/91` `DL-Val` `R882` baseline | Mean `1.1377941898164525`, max per-context abs diff `0.0`, bit-exact raw costs | `1.1377941898164525`, max error `0.0`, `eval_forward_bit_exact = true` | `PASS` |
| Static vs dynamic forward & grad parity | Initial and perturbed (`seed 190777`) `47`-tensor (`41` active + `6` `None`) parity | `static_grad_parity_bit_exact = true`, `static_grad_parity_perturbed_bit_exact = true` | `PASS` |
| Frozen `DOPE` & non-adapter guard | `DOPE` (`c51a56d0...`) & frozen (`8d1070fa...`) bit-identical across all `6,000` updates | Verified on every step (`29,211` trainable params, `41` active / `6` `None` grad tensors) | `PASS` |
| Exact `RENS10` init matching | Every `RENS10` member `m` uses exact initial weights of `ENS10` member `m` (`0` updates) | `rens10_uses_exact_ens10_inits = true`, `RENS10` L2 drift `0.0` (vs `ENS10` member mean median `0.236267`) | `PASS` |
| Independent fold partitions | `190101 -> 190141`, `190102 -> 190142`, `190103 -> 190143`, exact `205`-family `B` & `A` coverage | `independent_partitions_verified = true`, pairwise fold Jaccard `0.1027`, `0.1285`, `0.1240` | `PASS` |
| Stage-1 `B` lock before `A` | `pooled_b_selection_lock.json` written & `os.fsync`ed before any `A` or `DL-Val` scoring | `locked_before_a_confirmation = true` (`StageOrderGuard` enforced) | `PASS` |
| File-access audit & recount | Zero sealed/holdout reads; independent recount from `panels.npz` | `0` unexpected reads (`audit.json`), `100%` bit-exact recount (`recount_verification.json`) | `PASS` |

---

## 3. Stage-1 Pooled `B` Selection (`205` `DL-Train` Families, `1,219` Contexts)

Each outer seed $s \in \{190101, 190102, 190103\}$ forms its own `5`-fold cyclic `3/1/1` family partition (`123` `FIT` / `41` `B` / `41` `A` families per fold) using `fold_seed` $f_s \in \{190141, 190142, 190143\}$. Across the `5` outer folds, $B_{s,0} \cup \dots \cup B_{s,4}$ covers all `205` `DL-Train` families (`1,219` contexts) exactly once per seed.

### 3.1 Arm Summary on Pooled `B`

| Arm | Step | Per-Seed Equal-Family Utility (`190101, 190102, 190103`) | Median-Seed Utility (`95%` CI) | Seed-Averaged Utility (`95%` CI) | Per-Seed Context Utility | Bradley-Terry Elo vs `R882` (Median `[95% CI]` / Per-Seed) | Exact `-2.0` Picks per Seed | Choice Agreement with `R882` | Member Pairwise Choice Disagreement (Mean / Median / Per-Seed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `R882` | `0` | `[1.385355, 1.385355, 1.385355]` | `1.385355` (`[1.141118, 1.674772]`) | `1.385355` (`[1.141118, 1.674772]`) | `[1.423073, 1.423073, 1.423073]` | `1500.00` (`[1500.00, 1500.00]`) / `[1500.00, 1500.00, 1500.00]` | `[17, 17, 17]` | `[1.0000, 1.0000, 1.0000]` | — |
| `ENS10` | `40` | `[1.403114, 1.405394, 1.403762]` | `1.403762` (`[1.159747, 1.695782]`) | `1.404090` (`[1.159345, 1.695765]`) | `[1.432581, 1.435138, 1.432566]` | `1511.16` (`[1494.18, 1525.37]`) / `[1511.16, 1508.56, 1511.47]` | `[16, 15, 16]` | `[0.4922, 0.5021, 0.5021]` | `0.3417` / `0.3422` / `[0.3422, 0.3423, 0.3405]` |
| `RENS10` | `0` | `[1.400962, 1.401648, 1.403860]` | `1.401648` (`[1.157590, 1.693954]`) | `1.402156` (`[1.158240, 1.693730]`) | `[1.432696, 1.431128, 1.432777]` | `1508.26` (`[1491.07, 1522.65]`) / `[1508.26, 1505.48, 1508.56]` | `[16, 16, 16]` | `[0.4938, 0.4988, 0.4996]` | `0.3352` / `0.3359` / `[0.3323, 0.3373, 0.3359]` |

### 3.2 Paired Contrasts & Selector Decision on Pooled `B` (`10,000` Family Bootstrap Draws, `boot_seed = 190150`, `axis = 0`)

| Contrast | Role | Per-Seed Gain (`190101, 190102, 190103`) | Pos Seeds | Median-Seed Gain (`95%` CI) | Mean Gain (`95%` CI) | One-Sided $p$ (Median / Mean) | Holm $p_{m=2}$ (Median / Mean) | Admitted (`> +0.001`, `LCB95 > 0`, Holm $p < 0.05$) | Locked Step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ENS10 - R882` | Primary ($m=2$) | `[+0.017758, +0.020039, +0.018407]` | `3/3` | `+0.018407` (`[-0.012231, +0.053469]`) | `+0.018735` (`[-0.012133, +0.053318]`) | `0.121188` / `0.122788` | `0.242376` / `0.245575` | `false` | `0` (`R882`) |
| `RENS10 - R882` | Control ($m=2$) | `[+0.015606, +0.016292, +0.018504]` | `3/3` | `+0.016292` (`[-0.013844, +0.051190]`) | `+0.016801` (`[-0.013970, +0.050976]`) | `0.146085` / `0.148485` | `0.242376` / `0.245575` | `false` | `0` (`R882`) |
| `ENS10 - RENS10` | Diagnostic | `[+0.002152, +0.003747, -0.000097]` | `2/3` | `+0.002152` (`[+0.000222, +0.005053]`) | `+0.001934` (`[+0.000377, +0.003929]`) | `0.012999` / `0.004200` | — | — | — |

---

## 4. Stage-2 Pooled `A` Confirmation (`205` `DL-Train` Families, `1,219` Contexts)

Across the `5` outer folds per seed, $A_{s,0} \cup \dots \cup A_{s,4}$ covers all `205` `DL-Train` families (`1,219` contexts) disjoint from `FIT_{s, outer}` and `B_{s, outer}` in every fold. (Selection caveat: across all `5` folds, each family appears once in `B` and once in `A` under models trained on `2/3`-overlapping `123`-family `FIT` sets.)

### 4.1 Stage-1 Selected System on Pooled `A` (`selected_step = 0` Fallback)

Because Stage 1 locked `selected_step = 0` for both `ENS10` and `RENS10`, the selected system on `A` is identical to `R882`:
- `ENS10 (selected)` & `RENS10 (selected)` equal-family utility: `1.385355` (`95%` CI `[1.141118, 1.674772]`), paired gain vs `R882`: **`0.000000`** (`[0.000000, 0.000000]`, Holm $p = 1.0$), Bradley-Terry Elo **`1500.00`**, exact `-2.0` picks `[17, 17, 17]`.

### 4.2 Unconditional Prespecified Candidate Readout on Pooled `A` (`step = 40` for `ENS10`, `step = 0` for `RENS10`)

| Arm / Contrast | Per-Seed Utility or Gain (`190101, 190102, 190103`) | Pos Seeds | Median-Seed Utility or Gain (`95%` CI) | Mean Utility or Gain (`95%` CI) | One-Sided $p$ (Median / Mean) | Holm $p_{m=2}$ (Median / Mean) | Bradley-Terry Elo vs `R882` (Median `[95% CI]` / Per-Seed) | Exact `-2.0` Picks per Seed | Choice Agreement with `R882` | Member Disagreement (Mean / Median / Per-Seed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `R882` | `[1.385355, 1.385355, 1.385355]` | — | `1.385355` (`[1.141118, 1.674772]`) | `1.385355` (`[1.141118, 1.674772]`) | — | — | `1500.00` (`[1500.00, 1500.00]`) / `[1500.00, 1500.00, 1500.00]` | `[17, 17, 17]` | `[1.0000, 1.0000, 1.0000]` | — |
| `ENS10` (`step 40`) | `[1.406812, 1.408432, 1.404346]` | — | `1.406812` (`[1.161802, 1.698255]`) | `1.406530` (`[1.161979, 1.697860]`) | — | — | `1512.44` (`[1495.48, 1526.89]`) / `[1513.08, 1509.42, 1512.44]` | `[15, 14, 16]` | `[0.4947, 0.5037, 0.5012]` | `0.3423` / `0.3430` / `[0.3432, 0.3430, 0.3407]` |
| `RENS10` (`step 0`) | `[1.400962, 1.401648, 1.403860]` | — | `1.401648` (`[1.157590, 1.693954]`) | `1.402156` (`[1.158240, 1.693730]`) | — | — | `1507.70` (`[1490.97, 1522.28]`) / `[1507.70, 1506.30, 1508.29]` | `[16, 16, 16]` | `[0.4938, 0.4988, 0.4996]` | `0.3352` / `0.3359` / `[0.3323, 0.3373, 0.3359]` |
| `ENS10 - R882` | `[+0.021457, +0.023076, +0.018990]` | `3/3` | `+0.021457` (`[-0.009456, +0.055994]`) | `+0.021174` (`[-0.009697, +0.055344]`) | `0.092191` / `0.093291` | `0.184382` / `0.186581` | — | — | — | — |
| `RENS10 - R882` | `[+0.015606, +0.016292, +0.018504]` | `3/3` | `+0.016292` (`[-0.013844, +0.051190]`) | `+0.016801` (`[-0.013970, +0.050976]`) | `0.146085` / `0.148485` | `0.184382` / `0.186581` | — | — | — | — |
| `ENS10 - RENS10` (diag) | `[+0.005851, +0.006784, +0.000486]` | `3/3` | `+0.005851` (`[+0.000907, +0.009853]`) | `+0.004374` (`[+0.001450, +0.007981]`) | `0.002100` / `0.000100` | — | — | — | — | — |

### 4.3 Diagnostic Nested `k = 5` Prefix (`ENS5` / `RENS5`, Members `1..5`)

- **Pooled `B` (`k = 5` prefix)**:
  - `ENS5 - R882`: per-seed gains `[+0.010185, +0.012465, +0.012334]`, `3/3` positive, median-seed gain `+0.012334` (`[-0.021036, +0.047240]`), mean gain `+0.011661` (`[-0.019553, +0.047010]`), Elo `1511.61` (`[1495.55, 1528.23]`), `-2.0` picks `[17, 17, 16]`, member disagreement `0.3405`.
  - `RENS5 - R882`: per-seed gains `[+0.005909, +0.011932, +0.011825]`, `3/3` positive, median-seed gain `+0.011825` (`[-0.021919, +0.044779]`), mean gain `+0.009889` (`[-0.021709, +0.045185]`), Elo `1506.16` (`[1491.63, 1523.09]`), `-2.0` picks `[18, 17, 16]`.
- **Confirmation `A` Unconditional (`k = 5` prefix)**:
  - `ENS5 - R882`: per-seed gains `[+0.011267, +0.014567, +0.014972]`, `3/3` positive, median-seed gain `+0.014567` (`[-0.018510, +0.048870]`), mean gain `+0.013602` (`[-0.017792, +0.049174]`), Elo `1510.16` (`[1494.83, 1527.26]`), `-2.0` picks `[17, 17, 16]`, member disagreement `0.3415`.
  - `ENS5 - RENS5`: per-seed gains `[+0.005358, +0.002635, +0.003147]`, `3/3` positive, median-seed gain `+0.003147` (`[+0.000479, +0.007672]`), mean gain `+0.003713` (`[+0.000954, +0.007282]`).
  - Increasing ensemble size from `k = 5` (`+0.013602` mean / `+0.014567` median on `A`) to `k = 10` (`+0.021174` mean / `+0.021457` median on `A`) adds `+0.007572` mean (`+0.006890` median) equal-family utility and cuts `-2.0` cliff picks from `[17, 17, 16]` to `[15, 14, 16]`.

---

## 5. Partition-Variance Component Decomposition (`H190` Independent Folds vs `H182` Shared Folds)

Because `H190` assigns each outer seed its own independent `3/1/1` family fold partition (`190141, 190142, 190143`), we can directly decompose total gain variance across the $F = 205$ `DL-Train` families and $S = 3$ outer seeds into:
1. **Between-Seed Variance (`between_seed_var_ddof1` / `between_seed_sd_ddof1`)**: dispersion of the 3 seed-level equal-family gains $\bar{\Delta}_s = \frac{1}{205}\sum_{f=1}^{205} \Delta_{s,f}$.
2. **Within-Family Between-Seed Variance (`within_family_between_seed_var_ddof1`)**: mean across the `205` families of the sample variance ($\text{ddof}=1$) of family $f$'s gain across the 3 independent fold partitions and member seed sets. Note that for `RENS10 - R882` (`0` updates), each member is evaluated before any `FIT` updates, so within-family between-seed variance (`0.003098`, SD `0.05566`) reflects pure 10-member random-init sampling variance, whereas for `ENS10 - RENS10` (`0.001118` on `B`, `0.002255` on `A`), it reflects the combined `FIT`-partition + SGD-stream variance of the 40-update training delta.
3. **Between-Family Variance (`between_family_var_ddof1`)**: sample variance ($\text{ddof}=1$) across the `205` families of their seed-averaged gain $\bar{\Delta}_{\cdot, f} = \frac{1}{3}\sum_{s=1}^3 \Delta_{s,f}$, which governs the `10,000`-draw paired family bootstrap standard error $\text{SE}_{\text{family}} \approx \sqrt{\text{Var}_{\text{between-family}} / 205}$.

| Panel & Contrast | Per-Seed Gains (`190101, 190102, 190103`) | Between-Seed SD (`ddof=1`) | Between-Seed Range | Between-Seed SE of Mean | Within-Family Between-Seed Var (SD) | Between-Family Var (`205` fam) | Implied Family Bootstrap SE ($\sqrt{\text{Var}_{\text{fam}}/205}$) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Pooled `B`**: `ENS10 - R882` | `[+0.017758, +0.020039, +0.018407]` | `0.001175` | `0.002281` | `0.000678` | `0.003445` (`0.05870`) | `0.055174` | `0.016406` |
| **Pooled `B`**: `RENS10 - R882` | `[+0.015606, +0.016292, +0.018504]` | `0.001515` | `0.002898` | `0.000874` | `0.003098` (`0.05566`) | `0.054510` | `0.016306` |
| **Pooled `B`**: `ENS10 - RENS10` | `[+0.002152, +0.003747, -0.000097]` | `0.001931` | `0.003844` | `0.001115` | `0.001118` (`0.03344`) | `0.000173` | `0.000920` |
| **Confirm `A`**: `ENS10 - R882` | `[+0.021457, +0.023076, +0.018990]` | `0.002058` | `0.004086` | `0.001188` | `0.002311` (`0.04807`) | `0.055045` | `0.016386` |
| **Confirm `A`**: `RENS10 - R882` | `[+0.015606, +0.016292, +0.018504]` | `0.001515` | `0.002898` | `0.000874` | `0.003098` (`0.05566`) | `0.054510` | `0.016306` |
| **Confirm `A`**: `ENS10 - RENS10` | `[+0.005851, +0.006784, +0.000486]` | `0.003399` | `0.006298` | `0.001962` | `0.002255` (`0.04749`) | `0.000600` | `0.001710` |
| **`H182` Ref (Shared Fold `182140`)**: `ENS10 - R882` | `[+0.015200, +0.019116, +0.012835]` | `0.003172` | `0.006281` | `0.001831` | — | — | — |

**Key Structural Takeaway**:
- Independent outer-fold partitions do **not** destabilize `ENS10`: between-seed SD (`0.001175` on `B`, `0.002058` on `A`) is even tighter than `H182`'s shared-fold between-seed SD (`0.003172`), and all `3/3` independent-partition seeds remain strictly positive (`+0.0178` to `+0.0200` on `B`; `+0.0190` to `+0.0231` on `A`).
- Instead, `ENS10` fails the `LCB95 > 0` gate because fresh-init adapters change the chosen candidate on `~50%` of contexts relative to `R882`, inducing a large between-family variance (`0.0550`, $\text{SE}_{\text{family}} \approx 0.0164$, `95%` CI half-width `~0.0327`) that requires an effect size of `>+0.032` to clear a two-sided `95%` family bootstrap lower bound on `205` families.

---

## 6. Descriptive `DL-Val` Readout (`91` Contexts, `19` Families)

- **Stage-1 Selected (`selected_step = 0` fallback)**:
  - All arms equal `R882`: equal-family utility `1.153615` (`95%` CI `[0.742545, 1.650946]`), context utility `1.137794`, paired gain `0.000000 [0.000000, 0.000000]`, Elo `1500.00`, `-2.0` picks `[0, 0, 0]`.
- **Unconditional (`step = 40` for `ENS10`, `step = 0` for `RENS10`, averaged across the `5` fold models per seed before scoring)**:
  - `R882`: equal-family utility `1.153615` (`[0.742545, 1.650946]`), context utility `1.137794`, Elo `1500.00`, `-2.0` picks `[0, 0, 0]`.
  - `ENS10`: per-seed equal-family utility `[1.133119, 1.130811, 1.115007]`, median-seed gain vs `R882` **`-0.022804`** (`[-0.096087, +0.022282]`, `0/3` positive seeds), mean gain **`-0.027303`** (`[-0.103837, +0.020229]`), Bradley-Terry Elo vs `R882` **`1514.62`** (`[1476.83, 1556.50]`, per-seed `[1519.50, 1514.62, 1508.53]`), `-2.0` picks `[1, 1, 1]`.
  - `RENS10`: per-seed equal-family utility `[1.128775, 1.127833, 1.110402]`, median-seed gain vs `R882` **`-0.025783`** (`[-0.102373, +0.021530]`, `0/3` positive seeds), mean gain **`-0.031279`** (`[-0.109038, +0.018413]`), Bradley-Terry Elo vs `R882` **`1517.07`** (`[1473.17, 1565.36]`, per-seed `[1517.07, 1518.28, 1517.06]`), `-2.0` picks `[1, 1, 2]`.
  - `ENS10 - RENS10`: per-seed gains `[+0.004344, +0.002979, +0.004605]` (`3/3` positive seeds), median-seed gain `+0.004344` (`[-0.001397, +0.009891]`), mean gain `+0.003976` (`[-0.000850, +0.009818]`).

---

## 7. Artifacts & Reproducibility

- [PREREG.md](PREREG.md) — Preregistered protocol (PR `#1579`, `@rd_argon` review `5728910422`).
- [fold_manifest.json](fold_manifest.json) — Exact `5`-fold `3/1/1` family partitions for `190141`, `190142`, and `190143`.
- [analysis.py](analysis.py), [treatment.py](treatment.py), [runner.py](runner.py), [monitor.py](monitor.py), [test_h190.py](test_h190.py) — Guarded execution and statistical evaluation code.
- [metrics.json](metrics.json) — Complete results (`69 KB < 100 KB`), independently recounted from `panels.npz`.
- [custody.json](custody.json) — Versioned S3 custody manifest (`14` objects, `36,048,769` bytes, `VersionId` + SHA-256 readback verified).
