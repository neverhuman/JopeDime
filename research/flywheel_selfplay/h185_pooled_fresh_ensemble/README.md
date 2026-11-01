# H185: Power-Aware Pooled Confirmatory Test of Fresh-Init and Ensemble Candidates (`FRESH160`, `ENS5`, and `RENS5` vs `R882`)

- **Wave**: `H185`
- **Owner**: `gpu_tokbench_agy` (`[TOKBENCH-AGY]`)
- **Reserved by**: `ENG:hwave-loop`
- **Authority**: `/home/ubuntu/nudges/hwave_t0700_h185_tokagy.md` (`DECISIONS_20261029.md`)
- **Preregistration**: [PREREG.md](PREREG.md), [fold_manifest.json](fold_manifest.json), [manifest.json](manifest.json), [provenance.json](provenance.json) (merged in `#1547` at `60c856959a3c6a137a35468f91fda7fa89ee3fd5`)
- **Closeout Artifacts**: [metrics.json](metrics.json), [custody.json](custody.json)
- **Imported Selector & Calibration Anchor**:
  - Imported directly from [`research/flywheel_selfplay/h177_pooled_selector/analysis.py`](../h177_pooled_selector/analysis.py) (`H177_DECISION_RULE`, `pooled_b_select`, `StageOrderGuard`, `verify_pooled_b_coverage`, `resolve_h177_verdict`).
  - Calibrated by `H178` (`CALIBRATED_PASS_POWER_OK`, [`research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json`](../h178_pooled_calibration/h178_receipt.json), commit `6b70d803b3c51ca8cd46720244dbdfa872c08868`: `0/200` null/harmful admissions, `24`-slate family-clustered synthetic `MDE80 = +0.003038`, single-seed cloned `+0.016364`).
- **Execution Host**: `xbabe2` (`CPU-only`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `<= 4` threads)
- **Status**: **`COMPLETE — NO_ADVANCE`** (`candidate_admitted = false`, `lift_qualified = false`, `h178_calibration_status = EVALUATED_NO_ADMIT`, `serving_promotion = false`)

---

## 1. Hypothesis, Setup, & Executive Outcome

### Hypothesis
Prior fresh-initialization waves (`H179` `ENS5` `+0.0137`, `H180` `FRESH160` `+0.0106`, `H182` `ENS20` `+0.0168`) achieved positive point estimates across `3/3` outer seeds vs `R882` on `205` `DL-Train` families, but failed single-fold `41`-family `B` selection (`H175` family-clustered `MDE80 = +0.0640/+0.0947`). `H185` tests whether pooling `B` across all `5` cyclic cross-fit folds (`205` families, `1,219` contexts) under the imported `H177` two-stage selector calibrated in `H178` admits any of three candidate arms against `R882` ($m = 3$ Holm-Bonferroni) and confirms positive lift on Stage-2 pooled `A`:
1. **`FRESH160`**: `1` fresh-initialized `adapter_10tok` (`29,211` trainable parameters) trained for `160` updates on `FIT_k` (`H180` recipe).
2. **`ENS5`**: `5` independently seeded fresh-initialized `adapter_10tok` members trained for `40` updates each on `FIT_k` (`H179` recipe), averaging candidate cost predictions across the `5` members before `argmin`.
3. **`RENS5`**: Untrained random-structure control using the **exact same `5` fresh-initialized member states** as `ENS5` at `0` updates, averaging candidate cost predictions across the `5` members before `argmin`.

### Setup
| Parameter | Specification |
| :--- | :--- |
| **Reference Arm (`R882`)** | `round-882.pt` (`0` updates; `DL-Val` context utility `1.1377941898164525`, `DL-Val` equal-family utility `1.153615`, `DL-Train` equal-family utility `1.385355`, `1500.00` Elo) |
| **Outer Seeds (`3`)** | `185101, 185102, 185103` |
| **Member Seeds (`5` per outer seed)** | `185201–185205` (`185101`), `185301–185305` (`185102`), `185401–185405` (`185103`) |
| **Cross-Fit Fold Partition (`5` folds)** | `fold_seed = 185140` (`123 FIT / 41 B / 41 A` families per fold; `205` `DL-Train` families, `1,219` contexts; each family appears in `B_k` once and `A_k` once) |
| **Optimizer Updates** | `5,400` total (`(160 + 5 × 40 + 5 × 0) = 360` updates/cell × `15` cells; `90` trained trajectories) |
| **Stage-1 Selector (`Pooled B`)** | Imported `H177_DECISION_RULE` (`pooled_b_select` over `205` families, `10,000` paired family bootstrap draws with `boot_seed = 185141`, within-family 3-seed mean, $m = 3$ Holm: `gain > +0.001 + 1e-12`, `LCB95 > 0`, `holm_p < 0.05`, locked in `pooled_b_selection_lock.json` via `StageOrderGuard` before Stage 2) |
| **Stage-2 Readout (`Pooled A` + `DL-Val`)** | Correlated cross-fold `A` readout (`205` families, `1,219` contexts) + descriptive `DL-Val` (`19` families, `91` contexts) + active diagnostic contrasts (`ENS5 - RENS5`, `FRESH160 - RENS5`, `ENS5 - FRESH160`) |
| **Confirmatory Verdict Rule** | `resolve_h177_verdict(candidate_admitted, h178_receipt)` -> `ADVANCE` iff an arm is admitted on Stage-1 pooled `B` **and** achieves Stage-2 pooled `A` gain $> 0$ with `3/3` positive outer seeds (ties reject); else `NO_ADVANCE` |

### Executive Outcome
- **Gate 0 & Timing Pilot**: Exact `91/91` `DL-Val` `R882` baseline gate passed bit-identically (`1.1377941898164525`, `eval_forward_bit_exact = true`, `static_grad_parity_bit_exact = true`, `static_grad_parity_perturbed_bit_exact = true`, `41` active + `6` `None` adapter grad tensors, `29,211` trainable parameters, frozen `DOPE` SHA-256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` unchanged before/after). The 4-update `FIT_0` timing pilot measured `0.05973 s/update` (projected `1,522.55 s < 9,600 s`), locking full `160`-update `FRESH160` execution (`5,400` total updates).
- **Stage-1 Pooled `B` Selection (`205` Families)**: All three arms achieved positive 3-seed equal-family utility gains across `3/3` outer seeds vs `R882` (`1.385355`):
  - **`ENS5`**: mean gain **`+0.017895`**, 95% paired family bootstrap CI **`[-0.011261, +0.050499]`**, median seed gain **`+0.018177 [-0.011148, +0.051597]`**, one-sided $p = 0.1282$, Holm $m=3$ $p = 0.3846$, Elo **`1509.98 [1496.45, 1524.35]`** -> **`admitted = false` (`selected_step = 0`)**.
  - **`RENS5` (`0` updates)**: mean gain **`+0.017333`**, 95% CI **`[-0.012765, +0.051160]`**, median seed gain **`+0.017101 [-0.012122, +0.051401]`**, one-sided $p = 0.1504$, Holm $m=3$ $p = 0.3846$, Elo **`1507.98 [1494.34, 1522.32]`** -> **`admitted = false` (`selected_step = 0`)**.
  - **`FRESH160`**: mean gain **`+0.007968`**, 95% CI **`[-0.026080, +0.044387]`**, median seed gain **`+0.007634 [-0.028234, +0.044572]`**, one-sided $p = 0.3384$, Holm $m=3$ $p = 0.3846$, Elo **`1507.42 [1493.64, 1522.92]`** -> **`admitted = false` (`selected_step = 0`)**.
- **Critical Diagnostic (`ENS5` vs `RENS5`)**: On Stage-1 pooled `B`, `ENS5 - RENS5` is **`+0.000562 [-0.005213, +0.005580]`** ($p = 0.3998$, `2/3` positive seeds); on Stage-2 unconditional `A`, `ENS5 - RENS5` is **`+0.001925 [-0.001879, +0.005770]`** ($p = 0.1575$, `2/3` positive seeds). Thus **96.9% of `ENS5`'s Stage-1 gain (`+0.017333 / +0.017895`) and 90.0% of its Stage-2 unconditional `A` gain (`+0.017333 / +0.019258`) come from the untrained 5-member random-init ensemble `RENS5` (`0` updates)**—confirming `H180`/`H184`'s finding that fresh-adapter lift on `DL-Train` is predominantly a training-free structural shift (averaging out `R882`'s trained prior/attention adjustments while retaining the fixed revealed-history kernel), with `5 × 40` gradient updates contributing only `+0.0006..+0.0019`.
- **Why `205`-Family Pooled `B` Has `±0.031` CI Half-Width on Real Candidates**: `H178`'s synthetic calibration bank injected homogeneous per-context oracle shifts (`+delta`), yielding low between-family dispersion (`MDE80 = +0.003038`). In contrast, replacing `R882` with fresh-init adapters changes candidate picks on `~50%` of `DL-Train` contexts (`choice_agreement_with_r882 ≈ 0.493–0.503`), creating heavy-tailed between-family gain dispersion (`SD ≈ 0.228` across the `205` families, `SE = SD / sqrt(205) ≈ 0.0159`, 95% CI half-width `≈ ±0.031`). An effect of `+0.0179` therefore achieves $z \approx +1.13$ ($p = 0.1282$), well short of `LCB95 > 0` and Holm $p < 0.05$.
- **Verdict**: **`NO_ADVANCE`** (`0/3` arms admitted on Stage-1 pooled `B`; selected Stage-2 system is unchanged `R882` at `0.000000` gain and `1500.00` Elo; no serving promotion).

---

## 2. Disclosures

1. **Post-`H179`/`H180` Candidate Nomination Caveat**: `FRESH160`, `ENS5`, and `RENS5` were nominated after observing `H179` and `H180` outcomes on the same `205` `DL-Train` families (`1,219` contexts), evaluated here under fresh cyclic cross-fit folds (`fold_seed = 185140`) and fresh seeds (`185101–185103`, member seeds `185201–185405`).
2. **Shared Fold Partition & Correlated Cross-Fold `A` Readout**: All three outer seeds share the 5-fold family partition generated from `fold_seed = 185140` (`123 FIT / 41 B / 41 A` families per fold), so across-seed variation reflects fresh-adapter/member initialization and `FIT` minibatch sampling order. Because `A_k` and `B_k` cycle across the same `205` families and `FIT_k` overlaps by `82/123` families across folds, Stage-2 pooled `A` is a **correlated cross-fold confirmation readout**, not an independent sample split. `DL-Val` (`19` disjoint families, `91` contexts) is reported descriptively.
3. **Stage-2 Read-Only Postprocessing Disclosure**: During `attempt-001`, `runner.py` completed **100% of training (`5,400 / 5,400` optimizer updates across all `15 / 15` cells)**, saved all `165 / 165` `.npz` adapter checkpoints (`15` `FRESH160-step160`, `75` `ENS5-member{1..5}-step40`, `75` `RENS5-member{1..5}-step0`) and all `15 / 15` `.b_only.json` records, and wrote and locked `pooled_b_selection_lock.json` (`SHA-256: eead6066301d3e8a62fd7e7033f8988eed418d26829efac359a9d31138823c53`, `0/3` arms admitted) at `15:06Z`. At `2033.72 s` (`15:16Z`, during Stage-2 read-only `A` scoring at cell `10/15`), `monitor.py` sent `SIGTERM` because a transient 1-minute host load average of `89.51` crossed the `load > 85` check on 128-core `xbabe2` while our process tree was within all resource caps (`cpu = 26.3% <= 85%`, `max_cpu = 62.3%`, `rss = 4.13 GB < 6.0 GB`, `max_rss = 4.57 GB`, `threads = 3 <= 4`, `MemAvailable = 58.35 GiB >= 16 GiB`, `free_disk = 526.47 GB >= 40 GB`, `disk_used = 93.30% < 93.8%`). Synchronous zero-update (`0` optimizer updates) postprocessing (`complete_stage2_postprocess.py`) completed the remaining read-only `A` and `DL-Val` evaluations from the `165` saved `.npz` checkpoints and verified bit-exact equality with `pooled_b_selection_lock.json` (`recount_verification.json`: `status = PASS`, `pooled_b_decisions_bit_exact = true`, `pooled_b_summary_bit_exact = true`).

---

## 3. Primary & Secondary Metrics

### 3.1 Stage-1 Pooled `B` Selection (`205` Families, `1,219` Contexts, $m = 3$ Holm vs `R882`)

| Arm | Prespecified Step | Equal-Family Utility (`3`-Seed Mean `[95% CI]`) | Per-Seed Gain vs `R882` (`185101, 185102, 185103`) | `3`-Seed Mean Gain `[95% CI]` | Median Seed Gain `[95% CI]` (`axis=0`) | Positive Seeds | One-Sided $p$ | Holm $p$ ($m=3$) | Admitted (`selected_step`) | Elo vs `R882` (`Mean [95% CI]` / `Median`) | `-2.0` Picks per Seed | Choice Agreement w/ `R882` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R882`** | `0` | `1.385355 [1.143269, 1.679002]` | `[0.000000, 0.000000, 0.000000]` | `0.000000 [0.000000, 0.000000]` | `0.000000 [0.000000, 0.000000]` | `0/3` | `1.0000` | `1.0000` | Ref (`0`) | `1500.00 [1500.00, 1500.00]` / `1500.00` | `[17, 17, 17]` | `1.0000` |
| **`FRESH160`** | `160` | `1.393323 [1.148712, 1.690246]` | `[+0.007634, +0.003372, +0.012897]` | `+0.007968 [-0.026080, +0.044387]` | `+0.007634 [-0.028234, +0.044572]` | `3/3` | `0.3384` | `0.3846` | **`false` (`0`)** | `1507.42 [1493.64, 1522.92]` / `1507.89` | `[19, 20, 19]` | `0.4810` |
| **`ENS5`** | `40` (`×5`) | `1.403250 [1.157073, 1.699619]` | `[+0.019936, +0.015572, +0.018177]` | `+0.017895 [-0.011261, +0.050499]` | `+0.018177 [-0.011148, +0.051597]` | `3/3` | `0.1282` | `0.3846` | **`false` (`0`)** | `1509.98 [1496.45, 1524.35]` / `1510.18` | `[16, 17, 17]` | `0.4993` |
| **`RENS5`** | `0` (`×5`) | `1.402688 [1.157130, 1.698665]` | `[+0.017101, +0.022746, +0.012152]` | `+0.017333 [-0.012765, +0.051160]` | `+0.017101 [-0.012122, +0.051401]` | `3/3` | `0.1504` | `0.3846` | **`false` (`0`)** | `1507.98 [1494.34, 1522.32]` / `1507.85` | `[16, 15, 19]` | `0.4977` |

### 3.2 Stage-2 Pooled `A` Confirmation (`205` Families, `1,219` Contexts — Selected & Unconditional)

Because `0/3` arms were admitted on Stage-1 pooled `B`, the **Stage-2 Selected** policy for every arm is `selected_step = 0` (`R882`), yielding exact `0.000000 [0.000000, 0.000000]` gain, `1500.00` Elo, and `[17, 17, 17]` `-2.0` picks. The table below reports both **Stage-2 Selected** and **Stage-2 Unconditional** (evaluating each arm's prespecified final state on `A_k`):

| Readout Mode / Arm | Equal-Family Utility (`3`-Seed Mean `[95% CI]`) | Per-Seed Gain vs `R882` (`185101, 185102, 185103`) | `3`-Seed Mean Gain `[95% CI]` | Median Seed Gain `[95% CI]` (`axis=0`) | Positive Seeds | One-Sided $p$ | Holm $p$ ($m=3$) | Elo vs `R882` (`Mean [95% CI]` / `Median`) | `-2.0` Picks per Seed | Choice Agreement w/ `R882` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Selected (`FRESH160` / `ENS5` / `RENS5`)** | `1.385355 [1.143269, 1.679002]` | `[0.000000, 0.000000, 0.000000]` | `0.000000 [0.000000, 0.000000]` | `0.000000 [0.000000, 0.000000]` | `0/3` | `1.0000` | `1.0000` | `1500.00 [1500.00, 1500.00]` / `1500.00` | `[17, 17, 17]` | `1.0000` |
| **Unconditional `FRESH160`** | `1.393334 [1.147600, 1.689723]` | `[+0.009124, +0.000100, +0.014712]` | `+0.007979 [-0.019135, +0.038605]` | `+0.009124 [-0.020072, +0.040067]` | `3/3` | `0.3047` | `0.3555` | `1505.79 [1492.56, 1519.11]` / `1506.87` | `[18, 23, 17]` | `0.4777` |
| **Unconditional `ENS5`** | `1.404614 [1.159009, 1.700966]` | `[+0.020100, +0.020577, +0.017098]` | `+0.019258 [-0.010666, +0.052858]` | `+0.020100 [-0.010735, +0.053052]` | `3/3` | `0.1185` | `0.3555` | `1509.84 [1496.42, 1524.54]` / `1509.87` | `[16, 16, 17]` | `0.4996` |
| **Unconditional `RENS5`** | `1.402688 [1.157130, 1.698665]` | `[+0.017101, +0.022746, +0.012152]` | `+0.017333 [-0.012765, +0.051160]` | `+0.017101 [-0.012122, +0.051401]` | `3/3` | `0.1504` | `0.3555` | `1508.26 [1494.86, 1522.51]` / `1508.14` | `[16, 15, 19]` | `0.4977` |

*(Note: Because `RENS5` has `0` updates and uses the same 5 member initializations per outer seed across all 5 outer folds, its pooled `B` and pooled `A` panels cover the same `205` families with identical predictions.)*

### 3.3 Descriptive `DL-Val` (`19` Disjoint Families, `91` Contexts — Selected & Unconditional)

| Readout Mode / Arm | Equal-Family Utility (`3`-Seed Mean `[95% CI]`) | Context Utility (`3`-Seed Mean) | Per-Seed Equal-Family Gain vs `R882` | `3`-Seed Mean Gain `[95% CI]` | Median Seed Gain `[95% CI]` (`axis=0`) | Positive Seeds | One-Sided $p$ | Elo vs `R882` (`Mean [95% CI]` / `Median`) | `-2.0` Picks per Seed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R882` & Selected Arms** | `1.153615 [0.742260, 1.647608]` | `1.137794` | `[0.000000, 0.000000, 0.000000]` | `0.000000 [0.000000, 0.000000]` | `0.000000 [0.000000, 0.000000]` | `0/3` | `1.0000` | `1500.00 [1500.00, 1500.00]` / `1500.00` | `[0, 0, 0]` |
| **Unconditional `FRESH160`** | `1.114519 [0.695223, 1.612985]` | `1.096976` | `[-0.032593, -0.042922, -0.041773]` | `-0.039096 [-0.112856, +0.007485]` | `-0.041773 [-0.113736, +0.008366]` | `0/3` | `0.9125` | `1504.28 [1460.43, 1557.07]` / `1514.65` | `[1, 1, 1]` |
| **Unconditional `ENS5`** | `1.135467 [0.715387, 1.637372]` | `1.118862` | `[-0.020243, -0.017312, -0.016889]` | `-0.018148 [-0.092223, +0.026155]` | `-0.017312 [-0.091457, +0.026593]` | `0/3` | `0.6469` | `1538.50 [1495.71, 1591.83]` / `1540.50` | `[1, 1, 1]` |
| **Unconditional `RENS5`** | `1.132247 [0.712194, 1.633584]` | `1.115487` | `[-0.020411, -0.018009, -0.025684]` | `-0.021368 [-0.095729, +0.023578]` | `-0.020411 [-0.093612, +0.024906]` | `0/3` | `0.6640` | `1535.44 [1497.24, 1582.52]` / `1531.27` | `[1, 1, 1]` |

*(On `DL-Val`, all three fresh-init arms win pairwise win/loss frequency in Elo (`1504–1538`), but incur `1` invalid (`-2.0`) context pick per seed vs `0` for `R882`, pulling equal-family utility down by `-0.0181` to `-0.0391`.)*

### 3.4 Active Diagnostic Contrasts (`ENS5 - RENS5`, `FRESH160 - RENS5`, `ENS5 - FRESH160`)

| Split | Contrast | Per-Seed Gain (`185101, 185102, 185103`) | `3`-Seed Mean Gain `[95% CI]` | Median Seed Gain `[95% CI]` (`axis=0`) | Positive Seeds | One-Sided $p$ (Mean / Median) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage-1 Pooled `B` (`205` fam)** | `ENS5 - RENS5` | `[+0.002836, -0.007174, +0.006024]` | **`+0.000562 [-0.005213, +0.005580]`** | `+0.002836 [-0.001489, +0.005923]` | `2/3` | `0.3998` / `0.1054` |
| **Stage-1 Pooled `B` (`205` fam)** | `FRESH160 - RENS5` | `[-0.009466, -0.019374, +0.000745]` | **`-0.009365 [-0.024590, +0.003769]`** | `-0.009466 [-0.029581, +0.005168]` | `1/3` | `0.9100` / `0.8835` |
| **Stage-1 Pooled `B` (`205` fam)** | `ENS5 - FRESH160` | `[+0.012302, +0.012200, +0.005280]` | **`+0.009927 [-0.002991, +0.024611]`** | `+0.012200 [-0.003967, +0.029024]` | `3/3` | `0.0745` / `0.0875` |
| **Stage-2 Uncond `A` (`205` fam)** | `ENS5 - RENS5` | `[+0.002999, -0.002169, +0.004946]` | **`+0.001925 [-0.001879, +0.005770]`** | `+0.002999 [-0.001478, +0.005589]` | `2/3` | `0.1575` / `0.0830` |
| **Stage-2 Uncond `A` (`205` fam)** | `FRESH160 - RENS5` | `[-0.007976, -0.022646, +0.002560]` | **`-0.009354 [-0.021364, +0.001465]`** | `-0.007976 [-0.023727, +0.003781]` | `1/3` | `0.9536` / `0.9116` |
| **Stage-2 Uncond `A` (`205` fam)** | `ENS5 - FRESH160` | `[+0.010975, +0.020477, +0.002386]` | **`+0.011279 [+0.000782, +0.022631]`** | `+0.010975 [-0.000879, +0.025468]` | `3/3` | `0.0177` / `0.0350` |
| **Uncond `DL-Val` (`19` fam)** | `ENS5 - RENS5` | `[+0.000168, +0.000698, +0.008795]` | **`+0.003220 [-0.000056, +0.008936]`** | `+0.000698 [-0.000050, +0.003100]` | `3/3` | `0.0332` / `0.1104` |
| **Uncond `DL-Val` (`19` fam)** | `FRESH160 - RENS5` | `[-0.012182, -0.024913, -0.016089]` | **`-0.017728 [-0.031904, -0.005344]`** | `-0.016089 [-0.032906, -0.001448]` | `0/3` | `0.9991` / `0.9860` |
| **Uncond `DL-Val` (`19` fam)** | `ENS5 - FRESH160` | `[+0.012350, +0.025610, +0.024884]` | **`+0.020948 [+0.007530, +0.036286]`** | `+0.024884 [+0.006825, +0.038611]` | `3/3` | `0.0007` / `0.0009` |

### 3.5 Adapter Parameter Drift & Member Disagreement

| Diagnostic Metric | `FRESH160` (`160` updates) | `ENS5` (`5 × 40` updates) | `RENS5` (`5 × 0` updates) |
| :--- | :---: | :---: | :---: |
| **Adapter L2 Drift Across `15` Cells (`median [min, max]`)** | `1.3309 [1.0688, 1.5376]` | `0.2390 [0.2330, 0.2425]` (member mean) | `0.0000 [0.0000, 0.0000]` |
| **Pairwise Member Choice Disagreement (`Pooled B`)** | N/A (`1` member) | `0.3348` | `0.3294` |
| **Pairwise Member Choice Disagreement (`Confirm A`)** | N/A (`1` member) | `0.3364` | `0.3294` |

---

## 4. Versioned S3 Custody (`15` Objects, `21,409,744` Bytes)

All raw artifacts (including `panels.npz`, `pooled_b_selection_lock.json`, `audit.json`, `recount_verification.json`, `baseline.json`, `baseline_costs.npz`, `pilot_budget.json`, `attempt-001.resources.jsonl`, and `h185-checkpoints-and-records.tar.gz` containing all `165` `.npz` checkpoints and `30` per-cell JSON records) are stored in versioned S3 custody under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H185/gpu_tokbench_agy-h185-20261101/` and verified by SHA-256 readback in [custody.json](custody.json):

| Object | Bytes | VersionId | SHA-256 |
| :--- | ---: | :--- | :--- |
| `metrics.json` | `53,343` | `YmWKzafR7WtX36qvdzLFHrrgiPNar2h5` | `0bf93e5dbd33bd2017fe94a64631319cd200121946bc89774a9b6187345add8c` |
| `pooled_b_selection_lock.json` | `12,305` | `2.fLTN.vtwWSUH88CLRcX0sFhnLYHcVE` | `eead6066301d3e8a62fd7e7033f8988eed418d26829efac359a9d31138823c53` |
| `panels.npz` | `612,870` | `aXgQYZOHEo1YcovqjbqxeJiT7TYwRfTE` | `afbe3d1922683a72cde5f875225660b646920e37b073a8c0a0e7e4e6823a79a1` |
| `audit.json` | `169,764` | `ok7.WhYI7w6fnZp9nxqmoKJf7eunBmQ1` | `fe3f3212bc79ab8a9964a2144b8ccbfff829013a620a4df5cb9393286f8c0d7d` |
| `recount_verification.json` | `627` | `gBeH1FeFHRtW.jffTzwByvW9vfX_aclP` | `82de0abb7f7d96252922293bb574aa270f3bfe3f93d56e54acf18e4c938f139f` |
| `baseline.json` | `10,781` | `v9_1CWOS1.BHO5lO2IqIGzgsZiWX5P.Q` | `64fa9f657ff635a335d6e88d0cca4e29abc152b8bbeb71c28e99ae7ea06b3093` |
| `baseline_costs.npz` | `409,829` | `ylvmGD9YmXuGxyICW0NuFaFE1y8JrwQW` | `a5109aff14e9aed679752f9884059adf9ce580f1a2d266c2d0a88fe42194ee38` |
| `pilot_budget.json` | `524` | `9XT_gpmlEeR8sXlgy.0y6zJsvSR1XP9L` | `343afc216fd496a1091f9767443e8d8626ef3aab679c793df92a8e2738ab199e` |
| `attempt-001.stdout` | `693` | `mCtH112AI76gkGnDlyLOnooYoWgf3zMG` | `fabd157cc30d79b4a12574c705da49b2e55b1afea8711b558d7060c445446996` |
| `attempt-001.resources.jsonl` | `679,949` | `mvYxGyY5uFMd4qJaNOIrVaKxsJpFiloJ` | `405a81c4ff35a1f4e81d48c276d864789fd77fcede3da65350d347fd10d1800a` |
| `attempt-001.resource_summary.json` | `128` | `VnmS1I9WKfpvyLRsZjqIa.cjUpDhCGi.` | `cb0d0c6116eb7e0697db724b0f3e34091b9b081709bfa1a7b4d4777700e9abf2` |
| `launch.json` | `770` | `AHHxVmAj60CMYjrCq6taFf266Se542HZ` | `9e958b925cf6d899932281d3c984838da8f47621861ef51536595a25a463f42f` |
| `complete_stage2_postprocess.py` | `30,539` | `hv5QjJj3.FPTthsWkb8BMABT.w6cIaGm` | `a822d59e08e2cd7e0ea63c9d9ac3bd90ea6de4937d335fb36eacda811e748b57` |
| `smoke.json` | `1,987` | `J9_rcLzpc781i_ru.k1XK75yzinGjypF` | `3a516d86f3b92ae8c2a3e918a80ff2949f33460ee393430402b1c1797585341c` |
| `h185-checkpoints-and-records.tar.gz` | `19,425,635` | `h6u1pX1dqNYNQcMKRIDpikgdE7mlPC9D` | `137c025f3274666d4c6f0167475e2243bfff75255d7b98b1a7e33576dd4bc4fd` |

---

## 5. Verdict & Next Step

- **Verdict**: **`NO_ADVANCE`** (`resolve_h177_verdict(False, h178_receipt)` -> `EVALUATED_NO_ADMIT`; `0/3` arms admitted on `205`-family pooled `B`; no serving promotion).
- **Next Step**: Do not promote `FRESH160`, `ENS5`, or `RENS5` over `R882`. Because `RENS5` (`+0.017333` at `0` updates) accounts for `96.9%` of `ENS5`'s gain (`+0.017895` at `5 × 40` updates) and fresh-init replacement induces high between-family variance (`±0.031` CI half-width across `205` families) plus `-2.0` failure picks on `DL-Val`, future work should avoid wholesale fresh-adapter replacement and instead focus on anchored, low-variance corrections that preserve `R882`'s `0`-cliff validity on `DL-Val` while reducing between-family variance.

---

## 6. Verification Commands

```bash
PYTHONPATH=research/flywheel_selfplay/h185_pooled_fresh_ensemble pytest -q research/flywheel_selfplay/h185_pooled_fresh_ensemble/test_h185.py
```
