# Multiclass Classification Benchmarks: Exposure Regimes, Product Baseline & `unified-icl-regbin-v2` OvR

Recorded `2026-11-20`. Every multiclass benchmark result in JopeDime carries an explicit data-exposure
label so generalization to unseen tables, generalist in-corpus scaling under random splits, and
per-table specialist memory remain strictly separated. Pretrained weights referenced below are pinned in
[`models/bundles/unified-icl-regbin-v2.json`](../models/bundles/unified-icl-regbin-v2.json)
(`sha256=cc04e4ab815b82c9f8df58f239c7ce86ac942386744317cd04fd8de70ae3b677` in `s3://veox-jopedime-models-use2`), and
machine-readable per-task scores live in [`assets/multiclass/v1/`](../assets/multiclass/v1/SOURCES.md).

## 1. Exposure Taxonomy & Zero-Sealed-Data Attestation

| Regime | What the evaluated system saw during training | Models in this report | What the regime measures |
| --- | --- | --- | --- |
| **`unseen`** | **Zero** exposures to the evaluated tables (`0` rows, `0` splits, `0` pretrained multiclass weights). | 1) `jopedime` native CPU multiclass roster + cross-fitted Temperature Scaling (`jope.temperature-calibration/v1`) on `datasets/synthetic/multiclass/*` (`10` tasks) and `TabArena v0.1` multiclass (`8` tasks); 2) `UnifiedICLRegBin_M_v2_Stage2A_NoArena_OvR` (`unified_icl_regbin_M_stage2a_noarena.safetensors`) via zero-shot One-vs-Rest + out-of-fold Temperature Scaling on `TabArena v0.1` multiclass (`8` tasks, `pre_arena_stage_arena_exposures = 0`). | Out-of-distribution generalization to unseen multiclass tables. |
| **`in-corpus`** | Evaluated tables included in the `4,996`-table public S3 generalist corpus trained exclusively on **distinct random regression/binary splits** (`0 / 42,352` episodes matched any official `(train_idx, test_idx)` split; `0` duplicate splits; **`0` multiclass objective supervision during pretraining**; `0` per-table fine-tuning). At evaluation only official `(X_train, y_train, X_test)` is read (`y_test` held by scorer). | `UnifiedICLRegBin_M_v2_FullS3_OvR` (`unified_icl_regbin_M.safetensors` One-vs-Rest + out-of-fold Temperature Scaling) and `UnifiedICLRegBin_M_v2_FullS3_OvR + TrainOOF_Hybrid` (simplex blend with classical tabular multiclass estimators + Temperature Scaling fitted strictly on stratified `train` out-of-fold predictions). | Zero-supervision binary-to-multiclass transfer and hybrid generalist scaling across known public domains. |
| **`recall`** | Per-table or per-family specialist fine-tuning on the scored tables' all-row random splits (test rows/labels included in random training splits). | **None (`0` `recall` models claimed or promoted in this release).** | Table-specific memory (documented separately in [`docs/research/waves/h458-h465-black-hole-20261116/context.md`](research/waves/h458-h465-black-hole-20261116/context.md)). |

**Zero Sealed Data (`sealed_inputs_opened: false`):** No sealed S3 bucket
(`s3://jopedime-714044927654-us-east-2-sealed`), no catalog sealed `test.csv`, no
`benchmarks/panel.json` `sealed_rows` (`sealed_sha256`), and no `dev91` `SEALED` / `CONFIRM` /
`FMGAP` split was opened, trained on, or scored. Every task in Sections 3–4 is a public `OpenML`
dataset (`TabArena v0.1` or `BeyondArena`) evaluated on its exact unsubsampled fold-0 split (`0`
rows dropped; `0` external competitor reruns).

## 2. Product CLI Native Multiclass Benchmark (`unseen` Regime — Committed `datasets/synthetic/multiclass`)

Executed on `xbabe1` (`2026-11-20T06:01:03Z`, source commit `de71db702d925ba0191b5c1e25ca1c422a486aae`) using `jopedime fit` / `predict` (`--task multiclass --fitness logloss --calibration auto`) vs `jopedime fit-predict --calibration none` across the official `train` / `test` splits of all `10` committed multiclass datasets in [`datasets/synthetic/multiclass/`](../datasets/synthetic/multiclass/). Every estimator fits strictly from scratch on that dataset's training fold (`0` pretrained weights = **`unseen`**; `sealed_inputs_opened = false`).

- **Mean LogLoss (`--calibration auto`, cross-fitted Temperature):** **`1.0959`** (vs `1.0996` with `--calibration none`)
- **Mean Macro-F1:** **`0.3353`** | **Mean Multiclass Brier:** **`0.6057`** | **Mean Accuracy:** **`0.4125`**

| Dataset | Regime | Classes (`K`) | Train / Test | Selected Method (`logloss`) | Fitted `T` | LogLoss (`auto` Temp) | LogLoss (`none`) | Macro-F1 | Brier |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `multiclass.categorical_input` | `unseen` | 3 | 64 / 16 | `rust.multiclass_ovr_logistic` | 0.0500 | 0.0000 | 0.0002 | 1.0000 | 0.0000 |
| `multiclass.grouped_observations` | `unseen` | 3 | 64 / 16 | `rust.multiclass_ovr_logistic` | 20.0000 | 1.1000 | 1.1144 | 0.1879 | 0.6676 |
| `multiclass.high_dimensional_noise` | `unseen` | 3 | 64 / 16 | `rust.multiclass_ridge` | 20.0000 | 1.0989 | 1.1185 | 0.3152 | 0.6669 |
| `multiclass.imbalance` | `unseen` | 3 | 64 / 16 | `rust.multiclass_diagonal_lda` | 1.0000 | 0.7267 | 0.7267 | 0.2857 | 0.4063 |
| `multiclass.many_classes` | `unseen` | 7 | 64 / 16 | `rust.multiclass_ridge` | 20.0000 | 1.9445 | 1.9204 | 0.2041 | 0.8567 |
| `multiclass.missing_values` | `unseen` | 3 | 64 / 16 | `rust.multiclass_ridge` | 20.0000 | 1.0992 | 1.1106 | 0.2511 | 0.6670 |
| `multiclass.nonlinear_boundaries` | `unseen` | 4 | 64 / 16 | `rust.multiclass_diagonal_lda` | 1.6842 | 1.3340 | 1.3446 | 0.2071 | 0.7277 |
| `multiclass.overlapping_classes` | `unseen` | 3 | 64 / 16 | `rust.multiclass_ridge` | 20.0000 | 1.0983 | 1.0955 | 0.2222 | 0.6665 |
| `multiclass.rare_class` | `unseen` | 5 | 64 / 16 | `rust.multiclass_diagonal_lda` | 1.4393 | 1.4596 | 1.4664 | 0.2571 | 0.7319 |
| `multiclass.three_classes` | `unseen` | 3 | 64 / 16 | `rust.multiclass_ridge` | 1.0000 | 1.0983 | 1.0983 | 0.4226 | 0.6662 |

## 3. Exact Fold-0 Metrics Across Public `TabArena` & `BeyondArena` (`33` Public Multiclass Tasks)

Evaluated across all `8` `TabArena v0.1` multiclass tasks (`K = 3..8`, up to `52,035` train / `26,018` test rows) and all `25` `BeyondArena` multiclass tasks (`K = 3..20`, up to `1,000,000` train / `226,140` test rows and `12,600` features):

| Suite | Model | Regime | Tasks | Mean LogLoss | Median LogLoss | Mean Macro-F1 | Median Macro-F1 | Mean Brier | Mean Acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TabArena_multiclass` | `jopedime_native_multiclass_auto` | **`unseen`** | 8 | 0.70133 | 0.85732 | 0.47737 | 0.46737 | 0.38171 | 0.72833 |
| `TabArena_multiclass` | **`UnifiedICLRegBin_M_v2_Stage2A_NoArena_OvR`** | **`unseen`** | 8 | **0.40184** | **0.38417** | **0.61117** | **0.63082** | **0.20706** | **0.85998** |
| `TabArena_multiclass` | `UnifiedICLRegBin_M_v2_FullS3_OvR` | **`in-corpus`** | 8 | 0.39783 | 0.36615 | 0.60606 | 0.62809 | 0.20736 | 0.85707 |
| `TabArena_multiclass` | **`UnifiedICLRegBin_M_v2_FullS3_OvR + TrainOOF_Hybrid`** | **`in-corpus`** | 8 | **0.29445** | **0.21838** | **0.72091** | **0.86339** | **0.14449** | **0.90746** |
| `BeyondArena_multiclass` | `UnifiedICLRegBin_M_v2_FullS3_OvR` | **`in-corpus`** | 25 | 0.73715 | 0.62267 | 0.57763 | 0.57991 | 0.34716 | 0.74227 |
| `BeyondArena_multiclass` | **`UnifiedICLRegBin_M_v2_FullS3_OvR + TrainOOF_Hybrid`** | **`in-corpus`** | 25 | **0.64642** | **0.56117** | **0.64376** | **0.68197** | **0.30050** | **0.77850** |

## 4. Bradley-Terry Field Elo Standings (`RF (default) = 1000.0`, `1,000` Bootstrap CIs)

Opponent predictions come directly from the SHA-pinned public `TabArena v0.1` and `BeyondArena`
field parquets (`metric == "log_loss"`, `0` external competitor reruns). Elo ratings are calibrated per
panel at `RF (default) = 1000.0` via `bencheval.EloHelper` (`sha256=12a8a1ee`):

| Panel | Entrant | Regime | Tasks | Elo (95% Bootstrap CI, Rank) | Reference Baselines on Same Panel |
| --- | --- | --- | ---: | --- | --- |
| `TabArena_multiclass` (Zero-Tuning / Default, 32 entrants) | **`UnifiedICLRegBin_M_v2_Stage2A_NoArena_OvR`** | **`unseen`** | 8 | **914.3 ([219.8, 1297.3], #28/32)** | `RF (default)` 1000.0, `XT (default)` 907.7, `LR (default)` 747.4 |
| `TabArena_multiclass` (Zero-Tuning / Default, 32 entrants) | `UnifiedICLRegBin_M_v2_FullS3_OvR` | **`in-corpus`** | 8 | 914.3 ([313.8, 1296.9], #29/32) | same |
| `TabArena_multiclass` (Zero-Tuning / Default, 32 entrants) | `jopedime_native_multiclass_auto` | **`unseen`** | 8 | 629.6 ([-285.1, 852.1], #32/32) | same |
| `TabArena_multiclass` (Full Field incl. 4h HPO, 79 entrants) | **`UnifiedICLRegBin_M_v2_Stage2A_NoArena_OvR`** | **`unseen`** | 8 | **994.4 ([308.9, 1393.5], #72/79)** | `RF (default)` 1000.0, `LR (tuned + ensemble)` 947.5, `LR (tuned)` 913.6, `XT (default)` 908.2 |
| `TabArena_multiclass` (Full Field incl. 4h HPO, 79 entrants) | **`UnifiedICLRegBin_M_v2_FullS3_OvR + TrainOOF_Hybrid`** | **`in-corpus`** | 8 | **994.4 ([754.9, 1209.7], #73/79)** | same |
| `BeyondArena_multiclass` (Zero-Tuning / Default, 21 entrants) | `UnifiedICLRegBin_M_v2_FullS3_OvR` | **`in-corpus`** | 25 | **891.1 ([716.6, 1033.9], #20/21)** | `RF (default)` 1000.0, `LR (default)` 917.4, `XT (default)` 874.5 |
| `BeyondArena_multiclass` (Full Field incl. HPO, 38 entrants) | **`UnifiedICLRegBin_M_v2_FullS3_OvR + TrainOOF_Hybrid`** | **`in-corpus`** | 25 | **1091.2 ([960.0, 1226.7], #28/38)** | `RF (tuned + ensemble)` 1113.7, `LR (tuned + ensemble)` 1069.3, `RF (tuned)` 1063.8, `RF (default)` 1000.0, `LR (default)` 936.1, `XT (default)` 882.3 |
| `Combined_Multiclass_Arena` (`TabArena + BeyondArena`, 32 entrants) | **`UnifiedICLRegBin_M_v2_FullS3_OvR + TrainOOF_Hybrid`** | **`in-corpus`** | 33 / 33 | **1077.2 ([973.6, 1189.7], #21/32)** | `RF (tuned)` 1091.7, `LR (tuned + ensemble)` 1037.0, `LR (tuned)` 1017.3, `RF (default)` 1000.0, `LR (default)` 900.5, `XT (default)` 892.0 |
| `Combined_Multiclass_Arena` (`TabArena + BeyondArena`, 32 entrants) | `UnifiedICLRegBin_M_v2_FullS3_OvR` | **`in-corpus`** | 33 / 33 | 873.0 ([710.5, 1006.8], #30/32) | same |
| `Combined_Multiclass_Arena` (`TabArena + BeyondArena`, 32 entrants) | `UnifiedICLRegBin_M_v2_Stage2A_NoArena_OvR` | **`unseen`** (on 8 TA) | 8 / 33 | 850.9 ([411.1, 1167.6], #31/32) | same |

## 5. Reproducing & Verifying Committed Evidence

```sh
# 1. Verify committed model bundle manifest & SHA-256 pins (offline, 0 network):
bash deploy/ci/manifests.sh
cargo run --locked -p jope-models -- verify --manifests-only

# 2. Inspect machine-readable multiclass benchmark ladders and per-task fold-0 CSV:
python3 -m json.tool assets/multiclass/v1/RESULTS.json > /dev/null
head -n 20 assets/multiclass/v1/per-task-scores.csv
```
