# Binary Classification Benchmarks: Exposure Regimes, Product Baseline & `unified-icl-regbin-v2`

Recorded `2026-11-20`. Every benchmark score in JopeDime carries an explicit data-exposure label so
generalization to unseen tables, generalist in-corpus scaling under random splits, and per-table
specialist memory are never conflated. All weights referenced below are pinned in
[`models/bundles/unified-icl-regbin-v2.json`](../models/bundles/unified-icl-regbin-v2.json)
(`sha256=cc04e4ab815b82c9f8df58f239c7ce86ac942386744317cd04fd8de70ae3b677` in `s3://veox-jopedime-models-use2`), and
machine-readable per-task scores live in
[`assets/binary/unified-icl-regbin-v2/`](../assets/binary/unified-icl-regbin-v2/SOURCES.md).

## 1. Exposure Taxonomy & Zero-Sealed-Data Attestation

| Regime | What the evaluated system saw during training | Models in this report | What the regime measures |
| --- | --- | --- | --- |
| **`unseen`** | **Zero** exposures to the evaluated tables (`0` rows, `0` splits). | 1) `jopedime` native CPU binary roster + cross-fitted Platt scaling on `datasets/synthetic/binary/*`; 2) `Step0_Warmstart_r5_M` & `UnifiedICLRegBin_M_v2_Stage2A_NoArena` (`unified_icl_regbin_M_stage2a_noarena.safetensors`) on the 30 `TabArena v0.1` binary tasks (ingested `2026-11-18` after `r5`, `pre_arena_stage_arena_exposures = 0`). | Out-of-distribution generalization to unseen tables. |
| **`in-corpus`** | Evaluated tables included in the `4,996`-table public S3 generalist corpus (`~3.2%` table share), trained exclusively on **distinct random splits** (`0 / 42,352` episodes matched any official `(train_idx, test_idx)` split; `0` duplicate splits; `0` per-table specialist fine-tuning). Full unsplit public table CSVs participate in random training splits; at evaluation only official `(X_train, y_train, X_test)` is read (`y_test` held by scorer). | `UnifiedICLRegBin_M_v2_FullS3` (`unified_icl_regbin_M.safetensors`), `UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid` (blended on training out-of-fold predictions only), and `UnifiedICLRegBin_Ensemble_Mv2_L_v2` (`M` + `L` `134M`). | Multi-table generalist scaling across known public domains without per-table fine-tuning. |
| **`recall`** | Per-table or per-family specialist fine-tuning on the scored tables' all-row random splits (test rows/labels included in random training splits). | **None (`0` `recall` models claimed or promoted in this release).** | Table-specific memory (documented separately in [`docs/research/waves/h458-h465-black-hole-20261116/context.md`](research/waves/h458-h465-black-hole-20261116/context.md)). |

**Zero Sealed Data (`sealed_inputs_opened: false`):** No sealed S3 bucket
(`s3://jopedime-714044927654-us-east-2-sealed`), no catalog sealed `test.csv`, no
`benchmarks/panel.json` `sealed_rows` (`sealed_sha256`), and no `dev91` `SEALED` / `CONFIRM` /
`FMGAP` split was opened, trained on, or scored. Every task in Sections 3–4 is a public `OpenML`
dataset (`TabArena v0.1` or `BeyondArena`) evaluated on its exact unsubsampled fold-0 split (`0`
rows dropped; `0` external competitor reruns).

## 2. Product CLI Native Binary Benchmark (`unseen` Regime — Committed `datasets/synthetic/binary`)

Executed on `xbabe1` (`2026-11-20T00:55:20Z`, source commit `bf47ee09c49fc1613f194d528843c1d354d62591`) using `jopedime fit-predict --task binary --fitness logloss` over the official `train` / `test` splits of the 10 committed binary datasets in [`datasets/synthetic/binary/`](../datasets/synthetic/binary/). Every estimator fits strictly from scratch on that dataset's training fold (`0` pretrained weights, `0` prior exposures = **`unseen`**; `sealed_inputs_opened = false`).

- **Mean LogLoss (`--calibration auto`, cross-fitted Platt):** **`0.2802`** (vs `0.3089` with `--calibration none`; on `binary.nonlinear` with `rust.extra_trees`, cross-fitted Platt reduces held-out log-loss from `0.5003` to `0.2311`)
- **Mean ROC-AUC:** **`0.8959`** | **Mean Brier:** **`0.0790`**

| Dataset | Regime | Train / Test | Selected Method (`logloss`) | LogLoss (`auto` Platt) | LogLoss (`none`) | ROC-AUC | Brier |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `binary.categorical` | `unseen` | 64 / 32 | `rust.logit_ridge` | 0.0012 | 0.0012 | 1.0000 | 0.0000 |
| `binary.constant_features` | `unseen` | 64 / 32 | `rust.logit_ridge` | 0.0609 | 0.0609 | 1.0000 | 0.0164 |
| `binary.duplicated_rows` | `unseen` | 64 / 32 | `rust.logit_ridge` | 0.0827 | 0.0827 | 1.0000 | 0.0164 |
| `binary.grouped_observations` | `unseen` | 64 / 32 | `rust.logit_ridge` | 1.0555 | 1.0555 | 0.9883 | 0.2358 |
| `binary.imbalanced` | `unseen` | 64 / 32 | `rust.logit_ridge` | 0.2415 | 0.2689 | 0.5000 | 0.0598 |
| `binary.linear` | `unseen` | 64 / 32 | `rust.logit_ridge` | 0.0493 | 0.0692 | 1.0000 | 0.0063 |
| `binary.missing_values` | `unseen` | 64 / 32 | `rust.logit_ridge` | 0.1364 | 0.1364 | 0.9879 | 0.0445 |
| `binary.nonlinear` | `unseen` | 64 / 32 | `rust.extra_trees` | 0.2311 | 0.5003 | 0.9727 | 0.0720 |
| `binary.outliers` | `unseen` | 64 / 32 | `rust.logit_ridge` | 0.1995 | 0.1703 | 0.9879 | 0.0660 |
| `binary.overlapping` | `unseen` | 64 / 32 | `rust.portfolio` | 0.7439 | 0.7439 | 0.5223 | 0.2728 |

## 3. Exact Fold-0 Metrics Across Public `TabArena` & `BeyondArena` (`160` Public Tasks)

### Binary Classification (`30` `TabArena v0.1` + `73` `BeyondArena` = `103` Public Tasks)

| Suite | Model | Regime | Tasks | Mean ROC-AUC | Median ROC-AUC | Mean LogLoss | Mean Brier | Mean Acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `TabArena_binary` | `Step0_Warmstart_r5_M` | **`unseen`** | 30 | 0.81776 | 0.80414 | 0.32779 | 0.10152 | 0.85968 |
| `TabArena_binary` | `UnifiedICLRegBin_M_v2_Stage2A_NoArena` | **`unseen`** | 30 | 0.80792 | 0.80201 | 0.34209 | 0.10545 | 0.85281 |
| `TabArena_binary` | **`UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid`** | **`in-corpus`** | 30 | **0.83568** | **0.80519** | **0.31716** | **0.09789** | **0.86506** |
| `TabArena_binary` | `UnifiedICLRegBin_M_v2_FullS3` | **`in-corpus`** | 30 | 0.81170 | 0.80070 | 0.33962 | 0.10524 | 0.85377 |
| `BeyondArena_binary` | **`UnifiedICLRegBin_M_v2_FullS3`** | **`in-corpus`** | 73 | **0.78363** | **0.78128** | **0.34485** | **0.10730** | **0.84874** |

### Regression Preservation Control (`13` `TabArena v0.1` + `44` `BeyondArena` = `57` Public Tasks)

To verify that adding dual-task binary classification does not degrade the shared regression trunk,
all `57` public Arena regression tasks were scored on their exact fold-0 splits:

| Suite | Model | Regime | Tasks | Median NRMSE | Mean NRMSE | Median R² |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `TabArena_regression` | `Step0_Warmstart_r5_M` | `unseen_v2_curriculum`* | 13 | 0.37710 | 0.46603 | 0.85780 |
| `TabArena_regression` | `UnifiedICLRegBin_M_v2_Stage2A_NoArena` | `unseen_v2_curriculum`* | 13 | 0.37799 | 0.46451 | 0.85713 |
| `TabArena_regression` | **`UnifiedICLRegBin_M_v2_FullS3`** | **`in-corpus`** | 13 | **0.37296** | 0.46267 | **0.86090** |
| `BeyondArena_regression` | **`UnifiedICLRegBin_M_v2_FullS3`** | **`in-corpus`** | 44 | **0.64864** | **0.60318** | **0.57926** |

*Note: `unseen_v2_curriculum` indicates `0` exposures during `v2` Phase 1 + Stage 2A (`pre_arena_stage_arena_exposures = 0`), while upstream `r5` regression pretraining did not log per-episode split hashes.*

## 4. Bradley-Terry Field Elo Standings (`RF (default) = 1000.0`, `1,000` Bootstrap CIs)

Opponent predictions come directly from the SHA-pinned public `TabArena v0.1` and `BeyondArena`
field parquets (`0` external reruns). Elo values are calibrated per panel (`RF (default) = 1000.0`)
and are compared only within the same opponent pool:

| Panel | Entrant | Regime | Tasks | Elo (95% Bootstrap CI, Rank) | Reference Baselines on Same Panel |
| --- | --- | --- | ---: | --- | --- |
| `TabArena_binary` (Zero-Tuning / Default, 33 entrants) | `Step0_Warmstart_r5_M` | **`unseen`** | 30 | **1036.4 ([878.5, 1179.9], #27/33)** | `XT (default)` 1006.2, `RF (default)` 1000.0, `LR (default)` 918.5 |
| `TabArena_binary` (Zero-Tuning / Default, 33 entrants) | `UnifiedICLRegBin_M_v2_Stage2A_NoArena` | **`unseen`** | 30 | 875.5 ([686.1, 1023.3], #32/33) | same |
| `TabArena_binary` (Full Field incl. 4h HPO, 81 entrants) | **`UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid`** | **`in-corpus`** | 30 | **1075.0 ([934.2, 1202.8], #71/81)** | `XT (default)` 1021.4, `RF (default)` 1000.0, `LR (tuned + ensemble)` 996.1 |
| `Combined_Binary_Arena` (`TabArena + BeyondArena`, 34 entrants) | **`UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid`** | **`in-corpus`** | 30 / 103 | **1036.8 ([932.3, 1131.9], #24/34)** | `RF (tuned)` 1060.2, `RF (default)` 1000.0, `LR (tuned + ensemble)` 995.5, `XT (default)` 993.7 |
| `Combined_Binary_Arena` (`TabArena + BeyondArena`, 34 entrants) | `Step0_Warmstart_r5_M` | **`unseen`** (on 30 TA) | 30 / 103 | 993.7 ([867.2, 1107.0], #27/34) | same |
| `Combined_Binary_Arena` (`TabArena + BeyondArena`, 34 entrants) | `UnifiedICLRegBin_M_v2_FullS3` | **`in-corpus`** | 103 / 103 | 906.8 ([830.6, 981.3], #32/34) | same |

## 5. Reproducing & Verifying Committed Evidence

```sh
# 1. Verify committed model bundle manifest & SHA-256 pins (offline, 0 network):
bash deploy/ci/manifests.sh
cargo run --locked -p jope-models -- verify --manifests-only

# 2. Fetch the pinned v2 weights from s3://veox-jopedime-models-use2 and re-verify hashes:
AWS_PROFILE=veox-jepson-prod cargo run --locked -p jope-models -- fetch unified-icl-regbin-v2
cargo run --locked -p jope-models -- verify unified-icl-regbin-v2

# 3. Inspect machine-readable benchmark ladders and per-task fold-0 CSV:
python3 -m json.tool assets/binary/unified-icl-regbin-v2/RESULTS.json > /dev/null
head -n 20 assets/binary/unified-icl-regbin-v2/per-task-scores.csv
```
