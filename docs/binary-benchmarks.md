# Binary Classification Benchmarks: Exposure Regimes, Product Baseline & `unified-icl-regbin-v2`

Recorded `2026-11-20`. Every benchmark score in JopeDime carries an explicit data-exposure label
(`unseen`, `in-corpus`, or `recall`) so generalization to unseen tables, generalist in-corpus scaling
under random splits, and per-table specialist memory are never conflated. All weights referenced below
are pinned in [`models/bundles/unified-icl-regbin-v2.json`](../models/bundles/unified-icl-regbin-v2.json)
(`sha256=cc04e4ab815b82c9f8df58f239c7ce86ac942386744317cd04fd8de70ae3b677` in `s3://veox-jopedime-models-use2`),
machine-readable per-task scores live in
[`assets/binary/unified-icl-regbin-v2/`](../assets/binary/unified-icl-regbin-v2/SOURCES.md), and the
immutable `schema-registry/v1` S3 benchmark evaluation campaign (`406` objects, `813,753` B) is
published in `s3://veox-jopedime-learning-use2` (`COMMIT.7d04c2c005e148f8.json`).

## 1. Exposure Taxonomy & Zero-Sealed-Data Attestation

| Regime | What the evaluated system saw during training | Models in this report | What the regime measures |
| --- | --- | --- | --- |
| **`unseen`** | **Zero** exposures to the evaluated tables or any `C2` / `BENCH_OVERLAP` `S3_CATALOG` counterpart (`0` rows, `0` splits). | 1) `jopedime_native_binary_auto` (`jopedime` CPU binary roster + cross-fitted Platt scaling, `0` pretrained weights) on `datasets/synthetic/binary/*` (`10` tasks) and `TabArena v0.1` binary (`30` tasks); 2) `Step0_Warmstart_r5_M` on `TabArena v0.1` binary (`30` tasks; upstream `r5` trained on regression tables only, `0` binary/multiclass tables) and on the `5` `BENCH_OVERLAP` `in_r5 == "none"` `TabArena` regression tasks; 3) `UnifiedICLRegBin_M_v2_Stage2A_NoArena` on the `2` `TabArena` regression tasks with `0` `r5` and `0` `S3_CATALOG` overlap (`QSAR_fish_toxicity`, `wine_quality`). | Out-of-distribution generalization to unseen tables. |
| **`in-corpus`** | Evaluated tables (or older public `T1` `S3_CATALOG` OpenML counterparts identified by the `C2` / `BENCH_OVERLAP` screens) included in the `4,996`-table public `T1` S3 generalist corpus trained exclusively on **distinct random splits** (`0 / 42,352` episodes matched any official `(train_idx, test_idx)` split; `0` duplicate splits; `0` per-table specialist fine-tuning). At evaluation only official `(X_train, y_train, X_test)` is read (`y_test` held by scorer). | `UnifiedICLRegBin_M_v2_FullS3` (`160 / 160` Arena tasks: `103` binary + `57` regression), `UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid` (`30` `TabArena` binary + `13` `TabArena` regression), `UnifiedICLRegBin_Ensemble_Mv2_L_v2_Val0_FullS3` & `+ TrainOOF_Hybrid` (`20` `TabArena` binary + `13` `TabArena` regression), `UnifiedICLRegBin_M_v2_Stage2A_NoArena` on `TabArena` binary (`30` tasks; `pre_arena_stage_arena_exposures = 0`, reclassified from `unseen` to `in-corpus` because the `C2` screen identified older public `T1` `S3_CATALOG` OpenML copies) and `11` `TabArena` regression tasks, and `Step0_Warmstart_r5_M` on the `8` `TabArena` regression tasks present in `r5`. | Multi-table generalist scaling across known public domains without per-table fine-tuning. |
| **`recall`** | Per-table or per-family specialist fine-tuning on the scored tables' all-row random splits (test rows/labels included in random training splits). | **None (`0` `recall` models claimed or promoted in this release).** | Table-specific memory (documented separately in [`docs/research/waves/h458-h465-black-hole-20261116/context.md`](research/waves/h458-h465-black-hole-20261116/context.md)). |

**Zero Sealed Data (`sealed_inputs_opened: false`):** No sealed S3 bucket
(`s3://jopedime-714044927654-us-east-2-sealed`, `T2`/`T3`), no `benchmarks/panel.json` `sealed_rows`
(`sealed_sha256`), and no `dev91` `SEALED` / `CONFIRM` / `FMGAP` evaluation split was opened, trained
on, or scored. Every task in Sections 3–4 is a public `OpenML` dataset (`TabArena v0.1` or
`BeyondArena`) evaluated on its exact unsubsampled fold-0 split (`0` rows dropped; `0` external
competitor reruns), and generalist corpus pretraining read only public `T1`
`s3://veox-jopedime-datasets-use2` tables under distinct random splits.

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

All `160` public OpenML tasks in the Unified Regression + Binary Arena (`103` binary classification +
`57` regression preservation control; together with the `33` multiclass Arena tasks in
[`docs/multiclass-benchmarks.md`](multiclass-benchmarks.md), this covers `193 / 193 = 100%` of public
`TabArena v0.1` + `BeyondArena` tasks) were scored on exact fold-0 splits:

### Binary Classification (`30` `TabArena v0.1` + `73` `BeyondArena` = `103` Public Tasks)

| Suite | Model | Regime | Tasks | Mean ROC-AUC | Median ROC-AUC | Mean LogLoss | Mean Brier | Mean Acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `TabArena_binary` | `jopedime_native_binary_auto` | **`unseen`** | 30 | 0.68840 | 0.71265 | 0.44525 | 0.13111 | 0.81802 |
| `TabArena_binary` | **`Step0_Warmstart_r5_M`** | **`unseen`** | 30 | **0.81776** | **0.80414** | **0.32779** | **0.10152** | **0.85968** |
| `TabArena_binary` | `UnifiedICLRegBin_M_v2_Stage2A_NoArena` | **`in-corpus`** (`C2`*) | 30 | 0.80792 | 0.80201 | 0.34209 | 0.10545 | 0.85281 |
| `TabArena_binary` | **`UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid`** | **`in-corpus`** | 30 | **0.83568** | **0.80519** | **0.31716** | **0.09789** | **0.86506** |
| `TabArena_binary` | `UnifiedICLRegBin_M_v2_FullS3` | **`in-corpus`** | 30 | 0.81170 | 0.80070 | 0.33962 | 0.10524 | 0.85377 |
| `BeyondArena_binary` | **`UnifiedICLRegBin_M_v2_FullS3`** | **`in-corpus`** | 73 | **0.78363** | **0.78128** | **0.34485** | **0.10730** | **0.84874** |

### Regression Preservation Control (`13` `TabArena v0.1` + `44` `BeyondArena` = `57` Public Tasks)

To verify that adding dual-task binary classification does not degrade the shared regression trunk,
all `57` public Arena regression tasks were scored on their exact fold-0 splits:

| Suite | Model | Per-Task Regime Breakdown (`per-task-scores.csv`) | Tasks | Median NRMSE | Mean NRMSE | Median R² |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `TabArena_regression` | `Step0_Warmstart_r5_M` | `5` **`unseen`** (`in_r5=none`) + `8` **`in-corpus`** (`in_r5` present) | 13 | 0.37710 | 0.46603 | 0.85780 |
| `TabArena_regression` | `UnifiedICLRegBin_M_v2_Stage2A_NoArena` | `2` **`unseen`** + `11` **`in-corpus`** (`r5` / `S3_CATALOG` overlap) | 13 | 0.37799 | 0.46451 | 0.85713 |
| `TabArena_regression` | **`UnifiedICLRegBin_M_v2_FullS3`** | **`in-corpus`** (`13/13`) | 13 | **0.37296** | 0.46267 | **0.86090** |
| `TabArena_regression` | **`UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid`** | **`in-corpus`** (`13/13`) | 13 | **0.34901** | **0.44012** | **0.87819** |
| `BeyondArena_regression` | **`UnifiedICLRegBin_M_v2_FullS3`** | **`in-corpus`** (`44/44`) | 44 | **0.64864** | **0.60318** | **0.57926** |

*\*Note (`C2` screen): `Stage2A_NoArena` had `0` direct `TabArena`/`BeyondArena` exposures (`pre_arena_stage_arena_exposures = 0`), but the `C2` classification screen identified older public `T1` `S3_CATALOG` OpenML copies for `29/30` `TabArena_binary` datasets (`1` additional flagged); every row in `per-task-scores.csv` therefore carries `exposure_regime = "in-corpus"` with an explicit `exposure_note`.*

## 4. Bradley-Terry Field Elo Standings (`RF (default) = 1000.0`, `1,000` Bootstrap CIs)

Opponent predictions come directly from the SHA-pinned public `TabArena v0.1` and `BeyondArena`
field parquets (`0` external reruns). All `10` Bradley-Terry Elo ladders are calibrated per panel
(`RF (default) = 1000.0`) via `bencheval.EloHelper` (`sha256=12a8a1ee`) and stored in full in
[`assets/binary/unified-icl-regbin-v2/RESULTS.json`](../assets/binary/unified-icl-regbin-v2/RESULTS.json):

| Panel | Entrant | Regime | Tasks | Elo (95% Bootstrap CI, Rank) | Reference Baselines on Same Panel |
| --- | --- | --- | ---: | --- | --- |
| `TabArena_binary` (Zero-Tuning / Default, 34 entrants) | **`Step0_Warmstart_r5_M`** | **`unseen`** | 30 | **1035.2 ([881.2, 1176.6], #27/34)** | `XT (default)` 1006.0, `RF (default)` 1000.0, `LR (default)` 920.7 |
| `TabArena_binary` (Zero-Tuning / Default, 34 entrants) | `UnifiedICLRegBin_M_v2_FullS3` | **`in-corpus`** | 30 | 897.5 ([735.5, 1027.6], #31/34) | same |
| `TabArena_binary` (Zero-Tuning / Default, 34 entrants) | `UnifiedICLRegBin_M_v2_Stage2A_NoArena` | **`in-corpus`** | 30 | 884.5 ([702.2, 1024.3], #32/34) | same |
| `TabArena_binary` (Zero-Tuning / Default, 34 entrants) | `jopedime_native_binary_auto` | **`unseen`** | 30 | 672.4 ([389.2, 851.7], #33/34) | same |
| `TabArena_binary` (Full Field incl. 4h HPO, 82 entrants) | **`UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid`** | **`in-corpus`** | 30 | **1075.7 ([936.2, 1201.4], #71/82)** | `RF (tuned)` 1090.0, `XT (default)` 1021.0, `RF (default)` 1000.0, `LR (tuned + ensemble)` 996.2 |
| `TabArena_binary` (Full Field incl. 4h HPO, 82 entrants) | **`Step0_Warmstart_r5_M`** | **`unseen`** | 30 | **1058.2 ([901.1, 1195.0], #72/82)** | same |
| `BeyondArena_binary` (Zero-Tuning / Default, 21 entrants) | `UnifiedICLRegBin_M_v2_FullS3` | **`in-corpus`** | 73 | 886.6 ([788.3, 985.2], #21/21) | `RF (default)` 1000.0, `XT (default)` 968.5, `LR (default)` 950.5 |
| `BeyondArena_binary` (Full Field incl. HPO, 37 entrants) | `UnifiedICLRegBin_M_v2_FullS3` | **`in-corpus`** | 73 | 903.3 ([807.7, 999.3], #37/37) | `RF (tuned)` 1038.7, `RF (default)` 1000.0, `LR (tuned + ensemble)` 983.9, `XT (default)` 978.3 |
| `Combined_Binary_Arena` (`TabArena + BeyondArena`, 35 entrants) | **`UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid`** | **`in-corpus`** | 30 / 103 | **1039.2 ([936.9, 1133.1], #24/35)** | `RF (tuned)` 1060.4, `RF (default)` 1000.0, `LR (tuned + ensemble)` 995.5, `XT (default)` 993.8 |
| `Combined_Binary_Arena` (`TabArena + BeyondArena`, 35 entrants) | **`Step0_Warmstart_r5_M`** | **`unseen`** (on 30 TA) | 30 / 103 | **994.1 ([869.1, 1106.2], #27/35)** | same |
| `Combined_Binary_Arena` (`TabArena + BeyondArena`, 35 entrants) | `UnifiedICLRegBin_M_v2_FullS3` | **`in-corpus`** | **103 / 103** | 907.8 ([832.3, 982.0], #32/35) | same |
| `Combined_Unified_RegBin_Arena` (`103` Bin + `57` Reg = `160` tasks, 35 entrants) | **`UnifiedICLRegBin_M_v2_FullS3 + TrainOOF_Hybrid`** | **`in-corpus`** | 43 / 160 | **1068.2 ([987.9, 1140.1], #24/35)** | `RF (tuned)` 1082.0, `XT (default)` 1019.3, `RF (default)` 1000.0, `LR (tuned + ensemble)` 972.6, `LR (default)` 900.7 |
| `Combined_Unified_RegBin_Arena` (`103` Bin + `57` Reg = `160` tasks, 35 entrants) | **`UnifiedICLRegBin_M_v2_FullS3`** | **`in-corpus`** | **160 / 160** | **981.2 ([922.9, 1039.5], #29/35)** | same |

## 5. Reproducing & Verifying Committed & S3 Evidence

```sh
# 1. Verify committed model bundle manifest & SHA-256 pins (offline, 0 network):
bash deploy/ci/manifests.sh
cargo run --locked -p jope-models -- verify --manifests-only

# 2. Fetch the pinned v2 weights from s3://veox-jopedime-models-use2 and re-verify hashes:
AWS_PROFILE=veox-jepson-prod cargo run --locked -p jope-models -- fetch unified-icl-regbin-v2
cargo run --locked -p jope-models -- verify unified-icl-regbin-v2

# 3. Verify published S3 benchmark campaign objects & COMMIT marker via aws s3 ls:
AWS_PROFILE=veox-jepson-prod aws s3 ls s3://veox-jopedime-learning-use2/experiments/public-benchmark-evaluations/v1/campaigns/unified-icl-regbin-v2-arena160-20261120/ --recursive

# 4. Inspect machine-readable benchmark ladders and strict regime-labeled CSV (395 rows):
python3 -m json.tool assets/binary/unified-icl-regbin-v2/RESULTS.json > /dev/null
head -n 20 assets/binary/unified-icl-regbin-v2/per-task-scores.csv
```
