# Binary classification: objectives, calibration, service runs, and source export

Probability calibration, class-order authentication, and ranking metrics behave differently from
point-error regression. A credit risk scorer cares about ranking (`roc_auc`) or rare-event precision
(`pr_auc`), whereas a pricing or automated triage decision requires calibrated probabilities scored
by a strictly proper rule (`logloss` or `brier`). JopeDime binds the exact two-class order, objective
specification, out-of-fold Platt calibration, and multi-language predictor emission into one
authenticated lifecycle across the offline CLI, the local HTTP/MCP engine, and persisted
`jope.model-object/v1` artifacts:

| Surface | What binary classification configures |
| --- | --- |
| `jopedime objective --task binary --classes neg,pos <spec>` | Resolves shorthand names or `jope.classification-fitness/v1` JSON, binds `class_order: [c0, c1]`, and reports canonical objective metadata and SHA-256 identity. |
| `jopedime fit --task binary --fitness … --calibration auto\|platt\|none` | Evaluates every native binary estimator across stratified out-of-fold splits, fits fold-safe Platt scaling when probability-sensitive (`logloss`, `brier`), and writes an authenticated `jope.model-object/v1` envelope. |
| `jopedime run --task binary` / `POST /v1/runs` | Prepares stratified `FIT` / `HOLDOUT` partitions, verifies improvement against the positive-class prior comparator (`BASELINE_MEAN`), runs stratified `rung0` / `rung1` cross-validation, and finalizes a deterministic `ModelObject`. |
| `jopedime predict` / `POST /v1/predictions` | Emits two-column `[P(c0), P(c1)]` calibrated probabilities bound to the retained `class_order`. |
| `ModelObject::export_source` / `POST /v1/runs/{id}/exports` | Emits standalone zero-dependency Python (`float.fromhex`), Rust (`f64::from_bits`), JavaScript (`Object.freeze`), and TypeScript (`.d.ts` `BinaryProbabilityPair` & `CLASS_ORDER`) predictors with `<= 1e-12` parity against `ModelObject::predict_proba`. |

`GET /v1/system` advertises `tasks: ["regression", "binary"]` on store-bound engines. `Task::Auto`
selects binary classification automatically when the target column has exactly two distinct non-null
classes (`>= 2` rows per class), sorting class labels deterministically unless explicit classes are
supplied.

## Quick start

```sh
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
cargo build --locked --release -p jope-appliance
J=target/release/jopedime

# 1. Inspect and hash a binary objective specification
$J objective --task binary --classes 0,1 logloss --json
$J objective --task binary --classes benign,malignant f1 --json

# 2. Offline fit + calibrated probability prediction
$J fit --train train.csv --target label --task binary \
  --fitness logloss --calibration auto --model-root ./models --json > fit.json

MODEL=$(python3 -c 'import json; print(json.load(open("fit.json"))["outputs"][0]["path"])')
$J predict --model "$MODEL" --input test_features.csv --output proba.csv --json

# 3. Export standalone predictors (Python, Rust, JavaScript) from a Platt-calibrated binary model
$J model verify --model "$MODEL" --require-executable
```

## Built-in binary objectives (`jope.classification-fitness/v1`)

`--fitness` (offline `fit` / `fit-predict`) and `--objective` (`run` / `objective`) accept six
built-in shorthands or a full `jope.classification-fitness/v1` JSON payload (`inline` or `@path.json`).
Every specification authenticates an exact two-element `class_order: [negative_label, positive_label]`;
single-class, three-or-more-class, duplicate-label, or non-finite inputs fail closed.

| Shorthand | Direction | Probability-sensitive | Default `auto` calibration | Head contract & typical use |
| --- | --- | --- | --- | --- |
| `logloss` | minimize | yes | `platt` | Cross-entropy on clamped `[P(c0), P(c1)]`; `HeadKind::Logit`. Strictly proper probability scoring. |
| `brier` | minimize | yes | `platt` | Mean squared probability error `(P(c1) - y)^2`; `HeadKind::Logit`. Strictly proper quadratic scoring. |
| `roc_auc` | maximize | no | `none` | Area under the receiver operating characteristic curve; `HeadKind::UnitInterval`. Rank-ordering invariance under strictly monotone maps. |
| `pr_auc` | maximize | no | `none` | Precision-recall area for positive class `c1`; `HeadKind::UnitInterval`. Rare-positive detection. |
| `f1` | maximize | no | `none` | Positive-class harmonic mean of precision and recall at threshold `0.5`; `HeadKind::UnitInterval`. |
| `balanced_accuracy` | maximize | no | `none` | Mean of sensitivity (`TPR`) and specificity (`TNR`) at threshold `0.5`; `HeadKind::UnitInterval`. |

## Cross-fitted Platt calibration

Raw margins or uncalibrated tree/ensemble scores can rank well while distorting log-loss or Brier
score. JopeDime supports three `--calibration` modes on binary tasks:

- `auto` (default): resolves to `platt` when the objective is probability-sensitive (`logloss`,
  `brier`) and `none` when the objective depends only on rank order or fixed-threshold classification
  (`roc_auc`, `pr_auc`, `f1`, `balanced_accuracy`).
- `platt`: fits two scalar parameters `(a, b)` mapping raw positive log-odds `z = ln(p / (1 - p))`
  through `P(c1) = 1 / (1 + exp(-(a * z + b)))` and `P(c0) = 1 - P(c1)`. During candidate selection,
  `(a, b)` are estimated strictly out-of-fold so held-out objective scores never see test labels; on
  final refit, out-of-fold cross-predictions over the training table fit the retained `Calibration::Platt { a, b }`.
- `none`: retains raw estimator probabilities without post-hoc scaling (`Calibration::None`).

`FittedModel::validate` enforces binary invariants on load: `Calibration::Linear` (regression-only)
is rejected on binary artifacts, non-finite `(a, b)` parameters are refused, and class-count
mismatches fail closed before inference.

## Service discovery lifecycle (`POST /v1/runs`)

When running through `jopedime serve` or batch `jopedime run`:

1. **Stratified preparation (`q1_prepare`):** Validates exact two-class membership, records
   `class_order`, and partitions `FIT` vs `HOLDOUT` indices per class so minority prevalence is
   preserved in both splits without exposing `HOLDOUT` rows to worker processes.
2. **Prior comparator (`BASELINE_MEAN`):** Computes the empirical `FIT` positive rate
   `p_pos = count(y == c1) / n_fit` and records `[1 - p_pos, p_pos]` under the active binary
   objective so candidate champions are checked against constant-prior performance.
3. **Stratified worker cross-validation & deterministic refit:** Evaluates candidates on stratified
   `rung0` and `rung1` folds under the canonical `jope.classification-fitness/v1` specification,
   refits the winning program with normalized provenance (`consumed_seconds: 0.0` for deterministic
   `ModelObject::id()`), and scores `HOLDOUT` once at finalization.
4. **Predictions & bundle exports:** `POST /v1/predictions` writes CSV columns matching `class_order`,
   and `POST /v1/runs/{id}/exports` packages `require_all_languages: true` bundles even when
   `Calibration::Platt` is active.

## Standalone multi-language source export (`EXP-05`)

`ModelObject::export_source` emits self-contained, zero-runtime-dependency source files that embed
both the underlying portable predictor and any `Calibration::Platt { a, b }` wrapper:

- **Python (`predict_proba(rows)`):** Exact IEEE-754 constants via `float.fromhex(...)` and
  overflow-safe two-branch logistic evaluation returning `[p0, p1]`.
- **Rust (`predict_proba(row)` / `predict_batch(rows)`):** Exact IEEE-754 constants via
  `f64::from_bits(...)` returning `[f64; 2]`.
- **JavaScript & TypeScript (`predict` / `predictBatch` + `.d.ts`):** `Object.freeze` probability
  pairs with shortest round-trip IEEE-754 literals and typed `.d.ts` declarations
  (`export type BinaryProbabilityPair = readonly [number, number]; export declare const CLASS_ORDER: readonly [string, string];`).

Low-level raw container accessors (`ModelObject::fitted()` and `ModelObject::bound()`) remain
fail-closed when post-hoc `Calibration::Platt` is attached so callers cannot accidentally bypass the
calibrated probability layer.

## Verification commands

```sh
# Run on a build host with CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only:
cargo test --locked -p jope-runtime --test offline_fit_predict
cargo test --locked -p jope-runtime --lib model_object
cargo test --locked -p jope-service --lib http::runs::tests::run
cargo test --locked -p jope-service --lib export::dts
cargo test --locked -p jope-appliance --bin jopedime cmd::objective
```
