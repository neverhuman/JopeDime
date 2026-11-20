# Multiclass classification: objectives, temperature calibration, service runs, and source export

Multi-category decision problems (`K >= 3` classes) require authenticated class ordering, simplex
validation, and calibration transforms that do not scramble predicted class rankings. A clinical or
industrial triage table may optimize macro-averaged F1 (`macro_f1`), balanced recall across rare
categories (`balanced_accuracy`), asymmetric misclassification penalties (`cost_matrix_loss`), or
strictly proper probability scoring (`logloss`, multiclass `brier`). JopeDime binds the ordered
`class_order: [c_0, ..., c_{K-1}]`, canonical `jope.multiclass-fitness/v1` objective, out-of-fold
Temperature Scaling (`jope.temperature-calibration/v1`), and standalone multi-language source export
into a single fail-closed lifecycle across the offline CLI, local HTTP/MCP service, and persisted
`jope.model-object/v1` envelopes:

| Surface | What multiclass classification configures |
| --- | --- |
| `jopedime objective validate --task multiclass --classes c0,c1,c2 <spec>` | Validates shorthand names or `jope.multiclass-fitness/v1` JSON (`3..=1024` distinct classes), binds `class_order`, and prints canonical objective metadata and SHA-256 digest. |
| `jopedime fit --task multiclass --fitness … --calibration auto\|temperature\|none` | Scores native multiclass estimators across `K`-class stratified out-of-fold splits, cross-fits scalar Temperature Scaling when probability-sensitive (`logloss`, `brier`), and writes an authenticated `jope.model-object/v1` directory. |
| `jopedime run --task multiclass` / `POST /v1/runs` | Prepares `K`-class stratified `FIT` / `HOLDOUT` partitions, records the empirical `K`-vector prior comparator (`BASELINE_MEAN`), executes stratified `rung0` / `rung1` worker cross-validation, and finalizes a deterministic `ModelObject`. |
| `jopedime predict` / `POST /v1/predictions` | Emits `prediction,p_<c0>,...,p_<c_{K-1}>` CSV rows aligned with the retained `class_order` (`argmax` ties broken toward the earliest index in `class_order`). |
| `ModelObject::export_source` / `POST /v1/runs/{id}/exports` | Emits standalone zero-dependency Python (`float.fromhex`), Rust (`f64::from_bits`), JavaScript (`Object.freeze`), and TypeScript (`.d.ts` `MulticlassProbabilityTuple` & `CLASS_ORDER`) predictors with `<= 1e-12` elementwise parity against `ModelObject::predict`. |

`GET /v1/system` advertises `tasks: ["regression", "binary", "multiclass"]` on store-bound engines.
`Task::Auto` selects multiclass classification automatically when a non-numeric target column has
`3..=1024` distinct non-null class labels with at least `2` rows per class (numeric `3+`-value columns
remain regression under `Task::Auto` unless `--task multiclass` is passed explicitly).

## Quick start

```sh
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
cargo build --locked --release -p jope-appliance
J=target/release/jopedime

# 1. Inspect and hash a multiclass objective specification
$J objective validate --task multiclass --classes setosa,versicolor,virginica logloss --json
$J objective validate --task multiclass --classes low,medium,high macro_f1 --json

# 2. Offline fit + temperature-calibrated K-class probability prediction
$J fit --train train.csv --target species --task multiclass \
  --fitness logloss --calibration auto --model-root ./models --json > fit.json

MODEL=$(python3 -c 'import json; print(json.load(open("fit.json"))["outputs"][0]["path"])')
$J predict --model "$MODEL" --input test_features.csv --output proba.csv --json

# 3. Verify executable envelope integrity (including temperature.json capability binding)
$J model verify --model "$MODEL" --require-executable
```

## Built-in multiclass objectives (`jope.multiclass-fitness/v1`)

`--fitness` (offline `fit` / `fit-predict`) and `--objective` (`run` / `objective validate`) accept
five built-in shorthands or a full `jope.multiclass-fitness/v1` JSON specification (`inline` or
`@path.json`, including custom `K x K` `cost_matrix_loss`). Every specification authenticates an
ordered `class_order` of `3..=1024` distinct non-empty labels; fewer than three classes, duplicate
labels, unknown target labels, or simplex violations (`|sum_k p_k - 1| > 1e-6`) fail closed without
implicit renormalization.

| Objective | Shorthand | Direction | Probability-sensitive | Default `auto` calibration | Head contract & semantics |
| --- | --- | --- | --- | --- | --- |
| Multiclass log-loss | `logloss` | minimize | yes | `temperature` | Natural cross-entropy `-ln(p_{y})` (`HeadKind::Softmax`). Strictly proper probability scoring. |
| Multiclass Brier | `brier` | minimize | yes | `temperature` | Quadratic simplex score `sum_{k=0..K-1} (p_k - I[y == k])^2` (`HeadKind::Softmax`). Strictly proper quadratic scoring. |
| Macro F1 | `macro_f1` | maximize | no | `none` | Unweighted mean of per-class F1 scores over `argmax` predictions (`HeadKind::Simplex`); requires every declared class to appear in ground truth. |
| Multiclass accuracy | `accuracy` | maximize | no | `none` | Top-1 `argmax` match rate (`HeadKind::Simplex`). |
| Balanced accuracy | `balanced_accuracy` | maximize | no | `none` | Unweighted mean of per-class recall over `argmax` predictions (`HeadKind::Simplex`); requires every declared class to appear in ground truth. |
| Cost-matrix loss | JSON (`cost_matrix_loss`) | minimize | no | `none` | Asymmetric `K x K` penalty `C[true_idx][pred_idx]` with finite non-negative off-diagonals and zero diagonal (`HeadKind::Simplex`). |

## Native multiclass estimators & cross-fitted Temperature Scaling

The native CPU multiclass roster evaluates four deterministic schema-bound estimators across all
supported export languages (`rust`, `python`, `javascript`):

- `multiclass_ridge`: L2-regularized multinomial linear softmax head.
- `multiclass_ovr_logistic`: One-vs-Rest logistic linear ensemble normalized onto the `K`-simplex.
- `multiclass_diagonal_lda`: Shared-diagonal Gaussian linear discriminant head with empirical log-priors.
- `multiclass_nearest_centroid`: Standardized Euclidean class-centroid softmax head.

JopeDime supports three `--calibration` modes on multiclass tasks:

- `auto` (default): resolves to `temperature` when the active objective is probability-sensitive
  (`logloss`, `brier`) and `none` for discrete `argmax` objectives (`macro_f1`, `accuracy`,
  `balanced_accuracy`, `cost_matrix_loss`).
- `temperature`: cross-fits a single positive scalar temperature `T` bounded to `[0.05, 20.0]` on
  out-of-fold `K`-class probability vectors via shift-invariant softmax over
  `ln(max(p_k, 1e-15)) / T`. During model selection, each validation fold is scored using a scalar
  temperature fitted strictly on the remaining training folds; the stored `temperature.json` part
  (`jope.temperature-calibration/v1`, capability `org.jopedime/temperature-calibration/v1`) refits
  `T` on all out-of-fold rows and reverts to `1.0` whenever scaling does not strictly improve
  out-of-fold objective loss. Because `1 / T > 0` multiplies all class log-probabilities by the
  same positive constant, **`argmax` class ordering is strictly invariant on every row**.
- `none`: retains raw estimator simplex probabilities without post-hoc scaling (`Calibration::None`).

Incompatible calibration families (`shift`, `affine`, `platt`) fail closed with `FailureKind::Input`
on multiclass tasks, and `ModelObject` envelopes carrying both `platt.json` and `temperature.json`
are rejected as integrity errors.

## Service discovery lifecycle (`POST /v1/runs`)

When running through `jopedime serve` or batch `jopedime run`:

1. **`K`-class stratified preparation (`q1_prepare`):** Validates `3..=1024` distinct classes with at
   least `2` rows per class (allocating at least one `FIT` and one `HOLDOUT` row per class before
   proportional rounding so `macro_f1` and `balanced_accuracy` recall denominators remain well-defined
   on both partitions).
2. **Empirical prior comparator (`BASELINE_MEAN`):** Computes the `FIT` class-frequency vector
   `[count(y == c_0) / n_fit, ..., count(y == c_{K-1}) / n_fit]` and evaluates it under the active
   `jope.multiclass-fitness/v1` specification.
3. **Stratified worker cross-validation & deterministic refit:** Evaluates the native multiclass
   roster on `K`-class stratified `rung0` and `rung1` folds, cross-fits Temperature Scaling when
   probability-sensitive, refits the winning program with deterministic provenance, and scores
   `HOLDOUT` once during finalization.
4. **Predictions & bundle exports:** `POST /v1/predictions` writes `prediction,p_<c0>,...,p_<c_{K-1}>`
   CSV columns, and `POST /v1/runs/{id}/exports` packages `ExportKind::Bundle` archives across Rust,
   Python, and JavaScript/TypeScript for both uncalibrated and temperature-calibrated champions.

## Standalone multi-language source export

`ModelObject::export_source` emits self-contained, zero-dependency predictors that embed both the
portable multiclass base head and any `jope.temperature-calibration/v1` (`temperature.json`) wrapper:

- **Python (`predict_encoded(rows)` / `fitted_temperature(probs)`):** Exact IEEE-754 hex constants via
  `float.fromhex(...)`, simplex validation, and shift-invariant softmax alongside `TEMPERATURE_CLASS_ORDER`.
- **Rust (`predict_encoded(rows)` / `fitted_temperature(probs)`):** Exact IEEE-754 bit patterns via
  `f64::from_bits(...)` alongside `pub const TEMPERATURE_CLASS_ORDER: [&str; K]`.
- **JavaScript & TypeScript (`predictEncoded(rows)` / `fittedTemperature(probs)` + `.d.ts`):**
  `Object.freeze` class order with shortest round-trip IEEE-754 literals and arity-`K` TypeScript
  declarations (`export type MulticlassProbabilityTuple = [number, ...]; export declare const CLASS_ORDER: readonly [...];`).

Low-level uncalibrated accessors (`ModelObject::fitted()`, `ModelObject::bound()`, and
`ModelObject::math_pipeline()`) remain fail-closed when `temperature.json` is present so callers
cannot silently strip post-hoc Temperature Scaling.

## Verification commands

```sh
# Run on a build host with CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only:
cargo test --locked -p jope-runtime --lib fitness_multiclass
cargo test --locked -p jope-runtime --lib calibration_temperature
cargo test --locked -p jope-runtime --lib temperature_object_round_trip
cargo test --locked -p jope-runtime --test offline_fit_predict multiclass_objective_and_temperature_calibration
cargo test --locked -p jope-service --lib export::dts
cargo test --locked -p jope-service --lib http::runs::tests::run::multiclass_discovery
cargo test --locked -p jope-appliance --bin jopedime cmd::objective::tests
```
