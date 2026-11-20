# Export and prediction

The CLI exports a selected run model or generated source with `jopedime export` (or offline via `ModelObject::export_source`). Supported export kinds are `model`, `source`, and `bundle`; supported languages are Rust, Python, JavaScript, and TypeScript. `jopedime predict` accepts either a local model path (`ModelObject` directory or `.jope` file) or a completed run ID, plus input and output CSV paths.

For binary classification runs (`--task binary` or two-class `--task auto`; see [Binary classification](../binary-classification.md)):

- `POST /v1/predictions` and `jopedime predict` emit two calibrated probability columns matching the authenticated `class_order: [c0, c1]`.
- `ModelObject::export_source` and `POST /v1/runs/{id}/exports` (`EXP-05`, PR `#2390`) emit standalone zero-dependency predictors even when post-hoc `Calibration::Platt { a, b }` is attached (`logloss`, `brier`), wrapping the portable predictor across Python (`float.fromhex`), Rust (`f64::from_bits`), JavaScript (`Object.freeze`), and TypeScript (`.d.ts` `BinaryProbabilityPair` and `CLASS_ORDER`) with `<= 1e-12` parity against `ModelObject::predict_proba`.
- Raw container accessors (`ModelObject::fitted()` and `ModelObject::bound()`) remain fail-closed when post-hoc `Calibration::Platt` is present so callers cannot bypass calibrated probability scaling.

For multiclass classification runs (`--task multiclass` or non-numeric `K >= 3` `--task auto`; see [Multiclass classification](../multiclass-classification.md)):

- `POST /v1/predictions` and `jopedime predict` emit `prediction,p_<c0>,...,p_<c_{K-1}>` CSV columns aligned with the authenticated `class_order`.
- `ModelObject::export_source` and `POST /v1/runs/{id}/exports` (PR `#2398`) emit standalone zero-dependency predictors even when post-hoc `Calibration::Temperature` (`temperature.json`, `jope.temperature-calibration/v1`) is attached (`logloss`, `brier`), wrapping the portable predictor across Python (`float.fromhex`), Rust (`f64::from_bits`), JavaScript (`Object.freeze`), and TypeScript (`.d.ts` `MulticlassProbabilityTuple` and `CLASS_ORDER`) with `<= 1e-12` elementwise parity against `ModelObject::predict`.
- Raw container accessors (`ModelObject::fitted()`, `ModelObject::bound()`, and `ModelObject::math_pipeline()`) remain fail-closed when post-hoc `Calibration::Temperature` is present so callers cannot silently strip calibrated probability scaling.
