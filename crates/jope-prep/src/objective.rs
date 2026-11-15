//! Portable, WebAssembly-safe regression objective parser, validator, catalog, canonical SHA-256
//! identity, and standalone evaluator backed by `schemas/jope.regression-fitness.v1.schema.json`.
//!
//! Keeps `jope-prep` and `jope-carbon` (`edge` and `cli`) free of `jope-runtime` / `feat-math`
//! (`CT3` zero-IP-leakage invariant) while preserving 1:1 wire, identity, validation, and scoring
//! parity with `jope_runtime::fitness::Fitness`.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// Canonical objective specification schema version (`ADR-0005` / regression objectives v1).
pub const OBJECTIVE_VERSION: &str = "jope.regression-fitness/v1";
/// Maximum serialized byte size of an objective specification (`32 KiB`).
pub const MAX_OBJECTIVE_SPEC_BYTES: usize = 32 * 1024;
/// Maximum expression AST nodes across one objective specification.
pub const MAX_OBJECTIVE_EXPR_NODES: usize = 256;
/// Maximum distinct auxiliary columns referenced by one objective specification.
pub const MAX_OBJECTIVE_COLUMNS: usize = 16;
const MAX_COLUMN_NAME_BYTES: usize = 256;

const COMMITTED_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.regression-fitness.v1.schema.json");

/// Optimization direction of a regression objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

impl ObjectiveDirection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minimize => "minimize",
            Self::Maximize => "maximize",
        }
    }
}

/// Result of scoring aligned `(y, pred)` vectors under an [`ObjectiveSpec`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveScore {
    pub value: f64,
    /// Selection loss (`-value` when `direction == Maximize`, else `value`).
    pub loss: f64,
    pub direction: ObjectiveDirection,
    pub fitness_sha256: String,
}

/// Summary metadata for one built-in catalog objective.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveCatalogEntry {
    pub name: &'static str,
    pub default_spec: Value,
    pub direction: ObjectiveDirection,
    pub requires_order: bool,
    pub uses_resources: bool,
    pub description: &'static str,
}

/// Validated, canonicalized regression objective specification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    spec: Value,
}

const VALID_RESOURCES: &[&str] = &[
    "fit_seconds",
    "fit_cpu_seconds",
    "predict_seconds",
    "predict_us_per_row",
    "predict_rows_per_second",
    "row_latency_p50_us",
    "row_latency_p99_us",
    "fit_peak_heap_bytes",
    "predict_peak_heap_bytes",
    "model_bytes",
    "program_nodes",
    "stage_count",
    "train_rows",
    "hold_rows",
];

impl ObjectiveSpec {
    /// Return the committed `jope.regression-fitness.v1` JSON Schema document.
    pub fn schema_json() -> Value {
        serde_json::from_str(COMMITTED_SCHEMA_JSON).expect("valid committed fitness JSON schema")
    }

    /// Expand friendly CLI shorthand such as `pinball(0.9)` or `asymmetric_absolute(3,1)` into
    /// canonical `name:key=value` notation accepted by both `ObjectiveSpec` and `jope-runtime`.
    pub fn normalize_shorthand(raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.starts_with('{') || !trimmed.ends_with(')') {
            return trimmed.to_owned();
        }
        let Some((name, inside)) = trimmed[..trimmed.len() - 1].split_once('(') else {
            return trimmed.to_owned();
        };
        let name = name.trim();
        let args: Vec<&str> = inside
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if args.is_empty() {
            return name.to_owned();
        }
        if args.iter().all(|a| a.contains('=')) {
            return format!("{name}:{}", args.join(","));
        }
        let param_keys: &[&str] = match name {
            "pinball" | "expectile" => &["quantile"],
            "huber" | "pseudo_huber" => &["delta"],
            "mape" | "smape" | "rmspe" => &["epsilon"],
            "tweedie_deviance" => &["power"],
            "linex" => &["a"],
            "mase" => &["season"],
            "capture_at" => &["fraction"],
            "expected_shortfall" => &["alpha"],
            "asymmetric_absolute" | "asymmetric_squared" => &["under_cost", "over_cost"],
            _ => return trimmed.to_owned(),
        };
        if args.len() != param_keys.len() {
            return trimmed.to_owned();
        }
        let pairs: Vec<String> = param_keys
            .iter()
            .zip(args.iter())
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        format!("{name}:{}", pairs.join(","))
    }

    /// Parse a catalog name (`rmse`), parameter override (`pinball:quantile=0.9`), shorthand
    /// (`pinball(0.9)`), or strict inline JSON specification (`{"metric":...}`).
    pub fn parse(raw: &str) -> Result<Self, String> {
        if raw.len() > MAX_OBJECTIVE_SPEC_BYTES {
            return Err("fitness specification is too large".to_owned());
        }
        let normalized = Self::normalize_shorthand(raw);
        let spec = if normalized.starts_with('{') {
            serde_json::from_str::<Value>(&normalized)
                .map_err(|e| format!("invalid fitness JSON specification: {e}"))?
        } else {
            let (name, overrides) = match normalized.split_once(':') {
                Some((n, ov)) => (n.trim(), Some(ov)),
                None => (normalized.as_str(), None),
            };
            let mut base = builtin_default_value(name).ok_or_else(|| {
                format!("unknown fitness {name:?}; run `carbon fitness list` to inspect valid objectives")
            })?;
            if let Some(ov) = overrides {
                let obj = base.as_object_mut().expect("object spec");
                for pair in ov.split(',').filter(|p| !p.trim().is_empty()) {
                    let (k, v_raw) = pair
                        .split_once('=')
                        .ok_or_else(|| format!("override {pair:?} is not key=value"))?;
                    let key = k.trim();
                    if key == "metric" || !obj.contains_key(key) {
                        return Err(format!("{name} has no parameter {key:?}"));
                    }
                    let num: Value = serde_json::from_str(v_raw.trim())
                        .ok()
                        .filter(Value::is_number)
                        .ok_or_else(|| format!("parameter {key} needs a number, got {v_raw:?}"))?;
                    if key == "season" {
                        let season_u64 = num.as_u64().ok_or_else(|| {
                            format!("parameter season needs a positive integer, got {v_raw:?}")
                        })?;
                        obj.insert(key.to_owned(), json!(season_u64));
                    } else {
                        let f = num
                            .as_f64()
                            .ok_or_else(|| format!("parameter {key} out of f64 range"))?;
                        obj.insert(key.to_owned(), Value::from(f));
                    }
                }
            }
            base
        };

        let candidate = Self { spec };
        candidate.validate()?;
        Ok(candidate)
    }

    /// Return all 45 built-in catalog entries in canonical order.
    pub fn catalog() -> Vec<ObjectiveCatalogEntry> {
        CATALOG_ROWS
            .iter()
            .map(|&(name, desc)| {
                let spec = ObjectiveSpec {
                    spec: builtin_default_value(name).expect("catalog entry exists"),
                };
                ObjectiveCatalogEntry {
                    name,
                    default_spec: spec.spec.clone(),
                    direction: spec.direction(),
                    requires_order: spec.requires_order(),
                    uses_resources: spec.uses_execution_telemetry(),
                    description: desc,
                }
            })
            .collect()
    }

    /// Underlying JSON representation.
    pub fn as_value(&self) -> &Value {
        &self.spec
    }

    /// Wire representation suitable for `CarbonCreateRunRequest.objective` and `jope-runtime`:
    /// keeps plain catalog names compact (for example `"rmse"`) so `v1.1.0` default requests stay
    /// byte-identical while emitting canonical JSON for parameterized or custom objectives.
    pub fn to_wire_objective(&self) -> String {
        for &(name, _) in CATALOG_ROWS {
            if let Some(def) = builtin_default_value(name) {
                if self.spec == def {
                    return name.to_owned();
                }
            }
        }
        self.canonical_json()
    }

    /// Human-readable short name for display and reporting.
    pub fn name(&self) -> String {
        metric_name_of(&self.spec)
    }

    /// Optimization direction (`Minimize` or `Maximize`).
    pub fn direction(&self) -> ObjectiveDirection {
        direction_of(&self.spec)
    }

    /// Whether evaluation depends on chronological row order (requiring forward validation splits).
    pub fn requires_order(&self) -> bool {
        requires_order_of(&self.spec)
    }

    /// Auxiliary table columns referenced by custom expressions or row weights.
    pub fn columns(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        collect_columns_spec(&self.spec, &mut out);
        out
    }

    /// Resource variables referenced by this specification.
    pub fn resources(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        collect_resources_spec(&self.spec, &mut out);
        out
    }

    /// Whether evaluation requires candidate execution telemetry or resource measurements.
    pub fn uses_execution_telemetry(&self) -> bool {
        !self.resources().is_empty() || uses_telemetry_only_spec(&self.spec)
    }

    /// Whether a regression model trained with standard squared/L1 internal search heads benefits
    /// from fold-safe post-hoc output calibration (`a * pred + b`, optionally positive-floored) to
    /// align predictions with an asymmetric, quantile, deviance, or custom objective.
    pub fn benefits_from_post_hoc_calibration(&self) -> bool {
        needs_calibration_spec(&self.spec)
    }

    /// Positive domain floor required on predictions (`Some(1e-6)` for Poisson/Gamma/Tweedie,
    /// `Some(0.0)` for MSLE/RMSLE, `None` otherwise).
    pub fn prediction_lower_bound(&self) -> Option<f64> {
        match self.spec.get("metric").and_then(Value::as_str) {
            Some("poisson_deviance" | "gamma_deviance" | "tweedie_deviance") => Some(1e-6),
            Some("msle" | "rmsle") => Some(0.0),
            Some("weighted" | "constrained") => self.spec.get("fitness").and_then(|inner| {
                ObjectiveSpec {
                    spec: inner.clone(),
                }
                .prediction_lower_bound()
            }),
            _ => None,
        }
    }

    /// Canonical compact JSON with field order matching `jope_runtime::fitness::Fitness`.
    pub fn canonical_json(&self) -> String {
        let mut buf = Vec::with_capacity(128);
        write_canonical_spec(&self.spec, &mut buf);
        String::from_utf8(buf).expect("valid utf-8 canonical json")
    }

    /// Domain-separated SHA-256 hex digest matching `jope_runtime::fitness::Fitness::identity()`.
    pub fn identity_sha256(&self) -> String {
        let mut payload = Vec::with_capacity(160);
        payload.extend_from_slice(b"[\"");
        payload.extend_from_slice(OBJECTIVE_VERSION.as_bytes());
        payload.extend_from_slice(b"\",");
        write_canonical_spec(&self.spec, &mut payload);
        payload.push(b']');
        let digest = Sha256::digest(&payload);
        hex_lower(&digest)
    }

    /// Validate structural, numerical, recursion, node-count, and auxiliary-column invariants.
    pub fn validate(&self) -> Result<(), String> {
        let mut nodes = 0usize;
        let mut expr_nodes = 0usize;
        validate_spec_at(&self.spec, 0, &mut nodes, &mut expr_nodes, false)?;
        if self.columns().len() > MAX_OBJECTIVE_COLUMNS {
            return Err(format!(
                "a fitness specification may read at most {MAX_OBJECTIVE_COLUMNS} columns"
            ));
        }
        if self.canonical_json().len() > MAX_OBJECTIVE_SPEC_BYTES {
            return Err("fitness specification is too large".to_owned());
        }
        Ok(())
    }

    /// Evaluate aligned `y` and `pred` slices with optional auxiliary `columns`.
    pub fn evaluate(
        &self,
        y: &[f64],
        pred: &[f64],
        columns: &BTreeMap<String, Vec<f64>>,
    ) -> Result<ObjectiveScore, String> {
        self.validate()?;
        if y.is_empty() || y.len() != pred.len() {
            return Err("fitness needs equal, nonempty target and prediction vectors".to_owned());
        }
        if !y.iter().chain(pred.iter()).all(|v| v.is_finite()) {
            return Err("fitness inputs must be finite".to_owned());
        }
        for col_name in self.columns() {
            let vals = columns
                .get(&col_name)
                .ok_or_else(|| format!("fitness column {col_name:?} was not supplied"))?;
            if vals.len() != y.len() || !vals.iter().all(|v| v.is_finite()) {
                return Err(format!(
                    "fitness column {col_name:?} needs one finite value per scored row"
                ));
            }
        }
        let raw_val = eval_spec_value(&self.spec, y, pred, columns, None)?;
        let value = ensure_finite(raw_val)?;
        let direction = self.direction();
        let loss = if direction == ObjectiveDirection::Maximize {
            -value
        } else {
            value
        };
        Ok(ObjectiveScore {
            value,
            loss,
            direction,
            fitness_sha256: self.identity_sha256(),
        })
    }
}

const CATALOG_ROWS: &[(&str, &str)] = &[
    ("rmse", "Root mean squared error; emphasizes large misses."),
    ("mse", "Mean squared error."),
    ("mae", "Mean absolute error; linear cost per unit missed."),
    (
        "median_absolute_error",
        "Median absolute error; ignores the worst half of misses.",
    ),
    ("max_error", "Worst absolute error on scored rows."),
    (
        "mape",
        "Mean absolute percentage error (ratio) with a denominator floor.",
    ),
    (
        "smape",
        "Symmetric mean absolute percentage error (ratio, 0..2).",
    ),
    (
        "wape",
        "Total absolute error over total absolute target volume.",
    ),
    ("msle", "Mean squared log1p error; nonnegative values."),
    (
        "rmsle",
        "Root mean squared log1p error; nonnegative values.",
    ),
    ("r2", "Coefficient of determination (maximize)."),
    (
        "explained_variance",
        "Explained variance; ignores a constant bias (maximize).",
    ),
    ("pearson", "Pearson correlation (maximize)."),
    (
        "spearman",
        "Rank correlation, the information coefficient (maximize).",
    ),
    ("huber", "Huber loss: quadratic inside delta."),
    ("pseudo_huber", "Smooth Huber loss."),
    ("log_cosh", "log(cosh(error)); smooth and robust."),
    (
        "pinball",
        "Quantile (pinball) loss; quantile=0.9 makes shortages 9x as costly.",
    ),
    ("expectile", "Asymmetric squared error."),
    (
        "poisson_deviance",
        "Poisson deviance for counts; y>=0, pred>0.",
    ),
    ("gamma_deviance", "Gamma deviance for positive severities."),
    (
        "tweedie_deviance",
        "Tweedie deviance for zero-inflated positive costs.",
    ),
    (
        "linex",
        "Linear-exponential loss; a>0 punishes under-prediction exponentially.",
    ),
    (
        "asymmetric_absolute",
        "Newsvendor cost: under_cost per unit short, over_cost per unit over.",
    ),
    ("asymmetric_squared", "Squared newsvendor cost."),
    ("rmspe", "Root mean squared percentage error (ratio)."),
    (
        "mase",
        "MAE relative to the seasonal naive forecast; chronological rows.",
    ),
    (
        "mean_return",
        "Mean net strategy return per interval (maximize).",
    ),
    (
        "sharpe",
        "Annualized Sharpe ratio of net strategy returns (maximize).",
    ),
    (
        "sortino",
        "Annualized Sortino ratio; penalizes only downside (maximize).",
    ),
    (
        "max_drawdown",
        "Largest peak-to-trough loss of compounded equity.",
    ),
    (
        "calmar",
        "Compound annual growth over maximum drawdown (maximize).",
    ),
    (
        "omega",
        "Gains above threshold over shortfalls below it (maximize).",
    ),
    (
        "expected_shortfall",
        "Mean loss of the worst alpha fraction of net returns (CVaR).",
    ),
    (
        "hit_rate",
        "Share of nonzero targets whose sign is predicted (maximize).",
    ),
    (
        "normalized_gini",
        "Ranking quality: Gini over perfect Gini (maximize).",
    ),
    ("kge", "Kling-Gupta efficiency for hydrology (maximize)."),
    (
        "capture_at",
        "Share of total target in the top-ranked fraction (maximize).",
    ),
    (
        "train_time_seconds",
        "Wall-clock time in seconds spent fitting the candidate model.",
    ),
    (
        "inference_time_seconds",
        "Wall-clock time in seconds spent running candidate inference.",
    ),
    (
        "total_time_seconds",
        "Combined training and inference wall-clock time in seconds.",
    ),
    (
        "inference_us_per_row",
        "Per-row inference latency in microseconds.",
    ),
    (
        "train_rows_per_second",
        "Training throughput in rows per second (maximize for faster training).",
    ),
    (
        "inference_rows_per_second",
        "Inference throughput in rows per second (maximize for faster inference).",
    ),
    (
        "train_memory_bytes",
        "Peak/working heap memory in bytes used during candidate training.",
    ),
    (
        "inference_memory_bytes",
        "Peak/working heap memory in bytes used during candidate inference.",
    ),
    (
        "total_memory_bytes",
        "Combined training and inference memory footprint in bytes.",
    ),
    ("model_bytes", "Fitted model artifact footprint in bytes."),
    (
        "complexity_stages",
        "Pipeline stage count of the candidate.",
    ),
    (
        "complexity_parameters",
        "Fitted scalar parameter/node count of the candidate.",
    ),
    (
        "feature_count",
        "Active input feature count consumed by the candidate.",
    ),
    (
        "fold_std",
        "Standard deviation of loss across cross-validation folds or regime cuts.",
    ),
    (
        "prediction_turnover",
        "Mean absolute sequential change mean(|pred[t] - pred[t-1]|).",
    ),
];

fn builtin_default_value(name: &str) -> Option<Value> {
    Some(match name {
        "rmse"
        | "mse"
        | "mae"
        | "median_absolute_error"
        | "max_error"
        | "wape"
        | "msle"
        | "rmsle"
        | "r2"
        | "explained_variance"
        | "pearson"
        | "spearman"
        | "log_cosh"
        | "poisson_deviance"
        | "gamma_deviance"
        | "hit_rate"
        | "normalized_gini"
        | "kge"
        | "train_time_seconds"
        | "inference_time_seconds"
        | "total_time_seconds"
        | "inference_us_per_row"
        | "train_rows_per_second"
        | "inference_rows_per_second"
        | "train_memory_bytes"
        | "inference_memory_bytes"
        | "total_memory_bytes"
        | "model_bytes"
        | "complexity_stages"
        | "complexity_parameters"
        | "feature_count"
        | "fold_std"
        | "prediction_turnover" => json!({ "metric": name }),
        "mape" | "smape" | "rmspe" => json!({ "metric": name, "epsilon": 1e-8 }),
        "huber" | "pseudo_huber" => json!({ "metric": name, "delta": 1.0 }),
        "pinball" | "expectile" => json!({ "metric": name, "quantile": 0.5 }),
        "tweedie_deviance" => json!({ "metric": name, "power": 1.5 }),
        "linex" => json!({ "metric": name, "a": 1.0 }),
        "asymmetric_absolute" | "asymmetric_squared" => {
            json!({ "metric": name, "under_cost": 1.0, "over_cost": 1.0 })
        }
        "mase" => json!({ "metric": name, "season": 1u64 }),
        "mean_return" | "max_drawdown" => {
            json!({ "metric": name, "max_position": 1.0, "cost": 0.0 })
        }
        "sharpe" | "sortino" => json!({
            "metric": name,
            "max_position": 1.0,
            "cost": 0.0,
            "target": 0.0,
            "periods_per_year": 252.0
        }),
        "calmar" => json!({
            "metric": name,
            "max_position": 1.0,
            "cost": 0.0,
            "periods_per_year": 252.0
        }),
        "omega" => json!({
            "metric": name,
            "max_position": 1.0,
            "cost": 0.0,
            "threshold": 0.0
        }),
        "expected_shortfall" => json!({
            "metric": name,
            "max_position": 1.0,
            "cost": 0.0,
            "alpha": 0.05
        }),
        "capture_at" => json!({ "metric": name, "fraction": 0.1 }),
        _ => return None,
    })
}

fn expected_fields_for_metric(metric: &str) -> Option<&'static [&'static str]> {
    Some(match metric {
        "rmse"
        | "mse"
        | "mae"
        | "median_absolute_error"
        | "max_error"
        | "wape"
        | "msle"
        | "rmsle"
        | "r2"
        | "explained_variance"
        | "pearson"
        | "spearman"
        | "log_cosh"
        | "poisson_deviance"
        | "gamma_deviance"
        | "hit_rate"
        | "normalized_gini"
        | "kge"
        | "train_time_seconds"
        | "inference_time_seconds"
        | "total_time_seconds"
        | "inference_us_per_row"
        | "train_rows_per_second"
        | "inference_rows_per_second"
        | "train_memory_bytes"
        | "inference_memory_bytes"
        | "total_memory_bytes"
        | "model_bytes"
        | "complexity_stages"
        | "complexity_parameters"
        | "feature_count"
        | "fold_std"
        | "prediction_turnover" => &["metric"],
        "mape" | "smape" | "rmspe" => &["metric", "epsilon"],
        "huber" | "pseudo_huber" => &["metric", "delta"],
        "pinball" | "expectile" => &["metric", "quantile"],
        "tweedie_deviance" => &["metric", "power"],
        "linex" => &["metric", "a"],
        "asymmetric_absolute" | "asymmetric_squared" => &["metric", "under_cost", "over_cost"],
        "mase" => &["metric", "season"],
        "mean_return" | "max_drawdown" => &["metric", "max_position", "cost"],
        "sharpe" | "sortino" => &[
            "metric",
            "max_position",
            "cost",
            "target",
            "periods_per_year",
        ],
        "calmar" => &["metric", "max_position", "cost", "periods_per_year"],
        "omega" => &["metric", "max_position", "cost", "threshold"],
        "expected_shortfall" => &["metric", "max_position", "cost", "alpha"],
        "capture_at" => &["metric", "fraction"],
        "resource" => &["metric", "name"],
        "custom" => &["metric", "name", "direction", "expression", "aggregation"],
        "weighted" => &["metric", "weights", "fitness"],
        "composite" => &["metric", "terms"],
        "constrained" => &["metric", "fitness", "limits"],
        _ => return None,
    })
}

fn check_exact_keys(obj: &Map<String, Value>, allowed: &[&str], ctx: &str) -> Result<(), String> {
    for k in obj.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(format!("unknown field {k:?} in {ctx}"));
        }
    }
    for &req in allowed {
        if !obj.contains_key(req) {
            return Err(format!("missing field {req:?} in {ctx}"));
        }
    }
    Ok(())
}

fn req_f64(obj: &Map<String, Value>, key: &str) -> Result<f64, String> {
    obj.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("field {key:?} must be a finite number"))
}

fn is_pos(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

fn is_prob_open(x: f64) -> bool {
    x.is_finite() && x > 0.0 && x < 1.0
}

fn ensure_finite(x: f64) -> Result<f64, String> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err("non-finite fitness value".to_owned())
    }
}

fn metric_name_of(spec: &Value) -> String {
    let metric = spec
        .get("metric")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    match metric {
        "resource" => format!(
            "resource_{}",
            spec.get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        ),
        "custom" => spec
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("custom")
            .to_owned(),
        "weighted" => format!(
            "weighted_{}",
            spec.get("fitness")
                .map_or_else(|| "unknown".to_owned(), metric_name_of)
        ),
        "composite" => "composite".to_owned(),
        "constrained" => format!(
            "constrained_{}",
            spec.get("fitness")
                .map_or_else(|| "unknown".to_owned(), metric_name_of)
        ),
        other => other.to_owned(),
    }
}

fn direction_of(spec: &Value) -> ObjectiveDirection {
    match spec.get("metric").and_then(Value::as_str).unwrap_or("") {
        "r2"
        | "explained_variance"
        | "pearson"
        | "spearman"
        | "mean_return"
        | "sharpe"
        | "sortino"
        | "calmar"
        | "omega"
        | "hit_rate"
        | "normalized_gini"
        | "kge"
        | "capture_at"
        | "train_rows_per_second"
        | "inference_rows_per_second" => ObjectiveDirection::Maximize,
        "resource"
            if spec.get("name").and_then(Value::as_str) == Some("predict_rows_per_second") =>
        {
            ObjectiveDirection::Maximize
        }
        "custom" => match spec.get("direction").and_then(Value::as_str) {
            Some("maximize") => ObjectiveDirection::Maximize,
            _ => ObjectiveDirection::Minimize,
        },
        "weighted" | "constrained" => spec
            .get("fitness")
            .map_or(ObjectiveDirection::Minimize, direction_of),
        _ => ObjectiveDirection::Minimize,
    }
}

fn supports_weights_spec(spec: &Value) -> bool {
    match spec.get("metric").and_then(Value::as_str).unwrap_or("") {
        "max_error" | "spearman" | "mase" | "mean_return" | "sharpe" | "sortino"
        | "max_drawdown" | "calmar" | "omega" | "expected_shortfall" | "kge" | "capture_at"
        | "resource" | "weighted" => false,
        "custom" => !matches!(
            spec.get("aggregation")
                .and_then(|a| a.get("kind"))
                .and_then(Value::as_str),
            Some("max" | "min")
        ),
        "composite" => spec
            .get("terms")
            .and_then(Value::as_array)
            .is_some_and(|terms| {
                terms
                    .iter()
                    .all(|t| t.get("fitness").is_some_and(supports_weights_spec))
            }),
        "constrained" => spec.get("fitness").is_some_and(supports_weights_spec),
        _ => true,
    }
}

fn requires_order_of(spec: &Value) -> bool {
    match spec.get("metric").and_then(Value::as_str).unwrap_or("") {
        "mean_return"
        | "sharpe"
        | "sortino"
        | "max_drawdown"
        | "calmar"
        | "omega"
        | "expected_shortfall"
        | "mase"
        | "prediction_turnover" => true,
        "custom" => spec.get("expression").is_some_and(requires_order_expr),
        "weighted" => {
            spec.get("weights").is_some_and(requires_order_expr)
                || spec.get("fitness").is_some_and(requires_order_of)
        }
        "constrained" => spec.get("fitness").is_some_and(requires_order_of),
        "composite" => spec
            .get("terms")
            .and_then(Value::as_array)
            .is_some_and(|terms| {
                terms
                    .iter()
                    .any(|t| t.get("fitness").is_some_and(requires_order_of))
            }),
        _ => false,
    }
}

fn requires_order_expr(expr: &Value) -> bool {
    match canonical_expr_op(expr.get("op").and_then(Value::as_str).unwrap_or("")) {
        "prediction_turnover" | "row_index" | "row_fraction" => true,
        "abs" | "neg" | "square" | "sqrt" | "ln" | "log1p" | "exp" | "sign" | "tanh"
        | "sigmoid" | "cbrt" | "step" | "clip" => expr.get("arg").is_some_and(requires_order_expr),
        "add" | "sub" | "mul" | "div" | "min" | "max" | "pow" => {
            expr.get("left").is_some_and(requires_order_expr)
                || expr.get("right").is_some_and(requires_order_expr)
        }
        "where" => {
            expr.get("condition").is_some_and(requires_order_expr)
                || expr.get("then").is_some_and(requires_order_expr)
                || expr.get("else").is_some_and(requires_order_expr)
        }
        _ => false,
    }
}

fn needs_calibration_spec(spec: &Value) -> bool {
    match spec.get("metric").and_then(Value::as_str).unwrap_or("") {
        "pinball"
        | "expectile"
        | "linex"
        | "asymmetric_absolute"
        | "asymmetric_squared"
        | "poisson_deviance"
        | "gamma_deviance"
        | "tweedie_deviance"
        | "mape"
        | "smape"
        | "rmspe"
        | "wape"
        | "msle"
        | "rmsle"
        | "kge"
        | "mean_return"
        | "sharpe"
        | "sortino"
        | "calmar"
        | "omega"
        | "expected_shortfall"
        | "max_drawdown"
        | "custom" => true,
        "weighted" | "constrained" => spec.get("fitness").is_some_and(needs_calibration_spec),
        "composite" => spec
            .get("terms")
            .and_then(Value::as_array)
            .is_some_and(|terms| {
                terms
                    .iter()
                    .any(|t| t.get("fitness").is_some_and(needs_calibration_spec))
            }),
        _ => false,
    }
}

fn canonical_expr_op(op: &str) -> &str {
    match op {
        "train_time" => "train_time_seconds",
        "inference_time" => "inference_time_seconds",
        "total_time" => "total_time_seconds",
        "train_memory" => "train_memory_bytes",
        "inference_memory" => "inference_memory_bytes",
        "total_memory" => "total_memory_bytes",
        other => other,
    }
}

fn collect_columns_spec(spec: &Value, out: &mut BTreeSet<String>) {
    match spec.get("metric").and_then(Value::as_str).unwrap_or("") {
        "custom" => {
            if let Some(expr) = spec.get("expression") {
                collect_columns_expr(expr, out);
            }
        }
        "weighted" => {
            if let Some(w) = spec.get("weights") {
                collect_columns_expr(w, out);
            }
            if let Some(f) = spec.get("fitness") {
                collect_columns_spec(f, out);
            }
        }
        "composite" => {
            if let Some(terms) = spec.get("terms").and_then(Value::as_array) {
                for t in terms {
                    if let Some(f) = t.get("fitness") {
                        collect_columns_spec(f, out);
                    }
                }
            }
        }
        "constrained" => {
            if let Some(f) = spec.get("fitness") {
                collect_columns_spec(f, out);
            }
        }
        _ => {}
    }
}

fn collect_columns_expr(expr: &Value, out: &mut BTreeSet<String>) {
    match canonical_expr_op(expr.get("op").and_then(Value::as_str).unwrap_or("")) {
        "column" => {
            if let Some(name) = expr.get("name").and_then(Value::as_str) {
                out.insert(name.to_owned());
            }
        }
        "abs" | "neg" | "square" | "sqrt" | "ln" | "log1p" | "exp" | "sign" | "tanh"
        | "sigmoid" | "cbrt" | "step" | "clip" => {
            if let Some(arg) = expr.get("arg") {
                collect_columns_expr(arg, out);
            }
        }
        "add" | "sub" | "mul" | "div" | "min" | "max" | "pow" => {
            if let Some(l) = expr.get("left") {
                collect_columns_expr(l, out);
            }
            if let Some(r) = expr.get("right") {
                collect_columns_expr(r, out);
            }
        }
        "where" => {
            for k in ["condition", "then", "else"] {
                if let Some(sub) = expr.get(k) {
                    collect_columns_expr(sub, out);
                }
            }
        }
        _ => {}
    }
}

fn insert_metric_resources(tag: &str, out: &mut BTreeSet<String>) {
    match tag {
        "train_time_seconds" => {
            out.insert("fit_seconds".to_owned());
        }
        "inference_time_seconds" => {
            out.insert("predict_seconds".to_owned());
        }
        "total_time_seconds" => {
            out.insert("fit_seconds".to_owned());
            out.insert("predict_seconds".to_owned());
        }
        "inference_us_per_row" => {
            out.insert("predict_us_per_row".to_owned());
        }
        "train_rows_per_second" => {
            out.insert("fit_seconds".to_owned());
            out.insert("train_rows".to_owned());
        }
        "inference_rows_per_second" => {
            out.insert("predict_rows_per_second".to_owned());
        }
        "train_memory_bytes" => {
            out.insert("fit_peak_heap_bytes".to_owned());
        }
        "inference_memory_bytes" => {
            out.insert("predict_peak_heap_bytes".to_owned());
        }
        "total_memory_bytes" => {
            out.insert("fit_peak_heap_bytes".to_owned());
            out.insert("predict_peak_heap_bytes".to_owned());
        }
        "model_bytes" => {
            out.insert("model_bytes".to_owned());
        }
        "complexity_stages" => {
            out.insert("stage_count".to_owned());
        }
        "complexity_parameters" => {
            out.insert("program_nodes".to_owned());
        }
        _ => {}
    }
}

fn collect_resources_spec(spec: &Value, out: &mut BTreeSet<String>) {
    let metric = spec.get("metric").and_then(Value::as_str).unwrap_or("");
    match metric {
        "resource" => {
            if let Some(n) = spec.get("name").and_then(Value::as_str) {
                out.insert(n.to_owned());
            }
        }
        "custom" => {
            if let Some(expr) = spec.get("expression") {
                collect_resources_expr(expr, out);
            }
        }
        "weighted" => {
            if let Some(w) = spec.get("weights") {
                collect_resources_expr(w, out);
            }
            if let Some(f) = spec.get("fitness") {
                collect_resources_spec(f, out);
            }
        }
        "composite" => {
            if let Some(terms) = spec.get("terms").and_then(Value::as_array) {
                for t in terms {
                    if let Some(f) = t.get("fitness") {
                        collect_resources_spec(f, out);
                    }
                }
            }
        }
        "constrained" => {
            if let Some(limits) = spec.get("limits").and_then(Value::as_array) {
                for l in limits {
                    if let Some(r) = l.get("resource").and_then(Value::as_str) {
                        out.insert(r.to_owned());
                    }
                }
            }
            if let Some(f) = spec.get("fitness") {
                collect_resources_spec(f, out);
            }
        }
        other => insert_metric_resources(other, out),
    }
}

fn collect_resources_expr(expr: &Value, out: &mut BTreeSet<String>) {
    let op = canonical_expr_op(expr.get("op").and_then(Value::as_str).unwrap_or(""));
    match op {
        "resource" => {
            if let Some(n) = expr.get("name").and_then(Value::as_str) {
                out.insert(n.to_owned());
            }
        }
        "abs" | "neg" | "square" | "sqrt" | "ln" | "log1p" | "exp" | "sign" | "tanh"
        | "sigmoid" | "cbrt" | "step" | "clip" => {
            if let Some(arg) = expr.get("arg") {
                collect_resources_expr(arg, out);
            }
        }
        "add" | "sub" | "mul" | "div" | "min" | "max" | "pow" => {
            if let Some(l) = expr.get("left") {
                collect_resources_expr(l, out);
            }
            if let Some(r) = expr.get("right") {
                collect_resources_expr(r, out);
            }
        }
        "where" => {
            for k in ["condition", "then", "else"] {
                if let Some(sub) = expr.get(k) {
                    collect_resources_expr(sub, out);
                }
            }
        }
        other => insert_metric_resources(other, out),
    }
}

fn uses_telemetry_only_spec(spec: &Value) -> bool {
    match spec.get("metric").and_then(Value::as_str).unwrap_or("") {
        "feature_count" | "fold_std" => true,
        "custom" => spec.get("expression").is_some_and(uses_telemetry_only_expr),
        "weighted" => {
            spec.get("weights").is_some_and(uses_telemetry_only_expr)
                || spec.get("fitness").is_some_and(uses_telemetry_only_spec)
        }
        "composite" => spec
            .get("terms")
            .and_then(Value::as_array)
            .is_some_and(|terms| {
                terms
                    .iter()
                    .any(|t| t.get("fitness").is_some_and(uses_telemetry_only_spec))
            }),
        "constrained" => spec.get("fitness").is_some_and(uses_telemetry_only_spec),
        _ => false,
    }
}

fn uses_telemetry_only_expr(expr: &Value) -> bool {
    match canonical_expr_op(expr.get("op").and_then(Value::as_str).unwrap_or("")) {
        "feature_count" | "fold_std" => true,
        "abs" | "neg" | "square" | "sqrt" | "ln" | "log1p" | "exp" | "sign" | "tanh"
        | "sigmoid" | "cbrt" | "step" | "clip" => {
            expr.get("arg").is_some_and(uses_telemetry_only_expr)
        }
        "add" | "sub" | "mul" | "div" | "min" | "max" | "pow" => {
            expr.get("left").is_some_and(uses_telemetry_only_expr)
                || expr.get("right").is_some_and(uses_telemetry_only_expr)
        }
        "where" => {
            expr.get("condition").is_some_and(uses_telemetry_only_expr)
                || expr.get("then").is_some_and(uses_telemetry_only_expr)
                || expr.get("else").is_some_and(uses_telemetry_only_expr)
        }
        _ => false,
    }
}

fn validate_spec_at(
    spec: &Value,
    depth: usize,
    nodes: &mut usize,
    expr_nodes: &mut usize,
    weighted: bool,
) -> Result<(), String> {
    *nodes += 1;
    if depth > 8 || *nodes > 64 {
        return Err("fitness composition exceeds depth/node limit".to_owned());
    }
    let obj = spec
        .as_object()
        .ok_or_else(|| "fitness specification must be a JSON object".to_owned())?;
    let metric = obj
        .get("metric")
        .and_then(Value::as_str)
        .ok_or_else(|| "fitness specification requires string field \"metric\"".to_owned())?;
    let allowed = expected_fields_for_metric(metric)
        .ok_or_else(|| format!("unknown fitness metric {metric:?}"))?;
    check_exact_keys(obj, allowed, metric)?;

    if weighted && !supports_weights_spec(spec) {
        return Err(format!(
            "{} does not accept row weights",
            metric_name_of(spec)
        ));
    }

    let check_strategy = |obj: &Map<String, Value>| -> Result<(), String> {
        let mp = req_f64(obj, "max_position")?;
        let cost = req_f64(obj, "cost")?;
        if !is_pos(mp) || !cost.is_finite() || cost < 0.0 {
            return Err(
                "position must be positive; cost must be nonnegative; both finite".to_owned(),
            );
        }
        Ok(())
    };

    match metric {
        "mape" | "smape" | "rmspe" => {
            if !is_pos(req_f64(obj, "epsilon")?) {
                return Err("epsilon must be positive and finite".to_owned());
            }
        }
        "huber" | "pseudo_huber" => {
            if !is_pos(req_f64(obj, "delta")?) {
                return Err("delta must be positive and finite".to_owned());
            }
        }
        "pinball" | "expectile" => {
            if !is_prob_open(req_f64(obj, "quantile")?) {
                return Err("quantile must be in (0,1)".to_owned());
            }
        }
        "tweedie_deviance" => {
            let p = req_f64(obj, "power")?;
            if !(p.is_finite() && p > 1.0 && p < 2.0) {
                return Err(
                    "Tweedie power must be in (1,2); use Poisson/Gamma at endpoints".to_owned(),
                );
            }
        }
        "linex" => {
            let a = req_f64(obj, "a")?;
            if !a.is_finite() || a == 0.0 {
                return Err("LinEx a must be finite and nonzero".to_owned());
            }
        }
        "asymmetric_absolute" | "asymmetric_squared" => {
            if !is_pos(req_f64(obj, "under_cost")?) || !is_pos(req_f64(obj, "over_cost")?) {
                return Err("under_cost and over_cost must be positive and finite".to_owned());
            }
        }
        "mase" => {
            let season = obj
                .get("season")
                .and_then(Value::as_u64)
                .ok_or_else(|| "season must be an integer in 1..=10000".to_owned())?;
            if !(1..=10_000).contains(&season) {
                return Err("season must be in 1..=10000".to_owned());
            }
        }
        "mean_return" | "max_drawdown" => check_strategy(obj)?,
        "sharpe" | "sortino" => {
            check_strategy(obj)?;
            let target = req_f64(obj, "target")?;
            let ppy = req_f64(obj, "periods_per_year")?;
            if !target.is_finite() || !is_pos(ppy) {
                return Err("target must be finite and periods_per_year positive".to_owned());
            }
        }
        "calmar" => {
            check_strategy(obj)?;
            if !is_pos(req_f64(obj, "periods_per_year")?) {
                return Err("periods_per_year must be positive".to_owned());
            }
        }
        "omega" => {
            check_strategy(obj)?;
            if !req_f64(obj, "threshold")?.is_finite() {
                return Err("threshold must be finite".to_owned());
            }
        }
        "expected_shortfall" => {
            check_strategy(obj)?;
            if !is_prob_open(req_f64(obj, "alpha")?) {
                return Err("alpha must be in (0,1)".to_owned());
            }
        }
        "capture_at" => {
            let f = req_f64(obj, "fraction")?;
            if !(f.is_finite() && f > 0.0 && f <= 1.0) {
                return Err("fraction must be in (0,1]".to_owned());
            }
        }
        "resource" => {
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "resource metric requires string field \"name\"".to_owned())?;
            if !VALID_RESOURCES.contains(&name) {
                return Err(format!("unknown resource {name:?}"));
            }
        }
        "custom" => {
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "custom metric requires string \"name\"".to_owned())?;
            if name.is_empty()
                || name.len() > 64
                || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            {
                return Err(
                    "custom name needs 1..64 ASCII letters, digits or underscores".to_owned(),
                );
            }
            let dir = obj
                .get("direction")
                .and_then(Value::as_str)
                .ok_or_else(|| "custom metric requires direction".to_owned())?;
            if !matches!(dir, "minimize" | "maximize") {
                return Err(format!("invalid direction {dir:?}"));
            }
            validate_expr_at(&obj["expression"], 0, expr_nodes)?;
            validate_aggregation(&obj["aggregation"])?;
        }
        "weighted" => {
            if weighted {
                return Err("weighting cannot be nested".to_owned());
            }
            validate_expr_at(&obj["weights"], 0, expr_nodes)?;
            validate_spec_at(&obj["fitness"], depth + 1, nodes, expr_nodes, true)?;
        }
        "composite" => {
            let terms = obj
                .get("terms")
                .and_then(Value::as_array)
                .ok_or_else(|| "composite terms must be an array".to_owned())?;
            if terms.is_empty() || terms.len() > 16 {
                return Err("composite needs 1..16 terms".to_owned());
            }
            for t in terms {
                let tobj = t
                    .as_object()
                    .ok_or_else(|| "composite term must be an object".to_owned())?;
                check_exact_keys(tobj, &["weight", "scale", "fitness"], "composite term")?;
                if !is_pos(req_f64(tobj, "weight")?) || !is_pos(req_f64(tobj, "scale")?) {
                    return Err("term weight and scale must be positive and finite".to_owned());
                }
                validate_spec_at(&tobj["fitness"], depth + 1, nodes, expr_nodes, weighted)?;
            }
        }
        "constrained" => {
            let limits = obj
                .get("limits")
                .and_then(Value::as_array)
                .ok_or_else(|| "constrained limits must be an array".to_owned())?;
            if limits.is_empty() || limits.len() > 16 {
                return Err("constrained needs 1..16 resource limits".to_owned());
            }
            for l in limits {
                let lobj = l
                    .as_object()
                    .ok_or_else(|| "resource limit must be an object".to_owned())?;
                for k in lobj.keys() {
                    if !matches!(k.as_str(), "resource" | "max" | "min") {
                        return Err(format!("unknown field {k:?} in resource limit"));
                    }
                }
                let res = lobj
                    .get("resource")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "resource limit requires \"resource\"".to_owned())?;
                if !VALID_RESOURCES.contains(&res) {
                    return Err(format!("unknown resource {res:?}"));
                }
                let min_v = match lobj.get("min") {
                    None | Some(Value::Null) => None,
                    Some(v) => Some(
                        v.as_f64()
                            .ok_or_else(|| format!("limit min on {res} must be a number"))?,
                    ),
                };
                let max_v = match lobj.get("max") {
                    None | Some(Value::Null) => None,
                    Some(v) => Some(
                        v.as_f64()
                            .ok_or_else(|| format!("limit max on {res} must be a number"))?,
                    ),
                };
                if min_v.is_none() && max_v.is_none() {
                    return Err(format!("limit on {res} needs at least one of min or max"));
                }
                if let Some(mn) = min_v {
                    if !mn.is_finite() || mn < 0.0 {
                        return Err(format!("limit min on {res} must be finite and nonnegative"));
                    }
                }
                if let Some(mx) = max_v {
                    if !mx.is_finite() || mx < 0.0 {
                        return Err(format!("limit max on {res} must be finite and nonnegative"));
                    }
                }
                if let (Some(mn), Some(mx)) = (min_v, max_v) {
                    if mn > mx {
                        return Err(format!("limit on {res} has min ({mn}) > max ({mx})"));
                    }
                }
            }
            validate_spec_at(&obj["fitness"], depth + 1, nodes, expr_nodes, weighted)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_aggregation(agg: &Value) -> Result<(), String> {
    let obj = agg
        .as_object()
        .ok_or_else(|| "aggregation must be an object".to_owned())?;
    let kind = obj
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| "aggregation requires \"kind\"".to_owned())?;
    match kind {
        "mean" | "sum" | "min" | "max" | "median" | "std" | "variance" | "root_mean" => {
            check_exact_keys(obj, &["kind"], "aggregation")
        }
        "quantile" => {
            check_exact_keys(obj, &["kind", "quantile"], "aggregation")?;
            let q = req_f64(obj, "quantile")?;
            if !(q.is_finite() && (0.0..=1.0).contains(&q)) {
                return Err("aggregation quantile must be in [0,1]".to_owned());
            }
            Ok(())
        }
        "tail_mean" => {
            check_exact_keys(obj, &["kind", "fraction"], "aggregation")?;
            let f = req_f64(obj, "fraction")?;
            if !(f.is_finite() && f > 0.0 && f <= 1.0) {
                return Err("tail_mean fraction must be in (0,1]".to_owned());
            }
            Ok(())
        }
        other => Err(format!("unknown aggregation kind {other:?}")),
    }
}

fn validate_expr_at(expr: &Value, depth: usize, nodes: &mut usize) -> Result<(), String> {
    *nodes += 1;
    if depth > 16 || *nodes > MAX_OBJECTIVE_EXPR_NODES {
        return Err(format!(
            "fitness expressions exceed depth 16 or {MAX_OBJECTIVE_EXPR_NODES} nodes in one specification"
        ));
    }
    let obj = expr
        .as_object()
        .ok_or_else(|| "expression node must be a JSON object".to_owned())?;
    let raw_op = obj
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| "expression node requires string \"op\"".to_owned())?;
    let op = canonical_expr_op(raw_op);
    match op {
        "y"
        | "pred"
        | "train_time_seconds"
        | "inference_time_seconds"
        | "total_time_seconds"
        | "inference_us_per_row"
        | "train_rows_per_second"
        | "inference_rows_per_second"
        | "train_memory_bytes"
        | "inference_memory_bytes"
        | "total_memory_bytes"
        | "model_bytes"
        | "complexity_stages"
        | "complexity_parameters"
        | "feature_count"
        | "fold_std"
        | "prediction_turnover"
        | "row_index"
        | "row_fraction"
        | "row_count"
        | "residual"
        | "abs_error"
        | "squared_error"
        | "y_mean"
        | "y_std"
        | "pred_mean"
        | "pred_std" => check_exact_keys(obj, &["op"], op),
        "constant" => {
            check_exact_keys(obj, &["op", "value"], op)?;
            ensure_finite(req_f64(obj, "value")?).map(|_| ())
        }
        "column" => {
            check_exact_keys(obj, &["op", "name"], op)?;
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "column op requires string \"name\"".to_owned())?;
            if name.is_empty()
                || name.len() > MAX_COLUMN_NAME_BYTES
                || name.chars().any(char::is_control)
            {
                return Err(format!(
                    "column names need 1..{MAX_COLUMN_NAME_BYTES} bytes without control characters"
                ));
            }
            Ok(())
        }
        "resource" => {
            check_exact_keys(obj, &["op", "name"], op)?;
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "resource op requires string \"name\"".to_owned())?;
            if !VALID_RESOURCES.contains(&name) {
                return Err(format!("unknown resource {name:?}"));
            }
            Ok(())
        }
        "abs" | "neg" | "square" | "sqrt" | "ln" | "log1p" | "exp" | "sign" | "tanh"
        | "sigmoid" | "cbrt" | "step" => {
            check_exact_keys(obj, &["op", "arg"], op)?;
            validate_expr_at(&obj["arg"], depth + 1, nodes)
        }
        "clip" => {
            check_exact_keys(obj, &["op", "arg", "min", "max"], op)?;
            let mn = req_f64(obj, "min")?;
            let mx = req_f64(obj, "max")?;
            if !mn.is_finite() || !mx.is_finite() || mn > mx {
                return Err("clip needs finite min <= max".to_owned());
            }
            validate_expr_at(&obj["arg"], depth + 1, nodes)
        }
        "add" | "sub" | "mul" | "div" | "min" | "max" | "pow" => {
            check_exact_keys(obj, &["op", "left", "right"], op)?;
            validate_expr_at(&obj["left"], depth + 1, nodes)?;
            validate_expr_at(&obj["right"], depth + 1, nodes)
        }
        "where" => {
            check_exact_keys(obj, &["op", "condition", "then", "else"], op)?;
            validate_expr_at(&obj["condition"], depth + 1, nodes)?;
            validate_expr_at(&obj["then"], depth + 1, nodes)?;
            validate_expr_at(&obj["else"], depth + 1, nodes)
        }
        other => Err(format!("unknown expression op {other:?}")),
    }
}

struct CanonicalNumFmt;
impl serde_json::ser::Formatter for CanonicalNumFmt {
    fn write_f64<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        value: f64,
    ) -> std::io::Result<()> {
        let v = if value == 0.0 { 0.0 } else { value };
        serde_json::ser::CompactFormatter.write_f64(writer, v)
    }
}

fn write_json_scalar(v: &Value, out: &mut Vec<u8>) {
    serde::Serialize::serialize(
        v,
        &mut serde_json::Serializer::with_formatter(out, CanonicalNumFmt),
    )
    .expect("serialize scalar");
}

fn write_canonical_spec(spec: &Value, out: &mut Vec<u8>) {
    let obj = spec.as_object().expect("validated spec object");
    let metric = obj["metric"].as_str().expect("validated metric");
    let fields = expected_fields_for_metric(metric).expect("validated metric fields");
    out.push(b'{');
    for (idx, &key) in fields.iter().enumerate() {
        if idx > 0 {
            out.push(b',');
        }
        write_json_scalar(&Value::String(key.to_owned()), out);
        out.push(b':');
        let val = &obj[key];
        match (metric, key) {
            (_, "season") => {
                let u = val.as_u64().expect("validated season");
                write_json_scalar(&json!(u), out);
            }
            ("custom", "expression") | ("weighted", "weights") => write_canonical_expr(val, out),
            ("custom", "aggregation") => write_canonical_agg(val, out),
            ("weighted" | "constrained", "fitness") => write_canonical_spec(val, out),
            ("composite", "terms") => {
                out.push(b'[');
                for (t_idx, term) in val.as_array().expect("terms").iter().enumerate() {
                    if t_idx > 0 {
                        out.push(b',');
                    }
                    let tobj = term.as_object().expect("term");
                    out.extend_from_slice(b"{\"weight\":");
                    write_f64_val(&tobj["weight"], out);
                    out.extend_from_slice(b",\"scale\":");
                    write_f64_val(&tobj["scale"], out);
                    out.extend_from_slice(b",\"fitness\":");
                    write_canonical_spec(&tobj["fitness"], out);
                    out.push(b'}');
                }
                out.push(b']');
            }
            ("constrained", "limits") => {
                out.push(b'[');
                for (l_idx, lim) in val.as_array().expect("limits").iter().enumerate() {
                    if l_idx > 0 {
                        out.push(b',');
                    }
                    let lobj = lim.as_object().expect("limit");
                    out.extend_from_slice(b"{\"resource\":");
                    write_json_scalar(&lobj["resource"], out);
                    if let Some(mx) = lobj.get("max").filter(|v| !v.is_null()) {
                        out.extend_from_slice(b",\"max\":");
                        write_f64_val(mx, out);
                    }
                    if let Some(mn) = lobj.get("min").filter(|v| !v.is_null()) {
                        out.extend_from_slice(b",\"min\":");
                        write_f64_val(mn, out);
                    }
                    out.push(b'}');
                }
                out.push(b']');
            }
            _ if val.is_number() => write_f64_val(val, out),
            _ => write_json_scalar(val, out),
        }
    }
    out.push(b'}');
}

fn write_f64_val(val: &Value, out: &mut Vec<u8>) {
    let f = val.as_f64().expect("validated number");
    serde::Serialize::serialize(
        &f,
        &mut serde_json::Serializer::with_formatter(out, CanonicalNumFmt),
    )
    .expect("serialize f64");
}

fn write_canonical_agg(agg: &Value, out: &mut Vec<u8>) {
    let obj = agg.as_object().expect("agg");
    let kind = obj["kind"].as_str().expect("kind");
    out.extend_from_slice(b"{\"kind\":");
    write_json_scalar(&obj["kind"], out);
    match kind {
        "quantile" => {
            out.extend_from_slice(b",\"quantile\":");
            write_f64_val(&obj["quantile"], out);
        }
        "tail_mean" => {
            out.extend_from_slice(b",\"fraction\":");
            write_f64_val(&obj["fraction"], out);
        }
        _ => {}
    }
    out.push(b'}');
}

fn write_canonical_expr(expr: &Value, out: &mut Vec<u8>) {
    let obj = expr.as_object().expect("expr");
    let op = canonical_expr_op(obj["op"].as_str().expect("op"));
    out.extend_from_slice(b"{\"op\":");
    write_json_scalar(&Value::String(op.to_owned()), out);
    match op {
        "constant" => {
            out.extend_from_slice(b",\"value\":");
            write_f64_val(&obj["value"], out);
        }
        "column" | "resource" => {
            out.extend_from_slice(b",\"name\":");
            write_json_scalar(&obj["name"], out);
        }
        "abs" | "neg" | "square" | "sqrt" | "ln" | "log1p" | "exp" | "sign" | "tanh"
        | "sigmoid" | "cbrt" | "step" => {
            out.extend_from_slice(b",\"arg\":");
            write_canonical_expr(&obj["arg"], out);
        }
        "clip" => {
            out.extend_from_slice(b",\"arg\":");
            write_canonical_expr(&obj["arg"], out);
            out.extend_from_slice(b",\"min\":");
            write_f64_val(&obj["min"], out);
            out.extend_from_slice(b",\"max\":");
            write_f64_val(&obj["max"], out);
        }
        "add" | "sub" | "mul" | "div" | "min" | "max" | "pow" => {
            out.extend_from_slice(b",\"left\":");
            write_canonical_expr(&obj["left"], out);
            out.extend_from_slice(b",\"right\":");
            write_canonical_expr(&obj["right"], out);
        }
        "where" => {
            out.extend_from_slice(b",\"condition\":");
            write_canonical_expr(&obj["condition"], out);
            out.extend_from_slice(b",\"then\":");
            write_canonical_expr(&obj["then"], out);
            out.extend_from_slice(b",\"else\":");
            write_canonical_expr(&obj["else"], out);
        }
        _ => {}
    }
    out.push(b'}');
}

fn eval_spec_value(
    spec: &Value,
    y: &[f64],
    p: &[f64],
    columns: &BTreeMap<String, Vec<f64>>,
    w: Option<&[f64]>,
) -> Result<f64, String> {
    if w.is_some() && !supports_weights_spec(spec) {
        return Err(format!(
            "{} does not accept row weights",
            metric_name_of(spec)
        ));
    }
    let obj = spec.as_object().expect("validated");
    let metric = obj["metric"].as_str().expect("metric");
    let errors: Vec<f64> = y.iter().zip(p).map(|(yi, pi)| yi - pi).collect();
    if !matches!(
        metric,
        "custom"
            | "weighted"
            | "composite"
            | "constrained"
            | "resource"
            | "mean_return"
            | "sharpe"
            | "sortino"
            | "max_drawdown"
            | "calmar"
            | "omega"
            | "expected_shortfall"
            | "pearson"
            | "spearman"
            | "hit_rate"
            | "normalized_gini"
            | "kge"
            | "capture_at"
            | "train_time_seconds"
            | "inference_time_seconds"
            | "total_time_seconds"
            | "inference_us_per_row"
            | "train_rows_per_second"
            | "inference_rows_per_second"
            | "train_memory_bytes"
            | "inference_memory_bytes"
            | "total_memory_bytes"
            | "model_bytes"
            | "complexity_stages"
            | "complexity_parameters"
            | "feature_count"
            | "fold_std"
            | "prediction_turnover"
    ) && !errors.iter().all(|e| e.is_finite())
    {
        return Err("non-finite regression residual".to_owned());
    }
    let abs: Vec<f64> = errors.iter().map(|e| e.abs()).collect();
    let squared = || weighted_mean(&errors.iter().map(|e| e * e).collect::<Vec<_>>(), w);

    Ok(match metric {
        "rmse" => squared().sqrt(),
        "mse" => squared(),
        "mae" => weighted_mean(&abs, w),
        "median_absolute_error" => weighted_quantile(&abs, w, 0.5)?,
        "max_error" => abs.iter().copied().fold(0.0, f64::max),
        "mape" => {
            let eps = obj["epsilon"].as_f64().unwrap();
            weighted_mean(
                &y.iter()
                    .zip(p)
                    .map(|(&yi, &pi)| (yi - pi).abs() / yi.abs().max(eps))
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "smape" => {
            let eps = obj["epsilon"].as_f64().unwrap();
            weighted_mean(
                &y.iter()
                    .zip(p)
                    .map(|(&yi, &pi)| {
                        let scale = yi.abs().max(pi.abs()).max(eps);
                        let denom = (yi.abs() / scale + pi.abs() / scale).max(eps / scale);
                        ((yi - pi).abs() / scale) * (2.0 / denom)
                    })
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "rmspe" => {
            let eps = obj["epsilon"].as_f64().unwrap();
            weighted_mean(
                &y.iter()
                    .zip(p)
                    .map(|(&yi, &pi)| ((yi - pi) / yi.abs().max(eps)).powi(2))
                    .collect::<Vec<_>>(),
                w,
            )
            .sqrt()
        }
        "wape" => {
            let den = weighted_mean(&y.iter().map(|yi| yi.abs()).collect::<Vec<_>>(), w);
            if !den.is_finite() || den <= 0.0 {
                return Err("WAPE undefined for all-zero targets".to_owned());
            }
            weighted_mean(&abs, w) / den
        }
        "msle" | "rmsle" => {
            if !y.iter().chain(p).all(|&x| x >= 0.0) {
                return Err("log error needs nonnegative targets and predictions".to_owned());
            }
            let v = weighted_mean(
                &y.iter()
                    .zip(p)
                    .map(|(&yi, &pi)| (yi.ln_1p() - pi.ln_1p()).powi(2))
                    .collect::<Vec<_>>(),
                w,
            );
            if metric == "rmsle" {
                v.sqrt()
            } else {
                v
            }
        }
        "r2" | "explained_variance" => {
            let den = weighted_variance(y, w);
            if !den.is_finite() {
                return Err("R2/explained variance overflows: the target variance is not finite".to_owned());
            }
            if y.len() < 2 || den <= 0.0 {
                return Err("R2/explained variance undefined for constant or singleton targets".to_owned());
            }
            1.0 - if metric == "r2" {
                squared() / den
            } else {
                weighted_variance(&errors, w) / den
            }
        }
        "pearson" => pearson_corr(y, p, w)?,
        "spearman" => pearson_corr(&midranks(y), &midranks(p), None)?,
        "huber" => {
            let delta = obj["delta"].as_f64().unwrap();
            weighted_mean(
                &errors
                    .iter()
                    .map(|&e| {
                        let a = e.abs();
                        if a <= delta {
                            0.5 * a * a
                        } else {
                            delta * (a - 0.5 * delta)
                        }
                    })
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "pseudo_huber" => {
            let delta = obj["delta"].as_f64().unwrap();
            weighted_mean(
                &errors
                    .iter()
                    .map(|&e| delta * delta * ((1.0 + (e / delta).powi(2)).sqrt() - 1.0))
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "log_cosh" => weighted_mean(
            &errors
                .iter()
                .map(|&e| {
                    let a = e.abs();
                    a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2
                })
                .collect::<Vec<_>>(),
            w,
        ),
        "pinball" => {
            let q = obj["quantile"].as_f64().unwrap();
            weighted_mean(
                &errors
                    .iter()
                    .map(|&e| if e >= 0.0 { q * e } else { (q - 1.0) * e })
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "expectile" => {
            let q = obj["quantile"].as_f64().unwrap();
            weighted_mean(
                &errors
                    .iter()
                    .map(|&e| if e >= 0.0 { q * e * e } else { (1.0 - q) * e * e })
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "linex" => {
            let a = obj["a"].as_f64().unwrap();
            weighted_mean(&errors.iter().map(|&e| (a * e).exp_m1() - a * e).collect::<Vec<_>>(), w)
        }
        "asymmetric_absolute" => {
            let uc = obj["under_cost"].as_f64().unwrap();
            let oc = obj["over_cost"].as_f64().unwrap();
            weighted_mean(
                &errors
                    .iter()
                    .map(|&e| if e >= 0.0 { uc * e } else { -oc * e })
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "asymmetric_squared" => {
            let uc = obj["under_cost"].as_f64().unwrap();
            let oc = obj["over_cost"].as_f64().unwrap();
            weighted_mean(
                &errors
                    .iter()
                    .map(|&e| if e >= 0.0 { uc * e * e } else { oc * e * e })
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "poisson_deviance" => {
            if !y.iter().all(|&x| x >= 0.0) || !p.iter().all(|&x| x > 0.0) {
                return Err("Poisson requires y>=0 and pred>0".to_owned());
            }
            weighted_mean(
                &y.iter()
                    .zip(p)
                    .map(|(&yi, &pi)| 2.0 * if yi == 0.0 { pi } else { yi * (yi / pi).ln() - yi + pi })
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "gamma_deviance" => {
            if !y.iter().chain(p).all(|&x| x > 0.0) {
                return Err("Gamma requires y>0 and pred>0".to_owned());
            }
            weighted_mean(
                &y.iter()
                    .zip(p)
                    .map(|(&yi, &pi)| 2.0 * (-(yi / pi).ln() + yi / pi - 1.0))
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "tweedie_deviance" => {
            if !y.iter().all(|&x| x >= 0.0) || !p.iter().all(|&x| x > 0.0) {
                return Err("Tweedie requires y>=0 and pred>0".to_owned());
            }
            let q = obj["power"].as_f64().unwrap();
            weighted_mean(
                &y.iter()
                    .zip(p)
                    .map(|(&yi, &pi)| {
                        2.0 * (yi.powf(2.0 - q) / ((1.0 - q) * (2.0 - q))
                            - yi * pi.powf(1.0 - q) / (1.0 - q)
                            + pi.powf(2.0 - q) / (2.0 - q))
                    })
                    .collect::<Vec<_>>(),
                w,
            )
        }
        "mase" => {
            let season = obj["season"].as_u64().unwrap() as usize;
            if y.len() <= season {
                return Err("MASE needs more scored rows than its season".to_owned());
            }
            let naive = plain_mean(
                &y.windows(season + 1)
                    .map(|win| (win[season] - win[0]).abs())
                    .collect::<Vec<_>>(),
            );
            if !naive.is_finite() || naive <= 0.0 {
                return Err("MASE undefined when the seasonal naive forecast is exact".to_owned());
            }
            plain_mean(&abs) / naive
        }
        "mean_return" => {
            let rets = net_strategy_returns(y, p, obj["max_position"].as_f64().unwrap(), obj["cost"].as_f64().unwrap())?;
            plain_mean(&rets)
        }
        "sharpe" | "sortino" => {
            let rets = net_strategy_returns(y, p, obj["max_position"].as_f64().unwrap(), obj["cost"].as_f64().unwrap())?;
            let target = obj["target"].as_f64().unwrap();
            let ppy = obj["periods_per_year"].as_f64().unwrap();
            let den = if metric == "sortino" {
                plain_mean(&rets.iter().map(|r| (r - target).min(0.0).powi(2)).collect::<Vec<_>>()).sqrt()
            } else {
                plain_variance(&rets).sqrt()
            };
            if !(den > 0.0 && den.is_finite()) {
                return Err("risk ratio undefined with a zero or non-finite risk denominator".to_owned());
            }
            ensure_finite((plain_mean(&rets) - target) / den * ppy.sqrt())?
        }
        "max_drawdown" => {
            let rets = net_strategy_returns(y, p, obj["max_position"].as_f64().unwrap(), obj["cost"].as_f64().unwrap())?;
            compute_max_drawdown(&rets)?
        }
        "calmar" => {
            let rets = net_strategy_returns(y, p, obj["max_position"].as_f64().unwrap(), obj["cost"].as_f64().unwrap())?;
            let dd = compute_max_drawdown(&rets)?;
            if !dd.is_finite() || dd <= 0.0 {
                return Err("Calmar undefined without a drawdown".to_owned());
            }
            let end = *compute_equity_curve(&rets)?.last().expect("nonempty");
            let ppy = obj["periods_per_year"].as_f64().unwrap();
            let growth = end.powf(ppy / rets.len() as f64) - 1.0;
            growth / dd
        }
        "omega" => {
            let rets = net_strategy_returns(y, p, obj["max_position"].as_f64().unwrap(), obj["cost"].as_f64().unwrap())?;
            let thr = obj["threshold"].as_f64().unwrap();
            let gains: f64 = rets.iter().map(|r| (r - thr).max(0.0)).sum();
            let losses: f64 = rets.iter().map(|r| (thr - r).max(0.0)).sum();
            if !gains.is_finite() || !losses.is_finite() {
                return Err("Omega overflowed".to_owned());
            }
            if losses <= 0.0 {
                return Err("Omega undefined without a shortfall".to_owned());
            }
            gains / losses
        }
        "expected_shortfall" => {
            let rets = net_strategy_returns(y, p, obj["max_position"].as_f64().unwrap(), obj["cost"].as_f64().unwrap())?;
            let losses: Vec<f64> = rets.iter().map(|r| -r).collect();
            weighted_tail_mean(&losses, None, obj["alpha"].as_f64().unwrap())?
        }
        "hit_rate" => {
            let weight = |i: usize| w.map_or(1.0, |ws| ws[i]);
            let (mut hits, mut total) = (0.0, 0.0);
            for i in (0..y.len()).filter(|&i| y[i] != 0.0) {
                total += weight(i);
                if p[i] != 0.0 && (p[i] > 0.0) == (y[i] > 0.0) {
                    hits += weight(i);
                }
            }
            if !total.is_finite() || total <= 0.0 {
                return Err("hit rate undefined for all-zero targets".to_owned());
            }
            hits / total
        }
        "normalized_gini" => {
            let model = compute_lorenz_gini(&ranked_pairs(y, p, w))?;
            let perfect = compute_lorenz_gini(&ranked_pairs(y, y, w))?;
            if perfect == 0.0 {
                return Err("normalized Gini undefined for constant targets".to_owned());
            }
            model / perfect
        }
        "kge" => {
            if y.len() < 2 {
                return Err("KGE needs at least two rows".to_owned());
            }
            let r = pearson_corr(y, p, None)?;
            let alpha = (plain_variance(p) / plain_variance(y)).sqrt();
            let my = plain_mean(y);
            if my == 0.0 {
                return Err("KGE undefined for a zero-mean target".to_owned());
            }
            let beta = plain_mean(p) / my;
            1.0 - ((r - 1.0).powi(2) + (alpha - 1.0).powi(2) + (beta - 1.0).powi(2)).sqrt()
        }
        "capture_at" => {
            let total: f64 = y.iter().sum();
            if !(total > 0.0 && total.is_finite()) {
                return Err("capture needs a positive target total".to_owned());
            }
            let mass = obj["fraction"].as_f64().unwrap() * y.len() as f64;
            let (mut taken, mut captured) = (0.0f64, 0.0);
            for (_, target) in ranked_pairs(y, p, None) {
                let take = (mass - taken).min(1.0);
                if take <= 0.0 {
                    break;
                }
                captured += take * target;
                taken += take;
            }
            ensure_finite(captured / total)?
        }
        "prediction_turnover" => calc_turnover(p),
        "custom" => {
            let rows = eval_expr_rows(&obj["expression"], y, p, columns)?;
            let agg_obj = obj["aggregation"].as_object().unwrap();
            match agg_obj["kind"].as_str().unwrap() {
                "mean" => weighted_mean(&rows, w),
                "sum" => match w {
                    None => rows.iter().sum(),
                    Some(ws) => rows.iter().zip(ws).map(|(v, wi)| v * wi).sum(),
                },
                "min" => rows.iter().copied().fold(f64::INFINITY, f64::min),
                "max" => rows.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "median" => weighted_quantile(&rows, w, 0.5)?,
                "std" => weighted_variance(&rows, w).max(0.0).sqrt(),
                "variance" => weighted_variance(&rows, w).max(0.0),
                "quantile" => weighted_quantile(&rows, w, agg_obj["quantile"].as_f64().unwrap())?,
                "tail_mean" => weighted_tail_mean(&rows, w, agg_obj["fraction"].as_f64().unwrap())?,
                "root_mean" => {
                    let m = weighted_mean(&rows, w);
                    if m < 0.0 {
                        return Err("root_mean needs a nonnegative mean".to_owned());
                    }
                    m.sqrt()
                }
                other => return Err(format!("unknown aggregation {other:?}")),
            }
        }
        "weighted" => {
            if w.is_some() {
                return Err("weighting cannot be nested".to_owned());
            }
            let ws = eval_expr_rows(&obj["weights"], y, p, columns)?;
            let total: f64 = ws.iter().sum();
            if !ws.iter().all(|&wi| wi >= 0.0) || !total.is_finite() || total <= 0.0 {
                return Err("weights must be nonnegative with a positive finite total".to_owned());
            }
            eval_spec_value(&obj["fitness"], y, p, columns, Some(&ws))?
        }
        "composite" => {
            let mut sum = 0.0;
            for t in obj["terms"].as_array().unwrap() {
                let tobj = t.as_object().unwrap();
                let natural = ensure_finite(eval_spec_value(&tobj["fitness"], y, p, columns, w)?)?;
                let loss = if direction_of(&tobj["fitness"]) == ObjectiveDirection::Maximize {
                    -natural
                } else {
                    natural
                };
                sum += tobj["weight"].as_f64().unwrap() * loss / tobj["scale"].as_f64().unwrap();
            }
            sum
        }
        other => {
            return Err(format!(
                "fitness metric {other:?} requires candidate execution telemetry and cannot be evaluated on standalone prediction arrays"
            ))
        }
    })
}

fn eval_expr_rows(
    expr: &Value,
    y: &[f64],
    p: &[f64],
    columns: &BTreeMap<String, Vec<f64>>,
) -> Result<Vec<f64>, String> {
    let ym = plain_mean(y);
    let ys = plain_variance(y).max(0.0).sqrt();
    let pm = plain_mean(p);
    let ps = plain_variance(p).max(0.0).sqrt();
    let turnover = calc_turnover(p);
    let n = y.len();
    (0..n)
        .map(|row| {
            eval_expr_row(
                expr,
                y[row],
                p[row],
                row,
                n,
                (ym, ys, pm, ps, turnover),
                columns,
            )
        })
        .collect()
}

fn eval_expr_row(
    expr: &Value,
    yi: f64,
    pi: f64,
    row: usize,
    n: usize,
    stats: (f64, f64, f64, f64, f64),
    columns: &BTreeMap<String, Vec<f64>>,
) -> Result<f64, String> {
    let obj = expr.as_object().unwrap();
    let op = canonical_expr_op(obj["op"].as_str().unwrap());
    let rec = |sub: &Value| eval_expr_row(sub, yi, pi, row, n, stats, columns);
    ensure_finite(match op {
        "y" => yi,
        "pred" => pi,
        "constant" => obj["value"].as_f64().unwrap(),
        "column" => {
            let name = obj["name"].as_str().unwrap();
            *columns
                .get(name)
                .and_then(|v| v.get(row))
                .ok_or_else(|| format!("fitness column {name:?} is not available"))?
        }
        "prediction_turnover" => stats.4,
        "row_index" => row as f64,
        "row_fraction" => {
            if n <= 1 {
                1.0
            } else {
                (row as f64) / ((n - 1) as f64)
            }
        }
        "row_count" => n as f64,
        "residual" => yi - pi,
        "abs_error" => (yi - pi).abs(),
        "squared_error" => (yi - pi).powi(2),
        "y_mean" => stats.0,
        "y_std" => stats.1,
        "pred_mean" => stats.2,
        "pred_std" => stats.3,
        "abs" => rec(&obj["arg"])?.abs(),
        "neg" => -rec(&obj["arg"])?,
        "square" => rec(&obj["arg"])?.powi(2),
        "sqrt" => rec(&obj["arg"])?.sqrt(),
        "ln" => rec(&obj["arg"])?.ln(),
        "log1p" => rec(&obj["arg"])?.ln_1p(),
        "exp" => rec(&obj["arg"])?.exp(),
        "sign" => {
            let v = rec(&obj["arg"])?;
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        "tanh" => rec(&obj["arg"])?.tanh(),
        "sigmoid" => 1.0 / (1.0 + (-rec(&obj["arg"])?).exp()),
        "cbrt" => rec(&obj["arg"])?.cbrt(),
        "step" => {
            if rec(&obj["arg"])? > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        "clip" => {
            rec(&obj["arg"])?.clamp(obj["min"].as_f64().unwrap(), obj["max"].as_f64().unwrap())
        }
        "add" => rec(&obj["left"])? + rec(&obj["right"])?,
        "sub" => rec(&obj["left"])? - rec(&obj["right"])?,
        "mul" => rec(&obj["left"])? * rec(&obj["right"])?,
        "div" => rec(&obj["left"])? / rec(&obj["right"])?,
        "min" => rec(&obj["left"])?.min(rec(&obj["right"])?),
        "max" => rec(&obj["left"])?.max(rec(&obj["right"])?),
        "pow" => rec(&obj["left"])?.powf(rec(&obj["right"])?),
        "where" => {
            if rec(&obj["condition"])? > 0.0 {
                rec(&obj["then"])?
            } else {
                rec(&obj["else"])?
            }
        }
        other => {
            return Err(format!(
                "expression op {other:?} requires candidate execution telemetry"
            ))
        }
    })
}

fn plain_mean(v: &[f64]) -> f64 {
    v.iter().map(|x| x / v.len() as f64).sum()
}

fn weighted_mean(v: &[f64], w: Option<&[f64]>) -> f64 {
    match w {
        None => plain_mean(v),
        Some(ws) => {
            let total: f64 = ws.iter().sum();
            v.iter().zip(ws).map(|(x, wi)| x * (wi / total)).sum()
        }
    }
}

fn plain_variance(v: &[f64]) -> f64 {
    let m = plain_mean(v);
    plain_mean(&v.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>())
}

fn weighted_variance(v: &[f64], w: Option<&[f64]>) -> f64 {
    let m = weighted_mean(v, w);
    weighted_mean(&v.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>(), w)
}

fn weighted_quantile(v: &[f64], w: Option<&[f64]>, q: f64) -> Result<f64, String> {
    let Some(ws) = w else {
        let mut sorted = v.to_vec();
        sorted.sort_by(f64::total_cmp);
        let p = q * (sorted.len() - 1) as f64;
        let i = p.floor() as usize;
        return Ok(sorted[i] + (sorted[p.ceil() as usize] - sorted[i]) * (p - i as f64));
    };
    let mut order: Vec<usize> = (0..v.len()).filter(|&i| ws[i] > 0.0).collect();
    if order.is_empty() {
        return Err("every row has zero weight".to_owned());
    }
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]).then(a.cmp(&b)));
    let last = order.len() - 1;
    let span: f64 = order[..last].iter().map(|&i| ws[i]).sum();
    if last == 0 || q <= 0.0 {
        return Ok(v[order[0]]);
    }
    let mut below = 0.0;
    let positions: Vec<f64> = order
        .iter()
        .map(|&i| {
            let pos = below / span;
            below += ws[i];
            pos
        })
        .collect();
    let k = positions.partition_point(|&pos| pos < q).clamp(1, last);
    let (p0, p1) = (positions[k - 1], positions[k]);
    let (v0, v1) = (v[order[k - 1]], v[order[k]]);
    Ok(v0 + (v1 - v0) * ((q - p0) / (p1 - p0)).min(1.0))
}

fn weighted_tail_mean(v: &[f64], w: Option<&[f64]>, fraction: f64) -> Result<f64, String> {
    let weight = |i: usize| w.map_or(1.0, |ws| ws[i]);
    let total: f64 = (0..v.len()).map(weight).sum();
    let mass = fraction * total;
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    let (mut taken, mut sum) = (0.0, 0.0);
    for i in order {
        let take = weight(i).min(mass - taken);
        if take <= 0.0 {
            if weight(i) > 0.0 {
                break;
            }
            continue;
        }
        sum += take * v[i];
        taken += take;
    }
    if !taken.is_finite() || taken <= 0.0 {
        return Err("tail holds no weight".to_owned());
    }
    ensure_finite(sum / taken)
}

fn pearson_corr(a: &[f64], b: &[f64], w: Option<&[f64]>) -> Result<f64, String> {
    let am = weighted_mean(a, w);
    let bm = weighted_mean(b, w);
    let av = weighted_variance(a, w);
    let bv = weighted_variance(b, w);
    if !av.is_finite() || !bv.is_finite() {
        return Err("correlation overflows: the variance of a vector is not finite".to_owned());
    }
    if !(av > 0.0 && bv > 0.0) {
        return Err("correlation undefined for a constant vector".to_owned());
    }
    ensure_finite(
        weighted_mean(
            &a.iter()
                .zip(b)
                .map(|(x, y)| (x - am) * (y - bm))
                .collect::<Vec<_>>(),
            w,
        ) / av.sqrt()
            / bv.sqrt(),
    )
}

fn midranks(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && v[order[i]] == v[order[j]] {
            j += 1;
        }
        for k in i..j {
            out[order[k]] = (i + j - 1) as f64 / 2.0;
        }
        i = j;
    }
    out
}

fn ranked_pairs(y: &[f64], key: &[f64], w: Option<&[f64]>) -> Vec<(f64, f64)> {
    let weight = |i: usize| w.map_or(1.0, |ws| ws[i]);
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| key[b].total_cmp(&key[a]));
    let mut out = Vec::with_capacity(order.len());
    for group in order.chunk_by(|&a, &b| key[a] == key[b]) {
        let total: f64 = group.iter().map(|&i| weight(i)).sum();
        let target = if total > 0.0 {
            group.iter().map(|&i| weight(i) * y[i]).sum::<f64>() / total
        } else {
            group.iter().map(|&i| y[i]).sum::<f64>() / group.len() as f64
        };
        out.extend(group.iter().map(|&i| (weight(i), target)));
    }
    out
}

fn compute_lorenz_gini(ranked: &[(f64, f64)]) -> Result<f64, String> {
    let total_w: f64 = ranked.iter().map(|(w, _)| w).sum();
    let total_y: f64 = ranked.iter().map(|(w, y)| w * y).sum();
    if !(total_w > 0.0 && total_y.is_finite() && total_y != 0.0) {
        return Err("Gini needs a nonzero weighted target total".to_owned());
    }
    let (mut x0, mut y0, mut area) = (0.0, 0.0, 0.0);
    for &(w, y) in ranked {
        let x1 = x0 + w / total_w;
        let y1 = y0 + w * y / total_y;
        area += (x1 - x0) * (y0 + y1) / 2.0;
        (x0, y0) = (x1, y1);
    }
    ensure_finite(2.0 * area - 1.0)
}

fn net_strategy_returns(
    y: &[f64],
    p: &[f64],
    max_position: f64,
    cost: f64,
) -> Result<Vec<f64>, String> {
    let mut last = 0.0;
    let rets: Vec<f64> = y
        .iter()
        .zip(p)
        .map(|(&yi, &pi)| {
            let pos = pi.clamp(-max_position, max_position);
            let r = pos * yi - cost * (pos - last).abs();
            last = pos;
            r
        })
        .collect();
    if !rets.iter().all(|x| x.is_finite()) {
        return Err("non-finite strategy return".to_owned());
    }
    Ok(rets)
}

fn compute_equity_curve(returns: &[f64]) -> Result<Vec<f64>, String> {
    let mut eq = 1.0f64;
    let mut curve = Vec::with_capacity(returns.len());
    for &r in returns {
        let next = eq * (1.0 + r);
        if !next.is_finite() {
            return Err("the equity curve overflowed".to_owned());
        }
        eq = next.max(0.0);
        curve.push(eq);
    }
    Ok(curve)
}

fn compute_max_drawdown(returns: &[f64]) -> Result<f64, String> {
    let mut peak = 1.0f64;
    let mut worst = 0.0f64;
    for e in compute_equity_curve(returns)? {
        peak = peak.max(e);
        worst = worst.max(1.0 - e / peak);
    }
    Ok(worst)
}

fn calc_turnover(p: &[f64]) -> f64 {
    if p.len() < 2 {
        0.0
    } else {
        p.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / ((p.len() - 1) as f64)
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_covers_all_53_builtin_entries_and_identities_match_golden() {
        let entries = ObjectiveSpec::catalog();
        assert_eq!(entries.len(), 53);
        let rmse = ObjectiveSpec::parse("rmse").expect("rmse");
        assert_eq!(rmse.to_wire_objective(), "rmse");
        let expected_rmse_hash = {
            let d = Sha256::digest(br#"["jope.regression-fitness/v1",{"metric":"rmse"}]"#);
            hex_lower(&d)
        };
        assert_eq!(rmse.identity_sha256(), expected_rmse_hash);

        let pinball = ObjectiveSpec::parse("pinball(0.9)").expect("pinball shorthand");
        let expected_pinball_hash = {
            let d = Sha256::digest(
                br#"["jope.regression-fitness/v1",{"metric":"pinball","quantile":0.9}]"#,
            );
            hex_lower(&d)
        };
        assert_eq!(pinball.identity_sha256(), expected_pinball_hash);
        assert!(pinball.benefits_from_post_hoc_calibration());
    }

    #[test]
    fn custom_auxiliary_column_spec_parses_validates_and_scores() {
        let raw = r#"{
            "metric": "custom",
            "name": "newsvendor_profit",
            "direction": "maximize",
            "expression": {
                "op": "sub",
                "left": {
                    "op": "mul",
                    "left": { "op": "column", "name": "price" },
                    "right": { "op": "min", "left": { "op": "y" }, "right": { "op": "pred" } }
                },
                "right": {
                    "op": "mul",
                    "left": { "op": "column", "name": "unit_cost" },
                    "right": { "op": "pred" }
                }
            },
            "aggregation": { "kind": "mean" }
        }"#;
        let spec = ObjectiveSpec::parse(raw).expect("valid custom spec");
        let cols: Vec<String> = spec.columns().into_iter().collect();
        assert_eq!(cols, vec!["price".to_owned(), "unit_cost".to_owned()]);
        let mut aux = BTreeMap::new();
        aux.insert("price".to_owned(), vec![10.0, 10.0]);
        aux.insert("unit_cost".to_owned(), vec![3.0, 3.0]);
        let score = spec
            .evaluate(&[5.0, 10.0], &[6.0, 8.0], &aux)
            .expect("evaluate custom");
        // row 0: 10*min(5,6) - 3*6 = 50 - 18 = 32
        // row 1: 10*min(10,8) - 3*8 = 80 - 24 = 56 -> mean = 44, loss = -44
        assert!((score.value - 44.0).abs() < 1e-12);
        assert!((score.loss - (-44.0)).abs() < 1e-12);
    }
}
