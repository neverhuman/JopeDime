use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct TimeSeriesTable {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub features: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeSeriesManifestSpec {
    pub horizon: usize,
    pub seasonality: usize,
    pub frequency: String,
    pub target_is_final_column: bool,
    pub known_future_covariates: Vec<String>,
    pub past_only_covariates: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct TimeSeriesData {
    pub train: TimeSeriesTable,
    pub test: TimeSeriesTable,
    pub horizon: usize,
    pub seasonality: usize,
    pub frequency: String,
    pub known_future_covariates: Vec<String>,
    pub mase_denominator: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TimeSeriesMetrics {
    pub mase: f64,
    pub rmse: f64,
    pub mse: f64,
    pub mae: f64,
    pub smape: f64,
    pub wape: f64,
    pub prediction_min: f64,
    pub prediction_max: f64,
    pub target_min: f64,
    pub target_max: f64,
    pub mase_denominator: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SupervisedReduction {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub feature_names: Vec<String>,
}

/// v1 is the frozen recursive-rollout reduction; v2 is the direct
/// multi-horizon reduction. The version is pinned in shard manifests and
/// receipts so downstream consumers never cross-compare scores produced by
/// different reductions.
pub const TS_REDUCTION_V1: &str = "ts-reduction-v1";
pub const TS_REDUCTION_V2: &str = "ts-reduction-v2";

/// Deterministic cap on v2 supervised rows: origins are consumed newest-first
/// until the budget fills, so long series train on recent regimes (where every
/// lag is fully observed) and NaN lag cells can only appear on short series.
pub const DIRECT_REDUCTION_MAX_ROWS: usize = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionVersion {
    V1,
    V2,
}

impl ReductionVersion {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            TS_REDUCTION_V1 => Ok(Self::V1),
            TS_REDUCTION_V2 => Ok(Self::V2),
            other => bail!(
                "unknown timeseries reduction version '{other}' (expected {TS_REDUCTION_V1} or {TS_REDUCTION_V2})"
            ),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1 => TS_REDUCTION_V1,
            Self::V2 => TS_REDUCTION_V2,
        }
    }
}

/// Direct multi-horizon supervised table. `y` and every target-derived
/// feature column are in MASE units (divided by `target_scale`); model
/// predictions must be multiplied by `target_scale` before scoring.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectReduction {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub feature_names: Vec<String>,
    pub target_scale: f64,
}

pub fn read_timeseries_csv(path: &Path) -> Result<TimeSeriesTable> {
    let bytes =
        fs::read(path).with_context(|| format!("read time-series CSV {}", path.display()))?;
    read_timeseries_csv_bytes(&bytes, &path.display().to_string())
}

pub fn read_timeseries_csv_bytes(bytes: &[u8], source: &str) -> Result<TimeSeriesTable> {
    let text = std::str::from_utf8(bytes)
        .with_context(|| format!("read time-series CSV {source} as UTF-8"))?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut features = None;
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut values = line
            .split(',')
            .map(|field| {
                field
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("{source}:{} invalid numeric field", line_no + 1))
            })
            .collect::<Result<Vec<_>>>()?;
        if values.is_empty() {
            bail!("{source}:{} needs at least one target column", line_no + 1);
        }
        let width = values.len() - 1;
        match features {
            Some(expected) if expected != width => bail!(
                "{source}:{} expected {expected} features, found {width}",
                line_no + 1
            ),
            None => features = Some(width),
            _ => {}
        }
        let target = values.pop().expect("length checked");
        if !target.is_finite() {
            bail!("{source}:{} target is not finite", line_no + 1);
        }
        if values.iter().any(|value| !value.is_finite()) {
            bail!(
                "{source}:{} known-future covariate is not finite",
                line_no + 1
            );
        }
        x.push(values);
        y.push(target);
    }
    let features = features.context("empty CSV")?;
    Ok(TimeSeriesTable { x, y, features })
}

pub fn validate_timeseries_data(
    train: TimeSeriesTable,
    test: TimeSeriesTable,
    meta: &Value,
    spec: &TimeSeriesManifestSpec,
) -> Result<TimeSeriesData> {
    validate_manifest_spec(spec)?;
    if train.features != test.features {
        bail!(
            "train/test feature widths differ ({} vs {})",
            train.features,
            test.features
        );
    }
    if test.y.len() != spec.horizon {
        bail!(
            "test row count {} does not equal horizon {}",
            test.y.len(),
            spec.horizon
        );
    }
    validate_meta(meta, &train, &test, spec)?;
    let denominator = mase_denominator(&train.y, spec.seasonality)?;
    if let Some(meta_denominator) = number_at(meta, &["target_summary", "mase_denominator"]) {
        let tolerance = 1e-6 * meta_denominator.abs().max(1.0);
        if (denominator - meta_denominator).abs() > tolerance {
            bail!(
                "computed MASE denominator {denominator} does not match meta target_summary.mase_denominator {meta_denominator}"
            );
        }
    }
    Ok(TimeSeriesData {
        train,
        test,
        horizon: spec.horizon,
        seasonality: spec.seasonality,
        frequency: spec.frequency.clone(),
        known_future_covariates: spec.known_future_covariates.clone(),
        mase_denominator: denominator,
    })
}

pub fn mase_denominator(train_y: &[f64], seasonality: usize) -> Result<f64> {
    if seasonality == 0 {
        bail!("seasonality must be positive");
    }
    if train_y.len() <= seasonality {
        bail!(
            "train target length {} must exceed seasonality {seasonality}",
            train_y.len()
        );
    }
    let denominator = (seasonality..train_y.len())
        .map(|row| (train_y[row] - train_y[row - seasonality]).abs())
        .sum::<f64>()
        / (train_y.len() - seasonality) as f64;
    if !denominator.is_finite() || denominator <= 0.0 {
        bail!("seasonal MASE denominator must be finite and nonzero");
    }
    Ok(denominator)
}

pub fn score_timeseries(data: &TimeSeriesData, predictions: &[f64]) -> Result<TimeSeriesMetrics> {
    if predictions.len() != data.horizon || predictions.len() != data.test.y.len() {
        bail!(
            "prediction count {} does not equal horizon {}",
            predictions.len(),
            data.horizon
        );
    }
    if predictions.iter().any(|prediction| !prediction.is_finite()) {
        bail!("time-series predictions contain non-finite values");
    }
    let n = predictions.len();
    let mut absolute_error = 0.0;
    let mut squared_error = 0.0;
    let mut smape_sum = 0.0;
    let mut target_abs_sum = 0.0;
    for (target, prediction) in data.test.y.iter().zip(predictions) {
        let error = (target - prediction).abs();
        absolute_error += error;
        squared_error += error * error;
        let scale = target.abs() + prediction.abs();
        if scale > 0.0 {
            smape_sum += error / scale;
        }
        target_abs_sum += target.abs();
    }
    let mae = absolute_error / n as f64;
    let mse = squared_error / n as f64;
    let rmse = mse.sqrt();
    let mase = mae / data.mase_denominator;
    let smape = 200.0 * smape_sum / n as f64;
    let wape = if target_abs_sum > 0.0 {
        absolute_error / target_abs_sum
    } else if absolute_error == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let (prediction_min, prediction_max) = range(predictions);
    let (target_min, target_max) = range(&data.test.y);
    let metrics = TimeSeriesMetrics {
        mase,
        rmse,
        mse,
        mae,
        smape,
        wape,
        prediction_min,
        prediction_max,
        target_min,
        target_max,
        mase_denominator: data.mase_denominator,
    };
    if [
        metrics.mase,
        metrics.rmse,
        metrics.mse,
        metrics.mae,
        metrics.smape,
        metrics.wape,
        metrics.prediction_min,
        metrics.prediction_max,
        metrics.target_min,
        metrics.target_max,
        metrics.mase_denominator,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        bail!("computed time-series metric is non-finite");
    }
    Ok(metrics)
}

pub fn make_supervised_reduction(data: &TimeSeriesData) -> Result<SupervisedReduction> {
    if data.train.y.len() < 2 {
        bail!("time-series reduction needs at least two train rows");
    }
    let feature_names = feature_names(data);
    let mut x = Vec::with_capacity(data.train.y.len() - 1);
    let mut y = Vec::with_capacity(data.train.y.len() - 1);
    for row in 1..data.train.y.len() {
        x.push(feature_row(
            data,
            &data.train.y[..row],
            &data.train.x[row],
            row,
            1,
        ));
        y.push(data.train.y[row]);
    }
    Ok(SupervisedReduction {
        x,
        y,
        feature_names,
    })
}

pub fn make_forecast_features(
    data: &TimeSeriesData,
    history: &[f64],
    horizon_index: usize,
) -> Result<Vec<f64>> {
    if horizon_index >= data.horizon {
        bail!(
            "horizon index {horizon_index} exceeds horizon {}",
            data.horizon
        );
    }
    if history.is_empty() {
        bail!("recursive forecast history is empty");
    }
    Ok(feature_row(
        data,
        history,
        &data.test.x[horizon_index],
        data.train.y.len() + horizon_index,
        horizon_index + 1,
    ))
}

/// v2 direct multi-horizon reduction: one supervised row per (origin, lead)
/// pair with `1 <= lead <= min(horizon, train_len - origin)`, features built
/// from the strict prefix `y[..origin]` only, target `y[origin + lead - 1]`.
/// Origins are consumed newest-first until `DIRECT_REDUCTION_MAX_ROWS`, then
/// rows are ordered chronologically by (origin, lead). The target and every
/// target-derived feature are divided by the MASE denominator so the head
/// fits in MASE units.
pub fn make_direct_reduction(data: &TimeSeriesData) -> Result<DirectReduction> {
    if data.train.y.len() < 2 {
        bail!("time-series reduction needs at least two train rows");
    }
    let train_len = data.train.y.len();
    let scale = data.mase_denominator;
    let scaled: Vec<f64> = data.train.y.iter().map(|value| value / scale).collect();
    let mut pairs = Vec::new();
    for origin in (1..train_len).rev() {
        let max_lead = data.horizon.min(train_len - origin);
        for lead in 1..=max_lead {
            pairs.push((origin, lead));
        }
        if pairs.len() >= DIRECT_REDUCTION_MAX_ROWS {
            break;
        }
    }
    pairs.truncate(DIRECT_REDUCTION_MAX_ROWS);
    pairs.sort_unstable();
    let feature_names = direct_feature_names(data);
    let mut x = Vec::with_capacity(pairs.len());
    let mut y = Vec::with_capacity(pairs.len());
    for (origin, lead) in pairs {
        let target_index = origin + lead - 1;
        x.push(direct_feature_row(
            data,
            &scaled[..origin],
            &data.train.x[target_index],
            target_index,
            lead,
            train_len,
        ));
        y.push(scaled[target_index]);
    }
    Ok(DirectReduction {
        x,
        y,
        feature_names,
        target_scale: scale,
    })
}

/// v2 forecast rows: H rows for lead 1..=H, every one built from the full
/// train history (no prediction feedback), predicted in a single batch call.
pub fn make_direct_forecast_rows(data: &TimeSeriesData) -> Result<Vec<Vec<f64>>> {
    if data.train.y.is_empty() {
        bail!("direct forecast requires nonempty train history");
    }
    let train_len = data.train.y.len();
    let scale = data.mase_denominator;
    let scaled: Vec<f64> = data.train.y.iter().map(|value| value / scale).collect();
    (1..=data.horizon)
        .map(|lead| {
            let known_future = data
                .test
                .x
                .get(lead - 1)
                .with_context(|| format!("test covariate row {} missing", lead - 1))?;
            Ok(direct_feature_row(
                data,
                &scaled,
                known_future,
                train_len + lead - 1,
                lead,
                train_len,
            ))
        })
        .collect()
}

fn direct_feature_names(data: &TimeSeriesData) -> Vec<String> {
    direct_feature_names_for(&data.known_future_covariates, data.train.features)
}

/// v2 feature name list from manifest-level facts alone, so campaign drivers
/// can build the schema without loading the CSVs.
pub fn direct_feature_names_for(
    known_future_covariates: &[String],
    csv_features: usize,
) -> Vec<String> {
    let mut names = known_future_covariates.to_vec();
    if names.is_empty() {
        names.extend((0..csv_features).map(|column| format!("known_future_{column}")));
    }
    names.extend(
        [
            "lag_1",
            "lag_2",
            "lag_3",
            "seasonal_lag_1",
            "seasonal_lag_2",
            "diff_1",
            "diff_seasonal",
            "last_minus_rollmean_3",
            "rolling_mean_3",
            "rolling_mean_seasonal",
            "rolling_std_3",
            "rolling_std_seasonal",
            "rolling_absdiff_3",
            "rolling_absdiff_seasonal",
            "hist_mean",
            "hist_std",
            "hist_min",
            "hist_max",
            "time_frac",
            "log1p_time",
            "lead_frac",
            "lead_over_seasonality",
            "log1p_lead",
            "seasonal_phase",
            "seasonal_sin",
            "seasonal_cos",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    names
}

/// Features from a (scaled) strict history prefix. Lags and diffs that reach
/// beyond the observed history are NaN — the advanced-v3 input layer maps NaN
/// cells to its missing mask, so imputation stays genome-controlled instead of
/// silently fabricating `history[0]`.
fn direct_feature_row(
    data: &TimeSeriesData,
    history: &[f64],
    known_future: &[f64],
    time_index: usize,
    lead: usize,
    train_len: usize,
) -> Vec<f64> {
    let seasonality = data.seasonality;
    let mut row = Vec::with_capacity(known_future.len() + 26);
    row.extend_from_slice(known_future);
    row.push(lag_or_nan(history, 1));
    row.push(lag_or_nan(history, 2));
    row.push(lag_or_nan(history, 3));
    row.push(lag_or_nan(history, seasonality));
    row.push(lag_or_nan(history, seasonality.saturating_mul(2)));
    let last = history[history.len() - 1];
    row.push(diff_or_nan(history, 1));
    row.push(diff_or_nan(history, seasonality));
    row.push(last - rolling_mean(history, 3));
    row.push(rolling_mean(history, 3));
    row.push(rolling_mean(history, seasonality));
    row.push(rolling_std(history, 3));
    row.push(rolling_std(history, seasonality));
    row.push(rolling_absdiff_or_nan(history, 3));
    row.push(rolling_absdiff_or_nan(history, seasonality));
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    row.push(mean);
    row.push(population_std(history, mean));
    let (min, max) = range(history);
    row.push(min);
    row.push(max);
    row.push(time_index as f64 / (train_len - 1).max(1) as f64);
    row.push((time_index as f64).ln_1p());
    row.push(lead as f64 / data.horizon.max(1) as f64);
    row.push(lead as f64 / seasonality.max(1) as f64);
    row.push((lead as f64).ln_1p());
    let phase = (time_index % seasonality) as f64 / seasonality as f64;
    row.push(phase);
    row.push((std::f64::consts::TAU * phase).sin());
    row.push((std::f64::consts::TAU * phase).cos());
    row
}

fn lag_or_nan(history: &[f64], lag: usize) -> f64 {
    let lag = lag.max(1);
    if history.len() >= lag {
        history[history.len() - lag]
    } else {
        f64::NAN
    }
}

/// `y[last] - y[last - step]`, NaN when the earlier point is unobserved.
fn diff_or_nan(history: &[f64], step: usize) -> f64 {
    let step = step.max(1);
    if history.len() > step {
        history[history.len() - 1] - history[history.len() - 1 - step]
    } else {
        f64::NAN
    }
}

fn rolling_std(history: &[f64], window: usize) -> f64 {
    let window = window.max(1).min(history.len());
    let start = history.len() - window;
    let segment = &history[start..];
    let mean = segment.iter().sum::<f64>() / window as f64;
    population_std(segment, mean)
}

fn population_std(values: &[f64], mean: f64) -> f64 {
    (values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn rolling_absdiff_or_nan(history: &[f64], window: usize) -> f64 {
    if history.len() < 2 {
        return f64::NAN;
    }
    rolling_absdiff(history, window)
}

fn validate_manifest_spec(spec: &TimeSeriesManifestSpec) -> Result<()> {
    if spec.horizon == 0 {
        bail!("time-series horizon must be positive");
    }
    if spec.seasonality == 0 {
        bail!("time-series seasonality must be positive");
    }
    if spec.frequency.trim().is_empty() {
        bail!("time-series frequency must be nonempty");
    }
    if !spec.target_is_final_column {
        bail!("time-series manifests must declare target_is_final_column=true");
    }
    if !spec.past_only_covariates.is_empty() {
        bail!("past-only covariates are not allowed in time-series CSV features");
    }
    let mut seen = BTreeSet::new();
    for name in &spec.known_future_covariates {
        if name.trim().is_empty() {
            bail!("known-future covariate names must be nonempty");
        }
        if !seen.insert(name) {
            bail!("duplicate known-future covariate '{name}'");
        }
    }
    Ok(())
}

fn validate_meta(
    meta: &Value,
    train: &TimeSeriesTable,
    test: &TimeSeriesTable,
    spec: &TimeSeriesManifestSpec,
) -> Result<()> {
    if meta.get("task_type").and_then(Value::as_str) != Some("timeseries") {
        bail!("meta_json task_type must be timeseries");
    }
    if meta.get("target_is_final_column").and_then(Value::as_bool) != Some(true) {
        bail!("meta_json target_is_final_column must be true");
    }
    require_usize(meta, "horizon", spec.horizon)?;
    require_usize(meta, "seasonality", spec.seasonality)?;
    if meta.get("frequency").and_then(Value::as_str) != Some(spec.frequency.as_str()) {
        bail!("meta_json frequency does not match manifest frequency");
    }
    if let Some(features) = optional_usize(meta, "n_features") {
        if features != train.features {
            bail!(
                "meta_json n_features {features} does not match CSV width {}",
                train.features
            );
        }
    }
    if let Some(features) = number_at(meta, &["counts", "n_features"]) {
        if features as usize != train.features {
            bail!(
                "meta_json counts.n_features {} does not match CSV width {}",
                features as usize,
                train.features
            );
        }
    }
    check_optional_usize(meta, "n_train_rows", train.y.len())?;
    check_optional_usize(meta, "n_test_rows", test.y.len())?;
    check_nested_optional_usize(meta, &["counts", "train_rows"], train.y.len())?;
    check_nested_optional_usize(meta, &["counts", "test_rows"], test.y.len())?;

    let feature_columns = string_array_at(meta, &["feature_columns"])?;
    let known_dynamic = string_array_at(meta, &["known_dynamic_columns"])?;
    if feature_columns.len() != train.features {
        bail!(
            "meta_json feature_columns length {} does not match CSV width {}",
            feature_columns.len(),
            train.features
        );
    }
    if feature_columns != known_dynamic {
        bail!("time-series CSV features must be exactly known-future covariates");
    }
    if spec.known_future_covariates != known_dynamic {
        bail!("manifest known_future_covariates do not match meta_json known_dynamic_columns");
    }
    let past_excluded = string_array_at(meta, &["past_dynamic_columns_excluded"])?;
    if feature_columns
        .iter()
        .any(|column| past_excluded.iter().any(|past| past == column))
    {
        bail!("past-only covariate appears in time-series feature columns");
    }
    if let Some(policy) = meta
        .get("cleaning")
        .and_then(|value| value.get("past_only_covariates"))
        .and_then(Value::as_str)
    {
        if policy != "excluded" {
            bail!("meta_json cleaning.past_only_covariates must be excluded");
        }
    }
    Ok(())
}

fn require_usize(meta: &Value, field: &str, expected: usize) -> Result<()> {
    let Some(actual) = optional_usize(meta, field) else {
        bail!("meta_json missing numeric field {field}");
    };
    if actual != expected {
        bail!("meta_json {field} {actual} does not match manifest {expected}");
    }
    Ok(())
}

fn optional_usize(meta: &Value, field: &str) -> Option<usize> {
    meta.get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn check_optional_usize(meta: &Value, field: &str, expected: usize) -> Result<()> {
    if let Some(actual) = optional_usize(meta, field) {
        if actual != expected {
            bail!("meta_json {field} {actual} does not match CSV rows {expected}");
        }
    }
    Ok(())
}

fn check_nested_optional_usize(meta: &Value, path: &[&str], expected: usize) -> Result<()> {
    if let Some(actual) = number_at(meta, path) {
        if actual as usize != expected {
            bail!(
                "meta_json {} {} does not match CSV rows {expected}",
                path.join("."),
                actual as usize
            );
        }
    }
    Ok(())
}

fn number_at(meta: &Value, path: &[&str]) -> Option<f64> {
    let mut cursor = meta;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor.as_f64()
}

fn string_array_at(meta: &Value, path: &[&str]) -> Result<Vec<String>> {
    let mut cursor = meta;
    for key in path {
        let Some(next) = cursor.get(*key) else {
            return Ok(Vec::new());
        };
        cursor = next;
    }
    let Some(values) = cursor.as_array() else {
        bail!("meta_json {} must be an array", path.join("."));
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("meta_json {} contains a non-string", path.join(".")))
        })
        .collect()
}

fn feature_names(data: &TimeSeriesData) -> Vec<String> {
    let mut names = data.known_future_covariates.clone();
    if names.is_empty() {
        names.extend((0..data.train.features).map(|column| format!("known_future_{column}")));
    }
    names.extend(
        [
            "lag_1",
            "lag_2",
            "lag_3",
            "seasonal_lag_1",
            "seasonal_lag_2",
            "rolling_mean_3",
            "rolling_mean_seasonal",
            "rolling_absdiff_3",
            "rolling_absdiff_seasonal",
            "row_index",
            "horizon_index",
            "seasonal_phase",
            "seasonal_sin",
            "seasonal_cos",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    names
}

fn feature_row(
    data: &TimeSeriesData,
    history: &[f64],
    known_future: &[f64],
    row_index: usize,
    horizon_index: usize,
) -> Vec<f64> {
    let mut row = Vec::with_capacity(known_future.len() + 14);
    row.extend_from_slice(known_future);
    row.push(lag(history, 1));
    row.push(lag(history, 2));
    row.push(lag(history, 3));
    row.push(lag(history, data.seasonality));
    row.push(lag(history, data.seasonality.saturating_mul(2).max(1)));
    row.push(rolling_mean(history, 3));
    row.push(rolling_mean(history, data.seasonality));
    row.push(rolling_absdiff(history, 3));
    row.push(rolling_absdiff(history, data.seasonality));
    row.push(row_index as f64);
    row.push(horizon_index as f64);
    let phase = (row_index % data.seasonality) as f64 / data.seasonality as f64;
    row.push(phase);
    row.push((std::f64::consts::TAU * phase).sin());
    row.push((std::f64::consts::TAU * phase).cos());
    row
}

fn lag(history: &[f64], lag: usize) -> f64 {
    let lag = lag.max(1);
    if history.len() >= lag {
        history[history.len() - lag]
    } else {
        history[0]
    }
}

fn rolling_mean(history: &[f64], window: usize) -> f64 {
    let window = window.max(1).min(history.len());
    let start = history.len() - window;
    history[start..].iter().sum::<f64>() / window as f64
}

fn rolling_absdiff(history: &[f64], window: usize) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let pairs = window.max(1).min(history.len() - 1);
    let start = history.len() - pairs;
    (start..history.len())
        .map(|row| (history[row] - history[row - 1]).abs())
        .sum::<f64>()
        / pairs as f64
}

fn range(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec() -> TimeSeriesManifestSpec {
        TimeSeriesManifestSpec {
            horizon: 3,
            seasonality: 2,
            frequency: "D".to_string(),
            target_is_final_column: true,
            known_future_covariates: vec!["promo".to_string()],
            past_only_covariates: Vec::new(),
        }
    }

    fn meta() -> Value {
        json!({
            "task_type": "timeseries",
            "horizon": 3,
            "seasonality": 2,
            "frequency": "D",
            "target_is_final_column": true,
            "n_features": 1,
            "n_train_rows": 5,
            "n_test_rows": 3,
            "feature_columns": ["promo"],
            "known_dynamic_columns": ["promo"],
            "past_dynamic_columns_excluded": ["customers"],
            "cleaning": {"past_only_covariates": "excluded"},
            "target_summary": {"mase_denominator": 2.0}
        })
    }

    #[test]
    fn mase_matches_hand_calculated_fixture() {
        let denominator = mase_denominator(&[10.0, 12.0, 14.0, 18.0], 2).unwrap();
        assert_eq!(denominator, 5.0);
        let train = TimeSeriesTable {
            x: vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0], vec![0.0]],
            y: vec![10.0, 12.0, 12.0, 14.0, 14.0],
            features: 1,
        };
        let test = TimeSeriesTable {
            x: vec![vec![1.0], vec![0.0], vec![1.0]],
            y: vec![16.0, 18.0, 20.0],
            features: 1,
        };
        let data = validate_timeseries_data(train, test, &meta(), &spec()).unwrap();
        let metrics = score_timeseries(&data, &[15.0, 18.0, 23.0]).unwrap();
        assert_eq!(metrics.mae, 4.0 / 3.0);
        assert_eq!(metrics.mase, (4.0 / 3.0) / 2.0);
    }

    #[test]
    fn loader_allows_univariate_target_last_csv() {
        let table = read_timeseries_csv_bytes(b"1\n2\n3\n", "fixture").unwrap();
        assert_eq!(table.features, 0);
        assert_eq!(table.y, vec![1.0, 2.0, 3.0]);
        assert_eq!(table.x, vec![Vec::<f64>::new(), Vec::new(), Vec::new()]);
    }

    #[test]
    fn bad_width_nonfinite_target_and_zero_denominator_are_rejected() {
        assert!(read_timeseries_csv_bytes(b"1,2\n1,2,3\n", "fixture").is_err());
        assert!(read_timeseries_csv_bytes(b"1,NaN\n", "fixture").is_err());
        assert!(mase_denominator(&[1.0, 1.0, 1.0], 1).is_err());
    }

    #[test]
    fn manifest_validation_rejects_past_only_and_target_not_last() {
        let mut bad = spec();
        bad.past_only_covariates = vec!["customers".to_string()];
        assert!(validate_timeseries_data(
            TimeSeriesTable {
                x: vec![vec![0.0]; 5],
                y: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                features: 1,
            },
            TimeSeriesTable {
                x: vec![vec![0.0]; 3],
                y: vec![6.0, 7.0, 8.0],
                features: 1,
            },
            &meta(),
            &bad,
        )
        .is_err());
        bad = spec();
        bad.target_is_final_column = false;
        assert!(validate_manifest_spec(&bad).is_err());
    }

    fn fixture_data() -> TimeSeriesData {
        let train = TimeSeriesTable {
            x: vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0], vec![0.0]],
            y: vec![10.0, 12.0, 12.0, 14.0, 14.0],
            features: 1,
        };
        let test = TimeSeriesTable {
            x: vec![vec![1.0], vec![0.0], vec![1.0]],
            y: vec![16.0, 18.0, 20.0],
            features: 1,
        };
        validate_timeseries_data(train, test, &meta(), &spec()).unwrap()
    }

    fn feature_index(names: &[String], name: &str) -> usize {
        names.iter().position(|n| n == name).unwrap()
    }

    #[test]
    fn direct_reduction_covers_all_leads_and_scales_targets() {
        let data = fixture_data();
        let reduction = make_direct_reduction(&data).unwrap();
        // origins 1..=4 with leads min(horizon=3, 5-origin):
        // (1,1)(1,2)(1,3)(2,1)(2,2)(2,3)(3,1)(3,2)(4,1)
        assert_eq!(reduction.x.len(), 9);
        assert_eq!(reduction.target_scale, 2.0);
        let lead_frac = feature_index(&reduction.feature_names, "lead_frac");
        let observed: BTreeSet<i64> = reduction
            .x
            .iter()
            .map(|row| (row[lead_frac] * 3.0).round() as i64)
            .collect();
        assert_eq!(observed, BTreeSet::from([1, 2, 3]), "train must see every lead 1..=H");
        // first sorted pair is (origin=1, lead=1): target y[1]=12.0 scaled by 2.0
        assert_eq!(reduction.y[0], 6.0);
        // last sorted pair is (origin=4, lead=1): target y[4]=14.0 scaled by 2.0
        assert_eq!(reduction.y[8], 7.0);
        // scaling round-trips exactly on this fixture
        for (scaled, raw) in reduction.y.iter().zip([12.0, 12.0, 14.0, 12.0, 14.0, 14.0, 14.0, 14.0, 14.0]) {
            assert_eq!(scaled * reduction.target_scale, raw);
        }
    }

    #[test]
    fn direct_reduction_unavailable_lags_are_nan_not_fabricated() {
        let data = fixture_data();
        let reduction = make_direct_reduction(&data).unwrap();
        let names = &reduction.feature_names;
        let first = &reduction.x[0]; // origin=1: single-point history
        for name in ["lag_2", "lag_3", "seasonal_lag_1", "seasonal_lag_2", "diff_1", "diff_seasonal", "rolling_absdiff_3"] {
            assert!(first[feature_index(names, name)].is_nan(), "{name} must be NaN on a 1-point history");
        }
        assert_eq!(first[feature_index(names, "lag_1")], 5.0); // y[0]=10.0 / 2.0
        let last = &reduction.x[8]; // origin=4: 4-point history, seasonality 2
        assert_eq!(last[feature_index(names, "lag_2")], 6.0); // y[2]=12.0 / 2.0
        assert_eq!(last[feature_index(names, "seasonal_lag_2")], 5.0); // lag 4 = y[0]
        assert_eq!(last[feature_index(names, "diff_1")], 1.0); // (14-12)/2
        assert_eq!(last[feature_index(names, "diff_seasonal")], 1.0); // (14-12)/2
    }

    #[test]
    fn direct_reduction_is_prefix_honest_on_monotone_series() {
        // strictly increasing series: every target strictly exceeds every
        // prefix value, so any leaked future value would show up in hist_max
        // or a lag exceeding the target.
        let n = 40;
        let y: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let train = TimeSeriesTable {
            x: vec![Vec::new(); n],
            y: y.clone(),
            features: 0,
        };
        let test = TimeSeriesTable {
            x: vec![Vec::new(); 3],
            y: vec![41.0, 42.0, 43.0],
            features: 0,
        };
        let denominator = mase_denominator(&y, 2).unwrap();
        let data = TimeSeriesData {
            train,
            test,
            horizon: 3,
            seasonality: 2,
            frequency: "D".to_string(),
            known_future_covariates: Vec::new(),
            mase_denominator: denominator,
        };
        let reduction = make_direct_reduction(&data).unwrap();
        let names = &reduction.feature_names;
        let hist_max = feature_index(names, "hist_max");
        let lag_1 = feature_index(names, "lag_1");
        for (row, target) in reduction.x.iter().zip(&reduction.y) {
            assert!(row[hist_max] < *target, "hist_max must predate the target");
            assert!(row[lag_1] < *target, "lag_1 must predate the target");
        }
    }

    #[test]
    fn direct_forecast_rows_use_full_history_without_feedback() {
        let data = fixture_data();
        let reduction = make_direct_reduction(&data).unwrap();
        let rows = make_direct_forecast_rows(&data).unwrap();
        assert_eq!(rows.len(), data.horizon);
        let names = &reduction.feature_names;
        let lag_1 = feature_index(names, "lag_1");
        let lead_frac = feature_index(names, "lead_frac");
        let promo = feature_index(names, "promo");
        for (h, row) in rows.iter().enumerate() {
            // every forecast row sees the same, real train history: no
            // prediction feedback anywhere.
            assert_eq!(row[lag_1], 7.0); // y[4]=14.0 / 2.0
            assert_eq!(row[lead_frac], (h + 1) as f64 / 3.0);
            assert_eq!(row[promo], data.test.x[h][0]);
            assert_eq!(row.len(), names.len());
        }
    }

    #[test]
    fn direct_reduction_caps_rows_newest_first() {
        let n = 9000;
        let y: Vec<f64> = (0..n).map(|i| ((i % 7) + 1) as f64 + i as f64 * 0.01).collect();
        let denominator = mase_denominator(&y, 7).unwrap();
        let data = TimeSeriesData {
            train: TimeSeriesTable {
                x: vec![Vec::new(); n],
                y: y.clone(),
                features: 0,
            },
            test: TimeSeriesTable {
                x: vec![Vec::new(); 4],
                y: vec![1.0, 2.0, 3.0, 4.0],
                features: 0,
            },
            horizon: 4,
            seasonality: 7,
            frequency: "D".to_string(),
            known_future_covariates: Vec::new(),
            mase_denominator: denominator,
        };
        let reduction = make_direct_reduction(&data).unwrap();
        assert_eq!(reduction.x.len(), DIRECT_REDUCTION_MAX_ROWS);
        // newest-first budget: only the most recent 2048 origins (8192/4)
        // survive, so every retained row has a fully observed seasonal lag.
        let names = &reduction.feature_names;
        let seasonal_lag_2 = feature_index(names, "seasonal_lag_2");
        assert!(reduction.x.iter().all(|row| row[seasonal_lag_2].is_finite()));
    }

    #[test]
    fn supervised_reduction_uses_only_prior_targets() {
        let train = TimeSeriesTable {
            x: vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0], vec![0.0]],
            y: vec![10.0, 12.0, 12.0, 14.0, 14.0],
            features: 1,
        };
        let test = TimeSeriesTable {
            x: vec![vec![1.0], vec![0.0], vec![1.0]],
            y: vec![16.0, 18.0, 20.0],
            features: 1,
        };
        let data = validate_timeseries_data(train, test, &meta(), &spec()).unwrap();
        let reduction = make_supervised_reduction(&data).unwrap();
        assert_eq!(reduction.x.len(), 4);
        assert_eq!(reduction.y, vec![12.0, 12.0, 14.0, 14.0]);
        let lag_1 = reduction
            .feature_names
            .iter()
            .position(|name| name == "lag_1")
            .unwrap();
        assert_eq!(reduction.x[0][lag_1], 10.0);
        assert_eq!(reduction.x[1][lag_1], 12.0);
        let forecast = make_forecast_features(&data, &data.train.y, 0).unwrap();
        assert_eq!(forecast[lag_1], 14.0);
    }
}
