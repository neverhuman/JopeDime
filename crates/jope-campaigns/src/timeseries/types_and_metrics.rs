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
