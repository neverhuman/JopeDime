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
