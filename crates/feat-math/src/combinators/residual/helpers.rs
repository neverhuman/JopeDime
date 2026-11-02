use super::*;

pub(crate) fn generated_plan(
    row_ids: &[u64],
    fold_count: usize,
    seed: u64,
) -> Result<OOFFitPlan, FitError> {
    if fold_count < 2 || row_ids.len() < fold_count {
        return Err(FitError::InvalidPlan(
            "not enough rows for the inner OOF plan".into(),
        ));
    }
    let mut order: Vec<_> = (0..row_ids.len()).collect();
    order.sort_by_key(|&i| mix64(row_ids[i] ^ seed));
    let mut assignments = vec![0; row_ids.len()];
    for (rank, index) in order.into_iter().enumerate() {
        assignments[index] = rank % fold_count;
    }
    OOFFitPlan::new(assignments, fold_count, seed)
}

pub(crate) fn full_context(candidate_id: &str, seed: u64, row_ids: &[u64]) -> FitContext {
    FitContext {
        abi: COMBINATOR_ABI,
        candidate_id: candidate_id.into(),
        role: FitRole::FullTraining,
        seed,
        train_row_ids: row_ids.to_vec(),
    }
}

pub(crate) fn checked_prediction<M: FittedCandidate>(
    model: &M,
    rows: &[Vec<f64>],
) -> Result<Vec<f64>, FitError> {
    if model.prediction_target() != "original" {
        return Err(FitError::Prediction(
            "candidate output is not in the original target space".into(),
        ));
    }
    let values = model.predict(rows)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(FitError::Prediction(
            "candidate produced nonfinite values".into(),
        ));
    }
    Ok(values)
}

pub(crate) fn emitted_prediction<M: FittedCandidate>(
    model: &M,
    rows: &[Vec<f64>],
) -> Result<Vec<f64>, FitError> {
    if model.prediction_target() != "original" {
        return Err(FitError::Prediction(
            "emitted candidate output is not in original target space".into(),
        ));
    }
    let values = model.predict_emitted(rows)?.ok_or_else(|| {
        FitError::UnsupportedEmission("candidate has no portable fitted emitter".into())
    })?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(FitError::Prediction(
            "emitted candidate produced nonfinite values".into(),
        ));
    }
    Ok(values)
}

pub(crate) fn combine_predictions<M: FittedCandidate>(
    members: &[M],
    weights: &[f64],
    rows: &[Vec<f64>],
    emitted: bool,
) -> Result<Vec<f64>, FitError> {
    if members.is_empty()
        || members.len() != weights.len()
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
    {
        return Err(FitError::InvalidPlan(
            "member and weight shapes differ or weights are invalid".into(),
        ));
    }
    if (weights.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
        return Err(FitError::InvalidPlan(
            "stack weights must sum to one".into(),
        ));
    }
    let mut output = vec![0.0; rows.len()];
    for (member, &weight) in members.iter().zip(weights) {
        let values = if emitted {
            emitted_prediction(member, rows)?
        } else {
            checked_prediction(member, rows)?
        };
        if values.len() != rows.len() {
            return Err(FitError::Prediction(
                "member prediction width differs from input row count".into(),
            ));
        }
        for (sum, value) in output.iter_mut().zip(values) {
            *sum += weight * value;
        }
    }
    Ok(output)
}

pub(crate) fn fit_simplex_weights(
    predictions: &[Vec<f64>],
    target: &[f64],
    iterations: usize,
) -> Vec<f64> {
    let count = predictions.len();
    let mut weights = vec![1.0 / count as f64; count];
    let scale =
        predictions.iter().flatten().map(|x| x * x).sum::<f64>() / target.len().max(1) as f64;
    let step = 0.5 / (scale * count as f64 + 1e-12);
    for _ in 0..iterations {
        let blended: Vec<_> = (0..target.len())
            .map(|i| {
                predictions
                    .iter()
                    .zip(&weights)
                    .map(|(p, w)| p[i] * w)
                    .sum::<f64>()
            })
            .collect();
        let gradient: Vec<_> = (0..count)
            .map(|m| {
                2.0 * (0..target.len())
                    .map(|i| predictions[m][i] * (blended[i] - target[i]))
                    .sum::<f64>()
                    / target.len() as f64
            })
            .collect();
        weights = project_simplex(
            weights
                .iter()
                .zip(gradient)
                .map(|(w, g)| w - step * g)
                .collect(),
        );
    }
    weights
}

pub(crate) fn project_simplex(mut values: Vec<f64>) -> Vec<f64> {
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut cumulative = 0.0;
    let mut theta = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        cumulative += value;
        let candidate = (cumulative - 1.0) / (index + 1) as f64;
        if index + 1 == sorted.len() || sorted[index + 1] <= candidate {
            theta = candidate;
            break;
        }
    }
    for value in &mut values {
        *value = (*value - theta).max(0.0);
    }
    values
}

pub(crate) fn valid_shrinkage(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}
pub(crate) fn select_rows(rows: &[Vec<f64>], indices: &[usize]) -> Rows {
    indices.iter().map(|&i| rows[i].clone()).collect()
}
pub(crate) fn select_values<T: Copy>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| values[i]).collect()
}
pub(crate) fn fold_seed(seed: u64, part: u64) -> u64 {
    mix64(seed ^ part.wrapping_mul(0x9e3779b97f4a7c15))
}
pub(crate) fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}
