use super::*;

/// Fold of every row.
pub fn folds(kind: TaskKind, targets: &[f64], folds: usize) -> Vec<usize> {
    let mut out = vec![0; targets.len()];
    match kind {
        TaskKind::Binary | TaskKind::Multiclass => {
            let mut rank = std::collections::BTreeMap::<u64, usize>::new();
            for (row, value) in targets.iter().enumerate() {
                let r = rank.entry(value.to_bits()).or_default();
                out[row] = *r % folds;
                *r += 1;
            }
        }
        _ => {
            for (row, fold) in out.iter_mut().enumerate() {
                *fold = row % folds;
            }
        }
    }
    out
}

/// The loss name for a task.
pub fn loss_name(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Regression => "rmse",
        TaskKind::Binary => "1-auroc",
        _ => "log_loss",
    }
}

/// Loss of predictions against targets (values, or class indices).
pub fn loss(kind: TaskKind, targets: &[f64], predictions: &Predictions) -> Result<f64> {
    let value = match (kind, predictions) {
        (TaskKind::Regression, Predictions::Values(values)) => {
            feat_math::metrics::rmse(targets, values)
        }
        (TaskKind::Binary, Predictions::Probabilities { values, .. }) => {
            // IEEE -0 and +0 are the same probability and must tie in AUROC.
            let positive: Vec<f64> = values
                .iter()
                .map(|row| if row[1] == 0.0 { 0.0 } else { row[1] })
                .collect();
            1.0 - feat_math::metrics::auc(targets, &positive)
        }
        (TaskKind::Multiclass, Predictions::Probabilities { values, .. }) => {
            let sum: f64 = targets
                .iter()
                .zip(values)
                .map(|(target, row)| -row[*target as usize].max(PROBABILITY_EPSILON).ln())
                .sum();
            sum / targets.len() as f64
        }
        _ => anyhow::bail!("predictions do not match the task"),
    };
    anyhow::ensure!(value.is_finite(), "non-finite loss");
    Ok(value)
}
