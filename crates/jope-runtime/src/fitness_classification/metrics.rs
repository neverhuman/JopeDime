use super::*;

impl Metric {
    pub(super) fn direction(self) -> Direction {
        match self {
            Self::RocAuc | Self::PrAuc | Self::F1 | Self::BalancedAccuracy => Direction::Maximize,
            Self::Logloss | Self::Brier | Self::CostMatrixLoss => Direction::Minimize,
        }
    }

    pub(super) fn value(
        self,
        labels: &[usize],
        probabilities: &[[f64; 2]],
        aggregation: Aggregation,
        threshold: f64,
        cost_matrix: Option<[[f64; 2]; 2]>,
    ) -> Result<f64> {
        match self {
            Self::Logloss | Self::Brier => {
                let mut total = 0.0;
                for (&label, p) in labels.iter().zip(probabilities) {
                    total += match self {
                        Self::Logloss => {
                            if p[label] == 0.0 {
                                return Err(FitnessError::Undefined(
                                    "logloss has zero true-class probability (no clipping)".into(),
                                ));
                            }
                            -p[label].ln()
                        }
                        Self::Brier => (p[1] - label as f64).powi(2),
                        _ => unreachable!(),
                    };
                }
                Ok(match aggregation {
                    Aggregation::Mean {} => total / labels.len() as f64,
                    Aggregation::Sum {} => total,
                    _ => {
                        return Err(FitnessError::InvalidSpecification(
                            "classification aggregation supports only mean or sum".into(),
                        ));
                    }
                })
            }
            Self::RocAuc => roc_auc(labels, probabilities),
            Self::PrAuc => average_precision(labels, probabilities),
            Self::F1 => f1_score(labels, probabilities, threshold),
            Self::BalancedAccuracy => balanced_accuracy(labels, probabilities),
            Self::CostMatrixLoss => cost_matrix_loss(
                labels,
                probabilities,
                cost_matrix.expect("validated cost matrix"),
            ),
        }
    }
}

fn sorted_scores(
    labels: &[usize],
    probabilities: &[[f64; 2]],
    descending: bool,
) -> Vec<(f64, usize)> {
    let mut rows: Vec<_> = labels
        .iter()
        .zip(probabilities)
        .map(|(&label, probability)| (probability[1], label))
        .collect();
    rows.sort_by(|left, right| {
        let ordering = left.0.total_cmp(&right.0);
        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    });
    rows
}

fn roc_auc(labels: &[usize], probabilities: &[[f64; 2]]) -> Result<f64> {
    let positives = labels.iter().filter(|&&label| label == 1).count();
    let negatives = labels.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err(FitnessError::Undefined(
            "roc_auc requires both classes".into(),
        ));
    }
    let rows = sorted_scores(labels, probabilities, false);
    let mut negatives_before = 0usize;
    let mut concordant = 0.0;
    let mut index = 0;
    while index < rows.len() {
        let mut end = index + 1;
        while end < rows.len() && rows[end].0 == rows[index].0 {
            end += 1;
        }
        let group_positives = rows[index..end]
            .iter()
            .filter(|(_, label)| *label == 1)
            .count();
        let group_negatives = end - index - group_positives;
        concordant +=
            group_positives as f64 * (negatives_before as f64 + group_negatives as f64 / 2.0);
        negatives_before += group_negatives;
        index = end;
    }
    Ok(concordant / (positives as f64 * negatives as f64))
}

fn average_precision(labels: &[usize], probabilities: &[[f64; 2]]) -> Result<f64> {
    let positives = labels.iter().filter(|&&label| label == 1).count();
    if positives == 0 || positives == labels.len() {
        return Err(FitnessError::Undefined(
            "pr_auc requires both classes".into(),
        ));
    }
    let rows = sorted_scores(labels, probabilities, true);
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut total = 0.0;
    let mut index = 0;
    while index < rows.len() {
        let mut end = index + 1;
        while end < rows.len() && rows[end].0 == rows[index].0 {
            end += 1;
        }
        let new_positives = rows[index..end]
            .iter()
            .filter(|(_, label)| *label == 1)
            .count();
        true_positives += new_positives;
        false_positives += end - index - new_positives;
        total += (new_positives as f64 / positives as f64)
            * (true_positives as f64 / (true_positives + false_positives) as f64);
        index = end;
    }
    Ok(total)
}

fn f1_score(labels: &[usize], probabilities: &[[f64; 2]], threshold: f64) -> Result<f64> {
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut false_negatives = 0usize;
    for (&label, probability) in labels.iter().zip(probabilities) {
        let predicted_positive = probability[1] >= threshold;
        match (label == 1, predicted_positive) {
            (true, true) => true_positives += 1,
            (false, true) => false_positives += 1,
            (true, false) => false_negatives += 1,
            (false, false) => {}
        }
    }
    let denominator = 2 * true_positives + false_positives + false_negatives;
    if denominator == 0 {
        return Err(FitnessError::Undefined(
            "f1 is undefined for this fold".into(),
        ));
    }
    Ok((2 * true_positives) as f64 / denominator as f64)
}

fn balanced_accuracy(labels: &[usize], probabilities: &[[f64; 2]]) -> Result<f64> {
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut true_negatives = 0usize;
    let mut false_negatives = 0usize;
    for (&label, probability) in labels.iter().zip(probabilities) {
        match (
            label == 1,
            probability[1] >= DEFAULT_CLASSIFICATION_THRESHOLD,
        ) {
            (true, true) => true_positives += 1,
            (false, true) => false_positives += 1,
            (false, false) => true_negatives += 1,
            (true, false) => false_negatives += 1,
        }
    }
    if true_positives + false_negatives == 0 || true_negatives + false_positives == 0 {
        return Err(FitnessError::Undefined(
            "balanced_accuracy requires both classes".into(),
        ));
    }
    let sensitivity = true_positives as f64 / (true_positives + false_negatives) as f64;
    let specificity = true_negatives as f64 / (true_negatives + false_positives) as f64;
    Ok((sensitivity + specificity) / 2.0)
}

fn cost_matrix_loss(
    labels: &[usize],
    probabilities: &[[f64; 2]],
    cost_matrix: [[f64; 2]; 2],
) -> Result<f64> {
    let mut total = 0.0;
    for (&actual, probability) in labels.iter().zip(probabilities) {
        let predicted = usize::from(probability[1] >= DEFAULT_CLASSIFICATION_THRESHOLD);
        total += cost_matrix[actual][predicted];
    }
    Ok(total / labels.len() as f64)
}
