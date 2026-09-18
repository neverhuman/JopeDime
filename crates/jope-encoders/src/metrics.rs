use ordered_float::OrderedFloat;

pub fn binary_auc(labels: &[u8], scores: &[f64]) -> f64 {
    let n_pos = labels.iter().filter(|&&v| v == 1).count();
    let n_neg = labels.iter().filter(|&&v| v == 0).count();
    if n_pos == 0 || n_neg == 0 || labels.len() != scores.len() {
        return f64::NAN;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by_key(|&i| OrderedFloat(scores[i]));
    let mut ranks = vec![0f64; scores.len()];
    let mut i = 0usize;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && scores[order[j]] == scores[order[i]] {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for &idx in &order[i..j] {
            ranks[idx] = avg_rank;
        }
        i = j;
    }
    let rank_sum_pos = labels
        .iter()
        .zip(ranks.iter())
        .filter_map(|(&y, &r)| if y == 1 { Some(r) } else { None })
        .sum::<f64>();
    (rank_sum_pos - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0) / (n_pos as f64 * n_neg as f64)
}

pub fn r2_score(y: &[f32], pred: &[f64]) -> f64 {
    if y.len() != pred.len() || y.is_empty() {
        return f64::NAN;
    }
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / y.len() as f64;
    let ss_tot = y
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>();
    if ss_tot <= 1e-24 {
        return f64::NAN;
    }
    let ss_res = y
        .iter()
        .zip(pred)
        .map(|(&a, &b)| {
            let d = a as f64 - b;
            d * d
        })
        .sum::<f64>();
    1.0 - ss_res / ss_tot
}

pub fn rmse(y: &[f32], pred: &[f64]) -> f64 {
    if y.len() != pred.len() || y.is_empty() {
        return f64::NAN;
    }
    let mse = y
        .iter()
        .zip(pred)
        .map(|(&a, &b)| {
            let d = a as f64 - b;
            d * d
        })
        .sum::<f64>()
        / y.len() as f64;
    mse.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_auc_handles_ordering_and_ties() {
        assert_eq!(binary_auc(&[0, 1, 0, 1], &[0.1, 0.9, 0.2, 0.8]), 1.0);
        assert_eq!(binary_auc(&[0, 1, 0, 1], &[0.9, 0.1, 0.8, 0.2]), 0.0);
        assert_eq!(binary_auc(&[0, 1, 0, 1], &[0.5, 0.5, 0.5, 0.5]), 0.5);
    }

    #[test]
    fn binary_auc_returns_nan_for_invalid_inputs() {
        assert!(binary_auc(&[1, 1], &[0.1, 0.2]).is_nan());
        assert!(binary_auc(&[0, 1], &[0.1]).is_nan());
    }

    #[test]
    fn regression_metrics_handle_basic_cases() {
        let y = [1.0, 2.0, 3.0];
        assert_eq!(r2_score(&y, &[1.0, 2.0, 3.0]), 1.0);
        assert_eq!(rmse(&y, &[1.0, 2.0, 3.0]), 0.0);
        assert!(r2_score(&[1.0, 1.0], &[1.0, 1.0]).is_nan());
    }
}
