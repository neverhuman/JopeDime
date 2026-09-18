use super::*;

pub(super) fn detect_model_kind(weights: &HashMap<String, Tensor>) -> Result<StarforgeModelKind> {
    let has_classifier = weights.keys().any(|k| k.starts_with("many_class_decoder."));
    let has_regressor = weights.keys().any(|k| k.starts_with("output_projection."));
    match (has_classifier, has_regressor) {
        (true, false) => Ok(StarforgeModelKind::Classifier),
        (false, true) => Ok(StarforgeModelKind::Regressor),
        (true, true) => bail!("Starforge weights contain both classifier and regressor heads"),
        (false, false) => bail!("Starforge weights contain no recognized output head"),
    }
}

pub(super) fn target_mean_std(y: &[f32]) -> (f32, f32) {
    let n = y.len().max(1) as f32;
    let mean = y.iter().sum::<f32>() / n;
    let var = y
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    (mean, var.sqrt() + 1e-8)
}

pub(super) fn regression_query_chunk(n_context: usize, n_query: usize) -> usize {
    let room = REGRESSION_MAX_TOTAL_ROWS as isize - n_context as isize;
    let capped = if room > 0 {
        n_query.min(room as usize)
    } else {
        0
    };
    capped.max(8).min(n_query.max(1))
}

/// Per-row median of the 6 feature-view predictions. Each output row depends only on that row's own
/// per-view values, so a query scores identically alone or inside any batch (row-stable inference);
/// shared by the classifier and regression ensemble heads.
pub(super) fn median_of_views(per_view: &[Vec<f64>], n_query: usize) -> Vec<f64> {
    debug_assert_eq!(
        per_view.len(),
        6,
        "ensemble heads run exactly 6 feature views"
    );
    let mut out = vec![0.0; n_query];
    for (row, slot) in out.iter_mut().enumerate() {
        let mut values: Vec<f64> = per_view.iter().map(|pred| pred[row]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        *slot = 0.5 * (values[2] + values[3]);
    }
    out
}

// Retained for reference/tests: the classifier now aggregates feature views per row (median) for
// row-stable inference, so this batch-relative rank aggregation is no longer on the prediction path.
#[allow(dead_code)]
pub(super) fn rank_aggregate(predictions: &[Vec<f64>], n_query: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_query];
    if predictions.is_empty() || n_query == 0 {
        return out;
    }
    for pred in predictions {
        debug_assert_eq!(pred.len(), n_query);
        let ranks = average_ranks_one_based(pred);
        for (dst, rank) in out.iter_mut().zip(ranks) {
            *dst += rank / n_query as f64;
        }
    }
    let denom = predictions.len() as f64;
    for v in &mut out {
        *v /= denom;
    }
    out
}

pub(super) fn average_ranks_one_based(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0usize;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && values[order[j]] == values[order[i]] {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 * 0.5;
        for &idx in &order[i..j] {
            ranks[idx] = avg_rank;
        }
        i = j;
    }
    ranks
}
