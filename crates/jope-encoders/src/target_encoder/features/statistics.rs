pub(super) fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() > 1e-12 {
        numerator / denominator
    } else {
        0.0
    }
}

pub(super) fn mean_f64(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

pub(super) fn quantile(sorted: &[f64], position: f64) -> f64 {
    let index = (sorted.len() - 1) as f64 * position;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * index.fract()
}

pub(super) fn hard_histogram(values: &[f64], bins: usize) -> Vec<f64> {
    let mut counts = vec![0usize; bins];
    for &value in values {
        let index = if value >= 1.0 {
            bins - 1
        } else {
            (value * bins as f64).floor().max(0.0) as usize
        };
        counts[index.min(bins - 1)] += 1;
    }
    counts
        .into_iter()
        .map(|count| count as f64 / values.len().max(1) as f64)
        .collect()
}

pub(super) fn entropy(probabilities: &[f64]) -> f64 {
    -probabilities
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .map(|value| value * value.ln())
        .sum::<f64>()
        / (probabilities.len().max(2) as f64).ln()
}

pub(super) fn legendre(degree: usize, value: f64) -> f64 {
    if degree == 0 {
        return 1.0;
    }
    let mut previous = 1.0;
    let mut current = value;
    for order in 2..=degree {
        let next = ((2 * order - 1) as f64 * value * current - (order - 1) as f64 * previous)
            / order as f64;
        previous = current;
        current = next;
    }
    current
}
