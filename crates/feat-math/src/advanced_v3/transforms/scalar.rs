use super::*;

pub(super) fn yeo_johnson(value: f64, lambda: f64) -> f64 {
    if value >= 0.0 {
        if lambda.abs() < 1e-12 {
            value.ln_1p()
        } else {
            ((value + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < 1e-12 {
        -(-value).ln_1p()
    } else {
        -((1.0 - value).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
    }
}

pub(super) fn fit_yeo_lambda(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let data: Vec<f64> = values.collect();
    let n = data.len().max(1) as f64;
    let mut best = 1.0;
    let mut best_score = f64::NEG_INFINITY;
    for step in -4..=4 {
        let lambda = step as f64 * 0.5;
        let transformed: Vec<f64> = data.iter().map(|v| yeo_johnson(*v, lambda)).collect();
        let mean = transformed.iter().sum::<f64>() / n;
        let variance = transformed
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / n;
        let jacobian: f64 = data
            .iter()
            .map(|v| {
                if *v >= 0.0 {
                    (lambda - 1.0) * v.ln_1p()
                } else {
                    (1.0 - lambda) * (-v).ln_1p()
                }
            })
            .sum();
        let score = -0.5 * n * variance.max(1e-24).ln() + jacobian;
        if score > best_score {
            best_score = score;
            best = lambda;
        }
    }
    best
}

pub(super) fn piecewise_row(row: &[f64], edges: &[Vec<f64>]) -> Vec<f64> {
    let mut out = Vec::new();
    for (value, column_edges) in row.iter().zip(edges) {
        for interval in column_edges.windows(2) {
            let width = interval[1] - interval[0];
            let encoded = if width.abs() <= 1e-15 {
                f64::from(*value > interval[0])
            } else {
                ((*value - interval[0]) / width).clamp(0.0, 1.0)
            };
            out.push(encoded);
        }
    }
    out
}

pub(super) fn fit_pls_weights(x: &Rows, y: &[f64], k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = x.len();
    let cols = x.first().map_or(0, Vec::len);
    let mut means = vec![0.0; cols];
    for row in x {
        for (j, value) in row.iter().enumerate() {
            means[j] += value / n.max(1) as f64;
        }
    }
    let y_mean = y.iter().take(n).sum::<f64>() / n.max(1) as f64;
    let mut residual_x: Rows = x
        .iter()
        .map(|row| row.iter().enumerate().map(|(j, v)| v - means[j]).collect())
        .collect();
    let mut residual_y: Vec<f64> = y.iter().take(n).map(|v| v - y_mean).collect();
    let mut weights = Vec::new();
    for _ in 0..k.min(cols).min(n.saturating_sub(1)) {
        let mut weight = vec![0.0; cols];
        for (row, target) in residual_x.iter().zip(&residual_y) {
            for j in 0..cols {
                weight[j] += row[j] * target;
            }
        }
        let norm = weight.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= 1e-15 {
            break;
        }
        for value in &mut weight {
            *value /= norm;
        }
        let scores: Vec<f64> = residual_x
            .iter()
            .map(|row| row.iter().zip(&weight).map(|(a, b)| a * b).sum())
            .collect();
        let denom = scores.iter().map(|v| v * v).sum::<f64>().max(1e-24);
        let y_loading = scores
            .iter()
            .zip(&residual_y)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / denom;
        let mut loadings = vec![0.0; cols];
        for j in 0..cols {
            loadings[j] = scores
                .iter()
                .zip(&residual_x)
                .map(|(score, row)| score * row[j])
                .sum::<f64>()
                / denom;
        }
        for r in 0..n {
            for j in 0..cols {
                residual_x[r][j] -= scores[r] * loadings[j];
            }
            residual_y[r] -= scores[r] * y_loading;
        }
        weights.push(weight);
    }
    (means, weights)
}
