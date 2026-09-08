use super::{RouterArtifacts, RouterConfig};

fn adapt_dim<'a>(query: &'a [f64], dim: usize, scratch: &'a mut Vec<f64>) -> &'a [f64] {
    if query.len() == dim {
        return query;
    }
    scratch.clear();
    scratch.resize(dim, 0.0);
    for (dst, src) in scratch.iter_mut().zip(query.iter()) {
        *dst = if src.is_finite() { *src } else { 0.0 };
    }
    scratch
}

fn standardized_train_query<'a>(
    train: &[Vec<f64>],
    query: &'a [f64],
    scratch: &'a mut Vec<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let dim = train.first().map(|r| r.len()).unwrap_or(query.len());
    let q = adapt_dim(query, dim, scratch);
    let mut mu = vec![0.0; dim];
    for row in train {
        for (j, muj) in mu.iter_mut().enumerate() {
            *muj += row.get(j).copied().unwrap_or(0.0);
        }
    }
    for v in &mut mu {
        *v /= train.len().max(1) as f64;
    }
    let mut sd = vec![0.0; dim];
    for row in train {
        for (j, sdj) in sd.iter_mut().enumerate() {
            let d = row.get(j).copied().unwrap_or(0.0) - mu[j];
            *sdj += d * d;
        }
    }
    for v in &mut sd {
        *v = (*v / train.len().max(1) as f64).sqrt();
        if *v <= 1e-6 || !v.is_finite() {
            *v = 1.0;
        }
    }
    let x_train: Vec<Vec<f64>> = train
        .iter()
        .map(|row| {
            (0..dim)
                .map(|j| (row.get(j).copied().unwrap_or(0.0) - mu[j]) / sd[j])
                .collect()
        })
        .collect();
    let x_query = (0..dim).map(|j| (q[j] - mu[j]) / sd[j]).collect();
    (x_train, x_query)
}

pub fn adaptive_retrieval_scores(
    artifacts: &RouterArtifacts,
    query: &[f64],
    cfg: &RouterConfig,
) -> Vec<f64> {
    let mut scratch = Vec::new();
    let (x_train, x_query) =
        standardized_train_query(&artifacts.train_features, query, &mut scratch);
    let mut dist: Vec<(usize, f64)> = x_train
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let d2 = row
                .iter()
                .zip(x_query.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum::<f64>();
            (i, d2.max(0.0).sqrt())
        })
        .collect();
    dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let usable = cfg.k.max(1).min(dist.len().max(1));
    let nearest = &dist[..usable];
    let h = nearest
        .iter()
        .map(|(_, d)| *d)
        .fold(0.0_f64, f64::max)
        .mul_add(cfg.bandwidth_scale, 0.0)
        .max(1e-3);
    let mut weights: Vec<f64> = nearest
        .iter()
        .map(|(_, d)| (-0.5 * (d / h) * (d / h)).exp())
        .collect();
    let wsum = weights.iter().sum::<f64>().max(1e-12);
    for w in &mut weights {
        *w /= wsum;
    }
    let n_alg = artifacts.algorithms.len();
    let mut out = vec![0.0; n_alg];
    for ((row, _), w) in nearest.iter().zip(weights.iter()) {
        for (j, out_j) in out.iter_mut().enumerate() {
            *out_j += *w * artifacts.train_utilities[*row][j];
        }
    }
    out
}

pub fn source_family_mean_predict(
    artifacts: &RouterArtifacts,
    source_family: &str,
    shrink: f64,
) -> Vec<f64> {
    let prior = artifacts.prior();
    let rows: Vec<usize> = artifacts
        .train_source_families
        .iter()
        .enumerate()
        .filter_map(|(i, fam)| (fam == source_family).then_some(i))
        .collect();
    if rows.is_empty() {
        return prior;
    }
    let mut mean = vec![0.0; artifacts.algorithms.len()];
    for &r in &rows {
        for (j, v) in artifacts.train_utilities[r].iter().enumerate() {
            mean[j] += *v;
        }
    }
    for v in &mut mean {
        *v /= rows.len() as f64;
    }
    let weight = rows.len() as f64 / (rows.len() as f64 + shrink);
    mean.into_iter()
        .zip(prior)
        .map(|(m, p)| weight * m + (1.0 - weight) * p)
        .collect()
}

pub fn family_smooth_lifts(lifts: &mut [f64], family_ids: &[usize], weight: f64) {
    if weight <= 0.0 || family_ids.len() != lifts.len() {
        return;
    }
    let max_id = family_ids.iter().copied().max().unwrap_or(0);
    for fid in 0..=max_id {
        let cols: Vec<usize> = family_ids
            .iter()
            .enumerate()
            .filter_map(|(j, &id)| (id == fid).then_some(j))
            .collect();
        if cols.is_empty() {
            continue;
        }
        let mean = cols.iter().map(|&j| lifts[j]).sum::<f64>() / cols.len() as f64;
        for j in cols {
            lifts[j] = (1.0 - weight) * lifts[j] + weight * mean;
        }
    }
}

pub fn starlight_anchor_index(algorithms: &[String], prior: &[f64]) -> usize {
    let mut best_starlight: Option<usize> = None;
    for (i, alg) in algorithms.iter().enumerate() {
        if alg.to_ascii_lowercase().contains("starlight_core")
            && best_starlight.is_none_or(|b| prior[i] > prior[b])
        {
            best_starlight = Some(i);
        }
    }
    best_starlight.unwrap_or_else(|| argmax(prior))
}

pub fn certify_margin(scores: &[f64], anchor: usize, tau: f64) -> bool {
    let top = argmax(scores);
    top == anchor || (scores[top] - scores[anchor]) >= tau
}

pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        let vv = if v.is_finite() { v } else { f64::NEG_INFINITY };
        if vv > best_v {
            best = i;
            best_v = vv;
        }
    }
    best
}

pub fn elo_delta(wins: f64, losses: f64) -> f64 {
    let games = (wins + losses).max(1.0);
    let p = (wins / games).clamp(1e-6, 1.0 - 1e-6);
    400.0 * (p / (1.0 - p)).log10()
}

pub fn tiny_elo(selected: &[usize], winners: &[usize]) -> f64 {
    let mut wins = 0.0;
    let mut losses = 0.0;
    for (&s, &w) in selected.iter().zip(winners.iter()) {
        if s == w {
            wins += 1.0;
        } else {
            losses += 1.0;
        }
    }
    elo_delta(wins, losses)
}
