use super::*;

pub(super) fn load_csv(path: &str) -> Result<(Rows, Vec<f64>), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut x: Rows = Vec::new();
    let mut y = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut row: Vec<f64> = Vec::new();
        for cell in line.split(',') {
            let v: f64 = cell
                .trim()
                .parse()
                .map_err(|e| format!("{path}:{}: {e}", line_no + 1))?;
            row.push(v);
        }
        if row.len() < 2 {
            return Err(format!("{path}:{}: fewer than 2 columns", line_no + 1));
        }
        y.push(row.pop().expect("row has columns"));
        x.push(row);
    }
    if x.is_empty() {
        return Err(format!("{path}: no rows"));
    }
    let width = x[0].len();
    if x.iter().any(|r| r.len() != width) {
        return Err(format!("{path}: ragged rows"));
    }
    Ok((x, y))
}

pub(super) fn r2(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    if predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();
    let ss_res: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    if ss_tot <= 0.0 {
        return None;
    }
    Some(1.0 - ss_res / ss_tot)
}

pub(super) fn rmse(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    if predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mse: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    Some(mse.sqrt())
}

/// Copied from train_only_study.rs::paired_robust_fitness (mean - 0.25*sd,
/// scores clipped to >= -1 so one catastrophic split cannot dominate).
pub(super) fn robust_fitness(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return -1.0;
    }
    let clipped: Vec<f64> = scores.iter().map(|v| v.max(-1.0)).collect();
    let mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let sd =
        (clipped.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / clipped.len() as f64).sqrt();
    mean - 0.25 * sd.min(1.0)
}

pub(super) fn worst_split(scores: &[f64]) -> f64 {
    scores
        .iter()
        .map(|v| v.max(-1.0))
        .fold(f64::INFINITY, f64::min)
}

/// rank_scored ordering from train_only_study.rs: fitness desc, worst-split
/// desc, complexity asc, semantic asc (fully deterministic).
pub(super) fn rank(members: &mut [Member]) {
    members.sort_by(|a, b| {
        b.fitness_complete
            .cmp(&a.fitness_complete)
            .then_with(|| robust_fitness(&b.scores).total_cmp(&robust_fitness(&a.scores)))
            .then_with(|| worst_split(&b.scores).total_cmp(&worst_split(&a.scores)))
            .then_with(|| a.complexity.cmp(&b.complexity))
            .then_with(|| a.semantic.cmp(&b.semantic))
    });
}
