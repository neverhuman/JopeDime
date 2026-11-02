use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct CvScore {
    pub(crate) repeat_nrmse: Vec<f64>,
    pub(crate) fold_nrmse: Vec<Vec<f64>>,
    pub(crate) robust_loss: f64,
    pub(crate) worst_fold_nrmse: f64,
}

pub(crate) fn robust_cv_score(
    repeat_squared_error: &[f64],
    repeat_rows: &[usize],
    fold_nrmse: Vec<Vec<f64>>,
    target_sd: f64,
) -> Result<CvScore> {
    if repeat_squared_error.is_empty()
        || repeat_squared_error.len() != repeat_rows.len()
        || !target_sd.is_finite()
        || target_sd <= 0.0
    {
        bail!("robust CV score inputs are invalid");
    }
    let repeat_nrmse = repeat_squared_error
        .iter()
        .zip(repeat_rows)
        .map(|(squared_error, rows)| {
            if !squared_error.is_finite() || *squared_error < 0.0 || *rows == 0 {
                bail!("robust CV repeat is invalid");
            }
            Ok((squared_error / *rows as f64).sqrt() / target_sd)
        })
        .collect::<Result<Vec<_>>>()?;
    let mean = repeat_nrmse.iter().sum::<f64>() / repeat_nrmse.len() as f64;
    let population_sd = (repeat_nrmse
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / repeat_nrmse.len() as f64)
        .sqrt();
    let worst_fold_nrmse = fold_nrmse
        .iter()
        .flatten()
        .copied()
        .max_by(f64::total_cmp)
        .context("robust CV needs fold metrics")?;
    Ok(CvScore {
        repeat_nrmse,
        fold_nrmse,
        robust_loss: mean + 0.25 * population_sd,
        worst_fold_nrmse,
    })
}

pub(crate) fn target_sd(targets: &[f64]) -> Result<f64> {
    if targets.is_empty() || targets.iter().any(|value| !value.is_finite()) {
        bail!("target SD needs finite values");
    }
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let sd = (targets
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / targets.len() as f64)
        .sqrt();
    if sd <= 0.0 {
        bail!("target SD is zero");
    }
    Ok(sd)
}

#[derive(Clone, Debug)]
pub(crate) struct OneSeCandidate {
    pub(crate) semantic_sha256: String,
    pub(crate) complexity: usize,
    pub(crate) robust_loss: f64,
    pub(crate) repeat_nrmse: Vec<f64>,
    pub(crate) worst_fold_nrmse: f64,
}

pub(crate) fn one_standard_error_select(candidates: &[OneSeCandidate]) -> Result<usize> {
    let (best_index, best) = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.robust_loss.is_finite()
                && !candidate.repeat_nrmse.is_empty()
                && candidate.repeat_nrmse.iter().all(|value| value.is_finite())
        })
        .min_by(|(_, left), (_, right)| {
            left.robust_loss
                .total_cmp(&right.robust_loss)
                .then_with(|| left.worst_fold_nrmse.total_cmp(&right.worst_fold_nrmse))
                .then_with(|| left.complexity.cmp(&right.complexity))
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        })
        .context("one-SE selection has no valid candidates")?;
    let mean = best.repeat_nrmse.iter().sum::<f64>() / best.repeat_nrmse.len() as f64;
    let sample_sd = if best.repeat_nrmse.len() > 1 {
        (best
            .repeat_nrmse
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (best.repeat_nrmse.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    let threshold = best.robust_loss + sample_sd / (best.repeat_nrmse.len() as f64).sqrt();
    Ok(candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.robust_loss.is_finite() && candidate.robust_loss <= threshold
        })
        .min_by(|(_, left), (_, right)| {
            left.complexity
                .cmp(&right.complexity)
                .then_with(|| left.robust_loss.total_cmp(&right.robust_loss))
                .then_with(|| left.worst_fold_nrmse.total_cmp(&right.worst_fold_nrmse))
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        })
        .map(|(index, _)| index)
        .unwrap_or(best_index))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct EloRating {
    pub(crate) method_id: String,
    pub(crate) rating: f64,
}

pub(crate) fn bradley_terry_elo(
    scores: &BTreeMap<String, BTreeMap<String, f64>>,
    config: &ScoringConfig,
) -> Result<Vec<EloRating>> {
    let methods = scores
        .values()
        .flat_map(|dataset| dataset.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if methods.is_empty() {
        return Ok(Vec::new());
    }
    let method_index = methods
        .iter()
        .enumerate()
        .map(|(index, method)| (method.clone(), index))
        .collect::<HashMap<_, _>>();
    let mut matches = Vec::new();
    for dataset in scores.values() {
        if dataset.len() != methods.len() {
            bail!("Elo datasets must contain the same field");
        }
        let entries = dataset.iter().collect::<Vec<_>>();
        for left in 0..entries.len() {
            for right in (left + 1)..entries.len() {
                let (method_a, score_a) = entries[left];
                let (method_b, score_b) = entries[right];
                if !score_a.is_finite() || !score_b.is_finite() {
                    bail!("Elo score is non-finite");
                }
                let scale = score_a.abs().max(score_b.abs()).max(1.0);
                let outcome = if (score_a - score_b).abs() <= config.tie_relative_tolerance * scale
                {
                    0.5
                } else if score_a < score_b {
                    1.0
                } else {
                    0.0
                };
                matches.push((method_index[method_a], method_index[method_b], outcome));
            }
        }
    }
    let mut ratings = vec![config.elo_center; methods.len()];
    let elo_scale = 10_f64.ln() / 400.0;
    for _ in 0..config.bradley_terry_iterations {
        let mut gradient = vec![0.0; methods.len()];
        let mut information = vec![0.0; methods.len()];
        for &(left, right, outcome) in &matches {
            let delta = (elo_scale * (ratings[left] - ratings[right])).clamp(-700.0, 700.0);
            let probability = 1.0 / (1.0 + (-delta).exp());
            let residual = outcome - probability;
            let curvature = probability * (1.0 - probability);
            gradient[left] += elo_scale * residual;
            gradient[right] -= elo_scale * residual;
            let info = elo_scale * elo_scale * curvature;
            information[left] += info;
            information[right] += info;
        }
        let mut largest_update = 0.0_f64;
        for index in 0..ratings.len() {
            let regularized_gradient = gradient[index]
                - config.bradley_terry_regularization * (ratings[index] - config.elo_center);
            let update = (regularized_gradient
                / (information[index] + config.bradley_terry_regularization))
                .clamp(-40.0, 40.0);
            largest_update = largest_update.max(update.abs());
            ratings[index] += update;
        }
        if largest_update < config.bradley_terry_tolerance {
            break;
        }
    }
    let mean = ratings.iter().sum::<f64>() / ratings.len() as f64;
    for rating in &mut ratings {
        *rating += config.elo_center - mean;
    }
    let mut result = methods
        .into_iter()
        .zip(ratings)
        .map(|(method_id, rating)| EloRating { method_id, rating })
        .collect::<Vec<_>>();
    result.sort_by(|left, right| {
        right
            .rating
            .total_cmp(&left.rating)
            .then_with(|| left.method_id.cmp(&right.method_id))
    });
    Ok(result)
}
