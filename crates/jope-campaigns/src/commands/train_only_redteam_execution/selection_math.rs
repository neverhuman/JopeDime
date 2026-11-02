use super::*;

fn load_ranked_evaluation(row: &RankedCandidate) -> Result<CandidateEvaluation> {
    let (evaluation, digest): (CandidateEvaluation, _) =
        read_receipt(&relocated_input_path(&row.evaluation_path))?;
    if digest != row.evaluation_sha256
        || evaluation.candidate != row.candidate
        || evaluation.score.robust_loss != row.robust_loss
        || evaluation.score.worst_fold_nrmse != row.worst_fold_nrmse
        || !evaluation.development_only
        || evaluation.confirmatory_valid
    {
        bail!("ranked candidate evaluation link drift");
    }
    Ok(evaluation)
}

fn one_se_candidate(row: &RankedCandidate) -> Result<OneSeCandidate> {
    let evaluation = load_ranked_evaluation(row)?;
    Ok(OneSeCandidate {
        semantic_sha256: row.candidate.semantic_sha256.clone(),
        complexity: row.candidate.complexity,
        robust_loss: row.robust_loss,
        repeat_nrmse: evaluation.score.repeat_nrmse,
        worst_fold_nrmse: row.worst_fold_nrmse,
    })
}

pub(super) fn select_one_se(
    ranked: &[RankedCandidate],
    keep: impl Fn(&Candidate) -> bool,
) -> Result<Candidate> {
    let eligible = ranked
        .iter()
        .filter(|row| keep(&row.candidate))
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        bail!("one-SE counterfactual has no eligible candidates");
    }
    let summaries = eligible
        .iter()
        .map(|row| one_se_candidate(row))
        .collect::<Result<Vec<_>>>()?;
    Ok(eligible[one_standard_error_select(&summaries)?]
        .candidate
        .clone())
}

pub(super) fn selector_evaluations(
    ranked: &[RankedCandidate],
) -> Result<HashMap<String, CandidateEvaluation>> {
    ranked
        .iter()
        .map(|row| {
            Ok((
                row.candidate.semantic_sha256.clone(),
                load_ranked_evaluation(row)?,
            ))
        })
        .collect()
}

fn residual_correlation(
    left: &CandidateEvaluation,
    right: &CandidateEvaluation,
    targets: &[f64],
) -> Result<f64> {
    if left.repeat_predictions.len() != right.repeat_predictions.len()
        || left.repeat_predictions.is_empty()
    {
        bail!("residual correlation repeat mismatch");
    }
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (left_repeat, right_repeat) in left
        .repeat_predictions
        .iter()
        .zip(&right.repeat_predictions)
    {
        if left_repeat.len() != targets.len() || right_repeat.len() != targets.len() {
            bail!("residual correlation row mismatch");
        }
        x.extend(
            targets
                .iter()
                .zip(left_repeat)
                .map(|(truth, prediction)| truth - prediction),
        );
        y.extend(
            targets
                .iter()
                .zip(right_repeat)
                .map(|(truth, prediction)| truth - prediction),
        );
    }
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let covariance = x
        .iter()
        .zip(&y)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let scale_x = x.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    let scale_y = y.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    if scale_x <= 0.0 || scale_y <= 0.0 {
        return Ok(1.0);
    }
    Ok((covariance / (scale_x * scale_y).sqrt()).clamp(-1.0, 1.0))
}

fn weighted_prediction(
    evaluations: &HashMap<String, CandidateEvaluation>,
    candidates: &[Candidate],
    weights: &[f64],
    repeat: usize,
) -> Result<Vec<f64>> {
    if candidates.is_empty()
        || candidates.len() != weights.len()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
    {
        bail!("ensemble weights violate the convex contract");
    }
    let rows = evaluations
        .get(&candidates[0].semantic_sha256)
        .and_then(|evaluation| evaluation.repeat_predictions.get(repeat))
        .context("ensemble repeat is missing")?
        .len();
    let mut predictions = vec![0.0; rows];
    for (candidate, weight) in candidates.iter().zip(weights) {
        let values = evaluations
            .get(&candidate.semantic_sha256)
            .and_then(|evaluation| evaluation.repeat_predictions.get(repeat))
            .context("ensemble base evaluation is missing")?;
        if values.len() != rows {
            bail!("ensemble prediction row mismatch");
        }
        for (prediction, value) in predictions.iter_mut().zip(values) {
            *prediction += weight * value;
        }
    }
    Ok(predictions)
}

fn prediction_nrmse(targets: &[f64], predictions: &[f64], target_sd: f64) -> Result<f64> {
    Ok((squared_error(targets, predictions)? / targets.len() as f64).sqrt() / target_sd)
}

fn optimal_convex_alpha(targets: &[f64], current: &[f64], candidate: &[f64]) -> Result<f64> {
    if targets.len() != current.len() || targets.len() != candidate.len() || targets.is_empty() {
        bail!("convex mixing inputs have different shapes");
    }
    let numerator = targets
        .iter()
        .zip(current)
        .zip(candidate)
        .map(|((truth, current), candidate)| (truth - current) * (candidate - current))
        .sum::<f64>();
    let denominator = current
        .iter()
        .zip(candidate)
        .map(|(current, candidate)| (candidate - current).powi(2))
        .sum::<f64>();
    if denominator <= 0.0 {
        return Ok(0.0);
    }
    Ok((numerator / denominator).clamp(0.0, 1.0))
}

fn add_prediction(current: &[f64], candidate: &[f64], alpha: f64) -> Vec<f64> {
    current
        .iter()
        .zip(candidate)
        .map(|(current, candidate)| (1.0 - alpha) * current + alpha * candidate)
        .collect()
}

pub(super) fn ensemble_nrmse(
    evaluations: &HashMap<String, CandidateEvaluation>,
    candidates: &[Candidate],
    weights: &[f64],
    targets: &[f64],
) -> Result<f64> {
    let sd = target_sd(targets)?;
    let repeats = evaluations
        .get(&candidates[0].semantic_sha256)
        .context("ensemble evaluation is absent")?
        .repeat_predictions
        .len();
    Ok((0..repeats)
        .map(|repeat| {
            prediction_nrmse(
                targets,
                &weighted_prediction(evaluations, candidates, weights, repeat)?,
                sd,
            )
        })
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum::<f64>()
        / repeats as f64)
}

pub(super) fn build_ensemble(
    manifest: &TrainOnlyRedteamManifest,
    ranked: &[RankedCandidate],
    selected_single: &Candidate,
    evaluations: &HashMap<String, CandidateEvaluation>,
    targets: &[f64],
) -> Result<(Vec<Candidate>, Vec<f64>)> {
    let selected_evaluation = evaluations
        .get(&selected_single.semantic_sha256)
        .context("selected single evaluation is absent")?;
    if selected_evaluation.repeat_predictions.len() != 2 {
        bail!("selector ensemble requires exactly two repeats");
    }
    let mut leaders = vec![selected_single.clone()];
    let mut head_counts = HashMap::from([(selected_single.head_family.clone(), 1usize)]);
    for row in ranked {
        if leaders.len() == manifest.config.selector.ensemble_leaders {
            break;
        }
        if row.candidate.semantic_sha256 == selected_single.semantic_sha256
            || *head_counts.get(&row.candidate.head_family).unwrap_or(&0)
                >= manifest.config.selector.ensemble_head_family_cap
        {
            continue;
        }
        let evaluation = &evaluations[&row.candidate.semantic_sha256];
        if leaders.iter().any(|leader| {
            residual_correlation(evaluation, &evaluations[&leader.semantic_sha256], targets)
                .is_ok_and(|correlation| {
                    correlation > manifest.config.selector.residual_correlation_cap
                })
        }) {
            continue;
        }
        *head_counts
            .entry(row.candidate.head_family.clone())
            .or_default() += 1;
        leaders.push(row.candidate.clone());
    }

    let sd = target_sd(targets)?;
    let mut chosen = vec![selected_single.clone()];
    let mut weights = vec![1.0];
    while chosen.len() < manifest.config.selector.ensemble_max_models {
        let current_a = weighted_prediction(evaluations, &chosen, &weights, 0)?;
        let current_b = weighted_prediction(evaluations, &chosen, &weights, 1)?;
        let current_crossfit = (prediction_nrmse(targets, &current_a, sd)?
            + prediction_nrmse(targets, &current_b, sd)?)
            / 2.0;
        let mut best: Option<(f64, f64, Candidate)> = None;
        for candidate in &leaders {
            if chosen
                .iter()
                .any(|present| present.semantic_sha256 == candidate.semantic_sha256)
            {
                continue;
            }
            let evaluation = &evaluations[&candidate.semantic_sha256];
            let candidate_a = &evaluation.repeat_predictions[0];
            let candidate_b = &evaluation.repeat_predictions[1];
            let alpha_a = optimal_convex_alpha(targets, &current_a, candidate_a)?;
            let alpha_b = optimal_convex_alpha(targets, &current_b, candidate_b)?;
            let crossfit_loss = (prediction_nrmse(
                targets,
                &add_prediction(&current_b, candidate_b, alpha_a),
                sd,
            )? + prediction_nrmse(
                targets,
                &add_prediction(&current_a, candidate_a, alpha_b),
                sd,
            )?) / 2.0;
            let alpha = (alpha_a + alpha_b) / 2.0;
            if alpha > 0.0
                && best.as_ref().is_none_or(|(loss, _, best_candidate)| {
                    crossfit_loss < *loss
                        || (crossfit_loss == *loss
                            && candidate.semantic_sha256 < best_candidate.semantic_sha256)
                })
            {
                best = Some((crossfit_loss, alpha, candidate.clone()));
            }
        }
        let Some((loss, alpha, candidate)) = best else {
            break;
        };
        let improvement = (current_crossfit - loss) / current_crossfit.max(1e-15);
        if improvement
            < manifest
                .config
                .selector
                .crossfit_minimum_relative_improvement
        {
            break;
        }
        for weight in &mut weights {
            *weight *= 1.0 - alpha;
        }
        weights.push(alpha);
        chosen.push(candidate);
        let sum = weights.iter().sum::<f64>();
        for weight in &mut weights {
            *weight /= sum;
        }
    }
    Ok((chosen, weights))
}
