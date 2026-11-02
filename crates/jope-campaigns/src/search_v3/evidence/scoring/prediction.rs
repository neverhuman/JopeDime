use super::*;

pub fn score_prediction(
    prediction: &PredictionObjectV1,
    tape: &ScoringTapeV2,
    fitness: &RegisteredFitnessV1,
    complexity: u64,
) -> Result<(ScoreKeyV2, FitnessVectorV1), String> {
    prediction.validate()?;
    tape.validate()?;
    let prediction_sha256 = prediction.identity()?;
    if tape.prediction_object_sha256 != prediction_sha256
        || tape.validation_row_ids != prediction.validation_row_ids
    {
        return Err(
            "scoring tape does not bind the supplied prediction object and row order".into(),
        );
    }
    let fitness_sha256 = fitness.identity()?;
    let key = ScoreKeyV2 {
        schema: ScoreKeyV2::SCHEMA.into(),
        scoring_tape_sha256: tape.identity()?,
        registered_fitness_sha256: fitness_sha256,
    };
    key.validate()?;
    let (task, metric) = match fitness {
        RegisteredFitnessV1::RobustR2 { .. } => (SearchTaskV1::Regression, "robust_r2"),
        RegisteredFitnessV1::BinaryAuc { .. } => (SearchTaskV1::Binary, "auc"),
        RegisteredFitnessV1::MulticlassMacroF1 { .. } => (SearchTaskV1::Multiclass, "macro_f1"),
        RegisteredFitnessV1::NegativeMase { .. } => (SearchTaskV1::Timeseries, "negative_mase"),
    };
    if !prediction.valid {
        let result = invalid_fitness(
            task,
            metric,
            &prediction.candidate_semantic_sha256,
            complexity,
            prediction.invalid_reason.clone().unwrap(),
        );
        result.validate()?;
        return Ok((key, result));
    }

    let predictions = prediction.values();
    let truth = tape.truth();
    let result = match fitness {
        RegisteredFitnessV1::RobustR2 { max_rmse } => {
            if prediction.prediction_shape != vec![truth.len()]
                || tape.truth_shape != vec![truth.len()]
            {
                return Err("regression prediction and truth must both be one-dimensional".into());
            }
            let mean = truth.iter().sum::<f64>() / truth.len() as f64;
            let squared_error = predictions
                .iter()
                .zip(&truth)
                .map(|(prediction, truth)| (prediction - truth).powi(2))
                .sum::<f64>();
            let total = truth
                .iter()
                .map(|truth| (truth - mean).powi(2))
                .sum::<f64>();
            let rmse = (squared_error / truth.len() as f64).sqrt();
            let r2 = if total <= f64::EPSILON {
                if squared_error <= f64::EPSILON {
                    1.0
                } else {
                    0.0
                }
            } else {
                1.0 - squared_error / total
            };
            FitnessVectorV1 {
                schema: FitnessVectorV1::SCHEMA.into(),
                evidence_valid: true,
                invalid_reason: None,
                constraints_satisfied: rmse <= *max_rmse,
                primary: TypedMetricValueV1 {
                    task,
                    metric: metric.into(),
                    value: r2,
                },
                stability_guards: vec![GuardValueV1 {
                    name: "negative_rmse".into(),
                    value: -rmse,
                }],
                complexity,
                candidate_semantic_sha256: prediction.candidate_semantic_sha256.clone(),
            }
        }
        RegisteredFitnessV1::BinaryAuc {
            lift_depths,
            max_log_loss,
        } => {
            if prediction.prediction_shape != vec![truth.len()]
                || tape.truth_shape != vec![truth.len()]
            {
                return Err("binary prediction and truth must both be one-dimensional".into());
            }
            if predictions.iter().any(|value| !(0.0..=1.0).contains(value))
                || truth.iter().any(|value| *value != 0.0 && *value != 1.0)
            {
                return Ok((
                    key,
                    invalid_fitness(
                        task,
                        metric,
                        &prediction.candidate_semantic_sha256,
                        complexity,
                        "binary values are outside their registered domains".into(),
                    ),
                ));
            }
            let positives = truth.iter().filter(|value| **value == 1.0).count();
            let negatives = truth.len() - positives;
            if positives == 0 || negatives == 0 {
                return Ok((
                    key,
                    invalid_fitness(
                        task,
                        metric,
                        &prediction.candidate_semantic_sha256,
                        complexity,
                        "binary validation truth lacks class support".into(),
                    ),
                ));
            }
            let mut pairs: Vec<_> = predictions
                .iter()
                .copied()
                .zip(truth.iter().copied())
                .collect();
            pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
            let mut rank_sum = 0.0;
            let mut start = 0;
            while start < pairs.len() {
                let mut end = start + 1;
                while end < pairs.len() && pairs[end].0.to_bits() == pairs[start].0.to_bits() {
                    end += 1;
                }
                let average_rank = ((start + 1 + end) as f64) / 2.0;
                rank_sum += average_rank
                    * pairs[start..end]
                        .iter()
                        .filter(|(_, truth)| *truth == 1.0)
                        .count() as f64;
                start = end;
            }
            let auc = (rank_sum - (positives * (positives + 1) / 2) as f64)
                / (positives * negatives) as f64;
            let epsilon = 1e-15;
            let log_loss = predictions
                .iter()
                .zip(&truth)
                .map(|(prediction, truth)| {
                    let prediction = prediction.clamp(epsilon, 1.0 - epsilon);
                    -(truth * prediction.ln() + (1.0 - truth) * (1.0 - prediction).ln())
                })
                .sum::<f64>()
                / truth.len() as f64;
            pairs.sort_by(|left, right| right.0.total_cmp(&left.0));
            let prevalence = positives as f64 / truth.len() as f64;
            let mut guards = vec![GuardValueV1 {
                name: "negative_log_loss".into(),
                value: -log_loss,
            }];
            for depth in lift_depths {
                let count = ((*depth * pairs.len() as f64).ceil() as usize)
                    .max(1)
                    .min(pairs.len());
                let rate = pairs[..count]
                    .iter()
                    .filter(|(_, truth)| *truth == 1.0)
                    .count() as f64
                    / count as f64;
                guards.push(GuardValueV1 {
                    name: format!("lift_at_{:016x}", depth.to_bits()),
                    value: rate / prevalence,
                });
            }
            guards.sort_by(|left, right| left.name.cmp(&right.name));
            FitnessVectorV1 {
                schema: FitnessVectorV1::SCHEMA.into(),
                evidence_valid: true,
                invalid_reason: None,
                constraints_satisfied: log_loss <= *max_log_loss,
                primary: TypedMetricValueV1 {
                    task,
                    metric: metric.into(),
                    value: auc,
                },
                stability_guards: guards,
                complexity,
                candidate_semantic_sha256: prediction.candidate_semantic_sha256.clone(),
            }
        }
        RegisteredFitnessV1::MulticlassMacroF1 {
            classes,
            min_class_support,
            max_log_loss,
        } => {
            let classes = *classes as usize;
            if prediction.prediction_shape != vec![truth.len(), classes]
                || tape.truth_shape != vec![truth.len()]
            {
                return Err(
                    "multiclass prediction must be [rows, classes] with scalar class truth".into(),
                );
            }
            let mut support = vec![0u64; classes];
            let mut predicted = Vec::with_capacity(truth.len());
            let mut log_loss = 0.0;
            for (row, truth) in truth.iter().enumerate() {
                let class = *truth as usize;
                if *truth != class as f64 || class >= classes {
                    return Ok((
                        key,
                        invalid_fitness(
                            task,
                            metric,
                            &prediction.candidate_semantic_sha256,
                            complexity,
                            "multiclass truth contains an invalid class".into(),
                        ),
                    ));
                }
                support[class] += 1;
                let probabilities = &predictions[row * classes..(row + 1) * classes];
                let sum = probabilities.iter().sum::<f64>();
                if probabilities
                    .iter()
                    .any(|value| *value < 0.0 || *value > 1.0)
                    || (sum - 1.0).abs() > 1e-9
                {
                    return Ok((
                        key,
                        invalid_fitness(
                            task,
                            metric,
                            &prediction.candidate_semantic_sha256,
                            complexity,
                            "multiclass probabilities are outside the simplex".into(),
                        ),
                    ));
                }
                let winner = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|left, right| {
                        left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0))
                    })
                    .unwrap()
                    .0;
                predicted.push(winner);
                log_loss -= probabilities[class].clamp(1e-15, 1.0).ln();
            }
            log_loss /= truth.len() as f64;
            let mut macro_f1 = 0.0;
            for class in 0..classes {
                let mut true_positive = 0u64;
                let mut false_positive = 0u64;
                let mut false_negative = 0u64;
                for (row, truth) in truth.iter().enumerate() {
                    let actual = *truth as usize;
                    let predicted = predicted[row];
                    true_positive += u64::from(actual == class && predicted == class);
                    false_positive += u64::from(actual != class && predicted == class);
                    false_negative += u64::from(actual == class && predicted != class);
                }
                let denominator = 2 * true_positive + false_positive + false_negative;
                macro_f1 += if denominator == 0 {
                    0.0
                } else {
                    2.0 * true_positive as f64 / denominator as f64
                };
            }
            macro_f1 /= classes as f64;
            FitnessVectorV1 {
                schema: FitnessVectorV1::SCHEMA.into(),
                evidence_valid: true,
                invalid_reason: None,
                constraints_satisfied: support.iter().all(|count| *count >= *min_class_support)
                    && log_loss <= *max_log_loss,
                primary: TypedMetricValueV1 {
                    task,
                    metric: metric.into(),
                    value: macro_f1,
                },
                stability_guards: vec![
                    GuardValueV1 {
                        name: "minimum_class_support".into(),
                        value: *support.iter().min().unwrap() as f64,
                    },
                    GuardValueV1 {
                        name: "negative_log_loss".into(),
                        value: -log_loss,
                    },
                ],
                complexity,
                candidate_semantic_sha256: prediction.candidate_semantic_sha256.clone(),
            }
        }
        RegisteredFitnessV1::NegativeMase { max_mase } => {
            if prediction.prediction_shape != vec![truth.len()]
                || tape.truth_shape != vec![truth.len()]
            {
                return Err("timeseries prediction and truth must both be one-dimensional".into());
            }
            let Some(scale_bits) = tape.auxiliary_scale_f64_bits else {
                return Err("timeseries scoring tape requires the native MASE scale".into());
            };
            let scale = f64::from_bits(scale_bits);
            let mase = predictions
                .iter()
                .zip(&truth)
                .map(|(prediction, truth)| (prediction - truth).abs())
                .sum::<f64>()
                / truth.len() as f64
                / scale;
            FitnessVectorV1 {
                schema: FitnessVectorV1::SCHEMA.into(),
                evidence_valid: true,
                invalid_reason: None,
                constraints_satisfied: mase <= *max_mase,
                primary: TypedMetricValueV1 {
                    task,
                    metric: metric.into(),
                    value: -mase,
                },
                stability_guards: vec![GuardValueV1 {
                    name: "negative_native_mase".into(),
                    value: -mase,
                }],
                complexity,
                candidate_semantic_sha256: prediction.candidate_semantic_sha256.clone(),
            }
        }
    };
    result.validate()?;
    Ok((key, result))
}
