use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct NumericDataset {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
}

/// Parse a numeric CSV whose final column is the target.
pub fn parse_last_target_csv(csv: &str) -> PolicyResult<NumericDataset> {
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for (index, line) in csv.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_fields(line)
            .map_err(|message| error(format!("CSV row {}: {message}", index + 1)))?;
        let mut values = Vec::with_capacity(fields.len());
        for field in fields {
            let value = field
                .trim()
                .parse::<f64>()
                .map_err(|_| error(format!("CSV row {} contains a nonnumeric value", index + 1)))?;
            if !value.is_finite() {
                return Err(error(format!(
                    "CSV row {} contains a non-finite value",
                    index + 1
                )));
            }
            values.push(value);
        }
        if values.len() < 2 {
            return Err(error(format!(
                "CSV row {} must contain a feature and target",
                index + 1
            )));
        }
        let target = values.pop().expect("length checked");
        features.push(values);
        targets.push(target);
    }
    if features.is_empty() {
        return Err(error("dataset is empty"));
    }
    Ok(NumericDataset { features, targets })
}

fn parse_csv_fields(line: &str) -> Result<Vec<String>, &'static str> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.trim_end_matches('\r').chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(character),
        }
    }
    if quoted {
        return Err("unterminated quoted field");
    }
    fields.push(field);
    Ok(fields)
}

pub trait AdaptiveEstimator {
    fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) -> PolicyResult<()>;
    fn predict(&self, features: &[Vec<f64>]) -> PolicyResult<Vec<f64>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveFitness {
    pub task: String,
    pub metric: &'static str,
    pub score: f64,
    pub secondary_metric: &'static str,
    pub secondary_score: f64,
    pub fit_split: &'static str,
    pub fitness_split: &'static str,
    pub cross_validation_fitness: bool,
    pub test_adaptive: bool,
    pub benchmark_valid: bool,
    pub selection_bias: &'static str,
    pub valid: bool,
}

/// Fit exactly once on train.csv and calculate fitness only on test.csv.
pub fn evaluate_adaptive<E: AdaptiveEstimator>(
    task: &str,
    train_csv: &str,
    test_csv: &str,
    estimator: &mut E,
) -> PolicyResult<AdaptiveFitness> {
    let train = parse_last_target_csv(train_csv)?;
    let test = parse_last_target_csv(test_csv)?;
    if train.features[0].len() != test.features[0].len() {
        return Err(error("train/test feature widths differ"));
    }
    estimator.fit(&train.features, &train.targets)?;
    let predictions = estimator.predict(&test.features)?;
    if predictions.len() != test.targets.len() || predictions.iter().any(|value| !value.is_finite())
    {
        return Err(error("adaptive predictions are invalid"));
    }
    let (metric, score, secondary_metric, secondary_score) = match task {
        "regression" => {
            let score = r2_score(&test.targets, &predictions)?;
            let mean_squared_error = test
                .targets
                .iter()
                .zip(&predictions)
                .map(|(target, prediction)| (target - prediction).powi(2))
                .sum::<f64>()
                / test.targets.len() as f64;
            ("r2", score, "rmse", mean_squared_error.sqrt())
        }
        "binary" => {
            if train
                .targets
                .iter()
                .chain(&test.targets)
                .any(|target| *target != 0.0 && *target != 1.0)
            {
                return Err(error("binary adaptive targets must be zero/one"));
            }
            (
                "auc",
                auc_score(&test.targets, &predictions)?,
                "log_loss",
                log_loss(&test.targets, &predictions)?,
            )
        }
        _ => return Err(error(format!("unsupported adaptive task {task}"))),
    };
    Ok(AdaptiveFitness {
        task: task.to_owned(),
        metric,
        score,
        secondary_metric,
        secondary_score,
        fit_split: "train.csv",
        fitness_split: "test.csv",
        cross_validation_fitness: false,
        test_adaptive: true,
        benchmark_valid: false,
        selection_bias: SELFPLAY_SELECTION_BIAS,
        valid: score.is_finite() && secondary_score.is_finite(),
    })
}

pub fn r2_score(targets: &[f64], predictions: &[f64]) -> PolicyResult<f64> {
    validate_score_vectors(targets, predictions)?;
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let denominator = targets
        .iter()
        .map(|target| (target - mean).powi(2))
        .sum::<f64>();
    if denominator == 0.0 {
        return Ok(f64::NAN);
    }
    Ok(1.0
        - targets
            .iter()
            .zip(predictions)
            .map(|(target, prediction)| (target - prediction).powi(2))
            .sum::<f64>()
            / denominator)
}

pub fn auc_score(targets: &[f64], predictions: &[f64]) -> PolicyResult<f64> {
    validate_score_vectors(targets, predictions)?;
    if targets
        .iter()
        .any(|target| *target != 0.0 && *target != 1.0)
    {
        return Err(error("AUC targets must be zero/one"));
    }
    let positives = targets.iter().filter(|target| **target == 1.0).count();
    let negatives = targets.len() - positives;
    if positives == 0 || negatives == 0 {
        return Ok(f64::NAN);
    }
    let mut ordered = predictions
        .iter()
        .copied()
        .zip(targets.iter().copied())
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut rank_sum = 0.0;
    let mut offset = 0;
    while offset < ordered.len() {
        let mut end = offset + 1;
        while end < ordered.len() && ordered[end].0 == ordered[offset].0 {
            end += 1;
        }
        let average_rank = (offset + 1 + end) as f64 / 2.0;
        let tied_positives = ordered[offset..end]
            .iter()
            .filter(|row| row.1 == 1.0)
            .count();
        rank_sum += average_rank * tied_positives as f64;
        offset = end;
    }
    Ok((rank_sum - positives as f64 * (positives + 1) as f64 / 2.0)
        / (positives * negatives) as f64)
}

pub fn log_loss(targets: &[f64], predictions: &[f64]) -> PolicyResult<f64> {
    validate_score_vectors(targets, predictions)?;
    if targets
        .iter()
        .any(|target| *target != 0.0 && *target != 1.0)
    {
        return Err(error("log-loss targets must be zero/one"));
    }
    let epsilon = 1e-15;
    let total = targets
        .iter()
        .zip(predictions)
        .map(|(target, prediction)| {
            let probability = prediction.clamp(epsilon, 1.0 - epsilon);
            -(target * probability.ln() + (1.0 - target) * (1.0 - probability).ln())
        })
        .sum::<f64>();
    Ok(total / targets.len() as f64)
}

fn validate_score_vectors(targets: &[f64], predictions: &[f64]) -> PolicyResult<()> {
    if targets.is_empty() || targets.len() != predictions.len() {
        return Err(error(
            "targets and predictions must have the same nonzero length",
        ));
    }
    if targets
        .iter()
        .chain(predictions)
        .any(|value| !value.is_finite())
    {
        return Err(error("targets and predictions must be finite"));
    }
    Ok(())
}
