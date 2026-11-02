use super::*;

/// Dataset-level binary metrics. `f1` accepts the optional top-level threshold; the other
/// threshold metrics use the v1 default of 0.5. Cost-matrix loss accepts the top-level matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationMetric {
    Logloss,
    Brier,
    RocAuc,
    PrAuc,
    F1,
    BalancedAccuracy,
    CostMatrixLoss,
}

/// Compatibility path; the concrete enum has a domain-specific name.
pub type Metric = ClassificationMetric;
/// No omission, fallback score, probability clipping or replacement of non-finite predictions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndefinedPolicy {
    FailCandidate,
}

/// Core fields are required. Metric parameters are accepted only by their matching metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationFitness {
    pub schema: String,
    pub metric: Metric,
    pub class_order: [String; 2],
    #[serde(with = "classification_aggregation_serde")]
    pub aggregation: Aggregation,
    pub undefined: UndefinedPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_matrix: Option<[[f64; 2]; 2]>,
}

mod classification_aggregation_serde {
    use super::Aggregation;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &Aggregation, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Aggregation::Mean {} => "mean".serialize(serializer),
            Aggregation::Sum {} => "sum".serialize(serializer),
            _ => Err(serde::ser::Error::custom(
                "classification aggregation supports only mean or sum",
            )),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Aggregation, D::Error>
    where
        D: Deserializer<'de>,
    {
        match String::deserialize(deserializer)?.as_str() {
            "mean" => Ok(Aggregation::Mean {}),
            "sum" => Ok(Aggregation::Sum {}),
            other => Err(serde::de::Error::custom(format!(
                "unsupported classification aggregation `{other}`"
            ))),
        }
    }
}

/// Aligned held-out rows. Each probability pair follows `class_order`, which must match the
/// specification exactly. Labels are class names, never implicitly encoded numeric positions.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    pub class_order: [&'a str; 2],
    pub labels: &'a [&'a str],
    pub probabilities: &'a [[f64; 2]],
}

impl ClassificationFitness {
    /// Strict, bounded JSON: duplicate keys, unknown fields and trailing values are refused.
    pub fn parse(text: &str) -> Result<Self> {
        if text.len() > MAX_SPEC_BYTES {
            return Err(FitnessError::InvalidSpecification(
                "specification exceeds 32 KiB".into(),
            ));
        }
        let spec: Self = crate::strict_json::decode(text.as_bytes())
            .map_err(|e| FitnessError::InvalidSpecification(e.to_string()))?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != FITNESS_VERSION {
            return Err(FitnessError::InvalidSpecification(
                "unsupported schema version".into(),
            ));
        }
        if self.class_order[0] == self.class_order[1]
            || self
                .class_order
                .iter()
                .any(|name| name.is_empty() || name.chars().count() > 256)
        {
            return Err(FitnessError::InvalidSpecification(
                "class_order requires two distinct names of 1..256 characters".into(),
            ));
        }
        match self.metric {
            Metric::F1 => {
                let threshold = self.threshold.unwrap_or(DEFAULT_CLASSIFICATION_THRESHOLD);
                if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                    return Err(FitnessError::InvalidSpecification(
                        "F1 threshold must be finite and in [0,1]".into(),
                    ));
                }
                if self.cost_matrix.is_some() {
                    return Err(FitnessError::InvalidSpecification(
                        "cost_matrix is only valid for cost_matrix_loss".into(),
                    ));
                }
            }
            Metric::CostMatrixLoss => {
                if self.threshold.is_some() {
                    return Err(FitnessError::InvalidSpecification(
                        "threshold is only valid for f1".into(),
                    ));
                }
                let matrix = self.cost_matrix.ok_or_else(|| {
                    FitnessError::InvalidSpecification(
                        "cost_matrix is required for cost_matrix_loss".into(),
                    )
                })?;
                if matrix
                    .iter()
                    .flatten()
                    .any(|cost| !cost.is_finite() || *cost < 0.0)
                {
                    return Err(FitnessError::InvalidSpecification(
                        "cost_matrix entries must be finite and non-negative".into(),
                    ));
                }
            }
            _ => {
                if self.threshold.is_some() || self.cost_matrix.is_some() {
                    return Err(FitnessError::InvalidSpecification(
                        "threshold and cost_matrix are only valid for their matching metrics"
                            .into(),
                    ));
                }
            }
        }
        if matches!(
            self.metric,
            Metric::RocAuc
                | Metric::PrAuc
                | Metric::F1
                | Metric::BalancedAccuracy
                | Metric::CostMatrixLoss
        ) && self.aggregation != (Aggregation::Mean {})
        {
            return Err(FitnessError::InvalidSpecification(
                "dataset-level metrics require mean aggregation".into(),
            ));
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        let mut canonical = self.clone();
        if canonical.metric == Metric::F1 && canonical.threshold.is_none() {
            canonical.threshold = Some(DEFAULT_CLASSIFICATION_THRESHOLD);
        }
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|e| FitnessError::InvalidSpecification(e.to_string()))?;
        Ok(crate::sha256_hex(&bytes))
    }

    /// Every metric minimizes. New dataset-level metrics require both classes in each fold.
    pub fn evaluate(&self, inputs: Inputs<'_>) -> Result<Score> {
        self.validate()?;
        if inputs.class_order != [self.class_order[0].as_str(), self.class_order[1].as_str()] {
            return Err(FitnessError::Undefined(
                "prediction class order differs from the specification".into(),
            ));
        }
        if inputs.labels.is_empty() || inputs.labels.len() != inputs.probabilities.len() {
            return Err(FitnessError::Undefined(
                "labels and probability rows must be nonempty and aligned".into(),
            ));
        }
        let mut labels = Vec::with_capacity(inputs.labels.len());
        for (row, (label, probabilities)) in
            inputs.labels.iter().zip(inputs.probabilities).enumerate()
        {
            let index = inputs
                .class_order
                .iter()
                .position(|class| class == label)
                .ok_or_else(|| FitnessError::Undefined(format!("unknown label at row {row}")))?;
            if probabilities
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
                || (probabilities[0] + probabilities[1] - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
            {
                return Err(FitnessError::Undefined(format!(
                    "row {row} needs finite probabilities in [0,1] summing to one within 1e-12"
                )));
            }
            labels.push(index);
        }
        let needs_both_classes = matches!(
            self.metric,
            Metric::RocAuc
                | Metric::PrAuc
                | Metric::F1
                | Metric::BalancedAccuracy
                | Metric::CostMatrixLoss
        );
        if needs_both_classes && (!labels.contains(&0) || !labels.contains(&1)) {
            return Err(FitnessError::Undefined(
                "metric requires both classes in the fold".into(),
            ));
        }
        let value = self.metric.value(
            &labels,
            inputs.probabilities,
            self.aggregation.clone(),
            self.threshold.unwrap_or(DEFAULT_CLASSIFICATION_THRESHOLD),
            self.cost_matrix,
        )?;
        if !value.is_finite() {
            return Err(FitnessError::Undefined("non-finite objective value".into()));
        }
        let direction = self.metric.direction();
        Ok(Score {
            value,
            loss: if direction == Direction::Maximize {
                -value
            } else {
                value
            },
            direction,
            fitness_sha256: self.identity()?,
        })
    }
}
