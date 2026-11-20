//! Multiclass probability objectives (`jope.multiclass-fitness/v1`).
//!
//! Probability columns and labels follow the explicit ordered `class_order` (`K >= 3`) declared in
//! the specification. Logloss uses natural logarithms without clipping; multiclass Brier is
//! `sum_{k=0..K-1} (p_k - I[y == k])^2` aggregated by mean or sum across rows. `macro_f1`,
//! `accuracy`, `balanced_accuracy`, and `cost_matrix_loss` operate on deterministic `argmax`
//! predictions (breaking exact probability ties toward the earliest index in `class_order`) and
//! require every declared class to appear at least once in dataset-level recall denominators where
//! required (`macro_f1`, `balanced_accuracy`).

pub use crate::fitness::Aggregation;
use crate::fitness::{Direction, Score};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeSet;

pub const FITNESS_VERSION: &str = "jope.multiclass-fitness/v1";
pub const MAX_SPEC_BYTES: usize = 32 * 1024;
pub const MIN_CLASSES: usize = 3;
pub const MAX_CLASSES: usize = 1024;
/// Absolute tolerance for a K-class probability vector's sum (`1.0`). Inputs are never renormalized.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;

pub fn schema() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../schemas/jope.multiclass-fitness.v1.schema.json"
    ))
    .expect("committed multiclass fitness schema")
}

#[derive(Debug, thiserror::Error)]
pub enum FitnessError {
    #[error("invalid multiclass fitness specification: {0}")]
    InvalidSpecification(String),
    #[error("FITNESS_UNDEFINED: {0}")]
    Undefined(String),
}

type Result<T> = std::result::Result<T, FitnessError>;

/// Supported K-class (`K >= 3`) classification metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MulticlassMetric {
    Logloss,
    Brier,
    MacroF1,
    Accuracy,
    BalancedAccuracy,
    CostMatrixLoss,
}

pub type Metric = MulticlassMetric;

/// Candidate failure policy on undefined / non-finite predictions or losses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndefinedPolicy {
    FailCandidate,
}

/// Validated K-class (`K >= 3`) objective specification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MulticlassFitness {
    pub schema: String,
    pub metric: Metric,
    pub class_order: Vec<String>,
    #[serde(with = "multiclass_aggregation_serde")]
    pub aggregation: Aggregation,
    pub undefined: UndefinedPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_matrix: Option<Vec<Vec<f64>>>,
}

mod multiclass_aggregation_serde {
    use super::{Aggregation, Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &Aggregation, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Aggregation::Mean {} => "mean".serialize(serializer),
            Aggregation::Sum {} => "sum".serialize(serializer),
            _ => Err(serde::ser::Error::custom(
                "multiclass aggregation supports only mean or sum",
            )),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Aggregation, D::Error>
    where
        D: Deserializer<'de>,
    {
        match String::deserialize(deserializer)?.as_str() {
            "mean" => Ok(Aggregation::Mean {}),
            "sum" => Ok(Aggregation::Sum {}),
            other => Err(serde::de::Error::custom(format!(
                "unsupported multiclass aggregation `{other}`"
            ))),
        }
    }
}

/// Aligned held-out labels and K-column probability rows.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    pub class_order: &'a [&'a str],
    pub labels: &'a [&'a str],
    pub probabilities: &'a [Vec<f64>],
}

impl Metric {
    pub fn name(self) -> &'static str {
        match self {
            Self::Logloss => "logloss",
            Self::Brier => "brier",
            Self::MacroF1 => "macro_f1",
            Self::Accuracy => "accuracy",
            Self::BalancedAccuracy => "balanced_accuracy",
            Self::CostMatrixLoss => "cost_matrix_loss",
        }
    }

    pub fn from_shorthand_name(name: &str) -> Option<Self> {
        match name {
            "logloss" | "log_loss" => Some(Self::Logloss),
            "brier" => Some(Self::Brier),
            "macro_f1" | "f1_macro" => Some(Self::MacroF1),
            "accuracy" | "acc" => Some(Self::Accuracy),
            "balanced_accuracy" => Some(Self::BalancedAccuracy),
            "cost_matrix_loss" => Some(Self::CostMatrixLoss),
            _ => None,
        }
    }

    /// True when the shorthand is exclusively multiclass (not shared with binary shorthands).
    pub fn is_multiclass_only_shorthand(name: &str) -> bool {
        matches!(name, "macro_f1" | "f1_macro" | "accuracy" | "acc")
    }

    pub fn direction(self) -> Direction {
        match self {
            Self::MacroF1 | Self::Accuracy | Self::BalancedAccuracy => Direction::Maximize,
            Self::Logloss | Self::Brier | Self::CostMatrixLoss => Direction::Minimize,
        }
    }

    /// Whether post-hoc probability calibration (Temperature Scaling) can strictly improve the metric.
    pub fn probability_sensitive(self) -> bool {
        matches!(self, Self::Logloss | Self::Brier | Self::CostMatrixLoss)
    }
}

fn argmax_class(row: &[f64]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = row[0];
    for (idx, &val) in row.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}

impl MulticlassFitness {
    /// Strict, bounded JSON decoding (rejects duplicate keys, unknown fields, and > 32 KiB specs).
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

    /// Resolve a built-in multiclass shorthand (`logloss`/`log_loss`, `brier`, `macro_f1`,
    /// `accuracy`, `balanced_accuracy`, `logloss:aggregation=sum`, `brier:aggregation=sum`)
    /// against an explicit `class_order` (`len >= 3`).
    pub fn from_shorthand(text: &str, class_order: Vec<String>) -> Result<Self> {
        if text.len() > MAX_SPEC_BYTES {
            return Err(FitnessError::InvalidSpecification(
                "specification exceeds 32 KiB".into(),
            ));
        }
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(FitnessError::InvalidSpecification(
                "empty multiclass fitness specification".into(),
            ));
        }
        let (raw_name, overrides) = match trimmed.split_once(':') {
            Some((name, rest)) => (name.trim(), Some(rest)),
            None => (trimmed, None),
        };
        let metric = Metric::from_shorthand_name(raw_name).ok_or_else(|| {
            FitnessError::InvalidSpecification(format!("unknown multiclass fitness {raw_name:?}"))
        })?;
        if metric == Metric::CostMatrixLoss {
            return Err(FitnessError::InvalidSpecification(
                "cost_matrix_loss requires a JSON specification with cost_matrix".into(),
            ));
        }
        let mut aggregation = Aggregation::Mean {};
        if let Some(overrides) = overrides {
            if overrides.trim().is_empty() {
                return Err(FitnessError::InvalidSpecification(
                    "empty parameter list after `:`".into(),
                ));
            }
            for raw_pair in overrides.split(',') {
                let pair = raw_pair.trim();
                if pair.is_empty() {
                    return Err(FitnessError::InvalidSpecification(
                        "empty parameter entry in shorthand".into(),
                    ));
                }
                let (raw_key, raw_value) = pair.split_once('=').ok_or_else(|| {
                    FitnessError::InvalidSpecification(format!(
                        "override {pair:?} must be key=value"
                    ))
                })?;
                let key = raw_key.trim();
                let value = raw_value.trim();
                match key {
                    "aggregation" => {
                        aggregation = match value {
                            "mean" => Aggregation::Mean {},
                            "sum" => Aggregation::Sum {},
                            other => {
                                return Err(FitnessError::InvalidSpecification(format!(
                                    "unsupported multiclass aggregation `{other}`"
                                )));
                            }
                        };
                    }
                    other_key => {
                        return Err(FitnessError::InvalidSpecification(format!(
                            "{} has no shorthand parameter {other_key:?}",
                            metric.name()
                        )));
                    }
                }
            }
        }
        let spec = Self {
            schema: FITNESS_VERSION.to_owned(),
            metric,
            class_order,
            aggregation,
            undefined: UndefinedPolicy::FailCandidate,
            cost_matrix: None,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Parse either an inline `jope.multiclass-fitness/v1` JSON object (verifying `class_order`)
    /// or a built-in shorthand bound to `class_order`.
    pub fn parse_for_classes(text: &str, class_order: Vec<String>) -> Result<Self> {
        if text.len() > MAX_SPEC_BYTES {
            return Err(FitnessError::InvalidSpecification(
                "specification exceeds 32 KiB".into(),
            ));
        }
        if text.trim_start().starts_with('{') {
            let parsed = Self::parse(text)?;
            if parsed.class_order != class_order {
                return Err(FitnessError::InvalidSpecification(format!(
                    "specification class_order {:?} does not match dataset classes {:?}",
                    parsed.class_order, class_order
                )));
            }
            return Ok(parsed);
        }
        Self::from_shorthand(text, class_order)
    }

    pub fn name(&self) -> &'static str {
        self.metric.name()
    }

    pub fn direction(&self) -> Direction {
        self.metric.direction()
    }

    pub fn probability_sensitive(&self) -> bool {
        self.metric.probability_sensitive()
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != FITNESS_VERSION {
            return Err(FitnessError::InvalidSpecification(
                "unsupported schema version".into(),
            ));
        }
        let k = self.class_order.len();
        if !(MIN_CLASSES..=MAX_CLASSES).contains(&k) {
            return Err(FitnessError::InvalidSpecification(format!(
                "class_order requires {MIN_CLASSES}..={MAX_CLASSES} classes, got {k}"
            )));
        }
        let mut seen = BTreeSet::new();
        for name in &self.class_order {
            if name.is_empty() || name.chars().count() > 256 || !seen.insert(name.as_str()) {
                return Err(FitnessError::InvalidSpecification(
                    "class_order requires distinct names of 1..256 characters".into(),
                ));
            }
        }
        match self.metric {
            Metric::CostMatrixLoss => {
                let matrix = self.cost_matrix.as_ref().ok_or_else(|| {
                    FitnessError::InvalidSpecification(
                        "cost_matrix is required for cost_matrix_loss".into(),
                    )
                })?;
                if matrix.len() != k || matrix.iter().any(|row| row.len() != k) {
                    return Err(FitnessError::InvalidSpecification(format!(
                        "cost_matrix must have shape {k}x{k}"
                    )));
                }
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
                if self.cost_matrix.is_some() {
                    return Err(FitnessError::InvalidSpecification(
                        "cost_matrix is only valid for cost_matrix_loss".into(),
                    ));
                }
            }
        }
        if matches!(
            self.metric,
            Metric::MacroF1 | Metric::Accuracy | Metric::BalancedAccuracy | Metric::CostMatrixLoss
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
        let bytes = serde_json::to_vec(self)
            .map_err(|e| FitnessError::InvalidSpecification(e.to_string()))?;
        Ok(crate::sha256_hex(&bytes))
    }

    pub fn evaluate(&self, inputs: Inputs<'_>) -> Result<Score> {
        self.validate()?;
        let k = self.class_order.len();
        if inputs.class_order.len() != k
            || inputs
                .class_order
                .iter()
                .zip(&self.class_order)
                .any(|(left, right)| *left != right.as_str())
        {
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
        for (row_idx, (label, probs)) in inputs
            .labels
            .iter()
            .zip(inputs.probabilities.iter())
            .enumerate()
        {
            let class_idx = inputs
                .class_order
                .iter()
                .position(|c| c == label)
                .ok_or_else(|| {
                    FitnessError::Undefined(format!("unknown label at row {row_idx}"))
                })?;
            if probs.len() != k
                || probs
                    .iter()
                    .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            {
                return Err(FitnessError::Undefined(format!(
                    "row {row_idx} needs {k} finite probabilities in [0,1]"
                )));
            }
            let sum: f64 = probs.iter().sum();
            if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
                return Err(FitnessError::Undefined(format!(
                    "row {row_idx} probabilities sum to {sum}, outside 1 +- {PROBABILITY_SUM_TOLERANCE:e}"
                )));
            }
            labels.push(class_idx);
        }

        let value = match self.metric {
            Metric::Logloss => {
                let mut total = 0.0;
                for (&y, probs) in labels.iter().zip(inputs.probabilities.iter()) {
                    let p_true = probs[y];
                    if p_true == 0.0 {
                        return Err(FitnessError::Undefined(
                            "logloss has zero true-class probability (no clipping)".into(),
                        ));
                    }
                    total += -p_true.ln();
                }
                match self.aggregation {
                    Aggregation::Mean {} => total / labels.len() as f64,
                    Aggregation::Sum {} => total,
                    _ => unreachable!(),
                }
            }
            Metric::Brier => {
                let mut total = 0.0;
                for (&y, probs) in labels.iter().zip(inputs.probabilities.iter()) {
                    for (c, &p) in probs.iter().enumerate() {
                        let indicator = if c == y { 1.0 } else { 0.0 };
                        total += (p - indicator).powi(2);
                    }
                }
                match self.aggregation {
                    Aggregation::Mean {} => total / labels.len() as f64,
                    Aggregation::Sum {} => total,
                    _ => unreachable!(),
                }
            }
            Metric::Accuracy => {
                let correct = labels
                    .iter()
                    .zip(inputs.probabilities.iter())
                    .filter(|(&y, probs)| argmax_class(probs) == y)
                    .count();
                correct as f64 / labels.len() as f64
            }
            Metric::BalancedAccuracy => {
                let mut support = vec![0usize; k];
                let mut tp = vec![0usize; k];
                for (&y, probs) in labels.iter().zip(inputs.probabilities.iter()) {
                    support[y] += 1;
                    if argmax_class(probs) == y {
                        tp[y] += 1;
                    }
                }
                if support.contains(&0) {
                    return Err(FitnessError::Undefined(
                        "balanced_accuracy requires every declared class to appear in the fold"
                            .into(),
                    ));
                }
                let sum_recall: f64 = (0..k).map(|c| tp[c] as f64 / support[c] as f64).sum();
                sum_recall / k as f64
            }
            Metric::MacroF1 => {
                let mut support = vec![0usize; k];
                let mut predicted = vec![0usize; k];
                let mut tp = vec![0usize; k];
                for (&y, probs) in labels.iter().zip(inputs.probabilities.iter()) {
                    let pred = argmax_class(probs);
                    support[y] += 1;
                    predicted[pred] += 1;
                    if pred == y {
                        tp[y] += 1;
                    }
                }
                if support.contains(&0) {
                    return Err(FitnessError::Undefined(
                        "macro_f1 requires every declared class to appear in the fold".into(),
                    ));
                }
                let sum_f1: f64 = (0..k)
                    .map(|c| {
                        let denom = support[c] + predicted[c];
                        if denom == 0 {
                            0.0
                        } else {
                            (2 * tp[c]) as f64 / denom as f64
                        }
                    })
                    .sum();
                sum_f1 / k as f64
            }
            Metric::CostMatrixLoss => {
                let matrix = self.cost_matrix.as_ref().expect("validated cost_matrix");
                let total: f64 = labels
                    .iter()
                    .zip(inputs.probabilities.iter())
                    .map(|(&y, probs)| {
                        // Bayes optimal action under the calibrated row distribution:
                        let (best_action, _) =
                            (0..k).fold((0usize, f64::INFINITY), |(best_a, best_e), action| {
                                let expected: f64 = probs
                                    .iter()
                                    .zip(matrix.iter())
                                    .map(|(&p, cost_row)| p * cost_row[action])
                                    .sum();
                                if expected < best_e {
                                    (action, expected)
                                } else {
                                    (best_a, best_e)
                                }
                            });
                        matrix[y][best_action]
                    })
                    .sum();
                total / labels.len() as f64
            }
        };

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

#[cfg(test)]
mod tests {
    use super::*;

    fn classes() -> Vec<String> {
        vec!["cat".to_owned(), "dog".to_owned(), "fox".to_owned()]
    }

    #[test]
    fn committed_schema_parses() {
        let s = schema();
        assert_eq!(s["properties"]["schema"]["const"], FITNESS_VERSION);
    }

    #[test]
    fn shorthands_and_metrics_evaluate_cleanly() {
        let labels = ["cat", "dog", "fox", "cat", "dog", "fox"];
        let probs = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.1, 0.7, 0.2],
            vec![0.1, 0.2, 0.7],
            vec![0.6, 0.2, 0.2],
            vec![0.2, 0.6, 0.2],
            vec![0.2, 0.3, 0.5],
        ];
        let order = ["cat", "dog", "fox"];

        for (shorthand, dir) in [
            ("logloss", Direction::Minimize),
            ("log_loss", Direction::Minimize),
            ("brier", Direction::Minimize),
            ("macro_f1", Direction::Maximize),
            ("accuracy", Direction::Maximize),
            ("balanced_accuracy", Direction::Maximize),
        ] {
            let spec = MulticlassFitness::parse_for_classes(shorthand, classes()).unwrap();
            let score = spec
                .evaluate(Inputs {
                    class_order: &order,
                    labels: &labels,
                    probabilities: &probs,
                })
                .unwrap();
            assert_eq!(score.direction, dir, "{shorthand}");
            assert_eq!(score.fitness_sha256.len(), 64);
            if dir == Direction::Maximize {
                assert!((score.value - 1.0).abs() < 1e-12, "{shorthand}");
                assert_eq!(score.loss, -score.value);
            } else {
                assert!(score.value > 0.0 && score.value < 1.0);
                assert_eq!(score.loss, score.value);
            }
        }
    }

    #[test]
    fn cost_matrix_and_validation_refusals_work() {
        let json_spec = r#"{
            "schema": "jope.multiclass-fitness/v1",
            "metric": "cost_matrix_loss",
            "class_order": ["cat", "dog", "fox"],
            "aggregation": "mean",
            "undefined": "fail_candidate",
            "cost_matrix": [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [5.0, 2.0, 0.0]]
        }"#;
        let spec = MulticlassFitness::parse(json_spec).unwrap();
        let order = ["cat", "dog", "fox"];
        let labels = ["cat", "dog", "fox"];
        let probs = vec![
            vec![0.9, 0.05, 0.05],
            vec![0.05, 0.9, 0.05],
            vec![0.05, 0.05, 0.9],
        ];
        let score = spec
            .evaluate(Inputs {
                class_order: &order,
                labels: &labels,
                probabilities: &probs,
            })
            .unwrap();
        assert_eq!(score.value, 0.0);

        // Fewer than 3 classes refused
        assert!(
            MulticlassFitness::from_shorthand("logloss", vec!["a".into(), "b".into()]).is_err()
        );

        // Bad probability sum refused
        assert!(spec
            .evaluate(Inputs {
                class_order: &order,
                labels: &labels,
                probabilities: &[
                    vec![0.3, 0.3, 0.3],
                    vec![0.1, 0.8, 0.1],
                    vec![0.1, 0.1, 0.8]
                ],
            })
            .is_err());
    }
}
