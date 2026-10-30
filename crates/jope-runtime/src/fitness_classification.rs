//! Binary probability objectives (`jope.classification-fitness/v1`, spec §5.10).
//!
//! Probability columns and labels use the explicit, ordered class names in the specification.
//! `class_order[1]` is the positive class. Logloss uses natural logarithms without clipping;
//! Brier is the mean/sum of `(p_positive - is_positive)^2`, without a factor of two.
//! Invalid specifications fail before evaluation. Invalid inputs or undefined values return
//! [`FitnessError::Undefined`]; the caller must fail that candidate, never assign a placeholder.
//! Calibration, event publication and discovery-loop integration belong to their owning tasks.

use crate::fitness::{Direction, Score};
use serde::{Deserialize, Serialize};

pub const FITNESS_VERSION: &str = "jope.classification-fitness/v1";
pub const MAX_SPEC_BYTES: usize = 32 * 1024;
/// Absolute tolerance for a probability pair's sum. Inputs are never renormalized.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-12;

pub fn schema() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../schemas/jope.classification-fitness.v1.schema.json"
    ))
    .expect("committed classification fitness schema")
}

#[derive(Debug, thiserror::Error)]
pub enum FitnessError {
    #[error("invalid classification fitness specification: {0}")]
    InvalidSpecification(String),
    #[error("FITNESS_UNDEFINED: {0}")]
    Undefined(String),
}

type Result<T> = std::result::Result<T, FitnessError>;

/// OBJ-03 extension point: add the remaining binary metrics and their explicit parameters here,
/// in `Metric::value`, and in the schema. Rank/threshold metrics must define aggregation themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Logloss,
    Brier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aggregation {
    Mean,
    Sum,
}

/// No omission, fallback score, probability clipping or replacement of non-finite predictions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndefinedPolicy {
    FailCandidate,
}

/// All fields are required and contribute to the objective identity; there are no hidden defaults.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationFitness {
    pub schema: String,
    pub metric: Metric,
    pub class_order: [String; 2],
    pub aggregation: Aggregation,
    pub undefined: UndefinedPolicy,
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
        Ok(())
    }

    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|e| FitnessError::InvalidSpecification(e.to_string()))?;
        Ok(crate::sha256_hex(&bytes))
    }

    /// Both supported metrics minimize. A single-class fold is defined for these losses.
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
        let value = self
            .metric
            .value(&labels, inputs.probabilities, self.aggregation)?;
        if !value.is_finite() {
            return Err(FitnessError::Undefined("non-finite objective value".into()));
        }
        Ok(Score {
            value,
            loss: value,
            direction: Direction::Minimize,
            fitness_sha256: self.identity()?,
        })
    }
}

impl Metric {
    // OBJ-03 extension point: dispatch dataset-level rank/threshold/cost metrics here, retaining
    // validation above and returning FitnessError::Undefined for their undefined conditions.
    fn value(
        self,
        labels: &[usize],
        probabilities: &[[f64; 2]],
        aggregation: Aggregation,
    ) -> Result<f64> {
        let mut total = 0.0;
        for (&label, p) in labels.iter().zip(probabilities) {
            total += match self {
                Self::Logloss => {
                    if p[label] == 0.0 {
                        return Err(FitnessError::Undefined(
                            "logloss has zero true-class probability (no clipping)".into(),
                        ));
                    }
                    -p[label].ln()
                }
                Self::Brier => (p[1] - label as f64).powi(2),
            };
        }
        Ok(match aggregation {
            Aggregation::Mean => total / labels.len() as f64,
            Aggregation::Sum => total,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(metric: &str, aggregation: &str) -> ClassificationFitness {
        ClassificationFitness::parse(&format!(
            r#"{{"schema":"jope.classification-fitness/v1","metric":"{metric}",
            "class_order":["no","yes"],"aggregation":"{aggregation}",
            "undefined":"fail_candidate"}}"#
        ))
        .unwrap()
    }

    fn inputs<'a>(labels: &'a [&'a str], probabilities: &'a [[f64; 2]]) -> Inputs<'a> {
        Inputs {
            class_order: ["no", "yes"],
            labels,
            probabilities,
        }
    }

    #[test]
    fn hand_computed_golden_values_and_aggregations() {
        // True-class probabilities 1/2, 1/4, 3/4, 1: sum loss ln(32/3).
        // Brier row losses 1/4, 9/16, 1/16, 0: sum 7/8, mean 7/32.
        let labels = ["no", "yes", "no", "yes"];
        let probabilities = [[0.5, 0.5], [0.75, 0.25], [0.75, 0.25], [0.0, 1.0]];
        for (metric, mean, sum) in [
            ("logloss", 0.5917809035329042, 2.367123614131617),
            ("brier", 0.21875, 0.875),
        ] {
            for (aggregation, expected) in [("mean", mean), ("sum", sum)] {
                let objective = spec(metric, aggregation);
                let score = objective.evaluate(inputs(&labels, &probabilities)).unwrap();
                assert!((score.value - expected).abs() <= 1e-12);
                assert_eq!(score.loss, score.value);
                assert_eq!(score.direction, Direction::Minimize);
                assert_eq!(score.fitness_sha256, objective.identity().unwrap());
            }
        }
    }

    #[test]
    fn endpoints_single_class_and_small_positive_probabilities() {
        for metric in ["logloss", "brier"] {
            let objective = spec(metric, "mean");
            let perfect = objective
                .evaluate(inputs(&["yes", "yes"], &[[0.0, 1.0], [0.0, 1.0]]))
                .unwrap();
            assert_eq!(perfect.value, 0.0);
            assert_eq!(
                spec("brier", "mean")
                    .evaluate(inputs(&["no", "yes"], &[[0.0, 1.0], [1.0, 0.0]]))
                    .unwrap()
                    .value,
                1.0
            );
        }
        let objective = spec("logloss", "mean");
        assert!(matches!(
            objective.evaluate(inputs(&["yes"], &[[1.0, 0.0]])),
            Err(FitnessError::Undefined(_))
        ));
        let tiny = objective
            .evaluate(inputs(&["yes"], &[[1.0, f64::MIN_POSITIVE]]))
            .unwrap();
        assert!((tiny.value - 708.3964185322641).abs() <= 1e-12);
    }

    #[test]
    fn class_names_order_and_column_mapping_are_explicit() {
        let mut objective = spec("brier", "mean");
        objective.class_order = ["yes".into(), "no".into()];
        let original = inputs(&["yes", "no"], &[[0.25, 0.75], [0.75, 0.25]]);
        assert!(matches!(
            objective.evaluate(original),
            Err(FitnessError::Undefined(_))
        ));
        let swapped = Inputs {
            class_order: ["yes", "no"],
            labels: original.labels,
            probabilities: &[[0.75, 0.25], [0.25, 0.75]],
        };
        assert_eq!(objective.evaluate(swapped).unwrap().value, 0.0625);
    }

    #[test]
    fn malformed_specs_are_rejected_before_scoring() {
        let example = serde_json::to_value(spec("logloss", "mean")).unwrap();
        for field in [
            "schema",
            "metric",
            "class_order",
            "aggregation",
            "undefined",
        ] {
            let mut missing = example.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(ClassificationFitness::parse(&missing.to_string()).is_err());
        }
        for (field, value) in [
            (
                "schema",
                serde_json::json!("jope.classification-fitness/v2"),
            ),
            ("metric", serde_json::json!("roc_auc")),
            ("aggregation", serde_json::json!("median")),
            ("undefined", serde_json::json!("skip")),
            ("class_order", serde_json::json!(["yes", "yes"])),
            ("class_order", serde_json::json!(["", "yes"])),
            ("class_order", serde_json::json!(["no", "yes", "other"])),
            ("class_order", serde_json::json!([0, 1])),
            ("class_order", serde_json::json!(["no", "x".repeat(257)])),
            ("extra", serde_json::json!(true)),
        ] {
            let mut invalid = example.clone();
            invalid[field] = value;
            assert!(matches!(
                ClassificationFitness::parse(&invalid.to_string()),
                Err(FitnessError::InvalidSpecification(_))
            ));
        }
        let valid = example.to_string();
        for invalid in [
            format!("{valid} {{}}"),
            valid.replacen('{', r#"{"schema":"duplicate", "#, 1),
            " ".repeat(MAX_SPEC_BYTES + 1),
        ] {
            assert!(ClassificationFitness::parse(&invalid).is_err());
        }
        let mut invalid = spec("brier", "sum");
        invalid.schema = "wrong".into();
        assert!(matches!(
            invalid.evaluate(inputs(&[], &[])),
            Err(FitnessError::InvalidSpecification(_))
        ));
    }

    #[test]
    fn invalid_predictions_fail_only_the_candidate_without_a_score() {
        for metric in ["logloss", "brier"] {
            let objective = spec(metric, "mean");
            for invalid in [
                inputs(&[], &[]),
                inputs(&["yes"], &[]),
                inputs(&[], &[[0.5, 0.5]]),
                inputs(&["unknown"], &[[0.5, 0.5]]),
                inputs(&["yes"], &[[f64::NAN, 0.5]]),
                inputs(&["yes"], &[[0.5, f64::INFINITY]]),
                inputs(&["yes"], &[[-0.1, 1.1]]),
                inputs(&["yes"], &[[0.2, 0.3]]),
            ] {
                let error = objective.evaluate(invalid).unwrap_err();
                assert!(matches!(error, FitnessError::Undefined(_)));
                assert!(error.to_string().starts_with("FITNESS_UNDEFINED:"));
            }
            assert!(objective.evaluate(inputs(&["yes"], &[[0.5, 0.5]])).is_ok());
        }
    }

    #[test]
    fn probability_sum_tolerance_does_not_normalize() {
        let objective = spec("logloss", "mean");
        let accepted = objective
            .evaluate(inputs(&["yes"], &[[0.5 + 5e-13, 0.5]]))
            .unwrap();
        assert_eq!(accepted.value, std::f64::consts::LN_2);
        assert!(objective
            .evaluate(inputs(&["yes"], &[[0.5 + 2e-12, 0.5]]))
            .is_err());
    }

    #[test]
    fn schema_examples_roundtrip_and_identity_bind_every_field() {
        for example in schema()["examples"].as_array().unwrap() {
            let parsed = ClassificationFitness::parse(&example.to_string()).unwrap();
            assert_eq!(serde_json::to_value(&parsed).unwrap(), *example);
        }
        let original = spec("logloss", "mean");
        let identity = original.identity().unwrap();
        let reordered = serde_json::json!({
            "undefined": "fail_candidate", "aggregation": "mean", "metric": "logloss",
            "class_order": ["no", "yes"], "schema": FITNESS_VERSION
        });
        assert_eq!(
            ClassificationFitness::parse(&reordered.to_string())
                .unwrap()
                .identity()
                .unwrap(),
            identity
        );
        for changed in [spec("brier", "mean"), spec("logloss", "sum"), {
            let mut changed = original.clone();
            changed.class_order.swap(0, 1);
            changed
        }] {
            assert_ne!(changed.identity().unwrap(), identity);
        }
    }

    #[test]
    fn documented_input_example_has_the_golden_scores() {
        let document = schema();
        let example = &document["$defs"]["inputs"]["examples"][0];
        let order: [String; 2] = serde_json::from_value(example["class_order"].clone()).unwrap();
        let labels: Vec<String> = serde_json::from_value(example["labels"].clone()).unwrap();
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        let probabilities: Vec<[f64; 2]> =
            serde_json::from_value(example["probabilities"].clone()).unwrap();
        for (metric, expected) in [("logloss", 0.5917809035329042), ("brier", 0.21875)] {
            let score = spec(metric, "mean")
                .evaluate(Inputs {
                    class_order: [order[0].as_str(), order[1].as_str()],
                    labels: &labels,
                    probabilities: &probabilities,
                })
                .unwrap();
            assert!((score.value - expected).abs() <= 1e-12);
        }
    }
}
