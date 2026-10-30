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
pub const DEFAULT_CLASSIFICATION_THRESHOLD: f64 = 0.5;
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

/// Dataset-level binary metrics. `f1` accepts the optional top-level threshold; the other
/// threshold metrics use the v1 default of 0.5. Cost-matrix loss accepts the top-level matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Logloss,
    Brier,
    RocAuc,
    PrAuc,
    F1,
    BalancedAccuracy,
    CostMatrixLoss,
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

/// Core fields are required. Metric parameters are accepted only by their matching metric.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationFitness {
    pub schema: String,
    pub metric: Metric,
    pub class_order: [String; 2],
    pub aggregation: Aggregation,
    pub undefined: UndefinedPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_matrix: Option<[[f64; 2]; 2]>,
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
        ) && self.aggregation != Aggregation::Mean
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
            self.aggregation,
            self.threshold.unwrap_or(DEFAULT_CLASSIFICATION_THRESHOLD),
            self.cost_matrix,
        )?;
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
    fn value(
        self,
        labels: &[usize],
        probabilities: &[[f64; 2]],
        aggregation: Aggregation,
        threshold: f64,
        cost_matrix: Option<[[f64; 2]; 2]>,
    ) -> Result<f64> {
        match self {
            Self::Logloss | Self::Brier => {
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
                        _ => unreachable!(),
                    };
                }
                Ok(match aggregation {
                    Aggregation::Mean => total / labels.len() as f64,
                    Aggregation::Sum => total,
                })
            }
            Self::RocAuc => roc_auc(labels, probabilities),
            Self::PrAuc => average_precision(labels, probabilities),
            Self::F1 => f1_score(labels, probabilities, threshold),
            Self::BalancedAccuracy => balanced_accuracy(labels, probabilities),
            Self::CostMatrixLoss => cost_matrix_loss(
                labels,
                probabilities,
                cost_matrix.expect("validated cost matrix"),
            ),
        }
    }
}

fn sorted_scores(
    labels: &[usize],
    probabilities: &[[f64; 2]],
    descending: bool,
) -> Vec<(f64, usize)> {
    let mut rows: Vec<_> = labels
        .iter()
        .zip(probabilities)
        .map(|(&label, probability)| (probability[1], label))
        .collect();
    rows.sort_by(|left, right| {
        let ordering = left.0.total_cmp(&right.0);
        if descending {
            ordering.reverse()
        } else {
            ordering
        }
    });
    rows
}

fn roc_auc(labels: &[usize], probabilities: &[[f64; 2]]) -> Result<f64> {
    let positives = labels.iter().filter(|&&label| label == 1).count();
    let negatives = labels.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err(FitnessError::Undefined(
            "roc_auc requires both classes".into(),
        ));
    }
    let rows = sorted_scores(labels, probabilities, false);
    let mut negatives_before = 0usize;
    let mut concordant = 0.0;
    let mut index = 0;
    while index < rows.len() {
        let mut end = index + 1;
        while end < rows.len() && rows[end].0 == rows[index].0 {
            end += 1;
        }
        let group_positives = rows[index..end]
            .iter()
            .filter(|(_, label)| *label == 1)
            .count();
        let group_negatives = end - index - group_positives;
        concordant +=
            group_positives as f64 * (negatives_before as f64 + group_negatives as f64 / 2.0);
        negatives_before += group_negatives;
        index = end;
    }
    Ok(concordant / (positives as f64 * negatives as f64))
}

fn average_precision(labels: &[usize], probabilities: &[[f64; 2]]) -> Result<f64> {
    let positives = labels.iter().filter(|&&label| label == 1).count();
    if positives == 0 || positives == labels.len() {
        return Err(FitnessError::Undefined(
            "pr_auc requires both classes".into(),
        ));
    }
    let rows = sorted_scores(labels, probabilities, true);
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut total = 0.0;
    let mut index = 0;
    while index < rows.len() {
        let mut end = index + 1;
        while end < rows.len() && rows[end].0 == rows[index].0 {
            end += 1;
        }
        let new_positives = rows[index..end]
            .iter()
            .filter(|(_, label)| *label == 1)
            .count();
        true_positives += new_positives;
        false_positives += end - index - new_positives;
        total += (new_positives as f64 / positives as f64)
            * (true_positives as f64 / (true_positives + false_positives) as f64);
        index = end;
    }
    Ok(total)
}

fn f1_score(labels: &[usize], probabilities: &[[f64; 2]], threshold: f64) -> Result<f64> {
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut false_negatives = 0usize;
    for (&label, probability) in labels.iter().zip(probabilities) {
        let predicted_positive = probability[1] >= threshold;
        match (label == 1, predicted_positive) {
            (true, true) => true_positives += 1,
            (false, true) => false_positives += 1,
            (true, false) => false_negatives += 1,
            (false, false) => {}
        }
    }
    let denominator = 2 * true_positives + false_positives + false_negatives;
    if denominator == 0 {
        return Err(FitnessError::Undefined(
            "f1 is undefined for this fold".into(),
        ));
    }
    Ok((2 * true_positives) as f64 / denominator as f64)
}

fn balanced_accuracy(labels: &[usize], probabilities: &[[f64; 2]]) -> Result<f64> {
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut true_negatives = 0usize;
    let mut false_negatives = 0usize;
    for (&label, probability) in labels.iter().zip(probabilities) {
        match (
            label == 1,
            probability[1] >= DEFAULT_CLASSIFICATION_THRESHOLD,
        ) {
            (true, true) => true_positives += 1,
            (false, true) => false_positives += 1,
            (false, false) => true_negatives += 1,
            (true, false) => false_negatives += 1,
        }
    }
    if true_positives + false_negatives == 0 || true_negatives + false_positives == 0 {
        return Err(FitnessError::Undefined(
            "balanced_accuracy requires both classes".into(),
        ));
    }
    let sensitivity = true_positives as f64 / (true_positives + false_negatives) as f64;
    let specificity = true_negatives as f64 / (true_negatives + false_positives) as f64;
    Ok((sensitivity + specificity) / 2.0)
}

fn cost_matrix_loss(
    labels: &[usize],
    probabilities: &[[f64; 2]],
    cost_matrix: [[f64; 2]; 2],
) -> Result<f64> {
    let mut total = 0.0;
    for (&actual, probability) in labels.iter().zip(probabilities) {
        let predicted = usize::from(probability[1] >= DEFAULT_CLASSIFICATION_THRESHOLD);
        total += cost_matrix[actual][predicted];
    }
    Ok(total / labels.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(metric: &str, aggregation: &str) -> ClassificationFitness {
        spec_with_options(metric, aggregation, None, None)
    }

    fn spec_with_options(
        metric: &str,
        aggregation: &str,
        threshold: Option<f64>,
        cost_matrix: Option<[[f64; 2]; 2]>,
    ) -> ClassificationFitness {
        let mut value = serde_json::json!({
            "schema": "jope.classification-fitness/v1",
            "metric": metric,
            "class_order": ["no", "yes"],
            "aggregation": aggregation,
            "undefined": "fail_candidate"
        });
        if let Some(threshold) = threshold {
            value["threshold"] = serde_json::json!(threshold);
        }
        if let Some(cost_matrix) = cost_matrix {
            value["cost_matrix"] = serde_json::json!(cost_matrix);
        }
        ClassificationFitness::parse(&value.to_string()).unwrap()
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
    fn obj03_metrics_match_sklearn_goldens_with_score_ties() {
        let labels = ["no", "no", "yes", "yes", "no", "yes"];
        let probabilities = [
            [0.9, 0.1],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
        ];
        let cost_matrix = [[1.0, 4.0], [2.0, 0.0]];
        let goldens = [
            (spec("roc_auc", "mean"), 0.5),
            (spec("pr_auc", "mean"), 0.5),
            (spec("f1", "mean"), 4.0 / 7.0),
            (spec("balanced_accuracy", "mean"), 0.5),
            (
                spec_with_options("cost_matrix_loss", "mean", None, Some(cost_matrix)),
                11.0 / 6.0,
            ),
        ];
        for (objective, expected) in goldens {
            let score = objective.evaluate(inputs(&labels, &probabilities)).unwrap();
            assert!((score.value - expected).abs() <= 1e-12);
            assert_eq!(score.loss, score.value);
            assert_eq!(score.direction, Direction::Minimize);
        }

        // The threshold comparison is inclusive, including ties exactly at the threshold.
        let at_point_nine = spec_with_options("f1", "mean", Some(0.9), None)
            .evaluate(inputs(&labels, &probabilities))
            .unwrap();
        assert!((at_point_nine.value - 0.4).abs() <= 1e-12);
    }

    #[test]
    fn obj03_f1_default_threshold_is_canonical_in_objective_identity() {
        let defaulted = spec("f1", "mean");
        let explicit = spec_with_options("f1", "mean", Some(0.5), None);
        assert_eq!(defaulted.identity().unwrap(), explicit.identity().unwrap());
        assert_eq!(
            defaulted
                .evaluate(inputs(&["no", "yes"], &[[0.5, 0.5], [0.5, 0.5]]))
                .unwrap()
                .value,
            2.0 / 3.0
        );
    }

    #[test]
    fn obj03_single_class_folds_are_undefined_for_new_metrics() {
        let metrics = [
            spec("roc_auc", "mean"),
            spec("pr_auc", "mean"),
            spec("f1", "mean"),
            spec("balanced_accuracy", "mean"),
            spec_with_options(
                "cost_matrix_loss",
                "mean",
                None,
                Some([[1.0, 4.0], [2.0, 0.0]]),
            ),
        ];
        for objective in metrics {
            for single_class in [
                inputs(&["no"], &[[1.0, 0.0]]),
                inputs(&["yes"], &[[0.0, 1.0]]),
            ] {
                assert!(matches!(
                    objective.evaluate(single_class),
                    Err(FitnessError::Undefined(_))
                ));
            }
        }
    }

    #[test]
    fn obj03_metric_parameters_and_aggregation_are_validated() {
        for invalid in [
            serde_json::json!({"metric":"f1","threshold":-0.1}),
            serde_json::json!({"metric":"f1","threshold":1.1}),
            serde_json::json!({"metric":"roc_auc","threshold":0.5}),
            serde_json::json!({"metric":"brier","cost_matrix":[[0,1],[1,0]]}),
            serde_json::json!({"metric":"cost_matrix_loss"}),
            serde_json::json!({"metric":"cost_matrix_loss","cost_matrix":[[0,-1],[1,0]]}),
        ] {
            let mut value = serde_json::json!({
                "schema":"jope.classification-fitness/v1",
                "metric":"f1",
                "class_order":["no","yes"],
                "aggregation":"mean",
                "undefined":"fail_candidate"
            });
            for (key, item) in invalid.as_object().unwrap() {
                value[key] = item.clone();
            }
            assert!(matches!(
                ClassificationFitness::parse(&value.to_string()),
                Err(FitnessError::InvalidSpecification(_))
            ));
        }
        assert!(ClassificationFitness::parse(
            r#"{"schema":"jope.classification-fitness/v1","metric":"roc_auc","class_order":["no","yes"],"aggregation":"sum","undefined":"fail_candidate"}"#
        )
        .is_err());
        let mut non_finite = spec("f1", "mean");
        non_finite.threshold = Some(f64::NAN);
        assert!(matches!(
            non_finite.validate(),
            Err(FitnessError::InvalidSpecification(_))
        ));
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
            ("metric", serde_json::json!("unknown_metric")),
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
