//! Binary classification objective adapter over `jope_runtime::fitness_classification`.
//!
//! Probability columns follow the explicit two-class `class_order` bound at parse time
//! (`class_order[1]` is the positive class). Parsing, validation, SHA-256 identity and
//! candidate scoring delegate directly to `jope_runtime`.

use anyhow::Result;
use jope_runtime::fitness::Score;
use jope_runtime::fitness_classification::{ClassificationFitness, Inputs};

/// A validated binary classification objective bound to an ordered pair of class labels.
#[derive(Clone, Debug)]
pub struct BinaryObjective {
    spec: ClassificationFitness,
}

impl BinaryObjective {
    /// Parse either a built-in binary metric shorthand (`roc_auc`, `logloss`, `brier`, `pr_auc`,
    /// `balanced_accuracy`, `f1`, `f1:threshold=0.35`) or an inline `jope.classification-fitness/v1`
    /// JSON document whose `class_order` matches `class_order`.
    pub fn parse(specification: &str, class_order: [String; 2]) -> Result<Self> {
        Ok(Self {
            spec: ClassificationFitness::parse_for_classes(specification, class_order)?,
        })
    }

    /// Parse a standalone `jope.classification-fitness/v1` JSON document using its own `class_order`.
    pub fn parse_json(specification: &str) -> Result<Self> {
        Ok(Self {
            spec: ClassificationFitness::parse(specification)?,
        })
    }

    /// Canonical SHA-256 objective digest from `jope_runtime`.
    pub fn identity(&self) -> Result<String> {
        Ok(self.spec.identity()?)
    }

    /// Underlying validated `ClassificationFitness` specification.
    pub fn fitness(&self) -> &ClassificationFitness {
        &self.spec
    }

    /// Ordered class labels `[negative_class, positive_class]`.
    pub fn class_order(&self) -> [&str; 2] {
        [
            self.spec.class_order[0].as_str(),
            self.spec.class_order[1].as_str(),
        ]
    }

    /// Score held-out labels and `[p_class0, p_class1]` probability rows.
    pub fn evaluate(&self, labels: &[&str], probabilities: &[[f64; 2]]) -> Result<Score> {
        let inputs = Inputs {
            class_order: self.class_order(),
            labels,
            probabilities,
        };
        Ok(self.spec.evaluate(inputs)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_runtime::fitness::Direction;

    fn classes() -> [String; 2] {
        ["neg".to_owned(), "pos".to_owned()]
    }

    #[test]
    fn shorthand_and_json_objectives_match_runtime_scores_and_identities() {
        let labels = ["neg", "pos", "neg", "pos"];
        let probs = [[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7]];

        for (shorthand, expected_direction) in [
            ("roc_auc", Direction::Maximize),
            ("auroc", Direction::Maximize),
            ("logloss", Direction::Minimize),
            ("brier", Direction::Minimize),
            ("pr_auc", Direction::Maximize),
            ("balanced_accuracy", Direction::Maximize),
            ("f1:threshold=0.35", Direction::Maximize),
        ] {
            let objective = BinaryObjective::parse(shorthand, classes()).unwrap();
            let score = objective.evaluate(&labels, &probs).unwrap();
            let runtime = ClassificationFitness::parse_for_classes(shorthand, classes())
                .unwrap()
                .evaluate(Inputs {
                    class_order: ["neg", "pos"],
                    labels: &labels,
                    probabilities: &probs,
                })
                .unwrap();
            assert_eq!(score.direction, expected_direction, "{shorthand}");
            assert_eq!(score.value.to_bits(), runtime.value.to_bits());
            assert_eq!(score.loss.to_bits(), runtime.loss.to_bits());
            assert_eq!(score.fitness_sha256, runtime.fitness_sha256);
            assert_eq!(objective.identity().unwrap(), runtime.fitness_sha256);
        }

        let cost_json = r#"{
            "schema": "jope.classification-fitness/v1",
            "metric": "cost_matrix_loss",
            "class_order": ["neg", "pos"],
            "aggregation": "mean",
            "undefined": "fail_candidate",
            "cost_matrix": [[0.0, 1.0], [5.0, 0.0]]
        }"#;
        let from_classes = BinaryObjective::parse(cost_json, classes()).unwrap();
        let from_json = BinaryObjective::parse_json(cost_json).unwrap();
        assert_eq!(
            from_classes.identity().unwrap(),
            from_json.identity().unwrap()
        );
        let cost_score = from_classes.evaluate(&labels, &probs).unwrap();
        assert_eq!(cost_score.direction, Direction::Minimize);
        assert_eq!(cost_score.value, 0.0);
    }

    #[test]
    fn class_order_mismatch_and_invalid_candidates_are_refused() {
        let json_spec = r#"{
            "schema": "jope.classification-fitness/v1",
            "metric": "roc_auc",
            "class_order": ["pos", "neg"],
            "aggregation": "mean",
            "undefined": "fail_candidate"
        }"#;
        assert!(BinaryObjective::parse(json_spec, classes()).is_err());

        let objective = BinaryObjective::parse("roc_auc", classes()).unwrap();
        let digest = objective.identity().unwrap();
        assert!(objective.evaluate(&[], &[]).is_err());
        assert!(objective
            .evaluate(&["unknown", "pos"], &[[0.5, 0.5], [0.2, 0.8]])
            .is_err());
        assert!(objective
            .evaluate(&["neg", "pos"], &[[0.3, 0.3], [0.2, 0.8]])
            .is_err());
        assert!(objective
            .evaluate(&["neg", "neg"], &[[0.9, 0.1], [0.8, 0.2]])
            .is_err());
        assert_eq!(objective.identity().unwrap(), digest);
    }
}
