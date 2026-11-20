//! Multiclass (`K >= 3`) classification objective adapter over `jope_runtime::fitness_multiclass`.
//!
//! Probability columns follow the explicit ordered `class_order` (`len >= 3`) bound at parse time.
//! Parsing, validation, SHA-256 identity and candidate scoring delegate directly to `jope_runtime`.

use anyhow::Result;
use jope_runtime::fitness::Score;
use jope_runtime::fitness_multiclass::{Inputs, MulticlassFitness};

/// A validated multiclass classification objective bound to an ordered slice of `K >= 3` class labels.
#[derive(Clone, Debug)]
pub struct MulticlassObjective {
    spec: MulticlassFitness,
}

impl MulticlassObjective {
    /// Parse either a built-in multiclass metric shorthand (`logloss`/`log_loss`, `brier`,
    /// `macro_f1`, `accuracy`, `balanced_accuracy`) or an inline `jope.multiclass-fitness/v1`
    /// JSON document whose `class_order` matches `class_order`.
    pub fn parse(specification: &str, class_order: Vec<String>) -> Result<Self> {
        Ok(Self {
            spec: MulticlassFitness::parse_for_classes(specification, class_order)?,
        })
    }

    /// Parse a standalone `jope.multiclass-fitness/v1` JSON document using its own `class_order`.
    pub fn parse_json(specification: &str) -> Result<Self> {
        Ok(Self {
            spec: MulticlassFitness::parse(specification)?,
        })
    }

    /// Canonical SHA-256 objective digest from `jope_runtime`.
    pub fn identity(&self) -> Result<String> {
        Ok(self.spec.identity()?)
    }

    /// Underlying validated `MulticlassFitness` specification.
    pub fn fitness(&self) -> &MulticlassFitness {
        &self.spec
    }

    /// Ordered class labels (`len >= 3`).
    pub fn class_order(&self) -> &[String] {
        &self.spec.class_order
    }

    /// Score held-out labels and `K`-column probability rows.
    pub fn evaluate(&self, labels: &[&str], probabilities: &[Vec<f64>]) -> Result<Score> {
        let refs: Vec<&str> = self.spec.class_order.iter().map(String::as_str).collect();
        let inputs = Inputs {
            class_order: &refs,
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

    fn classes() -> Vec<String> {
        vec![
            "setosa".to_owned(),
            "versicolor".to_owned(),
            "virginica".to_owned(),
        ]
    }

    #[test]
    fn shorthand_and_json_multiclass_objectives_match_runtime_scores_and_identities() {
        let labels = [
            "setosa",
            "versicolor",
            "virginica",
            "setosa",
            "versicolor",
            "virginica",
        ];
        let probs = vec![
            vec![0.85, 0.10, 0.05],
            vec![0.10, 0.80, 0.10],
            vec![0.05, 0.15, 0.80],
            vec![0.70, 0.20, 0.10],
            vec![0.15, 0.70, 0.15],
            vec![0.10, 0.20, 0.70],
        ];
        let class_refs = ["setosa", "versicolor", "virginica"];

        for (shorthand, expected_direction) in [
            ("logloss", Direction::Minimize),
            ("log_loss", Direction::Minimize),
            ("brier", Direction::Minimize),
            ("macro_f1", Direction::Maximize),
            ("accuracy", Direction::Maximize),
            ("balanced_accuracy", Direction::Maximize),
        ] {
            let objective = MulticlassObjective::parse(shorthand, classes()).unwrap();
            let score = objective.evaluate(&labels, &probs).unwrap();
            let runtime = MulticlassFitness::parse_for_classes(shorthand, classes())
                .unwrap()
                .evaluate(Inputs {
                    class_order: &class_refs,
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
            "schema": "jope.multiclass-fitness/v1",
            "metric": "cost_matrix_loss",
            "class_order": ["setosa", "versicolor", "virginica"],
            "aggregation": "mean",
            "undefined": "fail_candidate",
            "cost_matrix": [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [3.0, 1.0, 0.0]]
        }"#;
        let from_classes = MulticlassObjective::parse(cost_json, classes()).unwrap();
        let from_json = MulticlassObjective::parse_json(cost_json).unwrap();
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
            "schema": "jope.multiclass-fitness/v1",
            "metric": "macro_f1",
            "class_order": ["virginica", "versicolor", "setosa"],
            "aggregation": "mean",
            "undefined": "fail_candidate"
        }"#;
        assert!(MulticlassObjective::parse(json_spec, classes()).is_err());

        let objective = MulticlassObjective::parse("macro_f1", classes()).unwrap();
        let digest = objective.identity().unwrap();
        assert!(objective.evaluate(&[], &[]).is_err());
        assert!(objective
            .evaluate(
                &["unknown", "versicolor", "virginica"],
                &[
                    vec![0.4, 0.3, 0.3],
                    vec![0.2, 0.6, 0.2],
                    vec![0.1, 0.2, 0.7]
                ]
            )
            .is_err());
        assert!(objective
            .evaluate(
                &["setosa", "versicolor", "virginica"],
                &[
                    vec![0.3, 0.3, 0.3],
                    vec![0.2, 0.6, 0.2],
                    vec![0.1, 0.2, 0.7]
                ]
            )
            .is_err());
        assert_eq!(objective.identity().unwrap(), digest);
    }
}
