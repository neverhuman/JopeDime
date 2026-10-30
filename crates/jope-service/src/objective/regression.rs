//! Regression objective adapter; parsing, scoring and identity use the runtime unchanged.
//!
//! The future `ProblemDigest` owner must include [`RegressionObjective::identity`] in the
//! problem identity. This module defines neither a shared objective trait nor a digest builder.

use anyhow::Result;
use jope_runtime::fitness::{Columns, Fitness, Score};

/// A validated regression fitness specification, including the bounded JSON grammar.
#[derive(Clone, Debug)]
pub struct RegressionObjective {
    fitness: Fitness,
}

impl RegressionObjective {
    /// Resolve a runtime catalog name, parameterized shorthand or inline JSON specification.
    /// File access (including the CLI's `@path` syntax) belongs to the caller.
    pub fn parse(specification: &str) -> Result<Self> {
        Ok(Self {
            fitness: Fitness::parse(specification)?,
        })
    }

    /// The runtime's exact objective identity; no additional hashing or normalization occurs.
    pub fn identity(&self) -> Result<String> {
        self.fitness.identity()
    }

    /// Score targets and predictions without auxiliary columns, retaining runtime errors.
    pub fn evaluate(&self, targets: &[f64], predictions: &[f64]) -> Result<Score> {
        self.fitness.evaluate(targets, predictions)
    }

    /// Score with aligned numeric auxiliary columns required by a custom objective.
    /// The returned runtime score includes the value, minimization loss, direction and identity.
    pub fn evaluate_with(
        &self,
        targets: &[f64],
        predictions: &[f64],
        columns: &Columns,
    ) -> Result<Score> {
        self.fitness.evaluate_with(targets, predictions, columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_runtime::fitness::Direction;

    // Targets/predictions from the committed examples/objectives/evaluation_request.json.
    const TARGETS: [f64; 3] = [10.0, 20.0, 30.0];
    const PREDICTIONS: [f64; 3] = [9.0, 22.0, 28.0];
    const GRID_COST: &str = include_str!("../../../../examples/objectives/grid_underforecast.json");

    #[test]
    fn builtins_and_committed_custom_fixture_retain_runtime_scores_and_identity() {
        for (specification, expected) in [
            ("rmse", 3.0_f64.sqrt()),
            ("mae", 5.0 / 3.0),
            (GRID_COST, 17.0 / 3.0),
        ] {
            let objective = RegressionObjective::parse(specification).unwrap();
            let score = objective.evaluate(&TARGETS, &PREDICTIONS).unwrap();
            let runtime = Fitness::parse(specification)
                .unwrap()
                .evaluate(&TARGETS, &PREDICTIONS)
                .unwrap();
            assert!((score.value - expected).abs() < 1e-12);
            assert_eq!(score.value.to_bits(), runtime.value.to_bits());
            assert_eq!(score.loss.to_bits(), runtime.loss.to_bits());
            assert_eq!(score.direction, Direction::Minimize);
            assert_eq!(score.fitness_sha256, runtime.fitness_sha256);
            assert_eq!(objective.identity().unwrap(), runtime.fitness_sha256);
        }
    }

    #[test]
    fn maximizing_custom_fixture_retains_the_runtime_minimization_loss() {
        let specification = include_str!("../../../../examples/objectives/inventory_profit.json");
        let objective = RegressionObjective::parse(specification).unwrap();
        let score = objective.evaluate(&TARGETS, &PREDICTIONS).unwrap();
        assert_eq!(score.direction, Direction::Maximize);
        assert!((score.value - 328.0 / 3.0).abs() < 1e-12);
        assert_eq!(score.loss, -score.value);
        assert_eq!(score.fitness_sha256, objective.identity().unwrap());
    }

    #[test]
    fn custom_auxiliary_columns_remain_required_and_aligned() {
        let specification =
            include_str!("../../../../examples/objectives/newsvendor_margin_columns.json");
        let objective = RegressionObjective::parse(specification).unwrap();
        assert!(objective.evaluate(&TARGETS, &PREDICTIONS).is_err());
        let mut columns = Columns::from([
            ("unit_margin".into(), vec![5.0; 3]),
            ("holding_cost".into(), vec![1.0; 3]),
        ]);
        let score = objective
            .evaluate_with(&TARGETS, &PREDICTIONS, &columns)
            .unwrap();
        assert!((score.value - 17.0 / 3.0).abs() < 1e-12);
        assert_eq!(score.fitness_sha256, objective.identity().unwrap());
        columns.get_mut("unit_margin").unwrap().pop();
        assert!(objective
            .evaluate_with(&TARGETS, &PREDICTIONS, &columns)
            .is_err());
        columns.get_mut("unit_margin").unwrap().push(f64::NAN);
        assert!(objective
            .evaluate_with(&TARGETS, &PREDICTIONS, &columns)
            .is_err());
    }

    #[test]
    fn canonical_alias_identity_is_unchanged_and_parameters_change_it() {
        let short = RegressionObjective::parse("rmse").unwrap();
        let json = RegressionObjective::parse(r#"{ "metric": "rmse" }"#).unwrap();
        assert_eq!(short.identity().unwrap(), json.identity().unwrap());
        assert_eq!(
            short.identity().unwrap(),
            Fitness::Rmse {}.identity().unwrap()
        );
        assert_ne!(
            RegressionObjective::parse("pinball:quantile=0.1")
                .unwrap()
                .identity()
                .unwrap(),
            RegressionObjective::parse("pinball:quantile=0.9")
                .unwrap()
                .identity()
                .unwrap()
        );
    }

    #[test]
    fn invalid_specs_and_undefined_candidates_fail_without_poisoning_the_objective() {
        for specification in [
            "unknown",
            r#"{"metric":"rmse","unknown":1}"#,
            "pinball:quantile=2",
            r#"{"metric":"custom"}"#,
        ] {
            assert!(RegressionObjective::parse(specification).is_err());
        }
        let objective = RegressionObjective::parse("rmse").unwrap();
        let identity = objective.identity().unwrap();
        for (targets, predictions) in [
            (&[][..], &[][..]),
            (&TARGETS[..], &[1.0][..]),
            (&[f64::NAN][..], &[1.0][..]),
            (&[1.0][..], &[f64::INFINITY][..]),
        ] {
            assert!(objective.evaluate(targets, predictions).is_err());
        }
        assert!(RegressionObjective::parse("r2")
            .unwrap()
            .evaluate(&[1.0, 1.0], &[1.0, 1.0])
            .is_err());
        assert!(objective
            .evaluate(&TARGETS, &PREDICTIONS)
            .unwrap()
            .value
            .is_finite());
        assert_eq!(objective.identity().unwrap(), identity);
    }
}
