//! Typed predictor boundary for the sealed test look (SC-DEC-184 JA-04 test-score extension).
//!
//! `test_score.rs` scores regression genomes through `feat_math::fit` and publishes to an on-disk
//! registry. This module is the typed boundary that a persisted fitted ensemble or classifier
//! would score through. It is pure: it holds an in-memory look ledger and calls no registry.
//!
//! Rules mirrored from the registry path:
//! - a predictor may take its test look once per dataset; a repeat is refused;
//! - a different predictor is allowed, and the earlier predictors on that dataset are listed
//!   as `earlier_looks`, so the choice among them is visible;
//! - a look is consumed when the test rows are read, even if scoring then fails, so a failed
//!   score cannot be retried until it succeeds.
//!
//! Scoring goes through `evaluation_score::score_predictions`, which refuses invalid
//! probabilities and class-order mismatches. Nothing here re-normalises a prediction.
use crate::evaluation_score::{score_predictions, Predictions};
use crate::experiment::Metric;
use anyhow::{bail, Result};
use jope_datasets::catalog::Task;
use serde::Serialize;
use std::collections::BTreeMap;

/// A predictor that can produce test-row predictions in the frozen form.
pub trait TypedPredictor {
    /// Stable identity of the fitted predictor (for example its artifact or ensemble hash).
    fn identity(&self) -> &str;
    /// Predictions for `rows`, aligned with the rows given.
    fn predict(&self, rows: &[usize]) -> Result<Predictions>;
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TypedTestScore {
    pub dataset_hash: String,
    pub predictor_identity: String,
    pub metric: Metric,
    pub loss: f64,
    /// Predictors that already took their look on this dataset, in the order they did.
    pub earlier_looks: Vec<String>,
}

/// The in-memory record of test looks. Process-local by design; durable records are the
/// registry's job.
#[derive(Clone, Debug, Default)]
pub struct LookLedger {
    looks: BTreeMap<String, Vec<String>>,
}

impl LookLedger {
    pub fn new() -> Self {
        Self::default()
    }

    fn taken(&self, dataset_hash: &str) -> &[String] {
        self.looks
            .get(dataset_hash)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

/// Takes the test look for `predictor` on `dataset_hash`. Refuses a repeat look by the same
/// predictor. The look is consumed before any row is read.
pub fn take_test_look(
    ledger: &mut LookLedger,
    dataset_hash: &str,
    task: Task,
    classes: &[String],
    test_labels: &[String],
    test_rows: &[usize],
    predictor: &dyn TypedPredictor,
) -> Result<TypedTestScore> {
    if ledger
        .taken(dataset_hash)
        .iter()
        .any(|id| id == predictor.identity())
    {
        bail!(
            "repeat test look refused: predictor {} already took its look on dataset {dataset_hash}",
            predictor.identity()
        );
    }
    let earlier_looks = ledger.taken(dataset_hash).to_vec();
    ledger
        .looks
        .entry(dataset_hash.to_owned())
        .or_default()
        .push(predictor.identity().to_owned());
    let labels: Vec<String> = test_rows.iter().map(|&i| test_labels[i].clone()).collect();
    let predictions = predictor.predict(test_rows)?;
    let score = score_predictions(task, classes, &labels, &predictions)?;
    Ok(TypedTestScore {
        dataset_hash: dataset_hash.to_owned(),
        predictor_identity: predictor.identity().to_owned(),
        metric: score.metric,
        loss: score.loss,
        earlier_looks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed {
        id: &'static str,
        out: Predictions,
    }

    impl TypedPredictor for Fixed {
        fn identity(&self) -> &str {
            self.id
        }
        fn predict(&self, _rows: &[usize]) -> Result<Predictions> {
            Ok(self.out.clone())
        }
    }

    fn regression_labels() -> Vec<String> {
        vec!["1".into(), "2".into(), "3".into()]
    }

    #[test]
    fn a_predictor_takes_its_test_look_once_per_dataset() {
        let mut ledger = LookLedger::new();
        let p = Fixed {
            id: "artifact-a",
            out: Predictions::Values(vec![1.0, 2.0, 3.0]),
        };
        let first = take_test_look(
            &mut ledger,
            "ds",
            Task::Regression,
            &[],
            &regression_labels(),
            &[0, 1, 2],
            &p,
        )
        .unwrap();
        assert_eq!(first.loss, 0.0);
        assert!(first.earlier_looks.is_empty());
        let repeat = take_test_look(
            &mut ledger,
            "ds",
            Task::Regression,
            &[],
            &regression_labels(),
            &[0, 1, 2],
            &p,
        );
        assert!(repeat.is_err());
    }

    #[test]
    fn a_second_predictor_is_allowed_and_lists_the_first_as_an_earlier_look() {
        let mut ledger = LookLedger::new();
        let a = Fixed {
            id: "artifact-a",
            out: Predictions::Values(vec![1.0, 2.0, 3.0]),
        };
        let b = Fixed {
            id: "artifact-b",
            out: Predictions::Values(vec![2.0, 2.0, 2.0]),
        };
        take_test_look(
            &mut ledger,
            "ds",
            Task::Regression,
            &[],
            &regression_labels(),
            &[0, 1, 2],
            &a,
        )
        .unwrap();
        let second = take_test_look(
            &mut ledger,
            "ds",
            Task::Regression,
            &[],
            &regression_labels(),
            &[0, 1, 2],
            &b,
        )
        .unwrap();
        assert_eq!(second.earlier_looks, vec!["artifact-a".to_string()]);
    }

    #[test]
    fn a_look_on_another_dataset_is_independent() {
        let mut ledger = LookLedger::new();
        let a = Fixed {
            id: "artifact-a",
            out: Predictions::Values(vec![1.0, 2.0, 3.0]),
        };
        take_test_look(
            &mut ledger,
            "ds-1",
            Task::Regression,
            &[],
            &regression_labels(),
            &[0, 1, 2],
            &a,
        )
        .unwrap();
        assert!(take_test_look(
            &mut ledger,
            "ds-2",
            Task::Regression,
            &[],
            &regression_labels(),
            &[0, 1, 2],
            &a
        )
        .is_ok());
    }

    #[test]
    fn a_failed_score_still_consumes_the_look() {
        let mut ledger = LookLedger::new();
        let bad_shape = Fixed {
            id: "artifact-bad",
            out: Predictions::Values(vec![1.0]),
        };
        assert!(take_test_look(
            &mut ledger,
            "ds",
            Task::Regression,
            &[],
            &regression_labels(),
            &[0, 1, 2],
            &bad_shape
        )
        .is_err());
        let retry = take_test_look(
            &mut ledger,
            "ds",
            Task::Regression,
            &[],
            &regression_labels(),
            &[0, 1, 2],
            &bad_shape,
        );
        assert!(retry.is_err(), "the failed look must not be retryable");
    }

    #[test]
    fn classification_goes_through_the_validated_scorer_and_refuses_invalid_probabilities() {
        let classes = vec!["no".to_string(), "yes".to_string()];
        let labels = vec!["no".to_string(), "yes".to_string()];
        let mut ledger = LookLedger::new();
        let good = Fixed {
            id: "clf-good",
            out: Predictions::Probabilities {
                classes: classes.clone(),
                values: vec![vec![0.9, 0.1], vec![0.2, 0.8]],
            },
        };
        let score = take_test_look(
            &mut ledger,
            "ds",
            Task::Binary,
            &classes,
            &labels,
            &[0, 1],
            &good,
        )
        .unwrap();
        assert!(score.loss.is_finite());
        let not_normalised = Fixed {
            id: "clf-bad",
            out: Predictions::Probabilities {
                classes: classes.clone(),
                values: vec![vec![0.9, 0.9], vec![0.2, 0.8]],
            },
        };
        assert!(take_test_look(
            &mut ledger,
            "ds",
            Task::Binary,
            &classes,
            &labels,
            &[0, 1],
            &not_normalised
        )
        .is_err());
    }
}
