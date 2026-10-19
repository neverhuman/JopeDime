//! Additive, versioned out-of-fold prediction bundle (SC-DEC-184 JA-03).
//!
//! `OofEvidence` in `search_plan.rs` holds measured fits and hashes, not predictions. This
//! bundle holds the row-aligned predictions themselves. Only the bake-off builds one, from
//! predictions of models that never saw the rows they predict. A scalar CV receipt cannot be
//! turned into a bundle, and neither can a full-train model's fitted values.
use crate::evaluation_score::Predictions;
use anyhow::{bail, ensure, Result};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const OOF_SCHEMA: &str = "jope.oof-prediction-bundle/v1";

/// Row-aligned out-of-fold predictions for one candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OofPredictionBundle {
    pub schema: String,
    pub task: Task,
    pub candidate_id: String,
    /// Identity of the request this bundle was produced under.
    pub request_sha256: String,
    /// Identity of the fitted artifacts that produced the predictions, in fold order.
    pub fitted_artifact_sha256: Vec<String>,
    /// Ordered row identifiers. Every row appears once.
    pub row_ids: Vec<String>,
    /// Fold of each row, aligned with `row_ids`.
    pub folds: Vec<u32>,
    /// Group of each row, aligned with `row_ids`. Empty when no grouping applies.
    pub group_ids: Vec<String>,
    /// Frozen class order. Empty for regression.
    pub classes: Vec<String>,
    /// Predictions aligned with `row_ids`.
    pub predictions: Predictions,
    /// Folds whose fit failed. Their rows are absent from the bundle, never filled in.
    pub failed_folds: Vec<u32>,
}

impl OofPredictionBundle {
    /// Refuses every misalignment, duplicate row, class-order mismatch, and invalid probability.
    /// Probabilities are never renormalised.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == OOF_SCHEMA,
            "OOF bundle schema {:?}",
            self.schema
        );
        ensure!(
            !self.candidate_id.is_empty(),
            "OOF bundle has no candidate id"
        );
        ensure!(
            is_sha256(&self.request_sha256),
            "OOF bundle request identity is not a sha256"
        );
        ensure!(
            self.fitted_artifact_sha256.iter().all(|h| is_sha256(h)),
            "OOF bundle fitted artifact identity is not a sha256"
        );
        let n = self.row_ids.len();
        ensure!(n > 0, "OOF bundle is empty");
        let unique: BTreeSet<_> = self.row_ids.iter().collect();
        ensure!(unique.len() == n, "OOF bundle repeats a row id");
        ensure!(self.folds.len() == n, "fold assignment is not row-aligned");
        ensure!(
            self.group_ids.is_empty() || self.group_ids.len() == n,
            "group identity is not row-aligned"
        );
        let failed: BTreeSet<u32> = self.failed_folds.iter().copied().collect();
        ensure!(
            self.folds.iter().all(|fold| !failed.contains(fold)),
            "a row belongs to a failed fold and must be absent"
        );
        match (self.task, &self.predictions) {
            (Task::Regression, Predictions::Values(values)) => {
                ensure!(self.classes.is_empty(), "regression bundle carries classes");
                ensure!(values.len() == n, "prediction rows are not row-aligned");
                ensure!(values.iter().all(|v| v.is_finite()), "nonfinite prediction");
            }
            (Task::Binary, Predictions::Probabilities { classes, values }) => {
                ensure!(classes == &self.classes, "prediction class order differs");
                ensure!(self.classes.len() == 2, "binary bundle needs two classes");
                ensure!(values.len() == n, "prediction rows are not row-aligned");
                crate::evaluation_score::validate_probabilities(values, 2)?;
            }
            (Task::Multiclass, Predictions::Probabilities { classes, values }) => {
                ensure!(classes == &self.classes, "prediction class order differs");
                ensure!(
                    self.classes.len() >= 3,
                    "multiclass bundle needs three classes"
                );
                ensure!(values.len() == n, "prediction rows are not row-aligned");
                crate::evaluation_score::validate_probabilities(values, self.classes.len())?;
            }
            (task, _) => bail!("predictions do not match task {task:?}"),
        }
        Ok(())
    }

    /// Canonical identity of the bundle. Used as the bundle's own sha256 in receipts.
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }
}

fn is_sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regression(values: Vec<f64>, folds: Vec<u32>) -> OofPredictionBundle {
        let n = values.len();
        OofPredictionBundle {
            schema: OOF_SCHEMA.into(),
            task: Task::Regression,
            candidate_id: "rust.ridge".into(),
            request_sha256: "a".repeat(64),
            fitted_artifact_sha256: vec!["b".repeat(64); 2],
            row_ids: (0..n).map(|i| format!("row-{i}")).collect(),
            folds,
            group_ids: vec![],
            classes: vec![],
            predictions: Predictions::Values(values),
            failed_folds: vec![],
        }
    }

    #[test]
    fn a_row_aligned_regression_bundle_validates_and_has_a_stable_identity() {
        let b = regression(vec![1.0, 2.0, 3.0], vec![0, 1, 0]);
        b.validate().unwrap();
        assert_eq!(b.identity().unwrap(), b.clone().identity().unwrap());
    }

    #[test]
    fn misaligned_folds_and_duplicate_rows_are_refused() {
        let mut short = regression(vec![1.0, 2.0], vec![0, 1]);
        short.folds.pop();
        assert!(short.validate().is_err());
        let mut dup = regression(vec![1.0, 2.0], vec![0, 1]);
        dup.row_ids[1] = dup.row_ids[0].clone();
        assert!(dup.validate().is_err());
    }

    #[test]
    fn a_row_from_a_failed_fold_is_refused_rather_than_filled() {
        let mut b = regression(vec![1.0, 2.0], vec![0, 1]);
        b.failed_folds = vec![1];
        assert!(b.validate().is_err());
    }

    #[test]
    fn invalid_probabilities_are_refused_not_renormalised() {
        let b = OofPredictionBundle {
            schema: OOF_SCHEMA.into(),
            task: Task::Binary,
            candidate_id: "rust.logit_ridge".into(),
            request_sha256: "a".repeat(64),
            fitted_artifact_sha256: vec!["b".repeat(64)],
            row_ids: vec!["r0".into()],
            folds: vec![0],
            group_ids: vec![],
            classes: vec!["no".into(), "yes".into()],
            predictions: Predictions::Probabilities {
                classes: vec!["no".into(), "yes".into()],
                values: vec![vec![0.7, 0.7]],
            },
            failed_folds: vec![],
        };
        assert!(b.validate().is_err());
    }
}
