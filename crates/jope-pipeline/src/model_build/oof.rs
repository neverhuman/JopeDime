//! Additive, versioned out-of-fold prediction bundle (SC-DEC-184 JA-03).
//!
//! `OofEvidence` in `search_plan.rs` holds measured fits and hashes, not predictions. This
//! bundle holds the row-aligned predictions themselves. Only the bake-off builds one, from
//! predictions of models that never saw the rows they predict. A scalar CV receipt cannot be
//! turned into a bundle, and neither can a full-train model's fitted values.
//!
//! `validate` checks the bundle's own shape. `validate_against` additionally binds it to the
//! admitted plan and to one verified fold record per fold. A caller-supplied SHA-shaped string
//! is not proof of physical fitting or of OOF production; the fold records carry that claim.
use crate::evaluation_score::Predictions;
use anyhow::{bail, ensure, Result};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

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
    /// One fitted-artifact identity per fold, in ascending fold order.
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

/// Terminal state of one fold's fit, as recorded by the worker that ran it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoldStatus {
    Verified,
    Failed,
    /// No terminal evidence exists. Unknown is never treated as a pass.
    Unknown,
}

/// One fold's record: the request it ran under, its artifact, and exactly which rows it fitted on
/// and which it scored. Its fields are private, and it has no public constructor. Admission does
/// not take it directly. It takes a `VerifiedFold`, which only a verified fit can produce.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldRecord {
    fold: u32,
    request_sha256: String,
    fitted_artifact_sha256: String,
    fitting_row_ids: Vec<String>,
    scoring_row_ids: Vec<String>,
    status: FoldStatus,
}

/// A fold record from a fit that returned verified evidence. It has no serde implementation and no
/// public constructor, so an externally read record cannot be turned into one. `AdmittedBundle`
/// admits only these.
///
/// ```compile_fail
/// use jope_pipeline::model_build::VerifiedFold;
/// fn needs_deserialize<'de, T: serde::Deserialize<'de>>() {}
/// needs_deserialize::<VerifiedFold>();
/// ```
///
/// ```compile_fail
/// use jope_pipeline::model_build::VerifiedFold;
/// let _ = VerifiedFold { record: unimplemented!() };
/// ```
///
/// ```compile_fail
/// use jope_pipeline::model_build::VerifiedFold;
/// let _ = VerifiedFold::from_verified(unimplemented!());
/// ```
///
/// ```compile_fail
/// use jope_pipeline::model_build::VerifiedFold;
/// fn needs_serialize<T: serde::Serialize>() {}
/// needs_serialize::<VerifiedFold>();
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedFold {
    record: FoldRecord,
}

impl VerifiedFold {
    /// Wraps a record only when its status is `Verified`. Every other status is refused here.
    pub(super) fn from_verified(record: FoldRecord) -> Option<Self> {
        (record.status == FoldStatus::Verified).then_some(Self { record })
    }

    pub fn record(&self) -> &FoldRecord {
        &self.record
    }
}

impl FoldRecord {
    pub(crate) fn new(
        fold: u32,
        request_sha256: String,
        fitted_artifact_sha256: String,
        fitting_row_ids: Vec<String>,
        scoring_row_ids: Vec<String>,
        status: FoldStatus,
    ) -> Self {
        Self {
            fold,
            request_sha256,
            fitted_artifact_sha256,
            fitting_row_ids,
            scoring_row_ids,
            status,
        }
    }

    pub fn fold(&self) -> u32 {
        self.fold
    }

    pub fn request_sha256(&self) -> &str {
        &self.request_sha256
    }

    pub fn fitted_artifact_sha256(&self) -> &str {
        &self.fitted_artifact_sha256
    }

    pub fn fitting_row_ids(&self) -> &[String] {
        &self.fitting_row_ids
    }

    pub fn scoring_row_ids(&self) -> &[String] {
        &self.scoring_row_ids
    }

    pub fn status(&self) -> FoldStatus {
        self.status
    }
}

/// The admitted plan that a bundle must reproduce exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedPlan {
    pub request_sha256: String,
    pub row_ids: Vec<String>,
    pub folds: Vec<u32>,
    pub group_ids: Vec<String>,
    pub classes: Vec<String>,
}

impl OofPredictionBundle {
    /// Refuses every misalignment, duplicate row, class-order mismatch, crossed group, empty or
    /// miscounted artifact list, and invalid probability. Probabilities are never renormalised.
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
        let n = self.row_ids.len();
        ensure!(n > 0, "OOF bundle is empty");
        let unique: BTreeSet<_> = self.row_ids.iter().collect();
        ensure!(unique.len() == n, "OOF bundle repeats a row id");
        ensure!(self.folds.len() == n, "fold assignment is not row-aligned");
        ensure!(
            self.group_ids.is_empty() || self.group_ids.len() == n,
            "group identity is not row-aligned"
        );
        let distinct_folds: BTreeSet<u32> = self.folds.iter().copied().collect();
        ensure!(
            !self.fitted_artifact_sha256.is_empty(),
            "OOF bundle has no fitted artifact identities"
        );
        ensure!(
            self.fitted_artifact_sha256.len() == distinct_folds.len(),
            "fitted artifact count {} does not match fold count {}",
            self.fitted_artifact_sha256.len(),
            distinct_folds.len()
        );
        ensure!(
            self.fitted_artifact_sha256.iter().all(|h| is_sha256(h)),
            "OOF bundle fitted artifact identity is not a sha256"
        );
        if !self.group_ids.is_empty() {
            let mut group_folds: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
            for (group, fold) in self.group_ids.iter().zip(&self.folds) {
                group_folds.entry(group.as_str()).or_default().insert(*fold);
            }
            ensure!(
                group_folds.values().all(|folds| folds.len() == 1),
                "a group spans more than one fold (crossed-group leakage)"
            );
        }
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

    /// Binds the bundle to the admitted plan and to one verified record per fold. Refuses unknown
    /// or failed folds, missing or extra folds, and any row whose fitting or scoring membership
    /// differs from the plan.
    pub fn validate_against(&self, expected: &ExpectedPlan, records: &[FoldRecord]) -> Result<()> {
        self.validate()?;
        ensure!(
            self.request_sha256 == expected.request_sha256,
            "bundle request differs from the admitted plan"
        );
        ensure!(
            self.row_ids == expected.row_ids,
            "bundle rows differ from the admitted plan"
        );
        ensure!(
            self.folds == expected.folds,
            "bundle folds differ from the admitted plan"
        );
        ensure!(
            self.group_ids == expected.group_ids,
            "bundle groups differ from the admitted plan"
        );
        ensure!(
            self.classes == expected.classes,
            "bundle class order differs from the admitted plan"
        );
        let folds: Vec<u32> = self
            .folds
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        ensure!(
            records.len() == folds.len(),
            "expected {} fold records, got {}",
            folds.len(),
            records.len()
        );
        let by_fold: BTreeMap<u32, &FoldRecord> = records.iter().map(|r| (r.fold, r)).collect();
        ensure!(by_fold.len() == records.len(), "duplicate fold record");
        for (index, fold) in folds.iter().enumerate() {
            let record = by_fold
                .get(fold)
                .ok_or_else(|| anyhow::anyhow!("no fold record for fold {fold}"))?;
            ensure!(
                record.status == FoldStatus::Verified,
                "fold {fold} is {:?}; only verified folds may enter a bundle",
                record.status
            );
            ensure!(
                record.request_sha256 == self.request_sha256,
                "fold {fold} ran under a different request"
            );
            ensure!(
                record.fitted_artifact_sha256 == self.fitted_artifact_sha256[index],
                "fold {fold} artifact differs from the bundle's artifact for that fold"
            );
            let scoring: Vec<&String> = self
                .row_ids
                .iter()
                .zip(&self.folds)
                .filter(|(_, f)| *f == fold)
                .map(|(id, _)| id)
                .collect();
            let fitting: Vec<&String> = self
                .row_ids
                .iter()
                .zip(&self.folds)
                .filter(|(_, f)| *f != fold)
                .map(|(id, _)| id)
                .collect();
            ensure!(
                record.scoring_row_ids.iter().collect::<Vec<_>>() == scoring,
                "fold {fold} scored rows differ from its fold membership"
            );
            ensure!(
                record.fitting_row_ids.iter().collect::<Vec<_>>() == fitting,
                "fold {fold} fitted on rows other than the other folds"
            );
        }
        Ok(())
    }

    /// Canonical identity of the bundle. Used as the bundle's own sha256 in receipts.
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }
}

/// A bundle that passed `validate_against` its admitted plan and verified fold records. The only
/// production constructor is `admit`, so downstream code cannot consume an unadmitted bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedBundle {
    bundle: OofPredictionBundle,
}

impl AdmittedBundle {
    pub fn admit(
        bundle: OofPredictionBundle,
        plan: &ExpectedPlan,
        records: &[VerifiedFold],
    ) -> Result<Self> {
        let plain: Vec<FoldRecord> = records.iter().map(|v| v.record().clone()).collect();
        bundle.validate_against(plan, &plain)?;
        Ok(Self { bundle })
    }

    pub fn bundle(&self) -> &OofPredictionBundle {
        &self.bundle
    }
}

/// Test-only admission from the bundle's own fold structure. Production admission must use
/// records produced by verified fits; this helper exists so unit tests can exercise consumers.
#[cfg(test)]
pub(crate) fn admit_for_test(bundle: OofPredictionBundle) -> Result<AdmittedBundle> {
    let folds: Vec<u32> = bundle
        .folds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let records: Vec<FoldRecord> = folds
        .iter()
        .enumerate()
        .map(|(index, &fold)| FoldRecord {
            fold,
            request_sha256: bundle.request_sha256.clone(),
            fitted_artifact_sha256: bundle.fitted_artifact_sha256[index].clone(),
            fitting_row_ids: bundle
                .row_ids
                .iter()
                .zip(&bundle.folds)
                .filter(|(_, f)| **f != fold)
                .map(|(id, _)| id.clone())
                .collect(),
            scoring_row_ids: bundle
                .row_ids
                .iter()
                .zip(&bundle.folds)
                .filter(|(_, f)| **f == fold)
                .map(|(id, _)| id.clone())
                .collect(),
            status: FoldStatus::Verified,
        })
        .collect();
    let plan = ExpectedPlan {
        request_sha256: bundle.request_sha256.clone(),
        row_ids: bundle.row_ids.clone(),
        folds: bundle.folds.clone(),
        group_ids: bundle.group_ids.clone(),
        classes: bundle.classes.clone(),
    };
    let verified: Vec<VerifiedFold> = records
        .into_iter()
        .map(|record| VerifiedFold::from_verified(record).expect("test records are verified"))
        .collect();
    AdmittedBundle::admit(bundle, &plan, &verified)
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
        let distinct: BTreeSet<u32> = folds.iter().copied().collect();
        OofPredictionBundle {
            schema: OOF_SCHEMA.into(),
            task: Task::Regression,
            candidate_id: "rust.ridge".into(),
            request_sha256: "a".repeat(64),
            fitted_artifact_sha256: vec!["b".repeat(64); distinct.len()],
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
    fn an_empty_fitted_artifact_vector_is_refused() {
        let mut b = regression(vec![1.0, 2.0], vec![0, 1]);
        b.fitted_artifact_sha256.clear();
        assert!(
            b.validate().is_err(),
            "an empty artifact list admits no fitting provenance"
        );
    }

    #[test]
    fn an_artifact_count_that_differs_from_the_fold_count_is_refused() {
        let mut b = regression(vec![1.0, 2.0, 3.0], vec![0, 1, 2]);
        b.fitted_artifact_sha256.pop();
        assert!(b.validate().is_err());
    }

    #[test]
    fn a_group_spanning_two_folds_is_refused_as_crossed_leakage() {
        let mut b = regression(vec![1.0, 2.0, 3.0], vec![0, 1, 0]);
        b.group_ids = vec!["g".into(), "g".into(), "h".into()];
        assert!(b.validate().is_err());
        b.group_ids = vec!["g".into(), "h".into(), "h".into()];
        // Rows 1 and 2 share group h but sit in folds 1 and 0: crossed, so refused.
        assert!(b.validate().is_err());
        b.group_ids = vec!["g".into(), "h".into(), "g".into()];
        // Rows 0 and 2 share group g and both sit in fold 0: not crossed, so accepted.
        b.validate().unwrap();
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

    fn plan_and_records() -> (OofPredictionBundle, ExpectedPlan, Vec<FoldRecord>) {
        let b = regression(vec![1.0, 2.0, 3.0, 4.0], vec![0, 1, 0, 1]);
        let expected = ExpectedPlan {
            request_sha256: b.request_sha256.clone(),
            row_ids: b.row_ids.clone(),
            folds: b.folds.clone(),
            group_ids: vec![],
            classes: vec![],
        };
        let records = vec![
            FoldRecord {
                fold: 0,
                request_sha256: b.request_sha256.clone(),
                fitted_artifact_sha256: b.fitted_artifact_sha256[0].clone(),
                fitting_row_ids: vec!["row-1".into(), "row-3".into()],
                scoring_row_ids: vec!["row-0".into(), "row-2".into()],
                status: FoldStatus::Verified,
            },
            FoldRecord {
                fold: 1,
                request_sha256: b.request_sha256.clone(),
                fitted_artifact_sha256: b.fitted_artifact_sha256[1].clone(),
                fitting_row_ids: vec!["row-0".into(), "row-2".into()],
                scoring_row_ids: vec!["row-1".into(), "row-3".into()],
                status: FoldStatus::Verified,
            },
        ];
        (b, expected, records)
    }

    #[test]
    fn a_bundle_matching_its_verified_fold_records_is_accepted() {
        let (b, expected, records) = plan_and_records();
        b.validate_against(&expected, &records).unwrap();
    }

    #[test]
    fn unknown_or_failed_fold_records_are_refused() {
        let (b, expected, mut records) = plan_and_records();
        records[1].status = FoldStatus::Unknown;
        assert!(b.validate_against(&expected, &records).is_err());
        records[1].status = FoldStatus::Failed;
        assert!(b.validate_against(&expected, &records).is_err());
    }

    #[test]
    fn a_fold_that_fitted_on_its_own_rows_is_refused() {
        let (b, expected, mut records) = plan_and_records();
        records[0].fitting_row_ids = vec!["row-0".into(), "row-1".into(), "row-3".into()];
        assert!(b.validate_against(&expected, &records).is_err());
    }

    #[test]
    fn a_record_under_a_different_request_or_artifact_is_refused() {
        let (b, expected, mut records) = plan_and_records();
        records[0].request_sha256 = "e".repeat(64);
        assert!(b.validate_against(&expected, &records).is_err());
        let (b, expected, mut records) = plan_and_records();
        records[1].fitted_artifact_sha256 = "f".repeat(64);
        assert!(b.validate_against(&expected, &records).is_err());
    }
}
