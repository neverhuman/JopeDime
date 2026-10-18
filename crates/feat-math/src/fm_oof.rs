//! Research-only admission and out-of-fold boundaries for foundation models.
//!
//! This module has no model downloader or default-enabled runtime. Callers must
//! supply a fitted operation and explicit evidence before an operation can run.

use serde::{Deserialize, Serialize};

/// A deliberately narrow interface for a foundation feature operator.
pub trait FoundationOp {
    /// Predict features for rows without observing their labels.
    fn transform(&self, rows: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FoundationError>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LicenseAdmission {
    pub license_id: String,
    pub license_verified: bool,
    pub expected_model_sha256: String,
    pub observed_model_sha256: String,
    pub product_eligible: bool,
    pub runtime_available: bool,
    pub required_compute_units: u64,
    pub available_compute_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Admission {
    ResearchEnabled,
    ResearchDisabled(Vec<AdmissionFailure>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionFailure {
    MissingLicense,
    ModelChecksumMismatch,
    ProductNotEligible,
    RuntimeUnavailable,
    ComputeBudgetExceeded,
}

impl LicenseAdmission {
    /// Admission is fail-closed; an empty license or digest is never evidence.
    pub fn evaluate(&self) -> Admission {
        let mut failures = Vec::new();
        if !self.license_verified || self.license_id.trim().is_empty() {
            failures.push(AdmissionFailure::MissingLicense);
        }
        if self.expected_model_sha256.trim().is_empty()
            || self.expected_model_sha256 != self.observed_model_sha256
        {
            failures.push(AdmissionFailure::ModelChecksumMismatch);
        }
        if !self.product_eligible {
            failures.push(AdmissionFailure::ProductNotEligible);
        }
        if !self.runtime_available {
            failures.push(AdmissionFailure::RuntimeUnavailable);
        }
        if self.required_compute_units > self.available_compute_units {
            failures.push(AdmissionFailure::ComputeBudgetExceeded);
        }
        if failures.is_empty() {
            Admission::ResearchEnabled
        } else {
            Admission::ResearchDisabled(failures)
        }
    }
}

/// Row roles are explicit so held-out and contextual rows cannot enter fitting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OofFold {
    pub train_rows: Vec<usize>,
    pub validation_rows: Vec<usize>,
    pub context_rows: Vec<usize>,
    pub sealed_rows: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OofFeatures {
    pub validation_rows: Vec<usize>,
    pub features: Vec<Vec<f64>>,
    pub inference_units: u64,
    pub model_load_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FoundationError {
    EmptyTrainingFold,
    InvalidRowIndex(usize),
    OverlappingRows(usize),
    MissingTrainLabels,
    PredictionCountMismatch,
    FeatureWidthMismatch,
    InferenceBudgetExceeded,
}

/// Fit each operator from that fold's train rows and transform only its
/// validation rows. Context and sealed rows are rejected even when cached.
pub fn fit_oof<F, O>(
    rows: &[Vec<f64>],
    labels: &[f64],
    fold: &OofFold,
    inference_budget: u64,
    model_load_units: u64,
    mut fit: F,
) -> Result<OofFeatures, FoundationError>
where
    F: FnMut(&[Vec<f64>], &[f64]) -> Result<O, FoundationError>,
    O: FoundationOp,
{
    if fold.train_rows.is_empty() {
        return Err(FoundationError::EmptyTrainingFold);
    }
    let mut seen = std::collections::BTreeSet::new();
    for &index in fold
        .train_rows
        .iter()
        .chain(&fold.validation_rows)
        .chain(&fold.context_rows)
        .chain(&fold.sealed_rows)
    {
        if index >= rows.len() {
            return Err(FoundationError::InvalidRowIndex(index));
        }
        if !seen.insert(index) {
            return Err(FoundationError::OverlappingRows(index));
        }
    }
    if fold.train_rows.iter().any(|&i| i >= labels.len()) {
        return Err(FoundationError::MissingTrainLabels);
    }
    let train_rows = fold
        .train_rows
        .iter()
        .map(|&i| rows[i].clone())
        .collect::<Vec<_>>();
    let train_labels = fold
        .train_rows
        .iter()
        .map(|&i| labels[i])
        .collect::<Vec<_>>();
    let validation_rows = fold
        .validation_rows
        .iter()
        .map(|&i| rows[i].clone())
        .collect::<Vec<_>>();
    let operation = fit(&train_rows, &train_labels)?;
    let features = operation.transform(&validation_rows)?;
    if features.len() != fold.validation_rows.len() {
        return Err(FoundationError::PredictionCountMismatch);
    }
    let width = features.first().map_or(0, Vec::len);
    if features.iter().any(|row| row.len() != width) {
        return Err(FoundationError::FeatureWidthMismatch);
    }
    let inference_units = fold.validation_rows.len() as u64;
    let charged = inference_units
        .checked_add(model_load_units)
        .ok_or(FoundationError::InferenceBudgetExceeded)?;
    if charged > inference_budget {
        return Err(FoundationError::InferenceBudgetExceeded);
    }
    Ok(OofFeatures {
        validation_rows: fold.validation_rows.clone(),
        features,
        inference_units,
        model_load_units,
    })
}

/// Distillation evidence keeps candidate identity and predictions separate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DistillationReceipt {
    pub teacher_artifact_sha256: String,
    pub distilled_artifact_sha256: String,
    pub teacher_prediction_sha256: String,
    pub distilled_prediction_sha256: String,
    pub max_abs_error: f64,
    pub max_abs_error_limit: f64,
    pub parity_verified: bool,
    pub quality_verified: bool,
}

impl DistillationReceipt {
    /// A portable-support claim requires distinct artifacts and explicit proof.
    pub fn portable_candidate_eligible(&self) -> bool {
        !self.teacher_artifact_sha256.is_empty()
            && !self.distilled_artifact_sha256.is_empty()
            && self.teacher_artifact_sha256 != self.distilled_artifact_sha256
            && !self.teacher_prediction_sha256.is_empty()
            && !self.distilled_prediction_sha256.is_empty()
            && self.teacher_prediction_sha256 != self.distilled_prediction_sha256
            && self.max_abs_error.is_finite()
            && self.max_abs_error_limit.is_finite()
            && self.max_abs_error <= self.max_abs_error_limit
            && self.parity_verified
            && self.quality_verified
    }
}
