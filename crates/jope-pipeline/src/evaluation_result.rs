//! Evaluation receipts with explicit unavailable resource measurements.
//! v1 remains readable and unchanged; new bounded campaigns use this version.
use crate::experiment::{sha256, EvaluationRequest, EvaluationStatus};
use anyhow::{ensure, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RESULT_SCHEMA: &str = "jope.evaluation-result/v2";

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Consumption {
    pub wall_seconds: Option<f64>,
    pub cpu_seconds: Option<f64>,
    pub gpu_seconds: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub cgroup_peak_memory_bytes: Option<u64>,
    pub sampled_peak_rss_bytes: Option<u64>,
    pub accounting_reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EvaluationResult {
    pub schema: String,
    pub request_sha256: String,
    pub status: EvaluationStatus,
    pub reason: Option<String>,
    pub primary: Option<f64>,
    pub secondary: BTreeMap<String, f64>,
    pub scored_rows: usize,
    pub predictions_sha256: Option<String>,
    pub fitted_state_sha256: Option<String>,
    pub consumption: Consumption,
    /// The worker's command receipt is independently retained and authenticated.
    pub worker_receipt_sha256: String,
    /// A reused fitted predictor retains its original receipt instead of claiming a new fit.
    pub reused_from_request_sha256: Option<String>,
}

impl EvaluationResult {
    pub fn validate(&self, request: &EvaluationRequest) -> Result<()> {
        request.validate()?;
        ensure!(
            self.schema == RESULT_SCHEMA && self.request_sha256 == request.identity()?,
            "stale or mismatched result identity"
        );
        ensure!(
            sha256(&self.worker_receipt_sha256),
            "invalid worker receipt identity"
        );
        ensure!(
            self.reused_from_request_sha256
                .as_ref()
                .is_none_or(|id| sha256(id) && *id != self.request_sha256),
            "invalid predictor reuse identity"
        );
        for value in [
            self.consumption.wall_seconds,
            self.consumption.cpu_seconds,
            self.consumption.gpu_seconds,
        ]
        .into_iter()
        .flatten()
        {
            ensure!(
                value.is_finite() && value >= 0.,
                "invalid resource measurement"
            );
        }
        ensure!(
            !self.consumption.accounting_reason.is_empty(),
            "missing measurement basis"
        );
        ensure!(
            self.secondary.values().all(|v| v.is_finite()),
            "nonfinite secondary score"
        );
        ensure!(
            self.predictions_sha256
                .as_ref()
                .is_none_or(|value| sha256(value))
                && self
                    .fitted_state_sha256
                    .as_ref()
                    .is_none_or(|value| sha256(value)),
            "invalid partial artifact identity"
        );
        if self.reused_from_request_sha256.is_some() {
            ensure!(
                self.consumption.wall_seconds == Some(0.)
                    && self.consumption.cpu_seconds == Some(0.)
                    && self.consumption.gpu_seconds.is_none(),
                "reused predictor must not duplicate original execution cost"
            );
        }
        if self.status == EvaluationStatus::Completed {
            ensure!(
                self.reason.is_none(),
                "completed result has a failure reason"
            );
            ensure!(
                self.scored_rows == request.scoring_rows,
                "zero or incomplete result"
            );
            request.metric.validate_score(
                self.primary
                    .ok_or_else(|| anyhow::anyhow!("completed result lacks primary score"))?,
            )?;
            ensure!(
                self.predictions_sha256.as_ref().is_some_and(|h| sha256(h))
                    && self.fitted_state_sha256.as_ref().is_some_and(|h| sha256(h)),
                "completed result lacks prediction/fitted evidence"
            );
        } else {
            ensure!(
                self.primary.is_none() && self.secondary.is_empty() && self.scored_rows == 0,
                "incomplete result cannot publish scores"
            );
            ensure!(
                self.reason.as_ref().is_some_and(|r| !r.is_empty()),
                "incomplete result requires reason"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{EvaluationStage, Metric, REQUEST_SCHEMA};
    use jope_datasets::catalog::Task;
    fn request() -> EvaluationRequest {
        EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: "a".repeat(64),
            dataset_sha256: "b".repeat(64),
            split_sha256: "c".repeat(64),
            fitting_rows_sha256: "d".repeat(64),
            scoring_rows_sha256: "e".repeat(64),
            fitting_rows: 10,
            scoring_rows: 2,
            recipe_sha256: "f".repeat(64),
            evaluator_sha256: "1".repeat(64),
            checkpoint_sha256: None,
            task: Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: 0,
            stage: EvaluationStage::Screen,
            seed: 42,
            numerical_policy: "synthetic-contract-only".into(),
        }
    }
    fn unknown(request: &EvaluationRequest) -> EvaluationResult {
        EvaluationResult {
            schema: RESULT_SCHEMA.into(),
            request_sha256: request.identity().unwrap(),
            status: EvaluationStatus::Failed,
            reason: Some("controller interrupted; resource counters unknown".into()),
            primary: None,
            secondary: BTreeMap::new(),
            scored_rows: 0,
            predictions_sha256: None,
            fitted_state_sha256: None,
            consumption: Consumption {
                accounting_reason: "original attempt interrupted before final counters".into(),
                ..Default::default()
            },
            worker_receipt_sha256: "2".repeat(64),
            reused_from_request_sha256: None,
        }
    }
    #[test]
    fn unknown_consumption_roundtrips_as_null_and_cannot_be_a_score() {
        let req = request();
        let mut result = unknown(&req);
        result.validate(&req).unwrap();
        let json = serde_json::to_value(&result).unwrap();
        assert!(json["consumption"]["cpu_seconds"].is_null());
        assert!(json["consumption"]["wall_seconds"].is_null());
        result.primary = Some(0.);
        assert!(result.validate(&req).is_err());
        result.primary = None;
        result.consumption.accounting_reason.clear();
        assert!(result.validate(&req).is_err());
    }
    #[test]
    fn stale_worker_identity_and_partial_scoring_refuse() {
        let req = request();
        let mut result = unknown(&req);
        result.worker_receipt_sha256 = "not-a-checksum".into();
        assert!(result.validate(&req).is_err());
        result.worker_receipt_sha256 = "2".repeat(64);
        result.status = EvaluationStatus::Completed;
        result.reason = None;
        result.primary = Some(0.1);
        result.scored_rows = 1;
        result.predictions_sha256 = Some("3".repeat(64));
        result.fitted_state_sha256 = Some("4".repeat(64));
        assert!(result.validate(&req).is_err());
        result.scored_rows = 2;
        result.validate(&req).unwrap();
        let mut changed = req;
        changed.seed += 1;
        assert!(result.validate(&changed).is_err());
    }
    #[test]
    fn reuse_cannot_duplicate_original_cost_or_point_to_itself() {
        let req = request();
        let mut result = unknown(&req);
        result.reused_from_request_sha256 = Some("5".repeat(64));
        assert!(result.validate(&req).is_err());
        result.consumption.wall_seconds = Some(0.);
        result.consumption.cpu_seconds = Some(0.);
        result.validate(&req).unwrap();
        result.reused_from_request_sha256 = Some(req.identity().unwrap());
        assert!(result.validate(&req).is_err());
    }
}
