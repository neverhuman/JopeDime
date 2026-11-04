//! Prediction from a retained model has no fit permission or fallback.
//!
//! The service verifies original completion custody and separate query access
//! before calling the private scorer. Successful artifacts remain unpublished
//! until the service commits them to its original journal.
use super::authority::monotonic_ns;
use super::{verify_completion_receipt, CompletionReceipt, ExecutionOutcome, ProcessIdentity};
use crate::{manifest::hash, prepare::Preprocessing};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod preparation;
pub(super) use preparation::load_retained_model;
use preparation::{
    execution_query_key, matrix_sha256, metrics, prepare_query, receipt_digest, valid_sha,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QueryContract {
    pub schema: String,
    pub task: String,
    pub metric: String,
    pub units: String,
    pub outcome_policy: String,
    pub feature_space: String,
}
impl QueryContract {
    pub fn regression_rmse() -> Self {
        Self {
            schema: "parallel.query_contract.v1".into(),
            task: "regression".into(),
            metric: "rmse".into(),
            units: "original_target".into(),
            outcome_policy: "strict_finite_v1".into(),
            feature_space: "raw_features_v1".into(),
        }
    }
    fn verify(&self) -> Result<()> {
        ensure!(
            self == &Self::regression_rmse(),
            "unsupported query scoring contract"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryInputs {
    pub schema: String,
    pub completion_sha256: String,
    pub scope: String,
    pub task_id: String,
    /// Exact raw, target-free row order. The retained training preprocessing is
    /// applied below; callers cannot substitute a learned feature view.
    pub rows: Vec<Vec<String>>,
    pub targets: Vec<f64>,
    pub preprocessing: Vec<u8>,
    pub feature_view: Vec<u8>,
    pub contract: QueryContract,
    pub access_receipt_sha256: String,
    pub deadline_monotonic_ns: u64,
}
impl QueryInputs {
    pub fn descriptor_sha256(&self) -> Result<String> {
        Ok(hash(&serde_json::to_vec(self)?))
    }
}

/// A fit scope, receipt hash, or caller Boolean cannot construct query access.
pub(super) struct VerifiedQueryAccess {
    descriptor_sha256: String,
}
impl VerifiedQueryAccess {
    pub(super) fn verify(&self, inputs: &QueryInputs) -> Result<()> {
        ensure!(
            self.descriptor_sha256 == inputs.descriptor_sha256()?,
            "query access does not bind exact request"
        );
        Ok(())
    }
}
pub(super) fn authorize(_inputs: &QueryInputs) -> Result<VerifiedQueryAccess> {
    bail!("query outcome access unavailable: original selection/access receipt verifier is not qualified; fitting scope grants no query access")
}
#[cfg(test)]
pub(super) fn synthetic_access_for_test(inputs: &QueryInputs) -> Result<VerifiedQueryAccess> {
    ensure!(
        inputs.scope == "synthetic",
        "private synthetic query scope required"
    );
    Ok(VerifiedQueryAccess {
        descriptor_sha256: inputs.descriptor_sha256()?,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryReceipt {
    pub schema: String,
    pub query_key: String,
    pub authority_instance: String,
    pub original_completion_sha256: String,
    pub fit_key: String,
    pub model_sha256: String,
    pub query_rows_sha256: String,
    pub targets_sha256: String,
    pub transformed_query_sha256: String,
    pub preprocessing_sha256: String,
    pub feature_view_sha256: String,
    pub contract: QueryContract,
    pub request_sha256: String,
    pub access_receipt_sha256: String,
    pub scope: String,
    pub task_id: String,
    pub rows: usize,
    pub features: usize,
    pub predictions_sha256: String,
    pub metrics_sha256: String,
    pub producer_source_sha256: String,
    pub predictor_source_sha256: String,
    pub preprocessing_source_sha256: String,
    pub producer_executable_sha256: String,
    pub producer_runtime_sha256: String,
    pub producer_process: ProcessIdentity,
    pub started_unix_ns: u64,
    pub started_monotonic_ns: u64,
    pub finished_monotonic_ns: u64,
    pub deadline_monotonic_ns: u64,
    pub prediction_elapsed_seconds: f64,
    /// Receipt publication stays in the service; prediction runs in its owned child.
    pub completion_kind: String,
    pub predictor_process: ProcessIdentity,
    pub predictor_exit_code: Option<i32>,
    pub receipt_sha256: String,
}

/// Fields are private: callers cannot turn plausible output into quality by
/// constructing an artifact bundle. Only the scorer below creates this value.
pub(super) struct QueryArtifacts {
    request: Vec<u8>,
    predictions: Vec<u8>,
    metrics: Vec<u8>,
    receipt_bytes: Vec<u8>,
    receipt: QueryReceipt,
}
impl QueryArtifacts {
    pub(super) fn request_bytes(&self) -> &[u8] {
        &self.request
    }
    pub(super) fn predictions_bytes(&self) -> &[u8] {
        &self.predictions
    }
    pub(super) fn metrics_bytes(&self) -> &[u8] {
        &self.metrics
    }
    pub(super) fn receipt_bytes(&self) -> &[u8] {
        &self.receipt_bytes
    }
    pub(super) fn receipt(&self) -> &QueryReceipt {
        &self.receipt
    }
}

pub(super) struct Prediction {
    pub values: Vec<f64>,
    pub process: ProcessIdentity,
    pub exit_code: Option<i32>,
}

pub(super) fn score_verified(
    completion: &[u8],
    model: &[u8],
    inputs: &QueryInputs,
    access: &VerifiedQueryAccess,
    runtime: &str,
    predict: impl FnOnce(&[u8], &[Vec<f64>], u64) -> Result<Prediction>,
) -> Result<QueryArtifacts> {
    access.verify(inputs)?;
    ensure!(valid_sha(runtime), "verified query producer runtime absent");
    let started_monotonic_ns = monotonic_ns()?;
    ensure!(
        started_monotonic_ns < inputs.deadline_monotonic_ns,
        "query deadline expired before evaluation"
    );
    let prepared = prepare_query(completion, model, inputs)?;
    let producer_executable_sha256 = hash(&std::fs::read("/proc/self/exe")?);
    let producer_process = ProcessIdentity::current()?;
    let started_unix_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_nanos()
        .try_into()?;
    ensure!(
        monotonic_ns()? < inputs.deadline_monotonic_ns,
        "query deadline expired during input verification"
    );
    let started = Instant::now();
    let prediction = predict(model, &prepared.rows, inputs.deadline_monotonic_ns)?;
    let prediction_elapsed_seconds = started.elapsed().as_secs_f64();
    let predictions: Vec<_> = prediction
        .values
        .into_iter()
        .map(|p| p * prepared.preprocessing.target_scale + prepared.preprocessing.target_mean)
        .collect();
    let metrics = metrics(&inputs.targets, &predictions)?;
    let predictions = serde_json::to_vec(&predictions)?;
    let request = serde_json::to_vec(inputs)?;
    let finished_monotonic_ns = monotonic_ns()?;
    ensure!(
        finished_monotonic_ns < inputs.deadline_monotonic_ns,
        "query deadline expired; no query quality committed"
    );
    let mut receipt = QueryReceipt {
        schema: "parallel.query_completion.v1".into(),
        query_key: execution_query_key(&prepared.query_key, &producer_executable_sha256, runtime)?,
        authority_instance: prepared.original.authority_instance,
        original_completion_sha256: inputs.completion_sha256.clone(),
        fit_key: prepared.original.fit_key,
        model_sha256: hash(model),
        query_rows_sha256: prepared.query_rows_sha256,
        targets_sha256: prepared.targets_sha256,
        transformed_query_sha256: matrix_sha256(&prepared.rows),
        preprocessing_sha256: hash(&inputs.preprocessing),
        feature_view_sha256: hash(&inputs.feature_view),
        contract: inputs.contract.clone(),
        request_sha256: hash(&request),
        access_receipt_sha256: inputs.access_receipt_sha256.clone(),
        scope: inputs.scope.clone(),
        task_id: inputs.task_id.clone(),
        rows: inputs.rows.len(),
        features: prepared.preprocessing.columns.len(),
        predictions_sha256: hash(&predictions),
        metrics_sha256: hash(&metrics),
        producer_source_sha256: hash(include_bytes!("query.rs")),
        predictor_source_sha256: hash(include_bytes!("../../../feat-math/src/model.rs")),
        preprocessing_source_sha256: hash(include_bytes!("../prepare.rs")),
        producer_executable_sha256,
        producer_runtime_sha256: runtime.into(),
        producer_process,
        started_unix_ns,
        started_monotonic_ns,
        finished_monotonic_ns,
        deadline_monotonic_ns: inputs.deadline_monotonic_ns,
        prediction_elapsed_seconds,
        completion_kind: "service_observed_predictor_exit".into(),
        predictor_process: prediction.process,
        predictor_exit_code: prediction.exit_code,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest(&receipt)?;
    let receipt_bytes = serde_json::to_vec(&receipt)?;
    verify_artifacts(
        completion,
        model,
        inputs,
        &receipt_bytes,
        &predictions,
        &metrics,
    )?;
    Ok(QueryArtifacts {
        request,
        predictions,
        metrics,
        receipt_bytes,
        receipt,
    })
}

/// Checks identities and metric arithmetic without predicting again. Durable
/// original-journal membership and producer approval must be checked by the
/// authority; a self-consistent receipt alone does not prove original execution.
pub fn verify_artifacts(
    completion: &[u8],
    model: &[u8],
    inputs: &QueryInputs,
    receipt: &[u8],
    predictions: &[u8],
    recorded_metrics: &[u8],
) -> Result<QueryReceipt> {
    let prepared = prepare_query(completion, model, inputs)?;
    let receipt: QueryReceipt = serde_json::from_slice(receipt)?;
    let values: Vec<f64> = serde_json::from_slice(predictions)?;
    ensure!(
        metrics(&inputs.targets, &values)? == recorded_metrics,
        "query metrics differ from retained predictions"
    );
    ensure!(
        receipt.schema == "parallel.query_completion.v1"
            && receipt.receipt_sha256 == receipt_digest(&receipt)?
            && receipt.query_key
                == execution_query_key(
                    &prepared.query_key,
                    &receipt.producer_executable_sha256,
                    &receipt.producer_runtime_sha256
                )?
            && receipt.authority_instance == prepared.original.authority_instance
            && receipt.original_completion_sha256 == inputs.completion_sha256
            && receipt.fit_key == prepared.original.fit_key
            && receipt.model_sha256 == hash(model)
            && receipt.query_rows_sha256 == prepared.query_rows_sha256
            && receipt.targets_sha256 == prepared.targets_sha256
            && receipt.transformed_query_sha256 == matrix_sha256(&prepared.rows)
            && receipt.preprocessing_sha256 == hash(&inputs.preprocessing)
            && receipt.feature_view_sha256 == hash(&inputs.feature_view)
            && receipt.contract == inputs.contract
            && receipt.request_sha256 == inputs.descriptor_sha256()?
            && receipt.access_receipt_sha256 == inputs.access_receipt_sha256
            && receipt.scope == inputs.scope
            && receipt.task_id == inputs.task_id
            && receipt.rows == inputs.rows.len()
            && receipt.features == prepared.preprocessing.columns.len()
            && receipt.predictions_sha256 == hash(predictions)
            && receipt.metrics_sha256 == hash(recorded_metrics),
        "query receipt identity or output mismatch"
    );
    ensure!(
        receipt.producer_source_sha256 == hash(include_bytes!("query.rs"))
            && receipt.predictor_source_sha256
                == hash(include_bytes!("../../../feat-math/src/model.rs"))
            && receipt.preprocessing_source_sha256 == hash(include_bytes!("../prepare.rs"))
            && valid_sha(&receipt.producer_executable_sha256)
            && valid_sha(&receipt.producer_runtime_sha256)
            && receipt.producer_process.pid > 0
            && receipt.producer_process.birth_ticks > 0
            && !receipt.producer_process.boot_id.is_empty()
            && receipt.started_unix_ns > 0
            && receipt.started_monotonic_ns <= receipt.finished_monotonic_ns
            && receipt.finished_monotonic_ns < receipt.deadline_monotonic_ns
            && receipt.deadline_monotonic_ns == inputs.deadline_monotonic_ns
            && receipt.prediction_elapsed_seconds.is_finite()
            && receipt.prediction_elapsed_seconds >= 0.
            && receipt.prediction_elapsed_seconds
                <= (receipt.finished_monotonic_ns - receipt.started_monotonic_ns) as f64 / 1e9
            && receipt.completion_kind == "service_observed_predictor_exit"
            && receipt.predictor_exit_code == Some(0)
            && receipt.predictor_process.pid > 0
            && receipt.predictor_process.birth_ticks > 0
            && receipt.predictor_process.boot_id == receipt.producer_process.boot_id,
        "query producer or timing evidence invalid"
    );
    Ok(receipt)
}

#[cfg(test)]
mod tests;
