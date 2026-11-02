//! Prediction from a retained model has no fit permission or fallback.
//!
//! The service verifies original completion custody and separate query access
//! before calling the private scorer. Successful artifacts remain unpublished
//! until the service commits them to its original journal.
use super::{verify_completion_receipt, CompletionReceipt, ExecutionOutcome, ProcessIdentity};
use crate::{manifest::hash, prepare::Preprocessing};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const MAX_ROWS: usize = 10_000;
const MAX_COLUMNS: usize = 1_024;
const MAX_CELLS: usize = 1_000_000;

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

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Decode only the retained fitted state. Every caller, including private
/// contract fixtures, uses the same rejection contract; no fitting fallback.
pub(super) fn load_retained_model(bytes: &[u8]) -> Result<feat_math::Model> {
    serde_json::from_slice(bytes)
        .context("retained model format unavailable; refitting is forbidden")
}

fn monotonic_ns() -> Result<u64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `time` points to a live, writable timespec and has the exact
    // layout expected by clock_gettime; the return code gates its use below.
    ensure!(
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } == 0,
        "query monotonic clock unavailable"
    );
    (time.tv_sec as u64)
        .checked_mul(1_000_000_000)
        .and_then(|n| n.checked_add(time.tv_nsec as u64))
        .context("query clock overflow")
}
fn vector_bits(values: &[f64]) -> Vec<u8> {
    let mut bytes = (values.len() as u64).to_le_bytes().to_vec();
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}
fn matrix_sha256(rows: &[Vec<f64>]) -> String {
    let mut bytes = (rows.len() as u64).to_le_bytes().to_vec();
    for row in rows {
        bytes.extend_from_slice(&vector_bits(row));
    }
    hash(&bytes)
}
fn receipt_digest(receipt: &QueryReceipt) -> Result<String> {
    let mut value = serde_json::to_value(receipt)?;
    value
        .as_object_mut()
        .context("query receipt object required")?
        .remove("receipt_sha256");
    Ok(hash(&serde_json::to_vec(&value)?))
}
fn execution_query_key(input_key: &str, executable: &str, runtime: &str) -> Result<String> {
    Ok(hash(&serde_json::to_vec(
        &json!({"schema":"parallel.query_execution_identity.v1",
        "inputs_sha256":input_key,"executable_sha256":executable,"runtime_sha256":runtime,
        "scorer_source_sha256":hash(include_bytes!("query.rs")),
        "predictor_source_sha256":hash(include_bytes!("../../../feat-math/src/model.rs")),
        "preprocessing_source_sha256":hash(include_bytes!("../prepare.rs"))}),
    )?))
}

struct PreparedQuery {
    original: CompletionReceipt,
    rows: Vec<Vec<f64>>,
    preprocessing: Preprocessing,
    query_key: String,
    query_rows_sha256: String,
    targets_sha256: String,
}
fn prepare_query(completion: &[u8], model: &[u8], inputs: &QueryInputs) -> Result<PreparedQuery> {
    ensure!(
        inputs.schema == "parallel.query_inputs.v1",
        "unsupported query input schema"
    );
    inputs.contract.verify()?;
    ensure!(
        valid_sha(&inputs.completion_sha256)
            && valid_sha(&inputs.access_receipt_sha256)
            && !inputs.scope.trim().is_empty()
            && !inputs.task_id.trim().is_empty(),
        "query completion/access identity or provenance absent"
    );
    ensure!(
        hash(completion) == inputs.completion_sha256,
        "original query completion changed"
    );
    let original = verify_completion_receipt(completion)?;
    ensure!(
        original.outcome == ExecutionOutcome::Success,
        "unsuccessful original fit has no query quality"
    );
    ensure!(
        original.model_sha256.as_deref() == Some(hash(model).as_str()),
        "retained query model unavailable or changed"
    );
    ensure!(
        hash(&inputs.preprocessing) == original.fit_identity.preprocessing_sha256
            && hash(&inputs.feature_view) == original.fit_identity.feature_view_sha256,
        "query preprocessing or learned feature view differs from original fit"
    );
    // Identity-only preprocessing does not retain a provable feature width.
    // Compact views need their own qualified reader and are deliberately denied.
    let pp: Preprocessing = serde_json::from_slice(&inputs.preprocessing)
        .context("query needs retained training preprocessing and feature dimensions")?;
    ensure!(
        pp.schema == "parallel.preprocessing.v1"
            && pp.labels.is_empty()
            && !pp.columns.is_empty()
            && pp.columns.len() <= MAX_COLUMNS
            && pp.target_mean.is_finite()
            && pp.target_scale.is_finite()
            && pp.target_scale > 0.,
        "unsupported or malformed regression preprocessing"
    );
    for column in &pp.columns {
        ensure!(
            matches!(column.kind.as_str(), "numeric" | "categorical")
                && column.impute.is_finite()
                && column.mean.is_finite()
                && column.scale.is_finite()
                && column.scale > 0.
                && column.categories.windows(2).all(|pair| pair[0] < pair[1]),
            "malformed retained feature transformation"
        );
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawView {
        protocol: String,
        training_sha256: String,
        preprocessing_sha256: String,
    }
    let view: RawView = serde_json::from_slice(&inputs.feature_view)?;
    ensure!(
        view.protocol == "raw_training_fold_v1"
            && valid_sha(&view.training_sha256)
            && view.preprocessing_sha256 == original.fit_identity.preprocessing_sha256,
        "unsupported compact or conflicting raw feature view"
    );
    ensure!(
        !inputs.rows.is_empty()
            && inputs.rows.len() <= MAX_ROWS
            && inputs
                .rows
                .len()
                .checked_mul(pp.columns.len())
                .is_some_and(|n| n <= MAX_CELLS)
            && inputs.rows.len() == inputs.targets.len()
            && inputs.targets.iter().all(|v| v.is_finite()),
        "invalid or oversized query rows/targets"
    );
    let rows = inputs
        .rows
        .iter()
        .map(|row| pp.features(row))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        rows.iter().flatten().all(|v| v.is_finite()),
        "nonfinite transformed query input"
    );
    let query_rows_sha256 = hash(&serde_json::to_vec(&inputs.rows)?);
    let targets_sha256 = hash(&vector_bits(&inputs.targets));
    let query_key = hash(&serde_json::to_vec(&json!({
        "schema":"parallel.query_identity.v1", "model_sha256":original.model_sha256,
        "original_completion_sha256":inputs.completion_sha256,
        "query_rows_sha256":query_rows_sha256,"targets_sha256":targets_sha256,
        "transformed_query_sha256":matrix_sha256(&rows),
        "preprocessing_sha256":original.fit_identity.preprocessing_sha256,
        "feature_view_sha256":original.fit_identity.feature_view_sha256,"contract":inputs.contract,
    }))?);
    Ok(PreparedQuery {
        original,
        rows,
        preprocessing: pp,
        query_key,
        query_rows_sha256,
        targets_sha256,
    })
}

fn metrics(targets: &[f64], predictions: &[f64]) -> Result<Vec<u8>> {
    ensure!(
        !targets.is_empty()
            && targets.len() == predictions.len()
            && targets.iter().chain(predictions).all(|v| v.is_finite()),
        "invalid query predictions or targets"
    );
    let mse = targets
        .iter()
        .zip(predictions)
        .map(|(y, p)| (y - p).powi(2))
        .sum::<f64>()
        / targets.len() as f64;
    ensure!(
        mse.is_finite() && mse >= 0.,
        "nonfinite query loss cannot create quality"
    );
    Ok(serde_json::to_vec(
        &json!({"schema":"parallel.query_metrics.v1",
        "task":"regression","metric":"rmse","units":"original_target",
        "outcome_policy":"strict_finite_v1","rows":targets.len(),"rmse":mse.sqrt()}),
    )?)
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
