//! Frozen completed-run models shared by HTTP, MCP and batch clients. No fitting here.
use super::auth::problem;
use crate::{
    dataset::register::RegisteredDataset,
    engine::DatasetState,
    events::EventLog,
    export::bundle::{self, Bundle, BundleManifest, StoreBundleWriter},
    idempotency::{CommandOutcome, CommandReceipt, RequestIdentity},
    store::{BlobRef, Record, RecordKey},
};
use axum::{
    body::{to_bytes, Body},
    extract::{Extension, Path, Query, Request},
    http::header,
    response::{IntoResponse, Response},
    Json,
};
use jope_api::{
    error::{ErrorCode, Problem},
    ids::{AlgorithmId, ArtifactId, RunId},
    *,
};
use jope_runtime::model_object::{Manifest, ModelObject};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

const MODEL_BYTES: u64 = 16 * 1024 * 1024;
const ARCHIVE_BYTES: u64 = 64 * 1024 * 1024;
type Result<T> = std::result::Result<T, Box<Problem>>;
fn error(code: ErrorCode, detail: &str) -> Box<Problem> {
    Box::new(Problem { r#type: "about:blank".into(), title: code.as_str().into(),
        status: code.http_status().unwrap_or(500), detail: detail.into(), instance: None,
        code, retriable: false, remedy: "Use a completed CPU run, its frozen model and a registered dataset; inspect supported export languages.".into(), field: None })
}
fn bad() -> Box<Problem> {
    error(ErrorCode::InvalidInput, "Invalid model request.")
}
fn integrity() -> Box<Problem> {
    error(
        ErrorCode::ModelIntegrityFailed,
        "Frozen model or artifact integrity failed.",
    )
}
fn missing() -> Box<Problem> {
    error(
        ErrorCode::NotFound,
        "Requested run, algorithm, model or dataset was not found.",
    )
}
fn unavailable() -> Box<Problem> {
    error(
        ErrorCode::CapabilityUnavailable,
        "This request is not supported by the frozen completed model.",
    )
}
fn response(e: &Problem) -> Response {
    problem(e.code, &e.detail, &e.remedy, e.field.as_deref())
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn object(state: &DatasetState, id: &ArtifactId) -> Result<(ModelObject, Record)> {
    let record = state
        .store
        .get(&RecordKey::Artifact(id.clone()))
        .map_err(|_| integrity())?
        .ok_or_else(missing)?;
    if !matches!(
        record.data["kind"].as_str(),
        Some("finalized_model" | "model_export")
    ) || record.data["export_kind"] == "source"
    {
        return Err(unavailable());
    }
    let first = record
        .blobs
        .first()
        .filter(|b| b.bytes <= MODEL_BYTES)
        .ok_or_else(integrity)?;
    let bytes = state.store.read_blob(first).map_err(|_| integrity())?;
    let manifest: Manifest = jope_runtime::strict_json::decode(&bytes).map_err(|_| integrity())?;
    manifest.validate(MODEL_BYTES).map_err(|_| integrity())?;
    let mut parts = BTreeMap::new();
    for part in &manifest.files {
        let reference = record
            .blobs
            .iter()
            .find(|b| b.bytes == part.bytes && hex(&b.sha256) == part.sha256)
            .ok_or_else(integrity)?;
        parts.insert(
            part.path.clone(),
            state.store.read_blob(reference).map_err(|_| integrity())?,
        );
    }
    let model = ModelObject::decode(bytes, parts, MODEL_BYTES).map_err(|_| integrity())?;
    if record.data["model_digest"] != model.id() {
        return Err(integrity());
    }
    model.verify_executable().map_err(|_| integrity())?;
    Ok((model, record))
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}
fn champion(
    state: &DatasetState,
    run: &RunId,
    alg: Option<&AlgorithmId>,
) -> Result<(RunSnapshot, ModelObject, Record)> {
    let snapshot = EventLog::new(Arc::clone(&state.store))
        .snapshot(run)
        .map_err(|_| integrity())?
        .ok_or_else(missing)?;
    if snapshot.state != RunState::Completed {
        return Err(error(
            ErrorCode::RevisionConflict,
            "Run finalization has not completed.",
        ));
    }
    let best = snapshot.best.as_ref().ok_or_else(missing)?;
    if alg.is_some_and(|id| *id != best.algorithm_id) {
        return Err(missing());
    }
    let (model, record) = object(state, best.model_artifact.as_ref().ok_or_else(missing)?)?;
    if model
        .manifest()
        .metadata
        .get("run_id")
        .is_some_and(|v| v != &json!(run))
    {
        return Err(integrity());
    }
    Ok((snapshot, model, record))
}

// Select only an already-fitted model whose role is bound by this run's completed
// finalization receipt. Export cannot open partition rows or start another fit.
fn export_champion(
    state: &DatasetState,
    run: &RunId,
    algorithm: &AlgorithmId,
    refit: Refit,
) -> Result<(ModelObject, Record)> {
    let (snapshot, model, record) = champion(state, run, Some(algorithm))?;
    let completed_refit = record.data.get("refit").cloned().unwrap_or(json!("fit"));
    if completed_refit == json!(refit) {
        return Ok((model, record));
    }
    if refit != Refit::Fit || completed_refit != json!(Refit::FitHoldout) {
        return Err(unavailable());
    }
    let key = RecordKey::Command {
        principal: "discover.finalization.v1".into(),
        idempotency_key: run.to_string(),
    };
    let completed = state
        .store
        .get(&key)
        .map_err(|_| integrity())?
        .ok_or_else(unavailable)?;
    if completed.data["state"] != "completed" {
        return Err(integrity());
    }
    let receipt: crate::discover::finalize::FinalizationReceipt =
        serde_json::from_value(completed.data["receipt"].clone()).map_err(|_| integrity())?;
    let best = snapshot.best.as_ref().ok_or_else(integrity)?;
    if receipt.run_id != *run
        || receipt.algorithm_id != *algorithm
        || receipt.refit != Refit::FitHoldout
        || Some(&receipt.model_artifact) != best.model_artifact.as_ref()
        || receipt.model_digest != model.id()
    {
        return Err(integrity());
    }
    let digest = receipt
        .holdout_model_digest
        .strip_prefix("sha256-")
        .ok_or_else(integrity)?;
    hex_digest(digest)?;
    let fit_id: ArtifactId = format!("art_{}", &digest[..24])
        .parse()
        .map_err(|_| integrity())?;
    let (fit, fit_record) = object(state, &fit_id)?;
    if fit.id() != receipt.holdout_model_digest
        || fit_record.data["refit"] != json!(Refit::Fit)
        || fit
            .manifest()
            .metadata
            .get("run_id")
            .is_some_and(|v| v != &json!(run))
    {
        return Err(integrity());
    }
    Ok((fit, fit_record))
}

mod algorithm;
mod export;

pub(super) use algorithm::*;
pub(super) use export::*;

#[cfg(test)]
mod tests;
