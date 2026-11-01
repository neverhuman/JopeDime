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

#[derive(Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AlgorithmQuery {
    language: Option<Language>,
    diff_from: Option<AlgorithmId>,
    after: Option<AlgorithmId>,
    limit: Option<usize>,
}
fn algorithm(
    state: &DatasetState,
    run: RunId,
    alg: Option<AlgorithmId>,
    query: AlgorithmQuery,
    list: bool,
) -> Result<Value> {
    if query.diff_from.is_some() || query.limit.is_some_and(|n| !(1..=256).contains(&n)) {
        return Err(bad());
    }
    let (snapshot, model, _) = champion(state, &run, alg.as_ref())?;
    let best = snapshot.best.as_ref().ok_or_else(missing)?;
    let language = query.language.unwrap_or(Language::Rust);
    let name = language_name(language);
    let source = String::from_utf8(model.export_source(name).map_err(|_| unavailable())?)
        .map_err(|_| integrity())?;
    let result = Algorithm {
        algorithm_id: best.algorithm_id.clone(),
        summary: "Frozen completed CPU regression model".into(),
        metric: best.metric.clone(),
        cv_value: best.cv_value,
        model_artifact: best.model_artifact.clone(),
        model_digest: model.id().into(),
        conditioning_digest: "none".into(),
        bundle_digest: "none".into(),
        revision: snapshot.revision,
        source: vec![AlgorithmSource { language, source }],
    };
    if list {
        Ok(
            json!({"algorithms":if query.after.as_ref().is_some_and(|id| id >= &best.algorithm_id) {vec![]} else {vec![result]},"next_after":null}),
        )
    } else {
        serde_json::to_value(result).map_err(|_| integrity())
    }
}
fn language_name(v: Language) -> &'static str {
    match v {
        Language::Rust => "rust",
        Language::Python => "python",
        Language::Javascript => "javascript",
        Language::Typescript => "typescript",
    }
}
pub(super) async fn get_algorithm(
    Extension(state): Extension<DatasetState>,
    path: std::result::Result<Path<(String, String)>, axum::extract::rejection::PathRejection>,
    query: std::result::Result<Query<AlgorithmQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let (Ok(Path((run, alg))), Ok(Query(query))) = (path, query) else {
        return response(&bad());
    };
    let (Ok(run), Ok(alg)) = (run.parse::<RunId>(), alg.parse::<AlgorithmId>()) else {
        return response(&bad());
    };
    finish_json(
        tokio::task::spawn_blocking(move || algorithm(&state, run, Some(alg), query, false)).await,
    )
}
pub(super) async fn list_algorithms(
    Extension(state): Extension<DatasetState>,
    path: std::result::Result<Path<String>, axum::extract::rejection::PathRejection>,
    query: std::result::Result<Query<AlgorithmQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let (Ok(Path(run)), Ok(Query(query))) = (path, query) else {
        return response(&bad());
    };
    let Ok(run) = run.parse::<RunId>() else {
        return response(&bad());
    };
    finish_json(
        tokio::task::spawn_blocking(move || algorithm(&state, run, None, query, true)).await,
    )
}
fn finish_json(result: std::result::Result<Result<Value>, tokio::task::JoinError>) -> Response {
    match result {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => response(&e),
        Err(_) => response(&integrity()),
    }
}

async fn command<T: DeserializeOwned + Serialize>(
    request: Request<Body>,
    operation: u8,
) -> Result<(T, RequestIdentity)> {
    let headers = request.headers().clone();
    let mut keys = headers.get_all("idempotency-key").iter();
    let Some(key) = keys.next().and_then(|v| v.to_str().ok()) else {
        return Err(bad());
    };
    if keys.next().is_some() {
        return Err(bad());
    }
    let target = request
        .uri()
        .path_and_query()
        .map(|p| p.as_str().to_owned())
        .ok_or_else(bad)?;
    let bytes = to_bytes(request.into_body(), 1024 * 1024)
        .await
        .map_err(|_| bad())?;
    let body: T = jope_runtime::strict_json::decode(&bytes).map_err(|_| bad())?;
    let value = serde_json::to_value(&body).map_err(|_| bad())?;
    let identity = RequestIdentity::new(
        "local-engine-owner",
        key,
        operation,
        &target,
        headers.get("if-match").and_then(|v| v.to_str().ok()),
        &value,
    )
    .map_err(|_| bad())?;
    Ok((body, identity))
}
fn export(
    state: &DatasetState,
    run: RunId,
    request: ExportRequest,
    identity: RequestIdentity,
) -> Response {
    match state.store.idempotent_receipt(&identity) {
        Ok(Some(r)) => return r.into_response(),
        Err(e) => return e.into_response(),
        Ok(None) => {}
    }
    let result = (|| -> Result<(ArtifactId, Record, ExportManifest)> {
        let (model, original) = export_champion(state, &run, &request.algorithm_id, request.refit)?;
        let mut entries = BTreeMap::new();
        let mut languages = vec![];
        if request.kind != ExportKind::Source {
            entries.insert(
                "model/manifest.json".into(),
                model.manifest_bytes().to_vec(),
            );
            for (name, bytes) in model.parts() {
                entries.insert(format!("model/{name}"), bytes.clone());
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        if request.kind != ExportKind::Model {
            for language in &request.languages {
                let name = language_name(*language);
                if !seen.insert(name) {
                    return Err(bad());
                }
                match model.export_source(name) {
                    Ok(bytes) => {
                        let extension = match language {
                            Language::Rust => "rs",
                            Language::Python => "py",
                            _ => "mjs",
                        };
                        let path = format!("src/{name}/predictor.{extension}");
                        entries.insert(path.clone(), bytes);
                        languages.push(LanguageExport::Exported {
                            language: *language,
                            path,
                        });
                    }
                    Err(_) if request.require_all_languages => return Err(unavailable()),
                    Err(_) => languages.push(LanguageExport::Unsupported {
                        language: *language,
                        operations: vec!["requested_language".into()],
                    }),
                }
            }
        }
        if request.kind == ExportKind::Source && entries.is_empty() {
            return Err(unavailable());
        }
        entries.insert(
            "export.json".into(),
            serde_json::to_vec(&request).map_err(|_| integrity())?,
        );
        if entries.values().map(|v| v.len() as u64).sum::<u64>() > ARCHIVE_BYTES {
            return Err(bad());
        }
        let measured =
            bundle::verified_model_parity(&state.store, &original, &model, &request.languages)
                .map_err(|_| integrity())?;
        if let Some((_, _, bytes)) = &measured {
            entries.insert("parity_receipt.json".into(), bytes.clone());
        }
        let pipeline = model.math_pipeline().map_err(|_| unavailable())?;
        let retains = bundle::retains_training_data(&pipeline.state.model);
        for language in &languages {
            let LanguageExport::Exported { language, .. } = language else {
                continue;
            };
            let name = language_name(*language);
            let usage = match language {
                Language::Python => "Run `python3 predictor.py --self-check` to verify the embedded checks. Import `predict_rows` for prediction batches.",
                Language::Rust => "Run `rustc --edition=2021 --test predictor.rs -o predictor-self-check` and `./predictor-self-check` to verify the embedded checks. Include the source as a module and call `predict_cells` for prediction batches.",
                _ => "Import the fitted predictor module using the prediction input contract for the bundled model.",
            };
            let privacy = if retains {
                "\nPrivacy notice: this fitted model retains training rows in its learned state. Treat the bundle as containing training data.\n"
            } else {
                ""
            };
            entries.insert(
                format!("src/{name}/README.md"),
                format!("# {name} fitted predictor\n\n{usage}\n\nPass raw feature cells in the bundled input-schema order; do not include the target column. The predictor applies its fitted preprocessing. Model identity: {}. Refit mode: {}.\n{privacy}", model.id(), serde_json::to_string(&request.refit).map_err(|_| integrity())?).into_bytes(),
            );
        }
        let archive_manifest = BundleManifest {
            schema_version: bundle::BUNDLE_SCHEMA_VERSION,
            model_object_digest: hash(model.manifest_bytes()),
            retains_training_data: retains,
            entries: entries.iter().map(|(k, v)| (k.clone(), hash(v))).collect(),
        };
        entries.insert(
            "manifest.json".into(),
            serde_json::to_vec(&archive_manifest).map_err(|_| integrity())?,
        );
        let published = bundle::publish(
            &Bundle {
                entries,
                manifest: archive_manifest,
            },
            &mut StoreBundleWriter {
                store: &state.store,
            },
        )
        .map_err(|_| integrity())?;
        if published.size > ARCHIVE_BYTES {
            return Err(bad());
        }
        let archive = BlobRef {
            sha256: hex_digest(&published.sha256)?,
            bytes: published.size,
        };
        let id: ArtifactId = format!("art_{}", &published.sha256[..24])
            .parse()
            .map_err(|_| integrity())?;
        let manifest = ExportManifest {
            artifact_id: id.clone(),
            algorithm_id: request.algorithm_id,
            kind: request.kind,
            model_digest: model.id().into(),
            emitter_version: env!("CARGO_PKG_VERSION").into(),
            qualified_set_id: measured
                .as_ref()
                .map_or_else(|| "not-qualified".into(), |(_, id, _)| id.clone()),
            refit: request.refit,
            retains_training_data: retains,
            languages,
            parity_status: if measured.is_some() {
                ParityStatus::Passed
            } else {
                ParityStatus::Unavailable
            },
            parity_receipt: measured.map(|(id, _, _)| id),
        };
        let mut refs = original.blobs;
        refs.push(archive.clone());
        let record = Record::metadata(
            json!({"kind":"model_export","export_kind":request.kind,"model_digest":model.id(),"archive":archive}),
            refs,
        );
        Ok((id, record, manifest))
    })();
    let (id, record, manifest) = match result {
        Ok(v) => v,
        Err(e) => return response(&e),
    };
    match state.store.execute_idempotent(&identity, move |tx| {
        let key = RecordKey::Artifact(id);
        if let Some(old) = tx.get(&key)? {
            if old != record {
                return Err(crate::store::StoreError::Integrity(
                    "artifact identity collision",
                ));
            }
        } else {
            tx.put(&key, &record)?
        }
        CommandOutcome::json(200, serde_json::to_value(manifest)?)
    }) {
        Ok(v) => v.into_response(),
        Err(e) => e.into_response(),
    }
}
fn hex_digest(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        return Err(integrity());
    }
    let mut bytes = [0; 32];
    for (i, slot) in bytes.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[2 * i..2 * i + 2], 16).map_err(|_| integrity())?
    }
    Ok(bytes)
}
pub(super) async fn export_model(
    Extension(state): Extension<DatasetState>,
    path: std::result::Result<Path<String>, axum::extract::rejection::PathRejection>,
    request: Request<Body>,
) -> Response {
    let Ok(Path(raw)) = path else {
        return response(&bad());
    };
    let Ok(run) = raw.parse::<RunId>() else {
        return response(&bad());
    };
    let (body, identity) = match command::<ExportRequest>(request, 13).await {
        Ok(v) => v,
        Err(e) => return response(&e),
    };
    match tokio::task::spawn_blocking(move || export(&state, run, body, identity)).await {
        Ok(v) => v,
        Err(_) => response(&integrity()),
    }
}
fn csv(receipt: CommandReceipt) -> Response {
    match receipt.outcome.body().as_str() {
        Some(v) => ([(header::CONTENT_TYPE, "text/csv")], v.to_owned()).into_response(),
        None => response(&integrity()),
    }
}
fn predict(state: &DatasetState, request: PredictRequest, identity: RequestIdentity) -> Response {
    match state.store.idempotent_receipt(&identity) {
        Ok(Some(r)) => return csv(r),
        Err(e) => return e.into_response(),
        Ok(None) => {}
    }
    let result = (|| -> Result<String> {
        let (model, _) = object(state, &request.model_artifact)?;
        let record = state
            .store
            .get(&RecordKey::Dataset(request.dataset.clone()))
            .map_err(|_| integrity())?
            .ok_or_else(missing)?;
        let dataset: RegisteredDataset =
            serde_json::from_value(record.data).map_err(|_| integrity())?;
        if dataset.id != request.dataset
            || !record.blobs.contains(&dataset.csv)
            || dataset.csv.bytes > MODEL_BYTES.min(state.config.limits.max_dataset_bytes.value)
        {
            return Err(bad());
        }
        let bytes = state
            .store
            .read_blob(&dataset.csv)
            .map_err(|_| integrity())?;
        let table =
            jope_runtime::table::parse_csv(&bytes, &Default::default()).map_err(|_| bad())?;
        let (values, _) = jope_runtime::budget::Budget::new(None, 1)
            .install(|| model.math_pipeline()?.predict_table(&table))
            .map_err(|_| unavailable())?
            .map_err(|_| bad())?;
        if values.len() != table.rows.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(error(
                ErrorCode::NumericalFailure,
                "Frozen model produced invalid predictions.",
            ));
        }
        let mut out = String::from("prediction\n");
        for value in values {
            out.push_str(&value.to_string());
            out.push('\n');
        }
        if out.len() as u64 > MODEL_BYTES {
            return Err(bad());
        }
        Ok(out)
    })();
    let output = match result {
        Ok(v) => v,
        Err(e) => return response(&e),
    };
    match state.store.execute_idempotent(&identity, move |_| {
        CommandOutcome::json(200, Value::String(output))
    }) {
        Ok(v) => csv(v),
        Err(e) => e.into_response(),
    }
}
pub(super) async fn predict_model(
    Extension(state): Extension<DatasetState>,
    request: Request<Body>,
) -> Response {
    let (body, identity) = match command::<PredictRequest>(request, 15).await {
        Ok(v) => v,
        Err(e) => return response(&e),
    };
    match tokio::task::spawn_blocking(move || predict(&state, body, identity)).await {
        Ok(v) => v,
        Err(_) => response(&integrity()),
    }
}
pub(super) async fn download(
    Extension(state): Extension<DatasetState>,
    path: std::result::Result<Path<String>, axum::extract::rejection::PathRejection>,
) -> Response {
    let Ok(Path(raw)) = path else {
        return response(&bad());
    };
    let Ok(id) = raw.parse::<ArtifactId>() else {
        return response(&bad());
    };
    let result = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let record = state
            .store
            .get(&RecordKey::Artifact(id))
            .map_err(|_| integrity())?
            .ok_or_else(missing)?;
        if !matches!(
            record.data["kind"].as_str(),
            Some("model_export" | "model_parity")
        ) {
            return Err(missing());
        }
        let blob: BlobRef =
            serde_json::from_value(record.data["archive"].clone()).map_err(|_| integrity())?;
        if blob.bytes > ARCHIVE_BYTES || !record.blobs.contains(&blob) {
            return Err(integrity());
        }
        state.store.read_blob(&blob).map_err(|_| integrity())
    })
    .await;
    match result {
        Ok(Ok(bytes)) => {
            let digest = hash(&bytes);
            (
                [
                    (header::CONTENT_TYPE.as_str(), "application/octet-stream"),
                    ("x-jope-artifact-sha256", digest.as_str()),
                ],
                bytes,
            )
                .into_response()
        }
        Ok(Err(e)) => response(&e),
        Err(_) => response(&integrity()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, dataset::register, store::Store};
    use axum::body::to_bytes;
    use jope_api::decimal::DecimalU64;
    use jope_runtime::{
        math_pipeline::{INPUT_PART, PIPELINE_PART},
        model_object::{Part, Producer, MATH_PIPELINE, SCHEMA},
    };
    fn open_state(path: &std::path::Path) -> DatasetState {
        DatasetState {
            store: Arc::new(Store::open(path).unwrap()),
            config: Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap(),
            home: path.into(),
        }
    }
    fn fixture() -> (
        tempfile::TempDir,
        DatasetState,
        RunId,
        ArtifactId,
        PredictRequest,
    ) {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/mcp05-model-tests");
        std::fs::create_dir_all(&root).unwrap();
        let dir = tempfile::tempdir_in(root).unwrap();
        let state = open_state(dir.path());
        let run = RunId::new();
        let mut input: Value = serde_json::from_slice(include_bytes!(
            "../../../jope-runtime/tests/fixtures/math-pipeline/input.json"
        ))
        .unwrap();
        input["prediction_version"] = json!(2);
        let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            (INPUT_PART.into(), serde_json::to_vec(&input).unwrap()),
            (
                PIPELINE_PART.into(),
                include_bytes!("../../../jope-runtime/tests/fixtures/math-pipeline/pipeline.json")
                    .to_vec(),
            ),
        ]);
        let model = ModelObject::new(
            Manifest {
                schema: SCHEMA.into(),
                minimum_reader_version: 1,
                format: MATH_PIPELINE.into(),
                producer: Producer::default(),
                primary: PIPELINE_PART.into(),
                required_capabilities: vec![],
                files: parts
                    .iter()
                    .map(|(p, b)| Part::new(p, "application/json", b))
                    .collect(),
                metadata: json!({"run_id":run}),
            },
            parts,
            MODEL_BYTES,
        )
        .unwrap();
        let id: ArtifactId = format!("art_{}", &model.id()[7..31]).parse().unwrap();
        let refs = std::iter::once(model.manifest_bytes())
            .chain(model.parts().values().map(Vec::as_slice))
            .map(|b| state.store.publish_blob(b).unwrap())
            .collect();
        let snapshot = RunSnapshot {
            run_id: run.clone(),
            revision: DecimalU64(1),
            state: RunState::Completed,
            phase: RunPhase::Finalizing,
            pending: None,
            effort: Effort::Fast,
            budget_used: BudgetUsed {
                evaluations: DecimalU64(1),
                wall_seconds: "0.1".into(),
            },
            budget_total: BudgetTotal {
                evaluations: DecimalU64(1),
                wall_seconds: DecimalU64(5),
            },
            best: Some(BestAlgorithm {
                algorithm_id: "alg_0001".parse().unwrap(),
                metric: "rmse".into(),
                cv_value: 0.25,
                model_artifact: Some(id.clone()),
            }),
            event_watermark: DecimalU64(0),
            warnings: vec![],
            completion_reason: Some(CompletionReason::BudgetEvaluations),
        };
        state
            .store
            .transaction(|tx| {
                tx.put(
                    &RecordKey::Artifact(id.clone()),
                    &Record::metadata(
                        json!({"kind":"finalized_model","model_digest":model.id()}),
                        refs,
                    ),
                )?;
                tx.put(
                    &RecordKey::Run(run.clone()),
                    &Record::run(
                        json!({"snapshot":snapshot,"checkpoint":{}}),
                        vec![],
                        RunState::Completed,
                        1,
                    ),
                )
            })
            .unwrap();
        let input = model.math_pipeline().unwrap().input;
        let mut csv = input
            .columns
            .iter()
            .map(|c| c.name.clone())
            .chain(std::iter::once(input.target.name.clone()))
            .collect::<Vec<_>>()
            .join(",");
        csv.push('\n');
        for i in 0..4 {
            csv.push_str(
                &input
                    .columns
                    .iter()
                    .map(|c| (c.min + i as f64).to_string())
                    .chain(std::iter::once((i + 1).to_string()))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            csv.push('\n');
        }
        let ds = register::register(&state.store, csv.as_bytes(), &Default::default()).unwrap();
        let request = PredictRequest {
            model_artifact: id.clone(),
            dataset: ds.id,
        };
        (dir, state, run, id, request)
    }
    fn identity<T: Serialize>(key: &str, operation: u8, target: &str, body: &T) -> RequestIdentity {
        RequestIdentity::new(
            "local-engine-owner",
            key,
            operation,
            target,
            None,
            &serde_json::to_value(body).unwrap(),
        )
        .unwrap()
    }
    fn export_request() -> ExportRequest {
        ExportRequest {
            algorithm_id: "alg_0001".parse().unwrap(),
            kind: ExportKind::Bundle,
            languages: vec![Language::Rust],
            require_all_languages: true,
            refit: Refit::Fit,
        }
    }
    async fn bytes(r: Response) -> Vec<u8> {
        to_bytes(r.into_body(), ARCHIVE_BYTES as usize)
            .await
            .unwrap()
            .to_vec()
    }
    #[tokio::test]
    async fn frozen_export_download_and_run_based_predict_use_one_model() {
        let (_dir, state, run, _, request) = fixture();
        let a = algorithm(
            &state,
            run.clone(),
            Some("alg_0001".parse().unwrap()),
            AlgorithmQuery::default(),
            false,
        )
        .unwrap();
        assert_eq!(a["algorithm_id"], "alg_0001");
        assert_eq!(a["model_artifact"], json!(request.model_artifact));
        let body = export_request();
        let target = format!("/v1/runs/{run}/exports");
        let r = export(
            &state,
            run.clone(),
            body.clone(),
            identity("export", 13, &target, &body),
        );
        assert_eq!(r.status(), 200);
        let manifest: ExportManifest = serde_json::from_slice(&bytes(r).await).unwrap();
        assert_eq!(manifest.parity_status, ParityStatus::Unavailable);
        assert!(manifest.parity_receipt.is_none());
        let downloaded = download(
            Extension(state.clone()),
            Ok(Path(manifest.artifact_id.to_string())),
        )
        .await;
        assert_eq!(downloaded.status(), 200);
        let digest = downloaded.headers()["x-jope-artifact-sha256"]
            .to_str()
            .unwrap()
            .to_owned();
        let archive = bytes(downloaded).await;
        assert_eq!(digest, hash(&archive));
        assert_eq!(
            manifest.model_digest,
            object(&state, &request.model_artifact).unwrap().0.id()
        );
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).unwrap();
        assert!(zip.by_name("model/manifest.json").is_ok());
        assert!(zip.by_name("src/rust/predictor.rs").is_ok());
        let exported = PredictRequest {
            model_artifact: manifest.artifact_id,
            dataset: request.dataset.clone(),
        };
        let prediction = predict(
            &state,
            exported.clone(),
            identity("predict-export", 15, "/v1/predictions", &exported),
        );
        assert_eq!(prediction.status(), 200);
        assert_eq!(prediction.headers()[header::CONTENT_TYPE], "text/csv");
        let output = bytes(prediction).await;
        let table = jope_runtime::table::parse_csv(&output, &Default::default()).unwrap();
        assert_eq!(table.rows.len(), 4);
        assert!(table.rows.iter().all(|r| r[0]
            .as_ref()
            .unwrap()
            .parse::<f64>()
            .unwrap()
            .is_finite()));
        let original = predict(
            &state,
            request.clone(),
            identity("predict-original", 15, "/v1/predictions", &request),
        );
        assert_eq!(bytes(original).await, output);
    }
    #[tokio::test]
    async fn export_and_prediction_replay_after_store_reopen() {
        let (dir, state, run, _, request) = fixture();
        let body = export_request();
        let target = format!("/v1/runs/{run}/exports");
        let first = bytes(export(
            &state,
            run.clone(),
            body.clone(),
            identity("export", 13, &target, &body),
        ))
        .await;
        let predicted = bytes(predict(
            &state,
            request.clone(),
            identity("predict", 15, "/v1/predictions", &request),
        ))
        .await;
        drop(state);
        let reopened = open_state(dir.path());
        assert_eq!(
            bytes(export(
                &reopened,
                run,
                body.clone(),
                identity("export", 13, &target, &body)
            ))
            .await,
            first
        );
        assert_eq!(
            bytes(predict(
                &reopened,
                request.clone(),
                identity("predict", 15, "/v1/predictions", &request)
            ))
            .await,
            predicted
        );
        let mut changed = request;
        changed.dataset = "ds_00112233445566778899aabb".parse().unwrap();
        assert_eq!(
            predict(
                &reopened,
                changed.clone(),
                identity("predict", 15, "/v1/predictions", &changed)
            )
            .status(),
            409
        );
    }
    #[tokio::test]
    async fn changed_export_body_conflicts_and_unsupported_language_has_no_receipt() {
        let (_dir, state, run, _, _) = fixture();
        let body = export_request();
        let target = format!("/v1/runs/{run}/exports");
        assert_eq!(
            export(
                &state,
                run.clone(),
                body.clone(),
                identity("same", 13, &target, &body)
            )
            .status(),
            200
        );
        let mut unsupported = body;
        unsupported.languages = vec![Language::Typescript];
        assert_eq!(
            export(
                &state,
                run.clone(),
                unsupported.clone(),
                identity("same", 13, &target, &unsupported)
            )
            .status(),
            409
        );
        assert_eq!(
            export(
                &state,
                run,
                unsupported.clone(),
                identity("unsupported", 13, &target, &unsupported)
            )
            .status(),
            422
        );
        assert!(state
            .store
            .get(&RecordKey::Command {
                principal: "local-engine-owner".into(),
                idempotency_key: "unsupported".into()
            })
            .unwrap()
            .is_none());
    }
    #[test]
    fn unknown_algorithm_and_corrupt_model_are_not_false_successes() {
        let (_dir, state, run, id, _) = fixture();
        assert_eq!(
            champion(&state, &run, Some(&"alg_9999".parse().unwrap()))
                .unwrap_err()
                .code,
            ErrorCode::NotFound
        );
        let record = state
            .store
            .get(&RecordKey::Artifact(id.clone()))
            .unwrap()
            .unwrap();
        let path = state.store.verified_blob_path(&record.blobs[1]).unwrap();
        std::fs::write(path, b"corrupted fixture").unwrap();
        assert_eq!(
            object(&state, &id).unwrap_err().code,
            ErrorCode::ModelIntegrityFailed
        );
    }
    #[tokio::test]
    async fn mutation_key_and_duplicate_json_refuse_before_effects() {
        for body in [
            r#"{"model_artifact":"art_00112233445566778899aabb","model_artifact":"art_ffeeddccbbaa998877665544","dataset":"ds_00112233445566778899aabb"}"#,
            r#"{}"#,
        ] {
            let req = Request::builder()
                .uri("/v1/predictions")
                .header("idempotency-key", "key")
                .body(Body::from(body))
                .unwrap();
            assert!(command::<PredictRequest>(req, 15).await.is_err());
        }
        let req = Request::builder()
            .uri("/v1/predictions")
            .body(Body::from("{}"))
            .unwrap();
        assert!(command::<PredictRequest>(req, 15).await.is_err());
    }
    #[tokio::test]
    async fn measured_export_parity_rechecks_model_sources_mode_and_all_rows() {
        let (_dir, state, run, id, _) = fixture();
        let (model, original) = object(&state, &id).unwrap();
        let pipeline = model.math_pipeline().unwrap();
        let rows: Vec<Vec<Option<String>>> = (0..64)
            .map(|i| {
                pipeline
                    .input
                    .columns
                    .iter()
                    .map(|c| Some((c.min + (c.max - c.min) * (i as f64 / 63.0)).to_string()))
                    .collect()
            })
            .collect();
        let borrowed = rows
            .iter()
            .map(|r| r.iter().map(Option::as_deref).collect())
            .collect::<Vec<_>>();
        let engine = pipeline.predict_cells(&borrowed).unwrap();
        // Controlled observations exercise receipt validation only. Actual toolchain
        // executions belong to the external qualification receipt, not this unit test.
        let receipt = bundle::MeasuredModelParity {
            schema: "jope.measured-model-parity/v1".into(),
            model_digest: model.id().into(),
            emitter_version: env!("CARGO_PKG_VERSION").into(),
            refit: Refit::Fit,
            source_sha256: BTreeMap::from([
                (
                    "python".into(),
                    hash(&model.export_source("python").unwrap()),
                ),
                ("rust".into(), hash(&model.export_source("rust").unwrap())),
            ]),
            toolchains: BTreeMap::from([
                ("python".into(), "controlled validator fixture".into()),
                ("rust".into(), "controlled validator fixture".into()),
            ]),
            rows,
            engine: engine.clone(),
            observed: BTreeMap::from([("python".into(), engine.clone()), ("rust".into(), engine)]),
        };
        for case in 0..5 {
            let mut bad = receipt.clone();
            match case {
                0 => bad.source_sha256.insert("python".into(), "0".repeat(64)),
                1 => {
                    bad.observed.get_mut("rust").unwrap()[0] += 1.;
                    None
                }
                2 => {
                    bad.rows.pop();
                    None
                }
                3 => {
                    bad.refit = Refit::FitHoldout;
                    None
                }
                _ => {
                    bad.model_digest = "different-model".into();
                    None
                }
            };
            assert!(bundle::record_model_parity(&state.store, &id, &model, &bad).is_err());
            assert_eq!(
                state
                    .store
                    .get(&RecordKey::Artifact(id.clone()))
                    .unwrap()
                    .unwrap(),
                original
            );
        }
        let parity_id = bundle::record_model_parity(&state.store, &id, &model, &receipt).unwrap();
        let mut body = export_request();
        body.languages = vec![Language::Python, Language::Rust];
        let target = format!("/v1/runs/{run}/exports");
        let response = export(
            &state,
            run,
            body.clone(),
            identity("measured-export", 13, &target, &body),
        );
        assert_eq!(response.status(), 200);
        let manifest: ExportManifest = serde_json::from_slice(&bytes(response).await).unwrap();
        assert_eq!(manifest.parity_status, ParityStatus::Passed);
        assert_eq!(manifest.parity_receipt, Some(parity_id.clone()));
        assert!(manifest.qualified_set_id.starts_with("measured-model:"));
        let downloaded = download(
            Extension(state.clone()),
            Ok(Path(manifest.artifact_id.to_string())),
        )
        .await;
        let mut archive =
            zip::ZipArchive::new(std::io::Cursor::new(bytes(downloaded).await)).unwrap();
        assert!(archive.by_name("parity_receipt.json").is_ok());
        assert!(archive.by_name("src/python/predictor.py").is_ok());
        let proof = download(Extension(state), Ok(Path(parity_id.to_string()))).await;
        assert_eq!(proof.status(), 200);
    }
    #[tokio::test]
    async fn export_refit_matches_the_completed_model_without_fitting_again() {
        let (_dir, state, run, id, _) = fixture();
        let mut body = export_request();
        body.refit = Refit::FitHoldout;
        let target = format!("/v1/runs/{run}/exports");
        assert_eq!(
            export(
                &state,
                run.clone(),
                body.clone(),
                identity("wrong-mode", 13, &target, &body)
            )
            .status(),
            422
        );
        state
            .store
            .transaction(|tx| {
                let key = RecordKey::Artifact(id.clone());
                let mut r = tx.get(&key)?.unwrap();
                r.data["refit"] = json!("fit_holdout");
                tx.put(&key, &r)
            })
            .unwrap();
        let response = export(
            &state,
            run,
            body.clone(),
            identity("completed-mode", 13, &target, &body),
        );
        assert_eq!(response.status(), 200);
        let manifest: ExportManifest = serde_json::from_slice(&bytes(response).await).unwrap();
        assert_eq!(manifest.refit, Refit::FitHoldout);
        assert_eq!(manifest.parity_status, ParityStatus::Unavailable);
    }
    #[test]
    fn explicit_fit_uses_only_the_model_bound_by_the_completed_run_receipt() {
        let (_dir, state, run, fit_id, _) = fixture();
        let (fit, _) = object(&state, &fit_id).unwrap();
        // Controlled selector fixture; operational proof uses real native fits.
        let mut manifest = fit.manifest().clone();
        manifest.metadata["refit"] = json!("fit_holdout");
        let full = ModelObject::new(manifest, fit.parts().clone(), MODEL_BYTES).unwrap();
        let full_id: ArtifactId = format!("art_{}", &full.id()[7..31]).parse().unwrap();
        let refs = std::iter::once(full.manifest_bytes())
            .chain(full.parts().values().map(Vec::as_slice))
            .map(|b| state.store.publish_blob(b).unwrap())
            .collect();
        let completed = json!({"run_id":run,"algorithm_id":"alg_0001","model_artifact":full_id,"model_digest":full.id(),"model_path":"unused","refit":"fit_holdout","scores":{"cv":0.25,"holdout":0.2,"test":null},"completion_reason":"budget_evaluations","budget_bound_by":"evaluations","budget_used":{"evaluations":"1","wall_seconds":"0.1"},"nondeterminism":{"count":"0","experiment_ids":[]},"holdout_model_digest":fit.id(),"holdout_scope":"holdout_scored_with_fit_only_model","test_scope":null,"fit_rows":4,"holdout_rows":1,"test_rows":0,"audit":null});
        let _: crate::discover::finalize::FinalizationReceipt =
            serde_json::from_value(completed.clone()).unwrap();
        let command = RecordKey::Command {
            principal: "discover.finalization.v1".into(),
            idempotency_key: run.to_string(),
        };
        state.store.transaction(|tx|{
            let fit_key=RecordKey::Artifact(fit_id.clone());
            let mut fit_record=tx.get(&fit_key)?.unwrap();
            fit_record.data["refit"]=json!(Refit::Fit);
            tx.put(&fit_key,&fit_record)?;
            tx.put(&RecordKey::Artifact(full_id.clone()),&Record::metadata(json!({"kind":"finalized_model","model_digest":full.id(),"refit":"fit_holdout"}),refs))?;
            let key=RecordKey::Run(run.clone());let mut record=tx.get(&key)?.unwrap();record.data["snapshot"]["best"]["model_artifact"]=json!(full_id);tx.put(&key,&record)?;
            tx.put(&command,&Record::metadata(json!({"state":"completed","receipt":completed}),vec![]))
        }).unwrap();
        let alg = "alg_0001".parse().unwrap();
        assert_eq!(
            export_champion(&state, &run, &alg, Refit::Fit)
                .unwrap()
                .0
                .id(),
            fit.id()
        );
        assert_eq!(
            export_champion(&state, &run, &alg, Refit::FitHoldout)
                .unwrap()
                .0
                .id(),
            full.id()
        );
        state
            .store
            .transaction(|tx| {
                let mut record = tx.get(&command)?.unwrap();
                record.data["receipt"]["run_id"] = json!(RunId::new());
                tx.put(&command, &record)
            })
            .unwrap();
        assert_eq!(
            export_champion(&state, &run, &alg, Refit::Fit)
                .unwrap_err()
                .code,
            ErrorCode::ModelIntegrityFailed
        );
    }
}
