//! Authenticated HTTP adapter for bounded dataset registration.

use super::auth::problem;
use crate::{
    dataset::register::{self, RegistrationError, RegistrationLimits},
    idempotency::{CommandOutcome, RequestIdentity},
};
use axum::{
    body::{to_bytes, Body},
    extract::{Extension, Query},
    http::{header, Request, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use jope_api::{
    decimal::DecimalU64,
    error::{ErrorCode, Problem},
    registry::DatasetRegistration,
    ColumnProfile, DatasetProfile, TargetSuggestion, Task, Warning,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

pub(super) async fn register(
    Extension(state): Extension<crate::engine::DatasetState>,
    request: Request<Body>,
) -> Response {
    let headers = request.headers().clone();
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let key = match headers
        .get("idempotency-key")
        .and_then(|value| value.to_str().ok())
    {
        Some(key) => key,
        None => return invalid_input("Idempotency-Key is required.", "Idempotency-Key"),
    };
    let limit = state.config.limits.max_dataset_bytes.value;
    let reserve = state.config.limits.disk_reserve.value;

    let parsed = if content_type.split(';').next().unwrap_or("").trim() == "text/csv" {
        let Query(query) = match Query::<DatasetQuery>::try_from_uri(request.uri()) {
            Ok(query) => query,
            Err(_) => {
                return invalid_input("Query must match dataset registration parameters.", "query");
            }
        };
        let (temp, bytes, digest) =
            match stage_csv(request.into_body(), &state.home, limit, reserve).await {
                Ok(value) => value,
                Err(response) => return response,
            };
        let limits = RegistrationLimits {
            max_dataset_bytes: limit,
            ..RegistrationLimits::default()
        };
        let staged = bytes;
        let prepared = match register::prepare_file_with_admission(
            &state.store,
            temp.path(),
            &limits,
            |next| admit_disk(&state.home, reserve, staged.saturating_add(next)),
        ) {
            Ok(value) => value,
            Err(error) => return registration_error(error),
        };
        (prepared, query.name, digest, bytes)
    } else if content_type.split(';').next().unwrap_or("").trim() == "application/json" {
        let bytes = match to_bytes(request.into_body(), 8 * 1024 * 1024).await {
            Ok(value) => value,
            Err(_) => return invalid_input("Request body is too large or unreadable.", "body"),
        };
        let registration: DatasetRegistration = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => return invalid_input("Body must match DatasetRegistration.", "body"),
        };
        match registration {
            DatasetRegistration::CsvText { csv_text, name } => {
                let csv = csv_text.into_bytes();
                if !admit_disk(&state.home, reserve, csv.len() as u64) {
                    return disk_reserve_error();
                }
                let digest = Sha256::digest(&csv).to_vec();
                let limits = RegistrationLimits {
                    max_dataset_bytes: limit,
                    ..RegistrationLimits::default()
                };
                let prepared = match register::prepare_with_admission(
                    &state.store,
                    csv.as_slice(),
                    &limits,
                    |next| admit_disk(&state.home, reserve, next),
                ) {
                    Ok(value) => value,
                    Err(error) => return registration_error(error),
                };
                (prepared, name, digest, csv.len() as u64)
            }
            DatasetRegistration::Path { path, name } => {
                let path = PathBuf::from(path);
                let metadata = match std::fs::metadata(&path) {
                    Ok(metadata) if metadata.is_file() => metadata,
                    _ => return unauthorized_path(),
                };
                if metadata.len() > limit {
                    return dataset_invalid("max_dataset_bytes");
                }
                if !admit_disk(&state.home, reserve, metadata.len()) {
                    return disk_reserve_error();
                }
                let limits = RegistrationLimits {
                    max_dataset_bytes: limit,
                    ..RegistrationLimits::default()
                };
                let prepared = match register::prepare_file_with_admission(
                    &state.store,
                    &path,
                    &limits,
                    |next| admit_disk(&state.home, reserve, next),
                ) {
                    Ok(value) => value,
                    Err(error) => return registration_error(error),
                };
                let digest = prepared.dataset.csv.sha256.to_vec();
                (prepared, name, digest, metadata.len())
            }
        }
    } else {
        return invalid_input(
            "Content-Type must be text/csv or application/json.",
            "Content-Type",
        );
    };

    let (prepared, name, digest, byte_count) = parsed;
    if !admit_disk(&state.home, reserve, byte_count) {
        return disk_reserve_error();
    }
    let profile = to_api_profile(&prepared.dataset, &prepared.profile, name);
    let body = match serde_json::to_value(profile) {
        Ok(body) => body,
        Err(_) => return internal_error(),
    };
    let identity_body = json!({
        "sha256": digest.iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "name": body.get("name").cloned().unwrap_or(Value::Null),
    });
    let identity = match RequestIdentity::new(
        "local-engine-owner",
        key,
        2,
        "/v1/datasets",
        headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()),
        &identity_body,
    ) {
        Ok(identity) => identity,
        Err(error) => return error.into_response(),
    };
    let outcome = match CommandOutcome::json(201, body) {
        Ok(outcome) => outcome,
        Err(_) => return internal_error(),
    };
    match state.store.execute_idempotent(&identity, |write| {
        prepared.commit(write)?;
        Ok(outcome)
    }) {
        Ok(receipt) => receipt.into_response(),
        Err(error) => error.into_response(),
    }
}

#[derive(Deserialize)]
struct DatasetQuery {
    name: Option<String>,
}

async fn stage_csv(
    body: Body,
    home: &std::path::Path,
    limit: u64,
    reserve: u64,
) -> Result<(tempfile::NamedTempFile, u64, Vec<u8>), Response> {
    let temp = tempfile::NamedTempFile::new_in(home).map_err(|_| disk_reserve_error())?;
    let mut output = tokio::fs::File::from_std(temp.reopen().map_err(|_| internal_error())?);
    let mut stream = body.into_data_stream();
    let mut bytes = 0_u64;
    let mut hash = Sha256::new();
    while let Some(chunk) = futures_util::StreamExt::next(&mut stream).await {
        let chunk = chunk.map_err(|_| invalid_input("CSV upload stream failed.", "body"))?;
        bytes = bytes
            .checked_add(chunk.len() as u64)
            .ok_or_else(|| dataset_invalid("max_dataset_bytes"))?;
        if bytes > limit {
            return Err(dataset_invalid("max_dataset_bytes"));
        }
        if !admit_disk(home, reserve, bytes) {
            return Err(disk_reserve_error());
        }
        output
            .write_all(&chunk)
            .await
            .map_err(|_| disk_reserve_error())?;
        hash.update(&chunk);
    }
    output.flush().await.map_err(|_| disk_reserve_error())?;
    output.sync_all().await.map_err(|_| disk_reserve_error())?;
    Ok((temp, bytes, hash.finalize().to_vec()))
}

pub(super) fn to_api_profile(
    dataset: &register::RegisteredDataset,
    profile: &crate::dataset::profile::DatasetProfile,
    name: Option<String>,
) -> DatasetProfile {
    let rows = profile.rows as u64;
    let columns = profile
        .columns
        .iter()
        .map(|column| ColumnProfile {
            name: column.name.clone(),
            data_type: match column.kind {
                jope_runtime::table::ColumnKind::Numeric => "numeric",
                jope_runtime::table::ColumnKind::Categorical => "categorical",
            }
            .into(),
            missing: DecimalU64((column.missing_rate * rows as f64).round() as u64),
            distinct: DecimalU64(column.distinct_count as u64),
        })
        .collect();
    let target_suggestions = profile
        .suggested_targets
        .iter()
        .map(|column| TargetSuggestion {
            column: column.clone(),
            task: Task::Auto,
            reason: "Column name is a non-binding target hint; select the task explicitly.".into(),
        })
        .collect();
    DatasetProfile {
        dataset_id: dataset.id.clone(),
        name: name.unwrap_or_else(|| dataset.id.to_string()),
        bytes: DecimalU64(dataset.csv.bytes),
        rows: DecimalU64(rows),
        columns,
        target_suggestions,
        warnings: Vec::<Warning>::new(),
    }
}

fn registration_error(error: RegistrationError) -> Response {
    match error {
        RegistrationError::Invalid(problem) => {
            (StatusCode::UNPROCESSABLE_ENTITY, Json(*problem)).into_response()
        }
        RegistrationError::Admission => disk_reserve_error(),
        RegistrationError::Store(_) => internal_error(),
    }
}

fn admit_disk(home: &std::path::Path, reserve: u64, incoming: u64) -> bool {
    disk_free(home).is_some_and(|free| free >= reserve.saturating_add(incoming))
}

#[cfg(unix)]
fn disk_free(path: &std::path::Path) -> Option<u64> {
    let stats = nix::sys::statvfs::statvfs(path).ok()?;
    Some((stats.blocks_available() as u64).saturating_mul(stats.fragment_size() as u64))
}

#[cfg(not(unix))]
fn disk_free(_: &std::path::Path) -> Option<u64> {
    None
}

fn unauthorized_path() -> Response {
    problem(
        ErrorCode::InvalidInput,
        "Local dataset path is unavailable or not a regular file.",
        "Use a readable regular CSV file on the local engine host.",
        Some("path"),
    )
}

fn disk_reserve_error() -> Response {
    problem(
        ErrorCode::BackendUnavailable,
        "Dataset registration would consume the configured disk reserve.",
        "Free disk space or lower limits.disk_reserve before registering.",
        Some("limits.disk_reserve"),
    )
}

fn dataset_invalid(field: &str) -> Response {
    let detail = format!("Dataset violates {field}.");
    let p = Problem {
        r#type: "about:blank".into(),
        title: "Dataset invalid".into(),
        status: 422,
        detail,
        instance: None,
        code: ErrorCode::DatasetInvalid,
        retriable: false,
        remedy: format!("Correct the CSV or explicitly adjust {field} before registering again."),
        field: Some(field.into()),
    };
    (StatusCode::UNPROCESSABLE_ENTITY, Json(p)).into_response()
}

fn invalid_input(detail: &str, field: &str) -> Response {
    problem(
        ErrorCode::InvalidInput,
        detail,
        "Correct the request and retry.",
        Some(field),
    )
}

fn internal_error() -> Response {
    problem(
        ErrorCode::IoFailure,
        "Dataset registration could not be committed.",
        "Retry with the same Idempotency-Key after checking the store.",
        None,
    )
}

#[cfg(test)]
#[path = "datasets/tests.rs"]
mod tests;
