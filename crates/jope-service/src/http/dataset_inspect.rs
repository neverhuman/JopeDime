//! Read-only profiles from registered, content-verified CSV blobs.

use super::auth::problem;
use crate::{dataset::register::RegisteredDataset, engine::DatasetState, store::RecordKey};
use axum::{
    extract::{Extension, Path, Query},
    response::{IntoResponse, Response},
    Json,
};
use jope_api::{error::ErrorCode, ids::DatasetId, registry::DatasetPage, DatasetProfile};

fn invalid(detail: &str) -> Response {
    problem(
        ErrorCode::InvalidInput,
        detail,
        "Use a valid dataset ID and list limit from 1 to 256.",
        None,
    )
}

fn unavailable() -> Response {
    problem(
        ErrorCode::IoFailure,
        "Stored dataset could not be profiled.",
        "Check the dataset store and its immutable CSV content.",
        None,
    )
}

fn profile(state: &DatasetState, id: &DatasetId) -> Result<DatasetProfile, Response> {
    let record = state
        .store
        .get(&RecordKey::Dataset(id.clone()))
        .map_err(|_| unavailable())?
        .ok_or_else(|| {
            problem(
                ErrorCode::NotFound,
                "Dataset was not found.",
                "List registered datasets and use an existing ID.",
                Some("dataset_id"),
            )
        })?;
    let dataset: RegisteredDataset =
        serde_json::from_value(record.data).map_err(|_| unavailable())?;
    if dataset.id != *id
        || !record.blobs.contains(&dataset.csv)
        || !record.blobs.contains(&dataset.matrix)
    {
        return Err(unavailable());
    }
    let limit = state.config.limits.max_dataset_bytes.value;
    if dataset.csv.bytes > limit {
        return Err(problem(
            ErrorCode::DatasetInvalid,
            "Stored dataset exceeds max_dataset_bytes.",
            "Explicitly adjust the dataset byte limit before inspection.",
            Some("limits.max_dataset_bytes"),
        ));
    }
    let csv = state
        .store
        .read_blob(&dataset.csv)
        .map_err(|_| unavailable())?;
    let limits = jope_runtime::limits::Limits {
        max_input_bytes: limit,
        ..Default::default()
    };
    let table = jope_runtime::table::parse_csv(&csv, &limits).map_err(|_| unavailable())?;
    if table.rows.len() as u64 != dataset.rows || table.headers.len() as u64 != dataset.columns {
        return Err(unavailable());
    }
    let stats = crate::dataset::profile::profile(&table);
    // Registration metadata predates display aliases; use its canonical ID.
    Ok(super::datasets::to_api_profile(&dataset, &stats, None))
}

pub(super) async fn get(
    Extension(state): Extension<DatasetState>,
    path: Result<Path<String>, axum::extract::rejection::PathRejection>,
) -> Response {
    let Ok(Path(raw)) = path else {
        return invalid("Invalid dataset path.");
    };
    let Ok(id) = raw.parse::<DatasetId>() else {
        return invalid("Invalid dataset ID.");
    };
    match tokio::task::spawn_blocking(move || profile(&state, &id)).await {
        Ok(Ok(profile)) => Json(profile).into_response(),
        Ok(Err(response)) => response,
        Err(_) => unavailable(),
    }
}

#[derive(Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ListQuery {
    after: Option<String>,
    limit: Option<usize>,
}

pub(super) async fn list(
    Extension(state): Extension<DatasetState>,
    query: Result<Query<ListQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let Ok(Query(query)) = query else {
        return invalid("Invalid dataset list query.");
    };
    let limit = query.limit.unwrap_or(100);
    if !(1..=256).contains(&limit) {
        return invalid("Invalid dataset list limit.");
    }
    let after = match query.after.map(|raw| raw.parse::<DatasetId>()).transpose() {
        Ok(id) => id,
        Err(_) => return invalid("Invalid dataset list cursor."),
    };
    match tokio::task::spawn_blocking(move || -> Result<DatasetPage, Response> {
        let mut ids = state
            .store
            .dataset_ids_after(after.as_ref(), limit + 1)
            .map_err(|_| unavailable())?;
        let more = ids.len() > limit;
        ids.truncate(limit);
        let next_after = if more {
            ids.last().map(ToString::to_string)
        } else {
            None
        };
        let datasets = ids
            .iter()
            .map(|id| profile(&state, id))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DatasetPage {
            datasets,
            next_after,
        })
    })
    .await
    {
        Ok(Ok(page)) => Json(page).into_response(),
        Ok(Err(response)) => response,
        Err(_) => unavailable(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, dataset::register, store::Store};
    use axum::{
        body::{to_bytes, Body},
        http::{header, Request, StatusCode},
    };
    use serde_json::{json, Value};
    use std::{collections::BTreeMap, sync::Arc};
    use tower::ServiceExt;

    const CSV: &[u8] = b"target,feature\n1,red\n2,blue\n,red\n";

    fn state(home: &std::path::Path) -> DatasetState {
        DatasetState {
            store: Arc::new(Store::open(home).unwrap()),
            config: Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap(),
            home: home.to_path_buf(),
        }
    }
    fn app(state: DatasetState) -> axum::Router {
        let auth = super::super::auth::LocalAuth::new(
            super::super::auth::BearerToken::from_hex(&"ab".repeat(32)).unwrap(),
            31234,
        )
        .unwrap();
        super::super::router::router_with_state(auth, state, None)
    }
    fn request(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
            .body(Body::empty())
            .unwrap()
    }
    async fn read(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap()).unwrap()
    }

    #[tokio::test]
    async fn inspect_returns_real_statistics_after_store_reopen() {
        let home = tempfile::tempdir().unwrap();
        let state = state(home.path());
        let saved = register::register(&state.store, CSV, &Default::default()).unwrap();
        drop(state);
        let state = self::state(home.path());
        let app = app(state.clone());
        let response = app
            .clone()
            .oneshot(request(&format!("/v1/datasets/{}", saved.id)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["jope-api-version"], "1.0");
        let body = read(response).await;
        let profile: DatasetProfile = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(profile.dataset_id, saved.id);
        assert_eq!(profile.bytes.0, CSV.len() as u64);
        assert_eq!(profile.rows.0, 3);
        assert_eq!(profile.columns.len(), 2);
        assert_eq!(profile.columns[0].data_type, "numeric");
        assert_eq!(profile.columns[0].missing.0, 1);
        assert_eq!(profile.columns[0].distinct.0, 2);
        assert_eq!(profile.columns[1].data_type, "categorical");
        assert_eq!(profile.columns[1].distinct.0, 2);
        assert_eq!(profile.target_suggestions[0].column, "target");
        assert!(!body.to_string().contains("red"));
        assert_eq!(
            state.store.dataset_ids_after(None, 257).unwrap(),
            vec![saved.id]
        );
        let list = read(app.oneshot(request("/v1/datasets")).await.unwrap()).await;
        assert_eq!(list, json!({"datasets":[body],"next_after":null}));
    }

    #[tokio::test]
    async fn list_paginates_exclusively_and_empty_store_has_no_phantom_profile() {
        let home = tempfile::tempdir().unwrap();
        let state = state(home.path());
        let app = app(state.clone());
        assert_eq!(
            read(app.clone().oneshot(request("/v1/datasets")).await.unwrap()).await,
            json!({"datasets":[],"next_after":null})
        );
        let mut ids = Vec::new();
        for csv in [CSV, b"target,feature\n3,green\n".as_slice()] {
            ids.push(
                register::register(&state.store, csv, &Default::default())
                    .unwrap()
                    .id,
            );
        }
        ids.sort_by_key(ToString::to_string);
        let first = read(
            app.clone()
                .oneshot(request("/v1/datasets?limit=1"))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(first["datasets"].as_array().unwrap().len(), 1);
        assert_eq!(first["datasets"][0]["dataset_id"], ids[0].to_string());
        assert_eq!(first["next_after"], ids[0].to_string());
        let second = read(
            app.oneshot(request(&format!("/v1/datasets?limit=1&after={}", ids[0])))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(second["datasets"].as_array().unwrap().len(), 1);
        assert_eq!(second["datasets"][0]["dataset_id"], ids[1].to_string());
        assert!(second["next_after"].is_null());
    }

    #[tokio::test]
    async fn malformed_missing_and_unauthenticated_reads_return_typed_problems() {
        let home = tempfile::tempdir().unwrap();
        let app = app(state(home.path()));
        for path in [
            "/v1/datasets/nope",
            "/v1/datasets?limit=0",
            "/v1/datasets?limit=257",
            "/v1/datasets?limit=wat",
            "/v1/datasets?after=bad",
            "/v1/datasets?unknown=1",
        ] {
            let response = app.clone().oneshot(request(path)).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
            assert_eq!(
                response.headers()[header::CONTENT_TYPE],
                jope_api::error::Problem::CONTENT_TYPE
            );
            let problem: jope_api::error::Problem =
                serde_json::from_value(read(response).await).unwrap();
            assert_eq!(problem.code, ErrorCode::InvalidInput);
        }
        let missing = "/v1/datasets/ds_000000000000000000000000";
        let response = app.clone().oneshot(request(missing)).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(read(response).await["code"], "NOT_FOUND");
        for path in ["/v1/datasets", missing] {
            let mut req = request(path);
            req.headers_mut().remove(header::AUTHORIZATION);
            assert_eq!(
                app.clone().oneshot(req).await.unwrap().status(),
                StatusCode::UNAUTHORIZED
            );
        }
    }

    #[tokio::test]
    async fn corrupt_csv_and_configured_limit_are_not_successful_profiles() {
        let home = tempfile::tempdir().unwrap();
        let mut state = state(home.path());
        let saved = register::register(&state.store, CSV, &Default::default()).unwrap();
        state.config.limits.max_dataset_bytes.value = 1;
        let limited = app(state.clone())
            .oneshot(request(&format!("/v1/datasets/{}", saved.id)))
            .await
            .unwrap();
        assert_eq!(limited.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(read(limited).await["code"], "DATASET_INVALID");
        state.config.limits.max_dataset_bytes.value = 1 << 20;
        let path = state.store.verified_blob_path(&saved.csv).unwrap();
        std::fs::write(path, b"broken").unwrap();
        let response = app(state)
            .oneshot(request(&format!("/v1/datasets/{}", saved.id)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(read(response).await["code"], "IO_FAILURE");
    }
}
