use super::*;
use crate::{config::Config, store::Store};
use axum::{
    body::{to_bytes, Body},
    http::{Method, Request},
};
use std::{collections::BTreeMap, sync::Arc};
use tower::ServiceExt;

const CSV: &str = "target,feature\n1,red\n2,blue\n";

fn fixture() -> (tempfile::TempDir, Arc<Store>, axum::Router) {
    let root = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::open(root.path()).unwrap());
    let config = Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    let auth = super::super::auth::LocalAuth::new(
        super::super::auth::BearerToken::from_hex(&"ab".repeat(32)).unwrap(),
        31234,
    )
    .unwrap();
    let app = super::super::router::router_with_state(
        auth,
        crate::engine::DatasetState {
            store: Arc::clone(&store),
            config,
            home: root.path().to_path_buf(),
        },
        None,
    );
    (root, store, app)
}

fn request(content_type: &str, key: &str, body: impl Into<Body>) -> Request<Body> {
    request_to("/v1/datasets", content_type, key, body)
}

fn request_to(uri: &str, content_type: &str, key: &str, body: impl Into<Body>) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header(header::CONTENT_TYPE, content_type)
        .header("Idempotency-Key", key)
        .body(body.into())
        .unwrap()
}

#[tokio::test]
async fn csv_query_name_is_returned_and_same_key_replay_matches() {
    let (_root, _store, app) = fixture();
    let first = app
        .clone()
        .oneshot(request_to(
            "/v1/datasets?name=rel05-s3-catalog-30-row",
            "text/csv",
            "named-upload",
            CSV,
        ))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::CREATED);
    let profile: DatasetProfile =
        serde_json::from_slice(&to_bytes(first.into_body(), 32 * 1024).await.unwrap()).unwrap();
    assert_eq!(profile.name, "rel05-s3-catalog-30-row");

    let replay = app
        .clone()
        .oneshot(request_to(
            "/v1/datasets?name=rel05-s3-catalog-30-row",
            "text/csv",
            "named-upload",
            CSV,
        ))
        .await
        .unwrap();
    assert_eq!(replay.status(), StatusCode::CREATED);
    let replayed: DatasetProfile =
        serde_json::from_slice(&to_bytes(replay.into_body(), 32 * 1024).await.unwrap()).unwrap();
    assert_eq!(replayed, profile);

    let changed_name = app
        .oneshot(request_to(
            "/v1/datasets?name=another-name",
            "text/csv",
            "named-upload",
            CSV,
        ))
        .await
        .unwrap();
    assert_eq!(changed_name.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn upload_replay_conflict_and_persisted_metadata() {
    let (_root, store, app) = fixture();
    let first = app
        .clone()
        .oneshot(request("text/csv", "upload-1", CSV))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::CREATED);
    let profile: DatasetProfile =
        serde_json::from_slice(&to_bytes(first.into_body(), 32 * 1024).await.unwrap()).unwrap();
    assert_eq!(profile.rows.0, 2);
    let key = crate::store::RecordKey::Dataset(profile.dataset_id.clone());
    let record = store.get(&key).unwrap().unwrap();
    let saved: register::RegisteredDataset = serde_json::from_value(record.data).unwrap();
    assert_eq!(store.read_blob(&saved.csv).unwrap(), CSV.as_bytes());
    assert_eq!(
        store.read_blob(&saved.matrix).unwrap().get(..4),
        Some(&b"JMAT"[..])
    );

    let replay = app
        .clone()
        .oneshot(request("text/csv", "upload-1", CSV))
        .await
        .unwrap();
    assert_eq!(replay.status(), StatusCode::CREATED);
    let replayed: DatasetProfile =
        serde_json::from_slice(&to_bytes(replay.into_body(), 32 * 1024).await.unwrap()).unwrap();
    assert_eq!(replayed.dataset_id, profile.dataset_id);

    let conflict = app
        .oneshot(request("text/csv", "upload-1", "target\n3\n"))
        .await
        .unwrap();
    assert_eq!(conflict.status(), StatusCode::CONFLICT);
    let problem: Problem =
        serde_json::from_slice(&to_bytes(conflict.into_body(), 32 * 1024).await.unwrap()).unwrap();
    assert_eq!(problem.code, ErrorCode::IdempotencyConflict);
}

#[tokio::test]
async fn csv_upload_and_authorized_path_produce_the_same_dataset() {
    let (root, _store, app) = fixture();
    let path = root.path().join("import.csv");
    std::fs::write(&path, CSV).unwrap();
    let uploaded = app
        .clone()
        .oneshot(request("text/csv", "equivalent-upload", CSV))
        .await
        .unwrap();
    let uploaded: DatasetProfile =
        serde_json::from_slice(&to_bytes(uploaded.into_body(), 32 * 1024).await.unwrap()).unwrap();
    let body = serde_json::to_vec(&json!({"path": path, "name": "fixture"})).unwrap();
    let local = app
        .oneshot(request("application/json", "equivalent-path", body))
        .await
        .unwrap();
    assert_eq!(local.status(), StatusCode::CREATED);
    let local: DatasetProfile =
        serde_json::from_slice(&to_bytes(local.into_body(), 32 * 1024).await.unwrap()).unwrap();
    assert_eq!(uploaded.dataset_id, local.dataset_id);
    assert_eq!(local.name, "fixture");
}

#[tokio::test]
async fn authentication_path_and_configured_refusals_are_enforced() {
    let (_root, _store, app) = fixture();
    let mut unauthorized = request(
        "application/json",
        "unauthorized-path",
        r#"{"path":"/etc/passwd"}"#,
    );
    unauthorized.headers_mut().remove(header::AUTHORIZATION);
    assert_eq!(
        app.clone().oneshot(unauthorized).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );

    let missing = app
        .clone()
        .oneshot(request(
            "application/json",
            "missing-path",
            r#"{"path":"/definitely/not/a/dataset.csv"}"#,
        ))
        .await
        .unwrap();
    assert_eq!(missing.status(), StatusCode::BAD_REQUEST);

    let limit_root = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::open(limit_root.path()).unwrap());
    let mut config = Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    config.limits.max_dataset_bytes.value = 4;
    let auth = super::super::auth::LocalAuth::new(
        super::super::auth::BearerToken::from_hex(&"ab".repeat(32)).unwrap(),
        31234,
    )
    .unwrap();
    let limited = super::super::router::router_with_state(
        auth,
        crate::engine::DatasetState {
            store,
            config,
            home: limit_root.path().to_path_buf(),
        },
        None,
    );
    let too_big = limited
        .oneshot(request("text/csv", "too-big", CSV))
        .await
        .unwrap();
    assert_eq!(too_big.status(), StatusCode::UNPROCESSABLE_ENTITY);

    let (reserve_root, store, _app) = fixture();
    let mut config = Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    config.limits.disk_reserve.value = u64::MAX;
    let auth = super::super::auth::LocalAuth::new(
        super::super::auth::BearerToken::from_hex(&"ab".repeat(32)).unwrap(),
        31234,
    )
    .unwrap();
    let reserved = super::super::router::router_with_state(
        auth,
        crate::engine::DatasetState {
            store,
            config,
            home: reserve_root.path().to_path_buf(),
        },
        None,
    );
    let rejected = reserved
        .oneshot(request("text/csv", "disk-reserve", CSV))
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::SERVICE_UNAVAILABLE);
}
