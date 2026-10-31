//! Store-parameterized destructive operations.
//!
//! Router/state wiring is deferred to SVC-13-F01; these handlers never obtain
//! a global store or start background work.

use super::auth::problem;
use crate::store::{gc::DeleteDataset, gc::DeleteRun, Store};
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use jope_api::{
    error::ErrorCode,
    ids::{DatasetId, RunId},
};
use serde::{Deserialize, Serialize};

/// Delete a dataset's metadata. Its blobs remain until explicit store GC.
pub fn delete_dataset(store: &Store, id: &str) -> Response {
    let id = match id.parse::<DatasetId>() {
        Ok(id) => id,
        Err(_) => {
            return problem(
                ErrorCode::InvalidInput,
                "The dataset ID is invalid.",
                "Supply a dataset ID returned by registration.",
                Some("id"),
            );
        }
    };
    match store.delete_dataset(&id) {
        Ok(DeleteDataset::Deleted) => StatusCode::NO_CONTENT.into_response(),
        Ok(DeleteDataset::NotFound) => problem(
            ErrorCode::NotFound,
            "The dataset does not exist.",
            "List registered datasets and retry with an existing ID.",
            None,
        ),
        Ok(DeleteDataset::Referenced) => problem(
            ErrorCode::ImmutableField,
            "The dataset is referenced by a run and cannot be deleted.",
            "Delete the referencing runs first.",
            None,
        ),
        Err(error) => store_failure(error),
    }
}

/// Delete a quiescent run and its event/evaluation history.
pub fn delete_run(store: &Store, id: &str) -> Response {
    let id = match id.parse::<RunId>() {
        Ok(id) => id,
        Err(_) => {
            return problem(
                ErrorCode::InvalidInput,
                "The run ID is invalid.",
                "Supply a run ID returned by the engine.",
                Some("id"),
            );
        }
    };
    match store.delete_run(&id) {
        Ok(DeleteRun::Deleted) => StatusCode::NO_CONTENT.into_response(),
        Ok(DeleteRun::NotFound) => problem(
            ErrorCode::NotFound,
            "The run does not exist.",
            "List runs and retry with an existing ID.",
            None,
        ),
        Ok(DeleteRun::Active) => problem(
            ErrorCode::ImmutableField,
            "An active run cannot be deleted.",
            "Wait for the run to pause or finish, then retry.",
            None,
        ),
        Err(error) => store_failure(error),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GcRequest {
    pub dry_run: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct GcResponse {
    pub dry_run: bool,
    /// The candidate list on a dry run and the exact removed list otherwise.
    pub blobs: Vec<crate::store::BlobRef>,
    pub bytes: u64,
}

/// List unreferenced blobs without mutation, or remove the same set atomically
/// with respect to Store writers.
pub fn collect_garbage(store: &Store, request: GcRequest) -> Response {
    match store.gc(request.dry_run) {
        Ok(report) => (
            StatusCode::OK,
            Json(GcResponse {
                dry_run: request.dry_run,
                blobs: report.blobs,
                bytes: report.bytes,
            }),
        )
            .into_response(),
        Err(error) => store_failure(error),
    }
}

fn store_failure(error: crate::store::StoreError) -> Response {
    problem(
        ErrorCode::IoFailure,
        &format!("Store operation failed: {error}"),
        "Inspect the local store and retry only after correcting the reported condition.",
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idempotency::RequestIdentity;
    use crate::store::{Record, RecordKey};
    use axum::body::to_bytes;
    use jope_api::{
        error::Problem,
        ids::RunId,
        registry::{HttpMethod, OPERATIONS},
        RunState,
    };
    use serde_json::{json, Value};
    use std::io::Cursor;
    use tempfile::TempDir;

    fn store() -> (TempDir, Store) {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        (dir, store)
    }

    async fn body(response: Response) -> Vec<u8> {
        to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap()
            .to_vec()
    }

    fn dataset_id() -> DatasetId {
        "ds_00112233445566778899aabb".parse().unwrap()
    }

    fn run_id() -> RunId {
        "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap()
    }

    fn run_delete_identity(
        principal: &str,
        key: &str,
        id: &RunId,
        if_match: Option<&str>,
    ) -> RequestIdentity {
        let operation = OPERATIONS
            .iter()
            .find(|operation| operation.name == "run_delete")
            .expect("run_delete is registered");
        assert_eq!(operation.method, HttpMethod::Delete);
        assert_eq!(operation.paths, &["/v1/runs/{id}"]);
        assert!(operation.requires_idempotency_key);

        RequestIdentity::new(
            principal,
            key,
            operation.id,
            &format!("/v1/runs/{id}"),
            if_match,
            &Value::Null,
        )
        .unwrap()
    }

    /// Pins request-identity inputs for the future run-delete adapter. This
    /// scaffold does not claim that the detached handler is mounted or
    /// idempotent yet; those boundaries remain owned by SVC-13/SVC-04.
    #[test]
    fn run_delete_identity_scaffold_binds_target_precondition_and_principal() {
        let id = run_id();
        let identity = run_delete_identity("local-principal", "delete-1", &id, Some("\"r7\""));
        let retry = run_delete_identity("local-principal", "delete-1", &id, Some("\"r7\""));
        let changed_precondition =
            run_delete_identity("local-principal", "delete-1", &id, Some("\"r8\""));
        let changed_target = run_delete_identity(
            "local-principal",
            "delete-1",
            &"run_01JBD8Q6N4V2T5Y8K3M7P9R1S1".parse().unwrap(),
            Some("\"r7\""),
        );
        let other_principal =
            run_delete_identity("other-principal", "delete-1", &id, Some("\"r7\""));
        let other_key = run_delete_identity("local-principal", "delete-2", &id, Some("\"r7\""));

        assert_eq!(identity.operation_id, 10);
        assert_eq!(identity.principal, "local-principal");
        assert_eq!(identity.key, "delete-1");
        assert_eq!(identity.digest, retry.digest);
        assert_ne!(identity.digest, changed_precondition.digest);
        assert_ne!(identity.digest, changed_target.digest);
        assert_eq!(identity.digest, other_principal.digest);
        assert_ne!(identity.principal, other_principal.principal);
        assert_eq!(identity.digest, other_key.digest);
        assert_ne!(identity.key, other_key.key);
    }

    #[tokio::test]
    async fn dataset_delete_returns_conflict_while_referenced_then_no_content() {
        let (_dir, store) = store();
        let id = dataset_id();
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Dataset(id.clone()),
                    &Record::metadata(json!({"id": id}), vec![]),
                )
            })
            .unwrap();
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Run(run_id()),
                    &Record::run(json!({"dataset": id}), vec![], RunState::Completed, 1),
                )
            })
            .unwrap();

        let blocked = delete_dataset(&store, &id.to_string());
        assert_eq!(blocked.status(), StatusCode::CONFLICT);
        let doc: Problem = serde_json::from_slice(&body(blocked).await).unwrap();
        assert_eq!(doc.code, ErrorCode::ImmutableField);
        assert!(doc.detail.contains("referenced by a run"));

        store.delete_run(&run_id()).unwrap();
        let deleted = delete_dataset(&store, &id.to_string());
        assert_eq!(deleted.status(), StatusCode::NO_CONTENT);
        assert!(body(deleted).await.is_empty());
        assert!(store.get(&RecordKey::Dataset(id)).unwrap().is_none());
    }

    #[tokio::test]
    async fn run_delete_refuses_active_and_deletes_completed_run() {
        let (_dir, store) = store();
        let id = run_id();
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Run(id.clone()),
                    &Record::run(json!({}), vec![], RunState::Running, 1),
                )
            })
            .unwrap();
        let blocked = delete_run(&store, &id.to_string());
        assert_eq!(blocked.status(), StatusCode::CONFLICT);
        let doc: Problem = serde_json::from_slice(&body(blocked).await).unwrap();
        assert_eq!(doc.code, ErrorCode::ImmutableField);

        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Run(id.clone()),
                    &Record::run(json!({}), vec![], RunState::Completed, 2),
                )
            })
            .unwrap();
        assert_eq!(
            delete_run(&store, &id.to_string()).status(),
            StatusCode::NO_CONTENT
        );
        assert!(store.get(&RecordKey::Run(id)).unwrap().is_none());
    }

    #[tokio::test]
    async fn gc_dry_run_lists_orphan_without_removal_then_actual_removes_it() {
        let (_dir, store) = store();
        let orphan = store.publish_blob(Cursor::new(b"orphan")).unwrap();
        let response = collect_garbage(&store, GcRequest { dry_run: true });
        assert_eq!(response.status(), StatusCode::OK);
        let dry: GcResponse = serde_json::from_slice(&body(response).await).unwrap();
        assert!(dry.dry_run);
        assert_eq!(dry.blobs, vec![orphan.clone()]);
        assert!(store.verify_blob(&orphan).is_ok());

        let response = collect_garbage(&store, GcRequest { dry_run: false });
        let removed: GcResponse = serde_json::from_slice(&body(response).await).unwrap();
        assert!(!removed.dry_run);
        assert_eq!(removed.blobs, dry.blobs);
        assert_eq!(removed.bytes, dry.bytes);
        assert!(store.verify_blob(&orphan).is_err());
    }
}
