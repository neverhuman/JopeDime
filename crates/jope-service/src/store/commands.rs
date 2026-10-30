//! Atomic effects and durable idempotency records (spec §4.1, §5.3).
use super::{Record, RecordKey, Store, StoreError, Write};
use crate::idempotency::{
    CommandOutcome, CommandReceipt, IdempotencyError, RequestIdentity, StoredCommand,
};

impl Store {
    /// The effect must write metadata ONLY through the supplied transaction.
    /// Prepare immutable CAS blobs first; schedule external work after a new receipt.
    /// Commit IO errors have unknown outcomes: read back/reopen and retry the same key.
    pub fn execute_idempotent(
        &self,
        request: &RequestIdentity,
        effect: impl FnOnce(&mut Write<'_>) -> super::Result<CommandOutcome>,
    ) -> Result<CommandReceipt, IdempotencyError> {
        let key = RecordKey::Command {
            principal: request.principal.clone(),
            idempotency_key: request.key.clone(),
        };
        let receipt = self.transaction(|write| {
            if let Some(record) = write.get(&key)? {
                if record.run.is_some() || !record.blobs.is_empty() {
                    return Err(StoreError::Integrity("invalid command record envelope"));
                }
                let stored: StoredCommand = serde_json::from_value(record.data)?;
                stored.outcome.validate()?;
                if stored.request_digest != request.digest
                    || stored.operation_id != request.operation_id
                {
                    return Ok(None);
                }
                return Ok(Some(CommandReceipt {
                    operation_id: stored.operation_id,
                    outcome: stored.outcome,
                    replayed: true,
                }));
            }
            let outcome = effect(write)?;
            outcome.validate()?;
            let stored = StoredCommand {
                request_digest: request.digest,
                operation_id: request.operation_id,
                outcome: outcome.clone(),
            };
            write.put(
                &key,
                &Record::metadata(serde_json::to_value(stored)?, vec![]),
            )?;
            Ok(Some(CommandReceipt {
                operation_id: request.operation_id,
                outcome,
                replayed: false,
            }))
        })?;
        receipt.ok_or(IdempotencyError::Conflict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::to_bytes, response::IntoResponse};
    use jope_api::error::{ErrorCode, Problem};
    use serde_json::json;
    use std::{
        fs,
        path::PathBuf,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    fn fixture() -> tempfile::TempDir {
        let root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/idempotency-fixtures");
        fs::create_dir_all(&root).unwrap();
        tempfile::tempdir_in(root).unwrap()
    }
    fn request(principal: &str, value: u64) -> RequestIdentity {
        RequestIdentity::new(
            principal,
            "fixture-key",
            6,
            "/v1/runs",
            None,
            &json!({"value":value}),
        )
        .unwrap()
    }
    fn command_key(principal: &str) -> RecordKey {
        RecordKey::Command {
            principal: principal.into(),
            idempotency_key: "fixture-key".into(),
        }
    }
    fn effect_key() -> RecordKey {
        RecordKey::Dataset("ds_00112233445566778899aabb".parse().unwrap())
    }
    fn effect(write: &mut Write<'_>) -> super::super::Result<CommandOutcome> {
        let n = write
            .get(&effect_key())?
            .map(|r| r.data.as_u64().unwrap())
            .unwrap_or(0)
            + 1;
        write.put(&effect_key(), &Record::metadata(json!(n), vec![]))?;
        CommandOutcome::json(
            201,
            json!({"run_id":"run_00000000000000000000000001","revision":"18446744073709551615"}),
        )
    }

    #[test]
    fn same_request_replays_original_outcome_and_operation_after_reopen() {
        let dir = fixture();
        let store = Store::open(dir.path()).unwrap();
        let first = store
            .execute_idempotent(&request("owner", 1), effect)
            .unwrap();
        assert!(!first.replayed);
        assert_eq!(first.operation_id, 6);
        assert_eq!(first.outcome.status(), 201);
        let record = store.get(&command_key("owner")).unwrap().unwrap();
        assert_eq!(record.data["operation_id"], 6);
        assert!(record.data.get("request_digest").is_some());
        assert_eq!(record.data["outcome"]["body"], *first.outcome.body());
        drop(store);
        let reopened = Store::open(dir.path()).unwrap();
        let replay = reopened
            .execute_idempotent(&request("owner", 1), |_| panic!("effect rerun"))
            .unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.outcome, first.outcome);
        assert_eq!(replay.operation_id, first.operation_id);
        assert_eq!(reopened.get(&effect_key()).unwrap().unwrap().data, json!(1));
    }

    #[test]
    fn principals_do_not_share_keys_and_concurrent_retries_execute_once() {
        let dir = fixture();
        let store = Arc::new(Store::open(dir.path()).unwrap());
        let calls = Arc::new(AtomicUsize::new(0));
        let mut children = vec![];
        for _ in 0..8 {
            let store = store.clone();
            let calls = calls.clone();
            children.push(std::thread::spawn(move || {
                store
                    .execute_idempotent(&request("owner", 1), |w| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        effect(w)
                    })
                    .unwrap()
            }));
        }
        let receipts: Vec<_> = children.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(receipts.iter().filter(|r| !r.replayed).count(), 1);
        assert!(receipts.iter().all(|r| r.outcome == receipts[0].outcome));
        assert!(
            !store
                .execute_idempotent(&request("other", 2), effect)
                .unwrap()
                .replayed
        );
        assert_eq!(store.get(&effect_key()).unwrap().unwrap().data, json!(2));
    }

    #[test]
    fn abort_rolls_back_both_effect_and_command_and_retry_can_succeed() {
        let dir = fixture();
        let store = Store::open(dir.path()).unwrap();
        let error = store.execute_idempotent(&request("owner", 1), |write| {
            effect(write)?;
            Err(StoreError::Invalid("fixture abort"))
        });
        assert!(matches!(error, Err(IdempotencyError::Store(_))));
        assert!(store.get(&effect_key()).unwrap().is_none());
        assert!(store.get(&command_key("owner")).unwrap().is_none());
        assert!(
            !store
                .execute_idempotent(&request("owner", 1), effect)
                .unwrap()
                .replayed
        );
        assert_eq!(store.get(&effect_key()).unwrap().unwrap().data, json!(1));
    }

    #[test]
    fn malformed_saved_receipt_fails_closed_without_running_effect() {
        let dir = fixture();
        let store = Store::open(dir.path()).unwrap();
        store
            .transaction(|w| {
                w.put(
                    &command_key("owner"),
                    &Record::metadata(json!({"outcome":"corrupt"}), vec![]),
                )
            })
            .unwrap();
        assert!(matches!(
            store.execute_idempotent(&request("owner", 1), |_| panic!("effect invoked")),
            Err(IdempotencyError::Store(_))
        ));
        assert!(store.get(&effect_key()).unwrap().is_none());
    }

    #[tokio::test]
    async fn conflict_is_409_problem_and_successful_http_replay_is_identical() {
        let dir = fixture();
        let store = Store::open(dir.path()).unwrap();
        let original = store
            .execute_idempotent(&request("owner", 1), effect)
            .unwrap()
            .into_response();
        let replay = store
            .execute_idempotent(&request("owner", 1), |_| panic!("replay effect"))
            .unwrap()
            .into_response();
        assert_eq!(original.status(), replay.status());
        assert_eq!(original.headers(), replay.headers());
        assert_eq!(
            to_bytes(original.into_body(), usize::MAX).await.unwrap(),
            to_bytes(replay.into_body(), usize::MAX).await.unwrap()
        );
        let conflict = store
            .execute_idempotent(&request("owner", 2), |_| panic!("conflict effect"))
            .unwrap_err()
            .into_response();
        assert_eq!(conflict.status(), 409);
        assert_eq!(conflict.headers()["content-type"], Problem::CONTENT_TYPE);
        assert_eq!(conflict.headers()["jope-api-version"], "1.0");
        let problem: Problem =
            serde_json::from_slice(&to_bytes(conflict.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(problem.code, ErrorCode::IdempotencyConflict);
        assert!(!problem.retriable);
        assert_eq!(store.get(&effect_key()).unwrap().unwrap().data, json!(1));
    }

    #[tokio::test]
    async fn request_boundary_replays_and_conflicts_without_reapplying_metadata() {
        use axum::{
            body::Body,
            extract::State,
            http::{HeaderMap, Request},
            routing::post,
            Json, Router,
        };
        use tower::ServiceExt;
        let dir = fixture();
        let store = Arc::new(Store::open(dir.path()).unwrap());
        let app = Router::new()
            .route(
                "/v1/runs",
                post(
                    |State(store): State<Arc<Store>>,
                     headers: HeaderMap,
                     Json(body): Json<serde_json::Value>| async move {
                        let key = headers
                            .get("Idempotency-Key")
                            .and_then(|h| h.to_str().ok())
                            .unwrap_or("");
                        match RequestIdentity::new(
                            "authenticated-fixture-owner",
                            key,
                            6,
                            "/v1/runs",
                            None,
                            &body,
                        ) {
                            Err(error) => error.into_response(),
                            Ok(identity) => match store.execute_idempotent(&identity, effect) {
                                Ok(receipt) => receipt.into_response(),
                                Err(error) => error.into_response(),
                            },
                        }
                    },
                ),
            )
            .with_state(store.clone());
        let request = |value| {
            Request::builder()
                .method("POST")
                .uri("/v1/runs")
                .header("Content-Type", "application/json")
                .header("Idempotency-Key", "http-fixture")
                .body(Body::from(format!(r#"{{"value":{value}}}"#)))
                .unwrap()
        };
        let first = app.clone().oneshot(request(1)).await.unwrap();
        let replay = app.clone().oneshot(request(1)).await.unwrap();
        assert_eq!(first.status(), 201);
        assert_eq!(replay.status(), 201);
        assert_eq!(
            to_bytes(first.into_body(), 4096).await.unwrap(),
            to_bytes(replay.into_body(), 4096).await.unwrap()
        );
        let conflict = app.oneshot(request(2)).await.unwrap();
        assert_eq!(conflict.status(), 409);
        let problem: Problem =
            serde_json::from_slice(&to_bytes(conflict.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(problem.code, ErrorCode::IdempotencyConflict);
        assert_eq!(store.get(&effect_key()).unwrap().unwrap().data, json!(1));
    }

    #[cfg(feature = "failpoints")]
    #[test]
    #[ignore = "subprocess-only abort target; explicitly invoked by crash supervisor"]
    fn crash_child() {
        let root = std::env::var_os("JOPE_IDEMPOTENCY_CRASH_ROOT").unwrap();
        let store = Store::open(PathBuf::from(root)).unwrap();
        let mode = std::env::var("JOPE_IDEMPOTENCY_CRASH_MODE").unwrap();
        if mode == "split-negative-control" {
            store.transaction(effect).unwrap();
            eprintln!("JOPE_IDEMPOTENCY_CRASH:split-negative-control");
            std::process::abort();
        }
        store
            .execute_idempotent(&request("owner", 1), |write| {
                let result = effect(write)?;
                if mode == "before_commit" {
                    eprintln!("JOPE_IDEMPOTENCY_CRASH:before_commit");
                    std::process::abort();
                }
                Ok(result)
            })
            .unwrap();
        panic!("requested crash boundary not reached");
    }

    #[cfg(feature = "failpoints")]
    #[test]
    fn actual_crash_retry_is_once_after_commit_and_clean_before_commit() {
        use std::process::Command;
        for mode in ["before_commit", "after_commit", "split-negative-control"] {
            let dir = fixture();
            drop(Store::open(dir.path()).unwrap());
            #[cfg(unix)]
            let mut child = {
                let mut c = Command::new("sh");
                c.args(["-c", "ulimit -c 0; exec \"$@\"", "idempotency-crash-child"]);
                c.arg(std::env::current_exe().unwrap());
                c
            };
            #[cfg(not(unix))]
            let mut child = Command::new(std::env::current_exe().unwrap());
            child
                .args([
                    "--exact",
                    "store::commands::tests::crash_child",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("JOPE_IDEMPOTENCY_CRASH_ROOT", dir.path())
                .env("JOPE_IDEMPOTENCY_CRASH_MODE", mode)
                .env_remove("JOPE_STORE_FAILPOINT");
            if mode == "after_commit" {
                child.env("JOPE_STORE_FAILPOINT", "after_commit");
            }
            let output = child.output().unwrap();
            assert!(!output.status.success(), "{mode}");
            let marker = if mode == "after_commit" {
                "JOPE_STORE_CRASH:after_commit".to_owned()
            } else {
                format!("JOPE_IDEMPOTENCY_CRASH:{mode}")
            };
            assert!(
                String::from_utf8_lossy(&output.stderr).contains(&marker),
                "{mode}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let reopened = Store::open(dir.path()).unwrap();
            let committed = mode == "after_commit";
            assert_eq!(
                reopened.get(&command_key("owner")).unwrap().is_some(),
                committed
            );
            assert_eq!(
                reopened.get(&effect_key()).unwrap().is_some(),
                mode != "before_commit"
            );
            let retry = reopened
                .execute_idempotent(&request("owner", 1), effect)
                .unwrap();
            assert_eq!(retry.replayed, committed);
            let expected = if mode == "split-negative-control" {
                2
            } else {
                1
            };
            assert_eq!(
                reopened.get(&effect_key()).unwrap().unwrap().data,
                json!(expected)
            );
        }
    }
}
