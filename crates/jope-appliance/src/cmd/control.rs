use crate::cli::ControlAction;
use anyhow::Context;
use jope_api::ids::RunId;
use jope_api::registry::RunControlRequest;
use jope_client::Transport;
use serde::Deserialize;
use serde_json::Value;
use std::str::FromStr;

pub fn run(
    run: String,
    action: ControlAction,
    body: Option<String>,
    idempotency_key: Option<String>,
) -> anyhow::Result<()> {
    let client = crate::common::resolve_client()?;
    let value = execute(
        &client,
        &run,
        action,
        body.as_deref(),
        idempotency_key.as_deref(),
    )?;
    crate::common::print_json(&value)
}

fn execute<T: Transport>(
    transport: &T,
    run: &str,
    action: ControlAction,
    body_file: Option<&str>,
    idempotency_key: Option<&str>,
) -> anyhow::Result<Value> {
    let id = RunId::from_str(run).map_err(|_| jope_runtime::Failure::input("invalid run id"))?;
    let body =
        match crate::common::read_json_body(body_file)? {
            Some(value) => {
                RunControlRequest::deserialize(&value).map_err(|error| {
                    jope_runtime::Failure::input(format!("invalid control request: {error}"))
                })?;
                // Retry identities bind the JSON body: retain explicit nulls after validation.
                value
            }
            None => serde_json::to_value(match action {
                ControlAction::Pause => RunControlRequest::Pause {
                    expected_revision: None,
                },
                ControlAction::Continue => RunControlRequest::Continue {
                    expected_revision: None,
                },
                ControlAction::Cancel => RunControlRequest::Cancel {
                    expected_revision: None,
                },
                ControlAction::Effort { level } => RunControlRequest::SetEffort {
                    effort: Some(serde_json::from_value(Value::String(level)).map_err(
                        |error| jope_runtime::Failure::input(format!("invalid effort: {error}")),
                    )?),
                    budget: None,
                    expected_revision: None,
                },
            })
            .context("serialize control request")?,
        };
    let key = crate::common::resolve_idempotency_key(idempotency_key);
    crate::common::announce_generated_key(idempotency_key, &key)?;
    let result = transport
        .run_control(&id, &body, Some(&key))
        .context("control run")?;
    serde_json::to_value(result).context("serialize control response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_client::{transport::HeaderValue, Request, Response};
    use serde_json::json;
    use std::sync::Mutex;

    const RUN: &str = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";
    const SNAPSHOT: &str = r#"{"best":null,"budget_total":{"evaluations":"10","wall_seconds":"10"},"budget_used":{"evaluations":"1","wall_seconds":"0.1"},"completion_reason":null,"effort":"medium","event_watermark":"2","pending":null,"phase":"searching","revision":"2","run_id":"run_01ARZ3NDEKTSV4RRFFQ69G5FAV","state":"paused","warnings":[]}"#;

    #[derive(Default)]
    struct Mock(Mutex<Vec<Request>>);
    impl jope_client::Transport for Mock {
        fn execute(&self, request: Request) -> jope_client::Result<Response> {
            self.0.lock().unwrap().push(request);
            let mut headers = jope_client::transport::HeaderMap::new();
            headers.insert("content-type", HeaderValue::from_static("application/json"));
            headers.insert("jope-api-version", HeaderValue::from_static("1.0"));
            Ok(Response::new(200, headers, SNAPSHOT.as_bytes()))
        }
    }

    #[test]
    fn pause_matches_golden_snapshot_and_registry_request() {
        let mock = Mock::default();
        let result = execute(&mock, RUN, ControlAction::Pause, None, Some("retry-1")).unwrap();
        assert_eq!(serde_json::to_string(&result).unwrap(), SNAPSHOT);
        let requests = mock.0.lock().unwrap();
        assert_eq!(requests[0].target(), format!("/v1/runs/{RUN}/control"));
        assert_eq!(
            serde_json::from_slice::<Value>(requests[0].body()).unwrap(),
            json!({"action":"pause"})
        );
    }

    #[test]
    fn effort_uses_catalogued_field_names() {
        let mock = Mock::default();
        execute(
            &mock,
            RUN,
            ControlAction::Effort {
                level: "high".into(),
            },
            None,
            None,
        )
        .unwrap();
        let requests = mock.0.lock().unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(requests[0].body()).unwrap(),
            json!({"action":"set_effort","effort":"high"})
        );
    }

    #[test]
    fn body_preserves_budget_and_revision() {
        let mock = Mock::default();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("body.json");
        std::fs::write(
            &path,
            br#"{"action":"set_effort","expected_revision":"7","budget":{"evaluations":"20"}}"#,
        )
        .unwrap();
        execute(
            &mock,
            RUN,
            ControlAction::Pause,
            path.to_str(),
            Some("retry-3"),
        )
        .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(mock.0.lock().unwrap()[0].body()).unwrap(),
            json!({"action":"set_effort","expected_revision":"7","budget":{"evaluations":"20"}})
        );
    }

    #[test]
    fn control_flags_use_canonical_actions_and_effort_values() {
        for (action, expected) in [
            (ControlAction::Pause, json!({"action":"pause"})),
            (ControlAction::Continue, json!({"action":"continue"})),
            (ControlAction::Cancel, json!({"action":"cancel"})),
        ]
        .into_iter()
        .chain(["fast", "medium", "high", "ultra"].map(|level| {
            (
                ControlAction::Effort {
                    level: level.into(),
                },
                json!({"action":"set_effort","effort":level}),
            )
        })) {
            let mock = Mock::default();
            execute(&mock, RUN, action, None, Some("flags")).unwrap();
            let requests = mock.0.lock().unwrap();
            let body: Value = serde_json::from_slice(requests[0].body()).unwrap();
            assert_eq!(body, expected);
            serde_json::from_value::<RunControlRequest>(body).unwrap();
        }
    }

    #[test]
    fn control_body_preserves_nullable_updates_and_retry_identity() {
        for body in [
            json!({"action":"pause","expected_revision":null}),
            json!({"action":"set_effort","effort":null,"budget":{"evaluations":"20"},"expected_revision":null}),
            json!({"action":"pause","expected_revision":"18446744073709551615"}),
            json!({"action":"continue","expected_revision":"0"}),
            json!({"action":"cancel","expected_revision":"2"}),
            json!({"action":"set_effort","budget":{"evaluations":null}}),
            json!({"action":"set_effort","effort":"ultra","budget":{"evaluations":"20","wall_seconds":null},"expected_revision":"7"}),
        ] {
            let mock = Mock::default();
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("body.json");
            std::fs::write(&path, serde_json::to_vec(&body).unwrap()).unwrap();
            for _ in 0..2 {
                execute(
                    &mock,
                    RUN,
                    ControlAction::Pause,
                    path.to_str(),
                    Some("same-key"),
                )
                .unwrap();
            }
            let requests = mock.0.lock().unwrap();
            assert_eq!(requests.len(), 2);
            for request in requests.iter() {
                assert_eq!(
                    serde_json::from_slice::<Value>(request.body()).unwrap(),
                    body
                );
                assert_eq!(request.headers["idempotency-key"], "same-key");
            }
            assert_eq!(requests[0].body(), requests[1].body());
        }
    }

    #[test]
    fn invalid_control_body_is_rejected_before_transport() {
        for body in [
            json!({"action":"pause","unexpected":true}),
            json!({"action":"pause","effort":"high"}),
            json!({"action":"set_effort","effort":"invalid"}),
            json!({"action":"set_effort"}),
            json!({"action":"set_effort","budget":{}}),
            json!({"action":"set_effort","budget":{"unexpected":"1"}}),
            json!({"action":"pause","expected_revision":7}),
            json!({"action":"pause","expected_revision":"07"}),
            json!({"action":"pause","expected_revision":"18446744073709551616"}),
        ] {
            let mock = Mock::default();
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("invalid.json");
            std::fs::write(&path, serde_json::to_vec(&body).unwrap()).unwrap();
            let result = execute(
                &mock,
                RUN,
                ControlAction::Pause,
                path.to_str(),
                Some("invalid-control"),
            );
            assert!(result.is_err(), "invalid body reached transport: {body}");
            assert_eq!(
                jope_runtime::error::kind_of(&result.unwrap_err()).exit_code(),
                2,
                "{body}"
            );
            assert!(mock.0.lock().unwrap().is_empty(), "{body}");
        }
    }
}
