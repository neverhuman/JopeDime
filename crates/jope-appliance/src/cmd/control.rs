use crate::cli::ControlAction;
use anyhow::Context;
use jope_api::ids::RunId;
use jope_client::Transport;
use serde_json::{json, Value};
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
    let key = crate::common::resolve_idempotency_key(idempotency_key);
    crate::common::announce_generated_key(idempotency_key, &key)?;
    let body = match crate::common::read_json_body(body_file)? {
        Some(value) => value,
        None => match action {
            ControlAction::Pause => json!({"action":"pause"}),
            ControlAction::Continue => json!({"action":"continue"}),
            ControlAction::Cancel => json!({"action":"cancel"}),
            ControlAction::Effort { level } => {
                json!({"action":"set_effort","effort":level})
            }
        },
    };
    let result = transport
        .run_control(&id, &body, Some(&key))
        .context("control run")?;
    serde_json::to_value(result).context("serialize control response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_client::{transport::HeaderValue, Request, Response};
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
    fn raw_body_is_passed_through_without_schema_invention() {
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
}
