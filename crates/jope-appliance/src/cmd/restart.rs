use anyhow::Context;
use jope_api::ids::RunId;
use jope_client::Transport;
use serde_json::{json, Value};
use std::str::FromStr;

pub fn run(
    run: String,
    new_seed: bool,
    warm: bool,
    body: Option<String>,
    idempotency_key: Option<String>,
) -> anyhow::Result<()> {
    let client = crate::common::resolve_client()?;
    let value = execute(
        &client,
        &run,
        new_seed,
        warm,
        body.as_deref(),
        idempotency_key.as_deref(),
    )?;
    crate::common::print_json(&value)
}

fn execute<T: Transport>(
    transport: &T,
    run: &str,
    new_seed: bool,
    warm: bool,
    body_file: Option<&str>,
    idempotency_key: Option<&str>,
) -> anyhow::Result<Value> {
    let id = RunId::from_str(run).map_err(|_| jope_runtime::Failure::input("invalid run id"))?;
    let key = crate::common::resolve_idempotency_key(idempotency_key);
    crate::common::announce_generated_key(idempotency_key, &key)?;
    let body = match crate::common::read_json_body(body_file)? {
        Some(value) => value,
        None => {
            let mut value = json!({"seed": if new_seed { "new" } else { "same" }});
            if warm {
                value["warm_start"] = Value::Bool(true);
            }
            value
        }
    };
    let result = transport
        .run_fork(&id, &body, Some(&key))
        .context("restart run")?;
    serde_json::to_value(result).context("serialize restart response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_client::Request;
    use std::sync::Mutex;

    const RUN: &str = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";
    const SNAPSHOT: &str = r#"{"best":null,"budget_total":{"evaluations":"10","wall_seconds":"10"},"budget_used":{"evaluations":"0","wall_seconds":"0"},"completion_reason":null,"effort":"medium","event_watermark":"0","pending":null,"phase":"preparing","revision":"2","run_id":"run_01ARZ3NDEKTSV4RRFFQ69G5FAV","state":"running","warnings":[]}"#;

    #[derive(Default)]
    struct Mock(Mutex<Vec<Request>>);
    impl jope_client::Transport for Mock {
        fn execute(&self, request: Request) -> jope_client::Result<jope_client::Response> {
            self.0.lock().unwrap().push(request);
            let mut headers = jope_client::transport::HeaderMap::new();
            headers.insert(
                "content-type",
                jope_client::transport::HeaderValue::from_static("application/json"),
            );
            headers.insert(
                "jope-api-version",
                jope_client::transport::HeaderValue::from_static("1.0"),
            );
            Ok(jope_client::Response::new(
                200,
                headers,
                SNAPSHOT.as_bytes(),
            ))
        }
    }

    #[test]
    fn restart_maps_flags_to_registry_fields_and_golden_response() {
        let mock = Mock::default();
        let result = execute(&mock, RUN, true, true, None, Some("retry-2")).unwrap();
        assert_eq!(serde_json::to_string(&result).unwrap(), SNAPSHOT);
        let requests = mock.0.lock().unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(requests[0].body()).unwrap(),
            json!({"seed":"new","warm_start":true})
        );
    }

    #[test]
    fn restart_without_flags_uses_same_seed_and_omits_optional_warm_start() {
        let mock = Mock::default();
        execute(&mock, RUN, false, false, None, None).unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(mock.0.lock().unwrap()[0].body()).unwrap(),
            json!({"seed":"same"})
        );
    }

    #[test]
    fn raw_body_preserves_overrides_for_pending_fork_schema() {
        let mock = Mock::default();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("fork.json");
        std::fs::write(
            &path,
            br#"{"seed":"new","warm_start":true,"overrides":{"effort":"fast"}}"#,
        )
        .unwrap();
        execute(
            &mock,
            RUN,
            false,
            false,
            path.to_str(),
            Some("fork-override"),
        )
        .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(mock.0.lock().unwrap()[0].body()).unwrap(),
            json!({"seed":"new","warm_start":true,"overrides":{"effort":"fast"}})
        );
    }
}
