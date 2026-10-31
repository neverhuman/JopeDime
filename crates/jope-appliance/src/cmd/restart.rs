use anyhow::Context;
use jope_api::ids::RunId;
use jope_api::registry::{ForkSeed, RunForkRequest};
use jope_client::Transport;
use serde::Deserialize;
use serde_json::Value;
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
    let body = match crate::common::read_json_body(body_file)? {
        Some(value) => {
            RunForkRequest::deserialize(&value).map_err(|error| {
                jope_runtime::Failure::input(format!("invalid restart request: {error}"))
            })?;
            // Validate with the DTO without changing a retry's null-bearing request body.
            value
        }
        None => serde_json::to_value(RunForkRequest {
            seed: if new_seed {
                ForkSeed::New
            } else {
                ForkSeed::Same
            },
            warm_start: warm.then_some(true),
            overrides: None,
        })
        .context("serialize restart request")?,
    };
    let key = crate::common::resolve_idempotency_key(idempotency_key);
    crate::common::announce_generated_key(idempotency_key, &key)?;
    let result: jope_api::RunSnapshot = (|| {
        let request = jope_client::Request::new(9, 0, &[("id", &id.to_string())])?
            .key(Some(&key))?
            .json(&body)?;
        transport.execute(request)?.json()
    })()
    .context("restart run")?;
    serde_json::to_value(result).context("serialize restart response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_client::Request;
    use serde_json::json;
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
    fn body_preserves_canonical_fork_overrides() {
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

    #[test]
    fn restart_flags_preserve_same_new_and_optional_warm_start() {
        for (new_seed, warm) in [(false, false), (false, true), (true, false), (true, true)] {
            let mock = Mock::default();
            execute(&mock, RUN, new_seed, warm, None, Some("flags")).unwrap();
            let requests = mock.0.lock().unwrap();
            let body: Value = serde_json::from_slice(requests[0].body()).unwrap();
            assert_eq!(body["seed"], if new_seed { "new" } else { "same" });
            assert_eq!(body.get("warm_start"), warm.then_some(&Value::Bool(true)));
            assert!(body.get("overrides").is_none());
            serde_json::from_value::<RunForkRequest>(body).unwrap();
        }
    }

    #[test]
    fn fork_body_preserves_nested_null_overrides_and_retry_identity() {
        for body in [
            json!({"seed":"same","warm_start":null,"overrides":null}),
            json!({"seed":"new","overrides":{"effort":null,"budget":null}}),
            json!({"seed":"same","warm_start":false}),
            json!({"seed":"new","warm_start":true,"overrides":{"effort":"high","test_dataset":null,"budget":{"evaluations":null,"wall_seconds":"60"},"require_all_languages":false}}),
        ] {
            let mock = Mock::default();
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("body.json");
            std::fs::write(&path, serde_json::to_vec(&body).unwrap()).unwrap();
            for _ in 0..2 {
                execute(&mock, RUN, false, false, path.to_str(), Some("same-key")).unwrap();
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
    fn invalid_fork_body_is_rejected_before_transport() {
        for body in [
            json!({"seed":"new","unexpected":true}),
            json!({"seed":"invalid"}),
            json!({"seed":"same","warm_start":"true"}),
            json!({"seed":"new","overrides":{"unexpected":true}}),
            json!({"seed":"new","overrides":{"seed":"same"}}),
            json!({"seed":"same","overrides":{"effort":"invalid"}}),
            json!({"seed":"new","overrides":{"budget":{"evaluations":20}}}),
            json!({"seed":"new","overrides":{"budget":{"evaluations":"020"}}}),
        ] {
            let mock = Mock::default();
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("invalid.json");
            std::fs::write(&path, serde_json::to_vec(&body).unwrap()).unwrap();
            let result = execute(
                &mock,
                RUN,
                false,
                false,
                path.to_str(),
                Some("invalid-fork"),
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
