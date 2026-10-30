use anyhow::Context;
use jope_api::ids::{AlgorithmId, RunId};
use jope_api::Language;
use jope_client::Transport;
use serde_json::Value;
use std::str::FromStr;

pub fn run(run: String, alg: Option<String>, lang: Option<String>) -> anyhow::Result<()> {
    let client = crate::common::resolve_client()?;
    let value = execute(&client, &run, alg.as_deref(), lang.as_deref())?;
    crate::common::print_json(&value)
}

fn execute<T: Transport>(
    transport: &T,
    run: &str,
    algorithm: Option<&str>,
    language: Option<&str>,
) -> anyhow::Result<Value> {
    let run_id =
        RunId::from_str(run).map_err(|_| jope_runtime::Failure::input("invalid run id"))?;
    let language = language
        .map(|language| {
            serde_json::from_value::<Language>(Value::String(language.to_ascii_lowercase()))
                .map_err(|_| jope_runtime::Failure::input("invalid source language"))
        })
        .transpose()?;
    let algorithm_id = match algorithm {
        Some(value) => AlgorithmId::from_str(value)
            .map_err(|_| jope_runtime::Failure::input("invalid algorithm id"))?,
        None => {
            transport
                .run_status(&run_id)
                .context("get run status")?
                .best
                .ok_or_else(|| jope_runtime::Failure::input("run has no accepted algorithm"))?
                .algorithm_id
        }
    };
    let mut result = transport
        .algorithm_get(&run_id, &algorithm_id)
        .context("get algorithm source")?;
    if let Some(language) = language {
        result.source.retain(|source| source.language == language);
        if result.source.is_empty() {
            return Err(jope_runtime::Failure::input(
                "algorithm has no source for the requested language",
            ));
        }
    }
    serde_json::to_value(result).context("serialize algorithm response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_client::{Request, Response};
    use std::sync::Mutex;

    const RUN: &str = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";
    const SNAPSHOT: &str = r#"{"run_id":"run_01ARZ3NDEKTSV4RRFFQ69G5FAV","revision":"2","state":"completed","phase":"finalizing","pending":null,"effort":"medium","budget_used":{"evaluations":"10","wall_seconds":"1.0"},"budget_total":{"evaluations":"10","wall_seconds":"10"},"best":{"algorithm_id":"alg_0001","metric":"rmse","cv_value":0.5,"model_artifact":null},"event_watermark":"2","warnings":[],"completion_reason":"budget_evaluations"}"#;
    const ALGORITHM: &str = r#"{"algorithm_id":"alg_0001","bundle_digest":"bundle","conditioning_digest":"conditioning","cv_value":0.5,"metric":"rmse","model_artifact":null,"model_digest":"model","revision":"2","source":[{"language":"python","source":"predict(x)"},{"language":"rust","source":"predict(x)"}],"summary":"golden"}"#;

    #[derive(Default)]
    struct Mock(Mutex<Vec<Request>>);
    impl jope_client::Transport for Mock {
        fn execute(&self, request: Request) -> jope_client::Result<Response> {
            let target = request.target().to_owned();
            self.0.lock().unwrap().push(request);
            let body = if target.ends_with("/status") || target == format!("/v1/runs/{RUN}") {
                SNAPSHOT
            } else {
                ALGORITHM
            };
            let mut headers = jope_client::transport::HeaderMap::new();
            headers.insert(
                "content-type",
                jope_client::transport::HeaderValue::from_static("application/json"),
            );
            headers.insert(
                "jope-api-version",
                jope_client::transport::HeaderValue::from_static("1.0"),
            );
            Ok(Response::new(200, headers, body.as_bytes()))
        }
    }

    #[test]
    fn code_defaults_to_best_and_filters_requested_language() {
        let mock = Mock::default();
        let result = execute(&mock, RUN, None, Some("python")).unwrap();
        let mut expected: Value = serde_json::from_str(ALGORITHM).unwrap();
        expected["source"] = serde_json::json!([{"language":"python","source":"predict(x)"}]);
        assert_eq!(result, expected);
        let requests = mock.0.lock().unwrap();
        assert_eq!(requests[0].target(), format!("/v1/runs/{RUN}"));
        assert_eq!(
            requests[1].target(),
            format!("/v1/runs/{RUN}/algorithms/alg_0001")
        );
    }

    #[test]
    fn explicit_algorithm_returns_golden_api_response() {
        let mock = Mock::default();
        let result = execute(&mock, RUN, Some("alg_0001"), None).unwrap();
        assert_eq!(serde_json::to_string(&result).unwrap(), ALGORITHM);
        assert_eq!(mock.0.lock().unwrap().len(), 1);
    }
}
