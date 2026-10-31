use crate::cli::RunsCommand;
use anyhow::Context;
use jope_api::ids::RunId;
use jope_client::Transport;
use serde_json::Value;
use std::str::FromStr;

pub fn run(command: RunsCommand) -> anyhow::Result<()> {
    let client = crate::common::resolve_client()?;
    let value = execute(&client, command)?;
    crate::common::print_json(&value)
}

fn execute<T: Transport>(transport: &T, command: RunsCommand) -> anyhow::Result<Value> {
    match command {
        RunsCommand::List => Ok(serde_json::to_value(
            transport.runs().context("list runs")?,
        )?),
        RunsCommand::Delete {
            run,
            idempotency_key,
        } => {
            let id = RunId::from_str(&run)
                .map_err(|_| jope_runtime::Failure::input("invalid run id"))?;
            let key = crate::common::resolve_idempotency_key(idempotency_key.as_deref());
            crate::common::announce_generated_key(idempotency_key.as_deref(), &key)?;
            Ok(serde_json::to_value(
                transport.run_delete(&id, &key).context("delete run")?,
            )?)
        }
        RunsCommand::ExportFeedback { run, out } => {
            let id = RunId::from_str(&run)
                .map_err(|_| jope_runtime::Failure::input("invalid run id"))?;
            let value = serde_json::to_value(
                transport
                    .feedback_export(&id)
                    .context("export run feedback")?,
            )?;
            let bytes = serde_json::to_vec_pretty(&value).context("serialize feedback response")?;
            std::fs::write(&out, bytes)
                .map_err(|error| jope_runtime::Failure::output(error.to_string()))?;
            Ok(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_client::{Request, Response};
    use std::sync::Mutex;

    const RUN: &str = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";
    #[derive(Default)]
    struct Mock(Mutex<Vec<Request>>);
    impl jope_client::Transport for Mock {
        fn execute(&self, request: Request) -> jope_client::Result<Response> {
            self.0.lock().unwrap().push(request);
            let response = match self.0.lock().unwrap().last().unwrap().target() {
                "/v1/runs" => r#"{"runs":[]}"#,
                path if path.ends_with("/feedback") => r#"{"records":[]}"#,
                _ => r#"{"deleted":true}"#,
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
            Ok(Response::new(200, headers, response.as_bytes()))
        }
    }

    #[test]
    fn list_has_golden_api_json_and_registry_route() {
        let mock = Mock::default();
        let value = execute(&mock, RunsCommand::List).unwrap();
        assert_eq!(
            serde_json::to_string(&value).unwrap(),
            r#"{"next_after":null,"runs":[]}"#
        );
        assert_eq!(mock.0.lock().unwrap()[0].target(), "/v1/runs");
    }

    #[test]
    fn delete_has_golden_api_json_and_uses_run_route() {
        let mock = Mock::default();
        let value = execute(
            &mock,
            RunsCommand::Delete {
                run: RUN.into(),
                idempotency_key: Some("delete-1".into()),
            },
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&value).unwrap(),
            r#"{"deleted":true}"#
        );
        assert_eq!(
            mock.0.lock().unwrap()[0].target(),
            format!("/v1/runs/{RUN}")
        );
    }

    #[test]
    fn feedback_export_writes_and_returns_golden_api_json() {
        let mock = Mock::default();
        let temp = tempfile::tempdir().unwrap();
        let out = temp.path().join("feedback.json");
        let value = execute(
            &mock,
            RunsCommand::ExportFeedback {
                run: RUN.into(),
                out: out.clone(),
            },
        )
        .unwrap();
        assert_eq!(serde_json::to_string(&value).unwrap(), r#"{"records":[]}"#);
        assert_eq!(
            serde_json::from_slice::<Value>(&std::fs::read(out).unwrap()).unwrap(),
            value
        );
    }
}
