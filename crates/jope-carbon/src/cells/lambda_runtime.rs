//! AWS Lambda Custom Runtime (`provided.al2023`) HTTP/1.1 event loop (`ADR-0007` Phase C4).
//!
//! Polls `http://${LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/next`, extracts
//! `Lambda-Runtime-Aws-Request-Id` and `Lambda-Runtime-Deadline-Ms`, invokes the cell handler,
//! and posts the response or structured runtime error back to the Lambda Runtime API.

use crate::error::CarbonErrCtx as _;
use std::io::Read;
use std::time::Duration;

/// `CarbonLambdaInvocation` is a struct representing one dequeued AWS Lambda Runtime API event.
#[derive(Clone, Debug)]
pub struct CarbonLambdaInvocation {
    pub request_id: String,
    pub deadline_epoch_ms: Option<u64>,
    pub payload: Vec<u8>,
}

/// Execute the AWS Lambda Custom Runtime loop against `runtime_api_host_port`, stopping cleanly
/// after `max_invocations` (when `Some`) or when the runtime server closes (`410 Gone` in `carbon emu`).
pub fn run_lambda_custom_runtime_loop<F>(
    runtime_api_host_port: &str,
    max_invocations: Option<usize>,
    mut handler: F,
) -> Result<usize, String>
where
    F: FnMut(CarbonLambdaInvocation) -> Result<Vec<u8>, String>,
{
    let base = format!(
        "http://{}/2018-06-01/runtime/invocation",
        runtime_api_host_port
            .trim_start_matches("http://")
            .trim_end_matches('/')
    );
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(905)))
        .http_status_as_error(false)
        .build()
        .into();

    let mut processed = 0usize;
    loop {
        if let Some(limit) = max_invocations {
            if processed >= limit {
                return Ok(processed);
            }
        }
        let next_url = format!("{base}/next");
        let mut next_resp = agent
            .get(&next_url)
            .call()
            .err_ctx("Lambda Runtime API /next poll failed")?;
        let status = next_resp.status().as_u16();
        if status == 410 {
            return Ok(processed);
        }
        if status != 200 {
            return Err(format!("Lambda Runtime API /next returned HTTP {status}"));
        }

        let request_id = next_resp
            .headers()
            .get("lambda-runtime-aws-request-id")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| "missing Lambda-Runtime-Aws-Request-Id header".to_owned())?
            .to_owned();
        let deadline_epoch_ms = next_resp
            .headers()
            .get("lambda-runtime-deadline-ms")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());

        const MAX_LAMBDA_EVENT_BYTES: u64 = 48 * 1_048_576;
        let mut payload = Vec::new();
        next_resp
            .body_mut()
            .as_reader()
            .take(MAX_LAMBDA_EVENT_BYTES.saturating_add(1))
            .read_to_end(&mut payload)
            .err_ctx("read invocation payload")?;
        if payload.len() as u64 > MAX_LAMBDA_EVENT_BYTES {
            return Err(format!(
                "Lambda Runtime API /next payload exceeds {MAX_LAMBDA_EVENT_BYTES} byte limit"
            ));
        }

        let invocation = CarbonLambdaInvocation {
            request_id: request_id.clone(),
            deadline_epoch_ms,
            payload,
        };
        match handler(invocation) {
            Ok(response_bytes) => {
                let resp_url = format!("{base}/{request_id}/response");
                let post_resp = agent
                    .post(&resp_url)
                    .send(&response_bytes)
                    .err_ctx("post invocation response failed")?;
                if post_resp.status().as_u16() >= 300 {
                    return Err(format!(
                        "Lambda Runtime API /response returned HTTP {}",
                        post_resp.status()
                    ));
                }
            }
            Err(err_msg) => {
                let err_url = format!("{base}/{request_id}/error");
                let err_json = serde_json::json!({
                    "errorMessage": err_msg,
                    "errorType": "CarbonHandlerError"
                });
                let body = crate::error::to_json_vec(&err_json)?;
                let _ = agent
                    .post(&err_url)
                    .header("content-type", "application/json")
                    .send(&body);
            }
        }
        processed = processed.saturating_add(1);
    }
}
