//! Shared SigV4 AWS Lambda Invoke (`2015-03-31`) and bounded HTTP RPC client (`RT-12`).
//!
//! Used by both `CarbonAwsLambdaEngineInvoker` (`edge -> engine`) and `CarbonAwsLambdaEvalInvoker`
//! (`engine -> eval`), supporting deterministic loopback endpoint overrides via
//! `CARBON_LAMBDA_ENDPOINT_URL` (`resolve_lambda_invoke_endpoint`) for real OS-process local
//! Lambda worker qualification.

use std::time::Duration;

use crate::{
    cells::engine_invoker::now_epoch_ms,
    security::sigv4::{
        compute_backoff_with_jitter_ms, resolve_lambda_invoke_endpoint,
        sign_aws_v4_headers_with_extra, CarbonAwsCredentials,
    },
    storage::cas::{read_bounded, MAX_RPC_RESPONSE_BYTES},
};

/// Bounded JSON `POST` helper shared by `CarbonHttpEngineInvoker` and `CarbonHttpEvalInvoker`.
pub(crate) fn post_json_http_bounded(
    agent: &ureq::Agent,
    url: &str,
    payload: &[u8],
    async_event: bool,
    label: &str,
) -> Result<Vec<u8>, String> {
    let mut req = agent
        .post(url)
        .config()
        .http_status_as_error(false)
        .build()
        .header("content-type", "application/json");
    if async_event {
        req = req.header("x-amz-invocation-type", "Event");
    }
    let mut resp = req
        .send(payload)
        .map_err(|e| format!("HTTP {label} failed: {e}"))?;
    let status = resp.status().as_u16();
    let body = read_bounded(resp.body_mut().as_reader(), MAX_RPC_RESPONSE_BYTES, label)?;
    if status >= 300 {
        return Err(format!(
            "{label} HTTP {status}: {}",
            String::from_utf8_lossy(&body)
        ));
    }
    Ok(body)
}

/// Shared SigV4 AWS Lambda `/2015-03-31/functions/{name}/invocations` client.
#[derive(Clone)]
pub struct CarbonLambdaRpcClient {
    region: String,
    function_name: String,
    role_label: &'static str,
    agent: ureq::Agent,
}

impl CarbonLambdaRpcClient {
    pub fn new(
        region: impl Into<String>,
        function_name: impl Into<String>,
        role_label: &'static str,
    ) -> Self {
        Self {
            region: region.into(),
            function_name: function_name.into(),
            role_label,
            agent: crate::tls::carbon_https_agent().clone(),
        }
    }

    pub fn invoke_bytes(
        &self,
        payload: &[u8],
        async_event: bool,
        max_attempts: usize,
    ) -> Result<Vec<u8>, String> {
        let creds = CarbonAwsCredentials::from_env()?;
        let (host, uri, canonical_query, url) =
            resolve_lambda_invoke_endpoint(&self.region, &self.function_name)?;
        let extra: &[(&str, &str)] = if async_event {
            &[("x-amz-invocation-type", "Event")]
        } else {
            &[]
        };
        let attempts = max_attempts.max(1);
        let mut last_err = String::new();
        for attempt in 0..attempts {
            if attempt > 0 {
                let wait_ms = compute_backoff_with_jitter_ms(attempt - 1, 120, 1_500);
                std::thread::sleep(Duration::from_millis(wait_ms));
            }
            let headers = sign_aws_v4_headers_with_extra(
                &creds,
                &self.region,
                "lambda",
                "POST",
                &host,
                &uri,
                &canonical_query,
                payload,
                now_epoch_ms() / 1_000,
                extra,
            );
            let mut req = self
                .agent
                .post(&url)
                .config()
                .http_status_as_error(false)
                .build()
                .header("content-type", "application/json");
            for (k, v) in &headers {
                req = req.header(k, v);
            }
            let mut resp = match req.send(payload) {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!(
                        "AWS Lambda {} invoke '{}' failed: {e}",
                        self.role_label, self.function_name
                    );
                    if attempt + 1 < attempts {
                        continue;
                    }
                    return Err(last_err);
                }
            };
            let status = resp.status().as_u16();
            let fn_err = resp
                .headers()
                .get("x-amz-function-error")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned);
            let body = read_bounded(
                resp.body_mut().as_reader(),
                MAX_RPC_RESPONSE_BYTES,
                "AWS Lambda invoke response",
            )?;
            if matches!(status, 429 | 502 | 503 | 504) && attempt + 1 < attempts {
                last_err = format!(
                    "AWS Lambda {} invoke HTTP {status}: {}",
                    self.role_label,
                    String::from_utf8_lossy(&body)
                );
                continue;
            }
            if status >= 300 {
                return Err(format!(
                    "AWS Lambda {} invoke HTTP {status}: {}",
                    self.role_label,
                    String::from_utf8_lossy(&body)
                ));
            }
            if let Some(err_kind) = fn_err {
                return Err(format!(
                    "AWS Lambda {} '{}' function error ({err_kind}): {}",
                    self.role_label,
                    self.function_name,
                    String::from_utf8_lossy(&body)
                ));
            }
            return Ok(body);
        }
        Err(last_err)
    }
}
