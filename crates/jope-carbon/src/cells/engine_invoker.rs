//! Synchronous (`RequestResponse`) and asynchronous (`Event`) `edge -> engine` cell dispatch (`RT-12`).
//!
//! Fast `SoloSingleLambda` and `/predict` calls execute synchronously for minimum latency;
//! multi-wave (`Medium` / `High` / `Ultra` / `HybridFanOut`) runs dispatch asynchronously via
//! AWS Lambda `X-Amz-Invocation-Type: Event` (`202 Accepted`) so `carbon-edge` never blocks on
//! multi-minute GP waves.

use std::io::Read;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use jope_api::carbon::{CarbonCreateRunRequest, CarbonPredictRequest, CarbonResumeRunRequest};
use serde::{Deserialize, Serialize};

/// Return current UNIX epoch timestamp in milliseconds.
pub fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

/// `CarbonEngineWireCommand` is an enum serialized across the `edge -> engine` cell boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum CarbonEngineWireCommand {
    StartRun {
        request: CarbonCreateRunRequest,
        region: String,
    },
    ResumeRun {
        run_id: String,
        resume: CarbonResumeRunRequest,
        region: String,
    },
    Predict {
        run_id: String,
        predict: CarbonPredictRequest,
    },
}

/// `CarbonEngineInvoker` is a trait abstracting synchronous and asynchronous `Event` dispatch from
/// `edge` to the sealed `engine` cell.
pub trait CarbonEngineInvoker: Send + Sync {
    /// Invoke `engine` synchronously (`RequestResponse`) and return its serialized JSON response.
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String>;

    /// Enqueue `command` on `engine` asynchronously (`Event` / `202 Accepted`) without blocking
    /// `carbon-edge` (`RT-12`).
    fn invoke_engine_async(&self, command: &CarbonEngineWireCommand) -> Result<(), String>;
}

/// `CarbonHttpEngineInvoker` is a struct invoking a remote `engine` HTTP endpoint (`carbon emu`).
pub struct CarbonHttpEngineInvoker {
    invoke_url: String,
    agent: ureq::Agent,
}

impl CarbonHttpEngineInvoker {
    pub fn new(invoke_url: impl Into<String>) -> Self {
        Self {
            invoke_url: invoke_url.into(),
            agent: crate::tls::carbon_https_agent().clone(),
        }
    }
}

impl CarbonEngineInvoker for CarbonHttpEngineInvoker {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
        let payload =
            serde_json::to_vec(command).map_err(|e| format!("encode engine command: {e}"))?;
        let mut resp = self
            .agent
            .post(&self.invoke_url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("content-type", "application/json")
            .send(&payload)
            .map_err(|e| format!("HTTP engine invoke failed: {e}"))?;
        let status = resp.status().as_u16();
        let mut body = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("read engine invoke body: {e}"))?;
        if status >= 300 {
            return Err(format!(
                "engine invoke HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        Ok(body)
    }

    fn invoke_engine_async(&self, command: &CarbonEngineWireCommand) -> Result<(), String> {
        let payload =
            serde_json::to_vec(command).map_err(|e| format!("encode async engine command: {e}"))?;
        let mut resp = self
            .agent
            .post(&self.invoke_url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("content-type", "application/json")
            .header("x-amz-invocation-type", "Event")
            .send(&payload)
            .map_err(|e| format!("HTTP async engine invoke failed: {e}"))?;
        let status = resp.status().as_u16();
        if status >= 300 {
            let mut body = Vec::new();
            let _ = resp.body_mut().as_reader().read_to_end(&mut body);
            return Err(format!(
                "async engine invoke HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        Ok(())
    }
}

/// `CarbonAwsLambdaEngineInvoker` is a struct invoking a real AWS Lambda `engine` function
/// (`lambda://<function_name>`) via signed AWS SigV4 `POST /2015-03-31/functions/{name}/invocations`.
pub struct CarbonAwsLambdaEngineInvoker {
    region: String,
    function_name: String,
    agent: ureq::Agent,
}

impl CarbonAwsLambdaEngineInvoker {
    pub fn new(region: impl Into<String>, function_name: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            function_name: function_name.into(),
            agent: crate::tls::carbon_https_agent().clone(),
        }
    }

    fn send_lambda_invoke(
        &self,
        command: &CarbonEngineWireCommand,
        async_event: bool,
    ) -> Result<Vec<u8>, String> {
        let payload =
            serde_json::to_vec(command).map_err(|e| format!("encode engine command: {e}"))?;
        let creds = crate::security::sigv4::CarbonAwsCredentials::from_env()?;
        let host = format!("lambda.{}.amazonaws.com", self.region);
        let (fn_base, qualifier) =
            crate::security::sigv4::split_lambda_name_and_qualifier(&self.function_name);
        let uri = format!(
            "/2015-03-31/functions/{}/invocations",
            crate::security::sigv4::url_encode_component(fn_base)
        );
        let canonical_query = match qualifier {
            Some(q) => format!(
                "Qualifier={}",
                crate::security::sigv4::url_encode_component(q)
            ),
            None => String::new(),
        };
        let url = if canonical_query.is_empty() {
            format!("https://{host}{uri}")
        } else {
            format!("https://{host}{uri}?{canonical_query}")
        };
        let extra: &[(&str, &str)] = if async_event {
            &[("x-amz-invocation-type", "Event")]
        } else {
            &[]
        };
        let headers = crate::security::sigv4::sign_aws_v4_headers_with_extra(
            &creds,
            &self.region,
            "lambda",
            "POST",
            &host,
            &uri,
            &canonical_query,
            &payload,
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
        let mut resp = req.send(&payload).map_err(|e| {
            format!(
                "AWS Lambda engine invoke '{}' failed: {e}",
                self.function_name
            )
        })?;
        let status = resp.status().as_u16();
        let fn_err = resp
            .headers()
            .get("x-amz-function-error")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let mut body = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("read AWS Lambda engine invoke body: {e}"))?;
        if status >= 300 {
            return Err(format!(
                "AWS Lambda engine invoke HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        if let Some(err_kind) = fn_err {
            return Err(format!(
                "AWS Lambda engine '{}' returned function error ({err_kind}): {}",
                self.function_name,
                String::from_utf8_lossy(&body)
            ));
        }
        Ok(body)
    }
}

impl CarbonEngineInvoker for CarbonAwsLambdaEngineInvoker {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
        self.send_lambda_invoke(command, false)
    }

    fn invoke_engine_async(&self, command: &CarbonEngineWireCommand) -> Result<(), String> {
        let _ = self.send_lambda_invoke(command, true)?;
        Ok(())
    }
}

/// Construct either a `CarbonAwsLambdaEngineInvoker` (`lambda://<function_name>`) or a
/// `CarbonHttpEngineInvoker` (`http(s)://...`).
pub fn open_engine_invoker(target: &str, default_region: &str) -> Arc<dyn CarbonEngineInvoker> {
    if let Some(fn_name) = target.trim().strip_prefix("lambda://") {
        let region = match std::env::var("AWS_REGION") {
            Ok(val) if !val.trim().is_empty() => val,
            _ => default_region.to_owned(),
        };
        return Arc::new(CarbonAwsLambdaEngineInvoker::new(region, fn_name.trim()));
    }
    Arc::new(CarbonHttpEngineInvoker::new(target))
}
