//! Synchronous (`RequestResponse`) and asynchronous (`Event`) `edge -> engine` cell dispatch (`RT-12`).
//!
//! Fast `SoloSingleLambda` and `/predict` calls execute synchronously for minimum latency;
//! multi-wave (`Medium` / `High` / `Ultra` / `HybridFanOut`) and public API Gateway runs dispatch
//! asynchronously via AWS Lambda `X-Amz-Invocation-Type: Event` (`202 Accepted`) so `carbon-edge`
//! never blocks on multi-minute GP waves or API Gateway HTTP API v2's 30-second ceiling.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use jope_api::carbon::{
    CarbonCreateRunRequest, CarbonExplainRequest, CarbonExportRequest, CarbonPredictRequest,
    CarbonResumeRunRequest,
};
use serde::{Deserialize, Serialize};

use crate::{
    cells::lambda_client::{post_json_http_bounded, CarbonLambdaRpcClient},
    error::to_json_vec,
};

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
    Explain {
        run_id: String,
        explain: CarbonExplainRequest,
    },
    Export {
        run_id: String,
        export: CarbonExportRequest,
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
        let payload = to_json_vec(command)?;
        post_json_http_bounded(
            &self.agent,
            &self.invoke_url,
            &payload,
            false,
            "engine invoke",
        )
    }

    fn invoke_engine_async(&self, command: &CarbonEngineWireCommand) -> Result<(), String> {
        let payload = to_json_vec(command)?;
        let _ = post_json_http_bounded(
            &self.agent,
            &self.invoke_url,
            &payload,
            true,
            "async engine invoke",
        )?;
        Ok(())
    }
}

/// `CarbonAwsLambdaEngineInvoker` is a struct invoking a real AWS Lambda `engine` function
/// (`lambda://<function_name>`) via signed AWS SigV4 `POST /2015-03-31/functions/{name}/invocations`.
pub struct CarbonAwsLambdaEngineInvoker {
    rpc: CarbonLambdaRpcClient,
}

impl CarbonAwsLambdaEngineInvoker {
    pub fn new(region: impl Into<String>, function_name: impl Into<String>) -> Self {
        Self {
            rpc: CarbonLambdaRpcClient::new(region, function_name, "engine"),
        }
    }
}

impl CarbonEngineInvoker for CarbonAwsLambdaEngineInvoker {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
        let payload = to_json_vec(command)?;
        self.rpc.invoke_bytes(&payload, false, 4)
    }

    fn invoke_engine_async(&self, command: &CarbonEngineWireCommand) -> Result<(), String> {
        let payload = to_json_vec(command)?;
        let _ = self.rpc.invoke_bytes(&payload, true, 4)?;
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
