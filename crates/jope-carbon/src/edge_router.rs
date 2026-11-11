//! Regional `carbon-edge` HTTP / AWS Lambda Function URL request router (`ADR-0007` Phase C4).
//!
//! Compiled under `--no-default-features --features edge`: depends ONLY on wire schemas,
//! `jope-prep` landmark header verification, `aws-lc-rs` presigning, and CAS state reads/writes.
//! Zero search, fitting, or neural crates enter this binary (`CT3` isolation).

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonCreateRunRequest, CarbonEarlyStopRequest, CarbonExecutionModeWire,
        CarbonPingResponse, CarbonPredictRequest, CarbonPredictResponse,
        CarbonPrepareUploadRequest, CarbonPrepareUploadResponse, CarbonPresignedTarget,
        CarbonPublicWaveProgress, CarbonRegionCatalog, CarbonResumeRunRequest, CarbonRunStateWire,
        CarbonTierWire,
    },
    decimal::DecimalU64,
    Effort,
};
use jope_prep::conditioning::FitLandmarkBundle;
use serde::{Deserialize, Serialize};

use crate::{
    cas::CarbonCasStore,
    planner::{plan_carbon_execution, CarbonExecutionMode, CarbonPlanInput},
    presign::CarbonPresigner,
    role::{CarbonEffortTier, CarbonProductTier},
    spot_advisor::{evaluate_spot_hybrid_gates, MonthlyWindowMetrics, SpotAdvisorThresholds},
};

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

/// `CarbonEngineInvoker` is a trait abstracting synchronous or async-queued dispatch from `edge`
/// to the sealed `engine` cell (Lambda Invoke HTTP endpoint, `carbon emu` runtime queue, or
/// in-process Enterprise `carbon serve` pool).
pub trait CarbonEngineInvoker: Send + Sync {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String>;
}

/// `CarbonHttpEngineInvoker` is a struct invoking a remote `engine` HTTP endpoint (`carbon emu`
/// or Lambda Function/Invoke proxy) via `carbon_https_agent()`.
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
}

/// `CarbonGatewayRequest` is a struct representing an HTTP or Lambda Function URL v2.0 request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonGatewayRequest {
    pub method: String,
    pub path: String,
    #[serde(default)]
    pub query: BTreeMap<String, String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub body_b64: Option<String>,
    #[serde(default)]
    pub raw_body: Vec<u8>,
}

impl CarbonGatewayRequest {
    pub fn body_bytes(&self) -> Result<Vec<u8>, String> {
        if !self.raw_body.is_empty() {
            return Ok(self.raw_body.clone());
        }
        if let Some(b64) = &self.body_b64 {
            return base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| format!("invalid base64 request body: {e}"));
        }
        Ok(Vec::new())
    }
}

/// `CarbonGatewayResponse` is a struct representing an HTTP / Lambda Function URL v2.0 response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonGatewayResponse {
    pub status: u16,
    pub content_type: String,
    #[serde(default)]
    pub body_b64: String,
}

impl CarbonGatewayResponse {
    pub fn json_ok<T: Serialize>(status: u16, value: &T) -> Self {
        match serde_json::to_vec(value) {
            Ok(bytes) => Self {
                status,
                content_type: "application/json".to_owned(),
                body_b64: base64::engine::general_purpose::STANDARD.encode(bytes),
            },
            Err(err) => Self::text(
                500,
                "application/json",
                &format!(
                    "{{\"error\":\"serialize_failed\",\"message\":{:?}}}",
                    err.to_string()
                ),
            ),
        }
    }

    pub fn text(status: u16, content_type: &str, text: &str) -> Self {
        Self {
            status,
            content_type: content_type.to_owned(),
            body_b64: base64::engine::general_purpose::STANDARD.encode(text.as_bytes()),
        }
    }

    pub fn error_json(status: u16, code: &str, message: &str) -> Self {
        Self::json_ok(
            status,
            &serde_json::json!({ "error": code, "message": message }),
        )
    }

    pub fn decoded_body(&self) -> Vec<u8> {
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&self.body_b64) {
            return bytes;
        }
        Vec::new()
    }
}

/// `CarbonEdgeConfig` is a struct configuring a regional `carbon-edge` or Enterprise gateway.
#[derive(Clone)]
pub struct CarbonEdgeConfig {
    pub region: String,
    pub public_base_url: String,
    pub enterprise_self_hosted: bool,
    pub catalog: CarbonRegionCatalog,
    pub presigner: CarbonPresigner,
}

/// `CarbonEdgeRouter` is a struct handling all public `/v1/carbon/*` and discovery endpoints.
pub struct CarbonEdgeRouter {
    config: CarbonEdgeConfig,
    cas: Arc<dyn CarbonCasStore>,
    engine: Arc<dyn CarbonEngineInvoker>,
}

impl CarbonEdgeRouter {
    pub fn new(
        config: CarbonEdgeConfig,
        cas: Arc<dyn CarbonCasStore>,
        engine: Arc<dyn CarbonEngineInvoker>,
    ) -> Self {
        Self {
            config,
            cas,
            engine,
        }
    }

    pub fn handle(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let method = req.method.to_ascii_uppercase();
        let path = req.path.trim_end_matches('/');
        let clean_path = if path.is_empty() { "/" } else { path };

        match (method.as_str(), clean_path) {
            ("GET", "/.well-known/carbon-regions.json") => {
                CarbonGatewayResponse::json_ok(200, &self.config.catalog)
            }
            ("GET", "/v1/carbon/ping") => CarbonGatewayResponse::json_ok(
                200,
                &CarbonPingResponse {
                    region: self.config.region.clone(),
                    healthy: true,
                    server_epoch_ms: DecimalU64(now_epoch_ms()),
                    enterprise_self_hosted: self.config.enterprise_self_hosted,
                },
            ),
            ("POST", "/v1/carbon/uploads/prepare") => self.handle_prepare_upload(req),
            ("POST", "/v1/carbon/runs") => self.handle_create_run(req),
            ("GET", "/v1/carbon/advisor/spot-hybrid") => self.handle_spot_advisor(req),
            _ if clean_path.starts_with("/v1/carbon/cas/") && method == "PUT" => {
                let sha = &clean_path["/v1/carbon/cas/".len()..];
                self.handle_put_cas_blob(sha, req)
            }
            _ if clean_path.starts_with("/v1/carbon/runs/") => {
                let rest = &clean_path["/v1/carbon/runs/".len()..];
                if let Some(run_id) = rest.strip_suffix("/events") {
                    if method == "GET" {
                        return self.handle_run_events(run_id, req);
                    }
                } else if let Some(run_id) = rest.strip_suffix("/stop") {
                    if method == "POST" {
                        return self.handle_early_stop(run_id, req);
                    }
                } else if let Some(run_id) = rest.strip_suffix("/resume") {
                    if method == "POST" {
                        return self.handle_resume(run_id, req);
                    }
                } else if let Some(run_id) = rest.strip_suffix("/predict") {
                    if method == "POST" {
                        return self.handle_predict(run_id, req);
                    }
                } else if !rest.contains('/') && method == "GET" {
                    return self.handle_get_run(rest, req);
                }
                CarbonGatewayResponse::error_json(404, "not_found", "unknown Carbon run route")
            }
            _ => CarbonGatewayResponse::error_json(404, "not_found", "unknown Carbon endpoint"),
        }
    }

    fn effective_tier(&self, requested: CarbonTierWire) -> CarbonProductTier {
        if self.config.enterprise_self_hosted {
            return CarbonProductTier::no_limit();
        }
        match requested {
            CarbonTierWire::Free => CarbonProductTier::Free,
            CarbonTierWire::Pro => CarbonProductTier::Pro,
            CarbonTierWire::Enterprise => CarbonProductTier::Enterprise,
        }
    }

    fn resolve_public_base_url(&self, req: &CarbonGatewayRequest) -> String {
        if !self.config.public_base_url.trim().is_empty() {
            return self.config.public_base_url.clone();
        }
        if let Some(host) = req.headers.get("host").or_else(|| req.headers.get("Host")) {
            let proto = req
                .headers
                .get("x-forwarded-proto")
                .map(String::as_str)
                .unwrap_or("http");
            return format!("{proto}://{host}");
        }
        "http://127.0.0.1:8080".to_owned()
    }

    fn handle_prepare_upload(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        let Ok(prep) = serde_json::from_slice::<CarbonPrepareUploadRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonPrepareUploadRequest",
            );
        };
        if prep.tenant_id.trim().is_empty() {
            return CarbonGatewayResponse::error_json(400, "invalid_tenant", "empty tenant_id");
        }
        let tier = self.effective_tier(prep.tier);
        let total_rows = (prep.fit_rows as usize).saturating_add(prep.holdout_rows as usize);
        if let Err(err) = tier.validate_admission(
            total_rows,
            prep.feature_count as usize,
            CarbonEffortTier::Fast,
        ) {
            return CarbonGatewayResponse::error_json(403, "tier_limit", &err.to_string());
        }
        let expires = (now_epoch_ms() / 1000).saturating_add(900);
        let base_url = self.resolve_public_base_url(req);
        let fit_url = self.config.presigner.sign_gateway_upload_url(
            &base_url,
            &prep.fit_jmat.sha256,
            prep.fit_jmat.bytes.0,
            expires,
        );
        let holdout_url = self.config.presigner.sign_gateway_upload_url(
            &base_url,
            &prep.holdout_jmat.sha256,
            prep.holdout_jmat.bytes.0,
            expires,
        );
        CarbonGatewayResponse::json_ok(
            200,
            &CarbonPrepareUploadResponse {
                upload_id: format!("upl_{}", ulid::Ulid::new()),
                fit_target: CarbonPresignedTarget {
                    url: fit_url,
                    method: "PUT".to_owned(),
                    sha256: prep.fit_jmat.sha256.to_ascii_lowercase(),
                    expires_epoch_secs: DecimalU64(expires),
                },
                holdout_target: CarbonPresignedTarget {
                    url: holdout_url,
                    method: "PUT".to_owned(),
                    sha256: prep.holdout_jmat.sha256.to_ascii_lowercase(),
                    expires_epoch_secs: DecimalU64(expires),
                },
            },
        )
    }

    fn handle_put_cas_blob(&self, sha: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Some(bytes_str) = req.query.get("bytes") else {
            return CarbonGatewayResponse::error_json(400, "missing_param", "missing bytes query");
        };
        let Some(exp_str) = req.query.get("exp") else {
            return CarbonGatewayResponse::error_json(400, "missing_param", "missing exp query");
        };
        let Some(sig) = req.query.get("sig") else {
            return CarbonGatewayResponse::error_json(401, "missing_sig", "missing sig query");
        };
        let (Ok(declared_bytes), Ok(exp_secs)) = (bytes_str.parse::<u64>(), exp_str.parse::<u64>())
        else {
            return CarbonGatewayResponse::error_json(400, "bad_query", "invalid bytes or exp");
        };
        let now_secs = now_epoch_ms() / 1000;
        if let Err(reason) = self.config.presigner.verify_gateway_upload(
            sha,
            declared_bytes,
            exp_secs,
            sig,
            now_secs,
        ) {
            return CarbonGatewayResponse::error_json(403, "presign_denied", reason);
        }
        let Ok(payload) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        if payload.len() as u64 != declared_bytes {
            return CarbonGatewayResponse::error_json(
                400,
                "size_mismatch",
                "uploaded body size differs from presigned byte count",
            );
        }
        if let Err(err) = self.cas.put_blob_verified(sha, &payload) {
            return CarbonGatewayResponse::error_json(400, "cas_integrity", &err);
        }
        CarbonGatewayResponse::json_ok(200, &serde_json::json!({ "stored_sha256": sha }))
    }

    fn handle_create_run(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        let Ok(mut create) = serde_json::from_slice::<CarbonCreateRunRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonCreateRunRequest",
            );
        };
        if self.config.enterprise_self_hosted {
            create.tier = CarbonTierWire::Enterprise;
        }
        let tier = self.effective_tier(create.tier);
        let effort = map_api_effort(create.effort);
        let total_rows = (create.fit_rows as usize).saturating_add(create.holdout_rows as usize);
        if let Err(err) = tier.validate_admission(total_rows, create.feature_names.len(), effort) {
            return CarbonGatewayResponse::error_json(403, "tier_limit", &err.to_string());
        }
        let Ok(lm_raw) =
            base64::engine::general_purpose::STANDARD.decode(&create.landmark_bundle_b64)
        else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_landmarks",
                "landmark_bundle_b64 is not valid base64",
            );
        };
        let Ok(lm_bundle) = FitLandmarkBundle::from_wire_bytes(&lm_raw) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_landmarks",
                "JLMB v1 FitLandmarkBundle failed integrity or bounds verification",
            );
        };
        if lm_bundle.feature_count() != create.feature_names.len() {
            return CarbonGatewayResponse::error_json(
                400,
                "landmark_feature_mismatch",
                "FitLandmarkBundle feature_count does not match feature_names",
            );
        }
        let plan_decision = match plan_carbon_execution(&CarbonPlanInput {
            product_tier: tier,
            effort,
            total_rows,
            fit_rows: create.fit_rows as usize,
            features: create.feature_names.len(),
            est_ms_per_candidate_1vcpu: 12.0,
        }) {
            Ok(d) => d,
            Err(err) => {
                return CarbonGatewayResponse::error_json(
                    400,
                    "planner_rejected",
                    &err.to_string(),
                );
            }
        };

        let run_str = create.run_id.to_string();
        let req_key = state_key(&create.tenant_id, &run_str, "request.json");
        let Ok(stored_req_bytes) = serde_json::to_vec(&create) else {
            return CarbonGatewayResponse::error_json(
                500,
                "serialize_failed",
                "failed to encode create request",
            );
        };
        if let Err(err) = self.cas.put_state_object(&req_key, &stored_req_bytes) {
            return CarbonGatewayResponse::error_json(500, "cas_write_failed", &err);
        }

        let initial_progress = CarbonPublicWaveProgress {
            run_id: create.run_id.clone(),
            state: CarbonRunStateWire::Queued,
            execution_mode: match plan_decision.mode {
                CarbonExecutionMode::SoloSingleLambda => CarbonExecutionModeWire::SoloSingleLambda,
                CarbonExecutionMode::HybridFanOut => CarbonExecutionModeWire::HybridFanOut,
            },
            eval_memory_mb: plan_decision.eval_class.memory_mb() as u32,
            eval_shards: plan_decision.shards_per_wave.max(1),
            wave_index: 0,
            waves_target: effort.planned_waves(),
            candidates_evaluated: DecimalU64(0),
            best_cv_loss: None,
            best_cv_metric: None,
            outer_holdout_loss: None,
            outer_holdout_metric: None,
            neural_seed_applied: false,
            can_resume: false,
            sealed_checkpoint_sha256: None,
            sealed_champion_sha256: None,
            elapsed_ms: DecimalU64(0),
            region: self.config.region.clone(),
            failure_message: None,
        };
        let Ok(status_bytes) = serde_json::to_vec(&initial_progress) else {
            return CarbonGatewayResponse::error_json(
                500,
                "serialize_failed",
                "failed to encode initial run progress",
            );
        };
        if let Err(err) = self.cas.put_state_object(
            &state_key(&create.tenant_id, &run_str, "status.json"),
            &status_bytes,
        ) {
            return CarbonGatewayResponse::error_json(500, "cas_write_failed", &err);
        }

        let cmd = CarbonEngineWireCommand::StartRun {
            request: create,
            region: self.config.region.clone(),
        };
        match self.engine.invoke_engine(&cmd) {
            Ok(resp_bytes) => {
                if let Ok(progress) =
                    serde_json::from_slice::<CarbonPublicWaveProgress>(&resp_bytes)
                {
                    CarbonGatewayResponse::json_ok(201, &progress)
                } else {
                    CarbonGatewayResponse::json_ok(201, &initial_progress)
                }
            }
            Err(err) => CarbonGatewayResponse::error_json(502, "engine_invoke_failed", &err),
        }
    }

    fn handle_get_run(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Some(tenant) = req.query.get("tenant_id").filter(|t| !t.trim().is_empty()) else {
            return CarbonGatewayResponse::error_json(
                400,
                "missing_tenant",
                "tenant_id query parameter is required",
            );
        };
        let key = state_key(tenant, run_id, "status.json");
        match self.cas.get_state_object(&key) {
            Ok(Some(bytes)) => match serde_json::from_slice::<CarbonPublicWaveProgress>(&bytes) {
                Ok(progress) => CarbonGatewayResponse::json_ok(200, &progress),
                Err(_) => {
                    CarbonGatewayResponse::error_json(500, "corrupt_status", "invalid status")
                }
            },
            Ok(None) => CarbonGatewayResponse::error_json(404, "run_not_found", "run not found"),
            Err(err) => CarbonGatewayResponse::error_json(500, "cas_read_failed", &err),
        }
    }

    fn handle_run_events(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Some(tenant) = req.query.get("tenant_id").filter(|t| !t.trim().is_empty()) else {
            return CarbonGatewayResponse::error_json(
                400,
                "missing_tenant",
                "tenant_id query parameter is required",
            );
        };
        let key = state_key(tenant, run_id, "status.json");
        match self.cas.get_state_object(&key) {
            Ok(Some(bytes)) => {
                let text = String::from_utf8_lossy(&bytes);
                let sse = format!("event: wave_progress\ndata: {text}\n\n");
                CarbonGatewayResponse::text(200, "text/event-stream", &sse)
            }
            Ok(None) => CarbonGatewayResponse::error_json(404, "run_not_found", "run not found"),
            Err(err) => CarbonGatewayResponse::error_json(500, "cas_read_failed", &err),
        }
    }

    fn handle_early_stop(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        let Ok(stop_req) = serde_json::from_slice::<CarbonEarlyStopRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonEarlyStopRequest",
            );
        };
        let stop_key = state_key(&stop_req.tenant_id, run_id, "stop.json");
        if let Err(err) = self.cas.put_state_object(&stop_key, &body) {
            return CarbonGatewayResponse::error_json(500, "cas_write_failed", &err);
        }
        CarbonGatewayResponse::json_ok(
            202,
            &serde_json::json!({ "run_id": run_id, "stop_signaled": true }),
        )
    }

    fn handle_resume(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        let Ok(resume_req) = serde_json::from_slice::<CarbonResumeRunRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonResumeRunRequest",
            );
        };
        let cmd = CarbonEngineWireCommand::ResumeRun {
            run_id: run_id.to_owned(),
            resume: resume_req,
            region: self.config.region.clone(),
        };
        match self.engine.invoke_engine(&cmd) {
            Ok(bytes) => match serde_json::from_slice::<CarbonPublicWaveProgress>(&bytes) {
                Ok(progress) => CarbonGatewayResponse::json_ok(200, &progress),
                Err(_) => CarbonGatewayResponse::error_json(
                    500,
                    "invalid_engine_response",
                    "malformed resume progress",
                ),
            },
            Err(err) => CarbonGatewayResponse::error_json(502, "engine_resume_failed", &err),
        }
    }

    fn handle_predict(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        let Ok(pred_req) = serde_json::from_slice::<CarbonPredictRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonPredictRequest",
            );
        };
        let cmd = CarbonEngineWireCommand::Predict {
            run_id: run_id.to_owned(),
            predict: pred_req,
        };
        match self.engine.invoke_engine(&cmd) {
            Ok(bytes) => match serde_json::from_slice::<CarbonPredictResponse>(&bytes) {
                Ok(resp) => CarbonGatewayResponse::json_ok(200, &resp),
                Err(_) => CarbonGatewayResponse::error_json(
                    500,
                    "invalid_engine_response",
                    "malformed predict response",
                ),
            },
            Err(err) => CarbonGatewayResponse::error_json(502, "engine_predict_failed", &err),
        }
    }

    fn handle_spot_advisor(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let spend = req
            .query
            .get("monthly_spend_usd")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(450.0);
        let p20 = req
            .query
            .get("p20_vcpus")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(2.0);
        let mean = req
            .query
            .get("mean_vcpus")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(5.0);
        let p95 = req
            .query
            .get("p95_vcpus")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(18.0);
        let cold = req
            .query
            .get("cold_start_rate")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.02);
        let months = req
            .query
            .get("months")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 12);
        let window =
            MonthlyWindowMetrics::for_region(&self.config.region, spend, p20, mean, p95, cold);
        let history = vec![window; months];
        let verdict = evaluate_spot_hybrid_gates(&history, &SpotAdvisorThresholds::default());
        CarbonGatewayResponse::json_ok(200, &verdict)
    }
}

pub(crate) fn state_key(tenant_id: &str, run_id: &str, leaf: &str) -> String {
    let clean_tenant: String = tenant_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let clean_run: String = run_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("runs/{clean_tenant}/{clean_run}/{leaf}")
}

pub(crate) fn map_api_effort(effort: Effort) -> CarbonEffortTier {
    match effort {
        Effort::Fast => CarbonEffortTier::Fast,
        Effort::Medium => CarbonEffortTier::Medium,
        Effort::High => CarbonEffortTier::High,
        Effort::Ultra => CarbonEffortTier::Ultra,
    }
}

pub(crate) fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}
