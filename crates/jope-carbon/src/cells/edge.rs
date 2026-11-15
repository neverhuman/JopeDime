//! Regional `carbon-edge` HTTP / AWS Lambda Function URL request router (`ADR-0007`).
//!
//! Compiled under `--no-default-features --features edge`: enforces `HMAC-SHA256` capability
//! tokens (`RT-02`), strict `tenant_id` isolation (`RT-08`), sliding-window rate limits (`RT-11`),
//! zero-knowledge `CUEN\x01`/`CUDK\x01` headers (`RT-01`), conditional `run_id` creation (`RT-03`),
//! size/SHA-pinned S3 `PUT` presigning (`RT-04`), `/predict` quotas (`RT-10`), async `202` dispatch
//! (`RT-12`), and `DELETE /v1/carbon/runs/{id}` cryptographic shred.

use std::collections::BTreeMap;
use std::sync::Arc;

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonCreateRunRequest, CarbonEarlyStopRequest, CarbonExecutionModeWire,
        CarbonExplainRequest, CarbonExplainResponse, CarbonExportRequest, CarbonExportResponse,
        CarbonPingResponse, CarbonPredictRequest, CarbonPredictResponse,
        CarbonPreparePredictRequest, CarbonPreparePredictResponse, CarbonPrepareUploadRequest,
        CarbonPrepareUploadResponse, CarbonPresignedTarget, CarbonPublicWaveProgress,
        CarbonRegionCatalog, CarbonRegisterFreeRequest, CarbonResumeRunRequest, CarbonRunStateWire,
        CarbonShredRunResponse, CarbonTierWire, CarbonVersionResponse,
    },
    decimal::DecimalU64,
    Effort,
};
use jope_prep::{
    conditioning::FitLandmarkBundle,
    cuen::{parse_cudk_v1, parse_cuen_v1, CuenPayloadKind},
    objective::ObjectiveSpec,
};
use serde::{Deserialize, Serialize};

use crate::{
    cells::engine_invoker::{now_epoch_ms, CarbonEngineInvoker, CarbonEngineWireCommand},
    modes::{
        planner::{
            estimate_ms_per_candidate_1vcpu, plan_carbon_execution, CarbonExecutionMode,
            CarbonPlanInput,
        },
        spot_advisor::{evaluate_spot_hybrid_gates, MonthlyWindowMetrics, SpotAdvisorThresholds},
    },
    role::{CarbonEffortTier, CarbonProductTier},
    security::{
        auth::{
            validate_strict_run_id, validate_strict_tenant_id, CarbonAuthVerifier,
            CarbonRateLimiter,
        },
        sigv4::CarbonPresigner,
    },
    storage::cas::{
        canonical_blob_scope, canonical_run_state_key, sha256_lower_hex, CarbonCasStore,
    },
};

const MAX_LANDMARK_WIRE_BYTES: usize = 16_384;
const MAX_CONTROL_JSON_BYTES: usize = 65_536;
const MAX_PREDICT_CAS_BYTES_FREE: u64 = 64 * 1_048_576;
const MAX_PREDICT_CAS_BYTES_PRO: u64 = 256 * 1_048_576;

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

    pub fn header_ci(&self, lower_name: &str) -> Option<&str> {
        if let Some(v) = self.headers.get(lower_name) {
            return Some(v.as_str());
        }
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(lower_name))
            .map(|(_, v)| v.as_str())
    }

    pub fn trusted_caller_ip<'a>(&'a self, default_ip: &'a str) -> &'a str {
        if let Some(ip) = self
            .header_ci("x-carbon-trusted-source-ip")
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return ip;
        }
        self.header_ci("x-forwarded-for")
            .and_then(|v| v.split(',').next())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(default_ip)
    }
}

/// `CarbonGatewayResponse` is a struct representing an HTTP / Lambda Function URL v2.0 response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonGatewayResponse {
    pub status: u16,
    pub content_type: String,
    #[serde(default)]
    pub body_b64: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

impl CarbonGatewayResponse {
    pub fn json_ok<T: Serialize>(status: u16, value: &T) -> Self {
        match serde_json::to_vec(value) {
            Ok(bytes) => Self {
                status,
                content_type: "application/json".to_owned(),
                body_b64: base64::engine::general_purpose::STANDARD.encode(bytes),
                headers: BTreeMap::new(),
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
            headers: BTreeMap::new(),
        }
    }

    pub fn bytes_ok(status: u16, content_type: &str, bytes: &[u8]) -> Self {
        Self {
            status,
            content_type: content_type.to_owned(),
            body_b64: base64::engine::general_purpose::STANDARD.encode(bytes),
            headers: BTreeMap::new(),
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers
            .insert(name.into().to_ascii_lowercase(), value.into());
        self
    }

    pub fn error_json(status: u16, code: &str, message: &str) -> Self {
        Self::json_ok(
            status,
            &serde_json::json!({ "error": code, "message": message }),
        )
    }

    pub fn decoded_body(&self) -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(&self.body_b64)
            .unwrap_or_default()
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
    pub auth_verifier: Option<CarbonAuthVerifier>,
    pub engine_kek_pubkey_b64: Option<String>,
}

/// `CarbonEdgeRouter` is a struct handling all public `/v1/carbon/*` and discovery endpoints.
pub struct CarbonEdgeRouter {
    config: CarbonEdgeConfig,
    cas: Arc<dyn CarbonCasStore>,
    engine: Arc<dyn CarbonEngineInvoker>,
    s3_upload_target: Option<(String, String)>,
    rate_limiter: CarbonRateLimiter,
    release_id: Option<String>,
    r882_digest: Option<String>,
}

impl CarbonEdgeRouter {
    pub fn new(
        config: CarbonEdgeConfig,
        cas: Arc<dyn CarbonCasStore>,
        engine: Arc<dyn CarbonEngineInvoker>,
    ) -> Self {
        let release_id = std::env::var("CARBON_RELEASE_ID")
            .ok()
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty());
        let r882_digest = std::env::var("CARBON_R882_DIGEST")
            .ok()
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty());
        Self {
            config,
            cas,
            engine,
            s3_upload_target: None,
            rate_limiter: CarbonRateLimiter::new(60),
            release_id,
            r882_digest,
        }
    }

    pub fn with_s3_upload_target(mut self, target: Option<(String, String)>) -> Self {
        self.s3_upload_target = target;
        self
    }

    pub fn with_release_provenance(
        mut self,
        release_id: Option<String>,
        r882_digest: Option<String>,
    ) -> Self {
        self.release_id = release_id;
        self.r882_digest = r882_digest;
        self
    }

    fn attach_provenance_headers(&self, mut resp: CarbonGatewayResponse) -> CarbonGatewayResponse {
        resp = resp
            .with_header("x-carbon-version", crate::CARBON_PKG_VERSION)
            .with_header("x-carbon-release-tag", crate::CARBON_RELEASE_TAG)
            .with_header("x-carbon-git-sha", crate::CARBON_GIT_SHA)
            .with_header("x-carbon-build-target", crate::CARBON_BUILD_TARGET);
        if let Some(rel) = &self.release_id {
            resp = resp.with_header("x-carbon-release-id", rel);
        }
        if let Some(dig) = &self.r882_digest {
            resp = resp.with_header("x-carbon-r882-digest", dig);
        }
        resp
    }

    fn finalize_gateway_response(
        &self,
        req_id: &str,
        is_sensitive_route: bool,
        mut resp: CarbonGatewayResponse,
    ) -> CarbonGatewayResponse {
        resp = self
            .attach_provenance_headers(resp)
            .with_header("x-carbon-request-id", req_id)
            .with_header(
                "strict-transport-security",
                "max-age=63072000; includeSubDomains",
            )
            .with_header("x-content-type-options", "nosniff");
        if is_sensitive_route && !resp.headers.contains_key("cache-control") {
            resp = resp.with_header("cache-control", "no-store");
        }
        if resp.status == 429 && !resp.headers.contains_key("retry-after") {
            resp = resp.with_header("retry-after", "1");
        }
        resp
    }

    pub fn handle_lambda_invocation(&self, payload: &[u8]) -> Result<Vec<u8>, String> {
        let value: serde_json::Value = serde_json::from_slice(payload)
            .map_err(|e| format!("invalid edge Lambda JSON payload: {e}"))?;
        if value.get("rawPath").is_some() || value.get("requestContext").is_some() {
            let gw_req = parse_lambda_url_v2_event(&value)?;
            let gw_resp = self.handle(&gw_req);
            return encode_lambda_url_v2_response(&gw_resp);
        }
        let gw_req: CarbonGatewayRequest = serde_json::from_value(value)
            .map_err(|e| format!("decode CarbonGatewayRequest: {e}"))?;
        let gw_resp = self.handle(&gw_req);
        serde_json::to_vec(&gw_resp).map_err(|e| format!("encode CarbonGatewayResponse: {e}"))
    }

    pub fn handle(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let req_id = crate::observability::generate_or_sanitize_request_id(
            req.header_ci("x-carbon-request-id"),
        );
        let method = req.method.to_ascii_uppercase();
        let path = req.path.trim_end_matches('/');
        let clean_path = if path.is_empty() { "/" } else { path };
        let is_sensitive_route = clean_path.starts_with("/v1/carbon/auth")
            || clean_path.starts_with("/v1/carbon/uploads")
            || clean_path.starts_with("/v1/carbon/runs")
            || clean_path.starts_with("/v1/carbon/cas")
            || clean_path.starts_with("/v1/carbon/advisor");

        let resp = match (method.as_str(), clean_path) {
            ("GET" | "HEAD", p) if p == "/carbon" || p.starts_with("/carbon/") => {
                self.handle_passive_download_redirect(&method, clean_path, &req_id, req)
            }
            ("GET", "/.well-known/carbon-regions.json") => {
                let mut cat = self.config.catalog.clone();
                for entry in &mut cat.regions {
                    if entry.endpoint_url.trim().is_empty() && entry.region == self.config.region {
                        entry.endpoint_url = self.resolve_public_base_url(req);
                    }
                }
                CarbonGatewayResponse::json_ok(200, &cat)
            }
            ("GET", "/v1/carbon/ping") => CarbonGatewayResponse::json_ok(
                200,
                &CarbonPingResponse {
                    region: self.config.region.clone(),
                    healthy: true,
                    server_epoch_ms: DecimalU64(now_epoch_ms()),
                    enterprise_self_hosted: self.config.enterprise_self_hosted,
                    engine_kek_pubkey_b64: self.config.engine_kek_pubkey_b64.clone(),
                },
            ),
            ("GET", "/v1/carbon/version") => CarbonGatewayResponse::json_ok(
                200,
                &CarbonVersionResponse {
                    version: crate::CARBON_PKG_VERSION.to_owned(),
                    release_tag: crate::CARBON_RELEASE_TAG.to_owned(),
                    git_sha: crate::CARBON_GIT_SHA.to_owned(),
                    build_target: crate::CARBON_BUILD_TARGET.to_owned(),
                    region: self.config.region.clone(),
                    enterprise_self_hosted: self.config.enterprise_self_hosted,
                    release_id: self.release_id.clone(),
                    r882_bundle_digest: self.r882_digest.clone(),
                    engine_kek_pubkey_b64: self.config.engine_kek_pubkey_b64.clone(),
                    server_epoch_ms: DecimalU64(now_epoch_ms()),
                },
            ),
            ("POST", "/v1/carbon/auth/anonymous") => self.handle_anonymous_auth(req),
            ("POST", "/v1/carbon/auth/register") => self.handle_register_free(req),
            ("POST", "/v1/carbon/uploads/prepare") => self.handle_prepare_upload(req),
            ("POST", "/v1/carbon/runs") => self.handle_create_run(req),
            ("GET", "/v1/carbon/advisor/spot-hybrid") => self.handle_spot_advisor(req),
            _ if clean_path.starts_with("/v1/carbon/cas/") && method == "PUT" => {
                let sha = &clean_path["/v1/carbon/cas/".len()..];
                self.handle_put_cas_blob(sha, req)
            }
            _ if clean_path.starts_with("/v1/carbon/cas/") && method == "GET" => {
                let sha = &clean_path["/v1/carbon/cas/".len()..];
                self.handle_get_cas_blob(sha, req)
            }
            _ if clean_path.starts_with("/v1/carbon/runs/") => {
                let rest = &clean_path["/v1/carbon/runs/".len()..];
                if let Some(run_id) = rest.strip_suffix("/events") {
                    if method == "GET" {
                        self.handle_run_events(run_id, req)
                    } else {
                        CarbonGatewayResponse::error_json(
                            404,
                            "not_found",
                            "unknown Carbon run route",
                        )
                    }
                } else if let Some(run_id) = rest.strip_suffix("/stop") {
                    if method == "POST" {
                        self.handle_early_stop(run_id, req)
                    } else {
                        CarbonGatewayResponse::error_json(
                            404,
                            "not_found",
                            "unknown Carbon run route",
                        )
                    }
                } else if let Some(run_id) = rest.strip_suffix("/resume") {
                    if method == "POST" {
                        self.handle_resume(run_id, req)
                    } else {
                        CarbonGatewayResponse::error_json(
                            404,
                            "not_found",
                            "unknown Carbon run route",
                        )
                    }
                } else if let Some(run_id) = rest.strip_suffix("/predict/prepare") {
                    if method == "POST" {
                        self.handle_prepare_predict(run_id, req)
                    } else {
                        CarbonGatewayResponse::error_json(
                            404,
                            "not_found",
                            "unknown Carbon run route",
                        )
                    }
                } else if let Some(run_id) = rest.strip_suffix("/predict") {
                    if method == "POST" {
                        self.handle_predict(run_id, req)
                    } else {
                        CarbonGatewayResponse::error_json(
                            404,
                            "not_found",
                            "unknown Carbon run route",
                        )
                    }
                } else if let Some(run_id) = rest.strip_suffix("/explain") {
                    if method == "POST" {
                        self.handle_explain(run_id, req)
                    } else {
                        CarbonGatewayResponse::error_json(
                            404,
                            "not_found",
                            "unknown Carbon run route",
                        )
                    }
                } else if let Some(run_id) = rest.strip_suffix("/export") {
                    if method == "POST" {
                        self.handle_export(run_id, req)
                    } else {
                        CarbonGatewayResponse::error_json(
                            404,
                            "not_found",
                            "unknown Carbon run route",
                        )
                    }
                } else if !rest.contains('/') {
                    if method == "GET" {
                        self.handle_get_run(rest, req)
                    } else if method == "DELETE" {
                        self.handle_shred_run(rest, req)
                    } else {
                        CarbonGatewayResponse::error_json(
                            404,
                            "not_found",
                            "unknown Carbon run route",
                        )
                    }
                } else {
                    CarbonGatewayResponse::error_json(404, "not_found", "unknown Carbon run route")
                }
            }
            _ => CarbonGatewayResponse::error_json(404, "not_found", "unknown Carbon endpoint"),
        };
        self.finalize_gateway_response(&req_id, is_sensitive_route, resp)
    }

    fn requires_cloud_encryption(&self) -> bool {
        !self.config.enterprise_self_hosted && self.config.engine_kek_pubkey_b64.is_some()
    }

    fn capability_auth_header(req: &CarbonGatewayRequest) -> Option<&str> {
        req.header_ci("x-carbon-capability-token").or_else(|| {
            req.header_ci("authorization")
                .filter(|v| !v.trim_start().starts_with("AWS4-HMAC-SHA256"))
        })
    }

    fn authorize_tenant(
        &self,
        req: &CarbonGatewayRequest,
        tenant_id: &str,
        requested_tier: Option<CarbonTierWire>,
    ) -> Result<(CarbonTierWire, bool), CarbonGatewayResponse> {
        if let Err(err) = validate_strict_tenant_id(tenant_id) {
            return Err(CarbonGatewayResponse::error_json(
                400,
                "invalid_tenant",
                &err.to_string(),
            ));
        }
        let cap_hdr = Self::capability_auth_header(req);
        if self.config.enterprise_self_hosted {
            if let (Some(verifier), Some(hdr)) = (&self.config.auth_verifier, cap_hdr) {
                let claims = verifier
                    .verify_bearer_header(Some(hdr), now_epoch_ms() / 1_000)
                    .map_err(|e| {
                        CarbonGatewayResponse::error_json(401, "unauthorized", &e.to_string())
                    })?;
                if claims.tenant_id != tenant_id {
                    return Err(CarbonGatewayResponse::error_json(
                        403,
                        "tenant_mismatch",
                        "capability token does not grant access to requested tenant_id",
                    ));
                }
            }
            return Ok((CarbonTierWire::Enterprise, true));
        }

        if requested_tier == Some(CarbonTierWire::Enterprise) {
            return Err(CarbonGatewayResponse::error_json(
                403,
                "tier_forbidden",
                "Enterprise tier is only available on self-hosted VPC `carbon serve`",
            ));
        }

        if let Some(verifier) = &self.config.auth_verifier {
            let claims = verifier
                .verify_bearer_header(cap_hdr, now_epoch_ms() / 1_000)
                .map_err(|e| {
                    CarbonGatewayResponse::error_json(401, "unauthorized", &e.to_string())
                })?;
            if claims.tenant_id != tenant_id {
                return Err(CarbonGatewayResponse::error_json(
                    403,
                    "tenant_mismatch",
                    "capability token tenant_id does not match request tenant_id",
                ));
            }
            if let Some(req_t) = requested_tier {
                let allowed = matches!(
                    (claims.tier, req_t),
                    (CarbonTierWire::Enterprise, _)
                        | (
                            CarbonTierWire::Pro,
                            CarbonTierWire::Pro | CarbonTierWire::Free
                        )
                        | (CarbonTierWire::Free, CarbonTierWire::Free)
                );
                if !allowed {
                    return Err(CarbonGatewayResponse::error_json(
                        403,
                        "tier_escalation_refused",
                        "capability token does not permit requested product tier",
                    ));
                }
                return Ok((req_t, claims.admin));
            }
            return Ok((claims.tier, claims.admin));
        }

        Ok((requested_tier.unwrap_or(CarbonTierWire::Free), false))
    }

    fn handle_anonymous_auth(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let now_secs = now_epoch_ms() / 1_000;
        let caller = req.trusted_caller_ip("anon_global");
        if !self
            .rate_limiter
            .check_and_record(&format!("anon:{caller}"), now_secs, 300)
        {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "anonymous token issuance rate limit exceeded",
            );
        }
        let Some(verifier) = &self.config.auth_verifier else {
            return CarbonGatewayResponse::error_json(
                501,
                "auth_not_configured",
                "anonymous token minting is not enabled on this gateway",
            );
        };
        match verifier.mint_anonymous_free(now_secs, self.config.engine_kek_pubkey_b64.clone()) {
            Ok(resp) => CarbonGatewayResponse::json_ok(200, &resp),
            Err(err) => {
                CarbonGatewayResponse::error_json(500, "auth_mint_failed", &err.to_string())
            }
        }
    }

    fn handle_register_free(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let now_secs = now_epoch_ms() / 1_000;
        let caller = req.trusted_caller_ip("reg_global");
        if !self
            .rate_limiter
            .check_and_record(&format!("reg:{caller}"), now_secs, 120)
        {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "free account registration rate limit exceeded (max 120/min)",
            );
        }
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        if body.len() > 1_024 {
            return CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                "registration payload exceeds 1 KiB limit",
            );
        }
        let reg_req = if body.iter().all(u8::is_ascii_whitespace) {
            CarbonRegisterFreeRequest::default()
        } else {
            match serde_json::from_slice::<CarbonRegisterFreeRequest>(&body) {
                Ok(r) => r,
                Err(_) => {
                    return CarbonGatewayResponse::error_json(
                        400,
                        "invalid_json",
                        "invalid CarbonRegisterFreeRequest JSON",
                    );
                }
            }
        };
        let Some(verifier) = &self.config.auth_verifier else {
            return CarbonGatewayResponse::error_json(
                501,
                "auth_not_configured",
                "free capability registration is not enabled on this gateway",
            );
        };
        match verifier.mint_registered_free(
            reg_req.label.as_deref(),
            now_secs,
            self.config.engine_kek_pubkey_b64.clone(),
        ) {
            Ok(resp) => CarbonGatewayResponse::json_ok(201, &resp),
            Err(err) => CarbonGatewayResponse::error_json(400, "invalid_label", &err.to_string()),
        }
    }

    fn handle_prepare_upload(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        if body.len() > MAX_CONTROL_JSON_BYTES {
            return CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                "upload prepare payload exceeds 64 KiB limit",
            );
        }
        let Ok(prep) = serde_json::from_slice::<CarbonPrepareUploadRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonPrepareUploadRequest",
            );
        };
        let (eff_tier_wire, _) = match self.authorize_tenant(req, &prep.tenant_id, Some(prep.tier))
        {
            Ok(v) => v,
            Err(resp) => return resp,
        };
        let now_secs = now_epoch_ms() / 1_000;
        if !self
            .rate_limiter
            .check_and_record(&format!("prep:{}", prep.tenant_id), now_secs, 300)
        {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "upload prepare rate limit exceeded",
            );
        }
        if self.requires_cloud_encryption() && !prep.encrypted_cuen {
            return CarbonGatewayResponse::error_json(
                400,
                "encryption_required",
                "cloud Carbon tiers require per-user CUEN v1 encrypted uploads (encrypted_cuen=true)",
            );
        }
        let tier = wire_to_product_tier(eff_tier_wire);
        let total_rows = (prep.fit_rows as usize).saturating_add(prep.holdout_rows as usize);
        if let Err(err) = tier.validate_admission(
            total_rows,
            prep.feature_count as usize,
            CarbonEffortTier::Fast,
        ) {
            return CarbonGatewayResponse::error_json(403, "tier_limit", &err.to_string());
        }
        if prep.fit_jmat.bytes.0 == 0
            || prep.holdout_jmat.bytes.0 == 0
            || !is_valid_sha256_hex(&prep.fit_jmat.sha256)
            || !is_valid_sha256_hex(&prep.holdout_jmat.sha256)
        {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_upload_slot",
                "fit_jmat and holdout_jmat require non-zero byte counts and 64-char hex SHA-256",
            );
        }
        if let Some(max_slot) = tier.max_upload_slot_bytes() {
            if prep.fit_jmat.bytes.0 > max_slot || prep.holdout_jmat.bytes.0 > max_slot {
                return CarbonGatewayResponse::error_json(
                    413,
                    "upload_slot_too_large",
                    &format!(
                        "{} tier upload slot exceeds maximum {} bytes",
                        tier.as_str(),
                        max_slot
                    ),
                );
            }
        }

        let upload_id = format!("upl_{}", ulid::Ulid::new());
        let Ok(blob_scope) = canonical_blob_scope(&prep.tenant_id, &upload_id) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_scope",
                "invalid tenant/upload scope",
            );
        };
        let fit_sha_lower = prep.fit_jmat.sha256.to_ascii_lowercase();
        let holdout_sha_lower = prep.holdout_jmat.sha256.to_ascii_lowercase();

        let expires = now_secs.saturating_add(900);
        let (fit_url, holdout_url) = if let Some((bucket, prefix)) = &self.s3_upload_target {
            let Ok(creds) = crate::security::sigv4::CarbonAwsCredentials::from_env() else {
                return CarbonGatewayResponse::error_json(
                    500,
                    "presign_creds_missing",
                    "missing AWS execution credentials for S3 presigning",
                );
            };
            let region = match std::env::var("AWS_REGION") {
                Ok(val) if !val.trim().is_empty() => val,
                _ => self.config.region.clone(),
            };
            let (_, amz_date) = crate::security::sigv4::format_amz_timestamp_utc(now_secs);
            let fit_key = join_s3_scoped_prefix(prefix, Some(&blob_scope), &fit_sha_lower);
            let holdout_key = join_s3_scoped_prefix(prefix, Some(&blob_scope), &holdout_sha_lower);
            let Ok(f_url) = CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
                &creds,
                &region,
                bucket,
                &fit_key,
                &amz_date,
                900,
                &fit_sha_lower,
                prep.fit_jmat.bytes.0,
            ) else {
                return CarbonGatewayResponse::error_json(
                    500,
                    "presign_failed",
                    "failed to sign size-pinned fit S3 PUT URL",
                );
            };
            let Ok(h_url) = CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
                &creds,
                &region,
                bucket,
                &holdout_key,
                &amz_date,
                900,
                &holdout_sha_lower,
                prep.holdout_jmat.bytes.0,
            ) else {
                return CarbonGatewayResponse::error_json(
                    500,
                    "presign_failed",
                    "failed to sign size-pinned holdout S3 PUT URL",
                );
            };
            (f_url, h_url)
        } else {
            let base_url = self.resolve_public_base_url(req);
            let f_url = self.config.presigner.sign_gateway_scoped_upload_url(
                &base_url,
                Some(&blob_scope),
                &fit_sha_lower,
                prep.fit_jmat.bytes.0,
                expires,
            );
            let h_url = self.config.presigner.sign_gateway_scoped_upload_url(
                &base_url,
                Some(&blob_scope),
                &holdout_sha_lower,
                prep.holdout_jmat.bytes.0,
                expires,
            );
            (f_url, h_url)
        };
        // Persist tenant-local upload_id pointers so legacy v1.0.0 CLIs omitting `upload_id` on
        // `CarbonCreateRunRequest` still resolve their scoped upload cleanly (`GAP-S3-3`).
        for sha_lc in [&fit_sha_lower, &holdout_sha_lower] {
            if let Ok(ptr_key) =
                canonical_run_state_key(&prep.tenant_id, "_uploads", &format!("{sha_lc}.uid"))
            {
                let _ = self.cas.put_state_object(&ptr_key, upload_id.as_bytes());
            }
        }
        CarbonGatewayResponse::json_ok(
            200,
            &CarbonPrepareUploadResponse {
                upload_id,
                fit_target: CarbonPresignedTarget {
                    url: fit_url,
                    method: "PUT".to_owned(),
                    sha256: fit_sha_lower,
                    max_bytes: prep.fit_jmat.bytes,
                    expires_epoch_secs: DecimalU64(expires),
                },
                holdout_target: CarbonPresignedTarget {
                    url: holdout_url,
                    method: "PUT".to_owned(),
                    sha256: holdout_sha_lower,
                    max_bytes: prep.holdout_jmat.bytes,
                    expires_epoch_secs: DecimalU64(expires),
                },
                engine_kek_pubkey_b64: self.config.engine_kek_pubkey_b64.clone(),
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
        let scope = req
            .query
            .get("scope")
            .map(String::as_str)
            .filter(|s| !s.is_empty());
        let (Ok(declared_bytes), Ok(exp_secs)) = (bytes_str.parse::<u64>(), exp_str.parse::<u64>())
        else {
            return CarbonGatewayResponse::error_json(400, "bad_query", "invalid bytes or exp");
        };
        let now_secs = now_epoch_ms() / 1_000;
        if let Err(reason) = self.config.presigner.verify_gateway_scoped_upload(
            scope,
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
        if let Err(err) = self.cas.put_scoped_blob_verified(scope, sha, &payload) {
            return CarbonGatewayResponse::error_json(400, "cas_integrity", &err);
        }
        CarbonGatewayResponse::json_ok(200, &serde_json::json!({ "stored_sha256": sha }))
    }

    fn handle_get_cas_blob(&self, sha: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Some(exp_str) = req.query.get("exp") else {
            return CarbonGatewayResponse::error_json(400, "missing_param", "missing exp query");
        };
        let Some(sig) = req.query.get("sig") else {
            return CarbonGatewayResponse::error_json(401, "missing_sig", "missing sig query");
        };
        let scope = req
            .query
            .get("scope")
            .map(String::as_str)
            .filter(|s| !s.is_empty());
        let Ok(exp_secs) = exp_str.parse::<u64>() else {
            return CarbonGatewayResponse::error_json(400, "bad_query", "invalid exp");
        };
        let now_secs = now_epoch_ms() / 1_000;
        if let Err(reason) = self
            .config
            .presigner
            .verify_gateway_scoped_download(scope, sha, exp_secs, sig, now_secs)
        {
            return CarbonGatewayResponse::error_json(403, "presign_denied", reason);
        }
        match self.cas.get_scoped_blob_verified(scope, sha) {
            Ok(bytes) => CarbonGatewayResponse::bytes_ok(200, "application/octet-stream", &bytes),
            Err(err) => CarbonGatewayResponse::error_json(404, "blob_not_found", &err),
        }
    }

    fn handle_create_run(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        if body.len() > MAX_CONTROL_JSON_BYTES {
            return CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                "create run request exceeds 64 KiB limit",
            );
        }
        let Ok(mut create) = serde_json::from_slice::<CarbonCreateRunRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonCreateRunRequest",
            );
        };
        let (eff_tier_wire, _) =
            match self.authorize_tenant(req, &create.tenant_id, Some(create.tier)) {
                Ok(v) => v,
                Err(resp) => return resp,
            };
        create.tier = eff_tier_wire;
        let run_str = create.run_id.to_string();
        if let Err(err) = validate_strict_run_id(&run_str) {
            return CarbonGatewayResponse::error_json(400, "invalid_run_id", &err.to_string());
        }
        if let Some(ref uid) = create.upload_id {
            if let Err(err) = canonical_blob_scope(&create.tenant_id, uid) {
                return CarbonGatewayResponse::error_json(400, "invalid_upload_id", &err);
            }
        } else if is_valid_sha256_hex(&create.fit_jmat_sha256) {
            let fit_lc = create.fit_jmat_sha256.to_ascii_lowercase();
            if let Ok(ptr_key) =
                canonical_run_state_key(&create.tenant_id, "_uploads", &format!("{fit_lc}.uid"))
            {
                if let Ok(Some(uid_bytes)) = self.cas.get_state_object(&ptr_key) {
                    if let Ok(uid_str) = String::from_utf8(uid_bytes) {
                        let uid_trimmed = uid_str.trim();
                        if canonical_blob_scope(&create.tenant_id, uid_trimmed).is_ok() {
                            create.upload_id = Some(uid_trimmed.to_owned());
                        }
                    }
                }
            }
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self
            .rate_limiter
            .check_and_record(&format!("runs:{}", create.tenant_id), now_secs, 120)
        {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "run creation rate limit exceeded",
            );
        }
        if self.is_run_shredded(&create.tenant_id, &run_str) {
            return CarbonGatewayResponse::error_json(
                409,
                "run_shredded",
                "run_id was previously shredded and cannot be reused",
            );
        }
        if self.requires_cloud_encryption()
            && (!create.encrypted_cuen || create.wrapped_u_dek_b64.is_none())
        {
            return CarbonGatewayResponse::error_json(
                400,
                "encryption_required",
                "cloud Carbon runs require encrypted_cuen=true and wrapped_u_dek_b64",
            );
        }

        if let Err(err) = ObjectiveSpec::parse(&create.objective) {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_objective",
                &format!("invalid fitness objective: {err}"),
            );
        }
        if let Some(ref mode) = create.split_mode {
            if !matches!(mode.as_str(), "random" | "forward") {
                return CarbonGatewayResponse::error_json(
                    400,
                    "invalid_split_mode",
                    "split_mode must be 'random' or 'forward'",
                );
            }
        }
        if let Some(ref cal) = create.calibration_mode {
            if !matches!(cal.as_str(), "auto" | "none" | "shift" | "affine") {
                return CarbonGatewayResponse::error_json(
                    400,
                    "invalid_calibration_mode",
                    "calibration_mode must be 'auto', 'none', 'shift', or 'affine'",
                );
            }
        }
        if create.aux_column_names.len() > 16 {
            return CarbonGatewayResponse::error_json(
                400,
                "too_many_aux_columns",
                "at most 16 auxiliary fitness columns are permitted",
            );
        }
        for (idx, aux_col) in create.aux_column_names.iter().enumerate() {
            if aux_col.trim().is_empty()
                || aux_col == &create.target_name
                || create.feature_names.contains(aux_col)
                || create.aux_column_names[..idx].contains(aux_col)
            {
                return CarbonGatewayResponse::error_json(
                    400,
                    "invalid_aux_column",
                    &format!(
                        "auxiliary fitness column '{aux_col}' must be non-empty, unique, and disjoint from feature_names and target_name"
                    ),
                );
            }
        }

        let tier = wire_to_product_tier(create.tier);
        let effort = map_api_effort(create.effort);
        let total_rows = (create.fit_rows as usize).saturating_add(create.holdout_rows as usize);
        if let Err(err) = tier.validate_admission(total_rows, create.feature_names.len(), effort) {
            return CarbonGatewayResponse::error_json(403, "tier_limit", &err.to_string());
        }
        if !is_valid_sha256_hex(&create.fit_jmat_sha256)
            || !is_valid_sha256_hex(&create.holdout_jmat_sha256)
        {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_sha256",
                "fit_jmat_sha256 and holdout_jmat_sha256 must be 64-char hex digests",
            );
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
        if lm_raw.len() > MAX_LANDMARK_WIRE_BYTES {
            return CarbonGatewayResponse::error_json(
                400,
                "landmarks_too_large",
                "landmark_bundle_b64 exceeds 16 KiB bound",
            );
        }
        if create.encrypted_cuen {
            if parse_cuen_v1(&lm_raw, CuenPayloadKind::LandmarkJlmb).is_err() {
                return CarbonGatewayResponse::error_json(
                    400,
                    "invalid_cuen_landmarks",
                    "landmark_bundle_b64 is not a valid CUEN v1 LandmarkJlmb envelope",
                );
            }
            let Some(cudk_b64) = &create.wrapped_u_dek_b64 else {
                return CarbonGatewayResponse::error_json(
                    400,
                    "missing_wrapped_u_dek",
                    "wrapped_u_dek_b64 is required when encrypted_cuen=true",
                );
            };
            let Ok(cudk_raw) = base64::engine::general_purpose::STANDARD.decode(cudk_b64) else {
                return CarbonGatewayResponse::error_json(
                    400,
                    "invalid_cudk_b64",
                    "wrapped_u_dek_b64 is not valid base64",
                );
            };
            if parse_cudk_v1(&cudk_raw).is_err() {
                return CarbonGatewayResponse::error_json(
                    400,
                    "invalid_cudk_envelope",
                    "wrapped_u_dek_b64 must be a 133-byte CUDK v1 envelope",
                );
            }
        } else {
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
        }

        let fit_rows = create.fit_rows as usize;
        let feature_count = create.feature_names.len();
        let plan_decision = match plan_carbon_execution(&CarbonPlanInput {
            product_tier: tier,
            effort,
            total_rows,
            fit_rows,
            features: feature_count,
            est_ms_per_candidate_1vcpu: estimate_ms_per_candidate_1vcpu(fit_rows, feature_count),
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

        let Ok(req_key) = canonical_run_state_key(&create.tenant_id, &run_str, "request.json")
        else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid run state key");
        };
        // Strip transient `wrapped_u_dek_b64` before persisting `request.json` so S3 CAS NEVER
        // stores even the wrapped U_DEK envelope at rest (`zero-knowledge shred hygiene`)!
        let mut persisted_req = create.clone();
        persisted_req.wrapped_u_dek_b64 = None;
        let Ok(stored_req_bytes) = serde_json::to_vec(&persisted_req) else {
            return CarbonGatewayResponse::error_json(
                500,
                "serialize_failed",
                "failed to encode create request",
            );
        };
        match self
            .cas
            .put_state_object_if_absent(&req_key, &stored_req_bytes)
        {
            Ok(true) => {}
            Ok(false) => {
                // Retry-idempotent create: if the exact same client retries a byte-identical
                // `POST /v1/carbon/runs` (e.g. after a transient gateway timeout), return the
                // existing progress as `202` without double-invoking `engine`. Any modified field
                // (`seed`, `dataset_id`, `landmark_bundle_b64` nonce, etc.) fails closed with `409`.
                let same_payload = matches!(
                    self.cas.get_state_object(&req_key),
                    Ok(Some(existing)) if existing == stored_req_bytes
                );
                if !same_payload {
                    return CarbonGatewayResponse::error_json(
                        409,
                        "duplicate_run_id",
                        "run_id already exists for this tenant; conditional create refused overwrite",
                    );
                }
                if let Ok(status_key) =
                    canonical_run_state_key(&create.tenant_id, &run_str, "status.json")
                {
                    if let Ok(Some(cur_bytes)) = self.cas.get_state_object(&status_key) {
                        if let Ok(cur_progress) =
                            serde_json::from_slice::<CarbonPublicWaveProgress>(&cur_bytes)
                        {
                            return CarbonGatewayResponse::json_ok(202, &cur_progress);
                        }
                    }
                }
            }
            Err(err) => {
                return CarbonGatewayResponse::error_json(500, "cas_write_failed", &err);
            }
        }

        let initial_progress = CarbonPublicWaveProgress {
            run_id: create.run_id.clone(),
            state: CarbonRunStateWire::Queued,
            execution_mode: match plan_decision.mode {
                CarbonExecutionMode::SoloSingleLambda => CarbonExecutionModeWire::SoloSingleLambda,
                CarbonExecutionMode::HybridFanOut => CarbonExecutionModeWire::HybridFanOut,
            },
            eval_memory_mb: plan_decision.eval_class.memory_mb() as u32,
            eval_shards: if plan_decision.mode == CarbonExecutionMode::HybridFanOut {
                plan_decision.shards_per_wave
            } else {
                0
            },
            eval_invocations: DecimalU64(0),
            eval_total_wall_ms: DecimalU64(0),
            wave_index: 0,
            waves_target: effort.planned_waves(),
            candidates_evaluated: DecimalU64(0),
            best_cv_loss: None,
            best_cv_metric: None,
            outer_holdout_loss: None,
            outer_holdout_metric: None,
            neural_seed_applied: false,
            neural_guidance_mode: None,
            bundle_digest: None,
            tier_applied: Some(create.tier),
            encrypted_cuen: create.encrypted_cuen,
            can_resume: false,
            sealed_checkpoint_sha256: None,
            sealed_champion_sha256: None,
            elapsed_ms: DecimalU64(0),
            region: self.config.region.clone(),
            failure_message: None,
            eval_fallback_shards: None,
        };
        let Ok(status_key) = canonical_run_state_key(&create.tenant_id, &run_str, "status.json")
        else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid status key");
        };
        let Ok(status_bytes) = serde_json::to_vec(&initial_progress) else {
            return CarbonGatewayResponse::error_json(
                500,
                "serialize_failed",
                "failed to encode initial run progress",
            );
        };
        if let Err(err) = self.cas.put_state_object(&status_key, &status_bytes) {
            return CarbonGatewayResponse::error_json(500, "cas_write_failed", &err);
        }

        let cell_volume = total_rows.saturating_mul(create.feature_names.len());
        let dispatch_async = effort >= CarbonEffortTier::Medium
            || plan_decision.mode == CarbonExecutionMode::HybridFanOut
            || cell_volume >= 2_500
            || total_rows >= 400;
        let cmd = CarbonEngineWireCommand::StartRun {
            request: create,
            region: self.config.region.clone(),
        };
        if dispatch_async {
            match self.engine.invoke_engine_async(&cmd) {
                Ok(()) => CarbonGatewayResponse::json_ok(202, &initial_progress),
                Err(err) => CarbonGatewayResponse::error_json(502, "engine_async_failed", &err),
            }
        } else {
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
    }

    fn handle_get_run(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Some(tenant) = req.query.get("tenant_id").filter(|t| !t.trim().is_empty()) else {
            return CarbonGatewayResponse::error_json(
                400,
                "missing_tenant",
                "tenant_id query parameter is required",
            );
        };
        if let Err(resp) = self.authorize_tenant(req, tenant, None) {
            return resp;
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self
            .rate_limiter
            .check_and_record(&format!("poll:{tenant}"), now_secs, 600)
        {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "run status poll rate limit exceeded",
            );
        }
        if validate_strict_run_id(run_id).is_err() {
            return CarbonGatewayResponse::error_json(400, "invalid_run_id", "invalid run_id");
        }
        if self.is_run_shredded(tenant, run_id) {
            return CarbonGatewayResponse::error_json(
                404,
                "run_not_found",
                "run has been shredded",
            );
        }
        let Ok(key) = canonical_run_state_key(tenant, run_id, "status.json") else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key");
        };
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
        if let Err(resp) = self.authorize_tenant(req, tenant, None) {
            return resp;
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self
            .rate_limiter
            .check_and_record(&format!("poll:{tenant}"), now_secs, 600)
        {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "run event poll rate limit exceeded",
            );
        }
        if validate_strict_run_id(run_id).is_err() || self.is_run_shredded(tenant, run_id) {
            return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
        }
        let Ok(key) = canonical_run_state_key(tenant, run_id, "status.json") else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key");
        };
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
        if body.len() > MAX_CONTROL_JSON_BYTES {
            return CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                "early stop payload exceeds 64 KiB limit",
            );
        }
        let Ok(stop_req) = serde_json::from_slice::<CarbonEarlyStopRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonEarlyStopRequest",
            );
        };
        if let Err(resp) = self.authorize_tenant(req, &stop_req.tenant_id, None) {
            return resp;
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self.rate_limiter.check_and_record(
            &format!("stop:{}", stop_req.tenant_id),
            now_secs,
            120,
        ) {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "early stop rate limit exceeded",
            );
        }
        let Ok(stop_key) = canonical_run_state_key(&stop_req.tenant_id, run_id, "stop.json") else {
            return CarbonGatewayResponse::error_json(400, "invalid_run_id", "invalid run_id");
        };
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
        if body.len() > MAX_CONTROL_JSON_BYTES {
            return CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                "resume payload exceeds 64 KiB limit",
            );
        }
        let Ok(resume_req) = serde_json::from_slice::<CarbonResumeRunRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonResumeRunRequest",
            );
        };
        if let Err(resp) = self.authorize_tenant(req, &resume_req.tenant_id, None) {
            return resp;
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self.rate_limiter.check_and_record(
            &format!("resume:{}", resume_req.tenant_id),
            now_secs,
            120,
        ) {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "resume rate limit exceeded",
            );
        }
        if validate_strict_run_id(run_id).is_err()
            || self.is_run_shredded(&resume_req.tenant_id, run_id)
        {
            return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
        }
        let Ok(req_key) = canonical_run_state_key(&resume_req.tenant_id, run_id, "request.json")
        else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key");
        };
        let stored_create = match self.cas.get_state_object(&req_key) {
            Ok(Some(bytes)) => match serde_json::from_slice::<CarbonCreateRunRequest>(&bytes) {
                Ok(c) => c,
                Err(_) => {
                    return CarbonGatewayResponse::error_json(
                        500,
                        "corrupt_request",
                        "corrupt stored request",
                    );
                }
            },
            Ok(None) => {
                return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
            }
            Err(err) => return CarbonGatewayResponse::error_json(500, "cas_read_failed", &err),
        };
        if let Some(new_eff) = resume_req.effort {
            let tier = wire_to_product_tier(stored_create.tier);
            let total_rows = (stored_create.fit_rows as usize)
                .saturating_add(stored_create.holdout_rows as usize);
            if let Err(err) = tier.validate_admission(
                total_rows,
                stored_create.feature_names.len(),
                map_api_effort(new_eff),
            ) {
                return CarbonGatewayResponse::error_json(403, "tier_limit", &err.to_string());
            }
        }
        if stored_create.encrypted_cuen && resume_req.wrapped_u_dek_b64.is_none() {
            return CarbonGatewayResponse::error_json(
                400,
                "missing_wrapped_u_dek",
                "encrypted runs require wrapped_u_dek_b64 to resume",
            );
        }

        let extra_waves = resume_req.additional_waves.unwrap_or(2);
        let stored_volume =
            (stored_create.fit_rows as usize).saturating_mul(stored_create.feature_names.len());
        let async_resume = extra_waves > 2
            || (extra_waves >= 2
                && (map_api_effort(resume_req.effort.unwrap_or(stored_create.effort))
                    >= CarbonEffortTier::Medium
                    || stored_volume >= 2_500
                    || stored_create.fit_rows >= 400));
        let cmd = CarbonEngineWireCommand::ResumeRun {
            run_id: run_id.to_owned(),
            resume: resume_req.clone(),
            region: self.config.region.clone(),
        };
        if async_resume {
            if let Ok(status_key) =
                canonical_run_state_key(&resume_req.tenant_id, run_id, "status.json")
            {
                if let Ok(Some(cur)) = self.cas.get_state_object(&status_key) {
                    if let Ok(mut prog) = serde_json::from_slice::<CarbonPublicWaveProgress>(&cur) {
                        prog.state = CarbonRunStateWire::Queued;
                        prog.can_resume = false;
                        if let Ok(updated) = serde_json::to_vec(&prog) {
                            let _ = self.cas.put_state_object(&status_key, &updated);
                        }
                        return match self.engine.invoke_engine_async(&cmd) {
                            Ok(()) => CarbonGatewayResponse::json_ok(202, &prog),
                            Err(err) => {
                                CarbonGatewayResponse::error_json(502, "engine_resume_failed", &err)
                            }
                        };
                    }
                }
            }
        }

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

    fn handle_prepare_predict(
        &self,
        run_id: &str,
        req: &CarbonGatewayRequest,
    ) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        if body.len() > MAX_CONTROL_JSON_BYTES {
            return CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                "predict prepare payload exceeds 64 KiB limit",
            );
        }
        let Ok(prep) = serde_json::from_slice::<CarbonPreparePredictRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonPreparePredictRequest",
            );
        };
        if let Err(resp) = self.authorize_tenant(req, &prep.tenant_id, None) {
            return resp;
        }
        if validate_strict_run_id(run_id).is_err() || self.is_run_shredded(&prep.tenant_id, run_id)
        {
            return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self.rate_limiter.check_and_record(
            &format!("pred_prep:{}", prep.tenant_id),
            now_secs,
            300,
        ) {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "predict prepare rate limit exceeded",
            );
        }
        if prep.predict_cuen.bytes.0 == 0 || !is_valid_sha256_hex(&prep.predict_cuen.sha256) {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_upload_slot",
                "predict_cuen requires non-zero byte count and 64-char hex SHA-256",
            );
        }
        let Ok(req_key) = canonical_run_state_key(&prep.tenant_id, run_id, "request.json") else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key");
        };
        let stored_create = match self.cas.get_state_object(&req_key) {
            Ok(Some(bytes)) => match serde_json::from_slice::<CarbonCreateRunRequest>(&bytes) {
                Ok(c) => c,
                Err(_) => {
                    return CarbonGatewayResponse::error_json(
                        500,
                        "corrupt_request",
                        "corrupt stored request",
                    );
                }
            },
            Ok(None) => {
                return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
            }
            Err(err) => return CarbonGatewayResponse::error_json(500, "cas_read_failed", &err),
        };
        let max_bytes = match stored_create.tier {
            CarbonTierWire::Free => MAX_PREDICT_CAS_BYTES_FREE,
            CarbonTierWire::Pro | CarbonTierWire::Enterprise => MAX_PREDICT_CAS_BYTES_PRO,
        };
        if prep.predict_cuen.bytes.0 > max_bytes {
            return CarbonGatewayResponse::error_json(
                413,
                "predict_payload_too_large",
                &format!("predict CAS upload exceeds maximum {max_bytes} bytes"),
            );
        }

        let predict_upload_id = format!("pupl_{}", ulid::Ulid::new());
        let Ok(blob_scope) = canonical_blob_scope(&prep.tenant_id, &predict_upload_id) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_scope",
                "invalid tenant/predict upload scope",
            );
        };
        let sha_lower = prep.predict_cuen.sha256.to_ascii_lowercase();
        let expires = now_secs.saturating_add(900);
        let put_url = if let Some((bucket, prefix)) = &self.s3_upload_target {
            let Ok(creds) = crate::security::sigv4::CarbonAwsCredentials::from_env() else {
                return CarbonGatewayResponse::error_json(
                    500,
                    "presign_creds_missing",
                    "missing AWS execution credentials for S3 presigning",
                );
            };
            let region = match std::env::var("AWS_REGION") {
                Ok(val) if !val.trim().is_empty() => val,
                _ => self.config.region.clone(),
            };
            let (_, amz_date) = crate::security::sigv4::format_amz_timestamp_utc(now_secs);
            let s3_key = join_s3_scoped_prefix(prefix, Some(&blob_scope), &sha_lower);
            let Ok(url) = CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
                &creds,
                &region,
                bucket,
                &s3_key,
                &amz_date,
                900,
                &sha_lower,
                prep.predict_cuen.bytes.0,
            ) else {
                return CarbonGatewayResponse::error_json(
                    500,
                    "presign_failed",
                    "failed to sign size-pinned predict S3 PUT URL",
                );
            };
            url
        } else {
            let base_url = self.resolve_public_base_url(req);
            self.config.presigner.sign_gateway_scoped_upload_url(
                &base_url,
                Some(&blob_scope),
                &sha_lower,
                prep.predict_cuen.bytes.0,
                expires,
            )
        };
        CarbonGatewayResponse::json_ok(
            200,
            &CarbonPreparePredictResponse {
                upload_id: predict_upload_id,
                input_target: CarbonPresignedTarget {
                    url: put_url,
                    method: "PUT".to_owned(),
                    sha256: sha_lower,
                    max_bytes: prep.predict_cuen.bytes,
                    expires_epoch_secs: DecimalU64(expires),
                },
            },
        )
    }

    fn handle_predict(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        if body.len() > 32 * 1_048_576 {
            return CarbonGatewayResponse::error_json(
                413,
                "predict_payload_too_large",
                "predict request exceeds 32 MiB wire limit",
            );
        }
        let Ok(pred_req) = serde_json::from_slice::<CarbonPredictRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonPredictRequest",
            );
        };
        if let Err(resp) = self.authorize_tenant(req, &pred_req.tenant_id, None) {
            return resp;
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self.rate_limiter.check_and_record(
            &format!("pred:{}", pred_req.tenant_id),
            now_secs,
            300,
        ) {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "predict rate limit exceeded",
            );
        }
        if validate_strict_run_id(run_id).is_err()
            || self.is_run_shredded(&pred_req.tenant_id, run_id)
        {
            return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
        }
        let Ok(req_key) = canonical_run_state_key(&pred_req.tenant_id, run_id, "request.json")
        else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key");
        };
        let stored_create = match self.cas.get_state_object(&req_key) {
            Ok(Some(bytes)) => match serde_json::from_slice::<CarbonCreateRunRequest>(&bytes) {
                Ok(c) => c,
                Err(_) => {
                    return CarbonGatewayResponse::error_json(
                        500,
                        "corrupt_request",
                        "corrupt stored request",
                    );
                }
            },
            Ok(None) => {
                return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
            }
            Err(err) => return CarbonGatewayResponse::error_json(500, "cas_read_failed", &err),
        };

        let tier = wire_to_product_tier(stored_create.tier);
        if stored_create.encrypted_cuen {
            let has_inline = pred_req.encrypted_rows_cuen_b64.is_some();
            let has_cas = pred_req.encrypted_rows_cas_sha256.is_some();
            if (!has_inline && !has_cas)
                || pred_req.client_response_pubkey_b64.is_none()
                || pred_req.wrapped_u_dek_b64.is_none()
                || !pred_req.rows.is_empty()
            {
                return CarbonGatewayResponse::error_json(
                    400,
                    "encrypted_predict_required",
                    "encrypted runs require encrypted_rows_cuen_b64 (or encrypted_rows_cas_sha256), client_response_pubkey_b64, and wrapped_u_dek_b64 with empty plaintext rows",
                );
            }
            if let Some(ref cas_sha) = pred_req.encrypted_rows_cas_sha256 {
                if !is_valid_sha256_hex(cas_sha) {
                    return CarbonGatewayResponse::error_json(
                        400,
                        "invalid_sha256",
                        "encrypted_rows_cas_sha256 must be a 64-char hex digest",
                    );
                }
            }
            if let Some(ref puid) = pred_req.predict_upload_id {
                if let Err(err) = canonical_blob_scope(&pred_req.tenant_id, puid) {
                    return CarbonGatewayResponse::error_json(
                        400,
                        "invalid_predict_upload_id",
                        &err,
                    );
                }
            }
        } else {
            if let Some(max_r) = tier.max_predict_rows_per_call() {
                if pred_req.rows.len() > max_r {
                    return CarbonGatewayResponse::error_json(
                        413,
                        "predict_row_limit",
                        &format!(
                            "{} tier admits at most {max_r} rows per /predict call",
                            tier.as_str()
                        ),
                    );
                }
            }
            if let Some(max_f) = tier.max_features() {
                if pred_req.feature_names.len() > max_f {
                    return CarbonGatewayResponse::error_json(
                        413,
                        "predict_feature_limit",
                        &format!(
                            "{} tier admits at most {max_f} features per /predict call",
                            tier.as_str()
                        ),
                    );
                }
            }
        }

        let quota_state = if let Some(max_calls) = tier.max_predict_calls_per_run() {
            let Ok(cnt_key) =
                canonical_run_state_key(&pred_req.tenant_id, run_id, "predict_count.json")
            else {
                return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key");
            };
            let current_calls: u32 = self
                .cas
                .get_state_object(&cnt_key)
                .ok()
                .flatten()
                .and_then(|b| serde_json::from_slice::<u32>(&b).ok())
                .unwrap_or(0);
            if current_calls >= max_calls {
                return CarbonGatewayResponse::error_json(
                    403,
                    "predict_quota_exceeded",
                    &format!(
                        "{} tier admits at most {max_calls} /predict calls per run",
                        tier.as_str()
                    ),
                );
            }
            Some((cnt_key, current_calls))
        } else {
            None
        };

        let pred_tenant_id = pred_req.tenant_id.clone();
        let pred_upload_id = pred_req.predict_upload_id.clone();
        let cmd = CarbonEngineWireCommand::Predict {
            run_id: run_id.to_owned(),
            predict: pred_req,
        };
        match self.engine.invoke_engine(&cmd) {
            Ok(bytes) => match serde_json::from_slice::<CarbonPredictResponse>(&bytes) {
                Ok(mut resp) => {
                    if let Some((cnt_key, current_calls)) = quota_state {
                        if let Ok(next_bytes) = serde_json::to_vec(&(current_calls + 1)) {
                            let _ = self.cas.put_state_object(&cnt_key, &next_bytes);
                        }
                    }
                    if let Some(ref out_sha) = resp.encrypted_predictions_cas_sha256 {
                        let out_scope = match pred_upload_id.as_deref() {
                            Some(puid) => canonical_blob_scope(&pred_tenant_id, puid).ok(),
                            None => canonical_blob_scope(&pred_tenant_id, run_id).ok(),
                        };
                        let dl_now = now_epoch_ms() / 1_000;
                        let dl_url = if let Some((bucket, prefix)) = &self.s3_upload_target {
                            let Ok(creds) =
                                crate::security::sigv4::CarbonAwsCredentials::from_env()
                            else {
                                return CarbonGatewayResponse::error_json(
                                    500,
                                    "presign_creds_missing",
                                    "missing AWS execution credentials for S3 GET presigning",
                                );
                            };
                            let region = match std::env::var("AWS_REGION") {
                                Ok(val) if !val.trim().is_empty() => val,
                                _ => self.config.region.clone(),
                            };
                            let (_, amz_date) =
                                crate::security::sigv4::format_amz_timestamp_utc(dl_now);
                            let s3_key = join_s3_scoped_prefix(
                                prefix,
                                out_scope.as_deref(),
                                &out_sha.to_ascii_lowercase(),
                            );
                            let Ok(url) = CarbonPresigner::presign_s3_get_v4_with_creds(
                                &creds, &region, bucket, &s3_key, &amz_date, 900,
                            ) else {
                                return CarbonGatewayResponse::error_json(
                                    500,
                                    "presign_failed",
                                    "failed to sign predict output S3 GET URL",
                                );
                            };
                            url
                        } else {
                            let base_url = self.resolve_public_base_url(req);
                            self.config.presigner.sign_gateway_scoped_download_url(
                                &base_url,
                                out_scope.as_deref(),
                                &out_sha.to_ascii_lowercase(),
                                dl_now.saturating_add(900),
                            )
                        };
                        resp.encrypted_predictions_cas_url = Some(dl_url);
                    }
                    CarbonGatewayResponse::json_ok(200, &resp)
                }
                Err(_) => CarbonGatewayResponse::error_json(
                    500,
                    "invalid_engine_response",
                    "malformed predict response",
                ),
            },
            Err(err) => CarbonGatewayResponse::error_json(502, "engine_predict_failed", &err),
        }
    }

    fn handle_shred_run(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Some(tenant) = req.query.get("tenant_id").filter(|t| !t.trim().is_empty()) else {
            return CarbonGatewayResponse::error_json(
                400,
                "missing_tenant",
                "tenant_id query parameter is required",
            );
        };
        if let Err(resp) = self.authorize_tenant(req, tenant, None) {
            return resp;
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self
            .rate_limiter
            .check_and_record(&format!("shred:{tenant}"), now_secs, 120)
        {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "shred rate limit exceeded",
            );
        }
        let Ok(parsed_run_id) = run_id.parse() else {
            return CarbonGatewayResponse::error_json(400, "invalid_run_id", "invalid RunId");
        };
        let mut deleted = 0u32;
        if let Ok(req_key) = canonical_run_state_key(tenant, run_id, "request.json") {
            if let Ok(Some(req_bytes)) = self.cas.get_state_object(&req_key) {
                if let Ok(stored) = serde_json::from_slice::<CarbonCreateRunRequest>(&req_bytes) {
                    let blob_scope = stored
                        .upload_id
                        .as_deref()
                        .and_then(|uid| canonical_blob_scope(tenant, uid).ok());
                    if self
                        .cas
                        .delete_scoped_blob(blob_scope.as_deref(), &stored.fit_jmat_sha256)
                        .unwrap_or(false)
                    {
                        deleted += 1;
                    }
                    if self
                        .cas
                        .delete_scoped_blob(blob_scope.as_deref(), &stored.holdout_jmat_sha256)
                        .unwrap_or(false)
                    {
                        deleted += 1;
                    }
                    if let Some(ref uid) = stored.upload_id {
                        for sha in [&stored.fit_jmat_sha256, &stored.holdout_jmat_sha256] {
                            let sha_lc = sha.to_ascii_lowercase();
                            if let Ok(ptr_key) = canonical_run_state_key(
                                tenant,
                                "_uploads",
                                &format!("{sha_lc}.uid"),
                            ) {
                                if let Ok(Some(cur)) = self.cas.get_state_object(&ptr_key) {
                                    if cur.as_slice() == uid.as_bytes() {
                                        let _ = self.cas.delete_state_object(&ptr_key);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for leaf in [
            "request.json",
            "status.json",
            "stop.json",
            "predict_count.json",
            "sealed_checkpoint.enc",
            "sealed_champion.enc",
        ] {
            if let Ok(k) = canonical_run_state_key(tenant, run_id, leaf) {
                if self.cas.get_state_object(&k).ok().flatten().is_some()
                    && self.cas.delete_state_object(&k).is_ok()
                {
                    deleted += 1;
                }
            }
        }
        if let Ok(tomb_key) = canonical_run_state_key(tenant, run_id, "shredded.json") {
            let _ = self
                .cas
                .put_state_object(&tomb_key, br#"{"shredded":true}"#);
        }
        CarbonGatewayResponse::json_ok(
            200,
            &CarbonShredRunResponse {
                run_id: parsed_run_id,
                shredded: true,
                objects_deleted: deleted,
            },
        )
    }

    fn handle_spot_advisor(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let now_secs = now_epoch_ms() / 1_000;
        if !self.config.enterprise_self_hosted {
            if let Some(verifier) = &self.config.auth_verifier {
                let claims = match verifier
                    .verify_bearer_header(Self::capability_auth_header(req), now_secs)
                {
                    Ok(c) => c,
                    Err(e) => {
                        return CarbonGatewayResponse::error_json(
                            401,
                            "unauthorized",
                            &e.to_string(),
                        );
                    }
                };
                if claims.tier == CarbonTierWire::Free && !claims.admin {
                    return CarbonGatewayResponse::error_json(
                        403,
                        "pro_or_admin_required",
                        "spot-hybrid advisor requires Pro or Admin capability token",
                    );
                }
                if !self.rate_limiter.check_and_record(
                    &format!("advisor:{}", claims.tenant_id),
                    now_secs,
                    60,
                ) {
                    return CarbonGatewayResponse::error_json(
                        429,
                        "rate_limited",
                        "spot-hybrid advisor rate limit exceeded",
                    );
                }
            }
        }
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

    fn handle_explain(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        if body.len() > MAX_CONTROL_JSON_BYTES {
            return CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                "explain payload exceeds 64 KiB limit",
            );
        }
        let Ok(explain_req) = serde_json::from_slice::<CarbonExplainRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonExplainRequest",
            );
        };
        if let Err(resp) = self.authorize_tenant(req, &explain_req.tenant_id, None) {
            return resp;
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self.rate_limiter.check_and_record(
            &format!("explain:{}", explain_req.tenant_id),
            now_secs,
            120,
        ) {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "explain rate limit exceeded",
            );
        }
        if validate_strict_run_id(run_id).is_err()
            || self.is_run_shredded(&explain_req.tenant_id, run_id)
        {
            return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
        }
        let Ok(req_key) = canonical_run_state_key(&explain_req.tenant_id, run_id, "request.json")
        else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key");
        };
        let stored_create = match self.cas.get_state_object(&req_key) {
            Ok(Some(bytes)) => match serde_json::from_slice::<CarbonCreateRunRequest>(&bytes) {
                Ok(c) => c,
                Err(_) => {
                    return CarbonGatewayResponse::error_json(
                        500,
                        "corrupt_request",
                        "corrupt stored request",
                    );
                }
            },
            Ok(None) => {
                return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
            }
            Err(err) => return CarbonGatewayResponse::error_json(500, "cas_read_failed", &err),
        };
        if stored_create.encrypted_cuen
            && (explain_req.client_response_pubkey_b64.is_none()
                || explain_req.wrapped_u_dek_b64.is_none())
        {
            return CarbonGatewayResponse::error_json(
                400,
                "encrypted_explain_required",
                "encrypted runs require client_response_pubkey_b64 and wrapped_u_dek_b64 on /explain",
            );
        }

        let cmd = CarbonEngineWireCommand::Explain {
            run_id: run_id.to_owned(),
            explain: explain_req,
        };
        match self.engine.invoke_engine(&cmd) {
            Ok(bytes) => match serde_json::from_slice::<CarbonExplainResponse>(&bytes) {
                Ok(resp) => CarbonGatewayResponse::json_ok(200, &resp),
                Err(_) => CarbonGatewayResponse::error_json(
                    500,
                    "invalid_engine_response",
                    "malformed explain response",
                ),
            },
            Err(err) => CarbonGatewayResponse::error_json(502, "engine_explain_failed", &err),
        }
    }

    fn handle_export(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let Ok(body) = req.body_bytes() else {
            return CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body");
        };
        if body.len() > MAX_CONTROL_JSON_BYTES {
            return CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                "export payload exceeds 64 KiB limit",
            );
        }
        let Ok(export_req) = serde_json::from_slice::<CarbonExportRequest>(&body) else {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_json",
                "invalid CarbonExportRequest",
            );
        };
        if let Err(resp) = self.authorize_tenant(req, &export_req.tenant_id, None) {
            return resp;
        }
        let now_secs = now_epoch_ms() / 1_000;
        if !self.rate_limiter.check_and_record(
            &format!("export:{}", export_req.tenant_id),
            now_secs,
            120,
        ) {
            return CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                "export rate limit exceeded",
            );
        }
        if validate_strict_run_id(run_id).is_err()
            || self.is_run_shredded(&export_req.tenant_id, run_id)
        {
            return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
        }
        let Ok(req_key) = canonical_run_state_key(&export_req.tenant_id, run_id, "request.json")
        else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key");
        };
        let stored_create = match self.cas.get_state_object(&req_key) {
            Ok(Some(bytes)) => match serde_json::from_slice::<CarbonCreateRunRequest>(&bytes) {
                Ok(c) => c,
                Err(_) => {
                    return CarbonGatewayResponse::error_json(
                        500,
                        "corrupt_request",
                        "corrupt stored request",
                    );
                }
            },
            Ok(None) => {
                return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
            }
            Err(err) => return CarbonGatewayResponse::error_json(500, "cas_read_failed", &err),
        };
        if stored_create.encrypted_cuen
            && (export_req.client_response_pubkey_b64.is_none()
                || export_req.wrapped_u_dek_b64.is_none())
        {
            return CarbonGatewayResponse::error_json(
                400,
                "encrypted_export_required",
                "encrypted runs require client_response_pubkey_b64 and wrapped_u_dek_b64 on /export",
            );
        }

        let cmd = CarbonEngineWireCommand::Export {
            run_id: run_id.to_owned(),
            export: export_req,
        };
        match self.engine.invoke_engine(&cmd) {
            Ok(bytes) => match serde_json::from_slice::<CarbonExportResponse>(&bytes) {
                Ok(resp) => CarbonGatewayResponse::json_ok(200, &resp),
                Err(_) => CarbonGatewayResponse::error_json(
                    500,
                    "invalid_engine_response",
                    "malformed export response",
                ),
            },
            Err(err) => CarbonGatewayResponse::error_json(502, "engine_export_failed", &err),
        }
    }

    fn handle_passive_download_redirect(
        &self,
        method: &str,
        clean_path: &str,
        req_id: &str,
        req: &CarbonGatewayRequest,
    ) -> CarbonGatewayResponse {
        let raw_path = req.header_ci("x-carbon-raw-path").unwrap_or(&req.path);
        let raw_lc = raw_path.to_ascii_lowercase();
        if raw_path.contains("..")
            || raw_lc.contains("%2e")
            || raw_lc.contains("%2f")
            || raw_lc.contains("%5c")
            || raw_path.contains('\\')
            || raw_path.bytes().any(|b| b <= 0x20 || b == 0x7f)
        {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_download_target",
                "download path contains disallowed traversal or encoded characters",
            )
            .with_header("cache-control", "private, no-store");
        }

        let default_pub_tag = std::env::var("CARBON_PUBLISHED_INSTALLER_TAG")
            .ok()
            .map(|v| v.trim().to_owned())
            .filter(|v| is_valid_release_tag(v) && v != "latest")
            .unwrap_or_else(|| "carbon-cli-v1.1.0".to_owned());

        let (tag, asset, target_url) = match clean_path {
            "/carbon/install.sh" => (
                default_pub_tag.clone(),
                "install.sh".to_owned(),
                format!(
                    "https://github.com/neverhuman/JopeDime/releases/download/{default_pub_tag}/install.sh"
                ),
            ),
            "/carbon/install.ps1" => (
                default_pub_tag.clone(),
                "install.ps1".to_owned(),
                format!(
                    "https://github.com/neverhuman/JopeDime/releases/download/{default_pub_tag}/install.ps1"
                ),
            ),
            "/carbon/carbon-clouds.json" => (
                "latest".to_owned(),
                "carbon-clouds.json".to_owned(),
                "https://github.com/neverhuman/JopeDime/releases/latest/download/carbon-clouds.json"
                    .to_owned(),
            ),
            p if p.starts_with("/carbon/releases/") => {
                let rem = &p["/carbon/releases/".len()..];
                let Some((tag_part, asset_part)) = rem.split_once('/') else {
                    return CarbonGatewayResponse::error_json(
                        400,
                        "invalid_download_target",
                        "expected /carbon/releases/{tag}/{asset}",
                    )
                    .with_header("cache-control", "private, no-store");
                };
                if asset_part.contains('/')
                    || !is_valid_release_tag(tag_part)
                    || !is_valid_release_asset(asset_part)
                {
                    return CarbonGatewayResponse::error_json(
                        400,
                        "invalid_download_target",
                        "requested release tag or asset is not on the Carbon download allowlist",
                    )
                    .with_header("cache-control", "private, no-store");
                }
                let url = if tag_part == "latest" {
                    format!(
                        "https://github.com/neverhuman/JopeDime/releases/latest/download/{asset_part}"
                    )
                } else {
                    format!(
                        "https://github.com/neverhuman/JopeDime/releases/download/{tag_part}/{asset_part}"
                    )
                };
                (tag_part.to_owned(), asset_part.to_owned(), url)
            }
            _ => {
                return CarbonGatewayResponse::error_json(
                    400,
                    "invalid_download_target",
                    "unknown Carbon download route",
                )
                .with_header("cache-control", "private, no-store");
            }
        };

        let now_ms = now_epoch_ms();
        let now_secs = now_ms / 1_000;
        let source_ip = req.trusted_caller_ip("unknown");
        let ip_day_hmac = self
            .config
            .presigner
            .daily_ip_pseudonym_hex(now_secs, source_ip);
        let mut client_hdr_names: Vec<&str> = req
            .headers
            .keys()
            .map(String::as_str)
            .filter(|k| {
                !k.eq_ignore_ascii_case("x-carbon-trusted-source-ip")
                    && !k.eq_ignore_ascii_case("x-carbon-raw-path")
            })
            .collect();
        client_hdr_names.sort_unstable();
        let header_names_hash =
            sha256_lower_hex(client_hdr_names.join(",").as_bytes())[..16].to_owned();
        let ua = sanitize_download_log_field(req.header_ci("user-agent"), 256);
        let accept_lang = sanitize_download_log_field(req.header_ci("accept-language"), 128);
        let sec_ch_ua = sanitize_download_log_field(req.header_ci("sec-ch-ua"), 128);
        let sec_ch_platform = sanitize_download_log_field(req.header_ci("sec-ch-ua-platform"), 64);
        let sec_ch_arch = sanitize_download_log_field(req.header_ci("sec-ch-ua-arch"), 64);
        let viewer_country = sanitize_download_log_field(
            req.header_ci("cloudfront-viewer-country")
                .or_else(|| req.header_ci("cf-ipcountry")),
            16,
        );
        let viewer_asn = sanitize_download_log_field(req.header_ci("cloudfront-viewer-asn"), 32);
        let tls_version = sanitize_download_log_field(
            req.header_ci("x-amzn-tls-version")
                .or_else(|| req.header_ci("cloudfront-viewer-tls")),
            64,
        );
        let dnt = matches!(req.header_ci("dnt"), Some("1"))
            || matches!(req.header_ci("sec-gpc"), Some("1"));
        let (inf_os, inf_arch, inf_libc, inf_archive) =
            infer_download_platform(&asset, ua.as_deref().unwrap_or(""));

        let telemetry_event = serde_json::json!({
            "schema": "carbon.download.v1",
            "ts_ms": now_ms,
            "request_id": req_id,
            "region": self.config.region,
            "method": method,
            "path": clean_path,
            "tag": tag,
            "asset": asset,
            "target_url": target_url,
            "source_ip": source_ip,
            "ip_day_hmac": ip_day_hmac,
            "header_names_hash": header_names_hash,
            "user_agent": ua,
            "accept_language": accept_lang,
            "sec_ch_ua": sec_ch_ua,
            "sec_ch_ua_platform": sec_ch_platform,
            "sec_ch_ua_arch": sec_ch_arch,
            "viewer_country": viewer_country,
            "viewer_asn": viewer_asn,
            "tls_version": tls_version,
            "dnt": dnt,
            "inferred_os": inf_os,
            "inferred_arch": inf_arch,
            "inferred_libc": inf_libc,
            "inferred_archive": inf_archive,
        });
        if let Ok(line) = serde_json::to_string(&telemetry_event) {
            eprintln!("{line}");
        }

        CarbonGatewayResponse::bytes_ok(302, "text/plain; charset=utf-8", b"")
            .with_header("location", target_url)
            .with_header("cache-control", "private, no-store")
    }

    fn is_run_shredded(&self, tenant_id: &str, run_id: &str) -> bool {
        canonical_run_state_key(tenant_id, run_id, "shredded.json")
            .ok()
            .and_then(|k| self.cas.get_state_object(&k).ok().flatten())
            .is_some()
    }

    fn resolve_public_base_url(&self, req: &CarbonGatewayRequest) -> String {
        if !self.config.public_base_url.trim().is_empty() {
            return self.config.public_base_url.clone();
        }
        if let Some(host) = req.header_ci("host") {
            let host_lc = host.to_ascii_lowercase();
            let default_proto =
                if host_lc.ends_with(".amazonaws.com") || host_lc.ends_with(".on.aws") {
                    "https"
                } else {
                    "http"
                };
            let proto = req.header_ci("x-forwarded-proto").unwrap_or(default_proto);
            return format!("{proto}://{host}");
        }
        "http://127.0.0.1:8080".to_owned()
    }
}

pub(crate) fn wire_to_product_tier(wire: CarbonTierWire) -> CarbonProductTier {
    match wire {
        CarbonTierWire::Free => CarbonProductTier::Free,
        CarbonTierWire::Pro => CarbonProductTier::Pro,
        CarbonTierWire::Enterprise => CarbonProductTier::no_limit(),
    }
}

pub(crate) fn map_api_effort(effort: Effort) -> CarbonEffortTier {
    match effort {
        Effort::Fast => CarbonEffortTier::Fast,
        Effort::Medium => CarbonEffortTier::Medium,
        Effort::High => CarbonEffortTier::High,
        Effort::Ultra => CarbonEffortTier::Ultra,
    }
}

fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn is_valid_release_tag(tag: &str) -> bool {
    if tag == "latest" {
        return true;
    }
    if tag.is_empty() || tag.len() > 64 || tag.contains("..") {
        return false;
    }
    let semver_part = if let Some(rest) = tag.strip_prefix("carbon-cli-v") {
        rest
    } else if let Some(rest) = tag.strip_prefix('v') {
        rest
    } else {
        return false;
    };
    let (core, prerelease) = match semver_part.split_once('-') {
        Some((c, pre)) => (c, Some(pre)),
        None => (semver_part, None),
    };
    let mut nums = core.split('.');
    let (Some(maj), Some(min), Some(pat), None) =
        (nums.next(), nums.next(), nums.next(), nums.next())
    else {
        return false;
    };
    let digits_ok = [maj, min, pat].iter().all(|part| {
        !part.is_empty() && part.len() <= 8 && part.bytes().all(|b| b.is_ascii_digit())
    });
    if !digits_ok {
        return false;
    }
    if let Some(pre) = prerelease {
        if pre.is_empty()
            || !pre
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        {
            return false;
        }
    }
    true
}

fn is_valid_release_asset(asset: &str) -> bool {
    if asset.is_empty()
        || asset.len() > 128
        || asset.contains("..")
        || !asset
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return false;
    }
    if matches!(
        asset,
        "SHA256SUMS.txt"
            | "SHA256SUMS.txt.sig"
            | "carbon-clouds.json"
            | "install.sh"
            | "install.ps1"
            | "carbon-linux-x86_64-musl"
            | "carbon-linux-aarch64-musl"
            | "carbon-linux-armv7-musleabihf"
            | "carbon-darwin-arm64"
            | "carbon-darwin-x86_64"
            | "carbon-darwin-universal"
    ) {
        return true;
    }
    let has_os_prefix = asset.starts_with("carbon-linux-")
        || asset.starts_with("carbon-darwin-")
        || asset.starts_with("carbon-windows-");
    let has_archive_suffix = asset.ends_with(".tar.zst")
        || asset.ends_with(".tar.xz")
        || asset.ends_with(".tar.gz")
        || asset.ends_with(".zip")
        || asset.ends_with(".exe");
    has_os_prefix && has_archive_suffix
}

fn sanitize_download_log_field(raw: Option<&str>, max_chars: usize) -> Option<String> {
    let text = raw?.trim();
    if text.is_empty() {
        return None;
    }
    let redacted = crate::observability::redact_secrets(text);
    let cleaned: String = redacted
        .chars()
        .filter(|c| !c.is_control())
        .take(max_chars)
        .collect();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn infer_download_platform(
    asset: &str,
    user_agent: &str,
) -> (&'static str, &'static str, &'static str, &'static str) {
    let archive = if asset.ends_with(".tar.zst") {
        "tar.zst"
    } else if asset.ends_with(".tar.gz") {
        "tar.gz"
    } else if asset.ends_with(".zip") {
        "zip"
    } else if asset.ends_with(".ps1") {
        "ps1"
    } else if asset.ends_with(".sh") {
        "sh"
    } else {
        "meta"
    };

    let ua_lc = user_agent.to_ascii_lowercase();
    let os = if asset.starts_with("carbon-linux-") || ua_lc.contains("linux") {
        "linux"
    } else if asset.starts_with("carbon-darwin-")
        || ua_lc.contains("darwin")
        || ua_lc.contains("macintosh")
        || ua_lc.contains("mac os")
    {
        "macos"
    } else if asset.starts_with("carbon-windows-")
        || asset.ends_with(".ps1")
        || ua_lc.contains("windows")
        || ua_lc.contains("powershell")
    {
        "windows"
    } else {
        "unknown"
    };

    let arch = if asset.contains("arm64")
        || asset.contains("aarch64")
        || ua_lc.contains("aarch64")
        || ua_lc.contains("arm64")
    {
        "arm64"
    } else if asset.contains("armv7") || ua_lc.contains("armv7") {
        "armv7"
    } else if asset.contains("x86_64")
        || asset.contains("amd64")
        || ua_lc.contains("x86_64")
        || ua_lc.contains("win64")
        || ua_lc.contains("x64")
    {
        "x86_64"
    } else if asset.contains("universal") {
        "universal"
    } else {
        "unknown"
    };

    let libc = if asset.contains("musleabihf") {
        "musleabihf"
    } else if asset.contains("musl") {
        "musl"
    } else if asset.contains("msvc") || os == "windows" {
        "msvc"
    } else if os == "macos" {
        "libsystem"
    } else if os == "linux" {
        "gnu_or_musl"
    } else {
        "unknown"
    };

    (os, arch, libc, archive)
}

fn join_s3_scoped_prefix(prefix: &str, scope: Option<&str>, sha_lower: &str) -> String {
    let rel = match scope {
        Some(sc) if !sc.is_empty() => format!("blobs/{sc}/{sha_lower}"),
        _ => format!("blobs/{sha_lower}"),
    };
    if prefix.is_empty() {
        rel
    } else {
        format!("{prefix}/{rel}")
    }
}

fn parse_lambda_url_v2_event(event: &serde_json::Value) -> Result<CarbonGatewayRequest, String> {
    let method = event
        .pointer("/requestContext/http/method")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("GET")
        .to_owned();
    let path = event
        .get("rawPath")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            event
                .pointer("/requestContext/http/path")
                .and_then(serde_json::Value::as_str)
        })
        .unwrap_or("/")
        .to_owned();

    let mut query = BTreeMap::new();
    if let Some(raw_q) = event
        .get("rawQueryString")
        .and_then(serde_json::Value::as_str)
    {
        if !raw_q.is_empty() {
            for pair in url::form_urlencoded::parse(raw_q.as_bytes()) {
                query.insert(pair.0.into_owned(), pair.1.into_owned());
            }
        }
    }

    let mut headers = BTreeMap::new();
    if let Some(hdr_map) = event.get("headers").and_then(serde_json::Value::as_object) {
        for (k, v) in hdr_map {
            // Never trust client-supplied `x-carbon-trusted-source-ip` or `x-carbon-raw-path` (`GAP-SEC-4`).
            if k.eq_ignore_ascii_case("x-carbon-trusted-source-ip")
                || k.eq_ignore_ascii_case("x-carbon-raw-path")
            {
                continue;
            }
            if let Some(text) = v.as_str() {
                headers.insert(k.to_ascii_lowercase(), text.to_owned());
            }
        }
    }
    headers.insert("x-carbon-raw-path".to_owned(), path.clone());
    if let Some(src_ip) = event
        .pointer("/requestContext/http/sourceIp")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.trim().is_empty())
    {
        let trimmed_ip = src_ip.trim().to_owned();
        headers.insert("x-carbon-trusted-source-ip".to_owned(), trimmed_ip.clone());
        if !headers.contains_key("x-forwarded-for") {
            headers.insert("x-forwarded-for".to_owned(), trimmed_ip);
        }
    }

    let is_b64 = event
        .get("isBase64Encoded")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let raw_body = match event.get("body").and_then(serde_json::Value::as_str) {
        Some(body_str) if is_b64 => base64::engine::general_purpose::STANDARD
            .decode(body_str)
            .map_err(|e| format!("invalid base64 Function URL v2 body: {e}"))?,
        Some(body_str) => body_str.as_bytes().to_vec(),
        None => Vec::new(),
    };

    Ok(CarbonGatewayRequest {
        method,
        path,
        query,
        headers,
        body_b64: None,
        raw_body,
    })
}

fn encode_lambda_url_v2_response(resp: &CarbonGatewayResponse) -> Result<Vec<u8>, String> {
    let (body_field, is_b64) = if resp
        .content_type
        .eq_ignore_ascii_case("application/octet-stream")
    {
        (resp.body_b64.clone(), true)
    } else {
        let raw_bytes = resp.decoded_body();
        match String::from_utf8(raw_bytes) {
            Ok(utf8_str) => (utf8_str, false),
            Err(_) => (resp.body_b64.clone(), true),
        }
    };
    let mut hdr_map = serde_json::Map::new();
    for (k, v) in &resp.headers {
        hdr_map.insert(k.clone(), serde_json::Value::String(v.clone()));
    }
    hdr_map.insert(
        "content-type".to_owned(),
        serde_json::Value::String(resp.content_type.clone()),
    );
    let out = serde_json::json!({
        "statusCode": resp.status,
        "headers": hdr_map,
        "body": body_field,
        "isBase64Encoded": is_b64,
    });
    serde_json::to_vec(&out).map_err(|e| format!("encode Function URL v2 response: {e}"))
}
