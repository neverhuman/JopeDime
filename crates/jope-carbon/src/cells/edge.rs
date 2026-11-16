//! Regional `carbon-edge` HTTP / AWS Lambda Function URL request router (`ADR-0007`).
//!
//! Compiled under `--no-default-features --features edge`: enforces `HMAC-SHA256` capability
//! tokens (`RT-02`), strict `tenant_id` isolation (`RT-08`), sliding-window rate limits (`RT-11`),
//! zero-knowledge `CUEN\x01`/`CUDK\x01` headers (`RT-01`), conditional `run_id` creation (`RT-03`),
//! size/SHA-pinned S3 `PUT` presigning (`RT-04`), atomic `/predict` quotas (`RT-10`), async `202`
//! dispatch (`RT-12` + API Gateway v2 30s ceiling guard), and `DELETE /v1/carbon/runs/{id}` shred.

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
use serde::{de::DeserializeOwned, ser::SerializeStruct, Deserialize, Serialize, Serializer};

use crate::{
    cells::engine_invoker::{now_epoch_ms, CarbonEngineInvoker, CarbonEngineWireCommand},
    error::{err_to_string, to_json_string, to_json_vec},
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
    storage::cas::{canonical_blob_scope, canonical_run_state_key, CarbonCasStore},
};

const MAX_LANDMARK_WIRE_BYTES: usize = 16_384;
const MAX_CONTROL_JSON_BYTES: usize = 65_536;
const MAX_PREDICT_CAS_BYTES_FREE: u64 = 64 * 1_048_576;
const MAX_PREDICT_CAS_BYTES_PRO: u64 = 256 * 1_048_576;

/// `CarbonGatewayRequest` is a struct representing an HTTP or Lambda Function URL v2.0 request.
#[derive(Clone, Debug, Deserialize)]
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

impl Serialize for CarbonGatewayRequest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Encode non-empty `raw_body` as compact Base64 across JSON Runtime API wires (`1.33x`)
        // instead of a JSON integer array (`~4.0x`), while keeping zero-copy `raw_body` in-process!
        let mut st = serializer.serialize_struct("CarbonGatewayRequest", 5)?;
        st.serialize_field("method", &self.method)?;
        st.serialize_field("path", &self.path)?;
        st.serialize_field("query", &self.query)?;
        st.serialize_field("headers", &self.headers)?;
        if let Some(b64) = &self.body_b64 {
            st.serialize_field("body_b64", &Some(b64))?;
        } else if !self.raw_body.is_empty() {
            let encoded = base64::engine::general_purpose::STANDARD.encode(&self.raw_body);
            st.serialize_field("body_b64", &Some(encoded))?;
        } else {
            st.serialize_field("body_b64", &None::<String>)?;
        }
        st.end()
    }
}

impl CarbonGatewayRequest {
    pub fn body_cow(&self) -> Result<std::borrow::Cow<'_, [u8]>, String> {
        if !self.raw_body.is_empty() || self.body_b64.is_none() {
            return Ok(std::borrow::Cow::Borrowed(&self.raw_body));
        }
        if let Some(b64) = &self.body_b64 {
            if b64.len() > 48 * 1_048_576 {
                return Err("base64 request body exceeds 48 MiB wire limit".to_owned());
            }
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(err_to_string)?;
            return Ok(std::borrow::Cow::Owned(decoded));
        }
        Ok(std::borrow::Cow::Borrowed(&[]))
    }

    pub fn body_bytes(&self) -> Result<Vec<u8>, String> {
        self.body_cow().map(std::borrow::Cow::into_owned)
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
        self.header_ci("x-carbon-trusted-source-ip")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(default_ip)
    }
}

/// `CarbonGatewayResponse` is a struct representing an HTTP / Lambda Function URL v2.0 response.
/// Holds unencoded `raw_body` in-process (`MEM-02`) so `serve` and `emu` never Base64-inflate
/// multi-hundred-megabyte CAS responses, encoding `body_b64` lazily only when serialized to JSON.
#[derive(Clone, Debug, Deserialize)]
pub struct CarbonGatewayResponse {
    pub status: u16,
    pub content_type: String,
    #[serde(default)]
    pub body_b64: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(skip)]
    raw_body: Option<Vec<u8>>,
}

impl Serialize for CarbonGatewayResponse {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let count = if self.headers.is_empty() { 3 } else { 4 };
        let mut st = serializer.serialize_struct("CarbonGatewayResponse", count)?;
        st.serialize_field("status", &self.status)?;
        st.serialize_field("content_type", &self.content_type)?;
        if let Some(raw) = &self.raw_body {
            let encoded = base64::engine::general_purpose::STANDARD.encode(raw);
            st.serialize_field("body_b64", &encoded)?;
        } else {
            st.serialize_field("body_b64", &self.body_b64)?;
        }
        if !self.headers.is_empty() {
            st.serialize_field("headers", &self.headers)?;
        }
        st.end()
    }
}

impl CarbonGatewayResponse {
    pub fn json_ok<T: Serialize>(status: u16, value: &T) -> Self {
        match serde_json::to_vec(value) {
            Ok(bytes) => Self::raw_bytes_ok(status, "application/json", bytes),
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
        Self::raw_bytes_ok(status, content_type, text.as_bytes().to_vec())
    }

    pub fn bytes_ok(status: u16, content_type: &str, bytes: &[u8]) -> Self {
        Self::raw_bytes_ok(status, content_type, bytes.to_vec())
    }

    pub fn raw_bytes_ok(status: u16, content_type: &str, bytes: Vec<u8>) -> Self {
        Self {
            status,
            content_type: content_type.to_owned(),
            body_b64: String::new(),
            headers: BTreeMap::new(),
            raw_body: Some(bytes),
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
        if let Some(raw) = &self.raw_body {
            return raw.clone();
        }
        base64::engine::general_purpose::STANDARD
            .decode(&self.body_b64)
            .unwrap_or_default()
    }

    /// Consume the response and return its raw body bytes with zero extra copy or Base64 pass.
    pub fn into_http_body_bytes(self) -> Vec<u8> {
        if let Some(raw) = self.raw_body {
            return raw;
        }
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
        let value: serde_json::Value = serde_json::from_slice(payload).map_err(err_to_string)?;
        if value.get("rawPath").is_some() || value.get("requestContext").is_some() {
            let gw_req = parse_lambda_url_v2_event(&value)?;
            let gw_resp = self.handle(&gw_req);
            return encode_lambda_url_v2_response(&gw_resp);
        }
        let gw_req: CarbonGatewayRequest = serde_json::from_value(value).map_err(err_to_string)?;
        let gw_resp = self.handle(&gw_req);
        to_json_vec(&gw_resp)
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
                self.route_run_subpath(&method, &clean_path["/v1/carbon/runs/".len()..], req)
            }
            _ => CarbonGatewayResponse::error_json(404, "not_found", "unknown Carbon endpoint"),
        };
        self.finalize_gateway_response(&req_id, is_sensitive_route, resp)
    }

    fn route_run_subpath(
        &self,
        method: &str,
        rest: &str,
        req: &CarbonGatewayRequest,
    ) -> CarbonGatewayResponse {
        if let Some(run_id) = rest.strip_suffix("/events") {
            return if method == "GET" {
                self.handle_run_events(run_id, req)
            } else {
                unknown_run_route()
            };
        }
        if let Some(run_id) = rest.strip_suffix("/stop") {
            return if method == "POST" {
                self.handle_early_stop(run_id, req)
            } else {
                unknown_run_route()
            };
        }
        if let Some(run_id) = rest.strip_suffix("/resume") {
            return if method == "POST" {
                self.handle_resume(run_id, req)
            } else {
                unknown_run_route()
            };
        }
        if let Some(run_id) = rest.strip_suffix("/predict/prepare") {
            return if method == "POST" {
                self.handle_prepare_predict(run_id, req)
            } else {
                unknown_run_route()
            };
        }
        if let Some(run_id) = rest.strip_suffix("/predict") {
            return if method == "POST" {
                self.handle_predict(run_id, req)
            } else {
                unknown_run_route()
            };
        }
        if let Some(run_id) = rest.strip_suffix("/explain") {
            return if method == "POST" {
                self.handle_explain(run_id, req)
            } else {
                unknown_run_route()
            };
        }
        if let Some(run_id) = rest.strip_suffix("/export") {
            return if method == "POST" {
                self.handle_export(run_id, req)
            } else {
                unknown_run_route()
            };
        }
        if !rest.contains('/') {
            return match method {
                "GET" => self.handle_get_run(rest, req),
                "DELETE" => self.handle_shred_run(rest, req),
                _ => unknown_run_route(),
            };
        }
        unknown_run_route()
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

    fn client_supports_v120_progress(req: &CarbonGatewayRequest) -> bool {
        let Some(raw) = req.header_ci("x-carbon-client-version") else {
            return false;
        };
        let clean = raw
            .trim()
            .trim_start_matches("carbon-cli-")
            .trim_start_matches('v');
        let mut parts = clean.split(['.', '-']);
        let Some(maj) = parts.next().and_then(|s| s.parse::<u32>().ok()) else {
            return false;
        };
        let Some(min) = parts.next().and_then(|s| s.parse::<u32>().ok()) else {
            return false;
        };
        (maj, min) >= (1, 2)
    }

    fn progress_json_for_client(
        status: u16,
        req: &CarbonGatewayRequest,
        mut progress: CarbonPublicWaveProgress,
    ) -> CarbonGatewayResponse {
        if !Self::client_supports_v120_progress(req) {
            progress.eval_fallback_shards = None;
        }
        CarbonGatewayResponse::json_ok(status, &progress)
    }

    fn preverify_bearer_before_json(
        &self,
        req: &CarbonGatewayRequest,
    ) -> Result<(), CarbonGatewayResponse> {
        if let Some(verifier) = &self.config.auth_verifier {
            let cap_hdr = Self::capability_auth_header(req);
            verifier
                .verify_bearer_header(cap_hdr, now_epoch_ms() / 1_000)
                .map_err(|e| {
                    CarbonGatewayResponse::error_json(401, "unauthorized", &e.to_string())
                })?;
        }
        Ok(())
    }

    fn parse_authed_json<T: DeserializeOwned>(
        &self,
        req: &CarbonGatewayRequest,
        max_bytes: usize,
        size_code: &str,
        size_msg: &str,
        type_name: &str,
    ) -> Result<(T, Vec<u8>), CarbonGatewayResponse> {
        let body = req.body_cow().map_err(|_| {
            CarbonGatewayResponse::error_json(400, "invalid_body", "malformed body")
        })?;
        if body.len() > max_bytes {
            return Err(CarbonGatewayResponse::error_json(413, size_code, size_msg));
        }
        self.preverify_bearer_before_json(req)?;
        let parsed = serde_json::from_slice::<T>(&body).map_err(|_| {
            CarbonGatewayResponse::error_json(400, "invalid_json", &format!("invalid {type_name}"))
        })?;
        Ok((parsed, body.into_owned()))
    }

    fn load_authorized_active_run(
        &self,
        req: &CarbonGatewayRequest,
        tenant_id: &str,
        run_id: &str,
        rate_prefix: &str,
        rate_limit_per_min: usize,
        rate_msg: &str,
    ) -> Result<CarbonCreateRunRequest, CarbonGatewayResponse> {
        self.authorize_tenant(req, tenant_id, None)?;
        let now_secs = now_epoch_ms() / 1_000;
        if !self.rate_limiter.check_and_record(
            &format!("{rate_prefix}:{tenant_id}"),
            now_secs,
            rate_limit_per_min,
        ) {
            return Err(CarbonGatewayResponse::error_json(
                429,
                "rate_limited",
                rate_msg,
            ));
        }
        if validate_strict_run_id(run_id).is_err() || self.is_run_shredded(tenant_id, run_id) {
            return Err(CarbonGatewayResponse::error_json(
                404,
                "run_not_found",
                "run not found",
            ));
        }
        let req_key = canonical_run_state_key(tenant_id, run_id, "request.json")
            .map_err(|_| CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key"))?;
        match self.cas.get_state_object(&req_key) {
            Ok(Some(bytes)) => {
                serde_json::from_slice::<CarbonCreateRunRequest>(&bytes).map_err(|_| {
                    CarbonGatewayResponse::error_json(
                        500,
                        "corrupt_request",
                        "corrupt stored request",
                    )
                })
            }
            Ok(None) => Err(CarbonGatewayResponse::error_json(
                404,
                "run_not_found",
                "run not found",
            )),
            Err(err) => Err(CarbonGatewayResponse::error_json(
                500,
                "cas_read_failed",
                &err,
            )),
        }
    }

    fn presign_scoped_put_url(
        &self,
        req: &CarbonGatewayRequest,
        blob_scope: &str,
        sha_lower: &str,
        exact_bytes: u64,
        now_secs: u64,
    ) -> Result<String, CarbonGatewayResponse> {
        let expires = now_secs.saturating_add(900);
        if let Some((bucket, prefix)) = &self.s3_upload_target {
            let creds = crate::security::sigv4::CarbonAwsCredentials::from_env().map_err(|_| {
                CarbonGatewayResponse::error_json(
                    500,
                    "presign_creds_missing",
                    "missing AWS execution credentials for S3 presigning",
                )
            })?;
            let region = self.active_aws_region();
            let (_, amz_date) = crate::security::sigv4::format_amz_timestamp_utc(now_secs);
            let s3_key = join_s3_scoped_prefix(prefix, Some(blob_scope), sha_lower);
            CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
                &creds,
                &region,
                bucket,
                &s3_key,
                &amz_date,
                900,
                sha_lower,
                exact_bytes,
            )
            .map_err(|_| {
                CarbonGatewayResponse::error_json(
                    500,
                    "presign_failed",
                    "failed to sign size-pinned S3 PUT URL",
                )
            })
        } else {
            let base_url = self.resolve_public_base_url(req);
            Ok(self.config.presigner.sign_gateway_scoped_upload_url(
                &base_url,
                Some(blob_scope),
                sha_lower,
                exact_bytes,
                expires,
            ))
        }
    }

    fn presign_scoped_get_url(
        &self,
        req: &CarbonGatewayRequest,
        out_scope: Option<&str>,
        sha_lower: &str,
        now_secs: u64,
    ) -> Result<String, CarbonGatewayResponse> {
        if let Some((bucket, prefix)) = &self.s3_upload_target {
            let creds = crate::security::sigv4::CarbonAwsCredentials::from_env().map_err(|_| {
                CarbonGatewayResponse::error_json(
                    500,
                    "presign_creds_missing",
                    "missing AWS execution credentials for S3 GET presigning",
                )
            })?;
            let region = self.active_aws_region();
            let (_, amz_date) = crate::security::sigv4::format_amz_timestamp_utc(now_secs);
            let s3_key = join_s3_scoped_prefix(prefix, out_scope, sha_lower);
            CarbonPresigner::presign_s3_get_v4_with_creds(
                &creds, &region, bucket, &s3_key, &amz_date, 900,
            )
            .map_err(|_| {
                CarbonGatewayResponse::error_json(
                    500,
                    "presign_failed",
                    "failed to sign predict output S3 GET URL",
                )
            })
        } else {
            let base_url = self.resolve_public_base_url(req);
            Ok(self.config.presigner.sign_gateway_scoped_download_url(
                &base_url,
                out_scope,
                sha_lower,
                now_secs.saturating_add(900),
            ))
        }
    }

    fn active_aws_region(&self) -> String {
        match std::env::var("AWS_REGION") {
            Ok(val) if !val.trim().is_empty() => val,
            _ => self.config.region.clone(),
        }
    }

    #[cfg(feature = "serve")]
    pub(crate) fn precheck_put_cas_presign(
        &self,
        raw_path: &str,
        raw_query: Option<&str>,
        req_id: &str,
    ) -> Result<u64, CarbonGatewayResponse> {
        let clean_path = raw_path.trim_end_matches('/');
        let sha = clean_path.strip_prefix("/v1/carbon/cas/").unwrap_or("");
        let mut query = BTreeMap::new();
        if let Some(rq) = raw_query {
            for pair in url::form_urlencoded::parse(rq.as_bytes()) {
                query.insert(pair.0.into_owned(), pair.1.into_owned());
            }
        }
        let wrap = |r: CarbonGatewayResponse| self.finalize_gateway_response(req_id, true, r);
        let Some(bytes_str) = query.get("bytes") else {
            return Err(wrap(CarbonGatewayResponse::error_json(
                400,
                "missing_param",
                "missing bytes query",
            )));
        };
        let Some(exp_str) = query.get("exp") else {
            return Err(wrap(CarbonGatewayResponse::error_json(
                400,
                "missing_param",
                "missing exp query",
            )));
        };
        let Some(sig) = query.get("sig") else {
            return Err(wrap(CarbonGatewayResponse::error_json(
                401,
                "missing_sig",
                "missing sig query",
            )));
        };
        let scope = query
            .get("scope")
            .map(String::as_str)
            .filter(|s| !s.is_empty());
        let (Ok(declared_bytes), Ok(exp_secs)) = (bytes_str.parse::<u64>(), exp_str.parse::<u64>())
        else {
            return Err(wrap(CarbonGatewayResponse::error_json(
                400,
                "bad_query",
                "invalid bytes or exp",
            )));
        };
        if declared_bytes > 268_435_456 {
            return Err(wrap(CarbonGatewayResponse::error_json(
                413,
                "payload_too_large",
                "CAS upload exceeds 256 MiB wire limit",
            )));
        }
        let now_secs = now_epoch_ms() / 1_000;
        if let Err(reason) = self.config.presigner.verify_gateway_scoped_upload(
            scope,
            sha,
            declared_bytes,
            exp_secs,
            sig,
            now_secs,
        ) {
            return Err(wrap(CarbonGatewayResponse::error_json(
                403,
                "presign_denied",
                reason,
            )));
        }
        Ok(declared_bytes)
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
        let Ok(body) = req.body_cow() else {
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
        let (prep, _) = match self.parse_authed_json::<CarbonPrepareUploadRequest>(
            req,
            MAX_CONTROL_JSON_BYTES,
            "payload_too_large",
            "upload prepare payload exceeds 64 KiB limit",
            "CarbonPrepareUploadRequest",
        ) {
            Ok(v) => v,
            Err(resp) => return resp,
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
        let fit_url = match self.presign_scoped_put_url(
            req,
            &blob_scope,
            &fit_sha_lower,
            prep.fit_jmat.bytes.0,
            now_secs,
        ) {
            Ok(u) => u,
            Err(resp) => return resp,
        };
        let holdout_url = match self.presign_scoped_put_url(
            req,
            &blob_scope,
            &holdout_sha_lower,
            prep.holdout_jmat.bytes.0,
            now_secs,
        ) {
            Ok(u) => u,
            Err(resp) => return resp,
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
        let Ok(payload) = req.body_cow() else {
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
            Ok(bytes) => {
                CarbonGatewayResponse::raw_bytes_ok(200, "application/octet-stream", bytes)
            }
            Err(err) => CarbonGatewayResponse::error_json(404, "blob_not_found", &err),
        }
    }

    fn handle_create_run(&self, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let (mut create, _) = match self.parse_authed_json::<CarbonCreateRunRequest>(
            req,
            MAX_CONTROL_JSON_BYTES,
            "payload_too_large",
            "create run request exceeds 64 KiB limit",
            "CarbonCreateRunRequest",
        ) {
            Ok(v) => v,
            Err(resp) => return resp,
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
        let Ok(stored_req_bytes) = to_json_vec(&persisted_req) else {
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
                            return Self::progress_json_for_client(202, req, cur_progress);
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
        let Ok(status_bytes) = to_json_vec(&initial_progress) else {
            return CarbonGatewayResponse::error_json(
                500,
                "serialize_failed",
                "failed to encode initial run progress",
            );
        };
        if let Err(err) = self.cas.put_state_object(&status_key, &status_bytes) {
            return CarbonGatewayResponse::error_json(500, "cas_write_failed", &err);
        }

        let via_public_apigw = req.header_ci("x-carbon-via-apigw-v2") == Some("1");
        let cell_volume = total_rows.saturating_mul(create.feature_names.len());
        let dispatch_async = via_public_apigw
            || effort >= CarbonEffortTier::Medium
            || plan_decision.mode == CarbonExecutionMode::HybridFanOut
            || cell_volume >= 2_500
            || total_rows >= 400;
        let cmd = CarbonEngineWireCommand::StartRun {
            request: create,
            region: self.config.region.clone(),
        };
        if dispatch_async {
            match self.engine.invoke_engine_async(&cmd) {
                Ok(()) => Self::progress_json_for_client(202, req, initial_progress),
                Err(err) => CarbonGatewayResponse::error_json(502, "engine_async_failed", &err),
            }
        } else {
            match self.engine.invoke_engine(&cmd) {
                Ok(resp_bytes) => {
                    if let Ok(progress) =
                        serde_json::from_slice::<CarbonPublicWaveProgress>(&resp_bytes)
                    {
                        Self::progress_json_for_client(201, req, progress)
                    } else {
                        Self::progress_json_for_client(201, req, initial_progress)
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
                Ok(progress) => Self::progress_json_for_client(200, req, progress),
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
        };
        if validate_strict_run_id(run_id).is_err() || self.is_run_shredded(tenant, run_id) {
            return CarbonGatewayResponse::error_json(404, "run_not_found", "run not found");
        }
        let Ok(key) = canonical_run_state_key(tenant, run_id, "status.json") else {
            return CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key");
        };
        match self.cas.get_state_object(&key) {
            Ok(Some(bytes)) => {
                let text = if !Self::client_supports_v120_progress(req) {
                    if let Ok(mut prog) = serde_json::from_slice::<CarbonPublicWaveProgress>(&bytes)
                    {
                        prog.eval_fallback_shards = None;
                        to_json_string(&prog)
                            .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned())
                    } else {
                        String::from_utf8_lossy(&bytes).into_owned()
                    }
                } else {
                    String::from_utf8_lossy(&bytes).into_owned()
                };
                let sse = format!("event: wave_progress\ndata: {text}\n\n");
                CarbonGatewayResponse::text(200, "text/event-stream", &sse)
            }
            Ok(None) => CarbonGatewayResponse::error_json(404, "run_not_found", "run not found"),
            Err(err) => CarbonGatewayResponse::error_json(500, "cas_read_failed", &err),
        }
    }

    fn handle_early_stop(&self, run_id: &str, req: &CarbonGatewayRequest) -> CarbonGatewayResponse {
        let (stop_req, body) = match self.parse_authed_json::<CarbonEarlyStopRequest>(
            req,
            MAX_CONTROL_JSON_BYTES,
            "payload_too_large",
            "early stop payload exceeds 64 KiB limit",
            "CarbonEarlyStopRequest",
        ) {
            Ok(v) => v,
            Err(resp) => return resp,
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
        if validate_strict_run_id(run_id).is_err() {
            return CarbonGatewayResponse::error_json(400, "invalid_run_id", "invalid run_id");
        }
        if self.is_run_shredded(&stop_req.tenant_id, run_id) {
            return CarbonGatewayResponse::error_json(
                410,
                "run_shredded",
                "run has been shredded and cannot accept stop signals",
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
        let (resume_req, _) = match self.parse_authed_json::<CarbonResumeRunRequest>(
            req,
            MAX_CONTROL_JSON_BYTES,
            "payload_too_large",
            "resume payload exceeds 64 KiB limit",
            "CarbonResumeRunRequest",
        ) {
            Ok(v) => v,
            Err(resp) => return resp,
        };
        let stored_create = match self.load_authorized_active_run(
            req,
            &resume_req.tenant_id,
            run_id,
            "resume",
            120,
            "resume rate limit exceeded",
        ) {
            Ok(c) => c,
            Err(resp) => return resp,
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

        let via_public_apigw = req.header_ci("x-carbon-via-apigw-v2") == Some("1");
        let extra_waves = resume_req.additional_waves.unwrap_or(2);
        let stored_volume =
            (stored_create.fit_rows as usize).saturating_mul(stored_create.feature_names.len());
        let async_resume = via_public_apigw
            || extra_waves > 2
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
                        if let Ok(updated) = to_json_vec(&prog) {
                            let _ = self.cas.put_state_object(&status_key, &updated);
                        }
                        return match self.engine.invoke_engine_async(&cmd) {
                            Ok(()) => Self::progress_json_for_client(202, req, prog),
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
                Ok(progress) => Self::progress_json_for_client(200, req, progress),
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
        let (prep, _) = match self.parse_authed_json::<CarbonPreparePredictRequest>(
            req,
            MAX_CONTROL_JSON_BYTES,
            "payload_too_large",
            "predict prepare payload exceeds 64 KiB limit",
            "CarbonPreparePredictRequest",
        ) {
            Ok(v) => v,
            Err(resp) => return resp,
        };
        let stored_create = match self.load_authorized_active_run(
            req,
            &prep.tenant_id,
            run_id,
            "pred_prep",
            300,
            "predict prepare rate limit exceeded",
        ) {
            Ok(c) => c,
            Err(resp) => return resp,
        };
        if prep.predict_cuen.bytes.0 == 0 || !is_valid_sha256_hex(&prep.predict_cuen.sha256) {
            return CarbonGatewayResponse::error_json(
                400,
                "invalid_upload_slot",
                "predict_cuen requires non-zero byte count and 64-char hex SHA-256",
            );
        }
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
        let now_secs = now_epoch_ms() / 1_000;
        let expires = now_secs.saturating_add(900);
        let put_url = match self.presign_scoped_put_url(
            req,
            &blob_scope,
            &sha_lower,
            prep.predict_cuen.bytes.0,
            now_secs,
        ) {
            Ok(u) => u,
            Err(resp) => return resp,
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
        let (pred_req, _) = match self.parse_authed_json::<CarbonPredictRequest>(
            req,
            32 * 1_048_576,
            "predict_payload_too_large",
            "predict request exceeds 32 MiB wire limit",
            "CarbonPredictRequest",
        ) {
            Ok(v) => v,
            Err(resp) => return resp,
        };
        let stored_create = match self.load_authorized_active_run(
            req,
            &pred_req.tenant_id,
            run_id,
            "pred",
            300,
            "predict rate limit exceeded",
        ) {
            Ok(c) => c,
            Err(resp) => return resp,
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

        // Cross-container linearizable `/predict` quota reservation (`RACE-01` / `RT-10`).
        let reserved_slot_key = if let Some(max_calls) = tier.max_predict_calls_per_run() {
            match self.reserve_predict_quota_slot(&pred_req.tenant_id, run_id, max_calls, &tier) {
                Ok(slot_key) => Some(slot_key),
                Err(resp) => return resp,
            }
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
                    if let Some(ref out_sha) = resp.encrypted_predictions_cas_sha256 {
                        let out_scope = match pred_upload_id.as_deref() {
                            Some(puid) => canonical_blob_scope(&pred_tenant_id, puid).ok(),
                            None => canonical_blob_scope(&pred_tenant_id, run_id).ok(),
                        };
                        self.record_predict_output_blob(
                            &pred_tenant_id,
                            run_id,
                            out_scope.as_deref(),
                            &out_sha.to_ascii_lowercase(),
                        );
                        let dl_now = now_epoch_ms() / 1_000;
                        let dl_url = match self.presign_scoped_get_url(
                            req,
                            out_scope.as_deref(),
                            &out_sha.to_ascii_lowercase(),
                            dl_now,
                        ) {
                            Ok(u) => u,
                            Err(err_resp) => return err_resp,
                        };
                        resp.encrypted_predictions_cas_url = Some(dl_url);
                    }
                    CarbonGatewayResponse::json_ok(200, &resp)
                }
                Err(_) => {
                    if let Some(ref sk) = reserved_slot_key {
                        let _ = self.cas.delete_state_object(sk);
                    }
                    CarbonGatewayResponse::error_json(
                        500,
                        "invalid_engine_response",
                        "malformed predict response",
                    )
                }
            },
            Err(err) => {
                if let Some(ref sk) = reserved_slot_key {
                    let _ = self.cas.delete_state_object(sk);
                }
                CarbonGatewayResponse::error_json(502, "engine_predict_failed", &err)
            }
        }
    }

    fn reserve_predict_quota_slot(
        &self,
        tenant_id: &str,
        run_id: &str,
        max_calls: u32,
        tier: &CarbonProductTier,
    ) -> Result<String, CarbonGatewayResponse> {
        let quota_err = || {
            CarbonGatewayResponse::error_json(
                403,
                "predict_quota_exceeded",
                &format!(
                    "{} tier admits at most {max_calls} /predict calls per run",
                    tier.as_str()
                ),
            )
        };
        if max_calls == 0 {
            return Err(quota_err());
        }
        let cnt_key = canonical_run_state_key(tenant_id, run_id, "predict_count.json")
            .map_err(|_| CarbonGatewayResponse::error_json(400, "invalid_key", "invalid key"))?;
        let hint: u32 = self
            .cas
            .get_state_object(&cnt_key)
            .ok()
            .flatten()
            .and_then(|b| serde_json::from_slice::<u32>(&b).ok())
            .unwrap_or(0);
        if hint >= max_calls {
            return Err(quota_err());
        }
        let probe_limit = max_calls.saturating_sub(hint).min(32);
        for offset in 0..probe_limit {
            let slot = hint.saturating_add(offset).saturating_add(1);
            let Ok(slot_key) =
                canonical_run_state_key(tenant_id, run_id, &format!("predict_slot_{slot}.json"))
            else {
                continue;
            };
            match self.cas.put_state_object_if_absent(&slot_key, b"1") {
                Ok(true) => {
                    if let Ok(next_bytes) = to_json_vec(&slot) {
                        let _ = self.cas.put_state_object(&cnt_key, &next_bytes);
                    }
                    return Ok(slot_key);
                }
                Ok(false) => continue,
                Err(err) => {
                    return Err(CarbonGatewayResponse::error_json(
                        500,
                        "cas_write_failed",
                        &err,
                    ));
                }
            }
        }
        let advanced = hint.saturating_add(probe_limit).min(max_calls);
        if let Ok(next_bytes) = to_json_vec(&advanced) {
            let _ = self.cas.put_state_object(&cnt_key, &next_bytes);
        }
        Err(quota_err())
    }

    fn record_predict_output_blob(
        &self,
        tenant_id: &str,
        run_id: &str,
        scope: Option<&str>,
        sha_lc: &str,
    ) {
        let Ok(list_key) = canonical_run_state_key(tenant_id, run_id, "predict_outputs.json")
        else {
            return;
        };
        let mut entries: Vec<(Option<String>, String)> = self
            .cas
            .get_state_object(&list_key)
            .ok()
            .flatten()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        if entries.len() < 128 {
            entries.push((scope.map(str::to_owned), sha_lc.to_owned()));
            if let Ok(bytes) = to_json_vec(&entries) {
                let _ = self.cas.put_state_object(&list_key, &bytes);
            }
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
        if validate_strict_run_id(run_id).is_err() {
            return CarbonGatewayResponse::error_json(400, "invalid_run_id", "invalid RunId");
        }
        let Ok(parsed_run_id) = run_id.parse() else {
            return CarbonGatewayResponse::error_json(400, "invalid_run_id", "invalid RunId");
        };
        // Persist tombstone FIRST (`RACE-01`) so concurrent GET/POST/stop/resume requests
        // immediately observe the run as shredded while artifacts are being erased.
        if let Ok(tomb_key) = canonical_run_state_key(tenant, run_id, "shredded.json") {
            if let Err(err) = self
                .cas
                .put_state_object(&tomb_key, br#"{"shredded":true}"#)
            {
                return CarbonGatewayResponse::error_json(500, "cas_write_failed", &err);
            }
        }
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
        if let Ok(out_key) = canonical_run_state_key(tenant, run_id, "predict_outputs.json") {
            if let Ok(Some(out_bytes)) = self.cas.get_state_object(&out_key) {
                if let Ok(entries) =
                    serde_json::from_slice::<Vec<(Option<String>, String)>>(&out_bytes)
                {
                    for (sc, sha_lc) in entries {
                        if self
                            .cas
                            .delete_scoped_blob(sc.as_deref(), &sha_lc)
                            .unwrap_or(false)
                        {
                            deleted += 1;
                        }
                    }
                }
                let _ = self.cas.delete_state_object(&out_key);
            }
        }
        let max_slot_to_sweep = canonical_run_state_key(tenant, run_id, "predict_count.json")
            .ok()
            .and_then(|ck| self.cas.get_state_object(&ck).ok().flatten())
            .and_then(|b| serde_json::from_slice::<u32>(&b).ok())
            .unwrap_or(0)
            .clamp(32, 1_000);
        for slot in 1..=max_slot_to_sweep {
            if let Ok(sk) =
                canonical_run_state_key(tenant, run_id, &format!("predict_slot_{slot}.json"))
            {
                let _ = self.cas.delete_state_object(&sk);
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
        let (explain_req, _) = match self.parse_authed_json::<CarbonExplainRequest>(
            req,
            MAX_CONTROL_JSON_BYTES,
            "payload_too_large",
            "explain payload exceeds 64 KiB limit",
            "CarbonExplainRequest",
        ) {
            Ok(v) => v,
            Err(resp) => return resp,
        };
        let stored_create = match self.load_authorized_active_run(
            req,
            &explain_req.tenant_id,
            run_id,
            "explain",
            120,
            "explain rate limit exceeded",
        ) {
            Ok(c) => c,
            Err(resp) => return resp,
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
        let (export_req, _) = match self.parse_authed_json::<CarbonExportRequest>(
            req,
            MAX_CONTROL_JSON_BYTES,
            "payload_too_large",
            "export payload exceeds 64 KiB limit",
            "CarbonExportRequest",
        ) {
            Ok(v) => v,
            Err(resp) => return resp,
        };
        let stored_create = match self.load_authorized_active_run(
            req,
            &export_req.tenant_id,
            run_id,
            "export",
            120,
            "export rate limit exceeded",
        ) {
            Ok(c) => c,
            Err(resp) => return resp,
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
        crate::cells::edge_downloads::execute_passive_download_redirect(
            &self.config.region,
            &self.config.presigner,
            &self.rate_limiter,
            method,
            clean_path,
            req_id,
            req,
        )
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

fn unknown_run_route() -> CarbonGatewayResponse {
    CarbonGatewayResponse::error_json(404, "not_found", "unknown Carbon run route")
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
            // Strip untrusted synthetic headers (`GAP-SEC-4`).
            if k.eq_ignore_ascii_case("x-carbon-trusted-source-ip")
                || k.eq_ignore_ascii_case("x-carbon-raw-path")
                || k.eq_ignore_ascii_case("x-carbon-via-apigw-v2")
            {
                continue;
            }
            if let Some(text) = v.as_str() {
                headers.insert(k.to_ascii_lowercase(), text.to_owned());
            }
        }
    }
    headers.insert("x-carbon-raw-path".to_owned(), path.clone());
    // Detect AWS API Gateway HTTP API v2.0 (30,000 ms hard timeout ceiling) vs direct Function URL
    // / direct Lambda invoke so `StartRun` / `ResumeRun` dispatch asynchronously (`202 Accepted`).
    let api_id_present = event
        .pointer("/requestContext/apiId")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| !s.trim().is_empty());
    let domain_name = event
        .pointer("/requestContext/domainName")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if api_id_present && !domain_name.contains(".lambda-url.") {
        headers.insert("x-carbon-via-apigw-v2".to_owned(), "1".to_owned());
    }

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
        Some(body_str) if is_b64 => {
            if body_str.len() > 48 * 1_048_576 {
                return Err("Function URL v2 base64 body exceeds 48 MiB wire limit".to_owned());
            }
            base64::engine::general_purpose::STANDARD
                .decode(body_str)
                .map_err(err_to_string)?
        }
        Some(body_str) => {
            if body_str.len() > 36 * 1_048_576 {
                return Err("Function URL v2 body exceeds 36 MiB wire limit".to_owned());
            }
            body_str.as_bytes().to_vec()
        }
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
    let raw_slice: Option<&[u8]> = resp.raw_body.as_deref();
    let (body_field, is_b64) = if resp
        .content_type
        .eq_ignore_ascii_case("application/octet-stream")
    {
        let b64 = match raw_slice {
            Some(bytes) => base64::engine::general_purpose::STANDARD.encode(bytes),
            None => resp.body_b64.clone(),
        };
        (b64, true)
    } else if let Some(bytes) = raw_slice {
        match std::str::from_utf8(bytes) {
            Ok(utf8_str) => (utf8_str.to_owned(), false),
            Err(_) => (
                base64::engine::general_purpose::STANDARD.encode(bytes),
                true,
            ),
        }
    } else {
        let decoded = resp.decoded_body();
        match String::from_utf8(decoded) {
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
    to_json_vec(&out)
}
