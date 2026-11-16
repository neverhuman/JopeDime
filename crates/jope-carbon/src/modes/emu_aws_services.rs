//! Loopback AWS SigV4 S3 CAS, Secrets Manager, Lambda Invoke, and API Gateway HTTP API v2.0
//! emulation for real OS-subprocess `carbon-edge` + `carbon-worker` testing (`carbon emu --real-workers`).
//!
//! Enforces bit-exact `AWS4-HMAC-SHA256` header and presigned-URL verification, per-role IAM
//! boundaries (`edge` vs `engine` vs `eval`), S3 `If-None-Match: *` -> `412 PreconditionFailed`,
//! Secrets Manager `VersionId` pinning, Lambda `200 + X-Amz-Function-Error: Unhandled` propagation,
//! and API Gateway v2.0's 30-second front-door timeout ceiling.

use std::collections::BTreeMap;
use std::net::IpAddr;
use std::path::PathBuf;
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path as AxumPath, State},
    http::{HeaderMap, Request, Response, StatusCode},
};
use base64::Engine as _;
use http_body_util::{BodyExt as _, Limited};
use sha2::{Digest, Sha256};

use crate::{
    cells::edge::{CarbonGatewayRequest, CarbonGatewayResponse},
    error::{err_to_string, to_json_vec},
    modes::{emu::EmulatorSharedState, runtime_driver::plain_axum_response as plain_status},
    security::sigv4::{derive_sigv4_signing_key, url_encode_component},
    storage::cas::{now_epoch_secs, sha256_lower_hex, CarbonCasStore},
};

pub const EMU_S3_BUCKET: &str = "carbon-emu-cas";
pub const EMU_EDGE_FUNCTION_NAME: &str = "carbon-emu-edge";
pub const EMU_ENGINE_FUNCTION_NAME: &str = "carbon-emu-engine";
pub const EMU_EVAL_FUNCTION_NAME: &str = "carbon-emu-eval";

pub const EMU_EDGE_ACCESS_KEY_ID: &str = "emu_edge_role_key_01";
pub const EMU_EDGE_SECRET_ACCESS_KEY: &str = "emu_edge_secret_key_00000000000000000001";
pub const EMU_ENGINE_ACCESS_KEY_ID: &str = "emu_engine_role_key2";
pub const EMU_ENGINE_SECRET_ACCESS_KEY: &str = "emu_engine_secret_key_000000000000000002";
pub const EMU_EVAL_ACCESS_KEY_ID: &str = "emu_eval_role_key_03";
pub const EMU_EVAL_SECRET_ACCESS_KEY: &str = "emu_eval_secret_key_00000000000000000003";
pub const EMU_CLIENT_ACCESS_KEY_ID: &str = "emu_client_role_key4";
pub const EMU_CLIENT_SECRET_ACCESS_KEY: &str = "emu_client_secret_key_000000000000000004";

pub const EMU_EDGE_SECRETS_VERSION: &str = "v1-edge";
pub const EMU_CELL_SECRETS_VERSION: &str = "v1-cell";

const MAX_EMU_AWS_BODY_BYTES: usize = 268_435_456;

/// Real OS-subprocess configuration for `carbon emu` (`carbon-edge` + `carbon-worker`).
#[derive(Clone, Debug)]
pub struct CarbonRealWorkerBinaries {
    pub edge_binary_path: PathBuf,
    pub worker_binary_path: PathBuf,
    pub simulate_apigw_v2: bool,
    pub apigw_timeout_ms: u64,
}

impl CarbonRealWorkerBinaries {
    pub fn new(edge_binary_path: PathBuf, worker_binary_path: PathBuf) -> Self {
        Self {
            edge_binary_path,
            worker_binary_path,
            simulate_apigw_v2: true,
            apigw_timeout_ms: 30_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EmuIamRole {
    Edge,
    Engine,
    Eval,
    Client,
}

pub(crate) struct EmuAwsVault {
    pub region: String,
    pub edge_secret_arn: String,
    pub edge_secret_json: String,
    pub cell_secret_arn: String,
    pub cell_secret_json: String,
    pub simulate_apigw_v2: bool,
    pub apigw_timeout_ms: u64,
}

impl EmuAwsVault {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        region: &str,
        presign_secret: [u8; 32],
        auth_secret: [u8; 32],
        seal_secret: [u8; 32],
        engine_kek_scalar: [u8; 32],
        engine_kek_pubkey_b64: &str,
        simulate_apigw_v2: bool,
        apigw_timeout_ms: u64,
    ) -> Self {
        let edge_secret_arn =
            format!("arn:aws:secretsmanager:{region}:000000000000:secret:carbon-emu-edge-secrets");
        let cell_secret_arn =
            format!("arn:aws:secretsmanager:{region}:000000000000:secret:carbon-emu-cell-secrets");
        // Edge secret bundle strictly omits `seal_secret_hex` and `engine_kek_hex` (`DEP-SEC-01`).
        let edge_secret_json = serde_json::json!({
            "presign_secret_hex": hex::encode(presign_secret),
            "auth_secret_hex": hex::encode(auth_secret),
            "engine_kek_pubkey_b64": engine_kek_pubkey_b64,
        })
        .to_string();
        let cell_secret_json = serde_json::json!({
            "seal_secret_hex": hex::encode(seal_secret),
            "engine_kek_hex": hex::encode(engine_kek_scalar),
            "engine_kek_pubkey_b64": engine_kek_pubkey_b64,
        })
        .to_string();
        Self {
            region: region.to_owned(),
            edge_secret_arn,
            edge_secret_json,
            cell_secret_arn,
            cell_secret_json,
            simulate_apigw_v2,
            apigw_timeout_ms: apigw_timeout_ms.max(50),
        }
    }
}

fn lookup_iam_access_key(access_key_id: &str) -> Option<(EmuIamRole, &'static str)> {
    match access_key_id {
        EMU_EDGE_ACCESS_KEY_ID => Some((EmuIamRole::Edge, EMU_EDGE_SECRET_ACCESS_KEY)),
        EMU_ENGINE_ACCESS_KEY_ID => Some((EmuIamRole::Engine, EMU_ENGINE_SECRET_ACCESS_KEY)),
        EMU_EVAL_ACCESS_KEY_ID => Some((EmuIamRole::Eval, EMU_EVAL_SECRET_ACCESS_KEY)),
        EMU_CLIENT_ACCESS_KEY_ID => Some((EmuIamRole::Client, EMU_CLIENT_SECRET_ACCESS_KEY)),
        _ => None,
    }
}

fn hmac_sha256_bytes(key: &[u8], data: &[u8]) -> [u8; 32] {
    let signing_key = aws_lc_rs::hmac::Key::new(aws_lc_rs::hmac::HMAC_SHA256, key);
    let tag = aws_lc_rs::hmac::sign(&signing_key, data);
    let mut out = [0u8; 32];
    out.copy_from_slice(tag.as_ref());
    out
}

fn parse_amz_timestamp_utc_secs(amz_date: &str) -> Option<u64> {
    if amz_date.len() != 16 || !amz_date.ends_with('Z') || amz_date.as_bytes()[8] != b'T' {
        return None;
    }
    let y: i64 = amz_date[0..4].parse().ok()?;
    let m: i64 = amz_date[4..6].parse().ok()?;
    let d: i64 = amz_date[6..8].parse().ok()?;
    let hh: u64 = amz_date[9..11].parse().ok()?;
    let mm: u64 = amz_date[11..13].parse().ok()?;
    let ss: u64 = amz_date[13..15].parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    // Howard Hinnant civil-to-days algorithm (inverse of `format_amz_timestamp_utc`).
    let y_adj = y - i64::from(m <= 2);
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = (y_adj - era * 400) as u64;
    let mp = (if m > 2 { m - 3 } else { m + 9 }) as u64;
    let doy = (153 * mp + 2) / 5 + (d as u64) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + (doe as i64) - 719_468;
    if days < 0 {
        return None;
    }
    Some((days as u64) * 86_400 + hh * 3_600 + mm * 60 + ss)
}

fn canonicalize_query_excluding(raw_query: Option<&str>, exclude_key: Option<&str>) -> String {
    let Some(q) = raw_query.filter(|s| !s.is_empty()) else {
        return String::new();
    };
    let mut pairs: Vec<(String, String)> = url::form_urlencoded::parse(q.as_bytes())
        .filter(|(k, _)| exclude_key.is_none_or(|ex| k.as_ref() != ex))
        .map(|(k, v)| (url_encode_component(&k), url_encode_component(&v)))
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    pairs
        .into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn verify_sigv4_header_request(
    expected_region: &str,
    expected_service: &str,
    method: &str,
    canonical_uri: &str,
    raw_query: Option<&str>,
    headers: &HeaderMap,
    body_bytes: &[u8],
) -> Result<EmuIamRole, (StatusCode, String)> {
    let auth_raw = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            (
                StatusCode::FORBIDDEN,
                "MissingAuthenticationToken: missing SigV4 Authorization header".to_owned(),
            )
        })?;
    let rest = auth_raw
        .trim()
        .strip_prefix("AWS4-HMAC-SHA256 ")
        .ok_or_else(|| {
            (
                StatusCode::FORBIDDEN,
                "InvalidSignature: expected AWS4-HMAC-SHA256".to_owned(),
            )
        })?;
    let mut cred_val = None;
    let mut signed_hdrs_val = None;
    let mut sig_hex_val = None;
    for part in rest.split(',') {
        let p = part.trim();
        if let Some(v) = p.strip_prefix("Credential=") {
            cred_val = Some(v.trim());
        } else if let Some(v) = p.strip_prefix("SignedHeaders=") {
            signed_hdrs_val = Some(v.trim());
        } else if let Some(v) = p.strip_prefix("Signature=") {
            sig_hex_val = Some(v.trim());
        }
    }
    let cred_parts: Vec<&str> = cred_val.unwrap_or("").split('/').collect();
    if cred_parts.len() != 5 || cred_parts[4] != "aws4_request" {
        return Err((
            StatusCode::FORBIDDEN,
            "IncompleteSignature: malformed Credential scope".to_owned(),
        ));
    }
    let (akid, date_stamp, region, service) =
        (cred_parts[0], cred_parts[1], cred_parts[2], cred_parts[3]);
    if region != expected_region || service != expected_service {
        return Err((
            StatusCode::FORBIDDEN,
            format!(
                "SignatureDoesNotMatch: scope {region}/{service} != {expected_region}/{expected_service}"
            ),
        ));
    }
    let (role, secret_key) = lookup_iam_access_key(akid).ok_or_else(|| {
        (
            StatusCode::FORBIDDEN,
            format!("InvalidClientTokenId: unknown access key '{akid}'"),
        )
    })?;
    let amz_date = headers
        .get("x-amz-date")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            (
                StatusCode::FORBIDDEN,
                "Missing x-amz-date header".to_owned(),
            )
        })?;
    if !amz_date.starts_with(date_stamp) {
        return Err((
            StatusCode::FORBIDDEN,
            "x-amz-date does not match Credential date_stamp".to_owned(),
        ));
    }
    let actual_payload_sha = sha256_lower_hex(body_bytes);
    let claimed_payload_sha = headers
        .get("x-amz-content-sha256")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if claimed_payload_sha != actual_payload_sha {
        return Err((
            StatusCode::FORBIDDEN,
            "XAmzContentSHA256Mismatch: payload digest mismatch".to_owned(),
        ));
    }
    let signed_hdrs_str = signed_hdrs_val.unwrap_or("");
    let mut canonical_headers = String::new();
    for h_name in signed_hdrs_str.split(';').filter(|s| !s.is_empty()) {
        let h_val = headers
            .get(h_name)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                (
                    StatusCode::FORBIDDEN,
                    format!("Missing signed header '{h_name}'"),
                )
            })?;
        canonical_headers.push_str(&h_name.to_ascii_lowercase());
        canonical_headers.push(':');
        canonical_headers.push_str(h_val.trim());
        canonical_headers.push('\n');
    }
    let canonical_query = canonicalize_query_excluding(raw_query, None);
    let normalized_uri = crate::security::sigv4::canonicalize_sigv4_uri(canonical_uri, service);
    let canonical_request = format!(
        "{}\n{normalized_uri}\n{canonical_query}\n{canonical_headers}\n{signed_hdrs_str}\n{actual_payload_sha}",
        method.to_ascii_uppercase()
    );
    let req_hash = format!("{:x}", Sha256::digest(canonical_request.as_bytes()));
    let cred_scope = format!("{date_stamp}/{region}/{service}/aws4_request");
    let string_to_sign = format!("AWS4-HMAC-SHA256\n{amz_date}\n{cred_scope}\n{req_hash}");
    let k_signing = derive_sigv4_signing_key(secret_key, date_stamp, region, service);
    let expected_sig = hmac_sha256_bytes(k_signing.as_ref(), string_to_sign.as_bytes());
    let caller_sig = hex::decode(sig_hex_val.unwrap_or("")).map_err(|_| {
        (
            StatusCode::FORBIDDEN,
            "SignatureDoesNotMatch: non-hex signature".to_owned(),
        )
    })?;
    if aws_lc_rs::constant_time::verify_slices_are_equal(&expected_sig, &caller_sig).is_err() {
        return Err((
            StatusCode::FORBIDDEN,
            "SignatureDoesNotMatch: SigV4 HMAC verification failed".to_owned(),
        ));
    }
    Ok(role)
}

fn verify_s3_presigned_query(
    expected_region: &str,
    method: &str,
    canonical_uri: &str,
    raw_query: &str,
    headers: &HeaderMap,
    body_bytes: &[u8],
) -> Result<EmuIamRole, (StatusCode, String)> {
    let mut params = BTreeMap::new();
    for (k, v) in url::form_urlencoded::parse(raw_query.as_bytes()) {
        params.insert(k.into_owned(), v.into_owned());
    }
    if params.get("X-Amz-Algorithm").map(String::as_str) != Some("AWS4-HMAC-SHA256") {
        return Err((
            StatusCode::FORBIDDEN,
            "AuthorizationQueryParametersError: unsupported X-Amz-Algorithm".to_owned(),
        ));
    }
    let cred_str = params
        .get("X-Amz-Credential")
        .ok_or_else(|| (StatusCode::FORBIDDEN, "Missing X-Amz-Credential".to_owned()))?;
    let cred_parts: Vec<&str> = cred_str.split('/').collect();
    if cred_parts.len() != 5 || cred_parts[3] != "s3" || cred_parts[4] != "aws4_request" {
        return Err((
            StatusCode::FORBIDDEN,
            "Malformed X-Amz-Credential".to_owned(),
        ));
    }
    let (akid, date_stamp, region) = (cred_parts[0], cred_parts[1], cred_parts[2]);
    if region != expected_region {
        return Err((
            StatusCode::FORBIDDEN,
            "Region mismatch in presigned S3 URL".to_owned(),
        ));
    }
    let (role, secret_key) = lookup_iam_access_key(akid).ok_or_else(|| {
        (
            StatusCode::FORBIDDEN,
            "InvalidAccessKeyId in presigned S3 URL".to_owned(),
        )
    })?;
    let amz_date = params
        .get("X-Amz-Date")
        .ok_or_else(|| (StatusCode::FORBIDDEN, "Missing X-Amz-Date".to_owned()))?;
    let expires_secs: u64 = params
        .get("X-Amz-Expires")
        .and_then(|s| s.parse().ok())
        .filter(|&e| (1..=604_800).contains(&e))
        .ok_or_else(|| (StatusCode::FORBIDDEN, "Invalid X-Amz-Expires".to_owned()))?;
    let issued_epoch = parse_amz_timestamp_utc_secs(amz_date)
        .ok_or_else(|| (StatusCode::FORBIDDEN, "Invalid X-Amz-Date".to_owned()))?;
    if now_epoch_secs() > issued_epoch.saturating_add(expires_secs) {
        return Err((
            StatusCode::FORBIDDEN,
            "AccessDenied: Request has expired".to_owned(),
        ));
    }
    let signed_hdrs_str = params.get("X-Amz-SignedHeaders").ok_or_else(|| {
        (
            StatusCode::FORBIDDEN,
            "Missing X-Amz-SignedHeaders".to_owned(),
        )
    })?;
    let sig_hex = params
        .get("X-Amz-Signature")
        .ok_or_else(|| (StatusCode::FORBIDDEN, "Missing X-Amz-Signature".to_owned()))?;

    let mut canonical_headers = String::new();
    let mut signed_sha_hdr: Option<String> = None;
    for h_name in signed_hdrs_str.split(';').filter(|s| !s.is_empty()) {
        let lc = h_name.to_ascii_lowercase();
        let val = if lc == "content-length" {
            let actual_len = body_bytes.len().to_string();
            if let Some(cl_hdr) = headers.get("content-length").and_then(|v| v.to_str().ok()) {
                if cl_hdr.trim() != actual_len {
                    return Err((
                        StatusCode::FORBIDDEN,
                        "SignatureDoesNotMatch: Content-Length mismatch".to_owned(),
                    ));
                }
            }
            actual_len
        } else {
            let v = headers
                .get(lc.as_str())
                .and_then(|hv| hv.to_str().ok())
                .ok_or_else(|| {
                    (
                        StatusCode::FORBIDDEN,
                        format!("Missing signed header '{lc}' on presigned request"),
                    )
                })?;
            if lc == "x-amz-content-sha256" {
                signed_sha_hdr = Some(v.trim().to_owned());
            }
            v.trim().to_owned()
        };
        canonical_headers.push_str(&lc);
        canonical_headers.push(':');
        canonical_headers.push_str(&val);
        canonical_headers.push('\n');
    }

    let payload_hash = if let Some(expected_sha) = signed_sha_hdr {
        let actual_sha = sha256_lower_hex(body_bytes);
        if expected_sha != actual_sha {
            return Err((
                StatusCode::FORBIDDEN,
                "SignatureDoesNotMatch: pinned x-amz-content-sha256 mismatch".to_owned(),
            ));
        }
        expected_sha
    } else {
        "UNSIGNED-PAYLOAD".to_owned()
    };

    let canonical_query = canonicalize_query_excluding(Some(raw_query), Some("X-Amz-Signature"));
    let canonical_request = format!(
        "{}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_hdrs_str}\n{payload_hash}",
        method.to_ascii_uppercase()
    );
    let req_hash = format!("{:x}", Sha256::digest(canonical_request.as_bytes()));
    let cred_scope = format!("{date_stamp}/{region}/s3/aws4_request");
    let string_to_sign = format!("AWS4-HMAC-SHA256\n{amz_date}\n{cred_scope}\n{req_hash}");
    let k_signing = derive_sigv4_signing_key(secret_key, date_stamp, region, "s3");
    let expected_sig = hmac_sha256_bytes(k_signing.as_ref(), string_to_sign.as_bytes());
    let caller_sig = hex::decode(sig_hex).map_err(|_| {
        (
            StatusCode::FORBIDDEN,
            "SignatureDoesNotMatch: invalid hex X-Amz-Signature".to_owned(),
        )
    })?;
    if aws_lc_rs::constant_time::verify_slices_are_equal(&expected_sig, &caller_sig).is_err() {
        return Err((
            StatusCode::FORBIDDEN,
            "SignatureDoesNotMatch: presigned S3 signature mismatch".to_owned(),
        ));
    }
    Ok(role)
}

/// Spawn real OS child processes (`carbon-edge` + `carbon-worker` engine + `carbon-worker` eval)
/// configured strictly with `AWS_LAMBDA_RUNTIME_API`, `s3://carbon-emu-cas/v1`, per-role IAM keys,
/// and Secrets Manager ARNs.
pub(crate) fn spawn_real_lambda_worker_processes(
    bins: &CarbonRealWorkerBinaries,
    vault: &EmuAwsVault,
    bound_addr: std::net::SocketAddr,
    data_dir: &std::path::Path,
    force_hybrid_fanout: bool,
    eval_worker_count: usize,
) -> Result<Vec<Child>, String> {
    let origin = format!("http://{bound_addr}");
    let s3_cas_url = format!("s3://{EMU_S3_BUCKET}/v1");
    let mut children = Vec::new();

    let spawn_err = |msg: String, spawned: &mut Vec<Child>| -> String {
        for mut c in spawned.drain(..) {
            let _ = c.kill();
            let _ = c.wait();
        }
        msg
    };

    // 1. Regional `carbon-edge` OS process (`AWS_LAMBDA_RUNTIME_API`, `CARBON_ROLE=edge`).
    let edge_child = ProcessCommand::new(&bins.edge_binary_path)
        .env(
            "AWS_LAMBDA_RUNTIME_API",
            format!("{bound_addr}/runtime/edge"),
        )
        .env("AWS_REGION", &vault.region)
        .env("CARBON_REGION", &vault.region)
        .env("CARBON_ROLE", "edge")
        .env("CARBON_PUBLIC_BASE_URL", &origin)
        .env("CARBON_CAS_URL", &s3_cas_url)
        .env(
            "CARBON_ENGINE_INVOKE_URL",
            format!("lambda://{EMU_ENGINE_FUNCTION_NAME}:live"),
        )
        .env("CARBON_S3_ENDPOINT_URL", &origin)
        .env("CARBON_SECRETSMANAGER_ENDPOINT_URL", &origin)
        .env("CARBON_LAMBDA_ENDPOINT_URL", &origin)
        .env("CARBON_SECRETS_ARN", &vault.edge_secret_arn)
        .env("CARBON_SECRETS_VERSION_ID", EMU_EDGE_SECRETS_VERSION)
        .env("AWS_ACCESS_KEY_ID", EMU_EDGE_ACCESS_KEY_ID)
        .env("AWS_SECRET_ACCESS_KEY", EMU_EDGE_SECRET_ACCESS_KEY)
        .env_remove("AWS_SESSION_TOKEN")
        .env_remove("AWS_PROFILE")
        .env_remove("CARBON_SEAL_SECRET_HEX")
        .env_remove("CARBON_ENGINE_KEK_HEX")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| {
            format!(
                "spawn real carbon-edge {}: {e}",
                bins.edge_binary_path.display()
            )
        })?;
    children.push(edge_child);

    // 2. Sealed `carbon-worker` (`CARBON_ROLE=engine`) OS processes.
    for idx in 0..2usize {
        let work_dir = data_dir.join(format!("real_engine_proc_{idx}"));
        let child = ProcessCommand::new(&bins.worker_binary_path)
            .env(
                "AWS_LAMBDA_RUNTIME_API",
                format!("{bound_addr}/runtime/engine"),
            )
            .env("AWS_REGION", &vault.region)
            .env("CARBON_REGION", &vault.region)
            .env("CARBON_ROLE", "engine")
            .env("CARBON_CAS_URL", &s3_cas_url)
            .env(
                "CARBON_EVAL_INVOKE_URL",
                format!("lambda://{EMU_EVAL_FUNCTION_NAME}:live"),
            )
            .env(
                "CARBON_FORCE_HYBRID_FANOUT",
                if force_hybrid_fanout { "1" } else { "0" },
            )
            .env("CARBON_WORK_DIR", &work_dir)
            .env("CARBON_S3_ENDPOINT_URL", &origin)
            .env("CARBON_SECRETSMANAGER_ENDPOINT_URL", &origin)
            .env("CARBON_LAMBDA_ENDPOINT_URL", &origin)
            .env("CARBON_SECRETS_ARN", &vault.cell_secret_arn)
            .env("CARBON_SECRETS_VERSION_ID", EMU_CELL_SECRETS_VERSION)
            .env("AWS_ACCESS_KEY_ID", EMU_ENGINE_ACCESS_KEY_ID)
            .env("AWS_SECRET_ACCESS_KEY", EMU_ENGINE_SECRET_ACCESS_KEY)
            .env_remove("AWS_SESSION_TOKEN")
            .env_remove("AWS_PROFILE")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                spawn_err(
                    format!(
                        "spawn real carbon-worker engine #{idx} {}: {e}",
                        bins.worker_binary_path.display()
                    ),
                    &mut children,
                )
            })?;
        children.push(child);
    }

    // 3. Stateless `carbon-worker` (`CARBON_ROLE=eval`, zero Secrets Manager access) OS processes.
    for idx in 0..eval_worker_count.max(1) {
        let scratch_dir = data_dir.join(format!("real_eval_proc_{idx}"));
        let child = ProcessCommand::new(&bins.worker_binary_path)
            .env(
                "AWS_LAMBDA_RUNTIME_API",
                format!("{bound_addr}/runtime/eval"),
            )
            .env("AWS_REGION", &vault.region)
            .env("CARBON_REGION", &vault.region)
            .env("CARBON_ROLE", "eval")
            .env("CARBON_CAS_URL", &s3_cas_url)
            .env("CARBON_SCRATCH_DIR", &scratch_dir)
            .env("CARBON_S3_ENDPOINT_URL", &origin)
            .env("CARBON_SECRETSMANAGER_ENDPOINT_URL", &origin)
            .env("CARBON_LAMBDA_ENDPOINT_URL", &origin)
            .env("AWS_ACCESS_KEY_ID", EMU_EVAL_ACCESS_KEY_ID)
            .env("AWS_SECRET_ACCESS_KEY", EMU_EVAL_SECRET_ACCESS_KEY)
            .env_remove("AWS_SESSION_TOKEN")
            .env_remove("AWS_PROFILE")
            .env_remove("CARBON_SECRETS_ARN")
            .env_remove("CARBON_SECRETS_VERSION_ID")
            .env_remove("CARBON_SEAL_SECRET_HEX")
            .env_remove("CARBON_ENGINE_KEK_HEX")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                spawn_err(
                    format!(
                        "spawn real carbon-worker eval #{idx} {}: {e}",
                        bins.worker_binary_path.display()
                    ),
                    &mut children,
                )
            })?;
        children.push(child);
    }

    Ok(children)
}

/// Handle `POST /` when `x-amz-target` targets AWS Secrets Manager (`secretsmanager.GetSecretValue`).
pub(crate) async fn handle_secretsmanager_rpc(
    state: Arc<EmulatorSharedState>,
    req: Request<Body>,
) -> Response<Body> {
    let raw_query = req.uri().query().map(str::to_owned);
    let headers = req.headers().clone();
    let Ok(collected) = Limited::new(req.into_body(), 256 * 1024).collect().await else {
        return plain_status(StatusCode::BAD_REQUEST, "SecretsManager body too large");
    };
    let body = collected.to_bytes();
    let caller_role = match verify_sigv4_header_request(
        &state.aws_vault.region,
        "secretsmanager",
        "POST",
        "/",
        raw_query.as_deref(),
        &headers,
        &body,
    ) {
        Ok(r) => r,
        Err((st, msg)) => return plain_status(st, msg),
    };

    let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return plain_status(StatusCode::BAD_REQUEST, "Invalid SecretsManager JSON");
    };
    let secret_id = parsed
        .get("SecretId")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let version_id = parsed.get("VersionId").and_then(|v| v.as_str());

    // Enforce strict per-role IAM policy (`DEP-SEC-01`, `RT-07`).
    match caller_role {
        EmuIamRole::Eval | EmuIamRole::Client => plain_status(
            StatusCode::FORBIDDEN,
            "AccessDeniedException: role is not authorized for secretsmanager:GetSecretValue",
        ),
        EmuIamRole::Edge => {
            if secret_id != state.aws_vault.edge_secret_arn {
                return plain_status(
                    StatusCode::FORBIDDEN,
                    "AccessDeniedException: carbon-edge IAM role cannot read cell/engine secrets",
                );
            }
            if version_id.is_some_and(|v| v != EMU_EDGE_SECRETS_VERSION) {
                return plain_status(
                    StatusCode::BAD_REQUEST,
                    "ResourceNotFoundException: VersionId mismatch for edge secret",
                );
            }
            let out = serde_json::json!({
                "ARN": state.aws_vault.edge_secret_arn,
                "VersionId": EMU_EDGE_SECRETS_VERSION,
                "SecretString": state.aws_vault.edge_secret_json,
            });
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/x-amz-json-1.1")
                .body(Body::from(out.to_string()))
                .unwrap_or_else(|_| Response::new(Body::empty()))
        }
        EmuIamRole::Engine => {
            if secret_id != state.aws_vault.cell_secret_arn {
                return plain_status(
                    StatusCode::FORBIDDEN,
                    "AccessDeniedException: carbon-engine IAM role restricted to cell-secrets ARN",
                );
            }
            if version_id.is_some_and(|v| v != EMU_CELL_SECRETS_VERSION) {
                return plain_status(
                    StatusCode::BAD_REQUEST,
                    "ResourceNotFoundException: VersionId mismatch for cell secret",
                );
            }
            let out = serde_json::json!({
                "ARN": state.aws_vault.cell_secret_arn,
                "VersionId": EMU_CELL_SECRETS_VERSION,
                "SecretString": state.aws_vault.cell_secret_json,
            });
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/x-amz-json-1.1")
                .body(Body::from(out.to_string()))
                .unwrap_or_else(|_| Response::new(Body::empty()))
        }
    }
}

/// Handle AWS Lambda `POST /2015-03-31/functions/{fn_name}/invocations` with SigV4 + IAM role checks.
pub(crate) async fn aws_lambda_invoke_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(fn_name): AxumPath<String>,
    req: Request<Body>,
) -> Response<Body> {
    let uri_path = req.uri().path().to_owned();
    let raw_query = req.uri().query().map(str::to_owned);
    let headers = req.headers().clone();
    let is_event = headers
        .get("x-amz-invocation-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|s| s.eq_ignore_ascii_case("Event"));
    let Ok(collected) = Limited::new(req.into_body(), MAX_EMU_AWS_BODY_BYTES)
        .collect()
        .await
    else {
        return plain_status(StatusCode::PAYLOAD_TOO_LARGE, "Lambda payload too large");
    };
    let payload = Vec::from(collected.to_bytes());

    let caller_role = match verify_sigv4_header_request(
        &state.aws_vault.region,
        "lambda",
        "POST",
        &uri_path,
        raw_query.as_deref(),
        &headers,
        &payload,
    ) {
        Ok(r) => r,
        Err((st, msg)) => return plain_status(st, msg),
    };

    let clean_fn = fn_name.split(':').next().unwrap_or(fn_name.as_str());
    let target_role = match (caller_role, clean_fn) {
        (EmuIamRole::Edge, EMU_ENGINE_FUNCTION_NAME) => "engine",
        (EmuIamRole::Engine, EMU_EVAL_FUNCTION_NAME) => "eval",
        (EmuIamRole::Client, EMU_EDGE_FUNCTION_NAME) => "edge",
        _ => {
            return plain_status(
                StatusCode::FORBIDDEN,
                format!(
                    "AccessDeniedException: IAM role {:?} is not permitted to lambda:InvokeFunction on '{clean_fn}'",
                    caller_role
                ),
            );
        }
    };

    let Some(queue) = state.role_queue(target_role) else {
        return plain_status(StatusCode::NOT_FOUND, "Function not found");
    };
    let lease_ms = if target_role == "edge" {
        30_000
    } else {
        state.engine_lease_ms
    };
    if is_event {
        if let Err(err) = queue.enqueue_async(payload, lease_ms).await {
            return plain_status(StatusCode::TOO_MANY_REQUESTS, err);
        }
        return plain_status(StatusCode::ACCEPTED, Body::empty());
    }
    match queue.dispatch(payload, lease_ms, 120_000).await {
        Ok(resp_bytes) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(resp_bytes))
            .unwrap_or_else(|_| Response::new(Body::empty())),
        Err(err) if err.starts_with("Lambda invocation ") || err.starts_with("HTTP 429") => {
            plain_status(StatusCode::TOO_MANY_REQUESTS, err)
        }
        Err(worker_err) => {
            // Real AWS Lambda returns HTTP 200 with `X-Amz-Function-Error: Unhandled` when the
            // runtime posts to `/runtime/invocation/{id}/error`.
            let err_json = serde_json::json!({
                "errorType": "RuntimeError",
                "errorMessage": worker_err,
            });
            Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .header("x-amz-function-error", "Unhandled")
                .body(Body::from(err_json.to_string()))
                .unwrap_or_else(|_| Response::new(Body::empty()))
        }
    }
}

/// Handle path-style AWS S3 CAS requests (`/{bucket}/{*key}`) for both SigV4 header (`CarbonS3Cas`)
/// and size/SHA-pinned SigV4 presigned query URLs (`CarbonClient` direct upload/download).
pub(crate) async fn aws_s3_bucket_handler(
    State(state): State<Arc<EmulatorSharedState>>,
    AxumPath(object_key): AxumPath<String>,
    req: Request<Body>,
) -> Response<Body> {
    let method = req.method().as_str().to_ascii_uppercase();
    let uri_path = req.uri().path().to_owned();
    let raw_query = req.uri().query().map(str::to_owned);
    let headers = req.headers().clone();
    let Ok(collected) = Limited::new(req.into_body(), MAX_EMU_AWS_BODY_BYTES)
        .collect()
        .await
    else {
        return plain_status(StatusCode::BAD_REQUEST, "S3 body exceeds stream ceiling");
    };
    let body_bytes = Vec::from(collected.to_bytes());

    let is_presigned = raw_query
        .as_deref()
        .is_some_and(|q| q.contains("X-Amz-Algorithm="));
    let caller_role = if is_presigned {
        match verify_s3_presigned_query(
            &state.aws_vault.region,
            &method,
            &uri_path,
            raw_query.as_deref().unwrap_or(""),
            &headers,
            &body_bytes,
        ) {
            Ok(r) => r,
            Err((st, msg)) => return plain_status(st, msg),
        }
    } else {
        match verify_sigv4_header_request(
            &state.aws_vault.region,
            "s3",
            &method,
            &uri_path,
            raw_query.as_deref(),
            &headers,
            &body_bytes,
        ) {
            Ok(r) => r,
            Err((st, msg)) => return plain_status(st, msg),
        }
    };

    let clean_key = object_key.trim_start_matches('/');
    let Some(rel_v1) = clean_key.strip_prefix("v1/") else {
        return plain_status(StatusCode::NOT_FOUND, "NoSuchKey");
    };

    if let Some(blob_rel) = rel_v1.strip_prefix("blobs/") {
        if caller_role == EmuIamRole::Eval && method != "GET" {
            return plain_status(
                StatusCode::FORBIDDEN,
                "AccessDenied: carbon-eval IAM role has read-only S3 blobs access",
            );
        }
        let segs: Vec<&str> = blob_rel.split('/').filter(|s| !s.is_empty()).collect();
        let (scope_opt, sha_str) = match segs.as_slice() {
            [sha] => (None, (*sha).to_owned()),
            [tenant, upload, sha] => (Some(format!("{tenant}/{upload}")), (*sha).to_owned()),
            _ => return plain_status(StatusCode::BAD_REQUEST, "Invalid S3 blob key layout"),
        };
        let cas = state.cas.clone();
        return match method.as_str() {
            "PUT" => {
                let res = tokio::task::spawn_blocking(move || {
                    cas.put_scoped_blob_verified(scope_opt.as_deref(), &sha_str, &body_bytes)
                })
                .await
                .map_err(err_to_string)
                .and_then(|r| r);
                match res {
                    Ok(()) => plain_status(StatusCode::OK, Body::empty()),
                    Err(err) => plain_status(StatusCode::BAD_REQUEST, err),
                }
            }
            "GET" => {
                let res = tokio::task::spawn_blocking(move || {
                    cas.get_scoped_blob_verified(scope_opt.as_deref(), &sha_str)
                })
                .await
                .map_err(err_to_string)
                .and_then(|r| r);
                match res {
                    Ok(bytes) => plain_status(StatusCode::OK, bytes),
                    Err(err) => plain_status(StatusCode::NOT_FOUND, err),
                }
            }
            "DELETE" => {
                let res = tokio::task::spawn_blocking(move || {
                    cas.delete_scoped_blob(scope_opt.as_deref(), &sha_str)
                })
                .await
                .map_err(err_to_string)
                .and_then(|r| r);
                match res {
                    Ok(true) => plain_status(StatusCode::NO_CONTENT, Body::empty()),
                    Ok(false) => plain_status(StatusCode::NOT_FOUND, Body::empty()),
                    Err(err) => plain_status(StatusCode::BAD_REQUEST, err),
                }
            }
            _ => plain_status(StatusCode::METHOD_NOT_ALLOWED, Body::empty()),
        };
    }

    if let Some(state_rel) = rel_v1.strip_prefix("state/") {
        if caller_role == EmuIamRole::Eval || is_presigned {
            return plain_status(
                StatusCode::FORBIDDEN,
                "AccessDenied: state prefix prohibited for caller",
            );
        }
        let state_key = state_rel.to_owned();
        let if_none_match = headers
            .get("if-none-match")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.trim() == "*");
        let cas = state.cas.clone();
        return match method.as_str() {
            "PUT" if if_none_match => {
                let res = tokio::task::spawn_blocking(move || {
                    cas.put_state_object_if_absent(&state_key, &body_bytes)
                })
                .await
                .map_err(err_to_string)
                .and_then(|r| r);
                match res {
                    Ok(true) => plain_status(StatusCode::OK, Body::empty()),
                    // Real AWS S3 returns `412 Precondition Failed` on `If-None-Match: *` collision!
                    Ok(false) => {
                        plain_status(StatusCode::PRECONDITION_FAILED, "PreconditionFailed")
                    }
                    Err(err) => plain_status(StatusCode::BAD_REQUEST, err),
                }
            }
            "PUT" => {
                let res = tokio::task::spawn_blocking(move || {
                    cas.put_state_object(&state_key, &body_bytes)
                })
                .await
                .map_err(err_to_string)
                .and_then(|r| r);
                match res {
                    Ok(()) => plain_status(StatusCode::OK, Body::empty()),
                    Err(err) => plain_status(StatusCode::BAD_REQUEST, err),
                }
            }
            "GET" => {
                let res = tokio::task::spawn_blocking(move || cas.get_state_object(&state_key))
                    .await
                    .map_err(err_to_string)
                    .and_then(|r| r);
                match res {
                    Ok(Some(bytes)) => plain_status(StatusCode::OK, bytes),
                    Ok(None) => plain_status(StatusCode::NOT_FOUND, "NoSuchKey"),
                    Err(err) => plain_status(StatusCode::BAD_REQUEST, err),
                }
            }
            "DELETE" => {
                let res = tokio::task::spawn_blocking(move || cas.delete_state_object(&state_key))
                    .await
                    .map_err(err_to_string)
                    .and_then(|r| r);
                match res {
                    Ok(()) => plain_status(StatusCode::NO_CONTENT, Body::empty()),
                    Err(err) => plain_status(StatusCode::BAD_REQUEST, err),
                }
            }
            _ => plain_status(StatusCode::METHOD_NOT_ALLOWED, Body::empty()),
        };
    }

    plain_status(StatusCode::NOT_FOUND, "NoSuchKey")
}

/// Encode a `CarbonGatewayRequest` as an AWS API Gateway HTTP API v2.0 JSON event (`apiId` set,
/// non-`.lambda-url.` `domainName`) so `carbon-edge` executes its exact production event parser.
pub(crate) fn encode_apigw_v2_event(
    gw_req: &CarbonGatewayRequest,
    peer_ip: IpAddr,
    region: &str,
) -> Result<Vec<u8>, String> {
    let raw_query = gw_req
        .query
        .iter()
        .map(|(k, v)| format!("{}={}", url_encode_component(k), url_encode_component(v)))
        .collect::<Vec<_>>()
        .join("&");
    let mut hdr_map = serde_json::Map::new();
    for (k, v) in &gw_req.headers {
        if k == "x-carbon-trusted-source-ip" || k == "x-carbon-raw-path" {
            continue;
        }
        hdr_map.insert(k.clone(), serde_json::Value::String(v.clone()));
    }
    let body_bytes = gw_req.body_cow()?;
    let event = if body_bytes.is_empty() {
        serde_json::json!({
            "version": "2.0",
            "routeKey": "$default",
            "rawPath": gw_req.path,
            "rawQueryString": raw_query,
            "headers": hdr_map,
            "requestContext": {
                "apiId": "emu-apigw-v2",
                "domainName": format!("emu-apigw-v2.execute-api.{region}.amazonaws.com"),
                "http": {
                    "method": gw_req.method,
                    "path": gw_req.path,
                    "sourceIp": peer_ip.to_string(),
                }
            },
            "isBase64Encoded": false,
        })
    } else {
        serde_json::json!({
            "version": "2.0",
            "routeKey": "$default",
            "rawPath": gw_req.path,
            "rawQueryString": raw_query,
            "headers": hdr_map,
            "requestContext": {
                "apiId": "emu-apigw-v2",
                "domainName": format!("emu-apigw-v2.execute-api.{region}.amazonaws.com"),
                "http": {
                    "method": gw_req.method,
                    "path": gw_req.path,
                    "sourceIp": peer_ip.to_string(),
                }
            },
            "body": base64::engine::general_purpose::STANDARD.encode(&body_bytes),
            "isBase64Encoded": true,
        })
    };
    to_json_vec(&event)
}

/// Decode either a Lambda Function URL / API GW v2 JSON response (`statusCode`) or an internal
/// `CarbonGatewayResponse` (`status`) into a `CarbonGatewayResponse`.
pub(crate) fn decode_edge_lambda_response(bytes: &[u8]) -> Result<CarbonGatewayResponse, String> {
    let val: serde_json::Value = serde_json::from_slice(bytes).map_err(err_to_string)?;
    if let Some(code) = val.get("statusCode").and_then(|v| v.as_u64()) {
        let status = u16::try_from(code).unwrap_or(500);
        let mut headers = BTreeMap::new();
        let mut content_type = "application/json".to_owned();
        if let Some(h_obj) = val.get("headers").and_then(|v| v.as_object()) {
            for (k, hv) in h_obj {
                if let Some(s) = hv.as_str() {
                    if k.eq_ignore_ascii_case("content-type") {
                        content_type = s.to_owned();
                    } else {
                        headers.insert(k.to_ascii_lowercase(), s.to_owned());
                    }
                }
            }
        }
        let is_b64 = val
            .get("isBase64Encoded")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let raw_body = match val.get("body").and_then(|v| v.as_str()) {
            Some(s) if is_b64 => base64::engine::general_purpose::STANDARD
                .decode(s)
                .map_err(err_to_string)?,
            Some(s) => s.as_bytes().to_vec(),
            None => Vec::new(),
        };
        let mut resp = CarbonGatewayResponse::raw_bytes_ok(status, &content_type, raw_body);
        resp.headers = headers;
        return Ok(resp);
    }
    serde_json::from_value(val).map_err(err_to_string)
}
