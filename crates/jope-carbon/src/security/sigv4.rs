//! AWS SigV4 (`AWS4-HMAC-SHA256`) S3 presigning, S3/Lambda HTTP request signing, and gateway
//! HMAC upload-token verification via `aws-lc-rs`.
//!
//! Hardened per `RT-04` (presigned S3 `PUT` URLs cryptographically pin both exact `content-length`
//! and `x-amz-content-sha256`) and `RT-14` (dynamic AWS CLI discovery with zero hardcoded host paths).

#[cfg(feature = "edge")]
use std::path::PathBuf;

#[cfg(feature = "edge")]
use aws_lc_rs::hmac;
#[cfg(feature = "edge")]
use sha2::{Digest, Sha256};
#[cfg(feature = "edge")]
use zeroize::Zeroizing;

#[cfg(feature = "edge")]
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let signing_key = hmac::Key::new(hmac::HMAC_SHA256, key);
    let tag = hmac::sign(&signing_key, data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&tag.as_ref()[..32]);
    out
}

/// `CarbonAwsCredentials` is a struct holding ephemeral or static AWS SigV4 signing credentials
/// read from the process environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`).
#[cfg(feature = "edge")]
#[derive(Clone)]
pub struct CarbonAwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: Zeroizing<String>,
    pub session_token: Option<Zeroizing<String>>,
}

#[cfg(feature = "edge")]
impl CarbonAwsCredentials {
    /// Load AWS credentials from standard Lambda / CLI environment variables (`AWS_ACCESS_KEY_ID`,
    /// `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`), falling back on CLI hosts to a cached
    /// `aws configure export-credentials --format process` lookup when `JOPE_AWS_PROFILE` or
    /// `AWS_PROFILE` is set.
    pub fn from_env() -> Result<Self, String> {
        if let (Ok(ak), Ok(sk)) = (
            std::env::var("AWS_ACCESS_KEY_ID"),
            std::env::var("AWS_SECRET_ACCESS_KEY"),
        ) {
            if !ak.trim().is_empty() && !sk.trim().is_empty() {
                let session_token = std::env::var("AWS_SESSION_TOKEN")
                    .ok()
                    .filter(|v| !v.trim().is_empty())
                    .map(Zeroizing::new);
                return Ok(Self {
                    access_key_id: ak,
                    secret_access_key: Zeroizing::new(sk),
                    session_token,
                });
            }
        }
        let profile = std::env::var("JOPE_AWS_PROFILE")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| {
                std::env::var("AWS_PROFILE")
                    .ok()
                    .filter(|v| !v.trim().is_empty())
            })
            .ok_or_else(|| {
                "missing AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY and AWS_PROFILE in environment"
                    .to_owned()
            })?;
        resolve_profile_credentials_cached(&profile)
    }
}

#[cfg(feature = "edge")]
pub(crate) fn resolve_aws_cli_binary() -> PathBuf {
    if let Some(explicit) = std::env::var_os("CARBON_AWS_CLI_BIN") {
        let p = PathBuf::from(explicit);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    if let Some(candidate) = crate::home_path::absolute_home_path(
        std::env::var_os("HOME").as_deref(),
        &[".local", "bin", "aws"],
    ) {
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("aws")
}

/// Split a `function_name[:qualifier]` target (such as `jopedime-carbon-dev-engine:live`) into
/// `(function_name, Some(qualifier))` so AWS Lambda `POST .../invocations?Qualifier=...` requests
/// sign `Qualifier` in the canonical SigV4 query string.
#[cfg(feature = "edge")]
pub fn split_lambda_name_and_qualifier(raw: &str) -> (&str, Option<&str>) {
    let trimmed = raw.trim();
    if !trimmed.starts_with("arn:") {
        if let Some((fn_name, qualifier)) = trimmed.split_once(':') {
            let fn_name = fn_name.trim();
            let qualifier = qualifier.trim();
            if !fn_name.is_empty() && !qualifier.is_empty() {
                return (fn_name, Some(qualifier));
            }
        }
    }
    (trimmed, None)
}

#[cfg(feature = "edge")]
fn resolve_profile_credentials_cached(profile: &str) -> Result<CarbonAwsCredentials, String> {
    use std::sync::{Mutex, OnceLock};
    static PROFILE_CACHE: OnceLock<
        Mutex<std::collections::BTreeMap<String, CarbonAwsCredentials>>,
    > = OnceLock::new();
    let cache = PROFILE_CACHE.get_or_init(|| Mutex::new(std::collections::BTreeMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(existing) = guard.get(profile) {
            return Ok(existing.clone());
        }
    }
    let aws_bin = resolve_aws_cli_binary();
    let output = std::process::Command::new(&aws_bin)
        .args(["configure", "export-credentials", "--format", "process"])
        .env("AWS_PROFILE", profile)
        .output()
        .map_err(|e| format!("failed to run aws configure export-credentials: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "aws configure export-credentials failed for profile '{profile}' ({})",
            output.status
        ));
    }
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("invalid export-credentials JSON for profile '{profile}': {e}"))?;
    let ak = parsed
        .get("AccessKeyId")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing AccessKeyId for profile '{profile}'"))?
        .to_owned();
    let sk = parsed
        .get("SecretAccessKey")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing SecretAccessKey for profile '{profile}'"))?
        .to_owned();
    let token = parsed
        .get("SessionToken")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(|s| Zeroizing::new(s.to_owned()));
    let creds = CarbonAwsCredentials {
        access_key_id: ak,
        secret_access_key: Zeroizing::new(sk),
        session_token: token,
    };
    if let Ok(mut guard) = cache.lock() {
        guard.insert(profile.to_owned(), creds.clone());
    }
    Ok(creds)
}

/// Convert UNIX `epoch_secs` to `(YYYYMMDD, YYYYMMDDTHHMMSSZ)` in UTC using Howard Hinnant's
/// exact civil-from-days algorithm.
#[cfg(feature = "edge")]
pub fn format_amz_timestamp_utc(epoch_secs: u64) -> (String, String) {
    let days = (epoch_secs / 86_400) as i64;
    let secs_of_day = epoch_secs % 86_400;
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    if m <= 2 {
        y += 1;
    }
    let date_stamp = format!("{y:04}{m:02}{d:02}");
    let amz_date = format!("{date_stamp}T{hour:02}{minute:02}{second:02}Z");
    (date_stamp, amz_date)
}

#[cfg(feature = "edge")]
fn derive_sigv4_signing_key(
    secret_access_key: &str,
    date_stamp: &str,
    region: &str,
    service: &str,
) -> [u8; 32] {
    let k_secret = Zeroizing::new(format!("AWS4{secret_access_key}").into_bytes());
    let k_date = hmac_sha256(&k_secret, date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// Sign an outbound HTTP request (`s3` or `lambda`) with AWS SigV4 (`AWS4-HMAC-SHA256`),
/// including any `extra_headers` (such as `if-none-match` or `x-amz-invocation-type`).
#[cfg(feature = "edge")]
#[allow(clippy::too_many_arguments)]
pub fn sign_aws_v4_headers_with_extra(
    creds: &CarbonAwsCredentials,
    region: &str,
    service: &str,
    method: &str,
    host: &str,
    canonical_uri: &str,
    canonical_query: &str,
    payload: &[u8],
    now_epoch_secs: u64,
    extra_headers: &[(&str, &str)],
) -> Vec<(String, String)> {
    let (date_stamp, amz_date) = format_amz_timestamp_utc(now_epoch_secs);
    let payload_hash = format!("{:x}", Sha256::digest(payload));

    let mut signed_headers: Vec<(String, String)> = vec![
        ("host".to_owned(), host.to_owned()),
        ("x-amz-content-sha256".to_owned(), payload_hash.clone()),
        ("x-amz-date".to_owned(), amz_date.clone()),
    ];
    for &(k, v) in extra_headers {
        signed_headers.push((k.to_ascii_lowercase(), v.to_owned()));
    }
    if let Some(tok) = &creds.session_token {
        signed_headers.push(("x-amz-security-token".to_owned(), tok.as_str().to_owned()));
    }
    signed_headers.sort_by(|a, b| a.0.cmp(&b.0));

    let mut canonical_headers_str = String::new();
    let mut signed_header_names = Vec::with_capacity(signed_headers.len());
    for (name, val) in &signed_headers {
        canonical_headers_str.push_str(name);
        canonical_headers_str.push(':');
        canonical_headers_str.push_str(val.trim());
        canonical_headers_str.push('\n');
        signed_header_names.push(name.as_str());
    }
    let signed_headers_list = signed_header_names.join(";");

    let canonical_request = format!(
        "{}\n{canonical_uri}\n{canonical_query}\n{canonical_headers_str}\n{signed_headers_list}\n{payload_hash}",
        method.to_ascii_uppercase()
    );
    let req_hash = format!("{:x}", Sha256::digest(canonical_request.as_bytes()));
    let credential_scope = format!("{date_stamp}/{region}/{service}/aws4_request");
    let string_to_sign = format!("AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{req_hash}");
    let k_signing =
        derive_sigv4_signing_key(&creds.secret_access_key, &date_stamp, region, service);
    let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));
    let auth_header = format!(
        "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, SignedHeaders={signed_headers_list}, Signature={signature}",
        creds.access_key_id
    );

    let mut out = vec![
        ("x-amz-date".to_owned(), amz_date),
        ("x-amz-content-sha256".to_owned(), payload_hash),
        ("authorization".to_owned(), auth_header),
    ];
    for &(k, v) in extra_headers {
        out.push((k.to_ascii_lowercase(), v.to_owned()));
    }
    if let Some(tok) = &creds.session_token {
        out.push(("x-amz-security-token".to_owned(), tok.as_str().to_owned()));
    }
    out
}

/// Sign an outbound HTTP request (`s3` or `lambda`) with AWS SigV4 (`AWS4-HMAC-SHA256`).
#[cfg(feature = "edge")]
#[allow(clippy::too_many_arguments)]
pub fn sign_aws_v4_headers(
    creds: &CarbonAwsCredentials,
    region: &str,
    service: &str,
    method: &str,
    host: &str,
    canonical_uri: &str,
    canonical_query: &str,
    payload: &[u8],
    now_epoch_secs: u64,
) -> Vec<(String, String)> {
    sign_aws_v4_headers_with_extra(
        creds,
        region,
        service,
        method,
        host,
        canonical_uri,
        canonical_query,
        payload,
        now_epoch_secs,
        &[],
    )
}

/// `CarbonPresigner` is a struct signing and verifying time-bounded, size- and SHA-256-pinned
/// upload URLs for both AWS S3 (`AWS4-HMAC-SHA256`) and self-hosted Enterprise / emulator gateways.
#[cfg(feature = "edge")]
#[derive(Clone)]
pub struct CarbonPresigner {
    secret: Zeroizing<[u8; 32]>,
}

#[cfg(feature = "edge")]
impl CarbonPresigner {
    pub fn from_secret(secret: [u8; 32]) -> Self {
        Self {
            secret: Zeroizing::new(secret),
        }
    }

    pub fn sign_gateway_upload_url(
        &self,
        gateway_base_url: &str,
        sha256_hex: &str,
        byte_len: u64,
        expires_epoch_secs: u64,
    ) -> String {
        let lower_sha = sha256_hex.to_ascii_lowercase();
        let canonical = format!("PUT\n{lower_sha}\n{byte_len}\n{expires_epoch_secs}");
        let sig = hex::encode(hmac_sha256(self.secret.as_ref(), canonical.as_bytes()));
        let base = gateway_base_url.trim_end_matches('/');
        format!(
            "{base}/v1/carbon/cas/{lower_sha}?bytes={byte_len}&exp={expires_epoch_secs}&sig={sig}"
        )
    }

    pub fn verify_gateway_upload(
        &self,
        sha256_hex: &str,
        declared_bytes: u64,
        expires_epoch_secs: u64,
        provided_sig_hex: &str,
        now_epoch_secs: u64,
    ) -> Result<(), &'static str> {
        if now_epoch_secs > expires_epoch_secs {
            return Err("presigned upload URL has expired");
        }
        let Ok(provided_bytes) = hex::decode(provided_sig_hex) else {
            return Err("malformed upload signature hex");
        };
        let lower_sha = sha256_hex.to_ascii_lowercase();
        let canonical = format!("PUT\n{lower_sha}\n{declared_bytes}\n{expires_epoch_secs}");
        let signing_key = hmac::Key::new(hmac::HMAC_SHA256, self.secret.as_ref());
        hmac::verify(&signing_key, canonical.as_bytes(), &provided_bytes)
            .map_err(|_| "presigned upload signature verification failed")
    }

    /// Size- and SHA-256-pinned AWS SigV4 S3 `PUT` presigner (`RT-04`): signs
    /// `content-length;host;x-amz-content-sha256` so S3 rejects any `PUT` whose byte length or
    /// SHA-256 payload digest differs from the edge-admitted values!
    #[allow(clippy::too_many_arguments)]
    pub fn presign_s3_put_v4_pinned_with_creds(
        creds: &CarbonAwsCredentials,
        region: &str,
        bucket: &str,
        object_key: &str,
        amz_date_iso8601: &str,
        expires_secs: u32,
        expected_sha256_hex: &str,
        exact_content_length: u64,
    ) -> Result<String, &'static str> {
        if amz_date_iso8601.len() < 8
            || expires_secs == 0
            || expires_secs > 604_800
            || expected_sha256_hex.len() != 64
            || !expected_sha256_hex.bytes().all(|b| b.is_ascii_hexdigit())
            || exact_content_length == 0
        {
            return Err("invalid SigV4 timestamp, expiration, SHA-256, or byte length");
        }
        let lower_sha = expected_sha256_hex.to_ascii_lowercase();
        let date_stamp = &amz_date_iso8601[..8];
        let host = format!("{bucket}.s3.{region}.amazonaws.com");
        let encoded_path = url_encode_s3_key(object_key);
        let credential_scope = format!("{date_stamp}/{region}/s3/aws4_request");
        let credential_raw = format!("{}/{credential_scope}", creds.access_key_id);
        let signed_hdrs = "content-length;host;x-amz-content-sha256";

        let mut query_pairs: Vec<(String, String)> = vec![
            ("X-Amz-Algorithm".to_owned(), "AWS4-HMAC-SHA256".to_owned()),
            (
                "X-Amz-Credential".to_owned(),
                url_encode_component(&credential_raw),
            ),
            (
                "X-Amz-Date".to_owned(),
                url_encode_component(amz_date_iso8601),
            ),
            ("X-Amz-Expires".to_owned(), expires_secs.to_string()),
            (
                "X-Amz-SignedHeaders".to_owned(),
                url_encode_component(signed_hdrs),
            ),
        ];
        if let Some(tok) = &creds.session_token {
            query_pairs.push((
                "X-Amz-Security-Token".to_owned(),
                url_encode_component(tok.as_str()),
            ));
        }
        query_pairs.sort_by(|a, b| a.0.cmp(&b.0));

        let canonical_query = query_pairs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");

        let canonical_request = format!(
            "PUT\n{encoded_path}\n{canonical_query}\ncontent-length:{exact_content_length}\nhost:{host}\nx-amz-content-sha256:{lower_sha}\n\n{signed_hdrs}\n{lower_sha}"
        );
        let req_hash = format!("{:x}", Sha256::digest(canonical_request.as_bytes()));
        let string_to_sign =
            format!("AWS4-HMAC-SHA256\n{amz_date_iso8601}\n{credential_scope}\n{req_hash}");

        let k_signing =
            derive_sigv4_signing_key(&creds.secret_access_key, date_stamp, region, "s3");
        let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

        Ok(format!(
            "https://{host}{encoded_path}?{canonical_query}&X-Amz-Signature={signature}"
        ))
    }

    pub fn presign_s3_put_v4_with_creds(
        creds: &CarbonAwsCredentials,
        region: &str,
        bucket: &str,
        object_key: &str,
        amz_date_iso8601: &str,
        expires_secs: u32,
        payload_hash_or_unsigned: &str,
    ) -> Result<String, &'static str> {
        if amz_date_iso8601.len() < 8 || expires_secs == 0 || expires_secs > 604_800 {
            return Err("invalid SigV4 timestamp or expiration window");
        }
        let date_stamp = &amz_date_iso8601[..8];
        let host = format!("{bucket}.s3.{region}.amazonaws.com");
        let encoded_path = url_encode_s3_key(object_key);
        let credential_scope = format!("{date_stamp}/{region}/s3/aws4_request");
        let credential_raw = format!("{}/{credential_scope}", creds.access_key_id);

        let mut query_pairs: Vec<(String, String)> = vec![
            ("X-Amz-Algorithm".to_owned(), "AWS4-HMAC-SHA256".to_owned()),
            (
                "X-Amz-Credential".to_owned(),
                url_encode_component(&credential_raw),
            ),
            (
                "X-Amz-Date".to_owned(),
                url_encode_component(amz_date_iso8601),
            ),
            ("X-Amz-Expires".to_owned(), expires_secs.to_string()),
            ("X-Amz-SignedHeaders".to_owned(), "host".to_owned()),
        ];
        if let Some(tok) = &creds.session_token {
            query_pairs.push((
                "X-Amz-Security-Token".to_owned(),
                url_encode_component(tok.as_str()),
            ));
        }
        query_pairs.sort_by(|a, b| a.0.cmp(&b.0));

        let canonical_query = query_pairs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");

        let canonical_request = format!(
            "PUT\n{encoded_path}\n{canonical_query}\nhost:{host}\n\nhost\n{payload_hash_or_unsigned}"
        );
        let req_hash = format!("{:x}", Sha256::digest(canonical_request.as_bytes()));
        let string_to_sign =
            format!("AWS4-HMAC-SHA256\n{amz_date_iso8601}\n{credential_scope}\n{req_hash}");

        let k_signing =
            derive_sigv4_signing_key(&creds.secret_access_key, date_stamp, region, "s3");
        let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

        Ok(format!(
            "https://{host}{encoded_path}?{canonical_query}&X-Amz-Signature={signature}"
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn presign_s3_put_v4(
        access_key_id: &str,
        secret_access_key: &str,
        region: &str,
        bucket: &str,
        object_key: &str,
        amz_date_iso8601: &str,
        expires_secs: u32,
        expected_sha256_hex: &str,
    ) -> Result<String, &'static str> {
        let creds = CarbonAwsCredentials {
            access_key_id: access_key_id.to_owned(),
            secret_access_key: Zeroizing::new(secret_access_key.to_owned()),
            session_token: None,
        };
        Self::presign_s3_put_v4_with_creds(
            &creds,
            region,
            bucket,
            object_key,
            amz_date_iso8601,
            expires_secs,
            &expected_sha256_hex.to_ascii_lowercase(),
        )
    }
}

/// RFC 3986 percent-encode an arbitrary query value or URI segment.
#[cfg(feature = "edge")]
pub fn url_encode_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Encode an S3 object key into a leading-slash canonical URI (`/prefix/blobs/...`).
#[cfg(feature = "edge")]
pub fn url_encode_s3_key(object_key: &str) -> String {
    let clean = object_key.trim_start_matches('/');
    if clean.is_empty() {
        return "/".to_owned();
    }
    let encoded_segments = clean
        .split('/')
        .map(url_encode_component)
        .collect::<Vec<_>>()
        .join("/");
    format!("/{encoded_segments}")
}
