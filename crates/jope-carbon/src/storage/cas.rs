//! Content-addressed blob (`blobs/<sha256>`) and per-run state object storage (`ADR-0007`).
//!
//! Hardened per `RT-03` (`put_state_object_if_absent` conditional create using `If-None-Match: "*"`
//! / `create_new(true)` / `?if_none_match=1`), `RT-08` (injective strict `tenant_id` state keys),
//! and zero-knowledge cryptographic shred (`delete_blob`).

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "edge")]
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::security::auth::{validate_strict_run_id, validate_strict_tenant_id};

/// `CarbonCasStore` is a trait abstracting content-addressed `JMAT`/`CUEN` blobs and run state keys.
pub trait CarbonCasStore: Send + Sync {
    /// Store `bytes` at `blobs/<expected_sha256>` after verifying `sha256(bytes) == expected_sha256`.
    fn put_blob_verified(&self, expected_sha256: &str, bytes: &[u8]) -> Result<(), String>;
    /// Read `blobs/<expected_sha256>` and verify `sha256(bytes) == expected_sha256` before returning.
    fn get_blob_verified(&self, expected_sha256: &str) -> Result<Vec<u8>, String>;
    /// Delete `blobs/<expected_sha256>` during `DELETE /v1/carbon/runs/{id}` cryptographic shred.
    fn delete_blob(&self, expected_sha256: &str) -> Result<bool, String>;
    /// Write a run state object (such as `runs/<tenant>/<run_id>/status.json` or `sealed_checkpoint.enc`).
    fn put_state_object(&self, key: &str, bytes: &[u8]) -> Result<(), String>;
    /// Conditionally create a run state object ONLY if `key` does not already exist (`RT-03`).
    /// Returns `Ok(true)` if created, `Ok(false)` if `key` already exists.
    fn put_state_object_if_absent(&self, key: &str, bytes: &[u8]) -> Result<bool, String>;
    /// Read a run state object, returning `Ok(None)` if absent.
    fn get_state_object(&self, key: &str) -> Result<Option<Vec<u8>>, String>;
    /// Remove a run state object (such as clearing `stop.json` on resume or shredding run state).
    fn delete_state_object(&self, key: &str) -> Result<(), String>;
}

/// Construct an injective, collision-free per-tenant run state key (`RT-08`).
pub fn canonical_run_state_key(
    tenant_id: &str,
    run_id: &str,
    leaf: &str,
) -> Result<String, &'static str> {
    validate_strict_tenant_id(tenant_id).map_err(|_| "invalid tenant_id")?;
    validate_strict_run_id(run_id).map_err(|_| "invalid run_id")?;
    if leaf.is_empty() || leaf.contains('/') || leaf.contains("..") || leaf.contains('\\') {
        return Err("invalid state leaf");
    }
    Ok(format!("runs/{tenant_id}/{run_id}/{leaf}"))
}

fn validate_hex_sha256(hex_digest: &str) -> Result<(), String> {
    if hex_digest.len() != 64 || !hex_digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid 64-char hex SHA-256 digest".to_owned());
    }
    Ok(())
}

fn validate_state_key(key: &str) -> Result<(), String> {
    if key.is_empty() || key.starts_with('/') || key.contains("..") || key.contains('\\') {
        return Err(format!("unsafe CAS state key '{key}'"));
    }
    Ok(())
}

/// Compute lowercase 64-character SHA-256 hex digest of `bytes`.
pub fn sha256_lower_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(feature = "edge")]
pub(crate) fn now_epoch_secs() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs(),
        Err(_) => 0,
    }
}

/// Parse an `s3://<bucket>[/<prefix>]` URL into `(bucket, trimmed_prefix)`.
#[cfg(feature = "edge")]
pub fn parse_s3_cas_url(cas_url: &str) -> Option<(String, String)> {
    let rest = cas_url.trim().strip_prefix("s3://")?;
    let trimmed = rest.trim_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.split_once('/') {
        Some((bucket, prefix)) if !bucket.is_empty() => {
            Some((bucket.to_owned(), prefix.trim_matches('/').to_owned()))
        }
        None => Some((trimmed.to_owned(), String::new())),
        _ => None,
    }
}

/// Open either a live `CarbonS3Cas` (`s3://bucket/prefix`) or `CarbonHttpCas` (`http(s)://...`).
#[cfg(feature = "edge")]
pub fn open_remote_cas(
    cas_url: &str,
    default_region: &str,
    internal_token: &str,
) -> Result<Arc<dyn CarbonCasStore>, String> {
    if let Some((bucket, prefix)) = parse_s3_cas_url(cas_url) {
        let region = match std::env::var("AWS_REGION") {
            Ok(val) if !val.trim().is_empty() => val,
            _ => default_region.to_owned(),
        };
        return Ok(Arc::new(CarbonS3Cas::new(region, bucket, prefix)));
    }
    Ok(Arc::new(CarbonHttpCas::new(cas_url, internal_token)))
}

/// `CarbonFsCas` is a struct storing verified CAS blobs and run state objects on a local directory.
#[derive(Clone, Debug)]
pub struct CarbonFsCas {
    root: PathBuf,
}

impl CarbonFsCas {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, String> {
        let root = root.into();
        fs::create_dir_all(root.join("blobs")).map_err(|e| format!("create CAS blobs dir: {e}"))?;
        fs::create_dir_all(root.join("state")).map_err(|e| format!("create CAS state dir: {e}"))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn atomic_write(dest: &Path, bytes: &[u8]) -> Result<(), String> {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("create parent dir: {e}"))?;
        }
        let nonce = ulid::Ulid::new().to_string();
        let tmp = dest.with_extension(format!("tmp.{nonce}"));
        fs::write(&tmp, bytes).map_err(|e| format!("write temp object: {e}"))?;
        fs::rename(&tmp, dest).map_err(|e| format!("atomic rename object: {e}"))?;
        Ok(())
    }
}

impl CarbonCasStore for CarbonFsCas {
    fn put_blob_verified(&self, expected_sha256: &str, bytes: &[u8]) -> Result<(), String> {
        validate_hex_sha256(expected_sha256)?;
        let expected_lower = expected_sha256.to_ascii_lowercase();
        let actual = sha256_lower_hex(bytes);
        if actual != expected_lower {
            return Err(format!(
                "CAS blob SHA-256 mismatch: declared {expected_lower}, actual {actual}"
            ));
        }
        let path = self
            .root
            .join("blobs")
            .join(&expected_lower[..2])
            .join(&expected_lower);
        if path.is_file() {
            return Ok(());
        }
        Self::atomic_write(&path, bytes)
    }

    fn get_blob_verified(&self, expected_sha256: &str) -> Result<Vec<u8>, String> {
        validate_hex_sha256(expected_sha256)?;
        let expected_lower = expected_sha256.to_ascii_lowercase();
        let path = self
            .root
            .join("blobs")
            .join(&expected_lower[..2])
            .join(&expected_lower);
        let bytes =
            fs::read(&path).map_err(|e| format!("missing CAS blob {expected_lower}: {e}"))?;
        let actual = sha256_lower_hex(&bytes);
        if actual != expected_lower {
            return Err(format!(
                "corrupted CAS blob {expected_lower}: computed {actual}"
            ));
        }
        Ok(bytes)
    }

    fn delete_blob(&self, expected_sha256: &str) -> Result<bool, String> {
        validate_hex_sha256(expected_sha256)?;
        let expected_lower = expected_sha256.to_ascii_lowercase();
        let path = self
            .root
            .join("blobs")
            .join(&expected_lower[..2])
            .join(&expected_lower);
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(format!("delete CAS blob {expected_lower}: {err}")),
        }
    }

    fn put_state_object(&self, key: &str, bytes: &[u8]) -> Result<(), String> {
        validate_state_key(key)?;
        let dest = self.root.join("state").join(key);
        Self::atomic_write(&dest, bytes)
    }

    fn put_state_object_if_absent(&self, key: &str, bytes: &[u8]) -> Result<bool, String> {
        validate_state_key(key)?;
        let dest = self.root.join("state").join(key);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("create parent dir: {e}"))?;
        }
        match OpenOptions::new().write(true).create_new(true).open(&dest) {
            Ok(mut file) => {
                file.write_all(bytes)
                    .map_err(|e| format!("write new state object '{key}': {e}"))?;
                Ok(true)
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
            Err(err) => Err(format!("create_new state object '{key}': {err}")),
        }
    }

    fn get_state_object(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        validate_state_key(key)?;
        let dest = self.root.join("state").join(key);
        match fs::read(&dest) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(format!("read state object '{key}': {err}")),
        }
    }

    fn delete_state_object(&self, key: &str) -> Result<(), String> {
        validate_state_key(key)?;
        let dest = self.root.join("state").join(key);
        match fs::remove_file(&dest) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(format!("delete state object '{key}': {err}")),
        }
    }
}

/// `CarbonS3Cas` is a struct implementing `CarbonCasStore` directly over AWS S3 (`AWS4-HMAC-SHA256`).
#[cfg(feature = "edge")]
#[derive(Clone)]
pub struct CarbonS3Cas {
    region: String,
    bucket: String,
    prefix: String,
    agent: Arc<ureq::Agent>,
}

#[cfg(feature = "edge")]
impl CarbonS3Cas {
    pub fn new(
        region: impl Into<String>,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Self {
        Self {
            region: region.into(),
            bucket: bucket.into(),
            prefix: prefix.into().trim_matches('/').to_owned(),
            agent: Arc::new(crate::tls::carbon_https_agent().clone()),
        }
    }

    pub fn join_object_key(&self, rel: &str) -> String {
        let clean_rel = rel.trim_start_matches('/');
        if self.prefix.is_empty() {
            clean_rel.to_owned()
        } else {
            format!("{}/{clean_rel}", self.prefix)
        }
    }

    fn s3_request_with_extra(
        &self,
        method: &str,
        object_key: &str,
        payload: &[u8],
        extra_headers: &[(&str, &str)],
    ) -> Result<(u16, Vec<u8>), String> {
        let creds = crate::security::sigv4::CarbonAwsCredentials::from_env()?;
        let host = format!("{}.s3.{}.amazonaws.com", self.bucket, self.region);
        let canonical_uri = crate::security::sigv4::url_encode_s3_key(object_key);
        let url = format!("https://{host}{canonical_uri}");
        let headers = crate::security::sigv4::sign_aws_v4_headers_with_extra(
            &creds,
            &self.region,
            "s3",
            method,
            &host,
            &canonical_uri,
            "",
            payload,
            now_epoch_secs(),
            extra_headers,
        );

        let mut resp = match method {
            "PUT" => {
                let mut req = self
                    .agent
                    .put(&url)
                    .config()
                    .http_status_as_error(false)
                    .build();
                for (k, v) in &headers {
                    req = req.header(k, v);
                }
                req.send(payload)
                    .map_err(|e| format!("S3 PUT {object_key}: {e}"))?
            }
            "GET" => {
                let mut req = self
                    .agent
                    .get(&url)
                    .config()
                    .http_status_as_error(false)
                    .build();
                for (k, v) in &headers {
                    req = req.header(k, v);
                }
                req.call()
                    .map_err(|e| format!("S3 GET {object_key}: {e}"))?
            }
            "DELETE" => {
                let mut req = self
                    .agent
                    .delete(&url)
                    .config()
                    .http_status_as_error(false)
                    .build();
                for (k, v) in &headers {
                    req = req.header(k, v);
                }
                req.call()
                    .map_err(|e| format!("S3 DELETE {object_key}: {e}"))?
            }
            other => return Err(format!("unsupported S3 method '{other}'")),
        };

        let status = resp.status().as_u16();
        let mut body = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("read S3 {method} {object_key} body: {e}"))?;
        Ok((status, body))
    }
}

#[cfg(feature = "edge")]
impl CarbonCasStore for CarbonS3Cas {
    fn put_blob_verified(&self, expected_sha256: &str, bytes: &[u8]) -> Result<(), String> {
        validate_hex_sha256(expected_sha256)?;
        let expected_lower = expected_sha256.to_ascii_lowercase();
        let actual = sha256_lower_hex(bytes);
        if actual != expected_lower {
            return Err(format!(
                "S3 CAS put SHA-256 mismatch: declared {expected_lower}, actual {actual}"
            ));
        }
        let key = self.join_object_key(&format!("blobs/{expected_lower}"));
        let (status, body) = self.s3_request_with_extra("PUT", &key, bytes, &[])?;
        if status >= 300 {
            return Err(format!(
                "S3 CAS put blob HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        Ok(())
    }

    fn get_blob_verified(&self, expected_sha256: &str) -> Result<Vec<u8>, String> {
        validate_hex_sha256(expected_sha256)?;
        let expected_lower = expected_sha256.to_ascii_lowercase();
        let key = self.join_object_key(&format!("blobs/{expected_lower}"));
        let (status, body) = self.s3_request_with_extra("GET", &key, &[], &[])?;
        if status != 200 {
            return Err(format!(
                "S3 CAS get blob '{expected_lower}' HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        let actual = sha256_lower_hex(&body);
        if actual != expected_lower {
            return Err(format!(
                "S3 CAS blob integrity mismatch: expected {expected_lower}, got {actual}"
            ));
        }
        Ok(body)
    }

    fn delete_blob(&self, expected_sha256: &str) -> Result<bool, String> {
        validate_hex_sha256(expected_sha256)?;
        let expected_lower = expected_sha256.to_ascii_lowercase();
        let key = self.join_object_key(&format!("blobs/{expected_lower}"));
        let (status, body) = self.s3_request_with_extra("DELETE", &key, &[], &[])?;
        if status == 200 || status == 204 {
            Ok(true)
        } else if status == 404 {
            Ok(false)
        } else {
            Err(format!(
                "S3 CAS delete blob '{expected_lower}' HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ))
        }
    }

    fn put_state_object(&self, key: &str, bytes: &[u8]) -> Result<(), String> {
        validate_state_key(key)?;
        let obj_key = self.join_object_key(&format!("state/{key}"));
        let (status, body) = self.s3_request_with_extra("PUT", &obj_key, bytes, &[])?;
        if status >= 300 {
            return Err(format!(
                "S3 CAS put state '{key}' HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        Ok(())
    }

    fn put_state_object_if_absent(&self, key: &str, bytes: &[u8]) -> Result<bool, String> {
        validate_state_key(key)?;
        let obj_key = self.join_object_key(&format!("state/{key}"));
        let (status, body) =
            self.s3_request_with_extra("PUT", &obj_key, bytes, &[("if-none-match", "*")])?;
        if status == 412 || status == 409 {
            return Ok(false);
        }
        if status >= 300 {
            return Err(format!(
                "S3 CAS conditional put state '{key}' HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        Ok(true)
    }

    fn get_state_object(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        validate_state_key(key)?;
        let obj_key = self.join_object_key(&format!("state/{key}"));
        let (status, body) = self.s3_request_with_extra("GET", &obj_key, &[], &[])?;
        if status == 404 {
            return Ok(None);
        }
        if status != 200 {
            return Err(format!(
                "S3 CAS get state '{key}' HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        Ok(Some(body))
    }

    fn delete_state_object(&self, key: &str) -> Result<(), String> {
        validate_state_key(key)?;
        let obj_key = self.join_object_key(&format!("state/{key}"));
        let (status, body) = self.s3_request_with_extra("DELETE", &obj_key, &[], &[])?;
        if status == 200 || status == 204 || status == 404 {
            Ok(())
        } else {
            Err(format!(
                "S3 CAS delete state '{key}' HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ))
        }
    }
}

/// `CarbonHttpCas` is a struct implementing `CarbonCasStore` over HTTP/HTTPS for `carbon emu`.
#[cfg(feature = "edge")]
#[derive(Clone)]
pub struct CarbonHttpCas {
    base_url: String,
    agent: Arc<ureq::Agent>,
    internal_token: String,
}

#[cfg(feature = "edge")]
impl CarbonHttpCas {
    pub fn new(base_url: impl Into<String>, internal_token: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            agent: Arc::new(crate::tls::carbon_https_agent().clone()),
            internal_token: internal_token.into(),
        }
    }
}

#[cfg(feature = "edge")]
impl CarbonCasStore for CarbonHttpCas {
    fn put_blob_verified(&self, expected_sha256: &str, bytes: &[u8]) -> Result<(), String> {
        validate_hex_sha256(expected_sha256)?;
        let expected_lower = expected_sha256.to_ascii_lowercase();
        let actual = sha256_lower_hex(bytes);
        if actual != expected_lower {
            return Err(format!(
                "HTTP CAS put SHA-256 mismatch: declared {expected_lower}, actual {actual}"
            ));
        }
        let url = format!("{}/blobs/{expected_lower}", self.base_url);
        let resp = self
            .agent
            .put(&url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("x-carbon-internal-token", &self.internal_token)
            .send(bytes)
            .map_err(|e| format!("HTTP CAS put blob failed: {e}"))?;
        if resp.status().as_u16() >= 300 {
            return Err(format!(
                "HTTP CAS put blob returned status {}",
                resp.status()
            ));
        }
        Ok(())
    }

    fn get_blob_verified(&self, expected_sha256: &str) -> Result<Vec<u8>, String> {
        validate_hex_sha256(expected_sha256)?;
        let expected_lower = expected_sha256.to_ascii_lowercase();
        let url = format!("{}/blobs/{expected_lower}", self.base_url);
        let mut resp = self
            .agent
            .get(&url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("x-carbon-internal-token", &self.internal_token)
            .call()
            .map_err(|e| format!("HTTP CAS get blob failed: {e}"))?;
        if resp.status().as_u16() != 200 {
            return Err(format!("HTTP CAS get blob status {}", resp.status()));
        }
        let mut buf = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut buf)
            .map_err(|e| format!("read HTTP CAS blob body: {e}"))?;
        let actual = sha256_lower_hex(&buf);
        if actual != expected_lower {
            return Err(format!(
                "HTTP CAS blob integrity mismatch: expected {expected_lower}, got {actual}"
            ));
        }
        Ok(buf)
    }

    fn delete_blob(&self, expected_sha256: &str) -> Result<bool, String> {
        validate_hex_sha256(expected_sha256)?;
        let expected_lower = expected_sha256.to_ascii_lowercase();
        let url = format!("{}/blobs/{expected_lower}", self.base_url);
        let resp = self
            .agent
            .delete(&url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("x-carbon-internal-token", &self.internal_token)
            .call()
            .map_err(|e| format!("HTTP CAS delete blob failed: {e}"))?;
        let code = resp.status().as_u16();
        if code == 200 || code == 204 {
            Ok(true)
        } else if code == 404 {
            Ok(false)
        } else {
            Err(format!("HTTP CAS delete blob status {code}"))
        }
    }

    fn put_state_object(&self, key: &str, bytes: &[u8]) -> Result<(), String> {
        validate_state_key(key)?;
        let url = format!("{}/state/{key}", self.base_url);
        let resp = self
            .agent
            .put(&url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("x-carbon-internal-token", &self.internal_token)
            .send(bytes)
            .map_err(|e| format!("HTTP CAS put state failed: {e}"))?;
        if resp.status().as_u16() >= 300 {
            return Err(format!("HTTP CAS put state status {}", resp.status()));
        }
        Ok(())
    }

    fn put_state_object_if_absent(&self, key: &str, bytes: &[u8]) -> Result<bool, String> {
        validate_state_key(key)?;
        let url = format!("{}/state/{key}?if_none_match=1", self.base_url);
        let resp = self
            .agent
            .put(&url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("x-carbon-internal-token", &self.internal_token)
            .send(bytes)
            .map_err(|e| format!("HTTP CAS conditional put state failed: {e}"))?;
        let code = resp.status().as_u16();
        if code == 409 || code == 412 {
            return Ok(false);
        }
        if code >= 300 {
            return Err(format!(
                "HTTP CAS conditional put state '{key}' status {code}"
            ));
        }
        Ok(true)
    }

    fn get_state_object(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        validate_state_key(key)?;
        let url = format!("{}/state/{key}", self.base_url);
        let mut resp = self
            .agent
            .get(&url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("x-carbon-internal-token", &self.internal_token)
            .call()
            .map_err(|e| format!("HTTP CAS get state '{key}' failed: {e}"))?;
        let code = resp.status().as_u16();
        if code == 404 {
            return Ok(None);
        }
        if code != 200 {
            return Err(format!("HTTP CAS get state '{key}' status {code}"));
        }
        let mut buf = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut buf)
            .map_err(|e| format!("read HTTP CAS state body: {e}"))?;
        Ok(Some(buf))
    }

    fn delete_state_object(&self, key: &str) -> Result<(), String> {
        validate_state_key(key)?;
        let url = format!("{}/state/{key}", self.base_url);
        let resp = self
            .agent
            .delete(&url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("x-carbon-internal-token", &self.internal_token)
            .call()
            .map_err(|e| format!("HTTP CAS delete state '{key}' failed: {e}"))?;
        let code = resp.status().as_u16();
        if code == 200 || code == 204 || code == 404 {
            Ok(())
        } else {
            Err(format!("HTTP CAS delete state '{key}' status {code}"))
        }
    }
}
