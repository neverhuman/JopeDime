//! Content-addressed blob (`blobs/<sha256>`) and per-run state object storage (`ADR-0007`).
//!
//! Used uniformly across cloud Lambda cells (`CarbonHttpCas`), Enterprise VPC/on-premises
//! `carbon serve` (`CarbonFsCas`), and `carbon emu` local CI simulation.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// `CarbonCasStore` is a trait abstracting content-addressed `JMAT` blobs and run state keys.
pub trait CarbonCasStore: Send + Sync {
    /// Store `bytes` at `blobs/<expected_sha256>` after verifying `sha256(bytes) == expected_sha256`.
    fn put_blob_verified(&self, expected_sha256: &str, bytes: &[u8]) -> Result<(), String>;
    /// Read `blobs/<expected_sha256>` and verify `sha256(bytes) == expected_sha256` before returning.
    fn get_blob_verified(&self, expected_sha256: &str) -> Result<Vec<u8>, String>;
    /// Write a run state object (such as `runs/<tenant>/<run_id>/status.json` or `sealed_checkpoint.enc`).
    fn put_state_object(&self, key: &str, bytes: &[u8]) -> Result<(), String>;
    /// Read a run state object, returning `Ok(None)` if absent.
    fn get_state_object(&self, key: &str) -> Result<Option<Vec<u8>>, String>;
    /// Remove a run state object (such as clearing `stop.json` on resume).
    fn delete_state_object(&self, key: &str) -> Result<(), String>;
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

pub(crate) fn sha256_lower_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// `CarbonFsCas` is a struct storing verified CAS blobs and run state objects on a local directory
/// (backing Enterprise `carbon serve` and the `carbon emu` S3 CAS backend).
#[derive(Clone, Debug)]
pub struct CarbonFsCas {
    root: PathBuf,
}

impl CarbonFsCas {
    /// Initialize or open a `CarbonFsCas` rooted at `root`.
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

    fn put_state_object(&self, key: &str, bytes: &[u8]) -> Result<(), String> {
        validate_state_key(key)?;
        let dest = self.root.join("state").join(key);
        Self::atomic_write(&dest, bytes)
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

/// `CarbonHttpCas` is a struct implementing `CarbonCasStore` over HTTP/HTTPS for stateless
/// `edge`, `engine`, and `eval` Lambda cells talking to S3 or the `carbon emu` S3 CAS server.
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

    fn put_state_object(&self, key: &str, bytes: &[u8]) -> Result<(), String> {
        validate_state_key(key)?;
        let url = format!("{}/state/{key}", self.base_url);
        let resp = self
            .agent
            .put(&url)
            .header("x-carbon-internal-token", &self.internal_token)
            .send(bytes)
            .map_err(|e| format!("HTTP CAS put state failed: {e}"))?;
        if resp.status().as_u16() >= 300 {
            return Err(format!("HTTP CAS put state status {}", resp.status()));
        }
        Ok(())
    }

    fn get_state_object(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        validate_state_key(key)?;
        let url = format!("{}/state/{key}", self.base_url);
        match self
            .agent
            .get(&url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("x-carbon-internal-token", &self.internal_token)
            .call()
        {
            Ok(mut resp) => {
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
            Err(err) => Err(format!("HTTP CAS get state '{key}' failed: {err}")),
        }
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
