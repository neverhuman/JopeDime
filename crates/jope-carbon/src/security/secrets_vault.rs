//! Runtime AWS Secrets Manager (`secretsmanager:GetSecretValue`) custody for Carbon cells (`GAP-SEC1`, `GAP-SEC5`).
//!
//! Eliminates plaintext private scalars (`CARBON_ENGINE_KEK_HEX`) and HMAC keys (`CARBON_SEAL_SECRET_HEX`,
//! `CARBON_AUTH_SECRET_HEX`, `CARBON_PRESIGN_SECRET_HEX`) from Lambda `FunctionConfiguration.Environment.Variables`
//! and `/proc/self/environ` by fetching `CARBON_SECRETS_ARN` once per container cold start over TLS 1.3 + SigV4
//! directly into `Zeroizing` memory, while preserving `[active, previous]` dual-key rotation and offline
//! environment-variable overrides for local unit/emulator harnesses.

use std::io::Read;
use std::sync::OnceLock;
use std::time::Duration;

use zeroize::Zeroizing;

use crate::{
    security::{
        auth::CarbonAuthVerifier,
        seal::CarbonSealKey,
        sigv4::{
            compute_backoff_with_jitter_ms, resolve_secretsmanager_endpoint,
            sign_aws_v4_headers_with_extra, CarbonAwsCredentials, CarbonPresigner,
        },
        user_envelope::{decode_p256_pubkey_b64, CarbonEngineKek},
    },
    storage::cas::now_epoch_secs,
    tls::carbon_https_agent,
};

/// Parsed in-memory cell secret document held strictly inside `Zeroizing` buffers.
#[derive(Clone)]
pub struct CarbonRuntimeSecrets {
    pub seal_secret_hex: Option<Zeroizing<String>>,
    pub previous_seal_secret_hex: Option<Zeroizing<String>>,
    pub presign_secret_hex: Option<Zeroizing<String>>,
    pub previous_presign_secret_hex: Option<Zeroizing<String>>,
    pub auth_secret_hex: Option<Zeroizing<String>>,
    pub previous_auth_secret_hex: Option<Zeroizing<String>>,
    pub engine_kek_hex: Option<Zeroizing<String>>,
    pub previous_engine_kek_hex: Option<Zeroizing<String>>,
    pub engine_kek_pubkey_b64: Option<String>,
}

impl std::fmt::Debug for CarbonRuntimeSecrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonRuntimeSecrets")
            .field("has_seal_secret", &self.seal_secret_hex.is_some())
            .field("has_presign_secret", &self.presign_secret_hex.is_some())
            .field("has_auth_secret", &self.auth_secret_hex.is_some())
            .field("has_engine_kek", &self.engine_kek_hex.is_some())
            .field("engine_kek_pubkey_b64", &self.engine_kek_pubkey_b64)
            .finish()
    }
}

fn decode_32byte_hex_named(label: &str, raw_hex: &str) -> Result<Zeroizing<[u8; 32]>, String> {
    let raw = Zeroizing::new(
        hex::decode(raw_hex.trim()).map_err(|_| format!("{label} must be 64-char hex"))?,
    );
    if raw.len() != 32 {
        return Err(format!("{label} must decode to exactly 32 bytes"));
    }
    let mut out = Zeroizing::new([0u8; 32]);
    out.copy_from_slice(&raw);
    Ok(out)
}

fn extract_nonempty_zeroized(
    obj: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<Zeroizing<String>> {
    for &k in keys {
        if let Some(s) = obj.get(k).and_then(|v| v.as_str()) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(Zeroizing::new(trimmed.to_owned()));
            }
        }
    }
    None
}

/// Parse a `SecretString` JSON object from AWS Secrets Manager into `CarbonRuntimeSecrets`.
/// When `scrub_engine_secrets` is `true` (e.g., `CARBON_ROLE=edge`), `seal_secret_hex` and
/// `engine_kek_hex` are never extracted into memory (`DEP-SEC-01` defense-in-depth).
pub fn parse_secret_bundle_json_for_role(
    secret_json: &str,
    scrub_engine_secrets: bool,
) -> Result<CarbonRuntimeSecrets, String> {
    let val: serde_json::Value = serde_json::from_str(secret_json)
        .map_err(|e| format!("invalid Secrets Manager JSON payload: {e}"))?;
    let obj = val
        .as_object()
        .ok_or_else(|| "Secrets Manager SecretString must be a JSON object".to_owned())?;

    let engine_kek_pubkey_b64 = obj
        .get("engine_kek_pubkey_b64")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);

    Ok(CarbonRuntimeSecrets {
        seal_secret_hex: if scrub_engine_secrets {
            None
        } else {
            extract_nonempty_zeroized(obj, &["seal_secret_hex"])
        },
        previous_seal_secret_hex: if scrub_engine_secrets {
            None
        } else {
            extract_nonempty_zeroized(obj, &["previous_seal_secret_hex"])
        },
        presign_secret_hex: extract_nonempty_zeroized(obj, &["presign_secret_hex"]),
        previous_presign_secret_hex: extract_nonempty_zeroized(
            obj,
            &["previous_presign_secret_hex"],
        ),
        auth_secret_hex: extract_nonempty_zeroized(obj, &["auth_secret_hex"]),
        previous_auth_secret_hex: extract_nonempty_zeroized(obj, &["previous_auth_secret_hex"]),
        engine_kek_hex: if scrub_engine_secrets {
            None
        } else {
            extract_nonempty_zeroized(obj, &["engine_kek_hex", "engine_kek_priv_hex"])
        },
        previous_engine_kek_hex: if scrub_engine_secrets {
            None
        } else {
            extract_nonempty_zeroized(obj, &["previous_engine_kek_hex"])
        },
        engine_kek_pubkey_b64,
    })
}

/// Parse a `SecretString` JSON object from AWS Secrets Manager into `CarbonRuntimeSecrets`,
/// automatically scrubbing engine-only keys when `CARBON_ROLE=edge`.
pub fn parse_secret_bundle_json(secret_json: &str) -> Result<CarbonRuntimeSecrets, String> {
    let is_edge = std::env::var("CARBON_ROLE")
        .map(|r| r.trim().eq_ignore_ascii_case("edge"))
        .unwrap_or(false);
    parse_secret_bundle_json_for_role(secret_json, is_edge)
}

fn infer_region_from_secret_arn(secret_id: &str) -> Option<String> {
    // arn:aws:secretsmanager:<region>:<account>:secret:<name>
    let parts: Vec<&str> = secret_id.trim().split(':').collect();
    if parts.len() >= 6 && parts[0] == "arn" && parts[2] == "secretsmanager" && !parts[3].is_empty()
    {
        Some(parts[3].to_owned())
    } else {
        None
    }
}

/// Fetch `SecretString` from AWS Secrets Manager (`secretsmanager.GetSecretValue`) via SigV4 over TLS.
pub fn fetch_secretsmanager_secret_string(
    default_region: &str,
    secret_id: &str,
    version_id: Option<&str>,
) -> Result<Zeroizing<String>, String> {
    let creds = CarbonAwsCredentials::from_env()?;
    let target_region = infer_region_from_secret_arn(secret_id).unwrap_or_else(|| {
        std::env::var("CARBON_SECRETS_REGION")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| default_region.to_owned())
    });
    let (host, url) = resolve_secretsmanager_endpoint(&target_region)?;
    let mut req_json = serde_json::Map::new();
    req_json.insert(
        "SecretId".to_owned(),
        serde_json::Value::String(secret_id.trim().to_owned()),
    );
    if let Some(vid) = version_id.map(str::trim).filter(|s| !s.is_empty()) {
        req_json.insert(
            "VersionId".to_owned(),
            serde_json::Value::String(vid.to_owned()),
        );
    }
    let payload = serde_json::to_vec(&serde_json::Value::Object(req_json))
        .map_err(|e| format!("encode GetSecretValue payload: {e}"))?;

    let agent = carbon_https_agent();
    let mut last_err = String::new();
    for attempt in 0..4usize {
        if attempt > 0 {
            let wait_ms = compute_backoff_with_jitter_ms(attempt - 1, 150, 2_500);
            std::thread::sleep(Duration::from_millis(wait_ms));
        }
        let headers = sign_aws_v4_headers_with_extra(
            &creds,
            &target_region,
            "secretsmanager",
            "POST",
            &host,
            "/",
            "",
            &payload,
            now_epoch_secs(),
            &[
                ("content-type", "application/x-amz-json-1.1"),
                ("x-amz-target", "secretsmanager.GetSecretValue"),
            ],
        );
        let mut req = agent
            .post(&url)
            .config()
            .http_status_as_error(false)
            .build();
        for (k, v) in &headers {
            req = req.header(k, v);
        }
        let mut resp = match req.send(&payload) {
            Ok(r) => r,
            Err(e) => {
                last_err = format!("SecretsManager GetSecretValue transport error: {e}");
                if attempt == 3 {
                    return Err(last_err);
                }
                continue;
            }
        };
        let status = resp.status().as_u16();
        let mut body = Zeroizing::new(Vec::new());
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("read GetSecretValue body: {e}"))?;
        if status == 200 {
            let parsed: serde_json::Value = serde_json::from_slice(&body)
                .map_err(|e| format!("decode GetSecretValue response JSON: {e}"))?;
            let secret_str = parsed
                .get("SecretString")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "SecretsManager response missing SecretString".to_owned())?;
            return Ok(Zeroizing::new(secret_str.to_owned()));
        }
        last_err = format!("SecretsManager GetSecretValue returned HTTP {status}");
        if !matches!(status, 408 | 429 | 500 | 502 | 503 | 504) || attempt == 3 {
            return Err(last_err);
        }
    }
    Err(last_err)
}

fn load_runtime_secrets_for_role(
    default_region: &str,
    scrub_engine_secrets: bool,
) -> Result<Option<CarbonRuntimeSecrets>, String> {
    static CACHED_EDGE: OnceLock<(String, CarbonRuntimeSecrets)> = OnceLock::new();
    static CACHED_FULL: OnceLock<(String, CarbonRuntimeSecrets)> = OnceLock::new();
    let Some(secret_arn) = std::env::var("CARBON_SECRETS_ARN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
    else {
        return Ok(None);
    };
    let version_id = std::env::var("CARBON_SECRETS_VERSION_ID")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let is_edge_env = std::env::var("CARBON_ROLE")
        .map(|r| r.trim().eq_ignore_ascii_case("edge"))
        .unwrap_or(false);
    let effective_scrub = scrub_engine_secrets || is_edge_env;
    let cache_key = format!(
        "{secret_arn}|{}|{effective_scrub}",
        version_id.as_deref().unwrap_or_default()
    );
    let slot = if effective_scrub {
        &CACHED_EDGE
    } else {
        &CACHED_FULL
    };
    if let Some((cached_key, existing)) = slot.get() {
        if *cached_key == cache_key {
            return Ok(Some(existing.clone()));
        }
    }
    let raw_secret =
        fetch_secretsmanager_secret_string(default_region, &secret_arn, version_id.as_deref())?;
    let parsed = parse_secret_bundle_json_for_role(&raw_secret, effective_scrub)?;
    let _ = slot.set((cache_key, parsed.clone()));
    Ok(Some(parsed))
}

/// Load `CARBON_SECRETS_ARN` once per container cold start (cached in `Zeroizing` RAM), or return
/// `Ok(None)` when `CARBON_SECRETS_ARN` is unset so local env-var tests and `RT-05` continue working.
pub fn load_cached_runtime_secrets(
    default_region: &str,
) -> Result<Option<&'static CarbonRuntimeSecrets>, String> {
    static CACHED_PUBLIC: OnceLock<CarbonRuntimeSecrets> = OnceLock::new();
    let Some(secret_arn) = std::env::var("CARBON_SECRETS_ARN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
    else {
        return Ok(None);
    };
    if let Some(existing) = CACHED_PUBLIC.get() {
        return Ok(Some(existing));
    }
    let version_id = std::env::var("CARBON_SECRETS_VERSION_ID")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let raw_secret =
        fetch_secretsmanager_secret_string(default_region, &secret_arn, version_id.as_deref())?;
    let parsed = parse_secret_bundle_json(&raw_secret)?;
    let _ = CACHED_PUBLIC.set(parsed);
    Ok(CACHED_PUBLIC.get())
}

/// Resolve `(CarbonPresigner, Option<CarbonAuthVerifier>, Option<String>)` for `carbon-edge` from
/// runtime Secrets Manager (`CARBON_SECRETS_ARN`) or environment variables (`RT-05` fail-closed on Lambda).
pub fn resolve_edge_secrets(
    region: &str,
    is_aws_production: bool,
) -> Result<(CarbonPresigner, Option<CarbonAuthVerifier>, Option<String>), String> {
    let vault = load_runtime_secrets_for_role(region, true)?;
    let vault_ref = vault.as_ref();

    let presign_hex_opt = std::env::var("CARBON_PRESIGN_SECRET_HEX")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(Zeroizing::new)
        .or_else(|| vault_ref.and_then(|v| v.presign_secret_hex.clone()));
    let prev_presign_opt = std::env::var("CARBON_PREVIOUS_PRESIGN_SECRET_HEX")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(Zeroizing::new)
        .or_else(|| vault_ref.and_then(|v| v.previous_presign_secret_hex.clone()));

    let presign_secret = match presign_hex_opt {
        Some(hex_val) => *decode_32byte_hex_named("CARBON_PRESIGN_SECRET_HEX", &hex_val)?,
        None if is_aws_production => {
            return Err("missing required CARBON_PRESIGN_SECRET_HEX".to_owned());
        }
        None => [0x33u8; 32],
    };
    let prev_presign = match prev_presign_opt {
        Some(hex_val) => Some(*decode_32byte_hex_named(
            "CARBON_PREVIOUS_PRESIGN_SECRET_HEX",
            &hex_val,
        )?),
        None => None,
    };
    let presigner = CarbonPresigner::from_secret(presign_secret).with_previous_secret(prev_presign);

    let auth_hex_opt = std::env::var("CARBON_AUTH_SECRET_HEX")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(Zeroizing::new)
        .or_else(|| vault_ref.and_then(|v| v.auth_secret_hex.clone()));
    let prev_auth_opt = std::env::var("CARBON_PREVIOUS_AUTH_SECRET_HEX")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(Zeroizing::new)
        .or_else(|| vault_ref.and_then(|v| v.previous_auth_secret_hex.clone()));

    let auth_verifier = match auth_hex_opt {
        Some(active_hex) => Some(
            CarbonAuthVerifier::from_hex_with_previous(
                &active_hex,
                prev_auth_opt.as_ref().map(|z| z.as_str()),
            )
            .map_err(|e| e.to_string())?,
        ),
        None if is_aws_production => {
            return Err("missing required CARBON_AUTH_SECRET_HEX on AWS Lambda".to_owned());
        }
        None => None,
    };

    let pubkey_opt = std::env::var("CARBON_ENGINE_KEK_PUBKEY_B64")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .or_else(|| vault_ref.and_then(|v| v.engine_kek_pubkey_b64.clone()));
    let engine_kek_pubkey_b64 = match pubkey_opt {
        Some(b64) => {
            let _ = decode_p256_pubkey_b64(&b64).map_err(|e| e.to_string())?;
            Some(b64)
        }
        None if is_aws_production => {
            return Err("missing required CARBON_ENGINE_KEK_PUBKEY_B64 on AWS Lambda".to_owned());
        }
        None => None,
    };

    Ok((presigner, auth_verifier, engine_kek_pubkey_b64))
}

/// Resolve `(CarbonSealKey, Option<CarbonEngineKek>)` for `carbon-engine` (`carbon-worker` / `carbon`)
/// from runtime Secrets Manager (`CARBON_SECRETS_ARN`) or environment variables (`RT-05` fail-closed).
pub fn resolve_engine_secrets(
    region: &str,
    is_aws_production: bool,
) -> Result<(CarbonSealKey, Option<CarbonEngineKek>), String> {
    let vault = load_runtime_secrets_for_role(region, false)?;
    let vault_ref = vault.as_ref();

    let seal_hex_opt = std::env::var("CARBON_SEAL_SECRET_HEX")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(Zeroizing::new)
        .or_else(|| vault_ref.and_then(|v| v.seal_secret_hex.clone()));
    let prev_seal_opt = std::env::var("CARBON_PREVIOUS_SEAL_SECRET_HEX")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(Zeroizing::new)
        .or_else(|| vault_ref.and_then(|v| v.previous_seal_secret_hex.clone()));

    let seal_bytes = match seal_hex_opt {
        Some(hex_val) => *decode_32byte_hex_named("CARBON_SEAL_SECRET_HEX", &hex_val)?,
        None if is_aws_production => {
            return Err("missing required CARBON_SEAL_SECRET_HEX on AWS Lambda".to_owned());
        }
        None => [0x44u8; 32],
    };
    let prev_seal_bytes = match prev_seal_opt {
        Some(hex_val) => Some(*decode_32byte_hex_named(
            "CARBON_PREVIOUS_SEAL_SECRET_HEX",
            &hex_val,
        )?),
        None => None,
    };
    let seal_key = CarbonSealKey::from_bytes(seal_bytes).with_previous_bytes(prev_seal_bytes);

    let kek_hex_opt = std::env::var("CARBON_ENGINE_KEK_HEX")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(Zeroizing::new)
        .or_else(|| vault_ref.and_then(|v| v.engine_kek_hex.clone()));
    let prev_kek_opt = std::env::var("CARBON_PREVIOUS_ENGINE_KEK_HEX")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(Zeroizing::new)
        .or_else(|| vault_ref.and_then(|v| v.previous_engine_kek_hex.clone()));

    let engine_kek = match kek_hex_opt {
        Some(active_hex) => Some(
            CarbonEngineKek::from_hex_with_previous(
                &active_hex,
                prev_kek_opt.as_ref().map(|z| z.as_str()),
            )
            .map_err(|e| e.to_string())?,
        ),
        None if is_aws_production => {
            return Err("missing required CARBON_ENGINE_KEK_HEX on AWS Lambda".to_owned());
        }
        None => None,
    };

    Ok((seal_key, engine_kek))
}
