//! Stateless `carbon eval` batch evaluation worker (`XS` `512 MB` .. `XL` `7,076 MB`, `ADR-0007`).
//!
//! Holds zero persistent cryptographic secrets (`RT-07`): decrypts `CUEN\x01(FitJmat)` strictly
//! in `Zeroizing` RAM using a 60-second ephemeral `FitJmat` slot subkey granted by `carbon-engine`,
//! bounds warm RAM caching to 2 entries with a 60-second TTL, and never accesses `holdout.jmat`.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use jope_prep::cuen::CuenPayloadKind;
use jope_service::{
    discover::r#loop::carbon::{evaluate_carbon_eval_shard, CarbonShardEvalOutcome},
    worker::Request,
};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::{
    cells::engine_invoker::now_epoch_ms,
    security::user_envelope::{decrypt_cuen_with_slot_subkey, CAS_BLOB_CUEN_SCOPE},
    storage::cas::{sha256_lower_hex, CarbonCasStore},
};

const EVAL_CACHE_TTL_SECS: u64 = 60;
const EVAL_CACHE_MAX_ENTRIES: usize = 2;

/// `CarbonEvalDecryptionGrant` is a struct carrying a 60-second ephemeral `FitJmat`-only slot
/// subkey from `carbon-engine` to `carbon-eval` (`RT-07`). It cannot decrypt `holdout.jmat`,
/// `JLMB` landmarks, or `/predict` payloads.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarbonEvalDecryptionGrant {
    pub tenant_id: String,
    pub fit_slot_subkey_b64: String,
    pub fit_plaintext_sha256: String,
    pub exp_epoch_secs: u64,
}

/// `CarbonEvalWireRequest` is a struct sent from `carbon engine` to a stateless `carbon eval` shard.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarbonEvalWireRequest {
    pub fit_jmat_sha256: String,
    pub request: Request,
    pub fit_champion: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decryption_grant: Option<CarbonEvalDecryptionGrant>,
}

/// `CarbonEvalInvoker` is a trait abstracting remote or local dispatch from `engine` to `eval` shards.
pub trait CarbonEvalInvoker: Send + Sync {
    fn invoke_eval(
        &self,
        payload: &CarbonEvalWireRequest,
    ) -> Result<CarbonShardEvalOutcome, String>;
}

/// `CarbonHttpEvalInvoker` is a struct dispatching `CarbonEvalWireRequest` over HTTP to `carbon emu`.
pub struct CarbonHttpEvalInvoker {
    invoke_url: String,
    agent: ureq::Agent,
}

impl CarbonHttpEvalInvoker {
    pub fn new(invoke_url: impl Into<String>) -> Self {
        Self {
            invoke_url: invoke_url.into(),
            agent: crate::tls::carbon_https_agent().clone(),
        }
    }
}

impl CarbonEvalInvoker for CarbonHttpEvalInvoker {
    fn invoke_eval(
        &self,
        payload: &CarbonEvalWireRequest,
    ) -> Result<CarbonShardEvalOutcome, String> {
        let req_bytes =
            serde_json::to_vec(payload).map_err(|e| format!("encode eval payload: {e}"))?;
        let mut resp = self
            .agent
            .post(&self.invoke_url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("content-type", "application/json")
            .send(&req_bytes)
            .map_err(|e| format!("HTTP eval invoke failed: {e}"))?;
        let status = resp.status().as_u16();
        let mut body = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("read eval invoke response: {e}"))?;
        if status >= 300 {
            return Err(format!(
                "eval shard HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        serde_json::from_slice(&body).map_err(|e| format!("decode eval outcome: {e}"))
    }
}

/// `CarbonAwsLambdaEvalInvoker` is a struct invoking a real AWS Lambda `eval` shard function
/// (`lambda://<function_name>`) via signed AWS SigV4 `POST /2015-03-31/functions/{name}/invocations`.
pub struct CarbonAwsLambdaEvalInvoker {
    region: String,
    function_name: String,
    agent: ureq::Agent,
}

impl CarbonAwsLambdaEvalInvoker {
    pub fn new(region: impl Into<String>, function_name: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            function_name: function_name.into(),
            agent: crate::tls::carbon_https_agent().clone(),
        }
    }
}

impl CarbonEvalInvoker for CarbonAwsLambdaEvalInvoker {
    fn invoke_eval(
        &self,
        payload: &CarbonEvalWireRequest,
    ) -> Result<CarbonShardEvalOutcome, String> {
        let req_bytes =
            serde_json::to_vec(payload).map_err(|e| format!("encode eval payload: {e}"))?;
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
        let headers = crate::security::sigv4::sign_aws_v4_headers(
            &creds,
            &self.region,
            "lambda",
            "POST",
            &host,
            &uri,
            &canonical_query,
            &req_bytes,
            now_epoch_ms() / 1_000,
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
        let mut resp = req.send(&req_bytes).map_err(|e| {
            format!(
                "AWS Lambda eval invoke '{}' failed: {e}",
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
            .map_err(|e| format!("read AWS Lambda eval invoke response: {e}"))?;
        if status >= 300 {
            return Err(format!(
                "AWS Lambda eval shard HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        if let Some(err_kind) = fn_err {
            return Err(format!(
                "AWS Lambda eval '{}' function error ({err_kind}): {}",
                self.function_name,
                String::from_utf8_lossy(&body)
            ));
        }
        serde_json::from_slice(&body).map_err(|e| format!("decode AWS Lambda eval outcome: {e}"))
    }
}

/// Construct either a `CarbonAwsLambdaEvalInvoker` (`lambda://<function_name>`) or a
/// `CarbonHttpEvalInvoker` (`http(s)://...`).
pub fn open_eval_invoker(target: &str, default_region: &str) -> Arc<dyn CarbonEvalInvoker> {
    if let Some(fn_name) = target.trim().strip_prefix("lambda://") {
        let region = match std::env::var("AWS_REGION") {
            Ok(val) if !val.trim().is_empty() => val,
            _ => default_region.to_owned(),
        };
        return Arc::new(CarbonAwsLambdaEvalInvoker::new(region, fn_name.trim()));
    }
    Arc::new(CarbonHttpEvalInvoker::new(target))
}

struct CachedFitEntry {
    expires_epoch_secs: u64,
    plaintext_sha256: String,
    bytes: Arc<Zeroizing<Vec<u8>>>,
}

struct LoadedFitPayload {
    bytes: Arc<Zeroizing<Vec<u8>>>,
    plaintext_sha256: String,
}

/// `CarbonEvalWorker` is a struct executing stateless candidate shard evaluations with short-lived
/// `Zeroizing` RAM caching (`<= 2` entries, `<= 60 s` TTL).
pub struct CarbonEvalWorker {
    cas: Arc<dyn CarbonCasStore>,
    scratch_root: PathBuf,
    fit_cache: Mutex<BTreeMap<(String, String), CachedFitEntry>>,
}

impl CarbonEvalWorker {
    pub fn new(cas: Arc<dyn CarbonCasStore>, scratch_root: PathBuf) -> Result<Self, String> {
        std::fs::create_dir_all(&scratch_root)
            .map_err(|e| format!("create eval scratch dir: {e}"))?;
        Ok(Self {
            cas,
            scratch_root,
            fit_cache: Mutex::new(BTreeMap::new()),
        })
    }

    fn load_fit_in_ram(
        &self,
        cas_sha256_hex: &str,
        grant: Option<&CarbonEvalDecryptionGrant>,
    ) -> Result<LoadedFitPayload, String> {
        let now_secs = now_epoch_ms() / 1_000;
        let cas_key = cas_sha256_hex.to_ascii_lowercase();
        let tenant_scope = grant
            .map(|g| g.tenant_id.clone())
            .unwrap_or_else(|| "vpc_plaintext".to_owned());

        if let Some(g) = grant {
            if now_secs > g.exp_epoch_secs {
                return Err("eval shard decryption grant has expired (TTL > 60s)".to_owned());
            }
        }

        let cache_key = (tenant_scope, cas_key.clone());
        if let Ok(mut guard) = self.fit_cache.lock() {
            guard.retain(|_, v| v.expires_epoch_secs >= now_secs);
            if let Some(hit) = guard.get(&cache_key) {
                if let Some(g) = grant {
                    // Still verify the caller supplied a valid subkey by checking length/base64,
                    // and match expected plaintext SHA-256!
                    if hit.plaintext_sha256 != g.fit_plaintext_sha256.to_ascii_lowercase() {
                        return Err("eval cache plaintext SHA-256 mismatch".to_owned());
                    }
                }
                return Ok(LoadedFitPayload {
                    bytes: Arc::clone(&hit.bytes),
                    plaintext_sha256: hit.plaintext_sha256.clone(),
                });
            }
        }

        let raw_blob = self.cas.get_blob_verified(&cas_key)?;
        let (pt_arc, pt_sha) = if let Some(g) = grant {
            let subkey_vec = Zeroizing::new(
                base64::engine::general_purpose::STANDARD
                    .decode(&g.fit_slot_subkey_b64)
                    .map_err(|e| format!("invalid fit_slot_subkey_b64: {e}"))?,
            );
            if subkey_vec.len() != 32 {
                return Err("fit_slot_subkey must be 32 bytes".to_owned());
            }
            let mut subkey = Zeroizing::new([0u8; 32]);
            subkey.copy_from_slice(&subkey_vec);
            let decrypted = decrypt_cuen_with_slot_subkey(
                &subkey,
                CuenPayloadKind::FitJmat,
                &g.tenant_id,
                CAS_BLOB_CUEN_SCOPE,
                &raw_blob,
            )
            .map_err(|e| format!("eval FitJmat CUEN decryption failed: {e}"))?;
            let actual_pt_sha = sha256_lower_hex(&decrypted);
            if actual_pt_sha != g.fit_plaintext_sha256.to_ascii_lowercase() {
                return Err("eval decrypted FitJmat SHA-256 mismatch".to_owned());
            }
            (Arc::new(decrypted), actual_pt_sha)
        } else {
            (Arc::new(Zeroizing::new(raw_blob)), cas_key)
        };

        let ttl_exp = grant
            .map(|g| {
                g.exp_epoch_secs
                    .min(now_secs.saturating_add(EVAL_CACHE_TTL_SECS))
            })
            .unwrap_or_else(|| now_secs.saturating_add(EVAL_CACHE_TTL_SECS));

        if let Ok(mut guard) = self.fit_cache.lock() {
            if guard.len() >= EVAL_CACHE_MAX_ENTRIES {
                guard.clear();
            }
            guard.insert(
                cache_key,
                CachedFitEntry {
                    expires_epoch_secs: ttl_exp,
                    plaintext_sha256: pt_sha.clone(),
                    bytes: Arc::clone(&pt_arc),
                },
            );
        }
        Ok(LoadedFitPayload {
            bytes: pt_arc,
            plaintext_sha256: pt_sha,
        })
    }

    pub fn handle_wire_bytes(&self, payload: &[u8]) -> Result<Vec<u8>, String> {
        let wire: CarbonEvalWireRequest =
            serde_json::from_slice(payload).map_err(|e| format!("invalid eval wire json: {e}"))?;
        let outcome = self.evaluate_shard(&wire)?;
        serde_json::to_vec(&outcome).map_err(|e| format!("serialize eval outcome: {e}"))
    }

    pub fn evaluate_shard(
        &self,
        wire: &CarbonEvalWireRequest,
    ) -> Result<CarbonShardEvalOutcome, String> {
        let LoadedFitPayload {
            bytes: fit_bytes,
            plaintext_sha256: pt_sha,
        } = self.load_fit_in_ram(&wire.fit_jmat_sha256, wire.decryption_grant.as_ref())?;
        evaluate_carbon_eval_shard(
            fit_bytes.as_slice(),
            &pt_sha,
            &self.scratch_root,
            &wire.request,
            wire.fit_champion,
        )
        .map_err(|e| format!("eval shard execution error: {e}"))
    }
}

impl CarbonEvalInvoker for CarbonEvalWorker {
    fn invoke_eval(
        &self,
        payload: &CarbonEvalWireRequest,
    ) -> Result<CarbonShardEvalOutcome, String> {
        self.evaluate_shard(payload)
    }
}
