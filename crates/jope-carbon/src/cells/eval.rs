//! Stateless `carbon eval` batch evaluation worker (`XS` `512 MB` .. `XL` `7,076 MB`, `ADR-0007`).
//!
//! Holds zero persistent cryptographic secrets (`RT-07`): decrypts `CUEN\x01(FitJmat)` strictly
//! in `Zeroizing` RAM using a 60-second ephemeral `FitJmat` slot subkey granted by `carbon-engine`,
//! bounds warm RAM caching to 2 entries with a 60-second TTL, and never accesses `holdout.jmat`.

use crate::error::CarbonErrCtx as _;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use jope_prep::cuen::CuenPayloadKind;
use jope_service::{
    discover::r#loop::carbon::{
        evaluate_carbon_eval_shard_with_executor, CarbonInProcessExecutor, CarbonShardEvalOutcome,
    },
    worker::{Request, Response, WorkerFailure},
};
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, Zeroizing};

use crate::{
    cells::{
        engine_invoker::now_epoch_ms,
        lambda_client::{post_json_http_bounded, CarbonLambdaRpcClient},
    },
    error::{err_to_string, to_json_vec},
    security::user_envelope::{decrypt_cuen_with_slot_subkey, CAS_BLOB_CUEN_SCOPE},
    storage::cas::{sha256_lower_hex, CarbonCasStore},
};

const EVAL_CACHE_TTL_SECS: u64 = 60;
const EVAL_CACHE_MAX_ENTRIES: usize = 2;

/// `CarbonEvalDecryptionGrant` is a struct carrying a 60-second ephemeral `FitJmat`-only slot
/// subkey from `carbon-engine` to `carbon-eval` (`RT-07`). It cannot decrypt `holdout.jmat`,
/// `JLMB` landmarks, or `/predict` payloads.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarbonEvalDecryptionGrant {
    pub tenant_id: String,
    pub fit_slot_subkey_b64: String,
    pub fit_plaintext_sha256: String,
    pub exp_epoch_secs: u64,
}

impl std::fmt::Debug for CarbonEvalDecryptionGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonEvalDecryptionGrant")
            .field("tenant_id", &self.tenant_id)
            .field("fit_slot_subkey_b64", &"[REDACTED]")
            .field("fit_plaintext_sha256", &self.fit_plaintext_sha256)
            .field("exp_epoch_secs", &self.exp_epoch_secs)
            .finish()
    }
}

impl Drop for CarbonEvalDecryptionGrant {
    fn drop(&mut self) {
        self.fit_slot_subkey_b64.zeroize();
    }
}

/// `CarbonEvalWireRequest` is a struct sent from `carbon engine` to a stateless `carbon eval` shard.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarbonEvalWireRequest {
    pub fit_jmat_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_scope: Option<String>,
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
        let req_bytes = to_json_vec(payload)?;
        let body = post_json_http_bounded(
            &self.agent,
            &self.invoke_url,
            &req_bytes,
            false,
            "eval shard",
        )?;
        serde_json::from_slice(&body).map_err(err_to_string)
    }
}

/// `CarbonAwsLambdaEvalInvoker` is a struct invoking a real AWS Lambda `eval` shard function
/// (`lambda://<function_name>`) via signed AWS SigV4 `POST /2015-03-31/functions/{name}/invocations`.
pub struct CarbonAwsLambdaEvalInvoker {
    rpc: CarbonLambdaRpcClient,
}

impl CarbonAwsLambdaEvalInvoker {
    pub fn new(region: impl Into<String>, function_name: impl Into<String>) -> Self {
        Self {
            rpc: CarbonLambdaRpcClient::new(region, function_name, "eval"),
        }
    }
}

impl CarbonEvalInvoker for CarbonAwsLambdaEvalInvoker {
    fn invoke_eval(
        &self,
        payload: &CarbonEvalWireRequest,
    ) -> Result<CarbonShardEvalOutcome, String> {
        let req_bytes = to_json_vec(payload)?;
        let body = self.rpc.invoke_bytes(&req_bytes, false, 3)?;
        serde_json::from_slice(&body).map_err(err_to_string)
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
    executor: Arc<CarbonInProcessExecutor>,
}

/// `CarbonEvalWorker` is a struct executing stateless candidate shard evaluations with short-lived
/// `Zeroizing` RAM `Matrix` caching (`<= 2` entries, `<= 60 s` TTL).
pub struct CarbonEvalWorker {
    cas: Arc<dyn CarbonCasStore>,
    scratch_root: PathBuf,
    fit_cache: Mutex<BTreeMap<(String, String, String), CachedFitEntry>>,
}

impl CarbonEvalWorker {
    pub fn new(cas: Arc<dyn CarbonCasStore>, scratch_root: PathBuf) -> Result<Self, String> {
        std::fs::create_dir_all(&scratch_root).err_ctx("create eval scratch dir")?;
        Ok(Self {
            cas,
            scratch_root,
            fit_cache: Mutex::new(BTreeMap::new()),
        })
    }

    fn load_fit_executor_in_ram(
        &self,
        cas_sha256_hex: &str,
        blob_scope: Option<&str>,
        grant: Option<&CarbonEvalDecryptionGrant>,
    ) -> Result<Arc<CarbonInProcessExecutor>, String> {
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

        let scope_key = blob_scope.unwrap_or("").to_owned();
        let cache_key = (tenant_scope, scope_key, cas_key.clone());
        if let Ok(mut guard) = self.fit_cache.lock() {
            guard.retain(|_, v| v.expires_epoch_secs >= now_secs);
            if let Some(hit) = guard.get(&cache_key) {
                if let Some(g) = grant {
                    if hit.plaintext_sha256 != g.fit_plaintext_sha256.to_ascii_lowercase() {
                        return Err("eval cache plaintext SHA-256 mismatch".to_owned());
                    }
                }
                return Ok(Arc::clone(&hit.executor));
            }
        }

        let raw_blob = self.cas.get_scoped_blob_verified(blob_scope, &cas_key)?;
        let (mut pt_vec, pt_sha) = if let Some(g) = grant {
            let subkey_vec = Zeroizing::new(
                base64::engine::general_purpose::STANDARD
                    .decode(&g.fit_slot_subkey_b64)
                    .err_ctx("invalid fit_slot_subkey_b64")?,
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
            .err_ctx("eval FitJmat CUEN decryption failed")?;
            let actual_pt_sha = sha256_lower_hex(&decrypted);
            if actual_pt_sha != g.fit_plaintext_sha256.to_ascii_lowercase() {
                return Err("eval decrypted FitJmat SHA-256 mismatch".to_owned());
            }
            (decrypted, actual_pt_sha)
        } else {
            (Zeroizing::new(raw_blob), cas_key)
        };

        let plain_owned = std::mem::take(&mut *pt_vec);
        let exec = Arc::new(
            CarbonInProcessExecutor::open_fit(plain_owned, &pt_sha, self.scratch_root.clone())
                .err_ctx("open eval FIT matrix failed")?,
        );

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
                    plaintext_sha256: pt_sha,
                    executor: Arc::clone(&exec),
                },
            );
        }
        Ok(exec)
    }

    pub fn handle_wire_bytes(&self, payload: &[u8]) -> Result<Vec<u8>, String> {
        self.handle_wire_bytes_with_deadline(payload, None)
    }

    pub fn handle_wire_bytes_with_deadline(
        &self,
        payload: &[u8],
        deadline_epoch_ms: Option<u64>,
    ) -> Result<Vec<u8>, String> {
        let wire: CarbonEvalWireRequest = serde_json::from_slice(payload).map_err(err_to_string)?;
        let outcome = self.evaluate_shard_with_deadline(&wire, deadline_epoch_ms)?;
        to_json_vec(&outcome)
    }

    pub fn evaluate_shard(
        &self,
        wire: &CarbonEvalWireRequest,
    ) -> Result<CarbonShardEvalOutcome, String> {
        self.evaluate_shard_with_deadline(wire, None)
    }

    pub fn evaluate_shard_with_deadline(
        &self,
        wire: &CarbonEvalWireRequest,
        deadline_epoch_ms: Option<u64>,
    ) -> Result<CarbonShardEvalOutcome, String> {
        const EVAL_DEADLINE_SAFETY_MARGIN_MS: u64 = 8_000;
        let mut bounded_req = wire.request.clone();
        if let Some(deadline_ms) = deadline_epoch_ms {
            let now_ms = now_epoch_ms();
            if now_ms.saturating_add(EVAL_DEADLINE_SAFETY_MARGIN_MS) >= deadline_ms {
                return Ok(CarbonShardEvalOutcome {
                    response: Response::failed(bounded_req.attempt, WorkerFailure::Timeout, 0),
                    fitted_model_bytes: None,
                });
            }
            let remaining_ms = deadline_ms
                .saturating_sub(now_ms)
                .saturating_sub(EVAL_DEADLINE_SAFETY_MARGIN_MS);
            if bounded_req.limits.wall_ms == 0 || bounded_req.limits.wall_ms > remaining_ms {
                bounded_req.limits.wall_ms = remaining_ms.max(1);
            }
        }
        let exec = self.load_fit_executor_in_ram(
            &wire.fit_jmat_sha256,
            wire.blob_scope.as_deref(),
            wire.decryption_grant.as_ref(),
        )?;
        evaluate_carbon_eval_shard_with_executor(&exec, &bounded_req, wire.fit_champion)
            .err_ctx("eval shard execution error")
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
