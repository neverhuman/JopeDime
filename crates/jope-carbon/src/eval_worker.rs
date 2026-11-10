//! Stateless `carbon eval` batch evaluation worker (`XS` `512 MB` .. `XL` `7,076 MB`, `ADR-0007` Phase C5).
//!
//! Fetches and caches `fit.jmat` by SHA-256 across warm Lambda invocations, evaluates candidate
//! fold requests via `evaluate_carbon_eval_shard`, and never touches `holdout.jmat`.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use jope_service::{
    discover::r#loop::carbon::{evaluate_carbon_eval_shard, CarbonShardEvalOutcome},
    worker::Request,
};
use serde::{Deserialize, Serialize};

use crate::cas::CarbonCasStore;

/// `CarbonEvalWireRequest` is a struct sent from `carbon engine` to a stateless `carbon eval` shard.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarbonEvalWireRequest {
    pub fit_jmat_sha256: String,
    pub request: Request,
    pub fit_champion: bool,
}

/// `CarbonEvalInvoker` is a trait abstracting remote or local dispatch from `engine` to `eval` shards.
pub trait CarbonEvalInvoker: Send + Sync {
    fn invoke_eval(
        &self,
        payload: &CarbonEvalWireRequest,
    ) -> Result<CarbonShardEvalOutcome, String>;
}

/// `CarbonHttpEvalInvoker` is a struct dispatching `CarbonEvalWireRequest` over HTTP to a
/// `carbon eval` Lambda Runtime API proxy (`carbon emu` or regional Lambda Invoke endpoint).
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

/// `CarbonEvalWorker` is a struct executing stateless candidate shard evaluations with warm-container
/// `fit.jmat` caching bounded to 4 entries.
pub struct CarbonEvalWorker {
    cas: Arc<dyn CarbonCasStore>,
    scratch_root: PathBuf,
    fit_cache: Mutex<BTreeMap<String, Arc<Vec<u8>>>>,
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

    fn load_fit_cached(&self, sha256_hex: &str) -> Result<Arc<Vec<u8>>, String> {
        let key = sha256_hex.to_ascii_lowercase();
        if let Ok(guard) = self.fit_cache.lock() {
            if let Some(hit) = guard.get(&key) {
                return Ok(Arc::clone(hit));
            }
        }
        let fetched = Arc::new(self.cas.get_blob_verified(&key)?);
        if let Ok(mut guard) = self.fit_cache.lock() {
            if guard.len() >= 4 {
                guard.clear();
            }
            guard.insert(key, Arc::clone(&fetched));
        }
        Ok(fetched)
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
        let fit_bytes = self.load_fit_cached(&wire.fit_jmat_sha256)?;
        evaluate_carbon_eval_shard(
            &fit_bytes,
            &wire.fit_jmat_sha256.to_ascii_lowercase(),
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
