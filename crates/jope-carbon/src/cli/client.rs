//! Zero-knowledge thin client SDK for Carbon regional Lambda cells and Enterprise `carbon serve`.
//!
//! Performs deterministic `FIT` / `HOLDOUT` splitting, `FIT`-only landmark conditioning (`JLMB` v1),
//! `JMAT` v1 encoding, and per-user `ECDH P-256 + HKDF-SHA256 + AES-256-GCM` (`CUEN\x01` / `CUDK\x01`)
//! envelope encryption strictly on the client before uploading to size/SHA-pinned presigned URLs.

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::{
    atomic::{AtomicU32, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonAnonymousAuthResponse, CarbonCreateRunRequest, CarbonEarlyStopRequest,
        CarbonPingResponse, CarbonPredictRequest, CarbonPredictResponse,
        CarbonPrepareUploadRequest, CarbonPrepareUploadResponse, CarbonPublicWaveProgress,
        CarbonRegionCatalog, CarbonRegisterFreeRequest, CarbonRegisterFreeResponse,
        CarbonResumeRunRequest, CarbonRunStateWire, CarbonShredRunResponse, CarbonTierWire,
        CarbonUploadSlotWire,
    },
    decimal::DecimalU64,
    ids::{DatasetId, RunId},
    Effort,
};
use jope_prep::{
    cuen::CuenPayloadKind, split_and_encode_jmat_v1, FitLandmarkBundle, PrepTable,
    SplitJmatArtifacts, SplitPlan,
};
use zeroize::Zeroizing;

use crate::{
    cells::engine_invoker::now_epoch_ms,
    security::{
        auth::{CarbonAuthVerifier, CarbonCapabilityClaims},
        sigv4::{sign_aws_v4_headers_with_extra, url_encode_component, CarbonAwsCredentials},
        user_envelope::{
            decode_p256_pubkey_b64, decrypt_cuen_payload, encrypt_cuen_payload,
            generate_random_u_dek, unwrap_u_dek_with_kek, wrap_u_dek_to_pubkey,
            CarbonEncryptedPredictPayload, CarbonEngineKek, CAS_BLOB_CUEN_SCOPE,
        },
    },
    storage::cas::sha256_lower_hex,
};

#[derive(Clone)]
struct UploadedEncryptionSecret {
    u_dek: Zeroizing<[u8; 32]>,
    fit_cuen_sha256: String,
    holdout_cuen_sha256: String,
}

/// `CarbonClientPreparedDataset` is a struct holding client-side split `JMAT` blobs and the
/// `FIT`-only `6,368`-byte `FitLandmarkBundle` ready for encryption and upload.
#[derive(Clone, Debug)]
pub struct CarbonClientPreparedDataset {
    pub split_plan: SplitPlan,
    pub landmark_bundle: FitLandmarkBundle,
    pub artifacts: SplitJmatArtifacts,
    pub feature_names: Vec<String>,
    pub target_name: String,
    pub fit_rows: u32,
    pub holdout_rows: u32,
}

/// `CarbonProbedEndpoint` is a struct recording one regional `/v1/carbon/ping` RTT measurement.
#[derive(Clone, Debug)]
pub struct CarbonProbedEndpoint {
    pub base_url: String,
    pub region: String,
    pub rtt_ms: f64,
    pub enterprise_self_hosted: bool,
    pub engine_kek_pubkey_b64: Option<String>,
}

static LOAD_SPREAD_SEQ: AtomicUsize = AtomicUsize::new(0);

/// Callback invoked on initial run creation/resume and on each async poll status update.
pub type CarbonProgressObserver = Arc<dyn Fn(&CarbonPublicWaveProgress) + Send + Sync>;

/// `CarbonThinClient` is a struct driving zero-knowledge end-to-end Carbon inference, search,
/// async `202` polling, pause/resume, cryptographic shred, and multi-region failover.
#[derive(Clone)]
pub struct CarbonThinClient {
    endpoints: Vec<String>,
    active_idx: Arc<AtomicUsize>,
    enterprise_self_hosted: bool,
    endpoint_keks: Arc<Mutex<BTreeMap<String, [u8; 65]>>>,
    pinned_keks: BTreeMap<String, String>,
    capability_token: Option<String>,
    uploaded_secrets: Arc<Mutex<BTreeMap<String, UploadedEncryptionSecret>>>,
    run_udeks: Arc<Mutex<BTreeMap<String, Zeroizing<[u8; 32]>>>>,
    poll_requests: Arc<AtomicU32>,
    http_request_bytes: Arc<AtomicUsize>,
    http_response_bytes: Arc<AtomicUsize>,
    progress_observer: Option<CarbonProgressObserver>,
    agent: ureq::Agent,
}

impl CarbonThinClient {
    /// Probe `/v1/carbon/ping` across all candidates in parallel and select a lowest-RTT healthy cell.
    pub fn connect_fastest(
        candidate_urls: Vec<String>,
    ) -> Result<(Self, Vec<CarbonProbedEndpoint>), String> {
        Self::connect_fastest_pinned(candidate_urls, &BTreeMap::new())
    }

    /// Probe `/v1/carbon/ping` concurrently across all candidate regional URLs, enforce any pinned
    /// `engine_kek_pubkey_b64` per endpoint, and spread load across healthy cells within a `25 ms`
    /// RTT band of the fastest cell.
    pub fn connect_fastest_pinned(
        candidate_urls: Vec<String>,
        pinned_keks: &BTreeMap<String, String>,
    ) -> Result<(Self, Vec<CarbonProbedEndpoint>), String> {
        let mut deduped = Vec::new();
        for raw_url in candidate_urls {
            let trimmed = raw_url.trim().trim_end_matches('/').to_owned();
            if !trimmed.is_empty() && !deduped.contains(&trimmed) {
                deduped.push(trimmed);
            }
        }
        if deduped.is_empty() {
            return Err("CarbonThinClient requires at least one endpoint URL".to_owned());
        }
        let normalized_pins: BTreeMap<String, String> = pinned_keks
            .iter()
            .map(|(k, v)| {
                (
                    k.trim().trim_end_matches('/').to_owned(),
                    v.trim().to_owned(),
                )
            })
            .collect();

        let agent = crate::tls::carbon_https_agent().clone();
        let probe_outcomes: Vec<Result<CarbonProbedEndpoint, String>> = thread::scope(|s| {
            let mut handles = Vec::with_capacity(deduped.len());
            for base in &deduped {
                let base_url = base.clone();
                let ag = agent.clone();
                let expected_kek = normalized_pins.get(&base_url).cloned();
                handles.push(s.spawn(move || {
                    let t0 = Instant::now();
                    let (code, buf) = send_raw_edge_http_with_agent(
                        &ag,
                        &base_url,
                        "GET",
                        "/v1/carbon/ping",
                        &[],
                        None,
                    )
                    .map_err(|e| format!("{base_url}: {e}"))?;
                    if code != 200 {
                        return Err(format!("{base_url}: /v1/carbon/ping returned HTTP {code}"));
                    }
                    let ping: CarbonPingResponse = serde_json::from_slice(&buf)
                        .map_err(|e| format!("{base_url}: invalid ping JSON: {e}"))?;
                    if !ping.healthy {
                        return Err(format!("{base_url}: reported unhealthy"));
                    }
                    if let Some(pin_b64) = expected_kek {
                        if ping.engine_kek_pubkey_b64.as_deref() != Some(pin_b64.as_str()) {
                            return Err(format!(
                                "{base_url}: engine_kek_pubkey_b64 mismatch against pinned manifest"
                            ));
                        }
                    }
                    Ok(CarbonProbedEndpoint {
                        base_url,
                        region: ping.region,
                        rtt_ms: t0.elapsed().as_secs_f64() * 1_000.0,
                        enterprise_self_hosted: ping.enterprise_self_hosted,
                        engine_kek_pubkey_b64: ping.engine_kek_pubkey_b64,
                    })
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("regional ping probe thread panicked".to_owned()))
                })
                .collect()
        });

        let mut healthy = Vec::new();
        let mut errors = Vec::new();
        let mut endpoint_keks = BTreeMap::new();
        for outcome in probe_outcomes {
            match outcome {
                Ok(probed) => {
                    if let Some(b64) = &probed.engine_kek_pubkey_b64 {
                        let decoded = decode_p256_pubkey_b64(b64).map_err(|e| e.to_string())?;
                        endpoint_keks.insert(probed.base_url.clone(), decoded);
                    }
                    healthy.push(probed);
                }
                Err(err) => errors.push(err),
            }
        }
        if healthy.is_empty() {
            return Err(format!(
                "all candidate Carbon regional endpoints failed /v1/carbon/ping ({})",
                errors.join("; ")
            ));
        }
        healthy.sort_by(|a, b| {
            a.rtt_ms
                .partial_cmp(&b.rtt_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let min_rtt = healthy[0].rtt_ms;
        let band_len = healthy
            .iter()
            .take_while(|h| (h.rtt_ms - min_rtt) <= 25.0)
            .count();
        if band_len > 1 {
            let seq = LOAD_SPREAD_SEQ.fetch_add(1, Ordering::Relaxed);
            let chosen = (seq
                .wrapping_add((now_epoch_ms() as usize) & 0xff)
                .wrapping_add(std::process::id() as usize))
                % band_len;
            if chosen > 0 {
                let picked = healthy.remove(chosen);
                healthy.insert(0, picked);
            }
        }
        let enterprise_self_hosted = healthy[0].enterprise_self_hosted;
        let ordered_endpoints: Vec<String> = healthy.iter().map(|h| h.base_url.clone()).collect();
        Ok((
            Self {
                endpoints: ordered_endpoints,
                active_idx: Arc::new(AtomicUsize::new(0)),
                enterprise_self_hosted,
                endpoint_keks: Arc::new(Mutex::new(endpoint_keks)),
                pinned_keks: normalized_pins,
                capability_token: None,
                uploaded_secrets: Arc::new(Mutex::new(BTreeMap::new())),
                run_udeks: Arc::new(Mutex::new(BTreeMap::new())),
                poll_requests: Arc::new(AtomicU32::new(0)),
                http_request_bytes: Arc::new(AtomicUsize::new(0)),
                http_response_bytes: Arc::new(AtomicUsize::new(0)),
                progress_observer: None,
                agent,
            },
            healthy,
        ))
    }

    /// Attach an observer invoked on initial run creation/resume and on each async poll update.
    pub fn with_progress_observer<F>(mut self, observer: F) -> Self
    where
        F: Fn(&CarbonPublicWaveProgress) + Send + Sync + 'static,
    {
        self.progress_observer = Some(Arc::new(observer));
        self
    }

    /// Return `(poll_requests, http_request_bytes, http_response_bytes)` observed on this client.
    pub fn telemetry_counters(&self) -> (u32, usize, usize) {
        (
            self.poll_requests.load(Ordering::Relaxed),
            self.http_request_bytes.load(Ordering::Relaxed),
            self.http_response_bytes.load(Ordering::Relaxed),
        )
    }

    /// Reset per-run HTTP poll and byte counters on this client.
    pub fn reset_telemetry_counters(&self) {
        self.poll_requests.store(0, Ordering::Relaxed);
        self.http_request_bytes.store(0, Ordering::Relaxed);
        self.http_response_bytes.store(0, Ordering::Relaxed);
    }

    /// Attach a signed `ctk1.*` capability token for Pro, Free, or Admin requests.
    pub fn with_capability_token(mut self, token: impl Into<String>) -> Self {
        let tok = token.into();
        if !tok.trim().is_empty() {
            self.capability_token = Some(tok);
        }
        self
    }

    /// Mint a 1-hour anonymous Free-tier capability token from `POST /v1/carbon/auth/anonymous`
    /// with automatic multi-region failover, and attach it to this client.
    pub fn authenticate_anonymous(&mut self) -> Result<CarbonAnonymousAuthResponse, String> {
        let bytes = self.request_checked_with_policy(
            "POST",
            "/v1/carbon/auth/anonymous",
            None,
            None,
            Some(b"{}"),
            true,
        )?;
        let resp: CarbonAnonymousAuthResponse = serde_json::from_slice(&bytes)
            .map_err(|e| format!("decode CarbonAnonymousAuthResponse: {e}"))?;
        self.record_verified_endpoint_kek(
            self.active_endpoint(),
            resp.engine_kek_pubkey_b64.as_deref(),
        )?;
        self.capability_token = Some(resp.capability_token.clone());
        Ok(resp)
    }

    /// Register a persistent 30-day Free-tier capability token via `POST /v1/carbon/auth/register`
    /// with automatic multi-region failover, and attach it to this client.
    pub fn register_free(
        &mut self,
        label: Option<&str>,
    ) -> Result<CarbonRegisterFreeResponse, String> {
        let req_body = serde_json::to_vec(&CarbonRegisterFreeRequest {
            label: label.map(str::to_owned),
        })
        .map_err(|e| format!("encode CarbonRegisterFreeRequest: {e}"))?;
        let bytes = self.request_checked_with_policy(
            "POST",
            "/v1/carbon/auth/register",
            None,
            None,
            Some(&req_body),
            true,
        )?;
        let resp: CarbonRegisterFreeResponse = serde_json::from_slice(&bytes)
            .map_err(|e| format!("decode CarbonRegisterFreeResponse: {e}"))?;
        self.record_verified_endpoint_kek(
            self.active_endpoint(),
            resp.engine_kek_pubkey_b64.as_deref(),
        )?;
        self.capability_token = Some(resp.capability_token.clone());
        Ok(resp)
    }

    fn record_verified_endpoint_kek(
        &self,
        endpoint: &str,
        advertised_b64: Option<&str>,
    ) -> Result<(), String> {
        let norm = endpoint.trim().trim_end_matches('/');
        if let Some(pin_b64) = self.pinned_keks.get(norm) {
            if advertised_b64 != Some(pin_b64.as_str()) {
                return Err(format!(
                    "{norm}: advertised engine_kek_pubkey_b64 did not match pinned manifest"
                ));
            }
        }
        if let Some(b64) = advertised_b64 {
            let decoded = decode_p256_pubkey_b64(b64).map_err(|e| e.to_string())?;
            if let Ok(mut guard) = self.endpoint_keks.lock() {
                guard.insert(norm.to_owned(), decoded);
            }
        }
        Ok(())
    }

    fn active_kek_pubkey(&self) -> Option<[u8; 65]> {
        let active = self.active_endpoint();
        self.endpoint_keks
            .lock()
            .ok()
            .and_then(|g| g.get(active).copied())
    }

    pub fn active_endpoint(&self) -> &str {
        let idx = self
            .active_idx
            .load(Ordering::Relaxed)
            .min(self.endpoints.len().saturating_sub(1));
        &self.endpoints[idx]
    }

    pub fn enterprise_self_hosted(&self) -> bool {
        self.enterprise_self_hosted
    }

    pub fn has_zero_knowledge_kek(&self) -> bool {
        self.active_kek_pubkey().is_some()
            || self
                .endpoint_keks
                .lock()
                .map(|g| !g.is_empty())
                .unwrap_or(false)
    }

    pub fn candidate_endpoints(&self) -> &[String] {
        &self.endpoints
    }

    pub fn ping_active(&self) -> Result<CarbonPingResponse, String> {
        let bytes =
            self.request_checked_with_policy("GET", "/v1/carbon/ping", None, None, None, true)?;
        let resp: CarbonPingResponse = serde_json::from_slice(&bytes)
            .map_err(|e| format!("decode CarbonPingResponse: {e}"))?;
        self.record_verified_endpoint_kek(
            self.active_endpoint(),
            resp.engine_kek_pubkey_b64.as_deref(),
        )?;
        Ok(resp)
    }

    pub fn fetch_region_catalog(&self) -> Result<CarbonRegionCatalog, String> {
        let bytes = self.request_checked_with_policy(
            "GET",
            "/.well-known/carbon-regions.json",
            None,
            None,
            None,
            true,
        )?;
        serde_json::from_slice(&bytes).map_err(|e| format!("decode CarbonRegionCatalog: {e}"))
    }

    /// Perform client-side `SplitPlan::deterministic`, `FitLandmarkBundle::extract_fit_only`,
    /// and separated `fit.jmat` / `holdout.jmat` encoding so holdout labels never touch landmarks.
    pub fn prepare_numeric_dataset(
        headers: Vec<String>,
        numeric_rows: &[Vec<f64>],
        target_name: &str,
        seed: u64,
        holdout_fraction: f64,
    ) -> Result<CarbonClientPreparedDataset, String> {
        let target_idx = headers
            .iter()
            .position(|h| h == target_name)
            .ok_or_else(|| format!("target column '{target_name}' not found in headers"))?;
        let feature_indices: Vec<usize> = (0..headers.len()).filter(|&i| i != target_idx).collect();
        if feature_indices.is_empty() {
            return Err("dataset must have at least one feature column".to_owned());
        }
        let feature_names: Vec<String> = feature_indices
            .iter()
            .map(|&i| headers[i].clone())
            .collect();
        let total_rows = numeric_rows.len();
        let split_plan = SplitPlan::deterministic(total_rows, seed, holdout_fraction)
            .map_err(|e| format!("split plan failed: {e}"))?;

        let mut full_x = Vec::with_capacity(total_rows * feature_indices.len());
        let mut full_y = Vec::with_capacity(total_rows);
        let mut string_rows = Vec::with_capacity(total_rows);
        for (r_idx, row) in numeric_rows.iter().enumerate() {
            if row.len() != headers.len() {
                return Err(format!("row {r_idx} column count mismatch"));
            }
            for &fi in &feature_indices {
                full_x.push(row[fi] as f32);
            }
            full_y.push(row[target_idx] as f32);
            string_rows.push(row.iter().map(|v| Some(format!("{v:.10}"))).collect());
        }

        let landmark_bundle = FitLandmarkBundle::extract_fit_only(
            &full_x,
            total_rows,
            feature_indices.len(),
            &full_y,
            &split_plan,
        )
        .map_err(|e| format!("FIT-only landmark conditioning failed: {e}"))?;

        let table = PrepTable {
            headers,
            rows: string_rows,
        };
        let artifacts = split_and_encode_jmat_v1(&table, &split_plan)
            .map_err(|e| format!("JMAT split encoding failed: {e}"))?;

        Ok(CarbonClientPreparedDataset {
            fit_rows: split_plan.fit_rows().len() as u32,
            holdout_rows: split_plan.holdout_rows().len() as u32,
            split_plan,
            landmark_bundle,
            artifacts,
            feature_names,
            target_name: target_name.to_owned(),
        })
    }

    /// Encrypt `fit.jmat` and `holdout.jmat` with a fresh per-upload 256-bit `U_DEK` (`CUEN\x01`)
    /// when `engine_kek_pubkey` is advertised, request size/SHA-pinned presigned URLs, and `PUT` both blobs.
    pub fn upload_prepared(
        &self,
        tenant_id: &str,
        tier: CarbonTierWire,
        prepared: &CarbonClientPreparedDataset,
    ) -> Result<CarbonPrepareUploadResponse, String> {
        let pt_fit_sha = hex::encode(prepared.artifacts.fit_sha256);
        let (fit_wire, fit_sha, holdout_wire, holdout_sha, encrypted_cuen, secret_opt) =
            if self.has_zero_knowledge_kek() {
                let u_dek = generate_random_u_dek().map_err(|e| e.to_string())?;
                let fit_enc = encrypt_cuen_payload(
                    &u_dek,
                    CuenPayloadKind::FitJmat,
                    tenant_id,
                    CAS_BLOB_CUEN_SCOPE,
                    &prepared.artifacts.fit_jmat,
                )
                .map_err(|e| e.to_string())?;
                let hold_enc = encrypt_cuen_payload(
                    &u_dek,
                    CuenPayloadKind::HoldoutJmat,
                    tenant_id,
                    CAS_BLOB_CUEN_SCOPE,
                    &prepared.artifacts.holdout_jmat,
                )
                .map_err(|e| e.to_string())?;
                let f_sha = sha256_lower_hex(&fit_enc);
                let h_sha = sha256_lower_hex(&hold_enc);
                let secret = UploadedEncryptionSecret {
                    u_dek,
                    fit_cuen_sha256: f_sha.clone(),
                    holdout_cuen_sha256: h_sha.clone(),
                };
                (fit_enc, f_sha, hold_enc, h_sha, true, Some(secret))
            } else {
                let f_sha = pt_fit_sha.clone();
                let h_sha = hex::encode(prepared.artifacts.holdout_sha256);
                (
                    prepared.artifacts.fit_jmat.clone(),
                    f_sha,
                    prepared.artifacts.holdout_jmat.clone(),
                    h_sha,
                    false,
                    None,
                )
            };

        let prep_req = CarbonPrepareUploadRequest {
            tenant_id: tenant_id.to_owned(),
            tier,
            fit_rows: prepared.fit_rows,
            holdout_rows: prepared.holdout_rows,
            feature_count: prepared.feature_names.len() as u32,
            fit_jmat: CarbonUploadSlotWire {
                sha256: fit_sha,
                bytes: DecimalU64(fit_wire.len() as u64),
            },
            holdout_jmat: CarbonUploadSlotWire {
                sha256: holdout_sha,
                bytes: DecimalU64(holdout_wire.len() as u64),
            },
            encrypted_cuen,
        };
        let payload = serde_json::to_vec(&prep_req).map_err(|e| format!("encode JSON: {e}"))?;
        let resp_bytes = self.request_checked_with_policy(
            "POST",
            "/v1/carbon/uploads/prepare",
            Some(tenant_id),
            Some(tier),
            Some(&payload),
            true,
        )?;
        let prep_resp: CarbonPrepareUploadResponse = serde_json::from_slice(&resp_bytes)
            .map_err(|e| format!("decode upload prepare response: {e}"))?;
        self.record_verified_endpoint_kek(
            self.active_endpoint(),
            prep_resp.engine_kek_pubkey_b64.as_deref(),
        )?;

        self.put_presigned_bytes(&prep_resp.fit_target.url, &fit_wire)?;
        self.put_presigned_bytes(&prep_resp.holdout_target.url, &holdout_wire)?;
        if let Some(secret) = secret_opt {
            if let Ok(mut guard) = self.uploaded_secrets.lock() {
                guard.insert(format!("{tenant_id}:{pt_fit_sha}"), secret);
            }
        }
        Ok(prep_resp)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_run(
        &self,
        run_id: RunId,
        dataset_id: DatasetId,
        tenant_id: &str,
        tier: CarbonTierWire,
        effort: Effort,
        objective: &str,
        seed: u64,
        prepared: &CarbonClientPreparedDataset,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let run_str = run_id.to_string();
        let pt_fit_sha = hex::encode(prepared.artifacts.fit_sha256);
        let lm_wire = prepared.landmark_bundle.to_wire_bytes();

        let req = if let Some(kek_pub) = self.active_kek_pubkey() {
            let secret = self
                .uploaded_secrets
                .lock()
                .map_err(|_| "uploaded_secrets mutex poisoned".to_owned())?
                .get(&format!("{tenant_id}:{pt_fit_sha}"))
                .cloned()
                .ok_or_else(|| {
                    "upload_prepared must be called on this client before start_run in zero-knowledge mode"
                        .to_owned()
                })?;
            let enc_lm = encrypt_cuen_payload(
                &secret.u_dek,
                CuenPayloadKind::LandmarkJlmb,
                tenant_id,
                &run_str,
                &lm_wire,
            )
            .map_err(|e| e.to_string())?;
            let cudk = wrap_u_dek_to_pubkey(&secret.u_dek, &kek_pub, tenant_id, &run_str)
                .map_err(|e| e.to_string())?;
            if let Ok(mut run_map) = self.run_udeks.lock() {
                run_map.insert(format!("{tenant_id}:{run_str}"), secret.u_dek.clone());
            }
            CarbonCreateRunRequest {
                run_id,
                dataset_id,
                tenant_id: tenant_id.to_owned(),
                tier,
                fit_jmat_sha256: secret.fit_cuen_sha256,
                holdout_jmat_sha256: secret.holdout_cuen_sha256,
                fit_rows: prepared.fit_rows,
                holdout_rows: prepared.holdout_rows,
                feature_names: prepared.feature_names.clone(),
                target_name: prepared.target_name.clone(),
                objective: objective.to_owned(),
                effort,
                seed: DecimalU64(seed),
                landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(enc_lm),
                encrypted_cuen: true,
                wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk)),
            }
        } else {
            CarbonCreateRunRequest {
                run_id,
                dataset_id,
                tenant_id: tenant_id.to_owned(),
                tier,
                fit_jmat_sha256: pt_fit_sha,
                holdout_jmat_sha256: hex::encode(prepared.artifacts.holdout_sha256),
                fit_rows: prepared.fit_rows,
                holdout_rows: prepared.holdout_rows,
                feature_names: prepared.feature_names.clone(),
                target_name: prepared.target_name.clone(),
                objective: objective.to_owned(),
                effort,
                seed: DecimalU64(seed),
                landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(lm_wire),
                encrypted_cuen: false,
                wrapped_u_dek_b64: None,
            }
        };

        let payload = serde_json::to_vec(&req).map_err(|e| format!("encode JSON: {e}"))?;
        let bytes = self.request_checked(
            "POST",
            "/v1/carbon/runs",
            Some(tenant_id),
            Some(tier),
            Some(&payload),
        )?;
        let initial: CarbonPublicWaveProgress = serde_json::from_slice(&bytes)
            .map_err(|e| format!("decode start_run progress: {e}"))?;
        self.wait_for_run_settlement(tenant_id, &run_str, initial)
    }

    pub fn get_run_status(
        &self,
        tenant_id: &str,
        run_id: &str,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let path = format!("/v1/carbon/runs/{run_id}?tenant_id={tenant_id}");
        let bytes = self.request_checked("GET", &path, Some(tenant_id), None, None)?;
        serde_json::from_slice(&bytes).map_err(|e| format!("decode run status: {e}"))
    }

    pub fn get_run_events_sse(&self, tenant_id: &str, run_id: &str) -> Result<String, String> {
        let path = format!("/v1/carbon/runs/{run_id}/events?tenant_id={tenant_id}");
        let bytes = self.request_checked("GET", &path, Some(tenant_id), None, None)?;
        String::from_utf8(bytes).map_err(|e| format!("SSE UTF-8 error: {e}"))
    }

    pub fn signal_early_stop(
        &self,
        tenant_id: &str,
        run_id: &str,
        reason: Option<String>,
    ) -> Result<(), String> {
        let path = format!("/v1/carbon/runs/{run_id}/stop");
        let req = CarbonEarlyStopRequest {
            tenant_id: tenant_id.to_owned(),
            reason,
        };
        let payload = serde_json::to_vec(&req).map_err(|e| format!("encode JSON: {e}"))?;
        let _ = self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
        Ok(())
    }

    pub fn resume_run(
        &self,
        tenant_id: &str,
        run_id: &str,
        effort: Option<Effort>,
        additional_waves: Option<u32>,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let path = format!("/v1/carbon/runs/{run_id}/resume");
        let wrapped_u_dek_b64 = if let Some(kek_pub) = self.active_kek_pubkey() {
            let u_dek_opt = self
                .run_udeks
                .lock()
                .ok()
                .and_then(|g| g.get(&format!("{tenant_id}:{run_id}")).cloned());
            if let Some(u_dek) = u_dek_opt {
                let cudk = wrap_u_dek_to_pubkey(&u_dek, &kek_pub, tenant_id, run_id)
                    .map_err(|e| e.to_string())?;
                Some(base64::engine::general_purpose::STANDARD.encode(cudk))
            } else {
                None
            }
        } else {
            None
        };
        let req = CarbonResumeRunRequest {
            tenant_id: tenant_id.to_owned(),
            effort,
            additional_waves,
            wrapped_u_dek_b64,
        };
        let payload = serde_json::to_vec(&req).map_err(|e| format!("encode JSON: {e}"))?;
        let bytes = self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
        let initial: CarbonPublicWaveProgress =
            serde_json::from_slice(&bytes).map_err(|e| format!("decode resume progress: {e}"))?;
        self.wait_for_run_settlement(tenant_id, run_id, initial)
    }

    pub fn predict(
        &self,
        tenant_id: &str,
        run_id: &str,
        feature_names: Vec<String>,
        rows: Vec<Vec<Option<f64>>>,
    ) -> Result<CarbonPredictResponse, String> {
        let path = format!("/v1/carbon/runs/{run_id}/predict");
        let u_dek_opt = self
            .run_udeks
            .lock()
            .ok()
            .and_then(|g| g.get(&format!("{tenant_id}:{run_id}")).cloned());

        if let (Some(kek_pub), Some(u_dek)) = (self.active_kek_pubkey(), u_dek_opt) {
            let resp_kek = CarbonEngineKek::generate().map_err(|e| e.to_string())?;
            let enc_body = serde_json::to_vec(&CarbonEncryptedPredictPayload {
                feature_names,
                rows,
            })
            .map_err(|e| format!("encode encrypted predict payload: {e}"))?;
            let cuen_req = encrypt_cuen_payload(
                &u_dek,
                CuenPayloadKind::PredictFeatures,
                tenant_id,
                run_id,
                &enc_body,
            )
            .map_err(|e| e.to_string())?;
            let cudk_req = wrap_u_dek_to_pubkey(&u_dek, &kek_pub, tenant_id, run_id)
                .map_err(|e| e.to_string())?;
            let req = CarbonPredictRequest {
                tenant_id: tenant_id.to_owned(),
                feature_names: Vec::new(),
                rows: Vec::new(),
                encrypted_rows_cuen_b64: Some(
                    base64::engine::general_purpose::STANDARD.encode(cuen_req),
                ),
                client_response_pubkey_b64: Some(resp_kek.pubkey_b64()),
                wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk_req)),
            };
            let payload = serde_json::to_vec(&req).map_err(|e| format!("encode JSON: {e}"))?;
            let bytes =
                self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
            let mut resp: CarbonPredictResponse = serde_json::from_slice(&bytes)
                .map_err(|e| format!("decode predict response: {e}"))?;
            let cudk_b64 = resp
                .response_cudk_b64
                .as_deref()
                .ok_or_else(|| "missing response_cudk_b64 on encrypted predict".to_owned())?;
            let cuen_b64 = resp
                .encrypted_predictions_cuen_b64
                .as_deref()
                .ok_or_else(|| "missing encrypted_predictions_cuen_b64".to_owned())?;
            let cudk_raw = base64::engine::general_purpose::STANDARD
                .decode(cudk_b64)
                .map_err(|e| format!("decode response_cudk_b64: {e}"))?;
            let cuen_raw = base64::engine::general_purpose::STANDARD
                .decode(cuen_b64)
                .map_err(|e| format!("decode encrypted_predictions_cuen_b64: {e}"))?;
            let r_dek = unwrap_u_dek_with_kek(&resp_kek, &cudk_raw, tenant_id, run_id)
                .map_err(|e| e.to_string())?;
            let pred_le = decrypt_cuen_payload(
                &r_dek,
                CuenPayloadKind::PredictOutputs,
                tenant_id,
                run_id,
                &cuen_raw,
            )
            .map_err(|e| e.to_string())?;
            if sha256_lower_hex(&pred_le) != resp.predictions_sha256 {
                return Err("decrypted predictions SHA-256 mismatch".to_owned());
            }
            if pred_le.len() != (resp.row_count as usize) * 8 {
                return Err("decrypted predictions byte length mismatch".to_owned());
            }
            let mut decoded_preds = Vec::with_capacity(resp.row_count as usize);
            for chunk in pred_le.chunks_exact(8) {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(chunk);
                decoded_preds.push(f64::from_le_bytes(arr));
            }
            resp.predictions = decoded_preds;
            return Ok(resp);
        }

        let req = CarbonPredictRequest {
            tenant_id: tenant_id.to_owned(),
            feature_names,
            rows,
            encrypted_rows_cuen_b64: None,
            client_response_pubkey_b64: None,
            wrapped_u_dek_b64: None,
        };
        let payload = serde_json::to_vec(&req).map_err(|e| format!("encode JSON: {e}"))?;
        let bytes = self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("decode predict response: {e}"))
    }

    /// Cryptographically shred all CAS blobs and state objects for `run_id` and purge `U_DEK` from client RAM.
    pub fn shred_run(
        &self,
        tenant_id: &str,
        run_id: &str,
    ) -> Result<CarbonShredRunResponse, String> {
        let path = format!("/v1/carbon/runs/{run_id}?tenant_id={tenant_id}");
        let bytes = self.request_checked("DELETE", &path, Some(tenant_id), None, None)?;
        if let Ok(mut g) = self.run_udeks.lock() {
            g.remove(&format!("{tenant_id}:{run_id}"));
        }
        serde_json::from_slice(&bytes).map_err(|e| format!("decode shred response: {e}"))
    }

    fn wait_for_run_settlement(
        &self,
        tenant_id: &str,
        run_id: &str,
        mut progress: CarbonPublicWaveProgress,
    ) -> Result<CarbonPublicWaveProgress, String> {
        if let Some(obs) = &self.progress_observer {
            obs(&progress);
        }
        if progress.state == CarbonRunStateWire::Failed {
            return Err(format!(
                "run {run_id} failed: {}",
                progress.failure_message.unwrap_or_default()
            ));
        }
        let started = Instant::now();
        let deadline = started + Duration::from_secs(960);
        let mut sleep_ms = 60u64;
        while matches!(
            progress.state,
            CarbonRunStateWire::Queued | CarbonRunStateWire::Running
        ) {
            if Instant::now() >= deadline {
                return Err(format!("timed out waiting for async run {run_id}"));
            }
            thread::sleep(Duration::from_millis(sleep_ms));
            let cap_ms = if started.elapsed() >= Duration::from_secs(5) {
                1_000
            } else {
                500
            };
            sleep_ms = (sleep_ms * 2).min(cap_ms);
            self.poll_requests.fetch_add(1, Ordering::Relaxed);
            progress = self.get_run_status(tenant_id, run_id)?;
            if let Some(obs) = &self.progress_observer {
                obs(&progress);
            }
            if progress.state == CarbonRunStateWire::Failed {
                return Err(format!(
                    "run {run_id} failed: {}",
                    progress.failure_message.unwrap_or_default()
                ));
            }
        }
        Ok(progress)
    }

    fn resolve_bearer_header(
        &self,
        tenant_id: Option<&str>,
        requested_tier: Option<CarbonTierWire>,
    ) -> Option<String> {
        if let Some(tok) = &self.capability_token {
            return Some(format!("Bearer {}", tok.trim()));
        }
        if let Ok(tok) = std::env::var("CARBON_AUTH_TOKEN") {
            if !tok.trim().is_empty() {
                return Some(format!("Bearer {}", tok.trim()));
            }
        }
        if let (Some(tid), Ok(secret_hex)) = (tenant_id, std::env::var("CARBON_AUTH_SECRET_HEX")) {
            if let Ok(verifier) = CarbonAuthVerifier::from_hex(&secret_hex) {
                let tier = requested_tier.unwrap_or(CarbonTierWire::Pro);
                if let Ok(tok) = verifier.mint_token(&CarbonCapabilityClaims {
                    tenant_id: tid.to_owned(),
                    tier,
                    exp_epoch_secs: (now_epoch_ms() / 1_000).saturating_add(3_600),
                    admin: false,
                }) {
                    return Some(format!("Bearer {tok}"));
                }
            }
        }
        None
    }

    fn request_checked(
        &self,
        method: &str,
        path_and_query: &str,
        tenant_id: Option<&str>,
        requested_tier: Option<CarbonTierWire>,
        body: Option<&[u8]>,
    ) -> Result<Vec<u8>, String> {
        self.request_checked_with_policy(
            method,
            path_and_query,
            tenant_id,
            requested_tier,
            body,
            false,
        )
    }

    fn request_checked_with_policy(
        &self,
        method: &str,
        path_and_query: &str,
        tenant_id: Option<&str>,
        requested_tier: Option<CarbonTierWire>,
        body: Option<&[u8]>,
        allow_region_failover: bool,
    ) -> Result<Vec<u8>, String> {
        let bearer = self.resolve_bearer_header(tenant_id, requested_tier);
        let extra: Vec<(&str, &str)> = match &bearer {
            Some(b) => vec![("x-carbon-capability-token", b.as_str())],
            None => Vec::new(),
        };
        let n_endpoints = self.endpoints.len().max(1);
        let max_attempts = if allow_region_failover {
            n_endpoints.max(3)
        } else {
            3usize
        };
        let start_idx = self
            .active_idx
            .load(Ordering::Relaxed)
            .min(n_endpoints.saturating_sub(1));
        let mut last_err = String::new();
        for attempt in 0..max_attempts {
            let idx = if allow_region_failover && n_endpoints > 1 {
                (start_idx + attempt) % n_endpoints
            } else {
                start_idx
            };
            if attempt > 0 && !(allow_region_failover && attempt < n_endpoints) {
                let wait_ms = (120u64 * attempt as u64) + ((now_epoch_ms() ^ attempt as u64) % 70);
                thread::sleep(Duration::from_millis(wait_ms));
            }
            let target_base = &self.endpoints[idx];
            if let Some(b) = body {
                self.http_request_bytes
                    .fetch_add(b.len(), Ordering::Relaxed);
            }
            match send_raw_edge_http_with_agent(
                &self.agent,
                target_base,
                method,
                path_and_query,
                &extra,
                body,
            ) {
                Ok((code, resp_body)) => {
                    self.http_response_bytes
                        .fetch_add(resp_body.len(), Ordering::Relaxed);
                    if code < 300 {
                        if allow_region_failover && idx != start_idx {
                            self.active_idx.store(idx, Ordering::Relaxed);
                        }
                        return Ok(resp_body);
                    }
                    let msg = String::from_utf8_lossy(&resp_body);
                    let lc = msg.to_ascii_lowercase();
                    let crypto_rejected = lc.contains("cudk")
                        || lc.contains("cuen")
                        || lc.contains("decrypt")
                        || lc.contains("unwrap");
                    let retriable_status = code == 429
                        || ((code == 502 || code == 503 || code == 504) && !crypto_rejected);
                    last_err = format!("HTTP {code}: {msg}");
                    if !retriable_status || attempt + 1 >= max_attempts {
                        return Err(last_err);
                    }
                }
                Err(err) => {
                    last_err = err;
                    if attempt + 1 >= max_attempts {
                        return Err(last_err);
                    }
                }
            }
        }
        Err(last_err)
    }

    fn put_presigned_bytes(&self, url: &str, bytes: &[u8]) -> Result<(), String> {
        let mut last_err = String::new();
        for attempt in 0..3usize {
            if attempt > 0 {
                let wait_ms = (120u64 * attempt as u64) + ((now_epoch_ms() ^ attempt as u64) % 60);
                thread::sleep(Duration::from_millis(wait_ms));
            }
            match send_raw_put_url_with_agent(&self.agent, url, bytes, true) {
                Ok((code, body)) => {
                    if code < 300 {
                        return Ok(());
                    }
                    last_err = format!("PUT HTTP {code}: {}", String::from_utf8_lossy(&body));
                    if !matches!(code, 429 | 500 | 502 | 503 | 504) || attempt == 2 {
                        return Err(last_err);
                    }
                }
                Err(e) => {
                    last_err = e;
                    if attempt == 2 {
                        return Err(last_err);
                    }
                }
            }
        }
        Err(last_err)
    }
}

/// Send a `PUT` to a presigned S3 or gateway URL (setting `x-amz-content-sha256` when `include_sha_header` is true).
pub fn send_raw_put_url(
    url: &str,
    bytes: &[u8],
    include_sha_header: bool,
) -> Result<(u16, Vec<u8>), String> {
    let agent = crate::tls::carbon_https_agent();
    send_raw_put_url_with_agent(agent, url, bytes, include_sha_header)
}

fn send_raw_put_url_with_agent(
    agent: &ureq::Agent,
    url: &str,
    bytes: &[u8],
    include_sha_header: bool,
) -> Result<(u16, Vec<u8>), String> {
    let mut req = agent.put(url).config().http_status_as_error(false).build();
    if include_sha_header && url.contains(".s3.") && url.contains("X-Amz-Signature=") {
        let sha_hex = sha256_lower_hex(bytes);
        req = req.header("x-amz-content-sha256", &sha_hex);
    }
    let mut resp = req
        .send(bytes)
        .map_err(|e| format!("PUT {url} failed: {e}"))?;
    let code = resp.status().as_u16();
    let mut out = Vec::new();
    let _ = resp.body_mut().as_reader().read_to_end(&mut out);
    Ok((code, out))
}

/// Low-level Carbon edge HTTP/Lambda dispatcher supporting `GET`, `POST`, `PUT`, `DELETE`,
/// `AWS_IAM` SigV4 Function URLs, and `lambda://` direct Function URL v2 invocation.
pub fn send_raw_edge_http(
    base_endpoint: &str,
    method: &str,
    path_and_query: &str,
    extra_headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Result<(u16, Vec<u8>), String> {
    let agent = crate::tls::carbon_https_agent();
    send_raw_edge_http_with_agent(
        agent,
        base_endpoint,
        method,
        path_and_query,
        extra_headers,
        body,
    )
}

fn send_raw_edge_http_with_agent(
    agent: &ureq::Agent,
    base_endpoint: &str,
    method: &str,
    path_and_query: &str,
    extra_headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Result<(u16, Vec<u8>), String> {
    let base = base_endpoint.trim_end_matches('/');
    if let Some(rest) = base.strip_prefix("lambda://") {
        let (region, fn_name) = match rest.split_once('/') {
            Some((r, f)) if !r.is_empty() && !f.is_empty() => (r.to_owned(), f.to_owned()),
            _ => {
                let resolved = std::env::var("CARBON_REGION")
                    .or_else(|_| std::env::var("AWS_REGION"))
                    .ok()
                    .filter(|v| !v.trim().is_empty())
                    .unwrap_or_else(|| "us-east-2".to_owned());
                (resolved, rest.to_owned())
            }
        };
        let (raw_path, raw_query) = match path_and_query.split_once('?') {
            Some((p, q)) => (p, q),
            None => (path_and_query, ""),
        };
        let mut hdr_map = serde_json::Map::new();
        if body.is_some() {
            hdr_map.insert(
                "content-type".to_owned(),
                serde_json::Value::String("application/json".to_owned()),
            );
        }
        for &(hk, hv) in extra_headers {
            hdr_map.insert(
                hk.to_ascii_lowercase(),
                serde_json::Value::String(hv.to_owned()),
            );
        }
        let fn_url_event = match body {
            Some(b) => serde_json::json!({
                "version": "2.0",
                "rawPath": raw_path,
                "rawQueryString": raw_query,
                "headers": hdr_map,
                "requestContext": { "http": { "method": method, "path": raw_path } },
                "body": base64::engine::general_purpose::STANDARD.encode(b),
                "isBase64Encoded": true
            }),
            None => serde_json::json!({
                "version": "2.0",
                "rawPath": raw_path,
                "rawQueryString": raw_query,
                "headers": hdr_map,
                "requestContext": { "http": { "method": method, "path": raw_path } },
                "isBase64Encoded": false
            }),
        };
        let event_bytes = serde_json::to_vec(&fn_url_event)
            .map_err(|e| format!("encode Lambda Function URL v2 event: {e}"))?;
        let creds = CarbonAwsCredentials::from_env()?;
        let host = format!("lambda.{region}.amazonaws.com");
        let canonical_uri = format!(
            "/2015-03-31/functions/{}/invocations",
            url_encode_component(&fn_name)
        );
        let url = format!("https://{host}{canonical_uri}");
        let now_secs = now_epoch_ms() / 1_000;
        let headers = sign_aws_v4_headers_with_extra(
            &creds,
            &region,
            "lambda",
            "POST",
            &host,
            &canonical_uri,
            "",
            &event_bytes,
            now_secs,
            &[],
        );
        let mut req = agent
            .post(&url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("content-type", "application/json");
        for (hk, hv) in &headers {
            req = req.header(hk, hv);
        }
        let mut resp = req
            .send(&event_bytes)
            .map_err(|e| format!("Lambda edge invoke {fn_name} failed: {e}"))?;
        let invoke_code = resp.status().as_u16();
        let fn_err = resp
            .headers()
            .get("x-amz-function-error")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_owned());
        let mut out_bytes = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut out_bytes)
            .map_err(|e| format!("read Lambda edge invoke body: {e}"))?;
        if invoke_code >= 300 {
            return Err(format!(
                "Lambda invoke HTTP {invoke_code}: {}",
                String::from_utf8_lossy(&out_bytes)
            ));
        }
        if let Some(fe) = fn_err {
            return Err(format!(
                "Lambda edge function error ({fe}): {}",
                String::from_utf8_lossy(&out_bytes)
            ));
        }
        let envelope: serde_json::Value = serde_json::from_slice(&out_bytes)
            .map_err(|e| format!("decode Function URL v2 response envelope: {e}"))?;
        let status_code = envelope
            .get("statusCode")
            .and_then(|v| v.as_u64())
            .unwrap_or(200) as u16;
        let is_b64 = envelope
            .get("isBase64Encoded")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let body_str = envelope.get("body").and_then(|v| v.as_str()).unwrap_or("");
        let decoded_body = if is_b64 {
            base64::engine::general_purpose::STANDARD
                .decode(body_str)
                .map_err(|e| format!("decode base64 Lambda response body: {e}"))?
        } else {
            body_str.as_bytes().to_vec()
        };
        return Ok((status_code, decoded_body));
    }

    let url = format!("{base}{path_and_query}");
    let payload_slice = body.unwrap_or(&[]);
    let is_fn_url = base.contains(".lambda-url.");
    let signed_fn_url_headers = if is_fn_url {
        if let Ok(parsed) = url::Url::parse(&url) {
            if let Some(host) = parsed.host_str() {
                let parts: Vec<&str> = host.split('.').collect();
                let region = if parts.len() >= 3 && parts[1] == "lambda-url" {
                    parts[2]
                } else {
                    "us-east-2"
                };
                if let Ok(creds) = CarbonAwsCredentials::from_env() {
                    let query = parsed.query().unwrap_or("");
                    let now_secs = now_epoch_ms() / 1_000;
                    Some(sign_aws_v4_headers_with_extra(
                        &creds,
                        region,
                        "lambda",
                        method,
                        host,
                        parsed.path(),
                        query,
                        payload_slice,
                        now_secs,
                        extra_headers,
                    ))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let mut resp = match method {
        "GET" => {
            let mut req = agent.get(&url).config().http_status_as_error(false).build();
            if let Some(hdrs) = &signed_fn_url_headers {
                for (hk, hv) in hdrs {
                    req = req.header(hk, hv);
                }
            } else {
                for &(hk, hv) in extra_headers {
                    req = req.header(hk, hv);
                }
            }
            req.call().map_err(|e| format!("GET {url} failed: {e}"))?
        }
        "DELETE" => {
            let mut req = agent
                .delete(&url)
                .config()
                .http_status_as_error(false)
                .build();
            if let Some(hdrs) = &signed_fn_url_headers {
                for (hk, hv) in hdrs {
                    req = req.header(hk, hv);
                }
            } else {
                for &(hk, hv) in extra_headers {
                    req = req.header(hk, hv);
                }
            }
            req.call()
                .map_err(|e| format!("DELETE {url} failed: {e}"))?
        }
        "PUT" => {
            let mut req = agent
                .put(&url)
                .config()
                .http_status_as_error(false)
                .build()
                .header("content-type", "application/octet-stream");
            if let Some(hdrs) = &signed_fn_url_headers {
                for (hk, hv) in hdrs {
                    req = req.header(hk, hv);
                }
            } else {
                for &(hk, hv) in extra_headers {
                    req = req.header(hk, hv);
                }
            }
            req.send(payload_slice)
                .map_err(|e| format!("PUT {url} failed: {e}"))?
        }
        _ => {
            let mut req = agent
                .post(&url)
                .config()
                .http_status_as_error(false)
                .build()
                .header("content-type", "application/json");
            if let Some(hdrs) = &signed_fn_url_headers {
                for (hk, hv) in hdrs {
                    req = req.header(hk, hv);
                }
            } else {
                for &(hk, hv) in extra_headers {
                    req = req.header(hk, hv);
                }
            }
            req.send(payload_slice)
                .map_err(|e| format!("POST {url} failed: {e}"))?
        }
    };
    let code = resp.status().as_u16();
    let mut resp_body = Vec::new();
    resp.body_mut()
        .as_reader()
        .read_to_end(&mut resp_body)
        .map_err(|e| format!("read {method} body: {e}"))?;
    Ok((code, resp_body))
}

/// Default public HTTPS URL publishing live `dev` and `prod` Carbon cloud discovery endpoints.
pub const DEFAULT_PUBLIC_CLOUDS_URL: &str =
    "https://github.com/neverhuman/JopeDime/releases/latest/download/carbon-clouds.json";

/// Secondary HTTPS mirror serving identical release-pinned `carbon-clouds.json` discovery endpoints.
pub const FALLBACK_PUBLIC_CLOUDS_URL: &str =
    "https://downloads.neverhuman.org/carbon/carbon-clouds.json";

/// Resolve the user's `jopedime` config directory (`CARBON_CONFIG_DIR` or `$HOME/.config/jopedime`),
/// strictly refusing empty or relative paths.
pub fn verified_absolute_jopedime_config_dir() -> Result<std::path::PathBuf, String> {
    if let Ok(explicit) = std::env::var("CARBON_CONFIG_DIR") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            let p = std::path::PathBuf::from(trimmed);
            if !p.is_absolute() {
                return Err(format!(
                    "CARBON_CONFIG_DIR must be an absolute path, got '{trimmed}'"
                ));
            }
            return Ok(p);
        }
    }
    let home = std::env::var("HOME").map_err(|_| {
        "HOME environment variable is unset; pass --clouds-manifest or set absolute HOME".to_owned()
    })?;
    let home_trimmed = home.trim();
    if home_trimmed.is_empty() {
        return Err("HOME environment variable is empty; refusing relative config path".to_owned());
    }
    let home_path = std::path::PathBuf::from(home_trimmed);
    if !home_path.is_absolute() {
        return Err(format!(
            "HOME must be an absolute path, got '{home_trimmed}'"
        ));
    }
    Ok(home_path.join(".config").join("jopedime"))
}

/// `CarbonCloudRegionPin` is a struct binding one regional cell inside a named Carbon cloud
/// (`dev` or `prod`) to its stage-throttled HTTPS endpoint and pinned regional `ECDH P-256` KEK.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CarbonCloudRegionPin {
    pub region: String,
    pub public_endpoint: String,
    #[serde(default)]
    pub engine_kek_pubkey_b64: Option<String>,
    #[serde(default)]
    pub release_id: Option<String>,
}

/// `CarbonCloudCellEntry` is a struct binding a named public Carbon cloud (`dev` or `prod`) to its
/// primary and multi-region stage-throttled HTTPS endpoints and pinned `ECDH P-256` KEKs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CarbonCloudCellEntry {
    pub environment: String,
    pub region: String,
    pub public_endpoint: String,
    #[serde(default)]
    pub engine_kek_pubkey_b64: Option<String>,
    #[serde(default)]
    pub release_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<CarbonCloudRegionPin>,
}

impl CarbonCloudCellEntry {
    /// Return all candidate public regional endpoints for parallel RTT probing and failover.
    pub fn candidate_endpoints(&self) -> Vec<String> {
        let mut out = Vec::new();
        for r in &self.regions {
            let u = r.public_endpoint.trim().trim_end_matches('/').to_owned();
            if !u.is_empty() && !out.contains(&u) {
                out.push(u);
            }
        }
        let primary = self.public_endpoint.trim().trim_end_matches('/').to_owned();
        if !primary.is_empty() && !out.contains(&primary) {
            out.push(primary);
        }
        out
    }

    /// Return normalized base URL -> pinned `engine_kek_pubkey_b64` across all configured regions.
    pub fn pinned_keks_by_endpoint(&self) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        let primary = self.public_endpoint.trim().trim_end_matches('/').to_owned();
        if let Some(kek) = &self.engine_kek_pubkey_b64 {
            if !primary.is_empty() && !kek.trim().is_empty() {
                map.insert(primary, kek.trim().to_owned());
            }
        }
        for r in &self.regions {
            let url = r.public_endpoint.trim().trim_end_matches('/').to_owned();
            if let Some(kek) = &r.engine_kek_pubkey_b64 {
                if !url.is_empty() && !kek.trim().is_empty() {
                    map.insert(url, kek.trim().to_owned());
                }
            }
        }
        map
    }
}

/// `CarbonPublicCloudsManifest` is a struct mapping named Carbon clouds (`dev`, `prod`) for
/// zero-credential client discovery (`--cloud dev` / `--cloud prod`).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CarbonPublicCloudsManifest {
    pub schema: String,
    pub generated_epoch_ms: u64,
    pub clouds: BTreeMap<String, CarbonCloudCellEntry>,
}

fn fetch_clouds_manifest_bytes(source: &str) -> Result<Vec<u8>, String> {
    if source.starts_with("https://") || source.starts_with("http://") {
        let mut resp = crate::tls::carbon_https_agent()
            .get(source)
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .map_err(|e| format!("fetch Carbon clouds manifest {source} failed: {e}"))?;
        let status = resp.status().as_u16();
        let mut buf = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut buf)
            .map_err(|e| format!("read Carbon clouds manifest {source}: {e}"))?;
        if status >= 300 {
            return Err(format!(
                "Carbon clouds manifest {source} returned HTTP {status}: {}",
                String::from_utf8_lossy(&buf)
            ));
        }
        Ok(buf)
    } else {
        std::fs::read(source)
            .map_err(|e| format!("read local Carbon clouds manifest {source}: {e}"))
    }
}

/// Resolve a named public Carbon cloud (`dev`, `staging`, `prod`) from an explicit file/URL
/// override, `CARBON_CLOUDS_URL`, local `$HOME/.config/jopedime/carbon-clouds.json`, or
/// [`DEFAULT_PUBLIC_CLOUDS_URL`] (with automatic fallback to [`FALLBACK_PUBLIC_CLOUDS_URL`])
/// without AWS credentials.
pub fn resolve_public_cloud_endpoint(
    cloud_name: &str,
    manifest_override: Option<&str>,
) -> Result<CarbonCloudCellEntry, String> {
    let env_override = std::env::var("CARBON_CLOUDS_URL")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let local_cached = verified_absolute_jopedime_config_dir()
        .ok()
        .map(|d| d.join("carbon-clouds.json"))
        .filter(|p| p.is_file())
        .map(|p| p.to_string_lossy().into_owned());
    let explicit_source = manifest_override
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or(env_override.as_deref())
        .or(local_cached.as_deref());

    let (source_used, raw_bytes) = if let Some(src) = explicit_source {
        let b = fetch_clouds_manifest_bytes(src)?;
        (src.to_owned(), b)
    } else {
        match fetch_clouds_manifest_bytes(DEFAULT_PUBLIC_CLOUDS_URL) {
            Ok(b) => (DEFAULT_PUBLIC_CLOUDS_URL.to_owned(), b),
            Err(primary_err) => match fetch_clouds_manifest_bytes(FALLBACK_PUBLIC_CLOUDS_URL) {
                Ok(b) => (FALLBACK_PUBLIC_CLOUDS_URL.to_owned(), b),
                Err(fallback_err) => {
                    return Err(format!("{primary_err}; mirror fallback: {fallback_err}"));
                }
            },
        }
    };
    let source = source_used.as_str();

    let manifest: CarbonPublicCloudsManifest = serde_json::from_slice(&raw_bytes)
        .map_err(|e| format!("parse Carbon clouds manifest {source}: {e}"))?;
    let key = cloud_name.trim().to_ascii_lowercase();
    let entry = manifest.clouds.get(&key).cloned().ok_or_else(|| {
        let available: Vec<&str> = manifest.clouds.keys().map(String::as_str).collect();
        format!(
            "cloud '{key}' not found in {source} (available: {})",
            available.join(", ")
        )
    })?;
    if let Some(kek_b64) = &entry.engine_kek_pubkey_b64 {
        let _ = decode_p256_pubkey_b64(kek_b64)
            .map_err(|e| format!("invalid pinned engine_kek_pubkey_b64 for '{key}': {e}"))?;
    }
    for r in &entry.regions {
        if let Some(kek_b64) = &r.engine_kek_pubkey_b64 {
            let _ = decode_p256_pubkey_b64(kek_b64).map_err(|e| {
                format!(
                    "invalid pinned engine_kek_pubkey_b64 for '{key}' region '{}': {e}",
                    r.region
                )
            })?;
        }
    }
    Ok(entry)
}

/// `CarbonStoredCloudCredential` is a struct persisting a user's `ctk1.*` capability token for one
/// named Carbon cloud (`dev` or `prod`) inside `0600` `$HOME/.config/jopedime/carbon-credentials.json`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CarbonStoredCloudCredential {
    pub cloud: String,
    pub endpoint: String,
    pub tenant_id: String,
    pub tier: jope_api::carbon::CarbonTierWire,
    pub exp_epoch_secs: u64,
    pub capability_token: String,
}

impl std::fmt::Debug for CarbonStoredCloudCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonStoredCloudCredential")
            .field("cloud", &self.cloud)
            .field("endpoint", &self.endpoint)
            .field("tenant_id", &self.tenant_id)
            .field("tier", &self.tier)
            .field("exp_epoch_secs", &self.exp_epoch_secs)
            .field("capability_token", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
struct CarbonCredentialsStoreFile {
    schema: String,
    clouds: BTreeMap<String, CarbonStoredCloudCredential>,
}

fn write_private_credentials_file(
    config_dir: &std::path::Path,
    store: &CarbonCredentialsStoreFile,
) -> Result<(), String> {
    std::fs::create_dir_all(config_dir)
        .map_err(|e| format!("create config dir {}: {e}", config_dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(config_dir, std::fs::Permissions::from_mode(0o700));
    }
    let final_path = config_dir.join("carbon-credentials.json");
    let tmp_path = config_dir.join(format!(".carbon-credentials.{}.tmp", std::process::id()));
    let encoded = zeroize::Zeroizing::new(
        serde_json::to_vec_pretty(store).map_err(|e| format!("encode credentials store: {e}"))?,
    );
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp_path)
            .map_err(|e| format!("open {}: {e}", tmp_path.display()))?;
        f.write_all(&encoded)
            .map_err(|e| format!("write {}: {e}", tmp_path.display()))?;
        f.sync_all()
            .map_err(|e| format!("fsync {}: {e}", tmp_path.display()))?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(&tmp_path, &*encoded)
            .map_err(|e| format!("write {}: {e}", tmp_path.display()))?;
    }
    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| format!("activate {}: {e}", final_path.display()))?;
    Ok(())
}

/// Validate and store a `ctk1.*` capability token for `cloud_name` inside the `0600` credentials store.
pub fn save_stored_cloud_token(
    cloud_name: &str,
    endpoint: &str,
    token: &str,
) -> Result<crate::security::auth::CarbonCapabilityClaims, String> {
    let trimmed_tok = token.trim();
    let claims = crate::security::auth::CarbonCapabilityClaims::decode_unverified(trimmed_tok)
        .map_err(|e| e.to_string())?;
    let config_dir = verified_absolute_jopedime_config_dir()?;
    let cred_path = config_dir.join("carbon-credentials.json");
    let mut store = if cred_path.is_file() {
        let raw = zeroize::Zeroizing::new(
            std::fs::read(&cred_path).map_err(|e| format!("read {}: {e}", cred_path.display()))?,
        );
        serde_json::from_slice::<CarbonCredentialsStoreFile>(&raw)
            .map_err(|e| format!("parse {}: {e}", cred_path.display()))?
    } else {
        CarbonCredentialsStoreFile {
            schema: "jope-carbon-credentials-v1".to_owned(),
            clouds: BTreeMap::new(),
        }
    };
    let key = cloud_name.trim().to_ascii_lowercase();
    store.schema = "jope-carbon-credentials-v1".to_owned();
    store.clouds.insert(
        key.clone(),
        CarbonStoredCloudCredential {
            cloud: key,
            endpoint: endpoint.trim().to_owned(),
            tenant_id: claims.tenant_id.clone(),
            tier: claims.tier,
            exp_epoch_secs: claims.exp_epoch_secs,
            capability_token: trimmed_tok.to_owned(),
        },
    );
    write_private_credentials_file(&config_dir, &store)?;
    Ok(claims)
}

/// Load a previously stored `ctk1.*` capability token for `cloud_name`, if present.
pub fn load_stored_cloud_token(
    cloud_name: &str,
) -> Result<Option<CarbonStoredCloudCredential>, String> {
    let config_dir = verified_absolute_jopedime_config_dir()?;
    let cred_path = config_dir.join("carbon-credentials.json");
    if !cred_path.is_file() {
        return Ok(None);
    }
    let raw = zeroize::Zeroizing::new(
        std::fs::read(&cred_path).map_err(|e| format!("read {}: {e}", cred_path.display()))?,
    );
    let store: CarbonCredentialsStoreFile =
        serde_json::from_slice(&raw).map_err(|e| format!("parse {}: {e}", cred_path.display()))?;
    let key = cloud_name.trim().to_ascii_lowercase();
    Ok(store.clouds.get(&key).cloned())
}

/// Load all stored cloud credentials from `carbon-credentials.json` (empty map when absent).
pub fn list_stored_cloud_tokens() -> Result<BTreeMap<String, CarbonStoredCloudCredential>, String> {
    let config_dir = verified_absolute_jopedime_config_dir()?;
    let cred_path = config_dir.join("carbon-credentials.json");
    if !cred_path.is_file() {
        return Ok(BTreeMap::new());
    }
    let raw = zeroize::Zeroizing::new(
        std::fs::read(&cred_path).map_err(|e| format!("read {}: {e}", cred_path.display()))?,
    );
    let store: CarbonCredentialsStoreFile =
        serde_json::from_slice(&raw).map_err(|e| format!("parse {}: {e}", cred_path.display()))?;
    Ok(store.clouds)
}

/// Remove stored capability token(s) (`Some(cloud)` removes one cloud; `None` clears all clouds).
pub fn remove_stored_cloud_token(cloud_name: Option<&str>) -> Result<usize, String> {
    let config_dir = verified_absolute_jopedime_config_dir()?;
    let cred_path = config_dir.join("carbon-credentials.json");
    if !cred_path.is_file() {
        return Ok(0);
    }
    let raw = zeroize::Zeroizing::new(
        std::fs::read(&cred_path).map_err(|e| format!("read {}: {e}", cred_path.display()))?,
    );
    let mut store: CarbonCredentialsStoreFile =
        serde_json::from_slice(&raw).map_err(|e| format!("parse {}: {e}", cred_path.display()))?;
    let removed = match cloud_name {
        Some(c) => usize::from(
            store
                .clouds
                .remove(&c.trim().to_ascii_lowercase())
                .is_some(),
        ),
        None => {
            let n = store.clouds.len();
            store.clouds.clear();
            n
        }
    };
    if store.clouds.is_empty() {
        let _ = std::fs::remove_file(&cred_path);
    } else {
        write_private_credentials_file(&config_dir, &store)?;
    }
    Ok(removed)
}
