//! Zero-knowledge thin client SDK for Carbon regional Lambda cells and Enterprise `carbon serve`.
//!
//! Performs deterministic `FIT` / `HOLDOUT` splitting, `FIT`-only landmark conditioning (`JLMB` v1),
//! `JMAT` v1 encoding, and per-user `ECDH P-256 + HKDF-SHA256 + AES-256-GCM` (`CUEN\x01` / `CUDK\x01`)
//! envelope encryption strictly on the client before uploading to size/SHA-pinned presigned URLs.

use crate::error::CarbonErrCtx as _;
use std::collections::BTreeMap;
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
        CarbonExplainRequest, CarbonExplainResponse, CarbonExportLangWire, CarbonExportRequest,
        CarbonExportResponse, CarbonPingResponse, CarbonPredictRequest, CarbonPredictResponse,
        CarbonPreparePredictRequest, CarbonPreparePredictResponse, CarbonPrepareUploadRequest,
        CarbonPrepareUploadResponse, CarbonPublicWaveProgress, CarbonRegionCatalog,
        CarbonRegisterFreeRequest, CarbonRegisterFreeResponse, CarbonResumeRunRequest,
        CarbonRunStateWire, CarbonShredRunResponse, CarbonTierWire, CarbonUploadSlotWire,
        CarbonVersionResponse,
    },
    decimal::DecimalU64,
    ids::{DatasetId, RunId},
    Effort,
};
use jope_prep::{cuen::CuenPayloadKind, FitLandmarkBundle, SplitJmatArtifacts, SplitPlan};
use zeroize::Zeroizing;

use crate::{
    cells::engine_invoker::now_epoch_ms,
    role::CarbonProductTier,
    security::{
        auth::{CarbonAuthVerifier, CarbonCapabilityClaims},
        sigv4::{
            compute_backoff_with_jitter_ms, sign_aws_v4_headers_with_extra, CarbonAwsCredentials,
        },
        user_envelope::{
            decode_p256_pubkey_b64, decrypt_cuen_payload, encrypt_cuen_payload,
            generate_random_u_dek, unwrap_u_dek_with_kek, wrap_u_dek_to_pubkey,
            CarbonEncryptedExplainPayload, CarbonEncryptedPredictPayload, CarbonEngineKek,
            CAS_BLOB_CUEN_SCOPE,
        },
    },
    storage::cas::sha256_lower_hex,
};

const PREDICT_CAS_THRESHOLD_BYTES: usize = 1_572_864;

fn decode_wire_ping_response(bytes: &[u8]) -> Result<CarbonPingResponse, String> {
    serde_json::from_slice(bytes).err_ctx("invalid ping JSON")
}

fn is_strict_loopback_host(host: &str) -> bool {
    let clean = host.trim().trim_start_matches('[').trim_end_matches(']');
    if clean.eq_ignore_ascii_case("localhost") {
        return true;
    }
    clean
        .parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

/// Return `true` iff `raw_url` targets strict local loopback (`127.0.0.0/8`, `::1`, or `localhost`).
pub fn is_loopback_endpoint_url(raw_url: &str) -> bool {
    url::Url::parse(raw_url.trim())
        .ok()
        .and_then(|u| u.host_str().map(is_strict_loopback_host))
        .unwrap_or(false)
}

/// Validate that `raw_url` uses TLS (`https://` or SigV4 `lambda://`), permitting cleartext
/// `http://` exclusively for strict local loopback (`127.0.0.1`, `::1`, `localhost`).
pub fn validate_endpoint_transport(raw_url: &str) -> Result<(), String> {
    let trimmed = raw_url.trim();
    if trimmed.starts_with("lambda://") {
        return Ok(());
    }
    let parsed =
        url::Url::parse(trimmed).map_err(|e| format!("invalid endpoint URL '{trimmed}': {e}"))?;
    match parsed.scheme() {
        "https" => Ok(()),
        "http" => {
            let host = parsed.host_str().unwrap_or_default();
            if is_strict_loopback_host(host) {
                Ok(())
            } else {
                Err(format!(
                    "refusing insecure cleartext HTTP endpoint '{trimmed}'; use https:// (http:// is permitted only for local loopback 127.0.0.1, ::1, or localhost)"
                ))
            }
        }
        other => Err(format!(
            "unsupported endpoint URL scheme '{other}' in '{trimmed}' (expected https://)"
        )),
    }
}

fn parse_major_minor_semver(raw: &str) -> Option<(u32, u32)> {
    let clean = raw
        .trim()
        .trim_start_matches("carbon-cli-v")
        .trim_start_matches('v');
    let mut parts = clean.split(['.', '-']);
    let major = parts.next()?.parse::<u32>().ok()?;
    let minor = parts.next()?.parse::<u32>().ok()?;
    Some((major, minor))
}

fn server_supports_v120_wire(ver: Option<&CarbonVersionResponse>) -> bool {
    let Some(v) = ver else {
        return false;
    };
    parse_major_minor_semver(&v.version)
        .or_else(|| parse_major_minor_semver(&v.release_tag))
        .is_some_and(|(maj, min)| (maj, min) >= (1, 2))
}

#[derive(Clone)]
struct UploadedEncryptionSecret {
    u_dek: Zeroizing<[u8; 32]>,
    upload_id: String,
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
    pub aux_column_names: Vec<String>,
    pub split_mode: Option<String>,
    pub calibration_mode: Option<String>,
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
    allow_unencrypted_enterprise: bool,
    endpoint_keks: Arc<Mutex<BTreeMap<String, [u8; 65]>>>,
    pinned_keks: BTreeMap<String, String>,
    capability_token: Option<Zeroizing<String>>,
    uploaded_secrets: Arc<Mutex<BTreeMap<String, UploadedEncryptionSecret>>>,
    uploaded_ids: Arc<Mutex<BTreeMap<String, String>>>,
    run_udeks: Arc<Mutex<BTreeMap<String, Zeroizing<[u8; 32]>>>>,
    run_upload_keys: Arc<Mutex<BTreeMap<String, String>>>,
    cached_server_version: Arc<Mutex<Option<Option<CarbonVersionResponse>>>>,
    last_release_id: Arc<Mutex<Option<String>>>,
    poll_requests: Arc<AtomicU32>,
    http_request_bytes: Arc<AtomicUsize>,
    http_response_bytes: Arc<AtomicUsize>,
    progress_observer: Option<CarbonProgressObserver>,
    agent: ureq::Agent,
}

impl std::fmt::Debug for CarbonThinClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonThinClient")
            .field("endpoints", &self.endpoints)
            .field("active_idx", &self.active_idx.load(Ordering::Relaxed))
            .field("enterprise_self_hosted", &self.enterprise_self_hosted)
            .field(
                "capability_token",
                &self.capability_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

fn probe_single_endpoint(
    ag: &ureq::Agent,
    base_url: String,
    expected_kek: Option<String>,
) -> Result<CarbonProbedEndpoint, String> {
    let max_attempts = if is_loopback_endpoint_url(&base_url) {
        1usize
    } else {
        3usize
    };
    let mut last_err = String::new();
    for attempt in 0..max_attempts {
        if attempt > 0 {
            let backoff_ms = compute_backoff_with_jitter_ms(attempt - 1, 150, 800);
            thread::sleep(Duration::from_millis(backoff_ms));
        }
        let t0 = Instant::now();
        let (code, buf) = match send_raw_edge_http_with_agent(
            ag,
            &base_url,
            "GET",
            "/v1/carbon/ping",
            &[("x-carbon-client-version", crate::CARBON_PKG_VERSION)],
            None,
        ) {
            Ok(pair) => pair,
            Err(e) => {
                last_err = format!("{base_url}: {e}");
                continue;
            }
        };
        if code == 429 || code >= 500 {
            last_err = format!("{base_url}: /v1/carbon/ping returned HTTP {code}");
            continue;
        }
        if code != 200 {
            return Err(format!("{base_url}: /v1/carbon/ping returned HTTP {code}"));
        }
        let ping = decode_wire_ping_response(&buf).map_err(|e| format!("{base_url}: {e}"))?;
        if !ping.healthy {
            last_err = format!("{base_url}: reported unhealthy");
            continue;
        }
        if let Some(ref pin_b64) = expected_kek {
            if ping.engine_kek_pubkey_b64.as_deref() != Some(pin_b64.as_str()) {
                return Err(format!(
                    "{base_url}: engine_kek_pubkey_b64 mismatch against pinned manifest"
                ));
            }
        }
        return Ok(CarbonProbedEndpoint {
            base_url,
            region: ping.region,
            rtt_ms: t0.elapsed().as_secs_f64() * 1_000.0,
            enterprise_self_hosted: ping.enterprise_self_hosted,
            engine_kek_pubkey_b64: ping.engine_kek_pubkey_b64,
        });
    }
    Err(last_err)
}

impl CarbonThinClient {
    /// Construct a zero-knowledge `CarbonThinClient` directly from a verified model handle's
    /// pinned `endpoint` and `engine_kek_pubkey_b64` with zero `/v1/carbon/ping` round-trip.
    pub fn from_pinned_handle(endpoint: String, kek_pubkey_b64: &str) -> Result<Self, String> {
        let trimmed = endpoint.trim().trim_end_matches('/').to_owned();
        if trimmed.is_empty() {
            return Err("CarbonThinClient requires at least one endpoint URL".to_owned());
        }
        validate_endpoint_transport(&trimmed)?;
        let kek_trimmed = kek_pubkey_b64.trim().to_owned();
        let decoded = decode_p256_pubkey_b64(&kek_trimmed).map_err(crate::error::err_to_string)?;
        let mut endpoint_keks = BTreeMap::new();
        endpoint_keks.insert(trimmed.clone(), decoded);
        let mut pinned_keks = BTreeMap::new();
        pinned_keks.insert(trimmed.clone(), kek_trimmed);
        Ok(Self {
            endpoints: vec![trimmed],
            active_idx: Arc::new(AtomicUsize::new(0)),
            enterprise_self_hosted: false,
            allow_unencrypted_enterprise: false,
            endpoint_keks: Arc::new(Mutex::new(endpoint_keks)),
            pinned_keks,
            capability_token: None,
            uploaded_secrets: Arc::new(Mutex::new(BTreeMap::new())),
            uploaded_ids: Arc::new(Mutex::new(BTreeMap::new())),
            run_udeks: Arc::new(Mutex::new(BTreeMap::new())),
            run_upload_keys: Arc::new(Mutex::new(BTreeMap::new())),
            cached_server_version: Arc::new(Mutex::new(None)),
            last_release_id: Arc::new(Mutex::new(None)),
            poll_requests: Arc::new(AtomicU32::new(0)),
            http_request_bytes: Arc::new(AtomicUsize::new(0)),
            http_response_bytes: Arc::new(AtomicUsize::new(0)),
            progress_observer: None,
            agent: crate::tls::carbon_https_agent().clone(),
        })
    }

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
        Self::connect_candidates(candidate_urls, pinned_keks, false)
    }

    /// Hedged regional `/v1/carbon/ping` race: returns within a `<= 35 ms` grace window after the
    /// fastest healthy KEK-verified cell responds (instead of waiting on distant continents),
    /// while preserving all remaining candidate URLs for automatic regional failover.
    pub fn connect_fastest_hedged(
        candidate_urls: Vec<String>,
        pinned_keks: &BTreeMap<String, String>,
    ) -> Result<(Self, Vec<CarbonProbedEndpoint>), String> {
        Self::connect_candidates(candidate_urls, pinned_keks, true)
    }

    fn connect_candidates(
        candidate_urls: Vec<String>,
        pinned_keks: &BTreeMap<String, String>,
        early_return: bool,
    ) -> Result<(Self, Vec<CarbonProbedEndpoint>), String> {
        let mut deduped = Vec::new();
        for raw_url in candidate_urls {
            let trimmed = raw_url.trim().trim_end_matches('/').to_owned();
            if !trimmed.is_empty() && !deduped.contains(&trimmed) {
                validate_endpoint_transport(&trimmed)?;
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
        let probe_agent = crate::tls::carbon_probe_https_agent().clone();
        let use_grace_race = early_return
            && deduped.len() > 1
            && deduped.iter().all(|u| !is_loopback_endpoint_url(u));
        let probe_outcomes: Vec<Result<CarbonProbedEndpoint, String>> = if use_grace_race {
            let (tx, rx) = std::sync::mpsc::channel();
            for base in &deduped {
                let base_url = base.clone();
                let ag = probe_agent.clone();
                let expected_kek = normalized_pins.get(&base_url).cloned();
                let tx_i = tx.clone();
                thread::spawn(move || {
                    let _ = tx_i.send(probe_single_endpoint(&ag, base_url, expected_kek));
                });
            }
            drop(tx);
            let mut collected = Vec::with_capacity(deduped.len());
            let mut grace_deadline: Option<Instant> = None;
            for _ in 0..deduped.len() {
                let next_msg = if let Some(dl) = grace_deadline {
                    let rem = dl.saturating_duration_since(Instant::now());
                    if rem.is_zero() {
                        break;
                    }
                    rx.recv_timeout(rem).ok()
                } else {
                    rx.recv().ok()
                };
                let Some(outcome) = next_msg else {
                    break;
                };
                if grace_deadline.is_none() {
                    if let Ok(ref probed) = outcome {
                        let grace_ms = (probed.rtt_ms * 0.35).clamp(15.0, 35.0) as u64;
                        grace_deadline = Some(Instant::now() + Duration::from_millis(grace_ms));
                    }
                }
                collected.push(outcome);
            }
            collected
        } else {
            thread::scope(|s| {
                let mut handles = Vec::with_capacity(deduped.len());
                for base in &deduped {
                    let base_url = base.clone();
                    let ag = probe_agent.clone();
                    let expected_kek = normalized_pins.get(&base_url).cloned();
                    handles
                        .push(s.spawn(move || probe_single_endpoint(&ag, base_url, expected_kek)));
                }
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            Err("regional ping probe thread panicked".to_owned())
                        })
                    })
                    .collect()
            })
        };

        let mut healthy = Vec::new();
        let mut errors = Vec::new();
        let mut endpoint_keks = BTreeMap::new();
        for outcome in probe_outcomes {
            match outcome {
                Ok(probed) => {
                    if let Some(b64) = &probed.engine_kek_pubkey_b64 {
                        let decoded =
                            decode_p256_pubkey_b64(b64).map_err(crate::error::err_to_string)?;
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
        let mut ordered_endpoints: Vec<String> =
            healthy.iter().map(|h| h.base_url.clone()).collect();
        for fallback_ep in &deduped {
            if !ordered_endpoints.contains(fallback_ep) {
                ordered_endpoints.push(fallback_ep.clone());
            }
        }
        Ok((
            Self {
                endpoints: ordered_endpoints,
                active_idx: Arc::new(AtomicUsize::new(0)),
                enterprise_self_hosted,
                allow_unencrypted_enterprise: false,
                endpoint_keks: Arc::new(Mutex::new(endpoint_keks)),
                pinned_keks: normalized_pins,
                capability_token: None,
                uploaded_secrets: Arc::new(Mutex::new(BTreeMap::new())),
                uploaded_ids: Arc::new(Mutex::new(BTreeMap::new())),
                run_udeks: Arc::new(Mutex::new(BTreeMap::new())),
                run_upload_keys: Arc::new(Mutex::new(BTreeMap::new())),
                cached_server_version: Arc::new(Mutex::new(None)),
                last_release_id: Arc::new(Mutex::new(None)),
                poll_requests: Arc::new(AtomicU32::new(0)),
                http_request_bytes: Arc::new(AtomicUsize::new(0)),
                http_response_bytes: Arc::new(AtomicUsize::new(0)),
                progress_observer: None,
                agent,
            },
            healthy,
        ))
    }

    /// Explicitly opt into unencrypted dataset uploads when connecting to a private VPC
    /// `carbon serve` deployment configured without `--enable-user-envelope`.
    pub fn with_allow_unencrypted_enterprise(mut self, allow: bool) -> Self {
        self.allow_unencrypted_enterprise = allow;
        self
    }

    /// Attach an observer invoked on initial run creation/resume and on each async poll update.
    pub fn with_progress_observer<F>(mut self, observer: F) -> Self
    where
        F: Fn(&CarbonPublicWaveProgress) + Send + Sync + 'static,
    {
        self.progress_observer = Some(Arc::new(observer));
        self
    }

    /// Return `(run_udeks_len, uploaded_secrets_len, uploaded_ids_len)` currently retained in client RAM.
    pub fn cached_secret_counts(&self) -> (usize, usize, usize) {
        let udeks = self.run_udeks.lock().map(|g| g.len()).unwrap_or(0);
        let secrets = self.uploaded_secrets.lock().map(|g| g.len()).unwrap_or(0);
        let ids = self.uploaded_ids.lock().map(|g| g.len()).unwrap_or(0);
        (udeks, secrets, ids)
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
            self.capability_token = Some(Zeroizing::new(tok));
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
        let resp: CarbonAnonymousAuthResponse =
            serde_json::from_slice(&bytes).err_ctx("decode CarbonAnonymousAuthResponse")?;
        self.record_verified_endpoint_kek(
            self.active_endpoint(),
            resp.engine_kek_pubkey_b64.as_deref(),
        )?;
        self.capability_token = Some(Zeroizing::new(resp.capability_token.clone()));
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
        .err_ctx("encode CarbonRegisterFreeRequest")?;
        let bytes = self.request_checked_with_policy(
            "POST",
            "/v1/carbon/auth/register",
            None,
            None,
            Some(&req_body),
            true,
        )?;
        let resp: CarbonRegisterFreeResponse =
            serde_json::from_slice(&bytes).err_ctx("decode CarbonRegisterFreeResponse")?;
        self.record_verified_endpoint_kek(
            self.active_endpoint(),
            resp.engine_kek_pubkey_b64.as_deref(),
        )?;
        self.capability_token = Some(Zeroizing::new(resp.capability_token.clone()));
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
            let decoded = decode_p256_pubkey_b64(b64).map_err(crate::error::err_to_string)?;
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

    /// Return the base64-encoded uncompressed P-256 KEK public key of the active endpoint, if present.
    pub fn active_kek_pubkey_b64(&self) -> Option<String> {
        self.active_kek_pubkey()
            .map(|raw| base64::engine::general_purpose::STANDARD.encode(raw))
    }

    /// Return the currently attached `ctk1.*` or `ctk2.*` capability token string on this client, if any.
    pub fn active_capability_token(&self) -> Option<String> {
        self.capability_token
            .as_ref()
            .map(|z| z.as_str().to_owned())
    }

    /// Return the most recent `x-carbon-release-id` header (or `/v1/carbon/version` `release_id`)
    /// observed from the active Carbon cell, if advertised.
    pub fn active_release_id(&self) -> Option<String> {
        if let Ok(guard) = self.last_release_id.lock() {
            if let Some(rel) = &*guard {
                return Some(rel.clone());
            }
        }
        self.cached_server_version.lock().ok().and_then(|g| {
            g.as_ref()
                .and_then(|opt| opt.as_ref().and_then(|v| v.release_id.clone()))
        })
    }

    /// Export the 32-byte per-run zero-knowledge `u_dek` held in memory for `(tenant_id, run_id)`.
    pub fn export_run_udek(&self, tenant_id: &str, run_id: &str) -> Option<[u8; 32]> {
        self.run_udeks
            .lock()
            .ok()
            .and_then(|g| g.get(&format!("{tenant_id}:{run_id}")).map(|z| **z))
    }

    /// Import a 32-byte per-run zero-knowledge `u_dek` from a saved model handle for subsequent `/predict`.
    pub fn import_run_udek(&self, tenant_id: &str, run_id: &str, u_dek: [u8; 32]) {
        if let Ok(mut guard) = self.run_udeks.lock() {
            guard.insert(format!("{tenant_id}:{run_id}"), Zeroizing::new(u_dek));
        }
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
        let resp = decode_wire_ping_response(&bytes).err_ctx("decode CarbonPingResponse")?;
        self.record_verified_endpoint_kek(
            self.active_endpoint(),
            resp.engine_kek_pubkey_b64.as_deref(),
        )?;
        Ok(resp)
    }

    /// Fetch remote `/v1/carbon/version` build provenance from the active endpoint (`Ok(None)` on
    /// legacy `v1.0.0` gateways that do not yet expose `/v1/carbon/version`), with bounded 3-attempt
    /// cold-start retry resilience and per-client caching.
    pub fn fetch_server_version(&self) -> Result<Option<CarbonVersionResponse>, String> {
        if let Ok(guard) = self.cached_server_version.lock() {
            if let Some(cached) = &*guard {
                return Ok(cached.clone());
            }
        }
        let ep = self.active_endpoint();
        let max_attempts = if is_loopback_endpoint_url(ep) {
            1usize
        } else {
            3usize
        };
        let mut last_err = String::new();
        for attempt in 0..max_attempts {
            if attempt > 0 {
                let wait_ms = compute_backoff_with_jitter_ms(attempt - 1, 150, 800);
                thread::sleep(Duration::from_millis(wait_ms));
            }
            let (code, body) = match send_raw_edge_http_with_agent(
                crate::tls::carbon_probe_https_agent(),
                ep,
                "GET",
                "/v1/carbon/version",
                &[("x-carbon-client-version", crate::CARBON_PKG_VERSION)],
                None,
            ) {
                Ok(pair) => pair,
                Err(e) => {
                    last_err = e;
                    continue;
                }
            };
            if code == 404 || code == 405 {
                if let Ok(mut guard) = self.cached_server_version.lock() {
                    *guard = Some(None);
                }
                return Ok(None);
            }
            if code == 429 || code >= 500 {
                last_err = format!(
                    "GET /v1/carbon/version returned HTTP {code}: {}",
                    String::from_utf8_lossy(&body)
                );
                continue;
            }
            if code >= 300 {
                return Err(format!(
                    "GET /v1/carbon/version returned HTTP {code}: {}",
                    String::from_utf8_lossy(&body)
                ));
            }
            let ver: CarbonVersionResponse =
                serde_json::from_slice(&body).err_ctx("decode CarbonVersionResponse")?;
            if let Ok(mut guard) = self.cached_server_version.lock() {
                *guard = Some(Some(ver.clone()));
            }
            return Ok(Some(ver));
        }
        Err(last_err)
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
        serde_json::from_slice(&bytes).err_ctx("decode CarbonRegionCatalog")
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
        Self::prepare_numeric_dataset_with_options(
            headers,
            numeric_rows,
            target_name,
            &[],
            seed,
            holdout_fraction,
            None,
            None,
        )
    }

    /// Prepare a numeric dataset with optional auxiliary fitness columns (`aux_column_names`
    /// encoded into `fit.jmat`/`holdout.jmat` for objective scoring while strictly excluded from
    /// candidate feature matrix `X` and `FitLandmarkBundle`), `split_mode` (`"random"` | `"forward"`),
    /// and `calibration_mode` (`"auto"` | `"none"` | `"shift"` | `"affine"`).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_numeric_dataset_with_options(
        headers: Vec<String>,
        numeric_rows: &[Vec<f64>],
        target_name: &str,
        aux_column_names: &[String],
        seed: u64,
        holdout_fraction: f64,
        split_mode: Option<&str>,
        calibration_mode: Option<&str>,
    ) -> Result<CarbonClientPreparedDataset, String> {
        let target_idx = headers
            .iter()
            .position(|h| h == target_name)
            .ok_or_else(|| format!("target column '{target_name}' not found in headers"))?;
        let mut aux_indices = Vec::with_capacity(aux_column_names.len());
        for aux_name in aux_column_names {
            if aux_name == target_name {
                return Err(format!(
                    "auxiliary fitness column '{aux_name}' cannot also be the target column"
                ));
            }
            let idx = headers.iter().position(|h| h == aux_name).ok_or_else(|| {
                format!("auxiliary fitness column '{aux_name}' not found in dataset headers")
            })?;
            if !aux_indices.contains(&idx) {
                aux_indices.push(idx);
            }
        }
        let feature_indices: Vec<usize> = (0..headers.len())
            .filter(|i| *i != target_idx && !aux_indices.contains(i))
            .collect();
        if feature_indices.is_empty() {
            return Err("dataset must have at least one feature column".to_owned());
        }
        let feature_names: Vec<String> = feature_indices
            .iter()
            .map(|&i| headers[i].clone())
            .collect();
        let total_rows = numeric_rows.len();
        let normalized_split = match split_mode.map(str::trim).filter(|s| !s.is_empty()) {
            None | Some("random") => None,
            Some("forward") => Some("forward".to_owned()),
            Some(other) => {
                return Err(format!(
                    "unsupported split mode '{other}' (expected 'random' or 'forward')"
                ));
            }
        };
        let normalized_cal = match calibration_mode.map(str::trim).filter(|s| !s.is_empty()) {
            None | Some("auto") => None,
            Some(m @ ("none" | "shift" | "affine")) => Some(m.to_owned()),
            Some(other) => {
                return Err(format!(
                    "unsupported calibration mode '{other}' (expected 'auto', 'none', 'shift', or 'affine')"
                ));
            }
        };
        let split_plan = if normalized_split.as_deref() == Some("forward") {
            SplitPlan::chronological(total_rows, seed, holdout_fraction)
                .err_ctx("chronological split plan failed")?
        } else {
            SplitPlan::deterministic(total_rows, seed, holdout_fraction)
                .err_ctx("split plan failed")?
        };

        let mut full_x = Vec::with_capacity(total_rows * feature_indices.len());
        let mut full_y = Vec::with_capacity(total_rows);
        for (r_idx, row) in numeric_rows.iter().enumerate() {
            if row.len() != headers.len() {
                return Err(format!("row {r_idx} column count mismatch"));
            }
            for &fi in &feature_indices {
                full_x.push(row[fi] as f32);
            }
            full_y.push(row[target_idx] as f32);
        }

        let landmark_bundle = FitLandmarkBundle::extract_fit_only(
            &full_x,
            total_rows,
            feature_indices.len(),
            &full_y,
            &split_plan,
        )
        .err_ctx("FIT-only landmark conditioning failed")?;

        let artifacts = jope_prep::jmat_lite::split_and_encode_numeric_jmat_v1(
            &headers,
            numeric_rows,
            &split_plan,
        )
        .err_ctx("JMAT split encoding failed")?;

        Ok(CarbonClientPreparedDataset {
            fit_rows: split_plan.fit_rows().len() as u32,
            holdout_rows: split_plan.holdout_rows().len() as u32,
            split_plan,
            landmark_bundle,
            artifacts,
            feature_names,
            target_name: target_name.to_owned(),
            aux_column_names: aux_indices
                .into_iter()
                .map(|i| headers[i].clone())
                .collect(),
            split_mode: normalized_split,
            calibration_mode: normalized_cal,
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
        if !self.enterprise_self_hosted {
            let product_tier = match tier {
                CarbonTierWire::Free => CarbonProductTier::Free,
                CarbonTierWire::Pro => CarbonProductTier::Pro,
                CarbonTierWire::Enterprise => CarbonProductTier::no_limit(),
            };
            if let Some(max_slot) = product_tier.max_upload_slot_bytes() {
                let raw_max = prepared
                    .artifacts
                    .fit_jmat
                    .len()
                    .max(prepared.artifacts.holdout_jmat.len())
                    as u64;
                if raw_max > max_slot {
                    return Err(format!(
                        "HTTP 413 pre-flight guard: prepared dataset slot ({raw_max} bytes) exceeds {tier:?} tier upload cap ({max_slot} bytes)"
                    ));
                }
            }
        }
        if !prepared.aux_column_names.is_empty()
            || prepared.split_mode.is_some()
            || prepared.calibration_mode.is_some()
        {
            let remote_ver = self.fetch_server_version()?;
            if !server_supports_v120_wire(remote_ver.as_ref()) {
                let ver_label = remote_ver
                    .as_ref()
                    .map(|v| format!("{} ({})", v.version, v.release_tag))
                    .unwrap_or_else(|| "v1.0.0".to_owned());
                return Err(format!(
                    "endpoint '{}' is running Carbon server {ver_label} (< 1.2.0), which does not yet support v1.2.0 --split / --calibration / col(...) auxiliary fitness columns",
                    self.active_endpoint()
                ));
            }
        }
        if !self.has_zero_knowledge_kek()
            && !is_loopback_endpoint_url(self.active_endpoint())
            && !self.allow_unencrypted_enterprise
        {
            return Err(format!(
                "refusing unencrypted dataset upload to non-loopback endpoint '{}' because server did not advertise a P-256 zero-knowledge KEK (enable --enable-user-envelope on `carbon serve` or pass --allow-unencrypted-enterprise)",
                self.active_endpoint()
            ));
        }
        let pt_fit_sha = hex::encode(prepared.artifacts.fit_sha256);
        let (fit_wire, fit_sha, holdout_wire, holdout_sha, encrypted_cuen, mut secret_opt) =
            if self.has_zero_knowledge_kek() {
                let u_dek = generate_random_u_dek().map_err(crate::error::err_to_string)?;
                let fit_enc = encrypt_cuen_payload(
                    &u_dek,
                    CuenPayloadKind::FitJmat,
                    tenant_id,
                    CAS_BLOB_CUEN_SCOPE,
                    &prepared.artifacts.fit_jmat,
                )
                .map_err(crate::error::err_to_string)?;
                let hold_enc = encrypt_cuen_payload(
                    &u_dek,
                    CuenPayloadKind::HoldoutJmat,
                    tenant_id,
                    CAS_BLOB_CUEN_SCOPE,
                    &prepared.artifacts.holdout_jmat,
                )
                .map_err(crate::error::err_to_string)?;
                let f_sha = sha256_lower_hex(&fit_enc);
                let h_sha = sha256_lower_hex(&hold_enc);
                let secret = UploadedEncryptionSecret {
                    u_dek,
                    upload_id: String::new(),
                    fit_cuen_sha256: f_sha.clone(),
                    holdout_cuen_sha256: h_sha.clone(),
                };
                (
                    std::borrow::Cow::Owned(fit_enc),
                    f_sha,
                    std::borrow::Cow::Owned(hold_enc),
                    h_sha,
                    true,
                    Some(secret),
                )
            } else {
                let f_sha = pt_fit_sha.clone();
                let h_sha = hex::encode(prepared.artifacts.holdout_sha256);
                (
                    std::borrow::Cow::Borrowed(prepared.artifacts.fit_jmat.as_slice()),
                    f_sha,
                    std::borrow::Cow::Borrowed(prepared.artifacts.holdout_jmat.as_slice()),
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
        let payload = serde_json::to_vec(&prep_req).err_ctx("encode JSON")?;
        let resp_bytes = self.request_checked_with_policy(
            "POST",
            "/v1/carbon/uploads/prepare",
            Some(tenant_id),
            Some(tier),
            Some(&payload),
            true,
        )?;
        let prep_resp: CarbonPrepareUploadResponse =
            serde_json::from_slice(&resp_bytes).err_ctx("decode upload prepare response")?;
        self.record_verified_endpoint_kek(
            self.active_endpoint(),
            prep_resp.engine_kek_pubkey_b64.as_deref(),
        )?;

        thread::scope(|s| -> Result<(), String> {
            let fit_put =
                s.spawn(|| self.put_presigned_bytes(&prep_resp.fit_target.url, fit_wire.as_ref()));
            let hold_res =
                self.put_presigned_bytes(&prep_resp.holdout_target.url, holdout_wire.as_ref());
            let fit_res = fit_put
                .join()
                .map_err(|_| "fit S3 PUT worker thread panicked".to_owned())?;
            fit_res?;
            hold_res
        })?;
        let upload_key = format!("{tenant_id}:{pt_fit_sha}");
        if let Ok(mut id_guard) = self.uploaded_ids.lock() {
            while id_guard.len() >= 128 {
                let _ = id_guard.pop_first();
            }
            id_guard.insert(upload_key.clone(), prep_resp.upload_id.clone());
        }
        if let Some(ref mut secret) = secret_opt {
            secret.upload_id = prep_resp.upload_id.clone();
            if let Ok(mut guard) = self.uploaded_secrets.lock() {
                while guard.len() >= 128 {
                    let _ = guard.pop_first();
                }
                guard.insert(upload_key, secret.clone());
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
        let upload_key = format!("{tenant_id}:{pt_fit_sha}");
        let lm_wire = prepared.landmark_bundle.to_wire_bytes();
        // Legacy v1.0.0 gateways expose neither `x-carbon-release-id` nor `/v1/carbon/version`
        // and reject `upload_id` via `#[serde(deny_unknown_fields)]`.
        let v1_1_wire =
            self.active_release_id().is_some() || self.fetch_server_version()?.is_some();

        let req = if let Some(kek_pub) = self.active_kek_pubkey() {
            let secret = self
                .uploaded_secrets
                .lock()
                .map_err(|_| "uploaded_secrets mutex poisoned".to_owned())?
                .get(&upload_key)
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
            .map_err(crate::error::err_to_string)?;
            let cudk = wrap_u_dek_to_pubkey(&secret.u_dek, &kek_pub, tenant_id, &run_str)
                .map_err(crate::error::err_to_string)?;
            let run_key = format!("{tenant_id}:{run_str}");
            if let Ok(mut run_map) = self.run_udeks.lock() {
                while run_map.len() >= 128 {
                    let _ = run_map.pop_first();
                }
                run_map.insert(run_key.clone(), secret.u_dek.clone());
            }
            if let Ok(mut ukey_map) = self.run_upload_keys.lock() {
                while ukey_map.len() >= 128 {
                    let _ = ukey_map.pop_first();
                }
                ukey_map.insert(run_key, upload_key.clone());
            }
            CarbonCreateRunRequest {
                run_id,
                dataset_id,
                tenant_id: tenant_id.to_owned(),
                tier,
                upload_id: if !v1_1_wire || secret.upload_id.is_empty() {
                    None
                } else {
                    Some(secret.upload_id)
                },
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
                aux_column_names: prepared.aux_column_names.clone(),
                split_mode: prepared.split_mode.clone(),
                calibration_mode: prepared.calibration_mode.clone(),
            }
        } else {
            if let Ok(mut ukey_map) = self.run_upload_keys.lock() {
                while ukey_map.len() >= 128 {
                    let _ = ukey_map.pop_first();
                }
                ukey_map.insert(format!("{tenant_id}:{run_str}"), upload_key.clone());
            }
            let plain_upload_id = if v1_1_wire {
                self.uploaded_ids
                    .lock()
                    .ok()
                    .and_then(|g| g.get(&upload_key).cloned())
            } else {
                None
            };
            CarbonCreateRunRequest {
                run_id,
                dataset_id,
                tenant_id: tenant_id.to_owned(),
                tier,
                upload_id: plain_upload_id,
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
                aux_column_names: prepared.aux_column_names.clone(),
                split_mode: prepared.split_mode.clone(),
                calibration_mode: prepared.calibration_mode.clone(),
            }
        };

        let payload = serde_json::to_vec(&req).err_ctx("encode JSON")?;
        let bytes = self.request_checked(
            "POST",
            "/v1/carbon/runs",
            Some(tenant_id),
            Some(tier),
            Some(&payload),
        )?;
        let initial: CarbonPublicWaveProgress =
            serde_json::from_slice(&bytes).err_ctx("decode start_run progress")?;
        self.wait_for_run_settlement(tenant_id, &run_str, initial)
    }

    pub fn get_run_status(
        &self,
        tenant_id: &str,
        run_id: &str,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let path = format!("/v1/carbon/runs/{run_id}?tenant_id={tenant_id}");
        let bytes = self.request_checked("GET", &path, Some(tenant_id), None, None)?;
        serde_json::from_slice(&bytes).err_ctx("decode run status")
    }

    pub fn get_run_events_sse(&self, tenant_id: &str, run_id: &str) -> Result<String, String> {
        let path = format!("/v1/carbon/runs/{run_id}/events?tenant_id={tenant_id}");
        let bytes = self.request_checked("GET", &path, Some(tenant_id), None, None)?;
        String::from_utf8(bytes).err_ctx("SSE UTF-8 error")
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
        let payload = serde_json::to_vec(&req).err_ctx("encode JSON")?;
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
                    .map_err(crate::error::err_to_string)?;
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
        let payload = serde_json::to_vec(&req).err_ctx("encode JSON")?;
        let bytes = self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
        let initial: CarbonPublicWaveProgress =
            serde_json::from_slice(&bytes).err_ctx("decode resume progress")?;
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
            let resp_kek = CarbonEngineKek::generate().map_err(crate::error::err_to_string)?;
            let enc_body = serde_json::to_vec(&CarbonEncryptedPredictPayload {
                feature_names,
                rows,
            })
            .err_ctx("encode encrypted predict payload")?;
            let cuen_req = encrypt_cuen_payload(
                &u_dek,
                CuenPayloadKind::PredictFeatures,
                tenant_id,
                run_id,
                &enc_body,
            )
            .map_err(crate::error::err_to_string)?;
            let cudk_req = wrap_u_dek_to_pubkey(&u_dek, &kek_pub, tenant_id, run_id)
                .map_err(crate::error::err_to_string)?;

            let force_cas = std::env::var("CARBON_FORCE_PREDICT_CAS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let req = if cuen_req.len() > PREDICT_CAS_THRESHOLD_BYTES || force_cas {
                let prep_path = format!("/v1/carbon/runs/{run_id}/predict/prepare");
                let prep_req = CarbonPreparePredictRequest {
                    tenant_id: tenant_id.to_owned(),
                    predict_cuen: CarbonUploadSlotWire {
                        sha256: sha256_lower_hex(&cuen_req),
                        bytes: DecimalU64(cuen_req.len() as u64),
                    },
                };
                let prep_payload = serde_json::to_vec(&prep_req).err_ctx("encode JSON")?;
                let prep_bytes = self.request_checked(
                    "POST",
                    &prep_path,
                    Some(tenant_id),
                    None,
                    Some(&prep_payload),
                )?;
                let prep_resp: CarbonPreparePredictResponse =
                    serde_json::from_slice(&prep_bytes)
                        .err_ctx("decode predict prepare response")?;
                self.put_presigned_bytes(&prep_resp.input_target.url, &cuen_req)?;
                CarbonPredictRequest {
                    tenant_id: tenant_id.to_owned(),
                    feature_names: Vec::new(),
                    rows: Vec::new(),
                    encrypted_rows_cuen_b64: None,
                    encrypted_rows_cas_sha256: Some(prep_resp.input_target.sha256),
                    predict_upload_id: Some(prep_resp.upload_id),
                    prefer_cas_output: true,
                    client_response_pubkey_b64: Some(resp_kek.pubkey_b64()),
                    wrapped_u_dek_b64: Some(
                        base64::engine::general_purpose::STANDARD.encode(cudk_req),
                    ),
                }
            } else {
                CarbonPredictRequest {
                    tenant_id: tenant_id.to_owned(),
                    feature_names: Vec::new(),
                    rows: Vec::new(),
                    encrypted_rows_cuen_b64: Some(
                        base64::engine::general_purpose::STANDARD.encode(cuen_req),
                    ),
                    encrypted_rows_cas_sha256: None,
                    predict_upload_id: None,
                    prefer_cas_output: false,
                    client_response_pubkey_b64: Some(resp_kek.pubkey_b64()),
                    wrapped_u_dek_b64: Some(
                        base64::engine::general_purpose::STANDARD.encode(cudk_req),
                    ),
                }
            };

            let payload = serde_json::to_vec(&req).err_ctx("encode JSON")?;
            let bytes =
                self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
            let mut resp: CarbonPredictResponse =
                serde_json::from_slice(&bytes).err_ctx("decode predict response")?;
            let cudk_b64 = resp
                .response_cudk_b64
                .as_deref()
                .ok_or_else(|| "missing response_cudk_b64 on encrypted predict".to_owned())?;
            let cudk_raw = base64::engine::general_purpose::STANDARD
                .decode(cudk_b64)
                .err_ctx("decode response_cudk_b64")?;
            let cuen_raw = if let Some(cas_url) = resp.encrypted_predictions_cas_url.as_deref() {
                let exp_sha = resp
                    .encrypted_predictions_cas_sha256
                    .as_deref()
                    .unwrap_or("");
                self.get_presigned_bytes(cas_url, exp_sha)?
            } else {
                let cuen_b64 = resp
                    .encrypted_predictions_cuen_b64
                    .as_deref()
                    .ok_or_else(|| {
                        "missing encrypted_predictions_cuen_b64 on encrypted predict".to_owned()
                    })?;
                base64::engine::general_purpose::STANDARD
                    .decode(cuen_b64)
                    .err_ctx("decode encrypted_predictions_cuen_b64")?
            };
            let r_dek = unwrap_u_dek_with_kek(&resp_kek, &cudk_raw, tenant_id, run_id)
                .map_err(crate::error::err_to_string)?;
            let pred_le = decrypt_cuen_payload(
                &r_dek,
                CuenPayloadKind::PredictOutputs,
                tenant_id,
                run_id,
                &cuen_raw,
            )
            .map_err(crate::error::err_to_string)?;
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
            encrypted_rows_cas_sha256: None,
            predict_upload_id: None,
            prefer_cas_output: false,
            client_response_pubkey_b64: None,
            wrapped_u_dek_b64: None,
        };
        let payload = serde_json::to_vec(&req).err_ctx("encode JSON")?;
        let bytes = self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
        serde_json::from_slice(&bytes).err_ctx("decode predict response")
    }

    /// Compute holdout permutation feature importances (`POST /v1/carbon/runs/{id}/explain`) with
    /// end-to-end `CUEN\x01(ExplainOutputs)` encryption when connected to a zero-knowledge cell.
    pub fn explain_run(
        &self,
        tenant_id: &str,
        run_id: &str,
        top_k: Option<u32>,
    ) -> Result<CarbonExplainResponse, String> {
        let path = format!("/v1/carbon/runs/{run_id}/explain");
        let u_dek_opt = self
            .run_udeks
            .lock()
            .ok()
            .and_then(|g| g.get(&format!("{tenant_id}:{run_id}")).cloned());

        if let (Some(kek_pub), Some(u_dek)) = (self.active_kek_pubkey(), u_dek_opt) {
            let resp_kek = CarbonEngineKek::generate().map_err(crate::error::err_to_string)?;
            let cudk_req = wrap_u_dek_to_pubkey(&u_dek, &kek_pub, tenant_id, run_id)
                .map_err(crate::error::err_to_string)?;
            let req = CarbonExplainRequest {
                tenant_id: tenant_id.to_owned(),
                top_k,
                client_response_pubkey_b64: Some(resp_kek.pubkey_b64()),
                wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk_req)),
            };
            let payload = serde_json::to_vec(&req).err_ctx("encode JSON")?;
            let bytes =
                self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
            let mut resp: CarbonExplainResponse =
                serde_json::from_slice(&bytes).err_ctx("decode explain response")?;
            let cudk_b64 = resp
                .response_cudk_b64
                .as_deref()
                .ok_or_else(|| "missing response_cudk_b64 on encrypted explain".to_owned())?;
            let cuen_b64 = resp.encrypted_explain_cuen_b64.as_deref().ok_or_else(|| {
                "missing encrypted_explain_cuen_b64 on encrypted explain".to_owned()
            })?;
            let cudk_raw = base64::engine::general_purpose::STANDARD
                .decode(cudk_b64)
                .err_ctx("decode response_cudk_b64")?;
            let cuen_raw = base64::engine::general_purpose::STANDARD
                .decode(cuen_b64)
                .err_ctx("decode encrypted_explain_cuen_b64")?;
            let r_dek = unwrap_u_dek_with_kek(&resp_kek, &cudk_raw, tenant_id, run_id)
                .map_err(crate::error::err_to_string)?;
            let pt = decrypt_cuen_payload(
                &r_dek,
                CuenPayloadKind::ExplainOutputs,
                tenant_id,
                run_id,
                &cuen_raw,
            )
            .map_err(crate::error::err_to_string)?;
            let decoded: CarbonEncryptedExplainPayload =
                serde_json::from_slice(&pt).err_ctx("decode decrypted explain payload")?;
            resp.baseline_holdout_loss = decoded.baseline_holdout_loss;
            resp.feature_importances = decoded.feature_importances;
            return Ok(resp);
        }

        let req = CarbonExplainRequest {
            tenant_id: tenant_id.to_owned(),
            top_k,
            client_response_pubkey_b64: None,
            wrapped_u_dek_b64: None,
        };
        let payload = serde_json::to_vec(&req).err_ctx("encode JSON")?;
        let bytes = self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
        serde_json::from_slice(&bytes).err_ctx("decode explain response")
    }

    /// Export a run's winning champion to standalone predictor source code (`POST /v1/carbon/runs/{id}/export`)
    /// with end-to-end `CUEN\x01(ExportArtifact)` encryption in zero-knowledge cloud mode.
    pub fn export_run(
        &self,
        tenant_id: &str,
        run_id: &str,
        lang: CarbonExportLangWire,
    ) -> Result<CarbonExportResponse, String> {
        self.export_run_with_options(tenant_id, run_id, lang, false)
    }

    /// Export a run's winning champion with explicit CAS spill preference (`prefer_cas_output`)
    /// and `CZIP\x01` transparent compression inside `CUEN\x01(ExportArtifact)`.
    pub fn export_run_with_options(
        &self,
        tenant_id: &str,
        run_id: &str,
        lang: CarbonExportLangWire,
        prefer_cas_output: bool,
    ) -> Result<CarbonExportResponse, String> {
        let path = format!("/v1/carbon/runs/{run_id}/export");
        let u_dek_opt = self
            .run_udeks
            .lock()
            .ok()
            .and_then(|g| g.get(&format!("{tenant_id}:{run_id}")).cloned());

        if let (Some(kek_pub), Some(u_dek)) = (self.active_kek_pubkey(), u_dek_opt) {
            let resp_kek = CarbonEngineKek::generate().map_err(crate::error::err_to_string)?;
            let cudk_req = wrap_u_dek_to_pubkey(&u_dek, &kek_pub, tenant_id, run_id)
                .map_err(crate::error::err_to_string)?;
            let req = CarbonExportRequest {
                tenant_id: tenant_id.to_owned(),
                lang,
                accept_compressed: true,
                prefer_cas_output,
                client_response_pubkey_b64: Some(resp_kek.pubkey_b64()),
                wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk_req)),
            };
            let payload = serde_json::to_vec(&req).err_ctx("encode JSON")?;
            let bytes =
                self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
            let mut resp: CarbonExportResponse =
                serde_json::from_slice(&bytes).err_ctx("decode export response")?;
            let cudk_b64 = resp
                .response_cudk_b64
                .as_deref()
                .ok_or_else(|| "missing response_cudk_b64 on encrypted export".to_owned())?;
            let cudk_raw = base64::engine::general_purpose::STANDARD
                .decode(cudk_b64)
                .err_ctx("decode response_cudk_b64")?;
            let cuen_raw = if let Some(cas_url) = resp
                .encrypted_export_cas_url
                .as_deref()
                .filter(|u| !u.is_empty())
            {
                let exp_sha = resp.encrypted_export_cas_sha256.as_deref().unwrap_or("");
                self.get_presigned_bytes(cas_url, exp_sha)?
            } else {
                let cuen_b64 = resp.encrypted_export_cuen_b64.as_deref().ok_or_else(|| {
                    "missing encrypted_export_cuen_b64 on encrypted export".to_owned()
                })?;
                base64::engine::general_purpose::STANDARD
                    .decode(cuen_b64)
                    .err_ctx("decode encrypted_export_cuen_b64")?
            };
            let r_dek = unwrap_u_dek_with_kek(&resp_kek, &cudk_raw, tenant_id, run_id)
                .map_err(crate::error::err_to_string)?;
            let raw_pt = decrypt_cuen_payload(
                &r_dek,
                CuenPayloadKind::ExportArtifact,
                tenant_id,
                run_id,
                &cuen_raw,
            )
            .map_err(crate::error::err_to_string)?;
            let pt = crate::export_codec::decompress_export_payload_bounded(&raw_pt)?;
            if sha256_lower_hex(&pt) != resp.source_sha256 {
                return Err("decrypted export source SHA-256 mismatch".to_owned());
            }
            let src_str =
                String::from_utf8(pt.to_vec()).err_ctx("decrypted export source is not UTF-8")?;
            resp.source_code = Some(src_str);
            return Ok(resp);
        }

        let req = CarbonExportRequest {
            tenant_id: tenant_id.to_owned(),
            lang,
            accept_compressed: true,
            prefer_cas_output,
            client_response_pubkey_b64: None,
            wrapped_u_dek_b64: None,
        };
        let payload = serde_json::to_vec(&req).err_ctx("encode JSON")?;
        let bytes = self.request_checked("POST", &path, Some(tenant_id), None, Some(&payload))?;
        serde_json::from_slice(&bytes).err_ctx("decode export response")
    }

    /// Cryptographically shred all CAS blobs and state objects for `run_id` and purge `U_DEK` and upload secrets from client RAM.
    pub fn shred_run(
        &self,
        tenant_id: &str,
        run_id: &str,
    ) -> Result<CarbonShredRunResponse, String> {
        self.shred_run_with_upload(tenant_id, run_id, None, None, None)
    }

    /// Cryptographically shred all CAS blobs and state objects for `run_id`, including explicit
    /// staged `upload_id` / `fit_sha256` / `holdout_sha256` cleanup when a run failed before
    /// `request.json` was committed on the gateway.
    pub fn shred_run_with_upload(
        &self,
        tenant_id: &str,
        run_id: &str,
        upload_id: Option<&str>,
        fit_sha256: Option<&str>,
        holdout_sha256: Option<&str>,
    ) -> Result<CarbonShredRunResponse, String> {
        let mut qs = url::form_urlencoded::Serializer::new(String::new());
        qs.append_pair("tenant_id", tenant_id);
        if let Some(uid) = upload_id.map(str::trim).filter(|s| !s.is_empty()) {
            qs.append_pair("upload_id", uid);
        }
        if let Some(fsha) = fit_sha256.map(str::trim).filter(|s| !s.is_empty()) {
            qs.append_pair("fit_sha256", fsha);
        }
        if let Some(hsha) = holdout_sha256.map(str::trim).filter(|s| !s.is_empty()) {
            qs.append_pair("holdout_sha256", hsha);
        }
        let path = format!("/v1/carbon/runs/{run_id}?{}", qs.finish());
        let bytes = self.request_checked("DELETE", &path, Some(tenant_id), None, None)?;
        let run_key = format!("{tenant_id}:{run_id}");
        if let Ok(mut g) = self.run_udeks.lock() {
            g.remove(&run_key);
        }
        let released_upload_key = self.run_upload_keys.lock().ok().and_then(|mut g| {
            let ukey = g.remove(&run_key)?;
            let still_used = g.values().any(|v| v == &ukey);
            if still_used {
                None
            } else {
                Some(ukey)
            }
        });
        if let Some(ukey) = released_upload_key {
            if let Ok(mut s) = self.uploaded_secrets.lock() {
                s.remove(&ukey);
            }
            if let Ok(mut ids) = self.uploaded_ids.lock() {
                ids.remove(&ukey);
            }
        }
        serde_json::from_slice(&bytes).err_ctx("decode shred response")
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
        let mut sleep_ms = 45u64;
        let mut sent_sigint_stop = false;
        while matches!(
            progress.state,
            CarbonRunStateWire::Queued | CarbonRunStateWire::Running
        ) {
            if !sent_sigint_stop && crate::observability::sigint_received() {
                sent_sigint_stop = true;
                let _ = self.signal_early_stop(tenant_id, run_id, Some("cli_sigint".to_owned()));
            }
            if Instant::now() >= deadline {
                return Err(format!("timed out waiting for async run {run_id}"));
            }
            thread::sleep(Duration::from_millis(sleep_ms));
            if !sent_sigint_stop && crate::observability::sigint_received() {
                sent_sigint_stop = true;
                let _ = self.signal_early_stop(tenant_id, run_id, Some("cli_sigint".to_owned()));
            }
            let cap_ms = if started.elapsed() >= Duration::from_secs(5) {
                1_000
            } else {
                400
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
            Some(b) => vec![
                ("x-carbon-capability-token", b.as_str()),
                ("x-carbon-client-version", crate::CARBON_PKG_VERSION),
            ],
            None => vec![("x-carbon-client-version", crate::CARBON_PKG_VERSION)],
        };
        let n_endpoints = self.endpoints.len().max(1);
        let max_attempts = if allow_region_failover {
            n_endpoints.max(6)
        } else {
            6usize
        };
        let start_idx = self
            .active_idx
            .load(Ordering::Relaxed)
            .min(n_endpoints.saturating_sub(1));
        let mut last_err = String::new();
        let mut retry_after_override_ms: Option<u64> = None;
        for attempt in 0..max_attempts {
            let idx = if allow_region_failover && n_endpoints > 1 {
                (start_idx + attempt) % n_endpoints
            } else {
                start_idx
            };
            if attempt > 0 && !(allow_region_failover && attempt < n_endpoints) {
                let wait_ms = retry_after_override_ms
                    .take()
                    .unwrap_or_else(|| compute_backoff_with_jitter_ms(attempt - 1, 100, 4_000));
                thread::sleep(Duration::from_millis(wait_ms));
            } else {
                retry_after_override_ms = None;
            }
            let target_base = &self.endpoints[idx];
            if let Some(b) = body {
                self.http_request_bytes
                    .fetch_add(b.len(), Ordering::Relaxed);
            }
            match send_raw_edge_http_with_headers_and_agent(
                &self.agent,
                target_base,
                method,
                path_and_query,
                &extra,
                body,
            ) {
                Ok((code, resp_headers, resp_body)) => {
                    self.http_response_bytes
                        .fetch_add(resp_body.len(), Ordering::Relaxed);
                    if let Some(rel_hdr) = resp_headers
                        .get("x-carbon-release-id")
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                    {
                        if let Ok(mut guard) = self.last_release_id.lock() {
                            *guard = Some(rel_hdr.to_owned());
                        }
                    }
                    if code < 300 {
                        if allow_region_failover && idx != start_idx {
                            self.active_idx.store(idx, Ordering::Relaxed);
                        }
                        return Ok(resp_body);
                    }
                    let raw_msg = String::from_utf8_lossy(&resp_body);
                    let msg = crate::observability::redact_secrets(&raw_msg);
                    let lc = msg.to_ascii_lowercase();
                    let crypto_rejected = lc.contains("cudk")
                        || lc.contains("cuen")
                        || lc.contains("decrypt")
                        || lc.contains("unwrap");
                    let permanent_quota = lc.contains("predict_quota_exceeded");
                    let retriable_status = !permanent_quota
                        && (code == 408
                            || code == 429
                            || ((code == 502 || code == 503 || code == 504) && !crypto_rejected));
                    if code == 429 {
                        retry_after_override_ms = resp_headers
                            .get("retry-after")
                            .and_then(|v| v.trim().parse::<u64>().ok())
                            .map(|secs| secs.clamp(1, 15).saturating_mul(1_000));
                    }
                    let req_id_note = resp_headers
                        .get("x-carbon-request-id")
                        .map(|rid| format!(" [request_id={rid}]"))
                        .unwrap_or_default();
                    last_err = format!("HTTP {code}{req_id_note}: {msg}");
                    if !retriable_status || attempt + 1 >= max_attempts {
                        return Err(last_err);
                    }
                }
                Err(err) => {
                    last_err = crate::observability::redact_secrets(&err);
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
        for attempt in 0..6usize {
            if attempt > 0 {
                let wait_ms = compute_backoff_with_jitter_ms(attempt - 1, 100, 4_000);
                thread::sleep(Duration::from_millis(wait_ms));
            }
            match send_raw_put_url_with_agent(&self.agent, url, bytes, true) {
                Ok((code, body)) => {
                    if code < 300 {
                        return Ok(());
                    }
                    last_err = format!("PUT HTTP {code}: {}", String::from_utf8_lossy(&body));
                    if !matches!(code, 408 | 429 | 500 | 502 | 503 | 504) || attempt == 5 {
                        return Err(last_err);
                    }
                }
                Err(e) => {
                    last_err = e;
                    if attempt == 5 {
                        return Err(last_err);
                    }
                }
            }
        }
        Err(last_err)
    }

    fn get_presigned_bytes(&self, url: &str, expected_sha256: &str) -> Result<Vec<u8>, String> {
        let mut last_err = String::new();
        for attempt in 0..6usize {
            if attempt > 0 {
                let wait_ms = compute_backoff_with_jitter_ms(attempt - 1, 100, 4_000);
                thread::sleep(Duration::from_millis(wait_ms));
            }
            match send_raw_get_url_with_agent(&self.agent, url) {
                Ok((code, body)) => {
                    if code < 300 {
                        if !expected_sha256.is_empty()
                            && sha256_lower_hex(&body) != expected_sha256.to_ascii_lowercase()
                        {
                            return Err(
                                "downloaded CAS prediction blob SHA-256 mismatch".to_owned()
                            );
                        }
                        return Ok(body);
                    }
                    last_err = format!("GET HTTP {code}: {}", String::from_utf8_lossy(&body));
                    if !matches!(code, 408 | 429 | 500 | 502 | 503 | 504) || attempt == 5 {
                        return Err(last_err);
                    }
                }
                Err(e) => {
                    last_err = e;
                    if attempt == 5 {
                        return Err(last_err);
                    }
                }
            }
        }
        Err(last_err)
    }
}

fn send_raw_get_url_with_agent(agent: &ureq::Agent, url: &str) -> Result<(u16, Vec<u8>), String> {
    if url.starts_with("lambda://") {
        // Split `lambda://fn/v1/carbon/cas/...` into base `lambda://fn` and path `/v1/...`.
        let rest = url.trim_start_matches("lambda://");
        if let Some((first, tail)) = rest.split_once("/v1/") {
            let base = format!("lambda://{first}");
            let pq = format!("/v1/{tail}");
            return send_raw_edge_http_with_agent(agent, &base, "GET", &pq, &[], None);
        }
    }
    validate_endpoint_transport(url)?;
    if url.contains(".lambda-url.") {
        if let Ok(parsed) = url::Url::parse(url) {
            if let Some(host) = parsed.host_str() {
                let base = format!("{}://{host}", parsed.scheme());
                let pq = match parsed.query() {
                    Some(q) => format!("{}?{q}", parsed.path()),
                    None => parsed.path().to_owned(),
                };
                return send_raw_edge_http_with_agent(agent, &base, "GET", &pq, &[], None);
            }
        }
    }
    let mut resp = agent
        .get(url)
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .map_err(|e| format!("GET {url} failed: {e}"))?;
    let code = resp.status().as_u16();
    let out = crate::storage::cas::read_bounded(
        resp.body_mut().as_reader(),
        268_435_456,
        &format!("GET {url} body"),
    )?;
    Ok((code, out))
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
    validate_endpoint_transport(url)?;
    let mut req = agent.put(url).config().http_status_as_error(false).build();
    if include_sha_header
        && (url.contains(".s3.") || url.contains("x-amz-content-sha256"))
        && url.contains("X-Amz-Signature=")
    {
        let sha_hex = sha256_lower_hex(bytes);
        req = req.header("x-amz-content-sha256", &sha_hex);
    }
    let mut resp = req
        .send(bytes)
        .map_err(|e| format!("PUT {url} failed: {e}"))?;
    let code = resp.status().as_u16();
    let out =
        crate::storage::cas::read_bounded(resp.body_mut().as_reader(), 1024 * 1024, "PUT response")
            .unwrap_or_default();
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

/// Raw HTTP response tuple `(status_code, lowercase_headers, body_bytes)`.
pub type RawEdgeHttpResponse = (u16, BTreeMap<String, String>, Vec<u8>);

/// Low-level Carbon edge HTTP/Lambda dispatcher returning `(status, lowercase_headers, body_bytes)`.
pub fn send_raw_edge_http_with_headers(
    base_endpoint: &str,
    method: &str,
    path_and_query: &str,
    extra_headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Result<RawEdgeHttpResponse, String> {
    let agent = crate::tls::carbon_https_agent();
    send_raw_edge_http_with_headers_and_agent(
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
    let (code, _headers, body_out) = send_raw_edge_http_with_headers_and_agent(
        agent,
        base_endpoint,
        method,
        path_and_query,
        extra_headers,
        body,
    )?;
    Ok((code, body_out))
}

fn send_raw_edge_http_with_headers_and_agent(
    agent: &ureq::Agent,
    base_endpoint: &str,
    method: &str,
    path_and_query: &str,
    extra_headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> Result<RawEdgeHttpResponse, String> {
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
        let event_bytes =
            serde_json::to_vec(&fn_url_event).err_ctx("encode Lambda Function URL v2 event")?;
        let creds = CarbonAwsCredentials::from_env()?;
        let (host, canonical_uri, canonical_query, url) =
            crate::security::sigv4::resolve_lambda_invoke_endpoint(&region, &fn_name)?;
        let now_secs = now_epoch_ms() / 1_000;
        let headers = sign_aws_v4_headers_with_extra(
            &creds,
            &region,
            "lambda",
            "POST",
            &host,
            &canonical_uri,
            &canonical_query,
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
        let out_bytes = crate::storage::cas::read_bounded(
            resp.body_mut().as_reader(),
            268_435_456,
            "Lambda edge invoke body",
        )?;
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
            .err_ctx("decode Function URL v2 response envelope")?;
        let status_code = envelope
            .get("statusCode")
            .and_then(|v| v.as_u64())
            .unwrap_or(200) as u16;
        let mut resp_headers = BTreeMap::new();
        if let Some(hobj) = envelope.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in hobj {
                if let Some(s) = v.as_str() {
                    resp_headers.insert(k.to_ascii_lowercase(), s.to_owned());
                }
            }
        }
        let is_b64 = envelope
            .get("isBase64Encoded")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let body_str = envelope.get("body").and_then(|v| v.as_str()).unwrap_or("");
        let decoded_body = if is_b64 {
            base64::engine::general_purpose::STANDARD
                .decode(body_str)
                .err_ctx("decode base64 Lambda response body")?
        } else {
            body_str.as_bytes().to_vec()
        };
        return Ok((status_code, resp_headers, decoded_body));
    }

    validate_endpoint_transport(base)?;
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
    let mut resp_headers = BTreeMap::new();
    for (name, val) in resp.headers() {
        if let Ok(s) = val.to_str() {
            resp_headers.insert(name.as_str().to_ascii_lowercase(), s.to_owned());
        }
    }
    let resp_body = crate::storage::cas::read_bounded(
        resp.body_mut().as_reader(),
        268_435_456,
        &format!("{method} {url} body"),
    )?;
    Ok((code, resp_headers, resp_body))
}

/// Primary first-party HTTPS URL publishing live `dev` and `prod` Carbon cloud discovery endpoints (`INS-06`).
pub const DEFAULT_PUBLIC_CLOUDS_URL: &str =
    "https://downloads.neverhuman.org/carbon/carbon-clouds.json";

/// Secondary GitHub Releases HTTPS mirror serving identical release-pinned `carbon-clouds.json` discovery endpoints.
pub const FALLBACK_PUBLIC_CLOUDS_URL: &str =
    "https://github.com/neverhuman/JopeDime/releases/latest/download/carbon-clouds.json";

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
    #[cfg(windows)]
    let home_res = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
    #[cfg(not(windows))]
    let home_res = std::env::var("HOME");
    let home = home_res.map_err(|_| {
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

fn is_non_public_redirect_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            let octets = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                || octets[0] == 0
                || (octets[0] == 100 && (octets[1] & 0b1100_0000) == 64) // RFC 6598 CGNAT 100.64.0.0/10
        }
        std::net::IpAddr::V6(v6) => {
            if let Some(mapped_v4) = v6.to_ipv4_mapped() {
                return is_non_public_redirect_ip(std::net::IpAddr::V4(mapped_v4));
            }
            let seg0 = v6.segments()[0];
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (seg0 & 0xfe00) == 0xfc00 // RFC 4193 ULA fc00::/7
                || (seg0 & 0xffc0) == 0xfe80 // Link-local fe80::/10
        }
    }
}

/// Resolve a redirect `Location` header for unauthenticated `carbon-clouds.json` discovery,
/// refusing `https -> http` scheme downgrades and private/loopback/CGNAT/IMDS redirects from HTTPS origins.
pub fn resolve_clouds_redirect_target(current_url: &str, location: &str) -> Result<String, String> {
    let base = url::Url::parse(current_url)
        .map_err(|e| format!("invalid clouds manifest URL '{current_url}': {e}"))?;
    let next = base
        .join(location.trim())
        .map_err(|e| format!("invalid redirect Location '{location}' from '{current_url}': {e}"))?;
    if base.scheme() == "https" {
        if next.scheme() != "https" {
            return Err(format!(
                "refusing insecure redirect downgrade from '{current_url}' to '{next}'"
            ));
        }
        let raw_host = next
            .host_str()
            .unwrap_or_default()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .to_ascii_lowercase();
        let blocked_domain = raw_host == "localhost"
            || raw_host.ends_with(".localhost")
            || raw_host.ends_with(".local")
            || raw_host.ends_with(".internal");
        let blocked_ip = raw_host
            .parse::<std::net::IpAddr>()
            .map(is_non_public_redirect_ip)
            .unwrap_or(false);
        if blocked_domain || blocked_ip {
            return Err(format!(
                "refusing redirect from public HTTPS '{current_url}' to local/private/link-local '{next}'"
            ));
        }
    } else if next.scheme() != "http" && next.scheme() != "https" {
        return Err(format!("unsupported redirect scheme in '{next}'"));
    }
    Ok(next.into())
}

fn fetch_clouds_manifest_bytes(source: &str) -> Result<Vec<u8>, String> {
    if source.starts_with("https://") || source.starts_with("http://") {
        validate_endpoint_transport(source)?;
        let mut current = source.to_owned();
        for _hop in 0..=5 {
            validate_endpoint_transport(&current)?;
            let mut resp = crate::tls::carbon_https_agent()
                .get(&current)
                .config()
                .http_status_as_error(false)
                .build()
                .call()
                .map_err(|e| format!("fetch Carbon clouds manifest {current} failed: {e}"))?;
            let status = resp.status().as_u16();
            if matches!(status, 301 | 302 | 303 | 307 | 308) {
                let loc = resp
                    .headers()
                    .get("location")
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| {
                        format!(
                            "Carbon clouds manifest {current} returned redirect HTTP {status} without Location"
                        )
                    })?;
                current = resolve_clouds_redirect_target(&current, loc)?;
                continue;
            }
            let buf = crate::storage::cas::read_bounded(
                resp.body_mut().as_reader(),
                4 * 1024 * 1024,
                &format!("Carbon clouds manifest {current}"),
            )?;
            if status >= 300 {
                return Err(format!(
                    "Carbon clouds manifest {current} returned HTTP {status}: {}",
                    String::from_utf8_lossy(&buf)
                ));
            }
            return Ok(buf);
        }
        Err(format!(
            "Carbon clouds manifest {source} exceeded maximum 5 redirects"
        ))
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
    static CRED_TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::fs::create_dir_all(config_dir)
        .map_err(|e| format!("create config dir {}: {e}", config_dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(config_dir, std::fs::Permissions::from_mode(0o700));
    }
    let final_path = config_dir.join("carbon-credentials.json");
    let seq = CRED_TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_path = config_dir.join(format!(
        ".carbon-credentials.{}.{}.{}.tmp",
        std::process::id(),
        now_epoch_ms(),
        seq
    ));
    let encoded = zeroize::Zeroizing::new(
        serde_json::to_vec_pretty(store).err_ctx("encode credentials store")?,
    );
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp_path)
            .map_err(|e| format!("open {}: {e}", tmp_path.display()))?;
        if let Err(e) = f.write_all(&encoded).and_then(|()| f.sync_all()) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(format!("write {}: {e}", tmp_path.display()));
        }
    }
    #[cfg(not(unix))]
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(|e| format!("open {}: {e}", tmp_path.display()))?;
        if let Err(e) = f.write_all(&encoded).and_then(|()| f.sync_all()) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(format!("write {}: {e}", tmp_path.display()));
        }
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("icacls")
                .arg(&tmp_path)
                .args(["/inheritance:r", "/grant:r", "*S-1-3-4:(F)"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
    if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(format!("activate {}: {e}", final_path.display()));
    }
    Ok(())
}

struct CredentialsLockGuard {
    lock_path: std::path::PathBuf,
}

impl CredentialsLockGuard {
    fn acquire(config_dir: &std::path::Path) -> Result<Self, String> {
        std::fs::create_dir_all(config_dir)
            .map_err(|e| format!("create config dir {}: {e}", config_dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(config_dir, std::fs::Permissions::from_mode(0o700));
        }
        let lock_path = config_dir.join("carbon-credentials.json.lock");
        for attempt in 0..300u32 {
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o600);
            }
            match opts.open(&lock_path) {
                Ok(mut f) => {
                    use std::io::Write as _;
                    let _ = writeln!(f, "pid={} ts_ms={}", std::process::id(), now_epoch_ms());
                    return Ok(Self { lock_path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let is_stale = std::fs::metadata(&lock_path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| std::time::SystemTime::now().duration_since(t).ok())
                        .is_some_and(|age| age > Duration::from_secs(30));
                    if is_stale {
                        let _ = std::fs::remove_file(&lock_path);
                        continue;
                    }
                    if attempt == 299 {
                        return Err(format!(
                            "timed out acquiring credential store lock {}",
                            lock_path.display()
                        ));
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    return Err(format!(
                        "acquire credential lock {}: {e}",
                        lock_path.display()
                    ));
                }
            }
        }
        Err(format!(
            "timed out acquiring credential store lock {}",
            lock_path.display()
        ))
    }
}

impl Drop for CredentialsLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

/// Validate and store a `ctk1.*` capability token for `cloud_name` inside the `0600` credentials store.
pub fn save_stored_cloud_token(
    cloud_name: &str,
    endpoint: &str,
    token: &str,
) -> Result<crate::security::auth::CarbonCapabilityClaims, String> {
    let trimmed_tok = token.trim();
    let claims = crate::security::auth::CarbonCapabilityClaims::decode_unverified(trimmed_tok)
        .map_err(crate::error::err_to_string)?;
    let config_dir = verified_absolute_jopedime_config_dir()?;
    let _lock = CredentialsLockGuard::acquire(&config_dir)?;
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

fn read_credentials_store_unlocked(
    config_dir: &std::path::Path,
) -> Result<Option<CarbonCredentialsStoreFile>, String> {
    let cred_path = config_dir.join("carbon-credentials.json");
    if !cred_path.is_file() {
        return Ok(None);
    }
    let raw = zeroize::Zeroizing::new(
        std::fs::read(&cred_path).map_err(|e| format!("read {}: {e}", cred_path.display()))?,
    );
    let store: CarbonCredentialsStoreFile =
        serde_json::from_slice(&raw).map_err(|e| format!("parse {}: {e}", cred_path.display()))?;
    Ok(Some(store))
}

/// Load a previously stored `ctk1.*` capability token for `cloud_name`, if present.
pub fn load_stored_cloud_token(
    cloud_name: &str,
) -> Result<Option<CarbonStoredCloudCredential>, String> {
    let config_dir = verified_absolute_jopedime_config_dir()?;
    let Some(store) = read_credentials_store_unlocked(&config_dir)? else {
        return Ok(None);
    };
    let key = cloud_name.trim().to_ascii_lowercase();
    Ok(store.clouds.get(&key).cloned())
}

/// Load all stored cloud credentials from `carbon-credentials.json` (empty map when absent).
pub fn list_stored_cloud_tokens() -> Result<BTreeMap<String, CarbonStoredCloudCredential>, String> {
    let config_dir = verified_absolute_jopedime_config_dir()?;
    Ok(read_credentials_store_unlocked(&config_dir)?
        .map(|s| s.clouds)
        .unwrap_or_default())
}

/// Remove stored capability token(s) (`Some(cloud)` removes one cloud; `None` clears all clouds).
pub fn remove_stored_cloud_token(cloud_name: Option<&str>) -> Result<usize, String> {
    let config_dir = verified_absolute_jopedime_config_dir()?;
    let _lock = CredentialsLockGuard::acquire(&config_dir)?;
    let Some(mut store) = read_credentials_store_unlocked(&config_dir)? else {
        return Ok(0);
    };
    let cred_path = config_dir.join("carbon-credentials.json");
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
        let _ = crate::cli::model_store::shred_local_file(&cred_path);
    } else {
        write_private_credentials_file(&config_dir, &store)?;
    }
    Ok(removed)
}
