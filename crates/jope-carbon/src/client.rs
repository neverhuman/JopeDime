//! Thin client SDK for Carbon regional Lambda cells and Enterprise `carbon serve` (`ADR-0007` Phase C8).
//!
//! Performs deterministic `FIT` / `HOLDOUT` splitting, `FIT`-only landmark conditioning (`JLMB` v1),
//! and `JMAT` v1 encoding strictly on the client before probing regional `/v1/carbon/ping`
//! endpoints, uploading `fit.jmat` / `holdout.jmat` to presigned URLs, starting/stopping/resuming
//! runs, and requesting server-side `predict`.

use std::io::Read;
use std::time::Instant;

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonCreateRunRequest, CarbonEarlyStopRequest, CarbonPingResponse, CarbonPredictRequest,
        CarbonPredictResponse, CarbonPrepareUploadRequest, CarbonPrepareUploadResponse,
        CarbonPublicWaveProgress, CarbonRegionCatalog, CarbonResumeRunRequest, CarbonTierWire,
        CarbonUploadSlotWire,
    },
    decimal::DecimalU64,
    ids::{DatasetId, RunId},
    Effort,
};
use jope_prep::{
    split_and_encode_jmat_v1, FitLandmarkBundle, PrepTable, SplitJmatArtifacts, SplitPlan,
};

/// `CarbonClientPreparedDataset` is a struct holding client-side split `JMAT` blobs and the
/// `FIT`-only `6,368`-byte `FitLandmarkBundle` ready for upload.
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
}

/// `CarbonThinClient` is a struct driving end-to-end Carbon inference, search, pause/resume,
/// and multi-region failover over HTTPS/HTTP.
pub struct CarbonThinClient {
    endpoints: Vec<String>,
    active_endpoint: String,
    agent: ureq::Agent,
}

impl CarbonThinClient {
    /// Construct a `CarbonThinClient` with one or more candidate regional/Enterprise URLs,
    /// probing `/v1/carbon/ping` across all candidates and selecting the lowest-RTT healthy cell.
    pub fn connect_fastest(
        candidate_urls: Vec<String>,
    ) -> Result<(Self, Vec<CarbonProbedEndpoint>), String> {
        if candidate_urls.is_empty() {
            return Err("CarbonThinClient requires at least one endpoint URL".to_owned());
        }
        let agent = crate::tls::carbon_https_agent().clone();
        let mut healthy = Vec::new();
        for raw_url in &candidate_urls {
            let base = raw_url.trim_end_matches('/').to_owned();
            let ping_url = format!("{base}/v1/carbon/ping");
            let t0 = Instant::now();
            if let Ok(mut resp) = agent
                .get(&ping_url)
                .config()
                .http_status_as_error(false)
                .build()
                .call()
            {
                if resp.status().as_u16() == 200 {
                    let mut buf = Vec::new();
                    if resp.body_mut().as_reader().read_to_end(&mut buf).is_ok() {
                        if let Ok(ping) = serde_json::from_slice::<CarbonPingResponse>(&buf) {
                            if ping.healthy {
                                healthy.push(CarbonProbedEndpoint {
                                    base_url: base,
                                    region: ping.region,
                                    rtt_ms: t0.elapsed().as_secs_f64() * 1_000.0,
                                    enterprise_self_hosted: ping.enterprise_self_hosted,
                                });
                            }
                        }
                    }
                }
            }
        }
        if healthy.is_empty() {
            return Err(
                "all candidate Carbon regional endpoints failed /v1/carbon/ping".to_owned(),
            );
        }
        healthy.sort_by(|a, b| {
            a.rtt_ms
                .partial_cmp(&b.rtt_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let active_endpoint = healthy[0].base_url.clone();
        let ordered_endpoints: Vec<String> = healthy.iter().map(|h| h.base_url.clone()).collect();
        Ok((
            Self {
                endpoints: ordered_endpoints,
                active_endpoint,
                agent,
            },
            healthy,
        ))
    }

    pub fn active_endpoint(&self) -> &str {
        &self.active_endpoint
    }

    /// Fetch `/.well-known/carbon-regions.json` from the active endpoint.
    pub fn fetch_region_catalog(&self) -> Result<CarbonRegionCatalog, String> {
        let url = format!("{}/.well-known/carbon-regions.json", self.active_endpoint);
        let bytes = self.get_json(&url)?;
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

    /// Request presigned URLs and `PUT` both `fit.jmat` and `holdout.jmat`.
    pub fn upload_prepared(
        &self,
        tenant_id: &str,
        tier: CarbonTierWire,
        prepared: &CarbonClientPreparedDataset,
    ) -> Result<CarbonPrepareUploadResponse, String> {
        let fit_sha = hex::encode(prepared.artifacts.fit_sha256);
        let holdout_sha = hex::encode(prepared.artifacts.holdout_sha256);
        let prep_req = CarbonPrepareUploadRequest {
            tenant_id: tenant_id.to_owned(),
            tier,
            fit_rows: prepared.fit_rows,
            holdout_rows: prepared.holdout_rows,
            feature_count: prepared.feature_names.len() as u32,
            fit_jmat: CarbonUploadSlotWire {
                sha256: fit_sha,
                bytes: DecimalU64(prepared.artifacts.fit_jmat.len() as u64),
            },
            holdout_jmat: CarbonUploadSlotWire {
                sha256: holdout_sha,
                bytes: DecimalU64(prepared.artifacts.holdout_jmat.len() as u64),
            },
        };
        let prep_url = format!("{}/v1/carbon/uploads/prepare", self.active_endpoint);
        let resp_bytes = self.post_json(&prep_url, &prep_req)?;
        let prep_resp: CarbonPrepareUploadResponse = serde_json::from_slice(&resp_bytes)
            .map_err(|e| format!("decode upload prepare response: {e}"))?;

        self.put_raw_bytes(&prep_resp.fit_target.url, &prepared.artifacts.fit_jmat)?;
        self.put_raw_bytes(
            &prep_resp.holdout_target.url,
            &prepared.artifacts.holdout_jmat,
        )?;
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
        let lm_wire = prepared.landmark_bundle.to_wire_bytes();
        let req = CarbonCreateRunRequest {
            run_id,
            dataset_id,
            tenant_id: tenant_id.to_owned(),
            tier,
            fit_jmat_sha256: hex::encode(prepared.artifacts.fit_sha256),
            holdout_jmat_sha256: hex::encode(prepared.artifacts.holdout_sha256),
            fit_rows: prepared.fit_rows,
            holdout_rows: prepared.holdout_rows,
            feature_names: prepared.feature_names.clone(),
            target_name: prepared.target_name.clone(),
            objective: objective.to_owned(),
            effort,
            seed: DecimalU64(seed),
            landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(lm_wire),
        };
        let url = format!("{}/v1/carbon/runs", self.active_endpoint);
        let bytes = self.post_json(&url, &req)?;
        serde_json::from_slice(&bytes).map_err(|e| format!("decode start_run progress: {e}"))
    }

    pub fn get_run_status(
        &self,
        tenant_id: &str,
        run_id: &str,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let url = format!(
            "{}/v1/carbon/runs/{run_id}?tenant_id={tenant_id}",
            self.active_endpoint
        );
        let bytes = self.get_json(&url)?;
        serde_json::from_slice(&bytes).map_err(|e| format!("decode run status: {e}"))
    }

    pub fn get_run_events_sse(&self, tenant_id: &str, run_id: &str) -> Result<String, String> {
        let url = format!(
            "{}/v1/carbon/runs/{run_id}/events?tenant_id={tenant_id}",
            self.active_endpoint
        );
        let bytes = self.get_json(&url)?;
        String::from_utf8(bytes).map_err(|e| format!("SSE UTF-8 error: {e}"))
    }

    pub fn signal_early_stop(
        &self,
        tenant_id: &str,
        run_id: &str,
        reason: Option<String>,
    ) -> Result<(), String> {
        let url = format!("{}/v1/carbon/runs/{run_id}/stop", self.active_endpoint);
        let req = CarbonEarlyStopRequest {
            tenant_id: tenant_id.to_owned(),
            reason,
        };
        let _ = self.post_json(&url, &req)?;
        Ok(())
    }

    pub fn resume_run(
        &self,
        tenant_id: &str,
        run_id: &str,
        effort: Option<Effort>,
        additional_waves: Option<u32>,
    ) -> Result<CarbonPublicWaveProgress, String> {
        let url = format!("{}/v1/carbon/runs/{run_id}/resume", self.active_endpoint);
        let req = CarbonResumeRunRequest {
            tenant_id: tenant_id.to_owned(),
            effort,
            additional_waves,
        };
        let bytes = self.post_json(&url, &req)?;
        serde_json::from_slice(&bytes).map_err(|e| format!("decode resume progress: {e}"))
    }

    pub fn predict(
        &self,
        tenant_id: &str,
        run_id: &str,
        feature_names: Vec<String>,
        rows: Vec<Vec<Option<f64>>>,
    ) -> Result<CarbonPredictResponse, String> {
        let url = format!("{}/v1/carbon/runs/{run_id}/predict", self.active_endpoint);
        let req = CarbonPredictRequest {
            tenant_id: tenant_id.to_owned(),
            feature_names,
            rows,
        };
        let bytes = self.post_json(&url, &req)?;
        serde_json::from_slice(&bytes).map_err(|e| format!("decode predict response: {e}"))
    }

    pub fn candidate_endpoints(&self) -> &[String] {
        &self.endpoints
    }

    fn get_json(&self, url: &str) -> Result<Vec<u8>, String> {
        let mut resp = self
            .agent
            .get(url)
            .config()
            .http_status_as_error(false)
            .build()
            .call()
            .map_err(|e| format!("GET {url} failed: {e}"))?;
        let code = resp.status().as_u16();
        let mut body = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("read GET body: {e}"))?;
        if code >= 300 {
            return Err(format!("HTTP {code}: {}", String::from_utf8_lossy(&body)));
        }
        Ok(body)
    }

    fn post_json<T: serde::Serialize>(&self, url: &str, body_val: &T) -> Result<Vec<u8>, String> {
        let payload = serde_json::to_vec(body_val).map_err(|e| format!("encode JSON: {e}"))?;
        let mut resp = self
            .agent
            .post(url)
            .config()
            .http_status_as_error(false)
            .build()
            .header("content-type", "application/json")
            .send(&payload)
            .map_err(|e| format!("POST {url} failed: {e}"))?;
        let code = resp.status().as_u16();
        let mut body = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut body)
            .map_err(|e| format!("read POST body: {e}"))?;
        if code >= 300 {
            return Err(format!("HTTP {code}: {}", String::from_utf8_lossy(&body)));
        }
        Ok(body)
    }

    fn put_raw_bytes(&self, url: &str, bytes: &[u8]) -> Result<(), String> {
        let mut resp = self
            .agent
            .put(url)
            .config()
            .http_status_as_error(false)
            .build()
            .send(bytes)
            .map_err(|e| format!("PUT {url} failed: {e}"))?;
        let code = resp.status().as_u16();
        if code >= 300 {
            let mut body = Vec::new();
            let _ = resp.body_mut().as_reader().read_to_end(&mut body);
            return Err(format!(
                "PUT HTTP {code}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        Ok(())
    }
}
