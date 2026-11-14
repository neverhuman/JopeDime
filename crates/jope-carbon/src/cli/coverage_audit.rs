//! Full product & adversarial live/CI coverage qualification harness (`carbon coverage-audit`).
//!
//! Exercises CLI/SDK flows, gateway HTTP routes, S3/CAS upload tamper refusal, zero-knowledge
//! `CUEN\x01`/`CUDK\x01`/`CCKP\x02` envelopes, and all adversarial cases (`bad signature`,
//! `wrong KEK`, `replay`, `shred`, and `abort`/`resume` paths) identically against dual
//! `CarbonLocalLambdaEmulator` cells in CI and existing live AWS `dev` + `prod` endpoints.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonCreateRunRequest, CarbonEarlyStopRequest, CarbonPredictRequest,
        CarbonPredictResponse, CarbonPrepareUploadRequest, CarbonPrepareUploadResponse,
        CarbonPublicWaveProgress, CarbonRegionCatalog, CarbonResumeRunRequest, CarbonRunStateWire,
        CarbonShredRunResponse, CarbonTierWire, CarbonUploadSlotWire,
    },
    decimal::DecimalU64,
    ids::{DatasetId, RunId},
    Effort,
};
use jope_prep::cuen::CuenPayloadKind;
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::{
    cli::{
        client::{
            send_raw_edge_http, send_raw_put_url, CarbonClientPreparedDataset, CarbonThinClient,
        },
        redteam::{execute_carbon_redteam_suite, CarbonRedTeamSuiteReport},
        runner::generate_nonlinear_benchmark_rows,
    },
    security::user_envelope::{
        decode_p256_pubkey_b64, decrypt_cuen_payload, encrypt_cuen_payload, generate_random_u_dek,
        unwrap_u_dek_with_kek, verify_ecdsa_p256_spki_pem_signature, wrap_u_dek_to_pubkey,
        CarbonEncryptedPredictPayload, CarbonEngineKek, CAS_BLOB_CUEN_SCOPE,
    },
    storage::cas::sha256_lower_hex,
};

/// `CarbonCoverageCheckResult` is a struct recording one capability or adversarial check outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonCoverageCheckResult {
    pub check_id: String,
    pub domain: String,
    pub title: String,
    pub passed: bool,
    pub observed_http_status: u16,
    pub wall_ms: f64,
    pub evidence: String,
}

/// `CarbonCloudCoverageInput` is a struct supplying non-persisted test credentials and pins for one cloud.
pub struct CarbonCloudCoverageInput<'a> {
    pub cloud: &'a str,
    pub endpoint_url: &'a str,
    pub region: &'a str,
    pub expected_kek_pubkey_b64: &'a str,
    pub wrong_peer_kek_pubkey_b64: &'a str,
    pub pro_tenant_a: &'a str,
    pub pro_token_a: &'a str,
    pub pro_tenant_b: &'a str,
    pub pro_token_b: &'a str,
    pub expired_pro_token_a: &'a str,
    pub signing_pub_pem: &'a str,
}

/// `CarbonCloudCoverageReceipt` is a struct capturing all product & adversarial checks for one cloud.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonCloudCoverageReceipt {
    pub cloud: String,
    pub region: String,
    pub target_endpoint: String,
    pub version_endpoint_deployed: bool,
    pub total_checks: usize,
    pub passed_checks: usize,
    pub redteam_total: usize,
    pub redteam_passed: usize,
    pub residual_unshredded_runs: usize,
    pub all_passed: bool,
    pub wall_ms: f64,
    pub checks: Vec<CarbonCoverageCheckResult>,
    pub redteam: CarbonRedTeamSuiteReport,
}

/// `CarbonDualCloudCoverageReport` is a struct bundling `dev` and `prod` product coverage receipts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonDualCloudCoverageReport {
    pub schema: String,
    pub generated_epoch_ms: u64,
    pub mode: String,
    pub all_passed: bool,
    pub total_checks_per_cloud: usize,
    pub total_redteam_vectors_per_cloud: usize,
    pub clouds: BTreeMap<String, CarbonCloudCoverageReceipt>,
}

struct EncryptedUploadFixture {
    u_dek: Zeroizing<[u8; 32]>,
    prep_resp: CarbonPrepareUploadResponse,
}

fn make_ids(tag: &str) -> Result<(RunId, DatasetId), String> {
    let u = ulid::Ulid::new().to_string();
    let run_id: RunId = format!("run_{u}")
        .parse()
        .map_err(|_| format!("invalid RunId for {tag}"))?;
    let ds_hex = &sha256_lower_hex(format!("{tag}:{u}").as_bytes())[..24];
    let ds_id: DatasetId = format!("ds_{ds_hex}")
        .parse()
        .map_err(|_| format!("invalid DatasetId for {tag}"))?;
    Ok((run_id, ds_id))
}

fn assert_clean_receipt_json(json_text: &str) -> Result<(), String> {
    for marker in [
        "ctk1.",
        "AKIA",
        "ASIA",
        "PRIVATE KEY",
        "auth_secret_hex",
        "signing.password",
    ] {
        if json_text.contains(marker) {
            return Err(format!(
                "secret-leak guard triggered on coverage receipt: found '{marker}'"
            ));
        }
    }
    Ok(())
}

fn upload_encrypted_dataset(
    base: &str,
    bearer: &str,
    tenant_id: &str,
    tier: CarbonTierWire,
    prepared: &CarbonClientPreparedDataset,
) -> Result<EncryptedUploadFixture, String> {
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
    let prep_body = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: tenant_id.to_owned(),
        tier,
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_count: u32::try_from(prepared.feature_names.len()).unwrap_or(0),
        fit_jmat: CarbonUploadSlotWire {
            sha256: sha256_lower_hex(&fit_enc),
            bytes: DecimalU64(fit_enc.len() as u64),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: sha256_lower_hex(&hold_enc),
            bytes: DecimalU64(hold_enc.len() as u64),
        },
        encrypted_cuen: true,
    })
    .map_err(|e| e.to_string())?;
    let (code, resp_bytes) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("x-carbon-capability-token", bearer)],
        Some(&prep_body),
    )?;
    if code != 200 {
        return Err(format!(
            "upload prepare failed with HTTP {code}: {}",
            String::from_utf8_lossy(&resp_bytes)
        ));
    }
    let prep_resp: CarbonPrepareUploadResponse =
        serde_json::from_slice(&resp_bytes).map_err(|e| e.to_string())?;
    let (fc, _) = send_raw_put_url(&prep_resp.fit_target.url, &fit_enc, true)?;
    let (hc, _) = send_raw_put_url(&prep_resp.holdout_target.url, &hold_enc, true)?;
    if !matches!(fc, 200..=204) || !matches!(hc, 200..=204) {
        return Err(format!("presigned PUT failed: fit={fc}, holdout={hc}"));
    }
    Ok(EncryptedUploadFixture { u_dek, prep_resp })
}

#[allow(clippy::too_many_arguments)]
fn build_encrypted_create_req(
    run_id: RunId,
    dataset_id: DatasetId,
    tenant_id: &str,
    tier: CarbonTierWire,
    effort: Effort,
    seed: u64,
    v1_1_wire: bool,
    upload: &CarbonPrepareUploadResponse,
    prepared: &CarbonClientPreparedDataset,
    lm_cuen: &[u8],
    cudk: &[u8],
) -> Result<Vec<u8>, String> {
    serde_json::to_vec(&CarbonCreateRunRequest {
        run_id,
        dataset_id,
        tenant_id: tenant_id.to_owned(),
        tier,
        upload_id: v1_1_wire.then(|| upload.upload_id.clone()),
        fit_jmat_sha256: upload.fit_target.sha256.clone(),
        holdout_jmat_sha256: upload.holdout_target.sha256.clone(),
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_names: prepared.feature_names.clone(),
        target_name: prepared.target_name.clone(),
        objective: "rmse".to_owned(),
        effort,
        seed: DecimalU64(seed),
        landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(lm_cuen),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk)),
    })
    .map_err(|e| e.to_string())
}

fn poll_run_until_settled(
    base: &str,
    bearer: &str,
    tenant_id: &str,
    run_id_str: &str,
    timeout: Duration,
) -> Result<CarbonPublicWaveProgress, String> {
    let deadline = Instant::now() + timeout;
    let path = format!("/v1/carbon/runs/{run_id_str}?tenant_id={tenant_id}");
    loop {
        let (code, body) = send_raw_edge_http(
            base,
            "GET",
            &path,
            &[("x-carbon-capability-token", bearer)],
            None,
        )?;
        if code != 200 {
            return Err(format!(
                "poll {run_id_str} returned HTTP {code}: {}",
                String::from_utf8_lossy(&body)
            ));
        }
        let prog: CarbonPublicWaveProgress =
            serde_json::from_slice(&body).map_err(|e| e.to_string())?;
        if matches!(
            prog.state,
            CarbonRunStateWire::Completed | CarbonRunStateWire::Paused | CarbonRunStateWire::Failed
        ) {
            return Ok(prog);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timed out polling {run_id_str}; last state {:?}",
                prog.state
            ));
        }
        std::thread::sleep(Duration::from_millis(120));
    }
}

fn invoke_encrypted_predict(
    base: &str,
    bearer: &str,
    tenant_id: &str,
    run_id_str: &str,
    u_dek: &[u8; 32],
    cudk_bytes: &[u8],
    feature_names: &[String],
) -> Result<(u16, Vec<u8>), String> {
    let resp_kek = CarbonEngineKek::generate().map_err(|e| e.to_string())?;
    let sample_row = vec![vec![Some(0.2); feature_names.len()]];
    let pt = serde_json::to_vec(&CarbonEncryptedPredictPayload {
        feature_names: feature_names.to_vec(),
        rows: sample_row,
    })
    .map_err(|e| e.to_string())?;
    let pred_cuen = encrypt_cuen_payload(
        u_dek,
        CuenPayloadKind::PredictFeatures,
        tenant_id,
        run_id_str,
        &pt,
    )
    .map_err(|e| e.to_string())?;
    let req_bytes = serde_json::to_vec(&CarbonPredictRequest {
        tenant_id: tenant_id.to_owned(),
        feature_names: Vec::new(),
        rows: Vec::new(),
        encrypted_rows_cuen_b64: Some(base64::engine::general_purpose::STANDARD.encode(pred_cuen)),
        encrypted_rows_cas_sha256: None,
        predict_upload_id: None,
        prefer_cas_output: false,
        client_response_pubkey_b64: Some(resp_kek.pubkey_b64()),
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk_bytes)),
    })
    .map_err(|e| e.to_string())?;
    let (code, body) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{run_id_str}/predict"),
        &[("x-carbon-capability-token", bearer)],
        Some(&req_bytes),
    )?;
    if code == 200 {
        let parsed: CarbonPredictResponse =
            serde_json::from_slice(&body).map_err(|e| e.to_string())?;
        let r_dek = unwrap_u_dek_with_kek(
            &resp_kek,
            &base64::engine::general_purpose::STANDARD
                .decode(parsed.response_cudk_b64.as_deref().unwrap_or(""))
                .map_err(|e| e.to_string())?,
            tenant_id,
            run_id_str,
        )
        .map_err(|e| e.to_string())?;
        let dec = decrypt_cuen_payload(
            &r_dek,
            CuenPayloadKind::PredictOutputs,
            tenant_id,
            run_id_str,
            &base64::engine::general_purpose::STANDARD
                .decode(
                    parsed
                        .encrypted_predictions_cuen_b64
                        .as_deref()
                        .unwrap_or(""),
                )
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if dec.len() != 8 || sha256_lower_hex(&dec) != parsed.predictions_sha256 {
            return Err("decrypted prediction digest mismatch".to_owned());
        }
    }
    Ok((code, body))
}

fn delete_shred_checked(
    base: &str,
    bearer: &str,
    tenant_id: &str,
    run_id_str: &str,
) -> Result<(u16, Option<CarbonShredRunResponse>), String> {
    let (code, body) = send_raw_edge_http(
        base,
        "DELETE",
        &format!("/v1/carbon/runs/{run_id_str}?tenant_id={tenant_id}"),
        &[("x-carbon-capability-token", bearer)],
        None,
    )?;
    if code == 200 {
        let parsed: CarbonShredRunResponse =
            serde_json::from_slice(&body).map_err(|e| e.to_string())?;
        Ok((code, Some(parsed)))
    } else {
        Ok((code, None))
    }
}

fn run_api_and_signature_checks(
    inp: &CarbonCloudCoverageInput<'_>,
    base: &str,
    bearer_a: &str,
    expired_bearer_a: &str,
    prepared_small: &CarbonClientPreparedDataset,
    checks: &mut Vec<CarbonCoverageCheckResult>,
) -> Result<(bool, String, String), String> {
    // API-LIVE-01: GET /.well-known/carbon-regions.json
    let t = Instant::now();
    let (c_reg, reg_bytes) =
        send_raw_edge_http(base, "GET", "/.well-known/carbon-regions.json", &[], None)?;
    let cat_ok = serde_json::from_slice::<CarbonRegionCatalog>(&reg_bytes)
        .map(|c| !c.regions.is_empty())
        .unwrap_or(false);
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-01".to_owned(),
        domain: "API".to_owned(),
        title: "GET /.well-known/carbon-regions.json discovery catalog".to_owned(),
        passed: c_reg == 200 && cat_ok,
        observed_http_status: c_reg,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("HTTP {c_reg}, catalog_non_empty={cat_ok}"),
    });

    // API-LIVE-02: GET /v1/carbon/ping + KEK pin verification
    let t = Instant::now();
    let mut expected_pins = BTreeMap::new();
    expected_pins.insert(base.to_owned(), inp.expected_kek_pubkey_b64.to_owned());
    let (mut pinned_client, probed) =
        CarbonThinClient::connect_fastest_pinned(vec![base.to_owned()], &expected_pins)?;
    let ping_kek_match = probed
        .first()
        .and_then(|p| p.engine_kek_pubkey_b64.as_deref())
        == Some(inp.expected_kek_pubkey_b64);
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-02".to_owned(),
        domain: "API".to_owned(),
        title: "GET /v1/carbon/ping healthy & P-256 KEK matches pinned manifest".to_owned(),
        passed: ping_kek_match,
        observed_http_status: 200,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("region={}, kek_pin_verified={ping_kek_match}", inp.region),
    });

    // API-LIVE-03: GET /v1/carbon/version probe & legacy v1.0.0 graceful fallback
    let t = Instant::now();
    let (c_ver, _) = send_raw_edge_http(base, "GET", "/v1/carbon/version", &[], None)?;
    let ver_opt = pinned_client.fetch_server_version()?;
    let ver_deployed = ver_opt.is_some();
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-03".to_owned(),
        domain: "API".to_owned(),
        title: "GET /v1/carbon/version probe & thin-client backward-compat fallback".to_owned(),
        passed: (c_ver == 200 && ver_deployed) || (c_ver == 404 && !ver_deployed),
        observed_http_status: c_ver,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("HTTP {c_ver}, version_endpoint_deployed={ver_deployed}"),
    });

    // API-LIVE-04: POST /v1/carbon/auth/register persistent 30-day Free account
    let t = Instant::now();
    let reg_free = pinned_client.register_free(Some("covmatrix"))?;
    let free_ok =
        reg_free.tenant_id.starts_with("free_covmatrix_") && reg_free.tier == CarbonTierWire::Free;
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-04".to_owned(),
        domain: "API".to_owned(),
        title: "POST /v1/carbon/auth/register persistent 30-day Free capability".to_owned(),
        passed: free_ok,
        observed_http_status: 200,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("minted tenant={} (tier=Free)", reg_free.tenant_id),
    });

    // API-LIVE-05: GET /v1/carbon/advisor/spot-hybrid authorized Pro 200 OK
    let t = Instant::now();
    let (c_spot, spot_body) = send_raw_edge_http(
        base,
        "GET",
        "/v1/carbon/advisor/spot-hybrid",
        &[("x-carbon-capability-token", bearer_a)],
        None,
    )?;
    let spot_json_ok = serde_json::from_slice::<serde_json::Value>(&spot_body)
        .map(|v| v.is_object())
        .unwrap_or(false);
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-05".to_owned(),
        domain: "API".to_owned(),
        title: "GET /v1/carbon/advisor/spot-hybrid authorized Pro 3-Gate verdict".to_owned(),
        passed: c_spot == 200 && spot_json_ok,
        observed_http_status: c_spot,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Pro token -> HTTP {c_spot}, valid_json={spot_json_ok}"),
    });

    // API-LIVE-06: Direct unauthenticated PUT /v1/carbon/cas/{sha} refused
    let t = Instant::now();
    let fake_sha = "b".repeat(64);
    let (c_cas, _) = send_raw_edge_http(
        base,
        "PUT",
        &format!("/v1/carbon/cas/{fake_sha}"),
        &[],
        Some(b"unauthenticated-direct-cas-write"),
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-06".to_owned(),
        domain: "Storage".to_owned(),
        title: "Direct unauthenticated PUT /v1/carbon/cas/{sha} refused".to_owned(),
        passed: matches!(c_cas, 400 | 401 | 403),
        observed_http_status: c_cas,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("PUT /v1/carbon/cas/<sha> without grant -> HTTP {c_cas}"),
    });

    // ADV-SIG-01: Forged HMAC capability signature rejected on prepare & runs
    let t = Instant::now();
    let forged_hdr = format!(
        "Bearer {}.{}",
        inp.pro_token_a
            .rsplit_once('.')
            .map(|(p, _)| p)
            .unwrap_or("ctk1.e30"),
        "ff".repeat(32)
    );
    let dummy_prep = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: inp.pro_tenant_a.to_owned(),
        tier: CarbonTierWire::Pro,
        fit_rows: prepared_small.fit_rows,
        holdout_rows: prepared_small.holdout_rows,
        feature_count: 4,
        fit_jmat: CarbonUploadSlotWire {
            sha256: fake_sha.clone(),
            bytes: DecimalU64(256),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: fake_sha,
            bytes: DecimalU64(128),
        },
        encrypted_cuen: true,
    })
    .map_err(|e| e.to_string())?;
    let (c_sig_prep, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("x-carbon-capability-token", &forged_hdr)],
        Some(&dummy_prep),
    )?;
    let (c_sig_spot, _) = send_raw_edge_http(
        base,
        "GET",
        "/v1/carbon/advisor/spot-hybrid",
        &[("x-carbon-capability-token", &forged_hdr)],
        None,
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-SIG-01".to_owned(),
        domain: "Adversarial:Signature".to_owned(),
        title: "Forged ctk1 HMAC signature rejected across mutating & Pro routes".to_owned(),
        passed: c_sig_prep == 401 && c_sig_spot == 401,
        observed_http_status: c_sig_prep,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("forged HMAC -> prepare HTTP {c_sig_prep}, advisor HTTP {c_sig_spot}"),
    });

    // ADV-SIG-02: Authentic HMAC-signed expired ctk1 token rejected -> HTTP 401
    let t = Instant::now();
    let (c_exp_prep, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("x-carbon-capability-token", expired_bearer_a)],
        Some(&dummy_prep),
    )?;
    let (c_exp_spot, _) = send_raw_edge_http(
        base,
        "GET",
        "/v1/carbon/advisor/spot-hybrid",
        &[("x-carbon-capability-token", expired_bearer_a)],
        None,
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-SIG-02".to_owned(),
        domain: "Adversarial:Signature".to_owned(),
        title: "Expired ctk1 capability token (valid HMAC, exp < now) rejected".to_owned(),
        passed: c_exp_prep == 401 && c_exp_spot == 401,
        observed_http_status: c_exp_prep,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("expired token -> prepare HTTP {c_exp_prep}, advisor HTTP {c_exp_spot}"),
    });

    // ADV-SIG-03: Tampered S3 / CAS presigned PUT body (checksum/signature pin violation)
    let t = Instant::now();
    let u_dek_tamper = generate_random_u_dek().map_err(|e| e.to_string())?;
    let mut fit_ct = encrypt_cuen_payload(
        &u_dek_tamper,
        CuenPayloadKind::FitJmat,
        inp.pro_tenant_a,
        CAS_BLOB_CUEN_SCOPE,
        &prepared_small.artifacts.fit_jmat,
    )
    .map_err(|e| e.to_string())?;
    let hold_ct = encrypt_cuen_payload(
        &u_dek_tamper,
        CuenPayloadKind::HoldoutJmat,
        inp.pro_tenant_a,
        CAS_BLOB_CUEN_SCOPE,
        &prepared_small.artifacts.holdout_jmat,
    )
    .map_err(|e| e.to_string())?;
    let tamper_prep_req = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: inp.pro_tenant_a.to_owned(),
        tier: CarbonTierWire::Pro,
        fit_rows: prepared_small.fit_rows,
        holdout_rows: prepared_small.holdout_rows,
        feature_count: 4,
        fit_jmat: CarbonUploadSlotWire {
            sha256: sha256_lower_hex(&fit_ct),
            bytes: DecimalU64(fit_ct.len() as u64),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: sha256_lower_hex(&hold_ct),
            bytes: DecimalU64(hold_ct.len() as u64),
        },
        encrypted_cuen: true,
    })
    .map_err(|e| e.to_string())?;
    let (_, tamper_prep_body) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&tamper_prep_req),
    )?;
    let tamper_prep: CarbonPrepareUploadResponse =
        serde_json::from_slice(&tamper_prep_body).map_err(|e| e.to_string())?;
    let last_idx = fit_ct.len().saturating_sub(1);
    fit_ct[last_idx] ^= 0x5A;
    let (c_s3_tamper, _) = send_raw_put_url(&tamper_prep.fit_target.url, &fit_ct, true)?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-SIG-03".to_owned(),
        domain: "Adversarial:Signature".to_owned(),
        title: "Mutated ciphertext against SHA-256-pinned presigned PUT rejected".to_owned(),
        passed: matches!(c_s3_tamper, 400 | 403),
        observed_http_status: c_s3_tamper,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("1-byte mutated presigned PUT (exact length) -> HTTP {c_s3_tamper}"),
    });

    // ADV-SIG-04: Release ECDSA-P256 SPKI signature & manifest tamper refusal
    let t = Instant::now();
    let bogus_der = [0x30u8, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01];
    let sig_refused = verify_ecdsa_p256_spki_pem_signature(
        inp.signing_pub_pem,
        b"0123456789abcdef  carbon-linux-x86_64-musl\n",
        &bogus_der,
    )
    .is_err();
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-SIG-04".to_owned(),
        domain: "Adversarial:Signature".to_owned(),
        title: "ECDSA-P256 release signing.pub rejects forged DER signature".to_owned(),
        passed: sig_refused,
        observed_http_status: 0,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("verify_ecdsa_p256_spki_pem_signature forged_rejected={sig_refused}"),
    });

    Ok((
        ver_deployed,
        reg_free.tenant_id,
        format!("Bearer {}", reg_free.capability_token),
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_kek_replay_and_shred_checks(
    inp: &CarbonCloudCoverageInput<'_>,
    base: &str,
    v1_1_wire: bool,
    bearer_a: &str,
    bearer_b: &str,
    prepared_small: &CarbonClientPreparedDataset,
    checks: &mut Vec<CarbonCoverageCheckResult>,
) -> Result<usize, String> {
    let mut unshredded = 0usize;
    let kek_pub = decode_p256_pubkey_b64(inp.expected_kek_pubkey_b64).map_err(|e| e.to_string())?;
    let wrong_kek_pub =
        decode_p256_pubkey_b64(inp.wrong_peer_kek_pubkey_b64).map_err(|e| e.to_string())?;

    // ADV-KEK-01: Client-side wrong KEK pin fails closed before upload
    let t = Instant::now();
    let mut wrong_pins = BTreeMap::new();
    wrong_pins.insert(base.to_owned(), inp.wrong_peer_kek_pubkey_b64.to_owned());
    let client_kek_refused =
        CarbonThinClient::connect_fastest_pinned(vec![base.to_owned()], &wrong_pins).is_err();
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-KEK-01".to_owned(),
        domain: "Adversarial:WrongKEK".to_owned(),
        title: "Client-side wrong P-256 KEK pin fails closed prior to upload".to_owned(),
        passed: client_kek_refused,
        observed_http_status: 0,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("connect_fastest_pinned(peer_kek) refused={client_kek_refused}"),
    });

    // Upload valid encrypted small dataset for tenant_A
    let up_a = upload_encrypted_dataset(
        base,
        bearer_a,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        prepared_small,
    )?;

    // ADV-KEK-02: Server-side wrong P-256 KEK CUDK\x01 wrap on POST /v1/carbon/runs -> 502
    let t = Instant::now();
    let (r_wkek, d_wkek) = make_ids("wkek_run")?;
    let r_wkek_str = r_wkek.to_string();
    let lm_wkek = encrypt_cuen_payload(
        &up_a.u_dek,
        CuenPayloadKind::LandmarkJlmb,
        inp.pro_tenant_a,
        &r_wkek_str,
        &prepared_small.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk_wkek =
        wrap_u_dek_to_pubkey(&up_a.u_dek, &wrong_kek_pub, inp.pro_tenant_a, &r_wkek_str)
            .map_err(|e| e.to_string())?;
    let req_wkek = build_encrypted_create_req(
        r_wkek,
        d_wkek,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        Effort::Fast,
        7,
        v1_1_wire,
        &up_a.prep_resp,
        prepared_small,
        &lm_wkek,
        &cudk_wkek,
    )?;
    let (c_wkek_run, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&req_wkek),
    )?;
    let _ = delete_shred_checked(base, bearer_a, inp.pro_tenant_a, &r_wkek_str)?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-KEK-02".to_owned(),
        domain: "Adversarial:WrongKEK".to_owned(),
        title: "Server-side wrong P-256 KEK CUDK\\x01 wrap on POST /runs rejected".to_owned(),
        passed: c_wkek_run == 502,
        observed_http_status: c_wkek_run,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("CUDK wrapped to wrong KEK on POST /runs -> HTTP {c_wkek_run}"),
    });

    // Create legitimate encrypted Pro run_A (and a second upload + run_A2 so run_A's blobs remain until shredded)
    let up_legit = upload_encrypted_dataset(
        base,
        bearer_a,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        prepared_small,
    )?;
    let (r_legit, d_legit) = make_ids("cov_legit")?;
    let r_legit_str = r_legit.to_string();
    let lm_legit = encrypt_cuen_payload(
        &up_legit.u_dek,
        CuenPayloadKind::LandmarkJlmb,
        inp.pro_tenant_a,
        &r_legit_str,
        &prepared_small.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk_legit =
        wrap_u_dek_to_pubkey(&up_legit.u_dek, &kek_pub, inp.pro_tenant_a, &r_legit_str)
            .map_err(|e| e.to_string())?;
    let req_legit = build_encrypted_create_req(
        r_legit.clone(),
        d_legit.clone(),
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        Effort::Fast,
        42,
        v1_1_wire,
        &up_legit.prep_resp,
        prepared_small,
        &lm_legit,
        &cudk_legit,
    )?;
    let (c_legit_create, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&req_legit),
    )?;
    if !matches!(c_legit_create, 200..=202) {
        return Err(format!("legit Pro run failed: HTTP {c_legit_create}"));
    }
    unshredded += 1;
    let settled = poll_run_until_settled(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_legit_str,
        Duration::from_secs(45),
    )?;
    if settled.state != CarbonRunStateWire::Completed {
        return Err(format!("legit Pro run state {:?}", settled.state));
    }

    // API-LIVE-07: GET /v1/carbon/runs/{id}/events SSE stream + cross-tenant 403
    let t = Instant::now();
    let ev_path = format!(
        "/v1/carbon/runs/{r_legit_str}/events?tenant_id={}",
        inp.pro_tenant_a
    );
    let (c_ev_ok, ev_body) = send_raw_edge_http(
        base,
        "GET",
        &ev_path,
        &[("x-carbon-capability-token", bearer_a)],
        None,
    )?;
    let sse_text = String::from_utf8_lossy(&ev_body);
    let (c_ev_idor, _) = send_raw_edge_http(
        base,
        "GET",
        &ev_path,
        &[("x-carbon-capability-token", bearer_b)],
        None,
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-07".to_owned(),
        domain: "API".to_owned(),
        title: "GET /v1/carbon/runs/{id}/events SSE stream & cross-tenant 403".to_owned(),
        passed: c_ev_ok == 200
            && sse_text.starts_with("event: wave_progress\ndata: ")
            && c_ev_idor == 403,
        observed_http_status: c_ev_ok,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("owner SSE -> HTTP {c_ev_ok}, cross-tenant -> HTTP {c_ev_idor}"),
    });

    // ADV-KEK-03: Server-side wrong P-256 KEK CUDK\x01 wrap on POST /v1/carbon/runs/{id}/predict -> 502
    let t = Instant::now();
    let wrong_pred_cudk = wrap_u_dek_to_pubkey(
        &up_legit.u_dek,
        &wrong_kek_pub,
        inp.pro_tenant_a,
        &r_legit_str,
    )
    .map_err(|e| e.to_string())?;
    let (c_wkek_pred, _) = invoke_encrypted_predict(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_legit_str,
        &up_legit.u_dek,
        &wrong_pred_cudk,
        &prepared_small.feature_names,
    )?;
    let (c_good_pred, _) = invoke_encrypted_predict(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_legit_str,
        &up_legit.u_dek,
        &cudk_legit,
        &prepared_small.feature_names,
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-KEK-03".to_owned(),
        domain: "Adversarial:WrongKEK".to_owned(),
        title: "Server-side wrong P-256 KEK CUDK\\x01 wrap on POST /predict rejected".to_owned(),
        passed: c_wkek_pred == 502 && c_good_pred == 200,
        observed_http_status: c_wkek_pred,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "wrong KEK /predict -> HTTP {c_wkek_pred}, valid KEK -> HTTP {c_good_pred}"
        ),
    });

    // ADV-RPL-01: Duplicate run_id POST /v1/carbon/runs replay with altered seed -> 409
    let t = Instant::now();
    let dup_req = build_encrypted_create_req(
        r_legit.clone(),
        d_legit.clone(),
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        Effort::Fast,
        999,
        v1_1_wire,
        &up_legit.prep_resp,
        prepared_small,
        &lm_legit,
        &cudk_legit,
    )?;
    let (c_dup, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&dup_req),
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-RPL-01".to_owned(),
        domain: "Adversarial:Replay".to_owned(),
        title: "Duplicate run_id POST /runs replay with mutated seed refused".to_owned(),
        passed: c_dup == 409,
        observed_http_status: c_dup,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("duplicate POST /runs({r_legit_str}) -> HTTP {c_dup}"),
    });

    // ADV-RPL-03: Cross-run CUDK\x01 replay (run_A's wrapped_u_dek_b64 replayed on run_B /runs & /predict)
    let t = Instant::now();
    let up_rpl = upload_encrypted_dataset(
        base,
        bearer_a,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        prepared_small,
    )?;
    let (r_cudk_b, d_cudk_b) = make_ids("rpl_cudk")?;
    let r_cudk_b_str = r_cudk_b.to_string();
    let lm_cudk_b = encrypt_cuen_payload(
        &up_legit.u_dek,
        CuenPayloadKind::LandmarkJlmb,
        inp.pro_tenant_a,
        &r_cudk_b_str,
        &prepared_small.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let req_cudk_replay = build_encrypted_create_req(
        r_cudk_b,
        d_cudk_b,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        Effort::Fast,
        11,
        v1_1_wire,
        &up_rpl.prep_resp,
        prepared_small,
        &lm_cudk_b,
        &cudk_legit, // <-- bound to r_legit_str, replayed onto r_cudk_b_str!
    )?;
    let (c_rpl_cudk_run, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&req_cudk_replay),
    )?;
    let _ = delete_shred_checked(base, bearer_a, inp.pro_tenant_a, &r_cudk_b_str)?;
    let foreign_cudk = wrap_u_dek_to_pubkey(
        &up_legit.u_dek,
        &kek_pub,
        inp.pro_tenant_a,
        "run_01FOREIGN00000000000000000",
    )
    .map_err(|e| e.to_string())?;
    let (c_rpl_cudk_pred, _) = invoke_encrypted_predict(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_legit_str,
        &up_legit.u_dek,
        &foreign_cudk,
        &prepared_small.feature_names,
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-RPL-03".to_owned(),
        domain: "Adversarial:Replay".to_owned(),
        title: "Cross-run CUDK\\x01 envelope replay rejected by HKDF/GCM run_id AAD".to_owned(),
        passed: c_rpl_cudk_run == 502 && c_rpl_cudk_pred == 502,
        observed_http_status: c_rpl_cudk_run,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "replayed CUDK -> /runs HTTP {c_rpl_cudk_run}, /predict HTTP {c_rpl_cudk_pred}"
        ),
    });

    // ADV-RPL-04: Cross-run CUEN\x01 ciphertext replay (run_A's LandmarkJlmb replayed on run_C)
    let t = Instant::now();
    let up_cuen = upload_encrypted_dataset(
        base,
        bearer_a,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        prepared_small,
    )?;
    let (r_cuen_c, d_cuen_c) = make_ids("rpl_cuen")?;
    let r_cuen_c_str = r_cuen_c.to_string();
    let cudk_c_for_legit_dek =
        wrap_u_dek_to_pubkey(&up_legit.u_dek, &kek_pub, inp.pro_tenant_a, &r_cuen_c_str)
            .map_err(|e| e.to_string())?;
    let req_cuen_replay = build_encrypted_create_req(
        r_cuen_c,
        d_cuen_c,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        Effort::Fast,
        12,
        v1_1_wire,
        &up_cuen.prep_resp,
        prepared_small,
        &lm_legit, // <-- encrypted to r_legit_str AAD, replayed onto r_cuen_c_str!
        &cudk_c_for_legit_dek,
    )?;
    let (c_rpl_cuen, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&req_cuen_replay),
    )?;
    let _ = delete_shred_checked(base, bearer_a, inp.pro_tenant_a, &r_cuen_c_str)?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-RPL-04".to_owned(),
        domain: "Adversarial:Replay".to_owned(),
        title: "Cross-run CUEN\\x01 ciphertext replay rejected by run_id AAD binding".to_owned(),
        passed: matches!(c_rpl_cuen, 400 | 502),
        observed_http_status: c_rpl_cuen,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("run_A LandmarkJlmb replayed in run_C -> HTTP {c_rpl_cuen}"),
    });

    // ADV-RPL-05: Cross-tenant upload SHA substitution replay (tenant_B referencing tenant_A blobs)
    let t = Instant::now();
    let (r_xten, d_xten) = make_ids("rpl_xten")?;
    let r_xten_str = r_xten.to_string();
    let lm_xten = encrypt_cuen_payload(
        &up_legit.u_dek,
        CuenPayloadKind::LandmarkJlmb,
        inp.pro_tenant_b,
        &r_xten_str,
        &prepared_small.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk_xten = wrap_u_dek_to_pubkey(&up_legit.u_dek, &kek_pub, inp.pro_tenant_b, &r_xten_str)
        .map_err(|e| e.to_string())?;
    let req_xten = build_encrypted_create_req(
        r_xten,
        d_xten,
        inp.pro_tenant_b,
        CarbonTierWire::Pro,
        Effort::Fast,
        13,
        v1_1_wire,
        &up_legit.prep_resp, // <-- tenant_A's uploaded SHAs!
        prepared_small,
        &lm_xten,
        &cudk_xten,
    )?;
    let (c_xten, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_b)],
        Some(&req_xten),
    )?;
    let _ = delete_shred_checked(base, bearer_b, inp.pro_tenant_b, &r_xten_str)?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-RPL-05".to_owned(),
        domain: "Adversarial:Replay".to_owned(),
        title: "Cross-tenant upload SHA substitution replay refused by S3 prefix isolation"
            .to_owned(),
        passed: matches!(c_xten, 400 | 403 | 404 | 502),
        observed_http_status: c_xten,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("tenant_B referencing tenant_A fit/holdout SHA -> HTTP {c_xten}"),
    });

    // ADV-SHR-01: Cross-tenant DELETE /v1/carbon/runs/{run_A} refused -> 403, run_A preserved
    let t = Instant::now();
    let (c_idor_del, _) = delete_shred_checked(base, bearer_b, inp.pro_tenant_a, &r_legit_str)?;
    let (c_still_alive, _) = send_raw_edge_http(
        base,
        "GET",
        &format!(
            "/v1/carbon/runs/{r_legit_str}?tenant_id={}",
            inp.pro_tenant_a
        ),
        &[("x-carbon-capability-token", bearer_a)],
        None,
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-SHR-01".to_owned(),
        domain: "Adversarial:Shred".to_owned(),
        title: "Cross-tenant DELETE shred refused (403) & owner run remains intact".to_owned(),
        passed: c_idor_del == 403 && c_still_alive == 200,
        observed_http_status: c_idor_del,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "tenant_B DELETE -> HTTP {c_idor_del}, owner GET -> HTTP {c_still_alive}"
        ),
    });

    // ADV-SHR-02 & ADV-RPL-02: Authorized DELETE shred + complete post-shred 404 sweep + 409 anti-resurrection
    let t = Instant::now();
    let (c_shred_ok, shred_opt) =
        delete_shred_checked(base, bearer_a, inp.pro_tenant_a, &r_legit_str)?;
    let deleted_objs = shred_opt.as_ref().map(|s| s.objects_deleted).unwrap_or(0);
    if c_shred_ok == 200 {
        unshredded = unshredded.saturating_sub(1);
    }
    let (c_ps_get, _) = send_raw_edge_http(
        base,
        "GET",
        &format!(
            "/v1/carbon/runs/{r_legit_str}?tenant_id={}",
            inp.pro_tenant_a
        ),
        &[("x-carbon-capability-token", bearer_a)],
        None,
    )?;
    let (c_ps_ev, _) = send_raw_edge_http(
        base,
        "GET",
        &ev_path,
        &[("x-carbon-capability-token", bearer_a)],
        None,
    )?;
    let (c_ps_pred, _) = invoke_encrypted_predict(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_legit_str,
        &up_legit.u_dek,
        &cudk_legit,
        &prepared_small.feature_names,
    )?;
    let ps_resume_body = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: inp.pro_tenant_a.to_owned(),
        effort: None,
        additional_waves: Some(1),
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(&cudk_legit)),
    })
    .map_err(|e| e.to_string())?;
    let (c_ps_res, _) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{r_legit_str}/resume"),
        &[("x-carbon-capability-token", bearer_a)],
        Some(&ps_resume_body),
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-SHR-02".to_owned(),
        domain: "Adversarial:Shred".to_owned(),
        title: "Post-shred 404 lifecycle sweep (GET, /events, /predict, /resume)".to_owned(),
        passed: c_shred_ok == 200
            && deleted_objs >= 4
            && c_ps_get == 404
            && c_ps_ev == 404
            && c_ps_pred == 404
            && c_ps_res == 404,
        observed_http_status: c_ps_get,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "deleted={deleted_objs}, GET={c_ps_get}, events={c_ps_ev}, predict={c_ps_pred}, resume={c_ps_res}"
        ),
    });

    let t = Instant::now();
    let (c_resurrect, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&dup_req),
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-RPL-02".to_owned(),
        domain: "Adversarial:Replay".to_owned(),
        title: "Post-shred run_id resurrection replay blocked by tombstone (409)".to_owned(),
        passed: c_resurrect == 409,
        observed_http_status: c_resurrect,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("re-POST shredded {r_legit_str} -> HTTP {c_resurrect}"),
    });

    // ADV-SHR-03: ThinClient E2E run + shred_run (`--shred` workflow)
    let t = Instant::now();
    let mut sdk_pins = BTreeMap::new();
    sdk_pins.insert(base.to_owned(), inp.expected_kek_pubkey_b64.to_owned());
    let (raw_sdk_client, _) =
        CarbonThinClient::connect_fastest_pinned(vec![base.to_owned()], &sdk_pins)?;
    let sdk_client = raw_sdk_client.with_capability_token(inp.pro_token_a.to_owned());
    let _ = sdk_client.upload_prepared(inp.pro_tenant_a, CarbonTierWire::Pro, prepared_small)?;
    let (r_sdk, d_sdk) = make_ids("sdk_shred")?;
    let r_sdk_str = r_sdk.to_string();
    let sdk_prog = sdk_client.start_run(
        r_sdk,
        d_sdk,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        Effort::Fast,
        "rmse",
        21,
        prepared_small,
    )?;
    let sdk_shred = sdk_client.shred_run(inp.pro_tenant_a, &r_sdk_str)?;
    let (c_sdk_after, _) = send_raw_edge_http(
        base,
        "GET",
        &format!("/v1/carbon/runs/{r_sdk_str}?tenant_id={}", inp.pro_tenant_a),
        &[("x-carbon-capability-token", bearer_a)],
        None,
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-SHR-03".to_owned(),
        domain: "Storage".to_owned(),
        title: "CarbonThinClient --shred workflow purges S3 CAS & returns 404".to_owned(),
        passed: sdk_prog.state == CarbonRunStateWire::Completed
            && sdk_shred.shredded
            && sdk_shred.objects_deleted >= 4
            && c_sdk_after == 404,
        observed_http_status: c_sdk_after,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "sdk shred deleted={}, post-GET -> HTTP {c_sdk_after}",
            sdk_shred.objects_deleted
        ),
    });

    Ok(unshredded)
}

#[allow(clippy::too_many_arguments)]
fn run_abort_and_resume_checks(
    inp: &CarbonCloudCoverageInput<'_>,
    base: &str,
    v1_1_wire: bool,
    bearer_a: &str,
    bearer_b: &str,
    free_tenant: &str,
    free_bearer: &str,
    prepared_async: &CarbonClientPreparedDataset,
    prepared_small: &CarbonClientPreparedDataset,
    checks: &mut Vec<CarbonCoverageCheckResult>,
) -> Result<usize, String> {
    let mut unshredded = 0usize;
    let kek_pub = decode_p256_pubkey_b64(inp.expected_kek_pubkey_b64).map_err(|e| e.to_string())?;
    let wrong_kek_pub =
        decode_p256_pubkey_b64(inp.wrong_peer_kek_pubkey_b64).map_err(|e| e.to_string())?;

    // Run 1 for standalone abort (`ADV-ABT-01`), cross-tenant `/stop` (`ADV-ABT-03`), and wrong-KEK `/resume` (`ADV-KEK-04`)
    let up_abt = upload_encrypted_dataset(
        base,
        bearer_a,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        prepared_async,
    )?;
    let (r_abt, d_abt) = make_ids("cov_abort")?;
    let r_abt_str = r_abt.to_string();
    let lm_abt = encrypt_cuen_payload(
        &up_abt.u_dek,
        CuenPayloadKind::LandmarkJlmb,
        inp.pro_tenant_a,
        &r_abt_str,
        &prepared_async.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk_abt = wrap_u_dek_to_pubkey(&up_abt.u_dek, &kek_pub, inp.pro_tenant_a, &r_abt_str)
        .map_err(|e| e.to_string())?;
    let req_abt = build_encrypted_create_req(
        r_abt,
        d_abt,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        Effort::Medium,
        55,
        v1_1_wire,
        &up_abt.prep_resp,
        prepared_async,
        &lm_abt,
        &cudk_abt,
    )?;
    let t_abt = Instant::now();
    let (c_abt_start, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&req_abt),
    )?;
    if !matches!(c_abt_start, 200..=202) {
        return Err(format!(
            "abort test run failed to start: HTTP {c_abt_start}"
        ));
    }
    unshredded += 1;

    // ADV-ABT-03: Cross-tenant POST /stop with tenant_B token -> HTTP 403
    let t_xstop = Instant::now();
    let stop_body_owner = serde_json::to_vec(&CarbonEarlyStopRequest {
        tenant_id: inp.pro_tenant_a.to_owned(),
        reason: Some("cooperative_abort_test".to_owned()),
    })
    .map_err(|e| e.to_string())?;
    let (c_xstop, _) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{r_abt_str}/stop"),
        &[("x-carbon-capability-token", bearer_b)],
        Some(&stop_body_owner),
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-ABT-03".to_owned(),
        domain: "Adversarial:Abort".to_owned(),
        title: "Cross-tenant POST /v1/carbon/runs/{id}/stop refused (403)".to_owned(),
        passed: c_xstop == 403,
        observed_http_status: c_xstop,
        wall_ms: t_xstop.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("tenant_B POST /stop on tenant_A run -> HTTP {c_xstop}"),
    });

    // Authorized cooperative early-stop signal (`POST /stop` -> 202) WITHOUT resume
    let (c_stop_ok, _) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{r_abt_str}/stop"),
        &[("x-carbon-capability-token", bearer_a)],
        Some(&stop_body_owner),
    )?;
    let abt_settled = poll_run_until_settled(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_abt_str,
        Duration::from_secs(90),
    )?;
    let (c_abt_pred, _) = invoke_encrypted_predict(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_abt_str,
        &up_abt.u_dek,
        &cudk_abt,
        &prepared_async.feature_names,
    )?;

    // ADV-KEK-04: Wrong-KEK CUDK\x01 on POST /v1/carbon/runs/{id}/resume -> HTTP 502 (sync extra_waves=1)
    let t_wkek_res = Instant::now();
    let wrong_res_cudk =
        wrap_u_dek_to_pubkey(&up_abt.u_dek, &wrong_kek_pub, inp.pro_tenant_a, &r_abt_str)
            .map_err(|e| e.to_string())?;
    let wrong_res_body = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: inp.pro_tenant_a.to_owned(),
        effort: None,
        additional_waves: Some(1),
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(wrong_res_cudk)),
    })
    .map_err(|e| e.to_string())?;
    let (c_wkek_res, _) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{r_abt_str}/resume"),
        &[("x-carbon-capability-token", bearer_a)],
        Some(&wrong_res_body),
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-KEK-04".to_owned(),
        domain: "Adversarial:WrongKEK".to_owned(),
        title: "Server-side wrong P-256 KEK CUDK\\x01 wrap on POST /resume rejected".to_owned(),
        passed: c_wkek_res == 502,
        observed_http_status: c_wkek_res,
        wall_ms: t_wkek_res.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("wrong KEK on POST /resume -> HTTP {c_wkek_res}"),
    });

    let (c_abt_del, _) = delete_shred_checked(base, bearer_a, inp.pro_tenant_a, &r_abt_str)?;
    if c_abt_del == 200 {
        unshredded = unshredded.saturating_sub(1);
    }
    let (c_abt_after, _) = send_raw_edge_http(
        base,
        "GET",
        &format!("/v1/carbon/runs/{r_abt_str}?tenant_id={}", inp.pro_tenant_a),
        &[("x-carbon-capability-token", bearer_a)],
        None,
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-ABT-01".to_owned(),
        domain: "Adversarial:Abort".to_owned(),
        title: "Standalone cooperative POST /stop without /resume + interim /predict + shred".to_owned(),
        passed: c_stop_ok == 202
            && matches!(
                abt_settled.state,
                CarbonRunStateWire::Paused | CarbonRunStateWire::Completed
            )
            && c_abt_pred == 200
            && c_abt_after == 404,
        observed_http_status: c_stop_ok,
        wall_ms: t_abt.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "stop={c_stop_ok}, state={:?} (wave {}), predict={c_abt_pred}, post-shred GET={c_abt_after}",
            abt_settled.state, abt_settled.wave_index
        ),
    });

    // Run 2 for missing-UDek `/resume` (`ADV-ABT-04`) + valid encrypted `CCKP\x02` `/resume` (`ADV-ABT-02`)
    let up_res = upload_encrypted_dataset(
        base,
        bearer_a,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        prepared_async,
    )?;
    let (r_res, d_res) = make_ids("cov_resume")?;
    let r_res_str = r_res.to_string();
    let lm_res = encrypt_cuen_payload(
        &up_res.u_dek,
        CuenPayloadKind::LandmarkJlmb,
        inp.pro_tenant_a,
        &r_res_str,
        &prepared_async.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk_res = wrap_u_dek_to_pubkey(&up_res.u_dek, &kek_pub, inp.pro_tenant_a, &r_res_str)
        .map_err(|e| e.to_string())?;
    let req_res = build_encrypted_create_req(
        r_res,
        d_res,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        Effort::Medium,
        66,
        v1_1_wire,
        &up_res.prep_resp,
        prepared_async,
        &lm_res,
        &cudk_res,
    )?;
    let t_res_flow = Instant::now();
    let (c_res_start, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&req_res),
    )?;
    if !matches!(c_res_start, 200..=202) {
        return Err(format!(
            "resume test run failed to start: HTTP {c_res_start}"
        ));
    }
    unshredded += 1;
    let _ = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{r_res_str}/stop"),
        &[("x-carbon-capability-token", bearer_a)],
        Some(&stop_body_owner),
    )?;
    let paused_prog = poll_run_until_settled(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_res_str,
        Duration::from_secs(90),
    )?;

    // ADV-ABT-04: Missing wrapped_u_dek_b64 on encrypted POST /resume refused at Edge -> 400
    let t_noudek = Instant::now();
    let missing_dek_body = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: inp.pro_tenant_a.to_owned(),
        effort: None,
        additional_waves: Some(1),
        wrapped_u_dek_b64: None,
    })
    .map_err(|e| e.to_string())?;
    let (c_missing_dek, _) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{r_res_str}/resume"),
        &[("x-carbon-capability-token", bearer_a)],
        Some(&missing_dek_body),
    )?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-ABT-04".to_owned(),
        domain: "Adversarial:Abort".to_owned(),
        title: "Encrypted POST /resume missing wrapped_u_dek_b64 refused at Edge (400)".to_owned(),
        passed: c_missing_dek == 400,
        observed_http_status: c_missing_dek,
        wall_ms: t_noudek.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("POST /resume without wrapped_u_dek_b64 -> HTTP {c_missing_dek}"),
    });

    // ADV-ABT-02: Valid encrypted CCKP\x02 POST /resume + /predict + DELETE shred
    let fresh_res_cudk =
        wrap_u_dek_to_pubkey(&up_res.u_dek, &kek_pub, inp.pro_tenant_a, &r_res_str)
            .map_err(|e| e.to_string())?;
    let valid_res_body = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: inp.pro_tenant_a.to_owned(),
        effort: None,
        additional_waves: Some(1),
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(fresh_res_cudk)),
    })
    .map_err(|e| e.to_string())?;
    let (c_res_ok, _) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{r_res_str}/resume"),
        &[("x-carbon-capability-token", bearer_a)],
        Some(&valid_res_body),
    )?;
    let after_res = poll_run_until_settled(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_res_str,
        Duration::from_secs(90),
    )?;
    let (c_res_pred, _) = invoke_encrypted_predict(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &r_res_str,
        &up_res.u_dek,
        &cudk_res,
        &prepared_async.feature_names,
    )?;
    let (c_res_del, _) = delete_shred_checked(base, bearer_a, inp.pro_tenant_a, &r_res_str)?;
    if c_res_del == 200 {
        unshredded = unshredded.saturating_sub(1);
    }
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-ABT-02".to_owned(),
        domain: "Adversarial:Abort".to_owned(),
        title: "Cooperative /stop + encrypted CCKP\\x02 /resume + /predict + shred".to_owned(),
        passed: matches!(c_res_ok, 200..=202)
            && matches!(
                after_res.state,
                CarbonRunStateWire::Paused | CarbonRunStateWire::Completed
            )
            && after_res.wave_index >= paused_prog.wave_index
            && c_res_pred == 200
            && c_res_del == 200,
        observed_http_status: c_res_ok,
        wall_ms: t_res_flow.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "pre_wave={} -> post_wave={}, resume HTTP {c_res_ok}, predict={c_res_pred}",
            paused_prog.wave_index, after_res.wave_index
        ),
    });

    // ADV-ABT-05: Free-tier privilege escalation on POST /resume (effort: High) -> 403
    let t_esc = Instant::now();
    let up_free = upload_encrypted_dataset(
        base,
        free_bearer,
        free_tenant,
        CarbonTierWire::Free,
        prepared_small,
    )?;
    let (r_free, d_free) = make_ids("free_esc")?;
    let r_free_str = r_free.to_string();
    let lm_free = encrypt_cuen_payload(
        &up_free.u_dek,
        CuenPayloadKind::LandmarkJlmb,
        free_tenant,
        &r_free_str,
        &prepared_small.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk_free = wrap_u_dek_to_pubkey(&up_free.u_dek, &kek_pub, free_tenant, &r_free_str)
        .map_err(|e| e.to_string())?;
    let req_free = build_encrypted_create_req(
        r_free,
        d_free,
        free_tenant,
        CarbonTierWire::Free,
        Effort::Fast,
        77,
        v1_1_wire,
        &up_free.prep_resp,
        prepared_small,
        &lm_free,
        &cudk_free,
    )?;
    let (c_free_start, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", free_bearer)],
        Some(&req_free),
    )?;
    if matches!(c_free_start, 200..=202) {
        unshredded += 1;
    }
    let _ = poll_run_until_settled(
        base,
        free_bearer,
        free_tenant,
        &r_free_str,
        Duration::from_secs(45),
    )?;
    let esc_body = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: free_tenant.to_owned(),
        effort: Some(Effort::High),
        additional_waves: Some(1),
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(&cudk_free)),
    })
    .map_err(|e| e.to_string())?;
    let (c_esc, _) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{r_free_str}/resume"),
        &[("x-carbon-capability-token", free_bearer)],
        Some(&esc_body),
    )?;
    let (c_free_del, _) = delete_shred_checked(base, free_bearer, free_tenant, &r_free_str)?;
    if c_free_del == 200 {
        unshredded = unshredded.saturating_sub(1);
    }
    checks.push(CarbonCoverageCheckResult {
        check_id: "ADV-ABT-05".to_owned(),
        domain: "Adversarial:Abort".to_owned(),
        title: "Free-tier privilege escalation on POST /resume (effort=high) refused (403)"
            .to_owned(),
        passed: c_esc == 403 && c_free_del == 200,
        observed_http_status: c_esc,
        wall_ms: t_esc.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "Free run /resume(effort=high) -> HTTP {c_esc}, shredded={}",
            c_free_del == 200
        ),
    });

    Ok(unshredded)
}

/// Execute all 28 product/adversarial checks plus the 16-vector `carbon redteam` suite on one cloud.
pub fn execute_cloud_product_coverage_audit(
    inp: &CarbonCloudCoverageInput<'_>,
) -> Result<CarbonCloudCoverageReceipt, String> {
    let t0 = Instant::now();
    let base = inp.endpoint_url.trim_end_matches('/').to_owned();
    let bearer_a = format!("Bearer {}", inp.pro_token_a);
    let bearer_b = format!("Bearer {}", inp.pro_token_b);
    let expired_bearer_a = format!("Bearer {}", inp.expired_pro_token_a);

    // Run 16-vector redteam first (also ensures any 1-second TTL expired token has elapsed > 1s).
    let redteam = execute_carbon_redteam_suite(&base)?;

    let (h_small, r_small, tgt_small) = generate_nonlinear_benchmark_rows(80, 4, 314);
    let prepared_small =
        CarbonThinClient::prepare_numeric_dataset(h_small, &r_small, &tgt_small, 314, 0.25)?;
    let (h_async, r_async, tgt_async) = generate_nonlinear_benchmark_rows(120, 4, 2718);
    let prepared_async =
        CarbonThinClient::prepare_numeric_dataset(h_async, &r_async, &tgt_async, 2718, 0.25)?;

    let mut checks = Vec::with_capacity(28);
    let (ver_deployed, free_tenant, free_bearer) = run_api_and_signature_checks(
        inp,
        &base,
        &bearer_a,
        &expired_bearer_a,
        &prepared_small,
        &mut checks,
    )?;
    let rem_1 = run_kek_replay_and_shred_checks(
        inp,
        &base,
        ver_deployed,
        &bearer_a,
        &bearer_b,
        &prepared_small,
        &mut checks,
    )?;
    let rem_2 = run_abort_and_resume_checks(
        inp,
        &base,
        ver_deployed,
        &bearer_a,
        &bearer_b,
        &free_tenant,
        &free_bearer,
        &prepared_async,
        &prepared_small,
        &mut checks,
    )?;

    let passed_checks = checks.iter().filter(|c| c.passed).count();
    let total_checks = checks.len();
    let residual = rem_1 + rem_2;
    let all_passed = passed_checks == total_checks && redteam.all_passed && residual == 0;

    let receipt = CarbonCloudCoverageReceipt {
        cloud: inp.cloud.to_owned(),
        region: inp.region.to_owned(),
        target_endpoint: base,
        version_endpoint_deployed: ver_deployed,
        total_checks,
        passed_checks,
        redteam_total: redteam.total_vectors,
        redteam_passed: redteam.passed_vectors,
        residual_unshredded_runs: residual,
        all_passed,
        wall_ms: t0.elapsed().as_secs_f64() * 1_000.0,
        checks,
        redteam,
    };
    let receipt_json = serde_json::to_string(&receipt).map_err(|e| e.to_string())?;
    assert_clean_receipt_json(&receipt_json)?;
    Ok(receipt)
}

/// Execute the dual-cloud (`dev` + `prod`) live coverage qualification against existing AWS endpoints
/// using read-only Secrets Manager token minting (`ttl = 900s`) and automatic `--shred` cleanup.
#[cfg(feature = "engine")]
pub fn execute_live_aws_product_coverage_audit(
    clouds_manifest_path: &std::path::Path,
    signing_pub_path: &std::path::Path,
    json_out: Option<&std::path::Path>,
) -> Result<CarbonDualCloudCoverageReport, String> {
    use std::fs;

    use crate::{
        cells::engine_invoker::now_epoch_ms, cli::client::CarbonPublicCloudsManifest,
        cli::deploy::mint_capability_token_from_vault,
    };

    let manifest_bytes = fs::read(clouds_manifest_path)
        .map_err(|e| format!("read {}: {e}", clouds_manifest_path.display()))?;
    let manifest: CarbonPublicCloudsManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("parse {}: {e}", clouds_manifest_path.display()))?;
    let signing_pub_pem = fs::read_to_string(signing_pub_path)
        .map_err(|e| format!("read {}: {e}", signing_pub_path.display()))?;

    let dev_cell = manifest
        .clouds
        .get("dev")
        .ok_or_else(|| "missing `dev` cloud in carbon-clouds.json".to_owned())?;
    let prod_cell = manifest
        .clouds
        .get("prod")
        .ok_or_else(|| "missing `prod` cloud in carbon-clouds.json".to_owned())?;

    let suffix = ulid::Ulid::new().to_string().to_ascii_lowercase();
    let short_id = &suffix[suffix.len().saturating_sub(10)..];
    let dev_ten_a = format!("pro_covdeva_{short_id}");
    let dev_ten_b = format!("pro_covdevb_{short_id}");
    let prod_ten_a = format!("pro_covprda_{short_id}");
    let prod_ten_b = format!("pro_covprdb_{short_id}");

    // Mint 1-second TTL tokens first so they naturally expire during the 16-vector redteam preamble.
    let dev_exp = Zeroizing::new(mint_capability_token_from_vault(
        "dev",
        &dev_cell.region,
        &dev_ten_a,
        CarbonTierWire::Pro,
        1,
        false,
    )?);
    let prod_exp = Zeroizing::new(mint_capability_token_from_vault(
        "prod",
        &prod_cell.region,
        &prod_ten_a,
        CarbonTierWire::Pro,
        1,
        false,
    )?);
    std::thread::sleep(Duration::from_millis(1_500));

    let dev_tok_a = Zeroizing::new(mint_capability_token_from_vault(
        "dev",
        &dev_cell.region,
        &dev_ten_a,
        CarbonTierWire::Pro,
        900,
        false,
    )?);
    let dev_tok_b = Zeroizing::new(mint_capability_token_from_vault(
        "dev",
        &dev_cell.region,
        &dev_ten_b,
        CarbonTierWire::Pro,
        900,
        false,
    )?);
    let prod_tok_a = Zeroizing::new(mint_capability_token_from_vault(
        "prod",
        &prod_cell.region,
        &prod_ten_a,
        CarbonTierWire::Pro,
        900,
        false,
    )?);
    let prod_tok_b = Zeroizing::new(mint_capability_token_from_vault(
        "prod",
        &prod_cell.region,
        &prod_ten_b,
        CarbonTierWire::Pro,
        900,
        false,
    )?);

    let (_, dev_probed) = CarbonThinClient::connect_fastest_pinned(
        dev_cell.candidate_endpoints(),
        &dev_cell.pinned_keks_by_endpoint(),
    )?;
    let best_dev = dev_probed
        .first()
        .ok_or_else(|| "no reachable dev endpoint".to_owned())?;
    let dev_kek = best_dev
        .engine_kek_pubkey_b64
        .as_deref()
        .ok_or_else(|| "dev endpoint missing KEK pubkey".to_owned())?;

    // Select fastest reachable prod region from Wave-1 pins so KEK pin matches the probed endpoint.
    let (_, prod_probed) = CarbonThinClient::connect_fastest_pinned(
        prod_cell.candidate_endpoints(),
        &prod_cell.pinned_keks_by_endpoint(),
    )?;
    let best_prod = prod_probed
        .first()
        .ok_or_else(|| "no reachable prod Wave-1 endpoint".to_owned())?;
    let best_prod_kek = best_prod
        .engine_kek_pubkey_b64
        .as_deref()
        .ok_or_else(|| "prod endpoint missing KEK pubkey".to_owned())?;

    let dev_receipt = execute_cloud_product_coverage_audit(&CarbonCloudCoverageInput {
        cloud: "dev",
        endpoint_url: &best_dev.base_url,
        region: &best_dev.region,
        expected_kek_pubkey_b64: dev_kek,
        wrong_peer_kek_pubkey_b64: best_prod_kek,
        pro_tenant_a: &dev_ten_a,
        pro_token_a: &dev_tok_a,
        pro_tenant_b: &dev_ten_b,
        pro_token_b: &dev_tok_b,
        expired_pro_token_a: &dev_exp,
        signing_pub_pem: &signing_pub_pem,
    })?;

    let prod_receipt = execute_cloud_product_coverage_audit(&CarbonCloudCoverageInput {
        cloud: "prod",
        endpoint_url: &best_prod.base_url,
        region: &best_prod.region,
        expected_kek_pubkey_b64: best_prod_kek,
        wrong_peer_kek_pubkey_b64: dev_kek,
        pro_tenant_a: &prod_ten_a,
        pro_token_a: &prod_tok_a,
        pro_tenant_b: &prod_ten_b,
        pro_token_b: &prod_tok_b,
        expired_pro_token_a: &prod_exp,
        signing_pub_pem: &signing_pub_pem,
    })?;

    let mut clouds = BTreeMap::new();
    let total_checks = dev_receipt.total_checks;
    let total_rt = dev_receipt.redteam_total;
    let all_passed = dev_receipt.all_passed && prod_receipt.all_passed;
    clouds.insert("dev".to_owned(), dev_receipt);
    clouds.insert("prod".to_owned(), prod_receipt);

    let report = CarbonDualCloudCoverageReport {
        schema: "jopedime.carbon.live-product-coverage.v1".to_owned(),
        generated_epoch_ms: now_epoch_ms(),
        mode: "live-aws-existing-endpoints-shredded".to_owned(),
        all_passed,
        total_checks_per_cloud: total_checks,
        total_redteam_vectors_per_cloud: total_rt,
        clouds,
    };
    let serialized = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    assert_clean_receipt_json(&serialized)?;

    if let Some(out_path) = json_out {
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::write(out_path, format!("{serialized}\n")).map_err(|e| e.to_string())?;
    }
    Ok(report)
}
