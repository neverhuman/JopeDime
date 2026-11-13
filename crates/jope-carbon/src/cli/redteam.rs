//! 16-vector adversarial security & zero-knowledge red-team suite (`carbon redteam`, `V-01`..`V-16`).
//!
//! Runs identically against local `CarbonLocalLambdaEmulator` in CI and live AWS Lambda Function URLs.

use std::time::Instant;

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonCreateRunRequest, CarbonPredictRequest, CarbonPredictResponse,
        CarbonPrepareUploadRequest, CarbonRunStateWire, CarbonTierWire, CarbonUploadSlotWire,
    },
    decimal::DecimalU64,
    ids::{DatasetId, RunId},
    Effort,
};
use jope_prep::cuen::CuenPayloadKind;
use serde::{Deserialize, Serialize};

use crate::{
    cells::engine_invoker::now_epoch_ms,
    cli::{
        client::{send_raw_edge_http, send_raw_put_url, CarbonThinClient},
        runner::generate_nonlinear_benchmark_rows,
    },
    security::user_envelope::{
        decode_p256_pubkey_b64, decrypt_cuen_payload, encrypt_cuen_payload, generate_random_u_dek,
        unwrap_u_dek_with_kek, wrap_u_dek_to_pubkey, CarbonEncryptedPredictPayload,
        CarbonEngineKek, CAS_BLOB_CUEN_SCOPE,
    },
    storage::cas::sha256_lower_hex,
};

/// `CarbonRedTeamVectorResult` is a struct recording one adversarial attack vector outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonRedTeamVectorResult {
    pub vector_id: String,
    pub finding_ref: String,
    pub title: String,
    pub passed: bool,
    pub observed_http_status: u16,
    pub wall_ms: f64,
    pub evidence: String,
}

/// `CarbonRedTeamSuiteReport` is a struct capturing all 16 red-team vector results.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonRedTeamSuiteReport {
    pub timestamp_epoch_ms: u64,
    pub target_endpoint: String,
    pub region: String,
    pub total_vectors: usize,
    pub passed_vectors: usize,
    pub all_passed: bool,
    pub vectors: Vec<CarbonRedTeamVectorResult>,
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

/// Execute all 16 adversarial red-team vectors (`V-01`..`V-16`) against `endpoint`.
pub fn execute_carbon_redteam_suite(endpoint: &str) -> Result<CarbonRedTeamSuiteReport, String> {
    let (mut client_a, probed) = CarbonThinClient::connect_fastest(vec![endpoint.to_owned()])?;
    let best = probed
        .first()
        .ok_or_else(|| "no reachable Carbon endpoint for redteam".to_owned())?;
    let base = client_a.active_endpoint().to_owned();
    let kek_b64 = best
        .engine_kek_pubkey_b64
        .clone()
        .ok_or_else(|| "endpoint does not advertise engine_kek_pubkey_b64".to_owned())?;
    let kek_pub = decode_p256_pubkey_b64(&kek_b64).map_err(|e| e.to_string())?;

    let anon_a = client_a.authenticate_anonymous()?;
    let mut client_b = CarbonThinClient::connect_fastest(vec![base.clone()])?.0;
    let anon_b = client_b.authenticate_anonymous()?;

    let bearer_a = format!("Bearer {}", anon_a.capability_token);
    let bearer_b = format!("Bearer {}", anon_b.capability_token);

    let (headers, rows, target) = generate_nonlinear_benchmark_rows(80, 4, 777);
    let prepared = CarbonThinClient::prepare_numeric_dataset(headers, &rows, &target, 777, 0.25)?;
    let dummy_sha = "a".repeat(64);

    let mut vectors = Vec::with_capacity(16);

    // V-01: Unauthenticated upload prepare -> 401
    let t = Instant::now();
    let dummy_prep = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: anon_a.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        fit_rows: 60,
        holdout_rows: 20,
        feature_count: 4,
        fit_jmat: CarbonUploadSlotWire {
            sha256: dummy_sha.clone(),
            bytes: DecimalU64(512),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: dummy_sha.clone(),
            bytes: DecimalU64(256),
        },
        encrypted_cuen: true,
    })
    .map_err(|e| e.to_string())?;
    let (c01, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[],
        Some(&dummy_prep),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-01".to_owned(),
        finding_ref: "RT-02".to_owned(),
        title: "Unauthenticated upload prepare rejected".to_owned(),
        passed: c01 == 401,
        observed_http_status: c01,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("POST /v1/carbon/uploads/prepare without token -> HTTP {c01}"),
    });

    // V-02: Forged HMAC capability token -> 401
    let t = Instant::now();
    let forged = format!(
        "Bearer {}.{}",
        anon_a
            .capability_token
            .rsplit_once('.')
            .map(|(p, _)| p)
            .unwrap_or("ctk1.e30"),
        "00".repeat(32)
    );
    let (c02, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("x-carbon-capability-token", &forged)],
        Some(&dummy_prep),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-02".to_owned(),
        finding_ref: "RT-02".to_owned(),
        title: "Forged capability HMAC signature rejected in constant time".to_owned(),
        passed: c02 == 401,
        observed_http_status: c02,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Tampered ctk1 signature -> HTTP {c02}"),
    });

    // V-03: Free -> Pro tier escalation -> 403
    let t = Instant::now();
    let esc_prep = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: anon_a.tenant_id.clone(),
        tier: CarbonTierWire::Pro,
        fit_rows: 60,
        holdout_rows: 20,
        feature_count: 4,
        fit_jmat: CarbonUploadSlotWire {
            sha256: dummy_sha.clone(),
            bytes: DecimalU64(512),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: dummy_sha.clone(),
            bytes: DecimalU64(256),
        },
        encrypted_cuen: true,
    })
    .map_err(|e| e.to_string())?;
    let (c03, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("x-carbon-capability-token", &bearer_a)],
        Some(&esc_prep),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-03".to_owned(),
        finding_ref: "RT-02".to_owned(),
        title: "Free-to-Pro capability tier escalation refused".to_owned(),
        passed: c03 == 403,
        observed_http_status: c03,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Free token requesting tier=pro -> HTTP {c03}"),
    });

    // V-04: Cross-tenant IDOR (`tenant_b` querying `tenant_a`) -> 403
    let t = Instant::now();
    let (c04, _) = send_raw_edge_http(
        &base,
        "GET",
        &format!(
            "/v1/carbon/runs/run_01JCCARB0NRT00000000000001?tenant_id={}",
            anon_a.tenant_id
        ),
        &[("x-carbon-capability-token", &bearer_b)],
        None,
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-04".to_owned(),
        finding_ref: "RT-02".to_owned(),
        title: "Cross-tenant IDOR access refused".to_owned(),
        passed: c04 == 403,
        observed_http_status: c04,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Token B reading Tenant A run -> HTTP {c04}"),
    });

    // V-05: Path-traversal & non-injective `tenant_id` (`../etc` / `a..b`) -> 400
    let t = Instant::now();
    let (c05a, _) = send_raw_edge_http(
        &base,
        "GET",
        "/v1/carbon/runs/run_01JCCARB0NRT00000000000001?tenant_id=a..b",
        &[("x-carbon-capability-token", &bearer_a)],
        None,
    )?;
    let (c05b, _) = send_raw_edge_http(
        &base,
        "GET",
        "/v1/carbon/runs/run_01JCCARB0NRT00000000000001?tenant_id=..%2Fsecret",
        &[("x-carbon-capability-token", &bearer_a)],
        None,
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-05".to_owned(),
        finding_ref: "RT-08".to_owned(),
        title: "Path-traversal & non-injective tenant_id rejected".to_owned(),
        passed: c05a == 400 && c05b == 400,
        observed_http_status: c05a,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("tenant_id='a..b' -> {c05a}, '..%2Fsecret' -> {c05b}"),
    });

    // V-06: Presigned S3 / gateway PUT size-pinning & SHA-256 tamper -> 400/403
    let t = Instant::now();
    let u_dek_tamper = generate_random_u_dek().map_err(|e| e.to_string())?;
    let fit_enc_t = encrypt_cuen_payload(
        &u_dek_tamper,
        CuenPayloadKind::FitJmat,
        &anon_a.tenant_id,
        CAS_BLOB_CUEN_SCOPE,
        &prepared.artifacts.fit_jmat,
    )
    .map_err(|e| e.to_string())?;
    let hold_enc_t = encrypt_cuen_payload(
        &u_dek_tamper,
        CuenPayloadKind::HoldoutJmat,
        &anon_a.tenant_id,
        CAS_BLOB_CUEN_SCOPE,
        &prepared.artifacts.holdout_jmat,
    )
    .map_err(|e| e.to_string())?;
    let prep_t_bytes = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: anon_a.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_count: prepared.feature_names.len() as u32,
        fit_jmat: CarbonUploadSlotWire {
            sha256: sha256_lower_hex(&fit_enc_t),
            bytes: DecimalU64(fit_enc_t.len() as u64),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: sha256_lower_hex(&hold_enc_t),
            bytes: DecimalU64(hold_enc_t.len() as u64),
        },
        encrypted_cuen: true,
    })
    .map_err(|e| e.to_string())?;
    let (cp_ok, cp_body) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("x-carbon-capability-token", &bearer_a)],
        Some(&prep_t_bytes),
    )?;
    let cp_resp: jope_api::carbon::CarbonPrepareUploadResponse =
        serde_json::from_slice(&cp_body)
            .map_err(|e| format!("V-06 prepare decode ({cp_ok}): {e}"))?;
    let mut bloated = fit_enc_t.clone();
    bloated.extend_from_slice(&[0xAB; 64]);
    let (c06, _) = send_raw_put_url(&cp_resp.fit_target.url, &bloated, true)?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-06".to_owned(),
        finding_ref: "RT-04".to_owned(),
        title: "Presigned PUT byte-size / SHA-256 inflation rejected".to_owned(),
        passed: matches!(c06, 400 | 403),
        observed_http_status: c06,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Inflated PUT (+64B) to size-pinned URL -> HTTP {c06}"),
    });

    // V-07: Unencrypted cloud upload bypass (`encrypted_cuen=false`) -> 400
    let t = Instant::now();
    let plain_prep = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: anon_a.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_count: prepared.feature_names.len() as u32,
        fit_jmat: CarbonUploadSlotWire {
            sha256: dummy_sha.clone(),
            bytes: DecimalU64(512),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: dummy_sha,
            bytes: DecimalU64(256),
        },
        encrypted_cuen: false,
    })
    .map_err(|e| e.to_string())?;
    let (c07, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("x-carbon-capability-token", &bearer_a)],
        Some(&plain_prep),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-07".to_owned(),
        finding_ref: "RT-01".to_owned(),
        title: "Cleartext upload refused on zero-knowledge cloud cell".to_owned(),
        passed: c07 == 400,
        observed_http_status: c07,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("encrypted_cuen=false -> HTTP {c07}"),
    });

    // Upload legitimate encrypted blobs via `client_a` for V-08..V-16.
    let valid_upload =
        client_a.upload_prepared(&anon_a.tenant_id, CarbonTierWire::Free, &prepared)?;

    // V-08: CUEN slot-swap (`HoldoutJmat` passed as `landmark_bundle_b64`) -> 400
    let t = Instant::now();
    let (r08, d08) = make_ids("v08")?;
    let r08_str = r08.to_string();
    let wrong_slot_lm = encrypt_cuen_payload(
        &u_dek_tamper,
        CuenPayloadKind::HoldoutJmat,
        &anon_a.tenant_id,
        &r08_str,
        &prepared.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk_08 = wrap_u_dek_to_pubkey(&u_dek_tamper, &kek_pub, &anon_a.tenant_id, &r08_str)
        .map_err(|e| e.to_string())?;
    let swap_req = serde_json::to_vec(&CarbonCreateRunRequest {
        run_id: r08,
        dataset_id: d08,
        tenant_id: anon_a.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        upload_id: Some(valid_upload.upload_id.clone()),
        fit_jmat_sha256: valid_upload.fit_target.sha256.clone(),
        holdout_jmat_sha256: valid_upload.holdout_target.sha256.clone(),
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_names: prepared.feature_names.clone(),
        target_name: prepared.target_name.clone(),
        objective: "rmse".to_owned(),
        effort: Effort::Fast,
        seed: DecimalU64(1),
        landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(wrong_slot_lm),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk_08)),
    })
    .map_err(|e| e.to_string())?;
    let (c08, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", &bearer_a)],
        Some(&swap_req),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-08".to_owned(),
        finding_ref: "RT-01".to_owned(),
        title: "CUEN payload slot-swap (HoldoutJmat -> LandmarkJlmb) rejected".to_owned(),
        passed: c08 == 400,
        observed_http_status: c08,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Wrong CUEN slot tag in landmark_bundle_b64 -> HTTP {c08}"),
    });

    // V-09: Tampered CUDK wrapped_u_dek_b64 rejected by carbon-engine -> 502
    let t = Instant::now();
    let (r09, d09) = make_ids("v09")?;
    let r09_str = r09.to_string();
    let valid_lm_09 = encrypt_cuen_payload(
        &u_dek_tamper,
        CuenPayloadKind::LandmarkJlmb,
        &anon_a.tenant_id,
        &r09_str,
        &prepared.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let mut bad_cudk = wrap_u_dek_to_pubkey(&u_dek_tamper, &kek_pub, &anon_a.tenant_id, &r09_str)
        .map_err(|e| e.to_string())?;
    bad_cudk[132] ^= 0xFF;
    let bad_dek_req = serde_json::to_vec(&CarbonCreateRunRequest {
        run_id: r09,
        dataset_id: d09,
        tenant_id: anon_a.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        upload_id: Some(valid_upload.upload_id.clone()),
        fit_jmat_sha256: valid_upload.fit_target.sha256.clone(),
        holdout_jmat_sha256: valid_upload.holdout_target.sha256.clone(),
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_names: prepared.feature_names.clone(),
        target_name: prepared.target_name.clone(),
        objective: "rmse".to_owned(),
        effort: Effort::Fast,
        seed: DecimalU64(1),
        landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(valid_lm_09),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(bad_cudk)),
    })
    .map_err(|e| e.to_string())?;
    let (c09, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", &bearer_a)],
        Some(&bad_dek_req),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-09".to_owned(),
        finding_ref: "RT-01".to_owned(),
        title: "Tampered CUDK v1 wrapped U_DEK envelope rejected by engine".to_owned(),
        passed: c09 == 502,
        observed_http_status: c09,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Flipped GCM auth byte in CUDK -> HTTP {c09}"),
    });

    // Legitimate encrypted Free run `r_legit` for V-10, V-12, V-13, V-14, V-16.
    let (r_legit, d_legit) = make_ids("legit")?;
    let r_legit_str = r_legit.to_string();
    let legit_prog = client_a.start_run(
        r_legit.clone(),
        d_legit.clone(),
        &anon_a.tenant_id,
        CarbonTierWire::Free,
        Effort::Fast,
        "rmse",
        42,
        &prepared,
    )?;
    if legit_prog.state != CarbonRunStateWire::Completed {
        return Err(format!(
            "baseline encrypted run did not complete: {:?}",
            legit_prog.state
        ));
    }

    // V-10: Duplicate `run_id` overwrite attack -> 409 Conflict
    let t = Instant::now();
    let dup_lm = encrypt_cuen_payload(
        &u_dek_tamper,
        CuenPayloadKind::LandmarkJlmb,
        &anon_a.tenant_id,
        &r_legit_str,
        &prepared.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let dup_cudk = wrap_u_dek_to_pubkey(&u_dek_tamper, &kek_pub, &anon_a.tenant_id, &r_legit_str)
        .map_err(|e| e.to_string())?;
    let dup_req = serde_json::to_vec(&CarbonCreateRunRequest {
        run_id: r_legit,
        dataset_id: d_legit,
        tenant_id: anon_a.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        upload_id: Some(valid_upload.upload_id.clone()),
        fit_jmat_sha256: valid_upload.fit_target.sha256.clone(),
        holdout_jmat_sha256: valid_upload.holdout_target.sha256.clone(),
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_names: prepared.feature_names.clone(),
        target_name: prepared.target_name.clone(),
        objective: "rmse".to_owned(),
        effort: Effort::Fast,
        seed: DecimalU64(99),
        landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(dup_lm),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(dup_cudk)),
    })
    .map_err(|e| e.to_string())?;
    let (c10, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", &bearer_a)],
        Some(&dup_req),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-10".to_owned(),
        finding_ref: "RT-03".to_owned(),
        title: "Duplicate run_id state clobber refused via If-None-Match".to_owned(),
        passed: c10 == 409,
        observed_http_status: c10,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Second POST /v1/carbon/runs for {r_legit_str} -> HTTP {c10}"),
    });

    // V-11: Oversized (>16 KiB) landmark bundle -> 400
    let t = Instant::now();
    let (r11, d11) = make_ids("v11")?;
    let r11_str = r11.to_string();
    let cudk_11 = wrap_u_dek_to_pubkey(&u_dek_tamper, &kek_pub, &anon_a.tenant_id, &r11_str)
        .map_err(|e| e.to_string())?;
    let huge_lm = vec![0x55u8; 20_000];
    let huge_req = serde_json::to_vec(&CarbonCreateRunRequest {
        run_id: r11,
        dataset_id: d11,
        tenant_id: anon_a.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        upload_id: Some(valid_upload.upload_id),
        fit_jmat_sha256: valid_upload.fit_target.sha256,
        holdout_jmat_sha256: valid_upload.holdout_target.sha256,
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_names: prepared.feature_names.clone(),
        target_name: prepared.target_name.clone(),
        objective: "rmse".to_owned(),
        effort: Effort::Fast,
        seed: DecimalU64(1),
        landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(huge_lm),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk_11)),
    })
    .map_err(|e| e.to_string())?;
    let (c11, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", &bearer_a)],
        Some(&huge_req),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-11".to_owned(),
        finding_ref: "RT-13".to_owned(),
        title: "Oversized (>16 KiB) landmark wire payload refused at edge".to_owned(),
        passed: c11 == 400,
        observed_http_status: c11,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("20,000-byte landmark_bundle_b64 -> HTTP {c11}"),
    });

    // V-12: Unencrypted `/predict` rows bypass against encrypted run -> 400
    let t = Instant::now();
    let plain_pred = serde_json::to_vec(&CarbonPredictRequest {
        tenant_id: anon_a.tenant_id.clone(),
        feature_names: prepared.feature_names.clone(),
        rows: vec![vec![Some(0.5), Some(-0.5), Some(0.1), Some(0.2)]],
        encrypted_rows_cuen_b64: None,
        encrypted_rows_cas_sha256: None,
        predict_upload_id: None,
        prefer_cas_output: false,
        client_response_pubkey_b64: None,
        wrapped_u_dek_b64: None,
    })
    .map_err(|e| e.to_string())?;
    let (c12, _) = send_raw_edge_http(
        &base,
        "POST",
        &format!("/v1/carbon/runs/{r_legit_str}/predict"),
        &[("x-carbon-capability-token", &bearer_a)],
        Some(&plain_pred),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-12".to_owned(),
        finding_ref: "RT-01".to_owned(),
        title: "Cleartext /predict rows refused on encrypted run".to_owned(),
        passed: c12 == 400,
        observed_http_status: c12,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Plaintext rows on encrypted run -> HTTP {c12}"),
    });

    // V-13: Zero-cleartext E2E `/predict` wire inspection (fresh explicit U_DEK run so we inspect raw wire JSON)
    let t = Instant::now();
    let u_dek_13 = generate_random_u_dek().map_err(|e| e.to_string())?;
    let fit_13 = encrypt_cuen_payload(
        &u_dek_13,
        CuenPayloadKind::FitJmat,
        &anon_b.tenant_id,
        CAS_BLOB_CUEN_SCOPE,
        &prepared.artifacts.fit_jmat,
    )
    .map_err(|e| e.to_string())?;
    let hold_13 = encrypt_cuen_payload(
        &u_dek_13,
        CuenPayloadKind::HoldoutJmat,
        &anon_b.tenant_id,
        CAS_BLOB_CUEN_SCOPE,
        &prepared.artifacts.holdout_jmat,
    )
    .map_err(|e| e.to_string())?;
    let prep_13_req = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: anon_b.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_count: prepared.feature_names.len() as u32,
        fit_jmat: CarbonUploadSlotWire {
            sha256: sha256_lower_hex(&fit_13),
            bytes: DecimalU64(fit_13.len() as u64),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: sha256_lower_hex(&hold_13),
            bytes: DecimalU64(hold_13.len() as u64),
        },
        encrypted_cuen: true,
    })
    .map_err(|e| e.to_string())?;
    let (_, prep_13_body) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("x-carbon-capability-token", &bearer_b)],
        Some(&prep_13_req),
    )?;
    let prep_13_resp: jope_api::carbon::CarbonPrepareUploadResponse =
        serde_json::from_slice(&prep_13_body).map_err(|e| e.to_string())?;
    let _ = send_raw_put_url(&prep_13_resp.fit_target.url, &fit_13, true)?;
    let _ = send_raw_put_url(&prep_13_resp.holdout_target.url, &hold_13, true)?;
    let (r13, d13) = make_ids("v13")?;
    let r13_str = r13.to_string();
    let lm_13 = encrypt_cuen_payload(
        &u_dek_13,
        CuenPayloadKind::LandmarkJlmb,
        &anon_b.tenant_id,
        &r13_str,
        &prepared.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk_13 = wrap_u_dek_to_pubkey(&u_dek_13, &kek_pub, &anon_b.tenant_id, &r13_str)
        .map_err(|e| e.to_string())?;
    let run_13_req = serde_json::to_vec(&CarbonCreateRunRequest {
        run_id: r13,
        dataset_id: d13,
        tenant_id: anon_b.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        upload_id: Some(prep_13_resp.upload_id),
        fit_jmat_sha256: prep_13_resp.fit_target.sha256,
        holdout_jmat_sha256: prep_13_resp.holdout_target.sha256,
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_names: prepared.feature_names.clone(),
        target_name: prepared.target_name.clone(),
        objective: "rmse".to_owned(),
        effort: Effort::Fast,
        seed: DecimalU64(13),
        landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(lm_13),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk_13)),
    })
    .map_err(|e| e.to_string())?;
    let (c13_run, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", &bearer_b)],
        Some(&run_13_req),
    )?;
    let resp_kek = CarbonEngineKek::generate().map_err(|e| e.to_string())?;
    let enc_pred_pt = serde_json::to_vec(&CarbonEncryptedPredictPayload {
        feature_names: prepared.feature_names.clone(),
        rows: vec![vec![Some(0.25), Some(-0.25), Some(0.5), Some(-0.5)]],
    })
    .map_err(|e| e.to_string())?;
    let enc_pred_cuen = encrypt_cuen_payload(
        &u_dek_13,
        CuenPayloadKind::PredictFeatures,
        &anon_b.tenant_id,
        &r13_str,
        &enc_pred_pt,
    )
    .map_err(|e| e.to_string())?;
    let pred_cudk_13 = wrap_u_dek_to_pubkey(&u_dek_13, &kek_pub, &anon_b.tenant_id, &r13_str)
        .map_err(|e| e.to_string())?;
    let raw_pred_req = serde_json::to_vec(&CarbonPredictRequest {
        tenant_id: anon_b.tenant_id.clone(),
        feature_names: Vec::new(),
        rows: Vec::new(),
        encrypted_rows_cuen_b64: Some(
            base64::engine::general_purpose::STANDARD.encode(enc_pred_cuen),
        ),
        encrypted_rows_cas_sha256: None,
        predict_upload_id: None,
        prefer_cas_output: false,
        client_response_pubkey_b64: Some(resp_kek.pubkey_b64()),
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(pred_cudk_13)),
    })
    .map_err(|e| e.to_string())?;
    let (c13, wire_pred_bytes) = send_raw_edge_http(
        &base,
        "POST",
        &format!("/v1/carbon/runs/{r13_str}/predict"),
        &[("x-carbon-capability-token", &bearer_b)],
        Some(&raw_pred_req),
    )?;
    let wire_pred: CarbonPredictResponse =
        serde_json::from_slice(&wire_pred_bytes).map_err(|e| e.to_string())?;
    let wire_cleartext_empty = wire_pred.predictions.is_empty();
    let r_dek_13 = unwrap_u_dek_with_kek(
        &resp_kek,
        &base64::engine::general_purpose::STANDARD
            .decode(wire_pred.response_cudk_b64.as_deref().unwrap_or(""))
            .unwrap_or_default(),
        &anon_b.tenant_id,
        &r13_str,
    )
    .map_err(|e| e.to_string())?;
    let dec_out = decrypt_cuen_payload(
        &r_dek_13,
        CuenPayloadKind::PredictOutputs,
        &anon_b.tenant_id,
        &r13_str,
        &base64::engine::general_purpose::STANDARD
            .decode(
                wire_pred
                    .encrypted_predictions_cuen_b64
                    .as_deref()
                    .unwrap_or(""),
            )
            .unwrap_or_default(),
    )
    .map_err(|e| e.to_string())?;
    let _ = client_b.shred_run(&anon_b.tenant_id, &r13_str);
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-13".to_owned(),
        finding_ref: "RT-01".to_owned(),
        title: "Zero-cleartext E2E /predict wire inspection verified".to_owned(),
        passed: matches!(c13_run, 200..=202)
            && c13 == 200
            && wire_cleartext_empty
            && dec_out.len() == 8
            && sha256_lower_hex(&dec_out) == wire_pred.predictions_sha256,
        observed_http_status: c13,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "Raw JSON predictions.len()=0, client ECDH decrypted 8B matching sha256 {}",
            &wire_pred.predictions_sha256[..16]
        ),
    });

    // V-14: Free-tier `/predict` call-rate quota (`<= 20` calls/run -> 21st returns 429)
    let t = Instant::now();
    let mut c14_last = 200u16;
    for call_idx in 0..21 {
        let res = client_a.predict(
            &anon_a.tenant_id,
            &r_legit_str,
            prepared.feature_names.clone(),
            vec![vec![Some(0.1), Some(0.2), Some(0.3), Some(0.4)]],
        );
        if call_idx == 20 {
            c14_last = match res {
                Err(msg) if msg.contains("429") => 429,
                Err(_) => 500,
                Ok(_) => 200,
            };
        }
    }
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-14".to_owned(),
        finding_ref: "RT-10".to_owned(),
        title: "Free-tier per-run /predict model-extraction quota enforced".to_owned(),
        passed: c14_last == 429,
        observed_http_status: c14_last,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("21st /predict call on Free run -> HTTP {c14_last}"),
    });

    // V-15: Spot-hybrid advisor unauthenticated (401) & Free (403) refused
    let t = Instant::now();
    let (c15_unauth, _) =
        send_raw_edge_http(&base, "GET", "/v1/carbon/advisor/spot-hybrid", &[], None)?;
    let (c15_free, _) = send_raw_edge_http(
        &base,
        "GET",
        "/v1/carbon/advisor/spot-hybrid",
        &[("x-carbon-capability-token", &bearer_a)],
        None,
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-15".to_owned(),
        finding_ref: "RT-11".to_owned(),
        title: "Spot-hybrid advisor telemetry restricted to Pro/Admin".to_owned(),
        passed: c15_unauth == 401 && c15_free == 403,
        observed_http_status: c15_free,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("Unauth -> HTTP {c15_unauth}, Free -> HTTP {c15_free}"),
    });

    // V-16: Cryptographic DELETE shred & post-shred resurrection refusal
    let t = Instant::now();
    let shred_resp = client_a.shred_run(&anon_a.tenant_id, &r_legit_str)?;
    let (c16_get, _) = send_raw_edge_http(
        &base,
        "GET",
        &format!(
            "/v1/carbon/runs/{r_legit_str}?tenant_id={}",
            anon_a.tenant_id
        ),
        &[("x-carbon-capability-token", &bearer_a)],
        None,
    )?;
    let (c16_recreate, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", &bearer_a)],
        Some(&dup_req),
    )?;
    vectors.push(CarbonRedTeamVectorResult {
        vector_id: "V-16".to_owned(),
        finding_ref: "RT-06".to_owned(),
        title: "Cryptographic DELETE shred purges blobs/state & blocks resurrection".to_owned(),
        passed: shred_resp.shredded
            && shred_resp.objects_deleted >= 4
            && c16_get == 404
            && c16_recreate == 409,
        observed_http_status: c16_get,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "deleted={} objects, post-shred GET -> {c16_get}, recreate -> {c16_recreate}",
            shred_resp.objects_deleted
        ),
    });

    let passed_vectors = vectors.iter().filter(|v| v.passed).count();
    let total_vectors = vectors.len();
    Ok(CarbonRedTeamSuiteReport {
        timestamp_epoch_ms: now_epoch_ms(),
        target_endpoint: base,
        region: best.region.clone(),
        total_vectors,
        passed_vectors,
        all_passed: passed_vectors == total_vectors,
        vectors,
    })
}

/// Format a `CarbonRedTeamSuiteReport` as a GitHub-flavored Markdown verification report.
pub fn render_redteam_markdown(report: &CarbonRedTeamSuiteReport) -> String {
    let mut md = String::new();
    md.push_str("# Carbon Zero-Knowledge & Security Red-Team Receipt\n\n");
    md.push_str(&format!(
        "- **Target Endpoint**: `{}`\n- **Region**: `{}`\n- **Timestamp (epoch ms)**: `{}`\n- **Coverage**: `{}/{}` adversarial vectors passed\n- **Suite Verdict**: **{}**\n\n",
        report.target_endpoint,
        report.region,
        report.timestamp_epoch_ms,
        report.passed_vectors,
        report.total_vectors,
        if report.all_passed { "PASS" } else { "FAIL" }
    ));
    md.push_str("| Vector | Finding | Attack Scenario | HTTP | Wall (ms) | Verdict | Evidence |\n");
    md.push_str("| --- | --- | --- | ---: | ---: | --- | --- |\n");
    for v in &report.vectors {
        md.push_str(&format!(
            "| `{}` | `{}` | {} | `{}` | {:.1} | **{}** | {} |\n",
            v.vector_id,
            v.finding_ref,
            v.title,
            v.observed_http_status,
            v.wall_ms,
            if v.passed { "PASS" } else { "FAIL" },
            v.evidence
        ));
    }
    md
}
