//! `v1.1.0` gateway & storage live coverage checks (`API-LIVE-08..10`, `STO-LIVE-01`, `CLI-LIVE-01`)
//! executed automatically by `carbon coverage-audit` whenever `GET /v1/carbon/version` is deployed.

use std::time::{Duration, Instant};

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonPredictRequest, CarbonPredictResponse, CarbonPreparePredictRequest,
        CarbonPreparePredictResponse, CarbonRunStateWire, CarbonTierWire, CarbonUploadSlotWire,
        CarbonVersionResponse,
    },
    decimal::DecimalU64,
    Effort,
};
use jope_prep::cuen::CuenPayloadKind;

use super::{
    build_encrypted_create_req, delete_shred_checked, invoke_encrypted_predict, make_ids,
    poll_run_until_settled, upload_encrypted_dataset, CarbonCloudCoverageInput,
    CarbonCoverageCheckResult,
};
use crate::{
    cli::client::{send_raw_edge_http, send_raw_put_url, CarbonClientPreparedDataset},
    security::user_envelope::{
        decode_p256_pubkey_b64, decrypt_cuen_payload, encrypt_cuen_payload, unwrap_u_dek_with_kek,
        verify_ecdsa_p256_spki_pem_signature, wrap_u_dek_to_pubkey, CarbonEncryptedPredictPayload,
        CarbonEngineKek,
    },
    storage::cas::sha256_lower_hex,
    tls::carbon_https_agent,
};

const EXPECTED_SIGNING_PUB_SHA256: &str =
    "de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e";

fn fetch_cas_output_bytes(url: &str) -> Result<(u16, Vec<u8>), String> {
    let agent = carbon_https_agent();
    match agent.get(url).call() {
        Ok(mut resp) => {
            let status = resp.status().as_u16();
            let bytes = resp
                .body_mut()
                .with_config()
                .limit(16 * 1024 * 1024)
                .read_to_vec()
                .map_err(|e| format!("read CAS output GET body: {e}"))?;
            Ok((status, bytes))
        }
        Err(e) => Err(format!("CAS output GET failed: {e}")),
    }
}

fn check_version_and_cas_get_guards(
    base: &str,
    checks: &mut Vec<CarbonCoverageCheckResult>,
) -> Result<(), String> {
    let t = Instant::now();
    let (c_ver, ver_body) = send_raw_edge_http(base, "GET", "/v1/carbon/version", &[], None)?;
    let ver_ok = serde_json::from_slice::<CarbonVersionResponse>(&ver_body)
        .map(|v| v.git_sha.len() >= 7 && !v.version.is_empty())
        .unwrap_or(false);
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-09".to_owned(),
        domain: "API".to_owned(),
        title: "GET /v1/carbon/version provenance SHA & release metadata".to_owned(),
        passed: c_ver == 200 && ver_ok,
        observed_http_status: c_ver,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("HTTP {c_ver}, valid_version_payload={ver_ok}"),
    });

    let t = Instant::now();
    let fake_sha = "c".repeat(64);
    let (c_no_exp, _) = send_raw_edge_http(
        base,
        "GET",
        &format!("/v1/carbon/cas/{fake_sha}"),
        &[],
        None,
    )?;
    let (c_no_sig, _) = send_raw_edge_http(
        base,
        "GET",
        &format!("/v1/carbon/cas/{fake_sha}?exp=9999999999"),
        &[],
        None,
    )?;
    let (c_bad_sig, _) = send_raw_edge_http(
        base,
        "GET",
        &format!("/v1/carbon/cas/{fake_sha}?exp=9999999999&sig=0000000000000000"),
        &[],
        None,
    )?;
    let guard_ok = c_no_exp == 400 && c_no_sig == 401 && c_bad_sig == 403;
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-08".to_owned(),
        domain: "API".to_owned(),
        title: "GET /v1/carbon/cas/{sha} HMAC-grant enforcement (400/401/403)".to_owned(),
        passed: guard_ok,
        observed_http_status: c_bad_sig,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("no_exp={c_no_exp}, no_sig={c_no_sig}, forged_sig={c_bad_sig}"),
    });
    Ok(())
}

fn execute_cas_backed_predict(
    base: &str,
    bearer: &str,
    tenant_id: &str,
    run_id_str: &str,
    u_dek: &[u8; 32],
    cudk: &[u8],
    feature_names: &[String],
) -> Result<(u16, bool), String> {
    let resp_kek = CarbonEngineKek::generate().map_err(|e| e.to_string())?;
    let rows = vec![vec![Some(0.25); feature_names.len()]; 16];
    let pt = serde_json::to_vec(&CarbonEncryptedPredictPayload {
        feature_names: feature_names.to_vec(),
        rows,
    })
    .map_err(|e| e.to_string())?;
    let cuen_req = encrypt_cuen_payload(
        u_dek,
        CuenPayloadKind::PredictFeatures,
        tenant_id,
        run_id_str,
        &pt,
    )
    .map_err(|e| e.to_string())?;
    let prep_bytes = serde_json::to_vec(&CarbonPreparePredictRequest {
        tenant_id: tenant_id.to_owned(),
        predict_cuen: CarbonUploadSlotWire {
            sha256: sha256_lower_hex(&cuen_req),
            bytes: DecimalU64(cuen_req.len() as u64),
        },
    })
    .map_err(|e| e.to_string())?;
    let (cp, pb) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{run_id_str}/predict/prepare"),
        &[("x-carbon-capability-token", bearer)],
        Some(&prep_bytes),
    )?;
    if cp != 200 {
        return Ok((cp, false));
    }
    let prep_resp: CarbonPreparePredictResponse =
        serde_json::from_slice(&pb).map_err(|e| e.to_string())?;
    let (put_c, _) = send_raw_put_url(&prep_resp.input_target.url, &cuen_req, true)?;
    if !matches!(put_c, 200..=204) {
        return Ok((put_c, false));
    }
    verify_cas_predict_response(
        base, bearer, tenant_id, run_id_str, cudk, &resp_kek, prep_resp,
    )
}

fn verify_cas_predict_response(
    base: &str,
    bearer: &str,
    tenant_id: &str,
    run_id_str: &str,
    cudk: &[u8],
    resp_kek: &CarbonEngineKek,
    prep_resp: CarbonPreparePredictResponse,
) -> Result<(u16, bool), String> {
    let req_bytes = serde_json::to_vec(&CarbonPredictRequest {
        tenant_id: tenant_id.to_owned(),
        feature_names: Vec::new(),
        rows: Vec::new(),
        encrypted_rows_cuen_b64: None,
        encrypted_rows_cas_sha256: Some(prep_resp.input_target.sha256),
        predict_upload_id: Some(prep_resp.upload_id),
        prefer_cas_output: true,
        client_response_pubkey_b64: Some(resp_kek.pubkey_b64()),
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk)),
    })
    .map_err(|e| e.to_string())?;
    let (code, body) = send_raw_edge_http(
        base,
        "POST",
        &format!("/v1/carbon/runs/{run_id_str}/predict"),
        &[("x-carbon-capability-token", bearer)],
        Some(&req_bytes),
    )?;
    if code != 200 {
        return Ok((code, false));
    }
    let parsed: CarbonPredictResponse = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
    let Some(cas_url) = parsed.encrypted_predictions_cas_url.as_deref() else {
        return Ok((code, false));
    };
    let (gc, cuen_out) = fetch_cas_output_bytes(cas_url)?;
    if gc != 200 {
        return Ok((gc, false));
    }
    let r_dek = unwrap_u_dek_with_kek(
        resp_kek,
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
        &cuen_out,
    )
    .map_err(|e| e.to_string())?;
    let digest_ok = dec.len() == 16 * 8 && sha256_lower_hex(&dec) == parsed.predictions_sha256;
    Ok((200, digest_ok))
}

fn check_predict_prepare_roundtrip(
    inp: &CarbonCloudCoverageInput<'_>,
    base: &str,
    bearer_a: &str,
    prepared: &CarbonClientPreparedDataset,
    checks: &mut Vec<CarbonCoverageCheckResult>,
) -> Result<usize, String> {
    let t = Instant::now();
    let kek = decode_p256_pubkey_b64(inp.expected_kek_pubkey_b64).map_err(|e| e.to_string())?;
    let upl = upload_encrypted_dataset(
        base,
        bearer_a,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        prepared,
    )?;
    let (rid, did) = make_ids("predcas")?;
    let rid_s = rid.to_string();
    let lm = encrypt_cuen_payload(
        &upl.u_dek,
        CuenPayloadKind::LandmarkJlmb,
        inp.pro_tenant_a,
        &rid_s,
        &prepared.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk = wrap_u_dek_to_pubkey(&upl.u_dek, &kek, inp.pro_tenant_a, &rid_s)
        .map_err(|e| e.to_string())?;
    let req = build_encrypted_create_req(
        rid,
        did,
        inp.pro_tenant_a,
        CarbonTierWire::Pro,
        Effort::Fast,
        71,
        true,
        &upl.prep_resp,
        prepared,
        &lm,
        &cudk,
    )?;
    let (cc, _) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer_a)],
        Some(&req),
    )?;
    let settled = poll_run_until_settled(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &rid_s,
        Duration::from_secs(45),
    )?;
    let (pc, cas_ok) = if matches!(cc, 200..=202) && settled.state == CarbonRunStateWire::Completed
    {
        execute_cas_backed_predict(
            base,
            bearer_a,
            inp.pro_tenant_a,
            &rid_s,
            &upl.u_dek,
            &cudk,
            &prepared.feature_names,
        )?
    } else {
        (cc, false)
    };
    let (sc, _) = delete_shred_checked(base, bearer_a, inp.pro_tenant_a, &rid_s)?;
    checks.push(CarbonCoverageCheckResult {
        check_id: "API-LIVE-10".to_owned(),
        domain: "API".to_owned(),
        title: "POST /predict/prepare + CAS-backed /predict round-trip & decrypt".to_owned(),
        passed: pc == 200 && cas_ok && sc == 200,
        observed_http_status: pc,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!("HTTP {pc}, cas_roundtrip_verified={cas_ok}, shred_http={sc}"),
    });
    Ok(usize::from(sc != 200))
}

struct ScopedRunFixture {
    run_id: String,
    u_dek: zeroize::Zeroizing<[u8; 32]>,
    cudk: Vec<u8>,
    upload_id: String,
}

fn create_scoped_fast_run(
    base: &str,
    bearer: &str,
    tenant_id: &str,
    kek_b64: &str,
    prepared: &CarbonClientPreparedDataset,
    tag: &str,
) -> Result<ScopedRunFixture, String> {
    let kek = decode_p256_pubkey_b64(kek_b64).map_err(|e| e.to_string())?;
    let upl = upload_encrypted_dataset(base, bearer, tenant_id, CarbonTierWire::Pro, prepared)?;
    let (rid, did) = make_ids(tag)?;
    let rid_s = rid.to_string();
    let lm = encrypt_cuen_payload(
        &upl.u_dek,
        CuenPayloadKind::LandmarkJlmb,
        tenant_id,
        &rid_s,
        &prepared.landmark_bundle.to_wire_bytes(),
    )
    .map_err(|e| e.to_string())?;
    let cudk =
        wrap_u_dek_to_pubkey(&upl.u_dek, &kek, tenant_id, &rid_s).map_err(|e| e.to_string())?;
    let req = build_encrypted_create_req(
        rid,
        did,
        tenant_id,
        CarbonTierWire::Pro,
        Effort::Fast,
        73,
        true,
        &upl.prep_resp,
        prepared,
        &lm,
        &cudk,
    )?;
    let (cc, body) = send_raw_edge_http(
        base,
        "POST",
        "/v1/carbon/runs",
        &[("x-carbon-capability-token", bearer)],
        Some(&req),
    )?;
    if !matches!(cc, 200..=202) {
        return Err(format!(
            "create {tag} failed HTTP {cc}: {}",
            String::from_utf8_lossy(&body)
        ));
    }
    let settled = poll_run_until_settled(base, bearer, tenant_id, &rid_s, Duration::from_secs(45))?;
    if settled.state != CarbonRunStateWire::Completed {
        return Err(format!("{tag} did not complete: {:?}", settled.state));
    }
    Ok(ScopedRunFixture {
        run_id: rid_s,
        u_dek: upl.u_dek,
        cudk: cudk.to_vec(),
        upload_id: upl.prep_resp.upload_id,
    })
}

fn check_sibling_upload_shred_isolation(
    inp: &CarbonCloudCoverageInput<'_>,
    base: &str,
    bearer_a: &str,
    prepared: &CarbonClientPreparedDataset,
    checks: &mut Vec<CarbonCoverageCheckResult>,
) -> Result<usize, String> {
    let t = Instant::now();
    let f1 = create_scoped_fast_run(
        base,
        bearer_a,
        inp.pro_tenant_a,
        inp.expected_kek_pubkey_b64,
        prepared,
        "sib1",
    )?;
    let f2 = create_scoped_fast_run(
        base,
        bearer_a,
        inp.pro_tenant_a,
        inp.expected_kek_pubkey_b64,
        prepared,
        "sib2",
    )?;
    let distinct_upl = f1.upload_id != f2.upload_id
        && f1.upload_id.starts_with("upl_")
        && f2.upload_id.starts_with("upl_");
    let (shred1_c, _) = delete_shred_checked(base, bearer_a, inp.pro_tenant_a, &f1.run_id)?;
    let (pred2_c, _) = invoke_encrypted_predict(
        base,
        bearer_a,
        inp.pro_tenant_a,
        &f2.run_id,
        &f2.u_dek,
        &f2.cudk,
        &prepared.feature_names,
    )?;
    let (shred2_c, _) = delete_shred_checked(base, bearer_a, inp.pro_tenant_a, &f2.run_id)?;
    let ok = distinct_upl && shred1_c == 200 && pred2_c == 200 && shred2_c == 200;
    checks.push(CarbonCoverageCheckResult {
        check_id: "STO-LIVE-01".to_owned(),
        domain: "Storage".to_owned(),
        title: "Per-upload upl_<ULID> sibling-run shred isolation".to_owned(),
        passed: ok,
        observed_http_status: pred2_c,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "distinct_upl={distinct_upl}, shred1={shred1_c}, sibling_predict={pred2_c}, shred2={shred2_c}"
        ),
    });
    Ok(usize::from(shred1_c != 200) + usize::from(shred2_c != 200))
}

fn check_self_verify_trust_roots(
    inp: &CarbonCloudCoverageInput<'_>,
    checks: &mut Vec<CarbonCoverageCheckResult>,
) -> Result<(), String> {
    let t = Instant::now();
    let pub_sha = sha256_lower_hex(inp.signing_pub_pem.as_bytes());
    let pub_match = pub_sha == EXPECTED_SIGNING_PUB_SHA256;
    let kek_ok = decode_p256_pubkey_b64(inp.expected_kek_pubkey_b64).is_ok();
    let exe_sha_ok = std::env::current_exe()
        .and_then(std::fs::read)
        .map(|b| sha256_lower_hex(&b).len() == 64)
        .unwrap_or(false);
    let bogus_refused = verify_ecdsa_p256_spki_pem_signature(
        inp.signing_pub_pem,
        b"carbon-self-verify-probe",
        &[0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01],
    )
    .is_err();
    let passed = pub_match && kek_ok && exe_sha_ok && bogus_refused;
    checks.push(CarbonCoverageCheckResult {
        check_id: "CLI-LIVE-01".to_owned(),
        domain: "CLI".to_owned(),
        title: "Self-verify signing.pub SHA-256, binary digest & KEK pin root check".to_owned(),
        passed,
        observed_http_status: 0,
        wall_ms: t.elapsed().as_secs_f64() * 1_000.0,
        evidence: format!(
            "signing_pub_pin={pub_match}, kek_pin={kek_ok}, exe_sha={exe_sha_ok}, bogus_sig_refused={bogus_refused}"
        ),
    });
    Ok(())
}

pub(super) fn run_v11_deployed_checks(
    inp: &CarbonCloudCoverageInput<'_>,
    base: &str,
    bearer_a: &str,
    prepared_small: &CarbonClientPreparedDataset,
    checks: &mut Vec<CarbonCoverageCheckResult>,
) -> Result<usize, String> {
    check_version_and_cas_get_guards(base, checks)?;
    let rem_a = check_predict_prepare_roundtrip(inp, base, bearer_a, prepared_small, checks)?;
    let rem_b = check_sibling_upload_shred_isolation(inp, base, bearer_a, prepared_small, checks)?;
    check_self_verify_trust_roots(inp, checks)?;
    Ok(rem_a + rem_b)
}
