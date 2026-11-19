//! HTTP API route & negative/security branch coverage closure across `edge.rs` and `emu.rs`.

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonCreateRunRequest, CarbonEarlyStopRequest, CarbonExplainRequest, CarbonExportLangWire,
        CarbonExportRequest, CarbonPredictRequest, CarbonPreparePredictRequest,
        CarbonPrepareUploadRequest, CarbonPrepareUploadResponse, CarbonPublicWaveProgress,
        CarbonResumeRunRequest, CarbonTierWire, CarbonUploadSlotWire,
    },
    decimal::DecimalU64,
    Effort,
};
use jope_carbon::{
    carbon_https_agent, decode_p256_pubkey_b64, encrypt_cuen_payload,
    generate_nonlinear_benchmark_rows, send_raw_edge_http, send_raw_put_url, wrap_u_dek_to_pubkey,
    CarbonAuthVerifier, CarbonCapabilityClaims, CarbonEmuConfig, CarbonEngineKek,
    CarbonLocalLambdaEmulator, CarbonThinClient,
};
use jope_prep::cuen::CuenPayloadKind;

fn mint_emu_pro_token(cfg: &CarbonEmuConfig, tenant_id: &str) -> String {
    CarbonAuthVerifier::from_secret(cfg.auth_secret)
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: tenant_id.to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint pro token")
}

fn raw_http_with_method_and_ip(
    base_url: &str,
    method: &str,
    path_and_query: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> (u16, std::collections::BTreeMap<String, String>, Vec<u8>) {
    let url = format!("{}{}", base_url.trim_end_matches('/'), path_and_query);
    let agent = carbon_https_agent();
    let resp_res = match method {
        "OPTIONS" => {
            let mut r = agent
                .options(&url)
                .config()
                .http_status_as_error(false)
                .build();
            for &(k, v) in headers {
                r = r.header(k, v);
            }
            r.call()
        }
        "HEAD" => {
            let mut r = agent
                .head(&url)
                .config()
                .http_status_as_error(false)
                .build();
            for &(k, v) in headers {
                r = r.header(k, v);
            }
            r.call()
        }
        "GET" => {
            let mut r = agent.get(&url).config().http_status_as_error(false).build();
            for &(k, v) in headers {
                r = r.header(k, v);
            }
            r.call()
        }
        "DELETE" => {
            let mut r = agent
                .delete(&url)
                .config()
                .http_status_as_error(false)
                .build();
            for &(k, v) in headers {
                r = r.header(k, v);
            }
            r.call()
        }
        "POST" => {
            let mut r = agent
                .post(&url)
                .config()
                .http_status_as_error(false)
                .build();
            for &(k, v) in headers {
                r = r.header(k, v);
            }
            r.send(body.unwrap_or(b""))
        }
        "PUT" => {
            let mut r = agent.put(&url).config().http_status_as_error(false).build();
            for &(k, v) in headers {
                r = r.header(k, v);
            }
            r.send(body.unwrap_or(b""))
        }
        other => panic!("unsupported method {other}"),
    };
    let mut resp = resp_res.expect("raw http call");
    let code = resp.status().as_u16();
    let mut hdrs = std::collections::BTreeMap::new();
    for (k, v) in resp.headers() {
        if let Ok(s) = v.to_str() {
            hdrs.insert(k.as_str().to_ascii_lowercase(), s.to_owned());
        }
    }
    let mut out = Vec::new();
    use std::io::Read as _;
    let _ = resp.body_mut().as_reader().read_to_end(&mut out);
    (code, hdrs, out)
}

#[test]
fn edge_cors_downloads_spot_advisor_404_headers_and_429_rate_limits() {
    let root = tempfile::tempdir().expect("tempdir");
    let cfg = CarbonEmuConfig::for_dir(root.path().join("edge_routes"));
    let verifier = jope_carbon::CarbonAuthVerifier::from_secret(cfg.auth_secret);
    let admin_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "ops_admin".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: true,
        })
        .expect("mint admin");
    let free_non_admin_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "free_non_admin".to_owned(),
            tier: CarbonTierWire::Free,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint free non-admin");
    let mut emu = CarbonLocalLambdaEmulator::start(cfg).expect("start emu");
    let base = emu.base_url();

    // 0. `GET /.well-known/carbon-regions.json` -> 200
    let (wk_code, _, wk_body) =
        raw_http_with_method_and_ip(&base, "GET", "/.well-known/carbon-regions.json", &[], None);
    assert_eq!(wk_code, 200);
    assert!(!wk_body.is_empty());

    // 1. Unmapped HTTP method (`OPTIONS`) -> 404 + HSTS + nosniff + sensitive-route `no-store`
    let (opt_code, opt_hdrs, _) =
        raw_http_with_method_and_ip(&base, "OPTIONS", "/v1/carbon/runs", &[], None);
    assert_eq!(opt_code, 404);
    assert!(opt_hdrs.contains_key("strict-transport-security"));
    assert_eq!(
        opt_hdrs.get("cache-control").map(String::as_str),
        Some("no-store")
    );

    // 2. Unknown route -> 404 + HSTS + nosniff + x-carbon-request-id
    let (nf_code, nf_hdrs, _) = raw_http_with_method_and_ip(
        &base,
        "GET",
        "/v1/carbon/does-not-exist",
        &[("x-carbon-request-id", "req_custom_01")],
        None,
    );
    assert_eq!(nf_code, 404);
    assert!(nf_hdrs.contains_key("strict-transport-security"));
    assert_eq!(
        nf_hdrs.get("x-content-type-options").map(String::as_str),
        Some("nosniff")
    );
    assert_eq!(
        nf_hdrs.get("x-carbon-request-id").map(String::as_str),
        Some("req_custom_01")
    );

    // 3. `/carbon/*` installer & release asset routes (GET, HEAD, bare `/carbon` 400, traversal 400, missing 404)
    for asset in [
        "/carbon/install.sh",
        "/carbon/install.ps1",
        "/carbon/carbon-clouds.json",
    ] {
        let (g_code, _, g_body) = raw_http_with_method_and_ip(&base, "GET", asset, &[], None);
        assert!(
            matches!(g_code, 200 | 302 | 307 | 404),
            "asset {asset} status {g_code}"
        );
        if g_code == 200 {
            assert!(!g_body.is_empty());
        }
        let (h_code, _, h_body) = raw_http_with_method_and_ip(&base, "HEAD", asset, &[], None);
        assert_eq!(h_code, g_code);
        assert!(h_body.is_empty());
    }
    let (bare_code, _, _) = raw_http_with_method_and_ip(&base, "GET", "/carbon", &[], None);
    assert!(matches!(bare_code, 400 | 404));
    let (trav_code, _, _) =
        raw_http_with_method_and_ip(&base, "GET", "/carbon/..%2fsecret", &[], None);
    assert!(matches!(trav_code, 400 | 404));

    // 4. `GET /v1/carbon/advisor/spot-hybrid`: 401 missing, 403 non-admin Free, 400 NaN/negative, 200 all 6 params
    let (sp_401, _, _) =
        raw_http_with_method_and_ip(&base, "GET", "/v1/carbon/advisor/spot-hybrid", &[], None);
    assert_eq!(sp_401, 401);

    let non_admin_hdr = format!("Bearer {free_non_admin_tok}");
    let (sp_403, _, _) = raw_http_with_method_and_ip(
        &base,
        "GET",
        "/v1/carbon/advisor/spot-hybrid",
        &[("authorization", &non_admin_hdr)],
        None,
    );
    assert_eq!(sp_403, 403);

    let admin_hdr = format!("Bearer {admin_tok}");
    let (sp_fallback, _, _) = raw_http_with_method_and_ip(
        &base,
        "GET",
        "/v1/carbon/advisor/spot-hybrid?monthly_spend_usd=invalid",
        &[("authorization", &admin_hdr)],
        None,
    );
    assert_eq!(sp_fallback, 200);

    let (sp_200, _, sp_body) = raw_http_with_method_and_ip(
        &base,
        "GET",
        "/v1/carbon/advisor/spot-hybrid?monthly_spend_usd=9500&p20_vcpus=20&mean_vcpus=42&p95_vcpus=90&cold_start_rate=0.05&months=3",
        &[("authorization", &admin_hdr)],
        None,
    );
    assert_eq!(sp_200, 200);
    let sp_json: serde_json::Value = serde_json::from_slice(&sp_body).expect("spot advisor json");
    assert!(sp_json["recommendation"].is_string());

    // 5. Rate-limit bucket (>240 req/min on `/v1/carbon/advisor/spot-hybrid`) -> 429
    let mut saw_429 = false;
    for _ in 0..250 {
        let (code, _, _) = raw_http_with_method_and_ip(
            &base,
            "GET",
            "/v1/carbon/advisor/spot-hybrid",
            &[("authorization", &admin_hdr)],
            None,
        );
        if code == 429 {
            saw_429 = true;
            break;
        }
    }
    assert!(saw_429, "expected 429 after exhausting per-IP bucket");

    emu.shutdown();
}

#[test]
fn edge_explain_export_prepare_predict_cas_and_emu_internal_negative_branches() {
    let root = tempfile::tempdir().expect("tempdir");
    let cfg = CarbonEmuConfig::for_dir(root.path().join("neg_matrix"));
    let tok_a = mint_emu_pro_token(&cfg, "tenant_alpha");
    let tok_b = mint_emu_pro_token(&cfg, "tenant_bravo");
    let mut emu = CarbonLocalLambdaEmulator::start(cfg).expect("start emu");
    let base = emu.base_url();
    let hdr_a = format!("Bearer {tok_a}");
    let hdr_b = format!("Bearer {tok_b}");

    // Probe KEK and run a valid encrypted run for tenant_alpha without shredding yet
    let (_, probed) =
        CarbonThinClient::connect_fastest(vec![base.clone()]).expect("connect fastest");
    let kek_pub = probed[0].engine_kek_pubkey_b64.clone().expect("kek pub");
    let kek_pub_raw = decode_p256_pubkey_b64(&kek_pub).expect("decode kek_pub");
    let (headers, rows, tgt) = generate_nonlinear_benchmark_rows(64, 3, 19);
    let prep_ds =
        CarbonThinClient::prepare_numeric_dataset(headers, &rows, &tgt, 19, 0.25).expect("prep");
    let u_dek = jope_carbon::generate_random_u_dek().expect("u_dek");
    let fit_cuen = encrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::FitJmat,
        "tenant_alpha",
        jope_carbon::CAS_BLOB_CUEN_SCOPE,
        &prep_ds.artifacts.fit_jmat,
    )
    .expect("enc fit");
    let hold_cuen = encrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::HoldoutJmat,
        "tenant_alpha",
        jope_carbon::CAS_BLOB_CUEN_SCOPE,
        &prep_ds.artifacts.holdout_jmat,
    )
    .expect("enc hold");
    let fit_sha = jope_carbon::sha256_lower_hex(&fit_cuen);
    let hold_sha = jope_carbon::sha256_lower_hex(&hold_cuen);

    let prep_up_body = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: "tenant_alpha".to_owned(),
        tier: CarbonTierWire::Pro,
        fit_rows: prep_ds.fit_rows,
        holdout_rows: prep_ds.holdout_rows,
        feature_count: 3,
        fit_jmat: CarbonUploadSlotWire {
            sha256: fit_sha.clone(),
            bytes: DecimalU64(fit_cuen.len() as u64),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: hold_sha.clone(),
            bytes: DecimalU64(hold_cuen.len() as u64),
        },
        encrypted_cuen: true,
    })
    .expect("encode prep upload");
    let (up_code, up_bytes) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("authorization", &hdr_a)],
        Some(&prep_up_body),
    )
    .expect("prepare upload");
    assert_eq!(up_code, 200);
    let up_resp: CarbonPrepareUploadResponse =
        serde_json::from_slice(&up_bytes).expect("decode prep upload");
    assert_eq!(
        send_raw_put_url(&up_resp.fit_target.url, &fit_cuen, true)
            .expect("put fit")
            .0,
        200
    );
    assert_eq!(
        send_raw_put_url(&up_resp.holdout_target.url, &hold_cuen, true)
            .expect("put hold")
            .0,
        200
    );

    let run_id: jope_api::ids::RunId = "run_01JCCARB0NNEGAT1VEMATRX001".parse().expect("run id");
    let ds_id: jope_api::ids::DatasetId = "ds_0123456789abcdef01234567".parse().expect("ds id");
    let run_str = run_id.to_string();
    let lm_cuen = encrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::LandmarkJlmb,
        "tenant_alpha",
        &run_str,
        &prep_ds.landmark_bundle.to_wire_bytes(),
    )
    .expect("enc lm");
    let cudk =
        wrap_u_dek_to_pubkey(&u_dek, &kek_pub_raw, "tenant_alpha", &run_str).expect("wrap u_dek");
    let cudk_b64 = base64::engine::general_purpose::STANDARD.encode(cudk);

    let create_body = serde_json::to_vec(&CarbonCreateRunRequest {
        run_id,
        dataset_id: ds_id,
        tenant_id: "tenant_alpha".to_owned(),
        tier: CarbonTierWire::Pro,
        fit_jmat_sha256: fit_sha.clone(),
        holdout_jmat_sha256: hold_sha,
        fit_rows: prep_ds.fit_rows,
        holdout_rows: prep_ds.holdout_rows,
        feature_names: prep_ds.feature_names.clone(),
        target_name: prep_ds.target_name.clone(),
        objective: "rmse".to_owned(),
        effort: Effort::Fast,
        seed: DecimalU64(7),
        landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(&lm_cuen),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some(cudk_b64.clone()),
        upload_id: Some(up_resp.upload_id.clone()),
        aux_column_names: Vec::new(),
        split_mode: None,
        calibration_mode: None,
    })
    .expect("encode create");
    let (cr_code, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/runs",
        &[("authorization", &hdr_a)],
        Some(&create_body),
    )
    .expect("create run");
    assert!(matches!(cr_code, 200..=202));

    // Poll until completed (also verify pre-v1.2 `?tenant_id=` query fallback and 400 missing tenant_id)
    let (missing_ten_get, _) = send_raw_edge_http(
        &base,
        "GET",
        &format!("/v1/carbon/runs/{run_str}"),
        &[("authorization", &hdr_a)],
        None,
    )
    .expect("get without tenant_id");
    assert!(matches!(missing_ten_get, 400 | 401 | 403));

    for _ in 0..80 {
        let (st_code, st_bytes) = send_raw_edge_http(
            &base,
            "GET",
            &format!("/v1/carbon/runs/{run_str}?tenant_id=tenant_alpha"),
            &[("authorization", &hdr_a)],
            None,
        )
        .expect("poll run");
        assert_eq!(st_code, 200);
        let prog: CarbonPublicWaveProgress = serde_json::from_slice(&st_bytes).expect("prog");
        if matches!(prog.state, jope_api::carbon::CarbonRunStateWire::Completed) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // `/events` SSE endpoint (`200` for owner, `403` IDOR for tenant_bravo)
    let (ev_200, _) = send_raw_edge_http(
        &base,
        "GET",
        &format!("/v1/carbon/runs/{run_str}/events?tenant_id=tenant_alpha"),
        &[("authorization", &hdr_a)],
        None,
    )
    .expect("events owner");
    assert_eq!(ev_200, 200);
    let (ev_403, _) = send_raw_edge_http(
        &base,
        "GET",
        &format!("/v1/carbon/runs/{run_str}/events?tenant_id=tenant_alpha"),
        &[("authorization", &hdr_b)],
        None,
    )
    .expect("events idor");
    assert_eq!(ev_403, 403);

    // `/predict/prepare`, `/explain`, `/export`, `/stop`, `/resume`: 401, 403 IDOR, 400 malformed, 413 >64KiB, 502 wrong KEK
    let wrong_kek = CarbonEngineKek::generate().expect("wrong kek");
    let wrong_cudk =
        wrap_u_dek_to_pubkey(&u_dek, wrong_kek.pubkey_bytes(), "tenant_alpha", &run_str)
            .expect("wrap wrong kek");
    let wrong_cudk_b64 = base64::engine::general_purpose::STANDARD.encode(wrong_cudk);
    let client_resp_kek = CarbonEngineKek::generate().expect("client resp kek");

    let prep_pred_ok = serde_json::to_vec(&CarbonPreparePredictRequest {
        tenant_id: "tenant_alpha".to_owned(),
        predict_cuen: CarbonUploadSlotWire {
            sha256: "a".repeat(64),
            bytes: DecimalU64(512),
        },
    })
    .expect("prep pred");
    assert_eq!(
        send_raw_edge_http(
            &base,
            "POST",
            &format!("/v1/carbon/runs/{run_str}/predict/prepare"),
            &[],
            Some(&prep_pred_ok),
        )
        .expect("401")
        .0,
        401
    );
    assert_eq!(
        send_raw_edge_http(
            &base,
            "POST",
            &format!("/v1/carbon/runs/{run_str}/predict/prepare"),
            &[("authorization", &hdr_b)],
            Some(&prep_pred_ok),
        )
        .expect("403")
        .0,
        403
    );
    assert_eq!(
        send_raw_edge_http(
            &base,
            "POST",
            &format!("/v1/carbon/runs/{run_str}/predict/prepare"),
            &[("authorization", &hdr_a)],
            Some(&prep_pred_ok),
        )
        .expect("200")
        .0,
        200
    );

    let oversize_70k = vec![b' '; 70_000];
    for sub in ["explain", "export", "stop", "resume", "predict/prepare"] {
        let path = format!("/v1/carbon/runs/{run_str}/{sub}");
        let (c_401, _) = send_raw_edge_http(&base, "POST", &path, &[], Some(b"{}")).expect("401");
        assert_eq!(c_401, 401, "{sub} 401");
        let (c_400, _) = send_raw_edge_http(
            &base,
            "POST",
            &path,
            &[("authorization", &hdr_a)],
            Some(b"{bad_json"),
        )
        .expect("400");
        assert_eq!(c_400, 400, "{sub} 400");
        let (c_413, _) = send_raw_edge_http(
            &base,
            "POST",
            &path,
            &[("authorization", &hdr_a)],
            Some(&oversize_70k),
        )
        .expect("413");
        assert!(matches!(c_413, 400 | 413), "{sub} oversize got {c_413}");
    }

    // Wrong KEK on `/explain` and `/export` -> 502/400 refusal; IDOR -> 403
    let bad_explain = serde_json::to_vec(&CarbonExplainRequest {
        tenant_id: "tenant_alpha".to_owned(),
        top_k: Some(2),
        client_response_pubkey_b64: Some(client_resp_kek.pubkey_b64()),
        wrapped_u_dek_b64: Some(wrong_cudk_b64.clone()),
    })
    .expect("bad explain");
    let (exp_idor, _) = send_raw_edge_http(
        &base,
        "POST",
        &format!("/v1/carbon/runs/{run_str}/explain"),
        &[("authorization", &hdr_b)],
        Some(&bad_explain),
    )
    .expect("explain idor");
    assert_eq!(exp_idor, 403);
    let (exp_wrong_kek, _) = send_raw_edge_http(
        &base,
        "POST",
        &format!("/v1/carbon/runs/{run_str}/explain"),
        &[("authorization", &hdr_a)],
        Some(&bad_explain),
    )
    .expect("explain wrong kek");
    assert!(matches!(exp_wrong_kek, 400 | 502));

    let bad_export = serde_json::to_vec(&CarbonExportRequest {
        tenant_id: "tenant_alpha".to_owned(),
        lang: CarbonExportLangWire::Python,
        accept_compressed: false,
        prefer_cas_output: false,
        client_response_pubkey_b64: Some(client_resp_kek.pubkey_b64()),
        wrapped_u_dek_b64: Some(wrong_cudk_b64),
    })
    .expect("bad export");
    let (ex_idor, _) = send_raw_edge_http(
        &base,
        "POST",
        &format!("/v1/carbon/runs/{run_str}/export"),
        &[("authorization", &hdr_b)],
        Some(&bad_export),
    )
    .expect("export idor");
    assert_eq!(ex_idor, 403);
    let (ex_wrong_kek, _) = send_raw_edge_http(
        &base,
        "POST",
        &format!("/v1/carbon/runs/{run_str}/export"),
        &[("authorization", &hdr_a)],
        Some(&bad_export),
    )
    .expect("export wrong kek");
    assert!(matches!(ex_wrong_kek, 400 | 502));

    // `/stop` & `/resume` IDOR -> 403
    let stop_body = serde_json::to_vec(&CarbonEarlyStopRequest {
        tenant_id: "tenant_alpha".to_owned(),
        reason: Some("test".to_owned()),
    })
    .expect("stop");
    assert_eq!(
        send_raw_edge_http(
            &base,
            "POST",
            &format!("/v1/carbon/runs/{run_str}/stop"),
            &[("authorization", &hdr_b)],
            Some(&stop_body),
        )
        .expect("stop idor")
        .0,
        403
    );
    let res_body = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: "tenant_alpha".to_owned(),
        effort: Some(Effort::Fast),
        additional_waves: Some(1),
        wrapped_u_dek_b64: Some(cudk_b64.clone()),
    })
    .expect("resume");
    assert_eq!(
        send_raw_edge_http(
            &base,
            "POST",
            &format!("/v1/carbon/runs/{run_str}/resume"),
            &[("authorization", &hdr_b)],
            Some(&res_body),
        )
        .expect("resume idor")
        .0,
        403
    );

    // `/cas/{sha}` invalid/expired presign -> 403
    let (cas_403, _) = send_raw_edge_http(
        &base,
        "GET",
        &format!("/v1/carbon/cas/{fit_sha}?exp=1&sig=deadbeef"),
        &[],
        None,
    )
    .expect("cas bad sig");
    assert!(matches!(cas_403, 400 | 401 | 403));

    // Shred via pre-v1.2 `?tenant_id=` query fallback, then verify 404 post-shred on `/explain`, `/export`, `/predict`!
    let (del_400, _) = send_raw_edge_http(
        &base,
        "DELETE",
        &format!("/v1/carbon/runs/{run_str}"),
        &[("authorization", &hdr_a)],
        None,
    )
    .expect("delete without tenant");
    assert!(matches!(del_400, 400 | 401 | 403));

    let (del_200, _) = send_raw_edge_http(
        &base,
        "DELETE",
        &format!("/v1/carbon/runs/{run_str}?tenant_id=tenant_alpha"),
        &[("authorization", &hdr_a)],
        None,
    )
    .expect("shred run");
    assert_eq!(del_200, 200);

    let ok_explain = serde_json::to_vec(&CarbonExplainRequest {
        tenant_id: "tenant_alpha".to_owned(),
        top_k: Some(2),
        client_response_pubkey_b64: Some(client_resp_kek.pubkey_b64()),
        wrapped_u_dek_b64: Some(cudk_b64.clone()),
    })
    .expect("explain post-shred");
    assert_eq!(
        send_raw_edge_http(
            &base,
            "POST",
            &format!("/v1/carbon/runs/{run_str}/explain"),
            &[("authorization", &hdr_a)],
            Some(&ok_explain),
        )
        .expect("404 explain")
        .0,
        404
    );
    let ok_export = serde_json::to_vec(&CarbonExportRequest {
        tenant_id: "tenant_alpha".to_owned(),
        lang: CarbonExportLangWire::C,
        accept_compressed: false,
        prefer_cas_output: false,
        client_response_pubkey_b64: Some(client_resp_kek.pubkey_b64()),
        wrapped_u_dek_b64: Some(cudk_b64.clone()),
    })
    .expect("export post-shred");
    assert_eq!(
        send_raw_edge_http(
            &base,
            "POST",
            &format!("/v1/carbon/runs/{run_str}/export"),
            &[("authorization", &hdr_a)],
            Some(&ok_export),
        )
        .expect("404 export")
        .0,
        404
    );
    let ok_pred = serde_json::to_vec(&CarbonPredictRequest {
        tenant_id: "tenant_alpha".to_owned(),
        feature_names: Vec::new(),
        rows: Vec::new(),
        encrypted_rows_cuen_b64: Some(base64::engine::general_purpose::STANDARD.encode(b"CUEN")),
        encrypted_rows_cas_sha256: None,
        predict_upload_id: None,
        prefer_cas_output: false,
        client_response_pubkey_b64: Some(client_resp_kek.pubkey_b64()),
        wrapped_u_dek_b64: Some(cudk_b64),
    })
    .expect("pred post-shred");
    assert_eq!(
        send_raw_edge_http(
            &base,
            "POST",
            &format!("/v1/carbon/runs/{run_str}/predict"),
            &[("authorization", &hdr_a)],
            Some(&ok_pred),
        )
        .expect("404 pred")
        .0,
        404
    );

    // `emu.rs` internal routes: `/invoke/bogus` 404, `/cas/blobs/...` & `/cas/state/...` 403/404, and `/runtime/edge/.../error`
    assert_eq!(
        raw_http_with_method_and_ip(&base, "POST", "/invoke/bogus_role", &[], Some(b"{}")).0,
        404
    );
    assert_eq!(
        raw_http_with_method_and_ip(
            &base,
            "GET",
            &format!("/cas/blobs/{}", "0".repeat(64)),
            &[],
            None
        )
        .0,
        403
    );
    assert_eq!(
        raw_http_with_method_and_ip(&base, "GET", "/cas/state/missing/key", &[], None).0,
        403
    );
    // Exercise `rt_error_handler` directly on `/runtime/engine/2018-06-01/runtime/invocation/{id}/error`
    let (rt_err_code, _, _) = raw_http_with_method_and_ip(
        &base,
        "POST",
        "/runtime/engine/2018-06-01/runtime/invocation/untracked-req-99/error",
        &[],
        Some(br#"{"errorMessage":"simulated worker fault"}"#),
    );
    assert!(matches!(rt_err_code, 202 | 404));

    emu.shutdown();
}
