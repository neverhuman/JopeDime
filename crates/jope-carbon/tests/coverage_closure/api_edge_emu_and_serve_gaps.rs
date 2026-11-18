//! Section 14 coverage closure for `API-33` (`emu.rs:759-806`), `API-24` (`serve.rs:180-294`),
//! `TE-17..TE-24` (`serve.rs:420-506` foreground `SIGINT` clean shutdown), and `edge.rs` routes
//! `API-04` (`/auth/anonymous`), `API-05` (`/auth/register`), `API-13` (`/resume`),
//! `API-14` (`/predict/prepare`), and `API-15` (`/predict`).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::Command;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use jope_api::{
    carbon::{
        CarbonCreateRunRequest, CarbonExecutionModeWire, CarbonPredictRequest,
        CarbonPredictResponse, CarbonPreparePredictRequest, CarbonPublicWaveProgress,
        CarbonRegionCatalog, CarbonRegionDescriptor, CarbonResumeRunRequest, CarbonRunStateWire,
        CarbonTierWire, CarbonUploadSlotWire,
    },
    decimal::DecimalU64,
    ids::RunId,
    Effort,
};
use jope_carbon::{
    canonical_run_state_key, carbon_https_agent, CarbonAuthVerifier, CarbonCapabilityClaims,
    CarbonCasStore, CarbonEdgeConfig, CarbonEdgeRouter, CarbonEmuConfig, CarbonEngineInvoker,
    CarbonEngineKek, CarbonEngineWireCommand, CarbonEnterpriseServeConfig, CarbonEnterpriseServer,
    CarbonFsCas, CarbonGatewayRequest, CarbonLocalLambdaEmulator, CarbonPresigner,
};

use super::ScopedProcessEnv;

fn send_truncated_chunked_post(addr: SocketAddr, path: &str) -> String {
    let mut stream =
        TcpStream::connect_timeout(&addr, Duration::from_secs(3)).expect("tcp connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("set_read_timeout");
    stream
        .set_write_timeout(Some(Duration::from_secs(3)))
        .expect("set_write_timeout");
    // Send invalid chunk size ("ZZZ\r\n") under Transfer-Encoding: chunked so hyper/axum body
    // collection fails with a stream read error -> GatewayBodyReadError::BadRequest -> HTTP 400.
    let raw_req = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\nZZZ\r\nbadchunk\r\n0\r\n\r\n"
    );
    let _ = stream.write_all(raw_req.as_bytes());
    let _ = stream.flush();
    let mut resp = Vec::new();
    let _ = stream.read_to_end(&mut resp);
    String::from_utf8_lossy(&resp).into_owned()
}

fn post_oversized_body(base_url: &str, path: &str, body_len: usize) -> (u16, String) {
    let url = format!("{}{}", base_url.trim_end_matches('/'), path);
    let payload = vec![b'a'; body_len];
    let mut resp = carbon_https_agent()
        .post(&url)
        .config()
        .http_status_as_error(false)
        .build()
        .header("content-type", "application/json")
        .send(&payload)
        .expect("oversized post");
    let status = resp.status().as_u16();
    let mut text = String::new();
    let _ = resp.body_mut().as_reader().read_to_string(&mut text);
    (status, text)
}

#[test]
fn emu_public_url_413_400_and_enterprise_serve_sigint_foreground_clean_exit() {
    let root = tempfile::tempdir().expect("tempdir");

    // 1. API-33 (`emu.rs:759-806`): > 131 KiB body -> 413 payload_too_large, and malformed chunked -> 400.
    let emu_cfg = CarbonEmuConfig::for_dir(root.path().join("emu_413"));
    let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start emu");
    let (emu_413_code, emu_413_body) =
        post_oversized_body(&emu.base_url(), "/v1/carbon/runs", 140_000);
    assert_eq!(emu_413_code, 413, "emu 413 body: {emu_413_body}");
    assert!(emu_413_body.contains("payload_too_large"));

    let emu_addr: SocketAddr = emu
        .base_url()
        .trim_start_matches("http://")
        .parse()
        .expect("parse emu socket addr");
    let emu_chunk_resp = send_truncated_chunked_post(emu_addr, "/v1/carbon/runs");
    assert!(
        emu_chunk_resp.starts_with("HTTP/1.1 400") || emu_chunk_resp.contains("bad_request"),
        "expected HTTP 400 from malformed chunked stream on emu, got: {emu_chunk_resp}"
    );
    emu.shutdown();

    // 2. API-24 (`serve.rs:180-294`): CarbonEnterpriseServeConfig Debug redaction + build_router error
    // branches + > 131 KiB body (413) + malformed chunked (400) + bound_addr().
    let listen_zero: SocketAddr = "127.0.0.1:0".parse().expect("addr");
    let mut cfg_dbg =
        CarbonEnterpriseServeConfig::enterprise_default(listen_zero, root.path().join("srv_dbg"));
    cfg_dbg.auth_secret = Some([0x77; 32]);
    let dbg_str = format!("{cfg_dbg:?}");
    assert!(dbg_str.contains("[REDACTED]"));
    assert!(!dbg_str.contains("119")); // 0x77 decimal byte never printed

    let mut cfg_empty_cas =
        CarbonEnterpriseServeConfig::enterprise_default(listen_zero, root.path().join("srv_empty"));
    cfg_empty_cas.cas_url = Some("   ".to_owned());
    let empty_err = CarbonEnterpriseServer::spawn_background(cfg_empty_cas)
        .err()
        .expect("empty cas_url must fail");
    assert!(empty_err.contains("cas_url cannot be empty"), "{empty_err}");

    let mut cfg_no_local =
        CarbonEnterpriseServeConfig::enterprise_default(listen_zero, root.path().join("srv_noloc"));
    cfg_no_local.allow_local_fs_dev = false;
    cfg_no_local.cas_url = None;
    let noloc_err = CarbonEnterpriseServer::spawn_background(cfg_no_local)
        .err()
        .expect("allow_local_fs_dev=false without cas_url must fail");
    assert!(noloc_err.contains("requires --cas-url"), "{noloc_err}");

    let mut srv_cfg =
        CarbonEnterpriseServeConfig::enterprise_default(listen_zero, root.path().join("srv_live"));
    srv_cfg.enable_user_envelope = true;
    let mut srv =
        CarbonEnterpriseServer::spawn_background(srv_cfg).expect("spawn background serve");
    let srv_addr = srv.bound_addr();
    assert!(srv_addr.port() > 0);
    let (srv_413_code, srv_413_body) =
        post_oversized_body(&srv.base_url(), "/v1/carbon/runs", 140_000);
    assert_eq!(srv_413_code, 413, "srv 413 body: {srv_413_body}");
    assert!(srv_413_body.contains("payload_too_large"));

    let srv_chunk_resp = send_truncated_chunked_post(srv_addr, "/v1/carbon/runs");
    assert!(
        srv_chunk_resp.starts_with("HTTP/1.1 400") || srv_chunk_resp.contains("bad_request"),
        "expected HTTP 400 from malformed chunked stream on serve, got: {srv_chunk_resp}"
    );
    srv.shutdown();

    // 3. TE-17..TE-24 & CLI-38 (`serve.rs:467-506` `run_until_ctrl_c` + `bin/carbon.rs` `Command::Serve`):
    // Spawn `carbon serve` subprocess on a loopback port, wait for `/ping` 200 OK, send SIGINT, and
    // verify clean graceful exit (`ExitCode::SUCCESS`). Also test invalid secret & non-loopback refusals.
    let bin = env!("CARGO_BIN_EXE_carbon");

    let non_loopback_out = Command::new(bin)
        .args([
            "serve",
            "--listen",
            "0.0.0.0:0",
            "--data-dir",
            root.path().join("srv_nonloop").to_str().unwrap(),
        ])
        .env_remove("CARBON_CAS_URL")
        .output()
        .expect("non-loopback serve refusal");
    assert_eq!(non_loopback_out.status.code(), Some(2));

    let bad_auth_out = Command::new(bin)
        .args([
            "serve",
            "--listen",
            "127.0.0.1:0",
            "--data-dir",
            root.path().join("srv_badauth").to_str().unwrap(),
        ])
        .env("CARBON_AUTH_SECRET_HEX", "not_hex")
        .output()
        .expect("bad auth hex serve refusal");
    assert_eq!(bad_auth_out.status.code(), Some(2));

    #[cfg(unix)]
    {
        let probe_listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = probe_listener.local_addr().expect("local_addr").port();
        drop(probe_listener);

        let fg_dir = root.path().join("srv_fg_sigint");
        let child = Command::new(bin)
            .args([
                "serve",
                "--listen",
                &format!("127.0.0.1:{port}"),
                "--data-dir",
                fg_dir.to_str().unwrap(),
                "--allow-local-fs-dev",
                "--enable-user-envelope",
            ])
            .env("CARBON_AUTH_SECRET_HEX", "55".repeat(32))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn foreground carbon serve");

        let ping_url = format!("http://127.0.0.1:{port}/v1/carbon/ping");
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut ready = false;
        while Instant::now() < deadline {
            if let Ok(resp) = carbon_https_agent().get(&ping_url).call() {
                if resp.status().as_u16() == 200 {
                    ready = true;
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(40));
        }
        assert!(
            ready,
            "foreground carbon serve never became healthy on {ping_url}"
        );

        let kill_status = Command::new("kill")
            .args(["-s", "INT", &child.id().to_string()])
            .status()
            .expect("send SIGINT to carbon serve");
        assert!(kill_status.success());

        let wait_out = child
            .wait_with_output()
            .expect("wait for carbon serve SIGINT");
        assert!(
            wait_out.status.success(),
            "expected exit 0 on SIGINT, got {:?}: {}",
            wait_out.status.code(),
            String::from_utf8_lossy(&wait_out.stderr)
        );
    }
}

struct StubEdgeEngineInvoker {
    async_calls: AtomicUsize,
    sync_calls: AtomicUsize,
    fail_mode: bool,
    malformed_json: bool,
    return_cas_predict_sha: Option<String>,
}

impl CarbonEngineInvoker for StubEdgeEngineInvoker {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
        self.sync_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_mode {
            return Err("simulated_engine_failure".to_owned());
        }
        if self.malformed_json {
            return Ok(b"{not-valid-json".to_vec());
        }
        match command {
            CarbonEngineWireCommand::ResumeRun { run_id, .. } => {
                let prog = CarbonPublicWaveProgress {
                    run_id: run_id.parse().unwrap_or_else(|_| RunId::new()),
                    state: CarbonRunStateWire::Completed,
                    execution_mode: CarbonExecutionModeWire::SoloSingleLambda,
                    eval_memory_mb: 1769,
                    eval_shards: 0,
                    eval_invocations: DecimalU64(1),
                    eval_total_wall_ms: DecimalU64(12),
                    wave_index: 2,
                    waves_target: 2,
                    candidates_evaluated: DecimalU64(8),
                    best_cv_loss: Some(0.1),
                    best_cv_metric: Some(0.95),
                    outer_holdout_loss: Some(0.12),
                    outer_holdout_metric: Some(0.94),
                    neural_seed_applied: false,
                    neural_guidance_mode: None,
                    bundle_digest: None,
                    tier_applied: Some(CarbonTierWire::Pro),
                    encrypted_cuen: false,
                    can_resume: true,
                    sealed_checkpoint_sha256: Some("aa".repeat(32)),
                    sealed_champion_sha256: Some("bb".repeat(32)),
                    elapsed_ms: DecimalU64(20),
                    region: "us-east-2".to_owned(),
                    failure_message: None,
                    eval_fallback_shards: None,
                };
                serde_json::to_vec(&prog).map_err(|e| e.to_string())
            }
            CarbonEngineWireCommand::Predict { run_id, .. } => {
                let resp = CarbonPredictResponse {
                    run_id: run_id.parse().unwrap_or_else(|_| RunId::new()),
                    predictions: if self.return_cas_predict_sha.is_some() {
                        Vec::new()
                    } else {
                        vec![1.25, 2.5]
                    },
                    predictions_sha256: "cc".repeat(32),
                    row_count: 2,
                    encrypted_predictions_cuen_b64: None,
                    encrypted_predictions_cas_url: None,
                    encrypted_predictions_cas_sha256: self.return_cas_predict_sha.clone(),
                    response_cudk_b64: self
                        .return_cas_predict_sha
                        .as_ref()
                        .map(|_| "dGVzdA==".to_owned()),
                };
                serde_json::to_vec(&resp).map_err(|e| e.to_string())
            }
            _ => Ok(b"{}".to_vec()),
        }
    }

    fn invoke_engine_async(&self, _command: &CarbonEngineWireCommand) -> Result<(), String> {
        self.async_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_mode {
            return Err("simulated_async_engine_failure".to_owned());
        }
        Ok(())
    }
}

fn make_gw_req(
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    raw_body: Vec<u8>,
) -> CarbonGatewayRequest {
    let mut hdr_map = BTreeMap::new();
    for &(k, v) in headers {
        hdr_map.insert(k.to_owned(), v.to_owned());
    }
    CarbonGatewayRequest {
        method: method.to_owned(),
        path: path.to_owned(),
        query: BTreeMap::new(),
        headers: hdr_map,
        body_b64: None,
        raw_body,
    }
}

fn sample_stored_create_request(
    run_id: RunId,
    tenant_id: &str,
    tier: CarbonTierWire,
    encrypted_cuen: bool,
) -> CarbonCreateRunRequest {
    CarbonCreateRunRequest {
        run_id,
        dataset_id: "ds_0123456789abcdef01234567".parse().expect("dataset_id"),
        tenant_id: tenant_id.to_owned(),
        tier,
        fit_jmat_sha256: "11".repeat(32),
        holdout_jmat_sha256: "22".repeat(32),
        fit_rows: 32,
        holdout_rows: 8,
        feature_names: vec!["x1".to_owned(), "x2".to_owned()],
        target_name: "y".to_owned(),
        objective: "rmse".to_owned(),
        effort: Effort::Fast,
        seed: DecimalU64(42),
        landmark_bundle_b64: "dGVzdA==".to_owned(),
        encrypted_cuen,
        wrapped_u_dek_b64: if encrypted_cuen {
            Some("dGVzdA==".to_owned())
        } else {
            None
        },
        upload_id: Some("upl_01JTESTUPLOAD0000000000000".to_owned()),
        aux_column_names: Vec::new(),
        split_mode: None,
        calibration_mode: None,
    }
}

#[test]
fn edge_auth_resume_predict_prepare_predict_and_s3_presign_branches() {
    let root = tempfile::tempdir().expect("tempdir");
    let cas = Arc::new(CarbonFsCas::open(root.path().join("cas")).expect("open fs cas"));
    let kek = CarbonEngineKek::generate().expect("kek");
    let catalog = CarbonRegionCatalog {
        catalog_version: 1,
        regions: vec![CarbonRegionDescriptor {
            region: "us-east-2".to_owned(),
            geography: "us-east".to_owned(),
            endpoint_url: "https://edge.carbon.example".to_owned(),
            priority_rank: 1,
        }],
    };

    // A. Gateway WITHOUT `auth_verifier` -> 501 `auth_not_configured` on `/auth/anonymous` (API-04)
    // and `/auth/register` (API-05).
    let no_auth_invoker = Arc::new(StubEdgeEngineInvoker {
        async_calls: AtomicUsize::new(0),
        sync_calls: AtomicUsize::new(0),
        fail_mode: false,
        malformed_json: false,
        return_cas_predict_sha: None,
    });
    let no_auth_router = CarbonEdgeRouter::new(
        CarbonEdgeConfig {
            region: "us-east-2".to_owned(),
            public_base_url: "https://edge.carbon.example".to_owned(),
            enterprise_self_hosted: false,
            allow_public_free_mint: true,
            catalog: catalog.clone(),
            presigner: CarbonPresigner::from_secret([0x31; 32]),
            auth_verifier: None,
            engine_kek_pubkey_b64: Some(kek.pubkey_b64()),
        },
        Arc::clone(&cas) as Arc<dyn CarbonCasStore>,
        no_auth_invoker,
    )
    .with_release_provenance(
        Some("rel-2026-11-15".to_owned()),
        Some("sha256:r882test".to_owned()),
    );

    let anon_501 = no_auth_router.handle(&make_gw_req(
        "POST",
        "/v1/carbon/auth/anonymous",
        &[],
        Vec::new(),
    ));
    assert_eq!(anon_501.status, 501);
    assert_eq!(
        anon_501
            .headers
            .get("x-carbon-release-id")
            .map(String::as_str),
        Some("rel-2026-11-15")
    );
    let reg_501 = no_auth_router.handle(&make_gw_req(
        "POST",
        "/v1/carbon/auth/register",
        &[],
        Vec::new(),
    ));
    assert_eq!(reg_501.status, 501);

    // B. Gateway WITH `auth_verifier` and `s3_upload_target` enabled -> exercises API-04, API-05,
    // API-13, API-14, and API-15 branches including S3 presigning and rate limits.
    let auth_secret = [0x42; 32];
    let verifier = CarbonAuthVerifier::from_secret(auth_secret);
    let stub_invoker = Arc::new(StubEdgeEngineInvoker {
        async_calls: AtomicUsize::new(0),
        sync_calls: AtomicUsize::new(0),
        fail_mode: false,
        malformed_json: false,
        return_cas_predict_sha: Some("dd".repeat(32)),
    });
    let s3_router = CarbonEdgeRouter::new(
        CarbonEdgeConfig {
            region: "us-east-2".to_owned(),
            public_base_url: "https://edge.carbon.example".to_owned(),
            enterprise_self_hosted: false,
            allow_public_free_mint: true,
            catalog: catalog.clone(),
            presigner: CarbonPresigner::from_secret([0x32; 32]),
            auth_verifier: Some(verifier.clone()),
            engine_kek_pubkey_b64: Some(kek.pubkey_b64()),
        },
        Arc::clone(&cas) as Arc<dyn CarbonCasStore>,
        Arc::clone(&stub_invoker) as Arc<dyn CarbonEngineInvoker>,
    )
    .with_s3_upload_target(Some((
        "carbon-test-cas-bkt".to_owned(),
        "cell-a".to_owned(),
    )));

    // API-04: 300 anonymous calls succeed, 301st hits 429 `rate_limited` (with retry-after: 1).
    // Bound to 605 iterations so a UTC minute-boundary rollover mid-loop still reaches 301 in the new window.
    let mut anon_429_seen = false;
    for _ in 0..605 {
        let r = s3_router.handle(&make_gw_req(
            "POST",
            "/v1/carbon/auth/anonymous",
            &[("x-carbon-trusted-source-ip", "198.51.100.41")],
            Vec::new(),
        ));
        if r.status == 429 {
            anon_429_seen = true;
            assert_eq!(r.headers.get("retry-after").map(String::as_str), Some("1"));
            break;
        }
        assert_eq!(r.status, 200);
    }
    assert!(
        anon_429_seen,
        "expected 429 on /auth/anonymous after 300 calls"
    );

    // API-05: malformed base64 body (400), > 1 KiB payload (413), invalid JSON (400),
    // whitespace-only body (201), invalid label with newline (400), and 429 rate limit (> 120).
    let mut bad_b64_req = make_gw_req("POST", "/v1/carbon/auth/register", &[], Vec::new());
    bad_b64_req.body_b64 = Some("%%%not-base64%%%".to_owned());
    assert_eq!(s3_router.handle(&bad_b64_req).status, 400);

    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                "/v1/carbon/auth/register",
                &[],
                vec![b' '; 1_025]
            ))
            .status,
        413
    );
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                "/v1/carbon/auth/register",
                &[],
                b"{bad".to_vec()
            ))
            .status,
        400
    );
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                "/v1/carbon/auth/register",
                &[],
                b"   \n\t ".to_vec()
            ))
            .status,
        201
    );
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                "/v1/carbon/auth/register",
                &[],
                br#"{"label":"bad\ncontrol"}"#.to_vec()
            ))
            .status,
        400
    );
    let mut reg_429_seen = false;
    for _ in 0..250 {
        let r = s3_router.handle(&make_gw_req(
            "POST",
            "/v1/carbon/auth/register",
            &[("x-carbon-trusted-source-ip", "198.51.100.42")],
            Vec::new(),
        ));
        if r.status == 429 {
            reg_429_seen = true;
            break;
        }
    }
    assert!(
        reg_429_seen,
        "expected 429 on /auth/register after 120 calls"
    );

    // Mint Free & Pro capability tokens for tenant_edge_1.
    let tenant = "tenant_edge_1";
    let pro_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: tenant.to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("pro token");
    let auth_hdr = format!("Bearer {pro_tok}");

    // Seed stored runs in CAS:
    // 1) `run_corrupt`: invalid JSON in `request.json`
    // 2) `run_free_enc`: Free tier, `encrypted_cuen = true`
    // 3) `run_free_plain`: Free tier, `encrypted_cuen = false`
    let run_missing = RunId::new();
    let run_corrupt = RunId::new();
    let run_free_enc = RunId::new();
    let run_free_plain = RunId::new();

    cas.put_state_object(
        &canonical_run_state_key(tenant, &run_corrupt.to_string(), "request.json").unwrap(),
        b"{corrupt-json",
    )
    .unwrap();

    let create_free_enc =
        sample_stored_create_request(run_free_enc.clone(), tenant, CarbonTierWire::Free, true);
    cas.put_state_object(
        &canonical_run_state_key(tenant, &run_free_enc.to_string(), "request.json").unwrap(),
        &serde_json::to_vec(&create_free_enc).unwrap(),
    )
    .unwrap();
    let init_prog = CarbonPublicWaveProgress {
        run_id: run_free_enc.clone(),
        state: CarbonRunStateWire::Paused,
        execution_mode: CarbonExecutionModeWire::SoloSingleLambda,
        eval_memory_mb: 1769,
        eval_shards: 0,
        eval_invocations: DecimalU64(1),
        eval_total_wall_ms: DecimalU64(5),
        wave_index: 1,
        waves_target: 2,
        candidates_evaluated: DecimalU64(4),
        best_cv_loss: Some(0.2),
        best_cv_metric: Some(0.8),
        outer_holdout_loss: Some(0.25),
        outer_holdout_metric: Some(0.75),
        neural_seed_applied: false,
        neural_guidance_mode: None,
        bundle_digest: None,
        tier_applied: Some(CarbonTierWire::Free),
        encrypted_cuen: true,
        can_resume: true,
        sealed_checkpoint_sha256: Some("ee".repeat(32)),
        sealed_champion_sha256: Some("ff".repeat(32)),
        elapsed_ms: DecimalU64(10),
        region: "us-east-2".to_owned(),
        failure_message: None,
        eval_fallback_shards: Some(1),
    };
    cas.put_state_object(
        &canonical_run_state_key(tenant, &run_free_enc.to_string(), "status.json").unwrap(),
        &serde_json::to_vec(&init_prog).unwrap(),
    )
    .unwrap();

    let create_free_plain =
        sample_stored_create_request(run_free_plain.clone(), tenant, CarbonTierWire::Free, false);
    cas.put_state_object(
        &canonical_run_state_key(tenant, &run_free_plain.to_string(), "request.json").unwrap(),
        &serde_json::to_vec(&create_free_plain).unwrap(),
    )
    .unwrap();

    // API-13 (`/v1/carbon/runs/{id}/resume`):
    let resume_path_enc = format!("/v1/carbon/runs/{run_free_enc}/resume");
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &resume_path_enc,
                &[("authorization", &auth_hdr)],
                vec![b'x'; 70_000]
            ))
            .status,
        413
    );
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &resume_path_enc,
                &[("authorization", &auth_hdr)],
                b"{bad".to_vec()
            ))
            .status,
        400
    );
    let valid_resume_body = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: tenant.to_owned(),
        effort: None,
        additional_waves: Some(1),
        wrapped_u_dek_b64: Some("dGVzdA==".to_owned()),
    })
    .unwrap();
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &format!("/v1/carbon/runs/{run_missing}/resume"),
                &[("authorization", &auth_hdr)],
                valid_resume_body.clone()
            ))
            .status,
        404
    );
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &format!("/v1/carbon/runs/{run_corrupt}/resume"),
                &[("authorization", &auth_hdr)],
                valid_resume_body
            ))
            .status,
        500
    );
    // Free run escalated to Ultra -> 403 `tier_limit`
    let ultra_resume = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: tenant.to_owned(),
        effort: Some(Effort::Ultra),
        additional_waves: Some(1),
        wrapped_u_dek_b64: Some("dGVzdA==".to_owned()),
    })
    .unwrap();
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &resume_path_enc,
                &[("authorization", &auth_hdr)],
                ultra_resume
            ))
            .status,
        403
    );
    // Encrypted run missing `wrapped_u_dek_b64` -> 400 `missing_wrapped_u_dek`
    let missing_dek_resume = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: tenant.to_owned(),
        effort: None,
        additional_waves: Some(1),
        wrapped_u_dek_b64: None,
    })
    .unwrap();
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &resume_path_enc,
                &[("authorization", &auth_hdr)],
                missing_dek_resume
            ))
            .status,
        400
    );
    // Async resume (`additional_waves: Some(3)`) -> 202 Accepted + `invoke_engine_async` incremented
    let async_resume = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: tenant.to_owned(),
        effort: Some(Effort::Medium),
        additional_waves: Some(3),
        wrapped_u_dek_b64: Some("dGVzdA==".to_owned()),
    })
    .unwrap();
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &resume_path_enc,
                &[
                    ("authorization", &auth_hdr),
                    ("x-carbon-client-version", "carbon-cli-v1.2.0")
                ],
                async_resume
            ))
            .status,
        202
    );
    assert_eq!(stub_invoker.async_calls.load(Ordering::SeqCst), 1);

    // Sync resume (`additional_waves: Some(1)`, `Effort::Fast`) -> 200 OK
    let sync_resume = serde_json::to_vec(&CarbonResumeRunRequest {
        tenant_id: tenant.to_owned(),
        effort: Some(Effort::Fast),
        additional_waves: Some(1),
        wrapped_u_dek_b64: Some("dGVzdA==".to_owned()),
    })
    .unwrap();
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &resume_path_enc,
                &[("authorization", &auth_hdr)],
                sync_resume
            ))
            .status,
        200
    );

    // API-14 (`/v1/carbon/runs/{id}/predict/prepare`):
    let prep_path_enc = format!("/v1/carbon/runs/{run_free_enc}/predict/prepare");
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &prep_path_enc,
                &[("authorization", &auth_hdr)],
                vec![b'x'; 70_000]
            ))
            .status,
        413
    );
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &prep_path_enc,
                &[("authorization", &auth_hdr)],
                b"{bad".to_vec()
            ))
            .status,
        400
    );
    // Zero-byte slot -> 400 `invalid_upload_slot`
    let bad_slot_prep = serde_json::to_vec(&CarbonPreparePredictRequest {
        tenant_id: tenant.to_owned(),
        predict_cuen: CarbonUploadSlotWire {
            sha256: "ab".repeat(32),
            bytes: DecimalU64(0),
        },
    })
    .unwrap();
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &prep_path_enc,
                &[("authorization", &auth_hdr)],
                bad_slot_prep
            ))
            .status,
        400
    );
    let ok_slot_prep = serde_json::to_vec(&CarbonPreparePredictRequest {
        tenant_id: tenant.to_owned(),
        predict_cuen: CarbonUploadSlotWire {
            sha256: "ab".repeat(32),
            bytes: DecimalU64(4096),
        },
    })
    .unwrap();
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &format!("/v1/carbon/runs/{run_missing}/predict/prepare"),
                &[("authorization", &auth_hdr)],
                ok_slot_prep.clone()
            ))
            .status,
        404
    );
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &format!("/v1/carbon/runs/{run_corrupt}/predict/prepare"),
                &[("authorization", &auth_hdr)],
                ok_slot_prep.clone()
            ))
            .status,
        500
    );
    // Free tier > 64 MiB -> 413 `predict_payload_too_large`
    let huge_slot_prep = serde_json::to_vec(&CarbonPreparePredictRequest {
        tenant_id: tenant.to_owned(),
        predict_cuen: CarbonUploadSlotWire {
            sha256: "ab".repeat(32),
            bytes: DecimalU64(70 * 1_048_576),
        },
    })
    .unwrap();
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &prep_path_enc,
                &[("authorization", &auth_hdr)],
                huge_slot_prep
            ))
            .status,
        413
    );

    // S3 presign without AWS env creds -> 500 `presign_creds_missing`, then WITH scoped test creds -> 200 OK
    {
        let _no_aws = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", None),
            ("AWS_SECRET_ACCESS_KEY", None),
            ("AWS_SESSION_TOKEN", None),
            ("AWS_PROFILE", None),
            ("JOPE_AWS_PROFILE", None),
        ]);
        assert_eq!(
            s3_router
                .handle(&make_gw_req(
                    "POST",
                    &prep_path_enc,
                    &[("authorization", &auth_hdr)],
                    ok_slot_prep.clone()
                ))
                .status,
            500
        );
    }
    {
        let _with_aws = ScopedProcessEnv::set(&[
            ("AWS_ACCESS_KEY_ID", "TESTKEYID0001"),
            ("AWS_SECRET_ACCESS_KEY", "dummy-test-value-not-a-secret"),
            ("AWS_REGION", "us-east-2"),
        ]);
        let s3_prep_ok = s3_router.handle(&make_gw_req(
            "POST",
            &prep_path_enc,
            &[("authorization", &auth_hdr)],
            ok_slot_prep,
        ));
        assert_eq!(s3_prep_ok.status, 200);
        let body_str = String::from_utf8(s3_prep_ok.decoded_body()).unwrap();
        assert!(body_str.contains("X-Amz-Signature="), "{body_str}");
    }

    // API-15 (`/v1/carbon/runs/{id}/predict`):
    let pred_path_enc = format!("/v1/carbon/runs/{run_free_enc}/predict");
    let pred_path_plain = format!("/v1/carbon/runs/{run_free_plain}/predict");

    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &pred_path_enc,
                &[("authorization", &auth_hdr)],
                b"{bad".to_vec()
            ))
            .status,
        400
    );
    let base_enc_pred = CarbonPredictRequest {
        tenant_id: tenant.to_owned(),
        feature_names: vec!["x1".to_owned(), "x2".to_owned()],
        rows: Vec::new(),
        encrypted_rows_cuen_b64: Some("dGVzdA==".to_owned()),
        encrypted_rows_cas_sha256: None,
        predict_upload_id: None,
        prefer_cas_output: false,
        client_response_pubkey_b64: Some("dGVzdA==".to_owned()),
        wrapped_u_dek_b64: Some("dGVzdA==".to_owned()),
    };
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &format!("/v1/carbon/runs/{run_missing}/predict"),
                &[("authorization", &auth_hdr)],
                serde_json::to_vec(&base_enc_pred).unwrap()
            ))
            .status,
        404
    );
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &format!("/v1/carbon/runs/{run_corrupt}/predict"),
                &[("authorization", &auth_hdr)],
                serde_json::to_vec(&base_enc_pred).unwrap()
            ))
            .status,
        500
    );
    // Encrypted run sent plaintext rows -> 400 `encrypted_predict_required`
    let mut bad_enc_pred = base_enc_pred.clone();
    bad_enc_pred.rows = vec![vec![Some(1.0), Some(2.0)]];
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &pred_path_enc,
                &[("authorization", &auth_hdr)],
                serde_json::to_vec(&bad_enc_pred).unwrap()
            ))
            .status,
        400
    );
    // Invalid `encrypted_rows_cas_sha256` -> 400 `invalid_sha256`
    let mut bad_sha_pred = base_enc_pred.clone();
    bad_sha_pred.encrypted_rows_cuen_b64 = None;
    bad_sha_pred.encrypted_rows_cas_sha256 = Some("not-a-sha256".to_owned());
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &pred_path_enc,
                &[("authorization", &auth_hdr)],
                serde_json::to_vec(&bad_sha_pred).unwrap()
            ))
            .status,
        400
    );
    // Path-traversal `predict_upload_id` -> 400 `invalid_predict_upload_id`
    let mut bad_uid_pred = base_enc_pred.clone();
    bad_uid_pred.predict_upload_id = Some("../escape".to_owned());
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &pred_path_enc,
                &[("authorization", &auth_hdr)],
                serde_json::to_vec(&bad_uid_pred).unwrap()
            ))
            .status,
        400
    );

    // S3 GET output presigning on `/predict`: 500 without creds, 200 with creds
    {
        let _no_aws = ScopedProcessEnv::set_opt(&[
            ("AWS_ACCESS_KEY_ID", None),
            ("AWS_SECRET_ACCESS_KEY", None),
            ("AWS_SESSION_TOKEN", None),
            ("AWS_PROFILE", None),
            ("JOPE_AWS_PROFILE", None),
        ]);
        assert_eq!(
            s3_router
                .handle(&make_gw_req(
                    "POST",
                    &pred_path_enc,
                    &[("authorization", &auth_hdr)],
                    serde_json::to_vec(&base_enc_pred).unwrap()
                ))
                .status,
            500
        );
    }
    {
        let _with_aws = ScopedProcessEnv::set(&[
            ("AWS_ACCESS_KEY_ID", "TESTKEYID0001"),
            ("AWS_SECRET_ACCESS_KEY", "dummy-test-value-not-a-secret"),
            ("AWS_REGION", "us-east-2"),
        ]);
        let mut ok_cas_pred = base_enc_pred.clone();
        ok_cas_pred.predict_upload_id = Some("pupl_01JTESTPREDICT00000000000".to_owned());
        let pred_200 = s3_router.handle(&make_gw_req(
            "POST",
            &pred_path_enc,
            &[("authorization", &auth_hdr)],
            serde_json::to_vec(&ok_cas_pred).unwrap(),
        ));
        assert_eq!(pred_200.status, 200);
        let body_str = String::from_utf8(pred_200.decoded_body()).unwrap();
        assert!(body_str.contains("X-Amz-Signature="), "{body_str}");
    }

    // Plaintext Free run feature cap (> 64 features -> 413 `predict_feature_limit`) & quota cap (403 `predict_quota_exceeded`)
    let too_many_feats = CarbonPredictRequest {
        tenant_id: tenant.to_owned(),
        feature_names: (0..65).map(|i| format!("f{i}")).collect(),
        rows: vec![vec![Some(1.0); 65]],
        encrypted_rows_cuen_b64: None,
        encrypted_rows_cas_sha256: None,
        predict_upload_id: None,
        prefer_cas_output: false,
        client_response_pubkey_b64: None,
        wrapped_u_dek_b64: None,
    };
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &pred_path_plain,
                &[("authorization", &auth_hdr)],
                serde_json::to_vec(&too_many_feats).unwrap()
            ))
            .status,
        413
    );

    cas.put_state_object(
        &canonical_run_state_key(tenant, &run_free_plain.to_string(), "predict_count.json")
            .unwrap(),
        &serde_json::to_vec(&1_000u32).unwrap(),
    )
    .unwrap();
    let quota_pred = CarbonPredictRequest {
        tenant_id: tenant.to_owned(),
        feature_names: vec!["x1".to_owned(), "x2".to_owned()],
        rows: vec![vec![Some(1.0), Some(2.0)]],
        encrypted_rows_cuen_b64: None,
        encrypted_rows_cas_sha256: None,
        predict_upload_id: None,
        prefer_cas_output: false,
        client_response_pubkey_b64: None,
        wrapped_u_dek_b64: None,
    };
    assert_eq!(
        s3_router
            .handle(&make_gw_req(
                "POST",
                &pred_path_plain,
                &[("authorization", &auth_hdr)],
                serde_json::to_vec(&quota_pred).unwrap()
            ))
            .status,
        403
    );
}
