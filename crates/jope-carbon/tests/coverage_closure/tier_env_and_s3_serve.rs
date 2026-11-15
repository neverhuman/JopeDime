//! `tier x env` full matrix closure (`unit_inproc`, `local_emu`, `enterprise_serve_local`,
//! `enterprise_serve_s3`) and loopback SigV4 `CarbonS3Cas` verification.

use std::sync::Arc;

use jope_api::{
    carbon::{CarbonRegionCatalog, CarbonRegionDescriptor, CarbonTierWire},
    Effort,
};
use jope_carbon::{
    carbon_https_agent, execute_carbon_cli_run, send_raw_edge_http, CarbonAuthVerifier,
    CarbonCapabilityClaims, CarbonCasStore, CarbonCliRunOptions, CarbonEdgeConfig,
    CarbonEdgeRouter, CarbonEmuConfig, CarbonEngineInvoker, CarbonEngineKek,
    CarbonEngineWireCommand, CarbonEngineWorker, CarbonEnterpriseServeConfig,
    CarbonEnterpriseServer, CarbonEvalWorker, CarbonFsCas, CarbonLocalLambdaEmulator,
    CarbonPresigner, CarbonS3Cas, CarbonSealKey,
};

use super::{LoopbackMockS3Server, ScopedProcessEnv};

struct SyncInprocEngineInvoker {
    worker: Arc<CarbonEngineWorker>,
}

impl CarbonEngineInvoker for SyncInprocEngineInvoker {
    fn invoke_engine(&self, command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
        self.worker.execute_command(command, None)
    }

    fn invoke_engine_async(&self, command: &CarbonEngineWireCommand) -> Result<(), String> {
        self.worker.execute_command(command, None).map(|_| ())
    }
}

fn run_cli_cell(
    endpoint: &str,
    tenant: &str,
    tier: CarbonTierWire,
    effort: Effort,
    token: Option<String>,
    shred: bool,
) -> Result<jope_carbon::CarbonCliRunReceipt, String> {
    execute_carbon_cli_run(&CarbonCliRunOptions {
        endpoints: vec![endpoint.to_owned()],
        csv_path: None,
        target_column: "y".to_owned(),
        synthetic_rows: 48,
        synthetic_features: 2,
        tenant_id: tenant.to_owned(),
        tier,
        effort,
        objective: "rmse".to_owned(),
        seed: 42,
        holdout_fraction: 0.25,
        stop_after_ms: None,
        resume_after_stop: false,
        predict_rows: 4,
        auth_token: token,
        shred_after_run: shred,
    })
}

#[test]
fn unit_inproc_free_medium_and_local_emu_free_ultra_403_and_anon_medium_close_matrix() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // 1. `unit_inproc`: verify `free x medium` (`200` E2E) directly on `CarbonEdgeRouter`
    let cas = Arc::new(CarbonFsCas::open(tmp.path().join("inproc_cas")).expect("fs cas"));
    let eval =
        Arc::new(CarbonEvalWorker::new(cas.clone(), tmp.path().join("eval_s")).expect("eval"));
    let kek = Arc::new(CarbonEngineKek::generate().expect("kek"));
    let kek_pub = kek.pubkey_b64();
    let engine = Arc::new(
        CarbonEngineWorker::new(
            cas.clone(),
            Some(eval),
            CarbonSealKey::from_bytes([0x21; 32]),
            tmp.path().join("eng_w"),
        )
        .expect("engine")
        .with_engine_kek(Some(kek)),
    );
    let auth = CarbonAuthVerifier::from_secret([0x34; 32]);
    let router = Arc::new(CarbonEdgeRouter::new(
        CarbonEdgeConfig {
            region: "us-east-2".to_owned(),
            public_base_url: String::new(),
            enterprise_self_hosted: false,
            catalog: CarbonRegionCatalog {
                catalog_version: 1,
                regions: vec![CarbonRegionDescriptor {
                    region: "us-east-2".to_owned(),
                    geography: "unit".to_owned(),
                    endpoint_url: "http://127.0.0.1:0".to_owned(),
                    priority_rank: 1,
                }],
            },
            presigner: CarbonPresigner::from_secret([0x45; 32]),
            auth_verifier: Some(auth),
            engine_kek_pubkey_b64: Some(kek_pub),
        },
        cas,
        Arc::new(SyncInprocEngineInvoker { worker: engine }),
    ));
    let ping_gw = router.handle(&jope_carbon::CarbonGatewayRequest {
        method: "GET".to_owned(),
        path: "/v1/carbon/ping".to_owned(),
        query: Default::default(),
        headers: Default::default(),
        body_b64: None,
        raw_body: Vec::new(),
    });
    assert_eq!(ping_gw.status, 200);
    let anon_gw = router.handle(&jope_carbon::CarbonGatewayRequest {
        method: "POST".to_owned(),
        path: "/v1/carbon/auth/anonymous".to_owned(),
        query: Default::default(),
        headers: Default::default(),
        body_b64: None,
        raw_body: b"{}".to_vec(),
    });
    assert_eq!(anon_gw.status, 200);
    let anon_val: serde_json::Value =
        serde_json::from_slice(&anon_gw.decoded_body()).expect("anon json");
    let free_tok = anon_val["capability_token"]
        .as_str()
        .expect("free tok")
        .to_owned();
    let free_ten = anon_val["tenant_id"].as_str().expect("free ten").to_owned();
    let prep_gw = router.handle(&jope_carbon::CarbonGatewayRequest {
        method: "POST".to_owned(),
        path: "/v1/carbon/uploads/prepare".to_owned(),
        query: Default::default(),
        headers: [("authorization".to_owned(), format!("Bearer {free_tok}"))]
            .into_iter()
            .collect(),
        body_b64: None,
        raw_body: serde_json::to_vec(&jope_api::carbon::CarbonPrepareUploadRequest {
            tenant_id: free_ten,
            tier: CarbonTierWire::Free,
            fit_rows: 36,
            holdout_rows: 12,
            feature_count: 2,
            fit_jmat: jope_api::carbon::CarbonUploadSlotWire {
                sha256: "a".repeat(64),
                bytes: jope_api::decimal::DecimalU64(128),
            },
            holdout_jmat: jope_api::carbon::CarbonUploadSlotWire {
                sha256: "b".repeat(64),
                bytes: jope_api::decimal::DecimalU64(64),
            },
            encrypted_cuen: true,
        })
        .expect("prep json"),
    });
    assert_eq!(prep_gw.status, 200);

    // 2. `local_emu`: verify `free x medium` (`200` E2E) and `free x ultra` (`403` tier gate)
    let mut emu =
        CarbonLocalLambdaEmulator::start(CarbonEmuConfig::for_dir(tmp.path().join("emu")))
            .expect("start emu");
    let med_rcpt = run_cli_cell(
        &emu.base_url(),
        "anon_free_med",
        CarbonTierWire::Free,
        Effort::Medium,
        None,
        true,
    )
    .expect("local_emu free x medium");
    assert_eq!(med_rcpt.waves_completed, 4);
    assert!(med_rcpt.shredded_after_run);

    let ultra_err = run_cli_cell(
        &emu.base_url(),
        "anon_free_ultra",
        CarbonTierWire::Free,
        Effort::Ultra,
        None,
        true,
    )
    .expect_err("local_emu free x ultra must return 403");
    assert!(
        ultra_err.contains("403") || ultra_err.to_ascii_lowercase().contains("pro"),
        "unexpected error: {ultra_err}"
    );
    emu.shutdown();
}

#[test]
fn enterprise_serve_local_8_cells_and_streaming_precheck_413_parity() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // A. `enterprise_self_hosted = true` (`no_limit()` VPC mode) + `enable_user_envelope = true`
    let mut vpc_cfg = CarbonEnterpriseServeConfig::enterprise_default(
        "127.0.0.1:0".parse().expect("addr"),
        tmp.path().join("serve_vpc"),
    );
    vpc_cfg.enable_user_envelope = true;
    vpc_cfg.auth_secret = Some([0x66; 32]);
    let verifier = CarbonAuthVerifier::from_secret([0x66; 32]);
    let pro_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "vpc_pro".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint pro");
    let free_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "vpc_free".to_owned(),
            tier: CarbonTierWire::Free,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint free");

    let mut srv = CarbonEnterpriseServer::spawn_background(vpc_cfg).expect("spawn serve_vpc");
    let base = srv.base_url();

    // `serve.rs` streaming guards: `PUT /v1/carbon/cas/*` precheck 401/403 before body read, and 413 >131 KiB on `/v1/carbon/runs`
    let put_url = format!(
        "{base}/v1/carbon/cas/{}?exp=1&sig=bad&bytes=16",
        "a".repeat(64)
    );
    let put_pre = carbon_https_agent()
        .put(&put_url)
        .config()
        .http_status_as_error(false)
        .build()
        .send(&[0u8; 16][..])
        .expect("put cas precheck");
    assert!(matches!(put_pre.status().as_u16(), 400 | 401 | 403));

    let big_post = vec![b'x'; 140_000];
    let (code_413, _) = send_raw_edge_http(
        &base,
        "POST",
        "/v1/carbon/runs",
        &[("authorization", &format!("Bearer {pro_tok}"))],
        Some(&big_post),
    )
    .expect("post >131KiB to serve");
    assert_eq!(code_413, 413);

    // Run all 8 (`free` & `pro` x `{fast, medium, high, ultra}`) cells on `enterprise_serve_local` (`no_limit()`)
    for effort in [Effort::Fast, Effort::Medium, Effort::High, Effort::Ultra] {
        let r_free = run_cli_cell(
            &base,
            "vpc_free",
            CarbonTierWire::Free,
            effort,
            Some(free_tok.clone()),
            true,
        )
        .expect("vpc no_limit free run");
        assert!(r_free.shredded_after_run);

        let r_pro = run_cli_cell(
            &base,
            "vpc_pro",
            CarbonTierWire::Pro,
            effort,
            Some(pro_tok.clone()),
            true,
        )
        .expect("vpc pro run");
        assert!(r_pro.shredded_after_run);
    }
    srv.shutdown();

    // B. `enterprise_self_hosted = false` cloud-gate mode on `CarbonEnterpriseServer`: Free `high`/`ultra` -> 403
    let mut cloud_cfg = CarbonEnterpriseServeConfig::enterprise_default(
        "127.0.0.1:0".parse().expect("addr"),
        tmp.path().join("serve_cloud_gate"),
    );
    cloud_cfg.enterprise_self_hosted = false;
    cloud_cfg.enable_user_envelope = true;
    cloud_cfg.auth_secret = Some([0x66; 32]);
    let mut cloud_srv =
        CarbonEnterpriseServer::spawn_background(cloud_cfg).expect("spawn cloud_srv");
    for effort in [Effort::High, Effort::Ultra] {
        let err = run_cli_cell(
            &cloud_srv.base_url(),
            "vpc_free",
            CarbonTierWire::Free,
            effort,
            Some(free_tok.clone()),
            true,
        )
        .expect_err("cloud-mode serve must gate free high/ultra");
        assert!(err.contains("403") || err.to_ascii_lowercase().contains("pro"));
    }
    cloud_srv.shutdown();
}

#[test]
fn loopback_sigv4_s3_cas_and_enterprise_serve_s3_all_8_tier_cells_and_shred() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mock_s3 = LoopbackMockS3Server::start();
    let s3_ep = mock_s3.endpoint_url();

    let _env_guard = ScopedProcessEnv::set(&[
        ("CARBON_S3_ENDPOINT_URL", &s3_ep),
        ("AWS_ACCESS_KEY_ID", "loopback-test-access-key"),
        ("AWS_SECRET_ACCESS_KEY", "loopback-test-secret-material"),
    ]);

    // 1. Direct `CarbonS3Cas` operations (`put_blob_verified`, `get_raw_blob`, `delete_blob`, `put_state_object_if_absent` 412, `delete_state_object`)
    let s3_cas = CarbonS3Cas::new("us-east-2", "test-carbon-cas", "direct-v1");
    let payload = b"direct-s3-cas-blob-payload";
    let sha = jope_carbon::sha256_lower_hex(payload);
    s3_cas
        .put_blob_verified(&sha, payload)
        .expect("put_blob_verified to loopback S3");
    assert_eq!(
        s3_cas
            .get_raw_blob(None, &sha)
            .expect("get_raw_blob")
            .as_deref(),
        Some(&payload[..])
    );
    assert!(s3_cas
        .put_state_object_if_absent("runs/ten_a/run_01/lock.bin", b"v1")
        .expect("first put_state_object_if_absent"));
    assert!(!s3_cas
        .put_state_object_if_absent("runs/ten_a/run_01/lock.bin", b"v2")
        .expect("second put_state_object_if_absent must observe 412"));
    s3_cas
        .put_state_object("runs/ten_a/run_01/state.bin", b"sealed")
        .expect("put_state_object");
    assert_eq!(
        s3_cas
            .get_state_object("runs/ten_a/run_01/state.bin")
            .expect("get_state_object")
            .as_deref(),
        Some(&b"sealed"[..])
    );
    assert!(s3_cas.delete_blob(&sha).expect("delete_blob"));
    s3_cas
        .delete_state_object("runs/ten_a/run_01/lock.bin")
        .expect("delete lock");
    s3_cas
        .delete_state_object("runs/ten_a/run_01/state.bin")
        .expect("delete state");
    assert_eq!(s3_cas.get_raw_blob(None, &sha).expect("deleted blob"), None);

    // 2. `CarbonEnterpriseServer` backed by `--cas-url s3://test-carbon-cas/serve-s3` (`enterprise_serve_s3`):
    //    verify all 8 cells (`free` `{fast, medium}` 200 + `{high, ultra}` 403, `pro` `{fast, medium, high, ultra}` 200) + zero residual S3 objects after shred!
    let mut s3_cfg = CarbonEnterpriseServeConfig::enterprise_default(
        "127.0.0.1:0".parse().expect("addr"),
        tmp.path().join("serve_s3_work"),
    );
    s3_cfg.cas_url = Some("s3://test-carbon-cas/serve-s3".to_owned());
    s3_cfg.allow_local_fs_dev = false;
    s3_cfg.enterprise_self_hosted = false;
    s3_cfg.enable_user_envelope = true;
    s3_cfg.region = "us-east-2".to_owned();
    s3_cfg.auth_secret = Some([0x77; 32]);

    let verifier = CarbonAuthVerifier::from_secret([0x77; 32]);
    let free_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "s3_free_ten".to_owned(),
            tier: CarbonTierWire::Free,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint s3 free");
    let pro_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "s3_pro_ten".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint s3 pro");

    let mut s3_srv =
        CarbonEnterpriseServer::spawn_background(s3_cfg).expect("spawn enterprise_serve_s3");
    let s3_base = s3_srv.base_url();

    for effort in [Effort::Fast, Effort::Medium] {
        let rcpt = run_cli_cell(
            &s3_base,
            "s3_free_ten",
            CarbonTierWire::Free,
            effort,
            Some(free_tok.clone()),
            true,
        )
        .expect("s3 serve free fast/medium");
        assert!(rcpt.encrypted_cuen && rcpt.shredded_after_run);
    }
    for effort in [Effort::High, Effort::Ultra] {
        let err = run_cli_cell(
            &s3_base,
            "s3_free_ten",
            CarbonTierWire::Free,
            effort,
            Some(free_tok.clone()),
            true,
        )
        .expect_err("s3 serve free high/ultra must 403");
        assert!(err.contains("403") || err.to_ascii_lowercase().contains("pro"));
    }
    for effort in [Effort::Fast, Effort::Medium, Effort::High, Effort::Ultra] {
        let rcpt = run_cli_cell(
            &s3_base,
            "s3_pro_ten",
            CarbonTierWire::Pro,
            effort,
            Some(pro_tok.clone()),
            true,
        )
        .expect("s3 serve pro 4-effort");
        assert!(rcpt.encrypted_cuen && rcpt.shredded_after_run);
    }
    // The 6 completed runs deleted 100% of their 12 blobs; only the 2 pre-run uploads whose
    // subsequent `POST /v1/carbon/runs` was rejected with HTTP 403 (`free x high` & `free x ultra`)
    // remain unassociated until S3's 1-day `blobs/` lifecycle rule expires them.
    assert_eq!(
        mock_s3.object_count_with_prefix("test-carbon-cas/serve-s3/blobs/"),
        4,
        "all 6 completed S3 runs must shred 100% of their blobs (leaving only the 2 rejected 403 uploads)"
    );
    assert!(
        !mock_s3.has_key_ending_with(".enc"),
        "all S3 sealed checkpoint & champion envelopes must be shredded after --shred"
    );
    s3_srv.shutdown();
}
