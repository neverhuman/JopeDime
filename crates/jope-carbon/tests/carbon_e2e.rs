//! End-to-end integration proof for zero-knowledge Carbon serverless cells (`ADR-0007`):
//! - `CarbonLocalLambdaEmulator` driving the OS-isolated `carbon-edge` child binary + `engine` +
//!   `HybridFanOut` `eval` workers with capability-token auth + `CUEN\x01`/`CUDK\x01` encryption;
//! - Multi-region `/v1/carbon/ping` RTT probing and dead-endpoint failover;
//! - Client-side `FIT`-only landmarking (`JLMB` v1) + `JMAT` v1 split + `CUEN\x01` zero-cleartext CAS;
//! - Async `202` cooperative early-stop ("see it now") -> `Paused` -> E2E encrypted `predict` ->
//!   `CCKP\x02` tamper-rejection -> cross-container `resume` -> `DELETE` cryptographic shred;
//! - `CarbonEnterpriseServer` (`carbon serve`) `no_limit()` admission vs cloud `Free` tier rejection.

use std::fs;
use std::path::PathBuf;

use jope_api::{
    carbon::{
        CarbonExecutionModeWire, CarbonPrepareUploadRequest, CarbonRunStateWire, CarbonTierWire,
        CarbonUploadSlotWire,
    },
    decimal::DecimalU64,
    Effort,
};
use jope_carbon::{
    execute_carbon_cli_run, CarbonCliRunOptions, CarbonEdgeConfig, CarbonEdgeRouter,
    CarbonEmuConfig, CarbonEnterpriseServeConfig, CarbonEnterpriseServer, CarbonFsCas,
    CarbonHttpEngineInvoker, CarbonLocalLambdaEmulator, CarbonPresigner, CarbonThinClient,
};

fn build_synthetic_rows(count: usize) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(count);
    for i in 0..count {
        let x1 = (i as f64) * 0.15 - 4.5;
        let x2 = ((i % 11) as f64) * 0.4 - 2.0;
        let y = 3.0 * x1 - 1.5 * x2 + 0.5;
        rows.push(vec![x1, x2, y]);
    }
    rows
}

fn assert_wire_json_ip_clean(raw_json: &str) {
    let lower = raw_json.to_ascii_lowercase();
    for forbidden in [
        "\"program\"",
        "\"expression\"",
        "\"genome\"",
        "\"ast\"",
        "\"formula\"",
        "\"operator_tree\"",
    ] {
        assert!(
            !lower.contains(forbidden),
            "forbidden structure field {forbidden} leaked in public wire payload: {raw_json}"
        );
    }
}

#[test]
fn emu_lambda_runtime_zero_knowledge_hybrid_fanout_stop_resume_predict_shred() {
    let root = tempfile::tempdir().expect("create tempdir");
    let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
    assert!(
        edge_bin.is_file(),
        "compiled carbon-edge binary must exist at {}",
        edge_bin.display()
    );

    let mut cfg_us = CarbonEmuConfig::for_dir(root.path().join("cell_us_east_2"));
    cfg_us.region = "us-east-2".to_owned();
    cfg_us.force_hybrid_fanout = true;
    cfg_us.eval_worker_threads = 2;
    cfg_us.edge_binary_path = Some(edge_bin);
    let mut emu_us = CarbonLocalLambdaEmulator::start(cfg_us).expect("start us-east-2 emu");

    let mut cfg_eu = CarbonEmuConfig::for_dir(root.path().join("cell_eu_west_1"));
    cfg_eu.region = "eu-west-1".to_owned();
    cfg_eu.force_hybrid_fanout = true;
    cfg_eu.eval_worker_threads = 1;
    let mut emu_eu = CarbonLocalLambdaEmulator::start(cfg_eu).expect("start eu-west-1 emu");

    let dead_endpoint = "http://127.0.0.1:9".to_owned();
    let (_initial_client, probed) = CarbonThinClient::connect_fastest(vec![
        dead_endpoint,
        emu_us.base_url(),
        emu_eu.base_url(),
    ])
    .expect("probe healthy regional cells and skip dead cell");
    assert_eq!(probed.len(), 2, "both live cells must pass /v1/carbon/ping");

    let pro_token = emu_us
        .mint_capability_token("tenant_acme", CarbonTierWire::Pro, false)
        .expect("mint pro token");
    let (raw_client, _) =
        CarbonThinClient::connect_fastest(vec![emu_us.base_url()]).expect("connect us-east-2");
    assert!(raw_client.has_zero_knowledge_kek());
    let client = raw_client.with_capability_token(pro_token);
    let catalog = client
        .fetch_region_catalog()
        .expect("fetch /.well-known/carbon-regions.json");
    assert_eq!(catalog.regions[0].region, "us-east-2");

    let rows = build_synthetic_rows(72);
    let prepared = CarbonThinClient::prepare_numeric_dataset(
        vec!["x1".to_owned(), "x2".to_owned(), "y".to_owned()],
        &rows,
        "y",
        20261110,
        0.20,
    )
    .expect("prepare local dataset");

    let mut poisoned_rows = rows.clone();
    for &holdout_idx in prepared.split_plan.holdout_rows() {
        poisoned_rows[holdout_idx][2] = -999_999.0;
    }
    let poisoned_prep = CarbonThinClient::prepare_numeric_dataset(
        vec!["x1".to_owned(), "x2".to_owned(), "y".to_owned()],
        &poisoned_rows,
        "y",
        20261110,
        0.20,
    )
    .expect("prepare poisoned-holdout dataset");
    assert_eq!(
        prepared.landmark_bundle.to_wire_bytes(),
        poisoned_prep.landmark_bundle.to_wire_bytes(),
        "JLMB v1 landmarks must be 100% invariant to holdout labels"
    );

    let upload_resp = client
        .upload_prepared("tenant_acme", CarbonTierWire::Pro, &prepared)
        .expect("encrypt, presign, and upload fit + holdout CUEN blobs");

    // Verify stored CAS blobs on disk are `CUEN\x01` ciphertext with ZERO cleartext `JMAT` headers!
    let fit_sha_hex = &upload_resp.fit_target.sha256;
    let stored_fit_blob = fs::read(
        emu_us
            .cas_root()
            .join("blobs")
            .join(&fit_sha_hex[..2])
            .join(fit_sha_hex),
    )
    .expect("read stored fit CUEN blob");
    assert_eq!(&stored_fit_blob[..5], b"CUEN\x01");
    assert!(!stored_fit_blob.windows(4).any(|w| w == b"JMAT"));

    // Verify presigned URL rejects corrupted bytes of equal length (SHA-256 mismatch at CAS boundary).
    let mut corrupted = stored_fit_blob.clone();
    let flip_idx = corrupted.len() - 1;
    corrupted[flip_idx] ^= 0xFF;
    let bad_put = jope_carbon::carbon_https_agent()
        .put(&upload_resp.fit_target.url)
        .config()
        .http_status_as_error(false)
        .build()
        .send(&corrupted)
        .expect("send corrupted PUT");
    assert_eq!(
        bad_put.status().as_u16(),
        400,
        "CAS upload must reject payload whose SHA-256 differs from presigned digest"
    );

    let run_id_str = "run_01JCCARB0NE2E0000000000001";
    client
        .signal_early_stop(
            "tenant_acme",
            run_id_str,
            Some("user_clicked_see_it_now".to_owned()),
        )
        .expect("signal cooperative early stop");

    let paused = client
        .start_run(
            run_id_str.parse().expect("valid RunId"),
            "ds_222222222222222222222222"
                .parse()
                .expect("valid DatasetId"),
            "tenant_acme",
            CarbonTierWire::Pro,
            Effort::Medium,
            "rmse",
            77,
            &prepared,
        )
        .expect("start_run with async 202 + early stop");
    assert_eq!(paused.state, CarbonRunStateWire::Paused);
    assert_eq!(paused.execution_mode, CarbonExecutionModeWire::HybridFanOut);
    assert!(paused.encrypted_cuen);
    assert_eq!(paused.wave_index, 1);
    assert!(paused.can_resume);
    assert!(paused.best_cv_loss.is_some());
    assert!(paused.outer_holdout_loss.is_some());
    assert_wire_json_ip_clean(&serde_json::to_string(&paused).unwrap());

    // Verify `request.json` on disk NEVER persists `wrapped_u_dek_b64` at rest!
    let stored_req_text = fs::read_to_string(
        emu_us
            .cas_root()
            .join("state/runs/tenant_acme")
            .join(run_id_str)
            .join("request.json"),
    )
    .expect("read stored request.json");
    assert!(
        !stored_req_text.contains("wrapped_u_dek_b64"),
        "wrapped_u_dek_b64 must be stripped before CAS persistence: {stored_req_text}"
    );

    let sse = client
        .get_run_events_sse("tenant_acme", run_id_str)
        .expect("fetch SSE stream");
    assert!(sse.starts_with("event: wave_progress\ndata: "));
    assert_wire_json_ip_clean(&sse);

    // "See it now": E2E zero-cleartext encrypted `/predict` while Paused!
    let immediate_pred = client
        .predict(
            "tenant_acme",
            run_id_str,
            vec!["x2".to_owned(), "x1".to_owned()],
            vec![vec![Some(0.5), Some(2.0)], vec![None, Some(-1.0)]],
        )
        .expect("encrypted predict while paused");
    assert_eq!(immediate_pred.row_count, 2);
    assert_eq!(immediate_pred.predictions.len(), 2);
    assert!(immediate_pred.predictions.iter().all(|v| v.is_finite()));
    assert!(immediate_pred.encrypted_predictions_cuen_b64.is_some());
    assert_wire_json_ip_clean(&serde_json::to_string(&immediate_pred).unwrap());

    // Inspect persisted `sealed_checkpoint.enc`: verify `CCKP\x02` domain-bound header & tamper fail-closed!
    let ckp_disk_path = emu_us
        .cas_root()
        .join("state/runs/tenant_acme")
        .join(run_id_str)
        .join("sealed_checkpoint.enc");
    let orig_ckp_bytes = fs::read(&ckp_disk_path).expect("read sealed_checkpoint.enc");
    assert_eq!(&orig_ckp_bytes[..5], b"CCKP\x02");
    let ckp_lossy = String::from_utf8_lossy(&orig_ckp_bytes);
    assert!(!ckp_lossy.contains("matrix_sha256") && !ckp_lossy.contains("CSNP"));

    let mut tampered_ckp = orig_ckp_bytes.clone();
    let last_idx = tampered_ckp.len() - 1;
    tampered_ckp[last_idx] ^= 0x01;
    fs::write(&ckp_disk_path, &tampered_ckp).expect("write tampered checkpoint");
    let tamper_err = client
        .resume_run("tenant_acme", run_id_str, None, Some(1))
        .expect_err("tampered sealed_checkpoint.enc must fail AES-256-GCM authentication");
    assert!(tamper_err.contains("502"));

    fs::write(&ckp_disk_path, &orig_ckp_bytes).expect("restore authentic checkpoint");
    let completed = client
        .resume_run("tenant_acme", run_id_str, None, Some(1))
        .expect("resume paused run across fresh engine directory");
    assert_eq!(completed.state, CarbonRunStateWire::Completed);
    assert!(!completed.can_resume);
    assert!(completed.candidates_evaluated.0 >= paused.candidates_evaluated.0);
    assert!(completed.outer_holdout_loss.is_some());
    assert_wire_json_ip_clean(&serde_json::to_string(&completed).unwrap());

    // Cryptographically shred the completed run and verify blobs + state are purged!
    let shred = client
        .shred_run("tenant_acme", run_id_str)
        .expect("DELETE cryptographic shred");
    assert!(shred.shredded);
    assert!(shred.objects_deleted >= 5);
    assert!(!ckp_disk_path.exists());
    let post_shred = client
        .get_run_status("tenant_acme", run_id_str)
        .expect_err("shredded run must return 404");
    assert!(post_shred.contains("404"));

    let tel = emu_us.telemetry();
    assert!(tel.edge_invocations >= 8);
    assert!(tel.engine_invocations >= 3);
    assert!(tel.eval_invocations >= 4);

    emu_us.shutdown();
    emu_eu.shutdown();
}

#[test]
fn enterprise_serve_vpc_no_limit_and_cloud_free_tier_rejection() {
    let root = tempfile::tempdir().expect("create tempdir");

    let ent_cfg = CarbonEnterpriseServeConfig::enterprise_default(
        "127.0.0.1:0".parse().unwrap(),
        root.path().join("enterprise_vpc"),
    );
    let mut ent_srv =
        CarbonEnterpriseServer::spawn_background(ent_cfg).expect("spawn enterprise serve");
    let (ent_client, ent_probed) =
        CarbonThinClient::connect_fastest(vec![ent_srv.base_url()]).expect("connect enterprise");
    assert!(ent_probed[0].enterprise_self_hosted);

    let big_prep_req = CarbonPrepareUploadRequest {
        tenant_id: "corp_vpc".to_owned(),
        tier: CarbonTierWire::Free,
        fit_rows: 4_160,
        holdout_rows: 1_040,
        feature_count: 120,
        fit_jmat: CarbonUploadSlotWire {
            sha256: "a".repeat(64),
            bytes: DecimalU64(1_048_576),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: "b".repeat(64),
            bytes: DecimalU64(262_144),
        },
        encrypted_cuen: false,
    };
    let big_resp = jope_carbon::carbon_https_agent()
        .post(&format!("{}/v1/carbon/uploads/prepare", ent_srv.base_url()))
        .config()
        .http_status_as_error(false)
        .build()
        .header("content-type", "application/json")
        .send(&serde_json::to_vec(&big_prep_req).unwrap())
        .expect("POST enterprise big upload prepare");
    assert_eq!(big_resp.status().as_u16(), 200);

    let rows = build_synthetic_rows(50);
    let prepared = CarbonThinClient::prepare_numeric_dataset(
        vec!["x1".to_owned(), "x2".to_owned(), "y".to_owned()],
        &rows,
        "y",
        99,
        0.20,
    )
    .expect("prepare dataset");
    ent_client
        .upload_prepared("corp_vpc", CarbonTierWire::Enterprise, &prepared)
        .expect("upload to enterprise server");
    let ent_run_id = "run_01JCCARB0NENT0000000000001";
    let ent_done = ent_client
        .start_run(
            ent_run_id.parse().unwrap(),
            "ds_333333333333333333333333".parse().unwrap(),
            "corp_vpc",
            CarbonTierWire::Enterprise,
            Effort::Fast,
            "rmse",
            99,
            &prepared,
        )
        .expect("enterprise fast run");
    assert_eq!(ent_done.state, CarbonRunStateWire::Completed);
    let ent_pred = ent_client
        .predict(
            "corp_vpc",
            ent_run_id,
            vec!["x1".to_owned(), "x2".to_owned()],
            vec![vec![Some(0.0), Some(0.0)]],
        )
        .expect("enterprise predict");
    assert_eq!(ent_pred.row_count, 1);

    // Cloud gateway (`enterprise_self_hosted: false`, zero-knowledge enabled) rejects >5,000 rows & High on Free!
    let mut cloud_cfg = CarbonEnterpriseServeConfig::enterprise_default(
        "127.0.0.1:0".parse().unwrap(),
        root.path().join("cloud_free_check"),
    );
    cloud_cfg.enterprise_self_hosted = false;
    cloud_cfg.enable_user_envelope = true;
    cloud_cfg.auth_secret = Some([0x22; 32]);
    let mut cloud_srv =
        CarbonEnterpriseServer::spawn_background(cloud_cfg).expect("spawn cloud-mode gateway");
    let (mut free_client, _) =
        CarbonThinClient::connect_fastest(vec![cloud_srv.base_url()]).expect("connect cloud");
    let anon = free_client
        .authenticate_anonymous()
        .expect("mint anonymous free token");
    let (h5, r5, t5) = jope_carbon::generate_nonlinear_benchmark_rows(5_200, 4, 11);
    let big_prep = CarbonThinClient::prepare_numeric_dataset(h5, &r5, &t5, 11, 0.20).unwrap();
    let row_cap_err = free_client
        .upload_prepared(&anon.tenant_id, CarbonTierWire::Free, &big_prep)
        .expect_err("Cloud Free tier must reject >5,000 rows");
    assert!(row_cap_err.contains("403"));

    free_client
        .upload_prepared(&anon.tenant_id, CarbonTierWire::Free, &prepared)
        .expect("upload small encrypted dataset on Free tier");
    let high_err = free_client
        .start_run(
            "run_01JCCARB0NFRE0000000000001".parse().unwrap(),
            "ds_444444444444444444444444".parse().unwrap(),
            &anon.tenant_id,
            CarbonTierWire::Free,
            Effort::High,
            "rmse",
            1,
            &prepared,
        )
        .expect_err("Free tier must reject High effort");
    assert!(high_err.contains("403"));

    // Size/SHA-pinned S3 PUT presigner determinism (`RT-04`).
    let creds = jope_carbon::CarbonAwsCredentials {
        access_key_id: "I".repeat(20),
        secret_access_key: zeroize::Zeroizing::new("S".repeat(40)),
        session_token: None,
    };
    let presigned_s3 = CarbonPresigner::presign_s3_put_v4_pinned_with_creds(
        &creds,
        "us-east-2",
        "jopedime-carbon-cas-us-east-2",
        "blobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "20261110T213000Z",
        900,
        &"a".repeat(64),
        4_096,
    )
    .expect("presign size-pinned S3 PUT v4");
    assert!(presigned_s3.contains("content-length%3Bhost%3Bx-amz-content-sha256"));
    assert!(presigned_s3.contains("X-Amz-Signature="));

    let cli_receipt = execute_carbon_cli_run(&CarbonCliRunOptions {
        endpoints: vec![cloud_srv.base_url()],
        csv_path: None,
        target_column: "y".to_owned(),
        synthetic_rows: 120,
        synthetic_features: 4,
        tenant_id: "cli_anon_test".to_owned(),
        tier: CarbonTierWire::Free,
        effort: Effort::Fast,
        objective: "rmse".to_owned(),
        seed: 77,
        holdout_fraction: 0.20,
        stop_after_ms: None,
        resume_after_stop: false,
        predict_rows: 3,
        auth_token: None,
        shred_after_run: true,
    })
    .expect("execute_carbon_cli_run anonymous Free on cloud gateway");
    assert_eq!(cli_receipt.final_state, CarbonRunStateWire::Completed);
    assert!(cli_receipt.encrypted_cuen);
    assert!(cli_receipt.shredded_after_run);
    assert_eq!(cli_receipt.sample_predictions.len(), 3);

    let dummy_cas =
        std::sync::Arc::new(CarbonFsCas::open(root.path().join("fn_url_cas")).expect("fn_url_cas"));
    let dummy_invoker = std::sync::Arc::new(CarbonHttpEngineInvoker::new("http://127.0.0.1:9"));
    let fn_router = CarbonEdgeRouter::new(
        CarbonEdgeConfig {
            region: "us-east-2".to_owned(),
            public_base_url: "https://demo.lambda-url.us-east-2.on.aws".to_owned(),
            enterprise_self_hosted: false,
            catalog: jope_api::carbon::CarbonRegionCatalog::default_17_region_catalog(
                "carbon.jopedime.cloud",
            ),
            presigner: CarbonPresigner::from_secret([0x42; 32]),
            auth_verifier: None,
            engine_kek_pubkey_b64: None,
        },
        dummy_cas,
        dummy_invoker,
    )
    .with_s3_upload_target(Some((
        "veox-jopedime-carbon-dev".to_owned(),
        "cell-v1".to_owned(),
    )));
    let fn_url_event = br#"{
        "version": "2.0",
        "rawPath": "/v1/carbon/ping",
        "rawQueryString": "",
        "headers": {"host": "demo.lambda-url.us-east-2.on.aws"},
        "requestContext": {
            "http": {
                "method": "GET",
                "path": "/v1/carbon/ping"
            }
        },
        "isBase64Encoded": false
    }"#;
    let out_bytes = fn_router
        .handle_lambda_invocation(fn_url_event)
        .expect("Lambda Function URL v2.0 ping");
    let out_val: serde_json::Value = serde_json::from_slice(&out_bytes).expect("json out");
    assert_eq!(out_val["statusCode"], 200);

    ent_srv.shutdown();
    cloud_srv.shutdown();
}
