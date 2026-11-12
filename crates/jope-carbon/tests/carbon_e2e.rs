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
    assert_eq!(cli_receipt.sample_targets.len(), 3);
    assert!(cli_receipt.holdout_r2.unwrap_or(0.0) > 0.85);
    assert!(cli_receipt.mean_baseline_rmse.unwrap_or(0.0) > 1.0);
    assert!(cli_receipt.linear_baseline_rmse.is_some());
    assert!(cli_receipt.rmse_reduction_vs_mean_pct.unwrap_or(0.0) > 50.0);
    assert!(cli_receipt.e2e_predict_holdout_rmse.is_some());
    if let (Some(srv_ho), Some(e2e_ho)) = (
        cli_receipt.outer_holdout_score,
        cli_receipt.e2e_predict_holdout_rmse,
    ) {
        assert!(
            (srv_ho - e2e_ho).abs() < 1e-4,
            "E2E encrypted /predict holdout RMSE ({e2e_ho}) must match server holdout RMSE ({srv_ho})"
        );
    }

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

#[test]
fn cli_binary_free_and_pro_flags_login_whoami_logout_and_auto_tenant_binding() {
    use std::process::Command;

    let root = tempfile::tempdir().expect("create tempdir");
    let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
    let carbon_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));

    let mut cfg = CarbonEmuConfig::for_dir(root.path().join("cell_cli_flags"));
    cfg.region = "us-east-2".to_owned();
    cfg.force_hybrid_fanout = true;
    cfg.eval_worker_threads = 2;
    cfg.edge_binary_path = Some(edge_bin);
    let mut emu = CarbonLocalLambdaEmulator::start(cfg).expect("start emu");

    let (_, probed) =
        CarbonThinClient::connect_fastest(vec![emu.base_url()]).expect("probe emu ping");
    let kek_b64 = probed[0]
        .engine_kek_pubkey_b64
        .clone()
        .expect("emu kek pubkey");

    let mut clouds = std::collections::BTreeMap::new();
    clouds.insert(
        "dev".to_owned(),
        jope_carbon::CarbonCloudCellEntry {
            environment: "dev".to_owned(),
            region: "us-east-2".to_owned(),
            public_endpoint: emu.base_url(),
            engine_kek_pubkey_b64: Some(kek_b64.clone()),
            release_id: Some("testrel01".to_owned()),
            regions: Vec::new(),
        },
    );
    clouds.insert(
        "prod".to_owned(),
        jope_carbon::CarbonCloudCellEntry {
            environment: "prod".to_owned(),
            region: "us-east-2".to_owned(),
            public_endpoint: emu.base_url(),
            engine_kek_pubkey_b64: Some(kek_b64),
            release_id: Some("testrel01".to_owned()),
            regions: Vec::new(),
        },
    );
    let clouds_doc = jope_carbon::CarbonPublicCloudsManifest {
        schema: "carbon.clouds/v1".to_owned(),
        generated_epoch_ms: 1_794_400_000_000,
        clouds,
    };
    let config_dir = root.path().join("user_config");
    fs::create_dir_all(&config_dir).unwrap();
    fs::write(
        config_dir.join("carbon-clouds.json"),
        serde_json::to_vec_pretty(&clouds_doc).unwrap(),
    )
    .unwrap();

    // 1. Zero-flag `carbon ping` defaults to `--cloud prod` via local `$CARBON_CONFIG_DIR/carbon-clouds.json`!
    let ping_default = Command::new(&carbon_bin)
        .arg("ping")
        .env("CARBON_CONFIG_DIR", &config_dir)
        .env_remove("CARBON_ENDPOINT")
        .env_remove("CARBON_AUTH_TOKEN")
        .env_remove("CARBON_AUTH_SECRET_HEX")
        .output()
        .expect("run zero-flag carbon ping");
    assert!(
        ping_default.status.success(),
        "ping stderr: {}",
        String::from_utf8_lossy(&ping_default.stderr)
    );
    let ping_json: serde_json::Value = serde_json::from_slice(&ping_default.stdout).unwrap();
    assert_eq!(ping_json["healthy"], true);
    assert_eq!(ping_json["cloud"], "prod");
    assert_eq!(ping_json["kek_pin_verified"], true);

    // 2. `carbon whoami` before login reports `anonymous_free`
    let whoami_anon = Command::new(&carbon_bin)
        .arg("whoami")
        .env("CARBON_CONFIG_DIR", &config_dir)
        .env_remove("CARBON_AUTH_TOKEN")
        .output()
        .expect("run whoami anon");
    assert!(whoami_anon.status.success());
    let whoami_anon_json: serde_json::Value = serde_json::from_slice(&whoami_anon.stdout).unwrap();
    assert_eq!(whoami_anon_json["auth_mode"], "anonymous_free");

    // 3. `carbon run --pro` WITHOUT a token fails fast with exit code 2!
    let pro_no_tok = Command::new(&carbon_bin)
        .args([
            "run",
            "--cloud",
            "dev",
            "--pro",
            "--synthetic-rows",
            "60",
            "--features",
            "3",
        ])
        .env("CARBON_CONFIG_DIR", &config_dir)
        .env_remove("CARBON_AUTH_TOKEN")
        .env_remove("CARBON_AUTH_SECRET_HEX")
        .output()
        .expect("run --pro without token");
    assert_eq!(pro_no_tok.status.code(), Some(2));
    let pro_no_tok_err = String::from_utf8_lossy(&pro_no_tok.stderr);
    assert!(
        pro_no_tok_err.contains("requires a capability token"),
        "expected actionable token message, got: {pro_no_tok_err}"
    );

    // 4. `carbon run --pro --token <ctk1>` WITHOUT `--tenant` auto-extracts `tenant_id` from claims!
    let custom_pro_token = emu
        .mint_capability_token("custom_pro_tenant_42", CarbonTierWire::Pro, false)
        .expect("mint pro token");
    let pro_inline = Command::new(&carbon_bin)
        .args([
            "run",
            "--cloud",
            "dev",
            "--pro",
            "--token",
            &custom_pro_token,
            "--synthetic-rows",
            "72",
            "--features",
            "3",
            "--effort",
            "fast",
            "--shred",
        ])
        .env("CARBON_CONFIG_DIR", &config_dir)
        .env_remove("CARBON_AUTH_TOKEN")
        .env_remove("CARBON_AUTH_SECRET_HEX")
        .output()
        .expect("run --pro with inline token and no --tenant");
    assert!(
        pro_inline.status.success(),
        "pro inline stderr: {}",
        String::from_utf8_lossy(&pro_inline.stderr)
    );
    let pro_inline_json: serde_json::Value = serde_json::from_slice(&pro_inline.stdout).unwrap();
    assert_eq!(pro_inline_json["tier_applied"], "pro");
    assert_eq!(pro_inline_json["tenant_id"], "custom_pro_tenant_42");
    assert_eq!(pro_inline_json["final_state"], "completed");
    assert_eq!(pro_inline_json["shredded_after_run"], true);

    // 5. `carbon login --cloud prod --token-file` -> `whoami` -> `carbon run --pro` (zero token flags!)
    //    -> `carbon run --free` (overrides stored login!) -> `carbon logout --all`
    let tok_file = root.path().join("pro.tok");
    fs::write(&tok_file, format!("{custom_pro_token}\n")).unwrap();
    let login_out = Command::new(&carbon_bin)
        .args(["login", "--cloud", "prod", "--token-file"])
        .arg(&tok_file)
        .env("CARBON_CONFIG_DIR", &config_dir)
        .output()
        .expect("run carbon login");
    assert!(
        login_out.status.success(),
        "login stderr: {}",
        String::from_utf8_lossy(&login_out.stderr)
    );
    let login_json: serde_json::Value = serde_json::from_slice(&login_out.stdout).unwrap();
    assert_eq!(login_json["logged_in"], true);
    assert_eq!(login_json["tenant_id"], "custom_pro_tenant_42");

    let whoami_pro = Command::new(&carbon_bin)
        .arg("whoami")
        .env("CARBON_CONFIG_DIR", &config_dir)
        .env_remove("CARBON_AUTH_TOKEN")
        .output()
        .expect("run whoami after login");
    assert!(whoami_pro.status.success());
    let whoami_pro_json: serde_json::Value = serde_json::from_slice(&whoami_pro.stdout).unwrap();
    assert_eq!(whoami_pro_json["auth_mode"], "capability_token");
    assert_eq!(whoami_pro_json["tenant_id"], "custom_pro_tenant_42");
    assert_eq!(whoami_pro_json["tier"], "pro");

    // Zero `--cloud` and zero `--token` flags: uses default `prod` + stored Pro login!
    let stored_pro_run = Command::new(&carbon_bin)
        .args([
            "run",
            "--pro",
            "--synthetic-rows",
            "64",
            "--features",
            "3",
            "--shred",
        ])
        .env("CARBON_CONFIG_DIR", &config_dir)
        .env_remove("CARBON_ENDPOINT")
        .env_remove("CARBON_AUTH_TOKEN")
        .env_remove("CARBON_AUTH_SECRET_HEX")
        .output()
        .expect("run --pro with stored login");
    assert!(
        stored_pro_run.status.success(),
        "stored pro stderr: {}",
        String::from_utf8_lossy(&stored_pro_run.stderr)
    );
    let stored_pro_json: serde_json::Value =
        serde_json::from_slice(&stored_pro_run.stdout).unwrap();
    assert_eq!(stored_pro_json["tier_applied"], "pro");
    assert_eq!(stored_pro_json["tenant_id"], "custom_pro_tenant_42");
    assert_eq!(stored_pro_json["final_state"], "completed");

    // Explicit `--free` overrides stored Pro login and runs anonymous Free!
    let free_override_run = Command::new(&carbon_bin)
        .args([
            "run",
            "--free",
            "--synthetic-rows",
            "64",
            "--features",
            "3",
            "--shred",
        ])
        .env("CARBON_CONFIG_DIR", &config_dir)
        .env_remove("CARBON_ENDPOINT")
        .env_remove("CARBON_AUTH_TOKEN")
        .env_remove("CARBON_AUTH_SECRET_HEX")
        .output()
        .expect("run --free overriding stored login");
    assert!(
        free_override_run.status.success(),
        "free override stderr: {}",
        String::from_utf8_lossy(&free_override_run.stderr)
    );
    let free_override_json: serde_json::Value =
        serde_json::from_slice(&free_override_run.stdout).unwrap();
    assert_eq!(free_override_json["tier_applied"], "free");
    assert!(
        free_override_json["tenant_id"]
            .as_str()
            .unwrap_or_default()
            .starts_with("anon_"),
        "expected anonymous tenant_id under --free, got {}",
        free_override_json["tenant_id"]
    );
    assert_eq!(free_override_json["final_state"], "completed");

    let logout_out = Command::new(&carbon_bin)
        .args(["logout", "--all"])
        .env("CARBON_CONFIG_DIR", &config_dir)
        .output()
        .expect("run carbon logout --all");
    assert!(logout_out.status.success());
    let logout_json: serde_json::Value = serde_json::from_slice(&logout_out.stdout).unwrap();
    assert_eq!(logout_json["entries_removed"], 1);

    emu.shutdown();
}

#[test]
fn clean_install_signature_free_register_all_four_efforts_and_live_cost_accounting() {
    use std::process::Command;

    let root = tempfile::tempdir().expect("create tempdir");
    let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
    let compiled_carbon = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));
    assert!(edge_bin.is_file() && compiled_carbon.is_file());

    let mut emu_cfg = CarbonEmuConfig::for_dir(root.path().join("cell_e2e_full"));
    emu_cfg.region = "us-east-2".to_owned();
    emu_cfg.force_hybrid_fanout = true;
    emu_cfg.eval_worker_threads = 6;
    emu_cfg.edge_binary_path = Some(edge_bin);
    let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start full E2E emulator");

    let (_, probed) = CarbonThinClient::connect_fastest(vec![emu.base_url()]).expect("probe emu");
    let kek_b64 = probed[0]
        .engine_kek_pubkey_b64
        .clone()
        .expect("emu KEK pubkey");

    // 1. Stage a signed offline release directory using an ephemeral ECDSA P-256 keypair and
    //    prove `deploy/carbon/install.sh` verifies ECDSA-P256 + SHA-256 and refuses 1-byte tampering.
    let key_dir = root.path().join("test_keys");
    fs::create_dir_all(&key_dir).unwrap();
    let priv_pem = key_dir.join("test_signing.key");
    let pub_pem = key_dir.join("test_signing.pub");
    assert!(Command::new("openssl")
        .args([
            "ecparam",
            "-name",
            "prime256v1",
            "-genkey",
            "-noout",
            "-out"
        ])
        .arg(&priv_pem)
        .status()
        .expect("openssl ecparam")
        .success());
    assert!(Command::new("openssl")
        .args(["ec", "-in"])
        .arg(&priv_pem)
        .args(["-pubout", "-out"])
        .arg(&pub_pem)
        .status()
        .expect("openssl ec pubout")
        .success());

    let stage_rel = root.path().join("staged_release");
    fs::create_dir_all(&stage_rel).unwrap();
    let arch_asset = if cfg!(target_arch = "aarch64") {
        "carbon-linux-aarch64-musl"
    } else {
        "carbon-linux-x86_64-musl"
    };
    fs::copy(&compiled_carbon, stage_rel.join(arch_asset)).unwrap();

    let mut clouds = std::collections::BTreeMap::new();
    for cloud_name in ["dev", "prod"] {
        clouds.insert(
            cloud_name.to_owned(),
            jope_carbon::CarbonCloudCellEntry {
                environment: cloud_name.to_owned(),
                region: "us-east-2".to_owned(),
                public_endpoint: emu.base_url(),
                engine_kek_pubkey_b64: Some(kek_b64.clone()),
                release_id: Some("e2erel043".to_owned()),
                regions: Vec::new(),
            },
        );
    }
    let clouds_bytes = serde_json::to_vec_pretty(&jope_carbon::CarbonPublicCloudsManifest {
        schema: "carbon.clouds/v1".to_owned(),
        generated_epoch_ms: 1_794_490_000_000,
        clouds,
    })
    .unwrap();
    fs::write(stage_rel.join("carbon-clouds.json"), &clouds_bytes).unwrap();

    let bin_sha = jope_carbon::sha256_lower_hex(&fs::read(stage_rel.join(arch_asset)).unwrap());
    let clouds_sha = jope_carbon::sha256_lower_hex(&clouds_bytes);
    let sums_path = stage_rel.join("SHA256SUMS.txt");
    fs::write(
        &sums_path,
        format!("{bin_sha}  {arch_asset}\n{clouds_sha}  carbon-clouds.json\n"),
    )
    .unwrap();

    let sig_der = Command::new("openssl")
        .args(["dgst", "-sha256", "-sign"])
        .arg(&priv_pem)
        .arg(&sums_path)
        .output()
        .expect("openssl dgst -sign");
    assert!(sig_der.status.success());
    use base64::Engine as _;
    let sig_b64 = base64::engine::general_purpose::STANDARD.encode(&sig_der.stdout);
    fs::write(stage_rel.join("SHA256SUMS.txt.sig"), format!("{sig_b64}\n")).unwrap();

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let installer = repo_root.join("deploy/carbon/install.sh");

    // Negative check: valid signature on SHA256SUMS.txt, but 1 byte flipped in binary -> refused!
    let tampered_rel = root.path().join("tampered_release");
    fs::create_dir_all(&tampered_rel).unwrap();
    for fname in [
        arch_asset,
        "carbon-clouds.json",
        "SHA256SUMS.txt",
        "SHA256SUMS.txt.sig",
    ] {
        fs::copy(stage_rel.join(fname), tampered_rel.join(fname)).unwrap();
    }
    let mut bad_bin = fs::read(tampered_rel.join(arch_asset)).unwrap();
    let last_i = bad_bin.len() - 1;
    bad_bin[last_i] ^= 0x5A;
    fs::write(tampered_rel.join(arch_asset), bad_bin).unwrap();

    let tampered_bin_dir = root.path().join("tampered_bin");
    let tampered_cfg_dir = root.path().join("tampered_cfg");
    let tamper_install = Command::new("sh")
        .arg(&installer)
        .arg("--source-dir")
        .arg(&tampered_rel)
        .arg("--bin-dir")
        .arg(&tampered_bin_dir)
        .arg("--config-dir")
        .arg(&tampered_cfg_dir)
        .arg("--allow-test-pubkey")
        .arg(&pub_pem)
        .arg("--skip-ping")
        .output()
        .expect("run install.sh on tampered binary");
    assert!(
        !tamper_install.status.success(),
        "install.sh must reject binary whose SHA-256 differs from signed SHA256SUMS.txt"
    );
    assert!(!tampered_bin_dir.join("carbon").exists());

    // Positive check: clean user HOME install via `deploy/carbon/install.sh`!
    let user_home = root.path().join("fresh_user_home");
    let installed_bin_dir = user_home.join(".local/bin");
    let installed_cfg_dir = user_home.join(".config/jopedime");
    fs::create_dir_all(&user_home).unwrap();
    let ok_install = Command::new("sh")
        .arg(&installer)
        .arg("--source-dir")
        .arg(&stage_rel)
        .arg("--allow-test-pubkey")
        .arg(&pub_pem)
        .arg("--skip-ping")
        .env("HOME", &user_home)
        .output()
        .expect("run positive install.sh");
    assert!(
        ok_install.status.success(),
        "install.sh stderr: {}",
        String::from_utf8_lossy(&ok_install.stderr)
    );
    let installed_carbon = installed_bin_dir.join("carbon");
    assert!(installed_carbon.is_file());
    assert!(installed_cfg_dir.join("carbon-clouds.json").is_file());

    // Prepare a real CSV dataset file (`48` rows x `2` features + target `y`).
    let csv_path = root.path().join("nonlinear_sensor.csv");
    let mut csv_text = String::from("f1,f2,y\n");
    for i in 0..48 {
        let f1 = (i as f64) * 0.16 - 3.8;
        let f2 = ((i % 9) as f64) * 0.35 - 1.4;
        let y = 2.4 * f1 - 1.3 * f2 + 0.6 * f1 * f2 + 1.25;
        csv_text.push_str(&format!("{f1:.5},{f2:.5},{y:.5}\n"));
    }
    fs::write(&csv_path, csv_text).unwrap();

    // 2. Brand-new user registers for Free (`carbon register --free --cloud prod --label e2e_free`)
    let reg_out = Command::new(&installed_carbon)
        .args([
            "register", "--free", "--cloud", "prod", "--label", "e2e_free",
        ])
        .env("HOME", &user_home)
        .env_remove("CARBON_CONFIG_DIR")
        .env_remove("CARBON_AUTH_TOKEN")
        .env_remove("CARBON_AUTH_SECRET_HEX")
        .output()
        .expect("run carbon register --free");
    assert!(
        reg_out.status.success(),
        "register stderr: {}",
        String::from_utf8_lossy(&reg_out.stderr)
    );
    let reg_json: serde_json::Value = serde_json::from_slice(&reg_out.stdout).unwrap();
    assert_eq!(reg_json["registered"], true);
    assert_eq!(reg_json["tier"], "free");
    assert_eq!(reg_json["credentials_saved"], true);
    let reg_tenant = reg_json["tenant_id"].as_str().unwrap().to_owned();
    assert!(
        reg_tenant.starts_with("free_e2e_free_"),
        "unexpected registered Free tenant_id: {reg_tenant}"
    );
    assert!(reg_json["ttl_remaining_secs"].as_u64().unwrap() >= 2_591_900);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let cred_meta = fs::metadata(installed_cfg_dir.join("carbon-credentials.json")).unwrap();
        assert_eq!(cred_meta.permissions().mode() & 0o777, 0o600);
    }

    // `carbon whoami` reflects the registered Free account
    let whoami_free = Command::new(&installed_carbon)
        .args(["whoami", "--cloud", "prod"])
        .env("HOME", &user_home)
        .env_remove("CARBON_CONFIG_DIR")
        .env_remove("CARBON_AUTH_TOKEN")
        .output()
        .expect("run whoami after free register");
    assert!(whoami_free.status.success());
    let wf_json: serde_json::Value = serde_json::from_slice(&whoami_free.stdout).unwrap();
    assert_eq!(wf_json["auth_mode"], "capability_token");
    assert_eq!(wf_json["tier"], "free");
    assert_eq!(wf_json["tenant_id"], reg_tenant);

    // Registered Free user runs CSV dataset at default `fast` AND `--free --effort medium`,
    // reusing their persistent `free_e2e_free_<ulid>` identity on both!
    for (extra_flag, effort, expected_waves) in [(false, "fast", 1u64), (true, "medium", 4u64)] {
        let mut cmd = Command::new(&installed_carbon);
        cmd.arg("run")
            .args(["--cloud", "prod", "--effort", effort, "--csv"])
            .arg(&csv_path)
            .args(["--target", "y", "--shred"]);
        if extra_flag {
            cmd.arg("--free");
        }
        let run_out = cmd
            .env("HOME", &user_home)
            .env_remove("CARBON_CONFIG_DIR")
            .env_remove("CARBON_AUTH_TOKEN")
            .env_remove("CARBON_AUTH_SECRET_HEX")
            .output()
            .expect("run registered free CSV");
        assert!(
            run_out.status.success(),
            "registered free {effort} stderr: {}",
            String::from_utf8_lossy(&run_out.stderr)
        );
        let rj: serde_json::Value = serde_json::from_slice(&run_out.stdout).unwrap();
        assert_eq!(rj["tenant_id"], reg_tenant);
        assert_eq!(rj["tier_applied"], "free");
        assert_eq!(rj["final_state"], "completed");
        assert_eq!(rj["waves_completed"], expected_waves);
        assert!(rj["cost_estimate"]["total_estimated_usd"].as_f64().unwrap() > 0.0);
        assert!(rj["cost_estimate"]["runs_per_1_usd"].as_f64().unwrap() > 0.0);
    }

    // Registered Free user attempting `--effort high` is refused at the edge (`403`)
    let free_high_refused = Command::new(&installed_carbon)
        .args(["run", "--cloud", "prod", "--effort", "high", "--csv"])
        .arg(&csv_path)
        .args(["--target", "y"])
        .env("HOME", &user_home)
        .env_remove("CARBON_CONFIG_DIR")
        .env_remove("CARBON_AUTH_TOKEN")
        .env_remove("CARBON_AUTH_SECRET_HEX")
        .output()
        .expect("run free --effort high");
    assert!(!free_high_refused.status.success());

    // 3. Log in with a Pro capability token on `--cloud dev` and validate ALL FOUR efforts
    //    (`fast`, `medium`, `high`, `ultra`) with `HybridFanOut` telemetry and per-dataset cost!
    let pro_tok = emu
        .mint_capability_token("pro_effort_matrix_team", CarbonTierWire::Pro, false)
        .expect("mint pro token");
    let pro_tok_file = root.path().join("pro_dev.tok");
    fs::write(&pro_tok_file, format!("{pro_tok}\n")).unwrap();
    assert!(Command::new(&installed_carbon)
        .args(["login", "--cloud", "dev", "--token-file"])
        .arg(&pro_tok_file)
        .env("HOME", &user_home)
        .env_remove("CARBON_CONFIG_DIR")
        .status()
        .expect("carbon login dev")
        .success());

    let mut observed_costs = Vec::new();
    for (effort, expected_waves) in [
        ("fast", 1u64),
        ("medium", 4u64),
        ("high", 12u64),
        ("ultra", 32u64),
    ] {
        let out = Command::new(&installed_carbon)
            .args([
                "run", "--cloud", "dev", "--pro", "--effort", effort, "--csv",
            ])
            .arg(&csv_path)
            .args(["--target", "y", "--shred"])
            .env("HOME", &user_home)
            .env_remove("CARBON_CONFIG_DIR")
            .env_remove("CARBON_AUTH_TOKEN")
            .env_remove("CARBON_AUTH_SECRET_HEX")
            .output()
            .unwrap_or_else(|e| panic!("run pro effort {effort}: {e}"));
        assert!(
            out.status.success(),
            "pro effort {effort} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let rj: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(rj["tenant_id"], "pro_effort_matrix_team");
        assert_eq!(rj["tier_applied"], "pro");
        assert_eq!(rj["final_state"], "completed");
        assert_eq!(rj["waves_completed"], expected_waves);
        assert_eq!(rj["waves_target"], expected_waves);
        assert!(rj["candidates_evaluated"].as_u64().unwrap() > 0);
        assert!(rj["outer_holdout_score"].as_f64().is_some());
        assert!(rj["holdout_r2"].as_f64().is_some());
        let eval_calls = rj["eval_invocations"].as_u64().unwrap();
        assert!(
            eval_calls > 0,
            "expected HybridFanOut eval_invocations > 0 on forced-hybrid emu for {effort}"
        );
        assert!(rj["cost_estimate"]["lambda_eval_usd"].as_f64().unwrap() > 0.0);
        let total_usd = rj["cost_estimate"]["total_estimated_usd"].as_f64().unwrap();
        assert!(total_usd > 0.0);
        observed_costs.push((effort, total_usd, eval_calls));
    }

    // Verify monotonic eval shard calls and positive increasing cost across `fast` -> `medium` -> `high` -> `ultra`
    for win in observed_costs.windows(2) {
        let (e_prev, c_prev, inv_prev) = win[0];
        let (e_next, c_next, inv_next) = win[1];
        assert!(
            inv_next > inv_prev,
            "expected eval_invocations({e_next})={inv_next} > eval_invocations({e_prev})={inv_prev}"
        );
        assert!(
            c_next > c_prev,
            "expected total_estimated_usd({e_next})={c_next} > total_estimated_usd({e_prev})={c_prev}"
        );
    }

    emu.shutdown();
}

#[test]
fn multi_region_parallel_probe_kek_pin_and_failover() {
    use std::collections::BTreeMap;
    use std::process::Command;

    let root = tempfile::tempdir().expect("create tempdir");
    let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
    let carbon_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));

    // Boot two regional cells (`us-east-2` and `us-west-2`) sharing global `auth_secret` while
    // holding strictly isolated per-region ECDH P-256 `engine_kek_scalar` keys!
    let mut east_cfg = CarbonEmuConfig::for_dir(root.path().join("cell_east"));
    east_cfg.region = "us-east-2".to_owned();
    east_cfg.auth_secret = [0x5Au8; 32];
    east_cfg.engine_kek_scalar = [0x11u8; 32];
    east_cfg.edge_binary_path = Some(edge_bin.clone());
    let mut emu_east = CarbonLocalLambdaEmulator::start(east_cfg).expect("start east cell");

    let mut west_cfg = CarbonEmuConfig::for_dir(root.path().join("cell_west"));
    west_cfg.region = "us-west-2".to_owned();
    west_cfg.auth_secret = [0x5Au8; 32];
    west_cfg.engine_kek_scalar = [0x22u8; 32];
    west_cfg.edge_binary_path = Some(edge_bin);
    let mut emu_west = CarbonLocalLambdaEmulator::start(west_cfg).expect("start west cell");

    let (_, p_east) = CarbonThinClient::connect_fastest(vec![emu_east.base_url()]).unwrap();
    let (_, p_west) = CarbonThinClient::connect_fastest(vec![emu_west.base_url()]).unwrap();
    let kek_east = p_east[0].engine_kek_pubkey_b64.clone().unwrap();
    let kek_west = p_west[0].engine_kek_pubkey_b64.clone().unwrap();
    assert_ne!(
        kek_east, kek_west,
        "regional cells must hold distinct ECDH P-256 KEK keypairs"
    );

    // 1. Negative check: wrong pinned KEKs must fail closed in `connect_fastest_pinned`!
    let bogus_kek = jope_carbon::CarbonEngineKek::generate()
        .unwrap()
        .pubkey_b64();
    let mut bad_pins = BTreeMap::new();
    bad_pins.insert(emu_east.base_url(), bogus_kek.clone());
    bad_pins.insert(emu_west.base_url(), bogus_kek);
    let pin_err = match CarbonThinClient::connect_fastest_pinned(
        vec![emu_east.base_url(), emu_west.base_url()],
        &bad_pins,
    ) {
        Ok(_) => panic!("expected KEK pin mismatch rejection"),
        Err(err) => err,
    };
    assert!(
        pin_err.contains("mismatch"),
        "expected KEK mismatch error, got: {pin_err}"
    );

    // 2. Write a `jope-carbon-clouds-v2` multi-region manifest and verify parallel RTT probing +
    //    `carbon register --free` + mid-flight failover when `us-east-2` goes down!
    let mut clouds = BTreeMap::new();
    clouds.insert(
        "prod".to_owned(),
        jope_carbon::CarbonCloudCellEntry {
            environment: "prod".to_owned(),
            region: "us-east-2".to_owned(),
            public_endpoint: emu_east.base_url(),
            engine_kek_pubkey_b64: Some(kek_east.clone()),
            release_id: Some("mr043".to_owned()),
            regions: vec![
                jope_carbon::CarbonCloudRegionPin {
                    region: "us-east-2".to_owned(),
                    public_endpoint: emu_east.base_url(),
                    engine_kek_pubkey_b64: Some(kek_east.clone()),
                    release_id: Some("mr043".to_owned()),
                },
                jope_carbon::CarbonCloudRegionPin {
                    region: "us-west-2".to_owned(),
                    public_endpoint: emu_west.base_url(),
                    engine_kek_pubkey_b64: Some(kek_west.clone()),
                    release_id: Some("mr043".to_owned()),
                },
            ],
        },
    );
    let cfg_dir = root.path().join("mr_cfg");
    fs::create_dir_all(&cfg_dir).unwrap();
    fs::write(
        cfg_dir.join("carbon-clouds.json"),
        serde_json::to_vec_pretty(&jope_carbon::CarbonPublicCloudsManifest {
            schema: "jope-carbon-clouds-v2".to_owned(),
            generated_epoch_ms: 1_794_500_000_000,
            clouds,
        })
        .unwrap(),
    )
    .unwrap();

    let ping_both = Command::new(&carbon_bin)
        .args(["ping", "--cloud", "prod"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("run multi-region ping");
    assert!(ping_both.status.success());
    let ping_j: serde_json::Value = serde_json::from_slice(&ping_both.stdout).unwrap();
    assert_eq!(ping_j["kek_pin_verified"], true);
    assert_eq!(ping_j["healthy_regions"].as_array().unwrap().len(), 2);

    // Register a persistent Free account while both regions are up
    let reg = Command::new(&carbon_bin)
        .args([
            "register", "--free", "--cloud", "prod", "--label", "mr_user",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("register multi-region free");
    assert!(reg.status.success());

    // Also connect an in-flight SDK client BEFORE shutting down `us-east-2`, forcing index 0 = east!
    let mut good_pins = BTreeMap::new();
    good_pins.insert(emu_east.base_url(), kek_east);
    good_pins.insert(emu_west.base_url(), kek_west);
    let (mut inflight_client, _) = CarbonThinClient::connect_fastest_pinned(
        vec![emu_east.base_url(), emu_west.base_url()],
        &good_pins,
    )
    .unwrap();

    // Simulate complete regional outage of `us-east-2`!
    emu_east.shutdown();

    // In-flight SDK client automatically fails over `authenticate_anonymous` + `upload_prepared` +
    // `start_run` (wrapping `U_DEK` to `us-west-2`'s distinct KEK!) + `predict` + `shred_run`!
    let anon = inflight_client
        .authenticate_anonymous()
        .expect("in-flight client should fail over to us-west-2");
    assert_eq!(
        inflight_client.active_endpoint(),
        emu_west.base_url().trim_end_matches('/')
    );
    let (_, rows, _) = jope_carbon::generate_nonlinear_benchmark_rows(40, 2, 7);
    let prep = CarbonThinClient::prepare_numeric_dataset(
        vec!["x0".to_owned(), "x1".to_owned(), "y".to_owned()],
        &rows,
        "y",
        7,
        0.20,
    )
    .unwrap();
    inflight_client
        .upload_prepared(&anon.tenant_id, CarbonTierWire::Free, &prep)
        .expect("upload_prepared on failover region");
    let prog = inflight_client
        .start_run(
            "run_01JCCARB0NE2E0000000000002".parse().unwrap(),
            "ds_222222222222222222222222".parse().unwrap(),
            &anon.tenant_id,
            CarbonTierWire::Free,
            Effort::Fast,
            "rmse",
            7,
            &prep,
        )
        .expect("start_run wrapped with us-west-2 KEK");
    assert_eq!(prog.region, "us-west-2");
    let pred = inflight_client
        .predict(
            &anon.tenant_id,
            "run_01JCCARB0NE2E0000000000002",
            vec!["x0".to_owned(), "x1".to_owned()],
            vec![vec![Some(0.5), Some(-0.2)]],
        )
        .expect("encrypted predict on us-west-2");
    assert_eq!(pred.predictions.len(), 1);
    let shred = inflight_client
        .shred_run(&anon.tenant_id, "run_01JCCARB0NE2E0000000000002")
        .unwrap();
    assert!(shred.shredded);

    // CLI `carbon run --cloud prod` seamlessly fails over to `us-west-2` and authenticates the
    // stored `ctk1.*` token minted earlier!
    let cli_failover = Command::new(&carbon_bin)
        .args([
            "run",
            "--cloud",
            "prod",
            "--synthetic-rows",
            "48",
            "--features",
            "2",
            "--effort",
            "fast",
            "--shred",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env_remove("CARBON_ENDPOINT")
        .env_remove("CARBON_AUTH_TOKEN")
        .env_remove("CARBON_AUTH_SECRET_HEX")
        .output()
        .expect("run CLI failover to us-west-2");
    assert!(
        cli_failover.status.success(),
        "CLI failover failed: {}",
        String::from_utf8_lossy(&cli_failover.stderr)
    );
    let rj: serde_json::Value = serde_json::from_slice(&cli_failover.stdout).unwrap();
    assert_eq!(rj["selected_region"], "us-west-2");
    assert_eq!(rj["encrypted_cuen"], true);
    assert_eq!(rj["final_state"], "completed");
    assert!(rj["tenant_id"]
        .as_str()
        .unwrap_or_default()
        .starts_with("free_mr_user_"));

    emu_west.shutdown();
}

#[test]
fn cli_doctor_bom_csv_target_ergonomics_and_progress_summary() {
    use std::process::Command;

    let root = tempfile::tempdir().expect("create tempdir");
    let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
    let carbon_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));

    let mut emu_cfg = CarbonEmuConfig::for_dir(root.path().join("cell_ux_v044"));
    emu_cfg.region = "us-east-2".to_owned();
    emu_cfg.eval_worker_threads = 2;
    emu_cfg.edge_binary_path = Some(edge_bin);
    let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start ux v044 emulator");

    let (_, probed) = CarbonThinClient::connect_fastest(vec![emu.base_url()]).expect("probe emu");
    let kek_b64 = probed[0]
        .engine_kek_pubkey_b64
        .clone()
        .expect("emu KEK pubkey");

    let cfg_dir = root.path().join("user_cfg");
    fs::create_dir_all(&cfg_dir).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&cfg_dir, fs::Permissions::from_mode(0o700)).unwrap();
    }

    let mut clouds = std::collections::BTreeMap::new();
    for cname in ["dev", "prod"] {
        clouds.insert(
            cname.to_owned(),
            jope_carbon::CarbonCloudCellEntry {
                environment: cname.to_owned(),
                region: "us-east-2".to_owned(),
                public_endpoint: emu.base_url(),
                engine_kek_pubkey_b64: Some(kek_b64.clone()),
                release_id: Some("ux044".to_owned()),
                regions: vec![jope_carbon::CarbonCloudRegionPin {
                    region: "us-east-2".to_owned(),
                    public_endpoint: emu.base_url(),
                    engine_kek_pubkey_b64: Some(kek_b64.clone()),
                    release_id: Some("ux044".to_owned()),
                }],
            },
        );
    }
    let manifest_bytes = serde_json::to_vec_pretty(&jope_carbon::CarbonPublicCloudsManifest {
        schema: "jope-carbon-clouds-v2".to_owned(),
        generated_epoch_ms: 1_794_500_000_000,
        clouds,
    })
    .unwrap();
    fs::write(cfg_dir.join("carbon-clouds.json"), &manifest_bytes).unwrap();

    // 1. Unauthenticated `carbon doctor --cloud all` passes cleanly.
    let doc1 = Command::new(&carbon_bin)
        .args(["doctor", "--cloud", "all"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env_remove("CARBON_AUTH_TOKEN")
        .output()
        .expect("run carbon doctor");
    assert!(
        doc1.status.success(),
        "doctor stderr: {}",
        String::from_utf8_lossy(&doc1.stderr)
    );
    let doc1_json: serde_json::Value = serde_json::from_slice(&doc1.stdout).unwrap();
    assert_eq!(doc1_json["healthy"], true);
    assert_eq!(doc1_json["clouds"]["prod"]["kek_pins_verified"], true);
    assert_eq!(doc1_json["clouds"]["dev"]["healthy_regions"], 1);

    // 2. Register Free -> `carbon doctor` reports stored credentials with 0600 mode, and fails closed if 0644!
    let reg = Command::new(&carbon_bin)
        .args(["register", "--free", "--cloud", "prod", "--label", "doc_u"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("register free");
    assert!(reg.status.success());

    let doc2 = Command::new(&carbon_bin)
        .args(["doctor", "--cloud", "prod"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("doctor after register");
    assert!(doc2.status.success());
    let doc2_json: serde_json::Value = serde_json::from_slice(&doc2.stdout).unwrap();
    assert_eq!(doc2_json["healthy"], true);
    assert_eq!(doc2_json["credentials_permissions_ok"], true);
    assert_eq!(doc2_json["stored_credentials"]["prod"]["tier"], "free");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let cred_file = cfg_dir.join("carbon-credentials.json");
        fs::set_permissions(&cred_file, fs::Permissions::from_mode(0o644)).unwrap();
        let doc_bad = Command::new(&carbon_bin)
            .args(["doctor", "--cloud", "prod", "--offline"])
            .env("CARBON_CONFIG_DIR", &cfg_dir)
            .output()
            .expect("doctor with 0644 credentials");
        assert!(
            !doc_bad.status.success(),
            "doctor must flag world-readable 0644 credentials"
        );
        fs::set_permissions(&cred_file, fs::Permissions::from_mode(0o600)).unwrap();
    }

    // 3. UTF-8 BOM CSV (`\u{feff}feat_a,feat_b,target`), auto-selected `target` column, `--progress`
    //    summary card on `stderr` with pure JSON on `stdout`, and helpful error on missing column!
    let bom_csv = root.path().join("bom_target.csv");
    let mut csv_text = String::from("\u{feff}feat_a,feat_b,target\r\n");
    for i in 0..48 {
        let a = (i as f64) * 0.15 - 3.0;
        let b = ((i % 9) as f64) * 0.4;
        let t = 1.8 * a - 0.9 * b + 0.3 * a * b + 0.5;
        csv_text.push_str(&format!("{a:.5},{b:.5},{t:.5}\r\n"));
    }
    fs::write(&bom_csv, csv_text.as_bytes()).unwrap();

    let bad_col = Command::new(&carbon_bin)
        .arg("run")
        .arg("--csv")
        .arg(&bom_csv)
        .args(["--target", "price_usd", "--quiet"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run with missing target column");
    assert!(!bad_col.status.success());
    let bad_err = String::from_utf8_lossy(&bad_col.stderr);
    assert!(
        bad_err.contains("[feat_a, feat_b, target]") && bad_err.contains("--target target"),
        "expected discovered CSV headers and --target suggestion, got: {bad_err}"
    );

    let bom_run = Command::new(&carbon_bin)
        .arg("run")
        .arg("--csv")
        .arg(&bom_csv)
        .args(["--progress", "--shred"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run BOM CSV with auto-selected target and --progress");
    assert!(
        bom_run.status.success(),
        "BOM CSV stderr: {}",
        String::from_utf8_lossy(&bom_run.stderr)
    );
    let bom_stderr = String::from_utf8_lossy(&bom_run.stderr);
    assert!(bom_stderr.contains("Auto-selected target column 'target'"));
    assert!(bom_stderr.contains("Run Summary"));
    let bom_json: serde_json::Value =
        serde_json::from_slice(&bom_run.stdout).expect("stdout must remain pure valid JSON");
    assert_eq!(bom_json["final_state"], "completed");
    assert_eq!(bom_json["shredded_after_run"], true);

    emu.shutdown();
}
