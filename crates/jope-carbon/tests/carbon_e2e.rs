//! End-to-end integration proof for Carbon serverless cells (`ADR-0007` Phases C4–C8):
//! - `CarbonLocalLambdaEmulator` driving the real `carbon-edge` child binary over `LAMBDA_RUNTIME_API`
//!   plus `engine` and `HybridFanOut` `eval` Lambda Custom Runtime (`2018-06-01`) queues and HTTP CAS;
//! - Multi-region `/v1/carbon/ping` RTT probing and dead-endpoint failover;
//! - Client-side `FIT`-only landmarking (`JLMB` v1) + `JMAT` v1 split + presigned `PUT` integrity;
//! - Cooperative early-stop ("see it now") -> `Paused` -> immediate `predict` -> tamper-rejection ->
//!   cross-container `resume` -> `Completed`;
//! - `AES-256-GCM` (`CCKP` v1) ciphertext verification on disk and zero-IP wire verification;
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
    CarbonEmuConfig, CarbonEnterpriseServeConfig, CarbonEnterpriseServer,
    CarbonLocalLambdaEmulator, CarbonPresigner, CarbonThinClient,
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
fn emu_lambda_runtime_hybrid_fanout_stop_resume_predict_and_ip_isolation() {
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

    // Include a dead port first to verify automatic regional failover during probe!
    let dead_endpoint = "http://127.0.0.1:9".to_owned();
    let (_initial_client, probed) = CarbonThinClient::connect_fastest(vec![
        dead_endpoint,
        emu_us.base_url(),
        emu_eu.base_url(),
    ])
    .expect("probe healthy regional cells and skip dead cell");
    assert_eq!(probed.len(), 2, "both live cells must pass /v1/carbon/ping");

    // Re-connect pinned to `us-east-2` (running the OS-isolated `carbon-edge` child binary!)
    let (client, _) =
        CarbonThinClient::connect_fastest(vec![emu_us.base_url()]).expect("connect us-east-2");
    let catalog = client
        .fetch_region_catalog()
        .expect("fetch /.well-known/carbon-regions.json");
    assert_eq!(catalog.regions[0].region, "us-east-2");

    // Client-side FIT-only landmark conditioning + JMAT v1 split.
    let rows = build_synthetic_rows(72);
    let prepared = CarbonThinClient::prepare_numeric_dataset(
        vec!["x1".to_owned(), "x2".to_owned(), "y".to_owned()],
        &rows,
        "y",
        20261110,
        0.20,
    )
    .expect("prepare local dataset");

    // Prove holdout label invariance on client preparation: poisoning every holdout target
    // produces the exact same 6,368-byte JLMB v1 landmark wire payload!
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
        .expect("presign and upload fit.jmat + holdout.jmat");

    // Verify presigned URL rejects corrupted bytes (SHA-256 mismatch at CAS boundary).
    let mut corrupted = prepared.artifacts.fit_jmat.clone();
    corrupted[0] ^= 0xFF;
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

    // Pre-stage cooperative early-stop ("see it now") so `Effort::Medium` (4 planned waves)
    // executes Wave 1 across `HybridFanOut` `eval` shards, checkpoints, seals state, scores
    // outer holdout, and pauses deterministically at `wave_index == 1`!
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
        .expect("start_run with early stop");
    assert_eq!(paused.state, CarbonRunStateWire::Paused);
    assert_eq!(paused.execution_mode, CarbonExecutionModeWire::HybridFanOut);
    assert_eq!(paused.wave_index, 1);
    assert!(paused.can_resume);
    assert!(paused.best_cv_loss.is_some());
    assert!(
        paused.outer_holdout_loss.is_some(),
        "outer holdout loss must be evaluated separately on holdout.jmat"
    );
    assert_wire_json_ip_clean(&serde_json::to_string(&paused).unwrap());

    let sse = client
        .get_run_events_sse("tenant_acme", run_id_str)
        .expect("fetch SSE stream");
    assert!(sse.starts_with("event: wave_progress\ndata: "));
    assert_wire_json_ip_clean(&sse);

    // "See it now": serve predictions immediately from the paused Wave-1 sealed champion!
    let immediate_pred = client
        .predict(
            "tenant_acme",
            run_id_str,
            vec!["x2".to_owned(), "x1".to_owned()],
            vec![vec![Some(0.5), Some(2.0)], vec![None, Some(-1.0)]],
        )
        .expect("predict while paused");
    assert_eq!(immediate_pred.row_count, 2);
    assert!(immediate_pred.predictions.iter().all(|v| v.is_finite()));
    assert_wire_json_ip_clean(&serde_json::to_string(&immediate_pred).unwrap());

    // Inspect persisted `sealed_checkpoint.enc` and `sealed_champion.enc` in emulated S3 CAS:
    // verify `CCKP\x01` AES-256-GCM header, zero cleartext column names, and tamper fail-closed!
    let ckp_disk_path = emu_us
        .cas_root()
        .join("state/runs/tenant_acme")
        .join(run_id_str)
        .join("sealed_checkpoint.enc");
    let orig_ckp_bytes = fs::read(&ckp_disk_path).expect("read sealed_checkpoint.enc");
    assert_eq!(&orig_ckp_bytes[..8], b"CCKP\x01\x00\x00\x00");
    let ckp_lossy = String::from_utf8_lossy(&orig_ckp_bytes);
    assert!(
        !ckp_lossy.contains("matrix_sha256") && !ckp_lossy.contains("CSNP"),
        "sealed_checkpoint.enc must be encrypted AES-256-GCM ciphertext, never cleartext"
    );

    // Tamper 1 ciphertext byte -> `resume_run` MUST refuse with HTTP 502 authentication failure!
    let mut tampered_ckp = orig_ckp_bytes.clone();
    let last_idx = tampered_ckp.len() - 1;
    tampered_ckp[last_idx] ^= 0x01;
    fs::write(&ckp_disk_path, &tampered_ckp).expect("write tampered checkpoint");
    let tamper_err = client
        .resume_run("tenant_acme", run_id_str, None, Some(1))
        .expect_err("tampered sealed_checkpoint.enc must fail AES-256-GCM authentication");
    assert!(tamper_err.contains("502"));

    // Restore authentic ciphertext and resume for 1 additional wave -> `Completed`!
    fs::write(&ckp_disk_path, &orig_ckp_bytes).expect("restore authentic checkpoint");
    let completed = client
        .resume_run("tenant_acme", run_id_str, None, Some(1))
        .expect("resume paused run across fresh engine directory");
    assert_eq!(completed.state, CarbonRunStateWire::Completed);
    assert!(!completed.can_resume);
    assert!(completed.candidates_evaluated.0 >= paused.candidates_evaluated.0);
    assert!(completed.outer_holdout_loss.is_some());
    assert_wire_json_ip_clean(&serde_json::to_string(&completed).unwrap());

    let tel = emu_us.telemetry();
    assert!(
        tel.edge_invocations >= 6,
        "expected >= 6 edge runtime invocations, got {}",
        tel.edge_invocations
    );
    assert!(
        tel.engine_invocations >= 3,
        "expected >= 3 engine runtime invocations (start + predict + resume), got {}",
        tel.engine_invocations
    );
    assert!(
        tel.eval_invocations >= 4,
        "expected >= 4 stateless eval shard invocations in HybridFanOut, got {}",
        tel.eval_invocations
    );

    emu_us.shutdown();
    emu_eu.shutdown();
}

#[test]
fn enterprise_serve_vpc_no_limit_and_cloud_free_tier_rejection() {
    let root = tempfile::tempdir().expect("create tempdir");

    // 1. Enterprise VPC/on-prem server (`enterprise_self_hosted: true` -> `no_limit()`).
    let ent_cfg = CarbonEnterpriseServeConfig::enterprise_default(
        "127.0.0.1:0".parse().unwrap(),
        root.path().join("enterprise_vpc"),
    );
    let mut ent_srv =
        CarbonEnterpriseServer::spawn_background(ent_cfg).expect("spawn enterprise serve");
    let (ent_client, ent_probed) =
        CarbonThinClient::connect_fastest(vec![ent_srv.base_url()]).expect("connect enterprise");
    assert!(ent_probed[0].enterprise_self_hosted);

    // Admit > 5,000 rows (`5,200` total rows) on Enterprise `no_limit()`!
    let big_prep_req = CarbonPrepareUploadRequest {
        tenant_id: "corp_vpc".to_owned(),
        tier: CarbonTierWire::Free, // upgraded to `no_limit()` on self-hosted Enterprise!
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
    };
    let big_resp = jope_carbon::carbon_https_agent()
        .post(&format!("{}/v1/carbon/uploads/prepare", ent_srv.base_url()))
        .config()
        .http_status_as_error(false)
        .build()
        .header("content-type", "application/json")
        .send(&serde_json::to_vec(&big_prep_req).unwrap())
        .expect("POST enterprise big upload prepare");
    assert_eq!(
        big_resp.status().as_u16(),
        200,
        "Enterprise self-hosted server must admit >5,000 rows and >64 features (no_limit)"
    );

    // Run a complete end-to-end `Effort::Fast` + `predict` workflow on `carbon serve`.
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

    // 2. Cloud gateway (`enterprise_self_hosted: false`) MUST reject Free tier > 5,000 rows & High effort!
    let mut cloud_cfg = CarbonEnterpriseServeConfig::enterprise_default(
        "127.0.0.1:0".parse().unwrap(),
        root.path().join("cloud_free_check"),
    );
    cloud_cfg.enterprise_self_hosted = false;
    let mut cloud_srv =
        CarbonEnterpriseServer::spawn_background(cloud_cfg).expect("spawn cloud-mode gateway");
    let cloud_reject = jope_carbon::carbon_https_agent()
        .post(&format!(
            "{}/v1/carbon/uploads/prepare",
            cloud_srv.base_url()
        ))
        .config()
        .http_status_as_error(false)
        .build()
        .header("content-type", "application/json")
        .send(&serde_json::to_vec(&big_prep_req).unwrap())
        .expect("POST cloud free >5000 rows");
    assert_eq!(
        cloud_reject.status().as_u16(),
        403,
        "Cloud Free tier must reject > 5,000 rows with HTTP 403"
    );

    let (cloud_client, _) =
        CarbonThinClient::connect_fastest(vec![cloud_srv.base_url()]).expect("connect cloud");
    cloud_client
        .upload_prepared("anon_user", CarbonTierWire::Free, &prepared)
        .expect("upload small dataset on Free tier");
    let high_err = cloud_client
        .start_run(
            "run_01JCCARB0NFRE0000000000001".parse().unwrap(),
            "ds_444444444444444444444444".parse().unwrap(),
            "anon_user",
            CarbonTierWire::Free,
            Effort::High,
            "rmse",
            1,
            &prepared,
        )
        .expect_err("Free tier must reject High effort");
    assert!(high_err.contains("403"));

    // 3. Verify pure `aws-lc-rs` SigV4 S3 PUT presigner determinism.
    let presigned_s3 = CarbonPresigner::presign_s3_put_v4(
        "CARBON_SIGV4_TEST_ID",
        "carbon-sigv4-unit-test-fixture-input",
        "us-east-2",
        "jopedime-carbon-cas-us-east-2",
        "blobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "20261110T213000Z",
        900,
        &"a".repeat(64),
    )
    .expect("presign S3 PUT v4");
    assert!(presigned_s3.starts_with(
        "https://jopedime-carbon-cas-us-east-2.s3.us-east-2.amazonaws.com/blobs/aaaa"
    ));
    assert!(presigned_s3.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"));
    assert!(presigned_s3.contains("X-Amz-Signature="));

    ent_srv.shutdown();
    cloud_srv.shutdown();
}
