//! End-to-end integration tests for real OS-subprocess AWS Lambda Custom Runtime (`carbon-edge` +
//! `carbon-worker` `engine` + `carbon-worker` `eval`), loopback SigV4 S3 CAS (`If-None-Match: *` ->
//! `412 PreconditionFailed`), SigV4 Secrets Manager per-role IAM isolation (`DEP-SEC-01`), SigV4
//! Lambda Invoke (`lambda://` thin client + `X-Amz-Function-Error: Unhandled`), API Gateway HTTP API
//! v2.0 30-second timeout framing (`504`), `MEM-01` secret eviction, `champion_cache` warm hits +
//! wrong-`U_DEK` refusal, and passive `/downloads/*` redirect routes (`edge_downloads.rs`).

use std::fs;
use std::path::PathBuf;

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonExecutionModeWire, CarbonExplainRequest, CarbonExportLangWire,
        CarbonPrepareUploadRequest, CarbonPrepareUploadResponse, CarbonRunStateWire,
        CarbonTierWire, CarbonUploadSlotWire,
    },
    decimal::DecimalU64,
    ids::{DatasetId, RunId},
    Effort,
};
use jope_carbon::{
    decode_p256_pubkey_b64, fetch_secretsmanager_secret_string, generate_random_u_dek,
    now_epoch_ms, open_engine_invoker, open_eval_invoker, open_remote_cas, send_raw_edge_http,
    send_raw_put_url, sign_aws_v4_headers_with_extra, wrap_u_dek_to_pubkey, CarbonAwsCredentials,
    CarbonEmuConfig, CarbonEngineKek, CarbonEngineWireCommand, CarbonEvalWireRequest,
    CarbonLocalLambdaEmulator, CarbonRealWorkerBinaries, CarbonThinClient,
    EMU_CELL_SECRETS_VERSION, EMU_CLIENT_ACCESS_KEY_ID, EMU_CLIENT_SECRET_ACCESS_KEY,
    EMU_EDGE_ACCESS_KEY_ID, EMU_EDGE_FUNCTION_NAME, EMU_EDGE_SECRETS_VERSION,
    EMU_EDGE_SECRET_ACCESS_KEY, EMU_ENGINE_ACCESS_KEY_ID, EMU_ENGINE_FUNCTION_NAME,
    EMU_ENGINE_SECRET_ACCESS_KEY, EMU_EVAL_ACCESS_KEY_ID, EMU_EVAL_FUNCTION_NAME,
    EMU_EVAL_SECRET_ACCESS_KEY, EMU_S3_BUCKET,
};
use jope_service::worker::{Attempt, Limits, Request as WorkerRequest};

use super::ScopedProcessEnv;

fn build_synthetic_matrix(rows: usize) -> (Vec<String>, Vec<Vec<f64>>) {
    let names = vec!["x1".to_owned(), "x2".to_owned(), "y".to_owned()];
    let mut data = Vec::with_capacity(rows);
    for i in 0..rows {
        let x1 = (i as f64) * 0.12 - 3.0;
        let x2 = ((i % 9) as f64) * 0.35 - 1.4;
        let y = 2.25 * x1 - 1.15 * x2 + 0.75;
        data.push(vec![x1, x2, y]);
    }
    (names, data)
}

fn sample_worker_request() -> WorkerRequest {
    WorkerRequest {
        attempt: Attempt {
            run: RunId::new(),
            round: 0,
            slot: 0,
            attempt: 1,
        },
        matrix_path: PathBuf::from("fit.jmat"),
        fold_spec: serde_json::json!({"kind": "k_fold", "folds": 3}),
        dsl: "x1".to_owned(),
        objective: serde_json::json!("rmse"),
        seed: 7,
        limits: Limits {
            wall_ms: 10_000,
            rss_bytes: 256 * 1_048_576,
        },
    }
}

#[test]
fn real_os_subprocess_lambda_edge_engine_eval_e2e_and_lambda_transport() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
    let worker_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-worker"));
    assert!(edge_bin.is_file() && worker_bin.is_file());

    let mut cfg = CarbonEmuConfig::for_dir(tmp.path().join("real_lambda_cell"))
        .with_real_worker_binaries(CarbonRealWorkerBinaries::new(edge_bin, worker_bin));
    cfg.force_hybrid_fanout = true;
    cfg.eval_worker_threads = 2;

    let mut emu = CarbonLocalLambdaEmulator::start(cfg).expect("start real-worker Lambda emulator");
    let pro_tok = emu
        .mint_capability_token("tenant_real_pro", CarbonTierWire::Pro, false)
        .expect("mint pro token");

    // 1. Passive `/carbon/*` redirect telemetry routes (`edge_downloads.rs`) through real `carbon-edge` + API GW v2 framing
    let (dl_code, dl_body) = send_raw_edge_http(
        &emu.base_url(),
        "GET",
        "/carbon/releases/latest/carbon-linux-x86_64-musl",
        &[],
        None,
    )
    .expect("GET /carbon/releases/latest/...");
    assert!(
        matches!(dl_code, 200 | 302),
        "unexpected /carbon/releases status {dl_code}: {}",
        String::from_utf8_lossy(&dl_body)
    );
    let (dl_ver_code, _) = send_raw_edge_http(
        &emu.base_url(),
        "GET",
        "/carbon/releases/carbon-cli-v1.1.0/SHA256SUMS.txt",
        &[],
        None,
    )
    .expect("GET /carbon/releases/carbon-cli-v1.1.0/SHA256SUMS.txt");
    assert!(matches!(dl_ver_code, 200 | 302));
    let (dl_inst_code, _) =
        send_raw_edge_http(&emu.base_url(), "GET", "/carbon/install.sh", &[], None)
            .expect("GET /carbon/install.sh");
    assert!(matches!(dl_inst_code, 200 | 302));
    let (dl_bad_code, _) = send_raw_edge_http(
        &emu.base_url(),
        "GET",
        "/carbon/releases/latest/../../etc/passwd",
        &[],
        None,
    )
    .expect("GET traversal /carbon/releases");
    assert!(matches!(dl_bad_code, 400 | 404));

    // 2. Pro `HybridFanOut` E2E across real OS subprocesses (`carbon-edge` -> `carbon-worker engine` -> `carbon-worker eval`),
    //    plus `MEM-01` `cached_secret_counts()`, `champion_cache` warm hits, and forged `U_DEK` refusal.
    let (client, probed) = CarbonThinClient::connect_fastest(vec![emu.base_url()])
        .expect("connect_fastest real-worker emulator");
    assert_eq!(
        probed[0].engine_kek_pubkey_b64.as_deref(),
        Some(emu.engine_kek_pubkey_b64())
    );
    let client = client.with_capability_token(pro_tok.clone());
    assert_eq!(client.cached_secret_counts(), (0, 0, 0));

    let (col_names, rows) = build_synthetic_matrix(96);
    let prepared =
        CarbonThinClient::prepare_numeric_dataset(col_names.clone(), &rows, "y", 42, 0.20)
            .expect("prepare numeric dataset");
    let _upload_resp = client
        .upload_prepared("tenant_real_pro", CarbonTierWire::Pro, &prepared)
        .expect("upload encrypted dataset to SigV4 S3 CAS");
    assert_eq!(client.cached_secret_counts(), (0, 1, 1));

    let dataset_id: DatasetId = "ds_0123456789abcdef01234567".parse().expect("dataset_id");
    let run_a_ulid = RunId::new();
    let run_b_ulid = RunId::new();
    let settled_a = client
        .start_run(
            run_a_ulid.clone(),
            dataset_id.clone(),
            "tenant_real_pro",
            CarbonTierWire::Pro,
            Effort::Fast,
            "rmse",
            42,
            &prepared,
        )
        .expect("start & settle run_a");
    assert_eq!(settled_a.state, CarbonRunStateWire::Completed);
    assert_eq!(
        settled_a.execution_mode,
        CarbonExecutionModeWire::HybridFanOut
    );

    let settled_b = client
        .start_run(
            run_b_ulid.clone(),
            dataset_id,
            "tenant_real_pro",
            CarbonTierWire::Pro,
            Effort::Fast,
            "mae",
            43,
            &prepared,
        )
        .expect("start & settle sibling run_b sharing upload");
    assert_eq!(settled_b.state, CarbonRunStateWire::Completed);
    assert_eq!(client.cached_secret_counts(), (2, 1, 1));

    let telem = emu.telemetry();
    assert!(telem.edge_invocations >= 4);
    assert!(telem.engine_invocations >= 2);
    assert!(
        telem.eval_invocations >= 2,
        "expected real carbon-worker eval subprocess invocations, got {telem:?}"
    );

    let run_a_id = run_a_ulid.to_string();
    let pred_cols = vec!["x1".to_owned(), "x2".to_owned()];
    let pred_rows: Vec<Vec<Option<f64>>> = (0..12)
        .map(|i| vec![Some((i as f64) * 0.2), Some(-0.5 + (i as f64) * 0.1)])
        .collect();

    // First `/predict` populates `champion_cache`; second `/predict` + `/explain` + `/export` hit `champion_cache`!
    let pred_1 = client
        .predict(
            "tenant_real_pro",
            &run_a_id,
            pred_cols.clone(),
            pred_rows.clone(),
        )
        .expect("predict cold");
    let pred_2 = client
        .predict(
            "tenant_real_pro",
            &run_a_id,
            pred_cols.clone(),
            pred_rows.clone(),
        )
        .expect("predict warm champion_cache");
    assert_eq!(pred_1.predictions, pred_2.predictions);

    // CAS-backed `/predict` through real `carbon-edge` + `carbon-worker` SigV4 S3 CAS
    let pred_cas = {
        let _env = ScopedProcessEnv::set(&[("CARBON_FORCE_PREDICT_CAS", "1")]);
        client
            .predict(
                "tenant_real_pro",
                &run_a_id,
                pred_cols.clone(),
                pred_rows.clone(),
            )
            .expect("CAS-backed predict on real workers")
    };
    assert!(pred_cas.encrypted_predictions_cas_url.is_some());
    assert_eq!(pred_1.predictions, pred_cas.predictions);

    let explain_resp = client
        .explain_run("tenant_real_pro", &run_a_id, Some(2))
        .expect("explain warm champion_cache");
    assert_eq!(explain_resp.feature_importances.len(), 2);

    let export_py = client
        .export_run("tenant_real_pro", &run_a_id, CarbonExportLangWire::Python)
        .expect("export python warm champion_cache");
    assert!(export_py
        .source_code
        .as_deref()
        .is_some_and(|s| !s.is_empty()));

    // Cryptographic refusal when a forged/wrong `wrapped_u_dek_b64` is supplied while `champion_cache` is hot!
    let wrong_udek = generate_random_u_dek().expect("gen wrong udek");
    let kek_pub_bytes =
        decode_p256_pubkey_b64(emu.engine_kek_pubkey_b64()).expect("decode kek pubkey");
    let wrong_cudk =
        wrap_u_dek_to_pubkey(&wrong_udek, &kek_pub_bytes, "tenant_real_pro", &run_a_id)
            .expect("wrap wrong udek");
    let resp_kek = CarbonEngineKek::generate().expect("gen resp_kek");
    let forged_explain_body = serde_json::to_vec(&CarbonExplainRequest {
        tenant_id: "tenant_real_pro".to_owned(),
        top_k: Some(2),
        client_response_pubkey_b64: Some(resp_kek.pubkey_b64()),
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(wrong_cudk)),
    })
    .unwrap();
    let bearer_hdr = format!("Bearer {pro_tok}");
    let (forged_code, _) = send_raw_edge_http(
        &emu.base_url(),
        "POST",
        &format!("/v1/carbon/runs/{run_a_id}/explain"),
        &[("authorization", &bearer_hdr)],
        Some(&forged_explain_body),
    )
    .expect("POST explain with wrong wrapped_u_dek");
    assert!(
        forged_code >= 400,
        "champion_cache must refuse caller with mismatched U_DEK (got HTTP {forged_code})"
    );

    // `MEM-01`: Shred `run_a` -> `run_a` U_DEK evicted `(1, 1, 1)` while shared upload stays for `run_b`;
    // then shred `run_b` -> all cached secrets drop to `(0, 0, 0)`!
    let shred_a = client
        .shred_run("tenant_real_pro", &run_a_id)
        .expect("shred run_a");
    assert!(shred_a.shredded);
    assert_eq!(client.cached_secret_counts(), (1, 1, 1));

    let shred_b = client
        .shred_run("tenant_real_pro", &run_b_ulid.to_string())
        .expect("shred run_b");
    assert!(shred_b.shredded);
    assert_eq!(client.cached_secret_counts(), (0, 0, 0));

    // 3. Direct `lambda://` `CarbonThinClient` transport invoking `carbon-emu-edge:live` via SigV4 Lambda Invoke!
    {
        let lambda_ep = format!("lambda://us-east-2/{EMU_EDGE_FUNCTION_NAME}:live");
        let _env = ScopedProcessEnv::set_opt(&[
            (
                "CARBON_LAMBDA_ENDPOINT_URL",
                Some(&emu.lambda_invoke_endpoint_url()),
            ),
            ("AWS_ACCESS_KEY_ID", Some(EMU_CLIENT_ACCESS_KEY_ID)),
            ("AWS_SECRET_ACCESS_KEY", Some(EMU_CLIENT_SECRET_ACCESS_KEY)),
            ("AWS_SESSION_TOKEN", None),
        ]);
        let (mut lclient, lprobed) =
            CarbonThinClient::connect_fastest(vec![lambda_ep]).expect("connect lambda:// edge");
        assert_eq!(lprobed.len(), 1);
        let ver = lclient
            .fetch_server_version()
            .expect("lambda:// version")
            .expect("version present");
        assert!(!ver.version.is_empty());

        let reg = lclient
            .register_free(Some("lambda_transport_user"))
            .expect("lambda:// register_free");
        assert!(reg.capability_token.starts_with("ctk1."));

        let lprep = CarbonThinClient::prepare_numeric_dataset(col_names, &rows[..48], "y", 7, 0.20)
            .expect("prepare lambda dataset");
        lclient
            .upload_prepared(&reg.tenant_id, CarbonTierWire::Free, &lprep)
            .expect("lambda:// upload dataset");
        let lrun_id = RunId::new();
        let lsettled = lclient
            .start_run(
                lrun_id.clone(),
                "ds_fedcba9876543210fedcba98".parse().expect("ds"),
                &reg.tenant_id,
                CarbonTierWire::Free,
                Effort::Fast,
                "rmse",
                7,
                &lprep,
            )
            .expect("lambda:// start_run");
        assert_eq!(lsettled.state, CarbonRunStateWire::Completed);
        let lpred = lclient
            .predict(&reg.tenant_id, &lrun_id.to_string(), pred_cols, pred_rows)
            .expect("lambda:// predict");
        assert_eq!(lpred.row_count, 12);
        let lshred = lclient
            .shred_run(&reg.tenant_id, &lrun_id.to_string())
            .expect("lambda:// shred");
        assert!(lshred.shredded);
    }

    emu.shutdown();
}

#[test]
fn sigv4_s3_secrets_manager_lambda_iam_isolation_and_apigw_504() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
    let worker_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-worker"));

    let mut cfg = CarbonEmuConfig::for_dir(tmp.path().join("iam_iso_cell"))
        .with_real_worker_binaries(CarbonRealWorkerBinaries::new(edge_bin.clone(), worker_bin));
    cfg.eval_worker_threads = 1;
    let mut emu = CarbonLocalLambdaEmulator::start(cfg).expect("start emu");

    // 1. SigV4 S3 CAS: unsigned request -> 403, `put_state_object_if_absent` -> `true` then `false` (`412 PreconditionFailed`),
    //    and tampered presigned `PUT` -> 403/400.
    let (unauth_s3_code, _) = send_raw_edge_http(
        &emu.s3_cas_endpoint_url(),
        "GET",
        &format!("/{EMU_S3_BUCKET}/v1/state/runs/t1/run_01/status.json"),
        &[],
        None,
    )
    .expect("unsigned S3 GET");
    assert_eq!(unauth_s3_code, 403);

    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("CARBON_S3_ENDPOINT_URL", Some(&emu.s3_cas_endpoint_url())),
            ("AWS_ACCESS_KEY_ID", Some(EMU_ENGINE_ACCESS_KEY_ID)),
            ("AWS_SECRET_ACCESS_KEY", Some(EMU_ENGINE_SECRET_ACCESS_KEY)),
            ("AWS_SESSION_TOKEN", None),
        ]);
        let cas = open_remote_cas(&emu.s3_cas_url(), "us-east-2", "")
            .expect("open CarbonS3Cas against loopback emulator");
        let lease_key = "runs/tenant_ifnone/run_ifnone_01/lease.json";
        assert!(
            cas.put_state_object_if_absent(lease_key, br#"{"holder":"a"}"#)
                .expect("first If-None-Match PUT"),
            "first conditional PUT must succeed"
        );
        assert!(
            !cas.put_state_object_if_absent(lease_key, br#"{"holder":"b"}"#)
                .expect("second If-None-Match PUT -> 412 -> Ok(false)"),
            "second conditional PUT must observe 412 PreconditionFailed -> Ok(false)"
        );
        assert_eq!(
            cas.get_state_object(lease_key)
                .expect("GET lease")
                .as_deref(),
            Some(&br#"{"holder":"a"}"#[..])
        );
        cas.delete_state_object(lease_key).expect("DELETE lease");
        assert!(cas
            .get_state_object(lease_key)
            .expect("GET deleted")
            .is_none());
    }

    // Tampered presigned PUT payload against real `carbon-edge` `/v1/carbon/uploads/prepare`
    let (mut client, _) = CarbonThinClient::connect_fastest(vec![emu.base_url()]).unwrap();
    let anon = client.authenticate_anonymous().expect("anon auth");
    let anon_hdr = format!("Bearer {}", anon.capability_token);
    let prep_body = serde_json::to_vec(&CarbonPrepareUploadRequest {
        tenant_id: anon.tenant_id,
        tier: CarbonTierWire::Free,
        fit_rows: 20,
        holdout_rows: 5,
        feature_count: 2,
        fit_jmat: CarbonUploadSlotWire {
            sha256: "a".repeat(64),
            bytes: DecimalU64(16),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: "b".repeat(64),
            bytes: DecimalU64(16),
        },
        encrypted_cuen: true,
    })
    .unwrap();
    let (prep_code, prep_resp_bytes) = send_raw_edge_http(
        &emu.base_url(),
        "POST",
        "/v1/carbon/uploads/prepare",
        &[("authorization", &anon_hdr)],
        Some(&prep_body),
    )
    .expect("prepare upload");
    assert_eq!(
        prep_code,
        200,
        "prepare failed: {}",
        String::from_utf8_lossy(&prep_resp_bytes)
    );
    let prep_val: CarbonPrepareUploadResponse = serde_json::from_slice(&prep_resp_bytes).unwrap();
    let bad_put = send_raw_put_url(&prep_val.fit_target.url, b"wrong_16_bytes!!", true);
    if let Ok((code, _)) = bad_put {
        assert!(
            code >= 400,
            "tampered SHA-256 presigned PUT must be rejected, got {code}"
        );
    }

    // 2. SigV4 Secrets Manager per-role IAM isolation (`DEP-SEC-01`)
    let edge_arn = emu.edge_secret_arn().to_owned();
    let cell_arn = emu.cell_secret_arn().to_owned();
    {
        // `edge` role: allowed `edge_arn` (no seal/KEK scalar!), denied `cell_arn`, denied wrong VersionId
        let _env = ScopedProcessEnv::set_opt(&[
            (
                "CARBON_SECRETSMANAGER_ENDPOINT_URL",
                Some(&emu.secrets_manager_endpoint_url()),
            ),
            ("AWS_ACCESS_KEY_ID", Some(EMU_EDGE_ACCESS_KEY_ID)),
            ("AWS_SECRET_ACCESS_KEY", Some(EMU_EDGE_SECRET_ACCESS_KEY)),
            ("AWS_SESSION_TOKEN", None),
        ]);
        let edge_json = fetch_secretsmanager_secret_string(
            "us-east-2",
            &edge_arn,
            Some(EMU_EDGE_SECRETS_VERSION),
        )
        .expect("edge role read edge secret");
        assert!(!edge_json.contains("seal_secret_hex"));
        assert!(!edge_json.contains("engine_kek_hex"));
        assert!(
            fetch_secretsmanager_secret_string("us-east-2", &cell_arn, None).is_err(),
            "edge IAM role must be denied cell-secrets ARN"
        );
        assert!(
            fetch_secretsmanager_secret_string("us-east-2", &edge_arn, Some("wrong-ver")).is_err(),
            "wrong VersionId must fail closed"
        );
    }
    {
        // `engine` role: allowed `cell_arn`, denied `edge_arn`, denied wrong VersionId
        let _env = ScopedProcessEnv::set_opt(&[
            (
                "CARBON_SECRETSMANAGER_ENDPOINT_URL",
                Some(&emu.secrets_manager_endpoint_url()),
            ),
            ("AWS_ACCESS_KEY_ID", Some(EMU_ENGINE_ACCESS_KEY_ID)),
            ("AWS_SECRET_ACCESS_KEY", Some(EMU_ENGINE_SECRET_ACCESS_KEY)),
            ("AWS_SESSION_TOKEN", None),
        ]);
        let cell_json = fetch_secretsmanager_secret_string(
            "us-east-2",
            &cell_arn,
            Some(EMU_CELL_SECRETS_VERSION),
        )
        .expect("engine role read cell secret");
        assert!(cell_json.contains("seal_secret_hex"));
        assert!(fetch_secretsmanager_secret_string("us-east-2", &edge_arn, None).is_err());
        assert!(
            fetch_secretsmanager_secret_string("us-east-2", &cell_arn, Some("wrong-ver")).is_err()
        );
    }
    for (ak, sk) in [
        (EMU_EVAL_ACCESS_KEY_ID, EMU_EVAL_SECRET_ACCESS_KEY),
        (EMU_CLIENT_ACCESS_KEY_ID, EMU_CLIENT_SECRET_ACCESS_KEY),
    ] {
        let _env = ScopedProcessEnv::set_opt(&[
            (
                "CARBON_SECRETSMANAGER_ENDPOINT_URL",
                Some(&emu.secrets_manager_endpoint_url()),
            ),
            ("AWS_ACCESS_KEY_ID", Some(ak)),
            ("AWS_SECRET_ACCESS_KEY", Some(sk)),
            ("AWS_SESSION_TOKEN", None),
        ]);
        assert!(fetch_secretsmanager_secret_string("us-east-2", &edge_arn, None).is_err());
        assert!(fetch_secretsmanager_secret_string("us-east-2", &cell_arn, None).is_err());
    }

    // 3. SigV4 Lambda Invoke IAM isolation + `200 + X-Amz-Function-Error: Unhandled` propagation
    {
        // `eval` role cannot invoke `eval` or `engine`
        let _env = ScopedProcessEnv::set_opt(&[
            (
                "CARBON_LAMBDA_ENDPOINT_URL",
                Some(&emu.lambda_invoke_endpoint_url()),
            ),
            ("AWS_ACCESS_KEY_ID", Some(EMU_EVAL_ACCESS_KEY_ID)),
            ("AWS_SECRET_ACCESS_KEY", Some(EMU_EVAL_SECRET_ACCESS_KEY)),
            ("AWS_SESSION_TOKEN", None),
        ]);
        let eval_inv =
            open_eval_invoker(&format!("lambda://{EMU_EVAL_FUNCTION_NAME}"), "us-east-2");
        let bad_req = CarbonEvalWireRequest {
            fit_jmat_sha256: "0".repeat(64),
            blob_scope: None,
            request: sample_worker_request(),
            fit_champion: false,
            decryption_grant: None,
        };
        assert!(
            eval_inv.invoke_eval(&bad_req).is_err(),
            "eval role must not have lambda:InvokeFunction permission"
        );
    }
    {
        // `client` role cannot invoke internal `engine` Lambda
        let _env = ScopedProcessEnv::set_opt(&[
            (
                "CARBON_LAMBDA_ENDPOINT_URL",
                Some(&emu.lambda_invoke_endpoint_url()),
            ),
            ("AWS_ACCESS_KEY_ID", Some(EMU_CLIENT_ACCESS_KEY_ID)),
            ("AWS_SECRET_ACCESS_KEY", Some(EMU_CLIENT_SECRET_ACCESS_KEY)),
            ("AWS_SESSION_TOKEN", None),
        ]);
        let eng_inv =
            open_engine_invoker(&format!("lambda://{EMU_ENGINE_FUNCTION_NAME}"), "us-east-2");
        let cmd = CarbonEngineWireCommand::Explain {
            run_id: "run_01JCCARB0NINVOKER000000001".to_owned(),
            explain: CarbonExplainRequest {
                tenant_id: "t".to_owned(),
                top_k: None,
                client_response_pubkey_b64: None,
                wrapped_u_dek_b64: None,
            },
        };
        assert!(eng_inv.invoke_engine(&cmd).is_err());
    }
    {
        // `engine` role invoking real `carbon-worker` (`eval`) with missing CAS blob -> worker returns error ->
        // emulator encodes `200 OK` + `X-Amz-Function-Error: Unhandled` -> `CarbonAwsLambdaEvalInvoker` surfaces `Err`!
        let _env = ScopedProcessEnv::set_opt(&[
            (
                "CARBON_LAMBDA_ENDPOINT_URL",
                Some(&emu.lambda_invoke_endpoint_url()),
            ),
            ("AWS_ACCESS_KEY_ID", Some(EMU_ENGINE_ACCESS_KEY_ID)),
            ("AWS_SECRET_ACCESS_KEY", Some(EMU_ENGINE_SECRET_ACCESS_KEY)),
            ("AWS_SESSION_TOKEN", None),
        ]);
        let eval_inv = open_eval_invoker(
            &format!("lambda://{EMU_EVAL_FUNCTION_NAME}:live"),
            "us-east-2",
        );
        let missing_blob_req = CarbonEvalWireRequest {
            fit_jmat_sha256: "f".repeat(64),
            blob_scope: Some("t_err/u_err".to_owned()),
            request: sample_worker_request(),
            fit_champion: false,
            decryption_grant: None,
        };
        let err = eval_inv
            .invoke_eval(&missing_blob_req)
            .expect_err("missing CAS blob must propagate X-Amz-Function-Error");
        assert!(
            err.contains("Unhandled") || err.contains("missing") || err.contains("error"),
            "unexpected eval error string: {err}"
        );

        // Also send raw SigV4 `Event` invocation with `edge` credentials to verify `202 Accepted`
        let creds = CarbonAwsCredentials {
            access_key_id: EMU_EDGE_ACCESS_KEY_ID.to_owned(),
            secret_access_key: EMU_EDGE_SECRET_ACCESS_KEY.to_owned().into(),
            session_token: None,
        };
        let host = emu.bound_addr().to_string();
        let uri = format!("/2015-03-31/functions/{EMU_ENGINE_FUNCTION_NAME}/invocations");
        let body = br#"{"invalid_async_test":true}"#;
        let hdrs = sign_aws_v4_headers_with_extra(
            &creds,
            "us-east-2",
            "lambda",
            "POST",
            &host,
            &uri,
            "Qualifier=live",
            body,
            now_epoch_ms() / 1_000,
            &[("x-amz-invocation-type", "Event")],
        );
        let mut req = ureq::Agent::new_with_defaults()
            .post(&format!("{}{uri}?Qualifier=live", emu.base_url()))
            .config()
            .http_status_as_error(false)
            .build();
        for (k, v) in &hdrs {
            req = req.header(k, v);
        }
        let resp = req.send(&body[..]).expect("send async Event invoke");
        assert_eq!(resp.status().as_u16(), 202);
    }
    emu.shutdown();

    // 4. API Gateway v2 30-second timeout framing (`504 Gateway Timeout` when edge exceeds `apigw_timeout_ms`)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let slow_edge = tmp.path().join("slow_edge.sh");
        // Wrapper around `carbon-edge` that delays startup by 250ms while `apigw_timeout_ms` is 60ms!
        let script = format!(
            "#!/usr/bin/env bash\nsleep 0.25\nexec '{}' \"$@\"\n",
            edge_bin.display()
        );
        fs::write(&slow_edge, script).unwrap();
        fs::set_permissions(&slow_edge, fs::Permissions::from_mode(0o755)).unwrap();

        let mut slow_cfg = CarbonEmuConfig::for_dir(tmp.path().join("apigw_504_cell"));
        slow_cfg.edge_binary_path = Some(slow_edge);
        slow_cfg.simulate_apigw_v2 = true;
        slow_cfg.apigw_timeout_ms = 60;
        let mut slow_emu = CarbonLocalLambdaEmulator::start(slow_cfg).expect("start slow_emu");
        let (code_504, body_504) =
            send_raw_edge_http(&slow_emu.base_url(), "GET", "/v1/carbon/ping", &[], None)
                .expect("GET ping exceeding apigw_timeout_ms");
        assert_eq!(
            code_504,
            504,
            "expected API Gateway v2 504 timeout, got {code_504}: {}",
            String::from_utf8_lossy(&body_504)
        );
        slow_emu.shutdown();
    }
}
