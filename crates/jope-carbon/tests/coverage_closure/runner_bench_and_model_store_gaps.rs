//! Section 14 coverage closure for `src/cli/runner.rs` (`TE-57..TE-64` `execute_carbon_cli_benchmark*`,
//! `dry_run`, `show_stderr_progress`, `stop_after_ms` + `resume_after_stop`, `persist_handle`),
//! `src/cli/bench_suite.rs` (`--csv-dir` & Markdown rendering), and `src/cli/model_store.rs`
//! (`.tar.zst` self-upgrade, labeled CSV quantiles, `--include-inputs`, `--id-column`, `--out -`,
//! and fallback capability token resolution).

use std::collections::BTreeMap;
use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use aws_lc_rs::{
    rand::SystemRandom,
    signature::{EcdsaKeyPair, KeyPair, ECDSA_P256_SHA256_ASN1_SIGNING},
};
use axum::{
    body::Body,
    extract::Path as AxumPath,
    http::{Response, StatusCode},
    routing::get,
    Router,
};
use base64::Engine as _;

use jope_api::{carbon::CarbonTierWire, Effort};
use jope_carbon::{
    build_client_for_handle, build_deterministic_single_bin_tar, compute_residual_quantiles,
    execute_carbon_cli_benchmark, execute_carbon_cli_benchmark_with_token,
    execute_carbon_cli_run_with_ux, execute_predict_csv_with_handle,
    execute_scientific_benchmark_suite, execute_self_upgrade, execute_self_upgrade_with_options,
    load_model_handle_by_spec, render_benchmark_markdown, render_pretty_run_scorecard,
    render_scientific_benchmark_markdown, save_stored_cloud_token, sha256_lower_hex,
    CarbonCliRunOptions, CarbonCliRunUxConfig, CarbonEmuConfig, CarbonEnterpriseServeConfig,
    CarbonEnterpriseServer, CarbonLocalLambdaEmulator, CarbonScientificBenchmarkOptions,
    CARBON_BUILD_TARGET,
};

use super::{spawn_loopback_axum, ScopedProcessEnv};

const SPKI_P256_PREFIX: [u8; 26] = [
    0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x08, 0x2a,
    0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03, 0x42, 0x00,
];

fn ephemeral_signing_keypair() -> (EcdsaKeyPair, String) {
    let rng = SystemRandom::new();
    let pkcs8 =
        EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, &rng).expect("gen p256");
    let kp =
        EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, pkcs8.as_ref()).expect("parse");
    let mut spki = Vec::with_capacity(91);
    spki.extend_from_slice(&SPKI_P256_PREFIX);
    spki.extend_from_slice(kp.public_key().as_ref());
    let b64 = base64::engine::general_purpose::STANDARD.encode(&spki);
    let pem = format!(
        "-----BEGIN PUBLIC KEY-----\n{}\n{}\n-----END PUBLIC KEY-----\n",
        &b64[..64],
        &b64[64..]
    );
    (kp, pem)
}

#[test]
fn runner_ops_benchmark_free_pro_enterprise_and_ux_modes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg_dir = tmp.path().join("jopedime_cfg");
    fs::create_dir_all(&cfg_dir).expect("mkdir cfg_dir");

    // 1. Offline `ux.dry_run = true` across SoloSingleLambda + HybridFanOut + maximize objective (`r2`).
    let dry_solo = execute_carbon_cli_run_with_ux(
        &CarbonCliRunOptions {
            endpoints: Vec::new(),
            csv_path: None,
            target_column: "y".to_owned(),
            synthetic_rows: 120,
            synthetic_features: 4,
            tenant_id: "dry_t".to_owned(),
            tier: CarbonTierWire::Free,
            effort: Effort::Fast,
            objective: "r2".to_owned(),
            seed: 7,
            holdout_fraction: 0.20,
            stop_after_ms: None,
            resume_after_stop: false,
            predict_rows: 4,
            auth_token: None,
            shred_after_run: false,
        },
        &CarbonCliRunUxConfig {
            dry_run: true,
            ..CarbonCliRunUxConfig::default()
        },
    )
    .expect("dry_run solo r2");
    assert_eq!(dry_solo.run_id, "run_dry_run_preflight");
    assert_eq!(dry_solo.objective_direction.as_deref(), Some("maximize"));

    let dry_hybrid = execute_carbon_cli_run_with_ux(
        &CarbonCliRunOptions {
            endpoints: Vec::new(),
            csv_path: None,
            target_column: "y".to_owned(),
            synthetic_rows: 800,
            synthetic_features: 6,
            tenant_id: "dry_pro".to_owned(),
            tier: CarbonTierWire::Pro,
            effort: Effort::High,
            objective: "rmse".to_owned(),
            seed: 9,
            holdout_fraction: 0.20,
            stop_after_ms: None,
            resume_after_stop: false,
            predict_rows: 4,
            auth_token: None,
            shred_after_run: true,
        },
        &CarbonCliRunUxConfig {
            dry_run: true,
            ..CarbonCliRunUxConfig::default()
        },
    )
    .expect("dry_run hybrid");
    assert!(dry_hybrid.shredded_after_run);
    let scorecard = render_pretty_run_scorecard(&dry_hybrid);
    assert!(scorecard.contains("run_dry_run_preflight"));

    // 2. Live `CarbonLocalLambdaEmulator`: Pro ops benchmark (`1, 2, 3, 4, 4b, 5`) + `render_benchmark_markdown`.
    let mut emu_cfg = CarbonEmuConfig::for_dir(tmp.path().join("emu_bench"));
    emu_cfg.force_hybrid_fanout = false;
    emu_cfg.eval_worker_threads = 4;
    let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start emu");
    let pro_tok = emu
        .mint_capability_token("pro_bench_tenant", CarbonTierWire::Pro, false)
        .expect("mint pro token");

    let pro_bench = execute_carbon_cli_benchmark_with_token(
        vec![emu.base_url()],
        CarbonTierWire::Pro,
        "pro_bench_tenant",
        Some(pro_tok.clone()),
    )
    .expect("execute Pro ops benchmark");
    assert!(
        pro_bench.all_passed,
        "Pro ops bench scenarios failed: {:?}",
        pro_bench.scenarios
    );
    assert_eq!(pro_bench.scenarios.len(), 6);
    let bench_md = render_benchmark_markdown(&pro_bench);
    assert!(bench_md.contains("pro_hybrid_fanout_stop_predict_resume_shred"));
    assert!(bench_md.contains("pro_ultra_hybrid_fanout"));
    assert!(bench_md.contains("free_tier_row_cap_403"));

    // 3. `execute_carbon_cli_run_with_ux`: `show_stderr_progress`, `stop_after_ms` + `resume_after_stop`,
    //    and `persist_handle` + `save_model_path`.
    let saved_path = tmp.path().join("saved_ux_model.carbon.json");
    let ux_receipt = {
        let _env = ScopedProcessEnv::set_opt(&[
            ("CARBON_CONFIG_DIR", Some(cfg_dir.to_str().expect("utf8"))),
            ("CARBON_AUTH_TOKEN", None),
            ("CARBON_AUTH_SECRET_HEX", None),
        ]);
        execute_carbon_cli_run_with_ux(
            &CarbonCliRunOptions {
                endpoints: vec![emu.base_url()],
                csv_path: None,
                target_column: "y".to_owned(),
                synthetic_rows: 140,
                synthetic_features: 4,
                tenant_id: "pro_bench_tenant".to_owned(),
                tier: CarbonTierWire::Pro,
                effort: Effort::Medium,
                objective: "rmse".to_owned(),
                seed: 55,
                holdout_fraction: 0.20,
                stop_after_ms: Some(1),
                resume_after_stop: true,
                predict_rows: 5,
                auth_token: Some(pro_tok.clone()),
                shred_after_run: false,
            },
            &CarbonCliRunUxConfig {
                target_was_explicit: true,
                show_stderr_progress: true,
                persist_handle: true,
                save_model_path: Some(saved_path.clone()),
                cloud_name: Some("dev".to_owned()),
                embed_token: false,
                ..CarbonCliRunUxConfig::default()
            },
        )
        .expect("ux run with progress + stop/resume + save")
    };
    assert!(saved_path.is_file());
    assert!(ux_receipt.saved_model_path.is_some());
    emu.shutdown();
}

#[test]
fn runner_scientific_csv_dir_and_enterprise_benchmark_modes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut emu_cfg = CarbonEmuConfig::for_dir(tmp.path().join("emu_sci_bench"));
    emu_cfg.force_hybrid_fanout = false;
    emu_cfg.eval_worker_threads = 2;
    let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start emu");
    let pro_tok = emu
        .mint_capability_token("pro_bench_tenant", CarbonTierWire::Pro, false)
        .expect("mint pro token");

    // 4. `bench_suite.rs`: `--csv-dir` + `--datasets-dir` (`PreSplitTables`) + `render_scientific_benchmark_markdown`.
    let csv_dir = tmp.path().join("sci_csvs");
    fs::create_dir_all(&csv_dir).expect("mkdir csv_dir");
    let mut csv_a = String::from("x1,x2,y\n");
    for i in 0..64 {
        let x1 = (i as f64) * 0.1;
        let x2 = ((i % 7) as f64) - 3.0;
        let y = 2.5 * x1 - 0.8 * x2 + 1.0;
        csv_a.push_str(&format!("{x1:.4},{x2:.4},{y:.4}\n"));
    }
    fs::write(csv_dir.join("task_alpha.csv"), &csv_a).expect("write task_alpha.csv");

    let datasets_dir = tmp.path().join("sci_datasets");
    let ds_sub = datasets_dir.join("ds_real_01");
    fs::create_dir_all(&ds_sub).expect("mkdir ds_real_01");
    fs::write(ds_sub.join("train.csv"), &csv_a).expect("write train.csv");
    fs::write(ds_sub.join("test.csv"), &csv_a).expect("write test.csv");

    let sci_rep = execute_scientific_benchmark_suite(
        &CarbonScientificBenchmarkOptions {
            endpoints: vec![emu.base_url()],
            pinned_keks: BTreeMap::new(),
            tier: CarbonTierWire::Pro,
            effort: Effort::Fast,
            tenant_id: "pro_bench_tenant".to_owned(),
            auth_token: Some(pro_tok),
            suite: "smoke".to_owned(),
            datasets_dir: Some(datasets_dir),
            csv_dir: Some(csv_dir),
            seed: 42,
        },
        true,
    )
    .expect("scientific benchmark csv_dir + datasets_dir");
    assert_eq!(sci_rep.tasks.len(), 5);
    let sci_md = render_scientific_benchmark_markdown(&sci_rep);
    assert!(sci_md.contains("task_alpha") && sci_md.contains("ds_real_01"));

    emu.shutdown();

    // 5. `CarbonEnterpriseServer`: `execute_carbon_cli_benchmark` (`enterprise_self_hosted == true` branch).
    let listen: SocketAddr = "127.0.0.1:0".parse().expect("addr");
    let srv_cfg =
        CarbonEnterpriseServeConfig::enterprise_default(listen, tmp.path().join("ent_srv"));
    let mut srv = CarbonEnterpriseServer::spawn_background(srv_cfg).expect("spawn ent server");
    let ent_bench =
        execute_carbon_cli_benchmark(vec![srv.base_url()], CarbonTierWire::Free, "vpc_tenant")
            .expect("enterprise ops benchmark");
    assert!(ent_bench.enterprise_self_hosted);
    assert!(ent_bench.all_passed);
    assert!(ent_bench
        .scenarios
        .iter()
        .any(|s| s.scenario_id == "solo_fast_cold"));
    srv.shutdown();
}

#[test]
fn model_store_tarzst_upgrade_predict_csv_modes_and_fallback_tokens() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg_dir = tmp.path().join("cfg_ms");
    fs::create_dir_all(&cfg_dir).expect("mkdir cfg_ms");
    let _env = ScopedProcessEnv::set_opt(&[
        ("CARBON_CONFIG_DIR", Some(cfg_dir.to_str().expect("utf8"))),
        ("CARBON_AUTH_TOKEN", None),
        ("DO_NOT_TRACK", Some("1")),
    ]);

    // Quantile edge cases: empty / length mismatch / NaN / constant target (`sst == 0 -> r2: None`).
    assert!(compute_residual_quantiles(&[], &[], None).is_none());
    assert!(compute_residual_quantiles(&[1.0], &[1.0, 2.0], None).is_none());
    assert!(compute_residual_quantiles(&[f64::NAN], &[1.0], None).is_none());
    let const_q = compute_residual_quantiles(&[5.0, 5.0, 5.0], &[4.5, 5.0, 5.5], None)
        .expect("constant target quantiles");
    assert!(const_q.r2.is_none());

    // Empty registry errors for `latest` and `None`.
    assert!(load_model_handle_by_spec(None, Some("latest")).is_err());
    assert!(load_model_handle_by_spec(None, None).is_err());

    let mut emu_ms_cfg = CarbonEmuConfig::for_dir(tmp.path().join("emu_ms"));
    emu_ms_cfg.force_hybrid_fanout = false;
    let mut emu = CarbonLocalLambdaEmulator::start(emu_ms_cfg).expect("start emu_ms");
    let tok = emu
        .mint_capability_token("ms_tenant", CarbonTierWire::Pro, false)
        .expect("mint token");

    let train_csv = tmp.path().join("train.csv");
    let mut train_txt = String::from("id,x1,x2,y\n");
    for i in 0..48 {
        let x1 = (i as f64) * 0.25;
        let x2 = ((i % 5) as f64) * 1.5;
        let y = 3.0 * x1 + 0.5 * x2 + 2.0;
        train_txt.push_str(&format!("row_{i},{x1:.3},{x2:.3},{y:.3}\n"));
    }
    fs::write(&train_csv, &train_txt).expect("write train.csv");

    let explicit_handle = tmp.path().join("stripped_handle.carbon.json");
    let fit_rcpt = execute_carbon_cli_run_with_ux(
        &CarbonCliRunOptions {
            endpoints: vec![emu.base_url()],
            csv_path: Some(train_csv),
            target_column: "y".to_owned(),
            synthetic_rows: 48,
            synthetic_features: 2,
            tenant_id: "ms_tenant".to_owned(),
            tier: CarbonTierWire::Pro,
            effort: Effort::Fast,
            objective: "rmse".to_owned(),
            seed: 12,
            holdout_fraction: 0.25,
            stop_after_ms: None,
            resume_after_stop: false,
            predict_rows: 3,
            auth_token: Some(tok.clone()),
            shred_after_run: false,
        },
        &CarbonCliRunUxConfig {
            target_was_explicit: true,
            persist_handle: true,
            save_model_path: Some(explicit_handle.clone()),
            cloud_name: Some("dev".to_owned()),
            embed_token: false,
            ..CarbonCliRunUxConfig::default()
        },
    )
    .expect("fit handle");

    // Load via `latest`, explicit `--run`, and default `None`; check Debug redaction.
    let (h_latest, _) = load_model_handle_by_spec(None, Some("latest")).expect("load latest");
    let (h_by_id, _) =
        load_model_handle_by_spec(None, Some(&fit_rcpt.run_id)).expect("load by run_id");
    let (h_default, _) = load_model_handle_by_spec(None, None).expect("load default latest");
    assert_eq!(h_latest.run_id, h_by_id.run_id);
    assert_eq!(h_latest.run_id, h_default.run_id);
    let dbg_str = format!("{h_latest:?}");
    assert!(dbg_str.contains("<redacted>") && !dbg_str.contains(&tok));

    // Corrupt schema refusal.
    let bad_schema_file = tmp.path().join("bad_schema.json");
    let mut bad_val: serde_json::Value =
        serde_json::from_slice(&fs::read(&explicit_handle).expect("read")).expect("json");
    bad_val["schema"] = serde_json::Value::String("wrong-schema-v0".to_owned());
    fs::write(&bad_schema_file, serde_json::to_vec(&bad_val).expect("ser")).expect("write");
    assert!(load_model_handle_by_spec(Some(&bad_schema_file), None).is_err());

    // Fallback token resolution: stripped handle (`capability_token: None`) resolves via stored cloud token
    // when registry entry is removed!
    let (stripped_h, _) =
        load_model_handle_by_spec(Some(&explicit_handle), None).expect("load stripped");
    assert!(stripped_h.capability_token.is_none());
    fs::remove_file(
        cfg_dir
            .join("runs")
            .join(format!("{}.json", stripped_h.run_id)),
    )
    .expect("remove registry copy to test cloud fallback");
    save_stored_cloud_token("dev", &emu.base_url(), &tok).expect("save stored cloud token");
    let resolved_client =
        build_client_for_handle(&stripped_h, None).expect("build client via stored cloud token");
    assert_eq!(
        resolved_client.active_capability_token().as_deref(),
        Some(tok.as_str())
    );

    // `execute_predict_csv_with_handle`: labeled CSV + `include_inputs` (quoting `,` & `"`) + `id_column` + `-` stdout + empty CSV error.
    let score_csv = tmp.path().join("score_labeled.csv");
    fs::write(
        &score_csv,
        "id,x1,x2,y\n\"item,1\",1.0,2.0,6.0\n\"quote\"\"2\",2.0,4.0,10.0\n",
    )
    .expect("write score_labeled.csv");
    let out_inc = tmp.path().join("sub/preds_with_inputs.csv");
    let pred_inc = execute_predict_csv_with_handle(
        &stripped_h,
        &score_csv,
        Some(&out_inc),
        None,
        false,
        None,
        true,
        None,
    )
    .expect("predict include_inputs");
    assert_eq!(pred_inc.rows_scored, 2);
    assert!(pred_inc.evaluation_if_labeled.is_some());
    let inc_text = fs::read_to_string(&out_inc).expect("read preds_with_inputs.csv");
    assert!(inc_text.contains("\"item,1\"") && inc_text.contains("\"quote\"\"2\""));

    let out_id = tmp.path().join("preds_id.csv");
    execute_predict_csv_with_handle(
        &stripped_h,
        &score_csv,
        Some(&out_id),
        None,
        false,
        Some("id"),
        false,
        None,
    )
    .expect("predict id_column");
    assert!(fs::read_to_string(&out_id)
        .expect("read preds_id")
        .starts_with("id,prediction\n"));

    // Missing `id_column` error & `-` stdout mode & empty CSV error.
    assert!(execute_predict_csv_with_handle(
        &stripped_h,
        &score_csv,
        Some(&out_id),
        None,
        false,
        Some("missing_col"),
        false,
        None,
    )
    .is_err());

    execute_predict_csv_with_handle(
        &stripped_h,
        &score_csv,
        Some(Path::new("-")),
        None,
        false,
        None,
        false,
        None,
    )
    .expect("predict stdout");

    let empty_csv = tmp.path().join("empty_rows.csv");
    fs::write(&empty_csv, "id,x1,x2\n").expect("write empty_rows.csv");
    assert!(execute_predict_csv_with_handle(
        &stripped_h,
        &empty_csv,
        None,
        None,
        false,
        None,
        false,
        None,
    )
    .is_err());

    emu.shutdown();

    // `execute_self_upgrade` / `execute_self_upgrade_with_options`: operator guard, 302 redirect + `up_to_date`, and `.tar.zst` SHA check!
    let (kp, pub_pem) = ephemeral_signing_keypair();
    assert!(execute_self_upgrade(Some("http://127.0.0.1:9"), None, &pub_pem, false).is_err());

    let asset_name = if CARBON_BUILD_TARGET.contains("aarch64") {
        "carbon-linux-aarch64-musl"
    } else {
        "carbon-linux-x86_64-musl"
    };
    let cur_exe = std::env::current_exe().expect("current_exe");
    let cur_exe_sha = sha256_lower_hex(&fs::read(&cur_exe).expect("read current_exe"));
    let fake_bin = b"#!/bin/sh\necho upgraded\n";
    let tar_raw = build_deterministic_single_bin_tar("carbon", fake_bin).expect("build tar");
    let zst_bytes = zstd::encode_all(std::io::Cursor::new(&tar_raw), 1).expect("zstd encode");
    let zst_sha = sha256_lower_hex(&zst_bytes);
    let zst_asset = format!("{asset_name}.tar.zst");

    let sums_up_to_date = format!("{cur_exe_sha}  {asset_name}\n");
    let sig_up_to_date = base64::engine::general_purpose::STANDARD.encode(
        kp.sign(&SystemRandom::new(), sums_up_to_date.as_bytes())
            .expect("sign")
            .as_ref(),
    );

    let bad_zst_sums = format!("{}  {asset_name}\n{zst_sha}  {zst_asset}\n", "f".repeat(64));
    let bad_zst_sig = base64::engine::general_purpose::STANDARD.encode(
        kp.sign(&SystemRandom::new(), bad_zst_sums.as_bytes())
            .expect("sign")
            .as_ref(),
    );

    let files_map: Arc<BTreeMap<String, Vec<u8>>> = Arc::new(BTreeMap::from([
        ("v1/SHA256SUMS.txt".to_owned(), sums_up_to_date.into_bytes()),
        (
            "v1/SHA256SUMS.txt.sig".to_owned(),
            sig_up_to_date.into_bytes(),
        ),
        ("v2/SHA256SUMS.txt".to_owned(), bad_zst_sums.into_bytes()),
        ("v2/SHA256SUMS.txt.sig".to_owned(), bad_zst_sig.into_bytes()),
        (format!("v2/{zst_asset}"), zst_bytes),
    ]));
    let fm = Arc::clone(&files_map);
    let app = Router::new()
        .route(
            "/redir/{*rest}",
            get(|AxumPath(rest): AxumPath<String>| async move {
                Response::builder()
                    .status(StatusCode::FOUND)
                    .header("location", format!("/v1/{rest}"))
                    .body(Body::empty())
                    .expect("302")
            }),
        )
        .route(
            "/{*key}",
            get(move |AxumPath(key): AxumPath<String>| {
                let fm = Arc::clone(&fm);
                async move {
                    match fm.get(&key) {
                        Some(bytes) => Response::builder()
                            .status(StatusCode::OK)
                            .body(Body::from(bytes.clone()))
                            .expect("200"),
                        None => Response::builder()
                            .status(StatusCode::NOT_FOUND)
                            .body(Body::from("not found"))
                            .expect("404"),
                    }
                }
            }),
        );
    let (up_addr, up_tx, up_jh) = spawn_loopback_axum("mock-upgrade", app);

    // 302 redirect + `up_to_date == true` short-circuit (never mutates test binary!).
    let up_rcpt = execute_self_upgrade_with_options(
        Some(&format!("http://{up_addr}/redir")),
        Some("carbon-cli-v1.2.0"),
        &pub_pem,
        false,
        true,
    )
    .expect("up_to_date upgrade via 302 redirect");
    assert!(up_rcpt.signature_verified && up_rcpt.up_to_date);

    // `.tar.zst` download + extraction + post-extract binary SHA-256 mismatch refusal!
    let mismatch_err = execute_self_upgrade_with_options(
        Some(&format!("http://{up_addr}/v2")),
        None,
        &pub_pem,
        false,
        true,
    )
    .expect_err("expected extracted binary SHA mismatch");
    assert!(
        mismatch_err.contains("SHA-256 mismatch"),
        "unexpected err: {mismatch_err}"
    );

    let _ = up_tx.send(());
    let _ = up_jh.join();
}
