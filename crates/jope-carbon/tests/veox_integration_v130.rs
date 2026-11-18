//! End-to-end verification of all 7 `app.veox.ai` / `jAIn` integration requirements (`v1.3.0`):
//! 1. `1,202`-row (`airfoil` scale) & `3,840`-fit-row (`parkinsons` scale) `HybridFanOut` local-engine
//!    champion refit, `> 5.9 MB` Runtime API ceiling guard, and standalone `JS` / `C++17` export;
//! 2. Unconditional auto-shred of staged `CUEN\x01` CAS blobs when `carbon fit` / `run` fails mid-flight;
//! 3. `prod` (`allow_public_free_mint: false`) `403` refusal on `/auth/anonymous` & `/auth/register`;
//! 4. Asymmetric P-256 `ctk2.*` per-user capability tokens (`app.veox.ai` keygen, dual-key rotation,
//!    cross-tenant `403`, and `carbon token-keygen` / `carbon token-mint --signing-key-file`);
//! 5. `jope.carbon.fit_receipt.v1` (`schema_version = 1`, `holdout_rows_sha256`, `release_id`);
//! 6. Standalone `--save` handle `capability_token` persistence on ephemeral Free runs + `--cloud`/`--tier`
//!    CLI flags on `export`, `shred`, `status`, `explain`, and `predict`;
//! 7. Bidirectional `CUEN\x01` + `CUDK\x01` wire compatibility between pure-Rust WASM-ready
//!    `jope-envelope` and `aws-lc-rs` `jope-carbon::CarbonEngineKek`.

use std::fs;
use std::path::Path;
use std::process::Command;

use jope_api::{
    carbon::{CarbonExecutionModeWire, CarbonExportLangWire, CarbonRunStateWire, CarbonTierWire},
    Effort,
};
use jope_carbon::{
    compute_canonical_holdout_rows_sha256, decrypt_cuen_payload, encrypt_cuen_payload, envelope,
    execute_carbon_cli_run_with_ux, execute_carbon_redteam_suite_with_tokens,
    generate_nonlinear_benchmark_rows, generate_random_u_dek, send_raw_edge_http,
    unwrap_u_dek_with_kek, wrap_u_dek_to_pubkey, CarbonCapabilityClaims, CarbonCliRunOptions,
    CarbonCliRunUxConfig, CarbonCloudCellEntry, CarbonEmuConfig, CarbonEngineKek,
    CarbonLocalLambdaEmulator, CarbonPublicCloudsManifest, CarbonThinClient,
    CARBON_FIT_RECEIPT_SCHEMA, CARBON_FIT_RECEIPT_SCHEMA_VERSION, CAS_BLOB_CUEN_SCOPE,
};
use jope_prep::cuen::CuenPayloadKind;

fn count_regular_files_recursive(dir: &Path) -> usize {
    let Ok(rd) = fs::read_dir(dir) else {
        return 0;
    };
    let mut total = 0usize;
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            total += count_regular_files_recursive(&p);
        } else if p.is_file() {
            total += 1;
        }
    }
    total
}

#[test]
fn jope_envelope_and_aws_lc_rs_engine_kek_bidirectional_interop() {
    let aws_kek = CarbonEngineKek::generate().expect("generate aws-lc-rs KEK");
    let aws_pub_sec1 =
        jope_carbon::decode_p256_pubkey_b64(&aws_kek.pubkey_b64()).expect("decode aws KEK pubkey");
    let pure_kp = envelope::ClientEnvelopeKeypair::from_scalar_bytes(&[0x2Au8; 32])
        .expect("pure-Rust P-256 keypair");

    // 1. Client (`jope-envelope`) -> Engine (`aws-lc-rs`)
    let u_dek_1 = [0x7Bu8; 32];
    let cudk_1 = envelope::wrap_u_dek_to_pubkey(
        &u_dek_1,
        &aws_pub_sec1,
        "veox_tenant_interop",
        "run_01JINTEROP001",
    )
    .expect("pure-Rust wrap_u_dek_to_pubkey");
    let unwrapped_1 = unwrap_u_dek_with_kek(
        &aws_kek,
        &cudk_1,
        "veox_tenant_interop",
        "run_01JINTEROP001",
    )
    .expect("aws-lc-rs unwrap_u_dek_with_kek");
    assert_eq!(*unwrapped_1, u_dek_1);

    let pt_1 = b"jAIn-browser-wasm-encrypted-jmat-payload";
    let cuen_1 = envelope::encrypt_cuen_payload(
        &u_dek_1,
        CuenPayloadKind::FitJmat,
        "veox_tenant_interop",
        CAS_BLOB_CUEN_SCOPE,
        pt_1,
    )
    .expect("pure-Rust encrypt_cuen_payload");
    let dec_1 = decrypt_cuen_payload(
        &unwrapped_1,
        CuenPayloadKind::FitJmat,
        "veox_tenant_interop",
        CAS_BLOB_CUEN_SCOPE,
        &cuen_1,
    )
    .expect("aws-lc-rs decrypt_cuen_payload");
    assert_eq!(dec_1.as_slice(), pt_1);

    // 2. Engine (`aws-lc-rs`) -> Client (`jope-envelope`)
    let r_dek_2 = generate_random_u_dek().expect("CSPRNG r_dek");
    let cudk_2 = wrap_u_dek_to_pubkey(
        &r_dek_2,
        pure_kp.pubkey_bytes(),
        "veox_tenant_interop",
        "run_01JINTEROP002",
    )
    .expect("aws-lc-rs wrap_u_dek_to_pubkey");
    let unwrapped_2 = envelope::unwrap_u_dek_with_keypair(
        &pure_kp,
        &cudk_2,
        "veox_tenant_interop",
        "run_01JINTEROP002",
    )
    .expect("pure-Rust unwrap_u_dek_with_keypair");
    assert_eq!(*unwrapped_2, *r_dek_2);

    let pt_2 = br#"{"predictions":[1.25,-3.5,8.75]}"#;
    let cuen_2 = encrypt_cuen_payload(
        &r_dek_2,
        CuenPayloadKind::PredictOutputs,
        "veox_tenant_interop",
        "run_01JINTEROP002",
        pt_2,
    )
    .expect("aws-lc-rs encrypt_cuen_payload");
    let dec_2 = envelope::decrypt_cuen_payload(
        &unwrapped_2,
        CuenPayloadKind::PredictOutputs,
        "veox_tenant_interop",
        "run_01JINTEROP002",
        &cuen_2,
    )
    .expect("pure-Rust decrypt_cuen_payload");
    assert_eq!(dec_2.as_slice(), pt_2);
}

#[test]
fn prod_anonymous_closed_ctk2_dual_key_rotation_and_auto_shred_on_failed_fit() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // 1. Auto-shred on failed `fit`: open `dev` emulator where `upload_prepared` succeeds on Free
    //    tier (`<= 5,000` rows), then `start_run` with `Effort::High` is rejected (`403 tier_limit`).
    //    `ShredOnDropGuard` must immediately purge both staged `CUEN\x01` blobs from `s3_cas/blobs`!
    let dev_dir = tmp.path().join("dev_cell");
    let mut dev_emu =
        CarbonLocalLambdaEmulator::start(CarbonEmuConfig::for_dir(dev_dir.clone())).expect("dev");
    let fail_opts = CarbonCliRunOptions {
        endpoints: vec![dev_emu.base_url()],
        csv_path: None,
        target_column: "y".to_owned(),
        synthetic_rows: 120,
        synthetic_features: 4,
        tenant_id: "anon_placeholder".to_owned(),
        tier: CarbonTierWire::Free,
        effort: Effort::High, // Rejected at `POST /v1/carbon/runs` AFTER `upload_prepared` PUTs blobs!
        objective: "rmse".to_owned(),
        seed: 20261117,
        holdout_fraction: 0.20,
        stop_after_ms: None,
        resume_after_stop: false,
        predict_rows: 5,
        auth_token: None,
        shred_after_run: false, // Even WITHOUT `--shred`, failed fit must auto-shred its upload!
    };
    let err = execute_carbon_cli_run_with_ux(&fail_opts, &CarbonCliRunUxConfig::default())
        .expect_err("Free + Effort::High must fail at start_run");
    assert!(err.contains("403"), "unexpected error: {err}");
    let residual_blobs = count_regular_files_recursive(&dev_dir.join("s3_cas").join("blobs"));
    assert_eq!(
        residual_blobs, 0,
        "failed fit must auto-shred staged CUEN ciphertext blobs; found {residual_blobs} leftover blob files"
    );
    dev_emu.shutdown();

    // 2. `prod` cell (`allow_public_free_mint: false`) + asymmetric P-256 `ctk2.*` `[active, previous]` keys.
    let active_app_key =
        envelope::AppSigningKey::from_scalar_bytes(&[0x31u8; 32]).expect("active key");
    let previous_app_key =
        envelope::AppSigningKey::from_scalar_bytes(&[0x32u8; 32]).expect("previous key");
    let pubkeys_csv = format!(
        "{},{}",
        active_app_key.pubkey_b64(),
        previous_app_key.pubkey_b64()
    );

    let prod_dir = tmp.path().join("prod_cell");
    let mut prod_cfg = CarbonEmuConfig::for_dir(prod_dir);
    prod_cfg.allow_public_free_mint = false;
    prod_cfg.app_auth_pubkeys_csv = Some(pubkeys_csv);
    let hmac_secret = prod_cfg.auth_secret;
    let mut prod_emu = CarbonLocalLambdaEmulator::start(prod_cfg).expect("prod");
    let base_url = prod_emu.base_url();

    // Anonymous and self-service register must both return HTTP 403 on `prod`.
    let (anon_code, anon_body) = send_raw_edge_http(
        &base_url,
        "POST",
        "/v1/carbon/auth/anonymous",
        &[],
        Some(b"{}"),
    )
    .expect("POST anonymous");
    assert_eq!(anon_code, 403);
    assert!(String::from_utf8_lossy(&anon_body).contains("anonymous_disabled"));

    let (reg_code, reg_body) = send_raw_edge_http(
        &base_url,
        "POST",
        "/v1/carbon/auth/register",
        &[],
        Some(b"{}"),
    )
    .expect("POST register");
    assert_eq!(reg_code, 403);
    assert!(String::from_utf8_lossy(&reg_body).contains("registration_disabled"));

    // Mint per-user `ctk2.*` tokens signed by BOTH `previous` and `active` app keys (zero HMAC secret).
    let now_secs = jope_carbon::now_epoch_ms() / 1_000;
    let tok_alice = previous_app_key
        .mint_token(&envelope::AppCapabilityClaims {
            tenant_id: "veox_usr_alice".to_owned(),
            tier: envelope::AppTokenTier::Pro,
            exp_epoch_secs: now_secs + 3_600,
            admin: false,
            iss: Some("app.veox.ai".to_owned()),
            kid: Some("veox-prev".to_owned()),
        })
        .expect("mint alice ctk2");
    let tok_bob = active_app_key
        .mint_token(&envelope::AppCapabilityClaims {
            tenant_id: "veox_usr_bob".to_owned(),
            tier: envelope::AppTokenTier::Free,
            exp_epoch_secs: now_secs + 3_600,
            admin: false,
            iss: Some("app.veox.ai".to_owned()),
            kid: Some("veox-active".to_owned()),
        })
        .expect("mint bob ctk2");

    assert_eq!(
        CarbonCapabilityClaims::decode_unverified(&tok_alice)
            .expect("decode ctk2")
            .tenant_id,
        "veox_usr_alice"
    );

    // Run full zero-knowledge fit + predict for Alice (`Pro` `ctk2.*` on retiring key), keeping run unshredded briefly to test cross-tenant isolation.
    let alice_receipt = execute_carbon_cli_run_with_ux(
        &CarbonCliRunOptions {
            endpoints: vec![base_url.clone()],
            csv_path: None,
            target_column: "y".to_owned(),
            synthetic_rows: 160,
            synthetic_features: 5,
            tenant_id: "veox_usr_alice".to_owned(),
            tier: CarbonTierWire::Pro,
            effort: Effort::Fast,
            objective: "rmse".to_owned(),
            seed: 20261117,
            holdout_fraction: 0.20,
            stop_after_ms: None,
            resume_after_stop: false,
            predict_rows: 4,
            auth_token: Some(tok_alice.clone()),
            shred_after_run: false,
        },
        &CarbonCliRunUxConfig::default(),
    )
    .expect("alice ctk2 run");
    assert_eq!(alice_receipt.schema, CARBON_FIT_RECEIPT_SCHEMA);
    assert_eq!(
        alice_receipt.schema_version,
        CARBON_FIT_RECEIPT_SCHEMA_VERSION
    );
    assert_eq!(alice_receipt.final_state, CarbonRunStateWire::Completed);

    // Bob's valid `ctk2.*` token must get 403 `tenant_mismatch` if attempting to read or shred Alice's run.
    let (cross_code, cross_body) = send_raw_edge_http(
        &base_url,
        "GET",
        &format!(
            "/v1/carbon/runs/{}?tenant_id=veox_usr_alice",
            alice_receipt.run_id
        ),
        &[("x-carbon-capability-token", &format!("Bearer {tok_bob}"))],
        None,
    )
    .expect("cross-tenant GET");
    assert_eq!(cross_code, 403);
    assert!(String::from_utf8_lossy(&cross_body).contains("tenant_mismatch"));

    // Cleanly shred Alice's run using Alice's `ctk2.*` token.
    let (alice_client, _) = CarbonThinClient::connect_fastest(vec![base_url.clone()]).unwrap();
    let shred_resp = alice_client
        .with_capability_token(tok_alice)
        .shred_run("veox_usr_alice", &alice_receipt.run_id)
        .expect("alice shred");
    assert!(shred_resp.shredded);

    // Red-team suite with pre-minted Free tokens must pass all 17 vectors on anonymous-closed `prod`!
    let verifier = jope_carbon::CarbonAuthVerifier::from_secret(hmac_secret);
    let rt_a = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "rt_prod_free_a".to_owned(),
            tier: CarbonTierWire::Free,
            exp_epoch_secs: now_secs + 900,
            admin: false,
        })
        .unwrap();
    let rt_b = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "rt_prod_free_b".to_owned(),
            tier: CarbonTierWire::Free,
            exp_epoch_secs: now_secs + 900,
            admin: false,
        })
        .unwrap();
    let rt_report = execute_carbon_redteam_suite_with_tokens(&base_url, Some((rt_a, rt_b)))
        .expect("redteam on closed prod");
    assert!(
        rt_report.all_passed,
        "all 17 redteam vectors must pass on anonymous-closed prod: {:?}",
        rt_report.vectors
    );

    prod_emu.shutdown();
}

#[test]
fn large_table_airfoil_and_parkinsons_scale_hybrid_fanout_and_cli_save_export_shred() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut cfg = CarbonEmuConfig::for_dir(tmp.path().join("scale_cell"));
    cfg.force_hybrid_fanout = true;
    let mut emu = CarbonLocalLambdaEmulator::start(cfg).expect("start scale_cell");
    let base_url = emu.base_url();

    // 1. Parkinsons-scale table (`2,400` total rows -> `1,920` fit rows > `1,500` fast-pool threshold)
    //    running over `HybridFanOut` with local-engine Rung-2 champion refit + JS/C++17 export!
    let scale_receipt = execute_carbon_cli_run_with_ux(
        &CarbonCliRunOptions {
            endpoints: vec![base_url.clone()],
            csv_path: None,
            target_column: "y".to_owned(),
            synthetic_rows: 1_900,
            synthetic_features: 6,
            tenant_id: "anon_scale".to_owned(),
            tier: CarbonTierWire::Free,
            effort: Effort::Fast,
            objective: "rmse".to_owned(),
            seed: 20261117,
            holdout_fraction: 0.20,
            stop_after_ms: None,
            resume_after_stop: false,
            predict_rows: 8,
            auth_token: None,
            shred_after_run: true,
        },
        &CarbonCliRunUxConfig::default(),
    )
    .expect("2,400-row HybridFanOut fast run must complete with sealed champion");
    assert_eq!(scale_receipt.final_state, CarbonRunStateWire::Completed);
    assert_eq!(
        scale_receipt.execution_mode,
        CarbonExecutionModeWire::HybridFanOut
    );
    assert!(scale_receipt.e2e_predict_holdout_rmse.is_some());
    assert_eq!(scale_receipt.holdout_rows_sha256.len(), 64);
    assert_eq!(scale_receipt.holdout_jmat_sha256.len(), 64);

    // 2. CLI subprocess contract test:
    //    - `carbon token-keygen` + `carbon token-mint --signing-key-file`
    //    - `carbon fit <CSV> --save <PATH> --format json --cloud dev --free`
    //    - Verify `holdout_rows_sha256` matches `compute_canonical_holdout_rows_sha256`
    //    - Move to a DIFFERENT `CARBON_CONFIG_DIR` (no `~/.config/jopedime/runs/<id>.json`!) and prove
    //      `carbon export --model <PATH> --lang js --cloud dev --tier free`,
    //      `carbon export --model <PATH> --lang cpp --cloud dev --tier free`, and
    //      `carbon shred --model <PATH> --cloud dev --tier free --json` ALL succeed!
    let carbon_bin = env!("CARGO_BIN_EXE_carbon");
    let key_pem = tmp.path().join("veox-signing.pem");
    let keygen_out = Command::new(carbon_bin)
        .args([
            "token-keygen",
            "--out",
            key_pem.to_str().unwrap(),
            "--kid",
            "veox-k1",
        ])
        .output()
        .expect("carbon token-keygen");
    assert!(
        keygen_out.status.success(),
        "token-keygen failed: {}",
        String::from_utf8_lossy(&keygen_out.stderr)
    );
    let keygen_json: serde_json::Value =
        serde_json::from_slice(&keygen_out.stdout).expect("keygen json");
    assert!(
        keygen_json["public_key_sec1_b64"]
            .as_str()
            .unwrap_or("")
            .len()
            > 40
    );

    let mint_out = Command::new(carbon_bin)
        .args([
            "token-mint",
            "--signing-key-file",
            key_pem.to_str().unwrap(),
            "--tenant",
            "veox_cli_user",
            "--tier",
            "pro",
            "--ttl-secs",
            "1800",
            "--kid",
            "veox-k1",
        ])
        .output()
        .expect("carbon token-mint --signing-key-file");
    assert!(
        mint_out.status.success(),
        "token-mint failed: {}",
        String::from_utf8_lossy(&mint_out.stderr)
    );
    let minted_ctk2 = String::from_utf8_lossy(&mint_out.stdout).trim().to_owned();
    assert!(minted_ctk2.starts_with("ctk2."));

    // Write local `carbon-clouds.json` pointing `dev` to `emu`
    let (probe_client, probed) =
        CarbonThinClient::connect_fastest(vec![base_url.clone()]).expect("probe");
    let _ = probe_client;
    let kek_b64 = probed[0].engine_kek_pubkey_b64.clone();
    let mut clouds_map = std::collections::BTreeMap::new();
    clouds_map.insert(
        "dev".to_owned(),
        CarbonCloudCellEntry {
            environment: "dev".to_owned(),
            region: "us-east-2".to_owned(),
            public_endpoint: base_url.clone(),
            engine_kek_pubkey_b64: kek_b64,
            release_id: Some("rel20261117".to_owned()),
            regions: Vec::new(),
        },
    );
    let clouds_path = tmp.path().join("carbon-clouds.json");
    fs::write(
        &clouds_path,
        serde_json::to_vec_pretty(&CarbonPublicCloudsManifest {
            schema: "jope-carbon-clouds-v2".to_owned(),
            generated_epoch_ms: 1_795_000_000_000,
            clouds: clouds_map,
        })
        .unwrap(),
    )
    .unwrap();

    // Write a 320-row CSV (`fuel-economy` scale) and compute expected canonical `holdout_rows_sha256`
    let (hdrs, rows, tgt) = generate_nonlinear_benchmark_rows(320, 6, 20261117);
    let csv_path = tmp.path().join("fuel_like.csv");
    let mut csv_txt = format!("{}\n", hdrs.join(","));
    for r in &rows {
        let cols: Vec<String> = r.iter().map(|v| format!("{v:.8}")).collect();
        csv_txt.push_str(&cols.join(","));
        csv_txt.push('\n');
    }
    fs::write(&csv_path, csv_txt).unwrap();

    let cfg_dir_train = tmp.path().join("cfg_train");
    let cfg_dir_isolated = tmp.path().join("cfg_isolated_worker");
    fs::create_dir_all(&cfg_dir_train).unwrap();
    fs::create_dir_all(&cfg_dir_isolated).unwrap();
    let saved_handle_path = tmp.path().join("saved_free_model.carbon.json");

    let fit_out = Command::new(carbon_bin)
        .env("CARBON_CONFIG_DIR", &cfg_dir_train)
        .args([
            "fit",
            csv_path.to_str().unwrap(),
            "--target",
            &tgt,
            "--cloud",
            "dev",
            "--clouds-manifest",
            clouds_path.to_str().unwrap(),
            "--free",
            "--effort",
            "fast",
            "--seed",
            "20261117",
            "--save",
            saved_handle_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("carbon fit");
    assert!(
        fit_out.status.success(),
        "carbon fit failed: {}",
        String::from_utf8_lossy(&fit_out.stderr)
    );
    let fit_json: serde_json::Value =
        serde_json::from_slice(&fit_out.stdout).expect("parse fit json");
    assert_eq!(fit_json["schema"], CARBON_FIT_RECEIPT_SCHEMA);
    assert_eq!(
        fit_json["schema_version"],
        CARBON_FIT_RECEIPT_SCHEMA_VERSION
    );
    assert_eq!(fit_json["split_mode"], "random");
    assert_eq!(fit_json["seed"], 20261117);

    // Recompute canonical holdout digest directly from the CSV and confirm bit-for-bit equality!
    let raw_tbl = jope_carbon::load_delimited_table(&csv_path, None, true).unwrap();
    let enc_set =
        jope_carbon::prepare_encoded_training_set(&raw_tbl, Some(&tgt), 20261117, 0.20, 200)
            .unwrap();
    let expected_ho_sha = compute_canonical_holdout_rows_sha256(
        enc_set.prepared.split_plan.holdout_rows(),
        &enc_set.holdout_x,
        &enc_set.holdout_y,
    );
    assert_eq!(
        fit_json["holdout_rows_sha256"].as_str(),
        Some(expected_ho_sha.as_str())
    );

    // Now run `export` (JS & C++17) and `shred` from `cfg_isolated_worker` (NO home registry copy!)
    // AND passing `--cloud dev --tier free --clouds-manifest ...`!
    for lang in [
        CarbonExportLangWire::Js,
        CarbonExportLangWire::Cpp,
        CarbonExportLangWire::Python,
    ] {
        let lang_str = match lang {
            CarbonExportLangWire::Js => "js",
            CarbonExportLangWire::Cpp => "cpp",
            _ => "python",
        };
        let exp_out = Command::new(carbon_bin)
            .env("CARBON_CONFIG_DIR", &cfg_dir_isolated)
            .args([
                "export",
                "--model",
                saved_handle_path.to_str().unwrap(),
                "--lang",
                lang_str,
                "--cloud",
                "dev",
                "--tier",
                "free",
                "--clouds-manifest",
                clouds_path.to_str().unwrap(),
            ])
            .output()
            .expect("carbon export");
        assert!(
            exp_out.status.success(),
            "carbon export --lang {lang_str} failed in isolated config dir: {}",
            String::from_utf8_lossy(&exp_out.stderr)
        );
        assert!(!exp_out.stdout.is_empty());
    }

    let shred_out = Command::new(carbon_bin)
        .env("CARBON_CONFIG_DIR", &cfg_dir_isolated)
        .args([
            "shred",
            "--model",
            saved_handle_path.to_str().unwrap(),
            "--cloud",
            "dev",
            "--tier",
            "free",
            "--clouds-manifest",
            clouds_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("carbon shred");
    assert!(
        shred_out.status.success(),
        "carbon shred failed in isolated config dir: {}",
        String::from_utf8_lossy(&shred_out.stderr)
    );
    let shred_json: serde_json::Value =
        serde_json::from_slice(&shred_out.stdout).expect("shred json");
    assert_eq!(shred_json["remote_shredded"], true);
    assert!(!saved_handle_path.exists());

    emu.shutdown();
}

/// 4. Universal Binary Categorical Target Pre-Processing (`yes`/`no`, `true`/`false`, `<=50K`/`>50K`,
///    1-class constant refusal, and `>= 3`-class multiclass refusal) across `carbon inspect`,
///    `carbon fit --save`, `carbon predict --out`, `carbon export --lang js/cpp`, and `carbon shred`.
#[test]
fn e2e_binary_categorical_targets_yes_no_true_false_inspect_fit_predict_export_and_shred() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let emu_dir = tmp.path().join("emu_bin");
    let emu_cfg = CarbonEmuConfig::for_dir(emu_dir);
    let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start binary target emu");
    let base_url = emu.base_url();

    let cfg_dir = tmp.path().join("cfg_bin");
    fs::create_dir_all(&cfg_dir).expect("mkdir cfg_bin");
    let carbon_bin = env!("CARGO_BIN_EXE_carbon");

    // A. `yes`/`no` (mixed case + missing row + boolean/categorical/numeric features)
    let yn_train_csv = tmp.path().join("churn_yes_no.csv");
    let yn_test_csv = tmp.path().join("churn_yes_no_test.csv");
    let mut train_buf = String::from("tenure_months,monthly_usd,autopay,segment,churned\n");
    let mut test_buf = String::from("tenure_months,monthly_usd,autopay,segment,churned\n");
    for i in 0..180_usize {
        let tenure = (i % 48) as f64 + 1.0;
        let monthly = 35.0 + ((i * 7) % 85) as f64;
        let autopay = if i % 3 == 0 { "true" } else { "false" };
        let seg = match i % 3 {
            0 => "smb",
            1 => "midmarket",
            _ => "enterprise",
        };
        let score = 0.06 * (monthly - 65.0) - 0.09 * (tenure - 18.0)
            + if autopay == "false" { 1.1 } else { -0.8 };
        let label = if i == 17 {
            "NA"
        } else if score > 0.0 {
            if i % 2 == 0 {
                "yes"
            } else {
                "Yes"
            }
        } else if i % 2 == 0 {
            "no"
        } else {
            "NO"
        };
        let line = format!("{tenure:.1},${monthly:.2},{autopay},{seg},{label}\n");
        if i < 150 {
            train_buf.push_str(&line);
        } else {
            test_buf.push_str(&line);
        }
    }
    fs::write(&yn_train_csv, train_buf.as_bytes()).expect("write yn train");
    fs::write(&yn_test_csv, test_buf.as_bytes()).expect("write yn test");

    // `carbon inspect` without `--target` must auto-preview rightmost binary column `churned`!
    let insp_out = Command::new(carbon_bin)
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .args([
            "inspect",
            yn_train_csv.to_str().unwrap(),
            "--tier",
            "free",
            "--json",
        ])
        .output()
        .expect("carbon inspect binary csv");
    assert!(
        insp_out.status.success(),
        "carbon inspect failed on yes/no target: {}",
        String::from_utf8_lossy(&insp_out.stderr)
    );
    let insp_json: serde_json::Value =
        serde_json::from_slice(&insp_out.stdout).expect("inspect json");
    assert_eq!(insp_json["target_column"], "churned");
    assert_eq!(insp_json["valid_target_rows"], 149);
    assert_eq!(insp_json["dropped_missing_target_rows"], 1);
    let tgt_col_prof = insp_json["columns"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "churned")
        .expect("churned column profile");
    assert_eq!(tgt_col_prof["detected_type"], "binary_categorical");

    // `carbon fit --save` on `yes`/`no` target
    let yn_handle = tmp.path().join("churn_yn.carbon.json");
    let fit_out = Command::new(carbon_bin)
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .args([
            "fit",
            yn_train_csv.to_str().unwrap(),
            "--target",
            "churned",
            "--endpoint",
            &base_url,
            "--free",
            "--effort",
            "fast",
            "--seed",
            "20261118",
            "--save",
            yn_handle.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("carbon fit yes/no");
    assert!(
        fit_out.status.success(),
        "carbon fit failed on yes/no binary target: {}",
        String::from_utf8_lossy(&fit_out.stderr)
    );
    let fit_json: serde_json::Value = serde_json::from_slice(&fit_out.stdout).expect("fit json");
    assert_eq!(fit_json["final_state"], "completed");
    let bin_summary = &fit_json["binary_target"];
    assert!(bin_summary["negative_label"]
        .as_str()
        .unwrap()
        .eq_ignore_ascii_case("no"));
    assert!(bin_summary["positive_label"]
        .as_str()
        .unwrap()
        .eq_ignore_ascii_case("yes"));
    let roc_auc = bin_summary["holdout_roc_auc"]
        .as_f64()
        .expect("holdout_roc_auc");
    assert!(
        roc_auc > 0.85,
        "expected strong binary holdout ROC-AUC (> 0.85), got {roc_auc}"
    );

    // `carbon predict` on labeled `yes`/`no` test CSV -> populates `evaluation_if_labeled` and emits `predicted_label`!
    let pred_csv = tmp.path().join("churn_preds.csv");
    let pred_out = Command::new(carbon_bin)
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .args([
            "predict",
            "--model",
            yn_handle.to_str().unwrap(),
            yn_test_csv.to_str().unwrap(),
            "--out",
            pred_csv.to_str().unwrap(),
            "--format",
            "json",
        ])
        .output()
        .expect("carbon predict binary");
    assert!(
        pred_out.status.success(),
        "carbon predict failed on yes/no test csv: {}",
        String::from_utf8_lossy(&pred_out.stderr)
    );
    let pred_json: serde_json::Value = serde_json::from_slice(&pred_out.stdout).expect("pred json");
    assert!(
        pred_json["evaluation_if_labeled"].is_object(),
        "expected evaluation_if_labeled on yes/no test CSV, got: {pred_json}"
    );
    let pred_csv_text = fs::read_to_string(&pred_csv).expect("read pred_csv");
    let first_line = pred_csv_text.lines().next().unwrap_or("");
    assert_eq!(
        first_line,
        "prediction,predicted_probability,predicted_label"
    );
    for row_line in pred_csv_text.lines().skip(1) {
        let cols: Vec<&str> = row_line.split(',').collect();
        assert_eq!(cols.len(), 3);
        let prob: f64 = cols[1].parse().expect("probability f64");
        assert!((0.0..=1.0).contains(&prob));
        assert!(cols[2].eq_ignore_ascii_case("yes") || cols[2].eq_ignore_ascii_case("no"));
    }

    // Export standalone JS & C++17 and shred
    for lang in ["js", "cpp"] {
        let exp = Command::new(carbon_bin)
            .env("CARBON_CONFIG_DIR", &cfg_dir)
            .args([
                "export",
                "--model",
                yn_handle.to_str().unwrap(),
                "--lang",
                lang,
            ])
            .output()
            .expect("export binary model");
        assert!(
            exp.status.success(),
            "export {lang} failed: {}",
            String::from_utf8_lossy(&exp.stderr)
        );
    }
    let shred = Command::new(carbon_bin)
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .args(["shred", "--model", yn_handle.to_str().unwrap(), "--json"])
        .output()
        .expect("shred binary model");
    assert!(shred.status.success());

    // B. `true`/`false` JSONL + `<=50K`/`>50K` offline `--dry-run` + `>=3`-class multiclass refusal
    let tf_jsonl = tmp.path().join("events_tf.jsonl");
    let mut jsonl_buf = String::new();
    for i in 0..40_usize {
        let x1 = (i as f64) * 0.5;
        let x2 = ((i % 5) as f64) - 2.0;
        let flag = x1 + x2 > 9.0;
        jsonl_buf.push_str(&format!(
            "{{\"x1\":{x1:.2},\"x2\":{x2:.2},\"converted\":{flag}}}\n"
        ));
    }
    fs::write(&tf_jsonl, jsonl_buf.as_bytes()).expect("write tf jsonl");
    let dry_tf = Command::new(carbon_bin)
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .args([
            "fit",
            tf_jsonl.to_str().unwrap(),
            "--target",
            "converted",
            "--dry-run",
            "--format",
            "json",
        ])
        .output()
        .expect("dry-run true/false jsonl");
    assert!(
        dry_tf.status.success(),
        "dry-run true/false jsonl failed: {}",
        String::from_utf8_lossy(&dry_tf.stderr)
    );
    let dry_tf_json: serde_json::Value =
        serde_json::from_slice(&dry_tf.stdout).expect("dry-run json");
    assert_eq!(dry_tf_json["binary_target"]["negative_label"], "false");
    assert_eq!(dry_tf_json["binary_target"]["positive_label"], "true");

    let multi_csv = tmp.path().join("iris_3class.csv");
    let mut multi_buf = String::from("f1,f2,species\n");
    for i in 0..30_usize {
        let sp = match i % 3 {
            0 => "setosa",
            1 => "versicolor",
            _ => "virginica",
        };
        multi_buf.push_str(&format!("{}.0,{}.5,{sp}\n", i, i % 4));
    }
    fs::write(&multi_csv, multi_buf.as_bytes()).expect("write 3class csv");
    let multi_err = Command::new(carbon_bin)
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .args([
            "fit",
            multi_csv.to_str().unwrap(),
            "--target",
            "species",
            "--dry-run",
        ])
        .output()
        .expect("dry-run 3class");
    assert!(!multi_err.status.success());
    let multi_stderr = String::from_utf8_lossy(&multi_err.stderr);
    assert!(
        multi_stderr.contains("3 distinct categorical classes")
            && multi_stderr.contains("multiclass"),
        "expected clear multiclass error, got: {multi_stderr}"
    );

    emu.shutdown();
}

#[test]
fn synthesize_template_edge_tag_and_pinned_veox_pubkeys_validate_cleanly() {
    let base_tpl =
        jope_carbon::cli::deploy::synthesize_cell_template_for_release("1b9f6396cb3ff81b")
            .expect("base template");
    assert!(base_tpl.contains("\"EdgeVersion1b9f6396cb3ff81b\""));

    let tagged_tpl = jope_carbon::cli::deploy::synthesize_cell_template_for_release_with_edge_tag(
        "1b9f6396cb3ff81b",
        Some("a1b2c3d4"),
    )
    .expect("tagged template");
    assert!(tagged_tpl.contains("\"EdgeVersion1b9f6396cb3ff81ba1b2c3d4\""));
    assert!(tagged_tpl.contains("\"EngineVersion1b9f6396cb3ff81b\""));
    assert!(tagged_tpl.contains("\"EvalVersion1b9f6396cb3ff81b\""));

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for rel in [
        "deploy/carbon/pubkeys/veox-app-dev.pub",
        "deploy/carbon/pubkeys/veox-app-prod.pub",
    ] {
        let raw = fs::read_to_string(repo_root.join(rel)).expect("read pinned veox pubkey");
        let pub_b64 = raw.trim();
        let decoded = envelope::decode_p256_pubkey_b64(pub_b64).expect("valid 65-byte SEC1 pubkey");
        assert_eq!(decoded[0], 0x04);
    }
}

#[test]
fn v131_max_performance_pinned_handle_and_large_comparators() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pub_b64 = fs::read_to_string(repo_root.join("deploy/carbon/pubkeys/veox-app-prod.pub"))
        .expect("read prod pubkey")
        .trim()
        .to_owned();

    // 1. Zero-ping pinned-handle client construction + transport/key validation
    let pinned_client = jope_carbon::CarbonThinClient::from_pinned_handle(
        "https://us-east-2.carbon.neverhuman.org/".to_owned(),
        &pub_b64,
    )
    .expect("from_pinned_handle valid HTTPS + P-256 KEK");
    assert_eq!(
        pinned_client.active_endpoint(),
        "https://us-east-2.carbon.neverhuman.org"
    );
    assert!(pinned_client.has_zero_knowledge_kek());
    assert!(
        jope_carbon::CarbonThinClient::from_pinned_handle(
            "http://us-east-2.carbon.neverhuman.org".to_owned(),
            &pub_b64
        )
        .is_err(),
        "non-loopback HTTP must be rejected"
    );
    assert!(
        jope_carbon::CarbonThinClient::from_pinned_handle(
            "https://us-east-2.carbon.neverhuman.org".to_owned(),
            "not-a-valid-pubkey"
        )
        .is_err(),
        "malformed KEK must be rejected"
    );

    // 2. Medium-large fit split (`4,800` fit rows > `3,500`, e.g. `parkinsons-telemonitoring` scale)
    //    emits both `ridge_l2` and `lightgbm_style` comparators cleanly.
    let fit_x: Vec<Vec<f64>> = (0..4_800)
        .map(|i| {
            let u = (i as f64) * 0.01;
            vec![u.sin(), u.cos(), (u * 0.5).tanh(), (i % 17) as f64 * 0.1]
        })
        .collect();
    let fit_y: Vec<f64> = fit_x
        .iter()
        .map(|r| 2.5 * r[0] - 1.4 * r[1] + 0.8 * r[2] * r[3])
        .collect();
    let holdout_x: Vec<Vec<f64>> = (4_800..5_600)
        .map(|i| {
            let u = (i as f64) * 0.01;
            vec![u.sin(), u.cos(), (u * 0.5).tanh(), (i % 17) as f64 * 0.1]
        })
        .collect();
    let holdout_y: Vec<f64> = holdout_x
        .iter()
        .map(|r| 2.5 * r[0] - 1.4 * r[1] + 0.8 * r[2] * r[3])
        .collect();
    let scores = jope_carbon::cli::bench_suite::compute_quick_run_comparators(
        &fit_x, &fit_y, &holdout_x, &holdout_y, 42,
    );
    assert!(scores.iter().any(|c| c.key == "ridge_l2"));
    assert!(scores.iter().any(|c| c.key == "lightgbm_style"));
}
