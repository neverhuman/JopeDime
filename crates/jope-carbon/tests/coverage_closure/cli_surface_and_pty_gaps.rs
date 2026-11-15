//! Section 14A CLI surface & pseudo-TTY coverage closure (`CLI-20` `carbon runs`,
//! `CLI-24` `carbon fitness list`, `CLI-14` `carbon doctor`, `CLI-45` `carbon engine`,
//! `CLI-46` `carbon eval`, `CLI-34` `carbon redteam`, `CLI-35` `carbon token-mint`,
//! `CLI-10` `carbon register`, `CLI-11` `carbon login`, `CLI-29` `carbon export`,
//! `CLI-33` `carbon bench`).

use std::fs;
use std::path::Path;
use std::process::Command;

use jope_api::carbon::CarbonTierWire;
use jope_carbon::{
    cli::model_store::CARBON_MODEL_HANDLE_SCHEMA, cli::tabular::CarbonTabularEncoder,
    persist_model_handle_with_options, CarbonAuthVerifier, CarbonCapabilityClaims, CarbonEmuConfig,
    CarbonEngineKek, CarbonLocalLambdaEmulator, CarbonSavedModelHandle,
};

use super::ScopedProcessEnv;

fn write_test_clouds_manifest(path: &Path, endpoint: &str, kek_b64: &str) {
    let manifest = serde_json::json!({
        "schema": "jope-carbon-clouds-v1",
        "generated_epoch_ms": 1_780_000_000_000u64,
        "clouds": {
            "prod": {
                "environment": "prod",
                "region": "us-east-2",
                "public_endpoint": endpoint,
                "engine_kek_pubkey_b64": kek_b64,
                "regions": [{
                    "region": "us-east-2",
                    "public_endpoint": endpoint,
                    "engine_kek_pubkey_b64": kek_b64
                }]
            },
            "dev": {
                "environment": "dev",
                "region": "us-east-2",
                "public_endpoint": endpoint,
                "engine_kek_pubkey_b64": kek_b64,
                "regions": [{
                    "region": "us-east-2",
                    "public_endpoint": endpoint,
                    "engine_kek_pubkey_b64": kek_b64
                }]
            }
        }
    });
    fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).expect("write clouds manifest");
}

#[test]
fn cli_pty_tables_runs_fitness_and_doctor_online_and_error_modes() {
    let root = tempfile::tempdir().expect("tempdir");
    let bin = env!("CARGO_BIN_EXE_carbon");
    let cfg_dir = root.path().join("jope_cfg");
    fs::create_dir_all(&cfg_dir).expect("create cfg dir");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&cfg_dir, fs::Permissions::from_mode(0o700)).expect("chmod 0700");
    }

    let mut emu_cfg = CarbonEmuConfig::for_dir(root.path().join("emu_cli_pty"));
    emu_cfg.force_hybrid_fanout = false;
    let kek_b64 = CarbonEngineKek::from_scalar_bytes(&emu_cfg.engine_kek_scalar)
        .expect("emu kek")
        .pubkey_b64();
    let auth_secret = emu_cfg.auth_secret;
    let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start emu");
    let manifest_path = root.path().join("clouds.json");
    write_test_clouds_manifest(&manifest_path, &emu.base_url(), &kek_b64);

    // 1. `CLI-20` `carbon runs` relative CARBON_CONFIG_DIR error path (exit 1)
    let bad_runs = Command::new(bin)
        .arg("runs")
        .env("CARBON_CONFIG_DIR", "relative/not-allowed")
        .output()
        .expect("runs relative config dir");
    assert_eq!(bad_runs.status.code(), Some(1));

    // 2. Pseudo-TTY human-table rendering via `/usr/bin/script` (`CLI-20`, `CLI-24`, `CLI-14`)
    if Path::new("/usr/bin/script").is_file() {
        // 2a. Empty `carbon runs` under a TTY -> prints empty guidance message
        let pty_runs_empty = Command::new("/usr/bin/script")
            .args(["-qec", &format!("'{bin}' runs"), "/dev/null"])
            .env("CARBON_CONFIG_DIR", &cfg_dir)
            .output()
            .expect("pty carbon runs empty");
        assert!(
            pty_runs_empty.status.success(),
            "pty runs empty failed: {}",
            String::from_utf8_lossy(&pty_runs_empty.stderr)
        );
        let empty_stdout = String::from_utf8_lossy(&pty_runs_empty.stdout);
        assert!(
            empty_stdout.contains("No saved Carbon runs found"),
            "{empty_stdout}"
        );

        // Persist one saved handle into `cfg_dir` so `carbon runs` renders the non-empty table too.
        {
            let _env = ScopedProcessEnv::set(&[(
                "CARBON_CONFIG_DIR",
                cfg_dir.to_str().expect("utf8 cfg_dir"),
            )]);
            let handle = CarbonSavedModelHandle {
                schema: CARBON_MODEL_HANDLE_SCHEMA.to_owned(),
                created_epoch_ms: 1_780_000_010_000,
                run_id: "run_01JPTYTABLERUN000000000000".to_owned(),
                dataset_id: "ds_0123456789abcdef01234567".to_owned(),
                tenant_id: "pty_tenant".to_owned(),
                tier: CarbonTierWire::Free,
                cloud: Some("prod".to_owned()),
                endpoint: emu.base_url(),
                region: "us-east-2".to_owned(),
                engine_kek_pubkey_b64: Some(kek_b64.clone()),
                encrypted_cuen: true,
                u_dek_b64: Some("MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=".to_owned()),
                capability_token: None,
                encoder: CarbonTabularEncoder {
                    raw_feature_columns: vec!["x1".to_owned(), "x2".to_owned()],
                    dropped_columns: Vec::new(),
                    transforms: Vec::new(),
                    encoded_feature_names: vec!["x1".to_owned(), "x2".to_owned()],
                    target_column: "y".to_owned(),
                    delimiter: ',',
                    had_header: true,
                    extra_na_tokens: Vec::new(),
                    inf_as_nan: false,
                },
                fit_rows: 80,
                holdout_rows: 20,
                holdout_rmse: Some(0.123),
                holdout_mae: Some(0.095),
                holdout_r2: Some(0.98765),
                objective_name: Some("rmse".to_owned()),
                objective_direction: Some("minimize".to_owned()),
                feature_importances: Vec::new(),
            };
            persist_model_handle_with_options(&handle, None, false).expect("persist pty handle");
        }

        let pty_runs_nonempty = Command::new("/usr/bin/script")
            .args(["-qec", &format!("'{bin}' runs"), "/dev/null"])
            .env("CARBON_CONFIG_DIR", &cfg_dir)
            .output()
            .expect("pty carbon runs nonempty");
        assert!(pty_runs_nonempty.status.success());
        let nonempty_stdout = String::from_utf8_lossy(&pty_runs_nonempty.stdout);
        assert!(nonempty_stdout.contains("RUN_ID") && nonempty_stdout.contains("0.98765"));

        // 2b. `CLI-24` `carbon fitness list` under a TTY -> prints human catalog table
        let pty_fitness = Command::new("/usr/bin/script")
            .args(["-qec", &format!("'{bin}' fitness list"), "/dev/null"])
            .env("CARBON_CONFIG_DIR", &cfg_dir)
            .output()
            .expect("pty carbon fitness list");
        assert!(pty_fitness.status.success());
        let fit_stdout = String::from_utf8_lossy(&pty_fitness.stdout);
        assert!(fit_stdout.contains("OBJECTIVE") && fit_stdout.contains("DIRECTION"));

        // 2c. `CLI-14` `carbon doctor` online (`--cloud all`) under a TTY -> hits all `[doctor] PASS` stderr branches
        let pty_doctor_ok = Command::new("/usr/bin/script")
            .args([
                "-qec",
                &format!(
                    "'{bin}' doctor --cloud all --clouds-manifest '{}'",
                    manifest_path.display()
                ),
                "/dev/null",
            ])
            .env("CARBON_CONFIG_DIR", &cfg_dir)
            .output()
            .expect("pty carbon doctor online");
        assert!(pty_doctor_ok.status.success());
        let doc_out = String::from_utf8_lossy(&pty_doctor_ok.stdout);
        assert!(doc_out.contains("HEALTHY") || doc_out.contains("\"healthy\": true"));
    }

    // 3. `CLI-14` `carbon doctor` error & degraded branches:
    // - unsupported `--cloud bogus` -> exit 2
    let doc_bad_cloud = Command::new(bin)
        .args(["doctor", "--cloud", "bogus"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("doctor bad cloud");
    assert_eq!(doc_bad_cloud.status.code(), Some(2));

    // - corrupt `carbon-credentials.json` + insecure 0644 permissions + unreachable cloud -> exit 1
    let bad_cfg_dir = root.path().join("bad_doctor_cfg");
    fs::create_dir_all(&bad_cfg_dir).unwrap();
    let cred_file = bad_cfg_dir.join("carbon-credentials.json");
    fs::write(&cred_file, b"{corrupt-credentials").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&cred_file, fs::Permissions::from_mode(0o644)).unwrap();
    }
    let dead_manifest = root.path().join("dead_clouds.json");
    write_test_clouds_manifest(&dead_manifest, "http://127.0.0.1:9", &kek_b64);
    let doc_degraded = Command::new(bin)
        .args([
            "doctor",
            "--cloud",
            "prod",
            "--clouds-manifest",
            dead_manifest.to_str().unwrap(),
        ])
        .env("CARBON_CONFIG_DIR", &bad_cfg_dir)
        .output()
        .expect("doctor degraded");
    assert_eq!(doc_degraded.status.code(), Some(1));

    // 4. `CLI-10` `carbon register` & `CLI-11` `carbon login` validation and online branches:
    let reg_ok = Command::new(bin)
        .args([
            "register",
            "--cloud",
            "prod",
            "--clouds-manifest",
            manifest_path.to_str().unwrap(),
            "--label",
            "ci-gap-test",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("register ok");
    assert!(
        reg_ok.status.success(),
        "register failed: {}",
        String::from_utf8_lossy(&reg_ok.stderr)
    );

    let reg_bad_cloud = Command::new(bin)
        .args([
            "register",
            "--cloud",
            "nonexistent",
            "--clouds-manifest",
            manifest_path.to_str().unwrap(),
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("register bad cloud");
    assert_eq!(reg_bad_cloud.status.code(), Some(2));

    let reg_dead_ep = Command::new(bin)
        .args(["register", "--endpoint", "http://127.0.0.1:9"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("register dead ep");
    assert_eq!(reg_dead_ep.status.code(), Some(1));

    // `carbon login`: missing token (exit 2), malformed token (exit 2), expired token (exit 2),
    // unreachable endpoint (exit 1), and valid login via `CARBON_AUTH_TOKEN` + `--clouds-manifest` (exit 0).
    let login_missing = Command::new(bin)
        .args(["login", "--endpoint", &emu.base_url()])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env_remove("CARBON_AUTH_TOKEN")
        .output()
        .expect("login missing");
    assert_eq!(login_missing.status.code(), Some(2));

    let login_bad_tok = Command::new(bin)
        .args([
            "login",
            "--endpoint",
            &emu.base_url(),
            "--token",
            "not_a_ctk1",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("login bad tok");
    assert_eq!(login_bad_tok.status.code(), Some(2));

    let verifier = CarbonAuthVerifier::from_secret(auth_secret);
    let expired_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "exp_t".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 1_000,
            admin: false,
        })
        .unwrap();
    let login_exp = Command::new(bin)
        .args([
            "login",
            "--endpoint",
            &emu.base_url(),
            "--token",
            &expired_tok,
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("login expired");
    assert_eq!(login_exp.status.code(), Some(2));

    let valid_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "login_t".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .unwrap();
    let login_dead = Command::new(bin)
        .args([
            "login",
            "--endpoint",
            "http://127.0.0.1:9",
            "--token",
            &valid_tok,
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("login dead");
    assert_eq!(login_dead.status.code(), Some(1));

    let login_env_ok = Command::new(bin)
        .args([
            "login",
            "--cloud",
            "prod",
            "--clouds-manifest",
            manifest_path.to_str().unwrap(),
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env("CARBON_AUTH_TOKEN", &valid_tok)
        .output()
        .expect("login env ok");
    assert!(login_env_ok.status.success());

    // 5. `CLI-34` `carbon redteam` via `--cloud prod --clouds-manifest` + markdown/json outputs + error paths
    let rt_json = root.path().join("rt.json");
    let rt_md = root.path().join("rt.md");
    let rt_ok = Command::new(bin)
        .args([
            "redteam",
            "--cloud",
            "prod",
            "--clouds-manifest",
            manifest_path.to_str().unwrap(),
            "--json-out",
            rt_json.to_str().unwrap(),
            "--markdown-out",
            rt_md.to_str().unwrap(),
        ])
        .output()
        .expect("redteam cloud manifest ok");
    assert!(
        rt_ok.status.success(),
        "redteam failed: {}",
        String::from_utf8_lossy(&rt_ok.stderr)
    );
    assert!(rt_json.is_file() && rt_md.is_file());

    let rt_no_args = Command::new(bin).arg("redteam").output().unwrap();
    assert_eq!(rt_no_args.status.code(), Some(2));
    let rt_bad_cloud = Command::new(bin)
        .args([
            "redteam",
            "--cloud",
            "bogus",
            "--clouds-manifest",
            manifest_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(rt_bad_cloud.status.code(), Some(2));
    let rt_dead = Command::new(bin)
        .args(["redteam", "--endpoint", "http://127.0.0.1:9"])
        .output()
        .unwrap();
    assert_eq!(rt_dead.status.code(), Some(1));

    // 6. `CLI-33` `carbon bench` validation branches + `--csv-dir` CLI execution
    let bench_conflict = Command::new(bin)
        .args(["bench", "--free", "--pro", "--endpoint", &emu.base_url()])
        .output()
        .unwrap();
    assert_eq!(bench_conflict.status.code(), Some(2));

    let bench_bad_effort = Command::new(bin)
        .args([
            "bench",
            "--suite",
            "smoke",
            "--effort",
            "bogus",
            "--endpoint",
            &emu.base_url(),
            "--token",
            &valid_tok,
            "--tenant",
            "login_t",
        ])
        .output()
        .unwrap();
    assert_eq!(bench_bad_effort.status.code(), Some(2));

    let csv_dir = root.path().join("bench_csvs");
    fs::create_dir_all(&csv_dir).unwrap();
    let mut csv_body = String::from("x1,x2,y\n");
    for i in 0..48 {
        let x1 = (i as f64) * 0.2;
        let x2 = ((i % 5) as f64) * 0.4;
        let y = 2.0 * x1 - 0.5 * x2 + 1.0;
        csv_body.push_str(&format!("{x1:.4},{x2:.4},{y:.4}\n"));
    }
    fs::write(csv_dir.join("mini.csv"), csv_body).unwrap();
    let bench_json = root.path().join("sci_bench.json");
    let bench_md = root.path().join("sci_bench.md");
    let bench_csv_ok = Command::new(bin)
        .args([
            "bench",
            "--suite",
            "smoke",
            "--csv-dir",
            csv_dir.to_str().unwrap(),
            "--effort",
            "fast",
            "--endpoint",
            &emu.base_url(),
            "--token",
            &valid_tok,
            "--tenant",
            "login_t",
            "--json-out",
            bench_json.to_str().unwrap(),
            "--markdown-out",
            bench_md.to_str().unwrap(),
        ])
        .output()
        .expect("bench --csv-dir");
    assert!(
        bench_csv_ok.status.success(),
        "bench --csv-dir failed: {}",
        String::from_utf8_lossy(&bench_csv_ok.stderr)
    );
    assert!(bench_json.is_file() && bench_md.is_file());

    // 7. `CLI-29` `carbon export`: train quick model, then test `--out -`, `--json`, `--out <file> --json`, and bad `--lang`
    let saved_model = root.path().join("exp_model.carbon.json");
    let train_csv = csv_dir.join("mini.csv");
    let fit_ok = Command::new(bin)
        .args([
            "fit",
            train_csv.to_str().unwrap(),
            "--target",
            "y",
            "--effort",
            "fast",
            "--endpoint",
            &emu.base_url(),
            "--token",
            &valid_tok,
            "--tenant",
            "login_t",
            "--save",
            saved_model.to_str().unwrap(),
            "--embed-token",
            "--json",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("fit for export");
    assert!(fit_ok.status.success());

    let exp_bad_lang = Command::new(bin)
        .args([
            "export",
            "--model",
            saved_model.to_str().unwrap(),
            "--lang",
            "cobol",
        ])
        .output()
        .unwrap();
    assert_ne!(exp_bad_lang.status.code(), Some(0));

    // Exercise `parse_export_lang` `cpp`, `js`, and `rust-refittable` arms deterministically
    for lang in ["cpp", "js", "rust-refittable"] {
        let missing_out = Command::new(bin)
            .args([
                "export",
                "--model",
                root.path().join("missing_handle.json").to_str().unwrap(),
                "--lang",
                lang,
            ])
            .env("CARBON_CONFIG_DIR", &cfg_dir)
            .output()
            .unwrap();
        assert_ne!(missing_out.status.code(), Some(0));
    }

    let exp_dash = Command::new(bin)
        .args([
            "export",
            "--model",
            saved_model.to_str().unwrap(),
            "--lang",
            "c",
            "--out",
            "-",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .unwrap();
    assert!(
        exp_dash.status.success(),
        "export --lang c --out - failed: {}",
        String::from_utf8_lossy(&exp_dash.stderr)
    );

    let exp_json_stdout = Command::new(bin)
        .args([
            "export",
            "--model",
            saved_model.to_str().unwrap(),
            "--lang",
            "python",
            "--json",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .unwrap();
    assert!(
        exp_json_stdout.status.success(),
        "export --lang python --json failed: stdout={} stderr={}",
        String::from_utf8_lossy(&exp_json_stdout.stdout),
        String::from_utf8_lossy(&exp_json_stdout.stderr)
    );

    let exp_file = root.path().join("exported/model.py");
    let exp_file_json = Command::new(bin)
        .args([
            "export",
            "--model",
            saved_model.to_str().unwrap(),
            "--lang",
            "python",
            "--out",
            exp_file.to_str().unwrap(),
            "--json",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .unwrap();
    assert!(
        exp_file_json.status.success() && exp_file.is_file(),
        "export --lang python --out file --json failed: stdout={} stderr={}",
        String::from_utf8_lossy(&exp_file_json.stdout),
        String::from_utf8_lossy(&exp_file_json.stderr)
    );

    let exp_rs_file = root.path().join("exported/model.rs");
    let exp_file_human = Command::new(bin)
        .args([
            "export",
            "--model",
            saved_model.to_str().unwrap(),
            "--lang",
            "rust",
            "--out",
            exp_rs_file.to_str().unwrap(),
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .unwrap();
    assert!(
        exp_file_human.status.success() && exp_rs_file.is_file(),
        "export --lang rust --out file failed: stdout={} stderr={}",
        String::from_utf8_lossy(&exp_file_human.stdout),
        String::from_utf8_lossy(&exp_file_human.stderr)
    );

    emu.shutdown();
}

#[test]
fn cli_eval_engine_runtime_and_token_mint_error_branches() {
    let root = tempfile::tempdir().expect("tempdir");
    let bin = env!("CARGO_BIN_EXE_carbon");

    // 1. `CLI-35` `carbon token-mint` error branches: ttl 0, invalid tier, missing secret, bad hex, invalid tenant
    assert_eq!(
        Command::new(bin)
            .args(["token-mint", "--tenant", "t1", "--ttl-secs", "0"])
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(bin)
            .args(["token-mint", "--tenant", "t1", "--tier", "bogus"])
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(bin)
            .args(["token-mint", "--tenant", "t1", "--tier", "pro"])
            .env_remove("CARBON_AUTH_SECRET_HEX")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(bin)
            .args(["token-mint", "--tenant", "t1", "--tier", "pro"])
            .env("CARBON_AUTH_SECRET_HEX", "zz_not_hex")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(bin)
            .args(["token-mint", "--tenant", "bad/tenant", "--tier", "pro"])
            .env("CARBON_AUTH_SECRET_HEX", "44".repeat(32))
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );

    // 2. `CLI-46` `carbon eval` & `CLI-45` `carbon engine` runtime bootstrap error branches
    let scratch_dir = root.path().join("eval_scratch");
    let regular_file = root.path().join("not_a_dir");
    fs::write(&regular_file, b"file").unwrap();
    let local_cas = format!("file://{}", root.path().join("cas").display());

    // Missing AWS_LAMBDA_RUNTIME_API -> exit 2
    assert_eq!(
        Command::new(bin)
            .args(["eval", "--scratch-dir", scratch_dir.to_str().unwrap()])
            .env_remove("AWS_LAMBDA_RUNTIME_API")
            .env_remove("LAMBDA_RUNTIME_API")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(bin)
            .args(["engine", "--work-dir", scratch_dir.to_str().unwrap()])
            .env_remove("AWS_LAMBDA_RUNTIME_API")
            .env_remove("LAMBDA_RUNTIME_API")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );

    // Missing CARBON_CAS_URL -> exit 2
    assert_eq!(
        Command::new(bin)
            .args(["eval", "--scratch-dir", scratch_dir.to_str().unwrap()])
            .env("LAMBDA_RUNTIME_API", "127.0.0.1:9")
            .env_remove("CARBON_CAS_URL")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(bin)
            .args(["engine", "--work-dir", scratch_dir.to_str().unwrap()])
            .env("LAMBDA_RUNTIME_API", "127.0.0.1:9")
            .env_remove("CARBON_CAS_URL")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );

    // Unsupported CARBON_CAS_URL scheme -> exit 1 (`open_remote_cas` error)
    assert_eq!(
        Command::new(bin)
            .args(["eval", "--scratch-dir", scratch_dir.to_str().unwrap()])
            .env("LAMBDA_RUNTIME_API", "127.0.0.1:9")
            .env("CARBON_CAS_URL", "ftp://unsupported-cas")
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );
    assert_eq!(
        Command::new(bin)
            .args(["engine", "--work-dir", scratch_dir.to_str().unwrap()])
            .env("LAMBDA_RUNTIME_API", "127.0.0.1:9")
            .env("CARBON_CAS_URL", "ftp://unsupported-cas")
            .env("CARBON_SEAL_SECRET_HEX", "11".repeat(32))
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );

    // `scratch_dir` / `work_dir` is an existing regular file -> `Worker::new` fails -> exit 1
    assert_eq!(
        Command::new(bin)
            .args(["eval", "--scratch-dir", regular_file.to_str().unwrap()])
            .env("LAMBDA_RUNTIME_API", "127.0.0.1:9")
            .env("CARBON_CAS_URL", &local_cas)
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );
    assert_eq!(
        Command::new(bin)
            .args(["engine", "--work-dir", regular_file.to_str().unwrap()])
            .env("LAMBDA_RUNTIME_API", "127.0.0.1:9")
            .env("CARBON_CAS_URL", &local_cas)
            .env("CARBON_SEAL_SECRET_HEX", "11".repeat(32))
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );

    // Unreachable `LAMBDA_RUNTIME_API=127.0.0.1:9` -> runtime loop exits 1 cleanly
    assert_eq!(
        Command::new(bin)
            .args(["eval", "--scratch-dir", scratch_dir.to_str().unwrap()])
            .env("LAMBDA_RUNTIME_API", "127.0.0.1:9")
            .env("CARBON_CAS_URL", &local_cas)
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );
    assert_eq!(
        Command::new(bin)
            .args(["engine", "--work-dir", scratch_dir.to_str().unwrap()])
            .env("LAMBDA_RUNTIME_API", "127.0.0.1:9")
            .env("CARBON_CAS_URL", &local_cas)
            .env("CARBON_SEAL_SECRET_HEX", "11".repeat(32))
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );

    // Bootstrap role dispatch via `CARBON_ROLE` when no subcommand is given
    assert_eq!(
        Command::new(bin)
            .env("LAMBDA_RUNTIME_API", "127.0.0.1:9")
            .env("CARBON_ROLE", "unknown_role")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
}
