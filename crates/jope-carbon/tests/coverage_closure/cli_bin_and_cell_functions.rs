//! Function and closure coverage tests for `bin/carbon.rs` (`self-verify` ECDSA manifest & KEK
//! pins, `serve` secret decoders, `--env` stack endpoint resolvers, CLI subcommand branches) and
//! `cells/{engine_invoker,eval,lambda_runtime}.rs`.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use aws_lc_rs::{
    rand::SystemRandom,
    signature::{EcdsaKeyPair, KeyPair, ECDSA_P256_SHA256_ASN1_SIGNING},
};
use axum::{
    body::Body,
    http::{Response, StatusCode},
    routing::{get, post},
    Router,
};
use base64::Engine as _;

use jope_api::carbon::{CarbonExplainRequest, CarbonTierWire};
use jope_carbon::{
    open_engine_invoker, open_eval_invoker, run_lambda_custom_runtime_loop, sha256_lower_hex,
    verify_ecdsa_p256_spki_pem_signature, CarbonAuthVerifier, CarbonCapabilityClaims,
    CarbonCasStore, CarbonEmuConfig, CarbonEngineInvoker, CarbonEngineKek, CarbonEngineWireCommand,
    CarbonEvalDecryptionGrant, CarbonEvalInvoker, CarbonEvalWorker, CarbonFsCas,
    CarbonHttpEngineInvoker, CarbonHttpEvalInvoker, CarbonLocalLambdaEmulator,
};

use super::{
    spawn_loopback_axum, write_executable_script, ScopedProcessEnv, SingleShotLambdaRuntimeServer,
};

const SPKI_P256_PREFIX: [u8; 26] = [
    0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x08, 0x2a,
    0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03, 0x42, 0x00,
];

fn carbon_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_carbon"))
}

fn generate_ephemeral_spki_keypair() -> (EcdsaKeyPair, String) {
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
        "-----BEGIN PUBLIC KEY-----\r\n{}\r\n{}\r\n-----END PUBLIC KEY-----\r\n",
        &b64[..64],
        &b64[64..]
    );
    (kp, pem)
}

fn sign_manifest_b64(kp: &EcdsaKeyPair, bytes: &[u8]) -> String {
    let rng = SystemRandom::new();
    let sig = kp.sign(&rng, bytes).expect("sign manifest");
    base64::engine::general_purpose::STANDARD.encode(sig.as_ref())
}

#[test]
fn carbon_bin_self_verify_ecdsa_manifest_and_clouds_function_closure() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let bin = carbon_bin();
    let exe_sha = sha256_lower_hex(&fs::read(&bin).expect("read carbon bin"));
    let (kp, pub_pem) = generate_ephemeral_spki_keypair();
    let pub_path = tmp.path().join("test-signing.pub");
    fs::write(&pub_path, &pub_pem).expect("write test pub");

    // Direct library positive check on SPKI verifier
    let sample_sig = kp.sign(&SystemRandom::new(), b"hello-spki").expect("sign");
    verify_ecdsa_p256_spki_pem_signature(&pub_pem, b"hello-spki", sample_sig.as_ref())
        .expect("verify_ecdsa_p256_spki_pem_signature ok");
    assert!(
        verify_ecdsa_p256_spki_pem_signature(&pub_pem, b"tampered", sample_sig.as_ref()).is_err()
    );

    let kek = CarbonEngineKek::generate().expect("kek");
    let clouds_path = tmp.path().join("carbon-clouds.json");
    let clouds_json = serde_json::json!({
        "schema": "jope-carbon-clouds-v2",
        "generated_epoch_ms": 1_760_000_000_000u64,
        "clouds": {
            "dev": {
                "environment": "dev",
                "region": "us-east-2",
                "public_endpoint": "https://api-dev.example.com",
                "engine_kek_pubkey_b64": kek.pubkey_b64(),
                "regions": [{
                    "region": "us-east-2",
                    "public_endpoint": "https://api-dev.example.com",
                    "engine_kek_pubkey_b64": kek.pubkey_b64()
                }]
            }
        }
    })
    .to_string();
    fs::write(&clouds_path, &clouds_json).expect("write clouds");
    let clouds_sha = sha256_lower_hex(clouds_json.as_bytes());

    // Valid signed SHA256SUMS.txt (with comment, blank line, `*` binary marker, & clouds hash)
    let sums_path = tmp.path().join("SHA256SUMS.txt");
    let sig_path = tmp.path().join("SHA256SUMS.txt.sig");
    let sums_text = format!(
        "# Carbon release checksums\n\n{exe_sha} *carbon-linux-x86_64-musl\n{clouds_sha}  carbon-clouds.json\n"
    );
    fs::write(&sums_path, &sums_text).expect("write sums");
    fs::write(&sig_path, sign_manifest_b64(&kp, sums_text.as_bytes())).expect("write sig");

    let ok_out = Command::new(&bin)
        .args([
            "self-verify",
            "--manifest",
            sums_path.to_str().expect("utf8"),
            "--signature",
            sig_path.to_str().expect("utf8"),
            "--clouds-manifest",
            clouds_path.to_str().expect("utf8"),
            "--allow-test-pubkey",
            pub_path.to_str().expect("utf8"),
        ])
        .env("CARBON_CONFIG_DIR", tmp.path().join("cfg_ok"))
        .output()
        .expect("self-verify ok");
    assert!(
        ok_out.status.success(),
        "self-verify failed: {}",
        String::from_utf8_lossy(&ok_out.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&ok_out.stdout).expect("json");
    assert_eq!(parsed["manifest_ecdsa_verified"], true);
    assert_eq!(parsed["clouds_manifest_verified"], true);
    assert_eq!(parsed["kek_pins_verified"], 1);

    // Negative controls: missing pubkey, missing manifest (--require-manifest), bad signature,
    // malformed SHA256SUMS line, empty SHA256SUMS, binary hash mismatch, clouds hash mismatch
    let bad_pub = Command::new(&bin)
        .args(["self-verify", "--allow-test-pubkey", "/nonexistent-pub.pem"])
        .output()
        .expect("bad pub");
    assert!(!bad_pub.status.success());

    let req_missing = Command::new(&bin)
        .args(["self-verify", "--require-manifest"])
        .env("CARBON_CONFIG_DIR", tmp.path().join("cfg_empty"))
        .output()
        .expect("require manifest missing");
    assert!(!req_missing.status.success());

    for (idx, bad_manifest) in [
        "badhex  carbon\n".to_owned(),
        "# comments only\n\n".to_owned(),
        format!("{} *carbon\n", "00".repeat(32)),
        format!(
            "{exe_sha} *carbon\n{}  carbon-clouds.json\n",
            "ff".repeat(32)
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let mp = tmp.path().join(format!("bad-{idx}.txt"));
        let sp = tmp.path().join(format!("bad-{idx}.txt.sig"));
        fs::write(&mp, &bad_manifest).expect("write mp");
        fs::write(&sp, sign_manifest_b64(&kp, bad_manifest.as_bytes())).expect("write sp");
        let res = Command::new(&bin)
            .args([
                "self-verify",
                "--manifest",
                mp.to_str().expect("utf8"),
                "--signature",
                sp.to_str().expect("utf8"),
                "--clouds-manifest",
                clouds_path.to_str().expect("utf8"),
                "--allow-test-pubkey",
                pub_path.to_str().expect("utf8"),
            ])
            .output()
            .expect("bad self-verify");
        assert!(!res.status.success(), "case {idx} unexpectedly succeeded");
    }
}

fn write_small_csv(path: &Path) {
    let mut s = String::from("f1,f2,y\n");
    for i in 0..32 {
        let x1 = (i as f64) * 0.25;
        let x2 = ((i % 5) as f64) * 0.5;
        let y = 2.0 * x1 - x2 + 0.5;
        s.push_str(&format!("{x1:.3},{x2:.3},{y:.3}\n"));
    }
    fs::write(path, s).expect("write small csv");
}

#[test]
fn carbon_bin_serve_secrets_stack_endpoints_and_subcommand_branches() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg_dir = tmp.path().join("cfg");
    fs::create_dir_all(&cfg_dir).expect("mkdir cfg");
    let bin = carbon_bin();

    // 1. `carbon serve` secret validation helpers (`decode_required_32byte_hex` & `decode_or_ephemeral_32byte_secret`)
    let srv_no_cas = Command::new(&bin)
        .args([
            "serve",
            "--listen",
            "0.0.0.0:19999",
            "--data-dir",
            tmp.path().join("srv0").to_str().expect("utf8"),
        ])
        .output()
        .expect("serve no cas");
    assert!(!srv_no_cas.status.success());

    for (env_k, env_v) in [
        ("CARBON_SEAL_SECRET_HEX", ""),
        ("CARBON_SEAL_SECRET_HEX", "zz-not-hex"),
        ("CARBON_SEAL_SECRET_HEX", "1122"),
        ("CARBON_AUTH_SECRET_HEX", "bad-hex"),
    ] {
        let mut cmd = Command::new(&bin);
        cmd.args([
            "serve",
            "--listen",
            "0.0.0.0:19999",
            "--allow-local-fs-dev",
            "--data-dir",
            tmp.path().join("srv1").to_str().expect("utf8"),
        ]);
        if env_k == "CARBON_AUTH_SECRET_HEX" {
            cmd.arg("--dev-ephemeral").env(env_k, env_v);
        } else if !env_v.is_empty() {
            cmd.env(env_k, env_v);
        } else {
            cmd.env_remove("CARBON_SEAL_SECRET_HEX");
        }
        let out = cmd.output().expect("serve secret check");
        assert!(!out.status.success());
    }

    // 2. Start emulator + mock `aws` for `ping --env`, `redteam --env`, `CARBON_ENDPOINT`, `bench --csv-dir`, `run --format pretty`
    let emu_cfg = CarbonEmuConfig::for_dir(tmp.path().join("emu_cli"));
    let emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start emu");
    let mock_aws = tmp.path().join("aws-cli.sh");
    write_executable_script(
        &mock_aws,
        &format!(
            "#!/bin/sh\ncase \"$*\" in\n  *\"jopedime-carbon-dev\"*) printf '{{\"Stacks\":[{{\"Outputs\":[{{\"OutputKey\":\"EdgeEndpointUrl\",\"OutputValue\":\"{}\"}}]}}]}}\\n' ;;\n  *) exit 2 ;;\nesac\n",
            emu.base_url()
        ),
    );

    let ping_env = Command::new(&bin)
        .args(["ping", "--env", "dev", "--region", "us-east-2"])
        .env("CARBON_AWS_CLI_BIN", &mock_aws)
        .output()
        .expect("ping --env dev");
    assert!(ping_env.status.success());

    let ping_env_fail = Command::new(&bin)
        .args(["ping", "--env", "staging", "--region", "us-east-2"])
        .env("CARBON_AWS_CLI_BIN", &mock_aws)
        .output()
        .expect("ping --env fail");
    assert!(!ping_env_fail.status.success());

    let ping_var = Command::new(&bin)
        .arg("ping")
        .env("CARBON_ENDPOINT", emu.base_url())
        .output()
        .expect("ping CARBON_ENDPOINT");
    assert!(ping_var.status.success());

    let rt_env = Command::new(&bin)
        .args(["redteam", "--env", "dev", "--region", "us-east-2"])
        .env("CARBON_AWS_CLI_BIN", &mock_aws)
        .output()
        .expect("redteam --env dev");
    assert!(rt_env.status.success());

    // 3. `whoami` via `--token-file -` (stdin), expired token (`EXIT_AUTH_EXPIRED`), and invalid token
    let verifier = CarbonAuthVerifier::from_secret([0x77; 32]);
    let valid_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "stdin_user".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint valid");
    let mut child = Command::new(&bin)
        .args(["whoami", "--token-file", "-"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn whoami stdin");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(valid_tok.as_bytes())
        .expect("write tok");
    let who_ok = child.wait_with_output().expect("wait whoami");
    assert!(who_ok.status.success());

    let exp_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "exp_user".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 100,
            admin: false,
        })
        .expect("mint expired");
    let who_exp = Command::new(&bin)
        .args(["whoami", "--token", &exp_tok])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("whoami expired");
    assert!(!who_exp.status.success());

    let who_bad = Command::new(&bin)
        .args(["whoami", "--token", "malformed-token"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("whoami bad");
    assert!(!who_bad.status.success());

    // 4. `run --format pretty`, `bench --csv-dir`, `emu` without `--self-test`, `rollback` bad arity, `engine`/`eval` missing runtime API
    let run_pretty = Command::new(&bin)
        .args([
            "run",
            "--endpoint",
            &emu.base_url(),
            "--synthetic-rows",
            "36",
            "--features",
            "2",
            "--format",
            "pretty",
            "--shred",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("run pretty");
    assert!(run_pretty.status.success());

    let csv_dir = tmp.path().join("sci_csvs");
    fs::create_dir_all(&csv_dir).expect("mkdir sci_csvs");
    write_small_csv(&csv_dir.join("task_a.csv"));
    let sci_json = tmp.path().join("sci.json");
    let sci_md = tmp.path().join("sci.md");
    let sci_bench = Command::new(&bin)
        .args([
            "bench",
            "--endpoint",
            &emu.base_url(),
            "--suite",
            "smoke",
            "--csv-dir",
            csv_dir.to_str().expect("utf8"),
            "--effort",
            "fast",
            "--json-out",
            sci_json.to_str().expect("utf8"),
            "--markdown-out",
            sci_md.to_str().expect("utf8"),
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("bench --suite smoke --csv-dir");
    assert!(
        sci_bench.status.success() && sci_json.is_file() && sci_md.is_file(),
        "sci bench failed: {}",
        String::from_utf8_lossy(&sci_bench.stderr)
    );

    for bad_args in [
        vec!["emu"],
        vec!["rollback", "--env", "dev", "--to-versions", "1,2"],
        vec!["engine"],
        vec!["eval"],
    ] {
        let out = Command::new(&bin)
            .args(&bad_args)
            .env_remove("AWS_LAMBDA_RUNTIME_API")
            .output()
            .expect("cli negative check");
        assert!(!out.status.success());
    }
}

#[test]
fn cells_invokers_eval_worker_and_lambda_runtime_function_closure() {
    // 1. `open_engine_invoker`, `open_eval_invoker`, `CarbonHttpEngineInvoker`, `CarbonHttpEvalInvoker`
    {
        let _env = ScopedProcessEnv::set(&[("AWS_REGION", "eu-central-1")]);
        let _ = open_engine_invoker("lambda://jopedime-carbon-dev-engine:live", "us-east-2");
        let _ = open_eval_invoker("lambda://jopedime-carbon-dev-eval:live", "us-east-2");
    }
    let _ = open_engine_invoker("http://127.0.0.1:9", "us-east-2");
    let _ = open_eval_invoker("http://127.0.0.1:9", "us-east-2");

    let app_ok = Router::new()
        .route(
            "/ok",
            post(|| async {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from(br#"{"ok":true}"#.to_vec()))
                    .expect("resp")
            }),
        )
        .route(
            "/err",
            post(|| async {
                Response::builder()
                    .status(StatusCode::BAD_GATEWAY)
                    .body(Body::from(b"upstream fail".to_vec()))
                    .expect("resp")
            }),
        )
        .route(
            "/2018-06-01/runtime/invocation/next",
            get(|| async {
                Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from(b"{}".to_vec()))
                    .expect("no req id header")
            }),
        );
    let (addr, tx, jh) = spawn_loopback_axum("mock-invoker", app_ok);

    let cmd = CarbonEngineWireCommand::Explain {
        run_id: "run_01".to_owned(),
        explain: CarbonExplainRequest {
            tenant_id: "t1".to_owned(),
            top_k: Some(8),
            client_response_pubkey_b64: None,
            wrapped_u_dek_b64: None,
        },
    };
    let eng_ok = CarbonHttpEngineInvoker::new(format!("http://{addr}/ok"));
    assert_eq!(
        eng_ok.invoke_engine(&cmd).expect("sync ok"),
        br#"{"ok":true}"#
    );
    eng_ok.invoke_engine_async(&cmd).expect("async ok");

    let eng_err = CarbonHttpEngineInvoker::new(format!("http://{addr}/err"));
    assert!(eng_err.invoke_engine(&cmd).is_err());
    assert!(eng_err.invoke_engine_async(&cmd).is_err());

    // 2. `CarbonEvalDecryptionGrant` redacted Debug + `CarbonEvalWorker` error branches
    let grant = CarbonEvalDecryptionGrant {
        tenant_id: "t1".to_owned(),
        fit_slot_subkey_b64: "secret-subkey-material".to_owned(),
        fit_plaintext_sha256: "aa".repeat(32),
        exp_epoch_secs: 10,
    };
    let g_dbg = format!("{grant:?}");
    assert!(g_dbg.contains("[REDACTED]") && !g_dbg.contains("secret-subkey-material"));

    let tmp = tempfile::tempdir().expect("tempdir");
    let cas = Arc::new(CarbonFsCas::open(tmp.path().join("cas")).expect("cas"));
    let raw_blob = b"not-a-valid-cuen";
    let blob_sha = sha256_lower_hex(raw_blob);
    cas.put_scoped_blob_verified(Some("tenant_01/upload_01"), &blob_sha, raw_blob)
        .expect("put blob");
    let eval_worker = CarbonEvalWorker::new(cas, tmp.path().join("scratch")).expect("eval worker");
    assert!(eval_worker.handle_wire_bytes(b"not-json").is_err());

    let eval_err_http = CarbonHttpEvalInvoker::new(format!("http://{addr}/err"));
    let bad_wire_json = serde_json::json!({
        "fit_jmat_sha256": blob_sha,
        "blob_scope": "tenant_01/upload_01",
        "request": {
            "task_id": "t",
            "fold_index": 0,
            "seed": 1,
            "candidates": [],
            "objective": "rmse"
        },
        "fit_champion": false,
        "decryption_grant": {
            "tenant_id": "tenant_01",
            "fit_slot_subkey_b64": "AQID",
            "fit_plaintext_sha256": "aa".repeat(32),
            "exp_epoch_secs": 10
        }
    });
    if let Ok(parsed_wire) = serde_json::from_value(bad_wire_json.clone()) {
        let _ = eval_err_http.invoke_eval(&parsed_wire);
        assert!(eval_worker.invoke_eval(&parsed_wire).is_err());
    }
    let future_bad_subkey = serde_json::json!({
        "fit_jmat_sha256": blob_sha,
        "blob_scope": "tenant_01/upload_01",
        "request": bad_wire_json["request"],
        "fit_champion": false,
        "decryption_grant": {
            "tenant_id": "tenant_01",
            "fit_slot_subkey_b64": "AQID",
            "fit_plaintext_sha256": "aa".repeat(32),
            "exp_epoch_secs": 4_102_444_800u64
        }
    });
    let _ = eval_worker.handle_wire_bytes(&serde_json::to_vec(&future_bad_subkey).expect("vec"));

    // 3. `run_lambda_custom_runtime_loop`: missing request-id error, handler Err -> /error post, & max_invocations
    assert!(
        run_lambda_custom_runtime_loop(&addr.to_string(), Some(1), |_| Ok(Vec::new())).is_err()
    );
    let _ = tx.send(());
    let _ = jh.join();

    let rt_srv = SingleShotLambdaRuntimeServer::start(vec![b"ev1".to_vec(), b"ev2".to_vec()]);
    let n1 = run_lambda_custom_runtime_loop(&rt_srv.host_port(), Some(1), |_inv| {
        Err("simulated handler failure".to_owned())
    })
    .expect("handler err posted cleanly");
    assert_eq!(n1, 1);
    assert_eq!(rt_srv.errors().len(), 1);
}

#[test]
fn carbon_bin_error_closures_pretty_predict_and_fitness_branches() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg_dir = tmp.path().join("cfg_err");
    fs::create_dir_all(&cfg_dir).expect("mkdir cfg");
    let bin = carbon_bin();

    // 1. `serve --cas-url` non-loopback without CARBON_SEAL_SECRET_HEX (`L988`, `L3763`, `L3764`)
    let srv_cas = Command::new(&bin)
        .args([
            "serve",
            "--listen",
            "0.0.0.0:19998",
            "--cas-url",
            "  s3://test-bkt/pfx  ",
            "--data-dir",
            tmp.path().join("srv_cas").to_str().expect("utf8"),
        ])
        .env_remove("CARBON_SEAL_SECRET_HEX")
        .output()
        .expect("serve cas missing seal secret");
    assert!(!srv_cas.status.success());

    // 2. `whoami` via `CARBON_AUTH_TOKEN` (`L2734`, `L2735`) + missing `--token-file` (`L1091`)
    let verifier = CarbonAuthVerifier::from_secret([0x55; 32]);
    let valid_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "env_tok_user".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint");
    let who_env = Command::new(&bin)
        .arg("whoami")
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env("CARBON_AUTH_TOKEN", format!("  {valid_tok}  "))
        .output()
        .expect("whoami env");
    assert!(who_env.status.success());

    let missing_tok_file = tmp.path().join("no-such-token.txt");
    let who_no_file = Command::new(&bin)
        .args([
            "whoami",
            "--token-file",
            missing_tok_file.to_str().expect("utf8"),
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("whoami missing file");
    assert!(!who_no_file.status.success());

    // 3. `resolve_cli_target_and_auth` error closures (`L1158`, `L1168`, `L1181`, `L1190`, `L1193`, `L1194`, `L1242`, `L1286`)
    let emu =
        CarbonLocalLambdaEmulator::start(CarbonEmuConfig::for_dir(tmp.path().join("emu_err")))
            .expect("emu");
    let mock_aws_ep = tmp.path().join("aws-ep.sh");
    write_executable_script(
        &mock_aws_ep,
        &format!(
            "#!/bin/sh\nprintf '{{\"Stacks\":[{{\"Outputs\":[{{\"OutputKey\":\"EdgeEndpointUrl\",\"OutputValue\":\"{}\"}}]}}]}}\\n'\n",
            emu.base_url()
        ),
    );
    for (args, extra_env) in [
        (
            vec!["run", "--env", "dev", "--region", "us-east-2"],
            vec![("CARBON_AWS_CLI_BIN", "/nonexistent-aws")],
        ),
        (
            vec![
                "run",
                "--cloud",
                "bad_cloud",
                "--clouds-manifest",
                "/nonexistent-clouds.json",
            ],
            vec![],
        ),
        (vec!["run", "--endpoint", "http://127.0.0.1:9"], vec![]),
        (
            vec![
                "run",
                "--endpoint",
                emu.base_url().as_str(),
                "--token-file",
                missing_tok_file.to_str().expect("utf8"),
            ],
            vec![],
        ),
        (
            vec!["run", "--endpoint", emu.base_url().as_str(), "--pro"],
            vec![("CARBON_AUTH_TOKEN", "  malformed_env_token  ")],
        ),
        (
            vec!["run", "--env", "dev", "--region", "us-east-2", "--pro"],
            vec![
                ("CARBON_AWS_CLI_BIN", mock_aws_ep.to_str().expect("utf8")),
                ("CARBON_SECRETSMANAGER_ENDPOINT_URL", "http://127.0.0.1:9"),
                ("AWS_ACCESS_KEY_ID", "TESTKEYCLI"),
                ("AWS_SECRET_ACCESS_KEY", "test-sec-cli"),
            ],
        ),
    ] {
        let mut cmd = Command::new(&bin);
        cmd.args(&args).env("CARBON_CONFIG_DIR", &cfg_dir);
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let out = cmd.output().expect("neg run");
        assert!(!out.status.success(), "unexpected success for {args:?}");
    }

    // 4. `self-verify --manifest <m>` without `--signature` (`L1465`), `emu --self-test` bad dir (`L2086`),
    //    `redteam` bad output paths (`L2326`, `L2332`), `bench --csv-dir` default suite + bad effort (`L3487`)
    let m_only = tmp.path().join("SHA256SUMS.txt");
    fs::write(&m_only, "00\n").expect("write");
    assert!(!Command::new(&bin)
        .args(["self-verify", "--manifest", m_only.to_str().expect("utf8")])
        .output()
        .expect("self-verify no sig")
        .status
        .success());

    assert!(!Command::new(&bin)
        .args([
            "emu",
            "--self-test",
            "--work-dir",
            "/dev/null/impossible_emu"
        ])
        .output()
        .expect("emu bad workdir")
        .status
        .success());

    assert!(!Command::new(&bin)
        .args([
            "redteam",
            "--endpoint",
            &emu.base_url(),
            "--json-out",
            "/dev/null/impossible.json",
        ])
        .output()
        .expect("redteam bad json-out")
        .status
        .success());
    let ok_rt_json = tmp.path().join("rt_ok.json");
    assert!(!Command::new(&bin)
        .args([
            "redteam",
            "--endpoint",
            &emu.base_url(),
            "--json-out",
            ok_rt_json.to_str().expect("utf8"),
            "--markdown-out",
            "/dev/null/impossible.md",
        ])
        .output()
        .expect("redteam bad md-out")
        .status
        .success());

    assert!(!Command::new(&bin)
        .args([
            "bench",
            "--endpoint",
            &emu.base_url(),
            "--csv-dir",
            tmp.path().to_str().expect("utf8"),
            "--effort",
            "not_an_effort",
        ])
        .output()
        .expect("bench default suite bad effort")
        .status
        .success());

    // 5. `carbon engine` with `CARBON_EVAL_INVOKE_URL` & `CARBON_FORCE_HYBRID_FANOUT=1` (`L2008`, `L2009`, `L2026`)
    let eng_rt = SingleShotLambdaRuntimeServer::start(vec![b"not-valid-engine-wire".to_vec()]);
    let _ = Command::new(&bin)
        .args([
            "engine",
            "--work-dir",
            tmp.path().join("eng_w").to_str().expect("utf8"),
        ])
        .env("LAMBDA_RUNTIME_API", eng_rt.host_port())
        .env_remove("AWS_LAMBDA_RUNTIME_API")
        .env("CARBON_CAS_URL", "s3://test-bkt/pfx")
        .env("CARBON_SEAL_SECRET_HEX", "44".repeat(32))
        .env("CARBON_EVAL_INVOKE_URL", "http://127.0.0.1:9")
        .env("CARBON_FORCE_HYBRID_FANOUT", "1")
        .env("AWS_ACCESS_KEY_ID", "TESTKEYENG")
        .env("AWS_SECRET_ACCESS_KEY", "test-sec-eng")
        .output()
        .expect("carbon engine runtime");
    assert_eq!(eng_rt.errors().len(), 1);

    // 6. `fit` + `predict --format pretty` on variable AND constant-target labeled CSVs (`L3045`, `L3046`, `L3056`)
    let train_csv = tmp.path().join("train_p.csv");
    write_small_csv(&train_csv);
    let model_path = tmp.path().join("m.carbon.json");
    let fit_out = Command::new(&bin)
        .args([
            "fit",
            train_csv.to_str().expect("utf8"),
            "--target",
            "y",
            "--save",
            model_path.to_str().expect("utf8"),
            "--endpoint",
            &emu.base_url(),
            "--effort",
            "fast",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("fit");
    assert!(fit_out.status.success());

    let pred_var = Command::new(&bin)
        .args([
            "predict",
            train_csv.to_str().expect("utf8"),
            "--model",
            model_path.to_str().expect("utf8"),
            "--format",
            "pretty",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("predict pretty var");
    assert!(pred_var.status.success());

    let const_csv = tmp.path().join("const_p.csv");
    fs::write(&const_csv, "f1,f2,y\n1.0,2.0,5.0\n2.0,3.0,5.0\n").expect("write const csv");
    let pred_const = Command::new(&bin)
        .args([
            "predict",
            const_csv.to_str().expect("utf8"),
            "--model",
            model_path.to_str().expect("utf8"),
            "--format",
            "pretty",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("predict pretty const");
    assert!(pred_const.status.success());

    // 7. `fitness score` missing column (`L4351`) and non-finite cell (`L4385`)
    assert!(!Command::new(&bin)
        .args([
            "fitness",
            "score",
            "--spec",
            "rmse",
            "--csv",
            train_csv.to_str().expect("utf8"),
            "--target",
            "no_such_col",
            "--pred",
            "f1",
        ])
        .output()
        .expect("fitness missing col")
        .status
        .success());
    let bad_num_csv = tmp.path().join("bad_num.csv");
    fs::write(&bad_num_csv, "y,p\n1.0,not_a_number\n").expect("write bad num");
    assert!(!Command::new(&bin)
        .args([
            "fitness",
            "score",
            "--spec",
            "rmse",
            "--csv",
            bad_num_csv.to_str().expect("utf8"),
            "--target",
            "y",
            "--pred",
            "p",
        ])
        .output()
        .expect("fitness bad num")
        .status
        .success());
}
