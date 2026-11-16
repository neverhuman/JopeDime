//! Integration tests for atomic filesystem races (`FS-01` `put_state_object_if_absent`, `FS-02`
//! concurrent `save_stored_cloud_token`), `MEM-05` 64-stripe `CarbonRateLimiter`, read-only
//! multi-region drift verification (`execute_carbon_drift_check` + `carbon deploy --check-drift`),
//! offline & `openssl`-less `deploy/carbon/install.sh`, and CLI UX upgrades (`upgrade` aliases +
//! `.tar.xz` + GitHub tag filter + `.exe.old.*` sweep, `uninstall`, `--token-stdin`, `doctor`,
//! dynamic `completions`, `--keep` / `--ephemeral`, bare TTY banner, `emu --real-workers`, and `--local`).

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Barrier};
use std::thread;

use aws_lc_rs::{
    rand::SystemRandom,
    signature::{EcdsaKeyPair, KeyPair, ECDSA_P256_SHA256_ASN1_SIGNING},
};
use axum::{http::StatusCode, routing::get, Router};
use base64::Engine as _;
use flate2::{write::GzEncoder, Compression};
use jope_api::carbon::CarbonTierWire;
use jope_carbon::{
    build_deterministic_single_bin_tar, execute_carbon_drift_check,
    execute_self_upgrade_with_options, list_stored_cloud_tokens, persist_model_handle_with_options,
    save_stored_cloud_token, sha256_lower_hex, try_external_stream_compress, CarbonAuthVerifier,
    CarbonCapabilityClaims, CarbonCasStore, CarbonEmuConfig, CarbonEngineKek, CarbonFsCas,
    CarbonLocalLambdaEmulator, CarbonRateLimiter, CarbonSavedModelHandle, CarbonTabularEncoder,
    CARBON_BUILD_TARGET, CARBON_MODEL_HANDLE_SCHEMA, CARBON_PUBLISHED_INSTALLER_TAG,
};

use super::{spawn_loopback_axum, ScopedProcessEnv};

const SPKI_P256_PREFIX: [u8; 26] = [
    0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, 0x06, 0x08, 0x2a,
    0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03, 0x42, 0x00,
];

fn gen_test_signing_keypair() -> (EcdsaKeyPair, String) {
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
fn fs01_cas_atomic_create_new_and_fs02_credentials_concurrent_races() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // 1. `FS-01`: 8 threads racing `CarbonFsCas::put_state_object_if_absent` on identical key
    let cas = Arc::new(CarbonFsCas::open(tmp.path().join("cas_race")).expect("open cas"));
    let barrier = Arc::new(Barrier::new(8));
    let mut handles = Vec::with_capacity(8);
    for tid in 0..8usize {
        let cas_ref = Arc::clone(&cas);
        let bar_ref = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let body = format!(r#"{{"winner":{tid}}}"#);
            bar_ref.wait();
            cas_ref
                .put_state_object_if_absent(
                    "runs/tenant_race/run_race_01/lease.json",
                    body.as_bytes(),
                )
                .expect("put_state_object_if_absent")
        }));
    }
    let winners: usize = handles
        .into_iter()
        .map(|h| usize::from(h.join().expect("join cas race thread")))
        .sum();
    assert_eq!(
        winners, 1,
        "FS-01 O_CREAT|O_EXCL hard-link publication must admit exactly 1 winner out of 8 threads"
    );

    // 2. `FS-02`: 8 threads racing `save_stored_cloud_token` across `cloud_0..cloud_7`
    let cfg_dir = tmp.path().join("cfg_race");
    fs::create_dir_all(&cfg_dir).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&cfg_dir, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let verifier = CarbonAuthVerifier::from_secret([0x55u8; 32]);
    let tokens: Vec<(String, String)> = (0..8)
        .map(|i| {
            let tok = verifier
                .mint_token(&CarbonCapabilityClaims {
                    tenant_id: format!("tenant_race_{i}"),
                    tier: CarbonTierWire::Pro,
                    exp_epoch_secs: 4_102_444_800,
                    admin: false,
                })
                .unwrap();
            (format!("cloud_{i}"), tok)
        })
        .collect();
    {
        let _env = ScopedProcessEnv::set_opt(&[(
            "CARBON_CONFIG_DIR",
            Some(cfg_dir.to_str().expect("utf8")),
        )]);
        let bar = Arc::new(Barrier::new(8));
        thread::scope(|s| {
            for (cname, tok) in &tokens {
                let b = Arc::clone(&bar);
                s.spawn(move || {
                    b.wait();
                    save_stored_cloud_token(cname, "https://prod.example", tok)
                        .expect("concurrent save_stored_cloud_token");
                });
            }
        });
        let stored = list_stored_cloud_tokens().expect("list stored cloud tokens");
        assert_eq!(
            stored.len(),
            8,
            "FS-02 file lock lost concurrent credential updates: got {}",
            stored.len()
        );
    }

    // 3. `MEM-05`: 64-stripe `CarbonRateLimiter` quota enforcement and window rollover
    //    `check_and_record(&self, key: &str, now_secs: u64, max_per_window: u32)`
    let limiter = CarbonRateLimiter::new(10);
    for i in 0..80 {
        let key = format!("ip:198.51.100.{i}:ping");
        assert!(limiter.check_and_record(&key, 1_000, 2));
        assert!(limiter.check_and_record(&key, 1_001, 2));
        assert!(!limiter.check_and_record(&key, 1_002, 2));
        // Next window (epoch 1_015) resets counter cleanly
        assert!(limiter.check_and_record(&key, 1_015, 2));
    }
}

#[test]
#[cfg(unix)]
fn deploy_check_drift_clean_and_drifted_detection_and_cli_exit_codes() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().expect("tempdir");
    let mut emu_cfg = CarbonEmuConfig::for_dir(tmp.path().join("drift_emu"));
    emu_cfg.force_hybrid_fanout = false;
    let kek = CarbonEngineKek::from_scalar_bytes(&emu_cfg.engine_kek_scalar).unwrap();
    let kek_b64 = kek.pubkey_b64();
    let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start emu");

    // Loopback Secrets Manager mock for `ensure_cell_secrets_in_vault(..., read_only=true)`
    let sm_kek = Arc::new(std::sync::Mutex::new(kek_b64.clone()));
    let sm_kek_route = Arc::clone(&sm_kek);
    let sm_secret_hex = hex::encode([0x11u8; 32]);
    let sm_app = Router::new().route(
        "/",
        axum::routing::post(move || {
            let k = sm_kek_route.lock().unwrap().clone();
            let s_hex = sm_secret_hex.clone();
            async move {
                let secret_str = serde_json::json!({
                    "presign_secret_hex": "33".repeat(32),
                    "auth_secret_hex": "44".repeat(32),
                    "seal_secret_hex": "77".repeat(32),
                    "internal_token": "88".repeat(32),
                    "engine_kek_hex": s_hex,
                    "engine_kek_pubkey_b64": k,
                })
                .to_string();
                let body = serde_json::json!({
                    "ARN": "arn:aws:secretsmanager:us-east-2:123456789012:secret:jopedime/carbon/dev/cell-secrets",
                    "VersionId": "v1",
                    "SecretString": secret_str,
                })
                .to_string();
                (StatusCode::OK, body)
            }
        }),
    );
    let (sm_addr, sm_shutdown, sm_join) = spawn_loopback_axum("drift-sm", sm_app);
    let sm_url = format!("http://{sm_addr}");

    // Mock `aws` CLI in `mock_bin/aws` controlled via `MOCK_DRIFT_MODE` (`clean` vs `drifted`)
    let mock_bin = tmp.path().join("mock_bin");
    fs::create_dir_all(&mock_bin).unwrap();
    let aws_script = mock_bin.join("aws");
    let emu_url = emu.base_url();
    let script_body = format!(
        r#"#!/usr/bin/env bash
set -euo pipefail
ARGS="$*"
MODE="${{MOCK_DRIFT_MODE:-clean}}"
if [[ "$ARGS" == *"cloudformation describe-stacks"* ]]; then
  if [[ "$MODE" == "drifted" ]]; then
    STATUS="UPDATE_ROLLBACK_COMPLETE"
  else
    STATUS="UPDATE_COMPLETE"
  fi
  cat <<EOF
{{"Stacks":[{{"StackName":"jopedime-carbon-dev","StackStatus":"$STATUS","Outputs":[
  {{"OutputKey":"ReleaseId","OutputValue":"carbon-v1.2.0-test"}},
  {{"OutputKey":"PublicEdgeEndpointUrl","OutputValue":"{emu_url}"}},
  {{"OutputKey":"EdgeEndpointUrl","OutputValue":"{emu_url}"}}
]}}]}}
EOF
  exit 0
fi
if [[ "$ARGS" == *"lambda get-alias"* ]]; then
  if [[ "$MODE" == "drifted" && "$ARGS" == *"-edge"* ]]; then
    echo '{{"FunctionVersion":"$LATEST"}}'
  else
    echo '{{"FunctionVersion":"7"}}'
  fi
  exit 0
fi
echo '{{}}'
"#
    );
    fs::write(&aws_script, script_body).unwrap();
    fs::set_permissions(&aws_script, fs::Permissions::from_mode(0o755)).unwrap();

    let clouds_clean = tmp.path().join("clouds_clean.json");
    fs::write(
        &clouds_clean,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "jope-carbon-clouds-v1",
            "generated_epoch_ms": 1_780_000_000_000u64,
            "clouds": {
                "dev": {
                    "environment": "dev",
                    "region": "us-east-2",
                    "public_endpoint": emu.base_url(),
                    "engine_kek_pubkey_b64": kek_b64,
                    "regions": [{
                        "region": "us-east-2",
                        "public_endpoint": emu.base_url(),
                        "engine_kek_pubkey_b64": kek_b64
                    }]
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let wrong_manifest_kek = CarbonEngineKek::from_scalar_bytes(&[0x99u8; 32])
        .unwrap()
        .pubkey_b64();
    let clouds_drifted = tmp.path().join("clouds_drifted.json");
    fs::write(
        &clouds_drifted,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "jope-carbon-clouds-v1",
            "generated_epoch_ms": 1_780_000_000_000u64,
            "clouds": {
                "dev": {
                    "environment": "dev",
                    "region": "us-east-2",
                    "public_endpoint": "https://wrong.example.org",
                    "engine_kek_pubkey_b64": wrong_manifest_kek,
                    "regions": [{
                        "region": "us-east-2",
                        "public_endpoint": "https://wrong.example.org",
                        "engine_kek_pubkey_b64": wrong_manifest_kek
                    }]
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let orig_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{orig_path}", mock_bin.display());
    let carbon_bin = env!("CARGO_BIN_EXE_carbon");

    // Case A: Clean in-sync region -> `all_in_sync == true` & CLI exit code 0
    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("PATH", Some(&new_path)),
            ("CARBON_AWS_CLI_BIN", Some(aws_script.to_str().unwrap())),
            ("MOCK_DRIFT_MODE", Some("clean")),
            ("CARBON_SECRETSMANAGER_ENDPOINT_URL", Some(&sm_url)),
            ("AWS_ACCESS_KEY_ID", Some("test_drift_key_01")),
            ("AWS_SECRET_ACCESS_KEY", Some("test_drift_secret_01")),
            ("AWS_SESSION_TOKEN", None),
        ]);
        let rep =
            execute_carbon_drift_check("dev", &["us-east-2".to_owned()], clouds_clean.to_str())
                .expect("clean drift check");
        assert!(
            rep.all_in_sync,
            "expected clean drift report, got {:?}",
            rep.regions
        );

        let out_json = tmp.path().join("drift_clean_out.json");
        let cli_ok = Command::new(carbon_bin)
            .args([
                "deploy",
                "--env",
                "dev",
                "--regions",
                "us-east-2",
                "--check-drift",
                "--clouds-manifest",
                clouds_clean.to_str().unwrap(),
                "--json-out",
                out_json.to_str().unwrap(),
            ])
            .output()
            .expect("carbon deploy --check-drift clean");
        assert!(
            cli_ok.status.success(),
            "expected exit 0 on clean drift check: stderr={}",
            String::from_utf8_lossy(&cli_ok.stderr)
        );
        assert!(out_json.is_file());
    }

    // Case B: Drifted region (`$LATEST` alias + `UPDATE_ROLLBACK_COMPLETE` + Vault KEK != Ping KEK + Manifest drift) -> exit 1
    *sm_kek.lock().unwrap() = "BWrongVaultKek".to_owned();
    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("PATH", Some(&new_path)),
            ("CARBON_AWS_CLI_BIN", Some(aws_script.to_str().unwrap())),
            ("MOCK_DRIFT_MODE", Some("drifted")),
            ("CARBON_SECRETSMANAGER_ENDPOINT_URL", Some(&sm_url)),
            ("AWS_ACCESS_KEY_ID", Some("test_drift_key_01")),
            ("AWS_SECRET_ACCESS_KEY", Some("test_drift_secret_01")),
            ("AWS_SESSION_TOKEN", None),
        ]);
        let rep_bad =
            execute_carbon_drift_check("dev", &["us-east-2".to_owned()], clouds_drifted.to_str())
                .expect("drifted check");
        assert!(!rep_bad.all_in_sync);
        let reasons = rep_bad.regions[0].drift_reasons.join(" | ");
        assert!(reasons.contains("UPDATE_ROLLBACK_COMPLETE"), "{reasons}");
        assert!(reasons.contains("$LATEST"), "{reasons}");
        assert!(reasons.contains("endpoint drift"), "{reasons}");

        let cli_bad = Command::new(carbon_bin)
            .args([
                "deploy",
                "--env",
                "dev",
                "--regions",
                "us-east-2",
                "--check-drift",
                "--clouds-manifest",
                clouds_drifted.to_str().unwrap(),
            ])
            .output()
            .expect("carbon deploy --check-drift drifted");
        assert_eq!(cli_bad.status.code(), Some(1));
    }

    let _ = sm_shutdown.send(());
    let _ = sm_join.join();
    emu.shutdown();
}

#[test]
#[cfg(unix)]
fn installer_offline_source_dir_and_openssl_less_ecdsa_verification() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().expect("tempdir");
    let carbon_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));
    let bin_bytes = fs::read(&carbon_bin).expect("read carbon binary");
    let bin_sha = sha256_lower_hex(&bin_bytes);

    let (kp, pubkey_pem) = gen_test_signing_keypair();
    let pubkey_file = tmp.path().join("test_release_pub.pem");
    fs::write(&pubkey_file, &pubkey_pem).unwrap();

    // Build deterministic `.tar.gz` containing `carbon`
    let tar_raw = build_deterministic_single_bin_tar("carbon", &bin_bytes).unwrap();
    let mut gz = GzEncoder::new(Vec::new(), Compression::fast());
    gz.write_all(&tar_raw).unwrap();
    let targz_bytes = gz.finish().unwrap();
    let targz_sha = sha256_lower_hex(&targz_bytes);

    // Valid `carbon-clouds.json` with a real P-256 KEK SPKI pubkey
    let kek = CarbonEngineKek::from_scalar_bytes(&[0x22u8; 32]).unwrap();
    let clouds_bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "schema": "jope-carbon-clouds-v1",
        "generated_epoch_ms": 1_780_000_000_000u64,
        "clouds": {
            "prod": {
                "environment": "prod",
                "region": "us-east-2",
                "public_endpoint": "http://127.0.0.1:9",
                "engine_kek_pubkey_b64": kek.pubkey_b64(),
                "regions": [{
                    "region": "us-east-2",
                    "public_endpoint": "http://127.0.0.1:9",
                    "engine_kek_pubkey_b64": kek.pubkey_b64()
                }]
            }
        }
    }))
    .unwrap();
    let clouds_sha = sha256_lower_hex(&clouds_bytes);

    let src_dir = tmp.path().join("offline_bundle");
    fs::create_dir_all(&src_dir).unwrap();
    // Cover both Linux x86_64/aarch64 asset names so host uname always matches
    let mut sums = format!("{clouds_sha}  carbon-clouds.json\n");
    for base in ["carbon-linux-x86_64-musl", "carbon-linux-aarch64-musl"] {
        let archive_name = format!("{base}.tar.gz");
        fs::write(src_dir.join(&archive_name), &targz_bytes).unwrap();
        sums.push_str(&format!("{bin_sha}  {base}\n"));
        sums.push_str(&format!("{targz_sha}  {archive_name}\n"));
    }
    fs::write(src_dir.join("carbon-clouds.json"), &clouds_bytes).unwrap();
    fs::write(src_dir.join("SHA256SUMS.txt"), sums.as_bytes()).unwrap();

    let rng = SystemRandom::new();
    let sig = kp.sign(&rng, sums.as_bytes()).expect("sign SHA256SUMS.txt");
    let sig_b64 = base64::engine::general_purpose::STANDARD.encode(sig.as_ref());
    fs::write(src_dir.join("SHA256SUMS.txt.sig"), format!("{sig_b64}\n")).unwrap();

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let install_sh = repo_root.join("deploy/carbon/install.sh");

    // Build an `openssl`-less PATH directory containing standard POSIX utilities ONLY (zero `openssl`!)
    let no_openssl_bin = tmp.path().join("no_openssl_path");
    fs::create_dir_all(&no_openssl_bin).unwrap();
    for tool in [
        "sh",
        "bash",
        "uname",
        "tr",
        "sed",
        "grep",
        "awk",
        "cut",
        "head",
        "tail",
        "cat",
        "cp",
        "mv",
        "rm",
        "mkdir",
        "chmod",
        "mktemp",
        "tar",
        "gzip",
        "sha256sum",
        "dirname",
        "basename",
        "wc",
        "id",
        "env",
        "printf",
        "sleep",
        "date",
    ] {
        for sys_dir in ["/usr/bin", "/bin"] {
            let cand = Path::new(sys_dir).join(tool);
            if cand.is_file() {
                let _ = symlink(&cand, no_openssl_bin.join(tool));
                break;
            }
        }
    }
    assert!(
        !no_openssl_bin.join("openssl").exists(),
        "sandbox PATH must exclude openssl"
    );

    // 1. Offline `--source-dir` + `openssl`-less PATH + valid ECDSA signature -> installs cleanly (exit 0)!
    let dest_ok = tmp.path().join("bin_ok");
    let cfg_ok = tmp.path().join("cfg_ok");
    let stage_root = tmp.path().join("stage_root");
    fs::create_dir_all(&stage_root).unwrap();
    let ok_out = Command::new(no_openssl_bin.join("sh"))
        .arg(&install_sh)
        .args([
            "--source-dir",
            src_dir.to_str().unwrap(),
            "--bin-dir",
            dest_ok.to_str().unwrap(),
            "--allow-test-pubkey",
            pubkey_file.to_str().unwrap(),
        ])
        .env("PATH", &no_openssl_bin)
        .env("HOME", tmp.path())
        .env("CARBON_CONFIG_DIR", &cfg_ok)
        .env("CARBON_INSTALL_STAGE_ROOT", &stage_root)
        .output()
        .expect("run install.sh offline without openssl");
    assert!(
        ok_out.status.success(),
        "install.sh failed without openssl: stdout={} stderr={}",
        String::from_utf8_lossy(&ok_out.stdout),
        String::from_utf8_lossy(&ok_out.stderr)
    );
    assert!(dest_ok.join("carbon").is_file());
    let home_cfg = tmp.path().join(".config/jopedime");
    assert!(home_cfg.join("SHA256SUMS.txt").is_file());
    assert!(home_cfg.join("SHA256SUMS.txt.sig").is_file());
    assert!(home_cfg.join("carbon-clouds.json").is_file());

    // 2. Tamper 1 byte in `SHA256SUMS.txt.sig` under the SAME `openssl`-less PATH -> MUST fail closed!
    let tampered_src = tmp.path().join("offline_tampered");
    fs::create_dir_all(&tampered_src).unwrap();
    for entry in fs::read_dir(&src_dir).unwrap() {
        let e = entry.unwrap();
        fs::copy(e.path(), tampered_src.join(e.file_name())).unwrap();
    }
    let mut bad_sig_bytes = sig.as_ref().to_vec();
    bad_sig_bytes[10] ^= 0x5a;
    let bad_sig_b64 = base64::engine::general_purpose::STANDARD.encode(&bad_sig_bytes);
    fs::write(
        tampered_src.join("SHA256SUMS.txt.sig"),
        format!("{bad_sig_b64}\n"),
    )
    .unwrap();

    let dest_bad = tmp.path().join("bin_bad");
    let cfg_bad = tmp.path().join("cfg_bad");
    let bad_out = Command::new(no_openssl_bin.join("sh"))
        .arg(&install_sh)
        .args([
            "--source-dir",
            tampered_src.to_str().unwrap(),
            "--bin-dir",
            dest_bad.to_str().unwrap(),
            "--allow-test-pubkey",
            pubkey_file.to_str().unwrap(),
        ])
        .env("PATH", &no_openssl_bin)
        .env("HOME", tmp.path())
        .env("CARBON_CONFIG_DIR", &cfg_bad)
        .env("CARBON_INSTALL_STAGE_ROOT", &stage_root)
        .output()
        .expect("run install.sh with tampered sig without openssl");
    assert!(
        !bad_out.status.success(),
        "install.sh without openssl must reject tampered ECDSA signature"
    );
    assert!(
        !dest_bad.join("carbon").exists(),
        "install.sh must never write $BIN_DIR/carbon on failed signature check"
    );

    // 3. `CARBON_CA_BUNDLE` nonexistent file fails closed, and `install.sh --uninstall` sweeps `.upgrade.*` / `.new.*` / `.exe.old.*`
    let bad_ca_out = Command::new(no_openssl_bin.join("sh"))
        .arg(&install_sh)
        .args([
            "--source-dir",
            src_dir.to_str().unwrap(),
            "--bin-dir",
            dest_bad.to_str().unwrap(),
            "--allow-test-pubkey",
            pubkey_file.to_str().unwrap(),
        ])
        .env("PATH", &no_openssl_bin)
        .env("HOME", tmp.path())
        .env("CARBON_CA_BUNDLE", tmp.path().join("missing-corp-ca.pem"))
        .env("CARBON_INSTALL_STAGE_ROOT", &stage_root)
        .output()
        .expect("run install.sh with missing CARBON_CA_BUNDLE");
    assert!(!bad_ca_out.status.success());

    fs::write(dest_ok.join(".carbon.new.1234"), b"leftover").unwrap();
    fs::write(dest_ok.join("carbon.upgrade.1234"), b"leftover").unwrap();
    fs::write(dest_ok.join("carbon.exe.old.1234"), b"leftover").unwrap();
    let uninst_sh = Command::new(no_openssl_bin.join("sh"))
        .arg(&install_sh)
        .args(["--uninstall", "--bin-dir", dest_ok.to_str().unwrap()])
        .env("PATH", &no_openssl_bin)
        .env("HOME", tmp.path())
        .output()
        .expect("run install.sh --uninstall");
    assert!(uninst_sh.status.success());
    assert!(!dest_ok.join("carbon").exists());
    assert!(!dest_ok.join(".carbon.new.1234").exists());
    assert!(!dest_ok.join("carbon.upgrade.1234").exists());
    assert!(!dest_ok.join("carbon.exe.old.1234").exists());
}

#[test]
fn cli_ux_upgrade_aliases_tarxz_uninstall_token_stdin_doctor_completions_keep_ephemeral_and_local()
{
    let tmp = tempfile::tempdir().expect("tempdir");
    let carbon_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon"));
    let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
    let worker_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-worker"));
    let cfg_dir = tmp.path().join("ux_cfg");
    fs::create_dir_all(&cfg_dir).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&cfg_dir, fs::Permissions::from_mode(0o700)).unwrap();
    }

    // 1. Stale `.exe.old.*` upgrade backup sweep on startup + `self-update` / `update` aliases + `CARBON_GITHUB_RELEASES_API_URL`
    let copied_bin_dir = tmp.path().join("copied_cli");
    fs::create_dir_all(&copied_bin_dir).unwrap();
    let copied_bin = copied_bin_dir.join("carbon");
    fs::copy(&carbon_bin, &copied_bin).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&copied_bin, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let stale_bak = copied_bin_dir.join("carbon.exe.old.99999");
    fs::write(&stale_bak, b"stale windows upgrade backup").unwrap();

    // Mock GitHub `/releases?per_page=20` returning a non-Carbon tag first (`jope-engine-v9.9.9`),
    // a draft tag second (`carbon-cli-v9.9.9-draft`), and valid `carbon-cli-v0.0.0-offline-probe` third!
    let gh_app = Router::new().route(
        "/releases",
        get(|| async move {
            let body = serde_json::json!([
                {"tag_name": "jope-engine-v9.9.9", "draft": false, "prerelease": false},
                {"tag_name": "carbon-cli-v9.9.9-draft", "draft": true, "prerelease": false},
                {"tag_name": "carbon-cli-v0.0.0-offline-probe", "draft": false, "prerelease": false}
            ])
            .to_string();
            (StatusCode::OK, body)
        }),
    );
    let (gh_addr, gh_shutdown, gh_join) = spawn_loopback_axum("mock-gh-releases", gh_app);
    let gh_api_url = format!("http://{gh_addr}/releases");
    {
        let (_kp_gh, gh_pub_pem) = gen_test_signing_keypair();
        let _env = ScopedProcessEnv::set_opt(&[
            ("CARBON_GITHUB_RELEASES_API_URL", Some(&gh_api_url)),
            ("DO_NOT_TRACK", Some("1")),
        ]);
        let probe_err = execute_self_upgrade_with_options(None, None, &gh_pub_pem, true, true)
            .expect_err("offline probe tag should fail 404 or network after tag discovery");
        assert!(
            probe_err.contains("carbon-cli-v0.0.0-offline-probe"),
            "discover_latest_carbon_cli_tag must skip non-carbon & draft tags and select carbon-cli-v0.0.0-offline-probe, got: {probe_err}"
        );
    }
    let _ = gh_shutdown.send(());
    let _ = gh_join.join();
    let _ = CARBON_PUBLISHED_INSTALLER_TAG;

    // `.tar.xz` signed self-upgrade extraction + `self-update` / `update` CLI aliases + `.old.*` backup sweep!
    // Uses intentional post-extract binary SHA mismatch on replace so Cargo's running test runner executable is never overwritten!
    let fake_xz_bin = b"#!/bin/sh\necho upgraded-from-tarxz\n";
    let fake_xz_bin_sha = sha256_lower_hex(fake_xz_bin);
    let tar_bytes = build_deterministic_single_bin_tar("carbon", fake_xz_bin).unwrap();
    if let Some(tarxz_bytes) = try_external_stream_compress("xz", &["-c", "-1"], &tar_bytes) {
        let (kp, pubkey_pem) = gen_test_signing_keypair();
        let test_pub_path = tmp.path().join("alias_test_pub.pem");
        fs::write(&test_pub_path, &pubkey_pem).unwrap();
        let base_asset = if CARBON_BUILD_TARGET.contains("aarch64") {
            "carbon-linux-aarch64-musl"
        } else {
            "carbon-linux-x86_64-musl"
        };
        let tarxz_asset = format!("{base_asset}.tar.xz");
        let tarxz_sha = sha256_lower_hex(&tarxz_bytes);
        let sums = format!(
            "{}  {base_asset}\n{tarxz_sha}  {tarxz_asset}\n",
            "f".repeat(64)
        );
        let sig = kp
            .sign(&SystemRandom::new(), sums.as_bytes())
            .expect("sign tar.xz sums");
        let sig_b64 = base64::engine::general_purpose::STANDARD.encode(sig.as_ref());

        let upg_app = Router::new()
            .route(
                "/SHA256SUMS.txt",
                get({
                    let s = sums.clone();
                    move || async move { (StatusCode::OK, s) }
                }),
            )
            .route(
                "/SHA256SUMS.txt.sig",
                get(move || async move { (StatusCode::OK, sig_b64) }),
            )
            .route(
                &format!("/{tarxz_asset}"),
                get(move || async move { (StatusCode::OK, tarxz_bytes) }),
            );
        let (u_addr, u_shut, u_join) = spawn_loopback_axum("tarxz-upgrade", upg_app);
        let base_u = format!("http://{u_addr}");

        for alias in ["self-update", "update"] {
            let chk = Command::new(&copied_bin)
                .args([
                    alias,
                    "--check",
                    "--version",
                    "carbon-cli-v1.2.0",
                    "--base-url",
                    &base_u,
                    "--allow-test-pubkey",
                    test_pub_path.to_str().unwrap(),
                ])
                .env("CARBON_CONFIG_DIR", &cfg_dir)
                .output()
                .expect("upgrade alias --check");
            assert!(
                chk.status.success(),
                "carbon {alias} --check failed: stderr={}",
                String::from_utf8_lossy(&chk.stderr)
            );
            let v: serde_json::Value = serde_json::from_slice(&chk.stdout).unwrap();
            assert_eq!(v["target_release_tag"].as_str(), Some("carbon-cli-v1.2.0"));
            assert_eq!(v["signature_verified"], true);
        }
        assert!(
            !stale_bak.exists(),
            "startup sweep must clean up stale <exe>.old.* backup files"
        );

        let xz_err = execute_self_upgrade_with_options(
            Some(&base_u),
            Some("carbon-cli-v1.2.0"),
            &pubkey_pem,
            false,
            true,
        )
        .expect_err("expected post-tar.xz-extraction SHA verification check");
        assert!(
            xz_err.contains(&fake_xz_bin_sha),
            "expected extracted fake_xz_bin SHA ({fake_xz_bin_sha}) proving .tar.xz extraction succeeded, got: {xz_err}"
        );
        let _ = u_shut.send(());
        let _ = u_join.join();
    }

    // 2. Dynamic `carbon completions` across all 4 shells (`bash -n` + dynamically discovered subcommands/aliases/flags)
    for shell in ["bash", "zsh", "fish", "powershell"] {
        let out = Command::new(&carbon_bin)
            .args(["completions", shell])
            .output()
            .unwrap();
        assert!(out.status.success());
        let text = String::from_utf8(out.stdout).unwrap();
        for expected in [
            "uninstall",
            "self-update",
            "update",
            "--check-drift",
            "--token-stdin",
            "--real-workers",
            "--ephemeral",
            "--keep",
            "--local",
        ] {
            let needle = if shell == "fish" {
                expected.trim_start_matches("--")
            } else {
                expected
            };
            assert!(
                text.contains(needle),
                "dynamic {shell} completions missing '{needle}'"
            );
        }
        if shell == "bash" {
            let f = tmp.path().join("dyn_comp.bash");
            fs::write(&f, text.as_bytes()).unwrap();
            assert!(Command::new("bash")
                .arg("-n")
                .arg(&f)
                .status()
                .unwrap()
                .success());
        }
    }

    // 3. `carbon doctor --offline` (`signing_pub_fingerprint_ok`, `path_shadowed_by`) + bare `carbon` TTY quick-start banner
    let mut emu_cfg = CarbonEmuConfig::for_dir(tmp.path().join("ux_emu"));
    emu_cfg.force_hybrid_fanout = false;
    let mut emu = CarbonLocalLambdaEmulator::start(emu_cfg).expect("start ux_emu");
    let ux_clouds = tmp.path().join("ux_clouds.json");
    fs::write(
        &ux_clouds,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "jope-carbon-clouds-v1",
            "generated_epoch_ms": 1_780_000_000_000u64,
            "clouds": {
                "dev": {
                    "environment": "dev",
                    "region": "us-east-2",
                    "public_endpoint": emu.base_url(),
                    "engine_kek_pubkey_b64": emu.engine_kek_pubkey_b64(),
                    "regions": [{
                        "region": "us-east-2",
                        "public_endpoint": emu.base_url(),
                        "engine_kek_pubkey_b64": emu.engine_kek_pubkey_b64()
                    }]
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let shadow_dir = tmp.path().join("shadow_path");
    fs::create_dir_all(&shadow_dir).unwrap();
    let shadow_carbon = shadow_dir.join("carbon");
    fs::write(&shadow_carbon, b"#!/bin/sh\nexit 0\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&shadow_carbon, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let shadowed_path = format!("{}:{orig_path}", shadow_dir.display());
    let doc_out = Command::new(&carbon_bin)
        .args([
            "doctor",
            "--offline",
            "--cloud",
            "dev",
            "--clouds-manifest",
            ux_clouds.to_str().unwrap(),
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env("PATH", &shadowed_path)
        .output()
        .expect("doctor --offline");
    assert!(
        doc_out.status.success(),
        "doctor --offline failed: stdout={} stderr={}",
        String::from_utf8_lossy(&doc_out.stdout),
        String::from_utf8_lossy(&doc_out.stderr)
    );
    let doc_json: serde_json::Value = serde_json::from_slice(&doc_out.stdout).unwrap();
    assert_eq!(doc_json["checks"][0]["signing_pub_fingerprint_ok"], true);
    assert!(doc_json["checks"][0]["path_shadowed_by"].is_string());

    // Bare `carbon` non-TTY -> JSON version; bare `carbon` PTY -> human quick-start banner!
    let bare_json = Command::new(&carbon_bin).output().unwrap();
    assert!(bare_json.status.success());
    let bare_val: serde_json::Value = serde_json::from_slice(&bare_json.stdout).unwrap();
    assert!(bare_val["version"].is_string());

    if Path::new("/usr/bin/script").is_file() {
        let pty_bare = Command::new("/usr/bin/script")
            .args(["-qec", &format!("'{}'", carbon_bin.display()), "/dev/null"])
            .env("CARBON_CONFIG_DIR", &cfg_dir)
            .output()
            .expect("pty bare carbon");
        assert!(pty_bare.status.success());
        let pty_txt = String::from_utf8_lossy(&pty_bare.stdout);
        assert!(
            pty_txt.contains("Quick-Start") || pty_txt.contains("carbon inspect"),
            "expected interactive quick-start banner on TTY bare `carbon`, got: {pty_txt}"
        );
    }

    // 4. `carbon login --token-stdin`, `carbon whoami --token-stdin`, `--keep` vs `--ephemeral`
    let pro_tok = emu
        .mint_capability_token("stdin_tenant", CarbonTierWire::Pro, false)
        .unwrap();

    let mut login_child = Command::new(&carbon_bin)
        .args([
            "login",
            "--endpoint",
            &emu.base_url(),
            "--cloud",
            "dev",
            "--token-stdin",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn login --token-stdin");
    login_child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(format!("{pro_tok}\n").as_bytes())
        .unwrap();
    let login_res = login_child.wait_with_output().unwrap();
    assert!(
        login_res.status.success(),
        "login --token-stdin failed: {}",
        String::from_utf8_lossy(&login_res.stderr)
    );

    let mut who_child = Command::new(&carbon_bin)
        .args([
            "whoami",
            "--cloud",
            "dev",
            "--clouds-manifest",
            ux_clouds.to_str().unwrap(),
            "--token-stdin",
            "--json",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn whoami --token-stdin");
    who_child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(format!("{pro_tok}\n").as_bytes())
        .unwrap();
    let who_res = who_child.wait_with_output().unwrap();
    assert!(
        who_res.status.success(),
        "whoami --token-stdin failed: {}",
        String::from_utf8_lossy(&who_res.stderr)
    );

    let csv_file = tmp.path().join("keep_test.csv");
    let mut csv_txt = String::from("x1,x2,y\n");
    for i in 0..48 {
        let x1 = i as f64 * 0.2;
        let x2 = (i % 5) as f64;
        csv_txt.push_str(&format!("{x1},{x2},{}\n", 2.0 * x1 + x2));
    }
    fs::write(&csv_file, &csv_txt).unwrap();

    // Positional shorthand `carbon <csv> --keep` -> retains handle (`shredded_after_run == false`)
    let keep_out = Command::new(&carbon_bin)
        .args([csv_file.to_str().unwrap(), "--keep"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env("CARBON_ENDPOINT", emu.base_url())
        .output()
        .expect("carbon <csv> --keep");
    assert!(
        keep_out.status.success(),
        "carbon --keep failed: {}",
        String::from_utf8_lossy(&keep_out.stderr)
    );
    let keep_json: serde_json::Value = serde_json::from_slice(&keep_out.stdout).unwrap();
    assert_eq!(keep_json["shredded_after_run"], false);
    let kept_run_id = keep_json["run_id"].as_str().unwrap();
    assert!(cfg_dir.join(format!("runs/{kept_run_id}.json")).is_file());

    // `carbon fit <csv> --ephemeral` -> shreds immediately (`shredded_after_run == true`)
    let eph_out = Command::new(&carbon_bin)
        .args([
            "fit",
            csv_file.to_str().unwrap(),
            "--endpoint",
            &emu.base_url(),
            "--free",
            "--ephemeral",
            "--json",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("carbon fit --ephemeral");
    assert!(eph_out.status.success());
    let eph_json: serde_json::Value = serde_json::from_slice(&eph_out.stdout).unwrap();
    assert_eq!(eph_json["shredded_after_run"], true);

    emu.shutdown();

    // Verify bounded `.csv.xz` tabular conditioning (`carbon inspect <file.csv.xz> --json`)
    if let Some(csv_xz_bytes) =
        try_external_stream_compress("xz", &["-c", "-1"], &fs::read(&csv_file).unwrap())
    {
        let csv_xz_path = tmp.path().join("compressed_sample.csv.xz");
        fs::write(&csv_xz_path, csv_xz_bytes).unwrap();
        let insp_xz = Command::new(&carbon_bin)
            .args(["inspect", csv_xz_path.to_str().unwrap(), "--json"])
            .env("CARBON_CONFIG_DIR", &cfg_dir)
            .output()
            .expect("carbon inspect .csv.xz");
        assert!(
            insp_xz.status.success(),
            "carbon inspect .csv.xz failed: {}",
            String::from_utf8_lossy(&insp_xz.stderr)
        );
        let insp_doc: serde_json::Value = serde_json::from_slice(&insp_xz.stdout).unwrap();
        assert_eq!(insp_doc["raw_rows"], 48);
    }

    // 5. `carbon emu --real-workers --self-test`, foreground `carbon emu --listen` (`CARBON_EMU_MAX_MS`),
    //    and `carbon quickstart --local` + `carbon fit <csv> --local`!
    let emu_real_self = Command::new(&carbon_bin)
        .args([
            "emu",
            "--self-test",
            "--real-workers",
            "--edge-bin",
            edge_bin.to_str().unwrap(),
            "--worker-bin",
            worker_bin.to_str().unwrap(),
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("carbon emu --self-test --real-workers");
    assert!(
        emu_real_self.status.success(),
        "emu --self-test --real-workers failed: stdout={} stderr={}",
        String::from_utf8_lossy(&emu_real_self.stdout),
        String::from_utf8_lossy(&emu_real_self.stderr)
    );
    let self_json: serde_json::Value = serde_json::from_slice(&emu_real_self.stdout).unwrap();
    assert_eq!(self_json["real_workers"], true);
    assert_eq!(self_json["self_test"].as_str(), Some("ok"));

    let emu_fg = Command::new(&carbon_bin)
        .args(["emu", "--listen", "127.0.0.1:0"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .env("CARBON_EMU_MAX_MS", "120")
        .output()
        .expect("foreground carbon emu");
    assert!(emu_fg.status.success());

    let qs_local = Command::new(&carbon_bin)
        .args(["quickstart", "--local"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("quickstart --local");
    assert!(
        qs_local.status.success(),
        "quickstart --local failed: stderr={}",
        String::from_utf8_lossy(&qs_local.stderr)
    );

    let fit_local = Command::new(&carbon_bin)
        .args([
            "fit",
            csv_file.to_str().unwrap(),
            "--local",
            "--ephemeral",
            "--json",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("fit --local");
    assert!(
        fit_local.status.success(),
        "fit --local failed: stderr={}",
        String::from_utf8_lossy(&fit_local.stderr)
    );

    // 6. `carbon uninstall`: Cargo `target/` refusal (exit 2), `--keep-data`, non-TTY `--purge-all` refusal (exit 2),
    //    and `--purge-all --yes` cryptographic zero-overwrite & purge!
    let refuse_target = Command::new(&carbon_bin)
        .args(["uninstall", "--keep-data", "--yes"])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("uninstall cargo target refusal");
    assert_eq!(refuse_target.status.code(), Some(2));

    let bin_u1 = tmp.path().join("uninstall_bin_1");
    fs::copy(&carbon_bin, &bin_u1).unwrap();
    let keep_u = Command::new(&carbon_bin)
        .args([
            "uninstall",
            "--bin-path",
            bin_u1.to_str().unwrap(),
            "--keep-data",
            "--yes",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("uninstall --keep-data");
    assert!(keep_u.status.success());
    assert!(!bin_u1.exists());
    assert!(cfg_dir.join("carbon-credentials.json").is_file());

    // Write a dummy handle to verify `--purge-all --yes` shreds handles + credentials + config dir
    {
        let _env =
            ScopedProcessEnv::set_opt(&[("CARBON_CONFIG_DIR", Some(cfg_dir.to_str().unwrap()))]);
        let dummy_h = CarbonSavedModelHandle {
            schema: CARBON_MODEL_HANDLE_SCHEMA.to_owned(),
            created_epoch_ms: 1_780_000_000_000,
            run_id: "run_uninstall_01".to_owned(),
            dataset_id: "ds_uninstall_01".to_owned(),
            tenant_id: "t_uninstall".to_owned(),
            tier: CarbonTierWire::Free,
            cloud: None,
            endpoint: "http://127.0.0.1:9".to_owned(),
            region: "us-east-2".to_owned(),
            engine_kek_pubkey_b64: None,
            encrypted_cuen: false,
            u_dek_b64: None,
            capability_token: None,
            encoder: CarbonTabularEncoder {
                raw_feature_columns: vec!["x1".to_owned()],
                dropped_columns: Vec::new(),
                transforms: Vec::new(),
                encoded_feature_names: vec!["x1".to_owned()],
                target_column: "y".to_owned(),
                delimiter: ',',
                had_header: true,
                extra_na_tokens: Vec::new(),
                inf_as_nan: false,
            },
            fit_rows: 10,
            holdout_rows: 2,
            holdout_rmse: Some(0.1),
            holdout_mae: Some(0.1),
            holdout_r2: Some(0.9),
            objective_name: None,
            objective_direction: None,
            feature_importances: Vec::new(),
        };
        persist_model_handle_with_options(&dummy_h, None, false).unwrap();
    }

    let bin_u2 = tmp.path().join("uninstall_bin_2");
    fs::copy(&carbon_bin, &bin_u2).unwrap();
    let refuse_purge_no_yes = Command::new(&carbon_bin)
        .args([
            "uninstall",
            "--bin-path",
            bin_u2.to_str().unwrap(),
            "--purge-all",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .stdin(Stdio::null())
        .output()
        .expect("uninstall --purge-all without --yes");
    assert_eq!(refuse_purge_no_yes.status.code(), Some(2));
    assert!(bin_u2.exists());

    let purge_ok = Command::new(&carbon_bin)
        .args([
            "uninstall",
            "--bin-path",
            bin_u2.to_str().unwrap(),
            "--purge-all",
            "--yes",
        ])
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("uninstall --purge-all --yes");
    assert!(
        purge_ok.status.success(),
        "uninstall --purge-all --yes failed: {}",
        String::from_utf8_lossy(&purge_ok.stderr)
    );
    assert!(!bin_u2.exists());
    assert!(!cfg_dir.exists());
}
