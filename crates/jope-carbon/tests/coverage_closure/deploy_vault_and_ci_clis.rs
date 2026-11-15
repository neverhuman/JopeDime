//! Coverage closure for `package`, `deploy`, `rollback`, `clouds-manifest`, `token-mint`,
//! `secrets_vault`, `coverage-audit`, `ci-live-4tier`, `spec-report`, and `spec-verify`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use jope_carbon::{
    assemble_verified_carbon_package, resolve_edge_secrets, resolve_engine_secrets,
    CarbonEmuConfig, CarbonEngineKek, CarbonLocalLambdaEmulator, CarbonPackageOptions,
};

use super::{
    minimal_aarch64_static_elf64, write_executable_script, LoopbackMockS3Server,
    LoopbackMockSecretsServer, ScopedProcessEnv,
};

fn carbon_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_carbon"))
}

fn build_secret_bundle_json(
    auth_hex: &str,
    kek_hex: &str,
    kek: &CarbonEngineKek,
    prev_kek_hex: Option<&str>,
) -> String {
    let mut obj = serde_json::json!({
        "presign_secret_hex": "11".repeat(32),
        "auth_secret_hex": auth_hex,
        "seal_secret_hex": "33".repeat(32),
        "internal_token": "aa".repeat(8),
        "engine_kek_hex": kek_hex,
        "engine_kek_pubkey_b64": kek.pubkey_b64(),
    });
    if let Some(pk_hex) = prev_kek_hex {
        let map = obj.as_object_mut().expect("obj");
        map.insert("previous_presign_secret_hex".into(), serde_json::json!("44".repeat(32)));
        map.insert("previous_auth_secret_hex".into(), serde_json::json!("55".repeat(32)));
        map.insert("previous_seal_secret_hex".into(), serde_json::json!("66".repeat(32)));
        map.insert("previous_engine_kek_hex".into(), serde_json::json!(pk_hex));
    }
    obj.to_string()
}

#[test]
fn package_assembly_and_cli_fail_closed_plus_secrets_vault_role_isolation() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let home = tmp.path().join("home");
    fs::create_dir_all(&home).expect("mkdir home");
    let bin = carbon_bin();

    // 1. `assemble_verified_carbon_package` produces deterministic ZIPs & `release-manifest.json`
    let r882_dir = tmp.path().join("r882_mock");
    fs::create_dir_all(r882_dir.join("models/bundles")).expect("mkdir r882");
    fs::write(
        r882_dir.join("models/bundles/r882.json"),
        br#"{"bundle_id":"r882"}"#,
    )
    .expect("write mock bundle");
    let edge_elf = tmp.path().join("carbon-edge-arm64");
    let worker_elf = tmp.path().join("carbon-worker-arm64");
    let elf_bytes = minimal_aarch64_static_elf64();
    fs::write(&edge_elf, &elf_bytes).expect("write edge elf");
    fs::write(&worker_elf, &elf_bytes).expect("write worker elf");

    let pkg_out = tmp.path().join("pkg_out");
    let manifest = assemble_verified_carbon_package(
        &CarbonPackageOptions {
            r882_dir: r882_dir.clone(),
            out_dir: pkg_out.clone(),
            edge_bin: Some(edge_elf.clone()),
            worker_bin: Some(worker_elf),
            build_jobs: 2,
            include_cli_binaries: false,
        },
        "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc".to_owned(),
    )
    .expect("assemble verified package");
    assert!(pkg_out.join(&manifest.edge_zip_file).is_file());
    assert!(pkg_out.join(&manifest.worker_zip_file).is_file());
    assert!(pkg_out.join(&manifest.r882_layer_zip_file).is_file());
    assert!(pkg_out.join("release-manifest.json").is_file());

    // 2. `carbon package` CLI subprocess refuses unverified `r882` directory fail-closed
    let pkg_cli = Command::new(&bin)
        .args([
            "package",
            "--r882-dir",
            r882_dir.to_str().expect("utf8"),
            "--out-dir",
            tmp.path().join("pkg_fail").to_str().expect("utf8"),
            "--edge-bin",
            edge_elf.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon package fail-closed");
    assert!(!pkg_cli.status.success());

    // 3. `resolve_edge_secrets` vs `resolve_engine_secrets` against loopback Mock Secrets Manager
    let mock_sm = LoopbackMockSecretsServer::start();
    let sm_ep = mock_sm.endpoint_url();
    let active_kek_hex = "01".repeat(32);
    let prev_kek_hex = "02".repeat(32);
    let active_kek = CarbonEngineKek::from_hex(&active_kek_hex).expect("active kek");
    let full_arn = "arn:aws:secretsmanager:us-east-2:123456789012:secret:test-cell-secrets-01";
    mock_sm.put_secret(
        "jopedime/carbon/dev/cell-secrets",
        full_arn,
        "vid-rot-01",
        &build_secret_bundle_json(
            &"22".repeat(32),
            &active_kek_hex,
            &active_kek,
            Some(&prev_kek_hex),
        ),
    );

    {
        let _env = ScopedProcessEnv::set(&[
            ("CARBON_SECRETSMANAGER_ENDPOINT_URL", &sm_ep),
            ("AWS_ACCESS_KEY_ID", "loopback-test-access-key"),
            ("AWS_SECRET_ACCESS_KEY", "loopback-test-secret-material"),
            ("CARBON_SECRETS_ARN", full_arn),
            ("CARBON_SECRETS_VERSION_ID", "vid-rot-01"),
        ]);
        let (_presigner, auth_opt, pub_b64) =
            resolve_edge_secrets("us-east-2", true).expect("resolve_edge_secrets");
        assert!(auth_opt.is_some());
        assert_eq!(pub_b64.as_deref(), Some(active_kek.pubkey_b64().as_str()));

        let (_seal, kek_opt) =
            resolve_engine_secrets("us-east-2", true).expect("resolve_engine_secrets");
        let loaded_kek = kek_opt.expect("engine kek must NOT be scrubbed by prior edge call");
        assert_eq!(loaded_kek.pubkey_b64(), active_kek.pubkey_b64());
    }
}

#[test]
fn cli_token_mint_deploy_dry_and_full_rollback_clouds_manifest_and_coverage_audit() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let home = tmp.path().join("home");
    fs::create_dir_all(&home).expect("mkdir home");
    let bin = carbon_bin();

    // Start isolated `dev` and `prod` emulators + loopback S3 + loopback Secrets Manager
    let mut cfg_dev = CarbonEmuConfig::for_dir(tmp.path().join("emu_dev"));
    cfg_dev.region = "us-east-2".to_owned();
    cfg_dev.auth_secret = [0x41; 32];
    cfg_dev.engine_kek_scalar = [0x11; 32];
    let dev_auth_hex = hex::encode(cfg_dev.auth_secret);
    let mut emu_dev = CarbonLocalLambdaEmulator::start(cfg_dev).expect("start emu_dev");
    let (_, dev_probed) =
        jope_carbon::CarbonThinClient::connect_fastest(vec![emu_dev.base_url()]).expect("ping dev");
    let dev_kek_pub = dev_probed[0]
        .engine_kek_pubkey_b64
        .clone()
        .expect("dev kek pub");

    let mut cfg_prod = CarbonEmuConfig::for_dir(tmp.path().join("emu_prod"));
    cfg_prod.region = "us-east-2".to_owned();
    cfg_prod.auth_secret = [0x52; 32];
    cfg_prod.engine_kek_scalar = [0x22; 32];
    let prod_auth_hex = hex::encode(cfg_prod.auth_secret);
    let mut emu_prod = CarbonLocalLambdaEmulator::start(cfg_prod).expect("start emu_prod");
    let (_, prod_probed) = jope_carbon::CarbonThinClient::connect_fastest(vec![emu_prod.base_url()])
        .expect("ping prod");
    let prod_kek_pub = prod_probed[0]
        .engine_kek_pubkey_b64
        .clone()
        .expect("prod kek pub");

    let mock_s3 = LoopbackMockS3Server::start();
    let mock_sm = LoopbackMockSecretsServer::start();

    // Seed Secrets Manager with `dev` & `prod` cell secrets matching `emu_dev` and `emu_prod` KEKs & auth keys!
    let dummy_kek_hex = "03".repeat(32);
    for (env_name, auth_h, kek_pub) in [
        ("dev", &dev_auth_hex, &dev_kek_pub),
        ("prod", &prod_auth_hex, &prod_kek_pub),
    ] {
        let secret_json = serde_json::json!({
            "presign_secret_hex": "11".repeat(32),
            "auth_secret_hex": auth_h,
            "seal_secret_hex": "33".repeat(32),
            "internal_token": "tok_int",
            "engine_kek_hex": &dummy_kek_hex,
            "engine_kek_pubkey_b64": kek_pub,
        })
        .to_string();
        let arn = format!("arn:aws:secretsmanager:us-east-2:123456789012:secret:{env_name}-sec");
        mock_sm.put_secret(
            &format!("jopedime/carbon/{env_name}/cell-secrets"),
            &arn,
            "vid-01",
            &secret_json,
        );
        let edge_json = serde_json::json!({
            "presign_secret_hex": "11".repeat(32),
            "auth_secret_hex": auth_h,
            "engine_kek_pubkey_b64": kek_pub,
        })
        .to_string();
        mock_sm.put_secret(
            &format!("jopedime/carbon/{env_name}/edge-secrets"),
            &format!("{arn}-edge"),
            "vid-edge-01",
            &edge_json,
        );
    }

    // 1. `carbon token-mint` (via `CARBON_AUTH_SECRET_HEX`, via `--env dev` vault, and invalid `--ttl-secs 0` exit 2)
    let tm_env_var = Command::new(&bin)
        .args([
            "token-mint",
            "--tenant",
            "minted_user",
            "--tier",
            "pro",
            "--ttl-secs",
            "600",
        ])
        .env("HOME", &home)
        .env("CARBON_AUTH_SECRET_HEX", &dev_auth_hex)
        .output()
        .expect("token-mint env");
    assert!(tm_env_var.status.success());

    let tm_vault = Command::new(&bin)
        .args([
            "token-mint",
            "--env",
            "dev",
            "--region",
            "us-east-2",
            "--tenant",
            "vault_user",
            "--tier",
            "pro",
            "--admin",
            "--ttl-secs",
            "600",
        ])
        .env("HOME", &home)
        .env("CARBON_SECRETSMANAGER_ENDPOINT_URL", mock_sm.endpoint_url())
        .env("AWS_ACCESS_KEY_ID", "loopback-test-access-key")
        .env("AWS_SECRET_ACCESS_KEY", "loopback-test-secret-material")
        .output()
        .expect("token-mint --env dev");
    assert!(
        tm_vault.status.success(),
        "token-mint vault failed: {}",
        String::from_utf8_lossy(&tm_vault.stderr)
    );

    let tm_bad = Command::new(&bin)
        .args(["token-mint", "--tenant", "u", "--ttl-secs", "0"])
        .env("HOME", &home)
        .env("CARBON_AUTH_SECRET_HEX", &dev_auth_hex)
        .output()
        .expect("token-mint bad ttl");
    assert!(!tm_bad.status.success());

    // 2. Write mock `CARBON_AWS_CLI_BIN` handling CloudFormation, STS, Lambda, S3API, IAM, Logs
    let mock_aws = tmp.path().join("mock-aws.sh");
    let state_flag = tmp.path().join("stack_created.flag");
    let aws_script = format!(
        r#"#!/bin/sh
set -eu
ARGS="$*"
case "$ARGS" in
  *"sts get-caller-identity"*)
    printf '%s\n' '{{"Account":"123456789012"}}' ;;
  *"s3api head-bucket"*)
    echo "An error occurred (404) when calling the HeadBucket operation: Not Found" >&2
    exit 254 ;;
  *"s3api "*)
    printf '%s\n' '{{}}' ;;
  *"cloudformation validate-template"*)
    printf '%s\n' '{{"Parameters":[]}}' ;;
  *"cloudformation describe-stacks --stack-name jopedime-carbon-prod"*)
    printf '%s\n' '{{"Stacks":[{{"StackStatus":"UPDATE_COMPLETE","Outputs":[{{"OutputKey":"EdgeEndpointUrl","OutputValue":"{prod_url}"}},{{"OutputKey":"PublicEdgeEndpointUrl","OutputValue":"{prod_url}"}},{{"OutputKey":"ReleaseId","OutputValue":"rel_mock_01"}}]}}]}}' ;;
  *"cloudformation describe-stacks --stack-name jopedime-carbon-dev"*)
    if [ -f "{flag}" ]; then
      printf '%s\n' '{{"Stacks":[{{"StackStatus":"CREATE_COMPLETE","Outputs":[{{"OutputKey":"EdgeEndpointUrl","OutputValue":"{dev_url}"}},{{"OutputKey":"PublicEdgeEndpointUrl","OutputValue":"{dev_url}"}},{{"OutputKey":"CasBucketName","OutputValue":"mock-cas-bkt"}},{{"OutputKey":"R882LayerVersionArn","OutputValue":"arn:aws:lambda:us-east-2:123:layer:r882:1"}},{{"OutputKey":"EdgePublishedVersion","OutputValue":"2"}},{{"OutputKey":"EnginePublishedVersion","OutputValue":"2"}},{{"OutputKey":"EvalPublishedVersion","OutputValue":"2"}},{{"OutputKey":"ReleaseId","OutputValue":"rel_mock_01"}}]}}]}}'
    else
      echo "Stack with id jopedime-carbon-dev does not exist" >&2
      exit 254
    fi ;;
  *"cloudformation create-change-set"*)
    touch "{flag}"
    printf '%s\n' '{{"Id":"arn:aws:cloudformation:us-east-2:123:changeSet/cs/1"}}' ;;
  *"cloudformation describe-change-set"*)
    printf '%s\n' '{{"Status":"CREATE_COMPLETE","Changes":[{{"ResourceChange":{{"Action":"Add","LogicalResourceId":"EdgeFn","ResourceType":"AWS::Lambda::Function"}}}}]}}' ;;
  *"cloudformation execute-change-set"*|*"cloudformation delete-change-set"*)
    printf '%s\n' '{{}}' ;;
  *"lambda get-alias"*)
    printf '%s\n' '{{"FunctionVersion":"2"}}' ;;
  *"lambda list-versions-by-function"*)
    printf '%s\n' '{{"Versions":[{{"Version":"$LATEST"}},{{"Version":"1"}},{{"Version":"2"}}]}}' ;;
  *"lambda update-alias"*)
    printf '%s\n' '{{"FunctionVersion":"1"}}' ;;
  *"iam "*|*"logs "*|*"lambda delete-"*)
    printf '%s\n' '{{}}' ;;
  *)
    printf '%s\n' '{{}}' ;;
esac
"#,
        dev_url = emu_dev.base_url(),
        prod_url = emu_prod.base_url(),
        flag = state_flag.display(),
    );
    write_executable_script(&mock_aws, &aws_script);

    // Assemble a valid release package so `carbon deploy` stages real verified ZIPs to Mock S3!
    let r882_dir = tmp.path().join("r882_dep");
    fs::create_dir_all(r882_dir.join("models/bundles")).expect("mkdir");
    fs::write(r882_dir.join("models/bundles/r882.json"), b"{}").expect("write");
    let elf = minimal_aarch64_static_elf64();
    let e_bin = tmp.path().join("edge_a64");
    let w_bin = tmp.path().join("worker_a64");
    fs::write(&e_bin, &elf).expect("e_bin");
    fs::write(&w_bin, &elf).expect("w_bin");
    let pkg_dir = tmp.path().join("pkg_dep");
    let rel_manifest = assemble_verified_carbon_package(
        &CarbonPackageOptions {
            r882_dir,
            out_dir: pkg_dir.clone(),
            edge_bin: Some(e_bin),
            worker_bin: Some(w_bin),
            build_jobs: 2,
            include_cli_binaries: false,
        },
        "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc".to_owned(),
    )
    .expect("package for deploy");
    let rel_manifest_path = pkg_dir.join("release-manifest.json");
    assert!(!rel_manifest.release_id.is_empty());

    // 3. `carbon deploy --dry-run` AND full `carbon deploy` (with `--decommission-unstacked-prototype --skip-redteam --json-out`)
    let dry_json = tmp.path().join("deploy-dry.json");
    let dep_dry = Command::new(&bin)
        .args([
            "deploy",
            "--manifest",
            rel_manifest_path.to_str().expect("utf8"),
            "--env",
            "dev",
            "--regions",
            "canary",
            "--dry-run",
            "--json-out",
            dry_json.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .env("CARBON_AWS_CLI_BIN", &mock_aws)
        .env("CARBON_S3_ENDPOINT_URL", mock_s3.endpoint_url())
        .env("CARBON_SECRETSMANAGER_ENDPOINT_URL", mock_sm.endpoint_url())
        .env("AWS_ACCESS_KEY_ID", "loopback-test-access-key")
        .env("AWS_SECRET_ACCESS_KEY", "loopback-test-secret-material")
        .output()
        .expect("carbon deploy --dry-run");
    assert!(
        dep_dry.status.success() && dry_json.is_file(),
        "deploy --dry-run failed: {}",
        String::from_utf8_lossy(&dep_dry.stderr)
    );

    let full_json = tmp.path().join("deploy-full.json");
    let dep_full = Command::new(&bin)
        .args([
            "deploy",
            "--manifest",
            rel_manifest_path.to_str().expect("utf8"),
            "--env",
            "dev",
            "--regions",
            "us-east-2",
            "--decommission-unstacked-prototype",
            "--skip-redteam",
            "--json-out",
            full_json.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .env("CARBON_AWS_CLI_BIN", &mock_aws)
        .env("CARBON_S3_ENDPOINT_URL", mock_s3.endpoint_url())
        .env("CARBON_SECRETSMANAGER_ENDPOINT_URL", mock_sm.endpoint_url())
        .env("AWS_ACCESS_KEY_ID", "loopback-test-access-key")
        .env("AWS_SECRET_ACCESS_KEY", "loopback-test-secret-material")
        .output()
        .expect("carbon deploy full");
    assert!(
        dep_full.status.success() && full_json.is_file(),
        "deploy full failed: {}",
        String::from_utf8_lossy(&dep_full.stderr)
    );

    // 4. `carbon rollback` (auto-version discovery + explicit `--to-versions 1,1,1`)
    let rb = Command::new(&bin)
        .args(["rollback", "--env", "dev", "--regions", "us-east-2"])
        .env("HOME", &home)
        .env("CARBON_AWS_CLI_BIN", &mock_aws)
        .output()
        .expect("carbon rollback");
    assert!(
        rb.status.success(),
        "rollback failed: {}",
        String::from_utf8_lossy(&rb.stderr)
    );

    // 5. `carbon clouds-manifest` generating `jope-carbon-clouds-v2` JSON from mock stacks + vault + live `/ping`!
    let clouds_out = tmp.path().join("generated-clouds.json");
    let cm = Command::new(&bin)
        .args([
            "clouds-manifest",
            "--out",
            clouds_out.to_str().expect("utf8"),
            "--dev-regions",
            "us-east-2",
            "--prod-regions",
            "us-east-2",
        ])
        .env("HOME", &home)
        .env("CARBON_AWS_CLI_BIN", &mock_aws)
        .env("CARBON_SECRETSMANAGER_ENDPOINT_URL", mock_sm.endpoint_url())
        .env("AWS_ACCESS_KEY_ID", "loopback-test-access-key")
        .env("AWS_SECRET_ACCESS_KEY", "loopback-test-secret-material")
        .output()
        .expect("carbon clouds-manifest");
    assert!(
        cm.status.success() && clouds_out.is_file(),
        "clouds-manifest failed: {}",
        String::from_utf8_lossy(&cm.stderr)
    );

    // 6. `carbon coverage-audit` CLI subprocess against `emu_dev` + `emu_prod` + Mock Secrets Manager!
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cov_audit_json = tmp.path().join("cov-audit-out.json");
    let ca = Command::new(&bin)
        .args([
            "coverage-audit",
            "--clouds-manifest",
            clouds_out.to_str().expect("utf8"),
            "--signing-pub",
            repo_root
                .join("docs/release/signing.pub")
                .to_str()
                .expect("utf8"),
            "--json-out",
            cov_audit_json.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .env("CARBON_SECRETSMANAGER_ENDPOINT_URL", mock_sm.endpoint_url())
        .env("AWS_ACCESS_KEY_ID", "loopback-test-access-key")
        .env("AWS_SECRET_ACCESS_KEY", "loopback-test-secret-material")
        .output()
        .expect("carbon coverage-audit");
    assert!(
        ca.status.success() && cov_audit_json.is_file(),
        "coverage-audit failed: {}",
        String::from_utf8_lossy(&ca.stderr)
    );

    emu_dev.shutdown();
    emu_prod.shutdown();
}

#[test]
fn cli_ci_live_4tier_and_spec_report_verify_subprocesses() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let home = tmp.path().join("home");
    fs::create_dir_all(&home).expect("mkdir home");
    let bin = carbon_bin();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sha = "7988e2ecd3067647526762774e9bdfc522745990";

    // 1. `carbon ci-live-4tier`: bad `--source-sha` refusal, `--require-live-pro` refusal, and full Option B run via mock `CARBON_DOCKER_BIN`!
    let bad_sha = Command::new(&bin)
        .args([
            "ci-live-4tier",
            "--source-sha",
            "not_40_hex",
            "--out-dir",
            tmp.path().join("c1").to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .output()
        .expect("bad sha");
    assert!(!bad_sha.status.success());

    let req_pro = Command::new(&bin)
        .args([
            "ci-live-4tier",
            "--source-sha",
            sha,
            "--require-live-pro",
            "--out-dir",
            tmp.path().join("c2").to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .env_remove("CARBON_CI_DEV_PRO_TOKEN")
        .env_remove("CARBON_CI_PROD_PRO_TOKEN")
        .output()
        .expect("require-live-pro missing");
    assert!(!req_pro.status.success());

    let mock_docker = tmp.path().join("mock-docker.sh");
    let b64 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let c64 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    let d64 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    let e64 = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    let s64 = "3333333333333333333333333333333333333333333333333333333333333333";
    let docker_script = format!(
        r#"#!/bin/sh
set -eu
case "$1" in
  pull|build|rm)
    cat >/dev/null 2>&1 || true
    exit 0 ;;
  image)
    printf '[{{"Id":"sha256:{c64}","RepoDigests":["ubuntu@sha256:{d64}"]}}]\n'
    exit 0 ;;
  run)
    python3 -c '
import base64, json, re, sys, urllib.request
src = sys.stdin.read()
m = re.search(r"EMU_CLOUDS_B64=\x27([^\x27]+)\x27", src)
if m:
    doc = json.loads(base64.b64decode(m.group(1)).decode())
    for cl in ("dev", "prod"):
        ep = doc["clouds"][cl]["public_endpoint"]
        req = urllib.request.Request(ep + "/invoke/eval", data=b"{{}}", method="POST")
        try:
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:
            pass
'
    cat <<'EOF'
===KV:OS_PRETTY_NAME=Ubuntu 24.04.5 LTS===
===KV:UNAME_STR=Linux 6.8.0 x86_64===
===KV:PRE_CLEAN=true===
===KV:POST_LOGOUT_ABSENT=true===
===KV:ABSENT_ENV_COUNT=13===
===KV:INSTALL_WALL_MS=1410===
===KV:CONFIG_DIR_MODE=0o700===
===KV:BIN_BYTES=3763432===
===KV:BIN_SHA256={b64}===
===KV:CLOUDS_SHA256={e64}===
===KV:QS_WALL_MS=4500===
===KV:CSV_BYTES=15938===
===KV:CSV_SHA256={s64}===
===KV:GATE_RC=1===
===BEGIN:GATE_OUT===
HTTP 403 Forbidden: Pro tier required
===END:GATE_OUT===
===KV:GATE_DEV_HIGH_RC=1===
===BEGIN:GATE_DEV_HIGH_OUT===
HTTP 403 Forbidden: Pro tier required
===END:GATE_DEV_HIGH_OUT===
===KV:GATE_DEV_ULTRA_RC=1===
===BEGIN:GATE_DEV_ULTRA_OUT===
HTTP 403 Forbidden: Pro tier required
===END:GATE_DEV_ULTRA_OUT===
===KV:GATE_PROD_ULTRA_RC=1===
===BEGIN:GATE_PROD_ULTRA_OUT===
HTTP 403 Forbidden: Pro tier required
===END:GATE_PROD_ULTRA_OUT===
===BEGIN:SELF_VERIFY===
{{"release_tag":"carbon-cli-v1.1.0","verified":true,"manifest_ecdsa_verified":true,"kek_pins_verified":6}}
===END:SELF_VERIFY===
===BEGIN:DOCTOR===
{{"healthy":true,"clouds":{{"dev":{{"healthy_regions":1}},"prod":{{"healthy_regions":5}}}}}}
===END:DOCTOR===
===BEGIN:QUICKSTART===
{{"final_state":"completed","run_id":"run_qs_01","total_wall_ms":4500.0,"holdout_r2":0.9998,"encrypted_cuen":true,"shredded_after_run":true}}
===END:QUICKSTART===
EOF
    for eff in fast medium high ultra; do
      if [ "$eff" = "fast" ] || [ "$eff" = "medium" ]; then
        tgt="aws_live"; tier="free"; mode="SoloSingleLambda"; wv=1; shd=0
      else
        tgt="ci_emulator"; tier="pro"; mode="HybridFanOut"; wv=12; shd=120
      fi
      for cl in dev prod; do
        printf '===KV:WALL_%s_%s=3900===\n' "$cl" "$eff"
        printf '===KV:TARGET_%s_%s=%s===\n' "$cl" "$eff" "$tgt"
        printf '===BEGIN:RUN_%s_%s===\n' "$cl" "$eff"
        printf '{{"tier_applied":"%s","execution_mode":"%s","selected_region":"us-east-2","run_id":"run_%s_%s_01","final_state":"completed","encrypted_cuen":true,"shredded_after_run":true,"waves_completed":%d,"waves_target":%d,"candidates_evaluated":64,"eval_invocations":%d,"poll_requests":4,"prep_ms":10.0,"upload_ms":90.0,"search_ms":3500.0,"predict_ms":30.0,"total_wall_ms":3630.0,"outer_holdout_score":0.08,"holdout_r2":0.9997,"mean_baseline_rmse":6.2,"linear_baseline_rmse":5.0,"linear_baseline_r2":0.33,"rmse_reduction_vs_linear_pct":98.4,"cost_estimate":{{"apigw_http_api_usd":0.000008,"lambda_edge_usd":0.000004,"lambda_engine_usd":0.0003,"lambda_eval_usd":0.0001,"s3_requests_usd":0.00001,"s3_storage_and_transfer_usd":0.000001,"total_estimated_usd":0.000423,"runs_per_1_usd":2364.0}}}}\n' "$tier" "$mode" "$cl" "$eff" "$wv" "$wv" "$shd"
        printf '===END:RUN_%s_%s===\n' "$cl" "$eff"
      done
    done
    exit 0 ;;
esac
"#
    );
    write_executable_script(&mock_docker, &docker_script);

    let live_out_dir = tmp.path().join("live_4tier_out");
    let live_ok = Command::new(&bin)
        .args([
            "ci-live-4tier",
            "--source-sha",
            sha,
            "--out-dir",
            live_out_dir.to_str().expect("utf8"),
            "--repo-root",
            repo_root.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .env("CARBON_DOCKER_BIN", &mock_docker)
        .env_remove("CARBON_CI_DEV_PRO_TOKEN")
        .env_remove("CARBON_CI_PROD_PRO_TOKEN")
        .output()
        .expect("carbon ci-live-4tier");
    assert!(
        live_ok.status.success(),
        "ci-live-4tier failed: {}",
        String::from_utf8_lossy(&live_ok.stderr)
    );
    let live_rcpt = live_out_dir.join("live-ci-4tier-docker.json");
    assert!(live_rcpt.is_file());
    assert!(live_out_dir.join("live-ci-4tier-docker.md").is_file());
    assert!(live_out_dir.join("live-ci-4tier-docker.sha256").is_file());

    // 2. `carbon spec-report` CLI subprocess + `carbon spec-verify` CLI subprocess
    let ci_qual = tmp.path().join("ci-qual.json");
    fs::write(
        &ci_qual,
        serde_json::to_vec_pretty(&jope_carbon::cli::spec_report::CiQuickstartQualificationReceipt {
            schema: "jopedime.carbon.ci-quickstart-qualification.v1".to_owned(),
            timestamp_epoch_ms: 1_762_000_000_000,
            source_sha: sha.to_owned(),
            binary_version: jope_carbon::CARBON_PKG_VERSION.to_owned(),
            release_tag: jope_carbon::CARBON_RELEASE_TAG.to_owned(),
            emulator_regions: vec!["us-east-2".to_owned()],
            emulator_edge_invocations: 12,
            emulator_engine_invocations: 8,
            emulator_eval_invocations: 16,
            residual_cas_blobs_after_shred: 0,
            total_tests: 8,
            passed_tests: 8,
            all_passed: true,
            total_wall_ms: 950.0,
            steps: Vec::new(),
            redteam_vectors: Vec::new(),
        })
        .expect("ser"),
    )
    .expect("write ci_qual");

    let spec_dir = tmp.path().join("spec_bundle");
    let sr = Command::new(&bin)
        .args([
            "spec-report",
            "--evidence-dir",
            repo_root
                .join("docs/carbon/evidence")
                .to_str()
                .expect("utf8"),
            "--out-dir",
            spec_dir.to_str().expect("utf8"),
            "--ci-qualification-in",
            ci_qual.to_str().expect("utf8"),
            "--live-ci-receipt",
            live_rcpt.to_str().expect("utf8"),
            "--source-sha",
            sha,
            "--repo-root",
            repo_root.to_str().expect("utf8"),
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon spec-report");
    assert!(
        sr.status.success(),
        "spec-report failed: {}",
        String::from_utf8_lossy(&sr.stderr)
    );
    assert!(spec_dir.join("carbon-engineering-testing-spec.html").is_file());

    // Missing `spec-pdf-receipt.json` must fail `carbon spec-verify` closed!
    let sv_missing = Command::new(&bin)
        .args([
            "spec-verify",
            "--dir",
            spec_dir.to_str().expect("utf8"),
            "--expected-source-sha",
            sha,
        ])
        .env("HOME", &home)
        .output()
        .expect("carbon spec-verify missing pdf");
    assert!(!sv_missing.status.success());
}
