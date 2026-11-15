//! Function and closure coverage tests for `cli/deploy.rs` and `cli/model_store.rs`:
//! cross-compile packager, ELF/archive validators, vault secret bootstrap/rotation, prototype
//! decommission, multi-region canary gate, and `< 1 s` automatic alias rollback.

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    http::{Response, StatusCode},
    routing::post,
    Router,
};
use jope_api::carbon::CarbonTierWire;
use jope_carbon::{
    assemble_verified_carbon_package, build_deterministic_single_bin_tar,
    build_deterministic_single_bin_targz, build_deterministic_zip_bytes,
    compute_residual_quantiles, execute_carbon_deploy, execute_carbon_package,
    execute_carbon_rollback, execute_carbon_rollback_multi, expand_region_targets,
    generate_public_clouds_manifest, load_model_handle_by_spec, mint_capability_token_from_vault,
    resolve_cell_stack_endpoint, resolve_cell_stack_public_endpoint,
    synthesize_cell_template_for_release, try_external_stream_compress, verify_arm64_lambda_binary,
    verify_static_linux_elf, CarbonDeployOptions, CarbonEmuConfig, CarbonEngineKek,
    CarbonLiveAliasVersions, CarbonLocalLambdaEmulator, CarbonPackageOptions,
    CarbonRollbackOptions, CarbonZipEntry,
};

use super::{
    minimal_aarch64_static_elf64, minimal_x86_64_static_elf64, spawn_loopback_axum,
    write_executable_script, LoopbackMockS3Server, LoopbackMockSecretsServer, ScopedProcessEnv,
};

fn build_dynamic_elf_with_pt_interp() -> Vec<u8> {
    let mut elf = vec![0u8; 128];
    elf[0..4].copy_from_slice(b"\x7fELF");
    elf[4] = 2;
    elf[5] = 1;
    elf[6] = 1;
    elf[16..18].copy_from_slice(&2u16.to_le_bytes());
    elf[18..20].copy_from_slice(&0x00B7u16.to_le_bytes());
    elf[32..40].copy_from_slice(&64u64.to_le_bytes()); // e_phoff = 64
    elf[54..56].copy_from_slice(&56u16.to_le_bytes()); // e_phentsize = 56
    elf[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum = 1
    elf[64..68].copy_from_slice(&3u32.to_le_bytes()); // PT_INTERP == 3
    elf
}

fn populate_mock_cross_target_dir(target_dir: &Path) {
    let arm_elf = minimal_aarch64_static_elf64();
    let x64_elf = minimal_x86_64_static_elf64();
    for rel in [
        "aarch64-unknown-linux-musl/carbon-cli/carbon-edge",
        "aarch64-unknown-linux-musl/dist/carbon-worker",
        "aarch64-unknown-linux-musl/carbon-cli/carbon",
        "armv7-unknown-linux-musleabihf/carbon-cli/carbon",
    ] {
        let p = target_dir.join(rel);
        fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        fs::write(p, &arm_elf).expect("write arm elf");
    }
    let x64_p = target_dir.join("x86_64-unknown-linux-musl/carbon-cli/carbon");
    fs::create_dir_all(x64_p.parent().expect("parent")).expect("mkdir");
    fs::write(x64_p, &x64_elf).expect("write x64 elf");
}

#[test]
fn deploy_elf_archive_template_and_region_validator_functions() {
    // Region target expansion & error guard
    let w1 = expand_region_targets(&[
        "  ".to_owned(),
        "canary".to_owned(),
        "wave1".to_owned(),
        "all-17".to_owned(),
        "eu-west-1".to_owned(),
    ])
    .expect("expand_region_targets");
    assert!(w1.len() >= 17);
    assert_eq!(
        expand_region_targets(&[]).expect("empty default"),
        vec!["us-east-2".to_owned()]
    );
    assert!(expand_region_targets(&["BAD_REGION!".to_owned()]).is_err());

    // Deterministic ZIP & USTAR tar(.gz) + stream compressor
    assert!(build_deterministic_zip_bytes(vec![CarbonZipEntry {
        rel_path: "../escape".to_owned(),
        bytes: b"x".to_vec(),
        unix_mode: 0o644,
    }])
    .is_err());
    let zip_ok = build_deterministic_zip_bytes(vec![
        CarbonZipEntry {
            rel_path: "empty.txt".to_owned(),
            bytes: Vec::new(),
            unix_mode: 0o644,
        },
        CarbonZipEntry {
            rel_path: "hello.txt".to_owned(),
            bytes: b"hello deterministic zip".to_vec(),
            unix_mode: 0o755,
        },
    ])
    .expect("zip_ok");
    assert!(zip_ok.starts_with(b"PK\x03\x04"));

    assert!(build_deterministic_single_bin_tar("", b"x").is_err());
    assert!(build_deterministic_single_bin_tar("sub/carbon", b"x").is_err());
    let tgz = build_deterministic_single_bin_targz("carbon", b"elf-payload").expect("targz");
    assert_eq!(&tgz[0..2], &[0x1f, 0x8b]);
    assert!(try_external_stream_compress("/nonexistent-compressor-xyz", &[], b"abc").is_none());
    assert!(try_external_stream_compress("sh", &["-c", "exit 2"], b"abc").is_none());
    let piped = try_external_stream_compress("cat", &[], b"stream-ok").expect("cat stream");
    assert_eq!(piped, b"stream-ok");

    // Static ELF & CT3 fence checks
    assert!(verify_static_linux_elf(b"short").is_err());
    let mut elf32 = minimal_aarch64_static_elf64();
    elf32[4] = 1;
    assert!(verify_static_linux_elf(&elf32).is_err());
    let mut bad_phdr = build_dynamic_elf_with_pt_interp();
    bad_phdr[56..58].copy_from_slice(&10u16.to_le_bytes()); // exceeds length
    assert!(verify_static_linux_elf(&bad_phdr).is_err());
    assert!(verify_static_linux_elf(&build_dynamic_elf_with_pt_interp()).is_err());

    let x64 = minimal_x86_64_static_elf64();
    verify_static_linux_elf(&x64).expect("valid x64 static elf");
    assert!(verify_arm64_lambda_binary(&x64, true).is_err());
    let mut ct3_bad = minimal_aarch64_static_elf64();
    ct3_bad.extend_from_slice(b"models/bundles/r882");
    assert!(verify_arm64_lambda_binary(&ct3_bad, true).is_err());
    verify_arm64_lambda_binary(&ct3_bad, false).expect("worker allows r882 symbol");

    // CloudFormation synthesizer
    assert!(synthesize_cell_template_for_release("ab").is_err());
    let tpl = synthesize_cell_template_for_release("rel20261115abcd").expect("synth template");
    assert!(tpl.contains("EdgeVersionrel20261115abcd"));
}

#[test]
fn deploy_cross_compile_zig_wrappers_and_cli_archives_function_closure() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let r882_dir = tmp.path().join("r882");
    fs::create_dir_all(r882_dir.join("sub")).expect("mkdir");
    fs::write(r882_dir.join("sub/weights.bin"), b"mock-weights").expect("write");

    // Unverified r882 dir fails execute_carbon_package; mismatched edge_bin/worker_bin fails assemble
    assert!(execute_carbon_package(&CarbonPackageOptions {
        r882_dir: r882_dir.clone(),
        out_dir: tmp.path().join("out_unverified"),
        edge_bin: None,
        worker_bin: None,
        build_jobs: 1,
        include_cli_binaries: false,
    })
    .is_err());
    assert!(assemble_verified_carbon_package(
        &CarbonPackageOptions {
            r882_dir: r882_dir.clone(),
            out_dir: tmp.path().join("out_half"),
            edge_bin: Some(tmp.path().join("only_edge")),
            worker_bin: None,
            build_jobs: 1,
            include_cli_binaries: false,
        },
        "sha256:00".to_owned(),
    )
    .is_err());

    // Symlink inside r882_dir is refused fail-closed
    #[cfg(unix)]
    {
        let sym_r882 = tmp.path().join("r882_sym");
        fs::create_dir_all(&sym_r882).expect("mkdir sym_r882");
        std::os::unix::fs::symlink(r882_dir.join("sub/weights.bin"), sym_r882.join("ln.bin"))
            .expect("symlink");
        let elf = minimal_aarch64_static_elf64();
        let eb = tmp.path().join("eb");
        let wb = tmp.path().join("wb");
        fs::write(&eb, &elf).expect("eb");
        fs::write(&wb, &elf).expect("wb");
        let sym_err = assemble_verified_carbon_package(
            &CarbonPackageOptions {
                r882_dir: sym_r882,
                out_dir: tmp.path().join("out_sym"),
                edge_bin: Some(eb),
                worker_bin: Some(wb),
                build_jobs: 1,
                include_cli_binaries: false,
            },
            "sha256:00".to_owned(),
        )
        .expect_err("symlink must fail");
        assert!(sym_err.contains("refusing symlink"));
    }

    // Mock `zig`, `cargo`, `rustup`, `zstd`, and `xz` to exercise full cross-compile + CLI archive pipeline
    let mock_bin = tmp.path().join("mock_bin");
    fs::create_dir_all(&mock_bin).expect("mkdir mock_bin");
    let mock_target = tmp.path().join("mock_target");
    populate_mock_cross_target_dir(&mock_target);

    let zig_bin = mock_bin.join("zig");
    write_executable_script(&zig_bin, "#!/bin/sh\nexit 0\n");
    write_executable_script(
        &mock_bin.join("rustup"),
        "#!/bin/sh\nprintf 'aarch64-unknown-linux-musl\\nx86_64-unknown-linux-musl\\narmv7-unknown-linux-musleabihf\\n'\n",
    );
    write_executable_script(&mock_bin.join("zstd"), "#!/bin/sh\ncat\n");
    write_executable_script(&mock_bin.join("xz"), "#!/bin/sh\ncat\n");

    let cargo_script = format!(
        r#"#!/bin/sh
if [ "${{MOCK_CARGO_FAIL:-0}}" = "1" ]; then
  exit 2
fi
case "$*" in
  *"metadata --no-deps"*)
    if [ "${{MOCK_META_FAIL:-0}}" = "1" ]; then
      exit 3
    fi
    printf '{{"target_directory":"{}"}}\n'
    ;;
  *)
    exit 0
    ;;
esac
"#,
        mock_target.display()
    );
    write_executable_script(&mock_bin.join("cargo"), &cargo_script);

    let orig_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{orig_path}", mock_bin.display());
    let out_pkg = tmp.path().join("out_full_cross");
    {
        let _env = ScopedProcessEnv::set(&[
            ("CARBON_ZIG_BIN", zig_bin.to_str().expect("utf8")),
            ("PATH", &new_path),
            ("JOPE_BUILD_SOURCE_SHA", &"ab".repeat(20)),
        ]);
        let manifest = assemble_verified_carbon_package(
            &CarbonPackageOptions {
                r882_dir: r882_dir.clone(),
                out_dir: out_pkg.clone(),
                edge_bin: None,
                worker_bin: None,
                build_jobs: 2,
                include_cli_binaries: true,
            },
            "sha256:11".to_owned(),
        )
        .expect("full cross-compile + include_cli_binaries");
        assert!(out_pkg.join(&manifest.edge_zip_file).is_file());
        assert!(out_pkg.join("carbon-linux-x86_64-musl.tar.gz").is_file());
        assert!(out_pkg.join("carbon-linux-aarch64-musl.tar.gz").is_file());
        assert!(out_pkg
            .join("carbon-linux-armv7-musleabihf.tar.gz")
            .is_file());
        assert!(out_pkg.join("CLI-SHA256SUMS").is_file());
    }

    // Also cover cargo build failure & cargo metadata failure & missing zig fallback
    let empty_home = tmp.path().join("no_zig_home");
    fs::create_dir_all(&empty_home).expect("mkdir");
    {
        let _env = ScopedProcessEnv::set_opt(&[
            ("CARBON_ZIG_BIN", Some("/nonexistent-zig")),
            ("HOME", Some(empty_home.to_str().expect("utf8"))),
            ("PATH", Some(&new_path)),
            ("MOCK_CARGO_FAIL", Some("1")),
        ]);
        assert!(assemble_verified_carbon_package(
            &CarbonPackageOptions {
                r882_dir: r882_dir.clone(),
                out_dir: tmp.path().join("out_cargo_fail"),
                edge_bin: None,
                worker_bin: None,
                build_jobs: 1,
                include_cli_binaries: false,
            },
            "sha256:11".to_owned(),
        )
        .is_err());
    }
    {
        let _env = ScopedProcessEnv::set(&[
            ("CARBON_ZIG_BIN", zig_bin.to_str().expect("utf8")),
            ("PATH", &new_path),
            ("MOCK_META_FAIL", "1"),
        ]);
        assert!(assemble_verified_carbon_package(
            &CarbonPackageOptions {
                r882_dir,
                out_dir: tmp.path().join("out_meta_fail"),
                edge_bin: None,
                worker_bin: None,
                build_jobs: 1,
                include_cli_binaries: false,
            },
            "sha256:11".to_owned(),
        )
        .is_err());
    }
}

fn prepare_test_release_manifest(dir: &Path) -> (PathBuf, String) {
    let r882_dir = dir.join("r882_m");
    fs::create_dir_all(r882_dir.join("models")).expect("mkdir");
    fs::write(r882_dir.join("models/a.json"), b"{}").expect("write a");
    fs::write(r882_dir.join("models/b.json"), b"{}").expect("write b");
    let elf = minimal_aarch64_static_elf64();
    let eb = dir.join("eb");
    let wb = dir.join("wb");
    fs::write(&eb, &elf).expect("eb");
    fs::write(&wb, &elf).expect("wb");
    let out_dir = dir.join("pkg_m");
    let m = assemble_verified_carbon_package(
        &CarbonPackageOptions {
            r882_dir,
            out_dir: out_dir.clone(),
            edge_bin: Some(eb),
            worker_bin: Some(wb),
            build_jobs: 1,
            include_cli_binaries: false,
        },
        "sha256:22".to_owned(),
    )
    .expect("assemble_verified_carbon_package");
    (out_dir.join("release-manifest.json"), m.release_id)
}

#[test]
fn deploy_vault_bootstrap_sync_canary_wave_and_auto_rollback_function_closure() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (manifest_path, rel_id) = prepare_test_release_manifest(tmp.path());
    let mock_s3 = LoopbackMockS3Server::start();
    let mock_sm = LoopbackMockSecretsServer::start();

    let mut cfg_emu = CarbonEmuConfig::for_dir(tmp.path().join("emu_canary"));
    cfg_emu.region = "us-east-2".to_owned();
    cfg_emu.auth_secret = [0x34; 32];
    cfg_emu.engine_kek_scalar = [0x19; 32];
    let auth_hex = hex::encode(cfg_emu.auth_secret);
    let emu = CarbonLocalLambdaEmulator::start(cfg_emu).expect("start emu");
    let (_, probed) =
        jope_carbon::CarbonThinClient::connect_fastest(vec![emu.base_url()]).expect("probe emu");
    let emu_kek_pub = probed[0]
        .engine_kek_pubkey_b64
        .clone()
        .expect("emu kek pub");

    let mock_aws = tmp.path().join("aws-rollout.sh");
    let mode_file = tmp.path().join("aws_mode.txt");
    fs::write(&mode_file, "fresh").expect("write mode");
    let aws_script = format!(
        r#"#!/bin/sh
set -eu
MODE="$(cat "{mode}")"
ARGS="$*"
case "$ARGS" in
  *"sts get-caller-identity"*) printf '{{"Account":"123456789012"}}\n' ;;
  *"s3api head-bucket"*)
    if [ "$MODE" = "existing" ] || [ "$MODE" = "noop" ]; then printf '{{}}\n'; else echo "404 Not Found" >&2; exit 254; fi ;;
  *"iam list-role-policies"*) printf '{{"PolicyNames":["inline-p1"]}}\n' ;;
  *"iam list-attached-role-policies"*) printf '{{"AttachedPolicies":[{{"PolicyArn":"arn:aws:iam::123:policy/p1"}}]}}\n' ;;
  *"cloudformation validate-template"*) printf '{{"Parameters":[]}}\n' ;;
  *"cloudformation describe-stacks --stack-name jopedime-carbon-only-edge"*)
    printf '{{"Stacks":[{{"StackStatus":"CREATE_COMPLETE","Outputs":[{{"OutputKey":"EdgeEndpointUrl","OutputValue":"{url}"}}]}}]}}\n' ;;
  *"cloudformation describe-stacks --stack-name jopedime-carbon-no-out"*)
    printf '{{"Stacks":[{{"StackStatus":"CREATE_COMPLETE","Outputs":[]}}]}}\n' ;;
  *"cloudformation describe-stacks"*)
    if [ "$MODE" = "fresh" ]; then
      echo "Stack does not exist" >&2; exit 254
    elif [ "$MODE" = "stack_fail" ]; then
      printf '{{"Stacks":[{{"StackStatus":"UPDATE_ROLLBACK_COMPLETE","Outputs":[]}}]}}\n'
    else
      printf '{{"Stacks":[{{"StackStatus":"UPDATE_COMPLETE","Outputs":[{{"OutputKey":"EdgeEndpointUrl","OutputValue":"{url}"}},{{"OutputKey":"PublicEdgeEndpointUrl","OutputValue":"{url}"}},{{"OutputKey":"CasBucketName","OutputValue":"cas-b"}},{{"OutputKey":"EdgePublishedVersion","OutputValue":"2"}},{{"OutputKey":"EnginePublishedVersion","OutputValue":"2"}},{{"OutputKey":"EvalPublishedVersion","OutputValue":"2"}},{{"OutputKey":"ReleaseId","OutputValue":"{rel_id}"}}]}}]}}\n'
    fi ;;
  *"cloudformation create-change-set"*)
    if [ "$MODE" = "fresh" ]; then printf 'existing' > "{mode}"; fi
    printf '{{"Id":"cs-1"}}\n' ;;
  *"cloudformation describe-change-set"*)
    if [ "$MODE" = "noop" ]; then
      printf '{{"Status":"FAILED","StatusReason":"The submitted information didn'\''t contain changes."}}\n'
    elif [ "$MODE" = "cs_fail" ]; then
      printf '{{"Status":"FAILED","StatusReason":"Hook denied"}}\n'
    else
      printf '{{"Status":"CREATE_COMPLETE","Changes":[{{"ResourceChange":{{"Action":"Modify","LogicalResourceId":"EdgeFn","ResourceType":"AWS::Lambda::Function"}}}}]}}\n'
    fi ;;
  *"cloudformation describe-stack-events"*)
    printf '{{"StackEvents":[{{"ResourceStatus":"UPDATE_FAILED","LogicalResourceId":"EdgeFn","ResourceStatusReason":"Memory limit"}}]}}\n' ;;
  *"lambda get-alias"*) printf '{{"FunctionVersion":"2"}}\n' ;;
  *"lambda list-versions-by-function"*) printf '{{"Versions":[{{"Version":"$LATEST"}},{{"Version":"1"}},{{"Version":"2"}}]}}\n' ;;
  *) printf '{{}}\n' ;;
esac
"#,
        mode = mode_file.display(),
        url = emu.base_url(),
        rel_id = rel_id,
    );
    write_executable_script(&mock_aws, &aws_script);

    let _env = ScopedProcessEnv::set(&[
        ("CARBON_AWS_CLI_BIN", mock_aws.to_str().expect("utf8")),
        ("CARBON_S3_ENDPOINT_URL", &mock_s3.endpoint_url()),
        (
            "CARBON_SECRETSMANAGER_ENDPOINT_URL",
            &mock_sm.endpoint_url(),
        ),
        ("AWS_ACCESS_KEY_ID", "TESTKEYDEPLOY"),
        ("AWS_SECRET_ACCESS_KEY", "test-secret-deploy"),
    ]);

    // 1. Direct stack endpoint resolvers + unsupported env guard + missing secret refusal
    assert_eq!(
        resolve_cell_stack_endpoint("only-edge", "us-east-2").expect("edge ep"),
        emu.base_url()
    );
    assert_eq!(
        resolve_cell_stack_public_endpoint("only-edge", "us-east-2").expect("fallback pub ep"),
        emu.base_url()
    );
    assert!(resolve_cell_stack_endpoint("no-out", "us-east-2").is_err());
    assert!(resolve_cell_stack_public_endpoint("no-out", "us-east-2").is_err());

    assert!(execute_carbon_deploy(&CarbonDeployOptions {
        manifest_path: manifest_path.clone(),
        environment: "invalid_env".to_owned(),
        regions: vec!["us-east-2".to_owned()],
        artifacts_bucket_override: None,
        bootstrap_secrets: false,
        decommission_unstacked_prototype: false,
        dry_run: true,
        skip_redteam: true,
    })
    .is_err());

    // Missing `staging` secret: `bootstrap_secrets: false` errors; `bootstrap_secrets: true` + `dry_run` + live CreateSecret succeed!
    assert!(execute_carbon_deploy(&CarbonDeployOptions {
        manifest_path: manifest_path.clone(),
        environment: "staging".to_owned(),
        regions: vec!["us-east-1".to_owned()],
        artifacts_bucket_override: None,
        bootstrap_secrets: false,
        decommission_unstacked_prototype: false,
        dry_run: true,
        skip_redteam: true,
    })
    .is_err());
    execute_carbon_deploy(&CarbonDeployOptions {
        manifest_path: manifest_path.clone(),
        environment: "staging".to_owned(),
        regions: vec!["us-east-1".to_owned()],
        artifacts_bucket_override: None,
        bootstrap_secrets: true,
        decommission_unstacked_prototype: false,
        dry_run: true,
        skip_redteam: true,
    })
    .expect("bootstrap_secrets + dry_run");

    // Live bootstrap in `us-east-1` + `decommission_unstacked_prototype` (with KEK mismatch triggering error after stack create)
    let boot_err = execute_carbon_deploy(&CarbonDeployOptions {
        manifest_path: manifest_path.clone(),
        environment: "staging".to_owned(),
        regions: vec!["us-east-1".to_owned()],
        artifacts_bucket_override: None,
        bootstrap_secrets: true,
        decommission_unstacked_prototype: true,
        dry_run: false,
        skip_redteam: true,
    })
    .expect_err("generated KEK differs from emu KEK");
    assert!(boot_err.contains("engine_kek_pubkey_b64 mismatch"));

    // Existing stack (`mode == existing`) with mismatched KEK triggers `< 1 s` automatic `:live` alias rollback!
    let rb_err = execute_carbon_deploy(&CarbonDeployOptions {
        manifest_path: manifest_path.clone(),
        environment: "staging".to_owned(),
        regions: vec!["us-gov-west-1".to_owned()],
        artifacts_bucket_override: None,
        bootstrap_secrets: false,
        decommission_unstacked_prototype: false,
        dry_run: false,
        skip_redteam: true,
    })
    .expect_err("auto rollback on KEK mismatch");
    assert!(rb_err.contains("automatically rolled :live aliases back"));

    // 2. Seed `dev` vault with mismatched initial `auth_secret_hex` & out-of-sync `edge-secrets` to exercise PutSecretValue sync,
    //    existing-stack `dry_run` ChangeSet cleanup, `noop` ChangeSet, AND 2-region Canary E2E inference+shred gate!
    let dummy_kek_hex = "05".repeat(32);
    let cell_doc_primary = serde_json::json!({
        "presign_secret_hex": "11".repeat(32),
        "auth_secret_hex": &auth_hex,
        "seal_secret_hex": "33".repeat(32),
        "internal_token": "tok_dev",
        "engine_kek_hex": &dummy_kek_hex,
        "engine_kek_pubkey_b64": &emu_kek_pub,
    })
    .to_string();
    let stale_edge_doc = serde_json::json!({
        "presign_secret_hex": "99".repeat(32),
        "auth_secret_hex": "88".repeat(32),
        "engine_kek_pubkey_b64": &emu_kek_pub,
    })
    .to_string();
    mock_sm.put_secret(
        "jopedime/carbon/dev/cell-secrets",
        "arn:aws:secretsmanager:us-east-2:123:secret:dev-cell",
        "v1",
        &cell_doc_primary,
    );
    mock_sm.put_secret(
        "jopedime/carbon/dev/edge-secrets",
        "arn:aws:secretsmanager:us-east-2:123:secret:dev-edge",
        "v1-edge",
        &stale_edge_doc,
    );

    // Multi-region (`cn-north-1` + `us-west-2`) live deploy runs S3 stage, edge-secrets PutSecretValue sync,
    // 16-vector live redteam, AND Wave-0 Canary E2E inference + shred gate!
    let multi_rep = execute_carbon_deploy(&CarbonDeployOptions {
        manifest_path: manifest_path.clone(),
        environment: "dev".to_owned(),
        regions: vec!["cn-north-1".to_owned(), "us-west-2".to_owned()],
        artifacts_bucket_override: Some("shared-art-bkt".to_owned()),
        bootstrap_secrets: false,
        decommission_unstacked_prototype: false,
        dry_run: false,
        skip_redteam: false,
    })
    .expect("2-region canary E2E + redteam rollout");
    assert_eq!(multi_rep.regions.len(), 2);

    // Existing stack + staged S3 artifacts + `dry_run: true` exercises ChangeSet creation + `delete-change-set` (`L2493-2535`)
    let dry_existing = execute_carbon_deploy(&CarbonDeployOptions {
        manifest_path: manifest_path.clone(),
        environment: "dev".to_owned(),
        regions: vec!["cn-north-1".to_owned()],
        artifacts_bucket_override: Some("shared-art-bkt".to_owned()),
        bootstrap_secrets: false,
        decommission_unstacked_prototype: false,
        dry_run: true,
        skip_redteam: true,
    })
    .expect("existing stack dry-run changeset");
    assert!(dry_existing.dry_run);

    // No-op ChangeSet (`didn't contain changes`), failed ChangeSet, and failed stack rollout diagnostics
    fs::write(&mode_file, "noop").expect("mode noop");
    execute_carbon_deploy(&CarbonDeployOptions {
        manifest_path: manifest_path.clone(),
        environment: "dev".to_owned(),
        regions: vec!["cn-north-1".to_owned()],
        artifacts_bucket_override: Some("shared-art-bkt".to_owned()),
        bootstrap_secrets: false,
        decommission_unstacked_prototype: false,
        dry_run: false,
        skip_redteam: true,
    })
    .expect("noop changeset deploy");

    for fail_mode in ["cs_fail", "stack_fail"] {
        fs::write(&mode_file, fail_mode).expect("set fail mode");
        assert!(execute_carbon_deploy(&CarbonDeployOptions {
            manifest_path: manifest_path.clone(),
            environment: "dev".to_owned(),
            regions: vec!["cn-north-1".to_owned()],
            artifacts_bucket_override: Some("shared-art-bkt".to_owned()),
            bootstrap_secrets: false,
            decommission_unstacked_prototype: false,
            dry_run: false,
            skip_redteam: true,
        })
        .is_err());
    }

    // Rollback multi + mint_capability_token_from_vault error branches + clouds-manifest KEK mismatch
    fs::write(&mode_file, "existing").expect("restore mode");
    let rb_multi = execute_carbon_rollback_multi(
        "dev",
        &["us-east-2".to_owned()],
        Some(CarbonLiveAliasVersions {
            edge_version: "1".to_owned(),
            engine_version: "1".to_owned(),
            eval_version: "1".to_owned(),
        }),
    )
    .expect("rollback multi explicit versions");
    assert_eq!(rb_multi.len(), 1);
    assert!(mint_capability_token_from_vault(
        "dev",
        "us-east-2",
        "t",
        CarbonTierWire::Pro,
        0,
        false
    )
    .is_err());
    assert!(mint_capability_token_from_vault(
        "missing_env",
        "us-east-2",
        "t",
        CarbonTierWire::Pro,
        60,
        false
    )
    .is_err());
    assert!(generate_public_clouds_manifest(&[], &["us-east-2".to_owned()]).is_err());
}

#[test]
fn model_store_quantiles_and_handle_error_function_closure() {
    assert!(compute_residual_quantiles(&[], &[], None).is_none());
    assert!(compute_residual_quantiles(&[1.0], &[1.0, 2.0], None).is_none());
    assert!(compute_residual_quantiles(&[f64::NAN], &[1.0], None).is_none());
    let q = compute_residual_quantiles(&[1.0, 2.0, 3.0, 4.0], &[1.1, 1.9, 3.2, 3.8], None)
        .expect("quantiles");
    assert!(q.rmse > 0.0 && q.r2.is_some());
    let q_ref = compute_residual_quantiles(&[1.0, 2.0, 3.0, 4.0], &[1.1, 1.9, 3.2, 3.8], Some(2.5))
        .expect("quantiles with reference mean");
    assert!(q_ref.rmse > 0.0 && q_ref.r2.is_some());
    let const_q = compute_residual_quantiles(&[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0], None)
        .expect("constant target quantiles");
    assert!(const_q.r2.is_none());

    let tmp = tempfile::tempdir().expect("tempdir");
    let _env = ScopedProcessEnv::set(&[(
        "CARBON_CONFIG_DIR",
        tmp.path().join("cfg").to_str().expect("utf8"),
    )]);
    assert!(load_model_handle_by_spec(None, None).is_err());
    assert!(load_model_handle_by_spec(None, Some("latest")).is_err());
    assert!(load_model_handle_by_spec(None, Some("nonexistent_run_id")).is_err());
    let bad_file = tmp.path().join("bad-handle.json");
    fs::write(&bad_file, br#"{"schema":"wrong"}"#).expect("write");
    assert!(load_model_handle_by_spec(Some(&bad_file), None).is_err());
    let _ = CarbonEngineKek::generate().expect("kek");
}

#[test]
fn deploy_secrets_stack_s3_and_packager_error_closure_matrix() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (manifest_path, _rel_id) = prepare_test_release_manifest(tmp.path());
    let elf_a64 = minimal_aarch64_static_elf64();
    let ok_bin = tmp.path().join("ok_bin");
    fs::write(&ok_bin, &elf_a64).expect("ok_bin");
    let r882_ok = tmp.path().join("r882_m");

    // 1. Packager filesystem & cargo-metadata error closures
    assert!(assemble_verified_carbon_package(
        &CarbonPackageOptions {
            r882_dir: tmp.path().join("missing_r882_dir"),
            out_dir: tmp.path().join("out_missing_r882"),
            edge_bin: Some(ok_bin.clone()),
            worker_bin: Some(ok_bin.clone()),
            build_jobs: 1,
            include_cli_binaries: false,
        },
        "sha256:00".to_owned(),
    )
    .is_err());
    assert!(assemble_verified_carbon_package(
        &CarbonPackageOptions {
            r882_dir: r882_ok.clone(),
            out_dir: PathBuf::from("/dev/null/impossible_pkg_out"),
            edge_bin: Some(ok_bin.clone()),
            worker_bin: Some(ok_bin.clone()),
            build_jobs: 1,
            include_cli_binaries: false,
        },
        "sha256:00".to_owned(),
    )
    .is_err());
    assert!(assemble_verified_carbon_package(
        &CarbonPackageOptions {
            r882_dir: r882_ok.clone(),
            out_dir: tmp.path().join("out_missing_eb"),
            edge_bin: Some(tmp.path().join("no_edge_bin")),
            worker_bin: Some(ok_bin.clone()),
            build_jobs: 1,
            include_cli_binaries: false,
        },
        "sha256:00".to_owned(),
    )
    .is_err());
    assert!(assemble_verified_carbon_package(
        &CarbonPackageOptions {
            r882_dir: r882_ok.clone(),
            out_dir: tmp.path().join("out_missing_wb"),
            edge_bin: Some(ok_bin.clone()),
            worker_bin: Some(tmp.path().join("no_worker_bin")),
            build_jobs: 1,
            include_cli_binaries: false,
        },
        "sha256:00".to_owned(),
    )
    .is_err());

    let zig_bin = tmp.path().join("mock-zig-err");
    write_executable_script(&zig_bin, "#!/bin/sh\nexit 0\n");
    let fake_bin = tmp.path().join("fake_cargo_err");
    fs::create_dir_all(&fake_bin).expect("mkdir");
    write_executable_script(
        &fake_bin.join("cargo"),
        r#"#!/bin/sh
case "$*" in
  metadata*)
    if [ "${MOCK_META_MODE:-}" = "bad_json" ]; then printf 'not-json\n'; exit 0; fi
    if [ "${MOCK_META_MODE:-}" = "no_target" ]; then printf '{}\n'; exit 0; fi
    exit 2 ;;
  *) exit 0 ;;
esac
"#,
    );
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let cargo_err_path = format!("{}:{orig_path}", fake_bin.display());
    for mode in ["bad_json", "no_target", "fail_rc"] {
        let _env = ScopedProcessEnv::set(&[
            ("CARBON_ZIG_BIN", zig_bin.to_str().expect("utf8")),
            ("PATH", &cargo_err_path),
            ("MOCK_META_MODE", mode),
        ]);
        // Lambda arm64 metadata error closures
        assert!(assemble_verified_carbon_package(
            &CarbonPackageOptions {
                r882_dir: r882_ok.clone(),
                out_dir: tmp.path().join(format!("lam_meta_{mode}")),
                edge_bin: None,
                worker_bin: None,
                build_jobs: 1,
                include_cli_binaries: false,
            },
            "sha256:00".to_owned(),
        )
        .is_err());
        // Thin CLI metadata error closures
        assert!(assemble_verified_carbon_package(
            &CarbonPackageOptions {
                r882_dir: r882_ok.clone(),
                out_dir: tmp.path().join(format!("cli_meta_{mode}")),
                edge_bin: Some(ok_bin.clone()),
                worker_bin: Some(ok_bin.clone()),
                build_jobs: 1,
                include_cli_binaries: true,
            },
            "sha256:00".to_owned(),
        )
        .is_err());
    }
    // Missing `cargo` on PATH -> spawn error closures (`L730`, `L905`)
    {
        let empty_dir = tmp.path().join("empty_path_dir");
        fs::create_dir_all(&empty_dir).expect("mkdir");
        let _env = ScopedProcessEnv::set(&[
            ("CARBON_ZIG_BIN", zig_bin.to_str().expect("utf8")),
            ("PATH", empty_dir.to_str().expect("utf8")),
        ]);
        assert!(assemble_verified_carbon_package(
            &CarbonPackageOptions {
                r882_dir: r882_ok.clone(),
                out_dir: tmp.path().join("lam_no_cargo"),
                edge_bin: None,
                worker_bin: None,
                build_jobs: 1,
                include_cli_binaries: false,
            },
            "sha256:00".to_owned(),
        )
        .is_err());
        assert!(assemble_verified_carbon_package(
            &CarbonPackageOptions {
                r882_dir: r882_ok,
                out_dir: tmp.path().join("cli_no_cargo"),
                edge_bin: Some(ok_bin.clone()),
                worker_bin: Some(ok_bin),
                build_jobs: 1,
                include_cli_binaries: true,
            },
            "sha256:00".to_owned(),
        )
        .is_err());
    }

    // 2. Scripted SecretsManager fault server + AWS CLI / Stack / S3 / Rollback error closures
    type FaultQueue = Arc<Mutex<VecDeque<(StatusCode, Vec<u8>)>>>;
    let sm_queue: FaultQueue = Arc::new(Mutex::new(VecDeque::new()));
    let sm_q_clone = Arc::clone(&sm_queue);
    let sm_app = Router::new().route(
        "/",
        post(move || {
            let q = Arc::clone(&sm_q_clone);
            async move {
                let (st, body) = q
                    .lock()
                    .expect("lock")
                    .pop_front()
                    .unwrap_or((StatusCode::INTERNAL_SERVER_ERROR, b"empty".to_vec()));
                Response::builder()
                    .status(st)
                    .body(Body::from(body))
                    .expect("resp")
            }
        }),
    );
    let (sm_addr, sm_tx, sm_jh) = spawn_loopback_axum("fault-sm", sm_app);
    let sm_url = format!("http://{sm_addr}");

    let aws_mode_file = tmp.path().join("aws_err_mode.txt");
    fs::write(&aws_mode_file, "ok").expect("mode");
    let aws_err_bin = tmp.path().join("aws-err.sh");
    write_executable_script(
        &aws_err_bin,
        &format!(
            r#"#!/bin/sh
MODE="$(cat "{m}")"
if [ "$MODE" = "bad_json" ]; then printf 'not-json\n'; exit 0; fi
case "$*" in
  *"sts get-caller-identity"*)
    if [ "$MODE" = "no_acct" ]; then printf '{{}}\n'; else printf '{{"Account":"123456789012"}}\n'; fi ;;
  *"describe-stacks --stack-name jopedime-carbon-no-outputs-key"*)
    printf '{{"Stacks":[{{"StackStatus":"CREATE_COMPLETE"}}]}}\n' ;;
  *"describe-stacks"*)
    if [ "$MODE" = "no_stack" ]; then echo "does not exist" >&2; exit 254; fi
    if [ "$MODE" = "only_edge_url" ]; then
      printf '{{"Stacks":[{{"StackStatus":"CREATE_COMPLETE","Outputs":[{{"OutputKey":"EdgeEndpointUrl","OutputValue":"http://127.0.0.1:9"}}]}}]}}\n'
    elif [ "$MODE" = "empty_outputs" ]; then
      printf '{{"Stacks":[{{"StackStatus":"CREATE_COMPLETE","Outputs":[]}}]}}\n'
    else
      printf '{{"Stacks":[{{"StackStatus":"CREATE_COMPLETE","Outputs":[{{"OutputKey":"EdgeEndpointUrl","OutputValue":"http://127.0.0.1:9"}}]}}]}}\n'
    fi ;;
  *"lambda get-alias"*)
    if [ "$MODE" = "no_alias" ]; then exit 254; else printf '{{"FunctionVersion":"2"}}\n'; fi ;;
  *"lambda list-versions-by-function"*)
    if [ "$MODE" = "no_prev_ver" ]; then printf '{{"Versions":[{{"Version":"$LATEST"}},{{"Version":"2"}}]}}\n'; else printf '{{"Versions":[{{"Version":"1"}},{{"Version":"2"}}]}}\n'; fi ;;
  *) printf '{{}}\n' ;;
esac
"#,
            m = aws_mode_file.display()
        ),
    );

    // Unreachable AWS binary (`L1153`, `L1177`) & unreachable SecretsManager (`L1237`)
    {
        let _env = ScopedProcessEnv::set(&[
            ("CARBON_AWS_CLI_BIN", "/nonexistent-aws-cli-binary"),
            ("CARBON_SECRETSMANAGER_ENDPOINT_URL", "http://127.0.0.1:9"),
            ("AWS_ACCESS_KEY_ID", "TESTKEYERR"),
            ("AWS_SECRET_ACCESS_KEY", "test-secret-err"),
        ]);
        assert!(resolve_cell_stack_endpoint("dev", "us-east-2").is_err());
        assert!(mint_capability_token_from_vault(
            "dev",
            "us-east-2",
            "tenant_1",
            CarbonTierWire::Pro,
            60,
            false
        )
        .is_err());
    }

    let _env = ScopedProcessEnv::set(&[
        ("CARBON_AWS_CLI_BIN", aws_err_bin.to_str().expect("utf8")),
        ("CARBON_SECRETSMANAGER_ENDPOINT_URL", &sm_url),
        ("CARBON_S3_ENDPOINT_URL", "http://127.0.0.1:9"),
        ("AWS_ACCESS_KEY_ID", "TESTKEYERR"),
        ("AWS_SECRET_ACCESS_KEY", "test-secret-err"),
    ]);

    // AWS invalid JSON (`L1165`, `L1199`), stack missing `Outputs` key (`L1653`, `L1688`), `sts` missing Account (`L2724`)
    fs::write(&aws_mode_file, "bad_json").expect("mode");
    assert!(resolve_cell_stack_endpoint("dev", "us-east-2").is_err());
    let _ = execute_carbon_rollback(&CarbonRollbackOptions {
        environment: "dev".to_owned(),
        region: "us-east-2".to_owned(),
        target_versions: None,
    });
    fs::write(&aws_mode_file, "ok").expect("mode");
    assert!(resolve_cell_stack_endpoint("no-outputs-key", "us-east-2").is_err());
    assert!(resolve_cell_stack_public_endpoint("no-outputs-key", "us-east-2").is_err());

    // Manifest read/parse errors (`L2705`, `L2707`) + `sts` missing Account (`L2724`)
    let base_opts = CarbonDeployOptions {
        manifest_path: manifest_path.clone(),
        environment: "dev".to_owned(),
        regions: vec!["us-east-2".to_owned()],
        artifacts_bucket_override: Some("art-bkt".to_owned()),
        bootstrap_secrets: true,
        decommission_unstacked_prototype: false,
        dry_run: false,
        skip_redteam: true,
    };
    let mut bad_m = base_opts.clone();
    bad_m.manifest_path = tmp.path().join("missing-manifest.json");
    assert!(execute_carbon_deploy(&bad_m).is_err());
    let bad_json_m = tmp.path().join("bad-manifest.json");
    fs::write(&bad_json_m, b"not-json").expect("write");
    bad_m.manifest_path = bad_json_m;
    assert!(execute_carbon_deploy(&bad_m).is_err());

    fs::write(&aws_mode_file, "no_acct").expect("mode");
    assert!(execute_carbon_deploy(&base_opts).is_err());
    fs::write(&aws_mode_file, "ok").expect("mode");

    // `mint_capability_token_from_vault` response fault matrix (`L1613`, `L1617`, `L1621`, `L1623`, `L1632`)
    let valid_cell_secret_str = serde_json::json!({
        "presign_secret_hex": "11".repeat(32),
        "auth_secret_hex": "22".repeat(32),
        "seal_secret_hex": "33".repeat(32),
        "internal_token": "tok",
        "engine_kek_hex": "05".repeat(32),
        "engine_kek_pubkey_b64": "AAAA",
    })
    .to_string();
    let bad_hex_secret_str = serde_json::json!({
        "presign_secret_hex": "11".repeat(32),
        "auth_secret_hex": "zz-not-hex",
        "seal_secret_hex": "33".repeat(32),
        "internal_token": "tok",
        "engine_kek_hex": "05".repeat(32),
        "engine_kek_pubkey_b64": "AAAA",
    })
    .to_string();
    for (resp_body, tid) in [
        (b"not-json".to_vec(), "tenant_ok"),
        (b"{}".to_vec(), "tenant_ok"),
        (br#"{"SecretString":"bad-schema"}"#.to_vec(), "tenant_ok"),
        (
            serde_json::to_vec(&serde_json::json!({"SecretString": bad_hex_secret_str})).unwrap(),
            "tenant_ok",
        ),
        (
            serde_json::to_vec(&serde_json::json!({"SecretString": valid_cell_secret_str}))
                .unwrap(),
            "x",
        ),
    ] {
        sm_queue
            .lock()
            .unwrap()
            .push_back((StatusCode::OK, resp_body));
        assert!(mint_capability_token_from_vault(
            "dev",
            "us-east-2",
            tid,
            CarbonTierWire::Pro,
            60,
            false
        )
        .is_err());
    }

    // `ensure_cell_secrets_in_vault` & `ensure_edge_secrets_in_vault` response fault matrix
    let valid_cell_get = serde_json::to_vec(&serde_json::json!({
        "ARN": "arn:cell",
        "VersionId": "vc1",
        "SecretString": valid_cell_secret_str,
    }))
    .unwrap();
    let stale_edge_get = serde_json::to_vec(&serde_json::json!({
        "ARN": "arn:edge",
        "VersionId": "ve1",
        "SecretString": "{\"presign_secret_hex\":\"99\",\"auth_secret_hex\":\"88\",\"engine_kek_pubkey_b64\":\"BBBB\"}",
    }))
    .unwrap();
    let rnf = (
        StatusCode::BAD_REQUEST,
        br#"{"__type":"ResourceNotFoundException"}"#.to_vec(),
    );

    let scenarios: Vec<Vec<(StatusCode, Vec<u8>)>> = vec![
        vec![(StatusCode::OK, b"not-json".to_vec())],
        vec![(StatusCode::OK, b"{}".to_vec())],
        vec![(StatusCode::OK, br#"{"ARN":"a"}"#.to_vec())],
        vec![(StatusCode::OK, br#"{"ARN":"a","VersionId":"v"}"#.to_vec())],
        vec![(
            StatusCode::OK,
            br#"{"ARN":"a","VersionId":"v","SecretString":"bad"}"#.to_vec(),
        )],
        vec![(StatusCode::INTERNAL_SERVER_ERROR, b"boom".to_vec())],
        // Cell CreateSecret faults (`L1558`, `L1562`, `L1567`, HTTP 500)
        vec![
            rnf.clone(),
            (StatusCode::INTERNAL_SERVER_ERROR, b"err".to_vec()),
        ],
        vec![rnf.clone(), (StatusCode::OK, b"not-json".to_vec())],
        vec![rnf.clone(), (StatusCode::OK, b"{}".to_vec())],
        vec![rnf.clone(), (StatusCode::OK, br#"{"ARN":"a"}"#.to_vec())],
        // Edge GetSecretValue faults (`L1331`, `L1335`, `L1340`, HTTP 500)
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            (StatusCode::INTERNAL_SERVER_ERROR, b"err".to_vec()),
        ],
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            (StatusCode::OK, b"not-json".to_vec()),
        ],
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            (StatusCode::OK, b"{}".to_vec()),
        ],
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            (StatusCode::OK, br#"{"ARN":"ea"}"#.to_vec()),
        ],
        // Edge PutSecretValue faults (`L1375`, `L1379`, HTTP 500)
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            (StatusCode::OK, stale_edge_get.clone()),
            (StatusCode::INTERNAL_SERVER_ERROR, b"err".to_vec()),
        ],
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            (StatusCode::OK, stale_edge_get.clone()),
            (StatusCode::OK, b"not-json".to_vec()),
        ],
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            (StatusCode::OK, stale_edge_get.clone()),
            (StatusCode::OK, b"{}".to_vec()),
        ],
        // Edge CreateSecret faults (`L1413`, `L1417`, `L1422`, HTTP 500)
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            rnf.clone(),
            (StatusCode::INTERNAL_SERVER_ERROR, b"err".to_vec()),
        ],
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            rnf.clone(),
            (StatusCode::OK, b"not-json".to_vec()),
        ],
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            rnf.clone(),
            (StatusCode::OK, b"{}".to_vec()),
        ],
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            rnf.clone(),
            (StatusCode::OK, br#"{"ARN":"ea"}"#.to_vec()),
        ],
        // Primary vault OK + regional vault OK + S3 PUT connection refused (`L1904`)
        vec![
            (StatusCode::OK, valid_cell_get.clone()),
            rnf.clone(),
            (
                StatusCode::OK,
                br#"{"ARN":"ea","VersionId":"ev1"}"#.to_vec(),
            ),
            (StatusCode::OK, valid_cell_get.clone()),
            rnf.clone(),
            (
                StatusCode::OK,
                br#"{"ARN":"ea","VersionId":"ev1"}"#.to_vec(),
            ),
        ],
    ];
    for steps in scenarios {
        let mut q = sm_queue.lock().unwrap();
        q.clear();
        q.extend(steps);
        drop(q);
        assert!(execute_carbon_deploy(&base_opts).is_err());
    }

    // Cell PutSecretValue faults during regional auth sync (`L1492`, `L1495`, `L1496`) + missing local artifact (`L1820`)
    let regional_diff_cell = serde_json::to_vec(&serde_json::json!({
        "ARN": "arn:cell2",
        "VersionId": "vc2",
        "SecretString": serde_json::json!({
            "presign_secret_hex": "11".repeat(32),
            "auth_secret_hex": "44".repeat(32),
            "seal_secret_hex": "33".repeat(32),
            "internal_token": "tok",
            "engine_kek_hex": "05".repeat(32),
            "engine_kek_pubkey_b64": "AAAA",
        }).to_string(),
    }))
    .unwrap();
    for put_resp in [
        (StatusCode::INTERNAL_SERVER_ERROR, b"err".to_vec()),
        (StatusCode::OK, b"not-json".to_vec()),
        (StatusCode::OK, b"{}".to_vec()),
    ] {
        let mut q = sm_queue.lock().unwrap();
        q.clear();
        q.extend([
            (StatusCode::OK, valid_cell_get.clone()),
            rnf.clone(),
            (
                StatusCode::OK,
                br#"{"ARN":"ea","VersionId":"ev1"}"#.to_vec(),
            ),
            (StatusCode::OK, regional_diff_cell.clone()),
            put_resp,
        ]);
        drop(q);
        let mut reg_opts = base_opts.clone();
        reg_opts.regions = vec!["us-west-2".to_owned()];
        assert!(execute_carbon_deploy(&reg_opts).is_err());
    }

    // Missing local artifact file (`L1820`)
    let broken_pkg_dir = tmp.path().join("broken_pkg");
    fs::create_dir_all(&broken_pkg_dir).expect("mkdir");
    let broken_manifest = broken_pkg_dir.join("release-manifest.json");
    fs::copy(&manifest_path, &broken_manifest).expect("copy manifest");
    {
        let mut q = sm_queue.lock().unwrap();
        q.clear();
        q.extend([
            (StatusCode::OK, valid_cell_get.clone()),
            rnf.clone(),
            (
                StatusCode::OK,
                br#"{"ARN":"ea","VersionId":"ev1"}"#.to_vec(),
            ),
            (StatusCode::OK, valid_cell_get),
            rnf,
            (
                StatusCode::OK,
                br#"{"ARN":"ea","VersionId":"ev1"}"#.to_vec(),
            ),
        ]);
    }
    let mut missing_art_opts = base_opts;
    missing_art_opts.manifest_path = broken_manifest;
    assert!(execute_carbon_deploy(&missing_art_opts).is_err());

    // Rollback & clouds-manifest error closures (`L2827`, `L2842`, `L2890`, `L2945`, `L2951`, `L2953`)
    for rb_mode in ["no_alias", "no_prev_ver", "no_stack"] {
        fs::write(&aws_mode_file, rb_mode).expect("mode");
        assert!(execute_carbon_rollback(&CarbonRollbackOptions {
            environment: "dev".to_owned(),
            region: "us-east-2".to_owned(),
            target_versions: if rb_mode == "no_stack" {
                Some(CarbonLiveAliasVersions {
                    edge_version: "1".to_owned(),
                    engine_version: "1".to_owned(),
                    eval_version: "1".to_owned(),
                })
            } else {
                None
            },
        })
        .is_err());
    }
    for cm_mode in ["no_stack", "empty_outputs", "only_edge_url"] {
        fs::write(&aws_mode_file, cm_mode).expect("mode");
        assert!(generate_public_clouds_manifest(&["us-east-2".to_owned()], &[]).is_err());
    }

    let _ = sm_tx.send(());
    let _ = sm_jh.join();
}
