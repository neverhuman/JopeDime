//! Automated CI adversarial security & zero-knowledge red-team suite (`V-01`..`V-16` + `RT-05`
//! fail-closed AWS Lambda secret startup enforcement).

use std::path::PathBuf;
use std::process::Command;

use jope_carbon::{execute_carbon_redteam_suite, CarbonEmuConfig, CarbonLocalLambdaEmulator};

#[test]
fn redteam_16_vectors_pass_against_isolated_lambda_emulator() {
    let root = tempfile::tempdir().expect("create redteam tempdir");
    let edge_bin = PathBuf::from(env!("CARGO_BIN_EXE_carbon-edge"));
    assert!(edge_bin.is_file());

    let mut cfg = CarbonEmuConfig::for_dir(root.path().join("redteam_cell"));
    cfg.edge_binary_path = Some(edge_bin);
    cfg.force_hybrid_fanout = true;
    cfg.eval_worker_threads = 2;
    let mut emu = CarbonLocalLambdaEmulator::start(cfg).expect("start redteam emulator");

    let report =
        execute_carbon_redteam_suite(&emu.base_url()).expect("run 16-vector redteam suite");
    for v in &report.vectors {
        assert!(
            v.passed,
            "red-team vector {} ({}) failed with HTTP {}: {}",
            v.vector_id, v.title, v.observed_http_status, v.evidence
        );
    }
    assert_eq!(report.total_vectors, 16);
    assert_eq!(report.passed_vectors, 16);
    assert!(report.all_passed);

    emu.shutdown();
}

#[test]
fn rt05_aws_lambda_production_boot_fails_closed_when_secrets_missing() {
    let edge_bin = env!("CARGO_BIN_EXE_carbon-edge");
    let edge_missing = Command::new(edge_bin)
        .env("AWS_LAMBDA_RUNTIME_API", "127.0.0.1:9")
        .env("CARBON_CAS_URL", "s3://dummy-bucket/cell-v1")
        .env("CARBON_ENGINE_INVOKE_URL", "lambda://dummy-engine")
        .env_remove("CARBON_PRESIGN_SECRET_HEX")
        .env_remove("CARBON_AUTH_SECRET_HEX")
        .env_remove("CARBON_ENGINE_KEK_PUBKEY_B64")
        .output()
        .expect("spawn carbon-edge without AWS secrets");
    assert!(
        !edge_missing.status.success(),
        "carbon-edge must refuse AWS Lambda boot without cryptographic secrets"
    );
    let edge_err = String::from_utf8_lossy(&edge_missing.stderr);
    assert!(edge_err.contains("CARBON_PRESIGN_SECRET_HEX"));

    let carbon_bin = env!("CARGO_BIN_EXE_carbon");
    let tmp = tempfile::tempdir().expect("engine tmp");
    let engine_missing = Command::new(carbon_bin)
        .env("AWS_LAMBDA_RUNTIME_API", "127.0.0.1:9")
        .env("CARBON_ROLE", "engine")
        .env("CARBON_WORK_DIR", tmp.path())
        .env("CARBON_CAS_URL", "s3://dummy-bucket/cell-v1")
        .env_remove("CARBON_SEAL_SECRET_HEX")
        .env_remove("CARBON_ENGINE_KEK_HEX")
        .output()
        .expect("spawn carbon engine without AWS secrets");
    assert!(
        !engine_missing.status.success(),
        "carbon engine must refuse AWS Lambda boot without CARBON_SEAL_SECRET_HEX / CARBON_ENGINE_KEK_HEX"
    );
    let engine_err = String::from_utf8_lossy(&engine_missing.stderr);
    assert!(engine_err.contains("CARBON_SEAL_SECRET_HEX"));
}
