//! Exporter-side guard regressions for the v1/v2 boundary.
//!
//! Full checkpoint export and backend parity require the admitted neural toolchain; this bounded
//! test confirms failed preflight cannot publish either weights or a receipt.

use jope_parallel::policy_export::{export, ExportRequest, RECEIPT_SCHEMA};
use std::{fs, path::PathBuf};

#[test]
fn export_preflight_failure_publishes_no_candidate_artifacts() {
    assert_eq!(RECEIPT_SCHEMA, "jope.checkpoint_export.v1");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/task-evidence/I14/policy-export-lineage")
        .join(std::process::id().to_string());
    fs::create_dir_all(&root).unwrap();
    let request = ExportRequest {
        checkpoint: root.join("missing-checkpoint"),
        output: root.join("candidate"),
        receipt: root.join("receipt.json"),
        expect_state_sha256: "7".repeat(64),
        parent: None,
    };
    std::env::set_var("CUDA_VISIBLE_DEVICES", "");
    assert!(export(&request).is_err());
    assert!(!request.output.exists());
    assert!(!request.receipt.exists());
}
