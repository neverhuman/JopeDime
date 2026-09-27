//! Public entry stays held even when a caller requests CPU or presents fixture flags.
#![cfg(feature = "neural")]
use std::process::Command;
#[test]
fn ranked_proposals_refuse_before_reading_inputs_or_writing_output() {
    let root = tempfile::tempdir().unwrap();
    let missing = root.path().join("absent.json");
    let result = Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root")
        .arg(root.path())
        .args(["propose-ranked", "--manifest"])
        .arg(&missing)
        .env("JOPE_RANKED_DEVICE", "cpu")
        .env("JOPE_RANKED_FIXTURE_ROOT", root.path())
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("original native GPU grant verifier unavailable"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
}
