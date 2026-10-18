//! I32 offline closure adapter. It never builds either fixture executable.
//! Missing or unpinned artifacts are explicit BLOCKED failures, not smoke passes.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
};

fn scratch(label: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/task-evidence/I32")
        .join(format!(
            "{}-{label}-{}",
            env!("CARGO_PKG_VERSION"),
            std::process::id()
        ));
    fs::create_dir_all(&root).expect("create lane-local I32 evidence directory");
    let child = root.join(format!("{}-{:?}", label, std::time::SystemTime::now()));
    fs::create_dir(&child).expect("create unique fixture output directory");
    child
}

fn pinned_binary(name: &str) -> PathBuf {
    let path = env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("BLOCKED: required prebuilt fixture {name} is unset"));
    assert!(
        path.is_absolute(),
        "BLOCKED: {name} must be an absolute path"
    );
    let metadata = fs::metadata(&path).unwrap_or_else(|_| panic!("BLOCKED: {name} is absent"));
    assert!(metadata.is_file(), "BLOCKED: {name} is not a regular file");
    assert!(
        metadata.permissions().mode() & 0o111 != 0,
        "BLOCKED: {name} is not executable"
    );
    let expected = env::var(format!("{name}_SHA256"))
        .unwrap_or_else(|_| panic!("BLOCKED: {name}_SHA256 is unset"));
    let bytes = fs::read(&path).expect("read fixture executable for digest");
    let actual = hex::encode(Sha256::digest(bytes));
    assert_eq!(
        actual, expected,
        "BLOCKED: fixture executable digest changed"
    );
    path
}

fn run_engine(output: &Path) -> Value {
    let binary = pinned_binary("JOPE_RESEARCH_LOOP_FIXTURE_BIN");
    let result = Command::new(binary)
        .arg("--output")
        .arg(output)
        .output()
        .expect("start the already-built native engine fixture");
    assert!(
        result.status.success(),
        "engine fixture failed: {}\n{}",
        result.status,
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).expect("engine fixture capsule JSON")
}

#[test]
fn native_update_crash_leakage_and_rejected_parent_trace() {
    let capsule = run_engine(&scratch("native"));
    assert_eq!(
        capsule["status"],
        "mechanism_fixture_partial_pending_independent_reload"
    );
    assert_eq!(capsule["permission"]["qualifying"], false);
    assert_eq!(capsule["dataset_role"], "synthetic_train_only");
    assert_eq!(capsule["native_optimizer_steps"], 1);
    assert_eq!(capsule["parameters_changed"], true);
    assert_eq!(capsule["actor_behavior_changed"], true);
    assert_eq!(capsule["leakage_canary"]["refused_before_objective"], true);
    assert_eq!(
        capsule["publication_crash"]["accepted_parent_pointer_unchanged"],
        true
    );
    assert_eq!(capsule["judge"]["decision"], "rejected");
    assert_eq!(capsule["scientific_benefit_claim"], false);
    assert_eq!(capsule["serving_promotion_claim"], false);
    assert_eq!(capsule["publication"], "pending");
}

#[test]
fn independent_candle_reload_and_next_actor_behavior() {
    let output = scratch("independent-reload");
    let capsule = run_engine(&output);
    let binary = pinned_binary("JOPE_CHALLENGER_FIXTURE_BIN");
    let checkpoint = capsule["child_checkpoint"]
        .as_str()
        .expect("engine capsule child checkpoint path");
    let reload_path = output.join("independent-reload.json");
    let result = Command::new(binary)
        .arg("--i32-reload-checkpoint")
        .arg(checkpoint)
        .arg("--output")
        .arg(&reload_path)
        .arg("--cpu")
        .arg("--dry-run-next-actor")
        .output()
        .expect("start the already-built I14 Candle fixture");
    assert!(
        result.status.success(),
        "I14 backend fixture failed: {}\n{}",
        result.status,
        String::from_utf8_lossy(&result.stderr)
    );
    let reload: Value = serde_json::from_slice(
        &fs::read(&reload_path).expect("I14 fixture must write its reload evidence"),
    )
    .expect("I14 reload receipt JSON");
    assert_eq!(reload["independent_reload_verified"], true);
    assert_eq!(reload["parameter_digest_matches_engine"], true);
    assert_eq!(reload["next_actor_behavior_changed"], true);
    assert_eq!(reload["dry_run_only"], true);
}
