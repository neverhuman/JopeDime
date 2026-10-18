//! I32 offline closure adapter. It never builds either fixture executable.
//! Missing or unpinned artifacts are explicit BLOCKED failures, not smoke passes.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

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

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("")
}

fn assert_blocked_refusal_guard_self_test(name: &str, dir: &Path) {
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let digest_name = format!("{name}_SHA256");
    let original_path = env::var_os(name);
    let original_digest = env::var_os(&digest_name);

    env::remove_var(name);
    let unset = std::panic::catch_unwind(|| pinned_binary(name))
        .expect_err("unset fixture pin must be refused");
    assert!(panic_message(unset.as_ref()).starts_with("BLOCKED:"));

    fs::create_dir_all(dir).expect("create guard self-test directory");
    let dummy = dir.join("dummy-fixture");
    fs::write(&dummy, b"guard-only dummy executable").expect("write dummy fixture");
    let mut permissions = fs::metadata(&dummy)
        .expect("read dummy fixture metadata")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&dummy, permissions).expect("make dummy fixture executable");
    env::set_var(name, &dummy);
    env::set_var(&digest_name, "mismatched-sha256");
    let mismatch = std::panic::catch_unwind(|| pinned_binary(name))
        .expect_err("mismatched fixture digest must be refused");
    assert!(panic_message(mismatch.as_ref()).contains("BLOCKED:"));

    match original_path {
        Some(value) => env::set_var(name, value),
        None => env::remove_var(name),
    }
    match original_digest {
        Some(value) => env::set_var(&digest_name, value),
        None => env::remove_var(&digest_name),
    }
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
#[ignore = "BLOCKED without prebuilt JOPE_RESEARCH_LOOP_FIXTURE_BIN and JOPE_CHALLENGER_FIXTURE_BIN (--features neural); run with --ignored"]
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
#[ignore = "BLOCKED without prebuilt JOPE_RESEARCH_LOOP_FIXTURE_BIN and JOPE_CHALLENGER_FIXTURE_BIN (--features neural); run with --ignored"]
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

#[test]
fn blocked_refusal_guard_self_test() {
    assert_blocked_refusal_guard_self_test(
        "JOPE_RESEARCH_LOOP_FIXTURE_BIN",
        &scratch("guard-self-test"),
    );
    assert_blocked_refusal_guard_self_test(
        "JOPE_CHALLENGER_FIXTURE_BIN",
        &scratch("reload-guard-self-test"),
    );
}
