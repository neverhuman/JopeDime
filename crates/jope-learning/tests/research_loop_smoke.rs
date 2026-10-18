//! I32 offline closure adapter. It uses pinned fixtures when supplied and local scripts in CI.

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

fn verify_binary(path: &Path, expected: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("fixture path must be absolute".into());
    }
    let metadata = fs::metadata(path).map_err(|_| "fixture is absent")?;
    if !metadata.is_file() {
        return Err("fixture is not a regular file".into());
    }
    if metadata.permissions().mode() & 0o111 == 0 {
        return Err("fixture is not executable".into());
    }
    let actual = hex::encode(Sha256::digest(
        fs::read(path).map_err(|_| "cannot read fixture")?,
    ));
    if actual != expected {
        return Err("fixture executable digest changed".into());
    }
    Ok(path.to_path_buf())
}

fn pinned_binary(name: &str) -> PathBuf {
    let path = env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("BLOCKED: required fixture {name} is unset"));
    let expected = env::var(format!("{name}_SHA256"))
        .unwrap_or_else(|_| panic!("BLOCKED: {name}_SHA256 is unset"));
    verify_binary(&path, &expected).unwrap_or_else(|error| panic!("BLOCKED: {name}: {error}"))
}

fn fixture_binary(name: &str, output: &Path, script: &str) -> PathBuf {
    if env::var_os(name).is_some() {
        return pinned_binary(name);
    }

    let path = output.join(format!("{name}.py"));
    fs::write(&path, script).expect("write local fixture executable");
    let mut permissions = fs::metadata(&path)
        .expect("stat local fixture executable")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&path, permissions).expect("make local fixture executable");

    let digest = hex::encode(Sha256::digest(fs::read(&path).expect("read local fixture")));
    assert_eq!(
        verify_binary(&path, &digest).expect("verify local fixture"),
        path
    );
    assert!(
        verify_binary(&path, &"0".repeat(64)).is_err(),
        "fixture verification must reject a wrong expected digest"
    );
    path
}

fn run_engine(output: &Path) -> Value {
    let binary = fixture_binary(
        "JOPE_RESEARCH_LOOP_FIXTURE_BIN",
        output,
        r##"#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
output = pathlib.Path(args[args.index("--output") + 1]).resolve()
output.mkdir(parents=True, exist_ok=True)
checkpoint = output / "child-checkpoint.bin"
checkpoint.write_bytes(b"synthetic one-step checkpoint")
capsule = {
    "status": "mechanism_fixture_partial_pending_independent_reload",
    "permission": {"qualifying": False},
    "dataset_role": "synthetic_train_only",
    "native_optimizer_steps": 1,
    "parameters_changed": True,
    "actor_behavior_changed": True,
    "leakage_canary": {"refused_before_objective": True},
    "publication_crash": {"accepted_parent_pointer_unchanged": True},
    "judge": {"decision": "rejected"},
    "scientific_benefit_claim": False,
    "serving_promotion_claim": False,
    "publication": "pending",
    "child_checkpoint": str(checkpoint),
}
(output / "capsule.json").write_text(json.dumps(capsule), encoding="utf-8")
print(json.dumps(capsule))
"##,
    );
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
    let binary = fixture_binary(
        "JOPE_CHALLENGER_FIXTURE_BIN",
        &output,
        r##"#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
checkpoint = pathlib.Path(args[args.index("--i32-reload-checkpoint") + 1])
if not checkpoint.is_file():
    raise SystemExit("engine checkpoint is missing")
output = pathlib.Path(args[args.index("--output") + 1])
output.write_text(json.dumps({
    "independent_reload_verified": True,
    "parameter_digest_matches_engine": True,
    "next_actor_behavior_changed": True,
    "dry_run_only": True,
}), encoding="utf-8")
"##,
    );
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
