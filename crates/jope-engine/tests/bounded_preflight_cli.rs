//! Operator dry-run acceptance: synthetic host census only, no dispatch.
#[path = "support/bounded_fixture.rs"]
mod fixture;
use fixture::*;
use jope_parallel::native_loop::bounded::controller::{self, Coordinator};
use std::{fs, path::Path, process::Command};

fn invoke(dir: &Path, digest: &str, census: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_jope-native-loop"))
        .args(["preflight", "--run-dir"])
        .arg(dir)
        .args(["--plan-sha256", digest, "--census"])
        .arg(census)
        .output()
        .unwrap()
}

#[test]
fn cli_preflight_matches_three_hosts_without_mutating_or_admitting() {
    let now = now_ms();
    let (_tmp, root) = tempdir();
    let dir = root.join("run");
    let c = contract(now, 3, &keys(3));
    let plan = controller::plan(&dir, &bytes(&c), now, None).unwrap();
    let census_path = root.join("census.json");
    let observed = serde_json::to_vec(&census(&c, now)).unwrap();
    fs::write(&census_path, &observed).unwrap();
    let before = tree(&root);
    for _ in 0..2 {
        let out = invoke(&dir, &plan.plan_sha256, &census_path);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(v["allocation_held"], false);
        assert_eq!(v["plan_sha256"], plan.plan_sha256);
        assert_eq!(v["census_sha256"], sha(&observed));
        assert_eq!(v["hosts"].as_object().unwrap().len(), 3);
    }
    assert_eq!(tree(&root), before);
    let fake = FakeSupervisor::default();
    let mut coordinator = Coordinator::open(&dir, &plan.plan_sha256, fake.clone(), now).unwrap();
    assert!(
        coordinator.dispatch_all(now).is_err(),
        "dry run must not authorize dispatch"
    );
    assert_eq!(fake.0.lock().unwrap().launches, 0);
    let before = tree(&root);
    let out = invoke(&dir, &plan.plan_sha256, &census_path);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("live coordinator"));
    assert_eq!(tree(&root), before);
    coordinator.preflight(&observed, now).unwrap();
    coordinator.dispatch_all(now).unwrap();
    drop(coordinator);
    let before = tree(&root);
    let out = invoke(&dir, &plan.plan_sha256, &census_path);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unstarted"));
    assert_eq!(tree(&root), before);
}

#[test]
fn cli_preflight_refuses_wrong_plan_stale_host_and_cancelled_run_read_only() {
    let now = now_ms();
    let (_tmp, root) = tempdir();
    let dir = root.join("run");
    let c = contract(now, 3, &keys(3));
    let plan = controller::plan(&dir, &bytes(&c), now, None).unwrap();
    let path = root.join("census.json");
    let observed = census(&c, now);
    fs::write(&path, serde_json::to_vec(&observed).unwrap()).unwrap();
    let before = tree(&root);
    let out = invoke(&dir, &"0".repeat(64), &path);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("frozen plan digest"));
    assert_eq!(tree(&root), before);
    let mut stale = observed.clone();
    stale.hosts[0].observed_unix_ms -= 120_000;
    fs::write(&path, serde_json::to_vec(&stale).unwrap()).unwrap();
    let before = tree(&root);
    let out = invoke(&dir, &plan.plan_sha256, &path);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("stale_telemetry"));
    assert_eq!(tree(&root), before);
    fs::write(&path, serde_json::to_vec(&observed).unwrap()).unwrap();
    controller::cancel(&dir, &plan.plan_sha256, "fixture", now).unwrap();
    let before = tree(&root);
    let out = invoke(&dir, &plan.plan_sha256, &path);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("uncancelled"));
    assert_eq!(tree(&root), before);
}

#[test]
fn cli_preflight_refuses_expired_grants_and_unsafe_census_files() {
    let now = now_ms();
    let (_tmp, root) = tempdir();
    let path = root.join("census.json");
    for (name, planned_at) in [("expired", now - 1_300_000), ("fresh", now)] {
        let dir = root.join(name);
        let c = contract(planned_at, 3, &keys(3));
        let plan = controller::plan(&dir, &bytes(&c), planned_at, None).unwrap();
        fs::write(&path, serde_json::to_vec(&census(&c, now)).unwrap()).unwrap();
        if name == "expired" {
            let before = tree(&root);
            let out = invoke(&dir, &plan.plan_sha256, &path);
            assert!(!out.status.success());
            assert!(String::from_utf8_lossy(&out.stderr).contains("deadline already elapsed"));
            assert_eq!(tree(&root), before);
        } else {
            let link = root.join("link.json");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            assert!(!invoke(&dir, &plan.plan_sha256, &link).status.success());
            fs::write(&path, vec![b' '; 1024 * 1024 + 1]).unwrap();
            let before = tree(&root);
            let out = invoke(&dir, &plan.plan_sha256, &path);
            assert!(!out.status.success());
            assert!(String::from_utf8_lossy(&out.stderr).contains("exceeds"));
            assert_eq!(tree(&root), before);
        }
    }
}
