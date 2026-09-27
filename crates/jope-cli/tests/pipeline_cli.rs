//! `jope pipeline` as a child process, the way a SmartCluster job launches it: `--source-commit`
//! keeps the run from spawning `git`, and SIGTERM stops a run between evaluations without a
//! receipt.

use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

fn dataset() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test/pmlb-229-pwlinear")
}

fn jope() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_jope"));
    command
        .env_remove("JOPE_GIT_SHA")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

/// A directory holding a `git` that records every invocation in `marker` and fails.
fn fake_git(dir: &Path, marker: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let git = bin.join("git");
    std::fs::write(
        &git,
        format!("#!/bin/sh\necho \"$@\" >> '{}'\nexit 1\n", marker.display()),
    )
    .unwrap();
    std::fs::set_permissions(&git, std::fs::Permissions::from_mode(0o755)).unwrap();
    bin
}

fn run_args(output: &Path, budget: &str) -> Vec<String> {
    [
        "pipeline",
        "run",
        "--dataset",
        dataset().to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--budget",
        budget,
        "--policy",
        "skip",
        "--emit",
        "rust",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect()
}

#[test]
fn a_given_source_commit_is_recorded_and_git_is_never_spawned() {
    let temp = tempfile::tempdir().unwrap();
    let marker = temp.path().join("git-invocations");
    let bin = fake_git(temp.path(), &marker);
    let commit = "0123456789abcdef0123456789abcdef01234567";

    let with = temp.path().join("with");
    let output = jope()
        .env("PATH", &bin)
        .args(run_args(&with, "evaluations=2,seconds=60"))
        .args(["--source-commit", commit])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!marker.exists(), "git was spawned despite --source-commit");
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(with.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["git_sha"], commit);

    // Positive control: without the flag the run does ask git (which fails here, so the
    // receipt simply carries no git_sha).
    let without = temp.path().join("without");
    let output = jope()
        .env("PATH", &bin)
        .args(run_args(&without, "evaluations=2,seconds=60"))
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(marker.exists(), "the control run never asked git");

    // A malformed commit is refused before anything is written.
    let bad = temp.path().join("bad");
    let output = jope()
        .args(run_args(&bad, "evaluations=2,seconds=60"))
        .args(["--source-commit", "HEAD"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(!bad.exists());
}

fn wait_for(what: &str, deadline: Duration, mut ready: impl FnMut() -> bool, child: &mut Child) {
    let started = Instant::now();
    while !ready() {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("the run exited ({status}) before {what}");
        }
        assert!(started.elapsed() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn terminate(child: &Child) {
    let status = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn sigterm_mid_search_stops_between_evaluations_and_writes_no_receipt() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let mut child = jope()
        .args(run_args(&output, "evaluations=400,seconds=600"))
        .spawn()
        .unwrap();
    // The search is running once the cuts are prepared.
    wait_for(
        "the search to start",
        Duration::from_secs(120),
        || output.join("prepared/prepared.json").exists(),
        &mut child,
    );
    terminate(&child);
    let result = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(143), "{stderr}");
    assert!(stderr.contains("termination signal"), "{stderr}");
    assert!(
        !output.join("receipt.json").exists(),
        "an interrupted run wrote a receipt"
    );
    assert!(!output.join("search/cut-0/search.json").exists());
}
