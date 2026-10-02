//! `jope pipeline` as a child process, the way a SmartCluster job launches it: `--source-commit`
//! keeps the run from spawning `git`, SIGTERM stops a run between evaluations without a
//! receipt, leaving only completed rounds in an evolve lineage, and a refused command exits 2
//! without writing anything.

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

/// A cut count outside the fold partition exits 2 and leaves nothing behind, for `evolve` and
/// `run`, so the corrected command runs into the same `--output`. (`evolve --rounds 2 --cuts 6`
/// used to leave `round-001/`, and the corrected retry failed with "already holds a lineage".)
#[test]
fn a_refused_cut_count_exits_2_leaves_nothing_and_the_corrected_retry_runs() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("lineage");
    let evolve = |cuts: &str, budget: &str| {
        jope()
            .args(["pipeline", "evolve", "--dataset"])
            .arg(dataset())
            .arg("--output")
            .arg(&output)
            .args(["--rounds", "2", "--cuts", cuts, "--budget", budget])
            .args(["--policy", "skip", "--emit", "rust"])
            .output()
            .unwrap()
    };
    for cuts in ["6", "0"] {
        let result = evolve(cuts, "evaluations=16,seconds=60");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert_eq!(result.status.code(), Some(2), "{stderr}");
        assert!(
            stderr.contains(&format!("--cuts must be 1..=5, got {cuts}")),
            "{stderr}"
        );
        assert!(!output.exists(), "--cuts {cuts} left {}", output.display());
    }
    let run = temp.path().join("run");
    let result = jope()
        .args(run_args(&run, "evaluations=4,seconds=60"))
        .args(["--cuts", "6"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(2), "{stderr}");
    assert!(!run.exists(), "run --cuts 6 left {}", run.display());
    // The corrected command, into the same --output (a small budget keeps it quick).
    let result = evolve("3", "evaluations=10,seconds=60");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let lineage: Value =
        serde_json::from_slice(&std::fs::read(output.join("lineage.json")).unwrap()).unwrap();
    assert_eq!(lineage["rounds"].as_array().map(Vec::len), Some(2));
    assert_eq!(lineage["cuts"], serde_json::json!([0, 1, 2]));
}

/// `evolve` into an `--output` that already holds something (here a stale `control/`) exits 2
/// before it writes anything; it used to run every round first, write `lineage.json` and only then
/// refuse, so the corrected retry was refused as "already holds a lineage". Once the directory is
/// emptied the same command runs into it.
#[test]
fn evolve_into_a_non_empty_output_exits_2_first_and_runs_once_it_is_emptied() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("lineage");
    std::fs::create_dir_all(output.join("control")).unwrap();
    std::fs::write(output.join("control/receipt.json"), b"{}\n").unwrap();
    let evolve = || {
        jope()
            .args(["pipeline", "evolve", "--dataset"])
            .arg(dataset())
            .arg("--output")
            .arg(&output)
            .args([
                "--rounds",
                "1",
                "--control",
                "--budget",
                "evaluations=4,seconds=60",
            ])
            .args(["--policy", "skip", "--emit", "rust"])
            .output()
            .unwrap()
    };
    let result = evolve();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("is not empty (control)"), "{stderr}");
    let names = |dir: &Path| {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    };
    assert_eq!(
        names(&output),
        ["control"],
        "the refusal wrote into --output"
    );
    assert_eq!(names(&output.join("control")), ["receipt.json"]);
    std::fs::remove_dir_all(output.join("control")).unwrap();
    let result = evolve();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let lineage: Value =
        serde_json::from_slice(&std::fs::read(output.join("lineage.json")).unwrap()).unwrap();
    assert_eq!(lineage["rounds"].as_array().map(Vec::len), Some(1));
    assert!(lineage["control"].is_object());
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

#[test]
fn sigterm_during_a_round_keeps_only_completed_rounds_in_the_lineage() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("lineage");
    let mut child = jope()
        .args([
            "pipeline",
            "evolve",
            "--dataset",
            dataset().to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--rounds",
            "3",
            "--control",
            "--budget",
            "evaluations=60,seconds=600",
            "--policy",
            "skip",
            "--emit",
            "rust",
        ])
        .spawn()
        .unwrap();
    let lineage = output.join("lineage.json");
    let rounds = || -> usize {
        std::fs::read(&lineage)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .map_or(0, |v| v["rounds"].as_array().map_or(0, Vec::len))
    };
    // Round 1 completed and is counted in the fsynced lineage; round 2's search has started.
    wait_for(
        "round 2 to start",
        Duration::from_secs(600),
        || rounds() == 1 && output.join("round-002/prepared/prepared.json").exists(),
        &mut child,
    );
    terminate(&child);
    let result = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(143), "{stderr}");
    assert_eq!(rounds(), 1, "only the completed round stays in the lineage");
    assert!(output.join("round-001/receipt.json").exists());
    assert!(!output.join("round-002/receipt.json").exists());
    assert!(!output.join("round-003").exists() && !output.join("control").exists());
    // What remains still verifies.
    let verify = jope()
        .args(["pipeline", "lineage", "verify", "--lineage"])
        .arg(&lineage)
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );
}

#[test]
fn pipeline_cli_emits_fitted_cpp_and_records_executable_build() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("fitted-run");
    let result = jope()
        .args(["pipeline", "run", "--dataset"])
        .arg(dataset())
        .args([
            "--emit",
            "cpp17-fitted",
            "--policy",
            "skip",
            "--budget",
            "evaluations=1,seconds=60",
            "--source-commit",
            "0123456789abcdef0123456789abcdef01234567",
            "--output",
        ])
        .arg(&out)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["ok"], true);
    assert_eq!(receipt["artifacts"].as_array().unwrap().len(), 3);
    let build = receipt["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["language"] == "cpp17-fitted-standalone")
        .unwrap()["build"]
        .as_str()
        .unwrap();
    let mut words = build.split_whitespace();
    let compiled = Command::new(words.next().unwrap())
        .args(words)
        .current_dir(out.join("emit"))
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert!(out.join("emit/model-cpp17-fitted").is_file());
    let verified = jope()
        .args(["pipeline", "verify-receipt", "--evidence", "--receipt"])
        .arg(out.join("receipt.json"))
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
}

#[test]
fn kernel2_profile_cli_publishes_v4_verifies_evidence_and_defaults_to_fitted_cpp() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("run");
    let mut argv = run_args(&directory, "evaluations=1,seconds=60");
    argv.truncate(argv.len() - 2); // No explicit --emit: exercise the profile's default.
    let result = jope()
        .args(argv)
        .args([
            "--search-profile",
            "kernel2-baseline",
            "--refine-top-k",
            "0",
            "--refine-trials",
            "0",
            "--source-commit",
            &"a".repeat(40),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(receipt["schema"], "jope.pipeline-receipt/v4");
    assert_eq!(receipt["search"]["profile_id"], "kernel2-baseline");
    assert_eq!(receipt["search"]["fault_policy"]["kind"], "reject_hard");
    assert_eq!(receipt["search"]["completed_evaluations"], 1);
    assert!(directory.join("emit/cpp17-fitted/standalone.cpp").is_file());
    let result = jope()
        .args(["pipeline", "verify-receipt", "--receipt"])
        .arg(directory.join("receipt.json"))
        .args(["--evidence", "--json"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap(),
        receipt
    );
    std::fs::write(directory.join("search/cut-0/observations.json"), b"[]").unwrap();
    let result = jope()
        .args(["pipeline", "verify-receipt", "--receipt"])
        .arg(directory.join("receipt.json"))
        .args(["--evidence", "--json"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap()["ok"],
        false
    );
}

#[test]
fn kernel2_cli_refuses_unsupported_bounds_and_the_corrected_command_succeeds() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("run");
    for bad in [
        ["--cuts", "2"],
        ["--threads", "2"],
        ["--islands", "2"],
        ["--incumbent", "missing"],
    ] {
        let result = jope()
            .args(run_args(&directory, "evaluations=4,seconds=60"))
            .args([
                "--search-profile",
                "kernel2-baseline",
                "--refine-top-k",
                "1",
                "--refine-trials",
                "4",
            ])
            .args(bad)
            .output()
            .unwrap();
        assert_eq!(
            result.status.code(),
            Some(2),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!directory.exists());
    }
    let result = jope()
        .args(run_args(&directory, "evaluations=4,seconds=60"))
        .args([
            "--search-profile",
            "kernel2-baseline",
            "--refine-top-k",
            "1",
            "--refine-trials",
            "4",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(directory.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["schema"], "jope.pipeline-receipt/v4");
}
