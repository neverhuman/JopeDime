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

/// The opt-in `--events` feed: every line one compact `jope.invent.progress.v1` object, numbers
/// only and evaluations never decreasing; delivery including the final record is best-effort.
fn assert_invent_progress(events: &Path, cuts: u64, eval_budget: u64) {
    let text = std::fs::read_to_string(events).unwrap();
    let records: Vec<Value> = text
        .split_inclusive('\n')
        .filter(|line| line.ends_with('\n'))
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect();
    assert!(records
        .iter()
        .all(|l| l["schema"] == "jope.invent.progress.v1"));
    assert!(text.split_inclusive('\n').all(|line| line.len() <= 2048));
    // One sequence over every kind.
    let mut previous_sequence = None;
    for record in &records {
        let sequence = record["sequence"].as_u64().unwrap();
        assert!(previous_sequence.is_none_or(|previous| sequence > previous));
        previous_sequence = Some(sequence);
        assert!(
            matches!(record["kind"].as_str(), Some("progress" | "candidate")),
            "{record}"
        );
        // Aside from fixed labels, numbers or null.
        assert!(
            record
                .as_object()
                .unwrap()
                .iter()
                .all(|(key, value)| !value.is_string()
                    || matches!(key.as_str(), "schema" | "kind" | "phase" | "status")),
            "{record}"
        );
    }
    let lines: Vec<&Value> = records.iter().filter(|r| r["kind"] == "progress").collect();
    let candidates: Vec<&Value> = records
        .iter()
        .filter(|r| r["kind"] == "candidate")
        .collect();
    let mut evaluations = 0;
    let mut emitted = 0;
    for line in &lines {
        let object = line.as_object().unwrap();
        assert!(matches!(
            line["phase"].as_str(),
            Some("exploration" | "comparison" | "done")
        ));
        for field in [
            "cut",
            "cuts",
            "generation",
            "evaluations",
            "eval_budget",
            "measured",
            "failed",
            "elapsed_s",
            "seconds_budget",
            "emitted_tokens",
            "emitted_tokens_per_second",
        ] {
            assert!(object[field].is_number(), "{field} in {line}");
        }
        for field in ["best_hold_rmse", "best_hold_r2", "generations"] {
            assert!(
                object[field].is_number() || object[field].is_null(),
                "{field} in {line}"
            );
        }
        match line["population_r2"].as_array() {
            Some(values) => {
                let values: Vec<f64> = values.iter().map(|v| v.as_f64().unwrap()).collect();
                assert!(values.len() <= 64, "{line}");
                assert!(values.iter().all(|v| v.is_finite()), "{line}");
                assert!(values.windows(2).all(|w| w[0] >= w[1]), "{line}");
            }
            None => assert!(line["population_r2"].is_null(), "{line}"),
        }
        assert_eq!(line["cuts"], cuts);
        assert_eq!(line["eval_budget"], eval_budget);
        assert!(line["cut"].as_u64().unwrap() < cuts);
        let now = line["evaluations"].as_u64().unwrap();
        assert!(now >= evaluations, "evaluations went backwards: {text}");
        assert_eq!(
            now,
            line["measured"].as_u64().unwrap() + line["failed"].as_u64().unwrap()
        );
        evaluations = now;
        let tokens = line["emitted_tokens"].as_u64().unwrap();
        assert!(tokens >= emitted, "emitted_tokens went backwards: {text}");
        let by = line["emitted_tokens_by_language"].as_object().unwrap();
        let languages: Vec<&str> = by.keys().map(String::as_str).collect();
        assert_eq!(
            languages,
            ["cpp17", "javascript", "python", "rust"],
            "{line}"
        );
        assert_eq!(
            by.values().map(|n| n.as_u64().unwrap()).sum::<u64>(),
            tokens
        );
        emitted = tokens;
    }
    if !lines.is_empty() {
        assert!(emitted > 0, "no emitted tokens counted: {text}");
    }
    // This small run's records fit the queue: both kinds and a snapshot arrive.
    assert!(!candidates.is_empty(), "no candidate records: {text}");
    assert!(lines
        .iter()
        .any(|l| l["population_r2"].as_array().is_some_and(|p| !p.is_empty())));
    assert!(
        !text.contains("x0") && !text.contains("x1"),
        "column name in the events file"
    );
    // Candidates: scores only, at most five a second (the end may add the waiting one).
    let times: Vec<f64> = candidates
        .iter()
        .map(|c| c["elapsed_s"].as_f64().unwrap())
        .collect();
    assert!(
        times[..times.len().saturating_sub(1)]
            .windows(2)
            .all(|w| w[1] - w[0] >= 0.199),
        "{times:?}"
    );
    for candidate in &candidates {
        assert!(candidate["generation"].is_u64() && candidate["cut"].as_u64().unwrap() < cuts);
        match candidate["status"].as_str() {
            Some("measured") => assert!(candidate["hold_rmse"].as_f64().unwrap().is_finite()),
            Some("failed") => assert!(candidate["hold_rmse"].is_null()),
            other => panic!("status {other:?}"),
        }
        assert!(candidate["hold_r2"].is_null() || candidate["hold_r2"].is_number());
        let keys: Vec<&str> = candidate
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "cut",
                "elapsed_s",
                "generation",
                "hold_r2",
                "hold_rmse",
                "kind",
                "schema",
                "sequence",
                "status"
            ]
        );
    }
    assert!(!text.contains("(pipe"), "program source in the events file");
}

fn no_progress_on_stderr(stderr: &[u8]) {
    let text = String::from_utf8_lossy(stderr);
    assert!(
        !text.contains("jope.invent.progress.v1"),
        "progress on stderr: {text}"
    );
}

// The test target root is tests/pipeline_cli.rs, so its sub-tests live under tests/pipeline_cli/.
#[cfg(test)]
mod pipeline_cli {
    use super::*;
    mod tests;
}
