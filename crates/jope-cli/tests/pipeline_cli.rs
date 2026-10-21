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

// laneD:B17-lm-refinement begin
/// `--refine-method lm` through the real binary: a kernel2 run whose one-literal seed (the one
/// jope-pipeline's LM run tests use) gives the damped least-squares refinement a `derive` stage to
/// fit; 8 trials hold that stage's surrogate minimum (one site plus five). The receipt records the
/// selection, a published refinement carries a surrogate stage, and `verify-receipt --evidence`,
/// which re-derives every refinement trace, accepts the run.
#[test]
fn kernel2_lm_refinement_records_its_selection_and_verifies_evidence() {
    use sha2::{Digest, Sha256};
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("run");
    let seed = "(pipe X (derive expr=\"1*c0\") (ridge))";
    let seeds = temp.path().join("seeds.json");
    let file = serde_json::json!({
        "schema": "jope.pipeline-seeds/v1",
        "genomes": [{"genome": seed, "program_sha256": format!("{:x}", Sha256::digest(seed))}],
    });
    std::fs::write(&seeds, serde_json::to_vec(&file).unwrap()).unwrap();
    let result = jope()
        .args(run_args(&directory, "evaluations=20,seconds=600"))
        .args([
            "--search-profile",
            "kernel2-baseline",
            "--refine-top-k",
            "1",
            "--refine-trials",
            "8",
            "--refine-method",
            "lm",
            "--rng-seed",
            "7",
            "--source-commit",
            &"a".repeat(40),
            "--json",
        ])
        .arg("--seed-genomes")
        .arg(&seeds)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(receipt["schema"], "jope.pipeline-receipt/v4");
    assert_eq!(
        receipt["search"]["engine_selection"],
        serde_json::json!({"refine": {
            "id": "feat-math.constant-refinement.damped-ls.v1",
            "jacobian": "sites",
            "tau": 1e-3,
        }})
    );
    let refinements: Value = serde_json::from_slice(
        &std::fs::read(directory.join("search/cut-0/refinements.json")).unwrap(),
    )
    .unwrap();
    assert!(
        refinements
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["refinement"]["trace"]["stages"]
                .as_array()
                .is_some_and(|stages| stages.iter().any(|s| s["kind"] == "surrogate"))),
        "{refinements}"
    );
    let verified = jope()
        .args(["pipeline", "verify-receipt", "--receipt"])
        .arg(directory.join("receipt.json"))
        .args(["--evidence", "--json"])
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&verified.stdout).unwrap(),
        receipt
    );
}
// laneD:B17-lm-refinement end
/// The opt-in `--events` feed: every line one compact `jope.invent.progress.v1` object, numbers
/// only, evaluations never decreasing, and a last line that is the end state.
fn assert_invent_progress(events: &Path, cuts: u64, eval_budget: u64) {
    let text = std::fs::read_to_string(events).unwrap();
    assert!(text.ends_with('\n'), "{text}");
    let lines: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect();
    assert!(!lines.is_empty(), "no progress records: {text}");
    assert!(lines
        .iter()
        .all(|l| l["schema"] == "jope.invent.progress.v1"));
    let mut evaluations = 0;
    let mut emitted = 0;
    for line in &lines {
        let object = line.as_object().unwrap();
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
        for field in ["best_hold_rmse", "best_hold_r2"] {
            assert!(
                object[field].is_number() || object[field].is_null(),
                "{field} in {line}"
            );
        }
        // Numbers only: no program source, genome or column name crosses stderr.
        assert!(
            object.iter().all(
                |(key, value)| matches!(key.as_str(), "schema" | "phase") || !value.is_string()
            ),
            "{line}"
        );
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
    assert!(emitted > 0, "no emitted tokens counted: {text}");
    let last = lines.last().unwrap();
    assert_eq!(last["phase"], "done");
    assert!(last["evaluations"].as_u64().unwrap() > 0);
    assert!(last["best_hold_rmse"].is_number());
    assert!(!text.contains("(pipe"), "program source in the events file");
}

fn no_progress_on_stderr(stderr: &[u8]) {
    let text = String::from_utf8_lossy(stderr);
    assert!(
        !text.contains("jope.invent.progress.v1"),
        "progress on stderr: {text}"
    );
}

/// `jope pipeline invent` the way jain-core's Jope engine launches it: a header CSV in, one
/// jope.candidates.v1 document out, `git` never spawned, and a refused input reported as JSON with
/// exit 2 before the work directory exists.
#[test]
fn invent_prints_candidates_and_refuses_bad_input_as_json() {
    let dir = tempfile::tempdir().unwrap();
    let train = dir.path().join("train.csv");
    let mut text = String::from("x0,x1,target\n");
    for i in 0..90 {
        let (a, b) = ((i % 9) as f64, (i as f64 * 0.7).cos());
        text.push_str(&format!(
            "{a},{b},{}\n",
            1.5 * a - 2.0 * b + 0.05 * (i % 4) as f64
        ));
    }
    std::fs::write(&train, text).unwrap();
    let marker = dir.path().join("git-calls");
    let path = format!(
        "{}:{}",
        fake_git(dir.path(), &marker).display(),
        std::env::var("PATH").unwrap()
    );
    let invent = |target: &str, work: &Path, events: Option<&Path>| {
        let mut command = jope();
        command
            .env("PATH", &path)
            .args(["pipeline", "invent", "--train", train.to_str().unwrap()])
            .args(["--target", target, "--work-dir", work.to_str().unwrap()])
            .args([
                "--budget",
                "evaluations=12,seconds=120",
                "--cuts",
                "2",
                "--top-k",
                "3",
            ])
            .args(["--source-commit", &"a".repeat(40), "--json"]);
        if let Some(events) = events {
            command.arg("--events").arg(events);
        }
        command.output().unwrap()
    };

    let work = dir.path().join("work");
    let events = dir.path().join("progress.ndjson");
    let result = invent("target", &work, Some(&events));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let doc: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(doc["schema"], "jope.candidates.v1");
    assert_eq!(doc["ok"], true);
    assert_eq!(doc["git_sha"], "a".repeat(40));
    assert_eq!(doc["feature_names"], serde_json::json!(["x0", "x1"]));
    let candidates = doc["candidates"].as_array().unwrap();
    assert!(!candidates.is_empty() && candidates.len() <= 3);
    assert!(candidates[0]["genome"]
        .as_str()
        .unwrap()
        .starts_with("(pipe X "));
    assert!(!marker.exists(), "git was spawned");
    assert_invent_progress(&events, 2, 24);
    no_progress_on_stderr(&result.stderr);

    // Without --events: no progress anywhere, stdout unchanged.
    let quiet = dir.path().join("quiet");
    let result = invent("target", &quiet, None);
    assert!(result.status.success());
    no_progress_on_stderr(&result.stderr);
    let doc: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(doc["schema"], "jope.candidates.v1");
    assert_eq!(doc["ok"], true);

    // --events may not alias the input, the work directory or a directory; refused before
    // anything is written, with the input left intact.
    let before = std::fs::read(&train).unwrap();
    let aliased = dir.path().join("aliased");
    for (events, says) in [
        (train.clone(), "--train"),
        (aliased.clone(), "--work-dir"),
        (aliased.join("progress.ndjson"), "--work-dir"),
        (dir.path().to_path_buf(), "directory"),
    ] {
        let result = invent("target", &aliased, Some(&events));
        assert_eq!(result.status.code(), Some(2), "{}", events.display());
        let doc: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(doc["ok"], false);
        assert!(doc["error"].as_str().unwrap().contains(says), "{doc}");
        assert!(!aliased.exists());
    }
    assert_eq!(std::fs::read(&train).unwrap(), before);

    let refused = dir.path().join("refused");
    let result = invent("missing", &refused, None);
    assert_eq!(result.status.code(), Some(2));
    let doc: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(doc["schema"], "jope.candidates.v1");
    assert_eq!(doc["ok"], false);
    assert!(doc["error"].as_str().unwrap().contains("--target"), "{doc}");
    assert!(!refused.exists());
}
