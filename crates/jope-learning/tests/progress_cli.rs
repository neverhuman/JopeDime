//! The shipped `jope-learning progress` verbs (W8 product integration): an engine corpus and
//! checkpoint reach the committed graph on an explicitly named `file://` store, and nothing is
//! written without one. No remote object store is contacted.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn pretty(v: &Value) -> Vec<u8> {
    serde_json::to_vec_pretty(v).unwrap()
}

/// A published corpus directory in the engine's `parallel.teaching_corpus.v1` layout.
fn write_corpus(dir: &Path) -> String {
    fs::create_dir_all(dir).unwrap();
    let records = b"abcdefghijkl".to_vec();
    let vocabulary = pretty(&json!({"tokens": ["pipe"]}));
    let labels = b"{\"label\":1}\n".to_vec();
    let contexts = sha(b"contexts");
    let index = pretty(
        &json!({"schema": "parallel.teaching_corpus.v1", "qualification": true,
        "vocabulary_sha256": sha(&vocabulary), "context_index_sha256": contexts,
        "records_sha256": sha(&records), "record_bytes": 4, "records": 3,
        "legal_sets": [], "contexts": [], "groups": []}),
    );
    let provenance = pretty(&json!({"index_sha256": sha(&index), "labels_sha256": sha(&labels)}));
    let publication = pretty(&json!({"schema": "parallel.corpus_publication.v1",
        "contexts_sha256": contexts, "vocabulary_sha256": sha(&vocabulary), "observations": [],
        "qualification": true}));
    for (name, bytes) in [
        ("records.bin", records),
        ("vocabulary.json", vocabulary),
        ("labels.jsonl", labels),
        ("index.json", index.clone()),
        ("provenance.json", provenance),
        ("publication.json", publication),
    ] {
        fs::write(dir.join(name), bytes).unwrap();
    }
    sha(&index)
}

/// Publish checkpoint `step` of `runs/train` in the engine's `parallel.checkpoint.v1` layout.
fn publish_checkpoint(root: &Path, step: u64, corpus_index: &str) {
    let dir = root.join(format!("runs/train/checkpoints/step-{step}"));
    fs::create_dir_all(&dir).unwrap();
    let objects = [
        (
            "model.ot",
            "model_sha256",
            format!("model {step}").into_bytes(),
        ),
        (
            "optimizer.ot",
            "optimizer_sha256",
            format!("optimizer {step}").into_bytes(),
        ),
        (
            "optimizer.json",
            "optimizer_metadata_sha256",
            b"{}".to_vec(),
        ),
        ("config.json", "config_sha256", b"{}".to_vec()),
    ];
    let mut state = json!({"schema": "parallel.checkpoint.v1", "step": step, "global_batch": 4,
        "corpus_sha256": corpus_index});
    for (name, field, bytes) in &objects {
        fs::write(dir.join(name), bytes).unwrap();
        state[*field] = sha(bytes).into();
    }
    let state = pretty(&state);
    fs::write(dir.join("state.json"), &state).unwrap();
    fs::write(
        root.join("runs/train/latest.json"),
        pretty(
            &json!({"checkpoint": format!("runs/train/checkpoints/step-{step}"),
            "state_sha256": sha(&state), "step": step}),
        ),
    )
    .unwrap();
}

fn cli(state: &Path, store: Option<&Path>, args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jope-learning"));
    cmd.env_remove("JOPE_LEARNING_URL")
        .arg("--state")
        .arg(state);
    if let Some(store) = store {
        cmd.arg("--store").arg(store);
    }
    cmd.args(args).output().unwrap()
}

fn json_out(out: &Output) -> Value {
    assert!(
        out.status.success(),
        "exit {:?}: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn progress_verbs_publish_engine_events_to_an_explicit_store_only() {
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let store = dir.path().join("store");
    fs::create_dir_all(&store).unwrap();
    let engine = dir.path().join("engine");
    let corpus_index = write_corpus(&engine.join("corpora/teaching"));
    let run = ["--run", "run-cli", "--publisher", "xbabe2"];
    let commit = "c4f60b4c7a2e27a31fc41f875650c89900dc32ed";

    // No store named: refused, and nothing is written.
    let mut init = vec!["progress", "init"];
    init.extend(run);
    init.extend(["--source-commit", commit, "--unit", "update:train"]);
    let out = cli(&state, None, &init);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("explicitly named store"));
    assert!(!state.exists(), "a refused verb wrote local state");

    // The effective configuration is the one adopted by owner delegation.
    let config = json_out(&cli(&state, None, &["progress", "config"]));
    assert_eq!(config["approval"], "adopted-by-owner-delegation-2026-09-26");
    assert_eq!(config["adopted_defaults"], true);
    assert_eq!(config["values"]["cadence"]["checkpoint_updates"], 100);

    // Init records the manifest and delivers it.
    let out = json_out(&cli(&state, Some(&store), &init));
    let unit = out["units"][0]["work_id"].as_str().unwrap().to_string();
    assert_eq!(
        out["config"]["approval"],
        "adopted-by-owner-delegation-2026-09-26"
    );

    // The corpus freeze commits at once (exit 0 means flushed).
    let corpus = engine.join("corpora/teaching");
    let mut args = vec!["progress", "corpus"];
    args.extend(run);
    args.push(corpus.to_str().unwrap());
    let out = json_out(&cli(&state, Some(&store), &args));
    assert_eq!(out["step"]["publication"]["outcome"], "advanced");
    assert_eq!(out["step"]["publication"]["height"], 0);

    // A checkpoint under the proposed cadence is counted, then shutdown makes it an update.
    publish_checkpoint(&engine, 20, &corpus_index);
    let mut args = vec!["progress", "checkpoint"];
    args.extend(run);
    args.extend([
        "--engine-root",
        engine.to_str().unwrap(),
        "--training-run",
        "runs/train",
        "--corpus",
        corpus.to_str().unwrap(),
    ]);
    let out = json_out(&cli(&state, Some(&store), &args));
    assert_eq!(out["step"]["checkpoint"]["decision"], "deferred");
    args.push("--shutdown");
    let out = json_out(&cli(&state, Some(&store), &args));
    assert_eq!(out["step"]["checkpoint"]["decision"], "mapped");
    assert_eq!(out["step"]["checkpoint"]["update_index"], 0);
    assert_eq!(out["step"]["checkpoint"]["consumed_rows"], 80);
    assert_eq!(out["step"]["publication"]["height"], 1);

    // Scheduling a planned unit, then flushing, commits it.
    let mut args = vec!["progress", "schedule"];
    args.extend(run);
    args.extend(["--unit", &unit]);
    json_out(&cli(&state, Some(&store), &args));
    let mut args = vec!["progress", "flush"];
    args.extend(run);
    let out = json_out(&cli(&state, Some(&store), &args));
    assert_eq!(out["step"]["publication"]["height"], 2);

    // Status is read-only and reports the committed graph.
    let before = walk(&store);
    let status = json_out(&cli(
        &state,
        Some(&store),
        &["progress", "status", "--run", "run-cli"],
    ));
    assert_eq!(status["head"]["height"], 2);
    assert_eq!(status["summary"]["checkpoints"], 1);
    assert_eq!(status["summary"]["corpora"], 1);
    assert_eq!(status["summary"]["consumed_rows"], 80);
    assert_eq!(walk(&store), before, "status wrote to the store");

    // A publisher the manifest does not name is refused.
    let args = [
        "progress",
        "flush",
        "--run",
        "run-cli",
        "--publisher",
        "xbabe3",
    ];
    let out = cli(&state, Some(&store), &args);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn follow_drives_the_cadence_until_the_run_completes() {
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let store = dir.path().join("store");
    fs::create_dir_all(&store).unwrap();
    let engine = dir.path().join("engine");
    let corpus = engine.join("corpora/teaching");
    let corpus_index = write_corpus(&corpus);
    let run = ["--run", "run-follow", "--publisher", "xbabe2"];
    let mut init = vec!["progress", "init"];
    init.extend(run);
    init.extend([
        "--source-commit",
        "c4f60b4c7a2e27a31fc41f875650c89900dc32ed",
        "--unit",
        "update:train",
    ]);
    json_out(&cli(&state, Some(&store), &init));
    let mut args = vec!["progress", "corpus"];
    args.extend(run);
    args.push(corpus.to_str().unwrap());
    json_out(&cli(&state, Some(&store), &args));

    let mut follow = vec!["progress", "follow"];
    follow.extend(run);
    follow.extend([
        "--engine-root",
        engine.to_str().unwrap(),
        "--training-run",
        "runs/train",
        "--corpus",
        corpus.to_str().unwrap(),
        "--poll-ms",
        "1",
    ]);
    // A running run: a published checkpoint below the cadence is counted, not published, and
    // the bounded follow stops without completing.
    publish_checkpoint(&engine, 30, &corpus_index);
    let mut bounded = follow.clone();
    bounded.extend(["--max-polls", "2"]);
    let out = json_out(&cli(&state, Some(&store), &bounded));
    assert_eq!(out["completed"], false);
    assert_eq!(out["polls"], 2);
    assert_eq!(out["head"]["height"], 0, "{out}");
    // The run completes: the follow shuts down, maps the counted steps as update 0 and exits.
    publish_checkpoint(&engine, 45, &corpus_index);
    fs::write(engine.join("runs/train/complete.json"), b"{}").unwrap();
    let out = json_out(&cli(&state, Some(&store), &follow));
    assert_eq!(out["completed"], true);
    assert_eq!(out["polls"], 1);
    assert_eq!(out["owed"], 0);
    assert_eq!(out["head"]["height"], 1);
    let mapped = &out["events"][0]["checkpoint"];
    assert_eq!(mapped["decision"], "mapped");
    assert_eq!(mapped["consumed_rows"], 45 * 4);
    assert_eq!(mapped["due"]["reason"], "shutdown");
}

/// Run the binary with an empty environment (plus `vars`) and `HOME` in a temporary directory:
/// no credential, profile or AWS configuration of this host can reach it.
fn cli_clean_env(state: &Path, home: &Path, vars: &[(&str, &str)], args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jope-learning"));
    cmd.env_clear().env("HOME", home);
    for (k, v) in vars {
        cmd.env(k, v);
    }
    cmd.arg("--state").arg(state).args(args).output().unwrap()
}

#[test]
fn an_s3_store_is_configuration_only_and_misconfiguration_is_refused_first() {
    let dir = tempfile::tempdir().unwrap();
    let (state, home) = (dir.path().join("state"), dir.path().join("home"));
    fs::create_dir_all(&home).unwrap();
    let status = ["progress", "status", "--run", "run-s3"];
    let refused = |vars: &[(&str, &str)], needle: &str| {
        let out = cli_clean_env(&state, &home, vars, &status);
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert_eq!(out.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains(needle), "{needle}: {stderr}");
        assert!(!state.exists(), "a refused store wrote local state");
    };
    // Refused before any store is opened: a bare bucket, unsigned requests, and conditional
    // writes turned off.
    refused(
        &[("JOPE_LEARNING_URL", "s3://jope-learning-bucket")],
        "bucket root is refused",
    );
    refused(
        &[
            ("JOPE_LEARNING_URL", "s3://jope-learning-bucket/runs/run-s3"),
            ("JOPE_DATASETS_ANON", "1"),
        ],
        "unsigned",
    );
    refused(
        &[
            ("JOPE_LEARNING_URL", "s3://jope-learning-bucket/runs/run-s3"),
            ("AWS_CONDITIONAL_PUT", "disabled"),
        ],
        "AWS_CONDITIONAL_PUT",
    );
    // A well-formed s3:// store passes the checks and reaches the credential step, which fails
    // fast and by name with no credential source: the URL is configuration, not code, and no
    // request was made.
    for vars in [
        &[("JOPE_LEARNING_URL", "s3://jope-learning-bucket/runs/run-s3")][..],
        &[
            ("JOPE_LEARNING_URL", "s3://jope-learning-bucket/runs/run-s3"),
            ("AWS_CONDITIONAL_PUT", "etag"),
            ("AWS_REGION", "us-east-2"),
        ][..],
    ] {
        refused(vars, "no complete AWS credential source");
    }
}

fn walk(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push((path.display().to_string(), sha(&fs::read(&path).unwrap())));
            }
        }
    }
    out.sort();
    out
}
