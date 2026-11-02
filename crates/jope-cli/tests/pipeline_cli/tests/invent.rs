use super::*;

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

    // Without --events: no progress feed; stdout retains its schema/channel.
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
