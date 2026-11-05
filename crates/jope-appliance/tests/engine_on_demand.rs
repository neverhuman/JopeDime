//! After a batch `run` (which uses an in-process engine and leaves none behind), the engine
//! commands start the shared engine on demand, and the model `run --out` wrote loads, verifies
//! and predicts on its own exactly as `predict --model RUN` does.
#![cfg(all(feature = "engine", feature = "mcp"))]
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_jopedime");

fn jopedime(home: &std::path::Path, dir: &std::path::Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .current_dir(dir)
        .env_clear()
        .env("HOME", dir)
        .env("JOPEDIME_HOME", home)
        .env("JAIN_ACCEPT_TERMS", "1")
        .env("JAIN_NO_UPDATE_CHECK", "1")
        .output()
        .unwrap()
}

fn ok(output: &Output, what: &str) {
    assert!(
        output.status.success(),
        "{what}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn engine_commands_start_the_engine_and_a_saved_model_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let demo = include_str!("../assets/demos/regression_parkinsons.csv");
    let lines: Vec<&str> = demo.lines().collect();
    std::fs::write(dir.path().join("train.csv"), lines[..996].join("\n") + "\n").unwrap();
    let mut test = vec![lines[0]];
    test.extend(&lines[lines.len() - 5..]);
    std::fs::write(dir.path().join("test.csv"), test.join("\n") + "\n").unwrap();

    let run = jopedime(
        &home,
        dir.path(),
        &[
            "run",
            "train.csv",
            "--target",
            "target",
            "--effort",
            "fast",
            "--budget",
            r#"{"evaluations":"4","wall_seconds":null}"#,
            "--out",
            "model",
        ],
    );
    ok(&run, "run --out");
    assert!(
        !home.join("daemon.json").exists(),
        "a batch run leaves no shared engine behind"
    );
    let receipt: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    let id = receipt["run"]["run_id"].as_str().unwrap().to_owned();

    // No engine is running: `status` starts one instead of failing.
    let status = jopedime(&home, dir.path(), &["status", &id, "--json"]);
    ok(&status, "status");
    let snapshot: serde_json::Value = serde_json::from_slice(&status.stdout).unwrap();
    assert_eq!(snapshot["state"], "completed");

    ok(
        &jopedime(
            &home,
            dir.path(),
            &[
                "predict",
                "--model",
                &id,
                "--input",
                "test.csv",
                "--output",
                "by_run.csv",
            ],
        ),
        "predict --model RUN",
    );
    let verify = jopedime(&home, dir.path(), &["model", "verify", "--model", "model"]);
    ok(&verify, "model verify on the --out directory");
    let inspected: serde_json::Value = serde_json::from_slice(&verify.stdout).unwrap();
    assert_eq!(inspected["executable"], true);
    for path in ["model", "model/model"] {
        let out = format!("by_{}.csv", path.replace('/', "_"));
        ok(
            &jopedime(
                &home,
                dir.path(),
                &[
                    "predict", "--model", path, "--input", "test.csv", "--output", &out,
                ],
            ),
            "predict --model DIR",
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join(&out)).unwrap(),
            std::fs::read_to_string(dir.path().join("by_run.csv")).unwrap(),
            "{path} predicts exactly as the run does"
        );
    }
    // A tampered part is refused, not silently used.
    let pipeline = dir.path().join("model/model/pipeline.json");
    let mut bytes = std::fs::read(&pipeline).unwrap();
    bytes.push(b' ');
    std::fs::write(&pipeline, bytes).unwrap();
    let tampered = jopedime(&home, dir.path(), &["model", "verify", "--model", "model"]);
    assert!(!tampered.status.success());

    let _ = jopedime(&home, dir.path(), &["serve", "--stop"]);
}
