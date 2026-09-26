//! W1 follow-up 2b: the shipped `jope-parallel outer-release` verbs.
//!
//! The shipped binary cannot produce an exported fitted subject (every fit is
//! fenced until a production namespace verifier exists), so the positive
//! grant -> score -> release path runs through the same clap command and
//! `outer_release_cli::run` in
//! `gp::fit_contract_tests::the_outer_release_cli_grants_scores_and_releases_once`.
//! Here the shipped entry point proves its wiring and its refusals: init,
//! events, a grant refused for an unexported subject and a release refused
//! without a withheld score are all recorded in the hash-chained ledger, and
//! nothing is ever released.

use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn cli(args: &[&str], root: &Path) -> (bool, Value, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let value = serde_json::from_str(stdout.trim()).unwrap_or(Value::Null);
    (
        output.status.success(),
        value,
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

#[test]
fn the_shipped_outer_release_verbs_record_refusals_and_release_nothing() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let authority = root.join("authority");
    let a = authority.to_str().unwrap();

    let (ok, init, error) = cli(&["outer-release", "init", "--authority", a], &root);
    assert!(ok, "outer-release init failed: {error}");
    assert_eq!(init["instance"].as_str().map(str::len), Some(64), "{init}");
    let (ok, again, _) = cli(&["outer-release", "init", "--authority", a], &root);
    assert!(!ok, "an existing authority was re-initialized: {again}");

    // A predictor with no handoff is not an exported sealed selection.
    fs::create_dir_all(root.join("search")).unwrap();
    fs::write(root.join("predictor.json"), b"{}").unwrap();
    fs::write(root.join("sealed.csv"), b"1,2\n").unwrap();
    let (ok, _, error) = cli(
        &[
            "outer-release",
            "grant",
            "--authority",
            a,
            "--root",
            root.to_str().unwrap(),
            "--search-output",
            root.join("search").to_str().unwrap(),
            "--predictor",
            root.join("predictor.json").to_str().unwrap(),
            "--test",
            root.join("sealed.csv").to_str().unwrap(),
        ],
        &root,
    );
    assert!(!ok, "a grant was issued for an unexported predictor");
    assert!(error.contains("handoff"), "{error}");

    let visible = root.join("released.json");
    let (ok, _, error) = cli(
        &[
            "outer-release",
            "release",
            "--authority",
            a,
            "--grant-id",
            &"a".repeat(64),
            "--output",
            visible.to_str().unwrap(),
        ],
        &root,
    );
    assert!(!ok, "a score was released without a withheld score");
    assert!(error.contains("no withheld score"), "{error}");
    assert!(!visible.exists());

    let (ok, events, error) = cli(&["outer-release", "events", "--authority", a], &root);
    assert!(ok, "{error}");
    let kinds: Vec<&str> = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["event"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["grant_refused", "release_refused"]);
    assert_eq!(events["events"][0]["labels_opened"], false);

    let help = Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&help.stdout).contains("outer-release"));
}
