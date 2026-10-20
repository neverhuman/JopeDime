use serde_json::Value;
use std::{fs, process::Command};

fn csv(root: &std::path::Path) -> std::path::PathBuf {
    let path = root.join("train.csv");
    let mut text = String::from("x0,x1,target\n");
    for i in 0..24 {
        text.push_str(&format!(
            "{},{},{}\n",
            i as f64 / 24.0,
            (i % 5) as f64,
            1.5 * i as f64 / 24.0 - (i % 5) as f64
        ));
    }
    fs::write(&path, text).unwrap();
    path
}
fn run(root: &std::path::Path, capture: Option<&std::path::Path>) -> std::process::Output {
    let input = csv(root);
    let mut command = Command::new(env!("CARGO_BIN_EXE_jope"));
    command.env_remove("JOPE_GP_SOURCE_DIR").args([
        "pipeline",
        "invent",
        "--train",
        input.to_str().unwrap(),
        "--target",
        "target",
        "--work-dir",
        root.join("work").to_str().unwrap(),
        "--budget",
        "evaluations=2,seconds=120",
        "--cuts",
        "1",
        "--top-k",
        "2",
        "--source-commit",
        &"a".repeat(40),
        "--json",
    ]);
    if let Some(path) = capture {
        command.env("JOPE_GP_SOURCE_DIR", path);
    }
    command.output().unwrap()
}
#[test]
fn invent_captures_all_draws_before_budget_and_preserves_results() {
    let enabled = tempfile::tempdir().unwrap();
    let disabled = tempfile::tempdir().unwrap();
    let journal = enabled.path().join("journals");
    let a = run(enabled.path(), Some(&journal));
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    let b = run(disabled.path(), None);
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let a: Value = serde_json::from_slice(&a.stdout).unwrap();
    let b: Value = serde_json::from_slice(&b.stdout).unwrap();
    let genomes = |v: &Value| {
        v["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["genome"].clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        genomes(&a),
        genomes(&b),
        "observational capture changed selection"
    );
    let mut records = Vec::new();
    for entry in fs::read_dir(journal).unwrap() {
        let path = entry.unwrap().path();
        for line in fs::read_to_string(path).unwrap().lines() {
            records.push(serde_json::from_str::<Value>(line).unwrap());
        }
    }
    assert!(
        records.len() > 2,
        "budgeted evaluations must not filter capture"
    );
    assert!(records.iter().any(|r| r["engine"] == "pipeline"));
    assert!(records
        .iter()
        .any(|r| r["coordinate"].as_str().unwrap().contains("random")));
    assert!(
        records
            .iter()
            .map(|r| r["program"].as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 2
    );
}
#[test]
fn enabled_capture_failure_stops_producer_instead_of_losing_source() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("not-a-directory");
    fs::write(&file, "retain this file").unwrap();
    let result = run(root.path(), Some(&file));
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("GP source capture initialization failed")
    );
    assert_eq!(fs::read_to_string(file).unwrap(), "retain this file");
}
