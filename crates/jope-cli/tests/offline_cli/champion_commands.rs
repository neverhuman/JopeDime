use super::*;

#[test]
fn champions_pull_and_push_fail_closed() {
    for flag in ["--pull", "--push"] {
        let output = jope(Path::new("."), &["champions", "--json", flag]);
        assert!(!output.status.success());
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["ok"], false);
        assert_eq!(report["schema"], "jope.champions.v1");
        let diagnostic = report.to_string();
        assert!(
            diagnostic
                .contains("live S3 champion access is unavailable in this command; use --local")
        );
        assert!(!diagnostic.contains("no AWS credentials"));
    }
}

#[test]
fn train_step_without_local_fails_closed() {
    let output = jope(Path::new("."), &["train-step", "--json", "--env", "dev"]);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert_eq!(report["schema"], "jope.train-step.v1");
}

#[test]
fn local_champions_and_train_step_report_supported_and_unsupported_fixtures() {
    let root = synthetic_root();
    let root = root.to_str().expect("utf8 path");
    let champions = jope(
        Path::new("."),
        &["champions", "--local", "--json", "--root", root],
    );
    assert!(
        champions.status.success(),
        "{}",
        String::from_utf8_lossy(&champions.stderr)
    );
    let bank: Value = serde_json::from_slice(&champions.stdout).unwrap();
    assert_eq!(bank["ok"], true);
    assert_eq!(bank["env"], "local");
    assert_eq!(bank["dataset_count"], 40);
    assert_eq!(bank["supported_count"], 8);
    assert_eq!(bank["unsupported_count"], 32);
    let rows = bank["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 8);
    assert!(rows
        .iter()
        .all(|row| row["task"] == "regression" && row["hold_r2"].as_f64().unwrap().is_finite()));
    let mut searched: Vec<&str> = rows
        .iter()
        .map(|row| row["scenario"].as_str().unwrap())
        .collect();
    searched.sort_unstable();
    assert_eq!(
        searched,
        [
            "collinearity",
            "constant_target",
            "extrapolation",
            "heavy_tailed",
            "heteroscedastic",
            "linear",
            "nonlinear",
            "sparse_signal"
        ]
    );
    let unsupported = bank["unsupported"].as_array().unwrap();
    assert_eq!(unsupported.len(), 32);
    for task in ["binary", "time_series", "multiclass"] {
        let excluded: Vec<_> = unsupported
            .iter()
            .filter(|row| row["task"] == task)
            .collect();
        assert_eq!(excluded.len(), 10, "{task}");
        let reason = format!("unsupported task {task}");
        assert!(excluded
            .iter()
            .all(|row| row["reason"].as_str().unwrap().contains(&reason)));
    }
    let mut regression: Vec<(&str, &str)> = unsupported
        .iter()
        .filter(|row| row["task"] == "regression")
        .map(|row| {
            (
                row["scenario"].as_str().unwrap(),
                row["reason"].as_str().unwrap(),
            )
        })
        .collect();
    regression.sort_unstable();
    assert_eq!(regression.len(), 2);
    for ((scenario, reason), expected) in regression.iter().zip(["categorical", "missing_values"]) {
        assert_eq!(*scenario, expected);
        assert!(reason.contains("integer, non-null"), "{scenario}: {reason}");
    }

    let step = jope(
        Path::new("."),
        &["train-step", "--local", "--json", "--root", root],
    );
    assert!(
        step.status.success(),
        "{}",
        String::from_utf8_lossy(&step.stderr)
    );
    let report: Value = serde_json::from_slice(&step.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["dataset_count"], 40);
    assert_eq!(report["supported_count"], 8);
    assert_eq!(report["unsupported_count"], 32);
    assert_eq!(report["labels"], bank["rows"]);
    assert_eq!(report["unsupported"], bank["unsupported"]);
    assert_eq!(report["router"]["dim"], 128);
    assert_eq!(
        report["router"]["classes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|class| class["n"].as_u64().unwrap())
            .sum::<u64>(),
        8
    );
    assert!(report["note"]
        .as_str()
        .unwrap()
        .contains("does not update jope-policy"));
}
