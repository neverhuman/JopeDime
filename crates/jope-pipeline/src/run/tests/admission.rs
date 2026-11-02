use super::*;

#[test]
fn classification_is_refused_before_writing_a_regression_run() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("binary");
    std::fs::create_dir(&input).unwrap();
    // An admitted binary catalog and genuine 0/1 labels must never reach
    // this regression-only search, even though both parse as numeric CSV.
    let csv: String = (0..60).map(|i| format!("{},{}\n", i, i % 2)).collect();
    let mut row: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../datasets/test/pmlb-229-pwlinear/metadata.json"
    ))
    .unwrap();
    row["task"] = "binary".into();
    row["target"]["type"] = "binary".into();
    row["shape"] = serde_json::json!({
        "rows":120,"features":1,"categorical_features":0,"columns":2
    });
    row["split"]["train_rows"] = 60.into();
    row["split"]["test_rows"] = 60.into();
    for name in ["train", "test"] {
        row["files"][name] = serde_json::json!({
            "sha256": crate::sha256_hex(csv.as_bytes()), "bytes": csv.len(),
            "rows": 60, "columns": 2, "has_header": false
        });
        std::fs::write(input.join(format!("{name}.csv")), &csv).unwrap();
    }
    std::fs::write(input.join("metadata.json"), row.to_string()).unwrap();
    let output = temp.path().join("run");
    let options = RunOptions {
        budget: Budget {
            evaluations: 1,
            seconds: 1,
        },
        seed: 1,
        rng_seed: 1,
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(input.display().to_string(), output.clone())
    };
    let error = format!("{:#}", run(&options).unwrap_err());
    assert!(error.contains("task binary is unsupported"), "{error}");
    assert!(!output.exists(), "refusal left a misleading run directory");
}

/// The dataset is unresolvable, so this also proves the policy is checked
/// before any other work — including anything that writes to `--output`.
#[test]
fn require_fails_closed_before_the_output_directory_is_touched() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let options = RunOptions {
        seed: 1,
        rng_seed: 1,
        languages: vec![Language::Rust],
        policy: Policy::Require,
        ..RunOptions::new("0000000000000000", output.clone())
    };
    let error = format!("{:#}", run(&options).unwrap_err());
    assert!(error.contains("--policy require"), "{error}");
    assert!(error.contains("--policy skip"), "{error}");
    assert!(!output.exists());
}
