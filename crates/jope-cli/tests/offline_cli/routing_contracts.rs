use super::*;

#[test]
fn regression_commands_refuse_other_tasks_and_unsupported_features() {
    let temporary = tempfile::tempdir().unwrap();
    let router = test_router(temporary.path());
    // A valid regression fixture whose second feature is declared nullable (values unchanged).
    let nullable = temporary.path().join("nullable.json");
    let mut data: Value = serde_json::from_slice(&fs::read(regression_linear()).unwrap()).unwrap();
    data["columns"][1]["nullable"] = Value::Bool(true);
    fs::write(&nullable, serde_json::to_vec(&data).unwrap()).unwrap();
    let mut cases = vec![(nullable, "integer, non-null")];
    for (fixture, reason) in [
        ("binary/linear.json", "unsupported task binary"),
        (
            "multiclass/three_classes.json",
            "unsupported task multiclass",
        ),
        ("time_series/trend.json", "unsupported task time_series"),
        ("regression/categorical.json", "integer, non-null"),
        ("regression/missing_values.json", "integer, non-null"),
    ] {
        cases.push((synthetic_root().join(fixture), reason));
    }
    for (input, reason) in &cases {
        for command in ["fit", "search", "route"] {
            assert_command_rejects(command, input, &router, reason);
        }
    }
}

#[test]
fn dataset_commands_validate_features_and_never_fall_back_to_test() {
    let temporary = tempfile::tempdir().unwrap();
    let router = test_router(temporary.path());
    let input = temporary.path().join("invalid.json");
    let original: Value = serde_json::from_slice(&fs::read(regression_linear()).unwrap()).unwrap();
    for (alteration, reason) in [
        (0, "feature width differs from schema"),
        (1, "expected integer feature"),
        (2, "empty or duplicate column name"),
        (3, "missing train/validation/test partition"),
    ] {
        let mut data = original.clone();
        match alteration {
            0 => data["rows"][0]["features"] = serde_json::json!([]),
            1 => data["rows"][0]["features"][0] = serde_json::json!(1.5),
            2 => data["columns"][1]["name"] = data["columns"][0]["name"].clone(),
            _ => {
                // Without validation rows there is no hold-out; test rows never stand in.
                for row in data["rows"].as_array_mut().unwrap() {
                    if row["split"] == "validation" {
                        row["split"] = serde_json::json!("test");
                    }
                }
            }
        }
        fs::write(&input, serde_json::to_vec(&data).unwrap()).unwrap();
        for command in ["encode", "fit", "search", "route"] {
            assert_command_rejects(command, &input, &router, reason);
        }
    }
}

#[test]
fn search_selection_and_validation_score_do_not_use_test_rows() {
    let temporary = tempfile::tempdir().unwrap();
    let input = temporary.path().join("changed-test.json");
    let original = jope_cli::search_dataset(&regression_linear()).unwrap();
    let fitted = jope_cli::fit_dataset(&regression_linear(), &original.genome).unwrap();
    let mut data: Value = serde_json::from_slice(&fs::read(regression_linear()).unwrap()).unwrap();
    for row in data["rows"].as_array_mut().unwrap() {
        if row["split"] == "test" {
            row["target"] = serde_json::json!(-999_999_999);
            for feature in row["features"].as_array_mut().unwrap() {
                *feature = serde_json::json!(999_999_999);
            }
        }
    }
    fs::write(&input, serde_json::to_vec(&data).unwrap()).unwrap();
    let changed = jope_cli::search_dataset(&input).unwrap();
    assert_eq!(changed.genome, original.genome);
    assert_eq!(changed.hold_r2, original.hold_r2);
    let refitted = jope_cli::fit_dataset(&input, &original.genome).unwrap();
    assert_eq!(refitted.hold_rows, 16);
    assert_eq!(refitted.hold_r2, fitted.hold_r2);
    assert_eq!(refitted.train_r2, fitted.train_r2);
}

#[test]
fn route_without_router_fails_closed() {
    let input = regression_linear();
    let output = jope(
        Path::new("."),
        &[
            "route",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
            "--router",
            "missing-router.json",
        ],
    );
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert_eq!(report["schema"], "jope.route.v1");
}

#[test]
fn train_step_writes_router_and_route_fits_a_ridge_genome() {
    let temporary = tempfile::tempdir().unwrap();
    let root = synthetic_root();
    let trained = jope(
        Path::new("."),
        &[
            "train-step",
            "--local",
            "--json",
            "--root",
            root.to_str().expect("utf8 path"),
            "--output",
            temporary.path().to_str().expect("utf8 path"),
        ],
    );
    assert!(
        trained.status.success(),
        "{}",
        String::from_utf8_lossy(&trained.stderr)
    );
    assert!(temporary.path().join("router.v1.json").is_file());
    let input = regression_linear();
    let routed = jope(
        Path::new("."),
        &[
            "route",
            "--json",
            "--input",
            input.to_str().expect("utf8 path"),
            "--router",
            temporary.path().to_str().expect("utf8 path"),
        ],
    );
    assert!(
        routed.status.success(),
        "{}",
        String::from_utf8_lossy(&routed.stderr)
    );
    let report: Value = serde_json::from_slice(&routed.stdout).unwrap();
    assert_eq!(report["ok"], true);
    assert_eq!(report["schema"], "jope.route.v1");
    assert!(report["genome"].as_str().unwrap().contains("ridge"));
    assert!(report["hold_r2"].as_f64().unwrap().is_finite());
}
