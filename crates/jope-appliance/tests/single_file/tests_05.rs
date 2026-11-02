use super::*;

#[test]
fn custom_fitness_cli_persists_the_objective_and_predicts_after_reload() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    train_csv(dir.path());
    let invalid = run(
        &exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--task",
            "regression",
            "--fitness",
            "unknown",
            "--output",
            "invalid.jope",
            "--json",
        ],
    );
    assert_eq!(code(&invalid), 2);
    assert!(!dir.path().join("invalid.jope").exists());
    std::fs::write(
        dir.path().join("fitness.json"),
        r#"{"metric":"pinball","quantile":0.9}"#,
    )
    .unwrap();
    let fitted = run(
        &exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--task",
            "regression",
            "--method",
            "auto",
            "--candidates",
            "ridge",
            "--folds",
            "3",
            "--fitness",
            "@fitness.json",
            "--output",
            "model.jope",
            "--json",
        ],
    );
    assert_eq!(
        code(&fitted),
        0,
        "{}",
        String::from_utf8_lossy(&fitted.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&fitted.stdout).unwrap();
    assert!(run_receipt_errors(&receipt).is_empty());
    assert_eq!(receipt["selection"]["metric"], "pinball");
    assert_eq!(receipt["selection"]["fitness"]["quantile"], 0.9);
    let model = jope_runtime::fitted::FittedModel::from_bytes(
        &std::fs::read(dir.path().join("model.jope")).unwrap(),
    )
    .unwrap();
    assert_eq!(model.header.metric, "pinball");
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../schemas/jope.appliance-fitted.v1.schema.json"
    ))
    .unwrap();
    assert!(jope_datasets::schema::validate(
        &schema,
        &serde_json::to_value(&model.header).unwrap()
    )
    .is_empty());
    let output = run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            "model.jope",
            "--input",
            "rows.csv",
            "--output",
            "predictions.csv",
            "--json",
        ],
    );
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut bad = model.clone();
    bad.header.training_procedure.selection.as_mut().unwrap()["fitness_sha256"] =
        "0".repeat(64).into();
    assert!(bad.validate().is_err());
    for remove_hash in [false, true] {
        let mut missing = model.clone();
        let selection = missing
            .header
            .training_procedure
            .selection
            .as_mut()
            .unwrap()
            .as_object_mut()
            .unwrap();
        selection.remove("fitness");
        if remove_hash {
            selection.remove("fitness_sha256");
        }
        assert!(
            missing.validate().is_err(),
            "a custom metric lost its objective"
        );
    }
}

#[test]
fn mcp_stdio_handshake_discovery_and_custom_scoring_work_in_the_packaged_binary() {
    use std::io::Write;
    use std::process::Stdio;
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    let mut child = Command::new(exe)
        .arg("mcp")
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let messages = [
        serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        serde_json::json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fitness_evaluate","arguments":{"fitness":{"metric":"mae"},"targets":[1,2,3],"predictions":[2,2,2]}}}),
    ];
    let mut input = child.stdin.take().unwrap();
    for message in messages {
        writeln!(input, "{message}").unwrap();
    }
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 3);
    assert_eq!(responses[1]["result"]["tools"].as_array().unwrap().len(), 4);
    assert!(
        (responses[2]["result"]["structuredContent"]["value"]
            .as_f64()
            .unwrap()
            - 2.0 / 3.0)
            .abs()
            < 1e-12
    );
    assert!(responses[1]["result"]["tools"][1]["inputSchema"]["$defs"]["fitness"].is_object());
}

#[test]
fn fitness_schema_accepts_examples_and_rejects_unknown_parameters() {
    let schema = jope_runtime::fitness::schema();
    for entry in jope_runtime::fitness::Fitness::catalog() {
        let value = serde_json::to_value(entry.fitness).unwrap();
        let failures = jope_datasets::schema::validate(&schema, &value);
        assert!(failures.is_empty(), "{value}: {failures:?}");
    }
    // Every checked-in recipe is valid under the published schema, not only the Rust parser.
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/objectives");
    let mut checked = 0;
    for entry in std::fs::read_dir(examples).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if path.extension().is_some_and(|e| e == "json") && !name.contains("request") {
            let value: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let failures = jope_datasets::schema::validate(&schema, &value);
            assert!(failures.is_empty(), "{name}: {failures:?}");
            checked += 1;
        }
    }
    assert!(checked >= 24, "{checked}");
    for bad in [
        serde_json::json!({"metric":"mae","ignored":true}),
        serde_json::json!({"metric":"pinball","quantile":1.5}),
        serde_json::json!({"metric":"custom","name":"x","direction":"minimize",
            "expression":{"op":"column"},"aggregation":{"kind":"mean"}}),
    ] {
        assert!(
            !jope_datasets::schema::validate(&schema, &bad).is_empty(),
            "{bad}"
        );
    }
}

#[test]
fn calibrated_objective_with_a_cost_column_round_trips_through_the_model_file() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    // Demand with a per-row unit margin; underage costs the margin, overage costs 1.
    let mut text = String::from("x,margin,demand\n");
    for i in 0..90 {
        let x = i as f64 / 10.0;
        let margin = 2.0 + (i % 4) as f64;
        let noise = ((i * 37) % 11) as f64 - 5.0;
        text.push_str(&format!("{x},{margin},{:.3}\n", 20.0 + 3.0 * x + noise));
    }
    std::fs::write(dir.path().join("train.csv"), text).unwrap();
    std::fs::write(dir.path().join("rows.csv"), "margin,x\n3,1.0\n5,4.0\n").unwrap();
    std::fs::write(
        dir.path().join("cost.json"),
        r#"{"metric":"custom","name":"stock_cost","direction":"minimize",
        "expression":{"op":"where","condition":{"op":"sub","left":{"op":"y"},"right":{"op":"pred"}},
          "then":{"op":"mul","left":{"op":"column","name":"margin"},"right":{"op":"sub","left":{"op":"y"},"right":{"op":"pred"}}},
          "else":{"op":"sub","left":{"op":"pred"},"right":{"op":"y"}}},
        "aggregation":{"kind":"mean"}}"#,
    )
    .unwrap();
    let validated = ok(run(
        &exe,
        dir.path(),
        &["fitness", "validate", "@cost.json"],
    ));
    let described: serde_json::Value = serde_json::from_slice(&validated.stdout).unwrap();
    assert_eq!(described["columns"], serde_json::json!(["margin"]));
    let fit = |calibration: &str, model: &str| {
        let output = ok(run(
            &exe,
            dir.path(),
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "demand",
                "--task",
                "regression",
                "--candidates",
                "ridge",
                "--folds",
                "3",
                "--fitness",
                "@cost.json",
                "--calibration",
                calibration,
                "--output",
                model,
                "--json",
            ],
        ));
        let receipt: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(run_receipt_errors(&receipt).is_empty());
        assert_eq!(receipt["task"]["features"], 1, "margin is not a feature");
        let predicted = ok(run(
            &exe,
            dir.path(),
            &[
                "predict",
                "--model",
                model,
                "--input",
                "rows.csv",
                "--output",
                &format!("{model}.csv"),
                "--json",
            ],
        ));
        let predict: serde_json::Value = serde_json::from_slice(&predicted.stdout).unwrap();
        assert_eq!(predict["ignored_columns"], serde_json::json!(["margin"]));
        let values: Vec<f64> = std::fs::read_to_string(dir.path().join(format!("{model}.csv")))
            .unwrap()
            .lines()
            .skip(1)
            .map(|l| l.parse().unwrap())
            .collect();
        (receipt, values)
    };
    let (raw, raw_values) = fit("none", "raw.jope");
    let (tuned, tuned_values) = fit("affine", "tuned.jope");
    assert!(raw["selection"].get("calibration").is_none());
    let candidate = &tuned["selection"]["candidates"][0];
    assert!(candidate["calibrated"].as_bool().unwrap());
    assert!(candidate["calibrated_loss"].as_f64() < candidate["uncalibrated_loss"].as_f64());
    // Shortages cost 2-5x overages, so the calibrated order quantity is higher.
    for (raw, tuned) in raw_values.iter().zip(&tuned_values) {
        assert!(tuned > raw, "{tuned} <= {raw}");
    }
    let model = jope_runtime::fitted::FittedModel::from_bytes(
        &std::fs::read(dir.path().join("tuned.jope")).unwrap(),
    )
    .unwrap();
    assert_eq!(model.header.auxiliary_columns, ["margin"]);
    let transform = model.header.output_transform.unwrap();
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../schemas/jope.appliance-fitted.v1.schema.json"
    ))
    .unwrap();
    assert!(jope_datasets::schema::validate(
        &schema,
        &serde_json::to_value(&model.header).unwrap()
    )
    .is_empty());
    // A model whose stored map disagrees with its selection record does not load.
    let mut tampered = model.clone();
    tampered.header.output_transform = Some(jope_runtime::calibration::OutputTransform {
        intercept: transform.intercept + 1.0,
        ..transform
    });
    assert!(tampered.validate().is_err());
    let mut dropped = model.clone();
    dropped.header.output_transform = None;
    assert!(dropped.validate().is_err());
    let mut renamed = model;
    renamed.header.auxiliary_columns = vec!["price".into()];
    assert!(renamed.validate().is_err());
    // Calibration without an objective is an input error.
    let refused = run(
        &exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "demand",
            "--calibration",
            "affine",
            "--output",
            "never.jope",
        ],
    );
    assert_eq!(code(&refused), 2);
}
