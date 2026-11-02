use super::*;

#[test]
fn platt_calibrated_objects_predict_whole_through_the_cli_and_model_tool() {
    use jope_runtime::model_object::{calibration_platt, ModelObject};
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.logistic"]);
    let mut train = String::from("x,y\n");
    for i in 0..40 {
        let label = if i % 10 > 4 { "yes" } else { "no" };
        train.push_str(&format!("{},{label}\n", i % 10));
    }
    std::fs::write(dir.path().join("train.csv"), train).unwrap();
    let rows = "x\n0\n3\n6\n9\n";
    std::fs::write(dir.path().join("rows.csv"), rows).unwrap();
    std::fs::write(dir.path().join("labelled.csv"), "x,y\n0,no\n9,yes\n").unwrap();
    let predict = |model: &str, input: &str, out: &str| {
        run(
            &exe,
            dir.path(),
            &[
                "predict",
                "--model",
                model,
                "--input",
                input,
                "--output",
                out,
                "--threads",
                "1",
                "--receipt",
                &format!("{out}.receipt.json"),
                "--json",
            ],
        )
    };
    let receipt = |out: &str| -> serde_json::Value {
        serde_json::from_slice(
            &std::fs::read(dir.path().join(format!("{out}.receipt.json"))).unwrap(),
        )
        .unwrap()
    };
    let fit: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--task",
                "binary",
                "--method",
                "logistic",
                "--threads",
                "1",
                "--json",
            ],
        ))
        .stdout,
    )
    .unwrap();
    let legacy_path = fit["outputs"][0]["path"].as_str().unwrap().to_owned();
    ok(predict(&legacy_path, "rows.csv", "legacy.csv"));
    let legacy = std::fs::read(dir.path().join("legacy.csv")).unwrap();
    ok(run(
        &exe,
        dir.path(),
        &[
            "model",
            "export",
            "--model",
            &legacy_path,
            "--format",
            "fitted",
            "--output",
            "legacy.jope",
        ],
    ));
    ok(predict("legacy.jope", "rows.csv", "standalone.csv"));
    let object = ModelObject::read(&dir.path().join(&legacy_path), 1 << 24).unwrap();
    let fitted = object.fitted().unwrap();
    let query = jope_runtime::table::parse_csv(rows.as_bytes(), &Default::default()).unwrap();
    let (table, _) =
        jope_runtime::table::prediction(&query, fitted.schema(), &fitted.header.target).unwrap();
    let raw = jope_runtime::api::predictions_csv(&fitted.predict(&table).unwrap()).unwrap();
    assert_eq!(legacy, raw, "legacy object prediction changed");
    assert_eq!(
        std::fs::read(dir.path().join("standalone.csv")).unwrap(),
        raw,
        "standalone fitted-file prediction changed"
    );

    let order = &fitted.header.class_order;
    let (mut scores, mut labels, mut folds) = (Vec::new(), Vec::new(), Vec::new());
    for fold in 0..3 {
        for i in 0..20 {
            scores.push(if i < 10 { 0.001 } else { 0.999 });
            let positive = i % 10 < if i < 10 { 2 } else { 8 };
            labels.push(order[usize::from(positive)].as_str());
            folds.push(fold);
        }
    }
    let calibration = calibration_platt::cross_fit(
        &scores,
        &labels,
        [order[0].as_str(), order[1].as_str()],
        &folds,
    )
    .unwrap()
    .calibration;
    for (format, base, selected) in [
        (
            "appliance",
            ModelObject::from_fitted(&fitted).unwrap(),
            fitted.header.method.id.clone(),
        ),
        (
            "bound",
            ModelObject::from_bound(&fitted.model).unwrap(),
            "org.jopedime/battle-artifact/v2".to_owned(),
        ),
    ] {
        let calibrated = base.with_platt_calibration(calibration.clone()).unwrap();
        let saved = calibrated
            .save(&dir.path().join("models"), 1 << 24)
            .unwrap();
        let path = saved
            .strip_prefix(dir.path())
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let expected =
            jope_runtime::api::predictions_csv(&calibrated.predict(&table).unwrap()).unwrap();
        assert_ne!(
            expected, raw,
            "{format}: calibration must change the probabilities"
        );
        let out = format!("{format}.csv");
        ok(predict(&path, "rows.csv", &out));
        assert_eq!(
            std::fs::read(dir.path().join(&out)).unwrap(),
            expected,
            "{format}: CLI differs from ModelObject::predict"
        );
        let header = format!("prediction,p_{},p_{}\n", order[0], order[1]);
        assert!(String::from_utf8(expected.clone())
            .unwrap()
            .starts_with(&header));
        let run_receipt = receipt(&out);
        assert!(run_receipt_errors(&run_receipt).is_empty(), "{run_receipt}");
        assert_eq!(run_receipt["method"]["selected"], selected.as_str());
        assert_eq!(run_receipt["task"].is_object(), format == "appliance");
        let response = model_tool(
            &exe,
            dir.path(),
            "model_predict",
            serde_json::json!({"id": calibrated.id(), "csv": rows}),
        );
        assert_eq!(response["result"]["isError"], false, "{response}");
        assert_eq!(
            response["result"]["structuredContent"]["predictions_csv"],
            String::from_utf8(expected).unwrap()
        );
        // The appliance head records its target, so a labelled input predicts and reports it
        // ignored; a bound head records none and refuses the extra column.
        let labelled = format!("{format}-labelled.csv");
        let output = predict(&path, "labelled.csv", &labelled);
        if format == "appliance" {
            ok(output);
            assert_eq!(
                receipt(&labelled)["ignored_columns"],
                serde_json::json!(["y"])
            );
        } else {
            assert_eq!(
                code(&output),
                2,
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
            assert!(!dir.path().join(&labelled).exists());
        }
        let refused = run(
            &exe,
            dir.path(),
            &[
                "model",
                "export",
                "--model",
                &path,
                "--format",
                "fitted",
                "--output",
                &format!("{format}.jope"),
            ],
        );
        assert_ne!(
            code(&refused),
            0,
            "{format}: fitted export must not drop the calibration"
        );
        assert_eq!(code(&refused), 4);
        let message = String::from_utf8_lossy(&refused.stderr);
        assert!(message.contains("standalone fitted export would drop Platt calibration"));
        assert!(
            !message.contains("BACKEND_UNAVAILABLE"),
            "a calibration-preservation refusal is not unavailable device selection"
        );
        assert!(!dir.path().join(format!("{format}.jope")).exists());
    }

    // A calibration part whose capability is not required, and a required capability whose part
    // is missing, are both refused before any output is written. `save` refuses the first, so
    // both are written as a foreign producer could: manifest and parts under their identity.
    let calibrated = ModelObject::from_fitted(&fitted)
        .unwrap()
        .with_platt_calibration(calibration.clone())
        .unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(calibrated.manifest_bytes()).unwrap();
    let mut stripped = manifest.clone();
    stripped["required_capabilities"] = serde_json::json!([]);
    let mut missing = manifest;
    missing["files"]
        .as_array_mut()
        .unwrap()
        .retain(|f| f["path"] != calibration_platt::PART);
    for (name, manifest, platt_part) in [
        ("capability-stripped", stripped, true),
        ("part-missing", missing, false),
    ] {
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let id = format!("sha256-{}", jope_runtime::sha256_hex(&bytes));
        let object_dir = dir.path().join("models").join(&id);
        std::fs::create_dir(&object_dir).unwrap();
        std::fs::write(object_dir.join("manifest.json"), &bytes).unwrap();
        for (part, data) in calibrated.parts() {
            if platt_part || part != calibration_platt::PART {
                std::fs::write(object_dir.join(part), data).unwrap();
            }
        }
        let path = format!("models/{id}");
        let out = format!("{name}.csv");
        let output = predict(&path, "rows.csv", &out);
        assert_eq!(
            code(&output),
            3,
            "{name}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("Platt"),
            "{name}"
        );
        assert!(!dir.path().join(&out).exists(), "{name}");
        let response = model_tool(
            &exe,
            dir.path(),
            "model_predict",
            serde_json::json!({"id": id, "csv": rows}),
        );
        assert_eq!(response["result"]["isError"], true, "{name}: {response}");
    }
}
