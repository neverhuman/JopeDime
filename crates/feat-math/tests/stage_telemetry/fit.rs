use super::*;

#[test]
fn fit_traced_serializes_to_fits_bytes_on_every_corpus_program() {
    let (train, target, test) = fitted_surface_data();
    let mut fitted = 0;
    let mut expression_stages = 0;
    for (name, source) in program_set() {
        if fit_skipped(&source) {
            continue;
        }
        let program = parse_program(&source);
        match (
            fit(&program, &train, &target),
            fit_traced(&program, &train, &target),
        ) {
            (Ok(model), Ok((traced, telemetry))) => {
                assert_eq!(
                    serialized(&traced),
                    serialized(&model),
                    "{name}: model bytes"
                );
                assert_eq!(telemetry.stages.len(), stage_count(&model), "{name}");
                assert_eq!(telemetry.input_sanitized, 0, "{name}: finite training rows");
                for (rows, which) in [(&test, "held-out"), (&train, "training")] {
                    let expected = bits(&model.predict(rows).unwrap());
                    for (fitted_by, m) in [("fit", &model), ("fit_traced", &traced)] {
                        let (predictions, _) = predict_traced(m, rows).expect("finite rows");
                        assert_eq!(
                            bits(&predictions),
                            expected,
                            "{name}: {which} predictions of the {fitted_by} model"
                        );
                    }
                }
                // The training pass is Model::predict's pass over the training rows.
                let (_, replayed) = predict_traced(&model, &train).expect("finite rows");
                assert_eq!(replayed, telemetry, "{name}: training-pass telemetry");
                let training = replay_ops(&format!("{name} training pass"), &model, &train);
                if matches!(model, Model::Pipeline { .. }) {
                    assert_eq!(training, telemetry.stages, "{name}: per-op replay");
                }
                replay_ops(&format!("{name} held-out rows"), &model, &test);
                expression_stages += telemetry
                    .stages
                    .iter()
                    .filter(|stage| stage.evaluated_cells > 0)
                    .count();
                fitted += 1;
            }
            (Err(legacy), Err(traced)) => {
                assert_eq!(
                    format!("{traced:#}"),
                    format!("{legacy:#}"),
                    "{name}: error"
                );
            }
            (legacy, traced) => panic!(
                "{name}: fit {} but fit_traced {}",
                if legacy.is_ok() { "succeeds" } else { "fails" },
                if traced.is_ok() { "succeeds" } else { "fails" }
            ),
        }
    }
    eprintln!("CORPUS fitted={fitted} expression_stages={expression_stages}");
    assert_eq!(fitted, 54);
    assert!(expression_stages > 20);
}

#[test]
fn two_runs_give_identical_telemetry() {
    let (train, target, test) = fitted_surface_data();
    for (name, source) in program_set() {
        if fit_skipped(&source) {
            continue;
        }
        let program = parse_program(&source);
        let (first_model, first) = fit_traced(&program, &train, &target).expect("fits");
        let (second_model, second) = fit_traced(&program, &train, &target).expect("fits");
        assert_eq!(first, second, "{name}: training telemetry");
        assert_eq!(
            serialized(&first_model),
            serialized(&second_model),
            "{name}"
        );
        let first_pass = predict_traced(&first_model, &test).expect("finite rows");
        let second_pass = predict_traced(&second_model, &test).expect("finite rows");
        assert_eq!(bits(&first_pass.0), bits(&second_pass.0), "{name}");
        assert_eq!(first_pass.1, second_pass.1, "{name}: inference telemetry");
        // The telemetry itself round-trips through serde_json.
        let text = serde_json::to_string(&first).expect("serialize the telemetry");
        let back: PipelineTelemetry = serde_json::from_str(&text).expect("parse the telemetry");
        assert_eq!(back, first, "{name}: serde round trip");
    }
}

#[test]
fn predict_traced_equals_predict_on_the_stored_models() {
    let (_, _, test) = fitted_surface_data();
    let fixture = fixture_json("compat_freeze/model_serde.v1.json");
    let stored = fixture["stored"].as_array().expect("stored");
    assert_eq!(stored.len(), 5);
    for entry in stored {
        let name = entry["name"].as_str().expect("name");
        let model: Model =
            serde_json::from_str(entry["json"].as_str().expect("json")).expect("a stored model");
        let (predictions, telemetry) = predict_traced(&model, &test).expect("finite rows");
        assert_eq!(
            bits(&predictions),
            bits(&model.predict(&test).unwrap()),
            "{name}"
        );
        assert_eq!(telemetry.stages.len(), stage_count(&model), "{name}");
        replay_ops(name, &model, &test);
    }
}
