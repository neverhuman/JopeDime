use jope_narrator::{read_journal, FactJournal, Limits as CaptureLimits, LineageFact};
use jope_runtime::{
    api::{fit, predict_with_narration, FitOptions},
    budget::Budget,
    fitted::{CreatedBy, FittedModel},
    limits::Limits,
    table::{parse_csv, prediction, training},
    task::{TaskChoice, TaskKind},
};

fn fixture() -> (FittedModel, feat_math::advanced_v3::RawTable) {
    let limits = Limits::default();
    let mut csv = String::from("private_feature,y\n");
    for i in 0..24 {
        csv.push_str(&format!("{i},{}\n", 2 * i + 1));
    }
    let table = parse_csv(csv.as_bytes(), &limits).unwrap();
    let prepared = training(
        &table,
        "y",
        TaskChoice::Fixed(TaskKind::Regression),
        &limits,
    )
    .unwrap();
    let fitted = fit(
        &prepared,
        &FitOptions {
            method: "ridge".into(),
            ..FitOptions::default()
        },
        &Budget::new(None, 1),
        &"0".repeat(64),
        CreatedBy::default(),
    )
    .unwrap()
    .fitted;
    let table = parse_csv(b"private_feature\n7\n11\n", &limits).unwrap();
    let (input, _) = prediction(&table, &prepared.schema, &prepared.target).unwrap();
    (fitted, input)
}
fn directory() -> tempfile::TempDir {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/narration-runtime-tests");
    std::fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}

#[test]
fn fixed_prediction_narration_preserves_bytes_and_binds_actual_inputs() {
    let (fitted, input) = fixture();
    let plain = predict_with_narration(&fitted, &input, &Budget::new(None, 1), None).unwrap();
    let dir = directory();
    let path = dir.path().join("events.jsonl");
    let (tap, journal) =
        FactJournal::create(&path, "prediction-fixture", CaptureLimits::default()).unwrap();
    let observed =
        predict_with_narration(&fitted, &input, &Budget::new(None, 1), Some(&tap)).unwrap();
    journal.finish().unwrap();
    assert_eq!(
        serde_json::to_vec(&plain).unwrap(),
        serde_json::to_vec(&observed).unwrap()
    );
    let page = read_journal(&path, None, 10, 1 << 20).unwrap();
    assert_eq!(page.events.len(), 2);
    let mut bindings = Vec::new();
    for event in &page.events {
        match &event.fact {
            LineageFact::PredictionBound {
                status,
                model_sha256,
                input_sha256,
                rows,
                seconds,
            } => {
                assert_eq!(*rows, 2);
                assert_eq!(model_sha256.len(), 64);
                assert_eq!(input_sha256.len(), 64);
                bindings.push((model_sha256, input_sha256));
                assert_eq!(seconds.is_none(), status == "started");
            }
            _ => panic!("fixed prediction must not invent search events"),
        }
    }
    assert_eq!(bindings[0], bindings[1]);
    let bytes = std::fs::read_to_string(path).unwrap();
    assert!(!bytes.contains("private_feature"));
}

#[test]
fn prediction_failures_are_retained_without_claiming_completion() {
    let (fitted, mut input) = fixture();
    input.headers[0] = "wrong_feature".into();
    let dir = directory();
    let path = dir.path().join("events.jsonl");
    let (tap, journal) =
        FactJournal::create(&path, "prediction-failure", CaptureLimits::default()).unwrap();
    assert!(predict_with_narration(&fitted, &input, &Budget::new(None, 1), Some(&tap)).is_err());
    journal.finish().unwrap();
    let page = read_journal(&path, None, 10, 1 << 20).unwrap();
    assert_eq!(page.events.len(), 2);
    assert!(
        matches!(&page.events[1].fact, LineageFact::PredictionBound { status, .. } if status == "failed")
    );
    assert!(!std::fs::read_to_string(path)
        .unwrap()
        .contains("wrong_feature"));
}

#[test]
fn detached_observer_does_not_change_prediction() {
    let (fitted, input) = fixture();
    let plain = predict_with_narration(&fitted, &input, &Budget::new(None, 1), None).unwrap();
    let (tap, receiver) = jope_narrator::fact_channel(1, 1024);
    drop(receiver);
    let observed =
        predict_with_narration(&fitted, &input, &Budget::new(None, 1), Some(&tap)).unwrap();
    assert_eq!(
        serde_json::to_vec(&plain).unwrap(),
        serde_json::to_vec(&observed).unwrap()
    );
}

fn observed_binding(
    fitted: &FittedModel,
    input: &feat_math::advanced_v3::RawTable,
) -> (String, String) {
    let recording = jope_narrator::RecordingTap::default();
    let plain = predict_with_narration(fitted, input, &Budget::new(None, 1), None).unwrap();
    let observed =
        predict_with_narration(fitted, input, &Budget::new(None, 1), Some(&recording.tap()))
            .unwrap();
    assert_eq!(
        serde_json::to_vec(&plain).unwrap(),
        serde_json::to_vec(&observed).unwrap()
    );
    match &recording.facts()[0] {
        LineageFact::PredictionBound {
            model_sha256,
            input_sha256,
            ..
        } => (model_sha256.clone(), input_sha256.clone()),
        _ => panic!("missing binding"),
    }
}

#[test]
fn bindings_change_with_real_inputs_and_fitted_state_independently() {
    let (fitted, input) = fixture();
    let base = observed_binding(&fitted, &input);
    let mut changed_input = input.clone();
    changed_input.rows[0][0] = "8".into();
    let input_ids = observed_binding(&fitted, &changed_input);
    assert_eq!(base.0, input_ids.0);
    assert_ne!(base.1, input_ids.1);
    let mut changed_model = fitted.clone();
    // A fixed threshold is bound state, even when this regression output ignores it.
    changed_model.model.threshold += 0.125;
    let model_ids = observed_binding(&changed_model, &input);
    assert_ne!(base.0, model_ids.0);
    assert_eq!(base.1, model_ids.1);
}

fn math_fixture() -> (
    jope_runtime::math_pipeline::MathPipeline,
    jope_runtime::table::CsvTable,
) {
    use jope_runtime::math_pipeline::{MathPipeline, INPUT_PART, PIPELINE_PART};
    let parts = std::collections::BTreeMap::from([
        (
            INPUT_PART.into(),
            include_bytes!("fixtures/math-pipeline/input.json").to_vec(),
        ),
        (
            PIPELINE_PART.into(),
            include_bytes!("fixtures/math-pipeline/pipeline.json").to_vec(),
        ),
    ]);
    let model = MathPipeline::decode(&parts, PIPELINE_PART).unwrap();
    let table = parse_csv(
        include_bytes!("fixtures/math-pipeline/rows.csv"),
        &Limits::default(),
    )
    .unwrap();
    (model, table)
}
fn math_binding(
    model: &jope_runtime::math_pipeline::MathPipeline,
    table: &jope_runtime::table::CsvTable,
) -> (String, String) {
    let plain = model.predict_table_with_narration(table, None).unwrap();
    let recording = jope_narrator::RecordingTap::default();
    let observed = model
        .predict_table_with_narration(table, Some(&recording.tap()))
        .unwrap();
    assert_eq!(
        plain.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        observed.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(plain.1, observed.1);
    let facts = recording.facts();
    assert_eq!(facts.len(), 2);
    assert!(matches!(&facts[1],LineageFact::PredictionBound{status,..} if status=="completed"));
    match &facts[0] {
        LineageFact::PredictionBound {
            model_sha256,
            input_sha256,
            ..
        } => (model_sha256.clone(), input_sha256.clone()),
        _ => panic!("missing GP binding"),
    }
}
#[test]
fn shipped_gp_path_preserves_prediction_bits_and_binds_model_and_cells() {
    let (model, table) = math_fixture();
    let base = math_binding(&model, &table);
    let mut changed = table.clone();
    changed.rows[0][0] = Some("200".into());
    // Cached CSV digest must not substitute for the actual cells.
    assert_eq!(changed.sha256, table.sha256);
    let input_ids = math_binding(&model, &changed);
    assert_eq!(base.0, input_ids.0);
    assert_ne!(base.1, input_ids.1);
    let mut changed = model.clone();
    changed.input.target.min += 1.0;
    let model_ids = math_binding(&changed, &table);
    assert_ne!(base.0, model_ids.0);
    assert_eq!(base.1, model_ids.1);
    let mut fitted = model.clone();
    if let feat_math::model::Model::Pipeline { head, .. } = &mut fitted.state.model {
        if let feat_math::heads::FittedHead::Ridge(r) = head.as_mut() {
            r.w[0] += 0.125;
        } else {
            panic!("fixture head changed")
        }
    } else {
        panic!("fixture model changed")
    }
    let fitted_ids = math_binding(&fitted, &table);
    assert_ne!(base.0, fitted_ids.0);
    assert_eq!(base.1, fitted_ids.1);
}

#[test]
fn environment_capture_observes_shipped_gp_and_startup_failure_is_best_effort() {
    use std::process::Command;
    const CHILD: &str = "JOPE_NARRATION_PREDICTION_CHILD";
    if let Ok(mode) = std::env::var(CHILD) {
        let (model, table) = math_fixture();
        let plain = model.predict_table_with_narration(&table, None).unwrap();
        let observed = model.predict_table(&table).unwrap();
        assert_eq!(plain.0, observed.0);
        if mode == "normal" {
            let root = std::path::PathBuf::from(std::env::var_os(jope_narrator::DIR_ENV).unwrap());
            let journal = std::fs::read_dir(root)
                .unwrap()
                .map(|e| e.unwrap().path())
                .find(|p| p.extension().is_some_and(|e| e == "jsonl"))
                .unwrap();
            let page = read_journal(&journal, None, 10, 1 << 20).unwrap();
            assert_eq!(page.events.len(), 3);
            assert_eq!(
                jope_narrator::read_commit(&journal).unwrap().status,
                "complete"
            );
            assert!(!std::fs::read_to_string(journal).unwrap().contains("cement"));
        }
        return;
    }
    let dir = directory();
    let bad = dir.path().join("not-a-directory");
    std::fs::write(&bad, b"occupied").unwrap();
    for (mode, path) in [
        ("normal", dir.path().join("capture")),
        ("startup-failure", bad),
    ] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "environment_capture_observes_shipped_gp_and_startup_failure_is_best_effort",
                "--nocapture",
            ])
            .env(CHILD, mode)
            .env(jope_narrator::DIR_ENV, path)
            .env_remove("JOPE_LINEAGE_FACT_DIR")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if mode == "startup-failure" {
            assert!(String::from_utf8_lossy(&output.stderr).contains("narration disabled"));
        }
    }
}
