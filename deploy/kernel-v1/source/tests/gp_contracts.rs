use jope_parallel::{gp, manifest::*, prepare::*};
use std::fs;
#[test]
fn sealed_memory_view_matches_file_evaluation_and_rejects_writes() {
    use std::io::Write;
    use std::os::fd::AsRawFd;
    let tmp = tempfile::tempdir().unwrap();
    let rows = (0..30)
        .map(|i| vec![i as f64, (i * i) as f64, 3.0 + i as f64 * 0.8])
        .collect::<Vec<_>>();
    let mut memory = gp::memory_view(&rows[..24], &rows[24..]).unwrap();
    assert!(memory.write_all(b"corrupt").is_err());
    let train = tmp.path().join("train.csv");
    let heldout = tmp.path().join("heldout.csv");
    let genome = tmp.path().join("p.dsl");
    fs::write(&train, csv_bytes(&rows[..24]).unwrap()).unwrap();
    fs::write(&heldout, csv_bytes(&rows[24..]).unwrap()).unwrap();
    fs::write(&genome, "(pipe X (ridge))").unwrap();
    let a = tmp.path().join("a.json");
    let b = tmp.path().join("b.json");
    gp::evaluate(&train, &heldout, &genome, &a).unwrap();
    gp::evaluate_memory(memory.as_raw_fd(), &genome, &b).unwrap();
    let a: serde_json::Value = read_json(&a).unwrap();
    let b: serde_json::Value = read_json(&b).unwrap();
    assert_eq!(a["semantic_sha256"], b["semantic_sha256"]);
    assert_eq!(a["rmse"], b["rmse"]);
}
#[test]
fn memory_evaluation_rejects_unsealed_regular_files() {
    use std::os::fd::AsRawFd;
    let tmp = tempfile::tempdir().unwrap();
    let file = fs::File::create(tmp.path().join("not-memory")).unwrap();
    let error = gp::evaluate_memory(
        file.as_raw_fd(),
        &tmp.path().join("missing-program"),
        &tmp.path().join("out"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("immutable"));
}
#[test]
fn fitted_predictor_roundtrip_matches_native_gp_and_restores_target_units() {
    let tmp = tempfile::tempdir().unwrap();
    let train = tmp.path().join("train.csv");
    let rows = (0..40)
        .map(|i| vec![i as f64, (i * i) as f64, 100.0 + i as f64 * 2.0])
        .collect::<Vec<_>>();
    fs::write(&train, csv_bytes(&rows).unwrap()).unwrap();
    let model = tmp.path().join("model.json");
    gp::fit(&train, "(pipe X (ridge))", &model).unwrap();
    let features = tmp.path().join("features.csv");
    let x = vec![vec![2.0, 4.0], vec![10.0, 100.0]];
    fs::write(&features, csv_bytes(&x).unwrap()).unwrap();
    let out = tmp.path().join("predictions.csv");
    gp::predict(&model, &features, &out).unwrap();
    let predicted = fs::read_to_string(out)
        .unwrap()
        .lines()
        .map(|s| s.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert!(predicted.iter().all(|p| *p > 90.0 && *p < 190.0));
    let p: gp::Predictor = read_json(&model).unwrap();
    let decoded: feat_math::Model =
        serde_json::from_slice(&serde_json::to_vec(&p.model).unwrap()).unwrap();
    let tx = x
        .iter()
        .map(|r| {
            p.preprocessing
                .features(&r.iter().map(f64::to_string).collect::<Vec<_>>())
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(p.model.predict(&tx), decoded.predict(&tx));
    let expected = p
        .model
        .predict(&tx)
        .iter()
        .map(|v| v * p.preprocessing.target_scale + p.preprocessing.target_mean)
        .collect::<Vec<_>>();
    assert_eq!(predicted, expected);
}
#[test]
fn score_rejects_unfrozen_receipt_before_opening_test() {
    let tmp = tempfile::tempdir().unwrap();
    let receipt = tmp.path().join("selection.json");
    write_json(
        &receipt,
        &gp::SelectionReceipt {
            schema: "parallel.selection.v1".into(),
            predictor_sha256: "0".repeat(64),
            final_test_sha256: "0".repeat(64),
            selection_evidence_path: tmp.path().join("missing-evidence"),
            selection_evidence_sha256: "0".repeat(64),
            selected_on: "meta_validation".into(),
            frozen: false,
        },
    )
    .unwrap();
    let error = gp::score(
        &tmp.path().join("missing-model"),
        &tmp.path().join("missing-test"),
        &receipt,
        &tmp.path().join("out"),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("frozen"));
}
#[test]
fn constant_target_r2_is_explicitly_undefined() {
    let result = gp::metrics(&[2., 2.], &[2., 3.]).unwrap();
    assert!(result["r2"].is_null());
    assert_eq!(result["r2_status"], "undefined_constant_target");
}
#[test]
fn reconstruction_recovers_missing_values_without_using_validation_targets() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("raw.csv");
    fs::write(&source, "x,target\n0,10\n,20\n10,30\n").unwrap();
    let original = Table {
        width: 2,
        header: false,
        rows: vec![vec!["0.5".into(), "0.5".into()]],
    };
    let m = serde_json::json!({"source_path":source,"target_generation":"original_numeric_column","feature_columns":["x"],"target_column":"target","feature_encodings":{"x":{"kind":"numeric","impute_value":5.0}},"feature_scaling":{"min":[0.0],"max":[10.0]},"target_scaling":{"min":10.0,"max":30.0}});
    let (raw, receipt) = reconstruct_numeric(&original, &m).unwrap();
    assert_eq!(raw.rows[0], vec!["", "20"]);
    assert_eq!(receipt["source_row_indices"], serde_json::json!([1]));
}

#[test]
fn reconstruction_uses_recorded_categories_only_to_recover_raw_training_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("raw.csv");
    fs::write(&source, "colour,target\nred,10\n,20\nblue,30\ngreen,40\n").unwrap();
    let m = serde_json::json!({"source_path":source,"target_generation":"original_numeric_column","feature_columns":["colour"],"target_column":"target",
        "feature_encodings":{"colour":{"kind":"categorical_codes","mapping":{"__MISSING__":0,"blue":1,"red":2,"green":3},"impute_value":0.0}},
        "feature_scaling":{"min":[0.0],"max":[3.0]},"target_scaling":{"min":10.0,"max":40.0}});
    let table = Table {
        width: 2,
        header: false,
        rows: vec![
            vec![(2.0f64 / 3.0).to_string(), "0".into()],
            vec!["0".into(), (1.0f64 / 3.0).to_string()],
        ],
    };
    let (raw, receipt) = reconstruct_numeric(&table, &m).unwrap();
    assert_eq!(raw.rows, vec![vec!["red", "10"], vec!["", "20"]]);
    assert_eq!(receipt["source_row_indices"], serde_json::json!([0, 1]));
    let pp = Preprocessing::fit(&raw.rows, "regression").unwrap();
    assert_eq!(pp.columns[0].categories, vec!["red"]);
    assert_eq!(pp.features(&["blue".into()]).unwrap(), vec![-1.0]);
}

#[test]
fn reconstruction_aligns_recorded_scalers_after_feature_removal_and_rejects_changed_targets() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("raw.csv");
    fs::write(&source, "removed,x,target\n100,20,3\n200,30,4\n").unwrap();
    let mut m = serde_json::json!({"source_path":source,"target_generation":"original_numeric_column","feature_columns":["x"],"target_column":"target",
        "original_headers":["removed","x","target"],"feature_encodings":{"x":{"kind":"numeric","impute_value":20.0}},
        "feature_scaling":{"min":[100.0,10.0],"max":[200.0,30.0]},"target_scaling":{"min":0.0,"max":4.0}});
    let table = Table {
        width: 2,
        header: false,
        rows: vec![vec!["0.5".into(), "0.75".into()]],
    };
    let (raw, receipt) = reconstruct_numeric(&table, &m).unwrap();
    assert_eq!(raw.rows[0], vec!["20", "3"]);
    assert_eq!(
        receipt["historical_scaler_feature_indices"],
        serde_json::json!([1])
    );
    m["target_fix"] = serde_json::json!({"changed":true});
    assert!(reconstruct_numeric(&table, &m)
        .unwrap_err()
        .to_string()
        .contains("changed legacy target"));
}
