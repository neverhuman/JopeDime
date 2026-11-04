use super::*;

#[test]
fn numeric_csv_matches_native_float_display() {
    assert_eq!(
        numeric_csv(&[vec![1.0, -0.0, 1.25], vec![-2.0, 0.0, 3.0]]),
        b"1,-0,1.25\n-2,0,3\n"
    );
}
#[test]
fn indices_refuse_negative_and_fractional() {
    assert!(indices(&serde_json::json!([-1])).is_err());
    assert!(indices(&serde_json::json!([0.5])).is_err());
}
#[test]
fn parent_selects_only_original_fitting_rows_after_full_hash_verification() {
    let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-fixtures");
    std::fs::create_dir_all(&fixture_root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("cpu-context-input-")
        .tempdir_in(&fixture_root)
        .unwrap();
    let root = scratch.path().canonicalize().unwrap();
    let csv = b"number,category,target\n1,A,10\n999,HOLD,DO_NOT_PARSE_TARGET\n3,B,30\n";
    let source = root.join("training-source.csv");
    std::fs::write(&source, csv).unwrap();
    let unused = Artifact {
        path: source.clone(),
        sha256: sha256_hex(csv),
    };
    let sources = ReferenceSources {
        index: unused.clone(),
        registry: unused.clone(),
        assembly: unused,
        metadata_root: root.clone(),
        component_root: root.clone(),
        context_root: root.clone(),
        source_root: root,
    };
    let meta = serde_json::json!({"storage":{"source":"training-source.csv","source_sha256":sha256_hex(csv),"feature_columns":["number","category"],"target_column":"target","original_training_row_map":[2,0]}});
    let pp = NativePreprocessing {
        schema: "parallel.preprocessing.v1".into(),
        columns: vec![
            NativeColumn {
                kind: "numeric".into(),
                impute: 0.0,
                mean: 1.0,
                scale: 2.0,
                categories: vec![],
            },
            NativeColumn {
                kind: "categorical".into(),
                impute: -1.0,
                mean: 0.0,
                scale: 1.0,
                categories: vec!["A".into(), "B".into()],
            },
        ],
        labels: vec![],
        target_mean: 10.0,
        target_scale: 10.0,
    };
    let mut custody = Vec::new();
    let actual = load_training_rows(&sources, &meta, &[1, 0], &pp, &mut custody).unwrap();
    assert_eq!(actual, vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 2.0]]);
    assert_eq!(custody.len(), 1);
    // A mutation outside selected fitting rows still invalidates the full
    // frozen source identity; it never licenses a silently changed input.
    std::fs::write(
        &source,
        b"number,category,target\n1,A,10\n0,HOLD,CHANGED\n3,B,30\n",
    )
    .unwrap();
    assert!(load_training_rows(&sources, &meta, &[1, 0], &pp, &mut custody).is_err());
}
