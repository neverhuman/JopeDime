use jope_datasets::catalog::Task;
use jope_datasets::hash::sha256_hex;
use jope_datasets::store::Store;
use jope_datasets::virtual_split::{
    generate, original, publish_from_files, read_scores, read_scores_verified, Row, Score, Source,
    SplitFiles, Track, SCORE_SCHEMA,
};

fn source(train_rows: u64, test_rows: u64) -> Source {
    Source {
        dataset_hash: "0123456789abcdef".into(),
        train_sha256: "1".repeat(64),
        test_sha256: "2".repeat(64),
        schema_sha256: "3".repeat(64),
        train_rows,
        test_rows,
    }
}

#[test]
fn grouped_repeats_partition_original_train_and_test_rows() {
    let rows: Vec<Row> = (0..20)
        .map(|i| Row {
            label: Some((i / 2 % 2).to_string()),
            labels: vec![],
            group: Some(format!("g{}", i / 2)),
            duplicate_key: format!("row{}", i / 2),
            time: None,
        })
        .collect();
    let source = source(10, 10);
    let bundle = generate(source.clone(), Task::Binary, &rows, 17).unwrap();
    assert_eq!(bundle.effective_folds, 5);
    assert_eq!(bundle.folds.len(), 25);
    bundle.validate().unwrap();
    for fold in &bundle.folds {
        let train = bundle.train_indices(fold).unwrap();
        assert_eq!(fold.test_indices.len() + train.len(), 20);
        for pair in 0..10 {
            assert_eq!(
                fold.test_indices.contains(&(2 * pair)),
                fold.test_indices.contains(&(2 * pair + 1))
            );
        }
    }
    assert!(bundle
        .folds
        .iter()
        .any(|fold| fold.test_indices.iter().any(|index| *index >= 10)));
    let mut corrupt = bundle.clone();
    corrupt.folds[0].test_indices.push(99);
    assert!(corrupt.validate().is_err());
    let mut rebound = bundle.clone();
    rebound.source.test_sha256 = "4".repeat(64);
    assert!(rebound.validate().is_err());
}

#[test]
fn sparse_class_reduces_folds_and_unworkable_group_gets_exception() {
    let rows: Vec<Row> = (0..8)
        .map(|i| Row {
            label: Some(if i < 2 { "rare" } else { "common" }.into()),
            labels: vec![],
            group: None,
            duplicate_key: format!("{i}"),
            time: None,
        })
        .collect();
    let bundle = generate(source(4, 4), Task::Binary, &rows, 5).unwrap();
    assert_eq!(bundle.effective_folds, 2);
    let mut blocked = rows;
    blocked[0].group = Some("one".into());
    blocked[1].group = Some("one".into());
    let exception = generate(source(4, 4), Task::Binary, &blocked, 5).unwrap();
    assert!(exception.exception.is_some());
    assert!(exception.folds.is_empty());
}

#[test]
fn chronological_cuts_use_time_even_when_virtual_rows_are_unsorted() {
    let times = [8, 1, 11, 3, 5, 9, 2, 12, 4, 10, 6, 7];
    let rows: Vec<Row> = times
        .iter()
        .enumerate()
        .map(|(i, time)| Row {
            label: None,
            labels: vec![],
            group: None,
            duplicate_key: format!("{i}"),
            time: Some(*time),
        })
        .collect();
    let bundle = generate(source(6, 6), Task::Timeseries, &rows, 7).unwrap();
    assert!(bundle.effective_folds >= 2);
    for fold in &bundle.folds {
        let train = bundle.train_indices(fold).unwrap();
        let latest_train = train
            .iter()
            .map(|index| times[*index as usize])
            .max()
            .unwrap();
        let earliest_test = fold
            .test_indices
            .iter()
            .map(|index| times[*index as usize])
            .min()
            .unwrap();
        assert!(latest_train < earliest_test);
    }
}

#[test]
fn original_scores_cannot_enter_exploratory_track() {
    let source = source(3, 2);
    let original = original(source.clone(), Task::Regression).unwrap();
    original.validate().unwrap();
    let score = Score {
        schema: SCORE_SCHEMA.into(),
        track: Track::Original,
        source_sha256: source.digest().unwrap(),
        split_sha256: original.digest().unwrap(),
        repeat: 0,
        fold: 0,
        algorithm: "legacy".into(),
        algorithm_version: "1".into(),
        metric: "rmse".into(),
        seed: 0,
        prediction_key: "predictions/original.bin".into(),
        prediction_bytes: 4,
        prediction_version: None,
        prediction_sha256: sha256_hex(b"pred"),
        value: 1.0,
    };
    let bytes = serde_json::to_vec(&vec![score]).unwrap();
    assert_eq!(
        read_scores(&bytes, &source, &original, Track::Original)
            .unwrap()
            .len(),
        1
    );
    assert!(read_scores(
        &bytes,
        &source,
        &original,
        Track::HistoricalTestExposedExploratory
    )
    .is_err());
    let mut altered = original.clone();
    altered.folds[0].test_indices[0] = 0;
    assert!(read_scores(&bytes, &source, &altered, Track::Original).is_err());
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(&format!("file://{}", temp.path().display())).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(store.put("predictions/original.bin", b"pred".to_vec()))
        .unwrap();
    assert_eq!(
        rt.block_on(read_scores_verified(
            &bytes,
            &source,
            &original,
            Track::Original,
            &store
        ))
        .unwrap()
        .len(),
        1
    );
    rt.block_on(store.put("predictions/original.bin", b"bad!".to_vec()))
        .unwrap();
    assert!(rt
        .block_on(read_scores_verified(
            &bytes,
            &source,
            &original,
            Track::Original,
            &store
        ))
        .is_err());
}

#[test]
fn publisher_binds_real_csv_rows_and_source_blobs() {
    let temp = tempfile::tempdir().unwrap();
    let train = temp.path().join("train.csv");
    let test = temp.path().join("test.csv");
    let schema = temp.path().join("meta.json");
    let annotations = temp.path().join("rows.jsonl");
    let all: Vec<String> = (0..10).map(|i| format!("{i},{}\n", i % 2)).collect();
    let train_bytes = all[..5].concat().into_bytes();
    let test_bytes = all[5..].concat().into_bytes();
    std::fs::write(&train, &train_bytes).unwrap();
    std::fs::write(&test, &test_bytes).unwrap();
    std::fs::write(&schema, b"{\"columns\":2}").unwrap();
    let rows: String = (0..10)
        .map(|i| format!("{{\"label\":\"{}\"}}\n", i % 2))
        .collect();
    std::fs::write(&annotations, rows).unwrap();
    let source = Source {
        dataset_hash: "0123456789abcdef".into(),
        train_sha256: sha256_hex(&train_bytes),
        test_sha256: sha256_hex(&test_bytes),
        schema_sha256: sha256_hex(b"{\"columns\":2}"),
        train_rows: 5,
        test_rows: 5,
    };
    let store_path = temp.path().join("store");
    std::fs::create_dir(&store_path).unwrap();
    let store = Store::open(&format!("file://{}", store_path.display())).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(store.put(
        &format!("blobs/sha256/{}", source.train_sha256),
        train_bytes,
    ))
    .unwrap();
    rt.block_on(store.put(&format!("blobs/sha256/{}", source.test_sha256), test_bytes))
        .unwrap();
    let out = temp.path().join("split-receipt.json");
    rt.block_on(publish_from_files(
        source.clone(),
        Task::Binary,
        SplitFiles {
            train: &train,
            test: &test,
            schema: &schema,
            annotations: &annotations,
            out: &out,
        },
        9,
        &store,
    ))
    .unwrap();
    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(out).unwrap()).unwrap();
    assert_eq!(receipt["bundles"].as_array().unwrap().len(), 2);
    for bundle in receipt["bundles"].as_array().unwrap() {
        let bytes = rt
            .block_on(store.get_bytes(bundle["key"].as_str().unwrap()))
            .unwrap()
            .unwrap();
        let loaded: jope_datasets::virtual_split::Bundle = serde_json::from_slice(&bytes).unwrap();
        loaded.validate().unwrap();
        assert_eq!(loaded.source, source);
    }
    std::fs::write(&test, b"changed\n").unwrap();
    let retry = temp.path().join("retry.json");
    assert!(rt
        .block_on(publish_from_files(
            source,
            Task::Binary,
            SplitFiles {
                train: &train,
                test: &test,
                schema: &schema,
                annotations: &annotations,
                out: &retry
            },
            9,
            &store,
        ))
        .is_err());
}
