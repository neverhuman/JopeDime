#[test]
fn typed_statistics_use_numeric_and_unsigned_byte_order_and_exclude_nonfinite_bounds() {
    let table = Table {
        columns: [
            ("n", ColumnType::Int64),
            ("s", ColumnType::Utf8),
            ("b", ColumnType::Boolean),
            ("d", ColumnType::Double),
        ]
        .into_iter()
        .map(|(name, ty)| ColumnSpec {
            name: name.into(),
            ty,
            required: false,
        })
        .collect(),
        rows: vec![
            vec![
                Cell::I64(-10),
                Cell::Str("é".into()),
                Cell::Bool(true),
                Cell::F64(-0.0),
            ],
            vec![
                Cell::I64(2),
                Cell::Str("z".into()),
                Cell::Bool(false),
                Cell::F64(0.0),
            ],
            vec![Cell::Null, Cell::Null, Cell::Null, Cell::Null],
        ],
        metadata: vec![],
    };
    let opts = WriteOptions {
        row_group_rows: 0,
        ..opts()
    };
    let bytes = parquet::write_with_statistics(&table, &opts, "fixture").unwrap();
    let reader = FileReader::open(&bytes).unwrap();
    let n = reader.statistics(0, 0).unwrap().unwrap();
    assert_eq!(n.null_count, Some(1));
    assert_eq!(n.min, Some(StatisticValue::I64(-10)));
    assert_eq!(n.max, Some(StatisticValue::I64(2)));
    assert_eq!(
        reader.statistics(0, 1).unwrap().unwrap().min,
        Some(StatisticValue::Bytes(b"z".to_vec()))
    );
    assert_eq!(
        reader.statistics(0, 2).unwrap().unwrap().min,
        Some(StatisticValue::Bool(false))
    );
    match reader.statistics(0, 3).unwrap().unwrap().min {
        Some(StatisticValue::F64(v)) => assert_eq!(v.to_bits(), (-0.0f64).to_bits()),
        _ => panic!("missing double bound"),
    }
    let mut zero = table.clone();
    zero.rows.truncate(1);
    zero.rows[0][3] = Cell::F64(0.0);
    let bytes = parquet::write_with_statistics(&zero, &opts, "fixture").unwrap();
    let reader = FileReader::open(&bytes).unwrap();
    let stats = reader.statistics(0, 3).unwrap().unwrap();
    assert_eq!(stats.min_is_exact, Some(false));
    assert_eq!(stats.max_is_exact, Some(true));
    assert_eq!(stats.nan_count, Some(0));
    if let Some(StatisticValue::F64(v)) = stats.min {
        assert_eq!(v.to_bits(), (-0.0f64).to_bits());
    } else {
        panic!("zero lower bound");
    }
    assert!(
        matches!(reader.read_column(0,3).unwrap().0,Values::F64(v) if v[0].unwrap().to_bits()==0.0f64.to_bits())
    );
    let mut nonfinite = table;
    nonfinite.rows[0][3] = Cell::F64(f64::NAN);
    let bytes = parquet::write_with_statistics(&nonfinite, &opts, "fixture").unwrap();
    let reader = FileReader::open(&bytes).unwrap();
    let stats = reader.statistics(0, 3).unwrap().unwrap();
    assert_eq!(stats.null_count, Some(1));
    assert_eq!(stats.nan_count, Some(1));
    assert_eq!(stats.min, None);
    assert_eq!(stats.max, None);
    assert!(
        matches!(reader.read_column(0,3).unwrap().0,Values::F64(v) if v[0].is_some_and(f64::is_nan))
    );
}

#[test]
fn fixed_fixture_manifest_and_source_root_are_authenticated_as_fixture_only() {
    let bytes = include_bytes!("../fixtures/observation-import-v2/producer-manifest.json");
    let pin = include_str!("../fixtures/observation-import-v2/producer-manifest.sha256").trim();
    assert_eq!(sha256_hex(bytes), pin);
    assert_eq!(
        spec().input_roots[0].sha256,
        sha256_hex(include_bytes!(
            "../fixtures/observation-import-v2/producer-source.txt"
        ))
    );
    let archive = bound(INDEX, bytes, pin);
    assert!(matches!(
        archive.producer_closure,
        ProducerClosure::Bound { .. }
    ));
    assert!(matches!(archive.historical, HistoricalStatus::Held { .. }));
    let historical: Value = serde_json::from_str(include_str!(
        "../fixtures/observation-import-v2/historical-status.json"
    ))
    .unwrap();
    assert_eq!(historical["historical_input"], "UNKNOWN");
    assert_eq!(historical["current_estate"], "UNKNOWN");
}

#[test]
fn malformed_or_ambiguous_index_is_refused_without_a_completeness_claim() {
    for bytes in [
        br#"{"schema":"jopedime.gp-results-index/v1","runs":[{"outcome":"failed"}],"runs":[]}"#
            .as_slice(),
        br#"{"schema":"jopedime.gp-results-index/v1","runs":[{"score":1.0,"score":2.0}]}"#,
        br#"{"schema":"jopedime.gp-results-index/v1","runs":[{"score":1e999}]}"#,
        br#"{"schema":"jopedime.gp-results-index/v1"}"#,
        br#"{"schema":"other","runs":[]}"#,
    ] {
        assert!(distilled::import_gp_results_index_complete(
            bytes,
            None,
            &known(),
            &BTreeMap::new(),
            None
        )
        .is_err());
        assert!(index_producer::produce_manifest(bytes, spec()).is_err());
    }
}
