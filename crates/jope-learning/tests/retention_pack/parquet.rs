use super::*;

#[test]
fn parquet_subset_round_trips_every_type_and_null() {
    let columns = vec![
        ColumnSpec {
            name: "s".into(),
            ty: ColumnType::Utf8,
            required: true,
        },
        ColumnSpec {
            name: "os".into(),
            ty: ColumnType::Utf8,
            required: false,
        },
        ColumnSpec {
            name: "i".into(),
            ty: ColumnType::Int64,
            required: false,
        },
        ColumnSpec {
            name: "d".into(),
            ty: ColumnType::Double,
            required: false,
        },
        ColumnSpec {
            name: "b".into(),
            ty: ColumnType::Boolean,
            required: true,
        },
    ];
    let rows: Vec<Vec<Cell>> = (0..37)
        .map(|n| {
            vec![
                Cell::Str(format!("row-{n}")),
                if n % 3 == 0 {
                    Cell::Null
                } else {
                    Cell::Str("x".repeat(n))
                },
                if n % 4 == 0 {
                    Cell::Null
                } else {
                    Cell::I64(n as i64 * -7)
                },
                if n % 5 == 0 {
                    Cell::Null
                } else {
                    Cell::F64(n as f64 / 3.0)
                },
                Cell::Bool(n % 2 == 0),
            ]
        })
        .collect();
    let table = Table {
        columns,
        rows,
        metadata: vec![("k".into(), "v".into())],
    };
    for zstd in [false, true] {
        let bytes = parquet::write(&table, zstd, "test").unwrap();
        assert_eq!(parquet::read(&bytes).unwrap(), table);
    }
    let empty = Table {
        rows: vec![],
        ..table.clone()
    };
    assert_eq!(
        parquet::read(&parquet::write(&empty, true, "test").unwrap()).unwrap(),
        empty
    );
    let mut bad = parquet::write(&table, true, "test").unwrap();
    bad[0] = b'X';
    assert!(parquet::read(&bad).is_err());
    let mut null_in_required = table.clone();
    null_in_required.rows[0][0] = Cell::Null;
    assert!(parquet::write(&null_in_required, true, "test").is_err());
    // Optional external cross-check: write a sample for an independent reader (pyarrow).
    if let Some(out) = std::env::var_os("JOPE_RETENTION_PARQUET_SAMPLE") {
        std::fs::write(&out, parquet::write(&table, true, "test").unwrap()).unwrap();
    }
}
