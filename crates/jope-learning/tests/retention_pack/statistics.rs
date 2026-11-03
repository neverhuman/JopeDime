use jope_learning::retention::parquet as parquet_api;

use super::*;

#[test]
fn optional_statistics_preserve_legacy_distilled_bytes_and_reader() {
    let rows = vec![
        row('a', 'b', Some(0.59), Outcome::Ok),
        row('b', 'b', None, Outcome::Failed),
    ];
    let legacy = distilled::write(&rows, &known(), &[]).unwrap();
    let table = parquet::read(&legacy).unwrap();
    let opts = parquet::WriteOptions {
        codec: parquet_api::Codec::Zstd,
        row_group_rows: 0,
        page_rows: 0,
        dictionary: false,
    };
    let with_stats = parquet::write_with_statistics(&table, &opts, "fixture").unwrap();
    assert_eq!(distilled::read(&with_stats, &known()).unwrap(), rows);
    assert_eq!(parquet::read(&with_stats).unwrap(), table);
    assert_eq!(distilled::write(&rows, &known(), &[]).unwrap(), legacy);
    let old_reader = parquet_api::FileReader::open(&legacy).unwrap();
    assert_eq!(old_reader.statistics(0, 7).unwrap(), None);
    let reader = parquet_api::FileReader::open(&with_stats).unwrap();
    let stats = reader.statistics(0, 7).unwrap().unwrap();
    assert_eq!(stats.value_count, 2);
    assert_eq!(stats.null_count, Some(1));
    assert_eq!(stats.min, Some(parquet_api::StatisticValue::F64(0.59)));
}

#[test]
fn nullable_multigroup_statistics_reconcile_to_legacy_streamed_values() {
    let rows = vec![
        row('a', 'b', Some(0.59), Outcome::Ok),
        row('b', 'b', None, Outcome::Timeout),
        row('c', 'b', Some(0.61), Outcome::Ok),
    ];
    let legacy = distilled::write(&rows, &known(), &[]).unwrap();
    let table = parquet::read(&legacy).unwrap();
    let bytes = parquet::write_with_statistics(
        &table,
        &parquet::WriteOptions {
            codec: parquet_api::Codec::Zstd,
            row_group_rows: 2,
            page_rows: 1,
            dictionary: true,
        },
        "fixture",
    )
    .unwrap();
    let reader = parquet_api::FileReader::open(&bytes).unwrap();
    assert_eq!(reader.row_groups.len(), 2);
    assert_eq!(reader.num_rows, 3);
    assert_eq!(
        reader.read_column(0, 7).unwrap().0,
        parquet::Values::F64(vec![Some(0.59), None])
    );
    assert_eq!(
        reader.read_column(1, 7).unwrap().0,
        parquet::Values::F64(vec![Some(0.61)])
    );
    assert_eq!(
        reader.statistics(0, 7).unwrap().unwrap().null_count,
        Some(1)
    );
    assert_eq!(
        reader.statistics(1, 7).unwrap().unwrap().null_count,
        Some(0)
    );
    assert!(reader.statistics(2, 7).is_err());
    assert!(reader.statistics(0, 200).is_err());
}
