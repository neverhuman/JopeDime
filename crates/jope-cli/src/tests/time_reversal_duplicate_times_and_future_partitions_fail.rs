use super::*;

#[test]
fn time_reversal_duplicate_times_and_future_partitions_fail() {
    for alteration in 0..4 {
        let mut data = dataset(Task::TimeSeries, 0);
        match alteration {
            0 => data.rows[1].timestamp = data.rows[0].timestamp,
            1 => data.rows.swap(0, 1),
            2 => data.rows[0].split = Split::Test,
            _ => data.rows[0].timestamp = None,
        }
        assert!(validate_dataset(&data).is_err(), "alteration {alteration}");
    }
    let mut multiple = named(Task::TimeSeries, "multiple_series");
    validate_dataset(&multiple).unwrap();
    for row in multiple
        .rows
        .iter_mut()
        .filter(|row| row.series.as_deref() == Some("series-0"))
    {
        row.timestamp = row.timestamp.map(|time| time + 100_000);
    }
    assert!(validate_dataset(&multiple)
        .unwrap_err()
        .to_string()
        .contains("overlap"));
}
