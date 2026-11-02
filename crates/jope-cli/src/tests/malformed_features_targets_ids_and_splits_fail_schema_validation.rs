use super::*;

#[test]
fn malformed_features_targets_ids_and_splits_fail_schema_validation() {
    for alteration in 0..9 {
        let mut data = dataset(Task::Binary, 0);
        match alteration {
            0 => data.rows[0].target = 2,
            1 => data.rows[0].features.clear(),
            2 => data.rows[0].features[0] = Value::Null,
            3 => data.rows[0].features[0] = json!(1.25),
            4 => data.rows[0].id = data.rows[1].id,
            5 => data
                .rows
                .iter_mut()
                .for_each(|row| row.split = Split::Train),
            6 => data.columns[0].name = data.columns[1].name.clone(),
            7 => data.rows[0].series = Some("unexpected".into()),
            _ => data.classes = vec![0, 2],
        }
        assert!(validate_dataset(&data).is_err(), "alteration {alteration}");
    }
    let mut categorical = named(Task::Binary, "categorical");
    validate_dataset(&categorical).unwrap();
    let column = categorical
        .columns
        .iter()
        .position(|column| column.kind == Kind::Categorical)
        .expect("the categorical scenario declares a categorical column");
    categorical.rows[0].features[column] = "undeclared".into();
    assert!(validate_dataset(&categorical)
        .unwrap_err()
        .to_string()
        .contains("unknown category"));
}
