use super::super::schema::ColumnSpec;
use super::*;

fn config(numeric: NumericMode, categorical: CategoricalMode, prior: f64) -> InputConfig {
    InputConfig {
        numeric,
        missing_indicator: false,
        categorical,
        max_onehot: 16,
        min_count: 1,
        prior,
        folds: 3,
    }
}

fn one_column_schema(kind: ColumnKind) -> DataSchema {
    DataSchema {
        columns: vec![ColumnSpec {
            name: "x".to_string(),
            kind,
        }],
        row_semantics: RowSemantics::Iid,
    }
}

#[test]
fn numeric_modes_treat_tokens_and_nonfinite_values_as_missing() {
    let table = RawTable::new(
        vec!["x".into()],
        ["", " na ", "NaN", "null", "inf", "-inf", "1", "3"]
            .into_iter()
            .map(|value| vec![value.to_string()])
            .collect(),
    )
    .unwrap();
    let schema = one_column_schema(ColumnKind::Numeric);
    let y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    for (mode, fill) in [
        (NumericMode::Median, 2.0_f64),
        (NumericMode::Mean, 2.0_f64),
        (NumericMode::Zero, 0.0_f64),
    ] {
        let (_, block) = FittedInput::fit_transform(
            config(mode, CategoricalMode::Native, 0.0),
            &schema,
            &table,
            &y,
            TaskKind::Regression,
        )
        .unwrap();
        assert!(block.values[..6]
            .iter()
            .all(|row| row[0].to_bits() == fill.to_bits()));
        assert!(block.missing_masks[..6].iter().all(|row| row[0]));
        assert_eq!(block.values[6][0], 1.0);
        assert_eq!(block.values[7][0], 3.0);
    }
}

#[test]
fn categorical_vocabulary_is_canonical_and_missing_nan_unseen_are_distinct() {
    let table = RawTable::new(
        vec!["x".into()],
        ["z", "", "nan", "a", "NA", "null"]
            .into_iter()
            .map(|value| vec![value.to_string()])
            .collect(),
    )
    .unwrap();
    let schema = one_column_schema(ColumnKind::Categorical);
    let y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let (fitted, block) = FittedInput::fit_transform(
        config(NumericMode::Zero, CategoricalMode::Native, 0.0),
        &schema,
        &table,
        &y,
        TaskKind::Regression,
    )
    .unwrap();
    let FittedColumn::Categorical { vocabulary, .. } = &fitted.columns[0] else {
        panic!("categorical column expected")
    };
    assert_eq!(vocabulary, &["\0MISSING", "a", "nan", "z"]);
    assert_eq!(block.roles, [FeatureRole::CategoricalNative]);

    let prediction = RawTable::new(
        vec!["x".into()],
        vec![vec!["".into()], vec!["nan".into()], vec!["unseen".into()]],
    )
    .unwrap();
    let encoded = fitted.transform(&prediction).unwrap();
    assert_ne!(encoded.values[0][0], encoded.values[1][0]);
    assert_eq!(encoded.values[2][0], -1.0);
    assert_eq!(encoded.missing_masks, [[true], [false], [false]]);
}

#[test]
fn every_categorical_mode_keeps_its_declared_roles() {
    let table = RawTable::new(
        vec!["x".into()],
        ["b", "a", "nan", ""]
            .into_iter()
            .map(|value| vec![value.to_string()])
            .collect(),
    )
    .unwrap();
    let schema = one_column_schema(ColumnKind::Categorical);
    let y = [0.1, 0.4, 0.7, 1.2];
    for (mode, expected) in [
        (
            CategoricalMode::Native,
            vec![FeatureRole::CategoricalNative],
        ),
        (
            CategoricalMode::Frequency,
            vec![FeatureRole::CategoricalFrequency],
        ),
        (
            CategoricalMode::Ordered,
            vec![FeatureRole::CategoricalOrdered],
        ),
    ] {
        let (_, block) = FittedInput::fit_transform(
            config(NumericMode::Zero, mode, 0.0),
            &schema,
            &table,
            &y,
            TaskKind::Regression,
        )
        .unwrap();
        assert_eq!(block.roles, expected);
    }

    let (_, onehot) = FittedInput::fit_transform(
        config(NumericMode::Zero, CategoricalMode::Onehot, 0.0),
        &schema,
        &table,
        &y,
        TaskKind::Regression,
    )
    .unwrap();
    assert_eq!(onehot.roles, vec![FeatureRole::CategoricalOneHot; 4]);

    let mut hybrid_config = config(NumericMode::Zero, CategoricalMode::Hybrid, 0.0);
    hybrid_config.max_onehot = 1;
    let (_, hybrid) =
        FittedInput::fit_transform(hybrid_config, &schema, &table, &y, TaskKind::Regression)
            .unwrap();
    assert_eq!(
        hybrid.roles,
        [
            FeatureRole::CategoricalFrequency,
            FeatureRole::CategoricalOrdered
        ]
    );
}

#[test]
fn iid_ordered_encoding_uses_only_fold_training_statistics() {
    let indices = vec![Some(0); 6];
    let y = [0.2, 10.0, 1.2, 20.0, 2.2, 30.0];
    let original = honest_ordered_values(&indices, &y, 0.0, 3);
    let mut permuted = y;
    permuted.swap(0, 3);
    let changed = honest_ordered_values(&indices, &permuted, 0.0, 3);
    assert_eq!(original[0].to_bits(), changed[0].to_bits());
    assert_eq!(original[3].to_bits(), changed[3].to_bits());

    let unique: Vec<Option<usize>> = (0..6).map(Some).collect();
    let encoded = honest_ordered_values(&unique, &y, 0.0, 3);
    for (row, &actual) in encoded.iter().enumerate().take(y.len()) {
        let fold = row % 3;
        let expected: Vec<f64> = y
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (index % 3 != fold).then_some(*value))
            .collect();
        let expected = expected.iter().sum::<f64>() / expected.len() as f64;
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn ordered_history_is_stable_for_ties_and_has_no_future_access() {
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::categorical("category"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".into()],
            order_by: vec!["time".into()],
        },
    };
    let table = RawTable::new(
        vec!["group".into(), "time".into(), "category".into()],
        vec![
            vec!["A".into(), "10".into(), "x".into()],
            vec!["A".into(), "2".into(), "x".into()],
            vec!["A".into(), "10".into(), "x".into()],
        ],
    )
    .unwrap();
    let mut ordered_config = config(NumericMode::Zero, CategoricalMode::Ordered, 0.0);
    ordered_config.folds = 2;
    let y = [20.0, 10.0, 30.0];
    let (fitted, block) = FittedInput::fit_transform(
        ordered_config.clone(),
        &schema,
        &table,
        &y,
        TaskKind::Regression,
    )
    .unwrap();
    assert_eq!(block.values[0][2], 10.0);
    assert_eq!(block.values[1][2], 0.0);
    assert_eq!(block.values[2][2], 15.0);

    let mut future_changed = y;
    future_changed[2] = 999.0;
    let (_, changed) = FittedInput::fit_transform(
        ordered_config,
        &schema,
        &table,
        &future_changed,
        TaskKind::Regression,
    )
    .unwrap();
    for row in 0..3 {
        assert_eq!(
            block.values[row][2].to_bits(),
            changed.values[row][2].to_bits()
        );
    }

    let prediction = RawTable::new(
        vec!["group".into(), "time".into(), "category".into()],
        vec![
            vec!["A".into(), "11".into(), "x".into()],
            vec!["A".into(), "12".into(), "unseen".into()],
            vec!["B".into(), "1".into(), "x".into()],
        ],
    )
    .unwrap();
    let predicted = fitted.transform(&prediction).unwrap();
    assert_eq!(predicted.values[0][2], 20.0);
    assert_eq!(predicted.values[1][2], 20.0);
    assert_eq!(predicted.values[2][2], 20.0);
}

#[test]
fn ordered_history_global_fallback_never_uses_later_group_targets() {
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::categorical("category"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".into()],
            order_by: vec!["time".into()],
        },
    };
    let table = RawTable::new(
        vec!["group".into(), "time".into(), "category".into()],
        vec![
            vec!["A".into(), "100".into(), "x".into()],
            vec!["B".into(), "1".into(), "x".into()],
            vec!["B".into(), "2".into(), "x".into()],
        ],
    )
    .unwrap();
    let ordered_config = config(NumericMode::Zero, CategoricalMode::Ordered, 0.0);
    let (_, baseline) = FittedInput::fit_transform(
        ordered_config.clone(),
        &schema,
        &table,
        &[999.0, 10.0, 20.0],
        TaskKind::Regression,
    )
    .unwrap();
    let (_, perturbed) = FittedInput::fit_transform(
        ordered_config,
        &schema,
        &table,
        &[-999.0, 10.0, 20.0],
        TaskKind::Regression,
    )
    .unwrap();

    assert_eq!(baseline.values[1][2].to_bits(), 0.0_f64.to_bits());
    assert_eq!(baseline.values[2][2].to_bits(), 10.0_f64.to_bits());
    for row in [1, 2] {
        assert_eq!(
            baseline.values[row][2].to_bits(),
            perturbed.values[row][2].to_bits()
        );
    }
}
