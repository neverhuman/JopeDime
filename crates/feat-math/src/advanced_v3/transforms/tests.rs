use super::*;

use super::super::schema::ColumnSpec;
use super::*;

fn ordered_schema() -> DataSchema {
    DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::numeric("value"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".to_string()],
            order_by: vec!["time".to_string()],
        },
    }
}

fn table(rows: &[(&str, &str, &str)]) -> RawTable {
    RawTable::new(
        vec!["group".to_string(), "time".to_string(), "value".to_string()],
        rows.iter()
            .map(|(group, time, value)| {
                vec![group.to_string(), time.to_string(), value.to_string()]
            })
            .collect(),
    )
    .unwrap()
}

fn one_feature(values: &[f64], missing: &[bool]) -> FeatureBlock {
    FeatureBlock::new(
        values.iter().map(|value| vec![*value]).collect(),
        vec![FeatureRole::Numeric],
        missing.iter().map(|value| vec![*value]).collect(),
        Vec::new(),
        vec![FeatureLineage::raw(2, "input:numeric")],
        3,
    )
    .unwrap()
}

fn column(block: &FeatureBlock, index: usize) -> Vec<f64> {
    block.values.iter().map(|row| row[index]).collect()
}

fn mask_column(block: &FeatureBlock, index: usize) -> Vec<bool> {
    block.missing_masks.iter().map(|row| row[index]).collect()
}

#[test]
fn lag_and_roll_are_group_order_aware_and_restore_caller_order() {
    // Caller order is deliberately neither group nor time order. The two
    // A/time=2 rows also prove original ordinal is the stable final key.
    let raw = table(&[
        ("B", "2", "200"),
        ("A", "2", "20"),
        ("A", "1", "10"),
        ("B", "1", "100"),
        ("A", "2", "21"),
    ]);
    let block = one_feature(&[200.0, 20.0, 10.0, 100.0, 21.0], &[false; 5]);
    let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();

    let (lag, _) =
        apply_row_operation(&block, &raw, &order, 1, false, RowOperation::Lag, &[]).unwrap();
    assert_eq!(column(&lag, 0), [100.0, 10.0, 0.0, 0.0, 20.0]);
    assert_eq!(mask_column(&lag, 0), [false, false, true, true, false]);

    let (roll, _) =
        apply_row_operation(&block, &raw, &order, 2, false, RowOperation::Roll, &[]).unwrap();
    assert_eq!(column(&roll, 0), [150.0, 15.0, 10.0, 100.0, 20.5]);
    assert_eq!(mask_column(&roll, 0), [false; 5]);
}

#[test]
fn rolling_all_missing_window_stays_missing() {
    let raw = table(&[("A", "1", ""), ("A", "2", ""), ("A", "3", "4")]);
    let block = one_feature(&[0.0, 0.0, 4.0], &[true, true, false]);
    let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
    let (roll, _) =
        apply_row_operation(&block, &raw, &order, 2, false, RowOperation::Roll, &[]).unwrap();
    assert_eq!(column(&roll, 0), [0.0, 0.0, 4.0]);
    assert_eq!(mask_column(&roll, 0), [true, true, false]);
}

#[test]
fn rolling_mean_uses_a_sequential_f64_fold() {
    let raw = table(&[("A", "1", "1e16"), ("A", "2", "1"), ("A", "3", "-1e16")]);
    let block = one_feature(&[1e16, 1.0, -1e16], &[false; 3]);
    let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
    let (roll, _) =
        apply_row_operation(&block, &raw, &order, 3, false, RowOperation::Roll, &[]).unwrap();
    assert_eq!(roll.values[2][0].to_bits(), 0.0f64.to_bits());
}

#[test]
fn typed_multicolumn_keys_cover_missing_nan_unicode_and_signed_zero() {
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("group_number"),
            ColumnSpec::categorical("group_name"),
            ColumnSpec::numeric("order_number"),
            ColumnSpec::categorical("order_name"),
            ColumnSpec::numeric("value"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group_number".to_string(), "group_name".to_string()],
            order_by: vec!["order_number".to_string(), "order_name".to_string()],
        },
    };
    let raw = RawTable::new(
        schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect(),
        vec![
            vec![
                "-0".into(),
                "雪".into(),
                "nan".into(),
                "nan".into(),
                "1".into(),
            ],
            vec![
                "0".into(),
                "雪".into(),
                "NA".into(),
                "nan".into(),
                "2".into(),
            ],
            vec![
                "nan".into(),
                "雪".into(),
                "0".into(),
                "é".into(),
                "3".into(),
            ],
            vec![
                "NA".into(),
                "雪".into(),
                "-0".into(),
                "é".into(),
                "4".into(),
            ],
        ],
    )
    .unwrap();
    let order = FittedTableOrder::from_schema(&schema).unwrap();
    let (groups, orders) = table_keys(&raw, &order).unwrap();
    assert_eq!(groups[0], groups[1], "signed zero is one numeric key");
    assert_eq!(groups[2], groups[3], "numeric nan/NA are missing keys");
    assert_ne!(groups[0], groups[2]);
    assert_eq!(
        orders[0], orders[1],
        "missing numeric order ties use ordinal"
    );
    assert_eq!(orders[2], orders[3], "signed-zero numeric order ties");
    assert_eq!(groups[0].len(), 2);
    assert_eq!(orders[0].len(), 2);
    assert_eq!(groups[0][1], SemanticKey::Categorical("雪".to_string()));
    assert_eq!(orders[0][0], SemanticKey::Missing);
    assert_eq!(orders[0][1], SemanticKey::Categorical("nan".to_string()));

    let block = FeatureBlock::new(
        vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
        vec![FeatureRole::Numeric],
        vec![vec![false]; 4],
        Vec::new(),
        vec![FeatureLineage::raw(4, "input:numeric")],
        5,
    )
    .unwrap();
    let (lag, _) =
        apply_row_operation(&block, &raw, &order, 1, false, RowOperation::Lag, &[]).unwrap();
    assert_eq!(column(&lag, 0), [0.0, 1.0, 0.0, 3.0]);
    assert_eq!(mask_column(&lag, 0), [true, false, true, false]);
}

#[test]
fn append_preserves_original_metadata_and_generated_missing_mask() {
    let raw = table(&[("A", "1", "5"), ("A", "2", "6")]);
    let block = one_feature(&[5.0, 6.0], &[false, false]);
    let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
    let (lag, _) =
        apply_row_operation(&block, &raw, &order, 1, true, RowOperation::Lag, &[]).unwrap();

    assert_eq!(lag.values, [vec![5.0, 0.0], vec![6.0, 5.0]]);
    assert_eq!(lag.missing_masks, [vec![false, true], vec![false, false]]);
    assert_eq!(lag.roles, [FeatureRole::Numeric, FeatureRole::Numeric]);
    assert_eq!(lag.lineage[0].origin, FeatureOrigin::RawDerived);
    assert_eq!(lag.lineage[1].origin, FeatureOrigin::GeneratedDense);
    assert_eq!(lag.lineage[1].source_columns, [2]);
    assert_eq!(lag.lineage[1].operations, ["input:numeric", "lag"]);
    assert_eq!(lag.widths().raw_derived, 1);
    assert_eq!(lag.widths().generated_dense, 1);
}

#[test]
fn prediction_uses_fitted_group_tails_and_unseen_groups_have_no_history() {
    let training_raw = table(&[("A", "1", "1"), ("A", "2", "2")]);
    let training = one_feature(&[1.0, 2.0], &[false, false]);
    let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
    let (_, tails) = apply_row_operation(
        &training,
        &training_raw,
        &order,
        2,
        false,
        RowOperation::Lag,
        &[],
    )
    .unwrap();
    let op = V3Op::Lag {
        k: 2,
        append: false,
        order,
        tails,
    };

    // A rows arrive in reverse chronological caller order. B is unseen.
    let prediction_raw = table(&[("A", "4", "4"), ("A", "3", "3"), ("B", "1", "10")]);
    let prediction = one_feature(&[4.0, 3.0, 10.0], &[false; 3]);
    let transformed = op.apply_block(&prediction, &prediction_raw).unwrap();
    assert_eq!(column(&transformed, 0), [2.0, 1.0, 0.0]);
    assert_eq!(mask_column(&transformed, 0), [false, false, true]);
}

#[test]
fn rolling_prediction_starts_with_fitted_tail() {
    let training_raw = table(&[("A", "1", "1"), ("A", "2", "2")]);
    let training = one_feature(&[1.0, 2.0], &[false, false]);
    let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
    let (_, tails) = apply_row_operation(
        &training,
        &training_raw,
        &order,
        3,
        false,
        RowOperation::Roll,
        &[],
    )
    .unwrap();
    let op = V3Op::Roll {
        k: 3,
        append: false,
        order,
        tails,
    };
    let raw = table(&[("A", "3", "3"), ("B", "1", "10")]);
    let prediction = one_feature(&[3.0, 10.0], &[false, false]);
    let transformed = op.apply_block(&prediction, &raw).unwrap();
    assert_eq!(column(&transformed, 0), [2.0, 10.0]);
    assert_eq!(mask_column(&transformed, 0), [false, false]);
}

#[test]
fn fitted_row_tails_reject_earlier_and_equal_prediction_orders() {
    let training_raw = table(&[("A", "1", "1"), ("A", "2", "2")]);
    let training = one_feature(&[1.0, 2.0], &[false, false]);
    let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();

    for operation in [RowOperation::Lag, RowOperation::Roll] {
        let (_, tails) =
            apply_row_operation(&training, &training_raw, &order, 2, false, operation, &[])
                .unwrap();

        for prediction_order in ["1", "2"] {
            let prediction_raw = table(&[("A", prediction_order, "99")]);
            let prediction = one_feature(&[99.0], &[false]);
            let error = apply_row_operation(
                &prediction,
                &prediction_raw,
                &order,
                2,
                false,
                operation,
                &tails,
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("strictly after fitted order"),
                "unexpected {operation:?} error: {error}"
            );
        }

        let future_raw = table(&[("A", "3", "3"), ("B", "1", "10")]);
        let future = one_feature(&[3.0, 10.0], &[false, false]);
        apply_row_operation(&future, &future_raw, &order, 2, false, operation, &tails).unwrap();
    }
}

#[test]
fn fitted_row_tails_reject_corrupt_serialized_key_state() {
    let training_raw = table(&[("A", "1", "1"), ("A", "2", "2")]);
    let training = one_feature(&[1.0, 2.0], &[false, false]);
    let order = FittedTableOrder::from_schema(&ordered_schema()).unwrap();
    let (_, tails) = apply_row_operation(
        &training,
        &training_raw,
        &order,
        2,
        false,
        RowOperation::Lag,
        &[],
    )
    .unwrap();
    let encoded = serde_json::to_value(&tails).unwrap();

    let mut missing_cutoff = encoded.clone();
    missing_cutoff[0]["last_order"][0] = serde_json::json!("Missing");
    let mut nonfinite_cutoff = encoded.clone();
    nonfinite_cutoff[0]["last_order"][0] = serde_json::json!({"Numeric": f64::NAN.to_bits()});
    let mut wrong_order_kind = encoded.clone();
    wrong_order_kind[0]["last_order"][0] = serde_json::json!({"Categorical": "2"});
    let mut wrong_group_kind = encoded.clone();
    wrong_group_kind[0]["key"][0] = serde_json::json!({"Numeric": 0});

    let prediction_raw = table(&[("A", "1", "99")]);
    let prediction = one_feature(&[99.0], &[false]);
    for corrupt in [
        missing_cutoff,
        nonfinite_cutoff,
        wrong_order_kind,
        wrong_group_kind,
    ] {
        let corrupt_tails: Vec<GroupTail> = serde_json::from_value(corrupt).unwrap();
        let error = apply_row_operation(
            &prediction,
            &prediction_raw,
            &order,
            2,
            false,
            RowOperation::Lag,
            &corrupt_tails,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("fitted row-operation tail"),
            "unexpected corrupt-tail error: {error}"
        );
    }

    let mut missing_field = encoded;
    missing_field[0]
        .as_object_mut()
        .unwrap()
        .remove("last_order");
    assert!(serde_json::from_value::<Vec<GroupTail>>(missing_field).is_err());

    let mut wrong_order = order.clone();
    wrong_order.order_columns[0].0 = 2;
    let op = V3Op::Lag {
        k: 2,
        append: false,
        order: wrong_order,
        tails,
    };
    assert!(op
        .validate_fitted_state(&ordered_schema())
        .unwrap_err()
        .to_string()
        .contains("does not match schema"));
}

#[test]
fn one_to_one_v3_transform_never_truncates_wide_raw_block() {
    let width = 1_024;
    let block = FeatureBlock::new(
        vec![vec![0.5; width]],
        vec![FeatureRole::Numeric; width],
        vec![vec![false; width]],
        Vec::new(),
        (0..width)
            .map(|source| FeatureLineage::raw(source, "input:numeric"))
            .collect(),
        width,
    )
    .unwrap();
    let raw = RawTable::new(vec!["x".to_string()], vec![vec!["0.5".to_string()]]).unwrap();
    let transformed = V3Op::MaxAbs {
        scales: vec![0.5; width],
        append: false,
    }
    .apply_block(&block, &raw)
    .unwrap();
    assert_eq!(transformed.values[0].len(), width);
    assert_eq!(transformed.lineage.len(), width);
    assert_eq!(transformed.values[0][width - 1], 1.0);
}
fn raw_transform_fixture(width: usize) -> FeatureBlock {
    FeatureBlock::new(
        vec![vec![2.0; width]],
        vec![FeatureRole::Numeric; width],
        vec![vec![false; width]],
        vec![],
        (0..width)
            .map(|source| FeatureLineage::raw(source, "input:numeric"))
            .collect(),
        width,
    )
    .unwrap()
}

#[test]
fn legacy_one_to_one_v3_preserves_wide_raw_columns_and_masks() {
    let width = 1024;
    let mut block = raw_transform_fixture(width);
    block.missing_masks[0][width - 1] = true;
    block.values[0][width - 1] = 0.0;
    let op = FittedOp::Standardize {
        means: vec![1.0; width],
        scales: vec![2.0; width],
        append: false,
    };
    // Legacy public behavior remains capped; v3 retains all original columns.
    assert_eq!(
        crate::ops::apply_op(&op, &block.values).unwrap()[0].len(),
        crate::MAX_WIDTH
    );
    let output = legacy_to_block(&op, &block).unwrap();
    assert_eq!(output.values[0].len(), width);
    assert_eq!(output.values[0][width - 2], 0.5);
    assert_eq!(output.values[0][width - 1], 0.0);
    assert_eq!(output.missing_masks, block.missing_masks);
    assert_eq!(output.widths().raw_derived, width);
    assert_eq!(output.widths().generated_dense, 0);
}

#[test]
fn legacy_append_v3_keeps_raw_plus_allowed_generated_width_on_reload() {
    let width = 300;
    let block = raw_transform_fixture(width);
    let op = V3Op::Legacy(FittedOp::RobustScale {
        medians: vec![0.0; width],
        scales: vec![2.0; width],
        append: true,
    });
    let raw = RawTable::new(vec!["x".into()], vec![vec!["2".into()]]).unwrap();
    let output = op.apply_block(&block, &raw).unwrap();
    assert_eq!(output.values[0].len(), 600);
    assert_eq!(output.values[0][299], 2.0);
    assert_eq!(output.values[0][599], 1.0);
    assert_eq!(output.widths().raw_derived, 300);
    assert_eq!(output.widths().generated_dense, 300);
    let reloaded: V3Op = serde_json::from_slice(&serde_json::to_vec(&op).unwrap()).unwrap();
    let replayed = reloaded.apply_block(&block, &raw).unwrap();
    assert_eq!(
        serde_json::to_value(&replayed).unwrap(),
        serde_json::to_value(&output).unwrap()
    );
    assert_eq!(reloaded.apply(&block.values), output.values);
}

#[test]
fn legacy_append_v3_rejects_dense_overflow_before_losing_feature_columns() {
    let block = raw_transform_fixture(512);
    let op = FittedOp::Standardize {
        means: vec![0.0; 512],
        scales: vec![1.0; 512],
        append: true,
    };
    let mut dense = block.clone();
    dense.roles = vec![FeatureRole::Generated; 512];
    dense.lineage = (0..512)
        .map(|source| FeatureLineage::generated_dense(vec![source], "synthetic:generated"))
        .collect();
    dense.validate().unwrap();
    let error = legacy_to_block(&op, &dense).unwrap_err().to_string();
    assert!(
        error.contains("generated dense width 1024 exceeds 512"),
        "{error}"
    );
    assert!(!error.contains("row/mask"));
}
