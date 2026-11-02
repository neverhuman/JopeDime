use super::*;
use crate::advanced_v3::schema::ColumnSpec;
use crate::advanced_v3::types::Objective;

fn numeric_schema(width: usize) -> DataSchema {
    DataSchema {
        columns: (0..width)
            .map(|idx| ColumnSpec::numeric(format!("c{idx}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

fn rows_to_table(x: &[Vec<f64>]) -> RawTable {
    let width = x[0].len();
    let headers = (0..width).map(|idx| format!("c{idx}")).collect();
    let rows = x
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect();
    RawTable::new(headers, rows).unwrap()
}

#[test]
fn retained_legacy_shortcut_declines_v3_overwidth_hinge_and_mismatched_state() {
    let genome = GenomeIr::parse("(pipe X (hinge knots=4 append=true) (ridge))").unwrap();
    let task = TaskSpec::regression(Objective::Rmse).unwrap();

    // 200 columns: legacy caps at 512 total columns, whereas V3 generates 200*4 = 800 > 512.
    let x200: Vec<Vec<f64>> = (0..16)
        .map(|r| {
            (0..200)
                .map(|c| ((r * 17 + c * 13) as f64 * 0.07).sin())
                .collect()
        })
        .collect();
    let y: Vec<f64> = (0..16).map(|r| (r as f64 * 0.19).cos()).collect();
    let retained_200 = crate::model::fit(&genome, &x200, &y).unwrap();
    let schema_200 = numeric_schema(200);
    assert!(
        try_bound_pipeline_from_retained_legacy_model(
            &genome,
            &task,
            &schema_200,
            &x200,
            &y,
            &retained_200
        )
        .unwrap()
        .is_none(),
        "200-column hinge (800 generated dense > 512) must be declined by retained shortcut"
    );

    // 120 columns: generated_dense = 480 <= 512, but V3 total width = 600 > 512 while legacy capped at 512.
    let x120: Vec<Vec<f64>> = x200.iter().map(|row| row[..120].to_vec()).collect();
    let retained_120 = crate::model::fit(&genome, &x120, &y).unwrap();
    let schema_120 = numeric_schema(120);
    assert!(
        try_bound_pipeline_from_retained_legacy_model(
            &genome,
            &task,
            &schema_120,
            &x120,
            &y,
            &retained_120
        )
        .unwrap()
        .is_none(),
        "120-column hinge (600 total > 512 legacy cap) must be declined by retained shortcut"
    );

    // Width-mismatched retained model (fitted on 4 columns, queried on 8-column schema).
    let x4: Vec<Vec<f64>> = x200.iter().map(|row| row[..4].to_vec()).collect();
    let x8: Vec<Vec<f64>> = x200.iter().map(|row| row[..8].to_vec()).collect();
    let retained_4 = crate::model::fit(&genome, &x4, &y).unwrap();
    let schema_8 = numeric_schema(8);
    assert!(
        try_bound_pipeline_from_retained_legacy_model(
            &genome,
            &task,
            &schema_8,
            &x8,
            &y,
            &retained_4
        )
        .unwrap()
        .is_none(),
        "width-mismatched retained model must be declined by retained shortcut"
    );

    // No-transform pipeline `(pipe X (ridge))` fitted on 4 columns, queried on 8-column schema (H4-R1-WIDTH).
    let ridge_only_genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let retained_ridge_4 = crate::model::fit(&ridge_only_genome, &x4, &y).unwrap();
    assert!(
        try_bound_pipeline_from_retained_legacy_model(
            &ridge_only_genome,
            &task,
            &schema_8,
            &x8,
            &y,
            &retained_ridge_4
        )
        .unwrap()
        .is_none(),
        "no-transform retained ridge fitted on 4 columns must be declined on 8-column schema"
    );
    // Matching 4-column schema is accepted and matches fit_table bit-for-bit.
    let schema_4 = numeric_schema(4);
    let train_table_4 = rows_to_table(&x4);
    let shortcut_ridge_4 = try_bound_pipeline_from_retained_legacy_model(
        &ridge_only_genome,
        &task,
        &schema_4,
        &x4,
        &y,
        &retained_ridge_4,
    )
    .unwrap()
    .expect("matching 4-column no-transform ridge should be accepted");
    let full_ridge_4 = fit_table(&ridge_only_genome, &task, &schema_4, &train_table_4, &y).unwrap();
    assert_eq!(
        serde_json::to_string(&shortcut_ridge_4).unwrap(),
        serde_json::to_string(&full_ridge_4).unwrap()
    );
}

#[test]
fn retained_legacy_shortcut_declines_missing_and_non_finite_training_inputs() {
    let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let task = TaskSpec::regression(Objective::Rmse).unwrap();
    let schema = numeric_schema(4);
    let mut x: Vec<Vec<f64>> = (0..16)
        .map(|r| {
            (0..4)
                .map(|c| ((r * 13 + c * 7) as f64 * 0.11).sin() + 2.0)
                .collect()
        })
        .collect();
    let y: Vec<f64> = (0..16).map(|r| (r as f64 * 0.19).cos()).collect();
    let retained_finite = crate::model::fit(&genome, &x, &y).unwrap();

    // Missing numeric cell represented as NaN (or empty cell in RawTable):
    // fit_table re-masks masked cells to 0.0 after standardize, whereas legacy model::fit
    // replaces NaN with 0.0 before standardize. The shortcut must decline (return Ok(None)).
    x[3][1] = f64::NAN;
    let retained_nan = crate::model::fit(&genome, &x, &y).unwrap();
    assert!(
        try_bound_pipeline_from_retained_legacy_model(
            &genome,
            &task,
            &schema,
            &x,
            &y,
            &retained_nan
        )
        .unwrap()
        .is_none(),
        "NaN/missing training cell must be declined so fit_table applies per-stage masks"
    );
    let mut table_with_empty = rows_to_table(&x);
    table_with_empty.rows[3][1] = String::new();
    let v3_bound = fit_table(&genome, &task, &schema, &table_with_empty, &y).unwrap();
    assert!(v3_bound.validate_binding().is_ok());

    // Infinite cell in x must also be declined.
    x[3][1] = f64::INFINITY;
    assert!(
        try_bound_pipeline_from_retained_legacy_model(
            &genome,
            &task,
            &schema,
            &x,
            &y,
            &retained_finite
        )
        .unwrap()
        .is_none(),
        "infinite training cell must be declined by retained shortcut"
    );

    // Non-finite target y must also be declined (and refused by fit_table).
    x[3][1] = 1.25;
    let mut y_nan = y.clone();
    y_nan[2] = f64::NAN;
    assert!(
        try_bound_pipeline_from_retained_legacy_model(
            &genome,
            &task,
            &schema,
            &x,
            &y_nan,
            &retained_finite
        )
        .unwrap()
        .is_none(),
        "non-finite target y must be declined by retained shortcut"
    );
}

#[test]
fn retained_legacy_shortcut_matches_fit_table_bit_for_bit_when_eligible() {
    let genome =
        GenomeIr::parse("(pipe X (standardize append=true) (hinge knots=3 append=true) (ridge))")
            .unwrap();
    let task = TaskSpec::regression(Objective::Rmse).unwrap();
    let x_train: Vec<Vec<f64>> = (0..24)
        .map(|r| {
            (0..8)
                .map(|c| ((r * 11 + c * 7) as f64 * 0.13).sin())
                .collect()
        })
        .collect();
    let y_train: Vec<f64> = (0..24)
        .map(|r| x_train[r][0] * 0.6 - x_train[r][1] * 0.3 + (r as f64 * 0.05).cos())
        .collect();
    let x_hold: Vec<Vec<f64>> = (0..12)
        .map(|r| {
            (0..8)
                .map(|c| ((r * 19 + c * 5) as f64 * 0.11).cos())
                .collect()
        })
        .collect();

    let schema = numeric_schema(8);
    let train_table = rows_to_table(&x_train);
    let hold_table = rows_to_table(&x_hold);

    let retained = crate::model::fit(&genome, &x_train, &y_train).unwrap();
    let shortcut_bound = try_bound_pipeline_from_retained_legacy_model(
        &genome, &task, &schema, &x_train, &y_train, &retained,
    )
    .unwrap()
    .expect("eligible within-512 pipeline should use retained shortcut");
    let full_bound = fit_table(&genome, &task, &schema, &train_table, &y_train).unwrap();

    assert_eq!(
        serde_json::to_string(&shortcut_bound).unwrap(),
        serde_json::to_string(&full_bound).unwrap()
    );
    let shortcut_preds: Vec<u64> = shortcut_bound
        .predict_table(&hold_table)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect();
    let full_preds: Vec<u64> = full_bound
        .predict_table(&hold_table)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect();
    assert_eq!(shortcut_preds, full_preds);
}
