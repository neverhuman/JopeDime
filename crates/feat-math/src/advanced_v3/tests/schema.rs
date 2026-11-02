use super::*;

#[test]
fn quoted_csv_parser_handles_commas_quotes_and_newlines() {
    let table =
        RawTable::from_csv("a,b\r\n1,\"x,y\"\r\n2,\"say \"\"hi\"\"\"\n3,\"two\nlines\"\n").unwrap();
    assert_eq!(table.headers, ["a", "b"]);
    assert_eq!(table.rows[0][1], "x,y");
    assert_eq!(table.rows[1][1], "say \"hi\"");
    assert_eq!(table.rows[2][1], "two\nlines");
}

#[test]
fn schema_mismatch_and_iid_row_ops_are_rejected() {
    let bad = RawTable::new(vec!["country".into(), "amount".into()], vec![]).unwrap();
    assert!(bad.validate_against(&schema()).is_err());
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let genome = GenomeIr::parse("(pipe X (lag k=1) (ridge))").unwrap();
    assert!(validate_v3_genome(&genome, &task, &schema()).is_err());
}

#[test]
fn ordered_training_values_do_not_reveal_unique_targets() {
    let schema = DataSchema {
        columns: vec![ColumnSpec::categorical("id")],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::new(
        vec!["id".into()],
        (0..10).map(|row| vec![format!("id-{row}")]).collect(),
    )
    .unwrap();
    let y: Vec<f64> = (0..10).map(|row| row as f64).collect();
    let stage =
        GenomeIr::parse("(pipe X (input categorical=ordered folds=5 prior=10) (ridge))").unwrap();
    let GenomeIr::Pipeline(dsl) = stage else {
        panic!()
    };
    let config = InputConfig::from_stage(Some(&dsl.stages[0])).unwrap();
    let (_, block) =
        FittedInput::fit_transform(config, &schema, &table, &y, TaskKind::Regression).unwrap();
    let expected: [f64; 10] = [5.0, 4.75, 4.5, 4.25, 4.0, 5.0, 4.75, 4.5, 4.25, 4.0];
    for (row, expected) in block.values.iter().zip(expected) {
        assert_eq!(row[0].to_bits(), expected.to_bits());
    }

    for validation_fold in 0..5 {
        let mut perturbed_y = y.clone();
        perturbed_y[validation_fold] += 1_000.0;
        perturbed_y[validation_fold + 5] -= 500.0;
        let config = InputConfig::from_stage(Some(&dsl.stages[0])).unwrap();
        let (_, perturbed) =
            FittedInput::fit_transform(config, &schema, &table, &perturbed_y, TaskKind::Regression)
                .unwrap();
        for row in [validation_fold, validation_fold + 5] {
            assert_eq!(
                block.values[row][0].to_bits(),
                perturbed.values[row][0].to_bits()
            );
        }
    }
}

#[test]
fn wide_raw_input_is_not_truncated() {
    let columns: Vec<ColumnSpec> = (0..1024)
        .map(|index| ColumnSpec::numeric(format!("c{index}")))
        .collect();
    let headers = columns.iter().map(|column| column.name.clone()).collect();
    let table = RawTable::new(headers, vec![vec!["1".into(); 1024]; 4]).unwrap();
    let schema = DataSchema {
        columns,
        row_semantics: RowSemantics::Iid,
    };
    let y = [0.0, 1.0, 0.0, 1.0];
    let (input, train) = FittedInput::fit_transform(
        InputConfig::compatibility(),
        &schema,
        &table,
        &y,
        TaskKind::Binary,
    )
    .unwrap();
    assert_eq!(train.values[0].len(), 1024);
    assert_eq!(input.transform(&table).unwrap().values[0].len(), 1024);
}

#[test]
fn logit_table_model_emits_finite_probabilities_and_handles_unknowns() {
    let task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let genome = GenomeIr::parse(
            "(pipe X (input numeric=median missing=indicator categorical=hybrid max_onehot=16 min_count=2 prior=10 folds=5) (logit_ridge class_weight=balanced l2=0.1))",
        )
        .unwrap();
    let y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
    let test =
        RawTable::from_csv("amount,country\n0.05,\"US\"\n1.3,\"unknown, place\"\n,DE\n").unwrap();
    let predictions = model.predict_table(&test).unwrap();
    assert_eq!(predictions.len(), 3);
    assert!(predictions.iter().all(|value| (0.0..=1.0).contains(value)));
    assert!(predictions[1] > predictions[0]);
}

#[test]
fn bound_model_recomputes_vocabulary_identity_before_use() {
    let task = TaskSpec::regression(Objective::Rmse).unwrap();
    let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let y = [0.0, 0.5, 0.25, 1.0, 0.75, 1.5, 1.25, 2.0];
    let mut model = fit_table(&genome, &task, &schema(), &training_table(), &y).unwrap();
    model.vocabulary_sha256 = "00".repeat(32);
    let error = model.predict_table(&training_table()).unwrap_err();
    assert!(error.to_string().contains("vocabulary hash mismatch"));
    let error = emit_bound(&model, &crate::transpile::EmitOptions::default()).unwrap_err();
    assert!(error.to_string().contains("vocabulary hash mismatch"));
}
