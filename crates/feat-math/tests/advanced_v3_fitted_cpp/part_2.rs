use super::*;

#[test]
fn iid_live_binary_head_and_calibration_corpus_matches_bits() {
    let task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let test = RawTable::from_csv("a,b\n0.125,-1.5\n,2.5\n2.125,6.5\n").unwrap();
    let cases = [
        "(pipe X (input numeric=median missing=indicator) (logit_ridge class_weight=balanced l2=0.1 calibration=platt threshold=f1))",
        "(pipe X (input numeric=median missing=indicator) (tab_boost rounds=8 bins=8 min_leaf=1))",
        "(pipe X (input numeric=median missing=indicator) (extra_trees trees=8 depth=3 min_leaf=1))",
        "(pipe X (input numeric=median missing=indicator) (gam_boost rounds=8 bins=4 calibration=isotonic threshold=youden))",
        "(pipe X (input numeric=median missing=indicator) (portfolio))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-binary-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for (index, source) in cases.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_table(&genome, &task, &numeric_schema(), &training_table(), &y)
            .unwrap_or_else(|error| panic!("case {index} fit: {error}"));
        let artifact =
            emit_fitted_cpp17(&model).unwrap_or_else(|error| panic!("case {index} emit: {error}"));
        let expected = model
            .predict_table(&test)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_and_run(&artifact.source, directory, false),
            expected,
            "case {index}: {source}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn iid_categorical_codec_modes_match_bits() {
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("amount"),
            ColumnSpec::categorical("kind"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    let train = RawTable::from_csv(
        "amount,kind\n0,A\n0.25,A\n0.5,B\n0.75,B\n1,C\n1.25,C\n1.5,A\n1.75,B\n2,C\n2.25,A\n",
    )
    .unwrap();
    let test = RawTable::from_csv("amount,kind\n0.125,A\n,B\n2.125,unknown\n").unwrap();
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let values = [0.125, 0.0, 0.0, 1.0, 2.125, -1.0];
    let missing = [0, 0, 1, 0, 0, 0];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-categorical-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for mode in ["native", "frequency", "onehot", "ordered", "hybrid"] {
        let genome = GenomeIr::parse(&format!(
            "(pipe X (input numeric=median missing=indicator categorical={mode} max_onehot=16 min_count=1 prior=10 folds=5) (ridge))"
        ))
        .unwrap();
        let model = fit_table(&genome, &task(), &schema, &train, &y).unwrap();
        let artifact = emit_fitted_cpp17(&model).unwrap();
        assert!(artifact
            .codec
            .contains("\"vocabulary\":[\"A\",\"B\",\"C\"]"));
        let expected = model
            .predict_table(&test)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_typed(&artifact.source, directory, false, &values, &missing, 3, 2),
            expected,
            "categorical mode {mode}"
        );
    }
    scratch.close().unwrap();
}

#[test]
fn ordered_historical_categorical_state_matches_bits() {
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::categorical("kind"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".to_string()],
            order_by: vec!["time".to_string()],
        },
    };
    let train = RawTable::from_csv(
        "group,time,kind\n1,1,A\n1,2,A\n1,3,B\n1,4,A\n1,5,B\n2,1,A\n2,2,B\n2,3,B\n2,4,A\n2,5,B\n",
    )
    .unwrap();
    let test = RawTable::from_csv("group,time,kind\n1,6,A\n2,6,B\n3,1,A\n1,7,unknown\n").unwrap();
    let y = [0.1, 0.2, 0.8, 0.4, 0.9, 0.3, 0.7, 0.6, 0.5, 1.0];
    let genome = GenomeIr::parse(
        "(pipe X (input numeric=median missing=indicator categorical=ordered prior=2) (ridge))",
    )
    .unwrap();
    let model = fit_table(&genome, &task(), &schema, &train, &y).unwrap();
    let artifact = emit_fitted_cpp17(&model).unwrap();
    let expected = model
        .predict_table(&test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let values = [1.0, 6.0, 0.0, 2.0, 6.0, 1.0, 3.0, 1.0, 0.0, 1.0, 7.0, -1.0];
    let missing = [0; 12];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-historical-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    assert_eq!(
        compile_typed(&artifact.source, directory, false, &values, &missing, 4, 3,),
        expected
    );
    scratch.close().unwrap();
}

#[test]
fn table_aware_lag_and_roll_match_bits_with_fitted_tails() {
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::numeric("value"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".to_string()],
            order_by: vec!["time".to_string()],
        },
    };
    let train = RawTable::from_csv(
        "group,time,value\nA,1,1\nA,2,2\nA,3,3\nA,4,4\nA,5,5\nB,1,10\nB,2,11\nB,3,12\nB,4,13\nB,5,14\n",
    )
    .unwrap();
    let test = RawTable::from_csv("group,time,value\nA,7,7\nA,6,6\nB,6,15\nC,1,20\n").unwrap();
    let y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let cases = [
        "(pipe X (input numeric=median categorical=native) (lag k=2 append=true) (ridge))",
        "(pipe X (input numeric=median categorical=native) (roll k=3 append=true) (ridge))",
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-table-ops-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for source in cases {
        let model = fit_table(
            &GenomeIr::parse(source).unwrap(),
            &task(),
            &schema,
            &train,
            &y,
        )
        .unwrap();
        let artifact = emit_fitted_cpp17(&model).unwrap();
        let encoded = EncodedTable::from_bound_model(&model, &test).unwrap();
        let expected = model
            .predict_table(&test)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            compile_typed(
                &artifact.source,
                directory,
                false,
                &encoded.values,
                &encoded.missing,
                encoded.rows,
                encoded.columns,
            ),
            expected,
            "{source}"
        );
    }
    scratch.close().unwrap();
}
