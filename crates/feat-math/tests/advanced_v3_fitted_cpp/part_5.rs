use super::*;

#[test]
fn table_row_operations_refuse_mismatched_fitted_tail_width() {
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
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-tail-width-")
        .tempdir()
        .unwrap();
    for (name, source) in [
        (
            "roll_tail_width",
            "(pipe X (input numeric=median categorical=native) (pair_prod topk=2 append=true) (roll k=3 append=true) (ridge))",
        ),
        (
            "lag_tail_width",
            "(pipe X (input numeric=median categorical=native) (pair_prod topk=2 append=true) (lag k=2 append=true) (ridge))",
        ),
    ] {
        let model = fit_table(
            &GenomeIr::parse(source).unwrap(),
            &task(),
            &schema,
            &train,
            &y,
        )
        .unwrap_or_else(|error| panic!("{name} fit: {error}"));
        assert!(model.predict_table(&test).is_ok(), "{name}: fitted model");
        let model = tampered(&model, |json| {
            let pairs = stage(json, 0)["Legacy"]["PairProd"]["pairs"]
                .as_array_mut()
                .unwrap();
            assert!(!pairs.is_empty(), "{name}: fitted pair");
            let extra = pairs[0].clone();
            pairs.push(extra);
        });
        assert!(model.predict_table(&test).is_err(), "{name}: Rust width check");
        let encoded = EncodedTable::from_bound_model(&model, &test).unwrap();
        let artifact = emit_fitted_cpp17(&model).unwrap();
        let (status, bits) = compile_status(
            &artifact.source,
            scratch.path(),
            name,
            true,
            &encoded.values,
            &encoded.missing,
            encoded.rows,
            encoded.columns,
        );
        assert_eq!(status, FITTED_WIDTH_STATUS, "{name}");
        assert!(bits.is_empty(), "{name}");
    }
    scratch.close().unwrap();
}

#[test]
fn empty_stage_outputs_match_the_runtime_bits_on_the_legacy_paths() {
    // One column leaves `pair_ratio` no pair, so its output rows are empty (the compat program
    // `empty_stage_output`), and `ops::finish` pads each to [0.0]. Revision 2 kept them empty.
    let rows = training_rows();
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let test = [vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]];
    let raw = legacy_table(&test);
    let values = test.iter().flatten().copied().collect::<Vec<_>>();
    let missing = vec![0; values.len()];
    let bits = |values: Vec<f64>| values.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-empty-stage-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    for source in [
        "(pipe X (select_corr k=1) (pair_ratio topk=4) (ridge))",
        // Revision 2 divided by zero in `roll` here.
        "(pipe X (select_corr k=1) (pair_ratio topk=4) (roll k=1) (ridge))",
        // Revision 2 read the head's first column past the end of the empty row here.
        "(pipe X (select_corr k=1) (pair_ratio topk=4) (elasticnet alpha=0.01 l1_ratio=0.5 sweeps=5))",
    ] {
        let genome = GenomeIr::parse(source).unwrap();
        let model = fit_legacy_numeric_bound(&genome, &rows, &y)
            .unwrap_or_else(|error| panic!("{source} fit: {error}"));
        let expected = bits(model.predict_table(&raw).unwrap());
        assert_eq!(
            bits(fit(&genome, &rows, &y).unwrap().predict(&test.to_vec()).unwrap()),
            expected,
            "{source}: the bound model is the legacy model"
        );
        let artifact = emit_fitted_cpp17(&model).unwrap();
        assert_eq!(
            compile_typed(&artifact.source, directory, false, &values, &missing, 3, 2),
            expected,
            "{source}"
        );
    }

    // A portfolio member's stages, and the portfolio's projection, pad as well: member 0 sees the
    // two raw columns and empties them as above; member 1 sees the projection, which keeps no
    // column and so is padded to [0.0]. Revision 2 divided by zero in both members' `roll`.
    let one_column = rows.iter().map(|row| vec![row[0]]).collect::<Vec<_>>();
    let member = fit(
        &GenomeIr::parse("(pipe X (select_corr k=1) (pair_ratio topk=4) (roll k=1) (ridge))")
            .unwrap(),
        &rows,
        &y,
    )
    .unwrap();
    let projected = fit(
        &GenomeIr::parse("(pipe X (roll k=1) (ridge))").unwrap(),
        &one_column,
        &y,
    )
    .unwrap();
    let mut select_none = legacy_stage("(pipe X (select_corr k=1) (ridge))", &rows, &y);
    select_none["SelectCorr"]["indices"] = serde_json::json!([]);
    let portfolio = tampered(
        &bound_with_serialized_legacy_head("(pipe X (portfolio folds=5 repeats=1))", &y),
        |json| {
            let portfolio = &mut json["predictor"]["Pipeline"]["head"]["Legacy"]["Portfolio"];
            assert!(portfolio.is_object(), "{portfolio}");
            assert_eq!(
                portfolio["mean_only"],
                serde_json::json!(false),
                "{portfolio}"
            );
            assert_eq!(portfolio["keep"], serde_json::json!([0, 1]), "{portfolio}");
            portfolio["proj"] = select_none;
            portfolio["members"] = serde_json::json!([
                { "name": "empty_stage_member", "proj_view": false, "state": { "Fitted": member } },
                { "name": "projected_member", "proj_view": true, "state": { "Fitted": projected } },
            ]);
            portfolio["weights"] = serde_json::json!([0.5, 0.5]);
        },
    );
    let portfolio_test = RawTable::from_csv("a,b\n0.125,-1.5\n0,2.5\n2.125,6.5\n").unwrap();
    let encoded = EncodedTable::from_bound_model(&portfolio, &portfolio_test).unwrap();
    let artifact = emit_fitted_cpp17(&portfolio).unwrap();
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
        bits(portfolio.predict_table(&portfolio_test).unwrap()),
        "portfolio member and projection"
    );
    scratch.close().unwrap();
}
