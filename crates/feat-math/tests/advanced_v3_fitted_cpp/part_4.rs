use super::part_3::FITTED_WIDTH_STATUS;
use super::*;

#[test]
fn fitted_state_that_does_not_fit_the_row_answers_status_8() {
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    // Column b of the first row is exactly zero, a cell that `impute` fills.
    let test = RawTable::from_csv("a,b\n0.125,0\n0.5,2.5\n2.125,6.5\n").unwrap();
    type Edit = fn(&mut serde_json::Value);
    let cases: [(&str, &str, Edit); 12] = [
        (
            "select_corr",
            "(pipe X (select_corr k=2) (ridge))",
            |json| {
                stage(json, 0)["Legacy"]["SelectCorr"]["indices"] = serde_json::json!([0, 99]);
            },
        ),
        // The row is narrower than the state: a fill and an indicator for six columns.
        (
            "impute_narrower",
            "(pipe X (impute add_indicator=true) (ridge))",
            |json| {
                let impute = &mut stage(json, 0)["Legacy"]["Impute"];
                impute["fill"]
                    .as_array_mut()
                    .unwrap()
                    .extend(std::iter::repeat_n(serde_json::json!(0.5), 4));
                impute["indicator_cols"] = serde_json::json!([5]);
            },
        ),
        // The row is wider than the state, with a zero cell past the fill: that cell reads it.
        ("impute_wider", "(pipe X (impute) (ridge))", |json| {
            stage(json, 0)["Legacy"]["Impute"]["fill"]
                .as_array_mut()
                .unwrap()
                .truncate(1);
        }),
        // One guard short: revision 2 read past the guards.
        (
            "pair_ratio",
            "(pipe X (pair_ratio topk=2) (ridge))",
            |json| {
                let eps = stage(json, 0)["Legacy"]["PairRatio"]["eps"]
                    .as_array_mut()
                    .unwrap();
                assert!(!eps.is_empty(), "pair_ratio fitted a pair");
                eps.pop();
            },
        ),
        ("pair_prod", "(pipe X (pair_prod topk=2) (ridge))", |json| {
            stage(json, 0)["Legacy"]["PairProd"]["pairs"] = serde_json::json!([[0, 99]]);
        }),
        (
            "poly_interact",
            "(pipe X (poly_interact) (ridge))",
            |json| {
                stage(json, 0)["Legacy"]["PolyInteract"]["pairs"] = serde_json::json!([[99, 1]]);
            },
        ),
        (
            "corr_screen",
            "(pipe X (corr_screen k=32) (ridge))",
            |json| {
                stage(json, 0)["CorrScreen"]["indices"] = serde_json::json!([0, 7]);
            },
        ),
        (
            "rff_cos_phase",
            "(pipe X (rff_cos n=5 seed=11) (ridge))",
            |json| {
                stage(json, 0)["Legacy"]["RffCos"]["b"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            },
        ),
        (
            "rff_sin_phase",
            "(pipe X (rff_sin n=5 seed=11) (ridge))",
            |json| {
                stage(json, 0)["Legacy"]["RffSin"]["b"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            },
        ),
        (
            "oblique_threshold",
            "(pipe X (oblique_hinge n=5 seed=11) (ridge))",
            |json| {
                stage(json, 0)["Legacy"]["ObliqueHinge"]["t"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            },
        ),
        (
            "empty_bin_encoding",
            "(pipe X (target_bin_encode bins=5) (ridge))",
            |json| {
                stage(json, 0)["Legacy"]["TargetBinEncode"]["enc"][0] = serde_json::json!([]);
            },
        ),
        (
            "empty_spline_knots",
            "(pipe X (spline knots=4) (ridge))",
            |json| {
                stage(json, 0)["Legacy"]["Spline"]["knots"][0] = serde_json::json!([]);
            },
        ),
    ];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-width-status-")
        .tempdir()
        .unwrap();
    let directory = scratch.path();
    let mut checked = Vec::new();
    let mut check = |name: &str, model: &BoundModel, test: &RawTable| {
        let encoded = EncodedTable::from_bound_model(model, test).unwrap();
        let artifact = emit_fitted_cpp17(model).unwrap();
        let (status, _) = compile_status(
            &artifact.source,
            directory,
            name,
            true,
            &encoded.values,
            &encoded.missing,
            encoded.rows,
            encoded.columns,
        );
        assert_eq!(status, FITTED_WIDTH_STATUS, "{name}");
        checked.push(name.to_string());
    };
    for (name, source, edit) in cases {
        let model = fit_table(
            &GenomeIr::parse(source).unwrap(),
            &task(),
            &numeric_schema(),
            &training_table(),
            &y,
        )
        .unwrap_or_else(|error| panic!("{name} fit: {error}"));
        assert!(
            model.predict_table(&test).is_ok(),
            "{name}: the fitted model"
        );
        let model = tampered(&model, edit);
        // The runtime refuses the edited state with an error, and the C++ answers status 8.
        assert!(model.predict_table(&test).is_err(), "{name}: the runtime");
        check(name, &model, &test);
    }

    // `roll` on an empty row, which only a hand edit builds: `select_corr` keeps no column and a
    // legacy `roll` follows it. The runtime refuses it; revision 2 divided by zero there.
    let roll = legacy_stage("(pipe X (roll k=1) (ridge))", &training_rows(), &y);
    let model = fit_table(
        &GenomeIr::parse("(pipe X (select_corr k=1) (ridge))").unwrap(),
        &task(),
        &numeric_schema(),
        &training_table(),
        &y,
    )
    .unwrap();
    let model = tampered(&model, |json| {
        stage(json, 0)["Legacy"]["SelectCorr"]["indices"] = serde_json::json!([]);
        json["predictor"]["Pipeline"]["ops"]
            .as_array_mut()
            .unwrap()
            .insert(1, serde_json::json!({ "Legacy": roll }));
    });
    assert!(model.predict_table(&test).is_err(), "roll: the runtime");
    check("roll_empty_row", &model, &test);

    // The table ABI, of a table-aware pipeline, answers the same status.
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
    let ordered_test =
        RawTable::from_csv("group,time,value\nA,7,7\nA,6,6\nB,6,15\nC,1,20\n").unwrap();
    let model = fit_table(
        &GenomeIr::parse(
            "(pipe X (input numeric=median categorical=native) (lag k=2 append=true) (select_corr k=2) (ridge))",
        )
        .unwrap(),
        &task(),
        &schema,
        &train,
        &y,
    )
    .unwrap();
    assert!(
        model.predict_table(&ordered_test).is_ok(),
        "table: the fitted model"
    );
    let model = tampered(&model, |json| {
        stage(json, 1)["Legacy"]["SelectCorr"]["indices"] = serde_json::json!([0, 99]);
    });
    assert!(
        model.predict_table(&ordered_test).is_err(),
        "table: the runtime"
    );
    check("table_select_corr", &model, &ordered_test);

    // A legacy predictor, the path compat_freeze pins, answers the same status. Its runtime,
    // `Model::predict`, gives no prediction there either: it panics.
    let rows = training_rows();
    let model = fit_legacy_numeric_bound(
        &GenomeIr::parse("(pipe X (select_corr k=2) (ridge))").unwrap(),
        &rows,
        &y,
    )
    .unwrap();
    let model = tampered(&model, |json| {
        json["predictor"]["Legacy"]["Pipeline"]["ops"][0]["SelectCorr"]["indices"] =
            serde_json::json!([0, 99]);
    });
    let legacy_test = legacy_table(&[vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]]);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.predict_table(&legacy_test)
    }));
    assert!(!matches!(outcome, Ok(Ok(_))), "legacy: the runtime");
    check("legacy_select_corr", &model, &legacy_test);

    assert_eq!(checked.len(), 15, "{checked:?}");
    scratch.close().unwrap();
}
