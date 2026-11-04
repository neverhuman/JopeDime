use super::*;

/// The first stage of a training pass.
fn first_stage(source: &str, x: &Rows, y: &[f64]) -> (Model, StageFaultCounts) {
    let program = parse_program(source);
    let (model, telemetry) = fit_traced(&program, x, y).expect("fits");
    assert_eq!(
        serialized(&model),
        serialized(&fit(&program, x, y).expect("fits")),
        "{source}"
    );
    (model, telemetry.stages[0].clone())
}

fn only_class(counts: &StageFaultCounts, class: FaultClass, n: u64) {
    let mut expected = [0; FAULT_SLOTS];
    expected[class.index()] = n;
    assert_eq!(counts.faulted_cells, expected, "only {class}");
}

#[test]
fn log_of_a_negative_column_counts_log_domain_and_of_a_positive_one_nothing() {
    let (x, y) = toy_data(64, 5, 42);
    let source = "(pipe X (derive expr=\"log(col(0))\") (ridge))";
    let (_, stage) = first_stage(source, &x, &y);
    let outside = x.iter().filter(|row| row[0] <= 1e-300).count() as u64;
    assert!(outside > 0, "toy_data has non-positive values in column 0");
    assert_eq!(stage.faulted(FaultClass::LogDomain), outside);
    only_class(&stage, FaultClass::LogDomain, outside);
    assert_eq!(stage.evaluated_cells, 64);
    assert_eq!(stage.hard_rows, 0);
    assert!(!stage.compile_failed && !stage.constant_output);

    let positive: Rows = x
        .iter()
        .map(|row| row.iter().map(|v| v.abs() + 1.0).collect())
        .collect();
    let (_, stage) = first_stage(source, &positive, &y);
    assert_eq!(stage.faulted(FaultClass::LogDomain), 0);
    assert_eq!(stage.faulted_cells, [0; FAULT_SLOTS]);
    assert_eq!(stage.evaluated_cells, 64);
}

#[test]
fn a_column_past_the_width_is_a_hard_row_and_the_legacy_zeros() {
    let (x, y) = toy_data(40, 5, 42);
    let x: Rows = x.iter().map(|row| row[..4].to_vec()).collect();
    let (model, stage) = first_stage("(pipe X (map expr=\"col(9)\") (ridge))", &x, &y);
    assert_eq!(stage.hard_rows, 40);
    assert_eq!(stage.evaluated_cells, 160);
    only_class(&stage, FaultClass::ColumnOob, 160);
    assert!(stage.constant_output && !stage.compile_failed);
    let Model::Pipeline { ops, .. } = &model else {
        panic!("a pipeline")
    };
    let (rows, _) = apply_op_traced(&ops[0], &x);
    assert_eq!(row_bits(&rows), row_bits(&apply_op(&ops[0], &x).unwrap()));
    assert!(rows.iter().flatten().all(|value| value.to_bits() == 0));
    assert_eq!(rows.len(), 40);
    assert!(rows.iter().all(|row| row.len() == 4));
}

#[test]
fn a_step_above_the_data_is_a_constant_output() {
    let (x, y) = toy_data(64, 5, 42);
    let (_, stage) = first_stage(
        "(pipe X (derive expr=\"step(col(0), 100)\") (ridge))",
        &x,
        &y,
    );
    assert!(stage.constant_output);
    assert_eq!(stage.faulted_cells, [0; FAULT_SLOTS]);
    assert_eq!(stage.hard_rows, 0);
    let (_, stage) = first_stage("(pipe X (derive expr=\"step(col(0), 0)\") (ridge))", &x, &y);
    assert!(!stage.constant_output);
    // Constant means bit-identical: +0.0 and -0.0 differ.
    let derive = fit_op(&parse_stage("(derive expr=\"c0\")"), &x, &y).expect("fits");
    let zeros: Rows = vec![vec![0.0], vec![-0.0]];
    assert!(!check_op("c0 on +0.0 and -0.0", &derive, &zeros).constant_output);
    let zeros: Rows = vec![vec![-0.0], vec![-0.0]];
    assert!(check_op("c0 on -0.0 twice", &derive, &zeros).constant_output);
}

#[test]
fn a_stage_that_emits_a_non_finite_value_counts_the_epilogue() {
    let (mut x, y) = toy_data(32, 5, 42);
    // symlog's asinh overflows to ±inf above about 1.34e154; the epilogue replaces it with 0.0.
    x[3][1] = 1e200;
    x[7][4] = -1e200;
    let (model, stage) = first_stage("(pipe X (symlog c=1) (ridge))", &x, &y);
    assert_eq!(stage.epilogue_sanitized, 2);
    assert_eq!(stage.evaluated_cells, 0);
    assert_eq!(stage.faulted_cells, [0; FAULT_SLOTS]);
    let (_, pass) = predict_traced(&model, &x).expect("no zero-width row");
    assert_eq!(pass.stages[0].epilogue_sanitized, 2);
    // A merged input cell counts too: an expression stage on raw rows, with append.
    let op = fit_op(&parse_stage("(map expr=\"x\" append=true)"), &x, &y).expect("fits");
    let raw = vec![vec![f64::NAN, 1.0]];
    let counts = check_op("map x append on a raw NaN", &op, &raw);
    assert_eq!(counts.epilogue_sanitized, 1);
    assert_eq!(counts.faulted(FaultClass::NonfiniteInput), 1);
    assert_eq!(counts.faulted(FaultClass::NonfiniteSanitized), 1);
}

#[test]
fn a_structurally_invalid_expression_is_a_compile_failure() {
    // parse and from_ast refuse this arity; serde decoding, a compatibility interface, does not.
    let bad: MathExpr = serde_json::from_value(serde_json::json!({
        "root": {"Call": ["Sin", [{"Const": 1.0}, {"Const": 2.0}]]}
    }))
    .expect("the historical AST encoding");
    assert!(bad.compile().is_err());
    let rows: Rows = vec![vec![1.0, 2.0, 3.0], vec![], vec![-1.0]];
    for append in [false, true] {
        let map = FittedOp::MapExpr {
            source: "sin(1, 2)".to_string(),
            expr: bad.clone(),
            append,
        };
        let (out, counts) = apply_op_traced(&map, &rows);
        assert_eq!(row_bits(&out), row_bits(&apply_op(&map, &rows).unwrap()));
        assert!(counts.compile_failed && counts.constant_output);
        assert_eq!(counts.evaluated_cells, 4);
        only_class(&counts, FaultClass::CompileFailed, 4);
        // The empty row generates no cell, so it is not a hard row.
        assert_eq!(counts.hard_rows, 2);
        assert_eq!(counts.epilogue_sanitized, 0);

        let derive = FittedOp::DeriveExpr {
            source: "sin(1, 2)".to_string(),
            expr: bad.clone(),
            append,
        };
        let (out, counts) = apply_op_traced(&derive, &rows);
        assert_eq!(row_bits(&out), row_bits(&apply_op(&derive, &rows).unwrap()));
        assert!(counts.compile_failed && counts.constant_output);
        assert_eq!(counts.evaluated_cells, 3);
        only_class(&counts, FaultClass::CompileFailed, 3);
        assert_eq!(counts.hard_rows, 3);
    }
    // A symbolic program does not compile: fit and predict call the interpreter, whose arity check
    // raises COMPILE_FAILED on each row. The stage's compile flag stays false.
    let (x, y) = toy_data(8, 5, 42);
    let program = GenomeIr::Symbolic {
        source: "sin(1, 2)".to_string(),
        expr: bad,
        affine: true,
    };
    let model = fit(&program, &x, &y).expect("fits");
    let (traced, telemetry) = fit_traced(&program, &x, &y).expect("fits");
    assert_eq!(serialized(&traced), serialized(&model));
    let stage = &telemetry.stages[0];
    assert!(!stage.compile_failed && stage.constant_output);
    assert_eq!(stage.hard_rows, 8);
    only_class(stage, FaultClass::CompileFailed, 8);
    let (predictions, pass) = predict_traced(&model, &x).expect("no zero-width row");
    assert_eq!(bits(&predictions), bits(&model.predict(&x).unwrap()));
    assert_eq!(pass, telemetry);
}

#[test]
fn the_symbolic_path_reads_raw_inputs_and_a_pipeline_sanitizes_them() {
    let (mut x, y) = toy_data(32, 5, 42);
    x[2][0] = f64::NAN;
    x[9][0] = f64::INFINITY;
    let (model, stage) = first_stage("(sym raw \"c0 + c1\")", &x, &y);
    assert_eq!(stage.evaluated_cells, 32);
    assert_eq!(stage.faulted(FaultClass::NonfiniteInput), 2);
    assert_eq!(stage.faulted(FaultClass::NonfiniteSanitized), 2);
    assert_eq!(stage.hard_rows, 0);
    assert_eq!(stage.epilogue_sanitized, 0);
    let (predictions, pass) = predict_traced(&model, &x).expect("no zero-width row");
    assert_eq!(bits(&predictions), bits(&model.predict(&x).unwrap()));
    assert_eq!(pass.input_sanitized, 0);
    assert_eq!(pass.stages[0], stage);

    // Through a pipeline, the entry sanitize replaces both cells first: the stage reads zeros.
    let program = parse_program("(pipe X (derive expr=\"c0 + c1\") (ridge))");
    let (_, telemetry) = fit_traced(&program, &x, &y).expect("fits");
    assert_eq!(telemetry.input_sanitized, 2);
    assert_eq!(telemetry.stages[0].faulted_cells, [0; FAULT_SLOTS]);

    // `x` outside a map: every row is a Hard fallback, and the affine fit takes its flat branch.
    let (model, stage) = first_stage("(sym affine \"x\")", &x, &y);
    assert_eq!(stage.hard_rows, 32);
    only_class(&stage, FaultClass::IllegalScope, 32);
    assert!(stage.constant_output);
    let Model::Symbolic { a, .. } = model else {
        panic!("a symbolic model")
    };
    assert_eq!(a.to_bits(), 0.0f64.to_bits());
}

#[test]
fn an_explicit_portfolio_is_fitted_by_fit_and_reports_no_stage() {
    let (mut train, target, mut test) = fitted_surface_data();
    // portfolio.rs replaces non-finite cells with 0.0 itself, in code that is not traced, so the
    // telemetry stays empty (input_sanitized 0) whatever the input holds.
    train[5][1] = f64::NAN;
    train[17][3] = f64::INFINITY;
    test[2][0] = f64::NEG_INFINITY;
    let program = parse_program(
        "(portfolio (member (pipe X (ridge))) (member (pipe X (standardize) (ridge))) \
         (blend kind=nnls folds=5 repeats=1 guard=clip))",
    );
    let model = fit(&program, &train, &target).expect("fits");
    let Model::Portfolio(portfolio) = &model else {
        panic!("an explicit portfolio model")
    };
    assert!(!portfolio.mean_only, "the predictions depend on the rows");
    let (traced, telemetry) = fit_traced(&program, &train, &target).expect("fits");
    assert_eq!(telemetry, PipelineTelemetry::default());
    // Serialized portfolios carry wall-clock fit times, so only the predictions are compared.
    let expected = bits(&model.predict(&test).unwrap());
    for m in [&model, &traced] {
        let (predictions, pass) = predict_traced(m, &test).expect("no zero-width row");
        assert_eq!(bits(&predictions), expected);
        assert_eq!(pass, PipelineTelemetry::default());
    }
    // The replacement is 0.0: zeros in place of the non-finite cells fit and predict the same
    // bits, and another value in place of one of them does not.
    let zeroed = |rows: &Rows| {
        let mut rows = rows.clone();
        sanitize(&mut rows);
        rows
    };
    let clean = fit(&program, &zeroed(&train), &target).expect("fits");
    assert_eq!(bits(&clean.predict(&zeroed(&test)).unwrap()), expected);
    assert_eq!(bits(&model.predict(&zeroed(&test)).unwrap()), expected);
    let mut moved_test = zeroed(&test);
    moved_test[2][0] = 1.0;
    assert_ne!(bits(&model.predict(&moved_test).unwrap()), expected);
    let mut moved_train = zeroed(&train);
    moved_train[5][1] = 1.0;
    let other = fit(&program, &moved_train, &target).expect("fits");
    assert_ne!(bits(&other.predict(&zeroed(&test)).unwrap()), expected);
}

#[test]
fn stage_level_empty_row_identities() {
    // WP-1.8 at stage level: the empty sum, l1 and l2 are identities; mean, std, min and max of
    // an empty row raise EMPTY_ROW_REDUCTION (min and max also reach the output filter).
    let empty: Rows = vec![Vec::new()];
    let cases = [
        ("row_sum", FaultSet::EMPTY),
        ("row_l1", FaultSet::EMPTY),
        ("row_l2", FaultSet::EMPTY),
        ("row_mean", FaultSet::EMPTY_ROW_REDUCTION),
        ("row_std", FaultSet::EMPTY_ROW_REDUCTION),
        (
            "row_min",
            FaultSet::EMPTY_ROW_REDUCTION.union(FaultSet::NONFINITE_SANITIZED),
        ),
        (
            "row_max",
            FaultSet::EMPTY_ROW_REDUCTION.union(FaultSet::NONFINITE_SANITIZED),
        ),
    ];
    for (source, faults) in cases {
        let expr = MathExpr::parse(source).expect("parses");
        let derive = FittedOp::DeriveExpr {
            source: source.to_string(),
            expr: expr.clone(),
            append: false,
        };
        let counts = check_op(source, &derive, &empty);
        assert_eq!(counts.evaluated_cells, 1, "{source}");
        let mut expected = [0; FAULT_SLOTS];
        for class in faults.iter() {
            expected[class.index()] = 1;
        }
        assert_eq!(counts.faulted_cells, expected, "{source}");
        assert_eq!(counts.hard_rows, 0, "{source}");
        // A map over an empty row generates nothing; the epilogue pads the row with one 0.0.
        let map = FittedOp::MapExpr {
            source: source.to_string(),
            expr,
            append: false,
        };
        let (rows, counts) = apply_op_traced(&map, &empty);
        assert_eq!(row_bits(&rows), vec![vec![0]]);
        assert_eq!(counts.evaluated_cells, 0);
        assert!(!counts.constant_output);
        assert_eq!(counts.epilogue_sanitized, 0);
    }
}
