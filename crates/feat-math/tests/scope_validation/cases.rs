use super::*;

// ---------------------------------------------------------------------------------------------
// The workplan's cases
// ---------------------------------------------------------------------------------------------

#[test]
fn the_workplan_cases_have_their_exact_results() {
    // `x` is the element a map reads.
    accepts(r#"(pipe X (map expr="x") (ridge))"#, 3);
    // A derived column reads a whole row, where there is no element.
    for width in [1, 3, 8, 512] {
        rejects(
            r#"(pipe X (derive expr="x") (ridge))"#,
            width,
            Some(0),
            ILLEGAL_SCOPE,
            "variable x is only valid in map expressions",
            "stage 0: ILLEGAL_SCOPE: variable x is only valid in map expressions",
        );
    }
    accepts(r#"(pipe X (derive expr="col(0)") (ridge))"#, 1);
    accepts(r#"(pipe X (map expr="col(0)") (ridge))"#, 1);
    rejects(
        r#"(pipe X (derive expr="col(7)") (ridge))"#,
        1,
        Some(0),
        COLUMN_OOB,
        "column index 7 out of bounds for width 1",
        "stage 0: COLUMN_OOB: column index 7 out of bounds for width 1",
    );
    // At input width 8, `select_corr k=2` leaves a lower width of 2.
    rejects(
        r#"(pipe X (select_corr k=2) (derive expr="col(3)") (ridge))"#,
        8,
        Some(1),
        COLUMN_OOB,
        "column index 3 out of bounds for width 2",
        "stage 1: COLUMN_OOB: column index 3 out of bounds for width 2",
    );
    accepts(
        r#"(pipe X (select_corr k=2) (derive expr="col(1)") (ridge))"#,
        8,
    );
    // `row_stats` is ten columns, whatever the input width.
    for width in [1, 3, 20] {
        accepts(
            r#"(pipe X (row_stats) (derive expr="col(9)") (ridge))"#,
            width,
        );
        rejects(
            r#"(pipe X (row_stats) (derive expr="col(10)") (ridge))"#,
            width,
            Some(1),
            COLUMN_OOB,
            "column index 10 out of bounds for width 10",
            "stage 1: COLUMN_OOB: column index 10 out of bounds for width 10",
        );
    }
}

#[test]
fn the_cases_rest_on_the_registry_and_the_runtime() {
    // `row_stats` is `ShapeRule::Fixed(10)` in the registry, and the runtime emits ten values.
    let row_stats = feat_math::registry::operator("row_stats").expect("row_stats is registered");
    assert_eq!(row_stats.shape, feat_math::registry::ShapeRule::Fixed(10));
    let (x, y) = data(6, 3);
    let op = fit_op(&stage("(row_stats)"), &x, &y).expect("row_stats fits");
    assert!(apply_op(&op, &x).unwrap().iter().all(|row| row.len() == 10));
    // `select_corr k=2` keeps two of eight columns.
    let (x, y) = data(12, 8);
    let op = fit_op(&stage("(select_corr k=2)"), &x, &y).expect("select_corr fits");
    assert!(apply_op(&op, &x).unwrap().iter().all(|row| row.len() == 2));
    // The runtime meets an out-of-range column or a misplaced `x` only when it evaluates; the
    // stage then substitutes zero, and the trace carries the class the static check reports.
    let column = MathExpr::parse("col(7)").expect("col(7) parses");
    assert!(column.eval_row(&[0.5]).is_err());
    assert_eq!(column.eval_row_traced(&[0.5]), (0.0, FaultSet::COLUMN_OOB));
    assert_eq!(column.eval_row_traced(&[0.5; 8]).1, FaultSet::EMPTY);
    let element = MathExpr::parse("x").expect("x parses");
    assert!(element.eval_row(&[0.5]).is_err());
    assert_eq!(
        element.eval_row_traced(&[0.5]),
        (0.0, FaultSet::ILLEGAL_SCOPE)
    );
    assert_eq!(
        element.eval_element_traced(0.5, &[0.5]),
        (0.5, FaultSet::EMPTY)
    );
}

// ---------------------------------------------------------------------------------------------
// Where an error points, and its class
// ---------------------------------------------------------------------------------------------

#[test]
fn the_error_names_the_stage_the_member_and_the_class() {
    fn is_error<E: std::error::Error + Clone + Eq>() {}
    is_error::<ScopeError>();
    // `x` is reported before any column, as `validate_columns` checks it first.
    rejects(
        r#"(pipe X (derive expr="x + c9") (ridge))"#,
        2,
        Some(0),
        ILLEGAL_SCOPE,
        "variable x is only valid in map expressions",
        "stage 0: ILLEGAL_SCOPE: variable x is only valid in map expressions",
    );
    rejects(
        r#"(pipe X (map expr="x + c9") (ridge))"#,
        2,
        Some(0),
        COLUMN_OOB,
        "column index 9 out of bounds for width 2",
        "stage 0: COLUMN_OOB: column index 9 out of bounds for width 2",
    );
    // The first column out of range in reading order, not the largest.
    rejects(
        r#"(pipe X (derive expr="col(9) + col(12)") (ridge))"#,
        8,
        Some(0),
        COLUMN_OOB,
        "column index 9 out of bounds for width 8",
        "stage 0: COLUMN_OOB: column index 9 out of bounds for width 8",
    );
    rejects(
        r#"(pipe X (derive expr="col(12) + col(9)") (ridge))"#,
        8,
        Some(0),
        COLUMN_OOB,
        "column index 12 out of bounds for width 8",
        "stage 0: COLUMN_OOB: column index 12 out of bounds for width 8",
    );
    // A target stage counts in the index.
    rejects(
        r#"(pipe X (target kind=log1p) (standardize) (derive expr="c5") (ridge))"#,
        3,
        Some(2),
        COLUMN_OOB,
        "column index 5 out of bounds for width 3",
        "stage 2: COLUMN_OOB: column index 5 out of bounds for width 3",
    );
    // An expression reads the width that reaches its stage; an appending stage keeps its input.
    accepts(
        r#"(pipe X (derive append=true expr="c0") (map expr="c1 + 1") (ridge))"#,
        1,
    );
    rejects(
        r#"(pipe X (derive expr="c0") (map expr="c1 + 1") (ridge))"#,
        2,
        Some(1),
        COLUMN_OOB,
        "column index 1 out of bounds for width 1",
        "stage 1: COLUMN_OOB: column index 1 out of bounds for width 1",
    );
    // A closed-form program is one expression in row context, with no stage.
    rejects(
        r#"(sym raw "x")"#,
        4,
        None,
        ILLEGAL_SCOPE,
        "variable x is only valid in map expressions",
        "ILLEGAL_SCOPE: variable x is only valid in map expressions",
    );
    accepts(r#"(sym affine "c4 + row_mean")"#, 5);
    rejects(
        r#"(sym affine "c4 + row_mean")"#,
        4,
        None,
        COLUMN_OOB,
        "column index 4 out of bounds for width 4",
        "COLUMN_OOB: column index 4 out of bounds for width 4",
    );
    // An explicit portfolio names the member, and the stage inside it.
    let portfolio = |expr: &str| {
        format!(
            r#"(portfolio (member (pipe X (ridge))) (member (pipe X (row_stats) (derive expr="{expr}") (ridge))) (blend))"#
        )
    };
    accepts(&portfolio("c9"), 4);
    let Err(error) = check(&portfolio("c10"), 4) else {
        panic!("the portfolio is accepted");
    };
    assert_eq!(
        (error.member, error.stage, error.class),
        (Some(1), Some(1), COLUMN_OOB)
    );
    assert_eq!(
        error.to_string(),
        "member 1, stage 1: COLUMN_OOB: column index 10 out of bounds for width 10"
    );
    let Err(error) = check("(portfolio (member (pipe X (bogus_op))) (blend))", 4) else {
        panic!("the portfolio is accepted");
    };
    assert_eq!(
        (error.member, error.stage, error.class),
        (Some(0), None, None)
    );
    assert_eq!(error.to_string(), "member 0: unknown stage bogus_op");
}

#[test]
fn a_program_that_cannot_be_checked_fails_with_its_own_reason() {
    // Not a string: the printer leaves a numeric-looking expression unquoted, and it reads back
    // as a number.
    rejects(
        r#"(pipe X (derive expr=0.5) (ridge))"#,
        3,
        Some(0),
        None,
        "derive needs a string expression",
        "stage 0: derive needs a string expression",
    );
    rejects(
        "(pipe X (standardize) (map) (ridge))",
        3,
        Some(1),
        None,
        "map needs a string expression",
        "stage 1: map needs a string expression",
    );
    // An expression that does not parse fails with the parser's own message.
    let too_many_nodes = ["c0"; 40].join(" + ");
    for source in [
        "c0 +",
        "",
        "sin(c0",
        "bogus_fn(c0)",
        "c0 $ c1",
        "col(-1)",
        too_many_nodes.as_str(),
    ] {
        let reason = MathExpr::parse(source)
            .expect_err("the expression does not parse")
            .to_string();
        for op in ["map", "derive"] {
            let Err(error) = validate_scope(&stage_program(op, source), 3) else {
                panic!("{op} {source:?} is accepted");
            };
            assert_eq!(
                (
                    error.member,
                    error.stage,
                    error.class,
                    error.reason.as_str()
                ),
                (None, Some(1), None, reason.as_str()),
                "{op} {source:?}"
            );
        }
    }
    // A stage layout the pipeline slots refuse.
    rejects(
        "(pipe X (bogus_op) (ridge))",
        3,
        None,
        None,
        "unknown stage bogus_op",
        "unknown stage bogus_op",
    );
    rejects(
        "(pipe X (ridge) (standardize) (ridge))",
        3,
        None,
        None,
        "misplaced ridge stage at 0",
        "misplaced ridge stage at 0",
    );
    rejects(
        "(pipe X (standardize) (target kind=log1p) (ridge))",
        3,
        None,
        None,
        "misplaced target stage at 1",
        "misplaced target stage at 1",
    );
    // An input encoding stage needs a table schema.
    rejects(
        "(pipe X (target kind=log1p) (input) (ridge))",
        3,
        Some(1),
        None,
        "input encoding needs a table schema binding",
        "stage 1: input encoding needs a table schema binding",
    );
    // A stage whose width only the table-v3 path computes, even with nothing reading after it.
    rejects(
        "(pipe X (standardize) (pls_project) (ridge))",
        3,
        Some(1),
        None,
        "pls_project needs table-v3 width binding",
        "stage 1: pls_project needs table-v3 width binding",
    );
    // An input width out of bounds, for every program form.
    for width in [0, MAX_INPUT_WIDTH + 1, usize::MAX] {
        let reason = format!("input width {width} is outside 1..=16777216");
        for source in [
            "(pipe X (poly_interact) (ridge))",
            r#"(sym raw "c0")"#,
            "(portfolio (member (pipe X (poly_interact) (ridge))) (blend))",
        ] {
            rejects(source, width, None, None, &reason, &reason);
        }
    }
    // The widest accepted input keeps every width computation exact.
    accepts(
        r#"(pipe X (poly_interact) (bin_onehot bins=32) (derive expr="c511") (ridge))"#,
        MAX_INPUT_WIDTH,
    );
    accepts(
        r#"(pipe X (poly_interact append=true) (derive expr="c511") (ridge))"#,
        MAX_INPUT_WIDTH,
    );
}
