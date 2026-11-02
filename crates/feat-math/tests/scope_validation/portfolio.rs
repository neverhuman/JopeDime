use super::*;

// ---------------------------------------------------------------------------------------------
// Portfolio members
// ---------------------------------------------------------------------------------------------

/// Sixty rows of four columns, the first one constant when `constant_first`, and a target of three
/// times column 3 plus a tenth of column 1.
fn portfolio_data(constant_first: bool) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x: Vec<Vec<f64>> = (0..60u32)
        .map(|i| {
            let t = f64::from(i);
            let first = if constant_first {
                1.0
            } else {
                (t * 0.23).sin()
            };
            vec![first, (t * 0.37).sin(), (t * 0.11).cos(), t / 10.0]
        })
        .collect();
    let y = x.iter().map(|row| 3.0 * row[3] + 0.1 * row[1]).collect();
    (x, y)
}

/// The share of the target's variance that `predictions` explain.
fn r2(y: &[f64], predictions: &[f64]) -> f64 {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let total: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum();
    let residual: f64 = y
        .iter()
        .zip(predictions)
        .map(|(v, p)| (v - p) * (v - p))
        .sum();
    1.0 - residual / total
}

#[test]
fn a_portfolio_member_is_checked_against_the_columns_it_can_see() {
    let member = |expr: &str| {
        format!(r#"(portfolio (member (pipe X (derive expr="{expr}") (ridge))) (blend))"#)
    };
    // The portfolio drops the columns that are constant in training before its members see the
    // rows, so a member may see a single column at any input width: until one of its stages sets
    // a width, column 0 is the only column it can read.
    for width in [1, 4, 600] {
        accepts(&member("c0"), width);
        let Err(error) = check(&member("c3"), width) else {
            panic!("a member reading c3 is accepted at width {width}");
        };
        assert_eq!(
            (
                error.member,
                error.stage,
                error.class,
                error.reason.as_str()
            ),
            (
                Some(0),
                Some(0),
                COLUMN_OOB,
                "column index 3 out of bounds for width 1"
            ),
            "width {width}"
        );
        assert_eq!(
            error.to_string(),
            "member 0, stage 0: COLUMN_OOB: column index 3 out of bounds for width 1"
        );
    }
    // The same expression in a pipeline reads the input columns themselves.
    accepts(r#"(pipe X (derive expr="c3") (ridge))"#, 4);
    // A stage of fixed width, or an appended column, sets what the later stages may read.
    accepts(
        r#"(portfolio (member (pipe X (row_stats) (derive expr="c9") (ridge))) (blend))"#,
        4,
    );
    accepts(
        r#"(portfolio (member (pipe X (derive append=true expr="c0") (map expr="x + c1") (ridge))) (blend))"#,
        4,
    );
    // The error names the member that fails.
    let two = r#"(portfolio (member (pipe X (derive expr="c0") (ridge))) (member (pipe X (derive expr="c1") (ridge))) (blend))"#;
    let Err(error) = check(two, 4) else {
        panic!("{two} is accepted");
    };
    assert_eq!(
        error.to_string(),
        "member 1, stage 0: COLUMN_OOB: column index 1 out of bounds for width 1"
    );

    // Every member is checked as its pipeline is at one column, whatever the input width.
    let mut expressions = vec![("derive", "x".to_string())];
    for column in 0..=16 {
        expressions.push(("derive", format!("c{column}")));
        expressions.push(("map", format!("x + c{column}")));
    }
    let widths = [1, 2, 3, 4, 5, 8, 16, 600, MAX_INPUT_WIDTH];
    let (mut comparisons, mut rejected) = (0, 0);
    for template in WIDTH_STAGES {
        for append in [false, true] {
            let first = if append {
                format!("{} append=true)", &template[..template.len() - 1])
            } else {
                template.to_string()
            };
            for (op, expression) in &expressions {
                let pipe = format!(r#"(pipe X {first} ({op} expr="{expression}") (ridge))"#);
                let alone = check(&pipe, 1)
                    .map_err(|error| (Some(0), error.stage, error.class, error.reason));
                let portfolio = program(&format!("(portfolio (member {pipe}) (blend))"));
                for width in widths {
                    let within = validate_scope(&portfolio, width)
                        .map_err(|error| (error.member, error.stage, error.class, error.reason));
                    assert_eq!(within, alone, "{pipe} as a member at width {width}");
                    comparisons += 1;
                    rejected += usize::from(within.is_err());
                }
            }
        }
    }
    eprintln!(
        "AGREEMENT members stages={} expressions={} widths={widths:?} comparisons={comparisons} \
         rejected={rejected}",
        WIDTH_STAGES.len() * 2,
        expressions.len()
    );
    assert!(rejected > 0 && rejected < comparisons);

    // At runtime, with input column 0 constant in training, the portfolio keeps columns 1 to 3: a
    // member's rows are three wide, so its `c3` does not exist and its `c2` is input column 3.
    let fit = |source: &str, (x, y): &(Vec<Vec<f64>>, Vec<f64>)| {
        let model = feat_math::fit(&program(source), x, y)
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        let predictions = model.predict(x).unwrap();
        (model, predictions)
    };
    let constant_first = portfolio_data(true);
    let (model, predictions) = fit(&member("c3"), &constant_first);
    let feat_math::Model::Portfolio(fitted) = &model else {
        panic!("a portfolio program fits a portfolio model");
    };
    assert_eq!(fitted.keep, [1, 2, 3]);
    let row: Vec<f64> = fitted
        .keep
        .iter()
        .map(|&column| constant_first.0[5][column])
        .collect();
    assert_eq!(
        MathExpr::parse("c3")
            .expect("c3 parses")
            .eval_row_traced(&row),
        (0.0, FaultSet::COLUMN_OOB)
    );
    // The member reads nothing, so the portfolio predicts one value everywhere.
    assert!(
        predictions
            .iter()
            .all(|value| value.to_bits() == predictions[0].to_bits()),
        "{predictions:?}"
    );
    let y = &constant_first.1;
    assert!(r2(y, &fit(&member("c2"), &constant_first).1) > 0.99);
    let pipeline = r#"(pipe X (derive expr="c3") (ridge))"#;
    assert!(r2(y, &fit(pipeline, &constant_first).1) > 0.99);
    // With no constant column, the member's `c3` is input column 3.
    let varying = portfolio_data(false);
    assert!(r2(&varying.1, &fit(&member("c3"), &varying).1) > 0.99);
}
