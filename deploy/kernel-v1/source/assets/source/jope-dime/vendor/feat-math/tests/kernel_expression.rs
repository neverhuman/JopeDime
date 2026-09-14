use feat_math::math_expr::{ExpressionScope, MathExpr};

#[test]
fn one_hundred_thousand_expression_children_pass_scope_and_execution_parity() {
    let mut parent = MathExpr::parse("0.1").unwrap();
    let donor = MathExpr::parse("sin(c9)+c0*0.0000000000001").unwrap();
    let mut changed = 0;
    for seed in 0..100_000 {
        let columns = [0, 1, 2, 32, 512][(seed / 1000 % 5) as usize];
        let scope = ExpressionScope {
            columns,
            elementwise: seed % 2 == 0,
            max_nodes: 64,
            max_depth: 16,
        };
        if scope.validate(&parent).is_err() {
            parent = MathExpr::parse("0.1").unwrap();
        }
        let edit = if seed % 3 == 0 {
            parent.crossover_v1(&donor, seed, scope).unwrap()
        } else {
            parent.mutate_v1(seed, scope).unwrap()
        };
        scope.validate(&edit.expression).unwrap();
        let compiled = edit.expression.compile().unwrap();
        let row: Vec<_> = (0..columns).map(|i| (i as f64 - 0.5) / 16.0).collect();
        if scope.elementwise {
            let expected: Vec<_> = row
                .iter()
                .map(|x| {
                    edit.expression
                        .eval_element(*x, &row)
                        .unwrap_or(0.0)
                        .to_bits()
                })
                .collect();
            assert_eq!(
                compiled
                    .map_row(&row)
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                expected,
                "seed {seed}"
            );
        } else {
            assert_eq!(
                compiled.eval_row(&row).ok().map(f64::to_bits),
                edit.expression.eval_row(&row).ok().map(f64::to_bits),
                "seed {seed}"
            );
        }
        changed += usize::from(edit.changed);
        parent = edit.expression;
    }
    assert!(changed > 60_000, "changed {changed} of 100000");
}

#[test]
fn compiled_operations_preserve_protected_math_and_error_boundaries() {
    for source in [
        "2^100",
        "-2^2",
        "x-row_mean+row_std-row_sum",
        "clip(x,c0,c1)",
        "min(clip(1,(-1)^0.5,(-1)^0.5),7)+1",
        "c9+1",
        "sign(x)+floor(x)+round(x)",
        "erf(x)+asinh(x)+atan(x)+exp(x)+log(x)",
    ] {
        let expr = MathExpr::parse(source).unwrap();
        let compiled = expr.compile().unwrap();
        for row in [
            vec![],
            vec![f64::NAN, f64::NAN],
            vec![-0.0, f64::INFINITY],
            vec![-0.5, 2.0],
        ] {
            assert_eq!(
                compiled
                    .map_row(&row)
                    .into_iter()
                    .map(f64::to_bits)
                    .collect::<Vec<_>>(),
                row.iter()
                    .map(|x| expr.eval_element(*x, &row).unwrap_or(0.0).to_bits())
                    .collect::<Vec<_>>(),
                "{source} {row:?}"
            );
        }
    }
}

#[test]
fn local_constant_fitting_recovers_a_linear_formula_with_a_hard_evaluation_budget() {
    let initial = MathExpr::parse("0.1*c0+0.1").unwrap();
    let rows: Vec<_> = (-4..=4).map(|i| vec![i as f64]).collect();
    let targets: Vec<_> = rows.iter().map(|r| 2.5 * r[0] - 0.75).collect();
    let (fitted, count) = initial.optimize_constants_v1(&rows, &targets, 400).unwrap();
    assert_eq!(count, 400);
    for (row, expected) in rows.iter().zip(targets) {
        assert!((fitted.eval_row(row).unwrap() - expected).abs() < 1e-8);
    }
}

#[test]
fn new_entrypoints_recheck_deserialized_ast_structure() {
    // Historical serde decoding is a compatibility interface, not admission.
    let mut root = serde_json::json!({"Const": 1.0});
    for _ in 0..24 {
        root = serde_json::json!({"Unary": ["Neg", root]});
    }
    let expr: MathExpr = serde_json::from_value(serde_json::json!({"root": root})).unwrap();
    let scope = ExpressionScope {
        columns: 1,
        elementwise: true,
        max_nodes: 64,
        max_depth: 16,
    };
    assert!(expr.compile().is_err());
    assert!(expr.to_source().is_err());
    assert!(expr.mutate_v1(0, scope).is_err());
    assert!(MathExpr::parse("x")
        .unwrap()
        .crossover_v1(&expr, 0, scope)
        .is_err());
    assert!(expr
        .optimize_constants_v1(&[vec![1.0]], &[1.0], 10)
        .is_err());
}
