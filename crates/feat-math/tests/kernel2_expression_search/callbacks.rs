use super::*;

#[test]
fn linear_scaled_refinement_recovers_frequency_intercept_and_scale() {
    let p = config();
    let rows: Vec<Vec<f64>> = (0..128)
        .map(|i| vec![-1.5 + 3.0 * i as f64 / 127.0])
        .collect();
    let targets: Vec<f64> = rows
        .iter()
        .map(|r| 3.0 * (1.7 * r[0]).sin() + 0.5)
        .collect();
    let expr = MathExpr::parse("sin(1.0*c0)").unwrap();
    let fit = expr
        .optimize_constants_v2(
            &p,
            scope(1, false),
            &rows,
            &targets,
            ConstantObjective::LinearScaled,
            512,
        )
        .unwrap();
    let text = fit.expression.to_source().unwrap();
    let coefficient = match fit.expression.ast() {
        Expr::Call(_, args) => match &args[0] {
            Expr::Binary(_, a, _) => match a.as_ref() {
                Expr::Const(v) => *v,
                _ => panic!("{text}"),
            },
            _ => panic!("{text}"),
        },
        _ => panic!("{text}"),
    };
    assert!((coefficient - 1.7).abs() < 1e-6, "{text}");
    let scale = fit.linear_scale.unwrap();
    assert!((scale[0] - 0.5).abs() < 1e-6 && (scale[1] - 3.0).abs() < 1e-6);
    assert!(fit.score < 1e-12 && fit.evaluations <= 512);
}

#[test]
fn callbacks_consume_only_supplied_training_rows_and_the_actual_budget() {
    let p = config();
    let rows = vec![vec![0.0], vec![1.0], vec![2.0]];
    let targets = vec![0.5, 2.5, 4.5];
    let expr = MathExpr::parse("0.5*c0+0.1").unwrap();
    for budget in [1, 2, 17, 101] {
        let calls = Cell::new(0);
        let mut objective =
            |candidate: &MathExpr, x: &[Vec<f64>], y: &[f64]| -> anyhow::Result<f64> {
                calls.set(calls.get() + 1);
                assert_eq!(x, rows);
                assert_eq!(y, targets);
                let sum: f64 = x
                    .iter()
                    .zip(y)
                    .map(|(r, y)| (candidate.eval_row(r).unwrap() - y).powi(2))
                    .sum();
                Ok(sum / x.len() as f64)
            };
        let fit = expr
            .optimize_constants_v2(
                &p,
                scope(1, false),
                &rows,
                &targets,
                ConstantObjective::Callback(&mut objective),
                budget,
            )
            .unwrap();
        assert!(fit.evaluations > 0 && fit.evaluations <= budget);
        assert_eq!(calls.get(), fit.evaluations - fit.rejected_trials);
        assert!(fit.linear_scale.is_none());
    }
}

#[test]
fn domain_and_hard_faults_never_reach_a_callback_as_valid_zero() {
    let p = config();
    for (text, rows) in [
        ("sqrt(c0-1.0)", vec![vec![0.0]]),
        ("c0+1.0", vec![vec![f64::NAN]]),
    ] {
        let calls = Cell::new(0);
        let mut objective = |_: &MathExpr, _: &[Vec<f64>], _: &[f64]| -> anyhow::Result<f64> {
            calls.set(calls.get() + 1);
            Ok(0.0)
        };
        let result = MathExpr::parse(text).unwrap().optimize_constants_v2(
            &p,
            scope(1, false),
            &rows,
            &[0.0],
            ConstantObjective::Callback(&mut objective),
            32,
        );
        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
    }
}

#[test]
fn faulting_trials_are_counted_rejected_and_never_returned_as_the_best() {
    let p = config();
    let rows = vec![vec![0.0], vec![0.1], vec![0.2]];
    let expr = MathExpr::parse("sqrt(c0+0.5)").unwrap();
    let fit = expr
        .optimize_constants_v2(
            &p,
            scope(1, false),
            &rows,
            &[0.0, 0.0, 0.0],
            ConstantObjective::LinearScaled,
            64,
        )
        .unwrap();
    assert!(fit.rejected_trials > 0 && fit.evaluations <= 64);
    for row in &rows {
        let (_, faults) = fit.expression.compile().unwrap().eval_row_traced(row);
        assert!(!faults.intersects(feat_math::math_expr::FaultSeverity::Domain.classes()));
    }
}

#[test]
fn malformed_profiles_and_empty_training_inputs_refuse_cleanly() {
    let p = config();
    let expr = MathExpr::parse("c0+0.5").unwrap();
    let mut bad = p.clone();
    for arm in bad.expression_edits.as_mut().unwrap() {
        arm.weight = 0;
    }
    assert!(expr.mutate_v2(&bad, scope(1, false), 0).is_err());
    let mut bad = p.clone();
    bad.expr.as_mut().unwrap().functions = vec!["exp"];
    assert!(MathExpr::grow_v2(&bad, scope(1, false), 0).is_err());
    assert!(expr
        .optimize_constants_v2(
            &p,
            scope(1, false),
            &[],
            &[],
            ConstantObjective::LinearScaled,
            10
        )
        .is_err());
    assert!(expr
        .optimize_constants_v2(
            &p,
            scope(1, false),
            &[vec![0.0]],
            &[0.0],
            ConstantObjective::LinearScaled,
            0
        )
        .is_err());
}

#[test]
fn one_node_scope_can_grow_x_but_never_read_it_outside_map() {
    let mut p = config();
    let pool = p.expr.as_mut().unwrap();
    pool.variables = vec!["x"];
    pool.column_upper_exclusive = 0;
    pool.birth_constants = vec![-0.0, -0.5];
    let s = ExpressionScope {
        columns: 0,
        elementwise: true,
        max_nodes: 1,
        max_depth: 1,
    };
    for seed in 0..1000 {
        let child = MathExpr::grow_v2(&p, s, seed).unwrap();
        assert_eq!(child.to_source().unwrap(), "x");
        let edit = child.mutate_v2(&p, s, seed).unwrap();
        assert_eq!(edit.expression.to_source().unwrap(), "x");
    }
    assert!(MathExpr::grow_v2(
        &p,
        ExpressionScope {
            elementwise: false,
            ..s
        },
        0
    )
    .is_err());
}

#[test]
fn elementwise_callback_checks_each_cell_and_refuses_empty_maps() {
    let p = config();
    let calls = Cell::new(0);
    let mut objective =
        |candidate: &MathExpr, rows: &[Vec<f64>], targets: &[f64]| -> anyhow::Result<f64> {
            calls.set(calls.get() + 1);
            Ok(rows
                .iter()
                .zip(targets)
                .map(|(row, y)| {
                    let prediction = candidate
                        .compile()
                        .unwrap()
                        .map_row(row)
                        .iter()
                        .sum::<f64>()
                        / row.len() as f64;
                    (prediction - y).powi(2)
                })
                .sum::<f64>()
                / rows.len() as f64)
        };
    let expr = MathExpr::parse("sqrt(x)+0.5").unwrap();
    let fit = expr
        .optimize_constants_v2(
            &p,
            scope(2, true),
            &[vec![1.0, 4.0]],
            &[2.0],
            ConstantObjective::Callback(&mut objective),
            1,
        )
        .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(fit.score, 0.0);
    for row in [vec![1.0, -1.0], vec![]] {
        assert!(expr
            .optimize_constants_v2(
                &p,
                scope(2, true),
                &[row],
                &[2.0],
                ConstantObjective::Callback(&mut objective),
                1
            )
            .is_err());
    }
    assert_eq!(
        calls.get(),
        1,
        "faulting/empty map must not invoke the objective"
    );
}

#[test]
fn elementwise_scope_without_x_refuses_empty_rows_before_callback() {
    let p = config();
    let expr = MathExpr::parse("row_l2+0.5").unwrap();
    let calls = Cell::new(0);
    let mut objective = |candidate: &MathExpr, rows: &[Vec<f64>], _: &[f64]| {
        calls.set(calls.get() + 1);
        assert!(!rows.iter().any(Vec::is_empty));
        assert_eq!(candidate.compile()?.map_row(&rows[0]), vec![5.5, 5.5]);
        Ok(0.0)
    };
    let fit = expr
        .optimize_constants_v2(
            &p,
            scope(2, true),
            &[vec![3.0, 4.0]],
            &[5.5],
            ConstantObjective::Callback(&mut objective),
            1,
        )
        .unwrap();
    assert_eq!(fit.score, 0.0);
    assert_eq!(calls.get(), 1);
    for rows in [vec![vec![]], vec![vec![3.0, 4.0], vec![]]] {
        let targets = vec![5.5; rows.len()];
        assert!(expr
            .optimize_constants_v2(
                &p,
                scope(2, true),
                &rows,
                &targets,
                ConstantObjective::Callback(&mut objective),
                1,
            )
            .is_err());
        assert_eq!(calls.get(), 1, "empty maps must not invoke the objective");
    }
    assert!(expr
        .optimize_constants_v2(
            &p,
            scope(2, true),
            &[vec![3.0, 4.0]],
            &[5.5],
            ConstantObjective::LinearScaled,
            1,
        )
        .is_err());
}
