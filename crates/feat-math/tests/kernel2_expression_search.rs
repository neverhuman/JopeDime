//! Track E2's versioned configuration and expression operator contract.
use feat_math::kernel::expression_search::{ConstantObjective, ExpressionArm};
use feat_math::kernel::profile::{profile, SearchProfile};
use feat_math::kernel::search::admit_child;
use feat_math::math_expr::{Expr, ExpressionScope, MathExpr};
use std::cell::Cell;

fn config() -> SearchProfile {
    profile("kernel2-baseline", 2).unwrap()
}
fn scope(width: usize, elementwise: bool) -> ExpressionScope {
    ExpressionScope::search_profile(width.min(8), elementwise)
}
fn size(node: &Expr) -> usize {
    1 + match node {
        Expr::Unary(_, a) => size(a),
        Expr::Binary(_, a, b) => size(a) + size(b),
        Expr::Call(_, args) => args.iter().map(size).sum(),
        _ => 0,
    }
}
fn admitted(expr: &MathExpr, p: &SearchProfile, width: usize, elementwise: bool) {
    let text = expr.to_source().unwrap();
    let reparsed = MathExpr::parse(&text).unwrap();
    scope(width, elementwise).validate(&reparsed).unwrap();
    let carrier = if elementwise { "map" } else { "derive" };
    admit_child(
        &format!("(pipe X ({carrier} expr=\"{text}\") (ridge))"),
        p,
        width,
    )
    .unwrap();
}

#[test]
fn expression_operator_revision_keeps_every_revision_one_identity() {
    for (id, pinned) in [
        (
            "kernel2-baseline",
            "78bb4c1cc430cd8179bd863e4e16d1e146a7e2e631fd75904c0e2d65a77c3b61",
        ),
        (
            "kernel2-experimental",
            "1989621b3450ff3347dfe1401aa75da1654966268dad2cf348b2656268d9e1d5",
        ),
    ] {
        assert_eq!(profile(id, 1).unwrap().sha256(), pinned);
        let revision_two =
            profile(id, 2).expect("E2 needs a separately versioned operator profile");
        let value: serde_json::Value =
            serde_json::from_str(&revision_two.canonical_json()).unwrap();
        assert_eq!(value["expression_edits"].as_array().unwrap().len(), 7);
        assert_ne!(revision_two.sha256(), pinned);
    }
}

#[test]
fn one_hundred_thousand_grown_children_have_variable_leaves_and_reparse_in_scope() {
    let p = config();
    for seed in 0..100_000 {
        let width = [1, 3, 8, 20, 600][seed as usize % 5];
        let elementwise = seed % 2 == 0;
        let child = MathExpr::grow_v2(&p, scope(width, elementwise), seed).unwrap();
        admitted(&child, &p, width, elementwise);
    }
}

fn exercise_arm(arm: ExpressionArm) {
    let mut p = config();
    for weight in p.expression_edits.as_mut().unwrap() {
        weight.weight = u32::from(weight.arm == arm);
    }
    let parent = MathExpr::parse("sin(c0)+(0.5*c0)").unwrap();
    let mut changed = 0;
    for seed in 0..100_000 {
        let width = [1, 3, 8, 20, 600][seed as usize % 5];
        let elementwise = seed % 2 == 0;
        let child = parent
            .mutate_v2(&p, scope(width, elementwise), seed)
            .unwrap();
        assert_eq!(child.arm, arm);
        changed += usize::from(child.changed);
        admitted(&child.expression, &p, width, elementwise);
    }
    assert!(
        changed > 1000,
        "{arm:?} must actually edit, rather than always return its parent"
    );
}
macro_rules! arm_test {
    ($name:ident, $arm:ident) => {
        #[test]
        fn $name() {
            exercise_arm(ExpressionArm::$arm);
        }
    };
}
arm_test!(point_arm_one_hundred_thousand_children, Point);
arm_test!(subtree_arm_one_hundred_thousand_children, Subtree);
arm_test!(hoist_arm_one_hundred_thousand_children, Hoist);
arm_test!(shrink_arm_one_hundred_thousand_children, Shrink);
arm_test!(
    constant_jitter_arm_one_hundred_thousand_children,
    ConstantJitter
);
arm_test!(wrap_arm_one_hundred_thousand_children, Wrap);
arm_test!(
    insert_binary_arm_one_hundred_thousand_children,
    InsertBinary
);

#[test]
fn sampled_arm_frequencies_follow_the_consumed_profile_weights() {
    let mut p = config();
    for (i, weight) in p.expression_edits.as_mut().unwrap().iter_mut().enumerate() {
        weight.weight = i as u32 + 1;
    }
    let parent = MathExpr::parse("sin(c0)+(0.5*c0)").unwrap();
    let mut counts = [0usize; 7];
    for seed in 0..100_000 {
        let child = parent.mutate_v2(&p, scope(3, false), seed).unwrap();
        counts[child.arm as usize] += 1;
    }
    let chi_square: f64 = counts
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let expected = 100_000.0 * (i + 1) as f64 / 28.0;
            (*n as f64 - expected).powi(2) / expected
        })
        .sum();
    assert!(
        chi_square < 22.458,
        "six-degree-of-freedom p=.001 threshold: {counts:?}, chi-square={chi_square}"
    );
}

#[test]
fn a_thousand_chained_edits_remain_bounded_and_do_not_grow_monotonically() {
    let p = config();
    let mut expr = MathExpr::parse("sin(c0)+(0.5*c0)").unwrap();
    let mut sizes = vec![size(expr.ast())];
    for seed in 0..1000 {
        expr = expr
            .mutate_v2(&p, scope(3, false), seed)
            .unwrap()
            .expression;
        admitted(&expr, &p, 3, false);
        sizes.push(size(expr.ast()));
    }
    assert!(sizes.windows(2).any(|p| p[1] < p[0]));
    assert!(sizes.windows(2).any(|p| p[1] > p[0]));
    sizes.sort();
    assert!(*sizes.last().unwrap() <= 15);
    assert!(sizes[sizes.len() / 2] < 15);
}

#[test]
fn one_hundred_thousand_crossovers_are_size_fair_and_use_recipient_scope() {
    let p = config();
    let recipient = MathExpr::parse("c0").unwrap();
    let donor = MathExpr::parse("(sin(c7)+row_mean)+(tanh(x)+row_l2)").unwrap();
    let mut changed = 0;
    for seed in 0..100_000 {
        let child = recipient
            .crossover_v2(&donor, &p, scope(1, false), seed)
            .unwrap();
        assert!(
            size(child.expression.ast()) <= 3,
            "a replacement of one node permits at most three donor nodes"
        );
        admitted(&child.expression, &p, 1, false);
        changed += usize::from(child.changed);
    }
    assert!(changed > 1000);
}

#[test]
fn every_version_two_operator_replays_from_its_seed() {
    let p = config();
    let s = scope(3, false);
    let parent = MathExpr::parse("sin(c0)+(0.5*c0)").unwrap();
    let donor = MathExpr::parse("row_mean+c1").unwrap();
    for seed in 0..1000 {
        assert_eq!(
            MathExpr::grow_v2(&p, s, seed).unwrap().to_source().unwrap(),
            MathExpr::grow_v2(&p, s, seed).unwrap().to_source().unwrap()
        );
        assert_eq!(
            parent
                .mutate_v2(&p, s, seed)
                .unwrap()
                .expression
                .to_source()
                .unwrap(),
            parent
                .mutate_v2(&p, s, seed)
                .unwrap()
                .expression
                .to_source()
                .unwrap()
        );
        assert_eq!(
            parent
                .crossover_v2(&donor, &p, s, seed)
                .unwrap()
                .expression
                .to_source()
                .unwrap(),
            parent
                .crossover_v2(&donor, &p, s, seed)
                .unwrap()
                .expression
                .to_source()
                .unwrap()
        );
    }
}

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
