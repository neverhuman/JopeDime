//! E6's per-trial cancellation/deadline seam, without changing the old optimizer.
use feat_math::kernel::expression_search::{
    ConstantObjective, ConstantRefinement, GuardedRefinementBudget,
};
use feat_math::kernel::profile::profile;
use feat_math::math_expr::{ExpressionScope, MathExpr};
use std::cell::Cell;

fn data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let rows: Vec<_> = (0..32).map(|i| vec![-1.55 + i as f64 / 10.0]).collect();
    let targets = rows
        .iter()
        .map(|r| 3.0 * (1.7 * r[0]).sin() + 0.5)
        .collect();
    (rows, targets)
}

fn same(expected: &ConstantRefinement, actual: &ConstantRefinement) {
    assert_eq!(
        actual.expression.to_source().unwrap(),
        expected.expression.to_source().unwrap()
    );
    assert_eq!(actual.score.to_bits(), expected.score.to_bits());
    assert_eq!(actual.evaluations, expected.evaluations);
    assert_eq!(actual.rejected_trials, expected.rejected_trials);
    assert_eq!(
        actual.linear_scale.map(|s| s.map(f64::to_bits)),
        expected.linear_scale.map(|s| s.map(f64::to_bits))
    );
}

#[test]
fn allowing_every_trial_keeps_existing_results_bit_for_bit() {
    let (rows, targets) = data();
    let p = profile("kernel2-baseline", 3).unwrap();
    for source in [
        "sin(0.5*c0)",
        "sin(0.5*c0)+(0.25*c0)",
        "sin(-0.5*c0)",
        "sin(c0)",
    ] {
        for budget in [1, 2, 3, 17, 101, 1024] {
            let expression = MathExpr::parse(source).unwrap();
            let expected = expression
                .optimize_constants_v2(
                    &p,
                    ExpressionScope::search_profile(1, false),
                    &rows,
                    &targets,
                    ConstantObjective::LinearScaled,
                    budget,
                )
                .unwrap();
            let starts = Cell::new(0);
            let mut guard = || {
                starts.set(starts.get() + 1);
                true
            };
            let actual = expression
                .optimize_constants_v2_guarded(
                    &p,
                    ExpressionScope::search_profile(1, false),
                    &rows,
                    &targets,
                    ConstantObjective::LinearScaled,
                    GuardedRefinementBudget {
                        max_evaluations: budget,
                        may_start: &mut guard,
                    },
                )
                .unwrap();
            same(&expected, actual.refinement.as_ref().unwrap());
            assert_eq!(starts.get(), actual.evaluations());
            assert!(!actual.stopped && actual.failure.is_none());
        }
    }
}

fn callback_loss(
    expression: &MathExpr,
    rows: &[Vec<f64>],
    targets: &[f64],
    elementwise: bool,
) -> anyhow::Result<f64> {
    let compiled = expression.compile()?;
    let mut loss = 0.0;
    for (row, target) in rows.iter().zip(targets) {
        let prediction = if elementwise {
            compiled.map_row(row)[0]
        } else {
            compiled.eval_row(row)?
        };
        loss += (prediction - target).powi(2);
    }
    Ok(loss / rows.len() as f64)
}

#[test]
fn callbacks_keep_row_and_elementwise_results_and_trial_order() {
    let (rows, targets) = data();
    let p = profile("kernel2-baseline", 3).unwrap();
    for (source, elementwise) in [("sin(0.5*c0)", false), ("sin(0.5*x)", true)] {
        for budget in [1, 3, 17, 101] {
            let expression = MathExpr::parse(source).unwrap();
            let mut expected_trials = Vec::new();
            let mut expected_objective = |e: &MathExpr, x: &[Vec<f64>], y: &[f64]| {
                expected_trials.push(e.to_source()?);
                callback_loss(e, x, y, elementwise)
            };
            let expected = expression
                .optimize_constants_v2(
                    &p,
                    ExpressionScope::search_profile(1, elementwise),
                    &rows,
                    &targets,
                    ConstantObjective::Callback(&mut expected_objective),
                    budget,
                )
                .unwrap();
            let mut actual_trials = Vec::new();
            let mut actual_objective = |e: &MathExpr, x: &[Vec<f64>], y: &[f64]| {
                actual_trials.push(e.to_source()?);
                callback_loss(e, x, y, elementwise)
            };
            let mut guard = || true;
            let actual = expression
                .optimize_constants_v2_guarded(
                    &p,
                    ExpressionScope::search_profile(1, elementwise),
                    &rows,
                    &targets,
                    ConstantObjective::Callback(&mut actual_objective),
                    GuardedRefinementBudget {
                        max_evaluations: budget,
                        may_start: &mut guard,
                    },
                )
                .unwrap();
            same(&expected, actual.refinement.as_ref().unwrap());
            assert_eq!(actual_trials, expected_trials);
            assert!(!actual.stopped && actual.failure.is_none());
        }
    }
}

#[test]
fn a_stop_before_the_initial_trial_starts_no_objective() {
    let (rows, targets) = data();
    let p = profile("kernel2-baseline", 3).unwrap();
    let calls = Cell::new(0);
    let mut objective = |_: &MathExpr, _: &[Vec<f64>], _: &[f64]| {
        calls.set(calls.get() + 1);
        Ok(1.0)
    };
    let mut guard = || false;
    let result = MathExpr::parse("sin(0.5*c0)")
        .unwrap()
        .optimize_constants_v2_guarded(
            &p,
            ExpressionScope::search_profile(1, false),
            &rows,
            &targets,
            ConstantObjective::Callback(&mut objective),
            GuardedRefinementBudget {
                max_evaluations: 64,
                may_start: &mut guard,
            },
        )
        .unwrap();
    assert!(result.stopped && result.refinement.is_none() && result.failure.is_none());
    assert_eq!(
        (result.evaluations(), result.rejected_trials(), calls.get()),
        (0, 0, 0)
    );
}

#[test]
fn stopping_between_trials_retains_the_same_best_and_actual_count() {
    let (rows, targets) = data();
    let p = profile("kernel2-baseline", 3).unwrap();
    let expression = MathExpr::parse("sin(0.5*c0)+(0.25*c0)").unwrap();
    for allowed in [1, 3, 9] {
        let expected = expression
            .optimize_constants_v2(
                &p,
                ExpressionScope::search_profile(1, false),
                &rows,
                &targets,
                ConstantObjective::LinearScaled,
                allowed,
            )
            .unwrap();
        let attempts = Cell::new(0);
        let mut guard = || {
            let n = attempts.get();
            attempts.set(n + 1);
            n < allowed
        };
        let actual = expression
            .optimize_constants_v2_guarded(
                &p,
                ExpressionScope::search_profile(1, false),
                &rows,
                &targets,
                ConstantObjective::LinearScaled,
                GuardedRefinementBudget {
                    max_evaluations: 64,
                    may_start: &mut guard,
                },
            )
            .unwrap();
        assert!(actual.stopped && actual.failure.is_none());
        same(&expected, actual.refinement.as_ref().unwrap());
        assert_eq!(actual.evaluations(), allowed);
        assert_eq!(attempts.get(), allowed + 1);
    }
}

#[test]
fn fault_refusals_spend_trials_and_never_reach_the_callback() {
    let rows: Vec<_> = (0..32).map(|i| vec![0.1 + i as f64 / 100.0]).collect();
    let targets = vec![1.0; rows.len()];
    let p = profile("kernel2-baseline", 3).unwrap();
    let calls = Cell::new(0);
    let mut objective = |e: &MathExpr, x: &[Vec<f64>], _: &[f64]| {
        calls.set(calls.get() + 1);
        Ok(x.iter().map(|r| e.eval_row(r).unwrap()).sum::<f64>())
    };
    let starts = Cell::new(0);
    let mut guard = || {
        let n = starts.get();
        starts.set(n + 1);
        n < 7
    };
    let result = MathExpr::parse("sqrt(c0+0.5)")
        .unwrap()
        .optimize_constants_v2_guarded(
            &p,
            ExpressionScope::search_profile(1, false),
            &rows,
            &targets,
            ConstantObjective::Callback(&mut objective),
            GuardedRefinementBudget {
                max_evaluations: 64,
                may_start: &mut guard,
            },
        )
        .unwrap();
    assert!(result.stopped && result.failure.is_none());
    assert_eq!(result.evaluations(), 7);
    assert!(result.rejected_trials() > 0);
    assert_eq!(calls.get() + result.rejected_trials(), result.evaluations());
}

#[test]
fn a_failed_initial_trial_is_reported_and_charged_once() {
    let (rows, targets) = data();
    let p = profile("kernel2-baseline", 3).unwrap();
    let calls = Cell::new(0);
    let mut objective = |_: &MathExpr, _: &[Vec<f64>], _: &[f64]| {
        calls.set(calls.get() + 1);
        Ok(0.0)
    };
    let mut guard = || true;
    let result = MathExpr::parse("sqrt(c0)")
        .unwrap()
        .optimize_constants_v2_guarded(
            &p,
            ExpressionScope::search_profile(1, false),
            &rows,
            &targets,
            ConstantObjective::Callback(&mut objective),
            GuardedRefinementBudget {
                max_evaluations: 64,
                may_start: &mut guard,
            },
        )
        .unwrap();
    assert!(result.refinement.is_none() && !result.stopped);
    assert!(result
        .failure
        .as_ref()
        .unwrap()
        .contains("faulting constant trial"));
    assert_eq!(
        (result.evaluations(), result.rejected_trials(), calls.get()),
        (1, 1, 0)
    );
}

#[test]
fn invalid_configuration_is_refused_before_consulting_the_guard() {
    let (rows, targets) = data();
    let p = profile("kernel2-baseline", 3).unwrap();
    let starts = Cell::new(0);
    let mut guard = || {
        starts.set(starts.get() + 1);
        true
    };
    let expression = MathExpr::parse("sin(0.5*c0)").unwrap();
    for (x, y, budget) in [
        (&rows[..], &targets[..], 0),
        (&rows[..0], &targets[..0], 8),
        (&rows[..], &targets[..1], 8),
    ] {
        assert!(expression
            .optimize_constants_v2_guarded(
                &p,
                ExpressionScope::search_profile(1, false),
                x,
                y,
                ConstantObjective::LinearScaled,
                GuardedRefinementBudget {
                    max_evaluations: budget,
                    may_start: &mut guard
                }
            )
            .is_err());
    }
    assert_eq!(starts.get(), 0);
}

fn check_invalid_elementwise_linear_objective(allow: bool) {
    let p = profile("kernel2-baseline", 3).unwrap();
    let scope = ExpressionScope::search_profile(1, true);
    let expression = MathExpr::parse("x+0.5").unwrap();
    let rows = [vec![1.0, 2.0]];
    let targets = [1.5];
    let old_error = expression
        .optimize_constants_v2(
            &p,
            scope,
            &rows,
            &targets,
            ConstantObjective::LinearScaled,
            8,
        )
        .unwrap_err();
    let starts = Cell::new(0);
    let mut guard = || {
        starts.set(starts.get() + 1);
        allow
    };
    let error = expression
        .optimize_constants_v2_guarded(
            &p,
            scope,
            &rows,
            &targets,
            ConstantObjective::LinearScaled,
            GuardedRefinementBudget {
                max_evaluations: 8,
                may_start: &mut guard,
            },
        )
        .unwrap_err();
    assert_eq!(format!("{error:#}"), format!("{old_error:#}"));
    assert_eq!(starts.get(), 0);
}

#[test]
fn invalid_pairing_precedes_false_guard() {
    check_invalid_elementwise_linear_objective(false);
}

#[test]
fn invalid_pairing_precedes_true_guard() {
    check_invalid_elementwise_linear_objective(true);
}
