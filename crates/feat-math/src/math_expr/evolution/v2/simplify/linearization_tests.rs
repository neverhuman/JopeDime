//! The bounded minimax linearization, degree 1 only (laneD:B19-minimax-linearize).
use super::*;
use crate::kernel::profile::profile;
use anyhow::Context;
use std::collections::BTreeSet;

fn baseline() -> SearchProfile {
    profile("kernel2-baseline", 3).unwrap()
}

/// The kernel2 caps (15 nodes, depth 4) over two columns.
fn scope() -> ExpressionScope {
    ExpressionScope::search_profile(2, false)
}

/// `points` rows with `c0` evenly spaced over `[lo, hi]` and `c1` fixed.
fn rows(lo: f64, hi: f64, points: usize) -> Vec<Vec<f64>> {
    (0..points)
        .map(|i| vec![lo + (hi - lo) * i as f64 / (points - 1) as f64, 0.25])
        .collect()
}

fn linearize(source: &str, rows: &[Vec<f64>]) -> Vec<SubtreeLinearization> {
    MathExpr::parse(source)
        .unwrap()
        .linearize_v2(&baseline(), scope(), rows)
        .unwrap()
}

/// What a record keeps of the live measurements.
fn evidence(l: &SubtreeLinearization) -> LinearizationEvidence {
    LinearizationEvidence {
        fit: l.fit,
        lowered_error: l.lowered_error,
        refusal: l.refusal.clone(),
    }
}

fn recorded(
    source: &str,
    evidence: &[LinearizationEvidence],
) -> anyhow::Result<Vec<SubtreeLinearization>> {
    MathExpr::parse(source)
        .unwrap()
        .linearize_recorded_v2(&baseline(), scope(), &mut |i| {
            evidence.get(i).cloned().context("missing")
        })
}

/// A 6-node expression that is nearly linear on [-0.5, 1.5].
const NEAR_LINEAR: &str = "sin((0.1*c0))*3.0";
/// A 6-node one-variable subtree beside a second variable.
const BESIDE: &str = "(sin(c0)+(c0*c0))+c1";

/// The native line at `path` of an admitted expression stays within the record's own measured
/// error of the subtree on every training row.
fn assert_within_lowered_error(l: &SubtreeLinearization, training: &[Vec<f64>]) {
    let admitted = l.expression.as_ref().unwrap();
    let native = MathExpr::from_ast(at(&admitted.root, &l.path).clone()).unwrap();
    let lowered = l.lowered_error.unwrap();
    for row in training {
        let (a, faults) = native.eval_row_traced(row);
        let (b, _) = l.subtree.eval_row_traced(row);
        assert!(faults.is_empty());
        assert!((a - b).abs() <= lowered, "{row:?}");
    }
}

/// The larger size of a recorded line at its sampled ends, `M`: between the samples the computed
/// line lies between its values there.
fn largest_end(fit: &LinearFit) -> f64 {
    [fit.lo, fit.hi]
        .map(|end| horner(&fit.coefficients, end).abs())
        .into_iter()
        .fold(0.0, f64::max)
}

/// The least error a recorded line can carry on its own record, for the subtree of `l`: at either
/// sampled end, its residual against that end's target and its size past `max|f|`; at the largest
/// target, `max|f| - M`; and never under zero.
fn least_error(l: &SubtreeLinearization, fit: &LinearFit) -> f64 {
    let mut found = Vec::new();
    variables(&l.subtree.root, &mut found);
    [fit.lo, fit.hi]
        .map(|end| {
            let value = horner(&fit.coefficients, end);
            let target = target_at(&l.subtree.root, found[0], end).unwrap();
            (value - target).abs().max(value.abs() - fit.max_abs)
        })
        .into_iter()
        .fold(fit.max_abs - largest_end(fit), f64::max)
        .max(0.0)
}

/// Exactly linear plus `1e-6 sin(c0)` on [1e6, 1e6 + 1]: the fit's error is about 1e-7, and the
/// monomial line `2 x - 2000001` loses about 1e-10 to rounding at this offset, far over the
/// bound's `1e-6` relative and `64 ε max|f|` (1.4e-14) allowances.
const OFF_CENTRE: &str = "((c0*2.0)-2000001.0)+(sin(c0)*0.000001)";
/// A 9-node expression, nearly linear on [-0.5, 1.5], whose values have no bound short of the
/// evaluator's range: no bounded call holds its variable.
const UNBOUNDED: &str = "((c0*c0)*0.001)+(c0*0.3)";
/// `(sin(c0)+c0)` twice: the root and two equal subtrees are considered.
const REPEATED: &str = "(sin(c0)+c0)*(sin(c0)+c0)";
mod cases_1;
mod cases_2;
mod cases_3;
mod cases_4;
