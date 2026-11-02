use super::*;

// ---------------------------------------------------------------------------------------------
// Lowering.

/// `Σ a_k x^k` in Horner form over the leaf `x`, as ordinary AST nodes: `Add`/`Sub` of finite
/// literals and `Mul` by `x`. A negative non-leading coefficient becomes a `Sub`, and a zero one
/// is skipped. No `Pow` (its precision is unspecified) and no new node type.
pub(crate) fn lower(coefficients: &[f64], x: &Expr) -> Option<Expr> {
    if coefficients.is_empty() || coefficients.iter().any(|c| !c.is_finite()) {
        return None;
    }
    let n = coefficients.len() - 1;
    let mut acc = literal(coefficients[n]);
    for &a in coefficients[..n].iter().rev() {
        let product = Expr::Binary(BinOp::Mul, Box::new(x.clone()), Box::new(acc));
        acc = if a == 0.0 {
            product
        } else if a > 0.0 {
            Expr::Binary(BinOp::Add, Box::new(product), Box::new(Expr::Const(a)))
        } else {
            Expr::Binary(BinOp::Sub, Box::new(product), Box::new(Expr::Const(-a)))
        };
    }
    Some(acc)
}

/// The lowered tree's value at `v`, in exactly the operation order [`lower`] builds: `x * acc`,
/// then `+ a` or `- (-a)`, which the evaluator computes with the same bits.
pub(crate) fn horner(coefficients: &[f64], v: f64) -> f64 {
    let n = coefficients.len() - 1;
    let mut acc = coefficients[n];
    for &a in coefficients[..n].iter().rev() {
        let product = v * acc;
        acc = if a == 0.0 {
            product
        } else if a > 0.0 {
            product + a
        } else {
            product - (-a)
        };
    }
    acc
}

/// The largest `|p(v) - f|` of the lowered polynomial over `unique` samples, in its own Horner
/// order, or `None` when a value leaves the finite range the output clamp keeps unchanged. The
/// monomial conversion is ill-conditioned away from 0, so this can be far above the fit's
/// Chebyshev `max_error`.
pub(crate) fn lowered_error(coefficients: &[f64], unique: &[(f64, f64)]) -> Option<f64> {
    if coefficients.is_empty() {
        return None;
    }
    unique.iter().try_fold(0.0f64, |worst, &(v, f)| {
        let value = horner(coefficients, v);
        (value.is_finite() && value.abs() <= MAX_ABS).then(|| worst.max((value - f).abs()))
    })
}

/// A lowering worth proposing: the lowered tree, its own error on the samples and its size.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Proposal {
    pub(crate) tree: Expr,
    /// What the proposal is held to: the lowered tree's error, never the fit's alone.
    pub(crate) lowered_error: f64,
    pub(crate) nodes: usize,
}

/// The lowered form of `fit` over the variable leaf `x`, when it is worth proposing in place of
/// `subtree`. It must have fewer nodes than `subtree`, fit `node_budget` and the expression limits
/// (64 nodes, depth 16) on its own, and stay, in its own evaluation order, within an allowance on
/// `samples` of `max_error (1 + 1e-6) + 64 ε max|f|`. Otherwise `None`. The lowered tree is
/// usually deeper than the subtree it replaces (depth `2n + 1`), so the caller must re-validate
/// the whole program under its scope, as [`MathExpr::linearize_v2`] does.
///
/// Its checks are [`assess`]'s, which also names the first that fails; outside the tests the
/// linearization calls `assess` directly.
#[cfg(test)]
pub(crate) fn propose(
    fit: &Minimax,
    samples: &[(f64, f64)],
    x: &Expr,
    subtree: &Expr,
    node_budget: usize,
) -> Option<Proposal> {
    let unique = prepare(samples).ok()?;
    let f_max = largest(&unique);
    assess(
        &fit.monomial(),
        fit.max_error,
        f_max,
        x,
        subtree,
        node_budget,
        |coefficients| lowered_error(coefficients, &unique),
    )
    .ok()
}

/// The largest `|f|` over samples.
pub(crate) fn largest(unique: &[(f64, f64)]) -> f64 {
    unique.iter().map(|(_, f)| f.abs()).fold(0.0f64, f64::max)
}

/// The minimax bound a lowered tree is held to on its samples: the fit's own largest error, a
/// relative `1e-6` over it, and the rounding level `64 ε max|f|`. It measures the precision lost
/// in lowering the fit to ordinary nodes, not the line's quality: a poor fit has a large
/// `max_error` and a bound to match. Whether the line is any good is for the native score of the
/// whole program to decide.
pub(crate) fn allowance(max_error: f64, f_max: f64) -> f64 {
    max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * f_max
}

/// `x` is not a variable leaf, so the samples do not give the lowered tree's values.
pub(crate) const NOT_A_LEAF: &str = "the lowered variable is not a leaf";

/// A refused lowering: the first check that failed, and what was measured before it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Refused {
    pub(crate) reason: &'static str,
    pub(crate) nodes: Option<usize>,
    pub(crate) lowered_error: Option<f64>,
}

/// The checks of a proposal, in order, for the polynomial `coefficients` (ascending powers) of a
/// fit whose largest error is `max_error`, on samples whose largest `|f|` is `f_max`. `measure`
/// gives the lowered tree's own largest error over those samples, in its own evaluation order, or
/// `None` when a value leaves the output range. The first check that fails is named:
/// - `x` must be a variable leaf, because the error is measured at the sampled abscissae;
/// - every coefficient must be finite, so the tree can be lowered;
/// - the tree must have fewer nodes than `subtree`, and no more than `node_budget`;
/// - it must fit the expression limits (64 nodes, depth 16) on its own;
/// - its own error must stay in the output range and within [`allowance`]: the minimax bound,
///   which holds the lowered tree to its fit's own training error, so it rejects lowerings that
///   lose precision and never judges the fit itself.
pub(crate) fn assess(
    coefficients: &[f64],
    max_error: f64,
    f_max: f64,
    x: &Expr,
    subtree: &Expr,
    node_budget: usize,
    measure: impl FnOnce(&[f64]) -> Option<f64>,
) -> Result<Proposal, Refused> {
    let refused = |reason, nodes, lowered_error| Refused {
        reason,
        nodes,
        lowered_error,
    };
    if !matches!(x, Expr::Var(_)) {
        return Err(refused(NOT_A_LEAF, None, None));
    }
    let Some(tree) = lower(coefficients, x) else {
        return Err(refused(LINEARIZE_NON_FINITE, None, None));
    };
    let nodes = count_nodes(&tree);
    if nodes >= count_nodes(subtree) {
        return Err(refused(LINEARIZE_NOT_SMALLER, Some(nodes), None));
    }
    if nodes > node_budget {
        return Err(refused(LINEARIZE_OVER_BUDGET, Some(nodes), None));
    }
    if MathExpr::from_ast(tree.clone()).is_err() {
        return Err(refused(LINEARIZE_OVER_LIMITS, Some(nodes), None));
    }
    let Some(lowered_error) = measure(coefficients) else {
        return Err(refused(LINEARIZE_OUT_OF_RANGE, Some(nodes), None));
    };
    if lowered_error > allowance(max_error, f_max) {
        return Err(refused(
            LINEARIZE_MISSES_BOUND,
            Some(nodes),
            Some(lowered_error),
        ));
    }
    Ok(Proposal {
        tree,
        lowered_error,
        nodes,
    })
}
