use super::*;

/// Every variable `expr` reads, in first-read order.
pub(super) fn variables(expr: &Expr, out: &mut Vec<Var>) {
    match expr {
        Expr::Var(v) => {
            if !out.contains(v) {
                out.push(*v);
            }
        }
        Expr::Const(_) => {}
        Expr::Unary(_, a) => variables(a, out),
        Expr::Binary(_, a, b) => {
            variables(a, out);
            variables(b, out);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| variables(a, out)),
    }
}

/// The one variable a subtree reads, and its `(v, f)` samples on the training rows. `None` when it
/// reads no variable or more than one, or any row raises any fault or gives a non-finite value.
pub(super) fn samples(subtree: &MathExpr, rows: &[Vec<f64>]) -> Option<(Var, Vec<(f64, f64)>)> {
    let mut found = Vec::new();
    variables(&subtree.root, &mut found);
    let [var] = found[..] else {
        return None;
    };
    let leaf = MathExpr::from_ast(Expr::Var(var)).ok()?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let (v, v_faults) = leaf.eval_row_traced(row);
        let (f, f_faults) = subtree.eval_row_traced(row);
        if !v_faults.is_empty() || !f_faults.is_empty() || !v.is_finite() || !f.is_finite() {
            return None;
        }
        out.push((v, f));
    }
    Some((var, out))
}

/// The target of `subtree`, which reads only `var`, where `var` reads the sampled value `v`: the
/// target [`samples`] records at such a row. A fault-free read of a finite value passes it on
/// unchanged, so the subtree with `var` read as the element `x` computes the same target from
/// `v`. `None` when that evaluation faults or is not finite, which no sampled row allows.
pub(super) fn target_at(subtree: &Expr, var: Var, v: f64) -> Option<f64> {
    let mut tree = subtree.clone();
    read_as_element(&mut tree, var);
    let (f, faults) = MathExpr::from_ast(tree).ok()?.eval_element_traced(v, &[]);
    (faults.is_empty() && f.is_finite()).then_some(f)
}

/// `expr` with each read of `var` made a read of the element `x`.
pub(super) fn read_as_element(expr: &mut Expr, var: Var) {
    match expr {
        Expr::Var(v) => {
            if *v == var {
                *v = Var::X;
            }
        }
        Expr::Const(_) => {}
        Expr::Unary(_, a) => read_as_element(a, var),
        Expr::Binary(_, a, b) => {
            read_as_element(a, var);
            read_as_element(b, var);
        }
        Expr::Call(_, args) => args.iter_mut().for_each(|a| read_as_element(a, var)),
    }
}

// ---------------------------------------------------------------------------------------------
// Bounded minimax linearization (laneD:B19-minimax-linearize).

/// The subtrees a line could replace, in preorder: each reads exactly one variable, has more
/// nodes than the smallest line (`x * a`, three nodes at depth two), and sits where that line fits
/// `max_depth`. Under the kernel2 depth limit of 4 these are the root and its children, and a
/// grandchild only when it is a call of three arguments.
pub(super) fn linearizable(root: &Expr, max_depth: usize) -> Vec<Vec<usize>> {
    paths(root)
        .into_iter()
        .filter(|path| {
            let node = at(root, path);
            let mut found = Vec::new();
            variables(node, &mut found);
            found.len() == 1 && count_nodes(node) > 3 && path.len() + 2 <= max_depth
        })
        .collect()
}

/// How a line's own largest error over the training samples is measured, from its coefficients.
pub(super) type Measure = Box<dyn FnOnce(&[f64]) -> Option<f64>>;

/// Where a linearization's training measurements come from: the training rows themselves, or a
/// record of what was measured on them, by subtree index.
pub(super) enum Source<'a> {
    Rows(&'a [Vec<f64>]),
    Recorded(&'a mut dyn FnMut(usize) -> Result<LinearizationEvidence>),
}

/// Whether a recorded fit agrees with its stop's own definition, as far as its recorded numbers
/// show: finite, a training range and largest `|f|` within the evaluator's range (`MAX_ABS`, the
/// only range [`samples`] produces), a range holding at least three distinct abscissae (a line's
/// fit needs three), a bracket that does not invert, with its two ends and `max|f|` of positive
/// sign (each is a maximum of absolute values from `+0.0`), steps within the cap, the stop's own
/// step (a closing stop returns the fit of the step it stops at, the cap stops after the last
/// step), and the stop's closing test (`ExactFit`, `Converged` and `NoiseLimited` are tested as
/// [`minimax`] tests them). [`minimax`] returns `ExactFit` at the first fit within its test, so a
/// fit with any other stop, the current one or the best seen, is outside it. A finite zero
/// coefficient is `+0.0`, and a fit whose targets are all zero is the zero polynomial, exact at
/// the first step. The range bound
/// stops `max_abs` only past 1e12: an inflated `max_abs` within it still meets the `ExactFit` test
/// for a `max_error` up to `4 ε 1e12` (about 8.9e-4) and the `NoiseLimited` test for a bracket up
/// to `64 ε 1e12` (about 0.014), and widens the bound with it. It cannot tell a consistent record
/// from the fit the training rows give: without the rows, an edit that keeps every number
/// consistent (say, a larger `max_error` and `levelled` with a closed bracket) passes.
pub(super) fn consistent(fit: &LinearFit) -> bool {
    let finite = [fit.lo, fit.hi, fit.levelled, fit.max_error, fit.max_abs]
        .iter()
        .all(|v| v.is_finite());
    let in_range = [fit.lo, fit.hi, fit.max_abs]
        .iter()
        .all(|v| v.abs() <= MAX_ABS);
    let width = fit.max_error - fit.levelled;
    let exact = fit.max_error <= 4.0 * f64::EPSILON * fit.max_abs;
    let closing = match fit.stop {
        MinimaxStop::ExactFit => exact,
        MinimaxStop::Converged => !exact && width <= TOLERANCE * fit.max_error,
        MinimaxStop::NoiseLimited => !exact && width <= 64.0 * f64::EPSILON * fit.max_abs,
        MinimaxStop::Stalled | MinimaxStop::Degenerate | MinimaxStop::IterationLimit => !exact,
    };
    let signed = [fit.levelled, fit.max_error, fit.max_abs]
        .iter()
        .all(|v| v.is_sign_positive());
    let stepped = match fit.stop {
        MinimaxStop::ExactFit | MinimaxStop::Converged | MinimaxStop::NoiseLimited => {
            fit.iterations == fit.steps
        }
        MinimaxStop::IterationLimit => fit.steps == MAX_ITERATIONS,
        MinimaxStop::Stalled | MinimaxStop::Degenerate => true,
    };
    // The monomial conversion adds rounded products to `+0.0`, so a finite zero coefficient is
    // `+0.0`: an exact cancellation rounds to `+0.0` too.
    let positive_zeros = fit
        .coefficients
        .iter()
        .all(|c| *c != 0.0 || c.is_sign_positive());
    // With every target zero the first solve gives the zero polynomial, whose fit is exact at the
    // first step; its coefficients are the zero line's on the range, as the conversion gives them.
    let zero_targets = fit.max_abs != 0.0
        || (fit.stop == MinimaxStop::ExactFit
            && fit.iterations == 1
            && fit.steps == 1
            && fit.max_error == 0.0
            && constant_line_recorded(fit.coefficients, 0.0, fit.lo, fit.hi));
    finite
        && in_range
        && signed
        && stepped
        && positive_zeros
        && zero_targets
        && fit.lo.next_up() < fit.hi
        && 0.0 <= fit.levelled
        && fit.levelled <= fit.max_error
        && fit.max_abs >= 0.0
        && 1 <= fit.iterations
        && fit.iterations <= fit.steps
        && fit.steps <= MAX_ITERATIONS
        && closing
}

/// The monomial pair [`Minimax::monomial`] gives a fit whose Chebyshev coefficients are `value`
/// and zero on `[lo, hi]`: the line of a first solve over constant targets (the reference
/// system's first column is all ones, so elimination leaves the other unknowns a zero right-hand
/// side). The conversion's own code, in its own order: `value` itself (`+0.0` for a zero), and a
/// zero slope, each NaN where the conversion scales zero by an infinite `α` or `β`. A zero slope's
/// sign never shows in the sums, which add it to `+0.0` or to `value`.
pub(super) fn constant_line(value: f64, lo: f64, hi: f64) -> [f64; 2] {
    let line = Minimax {
        lo,
        hi,
        chebyshev: vec![value, 0.0],
        levelled: 0.0,
        max_error: 0.0,
        iterations: 1,
        steps: 1,
        stop: Stop::ExactFit,
        reference: Vec::new(),
    }
    .monomial();
    [line[0], line[1]]
}

/// Whether recorded coefficients are the constant line's on `[lo, hi]`: bit for bit, a NaN where
/// the conversion gives one; or both NaN where it gives a NaN, as a published record keeps a pair
/// with any coefficient that is not finite (as absent, read back as NaN).
pub(super) fn constant_line_recorded(coefficients: [f64; 2], value: f64, lo: f64, hi: f64) -> bool {
    let expected = constant_line(value, lo, hi);
    let exact = coefficients.iter().zip(expected).all(|(c, e)| {
        if e.is_nan() {
            c.is_nan()
        } else {
            c.to_bits() == e.to_bits()
        }
    });
    exact || (expected.iter().any(|e| e.is_nan()) && coefficients.iter().all(|c| c.is_nan()))
}

/// What the recorded subtrees of one expression say about the samples they share. The same
/// subtree has the same samples, so the same fit, line error, or refusal before any fit; and
/// every subtree that reads a variable reads its values on the same rows, so its fits share one
/// training range, and a fit and a refusal of those values alone ([`FEW_ABSCISSAE`],
/// [`EMPTY_INTERVAL`]) cannot both occur. A record that disagrees with another of the same
/// expression is refused. A refusal that one subtree's own values faulted
/// ([`LINEARIZE_UNSAMPLED`]) has no such witness unless the same subtree occurs twice. Rows that
/// could not be built ([`LINEARIZE_INPUT_FAULTS`]) are missing for every subtree, so that refusal
/// is recorded for all of the expression's subtrees or none.
#[derive(Default)]
pub(super) struct Shared {
    /// Each recorded subtree with its fit, its line's error, and its refusal when it has no fit.
    pub(super) subtrees: Vec<(Expr, SampleEvidence)>,
    /// Per variable: the training range its fits record, and whether a fit refused its values.
    pub(super) variables: std::collections::BTreeMap<Var, (Option<(u64, u64)>, bool)>,
    /// Whether the expression's input rows could not be built, as its first record says.
    pub(super) input_faults: Option<bool>,
}

/// The fit, the line's measured error, and the refusal of a subtree without a fit, with every
/// number as its exact bits. A record compares with another bit for bit, NaN included: a fit
/// refused for coefficients that are not finite is published without them and read back as NaN,
/// which never equals itself, while two records of the same subtree hold the same bits.
pub(super) type SampleEvidence = (Option<FitBits>, Option<u64>, Option<String>);

/// A fit as exact bits: its stop, coefficients, range, bracket and largest `|f|`, and its steps.
pub(super) type FitBits = (MinimaxStop, [u64; 2], [u64; 5], [usize; 2]);

pub(super) fn fit_bits(fit: &LinearFit) -> FitBits {
    (
        fit.stop,
        fit.coefficients.map(f64::to_bits),
        [fit.lo, fit.hi, fit.levelled, fit.max_error, fit.max_abs].map(f64::to_bits),
        [fit.iterations, fit.steps],
    )
}

impl Shared {
    pub(super) fn check(
        &mut self,
        subtree: &Expr,
        var: Var,
        evidence: &LinearizationEvidence,
    ) -> Result<()> {
        let samples: SampleEvidence = (
            evidence.fit.as_ref().map(fit_bits),
            evidence.lowered_error.map(f64::to_bits),
            evidence
                .fit
                .is_none()
                .then(|| evidence.refusal.clone())
                .flatten(),
        );
        match self.subtrees.iter().find(|(seen, _)| seen == subtree) {
            Some((_, seen)) => {
                anyhow::ensure!(*seen == samples, "a repeated subtree records other samples")
            }
            None => self.subtrees.push((subtree.clone(), samples)),
        }
        let (range, refused) = self.variables.entry(var).or_default();
        if let Some(fit) = evidence.fit {
            let recorded = (fit.lo.to_bits(), fit.hi.to_bits());
            anyhow::ensure!(
                *range.get_or_insert(recorded) == recorded,
                "subtrees of one variable record different training ranges"
            );
        }
        let reason = evidence
            .refusal
            .as_deref()
            .and_then(|r| r.strip_prefix(LINEARIZE_FIT_REFUSED))
            .and_then(|r| r.strip_prefix(": "));
        if evidence.fit.is_none() && matches!(reason, Some(FEW_ABSCISSAE | EMPTY_INTERVAL)) {
            *refused = true;
        }
        anyhow::ensure!(
            !(range.is_some() && *refused),
            "a fit and a refusal of the same variable's values disagree"
        );
        let faulted =
            evidence.fit.is_none() && evidence.refusal.as_deref() == Some(LINEARIZE_INPUT_FAULTS);
        anyhow::ensure!(
            *self.input_faults.get_or_insert(faulted) == faulted,
            "a faulting input refuses only some of an expression's subtrees"
        );
        Ok(())
    }
}

/// Whether `a` and `b` are the same tree, each literal compared by its bits.
pub(super) fn same_tree(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Const(x), Expr::Const(y)) => x.to_bits() == y.to_bits(),
        (Expr::Var(x), Expr::Var(y)) => x == y,
        (Expr::Unary(o, x), Expr::Unary(p, y)) => o == p && same_tree(x, y),
        (Expr::Binary(o, x1, x2), Expr::Binary(p, y1, y2)) => {
            o == p && same_tree(x1, y1) && same_tree(x2, y2)
        }
        (Expr::Call(f, xs), Expr::Call(g, ys)) => {
            f == g && xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| same_tree(x, y))
        }
        _ => false,
    }
}

/// Whether `expr` is zero wherever it is finite: every evaluation, on any input, gives `+0.0`,
/// `-0.0` or a value that is not finite. The evaluator is deterministic, so `a - a` is `+0.0` for
/// a finite `a` and NaN otherwise; zero times anything, zero over anything (a divisor near zero
/// gives `0.0`), sums and differences of zeros, negation, a positive constant power, and the
/// calls that keep zero (`abs`, `sqrt`, `log1p`, `sin`, `tanh`, `atan`, `asinh`, `erf`, `sign`,
/// `floor`, `round`) each give zero from zero, and NaN or zero from NaN. A sample's target is
/// finite, so such a subtree's targets are all zero on any training rows. The test is
/// sufficient, not complete: `false` says nothing.
pub(super) fn zero_wherever_finite(expr: &Expr) -> bool {
    match expr {
        Expr::Const(c) => *c == 0.0,
        Expr::Var(_) => false,
        Expr::Unary(UnOp::Neg, a) => zero_wherever_finite(a),
        Expr::Binary(op, a, b) => match op {
            BinOp::Add => zero_wherever_finite(a) && zero_wherever_finite(b),
            BinOp::Sub => same_tree(a, b) || (zero_wherever_finite(a) && zero_wherever_finite(b)),
            BinOp::Mul => zero_wherever_finite(a) || zero_wherever_finite(b),
            BinOp::Div => zero_wherever_finite(a),
            BinOp::Pow => zero_wherever_finite(a) && matches!(**b, Expr::Const(c) if c > 0.0),
        },
        Expr::Call(func, args) => {
            matches!(
                func,
                Func::Abs
                    | Func::Sqrt
                    | Func::Log1p
                    | Func::Sin
                    | Func::Tanh
                    | Func::Atan
                    | Func::Asinh
                    | Func::Erf
                    | Func::Sign
                    | Func::Floor
                    | Func::Round
            ) && args.len() == 1
                && zero_wherever_finite(&args[0])
        }
    }
}

/// Whether `expr` takes one value wherever it is finite, up to the sign of a zero: its literals
/// alone, `a - a`, a subtree [`zero_wherever_finite`], or any operation or call of such values.
/// The evaluator is deterministic, and none of its operations or calls tells `+0.0` from `-0.0`
/// in size: each gives results of equal size from either, or faults (a divisor near zero, the
/// logarithm, a negative power), so every finite target of such a subtree has one size, on any
/// training rows. The test is sufficient, not complete.
pub(super) fn one_value_wherever_finite(expr: &Expr) -> bool {
    zero_wherever_finite(expr)
        || match expr {
            Expr::Const(_) => true,
            Expr::Var(_) => false,
            Expr::Unary(UnOp::Neg, a) => one_value_wherever_finite(a),
            Expr::Binary(op, a, b) => {
                (*op == BinOp::Sub && same_tree(a, b))
                    || (one_value_wherever_finite(a) && one_value_wherever_finite(b))
            }
            Expr::Call(_, args) => args.iter().all(one_value_wherever_finite),
        }
}

/// An upper bound on the size of every value of `expr` that is not NaN, on any input:
/// `+inf` when nothing smaller is known. Literals give their size, a subtree
/// [`zero_wherever_finite`] zero, sums, differences and products their operands' bounds, each
/// rounded up (a product with a zero factor zero), and the bounded calls their own range: `sin`,
/// `cos`, `tanh`, `erf`, `sigmoid`, `sign` and `step` at most 1, `atan` at most `π/2`, `abs` its
/// operand's, `sqrt` the root of its operand's, `floor` and `round` one more than their
/// operand's, `exp` of its clamped argument, `min`, `max` and `clip` the larger of their
/// operands' or bounds'. Quotients, powers, logarithms and `asinh` are unbounded here. A sample's
/// target is finite, so no target of a subtree is larger than its bound, on any training rows.
pub(super) fn largest_magnitude(expr: &Expr) -> f64 {
    let up = |v: f64| if v == 0.0 { 0.0 } else { v.next_up() };
    if zero_wherever_finite(expr) {
        return 0.0;
    }
    match expr {
        Expr::Const(c) if c.is_nan() => f64::INFINITY,
        Expr::Const(c) => c.abs(),
        Expr::Var(_) => f64::INFINITY,
        Expr::Unary(UnOp::Neg, a) => largest_magnitude(a),
        Expr::Binary(op, a, b) => {
            let (x, y) = (largest_magnitude(a), largest_magnitude(b));
            match op {
                BinOp::Add | BinOp::Sub => up(x + y),
                // A factor of size zero makes the product zero, or NaN.
                BinOp::Mul if x == 0.0 || y == 0.0 => 0.0,
                BinOp::Mul => up(x * y),
                BinOp::Div | BinOp::Pow => f64::INFINITY,
            }
        }
        Expr::Call(func, args) => {
            let bound = |i: usize| args.get(i).map_or(f64::INFINITY, largest_magnitude);
            match func {
                Func::Sin | Func::Cos | Func::Tanh | Func::Erf | Func::Sigmoid => 1.0,
                Func::Sign | Func::Step => 1.0,
                Func::Atan => std::f64::consts::FRAC_PI_2.next_up(),
                Func::Abs => bound(0),
                Func::Sqrt => up(bound(0).sqrt()),
                Func::Floor | Func::Round => up(bound(0) + 1.0),
                // The library's `exp` is within an ulp, so two ulps cover it.
                Func::Exp => up(up(bound(0).min(30.0).exp())),
                Func::Min | Func::Max => bound(0).max(bound(1)),
                Func::Clip => bound(1).max(bound(2)),
                Func::Log | Func::Log1p | Func::Asinh => f64::INFINITY,
            }
        }
    }
}
