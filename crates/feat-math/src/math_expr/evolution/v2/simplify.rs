//! The discrete minimax polynomial of a one-variable subtree on its observed training range,
//! lowered to ordinary AST nodes, and the bounded minimax linearization built on it.
//!
//! The one caller outside the tests is [`MathExpr::linearize_v2`]: a bounded minimax
//! linearization, degree 1 only, by the owner's ruling. Under the kernel2 caps (15 nodes, depth
//! 4) a dense Horner tree of degree `n` (every coefficient nonzero) has depth at least `2n + 1`
//! (a negative leading literal adds a unary minus), so no dense higher degree can be lowered into
//! an admissible expression; [`lower`] skips zero coefficients, so a sparse one can be shallower
//! (`[0, 0, 2]` lowers to `c0*(c0*2)`, depth 3), but none is built. This is not a general
//! polynomial or rational compression. A line is only a
//! proposal: it is offered when its fit's bracket closed, it is smaller than the subtree, its own
//! error on the training samples meets the minimax bound ([`assess`]), and the profile admits the
//! whole expression with it in place. The native protected re-score of the whole program then
//! decides, never the approximation error. The bound is the fit's own largest training error,
//! with room for rounding: it checks that lowering the fit to ordinary nodes lost no precision,
//! not that the line approximates the subtree well, which only the native score judges. The fit
//! of higher degrees, and the pole-free certificate a rational denominator would need
//! (`certify`, test-only), have no caller outside the tests.
//!
//! Algorithm: the second Remez exchange on the sample abscissae (Pachón and Trefethen, BIT 49,
//! 2009), in the Chebyshev basis on `t = (2v - (a + b)) / (b - a)`.
//! - It starts from the Chebyshev extrema.
//! - Each step solves `Σ c_j T_j(t_r) + (-1)^r h = f_r` on the reference by Gaussian elimination
//!   with partial pivoting.
//! - The new reference is the alternating run maxima with `|e| ≥ |h|`: same-sign neighbours are
//!   merged, and the `n + 2` consecutive points containing the global maximum are kept.
//! - The bracket is `[levelled, max|e|]`, with `levelled` the smaller of `|h|` and the smallest
//!   `|e|` on the reference, so it never inverts. Both ends are residuals of this module's own
//!   floating-point evaluation, not certified bounds: de la Vallée Poussin bounds the discrete
//!   minimax error E* by the exact errors, so E* lies within that evaluation's rounding of the
//!   bracket, not necessarily inside it. Rounding can also put `|h|` above `max|e|`; both stop
//!   tests bound that excess too.
//! - It stops when the bracket is within `1e-12 max|e|` (`Converged`). It stops `NoiseLimited`
//!   only when the bracket is within the rounding level `64 ε max|f|`, `|h|` is at least half of
//!   `max|e|`, the fit is the best seen, and the exchange is spent: it finds no new reference, or
//!   returns this one or one run before. The next reference depends on this one alone, so every
//!   later fit would repeat one already seen. A step whose `|h|` did not rise is not enough.
//!   Otherwise a run ends `Degenerate`, `Stalled` or at the iteration cap, with the best fit seen.
//! - Every tie goes to the lowest index and every sum runs in index order, so a fit is
//!   bit-reproducible and independent of sample order and duplicates.

// B19 core; the linearization and its kernel2 caller are laneD:B19-minimax-linearize.

use super::*;
use crate::kernel::expression_search::{
    LinearFit, LinearizationEvidence, MinimaxStop, SubtreeLinearization, LINEARIZATION_DEGREE,
    LINEARIZE_FIT_REFUSED, LINEARIZE_INPUT_FAULTS, LINEARIZE_MISSES_BOUND, LINEARIZE_NON_FINITE,
    LINEARIZE_NOT_SMALLER, LINEARIZE_OPEN_BRACKET, LINEARIZE_OUT_OF_RANGE, LINEARIZE_OVER_BUDGET,
    LINEARIZE_OVER_LIMITS, LINEARIZE_PROFILE_REFUSED, LINEARIZE_UNSAMPLED,
};

/// The largest degree fitted. What can be proposed is bounded separately, by the checks of
/// [`assess`]: a dense Horner tree over a variable leaf has depth at least `2n + 1`, so under the
/// expression depth limit of 16 the largest dense degree that lowers is 7 (a sparse one, its zero
/// coefficients skipped, can be shallower).
const MAX_DEGREE: usize = 12;
/// The exchange stops after this many steps.
const MAX_ITERATIONS: usize = 32;
/// The relative width of the bracket at which the exchange stops `Converged`.
const TOLERANCE: f64 = 1e-12;
/// The deepest subdivision [`certify`] accepts.
#[cfg(test)]
const MAX_CERTIFY_DEPTH: usize = 64;

/// Why a fit stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    /// `max|e| - levelled ≤ TOLERANCE max|e|` and `|h| - max|e| ≤ TOLERANCE max|e|`.
    Converged,
    /// Both bracket measures are within `64 ε max|f|`, the rounding level of the errors, but not
    /// within `TOLERANCE max|e|`; `|h| ≥ max|e| / 2`; the fit is the best seen; and the exchange
    /// is spent: it finds no new reference, or returns this one or one run before.
    NoiseLimited,
    /// The samples lie on a polynomial of the degree, to a few ulps.
    ExactFit,
    /// The exchange returned the same reference.
    Stalled,
    IterationLimit,
    /// Fewer than `n + 2` alternating extrema, or a singular reference system.
    Degenerate,
}

impl Stop {
    /// The public name of this stop, for records.
    fn public(self) -> MinimaxStop {
        match self {
            Self::Converged => MinimaxStop::Converged,
            Self::NoiseLimited => MinimaxStop::NoiseLimited,
            Self::ExactFit => MinimaxStop::ExactFit,
            Self::Stalled => MinimaxStop::Stalled,
            Self::IterationLimit => MinimaxStop::IterationLimit,
            Self::Degenerate => MinimaxStop::Degenerate,
        }
    }
}

/// A discrete minimax fit.
#[derive(Clone, Debug, PartialEq)]
struct Minimax {
    /// The fit interval: the smallest and largest sample abscissa. The error is controlled only at
    /// the samples.
    lo: f64,
    hi: f64,
    /// Chebyshev coefficients in `t`.
    chebyshev: Vec<f64>,
    /// The smaller of `|h|` and the smallest `|e|` on the final reference, never above
    /// `max_error`. In exact arithmetic it would bound the discrete minimax error from below (de
    /// la Vallée Poussin); as computed it does so only up to rounding.
    levelled: f64,
    /// The largest computed error over the samples, by Clenshaw at the rounded `t`. Not a
    /// certified upper bound: the exact errors, and the discrete minimax error with them, can
    /// exceed it by up to the rounding of that evaluation.
    max_error: f64,
    /// The exchange step that produced this fit.
    iterations: usize,
    /// The exchange steps run, at most `MAX_ITERATIONS`.
    steps: usize,
    stop: Stop,
    /// The final reference, as indices into the sorted unique samples.
    reference: Vec<usize>,
}

impl Minimax {
    #[cfg(test)]
    fn t(&self, v: f64) -> f64 {
        (2.0 * v - (self.lo + self.hi)) / (self.hi - self.lo)
    }

    #[cfg(test)]
    fn eval(&self, v: f64) -> f64 {
        clenshaw(&self.chebyshev, self.t(v))
    }

    /// Monomial coefficients in `v` (ascending powers). This conversion is ill-conditioned when
    /// the interval sits far from 0; the native re-score decides.
    fn monomial(&self) -> Vec<f64> {
        let alpha = 2.0 / (self.hi - self.lo);
        let beta = -(self.lo + self.hi) / (self.hi - self.lo);
        let n = self.chebyshev.len();
        let mut out = vec![0.0; n];
        let mut previous = vec![1.0];
        add_scaled(&mut out, &previous, self.chebyshev[0]);
        if n == 1 {
            return out;
        }
        let mut current = vec![beta, alpha];
        add_scaled(&mut out, &current, self.chebyshev[1]);
        for &c in &self.chebyshev[2..] {
            // T_{k+1} = 2 (α v + β) T_k - T_{k-1}.
            let mut next = vec![0.0; current.len() + 1];
            for (power, &a) in current.iter().enumerate() {
                next[power] += 2.0 * beta * a;
                next[power + 1] += 2.0 * alpha * a;
            }
            for (power, &a) in previous.iter().enumerate() {
                next[power] -= a;
            }
            add_scaled(&mut out, &next, c);
            previous = current;
            current = next;
        }
        out
    }
}

fn add_scaled(out: &mut [f64], polynomial: &[f64], scale: f64) {
    for (o, &a) in out.iter_mut().zip(polynomial) {
        *o += scale * a;
    }
}

/// `Σ c_k T_k(t)` by Clenshaw's recurrence.
fn clenshaw(c: &[f64], t: f64) -> f64 {
    let (mut b1, mut b2) = (0.0f64, 0.0f64);
    for &ck in c.iter().skip(1).rev() {
        let b0 = ck + 2.0 * t * b1 - b2;
        b2 = b1;
        b1 = b0;
    }
    c[0] + t * b1 - b2
}

/// `T_0(t) .. T_n(t)`.
fn chebyshev_row(t: f64, n: usize) -> Vec<f64> {
    let mut row = Vec::with_capacity(n + 1);
    row.push(1.0);
    if n >= 1 {
        row.push(t);
    }
    for k in 2..=n {
        row.push(2.0 * t * row[k - 1] - row[k - 2]);
    }
    row
}

/// `A x = b` by Gaussian elimination with partial pivoting (pivot ties go to the lowest row), or
/// `None` when a pivot is zero or a result is not finite.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let m = b.len();
    for k in 0..m {
        let pivot = (k..m).fold(k, |p, r| if a[r][k].abs() > a[p][k].abs() { r } else { p });
        if a[pivot][k] == 0.0 || !a[pivot][k].is_finite() {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        let (top, bottom) = a.split_at_mut(k + 1);
        let pivot_row = &top[k];
        let (b_top, b_bottom) = b.split_at_mut(k + 1);
        for (row, rhs) in bottom.iter_mut().zip(b_bottom.iter_mut()) {
            let factor = row[k] / pivot_row[k];
            if factor == 0.0 {
                continue;
            }
            for (value, &above) in row[k..].iter_mut().zip(&pivot_row[k..]) {
                *value -= factor * above;
            }
            *rhs -= factor * b_top[k];
        }
    }
    let mut x = vec![0.0; m];
    for k in (0..m).rev() {
        let mut s = b[k];
        for (coefficient, known) in a[k][k + 1..].iter().zip(&x[k + 1..]) {
            s -= coefficient * known;
        }
        x[k] = s / a[k][k];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// The sorted unique samples, or why the fit cannot start. A non-finite sample is refused, and so
/// is one above `MAX_ABS`, the evaluator's output range and the only range [`samples`] produces:
/// near `f64::MAX` the map to `t` and the Clenshaw sums overflow.
fn prepare(samples: &[(f64, f64)]) -> Result<Vec<(f64, f64)>, &'static str> {
    if samples
        .iter()
        .any(|(v, f)| !v.is_finite() || !f.is_finite())
    {
        return Err("a sample is not finite");
    }
    if samples
        .iter()
        .any(|(v, f)| v.abs() > MAX_ABS || f.abs() > MAX_ABS)
    {
        return Err(OUTSIDE_RANGE);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut unique: Vec<(f64, f64)> = Vec::with_capacity(sorted.len());
    for (v, f) in sorted {
        match unique.last() {
            // `==` so -0.0 and 0.0 are one abscissa, and ±0 targets are one target (an odd
            // subtree gives f(-0.0) = -0.0). The sort keeps the first of a merged pair.
            Some(&(lv, lf)) if lv == v => {
                if lf != f {
                    return Err(TWO_TARGETS);
                }
            }
            _ => unique.push((v, f)),
        }
    }
    Ok(unique)
}

/// The fit's refusal of too few distinct abscissae. With [`EMPTY_INTERVAL`], it depends on the
/// variable's values alone, never on the subtree's.
const FEW_ABSCISSAE: &str = "fewer than degree + 2 distinct abscissae";
/// [`prepare`]'s refusal of a sample past the evaluator's range.
const OUTSIDE_RANGE: &str = "a sample is outside the evaluator's range";
/// [`prepare`]'s refusal of one abscissa with two targets (merged signed zeros that differ).
const TWO_TARGETS: &str = "one abscissa has two targets";
/// The fit's refusal when its first reference system is singular.
const SINGULAR_FIRST: &str = "singular first reference";
/// The fit's refusal when its first fit's errors are not finite.
const NONFINITE_ERROR: &str = "an error is not finite";
/// Every reason the fit can refuse the samples [`samples`] gives for a line: [`prepare`]'s and
/// [`minimax`]'s own, less four. [`samples`] gives only fault-free finite samples, and the
/// evaluator raises a fault for every value it brings back into its range, so neither `a sample
/// is not finite` nor [`OUTSIDE_RANGE`] occurs; a line's degree is under the cap; and three
/// distinct abscissae within that range span an interval that is neither empty nor overflowing,
/// so [`EMPTY_INTERVAL`] does not occur either.
const FIT_REFUSALS: [&str; 4] = [TWO_TARGETS, FEW_ABSCISSAE, SINGULAR_FIRST, NONFINITE_ERROR];
/// The fit's refusal of an interval that is empty or overflows.
const EMPTY_INTERVAL: &str = "empty or overflowing interval";

/// The discrete minimax polynomial of `degree` on `samples` of `(v, f(v))`.
fn minimax(samples: &[(f64, f64)], degree: usize) -> Result<Minimax, &'static str> {
    if degree > MAX_DEGREE {
        return Err("degree above the cap");
    }
    let unique = prepare(samples)?;
    let n = degree;
    let count = unique.len();
    if count < n + 2 {
        return Err(FEW_ABSCISSAE);
    }
    let (lo, hi) = (unique[0].0, unique[count - 1].0);
    if lo >= hi || !(hi - lo).is_finite() {
        return Err(EMPTY_INTERVAL);
    }
    let t: Vec<f64> = unique
        .iter()
        .map(|(v, _)| (2.0 * v - (lo + hi)) / (hi - lo))
        .collect();
    let f: Vec<f64> = unique.iter().map(|(_, f)| *f).collect();
    let f_max = f.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    // The Chebyshev extrema, each at its nearest unused sample (ties to the lower index).
    let mut reference = Vec::with_capacity(n + 2);
    for j in 0..n + 2 {
        let target = -(std::f64::consts::PI * j as f64 / (n + 1) as f64).cos();
        let floor = reference.last().map_or(0, |&r: &usize| r + 1);
        let ceiling = count - (n + 2 - j);
        let nearest = (floor..=ceiling)
            .min_by(|&a, &b| {
                (t[a] - target)
                    .abs()
                    .total_cmp(&(t[b] - target).abs())
                    .then(a.cmp(&b))
            })
            .expect("a nonempty range");
        reference.push(nearest);
    }
    /// The best fit so far, stopped for `stop` after `steps` steps.
    fn ending(best: Option<Minimax>, stop: Stop, steps: usize) -> Option<Minimax> {
        best.map(|b| Minimax { stop, steps, ..b })
    }
    let noise = 64.0 * f64::EPSILON * f_max;
    let mut best: Option<Minimax> = None;
    // Every reference run so far. The next reference is a function of the current one alone, so
    // an exchange that returns one of these can only repeat fits already seen.
    let mut visited: Vec<Vec<usize>> = Vec::with_capacity(MAX_ITERATIONS);
    for iteration in 1..=MAX_ITERATIONS {
        let rows: Vec<Vec<f64>> = reference
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                let mut row = chebyshev_row(t[r], n);
                row.push(if i % 2 == 0 { 1.0 } else { -1.0 });
                row
            })
            .collect();
        let rhs: Vec<f64> = reference.iter().map(|&r| f[r]).collect();
        let Some(x) = solve(rows, rhs) else {
            return ending(best, Stop::Degenerate, iteration).ok_or(SINGULAR_FIRST);
        };
        let (chebyshev, h) = (x[..=n].to_vec(), x[n + 1].abs());
        let errors: Vec<f64> = (0..count)
            .map(|i| f[i] - clenshaw(&chebyshev, t[i]))
            .collect();
        // Overflow in the polynomial leaves no usable error, and a NaN would slip past the
        // maximum below.
        if errors.iter().any(|e| !e.is_finite()) {
            return ending(best, Stop::Degenerate, iteration).ok_or(NONFINITE_ERROR);
        }
        let (mut global, mut max_error) = (0usize, 0.0f64);
        for (i, e) in errors.iter().enumerate() {
            if e.abs() > max_error {
                max_error = e.abs();
                global = i;
            }
        }
        let smallest = reference
            .iter()
            .map(|&r| errors[r].abs())
            .fold(f64::INFINITY, f64::min);
        let fit = Minimax {
            lo,
            hi,
            chebyshev,
            levelled: h.min(smallest),
            max_error,
            iterations: iteration,
            steps: iteration,
            stop: Stop::IterationLimit,
            reference: reference.clone(),
        };
        let improved = best.as_ref().is_none_or(|b| fit.max_error < b.max_error);
        if improved {
            best = Some(fit.clone());
        }
        if max_error <= 4.0 * f64::EPSILON * f_max {
            return Ok(Minimax {
                stop: Stop::ExactFit,
                ..fit
            });
        }
        // The bracket width, and how far rounding put |h| above max|e| (an inverted bracket).
        let (width, excess) = (max_error - fit.levelled, h - max_error);
        if width <= TOLERANCE * max_error && excess <= TOLERANCE * max_error {
            return Ok(Minimax {
                stop: Stop::Converged,
                ..fit
            });
        }
        // Within the rounding level of the errors the bracket closes no further, but a fit whose
        // |h| is far below max|e| is not done, and neither is one the exchange can still improve.
        // A step whose |h| did not rise does not show that the next cannot raise it, so the noise
        // limit needs a spent exchange: no new reference, this one, or one run before.
        let noisy = width <= noise && excess <= noise && h >= max_error / 2.0 && improved;
        let next = exchange(&errors, h, global, n + 2)
            .or_else(|| single_exchange(&errors, &reference, global));
        visited.push(reference.clone());
        if noisy && next.as_ref().is_none_or(|next| visited.contains(next)) {
            return Ok(Minimax {
                stop: Stop::NoiseLimited,
                ..fit
            });
        }
        // Otherwise no new reference, or this one, is a degenerate or stalled run, and a reference
        // run before only repeats fits already seen until the cap.
        let Some(next) = next else {
            return Ok(ending(best, Stop::Degenerate, iteration).expect("set above"));
        };
        if next == reference {
            return Ok(ending(best, Stop::Stalled, iteration).expect("set above"));
        }
        reference = next;
    }
    Ok(ending(best, Stop::IterationLimit, MAX_ITERATIONS).expect("at least one step ran"))
}

/// The next reference: the maximum of each run of same-sign errors, those with `|e| ≥ |h|`,
/// same-sign neighbours merged to the larger, then `size` consecutive points containing the
/// global maximum. `None` when fewer than `size` alternate.
fn exchange(errors: &[f64], h: f64, global: usize, size: usize) -> Option<Vec<usize>> {
    let sign = |e: f64| (e > 0.0) as i8 - (e < 0.0) as i8;
    let mut extrema: Vec<usize> = Vec::new();
    let mut run: Option<(i8, usize)> = None;
    for (i, &e) in errors.iter().enumerate() {
        let s = sign(e);
        if s == 0 {
            continue;
        }
        match run {
            Some((rs, at)) if rs == s => {
                if e.abs() > errors[at].abs() {
                    run = Some((rs, i));
                }
            }
            Some((_, at)) => {
                extrema.push(at);
                run = Some((s, i));
            }
            None => run = Some((s, i)),
        }
    }
    if let Some((_, at)) = run {
        extrema.push(at);
    }
    // A reference point's error is |h| only up to rounding, so the bar sits a hair below it. On
    // an alternating set at or above it the next |h| is, in exact arithmetic, at least the bar (de
    // la Vallée Poussin); rounding can still lower it, so no stop rests on one step's |h|.
    let bar = h * (1.0 - 1e-9);
    let mut kept: Vec<usize> = Vec::new();
    for i in extrema
        .into_iter()
        .filter(|&i| errors[i].abs() >= bar || i == global)
    {
        match kept.last() {
            Some(&last) if sign(errors[last]) == sign(errors[i]) => {
                if errors[i].abs() > errors[last].abs() {
                    *kept.last_mut().expect("nonempty") = i;
                }
            }
            _ => kept.push(i),
        }
    }
    if kept.len() < size {
        return None;
    }
    let position = kept.iter().position(|&i| i == global)?;
    let start = position.saturating_sub(size - 1).min(kept.len() - size);
    Some(kept[start..start + size].to_vec())
}

/// The first Remez algorithm's single-point exchange (Pachón and Trefethen §2): the global
/// maximum replaces the reference point beside it whose error has the same sign, or enters at an
/// end, shifting the reference, so the signs keep alternating. `None` when the global maximum is
/// already a reference point.
fn single_exchange(errors: &[f64], reference: &[usize], global: usize) -> Option<Vec<usize>> {
    if reference.contains(&global) {
        return None;
    }
    let same = |a: usize, b: usize| (errors[a] > 0.0) == (errors[b] > 0.0);
    let mut next = reference.to_vec();
    let last = next.len() - 1;
    if global < next[0] {
        if same(global, next[0]) {
            next[0] = global;
        } else {
            next.pop();
            next.insert(0, global);
        }
    } else if global > next[last] {
        if same(global, next[last]) {
            next[last] = global;
        } else {
            next.remove(0);
            next.push(global);
        }
    } else {
        let j = next.iter().position(|&r| r > global)? - 1;
        if same(global, next[j]) {
            next[j] = global;
        } else {
            next[j + 1] = global;
        }
    }
    Some(next)
}

// ---------------------------------------------------------------------------------------------
// Lowering.

/// `Σ a_k x^k` in Horner form over the leaf `x`, as ordinary AST nodes: `Add`/`Sub` of finite
/// literals and `Mul` by `x`. A negative non-leading coefficient becomes a `Sub`, and a zero one
/// is skipped. No `Pow` (its precision is unspecified) and no new node type.
fn lower(coefficients: &[f64], x: &Expr) -> Option<Expr> {
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
fn horner(coefficients: &[f64], v: f64) -> f64 {
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
fn lowered_error(coefficients: &[f64], unique: &[(f64, f64)]) -> Option<f64> {
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
struct Proposal {
    tree: Expr,
    /// What the proposal is held to: the lowered tree's error, never the fit's alone.
    lowered_error: f64,
    nodes: usize,
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
fn propose(
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
fn largest(unique: &[(f64, f64)]) -> f64 {
    unique.iter().map(|(_, f)| f.abs()).fold(0.0f64, f64::max)
}

/// The minimax bound a lowered tree is held to on its samples: the fit's own largest error, a
/// relative `1e-6` over it, and the rounding level `64 ε max|f|`. It measures the precision lost
/// in lowering the fit to ordinary nodes, not the line's quality: a poor fit has a large
/// `max_error` and a bound to match. Whether the line is any good is for the native score of the
/// whole program to decide.
fn allowance(max_error: f64, f_max: f64) -> f64 {
    max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * f_max
}

/// `x` is not a variable leaf, so the samples do not give the lowered tree's values.
const NOT_A_LEAF: &str = "the lowered variable is not a leaf";

/// A refused lowering: the first check that failed, and what was measured before it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Refused {
    reason: &'static str,
    nodes: Option<usize>,
    lowered_error: Option<f64>,
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
fn assess(
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

/// Every variable `expr` reads, in first-read order.
fn variables(expr: &Expr, out: &mut Vec<Var>) {
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
fn samples(subtree: &MathExpr, rows: &[Vec<f64>]) -> Option<(Var, Vec<(f64, f64)>)> {
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
fn target_at(subtree: &Expr, var: Var, v: f64) -> Option<f64> {
    let mut tree = subtree.clone();
    read_as_element(&mut tree, var);
    let (f, faults) = MathExpr::from_ast(tree).ok()?.eval_element_traced(v, &[]);
    (faults.is_empty() && f.is_finite()).then_some(f)
}

/// `expr` with each read of `var` made a read of the element `x`.
fn read_as_element(expr: &mut Expr, var: Var) {
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
fn linearizable(root: &Expr, max_depth: usize) -> Vec<Vec<usize>> {
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
type Measure = Box<dyn FnOnce(&[f64]) -> Option<f64>>;

/// Where a linearization's training measurements come from: the training rows themselves, or a
/// record of what was measured on them, by subtree index.
enum Source<'a> {
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
fn consistent(fit: &LinearFit) -> bool {
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
    // first step; its coefficients are zero, or NaN where the conversion scales zero by infinity.
    let zero_targets = fit.max_abs != 0.0
        || (fit.stop == MinimaxStop::ExactFit
            && fit.iterations == 1
            && fit.steps == 1
            && fit.max_error == 0.0
            && fit.coefficients.iter().all(|c| c.is_nan() || *c == 0.0));
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
struct Shared {
    /// Each recorded subtree with its fit, its line's error, and its refusal when it has no fit.
    subtrees: Vec<(Expr, SampleEvidence)>,
    /// Per variable: the training range its fits record, and whether a fit refused its values.
    variables: std::collections::BTreeMap<Var, (Option<(u64, u64)>, bool)>,
    /// Whether the expression's input rows could not be built, as its first record says.
    input_faults: Option<bool>,
}

/// The fit, the line's measured error, and the refusal of a subtree without a fit, with every
/// number as its exact bits. A record compares with another bit for bit, NaN included: a fit
/// refused for coefficients that are not finite is published without them and read back as NaN,
/// which never equals itself, while two records of the same subtree hold the same bits.
type SampleEvidence = (Option<FitBits>, Option<u64>, Option<String>);

/// A fit as exact bits: its stop, coefficients, range, bracket and largest `|f|`, and its steps.
type FitBits = (MinimaxStop, [u64; 2], [u64; 5], [usize; 2]);

fn fit_bits(fit: &LinearFit) -> FitBits {
    (
        fit.stop,
        fit.coefficients.map(f64::to_bits),
        [fit.lo, fit.hi, fit.levelled, fit.max_error, fit.max_abs].map(f64::to_bits),
        [fit.iterations, fit.steps],
    )
}

impl Shared {
    fn check(&mut self, subtree: &Expr, var: Var, evidence: &LinearizationEvidence) -> Result<()> {
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
fn same_tree(a: &Expr, b: &Expr) -> bool {
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
fn zero_wherever_finite(expr: &Expr) -> bool {
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
fn one_value_wherever_finite(expr: &Expr) -> bool {
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
fn largest_magnitude(expr: &Expr) -> f64 {
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

/// Why a public linearization entry refuses a tree outside the structure limits, before any copy
/// of it.
const STRUCTURE_REFUSAL: &str =
    "the linearization refuses a tree outside the structure limits before it is copied";

/// The bounded minimax linearization of `expression`'s one-variable subtrees; see
/// [`MathExpr::linearize_v2`]. `expression` may be any tree, one decoded from JSON included, and
/// the copy and the admission's normalization both recurse, so its borrowed structure check
/// (`from_ast`'s limits, walked with an explicit stack) comes first.
fn linearize(
    expression: &MathExpr,
    profile: &SearchProfile,
    scope: ExpressionScope,
    mut source: Source<'_>,
) -> Result<Vec<SubtreeLinearization>> {
    expression
        .check_structure(MAX_NODES, MAX_DEPTH)
        .map_err(|error| error.context(STRUCTURE_REFUSAL))?;
    let search = Search::new(profile, scope)?;
    let input = search.finish(expression.root.clone())?;
    let total = count_nodes(&input.root);
    let mut out = Vec::new();
    let mut shared = Shared::default();
    for (index, path) in linearizable(&input.root, search.scope.max_depth)
        .into_iter()
        .enumerate()
    {
        let node = at(&input.root, &path);
        let nodes = count_nodes(node);
        let mut found = Vec::new();
        variables(node, &mut found);
        let x = Expr::Var(found[0]);
        let mut record = SubtreeLinearization {
            path: path.clone(),
            subtree: MathExpr::from_ast(node.clone())?,
            nodes,
            fit: None,
            line_nodes: None,
            lowered_error: None,
            expression: None,
            refusal: None,
        };
        // A recorded line in the evaluator's range at both sampled ends, with no measured error.
        let mut unmeasured_in_range = false;
        // The fit, and how the line's own error over the samples is measured.
        let (fit, measured): (LinearFit, Measure) = match &mut source {
            Source::Rows(rows) => {
                let Some((_, points)) = samples(&record.subtree, rows) else {
                    record.refusal = Some(LINEARIZE_UNSAMPLED.into());
                    out.push(record);
                    continue;
                };
                let fit = match minimax(&points, LINEARIZATION_DEGREE) {
                    Ok(fit) => fit,
                    Err(reason) => {
                        record.refusal = Some(format!("{LINEARIZE_FIT_REFUSED}: {reason}"));
                        out.push(record);
                        continue;
                    }
                };
                let unique = prepare(&points).map_err(anyhow::Error::msg)?;
                let line = fit.monomial();
                let public = LinearFit {
                    stop: fit.stop.public(),
                    coefficients: [line[0], line[1]],
                    lo: fit.lo,
                    hi: fit.hi,
                    levelled: fit.levelled,
                    max_error: fit.max_error,
                    max_abs: largest(&unique),
                    iterations: fit.iterations,
                    steps: fit.steps,
                };
                (
                    public,
                    Box::new(move |coefficients: &[f64]| lowered_error(coefficients, &unique)),
                )
            }
            Source::Recorded(recorded) => {
                let evidence = recorded(index)?;
                shared.check(node, found[0], &evidence)?;
                let Some(fit) = evidence.fit else {
                    anyhow::ensure!(
                        evidence.lowered_error.is_none() && evidence.refusal.is_some(),
                        "a linearization without a fit names no refusal"
                    );
                    // The live path refuses a subtree before any fit only as unsampled, for a
                    // reason the fit itself gives, or, from its caller, because the input rows
                    // could not be built (every subtree of the expression or none: `Shared`).
                    let refusal = evidence.refusal.as_deref().unwrap_or_default();
                    let reason = refusal
                        .strip_prefix(LINEARIZE_FIT_REFUSED)
                        .and_then(|r| r.strip_prefix(": "));
                    anyhow::ensure!(
                        refusal == LINEARIZE_UNSAMPLED
                            || refusal == LINEARIZE_INPUT_FAULTS
                            || reason.is_some_and(|r| FIT_REFUSALS.contains(&r)),
                        "a linearization without a fit names a refusal the fit never gives"
                    );
                    // The element `x` faults on every row, so a subtree of it is unsampled, or
                    // has too few values when there are no rows at all.
                    anyhow::ensure!(
                        found[0] != Var::X
                            || refusal == LINEARIZE_UNSAMPLED
                            || refusal == LINEARIZE_INPUT_FAULTS
                            || reason == Some(FEW_ABSCISSAE),
                        "a subtree of the element x names a refusal its samples never give"
                    );
                    record.refusal = evidence.refusal;
                    out.push(record);
                    continue;
                };
                anyhow::ensure!(consistent(&fit), "a recorded fit disagrees with its stop");
                // The live samples read each row's variable, and the element `x` faults on every
                // row, so a subtree of `x` is never sampled and has no fit.
                anyhow::ensure!(
                    found[0] != Var::X,
                    "a recorded fit names a subtree of the element x, which no training row samples"
                );
                // A subtree that is zero wherever it is finite gives every sample a zero target,
                // so its fit's largest `|f|` is zero, and `consistent` then holds the fit to the
                // zero polynomial, exact at the first step.
                anyhow::ensure!(
                    fit.max_abs == 0.0 || !zero_wherever_finite(node),
                    "a recorded fit gives a subtree that is zero on every sample a nonzero target"
                );
                // A row's L1 and L2 norms and its standard deviation are never negative, so a
                // fit of one has a range that is not either.
                anyhow::ensure!(
                    !matches!(found[0], Var::RowL1 | Var::RowL2 | Var::RowStd) || fit.lo >= 0.0,
                    "a recorded fit's range holds values its variable never takes"
                );
                // No target is larger than every value the subtree can take.
                anyhow::ensure!(
                    fit.max_abs <= largest_magnitude(node),
                    "a recorded fit's max|f| exceeds every value its subtree can take"
                );
                // `lo` and `hi` are sampled values of the subtree's one variable, so its targets
                // there follow from the subtree itself: fault-free, finite, and within `max|f|`.
                let mut targets = [0.0f64; 2];
                for (target, end) in targets.iter_mut().zip([fit.lo, fit.hi]) {
                    *target = target_at(node, found[0], end)
                        .filter(|f| f.abs() <= fit.max_abs)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "a recorded fit's sampled ends give targets its samples cannot have"
                            )
                        })?;
                }
                // A subtree with one value on every sample has targets of the size its target at
                // `lo` shows, so `max|f|` is that size. The first reference's solve, whose first
                // column is all ones, then gives that value at once with a zero slope and no
                // error, so the fit is exact at the first step, its coefficients that value and
                // zero (or NaN where the conversion scales zero by infinity); a singular first
                // solve gives no fit.
                let one_value = one_value_wherever_finite(node);
                if one_value {
                    let [a0, a1] = fit.coefficients;
                    anyhow::ensure!(
                        fit.max_abs == targets[0].abs()
                            && fit.stop == MinimaxStop::ExactFit
                            && fit.iterations == 1
                            && fit.steps == 1
                            && fit.max_error == 0.0
                            && (a0.is_nan() || a0 == targets[0])
                            && (a1.is_nan() || a1 == 0.0),
                        "a recorded fit of a subtree with one value on every sample is not that value's exact fit"
                    );
                }
                let lowered = evidence.lowered_error;
                // A line that can be lowered and is in the evaluator's range at both sampled ends
                // is in it at every sample between them, so the live measurement gives its error.
                let lowerable = fit.coefficients.iter().all(|c| c.is_finite());
                let ends = [fit.lo, fit.hi].map(|end| horner(&fit.coefficients, end));
                unmeasured_in_range = lowered.is_none()
                    && lowerable
                    && ends.iter().all(|v| v.is_finite() && v.abs() <= MAX_ABS);
                // The line's error is its largest absolute residual over the samples: a recorded
                // one that is negative, negative zero or not finite cannot have been measured.
                anyhow::ensure!(
                    lowered.is_none_or(|error| error.is_finite() && error.is_sign_positive()),
                    "a recorded line error is not a finite nonnegative residual"
                );
                if let Some(error) = lowered {
                    // The live measurement keeps a line only while it is in the evaluator's range
                    // at every sample, `lo` and `hi` are samples, and every target is within
                    // `max|f|`, which one of them attains. So every residual is at most
                    // `MAX_ABS + max|f|`, and the one at either end is `|p - f|` against that
                    // end's own target, which is within `max|f|`, so it is also at least
                    // `|p| - max|f|`. The line's computed value, a rounded product by the slope
                    // and then a rounded sum with the intercept, is monotone in the abscissa, so
                    // between the samples it lies between its values at the ends: with `M` the
                    // larger of their sizes, every residual is at most `M + max|f|`, and the one
                    // at the largest target at least `max|f| - M`. Rounding is monotone, so each
                    // bound holds for the computed values exactly, with no allowance. A line whose
                    // coefficients are not finite is refused before it is measured, as for every
                    // refusal before the measurement, so its ends are not read.
                    anyhow::ensure!(
                        error <= MAX_ABS + fit.max_abs,
                        "a recorded line error exceeds what the evaluator's range allows"
                    );
                    if lowerable {
                        let (mut largest_end, mut largest_residual) = (0.0f64, 0.0f64);
                        for (value, target) in ends.into_iter().zip(targets) {
                            anyhow::ensure!(
                                value.is_finite() && value.abs() <= MAX_ABS,
                                "a recorded line error names a line outside the evaluator's range at a sampled end"
                            );
                            // The residual at the sampled end itself, as the measurement takes it.
                            anyhow::ensure!(
                                error >= (value - target).abs(),
                                "a recorded line error is smaller than the line's residual at a sampled end's target"
                            );
                            largest_end = largest_end.max(value.abs());
                            largest_residual = largest_residual.max((value - target).abs());
                        }
                        // Against one value, the residual of a line monotone between its ends is
                        // largest at an end, and the measurement takes the ends: the error is
                        // exactly the larger end residual.
                        anyhow::ensure!(
                            !one_value || error == largest_residual,
                            "a recorded line error is not the larger end residual of a line against its subtree's one value"
                        );
                        anyhow::ensure!(
                            error <= largest_end + fit.max_abs,
                            "a recorded line error exceeds what its line and samples allow"
                        );
                        anyhow::ensure!(
                            error >= fit.max_abs - largest_end,
                            "a recorded line error is smaller than the residual at the largest target"
                        );
                    }
                }
                (fit, Box::new(move |_: &[f64]| lowered))
            }
        };
        record.fit = Some(fit);
        if !fit.stop.closed() {
            record.refusal = Some(format!("{LINEARIZE_OPEN_BRACKET}: {}", fit.stop.as_str()));
            out.push(record);
            continue;
        }
        // The node limit the rest of the expression leaves the line.
        let budget = search.scope.max_nodes - (total - nodes);
        match assess(
            &fit.coefficients,
            fit.max_error,
            fit.max_abs,
            &x,
            node,
            budget,
            measured,
        ) {
            Err(refused) => {
                // The measurement refuses a line as out of range only when some sample leaves the
                // range, which a line in range at both sampled ends cannot.
                anyhow::ensure!(
                    !(unmeasured_in_range && refused.reason == LINEARIZE_OUT_OF_RANGE),
                    "a recorded line with no measured error is in the evaluator's range at both sampled ends"
                );
                record.line_nodes = refused.nodes;
                record.lowered_error = refused.lowered_error;
                record.refusal = Some(refused.reason.into());
            }
            Ok(proposal) => {
                record.line_nodes = Some(proposal.nodes);
                record.lowered_error = Some(proposal.lowered_error);
                let mut root = input.root.clone();
                replace(&mut root, &path, proposal.tree);
                match search.finish(root) {
                    Ok(expression) => record.expression = Some(expression),
                    Err(error) => {
                        record.refusal = Some(format!("{LINEARIZE_PROFILE_REFUSED}: {error:#}"))
                    }
                }
            }
        }
        out.push(record);
    }
    Ok(out)
}

impl LinearFit {
    /// The minimax bound a line is held to on the training samples: `max_error (1 + 1e-6) + 64 ε
    /// max_abs`. The line's own largest error over those samples, in its own evaluation order,
    /// must not exceed it. Both are floating-point residuals, not certified bounds. The bound is
    /// the fit's own training error with room for rounding, so it catches a lowering that lost
    /// precision, not a poor line: the native score of the whole program judges the line.
    pub fn bound(&self) -> f64 {
        allowance(self.max_error, self.max_abs)
    }
}

impl MathExpr {
    /// The paths of the subtrees [`MathExpr::linearize_v2`] considers, in its order, from the
    /// expression's text alone: each reads exactly one variable, has more than three nodes, and
    /// sits where a line of three nodes and depth two fits the scope's depth limit. A tree outside
    /// the structure limits is refused before anything copies or walks it.
    pub fn linearizable_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
    ) -> Result<Vec<Vec<usize>>> {
        self.check_structure(MAX_NODES, MAX_DEPTH)
            .map_err(|error| error.context(STRUCTURE_REFUSAL))?;
        let search = Search::new(profile, scope)?;
        let input = search.finish(self.root.clone())?;
        Ok(linearizable(&input.root, search.scope.max_depth))
    }

    /// A bounded minimax linearization, degree 1 only, of the expression's one-variable subtrees
    /// on the given training rows. It draws no random number, and reads no row but `rows`.
    ///
    /// `self` must pass the same admission as every search edit (`Search::finish`: the pool, the
    /// caps and the negation rule), or this is an error. For each subtree of
    /// [`MathExpr::linearizable_v2`], in that order:
    /// 1. it is sampled on `rows` as `(x, f(x))` pairs of its variable and its value; a row that
    ///    faults, or gives a value that is not finite, refuses the subtree;
    /// 2. the discrete minimax line on those samples is fitted; a fit that refuses its samples
    ///    (fewer than three distinct values of the variable, say) refuses the subtree;
    /// 3. a fit whose bracket did not close ([`MinimaxStop::closed`]) refuses it: its error need
    ///    not be the minimax error;
    /// 4. the line is lowered to ordinary nodes, `x * a1 + a0`, and refused unless it is smaller
    ///    than the subtree, within the node limit the rest of the expression leaves it, and its
    ///    own largest error over the samples, in its own evaluation order, is within
    ///    [`LinearFit::bound`];
    /// 5. the line replaces the subtree, and the profile must admit the whole expression.
    ///
    /// Every considered subtree is returned, with the expression or the refusal. The minimax
    /// bound holds only on the training samples: off that range the line can be far from the
    /// subtree, so the caller re-scores each expression natively on its own data, and that score
    /// decides, never the line's error.
    pub fn linearize_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        rows: &[Vec<f64>],
    ) -> Result<Vec<SubtreeLinearization>> {
        linearize(self, profile, scope, Source::Rows(rows))
    }

    /// What [`MathExpr::linearize_v2`] returns, re-derived from a record of what it measured on
    /// the training rows: `recorded(i)` gives the `i`-th considered subtree's
    /// [`LinearizationEvidence`]. The subtrees, the bracket rule, the lowered line and its size,
    /// the bound, every refusal after the fit, and the admitted expression are recomputed. The
    /// fit itself and the line's measured error are taken as recorded, checked only against the
    /// fit's stop and the evaluator's range, the subtree's own targets at the sampled ends, what the
    /// subtree's structure fixes without the rows (a `max|f|` within the subtree's bound; for a
    /// subtree that is zero wherever it is finite, a zero `max|f|`; for one with a single value on
    /// every sample, that value's exact first-step fit and a line error equal to the larger end
    /// residual; a range within its variable's values; no fit, and only its own refusals, for a
    /// subtree of the element `x`), the error as a finite nonnegative residual within what the recorded line and
    /// samples allow (at most `1e12 + max|f|`; for a line that can be lowered, at least its
    /// residual at either sampled end and against that end's target, and between `max|f| - M` and
    /// `M + max|f|`, with `M` its larger size there; present whenever such a line in range at both
    /// ends reaches the measurement), a refusal before any fit as unsampled, one the fit gives, or
    /// [`LINEARIZE_INPUT_FAULTS`] for every subtree of the expression, and against the
    /// expression's other records (a repeated subtree records the same samples, bit for bit; the
    /// fits of one variable share its training range); without the rows they cannot be
    /// recomputed. A subtree with no recorded fit keeps its recorded refusal.
    pub fn linearize_recorded_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        recorded: &mut dyn FnMut(usize) -> Result<LinearizationEvidence>,
    ) -> Result<Vec<SubtreeLinearization>> {
        linearize(self, profile, scope, Source::Recorded(recorded))
    }
}

// ---------------------------------------------------------------------------------------------
// The pole-free certificate for a rational proposal's denominator. No proposal uses a
// denominator, so the certificate is compiled for its tests only.

/// A closed interval of reals, widened outward by one ulp after every operation.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Interval {
    lo: f64,
    hi: f64,
}

#[cfg(test)]
impl Interval {
    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    fn add(self, o: Self) -> Self {
        Self {
            lo: (self.lo + o.lo).next_down(),
            hi: (self.hi + o.hi).next_up(),
        }
    }

    fn sub(self, o: Self) -> Self {
        Self {
            lo: (self.lo - o.hi).next_down(),
            hi: (self.hi - o.lo).next_up(),
        }
    }

    fn mul(self, o: Self) -> Self {
        let p = [
            self.lo * o.lo,
            self.lo * o.hi,
            self.hi * o.lo,
            self.hi * o.hi,
        ];
        let lo = p.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self {
            lo: lo.next_down(),
            hi: hi.next_up(),
        }
    }

    fn half(self) -> Self {
        // Halving is exact unless it underflows; widen anyway.
        Self {
            lo: (self.lo * 0.5).next_down(),
            hi: (self.hi * 0.5).next_up(),
        }
    }
}

/// The verdict of [`certify`].
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Certificate {
    /// `Q ≥ margin` on the whole interval.
    Positive { margin: f64, depth: usize },
    /// `Q ≤ -margin` on the whole interval.
    Negative { margin: f64, depth: usize },
    /// Not proven within the subdivision depth: the proposal is refused.
    Refused,
}

/// Whether the polynomial `q` (ascending powers of `v`) stays at least `threshold` plus its
/// floating-point evaluation error away from zero on `[lo, hi]`. The proof is the Bernstein
/// enclosure (Garloff 1993) in outward-rounded interval arithmetic, subdividing at the midpoint
/// up to `max_depth`. Protected division returns 0 below |Q| = 1e-12, so a `threshold` below
/// 1e-12 is refused. So is a degree above `MAX_DEGREE` (the binomial ratios are exact well past
/// it), a `max_depth` above `MAX_CERTIFY_DEPTH`, which bounds the work, and an infinite end. So
/// is a `q` whose Horner evaluation could overflow on `[lo, hi]`: the evaluation-error bound
/// holds only when no intermediate does.
#[cfg(test)]
fn certify(q: &[f64], lo: f64, hi: f64, threshold: f64, max_depth: usize) -> Certificate {
    if q.is_empty()
        || q.len() > MAX_DEGREE + 1
        || threshold.is_nan()
        || threshold < 1e-12
        || max_depth > MAX_CERTIFY_DEPTH
        || !lo.is_finite()
        || !hi.is_finite()
        || lo >= hi
        || q.iter().any(|c| !c.is_finite())
    {
        return Certificate::Refused;
    }
    let m = q.len() - 1;
    // Horner's forward error bound (Higham): γ_{2m} Σ |q_j| M^j with γ_k = k u / (1 - k u) and
    // M = max(|lo|, |hi|). It needs every intermediate finite: at |v| ≤ M the product and the sum
    // of the step that adds q_k are at most (1 + γ_{2m}) B_k, B_k = Σ_{j ≥ k} |q_j| M^(j - k).
    // Horner's rule on |q| and M, rounded upward, gives each B_k (B_0 is the bound's sum), and all
    // must stay below f64::MAX / 2, which leaves a factor of 2 for that rounding growth.
    let u = f64::EPSILON / 2.0;
    let gamma = (2 * m) as f64 * u / (1.0 - (2 * m) as f64 * u);
    let reach = lo.abs().max(hi.abs());
    let mut bound = q[m].abs();
    let mut largest = bound;
    for &c in q[..m].iter().rev() {
        bound = ((reach * bound).next_up() + c.abs()).next_up();
        largest = largest.max(bound);
    }
    if largest > f64::MAX / 2.0 {
        return Certificate::Refused;
    }
    let margin = threshold + 2.0 * gamma * bound;
    // q̂(s) = q(lo + (hi - lo) s), expanded in powers of s.
    let width = Interval::point(hi).sub(Interval::point(lo));
    let base = Interval::point(lo);
    let mut shifted = vec![Interval::point(0.0); m + 1];
    for (j, &c) in q.iter().enumerate() {
        // c (lo + w s)^j = Σ_l C(j, l) c lo^(j - l) w^l s^l.
        for (l, slot) in shifted.iter_mut().enumerate().take(j + 1) {
            let mut term = Interval::point(c).mul(Interval::point(binomial(j, l)));
            for _ in 0..(j - l) {
                term = term.mul(base);
            }
            for _ in 0..l {
                term = term.mul(width);
            }
            *slot = slot.add(term);
        }
    }
    // Bernstein coefficients b_i = Σ_{l ≤ i} C(i, l) / C(m, l) q̂_l.
    let bernstein: Vec<Interval> = (0..=m)
        .map(|i| {
            (0..=i).fold(Interval::point(0.0), |sum, l| {
                let ratio = binomial(i, l) / binomial(m, l);
                let ratio = Interval {
                    lo: ratio.next_down(),
                    hi: ratio.next_up(),
                };
                sum.add(ratio.mul(shifted[l]))
            })
        })
        .collect();
    let mut pending = vec![(bernstein, 0usize)];
    let (mut positive, mut negative, mut deepest) = (true, true, 0usize);
    while let Some((b, depth)) = pending.pop() {
        deepest = deepest.max(depth);
        let all_above = b.iter().all(|c| c.lo > margin);
        let all_below = b.iter().all(|c| c.hi < -margin);
        if all_above {
            negative = false;
            continue;
        }
        if all_below {
            positive = false;
            continue;
        }
        if depth == max_depth {
            return Certificate::Refused;
        }
        let (left, right) = de_casteljau_halves(&b);
        pending.push((right, depth + 1));
        pending.push((left, depth + 1));
    }
    match (positive, negative) {
        (true, false) => Certificate::Positive {
            margin,
            depth: deepest,
        },
        (false, true) => Certificate::Negative {
            margin,
            depth: deepest,
        },
        _ => Certificate::Refused,
    }
}

#[cfg(test)]
fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The Bernstein coefficients of the two halves of the interval (de Casteljau at 1/2).
#[cfg(test)]
fn de_casteljau_halves(b: &[Interval]) -> (Vec<Interval>, Vec<Interval>) {
    let m = b.len();
    let mut work = b.to_vec();
    let mut left = Vec::with_capacity(m);
    let mut right = vec![Interval::point(0.0); m];
    left.push(work[0]);
    right[m - 1] = work[m - 1];
    for level in 1..m {
        for i in 0..(m - level) {
            work[i] = work[i].add(work[i + 1]).half();
        }
        left.push(work[0]);
        right[m - 1 - level] = work[m - 1 - level];
    }
    (left, right)
}

#[cfg(test)]
mod linearization_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pcg32;

    fn grid(lo: f64, hi: f64, points: usize, f: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
        (0..points)
            .map(|i| {
                let v = lo + (hi - lo) * i as f64 / (points - 1) as f64;
                (v, f(v))
            })
            .collect()
    }

    fn close(actual: f64, expected: f64, relative: f64) -> bool {
        (actual - expected).abs() <= relative * expected.abs()
    }

    #[test]
    fn exp_on_the_unit_interval_matches_the_published_minimax_errors() {
        // Atkinson, An Introduction to Numerical Analysis, §4.4: ρ_n(e^x) on [-1, 1], n = 1..9.
        let published = [
            2.79e-1, 4.50e-2, 5.53e-3, 5.47e-4, 4.52e-5, 3.21e-6, 2.00e-7, 1.11e-8, 5.52e-10,
        ];
        let samples = grid(-1.0, 1.0, 4001, f64::exp);
        for (n, rho) in published.iter().enumerate().map(|(i, r)| (i + 1, *r)) {
            let fit = minimax(&samples, n).unwrap();
            assert!(fit.levelled <= fit.max_error, "n={n}: {fit:?}");
            // Three printed significant digits: half a unit in the last place is at most 5e-3
            // relative. The discrete minimax can only be at or below the continuous one.
            assert!(
                close(fit.max_error, rho, 5e-3),
                "n={n}: {} vs {rho}",
                fit.max_error
            );
            assert!(
                matches!(
                    fit.stop,
                    Stop::Converged | Stop::NoiseLimited | Stop::Stalled
                ),
                "n={n}: {:?}",
                fit.stop
            );
            if fit.stop == Stop::Converged {
                assert!(fit.max_error - fit.levelled <= TOLERANCE * fit.max_error);
            }
            assert!(fit.iterations <= 20, "n={n}: {} steps", fit.iterations);
        }
    }

    #[test]
    fn the_linear_fit_is_the_closed_form_when_the_alternation_point_is_sampled() {
        // a1 = sinh 1, x* = ln(sinh 1), a0 = (e - sinh(1) x*) / 2, ρ1 = (1/e + sinh(1) x*) / 2.
        let s = 1f64.sinh();
        let star = s.ln();
        let a0 = (1f64.exp() - s * star) / 2.0;
        let rho = ((-1f64).exp() + s * star) / 2.0;
        let mut samples = grid(-1.0, 1.0, 801, f64::exp);
        samples.push((star, star.exp()));
        let fit = minimax(&samples, 1).unwrap();
        let monomial = fit.monomial();
        assert!(close(monomial[1], s, 1e-14), "{monomial:?}");
        assert!(close(monomial[0], a0, 1e-14), "{monomial:?}");
        assert!(
            close(fit.max_error, rho, 1e-14),
            "{} vs {rho}",
            fit.max_error
        );
        assert!(close(fit.levelled, rho, 1e-13));
    }

    #[test]
    fn quadratic_and_cubic_coefficients_match_their_published_digits() {
        let samples = grid(-1.0, 1.0, 4001, f64::exp);
        let quadratic = minimax(&samples, 2).unwrap().monomial();
        for (a, b) in quadratic.iter().zip([0.9890, 1.1302, 0.5540]) {
            assert!((a - b).abs() <= 1e-4, "{quadratic:?}");
        }
        let cubic = minimax(&samples, 3).unwrap().monomial();
        for (a, b) in cubic.iter().zip([0.994579, 0.995668, 0.542973, 0.179533]) {
            assert!((a - b).abs() <= 2e-6, "{cubic:?}");
        }
    }

    #[test]
    fn higher_degree_fixtures_from_sollya_and_chebfun() {
        // Sollya: remez(exp(x), 5, [0;1]) has error 1.129569802274787e-6.
        let fit = minimax(&grid(0.0, 1.0, 4001, f64::exp), 5).unwrap();
        assert!(
            close(fit.max_error, 1.129569802274787e-6, 1e-3),
            "{}",
            fit.max_error
        );
        // Pachón and Trefethen, Table 1, degree 10 on [-1, 1].
        let f1 = |x: f64| (x + 0.5).tanh() - (x - 0.5).tanh();
        let fit = minimax(&grid(-1.0, 1.0, 8001, f1), 10).unwrap();
        assert!(
            close(fit.max_error, 0.000_000_300_091_95, 1e-3),
            "{}",
            fit.max_error
        );
        let f2 = |x: f64| x.exp().sin();
        let fit = minimax(&grid(-1.0, 1.0, 8001, f2), 10).unwrap();
        assert!(
            close(fit.max_error, 0.000_001_786_234_00, 1e-3),
            "{}",
            fit.max_error
        );
    }

    #[test]
    fn the_final_reference_alternates_at_the_levelled_error() {
        let samples = grid(-1.0, 1.0, 2001, |x| (2.0 * x).sin() + 0.3 * x);
        let fit = minimax(&samples, 4).unwrap();
        let unique = prepare(&samples).unwrap();
        let errors: Vec<f64> = fit
            .reference
            .iter()
            .map(|&r| unique[r].1 - fit.eval(unique[r].0))
            .collect();
        for pair in errors.windows(2) {
            assert!(pair[0] * pair[1] < 0.0, "{errors:?}");
        }
        for e in &errors {
            assert!(
                close(e.abs(), fit.levelled, 1e-9),
                "{errors:?} vs {}",
                fit.levelled
            );
        }
    }

    #[test]
    fn fits_ignore_sample_order_and_duplicates() {
        let samples = grid(-2.0, 3.0, 701, |x| (0.7 * x).cos() * x);
        let reference = minimax(&samples, 5).unwrap();
        let mut rng = Pcg32::new(0xB19, 1);
        for _ in 0..4 {
            let mut shuffled = samples.clone();
            shuffled.extend(samples.iter().step_by(7).copied());
            for i in (1..shuffled.len()).rev() {
                let j = rng.next_u32() as usize % (i + 1);
                shuffled.swap(i, j);
            }
            let fit = minimax(&shuffled, 5).unwrap();
            let bits = |m: &Minimax| m.chebyshev.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&fit), bits(&reference));
            assert_eq!(fit.reference, reference.reference);
            assert_eq!(fit.max_error.to_bits(), reference.max_error.to_bits());
        }
    }

    #[test]
    fn exact_polynomials_and_refusals() {
        let cubic = |x: f64| 0.5 - 1.25 * x + 0.75 * x * x * x;
        let fit = minimax(&grid(-1.0, 1.0, 101, cubic), 3).unwrap();
        assert_eq!(fit.stop, Stop::ExactFit);
        assert!(fit.max_error <= 1e-15);
        assert_eq!(
            minimax(&grid(0.0, 1.0, 4, f64::exp), 3).unwrap_err(),
            "fewer than degree + 2 distinct abscissae"
        );
        assert!(minimax(&[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 1.0)], 1).is_err());
        // A non-finite target is refused by name, also where it would sit off the reference: the
        // exchange cannot see a NaN error, and an infinite one makes any fit look exact.
        assert_eq!(
            minimax(&[(1.0, f64::NAN), (2.0, 1.0), (3.0, 1.0)], 1).unwrap_err(),
            "a sample is not finite"
        );
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut interior = grid(0.0, 1.0, 50, f64::exp);
            interior[25].1 = bad;
            assert_eq!(
                minimax(&interior, 2).unwrap_err(),
                "a sample is not finite",
                "{bad}"
            );
        }
        assert!(minimax(&grid(0.0, 1.0, 100, f64::exp), MAX_DEGREE + 1).is_err());
        let one_point: Vec<(f64, f64)> = (0..10).map(|_| (2.0, 1.0)).collect();
        assert!(minimax(&one_point, 1).is_err());
    }

    fn native(expr: &Expr, v: f64) -> (f64, FaultSet) {
        MathExpr::from_ast(expr.clone())
            .unwrap()
            .eval_row_traced(&[v])
    }

    #[test]
    fn lowering_uses_ordinary_nodes_within_the_published_budget() {
        let x = Expr::Var(Var::Col(0));
        let cases: [(&[f64], usize, usize); 4] = [
            (&[1.25, 0.5], 5, 3),
            (&[1.25, 0.5, 0.75], 9, 5),
            (&[1.25, 0.5, 0.75, 0.125], 13, 7),
            // Negative non-leading coefficients become Sub at no extra cost.
            (&[-1.25, -0.5, 0.75], 9, 5),
        ];
        for (coefficients, nodes, height) in cases {
            let tree = lower(coefficients, &x).unwrap();
            assert_eq!(count_nodes(&tree), nodes, "{coefficients:?}");
            assert_eq!(depth(&tree), height, "{coefficients:?}");
            for v in [-1.5, -0.25, 0.0, 0.6, 2.0] {
                let expected = coefficients.iter().rev().fold(0.0, |acc, c| acc * v + c);
                let (value, faults) = native(&tree, v);
                assert!(faults.is_empty());
                assert!(
                    (value - expected).abs() <= 1e-14 * expected.abs().max(1.0),
                    "{coefficients:?} at {v}"
                );
            }
        }
        // A negative leading coefficient is a literal, two nodes.
        let tree = lower(&[1.0, -2.0], &x).unwrap();
        assert_eq!(count_nodes(&tree), 6);
        assert!(lower(&[1.0, f64::NAN], &x).is_none());
        assert!(lower(&[], &x).is_none());
    }

    #[test]
    fn a_one_variable_subtree_compresses_and_the_native_value_decides_outside_the_range() {
        let rows: Vec<Vec<f64>> = (0..201)
            .map(|i| vec![-1.0 + f64::from(i) / 100.0, 7.0])
            .collect();
        // 18 nodes, compressed to a cubic of at most 14.
        let subtree =
            MathExpr::parse("((exp(0.5*c0)+0.25)*(cos(0.3*c0)+1.5))+sin(0.2*c0)").unwrap();
        assert_eq!(count_nodes(&subtree.root), 18);
        let (var, points) = samples(&subtree, &rows).unwrap();
        assert_eq!(var, Var::Col(0));
        let fit = minimax(&points, 3).unwrap();
        let leaf = Expr::Var(var);
        let proposal = propose(&fit, &points, &leaf, &subtree.root, 64).unwrap();
        assert!(proposal.nodes <= 14, "{proposal:?}");
        let compressed = MathExpr::from_ast(proposal.tree.clone()).unwrap();
        // On the training range the native lowered tree is within its own measured error, which
        // is within the fit's bound.
        for row in &rows {
            let (a, faults) = compressed.eval_row_traced(row);
            let (b, _) = subtree.eval_row_traced(row);
            assert!(faults.is_empty());
            assert!((a - b).abs() <= proposal.lowered_error, "{row:?}");
        }
        assert!(
            proposal.lowered_error <= fit.max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * 20.0
        );
        // Outside it the polynomial drifts: only a native re-score on held-out rows can tell.
        let (a, _) = compressed.eval_row_traced(&[4.0, 7.0]);
        let (b, _) = subtree.eval_row_traced(&[4.0, 7.0]);
        assert!((a - b).abs() > 100.0 * fit.max_error, "{a} vs {b}");
        // A subtree no smaller than the lowered tree is never proposed (the 6-node
        // exp(0.5*c0)+0.25 against a cubic), and neither is a tree over the node budget.
        let small = MathExpr::parse("exp(0.5*c0)+0.25").unwrap();
        let (_, small_points) = samples(&small, &rows).unwrap();
        let small_fit = minimax(&small_points, 3).unwrap();
        assert!(propose(&small_fit, &small_points, &leaf, &small.root, 64).is_none());
        assert!(propose(&fit, &points, &leaf, &subtree.root, 12).is_none());
        // Subtrees reading two variables, or faulting on a row, are not sampled.
        assert!(samples(&MathExpr::parse("c0*c1").unwrap(), &rows).is_none());
        assert!(samples(&MathExpr::parse("log(c0)").unwrap(), &rows).is_none());
        assert!(samples(&MathExpr::parse("3.0").unwrap(), &rows).is_none());
    }

    #[test]
    fn the_bernstein_certificate_proves_only_pole_free_denominators() {
        // 1 + 0.5 v² on [0, 1]: at least 1.
        assert!(matches!(
            certify(&[1.0, 0.0, 0.5], 0.0, 1.0, 1e-12, 4),
            Certificate::Positive { .. }
        ));
        // A root inside the interval: v - 0.3 and v² - 0.25 on [0, 1].
        assert_eq!(
            certify(&[-0.3, 1.0], 0.0, 1.0, 1e-12, 6),
            Certificate::Refused
        );
        assert_eq!(
            certify(&[-0.25, 0.0, 1.0], 0.0, 1.0, 1e-12, 6),
            Certificate::Refused
        );
        // A minimum of 5e-13 sits below the protected-division threshold.
        assert_eq!(
            certify(&[5e-13 + 0.25, -1.0, 1.0], 0.0, 1.0, 1e-12, 8),
            Certificate::Refused
        );
        // A negative denominator is certified as such, and subdivision tightens the enclosure.
        assert!(matches!(
            certify(&[-2.0, 0.0, 1.0], -1.0, 1.0, 1e-12, 4),
            Certificate::Negative { .. }
        ));
        match certify(&[0.3, -1.0, 1.0], 0.0, 1.0, 1e-12, 8) {
            Certificate::Positive { depth, .. } => assert!(depth >= 1),
            other => panic!("{other:?}"),
        }
        assert_eq!(certify(&[1.0], 1.0, 1.0, 1e-12, 4), Certificate::Refused);
        assert_eq!(
            certify(&[f64::NAN], 0.0, 1.0, 1e-12, 4),
            Certificate::Refused
        );
    }

    #[test]
    fn interval_operations_enclose_their_exact_results() {
        let a = Interval::point(0.1);
        let b = Interval::point(0.2);
        let sum = a.add(b);
        assert!(sum.lo < 0.30000000000000004 && sum.hi > 0.3);
        let product = Interval { lo: -1.0, hi: 2.0 }.mul(Interval { lo: -3.0, hi: 0.5 });
        assert!(product.lo <= -6.0 && product.hi >= 3.0);
        let halves = de_casteljau_halves(&[Interval::point(1.0), Interval::point(3.0)]);
        assert!(halves.0[1].lo <= 2.0 && halves.0[1].hi >= 2.0);
        assert_eq!(halves.1[1], Interval::point(3.0));
    }

    #[test]
    fn an_off_centre_lowering_is_held_to_its_own_error() {
        let f = |x: f64| (0.37 * x).sin() + (x / 50.0).cos();
        let leaf = Expr::Var(Var::Col(0));
        // A 41-node stand-in for the replaced subtree, so that only the error gate can refuse.
        let big = (0..20).fold(leaf.clone(), |acc, _| {
            Expr::Binary(BinOp::Add, Box::new(acc), Box::new(leaf.clone()))
        });
        // Centred, the lowered tree keeps the fit's error.
        let centred = grid(-1.0, 1.0, 2001, f);
        let fit = minimax(&centred, 6).unwrap();
        let error = lowered_error(&fit.monomial(), &prepare(&centred).unwrap()).unwrap();
        assert!(
            error <= fit.max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * 2.0,
            "{error} vs {}",
            fit.max_error
        );
        assert!(propose(&fit, &centred, &leaf, &big, 64).is_some());
        // Off centre the monomial conversion loses the fit, so nothing is proposed.
        let shifted = grid(1000.0, 1001.0, 2001, f);
        let fit = minimax(&shifted, 6).unwrap();
        let error =
            lowered_error(&fit.monomial(), &prepare(&shifted).unwrap()).unwrap_or(f64::INFINITY);
        assert!(error > 1e3 * fit.max_error, "{error} vs {}", fit.max_error);
        assert!(propose(&fit, &shifted, &leaf, &big, 64).is_none());
    }

    #[test]
    fn signed_zero_abscissae_with_signed_zero_targets_are_one_sample() {
        // Odd subtrees give f(-0.0) = -0.0 and f(0.0) = 0.0 (0.5*c0, sin(c0), c0^3).
        let points0 = [
            (-0.0, -0.0),
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (-1.0, -1.0),
        ];
        let fit = minimax(&points0, 1).unwrap();
        assert_eq!(fit.stop, Stop::ExactFit);
        assert_eq!(prepare(&points0).unwrap().len(), 4);
        let mut reversed = points0.to_vec();
        reversed.reverse();
        let bits = |m: &Minimax| m.chebyshev.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&minimax(&reversed, 1).unwrap()), bits(&fit));
        // Through the sampler: a column holding both zeros.
        let rows: Vec<Vec<f64>> = [-1.0, -0.0, 0.0, 0.5, 1.0]
            .iter()
            .map(|&v| vec![v])
            .collect();
        let (_, points) = samples(&MathExpr::parse("0.5*c0").unwrap(), &rows).unwrap();
        assert!(minimax(&points, 1).is_ok());
        // A genuine conflict at zero is still refused.
        assert!(minimax(&[(0.0, 1.0), (-0.0, -1.0), (1.0, 2.0), (2.0, 3.0)], 1).is_err());
    }

    #[test]
    fn every_stop_path_is_reached_and_the_cap_returns_the_best_fit() {
        // Abscissae 2^i with targets +1, -1, -1 repeating: exact data, badly conditioned in t.
        let geometric = |points: usize| -> Vec<(f64, f64)> {
            (0..points)
                .map(|i| (f64::from(1u32 << i), if i % 3 == 0 { 1.0 } else { -1.0 }))
                .collect()
        };
        // The exchange cycles between two references from step 3 on, with max errors above
        // 1e15, so the cap must return step 2's fit, not the last one.
        let capped = minimax(&geometric(15), 11).unwrap();
        assert_eq!(
            (capped.stop, capped.iterations),
            (Stop::IterationLimit, 2),
            "{capped:?}"
        );
        assert!(capped.max_error < 2.0, "{capped:?}");
        assert!(capped.levelled <= capped.max_error);
        // |x| at 15 points, degree 12: a two-step cycle a few 1e-12 above the bracket; the best
        // fit is step 9's.
        let abs = minimax(&grid(-1.0, 1.0, 15, f64::abs), 12).unwrap();
        assert_eq!(
            (abs.stop, abs.iterations),
            (Stop::IterationLimit, 9),
            "{abs:?}"
        );
        assert!(abs.levelled <= abs.max_error);
        // The exchange returns the same reference.
        let stalled = minimax(&geometric(8), 6).unwrap();
        assert_eq!(stalled.stop, Stop::Stalled, "{stalled:?}");
        assert!(stalled.levelled <= stalled.max_error);
        // Fewer than n + 2 alternating extrema, with the global maximum already a reference
        // point.
        let no_exchange = minimax(&geometric(10), 8).unwrap();
        assert_eq!(no_exchange.stop, Stop::Degenerate, "{no_exchange:?}");
        // The second reference system is singular: the best (first) fit comes back.
        let singular = minimax(&geometric(27), 3).unwrap();
        assert_eq!(
            (singular.stop, singular.iterations),
            (Stop::Degenerate, 1),
            "{singular:?}"
        );
    }

    #[test]
    fn an_offset_noise_floor_stop_is_not_reported_as_converged() {
        // A constant offset lifts max|f|, and with it the rounding level of the errors, far above
        // 1e-12 of the minimax error.
        let offset = minimax(&grid(-1.0, 1.0, 4001, |x| 1e6 + x.exp()), 2).unwrap();
        assert_eq!(offset.stop, Stop::NoiseLimited, "{offset:?}");
        assert!(offset.max_error - offset.levelled > TOLERANCE * offset.max_error);
        assert!(offset.max_error - offset.levelled <= 64.0 * f64::EPSILON * (1e6 + 1f64.exp()));
        // Without the offset the same fit closes the documented bracket.
        let plain = minimax(&grid(-1.0, 1.0, 4001, f64::exp), 2).unwrap();
        assert_eq!(plain.stop, Stop::Converged, "{plain:?}");
        assert!(plain.max_error - plain.levelled <= TOLERANCE * plain.max_error);
    }

    /// Lane D pre-review of B19: a noise-floor stop came while the exchange could still raise |h|.
    /// On 1e3 + cos(0.7x)x at degree 9 it stopped at step 5 with max|e| = 1.88e-11, where step 8
    /// reaches 7.73e-12; on 1e6 + cos(0.7x)x at degree 7 it stopped at step 1 with |h| ≈ 1e-17
    /// against max|e| = 1.09e-8, where step 5 reaches 5.59e-9.
    #[test]
    fn a_noise_floor_stop_waits_until_the_exchange_cannot_raise_h() {
        let fit = minimax(&grid(-1.0, 1.0, 2001, |x| 1e3 + (0.7 * x).cos() * x), 9).unwrap();
        assert!(fit.max_error <= 1.05 * 7.731e-12, "{fit:?}");
        let fit = minimax(&grid(-1.0, 1.0, 2001, |x| 1e6 + (0.7 * x).cos() * x), 7).unwrap();
        assert!(fit.levelled >= fit.max_error / 2.0, "{fit:?}");
        assert!(fit.max_error <= 1.05 * 5.588e-9, "{fit:?}");
    }

    /// Lane D pre-review of B19: rounding can put |h| above max|e|, and such inverted brackets were
    /// reported as Converged (1e10 + tanh(5x) by 3.7e-7, 7.7e5 times the tolerance). `levelled` now
    /// never exceeds `max_error`, and an |h| above it by more than the tolerance is not Converged.
    #[test]
    fn no_reported_bracket_inverts_and_an_inverted_one_is_not_converged() {
        for (samples, n) in [
            (grid(-1.0, 1.0, 2001, |x| 1e10 + (5.0 * x).tanh()), 2),
            (grid(-1.0, 1.0, 501, |x| 10.0 + (2.0 + x).ln()), 8),
            (
                grid(-1.0, 1.0, 1001, |x| 3.0 + ((2.0 * x).sin() + 0.3 * x)),
                10,
            ),
        ] {
            let fit = minimax(&samples, n).unwrap();
            assert_ne!(fit.stop, Stop::Converged, "n={n}: {fit:?}");
            assert!(fit.levelled <= fit.max_error, "n={n}: {fit:?}");
        }
        let functions: [fn(f64) -> f64; 3] = [f64::exp, f64::sin, |x| (5.0 * x).tanh()];
        for f in functions {
            for offset in [0.0, 10.0, 1e3, 1e6, 1e10] {
                let samples = grid(-1.0, 1.0, 501, |x| offset + f(x));
                for n in 1..=10 {
                    let fit = minimax(&samples, n).unwrap();
                    assert!(fit.levelled <= fit.max_error, "{offset} n={n}: {fit:?}");
                    if fit.stop == Stop::Converged {
                        assert!(fit.max_error - fit.levelled <= TOLERANCE * fit.max_error);
                    }
                }
            }
        }
    }

    /// Lane D pre-review of B19: near `f64::MAX` the map to t and the Clenshaw sums overflowed, and
    /// an infinite max|e| passed the Converged test. A sample outside the evaluator's range is
    /// refused by name; the range itself is accepted.
    #[test]
    fn samples_outside_the_evaluator_range_are_refused() {
        let refused = "a sample is outside the evaluator's range";
        let wide =
            |amplitude: f64, k: f64| grid(-1.0, 1.0, 101, move |x| amplitude * (k * x).sin());
        assert_eq!(minimax(&wide(1e308, 3.0), 3).unwrap_err(), refused);
        assert_eq!(minimax(&wide(5e307, 6.0), 6).unwrap_err(), refused);
        let near: Vec<(f64, f64)> = (0..10)
            .map(|i| (1e308 + f64::from(i) * 7e306, f64::from(i)))
            .collect();
        assert_eq!(minimax(&near, 1).unwrap_err(), refused);
        let edge = minimax(&grid(-MAX_ABS, MAX_ABS, 101, |x| 0.5 * x), 1).unwrap();
        assert_eq!(edge.stop, Stop::ExactFit);
    }

    /// Lane D pre-review of B19: only the expression limits stop a degree-8 lowering (depth 17 or
    /// 18) from being proposed, and no test pinned that boundary. And `x` must be the sampled
    /// variable: the error is measured at the sampled abscissae.
    #[test]
    fn the_depth_limit_and_the_leaf_bound_what_is_proposed() {
        let f = |x: f64| (0.37 * x).sin() + (x / 50.0).cos();
        let leaf = Expr::Var(Var::Col(0));
        let big = (0..20).fold(leaf.clone(), |acc, _| {
            Expr::Binary(BinOp::Add, Box::new(acc), Box::new(leaf.clone()))
        });
        let samples = grid(-1.0, 1.0, 2001, f);
        let seven = propose(&minimax(&samples, 7).unwrap(), &samples, &leaf, &big, 64).unwrap();
        assert!(depth(&seven.tree) <= 16, "{seven:?}");
        assert!(propose(&minimax(&samples, 8).unwrap(), &samples, &leaf, &big, 64).is_none());
        let cubic = minimax(&samples, 3).unwrap();
        assert!(propose(&cubic, &samples, &leaf, &big, 64).is_some());
        let square = Expr::Binary(BinOp::Mul, Box::new(leaf.clone()), Box::new(leaf));
        assert!(propose(&cubic, &samples, &square, &big, 64).is_none());
    }

    /// Lane D pre-review of B19: the certificate's soundness margins had no test that fails when
    /// they are removed (the evaluation-error term, and the outward rounding of mul, half and the
    /// upper side of add), and its threshold, degree and depth were trusted.
    #[test]
    fn the_certificate_keeps_its_margins_and_refuses_inputs_it_cannot_bound() {
        // 1e6 + v² on [-1, 1]: the margin includes 2 γ_4 (1e6 + 1) ≈ 8.9e-10.
        match certify(&[1e6, 0.0, 1.0], -1.0, 1.0, 1e-12, 4) {
            Certificate::Positive { margin, .. } => assert!(margin > 1e-12 + 8e-10, "{margin}"),
            other => panic!("{other:?}"),
        }
        // A margin of the threshold alone would certify 1.1e-12 + 1000 v on [0, 1]; the Horner
        // term lifts it to 1.44e-12.
        assert_eq!(
            certify(&[1.1e-12, 1000.0], 0.0, 1.0, 1e-12, 8),
            Certificate::Refused
        );
        assert!(matches!(
            certify(&[1.5e-12, 1000.0], 0.0, 1.0, 1e-12, 8),
            Certificate::Positive { .. }
        ));
        // Each operation encloses a result that rounding would otherwise lose.
        let product = Interval::point(0.1).mul(Interval::point(0.2));
        assert!(
            product.lo < 0.1 * 0.2 && 0.1 * 0.2 < product.hi,
            "{product:?}"
        );
        let half = Interval::point(5e-324).half();
        assert!(half.lo <= 0.0 && half.hi >= 5e-324, "{half:?}");
        assert!(Interval::point(0.1).add(Interval::point(0.2)).hi > 0.30000000000000004);
        let sum = Interval::point(1.0).add(Interval::point(1e-20));
        assert!(sum.hi > 1.0, "{sum:?}");
        let difference = Interval::point(1.0).sub(Interval::point(1e-20));
        assert!(difference.lo < 1.0, "{difference:?}");
        // v - 0.5 has a root in [0, 1], and -0.5 is negative: a threshold below 1e-12 is refused
        // instead of certifying either.
        assert_eq!(
            certify(&[-0.5, 1.0], 0.0, 1.0, -1.0, 4),
            Certificate::Refused
        );
        assert_eq!(certify(&[-0.5], 0.0, 1.0, 0.0, 4), Certificate::Refused);
        let positive = [1.0, 0.0, 0.5];
        assert_eq!(
            certify(&positive, 0.0, 1.0, f64::NAN, 4),
            Certificate::Refused
        );
        let deep = MAX_CERTIFY_DEPTH + 1;
        assert_eq!(
            certify(&positive, 0.0, 1.0, 1e-12, deep),
            Certificate::Refused
        );
        assert!(matches!(
            certify(&positive, 0.0, 1.0, 1e-12, MAX_CERTIFY_DEPTH),
            Certificate::Positive { .. }
        ));
        let ones = [1.0; MAX_DEGREE + 2];
        assert_eq!(certify(&ones, 0.0, 1.0, 1e-12, 4), Certificate::Refused);
        assert!(matches!(
            certify(&ones[..=MAX_DEGREE], 0.0, 1.0, 1e-12, 4),
            Certificate::Positive { .. }
        ));
    }

    /// Lane D pre-review of B19: `iterations` is the step of the retained fit, so a capped run
    /// reported 2 after 32 solves; `steps` counts the solves.
    #[test]
    fn steps_count_the_exchange_beside_the_retained_step() {
        let geometric = |points: usize| -> Vec<(f64, f64)> {
            (0..points)
                .map(|i| (f64::from(1u32 << i), if i % 3 == 0 { 1.0 } else { -1.0 }))
                .collect()
        };
        let capped = minimax(&geometric(15), 11).unwrap();
        assert_eq!((capped.iterations, capped.steps), (2, MAX_ITERATIONS));
        let stalled = minimax(&geometric(8), 6).unwrap();
        assert_eq!((stalled.iterations, stalled.steps), (1, 1));
        let singular = minimax(&geometric(27), 3).unwrap();
        assert_eq!((singular.iterations, singular.steps), (1, 2));
    }

    /// Independent review of B19 (R1): one step whose |h| fell was taken for a spent exchange. On
    /// these 18 samples at degree 5 the run stopped NoiseLimited at step 9 with max|e| = 2.29e-5,
    /// where the next exchange raises |h| and reaches 1.72e-5.
    #[test]
    fn a_noise_floor_stop_needs_a_spent_exchange_not_a_falling_h() {
        let samples = [
            (-0.999990739939228, 9999999999.999983),
            (-0.9978979429134074, 10000000000.000017),
            (0.7074203759847726, 10000000000.00001),
            (0.912930560725927, 10000000000.000006),
            (-0.6831402736477055, 9999999999.999983),
            (-0.977500969634282, 9999999999.999985),
            (-0.8789742671837542, 10000000000.000002),
            (-0.9985609737561163, 9999999999.999983),
            (-0.8468934336899454, 10000000000.000017),
            (-0.3649387527514045, 10000000000.000015),
            (0.23064368667000013, 9999999999.999989),
            (-0.8362009505253496, 9999999999.999985),
            (-0.309612662189795, 9999999999.999989),
            (0.44993701364736216, 9999999999.999989),
            (-0.526377661688965, 9999999999.999992),
            (-0.010832884454037295, 10000000000.00001),
            (-0.8262491006272136, 10000000000.000004),
            (0.6818349395106322, 10000000000.000015),
        ];
        let fit = minimax(&samples, 5).unwrap();
        assert_eq!(
            (fit.stop, fit.iterations),
            (Stop::NoiseLimited, 10),
            "{fit:?}"
        );
        assert!(fit.max_error <= 1.71661376953125e-5, "{fit:?}");
        assert!(fit.levelled <= fit.max_error, "{fit:?}");
    }

    /// `a + b` as the rounded sum and its exact error (Knuth's TwoSum).
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let sum = a + b;
        let b_part = sum - a;
        (sum, (a - (sum - b_part)) + (b - b_part))
    }

    /// `a b` as the rounded product and its exact error (Dekker's TwoProduct, splitting at
    /// 2^27 + 1, with no fused multiply-add).
    fn two_product(a: f64, b: f64) -> (f64, f64) {
        let split = |x: f64| {
            let scaled = 134_217_729.0 * x;
            let high = scaled - (scaled - x);
            (high, x - high)
        };
        let product = a * b;
        let ((a_high, a_low), (b_high, b_low)) = (split(a), split(b));
        let error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low;
        (product, error)
    }

    /// The sign of the exact sum of `terms`. Shewchuk's grow-expansion keeps the sum exactly, as
    /// nonoverlapping components of increasing magnitude, and the largest one decides.
    fn exact_sign(terms: &[f64]) -> i8 {
        let mut expansion: Vec<f64> = Vec::new();
        for &term in terms {
            let mut carry = term;
            let mut grown = Vec::with_capacity(expansion.len() + 1);
            for &component in &expansion {
                let (sum, error) = two_sum(carry, component);
                if error != 0.0 {
                    grown.push(error);
                }
                carry = sum;
            }
            if carry != 0.0 {
                grown.push(carry);
            }
            expansion = grown;
        }
        expansion
            .last()
            .map_or(0, |&c| (c > 0.0) as i8 - (c < 0.0) as i8)
    }

    /// Independent review of B19 (R2): `levelled` and `max_error` are residuals of this module's
    /// floating-point evaluation, not certified bounds on the discrete minimax error E*. On 1e10 +
    /// tanh(5x) at degree 2 the exact errors of the returned polynomial alternate on the reference
    /// 3.7e-7 above `max_error`, so E* exceeds it (de la Vallée Poussin). E* still lies within the
    /// rounding level `64 ε max|f|` of it, because every exact error does.
    #[test]
    fn the_bracket_is_a_floating_point_residual_not_a_certified_bound() {
        let samples = grid(-1.0, 1.0, 2001, |x| 1e10 + (5.0 * x).tanh());
        let fit = minimax(&samples, 2).unwrap();
        assert!(fit.levelled <= fit.max_error, "{fit:?}");
        let unique = prepare(&samples).unwrap();
        let [c0, c1, c2] = fit.chebyshev[..] else {
            panic!("{fit:?}")
        };
        // f - (c0 + c1 t + c2 (2 t² - 1)) as an exact sum of doubles. The map to t is exact on
        // [-1, 1], so these are the errors at the sample abscissae themselves.
        let error_terms = |(v, f): (f64, f64)| -> [f64; 9] {
            let t = fit.t(v);
            assert_eq!(t.to_bits(), v.to_bits());
            let (square, square_error) = two_product(t, t);
            let (linear, linear_error) = two_product(c1, t);
            let (high, high_error) = two_product(c2, square);
            let (low, low_error) = two_product(c2, square_error);
            [
                f,
                -c0,
                c2,
                -linear,
                -linear_error,
                -2.0 * high,
                -2.0 * high_error,
                -2.0 * low,
                -2.0 * low_error,
            ]
        };
        let signed = |terms: [f64; 9], sign: i8, extra: [f64; 2]| -> Vec<f64> {
            let sign = f64::from(sign);
            terms.iter().map(|x| sign * x).chain(extra).collect()
        };
        // On the reference the exact errors alternate, each more than 3e-7 above max_error.
        let mut previous = 0;
        for &r in &fit.reference {
            let terms = error_terms(unique[r]);
            let sign = exact_sign(&terms);
            assert!(sign != 0 && sign != previous, "{r}");
            let excess = signed(terms, sign, [-fit.max_error, -3e-7]);
            assert_eq!(exact_sign(&excess), 1, "{r}: {fit:?}");
            previous = sign;
        }
        // Every exact error, and so E* too, is within the rounding level of max_error.
        let f_max = unique.iter().map(|(_, f)| f.abs()).fold(0.0f64, f64::max);
        let noise = 64.0 * f64::EPSILON * f_max;
        for &sample in &unique {
            let terms = error_terms(sample);
            for sign in [1, -1] {
                let excess = signed(terms, sign, [-fit.max_error, -noise]);
                assert_eq!(exact_sign(&excess), -1, "{sample:?}");
            }
        }
    }

    /// Independent review of B19 (R3): Horner's error bound holds only when no intermediate
    /// overflows. 1 + 1e308 v + 1e308 v² on [0.8, 0.9] was certified with a finite margin although
    /// Horner overflows at 0.9 (0.9 · 1e308 + 1e308): the weighted sum Σ |q_j| M^j stays finite.
    #[test]
    fn the_certificate_refuses_a_horner_evaluation_that_can_overflow() {
        let q = [1.0, 1e308, 1e308];
        assert!(horner(&q, 0.9).is_infinite());
        assert_eq!(certify(&q, 0.8, 0.9, 1e-12, 8), Certificate::Refused);
        // Every coefficient below f64::MAX / 2, and a partial sum still overflows at 0.7.
        let q = [1.0, 8.9e307, 8.9e307, 8.9e307, 8.9e307];
        assert!(horner(&q, 0.7).is_infinite());
        assert_eq!(certify(&q, 0.6, 0.7, 1e-12, 8), Certificate::Refused);
        // Partial sums within f64::MAX / 2 are certified, with a finite margin.
        match certify(&[1.0, 4e307, 4e307], 0.8, 0.9, 1e-12, 8) {
            Certificate::Positive { margin, .. } => assert!(margin.is_finite(), "{margin}"),
            other => panic!("{other:?}"),
        }
        // An infinite end leaves M unbounded.
        assert_eq!(
            certify(&[1.0, 0.0, 1.0], f64::NEG_INFINITY, 0.0, 1e-12, 8),
            Certificate::Refused
        );
    }
}
