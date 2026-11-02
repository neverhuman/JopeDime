use super::*;

/// The largest degree fitted. What can be proposed is bounded separately, by the checks of
/// [`assess`]: a dense Horner tree over a variable leaf has depth at least `2n + 1`, so under the
/// expression depth limit of 16 the largest dense degree that lowers is 7 (a sparse one, its zero
/// coefficients skipped, can be shallower).
pub(crate) const MAX_DEGREE: usize = 12;
/// The exchange stops after this many steps.
pub(crate) const MAX_ITERATIONS: usize = 32;
/// The relative width of the bracket at which the exchange stops `Converged`.
pub(crate) const TOLERANCE: f64 = 1e-12;
/// The deepest subdivision [`certify`] accepts.
#[cfg(test)]
pub(crate) const MAX_CERTIFY_DEPTH: usize = 64;

/// Why a fit stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
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
    pub(crate) fn public(self) -> MinimaxStop {
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
pub(crate) struct Minimax {
    /// The fit interval: the smallest and largest sample abscissa. The error is controlled only at
    /// the samples.
    pub(crate) lo: f64,
    pub(crate) hi: f64,
    /// Chebyshev coefficients in `t`.
    pub(crate) chebyshev: Vec<f64>,
    /// The smaller of `|h|` and the smallest `|e|` on the final reference, never above
    /// `max_error`. In exact arithmetic it would bound the discrete minimax error from below (de
    /// la Vallée Poussin); as computed it does so only up to rounding.
    pub(crate) levelled: f64,
    /// The largest computed error over the samples, by Clenshaw at the rounded `t`. Not a
    /// certified upper bound: the exact errors, and the discrete minimax error with them, can
    /// exceed it by up to the rounding of that evaluation.
    pub(crate) max_error: f64,
    /// The exchange step that produced this fit.
    pub(crate) iterations: usize,
    /// The exchange steps run, at most `MAX_ITERATIONS`.
    pub(crate) steps: usize,
    pub(crate) stop: Stop,
    /// The final reference, as indices into the sorted unique samples.
    pub(crate) reference: Vec<usize>,
}

impl Minimax {
    #[cfg(test)]
    pub(crate) fn t(&self, v: f64) -> f64 {
        (2.0 * v - (self.lo + self.hi)) / (self.hi - self.lo)
    }

    #[cfg(test)]
    pub(crate) fn eval(&self, v: f64) -> f64 {
        clenshaw(&self.chebyshev, self.t(v))
    }

    /// Monomial coefficients in `v` (ascending powers). This conversion is ill-conditioned when
    /// the interval sits far from 0; the native re-score decides.
    pub(crate) fn monomial(&self) -> Vec<f64> {
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

pub(crate) fn add_scaled(out: &mut [f64], polynomial: &[f64], scale: f64) {
    for (o, &a) in out.iter_mut().zip(polynomial) {
        *o += scale * a;
    }
}

/// `Σ c_k T_k(t)` by Clenshaw's recurrence.
pub(crate) fn clenshaw(c: &[f64], t: f64) -> f64 {
    let (mut b1, mut b2) = (0.0f64, 0.0f64);
    for &ck in c.iter().skip(1).rev() {
        let b0 = ck + 2.0 * t * b1 - b2;
        b2 = b1;
        b1 = b0;
    }
    c[0] + t * b1 - b2
}

/// `T_0(t) .. T_n(t)`.
pub(crate) fn chebyshev_row(t: f64, n: usize) -> Vec<f64> {
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
pub(crate) fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
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
pub(crate) fn prepare(samples: &[(f64, f64)]) -> Result<Vec<(f64, f64)>, &'static str> {
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
pub(crate) const FEW_ABSCISSAE: &str = "fewer than degree + 2 distinct abscissae";
/// [`prepare`]'s refusal of a sample past the evaluator's range.
pub(crate) const OUTSIDE_RANGE: &str = "a sample is outside the evaluator's range";
/// [`prepare`]'s refusal of one abscissa with two targets (merged signed zeros that differ).
pub(crate) const TWO_TARGETS: &str = "one abscissa has two targets";
/// The fit's refusal when its first reference system is singular.
pub(crate) const SINGULAR_FIRST: &str = "singular first reference";
/// The fit's refusal when its first fit's errors are not finite.
pub(crate) const NONFINITE_ERROR: &str = "an error is not finite";
/// Every reason the fit can refuse the samples [`samples`] gives for a line: [`prepare`]'s and
/// [`minimax`]'s own, less four. [`samples`] gives only fault-free finite samples, and the
/// evaluator raises a fault for every value it brings back into its range, so neither `a sample
/// is not finite` nor [`OUTSIDE_RANGE`] occurs; a line's degree is under the cap; and three
/// distinct abscissae within that range span an interval that is neither empty nor overflowing,
/// so [`EMPTY_INTERVAL`] does not occur either.
pub(crate) const FIT_REFUSALS: [&str; 4] =
    [TWO_TARGETS, FEW_ABSCISSAE, SINGULAR_FIRST, NONFINITE_ERROR];
/// The fit's refusal of an interval that is empty or overflows.
pub(crate) const EMPTY_INTERVAL: &str = "empty or overflowing interval";

/// The discrete minimax polynomial of `degree` on `samples` of `(v, f(v))`.
pub(crate) fn minimax(samples: &[(f64, f64)], degree: usize) -> Result<Minimax, &'static str> {
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
    pub(crate) fn ending(best: Option<Minimax>, stop: Stop, steps: usize) -> Option<Minimax> {
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
pub(crate) fn exchange(errors: &[f64], h: f64, global: usize, size: usize) -> Option<Vec<usize>> {
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
pub(crate) fn single_exchange(
    errors: &[f64],
    reference: &[usize],
    global: usize,
) -> Option<Vec<usize>> {
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
