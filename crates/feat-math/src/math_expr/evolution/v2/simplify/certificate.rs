use super::*;

// ---------------------------------------------------------------------------------------------
// The pole-free certificate for a rational proposal's denominator. No proposal uses a
// denominator, so the certificate is compiled for its tests only.

/// A closed interval of reals, widened outward by one ulp after every operation.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Interval {
    pub(super) lo: f64,
    pub(super) hi: f64,
}

#[cfg(test)]
impl Interval {
    pub(super) fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    pub(super) fn add(self, o: Self) -> Self {
        Self {
            lo: (self.lo + o.lo).next_down(),
            hi: (self.hi + o.hi).next_up(),
        }
    }

    pub(super) fn sub(self, o: Self) -> Self {
        Self {
            lo: (self.lo - o.hi).next_down(),
            hi: (self.hi - o.lo).next_up(),
        }
    }

    pub(super) fn mul(self, o: Self) -> Self {
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

    pub(super) fn half(self) -> Self {
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
pub(super) enum Certificate {
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
pub(super) fn certify(
    q: &[f64],
    lo: f64,
    hi: f64,
    threshold: f64,
    max_depth: usize,
) -> Certificate {
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
pub(super) fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The Bernstein coefficients of the two halves of the interval (de Casteljau at 1/2).
#[cfg(test)]
pub(super) fn de_casteljau_halves(b: &[Interval]) -> (Vec<Interval>, Vec<Interval>) {
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
