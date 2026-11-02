use super::*;

/// `ka * a + kb * b` lane by lane, skipping a term whose coefficient is an exact zero.
pub(super) fn comb(p: usize, ka: f64, a: &Lanes, kb: f64, b: &Lanes) -> Lanes {
    let mut out = ZERO;
    if ka != 0.0 {
        for k in 0..p {
            out[k] = ka * a[k];
        }
    }
    if kb != 0.0 {
        for k in 0..p {
            out[k] += kb * b[k];
        }
    }
    out
}

pub(super) fn scale(p: usize, k: f64, a: &Lanes) -> Lanes {
    comb(p, k, a, 0.0, &ZERO)
}

/// `a / denominator` lane by lane.
pub(super) fn divide(p: usize, a: &Lanes, denominator: f64) -> Lanes {
    let mut out = ZERO;
    for k in 0..p {
        out[k] = a[k] / denominator;
    }
    out
}

/// The slope factor of `value.clamp(lo, hi)`: 1 inside, 0 outside or NaN, and the tie mean 1/2
/// on a bound.
pub(super) fn clamp_factor(value: f64, lo: f64, hi: f64) -> f64 {
    if value > lo && value < hi {
        1.0
    } else if value == lo || value == hi {
        0.5
    } else {
        0.0
    }
}

/// The tangent of `value.clamp(lo, hi)` with constant bounds.
pub(super) fn clamped(
    p: usize,
    value: f64,
    lo: f64,
    hi: f64,
    d: &Lanes,
    nonsmooth: &mut bool,
) -> Lanes {
    let factor = clamp_factor(value, lo, hi);
    if factor == 0.5 {
        *nonsmooth = true;
    }
    scale(p, factor, d)
}

/// The mean tangent of the operands whose value equals the selected `result` (Rust `==`, so -0.0
/// ties 0.0 and NaN never matches). Zero when nothing matches.
pub(super) fn tie_mean(p: usize, result: f64, operands: &[&Dual], nonsmooth: &mut bool) -> Lanes {
    let mut out = ZERO;
    let mut tied = 0usize;
    for operand in operands {
        if operand.v == result {
            for (lane, d) in out.iter_mut().zip(&operand.d).take(p) {
                *lane += d;
            }
            tied += 1;
        }
    }
    if tied > 1 {
        *nonsmooth = true;
        let share = 1.0 / tied as f64;
        for lane in out.iter_mut().take(p) {
            *lane *= share;
        }
    }
    out
}

pub(super) fn binary_tangent(
    p: usize,
    op: BinOp,
    a: &Dual,
    b: &Dual,
    v: f64,
    nonsmooth: &mut bool,
) -> Lanes {
    if !a.v.is_finite() || !b.v.is_finite() {
        *nonsmooth = true;
        return ZERO;
    }
    match op {
        BinOp::Add => comb(p, 1.0, &a.d, 1.0, &b.d),
        BinOp::Sub => comb(p, 1.0, &a.d, -1.0, &b.d),
        BinOp::Mul => comb(p, b.v, &a.d, a.v, &b.d),
        // The protected branch is a constant 0. Otherwise the slope is a'/b - a b'/b². Its second
        // term is formed on the mantissas and exponents of a, b' and b separately and joined
        // once, so no intermediate (q = a/b, q b', b²) under- or overflows: a lane is lost only
        // where it is itself out of range. A site lies in a or in b, so one term is zero; a term
        // with an exact zero factor is skipped, and an incoming lane that is not finite is passed
        // on.
        BinOp::Div if b.v.abs() < 1e-12 => ZERO,
        BinOp::Div => {
            let mut out = ZERO;
            let ((ma, ea), (mb, eb)) = (split(a.v), split(b.v));
            for (k, lane) in out.iter_mut().enumerate().take(p) {
                let direct = if a.d[k] == 0.0 { 0.0 } else { a.d[k] / b.v };
                let through_b = if b.d[k] == 0.0 || a.v == 0.0 {
                    0.0
                } else {
                    let (md, ed) = split(b.d[k]);
                    join(ma * md / (mb * mb), ea + ed - 2 * eb)
                };
                *lane = direct - through_b;
            }
            out
        }
        BinOp::Pow => pow_tangent(p, a, b, v, nonsmooth),
    }
}

/// `v = a.powf(e)` with `e = b.clamp(-8, 8)`, both operands finite.
pub(super) fn pow_tangent(p: usize, a: &Dual, b: &Dual, v: f64, nonsmooth: &mut bool) -> Lanes {
    let e = b.v.clamp(-8.0, 8.0);
    let exponent_factor = clamp_factor(b.v, -8.0, 8.0);
    if exponent_factor == 0.5 {
        *nonsmooth = true;
    }
    // e * a^(e - 1) keeps a tiny base's slope where v / a would lose it to underflow (e > 1). Where
    // it leaves the normal range anyway, `base_lanes` takes another form.
    let base_slope = |a: f64| if e == 0.0 { 0.0 } else { e * a.powf(e - 1.0) };
    let (by_base, by_exponent) = if a.v > 0.0 {
        (base_slope(a.v), v * a.v.ln())
    } else if a.v == 0.0 {
        if e == 1.0 {
            (1.0, 0.0)
        } else if e > 1.0 {
            (0.0, 0.0)
        } else if e > 0.0 {
            // The right slope is +inf.
            *nonsmooth = true;
            (0.0, 0.0)
        } else if e == 0.0 {
            // 0^e jumps among 0, 1 and inf at e = 0.
            *nonsmooth = true;
            (0.0, 0.0)
        } else {
            // inf, with POW_NONFINITE.
            (0.0, 0.0)
        }
    } else if e == e.trunc() {
        // A negative base has a value only at integral exponents.
        *nonsmooth = true;
        (base_slope(a.v), 0.0)
    } else {
        // NaN, with POW_NONFINITE.
        (0.0, 0.0)
    };
    // The base slope was computed, not set: a nonzero base (a negative one only at an integral
    // exponent) and e != 0. Only a computed slope can leave the normal range by rounding.
    let computed = e != 0.0 && (a.v > 0.0 || (a.v < 0.0 && e == e.trunc()));
    // Each term is checked: a finite slope times a large lane can still overflow. A site lies in
    // the base or the exponent, never both, so a lane's sum has one nonzero term and is checked
    // only defensively.
    let mut out = base_lanes(p, by_base, e, v, a, computed, nonsmooth);
    // A saturated exponent is a constant, so its term is an exact zero even where v ln a
    // overflows; so is a base of exactly 1, where ln a = 0.
    if exponent_factor != 0.0 && a.v > 0.0 && a.v != 1.0 {
        let by_exponent = exponent_factor * by_exponent;
        // v ln a is used directly while it and v are normal numbers. Otherwise (v under- or
        // overflowed, or is subnormal) each lane is formed in the log domain.
        let direct = by_exponent.is_finite()
            && by_exponent.abs() >= f64::MIN_POSITIVE
            && v >= f64::MIN_POSITIVE;
        let ln_a = a.v.ln();
        let log_c = exponent_factor.ln() + e * ln_a + ln_a.abs().ln();
        for (lane, d) in out.iter_mut().zip(&b.d).take(p) {
            if *d == 0.0 {
                continue;
            }
            let term = if direct {
                by_exponent * d
            } else {
                log_lane(ln_a < 0.0, log_c, *d)
            };
            let exponent_term = finite_or_zero(term, nonsmooth);
            *lane = finite_or_zero(*lane + exponent_term, nonsmooth);
        }
    }
    out
}

/// `value`, or 0 with the row marked nonsmooth when it is not finite.
pub(super) fn finite_or_zero(value: f64, nonsmooth: &mut bool) -> f64 {
    if value.is_finite() {
        value
    } else {
        *nonsmooth = true;
        0.0
    }
}

/// `e a^(e-1) a'` lane by lane. An explicit slope (`computed` false: a zero base, or e = 0), or a
/// computed `by_base` that is a finite normal number, is used directly. A computed one outside
/// that range is taken in the ratio form `(e v) (a'_k / a)` while `v` is normal and `e v` finite
/// (a large base with e < 0 underflows `e a^(e-1)` while `v` is normal); where `a'_k / a` or the
/// ratio leaves the range, the same product `e v a'_k / a` is formed on mantissas and exponents
/// ([`product`]). Otherwise (both underflow for a tiny base and e > 1) it is taken in the log
/// domain. A lane whose `a'_k` is 0 stays 0. A lane that is not finite is 0 and marks the row
/// nonsmooth.
pub(super) fn base_lanes(
    p: usize,
    by_base: f64,
    e: f64,
    v: f64,
    a: &Dual,
    computed: bool,
    nonsmooth: &mut bool,
) -> Lanes {
    let mut out = ZERO;
    if !computed || (by_base.is_finite() && by_base.abs() >= f64::MIN_POSITIVE) {
        if by_base != 0.0 {
            for (lane, d) in out.iter_mut().zip(&a.d).take(p) {
                *lane = finite_or_zero(by_base * d, nonsmooth);
            }
        }
        return out;
    }
    let ev = e * v;
    let ratio = v.abs() >= f64::MIN_POSITIVE && ev.is_finite();
    // e a^(e-1) is negative when e is, flipped for a negative base at an odd integral e - 1.
    let negative = (e < 0.0) != (a.v < 0.0 && (e - 1.0) % 2.0 != 0.0);
    let log_c = e.abs().ln() + (e - 1.0) * a.v.abs().ln();
    for (lane, d) in out.iter_mut().zip(&a.d).take(p) {
        if *d != 0.0 {
            let term = if ratio {
                // `a'_k / a` can leave the normal range (a tiny base) where the slope does not;
                // the same product is then formed on mantissas and exponents.
                let quotient = d / a.v;
                let direct = ev * quotient;
                if quotient.is_finite() && quotient.abs() >= f64::MIN_POSITIVE && direct.is_finite()
                {
                    direct
                } else {
                    product(&[e, v, *d], &[a.v])
                }
            } else {
                log_lane(negative, log_c, *d)
            };
            *lane = finite_or_zero(term, nonsmooth);
        }
    }
    out
}

pub(super) fn call_tangent(
    p: usize,
    func: Func,
    operands: &[&Dual],
    v: f64,
    nonsmooth: &mut bool,
) -> Lanes {
    if matches!(func, Func::Min | Func::Max | Func::Clip) {
        return tie_mean(p, v, operands, nonsmooth);
    }
    if operands.iter().any(|operand| !operand.v.is_finite()) {
        *nonsmooth = true;
        return ZERO;
    }
    let a = operands[0];
    match func {
        Func::Abs if a.v > 0.0 => a.d,
        Func::Abs if a.v < 0.0 => scale(p, -1.0, &a.d),
        Func::Abs => {
            // The mean of the one-sided tangents a' and -a'.
            *nonsmooth = true;
            ZERO
        }
        Func::Sqrt if a.v > 0.0 => scale(p, 0.5 / v, &a.d),
        Func::Sqrt => {
            // At 0 the right slope is +inf; below 0 the value is projected (SQRT_PROJECTED).
            if a.v == 0.0 {
                *nonsmooth = true;
            }
            ZERO
        }
        Func::Log1p if a.v > -1.0 => divide(p, &a.d, 1.0 + a.v),
        Func::Log1p => ZERO,
        Func::Sin => scale(p, a.v.cos(), &a.d),
        Func::Cos => scale(p, -a.v.sin(), &a.d),
        Func::Tanh => sech_squared_lanes(p, a.v, &a.d),
        Func::Sigmoid => {
            let factor = clamp_factor(a.v, -30.0, 30.0);
            if factor == 0.5 {
                *nonsmooth = true;
            }
            // σ (1 - σ) from the argument: within the clamp it is at least e^-30 / 4.
            let e = (-a.v.abs()).exp();
            scale(p, factor * (e / ((1.0 + e) * (1.0 + e))), &a.d)
        }
        Func::Exp => {
            let factor = clamp_factor(a.v, -30.0, 30.0);
            if factor == 0.5 {
                *nonsmooth = true;
            }
            // The clamp factor meets v before the lane: at the tie, v a' can overflow where the
            // half slope does not. Halving is exact, so elsewhere the bits are those of (v a') f.
            scale(p, factor * v, &a.d)
        }
        Func::Log if a.v > 1e-300 => divide(p, &a.d, a.v),
        Func::Log => ZERO,
        // The evaluator's asinh squares its argument too, so where v² overflows its value is not
        // finite and this slope is never used.
        Func::Asinh => divide(p, &a.d, (a.v * a.v + 1.0).sqrt()),
        // Past |v| ≈ 1.34e154 v² overflows while atan stays finite: a' / (1 + v²) is then taken
        // as (a' / v) / v, so the slope does not silently become 0.
        Func::Atan if (a.v * a.v).is_finite() => divide(p, &a.d, 1.0 + a.v * a.v),
        Func::Atan => divide(p, &divide(p, &a.d, a.v), a.v),
        Func::Erf => erf_lanes(p, a.v, &a.d),
        Func::Step | Func::Sign | Func::Floor | Func::Round => ZERO,
        Func::Min | Func::Max | Func::Clip => unreachable!("selectors return above"),
    }
}

/// `c a'` lane by lane for a positive slope coefficient `c = exp(log_c)`: directly while `c` is a
/// normal number, and in the log domain otherwise.
pub(super) fn tail_lanes(p: usize, c: f64, log_c: f64, d: &Lanes) -> Lanes {
    if c >= f64::MIN_POSITIVE {
        return scale(p, c, d);
    }
    let mut out = ZERO;
    for (lane, d) in out.iter_mut().zip(d).take(p) {
        *lane = log_lane(false, log_c, *d);
    }
    out
}

/// `sech²(x) a'` lane by lane, from the argument: `4 e / (1 + e)²` with `e = e^(-2|x|)`. Unlike
/// `1 - tanh²`, it does not cancel to 0 once `tanh` rounds to ±1 (`|x|` above about 19.1).
pub(super) fn sech_squared_lanes(p: usize, x: f64, d: &Lanes) -> Lanes {
    let e = (-2.0 * x.abs()).exp();
    let c = 4.0 * e / ((1.0 + e) * (1.0 + e));
    tail_lanes(p, c, 4f64.ln() - 2.0 * x.abs() - 2.0 * e.ln_1p(), d)
}

/// The derivative of the evaluator's `erf` (Abramowitz and Stegun 7.1.26, the same constants),
/// valid at 0: with `u = |v|`, `t = 1 / (1 + p u)`, `P(t) = a1 t + … + a5 t⁵` and `E = e^(-u²)`,
/// the value is `sign(v) (1 - P(t) E)`, whose slope is `E (p t² P'(t) + 2 u P(t))` on both sides.
/// The coefficients sum to `1 - 1e-9`, so the value itself steps by about `2e-9` across 0; this
/// slope ignores that step and does not mark the row nonsmooth for it. Where `E` leaves the normal
/// range (`u` above about 26.5) the lanes are formed in the log domain; where `u²` overflows the
/// slope is an exact 0.
pub(super) fn erf_lanes(p: usize, v: f64, d: &Lanes) -> Lanes {
    let u = v.abs();
    if !(u * u).is_finite() {
        return ZERO;
    }
    let bracket = erf_bracket(u);
    let tail = (-(u * u)).exp();
    tail_lanes(p, tail * bracket, bracket.ln() - u * u, d)
}

/// `p t² P'(t) + 2 u P(t)` of [`erf_lanes`], positive for `u ≥ 0`.
pub(super) fn erf_bracket(u: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * u);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let slope = 0.254829592
        + t * (2.0 * -0.284496736
            + t * (3.0 * 1.421413741 + t * (4.0 * -1.453152027 + t * (5.0 * 1.061405429))));
    0.3275911 * t * t * slope + 2.0 * u * poly
}
