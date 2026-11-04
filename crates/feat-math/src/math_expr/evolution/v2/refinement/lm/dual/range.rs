/// The most parameters one pass differentiates.
pub(crate) const MAX_PARAMS: usize = 16;

/// One tangent lane per parameter. Lanes at and above the parameter count stay zero.
pub(crate) type Lanes = [f64; MAX_PARAMS];

pub(crate) const ZERO: Lanes = [0.0; MAX_PARAMS];

/// 2^600 and 2^-600. Multiplying by either is exact while the product stays a normal number.
pub(super) const SCALE_UP: f64 = f64::from_bits((1023 + 600) << 52);
pub(super) const SCALE_DOWN: f64 = f64::from_bits((1023 - 600) << 52);

/// `x = m · 2^e` with `1 ≤ |m| < 2` for a finite nonzero `x`, subnormals included. Zero and
/// non-finite `x` come back as `(x, 0)`.
pub(crate) fn split(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    // A subnormal is first scaled into the normal range, exactly.
    let (x, shift) = if x.abs() < f64::MIN_POSITIVE {
        (x * SCALE_UP, -600)
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mantissa = f64::from_bits((bits & !(0x7ff_u64 << 52)) | (1023_u64 << 52));
    (mantissa, exponent + shift)
}

/// `m · 2^e`, rounded once: the steps before the last are exact power-of-two scalings that keep
/// the value normal, so only the last can round (to a subnormal) or overflow.
pub(crate) fn join(m: f64, e: i32) -> f64 {
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let (mut x, mut e) = (m, e);
    while e > 600 {
        x *= SCALE_UP;
        e -= 600;
    }
    while e < -600 {
        x *= SCALE_DOWN;
        e += 600;
    }
    x * f64::from_bits(((e + 1023) as u64) << 52)
}

/// `Π numerators / Π denominators`, formed on mantissas and exponents separately and joined once,
/// so no intermediate under- or overflows: the result is lost only where it is itself out of range.
/// A zero numerator gives 0; a denominator is never 0.
pub(crate) fn product(numerators: &[f64], denominators: &[f64]) -> f64 {
    let (mut m, mut e) = (1.0f64, 0i32);
    for &x in numerators {
        if x == 0.0 {
            return 0.0;
        }
        let (mx, ex) = split(x);
        m *= mx;
        e += ex;
    }
    for &x in denominators {
        let (mx, ex) = split(x);
        m /= mx;
        e -= ex;
    }
    join(m, e)
}

/// `±exp(log_c) · d` for a slope coefficient `±exp(log_c)` outside the normal range, formed in the
/// log domain so a large or small lane still gets its product. An underflow is an exact +0.
pub(crate) fn log_lane(negative: bool, log_c: f64, d: f64) -> f64 {
    if d == 0.0 || log_c == f64::NEG_INFINITY {
        return 0.0;
    }
    let r = (log_c + d.abs().ln()).exp();
    if r == 0.0 {
        0.0
    } else if negative != (d < 0.0) {
        -r
    } else {
        r
    }
}
