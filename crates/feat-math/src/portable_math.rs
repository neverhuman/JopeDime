//! Portable, bit-reproducible f64 transcendental kernel.
//!
//! Every routine here is expressed using ONLY IEEE-754 correctly-rounded
//! operations (`+ - * /`, `sqrt`) plus exact integer/bit manipulation
//! (`to_bits`/`from_bits`, shifts, `floor`/`trunc`). Those operations are
//! specified identically by IEEE-754 and by the language standards of Rust,
//! Python (CPython float), JavaScript (ECMAScript Number), and C++ (`double`),
//! so a faithful port of each algorithm computes the SAME bits on every
//! backend — unlike the platform `libm` `exp`/`log`/`sin`/... which differ
//! implementation-to-implementation.
//!
//! The algorithms are faithful ports of the public-domain SunSoft fdlibm
//! reference (also the basis of musl / Rust's `libm` crate). Constants are the
//! fdlibm constants verbatim. This is the deeper level of the same technique
//! the crate already uses for `asinh`/`erf`/`probit`: those removed the
//! divergent *composite*, this removes the divergent *primitive* underneath.
//!
//! This scalar increment covers the exp/log family, `pow`, `tanh`, `sin`,
//! `cos`, `atan`, and `atan2`; emitted-language routing is a separate gate.

// These are deliberate fdlibm constants and IEEE special-value constructions.
// Replacing them with rounded standard constants or stylistic NaN expressions
// would change the reference bits that every backend must reproduce.
#![allow(clippy::approx_constant, clippy::eq_op, clippy::excessive_precision)]

// ------------------------------- bit helpers -------------------------------

#[inline]
fn high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline]
fn low_word(x: f64) -> u32 {
    x.to_bits() as u32
}

/// Replace the high 32 bits of `x`, keeping its low 32 bits.
#[inline]
fn with_high_word(x: f64, hi: u32) -> f64 {
    f64::from_bits(((hi as u64) << 32) | (x.to_bits() & 0xffff_ffff))
}

/// Replace the low 32 bits of `x`, keeping its high 32 bits.
#[inline]
fn with_low_word(x: f64, lo: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff_0000_0000) | (lo as u64))
}

/// |x| via clearing the sign bit.
#[inline]
fn fabs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

#[inline]
fn canonical_nan() -> f64 {
    f64::from_bits(0x7ff8_0000_0000_0000)
}

/// 2^n as an exact power of two for n in the normal exponent range.
#[inline]
fn two_pow(n: i32) -> f64 {
    f64::from_bits((((0x3ff + n) as i64) as u64) << 52)
}

/// `x * 2^n`, correct (with the fdlibm/musl three-step guard) across the full
/// exponent range including gradual underflow — no `libm` dependency.
fn scalbn(x: f64, mut n: i32) -> f64 {
    // 2^1023, 2^-1022, 2^53 as exact bit patterns.
    let p1023 = two_pow(1023);
    let pm1022 = two_pow(-1022);
    let p53 = two_pow(53);
    let mut y = x;
    if n > 1023 {
        y *= p1023;
        n -= 1023;
        if n > 1023 {
            y *= p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // Keep the residual n > -53 so the final multiply rounds once.
        y *= pm1022 * p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= pm1022 * p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * two_pow(n)
}

// ---------------------------------- exp ------------------------------------

/// e^x. Faithful port of fdlibm `e_exp.c`.
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return canonical_nan();
    }
    const HALF: [f64; 2] = [0.5, -0.5];
    const LN2HI: [f64; 2] = [
        6.931_471_803_691_238_164_9e-1,
        -6.931_471_803_691_238_164_9e-1,
    ];
    const LN2LO: [f64; 2] = [
        1.908_214_929_270_587_700_0e-10,
        -1.908_214_929_270_587_700_0e-10,
    ];
    const INVLN2: f64 = 1.442_695_040_888_963_387_0;
    const P1: f64 = 1.666_666_666_666_660_190_37e-1;
    const P2: f64 = -2.777_777_777_701_559_338_42e-3;
    const P3: f64 = 6.613_756_321_437_934_361_17e-5;
    const P4: f64 = -1.653_390_220_546_525_153_90e-6;
    const P5: f64 = 4.138_136_797_057_238_460_39e-8;
    const HUGE: f64 = 1.0e300;
    const TWOM1000: f64 = 9.332_636_185_032_188_789_90e-302;
    const O_THRESHOLD: f64 = 7.097_827_128_933_839_730_96e2;
    const U_THRESHOLD: f64 = -7.451_332_191_019_411_084_20e2;

    let mut hx = high_word(x);
    let xsb = ((hx >> 31) & 1) as usize;
    hx &= 0x7fff_ffff;

    // Non-finite / overflow / underflow filter.
    if hx >= 0x4086_2E42 {
        if hx >= 0x7ff0_0000 {
            if ((hx & 0xf_ffff) | low_word(x)) != 0 {
                return x + x; // NaN
            }
            return if xsb == 0 { x } else { 0.0 }; // exp(+inf)=inf, exp(-inf)=0
        }
        if x > O_THRESHOLD {
            return HUGE * HUGE; // overflow -> +inf
        }
        if x < U_THRESHOLD {
            return TWOM1000 * TWOM1000; // underflow -> 0
        }
    }

    let mut k: i32 = 0;
    let mut hi = 0.0;
    let mut lo = 0.0;
    let xr;
    if hx > 0x3fd6_2e42 {
        // |x| > 0.5*ln2
        if hx < 0x3FF0_A2B2 {
            // |x| < 1.5*ln2
            hi = x - LN2HI[xsb];
            lo = LN2LO[xsb];
            k = 1 - (xsb as i32) - (xsb as i32);
        } else {
            k = (INVLN2 * x + HALF[xsb]) as i32;
            let t = k as f64;
            hi = x - t * LN2HI[0]; // t*ln2HI is exact
            lo = t * LN2LO[0];
        }
        xr = hi - lo;
    } else if hx < 0x3e30_0000 {
        // |x| < 2^-28: exp(x) ~= 1 + x
        return 1.0 + x;
    } else {
        xr = x; // k = 0
    }

    let t = xr * xr;
    let c = xr - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 {
        return 1.0 - ((xr * c) / (c - 2.0) - xr);
    }
    let y = 1.0 - ((lo - (xr * c) / (2.0 - c)) - hi);
    scalbn(y, k)
}

// ---------------------------------- ln -------------------------------------

/// Natural log. Faithful port of fdlibm `e_log.c`.
pub fn ln(x: f64) -> f64 {
    if x.is_nan() {
        return canonical_nan();
    }
    const LN2_HI: f64 = 6.931_471_803_691_238_164_9e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_0e-10;
    const TWO54: f64 = 1.801_439_850_948_198_4e16;
    const LG1: f64 = 6.666_666_666_666_735_130e-1;
    const LG2: f64 = 3.999_999_999_940_941_908e-1;
    const LG3: f64 = 2.857_142_874_366_239_149e-1;
    const LG4: f64 = 2.222_219_843_214_978_396e-1;
    const LG5: f64 = 1.818_357_216_161_805_012e-1;
    const LG6: f64 = 1.531_383_769_920_937_332e-1;
    const LG7: f64 = 1.479_819_860_511_658_591e-1;

    let mut hx = high_word(x) as i32;
    let lx = low_word(x);
    let mut k: i32 = 0;
    let mut xx = x;

    if hx < 0x0010_0000 {
        // x < 2^-1022 (subnormal or zero)
        if ((hx & 0x7fff_ffff) | (lx as i32)) == 0 {
            return -TWO54 / 0.0; // ln(+-0) = -inf
        }
        if hx < 0 {
            return canonical_nan(); // ln(negative) = NaN
        }
        k -= 54;
        xx *= TWO54; // normalize subnormal
        hx = high_word(xx) as i32;
    }
    if hx >= 0x7ff0_0000 {
        return x + x; // +inf or NaN
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x95f64) & 0x100000;
    xx = with_high_word(xx, (hx | (i ^ 0x3ff0_0000)) as u32); // normalize to [sqrt(2)/2, sqrt(2))
    k += i >> 20;
    let f = xx - 1.0;

    if (0x000f_ffff & (2 + hx)) < 3 {
        // |f| < 2^-20
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            let dk = k as f64;
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - 0.333_333_333_333_333_33 * f);
        if k == 0 {
            return f - r;
        }
        let dk = k as f64;
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }

    let s = f / (2.0 + f);
    let dk = k as f64;
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let i2 = hx - 0x6147a;
    let j2 = 0x6b851 - hx;
    if (i2 | j2) > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

// --------------------------------- log1p -----------------------------------

/// ln(1 + x), accurate near 0. Faithful port of musl `log1p.c`.
pub fn log1p(x: f64) -> f64 {
    if x.is_nan() {
        return canonical_nan();
    }
    const LN2_HI: f64 = 6.931_471_803_691_238_164_9e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_0e-10;
    const LG1: f64 = 6.666_666_666_666_735_130e-1;
    const LG2: f64 = 3.999_999_999_940_941_908e-1;
    const LG3: f64 = 2.857_142_874_366_239_149e-1;
    const LG4: f64 = 2.222_219_843_214_978_396e-1;
    const LG5: f64 = 1.818_357_216_161_805_012e-1;
    const LG6: f64 = 1.531_383_769_920_937_332e-1;
    const LG7: f64 = 1.479_819_860_511_658_591e-1;

    let hx = high_word(x);
    let mut k: i32 = 1;
    let mut f = 0.0;
    let mut c = 0.0;

    if hx < 0x3fda_827a || (hx >> 31) != 0 {
        // 1 + x < sqrt(2)
        if hx >= 0xbff0_0000 {
            // x <= -1.0
            if x == -1.0 {
                return x / 0.0; // -inf
            }
            return canonical_nan();
        }
        if (hx << 1) < (0x3ca0_0000u32 << 1) {
            // |x| < 2^-53
            return x;
        }
        if hx <= 0xbfd2_bec4 {
            // sqrt(2)/2 <= 1+x < sqrt(2)
            k = 0;
            c = 0.0;
            f = x;
        }
    } else if hx >= 0x7ff0_0000 {
        return x; // +inf or NaN
    }

    if k != 0 {
        let u = 1.0 + x;
        let mut hu = high_word(u);
        hu = hu.wrapping_add(0x3ff0_0000 - 0x3fe6_a09e);
        k = ((hu >> 20) as i32) - 0x3ff;
        // Correction term ~ log(1+x) - log(u), avoiding underflow in c/u.
        if k < 54 {
            c = if k >= 2 { 1.0 - (u - x) } else { x - (u - 1.0) };
            c /= u;
        } else {
            c = 0.0;
        }
        // Reduce u into [sqrt(2)/2, sqrt(2)].
        hu = (hu & 0x000f_ffff).wrapping_add(0x3fe6_a09e);
        f = with_high_word(u, hu) - 1.0;
    }

    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let dk = k as f64;
    s * (hfsq + r) + (dk * LN2_LO + c) - hfsq + f + dk * LN2_HI
}

// --------------------------------- expm1 -----------------------------------

/// e^x - 1, accurate near 0. Faithful port of musl `expm1.c`.
pub fn expm1(x: f64) -> f64 {
    if x.is_nan() {
        return canonical_nan();
    }
    const LN2_HI: f64 = 6.931_471_803_691_238_164_9e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
    const INVLN2: f64 = 1.442_695_040_888_963_387_0;
    const Q1: f64 = -3.333_333_333_333_313_154_11e-2;
    const Q2: f64 = 1.587_301_587_254_814_601_65e-3;
    const Q3: f64 = -7.936_507_578_674_879_424_73e-5;
    const Q4: f64 = 4.008_217_827_329_362_395_52e-6;
    const Q5: f64 = -2.010_992_181_836_243_713_26e-7;

    let hx = high_word(x) & 0x7fff_ffff;
    let sign = (high_word(x) >> 31) != 0;

    let mut c = 0.0;
    let k: i32;
    let mut xx = x;

    // Filter huge / non-finite.
    if hx >= 0x4043_687A {
        // |x| >= 56*ln2
        if sign {
            return -1.0; // x <= -56*ln2 -> -1
        }
        if hx >= 0x4086_2E42 {
            // |x| >= 709.78 -> overflow
            return x * two_pow(1023); // inf
        }
    }

    if hx > 0x3fd6_2e42 {
        // |x| > 0.5*ln2
        let hi;
        let lo;
        if hx < 0x3FF0_A2B2 {
            // |x| < 1.5*ln2
            if !sign {
                hi = x - LN2_HI;
                lo = LN2_LO;
                k = 1;
            } else {
                hi = x + LN2_HI;
                lo = -LN2_LO;
                k = -1;
            }
        } else {
            k = (INVLN2 * x + if sign { -0.5 } else { 0.5 }) as i32;
            let t = k as f64;
            hi = x - t * LN2_HI; // t*ln2_hi is exact
            lo = t * LN2_LO;
        }
        xx = hi - lo;
        c = (hi - xx) - lo;
    } else if hx < 0x3c90_0000 {
        // |x| < 2^-54
        return x;
    } else {
        k = 0;
    }

    // Primary range.
    let hfx = 0.5 * xx;
    let hxs = xx * hfx;
    let r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    let t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - xx * t));
    if k == 0 {
        return xx - (xx * e - hxs); // c is 0
    }
    e = xx * (e - c) - c;
    e -= hxs;
    // exp(x) ~ 2^k (x_reduced - e + 1)
    if k == -1 {
        return 0.5 * (xx - e) - 0.5;
    }
    if k == 1 {
        if xx < -0.25 {
            return -2.0 * (e - (xx + 0.5));
        }
        return 1.0 + 2.0 * (xx - e);
    }
    let twopk = two_pow(k); // 2^k
    if !(0..=56).contains(&k) {
        // exp(x) - 1 suffices
        let y = (xx - e + 1.0) * twopk;
        return y - 1.0;
    }
    let inv = two_pow(-k); // 2^-k
    if k < 20 {
        (xx - e + (1.0 - inv)) * twopk
    } else {
        (xx - e - inv + 1.0) * twopk
    }
}

// ---------------------------------- pow ------------------------------------

#[path = "portable_math/power.rs"]
mod power;
#[path = "portable_math/trig.rs"]
mod trig;

pub use self::power::{pow, tanh};
pub use self::trig::{atan, atan2, cos, sin};

#[path = "portable_math/frozen_s1.rs"]
pub mod frozen_s1;
#[cfg(test)]
#[path = "portable_math/tests.rs"]
mod tests;
