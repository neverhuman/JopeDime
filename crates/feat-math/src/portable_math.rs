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

/// x^y. Faithful port of fdlibm `e_pow.c`.
pub fn pow(x: f64, y: f64) -> f64 {
    const BP: [f64; 2] = [1.0, 1.5];
    const DP_H: [f64; 2] = [0.0, 5.849_624_872_207_641_601_56e-1];
    const DP_L: [f64; 2] = [0.0, 1.350_039_202_129_748_971_28e-8];
    const TWO53: f64 = 9_007_199_254_740_992.0;
    const HUGE: f64 = 1.0e300;
    const TINY: f64 = 1.0e-300;
    const L1: f64 = 5.999_999_999_999_946_487_25e-1;
    const L2: f64 = 4.285_714_285_785_501_842_52e-1;
    const L3: f64 = 3.333_333_298_183_774_329_18e-1;
    const L4: f64 = 2.727_281_238_085_340_064_89e-1;
    const L5: f64 = 2.306_607_457_755_613_663_31e-1;
    const L6: f64 = 2.069_750_178_003_384_177_84e-1;
    const P1: f64 = 1.666_666_666_666_660_190_37e-1;
    const P2: f64 = -2.777_777_777_701_559_338_42e-3;
    const P3: f64 = 6.613_756_321_437_934_361_17e-5;
    const P4: f64 = -1.653_390_220_546_525_153_90e-6;
    const P5: f64 = 4.138_136_797_057_238_460_39e-8;
    const LG2: f64 = 6.931_471_805_599_452_862_27e-1;
    const LG2_H: f64 = 6.931_471_824_645_996_093_75e-1;
    const LG2_L: f64 = -1.904_654_299_957_768_045_25e-9;
    const OVT: f64 = 8.008_566_259_537_294_437_2e-17;
    const CP: f64 = 9.617_966_939_259_755_543_29e-1;
    const CP_H: f64 = 9.617_967_009_544_372_558_59e-1;
    const CP_L: f64 = -7.028_461_650_952_758_265_16e-9;
    const IVLN2: f64 = 1.442_695_040_888_963_387_0;
    const IVLN2_H: f64 = 1.442_695_021_629_333_496_09;
    const IVLN2_L: f64 = 1.925_962_991_126_617_468_87e-8;

    let hx = high_word(x) as i32;
    let lx = low_word(x);
    let hy = high_word(y) as i32;
    let ly = low_word(y);
    let mut ix = hx & 0x7fff_ffff;
    let iy = hy & 0x7fff_ffff;

    // y == 0: x^0 = 1
    if (iy | (ly as i32)) == 0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return canonical_nan();
    }
    // ±NaN -> x + y
    if ix > 0x7ff0_0000
        || (ix == 0x7ff0_0000 && lx != 0)
        || iy > 0x7ff0_0000
        || (iy == 0x7ff0_0000 && ly != 0)
    {
        return canonical_nan();
    }

    // Is y an odd/even integer when x < 0?  0=not int, 1=odd, 2=even.
    let mut yisint = 0;
    if hx < 0 {
        if iy >= 0x4340_0000 {
            yisint = 2;
        } else if iy >= 0x3ff0_0000 {
            let k = (iy >> 20) - 0x3ff;
            if k > 20 {
                let j = ly >> (52 - k);
                if (j << (52 - k)) == ly {
                    yisint = 2 - (j & 1) as i32;
                }
            } else if ly == 0 {
                let j = iy >> (20 - k);
                if (j << (20 - k)) == iy {
                    yisint = 2 - (j & 1);
                }
            }
        }
    }

    // Special values of y.
    if ly == 0 {
        if iy == 0x7ff0_0000 {
            // y is ±inf
            if ((ix - 0x3ff0_0000) | (lx as i32)) == 0 {
                return 1.0; // (±1)^±inf = 1
            } else if ix >= 0x3ff0_0000 {
                return if hy >= 0 { y } else { 0.0 };
            } else {
                return if hy < 0 { -y } else { 0.0 };
            }
        }
        if iy == 0x3ff0_0000 {
            return if hy < 0 { 1.0 / x } else { x }; // y = ±1
        }
        if hy == 0x4000_0000 {
            return x * x; // y = 2
        }
        if hy == 0x3fe0_0000 && hx >= 0 {
            return x.sqrt(); // y = 0.5, x >= 0
        }
    }

    let mut ax = fabs(x);
    // Special values of x.
    if lx == 0 && (ix == 0x7ff0_0000 || ix == 0 || ix == 0x3ff0_0000) {
        let mut z = ax; // ±0, ±inf, ±1
        if hy < 0 {
            z = 1.0 / z;
        }
        if hx < 0 {
            if ((ix - 0x3ff0_0000) | yisint) == 0 {
                z = canonical_nan(); // (-1)^non-int = NaN
            } else if yisint == 1 {
                z = -z;
            }
        }
        return z;
    }

    let mut n = (hx >> 31) + 1;
    // (x<0)^(non-int) is NaN
    if (n | yisint) == 0 {
        return canonical_nan();
    }

    let mut s = 1.0; // sign of result
    if (n | (yisint - 1)) == 0 {
        s = -1.0; // (-ve)^(odd int)
    }

    let t1;
    let t2;
    if iy > 0x41e0_0000 {
        // |y| > 2^31
        if iy > 0x43f0_0000 {
            // |y| > 2^64: over/underflow
            if ix <= 0x3fef_ffff {
                return if hy < 0 { HUGE * HUGE } else { TINY * TINY };
            }
            if ix >= 0x3ff0_0000 {
                return if hy > 0 { HUGE * HUGE } else { TINY * TINY };
            }
        }
        if ix < 0x3fef_ffff {
            return if hy < 0 {
                s * HUGE * HUGE
            } else {
                s * TINY * TINY
            };
        }
        if ix > 0x3ff0_0000 {
            return if hy > 0 {
                s * HUGE * HUGE
            } else {
                s * TINY * TINY
            };
        }
        // |1-x| tiny: log(x) ~= x - x^2/2 + ...
        let t = ax - 1.0;
        let w = (t * t) * (0.5 - t * (0.333_333_333_333_333_333_333 - t * 0.25));
        let u = IVLN2_H * t;
        let v = t * IVLN2_L - w * IVLN2;
        let t1v = with_low_word(u + v, 0);
        t1 = t1v;
        t2 = v - (t1v - u);
    } else {
        n = 0;
        // subnormal x
        if ix < 0x0010_0000 {
            ax *= TWO53;
            n -= 53;
            ix = high_word(ax) as i32;
        }
        n += (ix >> 20) - 0x3ff;
        let j0 = ix & 0x000f_ffff;
        ix = j0 | 0x3ff0_0000; // normalize
        let k;
        if j0 <= 0x3988E {
            k = 0; // |x| < sqrt(3/2)
        } else if j0 < 0xBB67A {
            k = 1; // |x| < sqrt(3)
        } else {
            k = 0;
            n += 1;
            ix -= 0x0010_0000;
        }
        ax = with_high_word(ax, ix as u32);

        // ss = s_h + s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5)
        let u = ax - BP[k];
        let v = 1.0 / (ax + BP[k]);
        let ss = u * v;
        let s_h = with_low_word(ss, 0);
        // t_h = ax + bp[k] (high)
        let t_h = with_high_word(
            0.0,
            (((ix >> 1) | 0x2000_0000) + 0x0008_0000 + ((k as i32) << 18)) as u32,
        );
        let t_l = ax - (t_h - BP[k]);
        let s_l = v * ((u - s_h * t_h) - s_h * t_l);
        // log(ax)
        let s2 = ss * ss;
        let mut r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + ss);
        let s2b = s_h * s_h;
        let t_h2 = with_low_word(3.0 + s2b + r, 0);
        let t_l2 = r - ((t_h2 - 3.0) - s2b);
        // u + v = ss*(1 + ...)
        let u2 = s_h * t_h2;
        let v2 = s_l * t_h2 + t_l2 * ss;
        // 2/(3log2)*(ss + ...)
        let p_h = with_low_word(u2 + v2, 0);
        let p_l = v2 - (p_h - u2);
        let z_h = CP_H * p_h;
        let z_l = CP_L * p_h + p_l * CP + DP_L[k];
        // log2(ax) = n + dp_h + z_h + z_l
        let t = n as f64;
        let t1v = with_low_word(((z_h + z_l) + DP_H[k]) + t, 0);
        t1 = t1v;
        t2 = z_l - (((t1v - t) - DP_H[k]) - z_h);
    }

    // Split y into y1 + y2 and compute (y1+y2)*(t1+t2).
    let y1 = with_low_word(y, 0);
    let p_l = (y - y1) * t1 + y * t2;
    let mut p_h = y1 * t1;
    let z = p_l + p_h;
    let mut j = high_word(z) as i32;
    let i = low_word(z);
    if j >= 0x4090_0000 {
        // z >= 1024
        if ((j - 0x4090_0000) | (i as i32)) != 0 {
            return s * HUGE * HUGE; // overflow
        }
        if p_l + OVT > z - p_h {
            return s * HUGE * HUGE;
        }
    } else if (j & 0x7fff_ffff) >= 0x4090_cc00 {
        // z <= -1075
        if ((j.wrapping_sub(0xc090_cc00u32 as i32)) | (i as i32)) != 0 {
            return s * TINY * TINY; // underflow
        }
        if p_l <= z - p_h {
            return s * TINY * TINY;
        }
    }

    // Compute 2^(p_h + p_l).
    let iabs = j & 0x7fff_ffff;
    let mut k = (iabs >> 20) - 0x3ff;
    let mut nn = 0;
    if iabs > 0x3fe0_0000 {
        // |z| > 0.5: n = [z + 0.5]
        nn = j + (0x0010_0000 >> (k + 1));
        k = ((nn & 0x7fff_ffff) >> 20) - 0x3ff;
        let t = with_high_word(0.0, (nn & !(0x000f_ffff >> k)) as u32);
        nn = ((nn & 0x000f_ffff) | 0x0010_0000) >> (20 - k);
        if j < 0 {
            nn = -nn;
        }
        p_h -= t;
    }
    let t = with_low_word(p_l + p_h, 0);
    let u = t * LG2_H;
    let v = (p_l - (t - p_h)) * LG2 + t * LG2_L;
    let mut z2 = u + v;
    let w = v - (z2 - u);
    let tt = z2 * z2;
    let t1b = z2 - tt * (P1 + tt * (P2 + tt * (P3 + tt * (P4 + tt * P5))));
    let r = (z2 * t1b) / (t1b - 2.0) - (w + z2 * w);
    z2 = 1.0 - (r - z2);
    j = high_word(z2) as i32;
    j += nn << 20;
    if (j >> 20) <= 0 {
        z2 = scalbn(z2, nn); // subnormal output
    } else {
        z2 = with_high_word(z2, j as u32);
    }
    s * z2
}

// ---------------------------------- tanh -----------------------------------

/// tanh(x). Faithful port of musl `tanh.c` (built on `expm1`).
pub fn tanh(x: f64) -> f64 {
    if x.is_nan() {
        return canonical_nan();
    }
    let sign = (x.to_bits() >> 63) != 0;
    let ax = f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff); // |x|
    let w = (ax.to_bits() >> 32) as u32;

    let t;
    if w > 0x3fe1_93ea {
        // |x| > log(3)/2 ~= 0.5493 (or NaN)
        if w > 0x4034_0000 {
            // |x| > 20 (or NaN): tanh -> 1 (0/x avoids overflow; NaN propagates)
            t = 1.0 - 0.0 / ax;
        } else {
            let e = expm1(2.0 * ax);
            t = 1.0 - 2.0 / (e + 2.0);
        }
    } else if w > 0x3fd0_58ae {
        // |x| > log(5/3)/2 ~= 0.2554
        let e = expm1(2.0 * ax);
        t = e / (e + 2.0);
    } else if w >= 0x0010_0000 {
        // |x| >= 2^-1022
        let e = expm1(-2.0 * ax);
        t = -e / (e + 2.0);
    } else {
        // subnormal
        t = ax;
    }
    if sign {
        -t
    } else {
        t
    }
}

// ------------------------------ sin / cos ----------------------------------

/// fdlibm `__kernel_sin`: sin(x + y) for |x| <= pi/4, y a tail term.
fn kernel_sin(x: f64, y: f64, iy: i32) -> f64 {
    const S1: f64 = -1.666_666_666_666_663_243_48e-1;
    const S2: f64 = 8.333_333_333_322_489_461_24e-3;
    const S3: f64 = -1.984_126_982_985_794_931_34e-4;
    const S4: f64 = 2.755_731_370_707_006_767_89e-6;
    const S5: f64 = -2.505_076_025_340_686_341_95e-8;
    const S6: f64 = 1.589_690_995_211_550_102_21e-10;
    let z = x * x;
    let w = z * z;
    let r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
    let v = z * x;
    if iy == 0 {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

/// fdlibm `__kernel_cos`: cos(x + y) for |x| <= pi/4.
fn kernel_cos(x: f64, y: f64) -> f64 {
    const C1: f64 = 4.166_666_666_666_660_190_37e-2;
    const C2: f64 = -1.388_888_888_887_410_957_49e-3;
    const C3: f64 = 2.480_158_728_947_672_941_78e-5;
    const C4: f64 = -2.755_731_435_139_066_330_35e-7;
    const C5: f64 = 2.087_572_321_298_174_827_90e-9;
    const C6: f64 = -1.135_964_755_778_819_482_65e-11;
    let z = x * x;
    let w = z * z;
    let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let w2 = 1.0 - hz;
    w2 + (((1.0 - w2) - hz) + (z * r - x * y))
}

/// Argument reduction x -> n mod 4 (returned) + (y0, y1) with x = n*(pi/2) +
/// y0 + y1, |y0| <= pi/4. Uses fdlibm's exact medium Cody-Waite reduction for
/// |x| < 2^20*(pi/2) ~= 1.647e6 (covers every realistic genome use: RFF,
/// Box-Muller, standardized features). Beyond that, a double-double 2*pi
/// pre-reduction brings x into range: the result stays finite, deterministic,
/// and in [-1,1], but accuracy degrades for astronomically large |x| (which
/// carry no modeling meaning). Bit-identical across all backends regardless.
fn rem_pio2(x: f64) -> (i32, f64, f64) {
    const INVPIO2: f64 = 6.366_197_723_675_813_824_33e-1;
    const PIO2_1: f64 = 1.570_796_326_734_125_614_17;
    const PIO2_1T: f64 = 6.077_100_506_506_192_249_32e-11;
    const PIO2_2: f64 = 6.077_100_506_303_965_976_60e-11;
    const PIO2_2T: f64 = 2.022_266_248_795_950_631_54e-21;
    const PIO2_3: f64 = 2.022_266_248_711_166_455_80e-21;
    const PIO2_3T: f64 = 8.478_427_660_368_899_569_97e-32;
    const INV_TWO_PI: f64 = 1.591_549_430_918_953_357_69e-1; // 1/(2pi)
    const TWO_PI_HI: f64 = 6.283_185_307_179_586_231_99; // 2pi high
    const TWO_PI_LO: f64 = 2.449_293_598_294_706_414_71e-16; // 2pi tail

    let ix = high_word(x) & 0x7fff_ffff;

    if ix >= 0x7ff0_0000 {
        let z = canonical_nan(); // inf/nan
        return (0, z, z);
    }

    // Large-|x| pre-reduction into [-pi, pi] via double-double 2pi.
    let mut xr = x;
    while (high_word(xr) & 0x7fff_ffff) >= 0x4139_21fb {
        // |xr| >= 2^20*(pi/2)
        let k = (xr * INV_TWO_PI).round(); // exact for |k| < 2^53
        xr = (xr - k * TWO_PI_HI) - k * TWO_PI_LO;
    }

    // Medium Cody-Waite reduction (exact for |xr| < 2^20*(pi/2)).
    let ixr = high_word(xr) & 0x7fff_ffff;
    if ixr <= 0x3fe9_21fb {
        // |xr| <= pi/4: no reduction
        return (0, xr, 0.0);
    }
    let half = if xr > 0.0 { 0.5 } else { -0.5 };
    let n = (xr * INVPIO2 + half) as i32;
    let fnn = n as f64;
    let mut r = xr - fnn * PIO2_1;
    let mut w = fnn * PIO2_1T;
    let j = (ixr >> 20) as i32;
    let mut y0 = r - w;
    let mut hi = (high_word(y0) >> 20) as i32 & 0x7ff;
    let mut i = j - hi;
    if i > 16 {
        // 2nd iteration
        let t = r;
        w = fnn * PIO2_2;
        r = t - w;
        w = fnn * PIO2_2T - ((t - r) - w);
        y0 = r - w;
        hi = (high_word(y0) >> 20) as i32 & 0x7ff;
        i = j - hi;
        if i > 49 {
            // 3rd iteration
            let t = r;
            w = fnn * PIO2_3;
            r = t - w;
            w = fnn * PIO2_3T - ((t - r) - w);
            y0 = r - w;
        }
    }
    let y1 = (r - y0) - w;
    (n & 3, y0, y1)
}

/// sin(x). fdlibm dispatch over the quadrant from `rem_pio2`.
pub fn sin(x: f64) -> f64 {
    if x.is_nan() {
        return canonical_nan();
    }
    let ix = high_word(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        // |x| <= pi/4
        if ix < 0x3e50_0000 {
            // |x| < 2^-26
            return x;
        }
        return kernel_sin(x, 0.0, 0);
    }
    if ix >= 0x7ff0_0000 {
        return canonical_nan();
    }
    let (n, y0, y1) = rem_pio2(x);
    match n {
        0 => kernel_sin(y0, y1, 1),
        1 => kernel_cos(y0, y1),
        2 => -kernel_sin(y0, y1, 1),
        _ => -kernel_cos(y0, y1),
    }
}

/// cos(x).
pub fn cos(x: f64) -> f64 {
    if x.is_nan() {
        return canonical_nan();
    }
    let ix = high_word(x) & 0x7fff_ffff;
    if ix <= 0x3fe9_21fb {
        if ix < 0x3e46_a09e {
            // |x| < 2^-27
            return 1.0;
        }
        return kernel_cos(x, 0.0);
    }
    if ix >= 0x7ff0_0000 {
        return canonical_nan();
    }
    let (n, y0, y1) = rem_pio2(x);
    match n {
        0 => kernel_cos(y0, y1),
        1 => -kernel_sin(y0, y1, 1),
        2 => -kernel_cos(y0, y1),
        _ => kernel_sin(y0, y1, 1),
    }
}

// ------------------------------ atan / atan2 -------------------------------

/// atan(x). Faithful port of fdlibm `s_atan.c`.
pub fn atan(x: f64) -> f64 {
    if x.is_nan() {
        return canonical_nan();
    }
    const ATANHI: [f64; 4] = [
        4.636_476_090_008_060_935_15e-1,
        7.853_981_633_974_482_789_99e-1,
        9.827_937_232_473_290_540_82e-1,
        1.570_796_326_794_896_558_00,
    ];
    const ATANLO: [f64; 4] = [
        2.269_877_745_296_168_709_24e-17,
        3.061_616_997_868_383_017_93e-17,
        1.390_331_103_123_099_845_16e-17,
        6.123_233_995_736_766_035_87e-17,
    ];
    const AT: [f64; 11] = [
        3.333_333_333_333_293_180_27e-1,
        -1.999_999_999_987_648_324_76e-1,
        1.428_571_427_250_346_637_11e-1,
        -1.111_111_040_546_235_578_80e-1,
        9.090_887_133_436_506_561_96e-2,
        -7.691_876_205_044_829_994_95e-2,
        6.661_073_137_387_531_206_69e-2,
        -5.833_570_133_790_573_486_45e-2,
        4.976_877_994_615_932_360_17e-2,
        -3.653_157_274_421_691_552_70e-2,
        1.628_582_011_536_578_236_23e-2,
    ];

    let hx = high_word(x) as i32;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x4410_0000 {
        // |x| >= 2^66
        if ix > 0x7ff0_0000 || (ix == 0x7ff0_0000 && low_word(x) != 0) {
            return canonical_nan();
        }
        return if hx > 0 {
            ATANHI[3] + ATANLO[3]
        } else {
            -ATANHI[3] - ATANLO[3]
        };
    }

    let id;
    let mut xr = x;
    if ix < 0x3fdc_0000 {
        // |x| < 0.4375
        if ix < 0x3e40_0000 {
            // |x| < 2^-27
            return x;
        }
        id = -1i32;
    } else {
        xr = fabs(x);
        if ix < 0x3ff3_0000 {
            // |x| < 1.1875
            if ix < 0x3fe6_0000 {
                // 7/16 <= |x| < 11/16
                id = 0;
                xr = (2.0 * xr - 1.0) / (2.0 + xr);
            } else {
                // 11/16 <= |x| < 19/16
                id = 1;
                xr = (xr - 1.0) / (xr + 1.0);
            }
        } else if ix < 0x4003_8000 {
            // |x| < 2.4375
            id = 2;
            xr = (xr - 1.5) / (1.0 + 1.5 * xr);
        } else {
            // 2.4375 <= |x| < 2^66
            id = 3;
            xr = -1.0 / xr;
        }
    }

    let z = xr * xr;
    let w = z * z;
    let s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
    let s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));
    if id < 0 {
        return xr - xr * (s1 + s2);
    }
    let idx = id as usize;
    let z2 = ATANHI[idx] - ((xr * (s1 + s2) - ATANLO[idx]) - xr);
    if hx < 0 {
        -z2
    } else {
        z2
    }
}

/// atan2(y, x). Faithful port of fdlibm `e_atan2.c`.
pub fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return canonical_nan();
    }
    const TINY: f64 = 1.0e-300;
    const PI_O_4: f64 = 7.853_981_633_974_482_790_0e-1;
    const PI_O_2: f64 = 1.570_796_326_794_896_558_0;
    const PI: f64 = 3.141_592_653_589_793_116_0;
    const PI_LO: f64 = 1.224_646_799_147_353_177_2e-16;

    let hx = high_word(x) as i32;
    let ix = hx & 0x7fff_ffff;
    let lx = low_word(x);
    let hy = high_word(y) as i32;
    let iy = hy & 0x7fff_ffff;
    let ly = low_word(y);

    // x or y is NaN
    if (ix | (((lx | lx.wrapping_neg()) >> 31) as i32)) > 0x7ff0_0000
        || (iy | (((ly | ly.wrapping_neg()) >> 31) as i32)) > 0x7ff0_0000
    {
        return canonical_nan();
    }
    if hx == 0x3ff0_0000 && lx == 0 {
        return atan(y); // x = 1.0
    }
    let m = ((hy >> 31) & 1) | ((hx >> 30) & 2); // 2*sign(x) + sign(y)

    // y = 0
    if (iy | (ly as i32)) == 0 {
        return match m {
            0 | 1 => y,      // atan(+-0, +anything) = +-0
            2 => PI + TINY,  // atan(+0, -anything) = pi
            _ => -PI - TINY, // atan(-0, -anything) = -pi
        };
    }
    // x = 0
    if (ix | (lx as i32)) == 0 {
        return if hy < 0 {
            -PI_O_2 - TINY
        } else {
            PI_O_2 + TINY
        };
    }
    // x is inf
    if ix == 0x7ff0_0000 {
        if iy == 0x7ff0_0000 {
            return match m {
                0 => PI_O_4 + TINY,
                1 => -PI_O_4 - TINY,
                2 => 3.0 * PI_O_4 + TINY,
                _ => -3.0 * PI_O_4 - TINY,
            };
        }
        return match m {
            0 => 0.0,
            1 => -0.0,
            2 => PI + TINY,
            _ => -PI - TINY,
        };
    }
    // y is inf
    if iy == 0x7ff0_0000 {
        return if hy < 0 {
            -PI_O_2 - TINY
        } else {
            PI_O_2 + TINY
        };
    }

    // compute y/x
    let k = (iy - ix) >> 20;
    let z = if k > 60 {
        PI_O_2 + 0.5 * PI_LO // |y/x| > 2^60
    } else if hx < 0 && k < -60 {
        0.0 // |y|/x < -2^60
    } else {
        atan(fabs(y / x))
    };
    match m {
        0 => z,
        1 => -z,
        2 => PI - (z - PI_LO),
        _ => (z - PI_LO) - PI,
    }
}

// ---------------------------------- tests ----------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Units in the last place between two finite f64s of the same sign region.
    fn ulp_diff(a: f64, b: f64) -> f64 {
        if a == b {
            return 0.0;
        }
        if !a.is_finite() || !b.is_finite() {
            return if a.to_bits() == b.to_bits() {
                0.0
            } else {
                f64::INFINITY
            };
        }
        let ia = a.to_bits() as i64;
        let ib = b.to_bits() as i64;
        // Map to a monotone ordering across the sign boundary.
        let ma = if ia < 0 { i64::MIN - ia } else { ia };
        let mb = if ib < 0 { i64::MIN - ib } else { ib };
        (ma - mb).unsigned_abs() as f64
    }

    fn assert_close_to_glibc(
        name: &str,
        ours: fn(f64) -> f64,
        theirs: fn(f64) -> f64,
        grid: &[f64],
        max_ulp: f64,
    ) {
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f64;
        for &v in grid {
            let a = ours(v);
            let b = theirs(v);
            if b.is_nan() {
                assert!(
                    a.is_nan(),
                    "{name}({v}): ours={a} not NaN where glibc is NaN"
                );
                continue;
            }
            if !b.is_finite() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name}({v}): ours={a} glibc={b} (non-finite mismatch)"
                );
                continue;
            }
            let d = ulp_diff(a, b);
            if d > worst {
                worst = d;
                worst_at = v;
            }
        }
        assert!(
            worst <= max_ulp,
            "{name}: worst {worst} ULP vs glibc at x={worst_at} exceeds {max_ulp}"
        );
        println!(
            "portable::{name} worst {worst} ULP vs glibc (grid {})",
            grid.len()
        );
    }

    fn dense_grid() -> Vec<f64> {
        let mut g = Vec::new();
        // Wide log-spaced + linear coverage, both signs, plus edge values.
        let mut v = -700.0;
        while v <= 700.0 {
            g.push(v);
            v += 0.013;
        }
        for e in -60..60 {
            let m = 2f64.powi(e);
            g.push(m);
            g.push(-m);
            g.push(1.0 + m);
            g.push(1.0 - m);
        }
        g.extend_from_slice(&[
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 709.0, -745.0, 1e-300, 1e300,
        ]);
        g
    }

    #[test]
    fn exp_matches_glibc_within_1_ulp() {
        let grid: Vec<f64> = dense_grid()
            .into_iter()
            .filter(|v| v.abs() <= 709.0)
            .collect();
        assert_close_to_glibc("exp", exp, f64::exp, &grid, 1.0);
    }

    #[test]
    fn ln_matches_glibc_within_1_ulp() {
        let grid: Vec<f64> = dense_grid().into_iter().filter(|v| *v > 0.0).collect();
        assert_close_to_glibc("ln", ln, f64::ln, &grid, 1.0);
    }

    #[test]
    fn log1p_matches_glibc_within_1_ulp() {
        let grid: Vec<f64> = dense_grid().into_iter().filter(|v| *v > -1.0).collect();
        assert_close_to_glibc("log1p", log1p, f64::ln_1p, &grid, 1.0);
    }

    #[test]
    fn expm1_matches_glibc_within_1_ulp() {
        let grid: Vec<f64> = dense_grid()
            .into_iter()
            .filter(|v| v.abs() <= 709.0)
            .collect();
        assert_close_to_glibc("expm1", expm1, f64::exp_m1, &grid, 1.0);
    }

    #[test]
    fn tanh_matches_glibc_within_2_ulp() {
        // The glibc comparison is a coarse correctness sanity bound only; the
        // canonical value is the portable one (a faithful musl port), and musl
        // and glibc legitimately differ by up to ~2 ULP on tanh. Cross-LANGUAGE
        // bit-identity is enforced separately by the golden harness.
        let grid = dense_grid();
        assert_close_to_glibc("tanh", tanh, f64::tanh, &grid, 2.0);
    }

    #[test]
    fn pow_matches_glibc_within_2_ulp() {
        let bases: Vec<f64> = dense_grid()
            .into_iter()
            .filter(|v| *v > 0.0 && v.is_finite())
            .collect();
        let exps = [
            -8.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5,
            3.0, 4.0, 7.0, 8.0, 0.333_333, 2.718,
        ];
        let mut worst = 0.0f64;
        let (mut wb, mut we) = (0.0, 0.0);
        for &b in &bases {
            for &e in &exps {
                let a = pow(b, e);
                let g = b.powf(e);
                if !g.is_finite() {
                    assert_eq!(a.to_bits(), g.to_bits(), "pow({b},{e}): ours={a} glibc={g}");
                    continue;
                }
                let d = ulp_diff(a, g);
                if d > worst {
                    worst = d;
                    wb = b;
                    we = e;
                }
            }
        }
        assert!(
            worst <= 2.0,
            "pow: worst {worst} ULP vs glibc at ({wb},{we})"
        );
        println!("portable::pow worst {worst} ULP vs glibc");
    }

    #[test]
    fn pow_special_values() {
        assert_eq!(pow(2.0, 0.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(pow(0.0, 0.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(pow(f64::NAN, 0.0).to_bits(), 1.0f64.to_bits());
        assert!(pow(f64::NAN, 2.0).is_nan());
        assert_eq!(pow(2.0, 3.0), 8.0);
        assert_eq!(pow(9.0, 0.5), 3.0);
        assert_eq!(pow(2.0, -1.0), 0.5);
        assert_eq!(pow(-2.0, 2.0), 4.0);
        assert_eq!(pow(-2.0, 3.0), -8.0);
        assert!(pow(-2.0, 0.5).is_nan());
        assert_eq!(pow(f64::INFINITY, 2.0), f64::INFINITY);
        assert_eq!(pow(2.0, f64::INFINITY), f64::INFINITY);
        assert_eq!(pow(0.5, f64::INFINITY), 0.0);
    }

    #[test]
    fn sin_cos_match_glibc_in_realistic_range() {
        // Full accuracy over the medium-reduction range (covers RFF/Box-Muller/
        // standardized features). Larger |x| tested separately for finiteness.
        let grid: Vec<f64> = dense_grid()
            .into_iter()
            .filter(|v| v.is_finite() && v.abs() <= 1.0e5)
            .collect();
        assert_close_to_glibc("sin", sin, f64::sin, &grid, 2.0);
        assert_close_to_glibc("cos", cos, f64::cos, &grid, 2.0);
    }

    #[test]
    fn sin_cos_large_args_finite_bounded_deterministic() {
        for &v in &[1.0e6, 1.0e9, 1.0e12, -1.0e12, 1.234e11, 2f64.powi(50)] {
            let s = sin(v);
            let c = cos(v);
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-9, "sin({v})={s}");
            assert!(c.is_finite() && c.abs() <= 1.0 + 1e-9, "cos({v})={c}");
            assert_eq!(sin(v).to_bits(), sin(v).to_bits());
            assert_eq!(cos(v).to_bits(), cos(v).to_bits());
        }
    }

    #[test]
    fn atan_atan2_match_glibc() {
        let grid = dense_grid();
        assert_close_to_glibc("atan", atan, f64::atan, &grid, 1.0);
        // atan2 spot checks incl. all sign quadrants and axis/inf cases.
        let cases = [
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (3.0, 4.0),
            (-3.0, 4.0),
            (1e-8, 1e8),
            (1e8, 1e-8),
            (f64::INFINITY, 1.0),
            (1.0, f64::INFINITY),
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
        ];
        for &(yy, xx) in &cases {
            let a = atan2(yy, xx);
            let g = yy.atan2(xx);
            if g.is_finite() {
                assert!(
                    ulp_diff(a, g) <= 2.0,
                    "atan2({yy},{xx}): ours={a} glibc={g}"
                );
            } else {
                assert_eq!(
                    a.to_bits(),
                    g.to_bits(),
                    "atan2({yy},{xx}) non-finite mismatch"
                );
            }
        }
    }

    #[test]
    fn special_values_route_exactly() {
        const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;
        let signaling_nan = f64::from_bits(0x7ff0_0000_0000_0001);
        // exp
        assert_eq!(exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp(f64::NEG_INFINITY), 0.0);
        assert!(exp(f64::NAN).is_nan());
        assert_eq!(exp(0.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(exp(800.0), f64::INFINITY);
        assert_eq!(exp(-800.0), 0.0);
        // ln
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert_eq!(ln(-0.0), f64::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert_eq!(ln(f64::INFINITY), f64::INFINITY);
        assert!(ln(f64::NAN).is_nan());
        assert_eq!(ln(1.0).to_bits(), 0.0f64.to_bits());
        // log1p / expm1
        assert_eq!(log1p(-1.0), f64::NEG_INFINITY);
        assert!(log1p(-2.0).is_nan());
        assert_eq!(log1p(0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(expm1(f64::NEG_INFINITY), -1.0);
        assert_eq!(expm1(0.0).to_bits(), 0.0f64.to_bits());
        // Signed-zero quadrants must route before any signed high-word
        // arithmetic that can overflow in debug builds.
        assert_eq!(
            atan2(-0.0, -0.0).to_bits(),
            (-std::f64::consts::PI).to_bits()
        );
        for value in [
            exp(signaling_nan),
            ln(signaling_nan),
            log1p(signaling_nan),
            expm1(signaling_nan),
            tanh(signaling_nan),
            sin(signaling_nan),
            cos(signaling_nan),
            atan(signaling_nan),
            pow(signaling_nan, 2.0),
            atan2(signaling_nan, 1.0),
        ] {
            assert_eq!(value.to_bits(), CANONICAL_NAN_BITS);
        }
    }

    #[test]
    fn deterministic_and_pure() {
        // Same input -> identical bits every call (no global state, no rounding mode drift).
        for &v in &[0.3, -2.7, 123.4, -0.001, 1e-20, 700.0] {
            assert_eq!(exp(v).to_bits(), exp(v).to_bits());
            assert_eq!(expm1(v).to_bits(), expm1(v).to_bits());
        }
        for &v in &[0.3, 2.7, 123.4, 0.001, 1e-20, 1e200] {
            assert_eq!(ln(v).to_bits(), ln(v).to_bits());
            assert_eq!(log1p(v).to_bits(), log1p(v).to_bits());
        }
    }
}

/// Precision context for the complete frozen S1 worker fit/predict operation.
/// Default interpreter and emitted-artifact arithmetic remain unchanged.
pub mod frozen_s1 {
    // Scoped precision for the frozen S1 worker's bounded transcendental inputs.
    // The normal interpreter keeps its existing platform arithmetic.
    use std::cell::Cell;

    thread_local! { static S1: Cell<bool> = const { Cell::new(false) }; }

    /// Apply the frozen worker's numerical context without changing other callers.
    pub fn with_frozen_s1_math<T>(run: impl FnOnce() -> T) -> T {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                S1.with(|state| state.set(self.0));
            }
        }
        let old = S1.with(|state| state.replace(true));
        let _restore = Restore(old);
        run()
    }

    fn scoped() -> bool {
        !cfg!(target_os = "linux") && S1.with(Cell::get)
    }
    pub fn ln(x: f64) -> f64 {
        if scoped() {
            precise_ln(x)
        } else {
            x.ln()
        }
    }
    pub fn exp(x: f64) -> f64 {
        if scoped() {
            precise_exp(x)
        } else {
            x.exp()
        }
    }
    pub fn sin(x: f64) -> f64 {
        if scoped() {
            precise_sincos(x).0
        } else {
            x.sin()
        }
    }
    pub fn cos(x: f64) -> f64 {
        if scoped() {
            precise_sincos(x).1
        } else {
            x.cos()
        }
    }

    pub(crate) fn precise_ln(x: f64) -> f64 {
        linux_reference::ln(x)
    }
    pub(crate) fn precise_exp(x: f64) -> f64 {
        linux_reference::exp(x)
    }
    pub(crate) fn precise_sincos(x: f64) -> (f64, f64) {
        (linux_reference::sin(x), linux_reference::cos(x))
    }

    // This submodule is LGPL-2.1-or-later; surrounding original code remains Apache-2.0.
    mod linux_reference {
        // SPDX-License-Identifier: LGPL-2.1-or-later
        // Derived from glibc 2.36 e_log.c/e_exp.c/s_sin.c and their tables.
        // Copyright (C) 2001-2022 Free Software Foundation, Inc.
        // IBM Accurate Mathematical Library (sin/cos); original log/exp by Arm contributors.
        // Rust translation uses explicit fused operations, independent of compiler contraction.
        // See licenses/frozen-s1-upstream.json for pinned URLs and hashes. No warranty.
        #![allow(dead_code)]
        const B: [f64; 11] = [
            f64::from_bits(0xbfe0000000000000),
            f64::from_bits(0x3fd5555555555577),
            f64::from_bits(0xbfcffffffffffdcb),
            f64::from_bits(0x3fc999999995dd0c),
            f64::from_bits(0xbfc55555556745a7),
            f64::from_bits(0x3fc24924a344de30),
            f64::from_bits(0xbfbfffffa4423d65),
            f64::from_bits(0x3fbc7184282ad6ca),
            f64::from_bits(0xbfb999eb43b068ff),
            f64::from_bits(0x3fb78182f7afd085),
            f64::from_bits(0xbfb5521375d145cd),
        ];
        const A: [f64; 5] = [
            f64::from_bits(0xbfe0000000000001),
            f64::from_bits(0x3fd555555551305b),
            f64::from_bits(0xbfcfffffffeb4590),
            f64::from_bits(0x3fc999b324f10111),
            f64::from_bits(0xbfc55575e506c89f),
        ];
        const INVC: [f64; 128] = [
            f64::from_bits(0x3ff734f0c3e0de9f),
            f64::from_bits(0x3ff713786a2ce91f),
            f64::from_bits(0x3ff6f26008fab5a0),
            f64::from_bits(0x3ff6d1a61f138c7d),
            f64::from_bits(0x3ff6b1490bc5b4d1),
            f64::from_bits(0x3ff69147332f0cba),
            f64::from_bits(0x3ff6719f18224223),
            f64::from_bits(0x3ff6524f99a51ed9),
            f64::from_bits(0x3ff63356aa8f24c4),
            f64::from_bits(0x3ff614b36b9ddc14),
            f64::from_bits(0x3ff5f66452c65c4c),
            f64::from_bits(0x3ff5d867b5912c4f),
            f64::from_bits(0x3ff5babccb5b90de),
            f64::from_bits(0x3ff59d61f2d91a78),
            f64::from_bits(0x3ff5805612465687),
            f64::from_bits(0x3ff56397cee76bd3),
            f64::from_bits(0x3ff54725e2a77f93),
            f64::from_bits(0x3ff52aff42064583),
            f64::from_bits(0x3ff50f22dbb2bddf),
            f64::from_bits(0x3ff4f38f4734ded7),
            f64::from_bits(0x3ff4d843cfde2840),
            f64::from_bits(0x3ff4bd3ec078a3c8),
            f64::from_bits(0x3ff4a27fc3e0258a),
            f64::from_bits(0x3ff4880524d48434),
            f64::from_bits(0x3ff46dce1b192d0b),
            f64::from_bits(0x3ff453d9d3391854),
            f64::from_bits(0x3ff43a2744b4845a),
            f64::from_bits(0x3ff420b54115f8fb),
            f64::from_bits(0x3ff40782da3ef4b1),
            f64::from_bits(0x3ff3ee8f5d57fe8f),
            f64::from_bits(0x3ff3d5d9a00b4ce9),
            f64::from_bits(0x3ff3bd60c010c12b),
            f64::from_bits(0x3ff3a5242b75dab8),
            f64::from_bits(0x3ff38d22cd9fd002),
            f64::from_bits(0x3ff3755bc5847a1c),
            f64::from_bits(0x3ff35dce49ad36e2),
            f64::from_bits(0x3ff34679984dd440),
            f64::from_bits(0x3ff32f5cceffcb24),
            f64::from_bits(0x3ff3187775a10d49),
            f64::from_bits(0x3ff301c8373e3990),
            f64::from_bits(0x3ff2eb4ebb95f841),
            f64::from_bits(0x3ff2d50a0219a9d1),
            f64::from_bits(0x3ff2bef9a8b7fd2a),
            f64::from_bits(0x3ff2a91c7a0c1bab),
            f64::from_bits(0x3ff293726014b530),
            f64::from_bits(0x3ff27dfa5757a1f5),
            f64::from_bits(0x3ff268b39b1d3bbf),
            f64::from_bits(0x3ff2539d838ff5bd),
            f64::from_bits(0x3ff23eb7aac9083b),
            f64::from_bits(0x3ff22a012ba940b6),
            f64::from_bits(0x3ff2157996cc4132),
            f64::from_bits(0x3ff201201dd2fc9b),
            f64::from_bits(0x3ff1ecf4494d480b),
            f64::from_bits(0x3ff1d8f5528f6569),
            f64::from_bits(0x3ff1c52311577e7c),
            f64::from_bits(0x3ff1b17c74cb26e9),
            f64::from_bits(0x3ff19e010c2c1ab6),
            f64::from_bits(0x3ff18ab07bb670bd),
            f64::from_bits(0x3ff1778a25efbcb6),
            f64::from_bits(0x3ff1648d354c31da),
            f64::from_bits(0x3ff151b990275fdd),
            f64::from_bits(0x3ff13f0ea432d24c),
            f64::from_bits(0x3ff12c8b7210f9da),
            f64::from_bits(0x3ff11a3028ecb531),
            f64::from_bits(0x3ff107fbda8434af),
            f64::from_bits(0x3ff0f5ee0f4e6bb3),
            f64::from_bits(0x3ff0e4065d2a9fce),
            f64::from_bits(0x3ff0d244632ca521),
            f64::from_bits(0x3ff0c0a77ce2981a),
            f64::from_bits(0x3ff0af2f83c636d1),
            f64::from_bits(0x3ff09ddb98a01339),
            f64::from_bits(0x3ff08cabaf52e7df),
            f64::from_bits(0x3ff07b9f2f4e28fb),
            f64::from_bits(0x3ff06ab58c358f19),
            f64::from_bits(0x3ff059eea5ecf92c),
            f64::from_bits(0x3ff04949cdd12c90),
            f64::from_bits(0x3ff038c6c6f0ada9),
            f64::from_bits(0x3ff02865137932a9),
            f64::from_bits(0x3ff0182427ea7348),
            f64::from_bits(0x3ff008040614b195),
            f64::from_bits(0x3fefe01ff726fa1a),
            f64::from_bits(0x3fefa11cc261ea74),
            f64::from_bits(0x3fef6310b081992e),
            f64::from_bits(0x3fef25f63ceeadcd),
            f64::from_bits(0x3feee9c8039113e7),
            f64::from_bits(0x3feeae8078cbb1ab),
            f64::from_bits(0x3fee741aa29d0c9b),
            f64::from_bits(0x3fee3a91830a99b5),
            f64::from_bits(0x3fee01e009609a56),
            f64::from_bits(0x3fedca01e577bb98),
            f64::from_bits(0x3fed92f20b7c9103),
            f64::from_bits(0x3fed5cac66fb5cce),
            f64::from_bits(0x3fed272caa5ede9d),
            f64::from_bits(0x3fecf26e3e6b2ccd),
            f64::from_bits(0x3fecbe6da2a77902),
            f64::from_bits(0x3fec8b266d37086d),
            f64::from_bits(0x3fec5894bd5d5804),
            f64::from_bits(0x3fec26b533bb9f8c),
            f64::from_bits(0x3febf583eeece73f),
            f64::from_bits(0x3febc4fd75db96c1),
            f64::from_bits(0x3feb951e0c864a28),
            f64::from_bits(0x3feb65e2c5ef3e2c),
            f64::from_bits(0x3feb374867c9888b),
            f64::from_bits(0x3feb094b211d304a),
            f64::from_bits(0x3feadbe885f2ef7e),
            f64::from_bits(0x3feaaf1d31603da2),
            f64::from_bits(0x3fea82e63fd358a7),
            f64::from_bits(0x3fea5740ef09738b),
            f64::from_bits(0x3fea2c2a90ab4b27),
            f64::from_bits(0x3fea01a01393f2d1),
            f64::from_bits(0x3fe9d79f24db3c1b),
            f64::from_bits(0x3fe9ae2505c7b190),
            f64::from_bits(0x3fe9852ef297ce2f),
            f64::from_bits(0x3fe95cbaeea44b75),
            f64::from_bits(0x3fe934c69de74838),
            f64::from_bits(0x3fe90d4f2f6752e6),
            f64::from_bits(0x3fe8e6528effd79d),
            f64::from_bits(0x3fe8bfce9fcc007c),
            f64::from_bits(0x3fe899c0dabec30e),
            f64::from_bits(0x3fe87427aa2317fb),
            f64::from_bits(0x3fe84f00acb39a08),
            f64::from_bits(0x3fe82a49e8653e55),
            f64::from_bits(0x3fe8060195f40260),
            f64::from_bits(0x3fe7e22563e0a329),
            f64::from_bits(0x3fe7beb377dcb5ad),
            f64::from_bits(0x3fe79baa679725c2),
            f64::from_bits(0x3fe77907f2170657),
            f64::from_bits(0x3fe756cadbd6130c),
        ];
        const LOGC: [f64; 128] = [
            f64::from_bits(0xbfd7cc7f79e69000),
            f64::from_bits(0xbfd76feec20d0000),
            f64::from_bits(0xbfd713e31351e000),
            f64::from_bits(0xbfd6b85b38287800),
            f64::from_bits(0xbfd65d5590807800),
            f64::from_bits(0xbfd602d076180000),
            f64::from_bits(0xbfd5a8ca86909000),
            f64::from_bits(0xbfd54f4356035000),
            f64::from_bits(0xbfd4f637c36b4000),
            f64::from_bits(0xbfd49da7fda85000),
            f64::from_bits(0xbfd445923989a800),
            f64::from_bits(0xbfd3edf439b0b800),
            f64::from_bits(0xbfd396ce448f7000),
            f64::from_bits(0xbfd3401e17bda000),
            f64::from_bits(0xbfd2e9e2ef468000),
            f64::from_bits(0xbfd2941b3830e000),
            f64::from_bits(0xbfd23ec58cda8800),
            f64::from_bits(0xbfd1e9e129279000),
            f64::from_bits(0xbfd1956d2b48f800),
            f64::from_bits(0xbfd141679ab9f800),
            f64::from_bits(0xbfd0edd094ef9800),
            f64::from_bits(0xbfd09aa518db1000),
            f64::from_bits(0xbfd047e65263b800),
            f64::from_bits(0xbfcfeb224586f000),
            f64::from_bits(0xbfcf474a7517b000),
            f64::from_bits(0xbfcea4443d103000),
            f64::from_bits(0xbfce020d44e9b000),
            f64::from_bits(0xbfcd60a22977f000),
            f64::from_bits(0xbfccc00104959000),
            f64::from_bits(0xbfcc202956891000),
            f64::from_bits(0xbfcb81178d811000),
            f64::from_bits(0xbfcae2c9ccd3d000),
            f64::from_bits(0xbfca45402e129000),
            f64::from_bits(0xbfc9a877681df000),
            f64::from_bits(0xbfc90c6d69483000),
            f64::from_bits(0xbfc87120a645c000),
            f64::from_bits(0xbfc7d68fb4143000),
            f64::from_bits(0xbfc73cb83c627000),
            f64::from_bits(0xbfc6a39a9b376000),
            f64::from_bits(0xbfc60b3154b7a000),
            f64::from_bits(0xbfc5737d76243000),
            f64::from_bits(0xbfc4dc7b8fc23000),
            f64::from_bits(0xbfc4462c51d20000),
            f64::from_bits(0xbfc3b08abc830000),
            f64::from_bits(0xbfc31b996b490000),
            f64::from_bits(0xbfc2875490a44000),
            f64::from_bits(0xbfc1f3b9f879a000),
            f64::from_bits(0xbfc160c8252ca000),
            f64::from_bits(0xbfc0ce7f57f72000),
            f64::from_bits(0xbfc03cdc49fea000),
            f64::from_bits(0xbfbf57bdbc4b8000),
            f64::from_bits(0xbfbe370896404000),
            f64::from_bits(0xbfbd17983ef94000),
            f64::from_bits(0xbfbbf9674ed8a000),
            f64::from_bits(0xbfbadc79202f6000),
            f64::from_bits(0xbfb9c0c3e7288000),
            f64::from_bits(0xbfb8a646b372c000),
            f64::from_bits(0xbfb78d01b3ac0000),
            f64::from_bits(0xbfb674f145380000),
            f64::from_bits(0xbfb55e0e6d878000),
            f64::from_bits(0xbfb4485cdea1e000),
            f64::from_bits(0xbfb333d94d6aa000),
            f64::from_bits(0xbfb22079f8c56000),
            f64::from_bits(0xbfb10e4698622000),
            f64::from_bits(0xbfaffa6c6ad20000),
            f64::from_bits(0xbfadda8d4a774000),
            f64::from_bits(0xbfabbcece4850000),
            f64::from_bits(0xbfa9a1894012c000),
            f64::from_bits(0xbfa788583302c000),
            f64::from_bits(0xbfa5715e67d68000),
            f64::from_bits(0xbfa35c8a49658000),
            f64::from_bits(0xbfa149e364154000),
            f64::from_bits(0xbf9e72c082eb8000),
            f64::from_bits(0xbf9a55f152528000),
            f64::from_bits(0xbf963d62cf818000),
            f64::from_bits(0xbf9228fb8caa0000),
            f64::from_bits(0xbf8c317b20f90000),
            f64::from_bits(0xbf8419355daa0000),
            f64::from_bits(0xbf781203c2ec0000),
            f64::from_bits(0xbf60040979240000),
            f64::from_bits(0x3f6feff384900000),
            f64::from_bits(0x3f87dc41353d0000),
            f64::from_bits(0x3f93cea3c4c28000),
            f64::from_bits(0x3f9b9fc114890000),
            f64::from_bits(0x3fa1b0d8ce110000),
            f64::from_bits(0x3fa58a5bd001c000),
            f64::from_bits(0x3fa95c8340d88000),
            f64::from_bits(0x3fad276aef578000),
            f64::from_bits(0x3fb07598e598c000),
            f64::from_bits(0x3fb253f5e30d2000),
            f64::from_bits(0x3fb42edd8b380000),
            f64::from_bits(0x3fb606598757c000),
            f64::from_bits(0x3fb7da76356a0000),
            f64::from_bits(0x3fb9ab434e1c6000),
            f64::from_bits(0x3fbb78c7bb0d6000),
            f64::from_bits(0x3fbd431332e72000),
            f64::from_bits(0x3fbf0a3171de6000),
            f64::from_bits(0x3fc067152b914000),
            f64::from_bits(0x3fc147858292b000),
            f64::from_bits(0x3fc2266ecdca3000),
            f64::from_bits(0x3fc303d7a6c55000),
            f64::from_bits(0x3fc3dfc33c331000),
            f64::from_bits(0x3fc4ba366b7a8000),
            f64::from_bits(0x3fc5933928d1f000),
            f64::from_bits(0x3fc66acd2418f000),
            f64::from_bits(0x3fc740f8ec669000),
            f64::from_bits(0x3fc815c0f51af000),
            f64::from_bits(0x3fc8e92954f68000),
            f64::from_bits(0x3fc9bb3602f84000),
            f64::from_bits(0x3fca8bed1c2c0000),
            f64::from_bits(0x3fcb5b515c01d000),
            f64::from_bits(0x3fcc2967ccbcc000),
            f64::from_bits(0x3fccf635d5486000),
            f64::from_bits(0x3fcdc1bd3446c000),
            f64::from_bits(0x3fce8c01b8cfe000),
            f64::from_bits(0x3fcf5509c0179000),
            f64::from_bits(0x3fd00e6c121fb800),
            f64::from_bits(0x3fd071b80e93d000),
            f64::from_bits(0x3fd0d46b9e867000),
            f64::from_bits(0x3fd13687334bd000),
            f64::from_bits(0x3fd1980d67234800),
            f64::from_bits(0x3fd1f8ffe0cc8000),
            f64::from_bits(0x3fd2595fd7636800),
            f64::from_bits(0x3fd2b9300914a800),
            f64::from_bits(0x3fd3187210436000),
            f64::from_bits(0x3fd377266dec1800),
            f64::from_bits(0x3fd3d54ffbaf3000),
            f64::from_bits(0x3fd432eee32fe000),
        ];
        const ET: [u64; 256] = [
            0x0,
            0x3ff0000000000000,
            0x3c9b3b4f1a88bf6e,
            0x3feff63da9fb3335,
            0xbc7160139cd8dc5d,
            0x3fefec9a3e778061,
            0xbc905e7a108766d1,
            0x3fefe315e86e7f85,
            0x3c8cd2523567f613,
            0x3fefd9b0d3158574,
            0xbc8bce8023f98efa,
            0x3fefd06b29ddf6de,
            0x3c60f74e61e6c861,
            0x3fefc74518759bc8,
            0x3c90a3e45b33d399,
            0x3fefbe3ecac6f383,
            0x3c979aa65d837b6d,
            0x3fefb5586cf9890f,
            0x3c8eb51a92fdeffc,
            0x3fefac922b7247f7,
            0x3c3ebe3d702f9cd1,
            0x3fefa3ec32d3d1a2,
            0xbc6a033489906e0b,
            0x3fef9b66affed31b,
            0xbc9556522a2fbd0e,
            0x3fef9301d0125b51,
            0xbc5080ef8c4eea55,
            0x3fef8abdc06c31cc,
            0xbc91c923b9d5f416,
            0x3fef829aaea92de0,
            0x3c80d3e3e95c55af,
            0x3fef7a98c8a58e51,
            0xbc801b15eaa59348,
            0x3fef72b83c7d517b,
            0xbc8f1ff055de323d,
            0x3fef6af9388c8dea,
            0x3c8b898c3f1353bf,
            0x3fef635beb6fcb75,
            0xbc96d99c7611eb26,
            0x3fef5be084045cd4,
            0x3c9aecf73e3a2f60,
            0x3fef54873168b9aa,
            0xbc8fe782cb86389d,
            0x3fef4d5022fcd91d,
            0x3c8a6f4144a6c38d,
            0x3fef463b88628cd6,
            0x3c807a05b0e4047d,
            0x3fef3f49917ddc96,
            0x3c968efde3a8a894,
            0x3fef387a6e756238,
            0x3c875e18f274487d,
            0x3fef31ce4fb2a63f,
            0x3c80472b981fe7f2,
            0x3fef2b4565e27cdd,
            0xbc96b87b3f71085e,
            0x3fef24dfe1f56381,
            0x3c82f7e16d09ab31,
            0x3fef1e9df51fdee1,
            0xbc3d219b1a6fbffa,
            0x3fef187fd0dad990,
            0x3c8b3782720c0ab4,
            0x3fef1285a6e4030b,
            0x3c6e149289cecb8f,
            0x3fef0cafa93e2f56,
            0x3c834d754db0abb6,
            0x3fef06fe0a31b715,
            0x3c864201e2ac744c,
            0x3fef0170fc4cd831,
            0x3c8fdd395dd3f84a,
            0x3feefc08b26416ff,
            0xbc86a3803b8e5b04,
            0x3feef6c55f929ff1,
            0xbc924aedcc4b5068,
            0x3feef1a7373aa9cb,
            0xbc9907f81b512d8e,
            0x3feeecae6d05d866,
            0xbc71d1e83e9436d2,
            0x3feee7db34e59ff7,
            0xbc991919b3ce1b15,
            0x3feee32dc313a8e5,
            0x3c859f48a72a4c6d,
            0x3feedea64c123422,
            0xbc9312607a28698a,
            0x3feeda4504ac801c,
            0xbc58a78f4817895b,
            0x3feed60a21f72e2a,
            0xbc7c2c9b67499a1b,
            0x3feed1f5d950a897,
            0x3c4363ed60c2ac11,
            0x3feece086061892d,
            0x3c9666093b0664ef,
            0x3feeca41ed1d0057,
            0x3c6ecce1daa10379,
            0x3feec6a2b5c13cd0,
            0x3c93ff8e3f0f1230,
            0x3feec32af0d7d3de,
            0x3c7690cebb7aafb0,
            0x3feebfdad5362a27,
            0x3c931dbdeb54e077,
            0x3feebcb299fddd0d,
            0xbc8f94340071a38e,
            0x3feeb9b2769d2ca7,
            0xbc87deccdc93a349,
            0x3feeb6daa2cf6642,
            0xbc78dec6bd0f385f,
            0x3feeb42b569d4f82,
            0xbc861246ec7b5cf6,
            0x3feeb1a4ca5d920f,
            0x3c93350518fdd78e,
            0x3feeaf4736b527da,
            0x3c7b98b72f8a9b05,
            0x3feead12d497c7fd,
            0x3c9063e1e21c5409,
            0x3feeab07dd485429,
            0x3c34c7855019c6ea,
            0x3feea9268a5946b7,
            0x3c9432e62b64c035,
            0x3feea76f15ad2148,
            0xbc8ce44a6199769f,
            0x3feea5e1b976dc09,
            0xbc8c33c53bef4da8,
            0x3feea47eb03a5585,
            0xbc845378892be9ae,
            0x3feea34634ccc320,
            0xbc93cedd78565858,
            0x3feea23882552225,
            0x3c5710aa807e1964,
            0x3feea155d44ca973,
            0xbc93b3efbf5e2228,
            0x3feea09e667f3bcd,
            0xbc6a12ad8734b982,
            0x3feea012750bdabf,
            0xbc6367efb86da9ee,
            0x3fee9fb23c651a2f,
            0xbc80dc3d54e08851,
            0x3fee9f7df9519484,
            0xbc781f647e5a3ecf,
            0x3fee9f75e8ec5f74,
            0xbc86ee4ac08b7db0,
            0x3fee9f9a48a58174,
            0xbc8619321e55e68a,
            0x3fee9feb564267c9,
            0x3c909ccb5e09d4d3,
            0x3feea0694fde5d3f,
            0xbc7b32dcb94da51d,
            0x3feea11473eb0187,
            0x3c94ecfd5467c06b,
            0x3feea1ed0130c132,
            0x3c65ebe1abd66c55,
            0x3feea2f336cf4e62,
            0xbc88a1c52fb3cf42,
            0x3feea427543e1a12,
            0xbc9369b6f13b3734,
            0x3feea589994cce13,
            0xbc805e843a19ff1e,
            0x3feea71a4623c7ad,
            0xbc94d450d872576e,
            0x3feea8d99b4492ed,
            0x3c90ad675b0e8a00,
            0x3feeaac7d98a6699,
            0x3c8db72fc1f0eab4,
            0x3feeace5422aa0db,
            0xbc65b6609cc5e7ff,
            0x3feeaf3216b5448c,
            0x3c7bf68359f35f44,
            0x3feeb1ae99157736,
            0xbc93091fa71e3d83,
            0x3feeb45b0b91ffc6,
            0xbc5da9b88b6c1e29,
            0x3feeb737b0cdc5e5,
            0xbc6c23f97c90b959,
            0x3feeba44cbc8520f,
            0xbc92434322f4f9aa,
            0x3feebd829fde4e50,
            0xbc85ca6cd7668e4b,
            0x3feec0f170ca07ba,
            0x3c71affc2b91ce27,
            0x3feec49182a3f090,
            0x3c6dd235e10a73bb,
            0x3feec86319e32323,
            0xbc87c50422622263,
            0x3feecc667b5de565,
            0x3c8b1c86e3e231d5,
            0x3feed09bec4a2d33,
            0xbc91bbd1d3bcbb15,
            0x3feed503b23e255d,
            0x3c90cc319cee31d2,
            0x3feed99e1330b358,
            0x3c8469846e735ab3,
            0x3feede6b5579fdbf,
            0xbc82dfcd978e9db4,
            0x3feee36bbfd3f37a,
            0x3c8c1a7792cb3387,
            0x3feee89f995ad3ad,
            0xbc907b8f4ad1d9fa,
            0x3feeee07298db666,
            0xbc55c3d956dcaeba,
            0x3feef3a2b84f15fb,
            0xbc90a40e3da6f640,
            0x3feef9728de5593a,
            0xbc68d6f438ad9334,
            0x3feeff76f2fb5e47,
            0xbc91eee26b588a35,
            0x3fef05b030a1064a,
            0x3c74ffd70a5fddcd,
            0x3fef0c1e904bc1d2,
            0xbc91bdfbfa9298ac,
            0x3fef12c25bd71e09,
            0x3c736eae30af0cb3,
            0x3fef199bdd85529c,
            0x3c8ee3325c9ffd94,
            0x3fef20ab5fffd07a,
            0x3c84e08fd10959ac,
            0x3fef27f12e57d14b,
            0x3c63cdaf384e1a67,
            0x3fef2f6d9406e7b5,
            0x3c676b2c6c921968,
            0x3fef3720dcef9069,
            0xbc808a1883ccb5d2,
            0x3fef3f0b555dc3fa,
            0xbc8fad5d3ffffa6f,
            0x3fef472d4a07897c,
            0xbc900dae3875a949,
            0x3fef4f87080d89f2,
            0x3c74a385a63d07a7,
            0x3fef5818dcfba487,
            0xbc82919e2040220f,
            0x3fef60e316c98398,
            0x3c8e5a50d5c192ac,
            0x3fef69e603db3285,
            0x3c843a59ac016b4b,
            0x3fef7321f301b460,
            0xbc82d52107b43e1f,
            0x3fef7c97337b9b5f,
            0xbc892ab93b470dc9,
            0x3fef864614f5a129,
            0x3c74b604603a88d3,
            0x3fef902ee78b3ff6,
            0x3c83c5ec519d7271,
            0x3fef9a51fbc74c83,
            0xbc8ff7128fd391f0,
            0x3fefa4afa2a490da,
            0xbc8dae98e223747d,
            0x3fefaf482d8e67f1,
            0x3c8ec3bc41aa2008,
            0x3fefba1bee615a27,
            0x3c842b94c3a9eb32,
            0x3fefc52b376bba97,
            0x3c8a64a931d185ee,
            0x3fefd0765b6e4540,
            0xbc8e37bae43be3ed,
            0x3fefdbfdad9cbe14,
            0x3c77893b4d91cd9d,
            0x3fefe7c1819e90d8,
            0x3c5305c14160cc89,
            0x3feff3c22b8f71f1,
        ];
        const C: [f64; 4] = [
            f64::from_bits(0x3fdffffffffffdbd),
            f64::from_bits(0x3fc555555555543c),
            f64::from_bits(0x3fa55555cf172b91),
            f64::from_bits(0x3f81111167a4d017),
        ];
        const ST: [f64; 440] = [
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3f7fffeaaaaeeeef),
            f64::from_bits(0xbc1e45e2ec67b77c),
            f64::from_bits(0x3fefffc000155552),
            f64::from_bits(0x3c8f4a01a0196dae),
            f64::from_bits(0x3f8fffaaaaeeeed5),
            f64::from_bits(0xbc02ab639a9f0777),
            f64::from_bits(0x3fefff000155549f),
            f64::from_bits(0x3c828a28a03a5ef3),
            f64::from_bits(0x3f97ff7001033255),
            f64::from_bits(0x3bfefe2b51527336),
            f64::from_bits(0x3feffdc006bff7e6),
            f64::from_bits(0x3c8ae6dae86977bd),
            f64::from_bits(0x3f9ffeaaaeeee86f),
            f64::from_bits(0xbc3cd406fb224ae2),
            f64::from_bits(0x3feffc00155527d3),
            f64::from_bits(0xbc83b54492d89b5b),
            f64::from_bits(0x3fa3feb2b12d45d5),
            f64::from_bits(0x3c34ec54203d1c11),
            f64::from_bits(0x3feff9c03414a7ba),
            f64::from_bits(0x3c6991f4be6c59bf),
            f64::from_bits(0x3fa7fdc01032fba9),
            f64::from_bits(0xbc4599bdf46e997a),
            f64::from_bits(0x3feff7006bfdf99f),
            f64::from_bits(0xbc78b3b560648d5f),
            f64::from_bits(0x3fabfc6d78586dac),
            f64::from_bits(0x3c18e4fd03dbf236),
            f64::from_bits(0x3feff3c0c8103a31),
            f64::from_bits(0x3c74856dbddc0e66),
            f64::from_bits(0x3faffaaaeeed4edb),
            f64::from_bits(0xbc42d16d32684b69),
            f64::from_bits(0x3feff0015549f4d3),
            f64::from_bits(0x3c8328387b99426f),
            f64::from_bits(0x3fb1fc343d808bef),
            f64::from_bits(0xbc5f3d32e6f3be4f),
            f64::from_bits(0x3fefebc222a8ef9f),
            f64::from_bits(0x3c57934934f54c77),
            f64::from_bits(0x3fb3facb12d1755b),
            f64::from_bits(0xbc5921915299468c),
            f64::from_bits(0x3fefe7034129ef6f),
            f64::from_bits(0xbc6cbf4337c96f97),
            f64::from_bits(0x3fb5f911fd10b737),
            f64::from_bits(0xbc50184f02be9102),
            f64::from_bits(0x3fefe1c4c3c873eb),
            f64::from_bits(0xbc35a9c9057c4a02),
            f64::from_bits(0x3fb7f701032550e4),
            f64::from_bits(0x3c3afc2d1800501a),
            f64::from_bits(0x3fefdc06bf7e6b9b),
            f64::from_bits(0x3c831902b535f8db),
            f64::from_bits(0x3fb9f4902d55d1f9),
            f64::from_bits(0x3c52696d7eac1dc1),
            f64::from_bits(0x3fefd5c94b43e000),
            f64::from_bits(0xbc62e768cb4f92f9),
            f64::from_bits(0x3fbbf1b78568391d),
            f64::from_bits(0x3c5e91841dea4cc8),
            f64::from_bits(0x3fefcf0c800e99b1),
            f64::from_bits(0x3c6ea3d786d186ac),
            f64::from_bits(0x3fbdee6f16c1cce6),
            f64::from_bits(0xbc450f8e2fb71673),
            f64::from_bits(0x3fefc7d078d1bc88),
            f64::from_bits(0x3c8075d2447db685),
            f64::from_bits(0x3fbfeaaeee86ee36),
            f64::from_bits(0xbc4afcb2bcc6f03b),
            f64::from_bits(0x3fefc015527d5bd3),
            f64::from_bits(0x3c8b68f35094efb8),
            f64::from_bits(0x3fc0f3378ddd71d1),
            f64::from_bits(0x3c6d8468724f0f9e),
            f64::from_bits(0x3fefb7db2bfe0695),
            f64::from_bits(0x3c821dadf4f65ab1),
            f64::from_bits(0x3fc1f0d3d7afceaf),
            f64::from_bits(0xbc66ef95099769a5),
            f64::from_bits(0x3fefaf22263c4bd3),
            f64::from_bits(0xbc552ace133a2769),
            f64::from_bits(0x3fc2ee285e4ab88f),
            f64::from_bits(0xbc6e4d0f05dee058),
            f64::from_bits(0x3fefa5ea641c36f2),
            f64::from_bits(0x3c404da6ed17cc7c),
            f64::from_bits(0x3fc3eb312c5d66cb),
            f64::from_bits(0x3c647d666b66cb91),
            f64::from_bits(0x3fef9c340a7cc428),
            f64::from_bits(0x3c8c5b6b063b7462),
            f64::from_bits(0x3fc4e7ea4dc5f27b),
            f64::from_bits(0x3c5949db2ac072fc),
            f64::from_bits(0x3fef91ff40374d01),
            f64::from_bits(0xbc67d03f4d3a9e4c),
            f64::from_bits(0x3fc5e44fcfa126f3),
            f64::from_bits(0xbc66f443063f89b6),
            f64::from_bits(0x3fef874c2e1eecf6),
            f64::from_bits(0xbc8c6514e1332b16),
            f64::from_bits(0x3fc6e05dc05a4d4c),
            f64::from_bits(0xbbd32c5c8b81c940),
            f64::from_bits(0x3fef7c1afeffde24),
            f64::from_bits(0xbc78f55bc47540b1),
            f64::from_bits(0x3fc7dc102fbaf2b5),
            f64::from_bits(0x3c45ab50e23c97c3),
            f64::from_bits(0x3fef706bdf9ece1c),
            f64::from_bits(0xbc8698c80c36dcb4),
            f64::from_bits(0x3fc8d7632efaa944),
            f64::from_bits(0xbc620fa262cbb953),
            f64::from_bits(0x3fef643efeb82acd),
            f64::from_bits(0x3c76b00ac1fe28ac),
            f64::from_bits(0x3fc9d252d0cec312),
            f64::from_bits(0x3c59c43d80b1137d),
            f64::from_bits(0x3fef57948cff6797),
            f64::from_bits(0x3c6e3a0d3e03b1d5),
            f64::from_bits(0x3fcaccdb297a0765),
            f64::from_bits(0xbc59883b57d6cdeb),
            f64::from_bits(0x3fef4a6cbd1e3a79),
            f64::from_bits(0x3c813df0edaebb57),
            f64::from_bits(0x3fcbc6f84edc6199),
            f64::from_bits(0x3c69c1a56a7b0cab),
            f64::from_bits(0x3fef3cc7c3b3d16e),
            f64::from_bits(0xbc621a3ad28a3494),
            f64::from_bits(0x3fccc0a6588289a3),
            f64::from_bits(0xbc6868d09bc87c6b),
            f64::from_bits(0x3fef2ea5d753ffed),
            f64::from_bits(0x3c8cc4215f56d583),
            f64::from_bits(0x3fcdb9e15fb5a5d0),
            f64::from_bits(0xbc632e20d6cc6fc2),
            f64::from_bits(0x3fef20073086649f),
            f64::from_bits(0x3c7b940416c1984b),
            f64::from_bits(0x3fceb2a57f8ae5a3),
            f64::from_bits(0xbc60be06af572ceb),
            f64::from_bits(0x3fef10ec09c5873b),
            f64::from_bits(0x3c8d9072762c1283),
            f64::from_bits(0x3fcfaaeed4f31577),
            f64::from_bits(0xbc615d88508e32b8),
            f64::from_bits(0x3fef01549f7deea1),
            f64::from_bits(0x3c8d3c1e99e5cafd),
            f64::from_bits(0x3fd0515cbf65155c),
            f64::from_bits(0xbc79b8c29dfd8ec8),
            f64::from_bits(0x3feef141300d2f26),
            f64::from_bits(0xbc82aa1b08ded372),
            f64::from_bits(0x3fd0cd00cef36436),
            f64::from_bits(0xbc79fb0a0c93e2b5),
            f64::from_bits(0x3feee0b1fbc0f11c),
            f64::from_bits(0xbc4bfd2380bbc3b1),
            f64::from_bits(0x3fd14861aa94ddeb),
            f64::from_bits(0xbc6be881b5b615a4),
            f64::from_bits(0x3feecfa744d5efa1),
            f64::from_bits(0xbc556d0a4af541d0),
            f64::from_bits(0x3fd1c37d64c6b876),
            f64::from_bits(0x3c746076fe0dcff5),
            f64::from_bits(0x3feebe214f76efa8),
            f64::from_bits(0xbc802f9f12ba543e),
            f64::from_bits(0x3fd23e52111aaf36),
            f64::from_bits(0xbc74f080334eff18),
            f64::from_bits(0x3feeac2061bbaf4f),
            f64::from_bits(0x3c62c1d53e94658d),
            f64::from_bits(0x3fd2b8ddc43eb49f),
            f64::from_bits(0x3c61553899f2d807),
            f64::from_bits(0x3fee99a4c3a7cd83),
            f64::from_bits(0xbc82264b1bc53ce8),
            f64::from_bits(0x3fd3331e94049f87),
            f64::from_bits(0x3c7e0cb6b40c302c),
            f64::from_bits(0x3fee86aebf29a9ed),
            f64::from_bits(0x3c89397afdbb58a7),
            f64::from_bits(0x3fd3ad129769d3d8),
            f64::from_bits(0x3c003d5504878398),
            f64::from_bits(0x3fee733ea0193d40),
            f64::from_bits(0xbc86428b3546ce13),
            f64::from_bits(0x3fd426b7e69ee697),
            f64::from_bits(0xbc7f09c75705c59f),
            f64::from_bits(0x3fee5f54b436e9d0),
            f64::from_bits(0x3c87eb0fd02fc8bc),
            f64::from_bits(0x3fd4a00c9b0f3d20),
            f64::from_bits(0x3c7823ba6bb08ead),
            f64::from_bits(0x3fee4af14b2a449c),
            f64::from_bits(0xbc868ca02e8a6833),
            f64::from_bits(0x3fd5190ecf68a77a),
            f64::from_bits(0x3c7b357155eef0f3),
            f64::from_bits(0x3fee3614b680d6a5),
            f64::from_bits(0xbc727793aa015237),
            f64::from_bits(0x3fd591bc9fa2f597),
            f64::from_bits(0x3c67c74bac3fe0cb),
            f64::from_bits(0x3fee20bf49acd6c1),
            f64::from_bits(0xbc5660aec7ef636c),
            f64::from_bits(0x3fd60a1429078775),
            f64::from_bits(0x3c5b1fd80ba89133),
            f64::from_bits(0x3fee0af15a03dbce),
            f64::from_bits(0x3c5fe8e702771ae6),
            f64::from_bits(0x3fd682138a38d7f7),
            f64::from_bits(0xbc7d889202444aad),
            f64::from_bits(0x3fedf4ab3ebd875e),
            f64::from_bits(0xbc8e2d8a7e6736c4),
            f64::from_bits(0x3fd6f9b8e33a0255),
            f64::from_bits(0x3c742bc14ee9da0d),
            f64::from_bits(0x3feddded50f228d6),
            f64::from_bits(0xbc6e80c8d42ba2bf),
            f64::from_bits(0x3fd7710255764214),
            f64::from_bits(0xbc66ead7314bb6ce),
            f64::from_bits(0x3fedc6b7eb995912),
            f64::from_bits(0x3c54b364776dcd35),
            f64::from_bits(0x3fd7e7ee03c86d4e),
            f64::from_bits(0xbc7b63bcdabf5af2),
            f64::from_bits(0x3fedaf0b6b888e83),
            f64::from_bits(0x3c8a249e2b5e5cea),
            f64::from_bits(0x3fd85e7a12826949),
            f64::from_bits(0x3c78a40e9b5face0),
            f64::from_bits(0x3fed96e82f71a9dc),
            f64::from_bits(0x3c8ff61bd5d2039d),
            f64::from_bits(0x3fd8d4a4a774992f),
            f64::from_bits(0x3c744a02ea766326),
            f64::from_bits(0x3fed7e4e97e17b4a),
            f64::from_bits(0xbc63b770352bed94),
            f64::from_bits(0x3fd94a6be9f546c5),
            f64::from_bits(0xbc769ce13e683f58),
            f64::from_bits(0x3fed653f073e4040),
            f64::from_bits(0xbc876236434bec37),
            f64::from_bits(0x3fd9bfce02e80510),
            f64::from_bits(0x3c709e39a320b0a4),
            f64::from_bits(0x3fed4bb9e1c619e0),
            f64::from_bits(0x3c8f34bb77858f61),
            f64::from_bits(0x3fda34c91cc50cca),
            f64::from_bits(0xbc5a310e3b50cecd),
            f64::from_bits(0x3fed31bf8d8d7c06),
            f64::from_bits(0x3c7e60dd3089cbdd),
            f64::from_bits(0x3fdaa95b63a09277),
            f64::from_bits(0xbc66293eb13c0381),
            f64::from_bits(0x3fed1750727d94f0),
            f64::from_bits(0x3c80d52b1ec1a48e),
            f64::from_bits(0x3fdb1d8305321617),
            f64::from_bits(0xbc7ae242cb99f519),
            f64::from_bits(0x3fecfc6cfa52ad9f),
            f64::from_bits(0x3c88b5b5508f2a0d),
            f64::from_bits(0x3fdb913e30dbac43),
            f64::from_bits(0xbc7e38ad2f6c3ff1),
            f64::from_bits(0x3fece115909a82e5),
            f64::from_bits(0x3c81f139bb31109a),
            f64::from_bits(0x3fdc048b17b140a3),
            f64::from_bits(0x3c619fe6757e9fa7),
            f64::from_bits(0x3fecc54aa2b2972e),
            f64::from_bits(0x3c64ee162ba83a98),
            f64::from_bits(0x3fdc7767ec7fd19e),
            f64::from_bits(0xbc5eb14d1a3d5826),
            f64::from_bits(0x3feca90c9fc67d0b),
            f64::from_bits(0xbc646a81485e3462),
            f64::from_bits(0x3fdce9d2e3d4a51f),
            f64::from_bits(0xbc62fc8a12dae298),
            f64::from_bits(0x3fec8c5bf8ce1a84),
            f64::from_bits(0x3c7ab3d1a1590123),
            f64::from_bits(0x3fdd5bca34047661),
            f64::from_bits(0x3c728a44a75fc29c),
            f64::from_bits(0x3fec6f39208be53b),
            f64::from_bits(0xbc8741dbfbaadb42),
            f64::from_bits(0x3fddcd4c15329c9a),
            f64::from_bits(0x3c70d4c6e171fd9a),
            f64::from_bits(0x3fec51a48b8b175e),
            f64::from_bits(0xbc61bbb43b9aa880),
            f64::from_bits(0x3fde3e56c1582a69),
            f64::from_bits(0xbc50a4821099f88f),
            f64::from_bits(0x3fec339eb01ddd81),
            f64::from_bits(0xbc8caaf5ee82c5c0),
            f64::from_bits(0x3fdeaee8744b05f0),
            f64::from_bits(0xbc5789b43c9b027d),
            f64::from_bits(0x3fec1528065b7d50),
            f64::from_bits(0xbc8892111312e828),
            f64::from_bits(0x3fdf1eff6bc4f97b),
            f64::from_bits(0x3c717212f8a7525c),
            f64::from_bits(0x3febf641081e7536),
            f64::from_bits(0x3c8b7bd71628a9a1),
            f64::from_bits(0x3fdf8e99e76abc97),
            f64::from_bits(0x3c59d950af2d00a3),
            f64::from_bits(0x3febd6ea310294f5),
            f64::from_bits(0x3c731bbcc88c109d),
            f64::from_bits(0x3fdffdb628d2f57a),
            f64::from_bits(0x3c6f4a992e905b6a),
            f64::from_bits(0x3febb723fe630f32),
            f64::from_bits(0x3c772bd2452d0a39),
            f64::from_bits(0x3fe0362939c69955),
            f64::from_bits(0xbc82d8cd78397b01),
            f64::from_bits(0x3feb96eeef58840e),
            f64::from_bits(0x3c545a3cc78fade0),
            f64::from_bits(0x3fe06d3686946e5b),
            f64::from_bits(0x3c83f5ae4538ff1b),
            f64::from_bits(0x3feb764b84b704c2),
            f64::from_bits(0xbc8f5848c21b389b),
            f64::from_bits(0x3fe0a4021e9e1001),
            f64::from_bits(0xbc86f643a13914f6),
            f64::from_bits(0x3feb553a410c104e),
            f64::from_bits(0x3c58ff7947027a16),
            f64::from_bits(0x3fe0da8b26b5672e),
            f64::from_bits(0xbc8a58def0bee909),
            f64::from_bits(0x3feb33bba89c8948),
            f64::from_bits(0x3c8ea6a51d1f6ca9),
            f64::from_bits(0x3fe110d0c4b69c3b),
            f64::from_bits(0x3c8d918998809981),
            f64::from_bits(0x3feb11d04162a4c6),
            f64::from_bits(0x3c71dd561efbc0c2),
            f64::from_bits(0x3fe146d21f8b7f82),
            f64::from_bits(0x3c7bf9535e2739a8),
            f64::from_bits(0x3feaef78930bd275),
            f64::from_bits(0xbc7f836279746f94),
            f64::from_bits(0x3fe17c8e5f2eedb0),
            f64::from_bits(0x3c635e57102e2488),
            f64::from_bits(0x3feaccb526f69de5),
            f64::from_bits(0x3c88fb6a8dd6b6cc),
            f64::from_bits(0x3fe1b204acb02fdd),
            f64::from_bits(0xbc5f190c70cbb5ff),
            f64::from_bits(0x3feaa98688308913),
            f64::from_bits(0xbc0b83d607cd5070),
            f64::from_bits(0x3fe1e7343236574c),
            f64::from_bits(0x3c722a3fa4f41d5a),
            f64::from_bits(0x3fea85ed4373e02d),
            f64::from_bits(0x3c69be06385ec792),
            f64::from_bits(0x3fe21c1c1b0394cf),
            f64::from_bits(0x3c5e5b324b23aa31),
            f64::from_bits(0x3fea61e9e72586af),
            f64::from_bits(0x3c858330e2fd453f),
            f64::from_bits(0x3fe250bb93788bbb),
            f64::from_bits(0x3c7ea3d02457bcce),
            f64::from_bits(0x3fea3d7d0352bdcf),
            f64::from_bits(0xbc868dbaeca19669),
            f64::from_bits(0x3fe28511c917a067),
            f64::from_bits(0xbc801df1d9a16b70),
            f64::from_bits(0x3fea18a729aee445),
            f64::from_bits(0x3c395e25736c0358),
            f64::from_bits(0x3fe2b91dea88421e),
            f64::from_bits(0xbc8fa371db216ab0),
            f64::from_bits(0x3fe9f368ed912f85),
            f64::from_bits(0xbc81d200c5791606),
            f64::from_bits(0x3fe2ecdf279a3082),
            f64::from_bits(0x3c8d3557e0e7e37e),
            f64::from_bits(0x3fe9cdc2e3f25e5c),
            f64::from_bits(0x3c83f99112993f62),
            f64::from_bits(0x3fe32054b148bc4f),
            f64::from_bits(0x3c8f6b42095a135b),
            f64::from_bits(0x3fe9a7b5a36a6514),
            f64::from_bits(0x3c8722cfcc9fa7a9),
            f64::from_bits(0x3fe3537db9be0367),
            f64::from_bits(0x3c6b327e7af040f0),
            f64::from_bits(0x3fe98141c42e1310),
            f64::from_bits(0x3c8d1ff80488f08d),
            f64::from_bits(0x3fe386597456282b),
            f64::from_bits(0xbc710fada93b07a8),
            f64::from_bits(0x3fe95a67e00cb1fd),
            f64::from_bits(0xbc80befda21f862d),
            f64::from_bits(0x3fe3b8e715a2840a),
            f64::from_bits(0xbc797653a7d2f07b),
            f64::from_bits(0x3fe93328926d9e92),
            f64::from_bits(0xbc8bb77003600cda),
            f64::from_bits(0x3fe3eb25d36cd53a),
            f64::from_bits(0xbc5be570e1570fc0),
            f64::from_bits(0x3fe90b84784ddaf7),
            f64::from_bits(0xbc70feb10ab93b87),
            f64::from_bits(0x3fe41d14e4ba6790),
            f64::from_bits(0x3c84608fd287ecf5),
            f64::from_bits(0x3fe8e37c303d9ad1),
            f64::from_bits(0xbc6463a4b53d4bf8),
            f64::from_bits(0x3fe44eb381cf386b),
            f64::from_bits(0xbc83ed6c1e6a5505),
            f64::from_bits(0x3fe8bb105a5dc900),
            f64::from_bits(0x3c8863e03e9474c1),
            f64::from_bits(0x3fe48000e431159f),
            f64::from_bits(0xbc8b194a7463ed10),
            f64::from_bits(0x3fe89241985d871f),
            f64::from_bits(0x3c8c48d9c413ed84),
            f64::from_bits(0x3fe4b0fc46aab761),
            f64::from_bits(0x3c20da05738cc59a),
            f64::from_bits(0x3fe869108d77a6c6),
            f64::from_bits(0x3c7338ffe2bfe9dd),
            f64::from_bits(0x3fe4e1a4e54ed51b),
            f64::from_bits(0xbc8a492f89b7c76a),
            f64::from_bits(0x3fe83f7dde701ca0),
            f64::from_bits(0xbc4152cf609bc6e8),
            f64::from_bits(0x3fe511f9fd7b351c),
            f64::from_bits(0xbc85c0e861c48831),
            f64::from_bits(0x3fe8158a31916d5d),
            f64::from_bits(0xbc6de8b90b8228de),
            f64::from_bits(0x3fe541facddbb724),
            f64::from_bits(0x3c7232c28520d391),
            f64::from_bits(0x3fe7eb362eaa1488),
            f64::from_bits(0x3c5a1d65a4a5959f),
            f64::from_bits(0x3fe571a6966d59b3),
            f64::from_bits(0x3c5c843b4d0fb198),
            f64::from_bits(0x3fe7c0827f09e54f),
            f64::from_bits(0xbc6c73d6d72aee68),
            f64::from_bits(0x3fe5a0fc98813a12),
            f64::from_bits(0xbc8d82e2b7d4227b),
            f64::from_bits(0x3fe7956fcd7f6543),
            f64::from_bits(0xbc8ab276e9d45ae4),
            f64::from_bits(0x3fe5cffc16bf8f0d),
            f64::from_bits(0x3c896cb370eb578a),
            f64::from_bits(0x3fe769fec655211f),
            f64::from_bits(0xbc6827d5cf8c68c5),
            f64::from_bits(0x3fe5fea4552a9e57),
            f64::from_bits(0x3c80b6cef7ee20b7),
            f64::from_bits(0x3fe73e30174efba1),
            f64::from_bits(0xbc65d3ae3d94ad5f),
            f64::from_bits(0x3fe62cf49921ac79),
            f64::from_bits(0xbc8edd9855b6241a),
            f64::from_bits(0x3fe712046fa77678),
            f64::from_bits(0x3c8425b0a5029c81),
            f64::from_bits(0x3fe65aec2963e755),
            f64::from_bits(0x3c8126f96b71053c),
            f64::from_bits(0x3fe6e57c800cf55e),
            f64::from_bits(0x3c860286dedbd0a6),
            f64::from_bits(0x3fe6888a4e134b2f),
            f64::from_bits(0xbc86b7d37644d5e6),
            f64::from_bits(0x3fe6b898fa9efb5d),
            f64::from_bits(0x3c715ac786ccf4b2),
            f64::from_bits(0x3fe6b5ce50b7821a),
            f64::from_bits(0xbc65d5158f702e0f),
            f64::from_bits(0x3fe68b5a92eb6253),
            f64::from_bits(0xbc89a91ad985f89c),
            f64::from_bits(0x3fe6e2b77c40bde1),
            f64::from_bits(0xbc70e729857fad53),
            f64::from_bits(0x3fe65dc1fdeb8cba),
            f64::from_bits(0xbc597c1b47337c77),
            f64::from_bits(0x3fe70f451d0a8c40),
            f64::from_bits(0x3c697ede3885770d),
            f64::from_bits(0x3fe62fcff20191c7),
            f64::from_bits(0x3c6d9143895756ef),
            f64::from_bits(0x3fe73b7680dea578),
            f64::from_bits(0xbc72248306dc12a2),
            f64::from_bits(0x3fe6018526f563df),
            f64::from_bits(0x3c846ca5e0e432d0),
            f64::from_bits(0x3fe7674af6f7b524),
            f64::from_bits(0x3c7e9d3f94ac84a8),
            f64::from_bits(0x3fe5d2e255f1f17a),
            f64::from_bits(0x3c80314104c8892b),
            f64::from_bits(0x3fe792c1d0041d52),
            f64::from_bits(0xbc8abf05eeb354eb),
            f64::from_bits(0x3fe5a3e839824077),
            f64::from_bits(0x3c8428aa2759be62),
            f64::from_bits(0x3fe7bdda5e28b3c2),
            f64::from_bits(0x3c4ad1197ccd0393),
            f64::from_bits(0x3fe574978d8e83f2),
            f64::from_bits(0x3c8f4714af282d23),
            f64::from_bits(0x3fe7e893f5037959),
            f64::from_bits(0x3c80eefbaa650c4c),
            f64::from_bits(0x3fe544f10f592ca5),
            f64::from_bits(0xbc8e7ae8e6c7a62f),
            f64::from_bits(0x3fe812ede9ae4ba4),
            f64::from_bits(0xbc87830adf402dda),
            f64::from_bits(0x3fe514f57d7bf3da),
            f64::from_bits(0x3c747a108073c259),
        ];
        const S1: f64 = f64::from_bits(0xbfc5555555555555);
        const S2: f64 = f64::from_bits(0x3f81111111110ece);
        const S3: f64 = f64::from_bits(0xbf2a01a019db08b8);
        const S4: f64 = f64::from_bits(0x3ec71de27b9a7ed9);
        const S5: f64 = f64::from_bits(0xbe5addffc2fcdf59);
        const AA: f64 = f64::from_bits(0xbfc5558000000000);
        const BB: f64 = f64::from_bits(0x3ed5555555556e24);
        const BIG: f64 = f64::from_bits(0x42c8000000000000);
        const HP0: f64 = f64::from_bits(0x3ff921fb54442d18);
        const HP1: f64 = f64::from_bits(0x3c91a62633145c07);
        const MP1: f64 = f64::from_bits(0x3ff921fb58000000);
        const MP2: f64 = f64::from_bits(0xbe4dde973c000000);
        const MP3: f64 = f64::from_bits(0xbc8cb3b399d747f2);
        const PP3: f64 = f64::from_bits(0xbc8cb3b398000000);
        const PP4: f64 = f64::from_bits(0xbacd747f23e32ed7);
        const HPINV: f64 = f64::from_bits(0x3fe45f306dc9c883);
        const TOINT: f64 = f64::from_bits(0x4338000000000000);

        pub fn ln(x: f64) -> f64 {
            let mut ix = x.to_bits();
            if ix.wrapping_sub((1.0f64 - 0.0625).to_bits())
                < (1.0f64 + f64::from_bits(0x3fb0900000000000)).to_bits()
                    - (1.0f64 - 0.0625).to_bits()
            {
                if x == 1.0 {
                    return 0.0;
                }
                let r = x - 1.0;
                let r2 = r * r;
                let r3 = r * r2;
                let inner = r3.mul_add(B[10], r2.mul_add(B[9], r.mul_add(B[8], B[7])));
                let middle = r3.mul_add(inner, r2.mul_add(B[6], r.mul_add(B[5], B[4])));
                let polynomial = r3.mul_add(middle, r2.mul_add(B[3], r.mul_add(B[2], B[1])));
                let w = r * 134217728.0;
                let rhi = (r + w) - w;
                let rlo = r - rhi;
                let w = rhi * rhi * B[0];
                let hi = r + w;
                let lo = (B[0] * rlo).mul_add(rhi + r, (r - hi) + w);
                // The reference contracts this product with the low correction.
                // Rounding before adding lo changes rare inputs near one.
                return r3.mul_add(polynomial, lo) + hi;
            }
            if !x.is_finite() || x <= 0.0 {
                return x.ln();
            }
            if x < f64::MIN_POSITIVE {
                ix = (x * 4503599627370496.0).to_bits().wrapping_sub(52u64 << 52);
            }
            let tmp = ix.wrapping_sub(0x3fe6000000000000);
            let i = ((tmp >> 45) % 128) as usize;
            let k = (tmp as i64) >> 52;
            let z = f64::from_bits(ix.wrapping_sub(tmp & (0xfffu64 << 52)));
            let r = z.mul_add(INVC[i], -1.0);
            let kd = k as f64;
            let w = kd.mul_add(f64::from_bits(0x3fe62e42fefa3800), LOGC[i]);
            let hi = w + r;
            let lo = kd.mul_add(f64::from_bits(0x3d2ef35793c76730), (w - hi) + r);
            let r2 = r * r;
            let p = r2.mul_add(r.mul_add(A[4], A[3]), r.mul_add(A[2], A[1]));
            (r * r2).mul_add(p, r2.mul_add(A[0], lo)) + hi
        }
        pub fn exp(x: f64) -> f64 {
            if !x.is_finite() || x.abs() > 500.0 {
                return x.exp();
            }
            if x.abs() < f64::from_bits(0x3c90000000000000) {
                return 1.0 + x;
            }
            let z = x * f64::from_bits(0x40671547652b82fe);
            let shifted = z + TOINT;
            let ki = shifted.to_bits();
            let kd = shifted - TOINT;
            let r = kd.mul_add(
                f64::from_bits(0xbd0cf79abc9e3b3a),
                kd.mul_add(f64::from_bits(0xbf762e42fefa0000), x),
            );
            let idx = 2 * ((ki % 128) as usize);
            let tail = f64::from_bits(ET[idx]);
            let scale = f64::from_bits(ET[idx + 1].wrapping_add(ki << 45));
            let r2 = r * r;
            let tmp = (r2 * r2).mul_add(
                r.mul_add(C[3], C[2]),
                r2.mul_add(r.mul_add(C[1], C[0]), tail + r),
            );
            scale.mul_add(tmp, scale)
        }
        const SN3: f64 = f64::from_bits(0xbfc5555555555515);
        const SN5: f64 = f64::from_bits(0x3f811110e829872f);
        const CS2: f64 = f64::from_bits(0x3fe0000000000000);
        const CS4: f64 = f64::from_bits(0xbfa5555555555535);
        const CS6: f64 = f64::from_bits(0x3f56c16bedd9e239);
        fn table(u: f64) -> (f64, f64, f64, f64) {
            let k = ((u.to_bits() as u32) as usize) * 4;
            (ST[k], ST[k + 1], ST[k + 2], ST[k + 3])
        }
        fn cos_reduced(x: f64, mut dx: f64) -> f64 {
            if x < 0.0 {
                dx = -dx;
            }
            let u = BIG + x.abs();
            let x = x.abs() - (u - BIG) + dx;
            let xx = x * x;
            let s = (x * xx).mul_add(xx.mul_add(SN5, SN3), x);
            let c = xx * xx.mul_add(xx.mul_add(CS6, CS4), CS2);
            let (sn, ssn, cs, ccs) = table(u);
            let cor = (-sn).mul_add(s, (-cs).mul_add(c, (-s).mul_add(ssn, ccs)));
            cs + cor
        }
        fn sin_reduced(x: f64, mut dx: f64) -> f64 {
            let xx = x * x;
            if x.abs() < 0.126 {
                let p = S5
                    .mul_add(xx, S4)
                    .mul_add(xx, S3)
                    .mul_add(xx, S2)
                    .mul_add(xx, S1);
                return x + p.mul_add(x, -0.5 * dx).mul_add(xx, dx);
            }
            let old = x;
            if x <= 0.0 {
                dx = -dx;
            }
            let u = BIG + x.abs();
            let x = x.abs() - (u - BIG);
            let xx = x * x;
            let s = x + (x * xx).mul_add(xx.mul_add(SN5, SN3), dx);
            let c = x.mul_add(dx, xx * xx.mul_add(xx.mul_add(CS6, CS4), CS2));
            let (sn, ssn, cs, ccs) = table(u);
            let cor = cs.mul_add(s, (-sn).mul_add(c, s.mul_add(ccs, ssn)));
            (sn + cor).copysign(old)
        }
        fn reduce(x: f64) -> (f64, f64, i32) {
            let t = x.mul_add(HPINV, TOINT);
            let xn = t - TOINT;
            let y = (-xn).mul_add(MP2, (-xn).mul_add(MP1, x));
            let t1 = xn * PP3;
            let t2 = y - t1;
            let db = (y - t2) - t1;
            let t1 = xn * PP4;
            let b = t2 - t1;
            let db = db + ((t2 - b) - t1);
            (b, db, t.to_bits() as i32 & 3)
        }
        fn quadrant(a: f64, da: f64, n: i32) -> f64 {
            let y = if n & 1 == 1 {
                cos_reduced(a, da)
            } else {
                sin_reduced(a, da)
            };
            if n & 2 == 2 {
                -y
            } else {
                y
            }
        }
        pub fn sin(x: f64) -> f64 {
            let k = (x.to_bits() >> 32) as u32 & 0x7fffffff;
            if k < 0x3e500000 {
                x
            } else if k < 0x3feb6000 {
                sin_reduced(x, 0.0)
            } else if k < 0x400368fd {
                cos_reduced(HP0 - x.abs(), HP1).copysign(x)
            } else if k < 0x419921fb {
                let (a, da, n) = reduce(x);
                quadrant(a, da, n)
            } else {
                x.sin()
            }
        }
        pub fn cos(x: f64) -> f64 {
            let k = (x.to_bits() >> 32) as u32 & 0x7fffffff;
            if k < 0x3e400000 {
                1.0
            } else if k < 0x3feb6000 {
                cos_reduced(x, 0.0)
            } else if k < 0x400368fd {
                let y = HP0 - x.abs();
                let a = y + HP1;
                let da = (y - a) + HP1;
                sin_reduced(a, da)
            } else if k < 0x419921fb {
                let (a, da, n) = reduce(x);
                quadrant(a, da, n + 1)
            } else {
                x.cos()
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn frozen_linux_primitive_streams() {
            use sha2::{Digest, Sha256};
            let mut gaussian = Sha256::new();
            for (seed, p, k) in [(6807, 8, 23), (701, 24, 32), (4660, 50, 21), (1903, 9, 24)] {
                let mut rng = crate::rng::Pcg32::for_stage(seed, p, k);
                for _ in 0..512 {
                    let x = with_frozen_s1_math(|| rng.gaussian());
                    gaussian.update(x.to_le_bytes());
                }
            }
            assert_eq!(
                format!("{:x}", gaussian.finalize()),
                "216204e1ec4b33d9431c998197c6f56325b0defb967eeeeb9244d39188177e48"
            );
            let mut exponential = Sha256::new();
            let mut sine = Sha256::new();
            for i in -4096..=4096 {
                let x = f64::from(i) / 256.0;
                exponential.update(precise_exp(x).to_le_bytes());
                sine.update(precise_sincos(x).0.to_le_bytes());
            }
            assert_eq!(
                format!("{:x}", exponential.finalize()),
                "f7872a218ba468f9d07dfa1f82414b6e0acc3edd6321b50f5cd003f9d0453abd"
            );
            assert_eq!(
                format!("{:x}", sine.finalize()),
                "44bf28d88ebbe870b2cc83e3efcf0c8f84c9b796b6165932aa66409bdcf3fed1"
            );
        }

        #[test]
        fn near_one_log_contracts_the_final_product() {
            for (input, expected) in [
                (0x3fef57dc79f05fd8, 0xbf953c6c85d3098f),
                (0x3fee22c78c832d5d, 0xbfaebaf68cea8710),
            ] {
                assert_eq!(precise_ln(f64::from_bits(input)).to_bits(), expected);
            }
        }

        #[test]
        fn frozen_worker_gaussian_keeps_the_integer_stream() {
            let draw = || {
                let mut rng = crate::rng::Pcg32::for_stage(6807, 8, 23);
                (0..13).map(|_| rng.gaussian()).last().unwrap()
            };
            let old = draw().to_bits();
            let fixed = with_frozen_s1_math(draw);
            assert_eq!(fixed.to_bits(), 0x3feaaccafda20403);
            assert_eq!(
                draw().to_bits(),
                old,
                "ordinary interpreter arithmetic changed"
            );
        }

        #[test]
        fn scope_restores_on_nesting_unwind_and_other_threads() {
            assert!(!S1.with(Cell::get));
            with_frozen_s1_math(|| {
                assert!(S1.with(Cell::get));
                with_frozen_s1_math(|| assert!(S1.with(Cell::get)));
                assert!(S1.with(Cell::get));
                assert!(!std::thread::spawn(|| S1.with(Cell::get)).join().unwrap());
            });
            assert!(!S1.with(Cell::get));
            let result =
                std::panic::catch_unwind(|| with_frozen_s1_math(|| panic!("scope control")));
            assert!(result.is_err());
            assert!(!S1.with(Cell::get));
        }

        #[test]
        fn special_values_keep_legacy_behavior() {
            assert_eq!(precise_sincos(-0.0).0.to_bits(), (-0.0f64).to_bits());
            assert_eq!(precise_sincos(-0.0).1, 1.0);
            assert_eq!(precise_ln(0.0), f64::NEG_INFINITY);
            assert!(precise_ln(-1.0).is_nan());
            assert_eq!(precise_exp(f64::NEG_INFINITY), 0.0);
            assert_eq!(precise_exp(f64::INFINITY), f64::INFINITY);
            assert!(precise_sincos(f64::INFINITY).0.is_nan());
        }
    }
}
