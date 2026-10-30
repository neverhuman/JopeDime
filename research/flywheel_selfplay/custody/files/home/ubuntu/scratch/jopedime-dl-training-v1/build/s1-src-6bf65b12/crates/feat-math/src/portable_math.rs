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
