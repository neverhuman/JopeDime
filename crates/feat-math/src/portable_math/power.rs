use super::*;

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
