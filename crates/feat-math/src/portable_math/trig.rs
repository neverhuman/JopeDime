use super::*;

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
