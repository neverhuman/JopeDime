// Standalone cross-language parity reference: portable exp/ln.
// Bit-for-bit copy of vendor/feat-math/src/portable_math.rs (verified ≤1 ULP).
// Prints "fn input_hex output_hex" so C++/JS ports can be diffed exactly.

fn high_word(x: f64) -> u32 { (x.to_bits() >> 32) as u32 }
fn low_word(x: f64) -> u32 { x.to_bits() as u32 }
fn with_high_word(x: f64, hi: u32) -> f64 {
    f64::from_bits(((hi as u64) << 32) | (x.to_bits() & 0xffff_ffff))
}
fn two_pow(n: i32) -> f64 { f64::from_bits((((0x3ff + n) as i64) as u64) << 52) }

fn scalbn(x: f64, mut n: i32) -> f64 {
    let p1023 = two_pow(1023);
    let pm1022 = two_pow(-1022);
    let p53 = two_pow(53);
    let mut y = x;
    if n > 1023 {
        y *= p1023; n -= 1023;
        if n > 1023 { y *= p1023; n -= 1023; if n > 1023 { n = 1023; } }
    } else if n < -1022 {
        y *= pm1022 * p53; n += 1022 - 53;
        if n < -1022 { y *= pm1022 * p53; n += 1022 - 53; if n < -1022 { n = -1022; } }
    }
    y * two_pow(n)
}

fn exp(x: f64) -> f64 {
    const HALF: [f64; 2] = [0.5, -0.5];
    const LN2HI: [f64; 2] = [6.931_471_803_691_238_164_9e-1, -6.931_471_803_691_238_164_9e-1];
    const LN2LO: [f64; 2] = [1.908_214_929_270_587_700_0e-10, -1.908_214_929_270_587_700_0e-10];
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
    if hx >= 0x4086_2E42 {
        if hx >= 0x7ff0_0000 {
            if ((hx & 0xf_ffff) | low_word(x)) != 0 { return x + x; }
            return if xsb == 0 { x } else { 0.0 };
        }
        if x > O_THRESHOLD { return HUGE * HUGE; }
        if x < U_THRESHOLD { return TWOM1000 * TWOM1000; }
    }
    let mut k: i32 = 0;
    let mut hi = 0.0; let mut lo = 0.0; let xr;
    if hx > 0x3fd6_2e42 {
        if hx < 0x3FF0_A2B2 {
            hi = x - LN2HI[xsb]; lo = LN2LO[xsb]; k = 1 - (xsb as i32) - (xsb as i32);
        } else {
            k = (INVLN2 * x + HALF[xsb]) as i32;
            let t = k as f64; hi = x - t * LN2HI[0]; lo = t * LN2LO[0];
        }
        xr = hi - lo;
    } else if hx < 0x3e30_0000 { return 1.0 + x; } else { xr = x; }
    let t = xr * xr;
    let c = xr - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 { return 1.0 - ((xr * c) / (c - 2.0) - xr); }
    let y = 1.0 - ((lo - (xr * c) / (2.0 - c)) - hi);
    scalbn(y, k)
}

fn ln(x: f64) -> f64 {
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
        if ((hx & 0x7fff_ffff) | (lx as i32)) == 0 { return -TWO54 / 0.0; }
        if hx < 0 { return (x - x) / 0.0; }
        k -= 54; xx *= TWO54; hx = high_word(xx) as i32;
    }
    if hx >= 0x7ff0_0000 { return x + x; }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x95f64) & 0x100000;
    xx = with_high_word(xx, (hx | (i ^ 0x3ff0_0000)) as u32);
    k += i >> 20;
    let f = xx - 1.0;
    if (0x000f_ffff & (2 + hx)) < 3 {
        if f == 0.0 {
            if k == 0 { return 0.0; }
            let dk = k as f64; return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - 0.333_333_333_333_333_33 * f);
        if k == 0 { return f - r; }
        let dk = k as f64; return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }
    let s = f / (2.0 + f);
    let dk = k as f64;
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let i2 = (hx as i32) - 0x6147a;
    let j2 = 0x6b851 - (hx as i32);
    if (i2 | j2) > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 { f - (hfsq - s * (hfsq + r)) }
        else { dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f) }
    } else if k == 0 { f - s * (f - r) }
    else { dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f) }
}

fn grid() -> Vec<f64> {
    let mut g = Vec::new();
    let mut i = -1900i32;
    while i <= 1900 { g.push(i as f64 * 0.37); i += 1; }
    for e in -60..60 { g.push(two_pow(e)); g.push(-two_pow(e)); g.push(1.0 + two_pow(e)); }
    g.extend_from_slice(&[0.0, -0.0, 1.0, -1.0, 709.0, -745.0, 710.0, -746.0,
        f64::from_bits(1), f64::from_bits(0x000f_ffff_ffff_ffff), 1e300, 1e-300,
        f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    g
}

// NaN payload bits are not standardized across languages; canonicalize.
fn hexout(v: f64) -> String {
    if v.is_nan() { "NaN".to_string() } else { format!("{:016x}", v.to_bits()) }
}

fn main() {
    let mut out = String::new();
    for &x in &grid() {
        let xb = x.to_bits();
        out.push_str(&format!("exp {:016x} {}\n", xb, hexout(exp(x))));
        out.push_str(&format!("ln {:016x} {}\n", xb, hexout(ln(x))));
    }
    print!("{out}");
}
