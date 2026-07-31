// Full-kernel reference: reuses the ACTUAL portable_math.rs source (not a copy)
// via include!, so the Python transcription is verified against the real thing.
// The #[cfg(test)] module in portable_math.rs is excluded in this non-test build.
#[path = "../../../../vendor/feat-math/src/portable_math.rs"]
mod pm;
use pm::*;

fn two_pow(n: i32) -> f64 {
    f64::from_bits((((0x3ff + n) as i64) as u64) << 52)
}

fn grid() -> Vec<f64> {
    let mut g = Vec::new();
    let mut i = -900i32;
    while i <= 900 {
        g.push(i as f64 * 0.41);
        i += 1;
    }
    for e in -55..55 {
        g.push(two_pow(e));
        g.push(-two_pow(e));
        g.push(1.0 + two_pow(e));
    }
    g.extend_from_slice(&[
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.5,
        -3.5,
        709.0,
        -745.0,
        710.0,
        f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1e300,
        1e-300,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]);
    g
}

fn hexout(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else {
        format!("{:016x}", v.to_bits())
    }
}

fn main() {
    let g = grid();
    let mut out = String::new();
    for &x in &g {
        let xb = x.to_bits();
        out.push_str(&format!("exp {:016x} {}\n", xb, hexout(exp(x))));
        out.push_str(&format!("ln {:016x} {}\n", xb, hexout(ln(x))));
        out.push_str(&format!("log1p {:016x} {}\n", xb, hexout(log1p(x))));
        out.push_str(&format!("expm1 {:016x} {}\n", xb, hexout(expm1(x))));
        out.push_str(&format!("tanh {:016x} {}\n", xb, hexout(tanh(x))));
        out.push_str(&format!("sin {:016x} {}\n", xb, hexout(sin(x))));
        out.push_str(&format!("cos {:016x} {}\n", xb, hexout(cos(x))));
        out.push_str(&format!("atan {:016x} {}\n", xb, hexout(atan(x))));
    }
    // pow(base, exp) and atan2(y, x) over a smaller 2-D grid.
    let exps: [f64; 11] = [-8.0, -2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 3.0, 8.0, 0.3333];
    for &b in &g {
        if !(b > 0.0 && b.is_finite()) {
            continue;
        }
        for &e in &exps {
            out.push_str(&format!(
                "pow {:016x} {:016x} {}\n",
                b.to_bits(),
                e.to_bits(),
                hexout(pow(b, e))
            ));
        }
    }
    let ys: [f64; 7] = [-3.0, -1.0, 0.0, 1.0, 3.0, f64::INFINITY, f64::NEG_INFINITY];
    let xs: [f64; 7] = [-3.0, -1.0, 0.0, 1.0, 3.0, f64::INFINITY, f64::NEG_INFINITY];
    for &y in &ys {
        for &xv in &xs {
            out.push_str(&format!(
                "atan2 {:016x} {:016x} {}\n",
                y.to_bits(),
                xv.to_bits(),
                hexout(atan2(y, xv))
            ));
        }
    }
    print!("{out}");
}
