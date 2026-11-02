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
        -8.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
        4.0, 7.0, 8.0, 0.333_333, 2.718,
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
