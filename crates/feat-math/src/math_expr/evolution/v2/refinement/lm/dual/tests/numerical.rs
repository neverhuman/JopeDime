use super::super::range::{SCALE_DOWN, SCALE_UP};
use super::*;

#[test]
fn overflowing_pow_slopes_leave_finite_lanes() {
    let at = |source: &str, row: &[f64]| {
        let (expression, sites) = sites_of(source);
        let program = Program::lower(&expression.root, &sites).unwrap();
        let dual = program.row(&theta_of(&expression, &sites), row);
        assert_eq!(
            dual.value.to_bits(),
            native(&expression, row).0.to_bits(),
            "{source}"
        );
        dual
    };
    // A saturated exponent (9 > 8) is a constant, so its lane is exactly 0, although
    // v ln a = 6.6e307 * 88.6 overflows; 0 * inf made a NaN lane before.
    let saturated = at("log(c0^9.0)", &[3e38]);
    assert_eq!(saturated.tangent[0], 0.0);
    assert!(!saturated.nonsmooth);
    // Inside the clamp v ln a overflows (v = 1.3e308), although log(c0^e) has the finite slope
    // ln c0 = 89.8 that forward mode cannot carry through v: the lane is 0 and the row is
    // nonsmooth, never inf.
    let overflowed = at("log(c0^7.9)", &[1e39]);
    assert_eq!(overflowed.tangent[0], 0.0);
    assert!(overflowed.nonsmooth);
    // A tiny base with e < 1: e a^(e-1) overflows to -inf, and the ratio form gives the exact
    // slope e v (a'/a); the base's zero lane for the exponent no longer turns NaN.
    let tiny = at("(1.0*c0)^(-0.03)", &[1e-300]);
    let v = tiny.value;
    assert!(v > 1e8 && v < 1e10, "{v}");
    assert_eq!(tiny.tangent[0], -0.03 * v);
    assert_eq!(tiny.tangent[1], v * 1e-300f64.ln());
    assert!(!tiny.nonsmooth);
}

/// Independent review of #195, R3: a finite base slope times a large lane, and an exponent
/// term times a large lane, overflowed past the `^` rule into the normal equations. Both rows
/// have finite native values and no faults.
#[test]
fn power_terms_that_overflow_after_their_slopes_are_zero() {
    // Each case with the lane its overflow zeroes.
    for (source, row, zeroed) in [
        ("log((1.0*c0-c1+1e-300)^(-0.02))", vec![1e308, 1e308], 0),
        ("0.001*log(c2^(1.0*c0-c1))", vec![1e308, 1e308, 1e200], 1),
    ] {
        let (expression, sites) = sites_of(source);
        let program = Program::lower(&expression.root, &sites).unwrap();
        let dual = program.row(&theta_of(&expression, &sites), &row);
        let (value, faults) = native(&expression, &row);
        assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
        assert!(
            value.is_finite() && faults.is_empty(),
            "{source}: {value} {faults:?}"
        );
        assert!(
            dual.tangent.iter().all(|lane| lane.is_finite()),
            "{source}: {:?}",
            dual.tangent
        );
        assert_eq!(dual.tangent[zeroed].to_bits(), 0.0f64.to_bits(), "{source}");
        assert!(dual.nonsmooth, "{source}");
    }
}

/// The mirror of R3: for a large base and e < 0, e a^(e-1) underflows to 0 while v = a^e is a
/// normal number, which left a silent zero lane on a smooth row. The ratio form gives the slope.
#[test]
fn power_slopes_that_underflow_take_the_ratio_form() {
    // d/dθ log((θ c0)^e) = e / θ = e at θ = 1.
    for (source, row, slope) in [
        ("log((1.0*c0)^(-1.0))", 1e170, -1.0),
        ("log((1.0*c0)^(-7.9))", 1e37, -7.9),
    ] {
        let (expression, sites) = sites_of(source);
        let program = Program::lower(&expression.root, &sites).unwrap();
        let dual = program.row(&theta_of(&expression, &sites), &[row]);
        let (value, faults) = native(&expression, &[row]);
        assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
        assert!(
            value.is_finite() && faults.is_empty(),
            "{source}: {value} {faults:?}"
        );
        assert_eq!(dual.tangent[0], slope, "{source}: {:?}", dual.tangent);
        assert!(!dual.nonsmooth, "{source}");
    }
}

/// Independent review of #195, R4: past `|v|` of about 27.3 `e^(-v²)` is 0 while `2 v P(t)`
/// can overflow. The slope is an exact 0, not NaN, and the row stays smooth.
#[test]
fn erf_tails_have_an_exact_zero_slope() {
    let (expression, sites) = sites_of("erf(1.0*c0)");
    let program = Program::lower(&expression.root, &sites).unwrap();
    for x in [1e308, -1e308, 1e200, 40.0, -40.0] {
        let dual = program.row(&[1.0], &[x]);
        let (value, faults) = native(&expression, &[x]);
        assert_eq!(dual.value.to_bits(), value.to_bits(), "{x}");
        assert_eq!(value.abs(), 1.0, "{x}");
        assert!(faults.is_empty(), "{x}: {faults:?}");
        assert_eq!(dual.tangent[0].to_bits(), 0.0f64.to_bits(), "{x}");
        assert!(!dual.nonsmooth, "{x}");
    }
    // Before the tail the closed form still gives a positive, finite slope.
    let near = program.row(&[1.0], &[5.0]);
    assert!(near.tangent[0] > 0.0 && near.tangent[0].is_finite());
}

/// The same class as R3/R4 in other rules: an intermediate that overflows while the native
/// value and the slope are finite. `atan` squares its argument, and `/` forms q b'. Each
/// case's slope is checked against its closed form.
#[test]
fn large_arguments_keep_finite_slopes() {
    for (source, row, expected) in [
        // d/dθ atan(θ c0) = c0 / (1 + (θ c0)²) = 1 / 2e154 at θ c0 = 2e154.
        ("atan(1.0*c0)", vec![2e154], 1.0 / 2e154),
        // d/dθ c0 / (θ c1) = -c0 / (θ² c1) = -1e20 at θ = 1e-10, c0 = c1 = 1e300.
        ("c0/(1e-10*c1)", vec![1e300, 1e300], -1e20),
    ] {
        let (expression, sites) = sites_of(source);
        let program = Program::lower(&expression.root, &sites).unwrap();
        let dual = program.row(&theta_of(&expression, &sites), &row);
        let (value, faults) = native(&expression, &row);
        assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
        assert!(
            value.is_finite() && faults.is_empty(),
            "{source}: {value} {faults:?}"
        );
        let slope = dual.tangent[0];
        assert!(
            (slope - expected).abs() <= 1e-15 * expected.abs(),
            "{source}: {slope} against {expected}"
        );
        assert!(!dual.nonsmooth, "{source}");
    }
}

/// Independent review of #195, R5: in `(c0/(θ c1)) c2` at θ = 1e13 with c1 = 1e-24, b = 1e-11
/// and b' = 1e-24, so q b' fell below the smallest normal number before the division by b
/// scaled it back up. On the reviewer's row it rounded to 0, a silent zero lane on a smooth
/// row; on the second it kept about 18 bits (a relative error of 1.3e-6). The slope is
/// -value/θ.
#[test]
fn division_numerators_that_underflow_keep_their_slope() {
    assert_eq!(SCALE_UP, 2f64.powi(600));
    assert_eq!(SCALE_UP * SCALE_DOWN, 1.0);
    let (expression, sites) = sites_of("(c0/(1e13*c1))*c2");
    let program = Program::lower(&expression.root, &sites).unwrap();
    let theta = theta_of(&expression, &sites);
    // Each row with the relative tolerance its intermediate slope -q/θ allows: on the first
    // row that slope is about -1e-313, a subnormal with about 34 bits.
    for (row, tolerance) in [
        ([1e-311, 1e-24, 1e308], 1e-9),
        ([1e-305, 1e-24, 1e300], 1e-14),
    ] {
        let dual = program.row(&theta, &row);
        let (value, faults) = native(&expression, &row);
        assert_eq!(dual.value.to_bits(), value.to_bits(), "{row:?}");
        assert!(
            value.is_finite() && faults.is_empty(),
            "{row:?}: {value} {faults:?}"
        );
        let expected = -value / theta[0];
        let slope = dual.tangent[0];
        assert!(
            (slope - expected).abs() <= tolerance * expected.abs(),
            "{row:?}: {slope} against {expected}"
        );
        assert!(!dual.nonsmooth, "{row:?}");
    }
}

/// The slope of a one-site row, checked against the native value bit for bit and for no
/// fault, and against `expected` to `tolerance` relative, on a smooth row.
fn check_slope(source: &str, row: &[f64], expected: f64, tolerance: f64) {
    let (expression, sites) = sites_of(source);
    let program = Program::lower(&expression.root, &sites).unwrap();
    let dual = program.row(&theta_of(&expression, &sites), row);
    let (value, faults) = native(&expression, row);
    assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
    assert!(
        value.is_finite() && faults.is_empty(),
        "{source}: {value} {faults:?}"
    );
    let slope = dual.tangent[0];
    assert!(
        (slope - expected).abs() <= tolerance * expected.abs(),
        "{source}: {slope} against {expected}"
    );
    assert!(!dual.nonsmooth, "{source}");
}

/// Independent review of #195, R7: `q = a / b` can itself underflow to 0 while the slope at
/// the division, `-a b' / b²`, is normal. Each expected slope is formed in an order whose
/// intermediates are normal.
#[test]
fn division_quotients_that_underflow_keep_their_slope() {
    let a = f64::from_bits(1);
    let b = 1e-300 * 1e308;
    let slope = -(a * 1e308) / (b * b);
    check_slope("c0/(1e-300*c1)", &[a, 1e308], slope, 1e-14);
    check_slope(
        "(c0/(1e-300*c1))*c2",
        &[a, 1e308, 1e30],
        slope * 1e30,
        1e-14,
    );
}

/// Independent review of #195, R8: for a tiny base and e > 1 both `e a^(e-1)` and `v = a^e`
/// underflow, while the slope at the power, `e a^(e-1) a'`, is normal; and for a tiny base and
/// a large exponent lane, `v ln a` is subnormal while `v ln a b'` is normal.
#[test]
fn power_coefficients_that_underflow_keep_their_slope() {
    // a = 1e-200 and a' = 1e110: 3 a² a' = 3e-290.
    let a = 1e-310 * 1e110;
    let slope = 3.0 * (a * 1e100) * (a * 1e100) * 1e110 * 1e-200;
    check_slope("(1e-310*c0)^3.0", &[1e110], slope, 1e-12);
    // A non-integral exponent: a = 1e-150, 3.5 a^2.5 a' = 3.5e-265.
    let a = 1e-260f64 * 1e110;
    let slope = 3.5 * (a * 1e100).powf(2.5) * 1e110 * 1e-250;
    check_slope("(1e-260*c0)^3.5", &[1e110], slope, 1e-12);
    // The exponent term: a = 1e-40 and e = 7.9, so v = 1e-316 is subnormal, and
    // d/dθ a^(θ c1) = v ln(a) c1 = -7.3e-15.
    let (a, e) = (1e-40f64, 1e-299 * 7.9e299);
    let half = a.powf(e / 2.0);
    let slope = half * (half * (a.ln() * 7.9e299));
    check_slope("c0^(1e-299*c1)", &[a, 7.9e299], slope, 1e-12);
}

/// Independent review of #195, R6: `1 - tanh²` cancels to 0 once `tanh` rounds to 1, and the
/// coefficients of `tanh`, `erf` and `sigmoid` can leave the normal range while a large lane
/// brings their product back into it.
#[test]
fn saturating_functions_keep_their_slope() {
    // 20 sech²(20), to 100 digits 3.398683404233271e-16.
    check_slope("tanh(1.0*c0)", &[20.0], 3.398683404233271e-16, 1e-14);
    // sech²(x) = 4 e^-2x / (1 + e^-2x)² at x = 400 is about 1e-347; times a' = 1e301.
    let x = 4e-299f64 * 1e301;
    let slope = 4.0 * ((-x).exp() * 1e301) * (-x).exp();
    check_slope("tanh(4e-299*c0)", &[1e301], slope, 1e-12);
    // erf'(30) is about 1e-391: E = e^-900 is split in two halves.
    let u = 3e-300f64 * 1e301;
    let half = (-(u * u) / 2.0).exp();
    let slope = erf_bracket(u) * (half * 1e301) * half;
    check_slope("erf(3e-300*c0)", &[1e301], slope, 1e-12);
    // σ (1 - σ) at 29 from the argument; 1 - σ would keep only about 12 bits here.
    let e = (-29f64).exp();
    check_slope(
        "sigmoid(1.0*c0)",
        &[29.0],
        29.0 * e / ((1.0 + e) * (1.0 + e)),
        1e-14,
    );
}

/// Independent review of #195, R9 and R10: an intermediate overflowed although the slope at the
/// node is finite, in the `^` ratio form (`a'/a` for a tiny base) and in `exp` at its clamp tie
/// (`v a'` before the factor 1/2). Each expected slope is formed in an order whose
/// intermediates stay in range, and checked against the review's 100-digit values.
#[test]
fn ratio_and_clamp_factor_products_keep_their_slope() {
    // R9: a = 1e-312 × 5e-12 = 2^-1074, e = 0.01, a' = 5e-12: e v a'/a ≈ 5.917e306.
    let source = "(1e-312*c0)^0.01";
    let (expression, _) = sites_of(source);
    let (value, _) = native(&expression, &[5e-12]);
    let a = 1e-312f64 * 5e-12;
    let slope = (0.01 * value * 5e-12) / a;
    assert!(
        (slope - 5.917294631406019e306).abs() <= 1e-12 * slope,
        "{slope}"
    );
    check_slope(source, &[5e-12], slope, 1e-12);
    check_slope(
        "((1e-312*c0)^0.01)*c1",
        &[5e-12, 1e-300],
        slope * 1e-300,
        1e-12,
    );
    // R10: at exp's clamp tie (argument exactly 30) the slope is half of e^30 a', with
    // a' = 3e295: 0.5 e^30 3e295 ≈ 1.603e308 at the exp node, then times 0.01.
    let source = "0.01*exp(1.0*c0-c1+30.0)";
    let (expression, sites) = sites_of(source);
    let program = Program::lower(&expression.root, &sites).unwrap();
    let row = [3e295, 3e295];
    let dual = program.row(&theta_of(&expression, &sites), &row);
    let (value, faults) = native(&expression, &row);
    assert_eq!(dual.value.to_bits(), value.to_bits());
    assert!(value.is_finite() && faults.is_empty(), "{value} {faults:?}");
    let expected = 0.01 * ((0.5 * 30f64.exp()) * 3e295);
    assert!((expected - 1.6029711872286693e306).abs() <= 1e-12 * expected);
    // Sites in order: 0.01, 1.0, 30.0; the 1.0 lane is the one through the tie.
    assert_eq!(theta_of(&expression, &sites), vec![0.01, 1.0, 30.0]);
    let lane = dual.tangent[1];
    assert!(
        (lane - expected).abs() <= 1e-14 * expected,
        "{lane} against {expected}"
    );
    // The tie is differentiated, so the row is nonsmooth.
    assert!(dual.nonsmooth);
}

#[test]
fn lanes_are_independent_of_one_another() {
    let rows: Vec<Vec<f64>> = (0..7)
        .map(|i| vec![0.3 * f64::from(i) - 0.9, 1.0])
        .collect();
    for source in [
        "1.5*exp(0.4*c0)+0.25*sin(2.0*c1)",
        "clip(0.5*c0,-0.25,0.75)*min(1.0,2.0*c0)",
        "(1.2*c0+3.0)^1.7-erf(0.3*c0)",
    ] {
        let (expression, sites) = sites_of(source);
        let all = Program::lower(&expression.root, &sites).unwrap();
        let theta = theta_of(&expression, &sites);
        for (k, site) in sites.iter().enumerate() {
            let one = Program::lower(&expression.root, std::slice::from_ref(site)).unwrap();
            for row in &rows {
                let full = all.row(&theta, row).tangent[k];
                let alone = one.row(&[theta[k]], row).tangent[0];
                assert_eq!(
                    full.to_bits(),
                    alone.to_bits(),
                    "{source} lane {k} on {row:?}"
                );
            }
        }
    }
}
