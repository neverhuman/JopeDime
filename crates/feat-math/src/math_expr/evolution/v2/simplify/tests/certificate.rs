use super::*;

/// Lane D pre-review of B19: the certificate's soundness margins had no test that fails when
/// they are removed (the evaluation-error term, and the outward rounding of mul, half and the
/// upper side of add), and its threshold, degree and depth were trusted.
#[test]
fn the_certificate_keeps_its_margins_and_refuses_inputs_it_cannot_bound() {
    // 1e6 + v² on [-1, 1]: the margin includes 2 γ_4 (1e6 + 1) ≈ 8.9e-10.
    match certify(&[1e6, 0.0, 1.0], -1.0, 1.0, 1e-12, 4) {
        Certificate::Positive { margin, .. } => assert!(margin > 1e-12 + 8e-10, "{margin}"),
        other => panic!("{other:?}"),
    }
    // A margin of the threshold alone would certify 1.1e-12 + 1000 v on [0, 1]; the Horner
    // term lifts it to 1.44e-12.
    assert_eq!(
        certify(&[1.1e-12, 1000.0], 0.0, 1.0, 1e-12, 8),
        Certificate::Refused
    );
    assert!(matches!(
        certify(&[1.5e-12, 1000.0], 0.0, 1.0, 1e-12, 8),
        Certificate::Positive { .. }
    ));
    // Each operation encloses a result that rounding would otherwise lose.
    let product = Interval::point(0.1).mul(Interval::point(0.2));
    assert!(
        product.lo < 0.1 * 0.2 && 0.1 * 0.2 < product.hi,
        "{product:?}"
    );
    let half = Interval::point(5e-324).half();
    assert!(half.lo <= 0.0 && half.hi >= 5e-324, "{half:?}");
    assert!(Interval::point(0.1).add(Interval::point(0.2)).hi > 0.30000000000000004);
    let sum = Interval::point(1.0).add(Interval::point(1e-20));
    assert!(sum.hi > 1.0, "{sum:?}");
    let difference = Interval::point(1.0).sub(Interval::point(1e-20));
    assert!(difference.lo < 1.0, "{difference:?}");
    // v - 0.5 has a root in [0, 1], and -0.5 is negative: a threshold below 1e-12 is refused
    // instead of certifying either.
    assert_eq!(
        certify(&[-0.5, 1.0], 0.0, 1.0, -1.0, 4),
        Certificate::Refused
    );
    assert_eq!(certify(&[-0.5], 0.0, 1.0, 0.0, 4), Certificate::Refused);
    let positive = [1.0, 0.0, 0.5];
    assert_eq!(
        certify(&positive, 0.0, 1.0, f64::NAN, 4),
        Certificate::Refused
    );
    let deep = MAX_CERTIFY_DEPTH + 1;
    assert_eq!(
        certify(&positive, 0.0, 1.0, 1e-12, deep),
        Certificate::Refused
    );
    assert!(matches!(
        certify(&positive, 0.0, 1.0, 1e-12, MAX_CERTIFY_DEPTH),
        Certificate::Positive { .. }
    ));
    let ones = [1.0; MAX_DEGREE + 2];
    assert_eq!(certify(&ones, 0.0, 1.0, 1e-12, 4), Certificate::Refused);
    assert!(matches!(
        certify(&ones[..=MAX_DEGREE], 0.0, 1.0, 1e-12, 4),
        Certificate::Positive { .. }
    ));
}

/// Lane D pre-review of B19: `iterations` is the step of the retained fit, so a capped run
/// reported 2 after 32 solves; `steps` counts the solves.
#[test]
fn steps_count_the_exchange_beside_the_retained_step() {
    let geometric = |points: usize| -> Vec<(f64, f64)> {
        (0..points)
            .map(|i| (f64::from(1u32 << i), if i % 3 == 0 { 1.0 } else { -1.0 }))
            .collect()
    };
    let capped = minimax(&geometric(15), 11).unwrap();
    assert_eq!((capped.iterations, capped.steps), (2, MAX_ITERATIONS));
    let stalled = minimax(&geometric(8), 6).unwrap();
    assert_eq!((stalled.iterations, stalled.steps), (1, 1));
    let singular = minimax(&geometric(27), 3).unwrap();
    assert_eq!((singular.iterations, singular.steps), (1, 2));
}

/// Independent review of B19 (R1): one step whose |h| fell was taken for a spent exchange. On
/// these 18 samples at degree 5 the run stopped NoiseLimited at step 9 with max|e| = 2.29e-5,
/// where the next exchange raises |h| and reaches 1.72e-5.
#[test]
fn a_noise_floor_stop_needs_a_spent_exchange_not_a_falling_h() {
    let samples = [
        (-0.999990739939228, 9999999999.999983),
        (-0.9978979429134074, 10000000000.000017),
        (0.7074203759847726, 10000000000.00001),
        (0.912930560725927, 10000000000.000006),
        (-0.6831402736477055, 9999999999.999983),
        (-0.977500969634282, 9999999999.999985),
        (-0.8789742671837542, 10000000000.000002),
        (-0.9985609737561163, 9999999999.999983),
        (-0.8468934336899454, 10000000000.000017),
        (-0.3649387527514045, 10000000000.000015),
        (0.23064368667000013, 9999999999.999989),
        (-0.8362009505253496, 9999999999.999985),
        (-0.309612662189795, 9999999999.999989),
        (0.44993701364736216, 9999999999.999989),
        (-0.526377661688965, 9999999999.999992),
        (-0.010832884454037295, 10000000000.00001),
        (-0.8262491006272136, 10000000000.000004),
        (0.6818349395106322, 10000000000.000015),
    ];
    let fit = minimax(&samples, 5).unwrap();
    assert_eq!(
        (fit.stop, fit.iterations),
        (Stop::NoiseLimited, 10),
        "{fit:?}"
    );
    assert!(fit.max_error <= 1.71661376953125e-5, "{fit:?}");
    assert!(fit.levelled <= fit.max_error, "{fit:?}");
}

/// `a + b` as the rounded sum and its exact error (Knuth's TwoSum).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_part = sum - a;
    (sum, (a - (sum - b_part)) + (b - b_part))
}

/// `a b` as the rounded product and its exact error (Dekker's TwoProduct, splitting at
/// 2^27 + 1, with no fused multiply-add).
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let split = |x: f64| {
        let scaled = 134_217_729.0 * x;
        let high = scaled - (scaled - x);
        (high, x - high)
    };
    let product = a * b;
    let ((a_high, a_low), (b_high, b_low)) = (split(a), split(b));
    let error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low;
    (product, error)
}

/// The sign of the exact sum of `terms`. Shewchuk's grow-expansion keeps the sum exactly, as
/// nonoverlapping components of increasing magnitude, and the largest one decides.
fn exact_sign(terms: &[f64]) -> i8 {
    let mut expansion: Vec<f64> = Vec::new();
    for &term in terms {
        let mut carry = term;
        let mut grown = Vec::with_capacity(expansion.len() + 1);
        for &component in &expansion {
            let (sum, error) = two_sum(carry, component);
            if error != 0.0 {
                grown.push(error);
            }
            carry = sum;
        }
        if carry != 0.0 {
            grown.push(carry);
        }
        expansion = grown;
    }
    expansion
        .last()
        .map_or(0, |&c| (c > 0.0) as i8 - (c < 0.0) as i8)
}

/// Independent review of B19 (R2): `levelled` and `max_error` are residuals of this module's
/// floating-point evaluation, not certified bounds on the discrete minimax error E*. On 1e10 +
/// tanh(5x) at degree 2 the exact errors of the returned polynomial alternate on the reference
/// 3.7e-7 above `max_error`, so E* exceeds it (de la Vallée Poussin). E* still lies within the
/// rounding level `64 ε max|f|` of it, because every exact error does.
#[test]
fn the_bracket_is_a_floating_point_residual_not_a_certified_bound() {
    let samples = grid(-1.0, 1.0, 2001, |x| 1e10 + (5.0 * x).tanh());
    let fit = minimax(&samples, 2).unwrap();
    assert!(fit.levelled <= fit.max_error, "{fit:?}");
    let unique = prepare(&samples).unwrap();
    let [c0, c1, c2] = fit.chebyshev[..] else {
        panic!("{fit:?}")
    };
    // f - (c0 + c1 t + c2 (2 t² - 1)) as an exact sum of doubles. The map to t is exact on
    // [-1, 1], so these are the errors at the sample abscissae themselves.
    let error_terms = |(v, f): (f64, f64)| -> [f64; 9] {
        let t = fit.t(v);
        assert_eq!(t.to_bits(), v.to_bits());
        let (square, square_error) = two_product(t, t);
        let (linear, linear_error) = two_product(c1, t);
        let (high, high_error) = two_product(c2, square);
        let (low, low_error) = two_product(c2, square_error);
        [
            f,
            -c0,
            c2,
            -linear,
            -linear_error,
            -2.0 * high,
            -2.0 * high_error,
            -2.0 * low,
            -2.0 * low_error,
        ]
    };
    let signed = |terms: [f64; 9], sign: i8, extra: [f64; 2]| -> Vec<f64> {
        let sign = f64::from(sign);
        terms.iter().map(|x| sign * x).chain(extra).collect()
    };
    // On the reference the exact errors alternate, each more than 3e-7 above max_error.
    let mut previous = 0;
    for &r in &fit.reference {
        let terms = error_terms(unique[r]);
        let sign = exact_sign(&terms);
        assert!(sign != 0 && sign != previous, "{r}");
        let excess = signed(terms, sign, [-fit.max_error, -3e-7]);
        assert_eq!(exact_sign(&excess), 1, "{r}: {fit:?}");
        previous = sign;
    }
    // Every exact error, and so E* too, is within the rounding level of max_error.
    let f_max = unique.iter().map(|(_, f)| f.abs()).fold(0.0f64, f64::max);
    let noise = 64.0 * f64::EPSILON * f_max;
    for &sample in &unique {
        let terms = error_terms(sample);
        for sign in [1, -1] {
            let excess = signed(terms, sign, [-fit.max_error, -noise]);
            assert_eq!(exact_sign(&excess), -1, "{sample:?}");
        }
    }
}

/// Independent review of B19 (R3): Horner's error bound holds only when no intermediate
/// overflows. 1 + 1e308 v + 1e308 v² on [0.8, 0.9] was certified with a finite margin although
/// Horner overflows at 0.9 (0.9 · 1e308 + 1e308): the weighted sum Σ |q_j| M^j stays finite.
#[test]
fn the_certificate_refuses_a_horner_evaluation_that_can_overflow() {
    let q = [1.0, 1e308, 1e308];
    assert!(horner(&q, 0.9).is_infinite());
    assert_eq!(certify(&q, 0.8, 0.9, 1e-12, 8), Certificate::Refused);
    // Every coefficient below f64::MAX / 2, and a partial sum still overflows at 0.7.
    let q = [1.0, 8.9e307, 8.9e307, 8.9e307, 8.9e307];
    assert!(horner(&q, 0.7).is_infinite());
    assert_eq!(certify(&q, 0.6, 0.7, 1e-12, 8), Certificate::Refused);
    // Partial sums within f64::MAX / 2 are certified, with a finite margin.
    match certify(&[1.0, 4e307, 4e307], 0.8, 0.9, 1e-12, 8) {
        Certificate::Positive { margin, .. } => assert!(margin.is_finite(), "{margin}"),
        other => panic!("{other:?}"),
    }
    // An infinite end leaves M unbounded.
    assert_eq!(
        certify(&[1.0, 0.0, 1.0], f64::NEG_INFINITY, 0.0, 1e-12, 8),
        Certificate::Refused
    );
}
