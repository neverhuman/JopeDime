use super::*;

#[test]
fn exp_on_the_unit_interval_matches_the_published_minimax_errors() {
    // Atkinson, An Introduction to Numerical Analysis, §4.4: ρ_n(e^x) on [-1, 1], n = 1..9.
    let published = [
        2.79e-1, 4.50e-2, 5.53e-3, 5.47e-4, 4.52e-5, 3.21e-6, 2.00e-7, 1.11e-8, 5.52e-10,
    ];
    let samples = grid(-1.0, 1.0, 4001, f64::exp);
    for (n, rho) in published.iter().enumerate().map(|(i, r)| (i + 1, *r)) {
        let fit = minimax(&samples, n).unwrap();
        assert!(fit.levelled <= fit.max_error, "n={n}: {fit:?}");
        // Three printed significant digits: half a unit in the last place is at most 5e-3
        // relative. The discrete minimax can only be at or below the continuous one.
        assert!(
            close(fit.max_error, rho, 5e-3),
            "n={n}: {} vs {rho}",
            fit.max_error
        );
        assert!(
            matches!(
                fit.stop,
                Stop::Converged | Stop::NoiseLimited | Stop::Stalled
            ),
            "n={n}: {:?}",
            fit.stop
        );
        if fit.stop == Stop::Converged {
            assert!(fit.max_error - fit.levelled <= TOLERANCE * fit.max_error);
        }
        assert!(fit.iterations <= 20, "n={n}: {} steps", fit.iterations);
    }
}

#[test]
fn the_linear_fit_is_the_closed_form_when_the_alternation_point_is_sampled() {
    // a1 = sinh 1, x* = ln(sinh 1), a0 = (e - sinh(1) x*) / 2, ρ1 = (1/e + sinh(1) x*) / 2.
    let s = 1f64.sinh();
    let star = s.ln();
    let a0 = (1f64.exp() - s * star) / 2.0;
    let rho = ((-1f64).exp() + s * star) / 2.0;
    let mut samples = grid(-1.0, 1.0, 801, f64::exp);
    samples.push((star, star.exp()));
    let fit = minimax(&samples, 1).unwrap();
    let monomial = fit.monomial();
    assert!(close(monomial[1], s, 1e-14), "{monomial:?}");
    assert!(close(monomial[0], a0, 1e-14), "{monomial:?}");
    assert!(
        close(fit.max_error, rho, 1e-14),
        "{} vs {rho}",
        fit.max_error
    );
    assert!(close(fit.levelled, rho, 1e-13));
}

#[test]
fn quadratic_and_cubic_coefficients_match_their_published_digits() {
    let samples = grid(-1.0, 1.0, 4001, f64::exp);
    let quadratic = minimax(&samples, 2).unwrap().monomial();
    for (a, b) in quadratic.iter().zip([0.9890, 1.1302, 0.5540]) {
        assert!((a - b).abs() <= 1e-4, "{quadratic:?}");
    }
    let cubic = minimax(&samples, 3).unwrap().monomial();
    for (a, b) in cubic.iter().zip([0.994579, 0.995668, 0.542973, 0.179533]) {
        assert!((a - b).abs() <= 2e-6, "{cubic:?}");
    }
}

#[test]
fn higher_degree_fixtures_from_sollya_and_chebfun() {
    // Sollya: remez(exp(x), 5, [0;1]) has error 1.129569802274787e-6.
    let fit = minimax(&grid(0.0, 1.0, 4001, f64::exp), 5).unwrap();
    assert!(
        close(fit.max_error, 1.129569802274787e-6, 1e-3),
        "{}",
        fit.max_error
    );
    // Pachón and Trefethen, Table 1, degree 10 on [-1, 1].
    let f1 = |x: f64| (x + 0.5).tanh() - (x - 0.5).tanh();
    let fit = minimax(&grid(-1.0, 1.0, 8001, f1), 10).unwrap();
    assert!(
        close(fit.max_error, 0.000_000_300_091_95, 1e-3),
        "{}",
        fit.max_error
    );
    let f2 = |x: f64| x.exp().sin();
    let fit = minimax(&grid(-1.0, 1.0, 8001, f2), 10).unwrap();
    assert!(
        close(fit.max_error, 0.000_001_786_234_00, 1e-3),
        "{}",
        fit.max_error
    );
}

#[test]
fn the_final_reference_alternates_at_the_levelled_error() {
    let samples = grid(-1.0, 1.0, 2001, |x| (2.0 * x).sin() + 0.3 * x);
    let fit = minimax(&samples, 4).unwrap();
    let unique = prepare(&samples).unwrap();
    let errors: Vec<f64> = fit
        .reference
        .iter()
        .map(|&r| unique[r].1 - fit.eval(unique[r].0))
        .collect();
    for pair in errors.windows(2) {
        assert!(pair[0] * pair[1] < 0.0, "{errors:?}");
    }
    for e in &errors {
        assert!(
            close(e.abs(), fit.levelled, 1e-9),
            "{errors:?} vs {}",
            fit.levelled
        );
    }
}

#[test]
fn fits_ignore_sample_order_and_duplicates() {
    let samples = grid(-2.0, 3.0, 701, |x| (0.7 * x).cos() * x);
    let reference = minimax(&samples, 5).unwrap();
    let mut rng = Pcg32::new(0xB19, 1);
    for _ in 0..4 {
        let mut shuffled = samples.clone();
        shuffled.extend(samples.iter().step_by(7).copied());
        for i in (1..shuffled.len()).rev() {
            let j = rng.next_u32() as usize % (i + 1);
            shuffled.swap(i, j);
        }
        let fit = minimax(&shuffled, 5).unwrap();
        let bits = |m: &Minimax| m.chebyshev.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&fit), bits(&reference));
        assert_eq!(fit.reference, reference.reference);
        assert_eq!(fit.max_error.to_bits(), reference.max_error.to_bits());
    }
}

#[test]
fn exact_polynomials_and_refusals() {
    let cubic = |x: f64| 0.5 - 1.25 * x + 0.75 * x * x * x;
    let fit = minimax(&grid(-1.0, 1.0, 101, cubic), 3).unwrap();
    assert_eq!(fit.stop, Stop::ExactFit);
    assert!(fit.max_error <= 1e-15);
    assert_eq!(
        minimax(&grid(0.0, 1.0, 4, f64::exp), 3).unwrap_err(),
        "fewer than degree + 2 distinct abscissae"
    );
    assert!(minimax(&[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 1.0)], 1).is_err());
    // A non-finite target is refused by name, also where it would sit off the reference: the
    // exchange cannot see a NaN error, and an infinite one makes any fit look exact.
    assert_eq!(
        minimax(&[(1.0, f64::NAN), (2.0, 1.0), (3.0, 1.0)], 1).unwrap_err(),
        "a sample is not finite"
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut interior = grid(0.0, 1.0, 50, f64::exp);
        interior[25].1 = bad;
        assert_eq!(
            minimax(&interior, 2).unwrap_err(),
            "a sample is not finite",
            "{bad}"
        );
    }
    assert!(minimax(&grid(0.0, 1.0, 100, f64::exp), MAX_DEGREE + 1).is_err());
    let one_point: Vec<(f64, f64)> = (0..10).map(|_| (2.0, 1.0)).collect();
    assert!(minimax(&one_point, 1).is_err());
}
