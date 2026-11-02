use super::*;

#[test]
fn lowering_uses_ordinary_nodes_within_the_published_budget() {
    let x = Expr::Var(Var::Col(0));
    let cases: [(&[f64], usize, usize); 4] = [
        (&[1.25, 0.5], 5, 3),
        (&[1.25, 0.5, 0.75], 9, 5),
        (&[1.25, 0.5, 0.75, 0.125], 13, 7),
        // Negative non-leading coefficients become Sub at no extra cost.
        (&[-1.25, -0.5, 0.75], 9, 5),
    ];
    for (coefficients, nodes, height) in cases {
        let tree = lower(coefficients, &x).unwrap();
        assert_eq!(count_nodes(&tree), nodes, "{coefficients:?}");
        assert_eq!(depth(&tree), height, "{coefficients:?}");
        for v in [-1.5, -0.25, 0.0, 0.6, 2.0] {
            let expected = coefficients.iter().rev().fold(0.0, |acc, c| acc * v + c);
            let (value, faults) = native(&tree, v);
            assert!(faults.is_empty());
            assert!(
                (value - expected).abs() <= 1e-14 * expected.abs().max(1.0),
                "{coefficients:?} at {v}"
            );
        }
    }
    // A negative leading coefficient is a literal, two nodes.
    let tree = lower(&[1.0, -2.0], &x).unwrap();
    assert_eq!(count_nodes(&tree), 6);
    assert!(lower(&[1.0, f64::NAN], &x).is_none());
    assert!(lower(&[], &x).is_none());
}

#[test]
fn a_one_variable_subtree_compresses_and_the_native_value_decides_outside_the_range() {
    let rows: Vec<Vec<f64>> = (0..201)
        .map(|i| vec![-1.0 + f64::from(i) / 100.0, 7.0])
        .collect();
    // 18 nodes, compressed to a cubic of at most 14.
    let subtree = MathExpr::parse("((exp(0.5*c0)+0.25)*(cos(0.3*c0)+1.5))+sin(0.2*c0)").unwrap();
    assert_eq!(count_nodes(&subtree.root), 18);
    let (var, points) = samples(&subtree, &rows).unwrap();
    assert_eq!(var, Var::Col(0));
    let fit = minimax(&points, 3).unwrap();
    let leaf = Expr::Var(var);
    let proposal = propose(&fit, &points, &leaf, &subtree.root, 64).unwrap();
    assert!(proposal.nodes <= 14, "{proposal:?}");
    let compressed = MathExpr::from_ast(proposal.tree.clone()).unwrap();
    // On the training range the native lowered tree is within its own measured error, which
    // is within the fit's bound.
    for row in &rows {
        let (a, faults) = compressed.eval_row_traced(row);
        let (b, _) = subtree.eval_row_traced(row);
        assert!(faults.is_empty());
        assert!((a - b).abs() <= proposal.lowered_error, "{row:?}");
    }
    assert!(proposal.lowered_error <= fit.max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * 20.0);
    // Outside it the polynomial drifts: only a native re-score on held-out rows can tell.
    let (a, _) = compressed.eval_row_traced(&[4.0, 7.0]);
    let (b, _) = subtree.eval_row_traced(&[4.0, 7.0]);
    assert!((a - b).abs() > 100.0 * fit.max_error, "{a} vs {b}");
    // A subtree no smaller than the lowered tree is never proposed (the 6-node
    // exp(0.5*c0)+0.25 against a cubic), and neither is a tree over the node budget.
    let small = MathExpr::parse("exp(0.5*c0)+0.25").unwrap();
    let (_, small_points) = samples(&small, &rows).unwrap();
    let small_fit = minimax(&small_points, 3).unwrap();
    assert!(propose(&small_fit, &small_points, &leaf, &small.root, 64).is_none());
    assert!(propose(&fit, &points, &leaf, &subtree.root, 12).is_none());
    // Subtrees reading two variables, or faulting on a row, are not sampled.
    assert!(samples(&MathExpr::parse("c0*c1").unwrap(), &rows).is_none());
    assert!(samples(&MathExpr::parse("log(c0)").unwrap(), &rows).is_none());
    assert!(samples(&MathExpr::parse("3.0").unwrap(), &rows).is_none());
}

#[test]
fn the_bernstein_certificate_proves_only_pole_free_denominators() {
    // 1 + 0.5 v² on [0, 1]: at least 1.
    assert!(matches!(
        certify(&[1.0, 0.0, 0.5], 0.0, 1.0, 1e-12, 4),
        Certificate::Positive { .. }
    ));
    // A root inside the interval: v - 0.3 and v² - 0.25 on [0, 1].
    assert_eq!(
        certify(&[-0.3, 1.0], 0.0, 1.0, 1e-12, 6),
        Certificate::Refused
    );
    assert_eq!(
        certify(&[-0.25, 0.0, 1.0], 0.0, 1.0, 1e-12, 6),
        Certificate::Refused
    );
    // A minimum of 5e-13 sits below the protected-division threshold.
    assert_eq!(
        certify(&[5e-13 + 0.25, -1.0, 1.0], 0.0, 1.0, 1e-12, 8),
        Certificate::Refused
    );
    // A negative denominator is certified as such, and subdivision tightens the enclosure.
    assert!(matches!(
        certify(&[-2.0, 0.0, 1.0], -1.0, 1.0, 1e-12, 4),
        Certificate::Negative { .. }
    ));
    match certify(&[0.3, -1.0, 1.0], 0.0, 1.0, 1e-12, 8) {
        Certificate::Positive { depth, .. } => assert!(depth >= 1),
        other => panic!("{other:?}"),
    }
    assert_eq!(certify(&[1.0], 1.0, 1.0, 1e-12, 4), Certificate::Refused);
    assert_eq!(
        certify(&[f64::NAN], 0.0, 1.0, 1e-12, 4),
        Certificate::Refused
    );
}

#[test]
fn interval_operations_enclose_their_exact_results() {
    let a = Interval::point(0.1);
    let b = Interval::point(0.2);
    let sum = a.add(b);
    assert!(sum.lo < 0.30000000000000004 && sum.hi > 0.3);
    let product = Interval { lo: -1.0, hi: 2.0 }.mul(Interval { lo: -3.0, hi: 0.5 });
    assert!(product.lo <= -6.0 && product.hi >= 3.0);
    let halves = de_casteljau_halves(&[Interval::point(1.0), Interval::point(3.0)]);
    assert!(halves.0[1].lo <= 2.0 && halves.0[1].hi >= 2.0);
    assert_eq!(halves.1[1], Interval::point(3.0));
}

#[test]
fn an_off_centre_lowering_is_held_to_its_own_error() {
    let f = |x: f64| (0.37 * x).sin() + (x / 50.0).cos();
    let leaf = Expr::Var(Var::Col(0));
    // A 41-node stand-in for the replaced subtree, so that only the error gate can refuse.
    let big = (0..20).fold(leaf.clone(), |acc, _| {
        Expr::Binary(BinOp::Add, Box::new(acc), Box::new(leaf.clone()))
    });
    // Centred, the lowered tree keeps the fit's error.
    let centred = grid(-1.0, 1.0, 2001, f);
    let fit = minimax(&centred, 6).unwrap();
    let error = lowered_error(&fit.monomial(), &prepare(&centred).unwrap()).unwrap();
    assert!(
        error <= fit.max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * 2.0,
        "{error} vs {}",
        fit.max_error
    );
    assert!(propose(&fit, &centred, &leaf, &big, 64).is_some());
    // Off centre the monomial conversion loses the fit, so nothing is proposed.
    let shifted = grid(1000.0, 1001.0, 2001, f);
    let fit = minimax(&shifted, 6).unwrap();
    let error =
        lowered_error(&fit.monomial(), &prepare(&shifted).unwrap()).unwrap_or(f64::INFINITY);
    assert!(error > 1e3 * fit.max_error, "{error} vs {}", fit.max_error);
    assert!(propose(&fit, &shifted, &leaf, &big, 64).is_none());
}

#[test]
fn signed_zero_abscissae_with_signed_zero_targets_are_one_sample() {
    // Odd subtrees give f(-0.0) = -0.0 and f(0.0) = 0.0 (0.5*c0, sin(c0), c0^3).
    let points0 = [
        (-0.0, -0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (-1.0, -1.0),
    ];
    let fit = minimax(&points0, 1).unwrap();
    assert_eq!(fit.stop, Stop::ExactFit);
    assert_eq!(prepare(&points0).unwrap().len(), 4);
    let mut reversed = points0.to_vec();
    reversed.reverse();
    let bits = |m: &Minimax| m.chebyshev.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&minimax(&reversed, 1).unwrap()), bits(&fit));
    // Through the sampler: a column holding both zeros.
    let rows: Vec<Vec<f64>> = [-1.0, -0.0, 0.0, 0.5, 1.0]
        .iter()
        .map(|&v| vec![v])
        .collect();
    let (_, points) = samples(&MathExpr::parse("0.5*c0").unwrap(), &rows).unwrap();
    assert!(minimax(&points, 1).is_ok());
    // A genuine conflict at zero is still refused.
    assert!(minimax(&[(0.0, 1.0), (-0.0, -1.0), (1.0, 2.0), (2.0, 3.0)], 1).is_err());
}

#[test]
fn every_stop_path_is_reached_and_the_cap_returns_the_best_fit() {
    // Abscissae 2^i with targets +1, -1, -1 repeating: exact data, badly conditioned in t.
    let geometric = |points: usize| -> Vec<(f64, f64)> {
        (0..points)
            .map(|i| (f64::from(1u32 << i), if i % 3 == 0 { 1.0 } else { -1.0 }))
            .collect()
    };
    // The exchange cycles between two references from step 3 on, with max errors above
    // 1e15, so the cap must return step 2's fit, not the last one.
    let capped = minimax(&geometric(15), 11).unwrap();
    assert_eq!(
        (capped.stop, capped.iterations),
        (Stop::IterationLimit, 2),
        "{capped:?}"
    );
    assert!(capped.max_error < 2.0, "{capped:?}");
    assert!(capped.levelled <= capped.max_error);
    // |x| at 15 points, degree 12: a two-step cycle a few 1e-12 above the bracket; the best
    // fit is step 9's.
    let abs = minimax(&grid(-1.0, 1.0, 15, f64::abs), 12).unwrap();
    assert_eq!(
        (abs.stop, abs.iterations),
        (Stop::IterationLimit, 9),
        "{abs:?}"
    );
    assert!(abs.levelled <= abs.max_error);
    // The exchange returns the same reference.
    let stalled = minimax(&geometric(8), 6).unwrap();
    assert_eq!(stalled.stop, Stop::Stalled, "{stalled:?}");
    assert!(stalled.levelled <= stalled.max_error);
    // Fewer than n + 2 alternating extrema, with the global maximum already a reference
    // point.
    let no_exchange = minimax(&geometric(10), 8).unwrap();
    assert_eq!(no_exchange.stop, Stop::Degenerate, "{no_exchange:?}");
    // The second reference system is singular: the best (first) fit comes back.
    let singular = minimax(&geometric(27), 3).unwrap();
    assert_eq!(
        (singular.stop, singular.iterations),
        (Stop::Degenerate, 1),
        "{singular:?}"
    );
}

#[test]
fn an_offset_noise_floor_stop_is_not_reported_as_converged() {
    // A constant offset lifts max|f|, and with it the rounding level of the errors, far above
    // 1e-12 of the minimax error.
    let offset = minimax(&grid(-1.0, 1.0, 4001, |x| 1e6 + x.exp()), 2).unwrap();
    assert_eq!(offset.stop, Stop::NoiseLimited, "{offset:?}");
    assert!(offset.max_error - offset.levelled > TOLERANCE * offset.max_error);
    assert!(offset.max_error - offset.levelled <= 64.0 * f64::EPSILON * (1e6 + 1f64.exp()));
    // Without the offset the same fit closes the documented bracket.
    let plain = minimax(&grid(-1.0, 1.0, 4001, f64::exp), 2).unwrap();
    assert_eq!(plain.stop, Stop::Converged, "{plain:?}");
    assert!(plain.max_error - plain.levelled <= TOLERANCE * plain.max_error);
}

/// Lane D pre-review of B19: a noise-floor stop came while the exchange could still raise |h|.
/// On 1e3 + cos(0.7x)x at degree 9 it stopped at step 5 with max|e| = 1.88e-11, where step 8
/// reaches 7.73e-12; on 1e6 + cos(0.7x)x at degree 7 it stopped at step 1 with |h| ≈ 1e-17
/// against max|e| = 1.09e-8, where step 5 reaches 5.59e-9.
#[test]
fn a_noise_floor_stop_waits_until_the_exchange_cannot_raise_h() {
    let fit = minimax(&grid(-1.0, 1.0, 2001, |x| 1e3 + (0.7 * x).cos() * x), 9).unwrap();
    assert!(fit.max_error <= 1.05 * 7.731e-12, "{fit:?}");
    let fit = minimax(&grid(-1.0, 1.0, 2001, |x| 1e6 + (0.7 * x).cos() * x), 7).unwrap();
    assert!(fit.levelled >= fit.max_error / 2.0, "{fit:?}");
    assert!(fit.max_error <= 1.05 * 5.588e-9, "{fit:?}");
}

/// Lane D pre-review of B19: rounding can put |h| above max|e|, and such inverted brackets were
/// reported as Converged (1e10 + tanh(5x) by 3.7e-7, 7.7e5 times the tolerance). `levelled` now
/// never exceeds `max_error`, and an |h| above it by more than the tolerance is not Converged.
#[test]
fn no_reported_bracket_inverts_and_an_inverted_one_is_not_converged() {
    for (samples, n) in [
        (grid(-1.0, 1.0, 2001, |x| 1e10 + (5.0 * x).tanh()), 2),
        (grid(-1.0, 1.0, 501, |x| 10.0 + (2.0 + x).ln()), 8),
        (
            grid(-1.0, 1.0, 1001, |x| 3.0 + ((2.0 * x).sin() + 0.3 * x)),
            10,
        ),
    ] {
        let fit = minimax(&samples, n).unwrap();
        assert_ne!(fit.stop, Stop::Converged, "n={n}: {fit:?}");
        assert!(fit.levelled <= fit.max_error, "n={n}: {fit:?}");
    }
    let functions: [fn(f64) -> f64; 3] = [f64::exp, f64::sin, |x| (5.0 * x).tanh()];
    for f in functions {
        for offset in [0.0, 10.0, 1e3, 1e6, 1e10] {
            let samples = grid(-1.0, 1.0, 501, |x| offset + f(x));
            for n in 1..=10 {
                let fit = minimax(&samples, n).unwrap();
                assert!(fit.levelled <= fit.max_error, "{offset} n={n}: {fit:?}");
                if fit.stop == Stop::Converged {
                    assert!(fit.max_error - fit.levelled <= TOLERANCE * fit.max_error);
                }
            }
        }
    }
}

/// Lane D pre-review of B19: near `f64::MAX` the map to t and the Clenshaw sums overflowed, and
/// an infinite max|e| passed the Converged test. A sample outside the evaluator's range is
/// refused by name; the range itself is accepted.
#[test]
fn samples_outside_the_evaluator_range_are_refused() {
    let refused = "a sample is outside the evaluator's range";
    let wide = |amplitude: f64, k: f64| grid(-1.0, 1.0, 101, move |x| amplitude * (k * x).sin());
    assert_eq!(minimax(&wide(1e308, 3.0), 3).unwrap_err(), refused);
    assert_eq!(minimax(&wide(5e307, 6.0), 6).unwrap_err(), refused);
    let near: Vec<(f64, f64)> = (0..10)
        .map(|i| (1e308 + f64::from(i) * 7e306, f64::from(i)))
        .collect();
    assert_eq!(minimax(&near, 1).unwrap_err(), refused);
    let edge = minimax(&grid(-MAX_ABS, MAX_ABS, 101, |x| 0.5 * x), 1).unwrap();
    assert_eq!(edge.stop, Stop::ExactFit);
}

/// Lane D pre-review of B19: only the expression limits stop a degree-8 lowering (depth 17 or
/// 18) from being proposed, and no test pinned that boundary. And `x` must be the sampled
/// variable: the error is measured at the sampled abscissae.
#[test]
fn the_depth_limit_and_the_leaf_bound_what_is_proposed() {
    let f = |x: f64| (0.37 * x).sin() + (x / 50.0).cos();
    let leaf = Expr::Var(Var::Col(0));
    let big = (0..20).fold(leaf.clone(), |acc, _| {
        Expr::Binary(BinOp::Add, Box::new(acc), Box::new(leaf.clone()))
    });
    let samples = grid(-1.0, 1.0, 2001, f);
    let seven = propose(&minimax(&samples, 7).unwrap(), &samples, &leaf, &big, 64).unwrap();
    assert!(depth(&seven.tree) <= 16, "{seven:?}");
    assert!(propose(&minimax(&samples, 8).unwrap(), &samples, &leaf, &big, 64).is_none());
    let cubic = minimax(&samples, 3).unwrap();
    assert!(propose(&cubic, &samples, &leaf, &big, 64).is_some());
    let square = Expr::Binary(BinOp::Mul, Box::new(leaf.clone()), Box::new(leaf));
    assert!(propose(&cubic, &samples, &square, &big, 64).is_none());
}
