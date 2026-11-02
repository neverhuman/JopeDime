use super::*;

#[test]
fn impute_fills_zeros_and_appends_indicators() {
    // Column 0 has zeros (missing); column 1 is fully observed.
    let x: Rows = vec![
        vec![0.0, 1.0],
        vec![1.0, 3.0],
        vec![2.0, 5.0],
        vec![9.0, 7.0],
    ];
    let y = vec![0.0; 4];

    // Median of the non-zero values [1, 2, 9] is 2.0.
    let op = fit_op(
        &stage("(impute strategy=median add_indicator=true)"),
        &x,
        &y,
    )
    .unwrap();
    match &op {
        FittedOp::Impute {
            fill,
            indicator_cols,
            ..
        } => {
            assert_eq!(fill[0], 2.0);
            assert_eq!(indicator_cols, &vec![0]);
        }
        other => panic!("wrong variant: {other:?}"),
    }
    let out = apply_op(&op, &x).unwrap();
    assert_eq!(out[0], vec![2.0, 1.0, 1.0]); // filled + indicator hot
    assert_eq!(out[1], vec![1.0, 3.0, 0.0]);
    assert_eq!(out[3], vec![9.0, 7.0, 0.0]);

    // Mean of [1, 2, 9] is 4.0.
    let op = fit_op(&stage("(impute strategy=mean)"), &x, &y).unwrap();
    let out = apply_op(&op, &x).unwrap();
    assert_eq!(out[0], vec![4.0, 1.0]); // no indicator without the flag
    assert_eq!(out[0].len(), 2);

    // Const fill uses the value literal.
    let op = fit_op(&stage("(impute strategy=const value=7.5)"), &x, &y).unwrap();
    let out = apply_op(&op, &x).unwrap();
    assert_eq!(out[0][0], 7.5);

    // An all-zero column keeps 0.0 (nothing observed to summarize).
    let xz: Rows = vec![vec![0.0, 1.0], vec![0.0, 2.0], vec![0.0, 3.0]];
    let op = fit_op(&stage("(impute strategy=mean)"), &xz, &y[..3]).unwrap();
    let out = apply_op(&op, &xz).unwrap();
    assert_eq!(out[0][0], 0.0);

    // Determinism: double fit is bit-identical.
    let s = stage("(impute strategy=median add_indicator=true)");
    let a = apply_op(&fit_op(&s, &x, &y).unwrap(), &x).unwrap();
    let b = apply_op(&fit_op(&s, &x, &y).unwrap(), &x).unwrap();
    assert_eq!(a, b);
}

#[test]
fn svd_project_is_deterministic_and_orthonormal() {
    let (x, y) = toy(120, 8);
    let s = stage("(svd_project k=4 iters=20 seed=7)");
    let op1 = fit_op(&s, &x, &y).unwrap();
    let op2 = fit_op(&s, &x, &y).unwrap();
    let comps = match &op1 {
        FittedOp::SvdProject { components, .. } => components.clone(),
        other => panic!("wrong variant: {other:?}"),
    };
    match &op2 {
        FittedOp::SvdProject { components, .. } => {
            for (a, b) in comps.iter().zip(components) {
                for (u, v) in a.iter().zip(b) {
                    assert_eq!(u.to_bits(), v.to_bits());
                }
            }
        }
        other => panic!("wrong variant: {other:?}"),
    }
    assert_eq!(comps.len(), 4);
    for i in 0..comps.len() {
        assert!(
            (dot(&comps[i], &comps[i]) - 1.0).abs() < 1e-9,
            "component {i} not unit"
        );
        for j in 0..i {
            assert!(
                dot(&comps[i], &comps[j]).abs() < 1e-6,
                "components {i},{j} not orthogonal"
            );
        }
        // Sign convention: largest-|value| coordinate is positive.
        let mut best = 0;
        for j in 1..comps[i].len() {
            if comps[i][j].abs() > comps[i][best].abs() {
                best = j;
            }
        }
        assert!(comps[i][best] > 0.0);
    }
    let out = apply_op(&op1, &x).unwrap();
    assert_eq!(out[0].len(), 4);
}

#[test]
fn nystrom_landmarks_are_deterministic_train_rows() {
    let (x, y) = toy(60, 5);
    let s = stage("(nystrom k=8 gamma=1.5 seed=9)");
    let op1 = fit_op(&s, &x, &y).unwrap();
    let op2 = fit_op(&s, &x, &y).unwrap();
    let l1 = match &op1 {
        FittedOp::Nystrom { landmarks, .. } => landmarks.clone(),
        other => panic!("wrong variant: {other:?}"),
    };
    let l2 = match &op2 {
        FittedOp::Nystrom { landmarks, .. } => landmarks.clone(),
        other => panic!("wrong variant: {other:?}"),
    };
    assert_eq!(l1, l2);
    assert_eq!(l1.len(), 8);
    for l in &l1 {
        assert!(x.iter().any(|r| r == l), "landmark is not a train row");
    }
    for i in 0..l1.len() {
        for j in 0..i {
            assert_ne!(l1[i], l1[j], "landmarks {i},{j} not distinct");
        }
    }
    let out = apply_op(&op1, &x).unwrap();
    assert_eq!(out[0].len(), 8);
    for v in out.iter().flatten() {
        assert!(v.is_finite() && *v > 0.0 && *v <= 1.0);
    }
}

#[test]
fn map_and_derive_round_trip_expressions() {
    let (x, y) = toy(40, 3);
    let m = stage("(map expr=\"sigmoid(col(0))\")");
    let op = fit_op(&m, &x, &y).unwrap();
    match &op {
        FittedOp::MapExpr { source, .. } => assert_eq!(source, "sigmoid(col(0))"),
        other => panic!("wrong variant: {other:?}"),
    }
    let out = apply_op(&op, &x).unwrap();
    assert_eq!(out[0].len(), 3);
    for (i, r) in x.iter().enumerate() {
        let want = 1.0 / (1.0 + (-r[0].clamp(-30.0, 30.0)).exp());
        for v in &out[i] {
            assert!((v - want).abs() < 1e-15);
        }
    }
    let d = stage("(derive expr=\"row_mean + col(1)\")");
    let opd = fit_op(&d, &x, &y).unwrap();
    let outd = apply_op(&opd, &x).unwrap();
    assert_eq!(outd[0].len(), 1);
    for (i, r) in x.iter().enumerate() {
        let want = r.iter().sum::<f64>() / r.len() as f64 + r[1];
        assert!((outd[i][0] - want).abs() < 1e-12);
    }
    // Parse failure and missing expr bail at fit time.
    assert!(fit_op(&stage("(map expr=\"nope(\")"), &x, &y).is_err());
    assert!(fit_op(&stage("(derive)"), &x, &y).is_err());
}

#[test]
fn scalar_batch_functions_evaluate_with_guarded_semantics() {
    let (x, y) = toy(30, 3);
    // Every new function parses, fits, and matches its reference formula.
    // A scalar op under test, paired with the reference formula it must reproduce.
    type ScalarCase = (&'static str, fn(f64) -> f64);
    let cases: [ScalarCase; 8] = [
        ("exp", |v| v.clamp(-30.0, 30.0).exp().min(1.0e12)),
        ("log", |v| if v <= 1e-300 { 0.0 } else { v.ln() }),
        ("asinh", asinh_stable),
        ("atan", |v| v.atan()),
        ("erf", |v| {
            let a = v.abs();
            let s = if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            };
            let t = 1.0 / (1.0 + 0.3275911 * a);
            let poly = t
                * (0.254829592
                    + t * (-0.284496736
                        + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
            s * (1.0 - poly * (-(a * a)).exp())
        }),
        ("sign", |v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        }),
        ("floor", |v| v.floor()),
        ("round", |v| {
            if v >= 0.0 {
                (v + 0.5).floor()
            } else {
                (v - 0.5).ceil()
            }
        }),
    ];
    for (name, reference) in cases {
        let m = stage(&format!("(map expr=\"{name}(col(0))\")"));
        let op = fit_op(&m, &x, &y).unwrap();
        let out = apply_op(&op, &x).unwrap();
        for (i, r) in x.iter().enumerate() {
            let want = reference(r[0]);
            assert_eq!(out[i][0], want, "{name}({}) mismatch", r[0]);
            assert!(out[i][0].is_finite(), "{name} produced non-finite");
        }
    }
    // Hand values: erf accuracy, round half-away, sign at zero.
    let probe = |src: &str, v: f64| -> f64 {
        let rows = vec![vec![v]];
        let op = fit_op(&stage(src), &rows, &[0.0]).unwrap();
        apply_op(&op, &rows).unwrap()[0][0]
    };
    assert!((probe("(map expr=\"erf(col(0))\")", 1.0) - 0.8427007929497149).abs() < 3e-7);
    assert!((probe("(map expr=\"erf(col(0))\")", -2.0) + 0.9953222650189527).abs() < 3e-7);
    assert_eq!(probe("(map expr=\"round(col(0))\")", 2.5), 3.0);
    assert_eq!(probe("(map expr=\"round(col(0))\")", -2.5), -3.0);
    assert_eq!(probe("(map expr=\"round(col(0))\")", 0.5), 1.0);
    assert_eq!(probe("(map expr=\"sign(col(0))\")", 0.0), 0.0);
    assert_eq!(probe("(map expr=\"floor(col(0))\")", -1.5), -2.0);
    assert_eq!(probe("(map expr=\"log(col(0))\")", -5.0), 0.0);
    // exp's clamp caps the argument at 30; the expression's final safe()
    // then caps exp(30) ~= 1.07e13 at the 1e12 output bound.
    assert_eq!(probe("(map expr=\"exp(col(0))\")", 100.0), 1.0e12);
}

#[test]
fn fft_mag_and_row_spectral_are_finite() {
    let (x, y) = toy(30, 9);
    let op = fit_op(&stage("(fft_mag k=4)"), &x, &y).unwrap();
    let out = apply_op(&op, &x).unwrap();
    assert_eq!(out[0].len(), 4);
    for v in out.iter().flatten() {
        assert!(v.is_finite());
    }
    // Default k=8 clamps to d/2 = 4 for 9 columns.
    let op = fit_op(&stage("(fft_mag)"), &x, &y).unwrap();
    assert_eq!(apply_op(&op, &x).unwrap()[0].len(), 4);
    let op = fit_op(&stage("(row_spectral)"), &x, &y).unwrap();
    let out = apply_op(&op, &x).unwrap();
    assert_eq!(out[0].len(), 5);
    for r in &out {
        for v in r {
            assert!(v.is_finite());
        }
        assert!(r[0] >= 0.0, "entropy negative: {}", r[0]);
        assert!(
            r[3] >= 1.0 && r[3] <= 4.0,
            "peak freq out of range: {}",
            r[3]
        );
        assert!(r[4] >= 0.0);
    }
}

#[test]
fn polar_lag_roll_shapes_and_values() {
    let (x, y) = tiny();
    let out = apply_op(&fit_op(&stage("(polar)"), &x, &y).unwrap(), &x).unwrap();
    assert_eq!(out[0].len(), 3);
    assert!((out[0][0] - 5.0).abs() < 1e-12);
    assert!((out[0][1] - 4.0f64.atan2(3.0)).abs() < 1e-15);
    assert_eq!(out[0][2], 0.5);
    let out = apply_op(&fit_op(&stage("(lag k=1)"), &x, &y).unwrap(), &x).unwrap();
    assert_eq!(out[0], vec![0.0, 3.0, 4.0]);
    let out = apply_op(&fit_op(&stage("(lag k=5)"), &x, &y).unwrap(), &x).unwrap();
    assert_eq!(out[1], vec![0.0, 0.0, 0.0]);
    let out = apply_op(&fit_op(&stage("(roll k=1)"), &x, &y).unwrap(), &x).unwrap();
    assert_eq!(out[0], vec![0.5, 3.0, 4.0]);
    // k = 4 ≡ 1 (mod 3).
    let out = apply_op(&fit_op(&stage("(roll k=4)"), &x, &y).unwrap(), &x).unwrap();
    assert_eq!(out[0], vec![0.5, 3.0, 4.0]);
    // append=true keeps the input on the left.
    let out = apply_op(
        &fit_op(&stage("(lag k=1 append=true)"), &x, &y).unwrap(),
        &x,
    )
    .unwrap();
    assert_eq!(out[0], vec![3.0, 4.0, 0.5, 0.0, 3.0, 4.0]);
}

#[test]
fn sigmoid_norm_row_and_diff_stats_values() {
    let (x, y) = tiny();
    let out = apply_op(&fit_op(&stage("(sigmoid)"), &x, &y).unwrap(), &x).unwrap();
    assert!((out[1][1] - 0.5).abs() < 1e-15, "sigmoid(0) != 0.5");
    assert!(out.iter().flatten().all(|v| *v > 0.0 && *v < 1.0));
    let out = apply_op(&fit_op(&stage("(norm_row)"), &x, &y).unwrap(), &x).unwrap();
    let n0 = out[0].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((n0 - 1.0).abs() < 1e-12);
    // Row [3,4,0.5]: diffs [1,-3.5] → mean -1.25, sign ratio 1/2, lag-1
    // autocorr of the original row = -1 (n=2 Pearson), energy 6.625.
    let out = apply_op(&fit_op(&stage("(row_diff_stats)"), &x, &y).unwrap(), &x).unwrap();
    assert_eq!(out[0].len(), 8);
    assert!((out[0][0] + 1.25).abs() < 1e-12);
    assert!((out[0][5] - 0.5).abs() < 1e-12);
    assert!((out[0][6] + 1.0).abs() < 1e-12);
    assert!((out[0][7] - 6.625).abs() < 1e-12);
}

/// Clamp-audit property test: near-constant columns (values `c ± 1e-16`)
/// make interpolated quantiles cross by one ulp, which used to panic
/// `f64::clamp` in winsor (min > max). Every registry op must fit+apply
/// on such data without panicking and with all-finite output.
#[test]
fn every_op_survives_near_constant_columns() {
    // One stage per fit_op arm (aliases included), default params; map and
    // derive carry the minimal expr they require.
    const STAGES: &[&str] = &[
        "(standardize)",
        "(norm_features)",
        "(robust_scale)",
        "(rank)",
        "(blom)",
        "(quantile_bins)",
        "(bin_onehot)",
        "(hinge)",
        "(winsor_clip)",
        "(winsor_moments)",
        "(cheb)",
        "(poly_interact)",
        "(rff_cos)",
        "(rff_sin)",
        "(rand_proj)",
        "(landmark_dist)",
        "(oblique_hinge)",
        "(row_stats)",
        "(pair_prod)",
        "(target_bin_encode)",
        "(sigmoid)",
        "(norm_row)",
        "(polar)",
        "(lag)",
        "(roll)",
        "(fft_mag)",
        "(row_spectral)",
        "(row_diff_stats)",
        "(svd_project)",
        "(nystrom)",
        "(map expr=\"sigmoid(col(0))\")",
        "(derive expr=\"row_mean + col(0)\")",
        "(symlog)",
        "(symlog c=0.5)",
        "(select_corr)",
        "(select_corr k=4 by=corr)",
        "(select_corr k=4 by=var)",
        "(impute)",
        "(impute strategy=median add_indicator=true)",
        "(impute strategy=mean)",
        "(impute strategy=const value=1.5)",
        "(pair_ratio)",
        "(pair_ratio topk=4)",
        "(spline)",
        "(spline knots=3)",
        "(spline knots=12)",
    ];
    let mut rng = Pcg32::new(20260702, 9);
    for trial in 0..50u32 {
        // Row counts where the ulp crossing PROVABLY happens for the
        // centers below (q=0.05: e.g. n=24 & sqrt2 reproduces the logged
        // `min = 1.4142135623730951, max = 1.414213562373095` panic),
        // plus random sizes.
        let rows = match trial % 5 {
            0 => 24,
            1 => 37,
            2 => 39,
            _ => 20 + (rng.next_u32() as usize) % 80,
        };
        let cols = 3 + (rng.next_u32() as usize) % 5;
        // Per-column plan: col 0 exactly constant [c; n], col 1
        // near-constant (c ± 1e-16 jitter), last col varying, the rest a
        // random mix. Centers include the exact panic-log values.
        let centers = [
            std::f64::consts::SQRT_2,
            1.0 / 3.0,
            0.333333343,
            -0.05798013915354207,
        ];
        #[derive(Clone, Copy)]
        enum ColPlan {
            Exact(f64),
            Jitter(f64),
            Vary,
        }
        let pick_center = |rng: &mut Pcg32, forced: usize| -> f64 {
            if rng.next_f64() < 0.5 {
                centers[forced % centers.len()]
            } else {
                rng.range_f64(-2.0, 2.0)
            }
        };
        let plan: Vec<ColPlan> = (0..cols)
            .map(|j| {
                if j == 0 {
                    ColPlan::Exact(centers[trial as usize % centers.len()])
                } else if j == 1 {
                    ColPlan::Jitter(pick_center(&mut rng, trial as usize + 1))
                } else if j + 1 == cols {
                    ColPlan::Vary
                } else {
                    match rng.next_u32() % 3 {
                        0 => ColPlan::Exact(pick_center(&mut rng, trial as usize + j)),
                        1 => ColPlan::Jitter(pick_center(&mut rng, trial as usize + j)),
                        _ => ColPlan::Vary,
                    }
                }
            })
            .collect();
        let mut x: Rows = Vec::with_capacity(rows);
        let mut y: Vec<f64> = Vec::with_capacity(rows);
        for _ in 0..rows {
            let r: Vec<f64> = plan
                .iter()
                .map(|p| match p {
                    ColPlan::Exact(c) => *c,
                    // ±1e-16 jitter: rounds to c or a 1-ulp neighbour
                    ColPlan::Jitter(c) => c + (rng.next_f64() - 0.5) * 2e-16,
                    ColPlan::Vary => rng.range_f64(-2.0, 2.0),
                })
                .collect();
            y.push(2.0 * r[cols - 1] - r[0] + 0.1 * rng.gaussian());
            x.push(r);
        }
        for src in STAGES {
            let op = fit_op(&stage(src), &x, &y)
                .unwrap_or_else(|e| panic!("trial {trial}: fit {src} failed: {e}"));
            let out = apply_op(&op, &x).unwrap();
            assert_eq!(out.len(), rows, "trial {trial}: {src} row count");
            for (i, row) in out.iter().enumerate() {
                assert!(!row.is_empty(), "trial {trial}: {src} empty row {i}");
                for v in row {
                    assert!(v.is_finite(), "trial {trial}: {src} non-finite output {v}");
                }
            }
        }
    }
}
