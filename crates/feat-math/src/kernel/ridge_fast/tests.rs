use super::*;
use crate::heads::{cholesky_solve, RidgeModel};
use crate::Pcg32;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Bits, with every NaN mapped to one value: NaN positions are kept, while a NaN result's sign
/// and payload, which Rust leaves unspecified, are not compared.
fn canonical(values: &[f64]) -> Vec<u64> {
    values
        .iter()
        .map(|v| if v.is_nan() { u64::MAX } else { v.to_bits() })
        .collect()
}

/// The oracle: `RidgeModel::fit` and `std_fit` as they stood before B11 (main e043aebfd),
/// verbatim apart from names: per-column strided moments and one rank-one update of the
/// whole Gram triangle per row.
fn legacy_fit(x: &Rows, y: &[f64], lambda_frac: f64) -> RidgeModel {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = {
        let n = x.len().max(1) as f64;
        let mut means = vec![0.0; p];
        let mut scales = vec![1.0; p];
        for j in 0..p {
            let mut s = 0.0;
            for r in x {
                s += r[j];
            }
            means[j] = s / n;
            let mut v = 0.0;
            for r in x {
                let d = r[j] - means[j];
                v += d * d;
            }
            scales[j] = (v / n).sqrt().max(1e-9);
        }
        (means, scales)
    };
    let lambda = (lambda_frac * n as f64).max(1e-3);
    let y_mean = y[..n].iter().sum::<f64>() / n.max(1) as f64;
    let mut ata = vec![0.0f64; p * p];
    let mut atb = vec![0.0f64; p];
    let mut xs_row = vec![0.0f64; p];
    for r in 0..n {
        for j in 0..p {
            xs_row[j] = ((x[r][j] - means[j]) / scales[j]).clamp(-20.0, 20.0);
        }
        let yr = y[r] - y_mean;
        for i in 0..p {
            let xi = xs_row[i];
            atb[i] += xi * yr;
            for j in i..p {
                ata[i * p + j] += xi * xs_row[j];
            }
        }
    }
    for i in 0..p {
        ata[i * p + i] += lambda;
        for j in 0..i {
            ata[i * p + j] = ata[j * p + i];
        }
    }
    let coeffs = cholesky_solve(&mut ata, &mut atb, p);
    let mut w = Vec::with_capacity(p + 1);
    w.push(y_mean);
    w.extend(coeffs);
    RidgeModel { means, scales, w }
}

fn assert_identical(x: &Rows, y: &[f64], lambda_frac: f64) {
    let legacy = legacy_fit(x, y, lambda_frac);
    let fast = RidgeModel::fit(x, y, lambda_frac);
    assert_eq!(bits(&std_fit(x).0), bits(&legacy.means), "std_fit means");
    assert_eq!(bits(&std_fit(x).1), bits(&legacy.scales), "std_fit scales");
    assert_eq!(bits(&fast.means), bits(&legacy.means), "means");
    assert_eq!(bits(&fast.scales), bits(&legacy.scales), "scales");
    assert_eq!(bits(&fast.w), bits(&legacy.w), "weights");
    let probe: Rows = x.iter().rev().take(7).cloned().collect();
    assert_eq!(bits(&fast.predict(&probe)), bits(&legacy.predict(&probe)));
}

/// Gaussian columns mixed with the hard kinds: constant, heavy-tailed, near-duplicate,
/// integer-valued and signed zeros. A finite column standardizes to |z| ≤ √(n − 1), so designs
/// of at most 300 rows stay inside the ±20 clamp. The 500-row design of
/// `a_design_that_binds_the_clamp_matches_the_legacy_fit` passes it, and so do the ignored
/// `kernel_benchmark`'s 3333-row designs.
fn design(seed: u64, n: usize, p: usize) -> (Rows, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 11);
    let x: Rows = (0..n)
        .map(|r| {
            let base = rng.gaussian();
            (0..p)
                .map(|j| match j % 6 {
                    0 => 3.5,
                    1 => (3.0 * rng.gaussian()).exp(),
                    2 => base,
                    3 => base + 1e-12 * rng.gaussian(),
                    4 => f64::from((r % 7) as u32) - 3.0,
                    _ => {
                        if r % 3 == 0 {
                            -0.0
                        } else {
                            rng.gaussian()
                        }
                    }
                })
                .collect()
        })
        .collect();
    let y = x
        .iter()
        .map(|row| row.iter().take(4).sum::<f64>() * 0.3 + rng.gaussian())
        .collect();
    (x, y)
}

#[test]
fn fast_fit_is_the_legacy_fit_bit_for_bit() {
    for (seed, n, p) in [
        (1, 1, 1),
        (2, 2, 3),
        (3, 7, 5),
        (4, 64, 17),
        (5, 300, 48),
        (6, 64, 200),
        (7, 128, 131),
    ] {
        let (x, y) = design(seed, n, p);
        for lambda_frac in [0.02, 0.2, 1e-9] {
            assert_identical(&x, &y, lambda_frac);
        }
    }
}

#[test]
fn a_design_that_binds_the_clamp_matches_the_legacy_fit() {
    // A value that dominates its column standardizes to nearly ±√(n − 1), Samuelson's bound:
    // ≈ 22.3 for 500 rows, past the ±20 clamp, which both fits then apply. One positive and
    // one negative plant bind both bounds.
    let (mut x, y) = design(8, 500, 13);
    x[0][1] = 1e12;
    x[1][7] = -1e12;
    let (means, scales) = std_fit(&x);
    let high = (x[0][1] - means[1]) / scales[1];
    let low = (x[1][7] - means[7]) / scales[7];
    assert!(
        high > 20.0,
        "the positive plant must pass the clamp, z = {high}"
    );
    assert!(
        low < -20.0,
        "the negative plant must pass the clamp, z = {low}"
    );
    for lambda_frac in [0.02, 0.2, 1e-9] {
        assert_identical(&x, &y, lambda_frac);
    }
    // The certified design standardizes through the same clamp: its factor solves to the
    // fit's weights bit for bit, which fails if either side drops either bound. PRESS still
    // matches refits.
    let d = Design::from_rows(&x, &y, 0.02).unwrap();
    let factor = Factor::new(&d).unwrap();
    let fit = RidgeModel::fit(&x, &y, 0.02);
    assert_eq!(bits(&factor.coefficients()), bits(&fit.w[1..]));
    let loo = leave_one_out(&d, &factor).unwrap();
    let reference = brute_force_leave_one_out(&d).unwrap();
    let scale = reference.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    for (fast, slow) in loo.residuals.iter().zip(&reference) {
        assert!((fast - slow).abs() <= 1e-10 * scale);
    }
}

#[test]
fn more_rows_than_labels_and_short_label_sets_match() {
    // The moments use every row of `x`; the normal equations only the first min(x, y) rows.
    let (x, y) = design(8, 40, 9);
    assert_identical(&x, &y[..25], 0.02);
    let (x, y) = design(9, 12, 6);
    let mut longer = y.clone();
    longer.extend([5.0, -5.0]);
    assert_identical(&x, &longer, 0.02);
}

/// Kernel micro-benchmark, timing only: `cargo test -p feat-math --release --lib
/// ridge_fast::tests::kernel_benchmark -- --ignored --nocapture`.
#[test]
#[ignore]
fn kernel_benchmark() {
    use std::time::Instant;
    for p in [16, 64, 256, 512] {
        let (x, y) = design(42, 3333, p);
        assert_identical(&x, &y, 0.02);
        let repeats = (2_000_000 / (p * p)).clamp(3, 200);
        let time = |f: &dyn Fn() -> RidgeModel| {
            let started = Instant::now();
            for _ in 0..repeats {
                std::hint::black_box(f());
            }
            started.elapsed().as_secs_f64() / repeats as f64
        };
        let legacy = time(&|| legacy_fit(&x, &y, 0.02));
        let fast = time(&|| RidgeModel::fit(&x, &y, 0.02));
        println!(
            "B11_KERNEL n=3333 p={p} repeats={repeats} legacy_ms={:.3} fast_ms={:.3} speedup={:.2}",
            legacy * 1e3,
            fast * 1e3,
            legacy / fast
        );
    }
}

#[test]
fn the_factor_reproduces_the_legacy_solve_and_resumes_bit_for_bit() {
    for (seed, n, p) in [(21, 7, 5), (22, 64, 17), (23, 300, 48), (24, 64, 131)] {
        let (x, y) = design(seed, n, p);
        let legacy = RidgeModel::fit(&x, &y, 0.02);
        let d = Design::from_rows(&x, &y, 0.02).unwrap();
        let full = Factor::new(&d).unwrap();
        assert_eq!(bits(&full.coefficients()), bits(&legacy.w[1..]));
        for p_a in [0, 1, p / 2, p - 1, p] {
            let prefix = Factor::new(&d.leading(p_a)).unwrap();
            assert_eq!(Factor::extend(&prefix, &d).unwrap(), full, "p_a={p_a}");
            // Two steps through an intermediate width give the same factor.
            let middle = Factor::extend(&prefix, &d.leading((p_a + p) / 2)).unwrap();
            assert_eq!(Factor::extend(&middle, &d).unwrap(), full);
        }
    }
}

/// Gaussian standardized columns with labels driven by the first ones.
fn columns(seed: u64, n: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 13);
    let z: Vec<f64> = (0..n * p).map(|_| rng.gaussian()).collect();
    let y = (0..n)
        .map(|r| z[r] * 1.5 - z[n + r] * 0.5 + 0.3 * rng.gaussian())
        .collect();
    (z, y)
}

#[test]
fn schur_gain_is_the_objective_decrease_of_an_appended_block() {
    for (seed, n, p_a, p_b, lambda) in [
        (31, 50, 3, 1, 1.0),
        (32, 200, 20, 5, 4.0),
        (33, 200, 40, 24, 1e-3),
        (34, 50, 0, 6, 1.0),
    ] {
        let p = p_a + p_b;
        let (z, y) = columns(seed, n, p);
        let d = Design::from_columns(z, n, p, &y, lambda).unwrap();
        let full = Factor::new(&d).unwrap();
        let lead = Factor::new(&d.leading(p_a)).unwrap();
        let gain = full.schur_gain(p_a);
        assert!(gain >= 0.0);
        // Both objectives from explicit coefficients and residuals, independently of the
        // forward vector the gain is read from.
        let explicit = |factor: &Factor| {
            let w = factor.coefficients();
            let mut value = lambda * dot(&w, &w);
            for r in 0..n {
                let fitted: f64 = (0..w.len()).map(|j| d.z[j * n + r] * w[j]).sum();
                value += (d.centered[r] - fitted).powi(2);
            }
            value
        };
        let (lead_value, full_value) = (explicit(&lead), explicit(&full));
        let decrease = lead_value - full_value;
        assert!(
            (decrease - gain).abs() <= 1e-10 * lead_value,
            "seed {seed}: {decrease} vs {gain}"
        );
        assert!((full_value - full.objective(&d)).abs() <= 1e-10 * full_value);
        assert!((lead_value - lead.objective(&d)).abs() <= 1e-10 * lead_value);
    }
    // An all-zero appended block gains exactly nothing.
    let (mut z, y) = columns(35, 40, 3);
    z.extend(std::iter::repeat_n(0.0, 40 * 2));
    let d = Design::from_columns(z, 40, 5, &y, 0.8).unwrap();
    assert_eq!(Factor::new(&d).unwrap().schur_gain(3), 0.0);
}

#[test]
fn press_matches_brute_force_leave_one_out() {
    for (seed, n, p, lambda) in [
        (41, 12, 3, 0.24),
        (42, 40, 8, 0.8),
        (43, 128, 16, 2.56),
        (44, 96, 48, 1.92),
    ] {
        let (z, y) = columns(seed, n, p);
        let d = Design::from_columns(z, n, p, &y, lambda).unwrap();
        let loo = leave_one_out(&d, &Factor::new(&d).unwrap()).unwrap();
        let reference = brute_force_leave_one_out(&d).unwrap();
        let scale = reference.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        for (fast, slow) in loo.residuals.iter().zip(&reference) {
            assert!((fast - slow).abs() <= 1e-10 * scale, "seed {seed}");
        }
        let press: f64 = reference.iter().map(|e| e * e).sum();
        assert!((loo.press - press).abs() <= 1e-10 * press);
    }
    // The legacy surrogate design, with its heavy-tailed columns, agrees as well.
    let (x, y) = design(45, 60, 12);
    let d = Design::from_rows(&x, &y, 0.02).unwrap();
    let loo = leave_one_out(&d, &Factor::new(&d).unwrap()).unwrap();
    let reference = brute_force_leave_one_out(&d).unwrap();
    let scale = reference.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    for (fast, slow) in loo.residuals.iter().zip(&reference) {
        assert!((fast - slow).abs() <= 1e-10 * scale);
    }
}

#[test]
fn unsafe_pivots_high_leverage_and_bad_designs_are_refused() {
    // Two huge near-identical columns: the second pivot's roundoff dwarfs the penalty.
    let n = 30;
    let mut rng = Pcg32::new(51, 1);
    let base: Vec<f64> = (0..n).map(|_| 1e8 * rng.gaussian()).collect();
    let mut z = base.clone();
    z.extend(base.iter().map(|v| v + 1e-8));
    let d = Design::from_columns(z, n, 2, &vec![1.0; n], 1e-3).unwrap();
    assert!(matches!(
        Factor::new(&d),
        Err(Refusal::UnsafePivot { index: 1, .. })
    ));
    // One dominant row under a tiny penalty has leverage near one.
    let mut z = vec![0.01; 6];
    z[0] = 1e4;
    let d = Design::from_columns(z, 6, 1, &[3.0, 0.0, 1.0, 0.0, 1.0, 0.0], 1e-6).unwrap();
    assert!(matches!(
        leave_one_out(&d, &Factor::new(&d).unwrap()),
        Err(Refusal::Leverage { row: 0, .. })
    ));
    assert!(matches!(
        Design::from_columns(vec![1.0; 5], 2, 2, &[1.0, 2.0], 1.0),
        Err(Refusal::Shape(_))
    ));
    assert!(matches!(
        Design::from_columns(vec![f64::NAN, 1.0], 2, 1, &[1.0, 2.0], 1.0),
        Err(Refusal::NonFinite(_))
    ));
    assert!(matches!(
        Design::from_columns(vec![1.0, 2.0], 2, 1, &[1.0, 2.0], 0.0),
        Err(Refusal::NonFinite(_))
    ));
    // A prefix with a different penalty or more columns cannot be resumed.
    let (z, y) = columns(52, 20, 4);
    let d = Design::from_columns(z.clone(), 20, 4, &y, 1.0).unwrap();
    let other = Design::from_columns(z, 20, 4, &y, 2.0).unwrap();
    let prefix = Factor::new(&other.leading(2)).unwrap();
    assert!(matches!(
        Factor::extend(&prefix, &d),
        Err(Refusal::Shape(_))
    ));
    let wide = Factor::new(&d).unwrap();
    assert!(matches!(
        Factor::extend(&wide, &d.leading(2)),
        Err(Refusal::Shape(_))
    ));
    // Leave-one-out needs the design's own factor, not a narrower one or another penalty's.
    assert!(matches!(
        leave_one_out(&d, &Factor::new(&d.leading(2)).unwrap()),
        Err(Refusal::Shape(_))
    ));
    assert!(matches!(
        leave_one_out(&d, &Factor::new(&other).unwrap()),
        Err(Refusal::Shape(_))
    ));
    // A penalty under 2e-12 could accept a pivot the legacy solve replaces, so it refuses.
    assert!(matches!(
        Design::from_columns(vec![1.0, 2.0], 2, 1, &[1.0, 2.0], 1e-12),
        Err(Refusal::NonFinite(_))
    ));
    // The legacy design refuses a non-finite label and an empty table.
    let (x, mut y) = design(53, 6, 3);
    y[2] = f64::INFINITY;
    assert!(matches!(
        Design::from_rows(&x, &y, 0.02),
        Err(Refusal::NonFinite(_))
    ));
    assert!(matches!(
        Design::from_rows(&x, &[], 0.02),
        Err(Refusal::Shape(_))
    ));
    // Finite columns whose Gram entry overflows: the +inf pivot refuses instead of zeroing
    // every leverage.
    let d = Design::from_columns(vec![1e160, 1e160], 2, 1, &[1.0, 2.0], 1.0).unwrap();
    assert!(matches!(
        Factor::new(&d),
        Err(Refusal::UnsafePivot { index: 0, .. })
    ));
    // A finite pivot whose right-hand side overflows refuses too.
    let d = Design::from_columns(vec![1.0, -1.0], 2, 1, &[1e308, -1e308], 1.0).unwrap();
    assert!(matches!(Factor::new(&d), Err(Refusal::NonFinite(_))));
    // Labels whose centering overflows refuse at construction.
    assert!(matches!(
        Design::from_columns(vec![1.0, 2.0], 2, 1, &[f64::MAX, f64::MAX], 1.0),
        Err(Refusal::NonFinite(_))
    ));
    // A ragged table refuses instead of panicking, and n x p cannot overflow past the check.
    let ragged: Rows = vec![vec![1.0, 2.0], vec![3.0]];
    assert!(matches!(
        Design::from_rows(&ragged, &[1.0, 2.0], 0.02),
        Err(Refusal::Shape(_))
    ));
    assert!(matches!(
        Design::from_columns(Vec::new(), usize::MAX, 2, &[], 1.0),
        Err(Refusal::Shape(_))
    ));
}

#[test]
fn overflowing_and_non_finite_inputs_keep_the_legacy_bits_and_nan_positions() {
    let base = |r: usize| -> Vec<f64> { (0..4).map(|j| ((r * 3 + j) % 5) as f64).collect() };
    let labels: Vec<f64> = (0..9).map(|r| r as f64 * 0.7 - 2.0).collect();
    let table = || -> Rows { (0..9).map(base).collect() };
    let mut cases: Vec<(Rows, Vec<f64>)> = Vec::new();
    // A column whose sum overflows, and one whose squared deviations overflow.
    let mut x = table();
    x.iter_mut().for_each(|row| row[1] = f64::MAX);
    cases.push((x, labels.clone()));
    let mut x = table();
    for (r, row) in x.iter_mut().enumerate() {
        row[2] = if r % 2 == 0 { f64::MAX } else { -f64::MAX };
    }
    cases.push((x, labels.clone()));
    // A single +inf and a NaN cell.
    let mut x = table();
    x[8][0] = f64::INFINITY;
    cases.push((x, labels.clone()));
    let mut x = table();
    x[4][3] = f64::NAN;
    cases.push((x, labels.clone()));
    // Non-finite labels.
    let mut y = labels.clone();
    y[3] = f64::NEG_INFINITY;
    cases.push((table(), y));
    let mut y = labels.clone();
    y[5] = f64::NAN;
    cases.push((table(), y));
    for (index, (x, y)) in cases.iter().enumerate() {
        let legacy = legacy_fit(x, y, 0.02);
        let fast = RidgeModel::fit(x, y, 0.02);
        // Each case really leaves the finite range, so it cannot pass as a finite one.
        assert!(
            legacy
                .means
                .iter()
                .chain(&legacy.scales)
                .chain(&legacy.w)
                .any(|v| !v.is_finite()),
            "case {index} stays finite"
        );
        assert_eq!(
            canonical(&fast.means),
            canonical(&legacy.means),
            "case {index}"
        );
        assert_eq!(
            canonical(&fast.scales),
            canonical(&legacy.scales),
            "case {index}"
        );
        assert_eq!(canonical(&fast.w), canonical(&legacy.w), "case {index}");
        let probe: Rows = x.iter().rev().take(4).cloned().collect();
        assert_eq!(
            canonical(&fast.predict(&probe)),
            canonical(&legacy.predict(&probe)),
            "case {index}"
        );
    }
}

#[test]
fn zero_variance_and_huge_constant_columns_match_the_legacy_fit() {
    // Constant, 1e300 and signed-zero columns standardize to (±)0. The penalty floor keeps
    // every pivot at lambda or more, so `cholesky_solve`'s 1e-12 replacement branch (shared
    // code on both sides) is not reached here; `Factor` refuses any pivot that would reach it.
    let x: Rows = (0..5).map(|_| vec![2.0, 2.0, 2.0]).collect();
    assert_identical(&x, &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-12);
    let x: Rows = (0..9).map(|r| vec![f64::from(r), 1e300, -0.0]).collect();
    assert_identical(&x, &[0.5; 9], 0.02);
}
