//! Independent B11 oracle: no production Cholesky, PRESS identity, or brute-force helper.
use feat_math::kernel::ridge_fast::{leave_one_out, Design, Factor};

// Row-major normal equations, solved by partial-pivoted Gaussian elimination. The
// fixture passes already centered labels: dropping a row must not re-center them.
fn solve(rows: &[Vec<f64>], centered: &[f64], lambda: f64, skip: Option<usize>) -> Vec<f64> {
    let p = rows[0].len();
    let mut a = vec![vec![0.0; p + 1]; p];
    for (i, equation) in a.iter_mut().enumerate() {
        for (r, row) in rows.iter().enumerate() {
            if Some(r) != skip {
                for j in 0..p {
                    equation[j] += row[i] * row[j];
                }
                equation[p] += row[i] * centered[r];
            }
        }
        equation[i] += lambda;
    }
    for k in 0..p {
        let pivot = (k..p)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        a.swap(k, pivot);
        assert!(a[k][k].is_finite() && a[k][k] != 0.0);
        let pivot_row = a[k].clone();
        for equation in a.iter_mut().skip(k + 1) {
            let ratio = equation[k] / pivot_row[k];
            for (value, &pivot_value) in equation[k..].iter_mut().zip(&pivot_row[k..]) {
                *value -= ratio * pivot_value;
            }
        }
    }
    let mut weights = vec![0.0; p];
    for i in (0..p).rev() {
        let later: f64 = (i + 1..p).map(|j| a[i][j] * weights[j]).sum();
        weights[i] = (a[i][p] - later) / a[i][i];
    }
    weights
}

fn prediction(row: &[f64], weights: &[f64]) -> f64 {
    row.iter().zip(weights).map(|(x, w)| x * w).sum()
}

fn fixture(p: usize, salt: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for r in 0..p + 4 {
        let row: Vec<f64> = (0..p)
            .map(|j| {
                let col = if salt == 2 && j == p - 1 { 0 } else { j };
                ((r * (col + 3) + col * col + salt * 5) % 13) as f64 / 4.0 - 1.5
            })
            .collect();
        let offset = ((r * 7 + salt * 3) % 17 + 1) as f64 / 8.0;
        rows.push(row.clone());
        rows.push(row.iter().map(|x| -x).collect());
        labels.extend([4.0 + offset, 4.0 - offset]);
    }
    (rows, labels)
}

fn design(rows: &[Vec<f64>], labels: &[f64], lambda: f64) -> Design {
    let (n, p) = (rows.len(), rows[0].len());
    let z = (0..p)
        .flat_map(|j| rows.iter().map(move |row| row[j]))
        .collect();
    Design::from_columns(z, n, p, labels, lambda).unwrap()
}

#[test]
fn press_matches_independent_deleted_row_solves_with_frozen_mean_and_penalty() {
    let mut designs = 0;
    let mut residuals = 0;
    for p in 0..=8 {
        for salt in 0..3 {
            let (rows, labels) = fixture(p, salt);
            let centered: Vec<f64> = labels.iter().map(|y| y - 4.0).collect();
            assert_eq!(labels.iter().sum::<f64>() / labels.len() as f64, 4.0);
            for lambda in [0.25, 2.0, 16.0] {
                let d = design(&rows, &labels, lambda);
                let fast = leave_one_out(&d, &Factor::new(&d).unwrap()).unwrap();
                assert_eq!(fast.residuals.len(), rows.len());
                let mut reference_press = 0.0;
                for (r, row) in rows.iter().enumerate() {
                    let w = solve(&rows, &centered, lambda, Some(r));
                    let expected = centered[r] - prediction(row, &w);
                    assert!(
                        expected.abs() > 1e-6,
                        "oracle corpus needs nonzero residuals"
                    );
                    assert!(
                        (fast.residuals[r] - expected).abs() <= 1e-10 * expected.abs(),
                        "p={p} salt={salt} lambda={lambda} row={r}: {} != {expected}",
                        fast.residuals[r]
                    );
                    reference_press += expected * expected;
                    residuals += 1;
                }
                assert!((fast.press - reference_press).abs() <= 1e-10 * reference_press);
                designs += 1;
            }
        }
    }
    assert_eq!((designs, residuals), (81, 1296));
    println!("B11 independent PRESS: {designs} designs, {residuals} deleted-row solves");
}

fn objective(rows: &[Vec<f64>], centered: &[f64], lambda: f64) -> f64 {
    let weights = solve(rows, centered, lambda, None);
    let penalty = lambda * weights.iter().map(|w| w * w).sum::<f64>();
    penalty
        + rows
            .iter()
            .zip(centered)
            .map(|(row, y)| (y - prediction(row, &weights)).powi(2))
            .sum::<f64>()
}

#[test]
fn schur_gains_match_independent_objectives_at_every_prefix_width() {
    let mut prefixes = 0;
    for p in 0..=8 {
        for salt in 0..3 {
            let (rows, labels) = fixture(p, salt);
            let centered: Vec<f64> = labels.iter().map(|y| y - 4.0).collect();
            for lambda in [0.25, 2.0, 16.0] {
                let d = design(&rows, &labels, lambda);
                let full = Factor::new(&d).unwrap();
                let full_objective = objective(&rows, &centered, lambda);
                for width in 0..=p {
                    let prefix_rows: Vec<Vec<f64>> =
                        rows.iter().map(|row| row[..width].to_vec()).collect();
                    let prefix_objective = objective(&prefix_rows, &centered, lambda);
                    let expected = prefix_objective - full_objective;
                    assert!(
                        (full.schur_gain(width) - expected).abs() <= 1e-10 * prefix_objective,
                        "p={p} salt={salt} lambda={lambda} prefix={width}"
                    );
                    let prefix = Factor::new(&d.leading(width)).unwrap();
                    assert_eq!(Factor::extend(&prefix, &d).unwrap(), full);
                    prefixes += 1;
                }
            }
        }
    }
    assert_eq!(prefixes, 405);
    println!("B11 independent Schur: {prefixes} prefix objectives and exact extensions");
}
