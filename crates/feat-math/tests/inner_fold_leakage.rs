//! Held-out poison check for the DSL stages named in the 2026-10-11 gap close.
//!
//! `cv_r2` fits each fold with `fit_predict` on the rows whose `fold_of`
//! assignment is not that fold, then scores the original held-out features.
//! This test calls those public functions through `complement_predict`. It does
//! not call `cv_r2` and does not change it, so a `cv_r2` that fit every row
//! would not fail this test.

use feat_math::{fit_predict, fold_of, GenomeIr, Rows};

/// Train on the complement of `fold` and score caller-supplied rows.
/// `x_score` stays the original held-out features, so a poisoned feature is a
/// fit check and not a new prediction input.
fn complement_predict(
    genome: &GenomeIr,
    x: &Rows,
    y: &[f64],
    k: usize,
    key: &str,
    fold: usize,
    x_score: &Rows,
) -> Vec<f64> {
    let n = x.len().min(y.len());
    let mut x_fit = Vec::new();
    let mut y_fit = Vec::new();
    for r in 0..n {
        if fold_of(key, r, k) != fold {
            x_fit.push(x[r].clone());
            y_fit.push(y[r]);
        }
    }
    fit_predict(genome, &x_fit, &y_fit, x_score).unwrap()
}

/// Poisoning a fold's held-out rows must not change the model fitted on that
/// fold's fitting rows. Fitting those held-out rows as well must change the
/// predictions, so a leaked fit cannot hide.
#[test]
fn inner_fold_learned_stages_fit_only_on_fitting_rows() {
    let stages = [
        "(pipe X (target_bin_encode bins=4 prior=1) (ridge))",
        "(pipe X (quantile_bins bins=4) (ridge))",
        "(pipe X (winsor_clip q=0.1) (ridge))",
        "(pipe X (standardize) (ridge))",
        "(pipe X (impute strategy=median) (ridge))",
        "(pipe X (ridge))",
    ];
    let n = 60usize;
    let mut x: Rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let a = if i % 9 == 0 { 0.0 } else { (i as f64) * 0.17 };
        let b = ((i * 3) % 11) as f64 - 5.0;
        x.push(vec![a, b]);
        y.push(0.4 * (i as f64) - 0.2 * b);
    }
    let k = 5usize;
    let key = "gjudge-inner-fold";
    for src in stages {
        let genome = GenomeIr::parse(src).unwrap();
        let fold = (0..k)
            .find(|&fold| {
                let n_fit = (0..n).filter(|&r| fold_of(key, r, k) != fold).count();
                let n_held = n - n_fit;
                n_held > 0 && n_fit >= 8
            })
            .unwrap();
        let mut x_held = Vec::new();
        let mut held = Vec::new();
        for (r, row) in x.iter().enumerate().take(n) {
            if fold_of(key, r, k) == fold {
                x_held.push(row.clone());
                held.push(r);
            }
        }
        let honest = complement_predict(&genome, &x, &y, k, key, fold, &x_held);
        let leaked = fit_predict(&genome, &x, &y, &x_held).unwrap();
        assert_ne!(
            honest.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            leaked.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "negative control did not move {src}"
        );
        let mut x_poison = x.clone();
        let mut y_poison = y.clone();
        for &r in &held {
            x_poison[r] = vec![1.0e6, -1.0e6];
            y_poison[r] = 1.0e6;
        }
        let again = complement_predict(&genome, &x_poison, &y_poison, k, key, fold, &x_held);
        assert_eq!(
            honest.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "held-out poison changed the fold fit for {src}"
        );
        let mut y_only = y.clone();
        for &r in &held {
            y_only[r] = 1.0e6;
        }
        let target_only = complement_predict(&genome, &x, &y_only, k, key, fold, &x_held);
        assert_eq!(
            honest.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            target_only.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "held-out target leaked into the inner-fold fit for {src}"
        );
    }
}
