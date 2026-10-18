//! Held-out poison check for the DSL stages named in the 2026-10-11 gap close.
//!
//! `cv_r2` fits each fold with `fit_predict` on the rows whose `fold_of`
//! assignment is not that fold, then scores the original held-out features.
//! The complement-helper check is retained alongside a direct public `cv_r2` oracle test.
//! The latter compares per-fold and pooled OOF reports and exercises an active leaky control.

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

/// Independent oracle: fit public models on explicit row complements, then compare every fold
/// score and the pooled OOF score with the real public CV dispatcher. This deliberately shares
/// neither cv_r2's orchestration nor any private fitting helper.
#[test]
fn public_cv_r2_matches_independent_oof_oracle_and_held_label_poison() {
    use feat_math::{cv_r2, fit};
    let key = "i03-public-dispatch";
    let k = 5;
    let x: Rows = (0..80)
        .map(|i| vec![(i % 7) as f64, (i as f64 * 0.19).sin()])
        .collect();
    let y: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, r)| 0.7 * r[0] + r[1] + (i % 3) as f64 * 0.2)
        .collect();
    for source in [
        "(pipe X (target_bin_encode bins=4 prior=1) (ridge))",
        "(pipe X (standardize) (ridge))",
        "(pipe X (quantile_bins bins=4) (ridge))",
        "(pipe X (winsor_clip q=0.1) (ridge))",
        "(pipe X (impute strategy=median) (ridge))",
        "(pipe X (ridge))",
    ] {
        let genome = GenomeIr::parse(source).unwrap();
        let oracle = |labels: &[f64]| {
            let mut predictions = vec![0.0; x.len()];
            let mut scores = Vec::new();
            let mut states = Vec::new();
            let mut by_fold = Vec::new();
            for f in 0..k {
                let training: Vec<_> = (0..x.len()).filter(|&i| fold_of(key, i, k) != f).collect();
                let held: Vec<_> = (0..x.len()).filter(|&i| fold_of(key, i, k) == f).collect();
                assert!(training.len() >= 8 && !held.is_empty());
                let xt: Rows = training.iter().map(|&i| x[i].clone()).collect();
                let yt: Vec<_> = training.iter().map(|&i| labels[i]).collect();
                let model = fit(&genome, &xt, &yt).unwrap();
                states.push(serde_json::to_value(&model).unwrap());
                let held_x = held.iter().map(|&i| x[i].clone()).collect();
                let pred = model.predict(&held_x).unwrap();
                for (&i, &p) in held.iter().zip(&pred) {
                    predictions[i] = p;
                }
                let held_y: Vec<_> = held.iter().map(|&i| labels[i]).collect();
                scores.push(feat_math::metrics::r2(&held_y, &pred));
                by_fold.push(pred);
            }
            (predictions, scores, states, by_fold)
        };
        let before = oracle(&y);
        let actual = cv_r2(&genome, &x, &y, k, key).unwrap();
        assert_eq!(actual.fold_r2, before.1, "{source}");
        assert_eq!(actual.oof_r2, feat_math::metrics::r2(&y, &before.0));
        assert_eq!(actual.mean_r2, before.1.iter().sum::<f64>() / k as f64);
        let mut poison = y.clone();
        for (i, label) in poison.iter_mut().enumerate() {
            if fold_of(key, i, k) == 0 {
                *label = 30.0 + (i % 11) as f64;
            }
        }
        let after = oracle(&poison);
        let dispatched = cv_r2(&genome, &x, &poison, k, key).unwrap();
        assert_eq!(dispatched.fold_r2, after.1, "poisoned {source}");
        assert_eq!(dispatched.oof_r2, feat_math::metrics::r2(&poison, &after.0));
        assert_eq!(
            before.2[0], after.2[0],
            "held labels changed fitted state: {source}"
        );
        assert_eq!(
            before.3[0], after.3[0],
            "held labels changed predictions: {source}"
        );
        assert_ne!(
            actual.fold_r2[0], dispatched.fold_r2[0],
            "poison must change measured loss"
        );
        // Active negative control: fitting all rows changes both state and held predictions.
        let leaky_before = fit(&genome, &x, &y).unwrap();
        let leaky_after = fit(&genome, &x, &poison).unwrap();
        assert_ne!(
            serde_json::to_value(&leaky_before).unwrap(),
            serde_json::to_value(&leaky_after).unwrap()
        );
        let held: Rows = x
            .iter()
            .enumerate()
            .filter(|(i, _)| fold_of(key, *i, k) == 0)
            .map(|(_, r)| r.clone())
            .collect();
        assert_ne!(
            leaky_before.predict(&held).unwrap(),
            leaky_after.predict(&held).unwrap(),
            "inactive negative control: {source}"
        );
    }
}
