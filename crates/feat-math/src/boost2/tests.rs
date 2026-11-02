use super::*;
use crate::metrics::r2;
use crate::model::cv_r2;
use crate::GenomeIr;

fn toy() -> (Rows, Vec<f64>) {
    // Same toy as model.rs tests: y = 2*x0 - x1 + 0.3*sin(6*x2).
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut rng = Pcg32::new(1, 1);
    for _ in 0..240 {
        let a = rng.next_f64();
        let b = rng.next_f64();
        let c = rng.next_f64();
        x.push(vec![a, b, c]);
        y.push(2.0 * a - b + 0.3 * (6.0 * c).sin());
    }
    (x, y)
}

#[test]
fn histgbm2_is_deterministic() {
    let (x, y) = toy();
    let cfg = HistGbm2Params::default();
    let m1 = HistGbm2Model::fit(&x, &y, &cfg);
    let m2 = HistGbm2Model::fit(&x, &y, &cfg);
    let p1 = m1.predict(&x);
    let p2 = m2.predict(&x);
    assert_eq!(p1, p2);
    assert!(r2(&y, &p1) > 0.8, "train r2={}", r2(&y, &p1));
}

#[test]
fn histgbm2_beats_histgbm_on_toy_cv() {
    let (x, y) = toy();
    let old = cv_r2(
        &GenomeIr::parse("(pipe X (histgbm))").unwrap(),
        &x,
        &y,
        5,
        "toy",
    )
    .unwrap();
    let new = cv_r2(
        &GenomeIr::parse("(pipe X (histgbm2))").unwrap(),
        &x,
        &y,
        5,
        "toy",
    )
    .unwrap();
    println!(
        "accuracy[toy cv oof_r2] histgbm={} histgbm2={}",
        old.oof_r2, new.oof_r2
    );
    assert!(
        new.oof_r2 >= old.oof_r2,
        "histgbm2 oof={} < histgbm oof={}",
        new.oof_r2,
        old.oof_r2
    );
}

/// Backward compatibility: a genome that spells out every new parameter at
/// its default must score bit-identically to the bare `(histgbm2)`.
#[test]
fn histgbm2_explicit_defaults_match_bare_head() {
    let (x, y) = toy();
    let bare = GenomeIr::parse("(pipe X (histgbm2))").unwrap();
    let full = GenomeIr::parse(
        "(pipe X (histgbm2 leaves=31 lr=0.05 bags=3 rounds=500 row_subsample=0.7 \
             col_subsample=1.0 min_leaf=3 val_pct=20 val_shuffle=false))",
    )
    .unwrap();
    let p1 = crate::model::fit_predict(&bare, &x, &y, &x).unwrap();
    let p2 = crate::model::fit_predict(&full, &x, &y, &x).unwrap();
    assert_eq!(p1, p2, "explicit defaults drifted from the bare head");
}

#[test]
fn histgbm2_deep_config_is_deterministic_and_learns() {
    let (x, y) = toy();
    let g = GenomeIr::parse(
        "(pipe X (histgbm2 leaves=127 rounds=300 lr=0.05 bags=2 row_subsample=0.8 \
             col_subsample=0.7 min_leaf=2 val_pct=15 val_shuffle=true))",
    )
    .unwrap();
    let p1 = crate::model::fit_predict(&g, &x, &y, &x).unwrap();
    let p2 = crate::model::fit_predict(&g, &x, &y, &x).unwrap();
    assert_eq!(p1, p2);
    assert!(r2(&y, &p1) > 0.8, "train r2={}", r2(&y, &p1));
    // The shuffled 15% holdout actually moves the split point.
    let cfg = HistGbm2Params::from(match &g {
        GenomeIr::Pipeline(e) => e.stages.last().unwrap(),
        _ => unreachable!(),
    });
    assert_eq!(cfg.val_pct, 15);
    assert!(cfg.val_shuffle);
    assert_eq!(cfg.rounds, 300);
    assert_eq!(cfg.min_leaf, 2);
}

#[test]
fn histgbm2_param_clamps_hold() {
    let g = GenomeIr::parse(
        "(pipe X (histgbm2 leaves=4096 rounds=9999 row_subsample=0.1 col_subsample=0.0 \
             min_leaf=500 val_pct=90))",
    )
    .unwrap();
    let cfg = HistGbm2Params::from(match &g {
        GenomeIr::Pipeline(e) => e.stages.last().unwrap(),
        _ => unreachable!(),
    });
    assert_eq!(cfg.leaves, 255);
    assert_eq!(cfg.rounds, 1000);
    assert_eq!(cfg.row_subsample, 0.5);
    assert_eq!(cfg.col_subsample, 0.5);
    assert_eq!(cfg.min_leaf, 20);
    assert_eq!(cfg.val_pct, 40);
}

#[test]
fn histgbm2_small_n_falls_back_to_mean() {
    let x: Rows = (0..6).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let m = HistGbm2Model::fit(&x, &y, &HistGbm2Params::default());
    assert!(m.bags.is_empty());
    let p = m.predict(&x);
    assert!(p.iter().all(|v| (*v - 2.5).abs() < 1e-12));
}
