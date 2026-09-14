//! Step-4: multi-target stacking — sibling-OOF coverage, gate consistency, and k==1 inertness.
use feat_core::backend::{MockBackend, TrainSpec};
use feat_core::data::Matrix;
use feat_core::hashing::mix64;
use feat_core::metrics::stratified_kfold;
use feat_core::progress::NullSink;
use feat_core::stacking::{build_sibling_oof_singles, select_model_stacked, StackConfig};
use feat_core::task::TaskKind;
use std::time::Instant;

fn pr(i: usize, s: u64) -> f32 {
    (mix64(i as u64 ^ s) % 10_000) as f32 / 10_000.0 * 2.0 - 1.0 // deterministic [-1,1)
}

/// Column-major features + primary y + multi-target buffer (k cols, primary at index 0: a correlated
/// sibling and a noise sibling).
fn synth(n: usize, p: usize, seed: u64) -> (Matrix, Vec<f32>, Vec<f32>) {
    let k = 3;
    let mut data = vec![0f32; n * p];
    for j in 0..p {
        for i in 0..n {
            data[j * n + i] = pr(i * p + j, seed);
        }
    }
    let x = Matrix::new(data, n, p);
    let y: Vec<f32> = (0..n)
        .map(|i| {
            if x.at(i, 0) + x.at(i, 1) > 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let mut ym = vec![0f32; n * k];
    for i in 0..n {
        ym[i] = y[i]; // col 0 = primary
        ym[n + i] = y[i]; // col 1 = correlated sibling
        ym[2 * n + i] = if pr(i, seed ^ 99) > 0.0 { 1.0 } else { 0.0 }; // col 2 = noise sibling
    }
    (x, y, ym)
}

#[test]
fn sibling_library_full_coverage() {
    let (n, p) = (400, 4);
    let (x, _y, ym) = synth(n, p, 1);
    let (xh, _, _) = synth(120, p, 2);
    let folds = stratified_kfold(&(0..n).collect::<Vec<_>>(), &ym[..n], 5, 7);
    let cols: Vec<u32> = (0..p as u32).collect();
    let spec = TrainSpec::from_basic_fields(10, 3, 0.1, 1, 0);
    let names = vec!["primary".into(), "sib0".into(), "sib1".into()];
    let sib = build_sibling_oof_singles(
        &MockBackend,
        &x,
        &ym,
        3,
        0,
        n,
        &xh,
        &folds,
        &cols,
        &spec,
        &spec,
        &names,
    );
    assert_eq!(sib.oof.len(), 2, "k-1 siblings");
    for o in &sib.oof {
        assert_eq!(o.len(), n);
        assert!(o.iter().all(|v| v.is_finite()), "no OOF holes");
    }
    assert!(sib.infer.iter().all(|c| c.len() == xh.n));
}

#[test]
fn gate_is_consistent() {
    let (n, p) = (400, 4);
    let (x, y, ym) = synth(n, p, 1);
    let (xh, yh, _) = synth(120, p, 2);
    let folds = stratified_kfold(&(0..n).collect::<Vec<_>>(), &y, 5, 7);
    let cols: Vec<u32> = (0..p as u32).collect();
    let spec = TrainSpec::from_basic_fields(20, 3, 0.1, 1, 0);
    let names = vec!["primary".into(), "sib0".into(), "sib1".into()];
    let scfg = StackConfig {
        enabled: true,
        gate_margin: 1e-4,
        rank_normalize: true,
        seed: 1,
    };
    let r = select_model_stacked(
        &MockBackend,
        &MockBackend,
        &MockBackend,
        &x,
        &y,
        &xh,
        &yh,
        &ym,
        3,
        0,
        &names,
        &folds,
        &cols,
        &spec,
        &spec,
        TaskKind::Binary,
        2,
        &scfg,
        &NullSink,
        Instant::now(),
    );
    assert_eq!(r.sibling_count, 2);
    assert_eq!(r.final_holdout_preds.len(), xh.n);
    assert!(r.final_holdout_auc.is_finite());
    // The gate is a no-regression contract: multi ships ONLY if it strictly beats the single ceiling.
    if r.gate_passed {
        assert!(
            r.multi_lcb > r.single_lcb,
            "gate passed ⇒ multi beats single"
        );
        assert_eq!(r.family, "multi-target");
    } else {
        assert_eq!(r.family, "single-target");
    }
}

#[test]
fn k1_is_inert() {
    let (n, p) = (300, 4);
    let (x, y, _) = synth(n, p, 1);
    let (xh, yh, _) = synth(100, p, 2);
    let folds = stratified_kfold(&(0..n).collect::<Vec<_>>(), &y, 5, 7);
    let cols: Vec<u32> = (0..p as u32).collect();
    let spec = TrainSpec::from_basic_fields(20, 3, 0.1, 1, 0);
    let ym = y.clone(); // single target -> k=1
    let scfg = StackConfig {
        enabled: true,
        ..StackConfig::default()
    };
    let r = select_model_stacked(
        &MockBackend,
        &MockBackend,
        &MockBackend,
        &x,
        &y,
        &xh,
        &yh,
        &ym,
        1,
        0,
        &["primary".into()],
        &folds,
        &cols,
        &spec,
        &spec,
        TaskKind::Binary,
        2,
        &scfg,
        &NullSink,
        Instant::now(),
    );
    assert_eq!(r.family, "single-target", "k==1 is inert");
    assert!(!r.gate_passed);
}
