//! Tests for the Track-0 trust/eval foundation: stable hashing, FoldScope, leakage detectors,
//! rank/AUC-LCB metric primitives, and TrainSpec-v2 regime defaults. These lock in the invariants the
//! rest of the Apex upgrade depends on.
use feat_core::backend::{auto_defaults, infer_regime, BackendKind, SizeRegime};
use feat_core::data::Matrix;
use feat_core::foldscope::{FoldPolicy, FoldScope};
use feat_core::hashing::{dataset_hash, mix64, row_hash};
use feat_core::leakage::{scan, LeakageSeverity};
use feat_core::metrics::{fold_auc_lcb, rank01, stratified_kfold};
use std::collections::HashSet;

fn mat(cols: &[Vec<f32>], n: usize) -> Matrix {
    let mut flat = Vec::new();
    for c in cols {
        flat.extend_from_slice(c);
    }
    Matrix::new(flat, n, cols.len())
}

#[test]
fn hashing_is_deterministic_and_sensitive() {
    let x = mat(&[vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0, 0.0, 1.0]], 4);
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let names = vec!["a".to_string(), "b".to_string()];
    let h1 = dataset_hash(&x, &y, &names);
    assert_eq!(
        h1,
        dataset_hash(&x, &y, &names),
        "dataset_hash deterministic"
    );
    let x2 = mat(&[vec![1.0, 2.0, 3.0, 5.0], vec![0.0, 1.0, 0.0, 1.0]], 4);
    assert_ne!(
        h1,
        dataset_hash(&x2, &y, &names),
        "content change -> hash change"
    );
    assert_ne!(mix64(1), mix64(2));
    assert_eq!(row_hash(&x, 0), row_hash(&x, 0));
    assert_ne!(row_hash(&x, 0), row_hash(&x, 1));
}

#[test]
fn rank01_is_monotone_in_unit_interval() {
    let r = rank01(&[3.0, 1.0, 2.0, 4.0]);
    assert!((r[1] - 0.0).abs() < 1e-6, "smallest -> 0");
    assert!((r[3] - 1.0).abs() < 1e-6, "largest -> 1");
    assert!(r[2] > r[1] && r[0] > r[2], "monotone in value");
    for v in &r {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn foldscope_root_is_deterministic_and_partitions() {
    let y: Vec<f32> = (0..200)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    let scope = FoldScope::root_stratified(&y, 200, 5, 7);
    assert_eq!(scope.policy, FoldPolicy::Stratified { k: 5 });
    assert_eq!(scope.folds.len(), 5);
    let mut seen = HashSet::new();
    for (_, va) in &scope.folds {
        for &r in va {
            assert!(seen.insert(r), "each active row in exactly one val fold");
        }
    }
    assert_eq!(
        scope.fold_hash,
        FoldScope::root_stratified(&y, 200, 5, 7).fold_hash
    );
    assert_ne!(
        scope.fold_hash,
        FoldScope::root_stratified(&y, 200, 5, 8).fold_hash,
        "different seed -> different fold hash"
    );
}

#[test]
fn leakage_flags_exact_target_copy() {
    let n = 100;
    let y: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
    let copy = y.clone(); // a feature equal to the target -> AUC 1.0 -> Blocker
    let constant = vec![5.0f32; n];
    let noise: Vec<f32> = (0..n)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as f32)
        .collect();
    let x = mat(&[copy, constant, noise], n);
    let names = vec![
        "leak_feature".to_string(),
        "const".to_string(),
        "noise".to_string(),
    ];
    let rep = scan(&x, &y, &names, None);
    assert!(rep.n_blockers() >= 1, "exact target copy must be a blocker");
    assert!(rep.max_single_feature_auc >= 0.99);
    assert_eq!(rep.worst(), LeakageSeverity::Blocker);
}

#[test]
fn leakage_clean_data_has_no_blockers() {
    let n = 300;
    let y: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
    // hash-derived features, independent of parity -> no near-perfect separators
    let f0: Vec<f32> = (0..n)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as f32)
        .collect();
    let f1: Vec<f32> = (0..n)
        .map(|i| ((i as u32).wrapping_mul(40503) >> 20) as f32)
        .collect();
    let x = mat(&[f0, f1], n);
    let names = vec!["f0".to_string(), "f1".to_string()];
    let rep = scan(&x, &y, &names, None);
    assert_eq!(rep.n_blockers(), 0, "clean data must have no blockers");
}

#[test]
fn fold_auc_lcb_perfect_predictor() {
    let y: Vec<f32> = (0..100).map(|i| (i % 2) as f32).collect();
    let pred = y.clone();
    let folds = stratified_kfold(&(0..100).collect::<Vec<_>>(), &y, 5, 1);
    let s = fold_auc_lcb(&pred, &y, &folds, 1.0);
    assert!((s.mean - 1.0).abs() < 1e-9);
    assert!(s.lcb <= s.mean);
}

#[test]
fn oof_member_is_covered_and_separable() {
    use feat_core::backend::{MockBackend, TrainSpec};
    use feat_core::ensemble::build_oof_member;
    let p = 4;
    let mk = |seed: u64, rows: usize| -> (Matrix, Vec<f32>) {
        let mut x = vec![0f32; rows * p];
        let mut y = vec![0f32; rows];
        let mut s = seed | 1;
        let mut nxt = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for r in 0..rows {
            let mut lat = 0f32;
            for j in 0..p {
                let v = ((nxt() >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                x[j * rows + r] = v;
                if j < 2 {
                    lat += v;
                }
            }
            y[r] = if lat > 0.0 { 1.0 } else { 0.0 };
        }
        (Matrix::new(x, rows, p), y)
    };
    let (xt, yt) = mk(1, 300);
    let (xh, _yh) = mk(2, 100);
    let folds = stratified_kfold(&(0..300).collect::<Vec<_>>(), &yt, 5, 7);
    let cols: Vec<u32> = (0..p as u32).collect();
    let spec = TrainSpec::from_basic_fields(10, 3, 0.1, 1, 0);
    let mp = build_oof_member(
        "mock",
        &MockBackend,
        &xt,
        &yt,
        &xh,
        &folds,
        &cols,
        &spec,
        &spec,
        feat_core::task::TaskKind::Binary,
        2,
    );
    assert!(
        mp.oof.iter().all(|v| v.is_finite()),
        "OOF fully covered (no holes)"
    );
    assert_eq!(mp.infer.len(), 100);
    assert!(
        mp.mean_auc > 0.6,
        "separable signal -> out-of-fold AUC > 0.6, got {}",
        mp.mean_auc
    );
}

#[test]
fn ges_is_deterministic_and_prefers_better_member() {
    use feat_core::ensemble::{bagged_ges, GesConfig};
    let n = 200;
    let y: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
    let good: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.2 } else { 0.8 }).collect();
    let noisy: Vec<f32> = (0..n).map(|i| ((i * 7 % 5) as f32) / 5.0).collect();
    let cols = vec![good, noisy];
    let folds = stratified_kfold(&(0..n).collect::<Vec<_>>(), &y, 5, 1);
    let cfg = GesConfig::default();
    let a = bagged_ges(&cols, &y, &folds, &cfg);
    let b = bagged_ges(&cols, &y, &folds, &cfg);
    assert_eq!(a.0, b.0, "GES weights deterministic for a fixed seed");
    assert!(a.0[0] >= a.0[1], "the better member gets >= weight");
    assert!(a.1 >= 0.95, "GES OOF mean AUC tracks the strong member");
}

#[test]
fn missing_features_are_target_free_and_appended() {
    use feat_core::data::Dataset;
    use feat_core::missing::augment_missing;
    let n = 100;
    let p = 3;
    let mut data = vec![0f32; n * p];
    for i in 0..n {
        data[i] = if i % 5 == 0 { f32::NAN } else { i as f32 }; // col0: 20% missing
        data[n + i] = (i as f32) * 0.5; // col1: dense
        data[2 * n + i] = if i % 20 == 0 { f32::NAN } else { -(i as f32) }; // col2: 5% missing
    }
    let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let mk = |y: Vec<f32>| Dataset {
        x: Matrix::new(data.clone(), n, p),
        y,
        y_multi: None,
        names: names.clone(),
    };
    let train = mk((0..n).map(|i| (i % 2) as f32).collect());
    let hold = mk((0..n).map(|i| ((i / 7) % 2) as f32).collect());
    let (at, _ah) = augment_missing(&train, &hold, 0.01);
    assert_eq!(
        at.p(),
        p + 4,
        "base + 2 indicators + miss_count + miss_frac"
    );
    assert!(at.names.iter().any(|s| s == "miss::a"));
    assert!(at.names.iter().any(|s| s == "miss_frac"));
    assert!(
        !at.names.iter().any(|s| s == "miss::b"),
        "dense column must not get an indicator"
    );
    // target-free: a different label must not change the APPENDED missingness columns (the base columns
    // carry NaN, so compare only the appended part, which is finite 0/1/count/frac).
    let train2 = mk((0..n).map(|i| ((i * 3) % 2) as f32).collect());
    let (at2, _) = augment_missing(&train2, &hold, 0.01);
    let app = p * n;
    assert_eq!(
        at.x.data[app..],
        at2.x.data[app..],
        "missingness features are label-independent"
    );
}

#[test]
fn regime_inference_and_defaults() {
    assert_eq!(infer_regime(1500, 10, 0.3).size, SizeRegime::Tiny);
    assert_eq!(infer_regime(50000, 147, 0.337).size, SizeRegime::Medium);
    assert_eq!(infer_regime(2_000_000, 21, 0.5).size, SizeRegime::Large);
    assert!(infer_regime(300, 600, 0.5).wide, "p/n>0.5 is wide");
    assert!(infer_regime(2600, 1200, 0.5).wide, "p>1000 is wide");
    assert!(!infer_regime(2600, 500, 0.5).wide, "p/n=0.19 is NOT wide");
    // STS-like medium regime
    let s = auto_defaults(
        infer_regime(50000, 147, 0.337),
        BackendKind::CatBoost,
        0.337,
        7,
        8,
    );
    assert_eq!(s.depth, 6);
    assert!(s.early_stopping_rounds.is_some());
    assert!(s.use_best_iteration);
    // imbalanced overlay
    let imb = auto_defaults(
        infer_regime(5000, 20, 0.05),
        BackendKind::CatBoost,
        0.05,
        7,
        8,
    );
    assert!(imb.scale_pos_weight.is_some());
    assert!(imb.class_weights.is_some());
}
