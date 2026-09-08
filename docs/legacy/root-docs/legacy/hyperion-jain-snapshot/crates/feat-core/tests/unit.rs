//! Unit coverage for the algorithmic core (all via public APIs + MockBackend; no CatBoost needed).
use feat_core::backend::{Backend, MockBackend, TrainSpec};
use feat_core::buildmatrix::build_augmented;
use feat_core::data::{load_csv, split_holdout, Dataset, Matrix};
use feat_core::dsl::{ops, Bin, Expr, Un};
use feat_core::gp::{run_gp, GpConfig, GpData};
use feat_core::metrics::{auc, corr_std, standardize, stratified_kfold, stratified_subsample};
use feat_core::mvp::{reduce_features, MvpFastConfig};
use feat_core::phase2::{
    block_percentiles, importance_epochs, importance_epochs_detailed, P2Config,
};
use feat_core::phase3::SingleScorer;
use feat_core::progress::NullSink;
use feat_core::rng::{seeded, Rng};
use feat_core::select::{run_eda, EdaConfig};
use std::time::Instant;

// ---------------- metrics ----------------

#[test]
fn auc_perfect_and_inverse_and_ties() {
    let y = [0.0, 0.0, 1.0, 1.0];
    let perfect = [0.1, 0.2, 0.8, 0.9];
    assert!((auc(&y, &perfect) - 1.0).abs() < 1e-9);
    let inverse = [0.9, 0.8, 0.2, 0.1];
    assert!(auc(&y, &inverse).abs() < 1e-9);
    let all_tied = [0.5, 0.5, 0.5, 0.5];
    assert!((auc(&y, &all_tied) - 0.5).abs() < 1e-9);
    // single class -> 0.5
    assert!((auc(&[1.0, 1.0], &[0.1, 0.9]) - 0.5).abs() < 1e-9);
}

#[test]
fn standardize_and_corr() {
    let (s, ok) = standardize(&[1.0, 2.0, 3.0, 4.0]);
    assert!(ok);
    let mean: f32 = s.iter().sum::<f32>() / s.len() as f32;
    assert!(mean.abs() < 1e-5);
    // identical standardized vectors -> corr 1; degenerate -> not ok
    assert!((corr_std(&s, &s) - 1.0).abs() < 1e-4);
    let (_, ok2) = standardize(&[5.0, 5.0, 5.0]);
    assert!(!ok2);
}

#[test]
fn stratified_subsample_preserves_ratio() {
    let mut y = vec![0.0; 1000];
    y[..200].fill(1.0); // 20% positive
    let mut rng = seeded(1);
    let sub = stratified_subsample(&y, 300, &mut rng);
    let pos = sub.iter().filter(|&&i| y[i] > 0.5).count();
    assert!(sub.len() >= 295 && sub.len() <= 300);
    let ratio = pos as f64 / sub.len() as f64;
    assert!((ratio - 0.20).abs() < 0.03, "ratio {ratio}");
}

#[test]
fn kfold_disjoint_and_complete() {
    let rows: Vec<usize> = (0..100).collect();
    let mut y = vec![0.0; 100];
    y[..40].fill(1.0);
    let folds = stratified_kfold(&rows, &y, 5, 42);
    assert_eq!(folds.len(), 5);
    let mut all_val = std::collections::HashSet::new();
    for (tr, va) in &folds {
        for &i in va {
            assert!(all_val.insert(i), "val index {i} appeared twice");
        }
        for &i in va {
            assert!(!tr.contains(&i), "train/val overlap");
        }
    }
    assert_eq!(
        all_val.len(),
        100,
        "every row appears in exactly one val fold"
    );
}

// ---------------- dsl ----------------

#[test]
fn dsl_eval_basic_ops() {
    // column-major 3 rows x 2 cols
    let m = Matrix::new(vec![1.0, 2.0, f32::NAN, 10.0, 20.0, 30.0], 3, 2);
    assert_eq!(Expr::Col(1).eval(&m), vec![10.0, 20.0, 30.0]);
    // Miss flags the NaN in col0 row2
    assert_eq!(Expr::Miss(0).eval(&m), vec![0.0, 0.0, 1.0]);
    let add = Expr::Bin(Bin::Add, Box::new(Expr::Col(0)), Box::new(Expr::Col(1))).eval(&m);
    assert_eq!(add[0], 11.0);
    assert!(add[2].is_nan());
}

#[test]
fn dsl_safediv_and_erf_and_rank() {
    let mut v = vec![0.0, 1.0, -2.0, 100.0];
    let sd = ops::apply_bin(Bin::SafeDiv, &[1.0, 1.0], &[0.0, 2.0]);
    assert!(sd[0].is_nan() && (sd[1] - 0.5).abs() < 1e-6);
    assert!(ops::erf(0.0).abs() < 1e-6);
    assert!(ops::erf(5.0) > 0.999 && ops::erf(-5.0) < -0.999);
    ops::apply_un(Un::Rank, &mut v);
    assert!(v.iter().all(|&x| (0.0..=1.0).contains(&x)));
    // monotone: smallest -> 0, largest -> 1
    assert!((v[2] - 0.0).abs() < 1e-6 && (v[3] - 1.0).abs() < 1e-6);
}

#[test]
fn dsl_zscore_and_where() {
    let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    ops::apply_un(Un::Zscore, &mut v);
    assert!((v.iter().sum::<f32>()).abs() < 1e-4);
    let m = Matrix::new(vec![1.0, 0.0, 9.0, 9.0, -1.0, -1.0], 2, 3);
    // where(col0>0.5, col1, col2): row0 col0=1>0.5 -> col1=9 ; row1 col0=0 -> col2=-1
    let w = Expr::Where(
        Box::new(Expr::Col(0)),
        Box::new(Expr::Col(1)),
        Box::new(Expr::Col(2)),
    )
    .eval(&m);
    assert_eq!(w, vec![9.0, -1.0]);
}

// ---------------- gp ----------------

fn synth_matrix(n: usize, p: usize, seed: u64) -> (Matrix, Vec<f32>) {
    let mut rng = seeded(seed);
    let mut data = vec![0f32; n * p];
    let mut y = vec![0f32; n];
    for i in 0..n {
        for j in 0..p {
            data[j * n + i] = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let s = data[i] * 1.5 + data[2 * n + i] * data[3 * n + i];
        y[i] = if 1.0 / (1.0 + (-3.0 * s).exp()) > rng.gen::<f32>() {
            1.0
        } else {
            0.0
        };
    }
    (Matrix::new(data, n, p), y)
}

#[test]
fn gp_survivors_are_pairwise_decorrelated() {
    let (x, y) = synth_matrix(400, 6, 3);
    let r: Vec<f32> = y.iter().map(|&v| v - 0.5).collect(); // simple residual proxy
    let names: Vec<String> = (0..6).map(|j| format!("f{j}")).collect();
    let gd = GpData::new(x.clone(), names, r, &y, 9, 256);
    let cfg = GpConfig {
        pop: 200,
        gens: 3,
        islands: 2,
        depth: 4,
        tau: 0.5,
        subsample: 256,
        subsample_fit: 64,
        max_surv: 300,
        ..GpConfig::default()
    };
    let surv = run_gp(&gd, &cfg, &NullSink, Instant::now());
    assert!(!surv.is_empty(), "GP found no survivors");
    // formulas unique
    let mut set = std::collections::HashSet::new();
    for s in &surv {
        assert!(set.insert(s.formula.clone()), "duplicate formula emitted");
    }
    // signal sorted descending
    for w in surv.windows(2) {
        assert!(w[0].signal >= w[1].signal - 1e-6);
    }
    // core invariant: pairwise |corr| < tau on full data
    let std: Vec<Vec<f32>> = surv
        .iter()
        .map(|s| standardize(&s.expr.eval(&x)).0)
        .collect();
    let mut worst = 0f32;
    for i in 0..std.len() {
        for j in (i + 1)..std.len() {
            worst = worst.max(corr_std(&std[i], &std[j]));
        }
    }
    assert!(
        worst < cfg.tau + 1e-3,
        "pairwise corr {worst} exceeds tau {}",
        cfg.tau
    );
}

// ---------------- buildmatrix + phase2 ----------------

fn synth_dataset(n: usize, p: usize, seed: u64) -> Dataset {
    let (x, y) = synth_matrix(n, p, seed);
    let names: Vec<String> = (0..p).map(|j| format!("f{j}")).collect();
    Dataset {
        x,
        y,
        y_multi: None,
        names,
    }
}

fn invented_report(
    name: &str,
    cv_mean: f64,
    holdout: f64,
    holdout_rows: usize,
) -> feat_core::invention::InventionReport {
    feat_core::invention::InventionReport {
        finalists: vec![feat_core::invention::InventedFinalist {
            name: name.to_string(),
            dsl: "(pipe X (ridge))".into(),
            cv_mean,
            fold_scores: vec![cv_mean],
            holdout,
            holdout_preds: vec![holdout as f32; holdout_rows],
            python: String::new(),
            rust: String::new(),
        }],
        evals: 1,
        tokens_total: 1.0,
    }
}

#[test]
fn buildmatrix_no_nan_and_correct_shape() {
    let train = synth_dataset(300, 5, 1);
    let hold = synth_dataset(120, 5, 2);
    let r: Vec<f32> = train.y.iter().map(|&v| v - 0.5).collect();
    let gd = GpData::new(train.x.clone(), train.names.clone(), r, &train.y, 1, 256);
    let surv = run_gp(
        &gd,
        &GpConfig {
            pop: 120,
            gens: 2,
            islands: 2,
            depth: 4,
            subsample: 256,
            subsample_fit: 64,
            max_surv: 200,
            ..GpConfig::default()
        },
        &NullSink,
        Instant::now(),
    );
    let aug = build_augmented(&train, &hold, &surv, 100);
    assert_eq!(aug.features.len(), aug.train.c);
    assert!(aug.train.c >= train.p());
    assert_eq!(aug.train.n, train.n());
    assert_eq!(aug.hold.n, hold.n());
    assert!(
        aug.train.data.iter().all(|v| v.is_finite()),
        "train has NaN/inf after fill"
    );
    assert!(
        aug.hold.data.iter().all(|v| v.is_finite()),
        "hold has NaN/inf after fill"
    );
}

#[test]
fn phase2_memory_vector_normalized_with_floor() {
    let train = synth_dataset(800, 8, 5);
    let cfg = P2Config {
        epochs: 2,
        block: 4,
        subsample: 400,
        iters: 10,
        depth: 3,
        thread_count: 1,
        floor: 0.05,
        seed: 1,
    };
    let mv = importance_epochs(
        &train.x,
        &train.y,
        &MockBackend,
        &cfg,
        feat_core::backend::ObjectiveSpec::from_kind(feat_core::task::TaskKind::Binary),
        &NullSink,
        Instant::now(),
    );
    assert_eq!(mv.len(), train.p());
    let sum: f64 = mv.iter().map(|&v| v as f64).sum();
    assert!((sum - 1.0).abs() < 1e-4, "memory_vector sum {sum}");
    let floor_min = 0.05f32 / train.p() as f32 * 0.9;
    assert!(
        mv.iter().all(|&v| v >= floor_min),
        "every feature must keep nonzero weight"
    );
}

#[test]
fn phase2_smoothed_percentile_ranks_importance_within_block() {
    let pct = block_percentiles(&[0.2, 0.8, 0.1, 0.4]);
    assert_eq!(pct, vec![0.4, 0.8, 0.2, 0.6]);
}

#[test]
fn phase2_detailed_returns_importance_percentiles() {
    let train = synth_dataset(400, 6, 6);
    let cfg = P2Config {
        epochs: 1,
        block: 3,
        subsample: 200,
        iters: 6,
        depth: 3,
        thread_count: 1,
        floor: 0.05,
        seed: 2,
    };
    let out = importance_epochs_detailed(
        &train.x,
        &train.y,
        &MockBackend,
        &cfg,
        feat_core::backend::ObjectiveSpec::from_kind(feat_core::task::TaskKind::Binary),
        &NullSink,
        Instant::now(),
    );
    assert_eq!(out.memory_vector.len(), train.p());
    assert_eq!(out.importance_percentile.len(), train.p());
    assert!(out
        .importance_percentile
        .iter()
        .all(|&v| v > 0.0 && v < 1.0));
}

#[test]
fn mvp_fast_accepts_gp_by_percentile_and_keeps_top_fraction() {
    let n = 8;
    let x = Matrix::new(
        vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, // base 0
            0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, // base 1
            0.0, 0.1, 0.4, 0.9, 0.2, 0.8, 0.3, 0.7, // gp accepted, high
            0.2, 0.3, 0.1, 0.5, 0.4, 0.6, 0.8, 0.9, // gp accepted
            0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, // gp rejected
        ],
        n,
        5,
    );
    let features: Vec<feat_core::buildmatrix::FeatureInfo> = (0..5)
        .map(|j| feat_core::buildmatrix::FeatureInfo {
            source: if j < 2 { "base" } else { "gp" }.into(),
            formula: format!("f{j}"),
            expr: Expr::Col(j as u16),
            fill: 0.0,
            scale_min: 0.0,
            scale_span: 1.0,
            canary: None,
            canary_refs: Vec::new(),
        })
        .collect();
    let pct = vec![0.1, 0.1, 0.95, 0.70, 0.20];
    let sel = reduce_features(
        &x,
        &features,
        &pct,
        &MvpFastConfig {
            importance_pct: 0.30,
            target_new: 3,
            top_frac: 0.5,
            max_features: 4,
            corr_penalty: 0.25,
        },
    );
    assert_eq!(sel.report.accepted_gp, 2);
    assert_eq!(sel.report.shortfall, 1);
    assert_eq!(sel.report.selected_gp, 1);
    assert_eq!(sel.cols.len(), 3);
    assert!(sel.cols.contains(&0) && sel.cols.contains(&1));
    assert!(
        sel.cols.contains(&2),
        "highest accepted GP feature should win"
    );
}

#[test]
fn mvp_fast_reduces_base_when_base_exceeds_cap() {
    let n = 5;
    let x = Matrix::new(
        vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0,
        ],
        n,
        4,
    );
    let features: Vec<feat_core::buildmatrix::FeatureInfo> = (0..4)
        .map(|j| feat_core::buildmatrix::FeatureInfo {
            source: "base".into(),
            formula: format!("b{j}"),
            expr: Expr::Col(j as u16),
            fill: 0.0,
            scale_min: 0.0,
            scale_span: 1.0,
            canary: None,
            canary_refs: Vec::new(),
        })
        .collect();
    let sel = reduce_features(
        &x,
        &features,
        &[0.1, 0.9, 0.8, 0.7],
        &MvpFastConfig {
            max_features: 2,
            ..MvpFastConfig::default()
        },
    );
    assert_eq!(sel.cols.len(), 2);
    assert!(sel.cols.contains(&1));
}

#[test]
fn starforge_top_importance_cols_are_sorted_and_capped() {
    let mv = vec![0.1, 0.7, 0.2, 0.4];
    let cols = feat_core::starforge_integration::top_importance_cols(&mv, 3);
    assert_eq!(cols, vec![1, 3, 2]);
    let all = feat_core::starforge_integration::top_importance_cols(&mv, 99);
    assert_eq!(all.len(), mv.len());
}

// ---------------- select (EDA) ----------------

#[test]
fn eda_champion_valid_and_finite() {
    let train = synth_dataset(600, 6, 7);
    let hold = synth_dataset(200, 6, 8);
    let folds = stratified_kfold(&(0..train.n()).collect::<Vec<_>>(), &train.y, 3, 1);
    let scorer = SingleScorer {
        x: &train.x,
        y: &train.y,
        xh: &hold.x,
        yh: &hold.y,
        folds: &folds,
        be: &MockBackend,
        cv_spec: TrainSpec::from_basic_fields(10, 3, 0.1, 1, 0),
        ho_spec: TrainSpec::from_basic_fields(10, 3, 0.1, 1, 0),
        task: feat_core::task::TaskKind::Binary,
        n_classes: 2,
    };
    let mv = vec![1.0f32 / 6.0; 6];
    let cfg = EdaConfig {
        pop: 8,
        gens: 3,
        folds: 3,
        target_size: 4.0,
        ..EdaConfig::default()
    };
    let champ = run_eda(&mv, None, &scorer, &cfg, &NullSink, 3, Instant::now());
    assert!(!champ.cols.is_empty());
    assert!(
        champ.cols.iter().all(|&c| (c as usize) < train.p()),
        "champion index out of range"
    );
    assert!(champ.holdout_auc > 0.0 && champ.holdout_auc < 1.0);
}

// ---------------- data + progress ----------------

#[test]
fn load_csv_and_split() {
    let dir = std::env::temp_dir().join(format!("jain_csv_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("d.csv");
    std::fs::write(
        &path,
        "a,b,target\n1.0,2.0,1\n3.0,,0\n5.0,6.0,1\n7.0,8.0,0\n",
    )
    .unwrap();
    let ds = load_csv(&path, &["target"], "target").unwrap();
    assert_eq!(ds.n(), 4);
    assert_eq!(ds.p(), 2);
    assert!(ds.x.at(1, 1).is_nan(), "empty field should be NaN");
    assert_eq!(ds.y, vec![1.0, 0.0, 1.0, 0.0]);
    let (tr, ho) = split_holdout(&ds, 0.5, 1);
    assert_eq!(tr.n() + ho.n(), 4);
}

#[test]
fn mock_backend_learns_separable_signal() {
    let (x, y) = synth_matrix(500, 4, 11);
    let m = MockBackend.train_single(&x.data, x.n, x.c, &y, &TrainSpec::default());
    assert_eq!(m.output_dim(), 1);
    let pred = m.predict(&x.data, x.n, x.c, true);
    assert!(
        auc(&y, &pred) > 0.55,
        "mock should beat chance on separable data"
    );
    let imp = m.importance(x.c);
    assert_eq!(imp.len(), 4);
    assert!((imp.iter().sum::<f32>() - 100.0).abs() < 1.0);
}

#[test]
fn ensemble_select_returns_valid() {
    let train = synth_dataset(500, 6, 21);
    let hold = synth_dataset(150, 6, 22);
    let folds = stratified_kfold(&(0..train.n()).collect::<Vec<_>>(), &train.y, 3, 1);
    let cols: Vec<u32> = (0..train.p() as u32).collect();
    let spec = TrainSpec::from_basic_fields(10, 3, 0.1, 1, 0);
    let starforge = feat_core::starforge_integration::StarforgeConfig::disabled();
    let r = feat_core::ensemble::select_model(
        &MockBackend,
        &MockBackend,
        &MockBackend,
        &train.x,
        &train.y,
        &hold.x,
        &hold.y,
        &folds,
        &cols,
        &spec,
        &spec,
        feat_core::task::TaskKind::Binary,
        2,
        &starforge,
        &[],
        None,
        &NullSink,
        Instant::now(),
    );
    assert!(
        [
            "catboost",
            "xgboost",
            "lightgbm",
            "catboost_s2",
            "ensemble",
            "ges"
        ]
        .contains(&r.final_model.as_str()),
        "unexpected fuser: {}",
        r.final_model
    );
    assert!(r.final_holdout_auc > 0.0 && r.final_holdout_auc < 1.0);
    assert!(r.ensemble_cv > 0.0 && r.catboost_cv > 0.0);
    assert!(r.active_members >= 1 && r.oof_lcb > 0.0);
}

#[test]
fn final_selection_ships_higher_cv_invented_math_standalone() {
    let train = synth_dataset(160, 5, 131);
    let hold = synth_dataset(50, 5, 132);
    let folds = stratified_kfold(&(0..train.n()).collect::<Vec<_>>(), &train.y, 4, 11);
    let cols: Vec<u32> = (0..train.p() as u32).collect();
    let spec = TrainSpec::from_basic_fields(4, 3, 0.1, 1, 0);
    let starforge = feat_core::starforge_integration::StarforgeConfig::disabled();
    let invented = invented_report("math_forced", 9.0, 0.77, hold.n());

    let r = feat_core::ensemble::select_model(
        &MockBackend,
        &MockBackend,
        &MockBackend,
        &train.x,
        &train.y,
        &hold.x,
        &hold.y,
        &folds,
        &cols,
        &spec,
        &spec,
        feat_core::task::TaskKind::Binary,
        2,
        &starforge,
        &[],
        Some(&invented),
        &NullSink,
        Instant::now(),
    );

    assert_eq!(r.final_model, "math_forced");
    assert_eq!(r.selected_fuser, "math_forced");
    assert_eq!(r.active_members, 1);
    assert_eq!(r.final_cv_auc, 9.0);
    assert_eq!(r.final_holdout_preds, vec![0.77; hold.n()]);
    let plan = r
        .final_plan
        .as_ref()
        .expect("math winner must be persistable");
    assert_eq!(plan.members.len(), 1);
    assert_eq!(plan.members[0].backend, "math");
    assert_eq!(plan.members[0].weight, 1.0);
    assert!(plan.members[0].math.is_some());
}

#[test]
fn final_selection_keeps_native_when_invented_cv_is_lower() {
    let train = synth_dataset(160, 5, 141);
    let hold = synth_dataset(50, 5, 142);
    let folds = stratified_kfold(&(0..train.n()).collect::<Vec<_>>(), &train.y, 4, 12);
    let cols: Vec<u32> = (0..train.p() as u32).collect();
    let spec = TrainSpec::from_basic_fields(4, 3, 0.1, 1, 0);
    let starforge = feat_core::starforge_integration::StarforgeConfig::disabled();
    let invented = invented_report("math_low", -1.0, 0.77, hold.n());

    let r = feat_core::ensemble::select_model(
        &MockBackend,
        &MockBackend,
        &MockBackend,
        &train.x,
        &train.y,
        &hold.x,
        &hold.y,
        &folds,
        &cols,
        &spec,
        &spec,
        feat_core::task::TaskKind::Binary,
        2,
        &starforge,
        &[],
        Some(&invented),
        &NullSink,
        Instant::now(),
    );

    assert_ne!(r.final_model, "math_low");
    assert_ne!(r.selected_fuser, "math_low");
    assert!(r.final_cv_auc > -1.0);
    assert_ne!(r.final_plan.as_ref().unwrap().members[0].backend, "math");
}

#[test]
fn residual_starforge_fusion_skips_when_weights_are_unavailable() {
    let train = synth_dataset(180, 5, 41);
    let hold = synth_dataset(60, 5, 42);
    let folds = stratified_kfold(&(0..train.n()).collect::<Vec<_>>(), &train.y, 3, 1);
    let cols: Vec<u32> = (0..train.p() as u32).collect();
    let spec = TrainSpec::from_basic_fields(4, 3, 0.1, 1, 0);
    let missing =
        std::env::temp_dir().join(format!("jain-missing-starforge-{}", std::process::id()));
    let starforge = feat_core::starforge_integration::StarforgeConfig {
        weights_path: missing.with_extension("classifier"),
        regressor_weights_path: missing.with_extension("regressor"),
        ..feat_core::starforge_integration::StarforgeConfig::default()
    };
    let r = feat_core::ensemble::select_model(
        &MockBackend,
        &MockBackend,
        &MockBackend,
        &train.x,
        &train.y,
        &hold.x,
        &hold.y,
        &folds,
        &cols,
        &spec,
        &spec,
        feat_core::task::TaskKind::Binary,
        2,
        &starforge,
        &[],
        None,
        &NullSink,
        Instant::now(),
    );
    assert_ne!(r.final_model, "catboost_residual_starforge");
    assert_ne!(r.selected_fuser, "catboost_residual_starforge");
}
