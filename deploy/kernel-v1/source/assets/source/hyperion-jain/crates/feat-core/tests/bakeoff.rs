//! Phase-3.5 bake-off integration test (Mock backend; no CatBoost/Starforge needed).
//!
//! Drives the full pipeline with `bakeoff.enabled = true` and asserts the 5-strategy report:
//! 5 rows, the 2 Starforge rows unavailable (the test build has no `starforge-cpu` feature), the
//! available rows finite, and the shipped winner == argmax(CV) over the available strategies.
use feat_core::backend::{Backend, MockBackend, TrainSpec};
use feat_core::data::{Dataset, Matrix};
use feat_core::gp::GpConfig;
use feat_core::mvp::SelectionMode;
use feat_core::phase2::P2Config;
use feat_core::pipeline::{run_pipeline, PipelineConfig};
use feat_core::progress::NullSink;
use feat_core::rng::{seeded, Rng};
use feat_core::select::EdaConfig;

fn synth(n: usize, p: usize, seed: u64) -> Dataset {
    let mut rng = seeded(seed);
    let mut data = vec![0f32; n * p];
    let mut y = vec![0f32; n];
    for i in 0..n {
        for j in 0..p {
            data[j * n + i] = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let s = data[i] * 1.5 - data[n + i] + data[2 * n + i] * data[3 * n + i];
        let pr = 1.0 / (1.0 + (-3.0 * s).exp());
        y[i] = if rng.gen::<f32>() < pr { 1.0 } else { 0.0 };
    }
    let names: Vec<String> = (0..p).map(|j| format!("f{j}")).collect();
    Dataset {
        x: Matrix::new(data, n, p),
        y,
        y_multi: None,
        names,
    }
}

fn bakeoff_cfg() -> PipelineConfig {
    PipelineConfig {
        gp: GpConfig {
            pop: 200,
            gens: 2,
            islands: 2,
            depth: 4,
            subsample: 256,
            subsample_fit: 64,
            max_surv: 500,
            ..GpConfig::default()
        },
        p2: P2Config {
            epochs: 1,
            block: 4,
            subsample: 256,
            iters: 10,
            depth: 4,
            thread_count: 1,
            ..P2Config::default()
        },
        eda: EdaConfig {
            selection_mode: SelectionMode::MvpFast,
            pop: 8,
            gens: 2,
            folds: 3,
            ..EdaConfig::default()
        },
        cv_spec: TrainSpec::from_basic_fields(10, 4, 0.1, 1, 0),
        ho_spec: TrainSpec::from_basic_fields(20, 4, 0.1, 1, 0),
        resid_spec: TrainSpec::from_basic_fields(10, 4, 0.1, 1, 0),
        select_subsample: 500, // >= train n so the CV folds cover every row (dominance OOF needs it)
        max_gp: 200,
        do_mimo: false,
        seed: 7,
        task: feat_core::task::TaskKind::Binary,
        n_classes: 2,
        stack: feat_core::stacking::StackConfig::default(),
        dominance: feat_core::dominance::DominanceConfig::default(),
        starforge: feat_core::starforge_integration::StarforgeConfig::disabled(),
        router: feat_core::router::RouterConfig::default(),
        coreset_threshold: 50_000,
        coreset_guard_frac: 0.20,
        bakeoff: feat_core::bakeoff::BakeoffConfig { enabled: true },
        hyperion: Default::default(),
        invention: Default::default(),
        canary: Default::default(),
    }
}

#[test]
fn bakeoff_reports_five_strategies_and_ships_cv_winner() {
    let train = synth(400, 6, 1);
    let hold = synth(120, 6, 2);
    let cfg = bakeoff_cfg();

    // The bake-off requires the two extra backends (xgb, lgb); use mocks here.
    let extra: [&dyn Backend; 2] = [&MockBackend, &MockBackend];
    let res = run_pipeline(&train, &hold, &MockBackend, &extra, &cfg, &NullSink);

    let report = res.bakeoff.as_ref().expect("bake-off report present");
    assert_eq!(report.rows.len(), 5, "expected 5 strategies");

    // Starforge is not compiled into the test build -> both starforge rows must be unavailable.
    let sf: Vec<_> = report
        .rows
        .iter()
        .filter(|r| r.name.starts_with("starforge"))
        .collect();
    assert_eq!(sf.len(), 2, "two starforge strategies");
    assert!(
        sf.iter().all(|r| !r.available),
        "starforge rows must be unavailable without the feature"
    );

    // The available strategies (catboost x2, dominance) must produce finite, sane AUCs.
    let avail: Vec<_> = report.rows.iter().filter(|r| r.available).collect();
    assert!(
        avail.len() >= 3,
        "at least catboost x2 + dominance available"
    );
    for r in &avail {
        assert!(
            r.cv_auc.is_finite() && (0.0..=1.0).contains(&r.cv_auc),
            "{} cv {}",
            r.name,
            r.cv_auc
        );
        assert!(
            r.test_auc.is_finite() && (0.0..=1.0).contains(&r.test_auc),
            "{} test {}",
            r.name,
            r.test_auc
        );
    }

    // Winner = highest CV among available, and the shipped champion mirrors the winner's test AUC.
    let best = report
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.available)
        .max_by(|a, b| a.1.cv_auc.partial_cmp(&b.1.cv_auc).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(report.winner, best, "winner must be argmax CV");
    assert!(report.rows[report.winner].available);
    assert!(
        (res.champion_holdout_auc - report.rows[report.winner].test_auc).abs() < 1e-9,
        "shipped champion must equal winner test AUC"
    );
}
