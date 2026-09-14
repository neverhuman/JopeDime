//! End-to-end pipeline smoke test on synthetic data using the Mock backend (no CatBoost needed).
//! Validates that all four phases run, indices stay consistent, and metrics are finite/sane.
use feat_core::backend::{MockBackend, TrainSpec};
use feat_core::data::{Dataset, Matrix};
use feat_core::dominance::{select_dominant, DominanceConfig};
use feat_core::gp::GpConfig;
use feat_core::metrics::stratified_kfold;
use feat_core::phase2::P2Config;
use feat_core::phase4::MimoScorer;
use feat_core::pipeline::{run_pipeline, PipelineConfig};
use feat_core::progress::{Event, NullSink, Sink};
use feat_core::rng::{seeded, Rng};
use feat_core::router::{RouterArtifacts, RouterConfig};
use feat_core::select::{EdaConfig, MaskScorer};
use feat_core::task::TaskKind;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Default)]
struct CaptureSink {
    events: Mutex<Vec<Event>>,
}

impl Sink for CaptureSink {
    fn emit(&self, e: &Event) {
        self.events.lock().unwrap().push(e.clone());
    }
}

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

#[test]
fn pipeline_runs_end_to_end_with_mock() {
    let train = synth(400, 6, 1);
    let hold = synth(120, 6, 2);

    let cfg = PipelineConfig {
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
            pop: 8,
            gens: 2,
            folds: 3,
            ..EdaConfig::default()
        },
        cv_spec: TrainSpec::from_basic_fields(10, 4, 0.1, 1, 0),
        ho_spec: TrainSpec::from_basic_fields(20, 4, 0.1, 1, 0),
        resid_spec: TrainSpec::from_basic_fields(10, 4, 0.1, 1, 0),
        select_subsample: 300,
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
        bakeoff: Default::default(),
        hyperion: Default::default(),
        invention: Default::default(),
        canary: Default::default(),
    };

    let res = run_pipeline(&train, &hold, &MockBackend, &[], &cfg, &NullSink);
    assert!(
        res.baseline_holdout_auc > 0.0 && res.baseline_holdout_auc < 1.0,
        "baseline {}",
        res.baseline_holdout_auc
    );
    assert!(
        res.champion_holdout_auc > 0.0 && res.champion_holdout_auc < 1.0,
        "champ {}",
        res.champion_holdout_auc
    );
    assert!(res.champion_size >= 1, "champion empty");
    assert!(
        res.n_features >= train.p(),
        "augmented should be >= base feature count"
    );
    // Mock is linear; the synthetic target has linear signal so baseline should beat chance.
    assert!(
        res.baseline_holdout_auc > 0.55,
        "baseline should learn linear signal: {}",
        res.baseline_holdout_auc
    );
}

#[test]
fn pipeline_mvp_fast_respects_max_features_with_mock() {
    let train = synth(240, 6, 31);
    let hold = synth(80, 6, 32);
    let mut cfg = tiny_pipeline_config(31);
    cfg.eda.selection_mode = feat_core::mvp::SelectionMode::MvpFast;
    cfg.eda.mvp.max_features = 4;
    cfg.eda.mvp.importance_pct = 0.0;
    cfg.eda.mvp.top_frac = 1.0;

    let sink = CaptureSink::default();
    let res = run_pipeline(&train, &hold, &MockBackend, &[], &cfg, &sink);
    assert!(res.champion_size >= 1);
    assert!(
        res.champion_size <= 4,
        "champion_size={}",
        res.champion_size
    );
    assert!(res.champion_holdout_auc.is_finite());
    let events = sink.events.lock().unwrap();
    let mvp_done = events
        .iter()
        .find(|e| e.phase == 3 && e.event == "done" && e.model.as_deref() == Some("mvp-fast"))
        .expect("mvp-fast done event");
    assert!(mvp_done.accepted_features.is_some());
    assert!(mvp_done.selected_features.is_some());
    assert_eq!(mvp_done.final_features, Some(res.champion_size as u64));
}

#[test]
fn pipeline_streams_gp_winners_and_task_metric_with_mock() {
    let train = synth(240, 6, 41);
    let hold = synth(80, 6, 42);
    let mut cfg = tiny_pipeline_config(41);
    cfg.eda.selection_mode = feat_core::mvp::SelectionMode::MvpFast;
    cfg.eda.mvp.max_features = 4;
    cfg.eda.mvp.importance_pct = 0.0;
    cfg.eda.mvp.top_frac = 1.0;

    let sink = CaptureSink::default();
    let _res = run_pipeline(&train, &hold, &MockBackend, &[], &cfg, &sink);
    let events = sink.events.lock().unwrap();

    // GP winners must be attached exactly when survivors have been admitted (phase 1).
    let winners_seen = events
        .iter()
        .any(|e| e.phase == 1 && e.winners.as_ref().is_some_and(|w| !w.is_empty()));
    let survivors_seen = events
        .iter()
        .any(|e| e.phase == 1 && e.survivors.unwrap_or(0) > 0);
    assert_eq!(
        winners_seen, survivors_seen,
        "GP winners must accompany admitted survivors"
    );
    assert!(
        events.iter().all(|e| e.event != "canary_purge"),
        "canary-derived GP features should not produce bulk purge events"
    );
    if winners_seen {
        let w = events
            .iter()
            .filter_map(|e| e.winners.as_ref())
            .flatten()
            .next()
            .unwrap();
        assert!(!w.formula.is_empty(), "winner carries a formula");
    }

    // The mvp-fast selection 'done' event carries a task-aware metric label (Binary -> auc).
    let mvp_done = events
        .iter()
        .find(|e| e.phase == 3 && e.event == "done" && e.model.as_deref() == Some("mvp-fast"))
        .expect("mvp-fast done event");
    assert_eq!(mvp_done.metric.as_deref(), Some("auc"));
    // Every emitted event carries the schema version.
    assert!(events
        .iter()
        .all(|e| e.v == feat_core::progress::EVENT_SCHEMA_VERSION));
}

#[test]
fn pipeline_preemptive_cancel_skips_final_and_baseline() {
    use std::sync::atomic::{AtomicBool, Ordering};

    // Trips cancellation the moment Phase 3 finishes, so GP/importance/selection all complete
    // normally and only the uninterruptible Phase-5 / MIMO / baseline stages are guard-skipped.
    struct CancelAfterPhase3 {
        tripped: AtomicBool,
    }
    impl Sink for CancelAfterPhase3 {
        fn emit(&self, e: &Event) {
            if e.phase == 3 && e.event == "done" {
                self.tripped.store(true, Ordering::SeqCst);
            }
        }
        fn is_cancelled(&self) -> bool {
            self.tripped.load(Ordering::SeqCst)
        }
    }

    let train = synth(240, 6, 51);
    let hold = synth(80, 6, 52);
    let mut cfg = tiny_pipeline_config(51);
    cfg.eda.selection_mode = feat_core::mvp::SelectionMode::MvpFast;
    cfg.eda.mvp.max_features = 4;
    cfg.eda.mvp.importance_pct = 0.0;
    cfg.eda.mvp.top_frac = 1.0;

    // Two extra backends so Phase 5 WOULD run if we weren't cancelled — proving it is skipped.
    let extra: [&dyn feat_core::backend::Backend; 2] = [&MockBackend, &MockBackend];
    let sink = CancelAfterPhase3 {
        tripped: AtomicBool::new(false),
    };
    let res = run_pipeline(&train, &hold, &MockBackend, &extra, &cfg, &sink);

    assert!(res.cancelled, "pipeline should report cancellation");
    assert!(res.ensemble.is_none(), "Phase 5 must be skipped on cancel");
    assert_eq!(
        res.baseline_holdout_auc, 0.0,
        "baseline must be skipped on cancel"
    );
    assert_eq!(res.mimo_holdout_auc, None);
}

#[test]
fn pipeline_keeps_hyperion_and_canaries_search_only_unless_gp_selects_them() {
    let mut train = synth(180, 6, 71);
    let mut hold = synth(64, 6, 72);
    let names = vec![
        "f0".into(),
        "f1".into(),
        "f2".into(),
        "f3".into(),
        "hyperion::v000".into(),
        "hyperion::v001".into(),
    ];
    train.names = names.clone();
    hold.names = names;

    let mut cfg = tiny_pipeline_config(71);
    cfg.gp.gens = 0;
    cfg.max_gp = 0;
    cfg.eda.mvp.max_features = 4;
    cfg.eda.mvp.importance_pct = 0.0;
    cfg.eda.mvp.top_frac = 1.0;

    let res = run_pipeline(&train, &hold, &MockBackend, &[], &cfg, &NullSink);
    assert!(res.canary.enabled);
    assert_eq!(
        res.canary.count, 6,
        "one CANARY_FEATURES terminal per GP-visible source column"
    );
    assert_eq!(res.n_survivors, 0);
    assert_eq!(
        res.n_features, 4,
        "final model matrix should keep only visible base columns when GP selects nothing"
    );
    assert_eq!(res.artifact.features.len(), 4);
    assert!(res.artifact.features.iter().all(|f| f.source == "base"
        && !f.formula.starts_with("hyperion::")
        && !feat_core::canary::is_canary_name(&f.formula)));
    assert!(res.artifact.selected_cols.iter().all(|&c| c < 4));
    assert_eq!(res.canary.selected_canary_count, 0);
    assert_eq!(res.canary.top_importance_contaminated_count, 0);
}

fn tiny_pipeline_config(seed: u64) -> PipelineConfig {
    PipelineConfig {
        gp: GpConfig {
            pop: 48,
            gens: 1,
            islands: 1,
            depth: 3,
            subsample: 96,
            subsample_fit: 32,
            max_surv: 96,
            ..GpConfig::default()
        },
        p2: P2Config {
            epochs: 1,
            block: 3,
            subsample: 96,
            iters: 4,
            depth: 3,
            thread_count: 1,
            ..P2Config::default()
        },
        eda: EdaConfig {
            pop: 6,
            gens: 1,
            folds: 2,
            target_size: 4.0,
            ..EdaConfig::default()
        },
        cv_spec: TrainSpec::from_basic_fields(4, 3, 0.1, 1, seed),
        ho_spec: TrainSpec::from_basic_fields(4, 3, 0.1, 1, seed),
        resid_spec: TrainSpec::from_basic_fields(4, 3, 0.1, 1, seed),
        select_subsample: 96,
        max_gp: 96,
        do_mimo: false,
        seed,
        task: TaskKind::Binary,
        n_classes: 2,
        stack: feat_core::stacking::StackConfig::default(),
        dominance: DominanceConfig::default(),
        starforge: feat_core::starforge_integration::StarforgeConfig::disabled(),
        router: RouterConfig::default(),
        coreset_threshold: 50_000,
        coreset_guard_frac: 0.20,
        bakeoff: Default::default(),
        hyperion: Default::default(),
        invention: Default::default(),
        canary: Default::default(),
    }
}

#[test]
fn pipeline_records_router_decision_when_assets_are_enabled() {
    let train = synth(96, 5, 11);
    let hold = synth(36, 5, 12);
    let artifact = RouterArtifacts {
        algorithms: vec![
            "catboost".into(),
            "starlight_core_classifier".into(),
            "xgboost_hist".into(),
        ],
        train_features: vec![vec![0.0; 9], vec![1.0; 9], vec![0.5; 9]],
        train_utilities: vec![
            vec![0.70, 0.72, 0.68],
            vec![0.71, 0.75, 0.69],
            vec![0.69, 0.74, 0.70],
        ],
        train_source_families: vec!["a".into(), "b".into(), "c".into()],
        family_names: Vec::new(),
        family_ids: Vec::new(),
        standardize: true,
    };
    let path =
        std::env::temp_dir().join(format!("jain-router-pipeline-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_string(&artifact).unwrap()).unwrap();

    let mut cfg = tiny_pipeline_config(13);
    cfg.router = RouterConfig {
        enabled: true,
        require: true,
        assets_path: Some(path.clone()),
        z: 0.0,
        ..RouterConfig::default()
    };
    let extra: [&dyn feat_core::backend::Backend; 2] = [&MockBackend, &MockBackend];
    let res = run_pipeline(&train, &hold, &MockBackend, &extra, &cfg, &NullSink);

    let decision = res.router.expect("router decision");
    assert!(!decision.selected_algorithm.is_empty());
    assert!(decision.selected_label.starts_with("CVBEST"));
    let _ = std::fs::remove_file(path);
}

#[test]
fn phase4_mimo_scorer_scores_tiny_multi_target_data() {
    let n = 12;
    let x = Matrix::new(
        vec![
            0.0, 0.1, 0.9, 1.0, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55, 1.0, 0.9, 0.1, 0.0, 0.8,
            0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.5, 0.2, 0.4, 0.1, 0.6, 0.3, 0.7, 0.8, 0.9, 0.0,
            0.15, 0.85,
        ],
        n,
        3,
    );
    let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let sibling = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let mut y_multi = Vec::new();
    y_multi.extend_from_slice(&y);
    y_multi.extend_from_slice(&sibling);
    let xh = x.clone();
    let yh = y.clone();
    let folds = vec![
        ((4..12).collect::<Vec<_>>(), vec![0, 1, 2, 3]),
        ([0, 1, 2, 3, 8, 9, 10, 11].to_vec(), vec![4, 5, 6, 7]),
        ((0..8).collect::<Vec<_>>(), vec![8, 9, 10, 11]),
    ];
    let scorer = MimoScorer {
        x: &x,
        y: &y,
        y_multi: &y_multi,
        k: 2,
        primary: 0,
        xh: &xh,
        yh: &yh,
        folds: &folds,
        be: &MockBackend,
        cv_spec: TrainSpec::from_basic_fields(4, 2, 0.1, 1, 1),
        ho_spec: TrainSpec::from_basic_fields(4, 2, 0.1, 1, 1),
        task: TaskKind::Binary,
        n_classes: 2,
    };

    let (mean, std) = scorer.score_cv(&[0, 1, 2]);
    let holdout = scorer.score_holdout(&[0, 1, 2]);
    assert!(mean.is_finite());
    assert!(std.is_finite());
    assert!(holdout.is_finite());
}

#[test]
fn dominance_falls_back_for_regression_and_keeps_best_single_gate() {
    let train = synth(36, 4, 21);
    let hold = synth(16, 4, 22);
    let y_reg: Vec<f32> = (0..train.n())
        .map(|i| train.x.at(i, 0) * 0.7 - train.x.at(i, 1) * 0.2 + i as f32 * 0.001)
        .collect();
    let yh_reg: Vec<f32> = (0..hold.n())
        .map(|i| hold.x.at(i, 0) * 0.7 - hold.x.at(i, 1) * 0.2 + i as f32 * 0.001)
        .collect();
    let folds = vec![
        ((12..36).collect::<Vec<_>>(), (0..12).collect::<Vec<_>>()),
        (
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35,
            ]
            .to_vec(),
            (12..24).collect::<Vec<_>>(),
        ),
        ((0..24).collect::<Vec<_>>(), (24..36).collect::<Vec<_>>()),
    ];
    let cols = vec![0, 1, 2, 3];
    let spec = TrainSpec::from_basic_fields(4, 2, 0.1, 1, 1);

    let regression = select_dominant(
        &MockBackend,
        &MockBackend,
        &MockBackend,
        &train.x,
        &y_reg,
        &hold.x,
        &yh_reg,
        &folds,
        &cols,
        &spec,
        &spec,
        TaskKind::Regression,
        0,
        &DominanceConfig {
            enabled: true,
            bbc_bootstraps: 4,
            seed: 2,
        },
        &NullSink,
        Instant::now(),
    );
    assert_eq!(regression.family, "single-target");
    assert_eq!(regression.final_holdout_preds.len(), hold.n());
    assert!(regression.final_cv_auc.is_finite());

    let binary_folds = stratified_kfold(&(0..train.n()).collect::<Vec<_>>(), &train.y, 3, 3);
    let binary = select_dominant(
        &MockBackend,
        &MockBackend,
        &MockBackend,
        &train.x,
        &train.y,
        &hold.x,
        &hold.y,
        &binary_folds,
        &cols,
        &spec,
        &spec,
        TaskKind::Binary,
        2,
        &DominanceConfig {
            enabled: true,
            bbc_bootstraps: 4,
            seed: 3,
        },
        &NullSink,
        Instant::now(),
    );
    assert_eq!(binary.family, "dominance");
    assert_eq!(binary.final_holdout_preds.len(), hold.n());
    if binary.gate_passed {
        assert_ne!(binary.final_model, "best_single");
        assert!(binary.final_cv_auc >= binary.single_lcb - 1e-9);
    } else {
        assert_eq!(binary.final_model, "best_single");
    }
}
