//! apex_bench — the FROZEN, 100%-Rust fitness used by the Apex self-improvement funnel.
//!
//! It runs the full `run_pipeline` over a small deterministic synthetic basket using ONLY the
//! pure-Rust `MockBackend` (no CatBoost/XGB/LGB, no C, no Python), prints `APEX_METRIC <mean>` (the
//! mean champion holdout AUC across the basket — the headline the funnel ratchets), and asserts a
//! conservative no-regression floor so any change that breaks the pipeline fails `cargo test`.
//!
//! Run the metric:  cargo test -p feat-core --test apex_bench -- --nocapture
//! The funnel greps the `APEX_METRIC <f64>` line; the cross-generation ratchet vs the prior
//! baseline lives in ops/apex/lib/verify_land.sh, NOT here (here we only guard the floor).
//!
//! DO NOT weaken this file: removing datasets, lowering the floor, or deleting asserts is exactly
//! the "vibe" regression the funnel's jail-validate + honest verdict are built to reject.
use feat_core::backend::{MockBackend, TrainSpec};
use feat_core::bench::{no_regression_gate, BasketEntry, BasketReport, GateOutcome, MetricKind};
use feat_core::data::{Dataset, Matrix};
use feat_core::gp::GpConfig;
use feat_core::phase2::P2Config;
use feat_core::pipeline::{run_pipeline, PipelineConfig};
use feat_core::progress::{RunContext, StderrProgressSink};
use feat_core::rng::{seeded, Rng};
use feat_core::select::EdaConfig;

/// Deterministic synthetic dataset: target depends on a linear term + a pairwise interaction (signal
/// the linear Mock backend partly captures and the GP synthesis layer is meant to recover). `kind`
/// varies the generative process so the basket is not a single distribution.
fn synth(n: usize, p: usize, seed: u64, kind: u8) -> Dataset {
    let mut rng = seeded(seed);
    let mut data = vec![0f32; n * p];
    let mut y = vec![0f32; n];
    for i in 0..n {
        for j in 0..p {
            data[j * n + i] = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let (a, b, c, d) = (data[i], data[n + i], data[2 * n + i], data[3 * n + i]);
        let s = match kind {
            0 => 1.6 * a - 1.1 * b + 0.9 * c * d, // linear + interaction
            1 => 1.3 * a + 0.8 * (b * b - 0.33) + 0.7 * c, // linear + curvature
            _ => 1.4 * a - 0.9 * b + 1.0 * (c * d) - 0.5 * d, // mixed
        };
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

fn cfg(seed: u64) -> PipelineConfig {
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
        seed,
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
    }
}

/// The frozen floor every basket dataset must clear (conservative; well below honest performance, so
/// the gate only fires on a genuinely broken pipeline). The generational ratchet is external.
const FLOOR: f64 = 0.55;

#[test]
fn apex_metric_basket() {
    // (name, train_n, hold_n, p, seed, kind) — a tiny, deterministic, multi-distribution basket.
    let specs: [(&str, usize, usize, usize, u64, u8); 3] = [
        ("synth_lin", 600, 200, 6, 11, 0),
        ("synth_curv", 600, 200, 7, 23, 1),
        ("synth_mix", 600, 200, 8, 37, 2),
    ];

    let mut cur = BasketReport::default();
    let mut floor = BasketReport::default();
    let mut sum = 0.0f64;
    for (idx, (name, ntr, nho, p, seed, kind)) in specs.iter().enumerate() {
        let train = synth(*ntr, *p, *seed, *kind);
        let hold = synth(*nho, *p, seed.wrapping_add(1000), *kind);
        let cfg = cfg(*seed);
        let sink = StderrProgressSink::new(
            RunContext::new("apex-bench")
                .dataset(*name)
                .task("Binary")
                .rows(train.n(), hold.n())
                .features(train.p())
                .selection_mode(cfg.eda.selection_mode.as_str()),
        );
        let res = run_pipeline(&train, &hold, &MockBackend, &[], &cfg, &sink);
        let m = res.champion_holdout_auc as f64;
        assert!(
            m.is_finite() && m > 0.0 && m < 1.0,
            "{name}: non-finite/oob champion AUC {m}"
        );
        let hash = (idx as u64) + 1; // stable identity per basket slot
        println!(
            "APEX_DATASET {name} champion={m:.6} baseline={:.6} hash={hash} kind=auc",
            res.baseline_holdout_auc
        );
        cur.entries.push(BasketEntry {
            name: (*name).into(),
            metric: m,
            kind: MetricKind::Auc,
            adv_auc: 0.5,
            dataset_hash: hash,
            secs: 0.0,
        });
        floor.entries.push(BasketEntry {
            name: (*name).into(),
            metric: FLOOR,
            kind: MetricKind::Auc,
            adv_auc: 0.5,
            dataset_hash: hash,
            secs: 0.0,
        });
        sum += m;
    }

    let mean = sum / specs.len() as f64;
    // The single headline line the funnel greps. Keep the token exact: `APEX_METRIC <f64>`.
    println!("APEX_METRIC {mean:.6}");

    // No-regression floor: every basket dataset must clear FLOOR, else the pipeline is broken.
    match no_regression_gate(&floor, &cur, 0.0) {
        GateOutcome::Pass => {}
        GateOutcome::Regressed(bad) => {
            panic!("apex_bench floor breached (pipeline broken): {bad:?}")
        }
    }
    assert!(mean > FLOOR, "APEX_METRIC {mean:.4} below floor {FLOOR}");
}
