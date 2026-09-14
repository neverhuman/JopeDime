//! apex_leak_guard — honesty test for the Apex TRIPLE GATE (axis 3: NOT-GAMED).
//!
//! It plants a perfect single-feature separator (a column equal to the target) and asserts the
//! pipeline's leak guard (`drop_leak_columns`, |AUC|>=0.995) drops it. If a candidate weakens or
//! disables the leak gate to fake a "gain", this test fails `cargo test -p feat-core`, so the QUALITY
//! gate rejects the candidate. 100% Rust, C-free.
//!
//! FROZEN: jail-validate forbids candidates from editing this file. Do not weaken it.
use feat_core::backend::{MockBackend, TrainSpec};
use feat_core::data::{Dataset, Matrix};
use feat_core::gp::GpConfig;
use feat_core::phase2::P2Config;
use feat_core::pipeline::{run_pipeline, PipelineConfig};
use feat_core::progress::NullSink;
use feat_core::rng::{seeded, Rng};
use feat_core::select::EdaConfig;

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

/// `p` ordinary features + ONE planted leaky column (index `p`) equal to the target — a perfect
/// single-feature separator (AUC = 1.0) that the leak guard must drop.
fn synth_with_leak(n: usize, p: usize, seed: u64) -> Dataset {
    let mut rng = seeded(seed);
    let cols = p + 1;
    let mut data = vec![0f32; n * cols];
    let mut y = vec![0f32; n];
    for i in 0..n {
        for j in 0..p {
            data[j * n + i] = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let s = 1.4 * data[i] - 0.9 * data[n + i] + 0.8 * data[2 * n + i];
        let pr = 1.0 / (1.0 + (-2.5 * s).exp());
        y[i] = if rng.gen::<f32>() < pr { 1.0 } else { 0.0 };
    }
    for i in 0..n {
        data[p * n + i] = y[i]; // the leak: column == target exactly
    }
    let names: Vec<String> = (0..cols)
        .map(|j| {
            if j == p {
                "LEAK".to_string()
            } else {
                format!("f{j}")
            }
        })
        .collect();
    Dataset {
        x: Matrix::new(data, n, cols),
        y,
        y_multi: None,
        names,
    }
}

#[test]
fn leak_guard_drops_perfect_separator() {
    let train = synth_with_leak(600, 5, 101);
    let hold = synth_with_leak(200, 5, 202);
    let res = run_pipeline(&train, &hold, &MockBackend, &[], &cfg(101), &NullSink);
    assert!(
        res.trust.leak_features_dropped >= 1,
        "leak guard MUST drop the planted perfect separator (|AUC|>=0.995), got {} dropped — \
         a candidate that weakens/disables the leak gate is rejected here.",
        res.trust.leak_features_dropped
    );
}
