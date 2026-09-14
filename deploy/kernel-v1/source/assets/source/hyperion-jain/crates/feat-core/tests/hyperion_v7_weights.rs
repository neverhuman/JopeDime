//! Truth tests against the REAL v7 checkpoint. They skip gracefully when the weights file is not
//! present (CI), and run for real on the dev box (`cargo test -p feat-core --features hyperion-cpu`).
#![cfg(feature = "hyperion-cpu")]

use std::path::Path;
use std::sync::Mutex;

use feat_core::data::Matrix;
use feat_core::hyperion::{
    fit_transform_pair, transform_scoring_matrix, HyperionConfig, HyperionEngine, HyperionMode,
    HyperionSweepSettings,
};
use feat_core::hyperion_v7::backend::V7Model;
use feat_core::hyperion_v7::config::SweepPlan;
use feat_core::hyperion_v7::grid::U8Grid;
use feat_core::hyperion_v7::sweep::{run_composite_sweep, PatchModel};
use feat_core::hyperion_v7::EMB_KEEP;
use feat_core::progress::{Event, NullSink, Sink, SweepFrame};

const WEIGHTS: &str = "/home/ubuntu/hyperion/weights/v7/model.safetensors";
const MANIFEST: &str = "/home/ubuntu/hyperion/weights/v7/manifest.json";

fn load() -> Option<(V7Model, HyperionSweepSettings)> {
    if !Path::new(WEIGHTS).is_file() || !Path::new(MANIFEST).is_file() {
        eprintln!("skipping: v7 weights not present on this machine");
        return None;
    }
    // The manifest omits `seed` (hyperion.rs's loader injects it) — inject a fixed one here.
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(MANIFEST).unwrap()).unwrap();
    manifest["seed"] = serde_json::json!(0u64);
    let settings: HyperionSweepSettings = serde_json::from_value(manifest).unwrap();
    let model = V7Model::load(Path::new(WEIGHTS), &settings).unwrap();
    Some((model, settings))
}

/// A deterministic p×p canvas pair (values in model range, mask all-ones).
fn canvas(p: usize, quant_scale: f32, salt: u32) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..p * p)
        .map(|i| {
            let u = ((i as u32 * 37 + salt * 101 + 11) % 256) as f32;
            ((u / 255.0) * 2.0 - 1.0) * quant_scale
        })
        .collect();
    (x, vec![1.0; p * p])
}

#[test]
fn forward_shapes_are_finite() {
    let Some((model, settings)) = load() else {
        return;
    };
    let p = 3usize;
    let (x1, m1) = canvas(p, settings.quant_scale, 1);
    let (x2, m2) = canvas(p, settings.quant_scale, 2);
    let x: Vec<f32> = [x1, x2].concat();
    let m: Vec<f32> = [m1, m2].concat();
    let out = model.forward_batch(&x, &m, 2, p).unwrap();
    assert_eq!(out.emb.len(), 2 * model.emb_dim());
    assert_eq!(model.emb_dim(), settings.dim);
    assert_eq!(out.logits.len(), 2 * p * p);
    assert!(out.emb.iter().all(|v| v.is_finite()));
    assert!(out.logits.iter().all(|v| v.is_finite()));
    // Non-degenerate: the two different patches must embed differently.
    let d0 = &out.emb[..EMB_KEEP];
    let d1 = &out.emb[model.emb_dim()..model.emb_dim() + EMB_KEEP];
    let diff: f32 = d0.iter().zip(d1).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-3,
        "embeddings should differ across patches (got {diff})"
    );
}

#[test]
fn cropped_canvas_matches_reference_full_canvas() {
    let Some((model, settings)) = load() else {
        return;
    };
    // Compare the cropped p×p canvas against the reference max_patch canvas (masked ops must make
    // them equivalent on the valid window).
    let settings2 = settings.clone();
    let full = V7Model::load(Path::new(WEIGHTS), &settings2)
        .unwrap()
        .with_full_canvas(true);
    for p in [3usize, 5, 9] {
        let (x, m) = canvas(p, settings.quant_scale, p as u32);
        let a = model.forward_batch(&x, &m, 1, p).unwrap();
        let b = full.forward_batch(&x, &m, 1, p).unwrap();
        let emb_delta = a
            .emb
            .iter()
            .zip(&b.emb)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let logit_delta = a
            .logits
            .iter()
            .zip(&b.logits)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(emb_delta < 1e-3, "p={p}: emb parity delta {emb_delta}");
        assert!(
            logit_delta < 1e-3,
            "p={p}: logit parity delta {logit_delta}"
        );
    }
}

/// Captures Events AND SweepFrames — proves the emit_sweep plumbing inside fit_transform_pair.
#[derive(Default)]
struct CaptureAll {
    events: Mutex<Vec<Event>>,
    frames: Mutex<Vec<SweepFrame>>,
}

impl Sink for CaptureAll {
    fn emit(&self, e: &Event) {
        self.events.lock().unwrap().push(e.clone());
    }
    fn emit_sweep(&self, f: &SweepFrame) {
        self.frames.lock().unwrap().push(f.clone());
    }
}

#[test]
fn fit_transform_pair_v7_streams_frames_and_scoring_round_trips() {
    if !Path::new(WEIGHTS).is_file() {
        eprintln!("skipping: v7 weights not present on this machine");
        return;
    }
    let (n, c) = (30usize, 4usize);
    let mk = |rows: usize, salt: u32| -> Matrix {
        let mut data = vec![0.0f32; rows * c];
        for col in 0..c {
            for r in 0..rows {
                data[col * rows + r] =
                    ((r as f32 * 1.3 + col as f32 * 7.7 + salt as f32).sin()) * 10.0;
            }
        }
        Matrix::new(data, rows, c)
    };
    let train = mk(n, 1);
    let hold = mk(8, 2);
    let names: Vec<String> = (0..c).map(|j| format!("f{j}")).collect();

    let cfg = HyperionConfig {
        mode: HyperionMode::On,
        engine: HyperionEngine::V7,
        sweeps: 2, // fast truth test; sweep math is proven candle-free
        scales: vec![3, 5],
        batch: 64,
        ..HyperionConfig::default()
    };
    let sink = CaptureAll::default();
    let aug = fit_transform_pair(
        &train,
        &hold,
        &names,
        &cfg,
        &sink,
        std::time::Instant::now(),
    )
    .unwrap();

    // Feature block appended: 4 base + 128 hyperion columns; state records the v7 plan.
    assert_eq!(aug.train.c, c + 128);
    assert_eq!(aug.hold.c, c + 128);
    let state = aug.state.expect("v7 state");
    assert_eq!(state.engine, "v7");
    assert_eq!(state.sweeps, 2);
    assert_eq!(state.scales, vec![3, 5]);
    // Confidence surface + dataset vector are real and well-formed.
    let conf = aug.confidence.expect("confidence surface");
    assert_eq!((conf.n, conf.c), (n, c));
    assert!(conf.data.iter().all(|v| (0.0..=1.0).contains(v)));
    let hold_conf = aug.hold_confidence.expect("hold confidence surface");
    assert_eq!((hold_conf.n, hold_conf.c), (hold.n, c));
    assert!(hold_conf.data.iter().all(|v| (0.0..=1.0).contains(v)));
    let vector = aug.dataset_vector.expect("dataset vector");
    assert_eq!(vector.len(), EMB_KEEP);

    // Frames streamed through the sink with sane progress; phase-0 done event closed the stream.
    let frames = sink.frames.lock().unwrap();
    assert!(
        frames.len() >= 4,
        "expected streamed frames, got {}",
        frames.len()
    );
    assert!(frames
        .windows(2)
        .all(|w| w[0].done_frac <= w[1].done_frac + 1e-9));
    let events = sink.events.lock().unwrap();
    assert!(events
        .iter()
        .any(|e| e.event == "done" && e.hyperion_status.as_deref() == Some("used")));
    assert!(events.iter().any(|e| {
        e.event == "done"
            && e.hyperion_engine.as_deref() == Some("v7")
            && e.hyperion_device.as_deref().is_some()
    }));
    assert!(events.iter().any(|e| e.event == "block"));

    // Scoring replay through the artifact path appends the same-width block deterministically.
    let scored = transform_scoring_matrix(&mk(8, 2), &names, &state).unwrap();
    assert_eq!(scored.c, c + 128);
    let scored_again = transform_scoring_matrix(&mk(8, 2), &names, &state).unwrap();
    assert_eq!(
        scored.data.as_ref(),
        scored_again.data.as_ref(),
        "scoring is deterministic"
    );
}

#[test]
fn composite_sweep_end_to_end_on_synthetic() {
    let Some((model, settings)) = load() else {
        return;
    };
    let (h, w) = (24usize, 6usize);
    let data: Vec<u8> = (0..h * w).map(|i| ((i * 53 + 7) % 256) as u8).collect();
    let xu8 = U8Grid { data, h, w };
    let plan = SweepPlan {
        sweeps: 2, // keep the real-model test fast; the sweep math is proven candle-free
        scales: vec![3, 5],
        batch: 64,
        seed: 42,
        target_chunks_per_sweep: 3,
        fixed_perms: None,
    };
    let mut frames = 0usize;
    let mut last_vector: Vec<f32> = Vec::new();
    let out = run_composite_sweep(
        &model,
        &xu8,
        settings.pad_u8,
        settings.quant_scale,
        settings.max_patch,
        &plan,
        &NullSink,
        |report| {
            frames += 1;
            last_vector = report.accum.dataset_vector();
        },
    )
    .unwrap();
    assert!(!out.cancelled);
    assert!(
        frames >= 4,
        "expected chunked streaming, got {frames} chunks"
    );
    assert!(out.confidence.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert!(out.row_emb.iter().all(|v| v.is_finite()));
    // cell-map is dense, normalized to [0,1] over covered cells; row_emb is its col-mean.
    assert_eq!(out.cellmap.len(), h * w * EMB_KEEP);
    assert!(out.cellmap.iter().all(|&v| (0.0..=1.0).contains(&v)));
    // The last streamed vector is the live (unnormalized) preview: well-formed + finite (it is
    // intentionally NOT bit-equal to the final normalized `dataset_vec`).
    assert_eq!(last_vector.len(), EMB_KEEP);
    assert!(last_vector.iter().all(|v| v.is_finite()));
    assert_eq!(out.dataset_vec.len(), EMB_KEEP);
    // Confidence surface must be non-degenerate (the head actually discriminates cells).
    let (mn, mx) = out
        .confidence
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    assert!(mx - mn > 1e-4, "confidence surface is flat ({mn}..{mx})");
}
