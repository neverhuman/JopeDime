//! Encodes a small table with the served Hyperion bundle when `HYPERION_V7_WEIGHTS` names its
//! directory (`model.safetensors` + `manifest.json`). Without the variable the test prints why it
//! skipped and returns — it is never a silent pass, and the README names the gate.

use hyperion::data::Matrix;
use hyperion::hyperion::{
    default_manifest_path, default_weights_path, fit_transform_pair, HyperionConfig, HyperionMode,
    DEFAULT_OUTPUT_DIM, WEIGHTS_DIR_ENV,
};
use hyperion::progress::NullSink;
use std::io::Write;
use std::time::Instant;

#[test]
fn served_bundle_encodes_a_small_table() {
    let Some(dir) = std::env::var_os(WEIGHTS_DIR_ENV).filter(|value| !value.is_empty()) else {
        // Written straight to the process stderr so libtest's output capture cannot hide it: the
        // skip is visible in a plain `cargo test` run, not only under `--nocapture`.
        let _ = writeln!(
            std::io::stderr(),
            "SKIPPED served_bundle_encodes_a_small_table: {WEIGHTS_DIR_ENV} is not set \
             (point it at a directory holding model.safetensors + manifest.json)"
        );
        return;
    };
    let cfg = HyperionConfig {
        mode: HyperionMode::On,
        weights_path: default_weights_path(),
        manifest_path: default_manifest_path(),
        sweeps: 2,
        ..HyperionConfig::default()
    };
    assert!(
        cfg.weights_path.starts_with(&dir) && cfg.manifest_path.starts_with(&dir),
        "bundle paths must resolve under {WEIGHTS_DIR_ENV}"
    );
    let (n, c) = (24usize, 3usize);
    let data: Vec<f32> = (0..n * c)
        .map(|i| ((i as f32) * 0.37).sin() + (i % 5) as f32 * 0.1)
        .collect();
    let train = Matrix::new(data, n, c);
    let hold = Matrix::new(train.data[..2 * c].to_vec(), 2, c);
    let names: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let aug = fit_transform_pair(&train, &hold, &names, &cfg, &NullSink, Instant::now())
        .expect("Hyperion On with the served bundle must encode");
    assert!(aug.state.is_some(), "Hyperion did not produce state");
    assert_eq!(aug.train.c, c + DEFAULT_OUTPUT_DIM);
    assert_eq!(aug.hold.c, c + DEFAULT_OUTPUT_DIM);
    assert_eq!(aug.train.n, n);
    assert_eq!(aug.hold.n, 2);
    assert!(aug.train.data.iter().all(|v| v.is_finite()));
    assert!(aug.hold.data.iter().all(|v| v.is_finite()));
}
