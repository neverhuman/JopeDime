//! Hyperion v7 matrix-sweep feature injection.
//!
//! This module is intentionally data-layer shaped: it consumes the already encoded feature matrix
//! only, fits train-only quantile/sweep state, places a fixed 128-column block, and persists enough
//! state for scoring replay. Standard gated training prepends the block; direct loader and old
//! artifact replay keep the historical right-append layout. The target never enters this API.

#[cfg(feature = "hyperion-cuda")]
use crate::battle_gpu;
use crate::data::Matrix;
use crate::hashing::mix64;
use crate::progress::{
    Event, HyperionConcernRange, HyperionConcernRow, HyperionPreview, HyperionPreviewPoint, Sink,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub mod hvc2;
pub mod orig_v2;

pub const HYPERION_VERSION: &str = "v7";
pub const DEFAULT_OUTPUT_DIM: usize = 128;
pub const DEFAULT_SEED: u64 = 0x4859_5045_5249_4f4e;
// DEFAULT = the v5-128 arm B encoder (landmark-distill, retrained) — the validated better-routing
// model. It is a v5-geometry checkpoint (dim128, 4 layers, 4 bands, ws {4,7,11,15}); the manifest
// carries version:"v7" only to satisfy the loader's version gate.
// The served bundle is a directory holding `model.safetensors` + `manifest.json`: env
// `HYPERION_V7_WEIGHTS` names it, otherwise `$JOPE_HOME/models/bundles/hyperion-v7` (relative to
// the working directory when `JOPE_HOME` is unset). `JAIN_HYPERION_WEIGHTS` / `JAIN_HYPERION_MANIFEST`
// still override the individual files (see `HyperionConfig::from_env`).
pub const WEIGHTS_DIR_ENV: &str = "HYPERION_V7_WEIGHTS";
pub const JOPE_HOME_ENV: &str = "JOPE_HOME";
pub const DEFAULT_BUNDLE_SUBDIR: &str = "models/bundles/hyperion-v7";
pub const DEFAULT_WEIGHTS_FILE: &str = "model.safetensors";
pub const DEFAULT_MANIFEST_FILE: &str = "manifest.json";

/// The bundle directory: `$HYPERION_V7_WEIGHTS`, else `$JOPE_HOME/models/bundles/hyperion-v7`.
pub fn default_weights_dir() -> PathBuf {
    let non_empty = |key: &str| {
        std::env::var_os(key)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    if let Some(dir) = non_empty(WEIGHTS_DIR_ENV) {
        return dir;
    }
    match non_empty(JOPE_HOME_ENV) {
        Some(home) => home.join(DEFAULT_BUNDLE_SUBDIR),
        None => PathBuf::from(DEFAULT_BUNDLE_SUBDIR),
    }
}

/// `<bundle dir>/model.safetensors`.
pub fn default_weights_path() -> PathBuf {
    default_weights_dir().join(DEFAULT_WEIGHTS_FILE)
}

/// `<bundle dir>/manifest.json`.
pub fn default_manifest_path() -> PathBuf {
    default_weights_dir().join(DEFAULT_MANIFEST_FILE)
}
const PREVIEW_MAX_POINTS: usize = 160;
const PREVIEW_TOP_ROWS: usize = 8;
const CONCERN_Z_THRESHOLD: f64 = 3.5;

include!("hyperion/config.rs");
include!("hyperion/encoder.rs");
include!("hyperion/runtime.rs");
include!("hyperion/tests.rs");
