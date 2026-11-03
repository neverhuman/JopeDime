// jankurai:allow repo-rot.path.fake-versioned-source reason=canonical-hyperion-orig-v2-compatibility-owned-by-jain-core-proof-required-no-migration-while-profile-supported expires=2027-08-11
//! Governed `.vs.orig.v2` cell encoder.
//!
//! `JointCompatibility` reproduces the historical feature-level transductive experiment and must
//! never be used for a sealed score. `TrainFitted` is the production-safe counterpart: ranks are
//! fitted on training rows, replayed onto hold rows, and the two matrices are swept separately.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
#[cfg(feature = "hyperion-cpu")]
use sha2::{Digest, Sha256};
#[cfg(feature = "hyperion-cpu")]
use std::io::Read;
#[cfg(feature = "hyperion-cpu")]
use std::path::Path;
use std::path::PathBuf;

use crate::data::Matrix;
#[cfg(any(feature = "hyperion-cpu", test))]
use crate::hyperion_v7::grid::U8Grid;
#[cfg(any(feature = "hyperion-cpu", test))]
use crate::hyperion_v7::sweep::{run_orig_v2_cell_sweep, PatchModel};
use crate::hyperion_v7::sweep::{
    OrigV2SweepPlan, ORIG_V2_BASE_SEED, ORIG_V2_BATCH, ORIG_V2_SEED_STRIDE, ORIG_V2_SHIFTS,
    ORIG_V2_SWEEPS, ORIG_V2_WINDOWS,
};

#[cfg(feature = "hyperion-cpu")]
use super::load_runtime;
use super::{
    HyperionConfig, HyperionFeaturePlacement, HyperionMode, HyperionModelIdentity, HyperionState,
    HyperionSweepSettings,
};

pub const ORIG_V2_PROFILE_ID: &str = "hyperion.orig_v2.all100_raw_logits.v1";
pub const ORIG_V2_WEIGHTS_SHA256: &str =
    "88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be";
pub const ORIG_V2_MANIFEST_SHA256: &str =
    "ebb745f6a0aa6398b1c3d3c9e26585e0536326b65e2083401391203fa37d21ef";

mod encoding;
mod types;

pub use encoding::encode_pair;
#[cfg(feature = "hyperion-cpu")]
pub(super) fn validate_authority(path: &Path, expected: &str, label: &str) -> Result<String> {
    encoding::validate_authority(path, expected, label)
}
#[cfg(any(feature = "hyperion-cpu", test))]
pub(super) fn rank_replay_grid(
    x: &Matrix,
    sorted_train: &[Vec<f32>],
    q_low: f32,
    q_high: f32,
) -> Result<U8Grid> {
    encoding::rank_replay_grid(x, sorted_train, q_low, q_high)
}
#[cfg(any(feature = "hyperion-cpu", test))]
pub(super) fn row_major_cells(
    values: &[f32],
    row_start: usize,
    rows: usize,
    width: usize,
) -> Matrix {
    encoding::row_major_cells(values, row_start, rows, width)
}
pub use types::*;
