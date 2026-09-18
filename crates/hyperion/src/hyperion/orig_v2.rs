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

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrigV2PairSemantics {
    /// Historical experiment only: concatenate train then test features before ranking/sweeping.
    JointCompatibility,
    /// Production-safe: fit ranks on train, replay them onto hold, and sweep separately.
    TrainFitted,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OrigV2Config {
    pub weights_path: PathBuf,
    pub manifest_path: PathBuf,
    pub sweeps: u32,
    pub windows: Vec<usize>,
    pub shifts: Vec<(usize, usize)>,
    pub base_seed: u64,
    pub seed_stride: u64,
    pub batch: usize,
    pub require_gpu: bool,
    pub require_pinned_model: bool,
}

impl Default for OrigV2Config {
    fn default() -> Self {
        let hyperion = HyperionConfig::default();
        Self {
            weights_path: hyperion.weights_path,
            manifest_path: hyperion.manifest_path,
            sweeps: ORIG_V2_SWEEPS,
            windows: ORIG_V2_WINDOWS.to_vec(),
            shifts: ORIG_V2_SHIFTS.to_vec(),
            base_seed: ORIG_V2_BASE_SEED,
            seed_stride: ORIG_V2_SEED_STRIDE,
            batch: ORIG_V2_BATCH,
            require_gpu: false,
            require_pinned_model: true,
        }
    }
}

impl OrigV2Config {
    pub fn from_hyperion_config(config: &HyperionConfig) -> Self {
        Self {
            weights_path: config.weights_path.clone(),
            manifest_path: config.manifest_path.clone(),
            require_gpu: config.require_gpu,
            ..Self::default()
        }
    }

    #[cfg(any(feature = "hyperion-cpu", test))]
    fn plan(&self) -> OrigV2SweepPlan {
        OrigV2SweepPlan {
            sweeps: self.sweeps,
            windows: self.windows.clone(),
            shifts: self.shifts.clone(),
            batch: self.batch,
            base_seed: self.base_seed,
            seed_stride: self.seed_stride,
        }
    }

    #[cfg(any(feature = "hyperion-cpu", test))]
    fn validate_profile(&self) -> Result<()> {
        if self.sweeps != ORIG_V2_SWEEPS
            || self.windows != ORIG_V2_WINDOWS
            || self.shifts != ORIG_V2_SHIFTS
            || self.base_seed != ORIG_V2_BASE_SEED
            || self.seed_stride != ORIG_V2_SEED_STRIDE
            || self.batch != ORIG_V2_BATCH
        {
            bail!("Hyperion Orig V2 configuration drifts from {ORIG_V2_PROFILE_ID}");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrigV2Receipt {
    pub profile: String,
    pub pair_semantics: OrigV2PairSemantics,
    pub sweeps: u32,
    pub windows: Vec<usize>,
    pub shifts: Vec<(usize, usize)>,
    pub base_seed: u64,
    pub seed_stride: u64,
    pub batch: usize,
    pub raw_logits: bool,
    pub collapse: String,
    pub model: HyperionModelIdentity,
    pub weights_sha256: String,
    pub manifest_sha256: String,
    pub sweep: HyperionSweepSettings,
    pub device: String,
}

impl OrigV2Receipt {
    pub(super) fn validate_frozen(&self, required_semantics: OrigV2PairSemantics) -> Result<()> {
        if self.profile != ORIG_V2_PROFILE_ID
            || self.pair_semantics != required_semantics
            || self.sweeps != ORIG_V2_SWEEPS
            || self.windows != ORIG_V2_WINDOWS
            || self.shifts != ORIG_V2_SHIFTS
            || self.base_seed != ORIG_V2_BASE_SEED
            || self.seed_stride != ORIG_V2_SEED_STRIDE
            || self.batch != ORIG_V2_BATCH
            || !self.raw_logits
            || self.collapse != "arithmetic_mean(shifts,windows,sweeps)"
        {
            bail!("Hyperion Orig V2 receipt drifts from the frozen profile");
        }
        if self.weights_sha256 != ORIG_V2_WEIGHTS_SHA256
            || self.manifest_sha256 != ORIG_V2_MANIFEST_SHA256
            || self.model.weights_hash != ORIG_V2_WEIGHTS_SHA256
            || self.model.manifest_hash != ORIG_V2_MANIFEST_SHA256
        {
            bail!("Hyperion Orig V2 receipt model hashes do not match pinned authority");
        }
        if self.model.weights_path.trim().is_empty()
            || self.model.manifest_path.trim().is_empty()
            || self.model.weights_bytes == 0
            || self.model.tensor_count == 0
            || self.device.trim().is_empty()
        {
            bail!("Hyperion Orig V2 receipt model/device identity is incomplete");
        }
        if self.sweep.dim == 0
            || self.sweep.heads == 0
            || self.sweep.layers == 0
            || self.sweep.emb_dim == 0
            || self.sweep.min_patch == 0
            || self.sweep.max_patch < self.sweep.min_patch
            || self.sweep.n_bands == 0
            || self.sweep.ws_list.is_empty()
            || !self.sweep.q_low.is_finite()
            || !self.sweep.q_high.is_finite()
            || self.sweep.q_low > self.sweep.q_high
            || !self.sweep.quant_scale.is_finite()
            || self.sweep.quant_scale <= 0.0
            || self.sweep.seed != ORIG_V2_BASE_SEED
        {
            bail!("Hyperion Orig V2 receipt sweep identity is invalid");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrigV2ArtifactState {
    pub sorted_fit_columns: Vec<Vec<f32>>,
    pub sweeps: u32,
    pub windows: Vec<usize>,
    pub shifts: Vec<(usize, usize)>,
    pub base_seed: u64,
    pub seed_stride: u64,
    pub batch: usize,
}

impl OrigV2ArtifactState {
    pub fn plan(&self) -> OrigV2SweepPlan {
        OrigV2SweepPlan {
            sweeps: self.sweeps,
            windows: self.windows.clone(),
            shifts: self.shifts.clone(),
            batch: self.batch,
            base_seed: self.base_seed,
            seed_stride: self.seed_stride,
        }
    }
}

#[derive(Clone)]
pub struct OrigV2PairOutput {
    pub train_cells: Matrix,
    pub hold_cells: Matrix,
    pub receipt: OrigV2Receipt,
    /// Present only for the production-safe train-fitted profile. It contains feature values only.
    pub artifact_state: Option<OrigV2ArtifactState>,
}

/// Reusable Orig V2 encoder for sharded/offline workloads.
///
/// Loading validates the frozen profile and model authority once. Every subsequent pair reuses
/// the same resident model while fitting rank state independently from that pair's training rows.
pub struct OrigV2Encoder {
    config: OrigV2Config,
    #[cfg(feature = "hyperion-cpu")]
    model: crate::hyperion_v7::backend::V7Model,
    #[cfg(feature = "hyperion-cpu")]
    settings: HyperionSweepSettings,
    #[cfg(feature = "hyperion-cpu")]
    identity: HyperionModelIdentity,
    #[cfg(feature = "hyperion-cpu")]
    weights_sha256: String,
    #[cfg(feature = "hyperion-cpu")]
    manifest_sha256: String,
    #[cfg(feature = "hyperion-cpu")]
    device: String,
}

#[cfg(feature = "hyperion-cpu")]
impl OrigV2Encoder {
    pub fn load(config: &OrigV2Config) -> Result<Self> {
        use crate::hyperion_v7::backend::V7Model;

        config.validate_profile()?;
        let weights_sha256 = if config.require_pinned_model {
            validate_authority(&config.weights_path, ORIG_V2_WEIGHTS_SHA256, "weights")?
        } else {
            sha256_file(&config.weights_path, "weights")?
        };
        let manifest_sha256 = if config.require_pinned_model {
            validate_authority(&config.manifest_path, ORIG_V2_MANIFEST_SHA256, "manifest")?
        } else {
            sha256_file(&config.manifest_path, "manifest")?
        };
        let base = HyperionConfig {
            weights_path: config.weights_path.clone(),
            manifest_path: config.manifest_path.clone(),
            seed: config.base_seed,
            ..HyperionConfig::default()
        };
        let runtime = load_runtime(&base, super::DEFAULT_OUTPUT_DIM)?;
        let model = V7Model::load(&config.weights_path, &runtime.settings)?;
        let device = model.device_label().to_string();
        if config.require_gpu && !device.starts_with("cuda") && !device.starts_with("gpu") {
            bail!("Hyperion orig-v2 require_gpu requested but model loaded on {device}");
        }
        Ok(Self {
            config: config.clone(),
            model,
            settings: runtime.settings,
            identity: runtime.identity,
            weights_sha256,
            manifest_sha256,
            device,
        })
    }

    pub fn encode_pair(
        &self,
        train: &Matrix,
        hold: &Matrix,
        semantics: OrigV2PairSemantics,
    ) -> Result<OrigV2PairOutput> {
        encode_pair_with_model(
            &self.model,
            &self.settings,
            &self.identity,
            &self.weights_sha256,
            &self.manifest_sha256,
            &self.device,
            train,
            hold,
            semantics,
            &self.config,
        )
    }

    pub fn device_label(&self) -> &str {
        &self.device
    }

    pub fn model_hashes(&self) -> (&str, &str) {
        (&self.weights_sha256, &self.manifest_sha256)
    }
}

#[cfg(not(feature = "hyperion-cpu"))]
impl OrigV2Encoder {
    pub fn load(_config: &OrigV2Config) -> Result<Self> {
        bail!("Hyperion orig-v2 requires the 'hyperion-cpu' or 'hyperion-cuda' feature")
    }

    pub fn encode_pair(
        &self,
        _train: &Matrix,
        _hold: &Matrix,
        _semantics: OrigV2PairSemantics,
    ) -> Result<OrigV2PairOutput> {
        let _ = &self.config;
        bail!("Hyperion orig-v2 requires the 'hyperion-cpu' or 'hyperion-cuda' feature")
    }
}

impl OrigV2PairOutput {
    pub fn train_fitted_hyperion_state(
        &self,
        input_names: &[String],
        requested_mode: HyperionMode,
    ) -> Result<HyperionState> {
        let artifact_state = self
            .artifact_state
            .clone()
            .context("joint compatibility output cannot be serialized as a production artifact")?;
        if self.train_cells.c != input_names.len() {
            bail!("Hyperion orig-v2 state width does not match input names");
        }
        Ok(HyperionState {
            version: ORIG_V2_PROFILE_ID.into(),
            requested_mode,
            output_names: (0..input_names.len())
                .map(|index| format!("hyperion::cell::{index:05}"))
                .collect(),
            input_names: input_names.to_vec(),
            input_indices: (0..input_names.len()).collect(),
            quantile_bounds: Vec::new(),
            sweep: self.receipt.sweep.clone(),
            output_dim: input_names.len(),
            model: self.receipt.model.clone(),
            engine: "orig_v2".into(),
            sweeps: self.receipt.sweeps as usize,
            scales: self.receipt.windows.clone(),
            batch: self.receipt.batch,
            feature_placement: HyperionFeaturePlacement::AppendRight,
            orig_v2: Some(artifact_state),
        })
    }
}

#[cfg(any(feature = "hyperion-cpu", test))]
fn clean(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[cfg(any(feature = "hyperion-cpu", test))]
fn quantile_position(n: usize, q: f32) -> f32 {
    1.0 + (q.clamp(0.0, 100.0) / 100.0) * n.saturating_sub(1) as f32
}

#[cfg(any(feature = "hyperion-cpu", test))]
fn ordinal_to_u8(ordinal: f32, n: usize, q_low: f32, q_high: f32) -> u8 {
    let low = quantile_position(n, q_low);
    let high = quantile_position(n, q_high);
    let span = (high - low).max(1.0e-6);
    (((ordinal - low) / span).clamp(0.0, 1.0) * 255.0) as u8
}

#[cfg(any(feature = "hyperion-cpu", test))]
fn rank_fit_grid(x: &Matrix, q_low: f32, q_high: f32) -> (U8Grid, Vec<Vec<f32>>) {
    let mut data = vec![0u8; x.n * x.c];
    let mut sorted_columns = Vec::with_capacity(x.c);
    for col in 0..x.c {
        let values: Vec<f32> = x.col(col).iter().copied().map(clean).collect();
        let mut order: Vec<usize> = (0..x.n).collect();
        // Historical argsort(argsort) assigned every row a unique ordinal. Index is the explicit
        // deterministic tie break for the Rust authority.
        order.sort_by(|&left, &right| {
            values[left]
                .total_cmp(&values[right])
                .then_with(|| left.cmp(&right))
        });
        for (rank, &row) in order.iter().enumerate() {
            data[row * x.c + col] = ordinal_to_u8(rank as f32 + 1.0, x.n, q_low, q_high);
        }
        let mut sorted = values;
        sorted.sort_by(f32::total_cmp);
        sorted_columns.push(sorted);
    }
    (
        U8Grid {
            data,
            h: x.n,
            w: x.c,
        },
        sorted_columns,
    )
}

#[cfg(any(feature = "hyperion-cpu", test))]
pub(super) fn rank_replay_grid(
    x: &Matrix,
    sorted_train: &[Vec<f32>],
    q_low: f32,
    q_high: f32,
) -> Result<U8Grid> {
    if x.c != sorted_train.len() {
        bail!("Hyperion orig-v2 rank replay width mismatch");
    }
    let n_fit = sorted_train.first().map_or(0, Vec::len);
    if n_fit == 0 {
        bail!("Hyperion orig-v2 cannot replay ranks from an empty fit matrix");
    }
    let mut data = vec![0u8; x.n * x.c];
    for (col, sorted) in sorted_train.iter().enumerate() {
        for row in 0..x.n {
            let value = clean(x.col(col)[row]);
            let low = sorted.partition_point(|item| *item < value);
            let high = sorted.partition_point(|item| *item <= value);
            // Midrank in one-based ordinal coordinates. Values outside the fit range naturally
            // clamp through ordinal_to_u8.
            let ordinal = 0.5 + (low + high) as f32 * 0.5;
            data[row * x.c + col] = ordinal_to_u8(ordinal, n_fit, q_low, q_high);
        }
    }
    Ok(U8Grid {
        data,
        h: x.n,
        w: x.c,
    })
}

#[cfg(any(feature = "hyperion-cpu", test))]
fn join_rows(train: &Matrix, hold: &Matrix) -> Result<Matrix> {
    if train.c != hold.c {
        bail!(
            "Hyperion orig-v2 pair width mismatch: train {} hold {}",
            train.c,
            hold.c
        );
    }
    let rows = train.n + hold.n;
    let mut data = vec![0.0f32; rows * train.c];
    for col in 0..train.c {
        let offset = col * rows;
        data[offset..offset + train.n]
            .copy_from_slice(&train.data[col * train.n..(col + 1) * train.n]);
        data[offset + train.n..offset + rows]
            .copy_from_slice(&hold.data[col * hold.n..(col + 1) * hold.n]);
    }
    Ok(Matrix::new(data, rows, train.c))
}

#[cfg(any(feature = "hyperion-cpu", test))]
pub(super) fn row_major_cells(
    values: &[f32],
    row_start: usize,
    rows: usize,
    width: usize,
) -> Matrix {
    let mut data = vec![0.0f32; rows * width];
    for row in 0..rows {
        for col in 0..width {
            data[col * rows + row] = values[(row_start + row) * width + col];
        }
    }
    Matrix::new(data, rows, width)
}

#[cfg(feature = "hyperion-cpu")]
fn sha256_file(path: &Path, label: &str) -> Result<String> {
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {label} {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hashing {label} {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    let actual = format!("{:x}", digest.finalize());
    Ok(actual)
}

#[cfg(feature = "hyperion-cpu")]
pub(super) fn validate_authority(path: &Path, expected: &str, label: &str) -> Result<String> {
    let actual = sha256_file(path, label)?;
    if actual != expected {
        bail!("Hyperion orig-v2 {label} hash mismatch: {actual} != {expected}");
    }
    Ok(actual)
}

#[cfg(feature = "hyperion-cpu")]
pub fn encode_pair(
    train: &Matrix,
    hold: &Matrix,
    semantics: OrigV2PairSemantics,
    config: &OrigV2Config,
) -> Result<OrigV2PairOutput> {
    OrigV2Encoder::load(config)?.encode_pair(train, hold, semantics)
}

#[cfg(any(feature = "hyperion-cpu", test))]
#[allow(clippy::too_many_arguments)]
fn encode_pair_with_model<M: PatchModel>(
    model: &M,
    settings: &HyperionSweepSettings,
    identity: &HyperionModelIdentity,
    weights_sha256: &str,
    manifest_sha256: &str,
    device: &str,
    train: &Matrix,
    hold: &Matrix,
    semantics: OrigV2PairSemantics,
    config: &OrigV2Config,
) -> Result<OrigV2PairOutput> {
    config.validate_profile()?;
    if train.n == 0 || train.c == 0 || hold.n == 0 || train.c != hold.c {
        bail!("Hyperion orig-v2 requires non-empty equal-width train/hold matrices");
    }
    let plan = config.plan();

    let (train_cells, hold_cells, artifact_state) = match semantics {
        OrigV2PairSemantics::JointCompatibility => {
            let joint = join_rows(train, hold)?;
            let (grid, _) = rank_fit_grid(&joint, settings.q_low, settings.q_high);
            let output = run_orig_v2_cell_sweep(
                model,
                &grid,
                settings.quant_scale,
                settings.max_patch,
                &plan,
            )?;
            (
                row_major_cells(&output.logits, 0, train.n, train.c),
                row_major_cells(&output.logits, train.n, hold.n, hold.c),
                None,
            )
        }
        OrigV2PairSemantics::TrainFitted => {
            let (train_grid, sorted) = rank_fit_grid(train, settings.q_low, settings.q_high);
            let hold_grid = rank_replay_grid(hold, &sorted, settings.q_low, settings.q_high)?;
            let train_output = run_orig_v2_cell_sweep(
                model,
                &train_grid,
                settings.quant_scale,
                settings.max_patch,
                &plan,
            )?;
            let hold_output = run_orig_v2_cell_sweep(
                model,
                &hold_grid,
                settings.quant_scale,
                settings.max_patch,
                &plan,
            )?;
            (
                row_major_cells(&train_output.logits, 0, train.n, train.c),
                row_major_cells(&hold_output.logits, 0, hold.n, hold.c),
                Some(OrigV2ArtifactState {
                    sorted_fit_columns: sorted,
                    sweeps: config.sweeps,
                    windows: config.windows.clone(),
                    shifts: config.shifts.clone(),
                    base_seed: config.base_seed,
                    seed_stride: config.seed_stride,
                    batch: config.batch,
                }),
            )
        }
    };

    Ok(OrigV2PairOutput {
        train_cells,
        hold_cells,
        artifact_state,
        receipt: OrigV2Receipt {
            profile: ORIG_V2_PROFILE_ID.into(),
            pair_semantics: semantics,
            sweeps: config.sweeps,
            windows: config.windows.clone(),
            shifts: config.shifts.clone(),
            base_seed: config.base_seed,
            seed_stride: config.seed_stride,
            batch: config.batch,
            raw_logits: true,
            collapse: "arithmetic_mean(shifts,windows,sweeps)".into(),
            model: identity.clone(),
            weights_sha256: weights_sha256.into(),
            manifest_sha256: manifest_sha256.into(),
            sweep: settings.clone(),
            device: device.into(),
        },
    })
}

#[cfg(not(feature = "hyperion-cpu"))]
pub fn encode_pair(
    _train: &Matrix,
    _hold: &Matrix,
    _semantics: OrigV2PairSemantics,
    _config: &OrigV2Config,
) -> Result<OrigV2PairOutput> {
    bail!("Hyperion orig-v2 requires the 'hyperion-cpu' or 'hyperion-cuda' feature")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperion_v7::sweep::PatchOutputs;

    struct FakeModel;

    impl PatchModel for FakeModel {
        fn forward_batch(
            &self,
            x: &[f32],
            _m: &[f32],
            batch: usize,
            p: usize,
        ) -> Result<PatchOutputs> {
            Ok(PatchOutputs {
                emb: vec![0.0; batch],
                logits: x
                    .iter()
                    .map(|value| value.mul_add(0.25, p as f32))
                    .collect(),
            })
        }

        fn emb_dim(&self) -> usize {
            1
        }
    }

    fn fake_settings() -> HyperionSweepSettings {
        HyperionSweepSettings {
            version: "v7".into(),
            step: 0,
            dim: 128,
            layers: 1,
            heads: 1,
            emb_dim: 1,
            n_bands: 1,
            min_patch: 2,
            max_patch: 12,
            ws_list: vec![3, 6, 9, 12],
            q_low: 1.0,
            q_high: 99.0,
            pad_u8: 0,
            quant_scale: 1.0,
            seed: 42,
        }
    }

    fn fake_identity() -> HyperionModelIdentity {
        HyperionModelIdentity {
            weights_path: "weights".into(),
            manifest_path: "manifest".into(),
            weights_bytes: 1,
            weights_hash: ORIG_V2_WEIGHTS_SHA256.into(),
            manifest_hash: ORIG_V2_MANIFEST_SHA256.into(),
            tensor_count: 1,
        }
    }

    fn fake_encode(train: &Matrix, hold: &Matrix) -> OrigV2PairOutput {
        encode_pair_with_model(
            &FakeModel,
            &fake_settings(),
            &fake_identity(),
            ORIG_V2_WEIGHTS_SHA256,
            ORIG_V2_MANIFEST_SHA256,
            "test",
            train,
            hold,
            OrigV2PairSemantics::TrainFitted,
            &OrigV2Config::default(),
        )
        .unwrap()
    }

    #[test]
    fn rank_fit_has_old_unique_rank_then_robust_u8_contract() {
        let x = Matrix::new(vec![9.0, 1.0, 1.0, 5.0], 4, 1);
        let (grid, sorted) = rank_fit_grid(&x, 1.0, 99.0);
        assert_eq!(sorted, vec![vec![1.0, 1.0, 5.0, 9.0]]);
        assert_eq!(grid.h, 4);
        assert_eq!(grid.w, 1);
        // Index breaks the tied 1.0 values deterministically, just as argsort(argsort) produced
        // unique ordinals rather than tie-averaged ranks.
        assert!(grid.data[1] < grid.data[2]);
        assert_eq!(grid.data[0], 255);
    }

    #[test]
    fn train_fitted_rank_replay_uses_only_fit_distribution() {
        let train = Matrix::new(vec![0.0, 10.0, 20.0, 30.0], 4, 1);
        let hold = Matrix::new(vec![-5.0, 15.0, 35.0], 3, 1);
        let (_, sorted) = rank_fit_grid(&train, 1.0, 99.0);
        let replay = rank_replay_grid(&hold, &sorted, 1.0, 99.0).unwrap();
        assert_eq!(replay.data[0], 0);
        assert!(replay.data[1] > 0 && replay.data[1] < 255);
        assert_eq!(replay.data[2], 255);
    }

    #[test]
    fn joint_rows_preserve_train_then_hold_order() {
        let train = Matrix::new(vec![1.0, 2.0, 10.0, 20.0], 2, 2);
        let hold = Matrix::new(vec![3.0, 30.0], 1, 2);
        let joint = join_rows(&train, &hold).unwrap();
        assert_eq!((joint.n, joint.c), (3, 2));
        assert_eq!(joint.data.as_ref(), &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn defaults_pin_the_correct_all100_profile() {
        let config = OrigV2Config::default();
        assert_eq!(config.sweeps, 100);
        assert_eq!(config.windows, vec![3, 6, 9, 12]);
        assert_eq!(config.shifts, vec![(0, 0), (1, 1)]);
        assert_eq!(config.base_seed, 42);
        assert_eq!(config.seed_stride, 777);
        assert_eq!(config.batch, 4096);
        assert!(config.require_pinned_model);
    }

    #[test]
    fn reusable_encoder_path_is_deterministic() {
        let train = Matrix::new(vec![1.0, 3.0, 2.0, 4.0], 2, 2);
        let hold = Matrix::new(vec![5.0, 6.0], 1, 2);
        let one_shot = fake_encode(&train, &hold);
        let reused = fake_encode(&train, &hold);
        assert_eq!(one_shot.train_cells.data, reused.train_cells.data);
        assert_eq!(one_shot.hold_cells.data, reused.hold_cells.data);
        assert_eq!(one_shot.receipt, reused.receipt);
    }

    #[test]
    fn train_fitted_train_cache_is_invariant_to_test_values() {
        let train = Matrix::new(vec![1.0, 3.0, 2.0, 4.0], 2, 2);
        let first_hold = Matrix::new(vec![5.0, 6.0], 1, 2);
        let second_hold = Matrix::new(vec![-100.0, 100.0], 1, 2);
        let first = fake_encode(&train, &first_hold);
        let second = fake_encode(&train, &second_hold);
        assert_eq!(first.train_cells.data, second.train_cells.data);
    }

    #[test]
    fn frozen_receipt_validation_rejects_profile_drift() {
        let train = Matrix::new(vec![1.0, 2.0], 2, 1);
        let hold = Matrix::new(vec![3.0], 1, 1);
        let mut receipt = fake_encode(&train, &hold).receipt;
        receipt
            .validate_frozen(OrigV2PairSemantics::TrainFitted)
            .unwrap();

        receipt.sweeps -= 1;
        assert!(receipt
            .validate_frozen(OrigV2PairSemantics::TrainFitted)
            .is_err());
    }
}
