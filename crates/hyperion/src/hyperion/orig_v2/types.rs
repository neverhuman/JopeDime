use super::*;

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
    pub(crate) fn plan(&self) -> OrigV2SweepPlan {
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
    pub(crate) fn validate_profile(&self) -> Result<()> {
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
    pub(in crate::hyperion) fn validate_frozen(
        &self,
        required_semantics: OrigV2PairSemantics,
    ) -> Result<()> {
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
    pub(super) config: OrigV2Config,
    #[cfg(feature = "hyperion-cpu")]
    pub(super) model: crate::hyperion_v7::backend::V7Model,
    #[cfg(feature = "hyperion-cpu")]
    pub(super) settings: HyperionSweepSettings,
    #[cfg(feature = "hyperion-cpu")]
    pub(super) identity: HyperionModelIdentity,
    #[cfg(feature = "hyperion-cpu")]
    pub(super) weights_sha256: String,
    #[cfg(feature = "hyperion-cpu")]
    pub(super) manifest_sha256: String,
    #[cfg(feature = "hyperion-cpu")]
    pub(super) device: String,
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
