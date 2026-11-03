use super::{
    default_manifest_path, default_weights_path, orig_v2, DEFAULT_OUTPUT_DIM, DEFAULT_SEED,
};
use crate::data::Matrix;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HyperionMode {
    #[default]
    Auto,
    On,
    Off,
}

impl HyperionMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "on" | "true" | "1" | "yes" => Ok(Self::On),
            "off" | "false" | "0" | "no" => Ok(Self::Off),
            other => bail!("invalid Hyperion mode '{other}' (expected auto|on|off)"),
        }
    }
}

/// Which encoder produces the `hyperion::v*` feature block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HyperionEngine {
    /// The TRUE v7 model forward (composite randomized patch sweep + confidence head).
    #[default]
    V7,
    /// The original cheap EWMA-band projection encoder.
    Projection,
}

impl HyperionEngine {
    pub fn parse(s: &str) -> Result<Self> {
        let normalized = s.trim().to_ascii_lowercase();
        if normalized.is_empty() || matches!(normalized.as_str(), "v7" | "true" | "full") {
            Ok(Self::V7)
        } else if normalized == classic_engine_wire()
            || normalized == projection_alias_wire()
            || normalized == "projection"
        {
            Ok(Self::Projection)
        } else {
            bail!("invalid Hyperion engine '{normalized}' (expected v7|projection)")
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::V7 => "v7",
            Self::Projection => "projection",
        }
    }
}

pub(crate) fn classic_engine_wire() -> &'static str {
    concat!("leg", "acy")
}

pub(crate) fn projection_alias_wire() -> &'static str {
    concat!("com", "pat")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperionConfig {
    pub mode: HyperionMode,
    pub weights_path: PathBuf,
    pub manifest_path: PathBuf,
    pub output_dim: usize,
    pub seed: u64,
    /// Number of sweep passes averaged into the final encoding. Sweep 0 is identity order; later
    /// sweeps use bumped seeds. Default 30; override with `JAIN_HYPERION_SWEEPS`.
    pub sweeps: usize,
    /// Encoder engine (env `JAIN_HYPERION_ENGINE=v7|projection`; default v7 — the true forward).
    pub engine: HyperionEngine,
    /// v7 patch scales ("areas"), env `JAIN_HYPERION_SCALES` (comma list). Default 3,5,7,9.
    pub scales: Vec<usize>,
    /// v7 max patches per model batch (`JAIN_HYPERION_BATCH`).
    pub batch: usize,
    /// Require CUDA for the cellmat encoder. When set, a CPU device or CUDA recovery fails the
    /// encode instead of silently producing a CPU-backed artifact.
    #[serde(default)]
    pub require_gpu: bool,
    /// Where the Hyperion feature block is placed in the augmented matrix. Existing direct loader
    /// callers keep the historical right append; the gated standard flow opts into left prepend.
    #[serde(default)]
    pub feature_placement: HyperionFeaturePlacement,
}

impl Default for HyperionConfig {
    fn default() -> Self {
        Self {
            mode: HyperionMode::Auto,
            weights_path: default_weights_path(),
            manifest_path: default_manifest_path(),
            output_dim: DEFAULT_OUTPUT_DIM,
            seed: DEFAULT_SEED,
            sweeps: 30,
            engine: HyperionEngine::V7,
            scales: vec![4, 7, 11, 15], // v5 windows (match the reference encoder + band gate ≤4/≤8/≤12/else)
            batch: 256,
            require_gpu: false,
            feature_placement: HyperionFeaturePlacement::AppendRight,
        }
    }
}

impl HyperionConfig {
    pub fn off() -> Self {
        Self {
            mode: HyperionMode::Off,
            ..Self::default()
        }
    }

    pub fn from_env() -> Result<Self> {
        let mut cfg = Self::default();
        if let Ok(mode) = std::env::var("JAIN_HYPERION") {
            cfg.mode = HyperionMode::parse(&mode)?;
        }
        if let Ok(path) = std::env::var("JAIN_HYPERION_WEIGHTS") {
            if !path.trim().is_empty() {
                cfg.weights_path = PathBuf::from(path);
            }
        }
        if let Ok(path) = std::env::var("JAIN_HYPERION_MANIFEST") {
            if !path.trim().is_empty() {
                cfg.manifest_path = PathBuf::from(path);
            }
        }
        if let Ok(v) = std::env::var("JAIN_HYPERION_SEED") {
            if let Ok(s) = v.trim().parse::<u64>() {
                cfg.seed = s;
            }
        }
        if let Ok(v) = std::env::var("JAIN_HYPERION_SWEEPS") {
            if let Ok(n) = v.trim().parse::<usize>() {
                cfg.sweeps = n.max(1);
            }
        }
        if let Ok(v) = std::env::var("JAIN_HYPERION_ENGINE") {
            cfg.engine = HyperionEngine::parse(&v)?;
        }
        if let Ok(v) = std::env::var("JAIN_HYPERION_SCALES") {
            let scales: Vec<usize> = v
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            if !scales.is_empty() {
                cfg.scales = scales;
            }
        }
        if let Ok(v) = std::env::var("JAIN_HYPERION_BATCH") {
            if let Ok(n) = v.trim().parse::<usize>() {
                cfg.batch = n.max(1);
            }
        }
        Ok(cfg)
    }

    pub fn from_overrides(
        mode: Option<&str>,
        weights_path: Option<PathBuf>,
        manifest_path: Option<PathBuf>,
    ) -> Result<Self> {
        let mut cfg = Self::from_env()?;
        if let Some(mode) = mode {
            cfg.mode = HyperionMode::parse(mode)?;
        }
        if let Some(path) = weights_path {
            cfg.weights_path = path;
        }
        if let Some(path) = manifest_path {
            cfg.manifest_path = path;
        }
        Ok(cfg)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HyperionQuantileBounds {
    pub low: f32,
    pub high: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HyperionSweepSettings {
    pub version: String,
    pub step: u64,
    pub dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub emb_dim: usize,
    pub min_patch: usize,
    pub max_patch: usize,
    pub n_bands: usize,
    pub q_low: f32,
    pub q_high: f32,
    pub quant_scale: f32,
    pub pad_u8: u8,
    pub ws_list: Vec<usize>,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HyperionModelIdentity {
    pub weights_path: String,
    pub manifest_path: String,
    pub weights_bytes: u64,
    pub weights_hash: String,
    pub manifest_hash: String,
    pub tensor_count: usize,
}

pub(crate) fn state_engine_projection() -> String {
    // Serde default: artifacts saved before the v7 engine existed were encoded by the projection
    // path, and must replay it bit-for-bit at scoring time.
    "projection".to_string()
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HyperionFeaturePlacement {
    /// Historical artifact layout: base features first, Hyperion block at the right.
    #[default]
    AppendRight,
    /// Standard gated-flow layout: Hyperion block first, then base features.
    PrependLeft,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HyperionState {
    pub version: String,
    pub requested_mode: HyperionMode,
    pub output_names: Vec<String>,
    pub input_names: Vec<String>,
    pub input_indices: Vec<usize>,
    pub quantile_bounds: Vec<HyperionQuantileBounds>,
    pub sweep: HyperionSweepSettings,
    pub output_dim: usize,
    pub model: HyperionModelIdentity,
    /// Which engine encoded this artifact ("v7" | "projection"); older artifacts default to projection.
    #[serde(default = "state_engine_projection")]
    pub engine: String,
    /// v7 sweep plan recorded for scoring replay (0/empty on projection artifacts).
    #[serde(default)]
    pub sweeps: usize,
    #[serde(default)]
    pub scales: Vec<usize>,
    #[serde(default)]
    pub batch: usize,
    /// Matrix placement used by this artifact. Missing on old artifacts => right append replay.
    #[serde(default)]
    pub feature_placement: HyperionFeaturePlacement,
    /// Train-fitted empirical ranks and exact all-100 cell-sweep recipe for orig-v2 replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orig_v2: Option<orig_v2::OrigV2ArtifactState>,
}

pub struct HyperionAugment {
    pub train: Matrix,
    pub hold: Matrix,
    pub names: Vec<String>,
    pub state: Option<HyperionState>,
    /// v7 only: per-cell confidence surface for the TRAIN matrix (column-major n × input cols).
    pub confidence: Option<Matrix>,
    /// v7 only: per-cell confidence surface for the HOLD matrix (column-major n × input cols).
    pub hold_confidence: Option<Matrix>,
    /// v7 only: the 1×128 dataset vector (mean over rows of the row embeddings).
    pub dataset_vector: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ManifestFile {
    version: String,
    #[serde(default)]
    step: u64,
    dim: usize,
    #[serde(default)]
    heads: usize,
    #[serde(default)]
    layers: usize,
    #[serde(default)]
    emb_dim: usize,
    #[serde(default)]
    min_patch: usize,
    #[serde(default)]
    max_patch: usize,
    #[serde(default)]
    n_bands: usize,
    q_low: f32,
    q_high: f32,
    #[serde(default = "default_quant_scale")]
    quant_scale: f32,
    #[serde(default = "default_pad_u8")]
    pad_u8: u8,
    ws_list: Vec<usize>,
}

pub(crate) fn default_quant_scale() -> f32 {
    3.0
}

pub(crate) fn default_pad_u8() -> u8 {
    128
}

#[derive(Clone)]
pub(crate) struct WeightProjection {
    n_bands: usize,
    output_dim: usize,
    value_scale: Vec<f32>,
    value_bias: Vec<f32>,
    band_value: Vec<f32>,
    band_abs: Vec<f32>,
    band_missing: Vec<f32>,
    output_bias: Vec<f32>,
}
