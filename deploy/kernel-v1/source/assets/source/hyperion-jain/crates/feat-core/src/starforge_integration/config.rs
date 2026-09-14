use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::task::TaskKind;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StarforgeMode {
    #[default]
    Auto,
    On,
    Off,
}

impl StarforgeMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "true" | "1" => Ok(Self::On),
            "off" | "false" | "0" => Ok(Self::Off),
            other => bail!("unsupported starforge mode {other:?}; use auto, on, or off"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StarforgeDevice {
    #[default]
    Auto,
    Cpu,
    Cuda0,
}

impl StarforgeDevice {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "cuda:0" => Ok(Self::Cuda0),
            other => bail!("unsupported starforge device {other:?}; use auto, cpu, or cuda:0"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda0 => "cuda:0",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StarforgeDType {
    #[default]
    Auto,
    F32,
    BF16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StarforgeRegressionEngine {
    Auto,
    Starlight,
    #[default]
    Chimera,
}

impl StarforgeRegressionEngine {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "starlight" => Ok(Self::Starlight),
            "chimera" => Ok(Self::Chimera),
            other => bail!(
                "unsupported Starforge regression engine {other:?}; use chimera, auto, or deprecated starlight"
            ),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Starlight => "starlight",
            Self::Chimera => "chimera",
        }
    }
}

impl StarforgeDType {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "f32" | "float32" => Ok(Self::F32),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            other => bail!("unsupported starforge dtype {other:?}; use auto, f32, or bf16"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::F32 => "f32",
            Self::BF16 => "bf16",
        }
    }

    /// Resolve the concrete compute dtype for a device. BF16 is a CUDA-only optimization —
    /// candle's CPU backend has no bf16 matmul — so on CPU we always run f32. This keeps a
    /// bf16-configured model (e.g. one trained on a GPU box) servable/scorable on a CPU endpoint
    /// instead of failing every prediction with "unsupported dtype BF16 for op matmul". `Auto`
    /// picks bf16 on CUDA and f32 on CPU.
    pub fn effective(self, use_cuda: bool) -> StarforgeDType {
        if !use_cuda {
            return Self::F32;
        }
        match self {
            Self::Auto => Self::BF16,
            explicit => explicit,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarforgeConfig {
    pub mode: StarforgeMode,
    pub device: StarforgeDevice,
    pub dtype: StarforgeDType,
    pub require_gpu: bool,
    pub max_context: usize,
    pub top_features: usize,
    pub seed: u64,
    /// Binary/multilabel classifier weights. Kept as the historical field for compatibility.
    pub weights_path: PathBuf,
    /// Regression weights. Separate artifact so classifier deployments keep their existing model.
    pub regressor_weights_path: PathBuf,
    pub regression_engine: StarforgeRegressionEngine,
    pub chimera_weights_path: PathBuf,
    pub tabicl_regressor_weights_path: PathBuf,
}

impl Default for StarforgeConfig {
    fn default() -> Self {
        Self {
            mode: StarforgeMode::Auto,
            device: StarforgeDevice::Auto,
            dtype: StarforgeDType::Auto,
            require_gpu: false,
            max_context: 512,
            top_features: 500,
            seed: 20260616,
            weights_path: default_classifier_weights_path(),
            regressor_weights_path: default_regressor_weights_path(),
            regression_engine: StarforgeRegressionEngine::Chimera,
            chimera_weights_path: default_chimera_weights_path(),
            tabicl_regressor_weights_path: default_tabicl_regressor_weights_path(),
        }
    }
}

impl StarforgeConfig {
    pub fn disabled() -> Self {
        Self {
            mode: StarforgeMode::Off,
            ..Default::default()
        }
    }

    pub fn wants_starforge(&self) -> bool {
        self.mode != StarforgeMode::Off
    }

    pub fn is_required(&self) -> bool {
        self.mode == StarforgeMode::On || self.require_gpu
    }

    pub fn seed_for_context(&self) -> u64 {
        self.seed
    }

    pub fn weights_path_for_task(&self, task: TaskKind) -> &Path {
        match task {
            TaskKind::Regression => &self.regressor_weights_path,
            _ => &self.weights_path,
        }
    }

    pub fn chimera_weights_present(&self) -> bool {
        self.regressor_weights_path.is_file()
            && self.chimera_weights_path.is_file()
            && self.tabicl_regressor_weights_path.is_file()
    }
}

fn default_classifier_weights_path() -> PathBuf {
    if let Ok(path) = std::env::var("JAIN_STARFORGE_WEIGHTS") {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }
    // Prefer the zstd-compressed weights (shipped to shrink the container image)
    // and fall back to a plain `.safetensors`, first in the runtime image location
    // and then in the local dev tree.
    for candidate in [
        "/opt/jain/starforge/starlight_core.safetensors.zst",
        "/opt/jain/starforge/starlight_core.safetensors",
        "artifacts/starforge/starlight_core.safetensors.zst",
        "artifacts/starforge/starlight_core.safetensors",
    ] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    // Nothing present yet; return the conventional dev path so a subsequent load
    // surfaces a stable "missing weights" error.
    PathBuf::from("artifacts/starforge/starlight_core.safetensors")
}

fn default_regressor_weights_path() -> PathBuf {
    if let Ok(path) = std::env::var("JAIN_STARFORGE_REGRESSOR_WEIGHTS") {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }
    for candidate in [
        "/opt/jain/starforge/starlight_core_regressor.safetensors.zst",
        "/opt/jain/starforge/starlight_core_regressor.safetensors",
        "artifacts/starforge/starlight_core_regressor.safetensors.zst",
        "artifacts/starforge/starlight_core_regressor.safetensors",
    ] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from("artifacts/starforge/starlight_core_regressor.safetensors")
}

fn default_chimera_weights_path() -> PathBuf {
    if let Ok(path) = std::env::var("JAIN_STARFORGE_CHIMERA_WEIGHTS") {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }
    for candidate in [
        "/opt/jain/starforge/chimera_stackfusionreg.safetensors.zst",
        "/opt/jain/starforge/chimera_stackfusionreg.safetensors",
        "artifacts/starforge/chimera_stackfusionreg.safetensors.zst",
        "artifacts/starforge/chimera_stackfusionreg.safetensors",
    ] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from("artifacts/starforge/chimera_stackfusionreg.safetensors")
}

fn default_tabicl_regressor_weights_path() -> PathBuf {
    if let Ok(path) = std::env::var("JAIN_TABICL_REGRESSOR_WEIGHTS") {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }
    for candidate in [
        "/opt/jain/starforge/tabicl-regressor-v2-20260212.safetensors.zst",
        "/opt/jain/starforge/tabicl-regressor-v2-20260212.safetensors",
        "artifacts/foundation/tabicl-regressor-v2-20260212.safetensors.zst",
        "artifacts/foundation/tabicl-regressor-v2-20260212.safetensors",
    ] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from("artifacts/foundation/tabicl-regressor-v2-20260212.safetensors")
}

#[derive(Clone, Debug)]
pub struct StarforgeDiagnostics {
    pub compiled: bool,
    pub cuda_compiled: bool,
    pub cuda_available: bool,
    pub selected_device: String,
    pub selected_dtype: String,
    pub weights_present: bool,
    pub regressor_weights_present: bool,
    pub chimera_weights_present: bool,
    pub tabicl_regressor_weights_present: bool,
    pub regression_engine: String,
    pub warning: Option<String>,
}

impl StarforgeDiagnostics {
    pub fn for_config(cfg: &StarforgeConfig) -> Self {
        let weights_present = cfg.weights_path.is_file();
        let regressor_weights_present = cfg.regressor_weights_path.is_file();
        let chimera_weights_present = cfg.chimera_weights_path.is_file();
        let tabicl_regressor_weights_present = cfg.tabicl_regressor_weights_path.is_file();
        let cuda_available = cuda_available();
        let selected_device = if matches!(cfg.device, StarforgeDevice::Cpu) {
            "cpu"
        } else if cuda_available && cfg.device != StarforgeDevice::Cpu {
            "cuda:0"
        } else {
            "cpu"
        };
        // Report the dtype that will actually run (see StarforgeDType::effective): bf16 only on CUDA,
        // f32 on CPU — so doctor never advertises bf16 for a CPU device it cannot execute in bf16.
        let selected_dtype = cfg.dtype.effective(selected_device == "cuda:0").label();
        let needs_cpu_mode_warning =
            cfg.wants_starforge() && !cuda_available && cfg.device != StarforgeDevice::Cpu;
        let warning = if needs_cpu_mode_warning {
            Some(
                "CUDA is unavailable; Starforge will use CPU execution mode unless GPU is required"
                    .into(),
            )
        } else {
            None
        };
        Self {
            compiled: starforge_compiled(),
            cuda_compiled: starforge_cuda_compiled(),
            cuda_available,
            selected_device: selected_device.into(),
            selected_dtype: selected_dtype.into(),
            weights_present,
            regressor_weights_present,
            chimera_weights_present,
            tabicl_regressor_weights_present,
            regression_engine: cfg.regression_engine.label().into(),
            warning,
        }
    }
}

pub fn starforge_compiled() -> bool {
    cfg!(feature = "starforge-cpu")
}

pub fn starforge_cuda_compiled() -> bool {
    cfg!(feature = "starforge-cuda")
}

pub fn cuda_visible_hint() -> bool {
    if Path::new("/dev/nvidia0").exists() || Path::new("/proc/driver/nvidia/version").exists() {
        return true;
    }
    match std::env::var("NVIDIA_VISIBLE_DEVICES") {
        Ok(v) => {
            let v = v.trim();
            !v.is_empty() && v != "void" && v != "none"
        }
        Err(_) => false,
    }
}

#[cfg(feature = "starforge-cuda")]
pub fn cuda_available() -> bool {
    candle_core::Device::new_cuda(0).is_ok()
}

#[cfg(not(feature = "starforge-cuda"))]
pub fn cuda_available() -> bool {
    false
}

#[cfg(test)]
mod dtype_tests {
    use super::{StarforgeConfig, StarforgeDType, StarforgeRegressionEngine};

    #[test]
    fn default_regression_engine_is_chimera() {
        assert_eq!(
            StarforgeConfig::default().regression_engine,
            StarforgeRegressionEngine::Chimera
        );
    }

    #[test]
    fn effective_dtype_never_uses_bf16_on_cpu() {
        // CPU: always f32 (candle CPU has no bf16 matmul) — even an explicit bf16 request downcasts,
        // so a GPU-trained bf16 model stays servable on a CPU endpoint.
        assert_eq!(StarforgeDType::BF16.effective(false), StarforgeDType::F32);
        assert_eq!(StarforgeDType::Auto.effective(false), StarforgeDType::F32);
        assert_eq!(StarforgeDType::F32.effective(false), StarforgeDType::F32);
        // CUDA: auto picks bf16; explicit choices are honored.
        assert_eq!(StarforgeDType::Auto.effective(true), StarforgeDType::BF16);
        assert_eq!(StarforgeDType::BF16.effective(true), StarforgeDType::BF16);
        assert_eq!(StarforgeDType::F32.effective(true), StarforgeDType::F32);
    }
}
