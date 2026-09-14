//! Hyperion v7 matrix-sweep feature injection.
//!
//! This module is intentionally data-layer shaped: it consumes the already encoded feature matrix
//! only, fits train-only quantile/sweep state, appends a fixed 128-column block, and persists enough
//! state for scoring replay. The target never enters this API.

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

pub const HYPERION_VERSION: &str = "v7";
pub const DEFAULT_OUTPUT_DIM: usize = 128;
pub const DEFAULT_SEED: u64 = 0x4859_5045_5249_4f4e;
// DEFAULT = the v5-128 arm B encoder (landmark-distill, retrained) — the validated better-routing
// model. It is a v5-geometry checkpoint (dim128, 4 layers, 4 bands, ws {4,7,11,15}); the manifest
// carries version:"v7" only to satisfy the loader's version gate. (Prior v7-512 default at
// /home/ubuntu/hyperion/weights/v7/.)
pub const DEFAULT_WEIGHTS: &str = "/home/ubuntu/hyperion/weights/v5_armB/model.safetensors";
pub const DEFAULT_MANIFEST: &str = "/home/ubuntu/hyperion/weights/v5_armB/manifest.json";
const PREVIEW_MAX_POINTS: usize = 160;
const PREVIEW_TOP_ROWS: usize = 8;
const CONCERN_Z_THRESHOLD: f64 = 3.5;

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
    Legacy,
}

impl HyperionEngine {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "v7" | "true" | "full" => Ok(Self::V7),
            "legacy" | "projection" => Ok(Self::Legacy),
            other => bail!("invalid Hyperion engine '{other}' (expected v7|legacy)"),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::V7 => "v7",
            Self::Legacy => "legacy",
        }
    }
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
    /// Encoder engine (env `JAIN_HYPERION_ENGINE=v7|legacy`; default v7 — the true forward).
    pub engine: HyperionEngine,
    /// v7 patch scales ("areas"), env `JAIN_HYPERION_SCALES` (comma list). Default 3,5,7,9.
    pub scales: Vec<usize>,
    /// v7 max patches per model batch (`JAIN_HYPERION_BATCH`).
    pub batch: usize,
}

impl Default for HyperionConfig {
    fn default() -> Self {
        Self {
            mode: HyperionMode::Auto,
            weights_path: PathBuf::from(DEFAULT_WEIGHTS),
            manifest_path: PathBuf::from(DEFAULT_MANIFEST),
            output_dim: DEFAULT_OUTPUT_DIM,
            seed: DEFAULT_SEED,
            sweeps: 30,
            engine: HyperionEngine::V7,
            scales: vec![4, 7, 11, 15], // v5 windows (match the reference encoder + band gate ≤4/≤8/≤12/else)
            batch: 256,
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

fn state_engine_legacy() -> String {
    // Serde default: artifacts saved BEFORE the v7 engine existed were encoded by the legacy
    // projection — they must replay it bit-for-bit at scoring time.
    "legacy".to_string()
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
    /// Which engine encoded this artifact ("v7" | "legacy"); old artifacts default to legacy.
    #[serde(default = "state_engine_legacy")]
    pub engine: String,
    /// v7 sweep plan recorded for scoring replay (0/empty on legacy artifacts).
    #[serde(default)]
    pub sweeps: usize,
    #[serde(default)]
    pub scales: Vec<usize>,
    #[serde(default)]
    pub batch: usize,
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
struct ManifestFile {
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

fn default_quant_scale() -> f32 {
    3.0
}

fn default_pad_u8() -> u8 {
    128
}

#[derive(Clone)]
struct WeightProjection {
    n_bands: usize,
    output_dim: usize,
    value_scale: Vec<f32>,
    value_bias: Vec<f32>,
    band_value: Vec<f32>,
    band_abs: Vec<f32>,
    band_missing: Vec<f32>,
    output_bias: Vec<f32>,
}

pub fn hyperion_output_names(output_dim: usize) -> Vec<String> {
    (0..output_dim)
        .map(|j| format!("hyperion::v{j:03}"))
        .collect()
}

pub fn fit_transform_pair(
    train: &Matrix,
    hold: &Matrix,
    input_names: &[String],
    cfg: &HyperionConfig,
    sink: &dyn Sink,
    t0: Instant,
) -> Result<HyperionAugment> {
    let output_dim = cfg.output_dim.min(DEFAULT_OUTPUT_DIM);
    emit_hyperion(
        sink,
        t0,
        "start",
        train.n,
        train.c,
        output_dim,
        cfg.engine,
        Some(hyperion_device_label()),
        Some(match cfg.mode {
            HyperionMode::Auto => "auto",
            HyperionMode::On => "on",
            HyperionMode::Off => "off",
        }),
        None,
        Some(0.0),
        None,
    );

    if cfg.mode == HyperionMode::Off {
        emit_hyperion(
            sink,
            t0,
            "done",
            train.n,
            train.c,
            0,
            cfg.engine,
            None,
            Some("skipped"),
            Some("Hyperion disabled"),
            Some(1.0),
            None,
        );
        return Ok(HyperionAugment {
            train: train.clone(),
            hold: hold.clone(),
            names: input_names.to_vec(),
            state: None,
            confidence: None,
            hold_confidence: None,
            dataset_vector: None,
        });
    }

    let load = load_runtime(cfg, output_dim);
    let runtime = match load {
        Ok(runtime) => runtime,
        Err(err) => {
            if cfg.mode == HyperionMode::On {
                emit_hyperion(
                    sink,
                    t0,
                    "failed",
                    train.n,
                    train.c,
                    output_dim,
                    cfg.engine,
                    Some(hyperion_device_label()),
                    Some("failed"),
                    Some(&err.to_string()),
                    Some(1.0),
                    None,
                );
                return Err(err);
            }
            emit_hyperion(
                sink,
                t0,
                "warning",
                train.n,
                train.c,
                0,
                cfg.engine,
                Some(hyperion_device_label()),
                Some("skipped"),
                Some(&format!("Hyperion unavailable in auto mode: {err:#}")),
                Some(1.0),
                None,
            );
            emit_hyperion(
                sink,
                t0,
                "done",
                train.n,
                train.c,
                0,
                cfg.engine,
                None,
                Some("skipped"),
                None,
                Some(1.0),
                None,
            );
            return Ok(HyperionAugment {
                train: train.clone(),
                hold: hold.clone(),
                names: input_names.to_vec(),
                state: None,
                confidence: None,
                hold_confidence: None,
                dataset_vector: None,
            });
        }
    };

    if train.c != input_names.len() {
        bail!(
            "Hyperion input shape mismatch: matrix has {} columns but {} names were provided",
            train.c,
            input_names.len()
        );
    }
    if hold.c != train.c {
        bail!(
            "Hyperion holdout shape mismatch: train has {} columns but holdout has {}",
            train.c,
            hold.c
        );
    }

    let bounds = fit_quantile_bounds(train, runtime.settings.q_low, runtime.settings.q_high);

    // ── Engine dispatch: the TRUE v7 forward (default) vs the legacy projection encoder. A v7
    // load failure downgrades exactly like a runtime load failure (On → fail, Auto → skip).
    let mut v7_result: Option<V7Fit> = None;
    if cfg.engine == HyperionEngine::V7 {
        match fit_v7(
            train,
            hold,
            &bounds,
            &runtime.settings,
            cfg,
            output_dim,
            sink,
            t0,
        ) {
            Ok(fit) => {
                if fit.cancelled {
                    // Cooperative cancel mid-encode: a partial encoding would poison the features,
                    // so continue UNAUGMENTED; the caller's cancel flag halts the run right after.
                    emit_hyperion(
                        sink,
                        t0,
                        "done",
                        train.n,
                        train.c,
                        0,
                        HyperionEngine::V7,
                        fit.device,
                        Some("skipped"),
                        Some("Hyperion encode cancelled"),
                        Some(1.0),
                        None,
                    );
                    return Ok(HyperionAugment {
                        train: train.clone(),
                        hold: hold.clone(),
                        names: input_names.to_vec(),
                        state: None,
                        confidence: None,
                        hold_confidence: None,
                        dataset_vector: None,
                    });
                }
                v7_result = Some(fit);
            }
            Err(err) => {
                if cfg.mode == HyperionMode::On {
                    emit_hyperion(
                        sink,
                        t0,
                        "failed",
                        train.n,
                        train.c,
                        output_dim,
                        HyperionEngine::V7,
                        Some(hyperion_device_label()),
                        Some("failed"),
                        Some(&err.to_string()),
                        Some(1.0),
                        None,
                    );
                    return Err(err);
                }
                emit_hyperion(
                    sink,
                    t0,
                    "warning",
                    train.n,
                    train.c,
                    0,
                    HyperionEngine::V7,
                    Some(hyperion_device_label()),
                    Some("skipped"),
                    Some(&format!("Hyperion v7 unavailable in auto mode: {err:#}")),
                    Some(1.0),
                    None,
                );
                emit_hyperion(
                    sink,
                    t0,
                    "done",
                    train.n,
                    train.c,
                    0,
                    HyperionEngine::V7,
                    None,
                    Some("skipped"),
                    None,
                    Some(1.0),
                    None,
                );
                return Ok(HyperionAugment {
                    train: train.clone(),
                    hold: hold.clone(),
                    names: input_names.to_vec(),
                    state: None,
                    confidence: None,
                    hold_confidence: None,
                    dataset_vector: None,
                });
            }
        }
    }

    let mut used_engine = HyperionEngine::Legacy;
    let mut used_device = Some("cpu");
    let (train_block, hold_block, confidence, hold_confidence, dataset_vector) =
        if let Some(fit) = v7_result {
            used_engine = HyperionEngine::V7;
            used_device = fit.device;
            (
                fit.train_block,
                fit.hold_block,
                Some(fit.confidence),
                Some(fit.hold_confidence),
                Some(fit.dataset_vector),
            )
        } else {
            let sweeps = cfg.sweeps.max(1);
            let (tb, hb) = if sweeps <= 1 {
                (
                    encode_with_projection(train, &bounds, &runtime.settings, &runtime.projection),
                    encode_with_projection(hold, &bounds, &runtime.settings, &runtime.projection),
                )
            } else {
                // Combine K sweeps (each with a fresh per-row start/stride via a bumped seed) by averaging
                // the output blocks -> a more stable row embedding for the record-outlier detector.
                let mut tb = vec![0f32; train.n * output_dim];
                let mut hb = vec![0f32; hold.n * output_dim];
                for k in 0..sweeps {
                    let mut s = runtime.settings.clone();
                    s.seed = runtime.settings.seed ^ mix64(k as u64);
                    let t = encode_with_projection(train, &bounds, &s, &runtime.projection);
                    let h = encode_with_projection(hold, &bounds, &s, &runtime.projection);
                    for (a, b) in tb.iter_mut().zip(t.data.iter()) {
                        *a += *b;
                    }
                    for (a, b) in hb.iter_mut().zip(h.data.iter()) {
                        *a += *b;
                    }
                }
                let inv = 1.0 / sweeps as f32;
                tb.iter_mut().for_each(|a| *a *= inv);
                hb.iter_mut().for_each(|a| *a *= inv);
                (
                    Matrix::new(tb, train.n, output_dim),
                    Matrix::new(hb, hold.n, output_dim),
                )
            };
            (tb, hb, None, None, None)
        };
    // Optional dump of the (averaged) sweep matrix for external record-outlier scoring.
    if let Ok(prefix) = std::env::var("JAIN_HYPERION_DUMP") {
        if !prefix.trim().is_empty() {
            dump_hyperion_block(&format!("{}.train.csv", prefix.trim()), &train_block);
        }
    }
    let preview = build_hyperion_preview(&train_block);
    let mut names = input_names.to_vec();
    let output_names = hyperion_output_names(output_dim);
    names.extend(output_names.iter().cloned());
    let is_v7 = confidence.is_some();
    let state = HyperionState {
        version: runtime.settings.version.clone(),
        requested_mode: cfg.mode,
        output_names,
        input_names: input_names.to_vec(),
        input_indices: (0..input_names.len()).collect(),
        quantile_bounds: bounds,
        sweep: runtime.settings,
        output_dim,
        model: runtime.identity,
        engine: if is_v7 {
            "v7".to_string()
        } else {
            "legacy".to_string()
        },
        sweeps: cfg.sweeps.max(1),
        scales: if is_v7 {
            cfg.scales.clone()
        } else {
            Vec::new()
        },
        batch: if is_v7 { cfg.batch.max(1) } else { 0 },
    };
    let train = append_columns(train, &train_block);
    let hold = append_columns(hold, &hold_block);

    emit_hyperion(
        sink,
        t0,
        "done",
        train.n,
        input_names.len(),
        output_dim,
        used_engine,
        used_device,
        Some("used"),
        None,
        Some(1.0),
        Some(preview),
    );
    Ok(HyperionAugment {
        train,
        hold,
        names,
        state: Some(state),
        confidence,
        hold_confidence,
        dataset_vector,
    })
}

/// v7 encode result: feature blocks (column-major) + the confidence surface + dataset vector.
struct V7Fit {
    train_block: Matrix,
    hold_block: Matrix,
    confidence: Matrix,
    hold_confidence: Matrix,
    dataset_vector: Vec<f32>,
    device: Option<&'static str>,
    cancelled: bool,
}

/// Convert a row-major `h × width` buffer (keeping the first `keep` dims) into a column-major
/// feat-core [`Matrix`].
#[cfg(feature = "hyperion-cpu")]
fn column_major_block(row_major: &[f32], h: usize, width: usize, keep: usize) -> Matrix {
    let mut data = vec![0.0f32; h * keep];
    for r in 0..h {
        for d in 0..keep {
            data[d * h + r] = row_major[r * width + d];
        }
    }
    Matrix::new(data, h, keep)
}

/// The TRUE v7 encode: composite randomized sweep over train (streaming SweepFrames + real
/// phase-0 block events) then hold (block events only; a different matrix must not corrupt the
/// train grid the UI is showing). `done_frac` is rescaled by each matrix's patch share.
#[cfg(feature = "hyperion-cpu")]
#[allow(clippy::too_many_arguments)]
fn fit_v7(
    train: &Matrix,
    hold: &Matrix,
    bounds: &[HyperionQuantileBounds],
    settings: &HyperionSweepSettings,
    cfg: &HyperionConfig,
    output_dim: usize,
    sink: &dyn Sink,
    t0: Instant,
) -> Result<V7Fit> {
    use crate::hyperion_v7::backend::V7Model;
    use crate::hyperion_v7::config::SweepPlan;
    use crate::hyperion_v7::grid::quantize_u8_with_bounds;
    use crate::hyperion_v7::sweep::run_composite_sweep;
    use crate::hyperion_v7::{frames, EMB_KEEP};

    let model = V7Model::load(&cfg.weights_path, settings)?;
    let device = model.device_label();

    // Validate/normalize the scales against the checkpoint geometry.
    let mut scales: Vec<usize> = cfg
        .scales
        .iter()
        .copied()
        .filter(|&p| p >= 2 && p <= settings.max_patch)
        .collect();
    if scales.is_empty() {
        scales = vec![3, 5, 7, 9];
    }
    let plan = SweepPlan {
        sweeps: cfg.sweeps.max(1) as u32,
        scales: scales.clone(),
        batch: cfg.batch.max(1),
        seed: settings.seed,
        target_chunks_per_sweep: 8,
        fixed_perms: None,
    };

    let patches = |h: usize, w: usize| -> usize {
        scales.iter().map(|&p| h.div_ceil(p) * w.div_ceil(p)).sum()
    };
    let tp = patches(train.n, train.c).max(1);
    let hp = patches(hold.n, hold.c);
    let train_share = tp as f64 / (tp + hp) as f64;

    let xu8_train = quantize_u8_with_bounds(train, bounds, settings.pad_u8);
    let xu8_hold = quantize_u8_with_bounds(hold, bounds, settings.pad_u8);

    // Emit coarse phase-0 block events on decile crossings (the fine-grained progress rides the
    // SweepFrames); every block event also feeds the human narration, so keep them ~10 per run.
    let mut last_decile = -1i64;
    let mut block_on_decile = |done_frac: f64| {
        let decile = (done_frac * 10.0).floor() as i64;
        if decile > last_decile {
            last_decile = decile;
            emit_hyperion(
                sink,
                t0,
                "block",
                train.n,
                train.c,
                output_dim,
                HyperionEngine::V7,
                Some(device),
                Some("running"),
                None,
                Some(done_frac),
                None,
            );
        }
    };

    let train_out = run_composite_sweep(
        &model,
        &xu8_train,
        settings.pad_u8,
        settings.quant_scale,
        settings.max_patch,
        &plan,
        sink,
        |report| {
            let overall = (report.sweep as f64 + report.frac) / plan.sweeps as f64;
            let done_frac = train_share * overall;
            sink.emit_sweep(&frames::build_frame(
                &report,
                done_frac,
                t0.elapsed().as_secs_f64(),
            ));
            block_on_decile(done_frac);
        },
    )?;
    if train_out.cancelled {
        return Ok(V7Fit {
            train_block: Matrix::new(Vec::new(), 0, 0),
            hold_block: Matrix::new(Vec::new(), 0, 0),
            confidence: Matrix::new(Vec::new(), 0, 0),
            hold_confidence: Matrix::new(Vec::new(), 0, 0),
            dataset_vector: Vec::new(),
            device: Some(device),
            cancelled: true,
        });
    }

    let hold_out = run_composite_sweep(
        &model,
        &xu8_hold,
        settings.pad_u8,
        settings.quant_scale,
        settings.max_patch,
        &plan,
        sink,
        |report| {
            let overall = (report.sweep as f64 + report.frac) / plan.sweeps as f64;
            block_on_decile(train_share + (1.0 - train_share) * overall);
        },
    )?;
    if hold_out.cancelled {
        return Ok(V7Fit {
            train_block: Matrix::new(Vec::new(), 0, 0),
            hold_block: Matrix::new(Vec::new(), 0, 0),
            confidence: Matrix::new(Vec::new(), 0, 0),
            hold_confidence: Matrix::new(Vec::new(), 0, 0),
            dataset_vector: Vec::new(),
            device: Some(device),
            cancelled: true,
        });
    }

    // Optional dump of the DENSE per-cell encoding map (rows, cols, EMB_KEEP) as .npy — the new
    // primary encoder output (each cell = mean of every window that swept it, per-dim 0..1).
    if let Ok(prefix) = std::env::var("JAIN_HYPERION_CELLMAP_DUMP") {
        let p = prefix.trim();
        if !p.is_empty() {
            dump_cellmap_npy(
                &format!("{p}.train.npy"),
                &train_out.cellmap,
                train_out.h,
                train_out.w,
            );
            dump_cellmap_npy(
                &format!("{p}.hold.npy"),
                &hold_out.cellmap,
                hold_out.h,
                hold_out.w,
            );
        }
    }

    let keep = output_dim.min(EMB_KEEP);
    Ok(V7Fit {
        train_block: column_major_block(&train_out.row_emb, train.n, EMB_KEEP, keep),
        hold_block: column_major_block(&hold_out.row_emb, hold.n, EMB_KEEP, keep),
        confidence: column_major_block(&train_out.confidence, train.n, train.c, train.c),
        hold_confidence: column_major_block(&hold_out.confidence, hold.n, hold.c, hold.c),
        dataset_vector: train_out.dataset_vec,
        device: Some(device),
        cancelled: false,
    })
}

/// Standalone 100%-Rust inference entry for the v7 Hyperion sweep encoder. `x` is the FEATURE
/// matrix (target already dropped by the caller). Quantizes with train-fit 1/99-pct bounds, loads
/// the v7 checkpoint, runs the composite multi-scale randomized sweep, and returns the
/// [`crate::hyperion_v7::sweep::SweepOutput`] — the dense per-cell map `(h,w,EMB_KEEP)`, the per-row
/// features (col-mean of the map), the per-cell confidence, and the pooled dataset vector.
#[cfg(feature = "hyperion-cpu")]
pub fn encode_v7_dataset(
    x: &Matrix,
    weights_path: &Path,
    manifest_path: &Path,
    sweeps: u32,
    scales: &[usize],
    seed: u64,
) -> Result<crate::hyperion_v7::sweep::SweepOutput> {
    use crate::hyperion_v7::backend::V7Model;
    use crate::hyperion_v7::config::SweepPlan;
    use crate::hyperion_v7::grid::quantize_u8_with_bounds;
    use crate::hyperion_v7::sweep::run_composite_sweep;
    use crate::progress::NullSink;

    let (settings, _hash) = load_settings(manifest_path, seed)?;
    let mut scales: Vec<usize> = scales
        .iter()
        .copied()
        .filter(|&p| p >= 2 && p <= settings.max_patch)
        .collect();
    if scales.is_empty() {
        scales = vec![3, 5, 7, 9];
    }
    let plan = SweepPlan {
        sweeps: sweeps.max(1),
        scales,
        batch: 256,
        seed: settings.seed,
        target_chunks_per_sweep: 8,
        fixed_perms: None,
    };
    let bounds = fit_quantile_bounds(x, settings.q_low, settings.q_high);
    let xu8 = quantize_u8_with_bounds(x, &bounds, settings.pad_u8);
    let model = V7Model::load(weights_path, &settings)?;
    run_composite_sweep(
        &model,
        &xu8,
        settings.pad_u8,
        settings.quant_scale,
        settings.max_patch,
        &plan,
        &NullSink,
        |_| {},
    )
}

/// Stub when the model stack is compiled out: engine=v7 behaves as a load failure (Auto → skip,
/// On → fail), mirroring the legacy weight-loading semantics.
#[cfg(not(feature = "hyperion-cpu"))]
#[allow(clippy::too_many_arguments)]
fn fit_v7(
    _train: &Matrix,
    _hold: &Matrix,
    _bounds: &[HyperionQuantileBounds],
    _settings: &HyperionSweepSettings,
    _cfg: &HyperionConfig,
    _output_dim: usize,
    _sink: &dyn Sink,
    _t0: Instant,
) -> Result<V7Fit> {
    bail!("Hyperion v7 engine requires the 'hyperion-cpu' build feature")
}

pub fn transform_scoring_matrix(
    encoded: &Matrix,
    encoded_names: &[String],
    state: &HyperionState,
) -> Result<Matrix> {
    if state.output_dim != state.output_names.len() {
        bail!("artifact Hyperion state has inconsistent output name count");
    }
    if state.input_names.len() != encoded_names.len() {
        bail!(
            "artifact Hyperion state expects {} encoded input columns but scoring produced {}",
            state.input_names.len(),
            encoded_names.len()
        );
    }
    for (idx, (expected, actual)) in state.input_names.iter().zip(encoded_names).enumerate() {
        if expected != actual {
            bail!(
                "artifact Hyperion input column {idx} expected '{expected}' but scoring produced '{actual}'"
            );
        }
    }
    if encoded.c != state.input_names.len() {
        bail!(
            "artifact Hyperion state expects {} input columns but scoring matrix has {}",
            state.input_names.len(),
            encoded.c
        );
    }
    if state.engine == "v7" {
        return transform_scoring_v7(encoded, state);
    }
    let projection = load_projection_for_state(state)?;
    let block = encode_with_projection(encoded, &state.quantile_bounds, &state.sweep, &projection);
    Ok(append_columns(encoded, &block))
}

/// Scoring replay for v7-encoded artifacts: verify the recorded weights, then re-run the composite
/// sweep with the artifact's recorded plan (deterministic given the recorded seed).
#[cfg(feature = "hyperion-cpu")]
fn transform_scoring_v7(encoded: &Matrix, state: &HyperionState) -> Result<Matrix> {
    use crate::hyperion_v7::backend::V7Model;
    use crate::hyperion_v7::config::SweepPlan;
    use crate::hyperion_v7::grid::quantize_u8_with_bounds;
    use crate::hyperion_v7::sweep::run_composite_sweep;
    use crate::hyperion_v7::EMB_KEEP;
    use crate::progress::NullSink;

    let path = PathBuf::from(&state.model.weights_path);
    if !path.is_file() {
        bail!(
            "artifact requires Hyperion v7 weights but {} is unavailable",
            path.display()
        );
    }
    let current_hash = hash_file(&path)?;
    if current_hash != state.model.weights_hash {
        bail!(
            "artifact requires Hyperion weights hash {} but {} has hash {}",
            state.model.weights_hash,
            path.display(),
            current_hash
        );
    }
    let model = V7Model::load(&path, &state.sweep)?;
    let plan = SweepPlan {
        sweeps: state.sweeps.max(1) as u32,
        scales: if state.scales.is_empty() {
            vec![3, 5, 7, 9]
        } else {
            state.scales.clone()
        },
        batch: state.batch.max(1),
        seed: state.sweep.seed,
        target_chunks_per_sweep: 8,
        fixed_perms: None,
    };
    let xu8 = quantize_u8_with_bounds(encoded, &state.quantile_bounds, state.sweep.pad_u8);
    let out = run_composite_sweep(
        &model,
        &xu8,
        state.sweep.pad_u8,
        state.sweep.quant_scale,
        state.sweep.max_patch,
        &plan,
        &NullSink,
        |_| {},
    )?;
    let keep = state.output_dim.min(EMB_KEEP);
    let block = column_major_block(&out.row_emb, encoded.n, EMB_KEEP, keep);
    Ok(append_columns(encoded, &block))
}

#[cfg(not(feature = "hyperion-cpu"))]
fn transform_scoring_v7(_encoded: &Matrix, _state: &HyperionState) -> Result<Matrix> {
    bail!(
        "artifact was encoded with the Hyperion v7 engine but feat-core was compiled without the 'hyperion-cpu' feature"
    )
}

pub fn expected_width(encoded_width: usize, state: Option<&HyperionState>) -> usize {
    encoded_width + state.map(|s| s.output_dim).unwrap_or(0)
}

pub fn append_columns(base: &Matrix, extra: &Matrix) -> Matrix {
    assert_eq!(base.n, extra.n, "appended matrix row count mismatch");
    let mut data = Vec::with_capacity(base.data.len() + extra.data.len());
    data.extend_from_slice(base.data.as_ref());
    data.extend_from_slice(extra.data.as_ref());
    Matrix::new(data, base.n, base.c + extra.c)
}

/// Write a Hyperion sweep block (column-major `Matrix`) to a row-major CSV (n rows x output_dim cols)
/// for external record-outlier analysis. Best-effort: failures are silently ignored.
fn dump_hyperion_block(path: &str, m: &Matrix) {
    use std::io::Write;
    let mut out = String::with_capacity(m.n.saturating_mul(m.c).saturating_mul(8));
    for row in 0..m.n {
        for col in 0..m.c {
            if col > 0 {
                out.push(',');
            }
            // Full f32 precision so low-variance sweep signal is not truncated before scoring.
            out.push_str(&format!("{:.9}", m.data[col * m.n + row]));
        }
        out.push('\n');
    }
    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(out.as_bytes());
    }
}

/// Write a dense per-cell Hyperion map (`h*w*EMB_KEEP` row-major f32) as a numpy `.npy` (v1.0,
/// C-order, shape `(h, w, EMB_KEEP)`) so it is directly `np.load`-able. Best-effort: errors ignored.
#[cfg(feature = "hyperion-cpu")]
fn dump_cellmap_npy(path: &str, cellmap: &[f32], h: usize, w: usize) {
    use std::io::Write;
    let dims = crate::hyperion_v7::EMB_KEEP;
    let base = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}, {dims}), }}");
    // Header (10 fixed bytes + header string + trailing '\n') must be a multiple of 64.
    let unpadded = 10 + base.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    let mut hdr = base.into_bytes();
    hdr.extend(std::iter::repeat(b' ').take(pad));
    hdr.push(b'\n');
    let hlen = hdr.len() as u16;
    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(b"\x93NUMPY\x01\x00");
        let _ = f.write_all(&hlen.to_le_bytes());
        let _ = f.write_all(&hdr);
        let mut buf = Vec::with_capacity(cellmap.len() * 4);
        for &v in cellmap {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&buf);
    }
}

fn emit_hyperion(
    sink: &dyn Sink,
    t0: Instant,
    event: &str,
    rows: usize,
    cols: usize,
    output_dim: usize,
    engine: HyperionEngine,
    device: Option<&str>,
    status: Option<&str>,
    message: Option<&str>,
    done_frac: Option<f64>,
    preview: Option<HyperionPreview>,
) {
    let mut e = Event::new(0, event, t0.elapsed().as_secs_f64());
    e.rows = Some(rows as u64);
    e.cols = Some(cols as u64);
    e.output_dim = Some(output_dim as u64);
    e.done_frac = done_frac;
    e.hyperion_status = status.map(str::to_string);
    e.hyperion_engine = Some(engine.as_str().to_string());
    e.hyperion_device = device.map(str::to_string);
    e.hyperion_preview = preview;
    e.model = message.map(str::to_string);
    sink.emit(&e);
}

fn build_hyperion_preview(block: &Matrix) -> HyperionPreview {
    let mut scores = vec![0.0_f64; block.n];
    for col in 0..block.c.min(DEFAULT_OUTPUT_DIM) {
        let off = col * block.n;
        for (row, score) in scores.iter_mut().enumerate().take(block.n) {
            *score += block.data[off + row].abs() as f64;
        }
    }
    let z = robust_z_scores(&scores);
    let points = downsample_scores(&scores, PREVIEW_MAX_POINTS);
    let mut flagged: Vec<HyperionConcernRow> = scores
        .iter()
        .zip(&z)
        .enumerate()
        .filter_map(|(row, (&score, &z))| {
            (z >= CONCERN_Z_THRESHOLD).then_some(HyperionConcernRow {
                row: row as u64,
                score,
                z,
            })
        })
        .collect();
    flagged.sort_by(|a, b| {
        b.z.partial_cmp(&a.z)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.row.cmp(&b.row))
    });
    let top_rows = flagged.into_iter().take(PREVIEW_TOP_ROWS).collect();
    let concern_ranges = concern_ranges(&scores, &z, CONCERN_Z_THRESHOLD);
    HyperionPreview {
        rows: block.n as u64,
        threshold_z: CONCERN_Z_THRESHOLD,
        points,
        top_rows,
        concern_ranges,
    }
}

fn robust_z_scores(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let med = median(scores);
    let deviations = scores.iter().map(|s| (s - med).abs()).collect::<Vec<_>>();
    let mut mad = median(&deviations);
    if mad <= 1e-12 || !mad.is_finite() {
        mad = (med.abs() * 0.01).max(1e-6);
    }
    if !mad.is_finite() {
        return vec![0.0; scores.len()];
    }
    scores
        .iter()
        .map(|score| (0.6745 * (score - med) / mad).max(0.0))
        .collect()
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn downsample_scores(scores: &[f64], max_points: usize) -> Vec<HyperionPreviewPoint> {
    if scores.is_empty() || max_points == 0 {
        return Vec::new();
    }
    if scores.len() <= max_points {
        return scores
            .iter()
            .enumerate()
            .map(|(row, &score)| HyperionPreviewPoint {
                row: row as u64,
                score,
            })
            .collect();
    }
    let step = scores.len() as f64 / max_points as f64;
    (0..max_points)
        .filter_map(|i| {
            let start = (i as f64 * step).floor() as usize;
            let end = (((i + 1) as f64 * step).ceil() as usize).min(scores.len());
            if start >= end {
                return None;
            }
            let (row, score) = scores[start..end]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, &score)| (start + idx, score))?;
            Some(HyperionPreviewPoint {
                row: row as u64,
                score,
            })
        })
        .collect()
}

fn concern_ranges(scores: &[f64], z: &[f64], threshold: f64) -> Vec<HyperionConcernRange> {
    let mut ranges = Vec::new();
    let mut active: Option<HyperionConcernRange> = None;
    for row in 0..scores.len() {
        if z.get(row).copied().unwrap_or(0.0) < threshold {
            if let Some(range) = active.take() {
                ranges.push(range);
            }
            continue;
        }
        let score = scores[row];
        let zr = z[row];
        match active.as_mut() {
            Some(range) => {
                range.end = row as u64;
                if zr > range.max_z {
                    range.max_z = zr;
                    range.max_score = score;
                }
            }
            None => {
                active = Some(HyperionConcernRange {
                    start: row as u64,
                    end: row as u64,
                    max_score: score,
                    max_z: zr,
                });
            }
        }
    }
    if let Some(range) = active {
        ranges.push(range);
    }
    ranges
}

struct Runtime {
    settings: HyperionSweepSettings,
    projection: WeightProjection,
    identity: HyperionModelIdentity,
}

fn load_runtime(cfg: &HyperionConfig, output_dim: usize) -> Result<Runtime> {
    if !cfg.manifest_path.is_file() {
        bail!(
            "Hyperion manifest not found at {}",
            cfg.manifest_path.display()
        );
    }
    if !cfg.weights_path.is_file() {
        bail!(
            "Hyperion weights not found at {}",
            cfg.weights_path.display()
        );
    }
    let (settings, manifest_hash) = load_settings(&cfg.manifest_path, cfg.seed)?;
    if settings.dim < output_dim {
        bail!(
            "Hyperion manifest dim {} is smaller than requested output dim {output_dim}",
            settings.dim
        );
    }
    let (projection, tensor_count) =
        load_weight_projection(&cfg.weights_path, &settings, output_dim)?;
    let weights_hash = hash_file(&cfg.weights_path)?;
    let weights_bytes = std::fs::metadata(&cfg.weights_path)
        .with_context(|| format!("reading metadata for {}", cfg.weights_path.display()))?
        .len();
    Ok(Runtime {
        settings,
        projection,
        identity: HyperionModelIdentity {
            weights_path: cfg.weights_path.display().to_string(),
            manifest_path: cfg.manifest_path.display().to_string(),
            weights_bytes,
            weights_hash,
            manifest_hash,
            tensor_count,
        },
    })
}

fn load_settings(path: &Path, seed: u64) -> Result<(HyperionSweepSettings, String)> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let hash = hash_bytes(text.as_bytes());
    let raw: ManifestFile = serde_json::from_str(&text)
        .with_context(|| format!("parsing Hyperion manifest {}", path.display()))?;
    if raw.version != HYPERION_VERSION {
        bail!(
            "unsupported Hyperion manifest version '{}' (expected {HYPERION_VERSION})",
            raw.version
        );
    }
    if raw.dim == 0 || raw.ws_list.is_empty() {
        bail!("invalid Hyperion manifest: dim and ws_list must be non-empty");
    }
    let n_bands = if raw.n_bands == 0 {
        raw.ws_list.len()
    } else {
        raw.n_bands
    };
    if n_bands != raw.ws_list.len() {
        bail!(
            "invalid Hyperion manifest: n_bands={} but ws_list has {} entries",
            n_bands,
            raw.ws_list.len()
        );
    }
    Ok((
        HyperionSweepSettings {
            version: raw.version,
            step: raw.step,
            dim: raw.dim,
            heads: raw.heads,
            layers: raw.layers,
            emb_dim: if raw.emb_dim == 0 {
                raw.dim
            } else {
                raw.emb_dim
            },
            min_patch: raw.min_patch,
            max_patch: raw.max_patch,
            n_bands,
            q_low: raw.q_low,
            q_high: raw.q_high,
            quant_scale: raw.quant_scale,
            pad_u8: raw.pad_u8,
            ws_list: raw.ws_list,
            seed,
        },
        hash,
    ))
}

fn fit_quantile_bounds(x: &Matrix, q_low: f32, q_high: f32) -> Vec<HyperionQuantileBounds> {
    (0..x.c)
        .map(|j| {
            let mut vals: Vec<f32> = x.col(j).iter().copied().filter(|v| v.is_finite()).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if vals.is_empty() {
                return HyperionQuantileBounds {
                    low: 0.0,
                    high: 1.0,
                };
            }
            let low = percentile(&vals, q_low);
            let mut high = percentile(&vals, q_high);
            if !high.is_finite() || high <= low + 1e-12 {
                high = low + 1.0;
            }
            HyperionQuantileBounds { low, high }
        })
        .collect()
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 100.0) as f64 / 100.0;
    let idx = (q * (sorted.len().saturating_sub(1) as f64)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn encode_with_projection(
    x: &Matrix,
    bounds: &[HyperionQuantileBounds],
    settings: &HyperionSweepSettings,
    projection: &WeightProjection,
) -> Matrix {
    let output_dim = projection.output_dim;
    if x.n == 0 || output_dim == 0 {
        return Matrix::new(Vec::new(), x.n, output_dim);
    }
    let mut flat = vec![0.0; x.n * output_dim];
    if x.c == 0 || bounds.is_empty() {
        return Matrix::new(flat, x.n, output_dim);
    }
    let n_bands = settings.ws_list.len().min(projection.n_bands);
    let max_cols = x.c.min(2048);
    for row in 0..x.n {
        let mut band_ewma = vec![0.0_f32; n_bands];
        let mut band_abs = vec![0.0_f32; n_bands];
        let mut band_missing = vec![0.0_f32; n_bands];
        for (band, &window) in settings.ws_list.iter().take(n_bands).enumerate() {
            let alpha = (2.0_f32 / (window.max(1) as f32 + 1.0)).clamp(0.02, 1.0);
            let seed = mix64(settings.seed ^ ((row as u64) << 17) ^ (band as u64));
            let start = (seed as usize) % x.c;
            let mut stride = (mix64(seed ^ 0x9e37_79b9_7f4a_7c15) as usize % x.c).max(1);
            if stride.is_multiple_of(2) {
                stride += 1;
            }
            let mut ewma = 0.0_f32;
            let mut abs_ewma = 0.0_f32;
            let mut miss_ewma = 0.0_f32;
            for step in 0..max_cols {
                let col = (start + step.wrapping_mul(stride)) % x.c;
                let (q, missing) = normalized_cell(x.at(row, col), &bounds[col], settings);
                let cell = (q * projection.value_scale[band] + projection.value_bias[band]).tanh();
                ewma = alpha.mul_add(cell, (1.0 - alpha) * ewma);
                abs_ewma = alpha.mul_add(cell.abs(), (1.0 - alpha) * abs_ewma);
                miss_ewma = alpha.mul_add(missing, (1.0 - alpha) * miss_ewma);
            }
            band_ewma[band] = ewma;
            band_abs[band] = abs_ewma;
            band_missing[band] = miss_ewma;
        }
        for d in 0..output_dim {
            let mut z = projection.output_bias[d];
            for b in 0..n_bands {
                let off = b * output_dim + d;
                z += band_ewma[b] * projection.band_value[off];
                z += band_abs[b] * projection.band_abs[off];
                z += band_missing[b] * projection.band_missing[off];
            }
            flat[d * x.n + row] = z.tanh();
        }
    }
    Matrix::new(flat, x.n, output_dim)
}

fn normalized_cell(
    v: f32,
    bounds: &HyperionQuantileBounds,
    settings: &HyperionSweepSettings,
) -> (f32, f32) {
    if !v.is_finite() {
        let centered_pad = (settings.pad_u8 as f32 - 128.0) / 127.0;
        return (centered_pad.clamp(-1.0, 1.0), 1.0);
    }
    let span = (bounds.high - bounds.low).max(1e-12);
    let norm = ((v - bounds.low) / span).clamp(0.0, 1.0);
    (
        ((norm * 2.0 - 1.0) * settings.quant_scale).clamp(-8.0, 8.0),
        0.0,
    )
}

fn load_projection_for_state(state: &HyperionState) -> Result<WeightProjection> {
    #[cfg(not(feature = "hyperion-cpu"))]
    {
        let _ = state;
        bail!(
            "artifact requires Hyperion but feat-core was compiled without the 'hyperion-cpu' feature"
        );
    }
    #[cfg(feature = "hyperion-cpu")]
    {
        let path = PathBuf::from(&state.model.weights_path);
        if !path.is_file() {
            bail!(
                "artifact requires Hyperion weights but {} is unavailable",
                path.display()
            );
        }
        let current_hash = hash_file(&path)?;
        if current_hash != state.model.weights_hash {
            bail!(
                "artifact requires Hyperion weights hash {} but {} has hash {}",
                state.model.weights_hash,
                path.display(),
                current_hash
            );
        }
        load_weight_projection(&path, &state.sweep, state.output_dim)
            .map(|(projection, _)| projection)
    }
}

#[cfg(feature = "hyperion-cpu")]
fn load_weight_projection(
    path: &Path,
    settings: &HyperionSweepSettings,
    output_dim: usize,
) -> Result<(WeightProjection, usize)> {
    use candle_core::safetensors::MmapedSafetensors;

    // SAFETY: Candle's API marks memmap creation unsafe because OS file mappings can be invalidated
    // externally. Jain treats model artifacts as immutable during a run and never mutates this file.
    let tensors = unsafe { MmapedSafetensors::new(path) }
        .map_err(|e| anyhow::anyhow!("opening Hyperion safetensors {}: {e}", path.display()))?;
    let tensor_count = tensors.tensors().len();

    // Projection extraction is tiny and feeds both engines, so keep it on CPU. That avoids a GPU
    // allocation becoming a hard preflight failure before v7 has a chance to retry on CPU.
    let device = candle_core::Device::Cpu;
    let val0_w = load_tensor_vec(&tensors, "val.0.weight", &device)?;
    let val0_b = load_tensor_vec(&tensors, "val.0.bias", &device)?;
    let val2_w = load_tensor_vec(&tensors, "val.2.weight", &device)?;
    let val2_b = load_tensor_vec(&tensors, "val.2.bias", &device)?;
    let row0_w = load_tensor_vec(&tensors, "row_pool.proj.0.weight", &device)?;
    let row0_b = load_tensor_vec(&tensors, "row_pool.proj.0.bias", &device)?;
    let row2_w = load_tensor_vec(&tensors, "row_pool.proj.2.weight", &device)?;
    let row2_b = load_tensor_vec(&tensors, "row_pool.proj.2.bias", &device)?;
    let scale = load_tensor_vec(&tensors, "scale_emb.weight", &device)?;

    let n_bands = settings.n_bands;
    let mut value_scale = vec![0.0; n_bands];
    let mut value_bias = vec![0.0; n_bands];
    for b in 0..n_bands {
        let k = b * 37 % 512;
        value_scale[b] = val0_w.get(k * 2).copied().unwrap_or(0.25);
        value_bias[b] = val0_b.get(k).copied().unwrap_or(0.0)
            + scale.get((b % 23) * 512 + k).copied().unwrap_or(0.0) * 0.05;
    }

    let mut band_value = vec![0.0; n_bands * output_dim];
    let mut band_abs = vec![0.0; n_bands * output_dim];
    let mut band_missing = vec![0.0; n_bands * output_dim];
    let mut output_bias = vec![0.0; output_dim];
    for b in 0..n_bands {
        for d in 0..output_dim {
            let off = b * output_dim + d;
            let k = (d * 29 + b * 17) % 512;
            let val2 = val2_w.get(d * 512 + k).copied().unwrap_or(0.0);
            let pool_a = row0_w.get(d * 1536 + k).copied().unwrap_or(0.0);
            let pool_b = row0_w.get(d * 1536 + 512 + k).copied().unwrap_or(0.0);
            let pool_c = row2_w.get(d * 512 + k).copied().unwrap_or(0.0);
            let scale_b = scale.get((b % 23) * 512 + k).copied().unwrap_or(0.0);
            band_value[off] = squash_weight(val2 + pool_a * 0.5 + scale_b * 0.1);
            band_abs[off] = squash_weight(pool_b + pool_c * 0.25);
            band_missing[off] =
                squash_weight(scale_b + val0_w.get(k * 2 + 1).copied().unwrap_or(0.0));
        }
    }
    for (d, ob) in output_bias.iter_mut().enumerate() {
        *ob = squash_weight(
            val2_b.get(d).copied().unwrap_or(0.0)
                + row0_b.get(d).copied().unwrap_or(0.0)
                + row2_b.get(d).copied().unwrap_or(0.0),
        );
    }
    Ok((
        WeightProjection {
            n_bands,
            output_dim,
            value_scale,
            value_bias,
            band_value,
            band_abs,
            band_missing,
            output_bias,
        },
        tensor_count,
    ))
}

/// Compute-device candidates for Hyperion v7 tensor loads. CUDA is opportunistic: use it only when
/// available and the host load guard allows it, then keep CPU as the retry/fallback path.
#[cfg(feature = "hyperion-cpu")]
pub(crate) fn hyperion_device_candidates() -> Vec<candle_core::Device> {
    let mut devices = Vec::new();
    #[cfg(feature = "hyperion-cuda")]
    {
        if hyperion_cuda_allowed_by_load() {
            if let Ok(device) = candle_core::Device::cuda_if_available(0) {
                if device.is_cuda() {
                    devices.push(device);
                }
            }
        }
    }
    devices.push(candle_core::Device::Cpu);
    devices
}

/// Compute device for Hyperion tensor loads: prefer an allowed CUDA GPU, and use the CPU when no
/// accelerator is present, CUDA initialization fails, the host is loaded, or the build is CPU-only.
#[cfg(feature = "hyperion-cpu")]
pub(crate) fn hyperion_device() -> candle_core::Device {
    hyperion_device_candidates()
        .into_iter()
        .next()
        .unwrap_or(candle_core::Device::Cpu)
}

/// Human-readable label for the device Hyperion would use (for status/diagnostics).
#[cfg(feature = "hyperion-cpu")]
pub fn hyperion_device_label() -> &'static str {
    hyperion_device_label_for(&hyperion_device())
}

#[cfg(not(feature = "hyperion-cpu"))]
pub fn hyperion_device_label() -> &'static str {
    "cpu"
}

#[cfg(feature = "hyperion-cpu")]
pub(crate) fn hyperion_device_label_for(device: &candle_core::Device) -> &'static str {
    if device.is_cuda() {
        "cuda:0"
    } else {
        "cpu"
    }
}

#[cfg(feature = "hyperion-cuda")]
fn hyperion_cuda_allowed_by_load() -> bool {
    let max_ratio = std::env::var("JAIN_HYPERION_CUDA_MAX_LOAD_RATIO")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(0.90);
    let Some(load1) = linux_loadavg_1m() else {
        return true;
    };
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1) as f64;
    load1 / cores < max_ratio
}

#[cfg(feature = "hyperion-cuda")]
fn linux_loadavg_1m() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/loadavg").ok()?;
    text.split_whitespace().next()?.parse::<f64>().ok()
}

#[cfg(feature = "hyperion-cpu")]
fn load_tensor_vec(
    tensors: &candle_core::safetensors::MmapedSafetensors,
    name: &str,
    device: &candle_core::Device,
) -> Result<Vec<f32>> {
    let tensor = tensors
        .load(name, device)
        .map_err(|e| anyhow::anyhow!("loading Hyperion tensor '{name}': {e}"))?;
    tensor
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| anyhow::anyhow!("reading Hyperion tensor '{name}' as f32: {e}"))
}

#[cfg(not(feature = "hyperion-cpu"))]
fn load_weight_projection(
    _path: &Path,
    _settings: &HyperionSweepSettings,
    _output_dim: usize,
) -> Result<(WeightProjection, usize)> {
    bail!("feat-core was compiled without the 'hyperion-cpu' feature")
}

#[cfg(feature = "hyperion-cpu")]
fn squash_weight(v: f32) -> f32 {
    (v * 0.25).tanh()
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut buf = [0u8; 64 * 1024];
    let mut acc = 0x4859_5045_5249_4f4e_u64;
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        for chunk in buf[..n].chunks(8) {
            let mut word = 0u64;
            for (i, b) in chunk.iter().enumerate() {
                word |= (*b as u64) << (i * 8);
            }
            acc = mix64(acc ^ word);
        }
    }
    Ok(format!("{acc:016x}"))
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut acc = 0x484d_414e_4946_4553_u64;
    for chunk in bytes.chunks(8) {
        let mut word = 0u64;
        for (i, b) in chunk.iter().enumerate() {
            word |= (*b as u64) << (i * 8);
        }
        acc = mix64(acc ^ word);
    }
    format!("{acc:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Matrix;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn projection(n_bands: usize, output_dim: usize) -> WeightProjection {
        let mut band_value = vec![0.0; n_bands * output_dim];
        let mut band_abs = vec![0.0; n_bands * output_dim];
        let mut band_missing = vec![0.0; n_bands * output_dim];
        for b in 0..n_bands {
            for d in 0..output_dim {
                let off = b * output_dim + d;
                band_value[off] = ((b + d + 1) as f32 * 0.013).sin() * 0.3;
                band_abs[off] = ((b * 7 + d + 3) as f32 * 0.017).cos() * 0.2;
                band_missing[off] = ((b * 5 + d + 11) as f32 * 0.019).sin() * 0.1;
            }
        }
        WeightProjection {
            n_bands,
            output_dim,
            value_scale: (0..n_bands).map(|b| 0.5 + b as f32 * 0.03).collect(),
            value_bias: vec![0.0; n_bands],
            band_value,
            band_abs,
            band_missing,
            output_bias: vec![0.0; output_dim],
        }
    }

    fn settings() -> HyperionSweepSettings {
        HyperionSweepSettings {
            version: HYPERION_VERSION.into(),
            step: 2000,
            dim: 512,
            heads: 8,
            layers: 12,
            emb_dim: 512,
            min_patch: 2,
            max_patch: 24,
            n_bands: 3,
            q_low: 1.0,
            q_high: 99.0,
            quant_scale: 3.0,
            pad_u8: 128,
            ws_list: vec![4, 7, 11],
            seed: 42,
        }
    }

    fn restore_env(key: &str, value: Option<OsString>) {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    fn with_hyperion_sweeps_env(value: Option<&str>, f: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().expect("env lock poisoned");
        let key = "JAIN_HYPERION_SWEEPS";
        let previous = std::env::var_os(key);
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        f();
        restore_env(key, previous);
    }

    #[test]
    fn default_sweeps_is_thirty_and_env_override_still_works() {
        assert_eq!(HyperionConfig::default().sweeps, 30);
        with_hyperion_sweeps_env(None, || {
            assert_eq!(HyperionConfig::from_env().unwrap().sweeps, 30);
        });
        with_hyperion_sweeps_env(Some("4"), || {
            assert_eq!(HyperionConfig::from_env().unwrap().sweeps, 4);
        });
        with_hyperion_sweeps_env(Some("0"), || {
            assert_eq!(HyperionConfig::from_env().unwrap().sweeps, 1);
        });
    }

    #[test]
    fn output_names_are_stable() {
        let names = hyperion_output_names(128);
        assert_eq!(names.len(), 128);
        assert_eq!(names[0], "hyperion::v000");
        assert_eq!(names[127], "hyperion::v127");
    }

    #[test]
    fn train_bounds_replay_holdout_shape_and_order() {
        let train = Matrix::new(
            vec![
                0.0, 0.5, 1.0, 1.5, // col 0
                1.0, 0.0, 1.0, 0.0, // col 1
            ],
            4,
            2,
        );
        let hold = Matrix::new(vec![0.25, 0.75, 1.0, 0.0], 2, 2);
        let bounds = fit_quantile_bounds(&train, 1.0, 99.0);
        let settings = settings();
        let projection = projection(settings.n_bands, 128);
        let h1 = encode_with_projection(&hold, &bounds, &settings, &projection);
        let h2 = encode_with_projection(&hold, &bounds, &settings, &projection);
        assert_eq!(h1.n, 2);
        assert_eq!(h1.c, 128);
        assert_eq!(h1.data, h2.data);
        assert!(h1.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn degenerate_matrices_encode_to_valid_values() {
        let x = Matrix::new(Vec::new(), 3, 0);
        let settings = settings();
        let projection = projection(settings.n_bands, 128);
        let encoded = encode_with_projection(&x, &[], &settings, &projection);
        assert_eq!(encoded.n, 3);
        assert_eq!(encoded.c, 128);
        assert!(encoded.data.iter().all(|v| *v == 0.0));

        let x = Matrix::new(vec![f32::NAN, f32::NAN, f32::NAN], 3, 1);
        let bounds = fit_quantile_bounds(&x, 1.0, 99.0);
        let encoded = encode_with_projection(&x, &bounds, &settings, &projection);
        assert!(encoded.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preview_flags_robust_row_concerns_without_raw_values() {
        let mut data = Vec::new();
        for col in 0..128 {
            for row in 0..12 {
                let value = if (5..=6).contains(&row) {
                    0.9
                } else {
                    0.02 + col as f32 * 0.00001
                };
                data.push(value);
            }
        }
        let block = Matrix::new(data, 12, 128);
        let preview = build_hyperion_preview(&block);
        assert_eq!(preview.rows, 12);
        assert_eq!(preview.points.len(), 12);
        assert!(preview
            .top_rows
            .iter()
            .any(|row| row.row == 5 || row.row == 6));
        assert_eq!(preview.concern_ranges[0].start, 5);
        assert_eq!(preview.concern_ranges[0].end, 6);
    }

    #[test]
    fn missing_weights_modes_differ() {
        let cfg = HyperionConfig {
            mode: HyperionMode::Auto,
            weights_path: PathBuf::from("/definitely/missing/model.safetensors"),
            manifest_path: PathBuf::from("/definitely/missing/manifest.json"),
            ..HyperionConfig::default()
        };
        let x = Matrix::new(vec![0.0, 1.0], 2, 1);
        let names = vec!["x".to_string()];
        let auto = fit_transform_pair(
            &x,
            &x,
            &names,
            &cfg,
            &crate::progress::NullSink,
            Instant::now(),
        )
        .unwrap();
        assert!(auto.state.is_none());
        assert_eq!(auto.train.c, 1);

        let mut on = cfg;
        on.mode = HyperionMode::On;
        let err = match fit_transform_pair(
            &x,
            &x,
            &names,
            &on,
            &crate::progress::NullSink,
            Instant::now(),
        ) {
            Ok(_) => panic!("Hyperion on unexpectedly succeeded with missing weights"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("Hyperion manifest not found"));

        let mut off = on;
        off.mode = HyperionMode::Off;
        let skipped = fit_transform_pair(
            &x,
            &x,
            &names,
            &off,
            &crate::progress::NullSink,
            Instant::now(),
        )
        .unwrap();
        assert!(skipped.state.is_none());
        assert_eq!(skipped.train.c, 1);
    }
}
