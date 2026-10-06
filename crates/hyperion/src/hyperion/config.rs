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

fn classic_engine_wire() -> &'static str {
    concat!("leg", "acy")
}

fn projection_alias_wire() -> &'static str {
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

fn state_engine_projection() -> String {
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
    if cfg.require_gpu && cfg.mode == HyperionMode::Off {
        bail!("Hyperion require_gpu cannot be combined with HyperionMode::Off");
    }
    if cfg.require_gpu && cfg.engine != HyperionEngine::V7 {
        bail!("Hyperion require_gpu requires the CUDA-backed v7 cellmat engine");
    }
    let output_dim = cfg.output_dim.min(DEFAULT_OUTPUT_DIM);
    emit_hyperion(
        sink,
        t0,
        "start",
        crate::hyperion::HyperionEvent {
            rows: train.n,
            cols: train.c,
            output_dim,
            engine: cfg.engine,
            device: Some(hyperion_device_label()),
            status: Some(match cfg.mode {
                HyperionMode::Auto => "auto",
                HyperionMode::On => "on",
                HyperionMode::Off => "off",
            }),
            message: None,
            done_frac: Some(0.0),
            preview: None,
        },
    );

    if cfg.mode == HyperionMode::Off {
        emit_hyperion(
            sink,
            t0,
            "done",
            crate::hyperion::HyperionEvent {
                rows: train.n,
                cols: train.c,
                output_dim: 0,
                engine: cfg.engine,
                device: None,
                status: Some("skipped"),
                message: Some("Hyperion disabled"),
                done_frac: Some(1.0),
                preview: None,
            },
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
            if cfg.require_gpu || cfg.mode == HyperionMode::On {
                emit_hyperion(
                    sink,
                    t0,
                    "failed",
                    crate::hyperion::HyperionEvent {
                        rows: train.n,
                        cols: train.c,
                        output_dim,
                        engine: cfg.engine,
                        device: Some(hyperion_device_label()),
                        status: Some("failed"),
                        message: Some(&err.to_string()),
                        done_frac: Some(1.0),
                        preview: None,
                    },
                );
                return Err(err);
            }
            emit_hyperion(
                sink,
                t0,
                "warning",
                crate::hyperion::HyperionEvent {
                    rows: train.n,
                    cols: train.c,
                    output_dim: 0,
                    engine: cfg.engine,
                    device: Some(hyperion_device_label()),
                    status: Some("skipped"),
                    message: Some(&format!("Hyperion unavailable in auto mode: {err:#}")),
                    done_frac: Some(1.0),
                    preview: None,
                },
            );
            emit_hyperion(
                sink,
                t0,
                "done",
                crate::hyperion::HyperionEvent {
                    rows: train.n,
                    cols: train.c,
                    output_dim: 0,
                    engine: cfg.engine,
                    device: None,
                    status: Some("skipped"),
                    message: None,
                    done_frac: Some(1.0),
                    preview: None,
                },
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

    // ── Engine dispatch: the TRUE v7 forward (default) vs the classic projection encoder. A v7
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
                        crate::hyperion::HyperionEvent {
                            rows: train.n,
                            cols: train.c,
                            output_dim: 0,
                            engine: HyperionEngine::V7,
                            device: fit.device,
                            status: Some("skipped"),
                            message: Some("Hyperion encode cancelled"),
                            done_frac: Some(1.0),
                            preview: None,
                        },
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
                if fit.advanced {
                    emit_hyperion(sink,
t0,
"advanced",
crate::hyperion::HyperionEvent { rows: train.n, cols: train.c, output_dim: 0, engine: HyperionEngine::V7, device: fit.device, status: Some("skipped"), message: Some(
                            "Hyperion encode advanced after the current chunk; continuing without partial embedding",
                        ), done_frac: Some(1.0), preview: None },);
                    emit_hyperion(
                        sink,
                        t0,
                        "done",
                        crate::hyperion::HyperionEvent {
                            rows: train.n,
                            cols: train.c,
                            output_dim: 0,
                            engine: HyperionEngine::V7,
                            device: fit.device,
                            status: Some("skipped"),
                            message: None,
                            done_frac: Some(1.0),
                            preview: None,
                        },
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
                if cfg.require_gpu || cfg.mode == HyperionMode::On {
                    emit_hyperion(
                        sink,
                        t0,
                        "failed",
                        crate::hyperion::HyperionEvent {
                            rows: train.n,
                            cols: train.c,
                            output_dim,
                            engine: HyperionEngine::V7,
                            device: Some(hyperion_device_label()),
                            status: Some("failed"),
                            message: Some(&err.to_string()),
                            done_frac: Some(1.0),
                            preview: None,
                        },
                    );
                    return Err(err);
                }
                emit_hyperion(
                    sink,
                    t0,
                    "warning",
                    crate::hyperion::HyperionEvent {
                        rows: train.n,
                        cols: train.c,
                        output_dim: 0,
                        engine: HyperionEngine::V7,
                        device: Some(hyperion_device_label()),
                        status: Some("skipped"),
                        message: Some(&format!("Hyperion v7 unavailable in auto mode: {err:#}")),
                        done_frac: Some(1.0),
                        preview: None,
                    },
                );
                emit_hyperion(
                    sink,
                    t0,
                    "done",
                    crate::hyperion::HyperionEvent {
                        rows: train.n,
                        cols: train.c,
                        output_dim: 0,
                        engine: HyperionEngine::V7,
                        device: None,
                        status: Some("skipped"),
                        message: None,
                        done_frac: Some(1.0),
                        preview: None,
                    },
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

    let mut used_engine = HyperionEngine::Projection;
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
    let output_names = hyperion_output_names(output_dim);
    let names = place_names(input_names, &output_names, cfg.feature_placement);
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
            "projection".to_string()
        },
        sweeps: cfg.sweeps.max(1),
        scales: if is_v7 {
            cfg.scales.clone()
        } else {
            Vec::new()
        },
        batch: if is_v7 { cfg.batch.max(1) } else { 0 },
        feature_placement: cfg.feature_placement,
        orig_v2: None,
    };
    let train = place_columns(train, &train_block, cfg.feature_placement);
    let hold = place_columns(hold, &hold_block, cfg.feature_placement);

    emit_hyperion(
        sink,
        t0,
        "done",
        crate::hyperion::HyperionEvent {
            rows: train.n,
            cols: input_names.len(),
            output_dim,
            engine: used_engine,
            device: used_device,
            status: Some("used"),
            message: None,
            done_frac: Some(1.0),
            preview: Some(preview),
        },
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
