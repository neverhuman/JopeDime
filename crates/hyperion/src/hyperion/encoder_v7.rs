// jankurai:allow repo-rot.path.fake-versioned-source reason=canonical-hyperion-encoder-v7-surface-owned-by-jain-core-proof-required owner=hyperion expires=2027-04-01
use super::*;

/// v7 encode result: feature blocks (column-major) + the confidence surface + dataset vector.
pub(super) struct V7Fit {
    train_block: Matrix,
    hold_block: Matrix,
    confidence: Matrix,
    hold_confidence: Matrix,
    dataset_vector: Vec<f32>,
    device: Option<&'static str>,
    cancelled: bool,
    advanced: bool,
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
pub(super) fn fit_v7(
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
    if cfg.require_gpu && !device.starts_with("cuda") && !device.starts_with("gpu") {
        bail!("Hyperion require_gpu requested but cellmat loaded on {device}; no CPU recovery ran");
    }

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
                crate::hyperion::HyperionEvent {
                    rows: train.n,
                    cols: train.c,
                    output_dim,
                    engine: HyperionEngine::V7,
                    device: Some(device),
                    status: Some("running"),
                    message: None,
                    done_frac: Some(done_frac),
                    preview: None,
                },
            );
        }
    };

    let train_out = run_composite_sweep(
        &model,
        &xu8_train,
        crate::hyperion_v7::sweep::PatchInput {
            pad_u8: settings.pad_u8,
            quant_scale: settings.quant_scale,
            max_patch: settings.max_patch,
        },
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
            advanced: false,
        });
    }
    if train_out.advanced {
        return Ok(V7Fit {
            train_block: Matrix::new(Vec::new(), 0, 0),
            hold_block: Matrix::new(Vec::new(), 0, 0),
            confidence: Matrix::new(Vec::new(), 0, 0),
            hold_confidence: Matrix::new(Vec::new(), 0, 0),
            dataset_vector: Vec::new(),
            device: Some(device),
            cancelled: false,
            advanced: true,
        });
    }

    let hold_out = run_composite_sweep(
        &model,
        &xu8_hold,
        crate::hyperion_v7::sweep::PatchInput {
            pad_u8: settings.pad_u8,
            quant_scale: settings.quant_scale,
            max_patch: settings.max_patch,
        },
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
            advanced: false,
        });
    }
    if hold_out.advanced {
        return Ok(V7Fit {
            train_block: Matrix::new(Vec::new(), 0, 0),
            hold_block: Matrix::new(Vec::new(), 0, 0),
            confidence: Matrix::new(Vec::new(), 0, 0),
            hold_confidence: Matrix::new(Vec::new(), 0, 0),
            dataset_vector: Vec::new(),
            device: Some(device),
            cancelled: false,
            advanced: true,
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
        advanced: false,
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
        crate::hyperion_v7::sweep::PatchInput {
            pad_u8: settings.pad_u8,
            quant_scale: settings.quant_scale,
            max_patch: settings.max_patch,
        },
        &plan,
        &NullSink,
        |_| {},
    )
}

/// Reusable standalone v7 encoder for batch/offline jobs that need to encode many matrices with
/// the same checkpoint. This keeps the large v7 weights resident instead of reloading them for
/// every dataset.
#[cfg(feature = "hyperion-cpu")]
pub struct V7DatasetEncoder {
    settings: HyperionSweepSettings,
    model: crate::hyperion_v7::backend::V7Model,
    batch: usize,
}

#[cfg(feature = "hyperion-cpu")]
impl V7DatasetEncoder {
    pub fn load(
        weights_path: &Path,
        manifest_path: &Path,
        seed: u64,
        batch: usize,
    ) -> Result<Self> {
        use crate::hyperion_v7::backend::V7Model;

        let (settings, _hash) = load_settings(manifest_path, seed)?;
        let model = V7Model::load(weights_path, &settings)?;
        Ok(Self {
            settings,
            model,
            batch: batch.max(1),
        })
    }

    pub fn settings(&self) -> &HyperionSweepSettings {
        &self.settings
    }

    pub fn device_label(&self) -> &'static str {
        self.model.device_label()
    }

    pub fn encode_matrix(
        &self,
        x: &Matrix,
        sweeps: u32,
        scales: &[usize],
    ) -> Result<crate::hyperion_v7::sweep::SweepOutput> {
        use crate::hyperion_v7::config::SweepPlan;
        use crate::hyperion_v7::grid::quantize_u8_with_bounds;
        use crate::hyperion_v7::sweep::run_composite_sweep;
        use crate::progress::NullSink;

        let mut scales: Vec<usize> = scales
            .iter()
            .copied()
            .filter(|&p| p >= 2 && p <= self.settings.max_patch)
            .collect();
        if scales.is_empty() {
            scales = vec![3, 5, 7, 9];
        }
        let plan = SweepPlan {
            sweeps: sweeps.max(1),
            scales,
            batch: self.batch,
            seed: self.settings.seed,
            target_chunks_per_sweep: 8,
            fixed_perms: None,
        };
        let bounds = fit_quantile_bounds(x, self.settings.q_low, self.settings.q_high);
        let xu8 = quantize_u8_with_bounds(x, &bounds, self.settings.pad_u8);
        run_composite_sweep(
            &self.model,
            &xu8,
            crate::hyperion_v7::sweep::PatchInput {
                pad_u8: self.settings.pad_u8,
                quant_scale: self.settings.quant_scale,
                max_patch: self.settings.max_patch,
            },
            &plan,
            &NullSink,
            |_| {},
        )
    }

    pub fn dataset_vector(&self, x: &Matrix, sweeps: u32, scales: &[usize]) -> Result<Vec<f32>> {
        if std::env::var("JAIN_HYPERION_NO_NORMALIZE").is_ok_and(|v| v == "1") {
            use crate::hyperion_v7::config::SweepPlan;
            use crate::hyperion_v7::grid::quantize_u8_with_bounds;
            use crate::hyperion_v7::sweep::run_raw_dataset_vector_sweep;

            let mut scales: Vec<usize> = scales
                .iter()
                .copied()
                .filter(|&p| p >= 2 && p <= self.settings.max_patch)
                .collect();
            if scales.is_empty() {
                scales = vec![3, 5, 7, 9];
            }
            let plan = SweepPlan {
                sweeps: sweeps.max(1),
                scales,
                batch: self.batch,
                seed: self.settings.seed,
                target_chunks_per_sweep: 8,
                fixed_perms: None,
            };
            let bounds = fit_quantile_bounds(x, self.settings.q_low, self.settings.q_high);
            let xu8 = quantize_u8_with_bounds(x, &bounds, self.settings.pad_u8);
            return run_raw_dataset_vector_sweep(
                &self.model,
                &xu8,
                self.settings.pad_u8,
                self.settings.quant_scale,
                self.settings.max_patch,
                &plan,
            );
        }
        Ok(self.encode_matrix(x, sweeps, scales)?.dataset_vec)
    }
}

/// Stub when the model stack is compiled out: engine=v7 behaves as a load failure (Auto → skip,
/// On → fail), mirroring the classic weight-loading semantics.
#[cfg(not(feature = "hyperion-cpu"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_v7(
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
