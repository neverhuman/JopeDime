/// v7 encode result: feature blocks (column-major) + the confidence surface + dataset vector.
struct V7Fit {
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
        settings.pad_u8,
        settings.quant_scale,
        settings.max_patch,
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
            self.settings.pad_u8,
            self.settings.quant_scale,
            self.settings.max_patch,
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
    if state.engine == "orig_v2" {
        return transform_scoring_orig_v2(encoded, state);
    }
    let projection = load_projection_for_state(state)?;
    let block = encode_with_projection(encoded, &state.quantile_bounds, &state.sweep, &projection);
    Ok(place_columns(encoded, &block, state.feature_placement))
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
    Ok(place_columns(encoded, &block, state.feature_placement))
}

#[cfg(not(feature = "hyperion-cpu"))]
fn transform_scoring_v7(_encoded: &Matrix, _state: &HyperionState) -> Result<Matrix> {
    bail!(
        "artifact was encoded with the Hyperion v7 engine but feat-core was compiled without the 'hyperion-cpu' feature"
    )
}

#[cfg(feature = "hyperion-cpu")]
fn transform_scoring_orig_v2(encoded: &Matrix, state: &HyperionState) -> Result<Matrix> {
    use crate::hyperion_v7::backend::V7Model;
    use crate::hyperion_v7::sweep::run_orig_v2_cell_sweep;

    let orig = state
        .orig_v2
        .as_ref()
        .context("artifact orig-v2 engine is missing its train-fitted rank state")?;
    if orig.sorted_fit_columns.len() != encoded.c {
        bail!(
            "artifact orig-v2 rank state has {} columns but scoring has {}",
            orig.sorted_fit_columns.len(),
            encoded.c
        );
    }
    let weights = PathBuf::from(&state.model.weights_path);
    let manifest = PathBuf::from(&state.model.manifest_path);
    orig_v2::validate_authority(&weights, orig_v2::ORIG_V2_WEIGHTS_SHA256, "weights")?;
    orig_v2::validate_authority(&manifest, orig_v2::ORIG_V2_MANIFEST_SHA256, "manifest")?;
    let weights_hash = hash_file(&weights)?;
    let manifest_hash = hash_file(&manifest)?;
    if weights_hash != state.model.weights_hash || manifest_hash != state.model.manifest_hash {
        bail!("artifact orig-v2 model authority hash mismatch");
    }
    let model = V7Model::load(&weights, &state.sweep)?;
    let grid = orig_v2::rank_replay_grid(
        encoded,
        &orig.sorted_fit_columns,
        state.sweep.q_low,
        state.sweep.q_high,
    )?;
    let output = run_orig_v2_cell_sweep(
        &model,
        &grid,
        state.sweep.quant_scale,
        state.sweep.max_patch,
        &orig.plan(),
    )?;
    let block = orig_v2::row_major_cells(&output.logits, 0, encoded.n, encoded.c);
    Ok(place_columns(encoded, &block, state.feature_placement))
}

#[cfg(not(feature = "hyperion-cpu"))]
fn transform_scoring_orig_v2(_encoded: &Matrix, _state: &HyperionState) -> Result<Matrix> {
    bail!(
        "artifact was encoded with Hyperion orig-v2 but feat-core was compiled without the 'hyperion-cpu' feature"
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

pub fn prepend_columns(base: &Matrix, extra: &Matrix) -> Matrix {
    assert_eq!(base.n, extra.n, "prepended matrix row count mismatch");
    let mut data = Vec::with_capacity(base.data.len() + extra.data.len());
    data.extend_from_slice(extra.data.as_ref());
    data.extend_from_slice(base.data.as_ref());
    Matrix::new(data, base.n, base.c + extra.c)
}

pub fn place_columns(base: &Matrix, extra: &Matrix, placement: HyperionFeaturePlacement) -> Matrix {
    match placement {
        HyperionFeaturePlacement::AppendRight => append_columns(base, extra),
        HyperionFeaturePlacement::PrependLeft => prepend_columns(base, extra),
    }
}

pub fn place_names(
    input_names: &[String],
    output_names: &[String],
    placement: HyperionFeaturePlacement,
) -> Vec<String> {
    let mut names = Vec::with_capacity(input_names.len() + output_names.len());
    match placement {
        HyperionFeaturePlacement::AppendRight => {
            names.extend(input_names.iter().cloned());
            names.extend(output_names.iter().cloned());
        }
        HyperionFeaturePlacement::PrependLeft => {
            names.extend(output_names.iter().cloned());
            names.extend(input_names.iter().cloned());
        }
    }
    names
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
