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
    let gpu_device =
        device.is_some_and(|value| value.starts_with("cuda") || value.starts_with("gpu"));
    e.hyperion_backend = Some(if gpu_device {
        "cellmat-cuda".into()
    } else {
        "cellmat-cpu".into()
    });
    e.hyperion_elapsed_ms = Some(t0.elapsed().as_millis().min(u64::MAX as u128) as u64);
    e.hyperion_fallback = Some(!gpu_device);
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

pub fn load_settings(path: &Path, seed: u64) -> Result<(HyperionSweepSettings, String)> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse_settings(&text, seed, &path.display().to_string())
}

/// [`load_settings`] over manifest text the caller already holds, for example bytes it has just
/// authenticated: nothing is reopened by path. `label` names the source in parse errors.
pub fn parse_settings(
    text: &str,
    seed: u64,
    label: &str,
) -> Result<(HyperionSweepSettings, String)> {
    let hash = hash_bytes(text.as_bytes());
    let raw: ManifestFile =
        serde_json::from_str(text).with_context(|| format!("parsing Hyperion manifest {label}"))?;
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

pub fn fit_quantile_bounds(x: &Matrix, q_low: f32, q_high: f32) -> Vec<HyperionQuantileBounds> {
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
/// available and the host load guard allows it, then keep CPU as the retry path.
#[cfg(feature = "hyperion-cpu")]
pub(crate) fn hyperion_device_candidates() -> Vec<candle_core::Device> {
    let mut devices = Vec::new();
    #[cfg(feature = "hyperion-cuda")]
    {
        if !hyperion_cuda_allowed_by_load() {
            eprintln!(
                "[hyperion] {}",
                battle_gpu::queue::gpu_cpu_handoff_reason(
                    "Hyperion CUDA device selection",
                    "host load guard exceeded JAIN_HYPERION_CUDA_MAX_LOAD_RATIO"
                )
            );
        } else {
            match candle_core::Device::cuda_if_available(0) {
                Ok(device) if device.is_cuda() => devices.push(device),
                Ok(_) => eprintln!(
                    "[hyperion] {}",
                    battle_gpu::queue::gpu_cpu_handoff_reason(
                        "Hyperion CUDA device selection",
                        "CUDA device 0 is not available"
                    )
                ),
                Err(err) => eprintln!(
                    "[hyperion] {}",
                    battle_gpu::queue::gpu_cpu_handoff_reason(
                        "Hyperion CUDA device selection",
                        format!("{err}")
                    )
                ),
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
