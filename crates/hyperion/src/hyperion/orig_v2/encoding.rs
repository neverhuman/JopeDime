use super::*;

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
mod tests;
