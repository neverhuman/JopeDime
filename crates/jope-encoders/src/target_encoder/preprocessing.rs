use super::*;
use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;

pub(super) fn linear_tensor(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let dimensions = x.dims().to_vec();
    let input_width = *dimensions
        .last()
        .context("target linear input has no dimensions")?;
    let rows = x.elem_count() / input_width;
    let output_width = weight.dim(0)?;
    let mut output = x
        .reshape((rows, input_width))?
        .matmul(&weight.to_dtype(DType::F32)?.t()?)?;
    if let Some(bias) = bias {
        output = output.broadcast_add(&bias.to_dtype(DType::F32)?)?;
    }
    let mut output_shape = dimensions;
    *output_shape.last_mut().expect("linear shape is nonempty") = output_width;
    Ok(output.reshape(output_shape)?)
}

pub(super) fn preprocess(values: &[f32], task: TargetTask) -> Result<Vec<f32>> {
    let values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| match task {
            TargetTask::Binary => f32::from(value >= 0.5),
            TargetTask::Regression => value.clamp(0.0, 1.0),
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        bail!("cannot encode an empty target");
    }
    Ok(values)
}

pub(super) fn safe_unit(values: &[f32]) -> Vec<f32> {
    let values = values
        .iter()
        .map(|value| if value.is_finite() { *value } else { 0.0 })
        .collect::<Vec<_>>();
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 1e-12 || norm.is_nan() {
        vec![0.0; values.len()]
    } else {
        values.iter().map(|value| value / norm).collect()
    }
}

pub(super) fn raw_signature(values: &[f32], task: TargetTask, seed: u64) -> Result<Vec<f32>> {
    let windows = classical_windows(values.len(), task, seed)?;
    raw_signature_windows(values, task, &windows)
}

pub(super) fn classical_windows(
    length: usize,
    task: TargetTask,
    seed: u64,
) -> Result<Vec<Vec<usize>>> {
    let mut windows = Vec::with_capacity(SIGNATURE_VIEWS);
    for view in 0..SIGNATURE_VIEWS {
        let stream_seed = stable_seed(&[
            "fusion-signature-inference".to_owned(),
            seed.to_string(),
            task.id().to_string(),
            length.to_string(),
            view.to_string(),
            WINDOW_SIZE.to_string(),
        ])?;
        let mut rng = NumpyPcg64::new(stream_seed);
        windows.push(rng.choice(length, WINDOW_SIZE, length < WINDOW_SIZE)?);
    }
    Ok(windows)
}

pub(super) fn raw_signature_windows(
    values: &[f32],
    task: TargetTask,
    windows: &[Vec<usize>],
) -> Result<Vec<f32>> {
    let mut rows = Vec::with_capacity(SIGNATURE_VIEWS);
    for indices in windows {
        let sampled = indices
            .iter()
            .map(|&index| values[index])
            .collect::<Vec<_>>();
        rows.push(classical_summary(
            &sampled,
            task,
            values.len(),
            values.len().min(WINDOW_SIZE),
            if values.len() < WINDOW_SIZE {
                1.0 - values.len() as f64 / WINDOW_SIZE as f64
            } else {
                0.0
            },
        ));
    }
    if rows.iter().any(|row| row.len() != FEATURE_DIM) {
        bail!("target classical signature width changed");
    }
    let mut signature = Vec::with_capacity(SIGNATURE_DIM);
    for aggregate in 0..4 {
        for (column, _) in rows[0].iter().enumerate() {
            let a = rows[0][column];
            let b = rows[1][column];
            signature.push(match aggregate {
                0 => (a + b) * 0.5,
                1 => {
                    let mean = (a + b) * 0.5;
                    (((a - mean) * (a - mean) + (b - mean) * (b - mean)) * 0.5).sqrt()
                }
                2 => a.min(b),
                3 => a.max(b),
                _ => unreachable!(),
            });
        }
    }
    Ok(signature)
}

pub(super) fn neural_windows(length: usize, seed: u64) -> Result<Vec<Vec<usize>>> {
    let stream_seed = stable_seed(&[
        "encode".to_owned(),
        seed.to_string(),
        length.to_string(),
        WINDOW_SIZE.to_string(),
        WINDOW_SIZE.to_string(),
    ])?;
    let mut rng = NumpyPcg64::new(stream_seed);
    if length < WINDOW_SIZE {
        return Ok(vec![rng.choice(length, WINDOW_SIZE, true)?]);
    }
    let shuffled = rng.permutation(length);
    Ok(shuffled
        .chunks_exact(WINDOW_SIZE)
        .take(SIGNATURE_VIEWS)
        .map(<[usize]>::to_vec)
        .collect())
}

pub(super) fn stable_seed(parts: &[String]) -> Result<u64> {
    let mut hash = Blake2bVar::new(8).expect("eight-byte BLAKE2b is valid");
    hash.update(parts.join("|").as_bytes());
    let mut digest = [0u8; 8];
    hash.finalize_variable(&mut digest)
        .map_err(|error| anyhow::anyhow!("BLAKE2b finalize failed: {error}"))?;
    Ok(u64::from_be_bytes(digest) & 0x7fff_ffff_ffff_ffff)
}
