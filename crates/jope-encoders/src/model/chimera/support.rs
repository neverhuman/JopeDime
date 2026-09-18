use super::{RegressionExpertFeatures, CONF_DIM, GATE_META_DIM, N_EXPERTS};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

pub(super) fn linear_tensor(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let in_dim = *dims.last().context("linear input has no dims")?;
    let out_dim = w.dims()[0];
    let rows = x.elem_count() / in_dim;
    let flat = x.reshape((rows, in_dim))?;
    let mut y = flat.matmul(&w.t()?)?;
    if let Some(b) = b {
        y = y.broadcast_add(&b.reshape((1, out_dim))?)?;
    }
    let mut out_shape = dims;
    *out_shape.last_mut().unwrap() = out_dim;
    Ok(y.reshape(out_shape)?)
}

pub(super) fn load_safetensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let raw = if path.extension().and_then(|e| e.to_str()) == Some("zst") {
        let compressed =
            std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let bytes = crate::model::tensor_ops::decompress_zstd(&compressed)
            .with_context(|| format!("decompressing {}", path.display()))?;
        candle_core::safetensors::load_buffer(&bytes, device)?
    } else {
        candle_core::safetensors::load(path, device)?
    };
    raw.into_iter()
        .map(|(name, tensor)| Ok((name, tensor.to_dtype(DType::F32)?.to_device(device)?)))
        .collect()
}

pub(super) fn interleave_rows(left: &[f32], right: &[f32], width: usize) -> Vec<f32> {
    let rows = left.len() / width;
    let mut out = Vec::with_capacity(rows * width * 2);
    for row in 0..rows {
        out.extend_from_slice(&left[row * width..(row + 1) * width]);
        out.extend_from_slice(&right[row * width..(row + 1) * width]);
    }
    out
}

pub(super) fn uncertainty_tensor(
    a: &RegressionExpertFeatures,
    b: &RegressionExpertFeatures,
    device: &Device,
) -> Result<Tensor> {
    let mut unc = Vec::with_capacity(a.n_rows * N_EXPERTS);
    for row in 0..a.n_rows {
        unc.push(a.conf[row * CONF_DIM].max(1e-4));
        unc.push(b.conf[row * CONF_DIM].max(1e-4));
    }
    Tensor::from_vec(unc, (a.n_rows, N_EXPERTS), device).map_err(Into::into)
}

pub(super) fn gate_meta_vector(
    ctx_y: &[f32],
    n_query: usize,
    n_features: usize,
) -> [f32; GATE_META_DIM] {
    let n_total = ctx_y.len() + n_query;
    let mut y = Vec::with_capacity(n_total);
    y.extend_from_slice(ctx_y);
    y.resize(n_total, 0.0);
    let skew = regression_skew(&y);
    let spread = percentile(&y, 90.0) - percentile(&y, 10.0);
    [
        (ctx_y.len() as f32).ln_1p(),
        (n_query as f32).ln_1p(),
        (n_features as f32).ln_1p(),
        0.0,
        0.0,
        1.0,
        skew.tanh(),
        (spread * 2.0 - 1.0).tanh(),
    ]
}

pub(super) fn classification_gate_meta_vector(
    ctx_y: &[u8],
    n_query: usize,
    n_features: usize,
) -> [f32; GATE_META_DIM] {
    let zeros = ctx_y.iter().filter(|&&y| y == 0).count() + n_query;
    let ones = ctx_y.iter().filter(|&&y| y != 0).count();
    let total = (ctx_y.len() + n_query).max(1) as f32;
    let n_classes = usize::from(zeros > 0) + usize::from(ones > 0);
    let imbalance = if zeros > 0 && ones > 0 {
        (zeros.min(ones) as f32) / total
    } else {
        1.0
    };
    [
        (ctx_y.len() as f32).ln_1p(),
        (n_query as f32).ln_1p(),
        (n_features as f32).ln_1p(),
        n_classes as f32 / 10.0,
        imbalance,
        0.0,
        0.0,
        (-1.0f32).tanh(),
    ]
}

fn regression_skew(y: &[f32]) -> f32 {
    if y.is_empty() {
        return 0.0;
    }
    let n = y.len() as f32;
    let mean = y.iter().sum::<f32>() / n;
    let var = y
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let std = var.sqrt();
    let third = y
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d * d
        })
        .sum::<f32>()
        / n;
    third / (std * std * std + 1e-9)
}

fn percentile(values: &[f32], pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = (pct / 100.0) * (sorted.len() as f32 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f32;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}
