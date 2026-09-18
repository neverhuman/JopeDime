use super::EMBED;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};

/// Decompress a zstd frame fully into memory using the pure-Rust `ruzstd`
/// decoder (no FFI, no C runtime, no extra shared library in the image). Used
/// to transparently load `*.safetensors.zst` weights that are shipped
/// compressed in the container image and decompressed on launch.
pub(super) fn decompress_zstd(compressed: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(compressed)
        .map_err(|e| anyhow::anyhow!("initialising zstd decoder: {e:?}"))?;
    let mut raw = Vec::new();
    decoder
        .read_to_end(&mut raw)
        .context("reading zstd-decompressed weights")?;
    Ok(raw)
}

pub(super) fn add_train_y_column(x: &Tensor, y: &Tensor, n_train: usize) -> Result<Tensor> {
    let dims = x.dims();
    let c = dims[2];
    let y = y.unsqueeze(2)?.broadcast_as(vec![1, n_train, c, EMBED])?;
    let train = add_promote(&x.narrow(1, 0, n_train)?, &y)?;
    if dims[1] == n_train {
        Ok(train)
    } else {
        let rest = x.narrow(1, n_train, dims[1] - n_train)?;
        Ok(cat_promote(&[&train, &rest], 1)?)
    }
}

pub(super) fn add_train_y_icl(x: &Tensor, y: &Tensor, n_train: usize) -> Result<Tensor> {
    let dims = x.dims();
    let train = add_promote(&x.narrow(1, 0, n_train)?, y)?;
    if dims[1] == n_train {
        Ok(train)
    } else {
        let rest = x.narrow(1, n_train, dims[1] - n_train)?;
        Ok(cat_promote(&[&train, &rest], 1)?)
    }
}

pub(super) fn add_promote(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.dtype() == rhs.dtype() {
        Ok(lhs.broadcast_add(rhs)?)
    } else {
        Ok(lhs
            .to_dtype(DType::F32)?
            .broadcast_add(&rhs.to_dtype(DType::F32)?)?)
    }
}

pub(super) fn cat_promote(xs: &[&Tensor], dim: usize) -> Result<Tensor> {
    let Some(first) = xs.first() else {
        bail!("cat_promote called with no tensors");
    };
    if xs.iter().all(|x| x.dtype() == first.dtype()) {
        Ok(Tensor::cat(xs, dim)?)
    } else {
        let promoted = xs
            .iter()
            .map(|x| x.to_dtype(DType::F32))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let refs: Vec<&Tensor> = promoted.iter().collect();
        Ok(Tensor::cat(&refs, dim)?)
    }
}

pub(super) fn repeat_dim(x: &Tensor, dim: usize, repeat: usize) -> Result<Tensor> {
    let parts: Vec<&Tensor> = (0..repeat).map(|_| x).collect();
    Ok(Tensor::cat(&parts, dim)?)
}

pub(super) fn preprocess_group(
    ctx_x: &[f32],
    qx: &[f32],
    n: usize,
    m: usize,
    p: usize,
) -> Vec<f32> {
    let mut mean = vec![0f32; p];
    let mut count = vec![0usize; p];
    for row in ctx_x.chunks_exact(p) {
        for (j, v) in row.iter().enumerate() {
            if v.is_finite() {
                mean[j] += *v;
                count[j] += 1;
            }
        }
    }
    for j in 0..p {
        mean[j] = if count[j] > 0 {
            mean[j] / count[j] as f32
        } else {
            0.0
        };
    }
    let mut var = vec![0f32; p];
    for row in ctx_x.chunks_exact(p) {
        for (j, v) in row.iter().enumerate() {
            let d = if v.is_finite() { *v } else { mean[j] } - mean[j];
            var[j] += d * d;
        }
    }
    let mut std = vec![1f32; p];
    for j in 0..p {
        let s = (var[j] / n.max(1) as f32).sqrt();
        std[j] = if s > 1e-12 { s } else { 1.0 };
    }
    let rows = n + m;
    let mut normalized = vec![0f32; rows * p];
    for r in 0..rows {
        let src = if r < n {
            &ctx_x[r * p..(r + 1) * p]
        } else {
            let qr = r - n;
            &qx[qr * p..(qr + 1) * p]
        };
        for j in 0..p {
            let v = if src[j].is_finite() { src[j] } else { mean[j] };
            normalized[r * p + j] = (v - mean[j]) / std[j];
        }
    }
    let shifts = [1usize, 2, 4];
    let mut grouped = Vec::with_capacity(rows * p * 6);
    for r in 0..rows {
        for c in 0..p {
            for shift in shifts {
                grouped.push(normalized[r * p + ((c + shift) % p)]);
            }
            let src = if r < n {
                &ctx_x[r * p..(r + 1) * p]
            } else {
                let qr = r - n;
                &qx[qr * p..(qr + 1) * p]
            };
            for shift in shifts {
                grouped.push(missing_indicator(src[(c + shift) % p]));
            }
        }
    }
    grouped
}

#[derive(Clone, Copy, Debug)]
pub(super) enum FeatureView {
    Identity,
    SignedLog,
    QuantileNormal,
    QuantileUniform,
    YeoJohnson,
    Robust,
}

pub(super) fn apply_feature_view(
    view: FeatureView,
    ctx_x: &[f32],
    qx: &[f32],
    n: usize,
    m: usize,
    p: usize,
) -> (Vec<f32>, Vec<f32>) {
    match view {
        FeatureView::Identity => (ctx_x.to_vec(), qx.to_vec()),
        FeatureView::SignedLog => (
            ctx_x.iter().map(|&v| signed_log(v)).collect(),
            qx.iter().map(|&v| signed_log(v)).collect(),
        ),
        FeatureView::QuantileNormal => quantile_transform(ctx_x, qx, n, m, p, true),
        FeatureView::QuantileUniform => quantile_transform(ctx_x, qx, n, m, p, false),
        FeatureView::YeoJohnson => yeo_johnson_transform(ctx_x, qx, n, m, p),
        FeatureView::Robust => robust_transform(ctx_x, qx, n, m, p),
    }
}

fn missing_indicator(v: f32) -> f32 {
    if v.is_nan() {
        -2.0
    } else if v == f32::INFINITY {
        2.0
    } else if v == f32::NEG_INFINITY {
        4.0
    } else {
        0.0
    }
}

fn signed_log(v: f32) -> f32 {
    if v.is_finite() {
        v.signum() * v.abs().ln_1p()
    } else {
        v
    }
}

fn robust_transform(
    ctx_x: &[f32],
    qx: &[f32],
    n: usize,
    m: usize,
    p: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut med = vec![0.0; p];
    let mut scale = vec![1.0; p];
    for j in 0..p {
        let mut col = finite_column(ctx_x, n, p, j);
        if col.is_empty() {
            continue;
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        med[j] = percentile_sorted(&col, 0.5);
        let q25 = percentile_sorted(&col, 0.25);
        let q75 = percentile_sorted(&col, 0.75);
        let iqr = q75 - q25;
        scale[j] = if iqr.abs() > 1e-12 { iqr } else { 1.0 };
    }
    (
        transform_by_col(ctx_x, n, p, |j, v| {
            finite_or(v, med[j], |x| (x - med[j]) / scale[j])
        }),
        transform_by_col(qx, m, p, |j, v| {
            finite_or(v, med[j], |x| (x - med[j]) / scale[j])
        }),
    )
}

fn quantile_transform(
    ctx_x: &[f32],
    qx: &[f32],
    n: usize,
    m: usize,
    p: usize,
    normal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut sorted_cols = Vec::with_capacity(p);
    for j in 0..p {
        let mut col = finite_column(ctx_x, n, p, j);
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_cols.push(col);
    }
    let apply = |j: usize, v: f32| {
        if !v.is_finite() {
            return v;
        }
        let col = &sorted_cols[j];
        if col.len() <= 1 {
            return 0.0;
        }
        if normal {
            inverse_standard_normal(empirical_cdf(col, v).clamp(1e-7, 1.0 - 1e-7))
        } else {
            empirical_cdf(col, v).clamp(0.0, 1.0)
        }
    };
    (
        transform_by_col(ctx_x, n, p, apply),
        transform_by_col(qx, m, p, apply),
    )
}

fn yeo_johnson_transform(
    ctx_x: &[f32],
    qx: &[f32],
    n: usize,
    m: usize,
    p: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut lambda = vec![1.0; p];
    let mut mean = vec![0.0; p];
    let mut std = vec![1.0; p];
    for j in 0..p {
        let col = finite_column(ctx_x, n, p, j);
        if col.len() < 2 {
            continue;
        }
        lambda[j] = fit_yeo_johnson_lambda(&col);
        let transformed: Vec<f32> = col
            .iter()
            .map(|&v| yeo_johnson_value(v, lambda[j]))
            .collect();
        mean[j] = transformed.iter().sum::<f32>() / transformed.len() as f32;
        let var = transformed
            .iter()
            .map(|v| {
                let d = *v - mean[j];
                d * d
            })
            .sum::<f32>()
            / transformed.len() as f32;
        let s = var.sqrt();
        std[j] = if s > 1e-12 { s } else { 1.0 };
    }
    let apply = |j: usize, v: f32| {
        finite_or(v, mean[j], |x| {
            (yeo_johnson_value(x, lambda[j]) - mean[j]) / std[j]
        })
    };
    (
        transform_by_col(ctx_x, n, p, apply),
        transform_by_col(qx, m, p, apply),
    )
}

fn transform_by_col<F>(x: &[f32], n: usize, p: usize, mut f: F) -> Vec<f32>
where
    F: FnMut(usize, f32) -> f32,
{
    let mut out = Vec::with_capacity(n * p);
    for row in x.chunks_exact(p).take(n) {
        for (j, &v) in row.iter().enumerate() {
            out.push(f(j, v));
        }
    }
    out
}

fn finite_or<F>(v: f32, replacement: f32, f: F) -> f32
where
    F: FnOnce(f32) -> f32,
{
    if v.is_finite() {
        f(v)
    } else {
        replacement
    }
}

fn finite_column(x: &[f32], n: usize, p: usize, j: usize) -> Vec<f32> {
    x.chunks_exact(p)
        .take(n)
        .filter_map(|row| row[j].is_finite().then_some(row[j]))
        .collect()
}

fn percentile_sorted(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f32;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

fn empirical_cdf(sorted: &[f32], v: f32) -> f32 {
    let n = sorted.len();
    if n <= 1 {
        return 0.5;
    }
    if v <= sorted[0] {
        return 0.0;
    }
    if v >= sorted[n - 1] {
        return 1.0;
    }
    let upper = sorted.partition_point(|x| *x <= v);
    let lower = upper.saturating_sub(1);
    let a = sorted[lower];
    let b = sorted[upper];
    let frac = if (b - a).abs() > 1e-12 {
        (v - a) / (b - a)
    } else {
        0.0
    };
    (lower as f32 + frac) / (n - 1) as f32
}

fn inverse_standard_normal(p: f32) -> f32 {
    // Peter J. Acklam's rational approximation.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p = p.clamp(1e-12, 1.0 - 1e-12) as f64;
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    let out = if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    out as f32
}

fn fit_yeo_johnson_lambda(x: &[f32]) -> f32 {
    let mut best = 1.0;
    let mut best_ll = f32::NEG_INFINITY;
    for i in 0..=80 {
        let lambda = -2.0 + i as f32 * 0.05;
        let ll = yeo_johnson_log_likelihood(x, lambda);
        if ll > best_ll {
            best_ll = ll;
            best = lambda;
        }
    }
    let mut lo = (best - 0.05).max(-2.0);
    let mut hi = (best + 0.05).min(2.0);
    for _ in 0..24 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if yeo_johnson_log_likelihood(x, m1) < yeo_johnson_log_likelihood(x, m2) {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    (lo + hi) * 0.5
}

fn yeo_johnson_log_likelihood(x: &[f32], lambda: f32) -> f32 {
    let transformed: Vec<f32> = x.iter().map(|&v| yeo_johnson_value(v, lambda)).collect();
    let mean = transformed.iter().sum::<f32>() / transformed.len() as f32;
    let var = transformed
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / transformed.len() as f32;
    if var <= 1e-24 {
        return f32::NEG_INFINITY;
    }
    let log_jac = x
        .iter()
        .map(|&v| if v >= 0.0 { v.ln_1p() } else { -(-v).ln_1p() })
        .sum::<f32>();
    -0.5 * x.len() as f32 * var.ln() + (lambda - 1.0) * log_jac
}

fn yeo_johnson_value(x: f32, lambda: f32) -> f32 {
    if x >= 0.0 {
        if lambda.abs() < 1e-7 {
            x.ln_1p()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else {
        let two_minus = 2.0 - lambda;
        if two_minus.abs() < 1e-7 {
            -(-x).ln_1p()
        } else {
            -(((1.0 - x).powf(two_minus) - 1.0) / two_minus)
        }
    }
}
