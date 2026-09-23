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
    group_from_moments(ctx_x, qx, n, m, p, &mean, &std)
}

/// Stable CPU moments for the frozen foundation input path. Torch's scaler
/// computes population moments; its parallel CUDA reductions are not reproduced
/// bit-for-bit here. Accumulate in f64, then retain the original f32 transform.
/// Ordinary prediction paths continue to use `preprocess_group` above.
pub(super) fn preprocess_group_stable(
    ctx_x: &[f32],
    qx: &[f32],
    n: usize,
    m: usize,
    p: usize,
) -> Vec<f32> {
    let (mean, std) = stable_feature_moments(ctx_x, n, p);
    group_from_moments(ctx_x, qx, n, m, p, &mean, &std)
}

fn stable_feature_moments(ctx_x: &[f32], n: usize, p: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f64; p];
    let mut count = vec![0usize; p];
    for row in ctx_x.chunks_exact(p) {
        for (j, &value) in row.iter().enumerate() {
            if value.is_finite() {
                mean[j] += f64::from(value);
                count[j] += 1;
            }
        }
    }
    for j in 0..p {
        if count[j] > 0 {
            mean[j] /= count[j] as f64;
        }
    }
    let mut variance = vec![0.0f64; p];
    for row in ctx_x.chunks_exact(p) {
        for (j, &value) in row.iter().enumerate() {
            // Imputed context rows have zero centered deviation. Keep n as
            // the population denominator, including those imputed rows.
            if value.is_finite() {
                let centered = f64::from(value) - mean[j];
                variance[j] += centered * centered;
            }
        }
    }
    let std = variance
        .into_iter()
        .map(|sum| {
            let value = (sum / n.max(1) as f64).sqrt() as f32;
            if value > 1e-12 {
                value
            } else {
                1.0
            }
        })
        .collect();
    (mean.into_iter().map(|value| value as f32).collect(), std)
}

fn group_from_moments(
    ctx_x: &[f32],
    qx: &[f32],
    n: usize,
    m: usize,
    p: usize,
    mean: &[f32],
    std: &[f32],
) -> Vec<f32> {
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

#[cfg(test)]
mod stable_feature_tests {
    use super::*;

    #[test]
    fn foundation_moments_preserve_small_variance_at_large_offset() {
        // Exactly representable f32 values, with analytical mean and population
        // variance. Sequential f32 summation loses the small deviations here.
        let context = (0..768)
            .map(|i| 1_000_000.0f32 + (i % 3) as f32 * 0.125)
            .collect::<Vec<_>>();
        let (mean, std) = stable_feature_moments(&context, context.len(), 1);
        let expected_std = (2.0f64 * 0.125f64.powi(2) / 3.0).sqrt() as f32;
        assert_eq!(mean, vec![1_000_000.125]);
        assert_eq!(std, vec![expected_std]);
        let stable = preprocess_group_stable(&context, &[], context.len(), 0, 1);
        let original = preprocess_group(&context, &[], context.len(), 0, 1);
        let expected_first = -0.125f32 / expected_std;
        assert_eq!(stable[0], expected_first);
        assert!((original[0] - expected_first).abs() > 0.1);
    }

    #[test]
    fn foundation_moments_and_query_output_are_row_permutation_stable() {
        let context = (0..600)
            .flat_map(|i| [1_000_000.0f32 + (i % 3) as f32 * 0.125, (i % 7) as f32])
            .collect::<Vec<_>>();
        let reversed = context
            .chunks_exact(2)
            .rev()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            stable_feature_moments(&context, 600, 2),
            stable_feature_moments(&reversed, 600, 2)
        );
        let query = [1_000_000.125f32, 2.0];
        let first = preprocess_group_stable(&context, &query, 600, 1, 2);
        let second = preprocess_group_stable(&reversed, &query, 600, 1, 2);
        assert_eq!(&first[600 * 12..], &second[600 * 12..]);
        for row in 0..600 {
            assert_eq!(
                &first[row * 12..(row + 1) * 12],
                &second[(599 - row) * 12..(600 - row) * 12]
            );
        }
    }

    #[test]
    fn foundation_moments_preserve_imputation_population_and_indicators() {
        let context = [1.0f32, f32::NAN, 3.0, f32::INFINITY];
        let (mean, std) = stable_feature_moments(&context, 4, 1);
        assert_eq!(mean, vec![2.0]);
        // Two observed centered squares, divided by all four context rows.
        assert_eq!(std, vec![0.5f32.sqrt()]);
        let query = [f32::NEG_INFINITY, 1000.0];
        let grouped = preprocess_group_stable(&context, &query, 4, 2, 1);
        assert_eq!(&grouped[6..12], &[0.0, 0.0, 0.0, -2.0, -2.0, -2.0]);
        assert_eq!(&grouped[18..24], &[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]);
        assert_eq!(&grouped[24..30], &[0.0, 0.0, 0.0, 4.0, 4.0, 4.0]);
        assert_eq!(grouped[30], (1000.0f32 - mean[0]) / std[0]);
        assert_eq!(
            &grouped[..24],
            preprocess_group_stable(&context, &[], 4, 0, 1)
        );
    }

    #[test]
    fn foundation_moments_keep_constant_missing_and_scale_floor_semantics() {
        let constant = vec![0.1f32; 768];
        assert_eq!(
            stable_feature_moments(&constant, 768, 1),
            (vec![0.1], vec![1.0])
        );
        assert!(preprocess_group_stable(&constant, &[], 768, 0, 1)
            .iter()
            .all(|&value| value == 0.0));
        let missing = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(
            stable_feature_moments(&missing, 3, 1),
            (vec![0.0], vec![1.0])
        );
        assert_eq!(stable_feature_moments(&[-1e-13, 1e-13], 2, 1).1, vec![1.0]);
        assert_eq!(
            stable_feature_moments(&[-1e-11, 1e-11], 2, 1).1,
            vec![1e-11]
        );
    }
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
