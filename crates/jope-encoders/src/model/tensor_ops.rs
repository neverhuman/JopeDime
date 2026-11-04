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
        assert_eq!(mean, vec![1_000_000.0 + 0.125]);
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
        let query = [1_000_000.0f32 + 0.125, 2.0];
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

mod feature_views;
use feature_views::missing_indicator;
pub(super) use feature_views::{apply_feature_view, FeatureView};
