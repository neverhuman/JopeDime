use super::FoundationDatasetInputV1;
use anyhow::{bail, Context, Result};

pub(super) fn rectangular_width(rows: &[Vec<f64>], label: &str) -> Result<usize> {
    let width = rows.first().map(Vec::len).context("input rows are empty")?;
    if width < 2 || rows.iter().any(|row| row.len() != width) {
        bail!("{label} must be rectangular with at least one feature and one label");
    }
    Ok(width)
}

pub(super) fn python_9g_roundtrip(value: f64) -> Result<f64> {
    if !value.is_finite() {
        bail!("cannot encode a non-finite scaled value");
    }
    format!("{value:.8e}")
        .parse::<f64>()
        .context("parse nine-significant-digit encoding value")
}

pub(super) fn validate_input(input: &FoundationDatasetInputV1) -> Result<()> {
    if input.context_x.is_empty()
        || input.context_x.len() != input.context_y.len()
        || input.query_x.is_empty()
    {
        bail!("foundation input requires context/query rows and matching targets");
    }
    let width = input.context_x[0].len();
    if width == 0
        || input
            .context_x
            .iter()
            .chain(&input.query_x)
            .any(|row| row.len() != width)
    {
        bail!("foundation input rows must be rectangular and non-empty");
    }
    if input.context_y.iter().any(|value| !value.is_finite()) {
        bail!("foundation targets must be finite");
    }
    Ok(())
}

pub(super) fn take_indices(len: usize, cap: usize, seed: u64) -> Vec<usize> {
    if cap == 0 || len <= cap {
        return (0..len).collect();
    }
    let mut indexes: Vec<usize> = (0..len).collect();
    let mut state = seed;
    for index in (1..indexes.len()).rev() {
        state = splitmix64(state);
        indexes.swap(index, (state as usize) % (index + 1));
    }
    indexes.truncate(cap);
    indexes.sort_unstable();
    indexes
}

pub(super) fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

pub(super) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub(super) fn pool_mean_std_p10_p90(rows: &[Vec<f32>]) -> Vec<f32> {
    let d = rows.first().map_or(0, Vec::len);
    let n = rows.len().max(1) as f64;
    let mut mean = vec![0.0; d];
    for row in rows {
        for (j, value) in row.iter().enumerate() {
            mean[j] += *value as f64;
        }
    }
    for value in &mut mean {
        *value /= n;
    }
    let mut std = vec![0.0; d];
    for row in rows {
        for (j, value) in row.iter().enumerate() {
            std[j] += (*value as f64 - mean[j]).powi(2);
        }
    }
    for value in &mut std {
        *value = (*value / n).sqrt();
    }
    let mut p10 = vec![0.0; d];
    let mut p90 = vec![0.0; d];
    for j in 0..d {
        let mut column: Vec<f64> = rows.iter().map(|row| row[j] as f64).collect();
        column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        p10[j] = percentile(&column, 0.1);
        p90[j] = percentile(&column, 0.9);
    }
    mean.into_iter()
        .chain(std)
        .chain(p10)
        .chain(p90)
        .map(|value| value as f32)
        .collect()
}

pub(super) fn percentile(values: &[f64], q: f64) -> f64 {
    if values.len() <= 1 {
        return values.first().copied().unwrap_or(0.0);
    }
    // The retained Torch 2.7 graph calls quantile on FP32 hidden rows. Its
    // quantile tensor, rank multiplication and interpolation weights are FP32;
    // a double-precision rank changes pooled tail values on sparse distributions.
    let position = q as f32 * (values.len() - 1) as f32;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let start = values[low] as f32;
    let end = values[high] as f32;
    let weight = position - low as f32;
    // Match Torch's stable lerp branch, including the fused multiply-add used
    // by the original CUDA FP32 graph.
    if weight < 0.5 {
        weight.mul_add(end - start, start) as f64
    } else {
        (-(1.0 - weight)).mul_add(end - start, end) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foundation_percentiles_match_retained_torch27_f32_positions() {
        // Independently generated with Torch 2.7.0 CPU, the retained encoder's
        // version. Newer Torch versions use different quantile arithmetic.
        for (n, zeros, block, expected) in [
            (158, 142, 3, 1_077_936_256u32),
            (158, 16, 2, 1_088_421_884),
            (200, 180, 3, 1_065_351_680),
            (200, 20, 2, 1_091_567_612),
            (1, 0, 3, 1_092_616_192),
        ] {
            let rows = (0..n)
                .map(|i| vec![if i < zeros { 0.0 } else { 10.0 }])
                .collect::<Vec<_>>();
            assert_eq!(pool_mean_std_p10_p90(&rows)[block].to_bits(), expected);
        }
    }
}
