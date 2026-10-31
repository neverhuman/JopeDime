//! Frozen CPU arithmetic boundaries used before the sensitive SVD scalers.
//!
//! The reference uses eight-lane Welford moments, explicit FMA boundaries,
//! ordered f32 dense reductions and the SLEEF exp primitive. More precision
//! throughout the graph changes its near-constant gates. CUDA retains Candle
//! operations on its selected device and requires separate qualification.

use anyhow::{ensure, Result};
use candle_core_0_11_0::{Device, Tensor, D};

fn flat(x: &Tensor) -> Result<Vec<f32>> {
    Ok(x.flatten_all()?.to_vec1()?)
}

fn tensor(values: Vec<f32>, x: &Tensor) -> Result<Tensor> {
    Ok(Tensor::from_vec(values, x.shape(), &Device::Cpu)?)
}

pub(super) fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let input = x.dim(D::Minus1)?;
    let (output, width) = weight.dims2()?;
    ensure!(
        input > 0 && width == input && bias.dims() == [output],
        "linear shape"
    );
    let mut shape = x.dims().to_vec();
    *shape.last_mut().expect("linear rank") = output;
    if !x.device().is_cpu() {
        return Ok(x
            .reshape((x.elem_count() / input, input))?
            .matmul(&weight.t()?.contiguous()?)?
            .broadcast_add(bias)?
            .reshape(shape)?);
    }
    ensure!(
        weight.device().is_cpu() && bias.device().is_cpu(),
        "CPU weights required"
    );
    let weights = flat(weight)?;
    let bias = flat(bias)?;
    let mut values = Vec::with_capacity(x.elem_count() / input * output);
    for row in flat(x)?.chunks_exact(input) {
        for (i, weights) in weights.chunks_exact(input).enumerate() {
            // The frozen CPU BLAS rounds each product/add and adds bias last.
            let mut sum = 0f32;
            for (&a, &b) in row.iter().zip(weights) {
                sum += a * b;
            }
            values.push(sum + bias[i]);
        }
    }
    Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
}

/// The frozen router widths (116, 140, 256) occupy at most two chunks.
fn moments(row: &[f32]) -> (f32, f32) {
    let vectors = row.len() / 8;
    let (mut means, mut variances) = ([0f32; 8], [0f32; 8]);
    let mut count_vec = 0usize;
    for start in (0..vectors).step_by(16) {
        let take = (vectors - start).min(16);
        let (mut mean, mut variance) = ([0f32; 8], [0f32; 8]);
        for j in 0..take {
            let reciprocal = 1f32 / (j + 1) as f32;
            for k in 0..8 {
                let value = row[(start + j) * 8 + k];
                let delta = value - mean[k];
                mean[k] = delta.mul_add(reciprocal, mean[k]);
                variance[k] = delta.mul_add(value - mean[k], variance[k]);
            }
        }
        let next = count_vec + take;
        let ratio = take as f32 / next as f32;
        for k in 0..8 {
            let delta = mean[k] - means[k];
            means[k] = ratio.mul_add(delta, means[k]);
            variances[k] += (delta * delta * ratio).mul_add(count_vec as f32, variance[k]);
        }
        count_vec = next;
    }
    let (mut count, mut mean, mut variance) = (0usize, 0f32, 0f32);
    for &value in &row[vectors * 8..] {
        let delta = value - mean;
        count += 1;
        mean += delta / count as f32;
        variance = delta.mul_add(value - mean, variance);
    }
    for k in 0..8 {
        let next = count + vectors;
        let ratio = vectors as f32 / next as f32;
        let delta = means[k] - mean;
        mean = ratio.mul_add(delta, mean);
        variance += (delta * delta * ratio).mul_add(count as f32, variances[k]);
        count = next;
    }
    (mean, variance / row.len() as f32)
}

pub(super) fn norm(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() {
        let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        return Ok(centered
            .broadcast_mul(
                &centered
                    .sqr()?
                    .mean_keepdim(D::Minus1)?
                    .affine(1., 1e-5)?
                    .sqrt()?
                    .recip()?,
            )?
            .broadcast_mul(weight)?
            .broadcast_add(bias)?);
    }
    ensure!(
        weight.device().is_cpu() && bias.device().is_cpu(),
        "CPU weights required"
    );
    let width = x.dim(D::Minus1)?;
    ensure!(
        width > 0 && width <= 256 && weight.dims() == [width] && bias.dims() == [width],
        "frozen norm shape"
    );
    let (weight, bias) = (flat(weight)?, flat(bias)?);
    let mut values = Vec::with_capacity(x.elem_count());
    for row in flat(x)?.chunks_exact(width) {
        let (mean, variance) = moments(row);
        let reciprocal = 1f32 / (variance + 1e-5).sqrt();
        values.extend(
            row.iter()
                .enumerate()
                .map(|(i, v)| ((v - mean) * reciprocal).mul_add(weight[i], bias[i])),
        );
    }
    tensor(values, x)
}

// Scalar expf_u10 adapted from SLEEF 3.6.1, sleefsimdsp.c / misc.h.
// Copyright Naoki Shibata and contributors 2010-2024.
// Distributed under the Boost Software License 1.0; see BSL-1.0.txt.
// Explicit mul_add preserves its reference rounding; no fast-math flags.
fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x < -104. {
        return 0.;
    }
    if x > 100. {
        return f32::INFINITY;
    }
    let q = (x * std::f32::consts::LOG2_E).round_ties_even() as i32;
    let mut s = (q as f32).mul_add(-f32::from_bits(0x3f317200), x);
    s = (q as f32).mul_add(-f32::from_bits(0x35bfbe8e), s);
    let mut u = f32::from_bits(0x39502bda);
    u = u.mul_add(s, f32::from_bits(0x3ab696c9));
    u = u.mul_add(s, f32::from_bits(0x3c0888a6));
    u = u.mul_add(s, f32::from_bits(0x3d2aaa7a));
    u = u.mul_add(s, f32::from_bits(0x3e2aaaab));
    u = u.mul_add(s, 0.5);
    u = 1. + (s * s).mul_add(u, s);
    let first = q >> 1;
    u * f32::from_bits(((first + 127) as u32) << 23)
        * f32::from_bits(((q - first + 127) as u32) << 23)
}

pub(super) fn silu(x: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() {
        return Ok(x.silu()?);
    }
    tensor(
        flat(x)?.into_iter().map(|v| v / (1. + exp(-v))).collect(),
        x,
    )
}

pub(super) fn tanh(x: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() {
        return Ok(x.tanh()?);
    }
    // Round once at the f32 model boundary, not inside the transcendental.
    tensor(
        flat(x)?
            .into_iter()
            .map(|v| f64::from(v).tanh() as f32)
            .collect(),
        x,
    )
}

pub(super) fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if !a.device().is_cpu() {
        // Q/K/V column slices retain the packed QKV row stride. CUDA matmul
        // requires compact operands; materialize them on the selected device.
        return Ok(a.contiguous()?.matmul(&b.contiguous()?)?);
    }
    ensure!(b.device().is_cpu(), "CPU right operand required");
    let (m, k) = a.dims2()?;
    let (kb, n) = b.dims2()?;
    ensure!(k == kb, "matrix shape");
    let (a, b) = (flat(a)?, flat(b)?);
    let mut values = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0f32;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            values[i * n + j] = sum;
        }
    }
    Ok(Tensor::from_vec(values, (m, n), &Device::Cpu)?)
}

pub(super) fn scale_attention(x: &Tensor) -> Result<Tensor> {
    // Scalar division first converts sqrt(32) to the reference f32 divisor.
    let divisor = Tensor::from_vec(vec![32f32.sqrt()], (1, 1), x.device())?;
    Ok(x.broadcast_div(&divisor)?)
}

pub(super) fn softmax(x: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() {
        let exp = x.broadcast_sub(&x.max_keepdim(1)?)?.exp()?;
        return Ok(exp.broadcast_mul(&exp.sum_keepdim(1)?.recip()?)?);
    }
    let (rows, width) = x.dims2()?;
    ensure!(width == 8, "frozen attention width");
    let mut values = Vec::with_capacity(x.elem_count());
    for row in flat(x)?.chunks_exact(8) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|v| exp(*v - max)).collect();
        let sum = ((e[0] + e[4]) + (e[2] + e[6])) + ((e[1] + e[5]) + (e[3] + e[7]));
        let reciprocal = 1f32 / sum;
        values.extend(e.into_iter().map(|v| v * reciprocal));
    }
    Ok(Tensor::from_vec(values, (rows, width), &Device::Cpu)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bits(x: &Tensor) -> Vec<u32> {
        flat(x).unwrap().into_iter().map(f32::to_bits).collect()
    }
    #[test]
    fn frozen_norm_preserves_chunk_tail_and_affine_rounding() {
        for (width, expected) in [
            (116usize, [0xbf02f40a, 0xbec45a8d, 0xbdaecebd]),
            (140usize, [0xbf0433ce, 0xbe4a8dec, 0x3fa4deac]),
            (256usize, [0xbf03dc7d, 0x3f0bcd84, 0xbf8713d6]),
        ] {
            let x = Tensor::from_vec(
                (0..width)
                    .map(|i| 10000f32 + ((i % 11) as f32 - 5.) * 0.125)
                    .collect::<Vec<_>>(),
                (1, width),
                &Device::Cpu,
            )
            .unwrap();
            let weight = Tensor::from_vec(
                (0..width)
                    .map(|i| 1f32 + ((i % 7) as f32 - 3.) * 0.25)
                    .collect::<Vec<_>>(),
                width,
                &Device::Cpu,
            )
            .unwrap();
            let bias = Tensor::from_vec(
                (0..width)
                    .map(|i| ((i % 5) as f32 - 2.) * 0.0625)
                    .collect::<Vec<_>>(),
                width,
                &Device::Cpu,
            )
            .unwrap();
            let actual = bits(&norm(&x, &weight, &bias).unwrap());
            assert_eq!(
                [actual[0], actual[width / 2], actual[width - 1]],
                expected,
                "width {width}"
            );
        }
    }
    #[test]
    fn frozen_exp_preserves_range_reduction_and_extreme_values() {
        let inputs = [
            -104f32,
            -100.,
            -88.7,
            -20.,
            -1.,
            -0.,
            0.,
            0.125,
            0.5,
            1.,
            10.,
            88.,
            90.,
            100.,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        assert_eq!(
            inputs.map(|x| exp(x).to_bits()),
            [
                0x00000000, 0x0000001b, 0x0020bd47, 0x310da433, 0x3ebc5ab2, 0x3f800000, 0x3f800000,
                0x3f910b02, 0x3fd3094c, 0x402df854, 0x46ac14ee, 0x7ef882b7, 0x7f800000, 0x7f800000,
                0x00000000, 0x7f800000
            ]
        );
        assert!(exp(f32::NAN).is_nan());
        assert_eq!(exp(-105.).to_bits(), 0);
    }
    #[test]
    fn dense_reduction_rounds_f32_products_before_adding_bias() {
        let x =
            Tensor::from_vec(vec![16_777_216f32, 1., -16_777_216.], (1, 3), &Device::Cpu).unwrap();
        let weight =
            Tensor::from_vec(vec![1f32, 1., 1., 1., -1., 1.], (2, 3), &Device::Cpu).unwrap();
        let bias = Tensor::from_vec(vec![1f32, 1.], 2, &Device::Cpu).unwrap();
        assert_eq!(
            flat(&linear(&x, &weight, &bias).unwrap()).unwrap(),
            [1., 0.]
        );
        let bad = Tensor::zeros(4, candle_core_0_11_0::DType::F32, &Device::Cpu).unwrap();
        assert!(linear(&x, &weight, &bad).is_err());
    }
    #[test]
    fn attention_scaling_keeps_division_scalar_rounding() {
        let x =
            Tensor::from_vec(vec![0.1f32, 0.2, 0.7, 1.3, -0.7, 2.7], (1, 6), &Device::Cpu).unwrap();
        assert_eq!(
            bits(&scale_attention(&x).unwrap()),
            [0x3c90d0c3, 0x3d10d0c3, 0x3dfd6d54, 0x3e6b533c, 0xbdfd6d54, 0x3ef46049]
        );
        assert_ne!(
            bits(&scale_attention(&x).unwrap()),
            bits(&x.affine(1. / 32f64.sqrt(), 0.).unwrap())
        );
    }
    #[test]
    fn attention_softmax_preserves_exp_and_eight_lane_sum() {
        let x = Tensor::from_vec(
            vec![
                -7f32, -3., 0., 1., 2., 3., 5., 7., 2., -1., 0.3, 0.7, 1., -2., 3., 0.,
            ],
            (2, 8),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            bits(&softmax(&x).unwrap()),
            [
                0x353fca9b, 0x38239dd4, 0x3a4d6535, 0x3b0b94a3, 0x3bbdb5b0, 0x3c80ebcf, 0x3dee26be,
                0x3f5bf6d6, 0x3e57d087, 0x3c2bea98, 0x3d1db3fb, 0x3d6b43ec, 0x3d9ec998, 0x3b7cfa5f,
                0x3f12a944, 0x3ce9a899
            ]
        );
        let bad = Tensor::zeros((1, 7), candle_core_0_11_0::DType::F32, &Device::Cpu).unwrap();
        assert!(softmax(&bad).is_err());
    }
}
