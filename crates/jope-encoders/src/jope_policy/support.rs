use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Debug, Deserialize)]
pub struct JopePolicyConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub dataset_dim: usize,
    pub target_dim: usize,
    pub reward_ensemble_k: usize,
    pub embedding_rows: usize,
    pub routing_classes: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub padding_idx: usize,
    pub layernorm_eps: f64,
    pub gelu: String,
    pub norm_first: bool,
    pub attn_scale: f64,
}

#[derive(Debug, Deserialize)]
pub(super) struct PolicyMetadata {
    pub(super) model_version: String,
    pub(super) config: JopePolicyConfig,
}

pub(super) struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    pub(super) fn load_buffer(bytes: &[u8], device: &Device) -> Result<Self> {
        let map = candle_core::safetensors::load_buffer(bytes, device)
            .context("load verified JOPE policy safetensors")?
            .into_iter()
            .map(|(name, tensor)| Ok((name, tensor.to_dtype(DType::F32)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { map })
    }

    pub(super) fn get(&self, name: &str) -> Result<&Tensor> {
        self.map
            .get(name)
            .with_context(|| format!("JOPE policy tensor missing: {name}"))
    }
}

fn projection(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    if !x.device().is_cpu() {
        bail!("JOPE scalar projection supports CPU only; CUDA supports context encoding");
    }
    let input_width = *x.dims().last().context("JOPE linear input needs a rank")?;
    let (output_width, weight_width) = weight.dims2()?;
    if weight_width != input_width
        || bias
            .map(Tensor::dims1)
            .transpose()?
            .is_some_and(|width| width != output_width)
    {
        bail!("JOPE linear dimensions are inconsistent");
    }
    let input = x.flatten_all()?.to_vec1::<f32>()?;
    let weights = weight.flatten_all()?.to_vec1::<f32>()?;
    let rows = input.len() / input_width;
    let biases = bias.map(Tensor::to_vec1::<f32>).transpose()?;
    let mut output = vec![0.0f32; rows * output_width];
    // The frozen CPU oracle's reference SGEMM accumulates each dot product
    // from zero, then applies beta*C (the bias). Seeding the accumulator with
    // bias changes fp32 rounding and was the Gen31 fixed-prefix parity miss.
    for row in 0..rows {
        for column in 0..output_width {
            let output_index = row * output_width + column;
            let mut value = 0.0f32;
            for inner in 0..input_width {
                value += input[row * input_width + inner] * weights[column * input_width + inner];
            }
            output[output_index] = if let Some(biases) = &biases {
                value + biases[column]
            } else {
                value
            };
        }
    }
    let mut dimensions = x.dims().to_vec();
    *dimensions.last_mut().expect("JOPE linear rank checked") = output_width;
    Ok(Tensor::from_vec(output, dimensions, x.device())?)
}

pub(super) fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    projection(x, weight, Some(bias))
}

pub(super) fn layer_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    epsilon: f64,
) -> Result<Tensor> {
    if !x.device().is_cpu() {
        bail!("JOPE scalar normalization supports CPU only; CUDA supports context encoding");
    }
    let dimensions = x.dims();
    let width = *dimensions.last().context("JOPE layer norm needs a rank")?;
    let input = x.flatten_all()?.to_vec1::<f32>()?;
    let weights = weight.to_vec1::<f32>()?;
    let biases = bias.to_vec1::<f32>()?;
    if weights.len() != width || biases.len() != width || input.len() % width != 0 {
        bail!("JOPE layer norm dimensions are inconsistent");
    }
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width) {
        let mean_wide = row.iter().map(|&value| f64::from(value)).sum::<f64>() / width as f64;
        let variance = row
            .iter()
            .map(|&value| (f64::from(value) - mean_wide).powi(2))
            .sum::<f64>()
            / width as f64;
        let mean = mean_wide as f32;
        let inverse = (1.0 / (variance + epsilon).sqrt()) as f32;
        output.extend(
            row.iter()
                .zip(&weights)
                .zip(&biases)
                .map(|((&value, &scale), &offset)| {
                    ((value - mean) * inverse).mul_add(scale, offset)
                }),
        );
    }
    Ok(Tensor::from_vec(output, dimensions.to_vec(), x.device())?)
}

pub(super) fn causal_mask(sequence: usize, device: &Device) -> Result<Tensor> {
    let mut values = vec![0.0f32; sequence * sequence];
    for row in 0..sequence {
        for column in (row + 1)..sequence {
            values[row * sequence + column] = -1e9;
        }
    }
    Ok(Tensor::from_vec(values, (sequence, sequence), device)?)
}

pub(super) fn validate_config(config: &JopePolicyConfig) -> Result<()> {
    if config.d_model != 512
        || config.n_layers != 6
        || config.n_heads != 4
        || config.ffn_dim != 2048
        || config.dataset_dim != 1536
        || config.target_dim != 512
        || config.reward_ensemble_k != 4
        || config.embedding_rows != 1359
        || config.routing_classes != 14
        || config.vocab_size != 1103
        || config.max_seq_len != 128
        || config.padding_idx != 0
        || config.layernorm_eps != 1e-5
        || config.gelu != "erf"
        || !config.norm_first
        || config.attn_scale.to_bits() != (1.0f64 / 128.0f64.sqrt()).to_bits()
    {
        bail!("JOPE policy configuration is not the governed policy_v3.2 shape");
    }
    Ok(())
}
