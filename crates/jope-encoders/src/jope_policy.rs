//! Verified inference for the governed JOPE `policy_v3.2` model pack.
//!
//! The model is loaded only from a [`LoadedModelBundle`], so callers cannot pair
//! the decoder with unbound metadata or weights. The attention projection stays
//! packed, matching PyTorch's `MultiheadAttention` fp32 evaluation order.

use crate::bundle::LoadedModelBundle;
use anyhow::{bail, Context, Result};
use candle_core::{Device, IndexOp, Tensor, D};

mod context;
mod support;

pub use support::JopePolicyConfig;
use support::{causal_mask, layer_norm, linear, validate_config, PolicyMetadata, Weights};

const POLICY_METADATA: &str = "artifacts/model-bundle/jope_policy.meta.json";
const POLICY_WEIGHTS: &str = "artifacts/model-bundle/jope_policy.safetensors";

/// Opaque dataset/target context. The caller retains its associated model identity.
pub struct JopePolicyContext(Tensor);

pub struct JopePolicyModel {
    weights_sha256: String,
    weights: Weights,
    config: JopePolicyConfig,
    device: Device,
}

impl JopePolicyModel {
    /// Load policy metadata and fp32 weights after descriptor-relative bundle verification.
    pub fn load_bundle(bundle: &LoadedModelBundle, device: Device) -> Result<Self> {
        if !matches!(&device, Device::Cpu) {
            bail!("JOPE policy inference is CPU-only");
        }
        Self::load_bundle_on_device(bundle, device)
    }

    /// Explicit context-encoder device path. CUDA decoding/reward heads remain unsupported.
    pub fn load_bundle_on_device(bundle: &LoadedModelBundle, device: Device) -> Result<Self> {
        if !(device.is_cpu() || device.is_cuda()) {
            bail!("optional JOPE inference supports CPU or CUDA");
        }
        let metadata_bytes = bundle.read_verified(POLICY_METADATA)?;
        let metadata: PolicyMetadata =
            serde_json::from_slice(&metadata_bytes).context("decode verified JOPE metadata")?;
        if metadata.model_version != "policy_v3.2" {
            bail!("unsupported JOPE policy version {}", metadata.model_version);
        }
        validate_config(&metadata.config)?;
        let weight_bytes = bundle.read_verified(POLICY_WEIGHTS)?;
        let weights = Weights::load_buffer(&weight_bytes, &device)?;
        Ok(Self {
            weights_sha256: jope_runtime::sha256_hex(&weight_bytes),
            weights,
            config: metadata.config,
            device,
        })
    }

    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    pub fn config(&self) -> &JopePolicyConfig {
        &self.config
    }

    fn weight(&self, name: &str) -> Result<Tensor> {
        Ok(self.weights.get(name)?.clone())
    }

    /// Return logits for the next token after a fully valid prefix.
    pub fn next_token_logits(
        &self,
        context: &JopePolicyContext,
        token_ids: &[usize],
    ) -> Result<Vec<f32>> {
        if token_ids.len() >= self.config.max_seq_len {
            bail!("JOPE token prefix exceeds the configured sequence limit");
        }
        if token_ids
            .iter()
            .any(|&token| token >= self.config.vocab_size)
        {
            bail!("JOPE token prefix contains an out-of-vocabulary token");
        }
        let full = self.full_logits(&context.0, token_ids)?;
        Ok(full.i((0, token_ids.len()))?.to_vec1::<f32>()?)
    }

    fn full_logits(&self, context: &Tensor, token_ids: &[usize]) -> Result<Tensor> {
        let (hidden, _) = self.decoder_hidden(context, token_ids, false)?;
        let logits = self.tied_projection(&hidden)?;
        Ok(logits.narrow(D::Minus1, 0, self.config.vocab_size)?)
    }

    /// Project a qualified output-normalized state through the tied embedding.
    pub fn parity_logits_from_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.config.d_model || !hidden.iter().all(|value| value.is_finite()) {
            bail!("JOPE parity hidden state must contain 512 finite values");
        }
        let hidden = Tensor::from_vec(hidden.to_vec(), (1, 1, self.config.d_model), &self.device)?;
        let logits = self.tied_projection(&hidden)?;
        Ok(logits
            .narrow(D::Minus1, 0, self.config.vocab_size)?
            .i((0, 0))?
            .to_vec1::<f32>()?)
    }

    /// Apply only the governed decoder output normalization for qualification.
    pub fn parity_output_norm_from_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.config.d_model || !hidden.iter().all(|value| value.is_finite()) {
            bail!("JOPE parity hidden state must contain 512 finite values");
        }
        let hidden = Tensor::from_vec(hidden.to_vec(), (1, 1, self.config.d_model), &self.device)?;
        Ok(layer_norm(
            &hidden,
            &self.weight("out_norm.weight")?,
            &self.weight("out_norm.bias")?,
            self.config.layernorm_eps,
        )?
        .i((0, 0))?
        .to_vec1::<f32>()?)
    }

    fn tied_projection(&self, hidden: &Tensor) -> Result<Tensor> {
        if !self.device.is_cpu() {
            bail!("JOPE scalar vocabulary projection supports CPU only");
        }
        let (batch, sequence, dimension) = hidden.dims3()?;
        if dimension != self.config.d_model {
            bail!("JOPE tied projection hidden dimension changed");
        }
        let rows = batch
            .checked_mul(sequence)
            .context("JOPE tied projection row count overflow")?;
        let embedding_rows = self.config.embedding_rows;
        let left = hidden.flatten_all()?.to_vec1::<f32>()?;
        let right = self
            .weight("token_emb.weight")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut output = vec![0.0f32; rows * embedding_rows];
        for row in 0..rows {
            for inner in 0..dimension {
                let input_value = left[row * dimension + inner];
                for column in 0..embedding_rows {
                    output[row * embedding_rows + column] +=
                        input_value * right[column * dimension + inner];
                }
            }
        }
        Ok(Tensor::from_vec(
            output,
            (batch, sequence, embedding_rows),
            &self.device,
        )?)
    }

    /// Return the last-position state after every decoder layer and output norm.
    /// This is a bounded qualification seam, not a workload-facing inference API.
    pub fn parity_decoder_trace(
        &self,
        context: &JopePolicyContext,
        token_ids: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        if token_ids.len() >= self.config.max_seq_len
            || token_ids
                .iter()
                .any(|&token| token >= self.config.vocab_size)
        {
            bail!("invalid JOPE parity trace prefix");
        }
        Ok(self.decoder_hidden(&context.0, token_ids, true)?.1)
    }

    fn decoder_hidden(
        &self,
        context: &Tensor,
        token_ids: &[usize],
        capture_trace: bool,
    ) -> Result<(Tensor, Vec<Vec<f32>>)> {
        let dimension = self.config.d_model;
        let token_count = token_ids.len();
        let prefix = linear(
            context,
            &self.weight("ctx_proj.weight")?,
            &self.weight("ctx_proj.bias")?,
        )?
        .reshape((1, 1, dimension))?;
        let ids = token_ids
            .iter()
            .map(|&token| token as u32)
            .collect::<Vec<_>>();
        let tokens = self
            .weight("token_emb.weight")?
            .index_select(&Tensor::from_vec(ids, token_count, &self.device)?, 0)?
            .reshape((1, token_count, dimension))?;
        let mut hidden = Tensor::cat(&[&prefix, &tokens], 1)?;
        let sequence = token_count + 1;
        let positions = (0..sequence as u32).collect::<Vec<_>>();
        let position_embedding = self
            .weight("pos_emb.weight")?
            .index_select(&Tensor::from_vec(positions, sequence, &self.device)?, 0)?
            .reshape((1, sequence, dimension))?;
        hidden = hidden.broadcast_add(&position_embedding)?;
        let mask = causal_mask(sequence, &self.device)?;
        let mut trace = Vec::with_capacity(if capture_trace {
            self.config.n_layers + 1
        } else {
            0
        });
        for layer in 0..self.config.n_layers {
            hidden = self.decoder_layer(&hidden, layer, &mask)?;
            if capture_trace {
                trace.push(hidden.i((0, token_count))?.to_vec1::<f32>()?);
            }
        }
        let hidden = layer_norm(
            &hidden,
            &self.weight("out_norm.weight")?,
            &self.weight("out_norm.bias")?,
            self.config.layernorm_eps,
        )?;
        if capture_trace {
            trace.push(hidden.i((0, token_count))?.to_vec1::<f32>()?);
        }
        Ok((hidden, trace))
    }

    fn decoder_layer(&self, hidden: &Tensor, layer: usize, mask: &Tensor) -> Result<Tensor> {
        let prefix = format!("decoder.layers.{layer}");
        let normalized = layer_norm(
            hidden,
            &self.weight(&format!("{prefix}.norm1.weight"))?,
            &self.weight(&format!("{prefix}.norm1.bias"))?,
            self.config.layernorm_eps,
        )?;
        let attention = self.self_attention(&normalized, &prefix, mask)?;
        let hidden = (hidden + attention)?;
        let feed_forward = layer_norm(
            &hidden,
            &self.weight(&format!("{prefix}.norm2.weight"))?,
            &self.weight(&format!("{prefix}.norm2.bias"))?,
            self.config.layernorm_eps,
        )?;
        let feed_forward = linear(
            &feed_forward,
            &self.weight(&format!("{prefix}.linear1.weight"))?,
            &self.weight(&format!("{prefix}.linear1.bias"))?,
        )?
        .gelu_erf()?;
        let feed_forward = linear(
            &feed_forward,
            &self.weight(&format!("{prefix}.linear2.weight"))?,
            &self.weight(&format!("{prefix}.linear2.bias"))?,
        )?;
        Ok((hidden + feed_forward)?)
    }

    fn self_attention(&self, x: &Tensor, prefix: &str, mask: &Tensor) -> Result<Tensor> {
        let dimension = self.config.d_model;
        let heads = self.config.n_heads;
        let head_dimension = dimension / heads;
        let (batch, sequence, _) = x.dims3()?;
        let projection_weight = self.weight(&format!("{prefix}.self_attn.in_proj_weight"))?;
        let projection_bias = self.weight(&format!("{prefix}.self_attn.in_proj_bias"))?;
        let split = |tensor: Tensor| -> Result<Tensor> {
            Ok(tensor
                .reshape((batch, sequence, heads, head_dimension))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        // PyTorch's native CPU fast path adds the packed bias and rescales Q before
        // the QK matmul. Keeping that evaluation order is necessary for the frozen
        // fp32 logit oracle; scaling the completed score matrix drifts beyond 1e-5.
        let projected = x.broadcast_matmul(&projection_weight.t()?)?;
        let query = projected
            .narrow(D::Minus1, 0, dimension)?
            .broadcast_add(&projection_bias.narrow(0, 0, dimension)?)?;
        let query = split((query * self.config.attn_scale)?)?;
        let key = projected
            .narrow(D::Minus1, dimension, dimension)?
            .broadcast_add(&projection_bias.narrow(0, dimension, dimension)?)?;
        let key = split(key)?;
        let value = projected
            .narrow(D::Minus1, 2 * dimension, dimension)?
            .broadcast_add(&projection_bias.narrow(0, 2 * dimension, dimension)?)?;
        let value = split(value)?;
        let scores = query.matmul(&key.transpose(2, 3)?.contiguous()?)?;
        let scores = scores.broadcast_add(&mask.reshape((1, 1, sequence, sequence))?)?;
        let probabilities = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let output = probabilities
            .matmul(&value.contiguous()?)?
            .transpose(1, 2)?
            .reshape((batch, sequence, dimension))?;
        linear(
            &output,
            &self.weight(&format!("{prefix}.self_attn.out_proj.weight"))?,
            &self.weight(&format!("{prefix}.self_attn.out_proj.bias"))?,
        )
    }

    fn genome_embedding(&self, token_ids: &[usize]) -> Result<Tensor> {
        if token_ids.is_empty()
            || token_ids
                .iter()
                .any(|&token| token >= self.config.vocab_size)
        {
            bail!("JOPE reward tokens must be nonempty and in vocabulary");
        }
        let ids = token_ids
            .iter()
            .map(|&token| token as u32)
            .collect::<Vec<_>>();
        let embedding = self
            .weight("reward_token_emb.weight")?
            .index_select(&Tensor::from_vec(ids, token_ids.len(), &self.device)?, 0)?;
        let mean =
            (embedding.sum(0)? / token_ids.len() as f64)?.reshape((1, self.config.d_model))?;
        layer_norm(
            &mean,
            &self.weight("reward_norm.weight")?,
            &self.weight("reward_norm.bias")?,
            self.config.layernorm_eps,
        )
    }

    fn reward_features(&self, context: &Tensor, genome: &Tensor) -> Result<Tensor> {
        let gamma = linear(
            context,
            &self.weight("z_gamma.0.weight")?,
            &self.weight("z_gamma.0.bias")?,
        )?
        .tanh()?;
        let projection = linear(
            context,
            &self.weight("z_proj.weight")?,
            &self.weight("z_proj.bias")?,
        )?;
        let scaled = (genome * &gamma)?;
        let distance = (genome - &projection)?.abs()?;
        Ok(Tensor::cat(&[genome, &scaled, &distance], D::Minus1)?)
    }

    fn scalar_head(
        &self,
        input: &Tensor,
        weight0: &str,
        bias0: &str,
        weight1: &str,
        bias1: &str,
    ) -> Result<f32> {
        let hidden = linear(input, &self.weight(weight0)?, &self.weight(bias0)?)?.gelu_erf()?;
        Ok(
            linear(&hidden, &self.weight(weight1)?, &self.weight(bias1)?)?
                .i((0, 0))?
                .to_scalar::<f32>()?,
        )
    }

    pub fn reward_stats(
        &self,
        context: &JopePolicyContext,
        token_ids: &[usize],
    ) -> Result<(f32, f32)> {
        let genome = self.genome_embedding(token_ids)?;
        let features = self.reward_features(&context.0, &genome)?;
        let rewards = (0..self.config.reward_ensemble_k)
            .map(|index| {
                self.scalar_head(
                    &features,
                    &format!("reward_heads.{index}.0.weight"),
                    &format!("reward_heads.{index}.0.bias"),
                    &format!("reward_heads.{index}.3.weight"),
                    &format!("reward_heads.{index}.3.bias"),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let variance = rewards
            .iter()
            .map(|reward| (reward - mean).powi(2))
            .sum::<f32>()
            / (rewards.len() as f32 - 1.0);
        Ok((mean, variance.sqrt()))
    }

    pub fn p_bad(&self, context: &JopePolicyContext, token_ids: &[usize]) -> Result<f32> {
        let genome = self.genome_embedding(token_ids)?;
        let features = self.reward_features(&context.0, &genome)?;
        let logit = self.scalar_head(
            &features,
            "stability_head.0.weight",
            "stability_head.0.bias",
            "stability_head.2.weight",
            "stability_head.2.bias",
        )?;
        Ok(1.0 / (1.0 + (-logit).exp()))
    }
}
