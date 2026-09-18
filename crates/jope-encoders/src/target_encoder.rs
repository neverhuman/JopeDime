//! Pure-Rust port of the frozen fused target encoder.
//!
//! The deployed artifact contains the two neural checkpoints and the fitted
//! signature vectorizer.  This module deliberately reproduces the frozen
//! NumPy sampling stream as well as the PyTorch forward pass so callers encode
//! the supplied labels, rather than accepting precomputed fixture vectors.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;

use crate::bundle::LoadedModelBundle;

mod features;
mod preprocessing;
mod rng;
#[cfg(test)]
mod tests;

use features::{classical_summary, neural_explicit_stats, torch_unit_point};
use preprocessing::*;
#[cfg(test)]
use rng::seed_sequence;
pub(crate) use rng::NumpyPcg64;

const WINDOW_SIZE: usize = 32;
const SIGNATURE_VIEWS: usize = 2;
const FEATURE_DIM: usize = 586;
const SIGNATURE_DIM: usize = FEATURE_DIM * 4;
const CLASSICAL_DIM: usize = 384;
const EMBEDDING_DIM: usize = 64;
const TOKEN_DIM: usize = 512;
const HEADS: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetTask {
    Binary,
    Regression,
}

impl TargetTask {
    fn id(self) -> usize {
        match self {
            Self::Binary => 0,
            Self::Regression => 1,
        }
    }
}

impl std::str::FromStr for TargetTask {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "binary" => Ok(Self::Binary),
            "regression" => Ok(Self::Regression),
            _ => bail!("target task must be binary or regression"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FusedTargetEncoding {
    pub preprocessed_values: Vec<f32>,
    pub raw_signature: Vec<f32>,
    pub neural_best: Vec<f32>,
    pub neural_final: Vec<f32>,
    pub classical_component: Vec<f32>,
    pub vector: Vec<f32>,
}

pub struct FusedTargetEncoder {
    tensors: HashMap<String, Tensor>,
    device: Device,
    seed: u64,
}

impl FusedTargetEncoder {
    pub fn load(bundle: &LoadedModelBundle) -> Result<Self> {
        let identity = bundle
            .manifest
            .encoders
            .get("target")
            .context("model bundle has no target encoder")?;
        let bytes = bundle
            .read_verified(&identity.artifact)
            .context("read and re-verify target encoder")?;
        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load_buffer(&bytes, &device)
            .context("load verified target encoder")?;
        if tensors.len() != 171 {
            bail!("target encoder has {} tensors, expected 171", tensors.len());
        }
        Ok(Self {
            tensors,
            device,
            seed: bundle.target_seed(),
        })
    }

    pub fn encode(&self, values: &[f32], task: TargetTask) -> Result<FusedTargetEncoding> {
        let values = preprocess(values, task)?;
        let raw_signature = raw_signature(&values, task, self.seed)?;
        let classical_component = self.classical_component(&raw_signature)?;
        let windows = neural_windows(values.len(), self.seed)?;
        let neural_best = self.neural_embedding(&values, task, &windows, "neural_best")?;
        let neural_final = self.neural_embedding(&values, task, &windows, "neural_final")?;
        let mut vector = safe_unit(&neural_best);
        vector.extend(safe_unit(&neural_final));
        vector.extend(safe_unit(&classical_component));
        let vector = safe_unit(&vector);
        if vector.len() != 512 {
            bail!(
                "target encoder produced {} values, expected 512",
                vector.len()
            );
        }
        Ok(FusedTargetEncoding {
            preprocessed_values: values,
            raw_signature,
            neural_best,
            neural_final,
            classical_component,
            vector,
        })
    }

    fn classical_component(&self, signature: &[f32]) -> Result<Vec<f32>> {
        if signature.len() != SIGNATURE_DIM {
            bail!("target signature has invalid width {}", signature.len());
        }
        let mean = self.vec_f32("vectorizer.scaler.mean")?;
        let scale = self.vec_f32("vectorizer.scaler.scale")?;
        let scaled = signature
            .iter()
            .zip(mean)
            .zip(scale)
            .map(|((&value, mean), scale)| (value - mean) / scale)
            .collect::<Vec<_>>();
        let components = self
            .w("vectorizer.reducer.components")?
            .to_dtype(DType::F32)?;
        let mut result = Tensor::from_vec(scaled, (1, SIGNATURE_DIM), &self.device)?
            .matmul(&components.t()?)?
            .to_vec2::<f32>()?
            .remove(0);
        result.resize(CLASSICAL_DIM, 0.0);
        Ok(result)
    }

    fn neural_embedding(
        &self,
        values: &[f32],
        task: TargetTask,
        windows: &[Vec<usize>],
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let mut embeddings = Vec::with_capacity(windows.len());
        for indices in windows {
            let sampled = indices
                .iter()
                .map(|&index| values[index])
                .collect::<Vec<_>>();
            embeddings.push(self.neural_window(&sampled, task, values.len(), prefix)?);
        }
        let mut combined = vec![0.0f32; EMBEDDING_DIM];
        for embedding in &embeddings {
            for (dst, src) in combined.iter_mut().zip(embedding) {
                *dst += *src;
            }
        }
        for value in &mut combined {
            *value /= embeddings.len() as f32;
        }
        Ok(safe_unit(&combined))
    }

    fn neural_window(
        &self,
        values: &[f32],
        task: TargetTask,
        original_length: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        if values.len() != WINDOW_SIZE {
            bail!(
                "target neural window has {} values, expected {WINDOW_SIZE}",
                values.len()
            );
        }
        let values_t = Tensor::from_vec(values.to_vec(), (1, WINDOW_SIZE), &self.device)?;
        let value_column = values_t.unsqueeze(2)?;
        let centers = Tensor::from_vec(
            (0..16)
                .map(|index| torch_unit_point(index, 16))
                .collect::<Vec<_>>(),
            (1, 1, 16),
            &self.device,
        )?;
        let rbf = value_column
            .broadcast_sub(&centers)?
            .sqr()?
            .affine(-48.0, 0.0)?
            .exp()?;
        let frequencies = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &self.device)?;
        let angles = value_column
            .broadcast_mul(&frequencies)?
            .affine(2.0 * std::f64::consts::PI, 0.0)?;
        let sin = angles.sin()?;
        let cos = angles.cos()?;
        let mut task_values = Vec::with_capacity(WINDOW_SIZE * 2);
        for _ in 0..WINDOW_SIZE {
            task_values.extend(if task == TargetTask::Binary {
                [1.0f32, 0.0]
            } else {
                [0.0f32, 1.0]
            });
        }
        let task_one_hot = Tensor::from_vec(task_values, (1, WINDOW_SIZE, 2), &self.device)?;
        let mask = Tensor::ones((1, WINDOW_SIZE, 1), DType::F32, &self.device)?;
        let features = Tensor::cat(&[&value_column, &rbf, &sin, &cos, &task_one_hot, &mask], 2)?;
        if features.dims() != [1, WINDOW_SIZE, 28] {
            bail!(
                "target value features have invalid shape {:?}",
                features.dims()
            );
        }

        let mut tokens = self
            .linear(&features, &format!("{prefix}.token_mlp.0"))?
            .gelu_erf()?;
        tokens = self.residual_mlp(&tokens, &format!("{prefix}.token_mlp.2"))?;
        for layer in 0..4 {
            let block = format!("{prefix}.set_blocks.{layer}");
            let normalized = self.layer_norm(&tokens, &format!("{block}.norm"))?;
            let attended = self.self_attention(&normalized, &format!("{block}.attn"))?;
            let residual = (&tokens + &attended)?;
            tokens = self.residual_mlp(&residual, &format!("{block}.ff"))?;
        }
        let pooled_mean = tokens.mean(1)?;
        let pooled_max = tokens.max(1)?;
        let stats = neural_explicit_stats(values, task, original_length);
        if stats.len() != 442 {
            bail!(
                "target neural statistics have invalid width {}",
                stats.len()
            );
        }
        let stats = Tensor::from_vec(stats, (1, 442), &self.device)?;
        let mut stats_features = self
            .linear(&stats, &format!("{prefix}.stats_mlp.0"))?
            .gelu_erf()?;
        stats_features = self.residual_mlp(&stats_features, &format!("{prefix}.stats_mlp.2"))?;
        let fused = Tensor::cat(&[&pooled_mean, &pooled_max, &stats_features], 1)?;
        let mut embedding = self
            .linear(&fused, &format!("{prefix}.fusion.0"))?
            .gelu_erf()?;
        embedding = self.layer_norm(&embedding, &format!("{prefix}.fusion.2"))?;
        embedding = self
            .linear(&embedding, &format!("{prefix}.fusion.3"))?
            .gelu_erf()?;
        embedding = self.linear(&embedding, &format!("{prefix}.fusion.5"))?;
        let embedding = embedding.to_vec2::<f32>()?.remove(0);
        Ok(safe_unit(&embedding))
    }

    fn self_attention(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let in_weight = self
            .w(&format!("{prefix}.in_proj_weight"))?
            .to_dtype(DType::F32)?;
        let in_bias = self
            .w(&format!("{prefix}.in_proj_bias"))?
            .to_dtype(DType::F32)?;
        let q_weight = in_weight.narrow(0, 0, TOKEN_DIM)?;
        let k_weight = in_weight.narrow(0, TOKEN_DIM, TOKEN_DIM)?;
        let v_weight = in_weight.narrow(0, TOKEN_DIM * 2, TOKEN_DIM)?;
        let q_bias = in_bias.narrow(0, 0, TOKEN_DIM)?;
        let k_bias = in_bias.narrow(0, TOKEN_DIM, TOKEN_DIM)?;
        let v_bias = in_bias.narrow(0, TOKEN_DIM * 2, TOKEN_DIM)?;
        let project = |weight: &Tensor, bias: &Tensor| -> Result<Tensor> {
            let projected = linear_tensor(x, weight, Some(bias))?;
            Ok(projected
                .reshape((1, WINDOW_SIZE, HEADS, TOKEN_DIM / HEADS))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let query = project(&q_weight, &q_bias)?;
        let key = project(&k_weight, &k_bias)?;
        let value = project(&v_weight, &v_bias)?;
        let scores = query
            .matmul(&key.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / ((TOKEN_DIM / HEADS) as f64).sqrt(), 0.0)?;
        let probabilities = candle_nn::ops::softmax(&scores, 3)?;
        let context = probabilities
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, WINDOW_SIZE, TOKEN_DIM))?;
        self.linear_named(
            &context,
            &format!("{prefix}.out_proj.weight"),
            Some(&format!("{prefix}.out_proj.bias")),
        )
    }

    fn residual_mlp(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let normalized = self.layer_norm(x, &format!("{prefix}.net.0"))?;
        let hidden = self
            .linear(&normalized, &format!("{prefix}.net.1"))?
            .gelu_erf()?;
        Ok((x + self.linear(&hidden, &format!("{prefix}.net.3"))?)?)
    }

    fn layer_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normalized = centered.broadcast_div(&(variance + 1e-5)?.sqrt()?)?;
        Ok(normalized
            .broadcast_mul(self.w(&format!("{prefix}.weight"))?)?
            .broadcast_add(self.w(&format!("{prefix}.bias"))?)?)
    }

    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        self.linear_named(
            x,
            &format!("{prefix}.weight"),
            Some(&format!("{prefix}.bias")),
        )
    }

    fn linear_named(&self, x: &Tensor, weight: &str, bias: Option<&str>) -> Result<Tensor> {
        linear_tensor(
            x,
            self.w(weight)?,
            bias.map(|name| self.w(name)).transpose()?,
        )
    }

    fn w(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing target encoder tensor {name}"))
    }

    fn vec_f32(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.w(name)?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
    }
}
