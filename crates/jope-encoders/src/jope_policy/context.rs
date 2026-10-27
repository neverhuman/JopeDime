//! Context execution preserves the accepted CPU oracle and uses device kernels on CUDA.
use super::support::{layer_norm, linear};
use super::{JopePolicyContext, JopePolicyModel};
use anyhow::{bail, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::Module;

impl JopePolicyModel {
    /// Encode the 1536-value dataset embedding and 512-value target summary.
    pub fn context(&self, embedding: &[f32], target: &[f32]) -> Result<JopePolicyContext> {
        if embedding.len() != self.config.dataset_dim || target.len() != self.config.target_dim {
            bail!(
                "JOPE context dimensions must be ({}, {}), got ({}, {})",
                self.config.dataset_dim,
                self.config.target_dim,
                embedding.len(),
                target.len()
            );
        }
        if !embedding
            .iter()
            .chain(target)
            .all(|value| value.is_finite())
        {
            bail!("JOPE context inputs must be finite");
        }
        let mut values = Vec::with_capacity(embedding.len() + target.len());
        values.extend_from_slice(embedding);
        values.extend_from_slice(target);
        let x = Tensor::from_vec(
            values,
            (1, self.config.dataset_dim + self.config.target_dim),
            &self.device,
        )?;
        self.encode_context_tensor(&x)
    }

    /// Consume already-resident components without a device-to-host round trip.
    pub fn context_tensors(
        &self,
        embedding: &Tensor,
        target: &Tensor,
    ) -> Result<JopePolicyContext> {
        if embedding.dims() != [1, self.config.dataset_dim]
            || target.dims() != [1, self.config.target_dim]
            || embedding.dtype() != candle_core::DType::F32
            || target.dtype() != candle_core::DType::F32
            || !embedding.device().same_device(&self.device)
            || !target.device().same_device(&self.device)
        {
            bail!("JOPE resident context shapes, dtype or device differ");
        }
        self.encode_context_tensor(&Tensor::cat(&[embedding, target], 1)?)
    }

    fn encode_context_tensor(&self, x: &Tensor) -> Result<JopePolicyContext> {
        if self.device.is_cuda() {
            return self.encode_context_cuda(x);
        }
        let epsilon = self.config.layernorm_eps;
        let hidden = linear(
            x,
            &self.weight("encoder.0.weight")?,
            &self.weight("encoder.0.bias")?,
        )?
        .gelu_erf()?;
        let hidden = layer_norm(
            &hidden,
            &self.weight("encoder.2.weight")?,
            &self.weight("encoder.2.bias")?,
            epsilon,
        )?;
        let hidden = linear(
            &hidden,
            &self.weight("encoder.4.weight")?,
            &self.weight("encoder.4.bias")?,
        )?
        .gelu_erf()?;
        let encoded = layer_norm(
            &hidden,
            &self.weight("encoder.6.weight")?,
            &self.weight("encoder.6.bias")?,
            epsilon,
        )?;
        Ok(JopePolicyContext(encoded))
    }

    pub(crate) fn context_tensor<'a>(&self, context: &'a JopePolicyContext) -> &'a Tensor {
        &context.0
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn context_values(&self, context: &JopePolicyContext) -> Result<Vec<f32>> {
        Ok(context.0.i(0)?.to_vec1::<f32>()?)
    }

    /// Construct the decoder seam from a previously qualified 512-value context.
    pub fn context_from_values(&self, values: &[f32]) -> Result<JopePolicyContext> {
        if values.len() != self.config.d_model || !values.iter().all(|value| value.is_finite()) {
            bail!("JOPE encoded context must contain 512 finite values");
        }
        Ok(JopePolicyContext(Tensor::from_vec(
            values.to_vec(),
            (1, self.config.d_model),
            &self.device,
        )?))
    }

    /// This context-only path performs no host export. CPU oracle accumulation stays separate.
    fn encode_context_cuda(&self, x: &Tensor) -> Result<JopePolicyContext> {
        let linear_device = |x: &Tensor, layer: &str| -> Result<Tensor> {
            Ok(x.matmul(&self.weight(&format!("{layer}.weight"))?.t()?)?
                .broadcast_add(&self.weight(&format!("{layer}.bias"))?)?)
        };
        let norm_device = |x: &Tensor, layer: &str| -> Result<Tensor> {
            Ok(candle_nn::LayerNorm::new(
                self.weight(&format!("{layer}.weight"))?,
                self.weight(&format!("{layer}.bias"))?,
                self.config.layernorm_eps,
            )
            .forward(x)?)
        };
        let hidden = linear_device(x, "encoder.0")?.gelu_erf()?;
        let hidden = norm_device(&hidden, "encoder.2")?;
        let hidden = linear_device(&hidden, "encoder.4")?.gelu_erf()?;
        Ok(JopePolicyContext(norm_device(&hidden, "encoder.6")?))
    }
}
