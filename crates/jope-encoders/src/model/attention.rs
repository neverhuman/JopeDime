use super::tensor_ops::{add_promote, repeat_dim};
use super::{StarforgeModel, EMBED, HEAD16, HEAD64};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};

impl StarforgeModel {
    pub(super) fn cross_attention_block(
        &self,
        query: &Tensor,
        context: &Tensor,
        prefix: &str,
        scaling: bool,
    ) -> Result<Tensor> {
        let qn = self.rms(query, &format!("{prefix}.layernorm_q.weight"))?;
        let cn = self.rms(context, &format!("{prefix}.layernorm_kv.weight"))?;
        let scaling_prefix = if scaling {
            Some(format!("{prefix}.attn.softmax_scaling_layer"))
        } else {
            None
        };
        let attn = self.attention_projected(
            &qn,
            &cn,
            &format!("{prefix}.attn"),
            scaling_prefix.as_deref(),
            false,
        )?;
        let mut out = add_promote(query, &attn)?;
        let norm2 = self.rms(&out, &format!("{prefix}.layernorm2.weight"))?;
        let mlp = self.mlp(&norm2, &format!("{prefix}.mlp"))?;
        out = add_promote(&out, &mlp)?;
        Ok(out)
    }

    pub(super) fn self_attention(&self, x: &Tensor, prefix: &str, rope: bool) -> Result<Tensor> {
        self.attention_projected(x, x, prefix, None, rope)
    }

    pub(super) fn attention_projected(
        &self,
        query: &Tensor,
        context: &Tensor,
        prefix: &str,
        scaling_prefix: Option<&str>,
        rope: bool,
    ) -> Result<Tensor> {
        let qd = query.dims();
        let cd = context.dims();
        let (b, q_len, _) = (qd[0], qd[1], qd[2]);
        let k_len = cd[1];
        let q_proj = self
            .linear(query, &format!("{prefix}.q_projection.weight"), None)?
            .contiguous()?;
        let q_last = q_proj.dims()[2];
        let head_dim = if q_last == EMBED { HEAD16 } else { HEAD64 };
        let heads = q_last / head_dim;
        let mut q = q_proj.reshape(vec![b, q_len, heads, head_dim])?;
        let mut k = self
            .linear(context, &format!("{prefix}.k_projection.weight"), None)?
            .reshape(vec![b, k_len, heads, head_dim])?;
        let v = self
            .linear(context, &format!("{prefix}.v_projection.weight"), None)?
            .reshape(vec![b, k_len, heads, head_dim])?;
        if rope {
            q = self.apply_rope(&q)?;
            k = self.apply_rope(&k)?;
        }
        let out = self.scaled_attention(&q, &k, &v, scaling_prefix, k_len)?;
        self.linear(
            &out.reshape(vec![b, q_len, heads * head_dim])?,
            &format!("{prefix}.out_projection.weight"),
            None,
        )
    }

    pub(super) fn scaled_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scaling_prefix: Option<&str>,
        n_keys: usize,
    ) -> Result<Tensor> {
        let q_dims = q.dims();
        let hq = q_dims[2];
        let hk = k.dims()[2];
        let (k, v) = if hk != hq {
            if !hq.is_multiple_of(hk) {
                bail!("kv head count {hk} does not divide query heads {hq}");
            }
            (repeat_dim(k, 2, hq / hk)?, repeat_dim(v, 2, hq / hk)?)
        } else {
            (k.clone(), v.clone())
        };
        let q = if let Some(prefix) = scaling_prefix {
            self.softmax_scale(q, prefix, n_keys)?
        } else {
            q.clone()
        };
        let qt = q.transpose(1, 2)?.contiguous()?;
        let kt = k.transpose(1, 2)?.contiguous()?;
        let vt = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0f64 / (q_dims[3] as f64).sqrt();
        let kt_t = kt.transpose(2, 3)?.contiguous()?;
        if self.autocast_bf16 {
            let scores = (qt.matmul(&kt_t)? * scale)?.to_dtype(DType::F32)?;
            let probs = candle_nn::ops::softmax(&scores, 3)?;
            Ok(probs
                .matmul(&vt.to_dtype(DType::F32)?)?
                .to_dtype(DType::BF16)?
                .transpose(1, 2)?
                .contiguous()?)
        } else {
            let scores = (qt.matmul(&kt_t)? * scale)?;
            let probs = candle_nn::ops::softmax(&scores, 3)?;
            Ok(probs.matmul(&vt)?.transpose(1, 2)?.contiguous()?)
        }
    }

    pub(super) fn softmax_scale(&self, q: &Tensor, prefix: &str, n_keys: usize) -> Result<Tensor> {
        let d = q.dims()[3];
        let h = q.dims()[2];
        let logn = (n_keys.max(1) as f32).ln();
        let base_in =
            Tensor::from_vec(vec![logn], vec![1, 1], &self.device)?.to_dtype(q.dtype())?;
        let base = self
            .linear(
                &base_in,
                &format!("{prefix}.base_mlp.0.weight"),
                Some(&format!("{prefix}.base_mlp.0.bias")),
            )?
            .gelu_erf()?;
        let base = self
            .linear(
                &base,
                &format!("{prefix}.base_mlp.2.weight"),
                Some(&format!("{prefix}.base_mlp.2.bias")),
            )?
            .reshape(vec![1, 1, h, d])?;
        let modn = self
            .linear(
                q,
                &format!("{prefix}.query_mlp.0.weight"),
                Some(&format!("{prefix}.query_mlp.0.bias")),
            )?
            .gelu_erf()?;
        let modn = (self
            .linear(
                &modn,
                &format!("{prefix}.query_mlp.2.weight"),
                Some(&format!("{prefix}.query_mlp.2.bias")),
            )?
            .tanh()?
            + 1.0)?;
        Ok(q.broadcast_mul(&base)?.broadcast_mul(&modn)?)
    }

    pub(super) fn apply_rope(&self, q: &Tensor) -> Result<Tensor> {
        let dims = q.dims();
        let (b, s, h, d) = (dims[0], dims[1], dims[2], dims[3]);
        let half = d / 2;
        let t = q.transpose(1, 2)?; // B,H,S,D
        let t_f32 = t.to_dtype(DType::F32)?;
        let freqs = self
            .w("column_aggregator.rope.freqs")?
            .to_dtype(DType::F32)?;
        let pos = Tensor::arange(0f32, s as f32, &self.device)?.reshape(vec![s, 1])?;
        let angles = pos.broadcast_mul(&freqs.reshape(vec![1, half])?)?;
        let cos = Tensor::cat(&[&angles.cos()?, &angles.cos()?], 1)?
            .reshape(vec![1, 1, s, d])?
            .broadcast_as(vec![b, h, s, d])?;
        let sin = Tensor::cat(&[&angles.sin()?, &angles.sin()?], 1)?
            .reshape(vec![1, 1, s, d])?
            .broadcast_as(vec![b, h, s, d])?;
        let first = t_f32.narrow(3, 0, half)?;
        let second = t_f32.narrow(3, half, half)?;
        let rotated = Tensor::cat(&[&second.neg()?, &first], 3)?;
        let out = t_f32
            .broadcast_mul(&cos)?
            .broadcast_add(&rotated.broadcast_mul(&sin)?)?;
        Ok(out.to_dtype(q.dtype())?.transpose(1, 2)?.contiguous()?)
    }

    pub(super) fn mlp(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let h = self
            .linear(x, &format!("{prefix}.0.weight"), None)?
            .gelu_erf()?;
        self.linear(&h, &format!("{prefix}.2.weight"), None)
    }

    pub(super) fn rms(&self, x: &Tensor, weight: &str) -> Result<Tensor> {
        let x = if self.autocast_bf16 {
            x.to_dtype(DType::F32)?.contiguous()?
        } else {
            x.contiguous()?
        };
        candle_nn::ops::rms_norm(&x, self.w(weight)?, 1.192_092_9e-7).map_err(Into::into)
    }

    pub(super) fn linear(&self, x: &Tensor, weight: &str, bias: Option<&str>) -> Result<Tensor> {
        let w = self.w(weight)?;
        let dims = x.dims().to_vec();
        let in_dim = *dims.last().context("linear input has no dims")?;
        let out_dim = w.dims()[0];
        let rows = x.elem_count() / in_dim;
        let flat = x.reshape(vec![rows, in_dim])?;
        let (flat, w) = if self.autocast_bf16 {
            (flat.to_dtype(DType::BF16)?, w.to_dtype(DType::BF16)?)
        } else {
            (flat, w.clone())
        };
        let mut y = flat.matmul(&w.t()?)?;
        if let Some(bias) = bias {
            let b = if self.autocast_bf16 {
                self.w(bias)?.to_dtype(DType::BF16)?
            } else {
                self.w(bias)?.clone()
            };
            y = y.broadcast_add(&b.reshape(vec![1, out_dim])?)?;
        }
        let mut out_shape = dims;
        *out_shape.last_mut().unwrap() = out_dim;
        Ok(y.reshape(out_shape)?)
    }

    pub(super) fn w(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }
}
