use super::support::{linear_tensor, load_safetensors};
use super::{
    logit_scale_key, ClassificationExpertFeatures, ATTN_HEADS, ATTN_HEAD_DIM, ATTN_LAYERS,
    ATTN_MODEL_DIM, CONF_DIM, GATE_META_DIM, HIDDEN_DIM, LOGIT_DIM, N_EXPERTS,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

pub(super) struct AttnFusionCls {
    weights: HashMap<String, Tensor>,
    device: Device,
}

impl AttnFusionCls {
    pub(super) fn load(path: &Path, device: Device) -> Result<Self> {
        let weights = load_safetensors(path, &device)?;
        Ok(Self { weights, device })
    }

    pub(super) fn forward(
        &self,
        a: &ClassificationExpertFeatures,
        b: &ClassificationExpertFeatures,
        gate_meta: &[f32; GATE_META_DIM],
    ) -> Result<Vec<f64>> {
        let m = a.n_rows;
        if m == 0 {
            return Ok(Vec::new());
        }

        let logits0 = Tensor::from_vec(a.logits.clone(), (m, LOGIT_DIM), &self.device)?;
        let logits1 = Tensor::from_vec(b.logits.clone(), (m, LOGIT_DIM), &self.device)?;
        let cal0 = self.calibrate_logits(&logits0, 0)?;
        let cal1 = self.calibrate_logits(&logits1, 1)?;
        let cal = Tensor::cat(&[&cal0.unsqueeze(1)?, &cal1.unsqueeze(1)?], 1)?;

        let h0 = Tensor::from_vec(a.hidden.clone(), (m, HIDDEN_DIM), &self.device)?;
        let h1 = Tensor::from_vec(b.hidden.clone(), (m, HIDDEN_DIM), &self.device)?;
        let z0 = self.adapter(&h0, 0)?;
        let z1 = self.adapter(&h1, 1)?;

        let conf0 = Tensor::from_vec(a.conf.clone(), (m, CONF_DIM), &self.device)?;
        let conf1 = Tensor::from_vec(b.conf.clone(), (m, CONF_DIM), &self.device)?;
        let tok0_in = Tensor::cat(&[&z0, &cal0, &conf0], 1)?;
        let tok1_in = Tensor::cat(&[&z1, &cal1, &conf1], 1)?;
        let tok0 = self
            .linear(&tok0_in, "tok.weight", Some("tok.bias"))?
            .unsqueeze(1)?;
        let tok1 = self
            .linear(&tok1_in, "tok.weight", Some("tok.bias"))?
            .unsqueeze(1)?;

        let mut gate = Vec::with_capacity(m * GATE_META_DIM);
        for _ in 0..m {
            gate.extend_from_slice(gate_meta);
        }
        let gate = Tensor::from_vec(gate, (m, GATE_META_DIM), &self.device)?;
        let meta_tok = self
            .linear(&gate, "meta_tok.weight", Some("meta_tok.bias"))?
            .unsqueeze(1)?;
        let tokens = Tensor::cat(&[&meta_tok, &tok0, &tok1], 1)?;
        let type_emb =
            self.w("type_emb")?
                .unsqueeze(0)?
                .broadcast_as((m, N_EXPERTS + 1, ATTN_MODEL_DIM))?;
        let mut h = tokens.broadcast_add(&type_emb)?;
        for layer in 0..ATTN_LAYERS {
            h = self.encoder_layer(&h, layer)?;
        }

        let expert_h = h.narrow(1, 1, N_EXPERTS)?;
        let gate_logits = self
            .linear(&expert_h, "gate_head.weight", Some("gate_head.bias"))?
            .reshape((m, N_EXPERTS))?;
        let route = candle_nn::ops::softmax(&gate_logits, 1)?;
        let routed = route.unsqueeze(2)?.broadcast_mul(&cal)?.sum(1)?;
        let cls = h.narrow(1, 0, 1)?.reshape((m, ATTN_MODEL_DIM))?;
        let residual = (self.resid_head(&cls)? * self.scalar("resid_scale")? as f64)?;
        let final_logits = (routed + residual)?.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax(&final_logits, 1)?
            .narrow(1, 1, 1)?
            .reshape((m,))?
            .to_vec1::<f32>()?;
        Ok(probs.into_iter().map(f64::from).collect())
    }

    fn adapter(&self, h: &Tensor, expert: usize) -> Result<Tensor> {
        let prefix = format!("adapters.{expert}");
        let z = self.layer_norm(h, &format!("{prefix}.pre"))?;
        let z = self
            .linear(&z, &format!("{prefix}.down.weight"), None)?
            .gelu_erf()?;
        let z = self.linear(&z, &format!("{prefix}.up.weight"), None)?;
        let z = self.layer_norm(&z, &format!("{prefix}.post"))?;
        Ok(z.broadcast_mul(self.w(&format!("{prefix}.gain"))?)?)
    }

    fn calibrate_logits(&self, logits: &Tensor, expert: usize) -> Result<Tensor> {
        let scale = self
            .scalar(&format!("cal.{expert}.{}", logit_scale_key()))?
            .exp()
            .clamp(0.2, 5.0);
        Ok((logits / scale as f64)?.broadcast_add(
            &self
                .w(&format!("cal.{expert}.bias"))?
                .reshape((1, LOGIT_DIM))?,
        )?)
    }

    fn encoder_layer(&self, x: &Tensor, layer: usize) -> Result<Tensor> {
        let prefix = format!("tf.layers.{layer}");
        let attn = self.self_attention(x, &format!("{prefix}.self_attn"))?;
        let x = self.layer_norm(&(x + attn)?, &format!("{prefix}.norm1"))?;
        let ff = self
            .linear(
                &x,
                &format!("{prefix}.linear1.weight"),
                Some(&format!("{prefix}.linear1.bias")),
            )?
            .gelu_erf()?;
        let ff = self.linear(
            &ff,
            &format!("{prefix}.linear2.weight"),
            Some(&format!("{prefix}.linear2.bias")),
        )?;
        self.layer_norm(&(x + ff)?, &format!("{prefix}.norm2"))
    }

    fn self_attention(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let dims = x.dims();
        let (m, seq, d_model) = (dims[0], dims[1], dims[2]);
        let in_w = self.w(&format!("{prefix}.in_proj_weight"))?;
        let in_b = self.w(&format!("{prefix}.in_proj_bias"))?;
        let wq = in_w.narrow(0, 0, d_model)?;
        let wk = in_w.narrow(0, d_model, d_model)?;
        let wv = in_w.narrow(0, 2 * d_model, d_model)?;
        let bq = in_b.narrow(0, 0, d_model)?;
        let bk = in_b.narrow(0, d_model, d_model)?;
        let bv = in_b.narrow(0, 2 * d_model, d_model)?;
        let q = linear_tensor(x, &wq, Some(&bq))?
            .reshape((m, seq, ATTN_HEADS, ATTN_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = linear_tensor(x, &wk, Some(&bk))?
            .reshape((m, seq, ATTN_HEADS, ATTN_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = linear_tensor(x, &wv, Some(&bv))?
            .reshape((m, seq, ATTN_HEADS, ATTN_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)?
            * (1.0f64 / (ATTN_HEAD_DIM as f64).sqrt()))?;
        let probs = candle_nn::ops::softmax(&scores, 3)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((m, seq, d_model))?;
        self.linear(
            &ctx,
            &format!("{prefix}.out_proj.weight"),
            Some(&format!("{prefix}.out_proj.bias")),
        )
    }

    fn resid_head(&self, x: &Tensor) -> Result<Tensor> {
        let h = self
            .linear(x, "resid_head.0.weight", Some("resid_head.0.bias"))?
            .gelu_erf()?;
        self.linear(&h, "resid_head.3.weight", Some("resid_head.3.bias"))
    }

    fn layer_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let denom = (var + 1e-5)?.sqrt()?;
        let xn = xc.broadcast_div(&denom)?;
        Ok(xn
            .broadcast_mul(self.w(&format!("{prefix}.weight"))?)?
            .broadcast_add(self.w(&format!("{prefix}.bias"))?)?)
    }

    fn linear(&self, x: &Tensor, weight: &str, bias: Option<&str>) -> Result<Tensor> {
        let w = self.w(weight)?;
        let b = bias.map(|name| self.w(name)).transpose()?;
        linear_tensor(x, w, b)
    }

    fn scalar(&self, name: &str) -> Result<f32> {
        Ok(self.w(name)?.flatten_all()?.to_vec1::<f32>()?[0])
    }

    fn w(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .with_context(|| format!("missing Chimera classifier tensor {name}"))
    }
}
