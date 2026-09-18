use super::LN_EPS;
use crate::model::tabicl_common::{attention, linear, Rope, Ssmax, Weights};
use anyhow::Result;
use candle_core::{Tensor, D};

/// Bias-free LayerNorm over the last dim (`bias_free_ln=True`), eps 1e-5:
/// `(x - mean) / sqrt(var + eps) * weight`.
pub(super) fn layer_norm(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + LN_EPS)?.sqrt()?;
    let xn = xc.broadcast_div(&denom)?;
    Ok(xn.broadcast_mul(w)?)
}

/// Pre-norm Multihead-Attention block (norm1 -> attn -> residual -> norm2 -> ff
/// -> residual). Bias-free LayerNorm (weights only).
pub(super) struct MabWeights {
    norm1_w: Tensor,
    norm2_w: Tensor,
    in_w: Tensor,
    in_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    lin1_w: Tensor,
    lin1_b: Tensor,
    lin2_w: Tensor,
    lin2_b: Tensor,
    ssmax: Option<Ssmax>,
}

impl MabWeights {
    pub(super) fn load(w: &Weights, prefix: &str, nhead: usize, has_ssmax: bool) -> Result<Self> {
        let e = w.g(&format!("{prefix}.norm1.weight"))?.dim(0)?;
        let ssmax = if has_ssmax {
            Some(Ssmax::load(
                w,
                &format!("{prefix}.attn.ssmax_layer"),
                nhead,
                e / nhead,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1_w: w.g(&format!("{prefix}.norm1.weight"))?.clone(),
            norm2_w: w.g(&format!("{prefix}.norm2.weight"))?.clone(),
            in_w: w.g(&format!("{prefix}.attn.in_proj_weight"))?.clone(),
            in_b: w.g(&format!("{prefix}.attn.in_proj_bias"))?.clone(),
            out_w: w.g(&format!("{prefix}.attn.out_proj.weight"))?.clone(),
            out_b: w.g(&format!("{prefix}.attn.out_proj.bias"))?.clone(),
            lin1_w: w.g(&format!("{prefix}.linear1.weight"))?.clone(),
            lin1_b: w.g(&format!("{prefix}.linear1.bias"))?.clone(),
            lin2_w: w.g(&format!("{prefix}.linear2.weight"))?.clone(),
            lin2_b: w.g(&format!("{prefix}.linear2.bias"))?.clone(),
            ssmax,
        })
    }

    fn ff(&self, x: &Tensor) -> Result<Tensor> {
        let h = linear(x, &self.lin1_w, Some(&self.lin1_b))?.gelu_erf()?;
        linear(&h, &self.lin2_w, Some(&self.lin2_b))
    }

    /// Cross/self attention with explicit q and k/v sources.
    pub(super) fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        nhead: usize,
        rope: Option<&Rope>,
    ) -> Result<Tensor> {
        let qn = layer_norm(q, &self.norm1_w)?;
        let kn = layer_norm(k, &self.norm1_w)?;
        let vn = layer_norm(v, &self.norm1_w)?;
        let attn = attention(
            &qn,
            &kn,
            &vn,
            &self.in_w,
            &self.in_b,
            &self.out_w,
            &self.out_b,
            nhead,
            self.ssmax.as_ref(),
            rope,
        )?;
        let x = (q + attn)?;
        let ffx = self.ff(&layer_norm(&x, &self.norm2_w)?)?;
        Ok((x + ffx)?)
    }
}
