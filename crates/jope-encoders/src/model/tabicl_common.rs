//! Shared candle building blocks for the TabICL classifier and regressor ports.
//!
//! Every unit in this module is byte-for-byte identical across both checkpoints:
//! the weight loader (`Weights`), the `linear`/`softmax_last`/head-split helpers,
//! the QASSMax (`Ssmax`) scaling layer, RoPE (`Rope`), and the generic multi-head
//! `attention`. Keeping the single definition here and calling it from both
//! `tabicl_classifier` and `tabicl_regressor` preserves the exact numerics while
//! removing the copy-paste.
//!
//! The pieces that genuinely differ between the two ports intentionally stay in
//! each model file: LayerNorm (affine bias vs `bias_free_ln=True`), the
//! `MabWeights` block that wraps it, the target encoders (one-hot vs continuous),
//! and the output heads.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

/// Loaded weights keyed by the original PyTorch state_dict names.
pub(super) struct Weights {
    t: HashMap<String, Tensor>,
    dev: Device,
}

impl Weights {
    pub(super) fn load(path: &Path, dev: Device) -> Result<Self> {
        let raw = if path.extension().and_then(|e| e.to_str()) == Some("zst") {
            let compressed =
                std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            let bytes = super::tensor_ops::decompress_zstd(&compressed)
                .with_context(|| format!("decompressing {}", path.display()))?;
            candle_core::safetensors::load_buffer(&bytes, &dev)
                .with_context(|| format!("load decompressed {}", path.display()))?
        } else {
            candle_core::safetensors::load(path, &dev)
                .with_context(|| format!("load safetensors from {}", path.display()))?
        };
        let mut t = HashMap::new();
        for (name, tensor) in raw {
            t.insert(name, tensor.to_dtype(DType::F32)?.to_device(&dev)?);
        }
        Ok(Self { t, dev })
    }

    pub(super) fn device(&self) -> &Device {
        &self.dev
    }

    pub(super) fn g(&self, name: &str) -> Result<&Tensor> {
        self.t
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))
    }
}

/// `x @ w.T + b`, flattening all-but-last input dims (w: (out, in)).
pub(super) fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let last = *dims.last().unwrap();
    let lead: usize = dims[..dims.len() - 1].iter().product();
    let x2 = x.reshape((lead, last))?;
    let out_dim = w.dim(0)?;
    let y = x2.matmul(&w.t()?)?; // (lead, out)
    let y = match b {
        Some(b) => y.broadcast_add(b)?,
        None => y,
    };
    let mut new_dims = dims[..dims.len() - 1].to_vec();
    new_dims.push(out_dim);
    Ok(y.reshape(new_dims)?)
}

/// Numerically-stable softmax over the last dimension.
pub(super) fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    let sum = e.sum_keepdim(D::Minus1)?;
    Ok(e.broadcast_div(&sum)?)
}

/// Reshape (..., T, E) -> (..., nhead, T, head_dim).
pub(super) fn split_heads(x: &Tensor, nhead: usize) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let t = dims[dims.len() - 2];
    let e = dims[dims.len() - 1];
    let hd = e / nhead;
    let mut nd = dims[..dims.len() - 2].to_vec();
    nd.push(t);
    nd.push(nhead);
    nd.push(hd);
    let x = x.reshape(nd)?;
    // swap T and nhead -> (..., nhead, T, hd)
    let r = x.rank();
    Ok(x.transpose(r - 3, r - 2)?.contiguous()?)
}

/// Inverse of [`split_heads`]: (..., nhead, T, head_dim) -> (..., T, E).
pub(super) fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let r = x.rank();
    let x = x.transpose(r - 3, r - 2)?.contiguous()?; // (..., T, nhead, hd)
    let dims = x.dims().to_vec();
    let t = dims[dims.len() - 3];
    let nhead = dims[dims.len() - 2];
    let hd = dims[dims.len() - 1];
    let mut nd = dims[..dims.len() - 3].to_vec();
    nd.push(t);
    nd.push(nhead * hd);
    Ok(x.reshape(nd)?)
}

/// QASSMax-elementwise scaling layer.
pub(super) struct Ssmax {
    base0_w: Tensor,
    base0_b: Tensor,
    base2_w: Tensor,
    base2_b: Tensor,
    query0_w: Tensor,
    query0_b: Tensor,
    query2_w: Tensor,
    query2_b: Tensor,
    nhead: usize,
    head_dim: usize,
}

impl Ssmax {
    pub(super) fn load(w: &Weights, prefix: &str, nhead: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            base0_w: w.g(&format!("{prefix}.base_mlp.0.weight"))?.clone(),
            base0_b: w.g(&format!("{prefix}.base_mlp.0.bias"))?.clone(),
            base2_w: w.g(&format!("{prefix}.base_mlp.2.weight"))?.clone(),
            base2_b: w.g(&format!("{prefix}.base_mlp.2.bias"))?.clone(),
            query0_w: w.g(&format!("{prefix}.query_mlp.0.weight"))?.clone(),
            query0_b: w.g(&format!("{prefix}.query_mlp.0.bias"))?.clone(),
            query2_w: w.g(&format!("{prefix}.query_mlp.2.weight"))?.clone(),
            query2_b: w.g(&format!("{prefix}.query_mlp.2.bias"))?.clone(),
            nhead,
            head_dim,
        })
    }

    /// Scale q (..., nhead, Tq, head_dim) given key length `n`.
    pub(super) fn apply(&self, q: &Tensor, n: usize) -> Result<Tensor> {
        let dev = q.device();
        let logn = (n.max(1) as f64).ln();
        let logn_t = Tensor::from_vec(vec![logn as f32], (1, 1), dev)?;
        // base_mlp(logn) -> (1, nhead*head_dim)
        let base = linear(&logn_t, &self.base0_w, Some(&self.base0_b))?.gelu_erf()?;
        let base = linear(&base, &self.base2_w, Some(&self.base2_b))?; // (1, nhead*hd)
                                                                       // reshape to broadcast (1,..,1, nhead, 1, head_dim)
        let r = q.rank();
        let mut bshape = vec![1usize; r];
        bshape[r - 3] = self.nhead;
        bshape[r - 1] = self.head_dim;
        let base = base.reshape(bshape)?;
        // query_mlp(q) -> same shape as q
        let qm = linear(q, &self.query0_w, Some(&self.query0_b))?.gelu_erf()?;
        let qm = linear(&qm, &self.query2_w, Some(&self.query2_b))?;
        let modulation = (qm.tanh()? + 1.0)?;
        let scales = base.broadcast_mul(&modulation)?;
        Ok(q.broadcast_mul(&scales)?)
    }
}

/// Rotary position embedding (non-interleaved).
pub(super) struct Rope {
    freqs: Vec<f32>, // length head_dim/2
}

impl Rope {
    pub(super) fn new(head_dim: usize, base: f64) -> Self {
        let half = head_dim / 2;
        let freqs = (0..half)
            .map(|j| (1.0 / base.powf((2 * j) as f64 / head_dim as f64)) as f32)
            .collect();
        Self { freqs }
    }

    /// cos/sin tables of shape (T, head_dim) for positions 0..T-1.
    fn tables(&self, t: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let half = self.freqs.len();
        let hd = half * 2;
        let mut cos = vec![0f32; t * hd];
        let mut sin = vec![0f32; t * hd];
        for p in 0..t {
            for j in 0..half {
                let ang = p as f32 * self.freqs[j];
                let (c, s) = (ang.cos(), ang.sin());
                cos[p * hd + j] = c;
                cos[p * hd + half + j] = c;
                sin[p * hd + j] = s;
                sin[p * hd + half + j] = s;
            }
        }
        Ok((
            Tensor::from_vec(cos, (t, hd), dev)?,
            Tensor::from_vec(sin, (t, hd), dev)?,
        ))
    }

    /// Apply RoPE to x (..., nhead, T, head_dim) over positions 0..T-1.
    pub(super) fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let t = dims[dims.len() - 2];
        let hd = dims[dims.len() - 1];
        let half = hd / 2;
        let (cos, sin) = self.tables(t, x.device())?; // (T, hd)
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?; // rotate_half_contiguous
        let out = x.broadcast_mul(&cos)?.add(&rot.broadcast_mul(&sin)?)?;
        Ok(out)
    }
}

/// Generic multi-head attention. q:(B,Tq,E), k/v:(B,Tk,E).
#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    in_w: &Tensor, // (3E, E)
    in_b: &Tensor, // (3E,)
    out_w: &Tensor,
    out_b: &Tensor,
    nhead: usize,
    ssmax: Option<&Ssmax>,
    rope: Option<&Rope>,
) -> Result<Tensor> {
    let e = q.dim(D::Minus1)?;
    let hd = e / nhead;
    let wq = in_w.narrow(0, 0, e)?;
    let wk = in_w.narrow(0, e, e)?;
    let wv = in_w.narrow(0, 2 * e, e)?;
    let bq = in_b.narrow(0, 0, e)?;
    let bk = in_b.narrow(0, e, e)?;
    let bv = in_b.narrow(0, 2 * e, e)?;

    let qp = split_heads(&linear(q, &wq, Some(&bq))?, nhead)?; // (B,nh,Tq,hd)
    let kp = split_heads(&linear(k, &wk, Some(&bk))?, nhead)?;
    let vp = split_heads(&linear(v, &wv, Some(&bv))?, nhead)?;

    let (qp, kp) = match rope {
        Some(r) => (r.apply(&qp)?, r.apply(&kp)?),
        None => (qp, kp),
    };

    let tk = kp.dim(D::Minus2)?;
    let qp = match ssmax {
        Some(s) => s.apply(&qp, tk)?,
        None => qp,
    };

    let scale = 1.0 / (hd as f64).sqrt();
    let scores = (qp.matmul(&kp.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;
    let attn = softmax_last(&scores)?;
    let ctx = attn.matmul(&vp)?; // (B,nh,Tq,hd)
    let ctx = merge_heads(&ctx)?; // (B,Tq,E)
    linear(&ctx, out_w, Some(out_b))
}
