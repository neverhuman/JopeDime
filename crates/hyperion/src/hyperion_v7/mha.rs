//! Multi-head attention (vendored from the veox reference; the hard-coded dim=128/head_dim=32 are
//! parameterized for v7's dim=512/heads=8).

use candle_core::{Result, Tensor};

use super::ops::{linear3, softmax_last, LinearWeights};

pub struct MhaWeights {
    pub in_proj_weight: Tensor,  // (3*dim, dim)
    pub in_proj_bias: Tensor,    // (3*dim,)
    pub out_proj_weight: Tensor, // (dim, dim)
    pub out_proj_bias: Tensor,   // (dim,)
}

pub fn mha_forward(
    query: &Tensor,        // (B, Lq, dim)
    key_value: &Tensor,    // (B, Lk, dim)
    key_pad_mask: &Tensor, // (B, Lk) float32 (1.0 = valid, 0.0 = pad)
    w: &MhaWeights,
    dim: usize,
    heads: usize,
) -> Result<Tensor> {
    let head_dim = (dim / heads) as f64;

    let wq = w.in_proj_weight.narrow(0, 0, dim)?;
    let wk = w.in_proj_weight.narrow(0, dim, dim)?;
    let wv = w.in_proj_weight.narrow(0, 2 * dim, dim)?;

    let bq = w.in_proj_bias.narrow(0, 0, dim)?;
    let bk = w.in_proj_bias.narrow(0, dim, dim)?;
    let bv = w.in_proj_bias.narrow(0, 2 * dim, dim)?;

    let q = linear3(
        query,
        &LinearWeights {
            weight: wq,
            bias: bq,
        },
    )?;
    let k = linear3(
        key_value,
        &LinearWeights {
            weight: wk,
            bias: bk,
        },
    )?;
    let v = linear3(
        key_value,
        &LinearWeights {
            weight: wv,
            bias: bv,
        },
    )?;

    let split_heads = |t: &Tensor| -> Result<Tensor> {
        let (b, l, d) = t.dims3()?;
        t.reshape((b, l, heads, d / heads))?
            .transpose(1, 2)?
            .contiguous()
    };

    let q_h = split_heads(&q)?; // (B, heads, Lq, head_dim)
    let k_h = split_heads(&k)?; // (B, heads, Lk, head_dim)
    let v_h = split_heads(&v)?; // (B, heads, Lk, head_dim)

    let k_t = k_h.transpose(2, 3)?.contiguous()?;
    let scores = (q_h.matmul(&k_t)? / head_dim.sqrt())?;

    // key_pad_mask: 1.0 = valid, 0.0 = pad → very negative where padding occurs.
    let b = scores.dim(0)?;
    let l_k = key_pad_mask.dim(1)?;
    let mask_exp = key_pad_mask
        .reshape((b, 1, 1, l_k))?
        .broadcast_as(scores.shape())?;

    let is_pad = ((mask_exp.zeros_like()? + 1.0)? - &mask_exp)?; // 1.0 where pad
    let fill = (scores.zeros_like()? - 1e4)?; // exp(-1e4) underflows to exactly 0.0 in f32

    let scores_masked = is_pad
        .to_dtype(candle_core::DType::U8)?
        .where_cond(&fill, &scores)?;
    let attn = softmax_last(&scores_masked)?;

    let out = attn.matmul(&v_h.contiguous()?)?; // (B, heads, Lq, head_dim)

    // merge heads
    let (b_s, h_s, l_q, h_d) = out.dims4()?;
    let out_merged = out.transpose(1, 2)?.reshape((b_s, l_q, h_s * h_d))?;

    linear3(
        &out_merged,
        &LinearWeights {
            weight: w.out_proj_weight.clone(),
            bias: w.out_proj_bias.clone(),
        },
    )
}
