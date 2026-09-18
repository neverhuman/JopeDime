//! Masked pooled statistics (vendored verbatim from the veox reference — dim-agnostic).

use candle_core::{DType, Result, Tensor};

use super::ops::{linear3, relu, LinearWeights};

pub struct MpsWeights {
    pub proj_0: LinearWeights,
    pub proj_2: LinearWeights,
}

impl MpsWeights {
    /// `dim` selects the pooled axis: 2 (pool columns → row states) or 1 (pool rows → col states).
    pub fn forward(&self, x: &Tensor, mask: &Tensor, dim: usize) -> Result<Tensor> {
        let me = mask.unsqueeze(3)?; // (B, P, P, 1)
        let me_b = me.broadcast_as(x.shape())?; // (B, P, P, D)
        let cnt = mask.sum(dim)?.unsqueeze(2)?.clamp(1.0f64, 1e9f64)?; // (B, P, 1)
        let mean = (x * &me_b)?.sum(dim)?.broadcast_div(&cnt)?; // (B, P, D)
        let mean_unsq = mean.unsqueeze(dim)?; // (B, 1, P, D) or (B, P, 1, D)

        let diff = x.broadcast_sub(&mean_unsq)? * &me_b;
        let diff = diff?;
        let var = (&diff * &diff)?.sum(dim)?.broadcast_div(&cnt)?;
        let std = (var + 1e-6f64)?.sqrt()?;

        let is_valid = me.broadcast_as(x.shape())?.to_dtype(DType::U8)?;
        let fill = (x.zeros_like()? - 1e4f64)?;
        let mx = is_valid.where_cond(x, &fill)?.max(dim)?;

        let empty = mask.sum(dim)?.eq(&mask.sum(dim)?.zeros_like()?)?;
        let empty_u8 = empty
            .unsqueeze(2)?
            .broadcast_as(mean.shape())?
            .to_dtype(DType::U8)?;

        let mean = empty_u8.where_cond(&mean.zeros_like()?, &mean)?;
        let std = empty_u8.where_cond(&std.zeros_like()?, &std)?;
        let mx = empty_u8.where_cond(&mx.zeros_like()?, &mx)?;

        let cat = Tensor::cat(&[&mean, &std, &mx], 2)?; // (B, P, 3D)
        let out = relu(&linear3(&cat, &self.proj_0)?)?;
        linear3(&out, &self.proj_2)
    }
}

/// Masked global statistics over the length axis: `t` (B, P, D), `mask1d` (B, P) → (B, 3D)
/// concatenated as [mean, max, std].
pub(super) fn global_stats(t: &Tensor, mask1d: &Tensor) -> Result<Tensor> {
    let cnt = mask1d.sum(1)?.unsqueeze(1)?.clamp(1.0f64, 1e9f64)?; // (B, 1)
    let mask_u3 = mask1d.unsqueeze(2)?; // (B, P, 1)

    let sum_t = t.broadcast_mul(&mask_u3)?.sum(1)?;
    let mean = sum_t.broadcast_div(&cnt)?; // (B, D)

    let is_valid = mask_u3.broadcast_as(t.shape())?.to_dtype(DType::U8)?;
    let fill = (t.zeros_like()? - 1e4f64)?;
    let mx = is_valid.where_cond(t, &fill)?.max(1)?; // (B, D)

    let mean_unsq = mean.unsqueeze(1)?; // (B, 1, D)
    let diff = t
        .broadcast_sub(&mean_unsq)?
        .broadcast_mul(&mask_u3.broadcast_as(t.shape())?)?;
    let var = (&diff * &diff)?.sum(1)?.broadcast_div(&cnt)?;
    let std = (var + 1e-6f64)?.sqrt()?;

    let empty = mask1d.sum(1)?.eq(&mask1d.sum(1)?.zeros_like()?)?;
    let empty_u8 = empty
        .unsqueeze(1)?
        .broadcast_as(mean.shape())?
        .to_dtype(DType::U8)?;

    let mean = empty_u8.where_cond(&mean.zeros_like()?, &mean)?;
    let std = empty_u8.where_cond(&std.zeros_like()?, &std)?;
    let mx = empty_u8.where_cond(&mx.zeros_like()?, &mx)?;

    Tensor::cat(&[&mean, &mx, &std], 1) // (B, 3D)
}
