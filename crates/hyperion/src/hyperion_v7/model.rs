//! The v7 dual-axis matrix encoder (vendored from the veox reference, re-geometried for v7:
//! dim 512 / heads 8 / 12 blocks / 6 bands) plus the per-cell confidence head reconstructed from
//! the training reference (`MQOTRCheat.py::MatrixGhostUnified.forward`).

use candle_core::{DType, Result, Tensor};

use super::mha::{mha_forward, MhaWeights};
use super::ops::{layer_norm, linear3, relu, LinearWeights};
use super::pooling::{global_stats, MpsWeights};

/// Band-gated embedding head: shared block (128 dims for v7) + one active band block (64 dims),
/// inactive band slots zeroed → total `emb_dim` (512).
pub struct HgeWeights {
    pub shared_0: LinearWeights,
    pub shared_2: LinearWeights,
    pub bands_0: Vec<LinearWeights>,
    pub bands_2: Vec<LinearWeights>,
}

/// Band for a patch scale: the first `ws_list` window ≥ p (v7 ws_list \[4,7,11,15,19,23\] →
/// scales 3,5,7,9 → bands 0,1,1,2). Isolated so the rule is a one-line change if the v7 training
/// recipe surfaces a different mapping.
pub fn band_for(p_val: usize, ws_list: &[usize]) -> usize {
    ws_list
        .iter()
        .position(|&w| p_val <= w)
        .unwrap_or(ws_list.len().saturating_sub(1))
}

impl HgeWeights {
    pub fn forward(&self, g: &Tensor, band: usize) -> Result<Tensor> {
        let shared = relu(&linear3(g, &self.shared_0)?)?;
        let shared = linear3(&shared, &self.shared_2)?; // (B, 128)

        let idx = band.min(self.bands_0.len().saturating_sub(1));
        let band_out = relu(&linear3(g, &self.bands_0[idx])?)?;
        let band_out = linear3(&band_out, &self.bands_2[idx])?; // (B, 64)
        let band_dim = band_out.dim(1)?;

        let mut outputs = Vec::with_capacity(1 + self.bands_0.len());
        outputs.push(shared);
        for i in 0..self.bands_0.len() {
            if i == idx {
                outputs.push(band_out.clone());
            } else {
                outputs.push(g.zeros_like()?.narrow(1, 0, band_dim)?);
            }
        }
        Tensor::cat(&outputs, 1) // 128 + 6*64 = 512
    }
}

pub struct BlockWeights {
    pub r2c: MhaWeights,
    pub c2r: MhaWeights,
    pub lnr_weight: Tensor,
    pub lnr_bias: Tensor,
    pub lnc_weight: Tensor,
    pub lnc_bias: Tensor,
    pub ffr_0: LinearWeights,
    pub ffr_2: LinearWeights,
    pub ffc_0: LinearWeights,
    pub ffc_2: LinearWeights,
}

pub struct MguWeights {
    pub val_0: LinearWeights,
    pub val_2: LinearWeights,
    pub scale_emb: Tensor, // (max_patch - min_patch + 1, dim)
    pub row_pool: MpsWeights,
    pub col_pool: MpsWeights,
    pub blocks: Vec<BlockWeights>,
    pub emb_head: HgeWeights,
    // Confidence head (reconstructed from the training reference; weights present in v7).
    pub cell_mlp_0: LinearWeights, // (2*dim, 5*dim)
    pub cell_mlp_2: LinearWeights, // (dim, 2*dim)
    pub h_real: LinearWeights,     // (1, dim)
    // Geometry.
    pub dim: usize,
    pub heads: usize,
    pub max_patch: usize,
    pub ws_list: Vec<usize>,
}

impl MguWeights {
    /// Shared trunk through the dual-axis blocks. Returns the (scale-conditioned) cell tensor and
    /// the post-block row/col states + validity masks.
    fn rowcol(
        &self,
        x: &Tensor,    // (B, P, P)
        m: &Tensor,    // (B, P, P)
        p_id: &Tensor, // (B,) u32 scale ids
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let x_un = x.unsqueeze(3)?;
        let m_un = m.unsqueeze(3)?;
        let cell_in = Tensor::cat(&[&x_un, &m_un], 3)?; // (B, P, P, 2)

        let cell_hid = relu(&linear3(&cell_in, &self.val_0)?)?;
        let mut cell = linear3(&cell_hid, &self.val_2)?; // (B, P, P, dim)

        let scale_vec = self.scale_emb.index_select(p_id, 0)?; // (B, dim)
        let scale_unsq = scale_vec.unsqueeze(1)?.unsqueeze(2)?; // (B, 1, 1, dim)
        cell = cell.broadcast_add(&scale_unsq)?;

        let mut row_h = self
            .row_pool
            .forward(&cell, m, 2)?
            .broadcast_add(&scale_vec.unsqueeze(1)?)?;
        let mut col_h = self
            .col_pool
            .forward(&cell, m, 1)?
            .broadcast_add(&scale_vec.unsqueeze(1)?)?;

        let r_valid = (m.sum(2)?)
            .gt(&m.sum(2)?.zeros_like()?)?
            .to_dtype(DType::F32)?;
        let c_valid = (m.sum(1)?)
            .gt(&m.sum(1)?.zeros_like()?)?
            .to_dtype(DType::F32)?;

        for blk in &self.blocks {
            let r_att = mha_forward(&row_h, &col_h, &c_valid, &blk.r2c, self.dim, self.heads)?;
            let c_att = mha_forward(&col_h, &row_h, &r_valid, &blk.c2r, self.dim, self.heads)?;

            let r_ff_hid = relu(&linear3(&row_h, &blk.ffr_0)?)?;
            let r_ff = linear3(&r_ff_hid, &blk.ffr_2)?;
            let c_ff_hid = relu(&linear3(&col_h, &blk.ffc_0)?)?;
            let c_ff = linear3(&c_ff_hid, &blk.ffc_2)?;

            // row = LN(row + attention + FF(row)) — the reference's residual arrangement.
            row_h = layer_norm(
                &row_h.broadcast_add(&r_att)?.broadcast_add(&r_ff)?,
                &blk.lnr_weight,
                &blk.lnr_bias,
                1e-5,
            )?;
            col_h = layer_norm(
                &col_h.broadcast_add(&c_att)?.broadcast_add(&c_ff)?,
                &blk.lnc_weight,
                &blk.lnc_bias,
                1e-5,
            )?;

            let r_v_unsq = r_valid.unsqueeze(2)?;
            let c_v_unsq = c_valid.unsqueeze(2)?;
            row_h = row_h.broadcast_mul(&r_v_unsq)?;
            col_h = col_h.broadcast_mul(&c_v_unsq)?;
        }

        Ok((cell, row_h, col_h, r_valid, c_valid))
    }

    /// Global-stats tail → band-gated embedding (B, emb_dim=512).
    fn embed_head(
        &self,
        row_h: &Tensor,
        col_h: &Tensor,
        r_valid: &Tensor,
        c_valid: &Tensor,
        p_val: usize,
    ) -> Result<Tensor> {
        let r_stats = global_stats(row_h, r_valid)?;
        let c_stats = global_stats(col_h, c_valid)?;
        let p_scale_v = p_val as f64 / self.max_patch as f64;
        let p_scale = (r_stats.zeros_like()?.narrow(1, 0, 1)? + p_scale_v)?;
        let g = Tensor::cat(&[&r_stats, &c_stats, &p_scale], 1)?; // (B, 6*dim + 1 = 3073)
        self.emb_head.forward(&g, band_for(p_val, &self.ws_list))
    }

    /// Embedding-only forward: shared trunk plus the band-gated embedding head.
    pub fn forward_embed(
        &self,
        x: &Tensor,
        m: &Tensor,
        p_id: &Tensor,
        p_val: usize,
    ) -> Result<Tensor> {
        let (_, row_h, col_h, r_valid, c_valid) = self.rowcol(x, m, p_id)?;
        self.embed_head(&row_h, &col_h, &r_valid, &c_valid, p_val)
    }

    /// Full forward: (B, emb_dim) band-gated embedding + (B, P, P) per-cell confidence LOGITS.
    /// Head per the training reference: `cell_h = ReLU(cell_mlp(cat([cell, rb, cb, |rb−cb|,
    /// rb·cb])))` with the TRAILING ReLU, then `logits = h_real(cell_h)`; sigmoid is applied by the
    /// caller on AVERAGED logits.
    pub fn forward_full(
        &self,
        x: &Tensor,
        m: &Tensor,
        p_id: &Tensor,
        p_val: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (cell, row_h, col_h, r_valid, c_valid) = self.rowcol(x, m, p_id)?;
        let (b, p, _, d) = cell.dims4()?;

        let rb = row_h
            .unsqueeze(2)?
            .broadcast_as((b, p, p, d))?
            .contiguous()?;
        let cb = col_h
            .unsqueeze(1)?
            .broadcast_as((b, p, p, d))?
            .contiguous()?;
        let diff = (&rb - &cb)?.abs()?;
        let prod = (&rb * &cb)?;
        let cat = Tensor::cat(&[&cell, &rb, &cb, &diff, &prod], 3)?; // (B, P, P, 5*dim)

        let h = relu(&linear3(&cat, &self.cell_mlp_0)?)?; // (B, P, P, 2*dim)
        let h = relu(&linear3(&h, &self.cell_mlp_2)?)?; // (B, P, P, dim) — trailing ReLU
        let logits = linear3(&h, &self.h_real)?.squeeze(3)?; // (B, P, P)

        let emb = self.embed_head(&row_h, &col_h, &r_valid, &c_valid, p_val)?;
        Ok((emb, logits))
    }
}
