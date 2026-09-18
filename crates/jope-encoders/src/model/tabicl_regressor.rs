//! 100% Rust (candle) forward pass for the TabICL v2 tabular **regressor**.
//!
//! Architecture recovered from `tabicl-regressor-v2-20260212.safetensors`
//! (347 tensors) and the reference `tabicl` 2.1.1 source. It is the SAME three
//! stages as the binary classifier port (column embedder -> row interactor ->
//! ICL predictor); this file only differs in three ways that the checkpoint
//! forces:
//!
//! 1. **Bias-free LayerNorm** (`bias_free_ln=True`): the regressor checkpoint has
//!    NO `*.norm*.bias` / `ln.bias` / `out_ln.bias` tensors (44 fewer than the
//!    classifier). Every LayerNorm is `weight`-only: `(x-mean)/sqrt(var+eps)*w`.
//! 2. **Continuous target encoders**: `col_embedder.y_encoder` and
//!    `icl_predictor.y_encoder` are `nn.Linear(1, E)` (weight `(E,1)`), applied to
//!    the standardized continuous target, instead of the classifier's
//!    one-hot -> linear. `Ry[i,:] = y[i] * W.flatten() + b`.
//! 3. **Quantile head**: the ICL decoder `512 -> 1024 -> GELU -> num_quantiles`
//!    emits `num_quantiles=999` raw quantiles per test row (not `MAX_CLASSES`
//!    logits). For `output_type="mean"` (the sklearn default) the prediction is
//!    the mean over the 999 quantiles; because `QuantileToDistribution` only
//!    monotonically SORTS the quantiles (`crossing_method="sort"`), and sorting
//!    preserves the mean, the tail/spline machinery is irrelevant and the
//!    per-row prediction is exactly `raw_quantiles.mean(dim=-1)`.
//!
//! All LayerNorm eps = 1e-5; all GELU = exact (erf). Attention scale =
//! 1/sqrt(head_dim); QASSMax scales the query BEFORE that division. The
//! column/row/ICL transformer blocks, QASSMax and RoPE are bit-for-bit the same
//! as the classifier port.

use super::tabicl_common::{linear, Rope, Weights};
use super::RegressionExpertFeatures;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

const E: usize = 128; // embed_dim
const COL_NHEAD: usize = 8;
const COL_BLOCKS: usize = 3;
const COL_NUM_INDS: usize = 128;
const GROUP_SIZE: usize = 3;
const ROW_NHEAD: usize = 8;
const ROW_BLOCKS: usize = 3;
const NUM_CLS: usize = 4;
const ROW_ROPE_BASE: f64 = 100000.0;
const ICL_NHEAD: usize = 8;
const ICL_BLOCKS: usize = 12;
const LN_EPS: f64 = 1e-5;

mod attention_weights;
mod pooling;

use attention_weights::{layer_norm, MabWeights};
use pooling::pool_mean_std_p10_p90;

/// The full TabICL regressor model (quantile head extracted by the runner).
pub(super) struct TabIclReg {
    w: Weights,
    // col embedder
    in_lin_w: Tensor,
    in_lin_b: Tensor,
    col_y_w: Tensor,
    col_y_b: Tensor,
    col_blocks: Vec<(Tensor, MabWeights, MabWeights)>, // (ind_vectors, attn1, attn2)
    // row interactor
    cls_tokens: Tensor,
    row_blocks: Vec<MabWeights>,
    row_out_ln_w: Tensor,
    rope: Rope,
    // icl
    icl_y_w: Tensor,
    icl_y_b: Tensor,
    icl_blocks: Vec<MabWeights>,
    icl_ln_w: Tensor,
    dec0_w: Tensor,
    dec0_b: Tensor,
    dec2_w: Tensor,
    dec2_b: Tensor,
    num_quantiles: usize,
}

impl TabIclReg {
    pub(super) fn load(ckpt_path: &Path, device: Device) -> Result<Self> {
        let w = Weights::load(ckpt_path, device)?;
        let mut col_blocks = Vec::new();
        for i in 0..COL_BLOCKS {
            let p = format!("col_embedder.tf_col.blocks.{i}");
            let ind = w.g(&format!("{p}.ind_vectors"))?.clone();
            let a1 = MabWeights::load(&w, &format!("{p}.multihead_attn1"), COL_NHEAD, true)?;
            let a2 = MabWeights::load(&w, &format!("{p}.multihead_attn2"), COL_NHEAD, false)?;
            col_blocks.push((ind, a1, a2));
        }
        let mut row_blocks = Vec::new();
        for i in 0..ROW_BLOCKS {
            row_blocks.push(MabWeights::load(
                &w,
                &format!("row_interactor.tf_row.blocks.{i}"),
                ROW_NHEAD,
                false,
            )?);
        }
        let mut icl_blocks = Vec::new();
        for i in 0..ICL_BLOCKS {
            icl_blocks.push(MabWeights::load(
                &w,
                &format!("icl_predictor.tf_icl.blocks.{i}"),
                ICL_NHEAD,
                true,
            )?);
        }
        let dec2_w = w.g("icl_predictor.decoder.2.weight")?.clone();
        let num_quantiles = dec2_w.dim(0)?;
        Ok(Self {
            in_lin_w: w.g("col_embedder.in_linear.weight")?.clone(),
            in_lin_b: w.g("col_embedder.in_linear.bias")?.clone(),
            col_y_w: w.g("col_embedder.y_encoder.weight")?.clone(),
            col_y_b: w.g("col_embedder.y_encoder.bias")?.clone(),
            col_blocks,
            cls_tokens: w.g("row_interactor.cls_tokens")?.clone(),
            row_blocks,
            row_out_ln_w: w.g("row_interactor.out_ln.weight")?.clone(),
            rope: Rope::new(E / ROW_NHEAD, ROW_ROPE_BASE),
            icl_y_w: w.g("icl_predictor.y_encoder.weight")?.clone(),
            icl_y_b: w.g("icl_predictor.y_encoder.bias")?.clone(),
            icl_blocks,
            icl_ln_w: w.g("icl_predictor.ln.weight")?.clone(),
            dec0_w: w.g("icl_predictor.decoder.0.weight")?.clone(),
            dec0_b: w.g("icl_predictor.decoder.0.bias")?.clone(),
            dec2_w,
            dec2_b: w.g("icl_predictor.decoder.2.bias")?.clone(),
            num_quantiles,
            w,
        })
    }

    pub(super) fn device(&self) -> &Device {
        self.w.device()
    }

    /// Continuous target encoder `nn.Linear(1, D)`: `y[:,None] @ W.T + b`.
    /// `y` is the standardized continuous target; `wt` is `(D, 1)`.
    fn y_encode(&self, y: &[f32], wt: &Tensor, b: &Tensor) -> Result<Tensor> {
        let n = y.len();
        let y1 = Tensor::from_vec(y.to_vec(), (n, 1), self.device())?;
        linear(&y1, wt, Some(b)) // (n, D)
    }

    /// Column embedder: X (T,H) -> per-group embeddings (T, H, E).
    fn column_embed(&self, x: &Tensor, y_train: &[f32], train_size: usize) -> Result<Tensor> {
        let (t, h) = x.dims2()?;
        // Feature grouping "same": group g = cols [(g+1)%H,(g+2)%H,(g+4)%H].
        // Build features (H, T, GROUP_SIZE).
        let xv: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?; // row-major (t,h)
        let shifts = [1usize, 2, 4];
        let mut feat = vec![0f32; h * t * GROUP_SIZE];
        for g in 0..h {
            for ti in 0..t {
                for (s, &sh) in shifts.iter().enumerate() {
                    let col = (g + sh) % h;
                    feat[(g * t + ti) * GROUP_SIZE + s] = xv[ti * h + col];
                }
            }
        }
        let feat = Tensor::from_vec(feat, (h, t, GROUP_SIZE), self.device())?;
        // in_linear (GROUP_SIZE -> E)
        let mut src = linear(&feat, &self.in_lin_w, Some(&self.in_lin_b))?; // (H,T,E)
                                                                            // target-aware: add continuous y embedding to train rows of every group
        let y_emb = self.y_encode(y_train, &self.col_y_w, &self.col_y_b)?; // (train_size, E)
        let y_emb = y_emb.unsqueeze(0)?; // (1, train_size, E)
        let src_train = src.narrow(1, 0, train_size)?.broadcast_add(&y_emb)?;
        let src_rest = src.narrow(1, train_size, t - train_size)?;
        src = Tensor::cat(&[&src_train, &src_rest], 1)?;
        // 3 ISAB blocks
        for (ind, a1, a2) in &self.col_blocks {
            let ind_v = ind
                .unsqueeze(0)?
                .broadcast_as((h, COL_NUM_INDS, E))?
                .contiguous()?;
            let src_train = src.narrow(1, 0, train_size)?.contiguous()?;
            let hidden = a1.forward(&ind_v, &src_train, &src_train, COL_NHEAD, None)?;
            src = a2.forward(&src, &hidden, &hidden, COL_NHEAD, None)?;
        }
        // (H, T, E) -> (T, H, E)
        Ok(src.transpose(0, 1)?.contiguous()?)
    }

    /// Row interactor: per-group embeddings (T,H,E) -> row reps (T, 512).
    fn row_interact(&self, emb: &Tensor) -> Result<Tensor> {
        let (t, _h, _e) = emb.dims3()?;
        // prepend CLS tokens -> (T, NUM_CLS+H, E)
        let cls = self
            .cls_tokens
            .unsqueeze(0)?
            .broadcast_as((t, NUM_CLS, E))?
            .contiguous()?;
        let mut seq = Tensor::cat(&[&cls, emb], 1)?; // (T, NUM_CLS+H, E)
                                                     // blocks[:-1]: full self-attention
        for blk in &self.row_blocks[..ROW_BLOCKS - 1] {
            seq = blk.forward(&seq, &seq, &seq, ROW_NHEAD, Some(&self.rope))?;
        }
        // last block: q = CLS tokens, k=v=full seq
        let last = &self.row_blocks[ROW_BLOCKS - 1];
        let q = seq.narrow(1, 0, NUM_CLS)?.contiguous()?;
        let cls_out = last.forward(&q, &seq, &seq, ROW_NHEAD, Some(&self.rope))?; // (T, NUM_CLS, E)
        let cls_out = layer_norm(&cls_out, &self.row_out_ln_w)?;
        // flatten last two dims -> (T, NUM_CLS*E = 512)
        Ok(cls_out.reshape((t, NUM_CLS * E))?)
    }

    /// ICL predictor: row reps (T,512) + train targets -> raw quantiles
    /// (test_size, num_quantiles).
    fn icl_predict(&self, reps: &Tensor, y_train: &[f32], train_size: usize) -> Result<Tensor> {
        let (t, _d) = reps.dims2()?;
        let y_emb = self.y_encode(y_train, &self.icl_y_w, &self.icl_y_b)?; // (train_size, 512)
        let r_train = reps.narrow(0, 0, train_size)?.add(&y_emb)?;
        let r_rest = reps.narrow(0, train_size, t - train_size)?;
        let mut src = Tensor::cat(&[&r_train, &r_rest], 0)?; // (T, 512)
                                                             // 12 blocks, k=v = src[:train_size]
        for blk in &self.icl_blocks {
            let ctx = src.narrow(0, 0, train_size)?.contiguous()?;
            src = blk.forward(&src, &ctx, &ctx, ICL_NHEAD, None)?;
        }
        let src = layer_norm(&src, &self.icl_ln_w)?;
        // decoder
        let h = linear(&src, &self.dec0_w, Some(&self.dec0_b))?.gelu_erf()?;
        let q = linear(&h, &self.dec2_w, Some(&self.dec2_b))?; // (T, num_quantiles)
                                                               // test rows only
        Ok(q.narrow(0, train_size, t - train_size)?.contiguous()?)
    }

    /// Full forward for one ensemble member. `x`: (T, H) row-major preprocessed
    /// features (train rows first), `y_train`: standardized train targets
    /// (length train_size). Returns raw quantiles (test_size, num_quantiles).
    fn forward_member_features(
        &self,
        x: &Tensor,
        y_train: &[f32],
        train_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let emb = self.column_embed(x, y_train, train_size)?;
        let reps = self.row_interact(&emb)?;
        let q = self.icl_predict(&reps, y_train, train_size)?;
        let test_reps = reps
            .narrow(0, train_size, reps.dims2()?.0 - train_size)?
            .contiguous()?;
        Ok((test_reps, q))
    }

    pub(super) fn regression_fusion_features_single(
        &self,
        ctx_x: &[f32],
        ctx_y: &[f32],
        qx: &[f32],
        n_features: usize,
    ) -> Result<RegressionExpertFeatures> {
        let n = ctx_y.len();
        let m = qx.len() / n_features;
        if n == 0 {
            bail!("TabICL regression requires at least one context row");
        }
        if ctx_x.len() != n * n_features || qx.len() != m * n_features {
            bail!("bad TabICL regression feature matrix length");
        }
        let xz = zscore_train_query(ctx_x, qx, n, m, n_features);
        let (target_mean, target_std) = target_mean_std(ctx_y);
        let y_scaled: Vec<f32> = ctx_y
            .iter()
            .map(|&y| (y - target_mean) / target_std)
            .collect();
        let x = Tensor::from_vec(xz, (n + m, n_features), self.device())?;
        let (hidden, quant_raw) = self.forward_member_features(&x, &y_scaled, n)?;
        let hidden = hidden
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let qv = quant_raw
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut mean = Vec::with_capacity(m);
        let mut quantiles = Vec::with_capacity(m * TABICL_QUANTILE_PICKS.len());
        let mut conf = Vec::with_capacity(m * 3);
        for row in 0..m {
            let start = row * self.num_quantiles;
            let mut q = qv[start..start + self.num_quantiles].to_vec();
            q.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let quant_y: Vec<f32> = q.iter().map(|&v| v * target_std + target_mean).collect();
            let row_mean = quant_y.iter().sum::<f32>() / quant_y.len().max(1) as f32;
            mean.push(row_mean);
            for &idx in &TABICL_QUANTILE_PICKS {
                quantiles.push(quant_y[idx]);
            }
            conf.push(quant_y[899] - quant_y[99]);
            conf.push(0.0);
            conf.push(row_mean);
        }
        Ok(RegressionExpertFeatures {
            n_rows: m,
            hidden_dim: NUM_CLS * E,
            quantile_dim: TABICL_QUANTILE_PICKS.len(),
            conf_dim: 3,
            mean,
            hidden,
            quantiles,
            conf,
        })
    }

    /// Pooled 2048-d dataset foundation vector (V6 Lane A / jope foundation recipe).
    ///
    /// Runs the same TabICL forward as `regression_fusion_features_single`, but
    /// instead of the fusion features returns the query-row `row_interactor`
    /// hidden states (M, 512) pooled over rows into
    /// `concat[mean, std, p10, p90]` = 2048. `ctx_x`/`qx` are row-major
    /// min-max-normalized-but-NOT-yet-zscored features (the z-score by train
    /// stats happens here); `ctx_y` are the raw context targets (standardized
    /// here). Matches jope's Python `TabICLReg` foundation encoder.
    pub(super) fn foundation_2048(
        &self,
        ctx_x: &[f32],
        ctx_y: &[f32],
        qx: &[f32],
        n_features: usize,
    ) -> Result<Vec<f32>> {
        let n = ctx_y.len();
        let m = qx.len() / n_features;
        if n == 0 {
            bail!("TabICL foundation needs >=1 context row");
        }
        if ctx_x.len() != n * n_features || qx.len() != m * n_features {
            bail!("bad TabICL foundation matrix length");
        }
        let xz = zscore_train_query(ctx_x, qx, n, m, n_features);
        let (tm, ts) = target_mean_std(ctx_y);
        let y_scaled: Vec<f32> = ctx_y.iter().map(|&y| (y - tm) / ts).collect();
        let x = Tensor::from_vec(xz, (n + m, n_features), self.device())?;
        let (hidden, _q) = self.forward_member_features(&x, &y_scaled, n)?; // (M, 512)
        let hid = hidden.to_dtype(DType::F32)?.to_vec2::<f32>()?; // Vec<Vec<f32>> (M,512)
        Ok(pool_mean_std_p10_p90(&hid))
    }
}

/// Public wrapper exposing the pooled 2048-d foundation vector from a loaded
/// TabICL regressor checkpoint (the internals stay `pub(super)`).
pub struct TabIclFoundation {
    inner: TabIclReg,
}

impl TabIclFoundation {
    pub fn load(ckpt_path: &Path, device: &Device) -> Result<Self> {
        Ok(Self {
            inner: TabIclReg::load(ckpt_path, device.clone())?,
        })
    }

    /// Produce the pooled 2048-d foundation vector for one dataset. `ctx_x` /
    /// `query_x` are per-row feature vectors (min-max-normalized, NOT z-scored;
    /// the z-score by train stats happens inside); `ctx_y` are the raw context
    /// targets. `n_features` is inferred from `ctx_x[0].len()`.
    pub fn foundation_2048(
        &self,
        ctx_x: &[Vec<f32>],
        ctx_y: &[f32],
        query_x: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        let n_features = ctx_x.first().map_or(0, |r| r.len());
        if n_features == 0 {
            bail!("TabICL foundation needs >=1 feature (empty ctx_x)");
        }
        let mut cx = Vec::with_capacity(ctx_x.len() * n_features);
        for row in ctx_x {
            if row.len() != n_features {
                bail!("ragged ctx_x row: expected {n_features} features");
            }
            cx.extend_from_slice(row);
        }
        let mut qx = Vec::with_capacity(query_x.len() * n_features);
        for row in query_x {
            if row.len() != n_features {
                bail!("ragged query_x row: expected {n_features} features");
            }
            qx.extend_from_slice(row);
        }
        self.inner.foundation_2048(&cx, ctx_y, &qx, n_features)
    }
}

const TABICL_QUANTILE_PICKS: [usize; 19] = [
    0, 55, 110, 166, 221, 277, 332, 388, 443, 499, 554, 609, 665, 720, 776, 831, 887, 942, 998,
];

fn target_mean_std(y: &[f32]) -> (f32, f32) {
    let n = y.len().max(1) as f32;
    let mean = y.iter().sum::<f32>() / n;
    let var = y
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    (mean, var.sqrt() + 1e-8)
}

fn zscore_train_query(ctx_x: &[f32], qx: &[f32], n: usize, m: usize, p: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; p];
    for row in ctx_x.chunks_exact(p) {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += v;
        }
    }
    for v in &mut mean {
        *v /= n.max(1) as f32;
    }
    let mut scale = vec![0.0f32; p];
    for row in ctx_x.chunks_exact(p) {
        for (j, &v) in row.iter().enumerate() {
            let d = v - mean[j];
            scale[j] += d * d;
        }
    }
    for v in &mut scale {
        *v = (*v / n.max(1) as f32).sqrt() + 1e-6;
    }
    let mut out = Vec::with_capacity((n + m) * p);
    for row in ctx_x.chunks_exact(p) {
        for (j, &v) in row.iter().enumerate() {
            out.push(((v - mean[j]) / scale[j]).clamp(-100.0, 100.0));
        }
    }
    for row in qx.chunks_exact(p) {
        for (j, &v) in row.iter().enumerate() {
            out.push(((v - mean[j]) / scale[j]).clamp(-100.0, 100.0));
        }
    }
    out
}
