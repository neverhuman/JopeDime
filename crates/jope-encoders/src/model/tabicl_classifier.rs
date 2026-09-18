//! 100% Rust (candle) forward pass for the TabICL v2 binary classifier.
//!
//! Architecture recovered from the `tabicl-classifier-v2-20260212` checkpoint
//! (391 tensors; loaded at runtime from the converted `.safetensors`) and the
//! reference `tabicl` 2.1.1 source. Three stages:
//!
//! 1. **Column embedder** (`col_embedder`): per-feature-group distribution-aware
//!    embedding via a shared Set-Transformer (3 Induced-Self-Attention blocks).
//!    Feature grouping = "same" (circular shifts by 1,2,4), group size 3, target
//!    aware (one-hot y added to train rows), QASSMax-elementwise on the FIRST
//!    attention of each ISAB only, affine=False.
//! 2. **Row interactor** (`row_interactor`): per-row transformer (3 blocks) with
//!    RoPE (base 1e5, non-interleaved) over the feature axis, 4 learnable CLS
//!    tokens prepended; final block uses CLS-as-query over the full sequence,
//!    out-LayerNorm, then the 4 CLS outputs are concatenated -> 512-dim per row.
//! 3. **ICL predictor** (`icl_predictor`): 12-block transformer (d=512) with
//!    QASSMax-elementwise, train-context-only keys/values (every query attends
//!    only to the first `train_size` rows), one-hot y added to train rows, final
//!    LayerNorm, decoder 512->1024->GELU->10. Logits sliced to `num_classes`.
//!
//! All LayerNorm eps = 1e-5; all GELU = exact (erf). Attention scale =
//! 1/sqrt(head_dim); QASSMax scales the query BEFORE that division.

use super::tabicl_common::{attention, linear, Rope, Ssmax, Weights};
use super::ClassificationExpertFeatures;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor, D};
use std::path::Path;

const E: usize = 128; // embed_dim
const COL_NHEAD: usize = 8;
const COL_BLOCKS: usize = 3;
const COL_NUM_INDS: usize = 128;
const GROUP_SIZE: usize = 3;
const MAX_CLASSES: usize = 10;
const ROW_NHEAD: usize = 8;
const ROW_BLOCKS: usize = 3;
const NUM_CLS: usize = 4;
const ROW_ROPE_BASE: f64 = 100000.0;
const ICL_NHEAD: usize = 8;
const ICL_BLOCKS: usize = 12;
const LN_EPS: f64 = 1e-5;

/// LayerNorm over the last dim with affine weight/bias and eps 1e-5.
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + LN_EPS)?.sqrt()?;
    let xn = xc.broadcast_div(&denom)?;
    Ok(xn.broadcast_mul(w)?.broadcast_add(b)?)
}

/// Pre-norm Multihead-Attention block (norm1 -> attn -> residual -> norm2 -> ff -> residual).
struct MabWeights {
    norm1_w: Tensor,
    norm1_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
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
    fn load(w: &Weights, prefix: &str, nhead: usize, has_ssmax: bool) -> Result<Self> {
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
            norm1_b: w.g(&format!("{prefix}.norm1.bias"))?.clone(),
            norm2_w: w.g(&format!("{prefix}.norm2.weight"))?.clone(),
            norm2_b: w.g(&format!("{prefix}.norm2.bias"))?.clone(),
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
    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        nhead: usize,
        rope: Option<&Rope>,
    ) -> Result<Tensor> {
        let qn = layer_norm(q, &self.norm1_w, &self.norm1_b)?;
        let kn = layer_norm(k, &self.norm1_w, &self.norm1_b)?;
        let vn = layer_norm(v, &self.norm1_w, &self.norm1_b)?;
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
        let ffx = self.ff(&layer_norm(&x, &self.norm2_w, &self.norm2_b)?)?;
        Ok((x + ffx)?)
    }
}

/// The full TabICL classifier model (binary head extracted by Chimera).
pub(super) struct TabIclCls {
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
    row_out_ln_b: Tensor,
    rope: Rope,
    // icl
    icl_y_w: Tensor,
    icl_y_b: Tensor,
    icl_blocks: Vec<MabWeights>,
    icl_ln_w: Tensor,
    icl_ln_b: Tensor,
    dec0_w: Tensor,
    dec0_b: Tensor,
    dec2_w: Tensor,
    dec2_b: Tensor,
}

impl TabIclCls {
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
        Ok(Self {
            in_lin_w: w.g("col_embedder.in_linear.weight")?.clone(),
            in_lin_b: w.g("col_embedder.in_linear.bias")?.clone(),
            col_y_w: w.g("col_embedder.y_encoder.weight")?.clone(),
            col_y_b: w.g("col_embedder.y_encoder.bias")?.clone(),
            col_blocks,
            cls_tokens: w.g("row_interactor.cls_tokens")?.clone(),
            row_blocks,
            row_out_ln_w: w.g("row_interactor.out_ln.weight")?.clone(),
            row_out_ln_b: w.g("row_interactor.out_ln.bias")?.clone(),
            rope: Rope::new(E / ROW_NHEAD, ROW_ROPE_BASE),
            icl_y_w: w.g("icl_predictor.y_encoder.weight")?.clone(),
            icl_y_b: w.g("icl_predictor.y_encoder.bias")?.clone(),
            icl_blocks,
            icl_ln_w: w.g("icl_predictor.ln.weight")?.clone(),
            icl_ln_b: w.g("icl_predictor.ln.bias")?.clone(),
            dec0_w: w.g("icl_predictor.decoder.0.weight")?.clone(),
            dec0_b: w.g("icl_predictor.decoder.0.bias")?.clone(),
            dec2_w: w.g("icl_predictor.decoder.2.weight")?.clone(),
            dec2_b: w.g("icl_predictor.decoder.2.bias")?.clone(),
            w,
        })
    }

    pub(super) fn device(&self) -> &Device {
        self.w.device()
    }

    /// One-hot encode integer labels to `MAX_CLASSES` then linear-project.
    fn y_encode(&self, y: &[u8], wt: &Tensor, b: &Tensor) -> Result<Tensor> {
        let n = y.len();
        let mut oh = vec![0f32; n * MAX_CLASSES];
        for (i, &yi) in y.iter().enumerate() {
            let c = yi as usize;
            if c < MAX_CLASSES {
                oh[i * MAX_CLASSES + c] = 1.0;
            }
        }
        let oh = Tensor::from_vec(oh, (n, MAX_CLASSES), self.device())?;
        linear(&oh, wt, Some(b))
    }

    /// Column embedder: X (T,H) -> per-group embeddings (T, H, E).
    fn column_embed(&self, x: &Tensor, y_train: &[u8], train_size: usize) -> Result<Tensor> {
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
                                                                            // target-aware: add y embedding to train rows of every group
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
        let cls_out = layer_norm(&cls_out, &self.row_out_ln_w, &self.row_out_ln_b)?;
        // flatten last two dims -> (T, NUM_CLS*E = 512)
        Ok(cls_out.reshape((t, NUM_CLS * E))?)
    }

    /// ICL predictor: row reps (T,512) + train labels -> logits (test_size, MAX_CLASSES).
    fn icl_predict(&self, reps: &Tensor, y_train: &[u8], train_size: usize) -> Result<Tensor> {
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
        let src = layer_norm(&src, &self.icl_ln_w, &self.icl_ln_b)?;
        // decoder
        let h = linear(&src, &self.dec0_w, Some(&self.dec0_b))?.gelu_erf()?;
        let logits = linear(&h, &self.dec2_w, Some(&self.dec2_b))?; // (T, MAX_CLASSES)
                                                                    // test rows only
        Ok(logits.narrow(0, train_size, t - train_size)?.contiguous()?)
    }

    pub(super) fn classification_fusion_features_single(
        &self,
        ctx_x: &[f32],
        ctx_y: &[u8],
        qx: &[f32],
        n_features: usize,
    ) -> Result<ClassificationExpertFeatures> {
        let n = ctx_y.len();
        let m = qx.len() / n_features;
        if ctx_x.len() != n * n_features || qx.len() != m * n_features {
            bail!("bad TabICL classification feature matrix length");
        }
        if n == 0 {
            bail!("TabICL classification features require at least one context row");
        }
        let xz = zscore_concat(ctx_x, qx, n, m, n_features);
        let x = Tensor::from_vec(xz, (n + m, n_features), self.device())?;
        let (logits, hidden) = self.forward_member_features(&x, ctx_y, n)?;
        let logits = logits.narrow(1, 0, 2)?.to_dtype(DType::F32)?;
        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let hidden_vec = hidden
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut conf = Vec::with_capacity(m * 3);
        for pair in logits_vec.chunks_exact(2) {
            let p1 = softmax_class1(pair[0], pair[1]);
            let p0 = 1.0 - p1;
            let entropy = -(p0.max(1e-6).ln() * p0 + p1.max(1e-6).ln() * p1);
            conf.push(p0.max(p1));
            conf.push(entropy);
            conf.push(p1 - p0);
        }
        Ok(ClassificationExpertFeatures {
            n_rows: m,
            hidden_dim: NUM_CLS * E,
            logit_dim: 2,
            conf_dim: 3,
            logits: logits_vec,
            hidden: hidden_vec,
            conf,
        })
    }

    /// Full forward for one Chimera expert member. `x`: (T, H) row-major
    /// z-scored features (train rows first), `y_train`: train labels. Returns
    /// test logits and the row-interactor hidden rows used by Python's hook.
    fn forward_member_features(
        &self,
        x: &Tensor,
        y_train: &[u8],
        train_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let total = x.dims()[0];
        let emb = self.column_embed(x, y_train, train_size)?;
        let reps = self.row_interact(&emb)?;
        let hidden = reps
            .narrow(0, train_size, total - train_size)?
            .contiguous()?;
        let logits = self.icl_predict(&reps, y_train, train_size)?;
        Ok((logits, hidden))
    }
}

fn zscore_concat(ctx_x: &[f32], qx: &[f32], n: usize, m: usize, p: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; p];
    for row in ctx_x.chunks_exact(p) {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += v;
        }
    }
    for v in &mut mean {
        *v /= n.max(1) as f32;
    }
    let mut std = vec![0.0f32; p];
    for row in ctx_x.chunks_exact(p) {
        for (j, &v) in row.iter().enumerate() {
            let d = v - mean[j];
            std[j] += d * d;
        }
    }
    for v in &mut std {
        *v = (*v / n.max(1) as f32).sqrt() + 1e-6;
    }
    let mut out = Vec::with_capacity((n + m) * p);
    for row in ctx_x.chunks_exact(p).chain(qx.chunks_exact(p)) {
        for j in 0..p {
            out.push(((row[j] - mean[j]) / std[j]).clamp(-100.0, 100.0));
        }
    }
    out
}

fn softmax_class1(a: f32, b: f32) -> f32 {
    let mx = a.max(b);
    let ea = (a - mx).exp();
    let eb = (b - mx).exp();
    eb / (ea + eb).max(1e-12)
}
