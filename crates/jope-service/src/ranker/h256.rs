//! 100% Rust `.safetensors` loader and end-to-end inference engine for the H256
//! Pre-Condense Residual Flywheel & Top-8 Hybrid Reranker models.
//!
//! Supports zero-Python/zero-C++ inference directly from `.safetensors` for:
//! - `H256FoldScaler` & `H256PreCondenseBridge` (Stage-1 & Stage-2 92-token Pre-Condense
//!   context encoder, 4-head `SyncFreeCrossAttention`, bilinear context-program interaction,
//!   `delta_prior`, and Stage-2 causal `2+2` trajectory residual `delta_traj`).
//! - `H256CriticModels` (`64D` outcome-free candidate feature builder, Decomposed Hurdle /
//!   SOS & Self-Play linear critics, `h_cost`, and `run_2plus2_active_reveal_step`).
//! - `FlatGbdtModel`, `H256Top8NeuralReranker`, and `H256ProductionBundle` (`616D` enriched
//!   Top-8 feature builder, 4-head slot self-attention neural reranker, flat-tensor GBDT
//!   ranker/classifier trees, and `H256_STATIC_0REV_TOP8_SOFT_VETO` /
//!   `H256_ACTIVE_2PLUS2_UNIFIED` selection).

use anyhow::{ensure, Context, Result};
use safetensors::{Dtype, SafeTensors};

pub const DOPE_DIM: usize = 629;
pub const HIDDEN_DIM: usize = 256;
pub const COL_SLOTS: usize = 64;
pub const COL_RAW_DIM: usize = 17;
pub const COL_ENC_DIM: usize = 18;
pub const LOG_NORM_DIM: usize = 11;
pub const TOKEN_DIM: usize = 32;
pub const PRE_CONDENSE_TOKENS: usize = 92;
pub const TOPK_SLICE: usize = 8;
pub const TOP8_BASE_DIM: usize = 568;
pub const TOP8_MID_DIM: usize = 580;
pub const TOP8_ENRICHED_DIM: usize = 616;

pub struct SafetensorsMap<'a> {
    st: SafeTensors<'a>,
}

impl<'a> SafetensorsMap<'a> {
    pub fn deserialize(bytes: &'a [u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).context("parse H256 safetensors bundle")?;
        Ok(Self { st })
    }

    pub fn f32_vec(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
        let view = self
            .st
            .tensor(name)
            .with_context(|| format!("missing F32 tensor {name}"))?;
        ensure!(
            view.dtype() == Dtype::F32 && view.shape() == expected_shape,
            "tensor {name} expected F32 {:?}, got {:?} {:?}",
            expected_shape,
            view.dtype(),
            view.shape()
        );
        let values: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite F32 values in {name}"
        );
        Ok(values)
    }

    pub fn f64_vec(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<f64>> {
        let view = self
            .st
            .tensor(name)
            .with_context(|| format!("missing F64 tensor {name}"))?;
        ensure!(
            view.dtype() == Dtype::F64 && view.shape() == expected_shape,
            "tensor {name} expected F64 {:?}, got {:?} {:?}",
            expected_shape,
            view.dtype(),
            view.shape()
        );
        let values: Vec<f64> = view
            .data()
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes")))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite F64 values in {name}"
        );
        Ok(values)
    }

    pub fn i32_vec(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<i32>> {
        let view = self
            .st
            .tensor(name)
            .with_context(|| format!("missing I32 tensor {name}"))?;
        ensure!(
            view.dtype() == Dtype::I32 && view.shape() == expected_shape,
            "tensor {name} expected I32 {:?}, got {:?} {:?}",
            expected_shape,
            view.dtype(),
            view.shape()
        );
        Ok(view
            .data()
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect())
    }

    pub fn i64_vec(&self, name: &str, expected_shape: &[usize]) -> Result<Vec<i64>> {
        let view = self
            .st
            .tensor(name)
            .with_context(|| format!("missing I64 tensor {name}"))?;
        ensure!(
            view.dtype() == Dtype::I64 && view.shape() == expected_shape,
            "tensor {name} expected I64 {:?}, got {:?} {:?}",
            expected_shape,
            view.dtype(),
            view.shape()
        );
        Ok(view
            .data()
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
            .collect())
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .st
            .tensor(name)
            .with_context(|| format!("missing tensor {name}"))?
            .shape()
            .to_vec())
    }
}

#[derive(Clone, Debug)]
pub struct LinearLayer {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl LinearLayer {
    pub fn from_safetensors(
        st: &SafetensorsMap<'_>,
        name: &str,
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let weight = st.f32_vec(&format!("{name}.weight"), &[out_dim, in_dim])?;
        let bias = if has_bias {
            Some(st.f32_vec(&format!("{name}.bias"), &[out_dim])?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_dim,
            out_dim,
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), rows * self.in_dim);
        let mut out = vec![0.0_f32; rows * self.out_dim];
        for r in 0..rows {
            let x_row = &x[r * self.in_dim..(r + 1) * self.in_dim];
            let out_row = &mut out[r * self.out_dim..(r + 1) * self.out_dim];
            for o in 0..self.out_dim {
                let w_row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
                let mut sum = 0.0_f32;
                for i in 0..self.in_dim {
                    sum += x_row[i] * w_row[i];
                }
                if let Some(b) = &self.bias {
                    sum += b[o];
                }
                out_row[o] = sum;
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm1d {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub dim: usize,
    pub eps: f32,
}

impl LayerNorm1d {
    pub fn from_safetensors(st: &SafetensorsMap<'_>, name: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            weight: st.f32_vec(&format!("{name}.weight"), &[dim])?,
            bias: st.f32_vec(&format!("{name}.bias"), &[dim])?,
            dim,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), rows * self.dim);
        let mut out = vec![0.0_f32; rows * self.dim];
        for r in 0..rows {
            let row = &x[r * self.dim..(r + 1) * self.dim];
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / (self.dim as f64);
            let var = row
                .iter()
                .map(|&v| {
                    let d = f64::from(v) - mean;
                    d * d
                })
                .sum::<f64>()
                / (self.dim as f64);
            let inv_std = 1.0 / (var + f64::from(self.eps)).sqrt();
            let out_row = &mut out[r * self.dim..(r + 1) * self.dim];
            for c in 0..self.dim {
                let norm = (f64::from(row[c]) - mean) * inv_std;
                out_row[c] = (norm * f64::from(self.weight[c]) + f64::from(self.bias[c])) as f32;
            }
        }
        out
    }
}

fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

#[derive(Clone, Debug)]
pub struct StandardizedPreCondenseContext {
    pub dope629_z: Vec<f32>,
    pub block_log_norms: [f32; LOG_NORM_DIM],
    pub hidden256_z: Vec<f32>,
    pub col18_z: Vec<f32>,
    pub col_mask_64: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct H256FoldScaler {
    pub dope629_mean: Vec<f32>,
    pub dope629_std: Vec<f32>,
    pub hidden256_mean: Vec<f32>,
    pub hidden256_std: Vec<f32>,
    pub col18_mean: Vec<f32>,
    pub col18_std: Vec<f32>,
}

impl H256FoldScaler {
    pub fn from_safetensors(st: &SafetensorsMap<'_>, prefix: &str) -> Result<Self> {
        Ok(Self {
            dope629_mean: st.f32_vec(&format!("{prefix}.dope629_mean"), &[DOPE_DIM])?,
            dope629_std: st.f32_vec(&format!("{prefix}.dope629_std"), &[DOPE_DIM])?,
            hidden256_mean: st.f32_vec(&format!("{prefix}.hidden256_mean"), &[HIDDEN_DIM])?,
            hidden256_std: st.f32_vec(&format!("{prefix}.hidden256_std"), &[HIDDEN_DIM])?,
            col18_mean: st.f32_vec(&format!("{prefix}.col18_mean"), &[COL_ENC_DIM])?,
            col18_std: st.f32_vec(&format!("{prefix}.col18_std"), &[COL_ENC_DIM])?,
        })
    }

    pub fn compute_block_log_norms(dope_or_z: &[f32]) -> [f32; LOG_NORM_DIM] {
        fn rms(slice: &[f32]) -> f32 {
            let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / (slice.len() as f32);
            (mean_sq + 1e-8).sqrt().ln_1p()
        }
        fn max_abs(slice: &[f32]) -> f32 {
            slice
                .iter()
                .fold(0.0_f32, |acc, &v| acc.max(v.abs()))
                .ln_1p()
        }
        [
            rms(&dope_or_z[0..256]),
            max_abs(&dope_or_z[0..256]),
            rms(&dope_or_z[256..384]),
            max_abs(&dope_or_z[256..384]),
            rms(&dope_or_z[384..512]),
            max_abs(&dope_or_z[384..512]),
            rms(&dope_or_z[512..526]),
            rms(&dope_or_z[526..549]),
            rms(&dope_or_z[549..565]),
            rms(&dope_or_z[565..629]),
            max_abs(&dope_or_z[565..629]),
        ]
    }

    pub fn transform_context(
        &self,
        dope629: &[f32],
        hidden256: &[f32],
        col_lm_64x17: &[f32],
        col_mask_64: &[f32],
    ) -> Result<StandardizedPreCondenseContext> {
        ensure!(
            dope629.len() == DOPE_DIM
                && hidden256.len() == HIDDEN_DIM
                && col_lm_64x17.len() == COL_SLOTS * COL_RAW_DIM
                && col_mask_64.len() == COL_SLOTS,
            "invalid raw context dimensions"
        );
        let dope629_z: Vec<f32> = (0..DOPE_DIM)
            .map(|i| ((dope629[i] - self.dope629_mean[i]) / self.dope629_std[i]).clamp(-15.0, 15.0))
            .collect();
        let hidden256_z: Vec<f32> = (0..HIDDEN_DIM)
            .map(|i| {
                ((hidden256[i] - self.hidden256_mean[i]) / self.hidden256_std[i]).clamp(-15.0, 15.0)
            })
            .collect();
        let mut col18_z = vec![0.0_f32; COL_SLOTS * COL_ENC_DIM];
        for c in 0..COL_SLOTS {
            let mask_val = if col_mask_64[c] > 0.5 {
                1.0_f32
            } else {
                0.0_f32
            };
            for f in 0..COL_RAW_DIM {
                let raw = col_lm_64x17[c * COL_RAW_DIM + f];
                let z = ((raw - self.col18_mean[f]) / self.col18_std[f]).clamp(-15.0, 15.0);
                col18_z[c * COL_ENC_DIM + f] = z * mask_val;
            }
            let raw_gate = dope629[565 + c];
            let z_gate = ((raw_gate - self.col18_mean[17]) / self.col18_std[17]).clamp(-15.0, 15.0);
            col18_z[c * COL_ENC_DIM + 17] = z_gate * mask_val;
        }
        let block_log_norms = Self::compute_block_log_norms(&dope629_z);
        Ok(StandardizedPreCondenseContext {
            dope629_z,
            block_log_norms,
            hidden256_z,
            col18_z,
            col_mask_64: col_mask_64
                .iter()
                .map(|&m| if m > 0.5 { 1.0 } else { 0.0 })
                .collect(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct SyncFreeCrossAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub out_proj: LinearLayer,
    pub d_model: usize,
    pub n_heads: usize,
}

impl SyncFreeCrossAttention {
    pub fn from_safetensors(
        st: &SafetensorsMap<'_>,
        prefix: &str,
        d_model: usize,
        n_heads: usize,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: LinearLayer::from_safetensors(
                st,
                &format!("{prefix}.q_proj"),
                d_model,
                d_model,
                true,
            )?,
            k_proj: LinearLayer::from_safetensors(
                st,
                &format!("{prefix}.k_proj"),
                d_model,
                d_model,
                true,
            )?,
            v_proj: LinearLayer::from_safetensors(
                st,
                &format!("{prefix}.v_proj"),
                d_model,
                d_model,
                true,
            )?,
            out_proj: LinearLayer::from_safetensors(
                st,
                &format!("{prefix}.out_proj"),
                d_model,
                d_model,
                true,
            )?,
            d_model,
            n_heads,
        })
    }

    pub fn forward(
        &self,
        query: &[f32],
        sq: usize,
        key_value: &[f32],
        sk: usize,
        valid_mask: Option<&[f32]>,
    ) -> Vec<f32> {
        let q = self.q_proj.forward(query, sq);
        let k = self.k_proj.forward(key_value, sk);
        let v = self.v_proj.forward(key_value, sk);
        let d_head = self.d_model / self.n_heads;
        let scale = 1.0_f64 / (d_head as f64).sqrt();

        let mut context = vec![0.0_f32; sq * self.d_model];
        for h in 0..self.n_heads {
            let h_off = h * d_head;
            for qi in 0..sq {
                let q_slice = &q[qi * self.d_model + h_off..qi * self.d_model + h_off + d_head];
                let mut logits = vec![0.0_f64; sk];
                let mut max_logit = f64::NEG_INFINITY;
                for ki in 0..sk {
                    let k_slice = &k[ki * self.d_model + h_off..ki * self.d_model + h_off + d_head];
                    let mut dot = 0.0_f64;
                    for d in 0..d_head {
                        dot += f64::from(q_slice[d]) * f64::from(k_slice[d]);
                    }
                    let mut score = dot * scale;
                    if let Some(mask) = valid_mask {
                        if mask[ki] <= 0.5 {
                            score += -1e4;
                        }
                    }
                    logits[ki] = score;
                    if score > max_logit {
                        max_logit = score;
                    }
                }
                let mut sum_exp = 0.0_f64;
                for s in logits.iter_mut() {
                    *s = (*s - max_logit).exp();
                    sum_exp += *s;
                }
                let inv_sum = 1.0 / sum_exp.max(1e-12);
                for d in 0..d_head {
                    let mut acc = 0.0_f64;
                    for ki in 0..sk {
                        let prob = logits[ki] * inv_sum;
                        let v_val = v[ki * self.d_model + h_off + d];
                        acc += prob * f64::from(v_val);
                    }
                    context[qi * self.d_model + h_off + d] = acc as f32;
                }
            }
        }
        self.out_proj.forward(&context, sq)
    }
}

#[derive(Clone, Debug)]
pub struct PriorAndContextOutput {
    pub ctx_tokens_92x32: Vec<f32>,
    pub ctx_mask_92: Vec<f32>,
    pub scale_feat_8: [f32; 8],
    pub attn_ctx_32: Vec<f32>,
    pub bilinear_24: Vec<f32>,
    pub delta_prior: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct H256Stage1ContextEncoder {
    pub desc_proj: LinearLayer,
    pub task_proj: LinearLayer,
    pub dist_proj: LinearLayer,
    pub heads_proj: LinearLayer,
    pub colgate_proj: LinearLayer,
    pub full629_norm: LayerNorm1d,
    pub full629_type_emb: Vec<f32>,
    pub full629_gate_logits: Vec<f32>,

    pub hidden_proj: LinearLayer,
    pub hidden_norm: LayerNorm1d,
    pub hidden_type_emb: Vec<f32>,
    pub hidden_gate_logits: Vec<f32>,

    pub col_enc_0: LinearLayer,
    pub col_enc_2: LinearLayer,
    pub col_enc_3: LayerNorm1d,
    pub col_type_emb: Vec<f32>,
    pub col_gate_logit: f32,

    pub ctx_cross_attn: SyncFreeCrossAttention,
    pub ctx_bilinear_u: LinearLayer,
    pub scale_side_proj: LinearLayer,
}

impl H256Stage1ContextEncoder {
    pub fn from_safetensors(st: &SafetensorsMap<'_>, prefix: &str) -> Result<Self> {
        let p = |sub: &str| format!("{prefix}.{sub}");
        Ok(Self {
            desc_proj: LinearLayer::from_safetensors(st, &p("desc_proj"), 32, 32, true)?,
            task_proj: LinearLayer::from_safetensors(st, &p("task_proj"), 32, 32, true)?,
            dist_proj: LinearLayer::from_safetensors(st, &p("dist_proj"), 32, 32, true)?,
            heads_proj: LinearLayer::from_safetensors(st, &p("heads_proj"), 32, 32, true)?,
            colgate_proj: LinearLayer::from_safetensors(st, &p("colgate_proj"), 32, 32, true)?,
            full629_norm: LayerNorm1d::from_safetensors(st, &p("full629_norm"), 32)?,
            full629_type_emb: st.f32_vec(&p("full629_type_emb"), &[20, 32])?,
            full629_gate_logits: st.f32_vec(&p("full629_gate_logits"), &[20])?,

            hidden_proj: LinearLayer::from_safetensors(st, &p("hidden_proj"), 32, 32, true)?,
            hidden_norm: LayerNorm1d::from_safetensors(st, &p("hidden_norm"), 32)?,
            hidden_type_emb: st.f32_vec(&p("hidden_type_emb"), &[8, 32])?,
            hidden_gate_logits: st.f32_vec(&p("hidden_gate_logits"), &[8])?,

            col_enc_0: LinearLayer::from_safetensors(st, &p("col_encoder.0"), 18, 32, true)?,
            col_enc_2: LinearLayer::from_safetensors(st, &p("col_encoder.2"), 32, 32, true)?,
            col_enc_3: LayerNorm1d::from_safetensors(st, &p("col_encoder.3"), 32)?,
            col_type_emb: st.f32_vec(&p("col_type_emb"), &[1, 32])?,
            col_gate_logit: st.f32_vec(&p("col_gate_logit"), &[1])?[0],

            ctx_cross_attn: SyncFreeCrossAttention::from_safetensors(
                st,
                &p("ctx_cross_attn"),
                32,
                4,
            )?,
            ctx_bilinear_u: LinearLayer::from_safetensors(st, &p("ctx_bilinear_u"), 64, 24, false)?,
            scale_side_proj: LinearLayer::from_safetensors(st, &p("scale_side_proj"), 11, 8, true)?,
        })
    }

    pub fn encode_context_tokens(
        &self,
        ctx: &StandardizedPreCondenseContext,
    ) -> (Vec<f32>, Vec<f32>, [f32; 8]) {
        // 1. 20 Full629 tokens
        let desc_8 = self.desc_proj.forward(&ctx.dope629_z[0..256], 8);
        let task_4 = self.task_proj.forward(&ctx.dope629_z[256..384], 4);
        let dist_4 = self.dist_proj.forward(&ctx.dope629_z[384..512], 4);
        let mut heads_in = [0.0_f32; 64];
        heads_in[..53].copy_from_slice(&ctx.dope629_z[512..565]);
        heads_in[53..64].copy_from_slice(&ctx.block_log_norms);
        let heads_2 = self.heads_proj.forward(&heads_in, 2);
        let colg_2 = self.colgate_proj.forward(&ctx.dope629_z[565..629], 2);

        let mut raw_20 = Vec::with_capacity(20 * 32);
        raw_20.extend_from_slice(&desc_8);
        raw_20.extend_from_slice(&task_4);
        raw_20.extend_from_slice(&dist_4);
        raw_20.extend_from_slice(&heads_2);
        raw_20.extend_from_slice(&colg_2);
        for (v, &e) in raw_20.iter_mut().zip(&self.full629_type_emb) {
            *v += e;
        }
        let mut tok_20 = self.full629_norm.forward(&raw_20, 20);
        for t in 0..20 {
            let g = sigmoid_f32(self.full629_gate_logits[t]);
            for d in 0..32 {
                tok_20[t * 32 + d] *= g;
            }
        }

        // 2. 8 Hidden256 tokens
        let mut raw_h8 = self.hidden_proj.forward(&ctx.hidden256_z, 8);
        for (v, &e) in raw_h8.iter_mut().zip(&self.hidden_type_emb) {
            *v += e;
        }
        let mut tok_h8 = self.hidden_norm.forward(&raw_h8, 8);
        for t in 0..8 {
            let g = sigmoid_f32(self.hidden_gate_logits[t]);
            for d in 0..32 {
                tok_h8[t * 32 + d] *= g;
            }
        }

        // 3. 64 Column tokens
        let mut col_h = self.col_enc_0.forward(&ctx.col18_z, COL_SLOTS);
        silu_inplace(&mut col_h);
        let col_h2 = self.col_enc_2.forward(&col_h, COL_SLOTS);
        let mut col_tok = self.col_enc_3.forward(&col_h2, COL_SLOTS);
        let col_gate = sigmoid_f32(self.col_gate_logit);
        for c in 0..COL_SLOTS {
            let mask_val = if ctx.col_mask_64[c] > 0.5 {
                1.0_f32
            } else {
                0.0_f32
            };
            for d in 0..32 {
                col_tok[c * 32 + d] =
                    (col_tok[c * 32 + d] + self.col_type_emb[d]) * col_gate * mask_val;
            }
        }

        let mut tok_92 = Vec::with_capacity(PRE_CONDENSE_TOKENS * TOKEN_DIM);
        tok_92.extend_from_slice(&tok_20);
        tok_92.extend_from_slice(&tok_h8);
        tok_92.extend_from_slice(&col_tok);

        let mut mask_92 = vec![1.0_f32; PRE_CONDENSE_TOKENS];
        for c in 0..COL_SLOTS {
            mask_92[28 + c] = if ctx.col_mask_64[c] > 0.5 { 1.0 } else { 0.0 };
        }

        let scale_vec = self.scale_side_proj.forward(&ctx.block_log_norms, 1);
        let scale_8: [f32; 8] = scale_vec.try_into().expect("8 scale features");
        (tok_92, mask_92, scale_8)
    }

    pub fn compute_ctx_summary_64(ctx_tokens_92x32: &[f32], ctx_mask_92: &[f32]) -> [f32; 64] {
        let mask_sum = ctx_mask_92.iter().sum::<f32>().max(1.0);
        let mut mean_32 = [0.0_f32; 32];
        let mut max_32 = [-1e4_f32; 32];
        for t in 0..PRE_CONDENSE_TOKENS {
            let m = ctx_mask_92[t];
            let row = &ctx_tokens_92x32[t * 32..(t + 1) * 32];
            for d in 0..32 {
                mean_32[d] += row[d] * m;
                let val_for_max = if m > 0.5 { row[d] } else { -1e4 };
                if val_for_max > max_32[d] {
                    max_32[d] = val_for_max;
                }
            }
        }
        for d in 0..32 {
            mean_32[d] /= mask_sum;
        }
        let mut out = [0.0_f32; 64];
        out[..32].copy_from_slice(&mean_32);
        out[32..].copy_from_slice(&max_32);
        out
    }

    pub fn extract_s3_ctx_vec16(&self, ctx: &StandardizedPreCondenseContext) -> [f32; 16] {
        let (tok_92, mask_92, scale_8) = self.encode_context_tokens(ctx);
        let summary_64 = Self::compute_ctx_summary_64(&tok_92, &mask_92);
        let u_ctx = self.ctx_bilinear_u.forward(&summary_64, 1);
        let mut vec16 = [0.0_f32; 16];
        vec16[..12].copy_from_slice(&u_ctx[..12]);
        vec16[12..16].copy_from_slice(&scale_8[..4]);
        vec16
    }
}

#[derive(Clone, Debug)]
pub struct H256PreCondenseBridge {
    pub ctx_encoder: H256Stage1ContextEncoder,
    pub prog_bilinear_v: LinearLayer,

    pub prior_mlp_0: LinearLayer,
    pub prior_mlp_2: LayerNorm1d,
    pub prior_out: LinearLayer,

    pub hist_proj_0: LinearLayer,
    pub hist_proj_1: LayerNorm1d,
    pub null_hist_tok: Vec<f32>,
    pub traj_cross_attn: SyncFreeCrossAttention,
    pub ctx_kernel_scale_raw: f32,
    pub traj_mlp_0: LinearLayer,
    pub traj_mlp_2: LayerNorm1d,
    pub traj_out: LinearLayer,
}

impl H256PreCondenseBridge {
    pub fn from_safetensors(st: &SafetensorsMap<'_>, prefix: &str) -> Result<Self> {
        let p = |sub: &str| format!("{prefix}.{sub}");
        Ok(Self {
            ctx_encoder: H256Stage1ContextEncoder::from_safetensors(st, prefix)?,
            prog_bilinear_v: LinearLayer::from_safetensors(
                st,
                &p("prog_bilinear_v"),
                24,
                24,
                false,
            )?,
            prior_mlp_0: LinearLayer::from_safetensors(
                st,
                &p("prior_residual_mlp.0"),
                98,
                32,
                true,
            )?,
            prior_mlp_2: LayerNorm1d::from_safetensors(st, &p("prior_residual_mlp.2"), 32)?,
            prior_out: LinearLayer::from_safetensors(st, &p("prior_residual_out"), 32, 1, true)?,

            hist_proj_0: LinearLayer::from_safetensors(st, &p("hist_proj.0"), 59, 32, true)?,
            hist_proj_1: LayerNorm1d::from_safetensors(st, &p("hist_proj.1"), 32)?,
            null_hist_tok: st.f32_vec(&p("null_hist_tok"), &[1, 1, 32])?,
            traj_cross_attn: SyncFreeCrossAttention::from_safetensors(
                st,
                &p("traj_cross_attn"),
                32,
                4,
            )?,
            ctx_kernel_scale_raw: st.f32_vec(&p("ctx_kernel_scale_raw"), &[1])?[0],
            traj_mlp_0: LinearLayer::from_safetensors(st, &p("traj_residual_mlp.0"), 91, 32, true)?,
            traj_mlp_2: LayerNorm1d::from_safetensors(st, &p("traj_residual_mlp.2"), 32)?,
            traj_out: LinearLayer::from_safetensors(st, &p("traj_residual_out"), 32, 1, true)?,
        })
    }

    pub fn encode_context_tokens(
        &self,
        ctx: &StandardizedPreCondenseContext,
    ) -> (Vec<f32>, Vec<f32>, [f32; 8]) {
        self.ctx_encoder.encode_context_tokens(ctx)
    }

    pub fn extract_s3_ctx_vec16(&self, ctx: &StandardizedPreCondenseContext) -> [f32; 16] {
        self.ctx_encoder.extract_s3_ctx_vec16(ctx)
    }

    pub fn forward_prior_and_context(
        &self,
        ctx: &StandardizedPreCondenseContext,
        cand_q_32: &[f32],
        sem24: &[f32],
        gate_risk: &[f32],
        sp_prior: &[f32],
        n_cand: usize,
    ) -> Result<PriorAndContextOutput> {
        ensure!(
            cand_q_32.len() == n_cand * 32
                && sem24.len() == n_cand * 24
                && gate_risk.len() == n_cand
                && sp_prior.len() == n_cand,
            "invalid candidate input lengths in forward_prior_and_context"
        );
        let (ctx_tokens_92x32, ctx_mask_92, scale_feat_8) = self.encode_context_tokens(ctx);
        let summary_64 =
            H256Stage1ContextEncoder::compute_ctx_summary_64(&ctx_tokens_92x32, &ctx_mask_92);

        let attn_ctx_32 = self.ctx_encoder.ctx_cross_attn.forward(
            cand_q_32,
            n_cand,
            &ctx_tokens_92x32,
            PRE_CONDENSE_TOKENS,
            Some(&ctx_mask_92),
        );
        let u_ctx = self.ctx_encoder.ctx_bilinear_u.forward(&summary_64, 1);
        let v_prog = self.prog_bilinear_v.forward(sem24, n_cand);

        let mut bilinear_24 = vec![0.0_f32; n_cand * 24];
        let mut prior_in = vec![0.0_f32; n_cand * 98];
        for i in 0..n_cand {
            for d in 0..24 {
                bilinear_24[i * 24 + d] = u_ctx[d] * v_prog[i * 24 + d];
            }
            let row = &mut prior_in[i * 98..(i + 1) * 98];
            row[0..32].copy_from_slice(&attn_ctx_32[i * 32..(i + 1) * 32]);
            row[32..56].copy_from_slice(&bilinear_24[i * 24..(i + 1) * 24]);
            row[56..88].copy_from_slice(&cand_q_32[i * 32..(i + 1) * 32]);
            row[88..96].copy_from_slice(&scale_feat_8);
            row[96] = gate_risk[i];
            row[97] = sp_prior[i];
        }

        let mut h_prior = self.prior_mlp_0.forward(&prior_in, n_cand);
        silu_inplace(&mut h_prior);
        let h_prior_norm = self.prior_mlp_2.forward(&h_prior, n_cand);
        let raw_prior = self.prior_out.forward(&h_prior_norm, n_cand);

        let mean = raw_prior.iter().map(|&v| f64::from(v)).sum::<f64>() / (n_cand as f64);
        let delta_prior: Vec<f64> = raw_prior
            .iter()
            .map(|&v| f64::from((v as f64 - mean) as f32))
            .collect();

        Ok(PriorAndContextOutput {
            ctx_tokens_92x32,
            ctx_mask_92,
            scale_feat_8,
            attn_ctx_32,
            bilinear_24,
            delta_prior,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_causal_trajectory_residual(
        &self,
        attn_ctx_32: &[f32],
        bilinear_24: &[f32],
        cand_q_32: &[f32],
        sem24: &[f32],
        base_cost_plus_prior: &[f32],
        rev_idx: &[usize],
        rev_y_comb: &[f32],
        rev_y_f: &[f32],
        rev_gate: &[f32],
        n_cand: usize,
    ) -> Result<Vec<f64>> {
        let k_rev = rev_idx.len();
        ensure!(
            k_rev > 0
                && rev_y_comb.len() == k_rev
                && rev_y_f.len() == k_rev
                && rev_gate.len() == k_rev,
            "invalid revealed trajectory slice lengths"
        );

        let mut joint_feat = vec![0.0_f32; n_cand * 56];
        for i in 0..n_cand {
            let dst = &mut joint_feat[i * 56..(i + 1) * 56];
            dst[..24].copy_from_slice(&sem24[i * 24..(i + 1) * 24]);
            dst[24..56].copy_from_slice(&attn_ctx_32[i * 32..(i + 1) * 32]);
            let norm = dst.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-8);
            for v in dst.iter_mut() {
                *v /= norm;
            }
        }

        let mut r_rev = vec![0.0_f32; k_rev];
        let mut h_in = vec![0.0_f32; k_rev * 59];
        for (j, &idx) in rev_idx.iter().enumerate() {
            r_rev[j] = rev_y_comb[j] - base_cost_plus_prior[idx];
            let dst = &mut h_in[j * 59..(j + 1) * 59];
            dst[..24].copy_from_slice(&sem24[idx * 24..(idx + 1) * 24]);
            dst[24..56].copy_from_slice(&attn_ctx_32[idx * 32..(idx + 1) * 32]);
            dst[56] = r_rev[j];
            dst[57] = rev_y_f[j];
            dst[58] = rev_gate[j];
        }

        let kernel_scale = softplus_f32(self.ctx_kernel_scale_raw);
        let mut sem_ctx_adj = vec![0.0_f32; n_cand];
        for i in 0..n_cand {
            let jf_i = &joint_feat[i * 56..(i + 1) * 56];
            let mut num = 0.0_f32;
            let mut den = 0.0_f32;
            for (j, &idx) in rev_idx.iter().enumerate() {
                let jf_r = &joint_feat[idx * 56..(idx + 1) * 56];
                let cos_sim = jf_i
                    .iter()
                    .zip(jf_r)
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
                    .clamp(-1.0, 1.0);
                let d2 = (1.0 - cos_sim) * kernel_scale;
                let wk = 0.70 / (0.20 + d2) + 0.30 / (1.50 + d2);
                num += wk * r_rev[j];
                den += wk;
            }
            sem_ctx_adj[i] = num / (0.25 + den);
        }

        let h_lin = self.hist_proj_0.forward(&h_in, k_rev);
        let mut h_tok = self.hist_proj_1.forward(&h_lin, k_rev);
        silu_inplace(&mut h_tok);

        let mut kv_seq = Vec::with_capacity((1 + k_rev) * 32);
        kv_seq.extend_from_slice(&self.null_hist_tok);
        kv_seq.extend_from_slice(&h_tok);

        let traj_attn = self
            .traj_cross_attn
            .forward(cand_q_32, n_cand, &kv_seq, 1 + k_rev, None);
        let best_rev = rev_y_f.iter().copied().fold(f32::INFINITY, f32::min);

        let mut traj_in = vec![0.0_f32; n_cand * 91];
        for i in 0..n_cand {
            let row = &mut traj_in[i * 91..(i + 1) * 91];
            row[0..32].copy_from_slice(&attn_ctx_32[i * 32..(i + 1) * 32]);
            row[32..64].copy_from_slice(&traj_attn[i * 32..(i + 1) * 32]);
            row[64..88].copy_from_slice(&bilinear_24[i * 24..(i + 1) * 24]);
            row[88] = sem_ctx_adj[i];
            row[89] = base_cost_plus_prior[i];
            row[90] = best_rev;
        }

        let mut h_traj = self.traj_mlp_0.forward(&traj_in, n_cand);
        silu_inplace(&mut h_traj);
        let h_traj_norm = self.traj_mlp_2.forward(&h_traj, n_cand);
        let raw_traj = self.traj_out.forward(&h_traj_norm, n_cand);

        let mean = raw_traj.iter().map(|&v| f64::from(v)).sum::<f64>() / (n_cand as f64);
        Ok(raw_traj
            .iter()
            .map(|&v| f64::from((v as f64 - mean) as f32))
            .collect())
    }
}

#[derive(Clone, Debug)]
pub struct H256CriticModels {
    pub w_c_full: Vec<f64>,
    pub w_g_full: Vec<f64>,
    pub sp_dir_full: Vec<f64>,
    pub w_c_oof_5fold: Vec<f64>,
    pub w_g_oof_5fold: Vec<f64>,
    pub sp_dir_oof_5fold: Vec<f64>,
}

impl H256CriticModels {
    pub fn from_safetensors(st: &SafetensorsMap<'_>, prefix: &str) -> Result<Self> {
        Ok(Self {
            w_c_full: st.f64_vec(&format!("{prefix}.w_c_full"), &[65])?,
            w_g_full: st.f64_vec(&format!("{prefix}.w_g_full"), &[65])?,
            sp_dir_full: st.f64_vec(&format!("{prefix}.sp_dir_full"), &[24])?,
            w_c_oof_5fold: st.f64_vec(&format!("{prefix}.w_c_oof_5fold"), &[5, 65])?,
            w_g_oof_5fold: st.f64_vec(&format!("{prefix}.w_g_oof_5fold"), &[5, 65])?,
            sp_dir_oof_5fold: st.f64_vec(&format!("{prefix}.sp_dir_oof_5fold"), &[5, 24])?,
        })
    }

    /// Computes the 24D structural semantic signature `sem24` from `program_412` (`(n_cand, 412)`),
    /// matching `common_h252.compute_sem24_numpy`.
    pub fn compute_sem24_from_program_412(program_412: &[f32], n_cand: usize) -> Result<Vec<f32>> {
        ensure!(
            program_412.len() == n_cand * 412,
            "invalid program_412 length in compute_sem24_from_program_412"
        );
        let mut out = vec![0.0_f32; n_cand * 24];
        for i in 0..n_cand {
            let src = &program_412[i * 412..(i + 1) * 412];
            let mut padded = [0.0_f32; 18 * 24];
            padded[..412].copy_from_slice(src);
            let mut sig = [0.0_f32; 24];
            let mut norm_sq = 0.0_f32;
            for d in 0..24 {
                let mut sum_v = 0.0_f32;
                let mut max_v = f32::NEG_INFINITY;
                for r in 0..18 {
                    let v = padded[r * 24 + d];
                    sum_v += v;
                    if v > max_v {
                        max_v = v;
                    }
                }
                let s = (sum_v / 18.0) + 0.5 * max_v;
                sig[d] = s;
                norm_sq += s * s;
            }
            let norm = f64::from(norm_sq.sqrt()).max(1e-6);
            let dst = &mut out[i * 24..(i + 1) * 24];
            for d in 0..24 {
                dst[d] = (f64::from(sig[d]) / norm) as f32;
            }
        }
        Ok(out)
    }

    /// Constructs the outcome-free `(n_cand, 64)` candidate feature matrix `X_c`.
    pub fn build_candidate_features_64(
        program_412: &[f32],
        sem24: &[f32],
        rcost: &[f64],
        aprior: &[f64],
        atotal: &[f64],
        lm48_z: &[f32],
        n_cand: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            program_412.len() == n_cand * 412
                && sem24.len() == n_cand * 24
                && rcost.len() == n_cand
                && aprior.len() == n_cand
                && atotal.len() == n_cand
                && lm48_z.len() == 48,
            "invalid dimensions in build_candidate_features_64"
        );

        let r_mean = rcost.iter().sum::<f64>() / (n_cand as f64);
        let r_sd = (rcost
            .iter()
            .map(|&v| (v - r_mean) * (v - r_mean))
            .sum::<f64>()
            / (n_cand as f64))
            .sqrt()
            + 1e-6;
        let r_z: Vec<f64> = rcost.iter().map(|&v| (v - r_mean) / r_sd).collect();
        let r_min = rcost.iter().copied().fold(f64::INFINITY, f64::min);

        let mut order: Vec<usize> = (0..n_cand).collect();
        order.sort_by(|&a, &b| rcost[a].total_cmp(&rcost[b]).then_with(|| a.cmp(&b)));
        let r_top8 = rcost[order[8]];
        let mut ranks = vec![0.0_f64; n_cand];
        for (rank_pos, &idx) in order.iter().enumerate() {
            ranks[idx] = (rank_pos as f64) / (n_cand as f64);
        }

        let ap_mean = aprior.iter().sum::<f64>() / (n_cand as f64);
        let ap_sd = (aprior
            .iter()
            .map(|&v| (v - ap_mean) * (v - ap_mean))
            .sum::<f64>()
            / (n_cand as f64))
            .sqrt()
            + 1e-6;
        let ap_z: Vec<f64> = aprior.iter().map(|&v| (v - ap_mean) / ap_sd).collect();

        let adelta: Vec<f64> = (0..n_cand).map(|i| rcost[i] - atotal[i]).collect();
        let ad_mean = adelta.iter().sum::<f64>() / (n_cand as f64);
        let ad_sd = (adelta
            .iter()
            .map(|&v| (v - ad_mean) * (v - ad_mean))
            .sum::<f64>()
            / (n_cand as f64))
            .sqrt()
            + 1e-6;
        let ad_z: Vec<f64> = adelta.iter().map(|&v| (v - ad_mean) / ad_sd).collect();

        let top1_idx = order[0];
        let top8_idx = &order[..8];
        let mut sem_top8 = [0.0_f64; 24];
        for &idx in top8_idx {
            for d in 0..24 {
                sem_top8[d] += f64::from(sem24[idx * 24 + d]) / 8.0;
            }
        }
        let sem_top8_norm = sem_top8
            .iter()
            .map(|&v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1e-6);
        for v in sem_top8.iter_mut() {
            *v /= sem_top8_norm;
        }

        let mut sem_mean = [0.0_f64; 24];
        let mut sem_std = [0.0_f64; 24];
        for d in 0..24 {
            let m = (0..n_cand)
                .map(|i| f64::from(sem24[i * 24 + d]))
                .sum::<f64>()
                / (n_cand as f64);
            let s = ((0..n_cand)
                .map(|i| {
                    let diff = f64::from(sem24[i * 24 + d]) - m;
                    diff * diff
                })
                .sum::<f64>()
                / (n_cand as f64))
                .sqrt()
                + 1e-5;
            sem_mean[d] = m;
            sem_std[d] = s;
        }

        let mut st_mean = [0.0_f64; 20];
        let mut st_std = [0.0_f64; 20];
        let st_col_idx = |c: usize| if c < 12 { c } else { 48 + (c - 12) };
        for c in 0..20 {
            let col = st_col_idx(c);
            let m = (0..n_cand)
                .map(|i| f64::from(program_412[i * 412 + col]))
                .sum::<f64>()
                / (n_cand as f64);
            let s = ((0..n_cand)
                .map(|i| {
                    let diff = f64::from(program_412[i * 412 + col]) - m;
                    diff * diff
                })
                .sum::<f64>()
                / (n_cand as f64))
                .sqrt()
                + 1e-5;
            st_mean[c] = m;
            st_std[c] = s;
        }

        let mut out = vec![0.0_f32; n_cand * 64];
        for i in 0..n_cand {
            let row = &mut out[i * 64..(i + 1) * 64];
            let s_i = &sem24[i * 24..(i + 1) * 24];
            let s_top1 = &sem24[top1_idx * 24..(top1_idx + 1) * 24];
            let cos_top1 = (0..24)
                .map(|d| f64::from(s_i[d]) * f64::from(s_top1[d]))
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            let cos_top8 = (0..24)
                .map(|d| f64::from(s_i[d]) * sem_top8[d])
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            let mut max_cos_rev4 = f64::NEG_INFINITY;
            let mut sum_cos_rev4 = 0.0_f64;
            for r in 0..4 {
                let s_r = &sem24[r * 24..(r + 1) * 24];
                let c_val = (0..24)
                    .map(|d| f64::from(s_i[d]) * f64::from(s_r[d]))
                    .sum::<f64>()
                    .clamp(-1.0, 1.0);
                if c_val > max_cos_rev4 {
                    max_cos_rev4 = c_val;
                }
                sum_cos_rev4 += c_val;
            }
            let mean_cos_rev4 = sum_cos_rev4 / 4.0;

            row[0] = r_z[i] as f32;
            row[1] = ((rcost[i] - r_min) / r_sd) as f32;
            row[2] = ((rcost[i] - r_top8) / r_sd) as f32;
            row[3] = ap_z[i] as f32;
            row[4] = ad_z[i] as f32;
            row[5] = ranks[i] as f32;
            row[6] = cos_top1 as f32;
            row[7] = cos_top8 as f32;
            row[8] = max_cos_rev4 as f32;
            row[9] = mean_cos_rev4 as f32;

            for d in 0..24 {
                let z = ((f64::from(s_i[d]) - sem_mean[d]) / sem_std[d]).clamp(-5.0, 5.0);
                row[10 + d] = z as f32;
            }
            for c in 0..20 {
                let col = st_col_idx(c);
                let z = ((f64::from(program_412[i * 412 + col]) - st_mean[c]) / st_std[c])
                    .clamp(-5.0, 5.0);
                row[34 + c] = z as f32;
            }
            for k in 0..5 {
                let ctx_v = f64::from(lm48_z[k]);
                row[54 + k] = (r_z[i] * ctx_v) as f32;
                row[59 + k] = (ap_z[i] * ctx_v) as f32;
            }
        }
        Ok(out)
    }

    pub fn predict_priors_and_h_cost(
        &self,
        cand_feats_64: &[f32],
        sem24: &[f32],
        rcost: &[f64],
        n_cand: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f64>)> {
        ensure!(
            cand_feats_64.len() == n_cand * 64
                && sem24.len() == n_cand * 24
                && rcost.len() == n_cand,
            "invalid dimensions in predict_priors_and_h_cost"
        );
        let mut pc = vec![0.0_f64; n_cand];
        let mut pg = vec![0.0_f64; n_cand];
        let mut sp_proj = vec![0.0_f64; n_cand];

        for i in 0..n_cand {
            let x_row = &cand_feats_64[i * 64..(i + 1) * 64];
            let mut acc_c = self.w_c_full[64];
            let mut acc_g = self.w_g_full[64];
            for d in 0..64 {
                let xv = f64::from(x_row[d]);
                acc_c += xv * self.w_c_full[d];
                acc_g += xv * self.w_g_full[d];
            }
            pc[i] = acc_c;
            pg[i] = acc_g;

            let s_row = &sem24[i * 24..(i + 1) * 24];
            let mut dot_sp = 0.0_f64;
            for d in 0..24 {
                dot_sp += f64::from(s_row[d]) * self.sp_dir_full[d];
            }
            sp_proj[i] = -dot_sp;
        }

        let z_norm_f32 = |v: &[f64]| -> Vec<f32> {
            let mean = v.iter().sum::<f64>() / (v.len() as f64);
            let sd = (v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (v.len() as f64))
                .sqrt()
                + 1e-6;
            v.iter().map(|&x| ((x - mean) / sd) as f32).collect()
        };

        let pc_z = z_norm_f32(&pc);
        let pg_z = z_norm_f32(&pg);
        let sp_prior = z_norm_f32(&sp_proj);

        let mut sos_prior_2d = vec![0.0_f32; n_cand * 2];
        for i in 0..n_cand {
            sos_prior_2d[i * 2] = pc_z[i];
            sos_prior_2d[i * 2 + 1] = pg_z[i];
        }

        let pick_r882 = (0..n_cand)
            .min_by(|&a, &b| rcost[a].total_cmp(&rcost[b]).then_with(|| a.cmp(&b)))
            .unwrap_or(0);
        let gate_risk_r882 = f64::from(pg_z[pick_r882]);

        let mut h_cost = vec![0.0_f64; n_cand];
        for i in 0..n_cand {
            let gr = f64::from(pg_z[i]);
            let penalty = if gr > gate_risk_r882 + 0.10 {
                10.0
            } else {
                0.0
            };
            h_cost[i] =
                0.65 * f64::from(pc_z[i]) + 0.35 * gr + 0.15 * f64::from(sp_prior[i]) + penalty;
        }
        Ok((sos_prior_2d, sp_prior, h_cost))
    }

    /// Executes the Round-1 `2+2` active reveal step.
    #[allow(clippy::too_many_arguments)]
    pub fn run_2plus2_active_reveal_step(
        rcost: &[f64],
        h_cost: &[f64],
        sem24: &[f32],
        fitness: &[f64],
        logratio: &[f64],
        gate: &[f32],
        idx4: &[usize],
        delta_prior: Option<&[f64]>,
        alpha_prior: f64,
        n_cand: usize,
    ) -> Result<(Vec<usize>, Vec<f64>, Vec<f32>, Vec<f32>)> {
        ensure!(
            idx4.len() >= 2,
            "idx4 must contain at least 2 initial probes"
        );
        let mut x_sem = vec![0.0_f64; n_cand * 24];
        for d in 0..24 {
            let m = (0..n_cand)
                .map(|i| f64::from(sem24[i * 24 + d]))
                .sum::<f64>()
                / (n_cand as f64);
            let s = ((0..n_cand)
                .map(|i| {
                    let diff = f64::from(sem24[i * 24 + d]) - m;
                    diff * diff
                })
                .sum::<f64>()
                / (n_cand as f64))
                .sqrt()
                + 1e-6;
            for i in 0..n_cand {
                x_sem[i * 24 + d] = (f64::from(sem24[i * 24 + d]) - m) / s;
            }
        }

        let rcost_adj: Vec<f64> = match delta_prior {
            Some(dp) if alpha_prior != 0.0 => (0..n_cand)
                .map(|i| rcost[i] + alpha_prior * dp[i])
                .collect(),
            _ => rcost.to_vec(),
        };

        let idx_a = [idx4[0], idx4[1]];
        let ya_cost = [
            if gate[idx_a[0]] > 0.5 {
                fitness[idx_a[0]]
            } else {
                2.0
            },
            if gate[idx_a[1]] > 0.5 {
                fitness[idx_a[1]]
            } else {
                2.0
            },
        ];
        let ya_mean = 0.5 * (ya_cost[0] + ya_cost[1]);
        let ya_c = [ya_cost[0] - ya_mean, ya_cost[1] - ya_mean];
        let h_ca_mean = 0.5 * (h_cost[idx_a[0]] + h_cost[idx_a[1]]);

        let mut interim_cost = vec![0.0_f64; n_cand];
        for i in 0..n_cand {
            let mut k_vals = [0.0_f64; 2];
            for (j, &a_idx) in idx_a.iter().enumerate() {
                let mut sq = 0.0_f64;
                for d in 0..24 {
                    let diff = x_sem[i * 24 + d] - x_sem[a_idx * 24 + d];
                    sq += diff * diff;
                }
                sq /= 24.0;
                k_vals[j] = (-sq / (2.0 * 2.2 * 2.2)).exp();
            }
            let den = k_vals[0] + k_vals[1] + 0.2;
            let wa_dot = (k_vals[0] * ya_c[0] + k_vals[1] * ya_c[1]) / den;
            interim_cost[i] = rcost_adj[i] + 0.035 * wa_dot + 0.010 * (h_cost[i] - h_ca_mean);
        }

        let mut order: Vec<usize> = (0..n_cand).collect();
        order.sort_by(|&a, &b| {
            interim_cost[a]
                .total_cmp(&interim_cost[b])
                .then_with(|| a.cmp(&b))
        });

        let mut rev4 = vec![idx_a[0], idx_a[1]];
        for &cand in &order {
            if !rev4.contains(&cand) {
                rev4.push(cand);
                break;
            }
        }
        let mut best_b2 = None;
        let mut best_acq = 1e18_f64;
        for &cand in order.iter().take(6) {
            if rev4.contains(&cand) {
                continue;
            }
            let mut d_min = f64::INFINITY;
            for &r_idx in &rev4 {
                let mut sq = 0.0_f64;
                for d in 0..24 {
                    let diff = x_sem[cand * 24 + d] - x_sem[r_idx * 24 + d];
                    sq += diff * diff;
                }
                sq /= 24.0;
                if sq < d_min {
                    d_min = sq;
                }
            }
            let acq = interim_cost[cand] - 0.015 * d_min.sqrt();
            if acq < best_acq {
                best_acq = acq;
                best_b2 = Some(cand);
            }
        }
        if let Some(b2) = best_b2 {
            rev4.push(b2);
        }

        let y4_f_f64: Vec<f64> = rev4
            .iter()
            .map(|&idx| if gate[idx] > 0.5 { fitness[idx] } else { 2.0 })
            .collect();
        let y4_lr: Vec<f64> = rev4
            .iter()
            .map(|&idx| logratio[idx].clamp(-1.5, 1.5))
            .collect();
        let y4_f_mean = y4_f_f64.iter().sum::<f64>() / (rev4.len() as f64);
        let y4_lr_mean = y4_lr.iter().sum::<f64>() / (rev4.len() as f64);
        let y4_comb_f64: Vec<f64> = (0..rev4.len())
            .map(|j| 0.70 * (y4_f_f64[j] - y4_f_mean) + 0.30 * (y4_lr[j] - y4_lr_mean))
            .collect();
        let h_c4_mean = rev4.iter().map(|&idx| h_cost[idx]).sum::<f64>() / (rev4.len() as f64);

        let mut cost_act = vec![0.0_f64; n_cand];
        for i in 0..n_cand {
            let mut num = 0.0_f64;
            let mut den = 0.15_f64;
            for (j, &r_idx) in rev4.iter().enumerate() {
                let mut sq = 0.0_f64;
                for d in 0..24 {
                    let diff = x_sem[i * 24 + d] - x_sem[r_idx * 24 + d];
                    sq += diff * diff;
                }
                sq /= 24.0;
                let k4 =
                    0.6 * (-sq / (2.0 * 1.8 * 1.8)).exp() + 0.4 * (-sq / (2.0 * 3.2 * 3.2)).exp();
                num += k4 * y4_comb_f64[j];
                den += k4;
            }
            cost_act[i] = rcost[i] + 0.05 * (num / den) + 0.010 * (h_cost[i] - h_c4_mean);
        }
        for (j, &r_idx) in rev4.iter().enumerate() {
            cost_act[r_idx] += 0.08 * (y4_f_f64[j] - y4_f_mean);
        }

        let y4_comb: Vec<f32> = y4_comb_f64.iter().map(|&v| v as f32).collect();
        let y4_f: Vec<f32> = y4_f_f64.iter().map(|&v| v as f32).collect();
        Ok((rev4, cost_act, y4_comb, y4_f))
    }
}

#[derive(Clone, Debug)]
pub struct FlatGbdtModel {
    pub is_binary_clf: bool,
    pub num_leaves: Vec<i32>,
    pub split_feature: Vec<i32>,
    pub threshold: Vec<f64>,
    pub decision_type: Vec<i32>,
    pub left_child: Vec<i32>,
    pub right_child: Vec<i32>,
    pub leaf_value: Vec<f64>,
    pub max_internal: usize,
    pub max_leaves: usize,
}

impl FlatGbdtModel {
    pub fn from_safetensors(st: &SafetensorsMap<'_>, prefix: &str) -> Result<Self> {
        let is_bin = st.i32_vec(&format!("{prefix}.is_binary_clf"), &[1])?[0] != 0;
        let shape_sf = st.shape(&format!("{prefix}.split_feature"))?;
        let shape_lv = st.shape(&format!("{prefix}.leaf_value"))?;
        ensure!(
            shape_sf.len() == 2 && shape_lv.len() == 2 && shape_sf[0] == shape_lv[0],
            "invalid GBDT tensor shapes for {prefix}"
        );
        let n_trees = shape_sf[0];
        let max_internal = shape_sf[1];
        let max_leaves = shape_lv[1];

        Ok(Self {
            is_binary_clf: is_bin,
            num_leaves: st.i32_vec(&format!("{prefix}.num_leaves"), &[n_trees])?,
            split_feature: st
                .i32_vec(&format!("{prefix}.split_feature"), &[n_trees, max_internal])?,
            threshold: st.f64_vec(&format!("{prefix}.threshold"), &[n_trees, max_internal])?,
            decision_type: st
                .i32_vec(&format!("{prefix}.decision_type"), &[n_trees, max_internal])?,
            left_child: st.i32_vec(&format!("{prefix}.left_child"), &[n_trees, max_internal])?,
            right_child: st.i32_vec(&format!("{prefix}.right_child"), &[n_trees, max_internal])?,
            leaf_value: st.f64_vec(&format!("{prefix}.leaf_value"), &[n_trees, max_leaves])?,
            max_internal,
            max_leaves,
        })
    }

    pub fn predict_batch(&self, x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f64> {
        debug_assert_eq!(x.len(), n_rows * n_cols);
        let n_trees = self.num_leaves.len();
        let mut out = vec![0.0_f64; n_rows];
        for r in 0..n_rows {
            let row = &x[r * n_cols..(r + 1) * n_cols];
            let mut acc = 0.0_f64;
            for t in 0..n_trees {
                let nl = self.num_leaves[t];
                if nl <= 1 {
                    acc += self.leaf_value[t * self.max_leaves];
                    continue;
                }
                let mut node = 0_i32;
                let int_base = t * self.max_internal;
                while node >= 0 {
                    let u_node = node as usize;
                    let f_idx = self.split_feature[int_base + u_node] as usize;
                    let th = self.threshold[int_base + u_node];
                    let val = f64::from(row[f_idx]);
                    if val <= th {
                        node = self.left_child[int_base + u_node];
                    } else {
                        node = self.right_child[int_base + u_node];
                    }
                }
                let leaf_idx = (-1 - node) as usize;
                acc += self.leaf_value[t * self.max_leaves + leaf_idx];
            }
            out[r] = if self.is_binary_clf {
                1.0 / (1.0 + (-acc).exp())
            } else {
                acc
            };
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct H256Top8NeuralReranker {
    pub in_norm: LayerNorm1d,
    pub in_proj_0: LinearLayer,
    pub in_proj_1: LayerNorm1d,
    pub in_proj_4: LinearLayer,
    pub in_proj_5: LayerNorm1d,
    pub slot_attn_in_weight: Vec<f32>,
    pub slot_attn_in_bias: Vec<f32>,
    pub slot_attn_out: LinearLayer,
    pub post_norm: LayerNorm1d,
    pub score_head_0: LinearLayer,
    pub score_head_2: LinearLayer,
    pub inv_head_0: LinearLayer,
    pub inv_head_2: LinearLayer,
    pub feat_mean: Vec<f32>,
    pub feat_std: Vec<f32>,
}

impl H256Top8NeuralReranker {
    pub fn from_safetensors(st: &SafetensorsMap<'_>, prefix: &str) -> Result<Self> {
        let p = |sub: &str| format!("{prefix}.{sub}");
        Ok(Self {
            in_norm: LayerNorm1d::from_safetensors(st, &p("in_norm"), TOP8_ENRICHED_DIM)?,
            in_proj_0: LinearLayer::from_safetensors(
                st,
                &p("in_proj.0"),
                TOP8_ENRICHED_DIM,
                128,
                true,
            )?,
            in_proj_1: LayerNorm1d::from_safetensors(st, &p("in_proj.1"), 128)?,
            in_proj_4: LinearLayer::from_safetensors(st, &p("in_proj.4"), 128, 128, true)?,
            in_proj_5: LayerNorm1d::from_safetensors(st, &p("in_proj.5"), 128)?,
            slot_attn_in_weight: st.f32_vec(&p("slot_attn.in_proj_weight"), &[384, 128])?,
            slot_attn_in_bias: st.f32_vec(&p("slot_attn.in_proj_bias"), &[384])?,
            slot_attn_out: LinearLayer::from_safetensors(
                st,
                &p("slot_attn.out_proj"),
                128,
                128,
                true,
            )?,
            post_norm: LayerNorm1d::from_safetensors(st, &p("post_norm"), 128)?,
            score_head_0: LinearLayer::from_safetensors(st, &p("score_head.0"), 256, 64, true)?,
            score_head_2: LinearLayer::from_safetensors(st, &p("score_head.2"), 64, 1, true)?,
            inv_head_0: LinearLayer::from_safetensors(st, &p("inv_head.0"), 256, 64, true)?,
            inv_head_2: LinearLayer::from_safetensors(st, &p("inv_head.2"), 64, 1, true)?,
            feat_mean: st.f32_vec(&p("feat_mean"), &[TOP8_ENRICHED_DIM])?,
            feat_std: st.f32_vec(&p("feat_std"), &[TOP8_ENRICHED_DIM])?,
        })
    }

    pub fn forward_top8(&self, x_616: &[f32]) -> Result<([f64; TOPK_SLICE], [f64; TOPK_SLICE])> {
        ensure!(
            x_616.len() == TOPK_SLICE * TOP8_ENRICHED_DIM,
            "x_616 must be 8 x 616"
        );
        let mut x_norm = vec![0.0_f32; TOPK_SLICE * TOP8_ENRICHED_DIM];
        let mut base_neg_ck_z = [0.0_f32; TOPK_SLICE];
        for j in 0..TOPK_SLICE {
            let src = &x_616[j * TOP8_ENRICHED_DIM..(j + 1) * TOP8_ENRICHED_DIM];
            let dst = &mut x_norm[j * TOP8_ENRICHED_DIM..(j + 1) * TOP8_ENRICHED_DIM];
            for c in 0..TOP8_ENRICHED_DIM {
                dst[c] = ((src[c] - self.feat_mean[c]) / self.feat_std[c]).clamp(-10.0, 10.0);
            }
            dst[2] = src[2];
            base_neg_ck_z[j] = -src[2];
        }

        let h0 = self.in_norm.forward(&x_norm, TOPK_SLICE);
        let h1 = self.in_proj_0.forward(&h0, TOPK_SLICE);
        let mut h2 = self.in_proj_1.forward(&h1, TOPK_SLICE);
        silu_inplace(&mut h2);
        let h3 = self.in_proj_4.forward(&h2, TOPK_SLICE);
        let mut h = self.in_proj_5.forward(&h3, TOPK_SLICE);
        silu_inplace(&mut h);

        // 4-head MultiheadAttention (`embed_dim=128`, `num_heads=4`, `head_dim=32`)
        let qkv_proj = LinearLayer {
            weight: self.slot_attn_in_weight.clone(),
            bias: Some(self.slot_attn_in_bias.clone()),
            in_dim: 128,
            out_dim: 384,
        };
        let qkv = qkv_proj.forward(&h, TOPK_SLICE);
        let mut attn_concat = vec![0.0_f32; TOPK_SLICE * 128];
        let scale = 1.0_f64 / (32.0_f64).sqrt();
        for head in 0..4 {
            let h_off = head * 32;
            for qi in 0..TOPK_SLICE {
                let q_slice = &qkv[qi * 384 + h_off..qi * 384 + h_off + 32];
                let mut logits = [0.0_f64; TOPK_SLICE];
                let mut max_l = f64::NEG_INFINITY;
                for ki in 0..TOPK_SLICE {
                    let k_slice = &qkv[ki * 384 + 128 + h_off..ki * 384 + 128 + h_off + 32];
                    let mut dot = 0.0_f64;
                    for d in 0..32 {
                        dot += f64::from(q_slice[d]) * f64::from(k_slice[d]);
                    }
                    let l = dot * scale;
                    logits[ki] = l;
                    if l > max_l {
                        max_l = l;
                    }
                }
                let mut sum_exp = 0.0_f64;
                for l in logits.iter_mut() {
                    *l = (*l - max_l).exp();
                    sum_exp += *l;
                }
                for d in 0..32 {
                    let mut acc = 0.0_f64;
                    for ki in 0..TOPK_SLICE {
                        let prob = logits[ki] / sum_exp;
                        let v_val = qkv[ki * 384 + 256 + h_off + d];
                        acc += prob * f64::from(v_val);
                    }
                    attn_concat[qi * 128 + h_off + d] = acc as f32;
                }
            }
        }
        let attn_out = self.slot_attn_out.forward(&attn_concat, TOPK_SLICE);
        let mut h_plus_attn = vec![0.0_f32; TOPK_SLICE * 128];
        for i in 0..h_plus_attn.len() {
            h_plus_attn[i] = h[i] + attn_out[i];
        }
        let h_ctx = self.post_norm.forward(&h_plus_attn, TOPK_SLICE);

        let mut pair_h = vec![0.0_f32; TOPK_SLICE * 256];
        for j in 0..TOPK_SLICE {
            let row = &mut pair_h[j * 256..(j + 1) * 256];
            row[..128].copy_from_slice(&h_ctx[j * 128..(j + 1) * 128]);
            for d in 0..128 {
                row[128 + d] = h_ctx[j * 128 + d] - h_ctx[d];
            }
        }

        let mut sh0 = self.score_head_0.forward(&pair_h, TOPK_SLICE);
        silu_inplace(&mut sh0);
        let delta_score = self.score_head_2.forward(&sh0, TOPK_SLICE);

        let mut ih0 = self.inv_head_0.forward(&pair_h, TOPK_SLICE);
        silu_inplace(&mut ih0);
        let inv_logits = self.inv_head_2.forward(&ih0, TOPK_SLICE);

        let mut scores = [0.0_f64; TOPK_SLICE];
        let mut inv_probs = [0.0_f64; TOPK_SLICE];
        for j in 0..TOPK_SLICE {
            scores[j] = f64::from(0.25 * base_neg_ck_z[j] + delta_score[j]);
            inv_probs[j] = f64::from(sigmoid_f32(inv_logits[j]));
        }
        Ok((scores, inv_probs))
    }
}

pub fn standardize_scores_8(scores: &[f64; TOPK_SLICE]) -> [f64; TOPK_SLICE] {
    let mean = scores.iter().sum::<f64>() / (TOPK_SLICE as f64);
    let sd = (scores.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (TOPK_SLICE as f64))
        .sqrt()
        .max(1e-6);
    std::array::from_fn(|j| (scores[j] - mean) / sd)
}

pub fn apply_soft_and_hard_veto_8(
    scores: &[f64; TOPK_SLICE],
    p_inv: &[f64; TOPK_SLICE],
    lam: f64,
    tau: f64,
) -> [f64; TOPK_SLICE] {
    let z = standardize_scores_8(scores);
    let mut out = z;
    for j in 1..TOPK_SLICE {
        out[j] = z[j] - lam * p_inv[j];
        if tau < 1.0 && p_inv[j] > tau {
            out[j] = -1e9;
        }
    }
    out
}

pub fn select_pick_from_scores_8(
    scores: &[f64; TOPK_SLICE],
    p_inv: &[f64; TOPK_SLICE],
    lam: f64,
    tau: f64,
    margin: f64,
) -> usize {
    if margin.is_infinite() {
        return 0;
    }
    let s_prime = apply_soft_and_hard_veto_8(scores, p_inv, lam, tau);
    let mut best_j = 0usize;
    let mut best_val = s_prime[0];
    for (j, &val) in s_prime.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_j = j;
        }
    }
    if best_j > 0 && (best_val - s_prime[0]) > margin {
        best_j
    } else {
        0
    }
}

#[derive(Clone, Debug)]
pub struct EnrichedTop8Packet {
    pub top8_indices: [usize; TOPK_SLICE],
    pub x_568: Vec<f32>,
    pub x_616: Vec<f32>,
    pub top_zgap: f64,
}

/// Pure-Rust implementation of `build_enriched_topk_packet_for_context` (`568D` base + `48D` H256).
#[allow(clippy::too_many_arguments)]
pub fn build_enriched_top8_features_616(
    rcost: &[f64],
    aprior: &[f64],
    prog_full: &[f32],
    sem24_full: &[f32],
    proj16_full: &[f32],
    dope629: &[f32],
    hidden256: &[f32],
    col_lm_64x17: &[f32],
    col_mask_64: &[f32],
    lm48_z: &[f32],
    s3_ctx_vec16: &[f32; 16],
    dp_f_full: &[f64],
    dp_g_full: &[f64],
    h_cost_full: &[f64],
    gate_risk_full: &[f32],
    sp_prior_full: &[f32],
    n_cand: usize,
) -> Result<EnrichedTop8Packet> {
    ensure!(
        rcost.len() == n_cand
            && aprior.len() == n_cand
            && prog_full.len() == n_cand * 412
            && sem24_full.len() == n_cand * 24
            && proj16_full.len() == n_cand * 16
            && dp_f_full.len() == n_cand
            && dp_g_full.len() == n_cand
            && h_cost_full.len() == n_cand
            && gate_risk_full.len() == n_cand
            && sp_prior_full.len() == n_cand,
        "invalid candidate pool dimensions in build_enriched_top8_features_616"
    );

    let mut order: Vec<usize> = (0..n_cand).collect();
    order.sort_by(|&a, &b| rcost[a].total_cmp(&rcost[b]).then_with(|| a.cmp(&b)));
    let top8_indices: [usize; TOPK_SLICE] = std::array::from_fn(|j| order[j]);

    let r_mean = rcost.iter().sum::<f64>() / (n_cand as f64);
    let r_sd = (rcost
        .iter()
        .map(|&v| (v - r_mean) * (v - r_mean))
        .sum::<f64>()
        / (n_cand as f64))
        .sqrt()
        .max(1e-12);
    let r_z_all: Vec<f64> = rcost.iter().map(|&v| (v - r_mean) / r_sd).collect();

    let ck: [f64; TOPK_SLICE] = std::array::from_fn(|j| rcost[top8_indices[j]]);
    let ck_mean = ck.iter().sum::<f64>() / (TOPK_SLICE as f64);
    let ck_sd = (ck
        .iter()
        .map(|&v| (v - ck_mean) * (v - ck_mean))
        .sum::<f64>()
        / (TOPK_SLICE as f64))
        .sqrt()
        .max(1e-6);
    let ck_z: [f64; TOPK_SLICE] = std::array::from_fn(|j| (ck[j] - ck_mean) / ck_sd);
    let zk: [f64; TOPK_SLICE] = std::array::from_fn(|j| r_z_all[top8_indices[j]]);

    let softmax_neg = |vals: &[f64; TOPK_SLICE], tau: f64| -> [f64; TOPK_SLICE] {
        let min_v = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let mut exps = [0.0_f64; TOPK_SLICE];
        let mut sum = 0.0_f64;
        for j in 0..TOPK_SLICE {
            exps[j] = (-(vals[j] - min_v) / tau).exp();
            sum += exps[j];
        }
        std::array::from_fn(|j| exps[j] / sum)
    };
    let sm_tau1 = softmax_neg(&zk, 1.0);
    let sm_tau02 = softmax_neg(&zk, 0.2);
    let sm_ck = softmax_neg(&ck_z, 1.0);

    let ap_mean = aprior.iter().sum::<f64>() / (n_cand as f64);
    let ap_sd = (aprior
        .iter()
        .map(|&v| (v - ap_mean) * (v - ap_mean))
        .sum::<f64>()
        / (n_cand as f64))
        .sqrt()
        .max(1e-6);
    let ap_z_all: Vec<f64> = aprior.iter().map(|&v| (v - ap_mean) / ap_sd).collect();

    let ctx_top2_zgap = (ck[1] - ck[0]) / r_sd;
    let ctx_topk_spread_z = (ck[TOPK_SLICE - 1] - ck[0]) / r_sd;

    // Sem24 and Proj16 pool standardization
    let mut sem_mean = [0.0_f64; 24];
    let mut sem_std = [0.0_f64; 24];
    for d in 0..24 {
        let m = (0..n_cand)
            .map(|i| f64::from(sem24_full[i * 24 + d]))
            .sum::<f64>()
            / (n_cand as f64);
        let s = ((0..n_cand)
            .map(|i| {
                let diff = f64::from(sem24_full[i * 24 + d]) - m;
                diff * diff
            })
            .sum::<f64>()
            / (n_cand as f64))
            .sqrt()
            + 1e-5;
        sem_mean[d] = m;
        sem_std[d] = s;
    }

    let mut proj_mean = [0.0_f64; 16];
    let mut proj_std = [0.0_f64; 16];
    for d in 0..16 {
        let m = (0..n_cand)
            .map(|i| f64::from(proj16_full[i * 16 + d]))
            .sum::<f64>()
            / (n_cand as f64);
        let s = ((0..n_cand)
            .map(|i| {
                let diff = f64::from(proj16_full[i * 16 + d]) - m;
                diff * diff
            })
            .sum::<f64>()
            / (n_cand as f64))
            .sqrt()
            + 1e-5;
        proj_mean[d] = m;
        proj_std[d] = s;
    }

    let mut sem_z_topk = [[0.0_f32; 24]; TOPK_SLICE];
    let mut proj_z_topk = [[0.0_f32; 16]; TOPK_SLICE];
    let mut proj_unit_topk = [[0.0_f64; 16]; TOPK_SLICE];
    let mut sem_topk = [[0.0_f64; 24]; TOPK_SLICE];
    let mut sem_top8_centroid = [0.0_f64; 24];

    for j in 0..TOPK_SLICE {
        let idx = top8_indices[j];
        for d in 0..24 {
            let v = f64::from(sem24_full[idx * 24 + d]);
            sem_topk[j][d] = v;
            sem_top8_centroid[d] += v / 8.0;
            sem_z_topk[j][d] = ((v - sem_mean[d]) / sem_std[d]).clamp(-5.0, 5.0) as f32;
        }
        let mut p_norm_sq = 0.0_f32;
        for d in 0..16 {
            let v = f64::from(proj16_full[idx * 16 + d]);
            let pz = ((v - proj_mean[d]) / proj_std[d]).clamp(-5.0, 5.0) as f32;
            proj_z_topk[j][d] = pz;
            p_norm_sq += pz * pz;
        }
        let p_norm = p_norm_sq.sqrt().max(1e-6);
        for d in 0..16 {
            proj_unit_topk[j][d] = f64::from(proj_z_topk[j][d] / p_norm);
        }
    }
    let centroid_norm = sem_top8_centroid
        .iter()
        .map(|&v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1e-6);
    for v in sem_top8_centroid.iter_mut() {
        *v /= centroid_norm;
    }

    let mut est_ids = [-1_i32; TOPK_SLICE];
    let mut op_counts = [0.0_f64; TOPK_SLICE];
    for j in 0..TOPK_SLICE {
        let p_row = &prog_full[top8_indices[j] * 412..(top8_indices[j] + 1) * 412];
        let mut max_est = 0.0_f32;
        let mut arg_est = -1_i32;
        for e in 0..5 {
            let v = p_row[24 + e];
            if v > max_est {
                max_est = v;
                arg_est = e as i32;
            }
        }
        est_ids[j] = arg_est;
        op_counts[j] = p_row[..23].iter().sum::<f32>() as f64;
    }

    // 76D base context vector (`lm48_z` + `dope629[:25]` + `d_summary[3]`)
    let d_mean = dope629.iter().sum::<f32>() / (DOPE_DIM as f32);
    let d_std = (dope629
        .iter()
        .map(|&v| (v - d_mean) * (v - d_mean))
        .sum::<f32>()
        / (DOPE_DIM as f32))
        .sqrt();
    let d_rms = (dope629.iter().map(|&v| v * v).sum::<f32>() / (DOPE_DIM as f32)).sqrt();
    let mut ctx_vec76 = Vec::with_capacity(76);
    ctx_vec76.extend_from_slice(lm48_z);
    ctx_vec76.extend_from_slice(&dope629[..25]);
    ctx_vec76.extend_from_slice(&[d_mean, d_std, d_rms]);

    // 36D extra context vector (`norms_11` + `pre_condense_stats_9` + `s3_ctx_vec16`)
    let norms_11 = H256FoldScaler::compute_block_log_norms(dope629);
    let hid_mean = hidden256.iter().sum::<f32>() / (HIDDEN_DIM as f32);
    let hid_std = (hidden256
        .iter()
        .map(|&v| (v - hid_mean) * (v - hid_mean))
        .sum::<f32>()
        / (HIDDEN_DIM as f32))
        .sqrt();
    let hid_rms = hidden256.iter().map(|&v| v * v).sum::<f32>().sqrt() / 16.0;
    let col_mask_mean = col_mask_64.iter().sum::<f32>() / (COL_SLOTS as f32);

    let mut active_col_vals = Vec::new();
    let mut active_colgate_vals = Vec::new();
    for c in 0..COL_SLOTS {
        if col_mask_64[c] > 0.5 {
            active_col_vals
                .extend_from_slice(&col_lm_64x17[c * COL_RAW_DIM..(c + 1) * COL_RAW_DIM]);
            active_colgate_vals.push(dope629[565 + c]);
        }
    }
    if active_col_vals.is_empty() {
        active_col_vals.extend_from_slice(&col_lm_64x17[..COL_RAW_DIM]);
    }
    let act_col_mean = active_col_vals.iter().sum::<f32>() / (active_col_vals.len() as f32);
    let act_col_std = (active_col_vals
        .iter()
        .map(|&v| (v - act_col_mean) * (v - act_col_mean))
        .sum::<f32>()
        / (active_col_vals.len() as f32))
        .sqrt();
    let d_mid = &dope629[256..565];
    let d_mid_mean = d_mid.iter().sum::<f32>() / (d_mid.len() as f32);
    let d_mid_std = (d_mid
        .iter()
        .map(|&v| (v - d_mid_mean) * (v - d_mid_mean))
        .sum::<f32>()
        / (d_mid.len() as f32))
        .sqrt();
    let act_colgate_mean = if active_colgate_vals.is_empty() {
        0.0
    } else {
        active_colgate_vals.iter().sum::<f32>() / (active_colgate_vals.len() as f32)
    };

    let mut ctx_extra_36 = Vec::with_capacity(36);
    ctx_extra_36.extend_from_slice(&norms_11);
    ctx_extra_36.extend_from_slice(&[
        hid_mean,
        hid_std,
        hid_rms,
        col_mask_mean,
        act_col_mean,
        act_col_std,
        d_mid_mean,
        d_mid_std,
        act_colgate_mean,
    ]);
    ctx_extra_36.extend_from_slice(s3_ctx_vec16);

    let dp_f_k: [f64; TOPK_SLICE] = std::array::from_fn(|j| dp_f_full[top8_indices[j]]);
    let dp_g_k: [f64; TOPK_SLICE] = std::array::from_fn(|j| dp_g_full[top8_indices[j]]);
    let hc_k: [f64; TOPK_SLICE] = std::array::from_fn(|j| h_cost_full[top8_indices[j]]);
    let gr_k: [f64; TOPK_SLICE] =
        std::array::from_fn(|j| f64::from(gate_risk_full[top8_indices[j]]));
    let sp_k: [f64; TOPK_SLICE] =
        std::array::from_fn(|j| f64::from(sp_prior_full[top8_indices[j]]));
    let comb_k: [f64; TOPK_SLICE] =
        std::array::from_fn(|j| ck[j] + 0.025 * dp_f_k[j] + 0.025 * dp_g_k[j] + 0.010 * hc_k[j]);

    let dp_f_z = standardize_scores_8(&dp_f_k);
    let dp_g_z = standardize_scores_8(&dp_g_k);
    let hc_z = standardize_scores_8(&hc_k);
    let sp_z = standardize_scores_8(&sp_k);
    let comb_z = standardize_scores_8(&comb_k);

    let mut x_568 = vec![0.0_f32; TOPK_SLICE * TOP8_BASE_DIM];
    let mut x_616 = vec![0.0_f32; TOPK_SLICE * TOP8_ENRICHED_DIM];

    for j in 0..TOPK_SLICE {
        let others: Vec<usize> = (0..TOPK_SLICE).filter(|&o| o != j).collect();
        let mut base_row = Vec::with_capacity(TOP8_BASE_DIM);
        base_row.extend_from_slice(&[
            ck[j] as f32,
            zk[j] as f32,
            ck_z[j] as f32,
            j as f32,
            (j as f32) / ((TOPK_SLICE - 1) as f32),
            if j == 0 { 1.0 } else { 0.0 },
            (ck[j] - ck[0]) as f32,
            if j == 0 {
                0.0
            } else {
                ((ck[j] - ck[0]) / r_sd) as f32
            },
            ctx_top2_zgap as f32,
            ctx_topk_spread_z as f32,
            sm_tau1[j] as f32,
            sm_tau02[j] as f32,
            sm_ck[j] as f32,
            ap_z_all[top8_indices[j]] as f32,
        ]);
        for &o in &others {
            base_row.push((ck[j] - ck[o]) as f32);
        }
        for &o in &others {
            base_row.push(((ck[j] - ck[o]) / r_sd) as f32);
        }

        let cos_sem = |a: &[f64; 24], b: &[f64; 24]| -> f64 {
            (0..24).map(|d| a[d] * b[d]).sum::<f64>().clamp(-1.0, 1.0)
        };
        let cos_proj = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            (0..16).map(|d| a[d] * b[d]).sum::<f64>().clamp(-1.0, 1.0)
        };

        let mut cos_sem_other = [0.0_f64; TOPK_SLICE - 1];
        let mut cos_proj_other = [0.0_f64; TOPK_SLICE - 1];
        for (idx_o, &o) in others.iter().enumerate() {
            cos_sem_other[idx_o] = cos_sem(&sem_topk[j], &sem_topk[o]);
            cos_proj_other[idx_o] = cos_proj(&proj_unit_topk[j], &proj_unit_topk[o]);
        }
        let mean_sem_o = cos_sem_other.iter().sum::<f64>() / 7.0;
        let max_sem_o = cos_sem_other
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_proj_o = cos_proj_other.iter().sum::<f64>() / 7.0;
        let max_proj_o = cos_proj_other
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut sum_rev03 = 0.0_f64;
        for r in 0..4 {
            let s_r: [f64; 24] = std::array::from_fn(|d| f64::from(sem24_full[r * 24 + d]));
            sum_rev03 += cos_sem(&s_r, &sem_topk[j]);
        }
        let same_est_count = (0..TOPK_SLICE)
            .filter(|&o| est_ids[o] == est_ids[j])
            .count() as f32;

        base_row.extend_from_slice(&[
            cos_sem(&sem_topk[j], &sem_topk[0]) as f32,
            mean_sem_o as f32,
            max_sem_o as f32,
            cos_proj(&proj_unit_topk[j], &proj_unit_topk[0]) as f32,
            mean_proj_o as f32,
            max_proj_o as f32,
            cos_sem(&sem_topk[j], &sem_top8_centroid) as f32,
            (sum_rev03 / 4.0) as f32,
            op_counts[j] as f32,
            (op_counts[j] - op_counts[0]) as f32,
            if est_ids[j] == est_ids[0] { 1.0 } else { 0.0 },
            same_est_count,
        ]);
        base_row.extend_from_slice(&prog_full[top8_indices[j] * 412..(top8_indices[j] + 1) * 412]);
        base_row.extend_from_slice(&sem_z_topk[j]);
        base_row.extend_from_slice(&proj_z_topk[j]);
        base_row.extend_from_slice(&ctx_vec76);
        debug_assert_eq!(base_row.len(), TOP8_BASE_DIM);

        let cand_h256_12 = [
            dp_f_z[j] as f32,
            (dp_f_z[j] - dp_f_z[0]) as f32,
            dp_g_z[j] as f32,
            (dp_g_z[j] - dp_g_z[0]) as f32,
            hc_z[j] as f32,
            (hc_k[j] - hc_k[0]) as f32,
            gr_k[j] as f32,
            (gr_k[j] - gr_k[0]) as f32,
            sp_z[j] as f32,
            (sp_k[j] - sp_k[0]) as f32,
            comb_z[j] as f32,
            ((comb_k[j] - comb_k[0]) / r_sd) as f32,
        ];

        x_568[j * TOP8_BASE_DIM..(j + 1) * TOP8_BASE_DIM].copy_from_slice(&base_row);
        let dst_616 = &mut x_616[j * TOP8_ENRICHED_DIM..(j + 1) * TOP8_ENRICHED_DIM];
        dst_616[..TOP8_BASE_DIM].copy_from_slice(&base_row);
        dst_616[TOP8_BASE_DIM..TOP8_MID_DIM].copy_from_slice(&cand_h256_12);
        dst_616[TOP8_MID_DIM..TOP8_ENRICHED_DIM].copy_from_slice(&ctx_extra_36);
    }

    Ok(EnrichedTop8Packet {
        top8_indices,
        x_568,
        x_616,
        top_zgap: ctx_top2_zgap,
    })
}

#[derive(Clone, Debug)]
pub struct Top8PredictionReadout {
    pub nn_scores: [f64; TOPK_SLICE],
    pub nn_inv_probs: [f64; TOPK_SLICE],
    pub lgbm_ranker_568: [f64; TOPK_SLICE],
    pub lgbm_ranker_580: [f64; TOPK_SLICE],
    pub lgbm_ranker_616: [f64; TOPK_SLICE],
    pub lgbm_clf_568: [f64; TOPK_SLICE],
    pub lgbm_clf_580: [f64; TOPK_SLICE],
    pub lgbm_clf_616: [f64; TOPK_SLICE],
    pub bagged_scores: [f64; TOPK_SLICE],
    pub bagged_pinv: [f64; TOPK_SLICE],
    pub static_pick_j: usize,
}

#[derive(Clone, Debug)]
pub struct H256Stage2EnsembleBundle {
    pub soup_bridge: H256PreCondenseBridge,
    pub fold_bridges: [H256PreCondenseBridge; 5],
    pub full_train_scaler: H256FoldScaler,
    pub alpha_prior: f64,
    pub alpha_traj: f64,
    pub critic: H256CriticModels,
}

impl H256Stage2EnsembleBundle {
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self> {
        let st = SafetensorsMap::deserialize(bytes)?;
        let hp = st.f32_vec("ensemble_hyperparams", &[2])?;
        Ok(Self {
            soup_bridge: H256PreCondenseBridge::from_safetensors(&st, "soup_bridge")?,
            fold_bridges: [
                H256PreCondenseBridge::from_safetensors(&st, "fold0_bridge")?,
                H256PreCondenseBridge::from_safetensors(&st, "fold1_bridge")?,
                H256PreCondenseBridge::from_safetensors(&st, "fold2_bridge")?,
                H256PreCondenseBridge::from_safetensors(&st, "fold3_bridge")?,
                H256PreCondenseBridge::from_safetensors(&st, "fold4_bridge")?,
            ],
            full_train_scaler: H256FoldScaler::from_safetensors(&st, "full_train_scaler")?,
            alpha_prior: f64::from(hp[0]),
            alpha_traj: f64::from(hp[1]),
            critic: H256CriticModels::from_safetensors(&st, "critic")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct H256ProductionBundle {
    pub s3_encoder: H256Stage1ContextEncoder,
    pub s3_scaler: H256FoldScaler,
    pub stage2_bridge: H256PreCondenseBridge,
    pub stage2_arm_f_soup: H256PreCondenseBridge,
    pub stage2_arm_g_soup: H256PreCondenseBridge,
    pub stage2_scaler: H256FoldScaler,
    pub alpha_prior: f64,
    pub alpha_traj: f64,
    pub critic: H256CriticModels,
    pub top8_nn: H256Top8NeuralReranker,
    pub ranker_568: FlatGbdtModel,
    pub ranker_580: FlatGbdtModel,
    pub ranker_616: FlatGbdtModel,
    pub clf_568: FlatGbdtModel,
    pub clf_580: FlatGbdtModel,
    pub clf_616: FlatGbdtModel,
}

impl H256ProductionBundle {
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self> {
        let st = SafetensorsMap::deserialize(bytes)?;
        let hp = st.f32_vec("stage2_hyperparams", &[2])?;
        Ok(Self {
            s3_encoder: H256Stage1ContextEncoder::from_safetensors(&st, "s3_encoder")?,
            s3_scaler: H256FoldScaler::from_safetensors(&st, "s3_scaler")?,
            stage2_bridge: H256PreCondenseBridge::from_safetensors(&st, "stage2_bridge")?,
            stage2_arm_f_soup: H256PreCondenseBridge::from_safetensors(&st, "stage2_arm_f_soup")?,
            stage2_arm_g_soup: H256PreCondenseBridge::from_safetensors(&st, "stage2_arm_g_soup")?,
            stage2_scaler: H256FoldScaler::from_safetensors(&st, "stage2_scaler")?,
            alpha_prior: f64::from(hp[0]),
            alpha_traj: f64::from(hp[1]),
            critic: H256CriticModels::from_safetensors(&st, "critic")?,
            top8_nn: H256Top8NeuralReranker::from_safetensors(&st, "top8_nn")?,
            ranker_568: FlatGbdtModel::from_safetensors(&st, "gbdt.ranker_568")?,
            ranker_580: FlatGbdtModel::from_safetensors(&st, "gbdt.ranker_580")?,
            ranker_616: FlatGbdtModel::from_safetensors(&st, "gbdt.ranker_616")?,
            clf_568: FlatGbdtModel::from_safetensors(&st, "gbdt.clf_568")?,
            clf_580: FlatGbdtModel::from_safetensors(&st, "gbdt.clf_580")?,
            clf_616: FlatGbdtModel::from_safetensors(&st, "gbdt.clf_616")?,
        })
    }

    pub fn predict_top8_from_matrices(
        &self,
        x_568: &[f32],
        x_616: &[f32],
    ) -> Result<Top8PredictionReadout> {
        ensure!(
            x_568.len() == TOPK_SLICE * TOP8_BASE_DIM
                && x_616.len() == TOPK_SLICE * TOP8_ENRICHED_DIM,
            "invalid Top-8 feature matrix dimensions"
        );
        let mut x_580 = vec![0.0_f32; TOPK_SLICE * TOP8_MID_DIM];
        for j in 0..TOPK_SLICE {
            x_580[j * TOP8_MID_DIM..(j + 1) * TOP8_MID_DIM].copy_from_slice(
                &x_616[j * TOP8_ENRICHED_DIM..j * TOP8_ENRICHED_DIM + TOP8_MID_DIM],
            );
        }

        let (nn_scores, nn_inv_probs) = self.top8_nn.forward_top8(x_616)?;
        let r568: [f64; TOPK_SLICE] = self
            .ranker_568
            .predict_batch(x_568, TOPK_SLICE, TOP8_BASE_DIM)
            .try_into()
            .expect("8 slots");
        let r580: [f64; TOPK_SLICE] = self
            .ranker_580
            .predict_batch(&x_580, TOPK_SLICE, TOP8_MID_DIM)
            .try_into()
            .expect("8 slots");
        let r616: [f64; TOPK_SLICE] = self
            .ranker_616
            .predict_batch(x_616, TOPK_SLICE, TOP8_ENRICHED_DIM)
            .try_into()
            .expect("8 slots");

        let p568: [f64; TOPK_SLICE] = self
            .clf_568
            .predict_batch(x_568, TOPK_SLICE, TOP8_BASE_DIM)
            .try_into()
            .expect("8 slots");
        let p580: [f64; TOPK_SLICE] = self
            .clf_580
            .predict_batch(&x_580, TOPK_SLICE, TOP8_MID_DIM)
            .try_into()
            .expect("8 slots");
        let p616: [f64; TOPK_SLICE] = self
            .clf_616
            .predict_batch(x_616, TOPK_SLICE, TOP8_ENRICHED_DIM)
            .try_into()
            .expect("8 slots");

        let b568 = standardize_scores_8(&r568);
        let b580 = standardize_scores_8(&r580);
        let b616 = standardize_scores_8(&r616);
        let bnn = standardize_scores_8(&nn_scores);
        let bridge_neg: [f64; TOPK_SLICE] =
            std::array::from_fn(|j| -f64::from(x_616[j * TOP8_ENRICHED_DIM + 578]));
        let bbr = standardize_scores_8(&bridge_neg);

        let mut bagged_scores = [0.0_f64; TOPK_SLICE];
        let mut bagged_pinv = [0.0_f64; TOPK_SLICE];
        for j in 0..TOPK_SLICE {
            bagged_scores[j] =
                0.40 * b568[j] + 0.30 * b580[j] + 0.15 * b616[j] + 0.10 * bnn[j] + 0.05 * bbr[j];
            bagged_pinv[j] =
                0.45 * p568[j] + 0.30 * p580[j] + 0.15 * p616[j] + 0.10 * nn_inv_probs[j];
        }
        let static_pick_j = select_pick_from_scores_8(&bagged_scores, &bagged_pinv, 1.5, 1.0, 0.05);

        Ok(Top8PredictionReadout {
            nn_scores,
            nn_inv_probs,
            lgbm_ranker_568: r568,
            lgbm_ranker_580: r580,
            lgbm_ranker_616: r616,
            lgbm_clf_568: p568,
            lgbm_clf_580: p580,
            lgbm_clf_616: p616,
            bagged_scores,
            bagged_pinv,
            static_pick_j,
        })
    }

    pub fn aggregate_subsample_batch_picks(
        view_readouts: &[Top8PredictionReadout],
        view_weights: Option<&[f64]>,
        top8_indices: &[usize; TOPK_SLICE],
    ) -> Result<SubsampleBatchConsensus> {
        ensure!(
            !view_readouts.is_empty(),
            "view_readouts must contain at least one subsample view"
        );
        let b = view_readouts.len();
        let default_weights: Vec<f64> = (0..b)
            .map(|idx| if idx == 0 { 2.0 } else { 1.0 })
            .collect();
        let weights: &[f64] = match view_weights {
            Some(w) => {
                ensure!(
                    w.len() == b,
                    "view_weights length {} does not match view_readouts length {}",
                    w.len(),
                    b
                );
                w
            }
            None => &default_weights,
        };
        let w_sum: f64 = weights.iter().copied().sum();
        ensure!(w_sum > 0.0 && w_sum.is_finite(), "view_weights sum must be positive and finite");

        let mut consensus_bagged_scores = [0.0_f64; TOPK_SLICE];
        let mut consensus_bagged_pinv = [0.0_f64; TOPK_SLICE];
        let mut view_pick_j = Vec::with_capacity(b);
        let mut view_pick_indices = Vec::with_capacity(b);

        for (v_idx, ro) in view_readouts.iter().enumerate() {
            let w = weights[v_idx] / w_sum;
            for j in 0..TOPK_SLICE {
                consensus_bagged_scores[j] += w * ro.bagged_scores[j];
                consensus_bagged_pinv[j] += w * ro.bagged_pinv[j];
            }
            view_pick_j.push(ro.static_pick_j);
            view_pick_indices.push(top8_indices[ro.static_pick_j]);
        }

        let consensus_pick_j = select_pick_from_scores_8(
            &consensus_bagged_scores,
            &consensus_bagged_pinv,
            1.5,
            1.0,
            0.05,
        );
        let consensus_pick_index = top8_indices[consensus_pick_j];

        // Deduplicated diverse Generation-0 GP seed indices:
        // 1. Consensus Top-1 pick
        // 2. Individual subsample view picks in view priority order
        // 3. Remaining Top-8 candidates sorted by descending consensus_bagged_scores
        let mut gen0_seed_indices = Vec::with_capacity(TOPK_SLICE);
        gen0_seed_indices.push(consensus_pick_index);
        for &cand_idx in &view_pick_indices {
            if !gen0_seed_indices.contains(&cand_idx) {
                gen0_seed_indices.push(cand_idx);
            }
        }
        let mut slot_order: Vec<usize> = (0..TOPK_SLICE).collect();
        slot_order.sort_by(|&a, &b| {
            consensus_bagged_scores[b]
                .total_cmp(&consensus_bagged_scores[a])
                .then_with(|| a.cmp(&b))
        });
        for j in slot_order {
            let cand_idx = top8_indices[j];
            if !gen0_seed_indices.contains(&cand_idx) {
                gen0_seed_indices.push(cand_idx);
            }
        }

        Ok(SubsampleBatchConsensus {
            consensus_bagged_scores,
            consensus_bagged_pinv,
            consensus_pick_j,
            consensus_pick_index,
            view_pick_j,
            view_pick_indices,
            gen0_seed_indices,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn predict_static_top8_batch(
        &self,
        views: &[SubsampleBatchContextView<'_>],
        view_weights: Option<&[f64]>,
        r882_total_cost: &[f64],
        r882_anchor_prior: &[f64],
        program_412: &[f32],
        sem24: &[f32],
        proj16: &[f32],
        cand_q_32: &[f32],
        h_cost: &[f64],
        gate_risk: &[f32],
        sp_prior_1d: &[f32],
        n_cand: usize,
    ) -> Result<(Vec<Top8PredictionReadout>, SubsampleBatchConsensus)> {
        ensure!(
            !views.is_empty(),
            "views must contain at least one SubsampleBatchContextView"
        );
        let mut view_readouts = Vec::with_capacity(views.len());
        let mut top8_indices_ref = [0_usize; TOPK_SLICE];

        for (v_idx, v) in views.iter().enumerate() {
            let ct_s3 = self.s3_scaler.transform_context(
                v.raw_dope629,
                v.raw_hidden256,
                v.raw_col_lm_64x17,
                v.raw_col_mask_64,
            )?;
            let s3_ctx_vec16 = self.s3_encoder.extract_s3_ctx_vec16(&ct_s3);
            let std_ctx = self.stage2_scaler.transform_context(
                v.raw_dope629,
                v.raw_hidden256,
                v.raw_col_lm_64x17,
                v.raw_col_mask_64,
            )?;
            let dp_f_out = self.stage2_arm_f_soup.forward_prior_and_context(
                &std_ctx,
                cand_q_32,
                sem24,
                gate_risk,
                sp_prior_1d,
                n_cand,
            )?;
            let dp_g_out = self.stage2_arm_g_soup.forward_prior_and_context(
                &std_ctx,
                cand_q_32,
                sem24,
                gate_risk,
                sp_prior_1d,
                n_cand,
            )?;
            let pkt = build_enriched_top8_features_616(
                r882_total_cost,
                r882_anchor_prior,
                program_412,
                sem24,
                proj16,
                v.raw_dope629,
                v.raw_hidden256,
                v.raw_col_lm_64x17,
                v.raw_col_mask_64,
                v.raw_lm48_z,
                &s3_ctx_vec16,
                &dp_f_out.delta_prior,
                &dp_g_out.delta_prior,
                h_cost,
                gate_risk,
                sp_prior_1d,
                n_cand,
            )?;
            if v_idx == 0 {
                top8_indices_ref = pkt.top8_indices;
            }
            let ro = self.predict_top8_from_matrices(&pkt.x_568, &pkt.x_616)?;
            view_readouts.push(ro);
        }

        let consensus = Self::aggregate_subsample_batch_picks(
            &view_readouts,
            view_weights,
            &top8_indices_ref,
        )?;
        Ok((view_readouts, consensus))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SubsampleBatchContextView<'a> {
    pub raw_dope629: &'a [f32],
    pub raw_hidden256: &'a [f32],
    pub raw_col_lm_64x17: &'a [f32],
    pub raw_col_mask_64: &'a [f32],
    pub raw_lm48_z: &'a [f32],
}

#[derive(Clone, Debug)]
pub struct SubsampleBatchConsensus {
    pub consensus_bagged_scores: [f64; TOPK_SLICE],
    pub consensus_bagged_pinv: [f64; TOPK_SLICE],
    pub consensus_pick_j: usize,
    pub consensus_pick_index: usize,
    pub view_pick_j: Vec<usize>,
    pub view_pick_indices: Vec<usize>,
    pub gen0_seed_indices: Vec<usize>,
}

#[cfg(test)]
mod tests;
