//! CPU eval forward for the frozen h40b `adapter_10tok.predict_costs` graph.
//!
//! Synthetic implementation controls are not RES-07 real-golden M2 qualification.
//! Production parameters enter only through the authenticated bundle boundary.

use std::fmt;

use anyhow::{ensure, Context, Result};
use candle_core_0_11_0::{DType, Device, Tensor, D};
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;

use super::bundle::{check_tensor_header, BundleLimits, VerifiedBundle};

pub const PROG_FEATURES: usize = 412;
pub const PROJ_FEATURES: usize = 16;
pub const TOKENS: usize = 10;
pub const TOKEN_WIDTH: usize = 32;
pub const CONTEXT_FEATURES: usize = 584;
pub const REVEALED: usize = 4;
pub const SOS_FEATURES: usize = 2;
pub const SCALE: f32 = 0.4208;

/// Flattened F12 inputs for one batched ranker call.
///
/// Candidate arrays are row-major `[batch, candidates, feature]`; token arrays
/// are `[batch, 10, 32]` and masks are `[batch, 10]`, shared by the slate.
/// Context and prior inputs are the zero-valued constants specified by F12.
#[derive(Clone, Copy, Debug)]
pub struct PredictCostsInputs<'a> {
    pub batch: usize,
    pub candidates: usize,
    pub prog_412: &'a [f32],
    pub proj_16: &'a [f32],
    pub tokens: &'a [f32],
    pub token_mask: &'a [f32],
    pub revealed_indices: &'a [i64],
    pub revealed_fitness: &'a [f32],
    pub revealed_gates: &'a [f32],
    pub ctx_feat: &'a [f32],
    pub base_prior_cost: &'a [f32],
    pub sos_prior: &'a [f32],
    pub scale: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputShapeError {
    pub field: &'static str,
    pub expected: usize,
    pub actual: usize,
}

impl fmt::Display for InputShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "adapter input {} has {} elements; expected {}",
            self.field, self.actual, self.expected
        )
    }
}

impl std::error::Error for InputShapeError {}

impl PredictCostsInputs<'_> {
    /// Check the F12 flattened dimensions before converting them to backend tensors.
    pub fn validate_shapes(&self) -> std::result::Result<(), InputShapeError> {
        if self.batch == 0 {
            return Err(InputShapeError {
                field: "batch",
                expected: 1,
                actual: 0,
            });
        }
        if ![12, 20].contains(&self.candidates) {
            return Err(InputShapeError {
                field: "candidates",
                expected: 12,
                actual: self.candidates,
            });
        }
        let bn = self.batch.saturating_mul(self.candidates);
        let expected = [
            (
                "prog_412",
                bn.saturating_mul(PROG_FEATURES),
                self.prog_412.len(),
            ),
            (
                "proj_16",
                bn.saturating_mul(PROJ_FEATURES),
                self.proj_16.len(),
            ),
            (
                "tokens",
                self.batch.saturating_mul(TOKENS * TOKEN_WIDTH),
                self.tokens.len(),
            ),
            (
                "token_mask",
                self.batch.saturating_mul(TOKENS),
                self.token_mask.len(),
            ),
            (
                "revealed_indices",
                self.batch.saturating_mul(REVEALED),
                self.revealed_indices.len(),
            ),
            (
                "revealed_fitness",
                self.batch.saturating_mul(REVEALED),
                self.revealed_fitness.len(),
            ),
            (
                "revealed_gates",
                self.batch.saturating_mul(REVEALED),
                self.revealed_gates.len(),
            ),
            (
                "ctx_feat",
                self.batch.saturating_mul(CONTEXT_FEATURES),
                self.ctx_feat.len(),
            ),
            ("base_prior_cost", bn, self.base_prior_cost.len()),
            (
                "sos_prior",
                bn.saturating_mul(SOS_FEATURES),
                self.sos_prior.len(),
            ),
        ];

        for (field, expected, actual) in expected {
            if expected != actual {
                return Err(InputShapeError {
                    field,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

/// Costs are row-major `[batch, candidates]`. KL is computed over proposals
/// only, using unit-temperature softmax(-cost) against the frozen anchor.
/// This readout does not grant M2/Q1 qualification or change admission policy.
#[derive(Debug)]
pub struct PredictedCosts {
    pub total_cost: Vec<f32>,
    pub anchor_cost: Vec<f32>,
    pub proposal_kl_to_anchor: Vec<f64>,
    pub exceeds_kl_limit: bool,
}

/// Immutable CPU parameters for the inference-only graph. Unused parent-delta,
/// evolvability and training heads are not part of `predict_costs`.
#[derive(Debug)]
pub struct CpuAdapter {
    pub(super) parameters: BTreeMap<String, Tensor>,
    pub(super) device: Device,
}

fn parameter_specs() -> Vec<(String, Vec<usize>)> {
    fn norm(specs: &mut Vec<(String, Vec<usize>)>, name: &str, width: usize) {
        specs.push((format!("{name}.weight"), vec![width]));
        specs.push((format!("{name}.bias"), vec![width]));
    }
    fn linear(
        specs: &mut Vec<(String, Vec<usize>)>,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) {
        specs.push((format!("{name}.weight"), vec![output, input]));
        if bias {
            specs.push((format!("{name}.bias"), vec![output]));
        }
    }
    let mut specs = Vec::new();
    for (name, shape) in [
        ("prog_mean", vec![412]),
        ("prog_std", vec![412]),
        ("modality_type_emb", vec![10, 32]),
        ("modality_gate_logits", vec![10]),
        ("null_hist_token", vec![1, 1, 32]),
    ] {
        specs.push((name.into(), shape));
    }
    for name in [
        "anchor.alpha_prior",
        "anchor.alpha_struct_kernel",
        "anchor.alpha_attn",
        "anchor.gate_barrier_weight",
        "mm_prior_scale",
        "null_kernel_w0_raw",
        "sem_kernel_scale_raw",
        "emb_kernel_scale_raw",
        "traj_kernel_blend",
        "traj_attn_scale",
    ] {
        specs.push((name.into(), vec![]));
    }
    for (name, width) in [
        ("anchor.ctx_in.0", 584),
        ("anchor.prog_in.0", 558),
        ("anchor.landmark_router.0", 64),
        ("anchor.prior_head.0", 256),
        ("anchor.history_token_in.1", 128),
        ("anchor.traj_out.0", 260),
        ("anchor.gate_head.0", 258),
        ("modality_norm", 32),
        ("prog_412_bottleneck.1", 16),
        ("cand_query_proj.1", 32),
        ("hist_token_proj.1", 32),
    ] {
        norm(&mut specs, name, width);
    }
    for name in ["anchor.ctx_block", "anchor.prog_block"] {
        norm(&mut specs, &format!("{name}.norm"), 128);
        linear(&mut specs, &format!("{name}.fc_gate"), 128, 256, true);
        linear(&mut specs, &format!("{name}.fc_val"), 128, 256, true);
        linear(&mut specs, &format!("{name}.fc_out"), 256, 128, true);
    }
    for (name, input, output, bias) in [
        ("anchor.ctx_in.1", 584, 128, true),
        ("anchor.prog_in.1", 558, 128, true),
        ("anchor.landmark_router.1", 64, 64, true),
        ("anchor.landmark_router.3", 64, 16, true),
        ("anchor.prior_head.1", 256, 128, true),
        ("anchor.prior_head.3", 128, 1, true),
        ("anchor.history_token_in.0", 132, 128, true),
        ("anchor.traj_out.1", 260, 64, true),
        ("anchor.traj_out.3", 64, 1, true),
        ("anchor.gate_head.1", 258, 64, true),
        ("anchor.gate_head.3", 64, 1, true),
        ("prog_412_bottleneck.0", 412, 16, true),
        ("cand_query_proj.0", 66, 32, true),
        ("ctx_bilinear_u", 32, 16, false),
        ("prog_bilinear_v", 24, 16, false),
        ("mm_prior_out.0", 50, 24, true),
        ("mm_prior_out.2", 24, 1, true),
        ("hist_token_proj.0", 60, 32, true),
        ("traj_delta_head.0", 67, 24, true),
        ("traj_delta_head.2", 24, 1, true),
    ] {
        linear(&mut specs, name, input, output, bias);
    }
    for (name, width) in [
        ("anchor.cross_attn", 128),
        ("mm_cross_attn", 32),
        ("traj_cross_attn", 32),
    ] {
        specs.push((format!("{name}.in_proj_weight"), vec![3 * width, width]));
        specs.push((format!("{name}.in_proj_bias"), vec![3 * width]));
        linear(&mut specs, &format!("{name}.out_proj"), width, width, true);
    }
    specs
}

impl CpuAdapter {
    pub fn from_bundle(bundle: &VerifiedBundle) -> Result<Self> {
        Self::from_bytes(
            bundle
                .asset("jope.safetensors")
                .context("missing authenticated jope.safetensors")?,
        )
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_on_device(bytes, &Device::Cpu)
    }

    pub(super) fn from_bytes_on_device(bytes: &[u8], device: &Device) -> Result<Self> {
        check_tensor_header(bytes, BundleLimits::default())?;
        let tensors = SafeTensors::deserialize(bytes).context("parse adapter tensors")?;
        let mut parameters = BTreeMap::new();
        for (name, shape) in parameter_specs() {
            let full = format!("adapter_10tok.{name}");
            let tensor = tensors
                .tensor(&full)
                .with_context(|| format!("missing adapter tensor {full}"))?;
            ensure!(
                tensor.dtype() == Dtype::F32 && tensor.shape() == shape,
                "invalid adapter tensor {full} dtype/shape"
            );
            let values: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().expect("F32 bytes")))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "non-finite adapter tensor {full}"
            );
            if name == "prog_std" {
                ensure!(
                    values.iter().all(|v| *v > 0.0),
                    "program standard deviations must be positive"
                );
            }
            parameters.insert(name, Tensor::from_vec(values, shape.as_slice(), device)?);
        }
        Ok(Self {
            parameters,
            device: device.clone(),
        })
    }

    fn p(&self, name: &str) -> &Tensor {
        self.parameters
            .get(name)
            .expect("validated adapter parameter")
    }
    fn scalar(&self, name: &str) -> Result<f32> {
        Ok(self.p(name).to_scalar()?)
    }
    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let y = x.matmul(&self.p(&format!("{name}.weight")).t()?.contiguous()?)?;
        Ok(
            if let Some(bias) = self.parameters.get(&format!("{name}.bias")) {
                y.broadcast_add(bias)?
            } else {
                y
            },
        )
    }
    fn norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let output_dtype = x.dtype();
        // CPU f32 reductions lose the small variance of wide offset rows.
        // Keep normalization arithmetic in f64 and round at the model boundary.
        let compute_dtype = if x.device().is_cpu() {
            DType::F64
        } else {
            output_dtype
        };
        let x = x.to_dtype(compute_dtype)?;
        let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        let denom = centered
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, 1e-5)?
            .sqrt()?;
        Ok(centered
            .broadcast_div(&denom)?
            .broadcast_mul(&self.p(&format!("{name}.weight")).to_dtype(compute_dtype)?)?
            .broadcast_add(&self.p(&format!("{name}.bias")).to_dtype(compute_dtype)?)?
            .to_dtype(output_dtype)?)
    }
    fn residual(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let h = self.norm(x, &format!("{name}.norm"))?;
        let product = (self.linear(&h, &format!("{name}.fc_gate"))?.silu()?
            * self.linear(&h, &format!("{name}.fc_val"))?)?;
        Ok((x + self.linear(&product, &format!("{name}.fc_out"))?)?)
    }
    fn attention(
        &self,
        q: &Tensor,
        kv: &Tensor,
        name: &str,
        mask: Option<&[f32]>,
    ) -> Result<Tensor> {
        let (n, width) = q.dims2()?;
        let k = kv.dim(0)?;
        let head = width / 4;
        let mut projected = Vec::new();
        for (part, input) in [(0, q), (1, kv), (2, kv)] {
            projected.push(
                input
                    .matmul(
                        &self
                            .p(&format!("{name}.in_proj_weight"))
                            .narrow(0, part * width, width)?
                            .t()?
                            .contiguous()?,
                    )?
                    .broadcast_add(&self.p(&format!("{name}.in_proj_bias")).narrow(
                        0,
                        part * width,
                        width,
                    )?)?
                    .reshape((input.dim(0)?, 4, head))?
                    .transpose(0, 1)?
                    .contiguous()?,
            );
        }
        let mut logits = projected[0]
            .matmul(&projected[1].transpose(1, 2)?.contiguous()?)?
            .affine(1.0 / (head as f64).sqrt(), 0.0)?;
        if let Some(mask) = mask {
            let values = mask
                .iter()
                .map(|v| if *v < 0.5 { f32::NEG_INFINITY } else { 0.0 })
                .collect::<Vec<_>>();
            logits = logits.broadcast_add(&Tensor::from_vec(values, (1, 1, k), &self.device)?)?;
        }
        let exp = logits
            .broadcast_sub(&logits.max_keepdim(D::Minus1)?)?
            .exp()?;
        let probabilities = exp.broadcast_div(&exp.sum_keepdim(D::Minus1)?)?;
        let output = probabilities
            .matmul(&projected[2])?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n, width))?;
        self.linear(&output, &format!("{name}.out_proj"))
    }
}

fn flat(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.flatten_all()?.to_vec1()?)
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}
fn normalized_rows(x: &[f32], width: usize, eps: f32) -> Vec<f32> {
    x.chunks_exact(width)
        .flat_map(|row| {
            let norm = dot(row, row).sqrt().max(eps);
            row.iter().map(move |v| v / norm)
        })
        .collect()
}
fn centered_scaled(x: &[f32], scale: f32) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    x.iter().map(|v| (v - mean) * scale).collect()
}
fn semantic_signature(prog: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(prog.len() / 412 * 24);
    for row in prog.chunks_exact(412) {
        let p: Vec<f32> = row.iter().map(|v| v.max(0.0)).collect();
        let mut s: Vec<f32> = p[24..29].iter().map(|v| 1.5 * v.clamp(0.0, 2.0)).collect();
        s.extend([
            (p[22] + p[1] + p[18]).ln_1p(),
            (p[15] + p[13] + p[20]).ln_1p(),
            (p[2] + p[12] + p[6]).ln_1p(),
            (p[11] + p[10] + p[16] + p[17]).ln_1p(),
            (p[19] + p[14] + p[0] + p[3] + p[4]).ln_1p(),
            p[52].ln_1p() / 6.0,
            (p[46] / 6.0).clamp(0.0, 2.5),
            (100.0 * p[50]).ln_1p() / 4.0,
            p[58].ln_1p() / 3.0,
            p[60].ln_1p() / 5.0,
            (p[34] / 7.0).clamp(0.0, 2.5),
            (p[42] / 8.0).clamp(0.0, 2.5),
            (100.0 * p[86]).ln_1p() / 5.0,
            (p[411] / 6.0).clamp(0.0, 2.0),
            p[23].ln_1p() / 6.0,
            p[..23].iter().sum::<f32>().ln_1p() / 3.0,
            p[111..122].iter().sum::<f32>().clamp(0.0, 1.5),
            p[122..132].iter().sum::<f32>().clamp(0.0, 1.5),
            if p[411] >= 2.5 { 0.8 } else { 0.0 },
        ]);
        result.extend(normalized_rows(&s, 24, 1e-8));
    }
    result
}

#[path = "adapter/predict.rs"]
mod predict_impl;

fn proposal_kl(costs: &[f32], anchor: &[f32]) -> f64 {
    fn log_probabilities(costs: &[f32]) -> Vec<f64> {
        let min = costs.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let shifted: Vec<f64> = costs.iter().map(|v| min - f64::from(*v)).collect();
        let log_sum = shifted.iter().map(|v| v.exp()).sum::<f64>().ln();
        shifted.iter().map(|v| v - log_sum).collect()
    }
    let p = log_probabilities(costs);
    let q = log_probabilities(anchor);
    p.iter()
        .zip(q)
        .map(|(p, q)| p.exp() * (p - q))
        .sum::<f64>()
        .max(0.0)
}

#[cfg(test)]
#[path = "../../tests/support/cpu_parity.rs"]
pub(super) mod real_cpu_parity;

#[cfg(test)]
mod tests;
