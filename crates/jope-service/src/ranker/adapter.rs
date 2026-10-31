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
    parameters: BTreeMap<String, Tensor>,
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
            parameters.insert(
                name,
                Tensor::from_vec(values, shape.as_slice(), &Device::Cpu)?,
            );
        }
        Ok(Self { parameters })
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
        let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        let denom = centered
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, 1e-5)?
            .sqrt()?;
        Ok(centered
            .broadcast_div(&denom)?
            .broadcast_mul(self.p(&format!("{name}.weight")))?
            .broadcast_add(self.p(&format!("{name}.bias")))?)
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
            logits = logits.broadcast_add(&Tensor::from_vec(values, (1, 1, k), &Device::Cpu)?)?;
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

    /// Inference only. Constants and the four revealed measurements follow F12;
    /// all batch rows are evaluated independently, with no state mutation.
    pub fn predict_costs(&self, inputs: PredictCostsInputs<'_>) -> Result<PredictedCosts> {
        inputs.validate_shapes()?;
        ensure!(
            inputs.scale.to_bits() == SCALE.to_bits(),
            "F12 scale must be 0.4208"
        );
        for (name, values) in [
            ("prog_412", inputs.prog_412),
            ("proj_16", inputs.proj_16),
            ("tokens", inputs.tokens),
            ("fitness", inputs.revealed_fitness),
        ] {
            ensure!(values.iter().all(|v| v.is_finite()), "non-finite {name}");
        }
        ensure!(
            inputs
                .token_mask
                .iter()
                .chain(inputs.revealed_gates)
                .all(|v| *v == 0.0 || *v == 1.0),
            "mask and gates must be binary"
        );
        ensure!(
            inputs
                .ctx_feat
                .iter()
                .chain(inputs.base_prior_cost)
                .chain(inputs.sos_prior)
                .all(|v| *v == 0.0),
            "F12 context and priors must be zero"
        );
        for revealed in inputs.revealed_indices.chunks_exact(REVEALED) {
            for (i, index) in revealed.iter().enumerate() {
                ensure!(
                    *index >= 0
                        && (*index as usize) < inputs.candidates
                        && !revealed[..i].contains(index),
                    "invalid revealed indices"
                );
            }
        }
        let mut total_cost = Vec::new();
        let mut anchor_cost = Vec::new();
        let mut proposal_kl_to_anchor = Vec::new();
        for b in 0..inputs.batch {
            let (total, anchor) = self.forward_row(&inputs, b)?;
            ensure!(
                total.iter().chain(&anchor).all(|v| v.is_finite()),
                "adapter produced non-finite costs"
            );
            proposal_kl_to_anchor.push(proposal_kl(&total[REVEALED..], &anchor[REVEALED..]));
            total_cost.extend(total);
            anchor_cost.extend(anchor);
        }
        let exceeds_kl_limit = proposal_kl_to_anchor.iter().any(|v| *v > 0.35);
        Ok(PredictedCosts {
            total_cost,
            anchor_cost,
            proposal_kl_to_anchor,
            exceeds_kl_limit,
        })
    }

    fn forward_row(
        &self,
        inputs: &PredictCostsInputs<'_>,
        b: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = inputs.candidates;
        let prog = &inputs.prog_412[b * n * 412..(b + 1) * n * 412];
        let proj = &inputs.proj_16[b * n * 16..(b + 1) * n * 16];
        let revealed: Vec<usize> = inputs.revealed_indices[b * 4..b * 4 + 4]
            .iter()
            .map(|i| *i as usize)
            .collect();
        let fitness = &inputs.revealed_fitness[b * 4..b * 4 + 4];
        let gates = &inputs.revealed_gates[b * 4..b * 4 + 4];
        let (anchor, anchor_prior) = self.anchor(prog, proj, &revealed, fitness, gates)?;
        let raw_tokens = Tensor::from_vec(
            inputs.tokens[b * 320..b * 320 + 320].to_vec(),
            (10, 32),
            &Device::Cpu,
        )?;
        let mut mask = inputs.token_mask[b * 10..b * 10 + 10].to_vec();
        mask[0] = 1.0;
        let gate = flat(self.p("modality_gate_logits"))?
            .iter()
            .zip(&mask)
            .map(|(g, m)| sigmoid(*g) * m)
            .collect();
        let tokens = self
            .norm(
                &(&raw_tokens + self.p("modality_type_emb"))?,
                "modality_norm",
            )?
            .broadcast_mul(&Tensor::from_vec(gate, (10, 1), &Device::Cpu)?)?;
        let ctx = tokens
            .sum_keepdim(0)?
            .affine(1.0 / f64::from(mask.iter().sum::<f32>()), 0.0)?;
        let sem = semantic_signature(prog);
        let sem_t = Tensor::from_vec(sem.clone(), (n, 24), &Device::Cpu)?;
        let centered = sem_t.broadcast_sub(&sem_t.mean_keepdim(0)?)?;
        let mean = flat(self.p("prog_mean"))?;
        let std = flat(self.p("prog_std"))?;
        let standardized = prog
            .iter()
            .enumerate()
            .map(|(i, x)| {
                ((x.signum() * x.abs().ln_1p() - mean[i % 412]) / std[i % 412]).clamp(-8.0, 8.0)
            })
            .collect();
        let bottleneck = self
            .norm(
                &self.linear(
                    &Tensor::from_vec(standardized, (n, 412), &Device::Cpu)?,
                    "prog_412_bottleneck.0",
                )?,
                "prog_412_bottleneck.1",
            )?
            .silu()?;
        let zero2 = Tensor::zeros((n, 2), DType::F32, &Device::Cpu)?;
        let q = self
            .norm(
                &self.linear(
                    &Tensor::cat(&[&sem_t, &centered, &bottleneck, &zero2], 1)?,
                    "cand_query_proj.0",
                )?,
                "cand_query_proj.1",
            )?
            .silu()?;
        let mm_attn = self.attention(&q, &tokens, "mm_cross_attn", Some(&mask))?;
        let bilinear = self
            .linear(&sem_t, "prog_bilinear_v")?
            .broadcast_mul(&self.linear(&ctx, "ctx_bilinear_u")?)?;
        let mm_raw = flat(
            &self.linear(
                &self
                    .linear(
                        &Tensor::cat(&[&mm_attn, &bilinear, &zero2], 1)?,
                        "mm_prior_out.0",
                    )?
                    .silu()?,
                "mm_prior_out.2",
            )?,
        )?;
        let mm = centered_scaled(&mm_raw, self.scalar("mm_prior_scale")?.tanh());
        let eff: Vec<f32> = anchor_prior.iter().zip(&mm).map(|(a, m)| a + m).collect();
        let qv = flat(&q)?;
        let qnorm = normalized_rows(&qv, 32, 1e-8);
        let capped: Vec<f32> = fitness
            .iter()
            .zip(gates)
            .map(|(f, g)| if *g > 0.5 { *f } else { f.min(2.5) })
            .collect();
        let residual: Vec<f32> = (0..4)
            .map(|i| capped[i] / SCALE - eff[revealed[i]])
            .collect();
        let sem_scale = softplus(self.scalar("sem_kernel_scale_raw")?);
        let emb_scale = softplus(self.scalar("emb_kernel_scale_raw")?);
        let w0 = softplus(self.scalar("null_kernel_w0_raw")?) + 0.15;
        let mut sem_adj = Vec::with_capacity(n);
        for i in 0..n {
            let mut distances = Vec::with_capacity(4);
            for (j, r) in revealed.iter().enumerate() {
                let cos_sem =
                    dot(&sem[i * 24..i * 24 + 24], &sem[r * 24..r * 24 + 24]).clamp(-1.0, 1.0);
                let cos_emb =
                    dot(&qnorm[i * 32..i * 32 + 32], &qnorm[r * 32..r * 32 + 32]).clamp(-1.0, 1.0);
                distances.push(((1.0 - cos_sem) * sem_scale + (1.0 - cos_emb) * emb_scale, j));
            }
            distances.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut num = 0.0;
            let mut den = 0.0;
            for (d, j) in distances {
                let w = 0.75 / (0.15 + d) + 0.25 / (1.5 + d);
                num += w * residual[j];
                den += w;
            }
            sem_adj.push(num / (w0 + den));
        }
        let mut history = Vec::with_capacity(4 * 60);
        for (j, r) in revealed.iter().enumerate() {
            history.extend_from_slice(&sem[r * 24..r * 24 + 24]);
            history.extend_from_slice(&qv[r * 32..r * 32 + 32]);
            history.extend_from_slice(&[
                residual[j],
                fitness[j] / SCALE,
                gates[j],
                j as f32 / 64.0,
            ]);
        }
        let history = self
            .norm(
                &self.linear(
                    &Tensor::from_vec(history, (4, 60), &Device::Cpu)?,
                    "hist_token_proj.0",
                )?,
                "hist_token_proj.1",
            )?
            .silu()?;
        let null = self.p("null_hist_token").reshape((1, 32))?;
        let attn = self.attention(
            &q,
            &Tensor::cat(&[&null, &history], 0)?,
            "traj_cross_attn",
            None,
        )?;
        let best = capped
            .iter()
            .map(|f| f / SCALE)
            .fold(f32::INFINITY, f32::min);
        let extra: Vec<f32> = (0..n).flat_map(|i| [eff[i], sem_adj[i], best]).collect();
        let raw = flat(
            &self.linear(
                &self
                    .linear(
                        &Tensor::cat(
                            &[&q, &attn, &Tensor::from_vec(extra, (n, 3), &Device::Cpu)?],
                            1,
                        )?,
                        "traj_delta_head.0",
                    )?
                    .silu()?,
                "traj_delta_head.2",
            )?,
        )?;
        let traj = centered_scaled(&raw, self.scalar("traj_attn_scale")?.tanh());
        let blend = sigmoid(self.scalar("traj_kernel_blend")?) * 0.45;
        let total = (0..n)
            .map(|i| anchor[i] + (mm[i] + blend * sem_adj[i] + traj[i]))
            .collect();
        Ok((total, anchor))
    }

    fn anchor(
        &self,
        prog: &[f32],
        proj: &[f32],
        revealed: &[usize],
        fitness: &[f32],
        gates: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = prog.len() / 412;
        let zero = Tensor::zeros((1, 584), DType::F32, &Device::Cpu)?;
        let ctx = self.residual(
            &self
                .linear(&self.norm(&zero, "anchor.ctx_in.0")?, "anchor.ctx_in.1")?
                .silu()?,
            "anchor.ctx_block",
        )?;
        let ctx_n = ctx.broadcast_as((n, 128))?;
        let lm_weights = flat(
            &self.linear(
                &self
                    .linear(
                        &self.norm(&zero.narrow(1, 520, 64)?, "anchor.landmark_router.0")?,
                        "anchor.landmark_router.1",
                    )?
                    .silu()?,
                "anchor.landmark_router.3",
            )?,
        )?;
        let clamped: Vec<f32> = prog.iter().map(|x| x.clamp(-15.0, 15.0)).collect();
        let prog_t = Tensor::from_vec(clamped.clone(), (n, 412), &Device::Cpu)?;
        let proj_t = Tensor::from_vec(proj.to_vec(), (n, 16), &Device::Cpu)?;
        let kron = Tensor::zeros((n, 128), DType::F32, &Device::Cpu)?;
        let sos = Tensor::zeros((n, 2), DType::F32, &Device::Cpu)?;
        let input = Tensor::cat(&[&prog_t, &proj_t, &kron, &sos], 1)?;
        let emb = self.residual(
            &self
                .linear(&self.norm(&input, "anchor.prog_in.0")?, "anchor.prog_in.1")?
                .silu()?,
            "anchor.prog_block",
        )?;
        let neural = flat(
            &self.linear(
                &self
                    .linear(
                        &self.norm(&Tensor::cat(&[&ctx_n, &emb], 1)?, "anchor.prior_head.0")?,
                        "anchor.prior_head.1",
                    )?
                    .silu()?,
                "anchor.prior_head.3",
            )?,
        )?;
        let alpha = f64::from(self.scalar("anchor.alpha_prior")?);
        let prior: Vec<f64> = (0..n)
            .map(|i| {
                alpha * f64::from((dot(&lm_weights, &proj[i * 16..i * 16 + 16]) + neural[i]).tanh())
            })
            .collect();
        let mut sig = Vec::with_capacity(n * 24);
        for row in clamped.chunks_exact(412) {
            let mut v = vec![0.0; 24];
            for (c, out) in v.iter_mut().enumerate() {
                let mut sum = 0.0;
                let mut max = f32::NEG_INFINITY;
                for r in 0..18 {
                    let x = row.get(r * 24 + c).copied().unwrap_or(0.0);
                    sum += x;
                    max = max.max(x);
                }
                *out = sum / 18.0 + 0.5 * max;
            }
            sig.extend(normalized_rows(&v, 24, 1e-6));
        }
        let smooth: Vec<f64> = (0..4)
            .map(|j| {
                f64::from(if gates[j] > 0.5 {
                    fitness[j]
                } else {
                    fitness[j].min(2.5)
                }) / f64::from(SCALE)
                    - prior[revealed[j]]
            })
            .collect();
        let raw: Vec<f64> = fitness
            .iter()
            .map(|f| f64::from(*f) / f64::from(SCALE))
            .collect();
        let mut w1 = Vec::new();
        let mut structural = Vec::new();
        let mut minimum = Vec::new();
        let mut failures = Vec::new();
        for i in 0..n {
            let mut distances = Vec::new();
            let mut joint = Vec::new();
            for (j, r) in revealed.iter().enumerate() {
                let d = proj[i * 16..i * 16 + 16]
                    .iter()
                    .zip(&proj[r * 16..r * 16 + 16])
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum::<f64>();
                distances.push((d, j));
                let sd = (1.0 - dot(&sig[i * 24..i * 24 + 24], &sig[r * 24..r * 24 + 24])).max(0.0)
                    * 4.0;
                joint.push((0.5 * (d as f32) + 2.0 * sd, j));
            }
            distances.sort_by(|a, b| a.0.total_cmp(&b.0));
            joint.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut num = 0.0;
            let mut den = 0.0;
            for (d, j) in distances {
                let w = 1.0 / (1.0 + d);
                num += w * raw[j];
                den += w;
            }
            w1.push(num / den.max(1e-12));
            minimum.push(joint[0].0.min(10.0) / 10.0);
            let mut sn = 0.0;
            let mut sd = 0.0;
            let mut fail = 0.0;
            for (d, j) in joint {
                let d = f64::from(d);
                let w = 0.7 / (0.25 + d) + 0.3 / (2.0 + d);
                sn += w * smooth[j];
                sd += w;
                fail += w * f64::from(1.0 - gates[j]);
            }
            structural.push(sn / sd.max(1e-12));
            failures.push((fail / sd.max(1e-12)) as f32);
        }
        let ev = flat(&emb)?;
        let mut history = Vec::new();
        for (j, r) in revealed.iter().enumerate() {
            history.extend_from_slice(&ev[r * 128..r * 128 + 128]);
            history.extend_from_slice(&[
                (smooth[j] as f32).clamp(-5.0, 5.0),
                gates[j],
                (prior[*r] as f32).clamp(-5.0, 5.0),
                [0.1, 0.4, 0.7, 1.0][j],
            ]);
        }
        let history = self
            .norm(
                &self.linear(
                    &Tensor::from_vec(history, (4, 132), &Device::Cpu)?,
                    "anchor.history_token_in.0",
                )?,
                "anchor.history_token_in.1",
            )?
            .silu()?;
        let attention = self.attention(&emb, &history, "anchor.cross_attn", None)?;
        let summary: Vec<f32> = (0..n)
            .flat_map(|i| {
                [
                    (w1[i] as f32).clamp(-5.0, 5.0),
                    (structural[i] as f32).clamp(-5.0, 5.0),
                    minimum[i],
                    failures[i],
                ]
            })
            .collect();
        let traj = Tensor::cat(
            &[
                &ctx_n,
                &attention,
                &Tensor::from_vec(summary, (n, 4), &Device::Cpu)?,
            ],
            1,
        )?;
        let delta = flat(
            &self
                .linear(
                    &self
                        .linear(&self.norm(&traj, "anchor.traj_out.0")?, "anchor.traj_out.1")?
                        .silu()?,
                    "anchor.traj_out.3",
                )?
                .tanh()?,
        )?;
        let extra: Vec<f32> = (0..n)
            .flat_map(|i| [prior[i] as f32, failures[i]])
            .collect();
        let gate_input = Tensor::cat(
            &[
                &ctx_n,
                &emb,
                &Tensor::from_vec(extra, (n, 2), &Device::Cpu)?,
            ],
            1,
        )?;
        let gate_logit = flat(
            &self.linear(
                &self
                    .linear(
                        &self.norm(&gate_input, "anchor.gate_head.0")?,
                        "anchor.gate_head.1",
                    )?
                    .silu()?,
                "anchor.gate_head.3",
            )?,
        )?;
        let w_struct = f64::from(sigmoid(self.scalar("anchor.alpha_struct_kernel")?)) * 0.35;
        let alpha_attn = f64::from(self.scalar("anchor.alpha_attn")?);
        let barrier = softplus(self.scalar("anchor.gate_barrier_weight")?);
        let total = (0..n)
            .map(|i| {
                (prior[i]
                    + (1.0 - w_struct) * w1[i]
                    + w_struct * structural[i]
                    + alpha_attn * f64::from(delta[i])
                    + f64::from(barrier * sigmoid(gate_logit[i] + 2.0 * failures[i])))
                    as f32
            })
            .collect();
        Ok((total, prior.iter().map(|v| *v as f32).collect()))
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
mod tests {
    use super::*;

    struct Fixture {
        prog_412: Vec<f32>,
        proj_16: Vec<f32>,
        tokens: Vec<f32>,
        token_mask: Vec<f32>,
        revealed_indices: Vec<i64>,
        revealed_fitness: Vec<f32>,
        revealed_gates: Vec<f32>,
        ctx_feat: Vec<f32>,
        base_prior_cost: Vec<f32>,
        sos_prior: Vec<f32>,
    }

    impl Fixture {
        fn serving(batch: usize, proposals: usize) -> Self {
            let candidates = REVEALED + proposals;
            let bn = batch * candidates;
            Self {
                prog_412: vec![0.0; bn * PROG_FEATURES],
                proj_16: vec![0.0; bn * PROJ_FEATURES],
                tokens: vec![0.0; batch * TOKENS * TOKEN_WIDTH],
                token_mask: vec![0.0; batch * TOKENS],
                revealed_indices: (0..batch).flat_map(|_| 0..REVEALED as i64).collect(),
                revealed_fitness: vec![0.0; batch * REVEALED],
                revealed_gates: vec![0.0; batch * REVEALED],
                ctx_feat: vec![0.0; batch * CONTEXT_FEATURES],
                base_prior_cost: vec![0.0; bn],
                sos_prior: vec![0.0; bn * SOS_FEATURES],
            }
        }

        fn inputs(&self, batch: usize, proposals: usize) -> PredictCostsInputs<'_> {
            PredictCostsInputs {
                batch,
                candidates: REVEALED + proposals,
                prog_412: &self.prog_412,
                proj_16: &self.proj_16,
                tokens: &self.tokens,
                token_mask: &self.token_mask,
                revealed_indices: &self.revealed_indices,
                revealed_fitness: &self.revealed_fitness,
                revealed_gates: &self.revealed_gates,
                ctx_feat: &self.ctx_feat,
                base_prior_cost: &self.base_prior_cost,
                sos_prior: &self.sos_prior,
                scale: SCALE,
            }
        }
    }

    #[test]
    fn accepts_both_serving_slate_shapes() {
        for proposals in [8, 16] {
            let fixture = Fixture::serving(2, proposals);
            assert_eq!(fixture.inputs(2, proposals).validate_shapes(), Ok(()));
        }
    }

    #[test]
    fn reports_a_mismatch_for_every_flattened_f12_input_shape() {
        for proposals in [8, 16] {
            let fixture = Fixture::serving(2, proposals);
            let valid = fixture.inputs(2, proposals);

            macro_rules! assert_one_short {
                ($field:ident, $name:literal) => {{
                    let actual = fixture.$field.len() - 1;
                    let malformed = PredictCostsInputs {
                        $field: &fixture.$field[..actual],
                        ..valid
                    };
                    let error = malformed.validate_shapes().unwrap_err();
                    assert_eq!(error.field, $name);
                    assert_eq!(error.expected, fixture.$field.len());
                    assert_eq!(error.actual, actual);
                }};
            }

            assert_one_short!(prog_412, "prog_412");
            assert_one_short!(proj_16, "proj_16");
            assert_one_short!(tokens, "tokens");
            assert_one_short!(token_mask, "token_mask");
            assert_one_short!(revealed_indices, "revealed_indices");
            assert_one_short!(revealed_fitness, "revealed_fitness");
            assert_one_short!(revealed_gates, "revealed_gates");
            assert_one_short!(ctx_feat, "ctx_feat");
            assert_one_short!(base_prior_cost, "base_prior_cost");
            assert_one_short!(sos_prior, "sos_prior");
        }
    }

    const TOTAL_12: [f32; 24] = [
        f32::from_bits(0x41a08967),
        f32::from_bits(0x430c0fd4),
        f32::from_bits(0x41f32476),
        f32::from_bits(0x41b15854),
        f32::from_bits(0x421ceae8),
        f32::from_bits(0x423c9991),
        f32::from_bits(0x4247eb94),
        f32::from_bits(0x424a5283),
        f32::from_bits(0x4247a1ca),
        f32::from_bits(0x424038a5),
        f32::from_bits(0x41a9ed52),
        f32::from_bits(0x4307eab9),
        f32::from_bits(0x410b3f37),
        f32::from_bits(0x41e6367b),
        f32::from_bits(0x416f0610),
        f32::from_bits(0x40b02f44),
        f32::from_bits(0x414a6145),
        f32::from_bits(0x414d8686),
        f32::from_bits(0x4156ecc2),
        f32::from_bits(0x415eb38b),
        f32::from_bits(0x41693c78),
        f32::from_bits(0x4159b47c),
        f32::from_bits(0x41bd0c4b),
        f32::from_bits(0x4189a479),
    ];
    const ANCHOR_12: [f32; 24] = [
        f32::from_bits(0x419d3a0d),
        f32::from_bits(0x430adab1),
        f32::from_bits(0x41ef1b9c),
        f32::from_bits(0x41a84428),
        f32::from_bits(0x4219aa0a),
        f32::from_bits(0x423a24e8),
        f32::from_bits(0x42448d3b),
        f32::from_bits(0x4247a344),
        f32::from_bits(0x4244a472),
        f32::from_bits(0x423d7d51),
        f32::from_bits(0x41a46e0d),
        f32::from_bits(0x43073277),
        f32::from_bits(0x40ee8d77),
        f32::from_bits(0x41bc7590),
        f32::from_bits(0x41440c3f),
        f32::from_bits(0x408628ac),
        f32::from_bits(0x411e5f20),
        f32::from_bits(0x412968a1),
        f32::from_bits(0x4135dad8),
        f32::from_bits(0x413a7222),
        f32::from_bits(0x413f80fe),
        f32::from_bits(0x4137fb46),
        f32::from_bits(0x41aacc23),
        f32::from_bits(0x416b7a6c),
    ];
    const TOTAL_20: [f32; 40] = [
        f32::from_bits(0x41a08662),
        f32::from_bits(0x430c0fd5),
        f32::from_bits(0x41f322c3),
        f32::from_bits(0x41b157d6),
        f32::from_bits(0x421cea8c),
        f32::from_bits(0x423c98bb),
        f32::from_bits(0x4247e979),
        f32::from_bits(0x424a5356),
        f32::from_bits(0x4247a252),
        f32::from_bits(0x42403907),
        f32::from_bits(0x41a9ed57),
        f32::from_bits(0x4307eab3),
        f32::from_bits(0x422e639c),
        f32::from_bits(0x421f1229),
        f32::from_bits(0x423238e4),
        f32::from_bits(0x423cad7b),
        f32::from_bits(0x4246953b),
        f32::from_bits(0x42494abc),
        f32::from_bits(0x424279db),
        f32::from_bits(0x422cda12),
        f32::from_bits(0x410b3a3a),
        f32::from_bits(0x41e6317b),
        f32::from_bits(0x416f04a0),
        f32::from_bits(0x40b034f2),
        f32::from_bits(0x414a5f4e),
        f32::from_bits(0x414d8e48),
        f32::from_bits(0x4156eb94),
        f32::from_bits(0x415ebd96),
        f32::from_bits(0x41693a09),
        f32::from_bits(0x4159c2e3),
        f32::from_bits(0x41bd1119),
        f32::from_bits(0x4189a3bd),
        f32::from_bits(0x413ea8cc),
        f32::from_bits(0x40e92d5d),
        f32::from_bits(0x4145be65),
        f32::from_bits(0x415236a2),
        f32::from_bits(0x415c9654),
        f32::from_bits(0x416224b4),
        f32::from_bits(0x4162cbac),
        f32::from_bits(0x4156b367),
    ];
    const ANCHOR_20: [f32; 40] = [
        f32::from_bits(0x419d3a0d),
        f32::from_bits(0x430adab1),
        f32::from_bits(0x41ef1b9c),
        f32::from_bits(0x41a84428),
        f32::from_bits(0x4219aa0a),
        f32::from_bits(0x423a24e8),
        f32::from_bits(0x42448d3b),
        f32::from_bits(0x4247a344),
        f32::from_bits(0x4244a472),
        f32::from_bits(0x423d7d51),
        f32::from_bits(0x41a46e0d),
        f32::from_bits(0x43073277),
        f32::from_bits(0x422bd528),
        f32::from_bits(0x421b32ed),
        f32::from_bits(0x422f9062),
        f32::from_bits(0x42399916),
        f32::from_bits(0x424390d7),
        f32::from_bits(0x42469419),
        f32::from_bits(0x423f8c80),
        f32::from_bits(0x422a4998),
        f32::from_bits(0x40ee8d77),
        f32::from_bits(0x41bc7590),
        f32::from_bits(0x41440c3f),
        f32::from_bits(0x408628ac),
        f32::from_bits(0x411e5f20),
        f32::from_bits(0x412968a1),
        f32::from_bits(0x4135dad8),
        f32::from_bits(0x413a7222),
        f32::from_bits(0x413f80fe),
        f32::from_bits(0x4137fb46),
        f32::from_bits(0x41aacc23),
        f32::from_bits(0x416b7a6c),
        f32::from_bits(0x4120996b),
        f32::from_bits(0x4095f12d),
        f32::from_bits(0x411eb10a),
        f32::from_bits(0x41303688),
        f32::from_bits(0x4138d3a9),
        f32::from_bits(0x413bdfc8),
        f32::from_bits(0x413f43dc),
        f32::from_bits(0x41389f45),
    ];

    type FixtureParameters = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

    fn synthetic_weights(change: impl FnOnce(&mut FixtureParameters)) -> Vec<u8> {
        let mut parameters: FixtureParameters = parameter_specs()
            .into_iter()
            .map(|(name, shape)| {
                let count = shape.iter().product::<usize>();
                let salt = name.bytes().map(usize::from).sum::<usize>() % 97;
                let values = (0..count)
                    .map(|i| {
                        let value = if name == "prog_std" {
                            1.0 + (i % 7) as f32 * 0.05
                        } else if name == "prog_mean" {
                            ((i % 17) as i32 - 8) as f32 * 0.01
                        } else if shape.is_empty() {
                            match name.as_str() {
                                "anchor.alpha_prior" => 0.08,
                                "anchor.alpha_struct_kernel" => 0.15,
                                "anchor.alpha_attn" | "anchor.gate_barrier_weight" => 0.05,
                                "mm_prior_scale" => 0.31,
                                "null_kernel_w0_raw" => -0.5,
                                "sem_kernel_scale_raw" => 1.2,
                                "emb_kernel_scale_raw" => 0.8,
                                "traj_kernel_blend" => 0.25,
                                "traj_attn_scale" => 0.27,
                                _ => panic!("unknown synthetic scalar"),
                            }
                        } else if name.ends_with(".weight") && shape.len() == 1 {
                            1.0 + (((i + salt) % 7) as i32 - 3) as f32 * 0.01
                        } else {
                            (((i * 37 + salt) % 101) as i32 - 50) as f32 * 0.002
                        };
                        if name == "mm_prior_out.2.weight" {
                            value * 50.0
                        } else {
                            value
                        }
                    })
                    .collect();
                (name, (shape, values))
            })
            .collect();
        change(&mut parameters);
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, (shape, values)) in parameters {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                format!("adapter_10tok.{name}"),
                serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        bytes
    }

    fn synthetic_adapter() -> CpuAdapter {
        CpuAdapter::from_bytes(&synthetic_weights(|_| {})).unwrap()
    }

    fn synthetic_fixture(proposals: usize) -> Fixture {
        let mut f = Fixture::serving(2, proposals);
        let n = 4 + proposals;
        for b in 0..2 {
            for c in 0..n {
                for k in 0..412 {
                    f.prog_412[(b * n + c) * 412 + k] =
                        (((b * 31 + c * 17 + k * 7) % 41) as i32 - 20) as f32 * 0.12;
                }
                for k in 0..16 {
                    f.proj_16[(b * n + c) * 16 + k] =
                        (((b * 13 + c * 3 + k * 11) % 29) as i32 - 14) as f32 * 0.07;
                }
            }
            for m in 0..10 {
                f.token_mask[b * 10 + m] = if (b + m) % 3 == 0 { 0.0 } else { 1.0 };
                for k in 0..32 {
                    f.tokens[(b * 10 + m) * 32 + k] =
                        (((b * 7 + m * 3 + k) % 23) as i32 - 11) as f32 * 0.09;
                }
            }
        }
        f.revealed_indices = vec![0, 1, 2, 3, 3, 1, 0, 2];
        f.revealed_fitness = vec![0.2, 100.0, 0.7, 3.7, 0.4, 12.5, 1.1, 5.0];
        f.revealed_gates = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        f
    }

    #[test]
    fn full_cpu_graph_matches_nonzero_python_controls_and_admission_on_both_slates() {
        use rand_chacha::{rand_core::SeedableRng, ChaCha12Rng};
        let model = synthetic_adapter();
        let mut decisive = 0;
        for (n, expected, expected_anchor) in [
            (12, TOTAL_12.as_slice(), ANCHOR_12.as_slice()),
            (20, TOTAL_20.as_slice(), ANCHOR_20.as_slice()),
        ] {
            let fixture = synthetic_fixture(n - 4);
            let result = model.predict_costs(fixture.inputs(2, n - 4)).unwrap();
            for (name, actual, oracle) in [
                ("total", result.total_cost.as_slice(), expected),
                ("anchor", result.anchor_cost.as_slice(), expected_anchor),
            ] {
                for (i, (a, e)) in actual.iter().zip(oracle).enumerate() {
                    assert!(
                        (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
                        "{name}[{i}] actual={a} expected={e}"
                    );
                    // Stronger synthetic control detects omitted small branches even
                    // when the large uncapped W1 failure cost loosens M2's rtol.
                    assert!(
                        (a - e).abs() < 8e-5,
                        "{name}[{i}] synthetic error {}",
                        (a - e).abs()
                    );
                }
            }
            assert!(result
                .total_cost
                .iter()
                .zip(&result.anchor_cost)
                .any(|(a, b)| (a - b).abs() > 0.1));
            for b in 0..2 {
                let mut order: Vec<usize> = (4..n).collect();
                order.sort_by(|left, right| {
                    expected[b * n + *left].total_cmp(&expected[b * n + *right])
                });
                let ranked = if n == 12 { 4 } else { 2 };
                let lo = expected[b * n + order[ranked - 1]];
                let hi = expected[b * n + order[ranked]];
                if hi - lo > 2.0 * (1e-5 + 1e-4 * lo.abs()) {
                    decisive += 1;
                    let actual: Vec<f64> = result.total_cost[b * n..b * n + n]
                        .iter()
                        .map(|v| f64::from(*v))
                        .collect();
                    let oracle: Vec<f64> = expected[b * n..b * n + n]
                        .iter()
                        .map(|v| f64::from(*v))
                        .collect();
                    let a = super::super::admission::select_costs(
                        &actual,
                        n == 20,
                        &mut ChaCha12Rng::seed_from_u64(71),
                    )
                    .unwrap();
                    let e = super::super::admission::select_costs(
                        &oracle,
                        n == 20,
                        &mut ChaCha12Rng::seed_from_u64(71),
                    )
                    .unwrap();
                    assert_eq!(a, e);
                }
            }
        }
        assert!(
            decisive > 0,
            "synthetic control must exercise a decision outside the tie band"
        );
    }

    #[test]
    fn batch_rows_share_only_their_own_conditioning() {
        let model = synthetic_adapter();
        for proposals in [8, 16] {
            let f = synthetic_fixture(proposals);
            let n = 4 + proposals;
            let all = f.inputs(2, proposals);
            let batched = model.predict_costs(all).unwrap();
            for b in 0..2 {
                let one = PredictCostsInputs {
                    batch: 1,
                    candidates: n,
                    prog_412: &f.prog_412[b * n * 412..(b + 1) * n * 412],
                    proj_16: &f.proj_16[b * n * 16..(b + 1) * n * 16],
                    tokens: &f.tokens[b * 320..b * 320 + 320],
                    token_mask: &f.token_mask[b * 10..b * 10 + 10],
                    revealed_indices: &f.revealed_indices[b * 4..b * 4 + 4],
                    revealed_fitness: &f.revealed_fitness[b * 4..b * 4 + 4],
                    revealed_gates: &f.revealed_gates[b * 4..b * 4 + 4],
                    ctx_feat: &f.ctx_feat[b * 584..b * 584 + 584],
                    base_prior_cost: &f.base_prior_cost[b * n..b * n + n],
                    sos_prior: &f.sos_prior[b * n * 2..(b + 1) * n * 2],
                    scale: SCALE,
                };
                assert_eq!(
                    model.predict_costs(one).unwrap().total_cost,
                    batched.total_cost[b * n..b * n + n]
                );
            }
        }
    }

    #[test]
    fn masked_tokens_are_inert_but_token_zero_is_always_present() {
        let model = synthetic_adapter();
        let mut f = synthetic_fixture(8);
        let original = model.predict_costs(f.inputs(2, 8)).unwrap().total_cost;
        for b in 0..2 {
            for m in 1..10 {
                if f.token_mask[b * 10 + m] == 0.0 {
                    for v in &mut f.tokens[(b * 10 + m) * 32..(b * 10 + m + 1) * 32] {
                        *v += 100.0;
                    }
                }
            }
        }
        assert_eq!(
            model.predict_costs(f.inputs(2, 8)).unwrap().total_cost,
            original
        );
        f.token_mask[0] = 1.0;
        assert_eq!(
            model.predict_costs(f.inputs(2, 8)).unwrap().total_cost,
            original
        );
        f.token_mask.fill(0.0);
        let first = model.predict_costs(f.inputs(2, 8)).unwrap().total_cost;
        for k in 0..32 {
            f.tokens[k] += k as f32 * 0.3;
        }
        let changed = model.predict_costs(f.inputs(2, 8)).unwrap().total_cost;
        assert!(first.iter().zip(changed).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn refuses_nonfinite_inputs_noncanonical_constants_masks_and_reveals() {
        let model = synthetic_adapter();
        for case in 0..16 {
            let mut f = synthetic_fixture(8);
            match case {
                0 => f.prog_412[0] = f32::NAN,
                1 => f.proj_16[0] = f32::INFINITY,
                2 => f.tokens[0] = f32::NAN,
                3 => f.revealed_fitness[0] = f32::INFINITY,
                4 => f.token_mask[0] = 0.5,
                5 => f.revealed_gates[0] = 2.0,
                6 => f.ctx_feat[0] = 1.0,
                7 => f.base_prior_cost[0] = 1.0,
                8 => f.sos_prior[0] = 1.0,
                9 => f.revealed_indices[0] = -1,
                10 => f.revealed_indices[0] = 12,
                11 => f.revealed_indices[0] = f.revealed_indices[1],
                _ => {}
            }
            let mut input = f.inputs(2, 8);
            match case {
                12 => input.scale = 1.0,
                13 => input.batch = 0,
                14 => input.candidates = 19,
                15 => input.scale = f32::NAN,
                _ => {}
            }
            assert!(
                model.predict_costs(input).is_err(),
                "accepted invalid input case {case}"
            );
        }
    }

    #[test]
    fn loader_refuses_missing_misshaped_nonfinite_and_bad_scaler_parameters() {
        for case in 0..5 {
            let bytes = synthetic_weights(|p| match case {
                0 => {
                    p.remove("anchor.alpha_attn");
                }
                1 => p.get_mut("modality_norm.weight").unwrap().0 = vec![1, 32],
                2 => p.get_mut("prog_mean").unwrap().1[0] = f32::NAN,
                3 => p.get_mut("prog_std").unwrap().1[0] = 0.0,
                4 => p.get_mut("prog_std").unwrap().1[0] = -1.0,
                _ => unreachable!(),
            });
            assert!(
                CpuAdapter::from_bytes(&bytes).is_err(),
                "accepted invalid parameter case {case}"
            );
        }
        let mut bytes = synthetic_weights(|_| {});
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = String::from_utf8(bytes[8..8 + header_len].to_vec())
            .unwrap()
            .replace("F32", "I32");
        bytes[8..8 + header_len].copy_from_slice(header.as_bytes());
        assert!(CpuAdapter::from_bytes(&bytes).is_err());
        assert!(CpuAdapter::from_bytes(&u64::MAX.to_le_bytes()).is_err());
    }

    #[test]
    fn kl_readout_has_a_known_zero_and_flags_large_proposal_shift() {
        assert_eq!(proposal_kl(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        let shifted = proposal_kl(&[0.0, 10.0], &[10.0, 0.0]);
        assert!((shifted - 9.9990920426).abs() < 1e-8);
        assert!(shifted > 0.35);
        assert!(proposal_kl(&[f32::MAX, -f32::MAX], &[-f32::MAX, f32::MAX]).is_finite());
    }

    #[test]
    #[ignore = "requires RES-07 real serving-slate fixtures for >=50 contexts; synthetic controls are not M2"]
    fn real_serving_slate_cost_and_admitted_set_m2_parity() {
        panic!("RES-07 must publish real input/output fixtures before the M2 parity gate can run");
    }
}
