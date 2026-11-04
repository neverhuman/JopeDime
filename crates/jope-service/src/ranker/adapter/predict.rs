use super::*;

impl CpuAdapter {
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
            &self.device,
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
            .broadcast_mul(&Tensor::from_vec(gate, (10, 1), &self.device)?)?;
        let ctx = tokens
            .sum_keepdim(0)?
            .affine(1.0 / f64::from(mask.iter().sum::<f32>()), 0.0)?;
        let sem = semantic_signature(prog);
        let sem_t = Tensor::from_vec(sem.clone(), (n, 24), &self.device)?;
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
                    &Tensor::from_vec(standardized, (n, 412), &self.device)?,
                    "prog_412_bottleneck.0",
                )?,
                "prog_412_bottleneck.1",
            )?
            .silu()?;
        let zero2 = Tensor::zeros((n, 2), DType::F32, &self.device)?;
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
                    &Tensor::from_vec(history, (4, 60), &self.device)?,
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
                            &[&q, &attn, &Tensor::from_vec(extra, (n, 3), &self.device)?],
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
        let zero = Tensor::zeros((1, 584), DType::F32, &self.device)?;
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
        let prog_t = Tensor::from_vec(clamped.clone(), (n, 412), &self.device)?;
        let proj_t = Tensor::from_vec(proj.to_vec(), (n, 16), &self.device)?;
        let kron = Tensor::zeros((n, 128), DType::F32, &self.device)?;
        let sos = Tensor::zeros((n, 2), DType::F32, &self.device)?;
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
                    &Tensor::from_vec(history, (4, 132), &self.device)?,
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
                &Tensor::from_vec(summary, (n, 4), &self.device)?,
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
                &Tensor::from_vec(extra, (n, 2), &self.device)?,
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
