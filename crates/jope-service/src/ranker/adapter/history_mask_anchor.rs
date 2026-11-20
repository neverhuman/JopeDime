//! Presence-masked frozen anchor branch (`FlywheelGameNet.predict_costs`) for
//! `CpuAdapter::predict_costs_with_history_mask`.

use super::{
    dot, flat, normalized_rows, sigmoid, softplus, ActiveSlot, CpuAdapter, DType, Result, Tensor,
    SCALE,
};

fn anchor_step_frac(k: usize, k_rev: usize) -> f32 {
    match k_rev {
        1 => 0.1,
        2 => [0.1, 1.0][k],
        3 => [0.1, 0.55, 1.0][k],
        4 => [0.1, 0.4, 0.7, 1.0][k],
        _ => 0.1 + 0.9 * (k as f32) / ((k_rev - 1) as f32),
    }
}

impl CpuAdapter {
    pub(super) fn anchor_masked(
        &self,
        prog: &[f32],
        proj: &[f32],
        base_prior: &[f32],
        active: &[ActiveSlot],
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
        let eff_prior: Vec<f64> = (0..n)
            .map(|i| {
                let delta = alpha
                    * f64::from((dot(&lm_weights, &proj[i * 16..i * 16 + 16]) + neural[i]).tanh());
                if base_prior[i] == 0.0 {
                    delta
                } else {
                    f64::from(base_prior[i]) + delta
                }
            })
            .collect();

        let barrier = softplus(self.scalar("anchor.gate_barrier_weight")?);
        if active.is_empty() {
            let extra: Vec<f32> = (0..n)
                .flat_map(|i| [eff_prior[i] as f32, 0.0_f32])
                .collect();
            let gate_input = Tensor::cat(
                &[
                    &ctx_n,
                    &emb,
                    &Tensor::from_vec(extra, (n, 2), &self.device)?,
                ],
                1,
            )?;
            let fail_logit = flat(
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
            let total = (0..n)
                .map(|i| (eff_prior[i] + f64::from(barrier * sigmoid(fail_logit[i]))) as f32)
                .collect();
            return Ok((total, eff_prior.iter().map(|&v| v as f32).collect()));
        }

        let k_rev = active.len();
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
        let smooth: Vec<f64> = active
            .iter()
            .map(|&(r, f, g)| {
                f64::from(if g > 0.5 { f } else { f.min(2.5) }) / f64::from(SCALE) - eff_prior[r]
            })
            .collect();
        let res_w1: Vec<f64> = active
            .iter()
            .map(|&(r, f, _)| {
                let raw = f64::from(f) / f64::from(SCALE);
                if base_prior[r] == 0.0 {
                    raw
                } else {
                    raw - f64::from(base_prior[r])
                }
            })
            .collect();
        let mut w1 = Vec::with_capacity(n);
        let mut structural = Vec::with_capacity(n);
        let mut minimum = Vec::with_capacity(n);
        let mut failures = Vec::with_capacity(n);
        for i in 0..n {
            let mut distances = Vec::with_capacity(k_rev);
            let mut joint = Vec::with_capacity(k_rev);
            for (k, &(r, _, _)) in active.iter().enumerate() {
                let d = proj[i * 16..i * 16 + 16]
                    .iter()
                    .zip(&proj[r * 16..r * 16 + 16])
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum::<f64>();
                distances.push((d, k));
                let sd = (1.0 - dot(&sig[i * 24..i * 24 + 24], &sig[r * 24..r * 24 + 24])).max(0.0)
                    * 4.0;
                joint.push((0.5 * (d as f32) + 2.0 * sd, k));
            }
            distances.sort_by(|a, b| a.0.total_cmp(&b.0));
            joint.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut num = 0.0;
            let mut den = 0.0;
            for (d, k) in distances {
                let w = 1.0 / (1.0 + d);
                num += w * res_w1[k];
                den += w;
            }
            w1.push(num / den.max(1e-12));
            minimum.push(joint[0].0.min(10.0) / 10.0);
            let mut sn = 0.0;
            let mut sd = 0.0;
            let mut fail = 0.0;
            for (d, k) in joint {
                let d = f64::from(d);
                let w = 0.7 / (0.25 + d) + 0.3 / (2.0 + d);
                sn += w * smooth[k];
                sd += w;
                fail += w * f64::from(1.0 - active[k].2);
            }
            structural.push(sn / sd.max(1e-12));
            failures.push((fail / sd.max(1e-12)) as f32);
        }
        let ev = flat(&emb)?;
        let mut history = Vec::with_capacity(k_rev * 132);
        for (k, &(r, _, g)) in active.iter().enumerate() {
            history.extend_from_slice(&ev[r * 128..r * 128 + 128]);
            history.extend_from_slice(&[
                (smooth[k] as f32).clamp(-5.0, 5.0),
                g,
                (eff_prior[r] as f32).clamp(-5.0, 5.0),
                anchor_step_frac(k, k_rev),
            ]);
        }
        let history = self
            .norm(
                &self.linear(
                    &Tensor::from_vec(history, (k_rev, 132), &self.device)?,
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
            .flat_map(|i| [eff_prior[i] as f32, failures[i]])
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
        let total = (0..n)
            .map(|i| {
                (eff_prior[i]
                    + (1.0 - w_struct) * w1[i]
                    + w_struct * structural[i]
                    + alpha_attn * f64::from(delta[i])
                    + f64::from(barrier * sigmoid(gate_logit[i] + 2.0 * failures[i])))
                    as f32
            })
            .collect();
        Ok((total, eff_prior.iter().map(|&v| v as f32).collect()))
    }
}
