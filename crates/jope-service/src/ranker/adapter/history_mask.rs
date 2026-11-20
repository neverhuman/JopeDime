//! Presence-masked history contract (`DV-02`) for `CpuAdapter`, matching Python
//! `FlywheelBlockFormerAdapterNet` (`MaskedPresenceV1`) across `K_rev = 0..=4`
//! while preserving exact `0-ULP` parity with `predict_costs` at `[true; 4]`.

use super::*;
use serde::{Deserialize, Serialize};

#[path = "history_mask_anchor.rs"]
mod anchor_impl;

/// Canonical schema identifier for the ranker presence-masked history contract (`DV-02`).
pub const HISTORY_CONTRACT_V1_SCHEMA: &str = "jope.ranker-history-contract/v1";

/// History conditioning mode for `CpuAdapter` ranker evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryMode {
    /// Fixed four-slot F12 contract requiring four distinct revealed candidates.
    LegacyF12,
    /// Explicit per-slot presence mask (`0..=4` observed history slots per context).
    MaskedPresenceV1,
}

pub(super) type ActiveSlot = (usize, f32, f32);

impl CpuAdapter {
    /// Evaluate ranker candidate costs under an explicit `[batch * REVEALED]` slot-presence mask.
    ///
    /// Absent slots (`history_present[b * 4 + j] == false`) never read `revealed_indices`,
    /// `revealed_fitness`, or `revealed_gates`. Active slots are compacted to `0..K_rev` so
    /// `K_rev = 0` skips history attention/trajectory delta heads cleanly, `1 <= K_rev <= 3`
    /// is invariant to which physical slots hold the observations, and `[true; 4]` with zero
    /// `base_prior_cost` is bit-for-bit (`0 ULP`) identical to [`CpuAdapter::predict_costs`].
    pub fn predict_costs_with_history_mask(
        &self,
        inputs: PredictCostsInputs<'_>,
        history_present: &[bool],
    ) -> Result<PredictedCosts> {
        inputs.validate_shapes_masked()?;
        ensure!(
            history_present.len() == inputs.batch * REVEALED,
            "history_present has {} elements; expected {}",
            history_present.len(),
            inputs.batch * REVEALED
        );
        ensure!(
            inputs.scale.to_bits() == SCALE.to_bits(),
            "F12 scale must be 0.4208"
        );
        for (name, values) in [
            ("prog_412", inputs.prog_412),
            ("proj_16", inputs.proj_16),
            ("tokens", inputs.tokens),
            ("base_prior_cost", inputs.base_prior_cost),
        ] {
            ensure!(values.iter().all(|v| v.is_finite()), "non-finite {name}");
        }
        ensure!(
            inputs.token_mask.iter().all(|v| *v == 0.0 || *v == 1.0),
            "token_mask must be binary"
        );
        ensure!(
            inputs
                .ctx_feat
                .iter()
                .chain(inputs.sos_prior)
                .all(|v| *v == 0.0),
            "F12 context and sos_prior must be zero"
        );

        let mut per_batch_active: Vec<Vec<ActiveSlot>> = Vec::with_capacity(inputs.batch);
        for b in 0..inputs.batch {
            let present = &history_present[b * REVEALED..(b + 1) * REVEALED];
            let mut active = Vec::with_capacity(REVEALED);
            for (slot_j, &is_present) in present.iter().enumerate() {
                if !is_present {
                    continue;
                }
                let f = inputs.revealed_fitness[b * REVEALED + slot_j];
                ensure!(f.is_finite(), "non-finite fitness on present history slot");
                let g = inputs.revealed_gates[b * REVEALED + slot_j];
                ensure!(g == 0.0 || g == 1.0, "present history gate must be binary");
                let idx = inputs.revealed_indices[b * REVEALED + slot_j];
                ensure!(
                    idx >= 0 && (idx as usize) < inputs.candidates,
                    "invalid revealed indices"
                );
                let r = idx as usize;
                ensure!(
                    active.iter().all(|&(prev_r, _, _)| prev_r != r),
                    "invalid revealed indices"
                );
                active.push((r, f, g));
            }
            per_batch_active.push(active);
        }

        let mut total_cost = Vec::with_capacity(inputs.batch * inputs.candidates);
        let mut anchor_cost = Vec::with_capacity(inputs.batch * inputs.candidates);
        let mut proposal_kl_to_anchor = Vec::with_capacity(inputs.batch);
        for (b, active) in per_batch_active.iter().enumerate() {
            let (total, anchor) = self.forward_row_masked(&inputs, b, active)?;
            ensure!(
                total.iter().chain(&anchor).all(|v| v.is_finite()),
                "adapter produced non-finite costs"
            );
            let kl = if active.len() == REVEALED && inputs.candidates > REVEALED {
                if inputs.candidates - REVEALED <= 1 {
                    0.0
                } else {
                    proposal_kl(&total[REVEALED..], &anchor[REVEALED..])
                }
            } else if active.is_empty() {
                if total.len() <= 1 {
                    0.0
                } else {
                    proposal_kl(&total, &anchor)
                }
            } else {
                let mut prop_total =
                    Vec::with_capacity(inputs.candidates.saturating_sub(active.len()));
                let mut prop_anchor =
                    Vec::with_capacity(inputs.candidates.saturating_sub(active.len()));
                for c in 0..inputs.candidates {
                    if active.iter().all(|&(r, _, _)| r != c) {
                        prop_total.push(total[c]);
                        prop_anchor.push(anchor[c]);
                    }
                }
                if prop_total.len() <= 1 {
                    0.0
                } else {
                    proposal_kl(&prop_total, &prop_anchor)
                }
            };
            proposal_kl_to_anchor.push(kl);
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

    fn forward_row_masked(
        &self,
        inputs: &PredictCostsInputs<'_>,
        b: usize,
        active: &[ActiveSlot],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = inputs.candidates;
        let prog = &inputs.prog_412[b * n * 412..(b + 1) * n * 412];
        let proj = &inputs.proj_16[b * n * 16..(b + 1) * n * 16];
        let base_prior = &inputs.base_prior_cost[b * n..(b + 1) * n];
        let (anchor, anchor_prior) = self.anchor_masked(prog, proj, base_prior, active)?;
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
        if active.is_empty() {
            let total = (0..n).map(|i| anchor[i] + mm[i]).collect();
            return Ok((total, anchor));
        }

        let k_rev = active.len();
        let eff: Vec<f32> = anchor_prior.iter().zip(&mm).map(|(a, m)| a + m).collect();
        let qv = flat(&q)?;
        let qnorm = normalized_rows(&qv, 32, 1e-8);
        let capped: Vec<f32> = active
            .iter()
            .map(|&(_, f, g)| if g > 0.5 { f } else { f.min(2.5) })
            .collect();
        let residual: Vec<f32> = active
            .iter()
            .enumerate()
            .map(|(k, &(r, _, _))| capped[k] / SCALE - eff[r])
            .collect();
        let sem_scale = softplus(self.scalar("sem_kernel_scale_raw")?);
        let emb_scale = softplus(self.scalar("emb_kernel_scale_raw")?);
        let w0 = softplus(self.scalar("null_kernel_w0_raw")?) + 0.15;
        let mut sem_adj = Vec::with_capacity(n);
        for i in 0..n {
            let mut distances = Vec::with_capacity(k_rev);
            for (k, &(r, _, _)) in active.iter().enumerate() {
                let cos_sem =
                    dot(&sem[i * 24..i * 24 + 24], &sem[r * 24..r * 24 + 24]).clamp(-1.0, 1.0);
                let cos_emb =
                    dot(&qnorm[i * 32..i * 32 + 32], &qnorm[r * 32..r * 32 + 32]).clamp(-1.0, 1.0);
                distances.push(((1.0 - cos_sem) * sem_scale + (1.0 - cos_emb) * emb_scale, k));
            }
            distances.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut num = 0.0;
            let mut den = 0.0;
            for (d, k) in distances {
                let w = 0.75 / (0.15 + d) + 0.25 / (1.5 + d);
                num += w * residual[k];
                den += w;
            }
            sem_adj.push(num / (w0 + den));
        }
        let mut history = Vec::with_capacity(k_rev * 60);
        for (k, &(r, f, g)) in active.iter().enumerate() {
            history.extend_from_slice(&sem[r * 24..r * 24 + 24]);
            history.extend_from_slice(&qv[r * 32..r * 32 + 32]);
            history.extend_from_slice(&[residual[k], f / SCALE, g, k as f32 / 64.0]);
        }
        let history = self
            .norm(
                &self.linear(
                    &Tensor::from_vec(history, (k_rev, 60), &self.device)?,
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
}
