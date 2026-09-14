//! Score generated programs using only the frozen policy and a saved training context.
//! Reward outputs are within-context ranking logits, not calibrated RMSE estimates.
use crate::{
    grammar::Vocab,
    manifest::hash,
    neural::{Bf16, Policy},
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use tch::{Device, Kind, Tensor};

pub const SELECTION_ID: &str = "mean_reward_heads_desc_program_hash_v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bindings {
    pub checkpoint_sha256: String,
    pub conditioning_context_sha256: String,
    pub vocabulary_sha256: String,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub device: Device,
    pub batch_size: usize,
    pub keep: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub program: String,
    pub program_sha256: String,
    pub token_count: usize,
    pub predicted_quality: f64,
    pub predicted_runtime_seconds: f64,
    pub reward_heads: [f64; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ranking {
    pub schema: String,
    pub selection_id: String,
    pub bindings: Bindings,
    pub pool_sha256: String,
    pub pool_size: usize,
    pub chosen_count: usize,
    pub scoring_batches: usize,
    pub batch_size: usize,
    pub backend: String,
    pub quality_units: String,
    pub ordering: String,
    pub training_context_only: bool,
    pub heldout_labels_read: bool,
    pub gp_scores_read: bool,
    /// Sorted best first. The first chosen_count entries are the selected pool.
    pub ranked: Vec<Prediction>,
}
impl Ranking {
    pub fn selected(&self) -> &[Prediction] {
        &self.ranked[..self.chosen_count]
    }
}

fn sorted_predictions(mut rows: Vec<Prediction>) -> Result<Vec<Prediction>> {
    ensure!(
        rows.iter().all(|p| p.predicted_quality.is_finite()
            && p.predicted_runtime_seconds.is_finite()
            && p.predicted_runtime_seconds >= 0.0
            && p.reward_heads.iter().all(|v| v.is_finite())),
        "nonfinite quality/runtime prediction"
    );
    rows.sort_by(|a, b| {
        // All values are finite; +/- zero are a tie and use program identity.
        b.predicted_quality
            .partial_cmp(&a.predicted_quality)
            .unwrap()
            .then(a.program_sha256.cmp(&b.program_sha256))
            .then(a.program.cmp(&b.program))
    });
    Ok(rows)
}

/// The caller supplies the already-loaded frozen checkpoint and the matching
/// vocabulary. There is deliberately no input for GP scores or held-out labels.
/// A shuffled-context arm must supply its donor context here as well as during
/// autoregressive generation. Retained/exploration candidates stay outside this pool.
pub fn score_pool(
    policy: &Policy,
    vocabulary: &Vocab,
    training_context: &[f32],
    programs: &[String],
    bindings: Bindings,
    options: Options,
) -> Result<Ranking> {
    crate::campaign::check_deadline()?;
    ensure!(
        matches!(options.device, Device::Cpu | Device::Cuda(_)),
        "unsupported scoring device"
    );
    ensure!(
        (1..=256).contains(&options.batch_size)
            && !programs.is_empty()
            && programs.len() <= 4096
            && options.keep > 0
            && options.keep <= programs.len(),
        "invalid proposal pool, retained count or scoring batch size"
    );
    let config = &policy.config;
    ensure!(
        config.dataset_dim == 1536
            && config.target_dim == 512
            && training_context.len() == 2048
            && training_context.iter().all(|v| v.is_finite()),
        "invalid saved training context"
    );
    ensure!(
        config.vocab_size == vocabulary.tokens.len() as i64,
        "scoring vocabulary differs from policy"
    );
    for sha in [
        &bindings.checkpoint_sha256,
        &bindings.conditioning_context_sha256,
        &bindings.vocabulary_sha256,
    ] {
        ensure!(
            sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid scoring identity hash"
        );
    }
    let bytes = training_context
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    ensure!(
        hash(&bytes) == bindings.conditioning_context_sha256,
        "scoring context bytes differ from its receipt"
    );
    let mut seen = BTreeSet::new();
    let mut encoded = Vec::with_capacity(programs.len());
    for program in programs {
        crate::campaign::check_deadline()?;
        ensure!(
            seen.insert(program),
            "duplicate generated program in ranking pool"
        );
        feat_math::GenomeIr::parse(program).context("invalid generated GP program")?;
        let tokens = vocabulary
            .encode(program)
            .context("generated program lacks exact frozen tokens")?;
        ensure!(
            tokens.len() > 1
                && tokens.len() <= config.max_seq as usize
                && tokens.first() == Some(&1)
                && tokens.last() == Some(&2),
            "generated program exceeds scoring sequence limit or lacks BOS/EOS"
        );
        encoded.push((hash(program.as_bytes()), program.clone(), tokens));
    }
    // Stable batching makes pool permutations use exactly the same arithmetic.
    encoded.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let pool_sha256 = hash(&serde_json::to_vec(
        &encoded.iter().map(|v| &v.0).collect::<Vec<_>>(),
    )?);
    let context = Tensor::from_slice(training_context)
        .view([1, 2048])
        .to_device(options.device);
    let mut predictions = Vec::with_capacity(encoded.len());
    let mut batches = 0;
    for batch in encoded.chunks(options.batch_size) {
        crate::campaign::check_deadline()?;
        let length = batch.iter().map(|v| v.2.len()).max().unwrap();
        let mut tokens = vec![0i64; batch.len() * length];
        for (i, (_, _, sequence)) in batch.iter().enumerate() {
            for (j, token) in sequence.iter().enumerate() {
                tokens[i * length + j] = *token as i64;
            }
        }
        let tokens = Tensor::from_slice(&tokens)
            .view([batch.len() as i64, length as i64])
            .to_device(options.device);
        let present = tokens.ne(0);
        let contexts = context.repeat([batch.len() as i64, 1]);
        let (quality, runtime, heads) =
            tch::no_grad(|| -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>)> {
                let _autocast = matches!(options.device, Device::Cuda(_)).then(Bf16::enter);
                let (reward, runtime) = policy.reward_runtime(
                    &contexts.narrow(1, 0, 1536),
                    &contexts.narrow(1, 1536, 512),
                    &tokens,
                    &present,
                )?;
                ensure!(
                    reward.size() == [batch.len() as i64, 4]
                        && runtime.size() == [batch.len() as i64],
                    "policy scoring head shape changed"
                );
                // Identical reduction and units to the training objectives.
                let quality = reward.mean_dim(&[1i64][..], false, Kind::Float);
                Ok((
                    Vec::<f32>::try_from(quality.to_device(Device::Cpu))?,
                    Vec::<f32>::try_from(runtime.to_kind(Kind::Float).to_device(Device::Cpu))?,
                    Vec::<Vec<f32>>::try_from(reward.to_kind(Kind::Float).to_device(Device::Cpu))?,
                ))
            })?;
        for (i, (sha, program, sequence)) in batch.iter().enumerate() {
            predictions.push(Prediction {
                program: program.clone(),
                program_sha256: sha.clone(),
                token_count: sequence.len(),
                predicted_quality: quality[i] as f64,
                predicted_runtime_seconds: runtime[i] as f64,
                reward_heads: heads[i]
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            });
        }
        batches += 1;
    }
    let ranked = sorted_predictions(predictions)?;
    Ok(Ranking {
        schema: "parallel.proposal_quality_ranking.v1".into(), selection_id: SELECTION_ID.into(),
        bindings, pool_sha256, pool_size: programs.len(), chosen_count: options.keep,
        scoring_batches: batches, batch_size: options.batch_size,
        backend: if matches!(options.device, Device::Cuda(_)) { "libtorch_cuda_bf16" } else { "libtorch_cpu_float32" }.into(),
        quality_units: "within_context_ranking_logit_not_calibrated_rmse".into(),
        ordering: "mean of four reward heads descending; exact ties by program SHA256 then text; runtime reported without a quality penalty".into(),
        training_context_only: true, heldout_labels_read: false, gp_scores_read: false, ranked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prediction(program: &str, quality: f64) -> Prediction {
        Prediction {
            program: program.into(),
            program_sha256: hash(program.as_bytes()),
            token_count: 5,
            predicted_quality: quality,
            predicted_runtime_seconds: 2.,
            reward_heads: [quality; 4],
        }
    }

    #[test]
    fn quality_wins_over_runtime_and_exact_ties_use_program_identity() {
        let mut better = prediction("better", 2.);
        better.predicted_runtime_seconds = 999.;
        let a = prediction("a", 1.);
        let b = prediction("b", 1.);
        let first = sorted_predictions(vec![a.clone(), better.clone(), b.clone()]).unwrap();
        let second = sorted_predictions(vec![b, a, better]).unwrap();
        assert_eq!(first[0].program, "better");
        assert_eq!(
            first.iter().map(|p| &p.program).collect::<Vec<_>>(),
            second.iter().map(|p| &p.program).collect::<Vec<_>>()
        );
        assert!(first[1].program_sha256 < first[2].program_sha256);
    }

    #[test]
    fn nonfinite_predictions_fail_closed() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(sorted_predictions(vec![prediction("bad", value)]).is_err());
            let mut row = prediction("bad runtime", 1.);
            row.predicted_runtime_seconds = value;
            assert!(sorted_predictions(vec![row]).is_err());
        }
        let mut row = prediction("negative runtime", 1.);
        row.predicted_runtime_seconds = -1.;
        assert!(sorted_predictions(vec![row]).is_err());
    }
}
