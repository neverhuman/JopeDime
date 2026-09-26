//! Fixed-logit sampler replay, kept separate from backend numerical parity (W4-T06).
//!
//! [`super::sample_batch_traced`] records, at every sampling step, the logits row the sampler
//! read, the automaton's legal set and phase, the uniform draw it consumed, and the token it
//! chose or the failure it raised. [`replay`] re-derives every decision from those records
//! alone, with no model and no random stream. Each step is checked against the grammar, and the
//! recorded token or failure must be reproduced exactly.
//!
//! Numerical parity is a different measurement. Two backends can agree within a tensor
//! threshold and still choose different tokens when a uniform draw falls near a sampling
//! boundary. [`max_legal_logit_difference`] reports the tensor scope and [`divergent_decision`]
//! reports the decision, so neither can stand in for the other.
use anyhow::{ensure, Context, Result};
use serde::Serialize;

use super::{choose, legal_weights};
use crate::grammar::Vocab;

/// One sampling step of a traced request.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SamplingStep {
    /// Row of the sampling batch.
    pub row: usize,
    /// Tokens already in the row (proposal prefix included) when this step sampled.
    pub position: usize,
    /// Automaton phase before the step.
    pub grammar_phase: &'static str,
    /// The automaton's legal set before the step, in grammar order.
    pub legal: Vec<usize>,
    /// The full logits row the sampler read.
    pub logits: Vec<f32>,
    pub temperature: f32,
    /// The uniform draw consumed; `None` when the step failed before drawing.
    pub uniform: Option<f64>,
    /// The chosen token; `None` for a failed step.
    pub token: Option<usize>,
    /// The typed failure's text; `None` for a successful step.
    pub failure: Option<String>,
}

/// Replay `steps` for `task`. Each step must continue its row in order, carry exactly the
/// automaton's legal set and phase, and reproduce its recorded token (or, for the final step of
/// a failed request, its recorded failure) from its own logits and uniform draw. Returns each
/// row's tokens, proposal prefix included.
pub fn replay(vocab: &Vocab, task: &str, steps: &[SamplingStep]) -> Result<Vec<Vec<usize>>> {
    let prefix = vocab.proposal_prefix(task)?;
    let start = vocab.prefix_state(task)?;
    let rows = steps.iter().map(|s| s.row + 1).max().unwrap_or(0);
    let mut sequences = vec![prefix; rows];
    let mut states = vec![start; rows];
    for (n, step) in steps.iter().enumerate() {
        let i = step.row;
        ensure!(
            step.position == sequences[i].len(),
            "step {n}: row {i} resumes at {} but has {} tokens",
            step.position,
            sequences[i].len()
        );
        ensure!(!states[i].done(), "step {n}: row {i} sampled after EOS");
        ensure!(
            step.legal == states[i].legal(vocab) && step.grammar_phase == states[i].phase_name(),
            "step {n}: row {i} legal set or phase differs from the grammar"
        );
        match legal_weights(&step.logits, &step.legal, step.temperature) {
            Ok(weights) => {
                let uniform = step
                    .uniform
                    .with_context(|| format!("step {n}: successful step without its draw"))?;
                ensure!(
                    (0.0..1.0).contains(&uniform),
                    "step {n}: uniform draw {uniform} outside [0, 1)"
                );
                let token = choose(&step.legal, &weights, uniform);
                ensure!(
                    step.token == Some(token) && step.failure.is_none(),
                    "step {n}: replay chooses {token}, the trace records {:?}",
                    step.token
                );
                states[i].advance(vocab, token)?;
                sequences[i].push(token);
            }
            Err(failure) => {
                ensure!(
                    step.token.is_none()
                        && step.uniform.is_none()
                        && step.failure.as_deref() == Some(failure.kind.to_string().as_str()),
                    "step {n}: replay fails with {}, the trace records {:?}",
                    failure.kind,
                    step.failure
                );
                ensure!(
                    n + 1 == steps.len(),
                    "step {n}: a failed step must end the trace"
                );
            }
        }
    }
    Ok(sequences)
}

/// The largest absolute difference between the step's legal logits and `other`'s: the tensor
/// parity measurement, scoped to the tokens the sampler can read.
pub fn max_legal_logit_difference(step: &SamplingStep, other: &[f32]) -> Result<f32> {
    step.legal.iter().try_fold(0f32, |worst, &id| {
        let a = *step
            .logits
            .get(id)
            .context("legal token outside the recorded row")?;
        let b = *other
            .get(id)
            .context("legal token outside the compared row")?;
        Ok(worst.max((a - b).abs()))
    })
}

/// The token `other` logits choose with the step's own uniform draw, when it differs from the
/// recorded token: the decision agreement measurement. Tensor parity within a threshold does
/// not imply `None`.
pub fn divergent_decision(step: &SamplingStep, other: &[f32]) -> Result<Option<usize>> {
    let uniform = step.uniform.context("step failed before drawing")?;
    let token = step.token.context("step failed before drawing")?;
    let weights = legal_weights(other, &step.legal, step.temperature)?;
    let alternative = choose(&step.legal, &weights, uniform);
    Ok((alternative != token).then_some(alternative))
}

#[cfg(test)]
mod tests {
    use super::super::{sample_batch, sample_batch_cached, sample_batch_traced, PAD};
    use super::*;
    use crate::config::PolicyConfig;
    use crate::model::PolicyModel;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn vocab() -> Vocab {
        Vocab::from_json(include_bytes!(
            "../../../jope-engine/assets/grammar/featmath.json"
        ))
        .unwrap()
    }

    fn model(vocab: &Vocab) -> PolicyModel {
        let mut config = PolicyConfig::regression(vocab.tokens.len());
        config.width = 16;
        config.heads = 4;
        config.ffn = 32;
        config.layers = 1;
        config.dropout = 0.0;
        PolicyModel::random_weights(config, 3).unwrap()
    }

    fn context() -> Vec<f32> {
        (0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect()
    }

    fn traced(
        cached: bool,
    ) -> (
        Vec<super::super::SampledRow>,
        Vec<SamplingStep>,
        ChaCha20Rng,
    ) {
        let vocab = vocab();
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let mut trace = Vec::new();
        let rows = sample_batch_traced(
            &model(&vocab),
            &vocab,
            &context(),
            "regression",
            24,
            25.0,
            cached,
            &mut rng,
            &mut trace,
        )
        .unwrap();
        (rows, trace, rng)
    }

    #[test]
    fn tracing_changes_nothing_and_fixed_logit_replay_reproduces_every_token() {
        let vocab = vocab();
        let model = model(&vocab);
        for cached in [false, true] {
            let mut plain_rng = ChaCha20Rng::seed_from_u64(7);
            let plain = if cached {
                sample_batch_cached(
                    &model,
                    &vocab,
                    &context(),
                    "regression",
                    0,
                    24,
                    25.0,
                    &mut plain_rng,
                )
            } else {
                sample_batch(
                    &model,
                    &vocab,
                    &context(),
                    "regression",
                    24,
                    25.0,
                    &mut plain_rng,
                )
            }
            .unwrap();
            let (rows, trace, mut rng) = traced(cached);
            for (a, b) in plain.iter().zip(&rows) {
                assert_eq!((&a.tokens, &a.program), (&b.tokens, &b.program));
            }
            for _ in 0..8 {
                assert_eq!(plain_rng.next_u64(), rng.next_u64(), "random stream tail");
            }
            // Cold replay from the records alone.
            let replayed = replay(&vocab, "regression", &trace).unwrap();
            for (row, tokens) in rows.iter().zip(&replayed) {
                let sampled = row
                    .tokens
                    .iter()
                    .copied()
                    .take_while(|&t| t != PAD)
                    .collect::<Vec<_>>();
                assert_eq!(&sampled, tokens);
            }
            assert!(trace.iter().all(|s| s.token.is_some()));
        }
    }

    #[test]
    fn cached_and_full_routes_share_draws_and_decisions_within_tensor_parity() {
        let (_, full, _) = traced(false);
        let (_, cached, _) = traced(true);
        assert_eq!(full.len(), cached.len());
        for (a, b) in full.iter().zip(&cached) {
            assert_eq!(
                (a.row, a.position, a.uniform, a.token),
                (b.row, b.position, b.uniform, b.token)
            );
            assert!(max_legal_logit_difference(a, &b.logits).unwrap() < 1e-4);
            assert_eq!(divergent_decision(a, &b.logits).unwrap(), None);
        }
    }

    #[test]
    fn a_decision_near_a_boundary_diverges_inside_the_tensor_threshold() {
        let (_, trace, _) = traced(false);
        let base = trace
            .iter()
            .find(|s| s.legal.len() >= 2)
            .expect("fixture has a step with a real choice")
            .clone();
        let (a, b) = (base.legal[0], base.legal[1]);
        // Two equal legal logits and every other legal token at -inf: the draw sits 2e-5 below
        // the boundary between `a` and `b`.
        let mut step = base;
        step.logits = vec![f32::NEG_INFINITY; step.logits.len()];
        step.logits[a] = 0.0;
        step.logits[b] = 0.0;
        step.temperature = 1.0;
        step.uniform = Some(0.49998);
        step.token = Some(a);
        let mut other = step.logits.clone();
        other[b] = 2e-4;
        let difference = max_legal_logit_difference(&step, &other).unwrap();
        assert!(difference <= 3e-4, "within the admission tensor threshold");
        assert_eq!(divergent_decision(&step, &other).unwrap(), Some(b));
        assert_eq!(
            divergent_decision(&step, &step.logits.clone()).unwrap(),
            None
        );
    }

    #[test]
    fn altered_traces_are_refused_and_failures_replay_exactly() {
        let vocab = vocab();
        let (_, trace, _) = traced(false);
        let choice = trace.iter().position(|s| s.legal.len() >= 2).unwrap();
        let other = |s: &SamplingStep| *s.legal.iter().find(|&&t| Some(t) != s.token).unwrap();
        let mut cases: Vec<(&str, Vec<SamplingStep>)> = Vec::new();
        let mut token = trace.clone();
        token[choice].token = Some(other(&token[choice]));
        cases.push(("replay chooses", token));
        let mut legal = trace.clone();
        legal[choice].legal.pop();
        cases.push(("legal set or phase", legal));
        let mut reordered = trace.clone();
        reordered.swap(0, 24);
        cases.push(("resumes at", reordered));
        let mut dropped = trace.clone();
        dropped[choice].uniform = None;
        cases.push(("without its draw", dropped));
        let mut out_of_range = trace.clone();
        out_of_range[choice].uniform = Some(1.0);
        cases.push(("outside [0, 1)", out_of_range));
        for (reason, steps) in cases {
            let err = replay(&vocab, "regression", &steps).unwrap_err();
            assert!(format!("{err:#}").contains(reason), "{reason}: {err:#}");
        }

        // A NaN legal logit ends the request; its recorded failure replays exactly.
        let mut failed = trace[..=choice].to_vec();
        let last = failed.last_mut().unwrap();
        let bad = last.legal[0];
        last.logits[bad] = f32::NAN;
        let text = format!("NaN logit for legal token {bad}");
        (last.uniform, last.token, last.failure) = (None, None, Some(text.clone()));
        replay(&vocab, "regression", &failed).unwrap();
        failed.last_mut().unwrap().failure = Some("empty legal token set".into());
        assert!(replay(&vocab, "regression", &failed).is_err());
        let mut continued = trace[..=choice + 1].to_vec();
        continued[choice].logits[bad] = f32::NAN;
        (
            continued[choice].uniform,
            continued[choice].token,
            continued[choice].failure,
        ) = (None, None, Some(text));
        let err = replay(&vocab, "regression", &continued).unwrap_err();
        assert!(format!("{err:#}").contains("must end the trace"), "{err:#}");
    }
}
