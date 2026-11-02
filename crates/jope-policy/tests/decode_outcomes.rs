//! W4-T03: decoding terminates with an explicit, bounded outcome (CPU/Candle path only).
//!
//! The fixtures drive the real decode entry points (`decode::sample_batch` and the testing
//! `pure_pool_unbound`, which shares `pure_pool`'s checked body) with a *fixed-logit* policy:
//! `out_norm` maps every hidden state to a vector of ones, so the tied projection returns each
//! `token_emb` row sum and the next-token logits do not depend on context, position or history.
//! Row `t` is `[logit(t), 0, ..., 0]`, so a fixture sets every token's logit exactly, including
//! NaN, ±Inf and preference orders separated by 1000 logit units (every non-preferred
//! weight underflows to exactly zero in f64, which makes decoding deterministic). A single large
//! entry keeps the embedded input well conditioned for LayerNorm; spreading the logit evenly
//! over the row (equal large entries, near-zero variance) produced NaN hidden states.
//!
//! No CUDA, LibTorch or accepted checkpoint is involved: this is fixture proof of the Candle
//! sampler, not native parity or live qualification.
#![cfg(feature = "testing")]

use candle_core::{Device, Tensor};
use jope_policy::config::PolicyConfig;
use jope_policy::context::PolicyContext;
use jope_policy::decode::{
    self, DecodeFailure, DrawId, DrawOutcome, FailureKind, SamplingOptions, FALLBACK_ATTEMPTS,
    MAX_BATCH, MAX_ROUNDS, POOL_SIZE,
};
use jope_policy::grammar::{bind_vocabulary, Slot, Vocab, EOS, MAX_PROGRAM_TOKENS};
use jope_policy::model::{expected_inventory, PolicyModel};
use jope_policy::seeds::{ORIGIN_NEURAL, ORIGIN_NEURAL_FALLBACK};
use jope_policy::testing::pure_pool_unbound;
use jope_policy::{sha256_hex, CHECKPOINT_VOCABULARY_SHA256};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeSet;
use std::path::Path;

/// `(stack)`: a regression head without slots, the largest regression head id.
const STACK: usize = 372;
/// Regression heads with many slot-value combinations (gbm, gpr, histgbm, histgbm2).
const WIDE_HEADS: [usize; 4] = [245, 257, 273, 286];
const PREFIX: [usize; 3] = [1, 3, 397];

fn vocab() -> Vocab {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar/featmath.json");
    bind_vocabulary(&std::fs::read(path).unwrap(), CHECKPOINT_VOCABULARY_SHA256)
        .unwrap()
        .vocab()
        .clone()
}

fn context() -> PolicyContext {
    PolicyContext::from_frozen_values((0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect())
        .unwrap()
}

fn config(max_seq: usize) -> PolicyConfig {
    let mut config = PolicyConfig::regression(401);
    config.width = 16;
    config.heads = 4;
    config.ffn = 32;
    config.layers = 1;
    config.dropout = 0.0;
    config.max_seq = max_seq;
    config
}

/// A policy whose next-token logit for token `t` is exactly `logit(t)` at every step.
fn fixed_logit_model(max_seq: usize, logit: impl Fn(usize) -> f32) -> PolicyModel {
    let config = config(max_seq);
    let width = config.width;
    let tensors = expected_inventory(&config)
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let numel = shape.iter().product::<usize>();
            let values = match name.as_str() {
                "out_norm.weight" => vec![0.0; numel],
                "out_norm.bias" => vec![1.0; numel],
                "token_emb.weight" => (0..numel)
                    .map(|j| {
                        if j % width == 0 {
                            logit(j / width)
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                _ => (0..numel)
                    .map(|j| ((i * 131 + j) as f32 * 0.37).sin() * 0.2)
                    .collect::<Vec<f32>>(),
            };
            let tensor = if shape.is_empty() {
                Tensor::new(values[0], &Device::Cpu)
            } else {
                Tensor::from_vec(values, shape, &Device::Cpu)
            }
            .unwrap();
            (name, tensor)
        })
        .collect::<Vec<_>>();
    PolicyModel::from_tensors(config, tensors, Device::Cpu).unwrap()
}

/// Deterministic logits: `preferred` tokens first, in order, then every other token by
/// descending id; adjacent ranks are separated by 1000 logit units.
fn ranked(preferred: &[usize]) -> impl Fn(usize) -> f32 + '_ {
    move |t| match preferred.iter().position(|&p| p == t) {
        Some(rank) => -1000.0 * rank as f32,
        None => -1000.0 * (preferred.len() + 401 - t.min(401)) as f32,
    }
}

/// Independent reference: the automaton walk that takes the legal token with the highest fixed
/// logit, stopped at `budget` tokens.
fn deterministic_walk(vocab: &Vocab, logit: &dyn Fn(usize) -> f32, budget: usize) -> Vec<usize> {
    let mut tokens = vocab.proposal_prefix("regression").unwrap();
    let mut state = vocab.prefix_state("regression").unwrap();
    while !state.done() && tokens.len() < budget {
        let legal = state.legal(vocab);
        let best = *legal
            .iter()
            .max_by(|a, b| logit(**a).total_cmp(&logit(**b)))
            .unwrap();
        state.advance(vocab, best).unwrap();
        tokens.push(best);
    }
    tokens
}

fn options(rounds: usize, batch: usize) -> SamplingOptions {
    SamplingOptions {
        rounds,
        batch,
        temperature: 1.0,
    }
}

/// The typed failure inside a refused request.
fn failure<T: std::fmt::Debug>(result: anyhow::Result<T>) -> DecodeFailure {
    let error = result.expect_err("the request was not refused");
    error
        .downcast_ref::<DecodeFailure>()
        .unwrap_or_else(|| panic!("untyped decode failure: {error:#}"))
        .clone()
}
mod logit_support;
mod sampling_limits;
mod terminal_outcomes;
mod token_budget;
