//! W4-T03: decoding terminates with an explicit, bounded outcome (CPU/Candle path only).
//!
//! The fixtures drive the real decode entry points (`decode::sample_batch` and the testing
//! `pure_pool_unbound`, which shares `pure_pool`'s checked body) with a *fixed-logit* policy:
//! `out_norm` maps every hidden state to a vector of ones, so the tied projection returns each
//! `token_emb` row sum and the next-token logits do not depend on context, position or history.
//! Row `t` is `[logit(t), 0, ..., 0]`, so a fixture sets every token's logit exactly, including
//! NaN, ±Inf and preference orders whose adjacent ranks are 1000 nats apart (every non-preferred
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
/// descending id; adjacent ranks are 1000 nats apart.
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

#[test]
fn pure_pool_refuses_an_unbounded_sampling_budget_before_any_draw() {
    let vocab = vocab();
    let model = fixed_logit_model(128, ranked(&[]));
    let ctx = context();
    // Positive control: jope-dime's own plan (8 rounds of 128 rows) is the largest accepted
    // budget. One distinct program is ever drawn, so every round runs.
    let (pool, _) = pure_pool_unbound(
        &model,
        &vocab,
        &ctx,
        "regression",
        1,
        2,
        &SamplingOptions::default(),
    )
    .unwrap();
    assert_eq!((MAX_ROUNDS, MAX_BATCH), (8, 128));
    assert_eq!(pool.attempted, 1024);
    assert_eq!(pool.unique_neural, 1);
    assert_eq!(pool.draws().len(), 1024);
    assert_eq!(pool.count(DrawOutcome::Completed), 1);
    assert_eq!(pool.count(DrawOutcome::Duplicate), 1023);
    assert_eq!(pool.draws()[1023].draw, DrawId { round: 7, row: 127 });
    // Without the bound the cases after (9, 1) never terminate or fail allocation.
    for (rounds, batch) in [
        (0, 1),
        (1, 0),
        (9, 1),
        (1, 129),
        (usize::MAX, 1),
        (1, usize::MAX),
    ] {
        let refused = failure(pure_pool_unbound(
            &model,
            &vocab,
            &ctx,
            "regression",
            1,
            2,
            &options(rounds, batch),
        ));
        assert_eq!(refused.kind, FailureKind::InvalidBudget { rounds, batch });
        assert_eq!(refused.draw, None);
    }
}

#[test]
fn sample_batch_refuses_an_unbounded_batch() {
    let vocab = vocab();
    let model = fixed_logit_model(128, ranked(&[]));
    let ctx = context();
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    assert_eq!(
        decode::sample_batch(&model, &vocab, &ctx, "regression", 128, 1.0, &mut rng)
            .unwrap()
            .len(),
        128
    );
    for batch in [0, 129] {
        let mut untouched = rng.clone();
        let refused = failure(decode::sample_batch(
            &model,
            &vocab,
            &ctx,
            "regression",
            batch,
            1.0,
            &mut rng,
        ));
        assert_eq!(
            refused.kind,
            FailureKind::InvalidBudget { rounds: 1, batch }
        );
        assert_eq!(rng.next_u64(), untouched.next_u64());
    }
}

#[test]
fn invalid_temperature_and_missing_token_budget_are_refused_without_a_token_step() {
    let vocab = vocab();
    let ctx = context();
    // Token budget min(128, max_seq) == 3 == the proposal prefix: no token can be sampled.
    let short = fixed_logit_model(3, ranked(&[]));
    let mut rng = ChaCha20Rng::seed_from_u64(5);
    let refused = failure(decode::sample_batch(
        &short,
        &vocab,
        &ctx,
        "regression",
        2,
        f32::NAN,
        &mut rng,
    ));
    assert!(matches!(refused.kind, FailureKind::InvalidTemperature(t) if t.is_nan()));
    let refused = failure(decode::sample_batch(
        &short,
        &vocab,
        &ctx,
        "regression",
        2,
        1.0,
        &mut rng,
    ));
    assert_eq!(
        refused.kind,
        FailureKind::NoTokenBudget {
            prefix: 3,
            budget: 3
        }
    );
    // The pool does not substitute grammar-uniform fallbacks for a policy that cannot sample.
    let refused = failure(pure_pool_unbound(
        &short,
        &vocab,
        &ctx,
        "regression",
        1,
        2,
        &options(1, 2),
    ));
    assert_eq!(refused.seed, Some(1));
    assert!(matches!(refused.kind, FailureKind::NoTokenBudget { .. }));
}

/// Regression protection: an invalid temperature is refused, consumes no random draw and never
/// becomes a greedy/argmax mode; a tiny positive temperature is ordinary sampling.
#[test]
fn invalid_temperature_never_draws_and_is_not_a_greedy_mode() {
    let vocab = vocab();
    let ctx = context();
    let model = fixed_logit_model(128, ranked(&[]));
    for temperature in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        let mut untouched = rng.clone();
        let refused = failure(decode::sample_batch(
            &model,
            &vocab,
            &ctx,
            "regression",
            3,
            temperature,
            &mut rng,
        ));
        assert!(matches!(refused.kind, FailureKind::InvalidTemperature(_)));
        assert_eq!(
            rng.next_u64(),
            untouched.next_u64(),
            "temperature {temperature} drew"
        );
        let options = SamplingOptions {
            rounds: 1,
            batch: 3,
            temperature,
        };
        assert!(options.validate().is_err());
        let refused = failure(pure_pool_unbound(
            &model,
            &vocab,
            &ctx,
            "regression",
            1,
            2,
            &options,
        ));
        assert!(matches!(refused.kind, FailureKind::InvalidTemperature(_)));
    }
    let walk = deterministic_walk(&vocab, &ranked(&[]), 128);
    let mut rng = ChaCha20Rng::seed_from_u64(9);
    let tiny = f32::from_bits(1);
    for row in decode::sample_batch(&model, &vocab, &ctx, "regression", 3, tiny, &mut rng).unwrap()
    {
        assert_eq!(row.program.as_ref(), Some(&walk));
    }
}

#[test]
fn nonfinite_logits_and_empty_support_fail_with_the_draw_identity() {
    let vocab = vocab();
    let ctx = context();
    let base = ranked(&[]);
    // Deterministic walk BOS featmath regression stack EOS.
    assert_eq!(
        deterministic_walk(&vocab, &base, 128),
        [1, 3, 397, STACK, EOS]
    );
    let nan_eos = fixed_logit_model(128, |t| if t == EOS { f32::NAN } else { base(t) });
    let inf_stack = fixed_logit_model(128, |t| if t == STACK { f32::INFINITY } else { base(t) });
    // Only the proposal prefix keeps finite logits (it is embedded as input).
    let no_support = fixed_logit_model(128, |t| {
        if PREFIX.contains(&t) {
            0.0
        } else {
            f32::NEG_INFINITY
        }
    });
    let cases: [(&PolicyModel, &[usize], &str); 3] = [
        (&nan_eos, &[1, 3, 397, STACK], "eos"),
        (&inf_stack, &PREFIX, "stage"),
        (&no_support, &PREFIX, "stage"),
    ];
    for (case, (model, tokens, phase)) in cases.into_iter().enumerate() {
        let check = |refused: &DecodeFailure| {
            match (case, &refused.kind) {
                (0, FailureKind::NonfiniteLogit { token, value }) => {
                    assert!(*token == EOS && value.is_nan())
                }
                (1, FailureKind::NonfiniteLogit { token, value }) => {
                    assert!(*token == STACK && *value == f32::INFINITY)
                }
                (2, FailureKind::EmptySupport) => {}
                (_, kind) => panic!("case {case}: unexpected failure {kind:?}"),
            }
            assert_eq!(refused.draw, Some(DrawId { round: 0, row: 0 }));
            assert_eq!(refused.tokens, tokens);
            assert_eq!(refused.grammar_phase, Some(phase));
        };
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let mut untouched = rng.clone();
        let result = decode::sample_batch(model, &vocab, &ctx, "regression", 4, 1.0, &mut rng);
        let message = format!("{:#}", result.as_ref().unwrap_err());
        assert!(message.contains("round 0, row 0"), "{message}");
        check(&failure(result));
        if case > 0 {
            // The first step failed before any random draw.
            assert_eq!(rng.next_u64(), untouched.next_u64());
        }
        // The pool entry fails too: no fallback program replaces a numerically invalid draw.
        let refused = failure(pure_pool_unbound(
            model,
            &vocab,
            &ctx,
            "regression",
            1,
            2,
            &options(2, 4),
        ));
        check(&refused);
        assert_eq!(refused.seed, Some(1));
    }
}

/// Regression protection: NaN or infinite logits of tokens that are never legal are masked
/// exactly as the reference `masked_fill(-inf)` masks them, so decoding is unchanged.
#[test]
fn nonfinite_logits_outside_the_legal_support_do_not_change_decoding() {
    let vocab = vocab();
    let ctx = context();
    let base = ranked(&[]);
    // PAD, the generate=false heads portfolio and ridge_gcv, a multiclass-only head and the
    // binary task token are never legal after the regression prefix (the prefix tokens
    // themselves are embedded as input, so they keep finite rows).
    let model = fixed_logit_model(128, |t| match t {
        0 | 369 | 371 | 324 => f32::NAN,
        398 => f32::INFINITY,
        _ => base(t),
    });
    let walk = deterministic_walk(&vocab, &base, 128);
    let mut rng = ChaCha20Rng::seed_from_u64(2);
    for row in decode::sample_batch(&model, &vocab, &ctx, "regression", 4, 1.0, &mut rng).unwrap() {
        assert_eq!(row.program.as_ref(), Some(&walk));
    }
}

/// Regression protection for the sampler arithmetic: extreme finite logits and exact ties never
/// move a draw outside the legal support, and tied tokens share the draws.
#[test]
fn extreme_finite_logits_and_ties_stay_inside_the_legal_support() {
    let logits = [f32::MAX, f32::MAX, -f32::MAX, 1e30, -1e30];
    let mut rng = ChaCha20Rng::seed_from_u64(12);
    let mut seen = BTreeSet::new();
    for _ in 0..256 {
        let token = decode::sample_legal(&logits, &[1, 2, 4], 1.0, &mut rng).unwrap();
        assert_eq!(token, 1);
        seen.insert(decode::sample_legal(&logits, &[0, 1, 2], 1.0, &mut rng).unwrap());
        let token = decode::sample_legal(&logits, &[2, 4], f32::from_bits(1), &mut rng).unwrap();
        assert_eq!(token, 4);
    }
    assert_eq!(seen, BTreeSet::from([0, 1]));
    // Through the decoder: three slot-free heads tie; every other stage token sits 1e30 below.
    let vocab = vocab();
    let ctx = context();
    let tied = [215usize, 370, STACK];
    let stage = vocab.prefix_state("regression").unwrap().legal(&vocab);
    let model = fixed_logit_model(128, |t| {
        if stage.contains(&t) && !tied.contains(&t) {
            -1e30
        } else {
            0.0
        }
    });
    let mut rng = ChaCha20Rng::seed_from_u64(13);
    let rows = decode::sample_batch(&model, &vocab, &ctx, "regression", 96, 1.0, &mut rng).unwrap();
    for head in tied {
        let n = rows.iter().filter(|r| r.tokens[3] == head).count();
        assert!(n >= 16, "tied head {head} drawn {n} of 96 times");
    }
    assert!(rows.iter().all(|r| tied.contains(&r.tokens[3])
        && r.program.as_deref() == Some(&[1, 3, 397, r.tokens[3], EOS][..])));
}

fn dead_end_vocab() -> Vocab {
    let mut vocab = vocab();
    // Adversarial vocabulary: `stack` gains a slot with no value and no absent token.
    vocab.ops.get_mut(&STACK).unwrap().slots = vec![Slot {
        param: "dead_end".into(),
        kind: "int".into(),
        val_ids: vec![],
        bin_texts: vec![],
        absent_id: None,
        is_expr: false,
        required: true,
        default: serde_json::Value::Null,
    }];
    vocab
}

#[test]
fn grammar_dead_end_fails_with_the_draw_identity() {
    let vocab = dead_end_vocab();
    let ctx = context();
    let model = fixed_logit_model(128, ranked(&[]));
    let mut rng = ChaCha20Rng::seed_from_u64(4);
    let result = decode::sample_batch(&model, &vocab, &ctx, "regression", 2, 1.0, &mut rng);
    let message = format!("{:#}", result.as_ref().unwrap_err());
    assert!(message.contains("round 0, row 0"), "{message}");
    let refused = failure(result);
    assert_eq!(refused.kind, FailureKind::EmptySupport);
    assert_eq!(refused.draw, Some(DrawId { round: 0, row: 0 }));
    assert_eq!(refused.tokens, [1, 3, 397, STACK]);
    assert_eq!(refused.grammar_phase, Some("value"));
    // The grammar-uniform generator stops at the same dead end with a typed failure.
    let refused = (0..2000u64)
        .find_map(|seed| {
            let mut rng = <rand_chacha::ChaCha8Rng as SeedableRng>::seed_from_u64(seed);
            decode::grammar_uniform_program(&vocab, "regression", &mut rng)
                .err()
                .map(|e| e.downcast_ref::<DecodeFailure>().cloned().expect("typed"))
        })
        .expect("a uniform walk reaches the dead end");
    assert_eq!(refused.kind, FailureKind::EmptySupport);
    assert_eq!(refused.draw, None);
    assert_eq!(refused.tokens.last(), Some(&STACK));
    assert_eq!(refused.grammar_phase, Some("value"));
}

#[test]
fn every_valid_draw_is_accounted_for() {
    let vocab = vocab();
    let ctx = context();
    // Uniform over four wide regression heads and their values; every other stage token has a
    // -Inf logit (zero probability, never embedded).
    let heads = vocab.engine_heads["featmath"].clone();
    let transforms = vocab.engine_transforms["featmath"].clone();
    let model = fixed_logit_model(128, |t| {
        if transforms.contains(&t) || (heads.contains(&t) && !WIDE_HEADS.contains(&t)) {
            f32::NEG_INFINITY
        } else {
            0.0
        }
    });
    let (pool, _) =
        pure_pool_unbound(&model, &vocab, &ctx, "regression", 21, 22, &options(1, 128)).unwrap();
    assert_eq!(pool.attempted, 128);
    assert_eq!(pool.unique_neural, 64);
    let surplus = pool.count(DrawOutcome::Surplus);
    assert!(surplus > 0, "fixture must fill the pool mid-round");
    // Every valid draw is pooled, a duplicate, or explicitly surplus.
    assert_eq!(
        pool.valid,
        pool.unique_neural + pool.within_neural_duplicates + surplus
    );
    assert_eq!(pool.count(DrawOutcome::Completed), pool.unique_neural);
    assert_eq!(
        pool.count(DrawOutcome::Duplicate),
        pool.within_neural_duplicates
    );
    // Completed draws are the pooled neural candidates, in order; surplus programs are unique
    // and never pooled; duplicates repeat an earlier valid draw.
    let completed = pool
        .draws()
        .iter()
        .filter(|d| d.outcome == DrawOutcome::Completed)
        .map(|d| d.program_sha256.clone().unwrap())
        .collect::<Vec<_>>();
    let pooled = pool.candidates()[..64]
        .iter()
        .map(|c| sha256_hex(c.genome().as_bytes()))
        .collect::<Vec<_>>();
    assert_eq!(completed, pooled);
    let mut earlier = BTreeSet::new();
    for record in pool.draws() {
        let sha = record.program_sha256.clone().unwrap();
        match record.outcome {
            DrawOutcome::Duplicate => assert!(earlier.contains(&sha)),
            DrawOutcome::Surplus => assert!(!pooled.contains(&sha) && earlier.insert(sha)),
            DrawOutcome::Completed => assert!(earlier.insert(sha)),
            outcome => panic!("unexpected {outcome:?}"),
        }
    }
    assert_eq!(pool.fallback_attempts, 0);
}

#[test]
fn every_scheduled_draw_has_exactly_one_terminal_outcome() {
    let vocab = vocab();
    let ctx = context();
    // Untrained weights at a high temperature with a 12-token budget: a mix of completed,
    // duplicate and incomplete draws.
    let model = PolicyModel::random_weights(config(12), 3).unwrap();
    let sampling = SamplingOptions {
        rounds: 2,
        batch: 32,
        temperature: 25.0,
    };
    let (pool, seen) =
        pure_pool_unbound(&model, &vocab, &ctx, "regression", 31, 32, &sampling).unwrap();
    assert_eq!(pool.token_budget, 12);
    assert_eq!((pool.options.rounds, pool.options.batch), (2, 32));
    assert_eq!(pool.attempted, 64);
    let draws = pool.draws();
    assert_eq!(draws.len(), pool.attempted);
    for (i, record) in draws.iter().enumerate() {
        assert_eq!(
            record.draw,
            DrawId {
                round: i / 32,
                row: i % 32
            }
        );
        assert_eq!(record.tokens[..3], PREFIX);
        // Replay: every sampled token was in the sampling-mode legal support.
        let mut state = vocab.prefix_state("regression").unwrap();
        for &token in &record.tokens[3..] {
            assert!(state.legal(&vocab).contains(&token));
            state.advance(&vocab, token).unwrap();
        }
        assert_eq!(state.phase_name(), record.grammar_phase);
        match record.outcome {
            DrawOutcome::Incomplete => {
                assert_eq!(record.tokens.len(), 12);
                assert_ne!(record.grammar_phase, "done");
                assert_eq!(record.program_sha256, None);
            }
            DrawOutcome::ParserRejected => panic!("the frozen grammar always parses here"),
            _ => {
                assert_eq!(record.grammar_phase, "done");
                assert_eq!(record.tokens.last(), Some(&EOS));
                let genome = decode::valid_genome(&vocab, &record.tokens).unwrap();
                assert!(seen.contains(&genome));
                assert_eq!(
                    record.program_sha256.as_deref(),
                    Some(sha256_hex(genome.as_bytes()).as_str())
                );
            }
        }
    }
    let incomplete = pool.count(DrawOutcome::Incomplete);
    assert!(incomplete > 0 && pool.count(DrawOutcome::Completed) > 0);
    assert_eq!(
        pool.attempted - pool.valid,
        incomplete + pool.count(DrawOutcome::ParserRejected)
    );
    assert_eq!(
        pool.sampled_tokens,
        draws.iter().map(|d| d.tokens.len() - 3).sum::<usize>()
    );
    // Fallback is bounded, labelled, and fills exactly the slots the policy left.
    assert!(pool.fallback_attempts > 0 && pool.fallback_attempts <= FALLBACK_ATTEMPTS);
    let origins = pool
        .candidates()
        .iter()
        .map(|c| c.origin())
        .collect::<Vec<_>>();
    assert_eq!(origins.len(), POOL_SIZE);
    assert!(origins[..pool.unique_neural]
        .iter()
        .all(|o| *o == ORIGIN_NEURAL));
    assert!(origins[pool.unique_neural..]
        .iter()
        .all(|o| *o == ORIGIN_NEURAL_FALLBACK));
}

#[test]
fn parser_rejection_is_an_explicit_outcome() {
    let mut vocab = vocab();
    // Adversarial vocabulary: the head name no longer renders as valid DSL.
    vocab.ops.get_mut(&STACK).unwrap().name = "stack)".into();
    assert!(decode::valid_genome(&vocab, &[1, 3, 397, STACK, EOS]).is_none());
    let ctx = context();
    let model = fixed_logit_model(128, ranked(&[]));
    let (pool, _) =
        pure_pool_unbound(&model, &vocab, &ctx, "regression", 5, 6, &options(1, 4)).unwrap();
    assert_eq!((pool.attempted, pool.valid, pool.unique_neural), (4, 0, 0));
    assert_eq!(pool.count(DrawOutcome::ParserRejected), 4);
    for record in pool.draws() {
        assert_eq!(record.tokens, [1, 3, 397, STACK, EOS]);
        assert_eq!(record.grammar_phase, "done");
        assert_eq!(record.program_sha256, None);
    }
    // Never relabelled: the pool is all fallback.
    assert!(pool
        .candidates()
        .iter()
        .all(|c| c.origin() == ORIGIN_NEURAL_FALLBACK));
}

/// Preference order that builds the longest program the sampling grammar admits: expression
/// stages first, then the transforms and head with the most slots; inside expressions, functions
/// of the highest arity.
fn longest_program_order(vocab: &Vocab) -> Vec<usize> {
    let length = |id: usize| {
        let op = &vocab.ops[&id];
        1 + op
            .slots
            .iter()
            .map(|s| {
                if s.is_expr {
                    vocab.expr.max_nodes + 1
                } else {
                    1
                }
            })
            .sum::<usize>()
            + usize::from(op.allow_append)
    };
    let mut order = (0..401).collect::<Vec<usize>>();
    order.sort_by_key(|&t| {
        let key = if vocab.ops.contains_key(&t) {
            (2, length(t))
        } else if let Some(&arity) = vocab.expr.func_arity.get(&t) {
            (1, arity)
        } else {
            (0, 0)
        };
        (std::cmp::Reverse(key), t)
    });
    order
}

#[test]
fn token_budget_boundary_is_exact() {
    let vocab = vocab();
    let ctx = context();
    let ascending = (0..401).collect::<Vec<_>>();
    let longest = longest_program_order(&vocab);
    let mut lengths = Vec::new();
    for logit in [
        Box::new(ranked(&[])) as Box<dyn Fn(usize) -> f32>,
        Box::new(ranked(&ascending)),
        Box::new(ranked(&longest)),
    ] {
        let walk = deterministic_walk(&vocab, &*logit, MAX_PROGRAM_TOKENS);
        assert_eq!(walk.last(), Some(&EOS));
        let n = walk.len();
        lengths.push(n);
        // Exactly at the budget: EOS is the last admissible token and the draw completes, on
        // the cached pool path and (for the shorter walks; it recomputes every prefix, which is
        // slow in a debug build) on the full-prefix reference path.
        let at = fixed_logit_model(n, &*logit);
        let over = fixed_logit_model(n - 1, &*logit);
        if n <= 32 {
            let mut rng = ChaCha20Rng::seed_from_u64(6);
            let rows = decode::sample_batch(&at, &vocab, &ctx, "regression", 1, 1.0, &mut rng);
            let [row] = &rows.unwrap()[..] else {
                panic!("one row requested")
            };
            assert_eq!(row.program.as_ref(), Some(&walk));
            assert_eq!(row.tokens, walk);
            assert_eq!(row.grammar_phase, "done");
            // One token short: the accepting EOS cannot be drawn and the draw is incomplete.
            let mut rng = ChaCha20Rng::seed_from_u64(6);
            let rows = decode::sample_batch(&over, &vocab, &ctx, "regression", 1, 1.0, &mut rng);
            let [row] = &rows.unwrap()[..] else {
                panic!("one row requested")
            };
            assert_eq!(row.program, None);
            assert_eq!(row.tokens, walk[..n - 1]);
            assert_eq!(row.grammar_phase, "eos");
        }
        let (pool, _) =
            pure_pool_unbound(&at, &vocab, &ctx, "regression", 1, 2, &options(1, 2)).unwrap();
        assert_eq!(pool.count(DrawOutcome::Completed), 1);
        assert_eq!(pool.count(DrawOutcome::Duplicate), 1);
        assert!(pool
            .draws()
            .iter()
            .all(|d| d.tokens == walk && d.grammar_phase == "done"));
        let (pool, _) =
            pure_pool_unbound(&over, &vocab, &ctx, "regression", 1, 2, &options(2, 2)).unwrap();
        assert_eq!((pool.attempted, pool.valid, pool.unique_neural), (4, 0, 0));
        assert_eq!(pool.count(DrawOutcome::Incomplete), 4);
        assert!(pool
            .draws()
            .iter()
            .all(|d| d.tokens == walk[..n - 1] && d.grammar_phase == "eos"));
    }
    // The longest program the sampling grammar admits (see `grammar_differential`) fits the
    // 128-token budget with room to spare.
    assert_eq!(lengths, [5, 25, 78]);
}
