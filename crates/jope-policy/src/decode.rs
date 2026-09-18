//! Batched grammar-masked decoding and the pure neural pool (jope-dime `proposals.rs:748-875`
//! and `proposal_population.rs:9-50`).
//!
//! Each round starts `batch` rows at `[BOS, featmath, task]`, and at every step samples the
//! next token of each unfinished row from `softmax(logits / temperature)` restricted to the
//! automaton's legal set, so every produced token is legal by construction. Finished rows are
//! padded with PAD (the only token jope-dime leaves legal for them) and are not recomputed.
//! Rows that have not reached EOS within 128 tokens are discarded. A finished row counts as
//! valid when it decodes and parses as a genome; the first 64 distinct valid programs form the
//! pure pool (origin `neural_policy`), and a short pool is filled with grammar-uniform programs
//! (origin `neural_rejection_fallback`).
//!
//! Sampling uses a seeded ChaCha20 stream, not torch's generator (see the module docs).
use anyhow::{bail, ensure, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::{ChaCha20Rng, ChaCha8Rng};
use std::collections::BTreeSet;

use super::grammar::{Vocab, EOS, MAX_PROGRAM_TOKENS, PAD};
use super::model::PolicyModel;
use super::seeds::{SeedCandidate, ORIGIN_NEURAL, ORIGIN_NEURAL_FALLBACK, ORIGIN_RANDOM};

/// jope-dime's pure pool protocol identifier.
pub const PURE_PROTOCOL: &str = "training_context_neural_pool_v1";
/// Number of pure (neural-arm) programs.
pub const POOL_SIZE: usize = 64;
/// Number of exploration programs appended by the seed writer.
pub const EXPLORATION_SIZE: usize = 16;

/// Sampling shape; the defaults are jope-dime's (8 rounds of 128 rows, temperature 1).
#[derive(Clone, Debug)]
pub struct SamplingOptions {
    pub rounds: usize,
    pub batch: usize,
    pub temperature: f32,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            rounds: 8,
            batch: 128,
            temperature: 1.0,
        }
    }
}

/// One sampled row: the tokens (PAD-padded to the batch length) and, when it reached EOS, the
/// program tokens `BOS..=EOS`.
#[derive(Clone, Debug)]
pub struct SampledRow {
    pub tokens: Vec<usize>,
    pub program: Option<Vec<usize>>,
}

/// Draw one token from the legal subset of `logits`.
pub fn sample_legal<R: Rng>(
    logits: &[f32],
    legal: &[usize],
    temperature: f32,
    rng: &mut R,
) -> Result<usize> {
    ensure!(!legal.is_empty(), "empty legal token set");
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "invalid sampling temperature"
    );
    let mut scaled = Vec::with_capacity(legal.len());
    for &id in legal {
        let value = *logits.get(id).ok_or_else(|| {
            anyhow::anyhow!("legal token {id} outside the logit row ({})", logits.len())
        })?;
        ensure!(!value.is_nan(), "NaN logit for legal token {id}");
        scaled.push(value as f64 / temperature as f64);
    }
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ensure!(max.is_finite(), "no finite logit among legal tokens");
    let weights = scaled.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    let mut draw = rng.gen::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        if draw < *w {
            return Ok(legal[i]);
        }
        draw -= w;
    }
    // Rounding left a remainder: take the last token with positive weight.
    Ok(legal[weights.iter().rposition(|w| *w > 0.0).unwrap_or(0)])
}

/// Sample `batch` rows under the grammar mask.
pub fn sample_batch<R: Rng>(
    model: &PolicyModel,
    vocab: &Vocab,
    context: &[f32],
    task: &str,
    batch: usize,
    temperature: f32,
    rng: &mut R,
) -> Result<Vec<SampledRow>> {
    ensure!(batch > 0, "empty sampling batch");
    ensure!(
        model.config().vocab_size == vocab.tokens.len(),
        "policy vocabulary size differs from the grammar"
    );
    let prefix = vocab.proposal_prefix(task)?;
    let start = vocab.prefix_state(task)?;
    let mut sequences = vec![prefix.clone(); batch];
    let mut states = vec![start; batch];
    let limit = MAX_PROGRAM_TOKENS.min(model.config().max_seq);
    for _ in prefix.len()..limit {
        let active = (0..batch)
            .filter(|&i| !states[i].done())
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        let rows = active
            .iter()
            .map(|&i| sequences[i].clone())
            .collect::<Vec<_>>();
        let logits = model.next_token_logits(context, &rows)?;
        let mut next = vec![PAD; batch];
        for (row, &i) in logits.iter().zip(&active) {
            let legal = states[i].legal(vocab);
            let token = sample_legal(row, &legal, temperature, rng)?;
            states[i].advance(vocab, token)?;
            next[i] = token;
        }
        for (sequence, token) in sequences.iter_mut().zip(next) {
            sequence.push(token);
        }
    }
    Ok(sequences
        .into_iter()
        .zip(states)
        .map(|(tokens, state)| {
            let program = state.done().then(|| {
                let end = tokens
                    .iter()
                    .position(|&t| t == EOS)
                    .expect("finished row has EOS");
                tokens[..=end].to_vec()
            });
            SampledRow { tokens, program }
        })
        .collect())
}

/// Decode a finished row into a genome that the pinned feat-math parses.
pub fn valid_genome(vocab: &Vocab, program: &[usize]) -> Option<String> {
    let genome = vocab.decode(program).ok()?;
    feat_math::GenomeIr::parse(&genome).ok()?;
    Some(genome)
}

/// jope-dime `push_pure`: false for a duplicate; the pool keeps only the first 64.
pub(crate) fn push_pure(
    pool: &mut Vec<SeedCandidate>,
    seen: &mut BTreeSet<String>,
    genome: String,
) -> bool {
    if !seen.insert(genome.clone()) {
        return false;
    }
    if pool.len() < POOL_SIZE {
        pool.push(SeedCandidate::new(genome, ORIGIN_NEURAL));
    }
    true
}

/// A program drawn uniformly over the legal tokens of the grammar (sampling mode, so
/// `generate=false` operators are excluded). `None` when the draw does not terminate within
/// the sequence limit, does not parse, or does not re-encode exactly.
pub fn grammar_uniform_program<R: Rng>(
    vocab: &Vocab,
    task: &str,
    rng: &mut R,
) -> Result<Option<String>> {
    let mut tokens = vocab.proposal_prefix(task)?;
    let mut state = vocab.prefix_state(task)?;
    while !state.done() && tokens.len() < MAX_PROGRAM_TOKENS {
        let legal = state.legal(vocab);
        ensure!(!legal.is_empty(), "grammar offered no legal token");
        let token = legal[rng.gen_range(0..legal.len())];
        state.advance(vocab, token)?;
        tokens.push(token);
    }
    if !state.done() {
        return Ok(None);
    }
    let Some(genome) = valid_genome(vocab, &tokens) else {
        return Ok(None);
    };
    Ok(match vocab.encode_for_task(&genome, task) {
        Ok(encoded) if encoded == tokens => Some(genome),
        _ => None,
    })
}

/// Append distinct grammar-uniform programs with `origin` until `pool` holds `target`.
/// Draw `offset` uses a fresh stream seeded with `seed + offset`.
pub(crate) fn fill_grammar_uniform(
    pool: &mut Vec<SeedCandidate>,
    seen: &mut BTreeSet<String>,
    vocab: &Vocab,
    task: &str,
    seed: u64,
    target: usize,
    origin: &str,
) -> Result<()> {
    for offset in 0..10_000u64 {
        if pool.len() >= target {
            return Ok(());
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(offset));
        if let Some(genome) = grammar_uniform_program(vocab, task, &mut rng)? {
            if seen.insert(genome.clone()) {
                pool.push(SeedCandidate::new(genome, origin));
            }
        }
    }
    if pool.len() >= target {
        return Ok(());
    }
    bail!("unable to fill {target} distinct {origin} programs")
}

/// jope-dime `fill_pure`, with the grammar-uniform generator.
pub(crate) fn fill_pure(
    pool: &mut Vec<SeedCandidate>,
    seen: &mut BTreeSet<String>,
    vocab: &Vocab,
    task: &str,
    seed: u64,
) -> Result<()> {
    fill_grammar_uniform(
        pool,
        seen,
        vocab,
        task,
        seed,
        POOL_SIZE,
        ORIGIN_NEURAL_FALLBACK,
    )
}

/// `count` distinct grammar-uniform programs, disjoint from `seen` (which they are added to).
pub fn uniform_programs(
    vocab: &Vocab,
    task: &str,
    seed: u64,
    count: usize,
    seen: &mut BTreeSet<String>,
) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(count);
    fill_grammar_uniform(&mut out, seen, vocab, task, seed, count, ORIGIN_RANDOM)?;
    Ok(out.into_iter().map(SeedCandidate::into_genome).collect())
}

/// Sixteen distinct exploration programs, disjoint from `seen` (which they are added to).
pub fn random_exploration(
    vocab: &Vocab,
    task: &str,
    seed: u64,
    seen: &mut BTreeSet<String>,
) -> Result<Vec<SeedCandidate>> {
    let mut out = Vec::with_capacity(EXPLORATION_SIZE);
    fill_grammar_uniform(
        &mut out,
        seen,
        vocab,
        task,
        seed,
        EXPLORATION_SIZE,
        ORIGIN_RANDOM,
    )?;
    Ok(out)
}

/// The pure pool and its counters (jope-dime `pure_pool_impl` return tuple). The candidates are
/// private: a pool is only built by sampling.
#[derive(Clone, Debug)]
pub struct PurePool {
    candidates: Vec<SeedCandidate>,
    pub attempted: usize,
    pub valid: usize,
    pub within_neural_duplicates: usize,
    pub unique_neural: usize,
    pub sampling_seed: u64,
    pub fallback_seed: u64,
    /// Tokens actually drawn by `sample_legal` (prefix and post-EOS PAD excluded); each was
    /// drawn from its checked legal set.
    pub sampled_tokens: usize,
}

impl PurePool {
    /// Neural samples first, then fallback programs.
    pub fn candidates(&self) -> &[SeedCandidate] {
        &self.candidates
    }
}

/// jope-dime `pure_pool_impl`: up to `rounds` batches, stopping once 64 distinct valid programs
/// exist, then filling the rest. Also returns the set of every program seen.
///
/// Samples are labelled `neural_policy`, so `model` must be the bound accepted policy (from
/// `PolicyModel::load_bound`); any other model is refused.
pub fn pure_pool(
    model: &PolicyModel,
    vocab: &Vocab,
    context: &[f32],
    task: &str,
    sampling_seed: u64,
    fallback_seed: u64,
    options: &SamplingOptions,
) -> Result<(PurePool, BTreeSet<String>)> {
    super::require_accepted_policy(model)?;
    pure_pool_checked(
        model,
        vocab,
        context,
        task,
        sampling_seed,
        fallback_seed,
        options,
    )
}

/// Test support (`testing`): [`pure_pool`] for untrained weights; refuses any bound
/// model.
#[cfg(any(test, feature = "testing"))]
pub fn pure_pool_unbound(
    model: &PolicyModel,
    vocab: &Vocab,
    context: &[f32],
    task: &str,
    sampling_seed: u64,
    fallback_seed: u64,
    options: &SamplingOptions,
) -> Result<(PurePool, BTreeSet<String>)> {
    ensure!(
        model.bound_identity().is_none(),
        "refused: the unbound test path only accepts untrained weights"
    );
    pure_pool_checked(
        model,
        vocab,
        context,
        task,
        sampling_seed,
        fallback_seed,
        options,
    )
}

/// [`pure_pool`] after the model identity check.
pub(crate) fn pure_pool_checked(
    model: &PolicyModel,
    vocab: &Vocab,
    context: &[f32],
    task: &str,
    sampling_seed: u64,
    fallback_seed: u64,
    options: &SamplingOptions,
) -> Result<(PurePool, BTreeSet<String>)> {
    ensure!(
        options.rounds > 0 && options.batch > 0,
        "invalid sampling options"
    );
    let prefix_len = vocab.proposal_prefix(task)?.len();
    let mut rng = ChaCha20Rng::seed_from_u64(sampling_seed);
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();
    let (mut attempted, mut valid, mut duplicate, mut sampled_tokens) = (0, 0, 0, 0);
    for _ in 0..options.rounds {
        let rows = sample_batch(
            model,
            vocab,
            context,
            task,
            options.batch,
            options.temperature,
            &mut rng,
        )?;
        attempted += rows.len();
        for row in rows {
            // Finished rows were sampled up to and including EOS; the PAD after it was not.
            let sampled_end = row.program.as_ref().map_or(row.tokens.len(), Vec::len);
            sampled_tokens += sampled_end - prefix_len;
            let Some(program) = row.program else {
                continue;
            };
            let Some(genome) = valid_genome(vocab, &program) else {
                continue;
            };
            valid += 1;
            if !push_pure(&mut candidates, &mut seen, genome) {
                duplicate += 1;
            }
        }
        if candidates.len() == POOL_SIZE {
            break;
        }
    }
    let unique_neural = candidates.len();
    fill_pure(&mut candidates, &mut seen, vocab, task, fallback_seed)?;
    Ok((
        PurePool {
            candidates,
            attempted,
            valid,
            within_neural_duplicates: duplicate,
            unique_neural,
            sampling_seed,
            fallback_seed,
            sampled_tokens,
        },
        seen,
    ))
}
