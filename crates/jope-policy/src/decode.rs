//! Batched grammar-masked decoding and the pure neural pool (jope-dime `proposals.rs:748-875`
//! and `proposal_population.rs:9-50`).
//!
//! Each round starts `batch` rows at `[BOS, featmath, task]`, and at every step samples the
//! next token of each unfinished row from `softmax(logits / temperature)` restricted to the
//! automaton's legal set, so every produced token is legal by construction. Finished rows are
//! padded with PAD (the only token jope-dime leaves legal for them) and are not recomputed.
//! Rows that have not reached EOS within the token budget (`min(128, max_seq)` tokens, prefix
//! included) are recorded as incomplete and never pooled. A finished row counts as
//! valid when it decodes and parses as a genome; the first 64 distinct valid programs form the
//! pure pool (origin `neural_policy`), and a short pool is filled with grammar-uniform programs
//! (origin `neural_rejection_fallback`).
//!
//! Sampling uses a seeded ChaCha20 stream, not torch's generator (see the module docs).
//!
//! # Terminal outcomes and bounds
//!
//! Every scheduled draw ends in exactly one [`DrawOutcome`] recorded in the pool with its
//! [`DrawId`], tokens and final automaton phase. A request either returns such a ledger or fails
//! as a whole with a typed [`DecodeFailure`] (invalid budget or temperature, no token budget,
//! NaN/+Inf legal logit, empty legal support, exhausted fallback): nothing is resampled,
//! repaired, or replaced by a fallback program after one of those. Work is bounded separately
//! and before it runs: at most [`MAX_ROUNDS`] rounds of [`MAX_BATCH`] rows (all charged to the
//! request before the round is sampled), at most `min(MAX_PROGRAM_TOKENS, max_seq)` tokens per
//! draw, and at most [`FALLBACK_ATTEMPTS`] grammar-uniform draws per fill. There is no greedy
//! mode: a zero temperature is invalid input.
use anyhow::{ensure, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::{ChaCha20Rng, ChaCha8Rng};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt;

use super::grammar::{Vocab, EOS, MAX_PROGRAM_TOKENS, PAD};
use super::model::PolicyModel;
use super::seeds::{SeedCandidate, ORIGIN_NEURAL, ORIGIN_NEURAL_FALLBACK, ORIGIN_RANDOM};
use super::sha256_hex;

/// jope-dime's pure pool protocol identifier.
pub const PURE_PROTOCOL: &str = "training_context_neural_pool_v1";
/// Number of pure (neural-arm) programs.
pub const POOL_SIZE: usize = 64;
/// Number of exploration programs appended by the seed writer.
pub const EXPLORATION_SIZE: usize = 16;
/// Most sampling rounds one request may schedule (jope-dime's plan).
pub const MAX_ROUNDS: usize = 8;
/// Most rows one sampling round may draw (jope-dime's plan).
pub const MAX_BATCH: usize = 128;
/// Grammar-uniform draws one fill may spend before it fails as exhausted.
pub const FALLBACK_ATTEMPTS: u64 = 10_000;

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
            rounds: MAX_ROUNDS,
            batch: MAX_BATCH,
            temperature: 1.0,
        }
    }
}

impl SamplingOptions {
    /// The bounded configuration every sampling entry accepts: `1..=MAX_ROUNDS` rounds of
    /// `1..=MAX_BATCH` rows at a finite, positive temperature.
    pub fn validate(&self) -> Result<()> {
        check_budget(self.rounds, self.batch)?;
        check_temperature(self.temperature)?;
        Ok(())
    }
}

/// One scheduled neural draw of a request: row `row` of sampling round `round` (a standalone
/// [`sample_batch`] is round 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DrawId {
    pub round: usize,
    pub row: usize,
}

/// How a scheduled neural draw ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawOutcome {
    /// First occurrence of a valid program; admitted to the pool with origin `neural_policy`.
    Completed,
    /// First occurrence of a valid program drawn after the pool already held [`POOL_SIZE`]
    /// programs in the same round; valid and unique, never pooled.
    Surplus,
    /// A valid program an earlier draw of the same request already produced.
    Duplicate,
    /// Reached EOS, but the tokens do not decode to a genome the pinned feat-math parses.
    ParserRejected,
    /// Did not reach EOS within the token budget.
    Incomplete,
}

/// The terminal record of one scheduled neural draw.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DrawRecord {
    pub draw: DrawId,
    pub outcome: DrawOutcome,
    /// The draw's tokens: the proposal prefix and every sampled token (through EOS when it
    /// finished); never the PAD written after EOS.
    pub tokens: Vec<usize>,
    /// Automaton phase when the draw ended (`done` after EOS).
    pub grammar_phase: &'static str,
    /// sha256 of the genome text, for valid draws (completed, surplus, duplicate).
    pub program_sha256: Option<String>,
}

/// Why a sampling request ended without a result.
#[derive(Clone, Debug, PartialEq)]
pub enum FailureKind {
    /// Rounds or batch outside `1..=MAX_ROUNDS` / `1..=MAX_BATCH`.
    InvalidBudget { rounds: usize, batch: usize },
    /// Temperature that is not finite and positive.
    InvalidTemperature(f32),
    /// `min(MAX_PROGRAM_TOKENS, max_seq)` leaves no token to sample after the prefix.
    NoTokenBudget { prefix: usize, budget: usize },
    /// A legal token's logit is NaN or +Inf.
    NonfiniteLogit { token: usize, value: f32 },
    /// No legal token, or every legal token's logit is -Inf.
    EmptySupport,
    /// A legal token has no logit (the logit row is shorter than the vocabulary).
    MissingLogit { token: usize, row: usize },
    /// A grammar-uniform fill spent [`FALLBACK_ATTEMPTS`] draws short of its target.
    FallbackExhausted {
        origin: String,
        target: usize,
        filled: usize,
    },
}

impl fmt::Display for FailureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBudget { rounds, batch } => write!(
                f,
                "invalid sampling options: {rounds} rounds of {batch} rows (bounded to 1..={MAX_ROUNDS} rounds of 1..={MAX_BATCH} rows)"
            ),
            Self::InvalidTemperature(t) => write!(f, "invalid sampling temperature {t}"),
            Self::NoTokenBudget { prefix, budget } => write!(
                f,
                "token budget {budget} leaves no token after the {prefix}-token proposal prefix"
            ),
            Self::NonfiniteLogit { token, value } => {
                write!(f, "{value} logit for legal token {token}")
            }
            Self::EmptySupport => {
                write!(f, "empty legal token set (no legal token has a finite logit)")
            }
            Self::MissingLogit { token, row } => {
                write!(f, "legal token {token} outside the logit row ({row})")
            }
            Self::FallbackExhausted {
                origin,
                target,
                filled,
            } => write!(
                f,
                "unable to fill {target} distinct {origin} programs ({filled} after {FALLBACK_ATTEMPTS} draws)"
            ),
        }
    }
}

/// A typed terminal failure of a sampling request, carried inside the returned
/// `anyhow::Error`; recover it with `err.downcast_ref::<DecodeFailure>()`.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeFailure {
    pub kind: FailureKind,
    /// The neural draw being sampled; `None` for request-level refusals and fallback fills.
    pub draw: Option<DrawId>,
    /// Seed of the random stream the failing draw belonged to, when the entry point knows it.
    pub seed: Option<u64>,
    /// The failing draw's tokens before the failing step (proposal prefix included).
    pub tokens: Vec<usize>,
    /// Automaton phase at the failing step.
    pub grammar_phase: Option<&'static str>,
}

impl From<FailureKind> for DecodeFailure {
    fn from(kind: FailureKind) -> Self {
        Self {
            kind,
            draw: None,
            seed: None,
            tokens: Vec::new(),
            grammar_phase: None,
        }
    }
}

impl fmt::Display for DecodeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        if let Some(DrawId { round, row }) = self.draw {
            write!(f, " at round {round}, row {row}")?;
        }
        if let Some(phase) = self.grammar_phase {
            write!(
                f,
                " after {} tokens in grammar phase {phase}",
                self.tokens.len()
            )?;
        }
        if let Some(seed) = self.seed {
            write!(f, " (stream seed {seed})")?;
        }
        Ok(())
    }
}

impl std::error::Error for DecodeFailure {}

fn check_budget(rounds: usize, batch: usize) -> Result<(), DecodeFailure> {
    if (1..=MAX_ROUNDS).contains(&rounds) && (1..=MAX_BATCH).contains(&batch) {
        Ok(())
    } else {
        Err(FailureKind::InvalidBudget { rounds, batch }.into())
    }
}

fn check_temperature(temperature: f32) -> Result<(), DecodeFailure> {
    if temperature.is_finite() && temperature > 0.0 {
        Ok(())
    } else {
        Err(FailureKind::InvalidTemperature(temperature).into())
    }
}

/// Attach the stream seed to a typed failure; other errors pass through unchanged.
fn with_seed(error: anyhow::Error, seed: u64) -> anyhow::Error {
    match error.downcast::<DecodeFailure>() {
        Ok(mut failure) => {
            failure.seed = Some(seed);
            failure.into()
        }
        Err(error) => error,
    }
}

/// Tokens (prefix and EOS included) one draw may reach with `model`.
fn token_budget(model: &PolicyModel) -> usize {
    MAX_PROGRAM_TOKENS.min(model.config().max_seq)
}

/// One sampled row: the tokens (PAD-padded to the batch length) and, when it reached EOS, the
/// program tokens `BOS..=EOS`.
#[derive(Clone, Debug)]
pub struct SampledRow {
    pub tokens: Vec<usize>,
    pub program: Option<Vec<usize>>,
    /// Automaton phase when the row stopped (`done` after EOS).
    pub grammar_phase: &'static str,
}

/// Draw one token from the legal subset of `logits`. Failures are typed [`DecodeFailure`]s
/// without a draw identity.
pub fn sample_legal<R: Rng>(
    logits: &[f32],
    legal: &[usize],
    temperature: f32,
    rng: &mut R,
) -> Result<usize> {
    Ok(draw_legal(logits, legal, temperature, rng)?)
}

/// [`sample_legal`] with the failure kind; consumes one random draw only when it returns a
/// token. NaN and +Inf legal logits fail; -Inf legal tokens have zero probability, like the
/// reference `masked_fill(-inf)` softmax; logits of illegal tokens are never read.
fn draw_legal<R: Rng>(
    logits: &[f32],
    legal: &[usize],
    temperature: f32,
    rng: &mut R,
) -> Result<usize, DecodeFailure> {
    if legal.is_empty() {
        return Err(FailureKind::EmptySupport.into());
    }
    check_temperature(temperature)?;
    let mut scaled = Vec::with_capacity(legal.len());
    for &id in legal {
        let value = *logits.get(id).ok_or(FailureKind::MissingLogit {
            token: id,
            row: logits.len(),
        })?;
        if value.is_nan() || value == f32::INFINITY {
            return Err(FailureKind::NonfiniteLogit { token: id, value }.into());
        }
        scaled.push(value as f64 / temperature as f64);
    }
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err(FailureKind::EmptySupport.into());
    }
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

/// Sample `batch` (`1..=MAX_BATCH`) rows under the grammar mask, as sampling round 0.
pub fn sample_batch<R: Rng>(
    model: &PolicyModel,
    vocab: &Vocab,
    context: &[f32],
    task: &str,
    batch: usize,
    temperature: f32,
    rng: &mut R,
) -> Result<Vec<SampledRow>> {
    sample_batch_impl(
        (model, context),
        vocab,
        task,
        (0, batch),
        temperature,
        rng,
        false,
    )
}

/// Cached production decoding of sampling round `round`. This returns raw sampled rows, never
/// an admitted origin or seed binding. `sample_batch` retains the full-prefix reference path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_batch_cached<R: Rng>(
    model: &PolicyModel,
    vocab: &Vocab,
    context: &[f32],
    task: &str,
    round: usize,
    batch: usize,
    temperature: f32,
    rng: &mut R,
) -> Result<Vec<SampledRow>> {
    sample_batch_impl(
        (model, context),
        vocab,
        task,
        (round, batch),
        temperature,
        rng,
        true,
    )
}

/// `shape` is `(round, batch)`: the round only names the draws in a failure.
fn sample_batch_impl<R: Rng>(
    policy: (&PolicyModel, &[f32]),
    vocab: &Vocab,
    task: &str,
    shape: (usize, usize),
    temperature: f32,
    rng: &mut R,
    cached: bool,
) -> Result<Vec<SampledRow>> {
    let (model, context) = policy;
    let (round, batch) = shape;
    // Everything that can refuse the request does so before the first forward pass or draw.
    check_budget(1, batch)?;
    check_temperature(temperature)?;
    ensure!(
        model.config().vocab_size == vocab.tokens.len(),
        "policy vocabulary size differs from the grammar"
    );
    let prefix = vocab.proposal_prefix(task)?;
    let start = vocab.prefix_state(task)?;
    let limit = token_budget(model);
    if limit <= prefix.len() {
        return Err(DecodeFailure::from(FailureKind::NoTokenBudget {
            prefix: prefix.len(),
            budget: limit,
        })
        .into());
    }
    let mut sequences = vec![prefix.clone(); batch];
    let mut states = vec![start; batch];
    let mut cache: Option<super::model::cached::DecodeCache<'_>> = None;
    for _ in prefix.len()..limit {
        let active = (0..batch)
            .filter(|&i| !states[i].done())
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        let logits = if cached {
            if let Some(cache) = cache.as_mut() {
                let tokens = active
                    .iter()
                    .map(|&i| *sequences[i].last().expect("sampling prefix is nonempty"))
                    .collect::<Vec<_>>();
                cache.advance(&active, &tokens)?
            } else {
                let rows = active
                    .iter()
                    .map(|&i| sequences[i].clone())
                    .collect::<Vec<_>>();
                let (created, logits) =
                    super::model::cached::DecodeCache::new(model, context, &rows)?;
                cache = Some(created);
                logits
            }
        } else {
            let rows = active
                .iter()
                .map(|&i| sequences[i].clone())
                .collect::<Vec<_>>();
            model.next_token_logits(context, &rows)?
        };
        ensure!(
            logits.len() == active.len(),
            "policy returned {} logit rows for {} active rows",
            logits.len(),
            active.len()
        );
        let mut next = vec![PAD; batch];
        for (row, &i) in logits.iter().zip(&active) {
            let legal = states[i].legal(vocab);
            let token =
                draw_legal(row, &legal, temperature, rng).map_err(|failure| DecodeFailure {
                    draw: Some(DrawId { round, row: i }),
                    tokens: sequences[i].clone(),
                    grammar_phase: Some(states[i].phase_name()),
                    ..failure
                })?;
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
            SampledRow {
                tokens,
                program,
                grammar_phase: state.phase_name(),
            }
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
        if legal.is_empty() {
            return Err(DecodeFailure {
                grammar_phase: Some(state.phase_name()),
                tokens,
                ..DecodeFailure::from(FailureKind::EmptySupport)
            }
            .into());
        }
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

/// Append distinct grammar-uniform programs with `origin` until `pool` holds `target`, spending
/// at most [`FALLBACK_ATTEMPTS`] draws; returns the draws spent. Draw `offset` uses a fresh
/// stream seeded with `seed + offset`.
pub(crate) fn fill_grammar_uniform(
    pool: &mut Vec<SeedCandidate>,
    seen: &mut BTreeSet<String>,
    vocab: &Vocab,
    task: &str,
    seed: u64,
    target: usize,
    origin: &str,
) -> Result<u64> {
    for offset in 0..FALLBACK_ATTEMPTS {
        if pool.len() >= target {
            return Ok(offset);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(offset));
        let program = grammar_uniform_program(vocab, task, &mut rng)
            .map_err(|error| with_seed(error, seed.wrapping_add(offset)))?;
        if let Some(genome) = program {
            if seen.insert(genome.clone()) {
                pool.push(SeedCandidate::new(genome, origin));
            }
        }
    }
    if pool.len() >= target {
        return Ok(FALLBACK_ATTEMPTS);
    }
    Err(DecodeFailure {
        seed: Some(seed),
        ..DecodeFailure::from(FailureKind::FallbackExhausted {
            origin: origin.into(),
            target,
            filled: pool.len(),
        })
    }
    .into())
}

/// jope-dime `fill_pure`, with the grammar-uniform generator; returns the draws spent.
pub(crate) fn fill_pure(
    pool: &mut Vec<SeedCandidate>,
    seen: &mut BTreeSet<String>,
    vocab: &Vocab,
    task: &str,
    seed: u64,
) -> Result<u64> {
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
    /// The validated sampling budget and temperature every draw used.
    pub options: SamplingOptions,
    /// Most tokens (prefix and EOS included) one draw could reach:
    /// `min(MAX_PROGRAM_TOKENS, max_seq)`.
    pub token_budget: usize,
    /// Grammar-uniform draws spent filling the pool (at most [`FALLBACK_ATTEMPTS`]); the
    /// `POOL_SIZE - unique_neural` programs they admitted carry origin
    /// `neural_rejection_fallback`.
    pub fallback_attempts: u64,
    /// One terminal record per scheduled draw, in draw order (`attempted` records).
    draws: Vec<DrawRecord>,
}

impl PurePool {
    /// Neural samples first, then fallback programs.
    pub fn candidates(&self) -> &[SeedCandidate] {
        &self.candidates
    }

    /// The terminal record of every scheduled neural draw, in draw order.
    pub fn draws(&self) -> &[DrawRecord] {
        &self.draws
    }

    /// Number of draws that ended with `outcome`.
    pub fn count(&self, outcome: DrawOutcome) -> usize {
        self.draws.iter().filter(|d| d.outcome == outcome).count()
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
    options.validate()?;
    let prefix_len = vocab.proposal_prefix(task)?.len();
    let mut rng = ChaCha20Rng::seed_from_u64(sampling_seed);
    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();
    let mut draws = Vec::new();
    let (mut attempted, mut valid, mut duplicate, mut sampled_tokens) = (0, 0, 0, 0);
    for round in 0..options.rounds {
        // The whole round is charged to the request before any of its rows is sampled.
        attempted += options.batch;
        let rows = sample_batch_cached(
            model,
            vocab,
            context,
            task,
            round,
            options.batch,
            options.temperature,
            &mut rng,
        )
        .map_err(|error| with_seed(error, sampling_seed))?;
        ensure!(rows.len() == options.batch, "sampling round lost rows");
        for (row, sampled) in rows.into_iter().enumerate() {
            // Finished rows were sampled up to and including EOS; the PAD after it was not.
            let sampled_end = sampled
                .program
                .as_ref()
                .map_or(sampled.tokens.len(), Vec::len);
            sampled_tokens += sampled_end - prefix_len;
            let genome = sampled
                .program
                .as_ref()
                .map(|program| valid_genome(vocab, program));
            let (outcome, program_sha256) = match genome {
                None => (DrawOutcome::Incomplete, None),
                Some(None) => (DrawOutcome::ParserRejected, None),
                Some(Some(genome)) => {
                    valid += 1;
                    let sha256 = sha256_hex(genome.as_bytes());
                    let pooled = candidates.len();
                    let outcome = if !push_pure(&mut candidates, &mut seen, genome) {
                        duplicate += 1;
                        DrawOutcome::Duplicate
                    } else if candidates.len() > pooled {
                        DrawOutcome::Completed
                    } else {
                        DrawOutcome::Surplus
                    };
                    (outcome, Some(sha256))
                }
            };
            let mut tokens = sampled.tokens;
            tokens.truncate(sampled_end);
            draws.push(DrawRecord {
                draw: DrawId { round, row },
                outcome,
                tokens,
                grammar_phase: sampled.grammar_phase,
                program_sha256,
            });
        }
        if candidates.len() == POOL_SIZE {
            break;
        }
    }
    let unique_neural = candidates.len();
    let fallback_attempts = fill_pure(&mut candidates, &mut seen, vocab, task, fallback_seed)?;
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
            options: options.clone(),
            token_budget: token_budget(model),
            fallback_attempts,
            draws,
        },
        seen,
    ))
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::config::PolicyConfig;
    use rand::RngCore;

    #[test]
    fn cached_sampling_preserves_tokens_grammar_finished_rows_and_rng_tail() {
        let vocab = Vocab::from_json(include_bytes!(
            "../../jope-engine/assets/grammar/featmath.json"
        ))
        .unwrap();
        let mut config = PolicyConfig::regression(vocab.tokens.len());
        config.width = 16;
        config.heads = 4;
        config.ffn = 32;
        config.layers = 1;
        config.dropout = 0.0;
        let model = PolicyModel::random_weights(config, 3).unwrap();
        assert!(model.bound_identity().is_none());
        let context = (0..2048)
            .map(|i| (i % 13) as f32 / 13.0 - 0.4)
            .collect::<Vec<_>>();
        let mut full_rng = ChaCha20Rng::seed_from_u64(7);
        let mut cached_rng = full_rng.clone();
        let full = sample_batch(
            &model,
            &vocab,
            &context,
            "regression",
            24,
            25.0,
            &mut full_rng,
        )
        .unwrap();
        let cached = sample_batch_cached(
            &model,
            &vocab,
            &context,
            "regression",
            0,
            24,
            25.0,
            &mut cached_rng,
        )
        .unwrap();
        assert_eq!(full.len(), cached.len());
        let mut finished_early = false;
        for (a, b) in full.iter().zip(&cached) {
            assert_eq!(a.tokens, b.tokens);
            assert_eq!(a.program, b.program);
            assert_eq!(a.grammar_phase, b.grammar_phase);
            let mut state = vocab.prefix_state("regression").unwrap();
            for &token in &b.tokens[3..] {
                if state.done() {
                    finished_early = true;
                    assert_eq!(token, PAD);
                } else {
                    assert!(state.legal(&vocab).contains(&token));
                    state.advance(&vocab, token).unwrap();
                }
            }
        }
        assert!(
            finished_early,
            "fixture must exercise removal of completed rows"
        );
        for _ in 0..16 {
            assert_eq!(full_rng.next_u64(), cached_rng.next_u64());
        }
    }
}
