//! Proposal ledger: the identity and the complete denominators of one policy proposal (W4-T07).
//!
//! [`seeds::propose`](super::seeds::propose) (and its challenger and test variants) records,
//! next to the 80-candidate seed population, one `jope.proposal-ledger/v1`:
//!
//! * every scheduled neural draw with its terminal outcome, tokens and grammar phase;
//! * every grammar-uniform draw of the fallback fill and of the exploration fill, each with its
//!   own stream seed and terminal outcome;
//! * one record per handed-off candidate: slot, origin, the draw it came from, the drawn tokens,
//!   the program text and its sha256, the parser's canonical recipe identity, the explicit
//!   adaptation edges applied at the proposal stage, and (for the 64 ranked pool slots) the
//!   predicted scores of exactly that program;
//! * the reconciled denominators: scheduled draws, complete sequences, valid, unique and
//!   canonical-unique programs, refusals, fallback and exploration draws, rank survivors and
//!   handed-off candidates.
//!
//! Draws that did not produce a handed-off candidate stay in the ledger as shadow records for
//! later ranking studies; [`ProposalLedger::shadow`] exposes them without fitting authority, and
//! nothing in this crate turns one into a seed.
//!
//! The ledger joins the W5-T01 GP lineage envelope rather than replacing it: a GP
//! `seed_population_slot` source names a slot of the seed file whose population sha256 the
//! producer artifact binds, and [`ProposalLedger::slot_record_sha256`] is the identity of that
//! slot's proposal record (original tokens and program, origin, scores, parser result and
//! proposal-stage adaptations). The GP-side vocabulary projection remains the W5 envelope's
//! single adaptation edge.
//!
//! [`ProposalLedger::reconcile`] re-derives every count and link from the records alone, so a
//! ledger read back from an artifact is checked, not trusted: an unterminated or missing draw, a
//! dropped failure, a fallback relabelled as neural, a duplicate or doubly used draw, a second
//! adaptation, or a score attached to another program is refused.
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::decode::{
    DrawOutcome, PurePool, UniformDrawRecord, UniformOutcome, EXPLORATION_SIZE, FALLBACK_ATTEMPTS,
    MAX_BATCH, MAX_ROUNDS, POOL_SIZE,
};
use super::grammar::{Vocab, EOS, MAX_PROGRAM_TOKENS};
use super::ranking::Ranking;
use super::seeds::{
    logprob_from_scores, SeedCandidate, ORIGIN_NEURAL, ORIGIN_NEURAL_FALLBACK, ORIGIN_RANDOM,
    SEED_COUNT,
};
use super::sha256_hex;

mod builder;
pub(crate) use builder::{build, ProposalRun};
mod decision_links;
pub use decision_links::{ProposalDecisionLinks, ProposalDrawLink};

/// Schema identifier for a reconciled proposal ledger.
pub const LEDGER_SCHEMA: &str = "jope.proposal-ledger/v1";
/// Schema identifier for decision links attached to an existing proposal ledger.
pub const DECISION_LINKS_SCHEMA: &str = "jope.proposal-decision-links/v1";

// I02's codec is shared as source until the integrator serializes the Cargo dependency hook.
// Extract I02's jope_records-importing doctests only in their owning crate. Runtime
// codec checks remain compiled here; I05's nonzero integration fixtures exercise this adapter.
#[cfg(not(doctest))]
// The adopted I02 Measured* enum names remain unchanged in this read-only source import.
#[allow(dead_code, clippy::enum_variant_names)]
#[path = "../../jope-records/src/wire_contract.rs"]
mod event_wire;

/// Adaptation edge: the drawn tokens decoded to the program text (the executable original).
pub const EDGE_DECODE: &str = "decode_tokens";
/// Adaptation edge: the program text re-encoded to the token sequence the reward heads read.
pub const EDGE_SCORE_ENCODING: &str = "score_encoding";
/// Parser whose canonical form identifies a recipe.
pub const PARSER: &str = "feat_math::GenomeIr::parse/canonical";

/// The sampling request the draws belong to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingRecord {
    pub sampling_seed: u64,
    pub fallback_seed: u64,
    pub exploration_seed: u64,
    pub rounds_scheduled: usize,
    /// Rounds actually sampled; fewer than scheduled only when the pool filled.
    pub rounds_run: usize,
    pub batch: usize,
    pub temperature: f32,
    pub token_budget: usize,
}

/// One scheduled neural draw and how it ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeuralDraw {
    pub round: usize,
    pub row: usize,
    pub outcome: DrawOutcome,
    /// Proposal prefix and every sampled token (through EOS when it finished).
    pub tokens: Vec<usize>,
    pub grammar_phase: String,
    pub program_sha256: Option<String>,
}

/// One grammar-uniform draw (fallback or exploration fill) and how it ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UniformDraw {
    /// The draw's own ChaCha8 stream seed (fill seed + draw offset).
    pub seed: u64,
    pub outcome: UniformOutcome,
    pub tokens: Vec<usize>,
    pub program_sha256: Option<String>,
}

/// Where a handed-off candidate came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SlotSource {
    NeuralDraw { round: usize, row: usize },
    FallbackDraw { seed: u64 },
    ExplorationDraw { seed: u64 },
}

/// One explicit proposal-stage transformation of a candidate's subject.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptationEdge {
    pub kind: String,
    pub input_sha256: String,
    pub output_sha256: String,
    /// For [`EDGE_SCORE_ENCODING`]: the scored token sequence differs from the drawn one.
    pub changed: bool,
}

/// The reward-head prediction attached to a ranked pool slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotPrediction {
    /// Position in the ranking (best first); equals the slot.
    pub rank: usize,
    pub predicted_quality: f64,
    pub reward_heads: [f64; 4],
    pub predicted_runtime_seconds: f64,
    /// Normalized critic rank weight of this slot in the seed file.
    pub rank_log_weight: f64,
    /// sha256 of the program the scores were computed for: the score subject.
    pub scored_program_sha256: String,
    pub scored_token_count: usize,
}

/// One handed-off candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotRecord {
    pub slot: usize,
    pub origin: String,
    pub source: SlotSource,
    /// The drawn tokens (proposal prefix through EOS): the original proposal.
    pub tokens: Vec<usize>,
    pub program: String,
    pub program_sha256: String,
    /// Parser result: sha256 of the parsed recipe's canonical form ([`PARSER`]).
    pub canonical_recipe_sha256: String,
    pub adaptations: Vec<AdaptationEdge>,
    /// Present for the 64 ranked pool slots, absent for exploration slots.
    pub prediction: Option<SlotPrediction>,
}

/// Reconciled counts of one proposal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Denominators {
    pub scheduled_neural_draws: usize,
    pub completed: usize,
    pub surplus: usize,
    pub duplicate: usize,
    pub parser_rejected: usize,
    pub incomplete: usize,
    /// Draws that reached EOS.
    pub complete_sequences: usize,
    /// Complete sequences that parse (completed + surplus + duplicate).
    pub valid_programs: usize,
    /// First occurrences among valid programs (completed + surplus).
    pub unique_programs: usize,
    pub fallback_draws: usize,
    pub fallback_admitted: usize,
    pub exploration_draws: usize,
    pub exploration_admitted: usize,
    /// Pool programs scored by the reward heads.
    pub scored: usize,
    pub rank_survivors: usize,
    pub handed_off: usize,
    /// Distinct canonical recipes among the handed-off candidates.
    pub unique_canonical_handed_off: usize,
    /// Draws kept as shadow records (never handed off).
    pub shadow: usize,
}

/// A draw that did not become a handed-off candidate: kept for studies, never fitted.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ShadowRecord<'a> {
    pub stage: &'static str,
    pub outcome: String,
    pub tokens: &'a [usize],
    pub program_sha256: Option<&'a str>,
    /// Always false: a shadow record has no fitting authority.
    pub fitting_authority: bool,
}

/// One draw's terminal outcome as a candidate ledger downstream records it (W4-T03).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrawOutcomeRecord {
    /// The draw: a scheduled neural draw, or a fallback or exploration stream.
    pub draw: SlotSource,
    /// Its terminal outcome, as the ledger spells it (`completed`, `parser_rejected`, ...).
    pub outcome: String,
    pub tokens_sha256: String,
    pub program_sha256: Option<String>,
    /// The seed slot the draw was handed off to, and that slot's proposal record identity.
    pub slot: Option<usize>,
    pub slot_record_sha256: Option<String>,
}

/// Every draw of one proposal with its terminal outcome, in ledger order (neural, fallback,
/// exploration), and the reconciled denominators.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrawOutcomes {
    pub ledger_sha256: String,
    pub draws: Vec<DrawOutcomeRecord>,
    pub denominators: Denominators,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalLedger {
    pub schema: String,
    /// The task whose proposal prefix every draw starts from.
    pub task: String,
    pub checkpoint_sha256: String,
    pub context_sha256: String,
    pub vocabulary_sha256: String,
    pub sampling: SamplingRecord,
    pub neural_draws: Vec<NeuralDraw>,
    pub fallback_draws: Vec<UniformDraw>,
    pub exploration_draws: Vec<UniformDraw>,
    pub candidates: Vec<SlotRecord>,
    pub denominators: Denominators,
}

fn tokens_sha256(tokens: &[usize]) -> Result<String> {
    Ok(sha256_hex(&serde_json::to_vec(tokens)?))
}
fn canonical_sha256(program: &str) -> Result<String> {
    let recipe = feat_math::GenomeIr::parse(program)?;
    Ok(sha256_hex(recipe.canonical().as_bytes()))
}

impl ProposalLedger {
    /// The denominators implied by the records.
    fn count(&self) -> Denominators {
        let neural = |o: DrawOutcome| self.neural_draws.iter().filter(|d| d.outcome == o).count();
        let admitted = |draws: &[UniformDraw]| {
            draws
                .iter()
                .filter(|d| d.outcome == UniformOutcome::Admitted)
                .count()
        };
        let (completed, surplus, duplicate) = (
            neural(DrawOutcome::Completed),
            neural(DrawOutcome::Surplus),
            neural(DrawOutcome::Duplicate),
        );
        let incomplete = neural(DrawOutcome::Incomplete);
        let scored = self
            .candidates
            .iter()
            .filter(|c| c.prediction.is_some())
            .count();
        let (fallback_admitted, exploration_admitted) = (
            admitted(&self.fallback_draws),
            admitted(&self.exploration_draws),
        );
        let draws =
            self.neural_draws.len() + self.fallback_draws.len() + self.exploration_draws.len();
        Denominators {
            scheduled_neural_draws: self.neural_draws.len(),
            completed,
            surplus,
            duplicate,
            parser_rejected: neural(DrawOutcome::ParserRejected),
            incomplete,
            complete_sequences: self.neural_draws.len() - incomplete,
            valid_programs: completed + surplus + duplicate,
            unique_programs: completed + surplus,
            fallback_draws: self.fallback_draws.len(),
            fallback_admitted,
            exploration_draws: self.exploration_draws.len(),
            exploration_admitted,
            scored,
            rank_survivors: scored,
            handed_off: self.candidates.len(),
            unique_canonical_handed_off: self
                .candidates
                .iter()
                .map(|c| &c.canonical_recipe_sha256)
                .collect::<BTreeSet<_>>()
                .len(),
            shadow: draws.saturating_sub(completed + fallback_admitted + exploration_admitted),
        }
    }
}

mod identity;
mod outcomes;
mod reconcile;
mod vocabulary;
