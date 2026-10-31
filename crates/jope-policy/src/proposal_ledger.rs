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

/// A companion references the existing ledger, including every shadow/refused draw. It adds
/// downstream decision joins without changing the bytes or denominators of proposal-ledger/v1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalDecisionLinks {
    pub schema: String,
    pub ledger_sha256: String,
    pub denominators: Denominators,
    pub draws: Vec<ProposalDrawLink>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalDrawLink {
    pub draw_id: String,
    pub draw_key: String,
    #[serde(rename = "draw_disposition")]
    pub outcome: String,
    pub tokens_sha256: String,
    pub program_sha256: Option<String>,
    pub slot_record_sha256: Option<String>,
    pub decision_ids: Vec<String>,
    /// No delivery/decision is not permission to fit a shadow proposal.
    pub missing_decision_reason: Option<String>,
}

impl ProposalDecisionLinks {
    /// Serialize this companion using the canonical JSON codec.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        Ok(
            event_wire::CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(self)?)?
                .as_bytes()
                .to_vec(),
        )
    }

    /// Verify that these decision joins exactly match the companion derived from `ledger`.
    pub fn verify(&self, ledger: &ProposalLedger) -> Result<()> {
        let mut joins = BTreeMap::new();
        for draw in &self.draws {
            ensure!(
                joins
                    .insert(draw.draw_id.clone(), draw.decision_ids.clone())
                    .is_none(),
                "duplicate proposal draw link"
            );
        }
        ensure!(
            *self == ledger.decision_links(&joins)?,
            "altered proposal decision companion"
        );
        Ok(())
    }
}
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
fn uniform_draws(records: &[UniformDrawRecord]) -> Vec<UniformDraw> {
    records
        .iter()
        .map(|r| UniformDraw {
            seed: r.seed,
            outcome: r.outcome,
            tokens: r.tokens.clone(),
            program_sha256: r.program_sha256.clone(),
        })
        .collect()
}

/// Everything [`build`] needs from one proposal run.
pub(crate) struct ProposalRun<'a> {
    pub task: &'a str,
    pub checkpoint_sha256: &'a str,
    pub context_sha256: &'a str,
    pub vocabulary_sha256: &'a str,
    pub pool: &'a PurePool,
    pub exploration_seed: u64,
    pub exploration_draws: &'a [UniformDrawRecord],
    pub ranking: &'a Ranking,
    pub population: &'a [SeedCandidate],
    pub vocab: &'a Vocab,
}

/// Record a proposal run; the result is reconciled against the vocabulary before it is returned.
pub(crate) fn build(run: &ProposalRun<'_>) -> Result<ProposalLedger> {
    let pool = run.pool;
    let batch = pool.options.batch;
    ensure!(
        batch > 0 && pool.attempted.is_multiple_of(batch),
        "pool attempts are not whole rounds"
    );
    let neural_draws = pool
        .draws()
        .iter()
        .map(|d| NeuralDraw {
            round: d.draw.round,
            row: d.draw.row,
            outcome: d.outcome,
            tokens: d.tokens.clone(),
            grammar_phase: d.grammar_phase.into(),
            program_sha256: d.program_sha256.clone(),
        })
        .collect::<Vec<_>>();
    let fallback_draws = uniform_draws(pool.fallback_draws());
    let exploration_draws = uniform_draws(run.exploration_draws);
    let selected = run.ranking.selected();
    ensure!(
        run.population.len() == SEED_COUNT && selected.len() == POOL_SIZE,
        "proposal population or ranking size changed"
    );
    let mut candidates = Vec::with_capacity(SEED_COUNT);
    for (slot, seed) in run.population.iter().enumerate() {
        let program = seed.genome().to_owned();
        let program_sha256 = sha256_hex(program.as_bytes());
        let (source, tokens) = match seed.origin() {
            ORIGIN_NEURAL => {
                let d = neural_draws
                    .iter()
                    .find(|d| {
                        d.outcome == DrawOutcome::Completed
                            && d.program_sha256.as_deref() == Some(program_sha256.as_str())
                    })
                    .context("neural candidate without its completed draw")?;
                (
                    SlotSource::NeuralDraw {
                        round: d.round,
                        row: d.row,
                    },
                    d.tokens.clone(),
                )
            }
            origin => {
                let (draws, fallback) = match origin {
                    ORIGIN_NEURAL_FALLBACK => (&fallback_draws, true),
                    ORIGIN_RANDOM => (&exploration_draws, false),
                    _ => bail!("unknown proposal origin {origin}"),
                };
                let d = draws
                    .iter()
                    .find(|d| {
                        d.outcome == UniformOutcome::Admitted
                            && d.program_sha256.as_deref() == Some(program_sha256.as_str())
                    })
                    .context("grammar-uniform candidate without its admitted draw")?;
                let source = if fallback {
                    SlotSource::FallbackDraw { seed: d.seed }
                } else {
                    SlotSource::ExplorationDraw { seed: d.seed }
                };
                (source, d.tokens.clone())
            }
        };
        let mut adaptations = vec![AdaptationEdge {
            kind: EDGE_DECODE.into(),
            input_sha256: tokens_sha256(&tokens)?,
            output_sha256: program_sha256.clone(),
            changed: false,
        }];
        let prediction = if slot < POOL_SIZE {
            let p = &selected[slot];
            // The score subject is this very program; the ranking re-encodes the text.
            ensure!(
                p.program == program && p.program_sha256 == program_sha256,
                "ranked prediction belongs to another program"
            );
            let scored = run.vocab.encode(&program)?;
            ensure!(
                scored.len() == p.token_count,
                "scored token count differs from the scoring encoding"
            );
            adaptations.push(AdaptationEdge {
                kind: EDGE_SCORE_ENCODING.into(),
                input_sha256: program_sha256.clone(),
                output_sha256: tokens_sha256(&scored)?,
                changed: scored != tokens,
            });
            Some(SlotPrediction {
                rank: slot,
                predicted_quality: p.predicted_quality,
                reward_heads: p.reward_heads,
                predicted_runtime_seconds: p.predicted_runtime_seconds,
                rank_log_weight: seed
                    .rank_log_weight()
                    .context("ranked pool slot without its rank weight")?,
                scored_program_sha256: p.program_sha256.clone(),
                scored_token_count: p.token_count,
            })
        } else {
            None
        };
        candidates.push(SlotRecord {
            slot,
            origin: seed.origin().into(),
            source,
            tokens,
            canonical_recipe_sha256: canonical_sha256(&program)?,
            program,
            program_sha256,
            adaptations,
            prediction,
        });
    }
    let mut ledger = ProposalLedger {
        schema: LEDGER_SCHEMA.into(),
        task: run.task.into(),
        checkpoint_sha256: run.checkpoint_sha256.into(),
        context_sha256: run.context_sha256.into(),
        vocabulary_sha256: run.vocabulary_sha256.into(),
        sampling: SamplingRecord {
            sampling_seed: pool.sampling_seed,
            fallback_seed: pool.fallback_seed,
            exploration_seed: run.exploration_seed,
            rounds_scheduled: pool.options.rounds,
            rounds_run: pool.attempted / batch,
            batch,
            temperature: pool.options.temperature,
            token_budget: pool.token_budget,
        },
        neural_draws,
        fallback_draws,
        exploration_draws,
        candidates,
        denominators: Denominators {
            scheduled_neural_draws: 0,
            completed: 0,
            surplus: 0,
            duplicate: 0,
            parser_rejected: 0,
            incomplete: 0,
            complete_sequences: 0,
            valid_programs: 0,
            unique_programs: 0,
            fallback_draws: 0,
            fallback_admitted: 0,
            exploration_draws: 0,
            exploration_admitted: 0,
            scored: 0,
            rank_survivors: 0,
            handed_off: 0,
            unique_canonical_handed_off: 0,
            shadow: 0,
        },
    };
    ledger.denominators = ledger.count();
    // The pool's own counters agree with the records.
    let d = &ledger.denominators;
    ensure!(
        d.scheduled_neural_draws == pool.attempted
            && d.valid_programs == pool.valid
            && d.duplicate == pool.within_neural_duplicates
            && d.completed == pool.unique_neural
            && d.fallback_draws as u64 == pool.fallback_attempts
            && run.ranking.chosen_count() == POOL_SIZE,
        "proposal counters differ from the draw records"
    );
    ledger.reconcile_with_vocabulary(run.vocab)?;
    Ok(ledger)
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

    /// sha256 of this ledger's canonical JSON.
    pub fn sha256(&self) -> Result<String> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Identity of one handed-off candidate's proposal record: what a GP lineage envelope's
    /// `seed_population_slot` source joins to.
    pub fn slot_record_sha256(&self, slot: usize) -> Result<String> {
        let record = self
            .candidates
            .get(slot)
            .context("slot outside the handed-off candidates")?;
        Ok(sha256_hex(&serde_json::to_vec(record)?))
    }

    /// Every draw's terminal outcome, for the candidate ledger of the search the population
    /// seeds (W4-T03): each record names the draw, its outcome, the digests of its tokens and
    /// program and, when it was handed off, its slot and slot record identity.
    pub fn draw_outcomes(&self) -> Result<DrawOutcomes> {
        fn label<T: Serialize>(outcome: &T) -> Result<String> {
            serde_json::to_value(outcome)?
                .as_str()
                .map(str::to_owned)
                .context("draw outcome is not a label")
        }
        let mut entries = Vec::with_capacity(
            self.neural_draws.len() + self.fallback_draws.len() + self.exploration_draws.len(),
        );
        for d in &self.neural_draws {
            let draw = SlotSource::NeuralDraw {
                round: d.round,
                row: d.row,
            };
            entries.push((draw, label(&d.outcome)?, &d.tokens, &d.program_sha256));
        }
        for d in &self.fallback_draws {
            let draw = SlotSource::FallbackDraw { seed: d.seed };
            entries.push((draw, label(&d.outcome)?, &d.tokens, &d.program_sha256));
        }
        for d in &self.exploration_draws {
            let draw = SlotSource::ExplorationDraw { seed: d.seed };
            entries.push((draw, label(&d.outcome)?, &d.tokens, &d.program_sha256));
        }
        let mut draws = Vec::with_capacity(entries.len());
        for (draw, outcome, tokens, program_sha256) in entries {
            let slot = self
                .candidates
                .iter()
                .find(|c| c.source == draw)
                .map(|c| c.slot);
            draws.push(DrawOutcomeRecord {
                slot_record_sha256: slot.map(|s| self.slot_record_sha256(s)).transpose()?,
                draw,
                outcome,
                tokens_sha256: tokens_sha256(tokens)?,
                program_sha256: program_sha256.clone(),
                slot,
            });
        }
        Ok(DrawOutcomes {
            ledger_sha256: self.sha256()?,
            draws,
            denominators: self.denominators.clone(),
        })
    }

    /// Export all existing draws and exact denominators, with stable I02 IDs and optional
    /// downstream decision references. A companion cannot invent a new draw or fitting grant.
    pub fn decision_links(
        &self,
        joins: &BTreeMap<String, Vec<String>>,
    ) -> Result<ProposalDecisionLinks> {
        self.reconcile()?;
        let outcomes = self.draw_outcomes()?;
        let mut used = BTreeSet::new();
        let mut draws = Vec::new();
        for draw in outcomes.draws {
            let key = match draw.draw {
                SlotSource::NeuralDraw { round, row } => format!("neural/{round}/{row}"),
                SlotSource::FallbackDraw { seed } => format!("fallback/{seed:016x}"),
                SlotSource::ExplorationDraw { seed } => format!("exploration/{seed:016x}"),
            };
            let payload = event_wire::CanonicalPayload::parse_and_canonicalize(
                &serde_json::to_vec(&serde_json::json!({
                    "schema": "jope.selfplay.draw-request/v1", "ledger_sha256": outcomes.ledger_sha256, "draw_key": key
                }))?,
            )?;
            let draw_id = payload.domain_separated_id()?.sha256_hex().to_owned();
            let decision_ids = joins.get(&draw_id).cloned().unwrap_or_default();
            ensure!(
                decision_ids.windows(2).all(|w| w[0] < w[1])
                    && decision_ids.iter().all(|s| super::is_sha256_hex(s)),
                "decision links must be sorted unique IDs"
            );
            used.insert(draw_id.clone());
            draws.push(ProposalDrawLink {
                draw_id, draw_key: key, outcome: draw.outcome, tokens_sha256: draw.tokens_sha256,
                program_sha256: draw.program_sha256, slot_record_sha256: draw.slot_record_sha256,
                missing_decision_reason: decision_ids.is_empty().then(|| "no downstream decision delivered; original shadow/outcome retained without fitting authority".into()),
                decision_ids,
            });
        }
        ensure!(
            joins.keys().all(|key| used.contains(key)),
            "orphan proposal draw/decision link"
        );
        Ok(ProposalDecisionLinks {
            schema: DECISION_LINKS_SCHEMA.into(),
            ledger_sha256: outcomes.ledger_sha256,
            denominators: outcomes.denominators,
            draws,
        })
    }

    /// Every draw that did not become a handed-off candidate, without fitting authority.
    pub fn shadow(&self) -> Vec<ShadowRecord<'_>> {
        let neural = self
            .neural_draws
            .iter()
            .filter(|d| d.outcome != DrawOutcome::Completed)
            .map(|d| ShadowRecord {
                stage: "neural",
                outcome: format!("{:?}", d.outcome),
                tokens: &d.tokens,
                program_sha256: d.program_sha256.as_deref(),
                fitting_authority: false,
            });
        fn uniform<'a>(
            stage: &'static str,
            draws: &'a [UniformDraw],
        ) -> impl Iterator<Item = ShadowRecord<'a>> + 'a {
            draws
                .iter()
                .filter(|d| d.outcome != UniformOutcome::Admitted)
                .map(move |d| ShadowRecord {
                    stage,
                    outcome: format!("{:?}", d.outcome),
                    tokens: &d.tokens,
                    program_sha256: d.program_sha256.as_deref(),
                    fitting_authority: false,
                })
        }
        neural
            .chain(uniform("fallback", &self.fallback_draws))
            .chain(uniform("exploration", &self.exploration_draws))
            .collect()
    }

    /// Re-derive every count and link from the records alone (no model, no vocabulary).
    pub fn reconcile(&self) -> Result<()> {
        ensure!(
            self.schema == LEDGER_SCHEMA,
            "unknown proposal ledger schema"
        );
        // `propose` samples regression programs only.
        ensure!(self.task == "regression", "unknown proposal task");
        for digest in [
            &self.checkpoint_sha256,
            &self.context_sha256,
            &self.vocabulary_sha256,
        ] {
            ensure!(
                super::is_sha256_hex(digest),
                "invalid proposal ledger identity"
            );
        }
        let s = &self.sampling;
        ensure!(
            (1..=MAX_ROUNDS).contains(&s.rounds_scheduled)
                && (1..=MAX_BATCH).contains(&s.batch)
                && (1..=s.rounds_scheduled).contains(&s.rounds_run)
                && s.temperature.is_finite()
                && s.temperature > 0.0
                && (1..=MAX_PROGRAM_TOKENS).contains(&s.token_budget),
            "sampling record outside the bounded configuration"
        );

        // Every scheduled neural draw has exactly one terminal record, in draw order.
        ensure!(
            self.neural_draws.len() == s.rounds_run * s.batch,
            "scheduled neural draws lack terminal records"
        );
        let mut seen = BTreeSet::new();
        let mut pooled = 0usize;
        for (i, d) in self.neural_draws.iter().enumerate() {
            ensure!(
                d.round == i / s.batch && d.row == i % s.batch,
                "neural draw records are out of order or missing"
            );
            ensure!(
                d.tokens.len() <= s.token_budget,
                "neural draw exceeds the token budget"
            );
            let finished = d.tokens.last() == Some(&EOS);
            match (d.outcome, d.program_sha256.as_deref()) {
                (DrawOutcome::Incomplete, None) => {
                    ensure!(!finished, "an incomplete draw reached EOS");
                    // The decoder ends a row only at EOS or at the token budget.
                    ensure!(
                        d.tokens.len() == s.token_budget,
                        "an incomplete draw did not exhaust its token budget"
                    )
                }
                (DrawOutcome::ParserRejected, None) => {
                    ensure!(finished, "a parser refusal did not reach EOS")
                }
                (DrawOutcome::Completed, Some(sha)) => ensure!(
                    finished && pooled < POOL_SIZE && seen.insert(sha.to_owned()),
                    "completed draw is not a new program entering an unfilled pool"
                ),
                (DrawOutcome::Surplus, Some(sha)) => ensure!(
                    finished && pooled == POOL_SIZE && seen.insert(sha.to_owned()),
                    "surplus draw is not a new program after the pool filled"
                ),
                (DrawOutcome::Duplicate, Some(sha)) => ensure!(
                    finished && seen.contains(sha),
                    "duplicate draw repeats no earlier program"
                ),
                _ => bail!("neural draw outcome and program identity disagree"),
            }
            if d.outcome == DrawOutcome::Completed {
                pooled += 1;
            }
            let last_of_round = d.row + 1 == s.batch;
            if last_of_round && d.round + 1 < s.rounds_run {
                ensure!(
                    pooled < POOL_SIZE,
                    "sampling continued after the pool filled"
                );
            }
        }
        ensure!(
            s.rounds_run == s.rounds_scheduled || pooled == POOL_SIZE,
            "sampling stopped before its scheduled rounds with an unfilled pool"
        );

        // The fallback fill tops the pool up to POOL_SIZE; the exploration fill draws
        // EXPLORATION_SIZE more programs; both from fresh consecutive streams.
        let mut filled = pooled;
        Self::reconcile_fill(
            &self.fallback_draws,
            s.fallback_seed,
            &mut seen,
            &mut filled,
            POOL_SIZE,
            "fallback",
        )?;
        let mut explored = 0;
        Self::reconcile_fill(
            &self.exploration_draws,
            s.exploration_seed,
            &mut seen,
            &mut explored,
            EXPLORATION_SIZE,
            "exploration",
        )?;

        // Each handed-off candidate comes from exactly one admitting draw, and every admitting
        // draw is handed off exactly once.
        ensure!(
            self.candidates.len() == SEED_COUNT,
            "handed-off candidates are not the {SEED_COUNT} seed slots"
        );
        let mut used = BTreeSet::new();
        let mut qualities = Vec::with_capacity(POOL_SIZE);
        for (slot, c) in self.candidates.iter().enumerate() {
            ensure!(
                c.slot == slot && sha256_hex(c.program.as_bytes()) == c.program_sha256,
                "candidate slot or program identity changed"
            );
            let pool_slot = slot < POOL_SIZE;
            let (key, admitted_tokens) = match (&c.source, c.origin.as_str()) {
                (SlotSource::NeuralDraw { round, row }, ORIGIN_NEURAL) if pool_slot => {
                    let d = self
                        .neural_draws
                        .get(round * s.batch + row)
                        .filter(|d| d.round == *round && d.row == *row)
                        .context("neural candidate names no scheduled draw")?;
                    ensure!(
                        d.outcome == DrawOutcome::Completed
                            && d.program_sha256.as_deref() == Some(c.program_sha256.as_str()),
                        "neural candidate is not its draw's completed program"
                    );
                    (format!("neural:{round}:{row}"), &d.tokens)
                }
                (SlotSource::FallbackDraw { seed }, ORIGIN_NEURAL_FALLBACK) if pool_slot => {
                    let d = Self::admitted(&self.fallback_draws, *seed, &c.program_sha256)?;
                    (format!("fallback:{seed}"), &d.tokens)
                }
                (SlotSource::ExplorationDraw { seed }, ORIGIN_RANDOM) if !pool_slot => {
                    let d = Self::admitted(&self.exploration_draws, *seed, &c.program_sha256)?;
                    (format!("exploration:{seed}"), &d.tokens)
                }
                _ => bail!("candidate {slot} origin, source and slot range disagree"),
            };
            ensure!(used.insert(key), "one draw is handed off more than once");
            ensure!(
                c.tokens == *admitted_tokens,
                "candidate tokens differ from its draw"
            );
            // Exactly the declared proposal-stage adaptations, each applied once.
            let expected = if pool_slot { 2 } else { 1 };
            ensure!(
                c.adaptations.len() == expected
                    && c.adaptations[0].kind == EDGE_DECODE
                    && c.adaptations[0].input_sha256 == tokens_sha256(&c.tokens)?
                    && c.adaptations[0].output_sha256 == c.program_sha256
                    && !c.adaptations[0].changed
                    && c.adaptations.get(1).is_none_or(|e| {
                        e.kind == EDGE_SCORE_ENCODING && e.input_sha256 == c.program_sha256
                    }),
                "candidate {slot} proposal-stage adaptations are not exactly one decode (and one scoring encoding)"
            );
            match (&c.prediction, pool_slot) {
                (Some(p), true) => {
                    ensure!(
                        p.rank == slot
                            && p.scored_program_sha256 == c.program_sha256
                            && p.predicted_quality.is_finite()
                            && p.rank_log_weight.is_finite()
                            && p.predicted_runtime_seconds.is_finite()
                            && p.predicted_runtime_seconds >= 0.0
                            && p.reward_heads.iter().all(|h| h.is_finite()),
                        "prediction is not attached to candidate {slot}'s own program"
                    );
                    qualities.push((p.predicted_quality, &c.program_sha256, &c.program));
                }
                (None, false) => {}
                _ => bail!("candidate {slot} prediction presence differs from its slot range"),
            }
        }
        ensure!(
            used.len()
                == self.denominators.completed
                    + self.denominators.fallback_admitted
                    + self.denominators.exploration_admitted,
            "an admitted draw was dropped from the handed-off candidates"
        );
        // Ranking order and the rank weights derived from it.
        for pair in qualities.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            ensure!(
                a.0 > b.0 || (a.0 == b.0 && (a.1, a.2) < (b.1, b.2)),
                "pool slots are not in the documented ranking order"
            );
        }
        let weights = logprob_from_scores(&qualities.iter().map(|q| q.0).collect::<Vec<_>>())?;
        for (c, w) in self.candidates.iter().zip(weights) {
            let p = c.prediction.as_ref().expect("pool slot prediction");
            ensure!(
                p.rank_log_weight == w,
                "rank weight is not derived from the ranked predictions"
            );
        }
        ensure!(
            self.denominators == self.count(),
            "denominators differ from the records"
        );
        Ok(())
    }

    fn admitted<'a>(draws: &'a [UniformDraw], seed: u64, sha: &str) -> Result<&'a UniformDraw> {
        let d = draws
            .iter()
            .find(|d| d.seed == seed)
            .context("grammar-uniform candidate names no draw")?;
        ensure!(
            d.outcome == UniformOutcome::Admitted && d.program_sha256.as_deref() == Some(sha),
            "grammar-uniform candidate is not its draw's admitted program"
        );
        Ok(d)
    }

    fn reconcile_fill(
        draws: &[UniformDraw],
        seed: u64,
        seen: &mut BTreeSet<String>,
        filled: &mut usize,
        target: usize,
        stage: &str,
    ) -> Result<()> {
        ensure!(
            draws.len() as u64 <= FALLBACK_ATTEMPTS,
            "{stage} fill exceeds its draw bound"
        );
        for (offset, d) in draws.iter().enumerate() {
            ensure!(
                d.seed == seed.wrapping_add(offset as u64) && *filled < target,
                "{stage} draws are not consecutive fresh streams of an unfilled fill"
            );
            let finished = d.tokens.last() == Some(&EOS);
            match (d.outcome, d.program_sha256.as_deref()) {
                (UniformOutcome::Admitted, Some(sha)) => {
                    ensure!(
                        finished && seen.insert(sha.to_owned()),
                        "{stage} admitted a program already seen"
                    );
                    *filled += 1;
                }
                (UniformOutcome::Duplicate, Some(sha)) => ensure!(
                    finished && seen.contains(sha),
                    "{stage} duplicate repeats no earlier program"
                ),
                (UniformOutcome::ReencodingMismatch, Some(_)) => {
                    ensure!(finished, "{stage} re-encoding refusal did not reach EOS")
                }
                (UniformOutcome::ParserRejected, None) => {
                    ensure!(finished, "{stage} parser refusal did not reach EOS")
                }
                (UniformOutcome::Incomplete, None) => {
                    ensure!(!finished, "{stage} incomplete draw reached EOS");
                    // A grammar-uniform walk ends only at EOS or at MAX_PROGRAM_TOKENS.
                    ensure!(
                        d.tokens.len() == MAX_PROGRAM_TOKENS,
                        "{stage} incomplete draw did not exhaust its token budget"
                    )
                }
                _ => bail!("{stage} draw outcome and program identity disagree"),
            }
        }
        ensure!(*filled == target, "{stage} fill did not complete");
        Ok(())
    }

    /// [`Self::reconcile`], plus the vocabulary-dependent links: every draw starts from the
    /// task's proposal prefix and its recorded outcome is what its tokens decode and parse to (so
    /// a valid program cannot be recorded as a refusal, nor a refusal as a program), each
    /// candidate's tokens decode to its program, its scoring encoding is the vocabulary's
    /// encoding of that program, and its canonical recipe is the parser's.
    pub fn reconcile_with_vocabulary(&self, vocab: &Vocab) -> Result<()> {
        self.reconcile()?;
        let prefix = vocab.proposal_prefix(&self.task)?;
        let program_of = |tokens: &[usize]| -> Result<Option<String>> {
            ensure!(
                tokens.starts_with(&prefix),
                "draw does not start from the task's proposal prefix"
            );
            Ok((tokens.last() == Some(&EOS))
                .then(|| super::decode::valid_genome(vocab, tokens))
                .flatten())
        };
        // Replay a draw through the sampling grammar: every drawn token is legal where it was
        // drawn; returns whether the walk accepted (reached EOS) and its final phase.
        let walk = |tokens: &[usize]| -> Result<Option<(bool, &'static str)>> {
            let mut state = vocab.prefix_state(&self.task)?;
            for &token in tokens.get(prefix.len()..).unwrap_or_default() {
                if !state.legal(vocab).contains(&token) {
                    return Ok(None);
                }
                state.advance(vocab, token)?;
            }
            Ok(Some((state.done(), state.phase_name())))
        };
        for d in &self.neural_draws {
            let program = program_of(&d.tokens)?;
            // An incomplete draw stopped before the grammar accepted; every other draw ended at
            // an accepting EOS. Both are legal walks ending in the recorded grammar phase.
            ensure!(
                walk(&d.tokens)?
                    == Some((
                        d.outcome != DrawOutcome::Incomplete,
                        d.grammar_phase.as_str()
                    )),
                "neural draw ({}, {}) is not a legal grammar walk ending in its recorded outcome",
                d.round,
                d.row
            );
            let valid = matches!(
                d.outcome,
                DrawOutcome::Completed | DrawOutcome::Surplus | DrawOutcome::Duplicate
            );
            ensure!(
                valid == program.is_some()
                    && program.map(|p| sha256_hex(p.as_bytes())) == d.program_sha256,
                "neural draw ({}, {}) outcome differs from its tokens",
                d.round,
                d.row
            );
        }
        for d in self.fallback_draws.iter().chain(&self.exploration_draws) {
            let program = program_of(&d.tokens)?;
            ensure!(
                walk(&d.tokens)?.map(|(accepted, _)| accepted)
                    == Some(d.outcome != UniformOutcome::Incomplete),
                "grammar-uniform draw {} is not a legal grammar walk ending in its recorded outcome",
                d.seed
            );
            let exact = program.as_ref().map(|p| {
                vocab.encode_for_task(p, &self.task).ok().as_deref() == Some(&d.tokens[..])
            });
            let consistent = match d.outcome {
                UniformOutcome::Admitted | UniformOutcome::Duplicate => exact == Some(true),
                UniformOutcome::ReencodingMismatch => exact == Some(false),
                UniformOutcome::ParserRejected | UniformOutcome::Incomplete => program.is_none(),
            };
            ensure!(
                consistent && program.map(|p| sha256_hex(p.as_bytes())) == d.program_sha256,
                "grammar-uniform draw {} outcome differs from its tokens",
                d.seed
            );
        }
        for c in &self.candidates {
            ensure!(
                vocab.decode(&c.tokens)? == c.program
                    && canonical_sha256(&c.program)? == c.canonical_recipe_sha256,
                "candidate {} tokens do not decode to its program",
                c.slot
            );
            if let Some(edge) = c.adaptations.get(1) {
                let scored = vocab.encode(&c.program)?;
                ensure!(
                    edge.output_sha256 == tokens_sha256(&scored)?
                        && edge.changed == (scored != c.tokens)
                        && c.prediction.as_ref().map(|p| p.scored_token_count)
                            == Some(scored.len()),
                    "candidate {} scoring encoding differs from the vocabulary's",
                    c.slot
                );
            }
        }
        Ok(())
    }

    /// The ledger describes exactly this seed population (slot order, program, origin, weight).
    pub fn matches_population(&self, population: &[SeedCandidate]) -> Result<()> {
        ensure!(
            population.len() == self.candidates.len(),
            "ledger and population sizes differ"
        );
        for (c, seed) in self.candidates.iter().zip(population) {
            ensure!(
                c.program == seed.genome()
                    && c.origin == seed.origin()
                    && c.prediction.as_ref().map(|p| p.rank_log_weight) == seed.rank_log_weight(),
                "ledger slot {} differs from the seed population",
                c.slot
            );
            // W4-T07 join: a seed slot that names a proposal record names exactly this slot's.
            if let Some(record) = seed.proposal_record_sha256() {
                ensure!(
                    record == sha256_hex(&serde_json::to_vec(c)?),
                    "seed slot {} names another proposal record",
                    c.slot
                );
            }
        }
        Ok(())
    }

    /// Canonical recipe identities in slot order.
    pub fn canonical_recipes(&self) -> Vec<&str> {
        self.candidates
            .iter()
            .map(|c| c.canonical_recipe_sha256.as_str())
            .collect()
    }

    /// The candidate records keyed by canonical recipe: canonical duplicates among handed-off
    /// candidates are visible rather than counted as independent proposals.
    pub fn canonical_groups(&self) -> BTreeMap<&str, Vec<usize>> {
        let mut groups = BTreeMap::<&str, Vec<usize>>::new();
        for c in &self.candidates {
            groups
                .entry(c.canonical_recipe_sha256.as_str())
                .or_default()
                .push(c.slot);
        }
        groups
    }
}

#[cfg(test)]
mod decision_link_tests {
    use super::*;

    #[test]
    fn companion_retains_every_draw_without_changing_legacy_ledger() -> Result<()> {
        let mut config = crate::config::PolicyConfig::regression(401);
        config.width = 16;
        config.heads = 4;
        config.ffn = 32;
        config.layers = 1;
        config.dropout = 0.;
        config.max_seq = 128;
        let model = crate::model::PolicyModel::random_weights(config.clone(), 5)?;
        let checkpoint = crate::config::CheckpointBinding::unbound_for_testing(
            config,
            crate::CHECKPOINT_VOCABULARY_SHA256,
        );
        let vocabulary = crate::grammar::bind_vocabulary(
            include_bytes!("../../jope-engine/assets/grammar/featmath.json"),
            crate::CHECKPOINT_VOCABULARY_SHA256,
        )?;
        let context = crate::context::PolicyContext::from_frozen_values(vec![0.; 2048])?;
        let hash = "a".repeat(64);
        let context_hash = crate::context_sha256(&context);
        let binding = crate::seeds::SeedBinding::new(&hash, 0, &hash, &context_hash)?;
        let proposal = crate::seeds::propose_unbound(
            &model,
            &checkpoint,
            &vocabulary,
            &context,
            &binding,
            &crate::seeds::ProposalSeeds::derive(7),
            &crate::decode::SamplingOptions {
                rounds: 1,
                batch: 8,
                temperature: 1.,
            },
        )?;
        let ledger = proposal.ledger();
        let before = serde_json::to_vec(ledger)?;
        let companion = ledger.decision_links(&BTreeMap::new())?;
        companion.verify(ledger)?;
        assert_eq!(
            serde_json::from_slice::<ProposalDecisionLinks>(&companion.canonical_bytes()?)?,
            companion
        );
        assert_eq!(companion.draws.len(), ledger.draw_outcomes()?.draws.len());
        assert_eq!(companion.denominators, ledger.denominators);
        assert!(companion
            .draws
            .iter()
            .any(|d| d.slot_record_sha256.is_none()));
        assert!(companion
            .draws
            .iter()
            .all(|d| d.missing_decision_reason.is_some()));
        let joins = [(companion.draws[0].draw_id.clone(), vec!["b".repeat(64)])].into();
        let linked = ledger.decision_links(&joins)?;
        linked.verify(ledger)?;
        assert_eq!(linked.draws[0].decision_ids.len(), 1);
        let mut missing = linked.clone();
        missing.draws.pop();
        assert!(missing.verify(ledger).is_err());
        let mut changed = linked;
        changed.draws[0].outcome = "invented".into();
        assert!(changed.verify(ledger).is_err());
        assert!(ledger
            .decision_links(&[("c".repeat(64), vec!["d".repeat(64)])].into())
            .is_err());
        assert_eq!(serde_json::to_vec(ledger)?, before);
        Ok(())
    }
}
