use super::recorder::{Event, EventKind, Recorder, Trace};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const STRATEGY_ID: &str = "jope.search.deterministic.v1";
pub const STRATEGY_VERSION: u32 = 1;
pub const STRATEGY_EVENT_VERSION: &str = "jope.search-events.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub strategy_id: String,
    pub strategy_version: u32,
    pub strategy_config_sha256: String,
    pub seed: u64,
    pub profile_id: String,
    pub profile_revision: u32,
    pub profile_sha256: String,
    pub evaluator_id: String,
    pub evaluator_version: String,
    pub seed_policy_id: String,
    pub cost_profile_id: String,
    /// Full commit SHA supplied by the driver.
    pub source_id: String,
    pub context_id: String,
    pub input_sha256: String,
    pub event_version: String,
}

impl Identity {
    pub fn new(
        profile_id: impl Into<String>,
        profile_revision: u32,
        profile_sha256: impl Into<String>,
        evaluator_id: impl Into<String>,
        evaluator_version: impl Into<String>,
        seed_policy_id: impl Into<String>,
        cost_profile_id: impl Into<String>,
    ) -> Self {
        Self {
            strategy_id: STRATEGY_ID.into(),
            strategy_version: STRATEGY_VERSION,
            strategy_config_sha256: String::new(),
            seed: 0,
            profile_id: profile_id.into(),
            profile_revision,
            profile_sha256: profile_sha256.into(),
            evaluator_id: evaluator_id.into(),
            evaluator_version: evaluator_version.into(),
            seed_policy_id: seed_policy_id.into(),
            cost_profile_id: cost_profile_id.into(),
            source_id: String::new(),
            context_id: String::new(),
            input_sha256: String::new(),
            event_version: STRATEGY_EVENT_VERSION.into(),
        }
    }

    pub fn sha256(&self) -> String {
        digest(&serde_json::to_vec(self).expect("identity is serializable"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyConfig {
    /// Maximum proposal slots, including duplicate proposals and failed evaluations.
    pub max_attempts: usize,
    /// Hard deterministic work cap in the allocator's declared cost units.
    pub cost_budget: u64,
    /// Candidate text accepted by the driver is kept byte-exact in this core.
    pub max_candidate_bytes: usize,
}

impl StrategyConfig {
    pub fn validate(&self) -> Result<(), StrategyError> {
        if self.max_attempts == 0 || self.cost_budget == 0 || self.max_candidate_bytes == 0 {
            return Err(StrategyError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub id: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    pub candidate: Candidate,
    /// Stable, source-defined proposal kind, such as seed, mutation or crossover.
    pub origin: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    pub units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub id: String,
    pub cost: Cost,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// `measured`, `parse_failed`, `fit_failed`, `score_failed`, `panicked`, or a versioned code.
    pub outcome: String,
    /// Finite score where lower is better; absence means the result cannot rank.
    pub score: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub candidate: Candidate,
    pub result: Evaluation,
    pub reservation: Reservation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StrategyError {
    InvalidConfig,
    InvalidIdentity,
    CandidateTooLarge,
    InvalidCost,
    NonFiniteScore,
    CostOverflow,
}

impl std::fmt::Display for StrategyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for StrategyError {}

/// Source of candidate proposals. Implementations must be deterministic for a fixed input state.
pub trait Proposer {
    fn propose(&mut self, attempt: usize) -> Option<Proposal>;
}

/// Resource adapter. Reservation policy and units are named by the immutable identity.
pub trait Allocator {
    fn reserve(&mut self, candidate: &Candidate, remaining: u64) -> Option<Reservation>;
}

/// Evaluation adapter. Effects, fitting and numeric identity remain in the caller.
pub trait Evaluator {
    fn evaluate(&mut self, candidate: &Candidate, reservation: &Reservation) -> Evaluation;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub identity: Identity,
    pub identity_sha256: String,
    pub consumed_cost: u64,
    pub outcomes: Vec<Outcome>,
    pub best: Option<usize>,
    pub trace: Trace,
}

#[derive(Clone, Debug)]
pub struct Strategy {
    pub identity: Identity,
    pub config: StrategyConfig,
}

/// Run proposal → duplicate admission → resource reservation → evaluation in stable order.
/// Completion order is owned by the driver; callers that evaluate concurrently must present
/// results back in reservation order before recording them here.
pub fn execute<P, A, E>(
    strategy: &Strategy,
    proposer: &mut P,
    allocator: &mut A,
    evaluator: &mut E,
) -> Result<Run, StrategyError>
where
    P: Proposer,
    A: Allocator,
    E: Evaluator,
{
    strategy.config.validate()?;
    if strategy.identity.strategy_id != STRATEGY_ID
        || strategy.identity.strategy_version != STRATEGY_VERSION
        || strategy.identity.event_version != STRATEGY_EVENT_VERSION
        || strategy.identity.profile_sha256.is_empty()
        || strategy.identity.evaluator_id.is_empty()
        || strategy.identity.evaluator_version.is_empty()
        || strategy.identity.seed_policy_id.is_empty()
        || strategy.identity.cost_profile_id.is_empty()
        || strategy.identity.context_id.is_empty()
        || strategy.identity.input_sha256.is_empty()
        || !is_full_sha(&strategy.identity.source_id)
    {
        return Err(StrategyError::InvalidIdentity);
    }
    let mut identity = strategy.identity.clone();
    identity.strategy_config_sha256 =
        digest(&serde_json::to_vec(&strategy.config).expect("strategy config is serializable"));
    let identity_sha256 = identity.sha256();
    let mut trace = Trace {
        event_version: STRATEGY_EVENT_VERSION.to_owned(),
        identity_sha256: identity_sha256.clone(),
        events: Vec::new(),
    };
    let mut seen = BTreeSet::new();
    let mut outcomes = Vec::new();
    let mut consumed = 0u64;
    let mut sequence = 0u64;
    for attempt in 0..strategy.config.max_attempts {
        let Some(proposal) = proposer.propose(attempt) else {
            break;
        };
        let candidate = proposal.candidate;
        let candidate_sha256 = digest(&candidate.bytes);
        record_with_origin(
            &mut trace,
            &mut sequence,
            EventKind::Proposed,
            &candidate,
            &candidate_sha256,
            None,
            None,
            None,
            Some(proposal.origin),
            None,
        );
        if candidate.bytes.len() > strategy.config.max_candidate_bytes {
            return Err(StrategyError::CandidateTooLarge);
        }
        if !seen.insert(candidate.bytes.clone()) {
            record(
                &mut trace,
                &mut sequence,
                EventKind::Duplicate,
                &candidate,
                &candidate_sha256,
                None,
                None,
                None,
            );
            continue;
        }
        let remaining = strategy.config.cost_budget.saturating_sub(consumed);
        let Some(reservation) = allocator.reserve(&candidate, remaining) else {
            record(
                &mut trace,
                &mut sequence,
                EventKind::ReservationRefused,
                &candidate,
                &candidate_sha256,
                None,
                None,
                None,
            );
            break;
        };
        if reservation.cost.units == 0 || reservation.cost.units > remaining {
            return Err(StrategyError::InvalidCost);
        }
        consumed = consumed
            .checked_add(reservation.cost.units)
            .ok_or(StrategyError::CostOverflow)?;
        record(
            &mut trace,
            &mut sequence,
            EventKind::Reserved,
            &candidate,
            &candidate_sha256,
            Some(reservation.id.clone()),
            Some(reservation.cost.units),
            None,
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            evaluator.evaluate(&candidate, &reservation)
        }))
        .unwrap_or_else(|_| Evaluation {
            outcome: "panicked".into(),
            score: None,
        });
        if result.score.is_some_and(|score| !score.is_finite()) {
            return Err(StrategyError::NonFiniteScore);
        }
        record_with_score(
            &mut trace,
            &mut sequence,
            EventKind::Evaluated,
            &candidate,
            &candidate_sha256,
            Some(reservation.id.clone()),
            Some(reservation.cost.units),
            Some(result.outcome.clone()),
            result.score.map(f64::to_bits),
        );
        outcomes.push(Outcome {
            candidate,
            result,
            reservation,
        });
    }
    let best = outcomes
        .iter()
        .enumerate()
        .filter_map(|(i, outcome)| outcome.result.score.map(|score| (i, score)))
        .min_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ia.cmp(ib)))
        .map(|(i, _)| i);
    Ok(Run {
        identity,
        identity_sha256,
        consumed_cost: consumed,
        outcomes,
        best,
        trace,
    })
}

/// Execute and deliver the completed causal event sequence to an injected sink.
/// The callback runs in sequence order after execution so a sink cannot affect algorithm choices.
pub fn execute_with_recorder<P, A, E, R>(
    strategy: &Strategy,
    proposer: &mut P,
    allocator: &mut A,
    evaluator: &mut E,
    recorder: &mut R,
) -> Result<Run, StrategyError>
where
    P: Proposer,
    A: Allocator,
    E: Evaluator,
    R: Recorder,
{
    let run = execute(strategy, proposer, allocator, evaluator)?;
    for event in &run.trace.events {
        recorder.record(event.clone());
    }
    Ok(run)
}

#[allow(clippy::too_many_arguments)]
fn record(
    trace: &mut Trace,
    sequence: &mut u64,
    kind: EventKind,
    candidate: &Candidate,
    candidate_sha256: &str,
    reservation_id: Option<String>,
    cost_units: Option<u64>,
    outcome: Option<String>,
) {
    record_with_origin(
        trace,
        sequence,
        kind,
        candidate,
        candidate_sha256,
        reservation_id,
        cost_units,
        outcome,
        None,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn record_with_score(
    trace: &mut Trace,
    sequence: &mut u64,
    kind: EventKind,
    candidate: &Candidate,
    candidate_sha256: &str,
    reservation_id: Option<String>,
    cost_units: Option<u64>,
    outcome: Option<String>,
    score_bits: Option<u64>,
) {
    record_with_origin(
        trace,
        sequence,
        kind,
        candidate,
        candidate_sha256,
        reservation_id,
        cost_units,
        outcome,
        None,
        score_bits,
    );
}

#[allow(clippy::too_many_arguments)]
fn record_with_origin(
    trace: &mut Trace,
    sequence: &mut u64,
    kind: EventKind,
    candidate: &Candidate,
    candidate_sha256: &str,
    reservation_id: Option<String>,
    cost_units: Option<u64>,
    outcome: Option<String>,
    origin: Option<String>,
    score_bits: Option<u64>,
) {
    trace.record(Event {
        sequence: *sequence,
        kind,
        candidate_id: candidate.id.clone(),
        candidate_sha256: candidate_sha256.to_owned(),
        origin,
        reservation_id,
        cost_units,
        outcome,
        score_bits,
    });
    *sequence += 1;
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_full_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
