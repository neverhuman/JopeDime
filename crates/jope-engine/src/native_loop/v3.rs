//! Draft v3 bounded campaign protocol for native multi-round research.
//!
//! Status: retained as a noncompiled design draft; no current entrypoint imports
//! this file. Owner: JopeDime engine maintainers. Review/expiry: 2027-01-01.
//! Migration plan: either promote it by adding explicit protocol tests and an
//! entrypoint that consumes it, or preserve its history and remove the draft
//! after its outstanding state transitions have been reconciled.
//!
//! This module is intentionally private to the worker allocation until the
//! integrator adds the version dispatcher, schema files, and observer hook.
//! It models durable protocol transitions; it does not run an optimizer or
//! confer serving eligibility.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const LOOP_V3: &str = "jope.native_loop.v3";
pub const COMPLETION_V3: &str = "jope.native_loop_completion.v3";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    pub id: String,
    pub campaign: String,
    pub deadline_unix: u64,
    pub max_rounds: u32,
    pub max_effort: u64,
    pub max_physical_retries: u32,
}

impl Grant {
    fn validate(&self) -> Result<(), Error> {
        if self.id.is_empty() || self.campaign.is_empty() {
            return Err(Error::Invalid("grant and campaign IDs must be nonempty"));
        }
        if self.deadline_unix == 0 || self.max_rounds == 0 || self.max_effort == 0 {
            return Err(Error::Invalid("grant deadline and caps must be positive"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Accepted,
    Rejected,
    Inconclusive,
    Quarantined,
    Aborted,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundEvidence {
    pub intent_id: String,
    pub committed_child_id: String,
    pub assessment_id: String,
    pub assessment_intent_id: String,
    pub assessment_child_id: String,
    pub compatible: bool,
}

impl RoundEvidence {
    fn validate(&self) -> Result<(), Error> {
        if self.intent_id.is_empty()
            || self.committed_child_id.is_empty()
            || self.assessment_id.is_empty()
            || self.assessment_intent_id.is_empty()
            || self.assessment_child_id.is_empty()
        {
            return Err(Error::Invalid(
                "intent, committed child, and assessment are required",
            ));
        }
        if self.assessment_intent_id != self.intent_id
            || self.assessment_child_id != self.committed_child_id
        {
            return Err(Error::Conflict(
                "assessment must bind the exact intent and committed child",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Round {
    pub number: u32,
    pub parent_id: String,
    pub child_id: String,
    pub evidence: RoundEvidence,
    pub decision: Decision,
    pub effort: u64,
    pub physical_retries: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    /// Stable caller-supplied idempotency key, persisted before applying effects.
    pub id: String,
    pub observed_at_unix: u64,
    pub round: Round,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Campaign {
    pub schema: String,
    pub grant: Grant,
    pub accepted_parent: String,
    pub rounds: Vec<Round>,
    pub spent_effort: u64,
    pub physical_retries: u32,
    pub applied_events: BTreeMap<String, Event>,
    pub terminal: Option<Decision>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Invalid(&'static str),
    Conflict(&'static str),
    Exhausted(&'static str),
    Terminal,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid campaign event: {message}"),
            Self::Conflict(message) => write!(f, "campaign conflict: {message}"),
            Self::Exhausted(message) => write!(f, "campaign grant exhausted: {message}"),
            Self::Terminal => f.write_str("campaign is terminal"),
        }
    }
}

impl std::error::Error for Error {}

impl Campaign {
    pub fn start(grant: Grant, initial_parent: impl Into<String>) -> Result<Self, Error> {
        grant.validate()?;
        let accepted_parent = initial_parent.into();
        if accepted_parent.is_empty() {
            return Err(Error::Invalid("initial parent is required"));
        }
        Ok(Self {
            schema: LOOP_V3.into(),
            grant,
            accepted_parent,
            rounds: Vec::new(),
            spent_effort: 0,
            physical_retries: 0,
            applied_events: BTreeMap::new(),
            terminal: None,
        })
    }

    /// Resume from persisted state while proving the original grant is unchanged.
    pub fn resume(&self, original_grant: &Grant, now_unix: u64) -> Result<(), Error> {
        original_grant.validate()?;
        if &self.grant != original_grant {
            return Err(Error::Conflict(
                "resume cannot replace or reset the original grant",
            ));
        }
        if self.schema != LOOP_V3 || self.rounds.len() as u32 > self.grant.max_rounds {
            return Err(Error::Invalid("schema or persisted round count is invalid"));
        }
        if self.spent_effort > self.grant.max_effort
            || self.physical_retries > self.grant.max_physical_retries
        {
            return Err(Error::Invalid("persisted aggregate use exceeds its grant"));
        }
        if now_unix >= self.grant.deadline_unix && self.terminal.is_none() {
            return Err(Error::Exhausted("original deadline has elapsed"));
        }
        Ok(())
    }

    /// Apply a persisted event exactly once. Exact replay is a no-op; key reuse
    /// with different contents is rejected. A rejected child is recorded but
    /// never replaces the accepted research parent.
    pub fn apply(&mut self, event: Event) -> Result<bool, Error> {
        if event.id.is_empty() {
            return Err(Error::Invalid("event id is required"));
        }
        if let Some(previous) = self.applied_events.get(&event.id) {
            return if previous == &event {
                Ok(false)
            } else {
                Err(Error::Conflict(
                    "idempotency key was reused with different event",
                ))
            };
        }
        if self.terminal.is_some() {
            return Err(Error::Terminal);
        }
        let event_id = event.id.clone();
        let mut round = event.round.clone();
        round.evidence.validate()?;
        if round.decision == Decision::Accepted && !round.evidence.compatible {
            return Err(Error::Conflict(
                "an incompatible assessment cannot advance the research parent",
            ));
        }
        if round.number != self.rounds.len() as u32 + 1 {
            return Err(Error::Conflict("round numbers must be contiguous"));
        }
        if round.number > self.grant.max_rounds {
            self.terminal = Some(Decision::Aborted);
            return Err(Error::Exhausted("maximum rounds exhausted"));
        }
        if round.parent_id != self.accepted_parent
            || round.child_id != round.evidence.committed_child_id
        {
            return Err(Error::Conflict(
                "parent or committed-child lineage mismatch",
            ));
        }
        if self.rounds.iter().any(|prior| {
            prior.evidence.intent_id == round.evidence.intent_id
                || prior.evidence.committed_child_id == round.evidence.committed_child_id
                || prior.evidence.assessment_id == round.evidence.assessment_id
        }) {
            return Err(Error::Conflict(
                "intent, committed child, and assessment IDs must be unique per campaign",
            ));
        }
        let effort = self
            .spent_effort
            .checked_add(round.effort)
            .ok_or(Error::Exhausted("effort overflow"))?;
        let retries = self
            .physical_retries
            .checked_add(round.physical_retries)
            .ok_or(Error::Exhausted("retry overflow"))?;
        self.spent_effort = effort;
        self.physical_retries = retries;
        if effort > self.grant.max_effort || retries > self.grant.max_physical_retries {
            round.decision = Decision::Aborted;
            self.rounds.push(round);
            self.applied_events.insert(event_id, event);
            self.terminal = Some(Decision::Aborted);
            return Ok(true);
        }
        let deadline_elapsed = event.observed_at_unix >= self.grant.deadline_unix;
        if deadline_elapsed {
            round.decision = Decision::Aborted;
        }
        self.rounds.push(round.clone());
        self.applied_events.insert(event_id, event);
        if deadline_elapsed {
            self.terminal = Some(Decision::Aborted);
            return Ok(true);
        }
        match round.decision {
            Decision::Accepted => self.accepted_parent = round.child_id.clone(),
            decision => self.terminal = Some(decision),
        }
        if self.terminal.is_none() && self.rounds.len() as u32 == self.grant.max_rounds {
            self.terminal = Some(Decision::Accepted);
        }
        Ok(true)
    }

    pub fn completion(&self) -> Completion {
        Completion {
            schema: COMPLETION_V3.into(),
            campaign: self.grant.campaign.clone(),
            grant_id: self.grant.id.clone(),
            terminal: self.terminal,
            accepted_parent: self.accepted_parent.clone(),
            rounds: self.rounds.clone(),
            spent_effort: self.spent_effort,
            physical_retries: self.physical_retries,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Completion {
    pub schema: String,
    pub campaign: String,
    pub grant_id: String,
    pub terminal: Option<Decision>,
    pub accepted_parent: String,
    pub rounds: Vec<Round>,
    pub spent_effort: u64,
    pub physical_retries: u32,
}
