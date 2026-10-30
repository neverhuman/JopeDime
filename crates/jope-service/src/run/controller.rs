use super::{AttemptKey, Evaluation, EvaluationOutcome, EvaluationRequest};
use jope_api::{error::ErrorCode, ids::RunId, CompletionReason, Effort, RunPhase, RunState};
use std::{collections::BTreeMap, num::NonZeroU32};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImmutableField {
    Objective,
    Target,
    Task,
    Dataset,
    Columns,
    Split,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Pause,
    Continue,
    Cancel,
    SetEffort(Effort),
    ChangeImmutable(ImmutableField),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlError {
    pub code: ErrorCode,
    pub revision: u64,
    pub detail: &'static str,
}

impl std::fmt::Display for ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} at revision {}: {}",
            self.code, self.revision, self.detail
        )
    }
}
impl std::error::Error for ControlError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResultDisposition {
    Buffered,
    DiscardedLate,
}

/// Internal view, not a replacement for the public RunSnapshot wire schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControllerSnapshot {
    pub run: RunId,
    pub state: RunState,
    pub phase: RunPhase,
    pub revision: u64,
    pub effort: Effort,
    pub pending_effort: Option<Effort>,
    pub next_round: u64,
    pub planned_rounds_in_flight: NonZeroU32,
    pub attempt_high_water: u64,
    pub pending_slots: usize,
    pub interrupted: bool,
    pub completion_reason: Option<CompletionReason>,
    pub failure: Option<ErrorCode>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommittedRound {
    pub round: u64,
    pub effort: Effort,
    pub evaluations: Vec<Evaluation>,
}

#[derive(Clone, Debug, PartialEq)]
struct Pending {
    request: EvaluationRequest,
    outcome: Option<EvaluationOutcome>,
}

#[derive(Clone, Debug, PartialEq)]
struct Round {
    effort: Effort,
    slots: BTreeMap<u32, Pending>,
}

/// In-memory transition/fencing kernel. Its owner must durably commit effects.
#[derive(Clone, Debug, PartialEq)]
pub struct RunController {
    view: ControllerSnapshot,
    checkpoint_phase: RunPhase,
    rounds: BTreeMap<u64, Round>,
    committed: Vec<CommittedRound>,
}

impl RunController {
    /// Validation precedes construction; R is fixed by the validated plan.
    pub fn new(run: RunId, effort: Effort, planned_rounds_in_flight: NonZeroU32) -> Self {
        Self {
            view: ControllerSnapshot {
                run,
                state: RunState::Running,
                phase: RunPhase::Preparing,
                revision: 1,
                effort,
                pending_effort: None,
                next_round: 0,
                planned_rounds_in_flight,
                attempt_high_water: 0,
                pending_slots: 0,
                interrupted: false,
                completion_reason: None,
                failure: None,
            },
            checkpoint_phase: RunPhase::Preparing,
            rounds: BTreeMap::new(),
            committed: Vec::new(),
        }
    }

    pub fn snapshot(&self) -> ControllerSnapshot {
        self.view.clone()
    }

    pub fn committed(&self) -> &[CommittedRound] {
        &self.committed
    }

    fn error(&self, code: ErrorCode, detail: &'static str) -> ControlError {
        ControlError {
            code,
            revision: self.view.revision,
            detail,
        }
    }

    fn invalid(&self, detail: &'static str) -> ControlError {
        self.error(ErrorCode::InvalidInput, detail)
    }

    fn next_revision(&self) -> Result<u64, ControlError> {
        self.view
            .revision
            .checked_add(1)
            .ok_or_else(|| self.invalid("revision exhausted"))
    }

    fn live(&self) -> bool {
        matches!(self.view.state, RunState::Running | RunState::Pausing)
    }

    pub fn control(&mut self, command: Control, if_match: Option<u64>) -> Result<(), ControlError> {
        if if_match.is_some_and(|revision| revision != self.view.revision) {
            return Err(self.error(
                ErrorCode::RevisionConflict,
                "re-read the snapshot and retry",
            ));
        }
        if matches!(command, Control::ChangeImmutable(_)) {
            return Err(self.error(
                ErrorCode::ImmutableField,
                "use restart to change this field",
            ));
        }
        let target = match (self.view.state, command) {
            (RunState::Running, Control::Pause) => RunState::Pausing,
            (RunState::Paused, Control::Continue) => RunState::Running,
            (RunState::Running | RunState::Pausing | RunState::Paused, Control::Cancel) => {
                RunState::Cancelled
            }
            (
                RunState::Running | RunState::Pausing | RunState::Paused,
                Control::SetEffort(effort),
            ) => {
                if self.view.pending_effort.unwrap_or(self.view.effort) == effort {
                    return Ok(());
                }
                let revision = self.next_revision()?;
                if self.rounds.is_empty() {
                    self.view.effort = effort;
                    self.view.pending_effort = None;
                } else {
                    self.view.pending_effort = Some(effort);
                }
                self.view.revision = revision;
                return Ok(());
            }
            _ => return Err(self.invalid("control is not allowed in this state")),
        };
        let revision = self.next_revision()?;
        if target == RunState::Cancelled {
            self.rounds.clear();
            self.view.pending_slots = 0;
            self.view.pending_effort = None;
        }
        if target == RunState::Running {
            self.view.interrupted = false;
        }
        self.view.state = target;
        self.view.revision = revision;
        Ok(())
    }

    pub fn advance_phase(&mut self, phase: RunPhase) -> Result<(), ControlError> {
        let valid = matches!(
            (self.view.phase, phase),
            (RunPhase::Preparing, RunPhase::Encoding)
                | (RunPhase::Encoding, RunPhase::Seeding)
                | (RunPhase::Seeding, RunPhase::Searching)
                | (RunPhase::Searching, RunPhase::Finalizing)
        );
        if self.view.state != RunState::Running || !self.rounds.is_empty() || !valid {
            return Err(self.invalid("phase must advance at a quiescent running boundary"));
        }
        let revision = self.next_revision()?;
        self.view.phase = phase;
        self.view.revision = revision;
        Ok(())
    }

    pub fn dispatch(
        &mut self,
        round: u64,
        slot: u32,
        seed: u64,
    ) -> Result<EvaluationRequest, ControlError> {
        if self.view.state != RunState::Running
            || !matches!(self.view.phase, RunPhase::Seeding | RunPhase::Searching)
        {
            return Err(self.invalid("evaluation admission requires a running search phase"));
        }
        if let Some(pending) = self.rounds.get(&round) {
            if pending.slots.contains_key(&slot) {
                return Err(self.invalid("slot already admitted"));
            }
        } else {
            let next = self
                .rounds
                .last_key_value()
                .map_or(Some(self.view.next_round), |(&last, _)| last.checked_add(1));
            if next != Some(round)
                || self.rounds.len() >= self.view.planned_rounds_in_flight.get() as usize
            {
                return Err(self.invalid("round is out of order or exceeds planned R"));
            }
        }
        let attempt = self
            .view
            .attempt_high_water
            .checked_add(1)
            .ok_or_else(|| self.invalid("attempt counter exhausted"))?;
        let pending = self.rounds.entry(round).or_insert_with(|| Round {
            effort: self.view.effort,
            slots: BTreeMap::new(),
        });
        let request = EvaluationRequest {
            key: AttemptKey {
                run: self.view.run.clone(),
                round,
                slot,
                attempt,
            },
            seed,
            effort: pending.effort,
        };
        pending.slots.insert(
            slot,
            Pending {
                request: request.clone(),
                outcome: None,
            },
        );
        self.view.attempt_high_water = attempt;
        self.view.pending_slots += 1;
        Ok(request)
    }

    pub fn receive(&mut self, result: Evaluation) -> Result<ResultDisposition, ControlError> {
        if !self.live() || result.key.run != self.view.run {
            return Ok(ResultDisposition::DiscardedLate);
        }
        let current = self
            .rounds
            .get(&result.key.round)
            .and_then(|r| r.slots.get(&result.key.slot));
        if !current.is_some_and(|p| p.request.key == result.key && p.outcome.is_none()) {
            return Ok(ResultDisposition::DiscardedLate);
        }
        match &result.outcome {
            EvaluationOutcome::Scored(score) if !score.is_finite() => {
                return Err(self.invalid("evaluation score must be finite"))
            }
            EvaluationOutcome::DiscardedAtPauseDeadline => {
                return Err(self.invalid("only the controller may revoke a pause-deadline attempt"))
            }
            _ => {}
        }
        self.rounds
            .get_mut(&result.key.round)
            .unwrap()
            .slots
            .get_mut(&result.key.slot)
            .unwrap()
            .outcome = Some(result.outcome);
        Ok(ResultDisposition::Buffered)
    }

    pub fn expire_pause(&mut self) -> Result<Vec<AttemptKey>, ControlError> {
        if self.view.state != RunState::Pausing {
            return Err(self.invalid("pause deadline requires pausing"));
        }
        let mut revoked = Vec::new();
        for round in self.rounds.values_mut() {
            for pending in round.slots.values_mut().filter(|p| p.outcome.is_none()) {
                revoked.push(pending.request.key.clone());
                pending.outcome = Some(EvaluationOutcome::DiscardedAtPauseDeadline);
            }
        }
        Ok(revoked)
    }

    /// Call in the owning store transaction. Never publishes events itself.
    pub fn round_committed(&mut self) -> Result<CommittedRound, ControlError> {
        if !self.live() {
            return Err(self.invalid("round commit requires running or pausing"));
        }
        let round_id = self.view.next_round;
        let next = match self.rounds.get(&round_id) {
            Some(round) if round.slots.values().all(|p| p.outcome.is_some()) => round_id
                .checked_add(1)
                .ok_or_else(|| self.invalid("round counter exhausted"))?,
            None if self.view.state == RunState::Pausing && self.rounds.is_empty() => round_id,
            _ => return Err(self.invalid("next round still has outstanding evaluations")),
        };
        let revision = self.next_revision()?;
        let round = self.rounds.remove(&round_id).unwrap_or(Round {
            effort: self.view.effort,
            slots: BTreeMap::new(),
        });
        let committed = CommittedRound {
            round: round_id,
            effort: round.effort,
            evaluations: round
                .slots
                .into_values()
                .map(|p| Evaluation {
                    key: p.request.key,
                    outcome: p.outcome.unwrap(),
                })
                .collect(),
        };
        self.view.pending_slots -= committed.evaluations.len();
        self.view.next_round = next;
        self.checkpoint_phase = self.view.phase;
        self.view.revision = revision;
        if let Some(effort) = self.view.pending_effort.take() {
            self.view.effort = effort;
        }
        if self.view.state == RunState::Pausing && self.rounds.is_empty() {
            self.view.state = RunState::Paused;
        }
        if !committed.evaluations.is_empty() {
            self.committed.push(committed.clone());
        }
        Ok(committed)
    }

    pub fn complete(&mut self, reason: CompletionReason) -> Result<(), ControlError> {
        if self.view.state != RunState::Running
            || self.view.phase != RunPhase::Finalizing
            || !self.rounds.is_empty()
        {
            return Err(self.invalid("completion requires quiescent running finalization"));
        }
        let revision = self.next_revision()?;
        self.view.state = RunState::Completed;
        self.view.completion_reason = Some(reason);
        self.view.pending_effort = None;
        self.view.revision = revision;
        Ok(())
    }

    pub fn fail(&mut self, code: ErrorCode) -> Result<(), ControlError> {
        if !self.live() {
            return Err(self.invalid("failure requires running or pausing"));
        }
        let revision = self.next_revision()?;
        self.rounds.clear();
        self.view.pending_slots = 0;
        self.view.pending_effort = None;
        self.view.state = RunState::Failed;
        self.view.failure = Some(code);
        self.view.revision = revision;
        Ok(())
    }

    /// Revoke all uncommitted work without resetting the retained fencing counter.
    /// A disk owner must reload that counter; constructing a new controller is not recovery.
    pub fn recover_interrupted(&mut self) -> Result<bool, ControlError> {
        if !self.live() {
            return Ok(false);
        }
        let revision = self.next_revision()?;
        self.rounds.clear();
        self.view.pending_slots = 0;
        self.view.phase = self.checkpoint_phase;
        if let Some(effort) = self.view.pending_effort.take() {
            self.view.effort = effort;
        }
        self.view.state = RunState::Paused;
        self.view.interrupted = true;
        self.view.revision = revision;
        Ok(true)
    }
}
