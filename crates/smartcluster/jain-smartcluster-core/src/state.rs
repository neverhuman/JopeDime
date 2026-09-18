use std::{fmt, num::NonZeroU32, num::NonZeroU64};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{FailureClass, JobId, JobRecord, Validate, ValidationError};

/// Milliseconds since the Unix epoch. Clock interpretation belongs to the daemon.
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(transparent)]
pub struct Timestamp(u64);

impl Timestamp {
    pub const fn from_unix_millis(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_unix_millis(self) -> u64 {
        self.0
    }
}

impl From<u64> for Timestamp {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// One-based execution-attempt number.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct AttemptNumber(NonZeroU32);

impl AttemptNumber {
    pub fn new(value: u32) -> Result<Self, ValidationError> {
        NonZeroU32::new(value)
            .map(Self)
            .ok_or_else(|| ValidationError::new("attempt.number", "must be greater than zero"))
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl TryFrom<u32> for AttemptNumber {
    type Error = ValidationError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl fmt::Display for AttemptNumber {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(formatter)
    }
}

/// Monotonically generated launch fence. A superseded worker cannot complete a newer attempt.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct FenceToken(NonZeroU64);

impl FenceToken {
    pub fn new(value: u64) -> Result<Self, ValidationError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or_else(|| ValidationError::new("attempt.fence", "must be greater than zero"))
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

impl TryFrom<u64> for FenceToken {
    type Error = ValidationError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl fmt::Display for FenceToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(formatter)
    }
}

/// Identity that must accompany every attempt-scoped mutation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FencedAttempt {
    pub job_id: JobId,
    pub number: AttemptNumber,
    pub fence: FenceToken,
}

impl FencedAttempt {
    pub const fn new(job_id: JobId, number: AttemptNumber, fence: FenceToken) -> Self {
        Self {
            job_id,
            number,
            fence,
        }
    }

    pub fn accepts(self, presented: Self) -> bool {
        self == presented
    }
}

/// Renewable ownership record for launching or adopting one fenced attempt.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AttemptLease {
    pub attempt: FencedAttempt,
    pub acquired_at: Timestamp,
    pub expires_at: Timestamp,
}

impl AttemptLease {
    pub fn validate_presented(&self, presented: FencedAttempt) -> Result<(), TransitionError> {
        if self.attempt == presented {
            Ok(())
        } else {
            Err(TransitionError::SupersededFence {
                expected: self.attempt,
                presented,
            })
        }
    }

    pub fn renew(
        &mut self,
        presented: FencedAttempt,
        expires_at: Timestamp,
    ) -> Result<(), TransitionError> {
        self.validate_presented(presented)?;
        if expires_at <= self.expires_at {
            return Err(TransitionError::LeaseDidNotAdvance);
        }
        self.expires_at = expires_at;
        Ok(())
    }
}

impl Validate for AttemptLease {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.expires_at <= self.acquired_at {
            return Err(ValidationError::new(
                "attempt.expires_at",
                "must be later than acquired_at",
            ));
        }
        Ok(())
    }
}

/// Durable lifecycle state for a job.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Admitted,
    Starting,
    Running,
    Paused,
    Suspended,
    RetryWait,
    Completed,
    Failed,
    Cancelled,
}

impl JobState {
    /// Explicit state-transition matrix. Terminal states have no successors.
    pub const fn allowed_successors(self) -> &'static [Self] {
        match self {
            Self::Queued => &[Self::Admitted, Self::Paused, Self::Failed, Self::Cancelled],
            Self::Admitted => &[Self::Queued, Self::Starting, Self::Failed, Self::Cancelled],
            Self::Starting => &[
                Self::Running,
                Self::RetryWait,
                Self::Failed,
                Self::Cancelled,
            ],
            Self::Running => &[
                Self::Paused,
                Self::Suspended,
                Self::RetryWait,
                Self::Completed,
                Self::Failed,
                Self::Cancelled,
            ],
            Self::Paused => &[
                Self::Queued,
                Self::Running,
                Self::Suspended,
                Self::RetryWait,
                Self::Failed,
                Self::Cancelled,
            ],
            Self::Suspended => &[Self::Queued, Self::RetryWait, Self::Failed, Self::Cancelled],
            Self::RetryWait => &[Self::Queued, Self::Failed, Self::Cancelled],
            Self::Completed | Self::Failed | Self::Cancelled => &[],
        }
    }

    pub fn can_transition_to(self, next: Self) -> bool {
        self.allowed_successors().contains(&next)
    }

    pub fn validate_transition(self, next: Self) -> Result<(), TransitionError> {
        if self.can_transition_to(next) {
            Ok(())
        } else {
            Err(TransitionError::IllegalTransition {
                from: self,
                to: next,
            })
        }
    }

    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// A transition record to persist before applying its external effect.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FencedTransition {
    pub job_id: JobId,
    pub from: JobState,
    pub to: JobState,
    pub attempt: Option<FencedAttempt>,
    pub at: Timestamp,
    pub reason: Option<String>,
}

impl FencedTransition {
    pub fn validate_against(&self, job: &JobRecord) -> Result<(), TransitionError> {
        if self.job_id != job.id {
            return Err(TransitionError::WrongJob {
                expected: job.id,
                presented: self.job_id,
            });
        }
        if self.from != job.state {
            return Err(TransitionError::StateMismatch {
                expected: job.state,
                presented: self.from,
            });
        }
        self.from.validate_transition(self.to)?;
        match (job.active_attempt, self.attempt) {
            (Some(expected), Some(presented)) if expected == presented => Ok(()),
            (Some(expected), Some(presented)) => Err(TransitionError::SupersededFence {
                expected,
                presented,
            }),
            (Some(_), None) => Err(TransitionError::MissingFence),
            (None, Some(_)) => Err(TransitionError::UnexpectedFence),
            (None, None) => Ok(()),
        }
    }
}

/// Durable information for an individual launch attempt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AttemptRecord {
    pub attempt: FencedAttempt,
    pub state: JobState,
    pub lease: AttemptLease,
    pub pid: Option<u32>,
    pub cgroup_path: Option<String>,
    pub started_at: Option<Timestamp>,
    pub finished_at: Option<Timestamp>,
    pub failure: Option<FailureClass>,
}

impl Validate for AttemptRecord {
    fn validate(&self) -> Result<(), ValidationError> {
        self.lease.validate()?;
        if self.lease.attempt != self.attempt {
            return Err(ValidationError::new(
                "attempt.lease",
                "must fence the same attempt",
            ));
        }
        if let (Some(started), Some(finished)) = (self.started_at, self.finished_at) {
            if finished < started {
                return Err(ValidationError::new(
                    "attempt.finished_at",
                    "must not precede started_at",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum TransitionError {
    #[error("illegal state transition from {from:?} to {to:?}")]
    IllegalTransition { from: JobState, to: JobState },
    #[error("state mismatch: expected {expected:?}, got {presented:?}")]
    StateMismatch {
        expected: JobState,
        presented: JobState,
    },
    #[error("wrong job: expected {expected}, got {presented}")]
    WrongJob { expected: JobId, presented: JobId },
    #[error("attempt-scoped transition is missing a fence")]
    MissingFence,
    #[error("transition unexpectedly carries an attempt fence")]
    UnexpectedFence,
    #[error("superseded attempt fence: expected {expected:?}, got {presented:?}")]
    SupersededFence {
        expected: FencedAttempt,
        presented: FencedAttempt,
    },
    #[error("lease renewal must advance the expiry")]
    LeaseDidNotAdvance,
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATES: [JobState; 10] = [
        JobState::Queued,
        JobState::Admitted,
        JobState::Starting,
        JobState::Running,
        JobState::Paused,
        JobState::Suspended,
        JobState::RetryWait,
        JobState::Completed,
        JobState::Failed,
        JobState::Cancelled,
    ];

    #[test]
    fn transition_matrix_is_explicit_and_has_no_self_transitions() {
        for from in STATES {
            for to in STATES {
                assert_eq!(
                    from.can_transition_to(to),
                    from.allowed_successors().contains(&to)
                );
                if from == to {
                    assert!(!from.can_transition_to(to));
                }
            }
        }
    }

    #[test]
    fn terminal_states_never_transition() {
        for terminal in [JobState::Completed, JobState::Failed, JobState::Cancelled] {
            assert!(terminal.is_terminal());
            assert!(terminal.allowed_successors().is_empty());
        }
    }

    #[test]
    fn superseded_attempt_cannot_renew_lease() {
        let job_id = JobId::new();
        let current = FencedAttempt::new(
            job_id,
            AttemptNumber::new(2).unwrap(),
            FenceToken::new(20).unwrap(),
        );
        let superseded = FencedAttempt::new(
            job_id,
            AttemptNumber::new(1).unwrap(),
            FenceToken::new(10).unwrap(),
        );
        let mut lease = AttemptLease {
            attempt: current,
            acquired_at: 100.into(),
            expires_at: 200.into(),
        };
        assert!(matches!(
            lease.renew(superseded, 300.into()),
            Err(TransitionError::SupersededFence { .. })
        ));
        assert_eq!(lease.expires_at, 200.into());
    }
}
