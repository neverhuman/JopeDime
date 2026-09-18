use std::{collections::BTreeMap, collections::BTreeSet, fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{
    FencedAttempt, IdempotencyKey, JobState, Principal, ProjectId, ResourceRequest, TenantId,
    Timestamp, Validate, ValidationError, Workload,
};

/// Stable, time-sortable job identifier.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct JobId(Ulid);

impl JobId {
    pub fn new() -> Self {
        Self(Ulid::new())
    }

    pub const fn from_ulid(value: Ulid) -> Self {
        Self(value)
    }

    pub const fn as_ulid(self) -> Ulid {
        self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for JobId {
    type Err = ulid::DecodeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        value.parse().map(Self)
    }
}

/// Strict scheduler class ordered from highest to lowest value.
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PriorityClass {
    System,
    Interactive,
    #[default]
    Batch,
    Opportunistic,
}

impl PriorityClass {
    pub const fn rank(self) -> u8 {
        match self {
            Self::System => 0,
            Self::Interactive => 1,
            Self::Batch => 2,
            Self::Opportunistic => 3,
        }
    }
}

/// Stable failure taxonomy used by retry and accounting policy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Infrastructure,
    GpuTransient,
    Network,
    HostOutOfMemory,
    SharedCudaOutOfMemory,
    Input,
    Dependency,
    Application,
    DeadlineExceeded,
    Preempted,
    DaemonRestart,
}

impl FailureClass {
    pub const fn is_retryable_by_default(self) -> bool {
        matches!(
            self,
            Self::Infrastructure
                | Self::GpuTransient
                | Self::Network
                | Self::HostOutOfMemory
                | Self::SharedCudaOutOfMemory
                | Self::Preempted
                | Self::DaemonRestart
        )
    }

    pub const fn consumes_failure_budget(self) -> bool {
        !matches!(self, Self::Preempted | Self::DaemonRestart)
    }
}

/// Failure-budget and capped exponential-backoff configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RetryPolicy {
    pub max_failures: u32,
    pub initial_backoff_ms: u64,
    pub maximum_backoff_ms: u64,
    pub jitter_per_mille: u16,
    pub retry_on: BTreeSet<FailureClass>,
}

const MAX_RETRY_FAILURES: u32 = 100;
const MIN_RETRY_BACKOFF_MS: u64 = 10;
const MAX_RETRY_BACKOFF_MS: u64 = 24 * 60 * 60 * 1_000;

impl RetryPolicy {
    pub fn allows(&self, failure: FailureClass) -> bool {
        self.retry_on.contains(&failure)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_failures: 3,
            initial_backoff_ms: 500,
            maximum_backoff_ms: 30_000,
            jitter_per_mille: 200,
            retry_on: [
                FailureClass::Infrastructure,
                FailureClass::GpuTransient,
                FailureClass::Network,
                FailureClass::HostOutOfMemory,
                FailureClass::SharedCudaOutOfMemory,
                FailureClass::Preempted,
                FailureClass::DaemonRestart,
            ]
            .into_iter()
            .collect(),
        }
    }
}

impl Validate for RetryPolicy {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.max_failures > MAX_RETRY_FAILURES {
            return Err(ValidationError::new(
                "retry.max_failures",
                format!("must not exceed {MAX_RETRY_FAILURES}"),
            ));
        }
        if self.max_failures > 0 && self.initial_backoff_ms < MIN_RETRY_BACKOFF_MS {
            return Err(ValidationError::new(
                "retry.initial_backoff_ms",
                format!("must be at least {MIN_RETRY_BACKOFF_MS} when retries are enabled"),
            ));
        }
        if self.initial_backoff_ms > self.maximum_backoff_ms {
            return Err(ValidationError::new(
                "retry.initial_backoff_ms",
                "must not exceed maximum_backoff_ms",
            ));
        }
        if self.maximum_backoff_ms > MAX_RETRY_BACKOFF_MS {
            return Err(ValidationError::new(
                "retry.maximum_backoff_ms",
                format!("must not exceed {MAX_RETRY_BACKOFF_MS}"),
            ));
        }
        if self.jitter_per_mille > 1000 {
            return Err(ValidationError::new(
                "retry.jitter_per_mille",
                "must not exceed 1000",
            ));
        }
        if let Some(failure) = self
            .retry_on
            .iter()
            .find(|failure| !failure.is_retryable_by_default())
        {
            return Err(ValidationError::new(
                "retry.retry_on",
                format!("{failure:?} is a deterministic terminal failure"),
            ));
        }
        Ok(())
    }
}

/// How a compatible worker is asked to establish a resumable boundary.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointPolicy {
    None,
    ControlFd { deadline_ms: u64 },
    Signal { signal: i32, grace_ms: u64 },
}

/// Whether and how the scheduler may reclaim this workload.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PreemptionPolicy {
    pub preemptible: bool,
    pub checkpoint: CheckpointPolicy,
}

impl PreemptionPolicy {
    pub const fn non_preemptible() -> Self {
        Self {
            preemptible: false,
            checkpoint: CheckpointPolicy::None,
        }
    }
}

impl Default for PreemptionPolicy {
    fn default() -> Self {
        Self::non_preemptible()
    }
}

impl Validate for PreemptionPolicy {
    fn validate(&self) -> Result<(), ValidationError> {
        if !self.preemptible && !matches!(self.checkpoint, CheckpointPolicy::None) {
            return Err(ValidationError::new(
                "preemption.checkpoint",
                "non-preemptible workloads cannot configure checkpointing",
            ));
        }
        match self.checkpoint {
            CheckpointPolicy::ControlFd { deadline_ms: 0 } => Err(ValidationError::new(
                "preemption.checkpoint.deadline_ms",
                "must be nonzero",
            )),
            CheckpointPolicy::Signal { grace_ms: 0, .. } => Err(ValidationError::new(
                "preemption.checkpoint.grace_ms",
                "must be nonzero",
            )),
            _ => Ok(()),
        }
    }
}

/// Child I/O attachment mode.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IoMode {
    #[default]
    Pipes,
    Pty {
        rows: u16,
        columns: u16,
    },
}

impl Validate for IoMode {
    fn validate(&self) -> Result<(), ValidationError> {
        if let Self::Pty { rows, columns } = self {
            if *rows == 0 || *columns == 0 {
                return Err(ValidationError::new(
                    "io",
                    "PTY rows and columns must be nonzero",
                ));
            }
        }
        Ok(())
    }
}

/// Complete public submission contract.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct JobSpec {
    pub tenant: TenantId,
    pub project: ProjectId,
    pub idempotency_key: IdempotencyKey,
    pub priority: PriorityClass,
    pub workload: Workload,
    pub resources: ResourceRequest,
    #[serde(default)]
    pub retry: RetryPolicy,
    #[serde(default)]
    pub preemption: PreemptionPolicy,
    #[serde(default)]
    pub io: IoMode,
    pub deadline: Option<Timestamp>,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

/// Uniqueness boundary for at-least-once submission deduplication.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct IdempotencyScope {
    pub principal: Principal,
    pub tenant: TenantId,
    pub project: ProjectId,
    pub key: IdempotencyKey,
}

impl IdempotencyScope {
    pub fn for_submission(principal: Principal, spec: &JobSpec) -> Self {
        Self {
            principal,
            tenant: spec.tenant.clone(),
            project: spec.project.clone(),
            key: spec.idempotency_key.clone(),
        }
    }
}

impl Validate for IdempotencyScope {
    fn validate(&self) -> Result<(), ValidationError> {
        self.principal.validate()?;
        self.tenant.validate()?;
        self.project.validate()?;
        self.key.validate()
    }
}

impl JobSpec {
    pub fn validate_for(&self, principal: &Principal) -> Result<(), ValidationError> {
        self.validate()?;
        self.workload.validate_for(principal)
    }
}

impl Validate for JobSpec {
    fn validate(&self) -> Result<(), ValidationError> {
        self.tenant.validate()?;
        self.project.validate()?;
        self.idempotency_key.validate()?;
        self.workload
            .validate()
            .map_err(|error| error.prefixed("spec"))?;
        self.resources
            .validate()
            .map_err(|error| error.prefixed("spec"))?;
        self.retry
            .validate()
            .map_err(|error| error.prefixed("spec"))?;
        self.preemption
            .validate()
            .map_err(|error| error.prefixed("spec"))?;
        self.io.validate().map_err(|error| error.prefixed("spec"))?;
        validate_labels(&self.labels)
    }
}

fn validate_labels(labels: &BTreeMap<String, String>) -> Result<(), ValidationError> {
    if labels.len() > 64 {
        return Err(ValidationError::new(
            "labels",
            "must contain at most 64 entries",
        ));
    }
    for (key, value) in labels {
        crate::validation::validate_identifier(&format!("labels.{key}"), key, 128)?;
        crate::validation::validate_text(&format!("labels.{key}"), value, 1024, true)?;
    }
    Ok(())
}

/// State-specific terminal outcome.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalStatus {
    Completed {
        exit_code: i32,
        at: Timestamp,
    },
    Failed {
        class: FailureClass,
        message: String,
        exit_code: Option<i32>,
        at: Timestamp,
    },
    Cancelled {
        message: Option<String>,
        at: Timestamp,
    },
}

impl TerminalStatus {
    pub const fn state(&self) -> JobState {
        match self {
            Self::Completed { .. } => JobState::Completed,
            Self::Failed { .. } => JobState::Failed,
            Self::Cancelled { .. } => JobState::Cancelled,
        }
    }
}

/// Durable externally visible job record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct JobRecord {
    pub id: JobId,
    pub spec: JobSpec,
    pub submitted_by: Principal,
    pub state: JobState,
    pub active_attempt: Option<FencedAttempt>,
    pub attempt_count: u32,
    pub failure_count: u32,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub terminal: Option<TerminalStatus>,
}

impl JobRecord {
    pub fn queued(id: JobId, spec: JobSpec, submitted_by: Principal, at: Timestamp) -> Self {
        Self {
            id,
            spec,
            submitted_by,
            state: JobState::Queued,
            active_attempt: None,
            attempt_count: 0,
            failure_count: 0,
            created_at: at,
            updated_at: at,
            terminal: None,
        }
    }
}

impl Validate for JobRecord {
    fn validate(&self) -> Result<(), ValidationError> {
        self.spec.validate_for(&self.submitted_by)?;
        self.submitted_by.validate()?;
        if self.updated_at < self.created_at {
            return Err(ValidationError::new(
                "job.updated_at",
                "must not precede created_at",
            ));
        }
        if let Some(attempt) = self.active_attempt {
            if attempt.job_id != self.id {
                return Err(ValidationError::new(
                    "job.active_attempt",
                    "belongs to a different job",
                ));
            }
            if attempt.number.get() > self.attempt_count {
                return Err(ValidationError::new(
                    "job.attempt_count",
                    "must include the active attempt",
                ));
            }
        }
        match (&self.terminal, self.state.is_terminal()) {
            (Some(terminal), true) if terminal.state() == self.state => Ok(()),
            (Some(_), true) => Err(ValidationError::new(
                "job.terminal",
                "does not match the terminal state",
            )),
            (None, true) => Err(ValidationError::new(
                "job.terminal",
                "is required for a terminal state",
            )),
            (Some(_), false) => Err(ValidationError::new(
                "job.terminal",
                "must be absent for a nonterminal state",
            )),
            (None, false) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuRequest, Environment, JainWorkload, MemoryRequest};

    fn valid_spec() -> JobSpec {
        JobSpec {
            tenant: TenantId::new("tenant-a").unwrap(),
            project: ProjectId::new("project-a").unwrap(),
            idempotency_key: IdempotencyKey::new("request-1").unwrap(),
            priority: PriorityClass::Batch,
            workload: Workload::Jain {
                template: JainWorkload::Training,
                arguments: Vec::new(),
                environment: Environment::default(),
            },
            resources: ResourceRequest {
                cpu: CpuRequest::fixed(1000),
                ram: MemoryRequest::fixed(1024),
                scratch_bytes: 0,
                gpu: None,
            },
            retry: RetryPolicy::default(),
            preemption: PreemptionPolicy::default(),
            io: IoMode::Pipes,
            deadline: None,
            labels: BTreeMap::new(),
        }
    }

    #[test]
    fn job_spec_validates_deeply() {
        assert!(valid_spec().validate().is_ok());
        let mut invalid = valid_spec();
        invalid.resources.ram.hard_bytes = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn nonterminal_record_cannot_carry_terminal_result() {
        let mut job =
            JobRecord::queued(JobId::new(), valid_spec(), Principal::unix(1000), 1.into());
        job.terminal = Some(TerminalStatus::Completed {
            exit_code: 0,
            at: 2.into(),
        });
        assert!(job.validate().is_err());
    }

    #[test]
    fn preemption_and_restart_do_not_consume_failure_budget() {
        assert!(!FailureClass::Preempted.consumes_failure_budget());
        assert!(!FailureClass::DaemonRestart.consumes_failure_budget());
        assert!(FailureClass::Infrastructure.consumes_failure_budget());
    }

    #[test]
    fn retry_policy_rejects_unbounded_or_immediate_storms() {
        let mut policy = RetryPolicy {
            max_failures: MAX_RETRY_FAILURES + 1,
            ..RetryPolicy::default()
        };
        assert!(policy.validate().is_err());

        policy = RetryPolicy {
            initial_backoff_ms: 0,
            ..RetryPolicy::default()
        };
        assert!(policy.validate().is_err());

        policy = RetryPolicy::default();
        policy.retry_on.insert(FailureClass::Input);
        assert!(policy.validate().is_err());

        policy = RetryPolicy {
            max_failures: 0,
            initial_backoff_ms: 0,
            maximum_backoff_ms: 0,
            jitter_per_mille: 0,
            retry_on: BTreeSet::new(),
        };
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn idempotency_is_scoped_by_effective_principal_and_project() {
        let spec = valid_spec();
        let alice = IdempotencyScope::for_submission(Principal::web("alice").unwrap(), &spec);
        let repeated = IdempotencyScope::for_submission(Principal::web("alice").unwrap(), &spec);
        let bob = IdempotencyScope::for_submission(Principal::web("bob").unwrap(), &spec);
        assert_eq!(alice, repeated);
        assert_ne!(alice, bob);
    }
}
