//! Crash-clean daemon state backed by redb.
//!
//! State-changing methods commit before returning a receipt. Callers must not
//! perform the corresponding launch, signal, cgroup, or notification effect
//! until the method succeeds.

use std::path::{Path, PathBuf};

use jain_smartcluster_core::{
    AttemptLease, AttemptNumber, AttemptRecord, FailureClass, FenceToken, FencedAttempt,
    FencedTransition, IdempotencyScope, JobId, JobRecord, JobState, TerminalStatus, Timestamp,
    TransitionError, Validate,
};
use redb::{Database, Durability, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

mod support;
use support::*;

const JOBS: TableDefinition<&str, &[u8]> = TableDefinition::new("jobs");
const ATTEMPTS: TableDefinition<&str, &[u8]> = TableDefinition::new("attempts");
const IDEMPOTENCY: TableDefinition<&str, &[u8]> = TableDefinition::new("idempotency");
const RESOURCE_HISTORY: TableDefinition<&str, &[u8]> = TableDefinition::new("resource-history");
const TRANSITION_OUTBOX: TableDefinition<&str, &[u8]> = TableDefinition::new("transition-outbox");
const META: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");
const QUEUED_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("queued-index");
const DEADLINE_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("deadline-index");

const VALUE_FORMAT_VERSION: u16 = 1;
const NEXT_FENCE_KEY: &str = "__scq.next-fence";
const FIRST_FENCE: u64 = 1;

pub type StoreResult<T> = Result<T, StoreError>;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("store backend error: {0}")]
    Backend(String),
    #[error("unsupported stored-value format version {found}")]
    UnsupportedFormat { found: u16 },
    #[error("stored value is corrupt: {0}")]
    Corrupt(String),
    #[error("invalid durable record: {0}")]
    InvalidRecord(String),
    #[error("job {0} was not found")]
    JobNotFound(JobId),
    #[error("job {0} already exists")]
    JobAlreadyExists(JobId),
    #[error("attempt {attempt} for job {job_id} was not found")]
    AttemptNotFound { job_id: JobId, attempt: u32 },
    #[error("idempotency key already names job {existing_job} with a different specification")]
    IdempotencyConflict { existing_job: JobId },
    #[error("job {0} has no active attempt")]
    NoActiveAttempt(JobId),
    #[error("attempt counter exhausted for job {0}")]
    AttemptCounterExhausted(JobId),
    #[error("global attempt fence counter exhausted")]
    FenceCounterExhausted,
    #[error("invalid transition mutation: {0}")]
    InvalidTransition(String),
    #[error(transparent)]
    Transition(#[from] TransitionError),
    #[error("metadata key is invalid")]
    InvalidMetaKey,
}

trait BackendResultExt<T> {
    fn store(self) -> StoreResult<T>;
}

impl<T, E> BackendResultExt<T> for Result<T, E>
where
    E: std::error::Error,
{
    fn store(self) -> StoreResult<T> {
        self.map_err(|error| StoreError::Backend(error.to_string()))
    }
}

/// Durable idempotency binding. Identity is included to prevent cross-principal
/// discovery through a tenant-wide key collision.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IdempotencyRecord {
    pub scope: IdempotencyScope,
    pub job_id: JobId,
    pub spec_digest: [u8; 32],
    pub created_at: Timestamp,
}

/// Persisted GPU observation history used by the fail-safe share policy.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResourceHistoryRecord {
    pub fingerprint: String,
    pub total_clean_exclusive_runs: u64,
    pub consecutive_clean_exclusive_runs: u32,
    pub max_observed_vram_bytes: u64,
    pub sharing_disabled: bool,
    pub exclusive_retry_pending: bool,
    pub updated_at: Timestamp,
}

impl ResourceHistoryRecord {
    pub fn new(fingerprint: impl Into<String>, at: Timestamp) -> StoreResult<Self> {
        let fingerprint = fingerprint.into();
        validate_fingerprint(&fingerprint)?;
        Ok(Self {
            fingerprint,
            total_clean_exclusive_runs: 0,
            consecutive_clean_exclusive_runs: 0,
            max_observed_vram_bytes: 0,
            sharing_disabled: false,
            exclusive_retry_pending: false,
            updated_at: at,
        })
    }

    pub fn record_clean_exclusive(&mut self, observed_vram_bytes: u64, at: Timestamp) {
        self.total_clean_exclusive_runs = self.total_clean_exclusive_runs.saturating_add(1);
        self.consecutive_clean_exclusive_runs =
            self.consecutive_clean_exclusive_runs.saturating_add(1);
        self.max_observed_vram_bytes = self.max_observed_vram_bytes.max(observed_vram_bytes);
        self.exclusive_retry_pending = false;
        if self.consecutive_clean_exclusive_runs >= 5 {
            self.sharing_disabled = false;
        }
        self.updated_at = at;
    }

    pub fn record_failed_exclusive(&mut self, observed_vram_bytes: u64, at: Timestamp) {
        self.consecutive_clean_exclusive_runs = 0;
        self.max_observed_vram_bytes = self.max_observed_vram_bytes.max(observed_vram_bytes);
        self.exclusive_retry_pending = false;
        self.updated_at = at;
    }

    pub fn record_shared_cuda_oom(&mut self, observed_vram_bytes: u64, at: Timestamp) {
        self.consecutive_clean_exclusive_runs = 0;
        self.max_observed_vram_bytes = self.max_observed_vram_bytes.max(observed_vram_bytes);
        self.sharing_disabled = true;
        self.exclusive_retry_pending = true;
        self.updated_at = at;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SubmitOutcome {
    Created(JobRecord),
    Existing(JobRecord),
}

impl SubmitOutcome {
    pub fn job(&self) -> &JobRecord {
        match self {
            Self::Created(job) | Self::Existing(job) => job,
        }
    }
}

/// Receipt authorizing a launch only after both the starting job state and
/// fenced attempt lease are durable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchReceipt {
    pub job: JobRecord,
    pub attempt: AttemptRecord,
    pub transition: FencedTransition,
}

/// Inputs for one atomic, fence-checked state change.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionMutation {
    pub transition: FencedTransition,
    pub terminal: Option<TerminalStatus>,
    pub failure: Option<FailureClass>,
}

impl TransitionMutation {
    pub fn nonterminal(transition: FencedTransition) -> Self {
        Self {
            transition,
            terminal: None,
            failure: None,
        }
    }
}

/// Receipt returned only after the transition transaction has committed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionReceipt {
    pub job: JobRecord,
    pub attempt: Option<AttemptRecord>,
    pub transition: FencedTransition,
}

/// A state transition committed atomically with its job/attempt mutation and
/// awaiting publication to the durable event segments.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PendingTransitionEvent {
    pub transition: FencedTransition,
}

pub struct Store {
    database: Database,
    path: PathBuf,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Store")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl Store {
    include!("store/open.rs");
    include!("store/submit.rs");
    include!("store/reads.rs");
    include!("store/transitions.rs");
    include!("store/leases.rs");
    include!("store/metadata.rs");
}

#[cfg(test)]
mod tests;
