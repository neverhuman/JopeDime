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
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).store()?;
        }
        let database = Database::create(&path).store()?;
        let mut transaction = database.begin_write().store()?;
        transaction.set_durability(Durability::Immediate);
        {
            let _table = transaction.open_table(JOBS).store()?;
        }
        {
            let _table = transaction.open_table(ATTEMPTS).store()?;
        }
        {
            let _table = transaction.open_table(IDEMPOTENCY).store()?;
        }
        {
            let _table = transaction.open_table(RESOURCE_HISTORY).store()?;
        }
        {
            let _table = transaction.open_table(TRANSITION_OUTBOX).store()?;
        }
        {
            let _table = transaction.open_table(META).store()?;
        }
        {
            let _table = transaction.open_table(QUEUED_INDEX).store()?;
        }
        {
            let _table = transaction.open_table(DEADLINE_INDEX).store()?;
        }
        let jobs = {
            let table = transaction.open_table(JOBS).store()?;
            let mut jobs = Vec::with_capacity(table.len().store()? as usize);
            for row in table.iter().store()? {
                let (_key, value) = row.store()?;
                jobs.push(decode::<JobRecord>(value.value())?);
            }
            jobs
        };
        rebuild_indexes(&transaction, &jobs)?;
        transaction.commit().store()?;
        Ok(Self { database, path })
    }

    /// Atomically create a queued job and its idempotency binding.
    pub fn submit(&self, job: &JobRecord) -> StoreResult<SubmitOutcome> {
        job.validate()
            .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;
        if job.state != JobState::Queued
            || job.active_attempt.is_some()
            || job.attempt_count != 0
            || job.terminal.is_some()
        {
            return Err(StoreError::InvalidRecord(
                "new jobs must be pristine queued records".into(),
            ));
        }

        let digest = digest(&job.spec)?;
        let scope = IdempotencyScope::for_submission(job.submitted_by.clone(), &job.spec);
        let idempotency_key = idempotency_table_key(&scope)?;
        let job_key = job.id.to_string();
        let transaction = self.database.begin_write().store()?;
        let existing_binding = {
            let table = transaction.open_table(IDEMPOTENCY).store()?;
            let binding = table
                .get(idempotency_key.as_str())
                .store()?
                .map(|value| decode::<IdempotencyRecord>(value.value()))
                .transpose()?;
            binding
        };

        if let Some(binding) = existing_binding {
            if binding.scope != scope {
                return Err(StoreError::Corrupt(
                    "idempotency index key does not match its stored scope".into(),
                ));
            }
            if binding.spec_digest != digest {
                return Err(StoreError::IdempotencyConflict {
                    existing_job: binding.job_id,
                });
            }
            let existing = {
                let table = transaction.open_table(JOBS).store()?;
                let binding_key = binding.job_id.to_string();
                let job = table
                    .get(binding_key.as_str())
                    .store()?
                    .map(|value| decode::<JobRecord>(value.value()))
                    .transpose()?
                    .ok_or(StoreError::Corrupt(
                        "idempotency binding references a missing job".into(),
                    ))?;
                job
            };
            transaction.abort().store()?;
            return Ok(SubmitOutcome::Existing(existing));
        }

        {
            let mut table = transaction.open_table(JOBS).store()?;
            if table.get(job_key.as_str()).store()?.is_some() {
                return Err(StoreError::JobAlreadyExists(job.id));
            }
            let bytes = encode(job)?;
            table.insert(job_key.as_str(), bytes.as_slice()).store()?;
        }
        {
            let binding = IdempotencyRecord {
                scope,
                job_id: job.id,
                spec_digest: digest,
                created_at: job.created_at,
            };
            let bytes = encode(&binding)?;
            let mut table = transaction.open_table(IDEMPOTENCY).store()?;
            table
                .insert(idempotency_key.as_str(), bytes.as_slice())
                .store()?;
        }
        index_job(&transaction, job)?;
        transaction.commit().store()?;
        Ok(SubmitOutcome::Created(job.clone()))
    }

    pub fn get_job(&self, job_id: JobId) -> StoreResult<Option<JobRecord>> {
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(JOBS).store()?;
        let key = job_id.to_string();
        table
            .get(key.as_str())
            .store()?
            .map(|value| decode(value.value()))
            .transpose()
    }

    pub fn all_jobs(&self) -> StoreResult<Vec<JobRecord>> {
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(JOBS).store()?;
        let mut jobs = Vec::with_capacity(table.len().store()? as usize);
        for row in table.iter().store()? {
            let (_key, value) = row.store()?;
            jobs.push(decode(value.value())?);
        }
        jobs.sort_by_key(|job: &JobRecord| (job.created_at, job.id));
        Ok(jobs)
    }

    /// Return queued jobs in scheduler order without scanning terminal or
    /// running records. The index is maintained in the same transaction as
    /// every job transition.
    pub fn queued_jobs(&self, limit: usize) -> StoreResult<Vec<JobRecord>> {
        let transaction = self.database.begin_read().store()?;
        let index = transaction.open_table(QUEUED_INDEX).store()?;
        let jobs = transaction.open_table(JOBS).store()?;
        let mut result = Vec::new();
        for row in index.iter().store()?.take(limit) {
            let (_key, value) = row.store()?;
            let job_id = indexed_job_id(value.value())?;
            let key = job_id.to_string();
            let Some(job) = jobs
                .get(key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
            else {
                return Err(StoreError::Corrupt(
                    "queued index references a missing job".to_owned(),
                ));
            };
            if job.state != JobState::Queued {
                return Err(StoreError::Corrupt(
                    "queued index references a non-queued job".to_owned(),
                ));
            }
            result.push(job);
        }
        Ok(result)
    }

    /// Return queued/admitted jobs whose indexed deadline has expired.
    pub fn expired_deadline_jobs(
        &self,
        now: Timestamp,
        limit: usize,
    ) -> StoreResult<Vec<JobRecord>> {
        let transaction = self.database.begin_read().store()?;
        let index = transaction.open_table(DEADLINE_INDEX).store()?;
        let jobs = transaction.open_table(JOBS).store()?;
        let end = deadline_index_key(now, "~");
        let mut result = Vec::new();
        for row in index.range(..=end.as_str()).store()?.take(limit) {
            let (_key, value) = row.store()?;
            let job_id = indexed_job_id(value.value())?;
            let key = job_id.to_string();
            let Some(job) = jobs
                .get(key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
            else {
                return Err(StoreError::Corrupt(
                    "deadline index references a missing job".to_owned(),
                ));
            };
            if matches!(job.state, JobState::Queued | JobState::Admitted)
                && job.spec.deadline.is_some_and(|deadline| deadline <= now)
            {
                result.push(job);
            }
        }
        Ok(result)
    }

    pub fn get_attempt(&self, attempt: FencedAttempt) -> StoreResult<Option<AttemptRecord>> {
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(ATTEMPTS).store()?;
        let key = attempt_table_key(attempt.job_id, attempt.number);
        let record = table
            .get(key.as_str())
            .store()?
            .map(|value| decode::<AttemptRecord>(value.value()))
            .transpose()?;
        match record {
            Some(record) if record.attempt == attempt => Ok(Some(record)),
            Some(_) => Err(StoreError::Corrupt(
                "attempt key and stored fence disagree".into(),
            )),
            None => Ok(None),
        }
    }

    pub fn attempts_for_job(&self, job_id: JobId) -> StoreResult<Vec<AttemptRecord>> {
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(ATTEMPTS).store()?;
        let prefix = format!("{job_id}/");
        let mut attempts = Vec::new();
        for row in table.iter().store()? {
            let (key, value) = row.store()?;
            if key.value().starts_with(&prefix) {
                attempts.push(decode(value.value())?);
            }
        }
        attempts.sort_by_key(|record: &AttemptRecord| record.attempt.number);
        Ok(attempts)
    }

    /// Atomically allocate a never-reused fence and persist `Starting` before a
    /// process may be launched.
    pub fn begin_attempt_before_launch(
        &self,
        job_id: JobId,
        acquired_at: Timestamp,
        expires_at: Timestamp,
    ) -> StoreResult<LaunchReceipt> {
        if expires_at <= acquired_at {
            return Err(StoreError::InvalidRecord(
                "attempt lease expiry must be later than acquisition".into(),
            ));
        }
        let transaction = self.database.begin_write().store()?;
        let job_key = job_id.to_string();
        let mut job = {
            let table = transaction.open_table(JOBS).store()?;
            let job = table
                .get(job_key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::JobNotFound(job_id))?;
            job
        };
        let previous_job = job.clone();
        if job.state != JobState::Admitted {
            return Err(StoreError::InvalidTransition(format!(
                "job must be admitted before launch, found {:?}",
                job.state
            )));
        }
        if job.active_attempt.is_some() {
            return Err(StoreError::InvalidTransition(
                "admitted job already has an active attempt".into(),
            ));
        }
        if acquired_at < job.updated_at {
            return Err(StoreError::InvalidRecord(
                "attempt acquisition timestamp precedes the job's last transition".into(),
            ));
        }

        let number_value = job
            .attempt_count
            .checked_add(1)
            .ok_or(StoreError::AttemptCounterExhausted(job_id))?;
        let number = AttemptNumber::new(number_value)
            .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;
        let fence_value = {
            let mut table = transaction.open_table(META).store()?;
            let current = table
                .get(NEXT_FENCE_KEY)
                .store()?
                .map(|value| decode::<u64>(value.value()))
                .transpose()?
                .unwrap_or(FIRST_FENCE);
            let next = current
                .checked_add(1)
                .ok_or(StoreError::FenceCounterExhausted)?;
            let bytes = encode(&next)?;
            table.insert(NEXT_FENCE_KEY, bytes.as_slice()).store()?;
            current
        };
        let fence = FenceToken::new(fence_value)
            .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;
        let fenced = FencedAttempt::new(job_id, number, fence);
        let lease = AttemptLease {
            attempt: fenced,
            acquired_at,
            expires_at,
        };
        let attempt = AttemptRecord {
            attempt: fenced,
            state: JobState::Starting,
            lease,
            pid: None,
            cgroup_path: None,
            started_at: None,
            finished_at: None,
            failure: None,
        };
        attempt
            .validate()
            .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;

        let transition = FencedTransition {
            job_id,
            from: JobState::Admitted,
            to: JobState::Starting,
            attempt: Some(fenced),
            at: acquired_at,
            reason: Some("fenced attempt lease committed before launch".into()),
        };
        job.state = JobState::Starting;
        job.active_attempt = Some(fenced);
        job.attempt_count = number_value;
        job.updated_at = acquired_at;
        {
            let bytes = encode(&job)?;
            let mut table = transaction.open_table(JOBS).store()?;
            table.insert(job_key.as_str(), bytes.as_slice()).store()?;
        }
        {
            let key = attempt_table_key(job_id, number);
            let bytes = encode(&attempt)?;
            let mut table = transaction.open_table(ATTEMPTS).store()?;
            table.insert(key.as_str(), bytes.as_slice()).store()?;
        }
        update_indexes(&transaction, &previous_job, &job)?;
        insert_pending_transition(&transaction, &transition)?;
        transaction.commit().store()?;
        Ok(LaunchReceipt {
            job,
            attempt,
            transition,
        })
    }

    /// Persist a fence-checked transition before its external effect.
    pub fn transition_before_effect(
        &self,
        mutation: TransitionMutation,
    ) -> StoreResult<TransitionReceipt> {
        validate_transition_mutation(&mutation)?;
        let transaction = self.database.begin_write().store()?;
        let job_key = mutation.transition.job_id.to_string();
        let mut job = {
            let table = transaction.open_table(JOBS).store()?;
            let job = table
                .get(job_key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::JobNotFound(mutation.transition.job_id))?;
            job
        };
        let previous_job = job.clone();
        mutation.transition.validate_against(&job)?;
        if mutation.transition.at < job.updated_at {
            return Err(StoreError::InvalidTransition(
                "transition timestamp precedes the current durable state".into(),
            ));
        }

        let mut attempt = match mutation.transition.attempt {
            Some(fenced) => {
                let key = attempt_table_key(fenced.job_id, fenced.number);
                let table = transaction.open_table(ATTEMPTS).store()?;
                let record = table
                    .get(key.as_str())
                    .store()?
                    .map(|value| decode::<AttemptRecord>(value.value()))
                    .transpose()?
                    .ok_or(StoreError::AttemptNotFound {
                        job_id: fenced.job_id,
                        attempt: fenced.number.get(),
                    })?;
                if record.attempt != fenced {
                    return Err(StoreError::Transition(TransitionError::SupersededFence {
                        expected: record.attempt,
                        presented: fenced,
                    }));
                }
                if record.state != mutation.transition.from {
                    return Err(StoreError::InvalidTransition(format!(
                        "attempt state {:?} does not match transition source {:?}",
                        record.state, mutation.transition.from
                    )));
                }
                Some(record)
            }
            None => None,
        };

        job.state = mutation.transition.to;
        job.updated_at = mutation.transition.at;
        job.terminal = mutation.terminal.clone();
        if mutation
            .failure
            .is_some_and(FailureClass::consumes_failure_budget)
        {
            job.failure_count = job.failure_count.saturating_add(1);
        }
        if let Some(record) = attempt.as_mut() {
            record.state = mutation.transition.to;
            if mutation.transition.to == JobState::Running && record.started_at.is_none() {
                record.started_at = Some(mutation.transition.at);
            }
            record.failure = mutation.failure;
            if retires_attempt(mutation.transition.to) {
                record.finished_at = Some(mutation.transition.at);
                job.active_attempt = None;
            }
        }
        job.validate()
            .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;

        {
            let bytes = encode(&job)?;
            let mut table = transaction.open_table(JOBS).store()?;
            table.insert(job_key.as_str(), bytes.as_slice()).store()?;
        }
        if let Some(record) = &attempt {
            let key = attempt_table_key(record.attempt.job_id, record.attempt.number);
            let bytes = encode(record)?;
            let mut table = transaction.open_table(ATTEMPTS).store()?;
            table.insert(key.as_str(), bytes.as_slice()).store()?;
        }
        update_indexes(&transaction, &previous_job, &job)?;
        insert_pending_transition(&transaction, &mutation.transition)?;
        transaction.commit().store()?;
        Ok(TransitionReceipt {
            job,
            attempt,
            transition: mutation.transition,
        })
    }

    pub fn renew_lease(
        &self,
        presented: FencedAttempt,
        expires_at: Timestamp,
    ) -> StoreResult<AttemptRecord> {
        let transaction = self.database.begin_write().store()?;
        let job_key = presented.job_id.to_string();
        let job = {
            let table = transaction.open_table(JOBS).store()?;
            let job = table
                .get(job_key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::JobNotFound(presented.job_id))?;
            job
        };
        let expected = job
            .active_attempt
            .ok_or(StoreError::NoActiveAttempt(presented.job_id))?;
        if expected != presented {
            return Err(StoreError::Transition(TransitionError::SupersededFence {
                expected,
                presented,
            }));
        }
        let key = attempt_table_key(presented.job_id, presented.number);
        let mut record = {
            let table = transaction.open_table(ATTEMPTS).store()?;
            let record = table
                .get(key.as_str())
                .store()?
                .map(|value| decode::<AttemptRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::AttemptNotFound {
                    job_id: presented.job_id,
                    attempt: presented.number.get(),
                })?;
            record
        };
        record.lease.renew(presented, expires_at)?;
        {
            let bytes = encode(&record)?;
            let mut table = transaction.open_table(ATTEMPTS).store()?;
            table.insert(key.as_str(), bytes.as_slice()).store()?;
        }
        transaction.commit().store()?;
        Ok(record)
    }

    pub fn record_spawned(
        &self,
        presented: FencedAttempt,
        pid: u32,
        cgroup_path: String,
    ) -> StoreResult<AttemptRecord> {
        if pid == 0 {
            return Err(StoreError::InvalidRecord(
                "spawned process id must be greater than zero".into(),
            ));
        }
        if cgroup_path.is_empty() {
            return Err(StoreError::InvalidRecord("cgroup path is empty".into()));
        }
        let transaction = self.database.begin_write().store()?;
        let job_key = presented.job_id.to_string();
        let job = {
            let table = transaction.open_table(JOBS).store()?;
            let job = table
                .get(job_key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::JobNotFound(presented.job_id))?;
            job
        };
        let expected = job
            .active_attempt
            .ok_or(StoreError::NoActiveAttempt(presented.job_id))?;
        if expected != presented {
            return Err(StoreError::Transition(TransitionError::SupersededFence {
                expected,
                presented,
            }));
        }
        let key = attempt_table_key(presented.job_id, presented.number);
        let mut record = {
            let table = transaction.open_table(ATTEMPTS).store()?;
            let record = table
                .get(key.as_str())
                .store()?
                .map(|value| decode::<AttemptRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::AttemptNotFound {
                    job_id: presented.job_id,
                    attempt: presented.number.get(),
                })?;
            record
        };
        if record.attempt != presented {
            return Err(StoreError::Transition(TransitionError::SupersededFence {
                expected: record.attempt,
                presented,
            }));
        }
        record.pid = Some(pid);
        record.cgroup_path = Some(cgroup_path);
        {
            let bytes = encode(&record)?;
            let mut table = transaction.open_table(ATTEMPTS).store()?;
            table.insert(key.as_str(), bytes.as_slice()).store()?;
        }
        transaction.commit().store()?;
        Ok(record)
    }

    pub fn get_resource_history(
        &self,
        fingerprint: &str,
    ) -> StoreResult<Option<ResourceHistoryRecord>> {
        validate_fingerprint(fingerprint)?;
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(RESOURCE_HISTORY).store()?;
        table
            .get(fingerprint)
            .store()?
            .map(|value| decode(value.value()))
            .transpose()
    }

    pub fn update_resource_history<F>(
        &self,
        fingerprint: &str,
        at: Timestamp,
        update: F,
    ) -> StoreResult<ResourceHistoryRecord>
    where
        F: FnOnce(&mut ResourceHistoryRecord),
    {
        validate_fingerprint(fingerprint)?;
        let transaction = self.database.begin_write().store()?;
        let mut record = {
            let table = transaction.open_table(RESOURCE_HISTORY).store()?;
            let record = table
                .get(fingerprint)
                .store()?
                .map(|value| decode::<ResourceHistoryRecord>(value.value()))
                .transpose()?
                .unwrap_or(ResourceHistoryRecord::new(fingerprint, at)?);
            record
        };
        if at < record.updated_at {
            return Err(StoreError::InvalidRecord(
                "resource-history timestamp moved backwards".into(),
            ));
        }
        update(&mut record);
        record.updated_at = at;
        if record.fingerprint != fingerprint {
            return Err(StoreError::InvalidRecord(
                "resource-history update changed its fingerprint".into(),
            ));
        }
        let bytes = encode(&record)?;
        {
            let mut table = transaction.open_table(RESOURCE_HISTORY).store()?;
            table.insert(fingerprint, bytes.as_slice()).store()?;
        }
        transaction.commit().store()?;
        Ok(record)
    }

    /// Return state events that were committed with redb but have not yet been
    /// acknowledged as durable in the event segment log.
    pub fn pending_transition_events(&self) -> StoreResult<Vec<PendingTransitionEvent>> {
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(TRANSITION_OUTBOX).store()?;
        let mut pending = Vec::with_capacity(table.len().store()? as usize);
        for row in table.iter().store()? {
            let (_key, value) = row.store()?;
            pending.push(decode(value.value())?);
        }
        pending.sort_by_key(|event: &PendingTransitionEvent| {
            (
                event.transition.at,
                event.transition.job_id,
                event.transition.from,
                event.transition.to,
            )
        });
        Ok(pending)
    }

    /// Idempotently acknowledge publication after the event segment append is
    /// durable. A mismatched value under the deterministic key is corruption.
    pub fn acknowledge_transition_event(&self, transition: &FencedTransition) -> StoreResult<()> {
        let key = transition_outbox_key(transition)?;
        let transaction = self.database.begin_write().store()?;
        {
            let mut table = transaction.open_table(TRANSITION_OUTBOX).store()?;
            let pending = match table.get(key.as_str()).store()? {
                Some(stored) => Some(decode::<PendingTransitionEvent>(stored.value())?),
                None => None,
            };
            if let Some(pending) = pending {
                if pending.transition != *transition {
                    return Err(StoreError::Corrupt(
                        "transition outbox key does not match its value".to_owned(),
                    ));
                }
                table.remove(key.as_str()).store()?;
            }
        }
        transaction.commit().store()?;
        Ok(())
    }

    pub fn put_meta<T: Serialize>(&self, key: &str, value: &T) -> StoreResult<()> {
        validate_meta_key(key)?;
        let bytes = encode(value)?;
        let transaction = self.database.begin_write().store()?;
        {
            let mut table = transaction.open_table(META).store()?;
            table.insert(key, bytes.as_slice()).store()?;
        }
        transaction.commit().store()?;
        Ok(())
    }

    pub fn get_meta<T: DeserializeOwned>(&self, key: &str) -> StoreResult<Option<T>> {
        validate_meta_key(key)?;
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(META).store()?;
        table
            .get(key)
            .store()?
            .map(|value| decode(value.value()))
            .transpose()
    }
}

#[cfg(test)]
mod tests;
