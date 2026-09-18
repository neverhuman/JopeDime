use std::str::FromStr;

use postcard::{from_bytes, to_stdvec};
use sha2::{Digest, Sha256};

use super::*;

pub(super) fn validate_transition_mutation(mutation: &TransitionMutation) -> StoreResult<()> {
    let transition = &mutation.transition;
    transition.from.validate_transition(transition.to)?;
    match (&mutation.terminal, transition.to.is_terminal()) {
        (Some(terminal), true) if terminal.state() == transition.to => {}
        (Some(_), true) => {
            return Err(StoreError::InvalidTransition(
                "terminal result does not match target state".into(),
            ));
        }
        (None, true) => {
            return Err(StoreError::InvalidTransition(
                "terminal target requires a terminal result".into(),
            ));
        }
        (Some(_), false) => {
            return Err(StoreError::InvalidTransition(
                "nonterminal target cannot carry a terminal result".into(),
            ));
        }
        (None, false) => {}
    }
    if mutation
        .terminal
        .as_ref()
        .map(terminal_timestamp)
        .is_some_and(|at| at != transition.at)
    {
        return Err(StoreError::InvalidTransition(
            "terminal timestamp must match its state transition".into(),
        ));
    }
    if mutation.failure.is_some() && transition.attempt.is_none() {
        return Err(StoreError::InvalidTransition(
            "attempt failure requires an attempt fence".into(),
        ));
    }
    if let Some(TerminalStatus::Failed { class, .. }) = mutation.terminal {
        let expected_failure = transition.attempt.map(|_| class);
        if mutation.failure != expected_failure {
            return Err(StoreError::InvalidTransition(
                "attempt-scoped failure must match the terminal failure class".into(),
            ));
        }
    }
    Ok(())
}

fn terminal_timestamp(terminal: &TerminalStatus) -> Timestamp {
    match terminal {
        TerminalStatus::Completed { at, .. }
        | TerminalStatus::Failed { at, .. }
        | TerminalStatus::Cancelled { at, .. } => *at,
    }
}

pub(super) fn retires_attempt(state: JobState) -> bool {
    matches!(
        state,
        JobState::Suspended
            | JobState::RetryWait
            | JobState::Completed
            | JobState::Failed
            | JobState::Cancelled
    )
}

pub(super) fn attempt_table_key(job_id: JobId, number: AttemptNumber) -> String {
    format!("{job_id}/{:010}", number.get())
}

fn queue_index_key(job: &JobRecord) -> String {
    format!("{:020}/{}", job.created_at.as_unix_millis(), job.id)
}

pub(super) fn deadline_index_key(deadline: Timestamp, job_id: &str) -> String {
    format!("{:020}/{job_id}", deadline.as_unix_millis())
}

pub(super) fn index_job(transaction: &redb::WriteTransaction, job: &JobRecord) -> StoreResult<()> {
    if job.state == JobState::Queued {
        let value = job.id.to_string();
        let mut table = transaction.open_table(QUEUED_INDEX).store()?;
        table
            .insert(queue_index_key(job).as_str(), value.as_bytes())
            .store()?;
    }
    if matches!(job.state, JobState::Queued | JobState::Admitted) {
        if let Some(deadline) = job.spec.deadline {
            let value = job.id.to_string();
            let mut table = transaction.open_table(DEADLINE_INDEX).store()?;
            table
                .insert(
                    deadline_index_key(deadline, &value).as_str(),
                    value.as_bytes(),
                )
                .store()?;
        }
    }
    Ok(())
}

fn remove_job_indexes(transaction: &redb::WriteTransaction, job: &JobRecord) -> StoreResult<()> {
    if job.state == JobState::Queued {
        let mut table = transaction.open_table(QUEUED_INDEX).store()?;
        table.remove(queue_index_key(job).as_str()).store()?;
    }
    if matches!(job.state, JobState::Queued | JobState::Admitted) {
        if let Some(deadline) = job.spec.deadline {
            let mut table = transaction.open_table(DEADLINE_INDEX).store()?;
            let id = job.id.to_string();
            table
                .remove(deadline_index_key(deadline, &id).as_str())
                .store()?;
        }
    }
    Ok(())
}

pub(super) fn update_indexes(
    transaction: &redb::WriteTransaction,
    previous: &JobRecord,
    current: &JobRecord,
) -> StoreResult<()> {
    remove_job_indexes(transaction, previous)?;
    index_job(transaction, current)
}

pub(super) fn rebuild_indexes(
    transaction: &redb::WriteTransaction,
    jobs: &[JobRecord],
) -> StoreResult<()> {
    {
        let mut table = transaction.open_table(QUEUED_INDEX).store()?;
        let keys: Vec<_> = table
            .iter()
            .store()?
            .map(|row| row.map(|(key, _)| key.value().to_owned()))
            .collect::<Result<_, _>>()
            .store()?;
        for key in keys {
            table.remove(key.as_str()).store()?;
        }
    }
    {
        let mut table = transaction.open_table(DEADLINE_INDEX).store()?;
        let keys: Vec<_> = table
            .iter()
            .store()?
            .map(|row| row.map(|(key, _)| key.value().to_owned()))
            .collect::<Result<_, _>>()
            .store()?;
        for key in keys {
            table.remove(key.as_str()).store()?;
        }
    }
    for job in jobs {
        index_job(transaction, job)?;
    }
    Ok(())
}

pub(super) fn indexed_job_id(bytes: &[u8]) -> StoreResult<JobId> {
    let value = std::str::from_utf8(bytes)
        .map_err(|error| StoreError::Corrupt(format!("invalid index job id: {error}")))?;
    JobId::from_str(value)
        .map_err(|error| StoreError::Corrupt(format!("invalid index job id: {error}")))
}

pub(super) fn insert_pending_transition(
    transaction: &redb::WriteTransaction,
    transition: &FencedTransition,
) -> StoreResult<()> {
    let key = transition_outbox_key(transition)?;
    let pending = PendingTransitionEvent {
        transition: transition.clone(),
    };
    let bytes = encode(&pending)?;
    let mut table = transaction.open_table(TRANSITION_OUTBOX).store()?;
    table.insert(key.as_str(), bytes.as_slice()).store()?;
    Ok(())
}

pub(super) fn transition_outbox_key(transition: &FencedTransition) -> StoreResult<String> {
    Ok(hex::encode(digest(transition)?))
}

pub(super) fn idempotency_table_key(scope: &IdempotencyScope) -> StoreResult<String> {
    Ok(hex::encode(digest(scope)?))
}

pub(super) fn validate_fingerprint(fingerprint: &str) -> StoreResult<()> {
    if fingerprint.is_empty() || fingerprint.len() > 512 || fingerprint.contains('\0') {
        return Err(StoreError::InvalidRecord(
            "workload fingerprint must contain 1 through 512 non-NUL bytes".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_meta_key(key: &str) -> StoreResult<()> {
    if key.is_empty() || key.len() > 256 || key.starts_with("__scq.") || key.contains('\0') {
        Err(StoreError::InvalidMetaKey)
    } else {
        Ok(())
    }
}

pub(super) fn encode<T: Serialize>(value: &T) -> StoreResult<Vec<u8>> {
    let body = to_stdvec(value).map_err(|error| StoreError::Corrupt(error.to_string()))?;
    let mut bytes = Vec::with_capacity(2 + body.len());
    bytes.extend_from_slice(&VALUE_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

pub(super) fn decode<T: DeserializeOwned>(bytes: &[u8]) -> StoreResult<T> {
    if bytes.len() < 2 {
        return Err(StoreError::Corrupt("stored value has no version".into()));
    }
    let found = u16::from_le_bytes([bytes[0], bytes[1]]);
    if found != VALUE_FORMAT_VERSION {
        return Err(StoreError::UnsupportedFormat { found });
    }
    from_bytes(&bytes[2..]).map_err(|error| StoreError::Corrupt(error.to_string()))
}

pub(super) fn digest<T: Serialize>(value: &T) -> StoreResult<[u8; 32]> {
    let bytes = to_stdvec(value).map_err(|error| StoreError::Corrupt(error.to_string()))?;
    Ok(Sha256::digest(bytes).into())
}
