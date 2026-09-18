use super::*;

pub(super) fn retry_delay(job: &JobRecord) -> Duration {
    let exponent = job.failure_count.saturating_sub(1).min(31);
    let base = job
        .spec
        .retry
        .initial_backoff_ms
        .saturating_mul(1_u64 << exponent)
        .min(job.spec.retry.maximum_backoff_ms);
    let spread = base.saturating_mul(u64::from(job.spec.retry.jitter_per_mille)) / 1_000;
    let seed = job.id.as_ulid().0 as u64 ^ u64::from(job.attempt_count);
    let jitter = if spread == 0 {
        0
    } else {
        seed % spread.saturating_add(1)
    };
    Duration::from_millis(
        base.saturating_sub(spread / 2)
            .saturating_add(jitter)
            .min(job.spec.retry.maximum_backoff_ms),
    )
}

pub(super) fn remaining_retry_delay(job: &JobRecord, now: Timestamp) -> Duration {
    let total = retry_delay(job);
    let elapsed = Duration::from_millis(
        now.as_unix_millis()
            .saturating_sub(job.updated_at.as_unix_millis()),
    );
    total.saturating_sub(elapsed)
}

pub(super) fn workload_has_environment(workload: &jain_smartcluster_core::Workload) -> bool {
    match workload {
        jain_smartcluster_core::Workload::Jain { environment, .. }
        | jain_smartcluster_core::Workload::TrustedService { environment, .. }
        | jain_smartcluster_core::Workload::AdministratorArgv { environment, .. } => {
            !environment.0.is_empty()
        }
    }
}

pub(super) fn bounded_control(
    mut control: jain_smartcluster_core::ControlMessage,
    execution: &ExecutionConfig,
) -> Result<jain_smartcluster_core::ControlMessage, ServerError> {
    control
        .validate()
        .map_err(|error| ServerError::new(ErrorCode::InvalidRequest, error.to_string()))?;
    if let jain_smartcluster_core::ControlMessage::Checkpoint { deadline_ms } = &mut control {
        *deadline_ms = (*deadline_ms).min(execution.checkpoint_timeout_ms);
    }
    let encoded = postcard::to_stdvec(&control)
        .map_err(|error| ServerError::new(ErrorCode::InvalidRequest, error.to_string()))?;
    if encoded.len() > execution.max_control_frame_bytes {
        return Err(ServerError::new(
            ErrorCode::InvalidRequest,
            "control message exceeds the configured frame limit",
        ));
    }
    Ok(control)
}

pub(super) fn deadline_expired(job: &JobRecord, now: Timestamp) -> bool {
    job.spec.deadline.is_some_and(|deadline| deadline <= now)
}

pub(super) fn next_lease_expiry(current: Timestamp, now: Timestamp) -> Timestamp {
    let minimum = now.as_unix_millis().saturating_add(ATTEMPT_LEASE_MS);
    let advance = current.as_unix_millis().saturating_add(1);
    Timestamp::from_unix_millis(minimum.max(advance))
}

pub(super) fn attempt_allocation_key(attempt: FencedAttempt) -> String {
    format!(
        "attempt-allocation/{}/{}/{}",
        attempt.job_id,
        attempt.number.get(),
        attempt.fence
    )
}

pub(super) fn attempt_pid_start_key(attempt: FencedAttempt) -> String {
    format!(
        "attempt-pid-start/{}/{}/{}",
        attempt.job_id,
        attempt.number.get(),
        attempt.fence
    )
}

pub(super) fn attempt_overload_pause_key(attempt: FencedAttempt) -> String {
    format!(
        "attempt-overload-pause/{}/{}/{}",
        attempt.job_id,
        attempt.number.get(),
        attempt.fence
    )
}

pub(super) fn exit_failure_class(
    exit_success: bool,
    wait_failed: bool,
    host_oom: bool,
    shared_cuda_oom: bool,
    reported: Option<FailureClass>,
) -> Option<FailureClass> {
    if shared_cuda_oom {
        Some(FailureClass::SharedCudaOutOfMemory)
    } else if host_oom {
        Some(FailureClass::HostOutOfMemory)
    } else if reported.is_some() {
        reported
    } else if wait_failed {
        Some(FailureClass::Infrastructure)
    } else if exit_success {
        None
    } else {
        Some(FailureClass::Application)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GpuHistoryAction {
    None,
    CleanExclusive,
    FailedExclusive,
    SharedCudaOom,
}

pub(super) fn gpu_history_action(
    exclusively_isolated: bool,
    observed_vram_bytes: u64,
    successful: bool,
    failure: Option<FailureClass>,
) -> GpuHistoryAction {
    if failure == Some(FailureClass::SharedCudaOutOfMemory) {
        GpuHistoryAction::SharedCudaOom
    } else if exclusively_isolated && successful && observed_vram_bytes > 0 {
        GpuHistoryAction::CleanExclusive
    } else if exclusively_isolated && !successful {
        GpuHistoryAction::FailedExclusive
    } else {
        GpuHistoryAction::None
    }
}

pub(super) fn tenant_matches_principal(
    tenant: &jain_smartcluster_core::TenantId,
    principal: &Principal,
) -> bool {
    tenant.as_str() == principal.to_string()
}

pub(super) fn now_timestamp() -> Timestamp {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    Timestamp::from_unix_millis(millis)
}

pub(super) fn unavailable() -> ServerError {
    ServerError::unavailable("scqd actor is unavailable", 1_000)
}

pub(super) fn auth_error(error: impl std::fmt::Display) -> ServerError {
    ServerError::new(ErrorCode::Forbidden, error.to_string())
}

pub(super) fn store_error(error: StoreError) -> ServerError {
    let code = match error {
        StoreError::JobNotFound(_) => ErrorCode::NotFound,
        StoreError::IdempotencyConflict { .. }
        | StoreError::InvalidTransition(_)
        | StoreError::JobAlreadyExists(_) => ErrorCode::Conflict,
        StoreError::Transition(jain_smartcluster_core::TransitionError::SupersededFence {
            ..
        }) => ErrorCode::SupersededAttempt,
        _ => ErrorCode::Internal,
    };
    ServerError::new(code, error.to_string())
}

pub(super) fn executor_error(error: impl std::fmt::Display) -> ServerError {
    ServerError::new(ErrorCode::Unavailable, error.to_string())
}

pub(super) fn event_error(error: impl std::fmt::Display) -> ServerError {
    ServerError::new(ErrorCode::Internal, error.to_string())
}

pub(super) fn actor_server_error(error: ActorError) -> ServerError {
    ServerError::new(ErrorCode::Internal, error.to_string())
}

pub(super) fn io_error(error: std::io::Error) -> ServerError {
    ServerError::new(ErrorCode::Unavailable, error.to_string())
}

pub(super) fn io_timeout_error() -> ServerError {
    ServerError::new(
        ErrorCode::Unavailable,
        "actor I/O write exceeded the configured timeout",
    )
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ActorError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Event(#[from] EventLogError),
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    #[error(transparent)]
    Cgroup(#[from] crate::cgroup::CgroupError),
    #[error(transparent)]
    Scheduler(#[from] crate::scheduler::SchedulerError),
    #[error(transparent)]
    Nix(#[from] nix::Error),
    #[error("recovery failed: {0}")]
    Recovery(String),
    #[error("protocol operation failed: {0}")]
    Protocol(#[from] ServerError),
    #[error("MPS safety operation failed: {0}")]
    MpsSafety(String),
}

#[derive(Debug, thiserror::Error)]
#[error("daemon startup recovery failed: {0}")]
pub struct ActorStartError(pub(super) String);

pub(super) fn attempt_cpu_pool_key(attempt: FencedAttempt) -> String {
    format!(
        "attempt-cpu-pool:{}:{}:{}",
        attempt.job_id,
        attempt.number.get(),
        attempt.fence
    )
}
