use std::{future::Future, time::Duration};

use jain_smartcluster_core::{
    EventFilter, EventOffset, IdempotencyKey, JainWorkload, JobId, JobRecord, JobSpec,
    PreemptionPolicy, PriorityClass, ProjectId, ResourceRequest, Response, RetryPolicy,
    ServerFrame, SubscribeRequest, TenantId, Workload,
};
use tokio::time::timeout;

use super::{CancelReceipt, DispatchError, JobHandle, JobStatus};

pub(super) fn typed_job_spec(
    workload: JainWorkload,
    idempotency_key: IdempotencyKey,
) -> Result<JobSpec, DispatchError> {
    Ok(JobSpec {
        tenant: TenantId::new("jain").map_err(DispatchError::Validation)?,
        project: ProjectId::new("typed-dispatch").map_err(DispatchError::Validation)?,
        idempotency_key,
        priority: PriorityClass::Interactive,
        workload: Workload::Jain {
            template: workload,
            arguments: Vec::new(),
            environment: Default::default(),
        },
        resources: ResourceRequest::cpu_and_ram(1_000, 1024 * 1024 * 1024),
        retry: RetryPolicy::default(),
        preemption: PreemptionPolicy::default(),
        io: Default::default(),
        deadline: None,
        labels: Default::default(),
    })
}

impl From<&JobRecord> for JobHandle {
    fn from(job: &JobRecord) -> Self {
        Self {
            job_id: job.id,
            attempt: job.active_attempt,
            event_offset: EventOffset::ZERO,
        }
    }
}

impl From<&JobRecord> for JobStatus {
    fn from(job: &JobRecord) -> Self {
        Self {
            job_id: job.id,
            state: job.state,
            attempt: job.active_attempt,
            terminal: job.terminal.clone(),
            event_offset: EventOffset::ZERO,
        }
    }
}

impl From<&JobRecord> for CancelReceipt {
    fn from(job: &JobRecord) -> Self {
        Self {
            job_id: job.id,
            attempt: job.active_attempt,
            event_offset: EventOffset::ZERO,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum StreamKind {
    Subscription,
    Attachment,
}

impl StreamKind {
    pub(super) const fn operation(self) -> &'static str {
        match self {
            Self::Subscription => "subscribe",
            Self::Attachment => "attach",
        }
    }

    pub(super) const fn expected_response(self) -> &'static str {
        match self {
            Self::Subscription => "subscribed",
            Self::Attachment => "attached",
        }
    }
}

pub(super) async fn with_timeout<T, F>(
    duration: Option<Duration>,
    operation: &'static str,
    future: F,
) -> Result<T, DispatchError>
where
    F: Future<Output = Result<T, DispatchError>>,
{
    if let Some(duration) = duration {
        timeout(duration, future)
            .await
            .map_err(|_| DispatchError::Timeout { operation })?
    } else {
        future.await
    }
}

fn response_name(response: &Response) -> &'static str {
    match response {
        Response::Status(_) => "status",
        Response::Submitted(_) => "submitted",
        Response::Job(_) => "job",
        Response::Jobs(_) => "jobs",
        Response::Waited(_) => "waited",
        Response::Mutated(_) => "mutated",
        Response::ControlAccepted(_) => "control_accepted",
        Response::Subscribed(_) => "subscribed",
        Response::Attached(_) => "attached",
        Response::TrustedFitChildPlaced(_) => "trusted_fit_child_placed",
        Response::AttemptInspected(_) => "attempt_inspected",
    }
}

pub(super) fn unexpected(expected: &'static str, response: &Response) -> DispatchError {
    DispatchError::UnexpectedResponse {
        expected,
        actual: response_name(response),
    }
}

pub(super) fn server_frame_name(frame: &ServerFrame) -> &'static str {
    match frame {
        ServerFrame::Response(_) => "response",
        ServerFrame::Event(_) => "event",
        ServerFrame::StreamClosed { .. } => "stream_closed",
        ServerFrame::StreamGap { .. } => "stream_gap",
    }
}

/// Convenience filter for observing one job from a durable offset.
pub fn job_subscription(job_id: JobId, offset: EventOffset) -> SubscribeRequest {
    SubscribeRequest {
        offset,
        filter: EventFilter {
            job_id: Some(job_id),
            ..EventFilter::default()
        },
    }
}
