use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use jain_smartcluster_core::{
    AttachRequest, CancelMode, CancelRequest, ClientFrame, DaemonStatus, ErrorCode, EventOffset,
    FencedAttempt, GetRequest, IdempotencyKey, JainWorkload, JobId, JobPage, JobRecord, JobSpec,
    JobState, ListRequest, OperationAck, PauseRequest, Principal, ProtocolVersion, Request,
    RequestEnvelope, RequestId, RequestIdentity, Response, ResumeRequest, SendControlRequest,
    ServerError, ServerFrame, StreamOpened, SubmitRequest, SubscribeRequest, TerminalStatus,
    Validate, ValidationError, WaitRequest,
};
use thiserror::Error;
use tokio::{net::UnixStream, time::timeout};

use crate::{Attachment, EventStream, FrameCodec, FrameError, DEFAULT_MAX_FRAME_BYTES};

mod connection;
mod monitor;
mod support;
pub use monitor::JobMonitor;
pub use support::job_subscription;
use support::*;

pub const DEFAULT_SOCKET_PATH: &str = "/run/jain-smartcluster/scqd.sock";

/// Stable result identity returned by the typed dispatcher.  The event offset is
/// initialized to zero for a newly submitted job and is advanced by the caller
/// as it acknowledges events from [`EventStream`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobHandle {
    pub job_id: JobId,
    pub attempt: Option<FencedAttempt>,
    pub event_offset: EventOffset,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobStatus {
    pub job_id: JobId,
    pub state: JobState,
    pub attempt: Option<FencedAttempt>,
    pub terminal: Option<TerminalStatus>,
    pub event_offset: EventOffset,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CancelReceipt {
    pub job_id: JobId,
    pub attempt: Option<FencedAttempt>,
    pub event_offset: EventOffset,
}

#[derive(Clone, Debug)]
pub struct ClientConfig {
    pub socket_path: PathBuf,
    pub max_frame_bytes: usize,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    /// `None` allows a terminal wait to last for the lifetime of the job.
    pub wait_timeout: Option<Duration>,
    pub stream_buffer: usize,
}

impl ClientConfig {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
            ..Self::default()
        }
    }

    fn validate(&self) -> Result<(), DispatchError> {
        if self.socket_path.as_os_str().is_empty() {
            return Err(DispatchError::InvalidConfig(
                "socket path must not be empty".into(),
            ));
        }
        if self.connect_timeout.is_zero() {
            return Err(DispatchError::InvalidConfig(
                "connect timeout must be nonzero".into(),
            ));
        }
        if self.request_timeout.is_zero() {
            return Err(DispatchError::InvalidConfig(
                "request timeout must be nonzero".into(),
            ));
        }
        if self.wait_timeout.is_some_and(|duration| duration.is_zero()) {
            return Err(DispatchError::InvalidConfig(
                "wait timeout must be nonzero when set".into(),
            ));
        }
        if self.stream_buffer == 0 {
            return Err(DispatchError::InvalidConfig(
                "stream buffer must be nonzero".into(),
            ));
        }
        FrameCodec::new(self.max_frame_bytes)?;
        Ok(())
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from(DEFAULT_SOCKET_PATH),
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            connect_timeout: Duration::from_secs(2),
            request_timeout: Duration::from_secs(30),
            wait_timeout: None,
            stream_buffer: 256,
        }
    }
}

#[derive(Debug, Error)]
pub enum DispatchError {
    #[error("invalid client configuration: {0}")]
    InvalidConfig(String),
    #[error("request validation failed: {0}")]
    Validation(#[from] ValidationError),
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error("failed to connect to {path}: {source}")]
    Connect { path: PathBuf, source: io::Error },
    #[error("{operation} timed out")]
    Timeout { operation: &'static str },
    #[error("daemon closed the connection")]
    ConnectionClosed,
    #[error("incompatible protocol: client {client}, server {server}")]
    ProtocolMismatch {
        client: ProtocolVersion,
        server: ProtocolVersion,
    },
    #[error("response request id mismatch: expected {expected}, got {actual}")]
    WrongRequestId {
        expected: RequestId,
        actual: RequestId,
    },
    #[error("stream id mismatch: expected {expected}, got {actual}")]
    WrongStream {
        expected: jain_smartcluster_core::StreamId,
        actual: jain_smartcluster_core::StreamId,
    },
    #[error(
        "event history gap: requested {requested}, available from {available_from}, latest {latest}: {reason}"
    )]
    StreamGap {
        requested: jain_smartcluster_core::EventOffset,
        available_from: jain_smartcluster_core::EventOffset,
        latest: jain_smartcluster_core::EventOffset,
        reason: String,
    },
    #[error("expected {expected} response, got {actual}")]
    UnexpectedResponse {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("unexpected server frame: {0}")]
    UnexpectedServerFrame(&'static str),
    #[error(transparent)]
    Server(#[from] ServerError),
}

#[derive(Clone, Debug)]
pub struct Client {
    config: Arc<ClientConfig>,
    codec: FrameCodec,
}

impl Default for Client {
    fn default() -> Self {
        Self::new(DEFAULT_SOCKET_PATH)
    }
}

#[async_trait]
pub trait WorkloadDispatcher: Send + Sync {
    async fn submit(
        &self,
        identity: RequestIdentity,
        workload: JainWorkload,
        idempotency: IdempotencyKey,
    ) -> Result<JobHandle, DispatchError>;
    async fn attach(&self, job_id: JobId, after: EventOffset)
        -> Result<EventStream, DispatchError>;
    async fn cancel(&self, job_id: JobId, mode: CancelMode)
        -> Result<CancelReceipt, DispatchError>;
    async fn status(&self, job_id: JobId) -> Result<JobStatus, DispatchError>;
}

#[async_trait]
impl WorkloadDispatcher for Client {
    async fn submit(
        &self,
        identity: RequestIdentity,
        workload: JainWorkload,
        idempotency: IdempotencyKey,
    ) -> Result<JobHandle, DispatchError> {
        self.dispatch_submit(identity, workload, idempotency).await
    }

    async fn attach(
        &self,
        job_id: JobId,
        after: EventOffset,
    ) -> Result<EventStream, DispatchError> {
        self.dispatch_attach(job_id, after).await
    }

    async fn cancel(
        &self,
        job_id: JobId,
        mode: CancelMode,
    ) -> Result<CancelReceipt, DispatchError> {
        self.dispatch_cancel(job_id, mode).await
    }

    async fn status(&self, job_id: JobId) -> Result<JobStatus, DispatchError> {
        self.dispatch_status(job_id).await
    }
}

#[cfg(test)]
mod tests;
