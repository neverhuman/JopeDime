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

impl Client {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        let config = ClientConfig::new(socket_path);
        Self {
            codec: FrameCodec::default(),
            config: Arc::new(config),
        }
    }

    pub fn with_config(config: ClientConfig) -> Result<Self, DispatchError> {
        config.validate()?;
        let codec = FrameCodec::new(config.max_frame_bytes)?;
        Ok(Self {
            config: Arc::new(config),
            codec,
        })
    }

    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    pub fn socket_path(&self) -> &Path {
        &self.config.socket_path
    }

    pub async fn status(&self) -> Result<DaemonStatus, DispatchError> {
        match self
            .round_trip(
                Request::Status(jain_smartcluster_core::StatusRequest),
                Some(self.config.request_timeout),
                "status",
            )
            .await?
        {
            Response::Status(status) => Ok(status),
            response => Err(unexpected("status", &response)),
        }
    }

    pub async fn submit(&self, spec: JobSpec) -> Result<JobRecord, DispatchError> {
        self.submit_request(SubmitRequest {
            spec,
            effective_principal: None,
        })
        .await
    }

    pub async fn submit_as(
        &self,
        spec: JobSpec,
        effective_principal: Principal,
    ) -> Result<JobRecord, DispatchError> {
        self.submit_request(SubmitRequest {
            spec,
            effective_principal: Some(effective_principal),
        })
        .await
    }

    async fn submit_request(&self, request: SubmitRequest) -> Result<JobRecord, DispatchError> {
        match self
            .round_trip(
                Request::Submit(Box::new(request)),
                Some(self.config.request_timeout),
                "submit",
            )
            .await?
        {
            Response::Submitted(job) => Ok(job),
            response => Err(unexpected("submitted", &response)),
        }
    }

    pub async fn get(&self, job_id: JobId) -> Result<Option<JobRecord>, DispatchError> {
        match self
            .round_trip(
                Request::Get(GetRequest { job_id }),
                Some(self.config.request_timeout),
                "get",
            )
            .await?
        {
            Response::Job(job) => Ok(job),
            response => Err(unexpected("job", &response)),
        }
    }

    pub async fn list(&self, request: ListRequest) -> Result<JobPage, DispatchError> {
        match self
            .round_trip(
                Request::List(request),
                Some(self.config.request_timeout),
                "list",
            )
            .await?
        {
            Response::Jobs(page) => Ok(page),
            response => Err(unexpected("jobs", &response)),
        }
    }

    pub async fn wait(&self, job_id: JobId) -> Result<JobRecord, DispatchError> {
        match self
            .round_trip(
                Request::Wait(WaitRequest { job_id }),
                self.config.wait_timeout,
                "wait",
            )
            .await?
        {
            Response::Waited(job) => Ok(job),
            response => Err(unexpected("waited", &response)),
        }
    }

    pub async fn cancel(&self, request: CancelRequest) -> Result<JobRecord, DispatchError> {
        self.mutate(Request::Cancel(request), "cancel").await
    }

    pub async fn pause(&self, job_id: JobId) -> Result<JobRecord, DispatchError> {
        self.mutate(Request::Pause(PauseRequest { job_id }), "pause")
            .await
    }

    pub async fn resume(&self, job_id: JobId) -> Result<JobRecord, DispatchError> {
        self.mutate(Request::Resume(ResumeRequest { job_id }), "resume")
            .await
    }

    async fn mutate(
        &self,
        request: Request,
        operation: &'static str,
    ) -> Result<JobRecord, DispatchError> {
        match self
            .round_trip(request, Some(self.config.request_timeout), operation)
            .await?
        {
            Response::Mutated(job) => Ok(job),
            response => Err(unexpected("mutated", &response)),
        }
    }

    /// Only a qualified native fit authority leader may call this operation.
    pub async fn inspect_attempt(
        &self,
        attempt: jain_smartcluster_core::FencedAttempt,
    ) -> Result<Option<jain_smartcluster_core::AttemptInspection>, DispatchError> {
        match self
            .round_trip(
                Request::InspectAttempt(jain_smartcluster_core::InspectAttemptRequest { attempt }),
                Some(self.config.request_timeout),
                "inspect_attempt",
            )
            .await?
        {
            Response::AttemptInspected(value) => Ok(value),
            response => Err(unexpected("attempt_inspected", &response)),
        }
    }

    pub async fn attach_trusted_fit_child(
        &self,
        request: jain_smartcluster_core::AttachTrustedFitChildRequest,
    ) -> Result<jain_smartcluster_core::TrustedFitChildPlacement, DispatchError> {
        match self
            .round_trip(
                Request::AttachTrustedFitChild(Box::new(request)),
                Some(self.config.request_timeout),
                "attach_trusted_fit_child",
            )
            .await?
        {
            Response::TrustedFitChildPlaced(receipt) => Ok(receipt),
            response => Err(unexpected("trusted_fit_child_placed", &response)),
        }
    }

    pub async fn send_control(
        &self,
        request: SendControlRequest,
    ) -> Result<OperationAck, DispatchError> {
        match self
            .round_trip(
                Request::SendControl(request),
                Some(self.config.request_timeout),
                "send_control",
            )
            .await?
        {
            Response::ControlAccepted(ack) => Ok(ack),
            response => Err(unexpected("control_accepted", &response)),
        }
    }

    pub async fn subscribe(&self, request: SubscribeRequest) -> Result<EventStream, DispatchError> {
        let (socket, opened) = self
            .open_stream(Request::Subscribe(request), StreamKind::Subscription)
            .await?;
        let (reader, writer) = socket.into_split();
        Ok(EventStream::spawn(
            opened.stream_id,
            reader,
            writer,
            self.codec,
            self.config.stream_buffer,
        ))
    }

    pub async fn attach(&self, request: AttachRequest) -> Result<Attachment, DispatchError> {
        let (socket, opened) = self
            .open_stream(Request::Attach(request), StreamKind::Attachment)
            .await?;
        let (reader, writer) = socket.into_split();
        let events = EventStream::spawn(
            opened.stream_id,
            reader,
            writer,
            self.codec,
            self.config.stream_buffer,
        );
        Ok(Attachment::new(events, self.codec))
    }

    async fn open_stream(
        &self,
        request: Request,
        kind: StreamKind,
    ) -> Result<(UnixStream, StreamOpened), DispatchError> {
        let mut socket = self.connect().await?;
        let response = self
            .exchange_on(
                &mut socket,
                request,
                Some(self.config.request_timeout),
                kind.operation(),
            )
            .await?;
        let opened = match (kind, response) {
            (StreamKind::Subscription, Response::Subscribed(opened))
            | (StreamKind::Attachment, Response::Attached(opened)) => opened,
            (_, response) => return Err(unexpected(kind.expected_response(), &response)),
        };
        Ok((socket, opened))
    }

    async fn round_trip(
        &self,
        request: Request,
        request_timeout: Option<Duration>,
        operation: &'static str,
    ) -> Result<Response, DispatchError> {
        let mut socket = self.connect().await?;
        self.exchange_on(&mut socket, request, request_timeout, operation)
            .await
    }

    async fn exchange_on(
        &self,
        socket: &mut UnixStream,
        request: Request,
        request_timeout: Option<Duration>,
        operation: &'static str,
    ) -> Result<Response, DispatchError> {
        request.validate()?;
        let envelope = RequestEnvelope::new(request);
        let request_id = envelope.request_id;
        let exchange = async {
            self.codec
                .write_frame(socket, &ClientFrame::Request(Box::new(envelope)))
                .await?;
            let frame = self
                .codec
                .read_frame::<ServerFrame, _>(socket)
                .await?
                .ok_or(DispatchError::ConnectionClosed)?;
            let ServerFrame::Response(response) = frame else {
                return Err(DispatchError::UnexpectedServerFrame(server_frame_name(
                    &frame,
                )));
            };
            if !ProtocolVersion::CURRENT.is_compatible_with(response.version) {
                return Err(DispatchError::ProtocolMismatch {
                    client: ProtocolVersion::CURRENT,
                    server: response.version,
                });
            }
            if response.request_id != request_id {
                return Err(DispatchError::WrongRequestId {
                    expected: request_id,
                    actual: response.request_id,
                });
            }
            response.result.map_err(DispatchError::from)
        };
        with_timeout(request_timeout, operation, exchange).await
    }

    async fn connect(&self) -> Result<UnixStream, DispatchError> {
        let path = self.config.socket_path.clone();
        match timeout(self.config.connect_timeout, UnixStream::connect(&path)).await {
            Ok(Ok(socket)) => Ok(socket),
            Ok(Err(source)) => Err(DispatchError::Connect { path, source }),
            Err(_) => Err(DispatchError::Timeout {
                operation: "connect",
            }),
        }
    }

    /// Submit an allowlisted Jain workload through the schema-versioned API.
    /// The lower-level [`JobSpec`] methods remain the compatibility adapter;
    /// production callers should use this method or [`WorkloadDispatcher`].
    pub async fn dispatch_submit(
        &self,
        identity: RequestIdentity,
        workload: JainWorkload,
        idempotency: IdempotencyKey,
    ) -> Result<JobHandle, DispatchError> {
        identity.validate()?;
        let spec = typed_job_spec(workload, idempotency)?;
        let job = if identity.is_delegated() {
            self.submit_as(spec, identity.effective).await?
        } else {
            self.submit(spec).await?
        };
        Ok(JobHandle::from(&job))
    }

    pub async fn dispatch_attach(
        &self,
        job_id: JobId,
        after: EventOffset,
    ) -> Result<EventStream, DispatchError> {
        Ok(self
            .attach(AttachRequest {
                job_id,
                offset: after,
                read_only: true,
            })
            .await?
            .into_parts()
            .0)
    }

    pub async fn dispatch_cancel(
        &self,
        job_id: JobId,
        mode: CancelMode,
    ) -> Result<CancelReceipt, DispatchError> {
        let job = self.cancel(CancelRequest { job_id, mode }).await?;
        Ok(CancelReceipt::from(&job))
    }

    pub async fn dispatch_status(&self, job_id: JobId) -> Result<JobStatus, DispatchError> {
        let job = self.get(job_id).await?.ok_or_else(|| {
            DispatchError::Server(ServerError::new(ErrorCode::NotFound, "job does not exist"))
        })?;
        Ok(JobStatus::from(&job))
    }
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
mod tests {
    use std::collections::BTreeMap;

    use jain_smartcluster_core::{
        AttachInput, CancelMode, CheckpointPolicy, ClientFrame, CpuRequest, DaemonMode,
        EffectiveCapacity, Environment, EventEnvelope, EventFilter, EventOffset, EventRecord,
        IdempotencyKey, IoMode, JainWorkload, JobEvent, JobPage, ListFilter, MemoryRequest,
        PreemptionPolicy, PressureStatus, PriorityClass, ProjectId, Request, ResourceRequest,
        ResponseEnvelope, RetryPolicy, SchedulerCapabilities, SendControlRequest, ServerFrame,
        StreamGap, StreamId, StreamOpened, TenantId, Timestamp, Workload,
    };
    use tempfile::TempDir;
    use tokio::net::UnixListener;

    use super::*;

    fn spec() -> JobSpec {
        JobSpec {
            tenant: TenantId::new("tenant-a").unwrap(),
            project: ProjectId::new("project-a").unwrap(),
            idempotency_key: IdempotencyKey::new("request-a").unwrap(),
            priority: PriorityClass::Batch,
            workload: Workload::Jain {
                template: JainWorkload::Training,
                arguments: Vec::new(),
                environment: Environment::default(),
            },
            resources: ResourceRequest {
                cpu: CpuRequest::fixed(1000),
                ram: MemoryRequest::fixed(1024 * 1024),
                scratch_bytes: 0,
                gpu: None,
            },
            retry: RetryPolicy::default(),
            preemption: PreemptionPolicy {
                preemptible: true,
                checkpoint: CheckpointPolicy::ControlFd { deadline_ms: 1000 },
            },
            io: IoMode::Pipes,
            deadline: None,
            labels: BTreeMap::new(),
        }
    }

    fn job() -> JobRecord {
        JobRecord::queued(
            JobId::new(),
            spec(),
            Principal::unix(1000),
            Timestamp::from_unix_millis(1),
        )
    }

    fn status() -> DaemonStatus {
        DaemonStatus {
            daemon_version: "8.0.0".into(),
            protocol_version: ProtocolVersion::CURRENT,
            mode: DaemonMode::Full,
            started_at: 1.into(),
            paused: false,
            capabilities: SchedulerCapabilities::default(),
            effective_capacity: EffectiveCapacity::default(),
            pressure: PressureStatus::default(),
            gpus: Vec::new(),
            queued_jobs: 1,
            running_jobs: 0,
        }
    }

    fn listener() -> (TempDir, PathBuf, UnixListener) {
        let directory = tempfile::tempdir().unwrap();
        let socket_path = directory.path().join("scqd.sock");
        let listener = UnixListener::bind(&socket_path).unwrap();
        (directory, socket_path, listener)
    }

    #[tokio::test]
    async fn every_unary_operation_maps_to_its_typed_response() {
        let (_directory, socket_path, listener) = listener();
        let expected_job = job();
        let server_job = expected_job.clone();
        let server = tokio::spawn(async move {
            let codec = FrameCodec::default();
            for _ in 0..9 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let ClientFrame::Request(envelope) = codec
                    .read_frame::<ClientFrame, _>(&mut socket)
                    .await
                    .unwrap()
                    .unwrap()
                else {
                    panic!("expected request frame");
                };
                let response = match envelope.request {
                    Request::Status(_) => Response::Status(status()),
                    Request::Submit(request) => {
                        assert_eq!(request.effective_principal, None);
                        Response::Submitted(server_job.clone())
                    }
                    Request::Get(_) => Response::Job(Some(server_job.clone())),
                    Request::List(_) => Response::Jobs(JobPage {
                        jobs: vec![server_job.clone()],
                        next_cursor: Some("next".into()),
                    }),
                    Request::Wait(_) => Response::Waited(server_job.clone()),
                    Request::Cancel(_) | Request::Pause(_) | Request::Resume(_) => {
                        Response::Mutated(server_job.clone())
                    }
                    Request::SendControl(_) => Response::ControlAccepted(OperationAck {
                        accepted_at: 2.into(),
                    }),
                    Request::Subscribe(_)
                    | Request::Attach(_)
                    | Request::AttachTrustedFitChild(_)
                    | Request::InspectAttempt(_) => panic!("unexpected operation"),
                };
                codec
                    .write_frame(
                        &mut socket,
                        &ServerFrame::Response(Box::new(ResponseEnvelope::ok(
                            envelope.request_id,
                            response,
                        ))),
                    )
                    .await
                    .unwrap();
            }
        });

        let client = Client::new(socket_path);
        assert_eq!(client.status().await.unwrap().daemon_version, "8.0.0");
        assert_eq!(client.submit(spec()).await.unwrap(), expected_job);
        assert_eq!(
            client.get(expected_job.id).await.unwrap(),
            Some(expected_job.clone())
        );
        assert_eq!(
            client
                .list(ListRequest {
                    filter: ListFilter::default(),
                    cursor: None,
                    limit: 10,
                })
                .await
                .unwrap()
                .jobs,
            vec![expected_job.clone()]
        );
        assert_eq!(client.wait(expected_job.id).await.unwrap(), expected_job);
        assert_eq!(
            client
                .cancel(CancelRequest {
                    job_id: expected_job.id,
                    mode: CancelMode::Graceful { grace_ms: 1000 },
                })
                .await
                .unwrap(),
            expected_job
        );
        assert_eq!(client.pause(expected_job.id).await.unwrap(), expected_job);
        assert_eq!(client.resume(expected_job.id).await.unwrap(), expected_job);
        assert_eq!(
            client
                .send_control(SendControlRequest {
                    job_id: expected_job.id,
                    control: jain_smartcluster_core::ControlMessage::Advance,
                })
                .await
                .unwrap()
                .accepted_at,
            Timestamp::from_unix_millis(2)
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn same_major_different_minor_response_is_rejected() {
        let (_directory, socket_path, listener) = listener();
        let server = tokio::spawn(async move {
            let codec = FrameCodec::default();
            let (mut socket, _) = listener.accept().await.unwrap();
            let ClientFrame::Request(envelope) = codec
                .read_frame::<ClientFrame, _>(&mut socket)
                .await
                .unwrap()
                .unwrap()
            else {
                panic!("expected request frame");
            };
            let mut response =
                ResponseEnvelope::ok(envelope.request_id, Response::Status(status()));
            response.version.minor += 1;
            codec
                .write_frame(&mut socket, &ServerFrame::response(response))
                .await
                .unwrap();
        });

        let error = Client::new(socket_path).status().await.unwrap_err();
        assert!(matches!(
            error,
            DispatchError::ProtocolMismatch { client, server }
                if client == ProtocolVersion::CURRENT
                    && server.major == ProtocolVersion::CURRENT.major
                    && server.minor == ProtocolVersion::CURRENT.minor + 1
        ));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn subscription_and_attachment_stream_events_and_input() {
        let (_directory, socket_path, listener) = listener();
        let job_id = JobId::new();
        let server = tokio::spawn(async move {
            let codec = FrameCodec::default();
            let stream_id = StreamId::new();
            let (mut socket, _) = listener.accept().await.unwrap();
            let ClientFrame::Request(envelope) = codec
                .read_frame::<ClientFrame, _>(&mut socket)
                .await
                .unwrap()
                .unwrap()
            else {
                panic!("expected subscribe request");
            };
            assert!(matches!(envelope.request, Request::Subscribe(_)));
            codec
                .write_frame(
                    &mut socket,
                    &ServerFrame::Response(Box::new(ResponseEnvelope::ok(
                        envelope.request_id,
                        Response::Subscribed(StreamOpened {
                            stream_id,
                            next_offset: EventOffset::new(7),
                        }),
                    ))),
                )
                .await
                .unwrap();
            codec
                .write_frame(
                    &mut socket,
                    &ServerFrame::StreamGap {
                        stream_id,
                        gap: StreamGap {
                            requested: EventOffset::new(5),
                            available_from: EventOffset::new(7),
                            latest: EventOffset::new(8),
                            reason: "event segments were compacted".to_owned(),
                        },
                    },
                )
                .await
                .unwrap();
            codec
                .write_frame(
                    &mut socket,
                    &ServerFrame::Event(EventEnvelope {
                        stream_id,
                        event: EventRecord {
                            offset: 7.into(),
                            job_id,
                            attempt: None,
                            at: 1.into(),
                            event: JobEvent::Stdout(b"hello".to_vec()),
                        },
                    }),
                )
                .await
                .unwrap();
            codec
                .write_frame(
                    &mut socket,
                    &ServerFrame::StreamClosed {
                        stream_id,
                        error: None,
                    },
                )
                .await
                .unwrap();

            let attach_stream_id = StreamId::new();
            let (mut socket, _) = listener.accept().await.unwrap();
            let ClientFrame::Request(envelope) = codec
                .read_frame::<ClientFrame, _>(&mut socket)
                .await
                .unwrap()
                .unwrap()
            else {
                panic!("expected attach request");
            };
            assert!(matches!(envelope.request, Request::Attach(_)));
            codec
                .write_frame(
                    &mut socket,
                    &ServerFrame::Response(Box::new(ResponseEnvelope::ok(
                        envelope.request_id,
                        Response::Attached(StreamOpened {
                            stream_id: attach_stream_id,
                            next_offset: EventOffset::ZERO,
                        }),
                    ))),
                )
                .await
                .unwrap();
            let input = codec
                .read_frame::<ClientFrame, _>(&mut socket)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                input,
                ClientFrame::StreamInput(jain_smartcluster_core::StreamInputFrame {
                    stream_id: attach_stream_id,
                    input: AttachInput::Resize {
                        rows: 40,
                        columns: 120,
                    },
                })
            );
            codec
                .write_frame(
                    &mut socket,
                    &ServerFrame::Event(EventEnvelope {
                        stream_id: attach_stream_id,
                        event: EventRecord {
                            offset: 8.into(),
                            job_id,
                            attempt: None,
                            at: 2.into(),
                            event: JobEvent::Pty(b"prompt".to_vec()),
                        },
                    }),
                )
                .await
                .unwrap();
            codec
                .write_frame(
                    &mut socket,
                    &ServerFrame::StreamClosed {
                        stream_id: attach_stream_id,
                        error: None,
                    },
                )
                .await
                .unwrap();
        });

        let client = Client::new(socket_path);
        let mut subscription = client
            .subscribe(SubscribeRequest {
                offset: EventOffset::new(5),
                filter: EventFilter::default(),
            })
            .await
            .unwrap();
        let gap = subscription.recv().await.unwrap().unwrap_err();
        assert!(matches!(
            gap,
            DispatchError::StreamGap {
                requested,
                available_from,
                latest,
                ..
            } if requested == EventOffset::new(5)
                && available_from == EventOffset::new(7)
                && latest == EventOffset::new(8)
        ));
        let event = subscription.recv().await.unwrap().unwrap();
        assert_eq!(event.event, JobEvent::Stdout(b"hello".to_vec()));
        assert!(subscription.recv().await.is_none());

        let mut attachment = client
            .attach(AttachRequest {
                job_id,
                offset: EventOffset::ZERO,
                read_only: false,
            })
            .await
            .unwrap();
        attachment.sender.resize(40, 120).await.unwrap();
        let event = attachment.events.recv().await.unwrap().unwrap();
        assert_eq!(event.event, JobEvent::Pty(b"prompt".to_vec()));
        assert!(attachment.events.recv().await.is_none());
        server.await.unwrap();
    }

    #[test]
    fn client_implements_object_safe_dispatch_interface() {
        fn accepts_dispatcher(_dispatcher: &dyn WorkloadDispatcher) {}
        let client = Client::default();
        accepts_dispatcher(&client);
    }
}
