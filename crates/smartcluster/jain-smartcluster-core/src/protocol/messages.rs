use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    Status(StatusRequest),
    Submit(Box<SubmitRequest>),
    Get(GetRequest),
    List(ListRequest),
    Wait(WaitRequest),
    Cancel(CancelRequest),
    Pause(PauseRequest),
    Resume(ResumeRequest),
    SendControl(SendControlRequest),
    Subscribe(SubscribeRequest),
    Attach(AttachRequest),
    AttachTrustedFitChild(Box<AttachTrustedFitChildRequest>),
    InspectAttempt(InspectAttemptRequest),
}

impl Validate for Request {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Status(_) => Ok(()),
            Self::Submit(request) => {
                request.spec.validate()?;
                if let Some(principal) = &request.effective_principal {
                    principal.validate()?;
                    request.spec.workload.validate_for(principal)?;
                }
                Ok(())
            }
            Self::List(request) => {
                if !(1..=1000).contains(&request.limit) {
                    return Err(ValidationError::new(
                        "list.limit",
                        "must be between 1 and 1000",
                    ));
                }
                if let Some(cursor) = &request.cursor {
                    crate::validation::validate_text("list.cursor", cursor, 4096, false)?;
                }
                Ok(())
            }
            Self::Cancel(request) => match request.mode {
                CancelMode::Graceful { grace_ms: 0 } => {
                    Err(ValidationError::new("cancel.grace_ms", "must be nonzero"))
                }
                CancelMode::CheckpointThenCancel {
                    checkpoint_deadline_ms: 0,
                } => Err(ValidationError::new(
                    "cancel.checkpoint_deadline_ms",
                    "must be nonzero",
                )),
                _ => Ok(()),
            },
            Self::SendControl(request) => request.control.validate(),
            Self::AttachTrustedFitChild(request) => request.validate(),
            Self::InspectAttempt(_)
            | Self::Get(_)
            | Self::Wait(_)
            | Self::Pause(_)
            | Self::Resume(_)
            | Self::Subscribe(_)
            | Self::Attach(_) => Ok(()),
        }
    }
}

impl Request {
    pub fn submit(request: SubmitRequest) -> Self {
        Self::Submit(Box::new(request))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct JobPage {
    pub jobs: Vec<JobRecord>,
    pub next_cursor: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OperationAck {
    pub accepted_at: Timestamp,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamOpened {
    pub stream_id: StreamId,
    pub next_offset: EventOffset,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Response {
    Status(DaemonStatus),
    Submitted(JobRecord),
    Job(Option<JobRecord>),
    Jobs(JobPage),
    Waited(JobRecord),
    Mutated(JobRecord),
    ControlAccepted(OperationAck),
    Subscribed(StreamOpened),
    Attached(StreamOpened),
    TrustedFitChildPlaced(TrustedFitChildPlacement),
    AttemptInspected(Option<AttemptInspection>),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    Unauthorized,
    Forbidden,
    NotFound,
    Conflict,
    SupersededAttempt,
    ResourceExhausted,
    Unavailable,
    DeadlineExceeded,
    Internal,
    ProtocolMismatch,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ServerError {
    pub code: ErrorCode,
    pub message: String,
    pub retry_after_ms: Option<u64>,
    #[serde(default)]
    pub details: BTreeMap<String, String>,
}

/// Local, agent-readable repair guidance derived from a typed protocol error.
/// This is not part of the wire envelope and can evolve without changing frames.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RepairGuidance {
    pub purpose: String,
    pub reason: String,
    pub common_fixes: Vec<String>,
    pub docs_url: String,
    pub repair_hint: String,
}

impl ServerError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retry_after_ms: None,
            details: BTreeMap::new(),
        }
    }

    pub fn unavailable(message: impl Into<String>, retry_after_ms: u64) -> Self {
        Self {
            code: ErrorCode::Unavailable,
            message: message.into(),
            retry_after_ms: Some(retry_after_ms),
            details: BTreeMap::new(),
        }
    }

    pub fn repair_guidance(&self) -> RepairGuidance {
        let common_fixes = match self.code {
            ErrorCode::Unauthorized | ErrorCode::Forbidden => vec![
                "verify the Unix peer identity and delegated owner".to_owned(),
                "do not retry as a broader principal".to_owned(),
            ],
            ErrorCode::Conflict | ErrorCode::SupersededAttempt => vec![
                "fetch the current job and active attempt fence".to_owned(),
                "retry only against the current state".to_owned(),
            ],
            ErrorCode::Unavailable | ErrorCode::ResourceExhausted => vec![
                "honor retry_after_ms when present".to_owned(),
                "inspect daemon status and scheduler capabilities".to_owned(),
            ],
            ErrorCode::InvalidRequest | ErrorCode::ProtocolMismatch => vec![
                "validate the request with the exact protocol version".to_owned(),
                "correct the named field before resubmitting".to_owned(),
            ],
            _ => vec![
                "inspect the job event stream and daemon health".to_owned(),
                "rerun the narrow proof lane named in agent/test-map.json".to_owned(),
            ],
        };
        RepairGuidance {
            purpose: "make an SCQ failure locally diagnosable without hidden daemon state"
                .to_owned(),
            reason: self.message.clone(),
            repair_hint: common_fixes.join("; "),
            common_fixes,
            docs_url: "docs/agent-native-standard.md".to_owned(),
        }
    }
}

impl fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for ServerError {}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RequestEnvelope {
    pub version: ProtocolVersion,
    pub request_id: RequestId,
    pub request: Request,
}

impl RequestEnvelope {
    pub fn new(request: Request) -> Self {
        Self {
            version: ProtocolVersion::CURRENT,
            request_id: RequestId::new(),
            request,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResponseEnvelope {
    pub version: ProtocolVersion,
    pub request_id: RequestId,
    pub result: Result<Response, ServerError>,
}

impl ResponseEnvelope {
    pub fn ok(request_id: RequestId, response: Response) -> Self {
        Self {
            version: ProtocolVersion::CURRENT,
            request_id,
            result: Ok(response),
        }
    }

    pub fn error(request_id: RequestId, error: ServerError) -> Self {
        Self {
            version: ProtocolVersion::CURRENT,
            request_id,
            result: Err(error),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachInput {
    Stdin(Vec<u8>),
    Resize { rows: u16, columns: u16 },
    Control(ControlMessage),
    CloseStdin,
}

impl Validate for AttachInput {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Stdin(data) => {
                if data.len() > 64 * 1024 {
                    Err(ValidationError::new(
                        "attach.stdin",
                        "must be at most 65536 bytes",
                    ))
                } else {
                    Ok(())
                }
            }
            Self::Resize { rows, columns } if *rows == 0 || *columns == 0 => Err(
                ValidationError::new("attach.resize", "rows and columns must be nonzero"),
            ),
            Self::Control(control) => control.validate(),
            Self::Resize { .. } | Self::CloseStdin => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamInputFrame {
    pub stream_id: StreamId,
    pub input: AttachInput,
}

/// Daemon-to-worker frame written on the typed control FD.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerControlFrame {
    pub version: ProtocolVersion,
    pub attempt: crate::FencedAttempt,
    pub sequence: u64,
    pub control: ControlMessage,
}

/// Worker-to-daemon reports written on the typed control FD.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerReport {
    Ready,
    CheckpointAcknowledged {
        checkpoint_id: String,
    },
    CheckpointCommitted(crate::CheckpointEvent),
    Progress(crate::StructuredProgress),
    Metrics(crate::MetricsSnapshot),
    Failure {
        class: FailureClass,
        message: String,
    },
    Application {
        name: String,
        payload: Vec<u8>,
    },
}

impl Validate for WorkerReport {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Ready => Ok(()),
            Self::CheckpointAcknowledged { checkpoint_id } => {
                crate::validation::validate_identifier(
                    "worker_report.checkpoint_id",
                    checkpoint_id,
                    256,
                )
            }
            Self::CheckpointCommitted(checkpoint) => checkpoint.validate(),
            Self::Progress(progress) => progress.validate(),
            Self::Metrics(metrics) => metrics.validate(),
            Self::Failure { message, .. } => crate::validation::validate_text(
                "worker_report.failure.message",
                message,
                4096,
                false,
            ),
            Self::Application { name, payload } => {
                crate::validation::validate_identifier("worker_report.name", name, 128)?;
                if payload.len() > 64 * 1024 {
                    return Err(ValidationError::new(
                        "worker_report.payload",
                        "must be at most 65536 bytes",
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkerReportFrame {
    pub version: ProtocolVersion,
    pub attempt: crate::FencedAttempt,
    pub sequence: u64,
    pub report: WorkerReport,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientFrame {
    Request(Box<RequestEnvelope>),
    StreamInput(StreamInputFrame),
    CloseStream { stream_id: StreamId },
}

impl ClientFrame {
    pub fn request(envelope: RequestEnvelope) -> Self {
        Self::Request(Box::new(envelope))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EventEnvelope {
    pub stream_id: StreamId,
    pub event: EventRecord,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerFrame {
    Response(Box<ResponseEnvelope>),
    Event(EventEnvelope),
    StreamClosed {
        stream_id: StreamId,
        error: Option<ServerError>,
    },
    /// A global stream marker, deliberately outside [`EventRecord`] because a
    /// compacted range is not owned by any one job.
    StreamGap {
        stream_id: StreamId,
        gap: crate::StreamGap,
    },
}

impl ServerFrame {
    pub fn response(envelope: ResponseEnvelope) -> Self {
        Self::Response(Box::new(envelope))
    }
}
