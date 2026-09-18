#![deny(unsafe_code)]

mod event;
mod identity;
mod job;
mod protocol;
mod resource;
mod state;
mod validation;
mod workload;

pub use event::{
    CheckpointEvent, EventOffset, EventRecord, GpuMetrics, JobEvent, MetricsSnapshot, StreamGap,
    StructuredProgress, MAX_EVENT_CHUNK_BYTES,
};
pub use identity::{
    IdempotencyKey, OwnerId, Principal, ProjectId, RequestIdentity, SystemPrincipal, TenantId,
};
pub use job::{
    CheckpointPolicy, FailureClass, IdempotencyScope, IoMode, JobId, JobRecord, JobSpec,
    PreemptionPolicy, PriorityClass, RetryPolicy, TerminalStatus,
};
pub use protocol::{
    AttachInput, AttachRequest, AttachTrustedFitChildRequest, AttemptInspection, CancelMode,
    CancelRequest, ClientFrame, ControlMessage, DaemonMode, DaemonStatus, EffectiveCapacity,
    ErrorCode, EventEnvelope, EventFilter, GetRequest, GpuDeviceStatus, InspectAttemptRequest,
    JobPage, ListFilter, ListRequest, NativeProcessIdentity, OperationAck, PauseRequest,
    PressureLevel, PressureStatus, ProtocolVersion, Request, RequestEnvelope, RequestId, Response,
    ResponseEnvelope, ResumeRequest, SchedulerCapabilities, SendControlRequest, ServerError,
    ServerFrame, StatusRequest, StreamId, StreamInputFrame, StreamOpened, SubmitRequest,
    SubscribeRequest, TrustedFitChildPlacement, WaitRequest, WorkerControlFrame, WorkerReport,
    WorkerReportFrame, PROTOCOL_MAJOR, PROTOCOL_MINOR,
};
pub use resource::{CpuRequest, GpuMode, GpuRequest, MemoryRequest, ResourceRequest};
pub use state::{
    AttemptLease, AttemptNumber, AttemptRecord, FenceToken, FencedAttempt, FencedTransition,
    JobState, Timestamp, TransitionError,
};
pub use validation::{Validate, ValidationError};
pub use workload::{Environment, JainWorkload, ServiceName, Workload};
