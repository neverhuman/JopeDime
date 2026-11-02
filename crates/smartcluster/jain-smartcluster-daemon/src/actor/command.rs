use super::*;

enum ActorCommand {
    Shutdown(oneshot::Sender<()>),
    ShutdownRetry,
    Request {
        peer: AuthenticatedPeer,
        request: Request,
        sink: ConnectionSink,
        reply: oneshot::Sender<Result<Response, ServerError>>,
    },
    StreamInput {
        peer: AuthenticatedPeer,
        frame: StreamInputFrame,
        reply: oneshot::Sender<Result<(), ServerError>>,
    },
    CloseStream {
        peer: AuthenticatedPeer,
        stream_id: StreamId,
    },
    StreamEnded(StreamId),
    WaitDisconnected {
        job_id: JobId,
        waiter_id: u64,
    },
    ActivateStream {
        peer: AuthenticatedPeer,
        stream_id: StreamId,
    },
    ProcessExited {
        attempt: FencedAttempt,
        status: std::io::Result<ExitStatus>,
    },
    AdoptedDrained {
        attempt: FencedAttempt,
        cgroup: JobCgroup,
    },
    CgroupDrained {
        attempt: FencedAttempt,
        cgroup: JobCgroup,
    },
    WorkerReport(WorkerReportFrame),
    ControlLost(FencedAttempt),
    OutputFailed {
        attempt: FencedAttempt,
        message: String,
    },
    OutputDrained(FencedAttempt),
    ProcessGroupDrained(FencedAttempt),
    Requeue(JobId),
    Deadline {
        job_id: JobId,
        deadline: Timestamp,
    },
    RenewLeases,
    RefreshCounts,
    RetryTransitionEvents,
    Schedule,
    Pressure(PsiEvent),
    PressureQuiet {
        resource: PsiResource,
        generation: u64,
    },
    Gpu(GpuEvent),
    Terminate {
        job_id: JobId,
        attempt: FencedAttempt,
        grace: Duration,
        retry: u8,
    },
    HardKill {
        job_id: JobId,
        attempt: FencedAttempt,
        retry: u8,
    },
    FreezerEffect {
        job_id: JobId,
        attempt: FencedAttempt,
        frozen: bool,
        retry: u8,
    },
}
