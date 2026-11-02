use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use jain_smartcluster_core::{
    AttachInput, CancelMode, ErrorCode, EventEnvelope, EventFilter, EventOffset, FailureClass,
    FencedAttempt, FencedTransition, JobEvent, JobId, JobPage, JobRecord, JobState, ListFilter,
    ListRequest, OperationAck, Principal, ProtocolVersion, Request, Response, ServerError,
    ServerFrame, StreamGap, StreamId, StreamInputFrame, StreamOpened, SubmitRequest,
    TerminalStatus, Timestamp, Validate, WorkerControlFrame, WorkerReport, WorkerReportFrame,
};
use nix::sys::signal::{kill, killpg, Signal};
use nix::unistd::Pid;
use tokio::io::{unix::AsyncFd, AsyncRead, AsyncReadExt, AsyncWriteExt, Interest};
use tokio::sync::{mpsc, oneshot, watch};

use crate::auth::{AuthenticatedPeer, Authorizer, PeerIdentity};
use crate::capacity::EffectiveCapacity;
use crate::cgroup::{CgroupManager, JobCgroup};
use crate::config::ExecutionConfig;
use crate::control::{ControlReceiver, ControlSender, DaemonControlFrameError};
use crate::event_log::{EventLog, EventLogError, LogItem};
use crate::executor::{
    ExecutionAllocation, Executor, ExecutorError, ProcessWaiter, RunningParts, WorkloadIo,
    WorkloadOutputOwned, WorkloadResolver,
};
use crate::fanout::{EventFanout, FanoutDelivery, FanoutSubscriber};
use crate::gpu::{GpuEvent, GpuInventory, MpsController};
use crate::health::RuntimeHealth;
use crate::overload::{OverloadController, OverloadStage, PressureLevel};
use crate::psi::{PsiEvent, PsiResource};
use crate::scheduler::{
    Allocation, EffectiveCapacity as SchedulerCapacity, GpuAllocationMode, GpuCompatibility,
    GpuDeviceKind, GpuLearningSnapshot, RunningAllocation, Scheduler, SchedulingCandidate,
    SchedulingSnapshot,
};
use crate::server::{ConnectionSink, RequestHandler};
use crate::store::{Store, StoreError, TransitionMutation, TransitionReceipt};

mod fit_child;
mod mps;
mod pressure;
mod recovery;
mod requests;
mod runtime;
mod scheduling;
mod state;
mod streams;
mod support;
#[cfg(test)]
mod tests;

use runtime::*;
use streams::*;
pub use support::ActorStartError;
use support::*;

mod api;
mod command;
pub use api::{DaemonHandle, DaemonServices};
use command::ActorCommand;

const ACTOR_CAPACITY: usize = 1_024;
const EVENT_READ_BATCH: usize = 256;
const OUTPUT_CHUNK_BYTES: usize = 64 * 1024;
const ATTEMPT_LEASE_MS: u64 = 30_000;
const ATTEMPT_LEASE_RENEW_MS: u64 = 10_000;
const MAX_ADMISSIONS_PER_PASS: usize = 64;
const MAX_WAITERS: usize = 256;
const MAX_WAITERS_PER_PEER: usize = 16;

struct DaemonActor {
    store: Arc<Store>,
    journal: EventJournal,
    authorizer: Authorizer,
    executor: Executor,
    resolver: WorkloadResolver,
    cgroups: CgroupManager,
    capacity: EffectiveCapacity,
    gpu: GpuInventory,
    mps_controller: Option<MpsController>,
    gpu_quarantine: BTreeSet<String>,
    gpu_global_quarantine: bool,
    health: RuntimeHealth,
    execution: ExecutionConfig,
    overload: OverloadController,
    recovery_quiet: Duration,
    pressure_generations: [u64; 3],
    pressure_levels: [PressureLevel; 3],
    scheduler: Scheduler,
    active: HashMap<JobId, ActiveWorkload>,
    shutdown_replies: Vec<oneshot::Sender<()>>,
    shutdown_started: BTreeSet<JobId>,
    queued_jobs: u64,
    queued_by_tenant: BTreeMap<jain_smartcluster_core::TenantId, u64>,
    max_queued_jobs: usize,
    max_queued_jobs_per_tenant: usize,
    tenant_weights: BTreeMap<jain_smartcluster_core::TenantId, u32>,
    streams: HashMap<StreamId, StreamRegistration>,
    waiters: HashMap<JobId, Vec<WaitRegistration>>,
    next_waiter_id: u64,
    receiver: mpsc::Receiver<ActorCommand>,
    sender: mpsc::Sender<ActorCommand>,
    schedule_pending: Arc<AtomicBool>,
}

struct ActiveWorkload {
    attempt: FencedAttempt,
    pid: u32,
    pid_start_time: Option<u64>,
    cgroup: Option<JobCgroup>,
    control: Option<ControlSender>,
    control_sequence: u64,
    io: Option<WorkloadIo>,
    resources: jain_smartcluster_core::ResourceRequest,
    priority: jain_smartcluster_core::PriorityClass,
    allocation: Allocation,
    tenant: jain_smartcluster_core::TenantId,
    preemptible: bool,
    started_at: Timestamp,
    fingerprint: Option<String>,
    max_observed_vram_bytes: u64,
    shared_cuda_oom: bool,
    pending_failure: Option<(FailureClass, String)>,
    mps_managed: bool,
    overload_paused: bool,
    cancelling: bool,
    leader_exit: Option<std::io::Result<ExitStatus>>,
    output_drains_pending: u8,
    control_drained: bool,
    process_tree_drained: bool,
    tree_watcher_started: bool,
}

struct StreamRegistration {
    peer: PeerIdentity,
    job_id: Option<JobId>,
    read_only: bool,
    cancel: watch::Sender<bool>,
    activate: Option<oneshot::Sender<()>>,
}

struct WaitRegistration {
    id: u64,
    peer: PeerIdentity,
    reply: oneshot::Sender<Result<Response, ServerError>>,
}

impl DaemonActor {
    async fn run(mut self, startup: oneshot::Sender<Result<(), String>>) {
        if let Err(error) = self.execution.resource_policy.validate(&self.capacity.cpus) {
            let _ = startup.send(Err(error));
            return;
        }
        if !self.execution.resource_policy.service_cpu_sets.is_empty()
            && self.cgroups.mode() == crate::cgroup::IsolationMode::SingleAllocation
        {
            let _ = startup.send(Err("service CPU pools require delegated cgroups".to_owned()));
            return;
        }
        if let Err(error) = self.recover().await {
            let _ = startup.send(Err(error.to_string()));
            return;
        }
        if let Err(error) = self.initialize_counts() {
            let _ = startup.send(Err(error.to_string()));
            return;
        }
        let _ = startup.send(Ok(()));
        self.schedule_count_refresh();
        self.schedule_lease_renewal();
        if self.execution.resource_policy.enabled() {
            let sender = self.sender.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    if sender.send(ActorCommand::Schedule).await.is_err() {
                        break;
                    }
                }
            });
        }

        self.request_schedule();
        while let Some(command) = self.receiver.recv().await {
            match command {
                ActorCommand::Shutdown(reply) => {
                    self.shutdown_replies.push(reply);
                    self.begin_shutdown();
                }
                ActorCommand::ShutdownRetry => self.begin_shutdown(),
                ActorCommand::Request {
                    peer,
                    request,
                    sink,
                    reply,
                } => {
                    if let Request::Wait(request) = request {
                        match self.authorized_job(&peer, request.job_id) {
                            Ok(job) if job.state.is_terminal() => {
                                let _ = reply.send(Ok(Response::Waited(job)));
                            }
                            Ok(_) => {
                                self.prune_waiters();
                                let global = self.waiters.values().map(Vec::len).sum::<usize>();
                                let peer_count = self
                                    .waiters
                                    .values()
                                    .flatten()
                                    .filter(|waiter| waiter.peer == peer.identity)
                                    .count();
                                if global >= MAX_WAITERS || peer_count >= MAX_WAITERS_PER_PEER {
                                    let _ = reply.send(Err(ServerError::new(
                                        ErrorCode::ResourceExhausted,
                                        "concurrent wait limit exceeded; poll get or retry later",
                                    )));
                                } else if !reply.is_closed() {
                                    self.next_waiter_id = self.next_waiter_id.wrapping_add(1);
                                    let waiter_id = self.next_waiter_id;
                                    self.waiters.entry(request.job_id).or_default().push(
                                        WaitRegistration {
                                            id: waiter_id,
                                            peer: peer.identity,
                                            reply,
                                        },
                                    );
                                    let sender = self.sender.clone();
                                    tokio::spawn(async move {
                                        sink.disconnected().await;
                                        let _ = sender
                                            .send(ActorCommand::WaitDisconnected {
                                                job_id: request.job_id,
                                                waiter_id,
                                            })
                                            .await;
                                    });
                                }
                            }
                            Err(error) => {
                                let _ = reply.send(Err(error));
                            }
                        }
                    } else {
                        let result = self.handle_request(&peer, request, sink).await;
                        let schedule = matches!(result, Ok(Response::Submitted(_)));
                        let _ = reply.send(result);
                        if schedule {
                            self.request_schedule();
                        }
                    }
                }
                ActorCommand::StreamInput { peer, frame, reply } => {
                    let result = self.handle_stream_input(&peer, frame).await;
                    let _ = reply.send(result);
                }
                ActorCommand::CloseStream { peer, stream_id } => {
                    if self
                        .streams
                        .get(&stream_id)
                        .is_some_and(|registration| registration.peer == peer.identity)
                    {
                        if let Some(registration) = self.streams.remove(&stream_id) {
                            let _ = registration.cancel.send(true);
                        }
                    }
                }
                ActorCommand::StreamEnded(stream_id) => {
                    self.streams.remove(&stream_id);
                }
                ActorCommand::WaitDisconnected { job_id, waiter_id } => {
                    if let Some(waiters) = self.waiters.get_mut(&job_id) {
                        waiters.retain(|waiter| waiter.id != waiter_id);
                        if waiters.is_empty() {
                            self.waiters.remove(&job_id);
                        }
                    }
                }
                ActorCommand::ActivateStream { peer, stream_id } => {
                    if self
                        .streams
                        .get(&stream_id)
                        .is_some_and(|registration| registration.peer == peer.identity)
                    {
                        if let Some(activate) = self
                            .streams
                            .get_mut(&stream_id)
                            .and_then(|registration| registration.activate.take())
                        {
                            let _ = activate.send(());
                        }
                    }
                }
                ActorCommand::ProcessExited { attempt, status } => {
                    if let Err(error) = self.observe_process_exit(attempt, status) {
                        tracing::error!(job_id = %attempt.job_id, %error, "process completion failed");
                    }
                }
                ActorCommand::AdoptedDrained { attempt, cgroup } => {
                    if let Err(error) = self.adopted_drained(attempt, &cgroup) {
                        tracing::error!(job_id = %attempt.job_id, %error, "adopted process recovery failed");
                    }
                }
                ActorCommand::CgroupDrained { attempt, cgroup } => {
                    if let Err(error) = self.observe_fresh_cgroup_drained(attempt, &cgroup) {
                        tracing::error!(job_id = %attempt.job_id, %error, "cgroup completion failed");
                    }
                }
                ActorCommand::ProcessGroupDrained(attempt) => {
                    if let Err(error) = self.observe_process_group_drained(attempt) {
                        tracing::error!(job_id = %attempt.job_id, %error, "process-group completion failed");
                    }
                }
                ActorCommand::ControlLost(attempt) => {
                    if let Some(active) = self.active.get_mut(&attempt.job_id) {
                        if active.attempt == attempt {
                            active.control_drained = true;
                        }
                    }
                    if let Err(error) =
                        self.begin_control_recovery(attempt, FailureClass::Infrastructure)
                    {
                        tracing::error!(job_id = %attempt.job_id, %error, "resident control recovery failed");
                    }
                    if let Err(error) = self.finish_process_if_drained(attempt) {
                        tracing::error!(job_id = %attempt.job_id, %error, "resident report drain completion failed");
                    }
                }
                ActorCommand::WorkerReport(frame) => {
                    if let Err(error) = self.handle_worker_report(frame).await {
                        tracing::warn!(%error, "worker control report rejected");
                    }
                }
                ActorCommand::OutputFailed { attempt, message } => {
                    if let Err(error) = self.fail_output(attempt, message) {
                        tracing::error!(job_id = %attempt.job_id, %error, "durable output failure handling failed");
                    }
                }
                ActorCommand::OutputDrained(attempt) => {
                    if let Err(error) = self.observe_output_drained(attempt) {
                        tracing::error!(job_id = %attempt.job_id, %error, "output-drain completion failed");
                    }
                }
                ActorCommand::Requeue(job_id) => {
                    if let Err(error) = self.requeue(job_id) {
                        tracing::error!(%job_id, %error, "retry requeue failed");
                    }
                    self.request_schedule();
                }
                ActorCommand::Deadline { job_id, deadline } => {
                    if let Err(error) = self.enforce_deadline(job_id, deadline) {
                        tracing::error!(%job_id, %error, "deadline enforcement failed");
                    }
                    self.request_schedule();
                }
                ActorCommand::RenewLeases => {
                    self.renew_live_leases();
                    self.schedule_lease_renewal();
                }
                ActorCommand::RefreshCounts => {
                    self.publish_counts();
                    self.schedule_count_refresh();
                }
                ActorCommand::RetryTransitionEvents => {
                    if let Err(error) = self.retry_transition_events() {
                        tracing::error!(%error, "transition event outbox retry failed");
                        self.schedule_transition_event_retry();
                    }
                }
                ActorCommand::Schedule => {
                    self.schedule_pending.store(false, Ordering::Release);
                    if let Err(error) = self.schedule_one().await {
                        tracing::error!(%error, "scheduling pass failed");
                    }
                }
                ActorCommand::Pressure(event) => self.handle_pressure(event).await,
                ActorCommand::PressureQuiet {
                    resource,
                    generation,
                } => self.handle_pressure_quiet(resource, generation).await,
                ActorCommand::Gpu(event) => {
                    if let Err(error) = self.handle_gpu_event(event).await {
                        tracing::error!(%error, "GPU event recovery failed");
                    }
                }
                ActorCommand::Terminate {
                    job_id,
                    attempt,
                    grace,
                    retry,
                } => {
                    if let Err(error) = self.handle_terminate(job_id, attempt, grace).await {
                        tracing::error!(%job_id, %error, "peer-safe termination failed closed");
                        self.retry_termination(job_id, attempt, grace, retry);
                    }
                }
                ActorCommand::HardKill {
                    job_id,
                    attempt,
                    retry,
                } => {
                    if let Err(error) = self.handle_hard_kill(job_id, attempt).await {
                        tracing::error!(%job_id, %error, "peer-safe hard kill failed closed");
                        self.retry_hard_kill(job_id, attempt, retry);
                    }
                }
                ActorCommand::FreezerEffect {
                    job_id,
                    attempt,
                    frozen,
                    retry,
                } => {
                    if let Err(error) = self.apply_freezer(job_id, attempt, frozen) {
                        tracing::error!(%job_id, %error, frozen, "freezer effect failed; retrying");
                        self.retry_freezer(job_id, attempt, frozen, retry);
                    }
                }
            }
            if !self.shutdown_replies.is_empty() && self.active.is_empty() {
                // An in-memory removal alone is not proof that its durable transition committed.
                let durable_drained = self.store.all_jobs().is_ok_and(|jobs| {
                    jobs.iter().all(|job| {
                        job.active_attempt.is_none()
                            || !matches!(
                                job.state,
                                JobState::Starting
                                    | JobState::Running
                                    | JobState::Paused
                                    | JobState::Suspended
                            )
                    })
                });
                if durable_drained && self.retry_transition_events().is_ok() {
                    for reply in self.shutdown_replies.drain(..) {
                        let _ = reply.send(());
                    }
                    break;
                }
                self.schedule_transition_event_retry();
            }
        }
    }

    fn begin_shutdown(&mut self) {
        // Finite jobs finish naturally under their existing deadlines. Only the
        // resident contract opts into EOF checkpoint/restart semantics.
        let attempts: Vec<_> = self.active.values().map(|active| active.attempt).collect();
        let mut retry = false;
        for attempt in attempts {
            if self.shutdown_started.contains(&attempt.job_id) {
                continue;
            }
            match self.begin_control_recovery(attempt, FailureClass::DaemonRestart) {
                Ok(()) => {
                    self.shutdown_started.insert(attempt.job_id);
                }
                Err(error) => {
                    tracing::error!(%error, "shutdown checkpoint initiation failed closed; retrying");
                    retry = true;
                }
            }
        }
        if retry {
            let sender = self.sender.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let _ = sender.send(ActorCommand::ShutdownRetry).await;
            });
        }
    }
}
