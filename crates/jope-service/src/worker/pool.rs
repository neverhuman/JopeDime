use super::{
    frame::{decode_frame, encode_frame},
    process::{self, Containment},
    Attempt, Request, Response, WorkerFailure,
};
use jope_api::ids::RunId;
use std::{
    collections::BTreeMap,
    io::{self, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        mpsc::{self, Receiver, SyncSender},
        Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

/// Worker count and limits are resolved in the plan, rather than reinterpreted by the pool.
#[derive(Clone)]
pub struct PoolConfig {
    pub executable: PathBuf,
    pub workers: usize,
    #[cfg(test)]
    pub(crate) fixture: bool,
    #[cfg(test)]
    pub(crate) wrong_fence: bool,
    #[cfg(test)]
    pub(crate) fixture_test: Option<&'static str>,
}

impl PoolConfig {
    pub fn new(executable: PathBuf, workers: usize) -> Self {
        Self {
            executable,
            workers,
            #[cfg(test)]
            fixture: false,
            #[cfg(test)]
            wrong_fence: false,
            #[cfg(test)]
            fixture_test: None,
        }
    }
}

type SlotKey = (RunId, u64, u32);
struct Fence {
    attempt: u64,
    active: bool,
}
struct State {
    live: usize,
    in_flight: usize,
    parked: Vec<Option<PoolLaunch>>,
    available: Vec<usize>,
    processes: Vec<Option<Process>>,
}

/// Blocking bounded pool. Concurrent callers lease independent workers; each candidate failure
/// is returned as a failed evaluation and the next call receives a replacement process.
pub struct WorkerPool {
    config: PoolConfig,
    state: Mutex<State>,
    available: Condvar,
    fences: Mutex<BTreeMap<SlotKey, Fence>>,
}

impl WorkerPool {
    pub fn new(config: PoolConfig) -> io::Result<Self> {
        if config.workers == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "planned worker count must be positive",
            ));
        }
        let state = State {
            live: config.workers,
            in_flight: 0,
            parked: (0..config.workers).map(|_| None).collect(),
            available: (0..config.workers).rev().collect(),
            processes: (0..config.workers).map(|_| None).collect(),
        };
        Ok(Self {
            config,
            state: Mutex::new(state),
            available: Condvar::new(),
            fences: Mutex::new(BTreeMap::new()),
        })
    }

    /// Current enabled slots. Processes in cold slots are still started lazily.
    pub fn live_workers(&self) -> usize {
        self.state.lock().unwrap().live
    }

    /// Apply one absolute governor target at a drained boundary. The owning
    /// controller closes admission before calling; running AND queued evaluations
    /// are refused. An already-applied target is an idempotent acknowledgement.
    /// Dispatcher/fit threads, fences, planned capacity and evaluation limits stay fixed.
    pub fn set_live_workers(&self, workers: usize) -> io::Result<()> {
        let mut state = self.state.lock().unwrap();
        if workers == 0 || workers > self.config.workers || workers.abs_diff(state.live) > 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "worker target must be one slot within the immutable planned capacity",
            ));
        }
        if workers == state.live {
            return Ok(());
        }
        if state.in_flight != 0 {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "worker resize requires all running and queued evaluations to drain",
            ));
        }
        if workers < state.live {
            let index = state.live - 1;
            if let Some(process) = state.processes[index].take() {
                state.parked[index] = Some(PoolLaunch {
                    matrix: process.matrix.clone(),
                    cap: process.cap,
                });
                // On error the slot remains enabled but empty. Retrying the SAME
                // absolute target cannot retire a second slot/process.
                process.retire()?;
            }
            state.available.retain(|&slot| slot != index);
        } else {
            let index = state.live;
            if let Some(launch) = state.parked[index].clone() {
                // Creation happens on its persistent pipe thread, never on this
                // possibly temporary governor caller (Linux PDEATHSIG lifetime).
                state.processes[index] = Some(Process::start(&self.config, launch)?);
            }
            state.available.push(index);
        }
        state.live = workers;
        self.available.notify_all();
        Ok(())
    }

    /// Fence an admitted attempt at pause/cancellation. A later admission must use a higher
    /// controller attempt number; no worker result can reactivate the revoked attempt.
    pub fn revoke(&self, attempt: &Attempt) {
        if let Some(fence) = self.fences.lock().unwrap().get_mut(&slot_key(attempt)) {
            if fence.attempt == attempt.attempt {
                fence.active = false;
            }
        }
    }

    fn current(&self, attempt: &Attempt) -> bool {
        self.fences
            .lock()
            .unwrap()
            .get(&slot_key(attempt))
            .is_some_and(|f| f.attempt == attempt.attempt && f.active)
    }

    pub fn evaluate(&self, request: &Request) -> Response {
        if request.attempt.attempt == 0
            || request.limits.wall_ms == 0
            || request.limits.rss_bytes == 0
        {
            return Response::failed(request.attempt.clone(), WorkerFailure::InvalidRequest, 0);
        }
        // Reject oversized/invalid requests before touching a worker or consuming the fence.
        let bytes = match serde_json::to_value(request)
            .map_err(io::Error::other)
            .and_then(|v| encode_frame(&v).map_err(io::Error::other))
        {
            Ok(bytes) => bytes,
            Err(_) => {
                return Response::failed(request.attempt.clone(), WorkerFailure::InvalidRequest, 0)
            }
        };
        // Include fence admission and waiting callers in the drained-boundary
        // check. This guard outlives the lease, so the process is returned first.
        let _evaluation = {
            self.state.lock().unwrap().in_flight += 1;
            PoolCall { pool: self }
        };
        {
            let mut fences = self.fences.lock().unwrap();
            let key = slot_key(&request.attempt);
            if fences
                .get(&key)
                .is_some_and(|f| f.attempt >= request.attempt.attempt)
            {
                return Response::failed(request.attempt.clone(), WorkerFailure::StaleAttempt, 0);
            }
            fences.insert(
                key,
                Fence {
                    attempt: request.attempt.attempt,
                    active: true,
                },
            );
        }
        let mut state = self.state.lock().unwrap();
        let index = loop {
            if !self.current(&request.attempt) {
                return Response::failed(request.attempt.clone(), WorkerFailure::StaleAttempt, 0);
            }
            if let Some(index) = state.available.pop() {
                break index;
            }
            state = self
                .available
                .wait_timeout(state, SAMPLE_INTERVAL)
                .unwrap()
                .0;
        };
        let mut lease = Lease {
            pool: self,
            index,
            process: state.processes[index].take(),
        };
        drop(state);
        let started = Instant::now();
        let elapsed = || started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        let mut retried_idle_exit = false;
        loop {
            if !self.current(&request.attempt) {
                return Response::failed(
                    request.attempt.clone(),
                    WorkerFailure::StaleAttempt,
                    elapsed(),
                );
            }
            if started.elapsed() >= Duration::from_millis(request.limits.wall_ms) {
                return Response::failed(
                    request.attempt.clone(),
                    WorkerFailure::Timeout,
                    elapsed(),
                );
            }
            // A matrix/cap change requires a fresh process and one new mapping.
            if lease.process.as_mut().is_some_and(|p| {
                p.matrix != request.matrix_path
                    || p.cap != request.limits.rss_bytes
                    || p.child.try_wait().ok().flatten().is_some()
            }) {
                lease.process = None;
            }
            if lease.process.is_none() {
                match Process::spawn(&self.config, request) {
                    Ok(process) => lease.process = Some(process),
                    Err(_) => {
                        return Response::failed(
                            request.attempt.clone(),
                            WorkerFailure::ProcessExited,
                            elapsed(),
                        )
                    }
                }
            }
            let process = lease.process.as_mut().unwrap();
            let previously_healthy = process.evaluations > 0;
            let result = self.exchange(process, request, bytes.clone(), started);
            match result {
                Ok(response) => {
                    process.evaluations += 1;
                    if process.baseline == 0 {
                        process.baseline = process.last_rss;
                    }
                    if response.failure.is_some()
                        || process.evaluations >= 500
                        || process.last_rss == 0
                        || (process.baseline > 0
                            && process.last_rss > process.baseline.saturating_mul(2))
                    {
                        lease.process = None;
                    }
                    return response;
                }
                Err(WorkerFailure::ProcessExited) if previously_healthy && !retried_idle_exit => {
                    // A send can race CLEAN retirement after the preceding response. Only
                    // that verified retirement may retransmit the pending request. A crash must
                    // fail this attempt so the controller allocates a fresh attempt number.
                    // Waiting for the exit and retransmission share the ORIGINAL deadline.
                    let clean = process.retired_after_reply(
                        started,
                        Duration::from_millis(request.limits.wall_ms),
                    );
                    lease.process = None;
                    if !clean {
                        let failure = if !self.current(&request.attempt) {
                            WorkerFailure::StaleAttempt
                        } else if started.elapsed() >= Duration::from_millis(request.limits.wall_ms)
                        {
                            WorkerFailure::Timeout
                        } else {
                            WorkerFailure::ProcessExited
                        };
                        return Response::failed(request.attempt.clone(), failure, elapsed());
                    }
                    retried_idle_exit = true;
                }
                Err(failure) => {
                    lease.process = None;
                    return Response::failed(request.attempt.clone(), failure, elapsed());
                }
            }
        }
    }

    fn exchange(
        &self,
        process: &mut Process,
        request: &Request,
        bytes: Vec<u8>,
        started: Instant,
    ) -> Result<Response, WorkerFailure> {
        process
            .requests
            .send(bytes)
            .map_err(|_| WorkerFailure::ProcessExited)?;
        let deadline = Duration::from_millis(request.limits.wall_ms);
        loop {
            if !self.current(&request.attempt) {
                return Err(WorkerFailure::StaleAttempt);
            }
            if started.elapsed() >= deadline {
                return Err(WorkerFailure::Timeout);
            }
            match crate::sampler::sample(process.child.id()) {
                Ok((_, rss)) => {
                    if rss > request.limits.rss_bytes {
                        return Err(WorkerFailure::MemoryLimit);
                    }
                    process.last_rss = rss;
                }
                Err(_) => {
                    if process.child.try_wait().ok().flatten().is_none() {
                        return Err(WorkerFailure::Unavailable);
                    }
                }
            }
            let remaining = deadline
                .saturating_sub(started.elapsed())
                .min(SAMPLE_INTERVAL);
            match process.responses.recv_timeout(remaining) {
                Ok(Ok(response)) => {
                    if response.attempt != request.attempt || !response.valid() {
                        return Err(WorkerFailure::Protocol);
                    }
                    // Startup samples can precede page faults/matrix load. Capture the actual
                    // post-load RSS before retaining a process; a disappeared worker retires.
                    match crate::sampler::sample(process.child.id()) {
                        Ok((_, rss)) => {
                            if rss > request.limits.rss_bytes {
                                return Err(WorkerFailure::MemoryLimit);
                            }
                            process.last_rss = rss;
                        }
                        Err(_) => process.last_rss = 0,
                    }
                    if !self.current(&request.attempt) {
                        return Err(WorkerFailure::StaleAttempt);
                    }
                    if started.elapsed() >= deadline {
                        return Err(WorkerFailure::Timeout);
                    }
                    return Ok(response);
                }
                Ok(Err(_)) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(WorkerFailure::ProcessExited)
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
}

fn slot_key(attempt: &Attempt) -> SlotKey {
    (attempt.run.clone(), attempt.round, attempt.slot)
}

struct PoolCall<'a> {
    pool: &'a WorkerPool,
}
impl Drop for PoolCall<'_> {
    fn drop(&mut self) {
        self.pool.state.lock().unwrap().in_flight -= 1;
    }
}

#[derive(Clone)]
struct PoolLaunch {
    matrix: PathBuf,
    cap: u64,
}

struct Lease<'a> {
    pool: &'a WorkerPool,
    index: usize,
    process: Option<Process>,
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let mut state = self.pool.state.lock().unwrap();
        state.processes[self.index] = self.process.take();
        state.available.push(self.index);
        self.pool.available.notify_one();
    }
}

mod process;
use process::Process;

#[cfg(test)]
mod resize_tests;
