use super::{
    frame::{decode_frame, encode_frame},
    process::{self, Containment},
    Attempt, Failure, Request, Response,
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
pub struct PoolConfig {
    pub executable: PathBuf,
    pub workers: usize,
    #[cfg(test)]
    pub(crate) fixture: bool,
    #[cfg(test)]
    pub(crate) wrong_fence: bool,
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
        }
    }
}

type SlotKey = (RunId, u64, u32);
struct Fence {
    attempt: u64,
    active: bool,
}
struct State {
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
            return Response::failed(request.attempt.clone(), Failure::InvalidRequest, 0);
        }
        // Reject oversized/invalid requests before touching a worker or consuming the fence.
        let bytes = match serde_json::to_value(request)
            .map_err(io::Error::other)
            .and_then(|v| encode_frame(&v).map_err(io::Error::other))
        {
            Ok(bytes) => bytes,
            Err(_) => return Response::failed(request.attempt.clone(), Failure::InvalidRequest, 0),
        };
        {
            let mut fences = self.fences.lock().unwrap();
            let key = slot_key(&request.attempt);
            if fences
                .get(&key)
                .is_some_and(|f| f.attempt >= request.attempt.attempt)
            {
                return Response::failed(request.attempt.clone(), Failure::StaleAttempt, 0);
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
                return Response::failed(request.attempt.clone(), Failure::StaleAttempt, 0);
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
                return Response::failed(request.attempt.clone(), Failure::StaleAttempt, elapsed());
            }
            if started.elapsed() >= Duration::from_millis(request.limits.wall_ms) {
                return Response::failed(request.attempt.clone(), Failure::Timeout, elapsed());
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
                            Failure::ProcessExited,
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
                Err(Failure::ProcessExited) if previously_healthy && !retried_idle_exit => {
                    // A worker may retire immediately after its last successful reply. A send
                    // can race that exit. Retry once on a replacement, within the ORIGINAL
                    // deadline and fence. Evaluators are stateless and have no store effects.
                    // Timeout/RSS/panic/protocol failures are never retried here.
                    lease.process = None;
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
    ) -> Result<Response, Failure> {
        process
            .requests
            .send(bytes)
            .map_err(|_| Failure::ProcessExited)?;
        let deadline = Duration::from_millis(request.limits.wall_ms);
        loop {
            if !self.current(&request.attempt) {
                return Err(Failure::StaleAttempt);
            }
            if started.elapsed() >= deadline {
                return Err(Failure::Timeout);
            }
            match crate::sampler::sample(process.child.id()) {
                Ok((_, rss)) => {
                    if rss > request.limits.rss_bytes {
                        return Err(Failure::MemoryLimit);
                    }
                    process.last_rss = rss;
                }
                Err(_) => {
                    if process.child.try_wait().ok().flatten().is_none() {
                        return Err(Failure::Unavailable);
                    }
                }
            }
            let remaining = deadline
                .saturating_sub(started.elapsed())
                .min(SAMPLE_INTERVAL);
            match process.responses.recv_timeout(remaining) {
                Ok(Ok(response)) => {
                    if response.attempt != request.attempt || !response.valid() {
                        return Err(Failure::Protocol);
                    }
                    // Startup samples can precede page faults/matrix load. Capture the actual
                    // post-load RSS before retaining a process; a disappeared worker retires.
                    match crate::sampler::sample(process.child.id()) {
                        Ok((_, rss)) => {
                            if rss > request.limits.rss_bytes {
                                return Err(Failure::MemoryLimit);
                            }
                            process.last_rss = rss;
                        }
                        Err(_) => process.last_rss = 0,
                    }
                    if !self.current(&request.attempt) {
                        return Err(Failure::StaleAttempt);
                    }
                    if started.elapsed() >= deadline {
                        return Err(Failure::Timeout);
                    }
                    return Ok(response);
                }
                Ok(Err(_)) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(Failure::ProcessExited)
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
}

fn slot_key(attempt: &Attempt) -> SlotKey {
    (attempt.run.clone(), attempt.round, attempt.slot)
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

struct Process {
    child: Child,
    // Closing this handle kills all Windows job members; it is never inherited by the child.
    containment: Option<Containment>,
    requests: SyncSender<Vec<u8>>,
    responses: Receiver<io::Result<Response>>,
    pipe_thread: Option<JoinHandle<()>>,
    matrix: PathBuf,
    cap: u64,
    evaluations: u32,
    baseline: u64,
    last_rss: u64,
}

impl Process {
    fn spawn(config: &PoolConfig, request: &Request) -> io::Result<Self> {
        let mut command = Command::new(&config.executable);
        #[cfg(test)]
        if config.fixture {
            command
                .args([
                    "--exact",
                    "worker::tests::child_helper",
                    "--ignored",
                    "--nocapture",
                    "--quiet",
                ])
                .env("JOPE_WORKER_FIXTURE", "worker");
        } else {
            command.args(["__worker", "--protocol", "1"]);
        }
        #[cfg(not(test))]
        command.args(["__worker", "--protocol", "1"]);
        #[cfg(test)]
        if config.wrong_fence {
            command.env("JOPE_WORKER_WRONG_FENCE", "1");
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        process::prepare(&mut command);
        let mut child = command.spawn()?;
        // No request is sent before Job Object assignment. The trusted executable only boots
        // and waits for input before that point; stdin EOF also covers the assignment race.
        let containment = match Containment::assign(&child, request.limits.rss_bytes) {
            Ok(value) => value,
            Err(error) => {
                process::kill(&mut child);
                return Err(error);
            }
        };
        let mut input = child.stdin.take().unwrap();
        let mut output = child.stdout.take().unwrap();
        let (requests, input_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (output_tx, responses) = mpsc::sync_channel(1);
        #[cfg(test)]
        let fixture = config.fixture;
        let pipe_thread = thread::Builder::new()
            .name("worker-pipe".into())
            .spawn(move || {
                #[cfg(test)]
                if fixture {
                    use std::io::{BufRead, BufReader};
                    let mut banner = BufReader::new(output);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        if banner.read_line(&mut line).unwrap_or(0) == 0 {
                            return;
                        }
                        if line.trim() == "WORKER_FIXTURE_READY" {
                            break;
                        }
                    }
                    // No request has been sent, so the buffered reader cannot hold frame bytes.
                    output = banner.into_inner();
                }
                while let Ok(bytes) = input_rx.recv() {
                    let result = input
                        .write_all(&bytes)
                        .and_then(|()| input.flush())
                        .and_then(|()| decode_frame(&mut output).map_err(io::Error::other))
                        .and_then(|v| serde_json::from_value(v).map_err(io::Error::other));
                    let failed = result.is_err();
                    if output_tx.send(result).is_err() || failed {
                        break;
                    }
                }
            });
        let pipe_thread = match pipe_thread {
            Ok(handle) => handle,
            Err(error) => {
                process::kill(&mut child);
                return Err(error);
            }
        };
        Ok(Self {
            child,
            containment: Some(containment),
            requests,
            responses,
            pipe_thread: Some(pipe_thread),
            matrix: request.matrix_path.clone(),
            cap: request.limits.rss_bytes,
            evaluations: 0,
            baseline: 0,
            last_rss: 0,
        })
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        process::kill(&mut self.child);
        self.containment.take();
        // A completed pipe thread can be waiting for its next request; disconnect that queue.
        let (closed_tx, _) = mpsc::sync_channel(1);
        let old = std::mem::replace(&mut self.requests, closed_tx);
        drop(old);
        if let Some(handle) = self.pipe_thread.take() {
            let _ = handle.join();
        }
    }
}
