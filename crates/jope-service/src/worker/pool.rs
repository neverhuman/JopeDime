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
    fn retired_after_reply(&mut self, started: Instant, deadline: Duration) -> bool {
        let waiting = Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return status.code() == Some(process::RETIRE_EXIT_CODE),
                Err(_) => return false,
                Ok(None) => {}
            }
            if waiting.elapsed() >= SAMPLE_INTERVAL || started.elapsed() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn retire(mut self) -> io::Result<()> {
        process::kill(&mut self.child);
        if self.child.try_wait()?.is_none() {
            return Err(io::Error::other("retired worker was not reaped"));
        }
        // Drop joins the pipe helper before this operation returns.
        Ok(())
    }

    fn spawn(config: &PoolConfig, request: &Request) -> io::Result<Self> {
        Self::start(
            config,
            PoolLaunch {
                matrix: request.matrix_path.clone(),
                cap: request.limits.rss_bytes,
            },
        )
    }

    fn start(config: &PoolConfig, launch: PoolLaunch) -> io::Result<Self> {
        let config = config.clone();
        let (requests, input_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (output_tx, responses) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let cap = launch.cap;
        let pipe_thread = thread::Builder::new()
            .name("worker-pipe".into())
            .spawn(move || {
                // Linux ties PDEATHSIG to the creating THREAD. This same thread
                // must own startup and remain alive for the entire process lifetime.
                let startup = (|| {
                    let mut command = Command::new(&config.executable);
                    #[cfg(test)]
                    if config.fixture {
                        command
                            .args([
                                "--exact",
                                config.fixture_test.unwrap_or("worker::tests::child_helper"),
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
                    // No request precedes Job Object assignment; the executable
                    // only boots and waits for input at this point.
                    let containment = match Containment::assign(&child, cap) {
                        Ok(value) => value,
                        Err(error) => {
                            process::kill(&mut child);
                            return Err(error);
                        }
                    };
                    let input = child.stdin.take().unwrap();
                    let output = child.stdout.take().unwrap();
                    Ok((child, containment, input, output))
                })();
                let (child, containment, mut input, mut output) = match startup {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                if let Err(error) = ready_tx.send(Ok((child, containment))) {
                    if let Ok((mut child, _containment)) = error.0 {
                        process::kill(&mut child);
                    }
                    return;
                }
                #[cfg(test)]
                if config.fixture {
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
            })?;
        let startup = ready_rx.recv().unwrap_or_else(|_| {
            Err(io::Error::other(
                "worker creator thread exited during startup",
            ))
        });
        let (child, containment) = match startup {
            Ok(value) => value,
            Err(error) => {
                let _ = pipe_thread.join();
                return Err(error);
            }
        };
        Ok(Self {
            child,
            containment: Some(containment),
            requests,
            responses,
            pipe_thread: Some(pipe_thread),
            matrix: launch.matrix,
            cap,
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

#[cfg(test)]
mod resize_tests {
    use super::*;
    use std::{fs, io::Write, sync::Arc};

    fn matrix(root: &std::path::Path) -> PathBuf {
        let mut csv = (0..21)
            .map(|i| format!("x{i}"))
            .chain(["target".into()])
            .collect::<Vec<_>>()
            .join(",")
            .into_bytes();
        csv.push(b'\n');
        csv.extend(include_bytes!(
            "../../../../test-support/datasets/v1/d2-regression/train.csv"
        ));
        let mut table =
            jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
        table.rows.truncate(1); // TRAIN-derived protocol fixture; not a native quality measurement.
        let path = root.join("protocol.jmat");
        fs::write(&path, crate::dataset::jmat::encode(&table).unwrap()).unwrap();
        path
    }
    fn request(matrix: &std::path::Path, slot: u32) -> Request {
        Request {
            attempt: Attempt {
                run: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
                round: 0,
                slot,
                attempt: 1,
            },
            matrix_path: matrix.to_path_buf(),
            fold_spec: serde_json::json!({}),
            dsl: "pid fixture".into(),
            objective: serde_json::json!("protocol fixture"),
            seed: 73,
            limits: super::super::Limits {
                wall_ms: 10000,
                rss_bytes: 256 * 1024 * 1024,
            },
        }
    }
    fn pool(executable: PathBuf, workers: usize) -> Arc<WorkerPool> {
        let mut config = PoolConfig::new(executable, workers);
        config.fixture = true;
        config.fixture_test = Some("worker::pool::resize_tests::resize_child");
        Arc::new(WorkerPool::new(config).unwrap())
    }
    fn until(mut predicate: impl FnMut() -> bool) {
        let start = Instant::now();
        while !predicate() {
            assert!(
                start.elapsed() < Duration::from_secs(8),
                "owned worker boundary timed out"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }
    fn pids(pool: &WorkerPool) -> Vec<u32> {
        pool.state
            .lock()
            .unwrap()
            .processes
            .iter()
            .flatten()
            .map(|p| p.child.id())
            .collect()
    }
    #[cfg(target_os = "linux")]
    fn reaped(pid: u32) {
        let mut status = 0;
        #[allow(unsafe_code)]
        // SAFETY: this PID came from our pool's owned Child; WNOHANG cannot block.
        let result = unsafe { libc::waitpid(pid as libc::pid_t, &mut status, libc::WNOHANG) };
        assert_eq!(result, -1, "worker{pid} was not already reaped");
        assert_eq!(
            io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD)
        );
    }
    #[cfg(not(target_os = "linux"))]
    fn reaped(_: u32) {}
    fn warm(pool: &WorkerPool, matrix: &std::path::Path, root: &std::path::Path) -> Vec<Response> {
        let release = root.join("release");
        thread::scope(|scope| {
            let jobs: Vec<_> = (0..pool.live_workers())
                .map(|slot| {
                    let mut request = request(matrix, slot as u32);
                    request.fold_spec = serde_json::json!({"release":release});
                    scope.spawn(move || pool.evaluate(&request))
                })
                .collect();
            until(|| pool.state.lock().unwrap().available.is_empty());
            fs::write(release, b"go").unwrap();
            jobs.into_iter()
                .map(|job| {
                    let value = job.join().unwrap();
                    assert!(value.failure.is_none(), "{value:?}");
                    value
                })
                .collect()
        })
    }

    #[test]
    #[ignore = "owned resize supervisor child, explicitly launched by pool tests"]
    fn resize_child() {
        assert_eq!(std::env::var("JOPE_WORKER_FIXTURE").unwrap(), "worker");
        println!("WORKER_FIXTURE_READY");
        std::io::stdout().flush().unwrap();
        super::super::serve(1, |request, matrix| {
            if let Some(release) = request.fold_spec["release"].as_str() {
                until(|| std::path::Path::new(release).exists());
            }
            Ok(super::super::Scores {
                values: vec![
                    std::process::id() as f64,
                    matrix.bytes().as_ptr() as usize as f64,
                ],
                predictions_digest: "0".repeat(64),
            })
        })
        .unwrap();
        std::process::exit(0);
    }

    #[test]
    fn physical_pid_ladder_reaps_one_and_preserves_all_other_launches() {
        let directory = tempfile::tempdir().unwrap();
        let matrix = matrix(directory.path());
        let pool = pool(std::env::current_exe().unwrap(), 8);
        let replies = warm(&pool, &matrix, directory.path());
        let initial = pids(&pool);
        assert_eq!(initial.len(), 8);
        let mappings: BTreeMap<_, _> = replies
            .iter()
            .map(|r| (r.scores[0] as u32, r.scores[1]))
            .collect();
        assert_eq!(mappings.len(), 8);
        let mut inventory = vec![serde_json::json!({"target":8,"pids":initial})];
        for target in (1..8).rev().chain(2..=8) {
            let before = pids(&pool);
            let surviving: Vec<_> = pool
                .state
                .lock()
                .unwrap()
                .processes
                .iter()
                .flatten()
                .map(|p| {
                    (
                        p.child.id(),
                        p.pipe_thread.as_ref().unwrap().thread().id(),
                        p.matrix.clone(),
                        p.cap,
                    )
                })
                .collect();
            // The resize caller exits immediately. It must not be the creator
            // whose Linux lifetime owns the newly restored process.
            thread::scope(|scope| {
                scope
                    .spawn(|| pool.set_live_workers(target))
                    .join()
                    .unwrap()
            })
            .unwrap();
            assert_eq!(pool.live_workers(), target);
            let after = pids(&pool);
            assert_eq!(after.len(), target);
            assert_eq!(
                before.iter().filter(|pid| !after.contains(pid)).count(),
                usize::from(target < before.len())
            );
            assert_eq!(
                after.iter().filter(|pid| !before.contains(pid)).count(),
                usize::from(target > before.len())
            );
            for pid in before.iter().filter(|pid| !after.contains(pid)) {
                reaped(*pid);
            }
            let state = pool.state.lock().unwrap();
            for process in state.processes.iter().flatten() {
                assert_eq!(process.cap, 256 * 1024 * 1024);
                assert_eq!(process.matrix, matrix);
                if let Some((_, thread, path, cap)) =
                    surviving.iter().find(|entry| entry.0 == process.child.id())
                {
                    assert_eq!(process.pipe_thread.as_ref().unwrap().thread().id(), *thread);
                    assert_eq!(&process.matrix, path);
                    assert_eq!(process.cap, *cap);
                }
                #[cfg(target_os = "linux")]
                assert!(std::path::Path::new(&format!("/proc/{}", process.child.id())).exists());
            }
            drop(state);
            assert!(pool.set_live_workers(target).is_ok());
            assert_eq!(pids(&pool), after);
            inventory.push(serde_json::json!({"target":target,"pids":after}));
        }
        // Exercise the surviving original slot: it kept the same mapped matrix.
        // All enabled slots are idle; isolate its next lease deterministically.
        let original = initial[0];
        {
            let mut state = pool.state.lock().unwrap();
            state.available.retain(|&slot| slot != 0);
            state.available.push(0);
        }
        let response = pool.evaluate(&request(&matrix, 100));
        assert!(response.failure.is_none());
        assert_eq!(response.scores[0] as u32, original);
        assert_eq!(response.scores[1], mappings[&original]);
        let final_pids = pids(&pool);
        drop(pool);
        for pid in final_pids {
            reaped(pid);
        }
        if let Some(root) = std::env::var_os("JOPE_WRK04_RECEIPTS") {
            let root = PathBuf::from(root);
            fs::create_dir_all(&root).unwrap();
            fs::write(root.join("pid-ladder.json"),serde_json::to_vec_pretty(&serde_json::json!({"kind":"real PID protocol fixture, not NativeSpace S2","one_process_per_action":true,"inventory":inventory})).unwrap()).unwrap();
        }
    }

    #[test]
    fn running_and_queued_calls_refuse_resize_without_consuming_fences() {
        let directory = tempfile::tempdir().unwrap();
        let matrix = matrix(directory.path());
        let pool = pool(std::env::current_exe().unwrap(), 2);
        pool.set_live_workers(1).unwrap();
        let release = directory.path().join("queue-release");
        let mut first = request(&matrix, 0);
        first.fold_spec = serde_json::json!({"release":release});
        let second = request(&matrix, 1);
        thread::scope(|scope| {
            let running = scope.spawn(|| pool.evaluate(&first));
            until(|| {
                let state = pool.state.lock().unwrap();
                state.in_flight == 1 && state.available.is_empty()
            });
            let queued = scope.spawn(|| pool.evaluate(&second));
            until(|| pool.state.lock().unwrap().in_flight == 2);
            assert_eq!(
                pool.set_live_workers(2).unwrap_err().kind(),
                io::ErrorKind::WouldBlock
            );
            assert_eq!(pool.live_workers(), 1);
            fs::write(release, b"go").unwrap();
            assert!(running.join().unwrap().failure.is_none());
            assert!(queued.join().unwrap().failure.is_none());
        });
        assert_eq!(pool.state.lock().unwrap().in_flight, 0);
        pool.set_live_workers(2).unwrap();
        assert_eq!(
            pool.evaluate(&second).failure,
            Some(WorkerFailure::StaleAttempt)
        );
        pool.revoke(&first.attempt);
        assert_eq!(
            pool.evaluate(&first).failure,
            Some(WorkerFailure::StaleAttempt)
        );
        assert_eq!(pool.state.lock().unwrap().in_flight, 0);
    }

    #[test]
    fn absolute_retry_and_invalid_jumps_never_retire_an_extra_process() {
        let directory = tempfile::tempdir().unwrap();
        let matrix = matrix(directory.path());
        let pool = pool(std::env::current_exe().unwrap(), 2);
        warm(&pool, &matrix, directory.path());
        pool.set_live_workers(1).unwrap();
        // A controller can lose the acknowledgement AFTER this effect. Retrying
        // its unchanged absolute target cannot apply another physical retirement.
        let pids_after_lost_ack = pids(&pool);
        pool.set_live_workers(1).unwrap();
        assert_eq!(pids(&pool), pids_after_lost_ack);
        for invalid in [0, 3, usize::MAX] {
            assert_eq!(
                pool.set_live_workers(invalid).unwrap_err().kind(),
                io::ErrorKind::InvalidInput
            );
            assert_eq!(pids(&pool), pids_after_lost_ack);
        }
        let cold = WorkerPool::new(PoolConfig::new(std::env::current_exe().unwrap(), 8)).unwrap();
        assert_eq!(
            cold.set_live_workers(6).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(cold.live_workers(), 8);
    }

    #[cfg(unix)]
    #[test]
    fn failed_restore_keeps_the_target_retryable_and_joins_its_failed_creator() {
        let directory = tempfile::tempdir().unwrap();
        let matrix = matrix(directory.path());
        let executable = directory.path().join("owned-worker");
        std::os::unix::fs::symlink(std::env::current_exe().unwrap(), &executable).unwrap();
        let pool = pool(executable.clone(), 2);
        warm(&pool, &matrix, directory.path());
        pool.set_live_workers(1).unwrap();
        let before = pids(&pool);
        fs::remove_file(&executable).unwrap();
        assert!(pool.set_live_workers(2).is_err());
        assert_eq!(pool.live_workers(), 1);
        assert_eq!(pids(&pool), before);
        std::os::unix::fs::symlink(std::env::current_exe().unwrap(), &executable).unwrap();
        pool.set_live_workers(2).unwrap();
        assert_eq!(pool.live_workers(), 2);
        let after = pids(&pool);
        assert!(before.iter().all(|pid| after.contains(pid)));
        drop(pool);
        for pid in after {
            reaped(pid);
        }
    }
}
