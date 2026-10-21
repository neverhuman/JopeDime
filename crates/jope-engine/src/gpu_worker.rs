//! Resident CUDA ranks. Task clients transfer their existing GPU lease by fd;
//! disconnecting a client terminates only its worker, releasing that lease.
use crate::manifest::*;
#[path = "resident_protocol.rs"]
mod protocol;
use crate::sealed_storage::{receive_fd, send_fd};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    os::{
        fd::AsRawFd,
        unix::{
            fs::{MetadataExt, PermissionsExt},
            net::UnixListener,
        },
    },
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

thread_local! {
    static ACCEPTED_REQUEST: std::cell::RefCell<Option<Value>> = const { std::cell::RefCell::new(None) };
}

pub(crate) struct TaskDeadline {
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

pub(crate) fn deadline_watchdog() -> Result<TaskDeadline> {
    TaskDeadline::start(crate::task_contract::gpu_task_deadline_ms()?)
}

/// Canonical JSON bytes of this value, hashed by the authority itself, bind
/// the common signed demand. Per-rank request IDs remain separate receipts.
pub fn frozen_task_descriptor(request: &Value) -> Result<Value> {
    Ok(serde_json::to_value(protocol::common_task_descriptor(
        request,
    )?)?)
}
impl TaskDeadline {
    fn start(deadline: u64) -> Result<Self> {
        let window = protocol::remaining(deadline)?;
        let end = std::time::Instant::now()
            .checked_add(window)
            .context("task deadline overflow")?;
        let done = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&done);
        let thread = std::thread::spawn(move || {
            while !observed.load(Ordering::Acquire) {
                if std::time::Instant::now() >= end || protocol::remaining(deadline).is_err() {
                    unsafe {
                        libc::_exit(76);
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        });
        Ok(Self {
            done,
            thread: Some(thread),
        })
    }
}
impl Drop for TaskDeadline {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Only the verified service dispatch installs this in-process capability.
/// Caller environment flags alone cannot impersonate an accepted worker task.
pub fn require_accepted_task(root: &Path, task: &Task) -> Result<()> {
    ACCEPTED_REQUEST.with(|active| {
        let active = active.borrow();
        let request = active
            .as_ref()
            .context("GPU task lacks verified resident acceptance")?;
        ensure!(
            request["root"] == json!(root) && request["task"] == serde_json::to_value(task)?,
            "accepted resident task identity changed"
        );
        protocol::remaining(
            request["deadline_unix_ms"]
                .as_u64()
                .context("accepted deadline absent")?,
        )?;
        protocol::verify_request_inputs(request)
    })
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Task {
    Train {
        corpus: PathBuf,
        run: PathBuf,
        launch: PathBuf,
        world: usize,
        micro: usize,
        steps: usize,
        checkpoint_steps: usize,
        resume: bool,
        initialize: Option<PathBuf>,
        #[serde(default)]
        replay_policy: Option<PathBuf>,
        /// Explicit lineage-mode declaration; absent on requests that predate
        /// it, which keep their implicit meaning (W6 follow-up).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        continuation_mode: Option<crate::task_contract::ContinuationMode>,
    },
    Proposal {
        corpus: PathBuf,
        checkpoint: PathBuf,
        output: PathBuf,
        world: usize,
        cuts: usize,
        cut_start: usize,
        shuffled: bool,
    },
    Monitor {
        corpus: PathBuf,
        checkpoint: PathBuf,
        panel: PathBuf,
        output: PathBuf,
    },
}
impl Task {
    fn receipt(&self, rank: usize) -> PathBuf {
        match self {
            Self::Train { launch, .. } => launch.join(format!("complete-{rank}.json")),
            Self::Proposal { output, .. } => output.join(format!("complete-{rank}.json")),
            Self::Monitor { output, .. } => output.clone(),
        }
    }
    fn run(&self, root: &Path, rank: usize) -> Result<()> {
        match self {
            Self::Train {
                corpus,
                run,
                launch,
                world,
                micro,
                steps,
                checkpoint_steps,
                resume,
                initialize,
                replay_policy,
                continuation_mode,
            } => crate::training::rank_with_policy(
                root,
                corpus,
                run,
                launch,
                rank,
                *world,
                *micro,
                *steps,
                *checkpoint_steps,
                *resume,
                initialize.as_deref(),
                replay_policy.as_deref(),
                *continuation_mode,
            ),
            Self::Proposal {
                corpus,
                checkpoint,
                output,
                world,
                cuts,
                cut_start,
                shuffled,
            } => crate::proposals::rank(
                root, corpus, checkpoint, output, rank, *world, *cuts, *cut_start, *shuffled,
            ),
            Self::Monitor {
                corpus,
                checkpoint,
                panel,
                output,
            } => crate::monitoring::evaluate(root, corpus, checkpoint, panel, output),
        }
    }
}

pub fn enabled() -> bool {
    std::env::var_os("JOPE_GPU_POOL").is_some()
        && std::env::var_os("JOPE_GPU_LEASE_FD").is_some()
        && std::env::var_os("JOPE_GPU_WORKER_ACTIVE").is_none()
}

pub fn inherit_lease(command: &mut std::process::Command, lease: &fs::File) -> Result<()> {
    ensure!(
        unsafe { libc::fcntl(lease.as_raw_fd(), libc::F_SETFD, 0) } == 0,
        "cannot inherit GPU lease"
    );
    command.env("JOPE_GPU_LEASE_FD", lease.as_raw_fd().to_string());
    Ok(())
}

/// Public train-rank must carry the controller's lease and immutable run
/// identity before it can initialize CUDA or contact a resident worker.
///
/// Returns the execution-partition version the frozen run names (W7-T06).
/// The controller writes v1; v2 runs only when a frozen run record declares
/// exactly the v2 record of this topology, never by default.
pub fn verify_training_launch(
    root: &Path,
    corpus: &Path,
    run: &Path,
    launch: &Path,
    world: usize,
    micro: usize,
    steps: usize,
    resume: bool,
    initialize: Option<&Path>,
    policy: Option<&Path>,
) -> Result<(fs::File, crate::training::partition::Version)> {
    let lease = inherited_task_lease(root)?;
    let spec: Value = read_json(&run.join("run.json"))?;
    let partition_version =
        crate::training::partition::Version::of_record(&spec["execution_partition"])
            .context("train-rank controller contract changed")?;
    let expected_partition = match partition_version {
        crate::training::partition::Version::V1 => {
            crate::task_contract::training_partition(world, micro, 2048)?
        }
        crate::training::partition::Version::V2 => {
            crate::task_contract::training_partition_v2(world, micro, 2048)?
        }
    };
    let path_hash = |path: &Path| -> Result<String> {
        file_hash(&resolve(
            root,
            if path.is_absolute() {
                path.strip_prefix(root)?
            } else {
                path
            },
        )?)
    };
    let policy_sha = policy.map(path_hash).transpose()?;
    let initialization_sha = initialize
        .map(|path| path_hash(&path.join("state.json")))
        .transpose()?;
    let deadline = protocol::inherited_deadline()?;
    ensure!(
        spec["schema"] == "parallel.training_run.v1"
            && spec["corpus_sha256"] == file_hash(&corpus.join("index.json"))?
            && spec["executable_sha256"] == file_hash(Path::new("/proc/self/exe"))?
            && spec["execution_partition"] == expected_partition
            && spec["step_window"]["requested_step_increment"] == steps
            && spec["replay_policy_sha256"] == json!(policy_sha)
            && (resume || spec["initialization_sha256"] == json!(initialization_sha))
            && spec["deadline_unix_ms"] == deadline,
        "train-rank controller contract changed"
    );
    protocol::input_artifacts(
        root,
        &json!({"kind":"Train", "corpus":corpus, "run":run,
        "launch":launch, "resume":resume, "initialize":initialize, "replay_policy":policy}),
    )?;
    protocol::remaining(deadline)?;
    Ok((lease, partition_version))
}

pub fn inherited_task_lease(root: &Path) -> Result<fs::File> {
    crate::task_contract::require_original_gpu_grant("native-gpu-rank")?;
    protocol::inherited_deadline()?;
    use std::os::fd::FromRawFd;
    let key = if std::env::var_os("JOPE_GPU_WORKER_ACTIVE").is_some() {
        "JOPE_GPU_WORKER_LEASE_FD"
    } else {
        "JOPE_GPU_LEASE_FD"
    };
    let descriptor: i32 = std::env::var(key)
        .context("GPU rank requires its inherited controller lease")?
        .parse()?;
    let copied = unsafe { libc::dup(descriptor) };
    ensure!(copied >= 0, "training lease is unavailable");
    let lease = unsafe { fs::File::from_raw_fd(copied) };
    let actual = lease.metadata()?;
    let expected = root.join("runs/encoding-service.lock").metadata()?;
    ensure!(
        actual.dev() == expected.dev()
            && actual.ino() == expected.ino()
            && lease.try_lock().is_ok(),
        "training lease identity changed"
    );
    Ok(lease)
}

pub fn verify_proposal_launch(root: &Path, task: &Task) -> Result<(fs::File, Value)> {
    let lease = inherited_task_lease(root)?;
    let Task::Proposal {
        corpus,
        checkpoint,
        output,
        world,
        cuts,
        cut_start,
        shuffled,
    } = task
    else {
        anyhow::bail!("proposal controller task required")
    };
    let spec: Value = read_json(&output.join("run.json"))?;
    let deadline = protocol::inherited_deadline()?;
    let champion = PathBuf::from(
        std::env::var_os("JOPE_CHAMPION_SNAPSHOT").context("champion snapshot absent")?,
    );
    ensure!(
        spec["schema"] == "parallel.proposal_run.v1"
            && spec["deadline_unix_ms"] == deadline
            && spec["executable_sha256"] == file_hash(Path::new("/proc/self/exe"))?
            && spec["corpus_sha256"] == file_hash(&corpus.join("index.json"))?
            && spec["checkpoint_sha256"] == file_hash(&checkpoint.join("state.json"))?
            && spec["champion_snapshot_sha256"] == file_hash(&champion)?
            && spec["world"] == *world
            && spec["cuts"] == *cuts
            && spec["cut_start"].as_u64().unwrap_or(0) == *cut_start as u64
            && spec["shuffled"] == *shuffled,
        "proposal controller contract changed"
    );
    Ok((lease, spec))
}

pub fn submit(root: &Path, rank: usize, task: Task) -> Result<()> {
    crate::task_contract::require_original_gpu_grant("resident-submit")?;
    let pool = PathBuf::from(std::env::var_os("JOPE_GPU_POOL").context("resident pool absent")?);
    let lease = std::env::var("JOPE_GPU_LEASE_FD")?.parse::<i32>()?;
    let mut env = serde_json::Map::new();
    for key in [
        "JOPE_CHAMPION_SNAPSHOT",
        "JOPE_GP_EXECUTABLE",
        "JOPE_EXPERIMENT_DIR",
    ] {
        if let Ok(value) = std::env::var(key) {
            env.insert(key.into(), json!(value));
        }
    }
    let executable = file_hash(Path::new("/proc/self/exe"))?;
    let request = json!({"root":root,"rank":rank,"task":task,"env":env,
        "executable_sha256":executable});
    let request = protocol::freeze_request(
        &task.receipt(rank).with_extension("resident-request.json"),
        request,
        protocol::inherited_deadline()?,
    )?;
    let deadline = request["deadline_unix_ms"]
        .as_u64()
        .context("task deadline absent")?;
    protocol::verify_request_inputs(&request)?;
    // Original durable completion can be adopted even after its worker died.
    if protocol::committed(&pool, &request, &task.receipt(rank))?.is_some() {
        return Ok(());
    }
    protocol::ensure_undispatched(&pool, &request)?;
    let mut stream = protocol::connect_rank(
        &pool,
        rank,
        &executable,
        protocol::remaining(deadline)?.min(std::time::Duration::from_secs(30)),
    )?;
    stream.set_read_timeout(Some(protocol::remaining(deadline)?))?;
    stream.set_write_timeout(Some(protocol::remaining(deadline)?))?;
    protocol::verify_request_inputs(&request)?;
    protocol::remaining(deadline)?;
    protocol::authorize_dispatch(&pool, &request)?;
    // No retry is safe from the first attempted descriptor transfer onward.
    let response = (|| -> Result<Value> {
        send_fd(&stream, lease)?;
        protocol::write_until(&mut stream, &serde_json::to_vec(&request)?, deadline)?;
        protocol::write_until(&mut stream, b"\n", deadline)?;
        let response = protocol::read_line_until(&mut stream, 1024 * 1024, deadline)?;
        Ok(serde_json::from_str(&response)?)
    })();
    match response {
        Ok(result) => {
            protocol::remaining(deadline)?;
            protocol::verify_request_inputs(&request)?;
            ensure!(
                result["request_sha256"] == protocol::request_id(&request)?,
                "resident task response changed"
            );
            ensure!(
                result["success"] == true,
                "resident GPU task failed: {}",
                result["error"]
            );
            ensure!(
                protocol::committed(&pool, &request, &task.receipt(rank))? == Some(result),
                "resident response lacks its original committed receipt"
            );
            Ok(())
        }
        Err(error) => {
            protocol::verify_request_inputs(&request)?;
            if protocol::committed(&pool, &request, &task.receipt(rank))?.is_some() {
                return Ok(());
            }
            Err(error.context("resident descriptor transfer attempted; automatic replay refused"))
        }
    }
}

pub fn serve(root: &Path, pool: &Path, rank: usize) -> Result<()> {
    ensure!(
        rank < 8 && pool.starts_with(root.join("runs")),
        "invalid resident worker namespace"
    );
    fs::create_dir_all(pool)?;
    fs::set_permissions(pool, fs::Permissions::from_mode(0o700))?;
    let guard = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(pool.join(format!("rank-{rank}.lock")))?;
    ensure!(guard.try_lock().is_ok(), "resident rank already owned");
    let socket = pool.join(format!("rank-{rank}.sock"));
    if socket.exists() {
        fs::remove_file(&socket)?;
    }
    let listener = UnixListener::bind(&socket)?;
    let executable_sha = file_hash(Path::new("/proc/self/exe"))?;
    let worker = protocol::WorkerIdentity::capture(&socket, executable_sha.clone())?;
    std::env::set_var("JOPE_GPU_WORKER_ACTIVE", "1");
    crate::neural::configure_runtime();
    // Initialize CUDA lazily under the first transferred task lease. A
    // replacement rank must be able to accept its parent's still-held lease;
    // acquiring a new one here would deadlock communicator recovery.
    write_json(
        &pool.join(format!("rank-{rank}.json")),
        &json!({"rank":rank,"status":"idle",
        "pid":std::process::id(),"worker":worker,"timestamp":crate::telemetry::now()}),
    )?;
    for incoming in listener.incoming() {
        write_json(
            &pool.join(format!("rank-{rank}.json")),
            &json!({"rank":rank,"status":"idle",
            "pid":std::process::id(),"worker":worker,"timestamp":crate::telemetry::now()}),
        )?;
        let mut stream = incoming?;
        let peer = protocol::peer_credentials(&stream)?;
        ensure!(
            peer.uid == unsafe { libc::geteuid() },
            "resident task peer rejected"
        );
        let handshake_deadline = protocol::now_ms()?
            .checked_add(30_000)
            .context("handshake deadline overflow")?;
        stream.set_read_timeout(Some(protocol::remaining(handshake_deadline)?))?;
        let lease = receive_fd(&stream)?;
        let file = fs::File::from(lease);
        let expected = root.join("runs/encoding-service.lock").metadata()?;
        let actual = file.metadata()?;
        ensure!(
            expected.dev() == actual.dev()
                && expected.ino() == actual.ino()
                && file.try_lock().is_ok(),
            "task did not transfer the exclusive GPU lease"
        );
        let line = protocol::read_line_until(&mut stream, 1024 * 1024, handshake_deadline)?;
        let request: Value = serde_json::from_str(&line)?;
        crate::task_contract::require_original_gpu_grant("resident-dispatch")?;
        ensure!(
            request["root"] == json!(root)
                && request["rank"] == rank
                && request["executable_sha256"] == executable_sha,
            "resident task root/rank/executable identity changed"
        );
        let deadline = request["deadline_unix_ms"]
            .as_u64()
            .context("resident deadline absent")?;
        // Start before artifact verification and acceptance fsync. The monotonic
        // bound cannot be renewed by a wall-clock rollback during this task.
        let _deadline = TaskDeadline::start(deadline)?;
        protocol::verify_dispatch(pool, &request, peer.pid as u32)?;
        protocol::remaining(deadline)?;
        protocol::verify_request_inputs(&request)?;
        protocol::remaining(deadline)?;
        let task: Task = serde_json::from_value(request["task"].clone())?;
        let id = protocol::request_id(&request)?;
        if let Some(result) = protocol::committed(pool, &request, &task.receipt(rank))? {
            protocol::write_until(&mut stream, &serde_json::to_vec(&result)?, deadline)?;
            protocol::write_until(&mut stream, b"\n", deadline)?;
            continue;
        }
        let accepted = protocol::AcceptedTask::begin(pool, &request, &worker)?;
        stream.set_read_timeout(Some(protocol::remaining(deadline)?))?;
        stream.set_write_timeout(Some(protocol::remaining(deadline)?))?;
        for key in [
            "JOPE_CHAMPION_SNAPSHOT",
            "JOPE_GP_EXECUTABLE",
            "JOPE_EXPERIMENT_DIR",
        ] {
            if let Some(value) = request["env"][key].as_str() {
                std::env::set_var(key, value);
            } else {
                std::env::remove_var(key);
            }
        }
        std::env::set_var("JOPE_GPU_WORKER_LEASE_FD", file.as_raw_fd().to_string());
        std::env::set_var(protocol::DEADLINE_ENV, deadline.to_string());
        write_json(
            &pool.join(format!("rank-{rank}.json")),
            &json!({"rank":rank,"status":"busy",
            "pid":std::process::id(),"worker":worker,"request_sha256":id,"task":task,"timestamp":crate::telemetry::now()}),
        )?;
        let done = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&done);
        let watching = stream.try_clone()?;
        let watcher = std::thread::spawn(move || {
            while !observed.load(Ordering::Acquire) {
                if protocol::remaining(deadline).is_err() {
                    // A stuck collective cannot suppress the task watchdog.
                    // Acceptance remains unknown until the supervisor reconciles it.
                    unsafe {
                        libc::_exit(76);
                    }
                }
                let mut poll = libc::pollfd {
                    fd: watching.as_raw_fd(),
                    events: libc::POLLRDHUP,
                    revents: 0,
                };
                unsafe {
                    libc::poll(&mut poll, 1, 100);
                }
                if poll.revents & (libc::POLLRDHUP | libc::POLLHUP | libc::POLLERR) != 0
                    && !observed.load(Ordering::Acquire)
                {
                    // No live task client remains to supervise collectives. The
                    // pool supervisor restarts this rank from durable receipts.
                    unsafe {
                        libc::_exit(75);
                    }
                }
            }
        });
        ACCEPTED_REQUEST.with(|active| *active.borrow_mut() = Some(request.clone()));
        let result = task.run(root, rank);
        ACCEPTED_REQUEST.with(|active| *active.borrow_mut() = None);
        let result = result.and_then(|()| {
            protocol::remaining(deadline)?;
            protocol::verify_request_inputs(&request)
        });
        done.store(true, Ordering::Release);
        let _ = watcher.join();
        std::env::remove_var("JOPE_GPU_WORKER_LEASE_FD");
        std::env::remove_var(protocol::DEADLINE_ENV);
        let response =
            accepted.finish(&task.receipt(rank), result.err().map(|e| format!("{e:#}")))?;
        protocol::write_until(&mut stream, &serde_json::to_vec(&response)?, deadline)?;
        protocol::write_until(&mut stream, b"\n", deadline)?;
        write_json(
            &pool.join(format!("rank-{rank}.json")),
            &json!({"rank":rank,"status":"idle",
            "pid":std::process::id(),"worker":worker,"timestamp":crate::telemetry::now()}),
        )?;
        ensure!(
            response["success"] == true,
            "failed task requires a fresh CUDA worker"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixStream;
    #[test]
    fn transferred_lease_survives_sender_and_releases_with_worker() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("lease");
        let lease = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&p)
            .unwrap();
        lease.try_lock().unwrap();
        let (sender, receiver) = UnixStream::pair().unwrap();
        send_fd(&sender, lease.as_raw_fd()).unwrap();
        let received = receive_fd(&receiver).unwrap();
        drop(lease);
        let competing = fs::OpenOptions::new().write(true).open(p).unwrap();
        assert!(competing.try_lock().is_err());
        drop(received);
        competing.try_lock().unwrap();
    }
}
