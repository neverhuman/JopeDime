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
    io::{BufRead, BufReader, Read, Write},
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
    if std::env::var_os("JOPE_GPU_POOL").is_some() {
        ensure!(
            unsafe { libc::fcntl(lease.as_raw_fd(), libc::F_SETFD, 0) } == 0,
            "cannot inherit GPU lease"
        );
        command.env("JOPE_GPU_LEASE_FD", lease.as_raw_fd().to_string());
    }
    Ok(())
}

pub fn submit(root: &Path, rank: usize, task: Task) -> Result<()> {
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
    // Original durable completion can be adopted even after its worker died.
    if protocol::committed(&pool, &request, &task.receipt(rank))?.is_some() {
        return Ok(());
    }
    protocol::ensure_undispatched(&pool, &request)?;
    let mut stream =
        protocol::connect_rank(&pool, rank, &executable, std::time::Duration::from_secs(30))?;
    protocol::authorize_dispatch(&pool, &request)?;
    // No retry is safe from the first attempted descriptor transfer onward.
    let response = (|| -> Result<Value> {
        send_fd(&stream, lease)?;
        stream.write_all(&serde_json::to_vec(&request)?)?;
        stream.write_all(b"\n")?;
        let mut response = String::new();
        BufReader::new(stream)
            .take(1024 * 1024)
            .read_line(&mut response)?;
        Ok(serde_json::from_str(&response)?)
    })();
    match response {
        Ok(result) => {
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
    ensure!(
        unsafe { libc::flock(guard.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "resident rank already owned"
    );
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
        stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
        let lease = receive_fd(&stream)?;
        let file = fs::File::from(lease);
        let expected = root.join("runs/encoding-service.lock").metadata()?;
        let actual = file.metadata()?;
        ensure!(
            expected.dev() == actual.dev()
                && expected.ino() == actual.ino()
                && unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "task did not transfer the exclusive GPU lease"
        );
        let mut line = String::new();
        BufReader::new(stream.try_clone()?)
            .take(1024 * 1024)
            .read_line(&mut line)?;
        let request: Value = serde_json::from_str(&line)?;
        ensure!(
            request["root"] == json!(root)
                && request["rank"] == rank
                && request["executable_sha256"] == executable_sha,
            "resident task root/rank/executable identity changed"
        );
        protocol::verify_dispatch(pool, &request, peer.pid as u32)?;
        let task: Task = serde_json::from_value(request["task"].clone())?;
        let id = protocol::request_id(&request)?;
        if let Some(result) = protocol::committed(pool, &request, &task.receipt(rank))? {
            stream.write_all(&serde_json::to_vec(&result)?)?;
            stream.write_all(b"\n")?;
            continue;
        }
        let accepted = protocol::AcceptedTask::begin(pool, &request, &worker)?;
        stream.set_read_timeout(None)?;
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
        let result = task.run(root, rank);
        done.store(true, Ordering::Release);
        let _ = watcher.join();
        std::env::remove_var("JOPE_GPU_WORKER_LEASE_FD");
        let response =
            accepted.finish(&task.receipt(rank), result.err().map(|e| format!("{e:#}")))?;
        stream.write_all(&serde_json::to_vec(&response)?)?;
        stream.write_all(b"\n")?;
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
        assert_eq!(
            unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let (sender, receiver) = UnixStream::pair().unwrap();
        send_fd(&sender, lease.as_raw_fd()).unwrap();
        let received = receive_fd(&receiver).unwrap();
        drop(lease);
        let competing = fs::OpenOptions::new().write(true).open(p).unwrap();
        assert_ne!(
            unsafe { libc::flock(competing.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        drop(received);
        assert_eq!(
            unsafe { libc::flock(competing.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
    }
}
