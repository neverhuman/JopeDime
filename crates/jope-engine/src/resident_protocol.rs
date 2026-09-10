//! CPU-testable custody and immutable completion for resident GPU tasks.
use crate::manifest::{file_hash, hash, read_json, write_json};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    os::{
        fd::AsRawFd,
        unix::{fs::MetadataExt, net::UnixStream},
    },
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub boot_id: String,
    pub start_ticks: u64,
}

pub fn process_identity(pid: u32) -> Result<ProcessIdentity> {
    ensure!(pid > 1, "invalid resident process ID");
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    // comm may contain spaces and parentheses. Field 22 follows the final ')'.
    let fields: Vec<_> = stat
        .rsplit_once(')')
        .context("malformed process stat")?
        .1
        .split_whitespace()
        .collect();
    ensure!(fields.first() != Some(&"Z"), "resident process is a zombie");
    Ok(ProcessIdentity {
        pid,
        boot_id: fs::read_to_string("/proc/sys/kernel/random/boot_id")?
            .trim()
            .to_owned(),
        start_ticks: fields.get(19).context("process birth absent")?.parse()?,
    })
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct WorkerIdentity {
    pub process: ProcessIdentity,
    pub socket_device: u64,
    pub socket_inode: u64,
    pub executable_sha256: String,
}

impl WorkerIdentity {
    pub fn capture(socket: &Path, executable_sha256: String) -> Result<Self> {
        let metadata = socket.metadata()?;
        Ok(Self {
            process: process_identity(std::process::id())?,
            socket_device: metadata.dev(),
            socket_inode: metadata.ino(),
            executable_sha256,
        })
    }

    pub fn verify_connection(&self, socket: &Path, stream: &UnixStream) -> Result<()> {
        let metadata = socket.metadata()?;
        ensure!(
            metadata.dev() == self.socket_device && metadata.ino() == self.socket_inode,
            "resident socket was replaced"
        );
        let peer = peer_credentials(stream)?;
        ensure!(
            peer.uid == unsafe { libc::geteuid() } && peer.pid as u32 == self.process.pid,
            "resident socket peer changed"
        );
        ensure!(
            process_identity(peer.pid as u32)? == self.process,
            "resident process birth changed"
        );
        ensure!(
            file_hash(Path::new(&format!("/proc/{}/exe", peer.pid)))? == self.executable_sha256,
            "resident loaded executable changed"
        );
        Ok(())
    }
}

pub fn peer_credentials(stream: &UnixStream) -> Result<libc::ucred> {
    let mut peer: libc::ucred = unsafe { std::mem::zeroed() };
    let mut length = std::mem::size_of_val(&peer) as libc::socklen_t;
    ensure!(
        unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&mut peer as *mut libc::ucred).cast(),
                &mut length,
            )
        } == 0
            && length as usize == std::mem::size_of_val(&peer),
        "resident peer credentials unavailable"
    );
    Ok(peer)
}

pub fn connect_rank(
    pool: &Path,
    rank: usize,
    executable: &str,
    window: std::time::Duration,
) -> Result<UnixStream> {
    let started = std::time::Instant::now();
    let socket = pool.join(format!("rank-{rank}.sock"));
    loop {
        ensure!(
            started.elapsed() < window,
            "resident rank {rank} readiness deadline elapsed"
        );
        let pool_state = read_json::<Value>(&pool.join("pool.json")).unwrap_or_default();
        let state = read_json::<Value>(&pool.join(format!("rank-{rank}.json"))).unwrap_or_default();
        if pool_state["phase"] == "running" && state["status"] == "idle" {
            if let Ok(worker) = serde_json::from_value::<WorkerIdentity>(state["worker"].clone()) {
                if worker.executable_sha256 == executable
                    && pool_state["processes"][rank.to_string()] == json!(worker.process)
                    && process_identity(worker.process.pid).is_ok_and(|p| p == worker.process)
                {
                    // No descriptor has been transferred, so only ordinary
                    // readiness races are retryable within the original bound.
                    match UnixStream::connect(&socket) {
                        Ok(stream) => {
                            worker.verify_connection(&socket, &stream)?;
                            ensure!(
                                started.elapsed() < window,
                                "resident readiness verification exceeded its deadline"
                            );
                            return Ok(stream);
                        }
                        Err(e)
                            if matches!(
                                e.kind(),
                                std::io::ErrorKind::NotFound
                                    | std::io::ErrorKind::ConnectionRefused
                            ) => {}
                        Err(e) => return Err(e.into()),
                    }
                }
            }
        }
        std::thread::sleep(
            std::time::Duration::from_millis(100).min(window.saturating_sub(started.elapsed())),
        );
    }
}

pub fn request_id(request: &Value) -> Result<String> {
    Ok(hash(&serde_json::to_vec(request)?))
}

pub fn ensure_undispatched(pool: &Path, request: &Value) -> Result<()> {
    ensure!(!pool.join("dispatches").join(request_id(request)?).try_exists()?,
        "resident descriptor transfer was authorized; original completion or reconciliation required");
    Ok(())
}

/// The client records uncertainty before even attempting descriptor transfer.
/// An interrupted first write remains a tombstone and cannot be retried.
pub fn authorize_dispatch(pool: &Path, request: &Value) -> Result<()> {
    let parent = pool.join("dispatches");
    fs::create_dir_all(&parent)?;
    fs::File::open(pool)?.sync_all()?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(parent.join(request_id(request)?))
        .context("resident dispatch was already authorized")?;
    file.write_all(&serde_json::to_vec(
        &json!({"request_sha256":request_id(request)?,
        "client":process_identity(std::process::id())?}),
    )?)?;
    file.sync_all()?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

pub fn verify_dispatch(pool: &Path, request: &Value, peer_pid: u32) -> Result<()> {
    let id = request_id(request)?;
    let dispatch: Value = read_json(&pool.join("dispatches").join(&id))?;
    ensure!(
        dispatch["request_sha256"] == id
            && dispatch["client"] == json!(process_identity(peer_pid)?),
        "resident dispatch client custody changed"
    );
    Ok(())
}

/// A directory is a permanent acceptance tombstone, even if its first write
/// was interrupted. Only a validated committed original result permits reuse.
pub fn committed(pool: &Path, request: &Value, output: &Path) -> Result<Option<Value>> {
    let id = request_id(request)?;
    let directory = pool.join("tasks").join(&id);
    if !directory.try_exists()? {
        return Ok(None);
    }
    ensure!(
        read_json::<Value>(&directory.join("request.json"))? == *request,
        "resident task intent changed"
    );
    let result: Value = read_json(&directory.join("complete.json")).context(
        "resident task was accepted without a committed result; reconciliation required",
    )?;
    ensure!(
        result["schema"] == "parallel.resident_task_completion.v2"
            && result["request_sha256"] == id
            && result["success"] == true,
        "resident completion contract invalid"
    );
    let digest = result["receipt_sha256"]
        .as_str()
        .context("resident output digest absent")?;
    ensure!(
        digest.len() == 64 && file_hash(output)? == digest,
        "committed resident output changed"
    );
    let accepted: Value = read_json(&directory.join("accepted.json"))?;
    ensure!(
        accepted["request_sha256"] == id
            && result["accepted_sha256"] == file_hash(&directory.join("accepted.json"))?,
        "resident completion acceptance binding changed"
    );
    Ok(Some(result))
}

pub struct AcceptedTask {
    directory: PathBuf,
    id: String,
}

impl AcceptedTask {
    pub fn begin(pool: &Path, request: &Value, worker: &WorkerIdentity) -> Result<Self> {
        let id = request_id(request)?;
        let parent = pool.join("tasks");
        fs::create_dir_all(&parent)?;
        fs::File::open(pool)?.sync_all()?;
        let directory = parent.join(&id);
        fs::create_dir(&directory)
            .context("resident task already accepted; automatic replay refused")?;
        fs::File::open(&parent)?.sync_all()?;
        write_json(&directory.join("request.json"), request)?;
        write_json(
            &directory.join("accepted.json"),
            &json!({"schema":"parallel.resident_task_acceptance.v2",
            "request_sha256":id,"worker":worker}),
        )?;
        Ok(Self { directory, id })
    }

    pub fn finish(self, output: &Path, error: Option<String>) -> Result<Value> {
        let response = json!({"schema":"parallel.resident_task_completion.v2",
            "request_sha256":self.id,"success":error.is_none(),"error":error,
            "accepted_sha256":file_hash(&self.directory.join("accepted.json"))?,
            "receipt_sha256":if error.is_none(){Some(file_hash(output)?)}else{None}});
        let path = self.directory.join(if error.is_none() {
            "complete.json"
        } else {
            "failure.json"
        });
        // Custody is unique, and a terminal receipt must never overwrite one.
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(&serde_json::to_vec_pretty(&response)?)?;
        file.sync_all()?;
        fs::File::open(&self.directory)?.sync_all()?;
        Ok(response)
    }
}
