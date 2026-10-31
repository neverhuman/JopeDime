//! CPU-testable custody and immutable completion for resident GPU tasks.
use crate::manifest::{file_hash, hash, read_json, write_json};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::{fs::MetadataExt, net::UnixStream},
    },
    path::{Path, PathBuf},
};

pub const REQUEST_SCHEMA: &str = "parallel.resident_request.v3";
pub const DEADLINE_ENV: &str = "JOPE_GPU_TASK_DEADLINE_MS";

/// Rank-independent task bytes for an independently signed funding demand.
/// Reservation IDs/signatures and per-rank transport identities are deliberately
/// outside these bytes. Constructing this metadata grants no execution.
pub use crate::task_contract::CommonTaskDescriptor;

pub fn common_task_descriptor(request: &Value) -> Result<CommonTaskDescriptor> {
    verify_request_inputs(request)?;
    let root = PathBuf::from(
        request["root"]
            .as_str()
            .context("common task root absent")?,
    );
    ensure!(
        root.is_absolute() && root.canonicalize()? == root,
        "common task root must be canonical"
    );
    let environment: BTreeMap<String, String> = serde_json::from_value(request["env"].clone())?;
    ensure!(
        environment.keys().all(|key| [
            "JOPE_CHAMPION_SNAPSHOT",
            "JOPE_GP_EXECUTABLE",
            "JOPE_EXPERIMENT_DIR"
        ]
        .contains(&key.as_str())),
        "unbound common task environment"
    );
    let executable_sha256 = request["executable_sha256"]
        .as_str()
        .context("common executable absent")?;
    ensure!(
        executable_sha256.len() == 64
            && executable_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "invalid common executable digest"
    );
    let deadline_unix_ms = request["deadline_unix_ms"]
        .as_u64()
        .context("common original deadline absent")?;
    remaining(deadline_unix_ms)?;
    Ok(CommonTaskDescriptor {
        schema: "parallel.resident_common_task.v1".into(),
        root,
        task: request["task"].clone(),
        environment,
        input_artifacts: serde_json::from_value(request["input_artifacts"].clone())?,
        executable_sha256: executable_sha256.into(),
        deadline_unix_ms,
    })
}

pub fn now_ms() -> Result<u64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

pub fn remaining(deadline_ms: u64) -> Result<std::time::Duration> {
    let left = deadline_ms
        .checked_sub(now_ms()?)
        .filter(|left| *left > 0)
        .context("resident task absolute deadline elapsed")?;
    Ok(std::time::Duration::from_millis(left))
}

pub fn inherited_deadline() -> Result<u64> {
    let deadline = std::env::var(DEADLINE_ENV)
        .context("resident task requires a bound absolute deadline")?
        .parse()?;
    remaining(deadline)?;
    Ok(deadline)
}

pub fn read_line_until(stream: &mut UnixStream, limit: usize, deadline: u64) -> Result<String> {
    let mut bytes = Vec::new();
    loop {
        ensure!(
            bytes.len() < limit,
            "resident IPC message exceeds its bound"
        );
        stream.set_read_timeout(Some(remaining(deadline)?))?;
        let mut block = [0u8; 4096];
        let capacity = block.len().min(limit - bytes.len());
        let count = stream.read(&mut block[..capacity])?;
        ensure!(count > 0, "resident IPC closed before a complete message");
        bytes.extend_from_slice(&block[..count]);
        remaining(deadline)?;
        if let Some(end) = bytes.iter().position(|byte| *byte == b'\n') {
            ensure!(
                end + 1 == bytes.len(),
                "resident IPC contains trailing messages"
            );
            return Ok(String::from_utf8(bytes)?);
        }
    }
}

pub fn write_until(stream: &mut UnixStream, mut bytes: &[u8], deadline: u64) -> Result<()> {
    while !bytes.is_empty() {
        stream.set_write_timeout(Some(remaining(deadline)?))?;
        let written = stream.write(bytes)?;
        ensure!(written > 0, "resident IPC closed during write");
        bytes = &bytes[written..];
    }
    remaining(deadline)?;
    Ok(())
}

fn input_path(root: &Path, path: &Path) -> Result<PathBuf> {
    crate::manifest::resolve(
        root,
        if path.is_absolute() {
            path.strip_prefix(root)?
        } else {
            path
        },
    )
}

/// Bind input payloads as well as the manifests naming them. This deliberately
/// pays the verification read cost until a sealed verifier receipt is qualified.
/// It does not qualify the dynamic-library runtime closure.
pub fn input_artifacts(root: &Path, task: &Value) -> Result<BTreeMap<PathBuf, String>> {
    request_artifacts(root, task, &json!({}))
}

pub(crate) fn request_artifacts(
    root: &Path,
    task: &Value,
    environment: &Value,
) -> Result<BTreeMap<PathBuf, String>> {
    let mut paths: Vec<(PathBuf, Option<String>)> = Vec::new();
    let digest = |value: &Value| -> Result<String> {
        let value = value.as_str().context("input content digest absent")?;
        ensure!(
            value.len() == 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid input content digest"
        );
        Ok(value.to_owned())
    };
    let corpus = input_path(
        root,
        Path::new(task["corpus"].as_str().context("task corpus absent")?),
    )?;
    let index_bytes = fs::read(corpus.join("index.json"))?;
    let index: Value = serde_json::from_slice(&index_bytes)?;
    if task["kind"] == "Train" {
        crate::corpus_projection::verify_admitted_context_closure(&index)?;
    }
    paths.push((corpus.join("index.json"), Some(hash(&index_bytes))));
    paths.push((
        corpus.join("vocabulary.json"),
        Some(digest(&index["vocabulary_sha256"])?),
    ));
    if index["schema"] == "parallel.teaching_shards.v1" {
        paths.push((corpus.join("provenance.json"), None));
        paths.push((
            corpus.join("groups.bin"),
            Some(digest(&index["groups_sha256"])?),
        ));
        for shard in index["shards"].as_array().context("corpus shards absent")? {
            let path = input_path(
                root,
                Path::new(shard["path"].as_str().context("shard path absent")?),
            )?;
            paths.push((path.join("shard.json"), Some(digest(&shard["sha256"])?)));
            paths.push((
                path.join("records.bin"),
                Some(digest(&shard["records_sha256"])?),
            ));
            if task["kind"] == "Proposal" {
                paths.push((
                    path.join("labels.jsonl"),
                    Some(digest(&shard["labels_sha256"])?),
                ));
            }
        }
    } else {
        ensure!(
            index["schema"] == "parallel.teaching_corpus.v1",
            "unsupported corpus input contract"
        );
        paths.push((
            corpus.join("records.bin"),
            Some(digest(&index["records_sha256"])?),
        ));
        if task["kind"] == "Proposal" {
            let bytes = fs::read(corpus.join("provenance.json"))?;
            let provenance: Value = serde_json::from_slice(&bytes)?;
            paths.push((corpus.join("provenance.json"), Some(hash(&bytes))));
            // W1 round 8: the labels digest the index binds; a legacy index
            // without one keeps the provenance's (its labels stay unbound, and
            // teacher_programs refuses them).
            let labels = match index.get("labels_sha256") {
                Some(bound) => digest(bound)?,
                None => digest(&provenance["labels_sha256"])?,
            };
            paths.push((corpus.join("labels.jsonl"), Some(labels)));
        }
    }
    for context in index["contexts"]
        .as_array()
        .context("corpus contexts absent")?
    {
        paths.push((
            input_path(
                root,
                Path::new(context["path"].as_str().context("context path absent")?),
            )?,
            Some(digest(&context["sha256"])?),
        ));
        if task["kind"] == "Proposal" {
            let prepared = input_path(
                root,
                Path::new(
                    context["prepared"]
                        .as_str()
                        .context("prepared path absent")?,
                ),
            )?;
            paths.push((
                prepared.join("prepared.json"),
                Some(digest(&context["prepared_sha256"])?),
            ));
        }
    }
    let checkpoint = match task["kind"].as_str() {
        Some("Train") => {
            let run = input_path(
                root,
                Path::new(task["run"].as_str().context("task run absent")?),
            )?;
            paths.push((run.join("run.json"), None));
            if task["resume"] == true {
                paths.push((run.join("initialization.json"), None));
                let launch = input_path(
                    root,
                    Path::new(task["launch"].as_str().context("task launch absent")?),
                )?;
                let pointer = launch.join("resume-input.json");
                let bytes = fs::read(&pointer)?;
                let latest: Value = serde_json::from_slice(&bytes)?;
                paths.push((pointer, Some(hash(&bytes))));
                let checkpoint = input_path(
                    root,
                    Path::new(
                        latest["checkpoint"]
                            .as_str()
                            .context("resume checkpoint absent")?,
                    ),
                )?;
                paths.push((
                    checkpoint.join("state.json"),
                    Some(digest(&latest["state_sha256"])?),
                ));
                Some(
                    latest["checkpoint"]
                        .as_str()
                        .context("resume checkpoint absent")?
                        .to_owned(),
                )
            } else {
                task["initialize"].as_str().map(str::to_owned)
            }
        }
        Some("Proposal" | "Monitor") => Some(
            task["checkpoint"]
                .as_str()
                .context("task checkpoint absent")?
                .to_owned(),
        ),
        _ => anyhow::bail!("unsupported resident task input contract"),
    };
    if task["kind"] == "Proposal" {
        let output = input_path(
            root,
            Path::new(task["output"].as_str().context("proposal output absent")?),
        )?;
        let bytes = fs::read(output.join("run.json"))?;
        let run: Value = serde_json::from_slice(&bytes)?;
        paths.push((output.join("run.json"), Some(hash(&bytes))));
        let selection = output
            .parent()
            .context("proposal parent absent")?
            .join("coverage-selection.json");
        if !run["coverage_selection_sha256"].is_null() {
            paths.push((selection, Some(digest(&run["coverage_selection_sha256"])?)));
        } else {
            ensure!(
                !selection.try_exists()?,
                "unbound proposal coverage selection"
            );
        }
        let champion = environment["JOPE_CHAMPION_SNAPSHOT"]
            .as_str()
            .context("proposal champion snapshot absent")?;
        paths.push((
            input_path(root, Path::new(champion))?,
            Some(digest(&run["champion_snapshot_sha256"])?),
        ));
    }
    if let Some(checkpoint) = checkpoint {
        let checkpoint = input_path(root, Path::new(&checkpoint))?;
        let bytes = fs::read(checkpoint.join("state.json"))?;
        let state: Value = serde_json::from_slice(&bytes)?;
        paths.push((checkpoint.join("state.json"), Some(hash(&bytes))));
        for (name, field) in [
            ("config.json", "config_sha256"),
            ("model.ot", "model_sha256"),
        ] {
            paths.push((checkpoint.join(name), Some(digest(&state[field])?)));
        }
        if task["kind"] == "Train" {
            for (name, field) in [
                ("optimizer.ot", "optimizer_sha256"),
                ("optimizer.json", "optimizer_metadata_sha256"),
            ] {
                paths.push((checkpoint.join(name), Some(digest(&state[field])?)));
            }
        }
    }
    for field in ["panel", "replay_policy"] {
        if let Some(path) = task[field].as_str() {
            let path = input_path(root, Path::new(path))?;
            if field == "replay_policy" {
                let bytes = fs::read(&path)?;
                let policy: Value = serde_json::from_slice(&bytes)?;
                if policy["schema"] == crate::checkpoint_recovery::CONTINUATION_SCHEMA {
                    for (reference, sha) in
                        crate::checkpoint_recovery::continuation_references(&policy)?
                    {
                        paths.push((input_path(root, &reference)?, Some(sha)));
                    }
                }
                paths.push((path, Some(hash(&bytes))));
            } else {
                paths.push((path, None));
            }
        }
    }
    paths
        .into_iter()
        .map(|(path, expected)| {
            let checked = input_path(root, &path)?;
            let actual = file_hash(&checked)?;
            ensure!(
                expected.is_none_or(|expected| expected == actual),
                "input payload disagrees with its manifest: {}",
                checked.display()
            );
            Ok((checked.strip_prefix(root)?.to_path_buf(), actual))
        })
        .collect()
}

pub fn verify_request_inputs(request: &Value) -> Result<()> {
    ensure!(
        request["schema"] == REQUEST_SCHEMA,
        "resident request protocol is incompatible"
    );
    let root = Path::new(request["root"].as_str().context("request root absent")?);
    let inputs: BTreeMap<PathBuf, String> =
        serde_json::from_value(request["input_artifacts"].clone())?;
    let required = request_artifacts(root, &request["task"], &request["env"])?;
    ensure!(
        !inputs.is_empty() && inputs == required,
        "resident input artifacts changed or required closure is incomplete"
    );
    Ok(())
}

/// Freeze once before any transfer. A later invocation uses the original input
/// identities and deadline, including when latest.json advanced during the task.
pub fn freeze_request(path: &Path, mut proposed: Value, deadline_ms: u64) -> Result<Value> {
    remaining(deadline_ms)?;
    let root = Path::new(proposed["root"].as_str().context("request root absent")?);
    let path = input_path(root, path)?;
    if path.try_exists()? {
        let saved: Value = read_json(&path)?;
        for field in ["root", "rank", "task", "env", "executable_sha256"] {
            ensure!(
                saved[field] == proposed[field],
                "resident frozen request changed: {field}"
            );
        }
        verify_request_inputs(&saved)?;
        remaining(
            saved["deadline_unix_ms"]
                .as_u64()
                .context("original task deadline absent")?,
        )?;
        return Ok(saved);
    }
    proposed["input_artifacts"] = json!(request_artifacts(
        root,
        &proposed["task"],
        &proposed["env"]
    )?);
    proposed["schema"] = json!(REQUEST_SCHEMA);
    proposed["deadline_unix_ms"] = json!(deadline_ms);
    fs::create_dir_all(path.parent().context("request parent absent")?)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    file.write_all(&serde_json::to_vec_pretty(&proposed)?)?;
    file.sync_all()?;
    fs::File::open(path.parent().context("request parent absent")?)?.sync_all()?;
    remaining(deadline_ms)?;
    Ok(proposed)
}

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
        // The authenticated kernel peer credentials are compared without changing identity.
        ensure!(
            peer.uid == crate::host_os::effective_uid() && peer.pid as u32 == self.process.pid,
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
    let mut peer = std::mem::MaybeUninit::<libc::ucred>::uninit();
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    ensure!(
        // SAFETY: stream owns a live socket; peer is aligned writable ucred storage
        // and length advertises its exact capacity. Both success and returned size are
        // checked before the peer value escapes; the syscall retains neither pointer.
        unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                peer.as_mut_ptr().cast(),
                &mut length,
            )
        } == 0
            && length as usize == std::mem::size_of::<libc::ucred>(),
        "resident peer credentials unavailable"
    );
    // SAFETY: getsockopt succeeded and returned the exact ucred size.
    Ok(unsafe { peer.assume_init() })
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
                    match crate::sealed_storage::connect_bounded(
                        &socket,
                        window.saturating_sub(started.elapsed()),
                    ) {
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
