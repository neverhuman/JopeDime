use super::*;

pub(crate) fn exchange(policy: &InstalledPolicy, request: Request) -> Result<Response> {
    policy.revalidate()?;
    let grant = &policy.signed.body;
    authenticated_exchange(
        &grant.daemon_socket,
        grant.daemon_uid,
        &grant.daemon_executable_sha256,
        None,
        request,
    )
}
pub(crate) fn authenticated_exchange(
    socket_path: &Path,
    uid: u32,
    executable_sha256: &str,
    expected: Option<&ProcessIdentity>,
    request: Request,
) -> Result<Response> {
    // Callers include both synchronous native services and async graph/proof
    // drivers. Keep authentication on the actual stream without nesting Tokio
    // runtimes on a runtime thread.
    if tokio::runtime::Handle::try_current().is_ok() {
        return std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    authenticated_exchange(socket_path, uid, executable_sha256, expected, request)
                })
                .join()
                .map_err(|_| anyhow::anyhow!("authenticated exchange thread failed"))?
        });
    }
    request.validate()?;
    let stream = std::os::unix::net::UnixStream::connect(socket_path)?;
    let peer = crate::resident_protocol::peer_credentials(&stream)?;
    let before = ProcessIdentity::read(peer.pid as u32)?;
    // Pin the actual executable inode through hashing and the complete exchange.
    let executable = fs::File::open(format!("/proc/{}/exe", peer.pid))?;
    let pinned = PathBuf::from(format!(
        "/proc/self/fd/{}",
        std::os::fd::AsRawFd::as_raw_fd(&executable)
    ));
    ensure!(
        peer.uid == uid
            && crate::manifest::file_hash(&pinned)? == executable_sha256
            && expected.is_none_or(|value| *value == before)
            && ProcessIdentity::read(peer.pid as u32)? == before,
        "native daemon peer not qualified"
    );
    stream.set_nonblocking(true)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let response = runtime.block_on(async {
        let mut socket = tokio::net::UnixStream::from_std(stream)?;
        let envelope = RequestEnvelope::new(request);
        let id = envelope.request_id;
        let codec = FrameCodec::default();
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            codec
                .write_frame(&mut socket, &ClientFrame::Request(Box::new(envelope)))
                .await?;
            let Some(ServerFrame::Response(response)) =
                codec.read_frame::<ServerFrame, _>(&mut socket).await?
            else {
                anyhow::bail!("native daemon response missing")
            };
            ensure!(
                response.request_id == id
                    && ProtocolVersion::CURRENT.is_compatible_with(response.version),
                "native daemon response identity/protocol mismatch"
            );
            response.result.map_err(anyhow::Error::from)
        })
        .await?
    })?;
    let original_exe = executable.metadata()?;
    let current_exe = fs::metadata(format!("/proc/{}/exe", peer.pid))?;
    ensure!(
        ProcessIdentity::read(peer.pid as u32)? == before
            && original_exe.dev() == current_exe.dev()
            && original_exe.ino() == current_exe.ino(),
        "native daemon changed during exchange"
    );
    Ok(response)
}
pub(crate) fn current_job(policy: &InstalledPolicy, id: JobId) -> Result<JobRecord> {
    match exchange(policy, Request::Get(GetRequest { job_id: id }))? {
        Response::Job(record) => record.context("native allocation missing"),
        _ => anyhow::bail!("native job response differs"),
    }
}

pub(crate) fn active_process(
    socket: &Path,
    uid: u32,
    executable_sha256: &str,
    daemon: Option<&ProcessIdentity>,
    attempt: FencedAttempt,
) -> Result<ProcessIdentity> {
    let response = authenticated_exchange(
        socket,
        uid,
        executable_sha256,
        daemon,
        Request::InspectAttempt(jain_smartcluster::InspectAttemptRequest { attempt }),
    )?;
    let Response::AttemptInspected(Some(inspection)) = response else {
        anyhow::bail!("native active attempt inspection absent")
    };
    inspected_process(inspection, attempt)
}
pub(crate) fn inspected_process(
    inspection: jain_smartcluster::AttemptInspection,
    attempt: FencedAttempt,
) -> Result<ProcessIdentity> {
    inspection.record.validate()?;
    ensure!(
        inspection.record.attempt == attempt
            && matches!(
                inspection.record.state,
                JobState::Starting | JobState::Running
            ),
        "native active attempt inspection is stale/not running"
    );
    let process = inspection
        .active_process
        .context("native original leader is not active")?;
    ensure!(
        inspection.record.pid == Some(process.pid) && inspection.record.finished_at.is_none(),
        "native active process differs from original attempt record"
    );
    let process = ProcessIdentity {
        pid: process.pid,
        birth_ticks: process.birth_ticks,
        boot_id: process.boot_id,
    };
    ensure!(
        ProcessIdentity::read(process.pid)? == process,
        "native original leader identity changed"
    );
    process_parent(process.pid)?;
    ensure!(
        inspection.record.cgroup_path.as_deref() == cgroup(process.pid)?.to_str(),
        "native inspected leader cgroup changed"
    );
    Ok(process)
}
pub(crate) fn native_leader(
    policy: &InstalledPolicy,
    attempt: FencedAttempt,
) -> Result<ProcessIdentity> {
    let grant = &policy.signed.body;
    active_process(
        &grant.daemon_socket,
        grant.daemon_uid,
        &grant.daemon_executable_sha256,
        None,
        attempt,
    )
}

pub(crate) fn process_parent(pid: u32) -> Result<u32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let mut fields = stat
        .rsplit_once(')')
        .context("native ancestry malformed stat")?
        .1
        .split_whitespace();
    let state = fields.next().context("native ancestry state absent")?;
    ensure!(
        !matches!(state, "Z" | "X" | "x"),
        "native requester ancestry process exited"
    );
    fields
        .next()
        .context("native ancestry parent absent")?
        .parse()
        .context("native ancestry parent malformed")
}

pub(crate) fn verify_ancestry(requester: &ProcessIdentity, leader: &ProcessIdentity) -> Result<()> {
    ensure!(
        requester.boot_id == leader.boot_id && ProcessIdentity::read(leader.pid)? == *leader,
        "native leader identity changed"
    );
    let mut current = requester.clone();
    let mut chain: Vec<(ProcessIdentity, u32)> = Vec::new();
    for _ in 0..32 {
        ensure!(
            ProcessIdentity::read(current.pid)? == current,
            "native requester ancestry changed"
        );
        if current == *leader {
            for (process, parent) in chain {
                ensure!(
                    ProcessIdentity::read(process.pid)? == process
                        && process_parent(process.pid)? == parent,
                    "native requester ancestry changed after observation"
                );
            }
            ensure!(
                ProcessIdentity::read(leader.pid)? == *leader,
                "native leader exited during ancestry validation"
            );
            return Ok(());
        }
        let parent = process_parent(current.pid)?;
        ensure!(
            parent > 0 && parent != current.pid,
            "requester is not a native leader descendant"
        );
        chain.push((current, parent));
        current = ProcessIdentity::read(parent)?;
    }
    anyhow::bail!("native requester ancestry exceeds bound")
}

pub(crate) fn cgroup(pid: u32) -> Result<PathBuf> {
    let text = fs::read_to_string(format!("/proc/{pid}/cgroup"))?;
    let name = text
        .lines()
        .find_map(|l| l.strip_prefix("0::"))
        .context("unified native cgroup absent")?;
    Ok(Path::new("/sys/fs/cgroup").join(name.trim_start_matches('/')))
}
pub(crate) fn attempt_path(policy: &InstalledPolicy, attempt: FencedAttempt) -> PathBuf {
    policy.signed.body.cgroup_root.join(format!(
        "scq-{}-attempt-{}-fence-{}",
        attempt.job_id, attempt.number, attempt.fence
    ))
}
pub(crate) fn live(record: &JobRecord) -> Result<FencedAttempt> {
    ensure!(
        record.state == JobState::Running,
        "native allocation not live"
    );
    record.active_attempt.context("native attempt absent")
}
pub(crate) fn service(record: &JobRecord) -> Result<(&str, &[String])> {
    match &record.spec.workload {
        Workload::TrustedService {
            service, arguments, ..
        } => Ok((service.as_str(), arguments)),
        _ => anyhow::bail!("native fit requires fixed trusted service"),
    }
}
pub(crate) fn monotonic_ns() -> Result<u64> {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    ensure!(
        // SAFETY: `t` is a live writable timespec and CLOCK_MONOTONIC has no
        // pointer preconditions beyond that output buffer.
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) } == 0,
        "monotonic clock unavailable"
    );
    (t.tv_sec as u64)
        .checked_mul(1_000_000_000)
        .and_then(|v| v.checked_add(t.tv_nsec as u64))
        .context("clock overflow")
}
