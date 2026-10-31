//! Retained prediction has an owned process and a deadline, but no fit token.
use super::*;

const MAX_PREDICTIONS_BYTES: u64 = 1_000_000;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    model: Vec<u8>,
    rows: Vec<Vec<f64>>,
    runtime: String,
    contract: NumericalContract,
    parent: ProcessIdentity,
    deadline_monotonic_ns: u64,
    #[cfg(test)]
    sleep_ms: Option<u64>,
}

/// Private inherited socket only. This command cannot fit or publish quality.
pub fn execute_query_worker() -> Result<()> {
    let deadline = std::env::var("JOPE_QUERY_DEADLINE_MONOTONIC_NS")
        .context("original query deadline absent")?
        .parse::<u64>()?;
    remaining_request(deadline)?;
    let fd: i32 = std::env::var("JOPE_QUERY_CHANNEL_FD")
        .context("original query custody channel absent")?
        .parse()?;
    ensure!(fd > 2, "invalid query custody channel");
    // The environment value can name any open descriptor. Duplicate it before
    // transfer so this worker cannot create a second Rust owner for that number.
    // SAFETY: F_DUPFD_CLOEXEC returns a fresh descriptor with one owner here.
    let channel_fd = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
    ensure!(
        channel_fd >= 0,
        "query custody channel descriptor is not open"
    );
    // SAFETY: `channel_fd` is the fresh descriptor returned above and has not
    // been wrapped or otherwise given another owner.
    let mut channel = unsafe { UnixStream::from_raw_fd(channel_fd) };
    let parent = ProcessIdentity::read(crate::host_os::parent_pid())?;
    ensure!(
        authenticated_peer(&channel)? == parent,
        "query channel is not owned by original parent"
    );
    // Confine this child before it reports readiness and receives inputs.
    super::sandbox::lockdown()?;
    send_json(
        &mut channel,
        &ProcessIdentity::current()?,
        remaining_request(deadline)?,
    )?;
    let descriptor = File::from(receive_fd(&channel, remaining_request(deadline)?)?);
    let bytes = read_sealed(&descriptor, 2 * 1024 * 1024 * 1024)?;
    let input: Input = serde_json::from_slice(&bytes)?;
    ensure!(input.parent == parent, "query original parent changed");
    ensure!(
        input.deadline_monotonic_ns == deadline,
        "query original deadline changed"
    );
    remaining_request(input.deadline_monotonic_ns)?;
    ensure!(
        current_runtime_digest(&input.contract)? == input.runtime,
        "query child loaded runtime changed"
    );
    let model = query::load_retained_model(&input.model)?;
    #[cfg(test)]
    if let Some(ms) = input.sleep_ms {
        std::thread::sleep(Duration::from_millis(ms));
    }
    let predictions = model.predict(&input.rows)?;
    remaining_request(input.deadline_monotonic_ns)?;
    ensure!(
        current_runtime_digest(&input.contract)? == input.runtime,
        "query child loaded runtime changed during prediction"
    );
    let output = serde_json::to_vec(&predictions)?;
    ensure!(
        output.len() as u64 <= MAX_PREDICTIONS_BYTES,
        "oversized query predictions"
    );
    let descriptor = sealed(&output)?;
    send_fd(
        &channel,
        descriptor.as_raw_fd(),
        remaining_request(input.deadline_monotonic_ns)?,
    )?;
    Ok(())
}

pub(super) fn predict(
    state: &State,
    model: &[u8],
    rows: &[Vec<f64>],
    deadline: u64,
) -> Result<query::Prediction> {
    remaining_request(deadline)?;
    let input = sealed(&serde_json::to_vec(&Input {
        model: model.to_vec(),
        rows: rows.to_vec(),
        runtime: state.runtime.clone(),
        contract: state.config.contract.clone(),
        parent: ProcessIdentity::current()?,
        deadline_monotonic_ns: deadline,
        #[cfg(test)]
        sleep_ms: state.config.query_worker_sleep_ms,
    })?)?;
    let (mut channel, child_channel) = UnixStream::pair()?;
    let fd = child_channel.as_raw_fd();
    let parent = std::process::id();
    let confinement =
        super::sandbox::Confinement::query(&state.config, remaining_request(deadline)?)?;
    let attestation = confinement.attestation();
    let mut command = Command::new(&state.config.worker_executable);
    confinement.configure(&mut command);
    #[cfg(not(test))]
    command.arg("registry-service-query");
    #[cfg(test)]
    command.args(&state.config.query_worker_arguments);
    command
        .env("JOPE_QUERY_CHANNEL_FD", fd.to_string())
        .env("JOPE_QUERY_DEADLINE_MONOTONIC_NS", deadline.to_string())
        .env(
            "OMP_NUM_THREADS",
            state.config.contract.omp_threads.to_string(),
        )
        .env(
            "RAYON_NUM_THREADS",
            state.config.contract.rayon_threads.to_string(),
        )
        .env(
            "OPENBLAS_NUM_THREADS",
            state.config.contract.blas_threads.to_string(),
        )
        .env(
            "MKL_NUM_THREADS",
            state.config.contract.mkl_threads.to_string(),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    // SAFETY: the child callback runs before exec and uses only the inherited
    // channel number, integer process controls, and Confinement::enter's
    // syscall-only setup; it does not access parent-owned Rust state.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                || libc::getppid() as u32 != parent
                || libc::setpgid(0, 0) != 0
            {
                return Err(std::io::Error::last_os_error());
            }
            confinement.enter(fd)
        });
    }
    let mut child = command.spawn()?;
    drop(child_channel);
    // Install custody immediately, even when readiness or identity checks fail.
    let process = match ProcessIdentity::read(child.id()) {
        Ok(process) => process,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    };
    let mut original = OriginalChild {
        child,
        process: process.clone(),
        reaped: false,
    };
    let result = (|| {
        let ready: ProcessIdentity = receive_json(&mut channel, remaining_request(deadline)?)?;
        ensure!(ready == process, "query child readiness identity changed");
        attestation.verify(process.pid)?;
        ensure!(
            sha(&fs::read(format!("/proc/{}/exe", process.pid))?)
                == state.identity.executable_sha256
                && super::super::authority::process_runtime_digest(
                    process.pid,
                    &state.config.contract
                )? == state.runtime,
            "query child executable or loaded runtime is unapproved"
        );
        send_fd(&channel, input.as_raw_fd(), remaining_request(deadline)?)?;
        let output = File::from(receive_fd(&channel, remaining_request(deadline)?)?);
        let bytes = read_sealed(&output, MAX_PREDICTIONS_BYTES)?;
        let values: Vec<f64> = serde_json::from_slice(&bytes)?;
        ensure!(
            values.len() == rows.len() && values.iter().all(|v| v.is_finite()),
            "invalid retained predictions"
        );
        let exit = loop {
            remaining_request(deadline)?;
            if let Some(exit) = original.child.try_wait()? {
                break exit;
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        original.reaped = true;
        ensure!(
            exit.success(),
            "original query predictor did not exit successfully"
        );
        Ok(query::Prediction {
            values,
            process,
            exit_code: exit.code(),
        })
    })();
    // OriginalChild kills and reaps the whole owned group on timeout, malformed
    // output, or lost IPC. Its accepted query remains unavailable for replay.
    result.context("original retained predictor failed or exceeded its absolute deadline")
}
