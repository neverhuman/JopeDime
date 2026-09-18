use super::*;

pub(super) fn spawn_output_drains(
    output: WorkloadOutputOwned,
    attempt: FencedAttempt,
    journal: EventJournal,
    sender: mpsc::Sender<ActorCommand>,
) {
    match output {
        WorkloadOutputOwned::Pipes { stdout, stderr } => {
            if let Some(stdout) = stdout {
                spawn_output_reader(
                    attempt,
                    journal.clone(),
                    sender.clone(),
                    stdout,
                    OutputKind::Stdout,
                );
            }
            if let Some(stderr) = stderr {
                spawn_output_reader(attempt, journal, sender, stderr, OutputKind::Stderr);
            }
        }
        WorkloadOutputOwned::Pty(output) => {
            if let Some(output) = output {
                spawn_output_reader(attempt, journal, sender, output, OutputKind::Pty);
            }
        }
    }
}

pub(super) fn output_reader_count(output: &WorkloadOutputOwned) -> u8 {
    match output {
        WorkloadOutputOwned::Pipes { stdout, stderr } => {
            u8::from(stdout.is_some()) + u8::from(stderr.is_some())
        }
        WorkloadOutputOwned::Pty(output) => u8::from(output.is_some()),
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum OutputKind {
    Stdout,
    Stderr,
    Pty,
}

fn spawn_output_reader<R>(
    attempt: FencedAttempt,
    journal: EventJournal,
    sender: mpsc::Sender<ActorCommand>,
    mut reader: R,
    kind: OutputKind,
) where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let result = async {
            let mut buffer = vec![0_u8; OUTPUT_CHUNK_BYTES];
            loop {
                let count = match reader.read(&mut buffer).await {
                    Ok(0) => return Ok::<(), String>(()),
                    Err(error) if output_error_is_eof(kind, &error) => {
                        return Ok::<(), String>(());
                    }
                    Err(error) => return Err(format!("workload output read failed: {error}")),
                    Ok(count) => count,
                };
                let payload = buffer[..count].to_vec();
                let event = match kind {
                    OutputKind::Stdout => JobEvent::Stdout(payload),
                    OutputKind::Stderr => JobEvent::Stderr(payload),
                    OutputKind::Pty => JobEvent::Pty(payload),
                };
                let journal = journal.clone();
                let result = tokio::task::spawn_blocking(move || {
                    journal.append(attempt.job_id, Some(attempt), event, now_timestamp())
                })
                .await;
                match result {
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => {
                        return Err(format!("workload output durability failed: {error}"));
                    }
                    Err(error) => {
                        return Err(format!("workload output journal task failed: {error}"))
                    }
                }
            }
        }
        .await;
        if let Err(message) = result {
            let _ = sender
                .send(ActorCommand::OutputFailed { attempt, message })
                .await;
        }
        let _ = sender.send(ActorCommand::OutputDrained(attempt)).await;
    });
}

fn output_error_is_eof(kind: OutputKind, error: &std::io::Error) -> bool {
    kind == OutputKind::Pty && error.raw_os_error() == Some(nix::libc::EIO)
}

pub(super) fn spawn_process_waiter(
    process: ProcessWaiter,
    attempt: FencedAttempt,
    sender: mpsc::Sender<ActorCommand>,
) {
    tokio::spawn(async move {
        let status = process.wait().await;
        let _ = sender
            .send(ActorCommand::ProcessExited { attempt, status })
            .await;
    });
}

pub(super) fn spawn_control_reports(
    mut receiver: ControlReceiver,
    attempt: FencedAttempt,
    sender: mpsc::Sender<ActorCommand>,
) {
    tokio::spawn(async move {
        loop {
            let frame = match receiver.receive::<WorkerReportFrame>().await {
                Ok(frame) => frame,
                Err(ControlError::Io(error))
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::UnexpectedEof
                            | std::io::ErrorKind::BrokenPipe
                            | std::io::ErrorKind::ConnectionReset
                    ) =>
                {
                    let _ = sender.send(ActorCommand::ControlLost(attempt)).await;
                    return;
                }
                Err(error) => {
                    tracing::warn!(job_id = %attempt.job_id, %error, "worker control reader stopped");
                    let _ = sender.send(ActorCommand::ControlLost(attempt)).await;
                    return;
                }
            };
            if frame.attempt != attempt {
                tracing::warn!(job_id = %attempt.job_id, "discarded superseded control report fence");
                continue;
            }
            if sender
                .send(ActorCommand::WorkerReport(frame))
                .await
                .is_err()
            {
                return;
            }
        }
    });
}

pub(super) fn spawn_adopted_watcher(
    cgroup: JobCgroup,
    attempt: FencedAttempt,
    sender: mpsc::Sender<ActorCommand>,
) {
    spawn_cgroup_watcher(cgroup, attempt, sender, true);
}

pub(super) fn spawn_fresh_cgroup_watcher(
    cgroup: JobCgroup,
    attempt: FencedAttempt,
    sender: mpsc::Sender<ActorCommand>,
) {
    spawn_cgroup_watcher(cgroup, attempt, sender, false);
}

fn spawn_cgroup_watcher(
    cgroup: JobCgroup,
    attempt: FencedAttempt,
    sender: mpsc::Sender<ActorCommand>,
    adopted: bool,
) {
    tokio::spawn(async move {
        let path = cgroup.path.join("cgroup.events");
        let file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(error) => {
                tracing::error!(path = %path.display(), %error, "cannot monitor adopted cgroup");
                return;
            }
        };
        let file = match AsyncFd::with_interest(file, Interest::PRIORITY) {
            Ok(file) => file,
            Err(error) => {
                tracing::error!(path = %path.display(), %error, "cannot register adopted cgroup readiness");
                return;
            }
        };
        loop {
            if !cgroup_is_populated(file.get_ref()).unwrap_or(true) {
                let command = if adopted {
                    ActorCommand::AdoptedDrained { attempt, cgroup }
                } else {
                    ActorCommand::CgroupDrained { attempt, cgroup }
                };
                let _ = sender.send(command).await;
                return;
            }
            match file.ready(Interest::PRIORITY).await {
                Ok(mut readiness) => readiness.clear_ready(),
                Err(error) => {
                    tracing::error!(path = %path.display(), %error, "adopted cgroup watcher failed");
                    return;
                }
            }
        }
    });
}

pub(super) fn spawn_process_group_watcher(
    pid: u32,
    attempt: FencedAttempt,
    sender: mpsc::Sender<ActorCommand>,
) {
    tokio::spawn(async move {
        while process_group_exists(pid) {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let _ = sender
            .send(ActorCommand::ProcessGroupDrained(attempt))
            .await;
    });
}

pub(super) fn cgroup_is_populated(file: &std::fs::File) -> std::io::Result<bool> {
    let mut source = String::new();
    let mut file = file;
    file.seek(SeekFrom::Start(0))?;
    file.read_to_string(&mut source)?;
    match source
        .lines()
        .find_map(|line| line.strip_prefix("populated "))
    {
        Some("1") => Ok(true),
        Some("0") => Ok(false),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "missing or invalid cgroup populated state",
        )),
    }
}

pub(super) async fn send_control(
    active: &mut ActiveWorkload,
    control: jain_smartcluster_core::ControlMessage,
    write_timeout: Duration,
) -> Result<(), ExecutorError> {
    let channel = active
        .control
        .as_mut()
        .ok_or(ExecutorError::NoControlChannel)?;
    active.control_sequence = active.control_sequence.saturating_add(1);
    let frame = WorkerControlFrame {
        version: ProtocolVersion::CURRENT,
        attempt: active.attempt,
        sequence: active.control_sequence,
        control,
    };
    let send = channel.send(&frame);
    match tokio::time::timeout(write_timeout, send).await {
        Ok(result) => result.map_err(ExecutorError::from),
        Err(_) => {
            active.control = None;
            Err(ExecutorError::Control(ControlError::Timeout))
        }
    }
}

pub(super) async fn request_checkpoint(
    active: &mut ActiveWorkload,
    deadline: Duration,
    write_timeout: Duration,
) -> Result<(), ExecutorError> {
    send_control(
        active,
        jain_smartcluster_core::ControlMessage::Checkpoint {
            deadline_ms: deadline.as_millis().try_into().unwrap_or(u64::MAX),
        },
        write_timeout,
    )
    .await
}

pub(super) fn signal_active(
    cgroups: &CgroupManager,
    active: &ActiveWorkload,
    signal: Signal,
) -> Result<(), ExecutorError> {
    if let Some(cgroup) = &active.cgroup {
        cgroups.signal_all(cgroup, signal)?;
    } else if (active.leader_exit.is_some() && process_group_exists(active.pid))
        || (active.leader_exit.is_none() && same_process(active.pid, active.pid_start_time))
    {
        killpg(Pid::from_raw(active.pid as i32), signal)?;
    }
    Ok(())
}

pub(super) fn pid_start_time(pid: u32) -> Option<u64> {
    let source = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_name = source.rsplit_once(") ")?.1;
    after_name.split_whitespace().nth(19)?.parse().ok()
}

pub(super) fn same_process(pid: u32, expected_start: Option<u64>) -> bool {
    expected_start.is_some() && pid_start_time(pid) == expected_start
}

pub(super) fn process_exists(pid: u32) -> bool {
    matches!(
        kill(Pid::from_raw(pid as i32), None),
        Ok(()) | Err(nix::errno::Errno::EPERM)
    )
}

pub(super) fn process_group_exists(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    matches!(
        kill(Pid::from_raw(-pid), None),
        Ok(()) | Err(nix::errno::Errno::EPERM)
    )
}

#[cfg(test)]
mod output_tests {
    use super::*;

    #[test]
    fn pty_eio_after_slave_close_is_normal_eof() {
        let error = std::io::Error::from_raw_os_error(nix::libc::EIO);
        assert!(output_error_is_eof(OutputKind::Pty, &error));
        assert!(!output_error_is_eof(OutputKind::Stdout, &error));
    }
}
