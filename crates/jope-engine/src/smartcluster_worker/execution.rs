use super::*;

fn read_frame(stream: &mut UnixStream) -> Result<WorkerControlFrame> {
    let mut header = [0; 4];
    stream.read_exact(&mut header)?;
    let length = u32::from_be_bytes(header) as usize;
    ensure!(length <= FRAME_LIMIT, "control frame too large");
    let mut bytes = vec![0; length];
    stream.read_exact(&mut bytes)?;
    let (frame, remaining): (WorkerControlFrame, &[u8]) = postcard::take_from_bytes(&bytes)?;
    ensure!(remaining.is_empty(), "trailing control bytes");
    Ok(frame)
}
fn report(
    stream: &mut UnixStream,
    attempt: FencedAttempt,
    sequence: &mut u64,
    report: WorkerReport,
) -> Result<()> {
    report.validate()?;
    *sequence = sequence
        .checked_add(1)
        .context("report sequence overflow")?;
    let bytes = postcard::to_stdvec(&WorkerReportFrame {
        version: ProtocolVersion::CURRENT,
        attempt,
        sequence: *sequence,
        report,
    })?;
    ensure!(bytes.len() <= FRAME_LIMIT, "report too large");
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    stream.write_all(&(bytes.len() as u32).to_be_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

fn within_allocation(child: &TaskManifest, parent: &TaskManifest) -> Result<()> {
    ensure!(
        child.kind == TaskKind::ResidentGpu
            && child.checkpoint_allowance_ms.unwrap_or(1000)
                <= parent.checkpoint_allowance_ms.unwrap_or(1000)
            && child.run == parent.run
            && child.deadline <= parent.deadline
            && child.resources.cpu.maximum_millicores <= parent.resources.cpu.maximum_millicores
            && child.resources.ram.hard_bytes <= parent.resources.ram.hard_bytes
            && child.resources.scratch_bytes <= parent.resources.scratch_bytes
            && child.resources.gpu == parent.resources.gpu,
        "resident subtask exceeds native allocation"
    );
    Ok(())
}

/// Persist the coordinator's dispatch intent before `Client::send_control` delivery.
pub fn freeze_dispatch_intent(path: &Path, command: &SubtaskCommand) -> Result<()> {
    let bytes = serde_json::to_vec(command)?;
    if path.exists() {
        ensure!(
            fs::read(path)? == bytes,
            "conflicting subtask dispatch intent"
        );
        return Ok(());
    }
    create_immutable(path, &bytes)
}
/// Delivery acknowledgement is not completion; verify the durable receipt separately.
pub async fn dispatch_subtask(
    client: &Client,
    job_id: jain_smartcluster::JobId,
    intent: &Path,
    command: &SubtaskCommand,
) -> Result<jain_smartcluster::OperationAck> {
    let payload = serde_json::to_vec(command)?;
    ensure!(payload.len() <= 64 * 1024, "subtask command too large");
    freeze_dispatch_intent(intent, command)?;
    Ok(client
        .send_control(jain_smartcluster::SendControlRequest {
            job_id,
            control: ControlMessage::Application {
                name: CONTROL_NAME.to_owned(),
                payload,
            },
        })
        .await?)
}
pub(crate) fn write_immutable_record(path: &Path, value: &Value) -> Result<()> {
    create_immutable(path, &serde_json::to_vec(value)?)
}
fn create_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len() <= FRAME_LIMIT,
        "durable control record too large"
    );
    let parent = path.parent().context("record parent absent")?;
    fs::create_dir_all(parent)?;
    // Keep interrupted staging files: final names become visible only after a
    // complete fsynced payload, using an atomic no-replace rename publication.
    let temporary = tempfile::Builder::new()
        .prefix(".immutable-pending-")
        .tempfile_in(parent)?;
    let (mut file, temporary_path) = temporary.keep()?;
    file.write_all(bytes)?;
    file.sync_all()?;
    match crate::native_persistence::publish_staged(&temporary_path, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            ensure!(fs::read(path)? == bytes, "conflicting immutable record");
        }
        Err(error) => return Err(error.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    crate::native_persistence::discard_staging(&temporary_path)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

mod control;
mod runner;

pub(crate) use control::{
    record_control_stop, stop_control_with, ControlStopDeadline, ControlStopReason, DeadlineWatch,
    ScratchGuard,
};
pub(crate) use runner::execute;

pub fn serve(path: &Path, sha: &str) -> Result<()> {
    let policy = load_policy()?;
    let mut allocation = authenticate(policy, path, sha)?;
    let (_, input) = load_task(&allocation.policy, path, sha)?;
    let stop = Arc::new(AtomicBool::new(false));
    let mut reader = allocation.stream.try_clone()?;
    let observed = Arc::clone(&stop);
    let attempt = allocation.attempt;
    let control_state_root = allocation.policy.state_root.clone();
    let allowance = allocation.bootstrap.checkpoint_allowance_ms.unwrap_or(1000);
    let (send, receive) = mpsc::sync_channel::<SubtaskCommand>(1);
    std::thread::spawn(move || {
        let mut previous = 0;
        let reason = loop {
            let frame = match read_frame(&mut reader) {
                Ok(frame) => frame,
                Err(error) => {
                    break if error
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|e| e.kind() == std::io::ErrorKind::UnexpectedEof)
                    {
                        ControlStopReason::EndOfFile
                    } else {
                        ControlStopReason::FrameReadError
                    }
                }
            };
            if frame.attempt != attempt
                || frame.sequence <= previous
                || frame.version != ProtocolVersion::CURRENT
            {
                continue;
            }
            previous = frame.sequence;
            match frame.control {
                ControlMessage::Checkpoint { .. } => {
                    break ControlStopReason::Checkpoint;
                }
                ControlMessage::Application { name, payload } if name == CONTROL_NAME => {
                    match serde_json::from_slice(&payload) {
                        Ok(command) => match send.try_send(command) {
                            Ok(()) => {}
                            Err(mpsc::TrySendError::Full(_)) => break ControlStopReason::QueueFull,
                            Err(mpsc::TrySendError::Disconnected(_)) => {
                                break ControlStopReason::QueueDisconnected
                            }
                        },
                        Err(_) => {
                            break ControlStopReason::MalformedNamedControl;
                        }
                    }
                }
                _ => {
                    break ControlStopReason::UnknownControl;
                }
            }
        };
        if stop_control_with(
            &observed,
            Duration::from_millis(allowance),
            |deadline| {
                std::thread::spawn(move || {
                    std::thread::sleep(
                        deadline.saturating_duration_since(std::time::Instant::now()),
                    );
                    // SAFETY: _exit has no preconditions. Skipping destructors and flushes is
                    // intended: a worker still running past its checkpoint allowance is killed.
                    unsafe {
                        libc::_exit(76);
                    }
                });
            },
            |deadline| {
                record_control_stop(&control_state_root, attempt, previous, reason, deadline)
            },
        )
        .is_err()
        {
            eprintln!(
                "control stop diagnostic could not be persisted; numerical admission stopped"
            );
        }
    });
    let mut sequence = 0;
    report(
        &mut allocation.stream,
        attempt,
        &mut sequence,
        WorkerReport::Ready,
    )?;
    if !matches!(input.operation, Operation::Resident) {
        let result = execute(&allocation, path, sha, Arc::clone(&stop))?;
        return report(
            &mut allocation.stream,
            attempt,
            &mut sequence,
            result_progress(&result)?,
        );
    }
    while !stop.load(Ordering::Acquire) {
        let command = match receive.recv_timeout(Duration::from_millis(100)) {
            Ok(command) => command,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => break,
        };
        match execute(
            &allocation,
            &command.manifest,
            &command.manifest_sha256,
            Arc::clone(&stop),
        ) {
            Ok(result) => report(
                &mut allocation.stream,
                attempt,
                &mut sequence,
                result_progress(&result)?,
            )?,
            Err(error) => {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                report(
                    &mut allocation.stream,
                    attempt,
                    &mut sequence,
                    WorkerReport::Failure {
                        class: jain_smartcluster::FailureClass::Application,
                        message: format!("{error:#}").chars().take(4000).collect(),
                    },
                )?;
                return Err(error);
            }
        }
    }
    write_json(
        &allocation.policy.state_root.join(format!(
            "recovery-{}-{}.json",
            attempt.job_id, attempt.fence
        )),
        &json!({"schema":"jope.smartcluster_recovery.v1","attempt":attempt,"recovery_state_fsynced":true,"numerical_completion_requires_verified_task_receipt":true}),
    )?;
    Ok(())
}
fn result_progress(result: &Value) -> Result<WorkerReport> {
    Ok(WorkerReport::Progress(StructuredProgress {
        phase: "durable_subtask_result".into(),
        completed: 1,
        total: Some(1),
        message: None,
        fields: BTreeMap::from([("receipt".into(), serde_json::to_string(result)?)]),
    }))
}
