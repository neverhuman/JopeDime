use super::*;

pub enum WorkloadIo {
    Pipes {
        stdin: ChildStdin,
        stdout: Option<ChildStdout>,
        stderr: Option<ChildStderr>,
    },
    Pty {
        stream: tokio::fs::File,
        output: Option<tokio::fs::File>,
        resize: File,
    },
}

impl WorkloadIo {
    pub fn output(&mut self) -> WorkloadOutput<'_> {
        match self {
            Self::Pipes { stdout, stderr, .. } => WorkloadOutput::Pipes {
                stdout: stdout
                    .as_mut()
                    .map(|value| value as &mut (dyn AsyncRead + Unpin + Send)),
                stderr: stderr
                    .as_mut()
                    .map(|value| value as &mut (dyn AsyncRead + Unpin + Send)),
            },
            Self::Pty { output, .. } => WorkloadOutput::Pty(
                output
                    .as_mut()
                    .map(|value| value as &mut (dyn AsyncRead + Unpin + Send)),
            ),
        }
    }

    pub fn take_output(&mut self) -> WorkloadOutputOwned {
        match self {
            Self::Pipes { stdout, stderr, .. } => WorkloadOutputOwned::Pipes {
                stdout: stdout.take(),
                stderr: stderr.take(),
            },
            Self::Pty { output, .. } => WorkloadOutputOwned::Pty(output.take()),
        }
    }

    pub fn input(&mut self) -> WorkloadInput<'_> {
        match self {
            Self::Pipes { stdin, .. } => WorkloadInput::Pipe(stdin),
            Self::Pty { stream, .. } => WorkloadInput::Pty(stream),
        }
    }

    pub fn resize(&self, rows: u16, columns: u16) -> Result<(), ExecutorError> {
        if rows == 0 || columns == 0 {
            return Err(ExecutorError::InvalidPtySize { rows, columns });
        }
        let Self::Pty { resize, .. } = self else {
            return Err(ExecutorError::NotPty);
        };
        rustix::termios::tcsetwinsize(
            resize,
            rustix::termios::Winsize {
                ws_row: rows,
                ws_col: columns,
                ws_xpixel: 0,
                ws_ypixel: 0,
            },
        )?;
        Ok(())
    }
}

pub enum WorkloadOutput<'a> {
    Pipes {
        stdout: Option<&'a mut (dyn AsyncRead + Unpin + Send)>,
        stderr: Option<&'a mut (dyn AsyncRead + Unpin + Send)>,
    },
    Pty(Option<&'a mut (dyn AsyncRead + Unpin + Send)>),
}

pub enum WorkloadOutputOwned {
    Pipes {
        stdout: Option<ChildStdout>,
        stderr: Option<ChildStderr>,
    },
    Pty(Option<tokio::fs::File>),
}

pub enum WorkloadInput<'a> {
    Pipe(&'a mut (dyn AsyncWrite + Unpin + Send)),
    Pty(&'a mut (dyn AsyncWrite + Unpin + Send)),
}

pub struct RunningWorkload {
    pub attempt: FencedAttempt,
    pub pid: u32,
    pub cgroup: Option<JobCgroup>,
    pub child: Child,
    pub control: ControlChannel,
    pub io: WorkloadIo,
    pub(crate) cgroups: CgroupManager,
    pub(crate) execution: ExecutionConfig,
    pub(crate) control_sequence: u64,
    pub(crate) _single_allocation_permit: Option<OwnedSemaphorePermit>,
    pub(crate) launch_guard: LaunchGuard,
}

impl RunningWorkload {
    /// Releases the launch fence after the attempt and Running transition are durable.
    pub fn start(&mut self) -> Result<(), ExecutorError> {
        kill(Pid::from_raw(self.pid as i32), Signal::SIGCONT).map_err(ExecutorError::Continue)?;
        self.launch_guard.armed = false;
        Ok(())
    }

    pub fn leave_for_recovery(mut self) {
        self.launch_guard.armed = false;
    }

    pub async fn force_kill(mut self) {
        self.launch_guard.armed = false;
        terminate_failed_launch(&mut self.child, self.cgroup.as_ref(), &self.cgroups).await;
    }

    pub fn into_parts(mut self) -> RunningParts {
        self.launch_guard.armed = false;
        let output = self.io.take_output();
        RunningParts {
            attempt: self.attempt,
            pid: self.pid,
            cgroup: self.cgroup,
            control: self.control,
            io: self.io,
            output,
            process: ProcessWaiter {
                child: self.child,
                _single_allocation_permit: self._single_allocation_permit,
            },
        }
    }

    pub fn pause(&self) -> Result<(), ExecutorError> {
        let cgroup = self.cgroup.as_ref().ok_or(ExecutorError::NoFreezer)?;
        self.cgroups.freeze(cgroup)?;
        Ok(())
    }

    pub fn resume(&self) -> Result<(), ExecutorError> {
        let cgroup = self.cgroup.as_ref().ok_or(ExecutorError::NoFreezer)?;
        self.cgroups.resume(cgroup)?;
        Ok(())
    }

    pub async fn send_control(&mut self, control: ControlMessage) -> Result<(), ExecutorError> {
        self.control_sequence = self.control_sequence.saturating_add(1);
        self.control
            .send(&WorkerControlFrame {
                version: ProtocolVersion::CURRENT,
                attempt: self.attempt,
                sequence: self.control_sequence,
                control,
            })
            .await?;
        Ok(())
    }

    pub async fn cancel(
        &mut self,
        request_checkpoint: bool,
        grace: Option<Duration>,
    ) -> Result<CancelOutcome, ExecutorError> {
        let mut checkpoint = None;
        if request_checkpoint {
            let timeout = self.execution.checkpoint_timeout();
            self.control_sequence = self.control_sequence.saturating_add(1);
            let checkpoint_sequence = self.control_sequence;
            self.control
                .send(&WorkerControlFrame {
                    version: ProtocolVersion::CURRENT,
                    attempt: self.attempt,
                    sequence: checkpoint_sequence,
                    control: ControlMessage::Checkpoint {
                        deadline_ms: timeout.as_millis().try_into().unwrap_or(u64::MAX),
                    },
                })
                .await?;
            let deadline = tokio::time::Instant::now() + timeout;
            loop {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                let report = self
                    .control
                    .receive_until::<WorkerReportFrame>(remaining)
                    .await;
                match report {
                    Ok(frame)
                        if frame.version != ProtocolVersion::CURRENT
                            || frame.attempt != self.attempt
                            || frame.sequence != checkpoint_sequence =>
                    {
                        continue;
                    }
                    Ok(WorkerReportFrame {
                        report: WorkerReport::CheckpointCommitted(checkpoint_event),
                        ..
                    }) => {
                        checkpoint = Some(checkpoint_event.checkpoint_id);
                        break;
                    }
                    Ok(_) => continue,
                    Err(DaemonControlFrameError::Timeout) => break,
                    Err(error) => return Err(error.into()),
                }
            }
        }

        self.signal(Signal::SIGTERM)?;
        let graceful = grace.unwrap_or_else(|| self.execution.terminate_grace());
        match tokio::time::timeout(graceful, self.child.wait()).await {
            Ok(status) => Ok(CancelOutcome {
                checkpoint,
                forced: false,
                exit_status: Some(status?),
            }),
            Err(_) => {
                if let Some(cgroup) = &self.cgroup {
                    self.cgroups.kill(cgroup)?;
                } else {
                    signal_process_group(self.pid, Signal::SIGKILL)?;
                }
                let status = tokio::time::timeout(self.execution.kill_grace(), self.child.wait())
                    .await
                    .ok()
                    .transpose()?;
                Ok(CancelOutcome {
                    checkpoint,
                    forced: true,
                    exit_status: status,
                })
            }
        }
    }

    fn signal(&self, signal: Signal) -> Result<(), ExecutorError> {
        if let Some(cgroup) = &self.cgroup {
            self.cgroups.signal_all(cgroup, signal)?;
        } else {
            signal_process_group(self.pid, signal)?;
        }
        Ok(())
    }
}

pub(crate) struct LaunchGuard {
    pub(crate) cgroup: Option<JobCgroup>,
    pub(crate) pid: u32,
    pub(crate) cgroups: CgroupManager,
    pub(crate) armed: bool,
}

impl Drop for LaunchGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Some(cgroup) = &self.cgroup {
            let _ = self.cgroups.kill(cgroup);
        } else if let Ok(pid) = i32::try_from(self.pid) {
            let _ = kill(Pid::from_raw(-pid), Signal::SIGKILL);
        }
    }
}

fn signal_process_group(pid: u32, signal: Signal) -> Result<(), ExecutorError> {
    let pid = i32::try_from(pid).map_err(|_| ExecutorError::ProcessIdOutOfRange(pid))?;
    match kill(Pid::from_raw(-pid), signal) {
        Ok(()) | Err(nix::errno::Errno::ESRCH) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

pub struct RunningParts {
    pub attempt: FencedAttempt,
    pub pid: u32,
    pub cgroup: Option<JobCgroup>,
    pub control: ControlChannel,
    pub io: WorkloadIo,
    pub output: WorkloadOutputOwned,
    pub process: ProcessWaiter,
}

pub struct ProcessWaiter {
    child: Child,
    pub(crate) _single_allocation_permit: Option<OwnedSemaphorePermit>,
}

impl ProcessWaiter {
    pub async fn wait(mut self) -> io::Result<std::process::ExitStatus> {
        self.child.wait().await
    }
}

#[derive(Debug)]
pub struct CancelOutcome {
    pub checkpoint: Option<String>,
    pub forced: bool,
    pub exit_status: Option<std::process::ExitStatus>,
}
