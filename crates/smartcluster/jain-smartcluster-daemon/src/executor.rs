use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use jain_smartcluster_core::{
    ControlMessage, FencedAttempt, IoMode, JainWorkload, PriorityClass, ProtocolVersion,
    ResourceRequest, WorkerControlFrame, WorkerReport, WorkerReportFrame, Workload,
};
use nix::pty::{openpty, Winsize};
use nix::sys::signal::{kill, raise, Signal};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::{setsid, Pid};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::capacity::CpuSet;
use crate::cgroup::{
    AttemptKey, CgroupError, CgroupLimits, CgroupManager, IsolationMode, JobCgroup,
};
use crate::config::{ExecutionConfig, WorkerConfig};
use crate::control::{ControlChannel, ControlError, ControlPair, CONTROL_FD_ENV};

pub const EXEC_HELPER_MARKER: &str = "__scqd_exec_helper";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchPlan {
    pub program: PathBuf,
    pub arguments: Vec<String>,
    pub environment: BTreeMap<String, String>,
    pub working_directory: Option<PathBuf>,
}

pub struct ExecutionAllocation {
    pub resources: ResourceRequest,
    pub priority: PriorityClass,
    pub cpus: CpuSet,
    pub io_mode: IoMode,
    pub gpu_devices: Vec<String>,
    pub daemon_environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct WorkloadResolver {
    workers: WorkerConfig,
    jain_worker_identity: Arc<OnceLock<[u8; 32]>>,
    identity_started: Arc<AtomicBool>,
}

impl WorkloadResolver {
    pub fn new(workers: WorkerConfig) -> Self {
        Self {
            workers,
            jain_worker_identity: Arc::new(OnceLock::new()),
            identity_started: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A non-secret, versioned identity used only for conservative GPU learning.
    /// Environment values are deliberately excluded; arguments are length-delimited
    /// into the digest and are never persisted in plaintext.
    pub fn fingerprint(&self, workload: &Workload) -> Option<String> {
        let Workload::Jain {
            template,
            arguments,
            ..
        } = workload
        else {
            return None;
        };
        let Some(binary) = self.jain_worker_identity.get().copied() else {
            if !self.identity_started.swap(true, Ordering::AcqRel) {
                let path = self.workers.jain_worker.clone();
                let identity = Arc::clone(&self.jain_worker_identity);
                if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                    runtime.spawn_blocking(move || {
                        if let Ok(binary) = std::fs::read(path) {
                            let _ = identity.set(Sha256::digest(binary).into());
                        }
                    });
                } else {
                    self.identity_started.store(false, Ordering::Release);
                }
            }
            return None;
        };
        let mut digest = Sha256::new();
        digest.update(b"jain-worker-fingerprint-v2\0");
        digest.update(binary);
        digest.update(jain_subcommand(*template).as_bytes());
        for argument in arguments {
            digest.update(
                u64::try_from(argument.len())
                    .unwrap_or(u64::MAX)
                    .to_le_bytes(),
            );
            digest.update(argument.as_bytes());
        }
        Some(format!("jain-worker-v2:{}", hex::encode(digest.finalize())))
    }

    pub fn resolve(&self, workload: &Workload) -> Result<LaunchPlan, ExecutorError> {
        let plan = match workload {
            Workload::Jain {
                template,
                arguments,
                environment,
            } => {
                let mut argv = vec![jain_subcommand(*template).to_owned()];
                argv.extend(arguments.iter().cloned());
                LaunchPlan {
                    program: self.workers.jain_worker.clone(),
                    arguments: argv,
                    environment: environment.0.clone(),
                    working_directory: None,
                }
            }
            Workload::TrustedService {
                service,
                arguments,
                environment,
            } => LaunchPlan {
                program: self
                    .workers
                    .trusted_services
                    .get(service.as_str())
                    .cloned()
                    .ok_or_else(|| ExecutorError::UnknownService(service.to_string()))?,
                arguments: arguments.clone(),
                environment: environment.0.clone(),
                working_directory: None,
            },
            Workload::AdministratorArgv {
                program,
                arguments,
                environment,
                working_directory,
            } => LaunchPlan {
                program: PathBuf::from(program),
                arguments: arguments.clone(),
                environment: environment.0.clone(),
                working_directory: working_directory.as_ref().map(PathBuf::from),
            },
        };
        plan.validate()?;
        Ok(plan)
    }
}

fn jain_subcommand(template: JainWorkload) -> &'static str {
    match template {
        JainWorkload::DatasetProfile => "dataset-profile",
        JainWorkload::FeatureExpansion => "feature-expansion",
        JainWorkload::Training => "training",
        JainWorkload::ReportGeneration => "report-generation",
        JainWorkload::Prediction => "prediction",
        JainWorkload::Export => "export",
        JainWorkload::Chat => "chat",
        JainWorkload::Research => "research",
        JainWorkload::ManagedBrowser => "managed-browser",
        JainWorkload::JopeEncode => "jope-encode",
        JainWorkload::JopePropose => "jope-propose",
        JainWorkload::LimeRun => "lime-run",
        JainWorkload::LimeBatch => "lime-batch",
    }
}

impl LaunchPlan {
    fn validate(&self) -> Result<(), ExecutorError> {
        if !self.program.is_absolute() {
            return Err(ExecutorError::ProgramNotAbsolute(self.program.clone()));
        }
        if let Some(directory) = &self.working_directory {
            if !directory.is_absolute() {
                return Err(ExecutorError::WorkingDirectoryNotAbsolute(
                    directory.clone(),
                ));
            }
        }
        for name in self.environment.keys() {
            if reserved_environment_name(name) {
                return Err(ExecutorError::ReservedEnvironment(name.clone()));
            }
        }
        Ok(())
    }
}

fn reserved_environment_name(name: &str) -> bool {
    name == "PATH"
        || name.starts_with("LD_")
        || name.starts_with("DYLD_")
        || name.starts_with("SCQ_")
        || matches!(name, "PYTHONPATH" | "PYTHONHOME" | "PYTHONSTARTUP")
        || name == "CUDA_VISIBLE_DEVICES"
        || name.starts_with("CUDA_MPS_")
}

fn validate_daemon_environment(
    environment: &BTreeMap<String, String>,
) -> Result<(), ExecutorError> {
    const DAEMON_KEYS: [&str; 5] = [
        "SCQ_SCRATCH_ROOT",
        "SCQ_SCRATCH_RESERVATION_BYTES",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
        "CUDA_MPS_PIPE_DIRECTORY",
    ];
    if environment
        .keys()
        .all(|name| DAEMON_KEYS.contains(&name.as_str()))
    {
        Ok(())
    } else {
        Err(ExecutorError::InvalidDaemonEnvironment)
    }
}

#[derive(Clone)]
pub struct Executor {
    cgroups: CgroupManager,
    helper_executable: PathBuf,
    execution: ExecutionConfig,
    single_allocation: Arc<Semaphore>,
}

impl Executor {
    pub fn new(
        cgroups: CgroupManager,
        helper_executable: impl Into<PathBuf>,
        execution: ExecutionConfig,
    ) -> Result<Self, ExecutorError> {
        let helper_executable = helper_executable.into();
        if !helper_executable.is_absolute() {
            return Err(ExecutorError::HelperNotAbsolute(helper_executable));
        }
        Ok(Self {
            cgroups,
            helper_executable,
            execution,
            single_allocation: Arc::new(Semaphore::new(1)),
        })
    }

    pub async fn launch(
        &self,
        attempt: FencedAttempt,
        plan: LaunchPlan,
        allocation: ExecutionAllocation,
    ) -> Result<RunningWorkload, ExecutorError> {
        let ExecutionAllocation {
            resources,
            priority,
            cpus,
            io_mode,
            gpu_devices,
            daemon_environment,
        } = allocation;
        plan.validate()?;
        validate_daemon_environment(&daemon_environment)?;
        let permit = if self.cgroups.mode() == IsolationMode::SingleAllocation {
            Some(
                Arc::clone(&self.single_allocation)
                    .acquire_owned()
                    .await
                    .map_err(|_| ExecutorError::Stopped)?,
            )
        } else {
            None
        };
        let attempt_key = AttemptKey {
            job_id: attempt.job_id.to_string(),
            attempt: attempt.number.get(),
            fence: attempt.fence.to_string(),
        };
        let cgroup = if self.cgroups.mode() == IsolationMode::Delegated {
            Some(
                self.cgroups
                    .prepare(&attempt_key, &limits(resources, priority, cpus))?,
            )
        } else {
            None
        };

        let pair = ControlPair::create()?;
        let worker_fd = pair.worker_fd();
        let mut command = Command::new(&self.helper_executable);
        command
            .arg(EXEC_HELPER_MARKER)
            .arg(std::process::id().to_string())
            .arg(
                self.execution
                    .worker_uid
                    .map_or_else(|| "-".to_owned(), |uid| uid.to_string()),
            )
            .arg(
                self.execution
                    .worker_gid
                    .map_or_else(|| "-".to_owned(), |gid| gid.to_string()),
            )
            .arg(&plan.program)
            .args(&plan.arguments)
            .env_clear()
            .kill_on_drop(false);
        for (name, value) in &plan.environment {
            command.env(name, value);
        }
        for (name, value) in &daemon_environment {
            command.env(name, value);
        }
        command
            .env(
                "PATH",
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            )
            .env(CONTROL_FD_ENV, worker_fd.to_string())
            .env("SCQ_JOB_ID", attempt.job_id.to_string())
            .env("SCQ_ATTEMPT_NUMBER", attempt.number.to_string())
            .env("SCQ_ATTEMPT_FENCE", attempt.fence.to_string())
            .env("CUDA_VISIBLE_DEVICES", cuda_visible_devices(&gpu_devices));
        if let Some(directory) = &plan.working_directory {
            command.current_dir(directory);
        }
        let prepared_io = prepare_io(&mut command, io_mode)?;
        let mut child = command.spawn().map_err(|source| ExecutorError::Spawn {
            program: self.helper_executable.clone(),
            source,
        })?;
        let pid = child.id().ok_or(ExecutorError::NoPid)?;

        let stopped = tokio::time::timeout(
            self.execution.launch_stop_timeout(),
            wait_for_helper_stop(pid),
        )
        .await;
        match stopped {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                terminate_failed_launch(&mut child, cgroup.as_ref(), &self.cgroups).await;
                return Err(error);
            }
            Err(_) => {
                terminate_failed_launch(&mut child, cgroup.as_ref(), &self.cgroups).await;
                return Err(ExecutorError::HelperStopTimeout(pid));
            }
        }
        if let Some(cgroup) = &cgroup {
            if let Err(error) = self.cgroups.attach_pid(cgroup, pid) {
                terminate_failed_launch(&mut child, Some(cgroup), &self.cgroups).await;
                return Err(error.into());
            }
        }
        let control =
            match pair.close_worker_and_open_daemon(self.execution.max_control_frame_bytes) {
                Ok(control) => control,
                Err(error) => {
                    terminate_failed_launch(&mut child, cgroup.as_ref(), &self.cgroups).await;
                    return Err(error.into());
                }
            };
        let io = match prepared_io.finish(&mut child) {
            Ok(io) => io,
            Err(error) => {
                terminate_failed_launch(&mut child, cgroup.as_ref(), &self.cgroups).await;
                return Err(error);
            }
        };

        let launch_guard = LaunchGuard {
            cgroup: cgroup.clone(),
            pid,
            cgroups: self.cgroups.clone(),
            armed: true,
        };
        Ok(RunningWorkload {
            attempt,
            pid,
            cgroup,
            child,
            control,
            io,
            cgroups: self.cgroups.clone(),
            execution: self.execution.clone(),
            control_sequence: 0,
            _single_allocation_permit: permit,
            launch_guard,
        })
    }
}

fn limits(resources: ResourceRequest, priority: PriorityClass, cpus: CpuSet) -> CgroupLimits {
    let cpu_weight = match priority {
        PriorityClass::System => 10_000,
        PriorityClass::Interactive => 7_500,
        PriorityClass::Batch => 2_500,
        PriorityClass::Opportunistic => 100,
    };
    CgroupLimits {
        cpus,
        cpu_weight,
        cpu_max_millicores: u64::from(resources.cpu.maximum_millicores),
        memory_low_bytes: resources.ram.soft_bytes,
        memory_high_bytes: resources.ram.soft_bytes,
        memory_max_bytes: resources.ram.hard_bytes,
    }
}

fn cuda_visible_devices(devices: &[String]) -> String {
    if devices.is_empty() {
        "-1".to_owned()
    } else {
        devices.join(",")
    }
}

enum PreparedIo {
    Pipes,
    Pty(File),
}

impl PreparedIo {
    fn finish(self, child: &mut Child) -> Result<WorkloadIo, ExecutorError> {
        match self {
            Self::Pipes => Ok(WorkloadIo::Pipes {
                stdin: child
                    .stdin
                    .take()
                    .ok_or(ExecutorError::MissingPipe("stdin"))?,
                stdout: Some(
                    child
                        .stdout
                        .take()
                        .ok_or(ExecutorError::MissingPipe("stdout"))?,
                ),
                stderr: Some(
                    child
                        .stderr
                        .take()
                        .ok_or(ExecutorError::MissingPipe("stderr"))?,
                ),
            }),
            Self::Pty(master) => {
                let output = master.try_clone()?;
                let resize = master.try_clone()?;
                Ok(WorkloadIo::Pty {
                    stream: tokio::fs::File::from_std(master),
                    output: Some(tokio::fs::File::from_std(output)),
                    resize,
                })
            }
        }
    }
}

fn prepare_io(command: &mut Command, io_mode: IoMode) -> Result<PreparedIo, ExecutorError> {
    match io_mode {
        IoMode::Pipes => {
            command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            Ok(PreparedIo::Pipes)
        }
        IoMode::Pty { rows, columns } => {
            let pty = openpty(
                Some(&Winsize {
                    ws_row: rows,
                    ws_col: columns,
                    ws_xpixel: 0,
                    ws_ypixel: 0,
                }),
                None,
            )?;
            let master = File::from(pty.master);
            let slave = File::from(pty.slave);
            command
                .stdin(Stdio::from(slave.try_clone()?))
                .stdout(Stdio::from(slave.try_clone()?))
                .stderr(Stdio::from(slave));
            Ok(PreparedIo::Pty(master))
        }
    }
}

async fn wait_for_helper_stop(pid: u32) -> Result<(), ExecutorError> {
    tokio::task::spawn_blocking(move || {
        match waitpid(Pid::from_raw(pid as i32), Some(WaitPidFlag::WUNTRACED)) {
            Ok(WaitStatus::Stopped(_, Signal::SIGSTOP)) => Ok(()),
            Ok(status) => Err(ExecutorError::UnexpectedHelperStatus(status)),
            Err(source) => Err(ExecutorError::WaitPid(source)),
        }
    })
    .await
    .map_err(ExecutorError::Join)?
}

async fn terminate_failed_launch(
    child: &mut Child,
    cgroup: Option<&JobCgroup>,
    cgroups: &CgroupManager,
) {
    if let Some(cgroup) = cgroup {
        let _ = cgroups.kill(cgroup);
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

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
    cgroups: CgroupManager,
    execution: ExecutionConfig,
    control_sequence: u64,
    _single_allocation_permit: Option<OwnedSemaphorePermit>,
    launch_guard: LaunchGuard,
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
                    Err(ControlError::Timeout) => break,
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

struct LaunchGuard {
    cgroup: Option<JobCgroup>,
    pid: u32,
    cgroups: CgroupManager,
    armed: bool,
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
    _single_allocation_permit: Option<OwnedSemaphorePermit>,
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

pub fn run_exec_helper(arguments: &[String]) -> Result<(), ExecutorError> {
    let HelperInvocation {
        expected_parent,
        identity,
        program,
        arguments,
    } = parse_helper_arguments(arguments)?;
    nix::sys::prctl::set_pdeathsig(Signal::SIGKILL).map_err(ExecutorError::ArmParentDeathSignal)?;
    verify_helper_parent(expected_parent)?;
    setsid().map_err(ExecutorError::SetSid)?;
    // If stdin is the PTY slave, establish it as this fresh session's controlling
    // terminal. Pipes return ENOTTY and intentionally continue without one.
    let _ = rustix::process::ioctl_tiocsctty(std::io::stdin());
    raise(Signal::SIGSTOP).map_err(ExecutorError::StopSelf)?;
    nix::sys::prctl::set_pdeathsig(None::<Signal>)
        .map_err(ExecutorError::ClearParentDeathSignal)?;
    verify_helper_parent(expected_parent)?;
    if let Some((uid, gid)) = identity {
        nix::unistd::setgroups(&[]).map_err(ExecutorError::ClearSupplementaryGroups)?;
        nix::unistd::setgid(nix::unistd::Gid::from_raw(gid)).map_err(ExecutorError::SetGid)?;
        nix::unistd::setuid(nix::unistd::Uid::from_raw(uid)).map_err(ExecutorError::SetUid)?;
    }
    nix::sys::prctl::set_no_new_privs().map_err(ExecutorError::NoNewPrivileges)?;
    let error = std::process::Command::new(program).args(arguments).exec();
    Err(ExecutorError::Exec {
        program: PathBuf::from(program),
        source: error,
    })
}

struct HelperInvocation<'a> {
    expected_parent: Pid,
    identity: Option<(u32, u32)>,
    program: &'a str,
    arguments: &'a [String],
}

fn parse_helper_arguments(arguments: &[String]) -> Result<HelperInvocation<'_>, ExecutorError> {
    let [expected_parent, uid, gid, remainder @ ..] = arguments else {
        return Err(ExecutorError::HelperArguments);
    };
    let expected_parent = expected_parent
        .parse::<i32>()
        .ok()
        .filter(|pid| *pid > 0)
        .map(Pid::from_raw)
        .ok_or(ExecutorError::HelperParent)?;
    let identity = match (uid.as_str(), gid.as_str()) {
        ("-", "-") => None,
        ("-", _) | (_, "-") => return Err(ExecutorError::HelperIdentity),
        (uid, gid) => Some((
            uid.parse().map_err(|_| ExecutorError::HelperIdentity)?,
            gid.parse().map_err(|_| ExecutorError::HelperIdentity)?,
        )),
    };
    let (program, program_arguments) = remainder
        .split_first()
        .ok_or(ExecutorError::HelperArguments)?;
    Ok(HelperInvocation {
        expected_parent,
        identity,
        program,
        arguments: program_arguments,
    })
}

fn verify_helper_parent(expected: Pid) -> Result<(), ExecutorError> {
    let actual = nix::unistd::getppid();
    if actual == expected {
        Ok(())
    } else {
        Err(ExecutorError::HelperParentChanged { expected, actual })
    }
}

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("workload program must be absolute: {0}")]
    ProgramNotAbsolute(PathBuf),
    #[error("workload working directory must be absolute: {0}")]
    WorkingDirectoryNotAbsolute(PathBuf),
    #[error("executor helper must be absolute: {0}")]
    HelperNotAbsolute(PathBuf),
    #[error("trusted service is not configured: {0}")]
    UnknownService(String),
    #[error("workload environment may not override reserved variable {0}")]
    ReservedEnvironment(String),
    #[error("executor received a daemon environment variable outside the MPS allowlist")]
    InvalidDaemonEnvironment,
    #[error("single-allocation executor has stopped")]
    Stopped,
    #[error("cannot spawn executor helper {program}: {source}")]
    Spawn { program: PathBuf, source: io::Error },
    #[error("spawned helper has no pid")]
    NoPid,
    #[error("workload process id is outside the supported range: {0}")]
    ProcessIdOutOfRange(u32),
    #[error("executor helper {0} did not stop before migration")]
    HelperStopTimeout(u32),
    #[error("executor helper reported unexpected wait status: {0:?}")]
    UnexpectedHelperStatus(WaitStatus),
    #[error("waitpid failed: {0}")]
    WaitPid(nix::Error),
    #[error("stopped-helper task failed: {0}")]
    Join(tokio::task::JoinError),
    #[error("cannot continue migrated helper: {0}")]
    Continue(nix::Error),
    #[error("missing child {0} pipe")]
    MissingPipe(&'static str),
    #[error("freezer is unavailable in safe single-allocation mode")]
    NoFreezer,
    #[error("adopted workload has no recoverable control channel")]
    NoControlChannel,
    #[error("attach resize is only valid for PTY workloads")]
    NotPty,
    #[error("invalid PTY dimensions: {rows}x{columns}")]
    InvalidPtySize { rows: u16, columns: u16 },
    #[error("executor helper requires parent, identity, and program arguments")]
    HelperArguments,
    #[error("executor helper received an invalid expected parent process id")]
    HelperParent,
    #[error("executor helper parent changed from {expected} to {actual} during launch fencing")]
    HelperParentChanged { expected: Pid, actual: Pid },
    #[error("executor helper received an invalid worker uid/gid pair")]
    HelperIdentity,
    #[error("setsid failed in executor helper: {0}")]
    SetSid(nix::Error),
    #[error("executor helper could not enable no-new-privileges: {0}")]
    NoNewPrivileges(nix::Error),
    #[error("executor helper could not clear supplementary groups: {0}")]
    ClearSupplementaryGroups(nix::Error),
    #[error("executor helper could not set its group: {0}")]
    SetGid(nix::Error),
    #[error("executor helper could not set its user: {0}")]
    SetUid(nix::Error),
    #[error("executor helper could not stop itself: {0}")]
    StopSelf(nix::Error),
    #[error("executor helper could not arm its parent-death launch fence: {0}")]
    ArmParentDeathSignal(nix::Error),
    #[error("executor helper could not clear its parent-death launch fence: {0}")]
    ClearParentDeathSignal(nix::Error),
    #[error("direct exec of {program} failed: {source}")]
    Exec { program: PathBuf, source: io::Error },
    #[error(transparent)]
    Cgroup(#[from] CgroupError),
    #[error(transparent)]
    Control(#[from] ControlError),
    #[error(transparent)]
    Nix(#[from] nix::Error),
    #[error("terminal operation failed: {0}")]
    Rustix(#[from] rustix::io::Errno),
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[cfg(test)]
mod tests;
