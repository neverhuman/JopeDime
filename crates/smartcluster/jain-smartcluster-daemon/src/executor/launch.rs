use super::*;

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

pub(super) async fn terminate_failed_launch(
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
