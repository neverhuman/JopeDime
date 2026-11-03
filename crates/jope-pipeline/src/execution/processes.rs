use super::*;

pub(super) struct Running {
    pub(super) started: Started,
    pub(super) cpu: u32,
    watcher: Option<thread::JoinHandle<Result<WorkerOutcome>>>,
    started_path: PathBuf,
    pub(super) stop: Arc<AtomicBool>,
}
impl Running {
    pub(super) fn start(
        started: Started,
        cpu: u32,
        memory_bytes: u64,
        per_fit_seconds: u64,
        started_path: PathBuf,
        logs: PathBuf,
        cancellation: Arc<AtomicBool>,
    ) -> Result<Self> {
        fs::create_dir_all(logs.join(&started.job_sha256))?;
        let process_started = started.clone();
        let process_journal = started_path.clone();
        Self::spawn(started, cpu, started_path, move |thread_stop| {
            #[cfg(test)]
            tests::host_setup_probe(tests::HostSetupStage::BeforeSetup, &process_started)?;
            if cancellation.load(Ordering::Acquire) || thread_stop.load(Ordering::Acquire) {
                return pre_gate_cancellation(&process_started, now_ms()?);
            }
            // The bounded Running slot reserves a child CPU assignment. Setup
            // hashing stays in the controller group under the aggregate limits.
            // No worker gate opens on the supervisor's serial scheduling thread.
            let process = ProcessFit::start(process_started, cpu, memory_bytes)?;
            process.watch(
                &process_journal,
                per_fit_seconds,
                &logs,
                &cancellation,
                &thread_stop,
            )
        })
    }
    pub(super) fn spawn(
        started: Started,
        cpu: u32,
        started_path: PathBuf,
        watch: impl FnOnce(Arc<AtomicBool>) -> Result<WorkerOutcome> + Send + 'static,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let watcher = thread::Builder::new()
            .name(format!("fit-{}", &started.job_sha256[..12]))
            .spawn(move || watch(thread_stop))?;
        Ok(Self {
            started,
            cpu,
            watcher: Some(watcher),
            started_path,
            stop,
        })
    }
    pub(super) fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
    pub(super) fn is_done(&self) -> bool {
        self.watcher.as_ref().is_some_and(|h| h.is_finished())
    }
    pub(super) fn finish(mut self) -> Result<WorkerOutcome> {
        let result = self.watcher.take().context("fit watcher missing")?.join();
        let reason = match result {
            Ok(Ok(outcome)) => return Ok(outcome),
            Ok(Err(error)) => format!("fit supervision/output failure: {error:#}"),
            Err(_) => "fit watcher panicked; original outcome unknown".into(),
        };
        if !self.started_path.exists() {
            return pre_gate_failure(&self.started, now_ms()?, reason);
        }
        let mut outcome =
            WorkerOutcome::unstarted(&self.started.job, EvaluationStatus::Failed, reason)?;
        outcome.consumption = ObservedConsumption {
            accounting_reason: "supervision failed; final counters and exit time unknown".into(),
            ..Default::default()
        };
        Ok(outcome)
    }
}
impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(watcher) = self.watcher.take() {
            let _ = watcher.join();
        }
    }
}

pub(super) struct ProcessFit {
    started: Started,
    child: Option<Child>,
    status: Option<std::process::ExitStatus>,
    stdout: Option<thread::JoinHandle<std::io::Result<Vec<u8>>>>,
    stderr: Option<thread::JoinHandle<std::io::Result<Vec<u8>>>>,
    began: Instant,
    monotonic_deadline: Instant,
    finished_at: Option<Instant>,
    gate_released_unix_ms: Option<u64>,
    observed_exit_unix_ms: Option<u64>,
    stop_status: Option<EvaluationStatus>,
    sampled_peak_rss_bytes: Option<u64>,
    drained: bool,
}
impl ProcessFit {
    fn start(started: Started, cpu: u32, memory_bytes: u64) -> Result<Self> {
        if started.cgroup.exists() {
            ensure!(
                named_counter(&started.cgroup, "cgroup.events", "populated")? == 0,
                "pre-existing fit cgroup remains populated"
            );
            // Empty kernel setup from a launch that never released its gate.
            fs::remove_dir(&started.cgroup)?;
        }
        ensure!(
            !started.job.directory.starts_with("/tmp"),
            "fit directory under /tmp"
        );
        fs::create_dir_all(&started.job.directory)?;
        ensure!(
            fs::canonicalize(&started.job.directory)? == started.job.directory,
            "physical fit directory required"
        );
        let allowed: BTreeSet<_> = started
            .job
            .command
            .inputs
            .iter()
            .map(|a| a.path.as_path())
            .collect();
        let mut pending = vec![started.job.directory.clone()];
        while let Some(directory) = pending.pop() {
            for entry in fs::read_dir(directory)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    pending.push(entry.path());
                } else {
                    ensure!(
                        entry.file_type()?.is_file() && allowed.contains(entry.path().as_path()),
                        "undeclared file in private fit directory"
                    );
                }
            }
        }
        for output in &started.job.outputs {
            ensure!(
                !started.job.directory.join(output).try_exists()?,
                "stale declared output exists before fit"
            );
        }
        let remaining = started
            .fit_deadline_unix_ms
            .checked_sub(now_ms()?)
            .context("fit expired before launch")?;
        fs::create_dir(&started.cgroup)?;
        let began = Instant::now();
        let mut running = Self {
            started,
            child: None,
            status: None,
            stdout: None,
            stderr: None,
            began,
            monotonic_deadline: began + Duration::from_millis(remaining),
            finished_at: None,
            gate_released_unix_ms: None,
            observed_exit_unix_ms: None,
            stop_status: None,
            sampled_peak_rss_bytes: None,
            drained: false,
        };
        let group = &running.started.cgroup;
        let parent = group.parent().context("fit cgroup parent absent")?;
        control_write(
            group,
            "cpuset.mems",
            &control_text(parent, "cpuset.mems.effective")?,
        )?;
        control_write(group, "cpuset.cpus", &cpu.to_string())?;
        control_write(group, "cpu.max", "100000 100000")?;
        control_write(group, "memory.max", &memory_bytes.to_string())?;
        control_write(group, "memory.swap.max", "0")?;
        control_write(group, "memory.oom.group", "1")?;
        OpenOptions::new()
            .write(true)
            .open(group.join("cgroup.kill"))?;
        verify_cpu(group, 1)?;
        verify_limit(group, "memory.max", memory_bytes)?;
        ensure!(
            control_text(group, "memory.swap.max")? == "0"
                && parse_cpus(&control_text(group, "cpuset.cpus.effective")?)? == [cpu],
            "fit resource readback differs"
        );
        let job = &running.started.job;
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "read -r jope_gate || exit 125; test \"$jope_gate\" = go || exit 125; exec \"$@\"",
                "jope-fit-gate",
            ])
            .arg(&job.command.program.path)
            .args(&job.command.args)
            .current_dir(&job.directory)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LANG", "C")
            .env("LC_ALL", "C")
            .env("CUDA_VISIBLE_DEVICES", "-1")
            .env("TMPDIR", &job.directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        for (name, value) in &job.command.environment {
            command.env(name, value);
        }
        for name in THREAD_VARS {
            command.env(name, "1");
        }
        running.child = Some(command.spawn().context("spawn gated fit")?);
        let child = running.child.as_mut().context("gated child absent")?;
        let pid = child.id();
        // Prefer killing a fit over its controller under the aggregate memory cap.
        fs::write(format!("/proc/{pid}/oom_score_adj"), b"1000")?;
        running.stdout = Some(capture(child.stdout.take().context("stdout missing")?));
        running.stderr = Some(capture(child.stderr.take().context("stderr missing")?));
        control_write(group, "cgroup.procs", &pid.to_string())?;
        let expected = group.strip_prefix("/sys/fs/cgroup")?.to_string_lossy();
        ensure!(
            fs::read_to_string(format!("/proc/{pid}/cgroup"))?
                .lines()
                .any(|line| line == format!("0::/{expected}")),
            "fit attachment readback differs"
        );
        #[cfg(test)]
        tests::host_setup_probe(tests::HostSetupStage::Attached, &running.started)?;
        // Always rehash actual bytes here; preliminary deduplication never
        // admits a command and must not become metadata-only gate verification.
        job.verify_inputs()?;
        ensure!(
            now_ms()? < running.started.fit_deadline_unix_ms,
            "deadline exhausted before gate release"
        );
        // The watcher owns Child before the gate is released. No fit runs on
        // the controller's artifact-hashing/scheduling thread.
        Ok(running)
    }
    fn watch(
        mut self,
        started_path: &Path,
        per_fit_seconds: u64,
        logs: &Path,
        cancellation: &AtomicBool,
        local_stop: &AtomicBool,
    ) -> Result<WorkerOutcome> {
        let gate_ms = now_ms()?;
        self.started.started_unix_ms = gate_ms;
        self.started.fit_deadline_unix_ms = self.started.fit_deadline_unix_ms.min(
            gate_ms
                .checked_add(per_fit_seconds * 1000)
                .context("fit deadline overflow")?,
        );
        ensure!(
            gate_ms < self.started.fit_deadline_unix_ms,
            "deadline exhausted before gate release"
        );
        if cancellation.load(Ordering::Acquire) || local_stop.load(Ordering::Acquire) {
            // No executable was released. Drop kills the gated wrapper and the
            // supervisor retains/reconciles cleanup ownership before slot reuse.
            return pre_gate_cancellation(&self.started, gate_ms);
        }
        immutable(started_path, &self.started)?;
        self.began = Instant::now();
        self.gate_released_unix_ms = Some(now_ms()?);
        self.monotonic_deadline = self.began
            + Duration::from_millis(self.started.fit_deadline_unix_ms.saturating_sub(now_ms()?));
        self.child
            .as_mut()
            .context("gated child absent")?
            .stdin
            .take()
            .context("gate stdin absent")?
            .write_all(b"go\n")?;
        while !self
            .is_done(cancellation.load(Ordering::Acquire) || local_stop.load(Ordering::Acquire))?
        {
            thread::sleep(Duration::from_millis(POLL_MS));
        }
        self.finish(logs)
    }
    fn is_done(&mut self, cancellation: bool) -> Result<bool> {
        // Poll and timestamp the exit independently of all controller hashing.
        // Already-observed exits retain their result if cancellation arrives later.
        if self.status.is_none() {
            self.status = self
                .child
                .as_mut()
                .context("fit child absent")?
                .try_wait()?;
        }
        if self.status.is_some() {
            self.finished_at.get_or_insert_with(Instant::now);
            self.observed_exit_unix_ms.get_or_insert(now_ms()?);
            return Ok(true);
        }
        // The original cutoff wins if a simultaneous supervisor/user signal
        // arrives at that cutoff. Earlier explicit cancellation remains Cancelled.
        if Instant::now() >= self.monotonic_deadline
            || now_ms()? >= self.started.fit_deadline_unix_ms
        {
            self.stop_status = Some(EvaluationStatus::TimedOut);
        } else if cancellation {
            self.stop_status = Some(EvaluationStatus::Cancelled);
        }
        if self.stop_status.is_some() {
            return Ok(true);
        }
        if let Ok(Some(rss)) = sampled_rss(&self.started.cgroup) {
            self.sampled_peak_rss_bytes = Some(self.sampled_peak_rss_bytes.unwrap_or(0).max(rss));
        }
        Ok(false)
    }
    fn finish(mut self, logs: &Path) -> Result<WorkerOutcome> {
        let mut outcome = WorkerOutcome::unstarted(
            &self.started.job,
            EvaluationStatus::Failed,
            "fit did not complete".into(),
        )?;
        if let Some(status) = self.stop_status {
            outcome.command_status = status;
            outcome.reason = Some(
                if status == EvaluationStatus::Cancelled {
                    "cancelled during fit"
                } else {
                    "fit, arm, or original execution cutoff reached"
                }
                .into(),
            );
        } else if self.status.is_some_and(|s| s.success()) {
            outcome.command_status = EvaluationStatus::Completed;
            outcome.reason = None;
        } else if self.status.and_then(|s| s.code()) == Some(69) {
            outcome.command_status = EvaluationStatus::Unavailable;
            outcome.reason = Some("worker reported unavailable dependency (exit69)".into());
        } else {
            outcome.reason = Some(format!("worker exited {:?}", self.status));
        }
        // Even a successful leader may leave grandchildren. Kill and observe an
        // empty cgroup before releasing this CPU slot or reading final outputs.
        if self.status.is_none() {
            if let Some(child) = &self.child {
                let _ = killpg(Pid::from_raw(child.id().try_into()?), Signal::SIGKILL);
            }
        }
        drain_cgroup(&self.started.cgroup)?;
        self.status = Some(self.child.as_mut().context("fit child absent")?.wait()?);
        self.finished_at.get_or_insert_with(Instant::now);
        self.observed_exit_unix_ms.get_or_insert(now_ms()?);
        outcome.exit_code = self.status.and_then(|s| s.code());
        outcome.signal = self.status.and_then(|s| s.signal());
        if let Ok(oom_kills) = named_counter(&self.started.cgroup, "memory.events", "oom_kill") {
            if oom_kills > 0 {
                outcome.command_status = EvaluationStatus::Failed;
                outcome.reason = Some(format!(
                    "memory_limit: cgroup recorded {oom_kills} OOM kills"
                ));
            }
        }
        let cpu_seconds = named_counter(&self.started.cgroup, "cpu.stat", "usage_usec")
            .ok()
            .map(|v| v as f64 / 1_000_000.);
        let memory = control_text(&self.started.cgroup, "memory.peak")
            .ok()
            .and_then(|v| v.parse().ok());
        let fit_memory_limit_bytes = control_text(&self.started.cgroup, "memory.max")
            .ok()
            .and_then(|v| v.parse().ok());
        outcome.consumption = ObservedConsumption {
            fit_memory_limit_bytes,
            gate_released_unix_ms: self.gate_released_unix_ms,
            observed_exit_unix_ms: self.observed_exit_unix_ms,
            wall_seconds: Some(
                self.finished_at
                    .unwrap_or_else(Instant::now)
                    .duration_since(self.began)
                    .as_secs_f64(),
            ),
            cpu_seconds,
            peak_rss_bytes: None,
            cgroup_peak_memory_bytes: memory,
            sampled_peak_rss_bytes: self.sampled_peak_rss_bytes,
            accounting_reason: if cpu_seconds.is_some() {
                "final cgroup CPU; cgroup memory includes cache/kernel; RSS is sampled only".into()
            } else {
                "final CPU counter unavailable; wall measured; memory/RSS availability explicit"
                    .into()
            },
        };
        for (thread, filename, destination) in [
            (&mut self.stdout, "stdout.log", &mut outcome.stdout),
            (&mut self.stderr, "stderr.log", &mut outcome.stderr),
        ] {
            let bytes = thread
                .take()
                .context("capture thread absent")?
                .join()
                .map_err(|_| anyhow::anyhow!("capture thread panicked"))??;
            let path = logs.join(&self.started.job_sha256).join(filename);
            write_new(&path, &bytes)?;
            *destination = Some(Artifact {
                path,
                sha256: crate::sha256_hex(&bytes),
            });
        }
        if outcome.command_status == EvaluationStatus::Completed {
            let retained = self
                .started
                .job
                .outputs
                .iter()
                .map(|name| {
                    let path = self.started.job.directory.join(name);
                    regular(&path)?;
                    ensure!(
                        fs::canonicalize(&path)?.starts_with(&self.started.job.directory),
                        "fit output escapes directory"
                    );
                    Ok(Artifact {
                        sha256: hash_file(&path)?,
                        path,
                    })
                })
                .collect::<Result<Vec<_>>>();
            match retained.and_then(|outputs| {
                self.started.job.verify_inputs()?;
                Ok(outputs)
            }) {
                Ok(outputs) => outcome.outputs = outputs,
                Err(error) => {
                    outcome.command_status = EvaluationStatus::Failed;
                    outcome.reason = Some(format!("output/input integrity failure: {error:#}"));
                }
            }
        }
        self.drained = true;
        // Cgroup directories hold counters: leave them for receipt recovery,
        // never remove source/evidence directories as routine pool cleanup.
        Ok(outcome)
    }
}
impl Drop for ProcessFit {
    fn drop(&mut self) {
        if !self.drained {
            let _ = control_write(&self.started.cgroup, "cgroup.kill", "1");
            if let Some(mut child) = self.child.take() {
                if self.status.is_none() {
                    let _ = killpg(Pid::from_raw(child.id() as i32), Signal::SIGKILL);
                }
                let _ = child.kill();
                if !matches!(child.try_wait(), Ok(Some(_))) {
                    // Own the eventual reap without blocking cleanup on D-state.
                    thread::spawn(move || {
                        let _ = child.wait();
                    });
                }
            }
            #[cfg(test)]
            if let Err(error) =
                tests::host_setup_probe(tests::HostSetupStage::AfterDropKill, &self.started)
            {
                eprintln!("host setup fault injection failed: {error:#}");
            }
        }
    }
}
