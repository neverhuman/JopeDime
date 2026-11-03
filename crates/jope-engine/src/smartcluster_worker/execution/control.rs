use super::*;

/// Cooperative scratch accounting supplements the daemon's reservation. A quota
/// is not claimed: immutable numerical outputs remain on the guarded data root.
pub(crate) struct ScratchGuard {
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl ScratchGuard {
    pub(crate) fn start(
        allocation: &Allocation,
        task: &TaskManifest,
        state: &Path,
    ) -> Result<Self> {
        let done = Arc::new(AtomicBool::new(false));
        if task.resources.scratch_bytes == 0 {
            return Ok(Self { done, thread: None });
        }
        let inherited_root = PathBuf::from(
            std::env::var_os("SCQ_SCRATCH_ROOT").context("native scratch root absent")?,
        );
        let inherited_budget: u64 = std::env::var("SCQ_SCRATCH_RESERVATION_BYTES")?.parse()?;
        ensure!(
            inherited_root == allocation.policy.scratch_root
                && inherited_budget == allocation.bootstrap.resources.scratch_bytes
                && task.resources.scratch_bytes <= inherited_budget,
            "native scratch reservation changed"
        );
        let allocation_root = inherited_root.join(allocation.attempt.job_id.to_string());
        let path = allocation_root.join(hash(&serde_json::to_vec(&(
            &task.run,
            task.generation,
            &task.task_id,
        ))?));
        fs::create_dir_all(&path)?;
        let within = |usage: ScratchUsage| {
            usage.task <= task.resources.scratch_bytes
                && usage.allocation <= inherited_budget
                && usage.root <= SCRATCH_ROOT_LIMIT
        };
        ensure!(
            within(scratch_usage(&inherited_root, &allocation_root, &path)?),
            "retained scratch exceeds task reservation"
        );
        std::env::set_var("TMPDIR", &path);
        let observed = Arc::clone(&done);
        let budget = task.resources.scratch_bytes;
        let state = state.to_owned();
        let thread = std::thread::spawn(move || {
            while !observed.load(Ordering::Acquire) {
                // One walk of the shared root per tick yields all three totals (the task and
                // allocation directories are inside it); walking each tree separately stat'ed
                // the task's files three times every 100 ms.
                let usage = scratch_usage(&inherited_root, &allocation_root, &path);
                if !usage.is_ok_and(|usage| {
                    usage.task <= budget
                        && usage.allocation <= inherited_budget
                        && usage.root <= SCRATCH_ROOT_LIMIT
                }) {
                    let _ = write_json(
                        &state.join("scratch-violation.json"),
                        &json!({"budget":budget,"path":path,"action":"terminate_and_preserve_attempt"}),
                    );
                    // SAFETY: _exit has no preconditions. Skipping destructors and flushes is
                    // intended: the over-budget task stops now, after a best-effort record.
                    unsafe {
                        libc::_exit(78);
                    }
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        Ok(Self {
            done,
            thread: Some(thread),
        })
    }
}
impl Drop for ScratchGuard {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
/// Byte cap on the whole inherited scratch root.
const SCRATCH_ROOT_LIMIT: u64 = 32 * 1024 * 1024 * 1024;

/// Regular-file bytes under the scratch root, its allocation directory and the task directory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::smartcluster_worker::policy) struct ScratchUsage {
    pub(in crate::smartcluster_worker::policy) root: u64,
    pub(in crate::smartcluster_worker::policy) allocation: u64,
    pub(in crate::smartcluster_worker::policy) task: u64,
}

/// Sums [`ScratchUsage`] in one walk of `root` (`allocation` and `task` are inside it). A
/// symlink anywhere is refused. A file or directory that disappears between the directory read
/// and its `lstat` is a temporary file the task removed (TMPDIR points here), not a violation, so
/// it is skipped. The root, allocation and task directories themselves must exist (each
/// separate walk used to fail on a missing one): the root as a directory, possibly behind a link
/// as before (walkdir follows a root link), the other two as physical directories. Every other
/// error still fails closed. Entries renamed during a walk are counted on the next tick.
pub(in crate::smartcluster_worker::policy) fn scratch_usage(
    root: &Path,
    allocation: &Path,
    task: &Path,
) -> Result<ScratchUsage> {
    let root_metadata =
        fs::metadata(root).with_context(|| format!("scratch root {}", root.display()))?;
    ensure!(root_metadata.is_dir(), "scratch root is not a directory");
    for directory in [allocation, task] {
        let metadata = fs::symlink_metadata(directory)
            .with_context(|| format!("scratch directory {}", directory.display()))?;
        ensure!(
            metadata.is_dir(),
            "scratch path is not a physical directory: {}",
            directory.display()
        );
    }
    let vanished = |error: &std::io::Error| error.kind() == std::io::ErrorKind::NotFound;
    let mut usage = ScratchUsage {
        root: 0,
        allocation: 0,
        task: 0,
    };
    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) if error.depth() > 0 && error.io_error().is_some_and(vanished) => continue,
            Err(error) => return Err(error.into()),
        };
        ensure!(!entry.file_type().is_symlink(), "scratch symlink rejected");
        if !entry.file_type().is_file() {
            continue;
        }
        let bytes = match entry.metadata() {
            Ok(metadata) => metadata.len(),
            Err(error) if error.io_error().is_some_and(vanished) => continue,
            Err(error) => return Err(error.into()),
        };
        let add = |total: u64| total.checked_add(bytes).context("scratch size overflow");
        usage.root = add(usage.root)?;
        if entry.path().starts_with(allocation) {
            usage.allocation = add(usage.allocation)?;
            if entry.path().starts_with(task) {
                usage.task = add(usage.task)?;
            }
        }
    }
    Ok(usage)
}

pub(crate) struct DeadlineWatch {
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl DeadlineWatch {
    pub(crate) fn start(stop: Arc<AtomicBool>, deadline: u64, allowance: u64) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&done);
        let remaining =
            Duration::from_millis(deadline.saturating_sub(resident::now_ms().unwrap_or(u64::MAX)));
        let started = std::time::Instant::now();
        let thread = std::thread::spawn(move || {
            while !observed.load(Ordering::Acquire) {
                let now = resident::now_ms().unwrap_or(u64::MAX);
                if now >= deadline || started.elapsed() >= remaining {
                    stop.store(true, Ordering::Release);
                }
                if now >= deadline.saturating_add(allowance)
                    || started.elapsed()
                        >= remaining.saturating_add(Duration::from_millis(allowance))
                {
                    // SAFETY: _exit has no preconditions. Skipping destructors and flushes is
                    // intended: past deadline plus allowance the process is ended at once.
                    unsafe {
                        libc::_exit(76);
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        Self {
            done,
            thread: Some(thread),
        }
    }
}
impl Drop for DeadlineWatch {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ControlStopReason {
    EndOfFile,
    FrameReadError,
    Checkpoint,
    MalformedNamedControl,
    UnknownControl,
    QueueFull,
    QueueDisconnected,
}

#[derive(Clone, Copy)]
pub(crate) struct ControlStopDeadline {
    instant: std::time::Instant,
    monotonic_ns: Option<u64>,
}

pub(crate) fn stop_control_with(
    stop: &AtomicBool,
    allowance: Duration,
    watchdog: impl FnOnce(std::time::Instant),
    persist: impl FnOnce(ControlStopDeadline) -> Result<()>,
) -> Result<()> {
    let now = std::time::Instant::now();
    let instant = now.checked_add(allowance).unwrap_or(now);
    stop.store(true, Ordering::Release);
    // Arm independently of all diagnostic filesystem I/O.
    watchdog(instant);
    let monotonic_ns = crate::host_os::monotonic_ns().ok().and_then(|n| {
        u64::try_from(
            instant
                .saturating_duration_since(std::time::Instant::now())
                .as_nanos(),
        )
        .ok()
        .and_then(|v| n.checked_add(v))
    });
    persist(ControlStopDeadline {
        instant,
        monotonic_ns,
    })
}

pub(crate) fn record_control_stop(
    state_root: &Path,
    attempt: FencedAttempt,
    sequence: u64,
    reason: ControlStopReason,
    deadline: ControlStopDeadline,
) -> Result<()> {
    let value = json!({"schema":"jope.smartcluster_control_stop.v1","attempt":attempt,
        "sequence":sequence,"reason":reason,"observed_at_unix_ms":resident::now_ms()?,
        "checkpoint_deadline_monotonic_ns":deadline.monotonic_ns,
        "checkpoint_remaining_ms":deadline.instant.saturating_duration_since(std::time::Instant::now()).as_millis()});
    ensure!(
        serde_json::to_vec(&value)?.len() <= 4096,
        "control stop receipt exceeds bound"
    );
    write_immutable_record(
        &state_root.join(format!(
            "control-stop-{}-{}.json",
            attempt.job_id, attempt.fence
        )),
        &value,
    )
}
