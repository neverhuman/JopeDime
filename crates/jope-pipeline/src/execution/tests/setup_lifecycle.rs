use super::*;

pub(super) fn qualify_cancellation_and_resume(
    mut pool: ExecutionPool,
    physical: &Path,
    case: &str,
) {
    let script = physical.join("cancel-descendant.py");
    write_new(&script, b"import os,time\npid=os.fork()\nif pid==0:\n os.setsid()\n time.sleep(60)\nelse:\n open('descendant.pid','w').write(str(pid))\n time.sleep(60)\n").unwrap();
    let program = fs::canonicalize("/usr/bin/python3").unwrap();
    let mut job = fixture_job('a', 'a');
    job.request.experiment_sha256 = pool.experiment.identity().unwrap();
    job.directory = pool
        .directory
        .join("jobs")
        .join(job.request.identity().unwrap());
    job.command = WorkerCommand {
        program: Artifact {
            sha256: hash_file(&program).unwrap(),
            path: program,
        },
        args: vec![script.to_string_lossy().into_owned()],
        inputs: vec![Artifact {
            sha256: hash_file(&script).unwrap(),
            path: script,
        }],
        environment: BTreeMap::new(),
    };
    let cancelled = crate::supervisor::cancellation().unwrap();
    let original = pool.admit().unwrap().clone();
    let original_bytes = fs::read(pool.directory.join("window.json")).unwrap();
    let child_pid_path = job.directory.join("descendant.pid");
    thread::scope(|scope| {
        let signal = if case == "graceful-sigterm" {
            Some(scope.spawn(|| {
                let deadline = Instant::now() + Duration::from_secs(1);
                while !child_pid_path.is_file() {
                    assert!(Instant::now() < deadline, "real descendant did not start");
                    thread::sleep(Duration::from_millis(5));
                }
                nix::sys::signal::kill(Pid::this(), Signal::SIGTERM).unwrap();
            }))
        } else {
            None
        };
        let outcomes = pool.run(vec![job.clone()], &cancelled).unwrap();
        if let Some(signal) = signal {
            signal.join().unwrap();
        }
        assert_eq!(outcomes.len(), 1);
        let outcome = &outcomes[0];
        let expected = if case == "graceful-sigterm" {
            EvaluationStatus::Cancelled
        } else {
            EvaluationStatus::TimedOut
        };
        assert_eq!(outcome.command_status, expected);
        assert_eq!(
            cancelled.load(Ordering::Acquire),
            case == "graceful-sigterm"
        );
        assert!(outcome.consumption.gate_released_unix_ms.is_some());
        if case == "original-cutoff" {
            let started: Started = read_json(&pool.started_path(&job.identity().unwrap())).unwrap();
            assert_eq!(
                started.fit_deadline_unix_ms,
                original.deadline_unix_ms - pool.experiment.resources.reporting_seconds * 1000
            );
            assert!(
                outcome.consumption.observed_exit_unix_ms.unwrap() >= started.fit_deadline_unix_ms
            );
        }
        let allocation = fit_memory_allocation(&pool.experiment).unwrap();
        assert_eq!(
            outcome.consumption.fit_memory_limit_bytes,
            Some(allocation.per_fit_bytes)
        );
        let group = pool
            .supervisor
            .cgroup_root
            .join(format!("fit-{}", job.identity().unwrap()));
        assert!(
            !group.exists() || named_counter(&group, "cgroup.events", "populated").unwrap() == 0
        );
        assert!(!pool.cleanup_path(&job.identity().unwrap()).exists());
        let descendant: u32 = fs::read_to_string(&child_pid_path)
            .unwrap()
            .parse()
            .unwrap();
        if let Ok(stat) = fs::read_to_string(format!("/proc/{descendant}/stat")) {
            assert!(matches!(
                stat.rsplit(')').next().unwrap().split_whitespace().next(),
                Some("Z" | "X")
            ));
        }
        // A second run uses the same immutable original attempt/window; it
        // must neither release another gate nor grant a replacement deadline.
        let retained = fs::read(pool.finished_path(&job.identity().unwrap())).unwrap();
        let experiment = pool.experiment.clone();
        let journal = pool.directory.clone();
        let supervisor = pool.supervisor.clone();
        drop(pool);
        let mut resumed_pool = ExecutionPool::open(&experiment, &journal, supervisor).unwrap();
        let resumed = resumed_pool
            .run(vec![job.clone()], &AtomicBool::new(false))
            .unwrap();
        assert_eq!(
            serde_json::to_vec(&resumed).unwrap(),
            serde_json::to_vec(&outcomes).unwrap()
        );
        assert_eq!(
            fs::read(journal.join("window.json")).unwrap(),
            original_bytes
        );
        assert_eq!(
            fs::read(resumed_pool.finished_path(&job.identity().unwrap())).unwrap(),
            retained
        );
        assert_eq!(
            resumed_pool.window().unwrap().deadline_unix_ms,
            original.deadline_unix_ms
        );
        let ended = now_ms().unwrap();
        assert!(
            ended < original.deadline_unix_ms,
            "cleanup consumed beyond global deadline"
        );
        println!(
            "execution_host_receipt={}",
            serde_json::json!({
                "case":case,"original_window":original,"resumed_window":resumed_pool.window(),
                "journal":journal,"window_sha256":hash_file(&journal.join("window.json")).unwrap(),
                "finished_sha256":hash_file(&resumed_pool.finished_path(&job.identity().unwrap())).unwrap(),
                "allocation_file_sha256":hash_file(&journal.join("fit-memory-policy.json")).unwrap(),
                "outcome":outcome,"descendant_pid":descendant,"descendants_drained":true,
                "resumed_without_new_gate":true,"fit_memory_allocation":allocation,
                "execution_cutoff_unix_ms":original.deadline_unix_ms-experiment.resources.reporting_seconds*1000,
                "completed_unix_ms":ended,"remaining_global_ms":original.deadline_unix_ms-ended
            })
        );
    });
}

// Installed only by the explicitly ignored real-host fixture. No environment
// or fault-injection path is compiled into a production executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HostSetupStage {
    BeforeSetup,
    Attached,
    AfterDropKill,
}
pub(super) type HostSetupCallback =
    dyn Fn(HostSetupStage, &Started) -> Result<()> + Send + Sync + 'static;
static HOST_SETUP_PROBE: std::sync::Mutex<Option<Arc<HostSetupCallback>>> =
    std::sync::Mutex::new(None);

pub(super) fn host_setup_probe(stage: HostSetupStage, started: &Started) -> Result<()> {
    // Release the registry lock before a callback blocks. The barrier must
    // permit two actual ProcessFit::start calls to enter concurrently.
    let callback = HOST_SETUP_PROBE.lock().unwrap().clone();
    if let Some(callback) = callback {
        callback(stage, started)?;
    }
    Ok(())
}
pub(super) struct HostSetupGuard;
impl HostSetupGuard {
    pub(super) fn install(callback: Arc<HostSetupCallback>) -> Self {
        let mut probe = HOST_SETUP_PROBE.lock().unwrap();
        assert!(probe.is_none(), "another real-host setup probe is active");
        *probe = Some(callback);
        Self
    }
}
impl Drop for HostSetupGuard {
    fn drop(&mut self) {
        *HOST_SETUP_PROBE.lock().unwrap() = None;
    }
}
#[derive(Default)]
pub(super) struct OwnedSetupOrphan(std::sync::Mutex<Option<Child>>);
impl OwnedSetupOrphan {
    pub(super) fn insert(&self, started: &Started) -> Result<u32> {
        let mut slot = self.0.lock().unwrap();
        ensure!(slot.is_none(), "setup orphan was already injected");
        // This deliberately recreates a still-live owned group *after* the
        // ordinary ProcessFit Drop kill. It tests pool recovery, not Drop.
        // The guard reaps it on failure; systemd also bounds its whole tree.
        let child = Command::new("/bin/sleep")
            .arg("10")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()?;
        let pid = child.id();
        *slot = Some(child);
        control_write(&started.cgroup, "cgroup.procs", &pid.to_string())?;
        let expected = started
            .cgroup
            .strip_prefix("/sys/fs/cgroup")?
            .to_string_lossy();
        ensure!(
            fs::read_to_string(format!("/proc/{pid}/cgroup"))?
                .lines()
                .any(|line| line == format!("0::/{expected}")),
            "synthetic owned orphan attachment differs"
        );
        ensure!(
            named_counter(&started.cgroup, "cgroup.events", "populated")? == 1,
            "synthetic owned orphan was not alive in the failed group"
        );
        Ok(pid)
    }
    fn reaped_after_recovery(&self) -> Result<u32> {
        let mut slot = self.0.lock().unwrap();
        let child = slot.as_mut().context("no owned orphan was injected")?;
        let deadline = Instant::now() + Duration::from_millis(DRAIN_MS);
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            ensure!(
                Instant::now() < deadline,
                "pool returned while owned setup orphan remained alive"
            );
            thread::sleep(Duration::from_millis(POLL_MS));
        };
        use std::os::unix::process::ExitStatusExt;
        ensure!(
            status.signal() == Some(9),
            "owned orphan was not killed by recovery"
        );
        Ok(child.id())
    }
}
impl Drop for OwnedSetupOrphan {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
pub(super) fn setup_host_job(pool: &ExecutionPool, script: &Path, recipe: char) -> WorkerJob {
    let mut job = fixture_job('a', recipe);
    job.request.experiment_sha256 = pool.experiment.identity().unwrap();
    job.directory = pool
        .directory
        .join("jobs")
        .join(job.request.identity().unwrap());
    let program = fs::canonicalize("/usr/bin/python3").unwrap();
    job.command = WorkerCommand {
        program: Artifact {
            sha256: hash_file(&program).unwrap(),
            path: program,
        },
        args: vec![script.to_string_lossy().into_owned()],
        inputs: vec![Artifact {
            sha256: hash_file(script).unwrap(),
            path: script.to_owned(),
        }],
        environment: BTreeMap::new(),
    };
    job
}
pub(super) fn setup_group_drained(group: &Path) -> Result<bool> {
    Ok(!group.exists() || named_counter(group, "cgroup.events", "populated")? == 0)
}
