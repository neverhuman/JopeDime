use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use jain_smartcluster::{Client, ClientConfig, DispatchError, EventStream};
use jain_smartcluster_core::{
    AttachRequest, CancelMode, CancelRequest, CpuRequest, DaemonMode, Environment, EventOffset,
    EventRecord, IdempotencyKey, IoMode, JobEvent, JobId, JobRecord, JobSpec, JobState,
    MemoryRequest, PreemptionPolicy, PriorityClass, ProjectId, ResourceRequest, RetryPolicy,
    TenantId, Workload,
};
use jain_smartcluster_daemon::config::{
    AuthConfig, DaemonConfig, ExecutionConfig, SchedulerConfig, WorkerConfig,
};
use nix::sys::signal::{kill, Signal};
use nix::unistd::{getegid, geteuid, Pid};
use tempfile::TempDir;
use tokio::time::{sleep, timeout};

const INITIAL_READY_TIMEOUT: Duration = Duration::from_secs(10);
const RESTART_READY_BUDGET: Duration = Duration::from_secs(2);
const STATE_TIMEOUT: Duration = Duration::from_secs(10);
const RETRY_DELAY: Duration = Duration::from_millis(20);

struct RestartHarness {
    child: Option<Child>,
    socket_path: PathBuf,
    config_path: PathBuf,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    daemon_executable: PathBuf,
    cleanup_marker: String,
    _directory: TempDir,
}

impl RestartHarness {
    async fn start() -> Result<Self> {
        let directory = tempfile::tempdir().context("create restart test directory")?;
        let socket_path = directory.path().join("run/scqd.sock");
        let config_path = directory.path().join("scqd.toml");
        let stdout_path = directory.path().join("scqd.stdout");
        let stderr_path = directory.path().join("scqd.stderr");
        let daemon_executable = PathBuf::from(env!("CARGO_BIN_EXE_scqd"));
        let cleanup_marker = directory.path().display().to_string();
        let uid = geteuid().as_raw();
        let gid = getegid().as_raw();
        let config = DaemonConfig {
            socket_path: socket_path.clone(),
            state_path: directory.path().join("state/scqd.redb"),
            event_dir: directory.path().join("events"),
            cgroup_root: directory.path().join("not-a-cgroup"),
            socket_mode: 0o600,
            embedded: true,
            scheduler: SchedulerConfig {
                interactive_cpu_headroom_millicores: 0,
                interactive_ram_headroom_bytes: 0,
                recovery_quiet_ms: 50,
                ..SchedulerConfig::default()
            },
            auth: AuthConfig {
                admin_uids: vec![uid],
                system_uids: Vec::new(),
                system_principal: "sagemaker".to_owned(),
                web_uid: u32::MAX,
                web_principal: "jain-web".to_owned(),
                ..AuthConfig::default()
            },
            workers: WorkerConfig {
                jain_worker: daemon_executable.clone(),
                trusted_services: BTreeMap::new(),
                ..WorkerConfig::default()
            },
            execution: ExecutionConfig {
                worker_uid: (uid == 0).then_some(uid),
                worker_gid: (uid == 0).then_some(gid),
                checkpoint_timeout_ms: 200,
                terminate_grace_ms: 500,
                kill_grace_ms: 200,
                ..ExecutionConfig::default()
            },
            ..DaemonConfig::default()
        };
        fs::write(
            &config_path,
            toml::to_string(&config).context("serialize restart test config")?,
        )
        .context("write restart test config")?;

        let mut harness = Self {
            child: None,
            socket_path,
            config_path,
            stdout_path,
            stderr_path,
            daemon_executable,
            cleanup_marker,
            _directory: directory,
        };
        harness.spawn_child()?;
        harness.wait_ready(INITIAL_READY_TIMEOUT).await?;
        Ok(harness)
    }

    fn client(&self) -> Result<Client> {
        Client::with_config(ClientConfig {
            socket_path: self.socket_path.clone(),
            connect_timeout: Duration::from_millis(100),
            request_timeout: Duration::from_secs(2),
            wait_timeout: Some(STATE_TIMEOUT),
            ..ClientConfig::default()
        })
        .context("build restart test client")
    }

    fn spawn_child(&mut self) -> Result<()> {
        ensure!(self.child.is_none(), "scqd test child is already running");
        let stdout = append_log(&self.stdout_path).context("open daemon stdout log")?;
        let stderr = append_log(&self.stderr_path).context("open daemon stderr log")?;
        let child = Command::new(&self.daemon_executable)
            .arg("--config")
            .arg(&self.config_path)
            .env("RUST_LOG", "warn")
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .context("spawn isolated scqd")?;
        self.child = Some(child);
        Ok(())
    }

    async fn wait_ready(&mut self, budget: Duration) -> Result<Duration> {
        let started = Instant::now();
        let client = self.client()?;
        loop {
            let child = self.child.as_mut().context("scqd test child is absent")?;
            if let Some(status) = child.try_wait().context("inspect scqd process")? {
                bail!("scqd exited before readiness ({status})\n{}", self.logs());
            }
            if client.status().await.is_ok() {
                return Ok(started.elapsed());
            }
            if started.elapsed() >= budget {
                bail!(
                    "scqd readiness exceeded {budget:?} after {:?}\n{}",
                    started.elapsed(),
                    self.logs()
                );
            }
            sleep(RETRY_DELAY).await;
        }
    }

    fn crash(&mut self) -> Result<()> {
        let mut child = self.child.take().context("scqd test child is absent")?;
        kill(Pid::from_raw(child.id() as i32), Signal::SIGKILL).context("SIGKILL isolated scqd")?;
        let status = child.wait().context("reap crashed scqd")?;
        ensure!(!status.success(), "SIGKILLed scqd exited successfully");
        Ok(())
    }

    async fn restart(&mut self) -> Result<Duration> {
        let started = Instant::now();
        self.spawn_child()?;
        self.wait_ready(RESTART_READY_BUDGET).await?;
        let elapsed = started.elapsed();
        ensure!(
            elapsed <= RESTART_READY_BUDGET,
            "scqd restart took {elapsed:?}, budget is {RESTART_READY_BUDGET:?}"
        );
        Ok(elapsed)
    }

    fn logs(&self) -> String {
        format!(
            "stdout:\n{}\nstderr:\n{}",
            fs::read_to_string(&self.stdout_path).unwrap_or_else(|error| error.to_string()),
            fs::read_to_string(&self.stderr_path).unwrap_or_else(|error| error.to_string())
        )
    }
}

impl Drop for RestartHarness {
    fn drop(&mut self) {
        kill_processes_with_marker(&self.cleanup_marker);
        let Some(child) = self.child.as_mut() else {
            return;
        };
        if matches!(child.try_wait(), Ok(None)) {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[tokio::test]
async fn held_job_and_event_cursor_survive_abrupt_daemon_restart() -> Result<()> {
    let mut harness = RestartHarness::start().await?;
    let client = harness.client()?;
    let status = client
        .status()
        .await
        .context("read initial daemon status")?;
    assert_eq!(status.mode, DaemonMode::SafeSingleAllocation);
    assert!(!status.capabilities.cgroup_delegated);

    let blocker = client
        .submit(admin_spec(
            "restart-blocker",
            find_executable(&["/usr/bin/sleep", "/bin/sleep"])
                .context("direct sleep executable is required")?,
            vec!["30".to_owned()],
            Environment(BTreeMap::from([(
                "JAIN_SMARTCLUSTER_RESTART_TEST".to_owned(),
                harness.cleanup_marker.clone(),
            )])),
        ))
        .await
        .context("submit blocking allocation")?;
    wait_for_state(&client, blocker.id, JobState::Running).await?;

    let launch_marker = harness._directory.path().join("held-job-launched");
    let held_spec = admin_spec(
        "restart-held-job",
        find_executable(&["/usr/bin/touch", "/bin/touch"])
            .context("direct touch executable is required")?,
        vec![launch_marker.to_string_lossy().into_owned()],
        Environment::default(),
    );
    let queued = client
        .submit(held_spec.clone())
        .await
        .context("submit job behind blocking allocation")?;
    assert_eq!(queued.state, JobState::Queued);
    let held = client.pause(queued.id).await.context("hold queued job")?;
    assert_eq!(held.state, JobState::Paused);
    assert!(held.active_attempt.is_none());
    sleep(Duration::from_millis(50)).await;
    assert!(
        !launch_marker.exists(),
        "held workload launched before crash"
    );

    let mut before_crash = client
        .attach(AttachRequest {
            job_id: held.id,
            offset: EventOffset::ZERO,
            read_only: true,
        })
        .await
        .context("attach before daemon crash")?;
    let pause_event = next_transition(&mut before_crash.events, held.id, JobState::Paused).await?;
    drop(before_crash);

    let cancelled_blocker = client
        .cancel(CancelRequest {
            job_id: blocker.id,
            mode: CancelMode::Graceful { grace_ms: 200 },
        })
        .await
        .context("cancel blocking allocation")?;
    assert_eq!(cancelled_blocker.state, JobState::Cancelled);

    harness.crash()?;
    let restart_elapsed = harness.restart().await?;
    println!(
        "JAIN_RELEASE_METRIC {}",
        serde_json::json!({
            "schema_version": "jain.smartcluster.release-metric/v1",
            "metric": "durable_restart_ready",
            "observed": u64::try_from(restart_elapsed.as_nanos())
                .expect("restart duration fits u64 nanoseconds"),
            "limit": u64::try_from(RESTART_READY_BUDGET.as_nanos())
                .expect("restart budget fits u64 nanoseconds"),
            "comparison": "lte",
            "unit": "nanoseconds",
            "sample_count": 1,
        })
    );
    assert!(restart_elapsed <= RESTART_READY_BUDGET);

    let client = harness.client()?;
    let recovered = client
        .get(held.id)
        .await
        .context("get held job after restart")?
        .context("held job disappeared across restart")?;
    assert_eq!(recovered.state, JobState::Paused);
    assert!(recovered.active_attempt.is_none());
    let duplicate = client
        .submit(held_spec)
        .await
        .context("idempotently resubmit held job after restart")?;
    assert_eq!(duplicate.id, held.id);
    assert_eq!(duplicate.state, JobState::Paused);
    sleep(Duration::from_millis(50)).await;
    assert!(
        !launch_marker.exists(),
        "held workload launched during restart recovery"
    );

    let resume_offset = pause_event
        .offset
        .next()
        .context("pause event offset overflowed")?;
    let mut resumed = client
        .attach(AttachRequest {
            job_id: held.id,
            offset: resume_offset,
            read_only: true,
        })
        .await
        .context("resume durable event stream after restart")?;
    let cancelled = client
        .cancel(CancelRequest {
            job_id: held.id,
            mode: CancelMode::Graceful { grace_ms: 200 },
        })
        .await
        .context("cancel still-held job without releasing it")?;
    assert_eq!(cancelled.state, JobState::Cancelled);
    let cancel_event = next_transition(&mut resumed.events, held.id, JobState::Cancelled).await?;
    assert!(cancel_event.offset > pause_event.offset);
    drop(resumed);

    let mut replay = client
        .attach(AttachRequest {
            job_id: held.id,
            offset: EventOffset::ZERO,
            read_only: true,
        })
        .await
        .context("replay durable events after restart")?;
    let replayed_pause = next_transition(&mut replay.events, held.id, JobState::Paused).await?;
    let replayed_cancel = next_transition(&mut replay.events, held.id, JobState::Cancelled).await?;
    assert_eq!(replayed_pause.offset, pause_event.offset);
    assert_eq!(replayed_cancel.offset, cancel_event.offset);
    assert!(replayed_pause.offset < replayed_cancel.offset);
    assert!(!launch_marker.exists(), "held workload was ever launched");
    Ok(())
}

fn append_log(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

fn admin_spec(
    idempotency_key: &str,
    program: PathBuf,
    arguments: Vec<String>,
    environment: Environment,
) -> JobSpec {
    JobSpec {
        tenant: unix_tenant(),
        project: ProjectId::new("restart-chaos").expect("fixed project is valid"),
        idempotency_key: IdempotencyKey::new(idempotency_key)
            .expect("fixed idempotency key is valid"),
        priority: PriorityClass::System,
        workload: Workload::AdministratorArgv {
            program: program.to_string_lossy().into_owned(),
            arguments,
            environment,
            working_directory: None,
        },
        resources: ResourceRequest {
            cpu: CpuRequest::fixed(1),
            ram: MemoryRequest::fixed(1024 * 1024),
            scratch_bytes: 0,
            gpu: None,
        },
        retry: RetryPolicy {
            max_failures: 0,
            initial_backoff_ms: 1,
            maximum_backoff_ms: 1,
            jitter_per_mille: 0,
            retry_on: BTreeSet::new(),
        },
        preemption: PreemptionPolicy::non_preemptible(),
        io: IoMode::Pipes,
        deadline: None,
        labels: BTreeMap::new(),
    }
}

fn unix_tenant() -> TenantId {
    TenantId::new(format!("unix:{}", geteuid().as_raw())).expect("current UID is a valid tenant")
}

fn find_executable(candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(Path::new)
        .find(|path| path.is_file())
        .map(Path::to_path_buf)
}

async fn wait_for_state(client: &Client, job_id: JobId, expected: JobState) -> Result<JobRecord> {
    let started = Instant::now();
    loop {
        let remaining = STATE_TIMEOUT.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            bail!("job {job_id} did not reach {expected:?} within {STATE_TIMEOUT:?}");
        }
        let response = timeout(remaining, client.get(job_id))
            .await
            .with_context(|| {
                format!("job {job_id} state deadline while waiting for {expected:?}")
            })?;
        let job = match response {
            // GET is idempotent. Keep polling inside the existing state budget
            // when the fixture's shorter request deadline expires under load.
            Err(DispatchError::Timeout { .. }) => {
                sleep(RETRY_DELAY.min(remaining)).await;
                continue;
            }
            response => response
                .with_context(|| format!("get job {job_id} while waiting for {expected:?}"))?
                .with_context(|| format!("job {job_id} disappeared"))?,
        };
        if job.state == expected {
            return Ok(job);
        }
        if job.state.is_terminal() {
            bail!(
                "job {job_id} reached {:?} while waiting for {expected:?}",
                job.state
            );
        }
        if started.elapsed() >= STATE_TIMEOUT {
            bail!(
                "job {job_id} remained {:?} while waiting for {expected:?}",
                job.state
            );
        }
        sleep(RETRY_DELAY).await;
    }
}

async fn next_transition(
    events: &mut EventStream,
    job_id: JobId,
    expected: JobState,
) -> Result<EventRecord> {
    let started = Instant::now();
    loop {
        let remaining = STATE_TIMEOUT.saturating_sub(started.elapsed());
        ensure!(
            !remaining.is_zero(),
            "event stream did not reach {expected:?}"
        );
        let event = timeout(remaining, events.recv())
            .await
            .context("durable event read timed out")?
            .context("durable event stream closed")?
            .context("durable event stream returned an error")?;
        ensure!(
            event.job_id == job_id,
            "attachment leaked another job event"
        );
        if matches!(
            &event.event,
            JobEvent::StateTransition(transition) if transition.to == expected
        ) {
            return Ok(event);
        }
    }
}

fn kill_processes_with_marker(value: &str) {
    let marker = format!("JAIN_SMARTCLUSTER_RESTART_TEST={value}");
    let Ok(entries) = fs::read_dir("/proc") else {
        return;
    };
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(environment) = fs::read(entry.path().join("environ")) else {
            continue;
        };
        if environment
            .split(|byte| *byte == 0)
            .any(|entry| entry == marker.as_bytes())
        {
            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
        }
    }
}
