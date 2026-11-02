use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use jain_smartcluster::{Client, ClientConfig, DispatchError};
use jain_smartcluster_core::{
    AttachRequest, CancelMode, CancelRequest, DaemonMode, Environment, ErrorCode, EventOffset,
    IdempotencyKey, IoMode, JainWorkload, JobEvent, JobId, JobRecord, JobSpec, JobState,
    ListRequest, MemoryRequest, PreemptionPolicy, Principal, PriorityClass, ProjectId,
    ResourceRequest, RetryPolicy, TenantId, TerminalStatus, Workload,
};
use jain_smartcluster_daemon::config::{
    AuthConfig, DaemonConfig, ExecutionConfig, SchedulerConfig, WorkerConfig,
};
use nix::sys::signal::{kill, Signal};
use nix::unistd::{getegid, geteuid, Pid};
use tempfile::TempDir;
use tokio::time::{sleep, timeout, Instant};

const READY_TIMEOUT: Duration = Duration::from_secs(10);
const STATE_TIMEOUT: Duration = Duration::from_secs(10);
const RETRY_DELAY: Duration = Duration::from_millis(20);

struct DaemonHarness {
    child: Child,
    socket_path: PathBuf,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    daemon_executable: PathBuf,
    cleanup_marker: String,
    _directory: TempDir,
}

impl DaemonHarness {
    async fn spawn(administrator: bool) -> Result<Self> {
        let directory = tempfile::tempdir().context("create daemon test directory")?;
        let socket_path = directory.path().join("run/scqd.sock");
        let stdout_path = directory.path().join("scqd.stdout");
        let stderr_path = directory.path().join("scqd.stderr");
        let config_path = directory.path().join("scqd.toml");
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
                admin_uids: administrator.then_some(uid).into_iter().collect(),
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
            toml::to_string(&config).context("serialize daemon test config")?,
        )
        .context("write daemon test config")?;
        let stdout = File::create(&stdout_path).context("create daemon stdout log")?;
        let stderr = File::create(&stderr_path).context("create daemon stderr log")?;
        let child = Command::new(&daemon_executable)
            .arg("--config")
            .arg(&config_path)
            .env("RUST_LOG", "warn")
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .context("spawn isolated scqd")?;

        let mut harness = Self {
            child,
            socket_path,
            stdout_path,
            stderr_path,
            daemon_executable,
            cleanup_marker,
            _directory: directory,
        };
        harness.wait_ready().await?;
        Ok(harness)
    }

    fn client(&self) -> Result<Client> {
        Client::with_config(ClientConfig {
            socket_path: self.socket_path.clone(),
            connect_timeout: Duration::from_secs(1),
            request_timeout: Duration::from_secs(3),
            wait_timeout: Some(STATE_TIMEOUT),
            ..ClientConfig::default()
        })
        .context("build daemon test client")
    }

    async fn wait_ready(&mut self) -> Result<()> {
        let client = self.client()?;
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            if let Some(status) = self.child.try_wait().context("inspect scqd process")? {
                bail!("scqd exited before readiness ({status})\n{}", self.logs());
            }
            if client.status().await.is_ok() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                bail!("scqd readiness timed out\n{}", self.logs());
            }
            sleep(RETRY_DELAY).await;
        }
    }

    fn logs(&self) -> String {
        format!(
            "stdout:\n{}\nstderr:\n{}",
            fs::read_to_string(&self.stdout_path).unwrap_or_else(|error| error.to_string()),
            fs::read_to_string(&self.stderr_path).unwrap_or_else(|error| error.to_string())
        )
    }
}

impl Drop for DaemonHarness {
    fn drop(&mut self) {
        kill_processes_with_marker(&self.cleanup_marker);
        let Ok(None) = self.child.try_wait() else {
            return;
        };

        let daemon_pid = self.child.id();
        let _ = kill(Pid::from_raw(daemon_pid as i32), Signal::SIGSTOP);
        for descendant in descendants_of(daemon_pid).into_iter().rev() {
            let _ = kill(Pid::from_raw(descendant as i32), Signal::SIGKILL);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
#[path = "daemon_e2e/cases.rs"]
mod cases;

fn admin_spec(
    idempotency_key: &str,
    program: PathBuf,
    arguments: Vec<String>,
    environment: Environment,
) -> JobSpec {
    JobSpec {
        tenant: unix_tenant(),
        project: ProjectId::new("daemon-e2e").expect("fixed project is valid"),
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
            cpu: jain_smartcluster_core::CpuRequest::fixed(1),
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

fn jain_spec(idempotency_key: &str, tenant: TenantId, environment: Environment) -> JobSpec {
    JobSpec {
        tenant,
        project: ProjectId::new("daemon-e2e").expect("fixed project is valid"),
        idempotency_key: IdempotencyKey::new(idempotency_key)
            .expect("fixed idempotency key is valid"),
        priority: PriorityClass::System,
        workload: Workload::Jain {
            template: JainWorkload::Training,
            arguments: Vec::new(),
            environment,
        },
        resources: ResourceRequest::cpu_and_ram(1, 1024 * 1024),
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

fn sleep_executable() -> Result<PathBuf> {
    [Path::new("/usr/bin/sleep"), Path::new("/bin/sleep")]
        .into_iter()
        .find(|path| path.is_file())
        .map(Path::to_path_buf)
        .context("a direct sleep executable is required for the queueing test")
}

async fn wait_for_state(client: &Client, job_id: JobId, expected: JobState) -> Result<JobRecord> {
    let deadline = Instant::now() + STATE_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
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
        if Instant::now() >= deadline {
            bail!(
                "job {job_id} remained {:?} while waiting for {expected:?}",
                job.state
            );
        }
        sleep(RETRY_DELAY).await;
    }
}

async fn assert_durable_output(client: &Client, job_id: JobId, expected: &[u8]) -> Result<()> {
    let mut attachment = client
        .attach(AttachRequest {
            job_id,
            offset: EventOffset::ZERO,
            read_only: true,
        })
        .await
        .context("attach from initial durable offset")?;
    let deadline = Instant::now() + STATE_TIMEOUT;
    let mut output = Vec::new();
    let mut saw_completed = false;
    while !saw_completed
        || !output
            .windows(expected.len())
            .any(|window| window == expected)
    {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("durable attach did not replay completion and expected output");
        }
        let event = timeout(remaining, attachment.events.recv())
            .await
            .context("durable attach replay timed out")?
            .context("durable attach closed early")?
            .context("durable attach returned an error")?;
        match event.event {
            JobEvent::Stdout(bytes) => output.extend(bytes),
            JobEvent::StateTransition(transition) => {
                saw_completed |= transition.to == JobState::Completed;
            }
            _ => {}
        }
    }
    Ok(())
}

fn assert_server_code(result: Result<JobRecord, DispatchError>, expected: ErrorCode) {
    match result {
        Err(DispatchError::Server(error)) => assert_eq!(error.code, expected),
        Err(error) => panic!("expected server error {expected:?}, got {error}"),
        Ok(job) => panic!("expected server error {expected:?}, got job {}", job.id),
    }
}

fn descendants_of(root: u32) -> Vec<u32> {
    let mut descendants = Vec::new();
    let mut parents = vec![root];
    loop {
        let mut added = false;
        let Ok(entries) = fs::read_dir("/proc") else {
            break;
        };
        for entry in entries.flatten() {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u32>().ok())
            else {
                continue;
            };
            if descendants.contains(&pid) || pid == root {
                continue;
            }
            let Some(parent) = process_parent(pid) else {
                continue;
            };
            if parents.contains(&parent) {
                descendants.push(pid);
                parents.push(pid);
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    descendants
}

fn process_parent(pid: u32) -> Option<u32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_name = stat.rsplit_once(") ")?.1;
    after_name.split_whitespace().nth(1)?.parse().ok()
}

fn kill_processes_with_marker(value: &str) {
    let marker = format!("JAIN_SMARTCLUSTER_E2E={value}");
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
