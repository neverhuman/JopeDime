//! Real delegated-cgroup recovery tests. Never uses production sockets or process scans.
use anyhow::{Context, Result, ensure};
use jain_smartcluster::{Client, ClientConfig};
use jain_smartcluster_core::*;
use jain_smartcluster_daemon::{
    cgroup::{CgroupManager, IsolationMode},
    config::{AuthConfig, DaemonConfig, ExecutionConfig, SchedulerConfig, WorkerConfig},
};
use nix::{
    fcntl::{FcntlArg, FdFlag, fcntl},
    unistd::{getegid, geteuid},
};
use std::os::unix::process::CommandExt;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    str::FromStr,
    time::{Duration, Instant},
};
use tempfile::TempDir;
use tokio::time::sleep;

const SERVICE: &str = "jope-resident-gpu";
const WAIT: Duration = Duration::from_secs(30);

use lease_tests::populated;

// Safe raw-FD socket APIs keep the fixture compatible with workspace unsafe_code=deny.
struct Control(i32);
impl Read for Control {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        nix::sys::socket::recv(self.0, bytes, nix::sys::socket::MsgFlags::empty())
            .map_err(std::io::Error::from)
    }
}
impl Write for Control {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        nix::sys::socket::send(self.0, bytes, nix::sys::socket::MsgFlags::MSG_NOSIGNAL)
            .map_err(std::io::Error::from)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl Drop for Control {
    fn drop(&mut self) {
        let _ = nix::unistd::close(self.0);
    }
}

fn durable(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(!path.exists(), "immutable fixture receipt already exists");
    let temporary = path.with_extension(format!(
        "{}-{}.pending",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let mut out = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    out.write_all(bytes)?;
    out.sync_all()?;
    drop(out);
    fs::rename(&temporary, path)?;
    File::open(path.parent().context("receipt parent")?)?.sync_all()?;
    Ok(())
}

fn own_cgroup() -> Result<PathBuf> {
    let text = fs::read_to_string("/proc/self/cgroup")?;
    let rel = text
        .lines()
        .find_map(|v| v.strip_prefix("0::"))
        .context("unified cgroup")?;
    Ok(Path::new("/sys/fs/cgroup").join(rel.trim_start_matches('/')))
}

fn report(
    socket: &mut Control,
    attempt: FencedAttempt,
    sequence: u64,
    report: WorkerReport,
) -> Result<()> {
    let data = postcard::to_stdvec(&WorkerReportFrame {
        version: ProtocolVersion::CURRENT,
        attempt,
        sequence,
        report,
    })?;
    socket.write_all(&u32::try_from(data.len())?.to_be_bytes())?;
    socket.write_all(&data)?;
    Ok(())
}

struct Harness {
    dir: TempDir,
    cgroup: PathBuf,
    config: PathBuf,
    socket: PathBuf,
    daemon: PathBuf,
    child: Option<Child>,
}
impl Harness {
    async fn start(checkpoint_ms: u64) -> Result<Option<Self>> {
        Self::start_with_gpu_baselines(checkpoint_ms, BTreeMap::new()).await
    }
    async fn start_with_gpu_baselines(
        checkpoint_ms: u64,
        gpu_idle_baselines: BTreeMap<String, jain_smartcluster_daemon::gpu::QualifiedIdleBaseline>,
    ) -> Result<Option<Self>> {
        Self::start_inner(checkpoint_ms, gpu_idle_baselines, None).await
    }
    async fn start_inner(
        checkpoint_ms: u64,
        gpu_idle_baselines: BTreeMap<String, jain_smartcluster_daemon::gpu::QualifiedIdleBaseline>,
        fit_fixture: Option<PathBuf>,
    ) -> Result<Option<Self>> {
        let Some(base) = std::env::var_os("SCQ_TEST_CGROUP_ROOT") else {
            eprintln!("SKIP: SCQ_TEST_CGROUP_ROOT not supplied");
            return Ok(None);
        };
        let base = fs::canonicalize(base)?;
        ensure!(
            base.starts_with("/sys/fs/cgroup") && base.join("cgroup.controllers").is_file(),
            "test root must be a real delegated cgroup"
        );
        ensure!(
            fs::read_to_string(base.join("cgroup.procs"))?
                .trim()
                .is_empty(),
            "delegated test root must have an empty parent and supervisor subgroup"
        );
        fs::write(
            base.join("cgroup.subtree_control"),
            "+cpu +memory +pids +cpuset",
        )?;
        let mut dir = if let Some(path) = std::env::var_os("SCQ_TEST_ARTIFACTS") {
            fs::create_dir_all(&path)?;
            tempfile::Builder::new()
                .prefix("resident-kernel-")
                .tempdir_in(path)?
        } else {
            tempfile::tempdir()?
        };
        if std::env::var_os("SCQ_TEST_ARTIFACTS").is_some() {
            dir.disable_cleanup(true);
            eprintln!("kernel evidence: {}", dir.path().display());
        }
        let name = format!(
            "resident-kernel-{}-{}",
            std::process::id(),
            dir.path().file_name().unwrap().to_string_lossy()
        );
        let cgroup = base.join(name);
        if let Err(error) = fs::create_dir(&cgroup) {
            eprintln!("SKIP: cannot create delegated child: {error}");
            return Ok(None);
        }
        match CgroupManager::initialize(&cgroup, false) {
            Ok(manager) if manager.mode() == IsolationMode::Delegated => {}
            other => {
                eprintln!(
                    "SKIP: delegation unavailable: {}",
                    other
                        .err()
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| "safe mode".into())
                );
                let _ = fs::remove_dir(&cgroup);
                return Ok(None);
            }
        }
        fs::create_dir(cgroup.join("supervisor"))?;
        let socket = dir.path().join("run/scqd.sock");
        let config = dir.path().join("scqd.toml");
        let daemon = std::env::var_os("SCQ_TEST_DAEMON")
            .map(PathBuf::from)
            .or_else(|| option_env!("CARGO_BIN_EXE_scqd").map(PathBuf::from))
            .context("test scqd executable unavailable")?;
        let uid = geteuid().as_raw();
        let gid = getegid().as_raw();
        let mut cfg = DaemonConfig {
            socket_path: socket.clone(),
            state_path: dir.path().join("state/scqd.redb"),
            event_dir: dir.path().join("events"),
            cgroup_root: cgroup.clone(),
            embedded: true,
            socket_mode: 0o600,
            scheduler: SchedulerConfig {
                interactive_cpu_headroom_millicores: 0,
                interactive_ram_headroom_bytes: 0,
                recovery_quiet_ms: 20,
                ..Default::default()
            },
            auth: AuthConfig {
                admin_uids: vec![uid],
                system_uids: vec![],
                system_principal: "sagemaker".into(),
                web_uid: u32::MAX,
                web_principal: "jain-web".into(),
                ..AuthConfig::default()
            },
            workers: WorkerConfig {
                jain_worker: std::env::current_exe()?,
                trusted_services: BTreeMap::from([(SERVICE.into(), std::env::current_exe()?)]),
                ..WorkerConfig::default()
            },
            execution: ExecutionConfig {
                gpu_idle_baselines,
                worker_uid: (uid == 0).then_some(uid),
                worker_gid: (uid == 0).then_some(gid),
                checkpoint_timeout_ms: checkpoint_ms,
                terminate_grace_ms: 200,
                kill_grace_ms: 200,
                ..Default::default()
            },
            ..Default::default()
        };
        if let Some(fixture) = fit_fixture {
            use sha2::{Digest, Sha256};
            let hash = hex::encode(Sha256::digest(fs::read(&fixture)?));
            for name in ["jope-fit-authority", "jope-gp-search", "jope-continuation"] {
                cfg.workers
                    .trusted_services
                    .insert(name.into(), fixture.clone());
            }
            cfg.execution.trusted_fit_child_policy = Some(
                jain_smartcluster_daemon::fit_child_policy::TrustedFitChildPolicy {
                    tenant: format!("unix:{uid}"),
                    project: "resident-kernel".into(),
                    authority_executable_sha256: hash.clone(),
                    evaluator_executable_sha256: BTreeSet::from([hash]),
                },
            );
        }
        fs::write(&config, toml::to_string(&cfg)?)?;
        let mut h = Self {
            dir,
            cgroup,
            config,
            socket,
            daemon,
            child: None,
        };
        h.restart().await?;
        Ok(Some(h))
    }
    fn client(&self) -> Result<Client> {
        Ok(Client::with_config(ClientConfig {
            socket_path: self.socket.clone(),
            connect_timeout: Duration::from_millis(200),
            request_timeout: Duration::from_secs(2),
            ..Default::default()
        })?)
    }
    async fn restart(&mut self) -> Result<()> {
        ensure!(self.child.is_none(), "daemon already running");
        let out = OpenOptions::new()
            .append(true)
            .create(true)
            .open(self.dir.path().join("daemon.log"))?;
        self.child = Some(
            Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "resident_daemon_entry",
                    "--ignored",
                    "--nocapture",
                ])
                .env("RESIDENT_DAEMON_EXECUTABLE", &self.daemon)
                .env("RESIDENT_DAEMON_CONFIG", &self.config)
                .env("RESIDENT_DAEMON_CGROUP", self.cgroup.join("supervisor"))
                .stdin(Stdio::null())
                .stdout(out.try_clone()?)
                .stderr(out)
                .spawn()?,
        );
        let until = Instant::now() + WAIT;
        loop {
            ensure!(
                self.child.as_mut().unwrap().try_wait()?.is_none(),
                "test daemon exited; {}: {}",
                self.dir.path().display(),
                fs::read_to_string(self.dir.path().join("daemon.log")).unwrap_or_default()
            );
            if self.client()?.status().await.is_ok() {
                return Ok(());
            }
            ensure!(Instant::now() < until, "readiness timeout");
            sleep(Duration::from_millis(20)).await;
        }
    }
    fn crash(&mut self) -> Result<()> {
        let mut child = self.child.take().context("missing daemon")?;
        child.kill()?;
        child.wait()?;
        Ok(())
    }
    fn spec(directory: &Path, mode: &str, allowance: u64) -> Result<JobSpec> {
        let preemption = PreemptionPolicy {
            preemptible: true,
            checkpoint: CheckpointPolicy::ControlFd {
                deadline_ms: allowance,
            },
        };
        let spec = JobSpec {
            tenant: TenantId::new(format!("unix:{}", geteuid().as_raw()))?,
            project: ProjectId::new("resident-kernel")?,
            idempotency_key: IdempotencyKey::new(mode)?,
            priority: PriorityClass::System,
            workload: Workload::TrustedService {
                service: ServiceName::new(SERVICE)?,
                arguments: vec![
                    "--exact".into(),
                    "resident_fixture_entry".into(),
                    "--ignored".into(),
                    "--nocapture".into(),
                ],
                environment: Environment(BTreeMap::from([
                    (
                        "RESIDENT_KERNEL_DIRECTORY".into(),
                        directory.display().to_string(),
                    ),
                    ("RESIDENT_KERNEL_MODE".into(), mode.into()),
                ])),
            },
            resources: ResourceRequest {
                cpu: CpuRequest::fixed(100),
                ram: MemoryRequest::fixed(128 * 1024 * 1024),
                scratch_bytes: 0,
                gpu: None,
            },
            retry: RetryPolicy {
                max_failures: 3,
                initial_backoff_ms: 20,
                maximum_backoff_ms: 20,
                jitter_per_mille: 0,
                retry_on: BTreeSet::from([
                    FailureClass::Infrastructure,
                    FailureClass::DaemonRestart,
                ]),
            },
            preemption,
            io: IoMode::Pipes,
            deadline: None,
            labels: BTreeMap::new(),
        };
        spec.validate()?;
        Ok(spec)
    }
    async fn submit(&self, mode: &str, allowance: u64) -> Result<JobRecord> {
        Ok(self
            .client()?
            .submit(Self::spec(self.dir.path(), mode, allowance)?)
            .await?)
    }
    async fn file(&self, name: &str) -> Result<Vec<u8>> {
        let until = Instant::now() + WAIT;
        loop {
            if let Ok(v) = fs::read(self.dir.path().join(name)) {
                ensure!(v.len() <= 1024 * 1024, "fixture receipt exceeds 1MiB bound");
                if !name.ends_with(".json")
                    || serde_json::from_slice::<serde_json::Value>(&v).is_ok()
                {
                    return Ok(v);
                }
            }
            ensure!(
                Instant::now() < until,
                "fixture file {name} not produced; see {}",
                self.dir.path().display()
            );
            sleep(Duration::from_millis(20)).await;
        }
    }
    async fn record(&self, id: JobId) -> Result<JobRecord> {
        self.client()?.get(id).await?.context("job disappeared")
    }
    async fn running(&self, id: JobId) -> Result<()> {
        let until = Instant::now() + WAIT;
        loop {
            let record = self.record(id).await?;
            if record.state == JobState::Running {
                return Ok(());
            }
            ensure!(
                !record.state.is_terminal() && Instant::now() < until,
                "resident did not become running: {:?}",
                record.state
            );
            sleep(Duration::from_millis(20)).await;
        }
    }
    async fn no_retry(&self, id: JobId, old: &Path) -> Result<()> {
        for _ in 0..10 {
            ensure!(populated(old)?, "old recursive cgroup unexpectedly empty");
            ensure!(
                self.record(id).await?.attempt_count == 1,
                "new attempt overlapped old populated cgroup"
            );
            sleep(Duration::from_millis(25)).await;
        }
        Ok(())
    }
}

#[path = "resident_kernel/admission.rs"]
mod admission_tests;
#[path = "resident_kernel/lease.rs"]
mod lease_tests;
use lease_tests::populated;
#[path = "resident_kernel/placement.rs"]
mod placement_tests;
#[path = "resident_kernel/restarts.rs"]
mod restart_tests;
