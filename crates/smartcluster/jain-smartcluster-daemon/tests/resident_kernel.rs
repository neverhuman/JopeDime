//! Real delegated-cgroup recovery tests. Never uses production sockets or process scans.
use anyhow::{ensure, Context, Result};
use jain_smartcluster::{Client, ClientConfig};
use jain_smartcluster_core::*;
use jain_smartcluster_daemon::{
    cgroup::{CgroupManager, IsolationMode},
    config::{AuthConfig, DaemonConfig, ExecutionConfig, SchedulerConfig, WorkerConfig},
};
use nix::{
    fcntl::{fcntl, FcntlArg, FdFlag},
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

#[test]
#[ignore = "native daemon launcher used only by the isolated kernel harness"]
fn resident_daemon_entry() -> Result<()> {
    let Some(executable) = std::env::var_os("RESIDENT_DAEMON_EXECUTABLE") else {
        eprintln!("SKIP: daemon launcher is invoked only by its harness");
        return Ok(());
    };
    let cgroup =
        PathBuf::from(std::env::var_os("RESIDENT_DAEMON_CGROUP").context("daemon cgroup")?);
    ensure!(
        cgroup.starts_with("/sys/fs/cgroup")
            && cgroup.file_name().is_some_and(|n| n == "supervisor"),
        "invalid daemon subgroup"
    );
    fs::write(cgroup.join("cgroup.procs"), std::process::id().to_string())?;
    let error = Command::new(executable)
        .arg("--config")
        .arg(std::env::var_os("RESIDENT_DAEMON_CONFIG").context("daemon config")?)
        .exec();
    Err(error.into())
}

#[test]
fn resident_fixture_specs_pass_core_admission_validation() -> Result<()> {
    for (mode, allowance) in [
        ("nested", 15000),
        ("ignore-eof", 1000),
        ("dependency", 1000),
        ("result-crash", 1000),
    ] {
        let spec = Harness::spec(
            Path::new("/tmp/resident-kernel-construction"),
            mode,
            allowance,
        )?;
        spec.validate()?;
        ensure!(
            spec.preemption.preemptible,
            "checkpoint-capable fixture must be preemptible"
        );
    }
    Ok(())
}

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

/// Invoked directly by scqd through this native test executable, never a shell.
#[test]
#[ignore = "native fixture entry, launched only by the isolated kernel harness"]
fn resident_fixture_entry() -> Result<()> {
    let Some(root) = std::env::var_os("RESIDENT_KERNEL_DIRECTORY") else {
        eprintln!("SKIP: fixture is invoked only by its harness");
        return Ok(());
    };
    let root = PathBuf::from(root);
    let mode = std::env::var("RESIDENT_KERNEL_MODE")?;
    if mode == "descendant" {
        let group = PathBuf::from(std::env::var("RESIDENT_KERNEL_NESTED")?);
        fs::write(group.join("cgroup.procs"), std::process::id().to_string())?;
        durable(&root.join("descendant-ready"), b"nested\n")?;
        let start = Instant::now();
        while !root.join("release-descendant").exists() && start.elapsed() < Duration::from_secs(60)
        {
            std::thread::sleep(Duration::from_millis(20));
        }
        return Ok(());
    }
    let attempt = FencedAttempt::new(
        JobId::from_str(&std::env::var("SCQ_JOB_ID")?)?,
        AttemptNumber::new(std::env::var("SCQ_ATTEMPT_NUMBER")?.parse()?)?,
        FenceToken::new(std::env::var("SCQ_ATTEMPT_FENCE")?.parse()?)?,
    );
    let number = std::env::var("SCQ_ATTEMPT_NUMBER")?;
    let fd: i32 = std::env::var("SCQ_CONTROL_FD")?.parse()?;
    // This process owns the inherited descriptor; a nested descendant must not retain it.
    fcntl(fd, FcntlArg::F_GETFD)?;
    let mut socket = Control(fd);
    fcntl(fd, FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))?;
    let group = own_cgroup()?;
    if mode == "gpu-lease" && root.join("gpu-predecessor.json").is_file() {
        let predecessor: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("gpu-predecessor.json"))?)?;
        let old = PathBuf::from(
            predecessor["cgroup"]
                .as_str()
                .context("prior GPU allocation cgroup")?,
        );
        ensure!(old.parent() == group.parent(), "foreign predecessor cgroup");
        ensure!(
            !old.exists() || !populated(&old)?,
            "exclusive GPU successor started before predecessor cgroup drained"
        );
    }
    let mut previous = None;
    if number != "1" {
        let first: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("attempt-1.json"))?)?;
        let old = PathBuf::from(first["cgroup"].as_str().context("persisted first cgroup")?);
        ensure!(
            old.parent() == group.parent(),
            "prior attempt is outside this delegated job tree"
        );
        let current = serde_json::to_value(attempt)?;
        ensure!(
            first["attempt"]["job_id"] == current["job_id"]
                && first["attempt"]["fence"] != current["fence"],
            "prior attempt identity or fence mismatch"
        );
        match fs::read_to_string(old.join("cgroup.events")) {
            Ok(events) => ensure!(
                events.lines().any(|line| line == "populated 0"),
                "replacement started while old recursive cgroup is populated"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        previous = Some(first);
    }
    report(&mut socket, attempt, 1, WorkerReport::Ready)?;
    durable(
        &root.join(format!("attempt-{number}.json")),
        &serde_json::to_vec(
            &serde_json::json!({"attempt":attempt,"pid":std::process::id(),"cgroup":group,"previous_attempt":previous,"old_recursive_populated_at_start":false,"cuda_visible_devices":std::env::var("CUDA_VISIBLE_DEVICES").ok()}),
        )?,
    )?;
    if mode == "dependency" {
        report(
            &mut socket,
            attempt,
            2,
            WorkerReport::Failure {
                class: FailureClass::Dependency,
                message: "fixture missing immutable dependency".into(),
            },
        )?;
        // Exit immediately after write: the actor must drain reports before classifying completion.
        return Ok(());
    }
    if number == "1" && mode == "nested" {
        let nested = group.join("fixture-descendant");
        fs::create_dir(&nested)?;
        let _child = Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "resident_fixture_entry",
                "--ignored",
                "--nocapture",
            ])
            .env("RESIDENT_KERNEL_MODE", "descendant")
            .env("RESIDENT_KERNEL_NESTED", nested)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?;
        // Keep Child alive without waiting: the test intentionally covers leader exit first.
    }
    let mut sequence = 2;
    loop {
        let mut length = [0; 4];
        if let Err(error) = socket.read_exact(&mut length) {
            ensure!(
                error.kind() == std::io::ErrorKind::UnexpectedEof,
                "control read failed: {error}"
            );
            if mode == "ignore-eof" {
                std::thread::sleep(Duration::from_secs(60));
                return Ok(());
            }
            if !root.join("eof-receipt").exists() {
                durable(&root.join("eof-receipt"), b"checkpoint-fsynced-on-eof\n")?;
            }
            return Ok(());
        }
        let length = u32::from_be_bytes(length) as usize;
        ensure!(length <= 65536, "oversized fixture control");
        let mut data = vec![0; length];
        socket.read_exact(&mut data)?;
        let frame: WorkerControlFrame = postcard::from_bytes(&data)?;
        ensure!(frame.attempt == attempt, "stale control fence");
        if let ControlMessage::Application { name, payload } = frame.control {
            ensure!(
                name == "fixture.compute" && payload == b"task-1",
                "unexpected fixture input"
            );
            let result = root.join("task-1.result");
            if !result.exists() {
                durable(&result, b"task-1:deterministic-result\n")?;
            }
            ensure!(
                fs::read(&result)? == b"task-1:deterministic-result\n",
                "conflicting durable result"
            );
            if mode == "result-crash" && number == "1" {
                // Abrupt process exit after fsync, without unwinding or a completion report.
                std::process::exit(23);
            }
            report(
                &mut socket,
                attempt,
                sequence,
                WorkerReport::Progress(StructuredProgress {
                    phase: "durable-result".into(),
                    completed: 1,
                    total: Some(1),
                    message: None,
                    fields: BTreeMap::from([("receipt".into(), result.display().to_string())]),
                }),
            )?;
            sequence += 1;
            if !root.join("result-reused").exists() {
                durable(&root.join("result-reused"), b"task-1\n")?;
            }
        }
    }
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

/// Admission ownership only: native fixtures deliberately make no CUDA calls.
#[tokio::test]
#[ignore = "requires isolated delegated cgroup lane, one qualified idle GPU and baseline file"]
async fn exclusive_gpu_lease_blocks_competitor_until_recursive_drain() -> Result<()> {
    let Some(baseline_path) = std::env::var_os("SCQ_TEST_GPU_BASELINES") else {
        eprintln!("SKIP: SCQ_TEST_GPU_BASELINES not supplied");
        return Ok(());
    };
    let mut baseline_bytes = Vec::new();
    File::open(baseline_path)?
        .take(16385)
        .read_to_end(&mut baseline_bytes)?;
    ensure!(
        baseline_bytes.len() <= 16384,
        "oversized baseline configuration"
    );
    let baselines: BTreeMap<String, jain_smartcluster_daemon::gpu::QualifiedIdleBaseline> =
        serde_json::from_slice(&baseline_bytes)?;
    ensure!(
        baselines.len() == 1,
        "GPU test requires exactly one qualified device"
    );
    let uuid = baselines.keys().next().unwrap().clone();
    baselines[&uuid]
        .validate(&uuid)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        std::env::var("CUDA_VISIBLE_DEVICES")? == uuid,
        "test daemon visibility must be exactly the qualified UUID"
    );
    let inventory = jain_smartcluster_daemon::gpu::GpuInventory::discover(Some(&uuid));
    if !inventory.nvml_available {
        eprintln!("SKIP: NVML unavailable: {:?}", inventory.unavailable_reason);
        return Ok(());
    }
    ensure!(
        inventory.devices.len() == 1
            && inventory.devices[0].uuid == uuid
            && inventory.devices[0].processes.is_empty(),
        "GPU unavailable or other compute/graphics process present"
    );
    let Some(h) = Harness::start_with_gpu_baselines(1000, baselines).await? else {
        return Ok(());
    };
    durable(
        &h.dir.path().join("gpu-inventory.json"),
        &serde_json::to_vec(&inventory)?,
    )?;
    let first_dir = h.dir.path().join("gpu-first");
    let second_dir = h.dir.path().join("gpu-second");
    fs::create_dir(&first_dir)?;
    fs::create_dir(&second_dir)?;
    let make_spec = |directory: &Path, key: &str| -> Result<JobSpec> {
        let mut spec = Harness::spec(directory, "gpu-lease", 1000)?;
        spec.idempotency_key = IdempotencyKey::new(key)?;
        spec.retry.max_failures = 0;
        spec.resources.cpu = CpuRequest::fixed(1000);
        spec.resources.gpu = Some(GpuRequest {
            count: 1,
            vram_bytes: 1024 * 1024,
            compute_share_per_mille: 1000,
            mode: GpuMode::Exclusive,
        });
        spec.validate()?;
        Ok(spec)
    };
    let first = h
        .client()?
        .submit(make_spec(&first_dir, "exclusive-first")?)
        .await?;
    h.running(first.id).await?;
    let first_receipt = wait_gpu_attempt(&first_dir).await?;
    // Prove the competing job's CPU/RAM fit while the first lease runs. Without
    // this control, a queued GPU competitor could merely indicate CPU pressure.
    let control_dir = h.dir.path().join("cpu-capacity-control");
    fs::create_dir(&control_dir)?;
    let mut control_spec = make_spec(&control_dir, "cpu-capacity-control")?;
    control_spec.resources.gpu = None;
    let control = h.client()?.submit(control_spec).await?;
    h.running(control.id).await?;
    h.client()?
        .cancel(CancelRequest {
            job_id: control.id,
            mode: CancelMode::Graceful { grace_ms: 200 },
        })
        .await?;
    let old = PathBuf::from(first_receipt["cgroup"].as_str().context("first cgroup")?);
    ensure!(old.starts_with(&h.cgroup), "foreign first cgroup");
    durable(
        &second_dir.join("gpu-predecessor.json"),
        &serde_json::to_vec(&first_receipt)?,
    )?;
    let second = h
        .client()?
        .submit(make_spec(&second_dir, "exclusive-second")?)
        .await?;
    for _ in 0..40 {
        ensure!(
            h.record(first.id).await?.state == JobState::Running && populated(&old)?,
            "first allocation unexpectedly ended"
        );
        let held = h.record(second.id).await?;
        ensure!(
            held.state == JobState::Queued
                && held.attempt_count == 0
                && !second_dir.join("attempt-1.json").exists(),
            "exclusive GPU competitor launched concurrently"
        );
        sleep(Duration::from_millis(25)).await;
    }
    let cancelled = h
        .client()?
        .cancel(CancelRequest {
            job_id: first.id,
            mode: CancelMode::Graceful { grace_ms: 200 },
        })
        .await?;
    ensure!(
        cancelled.state == JobState::Cancelled,
        "first job cancellation failed"
    );
    h.running(second.id).await?;
    let second_receipt = wait_gpu_attempt(&second_dir).await?;
    ensure!(
        !old.exists() || !populated(&old)?,
        "exclusive allocations overlapped at handoff"
    );
    ensure!(
        first_receipt["cuda_visible_devices"].as_str() == Some(uuid.as_str())
            && second_receipt["cuda_visible_devices"].as_str() == Some(uuid.as_str()),
        "allocations do not identify the exact qualified GPU"
    );
    durable(
        &h.dir.path().join("exclusive-gpu-proof.json"),
        &serde_json::to_vec(
            &serde_json::json!({"schema":"jope.exclusive_gpu_allocation_test.v1","uuid":uuid,"first":first_receipt,"second":second_receipt,"blocked_samples":40,"gpu_utilization_tested":false}),
        )?,
    )?;
    Ok(())
}

async fn wait_gpu_attempt(directory: &Path) -> Result<serde_json::Value> {
    let until = Instant::now() + WAIT;
    loop {
        if let Ok(file) = File::open(directory.join("attempt-1.json")) {
            let mut bytes = Vec::new();
            file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
            ensure!(bytes.len() <= 1024 * 1024, "oversized GPU fixture receipt");
            if let Ok(value) = serde_json::from_slice(&bytes) {
                return Ok(value);
            }
        }
        ensure!(Instant::now() < until, "GPU fixture receipt timed out");
        sleep(Duration::from_millis(20)).await;
    }
}
fn populated(path: &Path) -> Result<bool> {
    Ok(fs::read_to_string(path.join("cgroup.events"))?
        .lines()
        .any(|l| l == "populated 1"))
}
fn remove_empty_tree(path: &Path) {
    if let Ok(entries) = fs::read_dir(path) {
        for e in entries.flatten() {
            if e.file_type().is_ok_and(|t| t.is_dir()) {
                remove_empty_tree(&e.path());
            }
        }
    }
    let _ = fs::remove_dir(path);
}
impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        } // Only this newly-created subtree, never process-name/marker scans.
        let _ = fs::write(self.cgroup.join("cgroup.kill"), "1");
        for _ in 0..100 {
            if !populated(&self.cgroup).unwrap_or(false) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        remove_empty_tree(&self.cgroup);
    }
}

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn resident_two_restarts_hold_nested_descendant_before_new_fence() -> Result<()> {
    let Some(mut h) = Harness::start(15000).await? else {
        return Ok(());
    };
    let job = h.submit("nested", 15000).await?;
    h.file("descendant-ready").await?;
    let first: serde_json::Value = serde_json::from_slice(&h.file("attempt-1.json").await?)?;
    let old = PathBuf::from(first["cgroup"].as_str().context("cgroup path")?);
    ensure!(old.starts_with(&h.cgroup), "foreign cgroup");
    h.crash()?;
    h.file("eof-receipt").await?;
    // Main leader may take a scheduling tick to exit. Observe the actual kernel procs file.
    let until = Instant::now() + WAIT;
    while !fs::read_to_string(old.join("cgroup.procs"))?
        .trim()
        .is_empty()
    {
        ensure!(Instant::now() < until, "leader did not exit");
        sleep(Duration::from_millis(20)).await;
    }
    ensure!(
        populated(&old)?,
        "nested descendant must keep old cgroup populated"
    );
    h.restart().await?;
    h.no_retry(job.id, &old).await?;
    h.crash()?;
    h.restart().await?;
    h.no_retry(job.id, &old).await?;
    durable(&h.dir.path().join("release-descendant"), b"release\n")?;
    let second: serde_json::Value = serde_json::from_slice(&h.file("attempt-2.json").await?)?;
    ensure!(
        !old.exists() || !populated(&old)?,
        "retry launched before recursive drain"
    );
    ensure!(
        first["attempt"]["fence"] != second["attempt"]["fence"],
        "attempt fence reused"
    );
    ensure!(
        h.record(job.id).await?.attempt_count == 2,
        "unexpected attempt count"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn resident_ignored_eof_is_bounded_by_checkpoint_allowance() -> Result<()> {
    let Some(mut h) = Harness::start(1000).await? else {
        return Ok(());
    };
    let job = h.submit("ignore-eof", 1000).await?;
    let first: serde_json::Value = serde_json::from_slice(&h.file("attempt-1.json").await?)?;
    let old = PathBuf::from(first["cgroup"].as_str().unwrap());
    h.crash()?;
    let started = Instant::now();
    h.restart().await?;
    h.file("attempt-2.json").await?;
    ensure!(
        started.elapsed() >= Duration::from_millis(800)
            && started.elapsed() < Duration::from_millis(3500),
        "checkpoint recovery {:?} violates 1000ms allowance (200ms tolerance) plus bounded escalation/readiness overhead",
        started.elapsed()
    );
    ensure!(
        !old.exists() || !populated(&old)?,
        "old worker overlaps replacement"
    );
    ensure!(
        h.record(job.id).await?.attempt_count == 2,
        "wrong retry count"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn resident_immediate_dependency_failure_report_is_not_lost() -> Result<()> {
    let Some(h) = Harness::start(1000).await? else {
        return Ok(());
    };
    let job = h.submit("dependency", 1000).await?;
    let until = Instant::now() + WAIT;
    loop {
        let record = h.record(job.id).await?;
        if record.state.is_terminal() {
            ensure!(
                matches!(
                    record.terminal,
                    Some(TerminalStatus::Failed {
                        class: FailureClass::Dependency,
                        ..
                    })
                ),
                "dependency report lost: {:?}",
                record.terminal
            );
            ensure!(record.attempt_count == 1, "dependency failure retried");
            break;
        }
        ensure!(Instant::now() < until, "failure reporting timeout");
        sleep(Duration::from_millis(20)).await;
    }
    Ok(())
}

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn resident_fsynced_result_before_crash_is_reused_exactly() -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let Some(h) = Harness::start(1000).await? else {
        return Ok(());
    };
    let job = h.submit("result-crash", 1000).await?;
    h.file("attempt-1.json").await?;
    let request = SendControlRequest {
        job_id: job.id,
        control: ControlMessage::Application {
            name: "fixture.compute".into(),
            payload: b"task-1".to_vec(),
        },
    };
    h.running(job.id).await?;
    h.client()?.send_control(request.clone()).await?;
    let bytes = h.file("task-1.result").await?;
    let before = fs::metadata(h.dir.path().join("task-1.result"))?;
    h.file("attempt-2.json").await?;
    h.running(job.id).await?;
    h.client()?.send_control(request).await?;
    h.file("result-reused").await?;
    let after = fs::metadata(h.dir.path().join("task-1.result"))?;
    ensure!(
        bytes == b"task-1:deterministic-result\n"
            && fs::read(h.dir.path().join("task-1.result"))? == bytes,
        "result bytes changed"
    );
    ensure!(
        (before.dev(), before.ino(), before.mtime_nsec())
            == (after.dev(), after.ino(), after.mtime_nsec()),
        "durable result was rewritten"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn graceful_sigterm_waits_for_checkpoint_and_recursive_drain() -> Result<()> {
    let Some(mut h) = Harness::start(15000).await? else {
        return Ok(());
    };
    let job = h.submit("nested", 15000).await?;
    h.file("descendant-ready").await?;
    let first: serde_json::Value = serde_json::from_slice(&h.file("attempt-1.json").await?)?;
    let old = PathBuf::from(first["cgroup"].as_str().context("old cgroup")?);
    ensure!(old.starts_with(&h.cgroup), "foreign worker cgroup");
    let daemon = h.child.as_ref().context("daemon missing")?.id();
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(daemon as i32),
        nix::sys::signal::Signal::SIGTERM,
    )?;
    h.file("eof-receipt").await?;
    // Main remains alive after checkpoint fsync while a nested child remains.
    for _ in 0..20 {
        ensure!(populated(&old)?, "nested child ended too early");
        ensure!(
            h.child.as_mut().unwrap().try_wait()?.is_none(),
            "main exited before recursive drain"
        );
        ensure!(
            !h.dir.path().join("attempt-2.json").exists(),
            "replacement launched during shutdown"
        );
        sleep(Duration::from_millis(25)).await;
    }
    // Repeat signal must not terminate main or extend the persisted recovery deadline.
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(daemon as i32),
        nix::sys::signal::Signal::SIGTERM,
    )?;
    durable(&h.dir.path().join("release-descendant"), b"release\n")?;
    let until = Instant::now() + WAIT;
    loop {
        if let Some(status) = h.child.as_mut().unwrap().try_wait()? {
            ensure!(status.success(), "graceful daemon failed: {status}");
            break;
        }
        ensure!(
            Instant::now() < until,
            "shutdown did not finish after drain"
        );
        sleep(Duration::from_millis(20)).await;
    }
    h.child.take();
    ensure!(
        !old.exists() || !populated(&old)?,
        "main exited with old subtree populated"
    );
    ensure!(
        !h.dir.path().join("attempt-2.json").exists(),
        "shutdown launched retry"
    );
    h.restart().await?;
    h.file("attempt-2.json").await?;
    ensure!(
        h.record(job.id).await?.attempt_count == 2,
        "durable retry not resumed exactly once"
    );
    Ok(())
}

/// Actual peer-authenticated placement, child-start gate and cancellation propagation.
#[tokio::test]
#[ignore = "requires real delegated cgroup and SCQ_TEST_FIT_FIXTURE native example binary"]
async fn fit_child_real_peer_placement_duplicate_and_target_cancellation() -> Result<()> {
    use jain_smartcluster::jope::{TaskKind, TaskManifest, MANIFEST_SCHEMA};
    use sha2::{Digest, Sha256};
    let fixture = fs::canonicalize(
        std::env::var_os("SCQ_TEST_FIT_FIXTURE").context("build and set native fit fixture")?,
    )?;
    let Some(h) = Harness::start_inner(1000, BTreeMap::new(), Some(fixture)).await? else {
        return Ok(());
    };
    let now_ms: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?;
    let now = Timestamp::from_unix_millis(now_ms);
    let input = serde_json::to_vec(
        &serde_json::json!({"schema":"jope.smartcluster_input.v1","operation":{"kind":"Search","evaluations":1,"seeds":null}}),
    )?;
    let input_sha = hex::encode(Sha256::digest(&input));
    durable(&h.dir.path().join("fit.input.json"), &input)?;
    let manifest = TaskManifest {
        schema: MANIFEST_SCHEMA.into(),
        run: "native-fit-kernel".into(),
        generation: 0,
        task_id: "target-0".into(),
        kind: TaskKind::GpSearch,
        input_manifest_sha256: input_sha.clone(),
        dependencies: BTreeMap::from([("input".into(), input_sha)]),
        parent_checkpoint_sha256: None,
        corpus_sha256: None,
        idempotency_key: IdempotencyKey::new("fit-kernel-target")?,
        deadline: Timestamp::from_unix_millis(now_ms + 120000),
        retry_allowance: 0,
        resources: ResourceRequest {
            cpu: CpuRequest::fixed(100),
            ram: MemoryRequest::fixed(128 * 1024 * 1024),
            scratch_bytes: 0,
            gpu: None,
        },
        expected_receipt_id: "fit-kernel-result".into(),
        checkpoint_allowance_ms: None,
    };
    let bytes = serde_json::to_vec(&manifest)?;
    let manifest_sha = hex::encode(Sha256::digest(&bytes));
    let path = h.dir.path().join("fit.json");
    durable(&path, &bytes)?;
    let target = h
        .client()?
        .submit(jain_smartcluster::jope::job_spec(
            &path,
            &manifest_sha,
            TenantId::new(format!("unix:{}", geteuid().as_raw()))?,
            ProjectId::new("resident-kernel")?,
            now,
        )?)
        .await?;
    h.running(target.id).await?;
    h.file("target-started.json").await?;
    let target_record = h.record(target.id).await?;
    durable(
        &h.dir.path().join("target-request.json"),
        &serde_json::to_vec(
            &serde_json::json!({"attempt":target_record.active_attempt.context("target fence")?,"manifest_sha256":manifest_sha}),
        )?,
    )?;
    let mut authority = Harness::spec(h.dir.path(), "fit-authority", 1000)?;
    authority.workload = Workload::TrustedService {
        service: ServiceName::new("jope-fit-authority")?,
        arguments: vec![
            "--authority".into(),
            h.dir.path().display().to_string(),
            h.socket.display().to_string(),
        ],
        environment: Environment::default(),
    };
    authority.preemption = PreemptionPolicy::non_preemptible();
    authority.deadline = Some(Timestamp::from_unix_millis(now_ms + 120000));
    authority.retry.max_failures = 0;
    let authority = h.client()?.submit(authority).await?;
    h.running(authority.id).await?;
    let placement: TrustedFitChildPlacement =
        serde_json::from_slice(&h.file("placed.json").await?)?;
    let request: AttachTrustedFitChildRequest =
        serde_json::from_slice(&h.file("request.json").await?)?;
    ensure!(
        placement.target_attempt == target_record.active_attempt.unwrap(),
        "wrong target fence"
    );
    ensure!(
        placement.cgroup_path.starts_with(&h.cgroup),
        "foreign target cgroup"
    );
    ensure!(
        h.client()?.attach_trusted_fit_child(request).await.is_err(),
        "same UID nonauthority peer accepted"
    );
    ensure!(
        !h.dir.path().join("child-started.json").exists(),
        "child passed gate before test permits start"
    );
    durable(
        &h.dir.path().join("allow-child-start"),
        b"start after receipt\n",
    )?;
    let started: serde_json::Value = serde_json::from_slice(&h.file("child-started.json").await?)?;
    ensure!(
        started["cgroup"] == placement.cgroup_path.display().to_string(),
        "child started outside target allocation"
    );
    h.client()?
        .cancel(CancelRequest {
            job_id: target.id,
            mode: CancelMode::Graceful { grace_ms: 200 },
        })
        .await?;
    let limit = Instant::now() + WAIT;
    loop {
        if h.record(target.id).await?.state.is_terminal() {
            // Observe the kernel AFTER the terminal receipt. A pre-RPC snapshot
            // may correctly be populated immediately before completion commits.
            ensure!(
                !placement.cgroup_path.exists() || !populated(&placement.cgroup_path)?,
                "terminal target released while child subtree populated"
            );
            break;
        }
        ensure!(
            Instant::now() < limit,
            "target subtree not drained before terminal release"
        );
        sleep(Duration::from_millis(20)).await;
    }
    h.file("child-reaped.json").await?;
    h.file("authority-survived.json").await?;
    ensure!(
        h.record(authority.id).await?.state == JobState::Running,
        "target cancellation killed authority"
    );
    durable(&h.dir.path().join("release-authority"), b"complete\n")?;
    Ok(())
}
