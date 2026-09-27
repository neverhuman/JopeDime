//! W7-T05/T07 same-host multi-process fixture through the real supervisor.
//!
//! Every rank has the topology of the three-host proof, collapsed onto loopback: its own real
//! SmartCluster daemon (`scqd`, unprivileged embedded mode, one allocation at a time) runs the
//! rank as a trusted-service job whose program is the real `jope-native-loop participant`; its
//! own real `jope-native-loop agent` process serves that daemon's `SmartClusterSupervisor`
//! (authenticated exchange: pinned uid and executable digest) over the pinned-TLS transport;
//! the coordinator's `RemoteSupervisor` routes each rank to its agent, and the coordinator runs
//! the real rendezvous over the same transport. The host census is collected from the host's
//! installation file. SSH is never involved.
//!
//! This is a FIXTURE: three ranks under one boot on one host are not the three-host proof
//! (W7-T07), and the CPU proof workload is not numerical training.
#[path = "support/bounded_fixture.rs"]
mod fixture;
#[path = "support/tls_fixture.rs"]
mod tls;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    agent::RemoteSupervisor,
    collect::{self, CommandOutput, CommandRunner},
    contract::{BoundedRunContract, Ceilings, WorkItem, WorkKind},
    controller::{self, Disposition, Outcome, Supervisor},
    driver::{self, CoordinateArgs},
    participant::{Installation, SmartClusterInstallation, INSTALLATION_SCHEMA},
    rendezvous,
    supervisor::{DaemonPin, HostSupervision, SmartClusterSupervisor, SERVICE},
    transport::HostCredential,
};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader},
    net::{SocketAddr, TcpListener},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, Stdio},
    sync::{atomic::AtomicBool, Arc, OnceLock},
    thread,
    time::{Duration, Instant},
};
use tls::*;

const MIB: u64 = 1024 * 1024;
const PROJECT: &str = "bounded-cpu-proof";
const RANKS: u32 = 3;
/// SmartCluster requires a job's tenant to be exactly the submitting peer's principal.
fn tenant() -> String {
    format!("unix:{}", unsafe { libc::geteuid() })
}
fn participant_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_jope-native-loop"))
}

/// No GPU probe is ever run by these fixtures.
struct NoGpu;
impl CommandRunner for NoGpu {
    fn run(&self, program: &str, _: &[&str]) -> anyhow::Result<Option<CommandOutput>> {
        panic!("the CPU-only fixture runs no {program}")
    }
}

/// The `scqd` binary of this target directory, built once if it is absent.
fn scqd_binary() -> PathBuf {
    static BUILT: OnceLock<PathBuf> = OnceLock::new();
    BUILT
        .get_or_init(|| {
            let exe = std::env::current_exe().unwrap();
            let profile_dir = exe.parent().unwrap().parent().unwrap().to_path_buf();
            let binary = profile_dir.join("scqd");
            if !binary.exists() {
                let profile = profile_dir
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                let mut cmd = Command::new(env!("CARGO"));
                cmd.args([
                    "build",
                    "--locked",
                    "--offline",
                    "-p",
                    "jain-smartcluster-daemon",
                    "--bin",
                    "scqd",
                ])
                .env("CARGO_TARGET_DIR", profile_dir.parent().unwrap());
                match profile.as_str() {
                    "debug" => {}
                    "release" => {
                        cmd.arg("--release");
                    }
                    other => {
                        cmd.args(["--profile", other]);
                    }
                }
                let status = cmd.status().expect("run cargo to build scqd");
                assert!(
                    status.success(),
                    "building scqd for the SmartCluster fixture failed"
                );
            }
            binary
        })
        .clone()
}

struct Scqd {
    child: Child,
    dir: PathBuf,
    socket: PathBuf,
    executable_sha256: String,
}
impl Scqd {
    fn start(dir: &Path) -> Self {
        fs::create_dir_all(dir.join("run")).unwrap();
        let binary = scqd_binary();
        let uid = unsafe { libc::geteuid() };
        let socket = dir.join("run/scqd.sock");
        let config = format!(
            r#"socket_path = "{socket}"
state_path = "{dir}/state/scqd.redb"
event_dir = "{dir}/events"
cgroup_root = "{dir}/not-a-cgroup"
socket_mode = 384
embedded = true

[scheduler]
interactive_cpu_headroom_millicores = 0
interactive_ram_headroom_bytes = 0
recovery_quiet_ms = 50

[auth]
admin_uids = [{uid}]
system_uids = []
system_principal = "sagemaker"
web_uid = 4294967295
web_principal = "jain-web"

[workers]
jain_worker = "{scqd}"

[workers.trusted_services]
{SERVICE} = "{participant}"

[execution]
checkpoint_timeout_ms = 200
terminate_grace_ms = 1000
kill_grace_ms = 500
"#,
            socket = socket.display(),
            dir = dir.display(),
            scqd = binary.display(),
            participant = participant_binary().display(),
        );
        fs::write(dir.join("scqd.toml"), config).unwrap();
        let log = fs::File::create(dir.join("scqd.log")).unwrap();
        let child = Command::new(&binary)
            .arg("--config")
            .arg(dir.join("scqd.toml"))
            .env("RUST_LOG", "warn")
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap();
        let start = Instant::now();
        while !socket.exists() {
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "scqd did not start: {}",
                fs::read_to_string(dir.join("scqd.log")).unwrap_or_default()
            );
            thread::sleep(Duration::from_millis(20));
        }
        Self {
            child,
            dir: dir.into(),
            socket,
            executable_sha256: sha(&fs::read(&binary).unwrap()),
        }
    }
}
impl Drop for Scqd {
    fn drop(&mut self) {
        unsafe { libc::kill(self.child.id() as i32, libc::SIGKILL) };
        let _ = self.child.wait();
    }
}

fn write_private(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn pkcs8(seed: [u8; 32]) -> Vec<u8> {
    [
        &[
            0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22,
            0x04, 0x20,
        ][..],
        &seed[..],
    ]
    .concat()
}
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// One rank's supervision: its own daemon and its own agent process.
struct RankHost {
    agent: Child,
    /// Kept open so the agent's stdout never breaks.
    _agent_stdout: BufReader<ChildStdout>,
    address: SocketAddr,
    supervision: HostSupervision,
    scqd: Scqd,
}
impl Drop for RankHost {
    fn drop(&mut self) {
        unsafe { libc::kill(self.agent.id() as i32, libc::SIGKILL) };
        let _ = self.agent.wait();
    }
}

/// Three CPU ranks of one host, each supervised through its own daemon and agent.
struct Host {
    _t: tempfile::TempDir,
    root: PathBuf,
    run_dir: PathBuf,
    sha: String,
    contract: BoundedRunContract,
    census: PathBuf,
    listen: SocketAddr,
    agents_file: PathBuf,
    ranks: Vec<RankHost>,
}
impl Host {
    fn new(planned_updates: u64) -> Self {
        let (t, root) = tempdir();
        let participant = participant_binary();
        write_closure(&root);
        // This host's installation, certificate and keys.
        let cert = current("fixture-coordinator-host", 31);
        let etc = root.join("etc");
        fs::create_dir_all(&etc).unwrap();
        fs::write(etc.join("host.der"), &cert.der).unwrap();
        write_private(&etc.join("host.key"), &cert.pkcs8);
        for (name, bytes) in [
            ("libjope-runtime.so", b"fixture runtime".as_slice()),
            ("compiler", b"fixture compiler"),
            ("worker.json", b"{\"fixture\":\"policy\"}"),
        ] {
            fs::write(etc.join(name), bytes).unwrap();
        }
        let k = keys(RANKS as usize);
        write_private(&etc.join("coordinator.key"), &pkcs8([7; 32]));
        // One daemon per rank; each rank's installation pins its own daemon (W7-D1).
        let daemons: Vec<Scqd> = (0..RANKS)
            .map(|rank| Scqd::start(&root.join(format!("rank-{rank}/scqd"))))
            .collect();
        let pin = |scqd: &Scqd| SmartClusterInstallation {
            daemon: DaemonPin {
                socket: scqd.socket.clone(),
                uid: unsafe { libc::geteuid() },
                executable_sha256: scqd.executable_sha256.clone(),
            },
            service: SERVICE.into(),
        };
        let installation = Installation {
            schema: INSTALLATION_SCHEMA.into(),
            address: "127.0.0.1".into(),
            certificate: etc.join("host.der"),
            certificate_key: etc.join("host.key"),
            source_commit: sha(b"fixture-source")[..40].to_string(),
            executable: participant.clone(),
            runtime_libraries: vec![etc.join("libjope-runtime.so")],
            compiler: etc.join("compiler"),
            cuda_libraries: Vec::new(),
            nccl_libraries: Vec::new(),
            installed_policies: vec![etc.join("worker.json")],
            gpu_inventory: false,
            input_root: root.clone(),
            smartcluster: pin(&daemons[0]),
        };
        // The census this host's collector writes.
        let now = now_ms();
        let (census, _) = collect::collect(&installation.collect_spec(None), &NoGpu, now).unwrap();
        let census_path = root.join("census.json");
        collect::write_census(&census_path, &census).unwrap();
        let identity = census.hosts[0].identity.clone();
        // Three CPU ranks on this one host.
        let mut c = contract(now, RANKS as usize, &k);
        let unique = root.file_name().unwrap().to_string_lossy().to_string();
        c.run = format!("w7-scq-{}", unique.trim_start_matches('.'));
        c.coordinator_host = identity.host.clone();
        c.counters.planned_updates = planned_updates;
        c.work_census.clear();
        for i in 0..RANKS as usize {
            let p = &mut c.participants[i];
            p.host = identity.clone();
            p.local_rank = i as u32;
            p.device = None;
            p.ceilings = Ceilings {
                cpu_millicores: 250,
                ram_bytes: 128 * MIB,
                gpu_vram_bytes: None,
                wall_ms: 600_000,
            };
            for (kind, cpu, wall, updates) in [
                (WorkKind::Numerical, 60_000, 300_000, planned_updates),
                (WorkKind::Verification, 5_000, 30_000, 0),
                (WorkKind::ArtifactMovement, 5_000, 20_000, 0),
                (WorkKind::Cleanup, 1_000, 10_000, 0),
                (WorkKind::FailureAllowance, 10_000, 60_000, 0),
            ] {
                c.work_census.push(WorkItem {
                    id: format!(
                        "rank-{i}-{}",
                        serde_json::to_value(kind).unwrap().as_str().unwrap()
                    ),
                    global_rank: i as u32,
                    kind,
                    cpu_core_ms: cpu,
                    gpu_ms: 0,
                    wall_ms: wall,
                    peak_ram_bytes: 64 * MIB,
                    peak_vram_bytes: 0,
                    updates,
                });
            }
        }
        for i in 0..RANKS as usize {
            let p = c.participants[i].clone();
            c.participants[i].grant = grant(&c, &p, &k);
        }
        c.validate_structure().unwrap();
        let run_dir = root.join("run");
        let sha = controller::plan(&run_dir, &bytes(&c), now, None)
            .unwrap()
            .plan_sha256;
        let listen: SocketAddr = format!("127.0.0.1:{}", free_port()).parse().unwrap();
        // Per rank: its daemon, its supervision and its agent process.
        let mut ranks = Vec::new();
        for (rank, scqd) in (0..RANKS).zip(daemons) {
            let dir = root.join(format!("rank-{rank}"));
            let key = dir.join("participant.key");
            write_private(&key, &pkcs8(rank_seed(rank as usize)));
            let installation_path = dir.join("installation.json");
            let own = Installation {
                smartcluster: pin(&scqd),
                ..installation.clone()
            };
            fs::write(&installation_path, serde_json::to_vec(&own).unwrap()).unwrap();
            let supervision = HostSupervision {
                host: identity.host.clone(),
                daemon: own.smartcluster.daemon.clone(),
                tenant: tenant(),
                project: PROJECT.into(),
                service: SERVICE.into(),
                plan: run_dir.join("plan.json"),
                state: dir.join("supervisor"),
                coordinator: listen.to_string(),
                installation: installation_path.clone(),
                participant_keys: BTreeMap::from([(rank, key)]),
            };
            let supervision_path = dir.join("supervision.json");
            fs::write(&supervision_path, serde_json::to_vec(&supervision).unwrap()).unwrap();
            let log = fs::File::create(dir.join("agent.log")).unwrap();
            let mut agent = Command::new(&participant)
                .arg("agent")
                .args(["--plan", run_dir.join("plan.json").to_str().unwrap()])
                .args(["--plan-sha256", &sha])
                .args(["--host", &identity.host])
                .args(["--rank", &rank.to_string()])
                .args(["--listen", "127.0.0.1:0"])
                .args(["--certificate", etc.join("host.der").to_str().unwrap()])
                .args(["--certificate-key", etc.join("host.key").to_str().unwrap()])
                .args(["--supervision", supervision_path.to_str().unwrap()])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(log)
                .spawn()
                .unwrap();
            let mut stdout = BufReader::new(agent.stdout.take().unwrap());
            let mut ready = String::new();
            stdout.read_line(&mut ready).unwrap();
            let ready: serde_json::Value = serde_json::from_str(&ready).unwrap_or_else(|e| {
                panic!(
                    "agent {rank} did not start ({e}): {}",
                    fs::read_to_string(dir.join("agent.log")).unwrap_or_default()
                )
            });
            assert_eq!(ready["agent"], "listening");
            let address: SocketAddr = ready["address"].as_str().unwrap().parse().unwrap();
            ranks.push(RankHost {
                agent,
                _agent_stdout: stdout,
                address,
                supervision,
                scqd,
            });
        }
        let agents: BTreeMap<String, String> = ranks
            .iter()
            .enumerate()
            .map(|(r, h)| (r.to_string(), h.address.to_string()))
            .collect();
        let agents_file = root.join("agents.json");
        fs::write(&agents_file, serde_json::to_vec(&agents).unwrap()).unwrap();
        Self {
            _t: t,
            root,
            run_dir,
            sha,
            contract: c,
            census: census_path,
            listen,
            agents_file,
            ranks,
        }
    }
    fn args(&self) -> CoordinateArgs {
        let etc = self.root.join("etc");
        CoordinateArgs {
            run_dir: self.run_dir.clone(),
            plan_sha256: self.sha.clone(),
            census: self.census.clone(),
            listen: self.listen,
            certificate: etc.join("host.der"),
            certificate_key: etc.join("host.key"),
            coordinator_key: etc.join("coordinator.key"),
            heartbeat: Duration::from_millis(100),
        }
    }
    /// The coordinator's supervisor: every rank through its own agent.
    fn supervisor(&self) -> RemoteSupervisor {
        let etc = self.root.join("etc");
        RemoteSupervisor::new(
            &self.contract,
            &self.sha,
            HostCredential::load(&etc.join("host.der"), &etc.join("host.key")).unwrap(),
            self.ranks
                .iter()
                .enumerate()
                .map(|(r, h)| (r as u32, h.address))
                .collect(),
        )
        .unwrap()
    }
    fn dispatch_dir(&self, rank: u32) -> PathBuf {
        self.ranks[rank as usize]
            .supervision
            .state
            .join(rendezvous::dispatch_id(&self.sha, 1, rank).unwrap())
    }
    fn wait_admitted(&self) {
        let start = Instant::now();
        while !(0..RANKS).all(|r| self.dispatch_dir(r).join("outbox/admitted").exists()) {
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "ranks were not admitted:\n{}",
                self.diagnose()
            );
            thread::sleep(Duration::from_millis(50));
        }
    }
    /// Every rank's outbox (including a participant's recorded refusal), the journal events,
    /// each daemon's and agent's log tail and each daemon's jobs, for assertion messages.
    fn diagnose(&self) -> String {
        let mut out = String::new();
        if let Ok(entries) = fs::read_dir(self.run_dir.join("journal")) {
            let mut names: Vec<_> = entries.flatten().map(|e| e.path()).collect();
            names.sort();
            for p in names {
                let v: serde_json::Value =
                    serde_json::from_slice(&fs::read(&p).unwrap_or_default()).unwrap_or_default();
                out += &format!("journal {}\n", v["event"]);
            }
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        for rank in 0..RANKS {
            let outbox = self.dispatch_dir(rank).join("outbox");
            let files: Vec<String> = fs::read_dir(&outbox)
                .map(|d| {
                    d.flatten()
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .collect()
                })
                .unwrap_or_default();
            out += &format!("rank {rank} outbox {files:?}\n");
            if let Ok(refusal) = fs::read_to_string(outbox.join("refusal.txt")) {
                out += &format!("rank {rank} refusal: {refusal}\n");
            }
            let h = &self.ranks[rank as usize];
            for (name, path) in [
                ("scqd", h.scqd.dir.join("scqd.log")),
                ("agent", self.root.join(format!("rank-{rank}/agent.log"))),
            ] {
                let log = fs::read_to_string(path).unwrap_or_default();
                let tail: Vec<&str> = log.lines().rev().take(10).collect();
                out += &format!("rank {rank} {name} log tail: {tail:?}\n");
            }
            let listed = runtime.block_on(async {
                jain_smartcluster::Client::new(h.supervision.daemon.socket.clone())
                    .list(jain_smartcluster::ListRequest::default())
                    .await
            });
            match listed {
                Ok(page) => {
                    for job in page.jobs {
                        out += &format!(
                            "rank {rank} job {} state {:?} attempts {} terminal {:?}\n",
                            job.id, job.state, job.attempt_count, job.terminal
                        );
                    }
                }
                Err(e) => out += &format!("rank {rank} scqd list failed: {e}\n"),
            }
        }
        out
    }
    /// Live processes whose command line names one of this run's dispatches.
    fn rank_processes(&self) -> Vec<i32> {
        let ids: Vec<String> = (0..RANKS)
            .map(|r| rendezvous::dispatch_id(&self.sha, 1, r).unwrap())
            .collect();
        let mut pids = Vec::new();
        for entry in fs::read_dir("/proc").unwrap().flatten() {
            let Ok(pid) = entry.file_name().to_string_lossy().parse::<i32>() else {
                continue;
            };
            if let Ok(cmdline) = fs::read(entry.path().join("cmdline")) {
                let cmdline = String::from_utf8_lossy(&cmdline);
                let state = fs::read_to_string(entry.path().join("stat")).unwrap_or_default();
                let zombie = state
                    .rsplit_once(')')
                    .is_some_and(|(_, t)| t.trim_start().starts_with('Z'));
                if !zombie && ids.iter().any(|id| cmdline.contains(id.as_str())) {
                    pids.push(pid);
                }
            }
        }
        pids
    }
    fn assert_no_orphans(&self) {
        wait(Duration::from_secs(20), || self.rank_processes().is_empty());
    }
    fn journal_contains(&self, needle: &str) -> bool {
        fs::read_dir(self.run_dir.join("journal"))
            .unwrap()
            .flatten()
            .any(|e| {
                fs::read_to_string(e.path())
                    .unwrap_or_default()
                    .contains(needle)
            })
    }
}
fn wait(limit: Duration, mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < limit, "fixture wait timed out");
        thread::sleep(Duration::from_millis(50));
    }
}
fn coordinate_in_background(
    host: &Host,
) -> (
    Arc<AtomicBool>,
    thread::JoinHandle<anyhow::Result<controller::StatusReport>>,
) {
    let stop = Arc::new(AtomicBool::new(false));
    let (args, supervisor, flag) = (host.args(), host.supervisor(), Arc::clone(&stop));
    (
        stop,
        thread::spawn(move || driver::coordinate(&args, supervisor, &flag)),
    )
}

#[test]
fn smartcluster_ranks_commit_the_whole_group_exactly_once() {
    let host = Host::new(5);
    let (_stop, t) = coordinate_in_background(&host);
    let status = t.join().unwrap().unwrap();
    assert_eq!(
        status.outcome,
        Some(Outcome::Committed),
        "{status:?}\n{}",
        host.diagnose()
    );
    assert_eq!(status.committed_updates_total, 60_005);
    assert!(status
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Committed)));
    let verified = controller::verify(&host.run_dir, Some(&host.sha)).unwrap();
    assert_eq!(verified.results_verified, 3);
    host.assert_no_orphans();
    // Every rank ran as a SmartCluster job of the real participant binary, one per daemon,
    // and its result records the job and attempt it proved it was (W7-D1).
    for rank in 0..RANKS {
        let outbox = host.dispatch_dir(rank).join("outbox");
        assert!(outbox.join("envelope.json").exists() && outbox.join("result.bin").exists());
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(outbox.join("result.bin")).unwrap()).unwrap();
        let mut own =
            SmartClusterSupervisor::new(host.ranks[rank as usize].supervision.clone()).unwrap();
        let handle = own
            .find(&rendezvous::dispatch_id(&host.sha, 1, rank).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(result["smartcluster_job"]["job_id"], handle.identity);
        assert_eq!(result["smartcluster_job"]["attempt"], 1);
        assert_eq!(result["smartcluster_job"]["tenant"], tenant());
    }
    // Running the coordinator again on a committed run changes nothing.
    let before = controller::status(&host.run_dir).unwrap().journal_records;
    let again =
        driver::coordinate(&host.args(), host.supervisor(), &AtomicBool::new(false)).unwrap();
    assert_eq!(again.outcome, Some(Outcome::Committed));
    assert_eq!(again.committed_updates_total, 60_005);
    assert_eq!(
        controller::status(&host.run_dir).unwrap().journal_records,
        before
    );
    // A host supervisor acts only on its own plan's jobs: bound to another plan artifact on
    // the same daemon, it neither finds nor probes rank 0's job.
    let dispatch = rendezvous::dispatch_id(&host.sha, 1, 0).unwrap();
    let mut own = SmartClusterSupervisor::new(host.ranks[0].supervision.clone()).unwrap();
    let handle = own.find(&dispatch).unwrap().expect("rank 0's job");
    let other_plan = host.root.join("other-plan.json");
    fs::write(&other_plan, b"{\"another\":\"plan\"}").unwrap();
    let mut foreign = SmartClusterSupervisor::new(HostSupervision {
        plan: other_plan,
        ..host.ranks[0].supervision.clone()
    })
    .unwrap();
    assert_eq!(foreign.find(&dispatch).unwrap(), None);
    for e in [
        foreign.probe(&handle).err(),
        foreign.terminate(&handle).err(),
        foreign.retained_result(&handle).err(),
    ] {
        let e = e.expect("refused");
        assert!(
            e.to_string()
                .contains("is not a job of this supervisor's plan"),
            "{e}"
        );
    }
}

#[test]
fn a_killed_rank_drains_the_whole_group_through_smartcluster() {
    let host = Host::new(400);
    let (_stop, t) = coordinate_in_background(&host);
    host.wait_admitted();
    // SIGKILL rank 1's worker process, which rank 1's SmartCluster daemon started.
    let victim = host.dispatch_dir(1);
    let pid = {
        let mut found = None;
        wait(Duration::from_secs(10), || {
            found = host.rank_processes().into_iter().find(|pid| {
                fs::read(format!("/proc/{pid}/cmdline"))
                    .map(|c| String::from_utf8_lossy(&c).contains(victim.to_str().unwrap()))
                    .unwrap_or(false)
            });
            found.is_some()
        });
        found.unwrap()
    };
    unsafe { libc::kill(pid, libc::SIGKILL) };
    let status = t.join().unwrap().unwrap();
    assert_eq!(
        status.outcome,
        Some(Outcome::Drained),
        "{status:?}\n{}",
        host.diagnose()
    );
    assert_eq!(status.committed_updates_total, 60_000);
    assert_eq!(status.ranks[1].disposition, Some(Disposition::Failed));
    for r in [0, 2] {
        assert_eq!(status.ranks[r].disposition, Some(Disposition::Drained));
    }
    assert!(!host.run_dir.join("results").exists());
    controller::verify(&host.run_dir, Some(&host.sha)).unwrap();
    host.assert_no_orphans();
}

#[test]
fn cancellation_drains_the_whole_group_through_smartcluster() {
    let host = Host::new(400);
    let (_stop, t) = coordinate_in_background(&host);
    host.wait_admitted();
    let report = controller::cancel(&host.run_dir, &host.sha, "operator cancel", now_ms()).unwrap();
    assert!(report.delivered_to_live_coordinator);
    let status = t.join().unwrap().unwrap();
    assert_eq!(
        status.outcome,
        Some(Outcome::Cancelled),
        "{status:?}\n{}",
        host.diagnose()
    );
    assert_eq!(status.committed_updates_total, 60_000);
    assert!(status
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Drained)));
    host.assert_no_orphans();
}

/// Rank 0's participant argv (as its supervisor would submit it), writing to `outbox`, with
/// its frozen launch record at `dir/launch.json`.
fn participant_argv(host: &Host, dir: &Path) -> Vec<String> {
    let s = &host.ranks[0].supervision;
    let record =
        jope_parallel::native_loop::bounded::launch::launch_record(&host.contract, &host.sha, 0)
            .unwrap();
    fs::create_dir_all(dir.join("outbox")).unwrap();
    fs::write(
        dir.join("launch.json"),
        serde_json::to_vec(&record).unwrap(),
    )
    .unwrap();
    let p = |p: &Path| p.display().to_string();
    vec![
        "participant".into(),
        "--plan".into(),
        p(&s.plan),
        "--plan-sha256".into(),
        host.sha.clone(),
        "--launch-record".into(),
        p(&dir.join("launch.json")),
        "--outbox".into(),
        p(&dir.join("outbox")),
        "--coordinator".into(),
        s.coordinator.clone(),
        "--participant-key".into(),
        p(&s.participant_keys[&0]),
        "--installation".into(),
        p(&s.installation),
    ]
}
/// Run the participant directly (not as a SmartCluster job); its exit status and refusal.
/// `control` becomes its descriptor 3 when given.
fn run_outside(
    host: &Host,
    case: &str,
    env: &[(&str, String)],
    control: Option<&std::os::unix::net::UnixStream>,
) -> (bool, String) {
    use std::os::{fd::AsRawFd, unix::process::CommandExt};
    let dir = host.root.join(format!("outside-{case}"));
    let mut command = Command::new(participant_binary());
    command
        .args(participant_argv(host, &dir))
        .env_clear()
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(control) = control {
        let fd = control.as_raw_fd();
        // SAFETY: dup2 is async-signal-safe; it only duplicates an open descriptor.
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(fd, 3) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    let status = command.status().unwrap();
    (
        status.success(),
        fs::read_to_string(dir.join("outbox/refusal.txt")).unwrap_or_default(),
    )
}
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

#[test]
fn a_participant_proves_its_smartcluster_job_before_any_work() {
    use jain_smartcluster::{
        AttemptNumber, FenceToken, FencedAttempt, IdempotencyKey, JobSpec, JobState, TerminalStatus,
    };
    use jope_parallel::native_loop::bounded::{
        controller::RankLaunch,
        job_identity::{verify_job_record, ExpectedJob},
        supervisor::{DISPATCH_LABEL, PLAN_LABEL},
    };
    let host = Host::new(5);
    let rank0 = &host.ranks[0].supervision;
    let dispatch = |rank| rendezvous::dispatch_id(&host.sha, 1, rank).unwrap();

    // Outside SmartCluster: started by hand, with no job environment.
    let (ok, refusal) = run_outside(&host, "no-job", &[], None);
    assert!(!ok);
    assert!(refusal.contains("SCQ_JOB_ID is unset"), "{refusal}");

    // Jobs of rank 0's own daemon, submitted directly with rank 0's frozen launch, whose
    // spec differs from its supervisor's only by `edit`.
    let client = jain_smartcluster::Client::new(rank0.daemon.socket.clone());
    let (mut genuine, mut other_job) = (None, None);
    let cases: [(&str, fn(&mut JobSpec, &str), &str); 3] = [
        // Positive control: it proves its job, then finds no coordinator listening.
        ("genuine", |_, _| {}, "connect 127.0.0.1"),
        (
            "wrong-dispatch",
            |spec, other| {
                spec.labels.insert(DISPATCH_LABEL.into(), other.to_string());
            },
            "was submitted for another dispatch",
        ),
        (
            "other-plan",
            |spec, _| {
                spec.labels.insert(PLAN_LABEL.into(), sha(b"another plan"));
            },
            "was submitted for another plan",
        ),
    ];
    for (case, edit, needle) in cases {
        let supervisor = SmartClusterSupervisor::new(HostSupervision {
            state: host.root.join(format!("jobs-{case}")),
            ..rank0.clone()
        })
        .unwrap();
        let (dir, mut spec) = supervisor
            .launch_spec(&RankLaunch {
                contract: &host.contract,
                plan_sha256: &host.sha,
                global_rank: 0,
                dispatch_id: &dispatch(0),
            })
            .unwrap();
        spec.idempotency_key = IdempotencyKey::new(format!("w7-d1-{case}")).unwrap();
        edit(&mut spec, &dispatch(1));
        let submitted = block_on(client.submit(spec)).unwrap();
        let mut job = None;
        wait(Duration::from_secs(60), || {
            job = block_on(client.get(submitted.id)).unwrap();
            job.as_ref().is_some_and(|j| j.terminal.is_some())
        });
        let job = job.unwrap();
        assert!(
            matches!(job.terminal, Some(TerminalStatus::Failed { .. })),
            "{case}: {:?}\n{}",
            job.terminal,
            host.diagnose()
        );
        let refusal = fs::read_to_string(dir.join("outbox/refusal.txt")).unwrap();
        assert!(refusal.contains(needle), "{case}: {refusal}");
        assert!(!dir.join("outbox/admitted").exists());
        if case == "genuine" {
            genuine = Some(job);
        } else {
            other_job = Some(job.id);
        }
    }

    // Started by hand with a genuine job's environment: no control socket, then a control
    // socket whose peer (this test, its parent) is not the pinned daemon executable.
    let genuine = genuine.unwrap();
    let forged = [
        ("SCQ_JOB_ID", genuine.id.to_string()),
        ("SCQ_ATTEMPT_NUMBER", "1".to_string()),
        ("SCQ_ATTEMPT_FENCE", "1".to_string()),
        ("SCQ_CONTROL_FD", "3".to_string()),
    ];
    let (ok, refusal) = run_outside(&host, "no-control", &forged, None);
    assert!(!ok);
    assert!(refusal.contains("is not an open descriptor"), "{refusal}");
    let (ours, _theirs) = std::os::unix::net::UnixStream::pair().unwrap();
    let (ok, refusal) = run_outside(&host, "impostor-daemon", &forged, Some(&ours));
    assert!(!ok);
    assert!(
        refusal.contains("is not the pinned SmartCluster executable"),
        "{refusal}"
    );

    // The daemon's record checks, on the genuine job's real record made live again.
    let attempt = FencedAttempt::new(
        genuine.id,
        AttemptNumber::new(1).unwrap(),
        FenceToken::new(7).unwrap(),
    );
    let mut live = genuine.clone();
    live.state = JobState::Running;
    live.terminal = None;
    live.active_attempt = Some(attempt);
    let arguments = match &live.spec.workload {
        jain_smartcluster::Workload::TrustedService { arguments, .. } => arguments
            .iter()
            .map(std::ffi::OsString::from)
            .collect::<Vec<_>>(),
        _ => unreachable!(),
    };
    let expected = ExpectedJob {
        plan_sha256: host.sha.clone(),
        run: host.contract.run.clone(),
        global_rank: 0,
        dispatch_id: dispatch(0),
        tenant: tenant(),
        service: SERVICE.into(),
        arguments,
    };
    verify_job_record(&attempt, &live, &expected).unwrap();
    let refused = |claim: &FencedAttempt, job: &jain_smartcluster::JobRecord, e: &ExpectedJob| {
        format!("{:#}", verify_job_record(claim, job, e).unwrap_err())
    };
    // A replayed attempt: the same job under another fence or attempt number.
    for claim in [
        FencedAttempt::new(
            genuine.id,
            AttemptNumber::new(1).unwrap(),
            FenceToken::new(6).unwrap(),
        ),
        FencedAttempt::new(
            genuine.id,
            AttemptNumber::new(2).unwrap(),
            FenceToken::new(7).unwrap(),
        ),
    ] {
        assert!(refused(&claim, &live, &expected).contains("stale or replayed attempt"));
    }
    // A job that is no longer live.
    assert!(refused(&attempt, &genuine, &expected).contains("not live"));
    // Another tenant, another service or other arguments.
    let other_tenant = ExpectedJob {
        tenant: "unix:0".into(),
        ..expected.clone()
    };
    assert!(refused(&attempt, &live, &other_tenant).contains("tenant"));
    let other_service = ExpectedJob {
        service: "another-service".into(),
        ..expected.clone()
    };
    assert!(refused(&attempt, &live, &other_service).contains("runs service"));
    let mut other_arguments = expected.clone();
    other_arguments.arguments.push("--extra".into());
    assert!(refused(&attempt, &live, &other_arguments).contains("arguments"));
    // Another job's answer to this job's claim.
    let mut another = live.clone();
    another.id = other_job.unwrap();
    assert!(refused(&attempt, &another, &expected).contains("answered with job"));

    // A supervision whose daemon is not the one the host installation pins is refused.
    let e = SmartClusterSupervisor::new(HostSupervision {
        daemon: host.ranks[1].supervision.daemon.clone(),
        ..rank0.clone()
    })
    .err()
    .unwrap();
    assert!(
        e.to_string().contains("pins another SmartCluster daemon"),
        "{e}"
    );
}

fn coordinate_cli(host: &Host) -> Child {
    let a = host.args();
    let log = fs::File::create(host.root.join(format!("coordinate-{}.log", now_ms()))).unwrap();
    Command::new(participant_binary())
        .arg("coordinate")
        .args(["--run-dir", a.run_dir.to_str().unwrap()])
        .args(["--plan-sha256", &a.plan_sha256])
        .args(["--census", a.census.to_str().unwrap()])
        .args(["--listen", &a.listen.to_string()])
        .args(["--certificate", a.certificate.to_str().unwrap()])
        .args(["--certificate-key", a.certificate_key.to_str().unwrap()])
        .args(["--coordinator-key", a.coordinator_key.to_str().unwrap()])
        .args(["--heartbeat-ms", "100"])
        .args(["--agents", host.agents_file.to_str().unwrap()])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(log)
        .spawn()
        .unwrap()
}

#[test]
fn a_restarted_coordinator_reconciles_and_drains_with_no_orphan() {
    let host = Host::new(400);
    let mut first = coordinate_cli(&host);
    host.wait_admitted();
    // The coordinator process dies; its ranks lose control and stop themselves.
    unsafe { libc::kill(first.id() as i32, libc::SIGKILL) };
    let _ = first.wait();
    let pending = controller::status(&host.run_dir).unwrap();
    assert_eq!(pending.outcome, None);
    assert!(!pending.coordinator_live);
    host.assert_no_orphans();
    // The restarted coordinator reconciles every owned rank through the same agents.
    let out = coordinate_cli(&host).wait_with_output().unwrap();
    assert!(out.status.success(), "{}", host.diagnose());
    let status: controller::StatusReport = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(status.outcome, Some(Outcome::Drained));
    assert_eq!(status.committed_updates_total, 60_000);
    assert_eq!(
        status.original_deadline_unix_ms,
        host.contract.original_deadline_unix_ms
    );
    assert!(host.journal_contains("coordinator_restarted"));
    assert!(status.ranks.iter().all(|r| r.disposition.is_some()));
    controller::verify(&host.run_dir, Some(&host.sha)).unwrap();
    host.assert_no_orphans();
}
